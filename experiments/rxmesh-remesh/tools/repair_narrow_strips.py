"""Repair narrow CAD strips and synchronize shared boundaries before GPU remeshing.

The input CADPART1 snapshot is immutable. Unsupported patches are retained and
reported. Output is committed only after topology, anchors and seams pass.
"""
import argparse
import json
import math
import shutil
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from narrow_strip_geometry import require, remesh, ring_of, quality, sample_distances


class NeighborRepairRequired(ValueError):
    def __init__(self, patch, detail):
        super().__init__(detail)
        self.patch=int(patch)


def read_snapshot(path):
    with path.open('rb') as stream:
        header=stream.read(72)
        require(len(header)==72 and header[:8]==b'CADPART1', 'expected a CADPART1 snapshot')
        nv,nf,nr,ne=struct.unpack_from('<4I',header,8)
        require(nv>0 and nf>0 and 0<nr<=nf, 'invalid snapshot counts')
        require(path.stat().st_size>=72+nv*24+nf*16+nr*92+ne*16, 'truncated snapshot')
        points=np.frombuffer(stream.read(nv*24),'<f8').reshape(nv,3).copy()
        records=np.frombuffer(stream.read(nf*16),'<u4').reshape(nf,4).copy()
        patches=[]
        for _ in range(nr):
            ph=list(struct.unpack('<5I',stream.read(20)))
            tail=stream.read(ph[4]*4+72)
            patches.append((ph,tail))
        features=np.frombuffer(stream.read(ne*16),'<u4').reshape(ne,4).copy()
        require(not stream.read(1), 'geometry or topology invariant failed: not stream.read(1)')
    require(np.isfinite(points).all(), 'non-finite input coordinates')
    require((records[:,:3]<nv).all() and (records[:,3]<nr).all(), 'invalid face or patch index')
    require((features[:,1:3]<nv).all(), 'invalid feature vertex')
    counts=np.bincount(records[:,3],minlength=nr)
    require(all(ph[3]==counts[i] for i,(ph,_) in enumerate(patches)), 'patch face counts do not match')
    return header,points,records,patches,features


def write_snapshot(path,header,p,records,patches,features):
    output_header=bytearray(header)
    struct.pack_into('<4I',output_header,8,len(p),len(records),len(patches),len(features))
    counts=np.bincount(records[:,3],minlength=len(patches))
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(output_header)
        stream.write(np.asarray(p,dtype='<f8').tobytes())
        stream.write(np.asarray(records,dtype='<u4').tobytes())
        for i,(ph,tail) in enumerate(patches):
            updated=ph.copy(); updated[3]=int(counts[i])
            stream.write(struct.pack('<5I',*updated));stream.write(tail)
        stream.write(np.asarray(features,dtype='<u4').tobytes())
    temporary.replace(path)


def effective_target(points, requested):
    if requested>0:
        return float(np.float32(requested))
    p=points.astype(np.float32)
    span=p.max(0)-p.min(0)
    diagonal=np.sqrt(np.float32(span[0]*span[0]+span[1]*span[1]+span[2]*span[2]))
    return float(np.float32(max(diagonal,np.float32(1e-6)))*np.float32(.01))


def residual_patches(path, target):
    if not path:
        return set()
    with path.open(encoding='utf-8') as stream:
        require(stream.readline().strip()=='ply', 'residual input must be ASCII PLY')
        require(stream.readline().strip()=='format ascii 1.0', 'residual PLY must be ASCII')
        nv=None
        while True:
            line=stream.readline()
            require(bool(line), 'truncated residual PLY header')
            if line.startswith('element vertex '): nv=int(line.split()[-1])
            if line.strip()=='end_header': break
        require(nv is not None and nv>0, 'residual PLY has no vertices')
        p=np.loadtxt(stream,max_rows=nv,ndmin=2)[:,:3]
        r=np.loadtxt(stream,dtype=np.int64,ndmin=2)
    require(r.shape[1]==5 and (r[:,0]==3).all(), 'residual PLY requires triangles with patch IDs')
    tri=p[r[:,1:4]]
    longest=np.linalg.norm(tri-np.roll(tri,-1,axis=1),axis=2).max(1)
    return set(r[longest>target*(4/3)*(1+1e-4),4].tolist())


def is_strip_candidate(points, records, patches, rid, target):
    header=patches[rid][0]
    if header[0] not in (1,6) or not 0<header[3]<=64:
        return False
    faces=records[records[:,3]==rid,:3]
    extent=np.sort(np.ptp(points[np.unique(faces)],axis=0))[::-1]
    return extent[0]>target*4/3 and extent[0]/max(extent[1],1e-9)>100


def select_candidates(points, records, patches, target, residual):
    remaining=residual_patches(residual,target)
    selected=[]
    for rid,(header,_) in enumerate(patches):
        # Freeform strips first; admit planar strips when the full GPU result
        # demonstrates a size failure there. Never special-case model IDs.
        if (header[0]!=6 and not (header[0]==1 and rid in remaining)) or not 0<header[3]<=64:
            continue
        if is_strip_candidate(points,records,patches,rid,target):
            selected.append(rid)
    return selected


def edge_data(faces):
    oriented=faces[:,[0,1,1,2,2,0]].reshape(-1,2).astype(np.uint64)
    lo=oriented.min(1); hi=oriented.max(1)
    keys=(lo<<32)|hi
    unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    winding=np.bincount(inverse,weights=np.where(oriented[:,0]<oriented[:,1],1,-1))
    return unique,counts,winding


def audit(p,records,features):
    faces=records[:,:3]
    keys,counts,winding=edge_data(faces)
    pp=p.astype(np.float32)
    tri=pp[faces]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    zero=np.einsum('ij,ij->i',normal,normal)<=0
    canonical=np.sort(faces,axis=1)
    duplicate=len(faces)-len(np.unique(canonical,axis=0))
    feature_keys=(np.minimum(features[:,1],features[:,2]).astype(np.uint64)<<32)|np.maximum(features[:,1],features[:,2])
    lo=(keys>>32).astype(np.int64);hi=(keys&0xffffffff).astype(np.int64)
    graph=coo_matrix((np.ones(len(keys),dtype=np.uint8),(lo,hi)),shape=(len(p),len(p))).tocsr()
    used=np.unique(faces)
    components=connected_components(graph[used][:,used],directed=False,return_labels=False)
    return dict(vertices=len(p),used_vertices=len(used),faces=len(faces),edges=len(keys),open_edges=int((counts==1).sum()),nonmanifold_edges=int((counts>2).sum()),inconsistent_interior_edges=int(((counts==2)&(winding!=0)).sum()),float32_zero_faces=int(zero.sum()),duplicate_faces=int(duplicate),connected_components=int(components),euler_characteristic=int(len(used)-len(keys)+len(faces)),missing_feature_edges=int((~np.isin(feature_keys,keys)).sum())),keys,counts


def triangulate_neighbor(poly,vertices,reference_normal,source_triangle):
    # Usually only one source edge receives samples. A fan from the opposite
    # vertex preserves the original triangle exactly and is linear in size.
    for anchor in range(len(poly)):
        ring=poly[anchor:]+poly[:anchor]
        faces=np.array([(ring[0],ring[i],ring[i+1]) for i in range(1,len(ring)-1)])
        exact=np.array([[vertices[int(v)] for v in face] for face in faces])
        exact_sign=np.cross(exact[:,1]-exact[:,0],exact[:,2]-exact[:,0])@reference_normal
        tri=exact.astype(np.float32)
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        if (exact_sign>np.dot(reference_normal,reference_normal)*1e-12).all() and (normals@reference_normal>0).all(): return faces.tolist()
    # A very thin original face may not admit any safe fan from its corners
    # after float32 rounding. An interior point avoids collinear boundary ears
    # without changing any shared boundary sample or original anchor.
    for weights in ((1/3,1/3,1/3),(.5,.25,.25),(.25,.5,.25),(.25,.25,.5)):
        center=np.asarray(weights)@source_triangle
        ring_points=np.array([vertices[v] for v in poly])
        exact=np.stack([np.repeat(center[None,:],len(poly),axis=0),ring_points,np.roll(ring_points,-1,axis=0)],axis=1)
        exact_sign=np.cross(exact[:,1]-exact[:,0],exact[:,2]-exact[:,0])@reference_normal
        tri=exact.astype(np.float32)
        signs=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])@reference_normal
        if (exact_sign>np.dot(reference_normal,reference_normal)*1e-12).all() and (signs>0).all():
            index=len(vertices);vertices.append(center)
            return [[index,a,b] for a,b in zip(poly,poly[1:]+poly[:1])]
    # Convex source triangle with several subdivided sides: quality-prioritized
    # ear removal, retaining every boundary sample and forbidding inversion.
    ring=poly.copy(); faces=[]
    while len(ring)>3:
        candidates=[]
        for i in range(len(ring)):
            ids=[ring[i-1],ring[i],ring[(i+1)%len(ring)]]
            exact=np.array([vertices[j] for j in ids])
            if np.dot(np.cross(exact[1]-exact[0],exact[2]-exact[0]),reference_normal)<=np.dot(reference_normal,reference_normal)*1e-12: continue
            tri=exact.astype(np.float32)
            if np.dot(np.cross(tri[1]-tri[0],tri[2]-tri[0]),reference_normal)<=0: continue
            rest=ring[:i]+ring[i+1:]
            rp=np.array([vertices[j] for j in rest]); base=rp[0]
            area=sum(np.dot(np.cross(rp[k]-base,rp[k+1]-base),reference_normal) for k in range(1,len(rp)-1))
            if area<=1e-14: continue
            candidates.append((quality(tri),i,ids))
        require(candidates, 'neighbor subdivision has no safe ear')
        _,i,ids=max(candidates);faces.append(ids);ring.pop(i)
    tri=np.array([vertices[j] for j in ring],dtype=np.float32)
    require(np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), reference_normal) > 0, 'geometry or topology invariant failed: np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), reference_normal) > 0')
    faces.append(ring)
    return faces



def repair(input_path, output_path, target_length=0., residual=None, previous_report=None, max_error=0., extra_candidates=()):
    started=time.perf_counter()
    require(input_path.resolve()!=output_path.resolve(), 'input snapshot must remain immutable')
    header,p,records,patches,features=read_snapshot(input_path)
    before,source_keys,source_counts=audit(p,records,features)
    vertices=list(p.copy()); chains=defaultdict(list); replacements={}; local_results={}
    target_length = effective_target(p, target_length)
    tolerance = min(target_length * 1e-6, max_error) if max_error else target_length * 1e-6
    requested = sorted(set(select_candidates(p, records, patches, target_length, residual)) | set(extra_candidates))
    if previous_report:
        previous = json.loads(previous_report.read_text(encoding="utf-8"))
        require(Path(previous['source_snapshot']).resolve()==input_path.resolve(), 'previous repair uses a different source')
        require(previous['target_length']==target_length, 'previous repair uses a different target')
        requested = sorted(set(requested) | set(previous["candidate_patch_ids"]))
        if requested == previous["candidate_patch_ids"]:
            shutil.copyfile(previous["output_snapshot"], output_path)
            previous.update(output_snapshot=str(output_path.resolve()), candidate_set_changed=False,
                            seconds=time.perf_counter()-started)
            return previous
    if not requested:
        shutil.copyfile(input_path, output_path)
        return dict(source_snapshot=str(input_path.resolve()), output_snapshot=str(output_path.resolve()),
                    target_length=target_length, candidate_patch_ids=[], candidate_set_changed=False,
                    rebuilt_patches={}, rejected_patches={}, seconds=time.perf_counter()-started)

    for field in ('nonmanifold_edges','inconsistent_interior_edges','float32_zero_faces','duplicate_faces','missing_feature_edges'):
        require(before[field]==0, f'input mesh failed {field}: {before[field]}')

    shared_grid=None
    if requested:
        shared_grid={}
        selected_vertices=np.unique(records[np.isin(records[:,3],requested),:3])
        step=.95*target_length
        for axis in range(3):
            # Shift the common grid away from source anchors to avoid creating
            # tiny end slivers solely because an anchor almost hits a station.
            residues=np.unique(np.remainder(p[selected_vertices,axis],step))
            ends=np.r_[residues[1:],residues[0]+step]
            best=int(np.argmax(ends-residues))
            phase=((residues[best]+ends[best])/2)%step
            low=p[:,axis].min();high=p[:,axis].max()
            require((high-low)/step < 100000, "shared-grid station budget exceeded")
            shared_grid[axis]=phase+step*np.arange(np.floor((low-phase)/step),np.ceil((high-phase)/step)+1)
    prepared={};rejected={}
    feature_keys={(int(min(a,b)),int(max(a,b))) for _,a,b,_ in features}
    for rid in requested:
        source_faces=records[records[:,3]==rid,:3]; ids=np.unique(source_faces)
        pp=p[ids]; ff=np.searchsorted(ids,source_faces)
        try:
            require(len(np.unique(pp,axis=0))==len(pp), 'duplicate source positions require topology-aware repair')
            ring=ring_of(ff)
            boundary_keys={tuple(sorted((int(ids[a]),int(ids[b])))) for a,b in zip(ring,ring[1:]+ring[:1])}
            internal_keys={tuple(sorted((int(a),int(b)))) for face in source_faces for a,b in zip(face,np.roll(face,-1))}-boundary_keys
            require(not internal_keys & feature_keys, 'internal protected edge requires explicit preservation')
            rp,rf,groups=remesh(pp,ff,target_length,shared_grid,tolerance)
            prepared[rid]=(ids,pp,ff,rp,rf,groups)
        except (KeyError,ValueError) as exc:
            rejected[rid]=str(exc)
    print(f'prepared={len(prepared)} rejected={len(rejected)}',flush=True)
    targets=tuple(prepared)
    if not targets:
        shutil.copyfile(input_path, output_path)
        return dict(source_snapshot=str(input_path.resolve()), output_snapshot=str(output_path.resolve()),
                    target_length=target_length, candidate_patch_ids=requested, candidate_set_changed=True,
                    rebuilt_patches={}, rejected_patches=rejected, seconds=time.perf_counter()-started)
    for rid,(ids,pp,ff,rp,rf,groups) in prepared.items():
        original_by_position={tuple(q):int(i) for q,i in zip(pp,ids)}
        source_ring=ring_of(ff); output_ring=set(ring_of(rf)); boundary=[]
        for a,b in zip(source_ring,source_ring[1:]+source_ring[:1]):
            ga,gb=int(ids[a]),int(ids[b]); boundary.append((ga,gb,p[ga],p[gb]))
        mapping=[]
        for i,q in enumerate(rp):
            if tuple(q) in original_by_position:
                mapping.append(original_by_position[tuple(q)]);continue
            if i in output_ring:
                found=[]
                for ga,gb,a,b in boundary:
                    d=b-a;t=float(np.dot(q-a,d)/np.dot(d,d))
                    if 0<t<1 and np.linalg.norm(q-(a+t*d))<1e-8:
                        key=tuple(sorted((ga,gb)));t=t if ga<gb else 1-t
                        found.append((key,t))
                require(len(found) == 1, 'boundary sample has ambiguous source edge')
                key,t=found[0]
                same=[idx for old_t,idx in chains[key] if abs(old_t-t)<1e-12]
                if same: g=same[0]
                else:
                    g=len(vertices);vertices.append(q);chains[key].append((t,g))
            else:
                g=len(vertices);vertices.append(q)
            mapping.append(g)
        replacements[rid]=np.asarray(mapping,dtype=np.uint32)[rf]
        local_results[rid]=dict(faces=len(rf),planar_components=groups)
    chains={key:[a]+[i for t,i in sorted(samples)]+[b] for (a,b),samples in chains.items() for key in [(a,b)]}
    keys_to_split=np.array([(int(a)<<32)|int(b) for a,b in chains],dtype=np.uint64)
    face_edges=records[:,[0,1,1,2,2,0]].reshape(-1,3,2).astype(np.uint64)
    face_keys=(face_edges.min(2)<<32)|face_edges.max(2)
    affected=np.isin(face_keys,keys_to_split).any(1)&~np.isin(records[:,3],targets)
    neighbor_records=[]; neighbor_checks=[]
    for fi in np.flatnonzero(affected):
        face=records[fi,:3]; ring=[]
        for a,b in zip(face,np.roll(face,-1)):
            key=tuple(sorted((int(a),int(b))))
            chain=chains.get(key,[key[0],key[1]])
            if a!=key[0]: chain=chain[::-1]
            ring.extend(chain[:-1])
        normal=np.cross(p[face[1]]-p[face[0]],p[face[2]]-p[face[0]])
        try:
            children=triangulate_neighbor(ring,vertices,normal,p[face])
        except ValueError as exc:
            rid=int(records[fi,3])
            if is_strip_candidate(p,records,patches,rid,target_length):
                raise NeighborRepairRequired(rid,f'neighbor face {fi}, patch {rid}: {exc}') from exc
            raise ValueError(f'neighbor face {fi}, patch {records[fi,3]}: {exc}') from exc
        for tri in children: neighbor_records.append([*tri,int(records[fi,3])])
        cp=np.array([[vertices[int(v)] for v in face] for face in children])
        area=np.linalg.norm(np.cross(cp[:,1]-cp[:,0],cp[:,2]-cp[:,0]),axis=1).sum()
        neighbor_checks.append(dict(source_face=int(fi),patch=int(records[fi,3]),children=len(children),relative_area_change=float(abs(area/np.linalg.norm(normal)-1))))
    keep=~np.isin(records[:,3],targets)&~affected
    blocks=[records[keep],np.array(neighbor_records,dtype=np.uint32).reshape(-1,4)]
    for rid,ff in replacements.items(): blocks.append(np.column_stack([ff,np.full(len(ff),rid,dtype=np.uint32)]))
    output_records=np.concatenate(blocks)
    output_p=np.array(vertices)
    output_features=[]
    for curve,a,b,kind in features:
        key=tuple(sorted((int(a),int(b))));chain=chains.get(key,[int(a),int(b)])
        for u,v in zip(chain[:-1],chain[1:]): output_features.append([curve,min(u,v),max(u,v),kind])
    output_features=np.array(output_features,dtype=np.uint32).reshape(-1,4)
    after,output_keys,output_counts=audit(output_p,output_records,output_features)
    for field in ('nonmanifold_edges','inconsistent_interior_edges','float32_zero_faces','duplicate_faces','missing_feature_edges'):
        require(after[field] == 0, (field, after[field]))
    for field in ('connected_components','euler_characteristic'):
        require(before[field] == after[field], (field, before[field], after[field]))
    # Exact edge-identity audit: each old shared edge is replaced by one common
    # sequence, every segment has the original number of incident faces.
    lookup=dict(zip(output_keys.tolist(),output_counts.tolist()))
    source_lookup=dict(zip(source_keys.tolist(),source_counts.tolist()))
    for (a,b),chain in chains.items():
        old=(int(a)<<32)|int(b)
        require(old not in lookup, ('old shared edge retained', a, b, lookup.get(old)))
        for u,v in zip(chain[:-1],chain[1:]):
            key=(int(min(u,v))<<32)|int(max(u,v))
            require(lookup[key] == source_lookup[old], 'geometry or topology invariant failed: lookup[key] == source_lookup[old]')
    require(np.array_equal(output_p[:len(p)], p), 'geometry or topology invariant failed: np.array_equal(output_p[:len(p)], p)')
    original_open=set(source_keys[source_counts==1].tolist())
    for (a,b),chain in chains.items():
        key=(int(a)<<32)|int(b)
        if key in original_open:
            original_open.remove(key)
            original_open.update((int(min(u,v))<<32)|int(max(u,v)) for u,v in zip(chain[:-1],chain[1:]))
    require(original_open == set(output_keys[output_counts == 1].tolist()), 'new hole or seam')
    forward=[]; forward32=[]
    quantized=output_p.astype(np.float32).astype(float)
    for rid, faces in replacements.items():
        source_faces=records[records[:,3]==rid,:3]
        forward.extend(sample_distances(output_p[faces].mean(1),p,source_faces))
        forward32.extend(sample_distances(quantized[faces].mean(1),p,source_faces))
    forward=np.asarray(forward); forward32=np.asarray(forward32)
    allowed=max_error if max_error else .2*target_length
    require(float(forward32.max()) <= allowed, "strip reconstruction exceeds geometry error budget")
    report=dict(before=before,after=after,rebuilt_patches=local_results,rejected_patches=rejected,shared_edges_resampled=len(chains),new_shared_boundary_vertices=sum(len(c)-2 for c in chains.values()),neighbor_source_faces=len(neighbor_checks),neighbor_output_faces=len(neighbor_records),neighbor_checks=neighbor_checks,all_original_vertices_unchanged=True,shared_edge_incidence_matches=True,open_boundary_identity_matches=True,changed_face_centroid_sample_error=float(forward.max()),float32_changed_face_centroid_sample_error=float(forward32.max()),source_snapshot=str(input_path.resolve()),output_snapshot=str(output_path.resolve()),target_length=target_length,candidate_patch_ids=requested,candidate_set_changed=True,seconds=time.perf_counter()-started)
    write_snapshot(output_path,header,output_p,output_records,patches,output_features)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--target',type=float,default=0.)
    parser.add_argument('--max-error',type=float,default=0.)
    parser.add_argument('--residual',type=Path)
    parser.add_argument('--previous-report',type=Path)
    args=parser.parse_args()
    try:
        require(args.input.resolve()!=args.output.resolve(), 'input snapshot must remain immutable')
        require(args.report.resolve() not in (args.input.resolve(),args.output.resolve()), 'report path collides with snapshot')
        require(math.isfinite(args.target) and args.target>=0, 'invalid target length')
        require(math.isfinite(args.max_error) and args.max_error>=0, 'invalid geometry error')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        expanded=set(); expansion_reasons=[];started=time.perf_counter()
        for attempt in range(9):
            try:
                report=repair(args.input,args.output,args.target,args.residual,args.previous_report,args.max_error,expanded)
                break
            except NeighborRepairRequired as exc:
                require(exc.patch not in expanded and attempt<8, f'neighbor expansion stopped: {exc}')
                expanded.add(exc.patch); expansion_reasons.append(str(exc))
                print(f'strip_repair expanding_neighbor={exc.patch}',flush=True)
        report['expanded_neighbors']=sorted(expanded)
        report['expansion_reasons']=expansion_reasons
        report['seconds']=time.perf_counter()-started
        report['status']='ok'
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f"strip_repair candidates={len(report['candidate_patch_ids'])} rebuilt={len(report['rebuilt_patches'])} rejected={len(report['rejected_patches'])} seconds={report['seconds']:.3f}")
        return 0
    except (OSError,ValueError,KeyError,struct.error) as exc:
        # Do not replace a source/output collision with a JSON error report.
        if args.report.resolve() not in (args.input.resolve(),args.output.resolve()):
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps({'status':'failed','error':str(exc)},indent=2),encoding='utf-8')
        print(f'strip repair failed: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
