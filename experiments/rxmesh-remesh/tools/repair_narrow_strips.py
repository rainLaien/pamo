"""Repair narrow CAD strips and synchronize shared boundaries before GPU remeshing.

The input CADPART1 snapshot is immutable. Unsupported patches are retained and
reported. Output is committed only after topology, anchors and seams pass.
"""
import argparse
import json
import hashlib
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


def read_patch_targets(path, source):
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    require(isinstance(data,dict), 'regional targets require a JSON object')
    if 'targets' in data:
        require(data.get('snapshot_sha256')==hashlib.sha256(source.read_bytes()).hexdigest(), 'regional selection snapshot hash mismatch')
        data=data['targets']
    require(isinstance(data,dict), 'regional targets require a JSON object')
    return data


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


def is_strip_candidate(points, records, patches, rid, target, face_index=None):
    header=patches[rid][0]
    if header[0] not in (1,6) or not 0<header[3]<=64:
        return False
    faces=records[face_index[rid],:3] if face_index is not None else records[records[:,3]==rid,:3]
    extent=np.sort(np.ptp(points[np.unique(faces)],axis=0))[::-1]
    return extent[0]>target*4/3 and extent[0]/max(extent[1],1e-9)>100


def select_candidates(points, records, patches, target, residual, face_index=None):
    remaining=residual_patches(residual,target)
    selected=[]
    for rid,(header,_) in enumerate(patches):
        # Freeform strips first; admit planar strips when the full GPU result
        # demonstrates a size failure there. Never special-case model IDs.
        if (header[0]!=6 and not (header[0]==1 and rid in remaining)) or not 0<header[3]<=64:
            continue
        if is_strip_candidate(points,records,patches,rid,target,face_index):
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



def regional_targets(records, requested, overrides, default_target):
    """Keep adjacent requested strips on one grid; bound the experimental halo."""
    if not overrides:
        return {rid:default_target for rid in requested}
    rows=records[np.isin(records[:,3],requested)]
    ee=rows[:,[0,1,1,2,2,0]].reshape(-1,2).astype(np.uint64)
    keys=(ee.min(1)<<32)|ee.max(1);order=np.argsort(keys)
    pairs=np.flatnonzero(keys[order[:-1]]==keys[order[1:]])
    owners=np.repeat(rows[:,3],3)
    graph={rid:set() for rid in requested}
    for a,b in zip(owners[order[pairs]],owners[order[pairs+1]]):
        if a!=b:graph[int(a)].add(int(b));graph[int(b)].add(int(a))
    sizes={rid:default_target for rid in requested};seen=set()
    for rid in requested:
        if rid in seen:continue
        component={rid};stack=[rid];seen.add(rid)
        while stack:
            for neighbor in graph[stack.pop()]-seen:
                seen.add(neighbor);component.add(neighbor);stack.append(neighbor)
        size=min([default_target]+[overrides[x] for x in component if x in overrides])
        for x in component:sizes[x]=size
    require(sum(v<default_target for v in sizes.values())<=32, 'regional connected halo exceeds experimental 32-patch budget')
    return sizes


def repair(input_path, output_path, target_length=0., residual=None, previous_report=None, max_error=0., extra_candidates=(), _source_cache=None, patch_targets=None, graded_neighbors=False, only_requested_regions=False, constrained_neighbors=False):
    started=time.perf_counter()
    require(input_path.resolve()!=output_path.resolve(), 'input snapshot must remain immutable')
    # This cache lives only for the bounded neighbor-expansion loop. The source
    # stays immutable; output audits are still performed for every result.
    cache={} if _source_cache is None else _source_cache
    if 'source' not in cache:
        cache['source']=read_snapshot(input_path)
        _,points,rows,regions,edges=cache['source']
        cache['audit']=audit(points,rows,edges)
        order=np.argsort(rows[:,3],kind='stable')
        offsets=np.r_[0,np.cumsum(np.bincount(rows[:,3],minlength=len(regions)))]
        cache['face_index']=[order[offsets[i]:offsets[i+1]] for i in range(len(regions))]
        cache['prepared']={}
    header,p,records,patches,features=cache['source']
    before,source_keys,source_counts=cache['audit']
    face_index=cache['face_index']
    vertices=list(p.copy()); chains=defaultdict(list); replacements={}; local_results={};regional_chain_targets={}
    target_length = effective_target(p, target_length)
    overrides={int(k):float(v) for k,v in (patch_targets or {}).items()}
    require(not graded_neighbors or overrides, 'graded neighbors require regional targets')
    require(not constrained_neighbors or graded_neighbors, 'constrained neighbors require graded neighbors')
    require(all(0<=k<len(patches) and math.isfinite(v) and 0<v<target_length for k,v in overrides.items()), 'invalid regional patch target')
    require(len(overrides)<=32, 'regional selection exceeds experimental 32-patch budget')
    tolerance = min(target_length * 1e-6, max_error) if max_error else target_length * 1e-6
    require(not only_requested_regions or overrides, "bounded regional rebuild requires explicit targets")
    default_requested=set(extra_candidates)
    if not only_requested_regions:
        default_requested |= set(select_candidates(p, records, patches, target_length, residual, face_index))
    requested = sorted(default_requested | set(overrides))
    if previous_report:
        previous = json.loads(previous_report.read_text(encoding="utf-8"))
        require(Path(previous['source_snapshot']).resolve()==input_path.resolve(), 'previous repair uses a different source')
        require(previous['target_length']==target_length, 'previous repair uses a different target')
        require({int(k):float(v) for k,v in previous.get('regional_patch_targets',{}).items()}==overrides, 'previous repair uses different regional targets')
        require(previous.get('graded_neighbors',False)==graded_neighbors, 'previous repair uses different neighbor candidate')
        require(previous.get('only_requested_regions',False)==only_requested_regions, 'previous repair uses different regional scope')
        require(previous.get('constrained_neighbors',False)==constrained_neighbors, 'previous repair uses different neighbor triangulation')
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
                    rebuilt_patches={}, rejected_patches={}, regional_patch_targets=overrides, graded_neighbors=graded_neighbors, only_requested_regions=only_requested_regions, constrained_neighbors=constrained_neighbors, seconds=time.perf_counter()-started)

    for field in ('nonmanifold_edges','inconsistent_interior_edges','float32_zero_faces','duplicate_faces','missing_feature_edges'):
        require(before[field]==0, f'input mesh failed {field}: {before[field]}')

    sizes=regional_targets(records,requested,overrides,target_length)
    grids={}
    for size in sorted(set(sizes.values())):
        shared_grid={}
        # Preserve the unchanged default grid when a regional patch is added.
        grid_patches=default_requested if size==target_length else [rid for rid in requested if sizes[rid]==size]
        selected_vertices=np.unique(records[np.isin(records[:,3],list(grid_patches)),:3])
        if not len(selected_vertices):continue
        step=.95*size
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
        grids[size]=shared_grid
    prepared={};rejected={}
    feature_keys={(int(min(a,b)),int(max(a,b))) for _,a,b,_ in features}
    for rid in requested:
        local_target=sizes[rid];shared_grid=grids[local_target]
        grid_key=tuple(shared_grid[axis].tobytes() for axis in range(3))
        source_faces=records[face_index[rid],:3]; ids=np.unique(source_faces)
        pp=p[ids]; ff=np.searchsorted(ids,source_faces)
        try:
            cache_key=(rid,local_target,tolerance,grid_key)
            if cache_key in cache['prepared']:
                prepared[rid]=cache['prepared'][cache_key]
                continue
            require(len(np.unique(pp,axis=0))==len(pp), 'duplicate source positions require topology-aware repair')
            ring=ring_of(ff)
            boundary_keys={tuple(sorted((int(ids[a]),int(ids[b])))) for a,b in zip(ring,ring[1:]+ring[:1])}
            internal_keys={tuple(sorted((int(a),int(b)))) for face in source_faces for a,b in zip(face,np.roll(face,-1))}-boundary_keys
            require(not internal_keys & feature_keys, 'internal protected edge requires explicit preservation')
            rp,rf,groups=remesh(pp,ff,local_target,shared_grid,tolerance)
            prepared[rid]=(ids,pp,ff,rp,rf,groups)
            cache['prepared'][cache_key]=prepared[rid]
        except (KeyError,ValueError) as exc:
            rejected[rid]=str(exc)
    print(f'prepared={len(prepared)} rejected={len(rejected)}',flush=True)
    targets=tuple(prepared)
    if not targets:
        shutil.copyfile(input_path, output_path)
        return dict(source_snapshot=str(input_path.resolve()), output_snapshot=str(output_path.resolve()),
                    target_length=target_length, candidate_patch_ids=requested, candidate_set_changed=True,
                    rebuilt_patches={}, rejected_patches=rejected, regional_patch_targets=overrides, graded_neighbors=graded_neighbors, only_requested_regions=only_requested_regions, constrained_neighbors=constrained_neighbors, seconds=time.perf_counter()-started)
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
                if sizes[rid]<target_length:regional_chain_targets[key]=sizes[rid]
                same=[idx for old_t,idx in chains[key] if abs(old_t-t)<1e-12]
                if same: g=same[0]
                else:
                    g=len(vertices);vertices.append(q);chains[key].append((t,g))
            else:
                g=len(vertices);vertices.append(q)
            mapping.append(g)
        replacements[rid]=np.asarray(mapping,dtype=np.uint32)[rf]
        local_results[rid]=dict(faces=len(rf),planar_components=groups,target_length=sizes[rid])
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
        graded_info=None
        try:
            fine_edges=[i for i,(a,b) in enumerate(zip(face,np.roll(face,-1))) if tuple(sorted((int(a),int(b)))) in regional_chain_targets]
            if graded_neighbors and fine_edges:
                from graded_neighbor_geometry import triangulate_graded
                local_h=min(regional_chain_targets[tuple(sorted((int(face[i]),int(face[(i+1)%3]))))] for i in fine_edges)
                children,graded_info=triangulate_graded(ring,vertices,normal,p[face],fine_edges,local_h,target_length,constrained_neighbors)
            else:children=triangulate_neighbor(ring,vertices,normal,p[face])
        except ValueError as exc:
            rid=int(records[fi,3])
            if is_strip_candidate(p,records,patches,rid,target_length,face_index):
                raise NeighborRepairRequired(rid,f'neighbor face {fi}, patch {rid}: {exc}') from exc
            raise ValueError(f'neighbor face {fi}, patch {records[fi,3]}: {exc}') from exc
        for tri in children: neighbor_records.append([*tri,int(records[fi,3])])
        cp=np.array([[vertices[int(v)] for v in face] for face in children])
        if graded_info:
            quantized_children=cp.astype(np.float32).astype(float)
            child_points=quantized_children.reshape(-1,3)
            child_faces=np.arange(len(child_points)).reshape(-1,3)
            reference=p[face]
            forward_neighbor=sample_distances(np.vstack([child_points,quantized_children.mean(1)]),reference,np.array([[0,1,2]]))
            reverse_neighbor=sample_distances(np.vstack([reference,(reference+np.roll(reference,-1,axis=0))*.5,reference.mean(0)]),child_points,child_faces)
            graded_info['float32_forward_sample_error']=float(forward_neighbor.max(initial=0))
            graded_info['float32_reverse_sample_error']=float(reverse_neighbor.max(initial=0))
            require(max(graded_info['float32_forward_sample_error'],graded_info['float32_reverse_sample_error'])<=(max_error if max_error else .2*target_length), 'graded neighbor exceeds unchanged geometry budget')
        area=np.linalg.norm(np.cross(cp[:,1]-cp[:,0],cp[:,2]-cp[:,0]),axis=1).sum()
        neighbor_checks.append(dict(source_face=int(fi),patch=int(records[fi,3]),children=len(children),relative_area_change=float(abs(area/np.linalg.norm(normal)-1))))
        if graded_info:neighbor_checks[-1]['graded_candidate']=graded_info
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
        source_faces=records[face_index[rid],:3]
        forward.extend(sample_distances(output_p[faces].mean(1),p,source_faces))
        forward32.extend(sample_distances(quantized[faces].mean(1),p,source_faces))
    forward=np.asarray(forward); forward32=np.asarray(forward32)
    allowed=max_error if max_error else .2*target_length
    require(float(forward32.max()) <= allowed, "strip reconstruction exceeds geometry error budget")
    report=dict(before=before,after=after,rebuilt_patches=local_results,rejected_patches=rejected,shared_edges_resampled=len(chains),new_shared_boundary_vertices=sum(len(c)-2 for c in chains.values()),neighbor_source_faces=len(neighbor_checks),neighbor_output_faces=len(neighbor_records),neighbor_checks=neighbor_checks,all_original_vertices_unchanged=True,shared_edge_incidence_matches=True,open_boundary_identity_matches=True,changed_face_centroid_sample_error=float(forward.max()),float32_changed_face_centroid_sample_error=float(forward32.max()),source_snapshot=str(input_path.resolve()),output_snapshot=str(output_path.resolve()),target_length=target_length,candidate_patch_ids=requested,candidate_set_changed=True,seconds=time.perf_counter()-started)
    report['regional_patch_targets']=overrides
    report['graded_neighbors']=graded_neighbors
    report['only_requested_regions']=only_requested_regions
    report['constrained_neighbors']=constrained_neighbors
    report['regional_effective_patch_targets']={rid:size for rid,size in sizes.items() if size<target_length}
    regional_ids=set(report['regional_effective_patch_targets'])
    if regional_ids:
        ee=records[:,[0,1,1,2,2,0]].reshape(-1,2).astype(np.uint64)
        regional_edges=records[np.isin(records[:,3],list(regional_ids))][:,[0,1,1,2,2,0]].reshape(-1,2).astype(np.uint64)
        regional_keys=(regional_edges.min(1)<<32)|regional_edges.max(1)
        all_keys=((ee.min(1)<<32)|ee.max(1)).reshape(-1,3)
        report['regional_affected_patch_ids']=np.unique(records[np.isin(all_keys,regional_keys).any(1),3]).tolist()
        require(sum(local_results[rid]['faces'] for rid in regional_ids if rid in local_results)<=50000, 'regional faces exceed experimental 50000-face budget')
    else:report['regional_affected_patch_ids']=[]
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
    parser.add_argument('--patch-targets',type=Path,help='Experimental JSON patch ID to local target map; default sizes and GPU target remain unchanged')
    parser.add_argument('--graded-neighbors',action='store_true',help='Experimental interior size transition on faces touching regional stations')
    parser.add_argument("--only-requested-regions",action="store_true",help="Bounded reconstruction of explicit targets and required seam neighbors on a completed incumbent")
    parser.add_argument("--constrained-neighbors",action="store_true",help="Experimental complete boundary triangulation for graded neighbor candidates")
    args=parser.parse_args()
    try:
        require(args.input.resolve()!=args.output.resolve(), 'input snapshot must remain immutable')
        require(args.report.resolve() not in (args.input.resolve(),args.output.resolve()), 'report path collides with snapshot')
        require(math.isfinite(args.target) and args.target>=0, 'invalid target length')
        require(math.isfinite(args.max_error) and args.max_error>=0, 'invalid geometry error')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        expanded=set(); expansion_reasons=[];started=time.perf_counter();source_cache={}
        patch_targets=read_patch_targets(args.patch_targets,args.input) if args.patch_targets else None
        for attempt in range(9):
            try:
                report=repair(args.input,args.output,args.target,args.residual,args.previous_report,args.max_error,expanded,source_cache,patch_targets,args.graded_neighbors,args.only_requested_regions,args.constrained_neighbors)
                break
            except NeighborRepairRequired as exc:
                require(exc.patch not in expanded and attempt<8, f'neighbor expansion stopped: {exc}')
                expanded.add(exc.patch); expansion_reasons.append(str(exc))
                print(f'strip_repair expanding_neighbor={exc.patch}',flush=True)
        report['only_requested_regions']=args.only_requested_regions
        report['constrained_neighbors']=args.constrained_neighbors
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
