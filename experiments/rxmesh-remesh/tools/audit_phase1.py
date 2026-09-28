"""Independent same-cutoff comparison; sampled distances are not Hausdorff proofs."""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from repair_narrow_strips import read_snapshot, audit


def read_ply(path, *, respect_coordinate_types=True):
    with path.open(encoding='utf-8') as f:
        nv = None; element=None; properties=[]
        for line in f:
            parts=line.split()
            if parts[:1]==['element']:
                element=parts[1]
                if element=='vertex':nv=int(parts[2])
            elif parts[:1]==['property'] and element=='vertex':
                if len(parts)!=3:raise ValueError('vertex list properties are unsupported')
                properties.append((parts[2],parts[1]))
            if line.strip() == 'end_header': break
        if nv is None:raise ValueError('missing PLY vertex element')
        vertices=np.loadtxt(f,max_rows=nv,ndmin=2)
        columns={name:(i,kind) for i,(name,kind) in enumerate(properties)}
        coordinate_types={'float':np.float32,'float32':np.float32,'double':np.float64,'float64':np.float64}
        xyz=[]
        for name in ('x','y','z'):
            index,kind=columns[name]
            dtype=coordinate_types[kind] if respect_coordinate_types else np.float64
            xyz.append(vertices[:,index].astype(dtype).astype(np.float64))
        p=np.column_stack(xyz)
        rows = np.loadtxt(f, dtype=np.uint32, ndmin=2)
    return p, rows[:, 1:4], rows[:, 4]


def quality_regions(p, faces, labels, threshold, target):
    tri = p[faces]
    sides = np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2)
    area = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    q = np.divide(4*np.sqrt(3)*area, (sides*sides).sum(1), out=np.zeros(len(faces)), where=(sides*sides).sum(1)>0)
    ee = faces[:, [0,1,1,2,2,0]].reshape(-1, 2)
    keys = (ee.min(1).astype(np.uint64)<<32) | ee.max(1).astype(np.uint64)
    order = np.argsort(keys, kind='stable')
    paired = np.flatnonzero(keys[order[:-1]] == keys[order[1:]])
    a, b = order[paired]//3, order[paired+1]//3
    low = q < threshold
    low_ids = np.flatnonzero(low)
    mapping = np.full(len(faces), -1); mapping[low_ids] = np.arange(len(low_ids))
    selected = low[a] & low[b]
    graph = coo_matrix((np.ones(selected.sum()), (mapping[a[selected]], mapping[b[selected]])), shape=(len(low_ids),len(low_ids))).tocsr()
    count, component = connected_components(graph, directed=False)
    component_area = np.bincount(component, weights=area[low_ids], minlength=count)
    unique, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    edge = np.column_stack([unique>>32, unique & np.uint64(0xffffffff)]).astype(int)
    ratios = np.linalg.norm(p[edge[:,0]]-p[edge[:,1]],axis=1)/target
    scales=np.sqrt((sides*sides).mean(1))
    transition=np.maximum(scales[a],scales[b])/np.maximum(np.minimum(scales[a],scales[b]),np.finfo(float).tiny)
    metrics = dict(quality_mean=float(q.mean()), quality_p05=float(np.partition(q,len(q)//20)[len(q)//20]),
                   quality_min=float(q.min()), area_weighted_mean=float(np.dot(q,area)/area.sum()),
                   threshold=threshold, low_quality_faces=int(low.sum()), low_quality_area=float(area[low].sum()),
                   low_quality_components=int(count), largest_low_quality_area=float(component_area.max(initial=0)),
                   long_edges=int((ratios>(4/3)*(1+1e-4)).sum()), short_edges=int((ratios<.8).sum()),
                   edge_ratio_p05=float(np.quantile(ratios,.05)), edge_ratio_p95=float(np.quantile(ratios,.95)),
                   adjacent_face_size_ratio_p95=float(np.quantile(transition,.95)) if len(transition) else 1.,
                   adjacent_face_size_ratio_max=float(transition.max(initial=1.)),
                   faces=len(faces), vertices=len(p))
    return metrics, q, area, edge, (a,b), counts


def chain_audit(source_p, source_edges, output_p, output_edges, tolerance):
    # Retain every original anchor and the ordered geometric coverage of its
    # segments. Subdivided segments may have new IDs; coordinate welding is not used.
    distance, mapped = cKDTree(output_p).query(source_p)
    keys = set(((output_edges[:,0].astype(np.uint64)<<32)|output_edges[:,1].astype(np.uint64)).tolist())
    adjacency = [[] for _ in range(len(output_p))]
    for u,v in output_edges: adjacency[u].append(v); adjacency[v].append(u)
    lost = 0
    for a,b in source_edges:
        u,v = int(mapped[a]),int(mapped[b])
        if max(distance[a],distance[b])>tolerance: lost+=1; continue
        if (min(u,v)<<32)|max(u,v) in keys: continue
        origin=source_p[a]; delta=source_p[b]-origin; norm=float(delta@delta)
        if norm==0: lost+=1; continue
        queue=deque([u]); seen={u}; found=False
        while queue:
            x=queue.popleft()
            if x==v: found=True; break
            for y in adjacency[x]:
                if y in seen: continue
                t=float((output_p[y]-origin)@delta/norm)
                if -1e-6<=t<=1+1e-6 and np.linalg.norm(output_p[y]-(origin+t*delta))<=tolerance:
                    seen.add(y);queue.append(y)
        if not found: lost+=1
    anchors=np.unique(source_edges) if len(source_edges) else np.array([],dtype=int)
    return dict(segments=len(source_edges), missing_chains=lost,
                missing_anchors=int((distance[anchors]>tolerance).sum()),
                max_anchor_error=float(distance[anchors].max(initial=0)), tolerance=tolerance)


def distance_samples(mesh, points, query=None):
    from cached_proximity import CachedSurfaceQuery
    if query is None: query=CachedSurfaceQuery(mesh)
    distances=[]; normals=[]
    for start in range(0,len(points),64):
        _,d,ids=query.on_surface(points[start:start+64])
        distances.extend(d); normals.extend(mesh.face_normals[ids])
    return np.asarray(distances),np.asarray(normals)


def area_samples(mesh, count=2000):
    rng=np.random.default_rng(7401)
    ids=rng.choice(len(mesh.faces),count,p=mesh.area_faces/mesh.area)
    uv=rng.random((count,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    t=mesh.triangles[ids]
    return t[:,0]+uv[:,:1]*(t[:,1]-t[:,0])+uv[:,1:]*(t[:,2]-t[:,0])


def selected_region_quality(q, area, mask, patch_count, threshold):
    selected = q[mask]
    return dict(patches=patch_count, faces=len(selected),
                quality_mean=float(selected.mean()) if len(selected) else None,
                quality_p05=float(np.partition(selected, len(selected)//20)[len(selected)//20]) if len(selected) else None,
                low_quality_area=float(area[mask][selected < threshold].sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','snapshot','repair_report','baseline','candidate','output'):
        parser.add_argument('--'+name.replace('_','-'),required=True,type=Path)
    parser.add_argument('--threshold',required=True,type=float)
    parser.add_argument('--target',required=True,type=float)
    parser.add_argument('--max-error',required=True,type=float)
    args=parser.parse_args();start=time.perf_counter()
    _,p,records,patches,features=read_snapshot(args.snapshot)
    p=p.astype(np.float32).astype(float)
    source=trimesh.load(args.source,force='mesh',process=False)
    sp,inv=np.unique(source.vertices,axis=0,return_inverse=True)
    sf=inv[source.faces];source=trimesh.Trimesh(sp,sf,process=False)
    topology,source_keys,source_counts=audit(p,records,features)
    source_edge=np.column_stack([source_keys>>32,source_keys & np.uint64(0xffffffff)]).astype(int)
    srcq,_,_,_,(left,right),_=quality_regions(p,records[:,:3],records[:,3],args.threshold,args.target)
    source_seams=records[left,3]!=records[right,3]
    seam_keys=set()
    for a,b in zip(left[source_seams],right[source_seams]):
        ends=sorted(set(records[a,:3]) & set(records[b,:3])); seam_keys.add(tuple(ends))
    seam_edges=np.asarray(sorted(seam_keys),dtype=int).reshape(-1,2)
    strips=set(map(int,json.loads(args.repair_report.read_text())['rebuilt_patches']))
    result=dict(source=str(args.source.resolve()),snapshot=str(args.snapshot.resolve()),
                metric_basis='float64 arithmetic on declared PLY coordinate types; same cutoff and order-statistic P05; connected across patch interfaces',
                target=args.target,max_error=args.max_error,threshold=args.threshold,
                source_snapshot_topology=topology,source_snapshot_quality=srcq,outputs={})
    for name,path in [('baseline',args.baseline),('candidate',args.candidate)]:
        op,faces,labels=read_ply(path)
        mesh=trimesh.Trimesh(op,faces,process=False)
        row,q,area,edges,(a,b),counts=quality_regions(op,faces,labels,args.threshold,args.target)
        rows=np.column_stack([faces,labels]);row['topology']=audit(op,rows,np.empty((0,4),dtype=np.uint32))[0]
        row['feature_chains']=chain_audit(p,features[:,1:3],op,edges,max(1e-6,np.linalg.norm(np.ptp(p,axis=0))*1e-7))
        # Restrict seam paths to actual output partition interfaces.
        seam_set=set()
        for x,y in zip(a[labels[a]!=labels[b]],b[labels[a]!=labels[b]]):
            seam_set.add(tuple(sorted(set(faces[x]) & set(faces[y]))))
        row['partition_seams']=chain_audit(p,seam_edges,op,np.asarray(sorted(seam_set),dtype=int).reshape(-1,2),row['feature_chains']['tolerance'])
        row['open_boundary']=chain_audit(p,source_edge[source_counts==1],op,edges[counts==1],row['feature_chains']['tolerance'])
        strip_mask=np.isin(labels,list(strips))
        row['repaired_strips']=selected_region_quality(q,area,strip_mask,len(strips),args.threshold)
        samples=area_samples(mesh)
        low_ids=np.flatnonzero(q<args.threshold)
        if len(low_ids)>2000:low_ids=low_ids[np.linspace(0,len(low_ids)-1,2000,dtype=int)]
        points=np.concatenate([samples,mesh.triangles_center[low_ids]])
        forward,_=distance_samples(source,points)
        original_samples=area_samples(source)
        backward,_=distance_samples(mesh,original_samples)
        row['geometry_samples']=dict(forward_count=len(points),backward_count=2000,
            output_to_original_max=float(forward.max()),original_to_output_max=float(backward.max()),
            within_budget=bool(max(forward.max(),backward.max())<=args.max_error),proof=False,
            output_worst_point=points[int(np.argmax(forward))].tolist(),
            original_worst_point=original_samples[int(np.argmax(backward))].tolist())
        # Owner-specific exact nearest triangles for centroid orientation checks.
        flipped=0;checked=0;negative_area=0.;negative_in_strips=0;negative_by_type={};examples=[]
        source_order=np.argsort(records[:,3],kind='stable');source_offsets=np.r_[0,np.cumsum(np.bincount(records[:,3]))]
        out_order=np.argsort(labels,kind='stable');out_offsets=np.r_[0,np.cumsum(np.bincount(labels,minlength=len(source_offsets)-1))]
        for patch in range(len(source_offsets)-1):
            srcfaces=records[source_order[source_offsets[patch]:source_offsets[patch+1]],:3]
            indices=out_order[out_offsets[patch]:out_offsets[patch+1]]
            if not len(indices):continue
            ids=np.unique(srcfaces);ref=trimesh.Trimesh(p[ids],np.searchsorted(ids,srcfaces),process=False)
            reference_distances,normal=distance_samples(ref,mesh.triangles_center[indices])
            dots=np.einsum('ij,ij->i',mesh.face_normals[indices],normal)
            negative=np.flatnonzero(dots<=0)
            flipped+=len(negative);checked+=len(indices)
            negative_area+=float(area[indices[negative]].sum())
            if patch in strips:negative_in_strips+=len(negative)
            kind=str(patches[patch][0][0]);negative_by_type[kind]=negative_by_type.get(kind,0)+len(negative)
            for i in negative:
                f=int(indices[i]);examples.append(dict(face=f,patch_id=patch,patch_type=kind,
                    area=float(area[f]),quality=float(q[f]),reference_distance=float(reference_distances[i]),
                    normal_dot=float(dots[i]),centroid=mesh.triangles_center[f].tolist()))
        row['reference_orientation']=dict(centroids_checked=checked,nonpositive_normal_dots=flipped,
             negative_area=negative_area,negative_in_repaired_strips=negative_in_strips,
             negative_by_patch_type=negative_by_type,largest_area_examples=sorted(examples,key=lambda x:x['area'],reverse=True)[:16],
             basis='nearest source triangle in same patch at every output centroid; not a self-intersection proof')
        result['outputs'][name]=row
        print(name,json.dumps(row),flush=True)
    result['audit_seconds']=time.perf_counter()-start
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
