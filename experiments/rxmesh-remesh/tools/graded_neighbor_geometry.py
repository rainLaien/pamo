"""Bounded planar-source-face candidate with a gradual interior size field.

All supplied boundary stations are retained. No new outer boundary stations or
surface displacement are introduced. This is an optional candidate, not a gate.
"""
import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree
from narrow_strip_geometry import require


def triangulate_graded(poly, vertices, normal, source_triangle, refined_edges, local_h, global_h, constrained=False):
    origin=source_triangle[0]
    unit_normal=normal/np.linalg.norm(normal)
    u=source_triangle[1]-origin;u=u/np.linalg.norm(u)
    v=np.cross(unit_normal,u)
    boundary=np.asarray([vertices[x] for x in poly])
    basis=np.column_stack([u,v])
    uv=(boundary-origin)@basis
    n=int(np.ceil(np.linalg.norm(source_triangle-np.roll(source_triangle,-1,axis=0),axis=1).max()/local_h))
    require(n<=384, 'graded neighbor lattice exceeds 384-station work budget')
    i,j=np.triu_indices(max(n-1,0))
    # Positive barycentric coordinates, with all three sides excluded.
    aa=i+1;bb=j-i+1
    valid=aa+bb<n;aa,bb=aa[valid],bb[valid]
    samples=origin+aa[:,None]/n*(source_triangle[1]-origin)+bb[:,None]/n*(source_triangle[2]-origin) if n>1 else np.empty((0,3))
    h=np.full(len(samples),global_h)
    distances=[]
    for a,b in zip(source_triangle,np.roll(source_triangle,-1,axis=0)):
        d=b-a;t=np.clip(((samples-a)@d)/(d@d),0,1)
        distances.append(np.linalg.norm(samples-a-t[:,None]*d,axis=1))
    for edge in refined_edges:
        h=np.minimum(h,local_h+.5*distances[edge])
    mask=np.ones(len(samples),bool)
    for edge in set(range(3))-set(refined_edges):mask &= distances[edge]>=.45*h
    samples,h=samples[mask],h[mask]
    accepted=[];pending=[];tree=cKDTree(uv)
    for idx in np.argsort(h,kind='stable'):
        q=(samples[idx]-origin)@basis;spacing=.8*h[idx]
        if tree.query(q)[0]<spacing:continue
        if pending and np.linalg.norm(np.asarray(pending)-q,axis=1).min()<spacing:continue
        accepted.append(samples[idx]);pending.append(q)
        require(len(accepted)<=4096, 'graded neighbor interior exceeds 4096-point work budget')
        if len(pending)>=64:
            tree=cKDTree(np.vstack([uv,(np.asarray(accepted)-origin)@basis]));pending=[]
    points=np.vstack([boundary,np.asarray(accepted).reshape(-1,3)])
    if constrained:
        from constrained_neighbor_geometry import triangulate
        faces=triangulate(points,len(poly),normal)
    else:
        try:faces=Delaunay((points-origin)@basis).simplices.copy()
        except QhullError as exc:raise ValueError('graded neighbor triangulation failed') from exc
    require(np.isin(np.arange(len(poly)),faces).all(), 'graded neighbor lost boundary station')
    signs=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])@normal
    reverse=signs<0;faces[reverse]=faces[reverse][:,[0,2,1]]
    quantized=points.astype(np.float32).astype(float)
    signs32=np.cross(quantized[faces[:,1]]-quantized[faces[:,0]],quantized[faces[:,2]]-quantized[faces[:,0]])@normal
    require((signs32>0).all(), 'graded neighbor has float32 degenerate or reversed face')
    counts={}
    for f in faces:
        for a,b in zip(f,np.roll(f,-1)):
            key=tuple(sorted((int(a),int(b))));counts[key]=counts.get(key,0)+1
    expected={tuple(sorted((a,(a+1)%len(poly)))) for a in range(len(poly))}
    require({k for k,c in counts.items() if c==1}==expected, 'graded neighbor boundary is not conforming')
    generated=len(accepted)
    if constrained:
        used=np.unique(faces)
        require(np.isin(np.arange(len(poly)),used).all(), 'constrained neighbor lost boundary station')
        accepted=points[used[used>=len(poly)]]
        remap=np.full(len(points),-1,dtype=int);remap[used]=np.arange(len(used))
        faces=remap[faces]
    mapping=list(poly)
    for q in accepted:mapping.append(len(vertices));vertices.append(q)
    info=dict(interior_vertices=len(accepted),growth=.5,local_h=local_h,global_h=global_h)
    if constrained:info.update(interior_candidates=generated,triangulation='constrained')
    return np.asarray(mapping)[faces].tolist(),info
