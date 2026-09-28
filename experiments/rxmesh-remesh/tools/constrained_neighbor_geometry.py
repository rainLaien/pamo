"""Experimental constrained source-triangle triangulation in output precision.
Boundary IDs and segments remain intact. Existing source geometry is untouched.
"""
import numpy as np
from narrow_strip_geometry import quality,require

def relax_interior_points(points, faces, boundary_count, normal, threshold,
                          max_passes=4):
    """Optional fixed-connectivity candidate; all decisions use output precision.

    Boundary stations are immutable. Callers still evaluate the complete region
    against its incumbent: improving a reconstructed star is not acceptance.
    """
    points=np.asarray(points,float).copy();faces=np.asarray(faces,int)
    require(np.isfinite(points).all(), 'nonfinite relaxation input')
    require(0<=boundary_count<=len(points), 'invalid relaxation boundary')
    require(0<=max_passes<=4, 'relaxation exceeds four-pass work budget')
    unit=np.asarray(normal,float);length=np.linalg.norm(unit)
    require(np.isfinite(length) and length>0, 'invalid relaxation normal')
    unit=unit/length
    incidence=[[] for _ in points]
    for fi,face in enumerate(faces):
        for index in face:incidence[index].append(fi)
    def metrics(triangles):
        t=triangles.astype(np.float32).astype(float)
        e=t-np.roll(t,-1,axis=1)
        area2=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
        q=2*np.sqrt(3)*area2/(e*e).sum((1,2))
        return np.array([q.min(),q.mean(),np.dot(q,area2)/area2.sum(),
                         -area2[q<threshold].sum(),np.linalg.norm(e,axis=2).min()])
    moves=0
    for cycle in range(max_passes):
        changed=0
        for index in range(boundary_count,len(points)):
            star=faces[incidence[index]]
            if not len(star):continue
            neighbors=np.unique(star);neighbors=neighbors[neighbors!=index]
            origin=points[index].copy()
            shift=points[neighbors].mean(0)-origin
            shift-=np.dot(shift,unit)*unit
            before=metrics(points[star]);best=before;chosen=None
            for fraction in (.5,.25,.125):
                trial=origin+fraction*shift;points[index]=trial
                if not all(positive(points,face,normal) for face in star):continue
                after=metrics(points[star])
                tolerance=1e-12*np.maximum(1,np.abs(before))
                if (after>=before-tolerance).all() and after[1]>best[1]+1e-10:
                    chosen=trial.copy();best=after
            points[index]=origin if chosen is None else chosen
            if chosen is not None:changed+=1
        moves+=changed
        if not changed:break
    return points,dict(moves=moves,passes=cycle+1 if max_passes else 0)

def positive(points,ids,normal):
    t=points[np.asarray(ids)];q=t.astype(np.float32).astype(float)
    return (np.cross(t[1]-t[0],t[2]-t[0])@normal>0 and
            np.cross(q[1]-q[0],q[2]-q[0])@normal>0)

def polygon_dp(points,n,normal):
    require(3<=n<=128,'constrained boundary exceeds 128-point work budget')
    scores=np.full((n,n),-1.);choice={}
    for i in range(n-1):scores[i,i+1]=1.
    for width in range(2,n):
        for i in range(n-width):
            j=i+width
            for k in range(i+1,j):
                if min(scores[i,k],scores[k,j])<=0 or not positive(points,[i,k,j],normal):continue
                score=min(scores[i,k],scores[k,j],quality(points[[i,k,j]].astype(np.float32)))
                if score>scores[i,j]:scores[i,j]=score;choice[i,j]=k
    require(scores[0,n-1]>0,'no complete representable boundary triangulation')
    def visit(i,j):
        if j-i<2:return []
        k=choice[i,j];return [[i,k,j]]+visit(i,k)+visit(k,j)
    return visit(0,n-1)

def triangulate(points,boundary_count,normal):
    points=np.asarray(points,float);faces=polygon_dp(points,boundary_count,normal)
    require(len(points)-boundary_count<=4096,'constrained interior exceeds work budget')
    # Strict interior insertions keep the initial boundary triangulation intact.
    # A point on an existing diagonal is not mandatory and may remain unused.
    for index in range(boundary_count,len(points)):
        for fi,face in enumerate(faces):
            t=points[face];dots=np.array([np.cross(t[(k+1)%3]-t[k],points[index]-t[k])@normal for k in range(3)])
            if not (dots>1e-12*np.linalg.norm(normal)**2).all():continue
            children=[[face[k],face[(k+1)%3],index] for k in range(3)]
            if all(positive(points,c,normal) for c in children):faces[fi]=children[0];faces.extend(children[1:])
            break
    boundary={tuple(sorted((i,(i+1)%boundary_count))) for i in range(boundary_count)}
    # Improve unconstrained diagonals. Each accepted flip preserves a disk and
    # improves its minimum quality; repeated passes are bounded.
    for cycle in range(24):
        edges={}
        for fi,face in enumerate(faces):
            for a,b in zip(face,face[1:]+face[:1]):edges.setdefault(tuple(sorted((a,b))),[]).append((fi,a,b))
        touched=set();changed=0
        for edge,owners in list(edges.items()):
            if edge in boundary or len(owners)!=2:continue
            (fa,u,v),(fb,_,_)=owners
            if fa in touched or fb in touched:continue
            x=next(z for z in faces[fa] if z not in edge);y=next(z for z in faces[fb] if z not in edge)
            if x==y or tuple(sorted((x,y))) in edges:continue
            a,b=[x,y,v],[y,x,u]
            if not positive(points,a,normal) or not positive(points,b,normal):continue
            old=min(quality(points[faces[fa]].astype(np.float32)),quality(points[faces[fb]].astype(np.float32)))
            new=min(quality(points[a].astype(np.float32)),quality(points[b].astype(np.float32)))
            if new<=old+1e-9:continue
            faces[fa],faces[fb]=a,b;touched.update((fa,fb));changed+=1
            edges[tuple(sorted((x,y)))]=[]
        if not changed:break
    counts={}
    for face in faces:
        for a,b in zip(face,face[1:]+face[:1]):key=tuple(sorted((a,b)));counts[key]=counts.get(key,0)+1
    require({k for k,c in counts.items() if c==1}==boundary,'constrained boundary changed')
    require(all(c<=2 for c in counts.values()),'constrained nonmanifold edge')
    return np.asarray(faces,int)
