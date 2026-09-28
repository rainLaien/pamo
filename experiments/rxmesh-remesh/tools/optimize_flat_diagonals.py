"""Bounded experimental flat-diagonal candidates on a completed incumbent.
Exact axis-coordinate witnesses preserve the piecewise-linear surface. This is
an offline stage experiment, not a full-pipeline timing or endpoint certificate.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
from audit_phase1 import read_ply
from repair_narrow_strips import read_snapshot, require


def metrics(p, faces):
    t=p[faces]; n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); area=np.linalg.norm(n,axis=1)*.5
    d=t-np.roll(t,-1,axis=1); denom=(d*d).sum((1,2))
    q=np.divide(4*np.sqrt(3)*area,denom,out=np.zeros(len(faces)),where=denom>0)
    return q,area,n


def optimize(p, faces, labels, protected, target, threshold, passes=2):
    out=faces.copy(); reports=[]
    for cycle in range(passes):
        ends=out[:,[0,1,1,2,2,0]].reshape(-1,2)
        keys=(ends.min(1).astype(np.uint64)<<32)|ends.max(1).astype(np.uint64)
        order=keys.argsort(kind='stable'); unique,starts,counts=np.unique(keys[order],return_index=True,return_counts=True)
        pos=starts[counts==2]; a=order[pos]//3; b=order[pos+1]//3
        u=ends[order[pos],0]; v=ends[order[pos],1]
        mask=(labels[a]==labels[b]) & ~protected[u] & ~protected[v]
        a,b,u,v=a[mask],b[mask],u[mask],v[mask]
        x=np.sum(out[a],axis=1,dtype=np.int64)-u.astype(np.int64)-v.astype(np.int64)
        y=np.sum(out[b],axis=1,dtype=np.int64)-u.astype(np.int64)-v.astype(np.int64)
        quad=p[np.column_stack((u,v,x,y))]
        flat=(quad.min(1)==quad.max(1)).any(1)
        nk=(np.minimum(x,y).astype(np.uint64)<<32)|np.maximum(x,y).astype(np.uint64)
        ix=np.searchsorted(unique,nk); exists=unique[np.minimum(ix,len(unique)-1)]==nk
        oldlength=np.linalg.norm(p[u]-p[v],axis=1); newlength=np.linalg.norm(p[x]-p[y],axis=1)
        mask=flat & ~exists & (x!=y) & (newlength<=oldlength) & (newlength>=target*4/5) & (newlength<=target*4/3)
        a,b,u,v,x,y=[z[mask] for z in (a,b,u,v,x,y)]
        # Orient from the shared directed edge in the first triangle.
        fa=out[a]; forward=((fa==u[:,None]) & (np.roll(fa,-1,axis=1)==v[:,None])).any(1)
        n1=np.column_stack((x,y,np.where(forward,v,u)))
        n2=np.column_stack((y,x,np.where(forward,u,v)))
        oq,oa,on=metrics(p,np.vstack((out[a],out[b]))); nq,na,nn=metrics(p,np.vstack((n1,n2)))
        oq=oq.reshape(2,-1); nq=nq.reshape(2,-1); oa=oa.reshape(2,-1); na=na.reshape(2,-1)
        normok=(np.einsum('ij,ij->i',nn[:len(a)],on[:len(a)])>0)&(np.einsum('ij,ij->i',nn[len(a):],on[len(a):])>0)
        sortedold=np.sort(oq,axis=0); sortednew=np.sort(nq,axis=0)
        good=normok & (sortednew[0]>sortedold[0]+1e-9) & (nq.sum(0)>=oq.sum(0))
        good &= (nq*na).sum(0)>=(oq*oa).sum(0)
        good &= (na*(nq<threshold)).sum(0)<=(oa*(oq<threshold)).sum(0)
        used=set(); accepted=0; currentkeys=set(map(int,unique))
        for i in np.flatnonzero(good):
            if int(a[i]) in used or int(b[i]) in used:continue
            # Four distinct vertices and existing-diagonal exclusion preserve
            # the manifold disk. Face-disjoint choices avoid concurrent cavities.
            if len({int(u[i]),int(v[i]),int(x[i]),int(y[i])})!=4:continue
            newkey=(min(int(x[i]),int(y[i]))<<32)|max(int(x[i]),int(y[i]))
            if newkey in currentkeys:continue
            currentkeys.add(newkey)
            currentkeys.discard((min(int(u[i]),int(v[i]))<<32)|max(int(u[i]),int(v[i])))
            out[a[i]]=n1[i];out[b[i]]=n2[i];used.update((int(a[i]),int(b[i])));accepted+=1
        reports.append(dict(pass_index=cycle,certified_candidates=len(a),pair_eligible=int(good.sum()),accepted_flips=accepted))
        if not accepted:break
    return out,reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('incumbent','snapshot','output','report'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--target',type=float,required=True);parser.add_argument('--threshold',type=float,required=True)
    args=parser.parse_args();start=time.perf_counter()
    require(args.output.resolve() not in (args.incumbent.resolve(),args.snapshot.resolve()),'immutable source and incumbent required')
    require(args.report.resolve() not in (args.incumbent.resolve(),args.snapshot.resolve(),args.output.resolve()),'report/source collision')
    require(np.isfinite(args.target) and args.target>0 and np.isfinite(args.threshold) and 0<=args.threshold<=1,'invalid target or threshold')
    p,f,l=read_ply(args.incumbent);_,sp,sf,patches,features=read_snapshot(args.snapshot)
    sp=sp.astype(np.float32).astype(float)
    protected=np.zeros(len(p),bool)
    # Every vertex on any patch seam or open/nonmanifold boundary remains fixed.
    e=f[:,[0,1,1,2,2,0]].reshape(-1,2);k=(e.min(1).astype(np.uint64)<<32)|e.max(1).astype(np.uint64)
    o=k.argsort(kind='stable');un,s,c=np.unique(k[o],return_index=True,return_counts=True)
    boundary=c!=2; two=np.flatnonzero(c==2);boundary[two]=l[o[s[two]]//3]!=l[o[s[two]+1]//3]
    protected[e[o[s[boundary]]].reshape(-1)]=True
    # Coordinate lookup is only for overinclusive protection, never ID welding.
    tolerance=max(1e-6,np.linalg.norm(np.ptp(sp,axis=0))*1e-7)
    tree=cKDTree(p); distances,mapped=tree.query(sp)
    adjacency=[[] for _ in p]
    for key in un:
        u,v=int(key>>np.uint64(32)),int(key & np.uint64(0xffffffff))
        adjacency[u].append(v);adjacency[v].append(u)
    edgekeys=set(map(int,un))
    for a,b in features[:,1:3]:
        require(max(distances[a],distances[b])<=tolerance,'missing source constraint anchor')
        u,v=int(mapped[a]),int(mapped[b]);protected[u]=protected[v]=True
        for anchor in (a,b):
            protected[tree.query_ball_point(sp[anchor],tolerance)]=True
        if ((min(u,v)<<32)|max(u,v)) in edgekeys:continue
        delta=sp[b]-sp[a];norm=float(delta@delta);require(norm>0,'zero source constraint')
        queue=deque([u]);seen={u};found=False
        while queue:
            z=queue.popleft();protected[z]=True
            if z==v:found=True;break
            for w in adjacency[z]:
                if w in seen:continue
                t=float((p[w]-sp[a])@delta/norm)
                if -1e-6<=t<=1+1e-6 and np.linalg.norm(p[w]-(sp[a]+t*delta))<=tolerance:
                    protected[w]=True;seen.add(w);queue.append(w)
        require(found,'missing source constraint chain')
    out,cycles=optimize(p,f,l,protected,args.target,args.threshold)
    # Keep original header and vertex text byte-for-byte.
    with args.incumbent.open() as source,args.output.open('w',newline=chr(10)) as dest:
        for line in source:
            dest.write(line)
            if line.strip()=='end_header':break
        for _ in range(len(p)):dest.write(source.readline())
        np.savetxt(dest,np.column_stack((np.full(len(out),3),out,l)),fmt='%d')
    changed=np.any(f!=out,axis=1)
    report=dict(cycles=cycles,changed_faces=int(changed.sum()),changed_patches=np.unique(l[changed]).tolist(),seconds=time.perf_counter()-start,
                coordinates_unchanged=True,axis_plane_witness=True,geometry_tolerance_used=0,stage_only=True,
                acceptance='pair guards only; independent connected-region and hard-constraint audit required')
    args.report.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
