"""Export completed mesh for bounded regional reconstruction experiments.
No coordinate welding or geometric mutation. Constraint lineage is mapped only
through existing edges; ambiguous anchors fail rather than combining sheets.
"""
import argparse, json, time
from collections import deque
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from audit_phase1 import read_ply
from repair_narrow_strips import read_snapshot, write_snapshot, audit, require


def export(incumbent,source,output):
    start=time.perf_counter();p,f,labels=read_ply(incumbent)
    header,sp,records,patches,features=read_snapshot(source)
    sp=sp.astype(np.float32).astype(float)
    e=f[:,[0,1,1,2,2,0]].reshape(-1,2)
    keys=(e.min(1).astype(np.uint64)<<32)|e.max(1).astype(np.uint64)
    order=keys.argsort(kind='stable');un,starts,counts=np.unique(keys[order],return_index=True,return_counts=True)
    adjacency=[[] for _ in p]
    for key in un:
        a,b=int(key>>np.uint64(32)),int(key & np.uint64(0xffffffff))
        adjacency[a].append(b);adjacency[b].append(a)
    tree=cKDTree(p);distance,mapped=tree.query(sp)
    sourceanchors=np.unique(features[:,1:3]);require((distance[sourceanchors]==0).all(),'source anchor moved')
    for anchor in sourceanchors:
        require(len(tree.query_ball_point(sp[anchor],0))==1,'ambiguous source anchor; topology-aware mapping required')
    output_owners={int(k):tuple(sorted(map(int,labels[order[start:start+count]//3]))) for k,start,count in zip(un,starts,counts)}
    se=records[:,[0,1,1,2,2,0]].reshape(-1,2)
    sk=(se.min(1).astype(np.uint64)<<32)|se.max(1).astype(np.uint64)
    so=sk.argsort(kind='stable');su,ss,sc=np.unique(sk[so],return_index=True,return_counts=True)
    source_owners={int(k):tuple(sorted(map(int,records[so[start:start+count]//3,3]))) for k,start,count in zip(su,ss,sc)}
    constraints={};lookup=set(map(int,un));tol=max(1e-6,np.linalg.norm(np.ptp(sp,axis=0))*1e-7)
    for curve,a,b,hard in features:
        owner=source_owners[(min(int(a),int(b))<<32)|max(int(a),int(b))]
        u,v=int(mapped[a]),int(mapped[b]);key=(min(u,v)<<32)|max(u,v)
        if key in lookup and output_owners[key]==owner:path=[u,v]
        else:
            delta=sp[b]-sp[a];norm=float(delta@delta);require(norm>0,'zero source constraint')
            queue=deque([u]);previous={u:None}
            while queue and v not in previous:
                z=queue.popleft()
                for w in adjacency[z]:
                    if w in previous or output_owners[(min(z,w)<<32)|max(z,w)]!=owner:continue
                    t=float((p[w]-sp[a])@delta/norm)
                    if -1e-6<=t<=1+1e-6 and np.linalg.norm(p[w]-(sp[a]+t*delta))<=tol:
                        previous[w]=z;queue.append(w)
            require(v in previous,'source constraint chain missing')
            path=[v]
            while path[-1]!=u:path.append(previous[path[-1]])
            path.reverse()
        for u,v in zip(path,path[1:]):
            key=(min(u,v)<<32)|max(u,v);value=(int(curve),min(u,v),max(u,v),int(hard))
            require(key not in constraints or constraints[key]==value,f'constraint lineage conflict: {key}, {constraints.get(key)}, {value}, source edge {int(a)}, {int(b)}')
            constraints[key]=value
    # All output interfaces must have a proven source constraint lineage.
    mask=counts!=2;two=np.flatnonzero(counts==2)
    mask[two]=labels[order[starts[two]]//3]!=labels[order[starts[two]+1]//3]
    require(all(int(k) in constraints for k in un[mask]),'unmapped output boundary/interface')
    outrecords=np.column_stack((f,labels)).astype('uint32')
    outfeatures=np.asarray(list(constraints.values()),dtype='uint32').reshape(-1,4)
    topology,_,_=audit(p,outrecords,outfeatures)
    require(all(topology[k]==0 for k in ('nonmanifold_edges','inconsistent_interior_edges','float32_zero_faces','duplicate_faces','missing_feature_edges')),'invalid incumbent')
    write_snapshot(output,header,p,outrecords,patches,outfeatures)
    return dict(vertices=len(p),faces=len(f),constraint_segments=len(outfeatures),topology=topology,
        coordinates_unchanged=True,triangles_unchanged=True,geometry_reference='original source remains separate and authoritative',seconds=time.perf_counter()-start)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('incumbent','source','output','report'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(args.output.resolve() not in (args.source.resolve(),args.incumbent.resolve()),'immutable sources required')
    require(args.report.resolve() not in (args.source.resolve(),args.incumbent.resolve(),args.output.resolve()),'report/source collision')
    report=export(args.incumbent,args.source,args.output);args.report.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
