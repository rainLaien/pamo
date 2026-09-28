"""Restore immutable source-anchor arithmetic without changing float32 output.
This is a computation snapshot for bounded reconstruction, not a new geometry
reference. Original STL and source snapshot remain authoritative error references.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from repair_narrow_strips import read_snapshot,write_snapshot,require

def restore_arrays(current,reference,anchors):
    p=np.asarray(current,float);rp=np.asarray(reference,float)
    anchors=np.unique(anchors);tree=cKDTree(p.astype(np.float32).astype(float))
    rounded=rp[anchors].astype(np.float32).astype(float);distance,mapping=tree.query(rounded)
    require((distance==0).all(),'immutable source anchor no longer matches output')
    require(len(np.unique(mapping))==len(mapping),'ambiguous source-anchor identities')
    require(all(len(tree.query_ball_point(q,0))==1 for q in rounded),'coincident output anchors require topology mapping')
    output=p.copy();output[mapping]=rp[anchors]
    require(np.array_equal(output.astype(np.float32),p.astype(np.float32)),'float32 geometry changed')
    return output,dict(restored_source_anchors=len(anchors),float32_coordinates_bitwise_identical=True,
                      max_double_restore=float(np.linalg.norm(output-p,axis=1).max(initial=0)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('incumbent','reference','output','report'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(args.output.resolve() not in (args.incumbent.resolve(),args.reference.resolve()),'immutable sources required')
    require(args.report.resolve() not in (args.incumbent.resolve(),args.reference.resolve(),args.output.resolve()),'report/source collision')
    started=time.perf_counter();h,p,r,ph,f=read_snapshot(args.incumbent);_,rp,rr,rph,rf=read_snapshot(args.reference)
    restored,report=restore_arrays(p,rp,rf[:,1:3].reshape(-1));write_snapshot(args.output,h,restored,r,ph,f)
    _,op,rows,_,features=read_snapshot(args.output)
    require(np.array_equal(op.astype(np.float32),p.astype(np.float32)) and np.array_equal(rows,r) and np.array_equal(features,f),'saved invariant failed')
    report.update(triangles_and_features_unchanged=True,source=str(args.incumbent.resolve()),reference=str(args.reference.resolve()),
        source_sha256=hashlib.sha256(args.incumbent.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),seconds=time.perf_counter()-started,
        original_geometry_reference_required=True)
    args.report.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
