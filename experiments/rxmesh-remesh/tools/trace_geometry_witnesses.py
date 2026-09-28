"""Trace previously measured original-surface error witnesses through stages."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from repair_narrow_strips import read_snapshot
from audit_phase1 import read_ply


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--repaired',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prior=json.loads(args.audit.read_text(encoding='utf-8'))
    points=np.asarray([x['geometry_samples']['original_worst_point'] for x in prior['outputs'].values()])
    report=dict(points=points.tolist(),point_names=list(prior['outputs']),
                scope='Previously sampled witnesses only, not a new whole-mesh error certificate',stages={})
    for name,path in [('partition_snapshot',args.snapshot),('repaired_snapshot',args.repaired),('baseline',args.baseline),('phase1',args.candidate)]:
        if path.suffix=='.cadpart':
            _,p,r,_,_=read_snapshot(path);f,labels=r[:,:3],r[:,3]
        else:p,f,labels=read_ply(path)
        mesh=trimesh.Trimesh(p,f,process=False)
        closest,d,ids=trimesh.proximity.closest_point(mesh,points)
        report['stages'][name]=dict(path=str(path.resolve()),distances=d.tolist(),patches=labels[ids].tolist(),
                                  faces=ids.tolist(),closest_points=closest.tolist())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
