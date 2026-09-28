"""Extract complete existing compute blocks for bounded diagnostic experiments.

An extracted run is not a complete pipeline benchmark. External boundaries are
open and will be frozen by the existing batch policy. Keep the same global h/error.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from repair_narrow_strips import read_snapshot, write_snapshot


def extract(source, output, selected, block_size=16):
    if block_size<1:raise ValueError('block size must be positive')
    header,p,rows,patches,features=read_snapshot(source)
    if output.resolve()==source.resolve():raise ValueError('source must remain immutable')
    if not selected or any(x<0 or x>=len(patches) for x in selected):raise ValueError('invalid patch selection')
    blocks=sorted({x//block_size for x in selected})
    # Support records are part of the snapshot contract. Retain complete blocks
    # until dependency closure; never erase them merely to shrink an experiment.
    initial_blocks=blocks.copy()
    for _ in range(len(patches)):
        ids=[x for b in blocks for x in range(b*block_size,min((b+1)*block_size,len(patches)))]
        extra={int(x)//block_size for rid in ids for x in np.frombuffer(patches[rid][1][:patches[rid][0][4]*4],'<u4')}-set(blocks)
        if not extra:break
        blocks=sorted(set(blocks)|extra)
        if len(blocks)*block_size>1024:raise ValueError('support closure exceeds diagnostic 1024-patch budget')
    ids=np.asarray([x for b in blocks for x in range(b*block_size,min((b+1)*block_size,len(patches)))],int)
    chosen=rows[np.isin(rows[:,3],ids)].copy()
    vertices=np.unique(chosen[:,:3]);chosen[:,:3]=np.searchsorted(vertices,chosen[:,:3])
    chosen[:,3]=np.searchsorted(ids,chosen[:,3])
    edges=rows[np.isin(rows[:,3],ids)][:,[0,1,1,2,2,0]].reshape(-1,2).astype(np.uint64)
    keys=(edges.min(1)<<32)|edges.max(1)
    feature_keys=(features[:,1:3].min(1).astype(np.uint64)<<32)|features[:,1:3].max(1).astype(np.uint64)
    kept=features[np.isin(feature_keys,keys)].copy()
    kept[:,1:3]=np.searchsorted(vertices,kept[:,1:3])
    records=[]
    for rid in ids:
        ph,tail=patches[int(rid)];ph=ph.copy()
        supports=np.frombuffer(tail[:ph[4]*4],'<u4')
        if not np.isin(supports,ids).all():raise ValueError('external support patches must be retained before extraction')
        mapped=np.searchsorted(ids,supports).astype('<u4').tobytes()
        records.append((ph,mapped+tail[ph[4]*4:]))
    output.parent.mkdir(parents=True,exist_ok=True)
    write_snapshot(output,header,p[vertices],chosen,records,kept)
    return dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                output=str(output.resolve()),block_size=block_size,original_patch_ids=ids.tolist(),
                initial_blocks=initial_blocks,support_closed_blocks=blocks,
                original_vertex_ids=vertices.tolist(),faces=len(chosen),vertices=len(vertices),
                complete_pipeline=False,external_boundaries_frozen=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--patches',type=int,nargs='+',required=True)
    parser.add_argument('--block-size',type=int,default=16)
    args=parser.parse_args()
    if args.block_size<1:parser.error('block size must be positive')
    report=extract(args.source,args.output,args.patches,args.block_size)
    Path(str(args.output)+'.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['faces','vertices','original_patch_ids']}))


if __name__=='__main__':main()
