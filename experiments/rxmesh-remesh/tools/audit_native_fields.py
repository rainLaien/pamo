"""Validate exported native sizes and immutable source evidence independently.
This binds sidecars to saved coordinates and the supplied snapshot, not to a
certified CAD model. It does not replace geometry, topology or flip audits.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from audit_phase1 import read_ply
from repair_narrow_strips import read_snapshot

def verify(mesh_path,snapshot_path):
 start=time.perf_counter();prefix=str(mesh_path)
 metadata=json.loads(Path(prefix+'.fields.json').read_text())
 if metadata.get('complete') is not True:raise ValueError('incomplete export')
 p,f,labels=read_ply(mesh_path)
 vertices=np.loadtxt(prefix+'.vertices.tsv',skiprows=1,ndmin=2)
 edges=np.loadtxt(prefix+'.edges.tsv',skiprows=1,ndmin=2)
 source=np.loadtxt(prefix+'.source_constraints.tsv',skiprows=1,ndmin=2,dtype=np.uint64)
 if len(vertices)!=len(p) or len(edges)!=metadata['output_edges']:raise ValueError('field count mismatch')
 if not np.array_equal(vertices[:,0],np.arange(len(p))) or not np.array_equal(vertices[:,1:4].astype(np.float32).astype(float),p):raise ValueError('PLY coordinate identity mismatch')
 if not np.isfinite(vertices).all() or not np.isfinite(edges).all():raise ValueError('nonfinite field')
 ids=edges[:,:2].astype(np.int64)
 if not ((ids[:,0]<ids[:,1]) & (ids[:,0]>=0) & (ids[:,1]<len(p))).all():raise ValueError('invalid output edge identities')
 actual=np.sort(f[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1).astype(np.uint64)
 actual_keys=np.unique((actual[:,0]<<32)|actual[:,1])
 field_keys=(ids[:,0].astype(np.uint64)<<32)|ids[:,1].astype(np.uint64)
 if not np.array_equal(np.sort(field_keys),actual_keys):raise ValueError('output edge coverage mismatch')
 targets=vertices[:,4].astype(np.float32)
 stored=np.float32(.5)*(targets[ids[:,0]]+targets[ids[:,1]])
 if not np.allclose(edges[stored>0,4],stored[stored>0],rtol=0,atol=1e-7):raise ValueError('native target mismatch')
 q=p.astype(np.float32);delta=q[ids[:,0]]-q[ids[:,1]]
 ratios=np.sqrt(np.sum(delta*delta,axis=1,dtype=np.float32))/edges[:,4].astype(np.float32)
 if not np.allclose(ratios,edges[:,5],rtol=2e-6,atol=1e-7):raise ValueError('native ratio mismatch')
 ratios=edges[:,5].astype(np.float32)
 long=ratios>np.float32(metadata['split_ratio'])*np.float32(1.0001)
 short=ratios<np.float32(metadata['collapse_ratio'])
 constrained=(edges[:,6]>0)|((edges[:,2].astype(np.uint8)&9)!=0)
 edge_targets=targets[ids]
 growth=np.maximum(edge_targets[:,0],edge_targets[:,1])/np.maximum(np.minimum(edge_targets[:,0],edge_targets[:,1]),1e-30)
 face_targets=targets[f]
 transitioning=(face_targets.max(1)>face_targets.min(1)*1.05)
 tri=p[f]
 sides=np.linalg.norm(tri-np.roll(tri,-1,axis=1),axis=2)
 area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
 shape=np.divide(2*np.sqrt(3)*area2,np.sum(sides*sides,axis=1),out=np.zeros(len(f)),where=np.sum(sides*sides,axis=1)>0)
 report=json.loads(Path(prefix+'.json').read_text())['output_region_quality']
 if int(long.sum())!=report['long_edges'] or int(short.sum())!=report['short_edges']:raise ValueError('native size statistics mismatch')
 _,sp,records,patches,features=read_snapshot(snapshot_path)
 source_patch_summary=None
 if metadata.get('version',1)>=2:
  if patches is None or metadata.get('source_patches')!=len(patches):raise ValueError('source patch count mismatch')
  lines=Path(prefix+'.source_patches.tsv').read_text(encoding='utf-8').splitlines()
  if len(lines)!=len(patches)+1 or lines[0]!='source_patch\tpatch_type\tproducer_feature_role\tsupport_patch_ids':
   raise ValueError('source patch evidence format mismatch')
  fillets=0;support_relations=0;support_sets=[];roles=[]
  for i,(line,(header,tail)) in enumerate(zip(lines[1:],patches)):
   columns=line.split('\t')
   if len(columns)!=4:raise ValueError('source patch evidence columns mismatch')
   patch_id,patch_type,role=map(int,columns[:3])
   supports=[int(value) for value in columns[3].split(',')] if columns[3] else []
   expected=np.frombuffer(tail[:header[4]*4],dtype='<u4').tolist()
   if patch_id!=i or patch_type!=header[0] or role!=header[2] or supports!=expected:
    raise ValueError('source patch evidence mismatch')
   fillets+=int(role==1);support_relations+=len(supports)
   roles.append(role);support_sets.append(set(supports))
  source_patch_summary=dict(patches=len(patches),producer_fillet_patches=fillets,support_relations=support_relations,
                            cad_feature_certification=False)
 ee=np.sort(records[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1).astype(np.uint64)
 keys=(ee[:,0]<<32)|ee[:,1];order=np.argsort(keys,kind='stable');sorted_keys=keys[order]
 unique,starts,counts=np.unique(sorted_keys,return_index=True,return_counts=True)
 owners=np.repeat(records[:,3],3)[order]
 lower=np.minimum.reduceat(owners,starts);upper=np.maximum.reduceat(owners,starts)
 evidence=np.zeros(len(unique),np.uint8)
 evidence[counts==1]|=4;evidence[lower!=upper]|=8
 fk=(features[:,1:3].min(1).astype(np.uint64)<<32)|features[:,1:3].max(1).astype(np.uint64)
 where=np.searchsorted(unique,fk)
 if not np.array_equal(unique[where],fk):raise ValueError('source record is not an edge')
 evidence[where]|=1;evidence[where[features[:,3]>0]]|=2
 expected=evidence>0;sk=(source[:,0]<<32)|source[:,1];sort=np.argsort(sk)
 if not np.array_equal(sk[sort],unique[expected]) or not np.array_equal(source[sort,3],evidence[expected]):raise ValueError('source evidence mismatch')
 if source_patch_summary is not None:
  fillet_support=0;support_hard=0
  for row in source:
   a,b=int(row[4]),int(row[5])
   if a>=len(patches) or b>=len(patches):continue
   matched=(roles[a]==1 and b in support_sets[a]) or (roles[b]==1 and a in support_sets[b])
   if matched:
    fillet_support+=1;support_hard+=int(bool(int(row[3])&2))
  source_patch_summary['fillet_support_interface_edges']=fillet_support
  source_patch_summary['fillet_support_hard_edges']=support_hard
 feature_ids=dict(zip(fk.tolist(),features[:,0].tolist()))
 if any(int(row[2])!=feature_ids.get(int(key),0) for key,row in zip(sk,source)):raise ValueError('source feature identity mismatch')
 if metadata['source_vertices']!=len(sp) or metadata['source_faces']!=len(records):raise ValueError('source domain mismatch')
 return dict(verified=True,mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),source_evidence_counts=np.bincount(evidence[expected],minlength=16).tolist(),source_patch_evidence=source_patch_summary,native_long_edges=int(long.sum()),native_short_edges=int(short.sum()),native_short_constrained_edges=int((short&constrained).sum()),target_min=float(targets.min()),target_p05=float(np.quantile(targets,.05)),target_max=float(targets.max()),adjacent_target_growth_p99=float(np.quantile(growth,.99)),adjacent_target_growth_max=float(growth.max()),adjacent_target_growth_over_1_3=int((growth>1.3).sum()),transition_faces=int(transitioning.sum()),transition_skinny_faces=int((transitioning&(shape<.2)).sum()),transition_skinny_area=float(.5*area2[transitioning&(shape<.2)].sum()),transition_quality_p05=float(np.quantile(shape[transitioning],.05)) if transitioning.any() else None,seconds=time.perf_counter()-start,geometry_and_flip_certification=False)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mesh',type=Path,required=True);parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 result=verify(args.mesh,args.snapshot);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
