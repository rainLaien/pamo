"""Joint-region quality tradeoffs, separate from geometry/topology certification.
Budgets are explicit project policy. Missing budgets never imply acceptance.
Mean and P05 are reported, not individual patch vetoes.
"""
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from audit_phase1 import quality_regions

BUDGETS=('neighbor_loss_to_gain_max','area_weighted_quality_loss_max',
         'largest_component_growth_max','neighbor_component_area_max',
         'severe_area_growth_max','face_growth_max')

def validate_policy(policy):
 if policy is None:return None
 if set(policy)!=set(BUDGETS):raise ValueError('joint policy requires exactly: '+', '.join(BUDGETS))
 for key,value in policy.items():
  if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('invalid joint budget: '+key)
 return dict(policy)

def summary(points,faces,labels,threshold,target):
 if not len(faces):raise ValueError('empty joint region')
 if not (math.isfinite(threshold) and 0<threshold<=1 and math.isfinite(target) and target>0):raise ValueError('invalid joint metric parameters')
 metrics,q,area,_,adjacency,_=quality_regions(points,faces,labels,threshold,target)
 if not np.isfinite(q).all() or not np.isfinite(area).all() or not area.sum()>0:raise ValueError('invalid joint region geometry')
 metrics['surface_area']=float(area.sum())
 metrics['defect_severity']=float(np.dot(area,np.maximum(0,1-q/threshold)))
 # An additional diagnostic tail, not a universal acceptable-quality threshold.
 metrics['severe_diagnostic_threshold']=threshold/4
 metrics['severe_area']=float(area[q<threshold/4].sum())
 a,b=adjacency
 graph=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(faces),len(faces))).tocsr()
 metrics['spatial_components']=int(connected_components(graph,directed=False,return_labels=False))
 return metrics

def decide(before,after,patch_rows,policy=None):
 policy=validate_policy(policy)
 keys=('defect_severity','area_weighted_mean','largest_low_quality_area','severe_area','faces')
 for m in (before,after):
  if any(not math.isfinite(m[k]) or m[k]<0 for k in keys):raise ValueError('invalid joint metrics')
 gains=sum(max(0,r['before']['defect_severity']-r['after']['defect_severity']) for r in patch_rows)
 losses=sum(max(0,r['after']['defect_severity']-r['before']['defect_severity']) for r in patch_rows)
 requirements=dict(neighbor_loss_to_gain_max=losses/gains if gains else None,
  area_weighted_quality_loss_max=max(0,before['area_weighted_mean']-after['area_weighted_mean']),
  largest_component_growth_max=max(0,after['largest_low_quality_area']-before['largest_low_quality_area']),
  neighbor_component_area_max=max((r['after']['largest_low_quality_area'] for r in patch_rows if r['after']['defect_severity']>r['before']['defect_severity']+1e-12),default=0),
  severe_area_growth_max=max(0,after['severe_area']-before['severe_area']),
  face_growth_max=max(0,after['faces']-before['faces']))
 reasons=[]
 if after['defect_severity']>=before['defect_severity']-max(1e-12,before['defect_severity']*1e-6):reasons.append('no_joint_severity_progress')
 if policy is not None:
  for key,required in requirements.items():
   if required is None or required>policy[key]+1e-12:reasons.append('budget_exceeded:'+key)
 elif not reasons:reasons.append('explicit_tradeoff_policy_required')
 return dict(quality_tradeoff_pass=not reasons,reasons=reasons,policy=policy,
  required_budgets=requirements,gross_severity_gain=gains,gross_severity_loss=losses,
  net_severity_gain=before['defect_severity']-after['defect_severity'],
  mean_and_p05_are_advisory=True,geometry_topology_feature_and_size_checks_required=True)

def groups(faces,labels,scope):
 # Disconnected transactions cannot pay for one another's losses.
 scope=set(map(int,scope));parent={x:x for x in scope}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 edges={}
 for face,label in zip(faces,labels):
  label=int(label)
  if label not in scope:continue
  for a,b in zip(face,np.roll(face,-1)):
   key=tuple(sorted((int(a),int(b))))
   if key in edges:
    other=edges[key];x,y=find(label),find(other)
    if x!=y:parent[max(x,y)]=min(x,y)
   else:edges[key]=label
 result={}
 for x in sorted(scope):result.setdefault(find(x),[]).append(x)
 return list(result.values())

def evaluate(bp,bf,bl,cp,cf,cl,scope,threshold,target,policy=None):
 result=[]
 for group in groups(bf,bl,scope):
  before=summary(bp,bf[np.isin(bl,group)],bl[np.isin(bl,group)],threshold,target)
  after=summary(cp,cf[np.isin(cl,group)],cl[np.isin(cl,group)],threshold,target)
  rows=[]
  for rid in group:
   bm=summary(bp,bf[bl==rid],bl[bl==rid],threshold,target);am=summary(cp,cf[cl==rid],cl[cl==rid],threshold,target)
   rows.append(dict(patch=rid,before=bm,after=am))
  result.append(dict(patches=group,before=before,after=after,patches_detail=rows,decision=decide(before,after,rows,policy)))
  if before['spatial_components']!=1 or after['spatial_components']!=1:
   result[-1]['decision']['quality_tradeoff_pass']=False
   result[-1]['decision']['reasons'].append('connected_component_scope_required')
 return dict(groups=result,quality_tradeoff_pass=bool(result) and all(r['decision']['quality_tradeoff_pass'] for r in result),
             size_metric_basis='global_h_diagnostic_not_native_field',endpoint_accepted=False)
