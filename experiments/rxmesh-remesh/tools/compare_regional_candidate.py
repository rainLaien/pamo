"""Compare a synchronized regional candidate with its immutable incumbent.

Valid failed candidates are retained; local progress does not certify the endpoint.
"""
import argparse
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from audit_phase1 import read_ply, quality_regions, chain_audit, area_samples, distance_samples
from repair_narrow_strips import read_snapshot, audit
from cached_proximity import compact_mesh
from joint_region_quality import evaluate as evaluate_joint, validate_policy


def joint_transaction_decision(quality_ok, hard_ok, geometry_ok, size_ok, native_endpoint_ok):
    adoptable=bool(quality_ok and hard_ok and geometry_ok and size_ok)
    return adoptable,bool(adoptable and native_endpoint_ok)


def native_size_guard(evidence_path,baseline,candidate,budget):
    if evidence_path is None:return dict(passed=False,reason='native_size_evidence_required')
    evidence=json.loads(evidence_path.read_text(encoding='utf-8'))
    for name,path in [('baseline',baseline),('candidate',candidate)]:
        row=evidence[name]
        if row.get('verified') is not True or row.get('mesh_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('native size evidence is not bound to '+name)
    before,after=evidence['baseline'],evidence['candidate']
    for row in (before,after):
        for key in ('native_short_edges','native_long_edges'):
            value=row[key]
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ValueError('invalid native size count')
    growth=max(0,after['native_short_edges']-before['native_short_edges'])
    passed=bool(after['native_long_edges']<=before['native_long_edges'] and growth<=budget)
    return dict(passed=passed,before=before,after=after,short_edge_growth=growth,
                short_edge_growth_budget=budget,reason='passed' if passed else 'native_size_budget_exceeded')


def nonregression(before, after):
    reasons=[]
    for k in ['quality_mean','quality_p05','area_weighted_mean']:
        if after[k]<before[k]-1e-6:reasons.append(k)
    for k in ['low_quality_area','largest_low_quality_area']:
        if after[k]>before[k]+max(1e-12,before[k]*1e-6):reasons.append(k)
    return reasons


def seam_edges(faces, labels, adjacency):
    a,b=adjacency
    keys={tuple(sorted(set(faces[x]) & set(faces[y]))) for x,y in zip(a[labels[a]!=labels[b]],b[labels[a]!=labels[b]])}
    return np.asarray(sorted(keys),int).reshape(-1,2)


def canonical_triangles(points, faces):
    tri=points[faces]
    if not len(tri):return np.empty((0,9))
    first=np.lexsort((tri[:,:,2],tri[:,:,1],tri[:,:,0]),axis=1)[:,0]
    rotation=(first[:,None]+np.arange(3))%3
    flat=np.take_along_axis(tri,rotation[:,:,None],axis=1).reshape(-1,9)
    return flat[np.lexsort(flat.T[::-1])]


def candidate_decision(regional_safe, global_reasons, region_geometry_ok, geometry_ok, native_ok):
    # A provisional transaction can make progress while unrelated endpoint
    # defects remain. It must not be reported as an accepted final remesh.
    adoptable=bool(regional_safe and not global_reasons and region_geometry_ok)
    return adoptable, bool(adoptable and geometry_ok and native_ok)


def changed_patches(bp,bf,bi,cp,cf,ci):
    def groups(f,ids):
        order=np.argsort(ids,kind='stable');keys,counts=np.unique(ids,return_counts=True)
        offsets=np.r_[0,np.cumsum(counts)]
        return {int(k):f[order[offsets[i]:offsets[i+1]]] for i,k in enumerate(keys)}
    before,after=groups(bf,bi),groups(cf,ci)
    changed=[]
    for rid in sorted(set(before)|set(after)):
        if rid not in before or rid not in after or not np.array_equal(canonical_triangles(bp,before[rid]),canonical_triangles(cp,after[rid])):
            changed.append(rid)
    return changed


def compare(args):
    started=time.perf_counter()
    native=json.loads(Path(str(args.candidate)+'.json').read_text(encoding='utf-8-sig'))
    reference_native=json.loads(Path(str(args.baseline)+'.json').read_text(encoding='utf-8-sig'))
    for key in ['effective_target_length','max_geometry_error','max_iterations','smooth_passes','collapse_passes','flip_passes','strict_flip_quality','feature_refine']:
        if native[key]!=reference_native[key]:raise ValueError(f'algorithm comparison uses different {key}')
    repair=json.loads(args.repair_report.read_text(encoding='utf-8'))
    affected=repair.get('regional_affected_patch_ids',[])
    requested=set(map(int,repair.get('regional_patch_targets',{})))
    rebuilt=set(map(int,repair.get('rebuilt_patches',{})))
    _,p,records,patches,features=read_snapshot(args.snapshot)
    p=p.astype(np.float32).astype(float)
    baseline_p,baseline_f,baseline_ids=read_ply(args.baseline)
    candidate_p,candidate_f,candidate_ids=read_ply(args.candidate)
    actual_changes=changed_patches(baseline_p,baseline_f,baseline_ids,candidate_p,candidate_f,candidate_ids)
    outside=sorted(set(actual_changes)-set(affected))
    # GPU task grouping can affect more than geometric one-ring neighbors.
    # Audit actual changes too; do not silently claim the one-ring is sufficient.
    affected=sorted(set(affected)|set(actual_changes))
    joint_policy=None;size_evidence=None
    if getattr(args,'joint_quality_policy',None):
        joint_policy=json.loads(args.joint_quality_policy.read_text(encoding='utf-8'))
        if set(joint_policy)!={'quality_limits','native_short_edge_growth_max','justification'}:
            raise ValueError('joint policy requires quality_limits, native_short_edge_growth_max, justification')
        validate_policy(joint_policy['quality_limits'])
        budget=joint_policy['native_short_edge_growth_max']
        if isinstance(budget,bool) or not isinstance(budget,int) or budget<0 or not isinstance(joint_policy['justification'],str) or not joint_policy['justification'].strip():
            raise ValueError('joint size budget and policy justification must be explicit')
        size_evidence=native_size_guard(getattr(args,'native_size_evidence',None),args.baseline,args.candidate,budget)
    joint=evaluate_joint(baseline_p,baseline_f,baseline_ids,candidate_p,candidate_f,candidate_ids,
                         affected,args.threshold,args.target,joint_policy['quality_limits'] if joint_policy else None)
    print(f'regional comparison changed_patches={len(actual_changes)} outside_declared_neighborhood={len(outside)}',flush=True)
    before,bq,_,be,ba,bc=quality_regions(baseline_p,baseline_f,baseline_ids,args.threshold,args.target)
    after,cq,_,ce,ca,cc=quality_regions(candidate_p,candidate_f,candidate_ids,args.threshold,args.target)
    source_metric,_,_,se,sa,sc=quality_regions(p,records[:,:3],records[:,3],args.threshold,args.target)
    rows=[]
    for rid in affected:
        bm=quality_regions(baseline_p,baseline_f[baseline_ids==rid],np.full((baseline_ids==rid).sum(),rid),args.threshold,args.target)[0]
        cm=quality_regions(candidate_p,candidate_f[candidate_ids==rid],np.full((candidate_ids==rid).sum(),rid),args.threshold,args.target)[0]
        reasons=nonregression(bm,cm)
        progress=any(cm[k]>bm[k]+1e-6 for k in ['quality_mean','quality_p05','area_weighted_mean']) or any(cm[k]<bm[k]-max(1e-12,bm[k]*1e-6) for k in ['low_quality_area','largest_low_quality_area'])
        if rid in requested and bm['low_quality_faces']>0 and not progress:reasons.append('input_defect_without_progress')
        rows.append(dict(patch=rid,before=bm,after=cm,regressions=reasons,progress=progress))
    tol=max(1e-6,np.linalg.norm(np.ptp(p,axis=0))*1e-7)
    topology=audit(candidate_p,np.column_stack([candidate_f,candidate_ids]),np.empty((0,4),np.uint32))[0]
    source_topology=audit(p,records,features)[0]
    chains=dict(features=chain_audit(p,features[:,1:3],candidate_p,ce,tol),
                open_boundary=chain_audit(p,se[sc==1],candidate_p,ce[cc==1],tol),
                partition_seams=chain_audit(p,seam_edges(records[:,:3],records[:,3],sa),candidate_p,seam_edges(candidate_f,candidate_ids,ca),tol))
    mesh=trimesh.Trimesh(candidate_p,candidate_f,process=False)
    source=trimesh.load(args.source,force='mesh',process=False)
    sp,inv=np.unique(source.vertices,axis=0,return_inverse=True)
    source=trimesh.Trimesh(sp,inv[source.faces],process=False)
    low_ids=np.flatnonzero(cq<args.threshold)
    if len(low_ids)>2000:low_ids=low_ids[np.linspace(0,len(low_ids)-1,2000,dtype=int)]
    forward,_=distance_samples(source,np.concatenate([area_samples(mesh),mesh.triangles_center[low_ids]]))
    backward,_=distance_samples(mesh,area_samples(source))
    geometry=dict(output_to_original_max=float(forward.max()),original_to_output_max=float(backward.max()),
                  budget=args.max_error,within_budget=bool(max(forward.max(),backward.max())<=args.max_error),proof=False)
    orientation=[]
    for rid in affected:
        ref=compact_mesh(p,records[records[:,3]==rid,:3])
        row=dict(patch=rid)
        for name,pp,ff,ids in [('before',baseline_p,baseline_f,baseline_ids),('after',candidate_p,candidate_f,candidate_ids)]:
            local=compact_mesh(pp,ff[ids==rid])
            d,n=distance_samples(ref,local.triangles_center)
            dots=np.einsum('ij,ij->i',local.face_normals,n)
            row[name]=dict(nonpositive=int((dots<=0).sum()),area=float(local.area_faces[dots<=0].sum()),centroid_distance_max=float(d.max(initial=0)))
        row['regressed']=bool(row['after']['nonpositive']>row['before']['nonpositive'] or row['after']['area']>row['before']['area']+1e-6)
        orientation.append(row)
    hard=bool(all(topology[k]==0 for k in ['nonmanifold_edges','inconsistent_interior_edges','float32_zero_faces','duplicate_faces'])
              and all(topology[k]==source_topology[k] for k in ['connected_components','euler_characteristic'])
              and all(v['missing_chains']==0 and v['missing_anchors']==0 for v in chains.values()))
    regional_safe=bool(affected and requested<=rebuilt and hard and not any(x['regressions'] for x in rows) and not any(x['regressed'] for x in orientation))
    global_reasons=nonregression(before,after)
    adoptable,accepted=candidate_decision(regional_safe,global_reasons,
        all(x['after']['centroid_distance_max']<=args.max_error for x in orientation),geometry['within_budget'],native['quality_accepted'])
    if joint_policy is not None:
        regional_safe=bool(affected and requested<=rebuilt and hard and joint['quality_tradeoff_pass']
                           and not any(x['regressed'] for x in orientation))
        adoptable,accepted=joint_transaction_decision(regional_safe,hard,
            geometry['within_budget'] and all(x['after']['centroid_distance_max']<=args.max_error for x in orientation),
            size_evidence['passed'],native['quality_accepted'])
    return dict(baseline=str(args.baseline.resolve()),candidate=str(args.candidate.resolve()),
                regional_targets=repair.get('regional_patch_targets'),affected_patch_ids=affected,
                actual_changed_patch_ids=actual_changes,changes_outside_declared_neighborhood=outside,
                parameter_changes={'regional_reconstruction_targets':repair.get('regional_patch_targets')},
                algorithm_options={'graded_neighbors':repair.get('graded_neighbors',False),
                                   'explore_provisional_children':native.get('explore_provisional_children',False),
                                   'track_region_candidates':native.get('track_region_candidates',False),
                                   'size_feasible_final_refine':native.get('size_feasible_final_refine',False),
                                   'select_final_regions':native.get('select_final_regions',False),
                                   'child_candidate_policy':native.get('child_candidate_policy'),
                                   'immutable_local_sizing':native.get('immutable_local_sizing',False),
                                   'sizing_gradation':native.get('sizing_gradation'),
                                   'gpu_patch_targets':native.get('patch_targets',{})},
                baseline_metrics=before,candidate_metrics=after,global_regressions=global_reasons,
                patch_comparisons=rows,topology=topology,chains=chains,geometry_samples=geometry,
                joint_region_quality=joint,joint_policy=joint_policy,native_size_evidence=size_evidence,
                acceptance_mode='joint_budgeted_transaction' if joint_policy else 'legacy_strict_transaction',
                affected_orientation=orientation,hard_checks_pass=hard,regional_candidate_accepted=regional_safe,
                regional_transaction_adoptable=adoptable,
                native_quality_accepted=native['quality_accepted'],endpoint_accepted=accepted,
                stop_reason='accepted' if accepted else 'retained_provisional_candidate',seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','snapshot','repair-report','baseline','candidate','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ['threshold','target','max-error']:parser.add_argument('--'+name,type=float,required=True)
    parser.add_argument('--joint-quality-policy',type=Path,help='Explicit joint loss budgets and justification; omitting retains strict legacy transaction policy')
    parser.add_argument('--native-size-evidence',type=Path,help='JSON baseline/candidate reports from audit_native_fields, each bound to its mesh SHA256')
    args=parser.parse_args()
    if args.native_size_evidence and not args.joint_quality_policy:
        parser.error('--native-size-evidence requires --joint-quality-policy')
    report=compare(args)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['regional_candidate_accepted','endpoint_accepted','global_regressions','geometry_samples','seconds']}),flush=True)


if __name__=='__main__':main()
