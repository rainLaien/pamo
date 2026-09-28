"""Isolated strip size experiments. Outputs are NOT assembled production meshes.

Boundary resampling requires neighbor coordination before any integration.
The same Euclidean quality cutoff and original geometry budget apply to all rows.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from audit_phase1 import quality_regions, chain_audit, area_samples, distance_samples
from repair_narrow_strips import read_snapshot, audit
from narrow_strip_geometry import remesh, sample_distances


def interface_scope(p, records, selected_patches, candidates, tolerance):
    """Compare selected interface stations; enumerate neighbors needing splitting."""
    all_edges = records[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    keys = (all_edges.min(1).astype(np.uint64)<<32) | all_edges.max(1).astype(np.uint64)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    owners = np.repeat(records[:, 3], 3)[order]
    rows = []
    seen = set()
    for rid in selected_patches:
        f = records[records[:, 3] == rid, :3]
        ee = f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        local_keys, counts = np.unique((ee.min(1).astype(np.uint64)<<32) | ee.max(1).astype(np.uint64), return_counts=True)
        for key in local_keys[counts == 1]:
            if int(key) in seen: continue
            seen.add(int(key))
            neighbor = np.unique(owners[np.searchsorted(sorted_keys, key, 'left'):np.searchsorted(sorted_keys, key, 'right')]).tolist()
            a, b = int(key)>>32, int(key)&0xffffffff
            origin, delta = p[a], p[b]-p[a]
            stations = {}
            for owner in neighbor:
                if owner not in candidates: continue
                points = candidates[owner]
                t = ((points-origin)@delta)/(delta@delta)
                mask = (t >= -1e-6) & (t <= 1+1e-6) & (np.linalg.norm(points-origin-t[:, None]*delta, axis=1) <= tolerance)
                stations[owner] = np.unique(points[mask], axis=0)
            values = list(stations.values())
            row = dict(source_edge=[a,b], owners=neighbor, candidate_station_counts={str(k):len(v) for k,v in stations.items()},
                       unprocessed_neighbors=[k for k in neighbor if k not in selected_patches],
                       selected_station_sets_match=all(np.array_equal(values[0], v) for v in values[1:]))
            rows.append(row)
    return dict(edges=rows, mismatched_selected_interfaces=sum(not x['selected_station_sets_match'] for x in rows),
                unprocessed_neighbor_patch_ids=sorted({k for x in rows for k in x['unprocessed_neighbors']}),
                station_match_is_not_assembled_seam_validation=True)


def evaluate_candidate(p, f, target, threshold, budget, chain_tolerance=None, shared_grid=None, protected_edges=None):
    started = time.perf_counter()
    cp, cf, groups = remesh(p, f, target, shared_grid, min(target*1e-6, budget))
    # Production snapshots pass through float32 coordinates; audit that precision.
    cp = cp.astype(np.float32).astype(np.float64)
    metrics, _, area, edges, _, counts = quality_regions(cp, cf, np.zeros(len(cf), int), threshold, target)
    _, _, _, source_edges, _, source_counts = quality_regions(p, f, np.zeros(len(f), int), threshold, target)
    boundary = source_edges[source_counts == 1]
    tolerance = chain_tolerance if chain_tolerance is not None else max(1e-8, np.ptp(p, axis=0).max()*1e-7)
    chains = chain_audit(p, boundary, cp, edges, tolerance)
    source = trimesh.Trimesh(p, f, process=False)
    candidate = trimesh.Trimesh(cp, cf, process=False)
    # Fold lines inside a patch are constraints too, even without CAD provenance.
    adjacency = source.face_adjacency
    folded = np.einsum('ij,ij->i', source.face_normals[adjacency[:, 0]], source.face_normals[adjacency[:, 1]]) < 1-1e-10
    creases = chain_audit(p, source.face_adjacency_edges[folded], cp, edges, chains['tolerance'])
    protected = chain_audit(p, np.empty((0, 2), int) if protected_edges is None else protected_edges, cp, edges, chains['tolerance'])
    forward = sample_distances(np.concatenate([area_samples(candidate, 128), candidate.triangles_center]), p, f)
    # Extended-precision source distances avoid sliver query noise in forward audit.
    reverse, _ = distance_samples(candidate, np.concatenate([p, source.triangles_center, area_samples(source, 256)]))
    topology = audit(cp, np.column_stack([cf, np.zeros(len(cf), int)]), np.zeros((0, 4), np.uint32))[0]
    source_topology = audit(p, np.column_stack([f, np.zeros(len(f), int)]), np.zeros((0, 4), np.uint32))[0]
    _, normals = distance_samples(source, candidate.triangles_center)
    dots = np.einsum('ij,ij->i', candidate.face_normals, normals)
    relative_area = abs(area.sum()/source.area - 1)
    checks = dict(boundary=chains, interior_fold_chains=creases, protected_feature_chains=protected, topology=topology, groups=groups,
                  forward_sample_max=float(forward.max(initial=0)), reverse_sample_max=float(reverse.max(initial=0)),
                  nonpositive_reference_normal_dots=int((dots <= 0).sum()), relative_area_change=float(relative_area),
                  zero_area_faces=int((area <= 0).sum()), budget=budget,
                  geometry_samples_are_not_a_hausdorff_certificate=True,
                  boundary_subdivision_requires_neighbor_coordination=True)
    checks['local_checks_pass'] = bool(chains['missing_chains'] == 0 and chains['missing_anchors'] == 0
                                      and creases['missing_chains'] == 0 and creases['missing_anchors'] == 0
                                      and protected['missing_chains'] == 0 and protected['missing_anchors'] == 0
                                      and area.min() > 0 and relative_area < 1e-5
                                      and (dots > 0).all() and max(forward.max(initial=0), reverse.max(initial=0)) <= budget
                                      and all(topology[k] == 0 for k in ['nonmanifold_edges', 'inconsistent_interior_edges', 'float32_zero_faces', 'duplicate_faces'])
                                      and all(topology[k] == source_topology[k] for k in ['connected_components', 'euler_characteristic']))
    return cp, cf, dict(metrics=metrics, checks=checks, seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--reference-ply', type=Path, required=True)
    parser.add_argument('--patches', type=int, nargs='+', default=[3842, 3843, 3816])
    parser.add_argument('--targets', type=float, nargs='+', default=[7.623706340789795, 1.9, .95, .475, .2375])
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--max-error', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shared-grid', action='store_true', help='Use existing repair grid phase rule on the selected patch group')
    args = parser.parse_args()
    if not all(np.isfinite(x) and x > 0 for x in [*args.targets, args.threshold, args.max_error]):
        parser.error('sizes, cutoff and budget must be finite and positive')
    args.output.mkdir(parents=True, exist_ok=True)
    from audit_phase1 import read_ply
    rp, rf, labels = read_ply(args.reference_ply)
    _, p, records, _, features = read_snapshot(args.snapshot)
    # Identical to phase-one global chain audit, held fixed for all candidates.
    chain_tolerance = max(1e-6, np.linalg.norm(np.ptp(p, axis=0))*1e-7)
    report = dict(experiment='parameter-only local size sweep using existing reconstruction',
                  production_integrated=False, full_pipeline_timing=False,
                  snapshot=str(args.snapshot.resolve()), reference_ply=str(args.reference_ply.resolve()),
                  chain_tolerance=chain_tolerance, shared_grid=args.shared_grid, rows=[])
    candidates = {target: {} for target in args.targets}
    grids = {}
    if args.shared_grid:
        selected = np.unique(records[np.isin(records[:, 3], args.patches), :3])
        for target in args.targets:
            step = .95*target
            grid = {}
            for axis in range(3):
                residues = np.unique(np.remainder(p[selected, axis], step))
                ends = np.r_[residues[1:], residues[0]+step]
                best = int(np.argmax(ends-residues))
                phase = ((residues[best]+ends[best])/2) % step
                lo, hi = p[:, axis].min(), p[:, axis].max()
                grid[axis] = phase+step*np.arange(np.floor((lo-phase)/step), np.ceil((hi-phase)/step)+1)
            grids[target] = grid
    for rid in args.patches:
        faces = records[records[:, 3] == rid, :3]
        if not len(faces):
            raise ValueError(f'empty or invalid patch {rid}')
        ids, inverse = np.unique(faces, return_inverse=True)
        pp, ff = p[ids], inverse.reshape(-1, 3)
        feature_mask = np.isin(features[:, 1], ids) & np.isin(features[:, 2], ids)
        local_features = np.searchsorted(ids, features[feature_mask, 1:3])
        baseline = quality_regions(rp, rf[labels == rid], np.full((labels == rid).sum(), rid), args.threshold, args.targets[0])[0]
        # Baseline vertex count is global; replace it with the local count.
        baseline['vertices'] = len(np.unique(rf[labels == rid]))
        for target in args.targets:
            row = dict(patch=rid, target=target, reference_production_metrics=baseline)
            try:
                cp, cf, result = evaluate_candidate(pp, ff, target, args.threshold, args.max_error, chain_tolerance, grids.get(target), local_features)
                row.update(result)
                candidates[target][rid] = cp
                row['metrics_at_original_global_target'] = quality_regions(cp, cf, np.full(len(cf), rid), args.threshold, args.targets[0])[0]
                m = row['metrics']
                row['quality_nonregression_vs_reference'] = bool(all(m[k] >= baseline[k]-1e-6 for k in ['quality_mean', 'quality_p05', 'area_weighted_mean'])
                    and all(m[k] <= baseline[k]+1e-6 for k in ['low_quality_area', 'largest_low_quality_area']))
                np.savez_compressed(args.output/f'patch_{rid}_h_{target:.6g}.npz', vertices=cp, faces=cf)
            except (ValueError, RuntimeError) as exc:
                row['rejected'] = str(exc)
            report['rows'].append(row)
            (args.output/'sweep.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
    report['interfaces_by_target'] = {str(target):interface_scope(p, records, args.patches, candidates[target], chain_tolerance) for target in args.targets}
    (args.output/'sweep.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
