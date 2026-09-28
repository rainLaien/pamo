"""Try one bounded source-reference region recovery without replacing the incumbent.

The output PLY is a provisional candidate. The report states whether the
local, joint, global, native-size, topology, chain, orientation and sampled
original-STL checks passed. A caller must retain the incumbent on rejection.
"""
import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import trimesh

from audit_phase1 import (area_samples, chain_audit, distance_samples,
                          quality_regions, read_ply)
from compare_regional_candidate import changed_patches, seam_edges
from cached_proximity import CachedSurfaceQuery
from extract_partition_blocks import extract
from join_matching_patch import boundary, join
from joint_region_quality import BUDGETS, evaluate as evaluate_joint
from repair_narrow_strips import audit, read_snapshot


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_ply(path, points, faces, labels):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\n")
        stream.write("property uint patch_id\nend_header\n")
        for x, y, z in points:
            stream.write(f"{np.float32(x):.9g} {np.float32(y):.9g} {np.float32(z):.9g}\n")
        for (a, b, c), patch in zip(faces, labels):
            stream.write(f"3 {a} {b} {c} {patch}\n")


def native_targets(incumbent, fields_mesh, local_candidate, joined, selected, changed_patch):
    if sha256(incumbent) != sha256(fields_mesh):
        raise ValueError("incumbent and native-field mesh differ")
    bp, bf, bl = read_ply(incumbent)
    cp, cf, local_labels = read_ply(local_candidate)
    cl = np.asarray(selected, dtype=np.uint32)[local_labels]
    jp, jf, _ = read_ply(joined)
    baseline_rows = np.loadtxt(str(fields_mesh) + ".vertices.tsv", skiprows=1, ndmin=2)
    candidate_rows = np.loadtxt(str(local_candidate) + ".vertices.tsv", skiprows=1, ndmin=2)
    if not np.array_equal(baseline_rows[:, 1:4].astype(np.float32).astype(float), bp):
        raise ValueError("baseline field coordinates differ")
    if not np.array_equal(candidate_rows[:, 1:4].astype(np.float32).astype(float), cp):
        raise ValueError("candidate field coordinates differ")
    bt = baseline_rows[:, 4].astype(np.float32)
    ct = candidate_rows[:, 4].astype(np.float32)
    baseline_edges = np.loadtxt(str(fields_mesh) + ".edges.tsv", skiprows=1, ndmin=2)
    baseline_meta = json.loads(Path(str(fields_mesh) + ".fields.json").read_text())
    baseline_ratio = baseline_edges[:, 5].astype(np.float32)
    baseline_short = int((baseline_ratio < np.float32(baseline_meta["collapse_ratio"])).sum())
    baseline_long = int((baseline_ratio > np.float32(baseline_meta["split_ratio"]) *
                         np.float32(1 + baseline_meta["long_edge_tolerance"])).sum())
    baseline_quality = json.loads(Path(str(fields_mesh) + ".json").read_text())["output_region_quality"]
    if baseline_short != baseline_quality["short_edges"] or baseline_long != baseline_quality["long_edges"]:
        raise ValueError("baseline native size report differs from sidecar")
    old_boundary, old_edges = boundary(bp, bf[bl == changed_patch])
    new_boundary, new_edges = boundary(cp, cf[cl == changed_patch])
    if old_edges != new_edges or set(old_boundary) != set(new_boundary):
        raise ValueError("candidate boundary graph differs")
    mapping = {new_boundary[k]: old_boundary[k] for k in old_boundary}
    boundary_changes = [[int(old), float(bt[old]), float(ct[new])]
                        for new, old in mapping.items() if bt[old] != ct[new]]
    positions = list(bp.copy())
    targets = list(bt.copy())
    for vertex in np.unique(cf[cl == changed_patch]):
        vertex = int(vertex)
        if vertex not in mapping:
            mapping[vertex] = len(positions)
            positions.append(cp[vertex].copy())
            targets.append(ct[vertex])
    replacement = np.asarray([[mapping[int(v)] for v in face] for face in cf[cl == changed_patch]], dtype=np.int64)
    all_faces = np.vstack((bf[bl != changed_patch], replacement))
    used = np.unique(all_faces)
    if not np.array_equal(np.asarray(positions)[used].astype(np.float32).astype(float), jp) or not np.array_equal(
        np.searchsorted(used, all_faces), jf
    ):
        raise ValueError("native target mapping does not reproduce joined mesh")
    targets = np.asarray(targets, dtype=np.float32)[used]
    edge = np.unique(np.sort(jf[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    x = jp.astype(np.float32)
    lengths = np.sqrt(np.sum((x[edge[:, 0]] - x[edge[:, 1]]) ** 2, axis=1, dtype=np.float32))
    h = np.float32(.5) * (targets[edge[:, 0]] + targets[edge[:, 1]])
    ratio = lengths / h
    return dict(boundary_target_changes=boundary_changes, edges=len(edge),
                baseline_short_edges=baseline_short, baseline_long_edges=baseline_long,
                short_edges=int((ratio < np.float32(baseline_meta["collapse_ratio"])).sum()),
                long_edges=int((ratio > np.float32(baseline_meta["split_ratio"]) *
                                np.float32(1 + baseline_meta["long_edge_tolerance"])).sum()),
                target_min=float(targets.min()), target_max=float(targets.max()),
                mapping_verified=True)


def orientation(source_points, source_records, before, after, patch):
    source_faces = source_records[source_records[:, 3] == patch, :3]
    ids = np.unique(source_faces)
    reference = trimesh.Trimesh(source_points[ids], np.searchsorted(ids, source_faces), process=False)
    result = {}
    for name, (points, faces, labels) in (("before", before), ("after", after)):
        mesh = trimesh.Trimesh(points, faces[labels == patch], process=False)
        d, normals = distance_samples(reference, mesh.triangles_center)
        dots = np.einsum("ij,ij->i", mesh.face_normals, normals)
        bad = dots <= 0
        result[name] = dict(nonpositive=int(bad.sum()), area=float(mesh.area_faces[bad].sum()),
                            centroid_distance_max=float(d.max(initial=0)))
    result["nonregressing"] = (result["after"]["nonpositive"] <= result["before"]["nonpositive"]
                                and result["after"]["area"] <= result["before"]["area"] + 1e-6)
    return result


def chains(source_points, source_records, features, output, source_edges):
    op, of, ol = output
    source_keys, source_counts = source_edges
    se = np.column_stack((source_keys >> 32, source_keys & np.uint64(0xffffffff))).astype(int)
    _, _, _, source_edges, source_adjacency, _ = quality_regions(
        source_points, source_records[:, :3], source_records[:, 3], .026690566912293434, 1.)
    _, _, _, out_edges, out_adjacency, out_counts = quality_regions(
        op, of, ol, .026690566912293434, 1.)
    tol = max(1e-6, np.linalg.norm(np.ptp(source_points, axis=0)) * 1e-7)
    return dict(features=chain_audit(source_points, features[:, 1:3], op, out_edges, tol),
                open_boundary=chain_audit(source_points, se[source_counts == 1], op,
                                          out_edges[out_counts == 1], tol),
                partition_seams=chain_audit(source_points,
                    seam_edges(source_records[:, :3], source_records[:, 3], source_adjacency),
                    op, seam_edges(of, ol, out_adjacency), tol))


def sampled_original_geometry(original_stl, before, after, budget):
    source = trimesh.load(original_stl, force="mesh", process=False)
    points, inverse = np.unique(source.vertices, axis=0, return_inverse=True)
    source = trimesh.Trimesh(points, inverse[source.faces], process=False)
    source_query = CachedSurfaceQuery(source)
    original_samples = area_samples(source, 2000)
    result = {}
    for name, (p, f, _) in (("before", before), ("after", after)):
        mesh = trimesh.Trimesh(p, f, process=False)
        forward, _ = distance_samples(source, area_samples(mesh, 2000), source_query)
        reverse, _ = distance_samples(mesh, original_samples)
        result[name] = dict(output_to_original_max=float(forward.max()),
                            original_to_output_max=float(reverse.max()))
    result["budget"] = budget
    result["candidate_within_budget"] = max(result["after"].values()) <= budget
    result["sampled_not_continuous_proof"] = True
    return result


def run(args):
    started = time.perf_counter()
    policy = json.loads(args.quality_policy.read_text(encoding="utf-8"))
    if set(policy) != {"global_mean_loss_max", "justification"}:
        raise ValueError("quality policy requires a global mean loss budget and justification")
    budget = policy["global_mean_loss_max"]
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise ValueError("invalid global mean loss budget")
    if not isinstance(policy["justification"], str) or not policy["justification"].strip():
        raise ValueError("quality policy justification is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("output already exists")
    selected = sorted(set(args.patches))
    if not selected or len(selected) > 2:
        raise ValueError("this bounded stage supports one or two source patches")
    baseline_report = json.loads(Path(str(args.incumbent) + ".json").read_text(encoding="utf-8-sig"))
    expected = {"max_iterations": args.iterations, "smooth_passes": args.smooth_passes,
                "collapse_passes": args.collapse_passes, "flip_passes": args.flip_passes}
    if any(baseline_report[key] != value for key, value in expected.items()):
        raise ValueError("incumbent remesh parameters differ")
    for key, value in (("effective_target_length", args.target), ("max_geometry_error", args.max_error)):
        if not np.isclose(baseline_report[key], value, rtol=1e-6, atol=1e-6):
            raise ValueError("incumbent sizing or geometry budget differs")
    pipeline_path = args.incumbent.parent / "pipeline.json"
    if pipeline_path.exists():
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8-sig"))
        selected_snapshot = Path(pipeline["selected_snapshot"])
        if not selected_snapshot.exists() or sha256(selected_snapshot) != sha256(args.snapshot):
            raise ValueError("source snapshot differs from incumbent pipeline selection")
    stage = args.output.parent / (args.output.stem + "_stage")
    stage.mkdir(exist_ok=False)
    subset = stage / "source.cadpart"
    extraction = extract(args.snapshot, subset, selected, block_size=1)
    if extraction["original_patch_ids"] != selected:
        raise ValueError("support closure exceeds requested transaction")
    local = stage / "local.ply"
    command = [str(args.remesher), str(subset), str(local), "--workers", "1",
               "--gpu-concurrency", "1", "--patches-per-task", str(len(selected)),
               "--iters", str(args.iterations), "--smooth-passes", str(args.smooth_passes),
               "--collapse-passes", str(args.collapse_passes), "--flip-passes", str(args.flip_passes),
               "--max-error", str(args.max_error), "--target", str(args.target),
               "--low-quality-threshold", str(args.threshold),
               "--select-final-regions", "--global-quality-acceptance", "--audit-fields"]
    with (stage / "local.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode not in (0, 3, 4) or not local.exists():
        raise RuntimeError(f"local remesh failed with status {result.returncode}")
    local_report = json.loads(Path(str(local) + ".json").read_text(encoding="utf-8-sig"))
    for key in ("effective_target_length", "max_geometry_error", "max_iterations",
                "smooth_passes", "collapse_passes", "flip_passes", "strict_flip_quality"):
        if local_report[key] != baseline_report[key]:
            raise ValueError(f"local remesh changed {key}")
    before = read_ply(args.incumbent)
    cp, cf, local_labels = read_ply(local)
    cl = np.asarray(selected, dtype=np.uint32)[local_labels]
    selected_faces = np.isin(before[2], selected)
    changed = changed_patches(before[0], before[1][selected_faces], before[2][selected_faces], cp, cf, cl)
    if len(changed) != 1 or changed[0] not in selected:
        raise ValueError(f"expected one changed patch in requested transaction, got {changed}")
    joined = join(*before, cp, cf, cl, changed[0])
    save_ply(args.output, *joined)
    after = read_ply(args.output)
    if not all(np.array_equal(a, b) for a, b in zip(joined, after)):
        raise ValueError("saved candidate changed coordinates, faces or patch IDs")
    native = native_targets(args.incumbent, args.incumbent_fields_mesh, local, args.output, selected, changed[0])
    source_header, source_points, source_records, _, features = read_snapshot(args.snapshot)
    topology = audit(after[0], np.column_stack((after[1], after[2])).astype(np.uint32),
                     np.empty((0, 4), dtype=np.uint32))[0]
    source_topology, source_keys, source_counts = audit(source_points, source_records, features)
    chain = chains(source_points, source_records, features, after, (source_keys, source_counts))
    direction = orientation(source_points, source_records, before, after, changed[0])
    original = sampled_original_geometry(args.original_stl, before, after, args.max_error)
    joint_limits = {key: 0 for key in BUDGETS}
    joint = evaluate_joint(*before, *after, selected, args.threshold, args.target, joint_limits)
    bm = quality_regions(*before, args.threshold, args.target)[0]
    am = quality_regions(*after, args.threshold, args.target)[0]
    hard = (all(topology[key] == 0 for key in ("nonmanifold_edges", "inconsistent_interior_edges",
                                              "float32_zero_faces", "duplicate_faces"))
            and all(topology[key] == source_topology[key] for key in ("connected_components", "euler_characteristic"))
            and all(value["missing_chains"] == 0 and value["missing_anchors"] == 0 for value in chain.values()))
    mean_loss = max(0, bm["quality_mean"] - am["quality_mean"])
    quality = (joint["quality_tradeoff_pass"] and mean_loss <= budget
               and am["quality_p05"] >= bm["quality_p05"] - 1e-6
               and am["area_weighted_mean"] >= bm["area_weighted_mean"] - 1e-6
               and am["low_quality_area"] <= bm["low_quality_area"] + 1e-6
               and am["largest_low_quality_area"] <= bm["largest_low_quality_area"] + 1e-6)
    size = (not native["boundary_target_changes"] and native["short_edges"] <= native["baseline_short_edges"]
            and native["long_edges"] <= native["baseline_long_edges"])
    accepted = bool(hard and quality and size and direction["nonregressing"] and original["candidate_within_budget"])
    report = dict(accepted=accepted, candidate=str(args.output), incumbent=str(args.incumbent),
                  original_stl=str(args.original_stl), source_snapshot=str(args.snapshot),
                  candidate_sha256=sha256(args.output), incumbent_sha256=sha256(args.incumbent),
                  local_exit_code=result.returncode, selected_patches=selected, changed_patches=changed,
                  parameter_changes={"region_recovery": selected}, algorithm_options={"iterations": args.iterations,
                  "smooth_passes": args.smooth_passes, "collapse_passes": args.collapse_passes,
                  "flip_passes": args.flip_passes, "target": args.target, "max_error": args.max_error},
                  boundary_graph_exact=True, hard_checks_pass=hard, quality_checks_pass=quality,
                  quality_policy=policy, required_global_mean_loss_budget=mean_loss,
                  native_size_checks_pass=size, before=bm, after=am, joint_quality=joint,
                  native_size=native, topology=topology, chains=chain, orientation=direction,
                  original_geometry_samples=original, seconds=time.perf_counter()-started,
                  full_original_stl_pipeline_timing=False)
    Path(str(args.output) + ".recovery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "incumbent", "incumbent-fields-mesh", "original-stl", "remesher", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--patches", type=int, nargs="+", required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--max-error", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--quality-policy", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--smooth-passes", type=int, default=3)
    parser.add_argument("--collapse-passes", type=int, default=8)
    parser.add_argument("--flip-passes", type=int, default=8)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report[key] for key in ("accepted", "candidate", "changed_patches",
                                                   "hard_checks_pass", "quality_checks_pass",
                                                   "native_size_checks_pass", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
