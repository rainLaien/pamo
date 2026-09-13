"""Whole analytic charts plus PaMO optimization on complete reference surfaces."""
from __future__ import annotations

import time
import numpy as np

from cad_mesh.boundary_domains import prepare_surface_domains
from cad_mesh.remesh_pipeline import (
    DEFAULT_BATCH_FACE_LIMIT, RemeshResult, _pamo_module, _run_cuda_batches, _schedule_batches,
    _validate_output, _sampled_reference_deviation,
)


def _quality_assessment(vertices, faces, labels, patches, target=0.8):
    """Measure actual 3D shape even when optional diagnostics are disabled."""
    quality = _pamo_module("feature_optimize")._triangle_quality_values(vertices, faces)
    kinds = np.asarray([patch["type"] for patch in patches])[labels]

    def summarize(values):
        return {
            "face_count": int(len(values)),
            "mean_triangle_quality": float(values.mean()),
            "minimum_triangle_quality": float(values.min()),
            "fraction_below_0_2": float(np.mean(values < 0.2)),
            "fraction_below_target": float(np.mean(values < target)),
        }

    result = summarize(quality)
    result.update(metric="4*sqrt(3)*area/sum_squared_edge_lengths",
                  target_mean_triangle_quality=float(target),
                  target_met=bool(result["mean_triangle_quality"] + 1e-12 >= target),
                  per_surface_type={kind: summarize(quality[kinds == kind]) for kind in np.unique(kinds)})
    return result


def _analytic_stage(vertices, faces, labels, constraints, patches, target, deviation, surface,
                    maximum_normal_deviation_degrees=10.0, collect_diagnostics=True):
    from cad_mesh.analytic_remesh import remesh_analytic_patch
    order = np.argsort(labels, kind="stable")
    offsets = np.r_[0, np.flatnonzero(np.diff(labels[order])) + 1, len(faces)]
    extra_vertices, output_faces, output_labels, records = [], [], [], []
    per_type = {}
    accepted = set()
    vertex_count = len(vertices)
    constraint_keys = np.sort(constraints[:, 0] * np.int64(vertex_count) + constraints[:, 1])
    vertex_min = np.full(vertex_count, np.iinfo(np.int64).max, dtype=np.int64)
    vertex_max = np.full(vertex_count, -1, dtype=np.int64)
    np.minimum.at(vertex_min, faces.reshape(-1), np.repeat(labels, 3))
    np.maximum.at(vertex_max, faces.reshape(-1), np.repeat(labels, 3))
    for first, last in zip(offsets[:-1], offsets[1:]):
        group = order[first:last]
        patch_id = int(labels[group[0]])
        patch = patches[patch_id]
        patch_type = patch["type"]
        patch_stats = per_type.setdefault(patch_type, {"attempted": 0, "accepted": 0,
                                                      "input_faces": 0, "output_faces": 0})
        patch_stats["attempted"] += 1
        patch_stats["input_faces"] += int(len(group))
        stage_begin = time.perf_counter()
        diagnostic = {"accepted": False, "reason": "non_developable_surface"}
        output_v = output_f = None
        if patch_type in ("Plane", "Cylinder", "Cone"):
            used, local_f = np.unique(faces[group], return_inverse=True)
            local_f = local_f.reshape(-1, 3)
            # Query only this patch's edges against one shared index. Scanning
            # all model constraints and clearing a model-sized vertex map for
            # every small chart costs O(patches * model_size).
            actual = np.sort(faces[group][:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
            keys = np.unique(actual[:, 0] * np.int64(len(vertices)) + actual[:, 1])
            positions = np.searchsorted(constraint_keys, keys)
            present = positions < len(constraint_keys)
            present[present] &= constraint_keys[positions[present]] == keys[present]
            keys = keys[present]
            local_edges = np.searchsorted(used, np.column_stack((keys // len(vertices), keys % len(vertices))))
            chart_timing = {}
            begin_chart = time.perf_counter()
            output_v, output_f, diagnostic = remesh_analytic_patch(
                vertices[used], local_f, patch, local_edges, target, deviation,
                maximum_normal_deviation_degrees=maximum_normal_deviation_degrees,
            )
            chart_timing["rebuild_seconds"] = time.perf_counter() - begin_chart
            begin_chart = None
            if (patch_type in ('Cylinder', 'Cone') and
                    diagnostic.get('reason') in (
                        'source_mesh_exceeds_analytic_model_deviation',
                        'trimmed_cylinder_has_ambiguous_angular_triangle',
                        'trimmed_cone_seam_overlaps_in_development',
                        'parameter_chart_folds_or_collapses_source_triangles')):
                from cad_mesh.analytic_refit import reassess_analytic_patch
                begin_reassess = time.perf_counter()
                candidates, reassessment = reassess_analytic_patch(
                    vertices[used], local_f, local_edges, patch, deviation)
                chart_timing["reassessment_seconds"] = time.perf_counter() - begin_reassess
                diagnostic["reassessment_candidates"] = len(candidates)
                if candidates:
                    # Prefer the simpler cylinder when both models certify.
                    chosen = candidates[0]
                    for candidate in candidates:
                        begin_candidate = time.perf_counter()
                        candidate_v, candidate_f, candidate_diagnostic = remesh_analytic_patch(
                            vertices[used], local_f, candidate, local_edges, target, deviation,
                            maximum_normal_deviation_degrees=maximum_normal_deviation_degrees)
                        if "timing" in candidate_diagnostic:
                            candidate_diagnostic["timing"]["rebuild_candidate_seconds"] = (
                                candidate_diagnostic["timing"].get("chart_total_seconds", 0.0)
                                + (time.perf_counter() - begin_candidate)
                            )
                        chart_timing["candidate_seconds"] = chart_timing.get("candidate_seconds", 0.0) + (
                            time.perf_counter() - begin_candidate
                        )
                        output_v, output_f, diagnostic = candidate_v, candidate_f, candidate_diagnostic
                        chosen = candidate
                        if output_v is not None:
                            break
                    patch.update(type=chosen['type'], parameters=chosen['parameters'])
                else:
                    patch.update(type='Freeform', parameters=None)
                    diagnostic.update(accepted=False, reason='analytic_refit_failed_full_patch_validation')
                reassessment['selected_type'] = patch['type']
                diagnostic['model_reassessment'] = reassessment
            if output_v is not None:
                if not np.array_equal(output_v[:len(used)], vertices[used]):
                    raise RuntimeError("Analytic chart changed the shared source vertex prefix.")
                shared = np.flatnonzero(vertex_min[used] != vertex_max[used])
                if not np.all(np.isin(shared, output_f)):
                    output_v = output_f = None
                    diagnostic.update(accepted=False, reason="chart_discards_shared_vertex_contact")
            if output_v is not None:
                if collect_diagnostics:
                    before = surface.mesh_quality_metrics(vertices[used], local_f)
                    after = surface.mesh_quality_metrics(output_v, output_f)
                    diagnostic.update(source_metrics=before, final_metrics=after)
                else:
                    # The acceptance metric is the actual 3D triangle shape,
                    # even when detailed reporting is disabled.
                    quality = _pamo_module("feature_optimize")._triangle_quality_values
                    before = {"mean_triangle_quality": float(quality(vertices[used], local_f).mean())}
                    after = {"mean_triangle_quality": float(quality(output_v, output_f).mean())}
                # Do not replace a good patch with a worse chart merely because
                # a parameterization exists. The 3D path remains available.
                before_quality = before["mean_triangle_quality"]
                after_quality = after["mean_triangle_quality"]
                diagnostic["quality_comparison_metric"] = "euclidean"
                diagnostic["source_mean_triangle_quality"] = before_quality
                diagnostic["output_mean_triangle_quality"] = after_quality
                if after_quality + 1e-8 < before_quality:
                    output_v = output_f = None
                    diagnostic.update(accepted=False, reason="chart_reduces_mean_quality")
            if output_v is not None:
                patch_stats["accepted"] += 1
                patch_stats["output_faces"] += int(len(output_f))
                if "timing" in diagnostic:
                    diagnostic["timing"]["chart_total_seconds"] = float(chart_timing.get("rebuild_seconds", 0.0))
                extra = output_v[len(used):]
                global_ids = np.r_[used, np.arange(vertex_count, vertex_count + len(extra))]
                vertex_count += len(extra)
                extra_vertices.append(extra)
                output_faces.append(global_ids[output_f])
                output_labels.append(np.full(len(output_f), patch_id, dtype=np.int64))
                accepted.add(patch_id)
        if output_v is None:
            output_faces.append(faces[group])
            output_labels.append(labels[group])
        diagnostic["timing"] = {
            "stage_seconds": float(time.perf_counter() - stage_begin),
            **(diagnostic.get("timing") or {}),
        }
        records.append({"patch_id": patch_id, "type": patch_type, **diagnostic})
        if len(records) % 50 == 0:
            print(f"[surface] Chart assessment {len(records)}/{len(patches)}; {len(accepted)} accepted "
                  f"(last {patch_id} {patch_type} {diagnostic.get('reason')})", flush=True)
    vertices = np.vstack([vertices, *extra_vertices])
    summary = {
        "per_type": {
            key: {
                "attempted": int(value["attempted"]),
                "accepted": int(value["accepted"]),
                "input_faces": int(value["input_faces"]),
                "output_faces": int(value["output_faces"]),
            }
            for key, value in per_type.items()
        }
    }
    return vertices, np.vstack(output_faces), np.concatenate(output_labels), accepted, records, summary


def rebuild_surfaces(source, *, target_edge_length, sample_count=2000, split_passes=128,
                     collapse_passes=12, flip_passes=8, relax_iterations=3, seed=0,
                     maximum_boundary_splits=1000000, maximum_deviation=None,
                     maximum_normal_deviation_degrees=10.0,
                     batch_face_limit=DEFAULT_BATCH_FACE_LIMIT, projection_backend="cuda",
                     trace_surface_validation=False, collect_diagnostics=True):
    target = float(target_edge_length)
    if projection_backend not in ("cuda", "cpu"):
        raise ValueError("projection_backend must be 'cuda' or 'cpu'.")
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target_edge_length must be finite and positive.")
    for name, value, minimum in (("sample_count", sample_count, 1), ("split_passes", split_passes, 1),
                                ("collapse_passes", collapse_passes, 0), ("flip_passes", flip_passes, 0),
                                ("relax_iterations", relax_iterations, 0), ("batch_face_limit", batch_face_limit, 1),
                                ("maximum_boundary_splits", maximum_boundary_splits, 0)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    # Geometry fidelity must not change when only mesh density is changed.
    # Preserve the historical default at the default edge length (5% diagonal),
    # but anchor it to the immutable input mesh rather than the requested edge.
    model_diagonal = float(np.linalg.norm(np.ptp(source.vertices, axis=0)))
    deviation = model_diagonal * .00025 if maximum_deviation is None else float(maximum_deviation)
    if not np.isfinite(deviation) or deviation < 0:
        raise ValueError("maximum_deviation must be finite and nonnegative.")
    print(f"[surface] Maximum edge {target:.9g}; geometry deviation {deviation:.9g} "
          f"({'model-scale default' if maximum_deviation is None else 'explicit override'})", flush=True)
    import torch
    import trimesh
    if not torch.cuda.is_available():
        raise RuntimeError("PaMO surface remeshing requires an available CUDA GPU.")
    surface = _pamo_module("surface_sample")
    original = _pamo_module("original_constrained")
    begin = time.perf_counter()
    stage_timing = {"total": 0.0, "prepare_surface_domains": 0.0, "surface_split": 0.0,
                    "analytic_stage": 0.0, "chart_batching": 0.0, "cuda_batch": 0.0,
                    "validation_and_compaction": 0.0}
    stage_begin = time.perf_counter()
    prepared, policy = prepare_surface_domains(source)
    stage_timing["prepare_surface_domains"] = time.perf_counter() - stage_begin
    print(f"[surface] {len(source.report['patches'])} labels -> {len(prepared.report['patches'])} surface domains; "
          f"released {policy['released_label_edge_count']} non-geometric label edges", flush=True)
    stage_begin = time.perf_counter()
    vertices, faces, labels, boundary_splits, first_lineage = original._subdivide_labeled_patch_boundaries(
        prepared.vertices, prepared.faces, prepared.face_patch_ids, target * .75,
        explicit_constraint_edges=prepared.constraint_edges, return_lineage=True,
        maximum_splits=int(maximum_boundary_splits),
    )
    stage_timing["surface_split"] += time.perf_counter() - stage_begin
    geometry_constraints = first_lineage["constraint_edges"]
    # Every original interface is now geometric: topology, crease or distinct
    # analytic surface. The CPU pass samples it once for both incident domains.
    parent_keys = first_lineage["source_constraint_edges"][:, 0] * len(vertices) + first_lineage["source_constraint_edges"][:, 1]
    source_keys = prepared.constraint_edges[:, 0] * len(vertices) + prepared.constraint_edges[:, 1]
    if not np.array_equal(parent_keys, source_keys):
        raise RuntimeError("Surface-domain constraints disagree with source topology.")
    geometry_lineage = first_lineage["source_edge_indices"]
    print(f"[surface] Shared sampling: {boundary_splits} splits; assessing whole planar/cylindrical/conical charts", flush=True)
    stage_begin = time.perf_counter()
    vertices, faces, labels, analytic_ids, charts, chart_summary = _analytic_stage(
        vertices, faces, labels, geometry_constraints, prepared.report["patches"], target, deviation, surface,
        maximum_normal_deviation_degrees=maximum_normal_deviation_degrees,
        collect_diagnostics=collect_diagnostics,
    )
    stage_timing["analytic_stage"] = time.perf_counter() - stage_begin
    print(f"[surface] {len(analytic_ids)} complete charts accepted; remaining domains use PaMO surface projection", flush=True)
    if chart_summary["per_type"]:
        details = ", ".join(
            f"{name}: {value['accepted']}/{value['attempted']} charts, "
            f"{value['input_faces']}->{value['output_faces']} faces"
            for name, value in chart_summary["per_type"].items()
        )
        print(f"[surface] Chart summary by type: {details}", flush=True)
    is_analytic = np.isin(labels, list(analytic_ids))
    schedule = np.zeros(len(faces), dtype=np.int64)
    if np.any(~is_analytic):
        stage_begin = time.perf_counter()
        schedule[~is_analytic] = _schedule_batches(faces[~is_analytic], labels[~is_analytic], int(batch_face_limit))
        stage_timing["chart_batching"] += time.perf_counter() - stage_begin
    # Accepted charts never acquire arbitrary computational cuts.
    next_batch = int(schedule.max(initial=0)) + 1
    patch_count = len(prepared.report["patches"])
    chart_batches = np.zeros(patch_count, dtype=np.int64)
    chart_batches[sorted(analytic_ids)] = np.arange(next_batch, next_batch + len(analytic_ids))
    schedule[is_analytic] = chart_batches[labels[is_analytic]]
    stage_begin = time.perf_counter()
    combined = schedule * patch_count + labels
    vertices, faces, combined, batch_splits, second_lineage = original._subdivide_labeled_patch_boundaries(
        vertices, faces, combined, target * .75, explicit_constraint_edges=geometry_constraints,
        return_lineage=True, maximum_splits=max(0, int(maximum_boundary_splits) - boundary_splits),
    )
    stage_timing["surface_split"] += time.perf_counter() - stage_begin
    labels, batch_ids = combined % patch_count, combined // patch_count
    parents = second_lineage["source_constraint_edges"]
    parent_keys = parents[:, 0] * len(vertices) + parents[:, 1]
    geometry_keys = geometry_constraints[:, 0] * len(vertices) + geometry_constraints[:, 1]
    parent_map = np.full(len(parents), -1, dtype=np.int64)
    known = np.isin(parent_keys, geometry_keys)
    parent_map[known] = geometry_lineage[np.searchsorted(geometry_keys, parent_keys[known])]
    all_constraints = second_lineage["constraint_edges"]
    source_lineage = parent_map[second_lineage["source_edge_indices"]]
    genuine = source_lineage >= 0
    geometry_constraints = all_constraints[genuine]
    geometry_lineage = source_lineage[genuine]
    is_analytic = np.isin(labels, list(analytic_ids))
    gpu_begin = time.perf_counter()
    if np.any(~is_analytic):
        output_v, gpu_faces, gpu_labels, stats = _run_cuda_batches(
            surface, torch, trimesh, vertices, faces[~is_analytic], labels[~is_analytic], batch_ids[~is_analytic],
            all_constraints, prepared.corner_vertex_ids, target, sample_count, seed,
            split_passes, collapse_passes, flip_passes, relax_iterations, deviation,
            maximum_normal_deviation_degrees=maximum_normal_deviation_degrees,
            whole_patch_optimization=True,
            whole_patch_reference=(prepared.vertices, prepared.faces, prepared.face_patch_ids),
            projection_backend=projection_backend,
            analytic_patch_models=prepared.report["patches"],
            trace_surface_validation=trace_surface_validation,
            collect_diagnostics=collect_diagnostics,
        )
        output_f = np.vstack((faces[is_analytic], gpu_faces))
        output_labels = np.r_[labels[is_analytic], gpu_labels]
    else:
        output_v, output_f, output_labels = vertices.copy(), faces.copy(), labels.copy()
        stats = {"batch_count": 0, "batches": [], "sample_count": 0, "splits": 0,
                 "collapses": 0, "flips": 0, "remaining_long_edges": 0}
    stage_timing["cuda_batch"] = time.perf_counter() - gpu_begin
    print("[surface] Verifying global shared geometry, topology and target lengths", flush=True)
    stage_begin = time.perf_counter()
    validation, validated_edges = _validate_output(output_v, output_f, output_labels, vertices, faces, labels,
                                  geometry_constraints, prepared.corner_vertex_ids, target, return_edges=True)
    validation["all_expected_surface_domain_ids_preserved"] = validation.pop("all_input_patch_ids_preserved")
    validation["source_patch_ids_preserved_as_provenance"] = True
    deviation_stats = (_sampled_reference_deviation(output_v, output_f, source.vertices, source.faces)
                       if collect_diagnostics else None)
    referenced = np.zeros(len(output_v), dtype=bool)
    referenced[output_f.reshape(-1)] = True
    used = np.flatnonzero(referenced)
    mapping = np.full(len(output_v), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used))
    output_v, output_f = output_v[used], mapping[output_f]
    compact_edges = np.sort(mapping[geometry_constraints], axis=1)
    order = np.lexsort((compact_edges[:, 1], compact_edges[:, 0]))
    compact_edges, geometry_lineage = compact_edges[order], geometry_lineage[order]
    hard_set = set(map(tuple, prepared.hard_edges))
    hard_parent = np.array([tuple(e) in hard_set for e in prepared.constraint_edges], dtype=bool)
    hard_mask = hard_parent[geometry_lineage]
    stage_timing["validation_and_compaction"] = time.perf_counter() - stage_begin
    stage_begin = time.perf_counter()
    source_metrics = (surface.mesh_quality_metrics(source.vertices, source.faces)
                      if collect_diagnostics else None)
    final_metrics = (surface.mesh_quality_metrics(output_v, output_f, edges=mapping[validated_edges])
                     if collect_diagnostics else None)
    quality_assessment = _quality_assessment(output_v, output_f, output_labels, prepared.report["patches"])
    quality_status = "met" if quality_assessment["target_met"] else "not reached"
    print(f"[surface] 3D mean triangle quality {quality_assessment['mean_triangle_quality']:.6f}; "
          f"target 0.800000 {quality_status}; "
          f"{quality_assessment['fraction_below_0_2']:.2%} of faces below 0.2", flush=True)
    stage_timing["quality_metrics"] = time.perf_counter() - stage_begin
    stage_timing["total"] = time.perf_counter() - begin
    stats.update({
        "backend": "pamo.whole_surface + analytic_parameter_domain",
        "method": "surface", "reference_surface": "patch_scoped_triangle_mesh_or_analytic_chart",
        "projection_backend": projection_backend,
        "input_vertices": len(source.vertices), "input_faces": len(source.faces),
        "output_vertices": len(output_v), "output_faces": len(output_f),
        "input_patch_count": len(source.report["patches"]), "output_patch_count": patch_count,
        "input_labels_retained_as_provenance": True, "repartitioned": policy["merged_label_pairs"] > 0,
        "boundary_policy": policy, "target_edge_length": target, "requested_maximum_deviation": deviation,
        "deviation_policy": "model_diagonal_0.00025" if maximum_deviation is None else "explicit",
        "boundary_splits": boundary_splits + batch_splits,
        "computational_interfaces": int(np.count_nonzero(~genuine)),
        "computational_interfaces_are_output_patch_boundaries": False,
        "analytic_patch_count": len(analytic_ids), "analytic_charts": charts,
        "patch_projection_targets": {str(i): "analytic_surface" for i in analytic_ids},
        "source_metrics": source_metrics,
        "final_metrics": final_metrics,
        "quality_assessment": quality_assessment,
        "validation": validation, "sampled_reference_deviation": deviation_stats,
        "analytic_and_boundary_seconds": (stage_timing["prepare_surface_domains"]
                                           + stage_timing["surface_split"] + stage_timing["analytic_stage"]
                                           + stage_timing["chart_batching"]),
        "cuda_and_validation_seconds": stage_timing["cuda_batch"] + stage_timing["validation_and_compaction"],
        "timing_breakdown_seconds": stage_timing,
        "total_seconds": time.perf_counter() - begin,
        "analytic_chart_summary": chart_summary,
        "options": {"sample_count": int(sample_count), "split_passes": int(split_passes),
                    "collapse_passes": int(collapse_passes), "flip_passes": int(flip_passes),
                    "relax_iterations": int(relax_iterations), "seed": int(seed),
                    "batch_face_limit": int(batch_face_limit),
                    "trace_surface_validation": bool(trace_surface_validation),
                    "maximum_normal_deviation_degrees": float(maximum_normal_deviation_degrees)},
    })
    print("[surface] Timing: " + ", ".join(
        f"{name} {seconds:.2f}s" for name, seconds in stage_timing.items() if name != "total"), flush=True)
    return RemeshResult(output_v, output_f, output_labels, compact_edges[hard_mask],
                        compact_edges[~hard_mask], mapping[prepared.corner_vertex_ids], stats,
                        geometry_lineage, prepared_source=prepared)
