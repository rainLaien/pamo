"""Whole analytic charts plus PaMO optimization on complete reference surfaces."""
from __future__ import annotations

import time
import numpy as np

from cad_mesh.boundary_domains import prepare_surface_domains
from cad_mesh.remesh_pipeline import (
    RemeshResult, _pamo_module, _run_cuda_batches, _schedule_batches,
    _validate_output, _sampled_reference_deviation,
)


def _analytic_stage(vertices, faces, labels, constraints, patches, target, deviation, surface):
    from cad_mesh.analytic_remesh import remesh_analytic_patch
    order = np.argsort(labels, kind="stable")
    offsets = np.r_[0, np.flatnonzero(np.diff(labels[order])) + 1, len(faces)]
    extra_vertices, output_faces, output_labels, records = [], [], [], []
    accepted = set()
    vertex_count = len(vertices)
    vertex_min = np.full(vertex_count, np.iinfo(np.int64).max, dtype=np.int64)
    vertex_max = np.full(vertex_count, -1, dtype=np.int64)
    np.minimum.at(vertex_min, faces.reshape(-1), np.repeat(labels, 3))
    np.maximum.at(vertex_max, faces.reshape(-1), np.repeat(labels, 3))
    for first, last in zip(offsets[:-1], offsets[1:]):
        group = order[first:last]
        patch_id = int(labels[group[0]])
        patch = patches[patch_id]
        diagnostic = {"accepted": False, "reason": "non_developable_surface"}
        output_v = output_f = None
        if patch["type"] in ("Plane", "Cylinder", "Cone"):
            used, local_f = np.unique(faces[group], return_inverse=True)
            local_f = local_f.reshape(-1, 3)
            mapping = np.full(len(vertices), -1, dtype=np.int64)
            mapping[used] = np.arange(len(used))
            local_edges = constraints[np.all(mapping[constraints] >= 0, axis=1)]
            # Vertex-only contact with another patch is not a local edge.
            actual = np.unique(np.sort(faces[group][:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0)
            keys = actual[:, 0] * len(vertices) + actual[:, 1]
            local_edges = local_edges[np.isin(local_edges[:, 0] * len(vertices) + local_edges[:, 1], keys)]
            output_v, output_f, diagnostic = remesh_analytic_patch(
                vertices[used], local_f, patch, mapping[local_edges], target, deviation,
            )
            if (patch['type'] in ('Cylinder', 'Cone') and
                    diagnostic.get('reason') in (
                        'source_mesh_exceeds_analytic_model_deviation',
                        'trimmed_cylinder_has_ambiguous_angular_triangle',
                        'trimmed_cone_seam_overlaps_in_development',
                        'parameter_chart_folds_or_collapses_source_triangles')):
                from cad_mesh.analytic_refit import reassess_analytic_patch
                candidates, reassessment = reassess_analytic_patch(
                    vertices[used], local_f, mapping[local_edges], patch, deviation)
                if candidates:
                    # Prefer the simpler cylinder when both models certify.
                    chosen = candidates[0]
                    for candidate in candidates:
                        candidate_v, candidate_f, candidate_diagnostic = remesh_analytic_patch(
                            vertices[used], local_f, candidate, mapping[local_edges], target, deviation)
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
                before = surface.mesh_quality_metrics(vertices[used], local_f)
                after = surface.mesh_quality_metrics(output_v, output_f)
                diagnostic.update(source_metrics=before, final_metrics=after)
                # Do not replace a good patch with a worse chart merely because
                # a parameterization exists. The 3D path remains available.
                if after["mean_triangle_quality"] + 1e-8 < before["mean_triangle_quality"]:
                    output_v = output_f = None
                    diagnostic.update(accepted=False, reason="chart_reduces_mean_quality")
            if output_v is not None:
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
        records.append({"patch_id": patch_id, "type": patch["type"], **diagnostic})
        if len(records) % 250 == 0:
            print(f"[surface] Chart assessment {len(records)}/{len(patches)}; {len(accepted)} accepted", flush=True)
    vertices = np.vstack([vertices, *extra_vertices])
    return vertices, np.vstack(output_faces), np.concatenate(output_labels), accepted, records


def rebuild_surfaces(source, *, target_edge_length, sample_count=2000, split_passes=128,
                     collapse_passes=12, flip_passes=8, relax_iterations=3, seed=0,
                     maximum_boundary_splits=1000000, maximum_deviation=None,
                     batch_face_limit=40000, projection_backend="cuda"):
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
    prepared, policy = prepare_surface_domains(source)
    print(f"[surface] {len(source.report['patches'])} labels -> {len(prepared.report['patches'])} surface domains; "
          f"released {policy['released_label_edge_count']} non-geometric label edges", flush=True)
    vertices, faces, labels, boundary_splits, first_lineage = original._subdivide_labeled_patch_boundaries(
        prepared.vertices, prepared.faces, prepared.face_patch_ids, target * .75,
        explicit_constraint_edges=prepared.constraint_edges, return_lineage=True,
        maximum_splits=int(maximum_boundary_splits),
    )
    geometry_constraints = first_lineage["constraint_edges"]
    # Every original interface is now geometric: topology, crease or distinct
    # analytic surface. The CPU pass samples it once for both incident domains.
    parent_keys = first_lineage["source_constraint_edges"][:, 0] * len(vertices) + first_lineage["source_constraint_edges"][:, 1]
    source_keys = prepared.constraint_edges[:, 0] * len(vertices) + prepared.constraint_edges[:, 1]
    if not np.array_equal(parent_keys, source_keys):
        raise RuntimeError("Surface-domain constraints disagree with source topology.")
    geometry_lineage = first_lineage["source_edge_indices"]
    print(f"[surface] Shared sampling: {boundary_splits} splits; assessing whole planar/cylindrical/conical charts", flush=True)
    vertices, faces, labels, analytic_ids, charts = _analytic_stage(
        vertices, faces, labels, geometry_constraints, prepared.report["patches"], target, deviation, surface,
    )
    analytic_seconds = time.perf_counter() - begin
    print(f"[surface] {len(analytic_ids)} complete charts accepted; remaining domains use PaMO surface projection", flush=True)
    is_analytic = np.isin(labels, list(analytic_ids))
    schedule = np.zeros(len(faces), dtype=np.int64)
    if np.any(~is_analytic):
        schedule[~is_analytic] = _schedule_batches(faces[~is_analytic], labels[~is_analytic], int(batch_face_limit))
    # Accepted charts never acquire arbitrary computational cuts.
    next_batch = int(schedule.max(initial=0)) + 1
    for patch_id in sorted(analytic_ids):
        schedule[labels == patch_id] = next_batch
        next_batch += 1
    patch_count = len(prepared.report["patches"])
    combined = schedule * patch_count + labels
    vertices, faces, combined, batch_splits, second_lineage = original._subdivide_labeled_patch_boundaries(
        vertices, faces, combined, target * .75, explicit_constraint_edges=geometry_constraints,
        return_lineage=True, maximum_splits=max(0, int(maximum_boundary_splits) - boundary_splits),
    )
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
            whole_patch_optimization=True,
            whole_patch_reference=(prepared.vertices, prepared.faces, prepared.face_patch_ids),
            projection_backend=projection_backend,
        )
        output_f = np.vstack((faces[is_analytic], gpu_faces))
        output_labels = np.r_[labels[is_analytic], gpu_labels]
    else:
        output_v, output_f, output_labels = vertices.copy(), faces.copy(), labels.copy()
        stats = {"batch_count": 0, "batches": [], "sample_count": 0, "splits": 0,
                 "collapses": 0, "flips": 0, "remaining_long_edges": 0}
    print("[surface] Verifying global shared geometry, topology and target lengths", flush=True)
    validation = _validate_output(output_v, output_f, output_labels, vertices, faces, labels,
                                  geometry_constraints, prepared.corner_vertex_ids, target)
    validation["all_expected_surface_domain_ids_preserved"] = validation.pop("all_input_patch_ids_preserved")
    validation["source_patch_ids_preserved_as_provenance"] = True
    deviation_stats = _sampled_reference_deviation(output_v, output_f, source.vertices, source.faces)
    used, inverse = np.unique(output_f, return_inverse=True)
    mapping = np.full(len(output_v), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used))
    output_v, output_f = output_v[used], inverse.reshape(-1, 3)
    compact_edges = np.sort(mapping[geometry_constraints], axis=1)
    order = np.lexsort((compact_edges[:, 1], compact_edges[:, 0]))
    compact_edges, geometry_lineage = compact_edges[order], geometry_lineage[order]
    hard_set = set(map(tuple, prepared.hard_edges))
    hard_parent = np.array([tuple(e) in hard_set for e in prepared.constraint_edges], dtype=bool)
    hard_mask = hard_parent[geometry_lineage]
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
        "source_metrics": surface.mesh_quality_metrics(source.vertices, source.faces),
        "final_metrics": surface.mesh_quality_metrics(output_v, output_f),
        "validation": validation, "sampled_reference_deviation": deviation_stats,
        "analytic_and_boundary_seconds": analytic_seconds,
        "cuda_and_validation_seconds": time.perf_counter() - gpu_begin,
        "total_seconds": time.perf_counter() - begin,
        "options": {"sample_count": int(sample_count), "split_passes": int(split_passes),
                    "collapse_passes": int(collapse_passes), "flip_passes": int(flip_passes),
                    "relax_iterations": int(relax_iterations), "seed": int(seed),
                    "batch_face_limit": int(batch_face_limit)},
    })
    return RemeshResult(output_v, output_f, output_labels, compact_edges[hard_mask],
                        compact_edges[~hard_mask], mapping[prepared.corner_vertex_ids], stats,
                        geometry_lineage, prepared_source=prepared)
