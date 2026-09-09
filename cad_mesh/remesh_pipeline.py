"""Use PaMO's CUDA surface remesher with the existing CAD partition."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import time
import types

import numpy as np


@dataclass
class RemeshResult:
    vertices: np.ndarray
    faces: np.ndarray
    face_patch_ids: np.ndarray
    hard_edges: np.ndarray
    smooth_edges: np.ndarray
    corner_vertex_ids: np.ndarray
    stats: dict
    # Row indices into source.constraint_edges, aligned with the sorted union
    # of output hard_edges and smooth_edges (not C++ global edge IDs).
    source_constraint_edge_ids: np.ndarray
    # Whole-surface mode can consolidate redundant labels before remeshing.
    prepared_source: object | None = None


def _pamo_module(name):
    """Load this checkout's light modules without constructing the SDF model."""
    package_name = "_cadmesh_pamo_runtime"
    path = Path(__file__).resolve().parents[1] / "simp_cuda" / "pamo"
    if not (path / "surface_sample.py").is_file():
        raise RuntimeError(f"PaMO source checkout is missing: {path}")
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.{name}")


def _topology(faces, labels):
    directed = faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    minimum = np.full(len(edges), np.iinfo(np.int64).max, dtype=np.int64)
    maximum = np.full(len(edges), -1, dtype=np.int64)
    face_labels = np.repeat(labels, 3)
    np.minimum.at(minimum, inverse, face_labels)
    np.maximum.at(maximum, inverse, face_labels)
    orientation = np.zeros(len(edges), dtype=np.int64)
    np.add.at(orientation, inverse, np.where(directed[:, 0] < directed[:, 1], 1, -1))
    return edges, counts, minimum, maximum, orientation


def _edge_positions(edges, queries, vertex_count):
    keys = edges[:, 0] * np.int64(vertex_count) + edges[:, 1]
    query_keys = queries[:, 0] * np.int64(vertex_count) + queries[:, 1]
    indices = np.searchsorted(keys, query_keys)
    if np.any(indices >= len(edges)) or np.any(keys[indices] != query_keys):
        raise RuntimeError("A fixed shared boundary edge disappeared during remeshing.")
    return indices


def _validate_output(vertices, faces, labels, reference_vertices, reference_faces,
                     reference_labels, constraints, corners, target):
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise RuntimeError("PaMO returned invalid vertices or triangle indices.")
    if labels.shape != (len(faces),) or not np.array_equal(np.unique(labels), np.unique(reference_labels)):
        raise RuntimeError("PaMO lost a partition or returned inconsistent face labels.")
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if np.any(np.einsum("ij,ij->i", cross, cross) <= 0):
        raise RuntimeError("PaMO returned a degenerate triangle.")
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise RuntimeError("PaMO returned duplicate triangles.")
    locked = np.unique(np.concatenate((constraints.reshape(-1), corners)))
    if len(vertices) < len(reference_vertices) or not np.array_equal(vertices[locked], reference_vertices[locked]):
        raise RuntimeError("A shared boundary or corner moved during remeshing.")
    before = _topology(reference_faces, reference_labels)
    after = _topology(faces, labels)
    before_indices = _edge_positions(before[0], constraints, len(vertices))
    after_indices = _edge_positions(after[0], constraints, len(vertices))
    for index in (1, 2, 3, 4):
        if not np.array_equal(before[index][before_indices], after[index][after_indices]):
            raise RuntimeError("Boundary incidence, partition sides or winding changed.")
    interfaces = (after[1] != 2) | (after[2] != after[3])
    allowed = np.zeros(len(after[0]), dtype=bool)
    allowed[after_indices] = True
    if np.any(interfaces & ~allowed):
        raise RuntimeError("Remeshing created a new crack, nonmanifold edge or patch interface.")
    new_bad_winding = np.count_nonzero((after[1] == 2) & (after[4] != 0))
    old_bad_winding = np.count_nonzero((before[1] == 2) & (before[4] != 0))
    if new_bad_winding > old_bad_winding:
        raise RuntimeError("Remeshing introduced inconsistent face winding.")
    lengths = np.linalg.norm(vertices[after[0][:, 1]] - vertices[after[0][:, 0]], axis=1)
    longest = float(lengths.max(initial=0))
    if longest > target * (1 + 1e-6):
        raise RuntimeError(
            f"Maximum edge {longest:.9g} exceeds target {target:.9g}; "
            "increase split passes or choose a larger target edge length."
        )
    return {
        "passed": True, "all_input_patch_ids_preserved": True,
        "fixed_boundary_coordinates_exact": True, "corner_coordinates_exact": True,
        "shared_boundary_incidence_preserved": True, "duplicate_faces": 0,
        "degenerate_faces": 0, "new_cracks": 0, "new_nonmanifold_edges": 0,
        "new_inconsistent_winding_edges": 0, "maximum_edge_length": longest,
        "output_open_edges": int(np.count_nonzero(after[1] == 1)),
        "output_nonmanifold_edges": int(np.count_nonzero(after[1] > 2)),
    }


def _sampled_reference_deviation(vertices, faces, reference_vertices, reference_faces):
    import igl
    # Diagnostic samples, not a certified Hausdorff upper bound.
    count = min(6000, len(faces))
    indices = np.linspace(0, len(faces) - 1, count, dtype=np.int64)
    sampled_faces = faces[indices]
    vertex_ids = np.unique(sampled_faces)
    points = np.vstack((vertices[vertex_ids], vertices[sampled_faces].mean(axis=1)))
    distances, _, _ = igl.point_mesh_squared_distance(points, reference_vertices, reference_faces)
    return {
        "maximum": float(np.sqrt(np.maximum(distances, 0)).max(initial=0)),
        "sample_count": int(len(points)), "hausdorff_upper_bound": False,
        "direction": "output_samples_to_reference_triangle_mesh",
    }


def _schedule_batches(faces, labels, maximum_faces):
    """Pack whole patches and BFS chunks; these IDs are never output labels."""
    if len(faces) <= maximum_faces:
        return np.zeros(len(faces), dtype=np.int64)
    order = np.argsort(labels, kind="stable")
    offsets = np.r_[0, np.flatnonzero(np.diff(labels[order])) + 1, len(faces)]
    groups = [order[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    graph = None
    if max(map(len, groups)) > maximum_faces:
        from scipy.sparse import coo_matrix
        directed = faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
        _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
        edge_order = np.argsort(inverse, kind="stable")
        starts = np.r_[0, np.cumsum(counts[:-1])][counts == 2]
        left = edge_order[starts] // 3
        right = edge_order[starts + 1] // 3
        same = labels[left] == labels[right]
        left, right = left[same], right[same]
        rows, cols = np.r_[left, right], np.r_[right, left]
        graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                           shape=(len(faces), len(faces))).tocsr()
    visited = np.zeros(len(faces), dtype=bool)
    batch_ids = np.full(len(faces), -1, dtype=np.int64)
    batch = 0
    pending = []
    pending_size = 0
    for group in groups:
        if len(group) <= maximum_faces:
            if pending_size + len(group) > maximum_faces:
                batch_ids[np.concatenate(pending)] = batch
                batch += 1
                pending, pending_size = [], 0
            pending.append(group)
            pending_size += len(group)
            continue
        if pending:
            batch_ids[np.concatenate(pending)] = batch
            batch += 1
            pending, pending_size = [], 0
        traversal = []
        for seed in group:
            if visited[seed]:
                continue
            queue = [int(seed)]
            visited[seed] = True
            for face in queue:
                traversal.append(face)
                for neighbor in graph.indices[graph.indptr[face]:graph.indptr[face + 1]]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(int(neighbor))
        for start in range(0, len(traversal), maximum_faces):
            batch_ids[traversal[start:start + maximum_faces]] = batch
            batch += 1
    if pending:
        batch_ids[np.concatenate(pending)] = batch
    if np.any(batch_ids < 0):
        raise RuntimeError("Computational batch scheduling omitted a triangle.")
    return batch_ids


def _run_cuda_batches(surface, torch, trimesh, vertices, faces, labels,
                      batch_ids, constraints, corners, target, sample_count,
                      seed, split_passes, collapse_passes, flip_passes,
                      relax_iterations, deviation, whole_patch_optimization=False,
                      whole_patch_reference=None, projection_backend="cuda"):
    if whole_patch_optimization and isinstance(whole_patch_reference, tuple):
        whole_patch_reference = surface._make_reference_projector(*whole_patch_reference, backend=projection_backend)
    order = np.argsort(batch_ids, kind="stable")
    offsets = np.r_[0, np.flatnonzero(np.diff(batch_ids[order])) + 1, len(faces)]
    groups = [order[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    used_by_group = [np.unique(faces[group]) for group in groups]
    usage = np.zeros(len(vertices), dtype=np.int32)
    for used in used_by_group:
        usage[used] += 1
    # Include a geometric interface touched only at one vertex by this batch,
    # including interfaces belonging to an already rebuilt analytic chart.
    shared_corners = np.unique(np.r_[corners, constraints.reshape(-1), np.flatnonzero(usage > 1)])
    output_base = vertices.copy()
    new_vertices, all_faces, all_labels, records = [], [], [], []
    next_vertex = len(vertices)
    constraint_keys = constraints[:, 0] * len(vertices) + constraints[:, 1]
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    total_area = float(areas.sum())
    del triangles
    for number, (group, used) in enumerate(zip(groups, used_by_group)):
        batch_begin = time.perf_counter()
        local_map = np.full(len(vertices), -1, dtype=np.int64)
        local_map[used] = np.arange(len(used))
        local_vertices = vertices[used]
        local_faces = local_map[faces[group]]
        group_edges = np.unique(np.sort(faces[group][:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0)
        keys = group_edges[:, 0] * len(vertices) + group_edges[:, 1]
        constraint_indices = np.flatnonzero(np.isin(constraint_keys, keys))
        local_constraints = local_map[constraints[constraint_indices]]
        local_corners = local_map[np.intersect1d(shared_corners, used)]
        count = max(1, int(round(sample_count * float(areas[group].sum()) / total_area)))
        print(f"[remesh] CUDA batch {number + 1}/{len(groups)}: {len(group):,} faces, "
              f"{count} requested samples", flush=True)
        reference = trimesh.Trimesh(vertices=local_vertices, faces=local_faces, process=False)
        output_v, output_f, stats = surface.surface_sample_remesh(
            reference, torch.as_tensor(local_vertices, dtype=torch.float64, device="cuda"),
            torch.as_tensor(local_faces, dtype=torch.long, device="cuda"),
            sample_count=count, poisson_radius=target / 2, seed=int(seed) + number,
            external_face_patch_ids=labels[group], fixed_constraint_edges=local_constraints,
            fixed_corner_vertex_ids=local_corners, flip_passes=int(flip_passes),
            relax_iterations=int(relax_iterations), split_passes=int(split_passes),
            collapse_passes=int(collapse_passes), maximum_edge_ratio=2,
            minimum_edge_ratio=.5, maximum_surface_deviation_ratio=deviation / (target / 2),
            whole_patch_optimization=whole_patch_optimization,
            whole_patch_reference=whole_patch_reference,
            projection_backend=projection_backend,
        )
        output_labels = np.asarray(stats.pop("face_patch_ids"), dtype=np.int64)
        stats.pop("source_face_ids", None)
        output_edges = np.unique(
            np.sort(output_f[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0,
        )
        maximum_length = float(np.linalg.norm(
            output_v[output_edges[:, 1]] - output_v[output_edges[:, 0]], axis=1,
        ).max(initial=0))
        if maximum_length > target * (1.0 + 1e-6):
            raise RuntimeError(
                f"CUDA batch {number + 1} maximum edge {maximum_length:.9g} exceeds "
                f"target {target:.9g}; increase --split-passes or choose a larger target."
            )
        stats["verified_maximum_edge_length"] = maximum_length
        if not np.array_equal(output_v[local_corners], local_vertices[local_corners]):
            raise RuntimeError("A computational batch moved a shared global vertex.")
        output_base[used] = output_v[:len(used)]
        extra = output_v[len(used):]
        global_ids = np.r_[used, np.arange(next_vertex, next_vertex + len(extra))]
        next_vertex += len(extra)
        new_vertices.append(extra)
        all_faces.append(global_ids[output_f])
        all_labels.append(output_labels)
        stats["batch_seconds"] = time.perf_counter() - batch_begin
        print(f"[remesh] CUDA batch {number + 1}/{len(groups)} completed in "
              f"{stats['batch_seconds']:.2f}s", flush=True)
        records.append({"batch": number, "input_faces": len(group),
                        "output_faces": len(output_f), **stats})
        # Release cached allocations between batches on GPUs with limited VRAM.
        torch.cuda.empty_cache()
    output_vertices = np.vstack([output_base, *new_vertices])
    output_faces = np.vstack(all_faces)
    output_labels = np.concatenate(all_labels)
    stats = {"batch_count": len(groups), "batches": records,
             "final_metrics": surface.mesh_quality_metrics(output_vertices, output_faces),
             "sample_count": sum(r["sample_count"] for r in records),
             "splits": sum(r["splits"] for r in records),
             "collapses": sum(r["collapses"] for r in records),
             "flips": sum(r["flips"] for r in records),
             "remaining_long_edges": sum(r["remaining_long_edges"] for r in records),
             "external_partition": True}
    return output_vertices, output_faces, output_labels, stats


def remesh_partition(source, *, target_edge_length, sample_count=2000,
                     split_passes=128, collapse_passes=12, flip_passes=8,
                     relax_iterations=3, seed=0, maximum_boundary_splits=1000000,
                     maximum_deviation=None, batch_face_limit=40000, method="surface",
                     projection_backend="cuda"):
    """Remesh one shared mesh; preserve labels and both kinds of interfaces.

    The target edge length is in source units. Surface sampling/projection uses
    the reference triangles for all six CAD categories; no SDF or repartitioning
    takes place. The CPU boundary pass and CUDA optimization are PaMO routines.
    """
    if method == "surface":
        from cad_mesh.surface_rebuild import rebuild_surfaces
        return rebuild_surfaces(
            source, target_edge_length=target_edge_length, sample_count=sample_count,
            split_passes=split_passes, collapse_passes=collapse_passes,
            flip_passes=flip_passes, relax_iterations=relax_iterations, seed=seed,
            maximum_boundary_splits=maximum_boundary_splits,
            maximum_deviation=maximum_deviation, batch_face_limit=batch_face_limit,
            projection_backend=projection_backend,
        )
    if method != "legacy":
        raise ValueError("method must be 'surface' or 'legacy'.")
    target = float(target_edge_length)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target_edge_length must be finite and positive.")
    for name, value, minimum in (("sample_count", sample_count, 1), ("split_passes", split_passes, 1),
                                 ("collapse_passes", collapse_passes, 0), ("flip_passes", flip_passes, 0),
                                 ("relax_iterations", relax_iterations, 0),
                                 ("maximum_boundary_splits", maximum_boundary_splits, 0),
                                 ("batch_face_limit", batch_face_limit, 1)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    deviation = target * .005 if maximum_deviation is None else float(maximum_deviation)
    if not np.isfinite(deviation) or deviation < 0:
        raise ValueError("maximum_deviation must be finite and nonnegative.")
    try:
        import torch
        import trimesh
    except ImportError as error:
        raise RuntimeError("Run with the PaMO Python environment (.venv/Scripts/python.exe).") from error
    if not torch.cuda.is_available():
        raise RuntimeError("PaMO partition remeshing requires an available CUDA GPU.")
    original = _pamo_module("original_constrained")
    surface = _pamo_module("surface_sample")
    begin = time.perf_counter()
    print("[remesh] Finalizing one shared boundary graph with PaMO...", flush=True)
    # Leave headroom for the float comparison used by GPU length checks.
    schedule = _schedule_batches(source.faces, source.face_patch_ids, int(batch_face_limit))
    patch_count = len(source.report["patches"])
    combined_labels = schedule * patch_count + source.face_patch_ids
    vertices, faces, combined_labels, splits, lineage = original._subdivide_labeled_patch_boundaries(
        source.vertices, source.faces, combined_labels, target * .9,
        explicit_constraint_edges=source.constraint_edges, return_lineage=True,
        maximum_splits=int(maximum_boundary_splits),
    )
    parents = lineage["source_constraint_edges"]
    batch_ids, labels = combined_labels // patch_count, combined_labels % patch_count
    all_constraints = lineage["constraint_edges"]
    source_keys = source.constraint_edges[:, 0] * len(vertices) + source.constraint_edges[:, 1]
    parent_keys = parents[:, 0] * len(vertices) + parents[:, 1]
    parent_map = np.full(len(parents), -1, dtype=np.int64)
    known = np.isin(parent_keys, source_keys)
    parent_map[known] = np.searchsorted(source_keys, parent_keys[known])
    source_lineage = parent_map[lineage["source_edge_indices"]]
    genuine = source_lineage >= 0
    constraints = all_constraints[genuine]
    source_lineage = source_lineage[genuine]
    hard_keys = set(map(tuple, source.hard_edges))
    hard_parent = np.array([tuple(edge) in hard_keys for edge in source.constraint_edges], dtype=bool)
    is_hard = hard_parent[source_lineage]
    boundary_seconds = time.perf_counter() - begin
    print(f"[remesh] {splits:,} shared edge splits; optimizing {len(faces):,} faces "
          f"inside {len(np.unique(labels)):,} existing partitions on CUDA...", flush=True)
    gpu_begin = time.perf_counter()
    output_vertices, output_faces, output_labels, stats = _run_cuda_batches(
        surface, torch, trimesh, vertices, faces, labels, batch_ids,
        all_constraints, source.corner_vertex_ids, target, sample_count, seed,
        split_passes, collapse_passes, flip_passes, relax_iterations, deviation,
    )
    gpu_seconds = time.perf_counter() - gpu_begin
    print("[remesh] Checking shared boundaries, corners, topology and edge lengths...", flush=True)
    validation = _validate_output(output_vertices, output_faces, output_labels,
                                  vertices, faces, labels, constraints,
                                  source.corner_vertex_ids, target)
    sampled_deviation = _sampled_reference_deviation(
        output_vertices, output_faces, source.vertices, source.faces
    )
    # Compact only after all fixed global IDs have been independently verified.
    used, inverse = np.unique(output_faces, return_inverse=True)
    mapping = np.full(len(output_vertices), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used))
    output_vertices = output_vertices[used]
    output_faces = inverse.reshape(-1, 3)
    compact_edges = np.sort(mapping[constraints], axis=1)
    if np.any(compact_edges < 0) or np.any(mapping[source.corner_vertex_ids] < 0):
        raise RuntimeError("Remeshing removed a fixed boundary or corner vertex.")
    order = np.lexsort((compact_edges[:, 1], compact_edges[:, 0]))
    compact_edges = compact_edges[order]
    compact_hard = is_hard[order]
    stats.update({
        "backend": "pamo.original_constrained + pamo.surface_sample_remesh.cuda",
        "reference_surface": "original_triangle_mesh", "repartitioned": False,
        "target_edge_length": target, "requested_maximum_deviation": deviation,
        "input_vertices": int(len(source.vertices)), "input_faces": int(len(source.faces)),
        "output_vertices": int(len(output_vertices)), "output_faces": int(len(output_faces)),
        "input_patch_count": int(len(np.unique(source.face_patch_ids))),
        "output_patch_count": int(len(np.unique(output_labels))),
        "boundary_splits": int(splits), "boundary_seconds": boundary_seconds,
        "computational_interfaces": int(len(all_constraints) - len(constraints)),
        "computational_interfaces_are_output_patch_boundaries": False,
        "cuda_seconds": gpu_seconds, "total_seconds": time.perf_counter() - begin,
        "source_metrics": surface.mesh_quality_metrics(source.vertices, source.faces),
        "validation": validation, "sampled_reference_deviation": sampled_deviation,
        "options": {"sample_count": int(sample_count), "split_passes": int(split_passes),
                    "collapse_passes": int(collapse_passes), "flip_passes": int(flip_passes),
                    "relax_iterations": int(relax_iterations), "seed": int(seed),
                    "batch_face_limit": int(batch_face_limit)},
    })
    return RemeshResult(output_vertices, output_faces, output_labels,
                        compact_edges[compact_hard], compact_edges[~compact_hard],
                        mapping[source.corner_vertex_ids], stats,
                        source_lineage[order])
