"""CUDA topology cleanup for strict original-surface remeshing."""

import time

import numpy as np
import torch

from .surface_sample import _gpu_collapse_short_edges, _gpu_edge_topology


def cuda_available():
    """Return whether the strict constrained CUDA backend can run."""
    return bool(torch.cuda.is_available())


def _coplanar_patch_ids(
    vertices,
    faces,
    protected_edges,
    maximum_planar_angle_degrees,
):
    """Label face components joined through unprotected coplanar edges."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_count = len(faces)
    if face_count == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    face_edges = np.concatenate(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        ),
        axis=0,
    )
    face_edges.sort(axis=1)
    unique_edges, inverse, counts = np.unique(
        face_edges,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    occurrence_order = np.argsort(inverse, kind="stable")
    offsets = np.concatenate(((0,), np.cumsum(counts, dtype=np.int64)))
    occurrence_faces = np.tile(np.arange(face_count, dtype=np.int64), 3)

    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(crosses, axis=1)
    normals = np.divide(
        crosses,
        lengths[:, None],
        out=np.zeros_like(crosses),
        where=lengths[:, None] > np.finfo(np.float64).eps,
    )
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    parent = np.arange(face_count, dtype=np.int64)
    rank = np.zeros(face_count, dtype=np.int8)

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(first, second):
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    cosine_limit = np.cos(
        np.deg2rad(float(maximum_planar_angle_degrees))
    )
    for edge_index in np.flatnonzero(counts == 2):
        edge = tuple(map(int, unique_edges[edge_index]))
        if edge in protected:
            continue
        start = offsets[edge_index]
        incident = occurrence_faces[
            occurrence_order[start:start + 2]
        ]
        if (
            lengths[incident[0]] > np.finfo(np.float64).eps
            and lengths[incident[1]] > np.finfo(np.float64).eps
            and np.dot(normals[incident[0]], normals[incident[1]])
            >= cosine_limit
        ):
            union(incident[0], incident[1])

    roots = np.fromiter(
        (find(face_index) for face_index in range(face_count)),
        dtype=np.int64,
        count=face_count,
    )
    _, patch_ids = np.unique(roots, return_inverse=True)
    boundary_edges = unique_edges[counts != 2]
    return patch_ids.astype(np.int64, copy=False), boundary_edges


def _coplanar_patch_ids_cuda(
    vertices,
    faces,
    protected_edges,
    maximum_planar_angle_degrees,
):
    """Label coplanar face components without a CPU topology round trip."""
    face_count = len(faces)
    vertex_count = len(vertices)
    if face_count == 0:
        return (
            torch.empty(0, dtype=torch.long, device=faces.device),
            torch.empty((0, 2), dtype=torch.long, device=faces.device),
            torch.empty((0, 3), dtype=vertices.dtype, device=vertices.device),
            torch.empty((0, 3), dtype=vertices.dtype, device=vertices.device),
        )

    topology = _gpu_edge_topology(faces, vertex_count)
    counts = topology["counts"]
    manifold_groups = torch.nonzero(counts == 2).reshape(-1)

    triangles = vertices[faces]
    crosses = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    lengths = torch.linalg.norm(crosses, dim=1)
    epsilon = torch.finfo(vertices.dtype).eps
    normals = crosses / lengths.clamp_min(epsilon)[:, None]

    if len(manifold_groups):
        starts = topology["starts"][manifold_groups]
        first_entries = topology["order"][starts]
        second_entries = topology["order"][starts + 1]
        first_faces = topology["face_ids"][first_entries]
        second_faces = topology["face_ids"][second_entries]
        cosine_limit = float(
            np.cos(np.deg2rad(float(maximum_planar_angle_degrees)))
        )
        join = (
            (lengths[first_faces] > epsilon)
            & (lengths[second_faces] > epsilon)
            & (
                (normals[first_faces] * normals[second_faces]).sum(dim=1)
                >= cosine_limit
            )
        )

        if len(protected_edges):
            protected = torch.sort(protected_edges, dim=1).values
            protected_keys = torch.sort(
                torch.unique(
                    protected[:, 0] * vertex_count + protected[:, 1]
                )
            ).values
            manifold_edges = topology["edges"][manifold_groups]
            manifold_keys = (
                manifold_edges[:, 0] * vertex_count + manifold_edges[:, 1]
            )
            locations = torch.searchsorted(protected_keys, manifold_keys)
            clamped = locations.clamp_max(len(protected_keys) - 1)
            join &= ~(
                (locations < len(protected_keys))
                & (protected_keys[clamped] == manifold_keys)
            )

        first_faces = first_faces[join]
        second_faces = second_faces[join]
        labels = torch.arange(face_count, dtype=torch.long, device=faces.device)
        # Hook larger component roots to smaller roots, then pointer-jump.  A
        # fixed logarithmic iteration count avoids a device-to-host sync in
        # every component pass.
        component_passes = max(1, (face_count - 1).bit_length() + 1)
        for _ in range(component_passes):
            first_roots = labels[first_faces]
            second_roots = labels[second_faces]
            high = torch.maximum(first_roots, second_roots)
            low = torch.minimum(first_roots, second_roots)
            labels.scatter_reduce_(0, high, low, reduce="amin", include_self=True)
            labels = labels[labels]
        _, patch_ids = torch.unique(labels, sorted=True, return_inverse=True)
    else:
        patch_ids = torch.arange(
            face_count, dtype=torch.long, device=faces.device
        )

    return patch_ids, topology["edges"][counts != 2], triangles[:, 0], normals


def collapse_short_coplanar_edges_cuda(
    vertices,
    faces,
    protected_edges,
    maximum_short_edge_length,
    maximum_edge_length,
    maximum_planar_angle_degrees=0.1,
    passes=1,
    device=None,
):
    """Collapse strict coplanar short edges in conflict-free CUDA batches."""
    if not cuda_available():
        raise RuntimeError("CUDA constrained topology backend is unavailable.")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    protected_edges = np.asarray(
        protected_edges,
        dtype=np.int64,
    ).reshape(-1, 2)
    if device is None:
        device = torch.device("cuda")
    else:
        device = torch.device(device)

    preparation_start = time.perf_counter()
    transfer_start = time.perf_counter()
    gpu_vertices = torch.as_tensor(vertices, dtype=torch.float64, device=device)
    gpu_faces = torch.as_tensor(faces, dtype=torch.long, device=device)
    gpu_protected_edges = torch.as_tensor(
        protected_edges, dtype=torch.long, device=device
    )
    torch.cuda.synchronize(device)
    transfer_elapsed = time.perf_counter() - transfer_start

    topology_start = time.perf_counter()
    gpu_patch_ids, boundary_edges, gpu_origins, gpu_normals = (
        _coplanar_patch_ids_cuda(
            gpu_vertices,
            gpu_faces,
            gpu_protected_edges,
            maximum_planar_angle_degrees,
        )
    )
    gpu_face_ids = torch.arange(len(faces), dtype=torch.long, device=device)
    gpu_support_ids = torch.full(
        (len(vertices),),
        -1,
        dtype=torch.long,
        device=device,
    )
    gpu_protected_vertices = torch.zeros(
        len(vertices), dtype=torch.bool, device=device
    )
    if len(gpu_protected_edges):
        gpu_protected_vertices[gpu_protected_edges.reshape(-1)] = True
    if len(boundary_edges):
        gpu_protected_vertices[boundary_edges.reshape(-1)] = True
    torch.cuda.synchronize(device)
    topology_elapsed = time.perf_counter() - topology_start

    kernel_start = time.perf_counter()
    (
        gpu_vertices,
        gpu_faces,
        _,
        _,
        _,
        collapse_count,
    ) = _gpu_collapse_short_edges(
        gpu_vertices,
        gpu_faces,
        gpu_patch_ids,
        gpu_face_ids,
        gpu_support_ids,
        minimum_edge_length=float(maximum_short_edge_length),
        maximum_edge_length=float(maximum_edge_length),
        passes=int(passes),
        source_face_origins=gpu_origins,
        source_face_normals=gpu_normals,
        maximum_normal_deviation_degrees=float(
            maximum_planar_angle_degrees
        ),
        maximum_surface_deviation=max(
            float(np.ptp(vertices, axis=0).max()) * 1e-12,
            np.finfo(np.float64).eps,
        ),
        protected_vertex_mask=gpu_protected_vertices,
        strict_constraints=True,
        compact_vertices=False,
    )
    torch.cuda.synchronize(device)
    kernel_elapsed = time.perf_counter() - kernel_start
    result_faces = gpu_faces.cpu().numpy()
    result_vertices = gpu_vertices.cpu().numpy()
    total_elapsed = time.perf_counter() - preparation_start
    timings = {
        "total": total_elapsed,
        "transfer": transfer_elapsed,
        "kernel": topology_elapsed + kernel_elapsed,
        "preparation": (
            total_elapsed - transfer_elapsed - topology_elapsed - kernel_elapsed
        ),
    }
    return result_vertices, result_faces, int(collapse_count), timings
