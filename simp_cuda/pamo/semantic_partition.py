"""Geometry-aware mesh oversegmentation using curvature-change cues."""

from dataclasses import dataclass
import colorsys
import time

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class SemanticPartitionResult:
    labels: np.ndarray
    region_types: tuple
    boundary_edges: np.ndarray
    boundary_scores: np.ndarray
    face_normal_rates: np.ndarray
    stats: dict


def _compact_labels(labels):
    _, compact = np.unique(np.asarray(labels, dtype=np.int64), return_inverse=True)
    return compact.astype(np.int64, copy=False)


def _merge_small_regions(labels, adjacency, scores, crease, minimum_faces):
    """Merge tiny noise islands across their least significant soft boundary."""
    labels = _compact_labels(labels)
    minimum_faces = max(int(minimum_faces), 1)
    for _ in range(16):
        sizes = np.bincount(labels)
        small_mask = sizes < minimum_faces
        if not np.any(small_mask):
            break
        first = labels[adjacency[:, 0]]
        second = labels[adjacency[:, 1]]
        cross = (first != second) & ~crease
        edge_ids = np.flatnonzero(cross)
        if len(edge_ids) == 0:
            break
        first = first[edge_ids]
        second = second[edge_ids]
        edge_scores = scores[edge_ids]
        first_small = small_mask[first]
        second_small = small_mask[second]
        sources = np.concatenate((first[first_small], second[second_small]))
        targets = np.concatenate((second[first_small], first[second_small]))
        candidate_scores = np.concatenate(
            (edge_scores[first_small], edge_scores[second_small])
        )
        # Prefer merging a small island into an equal or larger neighbor. This
        # prevents simultaneous small-small swaps while retaining thin real
        # features surrounded only by strong creases.
        stable = (sizes[targets] > sizes[sources]) | (
            (sizes[targets] == sizes[sources]) & (targets < sources)
        )
        sources = sources[stable]
        targets = targets[stable]
        candidate_scores = candidate_scores[stable]
        if len(sources) == 0:
            break
        pair_keys = sources * len(sizes) + targets
        order = np.argsort(pair_keys, kind="stable")
        pair_keys = pair_keys[order]
        ordered_scores = candidate_scores[order]
        starts = np.flatnonzero(
            np.concatenate(([True], pair_keys[1:] != pair_keys[:-1]))
        )
        counts = np.diff(np.append(starts, len(pair_keys)))
        mean_scores = np.add.reduceat(ordered_scores, starts) / counts
        pair_sources = pair_keys[starts] // len(sizes)
        pair_targets = pair_keys[starts] % len(sizes)
        choice_order = np.lexsort((pair_targets, mean_scores, pair_sources))
        ordered_sources = pair_sources[choice_order]
        first_choice = np.concatenate(
            ([True], ordered_sources[1:] != ordered_sources[:-1])
        )
        chosen = choice_order[first_choice]
        mapping = np.arange(len(sizes), dtype=np.int64)
        mapping[pair_sources[chosen]] = pair_targets[chosen]
        changed = np.any(mapping != np.arange(len(sizes)))
        if not changed:
            break
        labels = _compact_labels(mapping[labels])
    return labels


def _classify_region(vertices, faces, face_ids, scale):
    """Fit inexpensive plane/cylinder hypotheses to one face region."""
    face_ids = np.asarray(face_ids, dtype=np.int64)
    region_faces = faces[face_ids]
    vertex_ids = np.unique(region_faces)
    points = vertices[vertex_ids]
    if len(points) < 3:
        return "small"

    triangles = vertices[region_faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(crosses, axis=1)
    valid = lengths > np.finfo(np.float64).eps
    normals = crosses[valid] / lengths[valid, None]
    if len(normals):
        reference = normals[0]
        aligned = np.where(
            (normals @ reference)[:, None] < 0.0, -normals, normals
        )
        mean_normal = aligned.mean(axis=0)
        mean_length = np.linalg.norm(mean_normal)
        if mean_length > np.finfo(np.float64).eps:
            mean_normal /= mean_length
            normal_angles = np.arccos(
                np.clip(aligned @ mean_normal, -1.0, 1.0)
            )
            if np.rad2deg(np.sqrt(np.mean(normal_angles ** 2))) <= 3.0:
                return "plane"

    origin = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
    plane_residual = np.sqrt(np.mean(((points - origin) @ vh[-1]) ** 2))
    if plane_residual <= max(float(scale) * 1e-5, 1e-12):
        return "plane"
    if len(face_ids) < 8:
        return "freeform"

    if np.count_nonzero(valid) < 4:
        return "freeform"
    normal_covariance = normals.T @ normals
    _, normal_vectors = np.linalg.eigh(normal_covariance)
    axis = normal_vectors[:, 0]
    axial_normal_rms = float(np.sqrt(np.mean((normals @ axis) ** 2)))
    if axial_normal_rms > 0.15:
        return "freeform"

    basis_u = vh[0] - float(np.dot(vh[0], axis)) * axis
    basis_u_length = np.linalg.norm(basis_u)
    if basis_u_length <= np.finfo(np.float64).eps:
        basis_u = np.cross(axis, (1.0, 0.0, 0.0))
        basis_u_length = np.linalg.norm(basis_u)
        if basis_u_length <= np.finfo(np.float64).eps:
            basis_u = np.cross(axis, (0.0, 1.0, 0.0))
            basis_u_length = np.linalg.norm(basis_u)
    basis_u /= basis_u_length
    basis_v = np.cross(axis, basis_u)
    projected = np.column_stack(((points - origin) @ basis_u, (points - origin) @ basis_v))
    system = np.column_stack((2.0 * projected, np.ones(len(projected))))
    rhs = np.einsum("ij,ij->i", projected, projected)
    circle, _, _, _ = np.linalg.lstsq(system, rhs, rcond=None)
    radius_squared = float(circle[2] + circle[0] ** 2 + circle[1] ** 2)
    if radius_squared <= 0.0:
        return "freeform"
    radius = np.sqrt(radius_squared)
    radial = np.linalg.norm(projected - circle[:2], axis=1)
    radial_rms = float(np.sqrt(np.mean((radial - radius) ** 2)))
    return "cylinder" if radial_rms / radius <= 3e-2 else "freeform"


def build_semantic_partitions(
    vertices,
    faces,
    feature_angle_degrees=15.0,
    curvature_gradient_degrees=1.0,
    minimum_region_faces=20,
    curvature_smoothing_iterations=2,
    minimum_gradient_chain_edges=4,
):
    """Partition faces using creases plus changes in discrete normal rate.

    A constant-radius fillet has a roughly constant normal rotation per unit
    distance.  Its tangent contact with a plane can have a tiny dihedral angle,
    but the normal-rate change from approximately ``1/r`` to zero remains a
    strong boundary cue.
    """
    started = time.perf_counter()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Semantic partitioning requires a nonempty mesh.")
    if not np.isfinite(vertices).all():
        raise ValueError("Semantic partition vertices contain NaN or infinity.")
    if not 0.0 <= float(feature_angle_degrees) < 180.0:
        raise ValueError("Feature angle must be in [0, 180) degrees.")
    if float(curvature_gradient_degrees) <= 0.0:
        raise ValueError("Curvature-gradient threshold must be positive.")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64), axis=1
    )
    if len(adjacency) == 0:
        labels = np.arange(len(faces), dtype=np.int64)
        return SemanticPartitionResult(
            labels=labels,
            region_types=tuple("small" for _ in labels),
            boundary_edges=np.empty((0, 2), dtype=np.int64),
            boundary_scores=np.empty(0, dtype=np.float64),
            face_normal_rates=np.zeros(len(faces), dtype=np.float64),
            stats={"partitions": len(faces), "elapsed": time.perf_counter() - started},
        )

    angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    dual_lengths = np.linalg.norm(
        centroids[adjacency[:, 0]] - centroids[adjacency[:, 1]], axis=1
    )
    scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    dual_lengths = np.maximum(dual_lengths, scale * 1e-12)
    normal_rates = angles / dual_lengths

    face_rate_sum = np.zeros(len(faces), dtype=np.float64)
    face_degree = np.zeros(len(faces), dtype=np.int64)
    np.add.at(face_rate_sum, adjacency[:, 0], normal_rates)
    np.add.at(face_rate_sum, adjacency[:, 1], normal_rates)
    np.add.at(face_degree, adjacency[:, 0], 1)
    np.add.at(face_degree, adjacency[:, 1], 1)
    face_rates = np.divide(
        face_rate_sum,
        face_degree,
        out=np.zeros_like(face_rate_sum),
        where=face_degree > 0,
    )
    for _ in range(max(int(curvature_smoothing_iterations), 0)):
        smoothed_sum = face_rates.copy()
        smoothed_weight = np.ones(len(faces), dtype=np.float64)
        np.add.at(smoothed_sum, adjacency[:, 0], face_rates[adjacency[:, 1]])
        np.add.at(smoothed_sum, adjacency[:, 1], face_rates[adjacency[:, 0]])
        np.add.at(smoothed_weight, adjacency[:, 0], 1.0)
        np.add.at(smoothed_weight, adjacency[:, 1], 1.0)
        face_rates = smoothed_sum / smoothed_weight

    gradient_equivalent = np.rad2deg(
        np.abs(face_rates[adjacency[:, 0]] - face_rates[adjacency[:, 1]])
        * dual_lengths
    )
    angle_degrees = np.rad2deg(angles)
    crease = angle_degrees > float(feature_angle_degrees)
    gradient_cut = gradient_equivalent > float(curvature_gradient_degrees)
    if np.any(gradient_cut) and int(minimum_gradient_chain_edges) > 1:
        candidate_edges = adjacency_edges[gradient_cut]
        candidate_graph = coo_matrix(
            (
                np.ones(len(candidate_edges) * 2, dtype=np.uint8),
                (
                    np.concatenate((candidate_edges[:, 0], candidate_edges[:, 1])),
                    np.concatenate((candidate_edges[:, 1], candidate_edges[:, 0])),
                ),
            ),
            shape=(len(vertices), len(vertices)),
        ).tocsr()
        _, vertex_components = connected_components(
            candidate_graph, directed=False
        )
        edge_components = vertex_components[candidate_edges[:, 0]]
        component_sizes = np.bincount(edge_components)
        coherent = (
            component_sizes[edge_components] >= int(minimum_gradient_chain_edges)
        )
        retained = np.zeros_like(gradient_cut)
        retained[np.flatnonzero(gradient_cut)[coherent]] = True
        gradient_cut = retained
    boundary_scores = np.maximum(
        angle_degrees / max(float(feature_angle_degrees), 1e-12),
        gradient_equivalent / float(curvature_gradient_degrees),
    )
    cut = crease | gradient_cut

    usable = adjacency[~cut]
    graph = coo_matrix(
        (
            np.ones(len(usable) * 2, dtype=np.uint8),
            (
                np.concatenate((usable[:, 0], usable[:, 1])),
                np.concatenate((usable[:, 1], usable[:, 0])),
            ),
        ),
        shape=(len(faces), len(faces)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    labels = _merge_small_regions(
        labels,
        adjacency,
        boundary_scores,
        angle_degrees
        > max(45.0, float(feature_angle_degrees) * 2.0),
        minimum_region_faces,
    )

    crossing = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
    boundary_edges = adjacency_edges[crossing]
    region_count = int(labels.max()) + 1
    region_sizes = np.bincount(labels, minlength=region_count)
    face_order = np.argsort(labels, kind="stable")
    region_faces = np.split(
        face_order,
        np.cumsum(region_sizes[:-1], dtype=np.int64),
    )
    def classify(region):
        return _classify_region(
            vertices,
            faces,
            region_faces[region],
            scale,
        )

    classification_started = time.perf_counter()
    region_types = tuple(classify(region) for region in range(region_count))
    classification_elapsed = time.perf_counter() - classification_started
    type_counts = {
        kind: int(sum(item == kind for item in region_types))
        for kind in sorted(set(region_types))
    }
    stats = {
        "partitions": region_count,
        "largest_partition_faces": int(region_sizes.max()),
        "median_partition_faces": float(np.median(region_sizes)),
        "crease_edges": int(np.count_nonzero(crease)),
        "gradient_edges": int(np.count_nonzero(gradient_cut & ~crease)),
        "boundary_edges": int(len(boundary_edges)),
        "region_types": type_counts,
        "classification_elapsed": classification_elapsed,
        "elapsed": time.perf_counter() - started,
    }
    return SemanticPartitionResult(
        labels=labels,
        region_types=region_types,
        boundary_edges=boundary_edges,
        boundary_scores=boundary_scores[crossing],
        face_normal_rates=face_rates,
        stats=stats,
    )


def export_semantic_partition_ply(path, vertices, faces, **kwargs):
    """Export an input mesh colored by geometry-aware partition labels."""
    result = build_semantic_partitions(vertices, faces, **kwargs)
    region_count = result.stats["partitions"]
    palette = np.empty((region_count, 4), dtype=np.uint8)
    golden_ratio = 0.6180339887498949
    for region in range(region_count):
        hue = (0.08 + region * golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
        palette[region, :3] = np.rint(np.asarray(rgb) * 255.0).astype(np.uint8)
        palette[region, 3] = 255
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = palette[result.labels]
    mesh.export(path)
    return result
