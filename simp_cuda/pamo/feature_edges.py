import heapq

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .segment_query import (
    closest_points_on_segments as _closest_points_on_segments,
)


def detect_reference_feature_edges(
    mesh,
    feature_edges=None,
    angle_degrees=45.0,
    include_boundaries=True,
):
    """Return original mesh edges which must be treated as feature curves."""
    if feature_edges is not None:
        edges = np.asarray(feature_edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Feature edges must have shape (n, 2).")
        if len(edges) == 0:
            raise ValueError("The feature-edge list is empty.")
        if edges.min() < 0 or edges.max() >= len(mesh.vertices):
            raise ValueError("Feature-edge indices are outside the input vertex range.")

        existing_edges = {
            tuple(edge) for edge in np.sort(mesh.edges_unique, axis=1)
        }
        missing = [
            tuple(edge)
            for edge in np.sort(edges, axis=1)
            if tuple(edge) not in existing_edges
        ]
        if missing:
            raise ValueError(
                "{} specified feature edges are not edges of the cleaned input "
                "mesh; first invalid edge: {}".format(len(missing), missing[0])
            )
        return np.unique(np.sort(edges, axis=1), axis=0)

    angle_degrees = float(angle_degrees)
    if not 0.0 < angle_degrees < 180.0:
        raise ValueError("Feature-edge angle must be between 0 and 180 degrees.")

    sharp_mask = mesh.face_adjacency_angles >= np.deg2rad(angle_degrees)
    sharp_edges = mesh.face_adjacency_edges[sharp_mask]

    if include_boundaries:
        edge_counts = np.bincount(
            mesh.edges_unique_inverse,
            minlength=len(mesh.edges_unique),
        )
        boundary_edges = mesh.edges_unique[edge_counts == 1]
        if len(boundary_edges):
            sharp_edges = np.vstack((sharp_edges, boundary_edges))

    if len(sharp_edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.sort(sharp_edges, axis=1), axis=0)


def _matched_output_feature_edges(
    vertices,
    faces,
    reference_segments,
    match_tolerance,
    output_angle_degrees,
    allow_branches=False,
):
    output_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    adjacency_angles = output_mesh.face_adjacency_angles
    sharp_mask = adjacency_angles >= np.deg2rad(output_angle_degrees)
    candidate_edges = output_mesh.face_adjacency_edges[sharp_mask]
    candidate_angles = adjacency_angles[sharp_mask]

    edge_counts = np.bincount(
        output_mesh.edges_unique_inverse,
        minlength=len(output_mesh.edges_unique),
    )
    boundary_edges = output_mesh.edges_unique[edge_counts == 1]
    if len(boundary_edges):
        candidate_edges = np.vstack((candidate_edges, boundary_edges))
        candidate_angles = np.concatenate(
            (
                candidate_angles,
                np.full(len(boundary_edges), np.pi, dtype=np.float64),
            )
        )
    if len(candidate_edges) == 0:
        return np.empty((0, 2), dtype=np.int64)

    candidate_edges = np.sort(candidate_edges, axis=1)
    unique_edges, unique_indices = np.unique(
        candidate_edges,
        axis=0,
        return_index=True,
    )
    candidate_edges = unique_edges
    candidate_angles = candidate_angles[unique_indices]

    segment_centers = reference_segments.mean(axis=1)
    tree = cKDTree(segment_centers)
    midpoints = vertices[candidate_edges].mean(axis=1)
    _, distances, segment_ids = _closest_points_on_segments(
        midpoints,
        reference_segments,
        tree,
    )
    candidate_lengths = np.linalg.norm(
        vertices[candidate_edges[:, 0]] - vertices[candidate_edges[:, 1]],
        axis=1,
    )
    # A broad voxel-based tolerance is useful for finding a feature after SDF,
    # but an edge much farther from the curve than its own length belongs to a
    # neighboring edge band rather than the feature chain itself.
    candidate_vectors = (
        vertices[candidate_edges[:, 1]]
        - vertices[candidate_edges[:, 0]]
    )
    reference_vectors = (
        reference_segments[segment_ids, 1]
        - reference_segments[segment_ids, 0]
    )
    candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
    reference_norms = np.linalg.norm(reference_vectors, axis=1)
    direction_cosine = np.divide(
        np.abs(
            np.einsum(
                "ij,ij->i",
                candidate_vectors,
                reference_vectors,
            )
        ),
        candidate_norms * reference_norms,
        out=np.zeros(len(candidate_edges), dtype=np.float64),
        where=(candidate_norms > 0.0) & (reference_norms > 0.0),
    )
    matched_mask = (
        distances
        <= np.minimum(
        match_tolerance,
        candidate_lengths * 0.5,
    )
    ) & (direction_cosine >= np.cos(np.deg2rad(30.0)))
    candidate_edges = candidate_edges[matched_mask]
    candidate_angles = candidate_angles[matched_mask]
    distances = distances[matched_mask]
    if len(candidate_edges) == 0:
        return np.empty((0, 2), dtype=np.int64)

    if allow_branches:
        return np.asarray(candidate_edges, dtype=np.int64)

    # Select the closest, sharpest non-branching chain. Without this degree
    # constraint, all sharp edges in a narrow band can be selected and then
    # project onto the same original curve.
    order = np.lexsort((-candidate_angles, distances))
    vertex_degrees = {}
    matched = []
    for index in order:
        edge = tuple(
            sorted(
                (
                    int(candidate_edges[index, 0]),
                    int(candidate_edges[index, 1]),
                )
            )
        )
        if (
            vertex_degrees.get(edge[0], 0) >= 2
            or vertex_degrees.get(edge[1], 0) >= 2
        ):
            continue
        matched.append(edge)
        vertex_degrees[edge[0]] = vertex_degrees.get(edge[0], 0) + 1
        vertex_degrees[edge[1]] = vertex_degrees.get(edge[1], 0) + 1

    if not matched:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(matched, dtype=np.int64)


def _face_edges(face):
    return (
        tuple(sorted((int(face[0]), int(face[1])))),
        tuple(sorted((int(face[1]), int(face[2])))),
        tuple(sorted((int(face[2]), int(face[0])))),
    )


def _oriented_split_faces(face, edge, midpoint_index):
    for index in range(3):
        first = int(face[index])
        second = int(face[(index + 1) % 3])
        if tuple(sorted((first, second))) == edge:
            opposite = int(face[(index + 2) % 3])
            return (
                [first, midpoint_index, opposite],
                [midpoint_index, second, opposite],
            )
    raise RuntimeError("Feature edge is missing from its incident face.")


def _triangle_area_squared(vertices, face):
    a = np.asarray(vertices[face[0]])
    b = np.asarray(vertices[face[1]])
    c = np.asarray(vertices[face[2]])
    cross = np.cross(b - a, c - a)
    return float(np.dot(cross, cross)) * 0.25


def split_feature_edges_only(
    vertices,
    faces,
    feature_edges,
    reference_segments,
    target_length,
    max_splits=100000,
):
    """
    Densify marked edges without collapse, flip, or off-curve smoothing.

    Each inserted vertex is projected onto the closest original feature
    segment. Splitting an interior manifold edge replaces its two incident
    triangles and therefore preserves mesh topology.
    """
    vertices_list = [
        np.asarray(vertex, dtype=np.float64).copy() for vertex in vertices
    ]
    faces_list = [
        [int(index) for index in face] for face in np.asarray(faces)
    ]
    active_edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(feature_edges)
    }

    edge_faces = {}
    for face_index, face in enumerate(faces_list):
        for edge in _face_edges(face):
            edge_faces.setdefault(edge, set()).add(face_index)

    segment_tree = cKDTree(reference_segments.mean(axis=1))
    heap = []
    for edge in active_edges:
        length = np.linalg.norm(
            vertices_list[edge[0]] - vertices_list[edge[1]]
        )
        heapq.heappush(heap, (-float(length), edge))

    split_count = 0
    skipped_count = 0
    area_scale = max(
        np.ptp(np.asarray(vertices), axis=0).max(),
        np.finfo(np.float64).eps,
    )
    minimum_area_squared = (area_scale * area_scale * 1e-10) ** 2
    # STL stores float32 coordinates. Use a scale-aware tolerance large
    # enough to prevent distinct inserted vertices from merging on export.
    point_tolerance = area_scale * 1e-6
    point_origin = np.min(np.asarray(vertices), axis=0)
    occupied_points = {}
    for vertex_index, vertex in enumerate(vertices_list):
        key = tuple(
            np.floor(
                (np.asarray(vertex) - point_origin) / point_tolerance
            ).astype(np.int64)
        )
        occupied_points.setdefault(key, []).append(vertex_index)

    neighbor_offsets = [
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    ]

    def point_is_occupied(point):
        key = tuple(
            np.floor(
                (point - point_origin) / point_tolerance
            ).astype(np.int64)
        )
        for offset in neighbor_offsets:
            neighbor_key = (
                key[0] + offset[0],
                key[1] + offset[1],
                key[2] + offset[2],
            )
            for vertex_index in occupied_points.get(neighbor_key, ()):
                if (
                    np.linalg.norm(point - vertices_list[vertex_index])
                    <= point_tolerance
                ):
                    return True, key
        return False, key

    while heap and split_count < max_splits:
        negative_length, edge = heapq.heappop(heap)
        if edge not in active_edges:
            continue

        current_length = np.linalg.norm(
            vertices_list[edge[0]] - vertices_list[edge[1]]
        )
        if current_length <= target_length:
            continue

        incident_faces = list(edge_faces.get(edge, ()))
        if len(incident_faces) not in (1, 2):
            active_edges.remove(edge)
            skipped_count += 1
            continue

        midpoint = (
            vertices_list[edge[0]] + vertices_list[edge[1]]
        ) * 0.5
        projected, projection_distance, _ = _closest_points_on_segments(
            midpoint[None, :],
            reference_segments,
            segment_tree,
        )
        projected = projected[0]
        if projection_distance[0] > current_length * 0.5:
            active_edges.remove(edge)
            skipped_count += 1
            continue
        occupied, point_key = point_is_occupied(projected)
        if occupied:
            active_edges.remove(edge)
            skipped_count += 1
            continue
        if (
            np.linalg.norm(projected - vertices_list[edge[0]])
            <= point_tolerance
            or np.linalg.norm(projected - vertices_list[edge[1]])
            <= point_tolerance
        ):
            active_edges.remove(edge)
            skipped_count += 1
            continue

        midpoint_index = len(vertices_list)
        vertices_list.append(projected)
        replacement_faces = []
        valid_split = True
        for face_index in incident_faces:
            replacements = _oriented_split_faces(
                faces_list[face_index],
                edge,
                midpoint_index,
            )
            if any(
                _triangle_area_squared(vertices_list, replacement)
                <= minimum_area_squared
                for replacement in replacements
            ):
                valid_split = False
                break
            replacement_faces.append((face_index, replacements))

        if not valid_split:
            vertices_list.pop()
            active_edges.remove(edge)
            skipped_count += 1
            continue

        for face_index, replacements in replacement_faces:
            old_face = faces_list[face_index]
            for old_edge in _face_edges(old_face):
                edge_faces[old_edge].discard(face_index)

            first_face, second_face = replacements
            faces_list[face_index] = first_face
            second_index = len(faces_list)
            faces_list.append(second_face)
            for new_edge in _face_edges(first_face):
                edge_faces.setdefault(new_edge, set()).add(face_index)
            for new_edge in _face_edges(second_face):
                edge_faces.setdefault(new_edge, set()).add(second_index)

        active_edges.remove(edge)
        first_edge = tuple(sorted((edge[0], midpoint_index)))
        second_edge = tuple(sorted((midpoint_index, edge[1])))
        occupied_points.setdefault(point_key, []).append(midpoint_index)
        active_edges.add(first_edge)
        active_edges.add(second_edge)
        for new_edge in (first_edge, second_edge):
            length = np.linalg.norm(
                vertices_list[new_edge[0]] - vertices_list[new_edge[1]]
            )
            heapq.heappush(heap, (-float(length), new_edge))
        split_count += 1

    result_vertices = np.asarray(vertices_list, dtype=np.float64)
    result_faces = np.asarray(faces_list, dtype=np.int64)
    final_lengths = np.asarray(
        [
            np.linalg.norm(
                result_vertices[edge[0]] - result_vertices[edge[1]]
            )
            for edge in active_edges
        ],
        dtype=np.float64,
    )
    return result_vertices, result_faces, {
        "splits": split_count,
        "skipped": skipped_count,
        "feature_edges": len(active_edges),
        "max_length": (
            float(final_lengths.max()) if len(final_lengths) else 0.0
        ),
        "hit_split_limit": bool(heap and split_count >= max_splits),
    }


def densify_remeshed_feature_edges(
    reference_mesh,
    vertices,
    faces,
    target_length,
    resolution,
    feature_edges=None,
    angle_degrees=45.0,
    match_tolerance=None,
    max_splits=100000,
):
    """Detect, match, and split remeshed edges near original feature curves."""
    target_length = float(target_length)
    if target_length <= 0.0:
        raise ValueError("Feature-edge target length must be positive.")
    max_splits = int(max_splits)
    if max_splits <= 0:
        raise ValueError("Feature-edge max splits must be positive.")

    reference_edges = detect_reference_feature_edges(
        reference_mesh,
        feature_edges=feature_edges,
        angle_degrees=angle_degrees,
        include_boundaries=True,
    )
    if len(reference_edges) == 0:
        print("Feature-edge densification: no reference feature edges found.")
        return np.asarray(vertices), np.asarray(faces)

    reference_vertices = np.asarray(reference_mesh.vertices, dtype=np.float64)
    reference_segments = reference_vertices[reference_edges]
    diameter = max(
        np.ptp(reference_vertices, axis=0).max(),
        np.finfo(np.float64).eps,
    )
    if match_tolerance is None:
        match_tolerance = diameter * 3.0 / int(resolution)
    match_tolerance = float(match_tolerance)
    if match_tolerance <= 0.0:
        raise ValueError("Feature-edge match tolerance must be positive.")

    output_angle = min(max(float(angle_degrees) * 0.5, 5.0), 45.0)
    matched_edges = _matched_output_feature_edges(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        reference_segments,
        match_tolerance,
        output_angle,
    )
    print(
        "Feature edges: {} reference, {} matched on remesh".format(
            len(reference_edges),
            len(matched_edges),
        )
    )
    if len(matched_edges) == 0:
        print(
            "Feature-edge densification: no output edge matched; "
            "the SDF surface may not contain an explicit edge chain."
        )
        return np.asarray(vertices), np.asarray(faces)

    vertices, faces, stats = split_feature_edges_only(
        vertices,
        faces,
        matched_edges,
        reference_segments,
        target_length,
        max_splits=max_splits,
    )
    print(
        "Feature-edge densification: {} splits, {} skipped, "
        "max edge length {:.6g}".format(
            stats["splits"],
            stats["skipped"],
            stats["max_length"],
        )
    )
    if stats["hit_split_limit"]:
        print(
            "Warning: feature-edge split limit reached; some matched edges "
            "remain longer than the target."
        )
    return vertices, faces
