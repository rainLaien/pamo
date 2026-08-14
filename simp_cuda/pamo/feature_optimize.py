"""Feature-constrained surface relocation and topology quality optimization."""

from dataclasses import dataclass

import igl
import numpy as np
from scipy.spatial import cKDTree

from .feature_edges import (
    _closest_points_on_segments,
    _matched_output_feature_edges,
    detect_reference_feature_edges,
)


@dataclass(frozen=True)
class FeatureConstraintMap:
    """Mapped original feature graph on a remeshed surface."""

    reference_edges: np.ndarray
    reference_segments: np.ndarray
    matched_edges: np.ndarray
    feature_vertices: np.ndarray
    feature_segment_ids: np.ndarray
    corner_vertices: np.ndarray
    corner_targets: np.ndarray
    match_tolerance: float


def _face_edges(face):
    return (
        tuple(sorted((int(face[0]), int(face[1])))),
        tuple(sorted((int(face[1]), int(face[2])))),
        tuple(sorted((int(face[2]), int(face[0])))),
    )


def _build_edge_faces(faces):
    edge_faces = {}
    for face_index, face in enumerate(faces):
        for edge in _face_edges(face):
            edge_faces.setdefault(edge, set()).add(face_index)
    return edge_faces


def _unique_edges(faces):
    if len(faces) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(
        np.sort(
            np.vstack(
                (
                    faces[:, (0, 1)],
                    faces[:, (1, 2)],
                    faces[:, (2, 0)],
                )
            ),
            axis=1,
        ),
        axis=0,
    )


def _triangle_cross_products(vertices, faces):
    triangles = vertices[faces]
    return np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )


def _triangle_quality_values(vertices, faces):
    triangles = vertices[faces]
    ab = triangles[:, 1] - triangles[:, 0]
    bc = triangles[:, 2] - triangles[:, 1]
    ca = triangles[:, 0] - triangles[:, 2]
    twice_area = np.linalg.norm(np.cross(ab, -ca), axis=1)
    squared_length_sum = (
        np.einsum("ij,ij->i", ab, ab)
        + np.einsum("ij,ij->i", bc, bc)
        + np.einsum("ij,ij->i", ca, ca)
    )
    return np.divide(
        2.0 * np.sqrt(3.0) * twice_area,
        squared_length_sum,
        out=np.zeros_like(twice_area),
        where=squared_length_sum > 0.0,
    ).clip(0.0, 1.0)


def mesh_quality_metrics(vertices, faces, edges=None):
    """Return a scale-invariant optimization energy and readable metrics."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if edges is None:
        edges = _unique_edges(faces)
    else:
        edges = np.asarray(edges, dtype=np.int64)
    lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]],
        axis=1,
    )
    mean_length = max(
        float(lengths.mean()) if len(lengths) else 0.0,
        np.finfo(np.float64).eps,
    )
    edge_cv = (
        float(lengths.std() / mean_length) if len(lengths) else 0.0
    )
    qualities = _triangle_quality_values(vertices, faces)
    shape_energy = (
        float(np.mean((1.0 - qualities) ** 2))
        if len(qualities)
        else 0.0
    )

    triangles = vertices[faces]
    angle_values = []
    for center, first, second in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        first_vector = triangles[:, first] - triangles[:, center]
        second_vector = triangles[:, second] - triangles[:, center]
        denominator = (
            np.linalg.norm(first_vector, axis=1)
            * np.linalg.norm(second_vector, axis=1)
        )
        cosine = np.divide(
            np.einsum("ij,ij->i", first_vector, second_vector),
            denominator,
            out=np.ones(len(faces), dtype=np.float64),
            where=denominator > 0.0,
        )
        angle_values.append(
            np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
        )
    minimum_angle = (
        float(np.min(np.column_stack(angle_values)))
        if len(faces)
        else 0.0
    )
    return {
        "energy": edge_cv * edge_cv + shape_energy,
        "edge_cv": edge_cv,
        "mean_triangle_quality": (
            float(qualities.mean()) if len(qualities) else 0.0
        ),
        "minimum_triangle_quality": (
            float(qualities.min()) if len(qualities) else 0.0
        ),
        "minimum_angle_degrees": minimum_angle,
    }


def _reference_feature_corners(
    reference_vertices,
    reference_edges,
    corner_angle_degrees,
):
    """Classify feature endpoints, junctions, and sharp polyline turns."""
    incident_edges = {}
    for edge_index, edge in enumerate(reference_edges):
        incident_edges.setdefault(int(edge[0]), []).append(edge_index)
        incident_edges.setdefault(int(edge[1]), []).append(edge_index)

    corners = []
    straight_cosine = -np.cos(np.deg2rad(float(corner_angle_degrees)))
    for vertex_index, edge_ids in incident_edges.items():
        if len(edge_ids) != 2:
            corners.append(vertex_index)
            continue

        directions = []
        for edge_index in edge_ids:
            edge = reference_edges[edge_index]
            other = int(edge[1]) if int(edge[0]) == vertex_index else int(edge[0])
            direction = (
                reference_vertices[other]
                - reference_vertices[vertex_index]
            )
            length = np.linalg.norm(direction)
            if length <= np.finfo(np.float64).eps:
                directions = []
                break
            directions.append(direction / length)
        if (
            len(directions) != 2
            or float(np.dot(directions[0], directions[1]))
            > straight_cosine
        ):
            corners.append(vertex_index)
    return np.asarray(sorted(set(corners)), dtype=np.int64)


def build_feature_constraint_map(
    reference_mesh,
    vertices,
    faces,
    resolution,
    feature_edges=None,
    feature_angle_degrees=30.0,
    match_tolerance=None,
    corner_angle_degrees=None,
):
    """Map original feature curves and corners onto a remeshed surface."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    reference_vertices = np.asarray(
        reference_mesh.vertices,
        dtype=np.float64,
    )
    reference_edges = detect_reference_feature_edges(
        reference_mesh,
        feature_edges=feature_edges,
        angle_degrees=feature_angle_degrees,
        include_boundaries=True,
    )
    diameter = max(
        float(np.ptp(reference_vertices, axis=0).max()),
        np.finfo(np.float64).eps,
    )
    if match_tolerance is None:
        match_tolerance = diameter * 3.0 / int(resolution)
    match_tolerance = float(match_tolerance)
    if match_tolerance <= 0.0:
        raise ValueError("Feature match tolerance must be positive.")

    if len(reference_edges) == 0:
        return FeatureConstraintMap(
            reference_edges=np.empty((0, 2), dtype=np.int64),
            reference_segments=np.empty((0, 2, 3), dtype=np.float64),
            matched_edges=np.empty((0, 2), dtype=np.int64),
            feature_vertices=np.empty(0, dtype=np.int64),
            feature_segment_ids=np.empty(0, dtype=np.int64),
            corner_vertices=np.empty(0, dtype=np.int64),
            corner_targets=np.empty((0, 3), dtype=np.float64),
            match_tolerance=match_tolerance,
        )

    reference_segments = reference_vertices[reference_edges]
    output_angle = min(
        max(float(feature_angle_degrees), 5.0),
        60.0,
    )
    matched_edges = _matched_output_feature_edges(
        vertices,
        faces,
        reference_segments,
        match_tolerance,
        output_angle,
        allow_branches=True,
    )
    if len(matched_edges):
        feature_vertices = np.unique(matched_edges)
        segment_tree = cKDTree(reference_segments.mean(axis=1))
        _, feature_distances, _ = _closest_points_on_segments(
            vertices[feature_vertices],
            reference_segments,
            segment_tree,
        )
        valid_feature_vertices = feature_vertices[
            feature_distances <= match_tolerance * 1.5
        ]
        valid_set = set(map(int, valid_feature_vertices))
        matched_edges = np.asarray(
            [
                edge
                for edge in matched_edges
                if int(edge[0]) in valid_set and int(edge[1]) in valid_set
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        feature_vertices = (
            np.unique(matched_edges)
            if len(matched_edges)
            else np.empty(0, dtype=np.int64)
        )
    else:
        feature_vertices = np.empty(0, dtype=np.int64)

    if len(feature_vertices):
        segment_tree = cKDTree(reference_segments.mean(axis=1))
        _, _, feature_segment_ids = _closest_points_on_segments(
            vertices[feature_vertices],
            reference_segments,
            segment_tree,
        )
    else:
        feature_segment_ids = np.empty(0, dtype=np.int64)

    if corner_angle_degrees is None:
        corner_angle_degrees = min(
            max(float(feature_angle_degrees), 15.0),
            60.0,
        )
    reference_corner_ids = _reference_feature_corners(
        reference_vertices,
        reference_edges,
        corner_angle_degrees,
    )

    corner_vertices = []
    corner_targets = []
    if len(reference_corner_ids):
        candidates = (
            feature_vertices
            if len(feature_vertices)
            else np.arange(len(vertices), dtype=np.int64)
        )
        candidate_tree = cKDTree(vertices[candidates])
        distances, nearest = candidate_tree.query(
            reference_vertices[reference_corner_ids],
            k=1,
        )
        order = np.argsort(distances)
        claimed_output_vertices = set()
        for index in order:
            if distances[index] > match_tolerance * 1.5:
                continue
            output_vertex = int(candidates[int(nearest[index])])
            if output_vertex in claimed_output_vertices:
                continue
            claimed_output_vertices.add(output_vertex)
            corner_vertices.append(output_vertex)
            corner_targets.append(
                reference_vertices[int(reference_corner_ids[index])]
            )

    return FeatureConstraintMap(
        reference_edges=reference_edges,
        reference_segments=reference_segments,
        matched_edges=np.asarray(matched_edges, dtype=np.int64).reshape(-1, 2),
        feature_vertices=np.asarray(feature_vertices, dtype=np.int64),
        feature_segment_ids=np.asarray(
            feature_segment_ids,
            dtype=np.int64,
        ),
        corner_vertices=np.asarray(corner_vertices, dtype=np.int64),
        corner_targets=np.asarray(corner_targets, dtype=np.float64).reshape(
            -1,
            3,
        ),
        match_tolerance=match_tolerance,
    )


def _neighbor_centroids(vertices, edges):
    sums = np.zeros_like(vertices)
    counts = np.zeros(len(vertices), dtype=np.int64)
    np.add.at(sums, edges[:, 0], vertices[edges[:, 1]])
    np.add.at(sums, edges[:, 1], vertices[edges[:, 0]])
    np.add.at(counts, edges[:, 0], 1)
    np.add.at(counts, edges[:, 1], 1)
    return np.divide(
        sums,
        counts[:, None],
        out=vertices.copy(),
        where=counts[:, None] > 0,
    )


def _vertex_normals(vertices, faces):
    face_cross = _triangle_cross_products(vertices, faces)
    normals = np.zeros_like(vertices)
    for column in range(3):
        np.add.at(normals, faces[:, column], face_cross)
    lengths = np.linalg.norm(normals, axis=1)
    return np.divide(
        normals,
        lengths[:, None],
        out=np.zeros_like(normals),
        where=lengths[:, None] > 0.0,
    )


def _locked_topology_vertices(faces, vertex_count):
    edge_faces = _build_edge_faces(faces)
    locked = np.zeros(vertex_count, dtype=bool)
    for edge, incident_faces in edge_faces.items():
        if len(incident_faces) != 2:
            locked[list(edge)] = True
    return locked


def _closest_points_on_assigned_segments(
    points,
    segments,
    segment_ids,
):
    selected = segments[np.asarray(segment_ids, dtype=np.int64)]
    starts = selected[:, 0]
    vectors = selected[:, 1] - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    parameters = np.divide(
        np.einsum("ij,ij->i", points - starts, vectors),
        lengths_squared,
        out=np.zeros(len(points), dtype=np.float64),
        where=lengths_squared > 0.0,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    return starts + parameters[:, None] * vectors


def _project_relocation(
    current,
    faces,
    reference_vertices,
    reference_faces,
    constraints,
    smoothing_step,
    mesh_edges,
    topology_locked,
    feature_segment_for_vertex,
):
    centroids = _neighbor_centroids(current, mesh_edges)
    normals = _vertex_normals(current, faces)
    displacement = centroids - current
    tangent = displacement - (
        np.einsum("ij,ij->i", displacement, normals)[:, None] * normals
    )
    raw = current + float(smoothing_step) * tangent

    feature_mask = np.zeros(len(current), dtype=bool)
    feature_mask[constraints.feature_vertices] = True
    corner_mask = np.zeros(len(current), dtype=bool)
    corner_mask[constraints.corner_vertices] = True
    if len(constraints.matched_edges):
        feature_centroids = _neighbor_centroids(
            current,
            constraints.matched_edges,
        )
        feature_displacement = feature_centroids - current
        raw[feature_mask] = (
            current[feature_mask]
            + float(smoothing_step) * feature_displacement[feature_mask]
        )

    proposed = current.copy()
    interior_mask = ~(feature_mask | topology_locked | corner_mask)
    if np.any(interior_mask):
        _, _, closest_points = igl.point_mesh_squared_distance(
            raw[interior_mask],
            reference_vertices,
            reference_faces,
        )
        proposed[interior_mask] = closest_points

    movable_feature_mask = feature_mask & ~corner_mask & ~topology_locked
    if np.any(movable_feature_mask) and len(constraints.reference_segments):
        movable_feature_ids = np.flatnonzero(movable_feature_mask)
        closest_points = _closest_points_on_assigned_segments(
            raw[movable_feature_ids],
            constraints.reference_segments,
            feature_segment_for_vertex[movable_feature_ids],
        )
        proposed[movable_feature_ids] = closest_points

    if len(constraints.corner_vertices):
        proposed[constraints.corner_vertices] = constraints.corner_targets
    # Boundary/non-manifold vertices and feature vertices whose exact snap
    # conflicts with the current triangulation remain fixed.
    proposed[topology_locked] = current[topology_locked]
    return proposed


def _valid_relocation(previous, proposed, faces, minimum_area_squared):
    old_cross = _triangle_cross_products(previous, faces)
    new_cross = _triangle_cross_products(proposed, faces)
    orientation = np.einsum("ij,ij->i", old_cross, new_cross)
    area_squared = np.einsum("ij,ij->i", new_cross, new_cross) * 0.25
    return bool(
        np.isfinite(proposed).all()
        and np.all(orientation > 0.0)
        and np.all(area_squared > minimum_area_squared)
    )


def _relocation_valid_faces(
    previous_cross,
    proposed,
    faces,
    minimum_area_squared,
    minimum_quality=None,
):
    """Return per-face validity relative to the pre-relocation orientation."""
    new_cross = _triangle_cross_products(proposed, faces)
    orientation = np.einsum("ij,ij->i", previous_cross, new_cross)
    area_squared = np.einsum("ij,ij->i", new_cross, new_cross) * 0.25
    finite_faces = np.isfinite(proposed[faces]).all(axis=(1, 2))
    valid = (
        finite_faces
        & (orientation > 0.0)
        & (area_squared > minimum_area_squared)
    )
    if minimum_quality is not None:
        valid &= _triangle_quality_values(proposed, faces) >= minimum_quality
    return valid


def _snap_to_feature_constraints(
    vertices,
    faces,
    constraints,
    minimum_area_squared,
):
    """
    Safely move mapped vertices toward their curve and corner constraints.

    A remeshed feature graph is only a geometric match to the original graph;
    its local triangles are not guaranteed to admit every exact snap. Apply the
    exact targets first, then backtrack only vertices incident to invalid faces.
    This preserves exact constraints wherever the current topology supports
    them without accepting flipped or degenerate triangles.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    targets = vertices.copy()
    if len(constraints.feature_vertices):
        closest_points = _closest_points_on_assigned_segments(
            vertices[constraints.feature_vertices],
            constraints.reference_segments,
            constraints.feature_segment_ids,
        )
        targets[constraints.feature_vertices] = closest_points
    if len(constraints.corner_vertices):
        targets[constraints.corner_vertices] = constraints.corner_targets

    constrained_vertices = np.unique(
        np.concatenate(
            (
                constraints.feature_vertices,
                constraints.corner_vertices,
            )
        )
    )
    if len(constrained_vertices) == 0:
        return vertices.copy(), constrained_vertices

    if not _valid_relocation(
        vertices,
        vertices,
        faces,
        minimum_area_squared,
    ):
        raise RuntimeError(
            "Feature optimization received a mesh that already contains "
            "flipped, degenerate, or non-finite triangles."
        )

    previous_cross = _triangle_cross_products(vertices, faces)
    # Exact feature placement must not create arbitrarily thin triangles.
    # Preserve at least half of each face's pre-snap shape quality; conflicted
    # vertices are backtracked below until this bound and orientation hold.
    minimum_quality = _triangle_quality_values(vertices, faces) * 0.5
    if np.all(
        _relocation_valid_faces(
            previous_cross,
            targets,
            faces,
            minimum_area_squared,
            minimum_quality=minimum_quality,
        )
    ):
        return targets, np.empty(0, dtype=np.int64)

    displacement = (
        targets[constrained_vertices] - vertices[constrained_vertices]
    )
    fractions = np.ones(len(constrained_vertices), dtype=np.float64)
    constrained_lookup = np.full(len(vertices), -1, dtype=np.int64)
    constrained_lookup[constrained_vertices] = np.arange(
        len(constrained_vertices),
        dtype=np.int64,
    )

    snapped = vertices.copy()
    valid_faces = np.zeros(len(faces), dtype=bool)
    for _ in range(32):
        snapped[constrained_vertices] = (
            vertices[constrained_vertices]
            + fractions[:, None] * displacement
        )
        valid_faces = _relocation_valid_faces(
            previous_cross,
            snapped,
            faces,
            minimum_area_squared,
            minimum_quality=minimum_quality,
        )
        if np.all(valid_faces):
            break

        conflicting = constrained_lookup[
            np.unique(faces[~valid_faces].reshape(-1))
        ]
        conflicting = np.unique(conflicting[conflicting >= 0])
        if len(conflicting) == 0:
            raise RuntimeError(
                "Feature snapping exposed invalid triangles which are not "
                "incident to any mapped feature vertex."
            )
        fractions[conflicting] *= 0.5
    else:
        # At zero displacement the input is known to be valid. Pin the last
        # conflicting vertices to their original positions as a deterministic
        # numerical fallback for extremely thin triangles.
        conflicting = constrained_lookup[
            np.unique(faces[~valid_faces].reshape(-1))
        ]
        conflicting = np.unique(conflicting[conflicting >= 0])
        fractions[conflicting] = 0.0
        snapped[constrained_vertices] = (
            vertices[constrained_vertices]
            + fractions[:, None] * displacement
        )
        valid_faces = _relocation_valid_faces(
            previous_cross,
            snapped,
            faces,
            minimum_area_squared,
            minimum_quality=minimum_quality,
        )
        if not np.all(valid_faces):
            fractions[:] = 0.0
            snapped = vertices.copy()

    if not _valid_relocation(
        vertices,
        snapped,
        faces,
        minimum_area_squared,
    ):
        raise RuntimeError(
            "Feature constraints could not be applied without invalidating "
            "the current mesh topology."
        )

    actually_moved = np.any(
        np.abs(displacement) > np.finfo(np.float64).eps,
        axis=1,
    )
    relaxed_vertices = constrained_vertices[
        (fractions < 1.0 - np.finfo(np.float64).eps) & actually_moved
    ]
    return snapped, relaxed_vertices


def _oriented_edge(face, edge):
    for index in range(3):
        first = int(face[index])
        second = int(face[(index + 1) % 3])
        if tuple(sorted((first, second))) == edge:
            return first, second, int(face[(index + 2) % 3])
    raise RuntimeError("Edge is missing from its incident triangle.")


def _flip_quality_edges(
    vertices,
    faces,
    protected_edges,
    passes,
    maximum_dihedral_degrees=None,
    maximum_edge_length=None,
    preferred_edges=None,
):
    """Greedily flip non-feature manifold edges when local quality improves."""
    faces = np.asarray(faces, dtype=np.int64).copy()
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    preferred = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(
            preferred_edges
            if preferred_edges is not None
            else np.empty((0, 2), dtype=np.int64),
            dtype=np.int64,
        ).reshape(-1, 2)
    }
    flip_count = 0
    diameter = max(
        float(np.ptp(vertices, axis=0).max()),
        np.finfo(np.float64).eps,
    )
    minimum_cross_squared = (diameter * diameter * 1e-12) ** 2
    if maximum_dihedral_degrees is None:
        minimum_normal_dot = None
    else:
        maximum_dihedral_degrees = float(maximum_dihedral_degrees)
        if not 0.0 <= maximum_dihedral_degrees < 180.0:
            raise ValueError("Maximum flip dihedral must be in [0, 180).")
        minimum_normal_dot = np.cos(np.deg2rad(maximum_dihedral_degrees))
    if maximum_edge_length is not None:
        maximum_edge_length = float(maximum_edge_length)
        if maximum_edge_length <= 0.0:
            raise ValueError("Maximum flipped edge length must be positive.")
        maximum_edge_length *= 1.0 + 1e-8

    for _ in range(int(passes)):
        edge_faces = _build_edge_faces(faces)
        pass_flips = 0
        for edge in sorted(edge_faces):
            if edge in protected:
                continue
            incident = edge_faces.get(edge)
            if incident is None or len(incident) != 2:
                continue
            first_face_id, second_face_id = sorted(incident)
            first_face = faces[first_face_id]
            second_face = faces[second_face_id]
            first, second, first_opposite = _oriented_edge(
                first_face,
                edge,
            )
            second_start, second_end, second_opposite = _oriented_edge(
                second_face,
                edge,
            )
            if (
                second_start != second
                or second_end != first
                or first_opposite == second_opposite
            ):
                continue

            new_edge = tuple(
                sorted((first_opposite, second_opposite))
            )
            if new_edge in edge_faces:
                continue

            new_edge_length = np.linalg.norm(
                vertices[first_opposite] - vertices[second_opposite]
            )
            if (
                maximum_edge_length is not None
                and new_edge_length > maximum_edge_length
            ):
                continue

            replacement_faces = np.asarray(
                (
                    [first_opposite, first, second_opposite],
                    [first_opposite, second_opposite, second],
                ),
                dtype=np.int64,
            )
            old_cross = _triangle_cross_products(
                vertices,
                np.asarray((first_face, second_face)),
            )
            if minimum_normal_dot is not None:
                cross_lengths = np.linalg.norm(old_cross, axis=1)
                if np.any(cross_lengths <= 0.0):
                    continue
                normal_dot = float(
                    np.dot(old_cross[0], old_cross[1])
                    / (cross_lengths[0] * cross_lengths[1])
                )
                if normal_dot < minimum_normal_dot:
                    continue
            new_cross = _triangle_cross_products(
                vertices,
                replacement_faces,
            )
            patch_normal = old_cross[0] + old_cross[1]
            if (
                np.dot(new_cross[0], patch_normal) <= 0.0
                or np.dot(new_cross[1], patch_normal) <= 0.0
                or np.dot(new_cross[0], new_cross[0])
                <= minimum_cross_squared
                or np.dot(new_cross[1], new_cross[1])
                <= minimum_cross_squared
            ):
                continue

            old_quality = _triangle_quality_values(
                vertices,
                np.asarray((first_face, second_face)),
            )
            new_quality = _triangle_quality_values(
                vertices,
                replacement_faces,
            )
            if edge in preferred:
                if float(new_quality.min()) < float(old_quality.min()) - 1e-8:
                    continue
            elif (
                float(new_quality.min())
                <= float(old_quality.min()) + 1e-8
                or float(new_quality.sum())
                < float(old_quality.sum()) - 1e-10
            ):
                continue

            for face_id in (first_face_id, second_face_id):
                for old_edge in _face_edges(faces[face_id]):
                    memberships = edge_faces.get(old_edge)
                    if memberships is not None:
                        memberships.discard(face_id)
                        if not memberships:
                            del edge_faces[old_edge]
            faces[first_face_id] = replacement_faces[0]
            faces[second_face_id] = replacement_faces[1]
            for face_id in (first_face_id, second_face_id):
                for replacement_edge in _face_edges(faces[face_id]):
                    edge_faces.setdefault(replacement_edge, set()).add(face_id)
            if edge in preferred:
                # Do not immediately undo a deliberate source-seam
                # replacement later in this optimization run.
                protected.add(new_edge)
            flip_count += 1
            pass_flips += 1
        if pass_flips == 0:
            break
    return faces, flip_count


def optimize_feature_constrained_mesh(
    reference_mesh,
    vertices,
    faces,
    resolution,
    feature_edges=None,
    feature_angle_degrees=30.0,
    match_tolerance=None,
    iterations=5,
    smoothing_step=0.2,
    flip_passes=2,
    maximum_edge_length=None,
):
    """
    Improve mesh quality while keeping mapped feature curves explicit.

    Corners target original feature vertices, feature-chain vertices target
    original feature segments, and ordinary vertices are tangentially
    relocated then projected to the original triangle surface. Constraints
    incompatible with the current local triangulation are safely backtracked
    and locked instead of creating flipped or near-degenerate faces. Only
    non-feature manifold edges may be flipped.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Feature optimization vertices must have shape (n, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Feature optimization faces must have shape (m, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError("Feature optimization vertices contain NaN or infinity.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Feature optimization face indices are out of range.")
    iterations = int(iterations)
    flip_passes = int(flip_passes)
    smoothing_step = float(smoothing_step)
    if iterations <= 0:
        raise ValueError("Feature quality iterations must be positive.")
    if flip_passes < 0:
        raise ValueError("Feature flip passes must be non-negative.")
    if not 0.0 < smoothing_step <= 1.0:
        raise ValueError("Feature quality step must be in (0, 1].")
    if maximum_edge_length is not None:
        maximum_edge_length = float(maximum_edge_length)
        if maximum_edge_length <= 0.0:
            raise ValueError("Maximum optimized edge length must be positive.")

    constraints = build_feature_constraint_map(
        reference_mesh,
        vertices,
        faces,
        resolution=resolution,
        feature_edges=feature_edges,
        feature_angle_degrees=feature_angle_degrees,
        match_tolerance=match_tolerance,
    )
    reference_vertices = np.asarray(
        reference_mesh.vertices,
        dtype=np.float64,
    )
    reference_faces = np.asarray(reference_mesh.faces, dtype=np.int64)
    current = vertices.copy()
    current_faces = faces.copy()
    mesh_edges = _unique_edges(current_faces)
    initial_metrics = mesh_quality_metrics(
        current,
        current_faces,
        edges=mesh_edges,
    )
    current_energy = initial_metrics["energy"]
    topology_locked = _locked_topology_vertices(
        current_faces,
        len(current),
    )
    feature_segment_tree = (
        cKDTree(constraints.reference_segments.mean(axis=1))
        if len(constraints.reference_segments)
        else None
    )
    feature_segment_for_vertex = np.full(
        len(current),
        -1,
        dtype=np.int64,
    )
    feature_segment_for_vertex[constraints.feature_vertices] = (
        constraints.feature_segment_ids
    )
    diameter = max(
        float(np.ptp(current, axis=0).max()),
        np.finfo(np.float64).eps,
    )
    minimum_area_squared = (diameter * diameter * 1e-12) ** 2
    current, relaxed_constraint_vertices = _snap_to_feature_constraints(
        current,
        current_faces,
        constraints,
        minimum_area_squared,
    )
    topology_locked[relaxed_constraint_vertices] = True
    constrained_metrics = mesh_quality_metrics(
        current,
        current_faces,
        edges=mesh_edges,
    )
    current_energy = constrained_metrics["energy"]
    accepted_iterations = 0

    for _ in range(iterations):
        accepted = False
        step = smoothing_step
        for _ in range(8):
            proposed = _project_relocation(
                current,
                current_faces,
                reference_vertices,
                reference_faces,
                constraints,
                step,
                mesh_edges,
                topology_locked,
                feature_segment_for_vertex,
            )
            proposed_metrics = mesh_quality_metrics(
                proposed,
                current_faces,
                edges=mesh_edges,
            )
            tolerance = (
                np.finfo(np.float64).eps
                * max(abs(current_energy), 1.0)
                * 32.0
            )
            if (
                proposed_metrics["energy"] < current_energy - tolerance
                and _valid_relocation(
                    current,
                    proposed,
                    current_faces,
                    minimum_area_squared,
                )
                and (
                    maximum_edge_length is None
                    or np.all(
                        np.linalg.norm(
                            proposed[mesh_edges[:, 0]]
                            - proposed[mesh_edges[:, 1]],
                            axis=1,
                        )
                        <= maximum_edge_length * (1.0 + 1e-8)
                    )
                )
            ):
                current = proposed
                current_energy = proposed_metrics["energy"]
                accepted = True
                accepted_iterations += 1
                break
            step *= 0.5
        if not accepted:
            break

    current_faces, flip_count = _flip_quality_edges(
        current,
        current_faces,
        constraints.matched_edges,
        flip_passes,
        maximum_edge_length=maximum_edge_length,
    )
    final_metrics = mesh_quality_metrics(current, current_faces)
    feature_distances = np.empty(0, dtype=np.float64)
    if len(constraints.feature_vertices):
        _, feature_distances, _ = _closest_points_on_segments(
            current[constraints.feature_vertices],
            constraints.reference_segments,
            feature_segment_tree,
        )
    corner_errors = (
        np.linalg.norm(
            current[constraints.corner_vertices]
            - constraints.corner_targets,
            axis=1,
        )
        if len(constraints.corner_vertices)
        else np.empty(0, dtype=np.float64)
    )
    relaxed_corner_count = len(
        np.intersect1d(
            relaxed_constraint_vertices,
            constraints.corner_vertices,
            assume_unique=True,
        )
    )
    print(
        "Feature constraints: {} reference edges, {} mapped output edges, "
        "{} constrained vertices, {} mapped corners, {} safety-relaxed "
        "vertices ({} corners)".format(
            len(constraints.reference_edges),
            len(constraints.matched_edges),
            len(constraints.feature_vertices),
            len(constraints.corner_vertices),
            len(relaxed_constraint_vertices),
            relaxed_corner_count,
        )
    )
    print(
        "Feature quality optimization: {} / {} relocations accepted, "
        "{} non-feature edges flipped".format(
            accepted_iterations,
            iterations,
            flip_count,
        )
    )
    print(
        "Feature fidelity: max curve distance {:.6g}, max corner error "
        "{:.6g}".format(
            float(feature_distances.max())
            if len(feature_distances)
            else 0.0,
            float(corner_errors.max()) if len(corner_errors) else 0.0,
        )
    )
    print(
        "Quality (input -> constrained -> optimized): edge CV "
        "{:.6g} -> {:.6g} -> {:.6g}; mean triangle quality "
        "{:.6g} -> {:.6g} -> {:.6g}; minimum angle "
        "{:.6g} -> {:.6g} -> {:.6g} degrees".format(
            initial_metrics["edge_cv"],
            constrained_metrics["edge_cv"],
            final_metrics["edge_cv"],
            initial_metrics["mean_triangle_quality"],
            constrained_metrics["mean_triangle_quality"],
            final_metrics["mean_triangle_quality"],
            initial_metrics["minimum_angle_degrees"],
            constrained_metrics["minimum_angle_degrees"],
            final_metrics["minimum_angle_degrees"],
        )
    )
    return current, current_faces, {
        "constraints": constraints,
        "initial_metrics": initial_metrics,
        "constrained_metrics": constrained_metrics,
        "final_metrics": final_metrics,
        "accepted_iterations": accepted_iterations,
        "flips": flip_count,
        "relaxed_constraint_vertices": relaxed_constraint_vertices,
        "relaxed_constraint_count": len(relaxed_constraint_vertices),
        "relaxed_corner_count": relaxed_corner_count,
        "maximum_feature_distance": (
            float(feature_distances.max())
            if len(feature_distances)
            else 0.0
        ),
        "maximum_corner_error": (
            float(corner_errors.max()) if len(corner_errors) else 0.0
        ),
    }
