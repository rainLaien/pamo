import heapq
import time

import numpy as np
import trimesh
import igl
from scipy.spatial import cKDTree, Delaunay, QhullError

try:
    import triangle as constrained_triangle
except ImportError:  # SciPy remains a conservative fallback.
    constrained_triangle = None

from .segment_query import (
    closest_points_on_segments as _closest_points_on_segments,
)
from .feature_optimize import _flip_quality_edges, _triangle_quality_values


def _ordered_cycle_from_edges(edges):
    adjacency = {}
    for first, second in edges:
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None
    start = min(adjacency)
    cycle = [start]
    previous = None
    current = start
    for _ in range(len(adjacency) - 1):
        choices = adjacency[current]
        following = choices[0] if choices[0] != previous else choices[1]
        if following == start or following in cycle:
            return None
        cycle.append(following)
        previous, current = current, following
    if start not in adjacency[current]:
        return None
    return cycle


def _ordered_cycles_from_edges(edges):
    remaining = {tuple(sorted(map(int, edge))) for edge in edges}
    cycles = []
    while remaining:
        seed = next(iter(remaining))
        component_vertices = set(seed)
        changed = True
        while changed:
            changed = False
            for edge in remaining:
                if edge[0] in component_vertices or edge[1] in component_vertices:
                    old_size = len(component_vertices)
                    component_vertices.update(edge)
                    changed = changed or len(component_vertices) != old_size
        component_edges = [
            edge for edge in remaining
            if edge[0] in component_vertices and edge[1] in component_vertices
        ]
        cycle = _ordered_cycle_from_edges(component_edges)
        if cycle is None:
            return None
        cycles.append(cycle)
        remaining.difference_update(component_edges)
    return cycles


def _points_in_polygon(points, polygon):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=np.float64)
    first = polygon
    second = np.roll(polygon, -1, axis=0)
    result = np.zeros(len(points), dtype=bool)
    for start, end in zip(first, second):
        crosses_y = (start[1] > points[:, 1]) != (end[1] > points[:, 1])
        intersection_x = (
            (end[0] - start[0]) * (points[:, 1] - start[1])
            / (end[1] - start[1] + np.finfo(np.float64).eps)
            + start[0]
        )
        result ^= crosses_y & (points[:, 0] < intersection_x)
    return result


def _polygon_signed_area(polygon):
    polygon = np.asarray(polygon, dtype=np.float64)
    following = np.roll(polygon, -1, axis=0)
    return 0.5 * float(
        np.sum(polygon[:, 0] * following[:, 1] - polygon[:, 1] * following[:, 0])
    )


def _recover_planar_constraint_edges(points, faces, required_edges):
    """Insert missing non-crossing PSLG edges by flipping intersecting edges."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).copy()
    edge_faces = _build_edge_faces(faces)
    protected = {edge for edge in required_edges if edge in edge_faces}
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    tolerance = scale * scale * 1e-14

    def orientation(first, second, third):
        first_point = points[first]
        second_point = points[second]
        third_point = points[third]
        return float(np.cross(second_point - first_point, third_point - first_point))

    for target in required_edges:
        if target in edge_faces:
            protected.add(target)
            continue
        target_first, target_second = target
        maximum_flips = max(len(faces) * 2, 100)
        for _ in range(maximum_flips):
            if target in edge_faces:
                protected.add(target)
                break
            crossing_edge = None
            for edge, memberships in edge_faces.items():
                if (
                    len(memberships) != 2
                    or edge in protected
                    or target_first in edge
                    or target_second in edge
                ):
                    continue
                first_side = orientation(target_first, target_second, edge[0])
                second_side = orientation(target_first, target_second, edge[1])
                target_side_first = orientation(edge[0], edge[1], target_first)
                target_side_second = orientation(edge[0], edge[1], target_second)
                if (
                    first_side * second_side < -tolerance
                    and target_side_first * target_side_second < -tolerance
                ):
                    crossing_edge = edge
                    break
            if crossing_edge is None:
                return faces, False
            first_face_id, second_face_id = tuple(edge_faces[crossing_edge])
            first_face = faces[first_face_id]
            second_face = faces[second_face_id]
            first_opposite = int(
                next(value for value in first_face if value not in crossing_edge)
            )
            second_opposite = int(
                next(value for value in second_face if value not in crossing_edge)
            )
            new_edge = tuple(sorted((first_opposite, second_opposite)))
            if new_edge in edge_faces:
                return faces, False
            replacement_faces = np.asarray(
                (
                    [first_opposite, second_opposite, crossing_edge[0]],
                    [second_opposite, first_opposite, crossing_edge[1]],
                ),
                dtype=np.int64,
            )
            for index in range(2):
                if orientation(*replacement_faces[index]) < 0.0:
                    replacement_faces[index, [1, 2]] = replacement_faces[
                        index, [2, 1]
                    ]
                if abs(orientation(*replacement_faces[index])) <= tolerance:
                    return faces, False
            for face_id in (first_face_id, second_face_id):
                for old_edge in _constraint_face_edges(faces[face_id]):
                    memberships = edge_faces.get(old_edge)
                    if memberships is not None:
                        memberships.discard(face_id)
                        if not memberships:
                            del edge_faces[old_edge]
            faces[first_face_id] = replacement_faces[0]
            faces[second_face_id] = replacement_faces[1]
            for face_id in (first_face_id, second_face_id):
                for replacement_edge in _constraint_face_edges(faces[face_id]):
                    edge_faces.setdefault(replacement_edge, set()).add(face_id)
        else:
            return faces, False
    return faces, True


def retriangulate_planar_annuli(
    vertices,
    faces,
    protected_edges=None,
    minimum_faces=20,
):
    """Uniformly retriangulate exact planar facets containing one or more holes."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(
            protected_edges
            if protected_edges is not None
            else np.empty((0, 2), dtype=np.int64),
            dtype=np.int64,
        ).reshape(-1, 2)
    }
    kept_faces = np.ones(len(faces), dtype=bool)
    appended_vertices = []
    appended_faces = []
    region_count = 0
    removed_face_count = 0
    old_quality_values = []
    new_quality_values = []
    candidate_count = 0
    boundary_rejection_count = 0
    quality_rejection_count = 0
    constraint_rejection_count = 0

    for facet, boundary_edges in zip(mesh.facets, mesh.facets_boundary):
        facet = np.asarray(facet, dtype=np.int64)
        if len(facet) < int(minimum_faces):
            continue
        cycles = _ordered_cycles_from_edges(boundary_edges)
        if cycles is None or len(cycles) < 2:
            continue
        candidate_count += 1
        boundary_edge_set = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in np.asarray(boundary_edges, dtype=np.int64)
        }
        facet_edge_set = {
            edge
            for face in faces[facet]
            for edge in _constraint_face_edges(face)
        }
        if (facet_edge_set & protected) - boundary_edge_set:
            constraint_rejection_count += 1
            continue

        facet_triangles = vertices[faces[facet]]
        crosses = np.cross(
            facet_triangles[:, 1] - facet_triangles[:, 0],
            facet_triangles[:, 2] - facet_triangles[:, 0],
        )
        normal = crosses.sum(axis=0)
        normal_length = np.linalg.norm(normal)
        if normal_length <= 0.0:
            continue
        normal /= normal_length
        boundary_vertex_ids = np.unique(np.asarray(boundary_edges, dtype=np.int64))
        origin = vertices[boundary_vertex_ids].mean(axis=0)
        _, _, basis_vh = np.linalg.svd(
            vertices[boundary_vertex_ids] - origin,
            full_matrices=False,
        )
        basis = basis_vh[:2]
        projected_cycles = [
            (vertices[np.asarray(cycle, dtype=np.int64)] - origin) @ basis.T
            for cycle in cycles
        ]
        areas = np.asarray(
            [abs(_polygon_signed_area(polygon)) for polygon in projected_cycles]
        )
        outer_index = int(np.argmax(areas))
        outer = projected_cycles[outer_index]
        holes = [
            polygon for index, polygon in enumerate(projected_cycles)
            if index != outer_index
        ]
        if any(not _points_in_polygon([hole.mean(axis=0)], outer)[0] for hole in holes):
            continue

        ordered_cycles = [cycles[outer_index]] + [
            cycle for index, cycle in enumerate(cycles) if index != outer_index
        ]
        polygons = [outer] + holes
        boundary_lengths = np.concatenate(
            [
                np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
                for polygon in polygons
            ]
        )
        spacing = float(np.median(boundary_lengths[boundary_lengths > 0.0]))
        if not np.isfinite(spacing) or spacing <= 0.0:
            continue

        all_boundary_2d = np.vstack(polygons)
        all_boundary_ids = np.concatenate(
            [np.asarray(cycle, dtype=np.int64) for cycle in ordered_cycles]
        )
        lower = outer.min(axis=0)
        upper = outer.max(axis=0)
        row_step = spacing * np.sqrt(3.0) * 0.5
        grid_points = []
        row = 0
        y = lower[1] + row_step
        segment_starts = np.vstack(polygons)
        segment_ends = np.vstack(
            [np.roll(polygon, -1, axis=0) for polygon in polygons]
        )
        segment_vectors = segment_ends - segment_starts
        segment_length_squared = np.maximum(
            np.sum(segment_vectors * segment_vectors, axis=1),
            np.finfo(np.float64).eps,
        )
        while y < upper[1]:
            x = lower[0] + spacing * (0.5 if row % 2 else 1.0)
            row_points = []
            while x < upper[0]:
                row_points.append((x, y))
                x += spacing
            if row_points:
                row_points = np.asarray(row_points, dtype=np.float64)
                inside = _points_in_polygon(row_points, outer)
                for hole in holes:
                    inside &= ~_points_in_polygon(row_points, hole)
                inside_points = row_points[inside]
                if len(inside_points):
                    differences = (
                        inside_points[:, None, :] - segment_starts[None, :, :]
                    )
                    fractions = np.clip(
                        np.sum(
                            differences * segment_vectors[None, :, :], axis=2
                        )
                        / segment_length_squared[None, :],
                        0.0,
                        1.0,
                    )
                    closest = (
                        segment_starts[None, :, :]
                        + fractions[:, :, None] * segment_vectors[None, :, :]
                    )
                    minimum_distances = np.sqrt(
                        np.min(
                            np.sum(
                                (inside_points[:, None, :] - closest) ** 2,
                                axis=2,
                            ),
                            axis=1,
                        )
                    )
                    grid_points.extend(
                        inside_points[minimum_distances >= spacing * 0.55]
                    )
            row += 1
            y += row_step

        interior_2d = np.asarray(grid_points, dtype=np.float64).reshape(-1, 2)
        all_2d = np.vstack((all_boundary_2d, interior_2d))
        offsets = np.cumsum([0] + [len(cycle) for cycle in ordered_cycles])
        required_edges = set()
        for start, end in zip(offsets[:-1], offsets[1:]):
            required_edges.update(
                tuple(sorted((index, start + (index - start + 1) % (end - start))))
                for index in range(start, end)
            )
        if constrained_triangle is not None:
            triangle_input = {
                "vertices": all_2d,
                "segments": np.asarray(sorted(required_edges), dtype=np.int32),
                "holes": np.asarray(
                    [hole.mean(axis=0) for hole in holes], dtype=np.float64
                ),
            }
            triangle_result = constrained_triangle.triangulate(
                triangle_input, "pQY"
            )
            if "triangles" not in triangle_result:
                boundary_rejection_count += 1
                continue
            all_2d = np.asarray(triangle_result["vertices"], dtype=np.float64)
            triangulation = np.asarray(
                triangle_result["triangles"], dtype=np.int64
            )
            interior_2d = all_2d[len(all_boundary_2d):]
        else:
            try:
                triangulation = Delaunay(all_2d).simplices.astype(np.int64)
            except QhullError:
                continue
        centroids = all_2d[triangulation].mean(axis=1)
        inside = _points_in_polygon(centroids, outer)
        for hole in holes:
            inside &= ~_points_in_polygon(centroids, hole)
        local_faces = triangulation[inside]
        if len(local_faces) == 0:
            continue

        local_edges = {
            tuple(sorted((int(first), int(second))))
            for face in local_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }
        if not required_edges.issubset(local_edges):
            boundary_rejection_count += 1
            continue

        new_3d = origin + interior_2d @ basis
        first_new_index = len(vertices) + len(appended_vertices)
        local_to_global = np.concatenate(
            (
                all_boundary_ids,
                np.arange(first_new_index, first_new_index + len(new_3d)),
            )
        )
        new_faces = local_to_global[local_faces]
        coordinate_pool = np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
                new_3d,
            )
        )
        new_triangles = coordinate_pool[new_faces]
        new_crosses = np.cross(
            new_triangles[:, 1] - new_triangles[:, 0],
            new_triangles[:, 2] - new_triangles[:, 0],
        )
        reverse = (new_crosses @ normal) < 0.0
        new_faces[reverse] = new_faces[reverse][:, [0, 2, 1]]
        old_quality = _triangle_quality_values(vertices, faces[facet])
        new_quality = _triangle_quality_values(coordinate_pool, new_faces)
        if (
            np.percentile(new_quality, 5.0)
            < np.percentile(old_quality, 5.0) - 1e-8
            or float(new_quality.mean()) < float(old_quality.mean()) - 1e-8
        ):
            quality_rejection_count += 1
            continue

        kept_faces[facet] = False
        appended_vertices.extend(new_3d)
        appended_faces.extend(new_faces)
        old_quality_values.extend(old_quality)
        new_quality_values.extend(new_quality)
        region_count += 1
        removed_face_count += len(facet)

    if not region_count:
        return vertices.copy(), faces.copy(), {
            "regions": 0, "candidates": candidate_count,
            "boundary_rejections": boundary_rejection_count,
            "quality_rejections": quality_rejection_count,
            "constraint_rejections": constraint_rejection_count,
            "new_vertices": 0, "removed_faces": 0,
            "new_faces": 0, "old_quality": 0.0, "new_quality": 0.0,
        }
    return (
        np.vstack((vertices, np.asarray(appended_vertices))),
        np.vstack((faces[kept_faces], np.asarray(appended_faces, dtype=np.int64))),
        {
            "regions": region_count,
            "candidates": candidate_count,
            "boundary_rejections": boundary_rejection_count,
            "quality_rejections": quality_rejection_count,
            "constraint_rejections": constraint_rejection_count,
            "new_vertices": len(appended_vertices),
            "removed_faces": removed_face_count,
            "new_faces": len(appended_faces),
            "old_quality": float(np.mean(old_quality_values)),
            "new_quality": float(np.mean(new_quality_values)),
        },
    )


def retriangulate_planar_fans(
    vertices,
    faces,
    protected_edges,
    minimum_valence=30,
    maximum_planar_angle_degrees=0.1,
):
    """Replace convex planar center fans by locally uniform triangulations."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    minimum_valence = int(minimum_valence)
    if minimum_valence < 6:
        raise ValueError("Planar fan valence must be at least 6.")
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    valence = np.bincount(faces.reshape(-1), minlength=len(vertices))
    candidates = np.flatnonzero(valence >= minimum_valence)
    kept_faces = np.ones(len(faces), dtype=bool)
    appended_vertices = []
    appended_faces = []
    replaced_fans = 0
    removed_faces = 0

    for center_index in candidates[np.argsort(valence[candidates])[::-1]]:
        incident_ids = np.flatnonzero(
            kept_faces & np.any(faces == center_index, axis=1)
        )
        if len(incident_ids) < minimum_valence:
            continue
        incident_faces = faces[incident_ids]
        boundary_edges = []
        for face in incident_faces:
            opposite = [int(value) for value in face if value != center_index]
            if len(opposite) != 2:
                boundary_edges = []
                break
            boundary_edges.append(tuple(opposite))
        cycle = _ordered_cycle_from_edges(boundary_edges)
        if cycle is None or len(cycle) != len(incident_faces):
            continue
        if any(
            tuple(sorted((int(center_index), boundary_vertex))) in protected
            for boundary_vertex in cycle
        ):
            continue

        triangles = vertices[incident_faces]
        crosses = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        cross_lengths = np.linalg.norm(crosses, axis=1)
        if np.any(cross_lengths <= 0.0):
            continue
        normals = crosses / cross_lengths[:, None]
        reference_normal = normals.sum(axis=0)
        reference_length = np.linalg.norm(reference_normal)
        if reference_length <= 0.0:
            continue
        reference_normal /= reference_length
        normal_dots = np.clip(normals @ reference_normal, -1.0, 1.0)
        if np.any(
            normal_dots
            < np.cos(np.deg2rad(maximum_planar_angle_degrees))
        ):
            continue

        center = vertices[center_index]
        boundary_3d = vertices[np.asarray(cycle, dtype=np.int64)]
        _, _, basis_vh = np.linalg.svd(boundary_3d - center, full_matrices=False)
        basis = basis_vh[:2]
        boundary_2d = (boundary_3d - center) @ basis.T
        signed_turns = np.cross(
            np.roll(boundary_2d, -1, axis=0) - boundary_2d,
            np.roll(boundary_2d, -2, axis=0)
            - np.roll(boundary_2d, -1, axis=0),
        )
        turn_tolerance = max(np.ptp(boundary_2d, axis=0).max(), 1.0) ** 2 * 1e-12
        nonzero_turns = signed_turns[np.abs(signed_turns) > turn_tolerance]
        if len(nonzero_turns) == 0 or np.any(nonzero_turns * nonzero_turns[0] < 0.0):
            continue
        if nonzero_turns[0] < 0.0:
            cycle.reverse()
            boundary_3d = boundary_3d[::-1]
            boundary_2d = boundary_2d[::-1]

        boundary_lengths = np.linalg.norm(
            np.roll(boundary_2d, -1, axis=0) - boundary_2d,
            axis=1,
        )
        spacing = float(np.median(boundary_lengths))
        if not np.isfinite(spacing) or spacing <= 0.0:
            continue
        lower = boundary_2d.min(axis=0)
        upper = boundary_2d.max(axis=0)
        row_step = spacing * np.sqrt(3.0) * 0.5
        grid_points = []
        row = 0
        y = lower[1] + row_step
        polygon_edges = np.roll(boundary_2d, -1, axis=0) - boundary_2d
        while y < upper[1] - row_step * 0.25:
            x = lower[0] + spacing * (0.5 if row % 2 else 1.0)
            while x < upper[0]:
                point = np.asarray((x, y))
                cross_values = np.cross(polygon_edges, point - boundary_2d)
                if np.all(cross_values >= -turn_tolerance):
                    segment_t = np.clip(
                        np.sum((point - boundary_2d) * polygon_edges, axis=1)
                        / np.maximum(
                            np.sum(polygon_edges * polygon_edges, axis=1),
                            np.finfo(np.float64).eps,
                        ),
                        0.0,
                        1.0,
                    )
                    closest = boundary_2d + segment_t[:, None] * polygon_edges
                    if np.min(np.linalg.norm(closest - point, axis=1)) >= spacing * 0.55:
                        grid_points.append(point)
                x += spacing
            row += 1
            y += row_step
        if not grid_points:
            continue

        interior_2d = np.asarray(grid_points, dtype=np.float64)
        all_2d = np.vstack((boundary_2d, interior_2d))
        try:
            local_faces = Delaunay(all_2d).simplices.astype(np.int64)
        except QhullError:
            # Leave numerically degenerate candidate patches unchanged.
            continue
        local_edges = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for face in local_faces
            for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        }
        required_boundary = {
            tuple(sorted((index, (index + 1) % len(cycle))))
            for index in range(len(cycle))
        }
        if not required_boundary.issubset(local_edges):
            continue

        new_3d = center + interior_2d @ basis
        first_new_index = len(vertices) + len(appended_vertices)
        local_to_global = np.concatenate(
            (
                np.asarray(cycle, dtype=np.int64),
                np.arange(
                    first_new_index,
                    first_new_index + len(new_3d),
                    dtype=np.int64,
                ),
            )
        )
        new_faces = local_to_global[local_faces]
        coordinate_pool = np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
                new_3d,
            )
        )
        new_triangles = coordinate_pool[new_faces]
        new_crosses = np.cross(
            new_triangles[:, 1] - new_triangles[:, 0],
            new_triangles[:, 2] - new_triangles[:, 0],
        )
        reverse = (new_crosses @ reference_normal) < 0.0
        new_faces[reverse] = new_faces[reverse][:, [0, 2, 1]]

        kept_faces[incident_ids] = False
        appended_vertices.extend(new_3d)
        appended_faces.extend(new_faces)
        replaced_fans += 1
        removed_faces += len(incident_faces)

    if not replaced_fans:
        return vertices.copy(), faces.copy(), {
            "fans": 0, "removed_faces": 0, "new_faces": 0, "new_vertices": 0
        }
    result_vertices = np.vstack((vertices, np.asarray(appended_vertices)))
    result_faces = np.vstack((faces[kept_faces], np.asarray(appended_faces, dtype=np.int64)))
    return result_vertices, result_faces, {
        "fans": replaced_fans,
        "removed_faces": removed_faces,
        "new_faces": len(appended_faces),
        "new_vertices": len(appended_vertices),
    }


def detect_hard_constraint_edges(mesh, feature_angle_degrees=5.0):
    """
    Return edges which must remain as explicit geometric edge chains.

    Every manifold edge whose adjacent-face dihedral is strictly greater than
    the threshold is hard. Boundary and non-manifold edges are also hard so
    refinement cannot change the input topology.
    """
    feature_angle_degrees = float(feature_angle_degrees)
    if not 0.0 <= feature_angle_degrees < 180.0:
        raise ValueError(
            "Constraint feature angle must be in [0, 180) degrees."
        )

    unique_edges = np.sort(
        np.asarray(mesh.edges_unique, dtype=np.int64),
        axis=1,
    )
    edge_counts = np.bincount(
        mesh.edges_unique_inverse,
        minlength=len(unique_edges),
    )
    hard = {
        tuple(edge)
        for edge in unique_edges[edge_counts != 2]
    }

    sharp = (
        np.asarray(mesh.face_adjacency_angles)
        > np.deg2rad(feature_angle_degrees)
    )
    hard.update(
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(mesh.face_adjacency_edges)[sharp]
    )
    if not hard:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(hard), dtype=np.int64)


def _constraint_face_edges(face):
    return (
        tuple(sorted((int(face[0]), int(face[1])))),
        tuple(sorted((int(face[1]), int(face[2])))),
        tuple(sorted((int(face[2]), int(face[0])))),
    )


def _constraint_split_face(face, edge, midpoint_index):
    for index in range(3):
        first = int(face[index])
        second = int(face[(index + 1) % 3])
        if tuple(sorted((first, second))) == edge:
            opposite = int(face[(index + 2) % 3])
            return (
                [first, midpoint_index, opposite],
                [midpoint_index, second, opposite],
            )
    raise RuntimeError("Split edge is missing from an incident face.")


def _build_edge_faces(faces):
    edge_faces = {}
    for face_index, face in enumerate(faces):
        for edge in _constraint_face_edges(face):
            edge_faces.setdefault(edge, set()).add(face_index)
    return edge_faces


def automatic_edge_length_limit(vertices, edge_lengths):
    """Choose 10% of the bounding-box diagonal as the maximum edge length."""
    vertices = np.asarray(vertices, dtype=np.float64)
    edge_lengths = np.asarray(edge_lengths, dtype=np.float64)
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    positive_lengths = edge_lengths[
        np.isfinite(edge_lengths) & (edge_lengths > 0.0)
    ]
    if diagonal <= 0.0 or len(positive_lengths) == 0:
        raise ValueError(
            "Automatic edge-length selection needs a non-degenerate mesh."
        )
    percentile_95 = float(np.percentile(positive_lengths, 95.0))
    # Keep the automatic rule predictable for large, already dense meshes.
    # Using the input edge distribution here can select an unnecessarily small
    # limit and trigger hundreds of thousands of conforming Python splits.
    limit = diagonal * 0.1
    return limit, diagonal, percentile_95


def automatic_split_limit(edge_lengths, max_edge_length):
    """Estimate a safe split budget from binary edge-bisection levels."""
    edge_lengths = np.asarray(edge_lengths, dtype=np.float64)
    ratios = edge_lengths[edge_lengths > max_edge_length] / max_edge_length
    if len(ratios) == 0:
        return 100000, 0
    levels = np.ceil(np.log2(ratios)).astype(np.int64)
    edge_only_estimate = int(np.sum(np.left_shift(1, levels) - 1))
    # Conforming face splits create additional long interior edges. Keep a
    # fourfold safety margin while retaining a finite runaway guard.
    return max(100000, edge_only_estimate * 4), edge_only_estimate


def refine_original_mesh_by_longest_edge(
    reference_mesh,
    max_edge_length=None,
    feature_angle_degrees=5.0,
    max_splits=None,
    coplanar_angle_degrees=0.1,
    flip_passes=8,
    flip_minimum_valence=None,
    planar_fan_minimum_valence=None,
    planar_annulus_minimum_faces=None,
):
    """
    Refine the original surface while retaining hard feature edge lineages.

    Hard edges (dihedral > threshold, boundary, or non-manifold) are tracked as
    lineages. Splitting a hard edge replaces it by two collinear hard children.
    All edges are bisected to the requested length, then quality-improving flips
    remove non-feature seams only inside coplanar patches.
    """
    vertices = np.asarray(reference_mesh.vertices, dtype=np.float64)
    faces = np.asarray(reference_mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Original-constrained refinement needs a nonempty mesh.")
    if not np.isfinite(vertices).all():
        raise ValueError("Input vertices contain NaN or infinity.")
    coplanar_angle_degrees = float(coplanar_angle_degrees)
    if not 0.0 <= coplanar_angle_degrees < 180.0:
        raise ValueError("Coplanar angle tolerance must be in [0, 180).")
    flip_passes = int(flip_passes)
    if flip_passes < 0:
        raise ValueError("Constraint flip passes must be non-negative.")

    print(
        "Constraint refinement input: {} vertices, {} faces."
        .format(len(vertices), len(faces)),
        flush=True,
    )

    hard_edges_array = detect_hard_constraint_edges(
        reference_mesh,
        feature_angle_degrees=feature_angle_degrees,
    )
    hard_edges = {
        tuple((int(edge[0]), int(edge[1])))
        for edge in hard_edges_array
    }
    coplanar_mask = (
        np.asarray(reference_mesh.face_adjacency_angles)
        <= np.deg2rad(coplanar_angle_degrees)
    )
    coplanar_edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(reference_mesh.face_adjacency_edges)[
            coplanar_mask
        ]
    }
    root_for_edge = {
        edge: root_index
        for root_index, edge in enumerate(sorted(hard_edges))
    }
    original_hard_lengths = np.asarray(
        [
            np.linalg.norm(vertices[edge[0]] - vertices[edge[1]])
            for edge in sorted(hard_edges)
        ],
        dtype=np.float64,
    )

    if planar_annulus_minimum_faces is not None:
        vertices, faces, annulus_stats = retriangulate_planar_annuli(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=planar_annulus_minimum_faces,
        )
        print(
            "Planar annulus retriangulation: {} region(s), {} new vertices, "
            "{} old -> {} new faces; mean quality {:.6g} -> {:.6g}; "
            "{} candidates ({} hard-constraint, {} boundary, {} quality "
            "rejected).".format(
                annulus_stats["regions"], annulus_stats["new_vertices"],
                annulus_stats["removed_faces"], annulus_stats["new_faces"],
                annulus_stats["old_quality"], annulus_stats["new_quality"],
                annulus_stats["candidates"],
                annulus_stats["constraint_rejections"],
                annulus_stats["boundary_rejections"],
                annulus_stats["quality_rejections"],
            ),
            flush=True,
        )

    edge_faces = _build_edge_faces(faces)
    initial_lengths = np.asarray(
        [
            np.linalg.norm(vertices[edge[0]] - vertices[edge[1]])
            for edge in edge_faces
        ],
        dtype=np.float64,
    )
    initial_max_length = (
        float(initial_lengths.max()) if len(initial_lengths) else 0.0
    )

    automatic_length = max_edge_length is None
    if automatic_length:
        (
            max_edge_length,
            bounding_box_diagonal,
            edge_length_p95,
        ) = automatic_edge_length_limit(vertices, initial_lengths)
        print(
            "Automatic maximum edge length: bounding-box diagonal 10% "
            "= {:.6g} (input edge P95 {:.6g}).".format(
                max_edge_length,
                edge_length_p95,
            )
        )
    max_edge_length = float(max_edge_length)
    if max_edge_length <= 0.0:
        raise ValueError("Constraint maximum edge length must be positive.")
    if max_splits is None:
        max_splits, edge_only_estimate = automatic_split_limit(
            initial_lengths,
            max_edge_length,
        )
        print(
            "Automatic split limit: {} (edge-only estimate {}, 4x "
            "conforming safety margin, minimum 100000).".format(
                max_splits,
                edge_only_estimate,
            )
        )
    max_splits = int(max_splits)
    if max_splits <= 0:
        raise ValueError("Constraint maximum splits must be positive.")

    relative_tolerance = 1e-8
    length_limit = max_edge_length * (1.0 + relative_tolerance)
    if initial_max_length <= length_limit:
        print(
            "Strict original constraints: current longest edge {:.6g} already "
            "satisfies the {:.6g} limit; no vertices were inserted.".format(
                initial_max_length,
                max_edge_length,
            )
        )
        optimized_vertices = vertices.copy()
        optimized_faces = faces.copy()
        fan_stats = {"fans": 0}
        if planar_fan_minimum_valence is not None:
            optimized_vertices, optimized_faces, fan_stats = (
                retriangulate_planar_fans(
                    optimized_vertices,
                    optimized_faces,
                    hard_edges_array,
                    minimum_valence=planar_fan_minimum_valence,
                    maximum_planar_angle_degrees=coplanar_angle_degrees,
                )
            )
            print(
                "Planar fan retriangulation: {} fan(s), {} new vertices, "
                "{} old -> {} new faces.".format(
                    fan_stats["fans"], fan_stats.get("new_vertices", 0),
                    fan_stats.get("removed_faces", 0),
                    fan_stats.get("new_faces", 0),
                ),
                flush=True,
            )
        optimized_faces, flip_count = _flip_quality_edges(
            optimized_vertices,
            optimized_faces,
            hard_edges_array,
            passes=flip_passes,
            maximum_dihedral_degrees=coplanar_angle_degrees,
            maximum_edge_length=max_edge_length,
            preferred_edges=np.asarray(
                sorted(coplanar_edges), dtype=np.int64
            ).reshape(-1, 2),
            minimum_candidate_valence=flip_minimum_valence,
        )
        return optimized_vertices, optimized_faces, {
            "hard_edges": len(hard_edges),
            "hard_edges_split": 0,
            "splits": 0,
            "initial_max_length": initial_max_length,
            "final_max_length": initial_max_length,
            "already_satisfied": True,
            "coplanar_flips": flip_count,
        }

    vertices_list = [vertex.copy() for vertex in vertices]
    faces_list = [list(map(int, face)) for face in faces]
    heap = []
    for edge, length in zip(edge_faces, initial_lengths):
        if length > length_limit:
            heapq.heappush(heap, (-float(length), edge))

    print(
        "Longest-edge refinement: {} initial edges exceed {:.6g}; "
        "split budget {}.".format(len(heap), max_edge_length, max_splits),
        flush=True,
    )

    split_count = 0
    split_hard_roots = set()
    split_start = time.perf_counter()
    progress_interval = 1000 if len(heap) < 20000 else 5000
    while heap and split_count < max_splits:
        negative_length, edge = heapq.heappop(heap)
        if edge not in edge_faces:
            continue
        current_length = float(
            np.linalg.norm(
                vertices_list[edge[0]] - vertices_list[edge[1]]
            )
        )
        if current_length <= length_limit:
            continue
        if current_length < -negative_length * (1.0 - 1e-12):
            continue

        incident_faces = sorted(edge_faces[edge])
        if not incident_faces:
            del edge_faces[edge]
            continue

        midpoint_index = len(vertices_list)
        midpoint = (
            vertices_list[edge[0]] + vertices_list[edge[1]]
        ) * 0.5
        vertices_list.append(midpoint)

        replacements = []
        for face_index in incident_faces:
            first_face, second_face = _constraint_split_face(
                faces_list[face_index],
                edge,
                midpoint_index,
            )
            replacements.append(
                (face_index, first_face, second_face)
            )

        for face_index, first_face, second_face in replacements:
            old_face = faces_list[face_index]
            for old_edge in _constraint_face_edges(old_face):
                incident = edge_faces.get(old_edge)
                if incident is None:
                    continue
                incident.discard(face_index)
                if not incident:
                    del edge_faces[old_edge]

            faces_list[face_index] = first_face
            second_index = len(faces_list)
            faces_list.append(second_face)
            for new_edge in _constraint_face_edges(first_face):
                edge_faces.setdefault(new_edge, set()).add(face_index)
            for new_edge in _constraint_face_edges(second_face):
                edge_faces.setdefault(new_edge, set()).add(second_index)

        first_child = tuple(sorted((edge[0], midpoint_index)))
        second_child = tuple(sorted((midpoint_index, edge[1])))
        root_index = root_for_edge.pop(edge, None)
        if root_index is not None:
            hard_edges.discard(edge)
            hard_edges.add(first_child)
            hard_edges.add(second_child)
            root_for_edge[first_child] = root_index
            root_for_edge[second_child] = root_index
            split_hard_roots.add(root_index)
        if edge in coplanar_edges:
            coplanar_edges.discard(edge)
            coplanar_edges.add(first_child)
            coplanar_edges.add(second_child)

        new_edges = set()
        for _, first_face, second_face in replacements:
            new_edges.update(_constraint_face_edges(first_face))
            new_edges.update(_constraint_face_edges(second_face))
        for new_edge in new_edges:
            new_length = float(
                np.linalg.norm(
                    vertices_list[new_edge[0]]
                    - vertices_list[new_edge[1]]
                )
            )
            if new_length > length_limit:
                heapq.heappush(heap, (-new_length, new_edge))
        split_count += 1
        if split_count % progress_interval == 0:
            elapsed = time.perf_counter() - split_start
            print(
                "  refinement progress: {} splits, {} queued candidates, "
                "{:.1f}s elapsed".format(split_count, len(heap), elapsed),
                flush=True,
            )

    remaining_long_edges = []
    final_lengths = []
    for edge in edge_faces:
        length = float(
            np.linalg.norm(
                vertices_list[edge[0]] - vertices_list[edge[1]]
            )
        )
        final_lengths.append(length)
        if length > length_limit:
            remaining_long_edges.append((edge, length))
    if remaining_long_edges:
        longest_remaining = max(length for _, length in remaining_long_edges)
        raise RuntimeError(
            "Constraint split limit {} was reached with {} edges still over "
            "the maximum length; longest remaining edge is {:.6g}. Increase "
            "max_splits or relax max_edge_length.".format(
                max_splits,
                len(remaining_long_edges),
                longest_remaining,
            )
        )

    refined_vertices = np.asarray(vertices_list, dtype=np.float64)
    refined_faces = np.asarray(faces_list, dtype=np.int64)
    if planar_fan_minimum_valence is not None:
        refined_vertices, refined_faces, fan_stats = retriangulate_planar_fans(
            refined_vertices,
            refined_faces,
            np.asarray(sorted(root_for_edge), dtype=np.int64).reshape(-1, 2),
            minimum_valence=planar_fan_minimum_valence,
            maximum_planar_angle_degrees=coplanar_angle_degrees,
        )
        print(
            "Planar fan retriangulation: {} fan(s), {} new vertices, "
            "{} old -> {} new faces.".format(
                fan_stats["fans"], fan_stats["new_vertices"],
                fan_stats["removed_faces"], fan_stats["new_faces"],
            ),
            flush=True,
        )
    if flip_passes:
        print(
            "Coplanar edge optimization: {} pass(es) over {} faces."
            .format(flip_passes, len(refined_faces)),
            flush=True,
        )
    refined_faces, flip_count = _flip_quality_edges(
        refined_vertices,
        refined_faces,
        np.asarray(sorted(root_for_edge), dtype=np.int64).reshape(-1, 2),
        passes=flip_passes,
        maximum_dihedral_degrees=coplanar_angle_degrees,
        maximum_edge_length=max_edge_length,
        preferred_edges=np.asarray(
            sorted(coplanar_edges), dtype=np.int64
        ).reshape(-1, 2),
        minimum_candidate_valence=flip_minimum_valence,
    )
    edge_faces = _build_edge_faces(refined_faces)

    hard_length_sums = np.zeros(len(original_hard_lengths), dtype=np.float64)
    for edge, root_index in root_for_edge.items():
        if edge not in edge_faces:
            raise RuntimeError(
                "A hard constraint edge disappeared during refinement: {}."
                .format(edge)
            )
        hard_length_sums[root_index] += np.linalg.norm(
            refined_vertices[edge[0]] - refined_vertices[edge[1]]
        )
    hard_tolerance = max(
        np.ptp(vertices, axis=0).max() * 1e-10,
        np.finfo(np.float64).eps,
    )
    if len(original_hard_lengths) and not np.allclose(
        hard_length_sums,
        original_hard_lengths,
        rtol=1e-10,
        atol=hard_tolerance,
    ):
        raise RuntimeError(
            "Hard constraint lineage validation failed after refinement."
        )

    final_lengths = [
        float(np.linalg.norm(refined_vertices[edge[0]] - refined_vertices[edge[1]]))
        for edge in edge_faces
    ]
    final_max_length = max(final_lengths) if final_lengths else 0.0
    print(
        "Strict original constraints: {} hard edges at > {:.6g} degrees; "
        "{} longest-edge splits ({} hard feature chains refined).".format(
            len(original_hard_lengths),
            float(feature_angle_degrees),
            split_count,
            len(split_hard_roots),
        )
    )
    print(
        "Coplanar optimization: {} quality-safe edge flips within "
        "{:.6g} degrees; hard feature edges were locked.".format(
            flip_count,
            coplanar_angle_degrees,
        )
    )
    print(
        "Global longest edge: {:.6g} -> {:.6g} (limit {:.6g}); "
        "all hard feature lineages preserved.".format(
            initial_max_length,
            final_max_length,
            max_edge_length,
        )
    )
    return (
        refined_vertices,
        refined_faces,
        {
            "hard_edges": len(original_hard_lengths),
            "hard_edges_split": len(split_hard_roots),
            "splits": split_count,
            "initial_max_length": initial_max_length,
            "final_max_length": final_max_length,
            "already_satisfied": False,
            "coplanar_flips": flip_count,
        },
    )


def remove_redundant_nested_shells(reference_mesh, vertices, faces):
    """Remove strictly nested SDF shells beyond the reference body count."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    sdf_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = trimesh.graph.connected_components(
        sdf_mesh.face_adjacency,
        nodes=np.arange(len(faces)),
        min_len=1,
    )
    reference_body_count = len(
        reference_mesh.split(only_watertight=False)
    )
    if len(components) <= reference_body_count:
        return vertices, faces

    component_data = []
    for component_index, face_ids in enumerate(components):
        vertex_ids = np.unique(faces[face_ids])
        bounds = np.vstack(
            (
                vertices[vertex_ids].min(axis=0),
                vertices[vertex_ids].max(axis=0),
            )
        )
        extent = np.maximum(bounds[1] - bounds[0], 0.0)
        component_data.append(
            {
                "index": component_index,
                "faces": np.asarray(face_ids, dtype=np.int64),
                "bounds": bounds,
                "box_volume": float(np.prod(extent)),
            }
        )

    diameter = max(
        np.ptp(vertices, axis=0).max(),
        np.finfo(np.float64).eps,
    )
    tolerance = diameter * 1e-6
    removed = set()
    smallest_first = sorted(
        component_data,
        key=lambda item: item["box_volume"],
    )
    for inner in smallest_first:
        if len(component_data) - len(removed) <= reference_body_count:
            break
        for outer in component_data:
            if inner["index"] == outer["index"]:
                continue
            inner_bounds = inner["bounds"]
            outer_bounds = outer["bounds"]
            contained = np.all(
                inner_bounds[0] >= outer_bounds[0] - tolerance
            ) and np.all(
                inner_bounds[1] <= outer_bounds[1] + tolerance
            )
            strictly_smaller = (
                inner["box_volume"]
                < outer["box_volume"] * (1.0 - 1e-6)
            )
            if contained and strictly_smaller:
                removed.add(inner["index"])
                break

    if not removed:
        return vertices, faces

    kept_face_ids = np.concatenate(
        [
            item["faces"]
            for item in component_data
            if item["index"] not in removed
        ]
    )
    kept_faces = faces[kept_face_ids]
    used_vertices, inverse = np.unique(kept_faces, return_inverse=True)
    kept_vertices = vertices[used_vertices]
    kept_faces = inverse.reshape(-1, 3)
    print(
        "Removed {} redundant nested SDF shell(s): verts {} -> {}, "
        "faces {} -> {}".format(
            len(removed),
            len(vertices),
            len(kept_vertices),
            len(faces),
            len(kept_faces),
        )
    )
    return kept_vertices, kept_faces


def classify_original_edges(
    mesh,
    coplanar_angle_degrees=0.1,
    coplanar_distance_ratio=1e-6,
):
    """
    Separate removable coplanar edges from protected original edges.

    Only manifold interior edges whose two incident triangles have matching
    normals and lie on the same plane are classified as removable. Boundary,
    non-manifold, and non-coplanar edges remain protected.
    """
    coplanar_angle_degrees = float(coplanar_angle_degrees)
    coplanar_distance_ratio = float(coplanar_distance_ratio)
    if not 0.0 <= coplanar_angle_degrees < 180.0:
        raise ValueError(
            "Coplanar angle tolerance must be in [0, 180) degrees."
        )
    if coplanar_distance_ratio < 0.0:
        raise ValueError("Coplanar distance ratio must be non-negative.")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    diameter = max(
        np.ptp(vertices, axis=0).max(),
        np.finfo(np.float64).eps,
    )
    distance_tolerance = diameter * coplanar_distance_ratio
    normal_cosine = np.cos(np.deg2rad(coplanar_angle_degrees))

    edge_counts = np.bincount(
        mesh.edges_unique_inverse,
        minlength=len(mesh.edges_unique),
    )
    manifold_interior = {
        tuple(edge)
        for edge in np.sort(
            mesh.edges_unique[edge_counts == 2],
            axis=1,
        )
    }

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    removable = []
    for edge, adjacent_faces in zip(
        mesh.face_adjacency_edges,
        mesh.face_adjacency,
    ):
        edge_key = tuple(sorted((int(edge[0]), int(edge[1]))))
        if edge_key not in manifold_interior:
            continue

        first_face = int(adjacent_faces[0])
        second_face = int(adjacent_faces[1])
        first_normal = normals[first_face]
        second_normal = normals[second_face]
        if np.dot(first_normal, second_normal) < normal_cosine:
            continue

        plane_origin = vertices[faces[first_face, 0]]
        combined_vertices = np.unique(
            np.concatenate((faces[first_face], faces[second_face]))
        )
        plane_distances = np.abs(
            (vertices[combined_vertices] - plane_origin) @ first_normal
        )
        if float(plane_distances.max()) <= distance_tolerance:
            removable.append(edge_key)

    removable_edges = (
        np.asarray(removable, dtype=np.int64).reshape(-1, 2)
        if removable
        else np.empty((0, 2), dtype=np.int64)
    )
    removable_set = {tuple(edge) for edge in removable_edges}
    protected_edges = np.asarray(
        [
            tuple(edge)
            for edge in np.sort(mesh.edges_unique, axis=1)
            if tuple(edge) not in removable_set
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    return protected_edges, removable_edges


def _face_cross_products(vertices, faces):
    triangles = vertices[faces]
    return np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )


def _adaptive_projection(vertices, faces, targets, movable_mask):
    """Project aggressively while backing off vertices on invalid faces."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    displacement = np.asarray(targets, dtype=np.float64) - vertices
    alpha = movable_mask.astype(np.float64)
    original_cross = _face_cross_products(vertices, faces)
    diameter = max(
        np.ptp(vertices, axis=0).max(),
        np.finfo(np.float64).eps,
    )
    minimum_area_squared = (diameter * diameter * 1e-14) ** 2

    proposed = vertices.copy()
    invalid = np.zeros(len(faces), dtype=bool)
    for _ in range(16):
        proposed = vertices + alpha[:, None] * displacement
        proposed_cross = _face_cross_products(proposed, faces)
        orientation = np.einsum(
            "ij,ij->i",
            original_cross,
            proposed_cross,
        )
        area_squared = (
            np.einsum("ij,ij->i", proposed_cross, proposed_cross) * 0.25
        )
        invalid = (orientation <= 0.0) | (
            area_squared <= minimum_area_squared
        )
        if not np.any(invalid):
            break

        bad_vertices = np.unique(faces[invalid])
        alpha[bad_vertices] *= 0.5
        alpha[alpha < 1.0 / 65536.0] = 0.0

    if np.any(invalid):
        bad_vertices = np.unique(faces[invalid])
        alpha[bad_vertices] = 0.0
        proposed = vertices + alpha[:, None] * displacement

    fully_projected = int(np.sum(alpha >= 1.0 - 1e-12))
    partially_projected = int(np.sum((alpha > 0.0) & (alpha < 1.0)))
    rejected = int(np.sum(movable_mask & (alpha == 0.0)))
    return proposed, {
        "fully_projected": fully_projected,
        "partially_projected": partially_projected,
        "rejected": rejected,
    }


def project_sdf_mesh_to_original_constraints(
    reference_mesh,
    vertices,
    faces,
    resolution,
    projection_distance=None,
    feature_snap_distance=None,
    coplanar_angle_degrees=0.1,
    coplanar_distance_ratio=1e-6,
):
    """
    Project SDF vertices to the original surface and protected original edges.

    Vertices farther than ``projection_distance`` remain on the SDF surface,
    preserving newly filled regions for which the original mesh has no target.
    """
    reference_vertices = np.asarray(
        reference_mesh.vertices,
        dtype=np.float64,
    )
    reference_faces = np.asarray(reference_mesh.faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    diameter = max(
        np.ptp(reference_vertices, axis=0).max(),
        np.finfo(np.float64).eps,
    )
    voxel_size = diameter / int(resolution)

    if projection_distance is None:
        projection_distance = voxel_size * 3.0
    if feature_snap_distance is None:
        feature_snap_distance = voxel_size
    projection_distance = float(projection_distance)
    feature_snap_distance = float(feature_snap_distance)
    if projection_distance <= 0.0:
        raise ValueError("Constraint projection distance must be positive.")
    if feature_snap_distance <= 0.0:
        raise ValueError("Constraint feature distance must be positive.")

    protected_edges, removable_edges = classify_original_edges(
        reference_mesh,
        coplanar_angle_degrees=coplanar_angle_degrees,
        coplanar_distance_ratio=coplanar_distance_ratio,
    )
    squared_distances, _, closest_points = igl.point_mesh_squared_distance(
        vertices,
        reference_vertices,
        reference_faces,
    )
    surface_distances = np.sqrt(
        np.maximum(np.asarray(squared_distances), 0.0)
    )
    movable_mask = surface_distances <= projection_distance
    targets = np.asarray(closest_points, dtype=np.float64)

    feature_vertex_mask = np.zeros(len(vertices), dtype=bool)
    if len(protected_edges):
        protected_segments = reference_vertices[protected_edges]
        segment_tree = cKDTree(protected_segments.mean(axis=1))
        feature_points, feature_distances, _ = (
            _closest_points_on_segments(
                vertices,
                protected_segments,
                segment_tree,
            )
        )
        feature_vertex_mask = movable_mask & (
            feature_distances <= feature_snap_distance
        )
        # Multiple SDF shells or nearby edge-band vertices can have the same
        # closest point on a protected segment. Keep the least-displaced
        # vertex for each target and leave the others on the surface target.
        target_tolerance = diameter * 1e-9
        quantized_targets = np.rint(
            (feature_points - reference_vertices.min(axis=0))
            / target_tolerance
        ).astype(np.int64)
        best_for_target = {}
        for vertex_index in np.flatnonzero(feature_vertex_mask):
            key = tuple(quantized_targets[vertex_index])
            displacement = feature_distances[vertex_index]
            previous = best_for_target.get(key)
            if previous is None or displacement < previous[0]:
                if previous is not None:
                    feature_vertex_mask[previous[1]] = False
                best_for_target[key] = (displacement, vertex_index)
            else:
                feature_vertex_mask[vertex_index] = False
        targets[feature_vertex_mask] = feature_points[feature_vertex_mask]

    projected, projection_stats = _adaptive_projection(
        vertices,
        faces,
        targets,
        movable_mask,
    )
    print(
        "Original constraints: {} protected edges, {} strictly coplanar "
        "removable edges".format(
            len(protected_edges),
            len(removable_edges),
        )
    )
    print(
        "Original projection: {} near-surface vertices, {} snapped to "
        "protected edges, {} new-fill vertices kept on SDF".format(
            int(movable_mask.sum()),
            int(feature_vertex_mask.sum()),
            int((~movable_mask).sum()),
        )
    )
    print(
        "Projection acceptance: {} full, {} partial, {} rejected".format(
            projection_stats["fully_projected"],
            projection_stats["partially_projected"],
            projection_stats["rejected"],
        )
    )
    return projected, faces, protected_edges
