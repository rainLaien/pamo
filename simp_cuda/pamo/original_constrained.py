import heapq
import colorsys
import time
from collections import deque

import numpy as np
import trimesh
import igl
from scipy.spatial import cKDTree, Delaunay, QhullError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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
    ordered_edges = list(remaining)
    vertex_edges = {}
    for edge_index, (first, second) in enumerate(ordered_edges):
        vertex_edges.setdefault(first, []).append(edge_index)
        vertex_edges.setdefault(second, []).append(edge_index)
    unvisited_edges = set(range(len(ordered_edges)))
    cycles = []
    while unvisited_edges:
        seed_index = min(unvisited_edges)
        component_edge_indices = set()
        pending_vertices = list(ordered_edges[seed_index])
        visited_vertices = set()
        while pending_vertices:
            vertex = pending_vertices.pop()
            if vertex in visited_vertices:
                continue
            visited_vertices.add(vertex)
            for edge_index in vertex_edges[vertex]:
                if edge_index not in unvisited_edges:
                    continue
                component_edge_indices.add(edge_index)
                pending_vertices.extend(ordered_edges[edge_index])
        component_edges = [
            ordered_edges[index] for index in sorted(component_edge_indices)
        ]
        cycle = _ordered_cycle_from_edges(component_edges)
        if cycle is None:
            return None
        cycles.append(cycle)
        unvisited_edges.difference_update(component_edge_indices)
    return cycles


def _points_in_polygon(points, polygon):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=np.float64)
    first = polygon
    second = np.roll(polygon, -1, axis=0)
    result = np.zeros(len(points), dtype=bool)
    if len(points) == 0 or len(polygon) == 0:
        return result
    # Vectorize ray crossings in bounded-memory edge chunks. The former Python
    # loop is called hundreds of times while seeding planar triangulations.
    chunk_size = max(1, min(len(polygon), 1_000_000 // max(len(points), 1)))
    point_x = points[:, 0, None]
    point_y = points[:, 1, None]
    epsilon = np.finfo(np.float64).eps
    for start_index in range(0, len(polygon), chunk_size):
        end_index = min(start_index + chunk_size, len(polygon))
        starts = first[start_index:end_index]
        ends = second[start_index:end_index]
        delta_y = ends[:, 1] - starts[:, 1]
        valid = np.abs(delta_y) > epsilon
        if not np.any(valid):
            continue
        starts = starts[valid]
        ends = ends[valid]
        delta_y = delta_y[valid]
        crosses_y = (starts[None, :, 1] > point_y) != (
            ends[None, :, 1] > point_y
        )
        intersection_x = (
            (ends - starts)[None, :, 0]
            * (point_y - starts[None, :, 1])
            / delta_y[None, :]
            + starts[None, :, 0]
        )
        result ^= np.logical_xor.reduce(
            crosses_y & (point_x < intersection_x), axis=1
        )
    return result


def _edge_rows_in_set(edges, edge_set):
    """Vectorized membership for sorted integer edge rows."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0 or not edge_set:
        return np.zeros(len(edges), dtype=bool)
    targets = np.asarray(list(edge_set), dtype=np.int64).reshape(-1, 2)
    maximum_vertex = int(max(edges.max(), targets.max()))
    base = np.int64(maximum_vertex + 1)
    edge_keys = edges[:, 0] * base + edges[:, 1]
    target_keys = targets[:, 0] * base + targets[:, 1]
    return np.isin(edge_keys, target_keys)


def _faces_touching_edge_set(faces, edge_set):
    """Mark faces incident to any edge in an integer edge set."""
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) == 0 or not edge_set:
        return np.zeros(len(faces), dtype=bool)
    face_edges = np.sort(
        faces[:, ((0, 1), (1, 2), (2, 0))], axis=2
    ).reshape(-1, 2)
    return _edge_rows_in_set(face_edges, edge_set).reshape(-1, 3).any(axis=1)


def _planar_facets_with_boundaries(mesh):
    """Return Trimesh planar facets with all boundaries grouped in one pass."""
    facets = [np.asarray(facet, dtype=np.int64) for facet in mesh.facets]
    if not facets:
        return facets, []

    face_to_facet = np.full(len(mesh.faces), -1, dtype=np.int64)
    for facet_id, facet in enumerate(facets):
        face_to_facet[facet] = facet_id

    face_edges = np.sort(
        np.asarray(mesh.faces, dtype=np.int64)[
            :, ((0, 1), (1, 2), (2, 0))
        ],
        axis=2,
    ).reshape(-1, 2)
    occurrence_facets = np.repeat(face_to_facet, 3)
    valid = occurrence_facets >= 0
    occurrence_facets = occurrence_facets[valid]
    face_edges = face_edges[valid]

    # Sort once by (facet, edge), then keep edges occurring exactly once in a
    # facet. Trimesh otherwise hashes the edges separately for every facet.
    vertex_count = max(len(mesh.vertices), 1)
    edge_keys = face_edges[:, 0] * vertex_count + face_edges[:, 1]
    order = np.lexsort((edge_keys, occurrence_facets))
    sorted_facets = occurrence_facets[order]
    sorted_keys = edge_keys[order]
    group_start_mask = np.ones(len(order), dtype=bool)
    group_start_mask[1:] = (
        (sorted_facets[1:] != sorted_facets[:-1])
        | (sorted_keys[1:] != sorted_keys[:-1])
    )
    starts = np.flatnonzero(group_start_mask)
    counts = np.diff(np.append(starts, len(order)))
    boundary_starts = starts[counts == 1]
    boundary_facets = sorted_facets[boundary_starts]
    boundary_edges = face_edges[order[boundary_starts]]

    facet_boundary_counts = np.bincount(
        boundary_facets, minlength=len(facets)
    )
    offsets = np.cumsum(facet_boundary_counts[:-1], dtype=np.int64)
    return facets, np.split(boundary_edges, offsets)


def _polygon_signed_area(polygon):
    polygon = np.asarray(polygon, dtype=np.float64)
    following = np.roll(polygon, -1, axis=0)
    return 0.5 * float(
        np.sum(polygon[:, 0] * following[:, 1] - polygon[:, 1] * following[:, 0])
    )


def _polygon_interior_point(polygon):
    """Return a point verified inside a possibly concave simple polygon."""
    polygon = np.asarray(polygon, dtype=np.float64)
    mean_point = polygon.mean(axis=0)
    if _points_in_polygon(mean_point[None, :], polygon)[0]:
        return mean_point
    try:
        simplices = Delaunay(polygon).simplices
    except QhullError:
        return None
    triangles = polygon[simplices]
    centroids = triangles.mean(axis=1)
    inside = _points_in_polygon(centroids, polygon)
    if not np.any(inside):
        return None
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    candidate_ids = np.flatnonzero(inside)
    return centroids[candidate_ids[np.argmax(np.abs(crosses[candidate_ids]))]]


def _uniform_target_spacing(
    measured_spacing, preferred_edge_length, gradual=True
):
    """Move generated interiors toward one target size."""
    measured_spacing = float(measured_spacing)
    if preferred_edge_length is None:
        return measured_spacing
    preferred_edge_length = float(preferred_edge_length)
    if not np.isfinite(preferred_edge_length) or preferred_edge_length <= 0.0:
        raise ValueError("Preferred remeshing edge length must be positive.")
    if not gradual:
        return preferred_edge_length
    gradual_lower_bound = min(
        preferred_edge_length * 0.8,
        measured_spacing * 1.25,
    )
    return min(max(measured_spacing, gradual_lower_bound), preferred_edge_length)


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
    minimum_holes=1,
    maximum_holes=None,
    maximum_target_edge_length=None,
    maximum_result_edge_length=None,
    minimum_angle_degrees=None,
    accept_strong_mean_gain=False,
    skip_satisfactory_quality=False,
    minimum_region_area=None,
    gradual_target_spacing=True,
    required_boundary_vertex_range=None,
    required_boundary_vertex_ids=None,
    minimum_boundary_length_ratio=None,
):
    """Uniformly retriangulate exact planar facets with constrained boundaries."""
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
    density_forced_count = 0
    density_simplification_count = 0

    planar_facets, planar_boundaries = _planar_facets_with_boundaries(mesh)
    for facet, boundary_edges in zip(planar_facets, planar_boundaries):
        facet = np.asarray(facet, dtype=np.int64)
        if len(facet) < int(minimum_faces):
            continue
        facet_triangles = vertices[faces[facet]]
        facet_area = 0.5 * float(
            np.linalg.norm(
                np.cross(
                    facet_triangles[:, 1] - facet_triangles[:, 0],
                    facet_triangles[:, 2] - facet_triangles[:, 0],
                ),
                axis=1,
            ).sum()
        )
        if (
            minimum_region_area is not None
            and facet_area < float(minimum_region_area)
        ):
            continue
        cycles = _ordered_cycles_from_edges(boundary_edges)
        if cycles is None:
            continue
        hole_count = len(cycles) - 1
        if hole_count < int(minimum_holes):
            continue
        if maximum_holes is not None and hole_count > int(maximum_holes):
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
        hole_points = [_polygon_interior_point(hole) for hole in holes]
        if any(point is None for point in hole_points):
            continue
        if any(
            not _points_in_polygon(point[None, :], outer)[0]
            for point in hole_points
        ):
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
        if (
            required_boundary_vertex_range is not None
            or required_boundary_vertex_ids is not None
        ):
            boundary_matches = False
            if required_boundary_vertex_range is not None:
                required_start, required_end = map(
                    int, required_boundary_vertex_range
                )
                boundary_matches |= bool(
                    np.any(
                        (boundary_vertex_ids >= required_start)
                        & (boundary_vertex_ids < required_end)
                    )
                )
            if required_boundary_vertex_ids is not None:
                required_ids = np.asarray(
                    required_boundary_vertex_ids, dtype=np.int64
                ).reshape(-1)
                boundary_matches |= bool(
                    np.any(np.isin(boundary_vertex_ids, required_ids))
                )
            if not boundary_matches:
                continue
        if minimum_boundary_length_ratio is not None:
            positive_boundary_lengths = boundary_lengths[
                boundary_lengths > 0.0
            ]
            if len(positive_boundary_lengths) == 0:
                continue
            fine_boundary_length = float(
                np.percentile(positive_boundary_lengths, 25.0)
            )
            if float(positive_boundary_lengths.max()) < (
                fine_boundary_length * float(minimum_boundary_length_ratio)
            ):
                continue
        if (
            maximum_result_edge_length is not None
            and len(boundary_lengths)
            and float(boundary_lengths.max())
            > float(maximum_result_edge_length) * (1.0 + 1e-8)
        ):
            boundary_rejection_count += 1
            continue
        old_quality = _triangle_quality_values(vertices, faces[facet])
        target_face_count = float(len(facet))
        if maximum_target_edge_length is not None:
            target_triangle_area = (
                np.sqrt(3.0)
                * float(maximum_target_edge_length) ** 2
                * 0.25
            )
            target_face_count = max(
                facet_area / max(target_triangle_area, 1e-30),
                float(max(len(boundary_vertex_ids) - 2, 1)),
            )
        density_excess = len(facet) > target_face_count * 2.0
        if (
            skip_satisfactory_quality
            and float(old_quality.mean()) >= 0.75
            and float(np.percentile(old_quality, 5.0)) >= 0.25
            and not density_excess
        ):
            continue
        if density_excess:
            density_forced_count += 1
        spacing = float(np.median(boundary_lengths[boundary_lengths > 0.0]))
        if not np.isfinite(spacing) or spacing <= 0.0:
            continue
        domain_area = max(
            float(areas[outer_index])
            - float(
                sum(
                    areas[index]
                    for index in range(len(areas))
                    if index != outer_index
                )
            ),
            0.0,
        )
        maximum_interior_points = max(len(facet) * 8, 100)
        density_spacing = np.sqrt(
            domain_area
            / max(
                (np.sqrt(3.0) * 0.5) * maximum_interior_points,
                np.finfo(np.float64).eps,
            )
        )
        spacing = max(spacing, density_spacing)
        spacing = _uniform_target_spacing(
            spacing,
            maximum_target_edge_length,
            gradual=gradual_target_spacing,
        )
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
                    # With a direct coarse interior target, keep the first
                    # grid ring close to a dense immutable boundary. This
                    # localizes the size transition and avoids a long empty
                    # moat whose triangles would violate the edge limit.
                    clearance_ratio = (
                        0.3
                        if maximum_target_edge_length is not None
                        and not gradual_target_spacing
                        else 0.55
                    )
                    grid_points.extend(
                        inside_points[
                            minimum_distances >= spacing * clearance_ratio
                        ]
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
            }
            if holes:
                triangle_input["holes"] = np.asarray(
                    hole_points, dtype=np.float64
                )
            triangle_options = "pQY"
            if minimum_angle_degrees is not None:
                triangle_options = "pq{:.8g}QY".format(
                    float(minimum_angle_degrees)
                )
            triangle_result = constrained_triangle.triangulate(
                triangle_input, triangle_options
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
        if maximum_result_edge_length is not None:
            result_edge_lengths = np.linalg.norm(
                new_triangles[:, (1, 2, 0)]
                - new_triangles[:, (0, 1, 2)],
                axis=2,
            )
            if (
                len(result_edge_lengths)
                and float(result_edge_lengths.max())
                > float(maximum_result_edge_length) * (1.0 + 1e-8)
            ):
                boundary_rejection_count += 1
                continue
        new_quality = _triangle_quality_values(coordinate_pool, new_faces)
        new_quality_p5 = float(np.percentile(new_quality, 5.0))
        old_quality_p5 = float(np.percentile(old_quality, 5.0))
        strong_mean_gain = (
            bool(accept_strong_mean_gain)
            and float(new_quality.mean()) >= float(old_quality.mean()) + 0.15
            and float(new_quality.min()) >= float(old_quality.min()) - 1e-8
        )
        safe_density_simplification = (
            density_excess
            and len(new_faces) <= len(facet) * 0.8
            and float(new_quality.mean()) >= 0.45
            and new_quality_p5 >= 0.12
        )
        if (
            (
                new_quality_p5 < old_quality_p5 - 1e-8
                or float(new_quality.mean()) < float(old_quality.mean()) - 1e-8
            )
            and not strong_mean_gain
            and not safe_density_simplification
        ):
            quality_rejection_count += 1
            continue
        if safe_density_simplification:
            density_simplification_count += 1

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
            "density_forced_regions": density_forced_count,
            "density_simplifications": density_simplification_count,
            "new_vertices": 0, "removed_faces": 0,
            "new_faces": 0, "old_quality": 0.0, "new_quality": 0.0,
        }
    return (
        np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
            )
        ),
        np.vstack((faces[kept_faces], np.asarray(appended_faces, dtype=np.int64))),
        {
            "regions": region_count,
            "candidates": candidate_count,
            "boundary_rejections": boundary_rejection_count,
            "quality_rejections": quality_rejection_count,
            "constraint_rejections": constraint_rejection_count,
            "density_forced_regions": density_forced_count,
            "density_simplifications": density_simplification_count,
            "new_vertices": len(appended_vertices),
            "removed_faces": removed_face_count,
            "new_faces": len(appended_faces),
            "old_quality": float(np.mean(old_quality_values)),
            "new_quality": float(np.mean(new_quality_values)),
        },
    )


def _fit_circle_loop(points):
    points = np.asarray(points, dtype=np.float64)
    origin = points.mean(axis=0)
    _, singular_values, basis_vh = np.linalg.svd(
        points - origin, full_matrices=False
    )
    if singular_values[0] <= 0.0:
        return None
    normal = basis_vh[-1]
    basis = basis_vh[:2]
    coordinates = (points - origin) @ basis.T
    system = np.column_stack(
        (2.0 * coordinates[:, 0], 2.0 * coordinates[:, 1], np.ones(len(points)))
    )
    right_hand_side = np.sum(coordinates * coordinates, axis=1)
    solution, _, _, _ = np.linalg.lstsq(system, right_hand_side, rcond=None)
    center_2d = solution[:2]
    center = origin + center_2d @ basis
    radii = np.linalg.norm(coordinates - center_2d, axis=1)
    radius = float(radii.mean())
    if radius <= 0.0:
        return None
    return {
        "center": center,
        "normal": normal,
        "radius": radius,
        "radius_error": float(radii.std() / radius),
        "plane_error": float(singular_values[-1] / singular_values[0]),
    }


def _discover_regular_cylinder_models(
    vertices, faces, protected_edges, minimum_faces=20, radius_tolerance=3e-2
):
    """Find reliable circular-ring cylinders to seed interrupted-wall recovery."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    protected = {
        tuple(sorted(map(int, edge)))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64), axis=1
    )
    usable = np.asarray(
        [tuple(map(int, edge)) not in protected for edge in adjacency_edges]
    )
    usable_adjacency = adjacency[usable]
    graph = coo_matrix(
        (
            np.ones(len(usable_adjacency) * 2, dtype=np.uint8),
            (
                np.concatenate((usable_adjacency[:, 0], usable_adjacency[:, 1])),
                np.concatenate((usable_adjacency[:, 1], usable_adjacency[:, 0])),
            ),
        ),
        shape=(len(faces), len(faces)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    models = []
    for component_id, component_size in enumerate(np.bincount(labels)):
        if not int(minimum_faces) <= component_size <= 50000:
            continue
        face_ids = np.flatnonzero(labels == component_id)
        component_faces = faces[face_ids]
        component_edges = np.sort(
            component_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
            axis=1,
        )
        unique_edges, edge_counts = np.unique(
            component_edges, axis=0, return_counts=True
        )
        boundary_edges = unique_edges[edge_counts == 1]
        cycles = _ordered_cycles_from_edges(boundary_edges)
        if cycles is None or len(cycles) != 2 or set(cycles[0]) & set(cycles[1]):
            continue
        boundary_set = {tuple(map(int, edge)) for edge in boundary_edges}
        if not boundary_set.issubset(protected):
            continue
        fits = [_fit_circle_loop(vertices[cycle]) for cycle in cycles]
        if any(fit is None for fit in fits):
            continue
        tolerance = float(radius_tolerance)
        if any(
            fit["plane_error"] > tolerance or fit["radius_error"] > tolerance
            for fit in fits
        ):
            continue
        center_delta = fits[1]["center"] - fits[0]["center"]
        height = float(np.linalg.norm(center_delta))
        if height <= 0.0:
            continue
        axis = center_delta / height
        if axis[np.argmax(np.abs(axis))] < 0.0:
            axis = -axis
            fits = fits[::-1]
        mean_radius = 0.5 * (fits[0]["radius"] + fits[1]["radius"])
        if abs(fits[0]["radius"] - fits[1]["radius"]) > mean_radius * tolerance:
            continue
        origin = fits[0]["center"]
        first_basis = vertices[cycles[0][0]] - origin
        first_basis -= axis * np.dot(first_basis, axis)
        basis_length = float(np.linalg.norm(first_basis))
        if basis_length <= 0.0:
            continue
        first_basis /= basis_length
        candidate = {
            "axis": axis,
            "origin": origin,
            "first_basis": first_basis,
            "second_basis": np.cross(axis, first_basis),
            "radius": mean_radius,
            "source_faces": int(component_size),
            "source_face_ids": face_ids.copy(),
            "axial_min": float(
                np.min((vertices[np.unique(component_faces)] - origin) @ axis)
            ),
            "axial_max": float(
                np.max((vertices[np.unique(component_faces)] - origin) @ axis)
            ),
        }
        duplicate = False
        for model in models:
            alignment = abs(float(np.dot(model["axis"], axis)))
            origin_delta = origin - model["origin"]
            line_distance = np.linalg.norm(
                origin_delta - np.dot(origin_delta, model["axis"]) * model["axis"]
            )
            radius_scale = max(mean_radius, model["radius"], 1e-30)
            if (
                alignment >= 0.999
                and line_distance <= radius_scale * tolerance
                and abs(mean_radius - model["radius"]) <= radius_scale * tolerance
            ):
                duplicate = True
                if candidate["source_faces"] > model["source_faces"]:
                    model.update(candidate)
                break
        if not duplicate:
            models.append(candidate)
    # Boolean-interrupted outer walls are sometimes represented by only a
    # short reliable ring. A longer coaxial cylinder supplies the body-height
    # interval without changing the short ring's independently fitted radius.
    # This keeps radial recovery away from unrelated top/bottom planar bands.
    for model in models:
        own_span = float(model["axial_max"] - model["axial_min"])
        best_bounds = None
        best_span = own_span
        for support in models:
            if support is model:
                continue
            alignment = abs(float(np.dot(model["axis"], support["axis"])))
            origin_delta = support["origin"] - model["origin"]
            line_distance = np.linalg.norm(
                origin_delta
                - np.dot(origin_delta, model["axis"]) * model["axis"]
            )
            radius_scale = max(
                float(model["radius"]), float(support["radius"]), 1e-30
            )
            support_span = float(
                support["axial_max"] - support["axial_min"]
            )
            if (
                alignment < 0.999
                or line_distance > radius_scale * float(radius_tolerance)
                or support_span <= max(own_span * 4.0, best_span)
            ):
                continue
            endpoints = np.asarray(
                (
                    support["origin"]
                    + support["axis"] * float(support["axial_min"]),
                    support["origin"]
                    + support["axis"] * float(support["axial_max"]),
                )
            )
            projected = (endpoints - model["origin"]) @ model["axis"]
            best_bounds = (float(projected.min()), float(projected.max()))
            best_span = support_span
        if best_bounds is not None:
            model["axial_min"], model["axial_max"] = best_bounds
    return models


def recover_analytic_cylinder_support(
    vertices,
    faces,
    protected_edges,
    cylinder_models,
    maximum_projection_distance,
):
    """Project connected near-cylinder support and relax its pseudo seams.

    A face is eligible only when all of its vertices are inside the explicit
    projection-distance guard. Eligible components must touch a much tighter
    cylinder seed, and vertices shared with an ineligible face remain fixed so
    embossed text, grooves, and other genuine boundary geometry are retained.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    protected = {
        tuple(sorted(map(int, edge)))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    distance_limit = float(maximum_projection_distance)
    if distance_limit <= 0.0:
        raise ValueError("Cylinder recovery distance must be positive.")
    empty_stats = {
        "models": 0,
        "components": 0,
        "faces": 0,
        "projected_vertices": 0,
        "relaxed_edges": 0,
        "maximum_displacement": 0.0,
        "mean_displacement": 0.0,
        "rejected_for_quality": 0,
        "old_quality_p5": 0.0,
        "new_quality_p5": 0.0,
    }
    if not cylinder_models or len(faces) == 0:
        return (
            vertices.copy(),
            np.asarray(sorted(protected), dtype=np.int64).reshape(-1, 2),
            empty_stats,
        )

    selection_edges, _ = _relax_false_cylinder_feature_edges(
        vertices,
        faces,
        np.asarray(sorted(protected), dtype=np.int64).reshape(-1, 2),
        cylinder_models,
    )
    edge_faces = _build_edge_faces(faces)
    vertex_faces = [[] for _ in range(len(vertices))]
    for face_id, face in enumerate(faces):
        for vertex_id in face:
            vertex_faces[int(vertex_id)].append(int(face_id))

    accepted_by_model = []
    accepted_components = 0
    used_models = 0
    for model_index, model in enumerate(cylinder_models):
        accepted, selection_stats = _select_connected_cylinder_support_faces(
            vertices,
            faces,
            selection_edges,
            model,
            distance_limit,
        )
        if np.any(accepted):
            accepted_by_model.append(
                (
                    model_index,
                    accepted,
                    selection_stats["displacement"],
                    selection_stats["radii"],
                )
            )
            accepted_components += selection_stats["components"]
            used_models += 1

    if not accepted_by_model:
        return (
            vertices.copy(),
            np.asarray(sorted(protected), dtype=np.int64).reshape(-1, 2),
            empty_stats,
        )

    recovered = vertices.copy()
    best_displacement = np.full(len(vertices), np.inf, dtype=np.float64)
    accepted_union = np.zeros(len(faces), dtype=bool)
    projected_vertices = set()
    for model_index, accepted, displacement, radii in accepted_by_model:
        model = cylinder_models[model_index]
        axis = np.asarray(model["axis"], dtype=np.float64)
        origin = np.asarray(model["origin"], dtype=np.float64)
        radius = float(model["radius"])
        accepted_union |= accepted
        accepted_vertex_ids = np.unique(faces[accepted])
        for vertex_id in accepted_vertex_ids:
            incident = vertex_faces[int(vertex_id)]
            if not incident or not np.all(accepted[np.asarray(incident)]):
                continue
            local_displacement = float(displacement[int(vertex_id)])
            if local_displacement >= best_displacement[int(vertex_id)]:
                continue
            offset = vertices[int(vertex_id)] - origin
            axial_value = float(np.dot(offset, axis))
            radial = offset - axial_value * axis
            radial_length = float(radii[int(vertex_id)])
            if radial_length <= 0.0:
                continue
            recovered[int(vertex_id)] = (
                origin + axial_value * axis + radial * (radius / radial_length)
            )
            best_displacement[int(vertex_id)] = local_displacement
            projected_vertices.add(int(vertex_id))

    relaxed = set()
    for edge in protected:
        memberships = edge_faces.get(edge, ())
        if len(memberships) == 2 and any(
            all(accepted[int(face_id)] for face_id in memberships)
            for _, accepted, _, _ in accepted_by_model
        ):
            relaxed.add(edge)
    remaining = np.asarray(sorted(protected - relaxed), dtype=np.int64).reshape(-1, 2)
    displacement_values = best_displacement[np.isfinite(best_displacement)]
    recovered_face_ids = np.flatnonzero(accepted_union)
    old_quality = _triangle_quality_values(vertices, faces[recovered_face_ids])
    new_quality = _triangle_quality_values(recovered, faces[recovered_face_ids])
    old_triangles = vertices[faces[recovered_face_ids]]
    new_triangles = recovered[faces[recovered_face_ids]]
    old_crosses = np.cross(
        old_triangles[:, 1] - old_triangles[:, 0],
        old_triangles[:, 2] - old_triangles[:, 0],
    )
    new_crosses = np.cross(
        new_triangles[:, 1] - new_triangles[:, 0],
        new_triangles[:, 2] - new_triangles[:, 0],
    )
    orientation_dot = np.sum(old_crosses * new_crosses, axis=1)
    old_p5 = float(np.percentile(old_quality, 5.0))
    new_p5 = float(np.percentile(new_quality, 5.0))
    quality_rejected = (
        np.any(orientation_dot <= 0.0)
        or not np.isfinite(new_quality).all()
        or new_p5 + 1e-12 < old_p5 * 0.9
        or float(new_quality.mean()) + 1e-12
        < float(old_quality.mean()) * 0.98
    )
    if quality_rejected:
        rejected_stats = dict(empty_stats)
        rejected_stats.update(
            {
                "models": int(used_models),
                "components": int(accepted_components),
                "faces": int(len(recovered_face_ids)),
                "rejected_for_quality": int(accepted_components),
                "old_quality_p5": old_p5,
                "new_quality_p5": new_p5,
            }
        )
        return (
            vertices.copy(),
            np.asarray(sorted(protected), dtype=np.int64).reshape(-1, 2),
            rejected_stats,
        )
    stats = {
        "models": int(used_models),
        "components": int(accepted_components),
        "faces": int(np.count_nonzero(accepted_union)),
        "projected_vertices": int(len(projected_vertices)),
        "relaxed_edges": int(len(relaxed)),
        "maximum_displacement": (
            float(displacement_values.max()) if len(displacement_values) else 0.0
        ),
        "mean_displacement": (
            float(displacement_values.mean()) if len(displacement_values) else 0.0
        ),
        "rejected_for_quality": 0,
        "old_quality_p5": old_p5,
        "new_quality_p5": new_p5,
    }
    return recovered, remaining, stats


def _relax_false_cylinder_feature_edges(
    vertices,
    faces,
    protected_edges,
    cylinder_models,
    minimum_normal_alignment=0.9,
):
    """Remove tessellation seams whose two sides share one cylinder support."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    protected = {
        tuple(sorted(map(int, edge)))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    if not protected or not cylinder_models:
        return np.asarray(sorted(protected), dtype=np.int64).reshape(-1, 2), 0

    edge_faces = _build_edge_faces(faces)
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    cross_lengths = np.linalg.norm(crosses, axis=1)
    normals = np.zeros_like(crosses)
    valid = cross_lengths > 0.0
    normals[valid] = crosses[valid] / cross_lengths[valid, None]
    relaxed = set()

    for model in cylinder_models:
        axis = np.asarray(model["axis"], dtype=np.float64)
        origin = np.asarray(model["origin"], dtype=np.float64)
        radius = float(model["radius"])
        centroids = triangles.mean(axis=1) - origin
        centroid_axial = centroids @ axis
        centroid_radial = centroids - centroid_axial[:, None] * axis
        centroid_radii = np.linalg.norm(centroid_radial, axis=1)
        alignment = np.abs(
            np.sum(normals * centroid_radial, axis=1)
            / np.maximum(centroid_radii, 1e-30)
        )
        broad_radius_tolerance = max(radius * 5e-2, 2e-2)
        support_tolerance = max(radius * 2.5e-3, 2e-2)
        broad = (
            valid
            & (alignment >= float(minimum_normal_alignment))
            & (np.abs(centroid_radii - radius) <= broad_radius_tolerance)
        )
        broad_ids = np.flatnonzero(broad)
        best = None
        for slope_candidate in np.linspace(-0.12, 0.12, 97):
            intercepts = (
                centroid_radii[broad_ids]
                - slope_candidate * centroid_axial[broad_ids]
            )
            bins = np.round(intercepts / support_tolerance).astype(np.int64)
            if len(bins) == 0:
                continue
            unique_bins, counts = np.unique(bins, return_counts=True)
            bin_index = int(np.argmax(counts))
            score = int(counts[bin_index])
            intercept_candidate = (
                float(unique_bins[bin_index]) * support_tolerance
            )
            if best is None or score > best[0]:
                best = (score, float(slope_candidate), intercept_candidate)
        if best is None:
            continue
        _, support_slope, support_intercept = best
        support_residual = centroid_radii - (
            support_intercept + support_slope * centroid_axial
        )
        fit_ids = np.flatnonzero(
            broad & (np.abs(support_residual) <= support_tolerance * 2.0)
        )
        if len(fit_ids) >= 2:
            system = np.column_stack(
                (
                    centroid_axial[fit_ids],
                    np.ones(len(fit_ids), dtype=np.float64),
                )
            )
            support_slope, support_intercept = np.linalg.lstsq(
                system,
                centroid_radii[fit_ids],
                rcond=None,
            )[0]
        for edge in protected:
            memberships = edge_faces.get(edge, ())
            if len(memberships) != 2:
                continue
            first, second = map(int, memberships)
            if (
                not valid[first]
                or not valid[second]
                or alignment[first] < float(minimum_normal_alignment)
                or alignment[second] < float(minimum_normal_alignment)
            ):
                continue
            local_vertex_ids = np.unique(faces[[first, second]].reshape(-1))
            local_offsets = vertices[local_vertex_ids] - origin
            local_axial = local_offsets @ axis
            local_radial = local_offsets - local_axial[:, None] * axis
            local_radii = np.linalg.norm(local_radial, axis=1)
            residual = local_radii - (
                support_slope * local_axial + support_intercept
            )
            if float(np.max(np.abs(residual))) <= support_tolerance * 2.0:
                relaxed.add(edge)

    remaining = np.asarray(sorted(protected - relaxed), dtype=np.int64).reshape(-1, 2)
    return remaining, len(relaxed)


def _select_connected_cylinder_support_faces(
    vertices,
    faces,
    protected_edges,
    model,
    maximum_projection_distance,
    smoothing_passes=8,
):
    """Select a seed-connected cylinder patch with an edge-coherent boundary."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    protected = {
        tuple(sorted(map(int, edge)))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    axis = np.asarray(model["axis"], dtype=np.float64)
    origin = np.asarray(model["origin"], dtype=np.float64)
    radius = float(model["radius"])
    distance_limit = float(maximum_projection_distance)
    offsets = vertices - origin
    axial = offsets @ axis
    radial = offsets - axial[:, None] * axis
    radii = np.linalg.norm(radial, axis=1)
    displacement = np.abs(radii - radius)
    face_displacement = displacement[faces]
    face_axial = axial[faces]
    axial_ok = np.ones(len(faces), dtype=bool)
    if "axial_min" in model and "axial_max" in model:
        axial_ok = (
            np.min(face_axial, axis=1)
            >= float(model["axial_min"]) - distance_limit
        ) & (
            np.max(face_axial, axis=1)
            <= float(model["axial_max"]) + distance_limit
        )
    core = axial_ok & (
        np.max(face_displacement, axis=1) <= distance_limit
    )
    growth = axial_ok & (
        np.max(face_displacement, axis=1) <= distance_limit * 1.25
    )
    seed_limit = min(
        distance_limit * 0.25,
        max(radius * 5e-4, np.finfo(np.float64).eps * radius * 100.0),
    )
    seed = axial_ok & (
        np.max(face_displacement, axis=1) <= seed_limit
    )
    source_face_ids = np.asarray(
        model.get("source_face_ids", ()), dtype=np.int64
    )
    valid_source = source_face_ids[
        (source_face_ids >= 0) & (source_face_ids < len(faces))
    ]
    seed[valid_source] = True

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64), axis=1
    )
    smooth_pair = np.asarray(
        [tuple(map(int, edge)) not in protected for edge in adjacency_edges]
    )
    smooth_adjacency = adjacency[smooth_pair]
    selected = core.copy()
    for _ in range(max(int(smoothing_passes), 0)):
        selected_neighbors = np.zeros(len(faces), dtype=np.int16)
        np.add.at(
            selected_neighbors,
            smooth_adjacency[:, 0],
            selected[smooth_adjacency[:, 1]],
        )
        np.add.at(
            selected_neighbors,
            smooth_adjacency[:, 1],
            selected[smooth_adjacency[:, 0]],
        )
        updated = selected.copy()
        updated[growth & ~selected & (selected_neighbors >= 2)] = True
        updated[selected & ~seed & (selected_neighbors <= 1)] = False
        if np.array_equal(updated, selected):
            break
        selected = updated

    usable = (
        selected[smooth_adjacency[:, 0]]
        & selected[smooth_adjacency[:, 1]]
    )
    selected_adjacency = smooth_adjacency[usable]
    graph = coo_matrix(
        (
            np.ones(len(selected_adjacency) * 2, dtype=np.uint8),
            (
                np.concatenate(
                    (selected_adjacency[:, 0], selected_adjacency[:, 1])
                ),
                np.concatenate(
                    (selected_adjacency[:, 1], selected_adjacency[:, 0])
                ),
            ),
        ),
        shape=(len(faces), len(faces)),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    accepted = np.zeros(len(faces), dtype=bool)
    accepted_components = 0
    for component_id in range(component_count):
        component_faces = np.flatnonzero(
            selected & (labels == component_id)
        )
        if len(component_faces) == 0 or not np.any(seed[component_faces]):
            continue
        accepted[component_faces] = True
        accepted_components += 1
    return accepted, {
        "components": int(accepted_components),
        "displacement": displacement,
        "radii": radii,
        "seed": seed,
        "core_faces": int(np.count_nonzero(core)),
        "growth_faces": int(np.count_nonzero(growth)),
    }


def build_feature_partition_face_colors(
    vertices,
    faces,
    feature_angle_degrees=15.0,
):
    """Color final smooth partitions for diagnostic PLY visualization."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    protected = detect_hard_constraint_edges(
        mesh,
        feature_angle_degrees=feature_angle_degrees,
    )
    models = _discover_regular_cylinder_models(
        vertices,
        faces,
        protected,
        minimum_faces=8,
        radius_tolerance=3e-2,
    )
    protected, relaxed = _relax_false_cylinder_feature_edges(
        vertices,
        faces,
        protected,
        models,
    )
    protected_set = {tuple(map(int, edge)) for edge in protected}
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64), axis=1
    )
    usable = np.asarray(
        [tuple(map(int, edge)) not in protected_set for edge in adjacency_edges]
    )
    smooth_adjacency = adjacency[usable]
    graph = coo_matrix(
        (
            np.ones(len(smooth_adjacency) * 2, dtype=np.uint8),
            (
                np.concatenate(
                    (smooth_adjacency[:, 0], smooth_adjacency[:, 1])
                ),
                np.concatenate(
                    (smooth_adjacency[:, 1], smooth_adjacency[:, 0])
                ),
            ),
        ),
        shape=(len(faces), len(faces)),
    ).tocsr()
    partition_count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=partition_count)
    order = np.argsort(-sizes, kind="stable")
    display_id = np.empty(partition_count, dtype=np.int64)
    display_id[order] = np.arange(partition_count, dtype=np.int64)

    palette = np.empty((partition_count, 4), dtype=np.uint8)
    golden_ratio = 0.6180339887498949
    for label in range(partition_count):
        index = int(display_id[label])
        hue = (0.08 + index * golden_ratio) % 1.0
        saturation = 0.58 + 0.16 * (index % 3)
        value = 0.88 if index % 2 == 0 else 0.72
        rgb = colorsys.hsv_to_rgb(hue, min(saturation, 0.9), value)
        palette[label, :3] = np.rint(np.asarray(rgb) * 255.0).astype(np.uint8)
        palette[label, 3] = 255
    return palette[labels], display_id[labels], {
        "partitions": int(partition_count),
        "largest_partition_faces": int(sizes.max()) if len(sizes) else 0,
        "cylinder_seams_relaxed": int(relaxed),
    }


def export_feature_partition_ply(
    path,
    vertices,
    faces,
    feature_angle_degrees=15.0,
):
    """Export a face-colored PLY showing final feature partitions."""
    face_colors, _, stats = build_feature_partition_face_colors(
        vertices,
        faces,
        feature_angle_degrees=feature_angle_degrees,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = face_colors
    mesh.remove_unreferenced_vertices()
    mesh.export(str(path), file_type="ply")
    return stats


def export_outer_cylinder_diagnostic_ply(
    path,
    reference_vertices,
    reference_faces,
    feature_angle_degrees=15.0,
    cylinder_minimum_faces=20,
    cylinder_radius_tolerance=3e-2,
    recovery_distance=0.2,
    export_remainder=False,
):
    """Export the exact input faces selected as one outer-cylinder candidate.

    No neighborhood or raised-feature shell is appended. Green marks reliable
    seed support, cyan marks radial candidate faces, and orange marks selected
    but non-radial disturbed faces which make this candidate risky to remesh.
    """
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    reference_faces = np.asarray(reference_faces, dtype=np.int64)
    reference_mesh = trimesh.Trimesh(
        vertices=reference_vertices,
        faces=reference_faces,
        process=False,
    )
    protected = detect_hard_constraint_edges(
        reference_mesh,
        feature_angle_degrees=feature_angle_degrees,
    )
    models = _discover_regular_cylinder_models(
        reference_vertices,
        reference_faces,
        protected,
        minimum_faces=cylinder_minimum_faces,
        radius_tolerance=cylinder_radius_tolerance,
    )
    if not models:
        raise ValueError("No fitted cylinder is available for diagnostic export.")
    model = max(models, key=lambda item: float(item["radius"]))
    selection_protected, _ = _relax_false_cylinder_feature_edges(
        reference_vertices,
        reference_faces,
        protected,
        models,
    )
    axis = np.asarray(model["axis"], dtype=np.float64)
    origin = np.asarray(model["origin"], dtype=np.float64)
    radius = float(model["radius"])
    distance_limit = float(recovery_distance)
    if distance_limit <= 0.0:
        raise ValueError("Cylinder diagnostic recovery distance must be positive.")

    triangles = reference_vertices[reference_faces]
    centroids = triangles.mean(axis=1)
    offsets = centroids - origin
    axial = offsets @ axis
    radial_vectors = offsets - axial[:, None] * axis
    centroid_radii = np.linalg.norm(radial_vectors, axis=1)
    selected, selection_stats = _select_connected_cylinder_support_faces(
        reference_vertices,
        reference_faces,
        selection_protected,
        model,
        distance_limit,
    )
    candidate_selected = selected.copy()
    centroid_radial_deviation = np.abs(centroid_radii - radius)
    shell_halfwidth = max(distance_limit * 2.5, radius * 5e-3)
    axial_margin = max(distance_limit, radius * 1e-3)
    outer_wall_shell = (
        (axial >= float(model["axial_min"]) - axial_margin)
        & (axial <= float(model["axial_max"]) + axial_margin)
        & (centroid_radial_deviation <= shell_halfwidth)
    )
    if export_remainder:
        selected = outer_wall_shell & ~candidate_selected
    selected_ids = np.flatnonzero(selected)
    if len(selected_ids) == 0:
        raise ValueError("The fitted outer cylinder has no candidate faces to export.")

    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    cross_lengths = np.linalg.norm(crosses, axis=1)
    normals = np.zeros_like(crosses)
    valid = cross_lengths > 0.0
    normals[valid] = crosses[valid] / cross_lengths[valid, None]
    radial_alignment = np.abs(
        np.sum(normals * radial_vectors, axis=1)
        / np.maximum(centroid_radii, 1e-30)
    )
    category = np.full(len(reference_faces), 3, dtype=np.int64)
    category[centroid_radial_deviation <= distance_limit] = 1
    category[
        (centroid_radial_deviation <= distance_limit)
        & (radial_alignment < 0.75)
    ] = 2
    category[selection_stats["seed"] & (radial_alignment >= 0.9)] = 0
    palette = np.asarray(
        (
            (46, 204, 113, 255),   # reliable support: green
            (52, 152, 219, 255),   # near-cylinder disturbance: cyan
            (243, 156, 18, 255),   # non-radial transition: orange
            (214, 48, 149, 255),   # reserved; excluded from this export
        ),
        dtype=np.uint8,
    )
    diagnostic = trimesh.Trimesh(
        vertices=reference_vertices.copy(),
        faces=reference_faces[selected_ids].copy(),
        process=False,
    )
    diagnostic.visual.face_colors = palette[category[selected_ids]]
    diagnostic.remove_unreferenced_vertices()
    diagnostic.export(str(path), file_type="ply")
    counts = np.bincount(category[selected_ids], minlength=4)
    return {
        "radius": radius,
        "faces": int(len(selected_ids)),
        "reliable_support_faces": int(counts[0]),
        "near_cylinder_faces": int(counts[1]),
        "transition_faces": int(counts[2]),
        "raised_feature_faces": int(counts[3]),
        "candidate_components": int(selection_stats["components"]),
        "shell_halfwidth": float(shell_halfwidth),
        "recovery_distance": distance_limit,
        "selection": "remainder" if export_remainder else "candidate",
    }


def retriangulate_interrupted_cylindrical_walls(
    vertices,
    faces,
    protected_edges,
    cylinder_models,
    minimum_faces=20,
    radius_tolerance=3e-2,
    preferred_edge_length=None,
    seam_angle_offset=0.0,
    minimum_support_alignment=0.94,
    seam_guard_degrees=1.0,
    selected_face_masks=None,
    project_interior_to_source=True,
    excluded_edges=None,
):
    """Recover cylindrical support patches interrupted by embossed features."""
    if constrained_triangle is None:
        raise RuntimeError(
            "Interrupted-cylinder remeshing needs the 'triangle' package."
        )
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    protected = {
        tuple(sorted(map(int, edge)))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    excluded = {
        tuple(sorted(map(int, edge)))
        for edge in np.asarray(
            excluded_edges if excluded_edges is not None else (),
            dtype=np.int64,
        ).reshape(-1, 2)
    }
    kept_faces = np.ones(len(faces), dtype=bool)
    claimed_faces = np.zeros(len(faces), dtype=bool)
    appended_vertices = []
    appended_faces = []
    candidates = accepted = removed_faces = 0
    rejection_counts = {}

    def reject(reason):
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    old_qualities = []
    new_qualities = []
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64), axis=1
    )
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    cross_lengths = np.linalg.norm(crosses, axis=1)
    valid_faces = cross_lengths > 0.0
    normals = np.zeros_like(crosses)
    normals[valid_faces] = crosses[valid_faces] / cross_lengths[valid_faces, None]
    excluded_faces = _faces_touching_edge_set(faces, excluded)
    protected_adjacency = _edge_rows_in_set(adjacency_edges, protected)

    for model_index, model in enumerate(cylinder_models):
        axis = np.asarray(model["axis"], dtype=np.float64)
        origin = np.asarray(model["origin"], dtype=np.float64)
        first_basis = np.asarray(model["first_basis"], dtype=np.float64)
        second_basis = np.asarray(model["second_basis"], dtype=np.float64)
        seed_radius = float(model["radius"])
        centroid_offsets = triangles.mean(axis=1) - origin
        centroid_z = centroid_offsets @ axis
        centroid_radial = centroid_offsets - centroid_z[:, None] * axis
        centroid_radii = np.linalg.norm(centroid_radial, axis=1)
        radial_alignment = np.abs(
            np.sum(normals * centroid_radial, axis=1)
            / np.maximum(centroid_radii, 1e-30)
        )
        broad_tolerance = seed_radius * max(float(radius_tolerance), 5e-3)
        broad = (
            valid_faces
            & (radial_alignment >= 0.94)
            & (np.abs(centroid_radii - seed_radius) <= broad_tolerance * 1.5)
        )
        if int(np.count_nonzero(broad)) < int(minimum_faces):
            continue

        # Embossed front faces can also point radially.  A small Hough search
        # finds the dominant r(z) support and separates those offset faces.
        support_tolerance = max(seed_radius * 2.5e-3, 2e-2)
        broad_ids = np.flatnonzero(broad)
        best = None
        for slope in np.linspace(-0.12, 0.12, 97):
            intercepts = centroid_radii[broad_ids] - slope * centroid_z[broad_ids]
            near_seed = np.abs(intercepts - seed_radius) <= broad_tolerance * 1.5
            values = intercepts[near_seed]
            if len(values) < int(minimum_faces):
                continue
            bins = np.round(values / support_tolerance).astype(np.int64)
            unique_bins, counts = np.unique(bins, return_counts=True)
            index = int(np.argmax(counts))
            score = int(counts[index])
            intercept = float(unique_bins[index]) * support_tolerance
            if best is None or score > best[0]:
                best = (score, float(slope), intercept)
        if best is None:
            continue
        _, slope, intercept = best
        residual = centroid_radii - (intercept + slope * centroid_z)
        fit_ids = np.flatnonzero(broad & (np.abs(residual) <= support_tolerance * 2.0))
        if len(fit_ids) >= 2:
            system = np.column_stack((centroid_z[fit_ids], np.ones(len(fit_ids))))
            slope, intercept = np.linalg.lstsq(
                system, centroid_radii[fit_ids], rcond=None
            )[0]

        vertex_offsets = triangles - origin
        vertex_z = vertex_offsets @ axis
        vertex_radial = vertex_offsets - vertex_z[:, :, None] * axis
        vertex_radii = np.linalg.norm(vertex_radial, axis=2)
        support_residual = vertex_radii - (intercept + slope * vertex_z)
        selected = (
            valid_faces
            & ~claimed_faces
            & ~excluded_faces
            & (radial_alignment >= float(minimum_support_alignment))
            & (np.max(np.abs(support_residual), axis=1) <= support_tolerance * 2.0)
        )
        if selected_face_masks is not None:
            selected = (
                np.asarray(selected_face_masks[model_index], dtype=bool)
                & ~claimed_faces
                & ~excluded_faces
                & valid_faces
            )
        if int(np.count_nonzero(selected)) < int(minimum_faces):
            continue

        centroid_angles = np.mod(
            np.arctan2(centroid_radial @ second_basis, centroid_radial @ first_basis),
            2.0 * np.pi,
        )
        histogram, edges = np.histogram(
            centroid_angles[selected], bins=180, range=(0.0, 2.0 * np.pi)
        )
        seam_index = int(np.argmin(histogram))
        seam_angle = (
            0.5 * (edges[seam_index] + edges[seam_index + 1])
            + float(seam_angle_offset)
        ) % (2.0 * np.pi)
        vertex_angles = np.arctan2(
            vertex_radial @ second_basis, vertex_radial @ first_basis
        )
        seam_distance = np.abs(
            np.arctan2(
                np.sin(vertex_angles - seam_angle),
                np.cos(vertex_angles - seam_angle),
            )
        )
        seam_guard = np.deg2rad(max(float(seam_guard_degrees), 0.0))
        if seam_guard > 0.0:
            selected &= np.min(seam_distance, axis=1) >= seam_guard

        usable = (
            selected[adjacency[:, 0]]
            & selected[adjacency[:, 1]]
            & ~protected_adjacency
        )
        usable_adjacency = adjacency[usable]
        graph = coo_matrix(
            (
                np.ones(len(usable_adjacency) * 2, dtype=np.uint8),
                (
                    np.concatenate(
                        (usable_adjacency[:, 0], usable_adjacency[:, 1])
                    ),
                    np.concatenate(
                        (usable_adjacency[:, 1], usable_adjacency[:, 0])
                    ),
                ),
            ),
            shape=(len(faces), len(faces)),
        ).tocsr()
        _, labels = connected_components(graph, directed=False)
        component_sizes = np.bincount(labels, weights=selected.astype(np.int64))

        for component_id in np.flatnonzero(
            (component_sizes >= int(minimum_faces)) & (component_sizes <= 50000)
        ):
            face_ids = np.flatnonzero((labels == component_id) & selected)
            component_faces = faces[face_ids]
            component_edges = np.sort(
                component_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
                axis=1,
            )
            unique_edges, edge_counts = np.unique(
                component_edges, axis=0, return_counts=True
            )
            boundary_edges = unique_edges[edge_counts == 1]
            cycles = _ordered_cycles_from_edges(boundary_edges)
            if cycles is None or len(cycles) == 0:
                continue
            candidates += 1
            boundary_set = {tuple(map(int, edge)) for edge in boundary_edges}
            internal_constraint_edges = sorted(
                (set(map(tuple, unique_edges)) & protected) - boundary_set
            )

            projected_cycles = []
            for cycle in cycles:
                offsets = vertices[np.asarray(cycle, dtype=np.int64)] - origin
                z_values = offsets @ axis
                radial = offsets - z_values[:, None] * axis
                raw_angles = (
                    np.arctan2(radial @ second_basis, radial @ first_basis)
                    - seam_angle
                )
                angular_steps = np.arctan2(
                    np.sin(raw_angles[1:] - raw_angles[:-1]),
                    np.cos(raw_angles[1:] - raw_angles[:-1]),
                )
                angles = np.concatenate(
                    ([raw_angles[0]], raw_angles[0] + np.cumsum(angular_steps))
                )
                angles -= 2.0 * np.pi * np.floor(angles.min() / (2.0 * np.pi))
                projected_cycles.append(
                    np.column_stack((seed_radius * angles, z_values))
                )
            areas = np.asarray(
                [abs(_polygon_signed_area(polygon)) for polygon in projected_cycles]
            )
            if not len(areas) or float(areas.max()) <= 1e-12:
                reject("parameter_boundary")
                continue
            outer_index = int(np.argmax(areas))
            outer = projected_cycles[outer_index]
            holes = [
                polygon
                for index, polygon in enumerate(projected_cycles)
                if index != outer_index
            ]
            # Every boundary cycle was unwrapped independently.  Move inner
            # cycles to the same 2*pi branch as the outer loop so a feature
            # crossing the artificial angular seam stays local in the
            # developed cylinder plane.
            circumference = 2.0 * np.pi * seed_radius
            outer_x_center = float(np.mean(outer[:, 0]))
            holes = [
                polygon
                + np.array(
                    [
                        circumference
                        * np.round(
                            (outer_x_center - float(np.mean(polygon[:, 0])))
                            / circumference
                        ),
                        0.0,
                    ]
                )
                for polygon in holes
            ]
            hole_points = [_polygon_interior_point(hole) for hole in holes]
            if any(point is None for point in hole_points):
                reject("hole_point")
                continue
            if any(
                not _points_in_polygon(point[None, :], outer)[0]
                for point in hole_points
            ):
                reject("hole_outside")
                continue
            ordered_cycles = [cycles[outer_index]] + [
                cycle for index, cycle in enumerate(cycles) if index != outer_index
            ]
            polygons = [outer] + holes
            all_boundary_2d = np.vstack(polygons)
            all_boundary_ids = np.concatenate(
                [np.asarray(cycle, dtype=np.int64) for cycle in ordered_cycles]
            )
            boundary_id_set = set(map(int, all_boundary_ids))
            internal_vertex_ids = np.asarray(
                sorted(
                    {
                        int(vertex_id)
                        for edge in internal_constraint_edges
                        for vertex_id in edge
                    }
                    - boundary_id_set
                ),
                dtype=np.int64,
            )
            if len(internal_vertex_ids):
                internal_offsets = vertices[internal_vertex_ids] - origin
                internal_z = internal_offsets @ axis
                internal_radial = (
                    internal_offsets - internal_z[:, None] * axis
                )
                internal_angles = np.mod(
                    np.arctan2(
                        internal_radial @ second_basis,
                        internal_radial @ first_basis,
                    )
                    - seam_angle,
                    2.0 * np.pi,
                )
                internal_x = seed_radius * internal_angles
                internal_x += circumference * np.round(
                    (outer_x_center - internal_x) / circumference
                )
                internal_2d = np.column_stack((internal_x, internal_z))
            else:
                internal_2d = np.empty((0, 2), dtype=np.float64)
            fixed_2d = np.vstack((all_boundary_2d, internal_2d))
            fixed_ids = np.concatenate((all_boundary_ids, internal_vertex_ids))
            fixed_lookup = {
                int(vertex_id): local_index
                for local_index, vertex_id in enumerate(fixed_ids)
            }
            boundary_lengths = np.concatenate(
                [
                    np.linalg.norm(
                        vertices[np.asarray(cycle, dtype=np.int64)]
                        - vertices[np.roll(np.asarray(cycle, dtype=np.int64), -1)],
                        axis=1,
                    )
                    for cycle in ordered_cycles
                ]
            )
            positive_lengths = boundary_lengths[boundary_lengths > 0.0]
            if len(positive_lengths) == 0:
                reject("boundary_length")
                continue
            spacing = float(np.median(positive_lengths))
            if preferred_edge_length is not None:
                spacing = max(spacing, float(preferred_edge_length))
            if not np.isfinite(spacing) or spacing <= 0.0:
                reject("spacing")
                continue

            lower = outer.min(axis=0)
            upper = outer.max(axis=0)
            row_step = spacing * np.sqrt(3.0) * 0.5
            segment_starts = np.vstack(polygons)
            segment_ends = np.vstack(
                [np.roll(polygon, -1, axis=0) for polygon in polygons]
            )
            if internal_constraint_edges:
                internal_segment_starts = np.asarray(
                    [
                        fixed_2d[fixed_lookup[int(edge[0])]]
                        for edge in internal_constraint_edges
                    ],
                    dtype=np.float64,
                )
                internal_segment_ends = np.asarray(
                    [
                        fixed_2d[fixed_lookup[int(edge[1])]]
                        for edge in internal_constraint_edges
                    ],
                    dtype=np.float64,
                )
                segment_starts = np.vstack(
                    (segment_starts, internal_segment_starts)
                )
                segment_ends = np.vstack((segment_ends, internal_segment_ends))
            segment_vectors = segment_ends - segment_starts
            segment_length_squared = np.maximum(
                np.sum(segment_vectors * segment_vectors, axis=1),
                np.finfo(np.float64).eps,
            )
            grid_points = []
            row = 0
            y_value = lower[1] + row_step
            while y_value < upper[1]:
                x_values = np.arange(
                    lower[0] + spacing * (0.5 if row % 2 == 0 else 1.0),
                    upper[0],
                    spacing,
                )
                if len(x_values):
                    points = np.column_stack(
                        (x_values, np.full(len(x_values), y_value))
                    )
                    inside = _points_in_polygon(points, outer)
                    for hole in holes:
                        inside &= ~_points_in_polygon(points, hole)
                    points = points[inside]
                    if len(points):
                        differences = points[:, None, :] - segment_starts[None, :, :]
                        fractions = np.clip(
                            np.sum(differences * segment_vectors[None, :, :], axis=2)
                            / segment_length_squared[None, :],
                            0.0,
                            1.0,
                        )
                        closest = (
                            segment_starts[None, :, :]
                            + fractions[:, :, None] * segment_vectors[None, :, :]
                        )
                        distances = np.sqrt(
                            np.min(
                                np.sum((points[:, None, :] - closest) ** 2, axis=2),
                                axis=1,
                            )
                        )
                        grid_points.extend(points[distances >= spacing * 0.55])
                row += 1
                y_value += row_step

            interior_2d = np.asarray(grid_points, dtype=np.float64).reshape(-1, 2)
            all_2d = np.vstack((fixed_2d, interior_2d))
            offsets = np.cumsum([0] + [len(cycle) for cycle in ordered_cycles])
            required_edges = set()
            for start, end in zip(offsets[:-1], offsets[1:]):
                required_edges.update(
                    tuple(
                        sorted(
                            (index, start + (index - start + 1) % (end - start))
                        )
                    )
                    for index in range(start, end)
                )
            required_edges.update(
                tuple(
                    sorted(
                        (
                            fixed_lookup[int(edge[0])],
                            fixed_lookup[int(edge[1])],
                        )
                    )
                )
                for edge in internal_constraint_edges
            )
            triangle_input = {
                "vertices": all_2d,
                "segments": np.asarray(sorted(required_edges), dtype=np.int32),
            }
            if holes:
                triangle_input["holes"] = np.asarray(hole_points, dtype=np.float64)
            result = constrained_triangle.triangulate(triangle_input, "pQY")
            if "triangles" not in result:
                reject("triangulation")
                continue
            result_2d = np.asarray(result["vertices"], dtype=np.float64)
            local_faces = np.asarray(result["triangles"], dtype=np.int64)
            if len(local_faces) == 0:
                reject("empty_result")
                continue
            local_edges = {
                tuple(sorted(map(int, edge)))
                for face in local_faces
                for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
            }
            if not required_edges.issubset(local_edges):
                reject(
                    "missing_boundary_{}_segments_{}".format(
                        len(required_edges - local_edges),
                        len(result.get("segments", ())),
                    )
                )
                continue
            interior_2d = result_2d[len(fixed_2d):]
            new_angles = interior_2d[:, 0] / seed_radius + seam_angle
            new_z = interior_2d[:, 1]
            new_radii = intercept + slope * new_z
            new_vertices = (
                origin
                + new_z[:, None] * axis
                + new_radii[:, None]
                * (
                    np.cos(new_angles)[:, None] * first_basis
                    + np.sin(new_angles)[:, None] * second_basis
                )
            )
            if len(new_vertices) and project_interior_to_source:
                _, _, new_vertices = igl.point_mesh_squared_distance(
                    new_vertices, vertices, component_faces
                )
                new_vertices = np.asarray(new_vertices, dtype=np.float64).reshape(-1, 3)
            first_new = len(vertices) + len(appended_vertices)
            local_to_global = np.concatenate(
                (
                    fixed_ids,
                    np.arange(first_new, first_new + len(new_vertices), dtype=np.int64),
                )
            )
            new_faces = local_to_global[local_faces]
            coordinate_pool = np.vstack(
                (
                    vertices,
                    np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
                    new_vertices,
                )
            )
            new_triangles = coordinate_pool[new_faces]
            new_crosses = np.cross(
                new_triangles[:, 1] - new_triangles[:, 0],
                new_triangles[:, 2] - new_triangles[:, 0],
            )
            old_triangles = vertices[component_faces]
            old_crosses = np.cross(
                old_triangles[:, 1] - old_triangles[:, 0],
                old_triangles[:, 2] - old_triangles[:, 0],
            )
            old_offsets = old_triangles.mean(axis=1) - origin
            old_z = old_offsets @ axis
            old_radial = old_offsets - old_z[:, None] * axis
            orientation = np.sign(np.sum(old_crosses * old_radial, axis=1).mean())
            new_offsets = new_triangles.mean(axis=1) - origin
            new_z_values = new_offsets @ axis
            new_radial = new_offsets - new_z_values[:, None] * axis
            reverse = np.sum(new_crosses * new_radial, axis=1) * orientation < 0.0
            new_faces[reverse] = new_faces[reverse][:, [0, 2, 1]]

            new_edges = np.sort(
                new_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1
            )
            new_unique, new_counts = np.unique(new_edges, axis=0, return_counts=True)
            count_map = {
                tuple(map(int, edge)): int(count)
                for edge, count in zip(new_unique, new_counts)
            }
            if any(count_map.get(edge, 0) != 1 for edge in boundary_set):
                reject("boundary_topology")
                continue
            if any(
                count != (1 if edge in boundary_set else 2)
                for edge, count in count_map.items()
            ):
                reject("result_topology")
                continue
            old_quality = _triangle_quality_values(vertices, component_faces)
            new_quality = _triangle_quality_values(coordinate_pool, new_faces)
            if (
                float(new_quality.mean()) < float(old_quality.mean()) - 1e-8
                or np.percentile(new_quality, 5.0)
                < np.percentile(old_quality, 5.0) - 1e-8
            ):
                reject("quality")
                continue
            kept_faces[face_ids] = False
            claimed_faces[face_ids] = True
            appended_vertices.extend(new_vertices)
            appended_faces.extend(new_faces)
            removed_faces += len(face_ids)
            accepted += 1
            old_qualities.extend(old_quality)
            new_qualities.extend(new_quality)

    if not accepted:
        return vertices.copy(), faces.copy(), {
            "candidates": candidates, "patches": 0, "new_vertices": 0,
            "removed_faces": 0, "new_faces": 0,
            "old_quality": 0.0, "new_quality": 0.0,
            "rejections": rejection_counts,
        }
    return (
        np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
            )
        ),
        np.vstack((faces[kept_faces], np.asarray(appended_faces, dtype=np.int64))),
        {
            "candidates": candidates, "patches": accepted,
            "new_vertices": len(appended_vertices), "removed_faces": removed_faces,
            "new_faces": len(appended_faces),
            "old_quality": float(np.mean(old_qualities)),
            "new_quality": float(np.mean(new_qualities)),
            "rejections": rejection_counts,
        },
    )


def retriangulate_partial_cylindrical_walls(
    vertices,
    faces,
    protected_edges,
    minimum_faces=20,
    radius_tolerance=2e-3,
    normal_tolerance=2e-2,
    minimum_angle_degrees=30.0,
    target_edge_ratio=1.0,
    isolate_rounded_faces=False,
    minimum_curvature_degrees=0.2,
    maximum_source_quality=None,
    minimum_triangle_angle_degrees=None,
    minimum_radial_alignment=0.98,
    preferred_edge_length=None,
    maximum_angle_degrees=None,
    virtual_boundary_edges=None,
    direct_target_spacing=False,
    allow_density_simplification=False,
    maximum_result_edge_length=None,
    maximum_face_count_ratio=None,
    model_guided_boundary_recovery=False,
):
    """Remesh open or irregularly trimmed cylinder patches with one boundary loop."""
    if constrained_triangle is None:
        raise RuntimeError(
            "Partial-cylinder constrained remeshing needs the 'triangle' package."
        )
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    virtual_boundaries = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(
            virtual_boundary_edges
            if virtual_boundary_edges is not None
            else np.empty((0, 2), dtype=np.int64),
            dtype=np.int64,
        ).reshape(-1, 2)
    }
    separators = protected | virtual_boundaries
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64), axis=1
    )
    usable = ~_edge_rows_in_set(adjacency_edges, separators)
    growth_graph = None
    growth_normals = None
    growth_centroids = None
    if model_guided_boundary_recovery:
        # The ordinary component graph stops at every protected edge.  The
        # model-guided graph may cross such an edge only when the neighboring
        # face still satisfies the same fitted cylinder.  This reconnects
        # false tessellation seams without swallowing caps, fillets, or joins.
        growth_usable = ~_edge_rows_in_set(
            adjacency_edges, virtual_boundaries
        )
        growth_adjacency = adjacency[growth_usable]
        growth_graph = coo_matrix(
            (
                np.ones(len(growth_adjacency) * 2, dtype=np.uint8),
                (
                    np.concatenate(
                        (growth_adjacency[:, 0], growth_adjacency[:, 1])
                    ),
                    np.concatenate(
                        (growth_adjacency[:, 1], growth_adjacency[:, 0])
                    ),
                ),
            ),
            shape=(len(faces), len(faces)),
        ).tocsr()
        growth_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        growth_centroids = vertices[faces].mean(axis=1)
    if isolate_rounded_faces:
        adjacency_angles = np.degrees(
            np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
        )
        face_curvature = np.zeros(len(faces), dtype=np.float64)
        np.maximum.at(face_curvature, adjacency[:, 0], adjacency_angles)
        np.maximum.at(face_curvature, adjacency[:, 1], adjacency_angles)
        rounded_faces = face_curvature >= float(minimum_curvature_degrees)
        if maximum_source_quality is not None:
            rounded_faces &= (
                _triangle_quality_values(vertices, faces)
                <= float(maximum_source_quality)
            )
        usable &= rounded_faces[adjacency[:, 0]] & rounded_faces[adjacency[:, 1]]
    usable_adjacency = adjacency[usable]
    graph = coo_matrix(
        (
            np.ones(len(usable_adjacency) * 2, dtype=np.uint8),
            (
                np.concatenate((usable_adjacency[:, 0], usable_adjacency[:, 1])),
                np.concatenate((usable_adjacency[:, 1], usable_adjacency[:, 0])),
            ),
        ),
        shape=(len(faces), len(faces)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    kept_faces = np.ones(len(faces), dtype=bool)
    appended_vertices = []
    appended_faces = []
    candidates = 0
    accepted = 0
    removed_faces = 0
    old_qualities = []
    new_qualities = []
    claimed_faces = np.zeros(len(faces), dtype=bool)

    for component_id in np.flatnonzero(
        (component_sizes >= int(minimum_faces)) & (component_sizes <= 50000)
    ):
        face_ids = np.flatnonzero(labels == component_id)
        if np.any(claimed_faces[face_ids]):
            continue
        component_faces = faces[face_ids]
        component_edges = np.sort(
            component_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
            axis=1,
        )
        unique_edges, edge_counts = np.unique(
            component_edges, axis=0, return_counts=True
        )
        boundary_edges = unique_edges[edge_counts == 1]
        cycles = _ordered_cycles_from_edges(boundary_edges)
        if model_guided_boundary_recovery:
            cylinder_seed = _fit_component_cylinder(
                vertices,
                component_faces,
                radius_tolerance=min(float(radius_tolerance), 3e-2),
                normal_tolerance=min(float(normal_tolerance), 8e-2),
            )
            if cylinder_seed is not None:
                axis = cylinder_seed["axis"]
                origin = cylinder_seed["origin"]
                radius = float(cylinder_seed["radius"])
                radial_tolerance = max(
                    radius * min(float(radius_tolerance), 3e-2), 1e-8
                )
                selected = set(map(int, face_ids))
                visited = set(selected)
                pending = deque(selected)
                while pending:
                    current = pending.popleft()
                    begin = growth_graph.indptr[current]
                    end = growth_graph.indptr[current + 1]
                    for neighbor in growth_graph.indices[begin:end]:
                        neighbor = int(neighbor)
                        if neighbor in visited or claimed_faces[neighbor]:
                            continue
                        visited.add(neighbor)
                        offset = growth_centroids[neighbor] - origin
                        axial = float(np.dot(offset, axis))
                        radial = offset - axial * axis
                        radial_length = float(np.linalg.norm(radial))
                        if radial_length <= 0.0:
                            continue
                        normal = growth_normals[neighbor]
                        if abs(float(np.dot(normal, axis))) > min(
                            float(normal_tolerance), 8e-2
                        ):
                            continue
                        if abs(radial_length - radius) > radial_tolerance:
                            continue
                        alignment = abs(float(np.dot(normal, radial))) / radial_length
                        if alignment < max(float(minimum_radial_alignment), 0.98):
                            continue
                        selected.add(neighbor)
                        pending.append(neighbor)
                if len(selected) > len(face_ids):
                    face_ids = np.asarray(sorted(selected), dtype=np.int64)
                    component_faces = faces[face_ids]
                    component_edges = np.sort(
                        component_faces[
                            :, ((0, 1), (1, 2), (2, 0))
                        ].reshape(-1, 2),
                        axis=1,
                    )
                    unique_edges, edge_counts = np.unique(
                        component_edges, axis=0, return_counts=True
                    )
                    boundary_edges = unique_edges[edge_counts == 1]
                    cycles = _ordered_cycles_from_edges(boundary_edges)
        if cycles is None or len(cycles) != 1:
            continue
        candidates += 1
        boundary_set = {tuple(map(int, edge)) for edge in boundary_edges}
        if not isolate_rounded_faces and not boundary_set.issubset(separators):
            continue
        if (set(map(tuple, unique_edges)) & separators) - boundary_set:
            continue

        triangles = vertices[component_faces]
        crosses = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        cross_lengths = np.linalg.norm(crosses, axis=1)
        if np.any(cross_lengths <= 0.0):
            continue
        normals = crosses / cross_lengths[:, None]
        normal_covariance = normals.T @ normals / len(normals)
        eigenvalues, eigenvectors = np.linalg.eigh(normal_covariance)
        axis = eigenvectors[:, 0]
        if float(np.sqrt(np.mean((normals @ axis) ** 2))) > normal_tolerance:
            continue
        first_basis = normals[0] - axis * np.dot(normals[0], axis)
        first_basis_length = np.linalg.norm(first_basis)
        if first_basis_length <= 0.0:
            continue
        first_basis /= first_basis_length
        second_basis = np.cross(axis, first_basis)
        component_vertex_ids = np.unique(component_faces)
        origin = vertices[component_vertex_ids].mean(axis=0)
        offsets = vertices[component_vertex_ids] - origin
        projected = np.column_stack(
            (offsets @ first_basis, offsets @ second_basis)
        )
        circle_system = np.column_stack(
            (
                2.0 * projected[:, 0],
                2.0 * projected[:, 1],
                np.ones(len(projected)),
            )
        )
        circle_rhs = np.sum(projected * projected, axis=1)
        circle_solution, _, _, _ = np.linalg.lstsq(
            circle_system, circle_rhs, rcond=None
        )
        center_2d = circle_solution[:2]
        radial_lengths = np.linalg.norm(projected - center_2d, axis=1)
        radius = float(radial_lengths.mean())
        if radius <= 0.0 or float(radial_lengths.std() / radius) > radius_tolerance:
            continue
        axis_origin = origin + center_2d @ np.vstack((first_basis, second_basis))

        face_centroids = triangles.mean(axis=1)
        centroid_offsets = face_centroids - axis_origin
        centroid_axial = centroid_offsets @ axis
        centroid_radial = centroid_offsets - centroid_axial[:, None] * axis
        centroid_radial_lengths = np.linalg.norm(centroid_radial, axis=1)
        if np.any(centroid_radial_lengths <= 0.0):
            continue
        radial_alignment = np.abs(
            np.sum(normals * centroid_radial, axis=1) / centroid_radial_lengths
        )
        if (
            float(np.percentile(radial_alignment, 5.0))
            < float(minimum_radial_alignment)
        ):
            continue

        all_offsets = vertices[component_vertex_ids] - axis_origin
        all_angles = np.mod(
            np.arctan2(all_offsets @ second_basis, all_offsets @ first_basis),
            2.0 * np.pi,
        )
        sorted_angles = np.sort(all_angles)
        angular_gaps = np.diff(
            np.concatenate((sorted_angles, sorted_angles[:1] + 2.0 * np.pi))
        )
        largest_gap_index = int(np.argmax(angular_gaps))
        coverage = 2.0 * np.pi - float(angular_gaps[largest_gap_index])
        if coverage < np.deg2rad(float(minimum_angle_degrees)):
            continue
        if (
            maximum_angle_degrees is not None
            and coverage >= np.deg2rad(float(maximum_angle_degrees))
        ):
            continue
        normal_angles = np.mod(
            np.arctan2(normals @ second_basis, normals @ first_basis),
            2.0 * np.pi,
        )
        sorted_normal_angles = np.sort(normal_angles)
        normal_gaps = np.diff(
            np.concatenate(
                (sorted_normal_angles, sorted_normal_angles[:1] + 2.0 * np.pi)
            )
        )
        normal_coverage = 2.0 * np.pi - float(normal_gaps.max())
        if normal_coverage < np.deg2rad(float(minimum_angle_degrees) * 0.5):
            continue
        cut_angle = float(
            sorted_angles[largest_gap_index]
            + 0.5 * angular_gaps[largest_gap_index]
        )

        boundary_cycle = np.asarray(cycles[0], dtype=np.int64)
        boundary_offsets = vertices[boundary_cycle] - axis_origin
        boundary_angles = np.mod(
            np.arctan2(
                boundary_offsets @ second_basis,
                boundary_offsets @ first_basis,
            )
            - cut_angle,
            2.0 * np.pi,
        )
        minimum_unwrapped_angle = float(
            np.min(np.mod(all_angles - cut_angle, 2.0 * np.pi))
        )
        boundary_u = radius * (boundary_angles - minimum_unwrapped_angle)
        boundary_z = boundary_offsets @ axis
        minimum_z = float(boundary_z.min())
        boundary_parameters = np.column_stack(
            (boundary_u, boundary_z - minimum_z)
        )
        if abs(_polygon_signed_area(boundary_parameters)) <= 1e-12:
            continue
        segments = np.column_stack(
            (
                np.arange(len(boundary_cycle), dtype=np.int32),
                np.roll(np.arange(len(boundary_cycle), dtype=np.int32), -1),
            )
        )
        boundary_lengths = np.linalg.norm(
            vertices[boundary_cycle]
            - vertices[np.roll(boundary_cycle, -1)],
            axis=1,
        )
        spacing = float(np.median(boundary_lengths)) * float(target_edge_ratio)
        spacing = _uniform_target_spacing(
            spacing,
            preferred_edge_length,
            gradual=not bool(direct_target_spacing),
        )
        if not np.isfinite(spacing) or spacing <= 0.0:
            continue
        lower = boundary_parameters.min(axis=0)
        upper = boundary_parameters.max(axis=0)
        row_step = spacing * np.sqrt(3.0) * 0.5
        grid_points = []
        row_index = 0
        y = lower[1] + row_step
        segment_starts = boundary_parameters
        segment_vectors = np.roll(boundary_parameters, -1, axis=0) - boundary_parameters
        segment_length_squared = np.maximum(
            np.sum(segment_vectors * segment_vectors, axis=1),
            np.finfo(np.float64).eps,
        )
        while y < upper[1]:
            x_values = np.arange(
                lower[0] + spacing * (0.5 if row_index % 2 == 0 else 1.0),
                upper[0],
                spacing,
            )
            if len(x_values):
                row_points = np.column_stack((x_values, np.full(len(x_values), y)))
                inside_points = row_points[
                    _points_in_polygon(row_points, boundary_parameters)
                ]
                if len(inside_points):
                    differences = inside_points[:, None, :] - segment_starts[None, :, :]
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
                    distances = np.sqrt(
                        np.min(
                            np.sum((inside_points[:, None, :] - closest) ** 2, axis=2),
                            axis=1,
                        )
                    )
                    clearance_ratio = 0.3 if direct_target_spacing else 0.55
                    grid_points.extend(
                        inside_points[distances >= spacing * clearance_ratio]
                    )
            row_index += 1
            y += row_step

        parameter_vertices = np.vstack(
            (
                boundary_parameters,
                np.asarray(grid_points, dtype=np.float64).reshape(-1, 2),
            )
        )
        triangle_options = "pQY"
        if minimum_triangle_angle_degrees is not None:
            triangle_options = "pq{:.8g}QY".format(
                float(minimum_triangle_angle_degrees)
            )
        triangle_result = constrained_triangle.triangulate(
            {"vertices": parameter_vertices, "segments": segments},
            triangle_options,
        )
        if "triangles" not in triangle_result:
            continue
        result_parameters = np.asarray(triangle_result["vertices"], dtype=np.float64)
        local_faces = np.asarray(triangle_result["triangles"], dtype=np.int64)
        if (
            maximum_face_count_ratio is not None
            and len(local_faces)
            > len(component_faces) * float(maximum_face_count_ratio)
        ):
            continue
        new_parameters = result_parameters[len(boundary_cycle):]
        new_angles = (
            new_parameters[:, 0] / radius
            + minimum_unwrapped_angle
            + cut_angle
        )
        new_axial = new_parameters[:, 1] + minimum_z
        new_vertices = (
            axis_origin
            + new_axial[:, None] * axis
            + radius
            * (
                np.cos(new_angles)[:, None] * first_basis
                + np.sin(new_angles)[:, None] * second_basis
            )
        )
        if len(new_vertices):
            _, _, new_vertices = igl.point_mesh_squared_distance(
                new_vertices, vertices, component_faces
            )
        new_vertices = np.asarray(
            new_vertices, dtype=np.float64
        ).reshape(-1, 3)
        first_new_index = len(vertices) + len(appended_vertices)
        local_to_global = np.concatenate(
            (
                boundary_cycle,
                np.arange(
                    first_new_index,
                    first_new_index + len(new_vertices),
                    dtype=np.int64,
                ),
            )
        )
        new_faces = local_to_global[local_faces]
        if np.any(
            (new_faces[:, 0] == new_faces[:, 1])
            | (new_faces[:, 1] == new_faces[:, 2])
            | (new_faces[:, 2] == new_faces[:, 0])
        ):
            continue
        coordinate_pool = np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
                new_vertices,
            )
        )
        if maximum_result_edge_length is not None:
            result_triangles = coordinate_pool[new_faces]
            result_lengths = np.linalg.norm(
                result_triangles[:, (1, 2, 0)]
                - result_triangles[:, (0, 1, 2)],
                axis=2,
            )
            if (
                len(result_lengths)
                and float(result_lengths.max())
                > float(maximum_result_edge_length) * (1.0 + 1e-8)
            ):
                continue
        new_edge_array = np.sort(
            new_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1
        )
        new_unique_edges, new_edge_counts = np.unique(
            new_edge_array, axis=0, return_counts=True
        )
        new_edge_count_map = {
            tuple(map(int, edge)): int(count)
            for edge, count in zip(new_unique_edges, new_edge_counts)
        }
        if any(new_edge_count_map.get(edge, 0) != 1 for edge in boundary_set):
            continue
        if any(
            count != (1 if edge in boundary_set else 2)
            for edge, count in new_edge_count_map.items()
        ):
            continue

        def boundary_directions(face_array):
            directions = {}
            for face in face_array:
                for first, second in (
                    (int(face[0]), int(face[1])),
                    (int(face[1]), int(face[2])),
                    (int(face[2]), int(face[0])),
                ):
                    edge = tuple(sorted((first, second)))
                    if edge in boundary_set:
                        directions[edge] = (first, second)
            return directions

        old_directions = boundary_directions(component_faces)
        new_directions = boundary_directions(new_faces)
        if set(old_directions) != boundary_set or set(new_directions) != boundary_set:
            continue
        matches = np.asarray(
            [old_directions[edge] == new_directions[edge] for edge in boundary_set]
        )
        if not np.all(matches):
            if np.all(~matches):
                new_faces = new_faces[:, [0, 2, 1]]
            else:
                continue
        old_quality = _triangle_quality_values(vertices, component_faces)
        new_quality = _triangle_quality_values(coordinate_pool, new_faces)
        component_area = 0.5 * float(cross_lengths.sum())
        density_excess = False
        if preferred_edge_length is not None:
            target_area = (
                np.sqrt(3.0) * float(preferred_edge_length) ** 2 * 0.25
            )
            target_faces = max(
                component_area / max(target_area, 1e-30),
                float(max(len(boundary_cycle) - 2, 1)),
            )
            density_excess = len(component_faces) > target_faces * 2.0
        safe_density_simplification = (
            bool(allow_density_simplification)
            and density_excess
            and len(new_faces) <= len(component_faces) * 0.8
            and float(new_quality.mean()) >= 0.45
            and float(np.percentile(new_quality, 5.0)) >= 0.12
        )
        if (
            (
                np.percentile(new_quality, 5.0)
                < np.percentile(old_quality, 5.0) - 1e-8
                or float(new_quality.mean()) < float(old_quality.mean()) - 1e-8
            )
            and not safe_density_simplification
        ):
            continue

        kept_faces[face_ids] = False
        claimed_faces[face_ids] = True
        appended_vertices.extend(new_vertices)
        appended_faces.extend(new_faces)
        old_qualities.extend(old_quality)
        new_qualities.extend(new_quality)
        accepted += 1
        removed_faces += len(face_ids)

    if not accepted:
        return vertices.copy(), faces.copy(), {
            "candidates": candidates, "patches": 0, "new_vertices": 0,
            "removed_faces": 0, "new_faces": 0,
            "old_quality": 0.0, "new_quality": 0.0,
            "old_quality_p5": 0.0, "new_quality_p5": 0.0,
        }
    return (
        np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
            )
        ),
        np.vstack((faces[kept_faces], np.asarray(appended_faces, dtype=np.int64))),
        {
            "candidates": candidates,
            "patches": accepted,
            "new_vertices": len(appended_vertices),
            "removed_faces": removed_faces,
            "new_faces": len(appended_faces),
            "old_quality": float(np.mean(old_qualities)),
            "new_quality": float(np.mean(new_qualities)),
            "old_quality_p5": float(np.percentile(old_qualities, 5.0)),
            "new_quality_p5": float(np.percentile(new_qualities, 5.0)),
        },
    )


def _fit_component_cylinder(vertices, component_faces, radius_tolerance,
                            normal_tolerance):
    """Fit an axis and radius without assuming planar circular end loops."""
    triangles = vertices[component_faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    cross_lengths = np.linalg.norm(crosses, axis=1)
    if np.any(cross_lengths <= 0.0):
        return None
    normals = crosses / cross_lengths[:, None]
    _, eigenvectors = np.linalg.eigh(normals.T @ normals / len(normals))
    axis = eigenvectors[:, 0]
    normal_error = float(np.sqrt(np.mean((normals @ axis) ** 2)))
    if normal_error > float(normal_tolerance):
        return None
    first_basis = normals[0] - axis * np.dot(normals[0], axis)
    basis_length = np.linalg.norm(first_basis)
    if basis_length <= 0.0:
        return None
    first_basis /= basis_length
    second_basis = np.cross(axis, first_basis)
    vertex_ids = np.unique(component_faces)
    origin = vertices[vertex_ids].mean(axis=0)
    offsets = vertices[vertex_ids] - origin
    projected = np.column_stack((offsets @ first_basis, offsets @ second_basis))
    system = np.column_stack(
        (2.0 * projected[:, 0], 2.0 * projected[:, 1], np.ones(len(projected)))
    )
    solution, _, _, _ = np.linalg.lstsq(
        system, np.sum(projected * projected, axis=1), rcond=None
    )
    center_2d = solution[:2]
    radii = np.linalg.norm(projected - center_2d, axis=1)
    radius = float(radii.mean())
    if radius <= 0.0 or float(radii.std() / radius) > float(radius_tolerance):
        return None
    axis_origin = origin + center_2d[0] * first_basis + center_2d[1] * second_basis
    centroid_offsets = triangles.mean(axis=1) - axis_origin
    centroid_axial = centroid_offsets @ axis
    centroid_radial = centroid_offsets - centroid_axial[:, None] * axis
    radial_lengths = np.linalg.norm(centroid_radial, axis=1)
    if np.any(radial_lengths <= 0.0):
        return None
    alignment = np.abs(np.sum(normals * centroid_radial, axis=1) / radial_lengths)
    if float(np.percentile(alignment, 5.0)) < 0.98:
        return None
    return {
        "axis": axis,
        "origin": axis_origin,
        "first_basis": first_basis,
        "second_basis": second_basis,
        "radius": radius,
        "normal_error": normal_error,
    }


def _unwrap_closed_cylinder_cycle(vertices, cycle, cylinder):
    """Return a connectivity-preserving, increasing one-turn (u, z) loop."""
    cycle = np.asarray(cycle, dtype=np.int64)
    offsets = vertices[cycle] - cylinder["origin"]
    angles = np.arctan2(
        offsets @ cylinder["second_basis"],
        offsets @ cylinder["first_basis"],
    )
    closed_deltas = np.arctan2(
        np.sin(np.roll(angles, -1) - angles),
        np.cos(np.roll(angles, -1) - angles),
    )
    winding = float(closed_deltas.sum())
    if not 1.5 * np.pi <= abs(winding) <= 2.5 * np.pi:
        return None
    if winding < 0.0:
        cycle = cycle[::-1]
        offsets = vertices[cycle] - cylinder["origin"]
        angles = np.arctan2(
            offsets @ cylinder["second_basis"],
            offsets @ cylinder["first_basis"],
        )
    deltas = np.arctan2(
        np.sin(angles[1:] - angles[:-1]),
        np.cos(angles[1:] - angles[:-1]),
    )
    # A trimmed cylinder boundary may wiggle slightly, but it must remain a graph
    # in the periodic angular coordinate to form a safe, non-self-intersecting strip.
    if len(deltas) and float(deltas.min()) < -1e-5:
        return None
    u = cylinder["radius"] * np.concatenate(([0.0], np.cumsum(deltas)))
    circumference = 2.0 * np.pi * cylinder["radius"]
    if u[-1] >= circumference or u[-1] < circumference * 0.5:
        return None
    return {
        "cycle": cycle,
        "u": u,
        "z": offsets @ cylinder["axis"],
        "start_angle": float(angles[0]),
    }


def retriangulate_trimmed_cylindrical_walls(
    vertices,
    faces,
    protected_edges,
    minimum_faces=20,
    radius_tolerance=1e-2,
    normal_tolerance=8e-2,
    target_edge_ratio=1.0,
    regular_radius_tolerance=1e-3,
    preferred_edge_length=None,
):
    """Remesh Boolean-trimmed cylinders bounded by two irregular closed loops."""
    if constrained_triangle is None:
        raise RuntimeError(
            "Trimmed-cylinder constrained remeshing needs the 'triangle' package."
        )
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64), axis=1
    )
    usable = np.asarray(
        [tuple(map(int, edge)) not in protected for edge in adjacency_edges]
    )
    usable_adjacency = adjacency[usable]
    graph = coo_matrix(
        (
            np.ones(len(usable_adjacency) * 2, dtype=np.uint8),
            (
                np.concatenate((usable_adjacency[:, 0], usable_adjacency[:, 1])),
                np.concatenate((usable_adjacency[:, 1], usable_adjacency[:, 0])),
            ),
        ),
        shape=(len(faces), len(faces)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    kept_faces = np.ones(len(faces), dtype=bool)
    appended_vertices = []
    appended_faces = []
    candidates = accepted = removed_faces = 0
    old_qualities = []
    new_qualities = []

    for component_id in np.flatnonzero(
        (component_sizes >= int(minimum_faces)) & (component_sizes <= 50000)
    ):
        face_ids = np.flatnonzero(labels == component_id)
        component_faces = faces[face_ids]
        component_edges = np.sort(
            component_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1
        )
        unique_edges, edge_counts = np.unique(
            component_edges, axis=0, return_counts=True
        )
        boundary_edges = unique_edges[edge_counts == 1]
        cycles = _ordered_cycles_from_edges(boundary_edges)
        if cycles is None or len(cycles) != 2 or set(cycles[0]) & set(cycles[1]):
            continue
        candidates += 1
        boundary_set = {tuple(map(int, edge)) for edge in boundary_edges}
        if not boundary_set.issubset(protected):
            continue
        if (set(map(tuple, unique_edges)) & protected) - boundary_set:
            continue
        # Regular circular ends are handled earlier by the stricter fast path.
        circle_fits = [_fit_circle_loop(vertices[cycle]) for cycle in cycles]
        if all(
            fit is not None
            and fit["plane_error"] <= regular_radius_tolerance
            and fit["radius_error"] <= regular_radius_tolerance
            for fit in circle_fits
        ):
            continue
        cylinder = _fit_component_cylinder(
            vertices, component_faces, radius_tolerance, normal_tolerance
        )
        if cylinder is None:
            continue
        unwrapped = [
            _unwrap_closed_cylinder_cycle(vertices, cycle, cylinder)
            for cycle in cycles
        ]
        if any(loop is None for loop in unwrapped):
            continue
        unwrapped.sort(key=lambda loop: float(np.mean(loop["z"])))
        lower, upper = unwrapped
        circumference = 2.0 * np.pi * cylinder["radius"]
        sample_u = np.linspace(0.0, circumference, 257)
        lower_z = np.interp(
            sample_u,
            np.append(lower["u"], circumference),
            np.append(lower["z"], lower["z"][0]),
        )
        upper_z = np.interp(
            sample_u,
            np.append(upper["u"], circumference),
            np.append(upper["z"], upper["z"][0]),
        )
        if float(np.min(upper_z - lower_z)) <= 1e-8:
            continue

        boundary_lengths = np.linalg.norm(
            vertices[boundary_edges[:, 0]] - vertices[boundary_edges[:, 1]], axis=1
        )
        spacing = float(np.median(boundary_lengths)) * float(target_edge_ratio)
        spacing = _uniform_target_spacing(spacing, preferred_edge_length)
        if not np.isfinite(spacing) or spacing <= 0.0:
            continue
        circumferential_count = max(
            3, int(np.round(circumference / spacing))
        )
        periodic_spacing = circumference / circumferential_count
        row_step = periodic_spacing * np.sqrt(3.0) * 0.5
        seam_low = float(lower["z"][0])
        seam_high = float(upper["z"][0])
        seam_z = np.arange(seam_low + row_step, seam_high - row_step * 0.25, row_step)

        parameter_vertices = []
        boundary_mapping = []
        parameter_vertices.extend(np.column_stack((lower["u"], lower["z"])))
        boundary_mapping.extend(lower["cycle"])
        parameter_vertices.append((circumference, seam_low))
        boundary_mapping.append(int(lower["cycle"][0]))
        right_seam_indices = []
        for z_value in seam_z:
            right_seam_indices.append(len(parameter_vertices))
            parameter_vertices.append((circumference, z_value))
            boundary_mapping.append(-1)
        parameter_vertices.append((circumference, seam_high))
        boundary_mapping.append(int(upper["cycle"][0]))
        parameter_vertices.extend(
            np.column_stack((upper["u"][::-1], upper["z"][::-1]))
        )
        boundary_mapping.extend(upper["cycle"][::-1])
        left_seam_indices = []
        for z_value in seam_z[::-1]:
            left_seam_indices.append(len(parameter_vertices))
            parameter_vertices.append((0.0, z_value))
            boundary_mapping.append(-1)
        parameter_vertices = np.asarray(parameter_vertices, dtype=np.float64)
        boundary_mapping = np.asarray(boundary_mapping, dtype=np.int64)
        segments = np.column_stack(
            (
                np.arange(len(parameter_vertices), dtype=np.int32),
                np.roll(np.arange(len(parameter_vertices), dtype=np.int32), -1),
            )
        )

        lower_bound = parameter_vertices.min(axis=0)
        upper_bound = parameter_vertices.max(axis=0)
        segment_vectors = np.roll(parameter_vertices, -1, axis=0) - parameter_vertices
        segment_lengths_squared = np.maximum(
            np.sum(segment_vectors * segment_vectors, axis=1),
            np.finfo(np.float64).eps,
        )
        grid_points = []
        row = 0
        y = lower_bound[1] + row_step
        while y < upper_bound[1]:
            if row % 2 == 0:
                x_values = (
                    np.arange(circumferential_count, dtype=np.float64) + 0.5
                ) * periodic_spacing
            else:
                x_values = (
                    np.arange(1, circumferential_count, dtype=np.float64)
                    * periodic_spacing
                )
            if len(x_values):
                points = np.column_stack((x_values, np.full(len(x_values), y)))
                points = points[_points_in_polygon(points, parameter_vertices)]
                if len(points):
                    differences = points[:, None, :] - parameter_vertices[None, :, :]
                    fractions = np.clip(
                        np.sum(differences * segment_vectors[None, :, :], axis=2)
                        / segment_lengths_squared[None, :],
                        0.0,
                        1.0,
                    )
                    closest = (
                        parameter_vertices[None, :, :]
                        + fractions[:, :, None] * segment_vectors[None, :, :]
                    )
                    distances = np.sqrt(
                        np.min(np.sum((points[:, None, :] - closest) ** 2, axis=2), axis=1)
                    )
                    grid_points.extend(
                        points[distances >= periodic_spacing * 0.55]
                    )
            row += 1
            y += row_step
        all_parameters = np.vstack(
            (parameter_vertices, np.asarray(grid_points).reshape(-1, 2))
        )
        triangle_result = constrained_triangle.triangulate(
            {"vertices": all_parameters, "segments": segments}, "pQY"
        )
        if "triangles" not in triangle_result:
            continue
        result_parameters = np.asarray(triangle_result["vertices"], dtype=np.float64)
        local_faces = np.asarray(triangle_result["triangles"], dtype=np.int64)
        interior_parameters = result_parameters[len(parameter_vertices):]
        new_parameters = np.vstack(
            (np.column_stack((np.zeros(len(seam_z)), seam_z)), interior_parameters)
        )
        wrapped_u = np.mod(new_parameters[:, 0], circumference)
        lower_at_u = np.interp(
            wrapped_u,
            np.append(lower["u"], circumference),
            np.append(lower["z"], lower["z"][0]),
        )
        upper_at_u = np.interp(
            wrapped_u,
            np.append(upper["u"], circumference),
            np.append(upper["z"], upper["z"][0]),
        )
        fractions = np.clip(
            (new_parameters[:, 1] - lower_at_u)
            / np.maximum(upper_at_u - lower_at_u, 1e-30),
            0.0,
            1.0,
        )
        seam_delta = np.arctan2(
            np.sin(upper["start_angle"] - lower["start_angle"]),
            np.cos(upper["start_angle"] - lower["start_angle"]),
        )
        angles = (
            wrapped_u / cylinder["radius"]
            + lower["start_angle"]
            + fractions * seam_delta
        )
        new_vertices = (
            cylinder["origin"]
            + new_parameters[:, 1, None] * cylinder["axis"]
            + cylinder["radius"]
            * (
                np.cos(angles)[:, None] * cylinder["first_basis"]
                + np.sin(angles)[:, None] * cylinder["second_basis"]
            )
        )
        if len(new_vertices):
            _, _, new_vertices = igl.point_mesh_squared_distance(
                new_vertices, vertices, component_faces
            )
            new_vertices = np.asarray(new_vertices, dtype=np.float64)
        first_new = len(vertices) + len(appended_vertices)
        local_to_global = np.empty(len(result_parameters), dtype=np.int64)
        existing = boundary_mapping >= 0
        local_to_global[:len(parameter_vertices)][existing] = boundary_mapping[existing]
        seam_ids = np.arange(first_new, first_new + len(seam_z), dtype=np.int64)
        for seam_index, local_index in enumerate(right_seam_indices):
            local_to_global[local_index] = seam_ids[seam_index]
        for seam_index, local_index in enumerate(reversed(left_seam_indices)):
            local_to_global[local_index] = seam_ids[seam_index]
        local_to_global[len(parameter_vertices):] = np.arange(
            first_new + len(seam_z), first_new + len(new_vertices), dtype=np.int64
        )
        new_faces = local_to_global[local_faces]
        if np.any(
            (new_faces[:, 0] == new_faces[:, 1])
            | (new_faces[:, 1] == new_faces[:, 2])
            | (new_faces[:, 2] == new_faces[:, 0])
        ):
            continue
        coordinate_pool = np.vstack(
            (vertices, np.asarray(appended_vertices).reshape(-1, 3), new_vertices)
        )
        new_triangles = coordinate_pool[new_faces]
        new_crosses = np.cross(
            new_triangles[:, 1] - new_triangles[:, 0],
            new_triangles[:, 2] - new_triangles[:, 0],
        )
        old_triangles = vertices[component_faces]
        old_crosses = np.cross(
            old_triangles[:, 1] - old_triangles[:, 0],
            old_triangles[:, 2] - old_triangles[:, 0],
        )
        old_offsets = old_triangles.mean(axis=1) - cylinder["origin"]
        old_radial = old_offsets - (old_offsets @ cylinder["axis"])[:, None] * cylinder["axis"]
        orientation = np.sign(np.sum(old_crosses * old_radial, axis=1).mean())
        new_offsets = new_triangles.mean(axis=1) - cylinder["origin"]
        new_radial = new_offsets - (new_offsets @ cylinder["axis"])[:, None] * cylinder["axis"]
        reverse = np.sum(new_crosses * new_radial, axis=1) * orientation < 0.0
        new_faces[reverse] = new_faces[reverse][:, [0, 2, 1]]

        new_edges = np.sort(
            new_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1
        )
        new_unique, new_counts = np.unique(new_edges, axis=0, return_counts=True)
        count_map = {
            tuple(map(int, edge)): int(count)
            for edge, count in zip(new_unique, new_counts)
        }
        if any(count_map.get(edge, 0) != 1 for edge in boundary_set):
            continue
        if any(
            count != (1 if edge in boundary_set else 2)
            for edge, count in count_map.items()
        ):
            continue

        def boundary_directions(face_array):
            directions = {}
            for face in face_array:
                for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    edge = tuple(sorted((int(first), int(second))))
                    if edge in boundary_set:
                        directions[edge] = (int(first), int(second))
            return directions

        old_directions = boundary_directions(component_faces)
        new_directions = boundary_directions(new_faces)
        if set(old_directions) != boundary_set or set(new_directions) != boundary_set:
            continue
        matches = np.asarray(
            [old_directions[edge] == new_directions[edge] for edge in boundary_set]
        )
        if not np.all(matches):
            if np.all(~matches):
                new_faces = new_faces[:, [0, 2, 1]]
            else:
                continue
        old_quality = _triangle_quality_values(vertices, component_faces)
        new_quality = _triangle_quality_values(coordinate_pool, new_faces)
        if (
            np.percentile(new_quality, 5.0) < np.percentile(old_quality, 5.0) - 1e-8
            or float(new_quality.mean()) < float(old_quality.mean()) - 1e-8
        ):
            continue
        kept_faces[face_ids] = False
        appended_vertices.extend(new_vertices)
        appended_faces.extend(new_faces)
        old_qualities.extend(old_quality)
        new_qualities.extend(new_quality)
        accepted += 1
        removed_faces += len(face_ids)

    if not accepted:
        return vertices.copy(), faces.copy(), {
            "candidates": candidates, "walls": 0, "new_vertices": 0,
            "removed_faces": 0, "new_faces": 0,
            "old_quality": 0.0, "new_quality": 0.0,
        }
    return (
        np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
            )
        ),
        np.vstack((faces[kept_faces], np.asarray(appended_faces, dtype=np.int64))),
        {
            "candidates": candidates, "walls": accepted,
            "new_vertices": len(appended_vertices), "removed_faces": removed_faces,
            "new_faces": len(appended_faces),
            "old_quality": float(np.mean(old_qualities)),
            "new_quality": float(np.mean(new_qualities)),
        },
    )


def retriangulate_cylindrical_walls(
    vertices,
    faces,
    protected_edges,
    minimum_faces=20,
    radius_tolerance=1e-3,
    target_edge_ratio=1.0,
    preferred_edge_length=None,
    direct_target_spacing=False,
    maximum_face_count_ratio=None,
):
    """Remesh safely detected cylindrical strips in an unwrapped parameter domain."""
    if constrained_triangle is None:
        raise RuntimeError(
            "Cylindrical constrained remeshing needs the 'triangle' package."
        )
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    radius_tolerance = float(radius_tolerance)
    target_edge_ratio = float(target_edge_ratio)
    if radius_tolerance <= 0.0:
        raise ValueError("Cylinder radius tolerance must be positive.")
    if target_edge_ratio <= 0.0:
        raise ValueError("Cylinder target edge ratio must be positive.")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64), axis=1
    )
    usable = np.asarray(
        [tuple(map(int, edge)) not in protected for edge in adjacency_edges]
    )
    usable_adjacency = adjacency[usable]
    graph = coo_matrix(
        (
            np.ones(len(usable_adjacency) * 2, dtype=np.uint8),
            (
                np.concatenate((usable_adjacency[:, 0], usable_adjacency[:, 1])),
                np.concatenate((usable_adjacency[:, 1], usable_adjacency[:, 0])),
            ),
        ),
        shape=(len(faces), len(faces)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    kept_faces = np.ones(len(faces), dtype=bool)
    appended_vertices = []
    appended_faces = []
    candidates = 0
    accepted = 0
    removed_faces = 0
    old_qualities = []
    new_qualities = []

    component_ids = np.flatnonzero(
        (component_sizes >= int(minimum_faces)) & (component_sizes <= 50000)
    )
    for component_id in component_ids:
        face_ids = np.flatnonzero(labels == component_id)
        component_faces = faces[face_ids]
        component_edges = np.sort(
            component_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
            axis=1,
        )
        unique_edges, edge_counts = np.unique(
            component_edges, axis=0, return_counts=True
        )
        boundary_edges = unique_edges[edge_counts == 1]
        cycles = _ordered_cycles_from_edges(boundary_edges)
        if cycles is None or len(cycles) != 2:
            continue
        if set(cycles[0]) & set(cycles[1]):
            continue
        candidates += 1
        boundary_set = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in boundary_edges
        }
        if not boundary_set.issubset(protected):
            continue
        if (set(map(tuple, unique_edges)) & protected) - boundary_set:
            continue

        fits = [_fit_circle_loop(vertices[cycle]) for cycle in cycles]
        if any(fit is None for fit in fits):
            continue
        if any(
            fit["plane_error"] > radius_tolerance
            or fit["radius_error"] > radius_tolerance
            for fit in fits
        ):
            continue
        center_delta = fits[1]["center"] - fits[0]["center"]
        height = float(np.linalg.norm(center_delta))
        if height <= 0.0:
            continue
        axis = center_delta / height
        if any(abs(np.dot(fit["normal"], axis)) < 1.0 - radius_tolerance for fit in fits):
            continue
        mean_radius = 0.5 * (fits[0]["radius"] + fits[1]["radius"])
        if abs(fits[0]["radius"] - fits[1]["radius"]) > mean_radius * radius_tolerance:
            continue
        component_vertex_ids = np.unique(component_faces)
        offsets = vertices[component_vertex_ids] - fits[0]["center"]
        axial_coordinates = offsets @ axis
        radial_vectors = offsets - axial_coordinates[:, None] * axis
        radial_lengths = np.linalg.norm(radial_vectors, axis=1)
        if (
            float(radial_lengths.std() / max(radial_lengths.mean(), 1e-30))
            > radius_tolerance
        ):
            continue

        first_radial = vertices[cycles[0][0]] - fits[0]["center"]
        first_radial -= axis * np.dot(first_radial, axis)
        first_radial_length = np.linalg.norm(first_radial)
        if first_radial_length <= 0.0:
            continue
        provisional_first_basis = first_radial / first_radial_length
        provisional_second_basis = np.cross(axis, provisional_first_basis)

        # Do not place the periodic seam at an arbitrary ring vertex.  A very
        # short wrap edge or a phase mismatch between the two end rings gets
        # propagated through the full wall as a column of sliver triangles.
        # Choose a lower/upper vertex pair whose preceding angular gaps are
        # both representative and whose phases agree.
        ring_angle_data = []
        for cycle, fit in zip(cycles, fits):
            radial = vertices[cycle] - fit["center"]
            angles = np.mod(
                np.arctan2(
                    radial @ provisional_second_basis,
                    radial @ provisional_first_basis,
                ),
                2.0 * np.pi,
            )
            order = np.argsort(angles)
            sorted_ring_angles = angles[order]
            gaps = np.diff(
                np.concatenate(
                    (sorted_ring_angles, sorted_ring_angles[:1] + 2.0 * np.pi)
                )
            )
            positive_gaps = gaps[gaps > 1e-12]
            if len(positive_gaps) == 0:
                ring_angle_data = []
                break
            ring_angle_data.append(
                {
                    "cycle": np.asarray(cycle, dtype=np.int64)[order],
                    "angles": sorted_ring_angles,
                    "gaps": gaps,
                    "median_gap": float(np.median(positive_gaps)),
                }
            )
        if len(ring_angle_data) != 2:
            continue
        lower_ring, upper_ring = ring_angle_data
        best_seam = None
        for lower_index, lower_angle in enumerate(lower_ring["angles"]):
            phase_delta = np.abs(
                np.arctan2(
                    np.sin(upper_ring["angles"] - lower_angle),
                    np.cos(upper_ring["angles"] - lower_angle),
                )
            )
            upper_index = int(np.argmin(phase_delta))
            lower_gap = max(
                float(lower_ring["gaps"][(lower_index - 1) % len(lower_ring["gaps"])]),
                1e-30,
            )
            upper_gap = max(
                float(upper_ring["gaps"][(upper_index - 1) % len(upper_ring["gaps"])]),
                1e-30,
            )
            gap_cost = abs(
                np.log(lower_gap / lower_ring["median_gap"])
            ) + abs(np.log(upper_gap / upper_ring["median_gap"]))
            phase_cost = float(phase_delta[upper_index]) / max(
                min(lower_ring["median_gap"], upper_ring["median_gap"]),
                1e-30,
            )
            cost = gap_cost + phase_cost
            if best_seam is None or cost < best_seam[0]:
                best_seam = (cost, lower_index)
        seam_vertex = int(lower_ring["cycle"][best_seam[1]])
        first_radial = vertices[seam_vertex] - fits[0]["center"]
        first_radial -= axis * np.dot(first_radial, axis)
        first_basis = first_radial / np.linalg.norm(first_radial)
        second_basis = np.cross(axis, first_basis)
        sorted_cycles = []
        sorted_angles = []
        for cycle, fit in zip(cycles, fits):
            radial = vertices[cycle] - fit["center"]
            angles = np.mod(
                np.arctan2(radial @ second_basis, radial @ first_basis),
                2.0 * np.pi,
            )
            order = np.argsort(angles)
            ordered_cycle = np.asarray(cycle, dtype=np.int64)[order]
            ordered_angles = angles[order]
            ordered_angles = np.mod(ordered_angles - ordered_angles[0], 2.0 * np.pi)
            sorted_cycles.append(ordered_cycle)
            sorted_angles.append(ordered_angles)

        circumference = 2.0 * np.pi * mean_radius
        lower_u = sorted_angles[0] * mean_radius
        upper_u = sorted_angles[1] * mean_radius
        boundary_lengths = np.concatenate(
            (
                np.linalg.norm(
                    vertices[sorted_cycles[0]]
                    - vertices[np.roll(sorted_cycles[0], -1)], axis=1
                ),
                np.linalg.norm(
                    vertices[sorted_cycles[1]]
                    - vertices[np.roll(sorted_cycles[1], -1)], axis=1
                ),
            )
        )
        spacing = float(np.median(boundary_lengths)) * target_edge_ratio
        spacing = _uniform_target_spacing(
            spacing,
            preferred_edge_length,
            gradual=not bool(direct_target_spacing),
        )
        if not np.isfinite(spacing) or spacing <= 0.0:
            continue
        circumferential_count = max(
            3, int(np.round(circumference / spacing))
        )
        periodic_spacing = circumference / circumferential_count
        row_step = periodic_spacing * np.sqrt(3.0) * 0.5
        seam_heights = []
        y = row_step
        while y < height - row_step * 0.25:
            seam_heights.append(y)
            y += row_step

        parameter_vertices = []
        boundary_mapping = []
        parameter_vertices.extend(np.column_stack((lower_u, np.zeros(len(lower_u)))))
        boundary_mapping.extend(sorted_cycles[0])
        parameter_vertices.append((circumference, 0.0))
        boundary_mapping.append(int(sorted_cycles[0][0]))
        right_seam_indices = []
        for seam_height in seam_heights:
            right_seam_indices.append(len(parameter_vertices))
            parameter_vertices.append((circumference, seam_height))
            boundary_mapping.append(-1)
        parameter_vertices.append((circumference, height))
        boundary_mapping.append(int(sorted_cycles[1][0]))
        parameter_vertices.extend(
            np.column_stack((upper_u[::-1], np.full(len(upper_u), height)))
        )
        boundary_mapping.extend(sorted_cycles[1][::-1])
        left_seam_indices = []
        for seam_height in reversed(seam_heights):
            left_seam_indices.append(len(parameter_vertices))
            parameter_vertices.append((0.0, seam_height))
            boundary_mapping.append(-1)
        parameter_vertices = np.asarray(parameter_vertices, dtype=np.float64)
        boundary_mapping = np.asarray(boundary_mapping, dtype=np.int64)
        segments = np.column_stack(
            (
                np.arange(len(parameter_vertices), dtype=np.int32),
                np.roll(np.arange(len(parameter_vertices), dtype=np.int32), -1),
            )
        )

        grid_points = []
        row_index = 0
        for y in seam_heights:
            if row_index % 2 == 0:
                x_values = (
                    np.arange(circumferential_count, dtype=np.float64) + 0.5
                ) * periodic_spacing
            else:
                x_values = (
                    np.arange(1, circumferential_count, dtype=np.float64)
                    * periodic_spacing
                )
            grid_points.extend((float(x), y) for x in x_values)
            row_index += 1
        all_parameter_vertices = np.vstack(
            (
                parameter_vertices,
                np.asarray(grid_points, dtype=np.float64).reshape(-1, 2),
            )
        )
        triangle_result = constrained_triangle.triangulate(
            {"vertices": all_parameter_vertices, "segments": segments},
            "pQY",
        )
        if "triangles" not in triangle_result:
            continue
        result_parameters = np.asarray(
            triangle_result["vertices"], dtype=np.float64
        )
        local_faces = np.asarray(triangle_result["triangles"], dtype=np.int64)
        if (
            maximum_face_count_ratio is not None
            and len(local_faces)
            > len(component_faces) * float(maximum_face_count_ratio)
        ):
            continue
        interior_parameters = result_parameters[len(parameter_vertices):]
        seam_parameters = np.column_stack(
            (
                np.zeros(len(seam_heights), dtype=np.float64),
                np.asarray(seam_heights, dtype=np.float64),
            )
        )
        new_parameters = np.vstack((seam_parameters, interior_parameters))
        axial_fraction = np.clip(new_parameters[:, 1] / height, 0.0, 1.0)
        lower_seam_angle = float(
            np.arctan2(
                (vertices[sorted_cycles[0][0]] - fits[0]["center"]) @ second_basis,
                (vertices[sorted_cycles[0][0]] - fits[0]["center"]) @ first_basis,
            )
        )
        upper_seam_angle = float(
            np.arctan2(
                (vertices[sorted_cycles[1][0]] - fits[1]["center"]) @ second_basis,
                (vertices[sorted_cycles[1][0]] - fits[1]["center"]) @ first_basis,
            )
        )
        seam_delta = np.arctan2(
            np.sin(upper_seam_angle - lower_seam_angle),
            np.cos(upper_seam_angle - lower_seam_angle),
        )
        angles = (
            new_parameters[:, 0] / mean_radius
            + lower_seam_angle
            + axial_fraction * seam_delta
        )
        radii = (
            fits[0]["radius"] * (1.0 - axial_fraction)
            + fits[1]["radius"] * axial_fraction
        )
        centers = (
            fits[0]["center"][None, :] * (1.0 - axial_fraction[:, None])
            + fits[1]["center"][None, :] * axial_fraction[:, None]
        )
        new_vertices = centers + radii[:, None] * (
            np.cos(angles)[:, None] * first_basis
            + np.sin(angles)[:, None] * second_basis
        )
        if len(new_vertices):
            _, _, new_vertices = igl.point_mesh_squared_distance(
                new_vertices,
                vertices,
                component_faces,
            )
            new_vertices = np.asarray(new_vertices, dtype=np.float64)
        first_new_index = len(vertices) + len(appended_vertices)
        local_to_global = np.empty(len(result_parameters), dtype=np.int64)
        existing_boundary = boundary_mapping >= 0
        local_to_global[:len(parameter_vertices)][existing_boundary] = (
            boundary_mapping[existing_boundary]
        )
        seam_global_ids = np.arange(
            first_new_index,
            first_new_index + len(seam_heights),
            dtype=np.int64,
        )
        for seam_index, local_index in enumerate(right_seam_indices):
            local_to_global[local_index] = seam_global_ids[seam_index]
        for seam_index, local_index in enumerate(reversed(left_seam_indices)):
            local_to_global[local_index] = seam_global_ids[seam_index]
        local_to_global[len(parameter_vertices):] = np.arange(
            first_new_index + len(seam_heights),
            first_new_index + len(new_vertices),
            dtype=np.int64,
        )
        new_faces = local_to_global[local_faces]
        if np.any(
            (new_faces[:, 0] == new_faces[:, 1])
            | (new_faces[:, 1] == new_faces[:, 2])
            | (new_faces[:, 2] == new_faces[:, 0])
        ):
            continue
        coordinate_pool = np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
                new_vertices,
            )
        )
        new_triangles = coordinate_pool[new_faces]
        new_crosses = np.cross(
            new_triangles[:, 1] - new_triangles[:, 0],
            new_triangles[:, 2] - new_triangles[:, 0],
        )
        original_triangles = vertices[component_faces]
        original_crosses = np.cross(
            original_triangles[:, 1] - original_triangles[:, 0],
            original_triangles[:, 2] - original_triangles[:, 0],
        )
        original_centroids = original_triangles.mean(axis=1)
        original_offsets = original_centroids - fits[0]["center"]
        original_radial = original_offsets - (original_offsets @ axis)[:, None] * axis
        orientation_sign = np.sign(
            np.sum(original_crosses * original_radial, axis=1).mean()
        )
        new_centroids = new_triangles.mean(axis=1)
        new_offsets = new_centroids - fits[0]["center"]
        new_radial = new_offsets - (new_offsets @ axis)[:, None] * axis
        reverse = (
            np.sum(new_crosses * new_radial, axis=1) * orientation_sign < 0.0
        )
        new_faces[reverse] = new_faces[reverse][:, [0, 2, 1]]

        new_edge_array = np.sort(
            new_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
            axis=1,
        )
        new_unique_edges, new_edge_counts = np.unique(
            new_edge_array, axis=0, return_counts=True
        )
        new_edge_count_map = {
            tuple(map(int, edge)): int(count)
            for edge, count in zip(new_unique_edges, new_edge_counts)
        }
        if any(new_edge_count_map.get(edge, 0) != 1 for edge in boundary_set):
            continue
        if any(
            count != (1 if edge in boundary_set else 2)
            for edge, count in new_edge_count_map.items()
        ):
            continue

        def boundary_directions(face_array):
            directions = {}
            for face in face_array:
                for first, second in (
                    (int(face[0]), int(face[1])),
                    (int(face[1]), int(face[2])),
                    (int(face[2]), int(face[0])),
                ):
                    edge = tuple(sorted((first, second)))
                    if edge in boundary_set:
                        directions[edge] = (first, second)
            return directions

        old_boundary_directions = boundary_directions(component_faces)
        new_boundary_directions = boundary_directions(new_faces)
        if set(old_boundary_directions) != boundary_set or set(
            new_boundary_directions
        ) != boundary_set:
            continue
        direction_matches = np.asarray(
            [
                old_boundary_directions[edge] == new_boundary_directions[edge]
                for edge in boundary_set
            ],
            dtype=bool,
        )
        if not np.all(direction_matches):
            if np.all(~direction_matches):
                new_faces = new_faces[:, [0, 2, 1]]
            else:
                continue
        old_quality = _triangle_quality_values(vertices, component_faces)
        new_quality = _triangle_quality_values(coordinate_pool, new_faces)
        if (
            np.percentile(new_quality, 5.0)
            < np.percentile(old_quality, 5.0) - 1e-8
            or float(new_quality.mean()) < float(old_quality.mean()) - 1e-8
        ):
            continue

        kept_faces[face_ids] = False
        appended_vertices.extend(new_vertices)
        appended_faces.extend(new_faces)
        old_qualities.extend(old_quality)
        new_qualities.extend(new_quality)
        accepted += 1
        removed_faces += len(face_ids)

    if not accepted:
        return vertices.copy(), faces.copy(), {
            "candidates": candidates, "walls": 0, "new_vertices": 0,
            "removed_faces": 0, "new_faces": 0,
            "old_quality": 0.0, "new_quality": 0.0,
        }
    return (
        np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
            )
        ),
        np.vstack((faces[kept_faces], np.asarray(appended_faces, dtype=np.int64))),
        {
            "candidates": candidates,
            "walls": accepted,
            "new_vertices": len(appended_vertices),
            "removed_faces": removed_faces,
            "new_faces": len(appended_faces),
            "old_quality": float(np.mean(old_qualities)),
            "new_quality": float(np.mean(new_qualities)),
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
    first, second, third = map(int, face)
    return (
        (first, second) if first <= second else (second, first),
        (second, third) if second <= third else (third, second),
        (third, first) if third <= first else (first, third),
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


def _stitch_faces_to_isolated_boundary(
    vertices, outside_faces, isolated_faces, original_boundary_edges
):
    """Insert isolated-remesh boundary vertices into adjacent outside faces."""
    vertices = np.asarray(vertices, dtype=np.float64)
    outside_faces = np.asarray(outside_faces, dtype=np.int64)
    isolated_faces = np.asarray(isolated_faces, dtype=np.int64)
    original_boundary_edges = np.asarray(
        original_boundary_edges, dtype=np.int64
    ).reshape(-1, 2)

    isolated_edges = np.sort(
        isolated_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1
    )
    unique_edges, edge_counts = np.unique(
        isolated_edges, axis=0, return_counts=True
    )
    boundary_vertex_ids = np.unique(unique_edges[edge_counts == 1])
    boundary_points = vertices[boundary_vertex_ids]
    subdivisions = {}
    for raw_edge in original_boundary_edges:
        edge = tuple(sorted((int(raw_edge[0]), int(raw_edge[1]))))
        start = vertices[edge[0]]
        vector = vertices[edge[1]] - start
        squared_length = float(np.dot(vector, vector))
        if squared_length <= 0.0:
            continue
        parameters = ((boundary_points - start) @ vector) / squared_length
        projections = start + parameters[:, None] * vector
        distances = np.linalg.norm(boundary_points - projections, axis=1)
        tolerance = max(np.sqrt(squared_length) * 1e-8, 1e-10)
        selected = (
            (parameters >= -1e-8)
            & (parameters <= 1.0 + 1e-8)
            & (distances <= tolerance)
        )
        ids = boundary_vertex_ids[selected]
        values = parameters[selected]
        if len(ids) <= 2:
            continue
        order = np.argsort(values)
        ordered = []
        ordered_parameters = []
        for vertex_id, parameter in zip(ids[order], values[order]):
            vertex_id = int(vertex_id)
            parameter = float(parameter)
            if ordered_parameters and abs(
                parameter - ordered_parameters[-1]
            ) <= 1e-8:
                continue
            ordered.append(vertex_id)
            ordered_parameters.append(parameter)
        interior = [
            vertex_id for vertex_id in ordered if vertex_id not in edge
        ]
        if interior:
            subdivisions[edge] = interior

    rebuilt = []
    interface_buffer_faces = []
    rebuilt_count = 0
    for face in outside_faces:
        polygon = []
        for index in range(3):
            first = int(face[index])
            second = int(face[(index + 1) % 3])
            polygon.append(first)
            key = tuple(sorted((first, second)))
            interior = subdivisions.get(key)
            if interior:
                polygon.extend(
                    interior if first == key[0] else reversed(interior)
                )
        if len(polygon) == 3:
            rebuilt.append(list(map(int, face)))
            continue

        original_normal = np.cross(
            vertices[int(face[1])] - vertices[int(face[0])],
            vertices[int(face[2])] - vertices[int(face[0])],
        )
        if constrained_triangle is None:
            raise RuntimeError(
                "The triangle package is required to stitch an isolated "
                "remesh transaction."
            )
        drop_axis = int(np.argmax(np.abs(original_normal)))
        polygon_points = vertices[np.asarray(polygon, dtype=np.int64)]
        polygon_2d = np.delete(polygon_points, drop_axis, axis=1)
        segments = np.column_stack(
            (
                np.arange(len(polygon), dtype=np.int64),
                np.roll(np.arange(len(polygon), dtype=np.int64), -1),
            )
        )
        result = constrained_triangle.triangulate(
            {"vertices": polygon_2d, "segments": segments}, "pQY"
        )
        result_triangles = np.asarray(
            result.get("triangles", ()), dtype=np.int64
        ).reshape(-1, 3)
        result_vertices = np.asarray(
            result.get("vertices", ()), dtype=np.float64
        ).reshape(-1, 2)
        if len(result_triangles) == 0 or len(result_vertices) != len(polygon):
            raise RuntimeError(
                "Constrained stitching changed the isolated interface "
                "vertex set."
            )
        polygon_ids = np.asarray(polygon, dtype=np.int64)
        face_triangles = polygon_ids[result_triangles].tolist()
        for triangle_index, triangle in enumerate(face_triangles):
            triangle_normal = np.cross(
                vertices[triangle[1]] - vertices[triangle[0]],
                vertices[triangle[2]] - vertices[triangle[0]],
            )
            if float(np.dot(triangle_normal, original_normal)) < 0.0:
                face_triangles[triangle_index] = [
                    triangle[0], triangle[2], triangle[1]
                ]
        rebuilt.extend(face_triangles)
        interface_buffer_faces.extend(face_triangles)
        rebuilt_count += 1
    if interface_buffer_faces:
        interface_buffer_faces = np.asarray(
            interface_buffer_faces, dtype=np.int64
        )
        interface_buffer_edges = np.unique(
            np.sort(
                interface_buffer_faces[
                    :, ((0, 1), (1, 2), (2, 0))
                ].reshape(-1, 2),
                axis=1,
            ),
            axis=0,
        )
    else:
        interface_buffer_edges = np.empty((0, 2), dtype=np.int64)
    return (
        np.asarray(rebuilt, dtype=np.int64),
        rebuilt_count,
        subdivisions,
        interface_buffer_edges,
    )


def _planar_transition_edge_targets(
    vertices,
    faces,
    protected_edges,
    minimum_faces,
    maximum_region_area,
    maximum_edge_length,
    maximum_quality_p5=0.25,
    growth_ratio=1.25,
    generated_vertex_range=None,
    seed_vertex_ids=None,
    minimum_boundary_ratio=4.0,
    minimum_local_target=None,
):
    """Find coarse boundaries around small, poor planar transition strips."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    targets = {}
    region_count = 0
    planar_facets, planar_boundaries = _planar_facets_with_boundaries(mesh)
    for facet, boundary_edges in zip(planar_facets, planar_boundaries):
        facet = np.asarray(facet, dtype=np.int64)
        if len(facet) < int(minimum_faces):
            continue
        facet_area = float(mesh.area_faces[facet].sum())
        if facet_area >= float(maximum_region_area):
            continue
        cycles = _ordered_cycles_from_edges(boundary_edges)
        if cycles is None or len(cycles) != 1:
            continue
        quality = _triangle_quality_values(vertices, faces[facet])
        if float(np.percentile(quality, 5.0)) >= float(maximum_quality_p5):
            continue
        boundary_edges = np.asarray(boundary_edges, dtype=np.int64).reshape(-1, 2)
        if generated_vertex_range is not None or seed_vertex_ids is not None:
            boundary_vertices = np.unique(boundary_edges)
            boundary_matches = False
            if generated_vertex_range is not None:
                generated_start, generated_end = map(
                    int, generated_vertex_range
                )
                boundary_matches |= bool(
                    np.any(
                        (boundary_vertices >= generated_start)
                        & (boundary_vertices < generated_end)
                    )
                )
            if seed_vertex_ids is not None:
                boundary_matches |= bool(
                    np.any(
                        np.isin(
                            boundary_vertices,
                            np.asarray(seed_vertex_ids, dtype=np.int64),
                        )
                    )
                )
            if not boundary_matches:
                continue
        boundary_set = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in boundary_edges
        }
        facet_edges = {
            edge
            for face in faces[facet]
            for edge in _constraint_face_edges(face)
        }
        if (facet_edges & protected) - boundary_set:
            continue
        lengths = np.linalg.norm(
            vertices[boundary_edges[:, 0]] - vertices[boundary_edges[:, 1]],
            axis=1,
        )
        positive = lengths[lengths > 0.0]
        if len(positive) == 0:
            continue
        fine_boundary_length = float(np.percentile(positive, 25.0))
        if float(positive.max()) < (
            fine_boundary_length * float(minimum_boundary_ratio)
        ):
            continue
        local_target = min(
            float(maximum_edge_length),
            fine_boundary_length * float(growth_ratio),
        )
        if minimum_local_target is not None:
            local_target = min(
                float(maximum_edge_length),
                max(local_target, float(minimum_local_target)),
            )
        if not np.isfinite(local_target) or local_target <= 0.0:
            continue
        selected = 0
        for edge, length in zip(boundary_edges, lengths):
            if float(length) <= local_target * (1.0 + 1e-8):
                continue
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            targets[key] = min(targets.get(key, local_target), local_target)
            selected += 1
        if selected:
            region_count += 1
    return targets, region_count


def _split_selected_edges_by_length(
    vertices,
    faces,
    edge_targets,
    edge_lineages=None,
    propagated_edges=None,
    maximum_splits=100000,
):
    """Conformingly bisect selected edges and propagate tracked edge state."""
    vertices_list = [
        np.asarray(vertex, dtype=np.float64).copy() for vertex in vertices
    ]
    faces_list = [list(map(int, face)) for face in np.asarray(faces)]
    active_targets = {
        tuple(sorted((int(edge[0]), int(edge[1])))): float(target)
        for edge, target in edge_targets.items()
    }
    lineages = dict(edge_lineages or {})
    propagated = set(propagated_edges or ())
    edge_faces = _build_edge_faces(np.asarray(faces_list, dtype=np.int64))
    heap = []
    for edge, target in active_targets.items():
        length = float(
            np.linalg.norm(vertices_list[edge[0]] - vertices_list[edge[1]])
        )
        heapq.heappush(heap, (-length, edge, target))

    split_count = 0
    while heap and split_count < int(maximum_splits):
        _, edge, queued_target = heapq.heappop(heap)
        target = active_targets.get(edge)
        if target is None or not np.isclose(target, queued_target):
            continue
        length = float(
            np.linalg.norm(vertices_list[edge[0]] - vertices_list[edge[1]])
        )
        if length <= target * (1.0 + 1e-8):
            del active_targets[edge]
            continue
        incident_faces = sorted(edge_faces.get(edge, ()))
        if not incident_faces:
            del active_targets[edge]
            continue

        midpoint_index = len(vertices_list)
        vertices_list.append(
            (vertices_list[edge[0]] + vertices_list[edge[1]]) * 0.5
        )
        replacements = []
        for face_index in incident_faces:
            first, second = _constraint_split_face(
                faces_list[face_index], edge, midpoint_index
            )
            replacements.append((face_index, first, second))
        for face_index, first, second in replacements:
            for old_edge in _constraint_face_edges(faces_list[face_index]):
                memberships = edge_faces.get(old_edge)
                if memberships is not None:
                    memberships.discard(face_index)
                    if not memberships:
                        del edge_faces[old_edge]
            faces_list[face_index] = first
            second_index = len(faces_list)
            faces_list.append(second)
            for new_edge in _constraint_face_edges(first):
                edge_faces.setdefault(new_edge, set()).add(face_index)
            for new_edge in _constraint_face_edges(second):
                edge_faces.setdefault(new_edge, set()).add(second_index)

        del active_targets[edge]
        children = (
            tuple(sorted((edge[0], midpoint_index))),
            tuple(sorted((midpoint_index, edge[1]))),
        )
        root_index = lineages.pop(edge, None)
        if root_index is not None:
            lineages[children[0]] = root_index
            lineages[children[1]] = root_index
        if edge in propagated:
            propagated.discard(edge)
            propagated.update(children)
        for child in children:
            active_targets[child] = target
            child_length = float(
                np.linalg.norm(
                    vertices_list[child[0]] - vertices_list[child[1]]
                )
            )
            heapq.heappush(heap, (-child_length, child, target))
        split_count += 1

    remaining_max = 0.0
    for edge in active_targets:
        remaining_max = max(
            remaining_max,
            float(
                np.linalg.norm(
                    vertices_list[edge[0]] - vertices_list[edge[1]]
                )
            ),
        )
    return (
        np.asarray(vertices_list, dtype=np.float64),
        np.asarray(faces_list, dtype=np.int64),
        lineages,
        propagated,
        {
            "splits": split_count,
            "remaining_max_length": remaining_max,
            "hit_split_limit": bool(active_targets),
        },
    )


def _build_edge_faces(faces):
    edge_faces = {}
    for face_index, face in enumerate(faces):
        for edge in _constraint_face_edges(face):
            edge_faces.setdefault(edge, set()).add(face_index)
    return edge_faces


def _edge_length_array(vertices, edges):
    """Compute edge lengths in one NumPy batch without scalar dispatch."""
    edge_array = np.asarray(list(edges), dtype=np.int64).reshape(-1, 2)
    if len(edge_array) == 0:
        return np.empty(0, dtype=np.float64)
    points = np.asarray(vertices, dtype=np.float64)
    vectors = points[edge_array[:, 0]] - points[edge_array[:, 1]]
    return np.sqrt(np.einsum("ij,ij->i", vectors, vectors))


def collapse_short_coplanar_edges(
    vertices,
    faces,
    protected_edges,
    maximum_short_edge_length,
    maximum_edge_length,
    maximum_planar_angle_degrees=0.1,
    passes=1,
):
    """Collapse only topology-safe short edges in strictly planar interiors."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).copy()
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    }
    protected_vertices = {vertex for edge in protected for vertex in edge}
    maximum_short_edge_length = float(maximum_short_edge_length)
    maximum_edge_length = float(maximum_edge_length)
    maximum_short_edge_length_squared = maximum_short_edge_length ** 2
    maximum_edge_length_squared = (
        maximum_edge_length * (1.0 + 1e-8)
    ) ** 2
    cosine_limit = np.cos(np.deg2rad(float(maximum_planar_angle_degrees)))
    collapse_count = 0
    active_vertices = None

    for _ in range(max(int(passes), 0)):
        face_count = len(faces)
        face_edges = np.concatenate(
            (
                faces[:, (0, 1)],
                faces[:, (1, 2)],
                faces[:, (2, 0)],
            ),
            axis=0,
        )
        face_edges.sort(axis=1)
        unique_edges, edge_inverse, edge_counts = np.unique(
            face_edges,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        edge_occurrence_order = np.argsort(edge_inverse, kind="stable")
        edge_occurrence_offsets = np.concatenate(
            ((0,), np.cumsum(edge_counts, dtype=np.int64))
        )
        occurrence_faces = np.tile(
            np.arange(face_count, dtype=np.int64), 3
        )
        vertex_faces = [set() for _ in range(len(vertices))]
        vertex_neighbors = [set() for _ in range(len(vertices))]
        for face_index, face in enumerate(faces):
            first, second, third = map(int, face)
            vertex_faces[first].add(face_index)
            vertex_faces[second].add(face_index)
            vertex_faces[third].add(face_index)
            vertex_neighbors[first].update((second, third))
            vertex_neighbors[second].update((first, third))
            vertex_neighbors[third].update((first, second))
        boundary_edges = unique_edges[edge_counts == 1]
        if len(boundary_edges):
            protected_vertices.update(map(int, boundary_edges.reshape(-1)))

        triangles = vertices[faces]
        crosses = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        cross_lengths = np.linalg.norm(crosses, axis=1)
        valid_faces = cross_lengths > np.finfo(np.float64).eps
        normals = np.zeros_like(crosses)
        normals[valid_faces] = crosses[valid_faces] / cross_lengths[valid_faces, None]
        face_quality = _triangle_quality_values(vertices, faces)
        candidate_edge_indices = np.flatnonzero(edge_counts == 2)
        candidate_edges = unique_edges[candidate_edge_indices]
        if len(candidate_edges):
            protected_vertex_mask = np.zeros(len(vertices), dtype=bool)
            if protected_vertices:
                protected_vertex_mask[
                    np.fromiter(protected_vertices, dtype=np.int64)
                ] = True
            unlocked = ~np.any(protected_vertex_mask[candidate_edges], axis=1)
            candidate_edge_indices = candidate_edge_indices[unlocked]
            candidate_edges = candidate_edges[unlocked]
        if len(candidate_edges) and active_vertices is not None:
            # A simultaneous pass only changes faces in the accepted edges'
            # one-rings.  On later passes, every newly viable collapse must
            # therefore touch that dirty vertex set; all other local tests are
            # unchanged from the preceding pass.
            active_vertex_mask = np.zeros(len(vertices), dtype=bool)
            if active_vertices:
                active_vertex_mask[
                    np.fromiter(active_vertices, dtype=np.int64)
                ] = True
            locally_changed = np.any(
                active_vertex_mask[candidate_edges], axis=1
            )
            candidate_edge_indices = candidate_edge_indices[locally_changed]
            candidate_edges = candidate_edges[locally_changed]
        candidates = []
        if len(candidate_edges):
            candidate_vectors = (
                vertices[candidate_edges[:, 0]]
                - vertices[candidate_edges[:, 1]]
            )
            candidate_length_squared = np.einsum(
                "ij,ij->i", candidate_vectors, candidate_vectors
            )
            candidates = [
                (
                    float(length_squared),
                    (int(edge[0]), int(edge[1])),
                    int(edge_index),
                )
                for length_squared, edge, edge_index in zip(
                    candidate_length_squared,
                    candidate_edges,
                    candidate_edge_indices,
                )
                if 0.0 < length_squared < maximum_short_edge_length_squared
            ]
        candidates.sort()

        # Build all topology-valid local collapse alternatives first.  Calling
        # NumPy separately for both directions of every short edge dominates
        # runtime on dense meshes; the geometry below is identical but is
        # evaluated in one merged batch.
        collapse_records = []
        option_faces = []
        option_normals = []
        option_offsets = [0]
        option_removed_kept = []
        for _, edge, edge_index in candidates:
            first, second = edge
            local_vertices = (
                {first, second}
                | vertex_neighbors[first]
                | vertex_neighbors[second]
            )
            occurrence_start = edge_occurrence_offsets[edge_index]
            occurrence_end = edge_occurrence_offsets[edge_index + 1]
            incident_edge_faces = occurrence_faces[
                edge_occurrence_order[occurrence_start:occurrence_end]
            ]
            opposite_vertices = {
                int(vertex)
                for face_index in incident_edge_faces
                for vertex in faces[face_index]
                if int(vertex) not in edge
            }
            if (
                vertex_neighbors[first] & vertex_neighbors[second]
            ) != opposite_vertices:
                continue
            local_face_ids = sorted(vertex_faces[first] | vertex_faces[second])
            if not local_face_ids or not np.all(valid_faces[local_face_ids]):
                continue
            reference_normal = normals[local_face_ids[0]]
            if np.any(normals[local_face_ids] @ reference_normal < cosine_limit):
                continue

            old_quality = face_quality[local_face_ids]
            record_options = []
            for removed, kept in ((first, second), (second, first)):
                local_faces = faces[local_face_ids].copy()
                local_faces[local_faces == removed] = kept
                nondegenerate = (
                    (local_faces[:, 0] != local_faces[:, 1])
                    & (local_faces[:, 1] != local_faces[:, 2])
                    & (local_faces[:, 2] != local_faces[:, 0])
                )
                local_faces = local_faces[nondegenerate]
                if len(local_faces) == 0:
                    continue
                option_index = len(option_faces)
                option_faces.append(local_faces)
                option_normals.append(reference_normal)
                option_offsets.append(option_offsets[-1] + len(local_faces))
                option_removed_kept.append((removed, kept))
                record_options.append(option_index)
            if record_options:
                collapse_records.append(
                    (
                        local_vertices,
                        float(old_quality.min()),
                        float(old_quality.mean()),
                        record_options,
                    )
                )

        option_scores = [None] * len(option_faces)
        if option_faces:
            merged_faces = np.concatenate(option_faces, axis=0)
            merged_triangles = vertices[merged_faces]
            first_vectors = merged_triangles[:, 1] - merged_triangles[:, 0]
            second_vectors = merged_triangles[:, 2] - merged_triangles[:, 0]
            option_ids = np.repeat(
                np.arange(len(option_faces), dtype=np.int64),
                np.diff(option_offsets),
            )
            merged_normals = np.asarray(option_normals)[option_ids]
            orientation = (
                (first_vectors[:, 1] * second_vectors[:, 2]
                 - first_vectors[:, 2] * second_vectors[:, 1])
                * merged_normals[:, 0]
                + (first_vectors[:, 2] * second_vectors[:, 0]
                   - first_vectors[:, 0] * second_vectors[:, 2])
                * merged_normals[:, 1]
                + (first_vectors[:, 0] * second_vectors[:, 1]
                   - first_vectors[:, 1] * second_vectors[:, 0])
                * merged_normals[:, 2]
            )
            merged_edge_vectors = (
                merged_triangles[:, (1, 2, 0)]
                - merged_triangles[:, (0, 1, 2)]
            )
            merged_length_squared = np.einsum(
                "ijk,ijk->ij", merged_edge_vectors, merged_edge_vectors
            ).max(axis=1)
            merged_quality = _triangle_quality_values(vertices, merged_faces)
            starts = np.asarray(option_offsets[:-1], dtype=np.int64)
            option_orientation_min = np.minimum.reduceat(orientation, starts)
            option_length_max = np.maximum.reduceat(
                merged_length_squared, starts
            )
            option_quality_min = np.minimum.reduceat(merged_quality, starts)
            option_quality_mean = np.add.reduceat(merged_quality, starts) / np.diff(
                option_offsets
            )
            for option_index in range(len(option_faces)):
                if (
                    option_orientation_min[option_index] > 1e-14
                    and option_length_max[option_index]
                    <= maximum_edge_length_squared
                ):
                    option_scores[option_index] = (
                        float(option_quality_min[option_index]),
                        float(option_quality_mean[option_index]),
                    )

        reserved_vertices = set()
        replacements = {}
        accepted_this_pass = 0
        for local_vertices, old_minimum, old_mean, record_options in collapse_records:
            if local_vertices & reserved_vertices:
                continue
            best = None
            for option_index in record_options:
                score = option_scores[option_index]
                if score is None:
                    continue
                if (
                    score[0] < old_minimum - 1e-8
                    or score[1] < old_mean - 1e-8
                ):
                    continue
                if best is None or score > best[0]:
                    removed, kept = option_removed_kept[option_index]
                    best = (score, removed, kept)
            if best is not None:
                _, removed, kept = best
                replacements[removed] = kept
                reserved_vertices.update(local_vertices)
                accepted_this_pass += 1

        if not replacements:
            break
        for removed, kept in replacements.items():
            faces[faces == removed] = kept
        nondegenerate = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 2] != faces[:, 0])
        )
        faces = faces[nondegenerate]
        collapse_count += accepted_this_pass
        # Candidates skipped by the independent-set reservation can have an
        # unchanged endpoint whose one-ring merely overlaps a changed region.
        # Include that outer ring so later passes preserve the exact selection
        # semantics of a full rescan.
        active_vertices = set(reserved_vertices)
        for vertex in reserved_vertices:
            active_vertices.update(vertex_neighbors[vertex])

    return vertices.copy(), faces, collapse_count


def collapse_short_coplanar_edges_with_backend(
    vertices,
    faces,
    protected_edges,
    maximum_short_edge_length,
    maximum_edge_length,
    maximum_planar_angle_degrees=0.1,
    passes=1,
    backend="cpu",
):
    """Dispatch strict short-edge cleanup to the selected topology backend."""
    backend = str(backend).lower()
    if backend == "cpu":
        return collapse_short_coplanar_edges(
            vertices,
            faces,
            protected_edges,
            maximum_short_edge_length,
            maximum_edge_length,
            maximum_planar_angle_degrees=maximum_planar_angle_degrees,
            passes=passes,
        )
    if backend != "cuda":
        raise ValueError("Constraint topology backend must be cpu or cuda.")
    from .constrained_gpu import collapse_short_coplanar_edges_cuda

    result_vertices, result_faces, count, timings = (
        collapse_short_coplanar_edges_cuda(
            vertices,
            faces,
            protected_edges,
            maximum_short_edge_length,
            maximum_edge_length,
            maximum_planar_angle_degrees=maximum_planar_angle_degrees,
            passes=passes,
        )
    )
    print(
        "  CUDA collapse detail: preparation {:.3f}s, transfer {:.3f}s, "
        "kernels {:.3f}s.".format(
            timings["preparation"],
            timings["transfer"],
            timings["kernel"],
        ),
        flush=True,
    )
    return result_vertices, result_faces, count


def automatic_edge_length_limit(vertices, edge_lengths):
    """Choose 5% of the bounding-box diagonal as the maximum edge length."""
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
    limit = diagonal * 0.05
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


def _subdivide_labeled_patch_boundaries(
    vertices,
    faces,
    labels,
    maximum_edge_length,
    patch_edge_limits=None,
    explicit_constraint_edges=None,
    return_lineage=False,
    maximum_splits=None,
):
    """Finalize shared constraints once and optionally retain source edge IDs.

    Topological boundaries, label transitions, and explicitly supplied mesh
    edges share one subdivision queue. Every original vertex keeps its index
    and position. The optional fifth return value maps each surviving child
    edge to the sorted original constraint edge that it subdivides.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    raw_faces = np.asarray(faces)
    raw_labels = np.asarray(labels)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Boundary subdivision vertices must have shape (N, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError("Boundary subdivision vertices must be finite.")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
        raise ValueError("Boundary subdivision faces must have shape (M, 3).")
    if not np.issubdtype(raw_faces.dtype, np.integer):
        raise ValueError("Boundary subdivision face indices must be integers.")
    faces = np.asarray(raw_faces, dtype=np.int64)
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Boundary subdivision face index is out of range.")
    if np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ):
        raise ValueError("Boundary subdivision faces must have distinct indices.")
    if raw_labels.shape != (len(faces),) or not np.issubdtype(
        raw_labels.dtype, np.integer
    ):
        raise ValueError("Boundary subdivision needs one integer label per face.")
    maximum_edge_length = float(maximum_edge_length)
    if not np.isfinite(maximum_edge_length) or maximum_edge_length <= 0.0:
        raise ValueError("Boundary maximum edge length must be finite and positive.")
    if maximum_splits is not None and (
        isinstance(maximum_splits, (bool, np.bool_))
        or not isinstance(maximum_splits, (int, np.integer))
        or maximum_splits < 0
    ):
        raise ValueError("Boundary maximum_splits must be a non-negative integer.")
    patch_edge_limits = {
        label: float(limit) for label, limit in (patch_edge_limits or {}).items()
    }
    if any(
        not np.isfinite(limit) or limit <= 0.0
        for limit in patch_edge_limits.values()
    ):
        raise ValueError("Per-patch boundary edge limits must be finite and positive.")
    vertices_list = [vertex.copy() for vertex in vertices]
    faces_list = [list(map(int, face)) for face in faces]
    labels_list = list(map(int, raw_labels))
    edge_faces = _build_edge_faces(np.asarray(faces_list, dtype=np.int64))
    boundary_edges = set()
    for edge, incidence in edge_faces.items():
        incidence = tuple(incidence)
        if (
            len(incidence) != 2
            or labels_list[incidence[0]] != labels_list[incidence[1]]
        ):
            boundary_edges.add(edge)
    if explicit_constraint_edges is not None:
        explicit_edges = np.asarray(explicit_constraint_edges)
        if explicit_edges.size == 0 and explicit_edges.shape in ((0,), (0, 2)):
            explicit_edges = np.empty((0, 2), dtype=np.int64)
        if (
            explicit_edges.ndim != 2
            or explicit_edges.shape[1] != 2
            or not np.issubdtype(explicit_edges.dtype, np.integer)
        ):
            raise ValueError(
                "Explicit constraint edges must be an integer (K, 2) array."
            )
        for endpoints in explicit_edges:
            edge = tuple(sorted(map(int, endpoints)))
            if edge not in edge_faces:
                raise ValueError(
                    "Explicit constraint is not a mesh edge: {}.".format(edge)
                )
            boundary_edges.add(edge)
    source_edges = sorted(boundary_edges)
    edge_sources = {edge: index for index, edge in enumerate(source_edges)}

    def edge_limit(edge):
        incidence = edge_faces.get(edge, ())
        owning_labels = {labels_list[index] for index in incidence}
        limits = [float(maximum_edge_length)]
        limits.extend(
            float(patch_edge_limits[label])
            for label in owning_labels
            if label in patch_edge_limits
        )
        return min(limits)

    heap = []
    for edge in boundary_edges:
        length = float(
            np.linalg.norm(vertices_list[edge[0]] - vertices_list[edge[1]])
        )
        if length > edge_limit(edge) * (1.0 + 1e-8):
            heapq.heappush(heap, (-length, edge))
    split_count = 0
    while heap:
        _, edge = heapq.heappop(heap)
        if edge not in boundary_edges or edge not in edge_faces:
            continue
        length = float(
            np.linalg.norm(vertices_list[edge[0]] - vertices_list[edge[1]])
        )
        if length <= edge_limit(edge) * (1.0 + 1e-8):
            continue
        if maximum_splits is not None and split_count >= maximum_splits:
            raise RuntimeError(
                "Boundary subdivision exceeds maximum_splits={}.".format(
                    maximum_splits
                )
            )
        incident_faces = sorted(edge_faces[edge])
        midpoint_index = len(vertices_list)
        midpoint = vertices_list[edge[0]] * 0.5 + vertices_list[edge[1]] * 0.5
        if np.array_equal(midpoint, vertices_list[edge[0]]) or np.array_equal(
            midpoint, vertices_list[edge[1]]
        ):
            raise RuntimeError("Boundary edge limit is below coordinate precision.")
        vertices_list.append(midpoint)
        replacements = []
        for face_index in incident_faces:
            first_face, second_face = _constraint_split_face(
                faces_list[face_index], edge, midpoint_index
            )
            replacements.append((face_index, first_face, second_face))
        for face_index, first_face, second_face in replacements:
            for old_edge in _constraint_face_edges(faces_list[face_index]):
                membership = edge_faces.get(old_edge)
                if membership is not None:
                    membership.discard(face_index)
                    if not membership:
                        del edge_faces[old_edge]
            faces_list[face_index] = first_face
            second_index = len(faces_list)
            faces_list.append(second_face)
            labels_list.append(labels_list[face_index])
            for new_edge in _constraint_face_edges(first_face):
                edge_faces.setdefault(new_edge, set()).add(face_index)
            for new_edge in _constraint_face_edges(second_face):
                edge_faces.setdefault(new_edge, set()).add(second_index)
        boundary_edges.discard(edge)
        children = (
            tuple(sorted((edge[0], midpoint_index))),
            tuple(sorted((midpoint_index, edge[1]))),
        )
        boundary_edges.update(children)
        source_index = edge_sources.pop(edge)
        for child in children:
            edge_sources[child] = source_index
            child_length = float(
                np.linalg.norm(
                    vertices_list[child[0]] - vertices_list[child[1]]
                )
            )
            if child_length > edge_limit(child) * (1.0 + 1e-8):
                heapq.heappush(heap, (-child_length, child))
        split_count += 1
    result = (
        np.asarray(vertices_list, dtype=np.float64).reshape(-1, 3),
        np.asarray(faces_list, dtype=np.int64).reshape(-1, 3),
        np.asarray(labels_list, dtype=np.int64),
        split_count,
    )
    if return_lineage:
        final_edges = sorted(boundary_edges)
        lineage = {
            "constraint_edges": np.asarray(
                final_edges, dtype=np.int64
            ).reshape(-1, 2),
            "source_constraint_edges": np.asarray(
                source_edges, dtype=np.int64
            ).reshape(-1, 2),
            "source_edge_indices": np.asarray(
                [edge_sources[edge] for edge in final_edges], dtype=np.int64
            ),
            "splits": int(split_count),
        }
        return result + (lineage,)
    return result


def _retriangulate_brep_cylinder_columns(
    vertices,
    faces,
    boundary_edges,
    maximum_edge_length,
    maximum_local_growth=2.0,
):
    """Preserve end-ring angular columns throughout a cylindrical fillet."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    fit = _fit_component_cylinder(
        vertices, faces, radius_tolerance=1e-2, normal_tolerance=3e-2
    )
    if fit is None:
        return None
    boundary_ids = np.unique(np.asarray(boundary_edges, dtype=np.int64))
    offsets = vertices[boundary_ids] - fit["origin"]
    axial = offsets @ fit["axis"]
    axial_min = float(axial.min())
    axial_max = float(axial.max())
    height = axial_max - axial_min
    axial_tolerance = max(height * 1e-7, 1e-9)
    lower_ids = boundary_ids[np.abs(axial - axial_min) <= axial_tolerance]
    upper_ids = boundary_ids[np.abs(axial - axial_max) <= axial_tolerance]
    if len(lower_ids) < 3 or len(lower_ids) != len(upper_ids):
        return None

    def raw_angles(vertex_ids):
        ring_offsets = vertices[vertex_ids] - fit["origin"]
        return np.arctan2(
            ring_offsets @ fit["second_basis"],
            ring_offsets @ fit["first_basis"],
        )

    lower_raw = raw_angles(lower_ids)
    sorted_raw = np.sort(np.mod(lower_raw, 2.0 * np.pi))
    gaps = np.diff(
        np.concatenate((sorted_raw, sorted_raw[:1] + 2.0 * np.pi))
    )
    largest_gap = int(np.argmax(gaps))
    cut_angle = float(
        sorted_raw[largest_gap] + 0.5 * gaps[largest_gap]
    )

    def order_ring(vertex_ids):
        unwrapped = np.mod(raw_angles(vertex_ids) - cut_angle, 2.0 * np.pi)
        order = np.argsort(unwrapped)
        return np.asarray(vertex_ids, dtype=np.int64)[order], unwrapped[order]

    lower_ids, lower_angles = order_ring(lower_ids)
    upper_ids, upper_angles = order_ring(upper_ids)
    angle_difference = np.arctan2(
        np.sin(upper_angles - lower_angles),
        np.cos(upper_angles - lower_angles),
    )
    if float(np.max(np.abs(angle_difference))) > 1e-3:
        return None
    column_angles = lower_angles + 0.5 * angle_difference + cut_angle
    lower_chords = np.linalg.norm(
        vertices[lower_ids[1:]] - vertices[lower_ids[:-1]], axis=1
    )
    upper_chords = np.linalg.norm(
        vertices[upper_ids[1:]] - vertices[upper_ids[:-1]], axis=1
    )
    local_chords = np.concatenate((lower_chords, upper_chords))
    local_chords = local_chords[local_chords > 0.0]
    if len(local_chords) == 0:
        return None
    local_size = float(np.median(local_chords))
    maximum_local_edge = min(
        float(maximum_edge_length),
        float(maximum_local_growth) * local_size,
    )
    side_ids = np.setdiff1d(
        boundary_ids, np.concatenate((lower_ids, upper_ids))
    )
    side_offsets = vertices[side_ids] - fit["origin"]
    side_axial = side_offsets @ fit["axis"] if len(side_ids) else np.empty(0)
    axial_levels = np.unique(
        np.concatenate(([axial_min], side_axial, [axial_max]))
    )

    new_vertices = []
    grid = np.empty((len(axial_levels), len(column_angles)), dtype=np.int64)
    grid[0] = lower_ids
    grid[-1] = upper_ids
    endpoint_angles = np.asarray((column_angles[0], column_angles[-1]))
    for row in range(1, len(axial_levels) - 1):
        level = float(axial_levels[row])
        for column, angle in enumerate(column_angles):
            existing = None
            if column in (0, len(column_angles) - 1) and len(side_ids):
                endpoint_index = 0 if column == 0 else 1
                side_angles = np.mod(raw_angles(side_ids) - cut_angle, 2.0 * np.pi)
                target_angle = np.mod(
                    endpoint_angles[endpoint_index] - cut_angle,
                    2.0 * np.pi,
                )
                angular_error = np.abs(
                    np.arctan2(
                        np.sin(side_angles - target_angle),
                        np.cos(side_angles - target_angle),
                    )
                )
                candidates = np.flatnonzero(
                    (np.abs(side_axial - level) <= axial_tolerance)
                    & (angular_error <= 1e-5)
                )
                if len(candidates):
                    existing = int(side_ids[candidates[0]])
            if existing is not None:
                grid[row, column] = existing
                continue
            if column in (0, len(column_angles) - 1):
                # A new point here would silently change a shared patch edge.
                # The two side chains must therefore define identical rows.
                return None
            radial = (
                np.cos(angle) * fit["first_basis"]
                + np.sin(angle) * fit["second_basis"]
            )
            point = fit["origin"] + level * fit["axis"] + fit["radius"] * radial
            grid[row, column] = len(vertices) + len(new_vertices)
            new_vertices.append(point)

    new_faces = []
    for row in range(len(axial_levels) - 1):
        for column in range(len(column_angles) - 1):
            lower_left = int(grid[row, column])
            lower_right = int(grid[row, column + 1])
            upper_left = int(grid[row + 1, column])
            upper_right = int(grid[row + 1, column + 1])
            if (row + column) % 2 == 0:
                new_faces.extend(
                    (
                        (lower_left, lower_right, upper_left),
                        (lower_right, upper_right, upper_left),
                    )
                )
            else:
                new_faces.extend(
                    (
                        (lower_left, lower_right, upper_right),
                        (lower_left, upper_right, upper_left),
                    )
                )
    output_vertices = np.vstack(
        (
            vertices,
            np.asarray(new_vertices, dtype=np.float64).reshape(-1, 3),
        )
    )
    output_faces = np.asarray(new_faces, dtype=np.int64)
    original_triangles = vertices[faces]
    original_crosses = np.cross(
        original_triangles[:, 1] - original_triangles[:, 0],
        original_triangles[:, 2] - original_triangles[:, 0],
    )
    original_centroids = original_triangles.mean(axis=1)
    original_offsets = original_centroids - fit["origin"]
    original_axial = original_offsets @ fit["axis"]
    original_radial = original_offsets - original_axial[:, None] * fit["axis"]
    orientation = np.sign(
        np.mean(np.sum(original_crosses * original_radial, axis=1))
    )
    new_triangles = output_vertices[output_faces]
    new_crosses = np.cross(
        new_triangles[:, 1] - new_triangles[:, 0],
        new_triangles[:, 2] - new_triangles[:, 0],
    )
    new_centroids = new_triangles.mean(axis=1)
    new_offsets = new_centroids - fit["origin"]
    new_axial = new_offsets @ fit["axis"]
    new_radial = new_offsets - new_axial[:, None] * fit["axis"]
    reverse = np.sum(new_crosses * new_radial, axis=1) * orientation < 0.0
    output_faces[reverse] = output_faces[reverse][:, [0, 2, 1]]
    return output_vertices, output_faces, {
        "patches": 1,
        "angular_columns": int(len(column_angles)),
        "axial_rows": int(len(axial_levels)),
        "local_size": local_size,
        "maximum_local_edge": maximum_local_edge,
        "new_vertices": int(len(new_vertices)),
    }


def retriangulate_brep_model_patches(
    vertices,
    faces,
    maximum_edge_length,
    cylinder_target_edge_ratio=2.0,
    minimum_triangle_angle_degrees=28.0,
):
    """Atomically rebuild strict B-rep patches and stitch shared boundaries."""
    from .brep_partition import (
        build_brep_model_partitions,
        patch_boundary_edges,
    )

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    partition = build_brep_model_partitions(vertices, faces)
    if partition is None:
        return None
    labels, stats = partition
    original_face_count = len(faces)
    cylinder_edge_limits = {}
    for patch_id, surface_type in enumerate(stats["surface_types"]):
        if surface_type != "cylinder":
            continue
        patch_faces = faces[labels == patch_id]
        patch_edges = np.sort(
            patch_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
            axis=1,
        )
        unique_edges, edge_counts = np.unique(
            patch_edges, axis=0, return_counts=True
        )
        boundary_edges = unique_edges[edge_counts == 1]
        boundary_lengths = np.linalg.norm(
            vertices[boundary_edges[:, 0]] - vertices[boundary_edges[:, 1]],
            axis=1,
        )
        positive_lengths = boundary_lengths[boundary_lengths > 0.0]
        if len(positive_lengths):
            cylinder_edge_limits[patch_id] = min(
                float(maximum_edge_length) * 0.7,
                2.0 * float(np.median(positive_lengths)),
            )
    vertices, faces, labels, boundary_split_count = (
        _subdivide_labeled_patch_boundaries(
            vertices,
            faces,
            labels,
            float(maximum_edge_length) * 0.7,
            patch_edge_limits=cylinder_edge_limits,
        )
    )
    surface_types = stats["surface_types"]
    original_boundaries = patch_boundary_edges(faces, labels)
    appended_vertices = []
    rebuilt_faces = []
    rebuilt_labels = []
    patch_records = []

    for patch_id, surface_type in enumerate(surface_types):
        patch_faces = faces[labels == patch_id]
        vertex_ids, inverse = np.unique(
            patch_faces.reshape(-1), return_inverse=True
        )
        local_vertices = vertices[vertex_ids]
        local_faces = inverse.reshape(-1, 3)
        local_edges = np.sort(
            local_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
            axis=1,
        )
        unique_edges, edge_counts = np.unique(
            local_edges, axis=0, return_counts=True
        )
        boundary_edges = unique_edges[edge_counts == 1]
        old_quality = _triangle_quality_values(local_vertices, local_faces)

        if surface_type == "cylinder":
            old_p5 = float(np.percentile(old_quality, 5.0))
            # Poor source triangles need more degrees of freedom.  The cap is
            # still finite, but rises smoothly from 2x to 3.25x as P5 quality
            # falls below 0.2.  Acceptance remains guarded by quality checks.
            quality_need = np.clip((0.2 - old_p5) / 0.2, 0.0, 1.0)
            face_ratio_cap = 2.0 + 1.25 * float(quality_need)
            structured_result = _retriangulate_brep_cylinder_columns(
                local_vertices,
                local_faces,
                boundary_edges,
                maximum_edge_length=maximum_edge_length,
                maximum_local_growth=2.0,
            )
            if structured_result is not None:
                rebuilt_vertices, local_output_faces, patch_stats = (
                    structured_result
                )
                face_ratio_cap = None
            else:
                rebuilt_vertices, local_output_faces, patch_stats = (
                    retriangulate_partial_cylindrical_walls(
                    local_vertices,
                    local_faces,
                    protected_edges=boundary_edges,
                    minimum_faces=3,
                    radius_tolerance=1e-2,
                    normal_tolerance=3e-2,
                    minimum_angle_degrees=5.0,
                    target_edge_ratio=cylinder_target_edge_ratio,
                    isolate_rounded_faces=False,
                    minimum_triangle_angle_degrees=(
                        minimum_triangle_angle_degrees
                    ),
                    minimum_radial_alignment=0.95,
                    preferred_edge_length=maximum_edge_length,
                    direct_target_spacing=True,
                    allow_density_simplification=True,
                    maximum_face_count_ratio=face_ratio_cap,
                )
                )
            accepted = patch_stats["patches"] == 1
        elif len(local_faces) <= 2:
            # A triangular or quadrilateral planar cap is already the minimal
            # constrained triangulation. Replacing its diagonal can only trade
            # quality between two equivalent faces, so retain it verbatim.
            rebuilt_vertices = local_vertices.copy()
            local_output_faces = local_faces.copy()
            patch_stats = {"regions": 1, "minimal_planar_patch": True}
            face_ratio_cap = None
            accepted = True
        else:
            rebuilt_vertices, local_output_faces, patch_stats = (
                retriangulate_planar_annuli(
                    local_vertices,
                    local_faces,
                    protected_edges=boundary_edges,
                    minimum_faces=1,
                    minimum_holes=0,
                    maximum_holes=0,
                    maximum_target_edge_length=(
                        float(maximum_edge_length) * 0.75
                    ),
                    maximum_result_edge_length=maximum_edge_length,
                    minimum_angle_degrees=minimum_triangle_angle_degrees,
                    accept_strong_mean_gain=True,
                    skip_satisfactory_quality=False,
                    gradual_target_spacing=False,
                )
            )
            face_ratio_cap = None
            accepted = patch_stats["regions"] == 1
        if not accepted:
            return None

        local_to_global = np.empty(len(rebuilt_vertices), dtype=np.int64)
        local_to_global[:len(local_vertices)] = vertex_ids
        new_vertex_count = len(rebuilt_vertices) - len(local_vertices)
        local_to_global[len(local_vertices):] = np.arange(
            len(vertices) + len(appended_vertices),
            len(vertices) + len(appended_vertices) + new_vertex_count,
            dtype=np.int64,
        )
        if new_vertex_count:
            appended_vertices.extend(rebuilt_vertices[len(local_vertices):])
        global_faces = local_to_global[local_output_faces]
        rebuilt_faces.append(global_faces)
        rebuilt_labels.append(
            np.full(len(global_faces), patch_id, dtype=np.int64)
        )
        new_quality = _triangle_quality_values(
            rebuilt_vertices, local_output_faces
        )
        patch_records.append(
            {
                "patch_id": int(patch_id),
                "surface_type": surface_type,
                "old_faces": int(len(local_faces)),
                "new_faces": int(len(local_output_faces)),
                "new_vertices": int(new_vertex_count),
                "old_quality": float(old_quality.mean()),
                "new_quality": float(new_quality.mean()),
                "old_quality_p5": float(np.percentile(old_quality, 5.0)),
                "new_quality_p5": float(np.percentile(new_quality, 5.0)),
                "face_ratio_cap": face_ratio_cap,
                "angular_columns": patch_stats.get("angular_columns"),
                "axial_rows": patch_stats.get("axial_rows"),
                "local_size": patch_stats.get("local_size"),
                "maximum_local_edge": patch_stats.get(
                    "maximum_local_edge"
                ),
            }
        )

    output_vertices = np.vstack(
        (
            vertices,
            np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
        )
    )
    output_faces = np.vstack(rebuilt_faces)
    output_labels = np.concatenate(rebuilt_labels)
    output_boundaries = patch_boundary_edges(output_faces, output_labels)
    if set(map(tuple, output_boundaries)) != set(map(tuple, original_boundaries)):
        return None
    edge_faces = _build_edge_faces(output_faces)
    if any(len(incidence) != 2 for incidence in edge_faces.values()):
        return None

    stats = dict(stats)
    stats.update(
        {
            "patch_records": patch_records,
            "output_patch_face_counts": np.bincount(output_labels).tolist(),
            "new_vertices": int(len(output_vertices) - len(vertices)),
            "old_faces": int(original_face_count),
            "new_faces": int(len(output_faces)),
            "boundary_edges": int(len(original_boundaries)),
            "boundary_splits": int(boundary_split_count),
            "cylinder_boundary_edge_limits": cylinder_edge_limits,
            "face_labels": output_labels,
        }
    )
    return output_vertices, output_faces, original_boundaries, stats


def refine_original_mesh_by_longest_edge(
    reference_mesh,
    max_edge_length=None,
    feature_angle_degrees=5.0,
    max_splits=None,
    coplanar_angle_degrees=0.1,
    flip_passes=8,
    flip_minimum_valence=None,
    flip_maximum_candidate_quality=None,
    planar_fan_minimum_valence=None,
    planar_annulus_minimum_faces=None,
    cylinder_minimum_faces=None,
    cylinder_radius_tolerance=1e-3,
    cylinder_target_edge_ratio=1.0,
    analytic_cylinder_recovery_distance=None,
    isolate_outer_cylinder_remainder=False,
    outer_cylinder_remainder_distance=0.2,
    trimmed_cylinder_minimum_faces=None,
    trimmed_cylinder_radius_tolerance=1e-2,
    trimmed_cylinder_normal_tolerance=8e-2,
    partial_cylinder_minimum_faces=None,
    partial_cylinder_radius_tolerance=2e-3,
    partial_cylinder_normal_tolerance=2e-2,
    partial_cylinder_minimum_angle=30.0,
    rounded_fillet_minimum_faces=None,
    rounded_fillet_minimum_curvature=0.2,
    rounded_fillet_maximum_source_quality=0.15,
    rounded_fillet_target_edge_ratio=4.0,
    rounded_fillet_minimum_triangle_angle=28.0,
    planar_region_minimum_faces=None,
    short_edge_collapse_ratio=0.5,
    short_edge_collapse_passes=3,
    topology_backend="auto",
    stop_after_curved_stages=False,
):
    """
    Refine the original surface while retaining hard feature edge lineages.

    Hard edges (dihedral > threshold, boundary, or non-manifold) are tracked as
    lineages. Splitting a hard edge replaces it by two collinear hard children.
    All edges are bisected to the requested length, then quality-improving flips
    remove non-feature seams only inside coplanar patches.
    """
    refinement_start = time.perf_counter()
    stage_timings = []
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
    short_edge_collapse_ratio = float(short_edge_collapse_ratio)
    short_edge_collapse_passes = int(short_edge_collapse_passes)
    if not 0.0 <= short_edge_collapse_ratio < 1.0:
        raise ValueError("Short-edge collapse ratio must be in [0, 1).")
    if short_edge_collapse_passes < 0:
        raise ValueError("Short-edge collapse passes must be non-negative.")
    requested_topology_backend = str(topology_backend).lower()
    if requested_topology_backend not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "Constraint topology backend must be auto, cpu, or cuda."
        )
    topology_backend = requested_topology_backend
    if requested_topology_backend == "auto":
        try:
            from .constrained_gpu import cuda_available

            # CUDA wins decisively on dense topology, while launch/context
            # overhead dominates the same operation on small local patches.
            topology_backend = (
                "cuda" if cuda_available() and len(faces) >= 16384 else "cpu"
            )
        except (ImportError, RuntimeError):
            topology_backend = "cpu"
    if topology_backend == "cuda":
        from .constrained_gpu import cuda_available

        if not cuda_available():
            raise RuntimeError(
                "CUDA constraint topology backend was requested but CUDA "
                "is unavailable."
            )
    print(
        "Constraint topology backend: {}.".format(topology_backend.upper()),
        flush=True,
    )
    if analytic_cylinder_recovery_distance is not None:
        analytic_cylinder_recovery_distance = float(
            analytic_cylinder_recovery_distance
        )
        if analytic_cylinder_recovery_distance <= 0.0:
            raise ValueError("Cylinder recovery distance must be positive.")
        if cylinder_minimum_faces is None:
            raise ValueError(
                "Cylinder recovery requires cylinder_minimum_faces."
            )
    outer_cylinder_remainder_distance = float(
        outer_cylinder_remainder_distance
    )
    if outer_cylinder_remainder_distance <= 0.0:
        raise ValueError("Outer-cylinder isolation distance must be positive.")
    if isolate_outer_cylinder_remainder and cylinder_minimum_faces is None:
        raise ValueError(
            "Outer-cylinder remainder isolation requires cylinder_minimum_faces."
        )

    print(
        "Constraint refinement input: {} vertices, {} faces."
        .format(len(vertices), len(faces)),
        flush=True,
    )

    stage_start = time.perf_counter()
    hard_edges_array = detect_hard_constraint_edges(
        reference_mesh,
        feature_angle_degrees=feature_angle_degrees,
    )
    stage_timings.append(("hard-edge detection", time.perf_counter() - stage_start))
    preferred_generated_edge_length = None
    if max_edge_length is not None:
        explicit_edge_limit = float(max_edge_length)
        if explicit_edge_limit <= 0.0:
            raise ValueError("Constraint maximum edge length must be positive.")
        # Hexagonal grid spacing is kept below the strict edge bound so
        # constrained-boundary transition triangles retain enough headroom.
        preferred_generated_edge_length = explicit_edge_limit * 0.5
        print(
            "Curved generated-edge target: {:.6g} (at most 25% growth per "
            "region toward the target; immutable short boundaries are "
            "preserved).".format(
                preferred_generated_edge_length
            ),
            flush=True,
        )

    brep_model_stats = None
    brep_target_edge_length = (
        float(max_edge_length) if max_edge_length is not None else None
    )
    if brep_target_edge_length is None:
        input_edges = _build_edge_faces(faces)
        input_lengths = _edge_length_array(vertices, input_edges)
        brep_target_edge_length, _, _ = automatic_edge_length_limit(
            vertices, input_lengths
        )
    stage_start = time.perf_counter()
    brep_result = retriangulate_brep_model_patches(
        vertices,
        faces,
        maximum_edge_length=brep_target_edge_length,
        cylinder_target_edge_ratio=rounded_fillet_target_edge_ratio,
        minimum_triangle_angle_degrees=(
            rounded_fillet_minimum_triangle_angle
        ),
    )
    if brep_result is not None:
        vertices, faces, hard_edges_array, brep_model_stats = brep_result
        stage_elapsed = time.perf_counter() - stage_start
        stage_timings.append(("B-rep patch remeshing", stage_elapsed))
        print(
            "Model-first B-rep remeshing: {} patch(es) = {} plane + {} "
            "cylinder; {} old -> {} new faces, {} interior vertices; "
            "{} shared boundary edges stitched exactly after {} boundary "
            "split(s); time {:.3f}s."
            .format(
                brep_model_stats["patches"],
                brep_model_stats["plane_patches"],
                brep_model_stats["cylinder_patches"],
                brep_model_stats["old_faces"],
                brep_model_stats["new_faces"],
                brep_model_stats["new_vertices"],
                brep_model_stats["boundary_edges"],
                brep_model_stats["boundary_splits"],
                stage_elapsed,
            ),
            flush=True,
        )
        for record in brep_model_stats["patch_records"]:
            print(
                "  patch {patch_id}: {surface_type}, {old_faces} -> "
                "{new_faces} faces, quality mean {old_quality:.6g} -> "
                "{new_quality:.6g}, P5 {old_quality_p5:.6g} -> "
                "{new_quality_p5:.6g}.".format(**record),
                flush=True,
            )
            if record.get("angular_columns") is not None:
                print(
                    "    preserved {} angular columns through {} axial "
                    "rows; local end size {:.6g}, interior edge limit "
                    "{:.6g} (2x).".format(
                        record["angular_columns"],
                        record["axial_rows"],
                        record["local_size"],
                        record["maximum_local_edge"],
                    ),
                    flush=True,
                )
        # The model-first transaction has already claimed every surface.
        # Generic heuristic passes must not repartition or overwrite it.
        cylinder_minimum_faces = None
        trimmed_cylinder_minimum_faces = None
        partial_cylinder_minimum_faces = None
        rounded_fillet_minimum_faces = None
        planar_annulus_minimum_faces = None
        planar_region_minimum_faces = None
        planar_fan_minimum_valence = None
        isolate_outer_cylinder_remainder = False
        short_edge_collapse_passes = 0
        flip_passes = 0

    curved_generated_vertex_start = len(vertices)
    isolated_transaction_edges = np.empty((0, 2), dtype=np.int64)
    isolated_faces_to_restore = np.empty((0, 3), dtype=np.int64)
    cylinder_seed_models = []
    cylinder_recovery_stats = None
    if cylinder_minimum_faces is not None:
        cylinder_seed_models = _discover_regular_cylinder_models(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=cylinder_minimum_faces,
            radius_tolerance=cylinder_radius_tolerance,
        )
        if analytic_cylinder_recovery_distance is not None:
            stage_start = time.perf_counter()
            selection_edges, selection_relaxed_edges = (
                _relax_false_cylinder_feature_edges(
                    vertices,
                    faces,
                    hard_edges_array,
                    cylinder_seed_models,
                )
            )
            selected_masks = []
            selected_components = 0
            selected_faces = 0
            for model in cylinder_seed_models:
                selected_mask, selected_stats = (
                    _select_connected_cylinder_support_faces(
                        vertices,
                        faces,
                        selection_edges,
                        model,
                        analytic_cylinder_recovery_distance,
                    )
                )
                selected_masks.append(selected_mask)
                selected_components += selected_stats["components"]
                selected_faces += int(np.count_nonzero(selected_mask))
            vertices, faces, cylinder_recovery_stats = (
                retriangulate_interrupted_cylindrical_walls(
                    vertices,
                    faces,
                    protected_edges=selection_edges,
                    cylinder_models=cylinder_seed_models,
                    minimum_faces=cylinder_minimum_faces,
                    radius_tolerance=cylinder_radius_tolerance,
                    preferred_edge_length=preferred_generated_edge_length,
                    minimum_support_alignment=0.75,
                    seam_guard_degrees=0.0,
                    selected_face_masks=selected_masks,
                    project_interior_to_source=False,
                )
            )
            hard_edges_array = selection_edges
            stage_elapsed = time.perf_counter() - stage_start
            stage_timings.append(("atomic analytic cylinder remesh", stage_elapsed))
            print(
                "Atomic analytic cylinder remesh: {} model(s), {} connected "
                "candidate component(s), {} selected faces; {} committed "
                "patch(es), {} old -> {} new faces, {} new vertices, {} "
                "pseudo seams relaxed; mean quality {:.6g} -> {:.6g} "
                "(guard {:.6g}); time {:.3f}s.".format(
                    len(cylinder_seed_models),
                    selected_components,
                    selected_faces,
                    cylinder_recovery_stats["patches"],
                    cylinder_recovery_stats["removed_faces"],
                    cylinder_recovery_stats["new_faces"],
                    cylinder_recovery_stats["new_vertices"],
                    selection_relaxed_edges,
                    cylinder_recovery_stats["old_quality"],
                    cylinder_recovery_stats["new_quality"],
                    float(analytic_cylinder_recovery_distance),
                    stage_elapsed,
                ),
                flush=True,
            )
            if cylinder_recovery_stats["rejections"]:
                print(
                    "Atomic analytic cylinder rollback reasons: {}."
                    .format(cylinder_recovery_stats["rejections"]),
                    flush=True,
                )
        hard_edges_array, relaxed_cylinder_edges = (
            _relax_false_cylinder_feature_edges(
                vertices,
                faces,
                hard_edges_array,
                cylinder_seed_models,
            )
        )
        if relaxed_cylinder_edges:
            print(
                "Cylinder support classification: relaxed {} false hard "
                "tessellation seam(s); embossed and boundary edges remain "
                "protected.".format(relaxed_cylinder_edges),
                flush=True,
            )
        outer_isolation_ready = False
        if isolate_outer_cylinder_remainder and cylinder_seed_models:
            isolation_probe_model = max(
                cylinder_seed_models,
                key=lambda item: float(item["radius"]),
            )
            _, isolation_probe_stats = (
                _select_connected_cylinder_support_faces(
                    vertices,
                    faces,
                    hard_edges_array,
                    isolation_probe_model,
                    outer_cylinder_remainder_distance,
                )
            )
            outer_isolation_ready = (
                isolation_probe_stats["components"] > 0
            )
            if not outer_isolation_ready:
                print(
                    "Outer-cylinder remainder isolation skipped: no "
                    "reliable connected cylinder candidate was found.",
                    flush=True,
                )
        if outer_isolation_ready:
            outer_model = max(
                cylinder_seed_models,
                key=lambda item: float(item["radius"]),
            )
            candidate_mask, isolation_selection_stats = (
                _select_connected_cylinder_support_faces(
                    vertices,
                    faces,
                    hard_edges_array,
                    outer_model,
                    outer_cylinder_remainder_distance,
                )
            )
            isolation_axis = np.asarray(
                outer_model["axis"], dtype=np.float64
            )
            isolation_origin = np.asarray(
                outer_model["origin"], dtype=np.float64
            )
            isolation_radius = float(outer_model["radius"])
            isolation_triangles = vertices[faces]
            isolation_centroids = isolation_triangles.mean(axis=1)
            isolation_offsets = isolation_centroids - isolation_origin
            isolation_axial = isolation_offsets @ isolation_axis
            isolation_radial = (
                isolation_offsets
                - isolation_axial[:, None] * isolation_axis
            )
            isolation_deviation = np.abs(
                np.linalg.norm(isolation_radial, axis=1) - isolation_radius
            )
            isolation_shell_halfwidth = max(
                outer_cylinder_remainder_distance * 2.5,
                isolation_radius * 5e-3,
            )
            isolation_axial_margin = max(
                outer_cylinder_remainder_distance,
                isolation_radius * 1e-3,
            )
            isolation_shell = (
                (
                    isolation_axial
                    >= float(outer_model["axial_min"])
                    - isolation_axial_margin
                )
                & (
                    isolation_axial
                    <= float(outer_model["axial_max"])
                    + isolation_axial_margin
                )
                & (isolation_deviation <= isolation_shell_halfwidth)
            )
            remainder_mask = isolation_shell & ~candidate_mask
            remainder_faces = faces[remainder_mask]
            remainder_edges = np.sort(
                remainder_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
                axis=1,
            )
            unique_remainder_edges, remainder_edge_counts = np.unique(
                remainder_edges,
                axis=0,
                return_counts=True,
            )
            virtual_boundary_edges = unique_remainder_edges[
                remainder_edge_counts == 1
            ]
            hard_edges_array = np.unique(
                np.vstack((hard_edges_array, virtual_boundary_edges)),
                axis=0,
            )
            print(
                "Outer-cylinder remainder isolation: {} input faces in {} "
                "candidate component(s), {} virtual boundary edges locked "
                "inside the full mesh (radius {:.6g}, guard {:.6g}).".format(
                    int(np.count_nonzero(remainder_mask)),
                    isolation_selection_stats["components"],
                    len(virtual_boundary_edges),
                    isolation_radius,
                    outer_cylinder_remainder_distance,
                ),
                flush=True,
            )
            local_reference_mesh = trimesh.Trimesh(
                vertices=vertices.copy(),
                faces=remainder_faces.copy(),
                process=False,
            )
            local_vertices, local_faces, local_remainder_stats = (
                refine_original_mesh_by_longest_edge(
                    local_reference_mesh,
                    max_edge_length=max_edge_length,
                    feature_angle_degrees=feature_angle_degrees,
                    max_splits=max_splits,
                    coplanar_angle_degrees=coplanar_angle_degrees,
                    flip_passes=flip_passes,
                    planar_fan_minimum_valence=planar_fan_minimum_valence,
                    planar_annulus_minimum_faces=(
                        planar_annulus_minimum_faces
                    ),
                    cylinder_minimum_faces=cylinder_minimum_faces,
                    cylinder_radius_tolerance=cylinder_radius_tolerance,
                    cylinder_target_edge_ratio=cylinder_target_edge_ratio,
                    analytic_cylinder_recovery_distance=None,
                    isolate_outer_cylinder_remainder=False,
                    trimmed_cylinder_minimum_faces=(
                        trimmed_cylinder_minimum_faces
                    ),
                    trimmed_cylinder_radius_tolerance=(
                        trimmed_cylinder_radius_tolerance
                    ),
                    trimmed_cylinder_normal_tolerance=(
                        trimmed_cylinder_normal_tolerance
                    ),
                    partial_cylinder_minimum_faces=(
                        partial_cylinder_minimum_faces
                    ),
                    partial_cylinder_radius_tolerance=(
                        partial_cylinder_radius_tolerance
                    ),
                    partial_cylinder_normal_tolerance=(
                        partial_cylinder_normal_tolerance
                    ),
                    partial_cylinder_minimum_angle=(
                        partial_cylinder_minimum_angle
                    ),
                    rounded_fillet_minimum_faces=(
                        rounded_fillet_minimum_faces
                    ),
                    rounded_fillet_minimum_curvature=(
                        rounded_fillet_minimum_curvature
                    ),
                    rounded_fillet_maximum_source_quality=(
                        rounded_fillet_maximum_source_quality
                    ),
                    rounded_fillet_target_edge_ratio=(
                        rounded_fillet_target_edge_ratio
                    ),
                    rounded_fillet_minimum_triangle_angle=(
                        rounded_fillet_minimum_triangle_angle
                    ),
                    planar_region_minimum_faces=(
                        planar_region_minimum_faces
                    ),
                    short_edge_collapse_ratio=short_edge_collapse_ratio,
                    short_edge_collapse_passes=short_edge_collapse_passes,
                    # A dense parent already selected CUDA, so keep it for
                    # nested patches to warm the context before the later
                    # global CUDA pass.  Standalone small meshes still use CPU.
                    topology_backend=topology_backend,
                    stop_after_curved_stages=False,
                )
            )
            kept_global_faces = faces[~remainder_mask]
            (
                kept_global_faces,
                stitched_outside_faces,
                stitched_boundary_edges,
                interface_buffer_edges,
            ) = _stitch_faces_to_isolated_boundary(
                local_vertices,
                kept_global_faces,
                local_faces,
                virtual_boundary_edges,
            )
            vertices = local_vertices
            isolated_faces_to_restore = local_faces.copy()
            faces = kept_global_faces
            # Replace every virtual parent interface edge with the actual
            # isolated output edge chain. This also covers parent edges whose
            # endpoints were duplicated or absorbed by a local rebuild.
            virtual_interface_set = {
                tuple(sorted((int(edge[0]), int(edge[1]))))
                for edge in virtual_boundary_edges
            }
            hard_edges_array = np.asarray(
                [
                    edge
                    for edge in hard_edges_array
                    if tuple(sorted((int(edge[0]), int(edge[1]))))
                    not in virtual_interface_set
                ],
                dtype=np.int64,
            ).reshape(-1, 2)
            local_owned_edges = np.unique(
                np.sort(
                    local_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
                    axis=1,
                ),
                axis=0,
            )
            local_edge_rows = np.sort(
                local_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
                axis=1,
            )
            local_unique_edges, local_edge_counts = np.unique(
                local_edge_rows, axis=0, return_counts=True
            )
            isolated_boundary_edges = local_unique_edges[
                local_edge_counts == 1
            ]
            isolated_transaction_edges = np.unique(
                np.vstack((isolated_boundary_edges, interface_buffer_edges)),
                axis=0,
            )
            hard_edges_array = np.unique(
                np.vstack(
                    (
                        hard_edges_array,
                        isolated_boundary_edges,
                        interface_buffer_edges,
                    )
                ),
                axis=0,
            )
            outside_edge_set = set(_build_edge_faces(faces))
            hard_edges_array = np.asarray(
                [
                    edge
                    for edge in hard_edges_array
                    if tuple(sorted((int(edge[0]), int(edge[1]))))
                    in outside_edge_set
                ],
                dtype=np.int64,
            ).reshape(-1, 2)
            print(
                "Complete isolated remainder remesh committed inside full "
                "mesh: {} old -> {} new local faces, {} adjacent outside "
                "face(s) conformed across {} subdivided interface edge(s), "
                "{} local edges held outside later global stages; the "
                "interface halo remains locked.".format(
                    len(remainder_faces),
                    len(local_faces),
                    stitched_outside_faces,
                    len(stitched_boundary_edges),
                    len(local_owned_edges),
                ),
                flush=True,
            )
            # The isolated transaction owns and locks all of its output edges.
            # Do not let its deliberately short feature/interface edges lower
            # the target lengths of later, unrelated global planar grading.
            curved_generated_vertex_start = len(vertices)
    working_reference_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    hard_edges = {
        tuple((int(edge[0]), int(edge[1])))
        for edge in hard_edges_array
    }
    coplanar_mask = (
        np.asarray(working_reference_mesh.face_adjacency_angles)
        <= np.deg2rad(coplanar_angle_degrees)
    )
    coplanar_edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(working_reference_mesh.face_adjacency_edges)[
            coplanar_mask
        ]
    }
    root_for_edge = {
        edge: root_index
        for root_index, edge in enumerate(sorted(hard_edges))
    }
    original_hard_lengths = _edge_length_array(vertices, sorted(hard_edges))
    reference_feature_edges = hard_edges_array.copy()
    reference_vertices = vertices.copy()
    reference_faces = (
        np.vstack((faces, isolated_faces_to_restore))
        if len(isolated_faces_to_restore)
        else faces.copy()
    )
    if cylinder_minimum_faces is not None:
        # Recover the base support around embossed cut-outs before closed-wall
        # candidates consume adjacent regular bands.  Once those bands are
        # replaced, the remaining outer support can become a multiply-holed
        # periodic domain whose original boundary cycles are no longer
        # available to the interrupted-wall pass.
        stage_start = time.perf_counter()
        vertices, faces, preclosed_interrupted_stats = (
            retriangulate_interrupted_cylindrical_walls(
                vertices,
                faces,
                protected_edges=hard_edges_array,
                cylinder_models=cylinder_seed_models,
                minimum_faces=cylinder_minimum_faces,
                radius_tolerance=cylinder_radius_tolerance,
                preferred_edge_length=preferred_generated_edge_length,
                minimum_support_alignment=0.75,
                excluded_edges=isolated_transaction_edges,
            )
        )
        stage_elapsed = time.perf_counter() - stage_start
        stage_timings.append(
            ("pre-closed interrupted cylindrical walls", stage_elapsed)
        )
        print(
            "Pre-closed interrupted cylinder recovery: {} / {} candidate "
            "patch(es), {} new vertices, {} old -> {} new faces; mean "
            "quality {:.6g} -> {:.6g}; time {:.3f}s.".format(
                preclosed_interrupted_stats["patches"],
                preclosed_interrupted_stats["candidates"],
                preclosed_interrupted_stats["new_vertices"],
                preclosed_interrupted_stats["removed_faces"],
                preclosed_interrupted_stats["new_faces"],
                preclosed_interrupted_stats["old_quality"],
                preclosed_interrupted_stats["new_quality"],
                stage_elapsed,
            ),
            flush=True,
        )

        stage_start = time.perf_counter()
        vertices, faces, cylinder_stats = retriangulate_cylindrical_walls(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=cylinder_minimum_faces,
            radius_tolerance=cylinder_radius_tolerance,
            target_edge_ratio=cylinder_target_edge_ratio,
            preferred_edge_length=preferred_generated_edge_length,
            direct_target_spacing=True,
            maximum_face_count_ratio=2.0,
        )
        stage_elapsed = time.perf_counter() - stage_start
        stage_timings.append(("cylindrical walls", stage_elapsed))
        print(
            "Cylindrical wall retriangulation: {} / {} candidate wall(s), "
            "{} new vertices, {} old -> {} new faces; mean quality "
            "{:.6g} -> {:.6g}; time {:.3f}s.".format(
                cylinder_stats["walls"], cylinder_stats["candidates"],
                cylinder_stats["new_vertices"], cylinder_stats["removed_faces"],
                cylinder_stats["new_faces"], cylinder_stats["old_quality"],
                cylinder_stats["new_quality"],
                stage_elapsed,
            ),
            flush=True,
        )

        stage_start = time.perf_counter()
        vertices, faces, interrupted_stats = (
            retriangulate_interrupted_cylindrical_walls(
                vertices,
                faces,
                protected_edges=hard_edges_array,
                cylinder_models=cylinder_seed_models,
                minimum_faces=cylinder_minimum_faces,
                radius_tolerance=cylinder_radius_tolerance,
                preferred_edge_length=preferred_generated_edge_length,
                minimum_support_alignment=0.75,
                excluded_edges=isolated_transaction_edges,
            )
        )
        stage_elapsed = time.perf_counter() - stage_start
        stage_timings.append(("interrupted cylindrical walls", stage_elapsed))
        print(
            "Interrupted cylindrical wall recovery: {} / {} candidate "
            "patch(es), {} new vertices, {} old -> {} new faces; mean "
            "quality {:.6g} -> {:.6g}; time {:.3f}s.".format(
                interrupted_stats["patches"], interrupted_stats["candidates"],
                interrupted_stats["new_vertices"],
                interrupted_stats["removed_faces"],
                interrupted_stats["new_faces"], interrupted_stats["old_quality"],
                interrupted_stats["new_quality"], stage_elapsed,
            ),
            flush=True,
        )

        stage_start = time.perf_counter()
        vertices, faces, periodic_stats = (
            retriangulate_interrupted_cylindrical_walls(
                vertices,
                faces,
                protected_edges=hard_edges_array,
                cylinder_models=cylinder_seed_models,
                minimum_faces=cylinder_minimum_faces,
                radius_tolerance=cylinder_radius_tolerance,
                preferred_edge_length=preferred_generated_edge_length,
                seam_angle_offset=np.pi,
                minimum_support_alignment=0.75,
                excluded_edges=isolated_transaction_edges,
            )
        )
        stage_elapsed = time.perf_counter() - stage_start
        stage_timings.append(("periodic cylinder seam closure", stage_elapsed))
        print(
            "Periodic cylinder seam closure: {} / {} candidate patch(es), "
            "{} new vertices, {} old -> {} new faces; mean quality "
            "{:.6g} -> {:.6g}; time {:.3f}s.".format(
                periodic_stats["patches"], periodic_stats["candidates"],
                periodic_stats["new_vertices"], periodic_stats["removed_faces"],
                periodic_stats["new_faces"], periodic_stats["old_quality"],
                periodic_stats["new_quality"], stage_elapsed,
            ),
            flush=True,
        )

        for phase_name, phase_offset in (
            ("Quarter-phase cylinder interface recovery", 0.5 * np.pi),
            ("Three-quarter-phase cylinder interface recovery", 1.5 * np.pi),
        ):
            stage_start = time.perf_counter()
            vertices, faces, phase_stats = (
                retriangulate_interrupted_cylindrical_walls(
                    vertices,
                    faces,
                    protected_edges=hard_edges_array,
                    cylinder_models=cylinder_seed_models,
                    minimum_faces=cylinder_minimum_faces,
                    radius_tolerance=cylinder_radius_tolerance,
                    preferred_edge_length=preferred_generated_edge_length,
                    seam_angle_offset=phase_offset,
                    minimum_support_alignment=0.75,
                    excluded_edges=isolated_transaction_edges,
                )
            )
            stage_elapsed = time.perf_counter() - stage_start
            stage_timings.append((phase_name.lower(), stage_elapsed))
            print(
                "{}: {} / {} candidate patch(es), {} new vertices, "
                "{} old -> {} new faces; mean quality {:.6g} -> {:.6g}; "
                "time {:.3f}s.".format(
                    phase_name,
                    phase_stats["patches"], phase_stats["candidates"],
                    phase_stats["new_vertices"], phase_stats["removed_faces"],
                    phase_stats["new_faces"], phase_stats["old_quality"],
                    phase_stats["new_quality"], stage_elapsed,
                ),
                flush=True,
            )

        # The guarded passes above avoid cutting a developed patch across its
        # angular branch.  Embossed features can pin a few faces inside every
        # shifted guard, however, leaving a conspicuous vertical sliver.  A
        # final branch-aware pass closes only those residual support patches.
        stage_start = time.perf_counter()
        vertices, faces, residual_seam_stats = (
            retriangulate_interrupted_cylindrical_walls(
                vertices,
                faces,
                protected_edges=hard_edges_array,
                cylinder_models=cylinder_seed_models,
                minimum_faces=2,
                radius_tolerance=cylinder_radius_tolerance,
                preferred_edge_length=preferred_generated_edge_length,
                minimum_support_alignment=0.75,
                seam_guard_degrees=0.0,
                excluded_edges=isolated_transaction_edges,
            )
        )
        stage_elapsed = time.perf_counter() - stage_start
        stage_timings.append(("residual cylinder seam closure", stage_elapsed))
        print(
            "Residual cylinder seam closure: {} / {} candidate patch(es), "
            "{} new vertices, {} old -> {} new faces; mean quality "
            "{:.6g} -> {:.6g}; time {:.3f}s.".format(
                residual_seam_stats["patches"],
                residual_seam_stats["candidates"],
                residual_seam_stats["new_vertices"],
                residual_seam_stats["removed_faces"],
                residual_seam_stats["new_faces"],
                residual_seam_stats["old_quality"],
                residual_seam_stats["new_quality"],
                stage_elapsed,
            ),
            flush=True,
        )

    # Surface ownership order is deliberate: every full, trimmed, or partial
    # cylinder claims its support before blend bands or generic surfaces.
    if trimmed_cylinder_minimum_faces is not None:
        stage_start = time.perf_counter()
        vertices, faces, trimmed_stats = retriangulate_trimmed_cylindrical_walls(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=trimmed_cylinder_minimum_faces,
            radius_tolerance=trimmed_cylinder_radius_tolerance,
            normal_tolerance=trimmed_cylinder_normal_tolerance,
            target_edge_ratio=cylinder_target_edge_ratio,
            regular_radius_tolerance=cylinder_radius_tolerance,
            preferred_edge_length=preferred_generated_edge_length,
        )
        stage_elapsed = time.perf_counter() - stage_start
        stage_timings.append(("trimmed cylinders", stage_elapsed))
        print(
            "Trimmed cylindrical wall retriangulation: {} / {} candidate "
            "wall(s), {} new vertices, {} old -> {} new faces; mean quality "
            "{:.6g} -> {:.6g}; time {:.3f}s.".format(
                trimmed_stats["walls"], trimmed_stats["candidates"],
                trimmed_stats["new_vertices"], trimmed_stats["removed_faces"],
                trimmed_stats["new_faces"], trimmed_stats["old_quality"],
                trimmed_stats["new_quality"],
                stage_elapsed,
            ),
            flush=True,
        )

    if partial_cylinder_minimum_faces is not None:
        stage_start = time.perf_counter()
        vertices, faces, partial_stats = (
            retriangulate_partial_cylindrical_walls(
                vertices,
                faces,
                protected_edges=hard_edges_array,
                minimum_faces=partial_cylinder_minimum_faces,
                radius_tolerance=partial_cylinder_radius_tolerance,
                normal_tolerance=partial_cylinder_normal_tolerance,
                minimum_angle_degrees=partial_cylinder_minimum_angle,
                target_edge_ratio=cylinder_target_edge_ratio,
                preferred_edge_length=preferred_generated_edge_length,
                direct_target_spacing=True,
                allow_density_simplification=True,
                maximum_face_count_ratio=2.0,
                model_guided_boundary_recovery=True,
            )
        )
        stage_elapsed = time.perf_counter() - stage_start
        stage_timings.append(("partial cylinders", stage_elapsed))
        print(
            "Partial cylindrical wall retriangulation: {} / {} candidate "
            "patch(es), {} new vertices, {} old -> {} new faces; mean "
            "quality {:.6g} -> {:.6g}; time {:.3f}s.".format(
                partial_stats["patches"], partial_stats["candidates"],
                partial_stats["new_vertices"], partial_stats["removed_faces"],
                partial_stats["new_faces"], partial_stats["old_quality"],
                partial_stats["new_quality"],
                stage_elapsed,
            ),
            flush=True,
        )

    if len(isolated_transaction_edges):
        curved_edge_faces = _build_edge_faces(faces)
        missing_owned = [
            tuple(map(int, edge))
            for edge in isolated_transaction_edges
            if tuple(map(int, edge)) not in curved_edge_faces
        ]
        if missing_owned:
            raise RuntimeError(
                "Cylinder stages crossed the isolated transaction halo; "
                "first missing edge: {}.".format(missing_owned[0])
            )

    # Persist cylinder ownership across all later stages.  Earlier versions
    # only kept a per-function claimed-face mask, so fillet and planar passes
    # could enter a cylinder that had already been successfully rebuilt.
    cylinder_owned_faces = np.zeros(len(faces), dtype=bool)
    if cylinder_seed_models:
        ownership_triangles = vertices[faces]
        ownership_crosses = np.cross(
            ownership_triangles[:, 1] - ownership_triangles[:, 0],
            ownership_triangles[:, 2] - ownership_triangles[:, 0],
        )
        ownership_lengths = np.linalg.norm(ownership_crosses, axis=1)
        ownership_normals = np.zeros_like(ownership_crosses)
        ownership_valid = ownership_lengths > 0.0
        ownership_normals[ownership_valid] = (
            ownership_crosses[ownership_valid]
            / ownership_lengths[ownership_valid, None]
        )
        ownership_centroids = ownership_triangles.mean(axis=1)
        for model in cylinder_seed_models:
            ownership_axis = np.asarray(model["axis"], dtype=np.float64)
            ownership_origin = np.asarray(model["origin"], dtype=np.float64)
            ownership_radius = float(model["radius"])
            ownership_offsets = ownership_centroids - ownership_origin
            ownership_axial = ownership_offsets @ ownership_axis
            ownership_radial = (
                ownership_offsets
                - ownership_axial[:, None] * ownership_axis
            )
            ownership_radii = np.linalg.norm(ownership_radial, axis=1)
            ownership_alignment = np.abs(
                np.sum(ownership_normals * ownership_radial, axis=1)
                / np.maximum(ownership_radii, 1e-30)
            )
            ownership_tolerance = max(
                ownership_radius * 5e-3,
                2e-2,
            )
            ownership_axial_ok = np.ones(len(faces), dtype=bool)
            if "axial_min" in model and "axial_max" in model:
                ownership_axial_ok = (
                    ownership_axial
                    >= float(model["axial_min"]) - ownership_tolerance
                ) & (
                    ownership_axial
                    <= float(model["axial_max"]) + ownership_tolerance
                )
            cylinder_owned_faces |= (
                ownership_valid
                & ownership_axial_ok
                & (ownership_alignment >= 0.75)
                & (
                    np.abs(ownership_radii - ownership_radius)
                    <= ownership_tolerance
                )
            )
    if np.any(cylinder_owned_faces):
        owned_edges = np.unique(
            np.sort(
                faces[cylinder_owned_faces][
                    :, ((0, 1), (1, 2), (2, 0))
                ].reshape(-1, 2),
                axis=1,
            ),
            axis=0,
        )
        new_owned_edges = [
            tuple(map(int, edge))
            for edge in owned_edges
            if tuple(map(int, edge)) not in root_for_edge
        ]
        if new_owned_edges:
            first_root = len(original_hard_lengths)
            new_owned_lengths = _edge_length_array(
                vertices, new_owned_edges
            )
            original_hard_lengths = np.concatenate(
                (original_hard_lengths, new_owned_lengths)
            )
            for offset, edge in enumerate(new_owned_edges):
                hard_edges.add(edge)
                root_for_edge[edge] = first_root + offset
                coplanar_edges.discard(edge)
        hard_edges_array = np.asarray(
            sorted(root_for_edge), dtype=np.int64
        ).reshape(-1, 2)
        reference_feature_edges = hard_edges_array.copy()
        reference_vertices = vertices.copy()
        reference_faces = (
            np.vstack((faces, isolated_faces_to_restore))
            if len(isolated_faces_to_restore)
            else faces.copy()
        )
        print(
            "Persistent cylinder ownership: {} face(s), {} edge(s) locked "
            "against fillet, generic curved, and planar stages.".format(
                int(np.count_nonzero(cylinder_owned_faces)),
                len(owned_edges),
            ),
            flush=True,
        )

    if rounded_fillet_minimum_faces is not None:
        stage_start = time.perf_counter()
        vertices, faces, fillet_stats = retriangulate_partial_cylindrical_walls(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=rounded_fillet_minimum_faces,
            radius_tolerance=max(partial_cylinder_radius_tolerance, 5e-2),
            normal_tolerance=max(partial_cylinder_normal_tolerance, 1.2e-1),
            minimum_angle_degrees=5.0,
            target_edge_ratio=rounded_fillet_target_edge_ratio,
            isolate_rounded_faces=True,
            minimum_curvature_degrees=rounded_fillet_minimum_curvature,
            maximum_source_quality=rounded_fillet_maximum_source_quality,
            minimum_triangle_angle_degrees=(
                rounded_fillet_minimum_triangle_angle
            ),
            minimum_radial_alignment=0.95,
            preferred_edge_length=preferred_generated_edge_length,
            model_guided_boundary_recovery=True,
            direct_target_spacing=True,
            allow_density_simplification=True,
            maximum_face_count_ratio=2.0,
        )
        stage_elapsed = time.perf_counter() - stage_start
        stage_timings.append(("rounded fillets", stage_elapsed))
        old_fillet_p5 = fillet_stats.get("old_quality_p5", 0.0)
        new_fillet_p5 = fillet_stats.get("new_quality_p5", 0.0)
        print(
            "Rounded fillet retriangulation: {} / {} candidate band(s), "
            "{} new vertices, {} old -> {} new faces; mean quality "
            "{:.6g} -> {:.6g}; P5 {:.6g} -> {:.6g}; time {:.3f}s.".format(
                fillet_stats["patches"], fillet_stats["candidates"],
                fillet_stats["new_vertices"], fillet_stats["removed_faces"],
                fillet_stats["new_faces"], fillet_stats["old_quality"],
                fillet_stats["new_quality"],
                old_fillet_p5, new_fillet_p5,
                stage_elapsed,
            ),
            flush=True,
        )

    if len(isolated_transaction_edges):
        fillet_edge_faces = _build_edge_faces(faces)
        missing_owned = [
            tuple(map(int, edge))
            for edge in isolated_transaction_edges
            if tuple(map(int, edge)) not in fillet_edge_faces
        ]
        if missing_owned:
            raise RuntimeError(
                "Fillet stage crossed the isolated transaction halo; "
                "first missing edge: {}.".format(missing_owned[0])
            )

    if stop_after_curved_stages:
        local_edge_faces = _build_edge_faces(faces)
        local_lengths = _edge_length_array(vertices, local_edge_faces)
        return vertices, faces, {
            "hard_edges": len(root_for_edge),
            "hard_edges_split": 0,
            "splits": 0,
            "initial_max_length": (
                float(local_lengths.max()) if len(local_lengths) else 0.0
            ),
            "final_max_length": (
                float(local_lengths.max()) if len(local_lengths) else 0.0
            ),
            "already_satisfied": False,
            "coplanar_flips": 0,
            "reference_feature_edges": np.asarray(
                sorted(root_for_edge), dtype=np.int64
            ).reshape(-1, 2),
            "reference_vertices": vertices.copy(),
            "reference_faces": faces.copy(),
            "analytic_cylinder_recovery": cylinder_recovery_stats,
            "brep_model_first": brep_model_stats,
        }

    curved_generated_vertex_end = len(vertices)

    def retriangulate_planar_after_boundary_refinement(
        current_vertices, current_faces, current_hard_edges, edge_target
    ):
        """Rebuild planes only after curved shared boundaries are subdivided."""
        # Use the same requested scale as the rest of constrained refinement.
        # Transition edges are checked against ``edge_target`` below, so a
        # planar rebuild that cannot satisfy the strict limit is rejected
        # instead of globally shrinking every large planar face.
        planar_edge_target = float(edge_target)
        minimum_planar_area = planar_edge_target * planar_edge_target * 2.0
        print(
            "Planar generated-edge target: {:.6g}; minimum rebuilt region "
            "area {:.6g}.".format(planar_edge_target, minimum_planar_area),
            flush=True,
        )
        if planar_annulus_minimum_faces is not None:
            planar_start = time.perf_counter()
            current_vertices, current_faces, annulus_stats = (
                retriangulate_planar_annuli(
                    current_vertices,
                    current_faces,
                    protected_edges=current_hard_edges,
                    minimum_faces=planar_annulus_minimum_faces,
                    maximum_target_edge_length=planar_edge_target,
                    maximum_result_edge_length=edge_target,
                    # Dense hole contours need quality-driven Steiner points
                    # so their small boundary edges grow into the coarse
                    # planar target through several transition rings.
                    minimum_angle_degrees=28.0,
                    accept_strong_mean_gain=True,
                    skip_satisfactory_quality=True,
                    minimum_region_area=minimum_planar_area,
                    gradual_target_spacing=False,
                )
            )
            planar_elapsed = time.perf_counter() - planar_start
            stage_timings.append(("planar regions with holes", planar_elapsed))
            print(
                "Constrained planar annulus retriangulation: {} region(s), "
                "{} new vertices, {} old -> {} new faces; mean quality "
                "{:.6g} -> {:.6g}; {} candidates ({} hard-constraint, {} "
                "boundary, {} quality rejected); time {:.3f}s.".format(
                    annulus_stats["regions"], annulus_stats["new_vertices"],
                    annulus_stats["removed_faces"], annulus_stats["new_faces"],
                    annulus_stats["old_quality"], annulus_stats["new_quality"],
                    annulus_stats["candidates"],
                    annulus_stats["constraint_rejections"],
                    annulus_stats["boundary_rejections"],
                    annulus_stats["quality_rejections"],
                    planar_elapsed,
                ),
                flush=True,
            )
            if (
                annulus_stats["boundary_rejections"]
                or annulus_stats["quality_rejections"]
            ):
                fallback_start = time.perf_counter()
                current_vertices, current_faces, fallback_stats = (
                    retriangulate_planar_annuli(
                        current_vertices,
                        current_faces,
                        protected_edges=current_hard_edges,
                        minimum_faces=planar_annulus_minimum_faces,
                        maximum_target_edge_length=planar_edge_target,
                        maximum_result_edge_length=edge_target,
                        minimum_angle_degrees=28.0,
                        accept_strong_mean_gain=True,
                        skip_satisfactory_quality=True,
                        minimum_region_area=minimum_planar_area,
                        # Complex multi-hole planes can reject the coarse
                        # first pass. Retry only those remaining regions with
                        # conservative growth from their dense boundary size.
                        gradual_target_spacing=True,
                    )
                )
                fallback_elapsed = time.perf_counter() - fallback_start
                stage_timings.append(
                    ("planar hole transition fallback", fallback_elapsed)
                )
                print(
                    "Conservative planar-hole transition fallback: {} "
                    "region(s), {} new vertices, {} old -> {} new faces; "
                    "mean quality {:.6g} -> {:.6g}; {} candidates ({} "
                    "hard-constraint, {} boundary, {} quality rejected); "
                    "time {:.3f}s.".format(
                        fallback_stats["regions"],
                        fallback_stats["new_vertices"],
                        fallback_stats["removed_faces"],
                        fallback_stats["new_faces"],
                        fallback_stats["old_quality"],
                        fallback_stats["new_quality"],
                        fallback_stats["candidates"],
                        fallback_stats["constraint_rejections"],
                        fallback_stats["boundary_rejections"],
                        fallback_stats["quality_rejections"],
                        fallback_elapsed,
                    ),
                    flush=True,
                )
        if planar_region_minimum_faces is not None:
            planar_start = time.perf_counter()
            current_vertices, current_faces, planar_stats = (
                retriangulate_planar_annuli(
                    current_vertices,
                    current_faces,
                    protected_edges=current_hard_edges,
                    minimum_faces=planar_region_minimum_faces,
                    minimum_holes=0,
                    maximum_holes=0,
                    maximum_target_edge_length=planar_edge_target,
                    maximum_result_edge_length=edge_target,
                    minimum_angle_degrees=20.0,
                    accept_strong_mean_gain=True,
                    skip_satisfactory_quality=True,
                    minimum_region_area=minimum_planar_area,
                    # Fine boundary segments must only constrain a local
                    # transition band. Seed the plane interior at the global
                    # target; Triangle adds quality-required Steiner points
                    # near the immutable boundary without propagating its
                    # small median spacing across the whole planar region.
                    gradual_target_spacing=False,
                )
            )
            planar_elapsed = time.perf_counter() - planar_start
            stage_timings.append(("solid planar regions", planar_elapsed))
            print(
                "Constrained solid planar retriangulation: {} region(s), {} "
                "new vertices, {} old -> {} new faces; mean quality {:.6g} "
                "-> {:.6g}; {} candidates ({} hard-constraint, {} boundary, "
                "{} quality rejected); time {:.3f}s.".format(
                    planar_stats["regions"], planar_stats["new_vertices"],
                    planar_stats["removed_faces"], planar_stats["new_faces"],
                    planar_stats["old_quality"], planar_stats["new_quality"],
                    planar_stats["candidates"],
                    planar_stats["constraint_rejections"],
                    planar_stats["boundary_rejections"],
                    planar_stats["quality_rejections"],
                    planar_elapsed,
                ),
                flush=True,
            )
            grading_start = time.perf_counter()
            current_vertices, current_faces, grading_stats = (
                retriangulate_planar_annuli(
                    current_vertices,
                    current_faces,
                    protected_edges=current_hard_edges,
                    minimum_faces=min(8, int(planar_region_minimum_faces)),
                    minimum_holes=0,
                    maximum_holes=0,
                    maximum_target_edge_length=planar_edge_target,
                    maximum_result_edge_length=edge_target,
                    minimum_angle_degrees=28.0,
                    accept_strong_mean_gain=True,
                    skip_satisfactory_quality=False,
                    minimum_region_area=None,
                    # Seed a coarse interior grid; Triangle adds only the
                    # quality-required rings near the dense curved boundary,
                    # producing a genuinely graded size transition.
                    gradual_target_spacing=False,
                    required_boundary_vertex_range=(
                        curved_generated_vertex_start,
                        curved_generated_vertex_end,
                    ),
                    minimum_boundary_length_ratio=4.0,
                )
            )
            grading_elapsed = time.perf_counter() - grading_start
            stage_timings.append(
                ("curve-adjacent planar grading", grading_elapsed)
            )
            print(
                "Curve-adjacent planar grading: {} region(s), {} new "
                "vertices, {} old -> {} new faces; mean quality {:.6g} -> "
                "{:.6g}; {} candidates ({} boundary, {} quality rejected); "
                "time {:.3f}s.".format(
                    grading_stats["regions"],
                    grading_stats["new_vertices"],
                    grading_stats["removed_faces"],
                    grading_stats["new_faces"],
                    grading_stats["old_quality"],
                    grading_stats["new_quality"],
                    grading_stats["candidates"],
                    grading_stats["boundary_rejections"],
                    grading_stats["quality_rejections"],
                    grading_elapsed,
                ),
                flush=True,
            )
            transition_targets, transition_regions = (
                _planar_transition_edge_targets(
                    current_vertices,
                    current_faces,
                    current_hard_edges,
                    minimum_faces=planar_region_minimum_faces,
                    maximum_region_area=minimum_planar_area,
                    maximum_edge_length=edge_target,
                    generated_vertex_range=(
                        curved_generated_vertex_start,
                        curved_generated_vertex_end,
                    ),
                )
            )
            if transition_targets:
                transition_start = time.perf_counter()
                transition_vertex_start = len(current_vertices)
                (
                    current_vertices,
                    current_faces,
                    updated_lineages,
                    updated_coplanar_edges,
                    split_stats,
                ) = _split_selected_edges_by_length(
                    current_vertices,
                    current_faces,
                    transition_targets,
                    edge_lineages=root_for_edge,
                    propagated_edges=coplanar_edges,
                )
                if split_stats["hit_split_limit"]:
                    raise RuntimeError(
                        "Planar transition-edge split limit was reached."
                    )
                root_for_edge.clear()
                root_for_edge.update(updated_lineages)
                coplanar_edges.clear()
                coplanar_edges.update(updated_coplanar_edges)
                current_hard_edges = np.asarray(
                    sorted(root_for_edge), dtype=np.int64
                ).reshape(-1, 2)
                current_vertices, current_faces, transition_stats = (
                    retriangulate_planar_annuli(
                        current_vertices,
                        current_faces,
                        protected_edges=current_hard_edges,
                        minimum_faces=planar_region_minimum_faces,
                        minimum_holes=0,
                        maximum_holes=0,
                        maximum_target_edge_length=planar_edge_target,
                        maximum_result_edge_length=edge_target,
                        minimum_angle_degrees=28.0,
                        accept_strong_mean_gain=True,
                        skip_satisfactory_quality=True,
                        # This recovery iteration intentionally includes the
                        # narrow planar strips skipped by the main area gate.
                        minimum_region_area=None,
                        # Keep fine curve contacts local. The constrained
                        # boundary and quality angle create the graded band;
                        # the rest of each plane uses the global target.
                        gradual_target_spacing=False,
                    )
                )
                transition_elapsed = time.perf_counter() - transition_start
                stage_timings.append(
                    ("curve-to-plane transition recovery", transition_elapsed)
                )
                print(
                    "Curve-to-plane transition recovery: {} narrow region(s), "
                    "{} conforming edge split(s), {} planar region(s), {} "
                    "new interior vertices, {} old -> {} new faces; mean "
                    "quality {:.6g} -> {:.6g}; time {:.3f}s.".format(
                        transition_regions,
                        split_stats["splits"],
                        transition_stats["regions"],
                        transition_stats["new_vertices"],
                        transition_stats["removed_faces"],
                        transition_stats["new_faces"],
                        transition_stats["old_quality"],
                        transition_stats["new_quality"],
                        transition_elapsed,
                    ),
                    flush=True,
                )
                outer_seed_range = (
                    transition_vertex_start,
                    len(current_vertices),
                )
                transition_base_target = float(
                    np.median(list(transition_targets.values()))
                )
                for ring_index in range(3):
                    ring_target = min(
                        float(edge_target),
                        transition_base_target * (2.0 ** (ring_index + 1)),
                    )
                    outer_targets, outer_regions = (
                        _planar_transition_edge_targets(
                            current_vertices,
                            current_faces,
                            current_hard_edges,
                            minimum_faces=min(
                                8, int(planar_region_minimum_faces)
                            ),
                            maximum_region_area=(
                                minimum_planar_area * (4.0 ** (ring_index + 1))
                            ),
                            maximum_edge_length=edge_target,
                            maximum_quality_p5=1.01,
                            growth_ratio=2.0,
                            generated_vertex_range=outer_seed_range,
                            minimum_boundary_ratio=2.0,
                            minimum_local_target=ring_target,
                        )
                    )
                    if not outer_targets:
                        break
                    outer_grading_start = time.perf_counter()
                    outer_vertex_start = len(current_vertices)
                    (
                        current_vertices,
                        current_faces,
                        updated_lineages,
                        updated_coplanar_edges,
                        outer_split_stats,
                    ) = _split_selected_edges_by_length(
                        current_vertices,
                        current_faces,
                        outer_targets,
                        edge_lineages=root_for_edge,
                        propagated_edges=coplanar_edges,
                    )
                    if outer_split_stats["hit_split_limit"]:
                        raise RuntimeError(
                            "Outer planar transition split limit was reached."
                        )
                    root_for_edge.clear()
                    root_for_edge.update(updated_lineages)
                    coplanar_edges.clear()
                    coplanar_edges.update(updated_coplanar_edges)
                    current_hard_edges = np.asarray(
                        sorted(root_for_edge), dtype=np.int64
                    ).reshape(-1, 2)
                    outer_split_end = len(current_vertices)
                    current_vertices, current_faces, outer_stats = (
                        retriangulate_planar_annuli(
                            current_vertices,
                            current_faces,
                            protected_edges=current_hard_edges,
                            minimum_faces=min(
                                8, int(planar_region_minimum_faces)
                            ),
                            minimum_holes=0,
                            maximum_holes=0,
                            maximum_target_edge_length=ring_target,
                            maximum_result_edge_length=edge_target,
                            minimum_angle_degrees=28.0,
                            accept_strong_mean_gain=True,
                            skip_satisfactory_quality=False,
                            minimum_region_area=None,
                            gradual_target_spacing=False,
                            required_boundary_vertex_range=(
                                outer_vertex_start,
                                outer_split_end,
                            ),
                        )
                    )
                    outer_grading_elapsed = (
                        time.perf_counter() - outer_grading_start
                    )
                    stage_timings.append(
                        (
                            "outer planar transition ring {}".format(
                                ring_index + 1
                            ),
                            outer_grading_elapsed,
                        )
                    )
                    print(
                        "Outer planar transition ring {} (target {:.6g}): {} "
                        "region(s), {} "
                        "conforming edge split(s), {} rebuilt region(s), {} "
                        "new interior vertices; mean quality {:.6g} -> "
                        "{:.6g}; time {:.3f}s.".format(
                            ring_index + 1,
                            ring_target,
                            outer_regions,
                            outer_split_stats["splits"],
                            outer_stats["regions"],
                            outer_stats["new_vertices"],
                            outer_stats["old_quality"],
                            outer_stats["new_quality"],
                            outer_grading_elapsed,
                        ),
                        flush=True,
                    )
                    outer_seed_range = (
                        outer_vertex_start,
                        len(current_vertices),
                    )
        return current_vertices, current_faces

    def print_timing_summary():
        total_elapsed = time.perf_counter() - refinement_start
        details = ", ".join(
            "{} {:.3f}s".format(name, elapsed)
            for name, elapsed in stage_timings
        )
        print(
            "Constraint timing summary: {}; total {:.3f}s.".format(
                details or "no optional stages", total_elapsed
            ),
            flush=True,
        )

    def print_edge_length_summary(lengths):
        lengths = np.asarray(lengths, dtype=np.float64)
        lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
        if len(lengths) == 0:
            return
        p10, median, p90 = np.percentile(lengths, (10.0, 50.0, 90.0))
        coefficient = float(lengths.std() / max(lengths.mean(), 1e-30))
        print(
            "Edge-length distribution: P10 {:.6g}, median {:.6g}, "
            "P90 {:.6g}, CV {:.6g}.".format(
                p10, median, p90, coefficient
            ),
            flush=True,
        )

    edge_faces = _build_edge_faces(faces)
    initial_lengths = _edge_length_array(vertices, edge_faces)
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
            "Automatic maximum edge length: bounding-box diagonal 5% "
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
        optimized_vertices, optimized_faces = (
            retriangulate_planar_after_boundary_refinement(
                optimized_vertices,
                optimized_faces,
                hard_edges_array,
                max_edge_length,
            )
        )
        optimized_hard_edges = np.asarray(
            sorted(root_for_edge), dtype=np.int64
        ).reshape(-1, 2)
        collapse_count = 0
        if short_edge_collapse_passes and short_edge_collapse_ratio > 0.0:
            stage_start = time.perf_counter()
            optimized_vertices, optimized_faces, collapse_count = (
                collapse_short_coplanar_edges_with_backend(
                    optimized_vertices,
                    optimized_faces,
                    optimized_hard_edges,
                    maximum_short_edge_length=(
                        max_edge_length * float(short_edge_collapse_ratio)
                    ),
                    maximum_edge_length=max_edge_length,
                    maximum_planar_angle_degrees=coplanar_angle_degrees,
                    passes=short_edge_collapse_passes,
                    backend=topology_backend,
                )
            )
            stage_elapsed = time.perf_counter() - stage_start
            stage_timings.append(("safe short-edge collapse", stage_elapsed))
            print(
                "Safe coplanar short-edge collapse: {} edge(s); time "
                "{:.3f}s.".format(collapse_count, stage_elapsed),
                flush=True,
            )
        fan_stats = {"fans": 0}
        if planar_fan_minimum_valence is not None:
            optimized_vertices, optimized_faces, fan_stats = (
                retriangulate_planar_fans(
                    optimized_vertices,
                    optimized_faces,
                    optimized_hard_edges,
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
        stage_start = time.perf_counter()
        optimized_faces, flip_count = _flip_quality_edges(
            optimized_vertices,
            optimized_faces,
            optimized_hard_edges,
            passes=flip_passes,
            maximum_dihedral_degrees=coplanar_angle_degrees,
            maximum_edge_length=max_edge_length,
            preferred_edges=np.asarray(
                sorted(coplanar_edges), dtype=np.int64
            ).reshape(-1, 2),
            minimum_candidate_valence=flip_minimum_valence,
            maximum_candidate_quality=flip_maximum_candidate_quality,
        )
        stage_timings.append(("coplanar edge flips", time.perf_counter() - stage_start))
        optimized_edge_faces = _build_edge_faces(optimized_faces)
        optimized_lengths = _edge_length_array(
            optimized_vertices, optimized_edge_faces
        )
        optimized_max_length = (
            float(optimized_lengths.max()) if len(optimized_lengths) else 0.0
        )
        if optimized_max_length > length_limit:
            raise RuntimeError(
                "Post-curve planar remeshing created an edge over the "
                "maximum length: {:.6g} > {:.6g}.".format(
                    optimized_max_length, max_edge_length
                )
            )
        print_edge_length_summary(optimized_lengths)
        print_timing_summary()
        return optimized_vertices, optimized_faces, {
            "hard_edges": len(hard_edges),
            "hard_edges_split": 0,
            "splits": 0,
            "initial_max_length": initial_max_length,
            "final_max_length": optimized_max_length,
            "already_satisfied": True,
            "coplanar_flips": flip_count,
            "short_edge_collapses": collapse_count,
            "reference_feature_edges": reference_feature_edges,
            "reference_vertices": reference_vertices,
            "reference_faces": reference_faces,
            "analytic_cylinder_recovery": cylinder_recovery_stats,
            "brep_model_first": brep_model_stats,
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

    stage_timings.append(
        ("longest-edge refinement", time.perf_counter() - split_start)
    )

    refined_vertices = np.asarray(vertices_list, dtype=np.float64)
    post_split_edges = list(edge_faces)
    final_lengths = _edge_length_array(refined_vertices, post_split_edges)
    remaining_long_edges = [
        (edge, float(length))
        for edge, length in zip(post_split_edges, final_lengths)
        if length > length_limit
    ]
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

    refined_faces = np.asarray(faces_list, dtype=np.int64)
    if len(isolated_transaction_edges):
        post_split_edges = _build_edge_faces(refined_faces)
        missing_lineage = [
            edge for edge in root_for_edge if edge not in post_split_edges
        ]
        if missing_lineage:
            raise RuntimeError(
                "Longest-edge refinement crossed the isolated transaction "
                "halo; first missing edge: {}.".format(missing_lineage[0])
            )
    current_hard_edges = np.asarray(
        sorted(root_for_edge), dtype=np.int64
    ).reshape(-1, 2)
    refined_vertices, refined_faces = (
        retriangulate_planar_after_boundary_refinement(
            refined_vertices,
            refined_faces,
            current_hard_edges,
            max_edge_length,
        )
    )
    if len(isolated_transaction_edges):
        post_planar_edges = _build_edge_faces(refined_faces)
        missing_lineage = [
            edge for edge in root_for_edge if edge not in post_planar_edges
        ]
        if missing_lineage:
            raise RuntimeError(
                "Planar stages crossed the isolated transaction halo; "
                "first missing edge: {}.".format(missing_lineage[0])
            )
    current_hard_edges = np.asarray(
        sorted(root_for_edge), dtype=np.int64
    ).reshape(-1, 2)
    collapse_count = 0
    if short_edge_collapse_passes and short_edge_collapse_ratio > 0.0:
        stage_start = time.perf_counter()
        refined_vertices, refined_faces, collapse_count = (
            collapse_short_coplanar_edges_with_backend(
                refined_vertices,
                refined_faces,
                current_hard_edges,
                maximum_short_edge_length=(
                    max_edge_length * float(short_edge_collapse_ratio)
                ),
                maximum_edge_length=max_edge_length,
                maximum_planar_angle_degrees=coplanar_angle_degrees,
                passes=short_edge_collapse_passes,
                backend=topology_backend,
            )
        )
        stage_elapsed = time.perf_counter() - stage_start
        stage_timings.append(("safe short-edge collapse", stage_elapsed))
        print(
            "Safe coplanar short-edge collapse: {} edge(s); time {:.3f}s."
            .format(collapse_count, stage_elapsed),
            flush=True,
        )
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
    stage_start = time.perf_counter()
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
        maximum_candidate_quality=flip_maximum_candidate_quality,
    )
    stage_timings.append(("coplanar edge flips", time.perf_counter() - stage_start))
    if len(isolated_faces_to_restore):
        refined_faces = np.vstack(
            (refined_faces, isolated_faces_to_restore)
        )
    edge_faces = _build_edge_faces(refined_faces)

    hard_edge_items = list(root_for_edge.items())
    for edge, _ in hard_edge_items:
        if edge not in edge_faces:
            raise RuntimeError(
                "A hard constraint edge disappeared during refinement: {}."
                .format(edge)
            )
    hard_length_sums = np.zeros(len(original_hard_lengths), dtype=np.float64)
    if hard_edge_items:
        hard_child_lengths = _edge_length_array(
            refined_vertices, (edge for edge, _ in hard_edge_items)
        )
        hard_root_indices = np.fromiter(
            (root_index for _, root_index in hard_edge_items),
            dtype=np.int64,
            count=len(hard_edge_items),
        )
        np.add.at(hard_length_sums, hard_root_indices, hard_child_lengths)
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

    final_lengths = _edge_length_array(refined_vertices, edge_faces)
    final_max_length = float(final_lengths.max()) if len(final_lengths) else 0.0
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
    print_edge_length_summary(final_lengths)
    print_timing_summary()
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
            "short_edge_collapses": collapse_count,
            "reference_feature_edges": reference_feature_edges,
            "reference_vertices": reference_vertices,
            "reference_faces": reference_faces,
            "analytic_cylinder_recovery": cylinder_recovery_stats,
            "brep_model_first": brep_model_stats,
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
