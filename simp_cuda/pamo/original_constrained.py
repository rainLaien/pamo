import heapq
import time

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
        delta_y = float(end[1] - start[1])
        if abs(delta_y) <= np.finfo(np.float64).eps:
            continue
        intersection_x = (
            (end[0] - start[0]) * (points[:, 1] - start[1])
            / delta_y
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

    for facet, boundary_edges in zip(mesh.facets, mesh.facets_boundary):
        facet = np.asarray(facet, dtype=np.int64)
        if len(facet) < int(minimum_faces):
            continue
        if minimum_region_area is not None:
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
            if facet_area < float(minimum_region_area):
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
            maximum_result_edge_length is not None
            and len(boundary_lengths)
            and float(boundary_lengths.max())
            > float(maximum_result_edge_length) * (1.0 + 1e-8)
        ):
            boundary_rejection_count += 1
            continue
        old_quality = _triangle_quality_values(vertices, faces[facet])
        if (
            skip_satisfactory_quality
            and float(old_quality.mean()) >= 0.75
            and float(np.percentile(old_quality, 5.0)) >= 0.25
        ):
            continue
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
        strong_mean_gain = (
            bool(accept_strong_mean_gain)
            and float(new_quality.mean()) >= float(old_quality.mean()) + 0.15
            and float(new_quality.min()) >= float(old_quality.min()) - 1e-8
        )
        if (
            (
                np.percentile(new_quality, 5.0)
                < np.percentile(old_quality, 5.0) - 1e-8
                or float(new_quality.mean()) < float(old_quality.mean()) - 1e-8
            )
            and not strong_mean_gain
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
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64), axis=1
    )
    usable = np.asarray(
        [tuple(map(int, edge)) not in protected for edge in adjacency_edges]
    )
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

    for component_id in np.flatnonzero(
        (component_sizes >= int(minimum_faces)) & (component_sizes <= 50000)
    ):
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
        if cycles is None or len(cycles) != 1:
            continue
        candidates += 1
        boundary_set = {tuple(map(int, edge)) for edge in boundary_edges}
        if not isolate_rounded_faces and not boundary_set.issubset(protected):
            continue
        if (set(map(tuple, unique_edges)) & protected) - boundary_set:
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
        spacing = _uniform_target_spacing(spacing, preferred_edge_length)
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
                    grid_points.extend(inside_points[distances >= spacing * 0.55])
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
            "candidates": candidates, "patches": 0, "new_vertices": 0,
            "removed_faces": 0, "new_faces": 0,
            "old_quality": 0.0, "new_quality": 0.0,
            "old_quality_p5": 0.0, "new_quality_p5": 0.0,
        }
    return (
        np.vstack((vertices, np.asarray(appended_vertices, dtype=np.float64))),
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
        row_step = spacing * np.sqrt(3.0) * 0.5
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
            x_values = np.arange(
                spacing * (0.5 if row % 2 == 0 else 1.0),
                circumference,
                spacing,
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
                    grid_points.extend(points[distances >= spacing * 0.55])
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
        np.vstack((vertices, np.asarray(appended_vertices, dtype=np.float64))),
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
        first_basis = first_radial / first_radial_length
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
        spacing = _uniform_target_spacing(spacing, preferred_edge_length)
        if not np.isfinite(spacing) or spacing <= 0.0:
            continue
        row_step = spacing * np.sqrt(3.0) * 0.5
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
            x = spacing * (0.5 if row_index % 2 == 0 else 1.0)
            while x < circumference:
                grid_points.append((x, y))
                x += spacing
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
        np.vstack((vertices, np.asarray(appended_vertices, dtype=np.float64))),
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
    cosine_limit = np.cos(np.deg2rad(float(maximum_planar_angle_degrees)))
    collapse_count = 0

    for _ in range(max(int(passes), 0)):
        edge_faces = _build_edge_faces(faces)
        protected_vertices.update(
            vertex
            for edge, memberships in edge_faces.items()
            if len(memberships) == 1
            for vertex in edge
        )
        vertex_faces = [set() for _ in range(len(vertices))]
        vertex_neighbors = [set() for _ in range(len(vertices))]
        for face_index, face in enumerate(faces):
            first, second, third = map(int, face)
            for vertex in (first, second, third):
                vertex_faces[vertex].add(face_index)
            vertex_neighbors[first].update((second, third))
            vertex_neighbors[second].update((first, third))
            vertex_neighbors[third].update((first, second))

        triangles = vertices[faces]
        crosses = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        cross_lengths = np.linalg.norm(crosses, axis=1)
        valid_faces = cross_lengths > np.finfo(np.float64).eps
        normals = np.zeros_like(crosses)
        normals[valid_faces] = crosses[valid_faces] / cross_lengths[valid_faces, None]
        candidates = []
        for edge, memberships in edge_faces.items():
            if (
                len(memberships) != 2
                or edge[0] in protected_vertices
                or edge[1] in protected_vertices
            ):
                continue
            length = float(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))
            if 0.0 < length < maximum_short_edge_length:
                candidates.append((length, edge))
        candidates.sort()

        reserved_vertices = set()
        replacements = {}
        accepted_this_pass = 0
        for _, edge in candidates:
            first, second = edge
            local_vertices = (
                {first, second}
                | vertex_neighbors[first]
                | vertex_neighbors[second]
            )
            if local_vertices & reserved_vertices:
                continue
            incident_edge_faces = edge_faces[edge]
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

            best = None
            old_quality = _triangle_quality_values(
                vertices, faces[local_face_ids]
            )
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
                local_triangles = vertices[local_faces]
                local_crosses = np.cross(
                    local_triangles[:, 1] - local_triangles[:, 0],
                    local_triangles[:, 2] - local_triangles[:, 0],
                )
                if np.any(local_crosses @ reference_normal <= 1e-14):
                    continue
                local_lengths = np.linalg.norm(
                    local_triangles[:, (1, 2, 0)]
                    - local_triangles[:, (0, 1, 2)],
                    axis=2,
                )
                if float(local_lengths.max()) > maximum_edge_length * (1.0 + 1e-8):
                    continue
                new_quality = _triangle_quality_values(vertices, local_faces)
                if (
                    float(new_quality.min()) < float(old_quality.min()) - 1e-8
                    or float(new_quality.mean()) < float(old_quality.mean()) - 1e-8
                ):
                    continue
                score = (float(new_quality.min()), float(new_quality.mean()))
                if best is None or score > best[0]:
                    best = (score, removed, kept)
            if best is None:
                continue
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

    return vertices.copy(), faces, collapse_count


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
    short_edge_collapse_ratio=0.25,
    short_edge_collapse_passes=0,
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

    if cylinder_minimum_faces is not None:
        stage_start = time.perf_counter()
        vertices, faces, cylinder_stats = retriangulate_cylindrical_walls(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=cylinder_minimum_faces,
            radius_tolerance=cylinder_radius_tolerance,
            target_edge_ratio=cylinder_target_edge_ratio,
            preferred_edge_length=preferred_generated_edge_length,
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

    if rounded_fillet_minimum_faces is not None:
        stage_start = time.perf_counter()
        vertices, faces, fillet_stats = retriangulate_partial_cylindrical_walls(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=rounded_fillet_minimum_faces,
            # Fillets created by blends and boolean joins are often only
            # approximately cylindrical. Projection is restricted to the
            # source band and the result still has to improve both mean and
            # lower-tail quality, so these relaxed fit gates remain safe.
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
                    # Solid planes do not have an inner contour to seed a
                    # graded transition. Keep their conservative growth rule
                    # to avoid long triangles along dense outer boundaries.
                    gradual_target_spacing=True,
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
        collapse_count = 0
        if short_edge_collapse_passes and short_edge_collapse_ratio > 0.0:
            stage_start = time.perf_counter()
            optimized_vertices, optimized_faces, collapse_count = (
                collapse_short_coplanar_edges(
                    optimized_vertices,
                    optimized_faces,
                    hard_edges_array,
                    maximum_short_edge_length=(
                        max_edge_length * float(short_edge_collapse_ratio)
                    ),
                    maximum_edge_length=max_edge_length,
                    maximum_planar_angle_degrees=coplanar_angle_degrees,
                    passes=short_edge_collapse_passes,
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
        stage_start = time.perf_counter()
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
            maximum_candidate_quality=flip_maximum_candidate_quality,
        )
        stage_timings.append(("coplanar edge flips", time.perf_counter() - stage_start))
        optimized_edge_faces = _build_edge_faces(optimized_faces)
        optimized_lengths = np.asarray(
            [
                float(
                    np.linalg.norm(
                        optimized_vertices[edge[0]]
                        - optimized_vertices[edge[1]]
                    )
                )
                for edge in optimized_edge_faces
            ],
            dtype=np.float64,
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
    collapse_count = 0
    if short_edge_collapse_passes and short_edge_collapse_ratio > 0.0:
        stage_start = time.perf_counter()
        refined_vertices, refined_faces, collapse_count = (
            collapse_short_coplanar_edges(
                refined_vertices,
                refined_faces,
                current_hard_edges,
                maximum_short_edge_length=(
                    max_edge_length * float(short_edge_collapse_ratio)
                ),
                maximum_edge_length=max_edge_length,
                maximum_planar_angle_degrees=coplanar_angle_degrees,
                passes=short_edge_collapse_passes,
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
