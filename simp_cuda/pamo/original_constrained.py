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


def _triangulate_simple_polygon_with_points(points, boundary_count):
    """Ear-clip a simple polygon, then insert strictly interior points."""
    points = np.asarray(points, dtype=np.float64)
    boundary_count = int(boundary_count)
    if boundary_count < 3 or len(points) < boundary_count:
        return None
    scale = max(float(np.ptp(points[:boundary_count], axis=0).max()), 1.0)
    tolerance = scale * scale * 1e-14

    def orientation(first, second, third):
        first = points[int(first)]
        second = points[int(second)]
        third = points[int(third)]
        return float(
            (second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0])
        )

    polygon = list(range(boundary_count))
    if _polygon_signed_area(points[:boundary_count]) < 0.0:
        polygon.reverse()
    triangles = []
    maximum_iterations = boundary_count * boundary_count
    iterations = 0
    while len(polygon) > 3 and iterations < maximum_iterations:
        ear_found = False
        for index in range(len(polygon)):
            previous = polygon[index - 1]
            current = polygon[index]
            following = polygon[(index + 1) % len(polygon)]
            area = orientation(previous, current, following)
            if area <= tolerance:
                continue
            first = points[previous]
            second = points[current]
            third = points[following]
            other_indices = [
                vertex
                for vertex in polygon
                if vertex not in (previous, current, following)
            ]
            contains_vertex = False
            if other_indices:
                others = points[other_indices]
                first_side = np.cross(second - first, others - first)
                second_side = np.cross(third - second, others - second)
                third_side = np.cross(first - third, others - third)
                contains_vertex = bool(
                    np.any(
                        (first_side >= -tolerance)
                        & (second_side >= -tolerance)
                        & (third_side >= -tolerance)
                    )
                )
            if contains_vertex:
                continue
            triangles.append((previous, current, following))
            del polygon[index]
            ear_found = True
            break
        if not ear_found:
            return None
        iterations += 1
    if len(polygon) != 3 or orientation(*polygon) <= tolerance:
        return None
    triangles.append(tuple(polygon))

    for point_index in range(boundary_count, len(points)):
        point = points[point_index]
        containing_triangle = None
        for triangle_index, triangle in enumerate(triangles):
            first, second, third = map(int, triangle)
            first_side = float(
                np.cross(points[second] - points[first], point - points[first])
            )
            second_side = float(
                np.cross(points[third] - points[second], point - points[second])
            )
            third_side = float(
                np.cross(points[first] - points[third], point - points[third])
            )
            if min(first_side, second_side, third_side) > tolerance:
                containing_triangle = triangle_index
                break
        if containing_triangle is None:
            continue
        first, second, third = triangles[containing_triangle]
        triangles[containing_triangle] = (first, second, point_index)
        triangles.append((second, third, point_index))
        triangles.append((third, first, point_index))
    return np.asarray(triangles, dtype=np.int64)


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


def classify_planar_face_regions(
    vertices,
    faces,
    protected_edges=None,
    minimum_faces=20,
    maximum_normal_angle_degrees=0.5,
    maximum_plane_distance_ratio=1e-6,
):
    """Classify connected planar patches without chaining across curvature.

    Each region grows against one fixed seed plane.  A sequence of slightly
    rotated faces on a fillet or general curved surface therefore cannot drift
    into a false planar patch.  Protected feature edges are never crossed.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    minimum_faces = int(minimum_faces)
    maximum_normal_angle_degrees = float(maximum_normal_angle_degrees)
    maximum_plane_distance_ratio = float(maximum_plane_distance_ratio)
    if minimum_faces < 1:
        raise ValueError("Planar classification minimum faces must be positive.")
    if not 0.0 <= maximum_normal_angle_degrees < 90.0:
        raise ValueError("Planar classification angle must be in [0, 90).")
    if maximum_plane_distance_ratio < 0.0:
        raise ValueError("Planar classification distance ratio must be non-negative.")
    if len(faces) == 0:
        return [], {
            "regions": 0,
            "planar_faces": 0,
            "remaining_faces": 0,
        }

    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_areas = np.linalg.norm(crosses, axis=1)
    valid_faces = double_areas > np.finfo(np.float64).eps
    normals = np.zeros_like(crosses)
    normals[valid_faces] = (
        crosses[valid_faces] / double_areas[valid_faces, None]
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64),
        axis=1,
    )
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(
            protected_edges
            if protected_edges is not None
            else np.empty((0, 2), dtype=np.int64),
            dtype=np.int64,
        ).reshape(-1, 2)
    }
    usable = np.asarray(
        [tuple(map(int, edge)) not in protected for edge in adjacency_edges],
        dtype=bool,
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

    cosine_limit = np.cos(np.deg2rad(maximum_normal_angle_degrees))
    diameter = max(
        float(np.linalg.norm(np.ptp(vertices, axis=0))),
        np.finfo(np.float64).eps,
    )
    distance_limit = max(
        diameter * maximum_plane_distance_ratio,
        diameter * 1e-12,
    )
    close_support = np.zeros(len(faces), dtype=np.int64)
    if len(adjacency):
        close = (
            np.einsum(
                "ij,ij->i",
                normals[adjacency[:, 0]],
                normals[adjacency[:, 1]],
            )
            >= cosine_limit
        ) & usable
        np.add.at(close_support, adjacency[close, 0], 1)
        np.add.at(close_support, adjacency[close, 1], 1)

    # Interior planar faces normally have the strongest coplanar support, so
    # let them seed regions before transition and fillet faces.
    seed_order = np.lexsort((-double_areas, -close_support))
    assigned = ~valid_faces.copy()
    regions = []
    for seed in seed_order:
        seed = int(seed)
        if assigned[seed]:
            continue
        seed_normal = normals[seed]
        seed_origin = triangles[seed, 0]
        assigned[seed] = True
        stack = [seed]
        region = []
        while stack:
            face_index = stack.pop()
            region.append(face_index)
            start = graph.indptr[face_index]
            end = graph.indptr[face_index + 1]
            for neighbor in graph.indices[start:end]:
                neighbor = int(neighbor)
                if assigned[neighbor]:
                    continue
                if float(np.dot(normals[neighbor], seed_normal)) < cosine_limit:
                    continue
                distances = np.abs(
                    (triangles[neighbor] - seed_origin) @ seed_normal
                )
                if float(distances.max()) > distance_limit:
                    continue
                assigned[neighbor] = True
                stack.append(neighbor)
        if len(region) >= minimum_faces:
            regions.append(np.asarray(sorted(region), dtype=np.int64))

    planar_face_count = int(sum(len(region) for region in regions))
    return regions, {
        "regions": len(regions),
        "planar_faces": planar_face_count,
        "remaining_faces": int(len(faces) - planar_face_count),
        "normal_angle_degrees": maximum_normal_angle_degrees,
        "distance_limit": distance_limit,
    }


def select_largest_opposed_planar_regions(
    vertices,
    faces,
    planar_regions,
    maximum_opposition_angle_degrees=5.0,
):
    """Select the largest planar patch and its largest opposite-facing mate.

    This isolates the two primary skins of a plate without assuming that the
    plate normal is aligned with a world axis.  Region area is measured from
    its triangles and the second patch must have the opposite oriented normal
    within ``maximum_opposition_angle_degrees``.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    maximum_opposition_angle_degrees = float(
        maximum_opposition_angle_degrees
    )
    if not 0.0 <= maximum_opposition_angle_degrees < 90.0:
        raise ValueError("Planar opposition angle must be in [0, 90).")
    if len(planar_regions) < 2:
        raise ValueError(
            "Largest opposed planar-pair selection needs at least two regions."
        )

    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    face_areas = 0.5 * np.linalg.norm(crosses, axis=1)
    region_areas = []
    region_normals = []
    normalized_regions = []
    for region in planar_regions:
        region = np.asarray(region, dtype=np.int64)
        normalized_regions.append(region)
        region_areas.append(float(face_areas[region].sum()))
        normal = crosses[region].sum(axis=0)
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= np.finfo(np.float64).eps:
            region_normals.append(np.zeros(3, dtype=np.float64))
        else:
            region_normals.append(normal / normal_length)

    region_areas = np.asarray(region_areas, dtype=np.float64)
    region_normals = np.asarray(region_normals, dtype=np.float64)
    primary_index = int(np.argmax(region_areas))
    opposition_limit = -np.cos(
        np.deg2rad(maximum_opposition_angle_degrees)
    )
    dot_products = region_normals @ region_normals[primary_index]
    candidates = np.flatnonzero(dot_products <= opposition_limit)
    candidates = candidates[candidates != primary_index]
    if len(candidates) == 0:
        raise ValueError(
            "The largest planar region has no opposite-facing planar mate "
            "within {:.6g} degrees.".format(
                maximum_opposition_angle_degrees
            )
        )
    opposite_index = int(candidates[np.argmax(region_areas[candidates])])
    return (
        [
            normalized_regions[primary_index],
            normalized_regions[opposite_index],
        ],
        {
            "primary_index": primary_index,
            "opposite_index": opposite_index,
            "primary_faces": int(len(normalized_regions[primary_index])),
            "opposite_faces": int(len(normalized_regions[opposite_index])),
            "primary_area": float(region_areas[primary_index]),
            "opposite_area": float(region_areas[opposite_index]),
            "primary_normal": region_normals[primary_index].copy(),
            "opposite_normal": region_normals[opposite_index].copy(),
            "normal_dot": float(dot_products[opposite_index]),
        },
    )


def classify_nonplanar_face_regions(
    vertices,
    faces,
    planar_regions,
    protected_edges=None,
    minimum_faces=4,
    axial_normal_tolerance=2e-2,
    radius_tolerance=1e-2,
    full_cylinder_coverage_degrees=300.0,
    maximum_region_dihedral_degrees=60.0,
):
    """Separate non-planar faces into cylinders, fillets and extrusions.

    Cylinder and extrusion normals share a common perpendicular axis.  A
    least-squares circle fit then separates constant-radius regions from
    general extruded/developable regions.  Constant-radius regions with broad
    angular coverage are labelled cylinders; narrower regions are labelled
    fillets.  Faces that do not satisfy the axial model remain general curved
    faces for a later surface-parameterization stage.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    minimum_faces = int(minimum_faces)
    if minimum_faces < 2:
        raise ValueError("Non-planar classification minimum faces must be >= 2.")
    if not 0.0 < float(axial_normal_tolerance) < 1.0:
        raise ValueError("Axial normal tolerance must be in (0, 1).")
    if float(radius_tolerance) <= 0.0:
        raise ValueError("Radius tolerance must be positive.")
    if not 0.0 < float(full_cylinder_coverage_degrees) <= 360.0:
        raise ValueError("Full-cylinder coverage must be in (0, 360].")
    if not 0.0 <= float(maximum_region_dihedral_degrees) < 180.0:
        raise ValueError("Region dihedral limit must be in [0, 180).")

    planar_mask = np.zeros(len(faces), dtype=bool)
    for region in planar_regions:
        planar_mask[np.asarray(region, dtype=np.int64)] = True
    active_mask = ~planar_mask
    if not np.any(active_mask):
        return {
            "cylinders": [],
            "fillets": [],
            "extrusions": [],
            "general_curved_faces": np.empty(0, dtype=np.int64),
        }, {
            "cylinder_regions": 0,
            "cylinder_faces": 0,
            "fillet_regions": 0,
            "fillet_faces": 0,
            "extrusion_regions": 0,
            "extrusion_faces": 0,
            "general_curved_faces": 0,
        }

    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_areas = np.linalg.norm(crosses, axis=1)
    valid = double_areas > np.finfo(np.float64).eps
    normals = np.zeros_like(crosses)
    normals[valid] = crosses[valid] / double_areas[valid, None]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.sort(
        np.asarray(mesh.face_adjacency_edges, dtype=np.int64), axis=1
    )
    adjacency_angles = np.degrees(
        np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    )
    protected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(
            protected_edges
            if protected_edges is not None
            else np.empty((0, 2), dtype=np.int64),
            dtype=np.int64,
        ).reshape(-1, 2)
    }

    # Extremely skinny triangles on an extruded surface carry a reliable
    # local axis in their longest edge even when old feature thresholds have
    # fragmented the surface into tiny components.  Grow those seeds by the
    # axial-normal model before attempting broader component fits.
    face_edge_vectors = (
        triangles[:, (1, 2, 0)] - triangles[:, (0, 1, 2)]
    )
    face_edge_lengths = np.linalg.norm(face_edge_vectors, axis=2)
    shortest_lengths = np.maximum(
        face_edge_lengths.min(axis=1), np.finfo(np.float64).eps
    )
    elongation = face_edge_lengths.max(axis=1) / shortest_lengths
    qualities = _triangle_quality_values(vertices, faces)
    sliver_seeds = np.flatnonzero(
        active_mask & valid & (qualities < 0.1) & (elongation >= 20.0)
    )
    if len(sliver_seeds):
        sliver_seeds = sliver_seeds[
            np.argsort(elongation[sliver_seeds])[::-1]
        ]
    face_neighbors = [[] for _ in range(len(faces))]
    for index, (first, second) in enumerate(adjacency):
        angle = float(adjacency_angles[index])
        face_neighbors[int(first)].append((int(second), angle))
        face_neighbors[int(second)].append((int(first), angle))

    sliver_claimed = np.zeros(len(faces), dtype=bool)
    extrusion_regions = []
    axial_growth_tolerance = max(float(axial_normal_tolerance), 5e-2)
    region_angle_limit = float(maximum_region_dihedral_degrees)
    for seed in sliver_seeds:
        seed = int(seed)
        if sliver_claimed[seed]:
            continue
        longest_edge_index = int(np.argmax(face_edge_lengths[seed]))
        axis = face_edge_vectors[seed, longest_edge_index]
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= np.finfo(np.float64).eps:
            continue
        axis /= axis_length
        visited = {seed}
        stack = [seed]
        while stack:
            face_index = stack.pop()
            for neighbor, angle in face_neighbors[face_index]:
                if (
                    neighbor in visited
                    or sliver_claimed[neighbor]
                    or not active_mask[neighbor]
                    or not valid[neighbor]
                    or angle > region_angle_limit
                    or abs(float(np.dot(normals[neighbor], axis)))
                    > axial_growth_tolerance
                ):
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        if len(visited) < 2:
            continue
        face_ids = np.asarray(sorted(visited), dtype=np.int64)
        sliver_claimed[face_ids] = True
        extrusion_regions.append(
            {
                "faces": face_ids,
                "axis": axis.copy(),
                "axial_normal_error": float(
                    np.sqrt(np.mean((normals[face_ids] @ axis) ** 2))
                ),
                "seeded_by_sliver": True,
            }
        )

    # Seed order must not create artificial region boundaries.  Merge
    # neighboring sliver-grown regions when their fitted axes agree and the
    # shared edge is smooth; this turns the old long seam into an internal edge
    # that the boundary-only reconstruction is allowed to discard.
    if len(extrusion_regions) > 1:
        face_region_ids = np.full(len(faces), -1, dtype=np.int64)
        for region_id, region in enumerate(extrusion_regions):
            face_region_ids[region["faces"]] = region_id
        parents = np.arange(len(extrusion_regions), dtype=np.int64)

        def find_root(region_id):
            region_id = int(region_id)
            while parents[region_id] != region_id:
                parents[region_id] = parents[int(parents[region_id])]
                region_id = int(parents[region_id])
            return region_id

        def union_regions(first_region, second_region):
            first_root = find_root(first_region)
            second_root = find_root(second_region)
            if first_root != second_root:
                parents[second_root] = first_root

        minimum_axis_alignment = np.cos(np.deg2rad(5.0))
        maximum_merge_dihedral = min(region_angle_limit, 10.0)
        for adjacency_index, (first_face, second_face) in enumerate(adjacency):
            first_region = int(face_region_ids[int(first_face)])
            second_region = int(face_region_ids[int(second_face)])
            if (
                first_region < 0
                or second_region < 0
                or first_region == second_region
                or float(adjacency_angles[adjacency_index])
                > maximum_merge_dihedral
            ):
                continue
            first_axis = extrusion_regions[first_region]["axis"]
            second_axis = extrusion_regions[second_region]["axis"]
            if abs(float(np.dot(first_axis, second_axis))) >= minimum_axis_alignment:
                union_regions(first_region, second_region)

        grouped_regions = {}
        for region_id, region in enumerate(extrusion_regions):
            grouped_regions.setdefault(find_root(region_id), []).append(region)
        merged_extrusion_regions = []
        for grouped in grouped_regions.values():
            if len(grouped) == 1:
                merged_extrusion_regions.append(grouped[0])
                continue
            face_ids = np.unique(
                np.concatenate([region["faces"] for region in grouped])
            )
            grouped_normals = normals[face_ids]
            covariance = grouped_normals.T @ grouped_normals / len(grouped_normals)
            _, eigenvectors = np.linalg.eigh(covariance)
            axis = eigenvectors[:, 0]
            axial_error = float(
                np.sqrt(np.mean((grouped_normals @ axis) ** 2))
            )
            if axial_error > axial_growth_tolerance:
                merged_extrusion_regions.extend(grouped)
                continue
            merged_extrusion_regions.append(
                {
                    "faces": face_ids,
                    "axis": axis,
                    "axial_normal_error": axial_error,
                    "seeded_by_sliver": True,
                }
            )
        extrusion_regions = merged_extrusion_regions

    remaining_active_mask = active_mask & ~sliver_claimed
    usable = np.asarray(
        [
            remaining_active_mask[first]
            and remaining_active_mask[second]
            and (
                tuple(map(int, edge)) not in protected
                or adjacency_angles[index]
                <= float(maximum_region_dihedral_degrees)
            )
            for index, ((first, second), edge) in enumerate(
                zip(adjacency, adjacency_edges)
            )
        ],
        dtype=bool,
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
    active_labels, active_counts = np.unique(
        labels[remaining_active_mask], return_counts=True
    )

    classified_mask = planar_mask | sliver_claimed
    cylinder_regions = []
    fillet_regions = []
    coverage_limit = np.deg2rad(float(full_cylinder_coverage_degrees))

    for component_id, component_size in zip(active_labels, active_counts):
        if int(component_size) < minimum_faces:
            continue
        face_ids = np.flatnonzero(
            remaining_active_mask & (labels == component_id)
        )
        if not np.all(valid[face_ids]):
            continue
        component_normals = normals[face_ids]
        normal_covariance = (
            component_normals.T @ component_normals / len(component_normals)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(normal_covariance)
        # A single planar direction leaves two null eigenvalues and must not
        # be mistaken for an axial surface merely because an axis can be fit.
        if float(eigenvalues[1]) <= 1e-8:
            continue
        axis = eigenvectors[:, 0]
        axial_error = float(
            np.sqrt(np.mean((component_normals @ axis) ** 2))
        )
        if axial_error > float(axial_normal_tolerance):
            continue

        first_basis = component_normals[0] - axis * np.dot(
            component_normals[0], axis
        )
        first_basis_length = float(np.linalg.norm(first_basis))
        if first_basis_length <= np.finfo(np.float64).eps:
            continue
        first_basis /= first_basis_length
        second_basis = np.cross(axis, first_basis)
        component_vertex_ids = np.unique(faces[face_ids])
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
        radii = np.linalg.norm(projected - center_2d, axis=1)
        radius = float(radii.mean())
        radius_error = (
            float(radii.std() / radius)
            if radius > np.finfo(np.float64).eps
            else np.inf
        )

        region = {
            "faces": face_ids,
            "axis": axis,
            "axial_normal_error": axial_error,
        }
        if radius_error <= float(radius_tolerance):
            axis_origin = origin + center_2d @ np.vstack(
                (first_basis, second_basis)
            )
            centroid_offsets = triangles[face_ids].mean(axis=1) - axis_origin
            centroid_axial = centroid_offsets @ axis
            centroid_radial = (
                centroid_offsets - centroid_axial[:, None] * axis
            )
            radial_lengths = np.linalg.norm(centroid_radial, axis=1)
            radial_alignment = np.zeros(len(face_ids), dtype=np.float64)
            nonzero_radial = radial_lengths > np.finfo(np.float64).eps
            radial_alignment[nonzero_radial] = np.abs(
                np.sum(
                    component_normals[nonzero_radial]
                    * centroid_radial[nonzero_radial],
                    axis=1,
                )
                / radial_lengths[nonzero_radial]
            )
            if (
                np.all(nonzero_radial)
                and float(np.percentile(radial_alignment, 5.0)) >= 0.98
            ):
                angles = np.mod(
                    np.arctan2(
                        offsets @ second_basis - center_2d[1],
                        offsets @ first_basis - center_2d[0],
                    ),
                    2.0 * np.pi,
                )
                sorted_angles = np.sort(angles)
                angular_gaps = np.diff(
                    np.concatenate(
                        (sorted_angles, sorted_angles[:1] + 2.0 * np.pi)
                    )
                )
                coverage = 2.0 * np.pi - float(angular_gaps.max())
                region.update(
                    {
                        "radius": radius,
                        "radius_error": radius_error,
                        "coverage_radians": coverage,
                    }
                )
                if coverage >= coverage_limit:
                    cylinder_regions.append(region)
                else:
                    fillet_regions.append(region)
                classified_mask[face_ids] = True
                continue

        extrusion_regions.append(region)
        classified_mask[face_ids] = True

    general_curved_faces = np.flatnonzero(~classified_mask)
    regions = {
        "cylinders": cylinder_regions,
        "fillets": fillet_regions,
        "extrusions": extrusion_regions,
        "general_curved_faces": general_curved_faces,
    }
    stats = {
        "cylinder_regions": len(cylinder_regions),
        "cylinder_faces": int(
            sum(len(region["faces"]) for region in cylinder_regions)
        ),
        "fillet_regions": len(fillet_regions),
        "fillet_faces": int(
            sum(len(region["faces"]) for region in fillet_regions)
        ),
        "extrusion_regions": len(extrusion_regions),
        "extrusion_faces": int(
            sum(len(region["faces"]) for region in extrusion_regions)
        ),
        "general_curved_faces": int(len(general_curved_faces)),
    }
    return regions, stats


def retriangulate_planar_annuli(
    vertices,
    faces,
    protected_edges=None,
    minimum_faces=20,
    minimum_holes=1,
    maximum_holes=None,
    maximum_target_edge_length=None,
    face_regions=None,
    solid_minimum_faces=None,
    holed_minimum_faces=None,
    minimum_triangle_angle_degrees=None,
    announce_fallback=True,
    enforce_quality_gate=True,
    fallback_flip_passes=5,
    allow_ear_clipping_fallback=False,
):
    """Rebuild planar patches from boundaries, discarding all interior edges."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    fallback_flip_passes = int(fallback_flip_passes)
    if fallback_flip_passes < 0:
        raise ValueError("Fallback flip passes must be non-negative.")
    if minimum_triangle_angle_degrees is not None:
        minimum_triangle_angle_degrees = float(
            minimum_triangle_angle_degrees
        )
        if not 0.0 < minimum_triangle_angle_degrees < 34.0:
            raise ValueError(
                "Planar minimum triangle angle must be in (0, 34) degrees."
            )
        if constrained_triangle is None and announce_fallback:
            print(
                "Planar minimum-angle package unavailable; using the SciPy "
                "boundary-layer constrained-Delaunay fallback.",
                flush=True,
            )
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

    if face_regions is None:
        region_boundaries = zip(mesh.facets, mesh.facets_boundary)
    else:
        region_boundaries = []
        for region in face_regions:
            region = np.asarray(region, dtype=np.int64)
            region_faces = faces[region]
            region_edges = np.sort(
                region_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
                axis=1,
            )
            unique_edges, edge_counts = np.unique(
                region_edges,
                axis=0,
                return_counts=True,
            )
            region_boundaries.append(
                (region, unique_edges[edge_counts == 1])
            )

    for facet, boundary_edges in region_boundaries:
        facet = np.asarray(facet, dtype=np.int64)
        if len(facet) < int(minimum_faces):
            continue
        cycles = _ordered_cycles_from_edges(boundary_edges)
        if cycles is None:
            continue
        hole_count = len(cycles) - 1
        if (
            hole_count == 0
            and solid_minimum_faces is not None
            and len(facet) < int(solid_minimum_faces)
        ):
            continue
        if (
            hole_count > 0
            and holed_minimum_faces is not None
            and len(facet) < int(holed_minimum_faces)
        ):
            continue
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
        if hole_count == 0:
            domain_area = max(float(areas[outer_index]), 0.0)
            maximum_interior_points = max(len(facet) * 8, 100)
            density_spacing = np.sqrt(
                domain_area
                / max(
                    (np.sqrt(3.0) * 0.5) * maximum_interior_points,
                    np.finfo(np.float64).eps,
                )
            )
            spacing = max(spacing, density_spacing)
        if maximum_target_edge_length is not None:
            spacing = min(spacing, float(maximum_target_edge_length))

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
        if minimum_triangle_angle_degrees is not None:
            # Build graded Steiner layers from each boundary into the planar
            # domain.  Level 0 places an approximately equilateral apex over
            # every boundary segment; subsequent levels double tangential and
            # normal spacing until they meet the coarse interior grid.
            boundary_layer_points = []
            for polygon in polygons:
                polygon = np.asarray(polygon, dtype=np.float64)
                polygon_edges = np.roll(polygon, -1, axis=0) - polygon
                positive_lengths = np.linalg.norm(polygon_edges, axis=1)
                positive_lengths = positive_lengths[positive_lengths > 0.0]
                if len(positive_lengths) == 0:
                    continue
                local_spacing = float(np.median(positive_lengths))
                if spacing <= local_spacing * 1.5:
                    continue
                level_count = min(
                    10,
                    max(
                        1,
                        int(
                            np.ceil(
                                np.log2(
                                    max(spacing / local_spacing, 1.0)
                                )
                            )
                        )
                        + 1,
                    ),
                )
                polygon_size = len(polygon)
                for level in range(level_count):
                    stride = min(1 << level, max(polygon_size - 1, 1))
                    for start_index in range(0, polygon_size, stride):
                        end_index = (start_index + stride) % polygon_size
                        start_point = polygon[start_index]
                        end_point = polygon[end_index]
                        chord = end_point - start_point
                        chord_length = float(np.linalg.norm(chord))
                        if chord_length <= 0.0:
                            continue
                        normal_2d = np.asarray(
                            (-chord[1], chord[0]), dtype=np.float64
                        ) / chord_length
                        height = min(
                            0.5 * np.sqrt(3.0) * chord_length,
                            row_step,
                        )
                        midpoint = (start_point + end_point) * 0.5
                        boundary_layer_points.extend(
                            (
                                midpoint + normal_2d * height,
                                midpoint - normal_2d * height,
                            )
                        )
            if boundary_layer_points:
                boundary_layer_points = np.asarray(
                    boundary_layer_points,
                    dtype=np.float64,
                )
                layer_inside = _points_in_polygon(
                    boundary_layer_points,
                    outer,
                )
                for hole in holes:
                    layer_inside &= ~_points_in_polygon(
                        boundary_layer_points,
                        hole,
                    )
                grid_points.extend(boundary_layer_points[layer_inside])
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
                    boundary_clearance = (
                        spacing * 0.25
                        if minimum_triangle_angle_degrees is not None
                        else spacing * 0.55
                    )
                    grid_points.extend(
                        inside_points[minimum_distances >= boundary_clearance]
                    )
            row += 1
            y += row_step

        interior_2d = np.asarray(grid_points, dtype=np.float64).reshape(-1, 2)
        if len(interior_2d):
            point_scale = max(
                float(np.ptp(all_boundary_2d, axis=0).max()),
                1.0,
            )
            quantization = point_scale * 1e-10
            keys = np.rint(interior_2d / quantization).astype(np.int64)
            _, unique_indices = np.unique(keys, axis=0, return_index=True)
            interior_2d = interior_2d[np.sort(unique_indices)]
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
                    [hole.mean(axis=0) for hole in holes], dtype=np.float64
                )
            triangle_options = "pQY"
            if minimum_triangle_angle_degrees is not None:
                triangle_options = "pq{:.12g}QY".format(
                    minimum_triangle_angle_degrees
                )
            triangle_result = constrained_triangle.triangulate(
                triangle_input,
                triangle_options,
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
            triangulation = None
            constraints_recovered = False
            for qhull_options in (None, "QJ"):
                try:
                    candidate = Delaunay(
                        all_2d,
                        qhull_options=qhull_options,
                    ).simplices.astype(np.int64)
                except QhullError:
                    continue
                candidate, constraints_recovered = (
                    _recover_planar_constraint_edges(
                        all_2d,
                        candidate,
                        required_edges,
                    )
                )
                if constraints_recovered:
                    triangulation = candidate
                    break
            if (
                not constraints_recovered
                and not holes
                and allow_ear_clipping_fallback
            ):
                candidate = _triangulate_simple_polygon_with_points(
                    all_2d,
                    len(all_boundary_2d),
                )
                if candidate is not None:
                    embedded_points = np.column_stack(
                        (
                            all_2d,
                            np.zeros(len(all_2d), dtype=np.float64),
                        )
                    )
                    candidate, _ = _flip_quality_edges(
                        embedded_points,
                        candidate,
                        np.asarray(sorted(required_edges), dtype=np.int64),
                        passes=fallback_flip_passes,
                        maximum_dihedral_degrees=0.1,
                    )
                    triangulation = candidate
                    constraints_recovered = True
            if not constraints_recovered:
                boundary_rejection_count += 1
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
        if enforce_quality_gate and (
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
        np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
            )
        ),
        np.vstack(
            (
                faces[kept_faces],
                np.asarray(appended_faces, dtype=np.int64).reshape(-1, 3),
            )
        ),
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


def _map_uv_points_to_surface(points, uv_vertices, uv_faces, vertices):
    """Map UV points to a source triangle surface by barycentric coordinates."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    uv_vertices = np.asarray(uv_vertices, dtype=np.float64)
    uv_faces = np.asarray(uv_faces, dtype=np.int64)
    embedded_uv = np.column_stack(
        (uv_vertices, np.zeros(len(uv_vertices), dtype=np.float64))
    )
    embedded_points = np.column_stack(
        (points, np.zeros(len(points), dtype=np.float64))
    )
    _, closest_faces, closest_points = igl.point_mesh_squared_distance(
        embedded_points,
        embedded_uv,
        uv_faces,
    )
    closest_faces = np.asarray(closest_faces, dtype=np.int64)
    closest_points = np.asarray(closest_points, dtype=np.float64)[:, :2]
    parameter_triangles = uv_vertices[uv_faces[closest_faces]]
    first = parameter_triangles[:, 0]
    second = parameter_triangles[:, 1]
    third = parameter_triangles[:, 2]
    denominator = (
        (second[:, 1] - third[:, 1]) * (first[:, 0] - third[:, 0])
        + (third[:, 0] - second[:, 0]) * (first[:, 1] - third[:, 1])
    )
    if np.any(np.abs(denominator) <= np.finfo(np.float64).eps):
        raise ValueError("Degenerate UV triangle encountered during mapping.")
    first_weight = (
        (second[:, 1] - third[:, 1])
        * (closest_points[:, 0] - third[:, 0])
        + (third[:, 0] - second[:, 0])
        * (closest_points[:, 1] - third[:, 1])
    ) / denominator
    second_weight = (
        (third[:, 1] - first[:, 1])
        * (closest_points[:, 0] - third[:, 0])
        + (first[:, 0] - third[:, 0])
        * (closest_points[:, 1] - third[:, 1])
    ) / denominator
    third_weight = 1.0 - first_weight - second_weight
    weights = np.column_stack(
        (first_weight, second_weight, third_weight)
    )
    source_triangles = np.asarray(vertices, dtype=np.float64)[
        uv_faces[closest_faces]
    ]
    return np.einsum("ij,ijk->ik", weights, source_triangles)


def retriangulate_developable_regions(
    vertices,
    faces,
    regions,
    minimum_faces=20,
    maximum_target_edge_length=None,
    minimum_triangle_angle_degrees=28.0,
    maximum_input_quality=0.05,
):
    """Unwrap classified developable regions and rebuild from boundaries.

    LSCM is used only as a two-dimensional chart.  The old internal edges are
    discarded in that chart, and all inserted vertices are mapped back to the
    original region by barycentric interpolation.  Region boundary vertices
    and boundary edges remain unchanged, so adjacent unprocessed surfaces stay
    conforming.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    minimum_faces = int(minimum_faces)
    if minimum_faces < 2:
        raise ValueError("Developable reconstruction minimum faces must be >= 2.")
    if maximum_target_edge_length is None:
        maximum_target_edge_length = (
            float(np.linalg.norm(np.ptp(vertices, axis=0))) * 0.05
        )
    maximum_target_edge_length = float(maximum_target_edge_length)
    if maximum_target_edge_length <= 0.0:
        raise ValueError("Developable target edge length must be positive.")
    maximum_input_quality = float(maximum_input_quality)
    if not 0.0 < maximum_input_quality <= 1.0:
        raise ValueError("Developable input quality threshold must be in (0, 1].")

    kept_faces = np.ones(len(faces), dtype=bool)
    appended_vertices = []
    appended_faces = []
    candidates = 0
    accepted = 0
    parameter_rejections = 0
    boundary_rejections = 0
    chart_rejections = 0
    chart_boundary_rejections = 0
    chart_constraint_rejections = 0
    chart_quality_rejections = 0
    quality_rejections = 0
    removed_faces = 0
    screened_out = 0
    old_qualities = []
    new_qualities = []
    last_quality_rejection = None

    for region in regions:
        face_ids = np.asarray(region["faces"], dtype=np.int64)
        if len(face_ids) < minimum_faces or not np.all(kept_faces[face_ids]):
            continue
        component_faces = faces[face_ids]
        old_quality = _triangle_quality_values(vertices, component_faces)
        component_triangles = vertices[component_faces]
        component_edge_lengths = np.linalg.norm(
            component_triangles[:, (1, 2, 0)]
            - component_triangles[:, (0, 1, 2)],
            axis=2,
        )
        if (
            float(np.percentile(old_quality, 5.0))
            >= maximum_input_quality
            or float(component_edge_lengths.max())
            <= maximum_target_edge_length * (1.0 + 1e-8)
        ):
            screened_out += 1
            continue
        candidates += 1
        if candidates == 1 or candidates % 25 == 0:
            print(
                "  developable reconstruction progress: {} candidates, {} "
                "accepted, {} parameter/boundary/chart/quality rejected."
                .format(
                    candidates,
                    accepted,
                    parameter_rejections
                    + boundary_rejections
                    + chart_rejections
                    + quality_rejections,
                ),
                flush=True,
            )
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
            boundary_rejections += 1
            continue

        component_vertex_ids = np.unique(component_faces)
        global_to_local = {
            int(vertex_id): local_index
            for local_index, vertex_id in enumerate(component_vertex_ids)
        }
        local_faces = np.asarray(
            [
                [global_to_local[int(vertex_id)] for vertex_id in face]
                for face in component_faces
            ],
            dtype=np.int64,
        )
        local_vertices = vertices[component_vertex_ids]
        boundary_cycle = np.asarray(cycles[0], dtype=np.int64)
        local_boundary = np.asarray(
            [global_to_local[int(vertex_id)] for vertex_id in boundary_cycle],
            dtype=np.int64,
        )
        if len(local_boundary) < 4:
            boundary_rejections += 1
            continue
        first_pin = int(local_boundary[0])
        first_distances = np.linalg.norm(
            local_vertices[local_boundary] - local_vertices[first_pin], axis=1
        )
        second_pin = int(local_boundary[int(np.argmax(first_distances))])
        pin_distance = float(
            np.linalg.norm(local_vertices[second_pin] - local_vertices[first_pin])
        )
        if pin_distance <= np.finfo(np.float64).eps:
            parameter_rejections += 1
            continue
        try:
            success, uv_vertices = igl.lscm(
                local_vertices,
                local_faces,
                np.asarray((first_pin, second_pin), dtype=np.int64),
                np.asarray(((0.0, 0.0), (pin_distance, 0.0)), dtype=np.float64),
            )
        except (RuntimeError, ValueError):
            success = False
        if not success:
            parameter_rejections += 1
            continue
        uv_vertices = np.asarray(uv_vertices, dtype=np.float64)
        if uv_vertices.shape != (len(local_vertices), 2) or not np.isfinite(
            uv_vertices
        ).all():
            parameter_rejections += 1
            continue

        parameter_triangles = uv_vertices[local_faces]
        signed_double_areas = (
            (parameter_triangles[:, 1, 0] - parameter_triangles[:, 0, 0])
            * (parameter_triangles[:, 2, 1] - parameter_triangles[:, 0, 1])
            - (parameter_triangles[:, 1, 1] - parameter_triangles[:, 0, 1])
            * (parameter_triangles[:, 2, 0] - parameter_triangles[:, 0, 0])
        )
        area_scale = max(float(np.ptp(uv_vertices, axis=0).max()) ** 2, 1.0)
        if np.any(np.abs(signed_double_areas) <= area_scale * 1e-14):
            parameter_rejections += 1
            continue
        positive_count = int(np.count_nonzero(signed_double_areas > 0.0))
        negative_count = len(signed_double_areas) - positive_count
        if min(positive_count, negative_count) > 0:
            parameter_rejections += 1
            continue
        if negative_count:
            uv_vertices[:, 1] *= -1.0

        local_edges = np.unique(
            np.sort(
                local_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
                axis=1,
            ),
            axis=0,
        )
        spatial_lengths = np.linalg.norm(
            local_vertices[local_edges[:, 0]]
            - local_vertices[local_edges[:, 1]],
            axis=1,
        )
        parameter_lengths = np.linalg.norm(
            uv_vertices[local_edges[:, 0]] - uv_vertices[local_edges[:, 1]],
            axis=1,
        )
        usable_lengths = parameter_lengths > np.finfo(np.float64).eps
        if not np.any(usable_lengths):
            parameter_rejections += 1
            continue
        uv_vertices *= float(
            np.median(spatial_lengths[usable_lengths] / parameter_lengths[usable_lengths])
        )

        local_boundary_edges = np.column_stack(
            (local_boundary, np.roll(local_boundary, -1))
        )
        planar_vertices = np.column_stack(
            (uv_vertices, np.zeros(len(uv_vertices), dtype=np.float64))
        )
        (
            rebuilt_parameter_vertices,
            rebuilt_faces,
            rebuilt_stats,
        ) = retriangulate_planar_annuli(
            planar_vertices,
            local_faces,
            protected_edges=local_boundary_edges,
            minimum_faces=1,
            minimum_holes=0,
            maximum_holes=None,
            maximum_target_edge_length=maximum_target_edge_length,
            face_regions=[np.arange(len(local_faces), dtype=np.int64)],
            solid_minimum_faces=1,
            holed_minimum_faces=1,
            minimum_triangle_angle_degrees=minimum_triangle_angle_degrees,
            announce_fallback=False,
            enforce_quality_gate=False,
            fallback_flip_passes=50,
            allow_ear_clipping_fallback=True,
        )
        if rebuilt_stats["regions"] != 1:
            chart_rejections += 1
            chart_boundary_rejections += rebuilt_stats["boundary_rejections"]
            chart_constraint_rejections += rebuilt_stats["constraint_rejections"]
            chart_quality_rejections += rebuilt_stats["quality_rejections"]
            continue

        new_uv_points = rebuilt_parameter_vertices[len(local_vertices):, :2]
        try:
            new_vertices = _map_uv_points_to_surface(
                new_uv_points,
                uv_vertices,
                local_faces,
                local_vertices,
            )
        except ValueError:
            parameter_rejections += 1
            continue
        first_new_index = len(vertices) + len(appended_vertices)
        local_to_global = np.concatenate(
            (
                component_vertex_ids,
                np.arange(
                    first_new_index,
                    first_new_index + len(new_vertices),
                    dtype=np.int64,
                ),
            )
        )
        new_faces = local_to_global[np.asarray(rebuilt_faces, dtype=np.int64)]
        coordinate_pool = np.vstack(
            (
                vertices,
                np.asarray(appended_vertices, dtype=np.float64).reshape(-1, 3),
                new_vertices,
            )
        )
        new_quality = _triangle_quality_values(coordinate_pool, new_faces)
        if (
            float(np.percentile(new_quality, 5.0))
            < float(np.percentile(old_quality, 5.0)) - 1e-8
            or float(new_quality.mean()) < float(old_quality.mean()) - 1e-8
        ):
            quality_rejections += 1
            last_quality_rejection = {
                "old_p5": float(np.percentile(old_quality, 5.0)),
                "new_p5": float(np.percentile(new_quality, 5.0)),
                "old_mean": float(old_quality.mean()),
                "new_mean": float(new_quality.mean()),
            }
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
            "candidates": candidates,
            "screened_out": screened_out,
            "regions": 0,
            "parameter_rejections": parameter_rejections,
            "boundary_rejections": boundary_rejections,
            "chart_rejections": chart_rejections,
            "chart_boundary_rejections": chart_boundary_rejections,
            "chart_constraint_rejections": chart_constraint_rejections,
            "chart_quality_rejections": chart_quality_rejections,
            "quality_rejections": quality_rejections,
            "new_vertices": 0,
            "removed_faces": 0,
            "new_faces": 0,
            "old_quality": 0.0,
            "new_quality": 0.0,
            "old_quality_p5": 0.0,
            "new_quality_p5": 0.0,
            "last_quality_rejection": last_quality_rejection,
        }
    return (
        np.vstack((vertices, np.asarray(appended_vertices, dtype=np.float64))),
        np.vstack((faces[kept_faces], np.asarray(appended_faces, dtype=np.int64))),
        {
            "candidates": candidates,
            "screened_out": screened_out,
            "regions": accepted,
            "parameter_rejections": parameter_rejections,
            "boundary_rejections": boundary_rejections,
            "chart_rejections": chart_rejections,
            "chart_boundary_rejections": chart_boundary_rejections,
            "chart_constraint_rejections": chart_constraint_rejections,
            "chart_quality_rejections": chart_quality_rejections,
            "quality_rejections": quality_rejections,
            "new_vertices": len(appended_vertices),
            "removed_faces": removed_faces,
            "new_faces": len(appended_faces),
            "old_quality": float(np.mean(old_qualities)),
            "new_quality": float(np.mean(new_qualities)),
            "old_quality_p5": float(np.percentile(old_qualities, 5.0)),
            "new_quality_p5": float(np.percentile(new_qualities, 5.0)),
            "last_quality_rejection": last_quality_rejection,
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
        if float(np.percentile(radial_alignment, 5.0)) < 0.98:
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
        triangle_result = constrained_triangle.triangulate(
            {"vertices": parameter_vertices, "segments": segments}, "pQY"
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
            new_vertices = np.asarray(new_vertices, dtype=np.float64)
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
        },
    )


def retriangulate_cylindrical_walls(
    vertices,
    faces,
    protected_edges,
    minimum_faces=20,
    radius_tolerance=1e-3,
    target_edge_ratio=1.0,
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


def sample_region_boundary_edges(
    vertices,
    faces,
    boundary_edges,
    target_edge_length,
    relative_tolerance=1e-8,
    tracked_edge_roots=None,
    tracked_face_labels=None,
):
    """Synchronously sample shared reconstruction boundaries.

    Only the supplied semantic-region boundary chains are subdivided.  The
    adjacent faces are split solely to keep the mesh conforming.  When a
    mutable ``tracked_edge_roots`` mapping is supplied, split hard edges are
    replaced in that mapping by their two children so feature lineage remains
    valid for later refinement stages.  Optional face labels are inherited by
    both children of every split face and returned in the statistics mapping;
    this preserves exact semantic-region membership without reclassification.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    target_edge_length = float(target_edge_length)
    if target_edge_length <= 0.0:
        raise ValueError("Region-boundary target edge length must be positive.")
    selected = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(boundary_edges, dtype=np.int64).reshape(-1, 2)
    }
    if tracked_face_labels is not None:
        tracked_face_labels = list(
            map(int, np.asarray(tracked_face_labels).reshape(-1))
        )
        if len(tracked_face_labels) != len(faces):
            raise ValueError(
                "Tracked face labels must match the input face count."
            )
    if not selected:
        return vertices.copy(), faces.copy(), {
            "initial_edges": 0,
            "splits": 0,
            "final_edges": 0,
            "initial_max_length": 0.0,
            "final_max_length": 0.0,
            "face_labels": (
                np.asarray(tracked_face_labels, dtype=np.int64)
                if tracked_face_labels is not None
                else None
            ),
        }

    vertices_list = [vertex.copy() for vertex in vertices]
    faces_list = [list(map(int, face)) for face in faces]
    edge_faces = _build_edge_faces(faces_list)
    selected.intersection_update(edge_faces)
    initial_lengths = {
        edge: float(
            np.linalg.norm(vertices_list[edge[0]] - vertices_list[edge[1]])
        )
        for edge in selected
    }
    length_limit = target_edge_length * (1.0 + float(relative_tolerance))
    heap = [
        (-length, edge)
        for edge, length in initial_lengths.items()
        if length > length_limit
    ]
    heapq.heapify(heap)
    active_boundary_edges = set(selected)
    split_count = 0
    maximum_splits = max(100000, len(heap) * 128)
    while heap and split_count < maximum_splits:
        negative_length, edge = heapq.heappop(heap)
        if edge not in active_boundary_edges or edge not in edge_faces:
            continue
        current_length = float(
            np.linalg.norm(vertices_list[edge[0]] - vertices_list[edge[1]])
        )
        if current_length <= length_limit:
            continue
        if current_length < -negative_length * (1.0 - 1e-12):
            continue
        incident_faces = sorted(edge_faces[edge])
        if not incident_faces:
            continue
        midpoint_index = len(vertices_list)
        vertices_list.append(
            (vertices_list[edge[0]] + vertices_list[edge[1]]) * 0.5
        )
        replacements = []
        for face_index in incident_faces:
            first_face, second_face = _constraint_split_face(
                faces_list[face_index], edge, midpoint_index
            )
            replacements.append((face_index, first_face, second_face))
        for face_index, _, _ in replacements:
            for old_edge in _constraint_face_edges(faces_list[face_index]):
                memberships = edge_faces.get(old_edge)
                if memberships is not None:
                    memberships.discard(face_index)
                    if not memberships:
                        del edge_faces[old_edge]
        for face_index, first_face, second_face in replacements:
            faces_list[face_index] = first_face
            second_face_index = len(faces_list)
            faces_list.append(second_face)
            if tracked_face_labels is not None:
                tracked_face_labels.append(tracked_face_labels[face_index])
            for new_edge in _constraint_face_edges(first_face):
                edge_faces.setdefault(new_edge, set()).add(face_index)
            for new_edge in _constraint_face_edges(second_face):
                edge_faces.setdefault(new_edge, set()).add(second_face_index)

        active_boundary_edges.discard(edge)
        children = (
            tuple(sorted((edge[0], midpoint_index))),
            tuple(sorted((midpoint_index, edge[1]))),
        )
        if tracked_edge_roots is not None:
            root_index = tracked_edge_roots.pop(edge, None)
            if root_index is not None:
                tracked_edge_roots[children[0]] = root_index
                tracked_edge_roots[children[1]] = root_index
        active_boundary_edges.update(children)
        for child in children:
            child_length = float(
                np.linalg.norm(
                    vertices_list[child[0]] - vertices_list[child[1]]
                )
            )
            if child_length > length_limit:
                heapq.heappush(heap, (-child_length, child))
        split_count += 1
    if heap:
        raise RuntimeError(
            "Region-boundary sampling exceeded its safety budget."
        )

    final_lengths = [
        float(np.linalg.norm(vertices_list[first] - vertices_list[second]))
        for first, second in active_boundary_edges
    ]
    return (
        np.asarray(vertices_list, dtype=np.float64),
        np.asarray(faces_list, dtype=np.int64),
        {
            "initial_edges": len(selected),
            "splits": split_count,
            "final_edges": len(active_boundary_edges),
            "initial_max_length": (
                max(initial_lengths.values()) if initial_lengths else 0.0
            ),
            "final_max_length": max(final_lengths) if final_lengths else 0.0,
            "face_labels": (
                np.asarray(tracked_face_labels, dtype=np.int64)
                if tracked_face_labels is not None
                else None
            ),
        },
    )


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
    feature_target_edge_length=None,
    max_splits=None,
    coplanar_angle_degrees=0.1,
    coplanar_distance_ratio=1e-6,
    flip_passes=8,
    flip_minimum_valence=None,
    flip_maximum_candidate_quality=None,
    planar_fan_minimum_valence=None,
    planar_annulus_minimum_faces=None,
    cylinder_minimum_faces=None,
    cylinder_radius_tolerance=1e-3,
    cylinder_target_edge_ratio=1.0,
    partial_cylinder_minimum_faces=None,
    partial_cylinder_radius_tolerance=2e-3,
    partial_cylinder_normal_tolerance=2e-2,
    partial_cylinder_minimum_angle=30.0,
    rounded_fillet_minimum_faces=None,
    rounded_fillet_minimum_curvature=0.2,
    extrusion_region_minimum_faces=None,
    extrusion_maximum_input_quality=0.05,
    planar_region_minimum_faces=None,
    planar_target_edge_length=None,
    planar_minimum_angle_degrees=None,
    planar_largest_opposed_pair_only=False,
):
    """
    Refine the original surface while retaining hard feature edge lineages.

    Hard edges (dihedral > threshold, boundary, or non-manifold) are tracked as
    lineages. Splitting a hard edge replaces it by two collinear hard children.
    All edges are bisected to the requested global length.  When
    ``feature_target_edge_length`` is supplied, hard feature chains use that
    finer independent bound while smooth high-quality regions may stay coarse.
    Quality-improving flips then remove non-feature seams only inside coplanar
    patches.
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
    coplanar_distance_ratio = float(coplanar_distance_ratio)
    if coplanar_distance_ratio < 0.0:
        raise ValueError("Coplanar distance ratio must be non-negative.")
    flip_passes = int(flip_passes)
    if flip_passes < 0:
        raise ValueError("Constraint flip passes must be non-negative.")
    if (
        feature_target_edge_length is not None
        and float(feature_target_edge_length) <= 0.0
    ):
        raise ValueError("Feature target edge length must be positive.")
    if feature_target_edge_length is not None:
        feature_target_edge_length = float(feature_target_edge_length)
    if (
        planar_target_edge_length is not None
        and float(planar_target_edge_length) <= 0.0
    ):
        raise ValueError("Planar target edge length must be positive.")
    if (
        extrusion_region_minimum_faces is not None
        and int(extrusion_region_minimum_faces) < 2
    ):
        raise ValueError(
            "Developable reconstruction minimum faces must be at least 2."
        )
    planar_largest_opposed_pair_only = bool(
        planar_largest_opposed_pair_only
    )
    if planar_largest_opposed_pair_only and (
        planar_annulus_minimum_faces is None
        and planar_region_minimum_faces is None
    ):
        raise ValueError(
            "Largest opposed planar-pair selection requires planar "
            "reconstruction."
        )
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

    planar_reconstruction_requested = (
        planar_annulus_minimum_faces is not None
        or planar_region_minimum_faces is not None
    )
    surface_classification_requested = (
        planar_reconstruction_requested
        or extrusion_region_minimum_faces is not None
    )
    if surface_classification_requested:
        classification_minimum_faces = min(
            value
            for value in (
                planar_annulus_minimum_faces,
                planar_region_minimum_faces,
                extrusion_region_minimum_faces,
            )
            if value is not None
        )
        all_planar_regions, classification_stats = classify_planar_face_regions(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=classification_minimum_faces,
            maximum_normal_angle_degrees=coplanar_angle_degrees,
            maximum_plane_distance_ratio=coplanar_distance_ratio,
        )
        nonplanar_regions, nonplanar_stats = classify_nonplanar_face_regions(
            vertices,
            faces,
            all_planar_regions,
            protected_edges=hard_edges_array,
            minimum_faces=4,
            axial_normal_tolerance=partial_cylinder_normal_tolerance,
            radius_tolerance=max(cylinder_radius_tolerance, 1e-2),
        )
        print(
            "Surface classification: {} planar patch(es) / {} faces; "
            "{} cylinder region(s) / {} faces; {} fillet region(s) / {} "
            "faces; {} extrusion region(s) / {} faces; {} general curved "
            "faces; "
            "normal tolerance {:.6g} degrees, plane distance {:.6g}.".format(
                classification_stats["regions"],
                classification_stats["planar_faces"],
                nonplanar_stats["cylinder_regions"],
                nonplanar_stats["cylinder_faces"],
                nonplanar_stats["fillet_regions"],
                nonplanar_stats["fillet_faces"],
                nonplanar_stats["extrusion_regions"],
                nonplanar_stats["extrusion_faces"],
                nonplanar_stats["general_curved_faces"],
                classification_stats["normal_angle_degrees"],
                classification_stats["distance_limit"],
            ),
            flush=True,
        )
        planar_regions = all_planar_regions
        if planar_largest_opposed_pair_only:
            planar_regions, planar_pair_stats = (
                select_largest_opposed_planar_regions(
                    vertices,
                    faces,
                    all_planar_regions,
                )
            )
            print(
                "Primary plate planes selected: {} / {} faces, area "
                "{:.6g} / {:.6g}, normals {} / {}, dot {:.6g}.".format(
                    planar_pair_stats["primary_faces"],
                    planar_pair_stats["opposite_faces"],
                    planar_pair_stats["primary_area"],
                    planar_pair_stats["opposite_area"],
                    np.array2string(
                        planar_pair_stats["primary_normal"], precision=5
                    ),
                    np.array2string(
                        planar_pair_stats["opposite_normal"], precision=5
                    ),
                    planar_pair_stats["normal_dot"],
                ),
                flush=True,
            )
        planar_edge_target = (
            float(planar_target_edge_length)
            if planar_target_edge_length is not None
            else float(np.linalg.norm(np.ptp(vertices, axis=0))) * 0.05
        )

        region_labels = np.full(len(faces), -1, dtype=np.int64)
        next_region_label = 0
        if planar_reconstruction_requested:
            requested_planar_minimum = min(
                value
                for value in (
                    planar_annulus_minimum_faces,
                    planar_region_minimum_faces,
                )
                if value is not None
            )
            for region in planar_regions:
                if len(region) >= requested_planar_minimum:
                    region_labels[region] = next_region_label
                    next_region_label += 1
        selected_planar_label_count = next_region_label
        if extrusion_region_minimum_faces is not None:
            for region in nonplanar_regions["extrusions"]:
                if len(region["faces"]) >= int(extrusion_region_minimum_faces):
                    region_labels[region["faces"]] = next_region_label
                    next_region_label += 1
        reconstruction_boundary_edges = []
        current_edge_faces = _build_edge_faces(faces)
        for edge, memberships in current_edge_faces.items():
            if len(memberships) != 2:
                continue
            first_face, second_face = tuple(memberships)
            first_label = int(region_labels[first_face])
            second_label = int(region_labels[second_face])
            if planar_largest_opposed_pair_only:
                selected_boundary = (
                    first_label != second_label
                    and (
                        0 <= first_label < selected_planar_label_count
                        or 0 <= second_label < selected_planar_label_count
                    )
                )
            else:
                selected_boundary = (
                    edge not in hard_edges
                    and first_label >= 0
                    and second_label >= 0
                    and first_label != second_label
                )
            if selected_boundary and (
                float(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))
                > planar_edge_target * (1.0 + 1e-8)
            ):
                reconstruction_boundary_edges.append(edge)
        if reconstruction_boundary_edges:
            vertices, faces, boundary_sample_stats = sample_region_boundary_edges(
                vertices,
                faces,
                reconstruction_boundary_edges,
                planar_edge_target,
                tracked_edge_roots=(
                    root_for_edge
                    if planar_largest_opposed_pair_only
                    else None
                ),
                tracked_face_labels=(
                    region_labels
                    if planar_largest_opposed_pair_only
                    else None
                ),
            )
            if planar_largest_opposed_pair_only:
                hard_edges = set(root_for_edge)
                hard_edges_array = np.asarray(
                    sorted(hard_edges), dtype=np.int64
                ).reshape(-1, 2)
            print(
                "Synchronized reconstruction boundaries: {} selected edge(s), "
                "{} conformity samples; longest boundary edge {:.6g} -> "
                "{:.6g}.".format(
                    boundary_sample_stats["initial_edges"],
                    boundary_sample_stats["splits"],
                    boundary_sample_stats["initial_max_length"],
                    boundary_sample_stats["final_max_length"],
                ),
                flush=True,
            )
            all_planar_regions, classification_stats = classify_planar_face_regions(
                vertices,
                faces,
                protected_edges=hard_edges_array,
                minimum_faces=classification_minimum_faces,
                maximum_normal_angle_degrees=coplanar_angle_degrees,
                maximum_plane_distance_ratio=coplanar_distance_ratio,
            )
            nonplanar_regions, nonplanar_stats = classify_nonplanar_face_regions(
                vertices,
                faces,
                all_planar_regions,
                protected_edges=hard_edges_array,
                minimum_faces=4,
                axial_normal_tolerance=partial_cylinder_normal_tolerance,
                radius_tolerance=max(cylinder_radius_tolerance, 1e-2),
            )
            if planar_largest_opposed_pair_only:
                inherited_labels = boundary_sample_stats["face_labels"]
                planar_regions = [
                    np.flatnonzero(inherited_labels == label)
                    for label in range(selected_planar_label_count)
                ]
            else:
                planar_regions = all_planar_regions

        if extrusion_region_minimum_faces is not None:
            vertices, faces, extrusion_stats = (
                retriangulate_developable_regions(
                    vertices,
                    faces,
                    nonplanar_regions["extrusions"],
                    minimum_faces=extrusion_region_minimum_faces,
                    maximum_target_edge_length=planar_edge_target,
                    minimum_triangle_angle_degrees=(
                        planar_minimum_angle_degrees
                        if planar_minimum_angle_degrees is not None
                        else 28.0
                    ),
                    maximum_input_quality=extrusion_maximum_input_quality,
                )
            )
            print(
                "Boundary-only developable reconstruction: {} / {} "
                "poor/oversized region(s), {} screened out, {} new "
                "vertices, {} old -> {} new faces; "
                "quality mean {:.6g} -> {:.6g}, P5 {:.6g} -> {:.6g}; "
                "{} parameter, {} boundary, {} "
                "chart, {} mapped-quality rejected.".format(
                    extrusion_stats["regions"],
                    extrusion_stats["candidates"],
                    extrusion_stats["screened_out"],
                    extrusion_stats["new_vertices"],
                    extrusion_stats["removed_faces"],
                    extrusion_stats["new_faces"],
                    extrusion_stats["old_quality"],
                    extrusion_stats["new_quality"],
                    extrusion_stats["old_quality_p5"],
                    extrusion_stats["new_quality_p5"],
                    extrusion_stats["parameter_rejections"],
                    extrusion_stats["boundary_rejections"],
                    extrusion_stats["chart_rejections"],
                    extrusion_stats["quality_rejections"],
                ),
                flush=True,
            )
            if extrusion_stats["regions"] and planar_reconstruction_requested:
                planar_minimum_faces = min(
                    value
                    for value in (
                        planar_annulus_minimum_faces,
                        planar_region_minimum_faces,
                    )
                    if value is not None
                )
                all_planar_regions, _ = classify_planar_face_regions(
                    vertices,
                    faces,
                    protected_edges=hard_edges_array,
                    minimum_faces=planar_minimum_faces,
                    maximum_normal_angle_degrees=coplanar_angle_degrees,
                    maximum_plane_distance_ratio=coplanar_distance_ratio,
                )
                planar_regions = all_planar_regions
                if planar_largest_opposed_pair_only:
                    planar_regions, _ = select_largest_opposed_planar_regions(
                        vertices,
                        faces,
                        all_planar_regions,
                    )

    if planar_reconstruction_requested:
        planar_minimum_faces = min(
            value
            for value in (
                planar_annulus_minimum_faces,
                planar_region_minimum_faces,
            )
            if value is not None
        )
        minimum_holes = 0 if planar_region_minimum_faces is not None else 1
        maximum_holes = None if planar_annulus_minimum_faces is not None else 0
        vertices, faces, planar_stats = retriangulate_planar_annuli(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=planar_minimum_faces,
            minimum_holes=minimum_holes,
            maximum_holes=maximum_holes,
            maximum_target_edge_length=planar_edge_target,
            face_regions=planar_regions,
            solid_minimum_faces=planar_region_minimum_faces,
            holed_minimum_faces=planar_annulus_minimum_faces,
            minimum_triangle_angle_degrees=planar_minimum_angle_degrees,
        )
        print(
            "Boundary-only planar reconstruction: {} region(s), {} new "
            "vertices, {} old -> {} new faces; mean quality {:.6g} -> "
            "{:.6g}; {} candidates ({} hard-constraint, {} boundary, {} "
            "quality rejected).".format(
                planar_stats["regions"], planar_stats["new_vertices"],
                planar_stats["removed_faces"], planar_stats["new_faces"],
                planar_stats["old_quality"], planar_stats["new_quality"],
                planar_stats["candidates"],
                planar_stats["constraint_rejections"],
                planar_stats["boundary_rejections"],
                planar_stats["quality_rejections"],
            ),
            flush=True,
        )

    if cylinder_minimum_faces is not None:
        vertices, faces, cylinder_stats = retriangulate_cylindrical_walls(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=cylinder_minimum_faces,
            radius_tolerance=cylinder_radius_tolerance,
            target_edge_ratio=cylinder_target_edge_ratio,
        )
        print(
            "Cylindrical wall retriangulation: {} / {} candidate wall(s), "
            "{} new vertices, {} old -> {} new faces; mean quality "
            "{:.6g} -> {:.6g}.".format(
                cylinder_stats["walls"], cylinder_stats["candidates"],
                cylinder_stats["new_vertices"], cylinder_stats["removed_faces"],
                cylinder_stats["new_faces"], cylinder_stats["old_quality"],
                cylinder_stats["new_quality"],
            ),
            flush=True,
        )

    if partial_cylinder_minimum_faces is not None:
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
            )
        )
        print(
            "Partial cylindrical wall retriangulation: {} / {} candidate "
            "patch(es), {} new vertices, {} old -> {} new faces; mean "
            "quality {:.6g} -> {:.6g}.".format(
                partial_stats["patches"], partial_stats["candidates"],
                partial_stats["new_vertices"], partial_stats["removed_faces"],
                partial_stats["new_faces"], partial_stats["old_quality"],
                partial_stats["new_quality"],
            ),
            flush=True,
        )

    if rounded_fillet_minimum_faces is not None:
        vertices, faces, fillet_stats = retriangulate_partial_cylindrical_walls(
            vertices,
            faces,
            protected_edges=hard_edges_array,
            minimum_faces=rounded_fillet_minimum_faces,
            radius_tolerance=max(partial_cylinder_radius_tolerance, 1e-2),
            normal_tolerance=max(partial_cylinder_normal_tolerance, 8e-2),
            minimum_angle_degrees=5.0,
            target_edge_ratio=cylinder_target_edge_ratio,
            isolate_rounded_faces=True,
            minimum_curvature_degrees=rounded_fillet_minimum_curvature,
        )
        print(
            "Rounded fillet retriangulation: {} / {} candidate band(s), "
            "{} new vertices, {} old -> {} new faces; mean quality "
            "{:.6g} -> {:.6g}.".format(
                fillet_stats["patches"], fillet_stats["candidates"],
                fillet_stats["new_vertices"], fillet_stats["removed_faces"],
                fillet_stats["new_faces"], fillet_stats["old_quality"],
                fillet_stats["new_quality"],
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
    relative_tolerance = 1e-8
    feature_split_edges = set()
    initial_feature_lengths = []
    if feature_target_edge_length is not None:
        feature_length_limit = feature_target_edge_length * (
            1.0 + relative_tolerance
        )
        for edge, length in zip(edge_faces, initial_lengths):
            if edge not in root_for_edge:
                continue
            initial_feature_lengths.append(float(length))
            if length > feature_length_limit:
                feature_split_edges.add(edge)
        print(
            "Feature-local refinement: {} hard edges exceed {:.6g}; smooth "
            "regions keep the global size bound.".format(
                len(feature_split_edges), feature_target_edge_length
            ),
            flush=True,
        )
    initial_feature_max_length = (
        max(initial_feature_lengths) if initial_feature_lengths else 0.0
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
        feature_safety_budget = 0
        feature_edge_estimate = 0
        if feature_target_edge_length is not None:
            feature_safety_budget, feature_edge_estimate = (
                automatic_split_limit(
                    np.asarray(initial_feature_lengths, dtype=np.float64),
                    feature_target_edge_length,
                )
            )
        max_splits = max(
            max_splits,
            feature_safety_budget,
        )
        print(
            "Automatic split limit: {} (global edge-only estimate {}, "
            "feature estimate {}, minimum "
            "100000).".format(
                max_splits,
                edge_only_estimate,
                feature_edge_estimate,
            )
        )
    max_splits = int(max_splits)
    if max_splits <= 0:
        raise ValueError("Constraint maximum splits must be positive.")

    length_limit = max_edge_length * (1.0 + relative_tolerance)
    if (
        initial_max_length <= length_limit
        and not feature_split_edges
    ):
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
            maximum_candidate_quality=flip_maximum_candidate_quality,
        )
        return optimized_vertices, optimized_faces, {
            "hard_edges": len(hard_edges),
            "hard_edges_split": 0,
            "splits": 0,
            "feature_driven_splits": 0,
            "initial_feature_split_edges": 0,
            "initial_max_length": initial_max_length,
            "final_max_length": initial_max_length,
            "initial_feature_max_length": initial_feature_max_length,
            "final_feature_max_length": initial_feature_max_length,
            "already_satisfied": True,
            "coplanar_flips": flip_count,
        }

    vertices_list = [vertex.copy() for vertex in vertices]
    faces_list = [list(map(int, face)) for face in faces]
    feature_length_limit = (
        feature_target_edge_length * (1.0 + relative_tolerance)
        if feature_target_edge_length is not None
        else None
    )
    def edge_requires_feature_split(edge, edge_length=None):
        if feature_length_limit is None or edge not in root_for_edge:
            return False
        if edge_length is None:
            edge_length = float(
                np.linalg.norm(
                    vertices_list[edge[0]] - vertices_list[edge[1]]
                )
            )
        return edge_length > feature_length_limit

    heap = []
    for edge, length in zip(edge_faces, initial_lengths):
        if (
            length > length_limit
            or edge in feature_split_edges
        ):
            heapq.heappush(heap, (-float(length), edge))

    print(
        "Longest-edge refinement: {} initial global/feature "
        "candidates; global limit {:.6g}, feature limit {}, split budget {}."
        .format(
            len(heap),
            max_edge_length,
            (
                "{:.6g}".format(feature_target_edge_length)
                if feature_target_edge_length is not None
                else "disabled"
            ),
            max_splits,
        ),
        flush=True,
    )

    split_count = 0
    feature_driven_split_count = 0
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
        global_split_required = current_length > length_limit
        feature_split_required = edge_requires_feature_split(
            edge,
            edge_length=current_length,
        )
        if not (
            global_split_required
            or feature_split_required
        ):
            continue
        if current_length < -negative_length * (1.0 - 1e-12):
            continue

        incident_faces = sorted(edge_faces[edge])
        if not incident_faces:
            del edge_faces[edge]
            continue
        if feature_split_required:
            feature_driven_split_count += 1
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
            if (
                new_length > length_limit
                or edge_requires_feature_split(
                    new_edge,
                    edge_length=new_length,
                )
            ):
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
    remaining_feature_edges = []
    if feature_target_edge_length is not None:
        for edge in root_for_edge:
            if edge not in edge_faces:
                continue
            length = float(
                np.linalg.norm(
                    vertices_list[edge[0]] - vertices_list[edge[1]]
                )
            )
            if length > feature_length_limit:
                remaining_feature_edges.append((edge, length))
    if remaining_feature_edges:
        longest_remaining = max(
            length for _, length in remaining_feature_edges
        )
        raise RuntimeError(
            "Constraint split limit {} was reached with {} hard feature "
            "edges still over the {:.6g} local target; longest remaining "
            "feature edge is {:.6g}. Increase max_splits or relax the "
            "feature target.".format(
                max_splits,
                len(remaining_feature_edges),
                feature_target_edge_length,
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
        maximum_candidate_quality=flip_maximum_candidate_quality,
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
    final_feature_lengths = [
        float(
            np.linalg.norm(
                refined_vertices[edge[0]] - refined_vertices[edge[1]]
            )
        )
        for edge in root_for_edge
        if edge in edge_faces
    ]
    final_feature_max_length = (
        max(final_feature_lengths) if final_feature_lengths else 0.0
    )
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
    if feature_target_edge_length is not None:
        print(
            "Feature-local longest edge: {:.6g} -> {:.6g} (limit {:.6g}); "
            "{} targeted feature splits.".format(
                initial_feature_max_length,
                final_feature_max_length,
                feature_target_edge_length,
                feature_driven_split_count,
            )
        )
    return (
        refined_vertices,
        refined_faces,
        {
            "hard_edges": len(original_hard_lengths),
            "hard_edges_split": len(split_hard_roots),
            "splits": split_count,
            "feature_driven_splits": feature_driven_split_count,
            "initial_feature_split_edges": len(feature_split_edges),
            "initial_max_length": initial_max_length,
            "final_max_length": final_max_length,
            "initial_feature_max_length": initial_feature_max_length,
            "final_feature_max_length": final_feature_max_length,
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
