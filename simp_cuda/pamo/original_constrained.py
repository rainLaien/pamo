import heapq

import numpy as np
import trimesh
import igl
from scipy.spatial import cKDTree

from .segment_query import (
    closest_points_on_segments as _closest_points_on_segments,
)
from .feature_optimize import _flip_quality_edges


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


def refine_original_mesh_by_longest_edge(
    reference_mesh,
    max_edge_length=None,
    feature_angle_degrees=5.0,
    max_splits=100000,
    coplanar_angle_degrees=0.1,
    flip_passes=8,
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

    if max_edge_length is None:
        print(
            "Strict original constraints: {} hard edges at > {:.6g} degrees; "
            "no maximum edge length requested, so no vertices were inserted."
            .format(len(hard_edges), float(feature_angle_degrees))
        )
        optimized_faces, flip_count = _flip_quality_edges(
            vertices,
            faces,
            hard_edges_array,
            passes=flip_passes,
            maximum_dihedral_degrees=coplanar_angle_degrees,
            preferred_edges=np.asarray(
                sorted(coplanar_edges), dtype=np.int64
            ).reshape(-1, 2),
        )
        return vertices.copy(), optimized_faces, {
            "hard_edges": len(hard_edges),
            "hard_edges_split": 0,
            "splits": 0,
            "initial_max_length": initial_max_length,
            "final_max_length": initial_max_length,
            "already_satisfied": True,
            "coplanar_flips": flip_count,
        }

    max_edge_length = float(max_edge_length)
    if max_edge_length <= 0.0:
        raise ValueError("Constraint maximum edge length must be positive.")
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
        optimized_faces, flip_count = _flip_quality_edges(
            vertices,
            faces,
            hard_edges_array,
            passes=flip_passes,
            maximum_dihedral_degrees=coplanar_angle_degrees,
            maximum_edge_length=max_edge_length,
            preferred_edges=np.asarray(
                sorted(coplanar_edges), dtype=np.int64
            ).reshape(-1, 2),
        )
        return vertices.copy(), optimized_faces, {
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

    split_count = 0
    split_hard_roots = set()
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
