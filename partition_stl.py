"""Partition an STL into connected, curvature-coherent surface patches."""

import argparse
import colorsys
import csv
from pathlib import Path
import time

import igl
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class _DisjointSet:
    def __init__(
        self,
        count,
        curvature_descriptors=None,
        directions=None,
        direction_reliability=None,
    ):
        self.parent = np.arange(count, dtype=np.int64)
        self.size = np.ones(count, dtype=np.int64)
        self.internal = np.zeros(count, dtype=np.float64)
        self.curvature_min = np.asarray(
            curvature_descriptors, dtype=np.float64
        ).copy()
        self.curvature_max = self.curvature_min.copy()
        self.curvature_sum = self.curvature_min.copy()
        self.curvature_squared_sum = self.curvature_min ** 2
        self.direction = np.asarray(directions, dtype=np.float64).copy()
        self.direction_weight = np.asarray(
            direction_reliability, dtype=np.float64
        ).copy()
        self.direction_resultant = (
            self.direction * self.direction_weight[:, None]
        )

    def find(self, item):
        item = int(item)
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            following = int(self.parent[item])
            self.parent[item] = root
            item = following
        return root

    def merge(self, first, second, edge_weight):
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return first
        if self.size[first] < self.size[second]:
            first, second = second, first
        first_direction = self.direction[first]
        second_resultant = self.direction_resultant[second]
        if float(np.dot(first_direction, second_resultant)) < 0.0:
            second_resultant = -second_resultant
        direction_sum = self.direction_resultant[first] + second_resultant
        direction_length = float(np.linalg.norm(direction_sum))
        self.parent[second] = first
        self.size[first] += self.size[second]
        self.internal[first] = max(
            float(edge_weight),
            float(self.internal[first]),
            float(self.internal[second]),
        )
        self.curvature_min[first] = np.minimum(
            self.curvature_min[first], self.curvature_min[second]
        )
        self.curvature_max[first] = np.maximum(
            self.curvature_max[first], self.curvature_max[second]
        )
        self.curvature_sum[first] += self.curvature_sum[second]
        self.curvature_squared_sum[first] += (
            self.curvature_squared_sum[second]
        )
        self.direction_weight[first] += self.direction_weight[second]
        self.direction_resultant[first] = direction_sum
        if direction_length > np.finfo(np.float64).eps:
            self.direction[first] = direction_sum / direction_length
        return first

    def globally_compatible(
        self,
        first,
        second,
        maximum_curvature_span,
        maximum_direction_change,
        minimum_direction_reliability=0.2,
    ):
        first = self.find(first)
        second = self.find(second)
        combined_size = self.size[first] + self.size[second]
        combined_sum = self.curvature_sum[first] + self.curvature_sum[second]
        combined_squared_sum = (
            self.curvature_squared_sum[first]
            + self.curvature_squared_sum[second]
        )
        combined_mean = combined_sum / combined_size
        combined_variance = np.maximum(
            combined_squared_sum / combined_size - combined_mean ** 2,
            0.0,
        )
        if np.any(
            np.sqrt(combined_variance) > float(maximum_curvature_span)
        ):
            return False
        first_reliability = (
            self.direction_weight[first] / self.size[first]
        )
        second_reliability = (
            self.direction_weight[second] / self.size[second]
        )
        if (
            first_reliability >= float(minimum_direction_reliability)
            and second_reliability >= float(minimum_direction_reliability)
        ):
            second_resultant = self.direction_resultant[second]
            if float(np.dot(self.direction[first], second_resultant)) < 0.0:
                second_resultant = -second_resultant
            resultant_length = float(
                np.linalg.norm(
                    self.direction_resultant[first] + second_resultant
                )
            )
            total_weight = (
                self.direction_weight[first] + self.direction_weight[second]
            )
            if (
                1.0 - resultant_length / max(total_weight, 1e-30)
                > float(maximum_direction_change)
            ):
                return False
        return True


def _compact_mesh(vertices, faces):
    vertex_ids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return vertices[vertex_ids], inverse.reshape(-1, 3)


def _color(patch):
    hue = (0.08 + int(patch) * 0.6180339887498949) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
    return np.asarray((*np.rint(np.asarray(rgb) * 255.0), 255), dtype=np.uint8)


def _write_patch_id_ply(path, vertices, faces, labels, patch_ids=None):
    """Write a display-safe PLY with an integer ID on vertices and faces.

    Vertices on a patch boundary are duplicated. Consequently a viewer that
    only understands vertex colours cannot interpolate between two patches.
    Readers supporting custom PLY properties can inspect ``patch_id`` directly.
    """
    path = Path(path)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(faces) != len(labels):
        raise ValueError("Every triangle must have exactly one patch label.")
    if patch_ids is None:
        patch_ids = np.unique(labels)

    output_vertices = []
    output_faces = []
    vertex_patch_ids = []
    face_patch_ids = []
    vertex_colors = []
    face_colors = []
    vertex_offset = 0
    for patch in patch_ids:
        patch = int(patch)
        selected_faces = faces[labels == patch]
        if len(selected_faces) == 0:
            continue
        compact_vertices, compact_faces = _compact_mesh(
            vertices, selected_faces
        )
        color = _color(patch)
        output_vertices.append(compact_vertices)
        output_faces.append(compact_faces + vertex_offset)
        vertex_patch_ids.append(
            np.full(len(compact_vertices), patch, dtype=np.int32)
        )
        face_patch_ids.append(
            np.full(len(compact_faces), patch, dtype=np.int32)
        )
        vertex_colors.append(np.tile(color, (len(compact_vertices), 1)))
        face_colors.append(np.tile(color, (len(compact_faces), 1)))
        vertex_offset += len(compact_vertices)

    output_vertices = np.vstack(output_vertices).astype("<f4", copy=False)
    output_faces = np.vstack(output_faces).astype("<i4", copy=False)
    vertex_patch_ids = np.concatenate(vertex_patch_ids)
    face_patch_ids = np.concatenate(face_patch_ids)
    vertex_colors = np.vstack(vertex_colors)
    face_colors = np.vstack(face_colors)

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("patch_id", "<i4"),
            ("red", "u1"), ("green", "u1"),
            ("blue", "u1"), ("alpha", "u1"),
        ]
    )
    face_dtype = np.dtype(
        [
            ("vertex_count", "u1"),
            ("vertex_indices", "<i4", (3,)),
            ("patch_id", "<i4"),
            ("red", "u1"), ("green", "u1"),
            ("blue", "u1"), ("alpha", "u1"),
        ]
    )
    vertex_records = np.empty(len(output_vertices), dtype=vertex_dtype)
    vertex_records["x"] = output_vertices[:, 0]
    vertex_records["y"] = output_vertices[:, 1]
    vertex_records["z"] = output_vertices[:, 2]
    vertex_records["patch_id"] = vertex_patch_ids
    for channel, index in (
        ("red", 0), ("green", 1), ("blue", 2), ("alpha", 3)
    ):
        vertex_records[channel] = vertex_colors[:, index]

    face_records = np.empty(len(output_faces), dtype=face_dtype)
    face_records["vertex_count"] = 3
    face_records["vertex_indices"] = output_faces
    face_records["patch_id"] = face_patch_ids
    for channel, index in (
        ("red", 0), ("green", 1), ("blue", 2), ("alpha", 3)
    ):
        face_records[channel] = face_colors[:, index]

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment one connected surface patch has one integer patch_id",
            f"element vertex {len(vertex_records)}",
            "property float x", "property float y", "property float z",
            "property int patch_id",
            "property uchar red", "property uchar green",
            "property uchar blue", "property uchar alpha",
            f"element face {len(face_records)}",
            "property list uchar int vertex_indices",
            "property int patch_id",
            "property uchar red", "property uchar green",
            "property uchar blue", "property uchar alpha",
            "end_header", "",
        ]
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertex_records.tofile(stream)
        face_records.tofile(stream)


def _coherent_edge_chains(mesh, candidate_mask, minimum_edges=4):
    retained = np.zeros(len(candidate_mask), dtype=bool)
    candidate_ids = np.flatnonzero(candidate_mask)
    if len(candidate_ids) == 0:
        return retained
    candidate_edges = np.asarray(
        mesh.face_adjacency_edges, dtype=np.int64
    )[candidate_ids]
    graph = coo_matrix(
        (
            np.ones(len(candidate_edges) * 2, dtype=np.uint8),
            (
                np.concatenate((candidate_edges[:, 0], candidate_edges[:, 1])),
                np.concatenate((candidate_edges[:, 1], candidate_edges[:, 0])),
            ),
        ),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    ).tocsr()
    _, vertex_components = connected_components(graph, directed=False)
    edge_components = vertex_components[candidate_edges[:, 0]]
    chain_sizes = np.bincount(edge_components)
    coherent = chain_sizes[edge_components] >= int(minimum_edges)
    retained[candidate_ids[coherent]] = True
    return retained


def _face_curvature_descriptors(vertices, faces):
    direction_1, direction_2, curvature_1, curvature_2 = (
        igl.principal_curvature(vertices, faces)
    )
    absolute_1 = np.abs(np.asarray(curvature_1, dtype=np.float64))
    absolute_2 = np.abs(np.asarray(curvature_2, dtype=np.float64))
    small_first = absolute_1 <= absolute_2
    small_curvature = np.where(small_first, absolute_1, absolute_2)
    large_curvature = np.where(small_first, absolute_2, absolute_1)
    small_direction = np.where(
        small_first[:, None], direction_1, direction_2
    )

    vertex_curvatures = np.column_stack((small_curvature, large_curvature))
    face_curvatures = np.median(vertex_curvatures[faces], axis=1)
    anisotropy = (
        face_curvatures[:, 1] - face_curvatures[:, 0]
    ) / np.maximum(face_curvatures[:, 1], np.finfo(np.float64).eps)

    directions = small_direction[faces]
    covariance = np.einsum("fvi,fvj->fij", directions, directions)
    _, eigenvectors = np.linalg.eigh(covariance)
    face_directions = eigenvectors[:, :, -1]
    return face_curvatures, anisotropy, face_directions


def _fit_cylindrical_face_set(vertices, faces, radius_tolerance=1e-3):
    """Return a cylinder fit for a connected set of planar tessellation strips."""
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    cross_lengths = np.linalg.norm(crosses, axis=1)
    if np.any(cross_lengths <= np.finfo(np.float64).eps):
        return None
    normals = crosses / cross_lengths[:, None]
    eigenvalues, eigenvectors = np.linalg.eigh(normals.T @ normals / len(normals))
    axis = eigenvectors[:, 0]
    normal_error = float(np.sqrt(np.mean((normals @ axis) ** 2)))
    if normal_error > 2e-2:
        return None

    vertex_ids = np.unique(faces)
    points = vertices[vertex_ids]
    origin = points.mean(axis=0)
    first_basis = normals[0] - axis * float(np.dot(normals[0], axis))
    basis_length = float(np.linalg.norm(first_basis))
    if basis_length <= np.finfo(np.float64).eps:
        return None
    first_basis /= basis_length
    second_basis = np.cross(axis, first_basis)
    offsets = points - origin
    projected = np.column_stack(
        (offsets @ first_basis, offsets @ second_basis)
    )
    system = np.column_stack(
        (2.0 * projected[:, 0], 2.0 * projected[:, 1], np.ones(len(projected)))
    )
    solution, _, _, _ = np.linalg.lstsq(
        system, np.sum(projected * projected, axis=1), rcond=None
    )
    center_2d = solution[:2]
    radii = np.linalg.norm(projected - center_2d, axis=1)
    radius = float(radii.mean())
    if (
        radius <= np.finfo(np.float64).eps
        or float(radii.std() / radius) > float(radius_tolerance)
    ):
        return None

    axis_origin = (
        origin + center_2d[0] * first_basis + center_2d[1] * second_basis
    )
    centroids = triangles.mean(axis=1)
    centroid_offsets = centroids - axis_origin
    axial = centroid_offsets @ axis
    radial = centroid_offsets - axial[:, None] * axis
    radial_lengths = np.linalg.norm(radial, axis=1)
    alignment = np.abs(np.sum(normals * radial, axis=1)) / np.maximum(
        radial_lengths, np.finfo(np.float64).eps
    )
    if float(np.percentile(alignment, 5.0)) < 0.98:
        return None
    return {
        "axis": axis,
        "origin": axis_origin,
        "radius": radius,
        "normal_error": normal_error,
    }


def build_brep_model_partitions(vertices, faces, minimum_cylinder_facets=3):
    """Recover connected planar and cylindrical B-rep surfaces when unambiguous."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    facets = [np.asarray(facet, dtype=np.int64) for facet in mesh.facets]
    if not facets:
        return None
    face_to_facet = np.full(len(faces), -1, dtype=np.int64)
    for facet_id, facet_faces in enumerate(facets):
        if np.any(face_to_facet[facet_faces] >= 0):
            return None
        face_to_facet[facet_faces] = facet_id
    # A model-first result must explain every triangle.  Otherwise use the
    # general curvature graph rather than inventing identities for leftovers.
    if np.any(face_to_facet < 0):
        return None

    facet_areas = np.asarray(
        [float(mesh.area_faces[facet].sum()) for facet in facets],
        dtype=np.float64,
    )
    positive_order = np.argsort(facet_areas)
    ordered_areas = facet_areas[positive_order]
    if len(ordered_areas) < 4 or np.any(ordered_areas <= 0.0):
        return None
    area_ratios = ordered_areas[1:] / ordered_areas[:-1]
    gap_index = int(np.argmax(area_ratios))
    if float(area_ratios[gap_index]) < 4.0:
        return None
    area_threshold = float(
        np.sqrt(ordered_areas[gap_index] * ordered_areas[gap_index + 1])
    )
    narrow_facets = facet_areas < area_threshold
    if int(np.count_nonzero(narrow_facets)) < int(minimum_cylinder_facets):
        return None

    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    facet_pairs = np.sort(face_to_facet[adjacency], axis=1)
    facet_pairs = np.unique(facet_pairs[facet_pairs[:, 0] != facet_pairs[:, 1]], axis=0)
    usable_pairs = facet_pairs[
        narrow_facets[facet_pairs[:, 0]] & narrow_facets[facet_pairs[:, 1]]
    ]
    graph = coo_matrix(
        (
            np.ones(len(usable_pairs) * 2, dtype=np.uint8),
            (
                np.concatenate((usable_pairs[:, 0], usable_pairs[:, 1])),
                np.concatenate((usable_pairs[:, 1], usable_pairs[:, 0])),
            ),
        ),
        shape=(len(facets), len(facets)),
    ).tocsr()
    _, facet_components = connected_components(graph, directed=False)

    cylinder_groups = []
    consumed_facets = np.zeros(len(facets), dtype=bool)
    for component_id in np.unique(facet_components[narrow_facets]):
        facet_ids = np.flatnonzero(
            narrow_facets & (facet_components == int(component_id))
        )
        if len(facet_ids) < int(minimum_cylinder_facets):
            continue
        component_faces = np.concatenate([facets[index] for index in facet_ids])
        fit = _fit_cylindrical_face_set(vertices, faces[component_faces])
        if fit is None:
            continue
        cylinder_groups.append((component_faces, fit))
        consumed_facets[facet_ids] = True

    # The area split is accepted only if every narrow tessellation strip is
    # explained by a cylinder.  This prevents a coincidental area gap from
    # turning small genuine planes into curved surfaces.
    if not cylinder_groups or np.any(narrow_facets & ~consumed_facets):
        return None

    labels = np.full(len(faces), -1, dtype=np.int64)
    surface_types = []
    for facet_id, facet_faces in enumerate(facets):
        if consumed_facets[facet_id]:
            continue
        labels[facet_faces] = len(surface_types)
        surface_types.append("plane")
    cylinder_models = []
    for component_faces, fit in cylinder_groups:
        labels[component_faces] = len(surface_types)
        surface_types.append("cylinder")
        cylinder_models.append(fit)
    if np.any(labels < 0):
        return None

    sizes = np.bincount(labels)
    return labels, {
        "patches": len(sizes),
        "smallest_patch_faces": int(sizes.min()),
        "median_patch_faces": float(np.median(sizes)),
        "largest_patch_faces": int(sizes.max()),
        "plane_patches": int(surface_types.count("plane")),
        "cylinder_patches": int(surface_types.count("cylinder")),
        "surface_types": surface_types,
        "cylinder_models": cylinder_models,
        "facet_area_threshold": area_threshold,
        "partition_method": "brep-model-first",
        "global_consistency_rejections": 0,
        "transition_patches": 0,
        "forced_two_face_assignments": 0,
        "forced_ownership_assignments": 0,
        "ownership_faces": 0,
        "curvature_floor": 0.0,
        "elapsed": 0.0,
    }


def _adjacency_weights(
    mesh,
    face_curvatures,
    anisotropy,
    face_directions,
    feature_angle_degrees,
):
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    neighbor_slots = np.repeat(
        np.arange(len(face_curvatures), dtype=np.int64)[:, None], 4, axis=1
    )
    degrees = np.zeros(len(face_curvatures), dtype=np.int64)
    for first_face, second_face in adjacency:
        first_face = int(first_face)
        second_face = int(second_face)
        if degrees[first_face] < 3:
            degrees[first_face] += 1
            neighbor_slots[first_face, degrees[first_face]] = second_face
        if degrees[second_face] < 3:
            degrees[second_face] += 1
            neighbor_slots[second_face, degrees[second_face]] = first_face
    for _ in range(2):
        face_curvatures = np.median(
            face_curvatures[neighbor_slots], axis=1
        )
    positive = face_curvatures[:, 1][face_curvatures[:, 1] > 0.0]
    curvature_floor = (
        float(np.percentile(positive, 5.0)) if len(positive) else 1e-12
    )
    curvature_floor = max(curvature_floor, np.finfo(np.float64).eps)
    first = adjacency[:, 0]
    second = adjacency[:, 1]
    log_first = np.log(face_curvatures[first] + curvature_floor)
    log_second = np.log(face_curvatures[second] + curvature_floor)
    curvature_change = np.linalg.norm(log_first - log_second, axis=1)
    curvature_change = np.minimum(curvature_change, 3.0)

    direction_change = 1.0 - np.abs(
        np.einsum(
            "ij,ij->i", face_directions[first], face_directions[second]
        )
    )
    direction_reliability = anisotropy * (
        face_curvatures[:, 1]
        / (face_curvatures[:, 1] + 4.0 * curvature_floor)
    )
    direction_change *= np.minimum(
        direction_reliability[first], direction_reliability[second]
    )
    angle_degrees = np.degrees(
        np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    )
    angle_change = angle_degrees / max(float(feature_angle_degrees), 1e-12)

    weights = np.sqrt(
        (0.7 * curvature_change) ** 2
        + (0.8 * direction_change) ** 2
        + angle_change ** 2
    )
    fallback_weights = weights.copy()
    # A visible crease is a topological partition constraint, not merely a
    # large soft cost.  Otherwise the high initial threshold of an adaptive
    # graph segmenter can absorb a small planar fan across a sharp junction.
    hard_crease = angle_degrees >= float(feature_angle_degrees)
    coherent_curvature_boundary = _coherent_edge_chains(
        mesh, curvature_change >= 1.25, minimum_edges=4
    )
    weights[hard_crease | coherent_curvature_boundary] = np.inf
    return (
        adjacency,
        weights,
        fallback_weights,
        np.log(face_curvatures + curvature_floor),
        direction_reliability,
    )


def build_connected_partitions(
    vertices,
    faces,
    segmentation_scale=10.0,
    minimum_patch_faces=50,
    feature_angle_degrees=15.0,
    small_patch_merge_limit=2.5,
    maximum_global_curvature_span=1.0,
    maximum_global_direction_change=0.35,
    prefer_brep_models=True,
):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Partitioning requires a nonempty triangle mesh.")
    started = time.perf_counter()
    if prefer_brep_models:
        model_result = build_brep_model_partitions(vertices, faces)
        if model_result is not None:
            model_labels, model_stats = model_result
            model_stats["elapsed"] = time.perf_counter() - started
            return model_labels, model_stats
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    curvatures, anisotropy, directions = _face_curvature_descriptors(
        vertices, faces
    )
    (
        adjacency,
        weights,
        fallback_weights,
        curvature_descriptors,
        direction_reliability,
    ) = _adjacency_weights(
        mesh,
        curvatures,
        anisotropy,
        directions,
        feature_angle_degrees,
    )
    triangles = vertices[faces]
    triangle_edges = np.linalg.norm(
        triangles[:, (1, 2, 0)] - triangles[:, (0, 1, 2)], axis=2
    )
    doubled_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    triangle_quality = (
        2.0 * np.sqrt(3.0) * doubled_areas
        / np.maximum(
            np.einsum("ij,ij->i", triangle_edges, triangle_edges),
            np.finfo(np.float64).eps,
        )
    )
    ownership_faces = triangle_quality <= 0.05
    mixed_ownership_edges = (
        ownership_faces[adjacency[:, 0]]
        != ownership_faces[adjacency[:, 1]]
    )
    weights[mixed_ownership_edges] = np.inf
    order = np.argsort(weights, kind="stable")
    components = _DisjointSet(
        len(faces),
        curvature_descriptors=curvature_descriptors,
        directions=directions,
        direction_reliability=direction_reliability,
    )
    scale = float(segmentation_scale)
    global_rejections = 0

    for edge_id in order:
        first, second = adjacency[edge_id]
        first_root = components.find(first)
        second_root = components.find(second)
        if first_root == second_root:
            continue
        first_threshold = (
            components.internal[first_root]
            + scale / components.size[first_root]
        )
        second_threshold = (
            components.internal[second_root]
            + scale / components.size[second_root]
        )
        if weights[edge_id] <= min(first_threshold, second_threshold):
            if components.globally_compatible(
                first_root,
                second_root,
                maximum_curvature_span=maximum_global_curvature_span,
                maximum_direction_change=maximum_global_direction_change,
            ):
                components.merge(first_root, second_root, weights[edge_id])
            else:
                global_rejections += 1

    # Small groups dominated by extremely poor triangles are ownership units,
    # not independent surfaces and not bridges between regular regions.  Give
    # each unit to one adjacent regular component using the unconstrained edge
    # cost, while leaving large low-quality sheets available as real patches.
    current_roots = np.fromiter(
        (components.find(face) for face in range(len(faces))),
        dtype=np.int64,
        count=len(faces),
    )
    low_quality_counts = {}
    for face_id, root in enumerate(current_roots):
        if ownership_faces[face_id]:
            low_quality_counts[int(root)] = low_quality_counts.get(int(root), 0) + 1
    ownership_roots = {
        root
        for root, low_count in low_quality_counts.items()
        if components.size[root] <= 32
        and low_count / components.size[root] >= 0.8
    }
    ownership_costs = {}
    for edge_id, (first, second) in enumerate(adjacency):
        first_root = components.find(first)
        second_root = components.find(second)
        if first_root == second_root:
            continue
        for source, target in (
            (first_root, second_root),
            (second_root, first_root),
        ):
            if source not in ownership_roots or target in ownership_roots:
                continue
            key = (source, target)
            total, count = ownership_costs.get(key, (0.0, 0))
            ownership_costs[key] = (
                total + float(fallback_weights[edge_id]), count + 1
            )
    ownership_targets = {}
    for (source, target), (total, count) in ownership_costs.items():
        score = total / count - 0.05 * np.log1p(count)
        previous = ownership_targets.get(source)
        if previous is None or score < previous[0]:
            ownership_targets[source] = (score, target)
    forced_ownership_assignments = 0
    for source, (score, target) in sorted(
        ownership_targets.items(), key=lambda item: item[1][0]
    ):
        source_root = components.find(source)
        target_root = components.find(target)
        if source_root == target_root or source_root not in ownership_roots:
            continue
        components.merge(source_root, target_root, score)
        forced_ownership_assignments += 1

    minimum_patch_faces = max(int(minimum_patch_faces), 1)
    initial_roots = np.fromiter(
        (components.find(face) for face in range(len(faces))),
        dtype=np.int64,
        count=len(faces),
    )
    root_neighbors = {}
    for first, second in adjacency:
        first_root = int(initial_roots[int(first)])
        second_root = int(initial_roots[int(second)])
        if first_root == second_root:
            continue
        first_neighbors = root_neighbors.setdefault(first_root, {})
        second_neighbors = root_neighbors.setdefault(second_root, {})
        first_neighbors[second_root] = first_neighbors.get(second_root, 0) + 1
        second_neighbors[first_root] = second_neighbors.get(first_root, 0) + 1
    transition_roots = {
        root
        for root, neighbors in root_neighbors.items()
        if 4 <= components.size[root] < minimum_patch_faces
        and sum(
            components.size[neighbor] >= minimum_patch_faces
            and shared_edges >= 2
            for neighbor, shared_edges in neighbors.items()
        ) >= 2
    }
    for edge_id in order:
        if weights[edge_id] > float(small_patch_merge_limit):
            break
        first, second = adjacency[edge_id]
        first_root = components.find(first)
        second_root = components.find(second)
        if first_root == second_root:
            continue
        if (
            first_root in transition_roots
            or second_root in transition_roots
        ):
            continue
        if (
            components.size[first_root] < minimum_patch_faces
            or components.size[second_root] < minimum_patch_faces
        ):
            if components.globally_compatible(
                first_root,
                second_root,
                maximum_curvature_span=maximum_global_curvature_span,
                maximum_direction_change=maximum_global_direction_change,
            ):
                components.merge(first_root, second_root, weights[edge_id])
            else:
                global_rejections += 1

    # A single triangle cannot carry a reliable surface identity.  After all
    # region decisions are complete, give it the geometrically closest
    # edge-neighbour without allowing that ownership choice to bridge two
    # already-established regions.
    fallback_order = np.argsort(fallback_weights, kind="stable")
    for edge_id in fallback_order:
        first, second = adjacency[edge_id]
        first_root = components.find(first)
        second_root = components.find(second)
        if first_root == second_root:
            continue
        if components.size[first_root] == 1 or components.size[second_root] == 1:
            components.merge(
                first_root, second_root, fallback_weights[edge_id]
            )

    # A highly elongated connected two-triangle sliver is too small and thin
    # to represent an independent B-rep surface.  Keep its two faces together
    # and assign the unit to exactly one edge-neighbour, choosing the side with
    # the lowest mean unconstrained geometry cost.  Compact two-triangle faces
    # (for example a valid rectangular B-rep face) and truly isolated source
    # components are intentionally retained.
    two_face_roots = {}
    for face_id in range(len(faces)):
        root = components.find(face_id)
        if components.size[root] == 2:
            two_face_roots.setdefault(root, []).append(face_id)
    sliver_roots = set()
    for root, face_ids in two_face_roots.items():
        points = vertices[np.unique(faces[face_ids])]
        singular_values = np.linalg.svd(
            points - points.mean(axis=0), compute_uv=False
        )
        if (
            len(singular_values) >= 2
            and singular_values[0]
            / max(singular_values[1], np.finfo(np.float64).eps)
            >= 4.0
        ):
            sliver_roots.add(root)

    pair_costs = {}
    for edge_id, (first, second) in enumerate(adjacency):
        first_root = components.find(first)
        second_root = components.find(second)
        if first_root == second_root:
            continue
        cost = float(fallback_weights[edge_id])
        for source, target in (
            (first_root, second_root),
            (second_root, first_root),
        ):
            if components.size[source] != 2 or source not in sliver_roots:
                continue
            key = (source, target)
            total, count = pair_costs.get(key, (0.0, 0))
            pair_costs[key] = (total + cost, count + 1)
    best_targets = {}
    for (source, target), (total, count) in pair_costs.items():
        score = total / count - 0.05 * np.log1p(count)
        previous = best_targets.get(source)
        if previous is None or score < previous[0]:
            best_targets[source] = (score, target)
    forced_two_face_assignments = 0
    for source, (score, target) in sorted(
        best_targets.items(), key=lambda item: item[1][0]
    ):
        source_root = components.find(source)
        target_root = components.find(target)
        if source_root == target_root or components.size[source_root] != 2:
            continue
        components.merge(source_root, target_root, score)
        forced_two_face_assignments += 1

    roots = np.fromiter(
        (components.find(face) for face in range(len(faces))),
        dtype=np.int64,
        count=len(faces),
    )
    _, labels = np.unique(roots, return_inverse=True)
    sizes = np.bincount(labels)
    positive_curvatures = curvatures[:, 1][curvatures[:, 1] > 0.0]
    return labels, {
        "patches": len(sizes),
        "smallest_patch_faces": int(sizes.min()),
        "median_patch_faces": float(np.median(sizes)),
        "largest_patch_faces": int(sizes.max()),
        "global_consistency_rejections": int(global_rejections),
        "transition_patches": int(len(transition_roots)),
        "forced_two_face_assignments": int(forced_two_face_assignments),
        "forced_ownership_assignments": int(forced_ownership_assignments),
        "ownership_faces": int(np.count_nonzero(ownership_faces)),
        "curvature_floor": (
            float(np.percentile(positive_curvatures, 5.0))
            if len(positive_curvatures)
            else 0.0
        ),
        "elapsed": time.perf_counter() - started,
    }


def partition_stl(
    input_path,
    output_directory,
    segmentation_scale=10.0,
    minimum_patch_faces=50,
    feature_angle_degrees=15.0,
    small_patch_merge_limit=2.5,
    maximum_global_curvature_span=1.0,
    maximum_global_direction_change=0.35,
    prefer_brep_models=True,
):
    input_path = Path(input_path).resolve()
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load_mesh(input_path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Input is not one triangle mesh: {input_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    labels, stats = build_connected_partitions(
        vertices,
        faces,
        segmentation_scale=segmentation_scale,
        minimum_patch_faces=minimum_patch_faces,
        feature_angle_degrees=feature_angle_degrees,
        small_patch_merge_limit=small_patch_merge_limit,
        maximum_global_curvature_span=maximum_global_curvature_span,
        maximum_global_direction_change=maximum_global_direction_change,
        prefer_brep_models=prefer_brep_models,
    )
    patch_count = stats["patches"]
    combined_path = output_directory / "_all_patches.ply"
    _write_patch_id_ply(combined_path, vertices, faces, labels)

    sizes = np.bincount(labels, minlength=patch_count)
    for stale_patch in output_directory.glob("patch_*_faces.ply"):
        stale_patch.unlink()
    surface_types = stats.get("surface_types", ["unknown"] * patch_count)
    manifest_path = output_directory / "patch_ids.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("patch_id", "surface_type", "face_count", "file"))
        for patch in range(patch_count):
            filename = f"patch_{patch:04d}_{int(sizes[patch])}_faces.ply"
            writer.writerow(
                (patch, surface_types[patch], int(sizes[patch]), filename)
            )
    for patch in range(patch_count):
        patch_faces = faces[labels == patch]
        patch_path = (
            output_directory
            / f"patch_{patch:04d}_{int(sizes[patch])}_faces.ply"
        )
        _write_patch_id_ply(
            patch_path,
            vertices,
            patch_faces,
            np.full(len(patch_faces), patch, dtype=np.int64),
            patch_ids=[patch],
        )
    stats["combined_path"] = combined_path
    stats["manifest_path"] = manifest_path
    stats["output_directory"] = output_directory
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Partition an STL into connected curvature-coherent PLY patches."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segmentation-scale", type=float, default=10.0)
    parser.add_argument("--minimum-patch-faces", type=int, default=50)
    parser.add_argument("--feature-angle", type=float, default=15.0)
    parser.add_argument("--small-patch-merge-limit", type=float, default=2.5)
    parser.add_argument(
        "--maximum-global-curvature-span", type=float, default=1.0
    )
    parser.add_argument(
        "--maximum-global-direction-change", type=float, default=0.35
    )
    parser.add_argument(
        "--disable-brep-models",
        action="store_true",
        help="Skip exact plane/cylinder reconstruction and use only graph segmentation",
    )
    args = parser.parse_args()
    stats = partition_stl(
        args.input,
        args.output_dir,
        segmentation_scale=args.segmentation_scale,
        minimum_patch_faces=args.minimum_patch_faces,
        feature_angle_degrees=args.feature_angle,
        small_patch_merge_limit=args.small_patch_merge_limit,
        maximum_global_curvature_span=args.maximum_global_curvature_span,
        maximum_global_direction_change=args.maximum_global_direction_change,
        prefer_brep_models=not args.disable_brep_models,
    )
    if stats.get("partition_method") == "brep-model-first":
        print(
            "Recovered {plane_patches} planar and {cylinder_patches} "
            "cylindrical/fillet B-rep surface(s).".format(**stats)
        )
    print(
        "Partitioned into {patches} connected patch(es); face counts min "
        "{smallest_patch_faces}, median {median_patch_faces:.1f}, max "
        "{largest_patch_faces}; {global_consistency_rejections} global merge "
        "rejection(s), {transition_patches} transition patch(es), "
        "{forced_ownership_assignments} low-quality ownership assignment(s), "
        "{forced_two_face_assignments} two-face assignment(s); time "
        "{elapsed:.3f}s.".format(**stats)
    )
    print(f"Combined patch-ID PLY: {stats['combined_path']}")
    print(f"Patch ID manifest: {stats['manifest_path']}")
    print(f"Individual patch directory: {stats['output_directory']}")


if __name__ == "__main__":
    main()
