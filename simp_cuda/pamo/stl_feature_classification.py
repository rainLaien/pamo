"""Classify connected surface features in a triangulated STL mesh.

STL stores triangles rather than CAD surface definitions.  The labels returned
by this module are therefore geometric inferences, not recovered B-Rep data.
Surface geometry and manufacturing meaning are deliberately kept separate:
a cylindrical patch can, for example, be a main wall or a fillet.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class SurfaceType(str, Enum):
    """Mutually exclusive geometric type of a connected triangle patch."""

    PLANE = "plane"
    CYLINDER = "cylinder"
    CONE = "cone"
    SPHERE = "sphere"
    TORUS = "torus"
    FREEFORM = "freeform"
    UNKNOWN = "unknown"


class TransitionType(str, Enum):
    """Optional engineering role of a surface patch."""

    NONE = "none"
    FILLET = "fillet"
    CHAMFER = "chamfer"


@dataclass(frozen=True)
class ClassificationConfig:
    """Scale-independent tolerances used by :func:`classify_mesh_features`."""

    crease_angle_degrees: float = 35.0
    plane_normal_angle_degrees: float = 2.0
    plane_distance_tolerance: float = 2e-3
    cylinder_radius_tolerance: float = 2e-2
    cylinder_normal_tolerance: float = 6e-2
    cone_radius_tolerance: float = 3e-2
    cone_normal_tolerance: float = 8e-2
    sphere_radius_tolerance: float = 2e-2
    torus_radius_tolerance: float = 3e-2
    torus_normal_tolerance: float = 1e-1
    minimum_planar_region_faces: int = 4
    # Retained for source compatibility. Plane acceptance no longer depends on
    # a fraction of the global mesh size.
    minimum_embedded_plane_span_ratio: float = 0.0
    fillet_maximum_coverage_degrees: float = 200.0
    fillet_maximum_radius_ratio: float = 0.2
    chamfer_maximum_width_ratio: float = 0.1
    chamfer_maximum_aspect_ratio: float = 0.25


@dataclass
class SurfacePatch:
    """One connected classified region of the input mesh."""

    patch_id: int
    face_indices: np.ndarray
    surface_type: SurfaceType
    confidence: float
    fit_error: float
    parameters: Dict[str, object] = field(default_factory=dict)
    transition_type: TransitionType = TransitionType.NONE
    neighbor_patch_ids: Tuple[int, ...] = ()

    @property
    def is_curved(self):
        return self.surface_type in {
            SurfaceType.CYLINDER,
            SurfaceType.CONE,
            SurfaceType.SPHERE,
            SurfaceType.TORUS,
            SurfaceType.FREEFORM,
        }


@dataclass(frozen=True)
class MeshFeatureClassification:
    """Patch labels and a face-to-patch lookup for the whole mesh."""

    patches: Tuple[SurfacePatch, ...]
    face_patch_ids: np.ndarray
    mesh_scale: float

    def patches_of_type(self, surface_type):
        surface_type = SurfaceType(surface_type)
        return tuple(
            patch for patch in self.patches
            if patch.surface_type == surface_type
        )


@dataclass(frozen=True)
class _SurfaceFit:
    surface_type: SurfaceType
    normalized_error: float
    fit_error: float
    parameters: Dict[str, object]


class _DisjointSet:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

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

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def _validate_config(config):
    if not 0.0 < config.crease_angle_degrees < 180.0:
        raise ValueError("Crease angle must be between 0 and 180 degrees.")
    if not 0.0 < config.plane_normal_angle_degrees < 90.0:
        raise ValueError("Plane normal angle must be between 0 and 90 degrees.")
    for name in (
        "plane_distance_tolerance",
        "cylinder_radius_tolerance",
        "cylinder_normal_tolerance",
        "cone_radius_tolerance",
        "cone_normal_tolerance",
        "sphere_radius_tolerance",
        "torus_radius_tolerance",
        "torus_normal_tolerance",
        "fillet_maximum_radius_ratio",
        "chamfer_maximum_width_ratio",
        "chamfer_maximum_aspect_ratio",
    ):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError("{} must be positive.".format(name))
    if config.minimum_planar_region_faces < 2:
        raise ValueError("minimum_planar_region_faces must be at least 2.")
    if float(config.minimum_embedded_plane_span_ratio) < 0.0:
        raise ValueError("minimum_embedded_plane_span_ratio cannot be negative.")
    if not 0.0 < config.fillet_maximum_coverage_degrees <= 360.0:
        raise ValueError("Fillet coverage must be in (0, 360] degrees.")


def _validate_mesh(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Vertices must have shape (n, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Faces must have shape (m, 3).")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Feature classification needs a non-empty mesh.")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices contain non-finite coordinates.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Face indices are outside the vertex array.")

    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_areas = np.linalg.norm(crosses, axis=1)
    if np.any(double_areas <= np.finfo(np.float64).eps):
        raise ValueError("Mesh contains degenerate triangles.")
    normals = crosses / double_areas[:, None]
    return vertices, faces, normals, 0.5 * double_areas


def _edge_memberships(faces):
    memberships = {}
    for face_index, face in enumerate(faces):
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            memberships.setdefault(edge, []).append(face_index)
    return memberships


def _adjacent_face_pairs(edge_memberships):
    pairs = []
    for memberships in edge_memberships.values():
        if len(memberships) == 2:
            pairs.append((int(memberships[0]), int(memberships[1])))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _connected_components(face_indices, adjacency_pairs, face_count):
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if len(face_indices) == 0:
        return []
    selected = np.zeros(face_count, dtype=bool)
    selected[face_indices] = True
    disjoint_set = _DisjointSet(face_count)
    for first, second in adjacency_pairs:
        if selected[first] and selected[second]:
            disjoint_set.union(first, second)
    grouped = {}
    for face_index in face_indices:
        grouped.setdefault(disjoint_set.find(face_index), []).append(face_index)
    return [
        np.asarray(indices, dtype=np.int64)
        for _, indices in sorted(
            grouped.items(), key=lambda item: min(item[1])
        )
    ]


def _unique_patch_points(vertices, faces, face_indices):
    vertex_ids = np.unique(faces[np.asarray(face_indices)].reshape(-1))
    return vertices[vertex_ids]


def _orthonormal_basis(axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    helper = np.zeros(3, dtype=np.float64)
    helper[int(np.argmin(np.abs(axis)))] = 1.0
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second


def _angular_coverage_degrees(coordinates):
    if len(coordinates) < 2:
        return 0.0
    angles = np.sort(
        np.mod(np.arctan2(coordinates[:, 1], coordinates[:, 0]), 2.0 * np.pi)
    )
    gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * np.pi)))
    return float(np.degrees(2.0 * np.pi - gaps.max()))


def _fit_plane(points, tolerance, tolerance_scale=None):
    origin = points.mean(axis=0)
    _, singular_values, basis_vh = np.linalg.svd(
        points - origin, full_matrices=False
    )
    projected_points = (points - origin) @ basis_vh[:2].T
    in_plane_spans = np.ptp(projected_points, axis=0)
    plane_scale = max(float(in_plane_spans.min()), 1e-30)
    distances = np.abs((points - origin) @ basis_vh[-1])
    relative_error = float(distances.max() / plane_scale)
    if tolerance_scale is None:
        tolerance_scale = plane_scale
    absolute_tolerance = max(float(tolerance) * tolerance_scale, 1e-30)
    normalized_error = float(distances.max() / absolute_tolerance)
    if normalized_error > 1.0:
        return None
    return _SurfaceFit(
        SurfaceType.PLANE,
        normalized_error,
        relative_error,
        {
            "origin": origin.tolist(),
            "normal": basis_vh[-1].tolist(),
            "in_plane_spans": sorted(
                (float(value) for value in in_plane_spans), reverse=True
            ),
            "singular_values": singular_values.tolist(),
        },
    )


def _fit_circle_2d(coordinates):
    system = np.column_stack(
        (
            2.0 * coordinates[:, 0],
            2.0 * coordinates[:, 1],
            np.ones(len(coordinates)),
        )
    )
    right_hand_side = np.sum(coordinates * coordinates, axis=1)
    solution, _, _, _ = np.linalg.lstsq(
        system, right_hand_side, rcond=None
    )
    center = solution[:2]
    radii = np.linalg.norm(coordinates - center, axis=1)
    radius = float(radii.mean())
    if radius <= 1e-30:
        return None
    return center, radii, radius


def _fit_cylinder(points, normals, radius_tolerance, normal_tolerance):
    normal_covariance = normals.T @ normals / max(len(normals), 1)
    _, eigenvectors = np.linalg.eigh(normal_covariance)
    axis = eigenvectors[:, 0]
    axial_normal_error = float(np.sqrt(np.mean((normals @ axis) ** 2)))
    if axial_normal_error > normal_tolerance:
        return None
    origin = points.mean(axis=0)
    first, second = _orthonormal_basis(axis)
    projected = np.column_stack(
        ((points - origin) @ first, (points - origin) @ second)
    )
    circle = _fit_circle_2d(projected)
    if circle is None:
        return None
    center_2d, radii, radius = circle
    radial_error = float(np.sqrt(np.mean((radii - radius) ** 2)) / radius)
    if radial_error > radius_tolerance:
        return None
    axis_origin = origin + center_2d[0] * first + center_2d[1] * second
    centered_coordinates = projected - center_2d
    coverage = _angular_coverage_degrees(centered_coordinates)
    normalized_error = max(
        radial_error / radius_tolerance,
        axial_normal_error / normal_tolerance,
    )
    return _SurfaceFit(
        SurfaceType.CYLINDER,
        normalized_error,
        max(radial_error, axial_normal_error),
        {
            "axis_origin": axis_origin.tolist(),
            "axis": axis.tolist(),
            "radius": radius,
            "angular_coverage_degrees": coverage,
            "radial_error": radial_error,
            "axial_normal_error": axial_normal_error,
        },
    )


def _fit_sphere(points, tolerance):
    system = np.column_stack((2.0 * points, np.ones(len(points))))
    right_hand_side = np.sum(points * points, axis=1)
    solution, _, _, _ = np.linalg.lstsq(
        system, right_hand_side, rcond=None
    )
    center = solution[:3]
    radii = np.linalg.norm(points - center, axis=1)
    radius = float(radii.mean())
    if radius <= 1e-30:
        return None
    relative_error = float(np.sqrt(np.mean((radii - radius) ** 2)) / radius)
    if relative_error > tolerance:
        return None
    return _SurfaceFit(
        SurfaceType.SPHERE,
        relative_error / tolerance,
        relative_error,
        {
            "center": center.tolist(),
            "radius": radius,
            "radial_error": relative_error,
        },
    )


def _axis_candidates(points, normals):
    candidates = []
    for covariance in (
        normals.T @ normals / max(len(normals), 1),
        np.cov((points - points.mean(axis=0)).T),
    ):
        _, eigenvectors = np.linalg.eigh(covariance)
        for column in range(3):
            candidate = eigenvectors[:, column]
            if not any(abs(float(candidate @ old)) > 1.0 - 1e-6 for old in candidates):
                candidates.append(candidate)
    return candidates


def _fit_cone(points, normals, radius_tolerance, normal_tolerance):
    best = None
    origin = points.mean(axis=0)
    for axis in _axis_candidates(points, normals):
        first, second = _orthonormal_basis(axis)
        projected = np.column_stack(
            ((points - origin) @ first, (points - origin) @ second)
        )
        circle = _fit_circle_2d(projected)
        if circle is None:
            continue
        center_2d = circle[0]
        radial = np.linalg.norm(projected - center_2d, axis=1)
        axis_origin = origin + center_2d[0] * first + center_2d[1] * second
        axial = (points - axis_origin) @ axis
        system = np.column_stack((axial, np.ones(len(axial))))
        slope_intercept, _, _, _ = np.linalg.lstsq(system, radial, rcond=None)
        slope, intercept = map(float, slope_intercept)
        if abs(slope) <= 2e-2:
            continue
        fitted_radius = system @ slope_intercept
        mean_radius = max(float(radial.mean()), 1e-30)
        radial_error = float(
            np.sqrt(np.mean((radial - fitted_radius) ** 2)) / mean_radius
        )
        axial_components = np.abs(normals @ axis)
        normal_error = float(np.std(axial_components))
        if radial_error > radius_tolerance or normal_error > normal_tolerance:
            continue
        normalized_error = max(
            radial_error / radius_tolerance,
            normal_error / normal_tolerance,
        )
        apex = axis_origin - axis * intercept / slope
        fit = _SurfaceFit(
            SurfaceType.CONE,
            normalized_error,
            max(radial_error, normal_error),
            {
                "apex": apex.tolist(),
                "axis": axis.tolist(),
                "half_angle_degrees": float(np.degrees(np.arctan(abs(slope)))),
                "radial_error": radial_error,
                "normal_error": normal_error,
                "angular_coverage_degrees": _angular_coverage_degrees(
                    projected - center_2d
                ),
            },
        )
        if best is None or fit.normalized_error < best.normalized_error:
            best = fit
    return best


def _fit_torus(points, sample_points, normals, tolerance, normal_tolerance):
    best = None
    origin = points.mean(axis=0)
    for axis in _axis_candidates(points, normals):
        axial = (points - origin) @ axis
        radial_vectors = points - origin - axial[:, None] * axis
        radial = np.linalg.norm(radial_vectors, axis=1)
        profile = np.column_stack((radial, axial))
        circle = _fit_circle_2d(profile)
        if circle is None:
            continue
        profile_center, profile_radii, minor_radius = circle
        major_radius = float(profile_center[0])
        if major_radius <= minor_radius * 1.05:
            continue
        relative_error = float(
            np.sqrt(np.mean((profile_radii - minor_radius) ** 2))
            / minor_radius
        )
        if relative_error > tolerance:
            continue
        center = origin + float(profile_center[1]) * axis
        sample_offsets = sample_points - center
        sample_axial = sample_offsets @ axis
        sample_radial = sample_offsets - sample_axial[:, None] * axis
        sample_radial_lengths = np.linalg.norm(sample_radial, axis=1)
        if np.any(sample_radial_lengths <= 1e-30):
            continue
        centerline_points = (
            center
            + major_radius
            * sample_radial
            / sample_radial_lengths[:, None]
        )
        expected_normals = sample_points - centerline_points
        expected_lengths = np.linalg.norm(expected_normals, axis=1)
        if np.any(expected_lengths <= 1e-30):
            continue
        expected_normals /= expected_lengths[:, None]
        alignments = np.abs(
            np.einsum("ij,ij->i", expected_normals, normals)
        )
        normal_error = float(
            np.sqrt(np.mean((1.0 - np.clip(alignments, 0.0, 1.0)) ** 2))
        )
        if normal_error > normal_tolerance:
            continue
        normalized_error = max(
            relative_error / tolerance,
            normal_error / normal_tolerance,
        )
        fit = _SurfaceFit(
            SurfaceType.TORUS,
            normalized_error,
            max(relative_error, normal_error),
            {
                "center": center.tolist(),
                "axis": axis.tolist(),
                "major_radius": major_radius,
                "minor_radius": minor_radius,
                "profile_error": relative_error,
                "normal_error": normal_error,
            },
        )
        if best is None or fit.normalized_error < best.normalized_error:
            best = fit
    return best


def _classify_surface(points, sample_points, normals, config):
    fits = [
        _fit_plane(points, config.plane_distance_tolerance),
        _fit_cylinder(
            points,
            normals,
            config.cylinder_radius_tolerance,
            config.cylinder_normal_tolerance,
        ),
        _fit_cone(
            points,
            normals,
            config.cone_radius_tolerance,
            config.cone_normal_tolerance,
        ),
        _fit_sphere(points, config.sphere_radius_tolerance),
        _fit_torus(
            points,
            sample_points,
            normals,
            config.torus_radius_tolerance,
            config.torus_normal_tolerance,
        ),
    ]
    fits = [fit for fit in fits if fit is not None]
    if not fits:
        return _SurfaceFit(SurfaceType.FREEFORM, 1.0, float("nan"), {})
    return min(fits, key=lambda fit: fit.normalized_error)


def _make_patch(face_indices, vertices, faces, normals, config):
    points = _unique_patch_points(vertices, faces, face_indices)
    sample_points = vertices[faces[face_indices]].mean(axis=1)
    fit = _classify_surface(
        points, sample_points, normals[face_indices], config
    )
    confidence = max(0.0, min(1.0, 1.0 - fit.normalized_error))
    return SurfacePatch(
        patch_id=-1,
        face_indices=np.asarray(face_indices, dtype=np.int64),
        surface_type=fit.surface_type,
        confidence=confidence,
        fit_error=fit.fit_error,
        parameters=fit.parameters,
    )


def _make_plane_patch(face_indices, vertices, faces, config):
    points = _unique_patch_points(vertices, faces, face_indices)
    fit = _fit_plane(points, config.plane_distance_tolerance)
    if fit is None:
        return None
    confidence = max(0.0, min(1.0, 1.0 - fit.normalized_error))
    return SurfacePatch(
        patch_id=-1,
        face_indices=np.asarray(face_indices, dtype=np.int64),
        surface_type=SurfaceType.PLANE,
        confidence=confidence,
        fit_error=fit.fit_error,
        parameters=fit.parameters,
    )


def _face_neighbors(adjacency_pairs, face_count):
    neighbors = [[] for _ in range(face_count)]
    for first, second in adjacency_pairs:
        first = int(first)
        second = int(second)
        neighbors[first].append(second)
        neighbors[second].append(first)
    return neighbors


def _grow_fixed_plane_region(
    seed_face,
    component_mask,
    claimed_mask,
    face_neighbors,
    vertices,
    faces,
    normals,
    normal_cosine_limit,
    distance_tolerance,
):
    """Fit a one-ring seed, then grow without changing its reference plane."""
    reference_triangle = vertices[faces[seed_face]]
    initial_origin = reference_triangle.mean(axis=0)
    initial_normal = normals[seed_face]
    seed_faces = [int(seed_face)]
    for neighbor in face_neighbors[seed_face]:
        if claimed_mask[neighbor] or not component_mask[neighbor]:
            continue
        if float(normals[neighbor] @ initial_normal) < normal_cosine_limit:
            continue
        neighbor_points = vertices[faces[neighbor]]
        distances = np.abs((neighbor_points - initial_origin) @ initial_normal)
        if float(distances.max()) <= distance_tolerance:
            seed_faces.append(neighbor)

    seed_vertex_ids = np.unique(
        faces[np.asarray(seed_faces, dtype=np.int64)].reshape(-1)
    )
    seed_points = vertices[seed_vertex_ids]
    reference_origin = seed_points.mean(axis=0)
    _, _, basis_vh = np.linalg.svd(
        seed_points - reference_origin, full_matrices=False
    )
    reference_normal = basis_vh[-1]
    if float(reference_normal @ initial_normal) < 0.0:
        reference_normal = -reference_normal

    region = set(seed_faces)
    queue = list(seed_faces)
    queue_index = 0
    while queue_index < len(queue):
        face_index = queue[queue_index]
        queue_index += 1
        for neighbor in face_neighbors[face_index]:
            if (
                neighbor in region
                or claimed_mask[neighbor]
                or not component_mask[neighbor]
            ):
                continue
            if float(normals[neighbor] @ reference_normal) < normal_cosine_limit:
                continue
            neighbor_points = vertices[faces[neighbor]]
            distances = np.abs(
                (neighbor_points - reference_origin) @ reference_normal
            )
            if float(distances.max()) > distance_tolerance:
                continue
            region.add(neighbor)
            queue.append(neighbor)
    return np.asarray(sorted(region), dtype=np.int64)


def _has_two_dimensional_planar_support(face_indices, faces):
    """Reject one-cell-wide polygon strips masquerading as CAD planes."""
    edge_counts = {}
    for face_index in face_indices:
        face = faces[face_index]
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_vertices = {
        vertex
        for edge, count in edge_counts.items()
        if count == 1
        for vertex in edge
    }
    vertex_face_counts = {}
    for vertex in faces[np.asarray(face_indices, dtype=np.int64)].reshape(-1):
        vertex = int(vertex)
        vertex_face_counts[vertex] = vertex_face_counts.get(vertex, 0) + 1
    has_interior_vertex = bool(set(vertex_face_counts) - boundary_vertices)
    has_branched_triangulation = max(vertex_face_counts.values()) >= 4
    return has_interior_vertex or has_branched_triangulation


def _extract_planar_seeds(
    component,
    face_neighbors,
    vertices,
    faces,
    normals,
    config,
):
    """Extract planes by growing against a fixed seed plane.

    Local normal agreement is intentionally not used as a transitive
    equivalence relation: doing so lets finely tessellated fillets bridge a
    true plane into a curved component. Each growth keeps the seed triangle's
    normal and offset fixed and validates the completed region with a global
    plane fit.
    """
    face_count = len(faces)
    component = np.asarray(component, dtype=np.int64)
    component_mask = np.zeros(face_count, dtype=bool)
    component_mask[component] = True
    claimed_mask = np.zeros(face_count, dtype=bool)
    exhausted_seeds = np.zeros(face_count, dtype=bool)
    component_points = _unique_patch_points(vertices, faces, component)
    component_scale = max(
        float(np.ptp(component_points, axis=0).max()),
        np.finfo(np.float64).eps,
    )
    distance_tolerance = max(
        component_scale * config.plane_distance_tolerance,
        component_scale * np.finfo(np.float64).eps * 64.0,
    )
    normal_cosine_limit = np.cos(
        np.deg2rad(config.plane_normal_angle_degrees)
    )

    planar_patches = []
    for seed_face in component:
        if claimed_mask[seed_face] or exhausted_seeds[seed_face]:
            continue
        candidate = _grow_fixed_plane_region(
            int(seed_face),
            component_mask,
            claimed_mask,
            face_neighbors,
            vertices,
            faces,
            normals,
            normal_cosine_limit,
            distance_tolerance,
        )
        if len(candidate) < config.minimum_planar_region_faces:
            exhausted_seeds[seed_face] = True
            continue
        patch = _make_plane_patch(
            candidate,
            vertices,
            faces,
            config,
        )
        if patch is None:
            exhausted_seeds[seed_face] = True
            continue
        if not _has_two_dimensional_planar_support(candidate, faces):
            # All vertices lie on the candidate boundary. This is normally a
            # single polygon strip from a tessellated cylinder or cone.
            exhausted_seeds[candidate] = True
            continue
        planar_patches.append(patch)
        claimed_mask[candidate] = True

    remaining = component[~claimed_mask[component]]
    return planar_patches, remaining


def _set_patch_neighbors(patches, face_patch_ids, edge_memberships):
    neighbors = [set() for _ in patches]
    for memberships in edge_memberships.values():
        patch_ids = {
            int(face_patch_ids[face_index]) for face_index in memberships
        }
        for patch_id in patch_ids:
            neighbors[patch_id].update(patch_ids - {patch_id})
    for patch, patch_neighbors in zip(patches, neighbors):
        patch.neighbor_patch_ids = tuple(sorted(patch_neighbors))


def _distinct_plane_neighbors(patch, patches):
    normals = []
    for neighbor_id in patch.neighbor_patch_ids:
        neighbor = patches[neighbor_id]
        if neighbor.surface_type != SurfaceType.PLANE:
            continue
        normal = np.asarray(neighbor.parameters.get("normal"), dtype=np.float64)
        if normal.shape == (3,):
            normals.append(normal / np.linalg.norm(normal))
    if len(normals) < 2:
        return False
    cosine_limit = np.cos(np.deg2rad(15.0))
    return any(
        abs(float(first @ second)) < cosine_limit
        for index, first in enumerate(normals)
        for second in normals[index + 1:]
    )


def _classify_transition_roles(patches, mesh_scale, config):
    for patch in patches:
        if not _distinct_plane_neighbors(patch, patches):
            continue
        if patch.surface_type == SurfaceType.CYLINDER:
            coverage = float(
                patch.parameters.get("angular_coverage_degrees", 360.0)
            )
            radius = float(patch.parameters.get("radius", np.inf))
            if (
                coverage <= config.fillet_maximum_coverage_degrees
                and radius <= mesh_scale * config.fillet_maximum_radius_ratio
            ):
                patch.transition_type = TransitionType.FILLET
        elif patch.surface_type == SurfaceType.TORUS:
            radius = float(patch.parameters.get("minor_radius", np.inf))
            if radius <= mesh_scale * config.fillet_maximum_radius_ratio:
                patch.transition_type = TransitionType.FILLET
        elif patch.surface_type == SurfaceType.PLANE:
            spans = patch.parameters.get("in_plane_spans", ())
            if len(spans) == 2 and spans[0] > 0.0:
                aspect_ratio = float(spans[1]) / float(spans[0])
                if (
                    aspect_ratio <= config.chamfer_maximum_aspect_ratio
                    and float(spans[1])
                    <= mesh_scale * config.chamfer_maximum_width_ratio
                ):
                    patch.transition_type = TransitionType.CHAMFER


def classify_mesh_features(vertices, faces=None, config=None):
    """Classify the surface patches of a triangle mesh.

    ``vertices`` may be either an ``(n, 3)`` array or a ``trimesh.Trimesh``.
    Classification first separates patches at sharp edges, extracts planar
    seeds with fixed-reference-plane region growth, then fits cylinder, cone,
    sphere, and torus models to the remaining faces. Unmatched smooth patches
    are labelled ``FREEFORM``. Fillet and chamfer are subsequent heuristic
    transition labels because STL contains no native CAD feature semantics.
    """
    if faces is None and hasattr(vertices, "vertices") and hasattr(vertices, "faces"):
        mesh = vertices
        vertices = mesh.vertices
        faces = mesh.faces
    if faces is None:
        raise ValueError("Faces are required when vertices is not a mesh object.")
    if config is None:
        config = ClassificationConfig()
    _validate_config(config)
    vertices, faces, normals, _ = _validate_mesh(vertices, faces)

    mesh_scale = float(np.ptp(vertices, axis=0).max())
    if mesh_scale <= 0.0:
        raise ValueError("Mesh has zero spatial extent.")
    edge_memberships = _edge_memberships(faces)
    adjacency_pairs = _adjacent_face_pairs(edge_memberships)
    if len(adjacency_pairs):
        normal_cosines = np.einsum(
            "ij,ij->i",
            normals[adjacency_pairs[:, 0]],
            normals[adjacency_pairs[:, 1]],
        )
        normal_cosines = np.clip(normal_cosines, -1.0, 1.0)
        smooth_pairs = adjacency_pairs[
            normal_cosines
            >= np.cos(np.deg2rad(config.crease_angle_degrees))
        ]
    else:
        smooth_pairs = adjacency_pairs
    face_neighbors = _face_neighbors(adjacency_pairs, len(faces))

    all_faces = np.arange(len(faces), dtype=np.int64)
    patches = []
    for component in _connected_components(
        all_faces, smooth_pairs, len(faces)
    ):
        direct_patch = _make_patch(component, vertices, faces, normals, config)
        if direct_patch.surface_type == SurfaceType.PLANE:
            patches.append(direct_patch)
            continue

        planar_patches, remaining = _extract_planar_seeds(
            component,
            face_neighbors,
            vertices,
            faces,
            normals,
            config,
        )
        if not planar_patches:
            patches.append(direct_patch)
            continue

        patches.extend(planar_patches)
        for curved_component in _connected_components(
            remaining, smooth_pairs, len(faces)
        ):
            patches.append(
                _make_patch(
                    curved_component, vertices, faces, normals, config
                )
            )

    patches.sort(key=lambda patch: int(patch.face_indices.min()))
    face_patch_ids = np.full(len(faces), -1, dtype=np.int64)
    for patch_id, patch in enumerate(patches):
        patch.patch_id = patch_id
        face_patch_ids[patch.face_indices] = patch_id
    if np.any(face_patch_ids < 0):
        raise RuntimeError("Internal error: not every input face was classified.")

    _set_patch_neighbors(patches, face_patch_ids, edge_memberships)
    _classify_transition_roles(patches, mesh_scale, config)
    return MeshFeatureClassification(
        patches=tuple(patches),
        face_patch_ids=face_patch_ids,
        mesh_scale=mesh_scale,
    )


def classify_stl_features(path, config=None):
    """Load, weld, and classify a binary or ASCII STL file."""
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError(
            "Reading STL files requires the project's 'trimesh' dependency."
        ) from error

    path = Path(path)
    if path.suffix.lower() != ".stl":
        raise ValueError("Feature input must be an STL file: {}".format(path))
    if not path.is_file():
        raise FileNotFoundError("STL file does not exist: {}".format(path))
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("STL does not contain a non-empty triangle mesh: {}".format(path))
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return classify_mesh_features(mesh, config=config)


__all__ = [
    "ClassificationConfig",
    "MeshFeatureClassification",
    "SurfacePatch",
    "SurfaceType",
    "TransitionType",
    "classify_mesh_features",
    "classify_stl_features",
]
