"""GPU surface sampling followed by feature-safe local retriangulation."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import igl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError, cKDTree

from .feature_edges import detect_reference_feature_edges
from .feature_optimize import mesh_quality_metrics


@dataclass(frozen=True)
class SurfaceSamples:
    """Points sampled directly from original triangles on a CUDA device."""

    points: torch.Tensor
    face_ids: torch.Tensor
    barycentric: torch.Tensor
    normals: torch.Tensor
    requested_count: int
    candidate_count: int


class _ReferencePatchProjector:
    """Closest-point queries against complete, immutable reference patches.

    Trees are built lazily once per patch. A query never searches another
    patch, even when another sheet is closer in Euclidean distance. Reference
    face IDs returned here are geometric support IDs, not triangle cages.
    """

    def __init__(self, vertices, faces, face_patch_ids):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces)
        self.face_patch_ids = np.asarray(face_patch_ids)
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                or not len(self.vertices) or not np.isfinite(self.vertices).all()
                or self.faces.ndim != 2 or self.faces.shape[1] != 3 or not len(self.faces)
                or not np.issubdtype(self.faces.dtype, np.integer)
                or self.faces.min() < 0 or self.faces.max() >= len(self.vertices)
                or self.face_patch_ids.shape != (len(self.faces),)
                or not np.issubdtype(self.face_patch_ids.dtype, np.integer)
                or np.any(self.face_patch_ids < 0)):
            raise ValueError("Reference patches require finite vertices, valid triangular faces and matching integer patch IDs.")
        self.faces = self.faces.astype(np.int64, copy=False)
        self.face_patch_ids = self.face_patch_ids.astype(np.int64, copy=False)
        ordered = np.argsort(self.face_patch_ids, kind="stable")
        labels, starts, counts = np.unique(self.face_patch_ids[ordered],
                                          return_index=True, return_counts=True)
        self.patch_faces = {
            int(label): ordered[start:start + count]
            for label, start, count in zip(labels, starts, counts)
        }
        self.trees = {}
        self.normal_support_trees = {}
        triangles = self.vertices[self.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(normal_lengths == 0):
            raise ValueError("Whole-patch reference contains a degenerate triangle.")
        self.normals = normals / normal_lengths
        maximum_edge_squared = np.max(np.sum(
            (triangles[:, (1, 2, 0)] - triangles) ** 2, axis=2), axis=1)
        # Dot-product barycentrics lose precision when the Gram determinant
        # nearly cancels. Keep these rare faces available for a stable
        # point-on-triangle check if the nearest support has the wrong normal.
        poorly_conditioned = normal_lengths[:, 0] / maximum_edge_squared < np.finfo(np.float64).eps ** .25
        self.thin_patch_faces = {
            label: ids[poorly_conditioned[ids]] for label, ids in self.patch_faces.items()
        }
        self.numeric_tolerance = max(float(np.ptp(self.vertices, axis=0).max()),
                                     np.finfo(np.float64).tiny) * 1e-12
        self.normal_tie_tolerance = max(float(np.abs(self.vertices).max()),
                                        float(np.ptp(self.vertices, axis=0).max()),
                                        np.finfo(np.float64).tiny) * np.finfo(np.float64).eps * 64

    def query_numpy(self, points, patch_ids):
        points = np.asarray(points, dtype=np.float64)
        patch_ids = np.asarray(patch_ids, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or patch_ids.shape != (len(points),):
            raise ValueError("Patch projection needs (n, 3) points and n patch IDs.")
        if not np.isfinite(points).all():
            raise ValueError("Patch projection points must be finite.")
        closest = np.empty_like(points)
        squared = np.empty(len(points), dtype=np.float64)
        source_ids = np.empty(len(points), dtype=np.int64)
        order = np.argsort(patch_ids, kind="stable")
        labels, starts, counts = np.unique(patch_ids[order], return_index=True,
                                          return_counts=True)
        for label, start, count in zip(labels, starts, counts):
            label = int(label)
            if label not in self.patch_faces:
                raise ValueError("Projection references an unknown reference patch.")
            if label not in self.trees:
                source_faces = self.patch_faces[label]
                vertex_ids, local_faces = np.unique(self.faces[source_faces], return_inverse=True)
                local_vertices = np.ascontiguousarray(self.vertices[vertex_ids])
                local_faces = np.ascontiguousarray(local_faces.reshape(-1, 3), dtype=np.int32)
                tree = igl.AABB_f64_3()
                tree.init(local_vertices, local_faces)
                self.trees[label] = (tree, local_vertices, local_faces, source_faces)
            tree, local_vertices, local_faces, source_faces = self.trees[label]
            # Bound temporary arrays for large patches without rebuilding the tree.
            for offset in range(start, start + count, 32768):
                rows = order[offset:min(offset + 32768, start + count)]
                distances, local_ids, positions = tree.squared_distance(
                    local_vertices, local_faces, np.ascontiguousarray(points[rows]),
                    return_index=True, return_closest_point=True,
                )
                closest[rows] = positions
                squared[rows] = np.asarray(distances).reshape(-1)
                source_ids[rows] = source_faces[np.asarray(local_ids).reshape(-1)]
        return closest, squared, source_ids

    def project(self, points, patch_ids):
        positions, _, source_ids = self.query_numpy(points.detach().cpu().numpy(),
                                                    patch_ids.detach().cpu().numpy())
        return (torch.as_tensor(positions, device=points.device, dtype=points.dtype),
                torch.as_tensor(source_ids, device=points.device, dtype=torch.long))

    def _normal_support_candidates(self, points, label, thin_only):
        """Conservative spatial broad phase for the exact normal-tie test.

        A triangle's bounding box is enclosed by a sphere about its box center.
        Radius buckets keep one long triangle from making every query scan the
        entire patch. All returned candidates still undergo the original box,
        plane, inward-edge and normal checks; no nearest-k approximation is used.
        """
        key = (int(label), bool(thin_only))
        if key not in self.normal_support_trees:
            ids = self.thin_patch_faces[label] if thin_only else self.patch_faces[label]
            triangles = self.vertices[self.faces[ids]]
            lower, upper = triangles.min(axis=1), triangles.max(axis=1)
            centers = lower * .5 + upper * .5
            radii = np.linalg.norm((upper - lower) * .5, axis=1)
            exponents = np.frexp(radii)[1]
            buckets = []
            for exponent in np.unique(exponents):
                members = np.flatnonzero(exponents == exponent)
                # Cover expansion in all three box coordinates, plus rounding
                # in the sphere centers/radii and KD-tree distance arithmetic.
                radius = float(radii[members].max()) + 4 * self.normal_tie_tolerance
                buckets.append((cKDTree(centers[members]), ids[members], radius))
            self.normal_support_trees[key] = buckets
        results = [[] for _ in range(len(points))]
        for tree, ids, radius in self.normal_support_trees[key]:
            nearby = tree.query_ball_point(points, radius, eps=0., return_sorted=True)
            for collected, local_ids in zip(results, nearby):
                if local_ids:
                    collected.append(ids[local_ids])
        # Preserve original source-face ordering when equally good supports tie.
        return [np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
                for parts in results]

    def valid_triangles(self, triangles, patch_ids, maximum_deviation,
                        maximum_normal_deviation_degrees, return_details=False):
        """Check seven surface samples plus the oriented centroid normal.

        This is a sampled deviation check, not a certified Hausdorff bound.
        Edge midpoints prevent a proposed diagonal from silently bridging a
        hole or cutting a curved surface despite having on-surface endpoints.
        """
        positions = triangles.detach().cpu().numpy()
        labels = patch_ids.detach().cpu().numpy()
        if len(positions) == 0:
            valid = torch.ones(0, dtype=torch.bool, device=triangles.device)
            return (valid, {}) if return_details else valid
        midpoints = 0.5 * (positions + positions[:, (1, 2, 0)])
        samples = np.concatenate((positions, midpoints, positions.mean(axis=1)[:, None]), axis=1)
        _, squared, source_ids = self.query_numpy(samples.reshape(-1, 3),
                                                  np.repeat(labels, 7))
        valid = np.ones(len(positions), dtype=bool)
        distance_valid = np.ones(len(positions), dtype=bool)
        normal_valid = np.ones(len(positions), dtype=bool)
        alignment = np.ones(len(positions))
        lengths = np.zeros(len(positions))
        resolved_normal_ties = np.zeros(len(positions), dtype=bool)
        if maximum_deviation is not None:
            distance_valid = squared.reshape(-1, 7).max(axis=1) <= (
                float(maximum_deviation) + self.numeric_tolerance) ** 2
            valid &= distance_valid
        if maximum_normal_deviation_degrees is not None:
            normals = np.cross(positions[:, 1] - positions[:, 0],
                               positions[:, 2] - positions[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            reference_normals = self.normals[source_ids.reshape(-1, 7)[:, -1]]
            alignment = np.einsum("ij,ij->i", normals, reference_normals) / lengths.clip(
                np.finfo(np.float64).tiny, None)
            cosine_limit = np.cos(np.deg2rad(maximum_normal_deviation_degrees)) - 1e-12
            normal_valid = (alignment >= cosine_limit) & (lengths > 0)
            centroid_distances = np.sqrt(np.maximum(squared.reshape(-1, 7)[:, -1], 0))
            ambiguous = np.flatnonzero(~normal_valid & (lengths > 0))
            support_candidates = {}
            for label in np.unique(labels[ambiguous]):
                patch_rows = ambiguous[labels[ambiguous] == label]
                for thin_only in (False, True):
                    rows = patch_rows[(centroid_distances[patch_rows] > self.normal_tie_tolerance) == thin_only]
                    if len(rows):
                        nearby = self._normal_support_candidates(samples[rows, -1], int(label), thin_only)
                        support_candidates.update(zip(rows, nearby))
            for row in ambiguous:
                # AABB can choose an overlapping face, or miss an extremely
                # thin face because its barycentric denominator cancels. A
                # stable plane/edge test must prove the centroid belongs to
                # an oriented reference triangle within floating-point error;
                # input provenance alone never grants an exemption.
                unit_normal = normals[row] / lengths[row]
                candidates = support_candidates[row]
                candidates = candidates[self.normals[candidates] @ unit_normal >= cosine_limit]
                if not len(candidates):
                    continue
                centroid = samples[row, -1]
                candidate_triangles = self.vertices[self.faces[candidates]]
                tolerance = self.normal_tie_tolerance
                nearby = ((centroid >= candidate_triangles.min(axis=1) - tolerance)
                          & (centroid <= candidate_triangles.max(axis=1) + tolerance)).all(axis=1)
                candidates = candidates[nearby]
                candidate_triangles = candidate_triangles[nearby]
                if not len(candidates):
                    continue
                candidate_normals = self.normals[candidates]
                plane_distances = np.abs(np.einsum(
                    "ij,ij->i", centroid - candidate_triangles[:, 0], candidate_normals))
                edges = candidate_triangles[:, (1, 2, 0)] - candidate_triangles
                inward_distances = np.einsum(
                    "ijk,ik->ij", np.cross(edges, centroid - candidate_triangles), candidate_normals,
                ) / np.linalg.norm(edges, axis=2)
                supports = np.flatnonzero((plane_distances <= tolerance)
                                          & (inward_distances >= -tolerance).all(axis=1))
                if len(supports):
                    reference_id = candidates[supports[np.argmin(plane_distances[supports])]]
                    alignment[row] = float(self.normals[reference_id] @ unit_normal)
                    source_ids.reshape(-1, 7)[row, -1] = reference_id
                    normal_valid[row] = True
                    resolved_normal_ties[row] = True
            valid &= normal_valid
        mask = torch.as_tensor(valid, device=triangles.device, dtype=torch.bool)
        if return_details:
            return mask, {
                "distance_valid": distance_valid, "normal_valid": normal_valid,
                "maximum_sample_distance": np.sqrt(np.maximum(squared.reshape(-1, 7).max(axis=1), 0)),
                "normal_deviation_degrees": np.rad2deg(np.arccos(np.clip(alignment, -1, 1))),
                "double_area": lengths,
                "centroid_reference_face_ids": source_ids.reshape(-1, 7)[:, -1],
                "normal_reference_ties_resolved": resolved_normal_ties,
            }
        return mask


def _make_reference_projector(vertices, faces, labels, *, backend="cuda", device="cuda"):
    if backend == "cpu":
        return _ReferencePatchProjector(vertices, faces, labels)
    if backend != "cuda":
        raise ValueError("projection_backend must be 'cuda' or 'cpu'.")
    try:
        from .cuda_surface_project import CudaReferencePatchProjector
    except ImportError as error:
        raise RuntimeError("CUDA surface projection requires Warp in the PaMO environment; use projection_backend='cpu' for the reference path.") from error
    return CudaReferencePatchProjector(vertices, faces, labels, device=device)


def adaptive_target_length_from_curvature(
    maximum_absolute_curvature,
    tolerance,
    minimum_edge_length,
    maximum_edge_length,
):
    """Return the CGAL/Dunyach curvature-adaptive target edge length.

    The tolerance is a world-space surface approximation error.  Zero
    curvature maps to the maximum length; increasing curvature continuously
    reduces the target until the minimum length clamp is reached.
    """
    curvature = np.asarray(maximum_absolute_curvature, dtype=np.float64)
    tolerance = float(tolerance)
    minimum_edge_length = float(minimum_edge_length)
    maximum_edge_length = float(maximum_edge_length)
    if tolerance <= 0.0:
        raise ValueError("Adaptive curvature tolerance must be positive.")
    if minimum_edge_length <= 0.0:
        raise ValueError("Adaptive minimum edge length must be positive.")
    if maximum_edge_length < minimum_edge_length:
        raise ValueError(
            "Adaptive maximum edge length must not be smaller than the minimum."
        )
    curvature = np.where(np.isfinite(curvature), np.abs(curvature), 0.0)
    target_squared = np.full(curvature.shape, np.inf, dtype=np.float64)
    nonzero = curvature > np.finfo(np.float64).eps
    target_squared[nonzero] = (
        6.0 * tolerance / curvature[nonzero]
        - 3.0 * tolerance * tolerance
    )
    target = np.sqrt(np.maximum(target_squared, 0.0))
    return np.clip(target, minimum_edge_length, maximum_edge_length)


def curvature_adaptive_source_face_lengths(
    vertices,
    faces,
    tolerance,
    minimum_edge_length,
    maximum_edge_length,
):
    """Estimate a patch-independent adaptive sizing field on an STL mesh."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Adaptive sizing requires a nonempty triangle mesh.")
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    cross_lengths = np.linalg.norm(crosses, axis=1)
    valid = cross_lengths > np.finfo(np.float64).eps
    if not np.any(valid):
        raise ValueError("Adaptive sizing mesh contains only degenerate faces.")
    normals = crosses[valid] / cross_lengths[valid, None]
    reference_normal = normals[0]
    if np.all(np.abs(normals @ reference_normal) >= 1.0 - 1e-12):
        vertex_lengths = np.full(
            len(vertices), float(maximum_edge_length), dtype=np.float64
        )
        return vertex_lengths, np.full(
            len(faces), float(maximum_edge_length), dtype=np.float64
        )
    _, _, first_curvature, second_curvature = igl.principal_curvature(
        vertices, faces
    )
    maximum_curvature = np.maximum(
        np.abs(np.asarray(first_curvature, dtype=np.float64)),
        np.abs(np.asarray(second_curvature, dtype=np.float64)),
    )
    vertex_lengths = adaptive_target_length_from_curvature(
        maximum_curvature,
        tolerance,
        minimum_edge_length,
        maximum_edge_length,
    )
    # CGAL tests an edge against the smaller endpoint target.  A conservative
    # face value gives the same behavior after source faces are subdivided.
    face_lengths = np.min(vertex_lengths[faces], axis=1)
    return vertex_lengths, face_lengths


def _triangle_cross_torch(vertices, faces):
    triangles = vertices[faces]
    return torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )


def _triangle_quality_torch(vertices, faces):
    triangles = vertices[faces]
    ab = triangles[:, 1] - triangles[:, 0]
    bc = triangles[:, 2] - triangles[:, 1]
    ca = triangles[:, 0] - triangles[:, 2]
    twice_area = torch.linalg.norm(torch.cross(ab, -ca, dim=1), dim=1)
    squared_length_sum = (
        (ab * ab).sum(dim=1)
        + (bc * bc).sum(dim=1)
        + (ca * ca).sum(dim=1)
    )
    epsilon = torch.finfo(vertices.dtype).eps
    return (
        2.0
        * np.sqrt(3.0)
        * twice_area
        / squared_length_sum.clamp_min(epsilon)
    ).clamp(0.0, 1.0)


def _triangle_quality_numpy(vertices, faces):
    triangles = np.asarray(vertices)[np.asarray(faces)]
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


def _unique_edges_torch(faces):
    edges = torch.cat(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        ),
        dim=0,
    )
    return torch.unique(torch.sort(edges, dim=1).values, dim=0)


def _mesh_energy_torch(vertices, faces):
    edges = _unique_edges_torch(faces)
    lengths = torch.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]],
        dim=1,
    )
    epsilon = torch.finfo(vertices.dtype).eps
    mean_length = lengths.mean().clamp_min(epsilon)
    edge_cv_squared = ((lengths - mean_length) ** 2).mean() / (
        mean_length * mean_length
    )
    quality = _triangle_quality_torch(vertices, faces)
    return edge_cv_squared + ((1.0 - quality) ** 2).mean()


def _grid_poisson_filter(
    points,
    face_ids,
    barycentric,
    normals,
    radius,
    maximum_count,
):
    """Select a radius-independent set using a CUDA spatial hash."""
    radius = float(radius)
    if radius <= 0.0 or len(points) == 0:
        return (
            points[:maximum_count],
            face_ids[:maximum_count],
            barycentric[:maximum_count],
            normals[:maximum_count],
        )

    cell_coordinates = torch.floor(
        (points - points.amin(dim=0)) / radius
    ).to(torch.int64)
    unique_cells, inverse, counts = torch.unique(
        cell_coordinates,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    order = torch.argsort(inverse)
    starts = torch.cumsum(counts, dim=0) - counts
    representatives = order[starts]
    if len(representatives) <= 1:
        keep = representatives[:maximum_count]
        return points[keep], face_ids[keep], barycentric[keep], normals[keep]

    representative_points = points[representatives]
    extents = unique_cells.amax(dim=0) + 1
    keys = (
        (unique_cells[:, 0] * extents[1] + unique_cells[:, 1])
        * extents[2]
        + unique_cells[:, 2]
    )
    sorted_keys, cell_order = torch.sort(keys)
    unique_cells = unique_cells[cell_order]
    representatives = representatives[cell_order]
    representative_points = representative_points[cell_order]

    neighbor_table = torch.full(
        (len(representatives), 27),
        -1,
        dtype=torch.long,
        device=points.device,
    )
    column = 0
    for x_offset in (-1, 0, 1):
        for y_offset in (-1, 0, 1):
            for z_offset in (-1, 0, 1):
                offset = torch.tensor(
                    (x_offset, y_offset, z_offset),
                    dtype=torch.long,
                    device=points.device,
                )
                query_cells = unique_cells + offset
                inside = (
                    (query_cells >= 0)
                    & (query_cells < extents[None, :])
                ).all(dim=1)
                query_keys = (
                    (
                        query_cells[:, 0] * extents[1]
                        + query_cells[:, 1]
                    )
                    * extents[2]
                    + query_cells[:, 2]
                )
                locations = torch.searchsorted(sorted_keys, query_keys)
                clamped = locations.clamp_max(len(sorted_keys) - 1)
                exists = (
                    inside
                    & (locations < len(sorted_keys))
                    & (sorted_keys[clamped] == query_keys)
                )
                neighbor_table[exists, column] = clamped[exists]
                column += 1

    clamped_neighbors = neighbor_table.clamp_min(0)
    neighbor_points = representative_points[clamped_neighbors]
    squared_distances = (
        (
            neighbor_points
            - representative_points[:, None, :]
        )
        ** 2
    ).sum(dim=2)
    neighbor_valid = (
        (neighbor_table >= 0)
        & (squared_distances < radius * radius)
    )

    priorities = torch.arange(
        len(representatives),
        dtype=torch.long,
        device=points.device,
    )
    active = torch.ones(
        len(representatives),
        dtype=torch.bool,
        device=points.device,
    )
    accepted = []
    sentinel = len(representatives)
    while bool(active.any()) and sum(len(batch) for batch in accepted) < maximum_count:
        neighbor_priorities = torch.where(
            neighbor_valid & active[clamped_neighbors],
            priorities[clamped_neighbors],
            sentinel,
        )
        local_minimum = active & (
            priorities <= neighbor_priorities.amin(dim=1)
        )
        selected = torch.nonzero(local_minimum).reshape(-1)
        if len(selected) == 0:
            break
        remaining = maximum_count - sum(len(batch) for batch in accepted)
        accepted.append(selected[:remaining])
        selected_mask = torch.zeros_like(active)
        selected_mask[selected] = True
        blocked = (
            neighbor_valid & selected_mask[clamped_neighbors]
        ).any(dim=1)
        active &= ~blocked

    if accepted:
        selected_representatives = torch.cat(accepted)
        keep = representatives[selected_representatives]
    else:
        keep = representatives[:0]
    return points[keep], face_ids[keep], barycentric[keep], normals[keep]


@torch.no_grad()
def gpu_area_poisson_sample(
    vertices,
    faces,
    sample_count,
    poisson_radius=None,
    oversample=4,
    seed=0,
    barycentric_margin=0.08,
    eligible_face_mask=None,
):
    """
    Generate area-weighted original-surface samples with CUDA tensor kernels.

    The returned source face and barycentric coordinates are retained so the
    samples never become an unlabelled point cloud.
    """
    if not torch.is_tensor(vertices) or not torch.is_tensor(faces):
        raise TypeError("Surface sampling expects PyTorch tensors.")
    if not vertices.is_cuda or not faces.is_cuda:
        raise ValueError("Surface sampling vertices and faces must be on CUDA.")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Surface sampling vertices must have shape (n, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Surface sampling faces must have shape (m, 3).")

    sample_count = int(sample_count)
    oversample = int(oversample)
    barycentric_margin = float(barycentric_margin)
    if sample_count <= 0:
        raise ValueError("Surface sample count must be positive.")
    if oversample <= 0:
        raise ValueError("Surface sample oversampling factor must be positive.")
    if not 0.0 <= barycentric_margin < 1.0 / 3.0:
        raise ValueError("Barycentric margin must be in [0, 1/3).")

    faces = faces.to(dtype=torch.long)
    cross = _triangle_cross_torch(vertices, faces)
    double_areas = torch.linalg.norm(cross, dim=1)
    if not bool(torch.isfinite(double_areas).all()):
        raise ValueError("Surface sampling mesh contains non-finite triangles.")
    if float(double_areas.sum().item()) <= 0.0:
        raise ValueError("Surface sampling mesh has zero total area.")
    sampling_weights = double_areas
    if eligible_face_mask is not None:
        eligible_face_mask = torch.as_tensor(
            eligible_face_mask,
            dtype=torch.bool,
            device=vertices.device,
        )
        if eligible_face_mask.shape != (len(faces),):
            raise ValueError(
                "Eligible surface face mask must have shape (face_count,)."
            )
        sampling_weights = torch.where(
            eligible_face_mask,
            double_areas,
            torch.zeros_like(double_areas),
        )
        if float(sampling_weights.sum().item()) <= 0.0:
            raise ValueError(
                "No original triangles satisfy the surface sampling "
                "eligibility thresholds."
            )

    use_poisson = poisson_radius is not None and float(poisson_radius) > 0.0
    candidate_count = sample_count * (oversample if use_poisson else 1)
    generator = torch.Generator(device=vertices.device)
    generator.manual_seed(int(seed))
    sampled_face_ids = torch.multinomial(
        sampling_weights,
        candidate_count,
        replacement=True,
        generator=generator,
    )

    random_values = torch.rand(
        (candidate_count, 2),
        dtype=vertices.dtype,
        device=vertices.device,
        generator=generator,
    )
    root = torch.sqrt(random_values[:, 0])
    barycentric = torch.stack(
        (
            1.0 - root,
            root * (1.0 - random_values[:, 1]),
            root * random_values[:, 1],
        ),
        dim=1,
    )
    if barycentric_margin > 0.0:
        barycentric = (
            (1.0 - 3.0 * barycentric_margin) * barycentric
            + barycentric_margin
        )

    sampled_triangles = vertices[faces[sampled_face_ids]]
    points = (
        sampled_triangles * barycentric[:, :, None]
    ).sum(dim=1)
    epsilon = torch.finfo(vertices.dtype).eps
    face_normals = cross / double_areas[:, None].clamp_min(epsilon)
    normals = face_normals[sampled_face_ids]

    points, sampled_face_ids, barycentric, normals = _grid_poisson_filter(
        points,
        sampled_face_ids,
        barycentric,
        normals,
        poisson_radius if use_poisson else 0.0,
        sample_count,
    )
    return SurfaceSamples(
        points=points,
        face_ids=sampled_face_ids,
        barycentric=barycentric,
        normals=normals,
        requested_count=sample_count,
        candidate_count=candidate_count,
    )


def _edge_keys(edges, vertex_count):
    edges = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    return edges[:, 0] * np.int64(vertex_count) + edges[:, 1]


def original_surface_patch_ids(faces, vertex_count, feature_edges):
    """Return smooth-patch labels separated by sharp and boundary edges."""
    faces = np.asarray(faces, dtype=np.int64)
    face_ids = np.arange(len(faces), dtype=np.int64)
    edges = np.vstack(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        )
    )
    edges = np.sort(edges, axis=1)
    owners = np.tile(face_ids, 3)
    keys = _edge_keys(edges, vertex_count)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_owners = owners[order]
    unique_keys, starts, counts = np.unique(
        sorted_keys,
        return_index=True,
        return_counts=True,
    )
    manifold = counts == 2
    if len(feature_edges):
        feature_keys = np.unique(_edge_keys(feature_edges, vertex_count))
        manifold &= ~np.isin(unique_keys, feature_keys, assume_unique=True)
    starts = starts[manifold]
    if len(starts) == 0:
        return np.arange(len(faces), dtype=np.int64)

    first = sorted_owners[starts]
    second = sorted_owners[starts + 1]
    rows = np.concatenate((first, second))
    columns = np.concatenate((second, first))
    graph = coo_matrix(
        (
            np.ones(len(rows), dtype=np.uint8),
            (rows, columns),
        ),
        shape=(len(faces), len(faces)),
    ).tocsr()
    _, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    return labels.astype(np.int64, copy=False)


def coplanar_internal_edges(
    mesh,
    feature_edges=None,
    angle_degrees=1.0,
):
    """Return manifold source edges that are coplanar and non-feature."""
    angle_degrees = float(angle_degrees)
    if not 0.0 <= angle_degrees < 180.0:
        raise ValueError("Coplanar edge angle must be in [0, 180).")
    adjacency_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    adjacency_angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    if len(adjacency_edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    coplanar = adjacency_angles <= np.deg2rad(angle_degrees)
    if feature_edges is not None and len(feature_edges):
        feature_keys = set(
            map(
                tuple,
                np.sort(np.asarray(feature_edges, dtype=np.int64), axis=1),
            )
        )
        coplanar &= np.asarray(
            [tuple(edge) not in feature_keys for edge in adjacency_edges],
            dtype=bool,
        )
    return np.unique(
        np.sort(adjacency_edges[coplanar], axis=1),
        axis=0,
    )


def subdivide_original_faces(
    vertices,
    faces,
    sample_points,
    sample_face_ids,
    sample_barycentric,
    edge_target_length=None,
    eligible_face_mask=None,
    protected_internal_edges=None,
):
    """
    Insert samples into their original triangles and retriangulate locally.

    Each source triangle is a convex 2D domain, so ordinary local Delaunay
    triangulation is constrained by its three original boundary edges. Adjacent
    source triangles therefore stitch with the exact original edge graph.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    sample_points = np.asarray(sample_points, dtype=np.float64)
    sample_face_ids = np.asarray(sample_face_ids, dtype=np.int64).reshape(-1)
    sample_barycentric = np.asarray(
        sample_barycentric,
        dtype=np.float64,
    )
    if len(sample_points) != len(sample_face_ids):
        raise ValueError("Surface samples and source face ids have different sizes.")
    if sample_barycentric.shape != (len(sample_points), 3):
        raise ValueError("Surface sample barycentrics must have shape (n, 3).")
    if len(sample_face_ids) and (
        sample_face_ids.min() < 0 or sample_face_ids.max() >= len(faces)
    ):
        raise ValueError("A surface sample references an invalid original face.")
    if eligible_face_mask is None:
        eligible_face_mask = np.ones(len(faces), dtype=bool)
    else:
        eligible_face_mask = np.asarray(
            eligible_face_mask,
            dtype=bool,
        )
        if eligible_face_mask.shape != (len(faces),):
            raise ValueError(
                "Eligible source face mask must have shape (face_count,)."
            )
    if len(sample_face_ids) and not np.all(
        eligible_face_mask[sample_face_ids]
    ):
        raise ValueError("A surface sample references an ineligible source face.")

    if len(sample_points) == 0:
        return (
            vertices.copy(),
            faces.copy(),
            np.arange(len(faces), dtype=np.int64),
            np.full(len(vertices), -1, dtype=np.int64),
        )

    stacked_edges = np.vstack(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        )
    )
    sorted_edges = np.sort(stacked_edges, axis=1)
    unique_edges, edge_inverse = np.unique(
        sorted_edges,
        axis=0,
        return_inverse=True,
    )
    face_edge_ids = edge_inverse.reshape(3, len(faces)).T
    active_edge_ids = np.unique(
        face_edge_ids[np.unique(sample_face_ids)].reshape(-1)
    )
    # Do not subdivide an edge when any incident source face is too small or
    # numerically thin for reliable float32 CUDA retriangulation.
    edge_is_safe = np.ones(len(unique_edges), dtype=bool)
    np.logical_and.at(
        edge_is_safe,
        edge_inverse,
        np.tile(eligible_face_mask, 3),
    )
    if protected_internal_edges is not None and len(protected_internal_edges):
        protected_keys = {
            tuple(edge)
            for edge in np.sort(
                np.asarray(protected_internal_edges, dtype=np.int64),
                axis=1,
            )
        }
        protected_edge_mask = np.asarray(
            [tuple(edge) in protected_keys for edge in unique_edges],
            dtype=bool,
        )
        edge_is_safe &= ~protected_edge_mask
    active_edge_ids = active_edge_ids[edge_is_safe[active_edge_ids]]
    edge_point_ids = {}
    edge_point_parameters = {}
    edge_points = []
    if edge_target_length is not None:
        edge_target_length = float(edge_target_length)
        if edge_target_length <= 0.0:
            raise ValueError("Surface edge target length must be positive.")
        active_edges = unique_edges[active_edge_ids]
        active_lengths = np.linalg.norm(
            vertices[active_edges[:, 1]] - vertices[active_edges[:, 0]],
            axis=1,
        )
        segment_counts = np.maximum(
            np.ceil(active_lengths / edge_target_length).astype(np.int64),
            1,
        )
        edge_point_base = len(vertices)
        for edge_id, edge, segment_count in zip(
            active_edge_ids,
            active_edges,
            segment_counts,
        ):
            if segment_count <= 1:
                continue
            parameters = (
                np.arange(1, segment_count, dtype=np.float64)
                / float(segment_count)
            )
            first_id = edge_point_base + len(edge_points)
            point_ids = first_id + np.arange(
                len(parameters),
                dtype=np.int64,
            )
            points = (
                (1.0 - parameters[:, None]) * vertices[int(edge[0])]
                + parameters[:, None] * vertices[int(edge[1])]
            )
            edge_points.extend(points)
            edge_point_ids[int(edge_id)] = point_ids
            edge_point_parameters[int(edge_id)] = parameters

    edge_points = np.asarray(edge_points, dtype=np.float64).reshape(-1, 3)
    interior_vertex_base = len(vertices) + len(edge_points)
    output_vertices = np.vstack((vertices, edge_points, sample_points))
    support_face_ids = np.full(len(output_vertices), -1, dtype=np.int64)
    support_face_ids[interior_vertex_base:] = sample_face_ids

    order = np.argsort(sample_face_ids, kind="stable")
    sorted_face_ids = sample_face_ids[order]
    group_faces, starts, counts = np.unique(
        sorted_face_ids,
        return_index=True,
        return_counts=True,
    )
    changed = np.zeros(len(faces), dtype=bool)
    changed[group_faces] = True
    if edge_point_ids:
        subdivided_edge_mask = np.zeros(len(unique_edges), dtype=bool)
        subdivided_edge_mask[
            np.fromiter(edge_point_ids, dtype=np.int64)
        ] = True
        changed |= np.any(subdivided_edge_mask[face_edge_ids], axis=1)
    output_faces = [faces[~changed]]
    output_source_faces = [
        np.flatnonzero(~changed).astype(np.int64, copy=False)
    ]

    sample_global_ids = interior_vertex_base + np.arange(
        len(sample_points),
        dtype=np.int64,
    )
    sample_groups = {
        int(source_face_id): order[start:start + count]
        for source_face_id, start, count in zip(
            group_faces,
            starts,
            counts,
        )
    }
    for source_face_id in np.flatnonzero(changed):
        local_sample_ids = sample_groups.get(
            int(source_face_id),
            np.empty(0, dtype=np.int64),
        )
        original_face = faces[int(source_face_id)]
        boundary_vertex_ids = []
        boundary_barycentric = []
        for edge_id in face_edge_ids[int(source_face_id)]:
            edge_id = int(edge_id)
            if edge_id not in edge_point_ids:
                continue
            edge = unique_edges[edge_id]
            point_ids = edge_point_ids[edge_id]
            parameters = edge_point_parameters[edge_id]
            first_position = int(np.flatnonzero(original_face == edge[0])[0])
            second_position = int(np.flatnonzero(original_face == edge[1])[0])
            barycentric = np.zeros((len(point_ids), 3), dtype=np.float64)
            barycentric[:, first_position] = 1.0 - parameters
            barycentric[:, second_position] = parameters
            boundary_vertex_ids.extend(point_ids)
            boundary_barycentric.extend(barycentric)

        boundary_vertex_ids = np.asarray(
            boundary_vertex_ids,
            dtype=np.int64,
        )
        boundary_barycentric = np.asarray(
            boundary_barycentric,
            dtype=np.float64,
        ).reshape(-1, 3)
        if len(local_sample_ids) == 1 and len(boundary_vertex_ids) == 0:
            point_id = int(sample_global_ids[local_sample_ids[0]])
            local_faces = np.asarray(
                (
                    [original_face[0], original_face[1], point_id],
                    [original_face[1], original_face[2], point_id],
                    [original_face[2], original_face[0], point_id],
                ),
                dtype=np.int64,
            )
        else:
            combined_barycentric = np.vstack(
                (
                    boundary_barycentric,
                    sample_barycentric[local_sample_ids],
                )
            )
            triangle_vertices = vertices[original_face]
            first_axis = triangle_vertices[1] - triangle_vertices[0]
            first_length = np.linalg.norm(first_axis)
            if first_length <= np.finfo(np.float64).eps:
                raise RuntimeError(
                    "Original face {} has a degenerate first edge.".format(
                        source_face_id
                    )
                )
            first_axis = first_axis / first_length
            triangle_normal = np.cross(
                triangle_vertices[1] - triangle_vertices[0],
                triangle_vertices[2] - triangle_vertices[0],
            )
            normal_length = np.linalg.norm(triangle_normal)
            if normal_length <= np.finfo(np.float64).eps:
                raise RuntimeError(
                    "Original face {} is degenerate.".format(source_face_id)
                )
            triangle_normal = triangle_normal / normal_length
            second_axis = np.cross(triangle_normal, first_axis)
            third_offset = triangle_vertices[2] - triangle_vertices[0]
            corner_coordinates = np.asarray(
                (
                    (0.0, 0.0),
                    (first_length, 0.0),
                    (
                        float(np.dot(third_offset, first_axis)),
                        float(np.dot(third_offset, second_axis)),
                    ),
                ),
                dtype=np.float64,
            )
            local_coordinates = np.vstack(
                (
                    corner_coordinates,
                    combined_barycentric @ corner_coordinates,
                )
            )
            local_vertex_ids = np.concatenate(
                (
                    original_face,
                    boundary_vertex_ids,
                    sample_global_ids[local_sample_ids],
                )
            )
            try:
                simplices = Delaunay(local_coordinates).simplices
            except QhullError:
                try:
                    simplices = Delaunay(
                        local_coordinates,
                        qhull_options="QJ",
                    ).simplices
                except QhullError as error:
                    raise RuntimeError(
                        "Local surface-sample triangulation failed for "
                        "original face {}.".format(source_face_id)
                    ) from error

            coordinates = local_coordinates[simplices]
            signed_twice_area = (
                (coordinates[:, 1, 0] - coordinates[:, 0, 0])
                * (coordinates[:, 2, 1] - coordinates[:, 0, 1])
                - (coordinates[:, 1, 1] - coordinates[:, 0, 1])
                * (coordinates[:, 2, 0] - coordinates[:, 0, 0])
            )
            nondegenerate = np.abs(signed_twice_area) > 1e-14
            simplices = simplices[nondegenerate].copy()
            signed_twice_area = signed_twice_area[nondegenerate]
            reverse = signed_twice_area < 0.0
            simplices[reverse, 1], simplices[reverse, 2] = (
                simplices[reverse, 2].copy(),
                simplices[reverse, 1].copy(),
            )
            local_faces = local_vertex_ids[simplices]

        local_edge_set = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for face in local_faces
            for edge in (
                face[[0, 1]],
                face[[1, 2]],
                face[[2, 0]],
            )
        }
        for edge_id in face_edge_ids[int(source_face_id)]:
            edge_id = int(edge_id)
            if edge_id not in edge_point_ids:
                continue
            edge = unique_edges[edge_id]
            chain = np.concatenate(
                (
                    edge[:1],
                    edge_point_ids[edge_id],
                    edge[1:],
                )
            )
            if any(
                tuple(sorted((int(first), int(second))))
                not in local_edge_set
                for first, second in zip(chain[:-1], chain[1:])
            ):
                raise RuntimeError(
                    "Local Delaunay triangulation did not retain the shared "
                    "source-edge chain on face {}.".format(source_face_id)
                )

        output_faces.append(local_faces)
        output_source_faces.append(
            np.full(len(local_faces), source_face_id, dtype=np.int64)
        )

    return (
        output_vertices,
        np.vstack(output_faces),
        np.concatenate(output_source_faces),
        support_face_ids,
    )


def _gpu_edge_topology(faces, vertex_count):
    """Build sorted edge ownership tables without leaving the CUDA device."""
    face_ids = torch.arange(
        len(faces),
        dtype=torch.long,
        device=faces.device,
    ).repeat(3)
    directed_edges = torch.cat(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        ),
        dim=0,
    )
    opposite = torch.cat(
        (faces[:, 2], faces[:, 0], faces[:, 1]),
        dim=0,
    )
    sorted_edges = torch.sort(directed_edges, dim=1).values
    keys = sorted_edges[:, 0] * int(vertex_count) + sorted_edges[:, 1]
    sorted_keys, order = torch.sort(keys)
    unique_keys, counts = torch.unique_consecutive(
        sorted_keys,
        return_counts=True,
    )
    starts = torch.cumsum(counts, dim=0) - counts
    edges = torch.stack(
        (
            torch.div(
                unique_keys,
                int(vertex_count),
                rounding_mode="floor",
            ),
            unique_keys % int(vertex_count),
        ),
        dim=1,
    )
    return {
        "face_ids": face_ids,
        "directed_edges": directed_edges,
        "opposite": opposite,
        "order": order,
        "counts": counts,
        "starts": starts,
        "edges": edges,
    }


def _gpu_independent_candidates(candidate_vertices, priorities, candidate_count):
    """
    Select candidates with disjoint vertex footprints.

    ``candidate_vertices`` contains unique (candidate, footprint vertex)
    pairs. A candidate is accepted only when it has the best priority at every
    vertex in its closed one-ring, so accepted collapses cannot interact.
    """
    if candidate_count == 0 or len(candidate_vertices) == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=candidate_vertices.device,
        )
    vertex_count = int(candidate_vertices[:, 1].max().item()) + 1
    pair_keys = (
        candidate_vertices[:, 1] * (int(candidate_count) + 1)
        + priorities[candidate_vertices[:, 0]]
    )
    order = torch.argsort(pair_keys)
    ordered_vertices = candidate_vertices[order, 1]
    first = torch.ones(
        len(order),
        dtype=torch.bool,
        device=order.device,
    )
    first[1:] = ordered_vertices[1:] != ordered_vertices[:-1]
    winners = candidate_vertices[order[first], 0]
    win_counts = torch.bincount(winners, minlength=candidate_count)
    footprint_counts = torch.bincount(
        candidate_vertices[:, 0],
        minlength=candidate_count,
    )
    return torch.nonzero(
        win_counts == footprint_counts
    ).reshape(-1)


def _gpu_split_long_edges(
    vertices,
    faces,
    face_patch_ids,
    face_source_ids,
    support_face_ids,
    maximum_edge_length,
    maximum_passes,
    protected_edges=None,
    source_face_maximum_edge_lengths=None,
):
    """
    Bisect conflict-free long edges until the requested bound is reached.

    This runs before quality flips. At that point every edge lies inside one
    original source triangle or on a shared source edge, so its midpoint is
    still exactly on the input STL surface.
    """
    maximum_edge_length = float(maximum_edge_length)
    maximum_passes = int(maximum_passes)
    if maximum_edge_length <= 0.0:
        raise ValueError("Maximum surface edge length must be positive.")
    if maximum_passes <= 0:
        raise ValueError("Maximum surface split passes must be positive.")

    vertices = vertices.clone()
    faces = faces.clone()
    face_patch_ids = face_patch_ids.clone()
    face_source_ids = face_source_ids.clone()
    support_face_ids = support_face_ids.clone()
    if protected_edges is not None:
        protected_edges = protected_edges.to(
            device=faces.device,
            dtype=torch.long,
        )
        protected_edges = torch.sort(protected_edges, dim=1).values
    else:
        protected_edges = None
    if source_face_maximum_edge_lengths is not None:
        source_face_maximum_edge_lengths = torch.as_tensor(
            source_face_maximum_edge_lengths,
            dtype=vertices.dtype,
            device=vertices.device,
        )
    total_splits = 0

    for _ in range(maximum_passes):
        topology = _gpu_edge_topology(faces, len(vertices))
        edge_lengths = torch.linalg.norm(
            vertices[topology["edges"][:, 1]]
            - vertices[topology["edges"][:, 0]],
            dim=1,
        )
        eligible_groups = torch.nonzero(
            topology["counts"] <= 2
        ).reshape(-1)
        group_maximum_lengths = torch.full(
            (len(topology["counts"]),),
            maximum_edge_length,
            dtype=vertices.dtype,
            device=vertices.device,
        )
        if source_face_maximum_edge_lengths is not None and len(eligible_groups):
            eligible_counts = topology["counts"][eligible_groups]
            first_entries = topology["order"][
                topology["starts"][eligible_groups]
            ]
            group_maximum_lengths[eligible_groups] = (
                source_face_maximum_edge_lengths[
                    face_source_ids[topology["face_ids"][first_entries]]
                ]
            )
            paired = eligible_counts == 2
            if bool(paired.any()):
                second_entries = topology["order"][
                    topology["starts"][eligible_groups[paired]] + 1
                ]
                group_maximum_lengths[eligible_groups[paired]] = torch.minimum(
                    group_maximum_lengths[eligible_groups[paired]],
                    source_face_maximum_edge_lengths[
                        face_source_ids[topology["face_ids"][second_entries]]
                    ],
                )
        candidate_groups = eligible_groups[
            edge_lengths[eligible_groups]
            > group_maximum_lengths[eligible_groups]
        ]
        if protected_edges is not None and len(candidate_groups):
            protected_edge_keys = (
                protected_edges[:, 0] * len(vertices)
                + protected_edges[:, 1]
            )
            edge_keys = (
                topology["edges"][:, 0] * len(vertices)
                + topology["edges"][:, 1]
            )
            candidate_groups = candidate_groups[
                ~torch.isin(
                    edge_keys[candidate_groups],
                    protected_edge_keys,
                )
            ]
        if len(candidate_groups) == 0:
            return (
                vertices,
                faces,
                face_patch_ids,
                face_source_ids,
                support_face_ids,
                total_splits,
                0,
            )

        candidate_counts = topology["counts"][candidate_groups]
        first_entries = topology["order"][
            topology["starts"][candidate_groups]
        ]
        second_mask = candidate_counts == 2
        second_entries = topology["order"][
            topology["starts"][candidate_groups[second_mask]] + 1
        ]
        candidate_ids = torch.arange(
            len(candidate_groups),
            dtype=torch.long,
            device=faces.device,
        )
        owner_candidates = torch.cat(
            (candidate_ids, candidate_ids[second_mask])
        )
        owner_entries = torch.cat((first_entries, second_entries))
        owner_faces = topology["face_ids"][owner_entries]

        # A face can be bisected along only one of its edges in a batch. Give
        # its longest candidate priority, which also makes convergence fast on
        # very stretched planar faces.
        candidate_lengths = edge_lengths[candidate_groups]
        length_order = torch.argsort(candidate_lengths, descending=True)
        priorities = torch.empty_like(length_order)
        priorities[length_order] = torch.arange(
            len(length_order),
            dtype=torch.long,
            device=faces.device,
        )
        owner_priority_keys = (
            owner_faces * (len(candidate_groups) + 1)
            + priorities[owner_candidates]
        )
        owner_order = torch.argsort(owner_priority_keys)
        ordered_owner_faces = owner_faces[owner_order]
        first_owner = torch.ones(
            len(owner_order),
            dtype=torch.bool,
            device=faces.device,
        )
        first_owner[1:] = (
            ordered_owner_faces[1:] != ordered_owner_faces[:-1]
        )
        winners = owner_candidates[owner_order[first_owner]]
        win_counts = torch.bincount(
            winners,
            minlength=len(candidate_groups),
        )
        selected = torch.nonzero(
            win_counts == candidate_counts
        ).reshape(-1)
        if len(selected) == 0:
            break

        selected_groups = candidate_groups[selected]
        selected_edges = topology["edges"][selected_groups]
        new_vertex_ids = torch.arange(
            len(vertices),
            len(vertices) + len(selected),
            dtype=torch.long,
            device=faces.device,
        )
        new_vertices = 0.5 * (
            vertices[selected_edges[:, 0]]
            + vertices[selected_edges[:, 1]]
        )

        selected_counts = topology["counts"][selected_groups]
        selected_first_entries = topology["order"][
            topology["starts"][selected_groups]
        ]
        selected_second_mask = selected_counts == 2
        selected_second_entries = topology["order"][
            topology["starts"][
                selected_groups[selected_second_mask]
            ] + 1
        ]
        selected_ids = torch.arange(
            len(selected),
            dtype=torch.long,
            device=faces.device,
        )
        incidence_candidates = torch.cat(
            (selected_ids, selected_ids[selected_second_mask])
        )
        incidence_entries = torch.cat(
            (selected_first_entries, selected_second_entries)
        )
        incidence_faces = topology["face_ids"][incidence_entries]
        starts = topology["directed_edges"][incidence_entries, 0]
        ends = topology["directed_edges"][incidence_entries, 1]
        opposite = topology["opposite"][incidence_entries]
        midpoint_ids = new_vertex_ids[incidence_candidates]
        old_patch_ids = face_patch_ids[incidence_faces]
        old_source_ids = face_source_ids[incidence_faces]

        faces[incidence_faces] = torch.stack(
            (starts, midpoint_ids, opposite),
            dim=1,
        )
        faces = torch.cat(
            (
                faces,
                torch.stack(
                    (midpoint_ids, ends, opposite),
                    dim=1,
                ),
            ),
            dim=0,
        )
        face_patch_ids = torch.cat(
            (face_patch_ids, old_patch_ids)
        )
        face_source_ids = torch.cat(
            (face_source_ids, old_source_ids)
        )
        first_owner_faces = topology["face_ids"][
            selected_first_entries
        ]
        new_support_face_ids = face_source_ids[first_owner_faces].clone()
        protected_chain = (
            support_face_ids[selected_edges] < -1
        ).all(dim=1)
        hard_split_edges = selected_counts != 2
        if bool(selected_second_mask.any()):
            first_selected_faces = topology["face_ids"][
                selected_first_entries[selected_second_mask]
            ]
            second_selected_faces = topology["face_ids"][
                selected_second_entries
            ]
            hard_split_edges[selected_second_mask] = (
                face_patch_ids[first_selected_faces]
                != face_patch_ids[second_selected_faces]
            )
        new_support_face_ids[
            protected_chain | hard_split_edges
        ] = -2
        support_face_ids = torch.cat(
            (
                support_face_ids,
                new_support_face_ids,
            )
        )
        vertices = torch.cat((vertices, new_vertices), dim=0)
        total_splits += len(selected)

    topology = _gpu_edge_topology(faces, len(vertices))
    final_lengths = torch.linalg.norm(
        vertices[topology["edges"][:, 1]]
        - vertices[topology["edges"][:, 0]],
        dim=1,
    )
    final_groups = torch.nonzero(topology["counts"] <= 2).reshape(-1)
    final_limits = torch.full_like(final_lengths, maximum_edge_length)
    if source_face_maximum_edge_lengths is not None and len(final_groups):
        final_counts = topology["counts"][final_groups]
        first_entries = topology["order"][topology["starts"][final_groups]]
        final_limits[final_groups] = source_face_maximum_edge_lengths[
            face_source_ids[topology["face_ids"][first_entries]]
        ]
        paired = final_counts == 2
        if bool(paired.any()):
            second_entries = topology["order"][
                topology["starts"][final_groups[paired]] + 1
            ]
            final_limits[final_groups[paired]] = torch.minimum(
                final_limits[final_groups[paired]],
                source_face_maximum_edge_lengths[
                    face_source_ids[topology["face_ids"][second_entries]]
                ],
            )
    remaining = int(
        (final_lengths[final_groups] > final_limits[final_groups]).sum().item()
    )
    return (
        vertices,
        faces,
        face_patch_ids,
        face_source_ids,
        support_face_ids,
        total_splits,
        remaining,
    )


def _gpu_vertex_ragged_pairs(edges, candidate_edges, vertex_count):
    """
    Return unique (candidate, neighboring vertex) pairs and common counts.

    The common count is used for the manifold edge-collapse link condition.
    """
    adjacency_vertices = torch.cat((edges[:, 0], edges[:, 1]))
    adjacency_neighbors = torch.cat((edges[:, 1], edges[:, 0]))
    adjacency_order = torch.argsort(adjacency_vertices)
    adjacency_vertices = adjacency_vertices[adjacency_order]
    adjacency_neighbors = adjacency_neighbors[adjacency_order]
    degrees = torch.bincount(
        adjacency_vertices,
        minlength=vertex_count,
    )
    starts = torch.cumsum(degrees, dim=0) - degrees

    first_degrees = degrees[candidate_edges[:, 0]]
    second_degrees = degrees[candidate_edges[:, 1]]
    totals = first_degrees + second_degrees
    candidate_ids = torch.repeat_interleave(
        torch.arange(
            len(candidate_edges),
            dtype=torch.long,
            device=edges.device,
        ),
        totals,
    )
    repeated_starts = torch.repeat_interleave(
        torch.cumsum(totals, dim=0) - totals,
        totals,
    )
    local_offsets = torch.arange(
        int(totals.sum().item()),
        dtype=torch.long,
        device=edges.device,
    ) - repeated_starts
    repeated_first_degrees = first_degrees[candidate_ids]
    from_first = local_offsets < repeated_first_degrees
    adjacency_offsets = torch.where(
        from_first,
        starts[candidate_edges[candidate_ids, 0]] + local_offsets,
        starts[candidate_edges[candidate_ids, 1]]
        + local_offsets
        - repeated_first_degrees,
    )
    neighbors = adjacency_neighbors[adjacency_offsets]
    pair_keys = candidate_ids * int(vertex_count) + neighbors
    sorted_pair_keys = torch.sort(pair_keys).values
    unique_pair_keys, pair_counts = torch.unique_consecutive(
        sorted_pair_keys,
        return_counts=True,
    )
    unique_pairs = torch.stack(
        (
            torch.div(
                unique_pair_keys,
                int(vertex_count),
                rounding_mode="floor",
            ),
            unique_pair_keys % int(vertex_count),
        ),
        dim=1,
    )
    common_counts = torch.bincount(
        unique_pairs[pair_counts == 2, 0],
        minlength=len(candidate_edges),
    )
    return unique_pairs, common_counts


def _gpu_candidate_face_pairs(faces, candidate_edges):
    """Expand candidate endpoints to unique candidate/incident-face pairs."""
    vertex_count = int(faces.max().item()) + 1
    flat_vertices = faces.reshape(-1)
    flat_face_ids = torch.arange(
        len(faces),
        dtype=torch.long,
        device=faces.device,
    ).repeat_interleave(3)
    order = torch.argsort(flat_vertices)
    sorted_vertices = flat_vertices[order]
    sorted_face_ids = flat_face_ids[order]
    degrees = torch.bincount(
        sorted_vertices,
        minlength=vertex_count,
    )
    starts = torch.cumsum(degrees, dim=0) - degrees

    first_degrees = degrees[candidate_edges[:, 0]]
    second_degrees = degrees[candidate_edges[:, 1]]
    totals = first_degrees + second_degrees
    candidate_ids = torch.repeat_interleave(
        torch.arange(
            len(candidate_edges),
            dtype=torch.long,
            device=faces.device,
        ),
        totals,
    )
    repeated_starts = torch.repeat_interleave(
        torch.cumsum(totals, dim=0) - totals,
        totals,
    )
    local_offsets = torch.arange(
        int(totals.sum().item()),
        dtype=torch.long,
        device=faces.device,
    ) - repeated_starts
    repeated_first_degrees = first_degrees[candidate_ids]
    from_first = local_offsets < repeated_first_degrees
    face_offsets = torch.where(
        from_first,
        starts[candidate_edges[candidate_ids, 0]] + local_offsets,
        starts[candidate_edges[candidate_ids, 1]]
        + local_offsets
        - repeated_first_degrees,
    )
    incident_faces = sorted_face_ids[face_offsets]
    pair_keys = candidate_ids * len(faces) + incident_faces
    unique_pair_keys = torch.unique(pair_keys, sorted=True)
    return torch.stack(
        (
            torch.div(
                unique_pair_keys,
                len(faces),
                rounding_mode="floor",
            ),
            unique_pair_keys % len(faces),
        ),
        dim=1,
    )


def _gpu_evaluate_collapse_direction(
    vertices,
    faces,
    candidate_edges,
    candidate_face_pairs,
    remove_first,
    maximum_edge_length,
    face_source_ids=None,
    source_face_origins=None,
    source_face_normals=None,
    maximum_surface_deviation=None,
    strict_quality=False,
    reference_patch_projector=None,
    face_patch_ids=None,
    maximum_normal_deviation_degrees=None,
    triangle_validation_cache=None,
    allowed_directions=None,
):
    """Validate one endpoint choice for every short-edge candidate."""
    candidate_ids = candidate_face_pairs[:, 0]
    face_ids = candidate_face_pairs[:, 1]
    local_faces = faces[face_ids]
    remove = torch.where(
        remove_first,
        candidate_edges[:, 0],
        candidate_edges[:, 1],
    )
    keep = torch.where(
        remove_first,
        candidate_edges[:, 1],
        candidate_edges[:, 0],
    )
    local_remove = remove[candidate_ids]
    local_keep = keep[candidate_ids]
    proposed_faces = torch.where(
        local_faces == local_remove[:, None],
        local_keep[:, None],
        local_faces,
    )
    nondegenerate = (
        (proposed_faces[:, 0] != proposed_faces[:, 1])
        & (proposed_faces[:, 1] != proposed_faces[:, 2])
        & (proposed_faces[:, 2] != proposed_faces[:, 0])
    )
    active_candidates = candidate_ids[nondegenerate]
    old_faces = local_faces[nondegenerate]
    new_faces = proposed_faces[nondegenerate]
    old_cross = _triangle_cross_torch(vertices, old_faces)
    new_cross = _triangle_cross_torch(vertices, new_faces)
    old_cross_squared = (old_cross * old_cross).sum(dim=1)
    new_cross_squared = (new_cross * new_cross).sum(dim=1)
    diameter = (
        vertices.amax(dim=0) - vertices.amin(dim=0)
    ).amax().clamp_min(torch.finfo(vertices.dtype).eps)
    minimum_cross_squared = (diameter * diameter * 1e-12) ** 2
    orientation = (old_cross * new_cross).sum(dim=1)
    new_triangles = vertices[new_faces]
    new_edge_lengths = torch.linalg.norm(
        new_triangles[:, (1, 2, 0)]
        - new_triangles[:, (0, 1, 2)],
        dim=2,
    )
    old_triangles = vertices[old_faces]
    old_edge_lengths = torch.linalg.norm(
        old_triangles[:, (1, 2, 0)]
        - old_triangles[:, (0, 1, 2)],
        dim=2,
    )
    old_quality = _triangle_quality_torch(vertices, old_faces)
    new_quality = _triangle_quality_torch(vertices, new_faces)
    surface_deviation_invalid = torch.zeros(
        len(new_faces),
        dtype=torch.bool,
        device=faces.device,
    )
    if (
        reference_patch_projector is None
        and maximum_surface_deviation is not None
        and face_source_ids is not None
        and source_face_origins is not None
        and source_face_normals is not None
    ):
        active_source_ids = face_source_ids[face_ids[nondegenerate]]
        source_origins = source_face_origins[active_source_ids]
        source_normals = source_face_normals[active_source_ids]
        plane_distances = (
            (
                new_triangles
                - source_origins[:, None, :]
            )
            * source_normals[:, None, :]
        ).sum(dim=2).abs()
        surface_deviation_invalid = (
            plane_distances.amax(dim=1)
            > float(maximum_surface_deviation)
        )
    if torch.is_tensor(maximum_edge_length):
        maximum_allowed_length = maximum_edge_length.to(
            dtype=vertices.dtype,
            device=vertices.device,
        )[active_candidates]
    else:
        maximum_allowed_length = torch.full_like(
            old_edge_lengths[:, 0],
            float(maximum_edge_length),
        )
    if not strict_quality:
        maximum_allowed_length = torch.maximum(
            maximum_allowed_length,
            old_edge_lengths.amax(dim=1),
        )
    invalid_face = (
        ~torch.isfinite(new_edge_lengths).all(dim=1)
        | (new_cross_squared <= minimum_cross_squared)
        | (orientation <= 0.0)
        | surface_deviation_invalid
        | (
            new_edge_lengths.amax(dim=1)
            > maximum_allowed_length * (1.0 + 1e-8)
        )
        | ((new_quality < old_quality * 0.5) & (not strict_quality))
    )
    invalid_counts = torch.zeros(
        len(candidate_edges),
        dtype=torch.long,
        device=faces.device,
    )
    invalid_counts.index_add_(
        0,
        active_candidates,
        invalid_face.to(torch.long),
    )
    energy_delta = torch.zeros(
        len(candidate_edges),
        dtype=vertices.dtype,
        device=faces.device,
    )
    removed_candidates = candidate_ids[~nondegenerate]
    if len(removed_candidates):
        removed_quality = _triangle_quality_torch(
            vertices,
            local_faces[~nondegenerate],
        )
        energy_delta.index_add_(
            0,
            removed_candidates,
            -((1.0 - removed_quality) ** 2),
        )
    energy_delta.index_add_(
        0,
        active_candidates,
        (1.0 - new_quality) ** 2 - (1.0 - old_quality) ** 2,
    )
    active_counts = torch.bincount(
        active_candidates,
        minlength=len(candidate_edges),
    )
    valid = (
        (invalid_counts == 0)
        & (energy_delta <= 0.0)
    )
    new_minimum = torch.ones(
        len(candidate_edges),
        dtype=vertices.dtype,
        device=vertices.device,
    )
    new_minimum.scatter_reduce_(
        0,
        active_candidates,
        new_quality,
        reduce="amin",
        include_self=True,
    )
    new_sum = torch.zeros_like(new_minimum)
    new_sum.index_add_(0, active_candidates, new_quality)
    new_mean = new_sum / active_counts.clamp_min(1).to(vertices.dtype)
    if reference_patch_projector is not None and not strict_quality:
        old_sum = torch.zeros_like(new_sum)
        old_sum.index_add_(0, candidate_ids, _triangle_quality_torch(vertices, local_faces))
        old_counts = torch.bincount(candidate_ids, minlength=len(candidate_edges))
        old_mean = old_sum / old_counts.clamp_min(1).to(vertices.dtype)
        valid &= (active_counts > 0) & (new_mean >= old_mean - 1e-10)
    if strict_quality:
        old_candidate_ids = candidate_ids
        old_quality_all = _triangle_quality_torch(vertices, local_faces)
        old_minimum = torch.ones_like(new_minimum)
        old_minimum.scatter_reduce_(
            0,
            old_candidate_ids,
            old_quality_all,
            reduce="amin",
            include_self=True,
        )
        old_sum = torch.zeros_like(new_minimum)
        old_sum.index_add_(0, old_candidate_ids, old_quality_all)
        old_counts = torch.bincount(
            old_candidate_ids,
            minlength=len(candidate_edges),
        )
        old_mean = old_sum / old_counts.clamp_min(1).to(vertices.dtype)
        valid = (
            (invalid_counts == 0)
            & (active_counts > 0)
            & (new_minimum >= old_minimum - 1e-8)
            & (new_mean >= old_mean - 1e-8)
        )
    if allowed_directions is not None:
        valid &= allowed_directions
    if reference_patch_projector is not None:
        # Closest-surface queries are much more expensive than the cavity
        # checks above. Query only faces of candidates that can still pass;
        # every surviving candidate retains the same full surface checks.
        query_face_ids = torch.nonzero(valid[active_candidates]).reshape(-1)
        if len(query_face_ids):
            query_labels = face_patch_ids[face_ids[nondegenerate][query_face_ids]]
            if triangle_validation_cache is not None:
                surface_valid = triangle_validation_cache.check(new_faces[query_face_ids], query_labels)
            else:
                surface_valid = reference_patch_projector.valid_triangles(
                    new_triangles[query_face_ids], query_labels,
                    maximum_surface_deviation, maximum_normal_deviation_degrees,
                )
            rejected_candidates = active_candidates[query_face_ids[~surface_valid]]
            valid[torch.unique(rejected_candidates)] = False
    return valid, energy_delta, new_minimum, new_mean


def _gpu_collapse_short_edges(
    vertices,
    faces,
    face_patch_ids,
    face_source_ids,
    support_face_ids,
    minimum_edge_length,
    maximum_edge_length,
    passes,
    protected_source_face_mask=None,
    source_face_origins=None,
    source_face_normals=None,
    maximum_normal_deviation_degrees=None,
    maximum_surface_deviation=None,
    minimum_collapse_quality=0.0,
    protected_vertex_mask=None,
    strict_constraints=False,
    compact_vertices=True,
    source_face_target_lengths=None,
    reference_patch_projector=None,
):
    """
    Collapse independent short edges inside smooth patches on CUDA.

    Hard-edge and boundary vertices are locked. The endpoint retained by each
    collapse is already on the original STL, avoiding an off-surface midpoint.
    Link-condition, source-lineage protection, source-normal cone, surface
    deviation, orientation, quality, and maximum-edge checks are applied before
    each parallel batch.
    """
    minimum_edge_length = float(minimum_edge_length)
    maximum_edge_length = float(maximum_edge_length)
    passes = int(passes)
    minimum_collapse_quality = float(minimum_collapse_quality)
    if minimum_edge_length < 0.0:
        raise ValueError("Minimum surface edge length must be non-negative.")
    if maximum_edge_length <= 0.0:
        raise ValueError("Maximum surface edge length must be positive.")
    if passes < 0:
        raise ValueError("Surface collapse passes must be non-negative.")
    if not 0.0 <= minimum_collapse_quality <= 1.0:
        raise ValueError("Minimum collapse quality must be in [0, 1].")
    if (
        maximum_normal_deviation_degrees is not None
        and not 0.0 <= float(maximum_normal_deviation_degrees) <= 180.0
    ):
        raise ValueError(
            "Maximum collapse normal deviation must be in [0, 180] degrees."
        )
    if (
        maximum_surface_deviation is not None
        and float(maximum_surface_deviation) < 0.0
    ):
        raise ValueError(
            "Maximum collapse surface deviation must be non-negative."
        )
    if minimum_edge_length == 0.0 or passes == 0:
        return (
            vertices,
            faces,
            face_patch_ids,
            face_source_ids,
            support_face_ids,
            0,
        )

    vertices = vertices.clone()
    faces = faces.clone()
    face_patch_ids = face_patch_ids.clone()
    face_source_ids = face_source_ids.clone()
    support_face_ids = support_face_ids.clone()
    if protected_source_face_mask is not None:
        protected_source_face_mask = protected_source_face_mask.to(
            device=faces.device,
            dtype=torch.bool,
        )
    if source_face_origins is not None:
        source_face_origins = source_face_origins.to(
            device=vertices.device,
            dtype=vertices.dtype,
        )
    if source_face_normals is not None:
        source_face_normals = source_face_normals.to(
            device=vertices.device,
            dtype=vertices.dtype,
        )
    if source_face_target_lengths is not None:
        source_face_target_lengths = torch.as_tensor(
            source_face_target_lengths,
            dtype=vertices.dtype,
            device=vertices.device,
        )
    if protected_vertex_mask is not None:
        protected_vertex_mask = protected_vertex_mask.to(
            device=faces.device,
            dtype=torch.bool,
        )
        if len(protected_vertex_mask) != len(vertices):
            raise ValueError(
                "Protected vertex mask must match the vertex count."
            )
    total_collapses = 0
    # Endpoint collapses retain all coordinates when the vertex prefix is not
    # compacted. Repeated ordered triples can therefore share exact validation
    # across both directions and all passes in this call.
    triangle_validation_cache = (
        (reference_patch_projector.make_triangle_cache(vertices,
            maximum_surface_deviation, maximum_normal_deviation_degrees)
         if getattr(reference_patch_projector, "is_cuda_backend", False)
         else _IndexedTriangleValidationCache(vertices, reference_patch_projector,
            maximum_surface_deviation, maximum_normal_deviation_degrees)
        )
        if reference_patch_projector is not None and not compact_vertices
        else None
    )

    for _ in range(passes):
        topology = _gpu_edge_topology(faces, len(vertices))
        counts = topology["counts"]
        starts = topology["starts"]
        order = topology["order"]
        manifold_groups = torch.nonzero(counts == 2).reshape(-1)
        if len(manifold_groups) == 0:
            break
        first_entries = order[starts[manifold_groups]]
        second_entries = order[starts[manifold_groups] + 1]
        first_faces = topology["face_ids"][first_entries]
        second_faces = topology["face_ids"][second_entries]
        same_patch = (
            face_patch_ids[first_faces]
            == face_patch_ids[second_faces]
        )

        hard_groups = torch.ones(
            len(counts),
            dtype=torch.bool,
            device=faces.device,
        )
        hard_groups[manifold_groups] = ~same_patch
        locked = torch.zeros(
            len(vertices),
            dtype=torch.bool,
            device=faces.device,
        )
        locked |= support_face_ids < -1
        if protected_vertex_mask is not None:
            locked |= protected_vertex_mask
        hard_edges = topology["edges"][hard_groups]
        if len(hard_edges):
            locked[hard_edges.reshape(-1)] = True

        manifold_edges = topology["edges"][manifold_groups]
        edge_lengths = torch.linalg.norm(
            vertices[manifold_edges[:, 1]]
            - vertices[manifold_edges[:, 0]],
            dim=1,
        )
        local_minimum_edge_lengths = torch.full_like(
            edge_lengths, minimum_edge_length
        )
        local_maximum_edge_lengths = torch.full_like(
            edge_lengths, maximum_edge_length
        )
        if source_face_target_lengths is not None:
            local_targets = torch.minimum(
                source_face_target_lengths[face_source_ids[first_faces]],
                source_face_target_lengths[face_source_ids[second_faces]],
            )
            local_minimum_edge_lengths = 0.8 * local_targets
            local_maximum_edge_lengths = (4.0 / 3.0) * local_targets
        face_quality = _triangle_quality_torch(vertices, faces)
        adjacent_quality = torch.minimum(
            face_quality[first_faces],
            face_quality[second_faces],
        )
        if strict_constraints:
            locked_candidate = (
                locked[manifold_edges[:, 0]]
                | locked[manifold_edges[:, 1]]
            )
        else:
            locked_candidate = (
                locked[manifold_edges[:, 0]]
                & locked[manifold_edges[:, 1]]
            )
        candidate_mask = (
            same_patch
            & (
                (edge_lengths < local_minimum_edge_lengths)
                | (
                    adjacent_quality
                    < minimum_collapse_quality
                )
            )
            & ~locked_candidate
        )
        candidates = torch.nonzero(candidate_mask).reshape(-1)
        if len(candidates) == 0:
            break
        candidate_edges = manifold_edges[candidates]
        candidate_lengths = edge_lengths[candidates]
        candidate_patches = face_patch_ids[first_faces[candidates]]
        candidate_maximum_lengths = local_maximum_edge_lengths[candidates]

        footprint_pairs, common_counts = _gpu_vertex_ragged_pairs(
            topology["edges"],
            candidate_edges,
            len(vertices),
        )
        link_valid = common_counts == 2
        if not bool(link_valid.all()):
            old_to_new = torch.full(
                (len(candidate_edges),),
                -1,
                dtype=torch.long,
                device=faces.device,
            )
            valid_ids = torch.nonzero(link_valid).reshape(-1)
            old_to_new[valid_ids] = torch.arange(
                len(valid_ids),
                dtype=torch.long,
                device=faces.device,
            )
            keep_footprints = link_valid[footprint_pairs[:, 0]]
            footprint_pairs = footprint_pairs[keep_footprints]
            footprint_pairs[:, 0] = old_to_new[
                footprint_pairs[:, 0]
            ]
            candidate_edges = candidate_edges[link_valid]
            candidate_lengths = candidate_lengths[link_valid]
            candidate_patches = candidate_patches[link_valid]
            candidate_maximum_lengths = candidate_maximum_lengths[link_valid]
        if len(candidate_edges) == 0:
            break

        candidate_face_pairs = _gpu_candidate_face_pairs(
            faces,
            candidate_edges,
        )
        pair_candidates = candidate_face_pairs[:, 0]
        pair_faces = candidate_face_pairs[:, 1]
        patch_mismatch = (
            face_patch_ids[pair_faces]
            != candidate_patches[pair_candidates]
        )
        mismatch_counts = torch.zeros(
            len(candidate_edges),
            dtype=torch.long,
            device=faces.device,
        )
        mismatch_counts.index_add_(
            0,
            pair_candidates,
            patch_mismatch.to(torch.long),
        )
        topology_valid = mismatch_counts == 0
        pair_source_ids = face_source_ids[pair_faces]
        if protected_source_face_mask is not None:
            protected_counts = torch.zeros(
                len(candidate_edges),
                dtype=torch.long,
                device=faces.device,
            )
            protected_counts.index_add_(
                0,
                pair_candidates,
                protected_source_face_mask[pair_source_ids].to(torch.long),
            )
            topology_valid &= protected_counts == 0
        if (
            source_face_normals is not None
            and maximum_normal_deviation_degrees is not None
            and float(maximum_normal_deviation_degrees) < 180.0
            and reference_patch_projector is None
        ):
            pair_normals = source_face_normals[pair_source_ids]
            normal_sums = torch.zeros(
                (len(candidate_edges), 3),
                dtype=vertices.dtype,
                device=vertices.device,
            )
            normal_sums.index_add_(0, pair_candidates, pair_normals)
            cone_axes = normal_sums / torch.linalg.norm(
                normal_sums,
                dim=1,
                keepdim=True,
            ).clamp_min(torch.finfo(vertices.dtype).eps)
            alignments = (
                pair_normals * cone_axes[pair_candidates]
            ).sum(dim=1)
            minimum_alignment = torch.ones(
                len(candidate_edges),
                dtype=vertices.dtype,
                device=vertices.device,
            )
            minimum_alignment.scatter_reduce_(
                0,
                pair_candidates,
                alignments,
                reduce="amin",
                include_self=True,
            )
            cosine_limit = float(
                np.cos(
                    np.deg2rad(
                        float(maximum_normal_deviation_degrees)
                    )
                )
            )
            topology_valid &= minimum_alignment >= cosine_limit
        if not bool(topology_valid.all()):
            valid_ids = torch.nonzero(topology_valid).reshape(-1)
            old_to_new = torch.full(
                (len(candidate_edges),),
                -1,
                dtype=torch.long,
                device=faces.device,
            )
            old_to_new[valid_ids] = torch.arange(
                len(valid_ids),
                dtype=torch.long,
                device=faces.device,
            )
            keep_pairs = topology_valid[candidate_face_pairs[:, 0]]
            candidate_face_pairs = candidate_face_pairs[keep_pairs]
            candidate_face_pairs[:, 0] = old_to_new[
                candidate_face_pairs[:, 0]
            ]
            keep_footprints = topology_valid[footprint_pairs[:, 0]]
            footprint_pairs = footprint_pairs[keep_footprints]
            footprint_pairs[:, 0] = old_to_new[
                footprint_pairs[:, 0]
            ]
            candidate_edges = candidate_edges[topology_valid]
            candidate_lengths = candidate_lengths[topology_valid]
            candidate_maximum_lengths = candidate_maximum_lengths[
                topology_valid
            ]
        if len(candidate_edges) == 0:
            break

        remove_first = torch.ones(
            len(candidate_edges),
            dtype=torch.bool,
            device=faces.device,
        )
        (
            first_valid,
            first_energy,
            first_minimum,
            first_mean,
        ) = _gpu_evaluate_collapse_direction(
            vertices,
            faces,
            candidate_edges,
            candidate_face_pairs,
            remove_first,
            candidate_maximum_lengths,
            face_source_ids=face_source_ids,
            source_face_origins=source_face_origins,
            source_face_normals=source_face_normals,
            maximum_surface_deviation=maximum_surface_deviation,
            strict_quality=strict_constraints,
            reference_patch_projector=reference_patch_projector,
            face_patch_ids=face_patch_ids,
            maximum_normal_deviation_degrees=maximum_normal_deviation_degrees,
            triangle_validation_cache=triangle_validation_cache,
            allowed_directions=~locked[candidate_edges[:, 0]],
        )
        (
            second_valid,
            second_energy,
            second_minimum,
            second_mean,
        ) = _gpu_evaluate_collapse_direction(
            vertices,
            faces,
            candidate_edges,
            candidate_face_pairs,
            ~remove_first,
            candidate_maximum_lengths,
            face_source_ids=face_source_ids,
            source_face_origins=source_face_origins,
            source_face_normals=source_face_normals,
            maximum_surface_deviation=maximum_surface_deviation,
            strict_quality=strict_constraints,
            reference_patch_projector=reference_patch_projector,
            face_patch_ids=face_patch_ids,
            maximum_normal_deviation_degrees=maximum_normal_deviation_degrees,
            triangle_validation_cache=triangle_validation_cache,
            allowed_directions=~locked[candidate_edges[:, 1]],
        )
        first_valid &= ~locked[candidate_edges[:, 0]]
        second_valid &= ~locked[candidate_edges[:, 1]]
        direction_valid = first_valid | second_valid
        if strict_constraints:
            first_is_better = (
                (first_minimum > second_minimum)
                | (
                    (first_minimum == second_minimum)
                    & (first_mean >= second_mean)
                )
            )
            choose_first = first_valid & (~second_valid | first_is_better)
        else:
            choose_first = first_valid & (
                ~second_valid | (first_energy <= second_energy)
            )
        if not bool(direction_valid.all()):
            valid_ids = torch.nonzero(direction_valid).reshape(-1)
            old_to_new = torch.full(
                (len(candidate_edges),),
                -1,
                dtype=torch.long,
                device=faces.device,
            )
            old_to_new[valid_ids] = torch.arange(
                len(valid_ids),
                dtype=torch.long,
                device=faces.device,
            )
            keep_footprints = direction_valid[footprint_pairs[:, 0]]
            footprint_pairs = footprint_pairs[keep_footprints]
            footprint_pairs[:, 0] = old_to_new[
                footprint_pairs[:, 0]
            ]
            candidate_edges = candidate_edges[direction_valid]
            candidate_lengths = candidate_lengths[direction_valid]
            candidate_maximum_lengths = candidate_maximum_lengths[
                direction_valid
            ]
            choose_first = choose_first[direction_valid]
        if len(candidate_edges) == 0:
            break

        length_order = torch.argsort(candidate_lengths)
        priorities = torch.empty_like(length_order)
        priorities[length_order] = torch.arange(
            len(length_order),
            dtype=torch.long,
            device=faces.device,
        )
        selected = _gpu_independent_candidates(
            footprint_pairs,
            priorities,
            len(candidate_edges),
        )
        if len(selected) == 0:
            break
        selected_edges = candidate_edges[selected]
        selected_directions = choose_first[selected]
        remove = torch.where(
            selected_directions,
            selected_edges[:, 0],
            selected_edges[:, 1],
        )
        keep = torch.where(
            selected_directions,
            selected_edges[:, 1],
            selected_edges[:, 0],
        )
        vertex_map = torch.arange(
            len(vertices),
            dtype=torch.long,
            device=faces.device,
        )
        vertex_map[remove] = keep
        faces = vertex_map[faces]
        nondegenerate = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 2] != faces[:, 0])
        )
        faces = faces[nondegenerate]
        face_patch_ids = face_patch_ids[nondegenerate]
        face_source_ids = face_source_ids[nondegenerate]

        if compact_vertices:
            used = torch.zeros(
                len(vertices),
                dtype=torch.bool,
                device=faces.device,
            )
            used[faces.reshape(-1)] = True
            compact_map = torch.cumsum(used.to(torch.long), dim=0) - 1
            faces = compact_map[faces]
            vertices = vertices[used]
            support_face_ids = support_face_ids[used]
            if protected_vertex_mask is not None:
                protected_vertex_mask = protected_vertex_mask[used]
        total_collapses += len(selected)

    return (
        vertices,
        faces,
        face_patch_ids,
        face_source_ids,
        support_face_ids,
        total_collapses,
    )


def _gpu_flip_quality_edges(
    vertices,
    faces,
    face_patch_ids,
    passes,
    maximum_edge_length=None,
    protected_vertex_mask=None,
    face_source_ids=None,
    protected_source_face_mask=None,
    source_face_origins=None,
    source_face_normals=None,
    maximum_normal_deviation_degrees=None,
    maximum_surface_deviation=None,
    protected_edges=None,
    source_face_target_lengths=None,
    reference_patch_projector=None,
    preserve_existing_maximum_edge=False,
):
    """Flip conflict-free same-patch, source-safe edges on the GPU."""
    faces = faces.clone()
    face_patch_ids = face_patch_ids.to(device=faces.device, dtype=torch.long)
    if face_source_ids is not None:
        face_source_ids = face_source_ids.to(
            device=faces.device,
            dtype=torch.long,
        )
    if protected_source_face_mask is not None:
        protected_source_face_mask = protected_source_face_mask.to(
            device=faces.device,
            dtype=torch.bool,
        )
    if source_face_origins is not None:
        source_face_origins = source_face_origins.to(
            device=vertices.device,
            dtype=vertices.dtype,
        )
    if source_face_normals is not None:
        source_face_normals = source_face_normals.to(
            device=vertices.device,
            dtype=vertices.dtype,
        )
    if protected_edges is not None:
        protected_edges = torch.sort(
            protected_edges.to(device=faces.device, dtype=torch.long),
            dim=1,
        ).values
    if source_face_target_lengths is not None:
        source_face_target_lengths = torch.as_tensor(
            source_face_target_lengths,
            dtype=vertices.dtype,
            device=vertices.device,
        )
    if (
        maximum_normal_deviation_degrees is not None
        and not 0.0 <= float(maximum_normal_deviation_degrees) <= 180.0
    ):
        raise ValueError(
            "Maximum flip normal deviation must be in [0, 180] degrees."
        )
    if (
        maximum_surface_deviation is not None
        and float(maximum_surface_deviation) < 0.0
    ):
        raise ValueError(
            "Maximum flip surface deviation must be non-negative."
        )
    vertex_count = len(vertices)
    total_flips = 0
    epsilon = torch.finfo(vertices.dtype).eps
    diameter = (
        vertices.amax(dim=0) - vertices.amin(dim=0)
    ).amax().clamp_min(epsilon)
    minimum_cross_squared = (diameter * diameter * 1e-12) ** 2
    if protected_vertex_mask is not None:
        protected_vertex_mask = protected_vertex_mask.to(
            device=faces.device,
            dtype=torch.bool,
        )

    for _ in range(int(passes)):
        face_ids = torch.arange(
            len(faces),
            dtype=torch.long,
            device=faces.device,
        ).repeat(3)
        directed_edges = torch.cat(
            (
                faces[:, (0, 1)],
                faces[:, (1, 2)],
                faces[:, (2, 0)],
            ),
            dim=0,
        )
        opposite = torch.cat(
            (faces[:, 2], faces[:, 0], faces[:, 1]),
            dim=0,
        )
        sorted_edges = torch.sort(directed_edges, dim=1).values
        keys = sorted_edges[:, 0] * vertex_count + sorted_edges[:, 1]
        sorted_keys, order = torch.sort(keys)
        unique_keys, counts = torch.unique_consecutive(
            sorted_keys,
            return_counts=True,
        )
        starts = torch.cumsum(counts, dim=0) - counts
        manifold = torch.nonzero(counts == 2).reshape(-1)
        if len(manifold) == 0:
            break

        first_entries = order[starts[manifold]]
        second_entries = order[starts[manifold] + 1]
        first_faces = face_ids[first_entries]
        second_faces = face_ids[second_entries]
        first_start = directed_edges[first_entries, 0]
        first_end = directed_edges[first_entries, 1]
        second_start = directed_edges[second_entries, 0]
        second_end = directed_edges[second_entries, 1]
        first_opposite = opposite[first_entries]
        second_opposite = opposite[second_entries]
        coplanar_source_edge = torch.zeros(
            len(manifold),
            dtype=torch.bool,
            device=faces.device,
        )
        if protected_edges is not None and len(protected_edges):
            current_edge_keys = (
                torch.minimum(first_start, first_end) * vertex_count
                + torch.maximum(first_start, first_end)
            )
            protected_edge_keys = (
                protected_edges[:, 0] * vertex_count
                + protected_edges[:, 1]
            )
            coplanar_source_edge = torch.isin(
                current_edge_keys,
                protected_edge_keys,
            )

        valid = (
            (second_start == first_end)
            & (second_end == first_start)
            & (first_opposite != second_opposite)
            & (
                face_patch_ids[first_faces]
                == face_patch_ids[second_faces]
            )
        )
        if (
            face_source_ids is not None
            and protected_source_face_mask is not None
        ):
            first_sources = face_source_ids[first_faces]
            second_sources = face_source_ids[second_faces]
            valid &= ~(
                protected_source_face_mask[first_sources]
                | protected_source_face_mask[second_sources]
            )
        if (
            face_source_ids is not None
            and source_face_normals is not None
            and maximum_normal_deviation_degrees is not None
            and float(maximum_normal_deviation_degrees) < 90.0
            and reference_patch_projector is None
        ):
            first_sources = face_source_ids[first_faces]
            second_sources = face_source_ids[second_faces]
            source_alignment = (
                source_face_normals[first_sources]
                * source_face_normals[second_sources]
            ).sum(dim=1)
            pairwise_limit = min(
                180.0,
                2.0 * float(maximum_normal_deviation_degrees),
            )
            valid &= source_alignment >= float(
                np.cos(np.deg2rad(pairwise_limit))
            )
        if protected_vertex_mask is not None:
            valid &= ~(
                protected_vertex_mask[first_start]
                & protected_vertex_mask[first_end]
            )
        if maximum_edge_length is not None:
            proposed_edge_lengths = torch.linalg.norm(
                vertices[first_opposite]
                - vertices[second_opposite],
                dim=1,
            )
            if source_face_target_lengths is None:
                proposed_maximum_lengths = torch.full_like(
                    proposed_edge_lengths, float(maximum_edge_length)
                )
            else:
                proposed_maximum_lengths = (4.0 / 3.0) * torch.minimum(
                    source_face_target_lengths[face_source_ids[first_faces]],
                    source_face_target_lengths[face_source_ids[second_faces]],
                )
            if preserve_existing_maximum_edge:
                first_triangles = vertices[faces[first_faces]]
                second_triangles = vertices[faces[second_faces]]
                old_maximum = torch.maximum(
                    torch.linalg.norm(first_triangles - first_triangles.roll(1, dims=1), dim=2).amax(dim=1),
                    torch.linalg.norm(second_triangles - second_triangles.roll(1, dims=1), dim=2).amax(dim=1),
                )
                proposed_maximum_lengths = torch.maximum(proposed_maximum_lengths, old_maximum)
            valid &= proposed_edge_lengths <= proposed_maximum_lengths * (1.0 + 1e-6)
        new_edge_start = torch.minimum(first_opposite, second_opposite)
        new_edge_end = torch.maximum(first_opposite, second_opposite)
        new_keys = new_edge_start * vertex_count + new_edge_end
        locations = torch.searchsorted(unique_keys, new_keys)
        clamped_locations = locations.clamp_max(len(unique_keys) - 1)
        new_edge_exists = (
            (locations < len(unique_keys))
            & (unique_keys[clamped_locations] == new_keys)
        )
        valid &= ~new_edge_exists
        candidate_ids = torch.nonzero(valid).reshape(-1)
        if len(candidate_ids) == 0:
            break

        first_faces = first_faces[candidate_ids]
        second_faces = second_faces[candidate_ids]
        first_start = first_start[candidate_ids]
        first_end = first_end[candidate_ids]
        first_opposite = first_opposite[candidate_ids]
        second_opposite = second_opposite[candidate_ids]
        replacement_first = torch.stack(
            (first_opposite, first_start, second_opposite),
            dim=1,
        )
        replacement_second = torch.stack(
            (first_opposite, second_opposite, first_end),
            dim=1,
        )

        old_first = faces[first_faces]
        old_second = faces[second_faces]
        old_cross_first = _triangle_cross_torch(vertices, old_first)
        old_cross_second = _triangle_cross_torch(vertices, old_second)
        new_cross_first = _triangle_cross_torch(
            vertices,
            replacement_first,
        )
        new_cross_second = _triangle_cross_torch(
            vertices,
            replacement_second,
        )
        patch_normal = old_cross_first + old_cross_second
        geometry_valid = (
            ((new_cross_first * patch_normal).sum(dim=1) > 0.0)
            & ((new_cross_second * patch_normal).sum(dim=1) > 0.0)
            & (
                (new_cross_first * new_cross_first).sum(dim=1)
                > minimum_cross_squared
            )
            & (
                (new_cross_second * new_cross_second).sum(dim=1)
                > minimum_cross_squared
            )
        )
        if (
            maximum_surface_deviation is not None
            and face_source_ids is not None
            and source_face_origins is not None
            and source_face_normals is not None
            and reference_patch_projector is None
        ):
            first_sources = face_source_ids[first_faces]
            second_sources = face_source_ids[second_faces]
            first_distances = (
                (
                    vertices[replacement_first]
                    - source_face_origins[first_sources, None, :]
                )
                * source_face_normals[first_sources, None, :]
            ).sum(dim=2).abs()
            second_distances = (
                (
                    vertices[replacement_second]
                    - source_face_origins[second_sources, None, :]
                )
                * source_face_normals[second_sources, None, :]
            ).sum(dim=2).abs()
            geometry_valid &= (
                first_distances.amax(dim=1)
                <= float(maximum_surface_deviation)
            ) & (
                second_distances.amax(dim=1)
                <= float(maximum_surface_deviation)
            )
        old_quality_first = _triangle_quality_torch(vertices, old_first)
        old_quality_second = _triangle_quality_torch(vertices, old_second)
        new_quality_first = _triangle_quality_torch(
            vertices,
            replacement_first,
        )
        new_quality_second = _triangle_quality_torch(
            vertices,
            replacement_second,
        )
        old_minimum = torch.minimum(
            old_quality_first,
            old_quality_second,
        )
        new_minimum = torch.minimum(
            new_quality_first,
            new_quality_second,
        )
        quality_valid = (
            (new_minimum > old_minimum + 1e-8)
            & (
                new_quality_first + new_quality_second
                >= old_quality_first + old_quality_second - 1e-10
            )
        )
        if bool(coplanar_source_edge[candidate_ids].any()):
            coplanar_candidates = coplanar_source_edge[candidate_ids]
            old_edge_start = torch.minimum(first_start, first_end)
            old_edge_end = torch.maximum(first_start, first_end)
            new_edge_start = torch.minimum(first_opposite, second_opposite)
            new_edge_end = torch.maximum(first_opposite, second_opposite)
            old_keys = old_edge_start * vertex_count + old_edge_end
            new_keys = new_edge_start * vertex_count + new_edge_end
            equal_or_better = (
                (new_minimum >= old_minimum - 1e-8)
                & (
                    new_quality_first + new_quality_second
                    >= old_quality_first + old_quality_second - 1e-8
                )
                & (new_keys > old_keys)
            )
            quality_valid |= coplanar_candidates & equal_or_better
        if reference_patch_projector is not None:
            provisional = torch.nonzero(geometry_valid & quality_valid).reshape(-1)
            if len(provisional):
                new_triangles = vertices[torch.cat((replacement_first[provisional],
                                                   replacement_second[provisional]))]
                labels = torch.cat((face_patch_ids[first_faces[provisional]],
                                    face_patch_ids[second_faces[provisional]]))
                surface_valid = reference_patch_projector.valid_triangles(
                    new_triangles, labels, maximum_surface_deviation,
                    maximum_normal_deviation_degrees,
                ).reshape(2, -1).all(dim=0)
                geometry_valid[provisional] &= surface_valid
        valid_candidates = torch.nonzero(
            geometry_valid & quality_valid
        ).reshape(-1)
        if len(valid_candidates) == 0:
            break

        first_faces = first_faces[valid_candidates]
        second_faces = second_faces[valid_candidates]
        first_start = first_start[valid_candidates]
        first_end = first_end[valid_candidates]
        first_opposite = first_opposite[valid_candidates]
        second_opposite = second_opposite[valid_candidates]
        replacement_first = replacement_first[valid_candidates]
        replacement_second = replacement_second[valid_candidates]

        conflict_vertices = torch.stack(
            (
                first_start,
                first_end,
                first_opposite,
                second_opposite,
            ),
            dim=1,
        )
        local_ids = torch.arange(
            len(conflict_vertices),
            dtype=torch.long,
            device=faces.device,
        )
        best = torch.full(
            (vertex_count,),
            len(conflict_vertices),
            dtype=torch.long,
            device=faces.device,
        )
        best.scatter_reduce_(
            0,
            conflict_vertices.reshape(-1),
            local_ids[:, None].expand(-1, 4).reshape(-1),
            reduce="amin",
            include_self=True,
        )
        selected = (
            best[conflict_vertices] == local_ids[:, None]
        ).all(dim=1)
        selected_ids = torch.nonzero(selected).reshape(-1)
        if len(selected_ids) == 0:
            break

        faces[first_faces[selected_ids]] = replacement_first[selected_ids]
        faces[second_faces[selected_ids]] = replacement_second[selected_ids]
        total_flips += int(len(selected_ids))

    return faces, total_flips


def _project_to_support_triangles(points, support_triangles):
    """Project points to supporting triangle planes and clamp barycentrics."""
    a = support_triangles[:, 0]
    ab = support_triangles[:, 1] - a
    ac = support_triangles[:, 2] - a
    ap = points - a
    d00 = (ab * ab).sum(dim=1)
    d01 = (ab * ac).sum(dim=1)
    d11 = (ac * ac).sum(dim=1)
    d20 = (ap * ab).sum(dim=1)
    d21 = (ap * ac).sum(dim=1)
    denominator = d00 * d11 - d01 * d01
    epsilon = torch.finfo(points.dtype).eps
    v = (d11 * d20 - d01 * d21) / denominator.clamp_min(epsilon)
    w = (d00 * d21 - d01 * d20) / denominator.clamp_min(epsilon)
    barycentric = torch.stack((1.0 - v - w, v, w), dim=1)
    barycentric = barycentric.clamp_min(1e-6)
    barycentric = barycentric / barycentric.sum(dim=1, keepdim=True)
    return (support_triangles * barycentric[:, :, None]).sum(dim=1)


def _gpu_source_sensitive_vertex_mask(
    faces,
    face_source_ids,
    source_face_normals,
    protected_source_face_mask=None,
    maximum_normal_deviation_degrees=None,
    vertex_count=None,
):
    """
    Mark vertices whose incident source lineage must not be optimized.

    Vertices touching a refinement-only source face are always marked. Vertices
    shared by source faces outside the requested normal cone are also marked so
    relaxation cannot pull a source-edge sample into just one side of a rounded
    region.
    """
    if vertex_count is None:
        vertex_count = int(faces.max().item()) + 1 if len(faces) else 0
    sensitive = torch.zeros(
        vertex_count,
        dtype=torch.bool,
        device=faces.device,
    )
    face_source_ids = face_source_ids.to(
        device=faces.device,
        dtype=torch.long,
    )
    if protected_source_face_mask is not None:
        protected_source_face_mask = protected_source_face_mask.to(
            device=faces.device,
            dtype=torch.bool,
        )
        protected_faces = protected_source_face_mask[face_source_ids]
        if bool(protected_faces.any()):
            sensitive[faces[protected_faces].reshape(-1)] = True
    if (
        source_face_normals is None
        or maximum_normal_deviation_degrees is None
        or float(maximum_normal_deviation_degrees) >= 180.0
        or len(faces) == 0
    ):
        return sensitive

    source_face_normals = source_face_normals.to(
        device=faces.device,
    )
    incidence_vertices = faces.reshape(-1)
    incidence_sources = face_source_ids[:, None].expand(-1, 3).reshape(-1)
    incidence_normals = source_face_normals[incidence_sources]
    normal_sums = torch.zeros(
        (vertex_count, 3),
        dtype=incidence_normals.dtype,
        device=faces.device,
    )
    normal_sums.index_add_(0, incidence_vertices, incidence_normals)
    cone_axes = normal_sums / torch.linalg.norm(
        normal_sums,
        dim=1,
        keepdim=True,
    ).clamp_min(torch.finfo(incidence_normals.dtype).eps)
    alignments = (
        incidence_normals * cone_axes[incidence_vertices]
    ).sum(dim=1)
    minimum_alignment = torch.ones(
        vertex_count,
        dtype=incidence_normals.dtype,
        device=faces.device,
    )
    minimum_alignment.scatter_reduce_(
        0,
        incidence_vertices,
        alignments,
        reduce="amin",
        include_self=True,
    )
    cosine_limit = float(
        np.cos(
            np.deg2rad(float(maximum_normal_deviation_degrees))
        )
    )
    sensitive |= minimum_alignment < cosine_limit
    return sensitive


def _gpu_relax_inserted_vertices(
    vertices,
    faces,
    original_vertices,
    original_faces,
    support_face_ids,
    iterations,
    smoothing_step,
    protected_source_face_mask=None,
    protected_vertex_mask=None,
    maximum_edge_length=None,
):
    """Move unprotected inserted points on their original source triangles."""
    current = vertices.clone()
    support_face_ids = support_face_ids.to(
        device=vertices.device,
        dtype=torch.long,
    )
    movable_mask = support_face_ids >= 0
    if protected_source_face_mask is not None:
        protected_source_face_mask = protected_source_face_mask.to(
            device=vertices.device,
            dtype=torch.bool,
        )
        supported = torch.nonzero(movable_mask).reshape(-1)
        movable_mask[supported] &= ~protected_source_face_mask[
            support_face_ids[supported]
        ]
    if protected_vertex_mask is not None:
        movable_mask &= ~protected_vertex_mask.to(
            device=vertices.device,
            dtype=torch.bool,
        )
    movable = torch.nonzero(movable_mask).reshape(-1)
    if len(movable) == 0:
        return current, 0

    support_triangles = original_vertices[
        original_faces[support_face_ids[movable]]
    ]
    accepted_iterations = 0
    current_energy = _mesh_energy_torch(current, faces)
    for _ in range(int(iterations)):
        edges = _unique_edges_torch(faces)
        sums = torch.zeros_like(current)
        counts = torch.zeros(
            (len(current), 1),
            dtype=current.dtype,
            device=current.device,
        )
        ones = torch.ones(
            (len(edges), 1),
            dtype=current.dtype,
            device=current.device,
        )
        sums.index_add_(0, edges[:, 0], current[edges[:, 1]])
        sums.index_add_(0, edges[:, 1], current[edges[:, 0]])
        counts.index_add_(0, edges[:, 0], ones)
        counts.index_add_(0, edges[:, 1], ones)
        centroids = sums / counts.clamp_min(1.0)

        accepted = False
        step = float(smoothing_step)
        old_cross = _triangle_cross_torch(current, faces)
        for _ in range(8):
            proposed = current.clone()
            raw = current[movable] + step * (
                centroids[movable] - current[movable]
            )
            proposed[movable] = _project_to_support_triangles(
                raw,
                support_triangles,
            )
            new_cross = _triangle_cross_torch(proposed, faces)
            orientation = (old_cross * new_cross).sum(dim=1)
            valid = bool(
                torch.isfinite(proposed).all()
                and (orientation > 0.0).all()
                and (
                    (new_cross * new_cross).sum(dim=1)
                    > torch.finfo(vertices.dtype).eps
                ).all()
            )
            if valid and maximum_edge_length is not None:
                proposed_lengths = torch.linalg.norm(
                    proposed[edges[:, 1]] - proposed[edges[:, 0]], dim=1,
                )
                valid = bool(
                    (proposed_lengths <= float(maximum_edge_length) * (1.0 + 1e-6)).all()
                )
            proposed_energy = _mesh_energy_torch(proposed, faces)
            if valid and bool(proposed_energy < current_energy):
                current = proposed
                current_energy = proposed_energy
                accepted = True
                accepted_iterations += 1
                break
            step *= 0.5
        if not accepted:
            break
    return current, accepted_iterations


class _IndexedTriangleValidationCache:
    """Bounded exact surface checks for a fixed-coordinate collapse phase.

    Keys contain the patch ID and all three ordered vertex IDs. They are full
    integer records, not rounded coordinates or hashes. Rejected triangles are
    cached as well, and mutation of the vertex tensor invalidates every entry.
    """

    def __init__(self, vertices, projector, maximum_deviation, normal_degrees,
                 maximum_entries=1000000):
        self.vertices = vertices
        self.vertex_version = vertices._version
        self.projector = projector
        self.maximum_deviation = maximum_deviation
        self.normal_degrees = normal_degrees
        self.maximum_entries = maximum_entries
        self.keys = np.empty(0, dtype='V32')
        self.valid = np.empty(0, dtype=bool)

    def check(self, faces, patch_ids):
        if self.vertices._version != self.vertex_version:
            self.keys = np.empty(0, dtype='V32')
            self.valid = np.empty(0, dtype=bool)
            self.vertex_version = self.vertices._version
        records = np.empty((len(faces), 4), dtype=np.int64)
        records[:, 0] = patch_ids.detach().cpu().numpy()
        records[:, 1:] = faces.detach().cpu().numpy()
        keys, first, inverse = np.unique(records.view('V32').reshape(-1),
                                         return_index=True, return_inverse=True)
        locations = np.searchsorted(self.keys, keys)
        hit = locations < len(self.keys)
        hit[hit] &= self.keys[locations[hit]] == keys[hit]
        valid = np.empty(len(keys), dtype=bool)
        valid[hit] = self.valid[locations[hit]]
        missing = ~hit
        if np.any(missing):
            rows = torch.as_tensor(first[missing], device=faces.device, dtype=torch.long)
            checked = self.projector.valid_triangles(
                self.vertices[faces[rows]], patch_ids[rows],
                self.maximum_deviation, self.normal_degrees,
            ).detach().cpu().numpy()
            valid[missing] = checked
            # Cache eviction changes cost only; every missing entry is checked
            # in full before use. Never grow a cache across batches or models.
            if len(self.keys) + len(checked) > self.maximum_entries:
                self.keys = keys[:self.maximum_entries].copy()
                self.valid = valid[:self.maximum_entries].copy()
            else:
                combined = np.concatenate((self.keys, keys[missing]))
                order = np.argsort(combined)
                self.keys = combined[order]
                self.valid = np.concatenate((self.valid, checked))[order]
        return torch.as_tensor(valid[inverse], device=faces.device, dtype=torch.bool)


class _TriangleValidationCache:
    """Reuse surface checks only while all three ordered vertices are identical.

    Relaxation repeatedly rolls back local proposals. Its topology, labels and
    tolerances stay fixed, so unchanged triangles have exactly the same surface
    result. This cache lives for one relaxation call and never skips the final
    independent whole-output validation.
    """

    def __init__(self, projector, patch_ids, maximum_deviation, normal_degrees):
        self.projector = projector
        self.patch_ids = patch_ids.clone()
        self.maximum_deviation = maximum_deviation
        self.normal_degrees = normal_degrees
        self.triangles = None
        self.valid = None

    def check(self, triangles):
        if self.triangles is None:
            self.valid = self.projector.valid_triangles(
                triangles, self.patch_ids, self.maximum_deviation, self.normal_degrees,
            ).clone()
        else:
            changed = torch.nonzero((triangles != self.triangles).any(dim=2).any(dim=1)).reshape(-1)
            if len(changed):
                self.valid[changed] = self.projector.valid_triangles(
                    triangles[changed], self.patch_ids[changed],
                    self.maximum_deviation, self.normal_degrees,
                )
        self.triangles = triangles.clone()
        # Callers combine the result with iteration-dependent orientation and
        # length checks in place; these must not modify the cached surface mask.
        return self.valid.clone()


def _gpu_relax_patch_vertices(
    vertices, faces, face_patch_ids, support_face_ids, reference_patch_projector,
    iterations, smoothing_step, maximum_edge_length,
    maximum_surface_deviation, maximum_normal_deviation_degrees,
):
    """Relax all used interior vertices on their complete reference patch.

    Original vertices and points inserted on old source edges are eligible.
    Only explicit fixed vertices and vertices incident to multiple patch IDs
    are immovable. The original vertex table is never reordered or compacted.
    """
    current = vertices.clone()
    vertex_ids = faces.reshape(-1)
    incidence_labels = face_patch_ids[:, None].expand(-1, 3).reshape(-1)
    minimum_labels = torch.full((len(vertices),), torch.iinfo(torch.long).max,
                                dtype=torch.long, device=faces.device)
    maximum_labels = torch.full_like(minimum_labels, -1)
    minimum_labels.scatter_reduce_(0, vertex_ids, incidence_labels, reduce="amin", include_self=True)
    maximum_labels.scatter_reduce_(0, vertex_ids, incidence_labels, reduce="amax", include_self=True)
    movable_mask = (minimum_labels == maximum_labels) & (support_face_ids >= -1)
    movable = torch.nonzero(movable_mask).reshape(-1)
    if len(movable) == 0 or int(iterations) <= 0:
        return current, 0
    affected_faces = torch.nonzero(movable_mask[faces].any(dim=1)).reshape(-1)
    edges = _unique_edges_torch(faces)
    counts = torch.zeros((len(current), 1), dtype=current.dtype, device=current.device)
    ones = torch.ones((len(edges), 1), dtype=current.dtype, device=current.device)
    counts.index_add_(0, edges[:, 0], ones)
    counts.index_add_(0, edges[:, 1], ones)
    diameter = (vertices.amax(dim=0) - vertices.amin(dim=0)).amax()
    minimum_cross_squared = (diameter * diameter * 1e-12) ** 2
    current_energy = _mesh_energy_torch(current, faces)
    _, patch_inverse, patch_counts = torch.unique(face_patch_ids, return_inverse=True, return_counts=True)

    def patch_mean_quality(points):
        sums = torch.zeros(len(patch_counts), dtype=points.dtype, device=points.device)
        sums.index_add_(0, patch_inverse, _triangle_quality_torch(points, faces))
        return sums / patch_counts.to(points.dtype)

    current_mean_quality = patch_mean_quality(current)
    surface_checks = _TriangleValidationCache(
        reference_patch_projector, face_patch_ids[affected_faces],
        maximum_surface_deviation, maximum_normal_deviation_degrees,
    )
    accepted_iterations = 0
    for _ in range(int(iterations)):
        sums = torch.zeros_like(current)
        sums.index_add_(0, edges[:, 0], current[edges[:, 1]])
        sums.index_add_(0, edges[:, 1], current[edges[:, 0]])
        centroids = sums / counts.clamp_min(1)
        old_cross = _triangle_cross_torch(current, faces[affected_faces])
        step = float(smoothing_step)
        accepted = False
        for _ in range(8):
            proposed = current.clone()
            raw = current[movable] + step * (centroids[movable] - current[movable])
            projected, _ = reference_patch_projector.project(
                raw, minimum_labels[movable],
            )
            proposed[movable] = projected
            # A bad local proposal must not freeze unrelated patches. Revert
            # its incident vertices and check the remaining coupled update.
            for _ in range(4):
                new_cross = _triangle_cross_torch(proposed, faces[affected_faces])
                valid_faces = (
                    (old_cross * new_cross).sum(dim=1) > 0
                ) & ((new_cross * new_cross).sum(dim=1) > minimum_cross_squared)
                valid_faces &= surface_checks.check(proposed[faces[affected_faces]])
                if maximum_edge_length is not None:
                    triangles = proposed[faces[affected_faces]]
                    lengths = torch.linalg.norm(
                        triangles[:, (1, 2, 0)] - triangles[:, (0, 1, 2)], dim=2,
                    )
                    valid_faces &= lengths.amax(dim=1) <= float(maximum_edge_length) * (1 + 1e-6)
                if bool(valid_faces.all()):
                    break
                reverted_vertices = faces[affected_faces[~valid_faces]].reshape(-1)
                proposed[reverted_vertices] = current[reverted_vertices]
            valid = bool(valid_faces.all() and torch.isfinite(proposed).all())
            proposed_energy = _mesh_energy_torch(proposed, faces)
            proposed_mean_quality = patch_mean_quality(proposed)
            if (valid and bool(proposed_energy < current_energy)
                    and bool((proposed_mean_quality >= current_mean_quality - 1e-12).all())):
                current = proposed
                current_energy = proposed_energy
                current_mean_quality = proposed_mean_quality
                accepted_iterations += 1
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return current, accepted_iterations


def _validated_external_partition(
    vertices, faces, face_patch_ids, constraint_edges, corner_vertex_ids=None,
):
    """Validate ownership and the complete shared interface graph once on CPU."""
    labels = np.asarray(face_patch_ids)
    if labels.shape != (len(faces),) or labels.dtype.kind not in "iu":
        raise ValueError("External face patch IDs must be an integer array of shape (face_count,).")
    if len(labels) == 0 or np.any(labels < 0) or np.any(labels > np.iinfo(np.int64).max):
        raise ValueError("External face patch IDs must be nonempty, nonnegative int64 values.")
    labels = labels.astype(np.int64, copy=True)
    if constraint_edges is None:
        raise ValueError("External partition mode requires fixed_constraint_edges, including an explicit empty array when appropriate.")
    constraints = np.asarray(constraint_edges)
    if constraints.shape == (0,):
        constraints = np.empty((0, 2), dtype=np.int64)
    if constraints.ndim != 2 or constraints.shape[1] != 2 or constraints.dtype.kind not in "iu":
        raise ValueError("Fixed constraint edges must be an integer array of shape (n, 2).")
    if len(constraints) and (
        np.any(constraints < 0) or np.any(constraints >= len(vertices))
        or np.any(constraints[:, 0] == constraints[:, 1])
    ):
        raise ValueError("Fixed constraint edges contain invalid vertex IDs or a self edge.")
    constraints = np.unique(np.sort(constraints.astype(np.int64), axis=1), axis=0)
    corners = np.asarray([] if corner_vertex_ids is None else corner_vertex_ids)
    if corners.shape == (0,):
        corners = np.empty(0, dtype=np.int64)
    if corners.ndim != 1 or corners.dtype.kind not in "iu":
        raise ValueError("Fixed corner vertex IDs must be a one-dimensional integer array.")
    if len(corners) and (np.any(corners < 0) or np.any(corners >= len(vertices))):
        raise ValueError("Fixed corner vertex IDs are outside the input vertex range.")
    corners = np.unique(corners.astype(np.int64))

    face_edges = np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
    edges, inverse, counts = np.unique(face_edges, axis=0, return_inverse=True, return_counts=True)
    minimum_labels = np.full(len(edges), np.iinfo(np.int64).max, dtype=np.int64)
    maximum_labels = np.full(len(edges), -1, dtype=np.int64)
    np.minimum.at(minimum_labels, inverse, np.repeat(labels, 3))
    np.maximum.at(maximum_labels, inverse, np.repeat(labels, 3))
    edge_keys = _edge_keys(edges, len(vertices))
    constraint_keys = _edge_keys(constraints, len(vertices))
    if not np.all(np.isin(constraint_keys, edge_keys)):
        raise ValueError("A fixed constraint is not an edge of the input mesh.")
    required = (counts != 2) | (minimum_labels != maximum_labels)
    if not np.all(np.isin(edge_keys[required], constraint_keys)):
        raise ValueError("Fixed constraints must cover every cross-patch, open, and non-manifold edge.")
    referenced = np.unique(faces)
    if len(corners) and not np.all(np.isin(corners, referenced)):
        raise ValueError("Fixed corner vertex IDs must be referenced by an input face.")
    return labels, constraints, corners


@torch.no_grad()
def surface_sample_remesh(
    reference_mesh,
    vertices,
    faces,
    sample_count,
    poisson_radius=None,
    oversample=4,
    seed=0,
    feature_edges=None,
    feature_angle_degrees=30.0,
    flip_passes=5,
    relax_iterations=3,
    smoothing_step=0.2,
    barycentric_margin=0.08,
    minimum_source_quality=1e-4,
    minimum_source_area_ratio=0.25,
    maximum_edge_ratio=2.0,
    minimum_edge_ratio=0.5,
    split_passes=64,
    collapse_passes=24,
    protected_source_quality=0.8,
    maximum_normal_deviation_degrees=5.0,
    maximum_surface_deviation_ratio=0.05,
    minimum_collapse_quality=0.25,
    coplanar_angle_degrees=1.0,
    curvature_adaptive=False,
    curvature_tolerance=None,
    adaptive_minimum_edge_length=None,
    adaptive_maximum_edge_length=None,
    external_face_patch_ids=None,
    fixed_constraint_edges=None,
    fixed_corner_vertex_ids=None,
    whole_patch_optimization=False,
    whole_patch_reference=None,
    projection_backend="cuda",
):
    """
    Sample the original surface, locally retriangulate, and optimize on CUDA.

    Feature edges remain exact. Long edges are bisected before topology
    optimization. High-quality source faces are refinement-only, while
    collapse and flip operations are bounded by a source-normal cone and a
    source-plane deviation tolerance to prevent rounded regions from being
    replaced by flat chords.

    External partition mode uses supplied ownership and fixed interfaces
    verbatim. Hard creases and smooth patch transitions share the same fixed
    topology contract here; neither is rediscovered from dihedral angles.
    Input CUDA coordinates must be float64 and exactly match reference_mesh.
    This mode retains vertex IDs and returns output face ownership and source
    face lineage in the statistics dictionary.

    With whole_patch_optimization, all interior vertices may move across old
    triangle edges on their own reference patch. Collapse, flip and relaxation
    use sampled distance and normal checks against that patch. A separate
    whole_patch_reference=(vertices, faces, patch_ids), or a cached
    _ReferencePatchProjector, can supply complete patches for a computational
    batch. Its patch IDs must use the same numbering as external_face_patch_ids.
    """
    if not vertices.is_cuda or not faces.is_cuda:
        raise ValueError("Surface sample remeshing requires CUDA tensors.")
    flip_passes = int(flip_passes)
    relax_iterations = int(relax_iterations)
    if flip_passes < 0 or relax_iterations < 0:
        raise ValueError("Surface flip and relaxation counts must be non-negative.")
    if not 0.0 < float(smoothing_step) <= 1.0:
        raise ValueError("Surface relaxation step must be in (0, 1].")
    minimum_source_quality = float(minimum_source_quality)
    minimum_source_area_ratio = float(minimum_source_area_ratio)
    maximum_edge_ratio = float(maximum_edge_ratio)
    minimum_edge_ratio = float(minimum_edge_ratio)
    split_passes = int(split_passes)
    collapse_passes = int(collapse_passes)
    protected_source_quality = float(protected_source_quality)
    maximum_normal_deviation_degrees = float(
        maximum_normal_deviation_degrees
    )
    maximum_surface_deviation_ratio = float(
        maximum_surface_deviation_ratio
    )
    minimum_collapse_quality = float(minimum_collapse_quality)
    coplanar_angle_degrees = float(coplanar_angle_degrees)
    curvature_adaptive = bool(curvature_adaptive)
    if not 0.0 <= minimum_source_quality < 1.0:
        raise ValueError("Minimum source quality must be in [0, 1).")
    if minimum_source_area_ratio < 0.0:
        raise ValueError("Minimum source area ratio must be non-negative.")
    if maximum_edge_ratio <= 0.0:
        raise ValueError("Maximum surface edge ratio must be positive.")
    if not 0.0 <= minimum_edge_ratio < maximum_edge_ratio:
        raise ValueError(
            "Minimum surface edge ratio must be non-negative and smaller "
            "than the maximum edge ratio."
        )
    if split_passes <= 0 or collapse_passes < 0:
        raise ValueError(
            "Surface split passes must be positive and collapse passes "
            "must be non-negative."
        )
    if not 0.0 <= protected_source_quality <= 1.0:
        raise ValueError("Protected source quality must be in [0, 1].")
    if not 0.0 <= maximum_normal_deviation_degrees <= 180.0:
        raise ValueError(
            "Maximum source-normal deviation must be in [0, 180] degrees."
        )
    if maximum_surface_deviation_ratio < 0.0:
        raise ValueError(
            "Maximum surface-deviation ratio must be non-negative."
        )
    if not 0.0 <= minimum_collapse_quality <= 1.0:
        raise ValueError("Minimum collapse quality must be in [0, 1].")
    if not 0.0 <= coplanar_angle_degrees < 180.0:
        raise ValueError("Coplanar angle must be in [0, 180).")

    reference_vertices = np.asarray(reference_mesh.vertices, dtype=np.float64)
    reference_faces = np.asarray(reference_mesh.faces, dtype=np.int64)
    external_partition = external_face_patch_ids is not None
    whole_patch_optimization = bool(whole_patch_optimization)
    if whole_patch_optimization and not external_partition:
        raise ValueError("Whole-patch optimization requires external_face_patch_ids.")
    if whole_patch_reference is not None and not whole_patch_optimization:
        raise ValueError("A whole-patch reference requires whole_patch_optimization.")
    fixed_source_vertices = np.empty(0, dtype=np.int64)
    partition_constraints = np.empty((0, 2), dtype=np.int64)
    if external_partition:
        if feature_edges is not None:
            raise ValueError("Use fixed_constraint_edges instead of feature_edges with external patch IDs.")
        if vertices.dtype != torch.float64:
            raise ValueError("External partition mode requires float64 CUDA vertices.")
        if faces.dtype != torch.long or not np.array_equal(faces.cpu().numpy(), reference_faces):
            raise ValueError("External partition CUDA faces must exactly match reference_mesh face ordering.")
        if not np.isfinite(reference_vertices).all() or not np.array_equal(vertices.cpu().numpy(), reference_vertices):
            raise ValueError("External partition CUDA vertices must exactly match finite reference_mesh coordinates.")
        patch_ids, partition_constraints, fixed_corners = _validated_external_partition(
            reference_vertices, reference_faces, external_face_patch_ids,
            fixed_constraint_edges, fixed_corner_vertex_ids,
        )
        fixed_source_vertices = np.unique(np.concatenate((partition_constraints.reshape(-1), fixed_corners)))
        reference_features = np.empty((0, 2), dtype=np.int64)
        coplanar_edges = np.empty((0, 2), dtype=np.int64)
    else:
        if fixed_constraint_edges is not None or fixed_corner_vertex_ids is not None:
            raise ValueError("Fixed partition constraints require external_face_patch_ids.")
        reference_features = detect_reference_feature_edges(
            reference_mesh, feature_edges=feature_edges,
            angle_degrees=feature_angle_degrees, include_boundaries=True,
        )
        coplanar_edges = coplanar_internal_edges(
            reference_mesh, feature_edges=reference_features,
            angle_degrees=coplanar_angle_degrees,
        )
        patch_ids = original_surface_patch_ids(
            reference_faces, len(reference_vertices), reference_features,
        )
    reference_patch_projector = None
    local_reference_projector = None
    if whole_patch_optimization:
        local_reference_projector = _make_reference_projector(
            reference_vertices, reference_faces, patch_ids, backend=projection_backend, device=vertices.device,
        )
        if whole_patch_reference is None:
            reference_patch_projector = local_reference_projector
        elif isinstance(whole_patch_reference, _ReferencePatchProjector):
            reference_patch_projector = whole_patch_reference
        else:
            if not isinstance(whole_patch_reference, (tuple, list)) or len(whole_patch_reference) != 3:
                raise ValueError("Whole-patch reference must be (vertices, faces, patch_ids).")
            reference_patch_projector = _make_reference_projector(*whole_patch_reference, backend=projection_backend, device=vertices.device)
        if not set(np.unique(patch_ids)) <= reference_patch_projector.patch_faces.keys():
            raise ValueError("Whole-patch reference is missing an external patch ID.")
    initial_metrics = mesh_quality_metrics(
        reference_vertices,
        reference_faces,
    )

    if poisson_radius is None:
        triangle_areas = np.asarray(reference_mesh.area_faces)
        total_area = max(
            float(triangle_areas.sum()),
            np.finfo(np.float64).eps,
        )
        poisson_radius = 0.65 * np.sqrt(total_area / int(sample_count))
    source_face_target_lengths = None
    adaptive_vertex_lengths = None
    if curvature_adaptive:
        adaptive_minimum_edge_length = (
            float(poisson_radius)
            if adaptive_minimum_edge_length is None
            else float(adaptive_minimum_edge_length)
        )
        adaptive_maximum_edge_length = (
            float(poisson_radius) * 4.0
            if adaptive_maximum_edge_length is None
            else float(adaptive_maximum_edge_length)
        )
        curvature_tolerance = (
            float(poisson_radius) * 0.1
            if curvature_tolerance is None
            else float(curvature_tolerance)
        )
        (
            adaptive_vertex_lengths,
            source_face_target_lengths,
        ) = curvature_adaptive_source_face_lengths(
            reference_vertices,
            reference_faces,
            tolerance=curvature_tolerance,
            minimum_edge_length=adaptive_minimum_edge_length,
            maximum_edge_length=adaptive_maximum_edge_length,
        )
    source_areas = np.asarray(reference_mesh.area_faces, dtype=np.float64)
    source_qualities = _triangle_quality_numpy(
        reference_vertices,
        reference_faces,
    )
    protected_source_faces = (
        protected_source_quality > 0.0
        and not curvature_adaptive
        and not whole_patch_optimization
    ) & (
        source_qualities >= protected_source_quality
    )
    source_triangles = reference_vertices[reference_faces]
    source_cross = np.cross(
        source_triangles[:, 1] - source_triangles[:, 0],
        source_triangles[:, 2] - source_triangles[:, 0],
    )
    source_cross_lengths = np.linalg.norm(source_cross, axis=1)
    source_normals = np.divide(
        source_cross,
        source_cross_lengths[:, None],
        out=np.zeros_like(source_cross),
        where=source_cross_lengths[:, None] > 0.0,
    )
    source_origins = source_triangles[:, 0]
    eligible_face_mask = (
        (source_qualities >= minimum_source_quality)
        & (
            source_areas
            >= minimum_source_area_ratio * float(poisson_radius) ** 2
        )
    )
    if not np.any(eligible_face_mask) and not external_partition:
        raise RuntimeError(
            "No original triangles are large and well-shaped enough for the "
            "requested surface sample spacing."
        )

    if np.any(eligible_face_mask):
        samples = gpu_area_poisson_sample(
            vertices,
            faces,
            sample_count=sample_count,
            poisson_radius=poisson_radius,
            oversample=oversample,
            seed=seed,
            barycentric_margin=barycentric_margin,
            eligible_face_mask=eligible_face_mask,
        )
    else:
        # A dense computational batch may already have sub-spacing triangles.
        # Still run constrained collapse/flip/split without inventing samples.
        samples = SurfaceSamples(
            vertices.new_empty((0, 3)),
            torch.empty(0, dtype=torch.long, device=vertices.device),
            vertices.new_empty((0, 3)), vertices.new_empty((0, 3)),
            int(sample_count), 0,
        )
    torch.cuda.synchronize(vertices.device)
    sampled_points = samples.points.cpu().numpy()
    sampled_face_ids = samples.face_ids.cpu().numpy()
    sampled_barycentric = samples.barycentric.cpu().numpy()
    rejected_source_faces = set()
    for _ in range(8):
        (
            output_vertices,
            output_faces,
            output_source_faces,
            support_face_ids,
        ) = subdivide_original_faces(
            reference_vertices,
            reference_faces,
            sampled_points,
            sampled_face_ids,
            sampled_barycentric,
            edge_target_length=poisson_radius,
            eligible_face_mask=eligible_face_mask,
            protected_internal_edges=(
                partition_constraints if external_partition
                else (None if curvature_adaptive else coplanar_edges)
            ),
        )
        float_vertices = output_vertices.astype(np.float64 if external_partition else np.float32)
        float_triangles = float_vertices[output_faces]
        float_cross = np.cross(
            float_triangles[:, 1] - float_triangles[:, 0],
            float_triangles[:, 2] - float_triangles[:, 0],
        )
        float_area_squared = np.einsum(
            "ij,ij->i",
            float_cross,
            float_cross,
        )
        invalid_faces = (
            ~np.isfinite(float_area_squared)
            | (float_area_squared <= 0.0)
        )
        if not np.any(invalid_faces):
            break

        bad_source_faces = np.unique(
            output_source_faces[invalid_faces]
        )
        new_bad_sources = [
            int(face_id)
            for face_id in bad_source_faces
            if int(face_id) not in rejected_source_faces
        ]
        if not new_bad_sources:
            raise RuntimeError(
                "Surface retriangulation still contains float32-degenerate "
                "faces after source-face rollback."
            )
        rejected_source_faces.update(new_bad_sources)
        eligible_face_mask[bad_source_faces] = False
        keep_samples = eligible_face_mask[sampled_face_ids]
        sampled_points = sampled_points[keep_samples]
        sampled_face_ids = sampled_face_ids[keep_samples]
        sampled_barycentric = sampled_barycentric[keep_samples]
    else:
        raise RuntimeError(
            "Surface retriangulation could not eliminate float32-degenerate "
            "faces within eight rollback passes."
        )
    sampled_metrics = mesh_quality_metrics(output_vertices, output_faces)

    device = vertices.device
    gpu_vertices = torch.as_tensor(
        output_vertices,
        dtype=vertices.dtype,
        device=device,
    )
    gpu_faces = torch.as_tensor(
        output_faces,
        dtype=torch.long,
        device=device,
    )
    gpu_patch_ids = torch.as_tensor(
        patch_ids[output_source_faces],
        dtype=torch.long,
        device=device,
    )
    gpu_source_face_ids = torch.as_tensor(
        output_source_faces,
        dtype=torch.long,
        device=device,
    )
    gpu_original_vertices = torch.as_tensor(
        reference_vertices,
        dtype=vertices.dtype,
        device=device,
    )
    gpu_original_faces = torch.as_tensor(
        reference_faces,
        dtype=torch.long,
        device=device,
    )
    gpu_protected_source_faces = torch.as_tensor(
        protected_source_faces,
        dtype=torch.bool,
        device=device,
    )
    gpu_source_origins = torch.as_tensor(
        source_origins,
        dtype=vertices.dtype,
        device=device,
    )
    gpu_source_normals = torch.as_tensor(
        source_normals,
        dtype=vertices.dtype,
        device=device,
    )
    gpu_support_face_ids = torch.as_tensor(
        support_face_ids,
        dtype=torch.long,
        device=device,
    )
    gpu_coplanar_edges = torch.as_tensor(
        coplanar_edges,
        dtype=torch.long,
        device=device,
    )
    if len(reference_features):
        gpu_support_face_ids[
            torch.as_tensor(
                np.unique(reference_features),
                dtype=torch.long,
                device=device,
            )
        ] = -2
    if external_partition:
        gpu_support_face_ids[torch.as_tensor(fixed_source_vertices, dtype=torch.long, device=device)] = -2
        topology_protected_edges = torch.as_tensor(partition_constraints, dtype=torch.long, device=device)
    else:
        topology_protected_edges = None if curvature_adaptive else gpu_coplanar_edges

    if curvature_adaptive:
        source_face_maximum_edge_lengths = (
            (4.0 / 3.0) * source_face_target_lengths
        )
        maximum_edge_length = float(source_face_maximum_edge_lengths.max())
        minimum_edge_length = float(
            0.8 * source_face_target_lengths.min()
        )
    else:
        source_face_maximum_edge_lengths = None
        maximum_edge_length = (
            float(poisson_radius) * maximum_edge_ratio
        )
        minimum_edge_length = (
            float(poisson_radius) * minimum_edge_ratio
        )
    reference_diameter = float(
        np.max(np.ptp(reference_vertices, axis=0))
    )
    numeric_surface_tolerance = max(
        reference_diameter,
        float(poisson_radius),
    ) * 1e-7
    maximum_surface_deviation = max(
        float(poisson_radius) * maximum_surface_deviation_ratio,
        numeric_surface_tolerance,
    )
    stage_seconds = {}
    torch.cuda.synchronize(gpu_vertices.device)
    stage_start = perf_counter()

    def finish_stage(name):
        nonlocal stage_start
        torch.cuda.synchronize(gpu_vertices.device)
        now = perf_counter()
        stage_seconds[name] = now - stage_start
        print(f"[remesh] {name}: {now - stage_start:.2f}s, {len(gpu_faces):,} faces", flush=True)
        stage_start = now

    # Remove poor source diagonals before their long edges are subdivided.
    # Spend part of the existing flip budget here, on the smaller mesh.
    # Fixed vertices stay fixed; only explicit constraint edges forbid flips.
    pre_flip_passes = min(2, flip_passes) if whole_patch_optimization else 0
    pre_flip_count = 0
    if pre_flip_passes:
        gpu_faces, pre_flip_count = _gpu_flip_quality_edges(
            gpu_vertices, gpu_faces, gpu_patch_ids, pre_flip_passes,
            maximum_edge_length=maximum_edge_length,
            face_source_ids=gpu_source_face_ids,
            protected_source_face_mask=gpu_protected_source_faces,
            source_face_origins=gpu_source_origins,
            source_face_normals=gpu_source_normals,
            maximum_normal_deviation_degrees=maximum_normal_deviation_degrees,
            maximum_surface_deviation=maximum_surface_deviation,
            protected_edges=topology_protected_edges,
            source_face_target_lengths=source_face_target_lengths,
            reference_patch_projector=reference_patch_projector,
            preserve_existing_maximum_edge=True,
        )
    finish_stage('pre-flip')
    pre_collapse_passes = min(2, collapse_passes) if whole_patch_optimization else collapse_passes
    (
        gpu_vertices,
        gpu_faces,
        gpu_patch_ids,
        gpu_source_face_ids,
        gpu_support_face_ids,
        pre_collapse_count,
    ) = _gpu_collapse_short_edges(
        gpu_vertices,
        gpu_faces,
        gpu_patch_ids,
        gpu_source_face_ids,
        gpu_support_face_ids,
        minimum_edge_length,
        maximum_edge_length,
        pre_collapse_passes,
        protected_source_face_mask=gpu_protected_source_faces,
        source_face_origins=gpu_source_origins,
        source_face_normals=gpu_source_normals,
        maximum_normal_deviation_degrees=(
            maximum_normal_deviation_degrees
        ),
        maximum_surface_deviation=maximum_surface_deviation,
        minimum_collapse_quality=minimum_collapse_quality,
        source_face_target_lengths=source_face_target_lengths,
        compact_vertices=not external_partition,
        reference_patch_projector=reference_patch_projector,
    )
    finish_stage('pre-collapse')
    (
        gpu_vertices,
        gpu_faces,
        gpu_patch_ids,
        gpu_source_face_ids,
        gpu_support_face_ids,
        split_count,
        remaining_long_edges,
    ) = _gpu_split_long_edges(
        gpu_vertices,
        gpu_faces,
        gpu_patch_ids,
        gpu_source_face_ids,
        gpu_support_face_ids,
        maximum_edge_length,
        split_passes,
        protected_edges=topology_protected_edges,
        source_face_maximum_edge_lengths=(
            source_face_maximum_edge_lengths
        ),
    )
    finish_stage('split')
    if whole_patch_optimization:
        # The splitter also marks an internal midpoint fixed when both old
        # endpoints were fixed. Only supplied interface/corner vertices are
        # fixed in this mode; the interior chord midpoint is movable.
        gpu_support_face_ids[gpu_support_face_ids < -1] = -1
        gpu_support_face_ids[torch.as_tensor(fixed_source_vertices, dtype=torch.long,
                                             device=device)] = -2
    (
        gpu_vertices,
        gpu_faces,
        gpu_patch_ids,
        gpu_source_face_ids,
        gpu_support_face_ids,
        post_collapse_count,
    ) = _gpu_collapse_short_edges(
        gpu_vertices,
        gpu_faces,
        gpu_patch_ids,
        gpu_source_face_ids,
        gpu_support_face_ids,
        minimum_edge_length,
        maximum_edge_length,
        collapse_passes - pre_collapse_passes if whole_patch_optimization else collapse_passes // 2,
        protected_source_face_mask=gpu_protected_source_faces,
        source_face_origins=gpu_source_origins,
        source_face_normals=gpu_source_normals,
        maximum_normal_deviation_degrees=(
            maximum_normal_deviation_degrees
        ),
        maximum_surface_deviation=maximum_surface_deviation,
        minimum_collapse_quality=minimum_collapse_quality,
        source_face_target_lengths=source_face_target_lengths,
        compact_vertices=not external_partition,
        reference_patch_projector=reference_patch_projector,
    )
    collapse_count = pre_collapse_count + post_collapse_count
    finish_stage('post-collapse')
    gpu_source_sensitive_vertices = _gpu_source_sensitive_vertex_mask(
        gpu_faces,
        gpu_source_face_ids,
        None if whole_patch_optimization else gpu_source_normals,
        protected_source_face_mask=gpu_protected_source_faces,
        maximum_normal_deviation_degrees=(
            maximum_normal_deviation_degrees
        ),
        vertex_count=len(gpu_vertices),
    )
    gpu_faces, flip_count = _gpu_flip_quality_edges(
        gpu_vertices,
        gpu_faces,
        gpu_patch_ids,
        flip_passes - pre_flip_passes,
        maximum_edge_length=maximum_edge_length,
        protected_vertex_mask=(
            (gpu_support_face_ids < -1)
            | gpu_source_sensitive_vertices
        ),
        face_source_ids=gpu_source_face_ids,
        protected_source_face_mask=gpu_protected_source_faces,
        source_face_origins=gpu_source_origins,
        source_face_normals=gpu_source_normals,
        maximum_normal_deviation_degrees=(
            maximum_normal_deviation_degrees
        ),
        maximum_surface_deviation=maximum_surface_deviation,
        protected_edges=topology_protected_edges,
        source_face_target_lengths=source_face_target_lengths,
        reference_patch_projector=reference_patch_projector,
    )
    flip_count += pre_flip_count
    finish_stage('post-flip')
    if whole_patch_optimization:
        gpu_vertices, accepted_relaxations = _gpu_relax_patch_vertices(
            gpu_vertices, gpu_faces, gpu_patch_ids, gpu_support_face_ids,
            reference_patch_projector, relax_iterations, smoothing_step,
            maximum_edge_length, maximum_surface_deviation,
            maximum_normal_deviation_degrees,
        )
    else:
        gpu_vertices, accepted_relaxations = _gpu_relax_inserted_vertices(
            gpu_vertices,
            gpu_faces,
            gpu_original_vertices,
            gpu_original_faces,
            gpu_support_face_ids,
            relax_iterations,
            smoothing_step,
            maximum_edge_length=maximum_edge_length if external_partition else None,
            protected_source_face_mask=gpu_protected_source_faces,
            protected_vertex_mask=_gpu_source_sensitive_vertex_mask(
                gpu_faces,
                gpu_source_face_ids,
                gpu_source_normals,
                protected_source_face_mask=gpu_protected_source_faces,
                maximum_normal_deviation_degrees=(
                    maximum_normal_deviation_degrees
                ),
                vertex_count=len(gpu_vertices),
            ),
        )
    finish_stage('relax')
    if whole_patch_optimization:
        # IDs remain local to reference_mesh even if projection used a larger
        # global patch. They describe nearest reference support, rather than
        # claiming the output triangle stays inside its ancestor face.
        _, gpu_source_face_ids = local_reference_projector.project(
            gpu_vertices[gpu_faces].mean(dim=1), gpu_patch_ids,
        )
        final_surface_valid, final_surface_details = reference_patch_projector.valid_triangles(
            gpu_vertices[gpu_faces], gpu_patch_ids, maximum_surface_deviation,
            maximum_normal_deviation_degrees, return_details=True,
        )
        if not bool(final_surface_valid.all()):
            invalid = ~final_surface_valid.cpu().numpy()
            bad_patches, bad_counts = np.unique(gpu_patch_ids.cpu().numpy()[invalid], return_counts=True)
            _, baseline_details = reference_patch_projector.valid_triangles(
                gpu_original_vertices[gpu_original_faces],
                torch.as_tensor(patch_ids, dtype=torch.long, device=device),
                maximum_surface_deviation, maximum_normal_deviation_degrees,
                return_details=True,
            )
            raise RuntimeError(
                "Whole-patch output violates sampled reference surface/normal tolerances: "
                f"{int(invalid.sum())}/{len(invalid)} invalid faces; "
                f"distance={int(np.count_nonzero(~final_surface_details['distance_valid']))} "
                f"(max={float(final_surface_details['maximum_sample_distance'].max()):.9g}, "
                f"limit={maximum_surface_deviation:.9g}), "
                f"normal={int(np.count_nonzero(~final_surface_details['normal_valid']))} "
                f"(max={float(final_surface_details['normal_deviation_degrees'].max()):.9g}deg, "
                f"limit={maximum_normal_deviation_degrees:.9g}deg); "
                f"patches={list(zip(bad_patches[:8].tolist(), bad_counts[:8].tolist()))}; "
                f"input baseline distance={int(np.count_nonzero(~baseline_details['distance_valid']))}, "
                f"normal={int(np.count_nonzero(~baseline_details['normal_valid']))}."
            )
    torch.cuda.synchronize(device)
    final_vertices = gpu_vertices.cpu().numpy()
    final_faces = gpu_faces.cpu().numpy()
    if external_partition:
        final_labels = gpu_patch_ids.cpu().numpy()
        final_source_faces = gpu_source_face_ids.cpu().numpy()
        if final_source_faces.shape != (len(final_faces),) or np.any(final_source_faces < 0) or np.any(final_source_faces >= len(reference_faces)):
            raise RuntimeError("External partition remeshing produced invalid source face lineage.")
        if not np.array_equal(final_vertices[fixed_source_vertices], reference_vertices[fixed_source_vertices]):
            raise RuntimeError("External partition remeshing moved a fixed interface vertex or corner.")
        if final_labels.shape != (len(final_faces),) or not np.array_equal(final_labels, patch_ids[final_source_faces]):
            raise RuntimeError("External partition remeshing lost face ownership or source lineage.")
        if not np.array_equal(np.unique(final_labels), np.unique(patch_ids)):
            raise RuntimeError("External partition remeshing removed an entire input patch.")
        # Revalidate complete interfaces on the final global mesh. This also
        # detects a missing fixed edge, new crack, or a cross-label diagonal.
        _validated_external_partition(final_vertices, final_faces, final_labels, partition_constraints, fixed_corners)
    final_metrics = mesh_quality_metrics(final_vertices, final_faces)

    print(
        "Original-surface CUDA sampling: {} requested, {} candidates, "
        "{} grid-Poisson samples retained on {} / {} eligible source faces "
        "(radius {:.6g})".format(
            samples.requested_count,
            samples.candidate_count,
            len(sampled_points),
            int(np.count_nonzero(eligible_face_mask)),
            len(reference_faces),
            float(poisson_radius),
        )
    )
    if external_partition:
        print(
            "External partition constraints: {} supplied patches, {} fixed "
            "interface edges, {} explicit corners; face ownership and "
            "shared vertex IDs retained.".format(
                len(np.unique(patch_ids)), len(partition_constraints), len(fixed_corners),
            )
        )
    if curvature_adaptive:
        print(
            "Curvature-adaptive sizing: tolerance {:.6g}, target range "
            "{:.6g} .. {:.6g}; vertex target p10/median/p90 "
            "{:.6g}/{:.6g}/{:.6g}.".format(
                curvature_tolerance,
                adaptive_minimum_edge_length,
                adaptive_maximum_edge_length,
                *np.percentile(adaptive_vertex_lengths, (10.0, 50.0, 90.0)),
            )
        )
    print(
        "Feature-safe retriangulation: {} smooth patches, {} hard feature "
        "edges, {} coplanar internal edges, {} / {} refinement-only source "
        "faces, {} source faces rolled "
        "back; {} CUDA long-edge splits "
        "({} still over {:.6g}), {} CUDA short-edge collapses, {} GPU "
        "flips, {} / {} relaxations accepted".format(
            len(np.unique(patch_ids)) if external_partition else (int(patch_ids.max()) + 1 if len(patch_ids) else 0),
            len(reference_features),
            len(coplanar_edges),
            int(np.count_nonzero(protected_source_faces)),
            len(reference_faces),
            len(rejected_source_faces),
            split_count,
            remaining_long_edges,
            maximum_edge_length,
            collapse_count,
            flip_count,
            accepted_relaxations,
            relax_iterations,
        )
    )
    print(
        "Quality (input -> sampled -> optimized): edge CV "
        "{:.6g} -> {:.6g} -> {:.6g}; mean triangle quality "
        "{:.6g} -> {:.6g} -> {:.6g}; minimum angle "
        "{:.6g} -> {:.6g} -> {:.6g} degrees".format(
            initial_metrics["edge_cv"],
            sampled_metrics["edge_cv"],
            final_metrics["edge_cv"],
            initial_metrics["mean_triangle_quality"],
            sampled_metrics["mean_triangle_quality"],
            final_metrics["mean_triangle_quality"],
            initial_metrics["minimum_angle_degrees"],
            sampled_metrics["minimum_angle_degrees"],
            final_metrics["minimum_angle_degrees"],
        )
    )
    finish_stage('final-validation-and-metrics')
    statistics = {
        "stage_seconds": stage_seconds,
        "pre_flip_count": pre_flip_count,
        "pre_collapse_count": pre_collapse_count,
        "sample_count": len(sampled_points),
        "poisson_radius": float(poisson_radius),
        "feature_edge_count": len(reference_features),
        "eligible_source_face_count": int(
            np.count_nonzero(eligible_face_mask)
        ),
        "rolled_back_source_face_count": len(rejected_source_faces),
        "patch_count": len(np.unique(patch_ids)) if external_partition else (int(patch_ids.max()) + 1 if len(patch_ids) else 0),
        "maximum_edge_length": maximum_edge_length,
        "minimum_edge_length": minimum_edge_length,
        "curvature_adaptive": curvature_adaptive,
        "whole_patch_optimization": whole_patch_optimization,
        "projection_backend": getattr(reference_patch_projector, "backend_name", "cpu_libigl") if whole_patch_optimization else "source_triangle",
        "curvature_tolerance": curvature_tolerance,
        "adaptive_minimum_edge_length": adaptive_minimum_edge_length,
        "adaptive_maximum_edge_length": adaptive_maximum_edge_length,
        "protected_source_face_count": int(
            np.count_nonzero(protected_source_faces)
        ),
        "protected_source_quality": protected_source_quality,
        "maximum_normal_deviation_degrees": (
            maximum_normal_deviation_degrees
        ),
        "maximum_surface_deviation": maximum_surface_deviation,
        "minimum_collapse_quality": minimum_collapse_quality,
        "coplanar_internal_edge_count": int(len(coplanar_edges)),
        "splits": split_count,
        "remaining_long_edges": remaining_long_edges,
        "collapses": collapse_count,
        "flips": flip_count,
        "accepted_relaxations": accepted_relaxations,
        "initial_metrics": initial_metrics,
        "sampled_metrics": sampled_metrics,
        "final_metrics": final_metrics,
    }
    if external_partition:
        statistics.update({
            "external_partition": True,
            "fixed_constraint_edge_count": len(partition_constraints),
            "fixed_corner_count": len(fixed_corners),
            "face_patch_ids": final_labels,
            "source_face_ids": final_source_faces,
        })
        if whole_patch_optimization:
            used_vertices = np.unique(final_faces)
            used_original = used_vertices[used_vertices < len(reference_vertices)]
            moved = np.linalg.norm(final_vertices[used_original] - reference_vertices[used_original], axis=1)
            statistics.update({
                "source_face_id_semantics": "nearest_local_reference_support_on_same_patch",
                "surface_validation": "vertices_edge_midpoints_centroid_and_oriented_centroid_normal",
                "surface_validation_is_hausdorff_bound": False,
                "reference_normal_distance_ties_resolved": int(final_surface_details["normal_reference_ties_resolved"].sum()),
                "projection_reference": "complete_supplied_patch" if whole_patch_reference is not None else "complete_input_patch",
                "original_interior_vertices_moved": int(np.count_nonzero(moved > numeric_surface_tolerance)),
                "original_vertices_unused": int(len(reference_vertices) - len(used_original)),
            })
    return final_vertices, final_faces, statistics
