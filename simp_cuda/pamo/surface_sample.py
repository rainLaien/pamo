"""GPU surface sampling followed by feature-safe local retriangulation."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError

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
    total_splits = 0

    for _ in range(maximum_passes):
        topology = _gpu_edge_topology(faces, len(vertices))
        edge_lengths = torch.linalg.norm(
            vertices[topology["edges"][:, 1]]
            - vertices[topology["edges"][:, 0]],
            dim=1,
        )
        candidate_groups = torch.nonzero(
            (edge_lengths > maximum_edge_length)
            & (topology["counts"] <= 2)
        ).reshape(-1)
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
    remaining = int(
        (
            torch.linalg.norm(
                vertices[topology["edges"][:, 1]]
                - vertices[topology["edges"][:, 0]],
                dim=1,
            )
            > maximum_edge_length
        ).sum().item()
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
        maximum_surface_deviation is not None
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
    invalid_face = (
        ~torch.isfinite(new_edge_lengths).all(dim=1)
        | (new_cross_squared <= minimum_cross_squared)
        | (orientation <= 0.0)
        | surface_deviation_invalid
        | (
            new_edge_lengths.amax(dim=1)
            > torch.maximum(
                torch.full_like(
                    old_edge_lengths[:, 0],
                    float(maximum_edge_length),
                ),
                old_edge_lengths.amax(dim=1),
            )
            * (1.0 + 1e-6)
        )
        | (new_quality < old_quality * 0.5)
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
    return valid, energy_delta


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
    total_collapses = 0

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
        hard_edges = topology["edges"][hard_groups]
        if len(hard_edges):
            locked[hard_edges.reshape(-1)] = True

        manifold_edges = topology["edges"][manifold_groups]
        edge_lengths = torch.linalg.norm(
            vertices[manifold_edges[:, 1]]
            - vertices[manifold_edges[:, 0]],
            dim=1,
        )
        face_quality = _triangle_quality_torch(vertices, faces)
        adjacent_quality = torch.minimum(
            face_quality[first_faces],
            face_quality[second_faces],
        )
        candidate_mask = (
            same_patch
            & (
                (edge_lengths < minimum_edge_length)
                | (
                    adjacent_quality
                    < minimum_collapse_quality
                )
            )
            & ~(
                locked[manifold_edges[:, 0]]
                & locked[manifold_edges[:, 1]]
            )
        )
        candidates = torch.nonzero(candidate_mask).reshape(-1)
        if len(candidates) == 0:
            break
        candidate_edges = manifold_edges[candidates]
        candidate_lengths = edge_lengths[candidates]
        candidate_patches = face_patch_ids[first_faces[candidates]]

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
        if len(candidate_edges) == 0:
            break

        remove_first = torch.ones(
            len(candidate_edges),
            dtype=torch.bool,
            device=faces.device,
        )
        first_valid, first_energy = _gpu_evaluate_collapse_direction(
            vertices,
            faces,
            candidate_edges,
            candidate_face_pairs,
            remove_first,
            maximum_edge_length,
            face_source_ids=face_source_ids,
            source_face_origins=source_face_origins,
            source_face_normals=source_face_normals,
            maximum_surface_deviation=maximum_surface_deviation,
        )
        second_valid, second_energy = _gpu_evaluate_collapse_direction(
            vertices,
            faces,
            candidate_edges,
            candidate_face_pairs,
            ~remove_first,
            maximum_edge_length,
            face_source_ids=face_source_ids,
            source_face_origins=source_face_origins,
            source_face_normals=source_face_normals,
            maximum_surface_deviation=maximum_surface_deviation,
        )
        first_valid &= ~locked[candidate_edges[:, 0]]
        second_valid &= ~locked[candidate_edges[:, 1]]
        direction_valid = first_valid | second_valid
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
            valid &= (
                proposed_edge_lengths
                <= float(maximum_edge_length) * (1.0 + 1e-6)
            )
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
):
    """
    Mark vertices whose incident source lineage must not be optimized.

    Vertices touching a refinement-only source face are always marked. Vertices
    shared by source faces outside the requested normal cone are also marked so
    relaxation cannot pull a source-edge sample into just one side of a rounded
    region.
    """
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
):
    """
    Sample the original surface, locally retriangulate, and optimize on CUDA.

    Feature edges remain exact. Long edges are bisected before topology
    optimization. High-quality source faces are refinement-only, while
    collapse and flip operations are bounded by a source-normal cone and a
    source-plane deviation tolerance to prevent rounded regions from being
    replaced by flat chords.
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
    reference_features = detect_reference_feature_edges(
        reference_mesh,
        feature_edges=feature_edges,
        angle_degrees=feature_angle_degrees,
        include_boundaries=True,
    )
    coplanar_edges = coplanar_internal_edges(
        reference_mesh,
        feature_edges=reference_features,
        angle_degrees=coplanar_angle_degrees,
    )
    patch_ids = original_surface_patch_ids(
        reference_faces,
        len(reference_vertices),
        reference_features,
    )
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
    source_areas = np.asarray(reference_mesh.area_faces, dtype=np.float64)
    source_qualities = _triangle_quality_numpy(
        reference_vertices,
        reference_faces,
    )
    protected_source_faces = (
        protected_source_quality > 0.0
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
    if not np.any(eligible_face_mask):
        raise RuntimeError(
            "No original triangles are large and well-shaped enough for the "
            "requested surface sample spacing."
        )

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
            protected_internal_edges=coplanar_edges,
        )
        float_vertices = output_vertices.astype(np.float32)
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
        collapse_passes,
        protected_source_face_mask=gpu_protected_source_faces,
        source_face_origins=gpu_source_origins,
        source_face_normals=gpu_source_normals,
        maximum_normal_deviation_degrees=(
            maximum_normal_deviation_degrees
        ),
        maximum_surface_deviation=maximum_surface_deviation,
        minimum_collapse_quality=minimum_collapse_quality,
    )
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
        protected_edges=gpu_coplanar_edges,
    )
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
        collapse_passes // 2,
        protected_source_face_mask=gpu_protected_source_faces,
        source_face_origins=gpu_source_origins,
        source_face_normals=gpu_source_normals,
        maximum_normal_deviation_degrees=(
            maximum_normal_deviation_degrees
        ),
        maximum_surface_deviation=maximum_surface_deviation,
        minimum_collapse_quality=minimum_collapse_quality,
    )
    collapse_count = pre_collapse_count + post_collapse_count
    gpu_source_sensitive_vertices = _gpu_source_sensitive_vertex_mask(
        gpu_faces,
        gpu_source_face_ids,
        gpu_source_normals,
        protected_source_face_mask=gpu_protected_source_faces,
        maximum_normal_deviation_degrees=(
            maximum_normal_deviation_degrees
        ),
    )
    gpu_faces, flip_count = _gpu_flip_quality_edges(
        gpu_vertices,
        gpu_faces,
        gpu_patch_ids,
        flip_passes,
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
        protected_edges=gpu_coplanar_edges,
    )
    gpu_vertices, accepted_relaxations = _gpu_relax_inserted_vertices(
        gpu_vertices,
        gpu_faces,
        gpu_original_vertices,
        gpu_original_faces,
        gpu_support_face_ids,
        relax_iterations,
        smoothing_step,
        protected_source_face_mask=gpu_protected_source_faces,
        protected_vertex_mask=_gpu_source_sensitive_vertex_mask(
            gpu_faces,
            gpu_source_face_ids,
            gpu_source_normals,
            protected_source_face_mask=gpu_protected_source_faces,
            maximum_normal_deviation_degrees=(
                maximum_normal_deviation_degrees
            ),
        ),
    )
    torch.cuda.synchronize(device)
    final_vertices = gpu_vertices.cpu().numpy()
    final_faces = gpu_faces.cpu().numpy()
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
    print(
        "Feature-safe retriangulation: {} smooth patches, {} hard feature "
        "edges, {} coplanar internal edges, {} / {} refinement-only source "
        "faces, {} source faces rolled "
        "back; {} CUDA long-edge splits "
        "({} still over {:.6g}), {} CUDA short-edge collapses, {} GPU "
        "flips, {} / {} relaxations accepted".format(
            int(patch_ids.max()) + 1 if len(patch_ids) else 0,
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
    return final_vertices, final_faces, {
        "sample_count": len(sampled_points),
        "poisson_radius": float(poisson_radius),
        "feature_edge_count": len(reference_features),
        "eligible_source_face_count": int(
            np.count_nonzero(eligible_face_mask)
        ),
        "rolled_back_source_face_count": len(rejected_source_faces),
        "patch_count": int(patch_ids.max()) + 1 if len(patch_ids) else 0,
        "maximum_edge_length": maximum_edge_length,
        "minimum_edge_length": minimum_edge_length,
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
