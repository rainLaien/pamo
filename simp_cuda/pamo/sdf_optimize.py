import itertools

import numpy as np
import torch
import torch.nn.functional as F


def _prepare_volume(sdf):
    volume = sdf
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(
            "Expected a 3D SDF volume, got shape {}.".format(tuple(sdf.shape))
        )
    return volume.contiguous()[None, None]


def _sample_volume(volume, points, axis_order):
    # grid_sample expects coordinates in W, H, D order.
    grid = points[:, axis_order].reshape(1, -1, 1, 1, 3)
    grid = grid * 2.0 - 1.0
    return F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(-1)


def find_sdf_axis_order(sdf, normalized_vertices):
    """Choose the volume-axis convention with the smallest surface residual."""
    volume = _prepare_volume(sdf)
    if len(normalized_vertices) > 8192:
        sample_ids = torch.linspace(
            0,
            len(normalized_vertices) - 1,
            8192,
            device=normalized_vertices.device,
        ).long()
        samples = normalized_vertices[sample_ids]
    else:
        samples = normalized_vertices

    best_order = None
    best_error = None
    with torch.no_grad():
        for order in itertools.permutations((0, 1, 2)):
            values = _sample_volume(volume, samples, order)
            error = float(values.abs().mean().item())
            if best_error is None or error < best_error:
                best_error = error
                best_order = order
    return best_order, best_error


def _sdf_values_and_gradients(volume, points, axis_order):
    # This helper is also called from public operations that may be wrapped in
    # ``torch.no_grad()``. Re-enable gradients locally for the SDF derivative
    # without retaining an optimization graph across iterations.
    with torch.enable_grad():
        query = points.detach().requires_grad_(True)
        values = _sample_volume(volume, query, axis_order)
        gradients = torch.autograd.grad(
            values.sum(),
            query,
            create_graph=False,
            retain_graph=False,
        )[0]
    return values.detach(), gradients.detach()


def _project_to_sdf(
    volume,
    points,
    axis_order,
    projection_steps,
    maximum_step,
):
    projected = points
    for _ in range(projection_steps):
        values, gradients = _sdf_values_and_gradients(
            volume,
            projected,
            axis_order,
        )
        gradient_squared = (gradients * gradients).sum(dim=1, keepdim=True)
        correction = (
            values[:, None]
            * gradients
            / gradient_squared.clamp_min(1e-12)
        )
        correction_length = torch.linalg.norm(
            correction,
            dim=1,
            keepdim=True,
        )
        correction = correction * torch.clamp(
            maximum_step / correction_length.clamp_min(1e-12),
            max=1.0,
        )
        projected = (projected - correction).clamp(0.0, 1.0)
    return projected.detach()


def project_vertices_to_sdf_zero(vertices, sdf, projection_steps=3):
    """
    Project DMC vertices onto the trilinearly interpolated SDF zero set.

    Dual Marching Cubes places one representative vertex per active cell, so
    its raw vertices are not guaranteed to have exactly zero sampled SDF.
    This small Newton correction preserves connectivity while restoring the
    requested isovalue semantics.
    """
    if not torch.is_tensor(sdf):
        raise TypeError("The SDF volume must be a torch tensor.")
    projection_steps = int(projection_steps)
    if projection_steps <= 0:
        raise ValueError("SDF projection steps must be positive.")
    if not sdf.is_floating_point():
        raise TypeError("The SDF volume must use a floating-point dtype.")

    volume = _prepare_volume(sdf)
    current = torch.as_tensor(
        vertices,
        dtype=sdf.dtype,
        device=sdf.device,
    )
    if current.ndim != 2 or current.shape[1] != 3 or len(current) == 0:
        raise ValueError("SDF projection vertices must have shape (n, 3).")
    if not bool(torch.isfinite(current).all().item()):
        raise ValueError("SDF projection vertices contain NaN or infinity.")
    axis_order, initial_error = find_sdf_axis_order(sdf, current)
    maximum_step = 2.0 / max(volume.shape[-3:])
    with torch.enable_grad():
        projected = _project_to_sdf(
            volume,
            current,
            axis_order,
            projection_steps,
            maximum_step,
        )
    final_error = float(
        _sample_volume(
            volume,
            projected,
            axis_order,
        ).abs().mean().item()
    )
    print(
        "DMC zero-set residual : {:.6g} -> {:.6g}".format(
            initial_error,
            final_error,
        )
    )
    return projected


def _mesh_topology(vertices, faces, angle_degrees):
    """Return unique edges and vertices locked by hard local topology."""
    face_edges = torch.cat(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        ),
        dim=0,
    )
    face_edges = torch.sort(face_edges, dim=1).values
    edges, inverse, counts = torch.unique(
        face_edges,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )

    # Boundary and non-manifold vertices do not have a unique surface tangent
    # neighborhood, so moving them with an unconstrained one-ring average can
    # shrink boundaries or merge sheets. Treat them as hard features.
    protected_edges = counts != 2
    # Use the broadly supported nonzero form for older PyTorch releases.
    manifold_edge_ids = torch.nonzero(counts == 2).reshape(-1)
    if len(manifold_edge_ids):
        edge_order = torch.argsort(inverse)
        group_starts = torch.cumsum(counts, dim=0) - counts
        first_entries = edge_order[group_starts[manifold_edge_ids]]
        second_entries = edge_order[group_starts[manifold_edge_ids] + 1]
        face_ids = torch.arange(
            len(faces),
            dtype=torch.long,
            device=faces.device,
        ).repeat(3)
        first_faces = face_ids[first_entries]
        second_faces = face_ids[second_entries]

        face_cross = _face_cross_products(vertices, faces)
        face_norms = torch.linalg.norm(face_cross, dim=1)
        normal_epsilon = torch.finfo(vertices.dtype).eps
        valid_first = face_norms[first_faces] > normal_epsilon
        valid_second = face_norms[second_faces] > normal_epsilon
        normals = face_cross / face_norms[:, None].clamp_min(
            normal_epsilon
        )
        cosine = (
            normals[first_faces] * normals[second_faces]
        ).sum(dim=1)
        sharp = (
            (~valid_first)
            | (~valid_second)
            | (
                cosine
                <= np.cos(np.deg2rad(float(angle_degrees)))
            )
        )
        protected_edges[manifold_edge_ids] = sharp

    locked_vertices = torch.zeros(
        len(vertices),
        dtype=torch.bool,
        device=vertices.device,
    )
    locked_vertices[edges[protected_edges].reshape(-1)] = True
    return edges, locked_vertices


def _neighbor_centroids(vertices, edges):
    sums = torch.zeros_like(vertices)
    counts = torch.zeros(
        (len(vertices), 1),
        dtype=vertices.dtype,
        device=vertices.device,
    )
    ones = torch.ones(
        (len(edges), 1),
        dtype=vertices.dtype,
        device=vertices.device,
    )
    sums.index_add_(0, edges[:, 0], vertices[edges[:, 1]])
    sums.index_add_(0, edges[:, 1], vertices[edges[:, 0]])
    counts.index_add_(0, edges[:, 0], ones)
    counts.index_add_(0, edges[:, 1], ones)
    return sums / counts.clamp_min(1.0)


def _face_cross_products(vertices, faces):
    triangles = vertices[faces]
    return torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )


def _valid_step(old_cross, proposed, faces, minimum_area_squared):
    new_cross = _face_cross_products(proposed, faces)
    orientation = (old_cross * new_cross).sum(dim=1)
    new_area_squared = (new_cross * new_cross).sum(dim=1) * 0.25
    return torch.all(
        (orientation > 0.0)
        & (new_area_squared > minimum_area_squared)
    )


def _mesh_quality_energy(vertices, faces, edges):
    """Scale-invariant edge-uniformity and triangle-shape objective."""
    edge_lengths = torch.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]],
        dim=1,
    )
    epsilon = torch.finfo(vertices.dtype).eps
    mean_length = edge_lengths.mean().clamp_min(epsilon)
    normalized_lengths = edge_lengths / mean_length
    edge_cv_squared = ((normalized_lengths - 1.0) ** 2).mean()

    triangles = vertices[faces]
    ab = triangles[:, 1] - triangles[:, 0]
    bc = triangles[:, 2] - triangles[:, 1]
    ca = triangles[:, 0] - triangles[:, 2]
    twice_area = torch.linalg.norm(torch.cross(ab, -ca, dim=1), dim=1)
    squared_length_sum = (
        (ab * ab).sum(dim=1)
        + (bc * bc).sum(dim=1)
        + (ca * ca).sum(dim=1)
    ).clamp_min(epsilon)
    # 1 for an equilateral triangle and 0 for a degenerate triangle.
    triangle_quality = (
        2.0
        * np.sqrt(3.0)
        * twice_area
        / squared_length_sum
    ).clamp(0.0, 1.0)
    shape_energy = ((1.0 - triangle_quality) ** 2).mean()
    return edge_cv_squared + shape_energy, torch.sqrt(edge_cv_squared)


def _edge_length_cv(vertices, edges):
    if len(edges) == 0:
        return 0.0
    lengths = torch.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]],
        dim=1,
    )
    mean = lengths.mean().clamp_min(torch.finfo(vertices.dtype).eps)
    variance = ((lengths - mean) ** 2).mean()
    return float((torch.sqrt(variance) / mean).item())


def optimize_mesh_on_sdf(
    vertices,
    faces,
    sdf,
    iterations=20,
    smoothing_step=0.2,
    projection_steps=3,
    feature_angle=45.0,
):
    """
    Improve triangle distribution while constraining vertices to an SDF.

    Connectivity and face count are unchanged. Each iteration performs
    tangential Laplacian relocation followed by Newton projection to the SDF
    zero level set. A quality-decreasing line search rejects geometric or SDF
    regressions. Vertices on sharp, boundary, or non-manifold edges stay fixed.
    """
    iterations = int(iterations)
    projection_steps = int(projection_steps)
    smoothing_step = float(smoothing_step)
    if iterations <= 0:
        raise ValueError("SDF optimization iterations must be positive.")
    if projection_steps <= 0:
        raise ValueError("SDF projection steps must be positive.")
    if not 0.0 < smoothing_step <= 1.0:
        raise ValueError("SDF smoothing step must be in (0, 1].")
    if not 0.0 < float(feature_angle) < 180.0:
        raise ValueError("SDF feature angle must be between 0 and 180.")
    if not torch.is_tensor(sdf):
        raise TypeError("The SDF volume must be a torch tensor.")
    if not sdf.is_floating_point():
        raise TypeError("The SDF volume must use a floating-point dtype.")

    device = sdf.device
    dtype = sdf.dtype
    current = torch.as_tensor(vertices, dtype=dtype, device=device).clone()
    if torch.is_tensor(faces):
        integer_face_types = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if faces.dtype not in integer_face_types:
            raise TypeError("SDF optimization face indices must be integers.")
    else:
        faces_array = np.asarray(faces)
        if not np.issubdtype(faces_array.dtype, np.integer):
            raise TypeError("SDF optimization face indices must be integers.")
    faces_tensor = torch.as_tensor(
        faces,
        dtype=torch.long,
        device=device,
    )
    if current.ndim != 2 or current.shape[1] != 3 or len(current) == 0:
        raise ValueError("SDF optimization vertices must have shape (n, 3).")
    if (
        faces_tensor.ndim != 2
        or faces_tensor.shape[1] != 3
        or len(faces_tensor) == 0
    ):
        raise ValueError("SDF optimization faces must have shape (m, 3).")
    if not bool(torch.isfinite(current).all().item()):
        raise ValueError("SDF optimization vertices contain NaN or infinity.")
    if (
        int(faces_tensor.min().item()) < 0
        or int(faces_tensor.max().item()) >= len(current)
    ):
        raise ValueError("SDF optimization face indices are out of range.")
    repeated_vertex = (
        (faces_tensor[:, 0] == faces_tensor[:, 1])
        | (faces_tensor[:, 1] == faces_tensor[:, 2])
        | (faces_tensor[:, 2] == faces_tensor[:, 0])
    )
    if bool(torch.any(repeated_vertex).item()):
        raise ValueError("SDF optimization faces must not repeat a vertex.")

    volume = _prepare_volume(sdf)
    axis_order, initial_sdf_error = find_sdf_axis_order(sdf, current)
    edges, feature_mask = _mesh_topology(
        current,
        faces_tensor,
        feature_angle,
    )

    initial_edge_cv = _edge_length_cv(current, edges)
    current_quality, _ = _mesh_quality_energy(
        current,
        faces_tensor,
        edges,
    )
    initial_quality = float(current_quality.item())
    voxel_size = 1.0 / max(volume.shape[-3:])
    maximum_projection_step = voxel_size * 2.0
    extent = (
        current.max(dim=0).values - current.min(dim=0).values
    ).max().clamp_min(1e-12)
    minimum_area_squared = float((extent * extent * 1e-14) ** 2)
    accepted_iterations = 0
    stop_reason = "iteration limit"
    residual_floor = torch.as_tensor(
        voxel_size * 1e-3,
        dtype=dtype,
        device=device,
    )

    for _ in range(iterations):
        values, gradients = _sdf_values_and_gradients(
            volume,
            current,
            axis_order,
        )
        gradient_norms = torch.linalg.norm(
            gradients,
            dim=1,
            keepdim=True,
        )
        gradient_epsilon = np.sqrt(torch.finfo(dtype).eps)
        reliable_gradient = gradient_norms[:, 0] > gradient_epsilon
        normals = gradients / gradient_norms.clamp_min(gradient_epsilon)
        displacement = _neighbor_centroids(current, edges) - current
        tangent = displacement - (
            displacement * normals
        ).sum(dim=1, keepdim=True) * normals
        movable = (~feature_mask) & reliable_gradient
        tangent[~movable] = 0.0
        if not bool(torch.any(movable).item()):
            stop_reason = "no movable vertices"
            break

        trial_step = smoothing_step
        accepted = False
        old_cross = _face_cross_products(current, faces_tensor)
        current_residual = values.abs().mean()
        residual_limit = torch.maximum(
            current_residual * 1.05,
            residual_floor,
        )
        quality_tolerance = (
            torch.finfo(dtype).eps
            * torch.maximum(
                current_quality.abs(),
                torch.ones((), dtype=dtype, device=device),
            )
            * 8.0
        )
        for _ in range(8):
            proposed = current + trial_step * tangent
            proposed = _project_to_sdf(
                volume,
                proposed,
                axis_order,
                projection_steps,
                maximum_projection_step,
            )
            proposed[feature_mask] = current[feature_mask]
            proposed_quality, _ = _mesh_quality_energy(
                proposed,
                faces_tensor,
                edges,
            )
            proposed_residual = _sample_volume(
                volume,
                proposed,
                axis_order,
            ).abs().mean()
            valid = _valid_step(
                old_cross,
                proposed,
                faces_tensor,
                minimum_area_squared,
            )
            improves_quality = (
                proposed_quality
                < current_quality - quality_tolerance
            )
            preserves_surface = proposed_residual <= residual_limit
            finite = (
                torch.isfinite(proposed_quality)
                & torch.isfinite(proposed_residual)
                & torch.isfinite(proposed).all()
            )
            if bool(
                (
                    valid
                    & improves_quality
                    & preserves_surface
                    & finite
                ).item()
            ):
                current = proposed
                current_quality = proposed_quality.detach()
                accepted = True
                accepted_iterations += 1
                break
            trial_step *= 0.5
        if not accepted:
            stop_reason = "quality-aware line search converged"
            break

    final_values = _sample_volume(
        volume,
        current,
        axis_order,
    ).detach()
    final_edge_cv = _edge_length_cv(current, edges)
    final_quality = float(current_quality.item())
    print(
        "SDF optimization: {} / {} iterations accepted ({}), "
        "{} feature/boundary vertices locked".format(
            accepted_iterations,
            iterations,
            stop_reason,
            int(feature_mask.sum().item()),
        )
    )
    print(
        "SDF residual mean: {:.6g} -> {:.6g}; edge CV: {:.6g} -> "
        "{:.6g}; quality energy: {:.6g} -> {:.6g}".format(
            initial_sdf_error,
            float(final_values.abs().mean().item()),
            initial_edge_cv,
            final_edge_cv,
            initial_quality,
            final_quality,
        )
    )
    return current.cpu().numpy(), faces_tensor.cpu().numpy()
