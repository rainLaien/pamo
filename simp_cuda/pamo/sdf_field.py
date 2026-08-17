import numpy as np
import torch


SDF_MODES = ("auto", "exact", "repair")


def resolve_sdf_mode(mesh, requested_mode):
    """
    Resolve the requested SDF semantics for an input mesh.

    ``exact`` extracts the zero set of the input surface and therefore needs a
    closed, consistently wound mesh. ``repair`` extracts an unsigned-distance
    envelope and is suitable for intentionally closing an invalid/open input.
    """
    mode = str(requested_mode).lower()
    if mode not in SDF_MODES:
        raise ValueError(
            "Unknown SDF mode '{}'. Expected one of: {}.".format(
                requested_mode,
                ", ".join(SDF_MODES),
            )
        )

    watertight = bool(mesh.is_watertight)
    winding_consistent = bool(mesh.is_winding_consistent)
    exact_compatible = watertight and winding_consistent

    if mode == "exact" and not exact_compatible:
        problems = []
        if not watertight:
            problems.append("the input is not watertight")
        if not winding_consistent:
            problems.append("face winding is inconsistent")
        raise ValueError(
            "Exact SDF=0 remeshing requires a watertight, consistently wound "
            "triangle mesh; {}. Repair the input first or use "
            "sdf_mode='repair' for an offset repair envelope.".format(
                " and ".join(problems)
            )
        )

    if mode == "auto":
        if exact_compatible:
            return "exact", "watertight input with consistent face winding"

        problems = []
        if not watertight:
            problems.append("input is not watertight")
        if not winding_consistent:
            problems.append("face winding is inconsistent")
        return "repair", " and ".join(problems)

    return mode, "explicitly requested"


def resolve_original_constraint_mode(
    mesh,
    requested_mode,
    allow_open_surface=False,
):
    """Resolve topology semantics for original-connectivity refinement.

    Original-constrained refinement does not evaluate an SDF: it operates on
    the input connectivity directly. A consistently wound open surface is
    therefore supported when the caller explicitly opts in; boundary and
    non-manifold edges remain hard constraints. Repair envelopes are never
    compatible because they replace the input connectivity and feature
    lineages.
    """
    mode = str(requested_mode).lower()
    if mode not in SDF_MODES:
        raise ValueError(
            "Unknown SDF mode '{}'. Expected one of: {}.".format(
                requested_mode,
                ", ".join(SDF_MODES),
            )
        )
    if mode == "repair":
        raise ValueError(
            "Original-constrained refinement cannot use an SDF repair "
            "envelope because it must retain the input connectivity."
        )

    watertight = bool(mesh.is_watertight)
    winding_consistent = bool(mesh.is_winding_consistent)
    if watertight and winding_consistent:
        return "exact", "watertight input with consistent face winding"

    if not allow_open_surface:
        problems = []
        if not watertight:
            problems.append("the input is not watertight")
        if not winding_consistent:
            problems.append("face winding is inconsistent")
        raise ValueError(
            "Strict original-constrained refinement requires a watertight, "
            "consistently wound mesh; {}. For a thin sheet, enable "
            "allow_open_surface so its boundary and non-manifold edges are "
            "retained as hard constraints.".format(" and ".join(problems))
        )

    if not winding_consistent:
        raise ValueError(
            "Open-surface original-constrained refinement requires "
            "consistent face winding."
        )

    edge_counts = np.bincount(
        np.asarray(mesh.edges_unique_inverse, dtype=np.int64),
        minlength=len(mesh.edges_unique),
    )
    boundary_count = int(np.count_nonzero(edge_counts == 1))
    nonmanifold_count = int(np.count_nonzero(edge_counts > 2))
    return (
        "open",
        "consistently wound surface with {} hard boundary edge(s) and {} "
        "hard non-manifold edge(s)".format(
            boundary_count,
            nonmanifold_count,
        ),
    )


def fast_winding_inside_mask(
    vertices,
    faces,
    resolution,
    threshold=0.5,
    max_query_points=4 * 1024 * 1024,
):
    """
    Classify cell-center samples with the generalized winding number.

    The query is evaluated in x slabs so a 256^3 grid does not require one
    giant ``(R^3, 3)`` allocation. Absolute winding supports globally reversed
    components while retaining the cancellation semantics of oriented cavity
    shells.
    """
    import igl

    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    resolution = int(resolution)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("SDF sign vertices must have shape (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("SDF sign faces must have shape (M, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError("SDF sign vertices contain NaN or infinity.")

    samples_per_slab = resolution * resolution
    slab_depth = max(
        1,
        min(
            resolution,
            int(max_query_points) // samples_per_slab,
        ),
    )
    axis = (
        np.arange(resolution, dtype=np.float32) + np.float32(0.5)
    ) / np.float32(resolution)
    inside = np.empty(
        (resolution, resolution, resolution),
        dtype=np.bool_,
    )

    for x_start in range(0, resolution, slab_depth):
        x_stop = min(x_start + slab_depth, resolution)
        queries = np.empty(
            (x_stop - x_start, resolution, resolution, 3),
            dtype=np.float32,
        )
        queries[..., 0] = axis[x_start:x_stop, None, None]
        queries[..., 1] = axis[None, :, None]
        queries[..., 2] = axis[None, None, :]
        winding = igl.fast_winding_number_for_meshes(
            vertices,
            faces,
            queries.reshape(-1, 3),
        )
        inside[x_start:x_stop] = (
            np.abs(np.asarray(winding)).reshape(
                x_stop - x_start,
                resolution,
                resolution,
            )
            > float(threshold)
        )

    inside_count = int(np.count_nonzero(inside))
    if inside_count == 0:
        raise RuntimeError(
            "Fast-winding sign classification found no interior grid cells. "
            "Increase the remesh resolution or use sdf_mode='repair' if an "
            "offset repair envelope is intended."
        )

    boundary_inside = (
        np.any(inside[0])
        or np.any(inside[-1])
        or np.any(inside[:, 0])
        or np.any(inside[:, -1])
        or np.any(inside[:, :, 0])
        or np.any(inside[:, :, -1])
    )
    if boundary_inside:
        raise RuntimeError(
            "The SDF interior reaches the volume boundary. This indicates an "
            "invalid sign classification or insufficient normalization margin."
        )

    return inside


def apply_inside_sign(unsigned_distance, inside):
    """Apply an inside-negative sign mask to a CUDA unsigned-distance grid."""
    if tuple(unsigned_distance.shape) != tuple(inside.shape):
        raise ValueError(
            "SDF distance and sign grids have different shapes: {} and {}."
            .format(tuple(unsigned_distance.shape), tuple(inside.shape))
        )
    inside_tensor = torch.as_tensor(
        inside,
        dtype=torch.bool,
        device=unsigned_distance.device,
    )
    return torch.where(
        inside_tensor,
        -unsigned_distance,
        unsigned_distance,
    )
