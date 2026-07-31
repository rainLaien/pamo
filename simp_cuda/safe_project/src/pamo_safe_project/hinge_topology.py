from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HingeTopology:
    """Oriented hinge indices and a summary of the input edge topology."""

    indices: np.ndarray
    unique_edge_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    inconsistent_winding_edge_count: int


def build_hinge_topology(faces: np.ndarray) -> HingeTopology:
    """Build one oriented hinge for every consistently wound manifold edge.

    Each returned row is ``[opposite_left, edge_low, edge_high,
    opposite_right]``. The first incident face traverses the shared edge from
    ``edge_low`` to ``edge_high`` and the second traverses it in reverse.

    The implementation groups the three directed edges of each triangle by
    their undirected vertex pair. It replaces the previous all-pairs triangle
    scan, whose quadratic launch size could exceed CUDA grid limits for dense
    remeshes.
    """

    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Expected triangle faces with shape (n, 3), got {faces.shape}"
        )
    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError(f"Face indices must be integers, got {faces.dtype}")
    if faces.size == 0:
        return HingeTopology(
            indices=np.empty((0, 4), dtype=np.int32),
            unique_edge_count=0,
            boundary_edge_count=0,
            nonmanifold_edge_count=0,
            inconsistent_winding_edge_count=0,
        )
    if np.min(faces) < 0:
        raise ValueError("Face indices must be non-negative")
    if np.max(faces) > np.iinfo(np.int32).max:
        raise ValueError("Face indices exceed the int32 range used by Warp")

    repeated_vertex = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    if np.any(repeated_vertex):
        first_bad = int(np.flatnonzero(repeated_vertex)[0])
        raise ValueError(
            f"Face {first_bad} repeats a vertex index: {faces[first_bad].tolist()}"
        )

    # Interleave a face's three directed edges so the corresponding opposite
    # vertex stays at the same flattened index.
    edge_starts = np.ascontiguousarray(faces[:, [0, 1, 2]]).reshape(-1)
    edge_ends = np.ascontiguousarray(faces[:, [1, 2, 0]]).reshape(-1)
    opposite_vertices = np.ascontiguousarray(faces[:, [2, 0, 1]]).reshape(-1)

    edge_lows = np.minimum(edge_starts, edge_ends)
    edge_highs = np.maximum(edge_starts, edge_ends)
    order = np.lexsort((edge_highs, edge_lows))
    sorted_lows = edge_lows[order]
    sorted_highs = edge_highs[order]

    group_start_mask = np.empty(order.shape[0], dtype=bool)
    group_start_mask[0] = True
    group_start_mask[1:] = (
        (sorted_lows[1:] != sorted_lows[:-1])
        | (sorted_highs[1:] != sorted_highs[:-1])
    )
    group_starts = np.flatnonzero(group_start_mask)
    group_counts = np.diff(
        np.append(group_starts, np.array([order.shape[0]], dtype=group_starts.dtype))
    )

    boundary_edge_count = int(np.count_nonzero(group_counts == 1))
    nonmanifold_edge_count = int(np.count_nonzero(group_counts > 2))

    manifold_group_starts = group_starts[group_counts == 2]
    first_entries = order[manifold_group_starts]
    second_entries = order[manifold_group_starts + 1]

    first_forward = edge_starts[first_entries] < edge_ends[first_entries]
    second_forward = edge_starts[second_entries] < edge_ends[second_entries]
    consistent_winding = first_forward != second_forward
    inconsistent_winding_edge_count = int(
        np.count_nonzero(~consistent_winding)
    )

    first_entries = first_entries[consistent_winding]
    second_entries = second_entries[consistent_winding]
    first_forward = first_forward[consistent_winding]

    forward_entries = np.where(first_forward, first_entries, second_entries)
    reverse_entries = np.where(first_forward, second_entries, first_entries)

    indices = np.column_stack(
        (
            opposite_vertices[forward_entries],
            edge_starts[forward_entries],
            edge_ends[forward_entries],
            opposite_vertices[reverse_entries],
        )
    ).astype(np.int32, copy=False)

    return HingeTopology(
        indices=np.ascontiguousarray(indices),
        unique_edge_count=int(group_starts.shape[0]),
        boundary_edge_count=boundary_edge_count,
        nonmanifold_edge_count=nonmanifold_edge_count,
        inconsistent_winding_edge_count=inconsistent_winding_edge_count,
    )
