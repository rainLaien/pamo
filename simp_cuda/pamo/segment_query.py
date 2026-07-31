"""Exact point-to-segment queries accelerated by a midpoint spatial index."""

import numpy as np


def _project_to_candidates(points, segments, candidate_ids):
    """Project each point onto its row of candidate segments."""
    candidates = segments[candidate_ids]
    starts = candidates[:, :, 0]
    vectors = candidates[:, :, 1] - starts
    lengths_squared = np.einsum("nki,nki->nk", vectors, vectors)
    offsets = points[:, None, :] - starts
    parameters = np.divide(
        np.einsum("nki,nki->nk", offsets, vectors),
        lengths_squared,
        out=np.zeros_like(lengths_squared),
        where=lengths_squared > 0.0,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = starts + parameters[:, :, None] * vectors
    distances_squared = np.einsum(
        "nki,nki->nk",
        closest - points[:, None, :],
        closest - points[:, None, :],
    )
    best_columns = np.argmin(distances_squared, axis=1)
    rows = np.arange(len(points))
    return (
        closest[rows, best_columns],
        distances_squared[rows, best_columns],
        candidate_ids[rows, best_columns],
    )


def closest_points_on_segments(
    points,
    segments,
    midpoint_tree,
    candidate_count=32,
):
    """
    Return exact closest points, distances, and segment indices.

    A fixed number of nearest segment midpoints is only an approximation: a
    long segment can pass close to a query while its midpoint is far away.
    This implementation first evaluates the nearest midpoint candidates, then
    uses the segment half-length bound

    ``distance(point, segment) >= distance(point, midpoint) - half_length``

    to certify the result. Only uncertified queries are expanded with a radius
    search. The common path remains vectorized while the returned nearest
    segment is exact (up to floating-point arithmetic).

    ``midpoint_tree`` must index ``segments.mean(axis=1)`` and provide the
    ``query`` and ``query_ball_point`` methods of ``scipy.spatial.cKDTree``.
    """
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Query points must have shape (n, 3).")
    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError("Segments must have shape (m, 2, 3).")
    if len(segments) == 0:
        raise ValueError("At least one segment is required.")
    if not np.isfinite(points).all() or not np.isfinite(segments).all():
        raise ValueError("Points and segments must contain only finite values.")

    candidate_count = int(candidate_count)
    if candidate_count <= 0:
        raise ValueError("Candidate count must be positive.")
    if hasattr(midpoint_tree, "n") and int(midpoint_tree.n) != len(segments):
        raise ValueError(
            "The midpoint tree size does not match the segment count."
        )
    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    candidate_count = min(candidate_count, len(segments))
    center_distances, candidate_ids = midpoint_tree.query(
        points,
        k=candidate_count,
    )
    center_distances = np.asarray(center_distances, dtype=np.float64)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if candidate_count == 1:
        center_distances = center_distances[:, None]
        candidate_ids = candidate_ids[:, None]

    closest, best_squared, best_ids = _project_to_candidates(
        points,
        segments,
        candidate_ids,
    )
    if candidate_count == len(segments):
        return closest, np.sqrt(best_squared), best_ids

    half_lengths = (
        np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) * 0.5
    )
    maximum_half_length = float(half_lengths.max())
    best_distances = np.sqrt(best_squared)

    # Every unqueried midpoint is at least as far away as the final queried
    # midpoint. If that lower bound already exceeds the best segment distance,
    # the vectorized candidate result is globally exact.
    unqueried_lower_bound = (
        center_distances[:, -1] - maximum_half_length
    )
    scale = np.maximum(
        np.maximum(best_distances, center_distances[:, -1]),
        maximum_half_length,
    )
    tolerance = np.finfo(np.float64).eps * np.maximum(scale, 1.0) * 8.0
    uncertain_rows = np.flatnonzero(
        best_distances > unqueried_lower_bound + tolerance
    )

    for row in uncertain_rows:
        # A segment outside this radius cannot beat the current result, even
        # when it has the maximum half length in the data set.
        radius = np.nextafter(
            best_distances[row] + maximum_half_length,
            np.inf,
        )
        expanded_ids = np.asarray(
            midpoint_tree.query_ball_point(points[row], radius),
            dtype=np.int64,
        )
        if expanded_ids.size == 0:
            continue
        expanded_closest, expanded_squared, expanded_best_ids = (
            _project_to_candidates(
                points[row : row + 1],
                segments,
                expanded_ids.reshape(1, -1),
            )
        )
        if expanded_squared[0] < best_squared[row]:
            closest[row] = expanded_closest[0]
            best_squared[row] = expanded_squared[0]
            best_ids[row] = expanded_best_ids[0]
            best_distances[row] = np.sqrt(expanded_squared[0])

    return closest, np.sqrt(best_squared), best_ids
