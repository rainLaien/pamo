import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


MODULE_PATH = (
    Path(__file__).parents[1]
    / "pamo"
    / "segment_query.py"
)
SPEC = importlib.util.spec_from_file_location("segment_query", MODULE_PATH)
segment_query = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = segment_query
SPEC.loader.exec_module(segment_query)
closest_points_on_segments = segment_query.closest_points_on_segments


class BruteMidpointTree:
    """Small cKDTree-compatible test double."""

    def __init__(self, centers):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.n = len(self.centers)

    def query(self, points, k):
        points = np.atleast_2d(points)
        distances = np.linalg.norm(
            points[:, None, :] - self.centers[None, :, :],
            axis=2,
        )
        order = np.argsort(distances, axis=1)[:, :k]
        selected = np.take_along_axis(distances, order, axis=1)
        if k == 1:
            return selected[:, 0], order[:, 0]
        return selected, order

    def query_ball_point(self, point, radius):
        distances = np.linalg.norm(self.centers - point, axis=1)
        return np.flatnonzero(distances <= radius).tolist()


def brute_force_query(points, segments):
    points = np.asarray(points, dtype=np.float64)
    starts = segments[:, 0]
    vectors = segments[:, 1] - starts
    lengths_squared = np.einsum("mi,mi->m", vectors, vectors)
    offsets = points[:, None, :] - starts[None, :, :]
    parameters = np.divide(
        np.einsum("nmi,mi->nm", offsets, vectors),
        lengths_squared[None, :],
        out=np.zeros((len(points), len(segments)), dtype=np.float64),
        where=lengths_squared[None, :] > 0.0,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    projected = (
        starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
    )
    distances_squared = np.sum(
        (projected - points[:, None, :]) ** 2,
        axis=2,
    )
    best = np.argmin(distances_squared, axis=1)
    rows = np.arange(len(points))
    return projected[rows, best], np.sqrt(distances_squared[rows, best]), best


class SegmentQueryTest(unittest.TestCase):
    def test_finds_long_segment_missed_by_fixed_midpoint_candidates(self):
        decoy_points = np.column_stack(
            (
                np.arange(1.0, 41.0),
                np.full(40, 10.0),
                np.zeros(40),
            )
        )
        decoys = np.stack((decoy_points, decoy_points), axis=1)
        long_segment = np.array([[[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]])
        segments = np.concatenate((decoys, long_segment), axis=0)
        tree = BruteMidpointTree(segments.mean(axis=1))

        closest, distances, segment_ids = closest_points_on_segments(
            np.array([[0.0, 0.1, 0.0]]),
            segments,
            tree,
            candidate_count=32,
        )

        np.testing.assert_allclose(closest, [[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(distances, [0.1])
        np.testing.assert_array_equal(segment_ids, [40])

    def test_matches_brute_force_for_mixed_and_degenerate_segments(self):
        rng = np.random.default_rng(7)
        segments = rng.normal(size=(80, 2, 3))
        segments[5, 1] = segments[5, 0]
        segments[27] *= 20.0
        points = rng.normal(size=(25, 3))
        tree = BruteMidpointTree(segments.mean(axis=1))

        actual = closest_points_on_segments(
            points,
            segments,
            tree,
            candidate_count=8,
        )
        expected = brute_force_query(points, segments)

        np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-12)
        np.testing.assert_array_equal(actual[2], expected[2])

    @unittest.skipIf(cKDTree is None, "SciPy is unavailable")
    def test_matches_brute_force_with_scipy_midpoint_tree(self):
        rng = np.random.default_rng(23)
        segments = rng.uniform(-5.0, 5.0, size=(250, 2, 3))
        segments[13] = [[-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
        points = rng.uniform(-2.0, 2.0, size=(100, 3))

        actual = closest_points_on_segments(
            points,
            segments,
            cKDTree(segments.mean(axis=1)),
            candidate_count=16,
        )
        expected = brute_force_query(points, segments)

        np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-12)
        np.testing.assert_array_equal(actual[2], expected[2])

    def test_validates_inputs_and_handles_empty_queries(self):
        segments = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        tree = BruteMidpointTree(segments.mean(axis=1))
        closest, distances, segment_ids = closest_points_on_segments(
            np.empty((0, 3)),
            segments,
            tree,
        )
        self.assertEqual(closest.shape, (0, 3))
        self.assertEqual(distances.shape, (0,))
        self.assertEqual(segment_ids.shape, (0,))

        with self.assertRaisesRegex(ValueError, "At least one segment"):
            closest_points_on_segments(
                np.zeros((1, 3)),
                np.empty((0, 2, 3)),
                BruteMidpointTree(np.empty((0, 3))),
            )
        with self.assertRaisesRegex(ValueError, "Candidate count"):
            closest_points_on_segments(
                np.zeros((1, 3)),
                segments,
                tree,
                candidate_count=0,
            )


if __name__ == "__main__":
    unittest.main()
