import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "pamo_safe_project"
    / "hinge_topology.py"
)
SPEC = importlib.util.spec_from_file_location("hinge_topology", MODULE_PATH)
hinge_topology = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hinge_topology
SPEC.loader.exec_module(hinge_topology)
build_hinge_topology = hinge_topology.build_hinge_topology


class HingeTopologyTest(unittest.TestCase):
    def test_builds_oriented_hinge_for_open_quad(self):
        faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)

        topology = build_hinge_topology(faces)

        np.testing.assert_array_equal(
            topology.indices,
            np.array([[0, 1, 2, 3]], dtype=np.int32),
        )
        self.assertEqual(topology.unique_edge_count, 5)
        self.assertEqual(topology.boundary_edge_count, 4)
        self.assertEqual(topology.nonmanifold_edge_count, 0)
        self.assertEqual(topology.inconsistent_winding_edge_count, 0)

    def test_excludes_nonmanifold_and_inconsistently_wound_edges(self):
        inconsistent = build_hinge_topology(
            np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        )
        self.assertEqual(inconsistent.indices.shape, (0, 4))
        self.assertEqual(inconsistent.inconsistent_winding_edge_count, 1)

        nonmanifold = build_hinge_topology(
            np.array(
                [
                    [0, 1, 2],
                    [1, 0, 3],
                    [0, 1, 4],
                ],
                dtype=np.int32,
            )
        )
        self.assertEqual(nonmanifold.indices.shape, (0, 4))
        self.assertEqual(nonmanifold.nonmanifold_edge_count, 1)

    def test_handles_more_faces_than_the_old_cuda_grid_limit(self):
        n_u = 190
        n_v = 180
        i, j = np.meshgrid(np.arange(n_u), np.arange(n_v), indexing="ij")
        a = i * n_v + j
        b = ((i + 1) % n_u) * n_v + j
        c = ((i + 1) % n_u) * n_v + (j + 1) % n_v
        d = i * n_v + (j + 1) % n_v
        faces = np.stack(
            (
                np.stack((a, b, c), axis=-1),
                np.stack((a, c, d), axis=-1),
            ),
            axis=2,
        ).reshape(-1, 3)
        self.assertGreater(faces.shape[0], 65535)

        topology = build_hinge_topology(faces)

        expected_edges = faces.shape[0] * 3 // 2
        self.assertEqual(topology.indices.shape, (expected_edges, 4))
        self.assertEqual(topology.unique_edge_count, expected_edges)
        self.assertEqual(topology.boundary_edge_count, 0)
        self.assertEqual(topology.nonmanifold_edge_count, 0)
        self.assertEqual(topology.inconsistent_winding_edge_count, 0)


if __name__ == "__main__":
    unittest.main()
