"""Shared partition boundaries keep one conforming subdivision and provenance."""

import importlib
from pathlib import Path
import sys
import types
import unittest

import numpy as np


# Exercise the CPU remeshing kernel without constructing the GPU PaMO wrapper.
PACKAGE_NAME = "_pamo_partition_boundary_lineage_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(Path(__file__).parents[1] / "pamo")]
sys.modules[PACKAGE_NAME] = package
original_constrained = importlib.import_module(
    PACKAGE_NAME + ".original_constrained"
)
subdivide = original_constrained._subdivide_labeled_patch_boundaries


class PartitionBoundaryLineageTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
             [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        self.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.labels = np.array([17, 23], dtype=np.int64)

    def test_shared_transition_inserts_one_midpoint_for_both_labels(self):
        vertices, faces, labels, splits, lineage = subdivide(
            self.vertices, self.faces, self.labels, 1.5, return_lineage=True
        )
        self.assertEqual(splits, 1)
        self.assertEqual(len(vertices), 5)
        self.assertEqual(len(faces), 4)
        np.testing.assert_array_equal(vertices[:4], self.vertices)
        np.testing.assert_array_equal(vertices[4], [0.0, 0.0, 0.0])
        self.assertEqual(np.sum(np.all(vertices == vertices[4], axis=1)), 1)
        edge_faces = original_constrained._build_edge_faces(faces)
        for child in ((0, 4), (2, 4)):
            self.assertEqual(len(edge_faces[child]), 2)
            self.assertEqual(set(labels[list(edge_faces[child])]), {17, 23})
        normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        self.assertTrue(np.all(normals[:, 2] > 0.0))
        self.assertEqual(lineage["splits"], splits)

    def test_explicit_crease_inside_one_label_is_subdivided(self):
        vertices = self.vertices.copy()
        vertices[3] = [0.0, 0.0, 1.0]
        labels = np.full(2, 17, dtype=np.int64)
        unchanged = subdivide(vertices, self.faces, labels, 1.5)
        self.assertEqual(unchanged[3], 0)
        output = subdivide(
            vertices, self.faces, labels, 1.5,
            explicit_constraint_edges=np.array([[2, 0], [0, 2]]),
            return_lineage=True,
        )
        self.assertEqual(output[3], 1)
        self.assertEqual(len(output[0]), 5)
        self.assertEqual(len(output[1]), 4)
        np.testing.assert_array_equal(output[2], [17, 17, 17, 17])
        self.assertEqual(len(output[4]["source_constraint_edges"]), 5)
        np.testing.assert_array_equal(output[0][:4], vertices)

    def test_default_return_is_four_items_and_matches_lineage_mode(self):
        ordinary = subdivide(self.vertices, self.faces, self.labels, 0.4)
        detailed = subdivide(
            self.vertices, self.faces, self.labels, 0.4, return_lineage=True
        )
        self.assertEqual(len(ordinary), 4)
        self.assertEqual(len(detailed), 5)
        for first, second in zip(ordinary, detailed):
            np.testing.assert_array_equal(first, second)

    def test_each_parent_is_covered_by_continuous_collinear_children(self):
        vertices, faces, _, splits, lineage = subdivide(
            self.vertices, self.faces, self.labels, 0.4, return_lineage=True
        )
        original_edges = lineage["source_constraint_edges"]
        child_edges = lineage["constraint_edges"]
        parent_indices = lineage["source_edge_indices"]
        self.assertEqual(parent_indices.dtype, np.dtype(np.int64))
        self.assertEqual(child_edges.tolist(), sorted(child_edges.tolist()))
        self.assertEqual(original_edges.tolist(), sorted(original_edges.tolist()))
        self.assertEqual(len(child_edges), len(original_edges) + splits)
        edge_faces = original_constrained._build_edge_faces(faces)
        self.assertTrue(all(tuple(edge) in edge_faces for edge in child_edges))
        np.testing.assert_array_equal(vertices[:len(self.vertices)], self.vertices)
        for parent_index, edge in enumerate(original_edges):
            children = child_edges[parent_indices == parent_index]
            self.assertGreater(len(children), 0)
            start, end = self.vertices[edge]
            vector = end - start
            points = vertices[children]
            parameter = np.einsum("ijk,k->ij", points - start, vector) / np.dot(vector, vector)
            projected = start + parameter[:, :, None] * vector
            np.testing.assert_allclose(points, projected, atol=1e-14)
            intervals = np.sort(parameter, axis=1)
            intervals = intervals[np.argsort(intervals[:, 0])]
            self.assertAlmostEqual(intervals[0, 0], 0.0)
            self.assertAlmostEqual(intervals[-1, 1], 1.0)
            np.testing.assert_allclose(intervals[:-1, 1], intervals[1:, 0], atol=1e-14)
            self.assertTrue(np.all(np.linalg.norm(points[:, 1] - points[:, 0], axis=1) <= 0.4 * (1.0 + 1e-8)))

    def test_budget_exhaustion_raises_without_mutating_inputs(self):
        originals = tuple(value.copy() for value in (self.vertices, self.faces, self.labels))
        with self.assertRaisesRegex(RuntimeError, "maximum_splits=1"):
            subdivide(
                self.vertices, self.faces, self.labels, 0.4,
                maximum_splits=1, return_lineage=True,
            )
        for actual, expected in zip((self.vertices, self.faces, self.labels), originals):
            np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(RuntimeError, "maximum_splits=0"):
            subdivide(self.vertices, self.faces, self.labels, 1.5, maximum_splits=0)
        self.assertEqual(subdivide(
            self.vertices, self.faces, self.labels, 1.5, maximum_splits=1
        )[3], 1)
        self.assertEqual(subdivide(
            self.vertices, self.faces, self.labels, 3.0, maximum_splits=0
        )[3], 0)

    def test_invalid_constraints_and_limits_are_rejected(self):
        for edges in ([[1, 3]], [[0, 4]], [[0, 0]], [[0.0, 2.5]], [0, 2]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                subdivide(
                    self.vertices, self.faces, self.labels, 1.5,
                    explicit_constraint_edges=edges,
                )
        for limit in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                subdivide(self.vertices, self.faces, self.labels, limit)
            with self.subTest(patch_limit=limit), self.assertRaises(ValueError):
                subdivide(
                    self.vertices, self.faces, self.labels, 1.5,
                    patch_edge_limits={17: limit},
                )
        for budget in (-1, 0.5, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                subdivide(
                    self.vertices, self.faces, self.labels, 1.5,
                    maximum_splits=budget,
                )

    def test_stricter_incident_patch_limit_controls_shared_children(self):
        vertices, _, _, _, lineage = subdivide(
            self.vertices, self.faces, self.labels, 3.0,
            patch_edge_limits={17: 0.5}, return_lineage=True,
        )
        parents = lineage["source_constraint_edges"]
        parent = int(np.flatnonzero(np.all(parents == [0, 2], axis=1))[0])
        children = lineage["constraint_edges"][lineage["source_edge_indices"] == parent]
        self.assertEqual(len(children), 4)
        np.testing.assert_allclose(
            np.linalg.norm(vertices[children[:, 1]] - vertices[children[:, 0]], axis=1),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
