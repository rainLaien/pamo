"""Topology changes retain a refinable source mesh around overlong edges."""
import unittest

import torch

from simp_cuda.tests.test_surface_sample import surface_sample


@unittest.skipIf(surface_sample is None, "Surface sampling dependencies are unavailable")
class PreRefinementConstraintsTest(unittest.TestCase):
    def collapse_fixture(self, interior_first=False):
        vertices = torch.tensor(
            [[0., 0., 0.], [3., 0., 0.], [0., 3., 0.], [.05, .05, 0.]],
            dtype=torch.float64,
        )
        faces = torch.tensor([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
        if interior_first:
            order = torch.tensor([3, 0, 1, 2])
            inverse = torch.argsort(order)
            vertices, faces = vertices[order], inverse[faces]
        return vertices, faces

    def collapse(self, vertices, faces, target, **options):
        return surface_sample._gpu_collapse_short_edges(
            vertices, faces, torch.zeros(len(faces), dtype=torch.long),
            torch.arange(len(faces)), torch.full((len(vertices),), -1),
            minimum_edge_length=.2, maximum_edge_length=target, passes=2,
            compact_vertices=False, **options,
        )

    def maximum_edge(self, vertices, faces):
        edges = surface_sample._unique_edges_torch(faces)
        return float(torch.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], dim=1).max())

    def test_pre_collapse_leaves_long_source_cavity_intact_in_both_directions(self):
        # Removing the interior point improves this planar fan's quality, but
        # retains the outer triangle's long sides. Reorder IDs to exercise
        # both choices of retained endpoint under the boundary lock.
        for interior_first in (False, True):
            with self.subTest(interior_first=interior_first):
                vertices, faces = self.collapse_fixture(interior_first)
                ordinary = self.collapse(vertices, faces, 1.)
                self.assertEqual(ordinary[-1], 1)
                self.assertGreater(self.maximum_edge(ordinary[0], ordinary[1]), 1.)
                constrained = self.collapse(
                    vertices, faces, 1., preserve_existing_maximum_edge=False,
                )
                self.assertEqual(constrained[-1], 0)
                self.assertTrue(torch.equal(constrained[0], vertices))
                self.assertTrue(torch.equal(constrained[1], faces))

    def test_pre_collapse_still_improves_a_cavity_within_target(self):
        vertices, faces = self.collapse_fixture()
        result = self.collapse(vertices, faces, 5., preserve_existing_maximum_edge=False)
        self.assertEqual(result[-1], 1)
        self.assertEqual(len(result[1]), 1)
        self.assertLessEqual(self.maximum_edge(result[0], result[1]), 5.)
        self.assertTrue(torch.equal(result[0], vertices))
        self.assertEqual(set(result[1].reshape(-1).tolist()), {0, 1, 2})
        self.assertGreater(
            float(surface_sample._triangle_quality_torch(result[0], result[1]).min()),
            float(surface_sample._triangle_quality_torch(vertices, faces).min()),
        )

    def test_short_flip_diagonal_does_not_exempt_long_replacement_sides(self):
        vertices = torch.tensor(
            [[0., 0., 0.], [3., 0., 0.], [0., 1., 0.], [1., 1., 0.]],
            dtype=torch.float64,
        )
        faces = torch.tensor([[0, 1, 2], [2, 1, 3]])
        labels = torch.zeros(2, dtype=torch.long)
        ordinary, count = surface_sample._gpu_flip_quality_edges(
            vertices, faces, labels, 1, maximum_edge_length=2.,
            preserve_existing_maximum_edge=True,
        )
        self.assertEqual(count, 1)
        self.assertLess(float(torch.linalg.norm(vertices[0] - vertices[3])), 2.)
        self.assertGreater(self.maximum_edge(vertices, ordinary), 2.)
        blocked, count = surface_sample._gpu_flip_quality_edges(
            vertices, faces, labels, 1, maximum_edge_length=2.,
            preserve_existing_maximum_edge=False,
        )
        self.assertEqual(count, 0)
        self.assertTrue(torch.equal(blocked, faces))
        permitted, count = surface_sample._gpu_flip_quality_edges(
            vertices, faces, labels, 1, maximum_edge_length=3.,
            preserve_existing_maximum_edge=False,
        )
        self.assertEqual(count, 1)
        self.assertLessEqual(self.maximum_edge(vertices, permitted), 3.)
        self.assertGreater(
            float(surface_sample._triangle_quality_torch(vertices, permitted).min()),
            float(surface_sample._triangle_quality_torch(vertices, faces).min()),
        )


if __name__ == "__main__":
    unittest.main()
