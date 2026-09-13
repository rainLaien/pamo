"""Reusing rejected cavities must agree with fully rechecking each pass."""
import unittest

import numpy as np
import torch

from simp_cuda.tests.test_surface_sample import surface_sample


@unittest.skipIf(surface_sample is None, "Surface sampling dependencies are unavailable")
class CollapseRejectionCacheTests(unittest.TestCase):
    def check_grid(self, split_labels):
        side = 9
        rng = np.random.default_rng(41)
        y, x = np.mgrid[:side, :side]
        points = np.c_[x.reshape(-1), y.reshape(-1), np.zeros(side * side)]
        boundary = (x == 0) | (y == 0) | (x == side - 1) | (y == side - 1)
        points[~boundary.reshape(-1), :2] += rng.uniform(
            -.35, .35, (np.count_nonzero(~boundary), 2))
        faces, labels = [], []
        for row in range(side - 1):
            for column in range(side - 1):
                a = row * side + column
                faces.extend(((a, a + 1, a + side), (a + 1, a + side + 1, a + side)))
                labels.extend([int(split_labels and column >= side // 2)] * 2)
        vertices = torch.as_tensor(points, dtype=torch.float64)
        faces = torch.tensor(faces, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        support = torch.full((len(vertices),), -1, dtype=torch.long)
        support[torch.from_numpy(boundary.reshape(-1))] = -2
        arguments = (vertices, faces, labels, torch.arange(len(faces)), support)
        options = dict(minimum_edge_length=1.25, maximum_edge_length=2.5,
                       minimum_collapse_quality=.6, compact_vertices=False,
                       preserve_existing_maximum_edge=False)
        cached = surface_sample._gpu_collapse_short_edges(*arguments, passes=6, **options)
        # Separate one-pass calls cannot retain the rejection cache. They check
        # every remaining candidate against each newly modified neighborhood.
        rechecked = arguments
        count = 0
        for _ in range(6):
            result = surface_sample._gpu_collapse_short_edges(*rechecked, passes=1, **options)
            rechecked = result[:5]
            count += result[-1]
        self.assertGreater(count, 0)
        self.assertEqual(cached[-1], count)
        for first, second in zip(cached[:5], rechecked):
            self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(cached[0], vertices))

    def test_changed_neighborhoods_are_rechecked(self):
        self.check_grid(False)

    def test_shared_patch_boundaries_are_preserved_across_cached_passes(self):
        self.check_grid(True)


if __name__ == '__main__':
    unittest.main()
