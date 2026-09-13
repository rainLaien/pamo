"""Exact cache lookup behavior with mixed hits, misses and collisions."""
import unittest
import numpy as np
import torch
from simp_cuda.tests.test_cuda_surface_project import cuda_module


@unittest.skipUnless(cuda_module is not None, 'CUDA and Warp required')
class CudaValidationCacheCompactionTests(unittest.TestCase):
    def test_mixed_hit_and_miss_order_with_collisions(self):
        reference_vertices = np.array([[0., 0., 0.], [4., 0., 0.],
                                       [4., 4., 0.], [0., 4., 0.]])
        reference_faces = np.array([[0, 1, 2], [0, 2, 3]])
        projector = cuda_module.CudaReferencePatchProjector(
            reference_vertices, reference_faces, np.array([2**40, 2**40]))
        rng = np.random.default_rng(9153)
        coordinates = rng.uniform(.2, 3.8, (100, 3))
        coordinates[:, 2] = 0.
        coordinates[::7, 2] = .2
        vertices = torch.as_tensor(coordinates, device='cuda')
        queries = torch.as_tensor(rng.integers(0, 100, (2048, 3)), device='cuda')
        for capacity in (4, 4096):
            with self.subTest(capacity=capacity):
                cache = projector.make_triangle_cache(vertices, .01, 5.)
                cache.capacity = capacity
                for rows in (queries[:800], queries[::3], queries.flip(0), queries[:800]):
                    labels = torch.full((len(rows),), 2**40, device='cuda')
                    expected = projector.valid_triangles(vertices[rows], labels, .01, 5.)
                    self.assertTrue(torch.equal(cache.check(rows, labels), expected))
                empty = torch.empty((0, 3), device='cuda', dtype=torch.long)
                self.assertEqual(cache.check(empty, torch.empty(0, device='cuda', dtype=torch.long)).numel(), 0)


if __name__ == '__main__':
    unittest.main()
