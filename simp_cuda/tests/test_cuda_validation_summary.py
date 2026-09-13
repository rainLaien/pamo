"""Fast final-validation flags retain the full geometric report's semantics."""
import unittest
import numpy as np
import torch
from simp_cuda.tests.test_cuda_surface_project import cuda_module


@unittest.skipUnless(cuda_module is not None, 'CUDA and Warp required')
class CudaValidationSummaryTests(unittest.TestCase):
    def compare(self, projector, triangles, labels, distance, degrees):
        expected, details = projector.valid_triangles(triangles, labels, distance, degrees, True)
        actual, ties = projector.valid_triangles_with_ties(triangles, labels, distance, degrees)
        self.assertTrue(torch.equal(actual, expected))
        np.testing.assert_array_equal(ties.cpu().numpy(), details['normal_reference_ties_resolved'])
        return expected, details

    def test_ties_on_valid_and_invalid_faces_and_exact_details_unchanged(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        projector = cuda_module.CudaReferencePatchProjector(
            vertices, np.array([[0, 2, 1], [0, 1, 2]]), np.array([2**40, 2**40]))
        queries = torch.as_tensor(np.array([
            vertices,
            [[-.1, 0., 0.], [1.1, 0., 0.], [0., 1.1, 0.]],
            vertices + [0., 0., .02],
            vertices[[0, 2, 1]],
        ]), device='cuda')
        labels = torch.full((len(queries),), 2**40, device='cuda')
        expected, details = self.compare(projector, queries, labels, .01, 5.)
        self.assertEqual(expected.tolist(), [True, False, False, True])
        self.assertEqual(details['normal_reference_ties_resolved'].tolist(), [True, True, False, False])
        self.assertGreater(details['maximum_sample_distance'][1], .01)
        self.assertGreater(details['maximum_sample_distance'][2], .01)
        _, repeated = projector.valid_triangles(queries, labels, .01, 5., True)
        for key in details:
            np.testing.assert_array_equal(repeated[key], details[key])

    def test_random_thin_faces_scales_and_disabled_limits(self):
        rng = np.random.default_rng(7920)
        source = rng.normal(size=(47, 3, 3))
        source[:5, 2] = source[:5, 0] + .3*(source[:5, 1]-source[:5, 0]) + rng.normal(size=(5, 3))*1e-9
        faces = np.arange(source.size//3).reshape(-1, 3)
        labels = np.where(np.arange(len(source))%2, 7, 2**40)
        for scale in (1e-3, 1., 1e3):
            vertices = (source.reshape(-1, 3)+(3000., 20., -15.))*scale
            projector = cuda_module.CudaReferencePatchProjector(vertices, faces, labels)
            queries = torch.as_tensor(vertices[faces].copy(), device='cuda')
            queries[5:10] += scale*.01
            queries[10:15] = queries[10:15].flip(1)
            query_labels = torch.as_tensor(labels, device='cuda')
            for distance, degrees in ((scale*.02, 5.), (0., 90.), (None, 5.), (scale*.02, None), (None, None)):
                with self.subTest(scale=scale, distance=distance, degrees=degrees):
                    self.compare(projector, queries, query_labels, distance, degrees)

    def test_empty_queries_and_invalid_inputs(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        projector = cuda_module.CudaReferencePatchProjector(vertices, np.array([[0, 1, 2]]), np.array([3]))
        triangles = torch.empty((0, 3, 3), dtype=torch.float64, device='cuda')
        mask, ties = projector.valid_triangles_with_ties(triangles, torch.empty(0, dtype=torch.long, device='cuda'), .01, 5.)
        self.assertEqual(mask.shape, (0,))
        self.assertEqual(ties.shape, (0,))
        query = torch.as_tensor(vertices[None], device='cuda')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            projector.valid_triangles_with_ties(query, torch.tensor([9], device='cuda'), .01, 5.)
        with self.assertRaisesRegex(ValueError, 'float64'):
            projector.valid_triangles_with_ties(query.float(), torch.tensor([3], device='cuda'), .01, 5.)
        query[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            projector.valid_triangles_with_ties(query, torch.tensor([3], device='cuda'), .01, 5.)


if __name__ == '__main__':
    unittest.main()
