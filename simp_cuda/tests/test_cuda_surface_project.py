"""CUDA nearest-surface queries versus CPU and geometric protection fixtures."""
import importlib
import importlib.util
import unittest
import numpy as np
import torch

from simp_cuda.tests.test_surface_sample import surface_sample
from simp_cuda.tests.test_whole_patch_surface import grid_fixture

AVAILABLE = torch.cuda.is_available() and importlib.util.find_spec('warp') is not None
cuda_module = None
if AVAILABLE and surface_sample is not None:
    cuda_module = importlib.import_module(surface_sample.__package__ + '.cuda_surface_project')


@unittest.skipUnless(cuda_module is not None, 'CUDA and Warp required')
class CudaSurfaceProjectionTests(unittest.TestCase):
    def projector(self, vertices, faces, labels):
        return cuda_module.CudaReferencePatchProjector(vertices, faces, labels)

    def test_device_cache_reuses_rejections_and_invalidates_coordinates(self):
        vertices = np.array([[0.,0,0],[1.,0,0],[0.,1,0]])
        gpu = self.projector(vertices, np.array([[0,1,2]]), np.array([2**40]))
        points = torch.tensor(vertices, device='cuda', dtype=torch.float64)
        faces = torch.tensor([[0,1,2],[0,2,1],[0,1,2]], device='cuda')
        labels = torch.full((3,), 2**40, device='cuda', dtype=torch.long)
        cache = gpu.make_triangle_cache(points, .01, 5.)
        # Force all keys into one slot to exercise collision ownership.
        cache.capacity = 1
        expected = gpu.valid_triangles(points[faces], labels, .01, 5.)
        for _ in range(2):
            self.assertTrue(torch.equal(cache.check(faces, labels), expected))
        points[:,2] += 1.
        self.assertFalse(bool(cache.check(faces, labels).any()))

    def test_exact_patch_nearest_queries_across_scales(self):
        rng = np.random.default_rng(520)
        triangles = rng.normal(size=(67,3,3))
        faces = np.arange(67*3).reshape(-1,3)
        labels = np.where(np.arange(67)%3 == 0, 2**40, 7)
        points = rng.normal(size=(83,3))*2
        query_labels = np.where(np.arange(83)%2 == 0, 2**40, 7)
        for scale in (1e-3,1.,1e3):
            with self.subTest(scale=scale):
                vertices = (triangles.reshape(-1,3) + (3000,20,-15))*scale
                queries = (points + (3000,20,-15))*scale
                cpu = surface_sample._ReferencePatchProjector(vertices,faces,labels)
                gpu = self.projector(vertices,faces,labels)
                expected = cpu.query_numpy(queries,query_labels)
                actual = gpu.query_numpy(queries,query_labels)
                np.testing.assert_allclose(actual[0],expected[0],rtol=0,atol=scale*1e-8)
                np.testing.assert_allclose(actual[1],expected[1],rtol=1e-8,atol=scale*scale*1e-10)
                np.testing.assert_array_equal(labels[actual[2]],query_labels)

    def test_complete_patch_query_never_selects_a_closer_other_sheet(self):
        vertices = np.array([[0.,0,0],[1.,0,0],[1.,1,0],[0.,1,0]])
        reference = np.vstack((vertices,vertices+(0,0,.01)))
        faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
        gpu = self.projector(reference,faces,np.array([7,7,9,9]))
        points = torch.tensor([[.2,.8,.009],[.2,.8,.009]],device='cuda',dtype=torch.float64)
        position, ids = gpu.project(points,torch.tensor([7,9],device='cuda'))
        np.testing.assert_allclose(position.cpu(),[[.2,.8,0],[.2,.8,.01]],atol=1e-14)
        np.testing.assert_array_equal(ids.cpu(),[1,3])

    def test_hole_bridge_and_reversed_triangles_are_rejected(self):
        vertices,faces,labels,_ = grid_fixture(hole=True)
        gpu = self.projector(vertices,faces,labels)
        triangles = torch.tensor([[[.25,.25,0],[.75,.25,0],[.5,.75,0]],
                                  vertices[faces[0,::-1]].tolist()],device='cuda',dtype=torch.float64)
        self.assertEqual(gpu.valid_triangles(triangles,torch.tensor([0,0],device='cuda'),1e-6,5).tolist(),[False,False])

    def test_thin_triangle_support_and_distinct_nearby_sheet(self):
        fixtures = [
            (np.array([[3063.46240234375,-24.833763122558594,-10.123769760131836],
                       [3063.46240234375,-29.833763122558594,-10.123785018920898],
                       [3063.46240234375,-29.833763122558594,-10.12376880645752],
                       [3063.069091796875,-24.833763122558594,-9.305299758911133]]),np.array([[3,0,2],[0,1,2]])),
            (np.array([[3185.849609375,94.21623992919922,-13.852574348449707],
                       [3185.849609375,94.21623992919922,-13.852519989013672],
                       [3184.040283203125,94.21623992919922,13.955145835876465],
                       [3187.689697265625,94.21623992919922,-18.787179946899414]]),np.array([[2,3,0],[0,1,2]])),
        ]
        for vertices,faces in fixtures:
            gpu = self.projector(vertices,faces,np.array([7,7]))
            triangle = torch.tensor(vertices[faces[1:]],device='cuda')
            valid,details = gpu.valid_triangles(triangle,torch.tensor([7],device='cuda'),1e-5,5,True)
            self.assertTrue(bool(valid[0]))
            self.assertLess(float(details['maximum_sample_distance'][0]),1e-5)
        vertices = np.array([[0.,0,0],[1.,0,0],[0.,1,0]])
        gpu = self.projector(np.vstack((vertices,vertices+(0,0,1e-7))),
                             np.array([[0,1,2],[3,5,4]]),np.array([0,0]))
        triangle = torch.tensor(vertices[np.array([[0,2,1]])],device='cuda')
        self.assertFalse(bool(gpu.valid_triangles(triangle,torch.tensor([0],device='cuda'),1e-6,5)[0]))

    def test_single_leaf_empty_unknown_labels_and_noncontiguous_queries(self):
        vertices = np.array([[0.,0,0],[1.,0,0],[0.,1,0]])
        gpu = self.projector(vertices,np.array([[0,1,2]]),np.array([2**40]))
        empty = torch.empty((0,3),device='cuda',dtype=torch.float64)
        self.assertEqual(gpu.project(empty,torch.empty(0,device='cuda',dtype=torch.long))[0].shape,(0,3))
        points = torch.tensor([[.1,.2],[.2,.1],[.3,.3]],device='cuda',dtype=torch.float64).T
        output,_ = gpu.project(points,torch.full((2,),2**40,device='cuda'))
        np.testing.assert_allclose(output.cpu(),[[.1,.2,0],[.2,.1,0]],atol=1e-14)
        with self.assertRaisesRegex(ValueError,'unknown'):
            gpu.project(points,torch.tensor([7,7],device='cuda'))
        points = points.clone()
        points[0,0] = float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            gpu.project(points,torch.full((2,),2**40,device='cuda'))

    def test_current_torch_stream_owns_reference_build_and_queries(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            vertices,faces,labels,_ = grid_fixture(curved=True)
            gpu = self.projector(vertices,faces,labels)
            positions = torch.tensor(vertices[faces].mean(axis=1),device='cuda')
            projected,ids = gpu.project(positions,torch.tensor(labels,device='cuda'))
        stream.synchronize()
        np.testing.assert_allclose(projected.cpu(),positions.cpu(),atol=1e-12)
        self.assertTrue(bool((ids>=0).all()))


if __name__ == '__main__':
    unittest.main()
