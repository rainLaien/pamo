"""Report-free remeshing preserves geometry and mandatory final validation."""
import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from simp_cuda.tests.test_whole_patch_surface import surface_sample, grid_fixture
from simp_cuda.tests import test_whole_patch_surface as whole_fixture


@unittest.skipIf(surface_sample is None or not torch.cuda.is_available(), 'CUDA/dependencies unavailable')
class OptionalRemeshDiagnosticsTests(unittest.TestCase):
    def remesh(self, vertices, faces, labels, constraints, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return whole_fixture.WholePatchCudaTests.remesh(self, vertices, faces, labels, constraints, **kwargs)

    def test_geometry_ownership_and_constraints_match_without_reports(self):
        vertices, faces, labels, constraints = grid_fixture(two_patches=True)
        projector = surface_sample._make_reference_projector(vertices, faces, labels, device='cuda')
        options = dict(whole_patch_reference=projector, collapse_passes=3, flip_passes=3,
                       relax_iterations=2, sample_count=4, minimum_source_area_ratio=0.)
        expected = self.remesh(vertices, faces, labels, constraints, **options)
        with patch.object(surface_sample, '_make_reference_projector', side_effect=AssertionError('unused local BVH built')), \
             patch.object(surface_sample, 'mesh_quality_metrics', side_effect=AssertionError('report metrics computed')):
            actual = self.remesh(vertices, faces, labels, constraints, collect_diagnostics=False, **options)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2]['face_patch_ids'], expected[2]['face_patch_ids'])
        whole_fixture.WholePatchCudaTests.assert_partition_geometry(self, vertices, faces, labels, constraints, actual)
        for key in ('splits', 'collapses', 'flips', 'remaining_long_edges', 'fixed_constraint_edge_count', 'fixed_corner_count'):
            self.assertEqual(actual[2][key], expected[2][key])
        for key in ('initial_metrics', 'sampled_metrics', 'final_metrics'):
            self.assertEqual(actual[2][key], {})
            self.assertTrue(expected[2][key])
        self.assertEqual(actual[2]['source_face_id_semantics'], 'operation_lineage_on_same_patch')
        self.assertNotIn('original_interior_vertices_moved', actual[2])
        self.assertNotIn('original_vertices_unused', actual[2])
        self.assertNotIn('reference_normal_distance_ties_resolved', actual[2])

    def test_local_reference_remains_when_required_for_geometry(self):
        vertices, faces, labels, constraints = grid_fixture()
        original = surface_sample._make_reference_projector
        with patch.object(surface_sample, '_make_reference_projector', wraps=original) as build:
            result = self.remesh(vertices, faces, labels, constraints, collect_diagnostics=False, relax_iterations=0)
        self.assertEqual(build.call_count, 1)
        whole_fixture.WholePatchCudaTests.assert_partition_geometry(self, vertices, faces, labels, constraints, result)

    def test_global_surface_and_complete_topology_checks_cannot_be_disabled(self):
        vertices, faces, labels, constraints = grid_fixture()
        projector = surface_sample._make_reference_projector(vertices, faces, labels, device='cuda')
        original = surface_sample._gpu_relax_patch_vertices
        for corruption in ('surface', 'topology'):
            for diagnostics in (True, False):
                def corrupted(*args, **kwargs):
                    output, count = original(*args, **kwargs)
                    if corruption == 'surface':
                        output = output.clone()
                        output[6, 2] += .2
                    else:
                        args[1][-1] = args[1][0]
                    return output, count
                with self.subTest(corruption=corruption, diagnostics=diagnostics), \
                     patch.object(surface_sample, '_gpu_relax_patch_vertices', side_effect=corrupted):
                    error = RuntimeError if corruption == 'surface' else ValueError
                    message = 'violates sampled reference' if corruption == 'surface' else 'constraint|Fixed constraints'
                    with self.assertRaisesRegex(error, message):
                        self.remesh(vertices, faces, labels, constraints, collect_diagnostics=diagnostics,
                                    whole_patch_reference=projector, relax_iterations=0)


if __name__ == '__main__':
    unittest.main()
