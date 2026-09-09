"""End-to-end whole-surface paths, separate from legacy contract tests."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

from cad_mesh.remesh_io import load_partition, write_remesh_result
from cad_mesh.remesh_pipeline import remesh_partition
from cad_mesh.tests.test_remesh_pipeline import (
    two_plane_fixture, annulus_fixture, triangle_area, boundary_component_count, write_partition_fixture,
)


@unittest.skipUnless(os.environ.get('CADMESH_TEST_GPU') == '1', 'set CADMESH_TEST_GPU=1')
class SurfaceRebuildTest(unittest.TestCase):
    def test_geometry_tolerance_does_not_follow_requested_edge_length(self):
        two_plane_fixture(self.directory)
        source = load_partition(self.directory)
        options = dict(sample_count=10, collapse_passes=0, flip_passes=0, relax_iterations=0)
        coarse = remesh_partition(source, target_edge_length=.5, **options)
        fine = remesh_partition(source, target_edge_length=.3, **options)
        self.assertEqual(coarse.stats['requested_maximum_deviation'],
                         fine.stats['requested_maximum_deviation'])
        explicit = remesh_partition(source, target_edge_length=.3, maximum_deviation=1e-6, **options)
        self.assertEqual(explicit.stats['requested_maximum_deviation'], 1e-6)
        self.assertEqual(explicit.stats['deviation_policy'], 'explicit')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_coplanar_labels_become_one_whole_chart_with_source_membership(self):
        two_plane_fixture(self.directory)
        source = load_partition(self.directory)
        result = remesh_partition(source, target_edge_length=.35, sample_count=100,
                                  collapse_passes=2, flip_passes=2, relax_iterations=1)
        self.assertEqual(result.stats['method'], 'surface')
        self.assertEqual(result.stats['output_patch_count'], 1)
        self.assertEqual(result.stats['analytic_patch_count'], 1)
        self.assertEqual(len(result.smooth_edges), 0)
        self.assertEqual(boundary_component_count(result.faces), 1)
        self.assertAlmostEqual(float(triangle_area(result.vertices, result.faces).sum()), 2., places=9)
        output = write_remesh_result(self.directory / 'out', result, source)
        report = json.loads(Path(output['remesh_report_json']).read_text())
        self.assertEqual(report['patches'][0]['source_patch_ids'], [0, 1])
        self.assertEqual(report['patches'][0]['projection_target'], 'analytic_surface')
        self.assertTrue(report['stats']['validation']['passed'])

    def test_planar_hole_remains_a_hole_in_the_output_chart(self):
        annulus_fixture(self.directory)
        source = load_partition(self.directory)
        result = remesh_partition(source, target_edge_length=.45, sample_count=100,
                                  collapse_passes=2, flip_passes=2, relax_iterations=1)
        self.assertEqual(result.stats['analytic_patch_count'], 1)
        self.assertEqual(boundary_component_count(result.faces), 2)
        self.assertAlmostEqual(float(triangle_area(result.vertices, result.faces).sum()), 12., places=8)
        centroids = result.vertices[result.faces].mean(axis=1)
        self.assertFalse(((centroids[:,0] > 1) & (centroids[:,0] < 3) &
                          (centroids[:,1] > 1) & (centroids[:,1] < 3)).any())
        write_remesh_result(self.directory / 'out', result, source)

    def test_gpu_patch_preserves_vertex_contact_with_an_accepted_chart(self):
        vertices = np.array([[.32,.44,0.],[.32,.44,1.],[.32,1.44,1.],
                             [0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
        faces = np.array([[0,1,2],[3,4,0],[4,5,0],[5,6,0],[6,3,0]])
        write_partition_fixture(self.directory, vertices, faces, [0,1,1,1,1])
        source = load_partition(self.directory)
        source.report['patches'][1].update(type='Freeform', parameters=None, projection_target='reference_mesh')
        # The contact is protected by global geometry, without an explicit corner hint.
        source.corner_vertex_ids = np.empty(0, dtype=np.int64)
        result = remesh_partition(source, target_edge_length=.4, sample_count=20,
                                  collapse_passes=2, flip_passes=2, relax_iterations=3)
        self.assertEqual(result.stats['analytic_patch_count'], 1)
        np.testing.assert_array_equal(result.vertices[0], vertices[0])
        incident = result.face_patch_ids[np.any(result.faces == 0, axis=1)]
        self.assertEqual(set(incident), {0,1})


if __name__ == '__main__':
    unittest.main()
