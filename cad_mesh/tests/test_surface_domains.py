"""Semantic seams may disappear; physical creases and holes must survive."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from cad_mesh.boundary_domains import prepare_surface_domains
from cad_mesh.remesh_io import load_partition, write_remesh_result
from cad_mesh.remesh_pipeline import RemeshResult
import json
from cad_mesh.tests.test_remesh_pipeline import write_partition_fixture, two_plane_fixture


class SurfaceDomainsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_coplanar_label_interface_is_released_with_provenance(self):
        two_plane_fixture(self.directory)
        source = load_partition(self.directory)
        domain, stats = prepare_surface_domains(source)
        self.assertEqual(len(domain.report['patches']), 1)
        self.assertEqual(domain.report['patches'][0]['source_patch_ids'], [0, 1])
        self.assertEqual(len(domain.smooth_edges), 0)
        self.assertGreater(stats['released_label_edge_count'], 0)
        np.testing.assert_array_equal(domain.hard_edges, source.hard_edges)
        np.testing.assert_array_equal(domain.vertices, source.vertices)
        np.testing.assert_array_equal(domain.faces, source.faces)
        self.assertEqual(len(source.report['patches']), 2)
        self.assertEqual(set(source.face_patch_ids), {0, 1})

    def test_explicit_crease_between_coplanar_faces_is_not_released(self):
        vertices = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
        faces = np.array([[0,1,2],[0,2,3]])
        write_partition_fixture(self.directory, vertices, faces, [0,1], hard_seam=True)
        source = load_partition(self.directory)
        domain, stats = prepare_surface_domains(source)
        self.assertEqual(len(domain.report['patches']), 2)
        self.assertEqual(stats['released_label_edge_count'], 0)
        np.testing.assert_array_equal(domain.hard_edges, source.hard_edges)

    def test_same_type_different_surface_is_not_a_removable_label(self):
        vertices = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,.1]])
        faces = np.array([[0,1,2],[0,2,3]])
        write_partition_fixture(self.directory, vertices, faces, [0,1])
        source = load_partition(self.directory)
        domain, stats = prepare_surface_domains(source)
        self.assertEqual(len(domain.report['patches']), 2)
        self.assertEqual(stats['released_label_edge_count'], 0)

    def test_unknown_freeform_labels_are_not_assumed_compatible(self):
        two_plane_fixture(self.directory)
        source = load_partition(self.directory)
        for patch in source.report['patches']:
            patch['type'] = 'Freeform'
            patch['parameters'] = None
        domain, stats = prepare_surface_domains(source)
        self.assertEqual(len(domain.report['patches']), 2)
        self.assertEqual(stats['released_label_edge_count'], 0)

    def test_writer_records_removed_labels_and_retained_geometric_interfaces(self):
        two_plane_fixture(self.directory)
        source = load_partition(self.directory)
        domain, policy = prepare_surface_domains(source)
        result = RemeshResult(
            domain.vertices, domain.faces, domain.face_patch_ids, domain.hard_edges,
            domain.smooth_edges, domain.corner_vertex_ids, {"boundary_policy": policy},
            np.arange(len(domain.constraint_edges)), prepared_source=domain,
        )
        output = write_remesh_result(self.directory / 'rebuilt', result, source)
        report = json.loads(Path(output['remesh_report_json']).read_text())
        self.assertEqual(report['source_original_patch_count'], 2)
        self.assertEqual(len(report['patches']), 1)
        self.assertEqual(report['patches'][0]['source_patch_ids'], [0, 1])
        self.assertGreater(report['boundary_policy']['released_label_edge_count'], 0)
        self.assertTrue(all(e['required_geometric_constraint'] for e in report['constraint_edges']))


if __name__ == '__main__':
    unittest.main()
