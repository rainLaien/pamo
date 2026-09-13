"""Binary handoffs retain ASCII geometry and all independent input validation."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cad_mesh.remesh_io import _read_ply, load_partition
from cad_mesh.tests.test_remesh_pipeline import two_plane_fixture


def binary_fixture(path, vertices, faces, labels, *, role_first=False, roles=True):
    properties = [('patch_id', 'int', 'i'), ('primitive_type', 'int', 'i')]
    colors = [('red', 'uchar', 'B'), ('green', 'uchar', 'B'), ('blue', 'uchar', 'B')]
    role = [('feature_role', 'int', 'i')] if roles else []
    properties += role + colors if role_first else colors + role
    header = (f'ply\nformat binary_little_endian 1.0\ncomment standard binary fixture\n'
              f'element vertex {len(vertices)}\nproperty double x\nproperty double y\nproperty double z\n'
              f'element face {len(faces)}\nproperty list uchar int vertex_indices\n'
              + ''.join(f'property {kind} {name}\n' for name, kind, _ in properties) + 'end_header\n')
    payload = bytearray(header.encode('ascii'))
    payload.extend(np.asarray(vertices, dtype='<f8').tobytes())
    face_struct = struct.Struct('<B3i' + ''.join(fmt for _, _, fmt in properties))
    for face, label in zip(faces, labels):
        values = {'patch_id': int(label), 'primitive_type': 1, 'feature_role': 0,
                  'red': 80, 'green': 120, 'blue': 160}
        payload.extend(face_struct.pack(3, *face, *(values[name] for name, _, _ in properties)))
    path.write_bytes(payload)
    return len(header), len(header) + len(vertices) * 24, face_struct.size


class BinaryPartitionReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report = two_plane_fixture(self.root)
        self.path = self.root / 'patch_result.ply'
        self.ascii = self.path.read_bytes()
        self.reference = load_partition(self.root)
        self.header_size, self.face_offset, self.face_size = binary_fixture(
            self.path, self.reference.vertices, self.reference.faces, self.reference.face_patch_ids)
        self.binary = self.path.read_bytes()

    def write_report(self, report):
        (self.root / 'patch_report.json').write_text(json.dumps(report), encoding='utf-8')

    def assert_same_geometry(self, actual):
        for name in ('vertices', 'faces', 'face_patch_ids', 'hard_edges', 'smooth_edges', 'corner_vertex_ids'):
            np.testing.assert_array_equal(getattr(actual, name), getattr(self.reference, name))

    def test_binary_matches_ascii_arrays_and_constraints(self):
        self.assertEqual(self.face_size, 28)
        self.assert_same_geometry(load_partition(self.root))
        self.path.write_bytes(self.ascii)
        self.assert_same_geometry(load_partition(self.root))

    def test_property_order_and_index_alias_preserve_layout(self):
        binary_fixture(self.path, self.reference.vertices, self.reference.faces,
                       self.reference.face_patch_ids, role_first=True)
        self.path.write_bytes(self.path.read_bytes().replace(b'vertex_indices', b'vertex_index'))
        self.assert_same_geometry(load_partition(self.root))

    def test_minimal_json_still_receives_full_loader_validation(self):
        minimal = {key: copy.deepcopy(self.report[key]) for key in (
            'schema', 'schema_version', 'indexing', 'partition_valid', 'mesh',
            'resolution', 'patches', 'constraints', 'constraint_edges')}
        minimal['patches'] = [{key: patch[key] for key in (
            'id', 'type', 'feature_role', 'support_patch_ids', 'triangle_count',
            'triangle_ids', 'parameters', 'projection_target')} for patch in minimal['patches']]
        minimal['constraints'] = {key: minimal['constraints'][key] for key in (
            'edge_ids', 'hard_feature_edge_ids', 'smooth_surface_transition_edge_ids', 'corner_vertex_ids')}
        self.write_report(minimal)
        self.assert_same_geometry(load_partition(self.root))
        minimal['patches'][0]['triangle_ids'][0] = 2
        self.write_report(minimal)
        with self.assertRaisesRegex(ValueError, 'membership|labels'):
            load_partition(self.root)

    def test_optional_role_property_requires_matching_legacy_json(self):
        binary_fixture(self.path, self.reference.vertices, self.reference.faces,
                       self.reference.face_patch_ids, roles=False)
        with self.assertRaisesRegex(ValueError, 'feature_role'):
            load_partition(self.root)
        for patch in self.report['patches']:
            patch.pop('feature_role')
        self.write_report(self.report)
        self.assert_same_geometry(load_partition(self.root))

    def test_truncated_and_trailing_payloads_are_rejected_before_array_read(self):
        for payload in (self.binary[:-1], self.binary[:self.face_offset - 1], self.binary + b'\n',
                        self.binary + b'\x00' * self.face_size):
            with self.subTest(size=len(payload)), patch('cad_mesh.partition_ply.np.fromfile') as read:
                self.path.write_bytes(payload)
                with self.assertRaisesRegex(ValueError, 'payload'):
                    _read_ply(self.path)
                read.assert_not_called()

    def test_malicious_counts_do_not_allocate_from_header(self):
        for count in (b'1000000000', b'9' * 80, b'-1', b'1.5'):
            with self.subTest(count=count), patch('cad_mesh.partition_ply.np.fromfile') as read:
                self.path.write_bytes(self.binary.replace(b'element vertex 6', b'element vertex ' + count))
                with self.assertRaises(ValueError):
                    _read_ply(self.path)
                read.assert_not_called()

    def test_nontriangle_list_count_is_rejected(self):
        for count in (0, 2, 4, 255):
            with self.subTest(count=count):
                raw = bytearray(self.binary)
                raw[self.face_offset] = count
                self.path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'triangle faces'):
                    load_partition(self.root)

    def test_invalid_indices_and_nonfinite_coordinates_are_rejected(self):
        for value in (-1, 6, 2**31 - 1):
            with self.subTest(index=value):
                raw = bytearray(self.binary)
                struct.pack_into('<i', raw, self.face_offset + 1, value)
                self.path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'out of range'):
                    load_partition(self.root)
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(coordinate=value):
                raw = bytearray(self.binary)
                struct.pack_into('<d', raw, self.header_size, value)
                self.path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'NaN or infinity'):
                    load_partition(self.root)

    def test_binary_input_does_not_bypass_membership_or_constraint_checks(self):
        for corruption in ('counts', 'membership', 'incidence', 'hard_flag', 'missing_edge'):
            with self.subTest(corruption=corruption):
                report = copy.deepcopy(self.report)
                if corruption == 'counts':
                    report['mesh']['triangle_count'] += 1
                elif corruption == 'membership':
                    report['patches'][1]['triangle_ids'][0] = 0
                elif corruption == 'incidence':
                    report['constraint_edges'][0]['incident_triangle_ids'] = [3]
                elif corruption == 'hard_flag':
                    report['constraint_edges'][0]['hard_feature'] = False
                else:
                    report['constraint_edges'].pop()
                self.write_report(report)
                with self.assertRaises(ValueError):
                    load_partition(self.root)

    def test_invalid_binary_headers_are_rejected(self):
        changes = (
            (b'binary_little_endian', b'binary_big_endian'),
            (b'property double x', b'property bogus x'),
            (b'property double y', b'property double x'),
            (b'property int patch_id', b'property float patch_id'),
            (b'property list uchar int vertex_indices', b'property list float int vertex_indices'),
            (b'property list uchar int vertex_indices', b'property list uchar double vertex_indices'),
            (b'property double z', b'property list uchar double z'),
            (b'property int feature_role', b'property list uchar int extra'),
            (b'property int patch_id', b'property int other'),
            (b'element face 4', b'element edge 4'),
            (b'end_header\n', b'format binary_little_endian 1.0\nend_header\n'),
            (b'end_header\n', b'unknown declaration\nend_header\n'),
        )
        for before, after in changes:
            with self.subTest(change=after):
                self.path.write_bytes(self.binary.replace(before, after))
                with self.assertRaises(ValueError):
                    _read_ply(self.path)

    def test_unterminated_or_oversized_header_is_rejected(self):
        for payload in (b'ply\nformat binary_little_endian 1.0\n',
                        b'ply\ncomment ' + b'x' * 65537 + b'\n'):
            self.path.write_bytes(payload)
            with self.assertRaises(ValueError):
                _read_ply(self.path)


if __name__ == '__main__':
    unittest.main()
