"""Acceptance tests for the C++ partition -> PaMO remeshing handoff.

Run with the project's Python environment. CPU checks do not import CUDA.
GPU remeshing checks are opt-in via CADMESH_TEST_GPU=1.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

CAD_MESH = Path(__file__).resolve().parents[1]
REPOSITORY = CAD_MESH.parent
sys.path.insert(0, str(CAD_MESH))
sys.path.insert(0, str(REPOSITORY))


def mesh_edges(faces):
    """Independent, index-based incidence (coincident vertices do not weld)."""
    result = defaultdict(list)
    for face_id, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            result[tuple(sorted((int(a), int(b))))].append(face_id)
    return result


def write_partition_fixture(directory, vertices, faces, labels, *, hard_seam=False):
    """Write small complete handoffs without depending on the production writer."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    incidence = mesh_edges(faces)
    edges, chains, hard_ids, smooth_ids = [], [], [], []
    patch_count = int(labels.max()) + 1
    patch_edges = [[] for _ in range(patch_count)]
    patch_chains = [[] for _ in range(patch_count)]
    neighbors = [set() for _ in range(patch_count)]
    adjacency_edges = defaultdict(list)
    for edge_id, (edge, face_ids) in enumerate(sorted(incidence.items())):
        owners = sorted(set(int(labels[i]) for i in face_ids))
        if len(face_ids) == 2 and len(owners) == 1:
            continue
        hard = len(face_ids) != 2 or hard_seam
        (hard_ids if hard else smooth_ids).append(edge_id)
        edges.append(dict(id=edge_id, vertex_ids=list(edge),
                          incident_triangle_ids=face_ids,
                          incident_patch_ids=owners, boundary_score=None,
                          open_boundary=len(face_ids) == 1,
                          non_manifold=len(face_ids) > 2,
                          constrained_feature=bool(hard_seam and len(owners) == 2),
                          hard_feature=hard))
        chain_id = len(chains)
        chains.append(dict(id=chain_id, edge_ids=[edge_id], vertex_ids=list(edge),
                           incident_patch_ids=owners,
                           initial_sample_vertex_ids=list(edge),
                           sampling_target='reference_mesh_polyline', closed=False,
                           non_manifold=False, inconsistent_winding=False,
                           hard_feature=hard,
                           boundary_kind=('hard_feature' if hard else
                                          'smooth_surface_transition')))
        for owner in owners:
            patch_edges[owner].append(edge_id)
            neighbors[owner].update(other for other in owners if other != owner)
            side = next(faces[i] for i in face_ids if labels[i] == owner)
            direction = 1 if edge in list(zip(side, np.roll(side, -1))) else -1
            patch_chains[owner].append(dict(chain_id=chain_id, direction=direction,
                                           orientation_status='consistent'))
        if len(owners) == 2:
            adjacency_edges[tuple(owners)].append(edge_id)
    patches = []
    for patch_id in range(patch_count):
        ids = np.flatnonzero(labels == patch_id).tolist()
        tri = vertices[faces[ids[0]]]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        normal /= np.linalg.norm(normal)
        patches.append(dict(id=patch_id, type='Plane', feature_role='Ordinary',
                            support_patch_ids=[], triangle_count=len(ids),
                            rms=0.0, max=0.0, normal_error=0.0, confidence=1.0,
                            triangle_ids=ids, neighbors=sorted(neighbors[patch_id]),
                            boundary_edge_ids=patch_edges[patch_id],
                            parameters=dict(origin=tri[0].tolist(), normal=normal.tolist()),
                            projection_target='analytic_surface',
                            consistent_face_orientation=True,
                            parameterization=dict(seam_assessment_required=False,
                                                  seam_generated=False),
                            boundary_chain_refs=patch_chains[patch_id]))
    edge_lengths = [np.linalg.norm(vertices[b] - vertices[a]) for a, b in incidence]
    boundary_vertices = sorted({v for edge in edges for v in edge['vertex_ids']})
    report = dict(schema='cadmesh.remesh_handoff', schema_version=1,
                  indexing=dict(base=0, vertices='clean_mesh_ply_vertex_order',
                                triangles='clean_mesh_ply_face_order',
                                edges='explicit_constraint_edge_ids'),
                  partition_valid=True,
                  mesh=dict(vertex_count=len(vertices), triangle_count=len(faces),
                            edge_count=len(incidence)),
                  resolution=dict(bbox_diagonal=float(np.linalg.norm(np.ptp(vertices, axis=0))),
                                  median_edge=float(np.median(edge_lengths)),
                                  fitting_tolerance=1e-6, angular_tolerance=0.00872664626),
                  cleanup=dict(input_triangles=len(faces), output_triangles=len(faces),
                               degenerate=0, duplicate=0, non_manifold_edges=0,
                               boundary_edges=sum(len(ids) == 1 for ids in incidence.values()),
                               warnings=[]),
                  patches=patches,
                  adjacency=[dict(patch0=a, patch1=b, edge_count=len(ids),
                                  confidence=None, shared_boundary_edge_ids=ids)
                             for (a, b), ids in adjacency_edges.items()],
                  constraints=dict(edge_ids=[edge['id'] for edge in edges],
                                   hard_feature_edge_ids=hard_ids,
                                   smooth_surface_transition_edge_ids=smooth_ids,
                                   junction_vertex_ids=[],
                                   corner_vertex_ids=boundary_vertices,
                                   corners=[dict(vertex_id=v, endpoint=True, junction=False,
                                                 sharp_corner=True, incidence_change=False)
                                            for v in boundary_vertices],
                                   boundary_chains=chains),
                  constraint_edges=edges)
    (directory / 'patch_report.json').write_text(json.dumps(report), encoding='utf-8')
    header = ('ply\nformat ascii 1.0\nelement vertex {}\nproperty double x\n'
              'property double y\nproperty double z\nelement face {}\n'
              'property list uchar int vertex_indices\nproperty int patch_id\n'
              'property int primitive_type\nproperty int feature_role\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    rows = [header.format(len(vertices), len(faces))]
    rows.extend(' '.join(format(float(v), '.17g') for v in vertex) + '\n' for vertex in vertices)
    rows.extend(f'3 {a} {b} {c} {int(label)} 1 0 80 120 160\n'
                for (a, b, c), label in zip(faces, labels))
    (directory / 'patch_result.ply').write_text(''.join(rows), encoding='ascii')
    return report


def two_plane_fixture(directory, *, hard_seam=False):
    vertices = [(0, 0, 0), (1, 0, 0), (2, 0, 0),
                (0, 1, 0), (1, 1, 0), (2, 1, 0)]
    faces = [(0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4)]
    return write_partition_fixture(directory, vertices, faces, [0, 0, 1, 1],
                                   hard_seam=hard_seam)


def annulus_fixture(directory):
    """A planar square annulus with an actual missing central square."""
    points, triangles, index = [], [], {}

    def vertex(x, y):
        if (x, y) not in index:
            index[x, y] = len(points)
            points.append((x, y, 0))
        return index[x, y]

    for y in range(4):
        for x in range(4):
            if 1 <= x < 3 and 1 <= y < 3:
                continue
            a, b, c, d = (vertex(x, y), vertex(x + 1, y),
                          vertex(x + 1, y + 1), vertex(x, y + 1))
            triangles.extend(((a, b, c), (a, c, d)))
    return write_partition_fixture(directory, points, triangles, [0] * len(triangles))


def boundary_component_count(faces):
    adjacency = defaultdict(set)
    for (a, b), ids in mesh_edges(faces).items():
        if len(ids) == 1:
            adjacency[a].add(b)
            adjacency[b].add(a)
    remaining = set(adjacency)
    count = 0
    while remaining:
        count += 1
        queue = [remaining.pop()]
        while queue:
            for vertex in adjacency[queue.pop()]:
                if vertex in remaining:
                    remaining.remove(vertex)
                    queue.append(vertex)
    return count


def triangle_area(vertices, faces):
    triangle = np.asarray(vertices)[np.asarray(faces)]
    return np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0],
                                   triangle[:, 2] - triangle[:, 0]), axis=1) * 0.5


def connected_face_count(faces):
    neighbors = [set() for _ in faces]
    for ids in mesh_edges(faces).values():
        for first in ids:
            neighbors[first].update(second for second in ids if second != first)
    remaining = set(range(len(faces)))
    count = 0
    while remaining:
        count += 1
        queue = [remaining.pop()]
        while queue:
            for face in neighbors[queue.pop()]:
                if face in remaining:
                    remaining.remove(face)
                    queue.append(face)
    return count


class FixtureTests(unittest.TestCase):
    """The acceptance fixtures themselves must have the advertised topology."""

    def test_independent_handoff_checker_accepts_fixtures(self):
        from cad_mesh.tests.verify_handoff import verify

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            two_plane_fixture(root / 'smooth')
            two_plane_fixture(root / 'hard', hard_seam=True)
            annulus_fixture(root / 'annulus')
            self.assertEqual(verify(root / 'smooth')['smooth_surface_transition_edges'], 1)
            self.assertEqual(verify(root / 'hard')['hard_feature_edges'], 7)
            self.assertEqual(verify(root / 'annulus')['triangles'], 24)


class PartitionLoaderTests(unittest.TestCase):
    def setUp(self):
        from cad_mesh.remesh_io import load_partition

        self.load_partition = load_partition
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.report = two_plane_fixture(self.directory)

    def write_report(self):
        (self.directory / 'patch_report.json').write_text(json.dumps(self.report),
                                                         encoding='utf-8')

    def edit_first_face(self, update):
        path = self.directory / 'patch_result.ply'
        lines = path.read_text(encoding='ascii').splitlines()
        index = lines.index('end_header') + 1 + self.report['mesh']['vertex_count']
        lines[index] = update(lines[index].split())
        path.write_text('\n'.join(lines) + '\n', encoding='ascii')

    def test_loader_keeps_global_indices_labels_and_distinct_boundary_kinds(self):
        data = self.load_partition(self.directory)
        self.assertEqual(data.vertices.shape, (6, 3))
        np.testing.assert_array_equal(data.faces, [(0, 1, 4), (0, 4, 3),
                                                  (1, 2, 5), (1, 5, 4)])
        np.testing.assert_array_equal(data.face_patch_ids, [0, 0, 1, 1])
        np.testing.assert_array_equal(data.smooth_edges, [(1, 4)])
        self.assertEqual(len(data.hard_edges), 6)
        self.assertEqual(len(data.constraint_edges), 7)
        self.assertNotIn((1, 4), set(map(tuple, data.hard_edges)))
        np.testing.assert_array_equal(data.corner_vertex_ids, np.arange(6))

    def test_explicit_hard_seam_remains_hard_even_for_coplanar_labels(self):
        two_plane_fixture(self.directory, hard_seam=True)
        data = self.load_partition(self.directory)
        self.assertEqual(len(data.smooth_edges), 0)
        self.assertIn((1, 4), set(map(tuple, data.hard_edges)))

    def test_load_via_ply_filename(self):
        data = self.load_partition(self.directory / 'patch_result.ply')
        self.assertEqual(len(data.faces), 4)

    def test_cpu_loader_does_not_import_torch(self):
        code = ('import sys; from cad_mesh.remesh_io import load_partition; '
                'load_partition(sys.argv[1]); '
                'assert "torch" not in sys.modules, "CPU loading imported torch"')
        completed = subprocess.run([sys.executable, '-c', code, str(self.directory)],
                                   cwd=REPOSITORY, capture_output=True, text=True,
                                   timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_hole_is_preserved_by_loading(self):
        annulus_fixture(self.directory)
        data = self.load_partition(self.directory)
        self.assertEqual(boundary_component_count(data.faces), 2)
        self.assertAlmostEqual(float(triangle_area(data.vertices, data.faces).sum()), 12.0)

    def test_reject_nontriangular_face(self):
        self.edit_first_face(lambda t: ' '.join(['4', *t[1:4], '3', *t[4:]]))
        with self.assertRaises(ValueError):
            self.load_partition(self.directory)

    def test_reject_ply_label_disagreement(self):
        self.edit_first_face(lambda t: ' '.join([*t[:4], '1', *t[5:]]))
        with self.assertRaises(ValueError):
            self.load_partition(self.directory)

    def test_reject_duplicate_or_missing_face_owner(self):
        self.report['patches'][1]['triangle_ids'][0] = 0
        self.write_report()
        with self.assertRaises(ValueError):
            self.load_partition(self.directory)

    def test_reject_missing_exported_constraint(self):
        self.report['constraint_edges'].pop()
        self.write_report()
        with self.assertRaises(ValueError):
            self.load_partition(self.directory)

    def test_reject_overlapping_hard_and_smooth_categories(self):
        self.report['constraints']['hard_feature_edge_ids'].extend(
            self.report['constraints']['smooth_surface_transition_edge_ids'])
        self.write_report()
        with self.assertRaises(ValueError):
            self.load_partition(self.directory)

    def test_reject_invalid_vertex_reference(self):
        self.edit_first_face(lambda t: ' '.join([t[0], '100', *t[2:]]))
        with self.assertRaises(ValueError):
            self.load_partition(self.directory)

    def test_reject_nonfinite_geometry(self):
        path = self.directory / 'patch_result.ply'
        text = path.read_text(encoding='ascii')
        path.write_text(text.replace('end_header\n0 0 0\n',
                                     'end_header\nnan 0 0\n', 1), encoding='ascii')
        with self.assertRaises(ValueError):
            self.load_partition(self.directory)

    def test_reject_stale_mesh_count(self):
        self.report['mesh']['triangle_count'] += 1
        self.write_report()
        with self.assertRaises(ValueError):
            self.load_partition(self.directory)

    def test_reject_unknown_schema_version(self):
        self.report['schema_version'] = 999
        self.write_report()
        with self.assertRaises(ValueError):
            self.load_partition(self.directory)

    def test_real_cpp_torus_handoff(self):
        fixture = CAD_MESH / 'debug' / 'torus_mothers_model_competition'
        if not fixture.is_dir():
            self.skipTest('optional exported C++ torus fixture is not present')
        data = self.load_partition(fixture)
        self.assertEqual(len(data.faces), 2112)
        self.assertIn('Torus', {patch['type'] for patch in data.report['patches']})
        self.assertGreater(len(data.hard_edges), 0)
        self.assertGreater(len(data.smooth_edges), 0)

    def test_real_cpp_unnamed_legacy_handoff(self):
        fixture = CAD_MESH / 'debug' / 'optimized_20260907'
        if not fixture.is_dir():
            self.skipTest('optional exported C++ Unnamed-Body fixture is not present')
        data = self.load_partition(fixture)
        self.assertEqual(data.vertices.shape, (136, 3))
        self.assertEqual(data.faces.shape, (268, 3))
        self.assertEqual(len(set(data.face_patch_ids)), 8)
        self.assertEqual(connected_face_count(data.faces), 1)
        self.assertEqual(boundary_component_count(data.faces), 0)
        self.assertEqual({patch['type'] for patch in data.report['patches']},
                         {'Plane', 'Cylinder'})


class RemeshWriterTests(unittest.TestCase):
    def setUp(self):
        from cad_mesh.remesh_io import load_partition, write_remesh_result

        self.write_remesh_result = write_remesh_result
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        two_plane_fixture(self.directory / 'source')
        self.source = source = load_partition(self.directory / 'source')
        self.result = SimpleNamespace(
            vertices=source.vertices.copy(), faces=source.faces.copy(),
            face_patch_ids=source.face_patch_ids.copy(), hard_edges=source.hard_edges.copy(),
            smooth_edges=source.smooth_edges.copy(), corner_vertex_ids=source.corner_vertex_ids.copy(),
            source_constraint_edge_ids=np.arange(len(source.constraint_edges)),
            stats={'backend': 'writer_fixture', 'metric': np.float64(0.0)})

    def test_writer_keeps_shared_indices_and_exports_fresh_membership(self):
        from cad_mesh.tests.verify_handoff import read_ply

        paths = self.write_remesh_result(self.directory / 'output', self.result, self.source)
        self.assertEqual(len(paths), 5)
        for path in paths.values():
            self.assertTrue(Path(path).is_file())
        for key in ('remesh_result_ply', 'surface_types_ply', 'feature_roles_ply'):
            exported = read_ply(Path(paths[key]))
            np.testing.assert_array_equal(np.asarray(exported['vertices']).reshape(-1, 3),
                                          self.source.vertices)
            np.testing.assert_array_equal(np.asarray(exported['faces']).reshape(-1, 3),
                                          self.source.faces)
            np.testing.assert_array_equal(exported['patch_ids'], self.source.face_patch_ids)
        report = json.loads(Path(paths['remesh_report_json']).read_text(encoding='utf-8'))
        self.assertEqual(report['schema'], 'cadmesh.partition_remesh')
        self.assertEqual(report['mesh']['vertex_count'], 6)
        self.assertEqual([patch['triangle_ids'] for patch in report['patches']], [[0, 1], [2, 3]])
        self.assertEqual(report['source_constraint_edge_indices'], list(range(7)))

    def test_writer_refuses_to_overwrite_partition_input(self):
        with self.assertRaises(ValueError):
            self.write_remesh_result(self.directory / 'source', self.result, self.source)

    def test_writer_rejects_nonfinite_quality_report_before_creating_output(self):
        self.result.stats['invalid'] = float('nan')
        with self.assertRaises(ValueError):
            self.write_remesh_result(self.directory / 'output', self.result, self.source)
        self.assertFalse((self.directory / 'output').exists())

    def test_writer_rejects_invalid_boundary_lineage(self):
        self.result.source_constraint_edge_ids[-1] = len(self.source.constraint_edges)
        with self.assertRaises(ValueError):
            self.write_remesh_result(self.directory / 'output', self.result, self.source)


@unittest.skipUnless(os.environ.get('CADMESH_TEST_GPU') == '1',
                     'set CADMESH_TEST_GPU=1 to run the CUDA acceptance checks')
class RemeshGpuTests(unittest.TestCase):
    def setUp(self):
        from cad_mesh.remesh_io import load_partition, write_remesh_result
        from cad_mesh.remesh_pipeline import remesh_partition

        self.load_partition = load_partition
        from functools import partial
        self.remesh_partition = partial(remesh_partition, method="legacy")
        self.write_remesh_result = write_remesh_result
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def assert_valid_result(self, source, result, *, target, area, boundary_loops):
        vertices = np.asarray(result.vertices)
        faces = np.asarray(result.faces)
        labels = np.asarray(result.face_patch_ids)
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)
        self.assertEqual(labels.shape, (len(faces),))
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue(np.issubdtype(faces.dtype, np.integer))
        self.assertGreaterEqual(int(faces.min()), 0)
        self.assertLess(int(faces.max()), len(vertices))
        self.assertEqual(set(labels), set(source.face_patch_ids))
        self.assertEqual(len(set(map(tuple, vertices))), len(vertices),
                         'shared partition boundaries must use one global vertex table')
        edges = mesh_edges(faces)
        self.assertTrue(all(len(ids) <= 2 for ids in edges.values()))
        self.assertEqual(connected_face_count(faces), 1,
                         'partition borders must not become disconnected sheets')
        self.assertEqual(boundary_component_count(faces), boundary_loops)
        areas = triangle_area(vertices, faces)
        self.assertTrue((areas > 1e-12).all())
        self.assertAlmostEqual(float(areas.sum()), area, places=8)
        lengths = np.linalg.norm(vertices[np.array(list(edges))[:, 0]] -
                                 vertices[np.array(list(edges))[:, 1]], axis=1)
        self.assertLessEqual(float(lengths.max()), target * 1.05)
        for vertex in source.vertices[source.corner_vertex_ids]:
            self.assertLess(float(np.linalg.norm(vertices - vertex, axis=1).min()), 1e-10)
        constraints = set(map(tuple, np.asarray(result.hard_edges).reshape(-1, 2)))
        constraints.update(map(tuple, np.asarray(result.smooth_edges).reshape(-1, 2)))
        self.assertTrue(constraints <= set(edges))
        self.assertEqual(len(result.source_constraint_edge_ids), len(constraints))

    def test_shared_smooth_partition_boundary_and_writer(self):
        from cad_mesh.tests.verify_handoff import read_ply

        two_plane_fixture(self.directory)
        source = self.load_partition(self.directory)
        result = self.remesh_partition(source, target_edge_length=0.35, sample_count=400,
                                       collapse_passes=3, flip_passes=3, relax_iterations=2)
        self.assert_valid_result(source, result, target=0.35, area=2.0, boundary_loops=1)
        self.assertGreater(len(result.faces), len(source.faces))
        np.testing.assert_allclose(result.vertices[:, 2], 0, atol=1e-10)
        edges = mesh_edges(result.faces)
        cross_label = [edge for edge, ids in edges.items()
                       if len({int(result.face_patch_ids[i]) for i in ids}) == 2]
        self.assertGreater(len(cross_label), 1)
        self.assertEqual(set(cross_label), set(map(tuple, result.smooth_edges)))
        for edge in cross_label:
            self.assertEqual(len(edges[edge]), 2)
            np.testing.assert_allclose(result.vertices[list(edge), 0], 1, atol=1e-10)
        centroid_x = result.vertices[result.faces].mean(axis=1)[:, 0]
        self.assertTrue((centroid_x[result.face_patch_ids == 0] < 1).all())
        self.assertTrue((centroid_x[result.face_patch_ids == 1] > 1).all())
        paths = self.write_remesh_result(self.directory / 'output', result, source)
        for name in ('remesh_result_ply', 'surface_types_ply', 'feature_roles_ply',
                     'remesh_result_stl', 'remesh_report_json'):
            self.assertTrue(Path(paths[name]).is_file(), name)
        exported = read_ply(Path(paths['remesh_result_ply']))
        np.testing.assert_array_equal(np.asarray(exported['faces']).reshape(-1, 3), result.faces)
        np.testing.assert_array_equal(exported['patch_ids'], result.face_patch_ids)
        np.testing.assert_allclose(np.asarray(exported['vertices']).reshape(-1, 3),
                                   result.vertices, atol=0, rtol=1e-15)

    def test_hole_and_boundary_loops_survive_remeshing(self):
        annulus_fixture(self.directory)
        source = self.load_partition(self.directory)
        result = self.remesh_partition(source, target_edge_length=0.45, sample_count=600,
                                       collapse_passes=3, flip_passes=3, relax_iterations=2)
        self.assert_valid_result(source, result, target=0.45, area=12.0, boundary_loops=2)
        centroids = result.vertices[result.faces].mean(axis=1)
        self.assertFalse(((centroids[:, 0] > 1) & (centroids[:, 0] < 3) &
                          (centroids[:, 1] > 1) & (centroids[:, 1] < 3)).any(),
                         'remeshing filled the planar hole')

    def test_computational_batches_preserve_only_original_patch_boundaries(self):
        two_plane_fixture(self.directory)
        source = self.load_partition(self.directory)
        result = self.remesh_partition(
            source, target_edge_length=0.35, sample_count=300,
            collapse_passes=2, flip_passes=2, relax_iterations=1,
            batch_face_limit=1)
        self.assert_valid_result(source, result, target=0.35, area=2.0, boundary_loops=1)
        self.assertEqual(result.stats['batch_count'], 4)
        self.assertEqual(set(result.face_patch_ids), {0, 1})
        self.assertGreater(result.stats['computational_interfaces'], 0)
        edges = mesh_edges(result.faces)
        cross = {edge for edge, ids in edges.items()
                 if len({int(result.face_patch_ids[i]) for i in ids}) == 2}
        self.assertEqual(cross, set(map(tuple, result.smooth_edges)))
        self.assertEqual(len(result.hard_edges), sum(len(ids) == 1 for ids in edges.values()))
        self.write_remesh_result(self.directory / 'batched_output', result, source)

    def test_dense_batch_without_eligible_sampling_faces_still_remeshes(self):
        two_plane_fixture(self.directory)
        source = self.load_partition(self.directory)
        result = self.remesh_partition(source, target_edge_length=8, sample_count=50,
                                       collapse_passes=2, flip_passes=2, relax_iterations=1)
        self.assert_valid_result(source, result, target=8, area=2.0, boundary_loops=1)
        self.assertEqual(result.stats['sample_count'], 0)


if __name__ == '__main__':
    unittest.main()
