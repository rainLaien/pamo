"""Whole-patch remeshing regressions independent of CAD patch fitting."""
from collections import defaultdict
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simp_cuda.tests.test_surface_sample import surface_sample


def edge_incidence(faces):
    result = defaultdict(list)
    for index, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            result[tuple(sorted((int(a), int(b))))].append(index)
    return result


def grid_fixture(size=5, *, two_patches=False, hole=False, curved=False):
    axis = np.linspace(0, 1, size)
    vertices = np.array([[x, y, 0.] for y in axis for x in axis])
    faces, labels = [], []
    for y in range(size - 1):
        for x in range(size - 1):
            if hole and 1 <= x < size - 2 and 1 <= y < size - 2:
                continue
            a = y * size + x
            faces.extend(((a, a + 1, a + 1 + size), (a, a + 1 + size, a + size)))
            labels.extend([int(two_patches and x >= (size - 1) // 2)] * 2)
    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    incidence = edge_incidence(faces)
    constraints = np.asarray([edge for edge, ids in incidence.items()
                              if len(ids) != 2 or labels[ids[0]] != labels[ids[1]]],
                             dtype=np.int64)
    fixed = np.unique(constraints)
    for vertex_id, delta in ((size + 1, (0.07, -0.04)),
                              (3 * size + 3, (-0.07, 0.04))):
        if vertex_id < len(vertices) and vertex_id not in fixed:
            vertices[vertex_id, :2] += delta
    if curved:
        angles = (vertices[:, 0] - .5) * 1.2
        vertices = np.c_[np.cos(angles), np.sin(angles), vertices[:, 1]]
    return vertices, faces, labels, constraints


@unittest.skipIf(surface_sample is None, 'surface remeshing dependencies unavailable')
class WholePatchProjectionTests(unittest.TestCase):
    def test_indexed_collapse_cache_preserves_orientation_labels_and_invalidates_on_motion(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
        reference = np.vstack((vertices, vertices + (0, 0, .01)))
        calls = []

        class CountedProjector(surface_sample._ReferencePatchProjector):
            def valid_triangles(self, triangles, *args, **kwargs):
                calls.append(len(triangles))
                return super().valid_triangles(triangles, *args, **kwargs)

        projector = CountedProjector(reference, np.array([[0, 1, 2], [3, 4, 5]]), np.array([7, 9]))
        positions = torch.tensor(vertices)
        faces = torch.tensor([[0,1,2], [0,1,2], [0,1,2], [0,2,1]])
        labels = torch.tensor([7,7,9,7])
        cache = surface_sample._IndexedTriangleValidationCache(positions, projector, 1e-6, 5)
        self.assertEqual(cache.check(faces, labels).tolist(), [True, True, False, False])
        self.assertEqual(cache.check(faces.flip(0), labels.flip(0)).tolist(), [False, False, True, True])
        self.assertEqual(calls, [3])
        positions[0, 2] = .1
        self.assertFalse(bool(cache.check(faces, labels).any()))
        positions[0, 2] = 0
        self.assertEqual(cache.check(faces, labels).tolist(), [True, True, False, False])
        self.assertEqual(calls, [3, 3, 3])
        for limit in (0, 1, 2):
            bounded = surface_sample._IndexedTriangleValidationCache(positions, projector, 1e-6, 5,
                                                                     maximum_entries=limit)
            for _ in range(2):
                self.assertEqual(bounded.check(faces, labels).tolist(), [True, True, False, False])
                self.assertLessEqual(len(bounded.keys), limit)

    def test_normal_support_index_is_conservative_across_scales_and_radius_buckets(self):
        rng = np.random.default_rng(18)
        sizes = np.geomspace(1e-5, 1e3, 48)
        triangles = rng.normal(size=(48, 3, 3)) * sizes[:, None, None]
        triangles += rng.normal(size=(48, 1, 3)) * 5
        # Also cover the special thin-reference-only search and a second label.
        triangles[::3, 2] = triangles[::3, 0] + .4 * (
            triangles[::3, 1] - triangles[::3, 0]) + (0., 0., 1e-10)
        labels = np.arange(48) % 2
        faces = np.arange(48 * 3).reshape(-1, 3)
        for scale in (1e-6, 1., 1e6):
            with self.subTest(scale=scale):
                vertices = (triangles.reshape(-1, 3) + (3000., 20., -15.)) * scale
                projector = surface_sample._ReferencePatchProjector(vertices, faces, labels)
                reference = vertices[faces]
                lower, upper = reference.min(axis=1), reference.max(axis=1)
                tolerance = projector.normal_tie_tolerance
                points = np.vstack((reference.mean(axis=1), lower - tolerance, upper + tolerance))
                for label in (0, 1):
                    for thin_only in (False, True):
                        ids = projector.thin_patch_faces[label] if thin_only else projector.patch_faces[label]
                        candidates = projector._normal_support_candidates(points, label, thin_only)
                        for point, found in zip(points, candidates):
                            inside = ((point >= lower[ids] - tolerance) &
                                      (point <= upper[ids] + tolerance)).all(axis=1)
                            self.assertTrue(set(ids[inside]) <= set(found))
                            self.assertTrue(set(found) <= set(ids))
                            np.testing.assert_array_equal(found, np.unique(found))

    def test_relaxation_cache_rechecks_changed_geometry_and_preserves_patch_scope(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
        reference = np.vstack((vertices, vertices + (0, 0, .01)))

        class CountedProjector(surface_sample._ReferencePatchProjector):
            def valid_triangles(self, triangles, *args, **kwargs):
                calls.append(len(triangles))
                return super().valid_triangles(triangles, *args, **kwargs)

        calls = []
        projector = CountedProjector(reference, np.array([[0, 1, 2], [3, 4, 5]]), np.array([7, 9]))
        triangles = torch.from_numpy(np.stack((vertices, vertices)))
        cache = surface_sample._TriangleValidationCache(projector, torch.tensor([7, 9]), 1e-6, 5)
        self.assertEqual(cache.check(triangles).tolist(), [True, False])
        # Identical coordinates on different sheets must not share validation.
        mask = cache.check(triangles)
        mask[:] = False
        self.assertEqual(cache.check(triangles).tolist(), [True, False])
        self.assertEqual(calls, [2])
        triangles[1, :, 2] = .01
        self.assertEqual(cache.check(triangles).tolist(), [True, True])
        triangles[0] = triangles[0, [0, 2, 1]].clone()
        self.assertEqual(cache.check(triangles).tolist(), [False, True])
        triangles[0] = torch.from_numpy(vertices)
        self.assertEqual(cache.check(triangles).tolist(), [True, True])
        # Even a sub-tolerance change is recomputed; cache equality is exact.
        triangles[0, 0, 2] += 1e-12
        self.assertEqual(cache.check(triangles).tolist(), [True, True])
        self.assertEqual(calls, [2, 1, 1, 1, 1])

    def test_thin_reference_support_uses_stable_point_on_triangle_check(self):
        # This source triangle's aspect ratio makes libigl's closest point
        # miss its own centroid and select the adjacent face's 25-degree normal.
        vertices = np.array([
            [3063.46240234375, -24.833763122558594, -10.123769760131836],
            [3063.46240234375, -29.833763122558594, -10.123785018920898],
            [3063.46240234375, -29.833763122558594, -10.12376880645752],
            [3063.069091796875, -24.833763122558594, -9.305299758911133],
        ])
        faces = np.array([[3, 0, 2], [0, 1, 2]])
        projector = surface_sample._ReferencePatchProjector(vertices, faces, np.array([7, 7]))
        triangle = torch.from_numpy(vertices[faces[1:]])
        valid, details = projector.valid_triangles(triangle, torch.tensor([7]), 1e-5, 5,
                                                  return_details=True)
        self.assertTrue(bool(valid[0]))
        self.assertEqual(int(details['centroid_reference_face_ids'][0]), 1)
        self.assertLess(float(details['normal_deviation_degrees'][0]), 1e-6)
        # A changed candidate must prove current geometric membership; the
        # nearby thin source face does not waive the oriented-normal check.
        changed = triangle.clone()
        changed[:, :, 2] += 1e-4
        self.assertFalse(bool(projector.valid_triangles(changed, torch.tensor([7]), .17, 5)[0]))

    def test_overlapping_opposite_reference_face_does_not_reject_unchanged_thin_triangle(self):
        vertices = np.array([
            [3185.849609375, 94.21623992919922, -13.852574348449707],
            [3185.849609375, 94.21623992919922, -13.852519989013672],
            [3184.040283203125, 94.21623992919922, 13.955145835876465],
            [3187.689697265625, 94.21623992919922, -18.787179946899414],
        ])
        faces = np.array([[2, 3, 0], [0, 1, 2]])
        projector = surface_sample._ReferencePatchProjector(vertices, faces, np.array([7, 7]))
        triangle = torch.from_numpy(vertices[faces[1:]])
        valid, details = projector.valid_triangles(triangle, torch.tensor([7]), .17, 5,
                                                  return_details=True)
        self.assertTrue(bool(valid[0]))
        self.assertLess(float(details['normal_deviation_degrees'][0]), 5)
        self.assertEqual(int(details['centroid_reference_face_ids'][0]), 1)

    def test_normal_tie_resolution_does_not_accept_a_distinct_nearby_sheet(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
        reference = np.vstack((vertices, vertices + (0, 0, 1e-7)))
        projector = surface_sample._ReferencePatchProjector(reference,
            np.array([[0, 1, 2], [3, 5, 4]]), np.array([0, 0]))
        reversed_triangle = torch.from_numpy(vertices[np.array([[0, 2, 1]])])
        valid = projector.valid_triangles(reversed_triangle, torch.tensor([0]), 1e-6, 5)
        self.assertFalse(bool(valid[0]))

    def test_collapse_can_remove_original_source_edges_without_compacting_prefix(self):
        vertices = torch.tensor([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0],
                                 [.05, .05, 0]], dtype=torch.float64)
        faces = torch.tensor([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        labels = torch.zeros(4, dtype=torch.long)
        support = torch.tensor([-2, -2, -2, -2, -1])
        projector = surface_sample._ReferencePatchProjector(vertices.numpy(), faces.numpy(), labels.numpy())
        result = surface_sample._gpu_collapse_short_edges(
            vertices, faces, labels, torch.arange(4), support, .2, 2., 2,
            compact_vertices=False, reference_patch_projector=projector,
            maximum_surface_deviation=1e-8, maximum_normal_deviation_degrees=5)
        self.assertEqual(result[-1], 1)
        self.assertTrue(torch.equal(result[0], vertices))
        self.assertEqual(len(result[1]), 2)
        self.assertNotIn(4, set(result[1].reshape(-1).tolist()))
        self.assertIn((0, 2), edge_incidence(result[1].numpy()))
        self.assertTrue((result[2] == 0).all())

    def test_projection_crosses_old_triangle_edge_but_never_changes_patch(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        reference_vertices = np.vstack((vertices, vertices + (0, 0, .01)))
        reference_faces = np.vstack((faces, faces + 4))
        projector = surface_sample._ReferencePatchProjector(
            reference_vertices, reference_faces, np.array([7, 7, 9, 9]))
        query = torch.tensor([[.2, .8, .009]], dtype=torch.float64)
        whole, sources = projector.project(query, torch.tensor([7]))
        legacy = surface_sample._project_to_support_triangles(
            query, torch.tensor(vertices[faces[:1]], dtype=torch.float64))
        self.assertLess(float(legacy[0, 1]), .8)
        np.testing.assert_allclose(whole.numpy(), [[.2, .8, 0]], atol=1e-14)
        self.assertEqual(int(sources[0]), 1)
        other, _ = projector.project(query, torch.tensor([9]))
        np.testing.assert_allclose(other.numpy(), [[.2, .8, .01]], atol=1e-14)
        self.assertEqual(len(projector.trees), 2)

    def test_surface_samples_reject_hole_bridge_even_with_on_surface_vertices(self):
        vertices, faces, labels, _ = grid_fixture(hole=True)
        projector = surface_sample._ReferencePatchProjector(vertices, faces, labels)
        bridging = torch.tensor([[[.25, .25, 0], [.75, .25, 0], [.5, .75, 0]]],
                                dtype=torch.float64)
        _, squared, _ = projector.query_numpy(bridging.numpy().reshape(-1, 3), [0, 0, 0])
        self.assertLess(float(squared.max()), 1e-20)
        valid = projector.valid_triangles(bridging, torch.tensor([0]), 1e-6, 5)
        self.assertFalse(bool(valid[0]), 'a plane-distance-only check would fill this hole')

    def test_oriented_normal_check_rejects_reversed_triangle(self):
        vertices, faces, labels, _ = grid_fixture()
        projector = surface_sample._ReferencePatchProjector(vertices, faces, labels)
        triangle = torch.tensor(vertices[faces[:1, ::-1].copy()], dtype=torch.float64)
        self.assertFalse(bool(projector.valid_triangles(triangle, torch.tensor([0]), 1e-6, 10)[0]))

    def test_original_interior_vertex_can_relax_while_boundary_prefix_stays_fixed(self):
        boundary = np.array([[0., 0, 0], [.5, 0, 0], [1, 0, 0], [1, .5, 0],
                             [1, 1, 0], [.5, 1, 0], [0, 1, 0], [0, .5, 0]])
        vertices = torch.tensor(np.vstack((boundary, [[.72, .32, 0]])), dtype=torch.float64)
        faces = torch.tensor([[i, (i + 1) % 8, 8] for i in range(8)], dtype=torch.long)
        labels = torch.zeros(8, dtype=torch.long)
        support = torch.tensor([-2] * 8 + [-1])
        projector = surface_sample._ReferencePatchProjector(vertices.numpy(), faces.numpy(), labels.numpy())
        legacy, legacy_accepted = surface_sample._gpu_relax_inserted_vertices(
            vertices, faces, vertices, faces, support, 5, .5)
        self.assertEqual(legacy_accepted, 0)
        self.assertTrue(torch.equal(legacy, vertices))
        result, accepted = surface_sample._gpu_relax_patch_vertices(
            vertices, faces, labels, support, projector, 5, .5, 2., 1e-8, 5)
        self.assertGreater(accepted, 0)
        self.assertGreater(float(torch.linalg.norm(result[8] - vertices[8])), .1)
        self.assertTrue(torch.equal(result[:8], vertices[:8]))
        self.assertLess(float(surface_sample._mesh_energy_torch(result, faces)),
                        float(surface_sample._mesh_energy_torch(vertices, faces)))
        np.testing.assert_allclose(result[:, 2], 0, atol=1e-14)


@unittest.skipIf(surface_sample is None or not torch.cuda.is_available(), 'CUDA/dependencies unavailable')
class WholePatchCudaTests(unittest.TestCase):
    def remesh(self, vertices, faces, labels, constraints, *, whole=True, **options):
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        arguments = dict(sample_count=1, poisson_radius=.25, minimum_source_area_ratio=100,
                         protected_source_quality=.01, collapse_passes=0, flip_passes=0,
                         split_passes=16, relax_iterations=5, smoothing_step=.5,
                         maximum_normal_deviation_degrees=15,
                         external_face_patch_ids=labels, fixed_constraint_edges=constraints,
                         fixed_corner_vertex_ids=np.unique(constraints)[:1],
                         whole_patch_optimization=whole)
        arguments.update(options)
        return surface_sample.surface_sample_remesh(
            mesh, torch.tensor(vertices, dtype=torch.float64, device='cuda'),
            torch.tensor(faces, dtype=torch.long, device='cuda'), **arguments)

    def assert_partition_geometry(self, original, faces, labels, constraints, result):
        vertices, output_faces, stats = result
        fixed = np.unique(constraints)
        np.testing.assert_array_equal(vertices[fixed], original[fixed])
        incidence = edge_incidence(output_faces)
        self.assertTrue(set(map(tuple, np.sort(constraints, axis=1))) <= incidence.keys())
        self.assertTrue(all(len(ids) <= 2 for ids in incidence.values()))
        self.assertEqual(set(stats['face_patch_ids']), set(labels))
        np.testing.assert_array_equal(stats['face_patch_ids'], labels[stats['source_face_ids']])
        edges = np.array(list(incidence))
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        self.assertLessEqual(float(lengths.max()), .5 * (1 + 1e-6))
        projector = surface_sample._ReferencePatchProjector(original, faces, labels)
        triangle_points = vertices[output_faces].reshape(-1, 3)
        _, squared, _ = projector.query_numpy(triangle_points,
                                              np.repeat(stats['face_patch_ids'], 3))
        self.assertLess(float(squared.max()), 1e-18)

    def test_whole_patch_bypasses_source_quality_cage_and_moves_original_vertices(self):
        vertices, faces, labels, constraints = grid_fixture(two_patches=True)
        legacy = self.remesh(vertices, faces, labels, constraints, whole=False)
        np.testing.assert_array_equal(legacy[0][:len(vertices)], vertices)
        whole = self.remesh(vertices, faces, labels, constraints)
        self.assert_partition_geometry(vertices, faces, labels, constraints, whole)
        self.assertEqual(whole[2]['protected_source_face_count'], 0)
        self.assertGreater(legacy[2]['protected_source_face_count'], 0)
        self.assertGreater(whole[2]['original_interior_vertices_moved'], 0)
        self.assertTrue(whole[2]['whole_patch_optimization'])
        centroids = whole[0][whole[1]].mean(axis=1)
        self.assertTrue((centroids[whole[2]['face_patch_ids'] == 0, 0] <= .5).all())
        self.assertTrue((centroids[whole[2]['face_patch_ids'] == 1, 0] >= .5).all())
        self.assertLess(whole[2]['final_metrics']['edge_cv'], legacy[2]['final_metrics']['edge_cv'])

    def test_curved_patch_uses_reference_triangles_and_preserves_boundary(self):
        vertices, faces, labels, constraints = grid_fixture(curved=True)
        result = self.remesh(vertices, faces, labels, constraints,
                             collapse_passes=3, flip_passes=3, maximum_surface_deviation_ratio=.1)
        self.assert_partition_geometry(vertices, faces, labels, constraints, result)
        self.assertGreater(result[2]['original_interior_vertices_moved'], 0)
        self.assertEqual(result[2]['source_face_id_semantics'],
                         'nearest_local_reference_support_on_same_patch')

    def test_planar_hole_is_not_filled_by_cross_source_operations(self):
        vertices, faces, labels, constraints = grid_fixture(hole=True)
        result = self.remesh(vertices, faces, labels, constraints, collapse_passes=3, flip_passes=3)
        self.assert_partition_geometry(vertices, faces, labels, constraints, result)
        before = edge_incidence(faces)
        after = edge_incidence(result[1])
        self.assertEqual(sum(len(ids) == 1 for ids in before.values()),
                         sum(len(ids) == 1 for ids in after.values()))
        centroids = result[0][result[1]].mean(axis=1)
        self.assertFalse(((centroids[:, 0] > .25) & (centroids[:, 0] < .75) &
                          (centroids[:, 1] > .25) & (centroids[:, 1] < .75)).any())

    def test_whole_mode_requires_external_labels_and_matching_reference_domain(self):
        import trimesh

        vertices, faces, labels, constraints = grid_fixture()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        with self.assertRaisesRegex(ValueError, 'requires external_face_patch_ids'):
            surface_sample.surface_sample_remesh(
                mesh, torch.tensor(vertices, device='cuda'), torch.tensor(faces, device='cuda'),
                1, whole_patch_optimization=True)
        with self.assertRaisesRegex(ValueError, 'missing an external patch ID'):
            self.remesh(vertices, faces, labels, constraints,
                        whole_patch_reference=(vertices, faces, labels + 100))

    def test_complete_cached_reference_retains_local_batch_source_indices(self):
        full_vertices, full_faces, full_labels, _ = grid_fixture()
        projector = surface_sample._ReferencePatchProjector(full_vertices, full_faces[::-1], full_labels)
        ids, local_faces = np.unique(full_faces[:16], return_inverse=True)
        local_vertices = full_vertices[ids]
        local_faces = local_faces.reshape(-1, 3)
        local_labels = np.zeros(len(local_faces), dtype=np.int64)
        constraints = np.array([edge for edge, owners in edge_incidence(local_faces).items()
                                if len(owners) == 1], dtype=np.int64)
        result = self.remesh(local_vertices, local_faces, local_labels, constraints,
                             whole_patch_reference=projector)
        self.assert_partition_geometry(local_vertices, local_faces, local_labels, constraints, result)
        self.assertEqual(result[2]['projection_reference'], 'complete_supplied_patch')
        local_projector = surface_sample._ReferencePatchProjector(local_vertices, local_faces, local_labels)
        _, _, nearest_sources = local_projector.query_numpy(
            result[0][result[1]].mean(axis=1), result[2]['face_patch_ids'])
        np.testing.assert_array_equal(result[2]['source_face_ids'], nearest_sources)
        tree = projector.trees[0][0]
        self.remesh(local_vertices, local_faces, local_labels, constraints,
                    whole_patch_reference=projector, relax_iterations=0)
        self.assertIs(projector.trees[0][0], tree)


if __name__ == '__main__':
    unittest.main()
