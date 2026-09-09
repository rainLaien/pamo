import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch


PAMO_PATH = Path(__file__).parents[1] / "pamo"
package_name = "_pamo_surface_sample_test"
package = types.ModuleType(package_name)
package.__path__ = [str(PAMO_PATH)]
sys.modules[package_name] = package

try:
    import igl  # noqa: F401
    import scipy  # noqa: F401
    import trimesh  # noqa: F401
except ImportError:
    surface_sample = None
else:
    for module_name in (
        "segment_query",
        "feature_edges",
        "feature_optimize",
        "surface_sample",
    ):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            PAMO_PATH / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)
    surface_sample = sys.modules[f"{package_name}.surface_sample"]


@unittest.skipIf(
    surface_sample is None,
    "Surface sampling dependencies are unavailable",
)
class SurfaceSampleTest(unittest.TestCase):
    def test_relaxation_backtracks_when_quality_improvement_exceeds_edge_limit(self):
        # A planar mesh where unconstrained smoothing lowers energy but makes
        # the longest edge longer. Fixed outer samples bound six interior points.
        t = np.linspace(0.0, 1.0, 6)[:-1]
        boundary = np.vstack((np.c_[t, t * 0], np.c_[t * 0 + 1, t],
                              np.c_[1 - t, t * 0 + 1], np.c_[t * 0, 1 - t]))
        interior = np.array([
            [0.49703165761580637, 0.4714812809839185],
            [0.16492129060322996, 0.28180625441857143],
            [0.05286299836398045, 0.39296097303423716],
            [0.5682857756332699, 0.43456889415465505],
            [0.8015921125777125, 0.6048421262691708],
            [0.28947552081014727, 0.7799199001862748],
        ])
        points = np.vstack((boundary, interior))
        vertices = torch.tensor(np.c_[points, np.zeros(len(points))], dtype=torch.float64)
        faces = torch.tensor([
            [24,12,20], [12,25,20], [23,24,20], [2,23,20], [23,2,3],
            [2,21,1], [21,2,20], [24,7,8], [7,23,6], [23,7,24],
            [24,11,12], [25,16,17], [16,14,15], [14,16,25], [4,23,3],
            [23,4,6], [6,4,5], [22,21,20], [25,22,20], [22,25,17],
            [11,9,10], [9,11,24], [9,24,8], [13,25,12], [13,14,25],
            [18,22,17], [1,19,0], [21,19,1], [22,19,21], [18,19,22],
        ], dtype=torch.long)
        reference_vertices = torch.tensor(
            [[-100., -100., 0.], [100., -100., 0.], [0., 100., 0.]], dtype=torch.float64,
        )
        arguments = (vertices, faces, reference_vertices, torch.tensor([[0, 1, 2]]),
                     torch.tensor([-2] * 20 + [0] * 6), 3, 0.5)
        edges = surface_sample._unique_edges_torch(faces)
        def longest(v):
            return float(torch.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], dim=1).max())
        target = longest(vertices)
        unconstrained, _ = surface_sample._gpu_relax_inserted_vertices(*arguments)
        self.assertGreater(longest(unconstrained), target * 1.001)
        constrained, accepted = surface_sample._gpu_relax_inserted_vertices(
            *arguments, maximum_edge_length=target,
        )
        self.assertGreater(accepted, 0)
        self.assertLessEqual(longest(constrained), target * (1.0 + 1e-6))
        self.assertTrue(torch.equal(constrained[:20], vertices[:20]))
        self.assertLess(float(surface_sample._mesh_energy_torch(constrained, faces)),
                        float(surface_sample._mesh_energy_torch(vertices, faces)))

    def test_external_partition_requires_complete_valid_interfaces(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        boundaries = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
        constraints = np.vstack((boundaries, [[0, 2]]))
        labels, checked, corners = surface_sample._validated_external_partition(
            vertices, faces, np.array([10, 20]), constraints, [1],
        )
        np.testing.assert_array_equal(labels, [10, 20])
        self.assertEqual(len(checked), 5)
        np.testing.assert_array_equal(corners, [1])
        for bad_labels in ([0], [0., 1.], [0, -1], [True, False]):
            with self.subTest(labels=bad_labels), self.assertRaises(ValueError):
                surface_sample._validated_external_partition(vertices, faces, bad_labels, constraints)
        for bad_edges in (None, [], boundaries, np.vstack((constraints, [[1, 3]])), [[0., 2.]]):
            with self.subTest(edges=bad_edges), self.assertRaises(ValueError):
                surface_sample._validated_external_partition(vertices, faces, [10, 20], bad_edges)
        for bad_corners in ([4], [-1], [0.5], [[0]]):
            with self.subTest(corners=bad_corners), self.assertRaises(ValueError):
                surface_sample._validated_external_partition(vertices, faces, [10, 20], constraints, bad_corners)

    def test_external_partition_requires_nonmanifold_constraints(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.]])
        faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
        edges = np.unique(np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0)
        surface_sample._validated_external_partition(vertices, faces, [0, 0, 0], edges)
        with self.assertRaisesRegex(ValueError, "non-manifold"):
            surface_sample._validated_external_partition(vertices, faces, [0, 0, 0], edges[~np.all(edges == [0, 1], axis=1)])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_external_partition_rejects_mismatched_geometry_or_precision(self):
        import trimesh
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        gpu_vertices = torch.as_tensor(vertices, device="cuda")
        gpu_faces = torch.as_tensor(faces, device="cuda")
        arguments = dict(sample_count=4, external_face_patch_ids=[0, 0], fixed_constraint_edges=[[0, 1], [1, 2], [2, 3], [0, 3]])
        for supplied_vertices, supplied_faces, message in (
            (gpu_vertices.float(), gpu_faces, "float64"),
            (gpu_vertices + 1.0, gpu_faces, "coordinates"),
            (gpu_vertices, gpu_faces.flip(0), "face ordering"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                surface_sample.surface_sample_remesh(mesh, supplied_vertices, supplied_faces, **arguments)
        with self.assertRaisesRegex(ValueError, "require external_face_patch_ids"):
            surface_sample.surface_sample_remesh(mesh, gpu_vertices, gpu_faces, sample_count=4, fixed_constraint_edges=[])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_external_coplanar_patches_keep_shared_interface_and_labels(self):
        import trimesh
        # The bridge subdivides interfaces before the GPU stage. Every fixed
        # segment is therefore shorter than this test's 0.6 edge bound.
        vertices = np.array([[x * 0.5, y * 0.5, 0.] for y in range(3) for x in range(3)])
        faces = np.array([[a, a + 1, a + 4] for a in (0, 1, 3, 4)] + [[a, a + 4, a + 3] for a in (0, 1, 3, 4)], dtype=np.int64)
        input_labels = np.array([10, 20, 10, 20, 10, 20, 10, 20])
        constraints = np.array([[0, 1], [1, 2], [2, 5], [5, 8], [7, 8], [6, 7], [3, 6], [0, 3], [1, 4], [4, 7]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        with mock.patch.object(surface_sample, "detect_reference_feature_edges", side_effect=AssertionError("Must not redetect features")), mock.patch.object(surface_sample, "original_surface_patch_ids", side_effect=AssertionError("Must not repartition")):
            output_vertices, output_faces, stats = surface_sample.surface_sample_remesh(
                mesh, torch.as_tensor(vertices, device="cuda"), torch.as_tensor(faces, device="cuda"),
                sample_count=24, poisson_radius=0.3, seed=37,
                protected_source_quality=0.0, collapse_passes=4,
                flip_passes=3, split_passes=12, relax_iterations=2,
                external_face_patch_ids=input_labels,
                fixed_constraint_edges=constraints, fixed_corner_vertex_ids=[0],
            )
        self.assertEqual(output_vertices.dtype, np.float64)
        np.testing.assert_array_equal(output_vertices[:9], vertices)
        self.assertGreater(len(output_faces), len(faces))
        self.assertEqual(stats["patch_count"], 2)
        self.assertEqual(stats["feature_edge_count"], 0)
        self.assertEqual(stats["fixed_constraint_edge_count"], 10)
        np.testing.assert_array_equal(stats["face_patch_ids"], input_labels[stats["source_face_ids"]])
        centroid = output_vertices[output_faces].mean(axis=1)
        self.assertTrue(np.all(centroid[stats["face_patch_ids"] == 10, 0] <= 0.5 + 1e-12))
        self.assertTrue(np.all(centroid[stats["face_patch_ids"] == 20, 0] >= 0.5 - 1e-12))
        final_mesh = trimesh.Trimesh(vertices=output_vertices, faces=output_faces, process=False)
        self.assertLessEqual(float(final_mesh.edges_unique_length.max()), 0.6 * (1.0 + 1e-7))
        for edge in constraints:
            self.assertTrue(np.any(np.all(np.sort(final_mesh.edges_unique, axis=1) == edge, axis=1)))
        edge_counts = np.bincount(final_mesh.edges_unique_inverse)
        self.assertEqual(int(np.count_nonzero(edge_counts == 1)), 8)
        self.assertTrue(np.all(edge_counts <= 2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_external_closed_partition_accepts_empty_constraints_and_fixed_corner(self):
        import trimesh
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        output_vertices, output_faces, stats = surface_sample.surface_sample_remesh(
            mesh, torch.as_tensor(vertices, device="cuda"), torch.as_tensor(faces, device="cuda"),
            sample_count=24, poisson_radius=0.25, seed=7,
            protected_source_quality=0.0, collapse_passes=3,
            flip_passes=2, split_passes=12, relax_iterations=2,
            external_face_patch_ids=np.zeros(4, dtype=np.int64),
            fixed_constraint_edges=[], fixed_corner_vertex_ids=[0],
        )
        np.testing.assert_array_equal(output_vertices[0], vertices[0])
        np.testing.assert_array_equal(stats["face_patch_ids"], np.zeros(len(output_faces), dtype=np.int64))
        self.assertEqual(stats["fixed_constraint_edge_count"], 0)
        self.assertTrue(trimesh.Trimesh(vertices=output_vertices, faces=output_faces, process=False).is_watertight)

    def test_cgal_adaptive_target_length_formula_and_clamps(self):
        tolerance = 0.1
        curvature = np.array([0.0, 1.0, 1000.0])
        lengths = surface_sample.adaptive_target_length_from_curvature(
            curvature,
            tolerance=tolerance,
            minimum_edge_length=0.2,
            maximum_edge_length=2.0,
        )

        self.assertEqual(lengths[0], 2.0)
        self.assertAlmostEqual(lengths[1], np.sqrt(0.57))
        self.assertEqual(lengths[2], 0.2)

    def test_planar_curvature_field_uses_maximum_target(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        vertex_lengths, face_lengths = (
            surface_sample.curvature_adaptive_source_face_lengths(
                vertices,
                faces,
                tolerance=0.01,
                minimum_edge_length=0.1,
                maximum_edge_length=1.0,
            )
        )

        np.testing.assert_allclose(vertex_lengths, 1.0)
        np.testing.assert_allclose(face_lengths, 1.0)

    def test_local_retriangulation_preserves_original_boundary(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        barycentric = np.array(
            [[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]]
        )
        source_faces = np.array([0, 1], dtype=np.int64)
        points = (
            vertices[faces[source_faces]]
            * barycentric[:, :, None]
        ).sum(axis=1)

        output_vertices, output_faces, _, support = (
            surface_sample.subdivide_original_faces(
                vertices,
                faces,
                points,
                source_faces,
                barycentric,
            )
        )

        self.assertEqual(len(output_vertices), 6)
        self.assertEqual(len(output_faces), 6)
        np.testing.assert_array_equal(support[:4], -1)
        np.testing.assert_array_equal(support[4:], source_faces)
        edges = {
            tuple(sorted(edge))
            for face in output_faces
            for edge in (
                face[[0, 1]],
                face[[1, 2]],
                face[[2, 0]],
            )
        }
        for boundary_edge in ((0, 1), (1, 2), (2, 3), (0, 3)):
            self.assertIn(boundary_edge, edges)

    def test_coplanar_shared_edge_is_not_subdivided(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        output_vertices, _, _, _ = surface_sample.subdivide_original_faces(
            vertices,
            faces,
            np.array(
                [
                    [3.0, 0.25, 0.0],
                    [2.0, 0.75, 0.0],
                ],
                dtype=np.float64,
            ),
            np.array([0, 1], dtype=np.int64),
            np.array(
                [
                    [0.25, 0.5, 0.25],
                    [0.25, 0.5, 0.25],
                ],
                dtype=np.float64,
            ),
            edge_target_length=0.5,
            protected_internal_edges=np.array([[0, 2]], dtype=np.int64),
        )

        diagonal_points = output_vertices[
            np.isclose(output_vertices[:, 0], output_vertices[:, 1])
        ]
        self.assertEqual(len(diagonal_points), 2)

    def test_feature_edge_separates_surface_patches(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        one_patch = surface_sample.original_surface_patch_ids(
            faces,
            vertex_count=4,
            feature_edges=np.array(
                [[0, 1], [1, 2], [2, 3], [0, 3]],
                dtype=np.int64,
            ),
        )
        two_patches = surface_sample.original_surface_patch_ids(
            faces,
            vertex_count=4,
            feature_edges=np.array([[0, 2]], dtype=np.int64),
        )

        self.assertEqual(len(np.unique(one_patch)), 1)
        self.assertEqual(len(np.unique(two_patches)), 2)

    def test_shared_edge_samples_form_an_exact_patch_boundary_chain(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        barycentric = np.array([[0.2, 0.3, 0.5]])
        point = (
            vertices[faces[[0]]] * barycentric[:, :, None]
        ).sum(axis=1)
        patches = surface_sample.original_surface_patch_ids(
            faces,
            vertex_count=4,
            feature_edges=np.array([[0, 2]], dtype=np.int64),
        )

        output_vertices, output_faces, source_faces, _ = (
            surface_sample.subdivide_original_faces(
                vertices,
                faces,
                point,
                np.array([0], dtype=np.int64),
                barycentric,
                edge_target_length=0.3,
            )
        )
        edge_owners = {}
        output_patches = patches[source_faces]
        for face_id, face in enumerate(output_faces):
            for edge in (
                face[[0, 1]],
                face[[1, 2]],
                face[[2, 0]],
            ):
                edge_owners.setdefault(tuple(sorted(edge)), []).append(
                    face_id
                )
        interface_edges = [
            edge
            for edge, owners in edge_owners.items()
            if len(owners) == 2
            and output_patches[owners[0]] != output_patches[owners[1]]
        ]
        interface_vertices = np.unique(interface_edges)
        segment_start = vertices[0]
        segment_vector = vertices[2] - segment_start
        offsets = output_vertices[interface_vertices] - segment_start
        cross = np.cross(offsets, segment_vector)

        self.assertEqual(len(interface_edges), 5)
        self.assertIn(0, interface_vertices)
        self.assertIn(2, interface_vertices)
        np.testing.assert_allclose(cross, 0.0, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_area_samples_remain_on_source_faces(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device="cuda")

        samples = surface_sample.gpu_area_poisson_sample(
            vertices,
            faces,
            sample_count=32,
            poisson_radius=None,
            seed=7,
        )

        self.assertEqual(len(samples.points), 32)
        self.assertTrue(bool((samples.face_ids == 0).all()))
        self.assertTrue(
            bool(
                torch.allclose(
                    samples.barycentric.sum(dim=1),
                    torch.ones(32, device="cuda"),
                )
            )
        )
        self.assertTrue(bool((samples.barycentric > 0.0).all()))
        self.assertTrue(bool((samples.points[:, 2] == 0.0).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_poisson_samples_respect_radius(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device="cuda")
        radius = 0.04
        samples = surface_sample.gpu_area_poisson_sample(
            vertices,
            faces,
            sample_count=100,
            poisson_radius=radius,
            oversample=8,
            seed=11,
            barycentric_margin=0.0,
        )
        distances = torch.cdist(samples.points, samples.points)
        distances.fill_diagonal_(float("inf"))

        self.assertGreater(len(samples.points), 1)
        self.assertGreaterEqual(
            float(distances.min()),
            radius - 1e-6,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_flip_improves_nonfeature_quad(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        faces = torch.tensor(
            [[0, 1, 2], [2, 1, 3]],
            dtype=torch.long,
            device="cuda",
        )
        patches = torch.zeros(2, dtype=torch.long, device="cuda")
        before = surface_sample._triangle_quality_torch(vertices, faces).min()

        output_faces, flip_count = surface_sample._gpu_flip_quality_edges(
            vertices,
            faces,
            patches,
            passes=1,
        )
        after = surface_sample._triangle_quality_torch(
            vertices,
            output_faces,
        ).min()

        self.assertEqual(flip_count, 1)
        self.assertGreater(float(after), float(before))

        protected_faces, protected_flip_count = (
            surface_sample._gpu_flip_quality_edges(
                vertices,
                faces,
                torch.tensor([0, 1], dtype=torch.long, device="cuda"),
                passes=1,
            )
        )
        self.assertEqual(protected_flip_count, 0)
        self.assertTrue(bool(torch.equal(protected_faces, faces)))

        explicit_feature_faces, explicit_feature_flip_count = (
            surface_sample._gpu_flip_quality_edges(
                vertices,
                faces,
                patches,
                passes=1,
                protected_vertex_mask=torch.tensor(
                    [False, True, True, False],
                    dtype=torch.bool,
                    device="cuda",
                ),
            )
        )
        self.assertEqual(explicit_feature_flip_count, 0)
        self.assertTrue(bool(torch.equal(explicit_feature_faces, faces)))

        protected_source_faces, protected_source_flip_count = (
            surface_sample._gpu_flip_quality_edges(
                vertices,
                faces,
                patches,
                passes=1,
                face_source_ids=torch.arange(
                    2,
                    dtype=torch.long,
                    device="cuda",
                ),
                protected_source_face_mask=torch.tensor(
                    [True, False],
                    dtype=torch.bool,
                    device="cuda",
                ),
            )
        )
        self.assertEqual(protected_source_flip_count, 0)
        self.assertTrue(bool(torch.equal(protected_source_faces, faces)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_long_edge_split_reaches_requested_bound(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        faces = torch.tensor(
            [[0, 1, 2], [0, 2, 3]],
            dtype=torch.long,
            device="cuda",
        )
        patches = torch.zeros(2, dtype=torch.long, device="cuda")
        sources = torch.arange(2, dtype=torch.long, device="cuda")
        support = torch.full(
            (4,),
            -1,
            dtype=torch.long,
            device="cuda",
        )

        (
            output_vertices,
            output_faces,
            _,
            _,
            _,
            split_count,
            remaining,
        ) = surface_sample._gpu_split_long_edges(
            vertices,
            faces,
            patches,
            sources,
            support,
            maximum_edge_length=0.75,
            maximum_passes=16,
        )
        edges = surface_sample._unique_edges_torch(output_faces)
        maximum = torch.linalg.norm(
            output_vertices[edges[:, 1]]
            - output_vertices[edges[:, 0]],
            dim=1,
        ).max()

        self.assertGreater(split_count, 0)
        self.assertEqual(remaining, 0)
        self.assertLessEqual(float(maximum), 0.75 + 1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_short_edge_collapse_locks_patch_boundary(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, 0.05, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        faces = torch.tensor(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            dtype=torch.long,
            device="cuda",
        )
        patches = torch.zeros(4, dtype=torch.long, device="cuda")
        sources = torch.arange(4, dtype=torch.long, device="cuda")
        support = torch.full(
            (5,),
            -1,
            dtype=torch.long,
            device="cuda",
        )

        (
            output_vertices,
            output_faces,
            _,
            _,
            _,
            collapse_count,
        ) = surface_sample._gpu_collapse_short_edges(
            vertices,
            faces,
            patches,
            sources,
            support,
            minimum_edge_length=0.2,
            maximum_edge_length=2.0,
            passes=2,
        )

        self.assertEqual(collapse_count, 1)
        self.assertEqual(len(output_vertices), 4)
        self.assertEqual(len(output_faces), 2)
        np.testing.assert_allclose(
            output_vertices.cpu().numpy(),
            vertices[:4].cpu().numpy(),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_short_edge_collapse_preserves_protected_source_one_ring(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, 0.05, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        faces = torch.tensor(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            dtype=torch.long,
            device="cuda",
        )
        patches = torch.zeros(4, dtype=torch.long, device="cuda")
        sources = torch.arange(4, dtype=torch.long, device="cuda")
        support = torch.full(
            (5,),
            -1,
            dtype=torch.long,
            device="cuda",
        )
        protected_sources = torch.tensor(
            [False, True, False, False],
            dtype=torch.bool,
            device="cuda",
        )

        (
            output_vertices,
            output_faces,
            _,
            _,
            _,
            collapse_count,
        ) = surface_sample._gpu_collapse_short_edges(
            vertices,
            faces,
            patches,
            sources,
            support,
            minimum_edge_length=0.2,
            maximum_edge_length=2.0,
            passes=2,
            protected_source_face_mask=protected_sources,
        )

        self.assertEqual(collapse_count, 0)
        self.assertTrue(bool(torch.equal(output_vertices, vertices)))
        self.assertTrue(bool(torch.equal(output_faces, faces)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_short_edge_collapse_preserves_rounded_normal_cone(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, 0.05, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        faces = torch.tensor(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            dtype=torch.long,
            device="cuda",
        )
        patches = torch.zeros(4, dtype=torch.long, device="cuda")
        sources = torch.arange(4, dtype=torch.long, device="cuda")
        support = torch.full(
            (5,),
            -1,
            dtype=torch.long,
            device="cuda",
        )
        angle = np.deg2rad(20.0)
        source_normals = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [np.sin(angle), 0.0, np.cos(angle)],
                [0.0, 0.0, 1.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        (
            output_vertices,
            output_faces,
            _,
            _,
            _,
            collapse_count,
        ) = surface_sample._gpu_collapse_short_edges(
            vertices,
            faces,
            patches,
            sources,
            support,
            minimum_edge_length=0.2,
            maximum_edge_length=2.0,
            passes=2,
            source_face_normals=source_normals,
            maximum_normal_deviation_degrees=5.0,
        )

        self.assertEqual(collapse_count, 0)
        self.assertTrue(bool(torch.equal(output_vertices, vertices)))
        self.assertTrue(bool(torch.equal(output_faces, faces)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_quality_driven_collapse_handles_thin_coplanar_fan(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 0.5, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        faces = torch.tensor(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            dtype=torch.long,
            device="cuda",
        )
        patches = torch.zeros(4, dtype=torch.long, device="cuda")
        sources = torch.arange(4, dtype=torch.long, device="cuda")
        support = torch.full(
            (5,),
            -1,
            dtype=torch.long,
            device="cuda",
        )

        _, _, _, _, _, collapse_count = (
            surface_sample._gpu_collapse_short_edges(
                vertices,
                faces,
                patches,
                sources,
                support,
                minimum_edge_length=0.5,
                maximum_edge_length=20.0,
                passes=2,
                minimum_collapse_quality=0.25,
            )
        )

        self.assertGreater(collapse_count, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_source_sensitive_vertices_lock_rounded_shared_edge(self):
        faces = torch.tensor(
            [[0, 1, 2], [2, 1, 3]],
            dtype=torch.long,
            device="cuda",
        )
        sources = torch.arange(2, dtype=torch.long, device="cuda")
        angle = np.deg2rad(20.0)
        source_normals = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        rounded_mask = surface_sample._gpu_source_sensitive_vertex_mask(
            faces,
            sources,
            source_normals,
            maximum_normal_deviation_degrees=5.0,
        )
        protected_mask = surface_sample._gpu_source_sensitive_vertex_mask(
            faces,
            sources,
            source_normals,
            protected_source_face_mask=torch.tensor(
                [True, False],
                dtype=torch.bool,
                device="cuda",
            ),
            maximum_normal_deviation_degrees=180.0,
        )

        self.assertTrue(
            bool(
                torch.equal(
                    rounded_mask,
                    torch.tensor(
                        [False, True, True, False],
                        dtype=torch.bool,
                        device="cuda",
                    ),
                )
            )
        )
        self.assertTrue(
            bool(
                torch.equal(
                    protected_mask,
                    torch.tensor(
                        [True, True, True, False],
                        dtype=torch.bool,
                        device="cuda",
                    ),
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
