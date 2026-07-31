import importlib.util
from pathlib import Path
import sys
import types
import unittest

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


if __name__ == "__main__":
    unittest.main()
