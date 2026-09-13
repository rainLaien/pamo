"""Mandatory refinement honors the requested reference-normal tolerance."""
from pathlib import Path
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simp_cuda.tests.test_surface_sample import surface_sample


@unittest.skipIf(surface_sample is None, "Surface sampling dependencies are unavailable")
class SplitNormalLimitTest(unittest.TestCase):
    def setUp(self):
        self.reference_vertices = torch.tensor(
            [[-1., 0., 0.], [0., 1e-4, 0.], [1., 0., 0.], [0., -1e-4, 0.]],
            dtype=torch.float64,
        )
        self.faces = torch.tensor([[0, 2, 1], [2, 0, 3]], dtype=torch.long)
        self.labels = torch.tensor([7, 7])
        self.sources = torch.tensor([17, 42])
        self.support = torch.full((4,), -2, dtype=torch.long)
        self.boundary = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]])
        self.projector = surface_sample._ReferencePatchProjector(
            self.reference_vertices.numpy(), self.faces.numpy(), self.labels.numpy(),
        )

    def tilted_vertices(self, upper_degrees, lower_degrees=None):
        if lower_degrees is None:
            lower_degrees = upper_degrees
        vertices = self.reference_vertices.clone()
        for index, degrees in ((1, upper_degrees), (3, lower_degrees)):
            angle = np.deg2rad(degrees)
            width = float(self.reference_vertices[index, 1])
            vertices[index, 1] = width * np.cos(angle)
            vertices[index, 2] = width * np.sin(angle)
        return vertices

    def chord_children(self, vertices):
        midpoint = .5 * (vertices[0] + vertices[2])
        return torch.stack((
            torch.stack((vertices[0], midpoint, vertices[1])),
            torch.stack((midpoint, vertices[2], vertices[1])),
            torch.stack((vertices[2], midpoint, vertices[3])),
            torch.stack((midpoint, vertices[0], vertices[3])),
        ))

    def split(self, vertices, **overrides):
        arguments = dict(
            vertices=vertices, faces=self.faces, face_patch_ids=self.labels,
            face_source_ids=self.sources, support_face_ids=self.support,
            maximum_edge_length=1.1, maximum_passes=8,
            protected_edges=self.boundary, reference_patch_projector=self.projector,
            maximum_surface_deviation=.01, maximum_normal_deviation_degrees=5.,
        )
        arguments.update(overrides)
        return surface_sample._gpu_split_long_edges(**arguments)

    def assert_unchanged(self, result, vertices):
        self.assertEqual(result[-2:], (0, 1))
        for actual, expected in zip(result[:5], (
            vertices, self.faces, self.labels, self.sources, self.support,
        )):
            self.assertTrue(torch.equal(actual, expected))

    def assert_valid_refinement(self, result, vertices):
        output_vertices, faces, labels, sources, support, splits, remaining = result
        self.assertEqual((splits, remaining), (1, 0))
        self.assertTrue(torch.equal(output_vertices[:4], vertices))
        self.assertTrue(torch.equal(support[:4], self.support))
        torch.testing.assert_close(output_vertices[4], vertices.new_zeros(3))
        self.assertEqual(len(faces), 4)
        self.assertEqual(labels.tolist(), [7, 7, 7, 7])
        self.assertEqual(sorted(sources.tolist()), [17, 17, 42, 42])
        edges = surface_sample._unique_edges_torch(faces)
        lengths = torch.linalg.norm(output_vertices[edges[:, 1]] - output_vertices[edges[:, 0]], dim=1)
        self.assertLessEqual(float(lengths.max()), 1.1)
        for boundary in self.boundary:
            self.assertTrue(bool((edges == boundary).all(dim=1).any()))
        self.assertTrue(bool(self.projector.valid_triangles(
            output_vertices[faces], labels, .01, 5.,
        ).all()))
        parents = vertices[self.faces]
        parent_cross = torch.cross(parents[:, 1] - parents[:, 0], parents[:, 2] - parents[:, 0], dim=1)
        children = output_vertices[faces]
        child_cross = torch.cross(children[:, 1] - children[:, 0], children[:, 2] - children[:, 0], dim=1)
        parent_ids = torch.tensor([{17: 0, 42: 1}[int(source)] for source in sources])
        self.assertTrue(bool(((child_cross * parent_cross[parent_ids]).sum(dim=1) > 0).all()))

    def test_refinement_reaches_target_when_parent_normals_exceed_half_limit(self):
        vertices = self.tilted_vertices(3.)
        children = self.chord_children(vertices)
        child_labels = self.labels.repeat_interleave(2)
        self.assertTrue(bool(self.projector.valid_triangles(vertices[self.faces], self.labels, .01, 5.).all()))
        self.assertTrue(bool(self.projector.valid_triangles(children, child_labels, .01, 5.).all()))
        self.assertFalse(bool(self.projector.valid_triangles(children, child_labels, .01, 2.5).any()))

        diagnostics = {}
        self.assert_valid_refinement(self.split(vertices, diagnostics=diagnostics), vertices)
        self.assertEqual(diagnostics["stop_reason"], "converged")
        self.assertEqual(diagnostics["remaining_long_edges"], 0)
        self.assertEqual(diagnostics["rejected_long_edges"], 0)

    def test_refinement_rejects_normals_beyond_requested_limit(self):
        for angles in ((6., 6.), (3., 6.), (6., 3.)):
            with self.subTest(angles=angles):
                vertices = self.tilted_vertices(*angles)
                children = self.chord_children(vertices)
                validity = self.projector.valid_triangles(children, self.labels.repeat_interleave(2), .01, 5.)
                self.assertEqual(validity.tolist(), [angle < 5. for angle in angles for _ in range(2)])
                diagnostics = {}
                self.assert_unchanged(self.split(vertices, diagnostics=diagnostics), vertices)
                self.assertEqual(diagnostics["stop_reason"], "geometric_rejections")
                self.assertEqual(diagnostics["rejected_long_edges"], 1)

    def test_explicit_candidate_normal_cap_remains_binding(self):
        vertices = self.tilted_vertices(3.)
        diagnostics = {}
        self.assert_unchanged(self.split(
            vertices, candidate_maximum_normal_deviation_degrees=2.5,
            diagnostics=diagnostics,
        ), vertices)
        self.assertEqual(diagnostics["stop_reason"], "geometric_rejections")
        self.assertEqual(diagnostics["rejected_long_edges"], 1)

    def test_candidate_normal_cap_cannot_override_global_limit(self):
        for candidate_limit in (5., 10.):
            with self.subTest(candidate_limit=candidate_limit):
                vertices = self.tilted_vertices(3.)
                self.assert_valid_refinement(self.split(
                    vertices, candidate_maximum_normal_deviation_degrees=candidate_limit,
                ), vertices)
                invalid_vertices = self.tilted_vertices(6.)
                self.assert_unchanged(self.split(
                    invalid_vertices, candidate_maximum_normal_deviation_degrees=candidate_limit,
                ), invalid_vertices)

    def test_surface_limit_remains_required_for_both_adjacent_faces(self):
        reference_vertices = self.reference_vertices.clone()
        reference_vertices[[1, 3], 1] *= 1000.
        projector = surface_sample._ReferencePatchProjector(
            reference_vertices.numpy(), self.faces.numpy(), self.labels.numpy(),
        )
        vertices = reference_vertices.clone()
        vertices[3, 2] = .003
        children = self.chord_children(vertices)
        child_labels = self.labels.repeat_interleave(2)
        self.assertTrue(bool(projector.valid_triangles(children, child_labels, None, 5.).all()))
        self.assertEqual(projector.valid_triangles(children, child_labels, 1e-4, 5.).tolist(),
                         [True, True, False, False])
        diagnostics = {}
        self.assert_unchanged(self.split(
            vertices, reference_patch_projector=projector,
            maximum_surface_deviation=1e-4, diagnostics=diagnostics,
        ), vertices)
        self.assertEqual(diagnostics["stop_reason"], "geometric_rejections")

    def test_inverted_projection_uses_orientation_preserving_chord(self):
        vertices = self.tilted_vertices(3.)

        def outside_edge(points, labels):
            return points + points.new_tensor([2., 0., 0.]), torch.zeros_like(labels)

        diagnostics = {}
        with mock.patch.object(self.projector, "project", side_effect=outside_edge):
            result = self.split(vertices, diagnostics=diagnostics)
        self.assert_valid_refinement(result, vertices)
        self.assertEqual(diagnostics["fallback_splits"], 1)

    def public_remesh(self, candidate_override=None):
        import trimesh

        vertices = self.tilted_vertices(3.).numpy()
        faces = self.faces.numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        with mock.patch.dict(os.environ):
            os.environ.pop("PAMO_CANDIDATE_MAX_NORMAL_DEVIATION_DEGREES", None)
            if candidate_override is not None:
                os.environ["PAMO_CANDIDATE_MAX_NORMAL_DEVIATION_DEGREES"] = str(candidate_override)
            result = surface_sample.surface_sample_remesh(
                mesh, torch.as_tensor(vertices, device="cuda"), torch.as_tensor(faces, device="cuda"),
                sample_count=1, poisson_radius=.55, collapse_passes=0,
                flip_passes=0, relax_iterations=0, split_passes=8,
                maximum_normal_deviation_degrees=5.,
                maximum_surface_deviation_ratio=.01 / .55,
                external_face_patch_ids=self.labels.numpy(),
                fixed_constraint_edges=self.boundary.numpy(),
                fixed_corner_vertex_ids=np.arange(4), whole_patch_optimization=True,
                whole_patch_reference=(self.reference_vertices.numpy(), faces, self.labels.numpy()),
            )
        return vertices, result

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_public_cuda_remesh_does_not_turn_default_headroom_into_split_cap(self):
        vertices, (output_vertices, output_faces, statistics) = self.public_remesh()
        np.testing.assert_array_equal(output_vertices[:4], vertices)
        np.testing.assert_array_equal(statistics["face_patch_ids"], np.full(len(output_faces), 7))
        self.assertEqual(statistics["splits"], 1)
        self.assertEqual(statistics["remaining_long_edges"], 0)
        self.assertEqual(statistics["split_diagnostics"]["stop_reason"], "converged")
        self.assertLessEqual(statistics["split_diagnostics"]["maximum_edge_length"], 1.1)
        self.assertEqual(statistics["fixed_constraint_edge_count"], 4)
        self.assertEqual(statistics["fixed_corner_count"], 4)
        self.assertTrue(bool(self.projector.valid_triangles(
            torch.as_tensor(output_vertices[output_faces]),
            torch.as_tensor(statistics["face_patch_ids"]), .01, 5.,
        ).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_public_cuda_remesh_respects_explicit_candidate_normal_override(self):
        vertices, (output_vertices, output_faces, statistics) = self.public_remesh(candidate_override=2.5)
        np.testing.assert_array_equal(output_vertices, vertices)
        np.testing.assert_array_equal(output_faces, self.faces.numpy())
        self.assertEqual(statistics["splits"], 0)
        self.assertEqual(statistics["remaining_long_edges"], 1)
        self.assertEqual(statistics["split_diagnostics"]["stop_reason"], "geometric_rejections")
        self.assertEqual(statistics["split_diagnostics"]["rejected_long_edges"], 1)
        self.assertEqual(statistics["split_diagnostics"]["maximum_edge_length"], 2.)


if __name__ == "__main__":
    unittest.main()
