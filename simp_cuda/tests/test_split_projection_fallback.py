"""Long-edge splits retain reference-safe alternatives to analytic proposals."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simp_cuda.tests.test_surface_sample import surface_sample


@unittest.skipIf(surface_sample is None, "Surface sampling dependencies are unavailable")
class SplitProjectionFallbackTest(unittest.TestCase):
    def setUp(self):
        # Only the interior diagonal exceeds the target. Thin CAD triangles
        # amplify the normal error from projecting a midpoint onto a fitted
        # surface that differs slightly from the immutable triangulation.
        self.vertices = torch.tensor(
            [[-1., 0., 0.], [0., 1e-4, 0.], [1., 0., 0.], [0., -1e-4, 0.]],
            dtype=torch.float64,
        )
        self.faces = torch.tensor([[0, 2, 1], [2, 0, 3]], dtype=torch.long)
        self.labels = torch.tensor([7, 7])
        self.sources = torch.tensor([17, 42])
        self.support = torch.full((4,), -2, dtype=torch.long)
        self.boundary = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]])
        self.projector = self.reference()

    def reference(self, vertices=None, faces=None, labels=None):
        return surface_sample._ReferencePatchProjector(
            self.vertices.numpy() if vertices is None else vertices,
            self.faces.numpy() if faces is None else faces,
            self.labels.numpy() if labels is None else labels,
        )

    def sphere(self, center=(0., 0., -1.), radius=1.1):
        return surface_sample._AnalyticSplitProjector(
            {7: {"type": "Sphere", "parameters": {"center": center, "radius": radius}}},
            torch.device("cpu"), 1e-12,
        )

    def test_batch_analytic_models_keep_active_projection_exact(self):
        models = {
            7: {"type": "Sphere", "parameters": {"center": [0., 0., -1.], "radius": 1.1}},
            12: {"type": "Cylinder", "parameters": {
                "axis_origin": [0., 0., 0.], "axis_direction": [0., 0., 1.], "radius": 2.}},
        }
        complete = surface_sample._AnalyticSplitProjector(models, "cpu", 1e-12)
        active = surface_sample._AnalyticSplitProjector(models, "cpu", 1e-12, active_patch_ids=[7, 15])
        points = torch.tensor([[1., 0., 0.], [2., 3., 4.]], dtype=torch.float64)
        labels = torch.tensor([7, 15])
        for before, after in zip(complete.project(points, labels), active.project(points, labels)):
            self.assertTrue(torch.equal(before, after))
        freeform = surface_sample._AnalyticSplitProjector(models, "cpu", 1e-12, active_patch_ids=[15])
        self.assertEqual(len(freeform.labels), 0)
        projected, curved, stable = freeform.project(points, torch.tensor([15, 15]))
        self.assertTrue(torch.equal(projected, points))
        self.assertFalse(bool(curved.any()))
        self.assertTrue(bool(stable.all()))

    def split(self, **overrides):
        arguments = dict(
            vertices=self.vertices, faces=self.faces, face_patch_ids=self.labels,
            face_source_ids=self.sources, support_face_ids=self.support,
            maximum_edge_length=1.1, maximum_passes=8,
            protected_edges=self.boundary, reference_patch_projector=self.projector,
            maximum_surface_deviation=1e-8, maximum_normal_deviation_degrees=5.,
            analytic_split_projector=self.sphere(),
        )
        arguments.update(overrides)
        return surface_sample._gpu_split_long_edges(**arguments)

    def assert_one_valid_split(self, result, midpoint=(0., 0., 0.)):
        vertices, faces, labels, sources, support, splits, remaining = result
        self.assertEqual((splits, remaining), (1, 0))
        self.assertTrue(torch.equal(vertices[:4], self.vertices))
        self.assertTrue(torch.equal(support[:4], self.support))
        torch.testing.assert_close(vertices[4], torch.tensor(midpoint, dtype=vertices.dtype))
        self.assertEqual(len(faces), 4)
        self.assertEqual(labels.tolist(), [7, 7, 7, 7])
        self.assertEqual(sorted(sources.tolist()), [17, 17, 42, 42])
        edges = surface_sample._unique_edges_torch(faces)
        lengths = torch.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], dim=1)
        self.assertLessEqual(float(lengths.max()), 1.1)
        self.assertTrue(bool(self.projector.valid_triangles(vertices[faces], labels, 1e-8, 2.5).all()))
        for boundary in self.boundary:
            self.assertTrue(bool((edges == boundary).all(dim=1).any()))

    def test_rejected_analytic_fit_falls_back_to_reference_midpoint(self):
        midpoint = torch.zeros((1, 3), dtype=torch.float64)
        analytic, curved, stable = self.sphere().project(midpoint, torch.tensor([7]))
        self.assertTrue(bool(curved[0] & stable[0]))
        self.assertGreater(float(torch.linalg.norm(analytic - midpoint)), 1e-8)
        diagnostics = {}
        with mock.patch.object(self.projector, "project", wraps=self.projector.project) as project:
            result = self.split(diagnostics=diagnostics)
        self.assert_one_valid_split(result)
        self.assertGreater(project.call_count, 0)
        torch.testing.assert_close(project.call_args.args[0], midpoint)
        self.assertEqual(diagnostics["stop_reason"], "converged")
        self.assertEqual(diagnostics["fallback_splits"], 1)
        self.assertEqual(diagnostics["remaining_long_edges"], 0)
        self.assertEqual(diagnostics["rejected_long_edges"], 0)

    def test_singular_analytic_projection_falls_back_to_reference(self):
        sphere = self.sphere(center=(0., 0., 0.), radius=1.)
        _, curved, stable = sphere.project(torch.zeros((1, 3), dtype=torch.float64), torch.tensor([7]))
        self.assertTrue(bool(curved[0]))
        self.assertFalse(bool(stable[0]))
        self.assert_one_valid_split(self.split(analytic_split_projector=sphere))

    def test_rejected_reference_projection_retries_original_chord(self):
        # Exercise the final alternative independently of closest-point query
        # roundoff: both projected points fail the real geometric validator.
        def displaced_reference(points, labels):
            return points + points.new_tensor([0., 0., .2]), torch.zeros_like(labels)

        with mock.patch.object(self.projector, "project", side_effect=displaced_reference):
            result = self.split()
        self.assert_one_valid_split(result)

    def test_valid_analytic_proposal_remains_preferred(self):
        sphere = self.sphere(center=(-.95, 0., 0.), radius=1.)
        with mock.patch.object(self.projector, "project", wraps=self.projector.project) as project:
            result = self.split(analytic_split_projector=sphere)
        self.assert_one_valid_split(result, midpoint=(.05, 0., 0.))
        project.assert_not_called()

    def test_each_adjacent_face_must_satisfy_reference_limits(self):
        for violation in ("surface", "normal"):
            with self.subTest(violation=violation):
                # The first parent's children are valid in both cases. The
                # other parent's children either cross absent reference area
                # or have opposite orientation to their reference surface.
                reference_faces = (self.faces[:1].numpy() if violation == "surface"
                                   else np.array([[0, 2, 1], [0, 2, 3]]))
                projector = self.reference(faces=reference_faces,
                                           labels=np.full(len(reference_faces), 7))
                midpoint = self.vertices.new_zeros(3)
                children = torch.stack((
                    torch.stack((self.vertices[0], midpoint, self.vertices[1])),
                    torch.stack((self.vertices[2], midpoint, self.vertices[3])),
                ))
                self.assertEqual(projector.valid_triangles(
                    children, self.labels, 1e-8, 2.5).tolist(), [True, False])
                diagnostics = {}
                result = self.split(reference_patch_projector=projector, diagnostics=diagnostics)
                self.assertEqual(result[-2:], (0, 1))
                self.assertEqual(diagnostics["stop_reason"], "geometric_rejections")
                self.assertEqual(diagnostics["rejected_long_edges"], 1)
                for actual, expected in zip(result[:5], (
                    self.vertices, self.faces, self.labels, self.sources, self.support,
                )):
                    self.assertTrue(torch.equal(actual, expected))

    def test_fallback_preserves_shared_interface_ids_and_source_ownership(self):
        vertices = torch.cat((self.vertices, self.vertices.new_tensor([[1., 2e-4, 0.]])))
        faces = torch.cat((self.faces, torch.tensor([[1, 2, 4]])))
        labels = torch.tensor([7, 7, 23])
        sources = torch.tensor([17, 42, 89])
        support = torch.full((5,), -2, dtype=torch.long)
        constraints = torch.cat((self.boundary, torch.tensor([[1, 4], [2, 4]])))
        projector = self.reference(vertices.numpy(), faces.numpy(), labels.numpy())
        output = self.split(
            vertices=vertices, faces=faces, face_patch_ids=labels, face_source_ids=sources,
            support_face_ids=support, protected_edges=constraints,
            reference_patch_projector=projector,
        )
        output_vertices, output_faces, output_labels, output_sources, output_support, splits, remaining = output
        self.assertEqual((splits, remaining), (1, 0))
        self.assertTrue(torch.equal(output_vertices[:5], vertices))
        self.assertTrue(torch.equal(output_support[:5], support))
        self.assertEqual(sorted(output_sources.tolist()), [17, 17, 42, 42, 89])
        for source, label in zip(output_sources.tolist(), output_labels.tolist()):
            self.assertEqual(label, {17: 7, 42: 7, 89: 23}[source])
        interface_faces = (output_faces == 1).any(dim=1) & (output_faces == 2).any(dim=1)
        self.assertEqual(sorted(output_labels[interface_faces].tolist()), [7, 23])
        edges = surface_sample._unique_edges_torch(output_faces)
        for constraint in constraints:
            self.assertTrue(bool((edges == constraint).all(dim=1).any()))
        self.assertTrue(bool(projector.valid_triangles(
            output_vertices[output_faces], output_labels, 1e-8, 2.5).all()))

    def test_protected_long_edge_reports_constraint_block(self):
        diagnostics = {}
        result = self.split(
            protected_edges=torch.cat((self.boundary, torch.tensor([[0, 2]]))),
            diagnostics=diagnostics,
        )
        self.assertEqual(result[-2:], (0, 1))
        self.assertTrue(torch.equal(result[0], self.vertices))
        self.assertTrue(torch.equal(result[1], self.faces))
        self.assertEqual(diagnostics["stop_reason"], "protected_constraints")
        self.assertEqual(diagnostics["attempted_passes"], 0)
        self.assertEqual(diagnostics["protected_long_edges"], 1)
        self.assertEqual(diagnostics["rejected_long_edges"], 0)
        self.assertEqual(diagnostics["maximum_edge_length"], 2.)

    def test_exhausted_split_budget_reports_pass_limit(self):
        diagnostics = {}
        result = self.split(
            maximum_edge_length=.4, maximum_passes=1, protected_edges=None,
            reference_patch_projector=None, analytic_split_projector=None,
            diagnostics=diagnostics,
        )
        self.assertEqual(result[-2], 1)
        self.assertGreater(result[-1], 0)
        self.assertEqual(diagnostics["stop_reason"], "pass_limit")
        self.assertEqual(diagnostics["attempted_passes"], 1)
        self.assertEqual(diagnostics["maximum_passes"], 1)
        self.assertEqual(diagnostics["remaining_long_edges"], result[-1])
        self.assertEqual(diagnostics["protected_long_edges"], 0)
        self.assertEqual(diagnostics["rejected_long_edges"], 0)

    def test_nonmanifold_long_edge_is_included_in_remaining_count(self):
        vertices = torch.cat((self.vertices, self.vertices.new_tensor([[0., 0., 1e-4]])))
        faces = torch.cat((self.faces, torch.tensor([[0, 2, 4]])))
        diagnostics = {}
        result = self.split(
            vertices=vertices, faces=faces, face_patch_ids=torch.tensor([7, 7, 7]),
            face_source_ids=torch.tensor([17, 42, 89]),
            support_face_ids=torch.full((5,), -2, dtype=torch.long),
            reference_patch_projector=None, analytic_split_projector=None,
            protected_edges=None, diagnostics=diagnostics,
        )
        self.assertEqual(result[-2:], (0, 1))
        self.assertTrue(torch.equal(result[0], vertices))
        self.assertTrue(torch.equal(result[1], faces))
        self.assertEqual(diagnostics["stop_reason"], "nonmanifold_edges")
        self.assertEqual(diagnostics["remaining_long_edges"], 1)
        self.assertEqual(diagnostics["nonmanifold_long_edges"], 1)
        self.assertEqual(diagnostics["attempted_passes"], 0)
        self.assertEqual(diagnostics["maximum_edge_length"], 2.)


if __name__ == "__main__":
    unittest.main()
