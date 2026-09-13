"""Actual 3D shape governs both chart acceptance and the reported target."""
import unittest
from unittest import mock

import numpy as np

from cad_mesh import analytic_remesh, surface_rebuild
from cad_mesh.remesh_pipeline import _pamo_module


class SurfaceQualityPolicyTests(unittest.TestCase):
    def test_directional_score_cannot_accept_a_worse_3d_chart(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        edges = np.array([[0, 1], [0, 3], [1, 2], [2, 3]])
        labels = np.zeros(2, dtype=np.int64)
        candidate_vertices = np.vstack((vertices, [.001, .001, 0.]))
        candidate_faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        # This is the earlier failure: a directional score improves while
        # ordinary triangles become substantially thinner.
        for collect in (True, False):
            diagnostic = {"accepted": True, "reason": "accepted", "anisotropic_sizing": True,
                          "source_mean_metric_quality": .2, "output_mean_metric_quality": .99}
            with self.subTest(collect_diagnostics=collect), mock.patch.object(
                    analytic_remesh, "remesh_analytic_patch",
                    return_value=(candidate_vertices, candidate_faces, diagnostic)):
                result = surface_rebuild._analytic_stage(
                    vertices, faces, labels, edges, [{"type": "Cylinder"}],
                    2., .01, _pamo_module("feature_optimize"), collect_diagnostics=collect)
            self.assertEqual(result[3], set())
            np.testing.assert_array_equal(result[0], vertices)
            np.testing.assert_array_equal(result[1], faces)
            self.assertEqual(result[4][0]["reason"], "chart_reduces_mean_quality")
            self.assertEqual(result[4][0]["quality_comparison_metric"], "euclidean")

    def test_mean_target_is_distinct_from_every_face_meeting_target(self):
        # One equilateral and one right triangle: the mean exceeds .8 even
        # though per-face thresholds and the minimum are separate quantities.
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [.5, np.sqrt(3.)/2., 0.],
                             [3., 0., 0.], [5., 0., 0.], [3., 1., 0.]])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        result = surface_rebuild._quality_assessment(
            vertices, faces, np.array([0, 1]), [{"type": "Cylinder"}, {"type": "Plane"}])
        self.assertTrue(result["target_met"])
        self.assertAlmostEqual(result["fraction_below_target"], .5)
        self.assertLess(result["minimum_triangle_quality"], .8)
        self.assertAlmostEqual(result["per_surface_type"]["Cylinder"]["mean_triangle_quality"], 1.)

    def test_slender_triangles_report_target_unmet(self):
        vertices = np.array([[0., 0., 0.], [10., 0., 0.], [.1, .01, 0.]])
        result = surface_rebuild._quality_assessment(
            vertices, np.array([[0, 1, 2]]), np.array([0]), [{"type": "Cylinder"}])
        self.assertFalse(result["target_met"])
        self.assertEqual(result["fraction_below_0_2"], 1.)


if __name__ == "__main__":
    unittest.main()
