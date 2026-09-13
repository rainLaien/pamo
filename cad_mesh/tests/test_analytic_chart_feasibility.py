"""Impossible fixed cylinder trims stop before repeated chart refinement."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cad_mesh import analytic_remesh
from cad_mesh.tests.test_analytic_remesh import boundary_edges, ruled_surface


class AnalyticChartFeasibilityTest(unittest.TestCase):
    def check(self, uv, limit=1., aliases=None, fixed=None):
        uv = np.asarray(uv, dtype=np.float64)
        loop = np.arange(len(uv))
        segments = np.column_stack((loop, np.roll(loop, -1)))
        aliases = loop if aliases is None else np.asarray(aliases)
        fixed = np.sort(aliases[segments], axis=1) if fixed is None else np.asarray(fixed)
        chart = {"uv": uv, "aliases": aliases, "circumferential_edge_length": limit}
        return analytic_remesh._validate_fixed_chart_edge_feasibility(chart, segments, fixed)

    def test_wide_fixed_rectangle_cannot_meet_internal_span_limit(self):
        with self.assertRaisesRegex(ValueError, "fixed_boundary_prevents_chart_edge_limits"):
            self.check([[0., 0.], [3., 0.], [3., 1.], [0., 1.]])

    def test_third_vertex_with_another_fixed_side_remains_permitted(self):
        # The ear (0, 1, 2) can retain its two long fixed sides. Its only
        # movable side, (0, 2), has angular span .5, below the limit of 1.
        self.check([[0., 0.], [3., 0.], [.5, 1.], [0., 1.]])

    def test_entire_fixed_triangle_is_not_rejected_for_long_spans(self):
        self.check([[0., 0.], [3., 0.], [.5, 1.]])

    def test_span_exactly_twice_limit_can_use_an_interior_midpoint(self):
        self.check([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])

    def test_parameter_seam_is_not_mistaken_for_an_immutable_trim(self):
        self.check([[0., 0.], [3., 0.], [3., 1.], [0., 1.]],
                   aliases=[-1, -2, -3, -4], fixed=np.empty((0, 2), dtype=np.int64))

    def test_non_cylindrical_charts_do_not_require_an_angular_cap(self):
        analytic_remesh._validate_fixed_chart_edge_feasibility({}, None, None)

    def test_impossible_cylinder_returns_to_surface_remesh_before_triangle(self):
        vertices, faces, patch = ruled_surface(count=24, partial=True)
        original_vertices, original_faces = vertices.copy(), faces.copy()
        with mock.patch.object(analytic_remesh, "_run_triangle", side_effect=AssertionError("Unnecessary triangulation")):
            output_vertices, output_faces, diagnostics = analytic_remesh.remesh_analytic_patch(
                vertices, faces, patch, boundary_edges(faces), 12., .1,
                maximum_normal_deviation_degrees=5.)
        self.assertIsNone(output_vertices)
        self.assertIsNone(output_faces)
        self.assertFalse(diagnostics["accepted"])
        self.assertEqual(diagnostics["reason"], "fixed_boundary_prevents_chart_edge_limits")
        np.testing.assert_array_equal(vertices, original_vertices)
        np.testing.assert_array_equal(faces, original_faces)


if __name__ == "__main__":
    unittest.main()
