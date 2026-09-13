"""Whole-patch analytic charts retain holes, fixed boundaries and periodic seams."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cad_mesh.analytic_remesh import remesh_analytic_patch
from cad_mesh.remesh_pipeline import _pamo_module


def boundary_edges(faces):
    edges, counts = np.unique(
        np.sort(faces[:, ((0, 1), (1, 2), (2, 0))], axis=2).reshape(-1, 2),
        axis=0, return_counts=True,
    )
    return edges[counts == 1]


def edge_incidence(faces):
    result = {}
    for triangle in faces:
        for a, b in zip(triangle, np.roll(triangle, -1)):
            result.setdefault(tuple(sorted((int(a), int(b)))), []).append((int(a), int(b)))
    return result


def annulus():
    vertices, faces, ids = [], [], {}

    def index(x, y):
        if (x, y) not in ids:
            ids[x, y] = len(vertices)
            vertices.append((x, y, 0.0))
        return ids[x, y]

    for y in range(4):
        for x in range(4):
            if 1 <= x < 3 and 1 <= y < 3:
                continue
            a, b, c, d = index(x, y), index(x + 1, y), index(x + 1, y + 1), index(x, y + 1)
            faces.extend(((a, b, c), (a, c, d)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def ruled_surface(kind="Cylinder", count=48, upper_phase=0.0, partial=False,
                  axial_rows=2):
    span = np.pi * .75 if partial else 2 * np.pi
    angles = np.linspace(0.0, span, count, endpoint=partial)
    ends = (0.0, 4.0) if kind == "Cylinder" else (2.0, 6.0)
    heights = np.linspace(*ends, axial_rows)
    vertices = []
    for row, height in enumerate(heights):
        radius = 2.0 if kind == "Cylinder" else height * .5
        for angle in angles + row / (axial_rows - 1) * upper_phase:
            vertices.append((radius * np.cos(angle), radius * np.sin(angle), height))
    faces = []
    for row in range(axial_rows - 1):
        lower, upper = row * count, (row + 1) * count
        for column in range(count - 1 if partial else count):
            following = (column + 1) % count
            faces.extend(((lower + column, lower + following, upper + following),
                          (lower + column, upper + following, upper + column)))
    parameters = {"axis_origin": [0.0, 0.0, 0.0], "axis_direction": [0.0, 0.0, 1.0]}
    if kind == "Cylinder":
        parameters["radius"] = 2.0
    else:
        parameters["semi_angle_radians"] = float(np.arctan(.5))
    return (np.asarray(vertices), np.asarray(faces, dtype=np.int64),
            {"type": kind, "parameters": parameters})


PLANE = {"type": "Plane", "parameters": {"origin": [0, 0, 0], "normal": [0, 0, 1]}}


def prepare_shared_boundaries(vertices, faces, length):
    # Use the same global constrained prepass as the integration. The chart
    # itself must never add independent samples to those finalized boundaries.
    result = _pamo_module("original_constrained")._subdivide_labeled_patch_boundaries(
        vertices, faces, np.zeros(len(faces), dtype=np.int64), length * .75)
    return result[0], result[1]


class AnalyticRemeshTests(unittest.TestCase):
    def test_cylinder_holes_touch_at_one_shared_vertex(self):
        vertices, faces, patch = ruled_surface(count=48, axial_rows=9)
        cell = np.arange(len(faces)) // 2
        row, column = cell // 48, cell % 48
        first = (row >= 2) & (row < 4) & (column >= 10) & (column < 12)
        second = (row >= 4) & (row < 6) & (column >= 12) & (column < 14)
        faces = faces[~(first | second)]
        out_v, out_f, stats = self.check_remesh(vertices, faces, patch, .6, .04)
        self.assertIn(4 * 48 + 12, out_f)
        self.assertTrue(stats['periodic_seam'])
        self.assertEqual(stats['boundary_loops'], 4)

    def test_adjacency_unwrap_crosses_branch_cut_without_changing_geometry(self):
        from cad_mesh.analytic_remesh import _adjacent_angles
        theta = np.radians([175., -175., -165., 170.])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        lifted = _adjacent_angles(theta, faces)
        np.testing.assert_allclose(np.cos(lifted), np.cos(theta), atol=1e-14)
        self.assertLess(np.ptp(lifted), np.pi)

    def test_periodic_cone_with_irregular_trims_and_hole(self):
        vertices, faces, patch = ruled_surface(kind='Cone', count=48, axial_rows=9)
        angles = np.arctan2(vertices[:, 1], vertices[:, 0])
        height = vertices[:, 2] + .15 * np.sin(3 * angles)
        vertices[:, :2] *= (height / vertices[:, 2])[:, None]
        vertices[:, 2] = height
        cell = np.arange(len(faces)) // 2
        row, column = cell // 48, cell % 48
        faces = faces[~((row >= 3) & (row < 5) & (column >= 10) & (column < 14))]
        _, _, stats = self.check_remesh(vertices, faces, patch, .6, .04)
        self.assertTrue(stats['trimmed_cone_chart'])
        self.assertEqual(stats['boundary_loops'], 3)
        self.assertEqual(stats['holes'], 1)

    def test_refit_compares_cylinder_and_cone_on_entire_region(self):
        from cad_mesh.analytic_refit import reassess_analytic_patch
        vertices, faces, patch = ruled_surface(count=48, axial_rows=6)
        wrong = dict(patch, type='Cone', parameters={
            'axis_origin': [0,0,-10], 'axis_direction': [0,0,1], 'semi_angle_radians': .2})
        candidates, diagnostic = reassess_analytic_patch(vertices, faces, boundary_edges(faces), wrong, .04)
        self.assertTrue(candidates, diagnostic)
        self.assertEqual(candidates[0]['type'], 'Cylinder')
        # A bad boundary cannot be hidden by fitting only the interior.
        vertices[0] *= 1.6
        candidates, diagnostic = reassess_analytic_patch(vertices, faces, boundary_edges(faces), wrong, .04)
        self.assertFalse(candidates, diagnostic)
        self.assertTrue(diagnostic['candidates'][-1]['accepted'])

    def test_periodic_cylinder_with_wavy_trims_and_side_hole(self):
        vertices, faces, patch = ruled_surface(count=48, axial_rows=9)
        angles = np.arctan2(vertices[:, 1], vertices[:, 0])
        vertices[:, 2] += .2 * np.sin(3 * angles)
        # Remove an interior rectangular window, away from the angular seam.
        cell = np.arange(len(faces)) // 2
        row, column = cell // 48, cell % 48
        faces = faces[~((row >= 3) & (row < 5) & (column >= 10) & (column < 14))]
        _, _, stats = self.check_remesh(vertices, faces, patch, .6, .04)
        self.assertTrue(stats['periodic_seam'])
        self.assertEqual(stats['boundary_loops'], 3)
        self.assertEqual(stats['holes'], 1)

    def test_periodic_cylinder_window_crossing_auxiliary_seam(self):
        vertices, faces, patch = ruled_surface(count=48, axial_rows=9)
        cell = np.arange(len(faces)) // 2
        row, column = cell // 48, cell % 48
        faces = faces[~((row >= 3) & (row < 5) & ((column >= 46) | (column < 2)))]
        _, _, stats = self.check_remesh(vertices, faces, patch, .6, .04)
        self.assertTrue(stats['periodic_seam'])
        self.assertEqual(stats['boundary_loops'], 3)

    def check_remesh(self, vertices, faces, patch, length, deviation):
        original_vertices, original_faces = vertices.copy(), faces.copy()
        prepared_vertices, prepared_faces = prepare_shared_boundaries(vertices, faces, length)
        edges = boundary_edges(prepared_faces)
        out_v, out_f, stats = remesh_analytic_patch(
            prepared_vertices, prepared_faces, patch, edges, length, deviation,
            maximum_normal_deviation_degrees=180.0)
        self.assertTrue(stats["accepted"], stats)
        self.assertIsNotNone(out_v)
        self.assertIsNotNone(out_f)
        np.testing.assert_array_equal(vertices, original_vertices)
        np.testing.assert_array_equal(faces, original_faces)
        np.testing.assert_array_equal(out_v[:len(prepared_vertices)], prepared_vertices)
        self.assertEqual(set(map(tuple, boundary_edges(out_f))), set(map(tuple, edges)))
        for incidence in edge_incidence(out_f).values():
            self.assertIn(len(incidence), (1, 2))
            if len(incidence) == 2:
                self.assertEqual(incidence[0], incidence[1][::-1])
        output_edges = np.array(list(edge_incidence(out_f)))
        self.assertLessEqual(float(np.linalg.norm(
            out_v[output_edges[:, 1]] - out_v[output_edges[:, 0]], axis=1).max()),
            length * (1.0 + 1e-6))
        self.assertFalse(stats["source_interior_triangulation_reused"])
        self.assertFalse(stats["hausdorff_upper_bound"])
        return out_v, out_f, stats

    def test_plane_replaces_the_whole_interior_and_keeps_boundary(self):
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.],
                             [0., 2., 0.], [.03, .04, 0.]])
        faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int64)
        out_v, out_f, stats = self.check_remesh(vertices, faces, PLANE, .6, 1e-9)
        self.assertNotIn(4, out_f)
        self.assertGreater(len(out_v), len(vertices))
        self.assertEqual(stats["holes"], 0)
        self.assertFalse(stats["periodic_seam"])
        triangle_area = np.cross(out_v[out_f[:, 1]] - out_v[out_f[:, 0]],
                                  out_v[out_f[:, 2]] - out_v[out_f[:, 0]])[:, 2] * .5
        self.assertAlmostEqual(float(triangle_area.sum()), 4.0)

    def test_plane_hole_remains_empty_after_global_triangulation(self):
        vertices, faces = annulus()
        out_v, out_f, stats = self.check_remesh(vertices, faces, PLANE, .7, 1e-9)
        centers = out_v[out_f].mean(axis=1)
        inside_hole = np.all((centers[:, :2] > 1.0) & (centers[:, :2] < 3.0), axis=1)
        self.assertFalse(np.any(inside_hole))
        self.assertEqual(stats["holes"], 1)
        self.assertEqual(stats["boundary_loops"], 2)
        triangle_area = np.cross(out_v[out_f[:, 1]] - out_v[out_f[:, 0]],
                                  out_v[out_f[:, 2]] - out_v[out_f[:, 0]])[:, 2] * .5
        self.assertAlmostEqual(float(triangle_area.sum()), 12.0)

    def test_complete_cylinder_has_one_stitched_periodic_seam(self):
        vertices, faces, patch = ruled_surface()
        out_v, out_f, stats = self.check_remesh(vertices, faces, patch, .6, .04)
        self.assertTrue(stats["periodic_seam"])
        self.assertGreater(stats["seam_vertex_count"], 0)
        self.assertEqual(stats["boundary_loops"], 2)
        self.assertTrue(np.allclose(np.linalg.norm(out_v[len(vertices):, :2], axis=1), 2.0))
        self.assertEqual(len(np.unique(out_f)) - len(edge_incidence(out_f)) + len(out_f), 0)

    def test_cylinder_seam_supports_different_end_ring_phases(self):
        vertices, faces, patch = ruled_surface(count=32, upper_phase=.04)
        _, _, stats = self.check_remesh(vertices, faces, patch, .7, .07)
        self.assertTrue(stats["periodic_seam"])

    def test_complete_conical_frustum_stitches_developed_sector(self):
        vertices, faces, patch = ruled_surface("Cone")
        out_v, out_f, stats = self.check_remesh(vertices, faces, patch, .65, .04)
        self.assertTrue(stats["periodic_seam"])
        self.assertGreater(stats["seam_vertex_count"], 0)
        new = out_v[len(vertices):]
        np.testing.assert_allclose(np.linalg.norm(new[:, :2], axis=1), new[:, 2] * .5, atol=1e-12)
        self.assertEqual(len(np.unique(out_f)) - len(edge_incidence(out_f)) + len(out_f), 0)

    def test_partial_cylinder_and_cone_use_uncut_whole_charts(self):
        for kind in ("Cylinder", "Cone"):
            for rows in (2, 9):
                with self.subTest(kind=kind, axial_rows=rows):
                    # Both a long immutable generator and a globally sampled
                    # shared generator work without independent boundary edits.
                    vertices, faces, patch = ruled_surface(
                        kind, count=24, partial=True, axial_rows=rows)
                    _, _, stats = self.check_remesh(vertices, faces, patch, .6, .04)
                    self.assertFalse(stats["periodic_seam"])
                    self.assertEqual(stats["boundary_loops"], 1)

    def test_periodic_chart_preserves_reversed_source_winding(self):
        vertices, faces, patch = ruled_surface()
        faces = faces[:, [0, 2, 1]]
        out_v, out_f, _ = self.check_remesh(vertices, faces, patch, .7, .05)
        triangles = out_v[out_f]
        crosses = np.cross(triangles[:, 1] - triangles[:, 0],
                            triangles[:, 2] - triangles[:, 0])
        centers = triangles.mean(axis=1)
        self.assertTrue(np.all(np.sum(crosses[:, :2] * centers[:, :2], axis=1) < 0.0))

    def test_normalized_chart_handles_small_units_and_translation(self):
        vertices, faces = annulus()
        for scale in (1e-8, 1000.0):
            with self.subTest(scale=scale):
                shift = np.array([123.0, -45.0, 67.0]) * scale
                points = vertices * scale + shift
                patch = {"type": "Plane", "parameters": {
                    "origin": shift.tolist(), "normal": [0.0, 0.0, 1.0]}}
                _, _, stats = self.check_remesh(points, faces, patch, .7 * scale, 1e-9 * scale)
                self.assertEqual(stats["holes"], 1)

    def test_source_deviation_rejection_is_explicit_and_does_not_mutate(self):
        vertices, faces, patch = ruled_surface(count=12)
        vertices, faces = prepare_shared_boundaries(vertices, faces, .6)
        copy = vertices.copy()
        out_v, out_f, stats = remesh_analytic_patch(
            vertices, faces, patch, boundary_edges(faces), .6, 1e-6)
        self.assertIsNone(out_v)
        self.assertIsNone(out_f)
        self.assertFalse(stats["accepted"])
        self.assertEqual(stats["reason"], "source_mesh_exceeds_analytic_model_deviation")
        np.testing.assert_array_equal(vertices, copy)

    def test_missing_boundary_or_unsupported_model_returns_to_fallback(self):
        vertices, faces = annulus()
        for patch, edges, reason in (
            (PLANE, boundary_edges(faces)[:-1], "constraints_do_not_equal_patch_boundary_loops"),
            ({"type": "Sphere", "parameters": {}}, boundary_edges(faces), "unsupported_surface_type"),
        ):
            with self.subTest(reason=reason):
                out_v, out_f, stats = remesh_analytic_patch(vertices, faces, patch, edges, 2.0, .01)
                self.assertIsNone(out_v)
                self.assertIsNone(out_f)
                self.assertEqual(stats["reason"], reason)

    def test_unsampled_long_shared_boundary_is_rejected(self):
        vertices, faces, patch = ruled_surface(partial=True, count=24)
        out_v, out_f, stats = remesh_analytic_patch(
            vertices, faces, patch, boundary_edges(faces), .6, .04)
        self.assertIsNone(out_v)
        self.assertIsNone(out_f)
        self.assertEqual(stats["reason"], "fixed_boundary_exceeds_target_edge_length")

    def test_small_planar_targets_certify_edges_after_chart_refinement(self):
        rectangle = np.array([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]])
        rectangle_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        for vertices, faces, target in ((rectangle, rectangle_faces, .35), (*annulus(), .45)):
            with self.subTest(target=target):
                _, _, stats = self.check_remesh(vertices, faces, PLANE, target, 1e-9)
                self.assertLessEqual(stats["maximum_edge_length"], target * (1.0 + 1e-6))
                self.assertGreaterEqual(stats["interior_refinement_passes"], 0)

    def test_repeated_plane_charts_keep_the_triangulator_alive(self):
        vertices, faces = annulus()
        for _ in range(8):
            _, _, stats = self.check_remesh(vertices, faces, PLANE, .7, 1e-9)
            self.assertTrue(stats["accepted"], stats)


if __name__ == "__main__":
    unittest.main()
