import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


PAMO_PATH = Path(__file__).parents[1] / "pamo"

try:
    import igl  # noqa: F401
    import scipy  # noqa: F401
    import trimesh
except ImportError:
    feature_optimize = None
else:
    package_name = "_pamo_feature_optimize_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PAMO_PATH)]
    sys.modules[package_name] = package

    for module_name in (
        "segment_query",
        "feature_edges",
        "feature_optimize",
        "original_constrained",
    ):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            PAMO_PATH / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)

    feature_optimize = sys.modules[
        f"{package_name}.feature_optimize"
    ]
    original_constrained = sys.modules[
        f"{package_name}.original_constrained"
    ]


@unittest.skipIf(
    feature_optimize is None,
    "Feature optimization dependencies are unavailable",
)
class FeatureOptimizeTest(unittest.TestCase):
    @staticmethod
    def _disturbed_cylinder_patch(center_radius):
        angles = (0.0, 0.2, 0.4)
        heights = (0.0, 1.0, 2.0)
        vertices = []
        for row, height in enumerate(heights):
            for column, angle in enumerate(angles):
                radius = center_radius if (row, column) == (1, 1) else 10.0
                vertices.append(
                    (radius * np.cos(angle), radius * np.sin(angle), height)
                )
        faces = []
        for row in range(2):
            for column in range(2):
                lower_left = row * 3 + column
                lower_right = lower_left + 1
                upper_left = lower_left + 3
                upper_right = upper_left + 1
                faces.extend(
                    (
                        (lower_left, lower_right, upper_left),
                        (lower_right, upper_right, upper_left),
                    )
                )
        model = {
            "axis": np.asarray((0.0, 0.0, 1.0)),
            "origin": np.asarray((0.0, 0.0, 0.0)),
            "radius": 10.0,
            "source_face_ids": np.asarray((0,), dtype=np.int64),
            "axial_min": 0.0,
            "axial_max": 2.0,
        }
        return np.asarray(vertices), np.asarray(faces), model

    @staticmethod
    def _irregular_trimmed_cylinder(radius, span_degrees, columns=31, rows=17):
        angles = np.linspace(
            -0.5 * np.deg2rad(span_degrees),
            0.5 * np.deg2rad(span_degrees),
            columns,
        )
        fractions = np.linspace(0.0, 1.0, rows)
        vertices = []
        for fraction in fractions:
            for angle in angles:
                lower = 1.2 * np.cos(2.0 * angle)
                upper = 10.0 - 0.8 * np.sin(1.5 * angle)
                height = lower + fraction * (upper - lower)
                vertices.append(
                    (radius * np.cos(angle), radius * np.sin(angle), height)
                )
        faces = []
        for row in range(rows - 1):
            for column in range(columns - 1):
                first = row * columns + column
                faces.extend(
                    (
                        (first, first + 1, first + columns),
                        (first + 1, first + columns + 1, first + columns),
                    )
                )
        boundary = []
        for column in range(columns - 1):
            boundary.extend(
                (
                    (column, column + 1),
                    ((rows - 1) * columns + column,
                     (rows - 1) * columns + column + 1),
                )
            )
        for row in range(rows - 1):
            boundary.extend(
                (
                    (row * columns, (row + 1) * columns),
                    (row * columns + columns - 1,
                     (row + 1) * columns + columns - 1),
                )
            )
        return (
            np.asarray(vertices),
            np.asarray(faces, dtype=np.int64),
            np.asarray(boundary, dtype=np.int64),
        )

    def test_analytic_cylinder_recovery_projects_and_relaxes_internal_edge(self):
        vertices, faces, model = self._disturbed_cylinder_patch(10.1)
        internal_edge = (1, 3)
        boundary_edge = (0, 1)

        recovered, protected, stats = (
            original_constrained.recover_analytic_cylinder_support(
                vertices,
                faces,
                np.asarray((internal_edge, boundary_edge)),
                (model,),
                maximum_projection_distance=0.2,
            )
        )

        self.assertAlmostEqual(np.linalg.norm(recovered[4, :2]), 10.0)
        protected_set = {tuple(edge) for edge in protected.tolist()}
        self.assertNotIn(internal_edge, protected_set)
        self.assertIn(boundary_edge, protected_set)
        self.assertEqual(stats["relaxed_edges"], 1)
        self.assertGreater(stats["projected_vertices"], 0)
        self.assertLessEqual(stats["maximum_displacement"], 0.2)

    def test_analytic_cylinder_recovery_preserves_geometry_beyond_guard(self):
        vertices, faces, model = self._disturbed_cylinder_patch(10.3)

        recovered, _, stats = (
            original_constrained.recover_analytic_cylinder_support(
                vertices,
                faces,
                np.empty((0, 2), dtype=np.int64),
                (model,),
                maximum_projection_distance=0.2,
            )
        )

        np.testing.assert_allclose(recovered[4], vertices[4])
        self.assertLessEqual(stats["maximum_displacement"], 0.2)

    def test_partition_colors_distinguish_disconnected_surface_regions(self):
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (4.0, 1.0, 0.0),
                (3.0, 1.0, 0.0),
            ),
            dtype=np.float64,
        )
        faces = np.asarray(
            ((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)),
            dtype=np.int64,
        )

        colors, labels, stats = (
            original_constrained.build_feature_partition_face_colors(
                vertices,
                faces,
                feature_angle_degrees=15.0,
            )
        )

        self.assertEqual(stats["partitions"], 2)
        self.assertEqual(colors.shape, (4, 4))
        self.assertTrue(np.all(colors[:, 3] == 255))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertFalse(np.array_equal(colors[0], colors[2]))

    def test_false_hard_edge_inside_one_cylinder_support_is_relaxed(self):
        angles = np.asarray([0.0, 0.2])
        vertices = np.asarray(
            [
                [10.0 * np.cos(angle), 10.0 * np.sin(angle), height]
                for height in (0.0, 1.0)
                for angle in angles
            ],
            dtype=np.float64,
        )
        faces = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int64)
        shared_diagonal = (1, 2)
        true_boundary = (0, 1)
        model = {
            "axis": np.asarray([0.0, 0.0, 1.0]),
            "origin": np.zeros(3),
            "radius": 10.0,
        }

        remaining, relaxed = (
            original_constrained._relax_false_cylinder_feature_edges(
                vertices,
                faces,
                np.asarray((shared_diagonal, true_boundary), dtype=np.int64),
                [model],
            )
        )
        remaining = {tuple(map(int, edge)) for edge in remaining}

        self.assertEqual(relaxed, 1)
        self.assertNotIn(shared_diagonal, remaining)
        self.assertIn(true_boundary, remaining)

    def test_targeted_edge_splits_are_conforming_and_keep_lineage(self):
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (4.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        diagonal = (0, 2)
        (
            result_vertices,
            result_faces,
            result_lineages,
            _,
            stats,
        ) = original_constrained._split_selected_edges_by_length(
            vertices,
            faces,
            {diagonal: 1.1},
            edge_lineages={diagonal: 7},
        )
        lineage_lengths = [
            np.linalg.norm(result_vertices[first] - result_vertices[second])
            for (first, second), root in result_lineages.items()
            if root == 7
        ]
        edge_memberships = original_constrained._build_edge_faces(result_faces)

        self.assertEqual(stats["splits"], 3)
        self.assertFalse(stats["hit_split_limit"])
        self.assertEqual(len(lineage_lengths), 4)
        self.assertLessEqual(max(lineage_lengths), 1.1)
        self.assertAlmostEqual(sum(lineage_lengths), np.sqrt(17.0))
        self.assertTrue(
            all(
                len(edge_memberships[edge]) == 2
                for edge, root in result_lineages.items()
                if root == 7
            )
        )

    def test_uniform_target_spacing_can_use_direct_planar_target(self):
        self.assertAlmostEqual(
            original_constrained._uniform_target_spacing(1.0, 10.0),
            1.25,
        )
        self.assertAlmostEqual(
            original_constrained._uniform_target_spacing(
                1.0, 10.0, gradual=False
            ),
            10.0,
        )

    def test_overdense_high_quality_plane_uses_local_boundary_grading(self):
        count = 21
        coordinates = np.linspace(0.0, 10.0, count)
        vertices = np.asarray(
            [(x, y, 0.0) for y in coordinates for x in coordinates]
        )
        faces = []
        for row in range(count - 1):
            for column in range(count - 1):
                first = row * count + column
                faces.extend(
                    (
                        (first, first + 1, first + count),
                        (first + 1, first + count + 1, first + count),
                    )
                )
        faces = np.asarray(faces, dtype=np.int64)
        boundary = []
        for index in range(count - 1):
            boundary.extend(
                (
                    (index, index + 1),
                    ((count - 1) * count + index,
                     (count - 1) * count + index + 1),
                    (index * count, (index + 1) * count),
                    (index * count + count - 1,
                     (index + 1) * count + count - 1),
                )
            )
        boundary = np.asarray(boundary, dtype=np.int64)

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_planar_annuli(
                vertices,
                faces,
                protected_edges=boundary,
                minimum_faces=20,
                minimum_holes=0,
                maximum_holes=0,
                maximum_target_edge_length=5.0,
                maximum_result_edge_length=5.0,
                minimum_angle_degrees=20.0,
                accept_strong_mean_gain=True,
                skip_satisfactory_quality=True,
                gradual_target_spacing=False,
            )
        )
        result_edges = np.sort(
            result_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
            axis=1,
        )
        result_edge_set = set(map(tuple, result_edges))
        lengths = np.linalg.norm(
            result_vertices[result_edges[:, 0]]
            - result_vertices[result_edges[:, 1]],
            axis=1,
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )

        self.assertEqual(stats["regions"], 1)
        self.assertEqual(stats["density_simplifications"], 1)
        self.assertLess(len(result_faces), len(faces) * 0.8)
        self.assertLessEqual(float(lengths.max()), 5.0 * (1.0 + 1e-8))
        self.assertGreaterEqual(float(qualities.mean()), 0.45)
        self.assertTrue(all(tuple(edge) in result_edge_set for edge in boundary))

    def test_short_coplanar_edge_collapse_preserves_boundary_and_quality(self):
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 2.0, 0.0),
                (0.0, 2.0, 0.0),
                (0.9, 1.0, 0.0),
                (1.1, 1.0, 0.0),
            )
        )
        faces = np.asarray(
            (
                (0, 1, 4),
                (1, 5, 4),
                (1, 2, 5),
                (2, 3, 5),
                (3, 4, 5),
                (3, 0, 4),
            ),
            dtype=np.int64,
        )
        old_quality = feature_optimize._triangle_quality_values(vertices, faces)

        result_vertices, result_faces, collapse_count = (
            original_constrained.collapse_short_coplanar_edges(
                vertices,
                faces,
                np.empty((0, 2), dtype=np.int64),
                maximum_short_edge_length=0.5,
                maximum_edge_length=3.0,
                passes=1,
            )
        )
        new_quality = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
        }

        self.assertEqual(collapse_count, 1)
        self.assertEqual(len(result_faces), 4)
        self.assertGreaterEqual(float(new_quality.min()), float(old_quality.min()))
        self.assertGreater(float(new_quality.mean()), float(old_quality.mean()))
        self.assertTrue(
            all(edge in result_edges for edge in ((0, 1), (1, 2), (2, 3), (0, 3)))
        )

    def test_original_refinement_collapses_short_coplanar_interiors_by_default(self):
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 2.0, 0.0),
                (0.0, 2.0, 0.0),
                (0.9, 1.0, 0.0),
                (1.1, 1.0, 0.0),
            )
        )
        faces = np.asarray(
            (
                (0, 1, 4),
                (1, 5, 4),
                (1, 2, 5),
                (2, 3, 5),
                (3, 4, 5),
                (3, 0, 4),
            ),
            dtype=np.int64,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        result_vertices, result_faces, stats = (
            original_constrained.refine_original_mesh_by_longest_edge(
                mesh,
                max_edge_length=3.0,
                flip_passes=0,
            )
        )
        result_edges = original_constrained._build_edge_faces(result_faces)
        result_lengths = [
            np.linalg.norm(result_vertices[first] - result_vertices[second])
            for first, second in result_edges
        ]

        self.assertTrue(stats["already_satisfied"])
        self.assertEqual(stats["short_edge_collapses"], 1)
        self.assertEqual(len(result_faces), 4)
        self.assertLessEqual(max(result_lengths), 3.0 * (1.0 + 1e-8))
        self.assertTrue(
            all(edge in result_edges for edge in ((0, 1), (1, 2), (2, 3), (0, 3)))
        )

    def test_concave_planar_loop_uses_verified_interior_point(self):
        polygon = np.asarray(
            (
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 1.0),
                (1.0, 1.0),
                (1.0, 3.0),
                (4.0, 3.0),
                (4.0, 4.0),
                (0.0, 4.0),
            )
        )
        mean_point = polygon.mean(axis=0)
        interior_point = original_constrained._polygon_interior_point(polygon)

        self.assertFalse(
            original_constrained._points_in_polygon(
                mean_point[None, :], polygon
            )[0]
        )
        self.assertIsNotNone(interior_point)
        self.assertTrue(
            original_constrained._points_in_polygon(
                interior_point[None, :], polygon
            )[0]
        )

    def test_two_fillet_bands_are_not_joined_through_a_planar_bridge(self):
        count = 33
        radius = 5.0
        length = 10.0
        left_angles = np.linspace(0.0, 0.5 * np.pi, count)
        right_angles = np.linspace(0.5 * np.pi, np.pi, count)

        def band(center_x, angles):
            return np.asarray(
                [
                    [
                        center_x + radius * np.cos(angle),
                        axial,
                        radius * np.sin(angle),
                    ]
                    for axial in (0.0, length)
                    for angle in angles
                ]
            )

        vertices = np.vstack((band(0.0, left_angles), band(20.0, right_angles)))
        faces = []
        for base in (0, 2 * count):
            for index in range(count - 1):
                faces.extend(
                    (
                        [base + index, base + index + 1, base + count + index],
                        [
                            base + index + 1,
                            base + count + index + 1,
                            base + count + index,
                        ],
                    )
                )
        # A two-triangle tangent plane joins the two different-radius-center bands.
        faces.extend(
            (
                [count - 1, 2 * count, 2 * count - 1],
                [2 * count, 3 * count, 2 * count - 1],
            )
        )
        faces = np.asarray(faces, dtype=np.int64)

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_partial_cylindrical_walls(
                vertices,
                faces,
                np.empty((0, 2), dtype=np.int64),
                minimum_faces=12,
                radius_tolerance=0.01,
                normal_tolerance=0.08,
                minimum_angle_degrees=5.0,
                target_edge_ratio=2.0,
                isolate_rounded_faces=True,
                minimum_curvature_degrees=0.2,
                maximum_source_quality=0.15,
                minimum_triangle_angle_degrees=28.0,
                model_guided_boundary_recovery=True,
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )

        self.assertEqual(stats["candidates"], 2)
        self.assertEqual(stats["patches"], 2)
        self.assertGreater(stats["new_quality"], 0.7)
        self.assertGreater(stats["new_quality_p5"], stats["old_quality_p5"])
        self.assertGreater(float(qualities.mean()), 0.5)

    def test_model_guided_fillet_growth_crosses_source_quality_breaks(self):
        radius = 5.0
        angles = np.linspace(0.0, 0.5 * np.pi, 33)
        axial_positions = np.asarray((0.0, 0.02, 0.22, 0.24))
        vertices = np.asarray(
            [
                [radius * np.cos(angle), axial, radius * np.sin(angle)]
                for axial in axial_positions
                for angle in angles
            ]
        )
        count = len(angles)
        faces = []
        for row in range(len(axial_positions) - 1):
            lower = row * count
            upper = (row + 1) * count
            for index in range(count - 1):
                faces.extend(
                    (
                        [lower + index, lower + index + 1, upper + index],
                        [
                            lower + index + 1,
                            upper + index + 1,
                            upper + index,
                        ],
                    )
                )
        faces = np.asarray(faces, dtype=np.int64)

        _, _, fragmented = (
            original_constrained.retriangulate_partial_cylindrical_walls(
                vertices,
                faces,
                np.empty((0, 2), dtype=np.int64),
                minimum_faces=12,
                radius_tolerance=0.05,
                normal_tolerance=0.12,
                minimum_angle_degrees=5.0,
                target_edge_ratio=2.0,
                isolate_rounded_faces=True,
                minimum_curvature_degrees=0.2,
                maximum_source_quality=0.15,
                minimum_triangle_angle_degrees=28.0,
            )
        )
        result_vertices, result_faces, recovered = (
            original_constrained.retriangulate_partial_cylindrical_walls(
                vertices,
                faces,
                np.empty((0, 2), dtype=np.int64),
                minimum_faces=12,
                radius_tolerance=0.05,
                normal_tolerance=0.12,
                minimum_angle_degrees=5.0,
                target_edge_ratio=2.0,
                isolate_rounded_faces=True,
                minimum_curvature_degrees=0.2,
                maximum_source_quality=0.15,
                minimum_triangle_angle_degrees=28.0,
                model_guided_boundary_recovery=True,
            )
        )

        self.assertEqual(fragmented["patches"], 2)
        self.assertEqual(recovered["patches"], 1)
        self.assertEqual(recovered["removed_faces"], len(faces))
        self.assertLess(len(result_faces), len(faces) * 0.5)
        self.assertGreater(recovered["new_quality"], 0.8)
        self.assertTrue(np.all(np.isfinite(result_vertices)))

    def test_slightly_tapered_fillet_band_uses_relaxed_safe_fit(self):
        count = 33
        angles = np.linspace(0.0, 0.5 * np.pi, count)
        lower_radius = 5.0
        upper_radius = 6.5
        length = 10.0
        vertices = np.vstack(
            (
                np.column_stack(
                    (
                        lower_radius * np.cos(angles),
                        np.zeros(count),
                        lower_radius * np.sin(angles),
                    )
                ),
                np.column_stack(
                    (
                        upper_radius * np.cos(angles),
                        np.full(count, length),
                        upper_radius * np.sin(angles),
                    )
                ),
            )
        )
        faces = []
        for index in range(count - 1):
            faces.extend(
                (
                    [index, index + 1, count + index],
                    [index + 1, count + index + 1, count + index],
                )
            )
        faces = np.asarray(faces, dtype=np.int64)
        common_options = dict(
            minimum_faces=12,
            normal_tolerance=0.12,
            minimum_angle_degrees=5.0,
            target_edge_ratio=2.0,
            isolate_rounded_faces=True,
            minimum_curvature_degrees=0.2,
            maximum_source_quality=0.15,
            minimum_triangle_angle_degrees=28.0,
            minimum_radial_alignment=0.95,
        )

        _, _, strict_stats = (
            original_constrained.retriangulate_partial_cylindrical_walls(
                vertices,
                faces,
                np.empty((0, 2), dtype=np.int64),
                radius_tolerance=0.01,
                **common_options,
            )
        )
        result_vertices, result_faces, relaxed_stats = (
            original_constrained.retriangulate_partial_cylindrical_walls(
                vertices,
                faces,
                np.empty((0, 2), dtype=np.int64),
                radius_tolerance=0.05,
                **common_options,
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )

        self.assertEqual(strict_stats["patches"], 0)
        self.assertEqual(relaxed_stats["patches"], 1)
        self.assertGreater(
            relaxed_stats["new_quality_p5"],
            relaxed_stats["old_quality_p5"],
        )
        self.assertGreater(float(qualities.mean()), 0.5)

    def test_solid_planar_fan_uses_uniform_constrained_mesh(self):
        count = 40
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                [0.0, 0.0, 0.0],
                np.column_stack(
                    (10 * np.cos(angles), 10 * np.sin(angles), np.zeros(count))
                ),
            )
        )
        faces = np.asarray(
            [[0, index + 1, (index + 1) % count + 1] for index in range(count)],
            dtype=np.int64,
        )
        boundary = np.asarray(
            [[index + 1, (index + 1) % count + 1] for index in range(count)],
            dtype=np.int64,
        )

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_planar_annuli(
                vertices,
                faces,
                protected_edges=boundary,
                minimum_faces=20,
                minimum_holes=0,
                maximum_holes=0,
                maximum_target_edge_length=5.0,
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )

        self.assertEqual(stats["regions"], 1)
        self.assertGreater(float(qualities.mean()), 0.9)
        self.assertGreater(float(np.percentile(qualities, 5.0)), 0.7)

    def test_half_cylinder_with_irregular_boundary_is_retriangulated(self):
        count = 31
        angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (10 * np.cos(angles), 10 * np.sin(angles), np.zeros(count))
                ),
                np.column_stack(
                    (10 * np.cos(angles), 10 * np.sin(angles), np.full(count, 10.0))
                ),
            )
        )
        faces = []
        for index in range(count - 1):
            faces.extend(
                (
                    [index, index + 1, count + index],
                    [index + 1, count + index + 1, count + index],
                )
            )
        faces = np.asarray(faces, dtype=np.int64)
        boundary = np.asarray(
            [[index, index + 1] for index in range(count - 1)]
            + [[count + index, count + index + 1] for index in range(count - 1)]
            + [[0, count], [count - 1, 2 * count - 1]],
            dtype=np.int64,
        )

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_partial_cylindrical_walls(
                vertices, faces, boundary, minimum_faces=20
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }

        self.assertEqual(stats["patches"], 1)
        self.assertGreater(float(np.percentile(qualities, 5.0)), 0.8)
        self.assertTrue(
            all(tuple(sorted(map(int, edge))) in result_edges for edge in boundary)
        )

    def test_half_cylinder_accepts_curved_nonparallel_end_boundaries(self):
        vertices, faces, boundary = self._irregular_trimmed_cylinder(4.0, 180.0)
        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_partial_cylindrical_walls(
                vertices,
                faces,
                boundary,
                minimum_faces=20,
                radius_tolerance=3e-3,
                normal_tolerance=3e-2,
                minimum_angle_degrees=30.0,
                preferred_edge_length=5.0,
                allow_density_simplification=True,
            )
        )
        radii = np.linalg.norm(result_vertices[:, :2], axis=1)

        self.assertEqual(stats["patches"], 1)
        self.assertLess(len(result_faces), len(faces))
        self.assertLess(float(np.max(np.abs(radii - 4.0))), 6e-3)

    def test_large_radius_shallow_arc_is_coarsened_anisotropically(self):
        vertices, faces, boundary = self._irregular_trimmed_cylinder(100.0, 20.0)
        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_partial_cylindrical_walls(
                vertices,
                faces,
                boundary,
                minimum_faces=20,
                radius_tolerance=3e-3,
                normal_tolerance=3e-2,
                minimum_angle_degrees=5.0,
                maximum_angle_degrees=30.0,
                preferred_edge_length=5.0,
                minimum_triangle_angle_degrees=20.0,
                direct_target_spacing=True,
                allow_density_simplification=True,
                maximum_result_edge_length=10.0,
            )
        )
        result_triangles = result_vertices[result_faces]
        edge_lengths = np.linalg.norm(
            result_triangles[:, (1, 2, 0)]
            - result_triangles[:, (0, 1, 2)],
            axis=2,
        )
        radii = np.linalg.norm(result_vertices[:, :2], axis=1)

        self.assertEqual(stats["patches"], 1)
        self.assertLess(len(result_faces), len(faces) * 0.3)
        self.assertLessEqual(float(edge_lengths.max()), 10.0 * (1.0 + 1e-8))
        self.assertLess(float(np.max(np.abs(radii - 100.0))), 3e-3)

    def test_short_cylinder_can_retriangulate_without_new_vertices(self):
        count = 30
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (10 * np.cos(angles), 10 * np.sin(angles), np.zeros(count))
                ),
                np.column_stack(
                    (10 * np.cos(angles), 10 * np.sin(angles), np.ones(count))
                ),
            )
        )
        faces = np.asarray(
            [
                face
                for index in range(count)
                for face in (
                    [index, (index + 1) % count, count + index],
                    [
                        (index + 1) % count,
                        count + (index + 1) % count,
                        count + index,
                    ],
                )
            ],
            dtype=np.int64,
        )
        boundaries = np.asarray(
            [[index, (index + 1) % count] for index in range(count)]
            + [
                [count + index, count + (index + 1) % count]
                for index in range(count)
            ],
            dtype=np.int64,
        )

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_cylindrical_walls(
                vertices,
                faces,
                boundaries,
                minimum_faces=20,
            )
        )

        self.assertEqual(stats["walls"], 1)
        self.assertEqual(len(result_vertices), len(vertices))
        self.assertEqual(result_vertices.shape[1], 3)
        self.assertEqual(len(result_faces), len(faces))

    def test_embossing_interrupted_cylinder_support_is_recovered(self):
        angular_count = 41
        axial_count = 6
        angles = np.linspace(-0.75 * np.pi, 0.75 * np.pi, angular_count)
        heights = np.asarray([0.0, 4.8, 4.9, 5.0, 5.1, 10.0])
        vertices = np.asarray(
            [
                [10.0 * np.cos(angle), 10.0 * np.sin(angle), height]
                for height in heights
                for angle in angles
            ],
            dtype=np.float64,
        )
        faces = []
        for axial_index in range(axial_count - 1):
            for angular_index in range(angular_count - 1):
                # A rectangular opening represents the support removed by
                # an embossed feature Boolean union.
                if 2 <= axial_index <= 3 and 16 <= angular_index <= 23:
                    continue
                lower = axial_index * angular_count + angular_index
                upper = lower + angular_count
                faces.extend(
                    (
                        [lower, lower + 1, upper],
                        [lower + 1, upper + 1, upper],
                    )
                )
        faces = np.asarray(faces, dtype=np.int64)
        edges = np.sort(
            faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1
        )
        unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
        boundary = unique_edges[edge_counts == 1]
        internal_constraints = np.asarray(
            [
                [
                    axial_index * angular_count + 10,
                    (axial_index + 1) * angular_count + 10,
                ]
                for axial_index in range(axial_count - 1)
            ],
            dtype=np.int64,
        )
        protected = np.vstack((boundary, internal_constraints))
        model = {
            "axis": np.asarray([0.0, 0.0, 1.0]),
            "origin": np.zeros(3),
            "first_basis": np.asarray([1.0, 0.0, 0.0]),
            "second_basis": np.asarray([0.0, 1.0, 0.0]),
            "radius": 10.0,
            "source_faces": len(faces),
        }
        old_quality = feature_optimize._triangle_quality_values(vertices, faces)

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_interrupted_cylindrical_walls(
                vertices,
                faces,
                protected,
                [model],
                minimum_faces=20,
                radius_tolerance=0.03,
                preferred_edge_length=1.5,
            )
        )
        new_quality = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        periodic_vertices, periodic_faces, periodic_stats = (
            original_constrained.retriangulate_interrupted_cylindrical_walls(
                result_vertices,
                result_faces,
                protected,
                [model],
                minimum_faces=20,
                radius_tolerance=0.03,
                preferred_edge_length=1.5,
                seam_angle_offset=np.pi,
            )
        )
        periodic_quality = feature_optimize._triangle_quality_values(
            periodic_vertices, periodic_faces
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in periodic_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }

        self.assertGreaterEqual(stats["patches"], 1, stats)
        self.assertGreaterEqual(periodic_stats["patches"], 1, periodic_stats)
        self.assertGreater(float(new_quality.mean()), float(old_quality.mean()))
        self.assertGreaterEqual(
            float(periodic_quality.mean()), float(new_quality.mean()) - 1e-8
        )
        self.assertTrue(
            all(tuple(map(int, edge)) in result_edges for edge in protected)
        )

    def test_mismatched_cylinder_rings_get_axial_transition_rows(self):
        lower_count = 60
        upper_count = 20
        lower_angles = np.arange(lower_count) * (2.0 * np.pi / lower_count)
        upper_angles = np.arange(upper_count) * (2.0 * np.pi / upper_count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (10 * np.cos(lower_angles), 10 * np.sin(lower_angles),
                     np.zeros(lower_count))
                ),
                np.column_stack(
                    (10 * np.cos(upper_angles), 10 * np.sin(upper_angles),
                     np.full(upper_count, 8.0))
                ),
            )
        )
        faces = []
        lower_index = 0
        upper_index = 0
        while lower_index < lower_count or upper_index < upper_count:
            next_lower = (
                (lower_index + 1) / lower_count
                if lower_index < lower_count else 2.0
            )
            next_upper = (
                (upper_index + 1) / upper_count
                if upper_index < upper_count else 2.0
            )
            if next_lower <= next_upper:
                faces.append(
                    [lower_index % lower_count,
                     (lower_index + 1) % lower_count,
                     lower_count + upper_index % upper_count]
                )
                lower_index += 1
            else:
                faces.append(
                    [lower_index % lower_count,
                     lower_count + (upper_index + 1) % upper_count,
                     lower_count + upper_index % upper_count]
                )
                upper_index += 1
        faces = np.asarray(faces, dtype=np.int64)
        boundaries = np.asarray(
            [[index, (index + 1) % lower_count] for index in range(lower_count)]
            + [
                [lower_count + index,
                 lower_count + (index + 1) % upper_count]
                for index in range(upper_count)
            ],
            dtype=np.int64,
        )

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_cylindrical_walls(
                vertices,
                faces,
                boundaries,
                minimum_faces=20,
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }
        edge_array = np.sort(
            result_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
            axis=1,
        )
        unique_edges, edge_counts = np.unique(
            edge_array, axis=0, return_counts=True
        )
        boundary_set = {
            tuple(sorted(map(int, edge))) for edge in boundaries
        }

        self.assertEqual(stats["walls"], 1)
        self.assertGreater(float(np.percentile(qualities, 5.0)), 0.5)
        self.assertTrue(
            all(tuple(sorted(map(int, edge))) in result_edges for edge in boundaries)
        )
        self.assertTrue(
            all(
                int(count) == (1 if tuple(map(int, edge)) in boundary_set else 2)
                for edge, count in zip(unique_edges, edge_counts)
            )
        )

    def test_closed_cylinder_uses_integer_periodic_column_count(self):
        count = 60
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (10.0 * np.cos(angles), 10.0 * np.sin(angles), np.zeros(count))
                ),
                np.column_stack(
                    (
                        10.0 * np.cos(angles),
                        10.0 * np.sin(angles),
                        np.full(count, 8.0),
                    )
                ),
            )
        )
        faces = np.asarray(
            [
                face
                for index in range(count)
                for face in (
                    [index, (index + 1) % count, count + index],
                    [
                        (index + 1) % count,
                        count + (index + 1) % count,
                        count + index,
                    ],
                )
            ],
            dtype=np.int64,
        )
        boundaries = np.asarray(
            [[index, (index + 1) % count] for index in range(count)]
            + [
                [count + index, count + (index + 1) % count]
                for index in range(count)
            ],
            dtype=np.int64,
        )

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_cylindrical_walls(
                vertices,
                faces,
                boundaries,
                minimum_faces=20,
                preferred_edge_length=1.5,
            )
        )
        centroids = result_vertices[result_faces].mean(axis=1)
        middle = (centroids[:, 2] > 1.0) & (centroids[:, 2] < 7.0)
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )[middle]

        self.assertEqual(stats["walls"], 1)
        self.assertGreater(float(qualities.min()), 0.65)

    def test_dense_cylinder_boundaries_only_refine_local_transition(self):
        count = 60
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (10.0 * np.cos(angles), 10.0 * np.sin(angles), np.zeros(count))
                ),
                np.column_stack(
                    (
                        10.0 * np.cos(angles),
                        10.0 * np.sin(angles),
                        np.full(count, 50.0),
                    )
                ),
            )
        )
        faces = np.asarray(
            [
                face
                for index in range(count)
                for face in (
                    [index, (index + 1) % count, count + index],
                    [
                        (index + 1) % count,
                        count + (index + 1) % count,
                        count + index,
                    ],
                )
            ],
            dtype=np.int64,
        )
        boundaries = np.asarray(
            [[index, (index + 1) % count] for index in range(count)]
            + [
                [count + index, count + (index + 1) % count]
                for index in range(count)
            ],
            dtype=np.int64,
        )

        _, gradual_faces, gradual_stats = (
            original_constrained.retriangulate_cylindrical_walls(
                vertices,
                faces,
                boundaries,
                minimum_faces=20,
                preferred_edge_length=5.0,
            )
        )
        direct_vertices, direct_faces, direct_stats = (
            original_constrained.retriangulate_cylindrical_walls(
                vertices,
                faces,
                boundaries,
                minimum_faces=20,
                preferred_edge_length=5.0,
                direct_target_spacing=True,
            )
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in direct_faces
            for first, second in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
        }

        self.assertEqual(gradual_stats["walls"], 1)
        self.assertEqual(direct_stats["walls"], 1)
        self.assertLess(len(direct_faces), len(gradual_faces) * 0.25)
        self.assertTrue(
            all(tuple(sorted(map(int, edge))) in result_edges for edge in boundaries)
        )
        self.assertTrue(np.isfinite(direct_vertices).all())

    def test_boolean_trimmed_cylinder_keeps_irregular_join_loops(self):
        count = 60
        angles = np.arange(count) * (2.0 * np.pi / count)
        lower_z = 0.8 * np.sin(2.0 * angles)
        upper_z = 8.0 + 0.7 * np.cos(3.0 * angles)
        vertices = np.vstack(
            (
                np.column_stack(
                    (10.0 * np.cos(angles), 10.0 * np.sin(angles), lower_z)
                ),
                np.column_stack(
                    (10.0 * np.cos(angles), 10.0 * np.sin(angles), upper_z)
                ),
            )
        )
        faces = []
        for index in range(count):
            following = (index + 1) % count
            faces.extend(
                (
                    [index, following, count + index],
                    [following, count + following, count + index],
                )
            )
        faces = np.asarray(faces, dtype=np.int64)
        boundaries = np.asarray(
            [[index, (index + 1) % count] for index in range(count)]
            + [
                [count + index, count + (index + 1) % count]
                for index in range(count)
            ],
            dtype=np.int64,
        )
        old_quality = feature_optimize._triangle_quality_values(vertices, faces)

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_trimmed_cylindrical_walls(
                vertices,
                faces,
                boundaries,
                minimum_faces=20,
                radius_tolerance=0.01,
                normal_tolerance=0.08,
            )
        )
        new_quality = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }

        self.assertEqual(stats["walls"], 1)
        self.assertGreater(float(new_quality.mean()), float(old_quality.mean()))
        self.assertGreater(float(np.percentile(new_quality, 5.0)), 0.5)
        self.assertTrue(
            all(tuple(sorted(map(int, edge))) in result_edges for edge in boundaries)
        )

    def test_planar_annulus_bridges_are_retriangulated_uniformly(self):
        count = 80
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (30.0 * np.cos(angles), 30.0 * np.sin(angles), np.zeros(count))
                ),
                np.column_stack(
                    (12.0 * np.cos(angles), 12.0 * np.sin(angles), np.zeros(count))
                ),
            )
        )
        faces = []
        for index in range(count):
            following = (index + 1) % count
            faces.extend(
                (
                    [index, following, count + index],
                    [following, count + following, count + index],
                )
            )
        faces = np.asarray(faces, dtype=np.int64)

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_planar_annuli(
                vertices,
                faces,
                minimum_faces=20,
                maximum_target_edge_length=10.0,
                maximum_result_edge_length=10.0,
                minimum_angle_degrees=28.0,
                gradual_target_spacing=False,
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        triangles = result_vertices[result_faces]
        edge_lengths = np.linalg.norm(
            triangles[:, (1, 2, 0)] - triangles[:, (0, 1, 2)], axis=2
        ).reshape(-1)
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }

        self.assertEqual(stats["regions"], 1)
        self.assertGreater(stats["new_vertices"], 0)
        self.assertGreater(float(np.percentile(qualities, 5.0)), 0.65)
        edge_percentiles = np.percentile(edge_lengths, (10.0, 50.0, 90.0))
        self.assertLess(float(edge_percentiles[0]), float(edge_percentiles[1]))
        self.assertLess(float(edge_percentiles[1]), float(edge_percentiles[2]))
        self.assertLessEqual(float(edge_lengths.max()), 10.0 + 1e-8)
        for offset in (0, count):
            self.assertTrue(
                all(
                    tuple(
                        sorted((offset + index, offset + (index + 1) % count))
                    )
                    in result_edges
                    for index in range(count)
                )
            )

    def test_planar_annulus_with_internal_hard_edge_is_not_replaced(self):
        count = 12
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (3 * np.cos(angles), 3 * np.sin(angles), np.zeros(count))
                ),
                np.column_stack(
                    (np.cos(angles), np.sin(angles), np.zeros(count))
                ),
            )
        )
        faces = []
        for index in range(count):
            following = (index + 1) % count
            faces.extend(
                (
                    [index, following, count + index],
                    [following, count + following, count + index],
                )
            )
        faces = np.asarray(faces, dtype=np.int64)
        internal_hard_edge = np.asarray([[0, count]], dtype=np.int64)

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_planar_annuli(
                vertices,
                faces,
                protected_edges=internal_hard_edge,
                minimum_faces=20,
            )
        )

        self.assertEqual(stats["regions"], 0)
        self.assertEqual(stats["constraint_rejections"], 1)
        np.testing.assert_array_equal(result_vertices, vertices)
        np.testing.assert_array_equal(result_faces, faces)

    def test_planar_circle_fan_is_retriangulated_uniformly(self):
        count = 81
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                [0.0, 0.0, 0.0],
                np.column_stack(
                    (28.0 * np.cos(angles), 28.0 * np.sin(angles), np.zeros(count))
                ),
            )
        )
        faces = np.asarray(
            [[0, index + 1, (index + 1) % count + 1] for index in range(count)],
            dtype=np.int64,
        )
        boundary = np.asarray(
            [[index + 1, (index + 1) % count + 1] for index in range(count)],
            dtype=np.int64,
        )

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_planar_fans(
                vertices,
                faces,
                boundary,
                minimum_valence=30,
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }

        self.assertEqual(stats["fans"], 1)
        self.assertEqual(np.count_nonzero(result_faces == 0), 0)
        self.assertGreater(float(qualities.min()), 0.6)
        self.assertTrue(
            all(tuple(sorted(map(int, edge))) in result_edges for edge in boundary)
        )

    def test_automatic_edge_limit_uses_five_percent_diagonal(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]
        )
        edge_lengths = np.array([1.0, 2.0, 3.0])

        limit, diagonal, percentile_95 = (
            original_constrained.automatic_edge_length_limit(
                vertices,
                edge_lengths,
            )
        )

        self.assertEqual(diagonal, 1000.0)
        self.assertAlmostEqual(percentile_95, 2.9)
        self.assertAlmostEqual(limit, 50.0)

    def test_automatic_split_limit_has_conformity_margin(self):
        budget, estimate = original_constrained.automatic_split_limit(
            np.array([5.0, 20.0, 40.0]),
            max_edge_length=10.0,
        )

        self.assertEqual(estimate, 4)
        self.assertEqual(budget, 100000)

    def test_feature_snap_backtracks_instead_of_flipping_triangle(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        corner_target = np.array([[2.0, 2.0, 0.0]])
        constraints = feature_optimize.FeatureConstraintMap(
            reference_edges=np.empty((0, 2), dtype=np.int64),
            reference_segments=np.empty((0, 2, 3), dtype=np.float64),
            matched_edges=np.empty((0, 2), dtype=np.int64),
            feature_vertices=np.empty(0, dtype=np.int64),
            feature_segment_ids=np.empty(0, dtype=np.int64),
            corner_vertices=np.array([0], dtype=np.int64),
            corner_targets=corner_target,
            match_tolerance=1.0,
        )

        snapped, relaxed = feature_optimize._snap_to_feature_constraints(
            vertices,
            faces,
            constraints,
            minimum_area_squared=1e-24,
        )

        np.testing.assert_array_equal(relaxed, [0])
        self.assertFalse(np.allclose(snapped[0], corner_target[0]))
        self.assertGreater(np.linalg.norm(snapped[0] - vertices[0]), 0.0)
        self.assertTrue(
            feature_optimize._valid_relocation(
                vertices,
                snapped,
                faces,
                minimum_area_squared=1e-24,
            )
        )

    def test_flips_only_when_local_triangle_quality_improves(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
        before = feature_optimize.mesh_quality_metrics(vertices, faces)

        flipped_faces, flip_count = feature_optimize._flip_quality_edges(
            vertices,
            faces,
            np.empty((0, 2), dtype=np.int64),
            passes=1,
        )
        after = feature_optimize.mesh_quality_metrics(
            vertices,
            flipped_faces,
        )

        self.assertEqual(flip_count, 1)
        self.assertIn(
            (0, 3),
            {
                edge
                for face in flipped_faces
                for edge in feature_optimize._face_edges(face)
            },
        )
        self.assertGreater(
            after["minimum_triangle_quality"],
            before["minimum_triangle_quality"],
        )

    def test_coplanar_flip_filter_preserves_fold_edge(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)

        filtered_faces, flip_count = feature_optimize._flip_quality_edges(
            vertices,
            faces,
            np.empty((0, 2), dtype=np.int64),
            passes=4,
            maximum_dihedral_degrees=0.1,
        )

        self.assertEqual(flip_count, 0)
        np.testing.assert_array_equal(filtered_faces, faces)

    def test_coplanar_flip_respects_maximum_new_edge_length(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)

        filtered_faces, flip_count = feature_optimize._flip_quality_edges(
            vertices,
            faces,
            np.empty((0, 2), dtype=np.int64),
            passes=4,
            maximum_dihedral_degrees=0.1,
            maximum_edge_length=1.0,
        )

        self.assertEqual(flip_count, 0)
        np.testing.assert_array_equal(filtered_faces, faces)

    def test_preferred_coplanar_seam_flips_at_equal_quality(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        flipped_faces, flip_count = feature_optimize._flip_quality_edges(
            vertices,
            faces,
            np.empty((0, 2), dtype=np.int64),
            passes=1,
            maximum_dihedral_degrees=0.1,
            preferred_edges=np.array([[0, 2]], dtype=np.int64),
        )

        self.assertEqual(flip_count, 1)
        output_edges = {
            edge
            for face in flipped_faces
            for edge in feature_optimize._face_edges(face)
        }
        self.assertNotIn((0, 2), output_edges)
        self.assertIn((1, 3), output_edges)

    def test_maps_boundary_features_and_geometric_corners(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        reference = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

        constraints = feature_optimize.build_feature_constraint_map(
            reference,
            vertices,
            faces,
            resolution=64,
            feature_angle_degrees=30.0,
        )

        self.assertEqual(len(constraints.reference_edges), 4)
        self.assertEqual(len(constraints.matched_edges), 4)
        np.testing.assert_array_equal(
            np.sort(constraints.corner_vertices),
            np.arange(4),
        )

    def test_relocates_interior_vertex_but_keeps_feature_corners(self):
        reference_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        reference_faces = np.array(
            [[0, 1, 2], [0, 2, 3]],
            dtype=np.int64,
        )
        reference = trimesh.Trimesh(
            vertices=reference_vertices,
            faces=reference_faces,
            process=False,
        )
        vertices = np.vstack(
            (reference_vertices, np.array([[0.7, 0.4, 0.0]]))
        )
        faces = np.array(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=np.int64,
        )

        optimized, optimized_faces, stats = (
            feature_optimize.optimize_feature_constrained_mesh(
                reference,
                vertices,
                faces,
                resolution=64,
                feature_angle_degrees=30.0,
                iterations=5,
                smoothing_step=0.5,
                flip_passes=0,
            )
        )

        np.testing.assert_allclose(optimized[:4], reference_vertices)
        self.assertLess(
            np.linalg.norm(optimized[4] - [0.5, 0.5, 0.0]),
            np.linalg.norm(vertices[4] - [0.5, 0.5, 0.0]),
        )
        self.assertLess(
            stats["final_metrics"]["energy"],
            stats["initial_metrics"]["energy"],
        )
        self.assertLessEqual(stats["maximum_feature_distance"], 1e-12)
        self.assertLessEqual(stats["maximum_corner_error"], 1e-12)
        self.assertEqual(stats["relaxed_constraint_count"], 0)
        np.testing.assert_array_equal(optimized_faces, faces)


if __name__ == "__main__":
    unittest.main()
