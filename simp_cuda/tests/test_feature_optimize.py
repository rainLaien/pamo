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
                vertices, faces, minimum_faces=20
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

        self.assertEqual(stats["regions"], 1)
        self.assertGreater(float(np.percentile(qualities, 5.0)), 0.7)
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

    def test_automatic_edge_limit_uses_ten_percent_diagonal(self):
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
        self.assertAlmostEqual(limit, 100.0)

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
