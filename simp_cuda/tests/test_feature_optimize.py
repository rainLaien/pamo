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


@unittest.skipIf(
    feature_optimize is None,
    "Feature optimization dependencies are unavailable",
)
class FeatureOptimizeTest(unittest.TestCase):
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
