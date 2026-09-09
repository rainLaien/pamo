import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


PAMO_PATH = Path(__file__).parents[1] / "pamo"
PACKAGE_NAME = "_pamo_semantic_partition_test"
PACKAGE = types.ModuleType(PACKAGE_NAME)
PACKAGE.__path__ = [str(PAMO_PATH)]
sys.modules[PACKAGE_NAME] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.semantic_partition",
    PAMO_PATH / "semantic_partition.py",
)
semantic_partition = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = semantic_partition
SPEC.loader.exec_module(semantic_partition)


def _extruded_cross_section(cross_section, rows=5):
    cross_section = np.asarray(cross_section, dtype=np.float64)
    y_values = np.linspace(0.0, 4.0, rows)
    vertices = np.asarray(
        [
            (point[0], y, point[1])
            for y in y_values
            for point in cross_section
        ],
        dtype=np.float64,
    )
    columns = len(cross_section)
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = lower_left + columns
            upper_right = upper_left + 1
            faces.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return vertices, np.asarray(faces, dtype=np.int64)


class SemanticPartitionTest(unittest.TestCase):
    def test_flat_surface_remains_one_region(self):
        vertices, faces = _extruded_cross_section(
            [(-2.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
        )
        result = semantic_partition.build_semantic_partitions(
            vertices,
            faces,
            feature_angle_degrees=15.0,
            curvature_gradient_degrees=1.0,
        )

        self.assertEqual(result.stats["partitions"], 1)
        self.assertEqual(result.stats["gradient_edges"], 0)
        self.assertEqual(result.region_types, ("plane",))

    def test_curvature_gradient_splits_tangent_plane_fillet_contacts(self):
        cross_section = [(-2.0, 0.0), (-1.0, 0.0), (0.0, 0.0)]
        cross_section.extend(
            (
                np.cos(np.deg2rad(angle)),
                1.0 + np.sin(np.deg2rad(angle)),
            )
            for angle in range(-80, 1, 10)
        )
        cross_section.extend(((1.0, 2.0), (1.0, 3.0)))
        vertices, faces = _extruded_cross_section(cross_section)

        result = semantic_partition.build_semantic_partitions(
            vertices,
            faces,
            feature_angle_degrees=15.0,
            curvature_gradient_degrees=1.0,
            minimum_region_faces=4,
        )

        self.assertEqual(result.stats["crease_edges"], 0)
        self.assertGreater(result.stats["gradient_edges"], 0)
        self.assertEqual(result.stats["partitions"], 3)
        self.assertEqual(result.region_types.count("plane"), 2)
        self.assertEqual(result.region_types.count("cylinder"), 1)


if __name__ == "__main__":
    unittest.main()
