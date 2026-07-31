import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


PAMO_PATH = Path(__file__).parents[1] / "pamo"

try:
    import scipy  # noqa: F401
    import trimesh  # noqa: F401
except ImportError:
    split_feature_edges_only = None
else:
    package_name = "_pamo_feature_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PAMO_PATH)]
    sys.modules[package_name] = package

    for module_name in ("segment_query", "feature_edges"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            PAMO_PATH / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)

    split_feature_edges_only = sys.modules[
        f"{package_name}.feature_edges"
    ].split_feature_edges_only


@unittest.skipIf(
    split_feature_edges_only is None,
    "Feature-edge dependencies are unavailable",
)
class FeatureEdgeSplitTest(unittest.TestCase):
    def test_refines_only_selected_edge_and_preserves_surface_topology(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        reference_segments = np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        )

        refined_vertices, refined_faces, stats = split_feature_edges_only(
            vertices,
            faces,
            np.array([[0, 1]], dtype=np.int64),
            reference_segments,
            target_length=0.3,
        )

        self.assertEqual(stats["splits"], 3)
        self.assertFalse(stats["hit_split_limit"])
        self.assertLessEqual(stats["max_length"], 0.3)
        self.assertEqual(refined_vertices.shape, (7, 3))
        self.assertEqual(refined_faces.shape, (5, 3))
        np.testing.assert_allclose(refined_vertices[4:, 1:], 0.0)
        self.assertEqual(
            len(np.unique(np.sort(refined_faces, axis=1), axis=0)),
            len(refined_faces),
        )


if __name__ == "__main__":
    unittest.main()
