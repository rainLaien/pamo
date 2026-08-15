import importlib.util
from pathlib import Path
import unittest

import numpy as np
import trimesh

SDF_FIELD_PATH = Path(__file__).parents[1] / "pamo" / "sdf_field.py"
SPEC = importlib.util.spec_from_file_location("_pamo_sdf_field_test", SDF_FIELD_PATH)
SDF_FIELD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SDF_FIELD)
resolve_original_surface_mode = SDF_FIELD.resolve_original_surface_mode


class OriginalSurfaceModeTest(unittest.TestCase):
    def test_open_consistently_wound_mesh_preserves_boundary(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0),
                )
            ),
            faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
            process=False,
        )

        mode, reason = resolve_original_surface_mode(mesh, "exact")

        self.assertEqual(mode, "exact")
        self.assertIn("4 open boundary edge(s) preserved", reason)

    def test_repair_mode_is_rejected_for_original_constraints(self):
        mesh = trimesh.creation.box()

        with self.assertRaisesRegex(ValueError, "offset repair envelope"):
            resolve_original_surface_mode(mesh, "repair")


if __name__ == "__main__":
    unittest.main()
