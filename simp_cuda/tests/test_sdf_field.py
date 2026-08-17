import importlib.util
from pathlib import Path
import unittest

import numpy as np
import trimesh


SDF_FIELD_PATH = Path(__file__).parents[1] / "pamo" / "sdf_field.py"
spec = importlib.util.spec_from_file_location(
    "_pamo_sdf_field_test",
    SDF_FIELD_PATH,
)
sdf_field = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdf_field)


class OriginalConstraintModeTest(unittest.TestCase):
    def setUp(self):
        self.open_sheet = trimesh.Trimesh(
            vertices=np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
                dtype=np.float64,
            ),
            faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
            process=False,
        )

    def test_open_sheet_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "allow_open_surface"):
            sdf_field.resolve_original_constraint_mode(
                self.open_sheet,
                "exact",
            )

    def test_open_sheet_is_accepted_with_hard_boundaries(self):
        mode, reason = sdf_field.resolve_original_constraint_mode(
            self.open_sheet,
            "exact",
            allow_open_surface=True,
        )

        self.assertEqual(mode, "open")
        self.assertIn("4 hard boundary edge", reason)
        self.assertIn("0 hard non-manifold edge", reason)

    def test_repair_envelope_is_never_accepted(self):
        with self.assertRaisesRegex(ValueError, "repair envelope"):
            sdf_field.resolve_original_constraint_mode(
                self.open_sheet,
                "repair",
                allow_open_surface=True,
            )


if __name__ == "__main__":
    unittest.main()
