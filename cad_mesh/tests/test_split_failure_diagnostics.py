"""Batch failures distinguish fixed geometry from a depleted split budget."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from cad_mesh.remesh_pipeline import _run_cuda_batches


class SplitFailureDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.labels = np.zeros(2, dtype=np.int64)
        self.constraints = np.array([[0, 1], [0, 3], [1, 2], [2, 3]])
        self.surface = SimpleNamespace(surface_sample_remesh=Mock())
        # This batch orchestration test must fail before any GPU-only work.
        self.torch = SimpleNamespace(float64=None, long=None,
                                     as_tensor=lambda value, **kwargs: np.asarray(value))
        self.trimesh = SimpleNamespace(Trimesh=lambda **kwargs: SimpleNamespace(**kwargs))

    def run_batch(self, target):
        return _run_cuda_batches(
            self.surface, self.torch, self.trimesh, self.vertices, self.faces,
            self.labels, self.labels, self.constraints, np.array([], dtype=np.int64),
            target, 10, 0, 16, 0, 0, 0, .01,
        )

    def test_overlong_fixed_interface_fails_before_remesher_is_called(self):
        with self.assertRaisesRegex(RuntimeError, "4 fixed interface edges over target"):
            self.run_batch(.5)
        self.surface.surface_sample_remesh.assert_not_called()

    def test_geometric_rejection_keeps_final_length_check_and_explains_blocker(self):
        self.surface.surface_sample_remesh.return_value = (
            self.vertices.copy(), self.faces.copy(),
            {"face_patch_ids": self.labels.copy(),
             "split_diagnostics": {"stop_reason": "geometric_rejections"}},
        )
        with self.assertRaisesRegex(RuntimeError, "increasing --split-passes alone cannot resolve"):
            self.run_batch(1.1)
        self.surface.surface_sample_remesh.assert_called_once()

    def test_exhausted_budget_keeps_actionable_pass_advice(self):
        self.surface.surface_sample_remesh.return_value = (
            self.vertices.copy(), self.faces.copy(),
            {"face_patch_ids": self.labels.copy(),
             "split_diagnostics": {"stop_reason": "pass_limit"}},
        )
        with self.assertRaisesRegex(RuntimeError, "budget was exhausted; increase --split-passes"):
            self.run_batch(1.1)

    def test_exhausted_budget_does_not_hide_other_blocked_edges(self):
        self.surface.surface_sample_remesh.return_value = (
            self.vertices.copy(), self.faces.copy(),
            {"face_patch_ids": self.labels.copy(), "split_diagnostics": {
                "stop_reason": "pass_limit", "eligible_long_edges": 2,
                "rejected_long_edges": 1, "protected_long_edges": 0,
                "nonmanifold_long_edges": 0,
            }},
        )
        with self.assertRaises(RuntimeError) as failure:
            self.run_batch(1.1)
        self.assertIn("eligible=2, rejected=1, fixed=0, nonmanifold=0", str(failure.exception))
        self.assertIn("Blocked edges also remain", str(failure.exception))


if __name__ == '__main__':
    unittest.main()
