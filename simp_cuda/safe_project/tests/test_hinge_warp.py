from pathlib import Path
import sys
import unittest

import numpy as np


PACKAGE_SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

try:
    import warp as wp

    from pamo_safe_project.energy import HingeEnergyCalculator
except (ImportError, OSError):
    wp = None
    HingeEnergyCalculator = None


@unittest.skipIf(wp is None, "Warp safe-projection dependencies are unavailable")
class HingeWarpTest(unittest.TestCase):
    def test_preprocesses_mesh_above_old_cuda_grid_limit(self):
        wp.init()
        try:
            device = wp.get_device("cuda:0")
        except (AssertionError, RuntimeError):
            self.skipTest("CUDA device is unavailable")

        n_u = 190
        n_v = 180
        u, v = np.meshgrid(
            np.arange(n_u) * (2.0 * np.pi / n_u),
            np.arange(n_v) * (2.0 * np.pi / n_v),
            indexing="ij",
        )
        vertices = np.stack(
            (
                (2.0 + 0.5 * np.cos(v)) * np.cos(u),
                (2.0 + 0.5 * np.cos(v)) * np.sin(u),
                0.5 * np.sin(v),
            ),
            axis=-1,
        ).reshape(-1, 3).astype(np.float32)

        i, j = np.meshgrid(np.arange(n_u), np.arange(n_v), indexing="ij")
        a = i * n_v + j
        b = ((i + 1) % n_u) * n_v + j
        c = ((i + 1) % n_u) * n_v + (j + 1) % n_v
        d = i * n_v + (j + 1) % n_v
        faces = np.stack(
            (
                np.stack((a, b, c), axis=-1),
                np.stack((a, c, d), axis=-1),
            ),
            axis=2,
        ).reshape(-1, 3).astype(np.int32)
        self.assertGreater(faces.shape[0], 65535)

        class Config:
            pass

        config = Config()
        config.max_particles = vertices.shape[0]

        class System:
            pass

        system = System()
        system.config = config
        system.device = device
        system.n_edges = faces.shape[0] * 3 // 2
        system.q_rest = wp.array(vertices, dtype=wp.vec3, device=device)

        calculator = HingeEnergyCalculator(system)
        calculator.preprocess(vertices, faces)
        wp.synchronize_device(device)

        self.assertEqual(calculator.n_hinges, system.n_edges)
        rest_lengths = calculator.rest_elens.numpy()[: calculator.n_hinges]
        self.assertTrue(np.all(np.isfinite(rest_lengths)))
        self.assertTrue(np.all(rest_lengths > 0.0))


if __name__ == "__main__":
    unittest.main()
