import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "simp_cuda"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pamo.original_constrained import retriangulate_brep_model_patches

PARTITION_PATH = PROJECT_ROOT / "partition_stl.py"
SPEC = importlib.util.spec_from_file_location(
    "_pamo_brep_partition_test", PARTITION_PATH
)
partition_stl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(partition_stl)


class BrepPartitionTest(unittest.TestCase):
    def test_unnamed_body_recovers_six_planes_and_two_fillets(self):
        mesh = trimesh.load(
            PROJECT_ROOT / "examples" / "Unnamed-Body.stl",
            force="mesh",
            process=True,
        )
        result = partition_stl.build_brep_model_partitions(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
        )

        self.assertIsNotNone(result)
        labels, stats = result
        self.assertEqual(stats["partition_method"], "brep-model-first")
        self.assertEqual(stats["patches"], 8)
        self.assertEqual(stats["plane_patches"], 6)
        self.assertEqual(stats["cylinder_patches"], 2)
        self.assertEqual(sorted(np.bincount(labels).tolist()), [2, 2, 2, 2, 64, 64, 66, 66])

    def test_export_contains_patch_id_on_vertices_and_faces(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            stats = partition_stl.partition_stl(
                PROJECT_ROOT / "examples" / "Unnamed-Body.stl",
                temporary_directory,
            )
            data = Path(stats["combined_path"]).read_bytes()
            header_end = data.index(b"end_header") + len(b"end_header")
            header = data[:header_end].decode("ascii")

            self.assertEqual(header.count("property int patch_id"), 2)
            self.assertIn("element face 268", header)
            manifest = Path(stats["manifest_path"]).read_text(
                encoding="utf-8-sig"
            )
            self.assertEqual(len(manifest.splitlines()), 9)
            self.assertIn("6,cylinder,64,patch_0006_64_faces.ply", manifest)
            self.assertIn("7,cylinder,64,patch_0007_64_faces.ply", manifest)

    def test_model_first_remesh_rebuilds_and_stitches_all_eight_patches(self):
        mesh = trimesh.load(
            PROJECT_ROOT / "examples" / "Unnamed-Body.stl",
            force="mesh",
            process=True,
        )
        result = retriangulate_brep_model_patches(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            maximum_edge_length=15.0,
            cylinder_target_edge_ratio=2.0,
            minimum_triangle_angle_degrees=28.0,
        )

        self.assertIsNotNone(result)
        vertices, faces, _, stats = result
        self.assertEqual(stats["patches"], 8)
        self.assertEqual(len(stats["output_patch_face_counts"]), 8)
        self.assertEqual(sum(stats["output_patch_face_counts"]), len(faces))
        self.assertEqual(
            stats["output_patch_face_counts"][6],
            stats["output_patch_face_counts"][7],
        )
        self.assertGreater(stats["boundary_splits"], 0)
        self.assertTrue(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).is_watertight
        )
        triangles = vertices[faces]
        edge_lengths = np.linalg.norm(
            triangles[:, (1, 2, 0)] - triangles[:, (0, 1, 2)], axis=2
        )
        self.assertLessEqual(float(edge_lengths.max()), 15.0 * (1.0 + 1e-8))
        for record in stats["patch_records"][6:]:
            self.assertGreater(record["new_quality"], 0.7)
            self.assertGreater(record["new_quality_p5"], 0.12)


if __name__ == "__main__":
    unittest.main()
