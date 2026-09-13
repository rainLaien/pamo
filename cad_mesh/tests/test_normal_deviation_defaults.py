"""Normal-budget defaults propagate through entrypoints without repartitioning."""
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cad_mesh import remesh_file, remesh_pipeline, surface_rebuild

partition_cli = importlib.import_module("cad_mesh.remesh_partition")
CAD_MESH = Path(__file__).resolve().parents[1]


class NormalDeviationDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        self.faces = np.array([[0, 1, 2]], dtype=np.int64)
        self.log = io.StringIO()

    def test_file_cli_default_and_explicit_normal_reuse_same_native_partition(self):
        source, executable = self.root / "source.stl", self.root / "segment.exe"
        source.write_bytes(b"same STL input")
        executable.write_bytes(b"same partitioner")
        native_calls, remesh_commands = [], []

        def run(command, **kwargs):
            if command[0] == str(executable):
                native_calls.append(command)
                destination = Path(command[2])
                destination.mkdir()
                for name in remesh_file._PARTITION_FILES:
                    (destination / name).write_bytes(b"unchanged partition export")
            else:
                remesh_commands.append(command)
            return SimpleNamespace(returncode=0)

        with patch.object(remesh_file, "__file__", str(self.root / "remesh_file.py")), \
                patch.object(remesh_file, "_ensure_segmenter", return_value=executable), \
                patch.object(remesh_file, "_partition_runtime", return_value={}), \
                patch.object(remesh_file.subprocess, "run", side_effect=run), redirect_stdout(self.log):
            for index, options in enumerate(([], ["--max-normal-deviation-degrees", "5"])):
                self.assertEqual(remesh_file.main([
                    str(source), "--output", str(self.root / f"output{index}"), *options]), 0)
        self.assertEqual(len(native_calls), 1)
        self.assertIn("cache hit", self.log.getvalue())
        self.assertEqual([float(c[c.index("--max-normal-deviation-degrees") + 1])
                          for c in remesh_commands], [10., 5.])

    def test_partition_cli_and_public_api_forward_default_and_explicit_normal(self):
        source = SimpleNamespace(source_directory=self.root / "partition",
                                 vertices=self.vertices, faces=self.faces)
        result = SimpleNamespace(faces=self.faces, stats={"input_patch_count": 1, "output_patch_count": 1})
        with patch.object(partition_cli, "load_partition", return_value=source), \
                patch.object(partition_cli, "write_remesh_result"), \
                patch.object(surface_rebuild, "rebuild_surfaces", return_value=result) as rebuild, \
                redirect_stdout(self.log):
            for options in ([], ["--max-normal-deviation-degrees", "5"]):
                self.assertEqual(partition_cli.main([str(source.source_directory), *options]), 0)
            remesh_pipeline.remesh_partition(source, target_edge_length=12.)
            remesh_pipeline.remesh_partition(source, target_edge_length=12.,
                                             maximum_normal_deviation_degrees=5.)
        self.assertEqual([call.kwargs["maximum_normal_deviation_degrees"]
                          for call in rebuild.call_args_list], [10., 5., 10., 5.])

    def test_batch_wrapper_preserves_default_and_explicit_normal_without_gpu(self):
        calls = []

        def remesh(*args, **kwargs):
            calls.append(kwargs["maximum_normal_deviation_degrees"])
            return self.vertices.copy(), self.faces.copy(), dict(
                face_patch_ids=np.array([0]), sample_count=1, splits=0, collapses=0,
                flips=0, remaining_long_edges=0)

        surface = SimpleNamespace(surface_sample_remesh=remesh)
        torch = SimpleNamespace(as_tensor=Mock(), float64="float64", long="long")
        trimesh = SimpleNamespace(Trimesh=Mock())
        with redirect_stdout(self.log):
            for options in ({}, {"maximum_normal_deviation_degrees": 5.}):
                remesh_pipeline._run_cuda_batches(
                    surface, torch, trimesh, self.vertices, self.faces,
                    np.array([0]), np.array([0]), np.empty((0, 2), dtype=np.int64),
                    np.empty(0, dtype=np.int64), target=12., sample_count=1, seed=0,
                    split_passes=1, collapse_passes=0, flip_passes=0, relax_iterations=0,
                    deviation=.1, collect_diagnostics=False, **options)
        self.assertEqual(calls, [10., 5.])

    @unittest.skipUnless(os.name == "nt" and shutil.which("powershell"), "Windows PowerShell entrypoints")
    def test_powershell_entrypoints_forward_default_and_explicit_normal(self):
        capture = self.root / "capture.ps1"
        capture.write_text(
            "[System.IO.File]::WriteAllText($env:CADMESH_TEST_ARGS, "
            "(ConvertTo-Json -InputObject @($args) -Compress))\n$global:LASTEXITCODE = 0\n",
            encoding="utf-8")
        # remsh has a fixed interpreter path. Redirect only that executable in
        # an isolated script copy; execute its actual parameter and argv logic.
        remsh = self.root / "remsh.ps1"
        text = (CAD_MESH / "remsh.ps1").read_text(encoding="utf-8")
        assignment = "$pythonExecutable = Join-Path $projectRoot '.venv/Scripts/python.exe'"
        self.assertEqual(text.count(assignment), 1)
        text = text.replace(assignment, "$pythonExecutable = '" + str(capture).replace("'", "''") + "'")
        remsh.write_text(text, encoding="utf-8")
        for script, extra in ((remsh, ["-OutputDirectory", str(self.root / "remsh-output")]),
                              (CAD_MESH / "run_remesh.ps1", ["-PartitionDirectory", str(self.root),
                                                            "-PythonExecutable", str(capture)])):
            for expected, options in ((10., []), (5., ["-MaxNormalDeviationDegrees", "5"])):
                with self.subTest(script=script.name, normal=expected):
                    output = self.root / "arguments.json"
                    environment = dict(os.environ, CADMESH_TEST_ARGS=str(output))
                    completed = subprocess.run([
                        shutil.which("powershell"), "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-File", str(script), *extra, *options], env=environment,
                        capture_output=True, text=True, timeout=20)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    arguments = json.loads(output.read_text(encoding="utf-8-sig"))
                    self.assertEqual(float(arguments[arguments.index("--max-normal-deviation-degrees") + 1]), expected)


if __name__ == "__main__":
    unittest.main()
