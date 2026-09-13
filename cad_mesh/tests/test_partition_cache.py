"""Native partition reuse never reuses remeshing or incomplete exports."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cad_mesh import remesh_file


class PartitionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'input.stl'
        self.source.write_bytes(b'stl input version 1')
        self.executable = self.root / 'segment.exe'
        self.executable.write_bytes(b'executable version 1')
        self.cache = self.root / 'cache'
        self.options = ['--analytic-seed-backend', 'cuda']
        self.env = {'PATH': str(self.root)}
        self.exports = {name: ('native ' + name).encode() for name in remesh_file._PARTITION_FILES}
        self.calls = 0
        self.log = io.StringIO()
        self.addCleanup(patch.stopall)
        patch.object(remesh_file, '_partition_runtime', return_value={}).start()
        self.native = patch.object(remesh_file.subprocess, 'run', side_effect=self.run_native).start()

    def run_native(self, command, **kwargs):
        self.calls += 1
        destination = Path(command[2])
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in self.exports.items():
            (destination / name).write_bytes(content)
        return SimpleNamespace(returncode=0)

    def run_partition(self, name, cache=True, options=None):
        destination = self.root / name
        with redirect_stdout(self.log):
            remesh_file._partition_with_cache(
                self.source, self.executable, destination, options or self.options,
                self.env, self.cache if cache else None)
        for filename, content in self.exports.items():
            self.assertEqual((destination / filename).read_bytes(), content)
        return destination

    def key(self):
        return remesh_file._partition_cache_key(self.source, self.executable, self.options, self.env)

    def test_hit_reuses_exact_exports_and_does_not_link_output_to_cache(self):
        first = self.run_partition('first')
        (first / 'patch_result.ply').write_bytes(b'user changed the first result')
        second = self.run_partition('second')
        (second / 'patch_report.json').write_bytes(b'user changed the second result')
        self.run_partition('third')
        self.assertEqual(self.calls, 1)
        self.assertIn('cache miss', self.log.getvalue())
        self.assertIn('cache hit', self.log.getvalue())
        self.assertTrue(remesh_file._cached_partition_valid(self.cache / self.key(), self.key()))

    def test_input_contents_invalidate_even_with_same_size(self):
        self.run_partition('first')
        self.source.write_bytes(b'stl input version 2')
        self.run_partition('changed')
        self.assertEqual(self.calls, 2)

    def test_read_only_cache_hit_does_not_copy_intermediate_files(self):
        self.run_partition('first')
        destination = self.root / 'unused_destination'
        with redirect_stdout(self.log), patch.object(remesh_file.shutil, 'copy2') as copy:
            path = remesh_file._partition_with_cache(
                self.source, self.executable, destination, self.options, self.env,
                self.cache, reuse_cached_files=True)
        self.assertEqual(path, self.cache / self.key())
        copy.assert_not_called()
        self.assertFalse(destination.exists())
        self.assertEqual(self.calls, 1)

    def test_executable_contents_invalidate_even_with_same_size(self):
        self.run_partition('first')
        self.executable.write_bytes(b'executable version 2')
        self.run_partition('changed')
        self.assertEqual(self.calls, 2)

    def test_backend_and_other_partition_options_invalidate(self):
        self.run_partition('cuda')
        self.run_partition('cpu', options=['--analytic-seed-backend', 'cpu'])
        self.run_partition('tolerance', options=self.options + ['--fit-tolerance-ratio', '0.001'])
        self.assertEqual(self.calls, 3)

    def test_runtime_or_relevant_environment_change_invalidates(self):
        self.run_partition('first')
        with patch.object(remesh_file, '_partition_runtime', return_value={'nvrtc.dll': 'new hash'}):
            self.run_partition('runtime')
        self.env['CUDA_VISIBLE_DEVICES'] = '1'
        self.run_partition('environment')
        self.assertEqual(self.calls, 3)

    def test_missing_or_corrupt_files_are_rebuilt_then_reused(self):
        self.run_partition('first')
        entry = self.cache / self.key()
        for index, mutation in enumerate(('missing', 'corrupt', 'json', 'record', 'key')):
            with self.subTest(mutation=mutation):
                if mutation == 'missing':
                    (entry / 'patch_report.json').unlink()
                elif mutation == 'corrupt':
                    path = entry / 'patch_result.ply'
                    path.write_bytes(b'x' * path.stat().st_size)
                elif mutation == 'json':
                    (entry / 'manifest.json').write_text('{incomplete', encoding='utf-8')
                else:
                    manifest = json.loads((entry / 'manifest.json').read_text())
                    if mutation == 'record':
                        manifest['files']['patch_result.ply'] = None
                    else:
                        manifest['key'] = 'wrong'
                    (entry / 'manifest.json').write_text(json.dumps(manifest))
                self.run_partition(f'rebuild{index}')
                self.assertEqual(self.calls, index + 2)
                self.run_partition(f'reuse{index}')
                self.assertEqual(self.calls, index + 2)
        self.assertIn('incomplete or damaged entry', self.log.getvalue())

    def test_incomplete_entry_without_manifest_is_rebuilt(self):
        entry = self.cache / self.key()
        entry.mkdir(parents=True)
        (entry / 'patch_result.ply').write_bytes(b'incomplete')
        self.run_partition('first')
        self.run_partition('second')
        self.assertEqual(self.calls, 1)

    def test_disabled_cache_runs_native_every_time_without_creating_cache(self):
        self.run_partition('first', cache=False)
        self.run_partition('second', cache=False)
        self.assertEqual(self.calls, 2)
        self.assertFalse(self.cache.exists())
        self.assertIn('cache disabled', self.log.getvalue())

    def test_failed_native_call_is_not_published(self):
        self.native.side_effect = subprocess.CalledProcessError(1, 'segment')
        with self.assertRaises(subprocess.CalledProcessError), redirect_stdout(self.log):
            self.run_partition('failed')
        self.assertFalse(self.cache.exists())

    def test_unwritable_cache_does_not_fail_partitioning(self):
        with patch.object(remesh_file, '_publish_partition_cache', side_effect=PermissionError('read-only')):
            self.run_partition('first')
        self.assertIn('cache was not saved', self.log.getvalue())

    def test_oversized_manifest_and_unexpected_paths_are_rejected(self):
        self.run_partition('first')
        entry, key = self.cache / self.key(), self.key()
        manifest = entry / 'manifest.json'
        manifest.write_text(' ' * 16385)
        self.assertFalse(remesh_file._cached_partition_valid(entry, key))
        manifest.write_text(json.dumps({'version': remesh_file._PARTITION_CACHE_VERSION,
                                        'key': key, 'files': {'../outside': {}}}))
        self.assertFalse(remesh_file._cached_partition_valid(entry, key))

    def test_cli_existing_output_remains_untouched(self):
        existing = self.root / 'existing'
        existing.mkdir()
        marker = existing / 'keep.json'
        marker.write_text('existing result')
        with self.assertRaises(SystemExit), redirect_stdout(self.log), \
                patch('sys.stderr', new=io.StringIO()), \
                patch.object(remesh_file, '_ensure_segmenter') as ensure:
            remesh_file.main([str(self.source), '--output', str(existing)])
        ensure.assert_not_called()
        self.native.assert_not_called()
        self.assertEqual(marker.read_text(), 'existing result')

    def test_cli_runs_remesh_again_when_target_changes_and_passes_disable(self):
        remesh_calls = []

        def run(command, **kwargs):
            if command[0] == str(self.executable):
                return self.run_native(command, **kwargs)
            remesh_calls.append(command)
            self.assertTrue(Path(command[2]).is_dir())
            return SimpleNamespace(returncode=0)

        self.native.side_effect = run
        with patch.object(remesh_file, '__file__', str(self.root / 'remesh_file.py')), \
                patch.object(remesh_file, '_ensure_segmenter', return_value=self.executable), \
                redirect_stdout(self.log):
            for index, target in enumerate(('12', '6', '3')):
                args = [str(self.source), '--output', str(self.root / f'output{index}'),
                        '--target-edge-length', target]
                if index == 2:
                    args.append('--no-partition-cache')
                self.assertEqual(remesh_file.main(args), 0)
        self.assertEqual(self.calls, 2)
        self.assertEqual(len(remesh_calls), 3)
        for command, target in zip(remesh_calls, ('12.0', '6.0', '3.0')):
            self.assertEqual(command[command.index('--target-edge-length') + 1], target)
        self.assertIn('total wall time:', self.log.getvalue())
        for index in range(3):
            self.assertFalse((self.root / f'output{index}' / 'partition').exists())

    def test_full_output_retains_only_the_two_handoff_files(self):
        commands = []

        def run(command, **kwargs):
            if command[0] == str(self.executable):
                self.assertIn('--remesh-handoff', command)
                return self.run_native(command, **kwargs)
            commands.append(command)
            self.assertTrue(Path(command[2]).is_dir())
            return SimpleNamespace(returncode=0)

        self.native.side_effect = run
        output = self.root / 'full_output'
        with patch.object(remesh_file, '__file__', str(self.root / 'remesh_file.py')), \
                patch.object(remesh_file, '_ensure_segmenter', return_value=self.executable), \
                redirect_stdout(self.log):
            self.assertEqual(remesh_file.main([str(self.source), '--output', str(output),
                                               '--full-output']), 0)
        self.assertEqual({p.name for p in (output / 'partition').iterdir()},
                         set(remesh_file._PARTITION_FILES))
        self.assertIn('--full-output', commands[0])


if __name__ == '__main__':
    unittest.main()
