"""Import a new triangle PLY/STL, partition it, then run surface remeshing."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid


_PARTITION_FILES = ('patch_result.ply', 'patch_report.json')
_PARTITION_CACHE_VERSION = 2


def _file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _partition_runtime(executable, env):
    """Fingerprint DLLs searched by the native importer and CUDA seed fitter."""
    directories = [executable.parent, Path(env.get('SystemRoot', 'C:/Windows')) / 'System32']
    directories.extend(Path(value.strip('"')) for value in env.get('PATH', '').split(os.pathsep)
                       if value)
    for name, value in env.items():
        if name.upper() == 'CUDA_PATH' or name.upper().startswith('CUDA_PATH_V'):
            directories.extend((Path(value) / 'bin', Path(value) / 'bin' / 'x64'))
    for ancestor in list(executable.parents)[:5]:
        directories.append(ancestor / '.venv' / 'Lib' / 'site-packages' / 'torch' / 'lib')
    cuda_root = Path(env.get('ProgramFiles', 'C:/Program Files')) / 'NVIDIA GPU Computing Toolkit' / 'CUDA'
    if cuda_root.is_dir():
        for version in cuda_root.iterdir():
            directories.extend((version / 'bin', version / 'bin' / 'x64'))
    files = set()
    explicit = env.get('CADMESH_NVRTC_DLL')
    if explicit and Path(explicit).is_file():
        files.add(Path(explicit).resolve())
        directories.append(Path(explicit).resolve().parent)
    for directory in set(directories):
        for pattern in ('libgcc_s_*.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll',
                        'nvrtc*.dll', 'nvcuda.dll'):
            files.update(path.resolve() for path in directory.glob(pattern) if path.is_file())
    return {str(path): _file_digest(path) for path in sorted(files)}


def _partition_cache_key(native_input, executable, options, env):
    # Hash the actual STL passed to the importer, including PLY conversion.
    # Remesh settings deliberately do not participate: native partitioning is
    # independent of the target edge length, projection, and optimization passes.
    signature = {
        'version': _PARTITION_CACHE_VERSION,
        'input_sha256': _file_digest(native_input),
        'executable_sha256': _file_digest(executable),
        'options': list(options),
        'environment': {name.upper(): value for name, value in env.items()
                        if name.upper().startswith(('CADMESH_', 'CUDA_'))
                        or name.upper() in ('PATH', 'SYSTEMROOT', 'PROGRAMFILES')},
        'runtime': _partition_runtime(executable, env),
    }
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode('utf-8')).hexdigest()


def _cached_partition_valid(entry, key):
    """Accept only complete, unchanged native exports; never deserialize pickle."""
    try:
        if entry.is_symlink() or not entry.is_dir():
            return False
        manifest_path = entry / 'manifest.json'
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 16384:
            return False
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (not isinstance(manifest, dict) or manifest.get('key') != key
                or manifest.get('version') != _PARTITION_CACHE_VERSION):
            return False
        records = manifest.get('files')
        if not isinstance(records, dict) or set(records) != set(_PARTITION_FILES):
            return False
        for name in _PARTITION_FILES:
            path, record = entry / name, records[name]
            if (path.is_symlink() or not path.is_file() or not isinstance(record, dict)
                    or path.stat().st_size != record.get('size')
                    or _file_digest(path) != record.get('sha256')):
                return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def _publish_partition_cache(partition, cache_root, key):
    """Publish complete exports together, retaining damaged entries for inspection."""
    cache_root.mkdir(parents=True, exist_ok=True)
    entry = cache_root / key
    if _cached_partition_valid(entry, key):
        return entry
    with tempfile.TemporaryDirectory(prefix='.pending-', dir=cache_root) as temporary:
        pending = Path(temporary) / 'complete'
        pending.mkdir()
        records = {}
        for name in _PARTITION_FILES:
            target = pending / name
            shutil.copy2(partition / name, target)
            records[name] = {'size': target.stat().st_size, 'sha256': _file_digest(target)}
        (pending / 'manifest.json').write_text(json.dumps({
            'version': _PARTITION_CACHE_VERSION, 'key': key, 'files': records,
        }, sort_keys=True), encoding='utf-8')
        if entry.exists():
            # A concurrent process may have published while these files copied.
            if _cached_partition_valid(entry, key):
                return entry
            if entry.is_symlink() or entry.resolve().parent != cache_root.resolve():
                raise OSError('Refusing to replace a partition cache link outside its directory.')
            entry.rename(cache_root / f'.invalid-{key}-{uuid.uuid4().hex}')
        pending.rename(entry)
    return entry


def _partition_with_cache(native_input, executable, partition, options, env, cache_root,
                          *, reuse_cached_files=False):
    start = time.perf_counter()
    key = None
    if cache_root is not None:
        try:
            key = _partition_cache_key(native_input, executable, options, env)
            entry = cache_root / key
            if _cached_partition_valid(entry, key):
                if not reuse_cached_files:
                    partition.mkdir()
                    for name in _PARTITION_FILES:
                        shutil.copy2(entry / name, partition / name)
                print(f'[partition] cache hit {key[:12]}; verified exports reused '
                      f'({time.perf_counter() - start:.2f}s)', flush=True)
                return entry if reuse_cached_files else partition
            reason = 'incomplete or damaged entry' if entry.exists() else 'no entry'
            print(f'[partition] cache miss {key[:12]} ({reason}); running native partitioner', flush=True)
        except OSError as error:
            # Caching is optional; read-only disks and unavailable runtime
            # fingerprint files must not prevent a normal partition run.
            key = None
            print(f'[partition] cache unavailable: {error}; running native partitioner', flush=True)
    else:
        print('[partition] cache disabled; running native partitioner', flush=True)
    subprocess.run([str(executable), str(native_input), str(partition), *options],
                   env=env, check=True)
    if key is not None:
        try:
            _publish_partition_cache(partition, cache_root, key)
        except OSError as error:
            print(f'[partition] cache was not saved: {error}', flush=True)
    print(f'[partition] native partition and cache: {time.perf_counter() - start:.2f}s', flush=True)
    return partition


def _ensure_segmenter(root, env):
    """Rebuild changed native sources before launching the partitioner."""
    build = root / 'build_mingw_release'
    executable = build / 'cad_mesh_segment.exe'
    inputs = [root / 'CMakeLists.txt']
    for folder in ('src', 'include', 'app'):
        inputs.extend((root / folder).rglob('*.cpp'))
        inputs.extend((root / folder).rglob('*.h'))
    if (executable.is_file()
            and all(path.stat().st_mtime_ns <= executable.stat().st_mtime_ns for path in inputs)):
        return executable
    if not (build / 'CMakeCache.txt').is_file():
        raise RuntimeError('Configure the native build with run_segmentation.ps1 first.')
    cmake = shutil.which('cmake')
    if cmake is None:
        bundled = Path('C:/Program Files/CMake/bin/cmake.exe')
        if bundled.is_file():
            cmake = str(bundled)
    if cmake is None:
        raise RuntimeError('CMake is required to rebuild the changed native partitioner.')
    print('[partition] Rebuilding changed native sources (cad_mesh_segment only)...', flush=True)
    subprocess.run([cmake, '--build', str(build), '--target', 'cad_mesh_segment', '--parallel'],
                   cwd=root.parent, env=env, check=True)
    if not executable.is_file():
        raise RuntimeError('Native build did not produce cad_mesh_segment.exe.')
    return executable


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-edge-length', type=float)
    parser.add_argument('--max-deviation', type=float,
                        help='Independent geometry error; default 0.00025 * input bbox diagonal.')
    parser.add_argument('--flip-passes', type=int, default=8)
    parser.add_argument('--batch-face-limit', type=int, default=75000,
                        help='Maximum source faces per CUDA batch (default 75000; comparison values: 50000 or 100000).')
    parser.add_argument('--analytic-seed-backend', choices=('auto', 'cpu', 'cuda'), default='cuda',
                        help='Native partition nonlinear seed fitting backend (default cuda; requires rebuilt segmenter).')
    parser.add_argument('--projection-backend', choices=('cpu', 'cuda'), default='cuda',
                        help='Projection backend for remesh; default is cuda for speed.')
    parser.add_argument('--max-normal-deviation-degrees', type=float, default=10.0,
                        help='Maximum output normal deviation angle in degrees (default 10).')
    parser.add_argument('--trace-surface-validation', action='store_true', default=False,
                        help='Enable expensive per-stage surface diagnostics; final validation always runs.')
    parser.add_argument('--no-partition-cache', action='store_true',
                        help='Run native partitioning again without reading or writing its content cache.')
    parser.add_argument('--full-output', action='store_true',
                        help='Keep the partition handoff and export remesh reports, STL and alternate PLY views; default only writes remesh_result.ply.')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() not in ('.ply', '.stl'):
        parser.error('Input must be an existing triangle PLY or STL file.')
    output = args.output.resolve()
    if output.exists():
        parser.error('Choose a new output directory to preserve existing results.')
    env = os.environ.copy()
    compilers = list(Path('D:/').glob('MinGW-W64*/*/bin/g++.exe'))
    if compilers:
        env['PATH'] = str(compilers[0].parent) + os.pathsep + env.get('PATH', '')
    try:
        executable = _ensure_segmenter(root, env)
    except (RuntimeError, OSError) as error:
        parser.error(str(error))
    # The native importer uses narrow paths. Stage input and native output in
    # an ASCII temporary directory, retaining Unicode paths in the Python API.
    with tempfile.TemporaryDirectory(prefix='cadmesh_') as temporary:
        stage = Path(temporary)
        native_input = stage / 'input.stl'
        if source.suffix.lower() == '.stl':
            shutil.copy2(source, native_input)
        else:
            import trimesh
            mesh = trimesh.load(source, force='mesh', process=False)
            if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                parser.error('PLY must contain mesh faces; point clouds require reconstruction first.')
            # ASCII STL avoids the float32 coordinate conversion of binary STL.
            native_input.write_text(trimesh.exchange.stl.export_stl_ascii(mesh), encoding='ascii')
        output.mkdir(parents=True)
        if not args.full_output:
            native_result = stage / 'remesh_result.ply'
            command = [str(executable), str(native_input), str(native_result),
                       '--analytic-seed-backend', args.analytic_seed_backend,
                       '--remesh', '--flip-passes', str(args.flip_passes),
                       '--max-normal-deviation-degrees',
                       str(args.max_normal_deviation_degrees),
                       '--target-mean-quality', '0.8']
            if args.target_edge_length is not None:
                command += ['--target-edge-length', str(args.target_edge_length)]
            if args.max_deviation is not None:
                command += ['--max-deviation', str(args.max_deviation)]
            if args.projection_backend == 'cuda':
                command.append('--require-remesh-cuda')
            print('[remesh-file] Running native in-memory partition and remesh...',
                  flush=True)
            returncode = subprocess.run(command, env=env).returncode
            if returncode == 0:
                shutil.copy2(native_result, output / 'remesh_result.ply')
        else:
            partition = _partition_with_cache(
                native_input, executable, stage / 'partition',
                ['--analytic-seed-backend', args.analytic_seed_backend,
                 '--remesh-handoff'], env,
                None if args.no_partition_cache else root / 'debug' / 'partition_cache',
                reuse_cached_files=True)
            shutil.copytree(partition, output / 'partition',
                            ignore=lambda _directory, names: [name for name in names if name not in _PARTITION_FILES])
            partition = output / 'partition'
            command = [sys.executable, str(root / 'remesh_partition.py'),
                       str(partition), '--output', str(output / 'remesh'),
                       '--projection-backend', args.projection_backend,
                       '--flip-passes', str(args.flip_passes),
                       '--batch-face-limit', str(args.batch_face_limit),
                       '--full-output']
            if args.target_edge_length is not None:
                command += ['--target-edge-length', str(args.target_edge_length)]
            if args.max_deviation is not None:
                command += ['--max-deviation', str(args.max_deviation)]
            if args.max_normal_deviation_degrees is not None:
                command += ['--max-normal-deviation-degrees',
                            str(args.max_normal_deviation_degrees)]
            if args.trace_surface_validation:
                command += ['--trace-surface-validation']
            returncode = subprocess.run(command, env=env).returncode
    print(f'[remesh-file] total wall time: {time.perf_counter() - started:.2f}s '
          f'(including partition, remesh, validation and export; exit {returncode})', flush=True)
    return returncode


if __name__ == '__main__':
    raise SystemExit(main())
