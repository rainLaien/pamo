"""Import a new triangle PLY/STL, partition it, then run surface remeshing."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-edge-length', type=float)
    parser.add_argument('--max-deviation', type=float,
                        help='Independent geometry error; default 0.00025 * input bbox diagonal.')
    parser.add_argument('--flip-passes', type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    executable = root / 'build_mingw_release/cad_mesh_segment.exe'
    if not executable.is_file():
        parser.error('Build cad_mesh_segment.exe with run_segmentation.ps1 first.')
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
        subprocess.run([str(executable), str(native_input), str(stage / 'partition')],
                       env=env, check=True)
        output.mkdir(parents=True)
        shutil.copytree(stage / 'partition', output / 'partition')
    command = [sys.executable, str(root / 'remesh_partition.py'), str(output / 'partition'),
               '--output', str(output / 'remesh'), '--projection-backend', 'cuda',
               '--flip-passes', str(args.flip_passes)]
    if args.target_edge_length is not None:
        command += ['--target-edge-length', str(args.target_edge_length)]
    if args.max_deviation is not None:
        command += ['--max-deviation', str(args.max_deviation)]
    return subprocess.run(command, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
