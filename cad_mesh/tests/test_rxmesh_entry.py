"""CUDA integration checks. Run with the built cad_mesh_rxmesh executable."""
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
from collections import Counter


def read_ply(path):
    with path.open('rb') as f:
        nv = nf = 0
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex '): nv = int(line.split()[-1])
            if line.startswith('element face '): nf = int(line.split()[-1])
            if line == 'end_header': break
            if not line: raise AssertionError('Incomplete PLY header')
        vertices = [struct.unpack('<3d', f.read(24)) for _ in range(nv)]
        faces = []
        for _ in range(nf):
            n, a, b, c, region = struct.unpack('<B4i', f.read(17))
            assert n == 3 and len({a,b,c}) == 3
            assert all(0 <= i < nv for i in (a,b,c))
            faces.append((a,b,c))
        assert not f.read(1)
    assert len({tuple(sorted(t)) for t in faces}) == len(faces)
    return vertices, faces


def write_stl(path, triangles):
    with path.open('wb') as f:
        f.write(b'RXMesh integration fixture'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(triangles)))
        for tri in triangles:
            f.write(struct.pack('<12fH', 0, 0, 0, *sum((list(p) for p in tri), []), 0))


def grid(n):
    result = []
    for y in range(n):
        for x in range(n):
            a, b, c, d = [(xx/n, yy/n, 0) for xx, yy in
                          [(x, y), (x+1, y), (x+1, y+1), (x, y+1)]]
            result.extend([(a, b, c), (a, c, d)])
    return result


def cube():
    p = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),
         (0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    fs = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
          (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    return [tuple(p[i] for i in t) for t in fs]


def nonmanifold_tetrahedra():
    faces = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
    return [tuple(points[i] for i in t)
            for points in [[(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
                           [(0,0,0),(1,0,0),(0,-1,0),(0,0,-1)]] for t in faces]


def main():
    exe = pathlib.Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix='pamo_rxmesh_') as temp:
        root = pathlib.Path(temp)
        for name, triangles, ratio in [('cube', cube(), '.15'), ('grid', grid(48), '.08'),
                                       ('nonmanifold', nonmanifold_tetrahedra(), '.15')]:
            source, output = root/(name+'.stl'), root/name
            write_stl(source, triangles)
            run = subprocess.run([str(exe), str(source), str(output),
                                  '--target-edge-ratio', ratio, '--max-deviation', '.001',
                                  '--iterations', '3'], capture_output=True, text=True, timeout=120)
            print(run.stdout, run.stderr)
            assert run.returncode == 0, name
            report = json.loads((output/'rxmesh_report.json').read_text())
            assert report['sampled_shape_and_topology_passed'], report
            assert report['missing_constraint_edges'] == report['moved_constraint_vertices'] == 0
            assert report['output_faces'] != len(triangles), 'No topology changes'
            vertices, faces = read_ply(output/'rxmesh_result.ply')
            incidence = Counter(tuple(sorted((t[k],t[(k+1)%3]))) for t in faces for k in range(3))
            if name == 'cube':
                assert report['boundary_edges'] == 0
                assert report['source_regions'] == 6
                assert report['boundary_splits'] > 0
                assert all(n == 2 for n in incidence.values()), 'Crack in closed cube'
                assert all(any(all(abs(vertices[v][axis]-side)<1e-8 for v in t)
                               for axis in range(3) for side in (0,1)) for t in faces), 'Cube face moved off its source plane'
            elif name == 'grid':
                assert report['output_faces'] < len(triangles), 'Dense plane was not coarsened'
                assert all(abs(p[2]) < 1e-8 for p in vertices)
                assert all(n in (1,2) for n in incidence.values())
            else:
                assert report['input_nonmanifold_edges'] == 1
                assert report['manifold_vertex_copies'] >= 2
                assert all(n == 2 for n in incidence.values()), 'Separated tetrahedra should remain closed'
            rerun = subprocess.run([str(exe), str(source), str(output)], capture_output=True)
            assert rerun.returncode != 0, 'Existing output must not be overwritten'
        bad = subprocess.run([str(exe), str(source), str(root/'bad'),
                              '--target-edge-ratio', 'garbage'], capture_output=True)
        assert bad.returncode != 0
    print('RXMesh integration checks passed')


if __name__ == '__main__':
    main()
