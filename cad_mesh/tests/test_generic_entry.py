"""Black-box regression tests for the independent native generic entry."""
import collections
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

EXE = Path(__file__).resolve().parents[1] / 'win/Release/cad_mesh_generic.exe'

def write_stl(path, vertices, faces):
    with path.open('w') as f:
        f.write('solid test\n')
        for t in faces:
            f.write('facet normal 0 0 0\nouter loop\n')
            for i in t:
                f.write('vertex %s %s %s\n' % tuple(vertices[i]))
            f.write('endloop\nendfacet\n')
        f.write('endsolid test\n')

def topology(path):
    with path.open('rb') as f:
        nv = nf = 0
        for line in f:
            if line.startswith(b'element vertex '): nv = int(line.split()[-1])
            if line.startswith(b'element face '): nf = int(line.split()[-1])
            if line.strip() == b'end_header': break
        vertices = [struct.unpack('<ddd', f.read(24)) for _ in range(nv)]
        edges = collections.Counter()
        for _ in range(nf):
            n, a, b, c, angle, flags = struct.unpack('<BiiifB', f.read(18))
            assert n == 3 and len({a,b,c}) == 3
            assert all(0 <= i < nv for i in (a,b,c))
            for x,y in ((a,b),(b,c),(c,a)): edges[tuple(sorted((x,y)))] += 1
        assert not f.read(1)
    return vertices, edges

class GenericEntryTests(unittest.TestCase):
    def run_mesh(self, vertices, faces, target=.6, analytic=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name); source = root/'input.stl'; output = root/'output'
        write_stl(source, vertices, faces)
        command = [str(EXE), str(source), str(output), '--target', str(target),
                   '--deviation', '.02', '--iterations', '3']
        if analytic:
            command += ['--analytic-guides', '1']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('classification=disabled', result.stderr)
        report = json.loads((output/'generic_report.json').read_text())
        self.assertTrue(report['sampled_shape_audit_passed'])
        self.assertFalse((output/'generic_candidate.ply').exists())
        points, edges = topology(output/'generic_result.ply')
        self.assertEqual(report['source_feature_segments'], report['preserved_feature_segments'])
        self.assertLess(report['maximum_feature_vertex_distance'], 1e-8)
        self.assertEqual(report['output']['degenerate'], 0)
        self.assertLessEqual(max(report['sampled_forward_distance'], report['sampled_reverse_distance']), .02)
        return report, points, edges

    def test_cylinder_sliver_band(self):
        n = 32; levels = [0,.01,1,4]
        p = [(math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n), z) for z in levels for i in range(n)]
        faces = []
        for row in range(len(levels)-1):
            for i in range(n):
                a=row*n+i; b=row*n+(i+1)%n; c=a+n; d=b+n
                faces.extend([(a,b,c),(c,b,d)])
        p.extend([(0,0,0),(0,0,4)])
        for i in range(n): faces.extend([(len(p)-2,(i+1)%n,i),(len(p)-1,3*n+i,3*n+(i+1)%n)])
        report, _, edges = self.run_mesh(p, faces)
        self.assertTrue(all(count == 2 for count in edges.values()))
        self.assertLess(report['output']['below_28_percent'], report['input']['below_28_percent'])

    def test_nonmanifold_connection_is_retained(self):
        p = [(0,0,0),(1,0,0),(.5,1,0),(.5,-1,0),(.5,0,1)]
        report, _, edges = self.run_mesh(p, [(0,1,2),(1,0,3),(0,1,4)], target=6)
        self.assertEqual(report['output']['faces'], 3)
        self.assertEqual(sum(count == 3 for count in edges.values()), 1)

    def test_open_planar_strip(self):
        xs = [0,.001,.5,1,2]
        p = [(x,y,0) for y in (0,.5,1,2) for x in xs]
        faces = []
        for j in range(3):
            for i in range(4):
                a=j*5+i; b=a+1; c=a+5; d=c+1
                faces.extend([(a,b,c),(c,b,d)])
        _, points, edges = self.run_mesh(p, faces)
        self.assertTrue(all(count <= 2 for count in edges.values()))
        self.assertTrue(all(abs(point[2]) < 1e-12 for point in points))
        self.assertGreater(sum(count == 1 for count in edges.values()), 0)

    def test_nonmanifold_adjacent_long_edges_are_not_split(self):
        p = [(0,0,0),(1,0,0),(.5,1,0),(.5,-1,0),(.5,0,1)]
        report, _, edges = self.run_mesh(p, [(0,1,2),(1,0,3),(0,1,4)], target=.1)
        self.assertEqual(report['output']['faces'], 3)
        self.assertEqual(report['output_nonmanifold_edges'], 1)
        self.assertEqual(sum(count == 3 for count in edges.values()), 1)

    def test_analytic_guides_cylinder_and_cone(self):
        for taper in (0, .15):
            with self.subTest(taper=taper):
                n = 48
                p = [((1+taper*z)*math.cos(2*math.pi*i/n),
                      (1+taper*z)*math.sin(2*math.pi*i/n), z)
                     for z in (0, 2, 4) for i in range(n)]
                faces = []
                for row in range(2):
                    for i in range(n):
                        a=row*n+i; b=row*n+(i+1)%n; c=a+n; d=b+n
                        faces.extend([(a,b,c),(c,b,d)])
                side_faces = len(faces)
                p.extend([(0,0,0),(0,0,4)])
                for i in range(n):
                    faces.extend([(len(p)-2,(i+1)%n,i),
                                  (len(p)-1,2*n+i,2*n+(i+1)%n)])
                report, _, edges = self.run_mesh(p, faces, analytic=True)
                self.assertTrue(report['analytic_guides_enabled'])
                self.assertGreater(report['analytic_cones' if taper else 'analytic_cylinders'], 0)
                self.assertGreater(report['analytic_projection_proposals'], 0)
                self.assertLessEqual(report['analytic_supported_faces'], side_faces)
                self.assertTrue(all(count == 2 for count in edges.values()))

    def test_analytic_guides_do_not_invent_a_curved_plane(self):
        p = [(x,y,0) for y in range(4) for x in range(4)]
        faces = []
        for j in range(3):
            for i in range(3):
                a=j*4+i
                faces.extend([(a,a+1,a+4),(a+4,a+1,a+5)])
        report, _, _ = self.run_mesh(p, faces, analytic=True)
        self.assertEqual(report['analytic_supported_faces'], 0)

if __name__ == '__main__':
    unittest.main()
