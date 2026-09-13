"""Locate small-angle faces in a generic binary PLY; no third-party packages.

Default: constraint-adjacent faces only. --ordinary includes unresolved ordinary
faces too. Constraint adjacency is a reason to inspect, not proof of necessity.
"""
import argparse
import csv
from pathlib import Path
import struct


def export(source, destination, ordinary=False):
    with source.open('rb') as mesh, destination.open('w', newline='') as output:
        header = []
        while True:
            line = mesh.readline()
            if not line:
                raise ValueError('Incomplete PLY header')
            header.append(line.strip())
            if line.strip() == b'end_header':
                break
        if b'format binary_little_endian 1.0' not in header or b'property uchar issue_flags' not in header:
            raise ValueError('Expected generic_result.ply from cad_mesh_generic')
        nv = int(next(line for line in header if line.startswith(b'element vertex ')).split()[-1])
        nf = int(next(line for line in header if line.startswith(b'element face ')).split()[-1])
        vertices = [struct.unpack('<ddd', mesh.read(24)) for _ in range(nv)]
        writer = csv.writer(output)
        writer.writerow(['output_face_id', 'center_x', 'center_y', 'center_z',
                         'minimum_angle_deg', 'constraint_adjacent', 'geometry_rejection_history', 'status'])
        count = 0
        for face_id in range(nf):
            n, a, b, c, angle, flags = struct.unpack('<BiiifB', mesh.read(18))
            if n != 3:
                raise ValueError('Expected triangle')
            if not flags & 1 or (not ordinary and not flags & 2):
                continue
            center = [sum(vertices[v][k] for v in (a, b, c)) / 3 for k in range(3)]
            writer.writerow([face_id, *center, angle, bool(flags & 2), bool(flags & 4),
                             'constraint_adjacent_requires_review' if flags & 2 else 'ordinary_unresolved'])
            count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ply', type=Path)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--ordinary', action='store_true')
    args = parser.parse_args()
    print(f'Exported {export(args.ply, args.csv, args.ordinary)} face locations to {args.csv}')
