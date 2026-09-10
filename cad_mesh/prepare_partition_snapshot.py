"""Package an exported partition for native remeshing; standard library only.

This does not weld, fit, segment, resample, or change any saved indices.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index(value):
    require(type(value) is int and 0 <= value < 2**31, 'Invalid nonnegative int32 index')
    return value


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid numeric value')
    return float(value)


def vector(value):
    require(isinstance(value, list) and len(value) == 3, 'Expected three coordinates')
    return [finite(x) for x in value]


def prepare(directory, destination):
    report = json.loads((directory / 'patch_report.json').read_text(encoding='utf-8-sig'))
    require(report.get('schema') == 'cadmesh.remesh_handoff' and report.get('schema_version') == 1,
            'Expected the compact export produced by --remesh-handoff')
    require(report.get('partition_valid') is True and report['indexing']['base'] == 0,
            'Expected a valid zero-based partition')
    nv = index(report['mesh']['vertex_count'])
    nf = index(report['mesh']['triangle_count'])
    patches = report['patches']
    constraints = report['constraint_edges']
    require(nv > 0 and nf > 0 and 0 < len(patches) <= nf, 'Empty or invalid partition')
    with (directory / 'patch_result.ply').open('rb') as source:
        lines = []
        while True:
            line = source.readline(4097)
            require(line and len(line) <= 4096 and len(lines) < 100, 'Invalid PLY header')
            text = line.decode('ascii').strip()
            if not text.startswith('comment '):
                lines.append(text)
            if text == 'end_header':
                break
        expected = ['ply', 'format binary_little_endian 1.0', f'element vertex {nv}',
                    'property double x', 'property double y', 'property double z',
                    f'element face {nf}', 'property list uchar int vertex_indices',
                    'property int patch_id', 'property int primitive_type', 'property uchar red',
                    'property uchar green', 'property uchar blue', 'property int feature_role', 'end_header']
        require(lines == expected, 'PLY layout does not match the native binary handoff export')
        require(os.fstat(source.fileno()).st_size - source.tell() == nv * 24 + nf * 28,
                'PLY size disagrees with JSON counts')
        vertices = source.read(nv * 24)
        faces = source.read(nf * 28)
    require(all(math.isfinite(x) for row in struct.iter_unpack('<3d', vertices) for x in row),
            'Non-finite PLY coordinate')
    labels, types, roles = [], [], []
    packed_faces = bytearray(nf * 16)
    for i, row in enumerate(struct.iter_unpack('<BiiiiiBBBi', faces)):
        count, a, b, c, label, kind, red, green, blue, role = row
        require(count == 3 and all(0 <= v < nv for v in (a, b, c)) and len({a,b,c}) == 3,
                'Invalid triangle in saved PLY')
        require(0 <= label < len(patches), 'Invalid PLY patch id')
        labels.append(label); types.append(kind); roles.append(role)
        struct.pack_into('<4I', packed_faces, i * 16, a, b, c, label)
    seen = bytearray(nf)
    patch_records = bytearray()
    type_ids = {name: i for i, name in enumerate(('Unknown','Plane','Cylinder','Cone','Sphere','Torus','Freeform'))}
    for pid, patch in enumerate(patches):
        require(index(patch['id']) == pid, 'Patch ids must retain exported order')
        kind = type_ids[patch['type']]
        require(patch['projection_target'] in ('analytic_surface', 'reference_mesh'), 'Invalid projection target')
        analytic = int(patch['projection_target'] == 'analytic_surface')
        require(patch['feature_role'] in ('Ordinary', 'Fillet'), 'Invalid feature role')
        role = int(patch['feature_role'] == 'Fillet')
        ids = patch['triangle_ids']
        require(len(ids) == index(patch['triangle_count']) and ids, 'Invalid patch triangle count')
        for face in ids:
            face = index(face)
            require(face < nf and not seen[face] and labels[face] == pid and
                    types[face] == kind and roles[face] == role, 'PLY/JSON patch ownership mismatch')
            seen[face] = 1
        supports = [index(x) for x in patch['support_patch_ids']]
        require(all(x < len(patches) for x in supports), 'Invalid support patch id')
        params = [0.] * 8
        if analytic:
            p = patch['parameters']
            if kind == 1:
                params[:3] = vector(p['origin']); params[3:6] = vector(p['normal'])
            elif kind == 4:
                params[:3] = vector(p['center']); params[6] = finite(p['radius'])
            elif kind in (2, 3, 5):
                params[:3] = vector(p['axis_origin']); params[3:6] = vector(p['axis_direction'])
                params[6] = finite(p[{2:'radius',3:'semi_angle_radians',5:'major_radius'}[kind]])
                if kind == 5: params[7] = finite(p['minor_radius'])
            else:
                raise ValueError('Unsupported analytic patch')
        deviation = finite(patch.get('max_sampled_surface_deviation', -1))
        require(deviation >= 0 or deviation == -1, 'Invalid sampled surface deviation')
        patch_records += struct.pack('<5I', kind, analytic, role, len(ids), len(supports))
        patch_records += struct.pack(f'<{len(supports)}I', *supports)
        patch_records += struct.pack('<9d', *params, deviation)
    require(all(seen), 'JSON omits PLY triangles')
    edge_records = bytearray()
    saved_ids = []
    for edge in constraints:
        eid = index(edge['id']); ends = edge['vertex_ids']
        require(len(ends) == 2 and all(index(x) < nv for x in ends), 'Invalid constraint endpoints')
        require(type(edge['hard_feature']) is bool, 'Invalid constraint flag')
        edge_records += struct.pack('<4I', eid, *ends, int(edge['hard_feature']))
        saved_ids.append(eid)
    require(len(set(saved_ids)) == len(saved_ids) and
            sorted(saved_ids) == sorted(report['constraints']['edge_ids']), 'Constraint id mismatch')
    resolution = report['resolution']
    values = [finite(resolution[name]) for name in ('bbox_diagonal','median_edge','weld_tolerance',
              'fitting_tolerance','curvature_tolerance','angular_tolerance')]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as out:
            temporary = Path(out.name)
            out.write(b'CADPART1')
            out.write(struct.pack('<4I6d', nv, nf, len(patches), len(constraints), *values))
            for block in (vertices, packed_faces, patch_records, edge_records): out.write(block)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('partition', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    prepare(args.partition.resolve(), args.output.resolve())
    print(f'[remesh-input] Saved partition packaged in {time.perf_counter()-start:.3f}s: {args.output}', flush=True)
