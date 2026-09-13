"""Bounded binary PLY handoff reader; geometry validation remains in remesh_io."""
import os
from pathlib import Path

import numpy as np


_SCALARS = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': '<i2', 'int16': '<i2', 'ushort': '<u2', 'uint16': '<u2',
    'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'uint32': '<u4',
    'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
}
_MAX_HEADER_BYTES = 1024 * 1024
_MAX_HEADER_LINE = 65536


def _header(stream):
    """Return binary schema, or None to retain the existing ASCII reader."""
    if stream.readline(_MAX_HEADER_LINE + 1).strip() != b'ply':
        raise ValueError('Expected a PLY header.')
    elements, encoding = [], None
    while stream.tell() <= _MAX_HEADER_BYTES:
        raw = stream.readline(_MAX_HEADER_LINE + 1)
        if not raw:
            raise ValueError('Unterminated PLY header.')
        if len(raw) > _MAX_HEADER_LINE:
            raise ValueError('PLY header line exceeds the supported length.')
        if stream.tell() > _MAX_HEADER_BYTES:
            raise ValueError('PLY header exceeds the supported length.')
        tokens = raw.decode('ascii').split()
        if not tokens or tokens[0] in ('comment', 'obj_info'):
            continue
        if tokens[0] == 'format':
            if encoding is not None or elements or len(tokens) != 3 or tokens[2] != '1.0':
                raise ValueError('Invalid or repeated PLY format declaration.')
            encoding = tokens[1]
            if encoding == 'ascii':
                return None
            if encoding != 'binary_little_endian':
                raise ValueError('Partition PLY requires ASCII or binary_little_endian 1.0.')
        elif tokens[0] == 'element':
            if encoding is None or len(tokens) != 3 or not tokens[2].isdecimal():
                raise ValueError('Invalid PLY element count or declaration.')
            count = int(tokens[2])
            if count > np.iinfo(np.intp).max:
                raise ValueError('PLY element count exceeds the supported index range.')
            elements.append((tokens[1], count, []))
        elif tokens[0] == 'property' and elements:
            if len(tokens) == 3 and tokens[1] in _SCALARS:
                prop = (tokens[2], _SCALARS[tokens[1]], None)
            elif (len(tokens) == 5 and tokens[1] == 'list'
                  and tokens[2] in _SCALARS and tokens[3] in _SCALARS):
                prop = (tokens[4], _SCALARS[tokens[3]], _SCALARS[tokens[2]])
                if np.dtype(prop[2]).kind not in 'iu':
                    raise ValueError('PLY list count must have an integer type.')
            else:
                raise ValueError('Invalid or unsupported PLY property declaration.')
            if any(existing[0] == prop[0] for existing in elements[-1][2]):
                raise ValueError(f'PLY {elements[-1][0]} contains duplicate properties.')
            elements[-1][2].append(prop)
        elif tokens == ['end_header']:
            if encoding is None:
                raise ValueError('Missing PLY format declaration.')
            if [element[0] for element in elements] != ['vertex', 'face']:
                raise ValueError('Partition PLY must contain vertex then face elements.')
            return elements
        else:
            raise ValueError('Unsupported PLY header declaration.')
    raise ValueError('PLY header exceeds the supported length.')


def _record_layout(properties, *, vertices):
    fields, names, count_name = [], {}, None
    indices = [name for name, _, _ in properties if name in ('vertex_indices', 'vertex_index')]
    if not vertices and len(indices) != 1:
        raise ValueError('PLY faces need exactly one vertex_indices list property.')
    index_name = indices[0] if indices else None
    for index, (name, dtype, count_type) in enumerate(properties):
        field = f'p{index}'
        names[name] = field
        if count_type is not None:
            if vertices or name != index_name:
                raise ValueError('Binary PLY supports only the triangle vertex_indices list.')
            if np.dtype(dtype).kind not in 'iu':
                raise ValueError('PLY triangle indices must have an integer type.')
            count_name = field + '_count'
            fields.extend(((count_name, count_type), (field, dtype, (3,))))
        else:
            if not vertices and name in ('patch_id', 'primitive_type', 'feature_role'):
                if np.dtype(dtype).kind not in 'iu':
                    raise ValueError(f'PLY {name} must have an integer type.')
            fields.append((field, dtype))
    required = ('x', 'y', 'z') if vertices else ('patch_id', 'primitive_type')
    if any(name not in names for name in required):
        raise ValueError('PLY vertices need x, y, and z properties.' if vertices
                         else 'PLY faces need patch_id and primitive_type properties.')
    if not vertices and count_name is None:
        raise ValueError('PLY faces need a vertex_indices list property.')
    return np.dtype(fields), names, index_name, count_name


def read_binary_partition_ply(path):
    """Read exact fixed triangle records, returning None for legacy ASCII.

    Header counts must match the physical payload before any mesh-sized array
    allocation. The declared list count is still checked in every face record;
    no producer marker or cache manifest substitutes for data validation.
    """
    with Path(path).open('rb') as stream:
        elements = _header(stream)
        if elements is None:
            return None
        (_, vertex_count, vertex_properties), (_, face_count, face_properties) = elements
        vertex_dtype, vertex_names, _, _ = _record_layout(vertex_properties, vertices=True)
        face_dtype, face_names, index_name, count_name = _record_layout(face_properties, vertices=False)
        expected_bytes = vertex_count * vertex_dtype.itemsize + face_count * face_dtype.itemsize
        remaining_bytes = os.fstat(stream.fileno()).st_size - stream.tell()
        if remaining_bytes != expected_bytes:
            raise ValueError('Binary PLY payload is truncated or contains trailing records beyond its declared counts.')
        vertex_records = np.fromfile(stream, dtype=vertex_dtype, count=vertex_count)
        face_records = np.fromfile(stream, dtype=face_dtype, count=face_count)
        if len(vertex_records) != vertex_count or len(face_records) != face_count or stream.read(1):
            raise ValueError('Binary PLY payload changed or is truncated.')
    if np.any(face_records[count_name] != 3):
        raise ValueError('Remeshing requires triangle faces; PLY contains a polygon.')
    vertices = np.column_stack([vertex_records[vertex_names[axis]] for axis in ('x', 'y', 'z')]).astype(np.float64)
    faces = face_records[face_names[index_name]].astype(np.int64)
    labels = face_records[face_names['patch_id']].astype(np.int64)
    types = face_records[face_names['primitive_type']].astype(np.int64)
    has_roles = 'feature_role' in face_names
    roles = (face_records[face_names['feature_role']].astype(np.int64) if has_roles
             else np.zeros(face_count, dtype=np.int64))
    return vertices, faces, labels, types, roles, has_roles
