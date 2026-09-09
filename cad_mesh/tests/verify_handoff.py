"""Independently verify an exported JSON/PLY remesh handoff (standard library).

Usage: python cad_mesh/tests/verify_handoff.py <segmentation-output-directory>

Properties are read by name. Mesh and patch incidence checks are linear in
faces, edges and patch references; auxiliary coloring views are checked too.
"""
from array import array
import collections
import json
import math
from pathlib import Path
import sys


def reject_nonfinite(value):
    raise AssertionError('non-standard JSON numeric value: ' + value)


def read_ply(path):
    """Stream ASCII PLY into compact arrays rather than retaining text/dicts."""
    result = dict(vertices=array('d'), faces=array('i'), patch_ids=array('i'),
                  primitive_types=array('i'), feature_roles=array('i'),
                  colors=array('B'), has_roles=False)
    with path.open(encoding='ascii') as stream:
        assert stream.readline().strip() == 'ply', f'{path}: missing PLY header'
        elements, ascii_format = [], False
        for raw in stream:
            tokens = raw.split()
            if not tokens or tokens[0] in ('comment', 'obj_info'):
                continue
            if tokens[0] == 'format':
                ascii_format = tokens[1:] == ['ascii', '1.0']
            elif tokens[0] == 'element':
                elements.append((tokens[1], int(tokens[2]), []))
            elif tokens[0] == 'property':
                assert elements, f'{path}: property before element'
                elements[-1][2].append((tokens[-1], tokens[1] == 'list'))
            elif tokens[0] == 'end_header':
                break
        else:
            raise AssertionError(f'{path}: unterminated PLY header')
        assert ascii_format, f'{path}: expected ASCII PLY 1.0'
        assert [name for name, _, _ in elements] == ['vertex', 'face']
        for name, count, properties in elements:
            names = [prop for prop, _ in properties]
            assert len(names) == len(set(names)), f'{path}: duplicate properties'
            if name == 'vertex':
                assert all(prop in names for prop in ('x', 'y', 'z'))
            else:
                assert all(prop in names for prop in
                           ('patch_id', 'primitive_type', 'red', 'green', 'blue'))
                assert 'vertex_indices' in names or 'vertex_index' in names
                result['has_roles'] = 'feature_role' in names
            for _ in range(count):
                tokens = stream.readline().split()
                values, cursor = {}, 0
                for prop, is_list in properties:
                    assert cursor < len(tokens), f'{path}: short {name} record'
                    if is_list:
                        size = int(tokens[cursor])
                        assert size >= 0 and cursor + size < len(tokens)
                        values[prop] = tuple(map(int, tokens[cursor + 1:cursor + 1 + size]))
                        cursor += size + 1
                    else:
                        values[prop] = tokens[cursor]
                        cursor += 1
                assert cursor == len(tokens), f'{path}: excess {name} properties'
                if name == 'vertex':
                    point = tuple(float(values[prop]) for prop in ('x', 'y', 'z'))
                    assert all(map(math.isfinite, point)), f'{path}: non-finite vertex'
                    result['vertices'].extend(point)
                else:
                    ids = values.get('vertex_indices', values.get('vertex_index'))
                    assert len(ids) == 3, f'{path}: non-triangular face'
                    result['faces'].extend(ids)
                    result['patch_ids'].append(int(values['patch_id']))
                    result['primitive_types'].append(int(values['primitive_type']))
                    result['feature_roles'].append(int(values.get('feature_role', 0)))
                    rgb = tuple(int(values[prop]) for prop in ('red', 'green', 'blue'))
                    assert all(0 <= value <= 255 for value in rgb)
                    result['colors'].extend(rgb)
        assert not stream.read().strip(), f'{path}: trailing PLY records'
    return result


def unique_ids(values, name):
    assert all(isinstance(value, int) and not isinstance(value, bool) for value in values), name
    result = set(values)
    assert len(result) == len(values), f'{name}: duplicate ids'
    return result


def verify(root):
    report = json.loads((root / 'patch_report.json').read_text(),
                        parse_constant=reject_nonfinite)
    mesh = read_ply(root / 'patch_result.ply')
    vertices, faces = mesh['vertices'], mesh['faces']
    patch_ids = mesh['patch_ids']
    vertex_count, face_count = len(vertices) // 3, len(faces) // 3
    assert face_count == report['mesh']['triangle_count']
    assert vertex_count == report['mesh']['vertex_count']
    assert report['cleanup']['output_triangles'] == face_count
    assert report['partition_valid'] is True
    patches = report['patches']
    patch_count = len(patches)
    assert [patch['id'] for patch in patches] == list(range(patch_count))
    surface_types = dict(Unknown=0, Plane=1, Cylinder=2, Cone=3,
                         Sphere=4, Torus=5, Freeform=6)
    role_ids = dict(Ordinary=0, Fillet=1)
    diagnostics = report.get('diagnostics')
    if diagnostics is not None:
        for name in ('differential_geometry', 'boundary_scores'):
            assert isinstance(diagnostics[name]['computed'], bool)
        if not diagnostics['differential_geometry']['computed']:
            for filename in ('mean_curvature.vtk', 'gaussian_curvature.vtk', 'k1.vtk', 'k2.vtk'):
                assert not (root / filename).exists(), 'uncomputed curvature retains a stale diagnostic file'
    has_roles = any('feature_role' in patch for patch in patches)
    if has_roles:
        assert mesh['has_roles'], 'semantic JSON requires feature_role in PLY'
    memberships = bytearray(face_count)
    for patch in patches:
        assert patch['type'] in surface_types
        assert patch['triangle_count'] == len(patch['triangle_ids'])
        assert patch['triangle_ids'], 'empty patch'
        sampled = patch.get('sampled_mesh_deviation')
        if sampled is not None:
            assert isinstance(sampled['computed'], bool)
            assert sampled['hausdorff_upper_bound'] is False
            if sampled['computed']:
                assert patch['projection_target'] == 'analytic_surface'
                assert math.isfinite(sampled['maximum']) and sampled['maximum'] >= 0
                assert sampled['maximum'] + max(1e-15, 1e-12 * sampled['maximum']) >= patch['max']
                assert sampled['direction'] == 'reference_mesh_to_analytic_surface'
            else:
                assert sampled['maximum'] is None, 'uncomputed mesh deviation must be null'
        if has_roles:
            role = patch['feature_role']
            assert role in role_ids, f"patch {patch['id']}: invalid feature role"
            supports = unique_ids(patch['support_patch_ids'], 'support_patch_ids')
            assert all(0 <= other < patch_count and other != patch['id'] for other in supports)
            assert supports <= set(patch['neighbors']), 'fillet support is not adjacent'
            if role == 'Ordinary':
                assert not supports, 'ordinary patch retains stale fillet supports'
            else:
                assert len(supports) == 2, 'confirmed fillet requires two mother surfaces'
                assert patch['type'] in ('Cylinder', 'Torus', 'Freeform')
        for triangle_id in patch['triangle_ids']:
            assert 0 <= triangle_id < face_count
            assert not memberships[triangle_id], 'triangle belongs to multiple patches'
            memberships[triangle_id] = 1
            assert patch_ids[triangle_id] == patch['id']
            assert mesh['primitive_types'][triangle_id] == surface_types[patch['type']]
            if has_roles:
                assert mesh['feature_roles'][triangle_id] == role_ids[patch['feature_role']]
    assert all(memberships), 'triangle has no patch owner'

    edge_incidence = collections.defaultdict(list)
    for triangle_id in range(face_count):
        a, b, c = faces[3 * triangle_id:3 * triangle_id + 3]
        assert all(0 <= vertex < vertex_count for vertex in (a, b, c))
        assert len({a, b, c}) == 3, 'repeated triangle vertex'
        for first, second in ((a, b), (b, c), (c, a)):
            key = min(first, second) * vertex_count + max(first, second)
            edge_incidence[key].append(triangle_id)
    assert len(edge_incidence) == report['mesh']['edge_count']
    constraints = report['constraints']
    edges = {edge['id']: edge for edge in report['constraint_edges']}
    assert len(edges) == len(report['constraint_edges'])
    edge_ids = unique_ids(constraints['edge_ids'], 'constraint edge ids')
    assert edge_ids == edges.keys()
    by_patch_edges = [set() for _ in patches]
    by_patch_chains = [set() for _ in patches]
    by_patch_neighbors = [set() for _ in patches]
    adjacency_edges = collections.defaultdict(set)
    exported_edge_keys = set()
    has_boundary_kinds = ('hard_feature_edge_ids' in constraints or
                          'smooth_surface_transition_edge_ids' in constraints)
    if has_boundary_kinds:
        hard = unique_ids(constraints['hard_feature_edge_ids'], 'hard feature edges')
        smooth = unique_ids(constraints['smooth_surface_transition_edge_ids'], 'smooth transitions')
        assert not hard & smooth, 'hard and smooth edge categories overlap'
        assert hard | smooth == edge_ids, 'hard/smooth categories do not cover all shared edges'
    for edge_id, edge in edges.items():
        first, second = sorted(edge['vertex_ids'])
        key = first * vertex_count + second
        assert key in edge_incidence and key not in exported_edge_keys
        exported_edge_keys.add(key)
        incidence = edge_incidence[key]
        assert sorted(edge['incident_triangle_ids']) == sorted(incidence)
        owners = sorted({patch_ids[tid] for tid in incidence})
        assert edge['incident_patch_ids'] == owners
        assert edge['open_boundary'] == (len(incidence) == 1)
        assert edge['non_manifold'] == (len(incidence) > 2)
        if diagnostics and not diagnostics['boundary_scores']['computed']:
            assert edge['boundary_score'] is None, 'uncomputed boundary score must be null'
        for owner in owners:
            by_patch_edges[owner].add(edge_id)
            by_patch_neighbors[owner].update(other for other in owners if other != owner)
        for index, owner in enumerate(owners):
            for other in owners[index + 1:]:
                adjacency_edges[(owner, other)].add(edge_id)
        if has_boundary_kinds:
            is_hard = edge['open_boundary'] or edge['non_manifold'] or edge['constrained_feature']
            assert edge['hard_feature'] is is_hard
            assert (edge_id in hard) == is_hard
            if edge_id in smooth:
                assert len(incidence) == 2 and len(owners) == 2, 'smooth transition needs two patch sides'
    for key, incidence in edge_incidence.items():
        if len(incidence) != 2 or patch_ids[incidence[0]] != patch_ids[incidence[1]]:
            assert key in exported_edge_keys, 'mesh/patch boundary missing from shared constraints'
    del edge_incidence
    observed_adjacency = set()
    for adjacency in report['adjacency']:
        pair = (adjacency['patch0'], adjacency['patch1'])
        assert pair not in observed_adjacency and pair in adjacency_edges
        observed_adjacency.add(pair)
        assert adjacency['edge_count'] == len(adjacency['shared_boundary_edge_ids'])
        assert unique_ids(adjacency['shared_boundary_edge_ids'], 'adjacency edges') == adjacency_edges[pair]
        if diagnostics and not diagnostics['boundary_scores']['computed']:
            assert adjacency['confidence'] is None, 'uncomputed boundary confidence must be null'
    assert observed_adjacency == adjacency_edges.keys(), 'missing patch adjacency'

    chains = constraints['boundary_chains']
    assert [chain['id'] for chain in chains] == list(range(len(chains)))
    chain_edges = collections.Counter(edge_id for chain in chains for edge_id in chain['edge_ids'])
    assert chain_edges == collections.Counter(edges.keys())
    for chain in chains:
        ids = chain['vertex_ids']
        assert chain['edge_ids'], 'empty boundary chain'
        assert len(ids) == len(chain['edge_ids']) + 1
        assert chain['closed'] == (ids[0] == ids[-1])
        assert chain['initial_sample_vertex_ids'] == ids
        assert all(0 <= vertex < vertex_count for vertex in ids)
        owners = unique_ids(chain['incident_patch_ids'], 'chain incident patches')
        assert all(0 <= owner < patch_count for owner in owners)
        for owner in owners:
            by_patch_chains[owner].add(chain['id'])
        if has_boundary_kinds:
            assert isinstance(chain['hard_feature'], bool)
            expected_kind = 'hard_feature' if chain['hard_feature'] else 'smooth_surface_transition'
            assert chain['boundary_kind'] == expected_kind
        for i, edge_id in enumerate(chain['edge_ids']):
            edge = edges[edge_id]
            assert sorted(ids[i:i + 2]) == sorted(edge['vertex_ids'])
            assert edge['incident_patch_ids'] == chain['incident_patch_ids']
            if has_boundary_kinds:
                assert edge['hard_feature'] == chain['hard_feature'], 'chain crosses boundary category'

    for patch in patches:
        patch_id = patch['id']
        assert unique_ids(patch['neighbors'], 'patch neighbors') == by_patch_neighbors[patch_id]
        if has_roles:
            assert set(patch['support_patch_ids']) <= by_patch_neighbors[patch_id], 'mother surface has no shared boundary'
        assert unique_ids(patch['boundary_edge_ids'], 'patch boundary edges') == by_patch_edges[patch_id]
        refs = {ref['chain_id']: ref for ref in patch['boundary_chain_refs']}
        assert len(refs) == len(patch['boundary_chain_refs'])
        assert set(refs) == by_patch_chains[patch_id]
        for chain_id, ref in refs.items():
            chain = chains[chain_id]
            if ref['orientation_status'] != 'consistent':
                assert ref['direction'] == 0, 'unresolved chain has invented direction'
                continue
            assert ref['direction'] in (-1, 1)
            for i, edge_id in enumerate(chain['edge_ids']):
                start_vertex, end_vertex = chain['vertex_ids'][i:i + 2]
                incident = [tid for tid in edges[edge_id]['incident_triangle_ids']
                            if patch_ids[tid] == patch_id]
                assert len(incident) == 1, 'oriented chain lacks a unique patch side'
                tid = incident[0]
                a, b, c = faces[3 * tid:3 * tid + 3]
                directed = ((a, b), (b, c), (c, a))
                expected = ((start_vertex, end_vertex) if ref['direction'] > 0
                            else (end_vertex, start_vertex))
                assert expected in directed, 'chain disagrees with incident triangle winding'

    checked_views = []
    for filename in ('surface_types.ply', 'feature_roles.ply'):
        path = root / filename
        if path.exists():
            view = read_ply(path)
            for field in ('vertices', 'faces', 'patch_ids', 'primitive_types', 'feature_roles', 'has_roles'):
                assert view[field] == mesh[field], f'{filename}: {field} differs from instance view'
            checked_views.append(filename)
            del view

    return dict(directory=str(root), vertices=vertex_count, triangles=face_count,
                patches=patch_count,
                patch_types=dict(collections.Counter(patch['type'] for patch in patches)),
                feature_roles=dict(collections.Counter(patch.get('feature_role', 'unavailable') for patch in patches)),
                constraint_edges=len(edges),
                hard_feature_edges=len(hard) if has_boundary_kinds else None,
                smooth_surface_transition_edges=len(smooth) if has_boundary_kinds else None,
                boundary_chains=len(chains), corners=len(constraints['corner_vertex_ids']),
                matching_semantic_views=checked_views, result='PASS')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    if not __debug__:
        raise SystemExit("Run this verifier without Python's -O option.")
    print(json.dumps(verify(Path(sys.argv[1])), indent=2))
