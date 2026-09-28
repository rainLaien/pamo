"""Report source patches whose triangles were retained, including round planar faces."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from audit_phase1 import read_ply
from repair_narrow_strips import read_snapshot


def circular_plane_patches(points, records, patches):
    owners = records[:, 3].astype(np.int64)
    order = np.argsort(owners, kind='stable')
    starts = np.searchsorted(owners[order], np.arange(len(patches) + 1))
    found = []
    for patch_id, (header, _) in enumerate(patches):
        if header[0] != 1 or header[3] < 12:
            continue
        faces = records[order[starts[patch_id]:starts[patch_id + 1]], :3].astype(np.int64)
        directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        edges = np.sort(directed, axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = unique[counts == 1]
        if len(boundary) < 10:
            continue
        vertices, degree = np.unique(boundary, return_counts=True)
        if not np.all(degree == 2):
            continue
        tri = points[faces]
        area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        perimeter = np.linalg.norm(points[boundary[:, 0]] - points[boundary[:, 1]], axis=1).sum()
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if .9 <= circularity <= 1.02:
            adjacency = {}
            for a, b in boundary:
                adjacency.setdefault(int(a), []).append(int(b))
                adjacency.setdefault(int(b), []).append(int(a))
            loop = [int(boundary[0, 0])]
            previous = -1
            current = loop[0]
            for _ in range(len(boundary) - 1):
                neighbors = adjacency[current]
                following = neighbors[0] if neighbors[0] != previous else neighbors[1]
                loop.append(following)
                previous, current = current, following
            if len(set(loop)) != len(boundary) or loop[0] not in adjacency[current]:
                continue
            polygon = points[loop]
            normal = np.cross(tri[0, 1] - tri[0, 0], tri[0, 2] - tri[0, 0])
            normal /= max(np.linalg.norm(normal), 1e-30)
            segments = np.roll(polygon, -1, axis=0) - polygon
            turns = np.cross(segments, np.roll(segments, -1, axis=0)) @ normal
            convex = bool(np.all(turns >= -1e-5) or np.all(turns <= 1e-5))
            found.append(dict(patch_id=patch_id, source_faces=int(len(faces)),
                              circularity=float(circularity), convex_boundary=convex))
    return found


def audit(mesh_path, snapshot_path):
    _, points, records, patches, _ = read_snapshot(snapshot_path)
    output_points, output_faces, labels = read_ply(mesh_path)
    if len(labels) != len(output_faces) or np.any(labels >= len(patches)):
        raise ValueError('output patch labels do not match source')
    detail = json.loads(Path(str(mesh_path) + '.json').read_text())
    task_retained = set()
    task_width = int(detail['patches_per_task'])
    failed = []
    for task in detail['patches']:
        task_retained.update(int(x) for x in task.get('unchanged_patch_ids', []))
        if not task['accepted']:
            first = int(task['id']) * task_width
            ids = list(range(first, min(first + task_width, len(patches))))
            task_retained.update(ids)
            failed.append(dict(task_id=int(task['id']), patch_ids=ids, reason=task.get('error', '')))
    source_counts = np.bincount(records[:, 3], minlength=len(patches))
    output_counts = np.bincount(labels.astype(np.int64), minlength=len(patches))
    source_tri = points[records[:, :3].astype(np.int64)]
    source_total_area = float((.5 * np.linalg.norm(np.cross(source_tri[:, 1] - source_tri[:, 0],
                                                              source_tri[:, 2] - source_tri[:, 0]), axis=1)).sum())
    source_order = np.argsort(records[:, 3], kind='stable')
    output_order = np.argsort(labels, kind='stable')
    source_starts = np.searchsorted(records[source_order, 3], np.arange(len(patches) + 1))
    output_starts = np.searchsorted(labels[output_order], np.arange(len(patches) + 1))
    source_float = points.astype(np.float32)
    output_float = output_points.astype(np.float32)

    def face_keys(vertices, faces):
        tri = vertices[faces]
        permutation = np.lexsort((tri[:, :, 2], tri[:, :, 1], tri[:, :, 0]), axis=1)
        canonical = np.take_along_axis(tri, permutation[:, :, None], axis=1)
        return np.sort(np.ascontiguousarray(canonical.reshape(-1, 9)).view('V36').reshape(-1))

    retained = set()
    for patch_id in range(len(patches)):
        if source_counts[patch_id] != output_counts[patch_id]:
            continue
        source_faces = records[source_order[source_starts[patch_id]:source_starts[patch_id + 1]], :3]
        produced_faces = output_faces[output_order[output_starts[patch_id]:output_starts[patch_id + 1]]]
        if np.array_equal(face_keys(source_float, source_faces), face_keys(output_float, produced_faces)):
            retained.add(patch_id)
    circular = circular_plane_patches(points, records, patches)
    circular_retained = [p for p in circular if p['patch_id'] in retained]
    circular_details = []
    for item in circular:
        patch_id = item['patch_id']
        source_faces = records[source_order[source_starts[patch_id]:source_starts[patch_id + 1]], :3].astype(np.int64)
        produced_faces = output_faces[output_order[output_starts[patch_id]:output_starts[patch_id + 1]]].astype(np.int64)
        def interior_count(faces):
            edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
            unique, counts = np.unique(edges, axis=0, return_counts=True)
            boundary = np.unique(unique[counts == 1])
            return int(len(np.unique(faces)) - len(boundary))
        circular_details.append(dict(patch_id=patch_id, source_faces=int(len(source_faces)),
                                     output_faces=int(len(produced_faces)),
                                     source_interior_vertices=interior_count(source_faces),
                                     output_interior_vertices=interior_count(produced_faces),
                                     circularity=item['circularity'],
                                     remesh_skip_reason=('nonconvex contour; preserved by safety guard' if not item['convex_boundary'] else
                                                         'interior candidate did not pass safety guard')
                                     if interior_count(produced_faces)==0 else ''))
    planar_retained = [int(i) for i in sorted(retained) if patches[i][0][0] == 1]
    failed_reason = {patch_id: item['reason'] for item in failed for patch_id in item['patch_ids']}
    retained_regions = []
    for patch_id in sorted(retained):
        source_faces = records[source_order[source_starts[patch_id]:source_starts[patch_id + 1]], :3].astype(np.int64)
        tri = points[source_faces]
        areas = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        region_points = points[np.unique(source_faces)]
        edges = np.sort(source_faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_vertices = np.unique(unique[counts == 1])
        interior_vertices = int(len(region_points) - len(boundary_vertices))
        if patch_id in failed_reason:
            reason = 'GPU task rejected: ' + failed_reason[patch_id]
        elif interior_vertices == 0:
            reason = 'all source vertices lie on the locked patch contour'
        else:
            reason = 'local candidate retained after quality and geometry guards'
        retained_regions.append(dict(patch_id=patch_id, patch_type=int(patches[patch_id][0][0]),
                                     source_faces=int(len(source_faces)), output_faces=int(output_counts[patch_id]),
                                     source_area=float(areas.sum()), source_interior_vertices=interior_vertices,
                                     centroid=np.average(tri.mean(axis=1), axis=0, weights=areas).tolist() if areas.sum()>0 else region_points.mean(axis=0).tolist(),
                                     bbox_min=region_points.min(axis=0).tolist(), bbox_max=region_points.max(axis=0).tolist(),
                                     reason=reason))
    retained_planar_area = sum(p['source_area'] for p in retained_regions if p['patch_type'] == 1)
    return dict(source_patches=len(patches), source_total_area=source_total_area, output_faces=int(len(output_faces)),
                task_report_retained_patch_count=len(task_retained),
                retained_patch_count=len(retained), retained_source_faces=int(source_counts[list(retained)].sum()) if retained else 0,
                retained_planar_patch_count=len(planar_retained), retained_planar_source_faces=int(source_counts[planar_retained].sum()) if planar_retained else 0,
                retained_planar_area=retained_planar_area,
                retained_planar_area_fraction=retained_planar_area/source_total_area if source_total_area>0 else 0.0,
                retained_regions=retained_regions,
                likely_circular_planar_patch_count=len(circular),
                retained_circular_planar_patches=circular_retained,
                retained_circular_planar_source_faces=sum(p['source_faces'] for p in circular_retained),
                circular_planar_interior_remeshed_count=sum(p['output_interior_vertices']>p['source_interior_vertices'] for p in circular_details),
                circular_planar_without_output_interior_count=sum(p['output_interior_vertices']==0 for p in circular_details),
                circular_planar_mean_output_interior_vertices=float(np.mean([p['output_interior_vertices'] for p in circular_details])) if circular_details else 0.0,
                circular_planar_details=circular_details,
                failed_tasks=failed, output_face_counts_for_retained_patches={str(i): int(output_counts[i]) for i in sorted(retained)},
                retained_patch_ids=sorted(retained),
                circularity_rule='single boundary with at least 10 edges; 4*pi*area/perimeter^2 >= 0.9',
                circular_classification_is_heuristic=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.mesh, args.snapshot)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('failed_tasks', 'output_face_counts_for_retained_patches', 'retained_circular_planar_patches', 'retained_patch_ids', 'circular_planar_details', 'retained_regions')}))


if __name__ == '__main__':
    main()
