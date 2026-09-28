"""Render before/after triangulation and retained-region location for review."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from audit_phase1 import read_ply
from repair_narrow_strips import read_snapshot


def render(mesh_path, snapshot_path, coverage_path, output_dir):
    _, source_points, source_records, source_patches, _ = read_snapshot(snapshot_path)
    output_points, output_faces, output_labels = read_ply(mesh_path)
    coverage = json.loads(coverage_path.read_text())
    circles = coverage['circular_planar_details']
    choices = [p for p in circles if p['output_interior_vertices'] >= 8 and p['source_faces'] >= 20]
    chosen = max(choices, key=lambda p: p['output_interior_vertices']) if choices else None
    saved = {}
    if chosen:
        patch = chosen['patch_id']
        sf = source_records[source_records[:, 3] == patch, :3].astype(np.int64)
        of = output_faces[output_labels == patch].astype(np.int64)
        source_local = source_points[np.unique(sf)]
        center = source_local.mean(axis=0)
        normal = np.cross(source_points[sf[0, 1]] - source_points[sf[0, 0]],
                          source_points[sf[0, 2]] - source_points[sf[0, 0]])
        normal /= np.linalg.norm(normal)
        u = source_local[0] - center
        u -= np.dot(u, normal) * normal
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        for ax, points, faces, title in ((axes[0], source_points, sf, f'Before: {len(sf)} faces'),
                                         (axes[1], output_points, of, f'After: {len(of)} faces')):
            ids, local = np.unique(faces, return_inverse=True)
            xy = np.column_stack(((points[ids] - center) @ u, (points[ids] - center) @ v))
            ax.triplot(mtri.Triangulation(xy[:, 0], xy[:, 1], local.reshape(-1, 3)),
                       color='#334155', linewidth=.45)
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.set_xlabel('local x')
            ax.set_ylabel('local y')
        fig.suptitle(f'Circular plane patch {patch}: interior remesh')
        path = output_dir / 'review_circular_plane.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved['circular_plane_image'] = str(path)
        saved['circular_plane_patch'] = patch
    retained = [p for p in coverage['retained_regions'] if p['patch_type'] == 1]
    if retained:
        coords = np.array([p['centroid'] for p in retained])
        areas = np.array([p['source_area'] for p in retained])
        marker = np.clip(np.sqrt(areas) * 3, 3, 36)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, (a, b), labels in zip(axes, ((0, 1), (0, 2), (1, 2)), (('X', 'Y'), ('X', 'Z'), ('Y', 'Z'))):
            ax.scatter(coords[:, a], coords[:, b], s=marker, color='#dc2626', alpha=.55, linewidths=0)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_aspect('equal', adjustable='box')
        fig.suptitle('Planar patches retaining source triangles; marker size tracks area')
        path = output_dir / 'review_retained_planar_regions.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved['retained_planar_image'] = str(path)
    field_path = Path(str(mesh_path) + '.vertices.tsv')
    if field_path.exists():
        targets = np.loadtxt(field_path, skiprows=1, ndmin=2)[:, 4]
        face_targets = targets[output_faces]
        transitioning = face_targets.max(axis=1) > face_targets.min(axis=1) * 1.05
        counts = np.bincount(output_labels.astype(np.int64), minlength=coverage['source_patches'])
        transition_counts = np.bincount(output_labels[transitioning].astype(np.int64), minlength=len(counts))
        planar = np.array([p[0][0] == 1 for p in source_patches])
        eligible = np.flatnonzero(planar & (counts >= 100) & (counts <= 800) & (transition_counts >= 50))
        if len(eligible):
            patch = int(eligible[np.argmax(transition_counts[eligible])])
            sf = source_records[source_records[:, 3] == patch, :3].astype(np.int64)
            of = output_faces[output_labels == patch].astype(np.int64)
            source_local = source_points[np.unique(sf)]
            center = source_local.mean(axis=0)
            _, _, basis = np.linalg.svd(source_local - center, full_matrices=False)
            u, v = basis[0], basis[1]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            for index, (ax, points, faces, title) in enumerate(((axes[0], source_points, sf, 'Before'),
                                                                (axes[1], output_points, of, 'After: local target length'))):
                ids, local = np.unique(faces, return_inverse=True)
                xy = np.column_stack(((points[ids] - center) @ u, (points[ids] - center) @ v))
                tri = mtri.Triangulation(xy[:, 0], xy[:, 1], local.reshape(-1, 3))
                if index == 0:
                    ax.triplot(tri, color='#334155', linewidth=.4)
                else:
                    values = targets[faces].mean(axis=1)
                    image = ax.tripcolor(tri, facecolors=values, vmin=targets.min(), vmax=targets.max(),
                                         cmap='viridis', edgecolors='#334155', linewidth=.13)
                    fig.colorbar(image, ax=ax, label='target edge length', shrink=.8)
                ax.set_aspect('equal')
                ax.set_title(title)
                ax.set_xlabel('local x');ax.set_ylabel('local y')
            fig.suptitle(f'Feature transition patch {patch} (blue = fine, yellow = coarse)')
            path = output_dir / 'review_feature_transition.png'
            fig.savefig(path, dpi=170)
            plt.close(fig)
            saved['feature_transition_image'] = str(path)
            saved['feature_transition_patch'] = patch
    (output_dir / 'review_visuals.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(render(args.mesh, args.snapshot, args.coverage, args.output_dir)))


if __name__ == '__main__':
    main()
