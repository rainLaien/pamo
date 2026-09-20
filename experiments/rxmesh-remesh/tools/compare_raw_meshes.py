"""Independent raw remesh audit. Distances are sampled, not Hausdorff proofs."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def samples(mesh, count=3000):
    rng = np.random.default_rng(7401)
    ids = rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    t = mesh.triangles[ids]
    return t[:, 0] + uv[:, :1] * (t[:, 1] - t[:, 0]) + uv[:, 1:] * (t[:, 2] - t[:, 0])


class DistanceQuery:
    def __init__(self, mesh):
        self.triangles = mesh.triangles
        self.centers = self.triangles.mean(axis=1)
        self.radius = np.linalg.norm(self.triangles - self.centers[:, None], axis=2).max()
        self.tree = cKDTree(self.centers)
        self.vertices = cKDTree(mesh.vertices)

    def distances(self, points):
        upper = self.vertices.query(points)[0]
        out = []
        for p, bound in zip(points, upper):
            # Every potentially closer triangle lies inside this sphere.
            ids = self.tree.query_ball_point(p, bound + self.radius + 1e-9)
            tri = self.triangles[ids]
            closest = trimesh.triangles.closest_point(tri, np.broadcast_to(p, (len(tri), 3)))
            out.append(np.linalg.norm(closest - p, axis=1).min())
        return np.asarray(out)


def metrics(source, output, h, source_query):
    triangles = output.triangles
    sides = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    q = 2 * np.sqrt(3) * area2 / (sides * sides).sum(axis=1)
    edges, counts = np.unique(np.sort(output.edges, axis=1), axis=0, return_counts=True)
    lengths = np.linalg.norm(output.vertices[edges[:, 0]] - output.vertices[edges[:, 1]], axis=1)
    forward = source_query.distances(np.concatenate([samples(output), output.vertices]))
    backward = DistanceQuery(output).distances(np.concatenate([samples(source), source.vertices]))
    canonical_faces = np.sort(output.faces, axis=1)
    return {
        "vertices": len(output.vertices), "faces": len(output.faces),
        "quality_mean": float(q.mean()), "quality_p05": float(np.quantile(q, .05)),
        "quality_min": float(q.min()), "faces_quality_below_0.1": int((q < .1).sum()),
        "min_angle_degrees": float(np.degrees(output.face_angles.min())),
        "min_angle_p05_degrees": float(np.quantile(np.degrees(output.face_angles.min(axis=1)), .05)),
        "edge_band_fraction": float(((lengths >= .8 * h) & (lengths <= 4 / 3 * h)).mean()),
        "open_edges": int((counts == 1).sum()), "nonmanifold_edges": int((counts > 2).sum()),
        "degenerate_faces": int((area2 < 1e-12).sum()),
        "duplicate_faces": len(output.faces) - len(np.unique(canonical_faces, axis=0)),
        "winding_consistent": bool(output.is_winding_consistent),
        "euler_number": int(len(output.vertices) - len(edges) + len(output.faces)),
        "volume_relative_error": float(abs(output.volume - source.volume) / max(abs(source.volume), 1e-30)),
        "sampled_output_to_source_max": float(forward.max()),
        "sampled_source_to_output_max": float(backward.max()),
        "sampled_bidirectional_max": float(max(forward.max(), backward.max())),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("outputs", nargs="+", type=Path)
    p.add_argument("--save", required=True, type=Path)
    p.add_argument("--length", type=float)
    args = p.parse_args()
    source = trimesh.load(args.source, force="mesh", process=True)
    h = args.length or np.linalg.norm(source.extents) * .01
    query = DistanceQuery(source)
    report = {"source": str(args.source.resolve()), "target_length": float(h), "results": {}}
    for path in args.outputs:
        mesh = trimesh.load(path, force="mesh", process=False)
        row = metrics(source, mesh, h, query)
        report["results"][path.stem] = row
        print(path.stem, json.dumps(row), flush=True)
    args.save.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
