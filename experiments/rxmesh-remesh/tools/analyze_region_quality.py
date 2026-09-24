"""Locate poor source triangles relative to compute seams and hard features."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def near_mask(neighbors: np.ndarray, seeds: np.ndarray, rings: int) -> np.ndarray:
    near = seeds.copy()
    frontier = np.flatnonzero(seeds)
    for _ in range(rings):
        frontier = neighbors[frontier].ravel()
        frontier = np.unique(frontier[frontier >= 0])
        frontier = frontier[~near[frontier]]
        if not len(frontier):
            break
        near[frontier] = True
    return near


def analyze(snapshot: Path, report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with snapshot.open("rb") as stream:
        header = stream.read(72)
        if len(header) != 72 or header[:8] != b"CADPART1":
            raise ValueError("expected CADPART1 snapshot")
        nv, nf, nr, ne = struct.unpack_from("<4I", header, 8)
        points = np.frombuffer(stream.read(nv * 24), dtype="<f8").reshape(nv, 3)
        records = np.frombuffer(stream.read(nf * 16), dtype="<u4").reshape(nf, 4)
        for _ in range(nr):
            patch_header = stream.read(20)
            supports = struct.unpack_from("<I", patch_header, 16)[0]
            stream.seek(supports * 4 + 72, 1)
        features = np.frombuffer(stream.read(ne * 16), dtype="<u4").reshape(ne, 4)

    faces, labels = records[:, :3], records[:, 3]
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    ab, ac, bc = b - a, c - a, c - b
    denominator = np.einsum("ij,ij->i", ab, ab) + np.einsum("ij,ij->i", ac, ac) + np.einsum("ij,ij->i", bc, bc)
    quality = np.divide(2 * np.sqrt(3) * np.linalg.norm(np.cross(ab, ac), axis=1),
                        denominator, out=np.zeros(nf), where=denominator > 0)

    keys = np.empty((nf, 3), dtype=np.uint64)
    for side in range(3):
        u, v = faces[:, side].astype(np.uint64), faces[:, (side + 1) % 3].astype(np.uint64)
        keys[:, side] = (np.minimum(u, v) << 32) | np.maximum(u, v)
    flat = keys.ravel()
    order = np.argsort(flat)
    paired = np.flatnonzero(flat[order[:-1]] == flat[order[1:]])
    left, right = order[paired], order[paired + 1]
    left_face, right_face = left // 3, right // 3
    same_region = labels[left_face] == labels[right_face]
    neighbors = np.full((nf, 3), -1, dtype=np.int32)
    neighbors.ravel()[left[same_region]] = right_face[same_region]
    neighbors.ravel()[right[same_region]] = left_face[same_region]

    seam = np.zeros(nf, dtype=bool)
    seam[left_face[~same_region]] = True
    seam[right_face[~same_region]] = True
    unpaired = np.ones(nf * 3, dtype=bool)
    unpaired[left] = False
    unpaired[right] = False
    seam[np.flatnonzero(unpaired) // 3] = True

    feature_keys = (features[:, 1].astype(np.uint64) << 32) | features[:, 2]
    feature = np.isin(keys, feature_keys).any(axis=1)
    degree = np.bincount(np.r_[features[:, 1], features[:, 2]], minlength=nv)
    corners = (degree > 0) & (degree != 2)
    corner = corners[faces].any(axis=1)
    near_seam = near_mask(neighbors, seam, 2)
    near_feature = near_mask(neighbors, feature, 2)
    near_corner = near_mask(neighbors, corner, 2)

    regions = []
    for patch in report["patches"]:
        rid = patch["id"]
        indices = np.flatnonzero(labels == rid)
        low = indices[np.argsort(quality[indices])[:max(1, len(indices) // 20)]]
        entry = {"id": rid, "unchanged": bool(patch["unchanged"]),
                 "faces": len(indices), "source_quality_p05": float(quality[low[-1]]),
                 "feature_face_fraction": float(np.mean(feature[indices])),
                 "seam_face_fraction": float(np.mean(seam[indices])),
                 "low_near_seam": float(np.mean(near_seam[low])),
                 "low_near_feature": float(np.mean(near_feature[low])),
                 "low_near_corner": float(np.mean(near_corner[low])),
                 "low_interior": float(np.mean(~near_seam[low] & ~near_feature[low]))}
        regions.append(entry)
    summary = {}
    for state in (False, True):
        group = [r for r in regions if r["unchanged"] == state]
        summary["unchanged" if state else "improved"] = {
            "regions": len(group),
            **{key: float(np.median([r[key] for r in group])) for key in
               ("feature_face_fraction", "seam_face_fraction", "low_near_seam",
                "low_near_feature", "low_near_corner", "low_interior")},
        }
    return {"summary": summary, "regions": regions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.snapshot, args.report)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("unchanged regions:")
    for region in result["regions"]:
        if region["unchanged"]:
            print(region)
