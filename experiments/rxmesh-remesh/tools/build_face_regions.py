"""Create connected, balanced remesh regions while preserving hard CAD edges."""

import argparse
from collections import deque
import heapq
import json
import struct
from pathlib import Path

import numpy as np


def build(source: Path, destination: Path, regions: int) -> dict:
    from_stl = source.suffix.lower() == ".stl"
    if from_stl:
        import trimesh
        mesh = trimesh.load(source, force="mesh", process=True)
        nv, nf = len(mesh.vertices), len(mesh.faces)
        source_patches, ne = 1, 0
        vertex_bytes = np.asarray(mesh.vertices, dtype="<f8").tobytes()
        initial_faces = np.zeros((nf, 4), dtype="<u4")
        initial_faces[:, :3] = mesh.faces
        face_bytes = bytearray(initial_faces.tobytes())
        header = b"CADPART1" + struct.pack("<4I6d", nv, nf, 1, 0, *([0.0] * 6))
        hard_edges = []
    else:
      with source.open("rb") as stream:
        header = stream.read(72)
        if len(header) != 72 or header[:8] != b"CADPART1":
            raise ValueError("expected CADPART1 input")
        nv, nf, source_patches, ne = struct.unpack_from("<4I", header, 8)
        if not 1 <= regions <= nf // 1024:
            raise ValueError("each requested region must have at least 1024 faces")
        vertex_bytes = stream.read(nv * 24)
        face_bytes = bytearray(stream.read(nf * 16))
        if len(vertex_bytes) != nv * 24 or len(face_bytes) != nf * 16:
            raise ValueError("truncated vertex or face data")
        for _ in range(source_patches):
            record = stream.read(20)
            if len(record) != 20:
                raise ValueError("truncated patch record")
            support_count = struct.unpack_from("<I", record, 16)[0]
            stream.seek(support_count * 4 + 72, 1)
        hard_edges = []
        for _ in range(ne):
            record = stream.read(16)
            if len(record) != 16:
                raise ValueError("truncated feature edge")
            if struct.unpack_from("<I", record, 12)[0]:
                hard_edges.append(record)
        if stream.read(1):
            raise ValueError("trailing snapshot data")

    if not 1 <= regions <= nf // 1024:
        raise ValueError("each requested region must have at least 1024 faces")

    vertices = np.frombuffer(vertex_bytes, dtype="<f8").reshape(nv, 3)
    faces = np.frombuffer(face_bytes, dtype="<u4").reshape(nf, 4)
    triangle = faces[:, :3]
    edge_keys = np.empty((nf, 3), dtype=np.uint64)
    for k in range(3):
        a = triangle[:, k].astype(np.uint64)
        b = triangle[:, (k + 1) % 3].astype(np.uint64)
        edge_keys[:, k] = (np.minimum(a, b) << 32) | np.maximum(a, b)
    keys = edge_keys.ravel()
    order = np.argsort(keys, kind="stable")
    same = keys[order[1:]] == keys[order[:-1]]
    if np.any(same[1:] & same[:-1]):
        raise ValueError("input has a non-manifold edge")
    paired = np.flatnonzero(same)
    left, right = order[paired], order[paired + 1]
    if from_stl:
        normals = np.asarray(mesh.face_normals)
        dot = np.einsum("ij,ij->i", normals[left // 3], normals[right // 3])
        crease = dot <= np.cos(np.deg2rad(30.0))
        crease_slots = left[crease]
        for feature_id, slot in enumerate(crease_slots):
            face, side = divmod(int(slot), 3)
            a = int(triangle[face, side])
            b = int(triangle[face, (side + 1) % 3])
            hard_edges.append(struct.pack("<4I", feature_id, min(a, b), max(a, b), 1))
    adjacency = np.full(nf * 3, -1, dtype=np.int32)
    adjacency[left] = right // 3
    adjacency[right] = left // 3
    del edge_keys, keys, order, same, paired, left, right

    centers = (vertices[triangle[:, 0]] + vertices[triangle[:, 1]] +
               vertices[triangle[:, 2]]) / 3.0
    mid = (centers.min(axis=0) + centers.max(axis=0)) * 0.5
    seeds = [int(np.argmax(np.sum((centers - mid) ** 2, axis=1)))]
    nearest = np.full(nf, np.inf)
    for _ in range(regions - 1):
        delta = centers - centers[seeds[-1]]
        nearest = np.minimum(nearest, np.einsum("ij,ij->i", delta, delta))
        seeds.append(int(np.argmax(nearest)))
    del centers, nearest

    labels = np.full(nf, -1, dtype=np.int16)
    counts = [0] * regions
    frontiers = [deque() for _ in range(regions)]
    heap = []
    neighbor = memoryview(adjacency)
    for region, seed in enumerate(seeds):
        labels[seed] = region
        counts[region] = 1
        for slot in range(3 * seed, 3 * seed + 3):
            if neighbor[slot] >= 0:
                frontiers[region].append(neighbor[slot])
        heapq.heappush(heap, (1, region))
    assigned = regions
    while assigned < nf:
        if not heap:
            raise ValueError("input has disconnected face components")
        count, region = heapq.heappop(heap)
        if count != counts[region]:
            continue
        frontier = frontiers[region]
        while frontier and labels[frontier[0]] >= 0:
            frontier.popleft()
        if not frontier:
            continue
        face = frontier.popleft()
        labels[face] = region
        counts[region] += 1
        assigned += 1
        for slot in range(3 * face, 3 * face + 3):
            next_face = neighbor[slot]
            if next_face >= 0 and labels[next_face] < 0:
                frontier.append(next_face)
        heapq.heappush(heap, (counts[region], region))
    if min(counts) < 1024:
        raise ValueError(f"a connected region is too small: {min(counts)} faces")
    faces[:, 3] = labels.astype(np.uint32)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        stream.write(header[:8])
        stream.write(struct.pack("<4I", nv, nf, regions, len(hard_edges)))
        stream.write(header[24:72])
        stream.write(vertex_bytes)
        stream.write(face_bytes)
        for count in counts:
            stream.write(struct.pack("<5I9d", 0, 0, 0, count, 0,
                                     *([0.0] * 8 + [-1.0])))
        for edge in hard_edges:
            stream.write(edge)
    result = {
        "input_faces": nf,
        "source_patches": source_patches,
        "source_kind": "stl" if from_stl else "cadpart",
        "compute_regions": regions,
        "region_faces": counts,
        "minimum_region_faces": min(counts),
        "maximum_region_faces": max(counts),
        "hard_feature_edges": len(hard_edges),
        "discarded_soft_label_edges": 0 if from_stl else ne - len(hard_edges),
        "connected_regions": True,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--regions", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination, args.regions)))
