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
    removed_degenerate_faces = 0
    if from_stl:
        import trimesh
        mesh = trimesh.load(source, force="mesh", process=True)
        # STL exporters occasionally emit zero-area triangles (often with two
        # identical vertex indices). They are not surface geometry and can
        # make an otherwise manifold shell look non-manifold after welding.
        valid_area = np.isfinite(mesh.area_faces) & (mesh.area_faces > 0.0)
        removed_degenerate_faces = int(len(mesh.faces) - np.count_nonzero(valid_area))
        if removed_degenerate_faces:
            mesh.update_faces(valid_area)
            mesh.remove_unreferenced_vertices()
        # A few STL exports contain tiny detached triangles that touch the
        # main shell along an edge with four incident faces. Keep the sheets
        # topologically separate (duplicated vertices) and pack those tiny
        # faces into a nearby main region instead of sending them through the
        # expensive CAD recognizer or making a one-face compute region.
        components = sorted(mesh.split(only_watertight=False),
                            key=lambda component: len(component.faces), reverse=True)
        if len(components) > 1:
            main_components = [component for component in components
                               if len(component.faces) >= 1024]
            small_components = [component for component in components
                                if len(component.faces) < 1024]
            if len(main_components) != 1:
                raise ValueError("STL has multiple large disconnected components; use CAD partitioning")
            main_faces = len(main_components[0].faces)
            mesh = trimesh.util.concatenate(main_components + small_components)
        else:
            main_faces = len(mesh.faces)
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

    partition_faces = main_faces if from_stl else nf
    if not 1 <= regions <= partition_faces // 1024:
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
    mid = (centers[:partition_faces].min(axis=0) + centers[:partition_faces].max(axis=0)) * 0.5
    seeds = [int(np.argmax(np.sum((centers[:partition_faces] - mid) ** 2, axis=1)))]
    nearest = np.full(partition_faces, np.inf)
    for _ in range(regions - 1):
        delta = centers[:partition_faces] - centers[seeds[-1]]
        nearest = np.minimum(nearest, np.einsum("ij,ij->i", delta, delta))
        seeds.append(int(np.argmax(nearest)))
    del nearest

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
    while assigned < partition_faces:
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
    # A seed enclosed by other growing regions can lose its frontier early.
    # Merge such undersized connected cells across their shared boundary
    # instead of emitting a tiny compute region.
    counts = np.bincount(labels[:partition_faces], minlength=regions).tolist()
    while len(counts) > 1 and min(counts) < 1024:
        small_region = int(np.argmin(counts))
        boundary_labels = []
        for face in np.flatnonzero(labels[:partition_faces] == small_region):
            for slot in range(3 * int(face), 3 * int(face) + 3):
                other = int(adjacency[slot])
                if other >= 0 and labels[other] != small_region:
                    boundary_labels.append(int(labels[other]))
        if not boundary_labels:
            raise ValueError("undersized region has no adjacent region to merge into")
        target = min(set(boundary_labels), key=lambda label: (-boundary_labels.count(label), label))
        labels[labels == small_region] = target
        active_labels = np.unique(labels[:partition_faces])
        remap = np.full(regions, -1, dtype=np.int16)
        remap[active_labels] = np.arange(len(active_labels), dtype=np.int16)
        labels[:partition_faces] = remap[labels[:partition_faces]]
        regions = len(active_labels)
        counts = np.bincount(labels[:partition_faces], minlength=regions).tolist()

    if partition_faces < nf:
        region_centers = np.stack([centers[:partition_faces][labels[:partition_faces] == r].mean(axis=0)
                                   for r in range(regions)])
        extra_delta = centers[partition_faces:, None, :] - region_centers[None, :, :]
        labels[partition_faces:] = np.argmin(np.einsum("ijk,ijk->ij", extra_delta, extra_delta), axis=1)
        counts = np.bincount(labels, minlength=regions).tolist()
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
        "removed_degenerate_faces": removed_degenerate_faces,
        "attached_small_components": int(nf - partition_faces) if from_stl else 0,
        "compute_regions": regions,
        "region_faces": counts,
        "minimum_region_faces": min(counts),
        "maximum_region_faces": max(counts),
        "hard_feature_edges": len(hard_edges),
        "discarded_soft_label_edges": 0 if from_stl else ne - len(hard_edges),
        "connected_regions": nf == partition_faces,
        "main_regions_connected": True,
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
