"""Merge only non-hard internal interfaces with an exact flat-plane witness.

The first implementation certifies axis-aligned planes without a geometric
tolerance. It does not demote arbitrary hard=0 records or modify coordinates,
triangles, hard records, open boundaries or unproven interfaces.
"""
import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
import numpy as np
from repair_narrow_strips import read_snapshot, write_snapshot, require


def merge(points, records, patches, features):
    nr = len(patches)
    labels = records[:, 3].astype(int)
    tri = records[:, :3]
    owner = np.repeat(labels, 3)
    minimum = np.full((nr, 3), np.inf)
    maximum = np.full((nr, 3), -np.inf)
    np.minimum.at(minimum, owner, points[tri.reshape(-1)])
    np.maximum.at(maximum, owner, points[tri.reshape(-1)])
    flat = minimum == maximum
    axes = np.where(flat.any(axis=1), flat.argmax(axis=1), -1)
    normals = np.cross(points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]])
    signs = np.sign(normals[np.arange(len(tri)), np.maximum(axes[labels], 0)])
    lower = np.full(nr, np.inf)
    upper = np.full(nr, -np.inf)
    np.minimum.at(lower, labels, signs)
    np.maximum.at(upper, labels, signs)
    # Special support/fillet roles are retained. Eligibility is a witness on
    # every source triangle, not agreement between fitted plane parameters.
    eligible = np.array([ph[0] == 1 and ph[1] == 1 and ph[2] == 0 and ph[4] == 0
                         for ph, _ in patches])
    eligible &= (axes >= 0) & (lower == upper) & (lower != 0)
    values = minimum[np.arange(nr), np.maximum(axes, 0)]

    ends = tri[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    keys = (ends.min(axis=1).astype(np.uint64) << 32) | ends.max(axis=1).astype(np.uint64)
    order = np.argsort(keys, kind='stable')
    unique, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    feature_keys = (features[:, 1:3].min(axis=1).astype(np.uint64) << 32) | features[:, 1:3].max(axis=1).astype(np.uint64)
    locations = np.searchsorted(unique, feature_keys)
    require((locations < len(unique)).all() and np.array_equal(unique[locations], feature_keys),
            'constraint record is not a source mesh edge')
    first = labels[order[starts[locations]] // 3]
    second = labels[order[np.minimum(starts[locations] + 1, len(order) - 1)] // 3]
    internal = counts[locations] == 2
    witness = internal & (features[:, 3] == 0) & (first != second)
    witness &= eligible[first] & eligible[second] & (axes[first] == axes[second])
    witness &= (values[first] == values[second]) & (lower[first] == lower[second])
    parent = np.arange(nr)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    for a, b in zip(first[witness], second[witness]):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(nr)])
    distinct, mapping = np.unique(roots, return_inverse=True)
    output_records = records.copy()
    output_records[:, 3] = mapping[labels]
    output_patches = []
    group_sizes = np.bincount(mapping)
    faces_per_group = np.bincount(output_records[:, 3], minlength=len(distinct))
    for new_id, root in enumerate(distinct):
        ph, tail = patches[root]
        if group_sizes[new_id] > 1:
            axis = int(axes[root])
            parameters = [0.] * 9
            parameters[axis] = float(values[root])
            parameters[axis + 3] = float(lower[root])
            output_patches.append(([1, 1, 0, int(faces_per_group[new_id]), 0], struct.pack('<9d', *parameters)))
        else:
            supports = np.frombuffer(tail[:ph[4] * 4], '<u4')
            rewritten = list(dict.fromkeys(int(mapping[p]) for p in supports))
            updated = ph.copy()
            updated[4] = len(rewritten)
            output_patches.append((updated, np.asarray(rewritten, '<u4').tobytes() + tail[ph[4] * 4:]))
    removed = witness & (mapping[first] == mapping[second])
    output_features = features[~removed].copy()
    require(np.array_equal(output_records[:, :3], records[:, :3]), 'triangle/winding mutation')
    require(np.array_equal(features[features[:, 3] == 1], output_features[output_features[:, 3] == 1]),
            'declared hard record mutation')
    require(not (removed & (counts[locations] != 2)).any(), 'open/nonmanifold constraint removal')
    report = dict(source_patches=nr, merged_patches=len(distinct), certified_flat_patches=int(eligible.sum()),
                  source_constraint_records=len(features), retained_constraint_records=len(output_features),
                  removed_internal_nonhard_records=int(removed.sum()),
                  declared_hard_records_retained=int((features[:, 3] == 1).sum()),
                  intrinsic_open_records_retained=int((counts[locations] == 1).sum()),
                  original_to_merged_patch=mapping.tolist(),
                  removed_record_ids=features[removed, 0].tolist(),
                  coordinates_unchanged=True, triangles_and_winding_unchanged=True,
                  geometric_tolerance_used=0, proof='all member source vertices share exact axis coordinate and face orientation; only two-face nonhard interfaces removed',
                  classification_limit='hard declaration is inherited, not full CAD certification; arbitrary nonhard interfaces remain constrained')
    return output_records, output_patches, output_features, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    require(args.source.resolve() != args.output.resolve(), 'source snapshot must remain immutable')
    require(args.report.resolve() not in (args.source.resolve(),args.output.resolve()), 'report/source collision')
    start = time.perf_counter()
    header, points, records, patches, features = read_snapshot(args.source)
    out_records, out_patches, out_features, report = merge(points, records, patches, features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_snapshot(args.output, header, points, out_records, out_patches, out_features)
    check_header, check_points, check_records, _, check_features = read_snapshot(args.output)
    require(np.array_equal(check_points, points) and np.array_equal(check_records[:, :3], records[:, :3]), 'saved geometry changed')
    require(np.array_equal(check_features, out_features), 'saved feature declarations changed')
    report.update(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), seconds=time.perf_counter() - start)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['source_patches', 'merged_patches', 'removed_internal_nonhard_records', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
