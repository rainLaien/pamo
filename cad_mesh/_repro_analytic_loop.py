"""Run analytic assessment on prepared domains 1700-1850 with per-patch timing."""
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cad_mesh.remesh_io import load_partition
from cad_mesh.boundary_domains import prepare_surface_domains
from cad_mesh.remesh_pipeline import _pamo_module
from cad_mesh.analytic_remesh import remesh_analytic_patch

print("load", flush=True)
source = load_partition(Path("cad_mesh/debug/112/partition"))
prepared, _ = prepare_surface_domains(source)
original = _pamo_module("original_constrained")
surface = _pamo_module("surface_sample")
target, deviation = 12.0, 0.171984914
print("split", flush=True)
vertices, faces, labels, splits, lineage = original._subdivide_labeled_patch_boundaries(
    prepared.vertices, prepared.faces, prepared.face_patch_ids, target * .75,
    explicit_constraint_edges=prepared.constraint_edges, return_lineage=True,
    maximum_splits=1000000)
print("splits", splits, "faces", len(faces), flush=True)
constraints = lineage["constraint_edges"]
patches = prepared.report["patches"]
order = np.argsort(labels, kind="stable")
offsets = np.r_[0, np.flatnonzero(np.diff(labels[order])) + 1, len(faces)]
constraint_keys = np.sort(constraints[:, 0] * np.int64(len(vertices)) + constraints[:, 1])
print("groups", len(offsets) - 1, flush=True)
for first, last in zip(offsets[:-1], offsets[1:]):
    group = order[first:last]
    patch_id = int(labels[group[0]])
    if patch_id < 1700 or patch_id > 1850:
        continue
    patch = patches[patch_id]
    patch_type = patch["type"]
    begin = time.perf_counter()
    reason = "skip"
    if patch_type in ("Plane", "Cylinder", "Cone"):
        used, local_f = np.unique(faces[group], return_inverse=True)
        local_f = local_f.reshape(-1, 3)
        actual = np.sort(faces[group][:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
        keys = np.unique(actual[:, 0] * np.int64(len(vertices)) + actual[:, 1])
        positions = np.searchsorted(constraint_keys, keys)
        present = positions < len(constraint_keys)
        present[present] &= constraint_keys[positions[present]] == keys[present]
        keys = keys[present]
        local_edges = np.searchsorted(used, np.column_stack((keys // len(vertices), keys % len(vertices))))
        _, _, diagnostic = remesh_analytic_patch(
            vertices[used], local_f, patch, local_edges, target, deviation,
            maximum_normal_deviation_degrees=5.0)
        reason = diagnostic.get("reason")
    print(f"domain {patch_id} {patch_type} faces={len(group)} {time.perf_counter()-begin:.3f}s {reason}",
          flush=True)
print("done", flush=True)
