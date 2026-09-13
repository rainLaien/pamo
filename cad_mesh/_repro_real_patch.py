"""Time analytic remesh on real partition cylinders near the stall."""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cad_mesh.analytic_remesh import remesh_analytic_patch
from cad_mesh.tests.test_analytic_remesh import (
    boundary_edges, prepare_shared_boundaries,
)


def extract_patch(mesh, patch):
    faces = mesh.faces[np.asarray(patch["triangle_ids"], dtype=np.int64)]
    used, inverse = np.unique(faces, return_inverse=True)
    return mesh.vertices[used], inverse.reshape(-1, 3).astype(np.int64)


def run_one(vertices, faces, patch, target, deviation):
    vertices, faces = prepare_shared_boundaries(vertices, faces, target)
    edges = boundary_edges(faces)
    begin = time.perf_counter()
    _, _, stats = remesh_analytic_patch(
        vertices, faces, patch, edges, target, deviation,
        maximum_normal_deviation_degrees=5.0)
    seconds = time.perf_counter() - begin
    return stats.get("accepted"), stats.get("reason"), seconds, stats.get("timing")


def _worker(payload, conn):
    try:
        vertices, faces, patch, target, deviation = payload
        result = run_one(vertices, faces, patch, target, deviation)
        conn.send(("ok",) + result)
    except Exception as error:
        conn.send(("error", type(error).__name__, str(error), None, None))
    finally:
        conn.close()


def run_isolated(vertices, faces, patch, target, deviation, timeout):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(False)
    process = ctx.Process(
        target=_worker, args=((vertices, faces, patch, target, deviation), child))
    process.start()
    child.close()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return "timeout", None, timeout, None
    if parent.poll():
        return parent.recv()
    return "no-result", process.exitcode, None, None


if __name__ == "__main__":
    root = Path("cad_mesh/debug/112/partition")
    report = json.loads((root / "patch_report.json").read_text())
    patches = {int(item["id"]): item for item in report["patches"]}
    print("loading mesh", flush=True)
    mesh = trimesh.load(root / "patch_result.ply", process=False)
    target = 12.0
    deviation = 0.171984914
    for patch_id in (1663, 1742, 1763, 1410, 362):
        patch = patches[patch_id]
        vertices, faces = extract_patch(mesh, patch)
        print(f"patch {patch_id} type={patch['type']} faces={len(faces)} verts={len(vertices)}",
              flush=True)
        result = run_isolated(vertices, faces, patch, target, deviation, 30)
        print(f"  {result}", flush=True)
