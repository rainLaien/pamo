"""Reproduce analytic chart hang on small-radius cylinders with coarse target."""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _synthetic():
    from cad_mesh.analytic_remesh import remesh_analytic_patch
    from cad_mesh.tests.test_analytic_remesh import (
        boundary_edges, prepare_shared_boundaries, ruled_surface,
    )
    vertices, faces, patch = ruled_surface(count=48, axial_rows=9)
    vertices, faces = prepare_shared_boundaries(vertices, faces, 12.0)
    begin = time.perf_counter()
    _, _, stats = remesh_analytic_patch(
        vertices, faces, patch, boundary_edges(faces), 12.0, 0.172,
        maximum_normal_deviation_degrees=5.0)
    return dict(stats), time.perf_counter() - begin


def _worker(kind, conn):
    try:
        if kind == "synthetic":
            stats, seconds = _synthetic()
            conn.send(("ok", stats.get("accepted"), stats.get("reason"), seconds,
                       stats.get("output_faces"), stats.get("timing")))
        else:
            conn.send(("bad-kind", None, None, None, None, None))
    except Exception as error:
        conn.send(("error", type(error).__name__, str(error), None, None, None))
    finally:
        conn.close()


def run_with_timeout(kind, seconds):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(False)
    process = ctx.Process(target=_worker, args=(kind, child))
    process.start()
    child.close()
    process.join(seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return "timeout", None
    if parent.poll():
        return parent.recv()
    return "no-result", process.exitcode


if __name__ == "__main__":
    print("synthetic start", flush=True)
    result = run_with_timeout("synthetic", 20)
    print("synthetic", result, flush=True)
