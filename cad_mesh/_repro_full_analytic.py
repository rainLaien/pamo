"""Full analytic stage with faulthandler to catch native crashes."""
from pathlib import Path
import faulthandler
import sys
import time

faulthandler.enable(file=sys.stderr, all_threads=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_mesh.remesh_io import load_partition
from cad_mesh.boundary_domains import prepare_surface_domains
from cad_mesh.remesh_pipeline import _pamo_module
from cad_mesh.surface_rebuild import _analytic_stage

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
print("analytic", splits, "faces", len(faces), flush=True)
begin = time.perf_counter()
vertices, faces, labels, analytic_ids, charts, summary = _analytic_stage(
    vertices, faces, labels, lineage["constraint_edges"], prepared.report["patches"],
    target, deviation, surface, maximum_normal_deviation_degrees=5.0)
print("done", len(analytic_ids), "accepted in", time.perf_counter() - begin,
      "summary", summary, flush=True)
