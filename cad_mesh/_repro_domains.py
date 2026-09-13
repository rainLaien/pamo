"""Inspect prepared surface domains around the stall."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cad_mesh.remesh_io import load_partition
from cad_mesh.boundary_domains import prepare_surface_domains

source = load_partition(Path("cad_mesh/debug/112/partition"))
prepared, policy = prepare_surface_domains(source)
print("merged", policy["merged_label_pairs"], "domains", len(prepared.report["patches"]))
patches = prepared.report["patches"]
print("large analytic domains:")
for patch in patches:
    if patch["triangle_count"] >= 200 and patch["type"] in ("Plane", "Cylinder", "Cone"):
        params = patch.get("parameters") or {}
        print(patch["id"], patch["type"], "tris", patch["triangle_count"],
              "r", params.get("radius"), "members", patch.get("source_patch_ids"))
print("domains 1740-1765:")
for patch in patches[1740:1766]:
    params = patch.get("parameters") or {}
    print(patch["id"], patch["type"], "tris", patch["triangle_count"],
          "r", params.get("radius"), "src", patch.get("source_patch_ids"))
