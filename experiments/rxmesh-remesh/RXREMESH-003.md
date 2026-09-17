# RXREMESH-003 — PAMO partition handoff and shape preservation

The user requested preservation of the original shape, holes and sharp edges,
with geometric partitioning connected to RXMesh. This extends the scope beyond
002's dirty-ring optimization. Validation uses `examples/Unnamed-Body.stl`;
stress testing is excluded at the user's request.

## Changes

- Reuse PAMO's model-first recognition and patch graph through its existing
  `--remesh-handoff --stop-after-partition` export and `CADPART1` packager.
- Add a transactional snapshot reader. Preserve indexed face ownership, plane
  and cylinder parameters, and fixed source feature vertices. Reject unsupported
  projection targets and unsupported within-patch feature curves explicitly.
- Permit GPU splits of interior diagonals with fixed endpoints. The new interior
  vertex is a surface vertex, while the endpoints and feature edges stay fixed.
- Subdivide long feature segments conformingly on the CPU before the GPU pass,
  retaining the exact source polyline and all original feature vertices.
- Size RXMesh dynamic LP mapping tables for potential post-slicing ribbon
  elements, including meshes with zero initial ribbon. The old initial-ribbon
  capacity could overflow when a growing patch was sliced, leaving invalid
  owner lookups during cleanup.
- Check triangle direction for flips and cavity fans during collapse, and
  backtrack smoothing moves that would reverse an incident triangle. Interior
  flips may use fixed boundary vertices without moving them; new face labels
  come from the original edge's geometric patch rather than a seam vertex.
- Add `--partition` and `--max-error` CLI options. CAD mode validates original
  feature vertices/edges, patch presence, triangle orientation and sampled
  analytic deviation before saving an OBJ. The default deviation limit is
  0.001 times the bounding-box diagonal.
- Preserve geometric patch groups and float precision in OBJ export.
- Add `run_cad_remesh.ps1` to run the complete file-to-file pipeline and save an
  independent shape/topology report. Add a comparison rendering tool.

## Verification

The supplied input partitions into six planes and two cylinders, with 142
constraint edges. The snapshot reader/validator unit test passes valid import,
fixed-point checks, unsupported surface rejection, truncation rejection,
transactional failure and inverted-face rejection.

The previous raw STL output was topologically closed but lost the source shape:
the input-to-output sampled distance reached 10.2818 and volume decreased 4.98%.
The GPU CAD pipeline result is pending compilation and verification.

## Current scope

Feature polylines may gain vertices on their original segments; this does not
refit or coarsen analytic boundary curves. Interior sizing remains constant. Surface-distance
checks sample vertices, centroids and edge midpoints and are not certified
Hausdorff bounds or general self-intersection proofs. Cone, sphere, torus and
reference-mesh projection are not supported by this adapter. Multi-patch GPU
migration robustness is not established by a successful run of this input.
