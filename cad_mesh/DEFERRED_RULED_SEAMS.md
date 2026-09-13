# Ruled surfaces and deferred shared-boundary sampling

The native path now builds cylinders and cones before the adjacent analytic
patches. The existing worker pool still handles independent patches in each
phase. Freeform jobs follow the analytic phases.

* Pre-sampling skips every interface incident to a cylinder or cone.
* A selected cone is one complete trimmed domain; short original triangles
  are not removed as preserved islands before construction.
* A source-topology parameter chart handles holes and periodic cuts. Physical
  boundary points inserted by the chart are proposals on original edges,
  identified by endpoint IDs and an edge parameter. They are positioned on
  the original edge, rather than independently projected onto each surface.
* Proposals from both incident patches are combined into one ordered sequence.
  Duplicate vertices are unified and incident triangles are subdivided on both
  the candidate mesh and the immutable reference mesh before adjacent charts
  are built. The reference subdivision preserves the original geometry.
* The original reference sidewall is transferred through parameter-space
  barycentric coordinates. Source internal edges with normal changes above
  five degrees are constrained and sampled; they are not used as hole loops.
  This prevents flips across geometric turns near narrow trims.
* Final seam sampling updates both incident sides without moving endpoints.
  Per-face result status and source ownership survive this subdivision.
* Distance and normal acceptance still apply. Each patch uses its own source
  triangle index for these queries. A failed patch falls back to its conforming
  reference triangles, including the shared boundary subdivisions.

## Fast verification

From the repository root:

```powershell
.\examples\remesh_10_cones.ps1 -TargetEdgeLength 6
```

The script fully partitions the input and selects the ten largest cone patches
by source face count (source ID breaks ties). It exports the whole contextual
mesh. Other patches are not remeshed, but their shared boundary triangles can
be subdivided for conformity. The classification distance defaults to 0.02;
the remesh distance at target length 6 defaults to 0.03 and normal tolerance
is 10 degrees.

```powershell
python .\examples\check_preview_mesh.py <output-directory>\remesh_result.ply
```

The PLY check reports selected source IDs, result states, open/nonmanifold edge
counts, zero-area rebuilt faces, and the minimum rebuilt angle. Passing shape
and connectivity checks does not imply a 28-degree minimum triangle angle:
fixed original boundary details can still produce very thin triangles.
