# Independent constrained refinement

STATUS: Experimental files retained for reference, but disconnected from the
active build and remesh path after the user's rollback request. The runtime
candidate kernel integration was removed. CADMESH_CDT_CUDA has no effect on
the active remesher. The implementation notes below are historical.

This code is independently authored. No Triangle source was copied, translated,
vendored or linked. The algorithmic reference is the published Delaunay
refinement description (circumcenter insertion, constrained-segment encroachment
and local legalization):
https://www.cs.cmu.edu/~quake/tripaper/triangle3.html
This is not a Triangle implementation or a claim of equivalent output.

## Implemented stage

`ConstrainedRefinement.h` exposes a batch geometric candidate interface, with
plain fixed-size arrays and no mesh-library dependency. A single original
function body computes host and CUDA device circumcenter/offcenter candidates,
shortest-edge lengths and priorities. PlanarDomain now snapshots triangle
connectivity, evaluates a batch, processes worst candidates first and rejects
stale connectivity before insertion. Boundary encroachment, domain membership,
local insertion and edge legalization remain in the CPU topology owner. Patch
workers keep their existing independent execution and immutable shared boundary.

CPU is the default. Set `CADMESH_CDT_CUDA=1` to opt in to CUDA candidate batches
of at least 2048 triangles when chart CUDA is permitted. CPU-only remesh overrides
this switch. Device evaluation uses 128 threads per block, one triangle per
thread. Driver/runtime failures fall back to CPU evaluation. Invalid/nonfinite
candidate records are ignored on either backend. Source changes participate in
the existing NVRTC/PTX cache key. This path does not claim a measured speedup:
it has uploads, allocations and readbacks, while topology and boundary queries
still cost CPU time. The existing per-thread CUDA runtime is reused.

## Explicit limits

Interior site relaxation now runs before and after optional candidate insertion.
Up to four sweeps move ordinary interior points toward the neighbor centroid,
with decreasing step lengths. Positive signed area, non-worsening minimum local
shape quality, lower reciprocal-quality energy, edge-length and cylinder angular
span bounds gate each individual move. Boundary and periodic aliases stay fixed.
Edge legalization follows each sweep. Cylinder collapse is followed by a fresh
adjacency map and another bounded relaxation. These are local optimization
decisions, not a whole-patch quality acceptance threshold. Regular structured
rings still bypass optional refinement and retain their row construction.

This is the first candidate/refinement stage, not a standalone PSLG mesher.
Existing chart seed triangulation, periodic unwrapping and constraint extraction
are still used. Existing fixed-boundary encroachment currently rejects a
candidate; it does not yet send a shared-boundary split request to neighbors.
Robust exact geometric predicates, coordinated boundary split transactions and
GPU conflict-free cavity updates remain to be implemented. Regular ring charts
continue to use their structured row path and do not invoke this optional
candidate refinement. Twenty-eight degrees remains a request, not a guarantee.

No compilation, tests, remesh runs or visual verification were performed, per
the user's instruction. In particular no CPU/GPU equivalence or speedup has
been established.
