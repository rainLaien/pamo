# Saved-partition analytic reconstruction

## Boundary compatibility and refinement progress

Global boundary sampling now considers the cylinder metric used by the local
constructor: circumferential arc travel is limited by curvature, while axial
travel is limited by TargetEdgeLength. Both incident patches consume the same
inserted vertex. Existing segments that meet both requirements are retained.
Only actual patch interfaces/open boundaries receive this additional cylinder
metric check; ordinary internal edges do not become shared boundaries. Sharp
feature constraints already supplied by the partition keep their prior role.

Cylinder construction checks all fixed boundary segments before triangulation.
Residual incompatibility (for example, an exhausted global split budget)
reports `fixed_boundary_requires_sampling` with the endpoint IDs and metric
length/target instead of trying to repair an immutable boundary from inside.
Virtual seams receive the same check when the local domain is initialized.

Refinement detects a midpoint coinciding with either endpoint and reports
`midpoint_precision_stall`. For cylinders, every 256 internal splits it checks
the remaining overlong-edge count and maximum length. Four consecutive windows
with no measurable improvement stop as `edge_repair_stalled`. This is a bounded
heuristic that retains the source patch, not a mathematical impossibility test.
The planar path keeps its previous refinement policy. Diagnostic local IDs
refer to chart vertices; nonnegative global IDs refer to the boundary-sampled
input and negative aliases denote generated points/seam copies. No build or
runtime validation was performed.

## Directional cylinder spacing and failure diagnostics

Cylinder reconstruction now applies the curvature-derived spacing only to
the circumferential direction. Axial spacing uses the requested target edge
length. Internally the axial UV coordinate is scaled by circumferential /
axial spacing for triangulation, then restored before mapping to 3D. This
avoids subdividing a long cylinder's straight axial direction at the small
radius's angular spacing. Both single-loop and full-ring paths use this
metric. Cylinder edge flips cannot replace a diagonal with one longer than
both the target and the old diagonal. The established planar path is unchanged.

Per-cylinder logs report both spacings, the last construction stage, generated
mesh size, work count, initial-legalization time, lattice time and edge-repair
time. A failure summary groups counts and accumulated worker time by reason.
Single-loop cylinders still use the constrained triangulation path; this
change does not claim to implement general clipped structured grids or support
all cylindrical trims. Default maximum deviation remains 0.1. No compilation
or runtime validation was performed.

## Irregular cylinder end trims and deviation tolerance

The saved-remesh script now defaults `-MaxDeviation` to `0.1` and always
passes it to the native executable. Explicit values still override the
default. This is the sampled input-to-analytic-surface deviation gate, in
model units; it does not disable boundary/topology or collision checks.

Two-loop cylinder rings now use `direct_periodic_strip`. Each trim may vary
in axial height as it travels around the circumference. Angularly monotone
trim curves are interpolated in the cylinder chart; interior rows are
generated directly between them, with spacing based on target length and
the existing normal-deviation bound. Original trim vertices stay shared;
periodic seam copies have identical aliases. Transition strips connect the
original boundary samples to the interior grid. Only remaining long edges
need local insertion; the full ring no longer runs lattice point location
or initial global Delaunay legalization.

Trim loops with angular backtracking/vertical segments, extra holes, or a
folding ruled parameterization remain unsupported and retain their source
mesh. The algorithm does not assume planar or equal-height end loops. Size
and work budgets remain in effect. Compilation and execution are left to
the user as requested.

## Cylinder-only experiment

```powershell
.\cad_mesh\remesh_saved.ps1 -CylindersOnly -AnalyticWorkers 4
```

Only cylinder patches are reconstructed in this opt-in mode. Other interiors
remain as supplied by the partition snapshot; shared boundary sampling still
updates both incident patches. Output stops before the full local split,
collapse and relaxation stages and does not compute triangle quality scores.
The collision guard and boundary/topology checks remain enabled. Use the PLY
`remeshed` face attribute to inspect accepted cylinder reconstructions.

Cylinders use u=radius*angle and v=axial height, then the same lattice and
local constrained Delaunay path as planes. Supported charts are contractible
single-loop trims spanning less than one period, and two-loop full rings
whose end loops are approximately constant in height. Full rings use paired
seam aliases with identical sampling; the seam cannot be independently split
by the two sides. Single winding loops, additional holes, and unsupported
ring boundaries retain the input patch instead of retrying the legacy slow
refiner. The existing normal-deviation setting may reduce interior spacing
below TargetEdgeLength for small radii. No absolute triangle-quality target
is imposed. Construction budgets and geometry checks can still reject jobs.

Logs include `cylinder constrained mesh` and `cylinders-only complete`.
`-PlanesOnly` and `-CylindersOnly` are mutually exclusive. This implementation
has not been compiled or run as part of the change.

## Planar patches with holes

`remesh_saved.ps1 -PlanesOnly` now includes every planar patch, including
multiple boundary loops. `-SimplePlanesOnly` remains a compatibility alias
and now also includes holes. Curved surfaces stay excluded in this mode.

For multiply connected planes, the source patch triangles provide the initial
domain topology. Existing interior sites are retained as seed points; interior
diagonals are legalized and lattice sites added. This avoids artificial bridges
between holes. All actual boundary segments are immutable constraints. An
even-odd scanline test excludes holes from lattice sampling and handles nested
loops/islands. Final boundary/topology and collision checks still apply;
triangle quality thresholds are not used. Work/size limits can still cause a
patch to retain its source mesh. `boundary_loops` in the per-patch log identifies
multi-loop cases; the PLY `remeshed` flag records accepted reconstruction.

No compilation or runtime validation was performed for this change.

## Current default: seeded planar constrained triangulation

To measure only single-boundary planar patch reconstruction:

```powershell
.\cad_mesh\remesh_saved.ps1 -SimplePlanesOnly -AnalyticWorkers 4
```

This opt-in mode skips construction for all other patch types, retains shared
boundary sampling and the analytic collision guard, and exports the partial
remesh before local splitting/collapse/flip/relaxation. The summary separates
boundary sampling, patch indexing, chart computation/ordered commits, face
assembly, and analytic time including collision guarding. Unprocessed patch
interiors may retain long edges; this mode therefore skips the full-mesh
maximum-edge acceptance test. Shared boundary insertion still affects both
incident patches. Without the switch the normal pipeline is unchanged.

CPU chart jobs are classified by surface type, boundary topology, and
power-of-two bands of estimated interior mesh size (area / target spacing
squared plus boundary vertices). Large jobs are submitted first; the fixed
worker pool takes the next job immediately without waiting for a patch-ID
batch to complete. Results remain keyed by patch ID and commits retain their
original order. Completed charts may consume more host memory while waiting
for earlier IDs. `computed` reports finished construction jobs independently
of committed patches. Waiting logs list running jobs and their elapsed times;
the final report lists the eight slowest construction jobs. This scheduling
change targets the current CPU construction path, not CUDA batch formation.

Single-boundary planar patches now bypass the older CUDA refinement path.
All patch interfaces and open boundaries are included in the global boundary
constraints and sampled once, preserving shared vertex IDs. The planar path
uses a bounded ear-clipped topology seed, local Delaunay legalization, a
staggered interior lattice at 0.85 times target spacing, and a local queue of
remaining overlong interior edges. Boundary segments cannot be flipped or
split by this per-patch path. Interior samples too close to the boundary are
omitted to avoid thin strips.

This implementation runs on CPU workers. It uses numerical orientation and
incircle predicates, not an exact-predicate geometry library. Construction
has explicit work/size limits. Failure retains the source patch without
retrying the previous planar refinement algorithm. Existing boundary,
topology, and quality acceptance checks still apply. Holes and periodic
surfaces retain their previous implementation; they are not covered by the
new planar path. `planar constrained mesh complete` reports attempted,
construction-failed, and accepted counts. No build or runtime validation was
performed for this change.

The command below is unchanged. `-CpuAnalytic` only controls the older CUDA
path and is unnecessary for the new default planar path.

Rebuild the native executable, then run from the repository root:

```powershell
.\cad_mesh\remesh_saved.ps1
```

## Previous CUDA path (superseded for single-boundary planes)

The script sets `CADMESH_REMESH_CHART_CUDA=1`. The older single-boundary planar
charts use CUDA after CPU boundary extraction and initial ear clipping.
Up to 16 consecutive patches are prepared together. CUDA maintains resident
point/triangle buffers while performing quality-improving flips, conforming
longest-edge splits, and independent vertex smoothing. Hash adjacency is
rebuilt on the GPU each pass; this version does not maintain incremental
adjacency. Only counters are downloaded between refinement rounds.

Each chart is handled by one cooperative thread block. Flips reserve all four
vertices; splits require both adjacent faces to select the same longest edge;
smoothing selects nonadjacent vertices. Boundary vertices are immutable.
Parallel operation ordering can change interior vertex IDs and triangulation.

Holes, closed surfaces, and charts with periodic surface coordinates retain
the CPU implementation. A single boundary triangle needs neither refinement
nor smoothing. GPU capacity exhaustion retries from the original chart with
larger buffers, bounded by 200,000 vertices/faces. Other failures, or exhausted
budgets, use the CPU implementation and report the fallback. CUDA runtime
failure disables subsequent chart GPU work for that remesh invocation.

Original topology, boundary, and quality acceptance checks still run before
ordered global commits. `completed` counts CUDA-produced charts; `accepted`
counts those that also pass the existing checks. Rejected results retain the
source patch. `gpu_pipeline_ms` measures the device timeline of the grouped
kernel launches, including launch gaps, excluding initial compilation/upload
and final geometry download; it is not the total wall time.

For a CPU comparison:

```powershell
.\cad_mesh\remesh_saved.ps1 -CpuAnalytic -AnalyticWorkers 4
```

Implementation has not been compiled or run as part of this change; execution
and mesh-quality validation are left to the user as requested.

## Local long-edge queue

The fallback `local split` stage builds adjacency once per invocation, then
keeps a queue of long edges. Each wave selects edges that are longest on all
incident faces. Only changed triangles and their edges update adjacency or
reenter the queue. An unselected edge sleeps until an incident neighborhood
changes. Equal lengths use the endpoint key to break ties deterministically.

This stage performs CPU local updates and does not upload the entire mesh to
CUDA each round. Boundary sampling still uses the existing CUDA classifier.
Incident triangle slots are replaced in place and children appended; arrays
remain dense without a full-array copy/compaction each round. Midpoints stay
on their source edges, patch labels are inherited, and constraint edges are
replaced by their two children. Edges touching frozen rebuilt patches, and
nonmanifold edges, are excluded from local splitting.

`SplitPasses` still bounds the number of waves per invocation. The final log
reports remaining eligible long edges, separately from frozen/nonmanifold
long edges, and whether the wave budget was exhausted. A zero pending queue
does not claim that the excluded edges satisfy the length target.
