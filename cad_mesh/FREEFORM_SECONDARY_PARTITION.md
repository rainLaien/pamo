# Local model partition after preserving good triangles

The saved partition remains unchanged. Only the Freeform remainder after input
quality selection is reconsidered. Existing constrained edges and kept triangles
block connectivity. Growth tests seven distance samples and the face normal
against a fixed candidate model; every accepted face passes these tests.

Candidates first reuse up to 64 analytic Plane/Cylinder models listed as neighbors
of the parent patch. New plane candidates use large remaining source triangles.
No new nonlinear cylinder fitting is introduced in this version. Models commit in
deterministic order. Cylinders still require at least eight faces. Planes have no
minimum face count: support area must reach 0.25 * target length squared, and
area / (twice the maximum vertex distance from a seed vertex) must exceed four
plane distance tolerances. This conservative width proxy rejects nearly collinear
support without rejecting a region just because it has only one or two triangles.
Per-parent limits
are 128 seed attempts and roughly one million face tests. Untouched remainder is
partitioned by edge connectivity and sent through reference-mesh repair.

Plane identification uses a separate distance tolerance: by default 0.0003 times
target length, capped by remesh MaximumDeviation, plus numerical slack. Its normal
tolerance defaults to one degree, also capped by the remesh normal limit. All faces
are tested against the fixed candidate plane. These are identification thresholds,
not stricter remesh output acceptance thresholds. NativeRemeshConfig exposes
SecondaryPlaneDistanceTolerance (zero means automatic),
SecondaryPlaneNormalToleranceDegrees and SecondaryPlaneMinimumAreaRatio; no new
command-line flags are introduced. With target length 6 and remesh deviation 0.1,
the plane distance threshold is approximately 0.0018 and minimum area is 9.
Logs include plane_distance_tolerance, plane_minimum_area, planes_below_8_faces
and plane_support_rejected (candidate attempts, not unique regions).

Accepted plane/cylinder subregions try the existing analytic chart remesher with
the outer surface-only filter explicitly bypassed: the outer selected patch is
still Freeform. Candidate outputs are sampled against the subregion's original
triangles before acceptance. A failed analytic attempt falls back to local generic
repair. Region-level generic failure retains that region's original faces.

Subregion interfaces reuse original global vertex IDs and existing boundary edges.
No independent interface sampling is permitted. An analytic chart whose fixed
interface cannot satisfy its requirements falls back rather than introducing a
one-sided split. A future shared subregion sampler can increase analytic coverage.
Collision replay uses the reference path to honor protected original geometry.

New logs: `freeform secondary partition` reports model counts, remaining components,
neighbor candidates, face checks, budget exhaustion and time. `freeform analytic
subregion` reports the local type and analytic acceptance. Child types are internal;
PLY patch IDs and primitive types still describe the original partition. Final
collision handling remains parent-patch scoped and can still revert a large patch.

No build, tests or remesh validation were run, per user instructions.
