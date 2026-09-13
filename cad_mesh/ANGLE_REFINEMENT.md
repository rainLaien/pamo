# Native 28-degree chart refinement

STATUS: Disabled from the active remesh path at the user's request after a
reported slowdown. PlanarDomain and Others preservation have been restored to
the committed pre-refinement baseline. The historical description below does
not describe the active default. Shared boundary sampling and patch workers
remain; cylinder axial sizing again uses the global target. No benchmark run.

Plane and cylinder charts request a 28-degree minimum angle after successful
construction. The implementation uses the existing native constrained edge
legalizer with circumcenter/off-center Steiner insertion; it does not call or
claim numerical equivalence to the Triangle library. Cylinder refinement runs
in physical (radius * angle, height) coordinates after anisotropic scaling is
undone, so the shape request is not measured in a distorted axial metric.

A boundary-derived size field uses min(target, 1.25 * segment_length +
0.3 * distance_to_segment). This is a spatial grading rule, not a guarantee
that adjacent triangles differ by exactly 25 percent. Fixed segments' diametral
disks are protected; physical boundaries and paired seams are never subdivided
independently. Interior source diagonals may change.

The request is bounded to 12 sweeps, at most 16000 additional points (also
limited by starting size), and two million boundary comparisons. No progress
ends refinement. A numerical failure leaves the previously constructed chart
available instead of restoring source patch geometry. Thus 28 degrees is a
best-effort request, particularly at small fixed angles and dense constraints.

Matching constant-height cylinder end rings keep the source angular columns.
Axial row spacing is limited using twice the median angular segment length.
Other trimmed/periodic cylinder constructions retain their existing layout and
receive the same physical-domain quality refinement.

Cylinder legalization also keeps the circumferential size bound after axial
scaling is undone. A flip cannot increase the angular span of its two incident
triangles or exceed the circumferential target. Matching end-ring layouts
protect axial column edges during construction and quality refinement. These
curvature constraints take priority over the 28-degree request, so improving
the planar angle cannot coarsen the cylinder's angular tessellation. Existing
shared boundary vertex positions remain fixed.

Preserved Others triangles now require both the target edge bound and minimum
angle of 28 degrees. Thin triangles enter repartition/remesh rather than being
marked as preserved. The Freeform isotropic solver itself is unchanged here;
the new constrained-chart insertion applies to planes and cylinders, including
newly recognized children, not arbitrary Freeform surfaces.

No build or runtime tests performed, as requested.

## Cylinder layout cleanup

Matching constant-height rings now size axial rows at approximately 0.866
times the median angular interval, capped by 0.85 times the physical axial
target. They skip scattered angle insertion and scalar diagonal splitting;
angular columns and axial row spacing set the geometry instead. This applies
only to the matching-ring path, not every trimmed cylindrical patch.

Interior flips are no longer prohibited merely because an edge is axial.
They still preserve the previous angular envelope and circumferential bound.
Irregular cylinder charts receive up to three short-edge collapse sweeps after
successful optional quality refinement. Only edges joining ordinary interior
vertices are eligible. The manifold link condition, positive UV orientation,
angular span, length and non-worsening local minimum shape quality are checked.
Existing endpoints are retained; fixed boundaries and periodic aliases cannot
collapse. Disjoint one-rings are processed per sweep and faces compacted once
per sweep. These are local construction checks, not whole-patch acceptance
tests or post-remesh collision/deviation rollback.

## Shared sizing and graded plane sampling

Cylinder axial spacing now uses min(global target, 2 * circumferential target).
The boundary splitter and chart construction use this same rule, so the demand
is applied once on shared edges and inherited by both incident patches. This
applies to analytic cylinders, including cylindrical fillets; it does not
reclassify freeform fillets. Boundary sampling no longer excludes a cylinder
solely because its fitted deviation exceeds the old acceptance threshold.

Plane initial sampling now queries a local immutable boundary BVH for
min(target, 1.25 * boundary segment length + 0.3 * distance). Adaptive square
cells subdivide near fine boundaries and grow toward the global target in the
interior. Even-odd containment handles holes and disconnected domains; candidates
too close to the boundary are omitted using the local size. Boundary vertices
and constraints are never changed by a chart worker. The BVH avoids scanning
every boundary segment for every initial sample. The later angle-refinement
pass retains its existing bounded implementation.

Initial sampling is limited to 200000 visited cells, 80000 chart points and
160000 chart faces, leaving capacity for edge repair. Reaching the optional
sampling budget keeps the current chart and proceeds to repair; construction
failures still follow the existing failure path. Cell traversal is breadth-first
to distribute work across the patch. The worker pool and deterministic assembly
are unchanged. Increased boundary and interior density can increase runtime and
face count; no measured performance or visual result is claimed.
