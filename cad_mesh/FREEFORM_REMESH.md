# Freeform patch remesh

Default runs and `-OtherFeaturesOnly` now process Freeform patches using an isolated
copy of each patch's triangles after shared boundary sampling. Plane/cylinder/cone
only modes continue to skip Freeform. Analytic patches do not enter this path.

The patch's original triangles form a local reference index. Boundary edges and
existing constraint edges are fixed; fixed vertices retain their global IDs.
Interior points are owned by the patch. Processing uses the local long-edge queue,
two collapse passes, two flip passes and two relaxation iterations. These local
operations reuse relative quality and normal checks, but no absolute quality target
or final whole-model quality measurement is introduced. Pass limits are independent
of the obsolete global postprocessing options.

Collapse, flip and relaxation now check candidate triangle vertices, edge midpoints
and centroids against the reference before committing. A deviation failure rejects
only that operation. The guard runs after cheap geometry/quality checks and before
collision queries; rejected candidates do not enter the accepted collision set.
Logs show per-stage rejected-operation counts, query counts, check time and the
last rejected sample coordinates/distance. Initial edge splits stay inside the
original triangles and do not need this projection check.

Final validation remains as a fallback and reports the failing local vertex/face/
edge IDs and sample coordinates, distance and tolerance if it fails.
Validation requires unchanged boundaries and constraints, valid edge incidence,
nondegenerate and unique triangles, and reference distance within MaximumDeviation
at used vertices, edge midpoints and face centers. This is sampled deviation, not a
continuous Hausdorff bound. Unchanged patches can complete successfully when no
eligible local operation is accepted. A bounded run does not guarantee all remaining
edges meet the requested target. The existing collision guard runs after assembly
and restores rejected patches from the original boundary-sampled faces.

Logs contain per-patch stage timing and failure details. `remeshed=1` and
`remesh_reason=0` now also include accepted Freeform processing. Per-type eligibility
is named `remesh_eligible`; historical `analytic` totals now include Freeform jobs.

Freeform collision checks now run only after assembly, using the existing guard
and rollback. Local collapse/flip/relax no longer build collision indexes or populate
dynamic collision sets. Boundary, topology, normal and sampled reference-deviation
checks still run before local edits are accepted. Deferring collisions can increase
final patch rollback if candidate geometry intersects; it does not establish that
fixed boundaries alone prevent intersections.

Reference deviation predicates stop as soon as any reference triangle is within
the tolerance radius. A last-hit triangle hint is tested first, with a complete
radius-pruned BVH search on misses. This is not cached acceptance: each new sample
is evaluated against the immutable reference. Exact nearest queries remain for
vertex projection and failed-sample diagnostics. The tolerance remains unchanged.

This version processes Freeform patches serially. Large patches may remain
expensive because collapse/flip/relax still rebuild local adjacency.
No build, tests or remesh runs were performed, as requested.
