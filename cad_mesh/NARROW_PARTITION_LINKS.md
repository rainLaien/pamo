# Narrow-link partition sampling

Current scheduling: ruled-strip proposals and an early cylinder/cone ring
absorption now precede strict planar core extraction. A second absorption stage
follows residual seed discovery. Both cylinders and cones participate, with
fixed fitted parameters; each stage allows 100000 evaluations (200000 total).
This gives already-recognized cylindrical fillets the same early opportunity,
but does not introduce a general arc/fillet detector. Torus/freeform blends are
not covered by this priority. Complete-component certification still runs first.

Certified cylinders now receive a frontier-only absorption stage after residual
seed discovery and before final residual classification. It ignores discovery
soft barriers and triangle length/aspect ratio, but uses the original hard-edge
filtered adjacency. Parameters and initial growth tolerances remain fixed.
Neighboring analytic models compete on the same ownership snapshot; ambiguous
near-ties across distinct models are deferred. Each accepted ring generates the
next frontier. Limits are 12 rings and 200000 compatibility evaluations per run.
Only unowned faces are eligible; established planar/other patch ownership is
not stolen. Updated cylinder statistics are recomputed without parameter refit.
This is adjacent-layer completion, not forced 360-degree closure or a mechanism
to cross a non-compatible fillet. Logs report accepted faces and budget status.

An additional ruled-strip proposal stage precedes generic curve discovery.
Slivers with two long edges within 10 percent in length and absolute direction
dot product at least 0.98 are eligible. Long-side adjacency assembles up to
128 faces, with length within 0.75--1/0.75 of the seed and direction within
0.35 radians (sign independent). At least six faces and resolved rotating
normals are required before proposing cylinder/cone fits. Existing support,
growth, refit and certification still decide acceptance. Recognized strip
faces take precedence over isolated slivers in subsequent seed selection.
The extra stage has at most min(256, ModelMaximumSeeds) proposals; it is not
debited from the original generic seed budget. CPU avoids speculative duplicate
fits; CUDA uses the existing prefetch mechanism. General discovery remains the
fallback for nonuniform, heavily trimmed or strongly convergent cone strips.
This implementation does not yet derive an explicit cylinder axis or cone apex
from the long edges; it uses them to select cleaner fitting support.

Sliver handling now supplements interface-width detection. A triangle with
2*area/longest_edge_squared below 0.05 defers traversal across its shortest
edge; its long edges remain eligible. This is only a discovery heuristic,
not a topology edit or a claim that all slender faces are spurious. Candidate
seeds prefer non-slivers within each cell, retaining slivers as a fallback.
Before residual classification, unowned slivers compare certified analytic
models incident to their two longer edges. Distinct models with scores within
0.1 remain ambiguous and are left for residual processing. Decisions use one
ownership snapshot; connected accepted groups are stored and can later merge
under the existing model-identity/union checks. No faces are deleted.

The model-first pipeline now aggregates shared interface length between
distinct provisional planar cells after cell construction. An interface is
deferred during discovery when its length is below 0.2 times the square root
of the smaller cell area. No mesh edges or faces are removed or marked as hard
features. This is a scale-invariant heuristic, not a proof of a geometric neck;
results still depend on provisional cell construction. Bottlenecks internal to
one cell are not detected by this version.

Cell-neighborhood proposals, model growth, exposed-component discovery and
last-chance residual sampling stay on one side of these links. Residual
classification uses the same separation so failed sides are not automatically
recombined merely by connectivity. Every remaining face is still stored.
Existing final analytic adjacency consolidation can reunite compatible sides
after model identity and geometric certification; small residual absorption
keeps its stricter existing normal checks. No extra stochastic trials or new
fit budget are introduced. Whole-component certification still precedes this
heuristic and may accept a complete component when it fits one model.

Diagnostics report `narrow sampling links` (face-neighbor link count). A zero
count means this heuristic found no such interface; it does not mean all thin
connections are absent. No compilation or model run was performed.

Cylinder/cone ring absorption now permits face-to-model normal error up to
30 degrees. Initial fitting, core certification, planes and other surface
types retain their configured normal thresholds. Fitted parameters and the
growth distance envelope stay fixed throughout absorption. Faces exceeding
the original normal threshold additionally require model normals at their
vertices to agree with the facet normal within 30 degrees. Interior chord
points are not tested against the vertex distance tolerance: a flat source
facet naturally deviates from its analytic surface there. A relaxed candidate
competing with a distinct compatible model is deferred. This is classification,
not a guarantee about continuous remeshed surface error. Hard-edge adjacency
barriers remain in force.
The absorption diagnostic includes `normal_limit_deg`, `relaxed_faces`,
and `normal_rejections`.

After residual classification, a separate bounded pass reconsiders Plane and
Freeform patches containing at most 32 faces, all marked slender. Whole strips
can move into a neighboring Cylinder/Cone with at least two long-side face
contacts, strict vertex distances and 30-degree face/model normal agreement.
Distinct analytic neighbors on long sides and competing incompatible ruled
models prevent reassignment. Decisions use frozen ownership and parameters;
source patches are transferred atomically, empty patches removed, and metadata
and ids rebuilt. No faces are deleted. Large Freeform regions are not split
by this pass. The pass is capped at 50,000 face/model evaluations.
