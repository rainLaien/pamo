# Model-first repartition of the Others remainder

This supersedes the connectivity-only dispatch in OTHERS_GENERIC_REMESH.md.

Selected Others patches retain triangles satisfying target edge length as status
3. The remaining faces, excluding these barriers, are copied into an indexed local
topology without welding or renumbering source faces. Existing constrained edges
are explicitly marked hard. The original mesh resolution and segmenter's
configuration are passed to PartitionBySurfaceModels, the same model-first engine
used in the initial partition. This includes independent plane/cylinder and other
analytic model searches, rather than only borrowing neighboring models.

Each returned child's source face IDs are mapped back to the parent input:

- Plane and Cylinder use their existing seeded constrained chart construction.
- Cone, Sphere and Torus use the existing corresponding analytic chart path.
- Freeform uses generic reference-mesh split/collapse/flip/relax.

Unknown/nonanalytic classifications become reference-mesh Freeform. If indexed
topology cannot be built or partition ownership is incomplete, the connected
reference regions remain available instead of losing source faces. Model fitting
still uses geometric certification to determine type; this is separate from
remesh-result acceptance. Child analytic reconstruction bypasses outer surface
selection, pre-construction model-deviation rejection and final result validation.
No post-assembly collision rollback is restored. An analytic constructor that
cannot produce a chart retains only that child's original faces as status 1; it
does not switch an analytic child to generic remeshing or discard sibling results.

All child boundary vertices alias their original global IDs. New interfaces are
fixed and are not independently resampled on either side. Long fixed interfaces
can therefore still cause analytic construction failures. Generated child faces
use status 2 and preserved input uses status 3. PLY patch_id and primitive_type
remain parent labels; the new local model-first partition log gives child counts
by type, input face count, fitting tolerance and partition time.

The saved-partition remesh_others.ps1 command remains unchanged. Independent
fitting adds work; no performance claim is made before a user run. No compilation,
tests or remesh validation were executed.
