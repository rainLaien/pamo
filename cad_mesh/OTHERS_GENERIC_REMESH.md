# Others: generic remesh with direct adoption

Others (Cone, Sphere, Torus, Freeform and Unknown) bypass analytic chart
construction and its model-deviation eligibility test. Plane/Cylinder chart
behavior is unchanged. Surface-only selection still applies, including Others
and Cone-only runs. The existing saved-partition entry point remains
`cad_mesh/remesh_others.ps1`.

For each selected parent patch:

1. Preserve input triangles whose longest edge satisfies target length, with
   existing numerical slack. Output these as per-face `remeshed=3`.
2. Remove those faces from the repair adjacency graph. Split the remaining graph
   into edge-connected regions without crossing constrained or nonmanifold edges.
   This is connectivity partitioning, not analytic model fitting.
3. Lock region boundaries, including interfaces with preserved triangles, using
   their original vertex IDs. Run the project's generic split/collapse/flip/relax
   implementation, projecting moved vertices to each region's source mesh.
4. Adopt the generated mesh directly as `remeshed=2`. No post-construction
   topology, quality, deviation or collision acceptance/whole-region rollback.

This uses the project's existing generic operations, not a VCGLib API call.
Deviation rejection is disabled during operations as well, and the local normal
change limit is opened to 180 degrees. Algorithmic link conditions and quality
improvement choices intrinsic to collapse/flip/relax still govern individual
operations; they do not reject the finished region. Nonmanifold edges are locked
and excluded from region connectivity rather than rejecting a whole parent.

The default remains bounded: long-edge queue followed by two collapse passes,
two flip passes and two relaxation iterations. A split stop diagnostic does not
discard the current mesh. No output quality or intersection-free guarantee is
implied by status 2. Status 3 means deliberately preserved input, not newly
generated faces that happen to satisfy target length.

Earlier secondary Plane/Cylinder identification inside Freeform is bypassed by
this path. No build, tests or remesh execution were performed for this change.
