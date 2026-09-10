# Native isotropic patch remesh

The temporary VCGLib adapter has been removed. IsotropicPatchRemesher.h owns the
iteration driver and tangential relaxation, reusing native edge-queue splitting
and local collapse/flip primitives. No VCGLib remeshing API is called. The project
still uses its pre-existing VCGLib dependency for other functionality such as STL
input; this change does not remove that dependency from the whole project.

Pipeline: preserve target-length input (status 3), independently repartition the
remainder using model-first recognition, dispatch analytic children to chart
construction and Freeform children to the native isotropic driver (status 2).
Source patch IDs remain the exported parent IDs; children are internal regions.

Each of five default iterations performs:

- Queue-based splitting above 4/3 of target length.
- Collapse below 0.8 of target length with link conditions and local orientation
  checks; relax the previous strict non-degradation veto.
- Four flip passes allowing minimum-quality improvement or valence improvement
  with limited quality loss (interior target valence 6, border 4).
- Four tangential relaxation sweeps with projection, local line search and
  disjoint-star commits. CSR adjacency and separate candidate positions/scores
  permit later batched/device evaluation without a mesh-library data model.

FeatureAngle defaults to 10 degrees: adjacent-face normals identify crease edges,
which are constrained together with shared boundaries. This differs from the
89-degree per-operation normal-change limit used to prevent inversion in the
collapse/flip primitives. The current crease policy fixes crease vertices rather
than moving them along curves. It may be more restrictive than VCGLib.

Unlike the abandoned selected-face VCGLib adapter, no entire boundary-adjacent
face ring is frozen. Only constrained edge endpoints are fixed. No final quality,
deviation or collision acceptance/whole-region rollback is introduced. Input
already preserved by edge length remains untouched even if skinny. Narrow regions
and dense fixed boundaries can still limit quality; status 2 is not certification.

This is a CPU implementation, not a CUDA kernel or an exact VCGLib reproduction.
Collapse and flip still use host adjacency containers; CSR relaxation is the first
explicit candidate/commit separation. Future GPU work also needs a device
reference index, conflict resolution and dynamic connectivity updates.

```powershell
.\cad_mesh\remesh_others.ps1 -GenericFeatureAngleDegrees 10 -GenericRemeshIterations 5
```

Log names: native isotropic iteration / native isotropic complete. No compilation,
tests or remesh execution were performed, per user instruction.
