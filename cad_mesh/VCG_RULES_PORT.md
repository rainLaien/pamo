# Native port of VCGLib isotropic remeshing rules

Supersedes NATIVE_ISOTROPIC_REMESH.md's approximate candidate rules. Implementation:
src/IsotropicPatchRemesher.h. Algorithm sources are the installed VCGLib headers
under D:/openSourceInstall/vcglib/vcg: complex/algorithms/isotropic_remeshing.h,
refine.h, smooth.h, space/triangle3.h and simplex/face/topology.h. The source file
preserves upstream attribution and GPL-2.0-or-later licensing for this port.
No VCGLib mesh types or remeshing APIs are called by this implementation.

The operator sequence is split, short-edge collapse, cross collapse, valence
improvement, Laplacian/fold relaxation, original-reference projection. It repeats
the configured number of iterations, without the former early convergence exit.

| Operator | Ported rule |
| --- | --- |
| Quality | Double triangle area divided by longest squared edge, not mean ratio |
| Split | One simultaneous midpoint pass above 4/3 target; all eight subdivision patterns and shorter-diagonal choice for two split edges |
| Collapse eligibility | Edge below 0.8 target OR incident area below minLength squared / 100 |
| Collapse placement | Midpoint if both endpoints movable, otherwise the fixed endpoint; no immediate projection |
| Crease movement | At most two incident feature edges, direction dot magnitude >= 0.9, collapse edge must itself be a feature |
| Collapse geometry | Every surviving face quality > half its old quality; normal dot >= 0.7; resulting incident edges <= maxLength |
| Cross collapse | Three/four incident faces, relaxed length/area conditions, same quality/normal/link tests |
| Flip | Sum of absolute valence errors, target 6 interior / 4 boundary; exact 0.5, 1.0, 1.5 quality-ratio alternatives |
| Flip orientation | Both new normals within 5 degrees of both old normals |
| Laplacian | Face-edge accumulation (interior edges counted twice), self-inclusive average, 0.2 blend; no tangential line search or global minimum-quality veto |
| Fold relaxation | 140-degree fold detection; two face-order sweeps with sweep-start accumulators |
| Projection | After the operators, against the immutable source region |

Creases are tagged initially (default feature angle 10 degrees), propagated through
split/collapse, and retained through flips. Degenerate support is treated as in the
upstream crease tagger using the radii-quality cutoff. Nonmanifold vertex stars
are excluded from vertex movement. Native storage uses indexed faces, deletion
flags, incidence sets and incrementally updated edge adjacency; sequential
face-order commits replace the prior disjoint candidate-batch approximation.

Explicit application-specific differences from an unrestricted VCGLib run:

- Original shared interface edges/endpoints remain fixed; VCGLib normally permits
  some boundary and crease collapses. Internal crease vertices can now move along
  qualifying creases, unlike the previous blanket crease lock.
- surfDistCheck is disabled and cleanup is disabled per the requested direct
  adoption policy. Initial input cleaning belongs to the saved topology stage.
- Projection uses our nearest-triangle BVH rather than VCGLib's finite-radius
  uniform-grid lookup, and skips fixed interface vertices.
- Face/vertex storage, floating-point evaluation and ordering can differ from
  VCGLib pointer-based allocation. Bitwise identical meshes are not claimed.
- Adaptive sizing, selected-face mode and optional cleanup variants are not
  exposed here; this port targets the nonadaptive per-subregion workflow.

Remaining Others dispatch and per-face 0/1/2/3 status semantics are unchanged.
Length-compliant skinny input remains preserved as requested. No final result
quality/deviation/collision rollback is reintroduced. This is a CPU reference
implementation; no GPU equivalence or measured performance is claimed.

Use the existing remesh_others.ps1 parameters GenericFeatureAngleDegrees (10)
and GenericRemeshIterations (5). Logs use native vcg-rules iteration/complete.
No compilation, tests or runtime comparison were performed, per user instruction.
