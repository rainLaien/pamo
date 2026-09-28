# Phase 26: identify the limiting interface before changing constraints

**Later source-evidence correction (phase 34):** CADPART1 labels patch 3842
as a producer-identified Fillet with support patches 607 and 3843. Thus the
near-zero 3842/3843 dihedral does not justify treating that interface as a
computation-only seam. It remains fixed pending a shared-curve motion
contract. See `PHASE34_PATCH_PROVENANCE.md`.

This phase reuses the accepted `2.stl` output and its completed-output
snapshot. No candidate was promoted, no default remesh algorithm was changed,
and no complete original-STL timing was claimed. The accepted full flow
remains 103.09 seconds at GPU concurrency 8.

## Controlled neighbor candidate

With the same target 5.0 and original geometry budget as phase 25, the
existing `--constrained-neighbors` mode rebuilt only patch 3842 and
retessellated adjacent source triangles while preserving all boundary
stations. Its output had 500527 faces, one connected component, Euler -1,
81 open edges, and no detected nonmanifold, winding, duplicate, or float32
zero-area faces. It was rejected before original-STL geometry and native-size
audit:

| Measure | Accepted incumbent | Constrained-neighbor candidate |
| --- | ---: | ---: |
| Mean quality | .568358022 | .567916524 |
| Area-weighted quality | .752342857 | .752023263 |
| Low-quality area | 4727.793 | 4671.301 |
| Largest low-quality component | 55.522 | 53.096 |

Patch 3842 low-quality area fell 59.770 to 3.210. Neighbor 489 rose 0.539 to
0.652, 607 rose 0.075 to 0.823, and 3843 fell 6.266 to 5.473 but lost P05.
The global mean and area-weighted quality regress, so the existing whole-mesh
quality gate correctly rejects this candidate. No geometry or speed benefit
is attributed to it.

An existing exact-axis-plane diagonal optimizer changed 130 pairs elsewhere
but none in 489/607. A bounded opt-in prototype allowing four float32 ULPs
of local noncoplanarity also changed **zero** pairs in 489/607. It passed a
small synthetic test, but gave no benefit on this input, so the prototype
source and test changes were removed. The saved experimental reports remain
under `results/phase26_neighbor_contract`; the accepted output is untouched.

## Why direct seam unlocking is unsafe

The completed-output snapshot carries feature records for every edge of the
relevant interfaces. Measured adjacent face-normal angles and source hard
bits are:

| Interface | Segments | Angle | hard=1 | hard=0 |
| --- | ---: | ---: | ---: | ---: |
| 489 / 3842 | 30 | 13.9289° | 0 | 30 |
| 489 / 3843 | 25 | 13.9289° | 25 | 0 |
| 607 / 3842 | 50 | 13.9272–13.9281° | 0 | 50 |
| 3842 / 3843 | 25 | 0° | 0 | 25 |

These are source-snapshot observations, not a full CAD feature
certification. They nevertheless show that treating all hard=0 partition
interfaces as computational seams would erase a roughly 14° crease. The
3842/3843 interface is coplanar and a plausible computational seam; the
489/607-side interfaces need a feature-curve contract. Of 396 (patch 489)
and 36 (patch 607) near-planar diagonal pairs that met the global-h length
screen, **all** touched a patch-seam vertex and were excluded by the current
hard boundary protection. More generous coplanarity tolerance alone cannot
solve the neighbor loss without changing the seam contract.

## Decision

Stop single-face neighbor triangulation and opportunistic diagonal flips for
this region. A future candidate needs a connected cavity across the
3842/3843 computational seam, with the 489/607-side crease classified and
retained as a shared curve. It may resample along that curve only if both
sides are retessellated together and original-input geometry, native sizing,
orientation, feature chains, and the complete regional quality transaction
pass. Preserve the incumbent for rollback. This is a bounded next design
target, not a claim that it will pass or meet 90 seconds.

`results/phase26_neighbor_contract/repair.json` and `quality.json` contain
the constrained-neighbor stage candidate and comparison. `flip.json` and
`near_flip.json` retain the two flip checks. The rejected candidates must not
be confused with the published output.
