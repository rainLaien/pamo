# Phase 33: fixed-boundary recovery in neighbor 607

The production remesher and accepted `2.stl` output remain unchanged. This is
a bounded completed-output stage experiment on the rejected phase-32
candidate; it is not a raw-STL complete-flow timing or final certification.

## Why patch 607 was selected

Conforming repair removed all 67 new native long edges, but patch 607's
area-weighted quality fell from 0.878766 in the accepted mesh to about
0.7662. The completed candidate has 371 faces in that planar patch, with a
235-vertex outer boundary and 69 interior vertices. An existing ordinary
interior diagonal divides it into two fixed-boundary cavities of 118 and 119
boundary vertices, within the 128-point triangulator work limit. All
feature records lie on the retained boundary; no curve station or vertex
was moved.

## One bounded retessellation

The existing constrained planar triangulator was applied separately to the
two cavities. It initially omitted two fixed interior stations from the
196-face cavity. One lay strictly inside a proposed face; one lay on an
internal diagonal. The stage tool restored both at their original positions
by splitting the containing face or both incident faces. It then required
all original vertices and the original face count to be retained. Both
cavities are representable and the resulting whole mesh has one component,
Euler characteristic -1, 81 open edges, and no detected nonmanifold,
winding, duplicate, float32 zero-area, or missing-feature edge.

The strict local rule accepted only the 175-face cavity: minimum, mean and
area-weighted quality did not decrease and low-quality area did not grow.
The 196-face cavity improved mean and area-weighted quality but increased
its low-quality area from 0.114073 to 0.117456, so the strict rule rejected
it. A separate diagnostic-only output includes both cavities for measuring
the full tradeoff; it was not accepted by that rule.

| Measure | Accepted mesh | Phase-32 conforming candidate | 607 strict cavity | 607 both-cavity diagnostic |
| --- | ---: | ---: | ---: | ---: |
| Whole-mesh mean quality | 0.568358 | 0.567936 | 0.567954 | 0.567969 |
| Whole-mesh area-weighted quality | 0.752343 | 0.752024 | 0.752095 | 0.752156 |
| Whole-mesh low-quality area | 4727.793 | 4670.041 | 4670.041 | 4670.044 |
| Largest poor connected area | 55.522 | 53.096 | 53.096 | 53.096 |
| Native short edges | 676458 | 677521 | 677533 | 677542 |
| Native long edges | 0 | 0 | 0 | 0 |

Patch 607's area-weighted quality recovers to about 0.802 in the
both-cavity diagnostic, still far below the accepted mesh's 0.879. Its
low-quality area grows from 0.075 to 0.286 while the whole mesh's native
short-edge count also grows. The fixed-boundary retessellation cannot recover the neighbor
without a material cost under these measured candidates.

## Decision and limit

Keep the accepted phase-24 output. Neither 607 variant meets the existing
whole-mesh quality endpoint or native short-edge nonregression. The
diagnostic-only variant also violates one cavity's low-quality-area guard.
No independent raw-STL geometry audit was run for these rejected variants;
the phase-32 conforming parent did pass the sampled audit. The 607 trial
retains coordinates and feature records, but that does not by itself certify
the new piecewise-linear surface against the original STL.

The next design must address boundary-station placement and size transition
as a connected region problem. Moving stations requires positive evidence of
which interfaces are computational seams and which are geometric curves;
the CADPART1 hard bit alone does not provide that distinction. Further
fixed-boundary diagonal or scalar-h sweeps on this same interface are not
justified by these results. The latest accepted complete-flow measurement
remains one 103.086 s run; the 90 s target is still open.

## Reproduce the stage comparison

```powershell
python experiments/rxmesh-remesh/results/phase27_cavity/probe_neighbor607.py
python experiments/rxmesh-remesh/results/phase27_cavity/probe_neighbor607.py --include-first-cavity-for-diagnosis
python experiments/rxmesh-remesh/results/phase27_cavity/audit_candidate_sizing.py --candidate experiments/rxmesh-remesh/results/phase27_cavity/candidate_neighbor607.ply --candidate-snapshot experiments/rxmesh-remesh/results/phase27_cavity/candidate_neighbor607.cadpart --output experiments/rxmesh-remesh/results/phase27_cavity/neighbor607_strict_sizing.json
python experiments/rxmesh-remesh/results/phase27_cavity/audit_candidate_sizing.py --candidate experiments/rxmesh-remesh/results/phase27_cavity/candidate_neighbor607_both.ply --candidate-snapshot experiments/rxmesh-remesh/results/phase27_cavity/candidate_neighbor607_both.cadpart --output experiments/rxmesh-remesh/results/phase27_cavity/neighbor607_both_sizing.json
```

No commit or push was made.
