# Phase 32: conforming repair of the local h=5 candidate

The accepted `2.stl` output and production algorithm remain unchanged. This
is one bounded completed-output stage candidate built from the rejected
phase-27 weighted cavity mesh. It is not a complete original-STL timing run.

## Precise remaining size defect

The phase-27 h=5 candidate had 67 prospective native long edges: 17 on
recorded feature/partition-interface edges, 50 ordinary same-patch interior
edges, and none on open boundaries. The local h=5 field was reconstructed
from the immutable canonical snapshot; the accepted mesh's saved native
targets reproduced its own reported 676458 short and zero long edges exactly.

`probe_conforming_size.py` split all 67 long edges once at their existing
piecewise-linear midpoints and retriangulated every incident face with its
full boundary subdivision. It split 17 feature records using the same
identity and hard flag, changed 111 source faces, and added 134 faces. It
did not move an existing vertex. This is a feasibility probe, not a general
connected-region production operator.

| Measure | Accepted phase-24 output | Conforming candidate |
| --- | ---: | ---: |
| Faces | 499935 | 500661 |
| Mean quality | 0.568358022 | 0.567935967 |
| P05 | 0.032024174 | 0.032281201 |
| Area-weighted quality | 0.752342857 | 0.752024182 |
| Low-quality area | 4727.792922 | 4670.040771 |
| Largest poor connected area | 55.522459 | 53.095598 |
| Native short edges | 676458 | 677521 (+1063) |
| Native long edges | 0 | 0 |

The connected [489, 607, 3842, 3843] area has defect severity 13.510978 to
3.396670 and low-quality area 66.650311 to 8.898160, but its area-weighted
quality falls from 0.519622 to 0.503988. Patch 3842's low-quality area falls
59.770 to 3.210. Patch 607's area-weighted quality falls 0.878766 to
0.7662, while its low-quality area grows 0.075 to 0.283. This is a material
neighbor regression, not a harmless numerical fluctuation.

## Hard checks against the original input

The independent checker found one connected component, Euler characteristic
-1, 81 open edges, and zero nonmanifold edges, winding conflicts, duplicate
faces, float32 zero-area faces, or missing feature records. Against the
canonical completed-output snapshot, all 100586 feature chains, 99806
partition-interface chains, and 81 open-boundary chains remain continuous,
with zero missing anchors. Raw `examples/2.stl` bidirectional sampled
distances are 0.977330 / 1.304286, both within the unchanged 1.524741292
budget. The candidate has the same 106 nonpositive nearest-source-normal
centroid samples as the accepted output, all outside changed patches. These
checks are sampled and topological evidence, not continuous Hausdorff or
self-intersection proofs.

## Decision

Reject as a default final result. Coordinated midpoint splitting eliminates
the 67 newly created long edges without breaking the observed feature,
seam, or topology constraints. The large gain in the thin strip comes with
substantial patch-607 quality loss, 1063 additional native short edges, and
a small whole-mesh mean and area-weighted loss. No quality tradeoff budget
was selected before this candidate, so these measurements cannot be turned
into a post hoc acceptance threshold. A naive offline postpass took 9.3 s
for the conforming split alone; the 89.4 s independent audit was separate.
Neither number is a full pipeline result, and this candidate has not been
timed from raw STL. The latest accepted complete-flow evidence remains
103.086 s, above the 90 s target.

The next quality implementation should optimize the connected planar
neighbor cavity around patch 607 under the shared-boundary and native-size
contract. It must recover neighbor area-weighted quality and short-edge
distribution while retaining the strip's low-defect gain; otherwise keep
the accepted mesh. Further scalar h sweeps or unguarded midpoint passes do
not address the measured tradeoff.

## Reproduce the stage evidence

```powershell
python experiments/rxmesh-remesh/results/phase27_cavity/audit_candidate_sizing.py
python experiments/rxmesh-remesh/results/phase27_cavity/probe_conforming_size.py
python experiments/rxmesh-remesh/results/phase27_cavity/audit_candidate_sizing.py --candidate experiments/rxmesh-remesh/results/phase27_cavity/candidate_conforming.ply --candidate-snapshot experiments/rxmesh-remesh/results/phase27_cavity/candidate_conforming.cadpart --output experiments/rxmesh-remesh/results/phase27_cavity/conforming_sizing.json
python experiments/rxmesh-remesh/tools/audit_phase1.py --source examples/2.stl --snapshot experiments/rxmesh-remesh/results/phase16_polygon/canonical_incumbent.cadpart --repair-report experiments/rxmesh-remesh/results/phase24_concurrency/full_gpu8/strip_repair/pass_01/repair.json --baseline experiments/rxmesh-remesh/results/phase24_concurrency/full_gpu8/remeshed.ply --candidate experiments/rxmesh-remesh/results/phase27_cavity/candidate_conforming.ply --output experiments/rxmesh-remesh/results/phase27_cavity/conforming_original_geometry.json --threshold 0.026690566912293434 --target 7.623706340789795 --max-error 1.524741292
```

No commit or push was made.
