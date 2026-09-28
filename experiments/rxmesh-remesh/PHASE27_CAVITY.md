# Phase 27: connected neighbor cavity candidate

The official `2.stl` result is unchanged. This is an offline completed-output
candidate using the phase-26 patch-3842 target-5.0 reconstruction as input.
It does not change the default pipeline, GPU operators, partitioning, or
published mesh. The complete original-STL timing remains 103.09 seconds at
GPU concurrency 8.

## Bounded region and method

The earlier reconstruction changed 98 adjacent source faces: 25 in patch 489,
49 in 607, and 24 in 3843. For 489 and 607, the probe selected faces
containing newly inserted vertices and one same-patch face ring. This yielded
five simple cavities in 489 and three in 607. They had no internal source
feature edges. The largest 607 cavity had 153 boundary vertices; an existing
internal, non-feature diagonal split it into two cavities of 77 and 78
boundary vertices, within the existing 128-point triangulator work limit.
All cavity boundaries, source vertices, and patch labels stayed fixed.

Each new triangulation was accepted locally only if minimum, mean, and
area-weighted quality did not decrease and low-quality area did not increase.
The first maximin candidate accepted six of nine cavities. A second,
explicitly bounded candidate used dynamic programming to maximize
sum(area × quality) for patch 607 while keeping its previous minimum-quality
floor. It accepted eight of nine cavities; one 607 cavity was rejected because
its low-quality area increased. Neither candidate changed vertex positions.

## Same completed-output comparison

| Measure | Official incumbent | Fixed-boundary candidate before cavity work | Weighted cavity candidate |
| --- | ---: | ---: | ---: |
| Faces | 499935 | 500527 | 500527 |
| Mean quality | .568358022 | .567916524 | .567943143 |
| P05 | .032024174 | — | .032316811 |
| Area-weighted quality | .752342857 | .752023263 | .752151085 |
| Low-quality area | 4727.793 | 4671.301 | 4670.761 |
| Largest low-quality component | 55.522 | 53.096 | 53.096 |
| Patch 607 low-quality area | .075 | .823 | .283 |

With a single global h as a **diagnostic**, short edges rise from 678731 to
679911 and the adjacent-face size-ratio P95 rises from 5.8947 to 5.9081.
These are not native local-size measurements. The P05 gain does not override
the mean, area-weighted, regional, and size concerns.

The cavity operation recovers some neighbor quality, especially in patch
607, but the final candidate still loses mean and area-weighted quality
against the official incumbent. Patch 489 low-quality area is .652 versus
.539 in the incumbent; patch 607 is .283 versus .075. The fixed
whole-mesh quality gate therefore rejects it. The narrow-strip benefit is
real as a stage result: patch 3842 low-quality area changed 59.770 to 3.210.
It cannot be counted as an accepted final improvement.

The candidate has one connected component, Euler characteristic -1, 81 open
edges, and no detected nonmanifold, winding, duplicate, or float32 zero-area
faces. The independent original-STL geometry/feature audit and the native
local-size audit were not run. The cavity step preserved coordinates and
feature records, but that alone does not certify the changed piecewise-linear
surface against the original input. No complete-flow timing was run for this
offline candidate.

## Decision and evidence

Keep the official incumbent. This experiment establishes that retessellating
connected cavities can recover part of the neighbor loss without moving
crease stations, but it does not satisfy the current final quality or native
size gate. Further work should either improve the crease resampling and
neighbor cavity together under an explicit regional tradeoff policy, or
retain the current mesh. Do not introduce this offline pass into the timed
pipeline based on these stage results.

The reproducible stage script and JSON reports are under
`results/phase27_cavity/`. From the repository root:

```powershell
python experiments/rxmesh-remesh/results/phase27_cavity/probe.py
python experiments/rxmesh-remesh/results/phase27_cavity/probe.py --weighted-607
```

They use the immutable completed-output snapshot and the phase-26 rejected
candidate; they never overwrite the official result.
