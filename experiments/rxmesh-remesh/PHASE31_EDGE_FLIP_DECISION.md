# Phase 31: protected vertices are not the only obstacle

The accepted `2.stl` output and production remesher are unchanged. This was
one bounded completed-output experiment; it was not an original-STL full-flow
run. The latest complete-flow evidence remains 103.086 s at GPU concurrency 8.

## Eligibility measured on the accepted output

The largest 12 low-quality connected components each span a thin geometric
strip. Their long bounding-box dimension ranges from about 113 to 544 units,
while the other dimensions are generally below 0.21. Every vertex in those
components touches at least one protected edge. Yet most edges *inside* their
low-quality face sets are ordinary: for the largest component, 198 of 222
internal edges have no protection flag; for the second, 149 of 150. The
existing flat-diagonal prototype excludes these edges because it checks
whether their endpoints touch any protected edge. This is overconservative
for a fixed-vertex diagonal flip, but it does not imply such a flip can
improve the poor strip while satisfying geometry and size guards.

## Bounded edge-level trial

The trial used the phase-19 edge-provenance sidecar, whose PLY SHA256 exactly
matches the phase-24 accepted output. It prohibited flipping a protected
edge or crossing patch IDs, kept every vertex fixed, required an exact
axis-aligned planar four-vertex witness, preserved face orientation, and
required pair minimum/mean/area-weighted quality and low-quality area not to
regress. A second variant used the saved native target at each endpoint to
ensure a new diagonal was not long and did not add or worsen a short edge.

| Measure | Accepted output | Native-size-guarded trial |
| --- | ---: | ---: |
| Accepted flips, at most two passes | — | 478 |
| Changed faces / patches | — | 939 / 178 |
| Mean quality | 0.568358022 | 0.568452507 |
| P05 | 0.032024174 | 0.032024174 |
| Area-weighted quality | 0.752342857 | 0.753522059 |
| Low-quality area | 4727.792922 | 4727.792922 |
| Largest poor component | 55.522459 | 55.522459 |
| Native short / long edges | 676458 / 0 | 676458 / 0 |

Exactly zero previously low-quality faces changed. The 478 flips improve
healthy planar areas but do not resolve any of the largest poor components.
The fixed global-h diagnostic short count changes by two because it does not
use the saved native targets; the native count is unchanged. The alternate
legacy size guard produced 433 flips, likewise without changing low-quality
area. These are stage metrics. No independent original-STL audit or full-flow
timing is claimed for the rejected trial.

## Decision

Do not integrate the edge-level flip relaxation. Its approximately seven
seconds of offline candidate generation would add work while leaving the
regional quality objective untouched. The production prototype was restored
after the experiment; a self-contained copy used solely to reproduce the
stage trial remains in ignored results. The evidence narrows the next
implementation target: the poor strips need coordinated boundary-station
and adjacent-face sizing, with an external seam contract, rather than a
different diagonal chosen from the existing fixed vertices.

Evidence and reproduction:

```powershell
python experiments/rxmesh-remesh/results/phase31_eligibility.py
python experiments/rxmesh-remesh/results/phase31_edgeflip.py --native-size-guard
```

The corresponding JSON reports and stage PLY are under `results/`. No commit
or push was made.
