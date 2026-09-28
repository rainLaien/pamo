# Phase 25: stop scalar-size tuning on the largest thin-strip defect

The accepted `2.stl` output is unchanged from phase 24. Its largest connected
low-quality region has area 55.52246 and spans patches 3842/3843. Its bounding
box is 351.073 by 0.201 by 0.194 model units. In patch 3842, low-quality
triangles have a median longest side of 7.2432 and shortest side of 0.09675.
This is direct evidence of a narrow geometric strip at the current target
length 7.6237063. The box alone does not certify a CAD feature or prove that
the low quality is unavoidable.

## Single bounded candidate

To check whether a modest local isotropic size change avoids the neighbor
loss of the earlier 0.95 experiment, set only patch 3842 to target 5.0 in the
already-completed incumbent snapshot. The repair tool expanded to 3843 and
rebuilt both, with affected neighbors 489/607/3841/3844. Source, global
target, geometry budget, and quality cutoff stayed fixed. This is a
completed-output stage experiment; it is not a full original-STL run.

| Measure | Incumbent | Target 5.0 candidate |
| --- | ---: | ---: |
| Faces | 499935 | 500601 |
| Mean quality | .568358022 | .567883748 |
| P05 | .032024174 | .032339438 |
| Area-weighted quality | .752342857 | .752020496 |
| Low-quality area | 4727.793 | 4668.938 |
| Largest low-quality component | 55.522 | 53.096 |
| Short edges using global h (diagnostic only) | 678731 | 680037 |
| Long edges using global h | 0 | 0 |

Patch 3842 low-quality area improved 59.770 to 3.210, and 3843 improved
6.266 to 2.987. Neighbor 489 worsened 0.539 to 0.775 and 607 worsened
0.075 to 0.823; both also lost area-weighted quality. The candidate retained
one connected component, Euler characteristic -1, 81 open edges and no
detected nonmanifold, inconsistent, duplicate, or float32 zero-area faces.

**Decision: reject.** The preselected whole-mesh non-regression rule rejects
the mean and area-weighted quality losses; local neighbor defects also grew.
The candidate was not promoted to the complete output. Because it failed the
quality gate, no independent original-STL feature, geometry, or native size
audit and no full-flow timing were run for it. Global-h short-edge counts are
not native sizing results. Reducing only this local isotropic size is not an
adequate fix under the current boundary treatment. Further scalar-size sweeps
on this interface are stopped.

The next implementation should address the shared boundary geometry and
neighbor retessellation as one region transaction, with the same feature,
original-error, native-size and global-quality guards. It must keep the
accepted incumbent available for rollback. This is a design direction, not a
claim that such a transaction will pass.

## Reproduce the stage candidate

Use `results/phase16_polygon/canonical_incumbent.cadpart` as the immutable
completed-output snapshot, write a SHA-bound target JSON with patch 3842 at
5.0, and run:

```powershell
python experiments/rxmesh-remesh/tools/repair_narrow_strips.py experiments/rxmesh-remesh/results/phase16_polygon/canonical_incumbent.cadpart experiments/rxmesh-remesh/results/phase25_target5/rebuilt.cadpart --report experiments/rxmesh-remesh/results/phase25_target5/repair.json --target 7.623706340789795 --max-error 1.524741292 --patch-targets experiments/rxmesh-remesh/results/phase25_target5/targets.json --graded-neighbors --only-requested-regions
```

The ignored result directory contains `targets.json`, `repair.json`,
`rebuilt.ply`, and `quality.json`; the latter is a stage comparison against
`results/phase24_concurrency/full_gpu8/remeshed.ply`. Do not count this
candidate as a production quality or timing improvement.
