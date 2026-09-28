# Phase 30: original geometry and local sizing audit of the cavity candidate

The accepted `2.stl` result remains the phase-24 output. No production
algorithm, parameter default, or output mesh changed. The complete original
STL runtime evidence remains one 103.086 s run at GPU concurrency 8. The
phase-27 candidate remains a completed-output stage experiment, so its audits
are not a full-flow timing result.

## Original-input geometry and topology

`audit_phase1.py` compared the accepted mesh and the weighted cavity candidate
against the same raw `examples/2.stl`, canonical completed-output snapshot,
quality cutoff, and 1.524741292 original-input error budget. Both output
meshes retained all 100586 feature chains, 99806 partition-interface chains,
and 81 open-boundary chains from this snapshot, with zero missing anchors or
chains. Both have one component, Euler characteristic -1, 81 open edges, and
no detected nonmanifold, inconsistent-winding, duplicate, or float32 zero-area
faces.

| Original-STL sampled distance | Accepted mesh | Cavity candidate |
| --- | ---: | ---: |
| Output to original max | 1.267034 | 1.162998 |
| Original to output max | 1.304286 | 1.304286 |

These are deterministic samples, not a continuous Hausdorff certificate.
The nearest-source-triangle centroid normal check found the same 106
nonpositive samples in each output, in unchanged patches 885 and 6725. Their
total area is 0.003154. This check does not prove all faces are correctly
oriented or that there are no self-intersections; it does show that this
candidate did not add samples failing that check.

## Prospective native-size field

The candidate PLY retains all 250007 accepted-output vertices in exactly the
same order and appends 296. `audit_candidate_sizing.py` binds the saved
incumbent native vertex targets to that exact coordinate identity. It
reconstructs the explicit local h=5.0, gradation=0.5 field from the immutable
snapshot's protected edges touching patch 3842, then evaluates every candidate
vertex on every incident patch. It uses the same float32 average endpoint
target and split/collapse ratio as `RegionQuality`. As a self-check, the
saved incumbent targets reproduce its reported 676458 short and zero long
edges exactly. The reconstructed candidate field is stage evidence, not a
production `--audit-fields` export.

| Native local-size measure | Accepted mesh | Cavity candidate |
| --- | ---: | ---: |
| Short edges | 676458 | 677299 (+841) |
| Long edges | 0 | 67 |
| Maximum edge/target ratio | — | 1.995898 |

The 67 candidate long edges touch faces in patches 489, 607, and 3843 (face
incidence counts 17, 72, and 22; an edge may touch two faces). This is a
direct consequence of applying the smaller target to the shared boundary and
its neighbor support without a subsequent size-conforming remesh of those
neighbors. It is not a reason to ignore or weaken the local target.

## Decision

Reject the candidate. It improves low-quality area 4727.793 to 4670.761 and
the largest poor connected component 55.522 to 53.096, and the sampled
geometry/chain checks pass. However, mean and area-weighted quality decline,
neighbor patches 489/607 retain new poor regions, the prospective native
field has 67 long edges and 841 more short edges, and no complete-flow run
exists. The candidate fails the size and quality endpoint independently of
the small mean-quality regression. Do not add it to the production pipeline.

Further work on this strip needs a connected region transaction that includes
the shared curve stations, adjacent faces, and their size transition, with a
native field export and the same original-geometry checks before adoption.
This audit closes the bounded h=5 plus fixed-boundary cavity direction; no
further scalar h sweeps are justified by these data.

## Reproduce the audits

From the repository root, use the saved phase-24 and phase-27 stage artifacts:

```powershell
python experiments/rxmesh-remesh/tools/audit_phase1.py --source examples/2.stl --snapshot experiments/rxmesh-remesh/results/phase16_polygon/canonical_incumbent.cadpart --repair-report experiments/rxmesh-remesh/results/phase24_concurrency/full_gpu8/strip_repair/pass_01/repair.json --baseline experiments/rxmesh-remesh/results/phase24_concurrency/full_gpu8/remeshed.ply --candidate experiments/rxmesh-remesh/results/phase27_cavity/candidate_weighted.ply --output experiments/rxmesh-remesh/results/phase27_cavity/original_geometry_audit.json --threshold 0.026690566912293434 --target 7.623706340789795 --max-error 1.524741292
python experiments/rxmesh-remesh/results/phase27_cavity/audit_candidate_sizing.py
```

Saved evidence: `results/phase27_cavity/original_geometry_audit.json` and
`candidate_sizing.json`. The original-input audit took 94.584 s outside the
timed full pipeline. No commit or push was made.
