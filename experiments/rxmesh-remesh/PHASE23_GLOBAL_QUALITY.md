# Phase 23: optional whole-mesh quality acceptance

## Change

`-GlobalQualityAcceptance` makes the final batch quality flag depend on the
existing whole-mesh quality endpoint guard instead of additionally requiring
every patch to pass its own quality endpoint. The guard still rejects regression
in mean quality, P05, area-weighted mean, low-quality area, largest connected
low-quality area, invalid faces, and winding consistency. When the input has
low-quality faces, measurable quality progress is required. Geometry, topology,
seam, and size checks remain in their existing stages. The strict patch rule
remains the default for `run_raw_partition.ps1`; `run_remesh.ps1` selects this
new policy for its normal-quality command line and reports residual patches.

The existing residual strip-repair pass still runs if unresolved patches exist.
It stops when no additional supported regions can be generated. The optional
flag cannot be combined with legacy coverage acceptance.

This changes the acceptance label, not candidate generation or mesh quality.
Do not interpret `quality_accepted=true` as proof that every region or size
target passes. Read pending/unresolved patches and short-edge counts alongside
the global fields.

## Same-parameter complete runs from original STL

| Input / mode | Full pipeline | Output SHA256 | Faces | Mean / P05 | Low-quality area / largest component | Long / short edges | Pending / unresolved patches | Quality accepted |
| --- | ---: | --- | ---: | --- | --- | --- | --- | --- |
| `2.stl`, strict baseline | 107.39 s | `f5d0b722...996` | 499935 | .568358 / .032024 | 4727.793 / 55.522 | 0 / 676458 | 2009 / 735 | no |
| `2.stl`, global policy | 113.32 s | `f5d0b722...996` | 499935 | .568358 / .032024 | 4727.793 / 55.522 | 0 / 676458 | 2009 / 735 | yes |
| `3.stl` (same STL as 浇道), strict script baseline | 8.41 s | `1ff599be...03` | 57184 | .7410 / .2289 | 6.026 / .435 | 0 / 34509 | 57 / 4 | no |
| `3.stl`, global-policy script | 7.30 s | `1ff599be...03` | 57184 | .7410 / .2289 | 6.026 / .435 | 0 / 34509 | 57 / 4 | yes |

Full SHA256 hashes, logs and full reports are in
`results/phase16_polygon/full_workers8`, `results/phase23_global_quality/full_2`,
`results/phase22_script/smoke`, and
`results/phase23_global_quality/jiaodao_script_final`. Each pair produced an
identical PLY, so neither apparent acceptance improvement nor runtime
difference is an algorithm-quality or speed gain. These are single timings.
`2.stl` still exceeds the 90 s target.

The previous phase-19 audit of the same exact `2.stl` PLY found one connected
topological component, Euler characteristic -1, no detected nonmanifold,
winding, duplicate-face, or zero-area issues, and sampled original-to-output
distance within its 1.5247413 geometry budget. Exact PLY identity carries
those observations over, but sampled distances are not a continuous geometric
certificate. For `3.stl`, phase-21 audit of the same exact PLY covered all
10692 source feature chains by exact dyadic owner-matched paths; it found no
nonpositive face normals. These audits have their own limits.

## Reproduce

```powershell
cmake --build experiments/rxmesh-remesh/build_rx --config Release --target cad_raw_partition_cli test_raw_batch test_region_quality --parallel 8
ctest --test-dir experiments/rxmesh-remesh/build_rx -C Release -R '^(constraint_audit|region_quality|region_candidate_selection|raw_batch_safety|raw_projection_safety)$' --output-on-failure
& experiments/rxmesh-remesh/run_raw_partition.ps1 -InputMesh examples/2.stl -OutputDirectory experiments/rxmesh-remesh/results/phase23_global_quality/reproduce_2 -ModelSeeds 250 -Workers 8 -GpuConcurrency 4 -Iterations 12 -SmoothPasses 3 -CollapsePasses 8 -FlipPasses 8 -TargetLength 7.6237063 -MaxError 1.5247413 -SelectFinalRegions -GlobalQualityAcceptance
& experiments/rxmesh-remesh/run_remesh.ps1 -InputMesh examples/3.stl -OutputDirectory experiments/rxmesh-remesh/results/phase23_global_quality/reproduce_3
```

Use new output directories for reruns. The `2.stl` command includes partition,
preprocessing, batch remeshing, and output. The `run_remesh.ps1` summary also
times its native size audit separately.
