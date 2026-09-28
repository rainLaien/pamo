# Phase 24: bounded GPU concurrency change

The batch scheduler used only four active jobs with `-Workers 8
-GpuConcurrency 4`. On `2.stl` the reported peak reservation was 562 MB
against a 7.07 GB scheduler budget. A single stage experiment using the
identical repaired snapshot, target size, geometry error, iteration settings,
and quality policy raised only `-GpuConcurrency` to 8. Peak active jobs became
8, batch time changed from 54.18 to 46.91 seconds, and output PLY SHA256 was
identical. This stage run reused a partition and is **not** a complete-flow
timing.

A complete run from the original `examples/2.stl` then changed only that
parameter. It includes partitioning, narrow-strip preprocessing, batch
remeshing, residual pass, and output:

| GPU concurrency | Full flow | Partition/packaging | Batch | PLY SHA256 |
| ---: | ---: | ---: | ---: | --- |
| 4 | 113.32 s | 43.70 s | 54.18 s | `f5d0b7222e451ce5d7676bf9919bea53fb5d5ba933eee4946147cbae4a9bf996` |
| 8 | 103.09 s | 43.98 s | 43.49 s | `f5d0b7222e451ce5d7676bf9919bea53fb5d5ba933eee4946147cbae4a9bf996` |

The identical output has 499935 faces, mean quality 0.568358, P05 0.032024,
low-quality area 4727.793, largest low-quality component area 55.522,
zero overlong and 676458 short edges, 2009 pending and 735 unresolved patches.
Quality and size have **not** improved. The 90-second goal is still unmet.
These are single timings, not a performance guarantee. The only changed input
parameter is GPU concurrency; no algorithm change is credited with the speed
gain.

For `examples/3.stl` (same content as the supplied 浇道 STL), the complete
script run was 7.30 seconds at 4 versus 6.46 seconds at 8; PLY SHA256 was
identical (`1ff599be41223e04cd569187c38b8854eabaa9bda5f36cea6ff2a868d917ab03`).
The convenience script now defaults to 8 and exposes `-GpuConcurrency` for
changing it. The lower-level pipeline keeps its previous default of 4.

## Reproduce

```powershell
& experiments/rxmesh-remesh/run_raw_partition.ps1 -InputMesh examples/2.stl -OutputDirectory experiments/rxmesh-remesh/results/phase24_concurrency/reproduce_2 -ModelSeeds 250 -Workers 8 -GpuConcurrency 8 -Iterations 12 -SmoothPasses 3 -CollapsePasses 8 -FlipPasses 8 -TargetLength 7.6237063 -MaxError 1.5247413 -SelectFinalRegions -GlobalQualityAcceptance
& experiments/rxmesh-remesh/run_remesh.ps1 -InputMesh examples/3.stl -GpuConcurrency 8 -OutputDirectory experiments/rxmesh-remesh/results/phase24_concurrency/reproduce_3
```

Each run needs a new output directory. Recorded results:
`results/phase23_global_quality/full_2`,
`results/phase24_concurrency/full_gpu8`,
`results/phase23_global_quality/jiaodao_script_final`, and
`results/phase24_concurrency/jiaodao_gpu8`.
