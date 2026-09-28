# Phase 35: original-STL partition GPU profile

The accepted `2.stl` remesh remains the phase-24 full-flow result: 103.09 s.
This phase changed no algorithm, output, or parameter. It profiled partitioning
from the original STL with `--max-model-seeds 250` to choose a bounded next
performance target. It is a partition-stage measurement, not a full-flow time.

| Measurement | Standard mixed launch with event timing | Per-type diagnostic launches |
| --- | ---: | ---: |
| Segmenter wall time | 42.317 s | 43.150 s |
| Model-first partition | 36.398 s | 37.333 s |
| GPU transfer + kernel + wait | 21.329 s | 22.633 s |
| GPU kernel | 19.776 s | 21.061 s |
| Host-to-device transfer | 0.472 s | 0.475 s |
| `patch_result.ply` and `patch_report.json` | Both byte-identical to phase 24 | Both byte-identical to phase 24 |

Per-type diagnostic kernel time: torus 11.305 s (53.9%), cone 5.487 s
(26.2%), cylinder 3.377 s (16.1%), sphere 0.807 s (3.8%). The per-type
diagnostic changes launch shape; its numbers identify the dominant model but
are not speed comparisons against the mixed-launch path.

The kernel accounts for about 93% of measured transfer/kernel/wait time in
the normal mixed launch. Thus changing prefetch batch size or transfer volume
alone cannot plausibly recover the roughly 13 s between the current full-flow
baseline and the 90 s target. A useful next partition trial must reduce torus
and cone kernel computation while preserving fitted models and all feature,
topology, and geometry checks. Do not reduce the analytic seed budget to claim
the same geometric constraint at lower cost.

Reproduce the diagnostic:

```powershell
$env:CADMESH_CUDA_PROFILE='1'
& cad_mesh/win/Release/cad_mesh_segment.exe examples/2.stl experiments/rxmesh-remesh/results/phase35_partition_profile --remesh-handoff --stop-after-partition --max-model-seeds 250
Remove-Item Env:CADMESH_CUDA_PROFILE
```

Logs: `results/phase35_partition_profile/partition.log` and
`results/phase35_partition_types/partition.log`. No commit or push was made.
