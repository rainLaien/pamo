# Phase 28: bounded partition prefetch experiment, reverted

The accepted `2.stl` full flow remains the phase-24 result: 103.09 seconds
with GPU concurrency 8 and the same output mesh. This phase did not change
the official remesh implementation. It tested one partition-only performance
hypothesis and restored the original source and executable afterward.

## Hypothesis and controlled test

`ModelFirstPartitioner.cpp` had an analytic-seed lookahead of 8. The
partition log showed many small CUDA calls. The experiment changed that
single constant to 16, rebuilt `cad_mesh_segment.exe`, and reran partitioning
from the original `examples/2.stl` with `--remesh-handoff
--stop-after-partition --max-model-seeds 250`. The source STL and all model
fit and geometry settings were unchanged. Both `patch_result.ply` and
`patch_report.json` had **identical SHA256 hashes** to the phase-24 partition.
This was a partition stage trial, not a full remesh timing.

| Measure | Lookahead 8 | Lookahead 16 |
| --- | ---: | ---: |
| Residual-neighborhood prefetch flushes | 157 | 85 |
| Residual-neighborhood queued seeds | 19326 | 19326 |
| Residual-neighborhood direct misses | 3139 | 3139 |
| All analytic CUDA batches | 11995 | 11908 |
| CUDA initialization | .638 s | 4.426 s |
| CUDA transfer/kernel/wait | 21.119 s | 22.055 s |
| Model-first partition | 35.718 s | 40.110 s |
| Segmenter wall time | 41.599 s | 45.756 s |

Single-run CUDA initialization variance affects the timing comparison, so
the data do not prove that lookahead 16 is intrinsically slower. They do
show that it removes only 87 of 11995 GPU launches and does not reduce the
3139 direct misses in the residual-neighborhood stage. This mechanism is
unlikely to supply the roughly 13 seconds needed for the 90-second goal.
The constant was restored to 8 and the Release segmenter was rebuilt.

The current GPU-8 batch report shows 426 jobs and 232 retries. Sum of job
wall times is 219.93 worker-seconds; at 8 active workers, the measured task
stage is 28.24 seconds, about 97% of ideal aggregate utilization. Adding
more scheduling concurrency cannot materially reduce that stage without
reducing the repeated work inside jobs. Its largest measured parts are
initial attempts (124.50 worker-seconds), gentle retries (56.75), and
quality-split attempts (34.29); these overlap with lower-level operation
timers and must not be summed again. Any retry optimization must preserve
the candidates and difficult-region fallback, and be compared at the same
target size and original geometry budget.

Saved partition-only output and log:
`results/phase28_prefetch16/partition/` and `partition.log`. These are
diagnostics, not a published remesh output. No commit or push was made.
