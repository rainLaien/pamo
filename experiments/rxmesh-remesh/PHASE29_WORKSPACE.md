# Phase 29: per-task GPU scratch reuse, reverted

The official `2.stl` output and algorithm are unchanged. The complete flow
baseline remains 103.09 seconds at GPU concurrency 8; the 90-second target
and regional quality work remain open.

## Measured retry cost

The phase-24 batch has 426 packed jobs and 232 reported retries. The retrying
jobs account for 177.00 of 219.93 aggregate worker-seconds. Initial passes
use 124.50 worker-seconds, gentle retries 56.75, and quality/size retries
34.29. These stage timings can overlap with lower-level operation timers;
do not add them to the measured batch time. Retry reasons were 154 local
quality endpoint failures, 51 geometry/locked-vertex failures, and 27
size-only retry rejections. These are real difficult paths, so bypassing
them merely to improve timing would not satisfy the remesh goal.

## Controlled scratch-cache trial

The CUDA remesher already caches device blocks within each call. This trial
kept the same arena across sequential retries inside one packed job and
released it before that job's scheduler memory reservation ended. It changed
no operator, candidate order, geometry target, or acceptance guard. The
Release CLI was rebuilt; five focused CTest cases passed. The exact same
repaired `2.stl` snapshot was run with workers 8, GPU concurrency 8,
target 7.6237063, geometry budget 1.5247413, iterations 12, smoothing 3,
collapse/flip passes 8, regional final selection, and global quality policy.

| Measure | Existing implementation | Scratch reuse trial |
| --- | ---: | ---: |
| Batch wall time | 43.495 s | 46.272 s |
| Task stage | 28.238 s | 29.022 s |
| Sum of job wall times | 219.930 worker-s | 218.349 worker-s |
| Peak scheduler reservation | 1.043 GB | 1.043 GB |
| Output PLY SHA256 | `f5d0b7222e451ce5d7676bf9919bea53fb5d5ba933eee4946147cbae4a9bf996` | identical |

The mean/P05, low-quality connected area, native short/long-edge counts,
pending patch count, face count, and output bytes were identical. The sum of
job durations improved only 1.58 worker-seconds, less than 0.2 seconds at
ideal eight-way utilization; observed batch wall time was worse in this
single trial. The extra lifecycle code was removed, and the original Release
CLI was rebuilt. No complete original-STL timing is attributed to this
reused-partition stage run.

The next performance investigation must reduce repeated algorithmic work
inside retries while retaining every candidate/fallback path, or address a
different measured bottleneck. Caching CUDA allocations across calls is not
an effective path on this input.

Saved stage report and PLY:
`results/phase29_workspace/stage_gpu8/`. No commit or push was made.
