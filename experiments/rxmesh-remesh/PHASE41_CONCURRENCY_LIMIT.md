# Phase 41: GPU concurrency ceiling trial, rejected

The phase-24 accepted `examples/2.stl` full flow remains the incumbent. This
trial used its repaired CADPART snapshot with identical target length
7.6237063, original geometry budget 1.5247413, patch packing 16, iteration
12, smoothing 3, collapse 8, flip 8, and final-region/global-quality policy.
It changed only worker and GPU concurrency from 8 to 12. The existing
validation cap of 8 was temporarily raised to 12 for the experiment.

| Measure | 8 (phase-24 stage) | 12 (this stage) |
| --- | ---: | ---: |
| Batch seconds | 43.495 | 46.783 |
| Task seconds | 28.238 | 29.083 |
| Assembly seconds | 11.634 | 13.683 |
| Peak active jobs | 8 | 12 |
| Peak scheduler reservation | 1.043 GB | 1.502 GB |
| Output PLY SHA256 | `f5d0b7222e451ce5d7676bf9919bea53fb5d5ba933eee4946147cbae4a9bf996` | identical |

The 12-way output has 499935 faces, mean quality 0.568358, P05 0.032024,
low-quality area 4727.793, largest low-quality component area 55.522,
676458 short edges, zero long edges, 2009 pending patches, and 735 unresolved
patches: all unchanged. The run exited successfully. Its invocation wall time
was 49.385 seconds; the table uses the program's batch timer for comparison.
This is a single stage comparison, not a new complete original-STL timing.

The larger concurrency worsened both task and assembly time. The cap and
Release executable were restored to their pre-trial state; the executable
SHA256 is `e26e46a73aba16b61de40ce17e495caa357d3ee70322ba334319536aee2d58b3`.
Increasing concurrent jobs is not an evidenced route to the 90-second goal
on this input. The accepted mesh and algorithm were not changed. No commit or
push was made.

Stage output and log: `results/phase41_concurrency12/`.
