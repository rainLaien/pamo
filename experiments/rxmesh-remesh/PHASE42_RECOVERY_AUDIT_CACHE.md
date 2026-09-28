# Phase 42: reuse immutable regional audit data

The optional original-source regional recovery trial spent 35.905 seconds
after the original-STL full flow. A profile of the same 4501/4510 trial showed
15.513 seconds in sampled original-STL geometry, including building the same
original-mesh triangle tree twice; the full source topology audit was also
repeated before chain checking.

The change reuses the immutable original-mesh nearest-surface query for both
forward sample sets and passes the already computed source edge keys/counts
to chain checking. Sampling points, tie handling, local remesh, size field,
quality policy, and acceptance checks are unchanged.

| Same-input candidate | Reported tool seconds | PLY SHA256 | Report differences |
| --- | ---: | --- | --- |
| Before | 35.905 | `5bff6bafe866c31be5c67a77f31cf91b536001c7e17554bf05586b4ad21b6c6e` | Reference |
| Reuse trial 1 | 33.146 | Same | Path and seconds only |
| Reuse trial 2 | 36.211 | Same | Path and seconds only |

The two runs do not establish a reliable wall-time improvement. The change
removes redundant index/audit construction while retaining exactly the same
candidate and decision on this input. `test_phase1_audit.py` passed all 10
cases, and the modified Python modules compiled. This remains an optional
candidate tool; no change to the default remesh output or full original-STL
timing is claimed. The candidate still costs roughly 33–36 seconds in these
runs and does not meet the 90-second full-flow target when appended to the
current pipeline.

Profile and outputs: `results/phase41_concurrency12/recovery_profile.pstats`,
`profiled_4501.ply`, `cached_4501.ply`, and `cached2_4501.ply`. No commit or
push was made.
