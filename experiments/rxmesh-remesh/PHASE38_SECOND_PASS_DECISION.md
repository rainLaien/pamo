# Phase 38: completed-mesh coarsening on freeform patch 4277, rejected

This bounded stage experiment used the accepted `2.stl` output and retained
every patch-4277 boundary station. The existing GPU remesher ran on one
extracted patch with the same scalar target 7.6237063 and geometry error
parameter 1.5247413. It used the completed patch as its *internal* reference,
so a separate original-STL audit was mandatory. It is neither a raw-STL
full-flow timing nor a production algorithm change.

The completed output was exported losslessly to CADPART1 (499,935 faces,
250,007 vertices; one component, Euler -1). The single patch contained
5,770 faces. Its exact boundary graph was required by `join_matching_patch`
before either candidate was inserted into the full mesh. No outside patch
faces or coordinates were edited.

| Measure | Accepted mesh | Normal second pass | Second pass with flip disabled |
| --- | ---: | ---: | ---: |
| Faces | 499,935 | 497,883 | 499,045 |
| Global mean | .5683580 | .5684666 | .5683307 |
| Global P05 | .0320242 | .0318049 | .0319201 |
| Area-weighted mean | .7523429 | .7524578 | .7523682 |
| Low-quality area | 4727.7929 | 4727.6939 | 4727.7882 |
| Largest poor component | 55.5225 | 55.5225 | 55.5225 |
| Global-h short edges (diagnostic) | 678,731 | 675,655 | 677,396 |
| New same-patch nonpositive normal samples | 0 | 1 | 0 |
| Sampled output-to-original maximum | 1.2670 | 1.2151 | 1.4145 |
| Sampled original-to-output maximum | 1.3043 | 1.3043 | 1.3043 |

All three sampled geometry maxima remain below the 1.5247 original-STL
budget. The two candidates retain topology, all source feature chains,
partition seam chains, and 81 open boundary chains under the independent
audit. These discrete checks are not a continuous Hausdorff or
self-intersection proof. The ordinary second pass adds one nonpositive
nearest-reference normal sample of area 0.000482 within patch 4277, so it
fails the orientation nonregression gate despite better local quality and
size. Disabling flips removes that regression and locally improves quality,
but global mean/P05 decline slightly, the largest bad component is
unchanged, and sampled forward geometry error rises to 1.4145. The native
per-vertex size field was not reconstructed for either manually joined full
mesh; the short-edge numbers here use the fixed global-h diagnostic and
cannot be substituted for the production native size measure.

Neither candidate is adopted. The first fails orientation nonregression.
The second offers a small size gain while consuming more of the original
geometry budget and requiring an extra remesh pass. An unrestricted second
pass could accumulate geometry error because its internal reference is the
completed output. Any future coarsening path must carry the immutable
original reference through that pass and prove a better regional tradeoff
before integration. No further patch sweep is justified by this result.

Artifacts and stage-only scripts are in `results/phase38_local_coarsen/`.
The independent audits took about 101 s each outside the candidate stage;
their cost is not part of any claimed remesh pipeline time. The accepted
full-flow `2.stl` result remains 103.09 s. No commit or push was made.
