# Phase 40: source-reference connected-region recovery

This phase adds `tools/try_original_region_recovery.py`, an explicit,
reversible candidate path. It extracts at most two requested source patches
from the **original repaired CADPART1 snapshot**, runs the existing native
remesher with unchanged target/error/operator settings, and accepts at most
one changed patch under an exact boundary graph contract. The incumbent PLY
is never overwritten. The candidate and a reviewable decision report are
saved separately. No partitioner, CUDA edit operator, seam scheduler, or
narrow-strip reconstruction code was changed.

## Why patches 4501 and 4510

The largest low-quality connected region entirely outside producer-labeled
Fillet patches has 12.503 area and 111 faces across patches 4501 and 4510.
Both have no producer support relation. Patch 4501 is Freeform; 4510 is
Planar. This is a diagnostic selection, not evidence that every other
Fillet or Freeform interface can be moved. The candidate retains all
interface stations. Only patch 4501 changes; patch 4510 is an unchanged
member of the connected transaction.

The isolated native run exits 4 under its strict per-patch quality endpoint:
patch 4501's arithmetic mean falls, although its P05, area-weighted mean,
size distribution and low-quality area improve. The tool keeps that native
result provisional and applies an explicit joint policy to the assembled
full mesh. It does not relabel the native exit as success.

## Same-input quality comparison

Fixed target 7.6237063, original geometry budget 1.5247413, low-quality
diagnostic cutoff 0.026690566912293434, iterations 12, smoothing 3,
collapse 8, flip 8, strict flip quality on:

| Metric | Incumbent | Candidate |
| --- | ---: | ---: |
| Faces | 499,935 | 499,079 |
| Mean quality | .56835802 | .56831633 |
| P05 | .03202417 | .03219494 |
| Area-weighted mean | .75234286 | .75243491 |
| Low-quality area | 4727.79292 | 4712.29332 |
| Largest connected low-quality area | 55.52246 | 55.52246 |
| Native short edges | 676,458 | 675,151 |
| Native long edges | 0 | 0 |
| Nearest-source nonpositive normal samples | 456 | 454 |
| Output-to-original sampled maximum | 1.26703 | 1.10024 |
| Original-to-output sampled maximum | 1.30429 | 1.30429 |

The connected transaction's defect severity falls from 6.490 to 0.672
without measured neighbor loss, area-weighted quality loss, severe-area
growth, largest-component growth or face growth. An explicit trial policy
allows at most 0.00005 absolute global mean loss; the measured requirement
is 0.00004169. This is a limited tradeoff decision, not a universal quality
threshold. The independent full-mesh audit found no missing source feature,
partition seam, or open-boundary chains, no new topology defects, and the
same one component and Euler characteristic -1. The sampled geometry
checks are not continuous Hausdorff or self-intersection proofs. The
candidate's native size field was reconstructed from vertex-ID lineage;
all shared-boundary targets match the incumbent, and the joined coordinates
and faces were checked against that lineage.

## Full original-STL timing and decision

The fresh run starts from `examples/2.stl` and includes partitioning,
packaging, narrow-strip preprocessing, GPU remesh, native field export,
second repair scan and output. It takes **109.013 s**. The source snapshot
and base PLY SHA256 hashes match phase 24 exactly. Regional candidate
generation, join and all tool checks take another **35.905 s**; the measured
sum is **144.918 s**. The recovered PLY is byte-identical to the earlier
stage candidate, so the existing independent 104.17 s whole-mesh audit
applies to this full-flow output. That offline audit is not included in
144.918 s. The phase-24 103.086 s baseline did not export audit fields;
the fair full-flow timing comparison for this candidate is 109.013 s versus
144.918 s within the same audit-enabled run. Do not attribute the
109.013-versus-103.086 difference solely to the algorithm.

This is a valid **optional, bounded regional candidate**, but it is not
enabled by default. Its 15.50 area reduction is useful for the chosen
ordinary region; the largest global bad region remains a constrained
Fillet interface, while the full flow exceeds the 90 s target by 54.92 s.
Automatically repeating this expensive path across many regions would not
meet the user's performance goal. A production adoption needs a faster
in-process region transaction and an evidence-based trigger; neither is
claimed here. The incumbent remains the normal output.

## Reproduce

```powershell
& experiments/rxmesh-remesh/run_raw_partition.ps1 -InputMesh examples/2.stl -OutputDirectory experiments/rxmesh-remesh/results/phase40_nonfillet/full_raw_reproduce -ModelSeeds 250 -Workers 8 -GpuConcurrency 8 -PatchesPerTask 16 -Iterations 12 -SmoothPasses 3 -CollapsePasses 8 -FlipPasses 8 -TargetLength 7.6237063 -MaxError 1.5247413 -LowQualityThreshold 0.026690566912293434 -SelectFinalRegions -GlobalQualityAcceptance -AuditFields

python experiments/rxmesh-remesh/tools/try_original_region_recovery.py --snapshot experiments/rxmesh-remesh/results/phase40_nonfillet/full_raw_reproduce/strip_repair/pass_01/input.cadpart --incumbent experiments/rxmesh-remesh/results/phase40_nonfillet/full_raw_reproduce/remeshed.ply --incumbent-fields-mesh experiments/rxmesh-remesh/results/phase40_nonfillet/full_raw_reproduce/strip_repair/pass_01/remeshed.ply --original-stl examples/2.stl --remesher experiments/rxmesh-remesh/build_rx/Release/cad_raw_partition_cli.exe --output experiments/rxmesh-remesh/results/phase40_nonfillet/full_raw_reproduce/recovered_4501.ply --patches 4501 4510 --target 7.6237063 --max-error 1.5247413 --threshold 0.026690566912293434 --quality-policy experiments/rxmesh-remesh/results/phase40_nonfillet/quality_policy.json --iterations 12 --smooth-passes 3 --collapse-passes 8 --flip-passes 8
```

The policy and all candidate/audit artifacts are retained in
`results/phase40_nonfillet/`. No commit or push was made.
