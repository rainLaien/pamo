# CAD Adaptive Remesher (RXMesh experiment)

Isolated prototype. It does **not** replace `cad_mesh/rxmesh/` or
`NativeRemesher`. PAMO keeps owning recognition, fitting, and patch graphs.
RXMesh only executes topology.

```text
CadAdaptiveRemesher
        │
        ├── CadSemanticModel     CPU  (PAMO types, mirrored)
        ├── RemeshField          CPU→GPU SoA
        ├── RemeshPolicy         CPU tests + device copy
        ├── GeometryProjector    plane / cylinder first
        └── IRemeshBackend
                  │
          ┌───────┴────────┐
          │                │
   CpuRemeshBackend   RxMeshBackend
```

Rule: **RXMesh does not know cylinders, fillets, or CAD features. It knows
topology.** Policy and projection stay above the backend.

Completed: [`RXREMESH-001.md`](RXREMESH-001.md) (001.0–001.5) and
[`RXREMESH-002.md`](RXREMESH-002.md) (002.0–002.2, CPU/GPU dirty 2-ring and GPU candidate counters).

A raw STL has no `PatchId` / `EdgeSharp`. The raw CLI path remains an unlabeled
topology experiment. For CAD shape preservation, use the PAMO handoff below.

## Raw STL quality baseline (`--gpu`)

```powershell
.\experiments\rxmesh-remesh\build_rx\Release\cad_adaptive_cli.exe `
  .\examples\Unnamed-Body.stl out.obj --gpu --iters 20
```

An unlabelled single-patch mesh now selects `gpu-raw-cuda`, a dedicated CUDA
operator implementation. It does not run the CPU remesher or VCGLib internally.
The RXMesh cavity backend continues to handle the existing labelled/analytic
path. The raw CUDA path uses GPU triangle split templates, short/small-area
collapse, a relaxed low-valence collapse pass, valence/quality flips, and
projected smoothing against an immutable copy of the input triangles. Creases
are classified once and propagated. Qualifying crease segments can coarsen;
this phase does not promise preservation of every original feature vertex.

CPU code constructs adjacency and compacts GPU output between topology passes.
Reference queries use an immutable stackless triangle BVH. Uniform and optional
feature/curvature sizing support the tested small reference meshes; large-mesh
scaling is not yet established. It does not exactly reproduce VCGLib's sequential scheduling, smoothing
step count or dedicated fold-relaxation pass. `--cpu` remains the separate
experimental CPU implementation.

For a real VCGLib comparison, configure with
`-DCAD_ADAPTIVE_VCGLIB_REFERENCE=ON -DVCGLIB_ROOT=D:/openSourceInstall/vcglib`, then
build `vcglib_reference`. Its arguments are
`INPUT OUTPUT [target_length] [iterations] [feature_angle] [max_error]`; defaults
match the raw CLI's length, iterations, feature angle and geometry tolerance.
The CLI's raw geometry report samples the original input, not the output itself.
`tools/compare_raw_meshes.py` independently audits exported meshes, including
sampled distances in both directions (requires NumPy, SciPy and trimesh).

Configure `-DBUILD_TESTING=ON` to build the raw-reference quality regression and
existing RXMesh isotropic, constraints and analytic-projection regressions.

## Raw feature and curvature refinement

```powershell
.\experiments\rxmesh-remesh\build_rx\Release\cad_adaptive_cli.exe `
  .\examples\Unnamed-Body.stl refined.obj --gpu --feature-refine
```

This opt-in mode refines curved regions, rather than all feature edges. Sharp
plane/plane intersections remain geometric constraints without automatically
receiving smaller elements. Coplanar triangles are grouped first; a chain of
normal changes through adjacent groups provides curvature evidence. An isolated
shallow junction between two planes is excluded as well as sharp dihedrals.
This is a discrete heuristic, not a complete CAD surface classifier.

`--feature-size H` sets the lower bound on curvature-driven size (default 0.25
of the regular length). It no longer sets a size on all creases. The curvature
size combines a sagitta bound with a normal-rotation bound controlled by
`--normal-degrees` (default 10). Lowering H alone does not force refinement when
curvature already permits a larger size. `--feature-band B` controls transition
width onto adjacent planes (default 0.75 of the regular length, previously 2).
Both options imply `--feature-refine`. H must be positive and smaller than the
regular length; B must be positive. Scope is raw single Unknown patch `--gpu`.

Immutable source curvature edges seed the field, evaluated on the GPU after
edits. Split/collapse use local targets; proposed edits and final seven-point
face samples use `min(max-error, 0.08 * local-target)`. The local sizing audit
reports local edge-band compliance. The other edge audit explicitly reports
only the global reference length. Shared curve/plane boundaries remain conforming
and require a narrow transition on the plane. The field uses Euclidean distance;
very close sheets, noisy input, and undersampled curves need further validation.
It preserves the STL surface, not an unprovided exact CAD surface.

For `Unnamed-Body.stl`, 20 cycles now produce 11,092 faces instead of the previous
feature-distance mode's 21,688. In fixed inspection regions, average edge length
changes from 0.87 to 0.59 on the round and from 0.46 to 1.82 on distant plane/plane
intersections (regular target 1.80). Mean triangle quality is 0.97358; independently
sampled bidirectional error is 0.02839. Regression tests cover a square prism,
a shallow two-plane junction, a smooth cylinder and the user's STL.

## Raw GPU performance

The raw CUDA implementation now reuses per-call device scratch storage, downloads
candidate counts instead of complete candidate records, and updates flip faces
without vertex compaction. Adjacency uses contiguous open-addressed lookup arrays
while preserving first-seen edge IDs. Vertex-star safety checks visit each face
once, and smoothing reuses its trial projection. Source projection uses a
stackless BVH with original-triangle tie ordering. The CLI reuses the backend's
final immutable-source GPU audit and metrics rather than repeating them on CPU;
independent export auditing remains available in `compare_raw_meshes.py`.

On RTX 3060 / `Unnamed-Body.stl`, `--gpu --feature-refine`, 20 iterations, medians
of three fresh CLI processes before/after optimization were:

| Timing | Before | After | Speedup |
|---|---:|---:|---:|
| Whole command, including startup and export | 8.965 s | 6.078 s | 1.47x |
| Remesh backend | 7.758 s | 4.639 s | 1.67x |
| Flip stage | 1.711 s | 0.568 s | 3.01x |

All six exported OBJ SHA256 values match. Iteration count, sizing and quality
thresholds are unchanged. These numbers describe one model on one machine, not a
general speedup guarantee. CPU adjacency reconstruction and per-pass topology
transfers remain; moving those operations and compaction onto persistent GPU
storage is the next larger architectural opportunity.

## Raw CUDA operation safety

Raw single-patch reprojection is staged: proposed positions are checked together
on every incident triangle before they are committed. Degenerate faces, normal
reversals, excessive quality loss and sampled source-distance violations cancel
the affected vertex moves. Partial rollbacks are rechecked because they can
affect neighboring triangles; failure to stabilize within 16 passes rolls back
the entire projection. This also applies with `--no-smooth`.

Split templates undergo the same source-distance check on their new triangles.
Rejected templates cancel shared edge splits consistently, then regenerate and
recheck their neighbors before committing. Neither operation deletes bad faces
or relaxes the final error budget. Topology is checked after each stage,
including smoothing/projection, and failures identify the cycle and stage.
Geometry checks sample triangle vertices, edge midpoints and centroids; they
are not continuous Hausdorff guarantees or a general self-intersection test.

Curvature sizing seeds now use an immutable spatial BVH with the same distance
and target-length formula as the linear search. `raw_projection_safety` covers
projection-induced degeneracy, projection-induced interior distance violations,
accepted safe projection, and indexed-versus-linear sizing equality.

These safeguards do not require a correct CAD partition. They cover the raw
single-patch backend; the separate analytic CAD backend has its own checks.
Invalid input or an unmet final constraint still fails explicitly.

## Raw CUDA tolerance-query acceleration

Operation feasibility uses a bounded source-surface query: BVH branches outside
the existing local error radius are skipped, and the first source point within
that radius proves acceptance. Exact nearest-point queries remain in projection
and final distance auditing. The acceptance threshold, source surface, operation
ordering and safety checks are unchanged.

On RTX 3060, `浇道.stl` (identical to `examples/3.stl`),
`--gpu --iters 20 --feature-refine`, fresh sequential processes measured:

| Backend stage | Before | After |
|---|---:|---:|
| Collapse | 110.13 s | 36.03 s |
| Smooth and projection | 50.38 s | 9.83 s |
| Total remesh | 180.24 s | 60.70 s |

The before/after exported OBJ files are byte-identical (SHA256
`CAEDFAA24A6DB51F69C507D26B3DD5C0E27A546F484180F326D3FBC8471FEAA8`).
An optimized repeat took 60.88 s with the same hash and non-timing metrics.
`Unnamed-Body.stl` with the same options improved from 4.40 s to 2.33 s,
also with byte-identical output. All five GPU tests and CUDA memcheck passed.
These are backend times for the tested models, not a general performance guarantee.
The `raw_projection_safety` regression compares bounded queries, with and without
a BVH, against brute-force nearest-point acceptance across patches, tolerance
boundaries and spatial sizing. The larger remaining work is persistent GPU
topology/compaction to reduce per-pass host adjacency rebuilds and transfers.

## Experimental PAMO partition + raw CUDA batch backend

The single-mesh `--gpu` entry remains available. To use PAMO's actual partitioner
with the raw CUDA operators (including reference-only/nonanalytic patches):

```powershell
.\experiments\rxmesh-remesh\run_raw_partition.ps1 `
  -InputMesh 'F:\work\sunjie\Apollo\Apollo\test_data\浇道.stl' `
  -OutputDirectory '.\experiments\rxmesh-remesh\results\runner-batch' `
  -Workers 4 -Iterations 20 -FeatureRefine
```

The same runner now includes narrow-strip repair by default (Python, NumPy and
SciPy are required). Start from the original STL; no manually repaired snapshot
or model-specific patch list is needed:

```powershell
.\experiments\rxmesh-remesh\run_raw_partition.ps1 `
  -InputMesh .\examples\2.stl `
  -OutputDirectory .\experiments\rxmesh-remesh\results\2stl-remesh
```

Before GPU task dispatch, `tools/repair_narrow_strips.py` identifies small,
extremely elongated Freeform regions, groups their convex planar components,
rebuilds their interior connectivity and synchronizes new boundary samples with
neighboring triangles. Source vertices, fold edges and feature chains are
preserved. If a neighboring thin planar region cannot be safely subdivided,
the repair expands to that region, with at most eight expansions. Truly curved,
nonconvex, multi-loop or internally protected regions are retained with explicit
rejection reasons. The original partition snapshot remains immutable; GPU
reference queries use the repaired snapshot after its source-geometry checks.

After each GPU run, residual oversized edges can admit additional narrow planar
regions. `-StripRepairPasses` bounds GPU attempts (default 3, maximum 5); an
unchanged candidate set ends retries. The runner selects a complete result
before a partial result, then minimizes oversized-edge count and worst edge
ratio. Remaining oversized edges are reported as a partial result (exit 3).
Use `-SkipStripRepair` for an explicit raw-operator comparison, and `-Python`
to select the Python interpreter.

Each `strip_repair/pass_XX/` directory contains its input snapshot, repair
report, GPU mesh and GPU report. `pipeline.json` records the immutable source,
selected snapshot, attempted rounds, stop reason, remaining oversized edges,
and total elapsed time including preprocessing. The selected mesh is copied to
the usual `remeshed.ply`, so the third workflow retains one user-facing entry.
The current repair eligibility is conservative: at most 64 source faces and a
bounding-box aspect ratio above 100. The validated case uses uniform sizing;
unsupported geometries remain visible in the reports.

End-to-end check on `examples/2.stl`: automatic segmentation, one neighboring
strip expansion, 231 repaired regions and one GPU pass completed in 142.8 seconds
(including partitioning and preprocessing). The exported mesh has 268,103 faces,
zero oversized edges, worst edge/target ratio 1.33331 and quality P05 0.0231345.
An independent PLY audit found no zero-area or duplicate faces, nonmanifold
edges, or inconsistent interior-edge orientation; the original 81 open edges,
one connected component and Euler characteristic -1 were retained. This is one
validated model/run, not a guarantee for all CAD inputs.

### Faster `2.stl` run

```powershell
.\experiments\rxmesh-remesh\run_raw_partition.ps1 `
  -InputMesh .\examples\2.stl `
  -OutputDirectory .\experiments\rxmesh-remesh\results\2stl-fast `
  -ModelSeeds 250 -Iterations 12 -SmoothPasses 3
```

`-ModelSeeds` controls the analytic recognition seed budget when creating a
fresh partition. Its default remains 2500. Lowering it changes the partition;
unrecognized surfaces continue through the Freeform path. The residual search
budget, geometry tolerance, boundary constraints and final oversized-edge
recovery remain enabled. The faster settings are a measured tradeoff for this
input, not a universal replacement for the default settings.

Strip repair now indexes source faces once and reuses the source topology
audit across neighbor expansion. It also reuses local reconstruction only when
the target, tolerance and shared slicing grid match exactly. The original
2500-seed snapshot repair fell from 19.60 s to 9.71 s with byte-identical output.
The GPU smoother now retains ownership of its final vertex buffer for odd pass
counts, enabling the three-pass configuration without a freed-buffer access.

For the supplied `examples/3.stl` / 浇道 model, the user-facing script can
enable cylinder and cone curvature sizing while retaining the same global
target and original-STL geometry budget:

```powershell
& experiments/rxmesh-remesh/run_remesh.ps1 -InputMesh examples/3.stl -CurvedSurfaceSizing -TargetLength 8.9206295 -MaxError 1.7841259
```

This option uses one patch per task. If the protected-boundary subdivision
budget is exceeded, the user-facing runner retries the repaired snapshot
with feature-aware sizing and 16 patches per task. This mode refines sampled
curvature and protected feature lines with a bounded local target. The feature
field now relaxes over four regular target lengths by default
(`-FeatureTransitionWidthRatio` controls this), so local sizes change
gradually instead of jumping across one triangle. Convex, near-circular plane
patches also receive a new layered interior triangulation while their shared
contours remain fixed; each local result must preserve or improve mean and
lower-tail triangle quality. If that
candidate fails the topology, boundary, global-quality or edge-size guards,
the runner retries uniform sizing. The summary records the effective sizing
mode, transition quality, and planar coverage. The ordinary uniform mode remains the default
for other inputs; `-FeatureAwareSizing` selects feature-driven sizing directly.
For `examples/2.stl`, a direct graded run is:

```powershell
& .\experiments\rxmesh-remesh\run_remesh.ps1 -InputMesh .\examples\2.stl -FeatureAwareSizing -TargetLength 7.6237063 -MaxError 1.5247413
```

The user-facing runner exits with code 0 once it has saved the mesh and
`summary.json`, including when size or regional quality targets remain unmet.
`summary.json` records `acceptance_code` and `result_status`; `run.log` retains
the unresolved counts. Optional size, coverage, and visual audits are reported
separately and do not discard a completed mesh.

The coverage audit compares per-patch triangle geometry with the partition
snapshot, and identifies planar patches whose source tessellation remains.
Those patches require a separate shared-contour coarsening strategy; the
runner reports them instead of treating a valid global mesh as complete
coverage.
See [CONE_WALL_PREVIEW.md](CONE_WALL_PREVIEW.md) for the latest
same-input quality, geometry, feature, orientation and size comparison.

On the RTX 3060, a fresh run from the original `examples/2.stl` completed in
**85.37 s**, versus the earlier 142.79 s default run. This includes segmentation
and packaging (44.87 s), strip repair (9.48 s), GPU batch processing (28.50 s),
loading, Python startup and output. It does not reuse a saved partition.
The 227 strip repairs and all 426 GPU tasks were accepted, with zero rejected
strip repairs, unchanged tasks, fallback tasks or oversized output edges.
The output has 257,297 faces, max edge/target ratio 1.33322, mean quality 0.448142
(previously 0.460707) and P05 quality 0.0231339 (previously 0.0231345).
The native topology, geometry and boundary checks passed. This is a single
complete timed run with about 4.6 s margin, not a hard runtime guarantee.
Artifacts: `results/2stl_speed_20260926/full_fast/pipeline.json`,
`remeshed.ply.json` and `remeshed.ply` in the same directory.

### Phase 1 regional quality acceptance

The default batch policy now separates a geometrically valid candidate from an
accepted quality endpoint. `accepted` in the task report means it can be
assembled; `quality_accepted` and `provisional` state its local quality outcome.
Changed topology or coordinates alone do not stop recovery. Existing recovery
strategies are tried with the same endpoint guard, and a valid provisional
candidate is kept when they cannot meet that guard. Quality failure alone does
not fall back to the complete source task. Hard topology/geometry failures still
use the existing safe fallback. No GPU operators or partition/seam scheduling
were replaced in this phase.

`RegionQuality` reports the original mean and order-statistic P05, area-weighted
mean, low-quality area, and connected low-quality regions. Components connect
across patch interfaces. All components contribute to totals; only the largest
32 are serialized with bounds and patch IDs. Long and short edges, components of
faces incident to those edges, and target-size transition ratios are also
reported. Short edges use the existing collapse ratio as a diagnostic cutoff;
they are not all assumed removable near constrained geometry.

`--low-quality-threshold` / `-LowQualityThreshold` specify a diagnostic cutoff.
The default is the immutable batch input P05, derived once before boundary
refinement and held across repair rounds. It locates the input's poor tail;
it is not a universal quality target. Both sides of a comparison use this same
cutoff. Endpoint acceptance conservatively requires non-regression of mean,
P05, area-weighted mean, bad area and largest bad component, globally AND for
every original patch in the final assembled output. Tolerances of 1e-6 for
quality and 1e-6 times source area accommodate numerical comparisons; the old
percentage quality-loss allowances cannot certify final success.
If the input region has detected poor faces, non-regression alone is insufficient:
at least one of those quality/area measurements must improve measurably. A poor
unchanged region follows recovery and remains pending when no candidate helps.

`pending_patch_ids` identifies remaining work, with endpoint measurements and
`failure_mask`: mean=1, P05=2, area-weighted mean=4, bad area=8, largest bad
region=16, invalid faces=32, winding=64, unresolved input defect=128. The original patch guard also protects
against improvements in one region hiding regression in another. An artificial
cut may subdivide these per-patch comparisons, so the cross-interface connected
region audit remains necessary. These guards establish non-regression, not an
application-calibrated absolute quality certificate.

Constraint roles are orthogonal: persistent feature, intrinsic open boundary,
and partition interface. `ConstraintMotion` describes the current movement
classification without changing operator constraints. CADPART1 does not encode
whether an interface is a true geometric edge or a computation cut, or whether
all persistent features are true CAD features; the audit explicitly leaves
that provenance unknown. Corner/locked vertices remain fixed. Resolving that
provenance and opening pure computation seams is deferred. `regressed_patch_ids`
and `unresolved_patch_ids` distinguish regression from a pre-existing defect that
still lacks measurable quality progress; both remain pending.

An endpoint that fails quality is saved and returns **4**. The runner continues
supported residual repair, ranks quality before size, and keeps the candidate
and diagnostics if no additional supported repair exists. It no longer forces
centroid subdivision merely to eliminate `unchanged`. `-LegacyCoverageAcceptance`
(`--legacy-coverage-acceptance` for the executable) restores the previous
acceptance and coverage-refinement decisions for comparison or rollback; the
new audits still run. It is not a way to certify a rejected endpoint.

Independent comparison, targeted tests, timing and reproduction details for
this change are in `PHASE1_REVIEW.md`. Independent distance samples and nearest
reference-normal checks are not proofs of Hausdorff distance or absence of
self-intersection.

The independent shared-boundary and input-preservation regression checks run as:

```powershell
python .\experiments\rxmesh-remesh\tests\test_repair_narrow_strips.py
```

For repeatable debugging, reuse the original `input.cadpart` with
`-SavedPartition`. Calling the executable directly bypasses the runner's repair:

```powershell
.\experiments\rxmesh-remesh\build_rx\Release\cad_raw_partition_cli.exe input.cadpart out.ply --workers 4 --iters 20 --feature-refine
```

`--workers 1` executes the same patch algorithm serially. `--memory-mb` limits
aggregate reserved device workspace; the default is 60% of free GPU memory.
The scheduler reuses PAMO's `RemeshWorkerPool`, estimates task size from source
faces, surface area and the finest requested length, and selects large ready
tasks that fit the remaining budget. Each task has a nonblocking CUDA stream;
each raw kernel still launches as many blocks as its current element count
requires. A large region is not restricted to one block. Admission reservations
are conservative estimates; actual cached buffer allocation has a hard cap per
task. Exceeding it retains that region's source mesh and reports the failure.

Shared boundaries are subdivided globally before workers start, using common
vertex IDs. Independent tasks cannot split, collapse or move those boundaries.
Locked vertices are mapped back by exact coordinates with uniqueness checks;
ambiguous identities reject the region. Workers read immutable input, and results
are assembled in patch order. Each patch passes the raw topology and sampled
reference-distance checks; assembly verifies shared edges and mesh topology.
Failed regions retain their prepared source geometry. Exit 3 explicitly denotes
this partial result; exit 0 requires every region to be accepted. `out.ply.json`
records per-region timings, allocation caps, acceptance and failure reasons.
The wrapper also records partition/packaging and end-to-end time in `pipeline.json`.

On the runner fixture, PAMO produced 457 unequal regions. An initial 20-round
comparison measured 120.37 s with one worker and 37.38 s with four, with identical
PLY SHA256 `778C3A806B93217448160817894BD166F0FFB648451D53603FB71A48568B1168`.
All 457 regions were accepted. These batch times include shared-boundary
preparation and assembly but exclude partitioning (about 1.8 s for this input).
The final quiet-logging build repeated at 38.27 s with the same PLY hash;
loading the saved snapshot through the wrapper and exporting took 39.34 s total.
The 8-region `Unnamed-Body.stl` case took 6.88 s in the batch backend and
7.80 s end-to-end, slower than its earlier 2.33 s whole-mesh backend timing.
Partitioning is therefore not enabled automatically for small models.

This is an experimental alternative, not a quality-equivalent replacement for
whole-mesh remeshing. Fixing all partition seams reduces freedom to improve
triangles: on this fixture mean/P05 quality changed from 0.9615/0.8461 for the
whole mesh to 0.9264/0.6064 for the batch output. Curvature sizing is evaluated
within each source region, so it can also differ near seams. The next geometry
step is coordinated seam improvement or relaxing non-feature partition seams.
GPU allocation, host adjacency reconstruction and host readback remain; tiny
regions are concurrent tasks, not yet packed into one kernel launch. Peak active
tasks is not a measurement of simultaneous GPU kernel occupancy. There is no
general self-intersection guarantee or continuous Hausdorff bound.

## CAD partition handoff

```powershell
.\experiments\rxmesh-remesh\run_cad_remesh.ps1
```

The default input is `examples/Unnamed-Body.stl`. This runs PAMO's existing
model-first partitioner and patch graph, packages its indexed `CADPART1`
snapshot, then runs the GPU backend with analytic plane/cylinder projection.
The result is `results/Unnamed-Body-cad/remeshed.obj`; OBJ groups retain the
geometric patch IDs. `validation.json` checks source feature edges, source
feature vertex positions, topology, volume, and bidirectional sampled distance.
The script requires the existing PAMO Release segmenter, the built GPU CLI,
and Python with NumPy. Override `-Segmenter`, `-Remesher`, `-InputMesh`,
`-OutputDirectory`, `-TargetLength`, `-MaxError`, or `-Iterations` as needed.

To reuse a saved partition without repeating recognition:

```powershell
.\experiments\rxmesh-remesh\build_rx\Release\cad_adaptive_cli.exe INPUT.cadpart OUTPUT.obj --partition --gpu
```

Source feature vertices and corners retain their positions. Before the GPU
stage, long feature segments are subdivided conformingly on the CPU without
changing their polyline geometry; all these boundary vertices remain fixed.
Interior edges can split even when their endpoints are fixed. This is currently
constant-length interior remeshing with analytic projection; it does not refit
or coarsen the source boundary curves. The default CAD surface tolerance is 0.001 times the bounding-box
diagonal, overridable with `--max-error`. Before export, CAD input mode checks
feature preservation, patch presence, face orientation and sampled analytic
surface deviation. Unsupported surfaces/reference-mesh targets and feature
edges inside a single geometric patch fail explicitly rather than running
unconstrained. This bridge does not replace PAMO's native remesher.

## CPU tests (001.0–001.4 + 002.0, no CUDA)

### Regional candidate experiments

`-UnifiedRegionalSizing` is an optional CPU/GPU shared size-field experiment,
used with explicit regional targets and a reference output. See
[PHASE6_SIZING.md](PHASE6_SIZING.md) for rejected results and reproduction.
The existing `-ComputeRegions 256` path reached 83.42 seconds on one complete
STL run, but failed quality and size checks; it is not an accepted solution.
See [PHASE7_COMPUTE.md](PHASE7_COMPUTE.md).

`-ExploreProvisionalChildren` (CLI `--explore-provisional-children`) optionally
compares a legal provisional packed-task candidate with single-patch recovery.
The current policy first chooses nonregressing ownership regions, retaining
their legal incumbents elsewhere, then checks the connected union globally.
The comparison checks defect quality per patch and globally, plus long and
short edge counts. Already healthy nonregressing regions reuse their incumbent.
Rejected exploration retains the legal incumbent.
It does not certify endpoint quality or the 90-second performance target.
All three options are off by default.

Final size recovery now selects regions from the actual assembled endpoint
targets, rather than packed-task ratio summaries. This removes a known omission
of 394 local-target long edges, but can worsen regional shape quality; endpoint
failure remains explicit. See [PHASE10_CONTRACT.md](PHASE10_CONTRACT.md) for
full timing, rejected quality tradeoffs, and the PLY coordinate-type audit fix.

```powershell
cmake -S experiments/rxmesh-remesh -B experiments/rxmesh-remesh/build -G Ninja
cmake --build experiments/rxmesh-remesh/build
ctest --test-dir experiments/rxmesh-remesh/build --output-on-failure
```

## GPU tests (001.0–001.5 + 002.1–002.2)

Uses the in-repo RXMesh pin `e468c34` and does not link CadMesh/VCG into CUDA TUs.

```powershell
.\experiments\rxmesh-remesh\run_gpu.ps1
```

Scale bench (writes `out.json`):

```powershell
.\experiments\rxmesh-remesh\build_rx\Release\cad_adaptive_cli.exe --grid 1000000 out.obj 0.002 --gpu --iters 2
```

Recorded on RTX 3060 (plane grid, constant `h`): 100K GPU 2.2s / CPU 4.5s; 1M GPU 16s / CPU 519s; 5M GPU 85s; 10M GPU 296s.

Those scale results are from 001. The 002 1M comparison records topology-stage
time of 3.900s → 2.816s, but total remesh time of 16.43s → 18.06s with different
finite-iteration outputs. See the ticket for the full report and measurement caveats.

Optional `-TrackRegionCandidates` / `--track-region-candidates` audits intermediate cycles against an immutable reference and only restores a snapshot if it does not regress the completed run. It remains disabled: the full experiment failed quality acceptance and took 105.89 s. See [PHASE11_CANDIDATES.md](PHASE11_CANDIDATES.md).

Optional `-SizeFeasibleFinalRefine` / `--size-feasible-final-refine` prioritizes size-feasible child edges and checks parent/child orientation. It remains disabled: 100.95 s full runtime, four explicit size defects and regional quality regressions. See [PHASE12_FEASIBLE.md](PHASE12_FEASIBLE.md).

The optional size-feasible refinement now reports residual selection/stopping, allows measurable intermediate severity reduction and searches a bounded finer candidate grid only after coarse candidates fail. The full result has zero size-long edges but still fails regional quality at 102.06 s; it remains disabled. See [PHASE13_RESIDUAL.md](PHASE13_RESIDUAL.md).

Optional `-SelectFinalRegions` / `--select-final-regions` compares both final refinement strategies and selects nonregressing connected groups with stable anchor identities. Ten actual regions passed independent quality checks, but the full endpoint remains unsatisfied at 109.49 s. Host defect-area comparisons now use their own metric scale and declared float coordinates are evaluated in double precision. CADPART1 also carries a partial hard-feature declaration currently ignored by the experimental loader; see [PHASE14_SELECTION.md](PHASE14_SELECTION.md).

## 阶段 15：完成态区域试验

见 [PHASE15_FLAT.md](PHASE15_FLAT.md)。严格平面合并在 2.stl 无可合并接口；完成态平面对角线仅改善普通平面，主要坏区域不变，因此未接入默认流程。新增 `repair_narrow_strips.py --only-requested-regions` 用于显式区域和必要接缝邻居的阶段重建，默认关闭，不能把复用输出的耗时当作完整流程。总质量和 90 秒目标尚未达到。

## 阶段 16：精度来源与邻区约束

见 [PHASE16_POLYGON.md](PHASE16_POLYGON.md)。完成态参考精度恢复保持 float32 输出坐标不变；两窄带联合重建可生成合法候选，但全局和邻区质量仍有退化，未采用。8 工作线程从原始 2.stl 完整流程为 107.39 秒，输出与 4 线程对照逐字节一致，终点质量与 90 秒目标均未达到。


## Optional constraint and native-size evidence

`run_raw_partition.ps1 -AuditFields` (`cad_raw_partition_cli --audit-fields`) writes immutable input constraint evidence and final native vertex/edge sizing sidecars. `pipeline.json` records which round contains the selected sidecars; input and output vertex IDs are different domains. No movement permission changes. See [PHASE19_CONSTRAINTS.md](PHASE19_CONSTRAINTS.md) for verification, full timing, and limitations.
The version-2 audit also exports `*.source_patches.tsv`: producer patch type,
Fillet role, and support patch IDs, independently checked against CADPART1.
This records model-recognition evidence without certifying a CAD curve or
unlocking any boundary; see [PHASE34_PATCH_PROVENANCE.md](PHASE34_PATCH_PROVENANCE.md).


## Optional joint-region tradeoff policy

`-RegionalQualityPolicy` and `-NativeSizeEvidence` enable an explicit bounded regional transaction policy. Neighbor means and P05 become advisory; feature/topology/original-geometry and native-size checks still gate adoption. No production loss budget is assumed. See [PHASE20_JOINT.md](PHASE20_JOINT.md) for measured candidate gains, costs, and remaining checks.

`tools/try_original_region_recovery.py` is a separate, optional candidate
tool for at most two explicitly selected source patches. It remeshes from the
original repaired snapshot, requires an exact boundary graph, reconstructs
native targets by vertex identity, and checks a supplied quality policy,
feature/seam/open chains, topology, direction and sampled original-STL error.
It never overwrites the incumbent. On `2.stl`, patches 4501/4510 reduce
low-quality area by 15.50 but take an additional 35.90 s; the complete
original-STL run with audit export and recovery takes 144.92 s. This is not
enabled by default and does not meet the 90 s target. See
[PHASE40_ORIGINAL_REGION_RECOVERY.md](PHASE40_ORIGINAL_REGION_RECOVERY.md).
