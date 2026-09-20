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
