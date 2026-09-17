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
