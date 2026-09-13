# RXMesh constrained remeshing

This opt-in CUDA executable starts with the cleaned STL triangulation. It does
not construct UV charts or replace patch interiors with boundary-only polygons.
RXMesh is pinned to `e468c34ffabc70cd207309bce662d4821a9ed3b7`.

## Run on Windows

From the repository root, with Visual Studio 2022, CUDA 12.6, CMake 3.25+,
Git, and VCGLib installed:

```powershell
.\cad_mesh\run_rxmesh.ps1 -InputMesh .\examples\2.stl `
  -TargetEdgeRatio 0.01 -MaximumDeviation 0.03
```

The first invocation downloads and builds RXMesh dependencies. Later runs may
use `-SkipBuild`. Build/download time is separate from remeshing time. The wrapper
defaults to CUDA architecture 89 (the development machine's RTX 4060).
`-VcglibRoot` and `-CudaVersion` override local toolchain paths/version.
`-RxMeshSource` optionally selects an existing checkout of the pinned RXMesh.

`TargetEdgeRatio` is relative to the **whole model's bounding-box diagonal**.
`TargetEdgeLength` overrides it with an absolute length. `MaximumDeviation` is
an independent absolute distance in STL coordinate units. Geometry guards retain
finer source sampling where coarsening exceeds the distance/normal limits.
Neither setting guarantees an exact uniform edge length.
`NormalDegrees` defaults to 10, `FeatureAngle` to 45, `Iterations` to 5.

By default, geometric regions are connected components separated by sharp
edges. `-Segment` uses the project's CAD segmentation first. An existing binary
remesh snapshot can instead be passed as `InputMesh` with `-PartitionSnapshot`.
These options reuse partition IDs/constraints, **not** analytic chart remeshing.

## Algorithm and constraints

1. Load/weld/clean through the existing mesh loader. Separate non-manifold
   connections into manifold sheets by duplicating face-corner vertices, keeping
   every cleaned triangle and its position. Inconsistent winding connections are
   cut as well. Preserve these triangles as the immutable reference surface.
2. Split long protected edges globally. Both incident triangles use the same
   new vertex ID. Fix all resulting boundary/feature vertices and edges.
3. Build a reference BVH for each geometric region and a uniform target-size field.
4. RXMesh creates independent computational patches. GPU cavities perform
   collapse, flip, split, and tangential relocation with region-restricted
   projection, sampled distance, normal, and local topology checks.
5. Export a compact mesh; audit protected seams, topology, and sampled distance
   in both directions. Record quality and total time, including loading,
   partitioning, GPU setup, remeshing, and final audit/output.

An accepted output is `rxmesh_result.ply`, with per-face `source_patch_id`.
`rxmesh_report.json` records counts, minimum-angle statistics, measured timing,
and constraint/distance failures. A failed final audit retains
`rxmesh_candidate.ply` and returns a nonzero exit code. Existing nonempty output
directories are rejected. The input is never overwritten.

Distance checks are sampled; this implementation does not certify the continuous
Hausdorff distance or perform global self-intersection detection. The report
states both limitations. The shape/topology pass flag is not a guarantee of
high triangle quality. Fixed dense/noisy feature boundaries may constrain
coarsening and leave narrow triangles. A 60-second result is a performance
target, not an enforced timeout or an established guarantee.
Manifold sheet separation preserves geometry but may create coincident open
seams. It does not fill input holes or guarantee a watertight volume.

## Integration checks

```powershell
python .\cad_mesh\tests\test_rxmesh_entry.py `
  .\cad_mesh\build_rxmesh\rxmesh\Release\cad_mesh_rxmesh.exe
```

Checks exercise shared cube edges, a dense open plane, actual topology changes,
sampled shape preservation, and output/argument handling on a CUDA device.

RXMesh's own source and license: https://github.com/owensgroup/RXMesh
