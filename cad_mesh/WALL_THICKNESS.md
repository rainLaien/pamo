# Optional wall thickness

`WallThicknessCalculator` adapts the rolling-ball search from
`F:/work/sunjie/Apollo/Apollo/src/algorithm/thickness/wall_thickness_calculator.cpp`.
It retains AABB traversal, inscribed-ball shrinking, and containment/contact
checks. Each query starts at an existing remesh vertex;
there is no subdivision, resampling, representative relocation, or mesh edit.
The sphere center is constrained to the inward vertex-normal line. The solver
shrinks the radius against nearby triangles, then checks containment and wall
contact; it does not search every possible sphere center in the mesh.
Its private `WallThicknessGeometry.h` implements vector operations, triangles,
slab ray/box tests and a median-split BVH in standard C++17. The public input
is double coordinate arrays and triangle indices. Neither the public header
nor the solver depends on VCGLib or Apollo runtime/model classes. CadMesh's
other modules still use their existing geometry dependencies.

## Run

Reconfigure/build the native executable after adding the new source, then:

```powershell
.\cad_mesh\cpu.ps1 -InputStl .\examples\3.stl -WallThickness -ThicknessWorkers 30
.\cad_mesh\cuda.ps1 -InputStl .\examples\3.stl -WallThickness -ThicknessWorkers 20
.\cad_mesh\remesh_all.ps1 -WallThickness
```

The same options are forwarded by remesh_saved.ps1 and remesh_others.ps1.
Omit `-WallThickness` to disable all thickness work and omit its PLY properties.
Native switches are `--wall-thickness`, `--thickness-workers` (1..128),
`--thickness-minimum` (default 0.01 model units; zero selects 0.0075 times the
smallest bounding-box dimension) and `--thickness-contact-angle-deg` (0..180,
default 0). The minimum is an acceptance floor, not convergence accuracy.
The flag requires remesh and cannot be combined with stop-after-partition.

For an already remeshed ASCII PLY, use the read-only measurement executable:

```powershell
cmake --build .\cad_mesh\build --target cad_mesh_thickness
.\cad_mesh\build\cad_mesh_thickness.exe input_remeshed.ply output_with_thickness.ply --workers 20
```

This path reads the existing vertices and triangular faces, computes face
properties, then copies the original coordinates and face data while appending
`wall_thickness`, `thickness_valid`, and `thickness_status` to each vertex row.

## Measurement and output

Measurements run after final mesh compaction, across the whole final mesh,
including preserved faces. Patch IDs do not limit opposite contacts.
The private measurement mesh copies face coordinates without changing geometry
or face order. Face-mode queries do not require adjacency or vertex welding;
shared-edge ray hits retain Apollo's distance-based deduplication. A read-only AABB tree is built
once, then workers claim batches of 32 vertices. The vertex direction is the
angle-weighted sum of incident unit face normals. Thickness is CPU-only even in a
CUDA remesh run, and its pool runs after remesh pools have completed.

The existing remesh_result.ply gains these **vertex** properties:

| Property | Meaning |
| --- | --- |
| wall_thickness | Accepted rolling-ball diameter in model units, or NaN |
| thickness_valid | 1 for accepted measurements, 0 otherwise |
| thickness_status | Diagnostic status independent of the numeric value |

Status values used by vertex sampling: 0 measured, 2 invalid normal,
5 invalid sphere, 6 nonfinite, 7 below minimum, 8 above maximum,
11 penetrating sphere,
12 local-surface contact rather than accepted wall contact.
The other Apollo status IDs are reserved in the public enum.
`remeshed` and `remesh_reason` retain their existing meanings.

The log reports valid vertex count/fraction, minimum, maximum, arithmetic mean,
preparation time and sampling time. Zero valid vertices is exported with all values invalid; zero-valued summary
statistics in that case are placeholders, not measured zero thickness.

This is a vertex-query adaptation of Apollo's rolling-ball search. It relies on
the input's face orientation to define inward directions and retains Apollo's
maximum radius of half the smallest bounding-box dimension. It does not repair
open/self-intersecting meshes. Structural input errors return an error; failed
individual sphere queries remain visible as invalid vertex measurements.

The BVH construction and traversal backend have changed, so floating-point tie
ordering may differ from Apollo. Identical numerical results are not claimed.
The optional CUDA path currently runs one thread per vertex to find the first
nonincident face along the inward normal, using the pointer-free BVH. The
rolling-ball shrink, containment, contact and inside/outside certification
still run on the CPU, using the GPU hit as their initial radius bound. Thus the
measurement definition is retained, while only ray initialization is offloaded.
Native CUDA remesh runs enable this stage; CPU runs do not. The standalone
measurement executable accepts `--cuda-ray-seeds` to request it and falls back
to CPU ray queries if CUDA initialization or execution fails.
Iteration counts need not match across vertices: CUDA threads can run their own
bounded loops, with warp divergence affecting speed rather than correctness.

The native executables build with this source. The earlier face-mode smoke
statistics do not describe this vertex-mode implementation.
