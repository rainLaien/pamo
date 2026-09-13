# Optional wall thickness

`WallThicknessCalculator` adapts the face-sampling rolling-ball algorithm from
`F:/work/sunjie/Apollo/Apollo/src/algorithm/thickness/wall_thickness_calculator.cpp`.
It retains AABB traversal, inscribed-ball shrinking, containment/contact checks
and `refineFaceRepresentative`. It does not replace these with normal-ray length.
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

## Measurement and output

Measurements run after final mesh compaction, across the whole final mesh,
including preserved faces. Patch IDs do not limit opposite contacts.
The private measurement mesh copies face coordinates without changing geometry
or face order. Face-mode queries do not require adjacency or vertex welding;
shared-edge ray hits retain Apollo's distance-based deduplication. A read-only AABB tree is built
once, then workers claim batches of 32 faces. Thickness is CPU-only even in a
CUDA remesh run, and its pool runs after remesh pools have completed.

The existing remesh_result.ply gains these **face** properties:

| Property | Meaning |
| --- | --- |
| wall_thickness | Accepted rolling-ball diameter in model units, or NaN |
| thickness_valid | 1 for accepted measurements, 0 otherwise |
| thickness_status | Diagnostic status independent of the numeric value |

Status values used by face sampling: 0 measured, 2 invalid normal,
5 invalid sphere, 6 nonfinite, 7 below minimum, 8 above maximum,
9 accepted relocated representative sample, 11 penetrating sphere,
12 local-surface contact rather than accepted wall contact.
The other Apollo status IDs are reserved in the public enum.
`remeshed` and `remesh_reason` retain their existing meanings.

The log reports valid face count, valid area fraction, minimum, maximum,
area-weighted mean over valid faces, preparation time and sampling time.
Zero valid faces is exported with all values invalid; zero-valued summary
statistics in that case are placeholders, not measured zero thickness.

This is the Apollo **face** mode. Vertex display interpolation, reconstruction
of failed values and extra sampling subdivision are not enabled. It relies on
the input's face orientation to define inward directions and retains Apollo's
maximum radius of half the smallest bounding-box dimension. It does not repair
open/self-intersecting meshes. Structural input errors return an error; failed
individual sphere queries remain visible as invalid face measurements.

The BVH construction and traversal backend have changed, so floating-point tie
ordering may differ from Apollo. Identical numerical results are not claimed.
The unused Apollo vertex-only source relocation/adjacency branches were removed;
face representative refinement and full-mesh containment remain enabled.

No compilation, tests or mesh runs were performed for this change.
