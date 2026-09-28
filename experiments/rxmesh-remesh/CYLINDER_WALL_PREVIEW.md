# Cylinder-wall remesh quality comparison (`examples/3.stl`)

The previous user-facing run used a global target length of 8.9206295 and
geometry budget of 1.7841259. Its native target P05 was also 8.9206295, so
the cylindrical walls had almost no curvature-driven local sizing. Merely
raising the analytic model seed budget from 250 to 2500 produced the same
source snapshot and mesh on this input. The existing
`BoundarySizingField::create(..., curvatureSizing=true)` implementation was
present but not reachable through the raw remesh CLI.

The CLI and PowerShell pipeline now expose an opt-in cylinder-curvature sizing
switch. The user-facing `run_remesh.ps1` exposes the same switch, uses the
original 2500 model-seed default, and automatically chooses one patch per
task when the switch is set. The original uniform mode remains available;
there is no blanket new default for all STL inputs. The switch cannot be
combined with the separate near-feature refinement or explicit patch target
mode.

## Same-input comparison

Both outputs start at raw `examples/3.stl`, use h=8.9206295,
max error=1.7841259, model seeds=2500 (the 250- and 2500-seed snapshots were
byte-identical), iterations=12, smoothing=3, collapse=8, flip=8, and the
same global quality policy. The accepted cylinder candidate changes only
curvature sizing and patch packing 16 to 1. A diagnostic curvature run at
pack 16 had four nearly reversed faces in cylinder patch 249; it was rejected.
Disabling flip globally or collapse globally was also rejected because it
produced poor quality or excessive short edges. Pack 1 retained the curvature
benefit without the observed fold.

| Metric | Uniform result | Cylinder curvature, pack 1 |
| --- | ---: | ---: |
| Faces | 57,184 | 94,604 |
| Mean quality | .740958 | .822075 |
| P05 | .228913 | .312935 |
| Area-weighted mean | .877129 | .892954 |
| Low-quality area | 6.025884 | 1.357629 |
| Largest low-quality component area | .435300 | .181137 |
| Cylindrical face quality mean / P05 | .760236 / .225454 | .864245 / .556140 |
| Cylinder edge length P05 / median / P95 | 1.256 / 7.660 / 11.102 | 1.950 / 4.581 / 7.216 |
| Native short edges | 34,509 of 85,776 (40.2%) | 42,661 of 141,906 (30.1%) |
| Native long edges | 0 | 0 |
| Unresolved patches | 4 | 3 |

The pack-1 candidate's native target field has min/P05/max
2.367/3.093/8.921. The preview `cylinder_wall_closeup.png` compares the same visible
cylinder-wall area directly. Short-edge count grows with the extra faces,
while the fraction of short edges falls under each mesh's native target.

## Geometry and constraints

An independent full-mesh audit of the pack-1 candidate found one connected
component, Euler characteristic -6, no nonmanifold edges, inconsistent
winding, duplicate or float32 zero-area faces, and zero nonpositive
nearest-source normal comparisons over 94,604 face centroids. Sampled
output-to-original and original-to-output maximum distances were
1.400536/1.684415, within the unchanged 1.784126 budget. These samples do
not prove a continuous Hausdorff bound or absence of self-intersections.

The original chain audit used tolerance 0.0000892, below one float32 ULP at
this model's coordinate magnitude, and reported 90 missing feature/seam
segments for the baseline and 115 for the candidate. All 25 additional
flagged segments retained their original endpoints and continuous output
feature paths. Their maximum departure from the source chord was
0.0001832. Using a predeclared coordinate-precision tolerance of two
float32 ULP (0.0004883), both outputs have zero missing feature, seam or
open-boundary chains and zero missing anchors. This is a quantization-scale
audit, not CAD certification. Native size and source-provenance sidecars
verified independently.

The user-facing script reproduced the accepted pack-1 PLY byte-for-byte from
raw STL (SHA256 `8b70336addbbf9b9d4338dddd87b060a9dfe1168087f1fa33860bce8259c3cb9`). Its full-flow
time was 25.010 s on one run; the earlier uniform script run was 6.856 s.
The user currently prioritizes quality, but the added time and 37,420 faces
are material costs. For `examples/2.stl`, 984 cylinder patches include radii
near 0.2 and the same curvature formula would request targets near 0.05;
this option has not been accepted or enabled by default for that model.

## Reproduce the accepted `3.stl` result

```powershell
& experiments/rxmesh-remesh/run_remesh.ps1 -InputMesh examples/3.stl -OutputDirectory experiments/rxmesh-remesh/results/cylinder_preview_20260928/reproduce -CylinderCurvatureSizing -TargetLength 8.9206295 -MaxError 1.7841259
```

The output directory must be new and empty. The actual PLY, STL, previews,
audits and rejected diagnostic variants are in
`results/cylinder_preview_20260928/`. No commit or push was made.
