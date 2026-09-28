# Cone-wall remesh preview (`examples/3.stl`)

The earlier `-CylinderCurvatureSizing` run refined cylinders but left all 61
recognized cone patches at the global target. The source snapshot contains
5,001 cone triangles. `BoundarySizingField::create` now computes each cone's
circumferential principal curvature from its semi-angle and smallest sampled
radial distance. The resulting patch target is bounded by the same chord,
normal-angle, split-ratio and original geometry-error inputs as cylinders.
This is a patch-wide conservative size: it does not yet vary along the cone.
The CLI and scripts expose `-CurvedSurfaceSizing`; the former cylinder switch
name remains an alias.

Both runs below start from the same raw STL with 2,500 model seeds,
target length 8.9206295, max geometry error 1.7841259, 12 iterations,
3 smoothing passes, 8 collapse/flip passes, one patch per GPU task, and the
same quality acceptance policy. The only algorithm change is adding cone
sizing to the existing cylinder sizing.

| Metric | Cylinder sizing only | Cylinder and cone sizing |
| --- | ---: | ---: |
| Total faces | 94,604 | 143,108 |
| Mean quality / P05 | .822075 / .312935 | .873693 / .607532 |
| Low-quality area | 1.357629 | 1.059569 |
| Largest low-quality component area | .181137 | .181137 |
| Cone faces | 8,435 | 51,042 |
| Cone edge length P05 / median / P95 | .923 / 4.810 / 10.315 | 1.077 / 2.069 / 3.196 |
| Cone quality mean / P05 | .587736 / .177731 | .898749 / .688277 |
| Cone low-quality area | .403015 | .026760 |
| Native long edges | 0 | 0 |
| Native short edges | 42,661 | 51,195 |
| Unresolved patches | 3 | 3 |
| Full pipeline time | 25.01 s | 30.06 s |

The preview `results/cone_preview_20260928/cone_wall_closeup.png` renders
the same cone patch with identical camera and axes. Its face count grows
from 332 to 1,726. `results/cone_preview_20260928/curved_full/remeshed.stl`
has 143,108 faces and float32 triangle coordinates identical to its PLY.

The independent original-STL audit found one connected component, Euler
characteristic -6, zero open/nonmanifold/inconsistently wound edges, duplicate
faces or float32 zero-area faces. All 143,108 output face centroids had
positive orientation relative to their nearest same-patch source triangle.
Sampled output-to-original and original-to-output maximum distances were
.678421 and .725803, within the unchanged 1.784126 budget. This sampling is
not a continuous Hausdorff or self-intersection certificate. Original feature
and seam chains were complete at two float32 ULP coordinate tolerance; the
tighter sub-ULP diagnostic reported 160 missing chains due to quantization.
Three planar patches (105, 106, 167) remain unresolved, so this does not
claim universal quality.

The complete run and three targeted regression tests passed. Reproduce with
an empty output directory:

```powershell
& experiments/rxmesh-remesh/run_remesh.ps1 -InputMesh examples/3.stl -OutputDirectory experiments/rxmesh-remesh/results/my_curved_run -CurvedSurfaceSizing -TargetLength 8.9206295 -MaxError 1.7841259
```

The sizing is opt-in. Other inputs, especially `examples/2.stl` with very
small cylinders, need an independent density and geometry review before
using it as a default. No commit or push was made.
