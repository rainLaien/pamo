# Phase 36: bounded torus norm trial, rejected

The phase-35 profile identified torus fitting as 53.9% of diagnostic
partition GPU kernel time. A narrow trial replaced the two torus `hypot`
sites in `CudaAnalyticKernels.h` with direct `sqrt(x*x+y*y)` in the common
finite range, retaining `hypot` for extreme magnitudes. The original
`cad_mesh_segment.exe` was saved before the trial. No remesh setting or
feature/geometry tolerance changed.

From raw `examples/2.stl`, model seed budget 250:

| Measure | Phase-35 original | Trial |
| --- | ---: | ---: |
| GPU kernel with timing events | 19.776 s | 21.065 s |
| Model-first partition | 36.398 s | 42.738 s |
| Segmenter wall | 42.317 s | 48.766 s |
| Partition PLY | Baseline | Byte-identical |
| Model/constraint JSON | Baseline | Different |

The timing is one run each and CUDA initialization varied (0.679 versus
5.365 s); it does not establish a precise speed penalty. It does establish
no observed kernel gain in this trial. The JSON differed in 1,085 scalar
values; patch 307's fitted torus major radius changed from 296.813 to
370.054 while the output partition faces stayed identical. Exact model
identity therefore failed even though the mesh hash matched. This is not a
safe output-identical performance optimization. It was rejected before a
full remesh run.

The source helper was removed, and `cad_mesh_segment.exe` was restored to
its saved SHA256 `86F081057EF3C64D36AF3BF8FB21DDBE33854FBD231E336DE15968C8C4E5CDAF`.
The unrelated preexisting changes in `cad_mesh/src/CudaAnalyticFitting.cpp`
were untouched. Logs and the saved binary are in
`results/phase36_torus_trial/`. No commit or push was made.
