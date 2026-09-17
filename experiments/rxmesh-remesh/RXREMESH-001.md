# RXREMESH-001 — locked first slice

Status: **001.0–001.5 COMPLETE** (2026-09-14)  
Date: 2026-09-14  
Next: [`RXREMESH-002.md`](RXREMESH-002.md)  
Branch policy: do not edit `cad_mesh/` mainline remeshers for this ticket.

This ticket is the first *product slice*, not “isotropic GPU remesh only”.
Section 37’s Phase 0–3 are **internal gates** inside this ticket. Later types
(Cone / Sphere / FeatureCurve / Freeform / Hausdorff / GPU BVH / PAMO wiring)
are out of scope.

---

## Why this exists

`cad_mesh/rxmesh/` already changes connectivity on GPU. It is **not** this
architecture:

| Existing `cad_mesh/rxmesh` | This experiment |
| --- | --- |
| `edit<Kind>` mixes split / flip / collapse | one operator per kernel |
| BVH projection for every new point | analytic plane / cylinder first |
| region id + fixed-source-id packed in `int3` | SoA semantic attributes |
| uniform `target_size` | `h(x)` from error / curvature / feature distance |
| CAD policy inside cavity lambdas | `RemeshPolicy` tested on CPU |
| double everywhere, CadMesh+VCG in input TU | float GPU working set; no VCG in CUDA TUs |

Keep `cad_mesh/rxmesh` as the cavity/scheduler reference and quality/perf
baseline. Do not “improve it into” this design.

---

## What already exists (reuse, do not rebuild)

| Need | Existing | How 001 uses it |
| --- | --- | --- |
| Patch types | `PatchSurfaceType` in `cad_mesh/include/CadMesh/Types.h` | Mirror as `PatchType`; same numeric order |
| Patch adjacency | `PatchAdjacency` + `PatchGraphBuilder` | Converter later (Phase 6). 001 takes explicit `PatchId` / boundary flags |
| Corners / locked verts | `RemeshConstraint::{Corner,Junction}VertexIds` | Input `VertexConstraint` |
| Boundary chains | `BoundaryChain` | 001 only needs `isPatchBoundary` + `featureCurveId` on edges |
| Analytic project | `ProjectAnalytic` in `NativeRemesher.cpp` | Copy plane/cylinder formulas into a VCG-free projector |
| Chord sizing | `GenericSurfaceRemesher` uses `sqrt(4 R ε)` | **Do not copy.** Lock `h = sqrt(8 R ε)` from sagitta `ε ≈ R θ² / 8` |
| GPU cavities | RXMesh `CavityManager` + scheduler | **This is TopologyScheduler v1.** No custom MIS / coloring |
| RXMesh pin | `e468c34ffabc70cd207309bce662d4821a9ed3b7` | Same pin as `cad_mesh/rxmesh` |
| Split ratio | existing `16/9` length² test | Keep `4/3` linear (`16/9` squared) |

PAMO recognition stays CPU and stays in `cad_mesh/`. 001 consumes **already
labeled** meshes.

---

## Locked architecture

```text
triangle mesh + PatchId + ConstraintMask [+ plane/cyl params]
        │
        ▼
 CadSemanticModel   (CPU, SoA host arrays)
        │
        ▼
 RemeshFieldBuilder
   h = clamp(min(hCurvature, hFeature, hPatch, hError), hMin, hMax)
        │
        ▼
 RemeshPolicy          GeometryProjector
   canSplit/Collapse     projectSurface(patchId, p)
   canFlip/relocate      projectFeature(curveId, p)   [stub: edge lerp]
        │
        ▼
 IRemeshBackend  ── CpuRemeshBackend (policy + tiny meshes)
                 └── RxMeshBackend   (dynamic connectivity)
                        │
                        ▼
              GPU Validation + JSON metrics
```

Semantic IDs **propagate with every operator**. Never re-guess a patch after
an edit.

### Split

```text
same-patch edge  →  PatchId(newV) = PatchId(edge)
boundary edge    →  newV.constraint = PatchBoundary, project onto feature
```

### Collapse

Apply the constraint matrix. Surviving vertex keeps the stricter constraint
and its patch / feature id.

### Flip

Forbidden on `FeatureEdge | PatchBoundary | ProtectedEdge`.

---

## GPU layout (SoA, not AoS)

Host and device use parallel arrays. No `HugeVertex`.

```text
VertexPosition[3][n]
VertexNormal[3][n]
VertexPatchId[n]
VertexConstraint[n]          uint8
VertexTargetLength[n]
VertexCurvature[n]           kmax = max(|k1|,|k2|)
VertexFeatureDistance[n]

FacePatchId[n]
FacePatchType[n]             uint8

EdgePatchLeft / EdgePatchRight
EdgeFlags                    boundary | sharp | protected
EdgeFeatureCurveId
```

Working precision on GPU: **float32**. CPU audit may use double. Existing
`cad_mesh/rxmesh` double path is the accuracy baseline, not the 10M-tri
memory budget.

Do not include CadMesh / VCG headers in `.cu` / `.cuh` files. The existing
RXMesh CMake already isolates that boundary.

---

## Sizing (locked formulas)

User knob for this ticket: `MaxGeometryError = ε` (model units).
Optional override: constant `hConst` (isotropic gate).

```text
hCurvature = C / sqrt(kmax + 1e-12)
  plane:     kmax = 0  →  hCurvature = hMax
  cylinder:  kmax = 1/R

hPatch:
  plane:     hMax
  cylinder:  sqrt(8 R ε)
  other:     hConst or hMax   (001 does not specialize cone/sphere)

hFeature = lerp(hFeatureEdge, hRegular, smoothstep(clamp(d / band, 0, 1)))
  band default = 4 * hRegular
  hFeatureEdge default = min(hRegular, 2ε) unless overridden

h(x) = clamp(min(hCurvature, hFeature, hPatch, hError), hMin, hMax)
hMin = ε
hMax = max(hConst, bboxDiagonal * 0.05)   if hConst unset
```

Split candidate:

```text
h = 0.5 * (h(v0) + h(v1))
candidate ⇔ length > (4/3) h
```

Collapse candidate:

```text
candidate ⇔ length < (4/5) h     # 0.8, not the CPU 0.65
```

`0.8` matches isotropic 4/5–4/3 bands. CPU NativeRemesher’s `0.65` stays
on the CPU path; do not mix the two.

---

## Collapse constraint matrix (locked, unit-tested)

Priority: `Locked > Corner > PatchBoundary/FeatureEdge > Surface > Free`.

| V0 | V1 | Result |
| --- | --- | --- |
| Free | Free | yes, Optimal |
| Surface | Surface, same patch | yes, Optimal (2D on patch) |
| Surface A | Surface B | **no** |
| Boundary | Surface | yes, **keep Boundary** (Surface → Boundary) |
| Surface | Boundary | yes, **keep Boundary** |
| Boundary | Boundary, same feature | yes, 1D optimal on feature |
| Boundary A | Boundary B | **no** |
| Corner | anything | yes only if the other collapses **onto Corner** |
| Locked | anything | yes only if the other collapses **onto Locked** |

Destination modes: `KeepV0 | KeepV1 | Optimal`.
001 Optimal = midpoint then project. Quadric optimal is later.

Also reject if any fail: topology link, normal flip, geometry error `> ε`,
triangle quality below `1e-5` (same floor as current RXMesh kernel).

---

## Operators and iteration (locked)

Do **not** put split / collapse / flip in one kernel.

```text
Iteration N
├── Update sizing field          (vertex kernel)
├── Classify edges               (all edges in 001; active set is 002)
├── Split:  score → cavity create → execute → project
├── Collapse: validate → cavity create → execute → project
├── Flip: objective → cavity create → execute
├── Tangential relaxation        (then project)
├── Error measurement
└── stop if converged or Nmax
```

Conflict resolution in 001: **RXMesh `CavityManager` + scheduler queue**.
No graph coloring. No custom hash-MIS. Active-set dirty rings are
**RXREMESH-002**.

Flip objective (001):

```text
E = wValence EValence + wShape EShape + wSizing ESizing
wNormal = 0 until analytic normals are on every face
accept ⇔ Eafter < Ebefore
```

Smoothing:

```text
interior:  tangent = d - n(n·d);  project to surface
boundary:  tangent = t(t·d) along feature;  project to feature
corner/locked:  no move
```

Projection (001):

```text
Surface  → plane or cylinder table[patchId]
Boundary → feature polyline lerp (polyline == current protected edges)
Freeform / unknown → **reject the operator** (no BVH in 001)
```

`IGeometryProjector` is a CPU test interface only. Device code uses a
`switch(patchType)` over SoA tables. No virtual calls on GPU.

---

## Internal gates (must pass in order)

Each gate is a mergeable checkpoint. Do not start the next until tests pass.

### 001.0 Import / attributes / neighbors

- OBJ/PLY in, compact mesh out
- SoA attributes round-trip
- one-ring query on a known mesh
- no CadMesh headers in CUDA TUs

### 001.1 Isotropic GPU remesh

- constant `h`
- Split / Collapse / Flip / tangential smooth
- no CAD
- topology `rx.validate()` after every iteration
- cube + plane grid fixtures (same spirit as `test_rxmesh_entry.py`)

### 001.2 Three constraints

- `PatchId` propagation
- `PatchBoundary` never flipped, never collapsed away
- `Locked` vertices stay put
- fixture: two patches sharing a straight seam

### 001.3 Plane + Cylinder projection

- new interior vertices land on the analytic surface within `1e-5 * R`
  (or `1e-5 * bbox` for planes)
- seam vertices stay on the shared generator (within `ε`)
- fixture: finite cylinder with two planar caps, sharp rims locked

### 001.4 Adaptive sizing

- constant still works (regression)
- cylinder `h ≈ sqrt(8 R ε)`
- feature-distance band visible in `VertexTargetLength`
- plane interior allowed to go to `hMax`

### 001.5 Metrics + scale

JSON report **required every run**:

```text
candidates / accepted / rejected  per operator
reject reasons                    topology, patch, feature, normal, quality, error
triangle quality                  mean, p05, min, Q = 4√3 A / (a²+b²+c²)
sizing error                      |L/h - 1|  mean and p95
geometry error                    sampled max distance to analytic or input
boundary audit                    missing / moved locked verts and boundary edges
timings                           setup, each phase, total
```

Benchmarks (same machine, vs CPU `NativeRemesher` isotropic **and** vs
`cad_mesh_rxmesh`): **100K / 1M / 5M / 10M** triangles. 001 records numbers;
it does **not** gate on a 60s slogan. Pass = no crash, valid topology,
constraints held, report written. Speedup is evidence, not a fail criterion
until 002.

Convergence (optional early stop, not required to match CPU):

```text
badSizingEdges < 0.5%
sampled max geometry error < ε
quality p05 > 0.35
```

---

## Proof bar (the only 001 success definition)

1. **Topology edits are reliable** — `validate()` true after every iteration
   on the fixtures plus one 1M-tri mesh.
2. **CAD boundaries are not washed out** — locked vertices move `< 1e-9`;
   patch-boundary edges remain a single closed (or input-open) chain;
   `PatchId` never jumps across the seam.
3. **Scale is real** — 1M / 5M / 10M complete with metrics. If GPU is not
   faster than CPU at 1M, that is a finding, not an excuse to skip the
   report.

Candidate-quality warning (not a fail): `accepted / candidates < 1%` on a
uniform plane means the classify kernel is too wide. Log it.

---

## File layout (create as gates land)

```text
experiments/rxmesh-remesh/
  README.md
  RXREMESH-001.md
  CMakeLists.txt
  include/cad_adaptive/
    Types.h
    SemanticMesh.h
    RemeshField.h
    RemeshPolicy.h
    GeometryProjector.h
    IRemeshBackend.h
  src/
    SemanticMesh.cpp
    RemeshField.cpp
    RemeshPolicy.cpp
    AnalyticProjector.cpp
    cpu/CpuRemeshBackend.cpp
    rxmesh/RxMeshBackend.cu
    rxmesh/kernels_split.cuh
    rxmesh/kernels_collapse.cuh
    rxmesh/kernels_flip.cuh
    rxmesh/kernels_smooth.cuh
  tests/
    test_collapse_matrix.cpp
    test_project_plane_cylinder.cpp
    test_sizing_field.cpp
    test_semantic_propagation.cpp
    test_cpu_isotropic.cpp
    test_rxmesh_roundtrip.py
    test_boundary_preservation.py
  fixtures/
    plane.obj
    cylinder.obj
    plane_cylinder_seam.obj
  tools/
    remesh_cli.cpp
```

`CpuRemeshBackend` is not optional. Policy, matrix, projection, and sizing
must fail in CPU tests before any CUDA debug.

---

## Public types (001)

```cpp
enum class PatchType : uint8_t {
  Unknown = 0, Plane, Cylinder, Cone, Sphere, Torus, Freeform
};

enum class VertexConstraint : uint8_t {
  Free = 0, Surface, FeatureEdge, PatchBoundary, Corner, Locked
};

struct CollapseDecision {
  bool allowed = false;
  enum class Dest { KeepV0, KeepV1, Optimal } dest = Dest::Optimal;
};
```

`PatchType` numeric values match `CadMesh::PatchSurfaceType`.

---

## NOT in scope (explicit)

- Editing `NativeRemesher`, `GenericSurfaceRemesher`, or `cad_mesh/rxmesh`
  kernels except read-only reference
- PAMO segmentation / fitting / snapshot integration (Phase 6)
- Cone, Sphere, Torus, fillet, freeform BVH projection
- Fitted FeatureCurve geometry (polyline edges are enough)
- Corner optimization
- Quadric collapse
- Hausdorff certification / global self-intersection
- Active-set / custom MIS scheduler / graph coloring
- `IRemeshBackend` as a plugin ABI; it is an in-tree C++ interface
- User-facing “target edge length” as the *only* control (constant `h` stays
  as the isotropic gate and debug override)
- Replacing production remesh entry points or `run_rxmesh.ps1`

---

## Implementation order for the next coding session

1. Headers + CPU `RemeshPolicy` + collapse-matrix tests
2. Analytic plane/cylinder projector tests (no CUDA)
3. Sizing-field tests (plane / cylinder / feature band)
4. CPU isotropic backend on `plane.obj`
5. RXMesh 001.0 round-trip
6. 001.1–001.5 in order

Do not start CUDA until step 3 is green.
