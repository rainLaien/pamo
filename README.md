# PaMO: Parallel Mesh Optimization for Intersection-Free Low-Poly Modeling on the GPU [PG2025]

*Seonghun Oh\*, Xiaodi Yuan\*, Xinyue Wei\*, Ruoxi Shi, Fanbo Xiang, Minghua Liu, Hao Su*
<br>[[Project Page]](https://seonghunn.github.io/pamo/) [[Paper]](https://arxiv.org/abs/2509.05595)

## Intro

We present a novel GPU-based mesh optimization pipeline with three core components:
1. Parallel remeshing: Converts arbitrary meshes into watertight, manifold, and intersection-free meshes while improving triangle quality.
2. Robust parallel simplification: Reduces mesh complexity with guaranteed intersection-free results.
3. Optimization-based safe projection: Realigns the simplified mesh to the original input, eliminating surface shifts from remeshing and restoring sharp features.

Our approach is highly efficient, simplifying a 2-million-face mesh to 2k triangles in just 3 seconds on an RTX 4090.


![teaser](teaser.png)
Left: Reducing the 2M-face “crab” to 0.1% in 2.29s. Right: Reducing the 7M-face “dragon” to 0.1% in 5.32s. (Only the output
meshes are shown above; qem [1], rolopm [2]).

## Installation
### Option 1: Docker environment
```
docker run --name pamo -i -t --gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility sarahwei0210/pamo:0.0.2 /bin/bash
```
PaMO environment is ready under `/workspace/pamo`

### Option 2: Install by anaconda (may take ~5min)
```
git clone --recurse-submodules https://github.com/SarahWeiii/pamo.git
conda env create -f env.yaml
conda activate pamo
bash setup.sh
```

Planar regions with holes use the `triangle` constrained-triangulation wheel
listed in `env.yaml`. In an existing Windows virtual environment, install it
with `python -m pip install "triangle>=20250106"`.

## Demo

```
bash demo.sh
```
We offer three meshes stored under `./mesh` folder (from [DTC dataset](https://ai.meta.com/blog/digital-twin-catalog-3d-reconstruction-shopify-reality-labs-research/)) for the demo. The results will be saved under `./examples` folder.

## Example
```
python example.py --input INPUT_DIR --output OUTPUT_DIR --ratio 0.001
```

- **`--input`**: Specify the path to the input mesh file. OBJ, STL, and PLY triangle meshes are supported. STL and PLY inputs are automatically cleaned by welding duplicate vertices and removing duplicate or degenerate faces. Point-cloud-only PLY files are not supported. If not provided, it defaults to `./mesh/crab.obj`.
- **`--output`**: Specify the output mesh path. Use a `.ply` suffix to export PLY geometry.
- **`--ratio`**: Set the simplification ratio to control the target reduction in the number of triangles. For example, `--ratio 0.001` (default) means reducing the number of triangles to 0.1% of the original.
- **`--min-vertex`**: Add this flag to constrain the minimum number of vertices after simplification, default=0.
- **`--disable_stage1`**: Add this flag to skip the remeshing process (stage 1), default=false.
- **`--disable_stage3`**: Add this flag to skip the safe projection process (stage 3), default=false.
- **`--remesh-only`**: Run only the SDF remeshing stage, without simplification or safe projection. Valid closed inputs use the original signed `SDF=0` surface by default.
- **`--feature-remesh`**: Run SDF remeshing followed by safe projection back toward the original mesh. This closes the surface without simplification and preserves planes, sharp edges, and other geometric features better than `--remesh-only`.
- **`--feature-optimize`**: Run the joint feature-preserving quality pipeline: SDF quality relocation, safe projection, optional feature-chain densification, corner/curve-constrained relocation, and quality-driven flips of non-feature edges.
- **`--surface-sample-remesh`**: Bypass SDF topology. Sample points directly on original triangles with CUDA area sampling and radius filtering, locally retriangulate source faces, collapse short edges only inside a smooth patch, bisect long edges, and run conflict-free CUDA quality flips. Detected feature-chain vertices are explicitly locked. High-quality source faces are refinement-only by default, and curved neighborhoods are protected by source-normal and source-plane deviation checks.
- **`--remesh-resolution`**: Set the SDF grid resolution used by the remesh modes. Supported values are 64, 128, and 256; higher values preserve more detail but use more GPU memory.
- **`--sdf-mode`**: Select `auto`, `exact`, or `repair`, default=`auto`. `auto` uses the original signed zero surface for watertight, consistently wound inputs and explicitly falls back to a repair envelope otherwise. `exact` rejects open or invalid inputs. `repair` extracts a 0.9-voxel unsigned-distance envelope.
- **`--projection-iterations`**: Set the number of safe-projection iterations used by `--feature-remesh`, default=5.
- **`--feature-edge-target-length`**: Enable feature-edge-only densification and set its maximum target edge length in the input mesh's coordinate units.
- **`--feature-edge-angle`**: Detect input feature edges whose dihedral angle is at least this value, default=45 degrees. Input boundary edges are also treated as features.
- **`--feature-edges`**: Optionally provide a text file containing explicit input edge vertex-index pairs instead of automatic angle detection.
- **`--feature-edge-match-tolerance`**: Set the maximum distance used to match SDF-remeshed edges to original feature curves. The default is three SDF voxels.
- **`--feature-edge-max-splits`**: Set a safety limit on inserted feature-edge vertices, default=100000.
- **`--sdf-optimize`**: Run SDF remeshing followed by topology-preserving tangential mesh optimization. It changes vertex positions only: no simplification, edge collapse, face-count change, or connectivity change.
- **`--sdf-optimize-iterations`**: Set the number of tangential optimization iterations, default=20.
- **`--sdf-smoothing-step`**: Set the tangential relocation step in `(0, 1]`, default=0.2.
- **`--sdf-projection-steps`**: Set the number of SDF Newton projection steps after each relocation, default=3.
- **`--sdf-feature-angle`**: Lock vertices on edges of the extracted SDF mesh sharper than this angle, default=45 degrees.
- **`--feature-quality-iterations`**: Set original-surface constrained relocation iterations used by `--feature-optimize`, default=5.
- **`--feature-quality-step`**: Set the feature-constrained relocation step in `(0, 1]`, default=0.2.
- **`--feature-flip-passes`**: Set the number of quality-driven non-feature edge-flip passes, default=2.
- **`--surface-sample-count`**: Requested candidate-point budget for original-surface sampling. Dense input meshes should use a conservative value; adding too many points over-refines the mesh.
- **`--surface-poisson-radius`**: Optional world-space minimum sample spacing. By default it is derived from surface area and `--surface-sample-count`.
- **`--surface-flip-passes`**: Number of conflict-free CUDA edge-flip batches for sampled remeshing, default=5.
- **`--surface-max-edge-ratio`**: Hard output edge-length bound divided by the Poisson radius, default=2.0. The two-radius default avoids over-refining narrow walls while still eliminating extreme long edges.
- **`--surface-min-edge-ratio`**: Short-edge collapse threshold divided by the Poisson radius, default=0.5. Collapses must remain inside one smooth patch and improve local quality.
- **`--surface-split-passes`**: Maximum conflict-free CUDA long-edge split batches, default=64.
- **`--surface-collapse-passes`**: Maximum feature-safe CUDA short-edge collapse batches before refinement, default=24; a smaller cleanup phase also runs after splitting.
- **`--surface-protect-source-quality`**: Make original triangles at or above this normalized quality refinement-only: they may be split, but their source lineage cannot be collapsed, flipped, or relaxed. The default is `0.8`; use `0` to disable this protection.
- **`--surface-max-normal-deviation`**: Maximum source-normal cone half-angle, in degrees, allowed in a collapse one-ring. The default is `5`; smaller values preserve rounded regions more aggressively.
- **`--surface-max-deviation-ratio`**: Maximum source-plane deviation introduced by collapse or flip, divided by the Poisson radius. The default is `0.05`; smaller values prevent flat chords across rounded regions more aggressively.
- **`--surface-min-collapse-quality`**: Also make edges adjacent to triangles below this normalized quality eligible for controlled collapse, even when the edge is not shorter than the minimum edge ratio. The default is `0.25`; use `0` to keep short-edge-only behavior.
- **`--surface-coplanar-angle`**: Treat adjacent source faces within this normal angle as one coplanar patch for source-edge refinement. Internal coplanar edges are no longer subdivided as hard source boundaries; the default is `1` degree.
- **`--original-constrained-remesh`**: Preserve sharp, boundary, and non-manifold input edges as hard constraint chains. After conforming longest-edge refinement, quality-improving flips remove non-feature seams inside coplanar patches.
- **`--constraint-feature-angle`**: Mark every original manifold edge whose adjacent-face dihedral is strictly greater than this angle as a hard feature, default=5 degrees.
- **`--constraint-max-edge-length`**: Globally bisect longest edges until every output edge satisfies this world-space length bound. When omitted, the limit is `10% of the bounding-box diagonal`. This predictable scale-based default avoids excessive refinement on large, already dense meshes.
- **`--constraint-max-splits`**: Optional safety limit for longest-edge bisection. When omitted, the edge-only binary-bisection estimate receives a 4x conformity margin with a minimum budget of 100000.
- **`--constraint-flip-passes`**: Number of quality-driven coplanar non-feature edge-flip passes after refinement, default=8. Hard feature chains are never flipped.
- **`--constraint-flip-minimum-valence`**: Restrict coplanar flips to edges touching unusually high-valence vertices. A value such as `12` efficiently removes planar center-fan triangulations without scanning every edge for expensive quality tests.
- **`--constraint-planar-fan-minimum-valence`**: Replace a convex planar center fan at or above this valence with a uniform local Delaunay triangulation while retaining its boundary edges. A value such as `30` handles dense circle-center fans directly.
- **`--constraint-planar-annulus-minimum-faces`**: Uniformly retriangulate planar facets containing holes while retaining every outer and inner boundary edge. A value such as `20` replaces direct inner-to-outer circle bridges with a graded interior triangulation.
- **`--constraint-quality-iterations`**: Number of feature-safe tangential relocation iterations after constrained refinement, default=20.
- **`--constraint-quality-step`**: Tangential relocation step for constrained quality optimization, default=0.4.
- **`--constraint-quality-flip-passes`**: Number of feature-safe global quality edge-flip passes, default=12. The maximum edge-length bound remains enforced.

For STL input and output:
```
python example.py --input ./model.stl --output ./examples/model_pamo.stl --ratio 0.001
```

For PLY input and output:
```
python example.py --input ./model.ply --output ./examples/model_pamo.ply --ratio 0.001
```

For feature-safe original-surface CUDA sampling without SDF topology:
```
python example.py \
  --input ./model.stl \
  --output ./examples/model_surface_sampled.stl \
  --surface-sample-remesh \
  --surface-sample-count 5000 \
  --feature-edge-angle 30 \
  --surface-max-edge-ratio 2.0 \
  --surface-min-edge-ratio 0.5 \
  --surface-protect-source-quality 0.8 \
  --surface-max-normal-deviation 5 \
  --surface-max-deviation-ratio 0.05 \
  --surface-flip-passes 32
```

The directly executable validation command is:
```
bash ./examples/test_surface_sample_remesh.sh
```

For remeshing without simplification:
```
python example.py --input ./model.stl --output ./examples/model_remeshed.stl --remesh-only --remesh-resolution 256 --sdf-mode exact
```

PLY meshes use the same remesh-only operation:
```
python example.py --input ./examples/geom_runner_sys.ply --output ./examples/geom_runner_sys_remeshed.ply --remesh-only --remesh-resolution 128
```

The remesh-only operation computes unsigned distance on the GPU and uses a
fast-winding classification for a reliable inside-negative sign. In `exact`
mode it extracts the original `SDF=0` surface: there is no hidden
`0.9 / resolution` isovalue shift. DMC vertices are projected back to the
trilinearly interpolated zero set, and the DMC `(R - 1)` grid coordinates are
converted to the SDF cell-center coordinates before returning world-space
vertices.

`repair` has deliberately different semantics. It extracts the level set at
an unsigned distance of 0.9 voxel, producing a closed offset envelope for
open or invalid inputs. This can close holes and change topology, and it is
not expected to coincide with the original surface. On an already closed
mesh it may produce both an inner and an outer envelope, so use `exact` when
surface fidelity is the goal.

For the repository test STL:
```
python example.py \
  --input ./examples/111.stl \
  --output ./examples/111_remeshed.stl \
  --remesh-only \
  --remesh-resolution 128 \
  --sdf-mode exact
```

PLY vertex colors and other custom attributes are not preserved; PaMO currently processes and exports mesh geometry only.

For watertight SDF remeshing with better feature preservation:
```
python example.py --input ./model.stl --output ./examples/model_feature_remeshed.stl --feature-remesh --remesh-resolution 256 --projection-iterations 5
```

To automatically detect sharp edges and only densify those feature edges:
```
python example.py \
  --input ./model.stl \
  --output ./examples/model_feature_dense.stl \
  --feature-remesh \
  --remesh-resolution 128 \
  --projection-iterations 5 \
  --feature-edge-angle 45 \
  --feature-edge-target-length 0.5
```

The target length uses the same coordinate units as the input mesh. During
feature-edge densification, matched edges can only be split: they are never
collapsed, flipped, or smoothed away from the original feature curve. New
vertices are projected onto the exact nearest original feature segment; a
midpoint spatial index accelerates the query without imposing a fixed-candidate
approximation.

To specify feature edges explicitly, use a UTF-8 text file with one zero-based
vertex-index pair per line. Commas, whitespace, blank lines, and `#` comments
are accepted:
```
# feature_edges.txt
12 13
13, 14
14 15
```

Then run:
```
python example.py \
  --input ./model.ply \
  --output ./examples/model_feature_dense.ply \
  --feature-remesh \
  --remesh-resolution 128 \
  --feature-edges ./feature_edges.txt \
  --feature-edge-target-length 0.5
```

Explicit indices refer to the cleaned input mesh seen by PaMO. Because STL
files do not store shared vertex indices and are welded during loading,
automatic angle detection is normally preferable for STL.

Feature remeshing keeps the SDF-generated connectivity and applies PaMO's
intersection-aware safe projection toward the input surface. It preserves
existing planar caps and sharp edges better than SDF remeshing alone. Regions
newly filled by the SDF have no corresponding surface in the input, so their
exact shape cannot be recovered from projection alone.

The first feature-edge densification implementation is conservative: it only
splits remeshed edges that already form an explicit sharp edge near an original
feature curve. It does not cut a feature curve through the interior of an
existing output triangle. This preserves the SDF mesh topology, but a feature
which is completely blurred away by a low-resolution SDF cannot be recreated
by densification alone; increase `--remesh-resolution` in that case.

For joint feature preservation and triangle-quality optimization:
```
python example.py \
  --input ./model.stl \
  --output ./examples/model_feature_optimized.stl \
  --feature-optimize \
  --remesh-resolution 128 \
  --sdf-mode exact \
  --feature-edge-angle 30 \
  --feature-edge-target-length 8 \
  --sdf-optimize-iterations 5 \
  --feature-quality-iterations 3 \
  --feature-flip-passes 1
```

The joint mode fixes mapped feature corners, permits feature-chain vertices to
move only on original feature segments, and projects ordinary relocated
vertices back to the original triangle surface. Edge flips are accepted only
for non-feature manifold edges when the local minimum triangle quality
improves without reversing or degenerating either triangle. This first version
keeps the base face count stable except for explicitly requested feature-edge
splits; it does not yet perform edge collapse. It requires an exact-compatible
watertight input because projecting a repair envelope to an incomplete
original surface would invalidate the repaired regions.

For direct SDF-constrained quality optimization without simplification:
```
python example.py \
  --input ./model.stl \
  --output ./examples/model_sdf_optimized.stl \
  --sdf-optimize \
  --remesh-resolution 128 \
  --sdf-optimize-iterations 20 \
  --sdf-smoothing-step 0.2 \
  --sdf-projection-steps 3
```

This mode keeps the SDF-extracted faces and connectivity fixed. Each iteration
moves vertices tangentially toward their one-ring neighbor centroid, then uses
the sampled SDF gradient to project them back to the zero level set. Its
backtracking line search accepts a step only when a scale-invariant combination
of edge-length variation and triangle shape improves, the SDF residual remains
bounded, and no triangle flips or degenerates. Sharp, boundary, and
non-manifold vertices are locked. The mode improves triangle distribution
while preserving the SDF surface, including regions that did not exist in an
open input mesh.

Only features already represented by the sampled SDF can be preserved by this
mode. A feature narrower than a voxel cannot be reconstructed by SDF
optimization; use a higher `--remesh-resolution` or combine the result with
the original-mesh feature projection when strict planes or sharp curves are
required.

For strict longest-edge refinement constrained by the original STL:
```
python example.py \
  --input ./model.stl \
  --output ./examples/model_original_constrained.stl \
  --original-constrained-remesh \
  --sdf-mode exact \
  --constraint-feature-angle 5 \
  --constraint-max-edge-length 0.5
```

The strict implementation deliberately does not use DMC connectivity. DMC
constructs a new triangulation and therefore cannot guarantee that an original
feature edge remains an explicit edge; snapping nearby DMC vertices only gives
an approximation. With exact `SDF=0` semantics, the original piecewise-linear
surface is already the zero surface, so this mode retains its connectivity as
the constraint skeleton and refines it directly.

Every hard feature edge is preserved geometrically. If it is longer than the
global length bound, it is replaced only by two collinear child edges and its
lineage is validated after refinement. Long non-feature edges are bisected in
the same conforming operation, then quality-improving flips replace redundant
diagonals inside coplanar patches. There is no hard-edge collapse, smoothing,
or nearest-edge matching.

Because an unsigned-distance repair envelope has different topology and is
offset from the original surface, strict edge identity and repair-envelope
closure cannot both be guaranteed. `--original-constrained-remesh` therefore
requires a watertight, consistently wound input and exact mode. Open inputs
must first be repaired into a valid closed mesh before strict refinement.

### Example smoke-test scripts

The scripts in `examples/` use the watertight `222.stl` input and write results
to `examples/test_outputs/`. They require no arguments and are intended as
direct functional smoke tests:
```
bash examples/test_remesh_only.sh
bash examples/test_feature_remesh.sh
bash examples/test_feature_optimize.sh
bash examples/test_sdf_optimize.sh
bash examples/test_original_constrained.sh
bash examples/test_full_pipeline.sh
```

Run every mode in sequence with:
```
bash examples/test_all.sh
```

## Usage
### Import
```
from pamo import PaMO
```
### Constructor
Creates an instance of the `PAMO` class using the input mesh data.
```
pamo = PaMO(input_mesh, use_stage1=True, use_stage3=True)
```
### Run
Performs mesh optimization to reduce the complexity of the mesh while preserving essential details according to specified parameters.
```
pamo.run(points, triangles, ratio, tolerance=4, threshold=1e-3, iter=100000)
```

### Remesh Only
Runs only the SDF and Dual Marching Cubes remeshing stage.
```
pamo = PaMO(input_mesh, use_stage1=True, use_stage3=False)
verts, faces = pamo.remesh_only(points, triangles, resolution=256)
```

### Feature Remesh
Runs SDF remeshing followed by safe projection, without simplification.
```
pamo = PaMO(input_mesh, use_stage1=True, use_stage3=True)
verts, faces = pamo.feature_remesh(
    points,
    triangles,
    resolution=256,
    projection_iterations=5,
    feature_edge_target_length=0.5,
    feature_edge_angle=45.0,
)
```

### Feature Optimize
Runs the combined feature-preserving quality pipeline.
```
pamo = PaMO(input_mesh, use_stage1=True, use_stage3=True)
verts, faces = pamo.feature_optimize(
    points,
    triangles,
    resolution=128,
    projection_iterations=2,
    feature_edge_target_length=8.0,
    feature_edge_angle=30.0,
    sdf_iterations=5,
    quality_iterations=3,
    flip_passes=1,
    sdf_mode="exact",
)
```

### SDF Optimize
Runs SDF remeshing followed by topology-preserving tangential optimization.
```
pamo = PaMO(input_mesh, use_stage1=True, use_stage3=False)
verts, faces = pamo.sdf_optimize(
    points,
    triangles,
    resolution=128,
    iterations=20,
    smoothing_step=0.2,
    projection_steps=3,
    feature_angle=45.0,
)
```

### Original-constrained Remesh
Runs strict original-connectivity refinement with hard feature lineages.
```
pamo = PaMO(input_mesh, use_stage1=True, use_stage3=False)
verts, faces = pamo.original_constrained_remesh(
    points,
    triangles,
    sdf_mode="exact",
    feature_angle=5.0,
    max_edge_length=0.5,
)
```

#### Parameters
**points** (`float Tensor`): Vertices of the mesh. A tensor of floating-point numbers representing the 3D coordinates of each vertex.

**triangles** (`int Tensor`): Faces of the mesh. A tensor of integers where each row represents a triangle in the mesh defined by indices into the points array.

**ratio** (`float`): Decimation ratio specifying the target reduction in the number of triangles.

**use_stage1** (`bool`, *default = True*): Whether to use a remeshing (stage 1) before simplification.

**use_stage3** (`bool`, *default = True*): Whether to use a safe projection (stage 3) after simplification.

**tolerance** (`int`, *default = 4*): Defines the number of iterations to run without edge collapses before stopping, accumulating invalid edges that do not qualify for collapsing. Lower values quicken termination, while higher values allow more iterations for potential optimization

## References
[1] Jiang, Zhongshi, et al. "Declarative Specification for Unstructured Mesh Editing Algorithms." ACM Trans. Graph. 41.6 (2022): 251-1.

[2] Chen, Zhen, et al. "Robust low-poly meshing for general 3d models." ACM Transactions on Graphics (TOG) 42.4 (2023): 1-20.

## Cite
```
@inproceedings{oh2025pamo,
  title={PaMO: Parallel Mesh Optimization for Intersection-Free Low-Poly Modeling on the GPU},
  author={Oh, Seonghun and Yuan, Xiaodi and Wei, Xinyue and Shi, Ruoxi and Xiang, Fanbo and Liu, Minghua and Su, Hao},
  booktitle={Computer Graphics Forum},
  pages={e70267},
  year={2025},
  organization={Wiley Online Library}
}
```
