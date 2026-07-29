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
- **`--remesh-only`**: Run only the SDF remeshing stage, without simplification or safe projection.
- **`--remesh-resolution`**: Set the SDF grid resolution used by `--remesh-only`. Supported values are 64, 128, and 256; higher values preserve more detail but use more GPU memory.

For STL input and output:
```
python example.py --input ./model.stl --output ./examples/model_pamo.stl --ratio 0.001
```

For PLY input and output:
```
python example.py --input ./model.ply --output ./examples/model_pamo.ply --ratio 0.001
```

For remeshing without simplification:
```
python example.py --input ./model.stl --output ./examples/model_remeshed.stl --remesh-only --remesh-resolution 256
```

PLY meshes use the same remesh-only operation:
```
python example.py --input ./examples/geom_runner_sys.ply --output ./examples/geom_runner_sys_remeshed.ply --remesh-only --remesh-resolution 128
```

The remesh-only operation uses an SDF and Dual Marching Cubes. It is designed to produce a watertight remesh, so it may close holes or otherwise change the input topology.
PLY vertex colors and other custom attributes are not preserved; PaMO currently processes and exports mesh geometry only.

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
