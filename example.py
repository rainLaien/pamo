import torch
import torch.nn as nn
import trimesh
from pamo import PaMO, PaSP
import numpy as np
import time
import networkx as nx
import argparse
from pathlib import Path


SUPPORTED_INPUT_EXTENSIONS = {'.obj', '.stl', '.ply'}


def load_input_mesh(input_path):
    """Load an OBJ, STL, or PLY triangle mesh and clean its topology."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError("Input mesh does not exist: {}".format(input_path))
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            "Unsupported input format '{}'. Supported formats: {}".format(
                suffix or "<none>",
                ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS)),
            )
        )

    input_mesh = trimesh.load(input_path, force='mesh', process=False)
    if not isinstance(input_mesh, trimesh.Trimesh) or input_mesh.is_empty:
        if suffix == '.ply':
            raise ValueError(
                "PLY input must contain polygon faces; point-cloud-only PLY files "
                "are not supported: {}".format(input_path)
            )
        raise ValueError("Input does not contain a valid mesh: {}".format(input_path))

    if suffix in {'.stl', '.ply'}:
        verts_before = len(input_mesh.vertices)
        faces_before = len(input_mesh.faces)

        # STL files commonly repeat all three vertices for every triangle.
        # PLY files may also contain unwelded or duplicate geometry.
        # Weld vertices before PaMO builds edge adjacency.
        input_mesh.merge_vertices()
        input_mesh.update_faces(input_mesh.unique_faces())
        input_mesh.update_faces(input_mesh.nondegenerate_faces())
        input_mesh.remove_unreferenced_vertices()
        input_mesh.fix_normals(multibody=True)

        print(
            "{} cleanup: verts {} -> {}, faces {} -> {}".format(
                suffix[1:].upper(),
                verts_before,
                len(input_mesh.vertices),
                faces_before,
                len(input_mesh.faces),
            )
        )

    if len(input_mesh.vertices) == 0 or len(input_mesh.faces) == 0:
        raise ValueError("Input mesh is empty after cleanup: {}".format(input_path))
    if input_mesh.faces.ndim != 2 or input_mesh.faces.shape[1] != 3:
        raise ValueError("PaMO requires a triangular mesh: {}".format(input_path))
    if not np.isfinite(input_mesh.vertices).all():
        raise ValueError("Input mesh contains non-finite vertex coordinates: {}".format(input_path))

    return input_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='./mesh/BirdHouse_B019SXLRJ2_MetalLeafRoofGreenWalls_TU.obj',
        help="Input triangle mesh in OBJ, STL, or PLY format",
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='./BirdHouse_pamo.obj',
        help="Output mesh path; use .ply to export PLY",
    )
    parser.add_argument('-r', '--ratio', type=float, default=0.1)
    parser.add_argument('-mv', '--min-vertex', type=int, default=0)
    parser.add_argument('--disable_stage1', action='store_true', help="Disable remeshing")
    parser.add_argument('--disable_stage3', action='store_true', help="Disable safe projection")
    parser.add_argument(
        '--remesh-only',
        action='store_true',
        help="Run only SDF remeshing; skip simplification and safe projection",
    )
    parser.add_argument(
        '--remesh-resolution',
        type=int,
        choices=(64, 128, 256),
        default=256,
        help="SDF grid resolution used with --remesh-only (default: 256)",
    )
    args = parser.parse_args()

    if args.remesh_only and args.disable_stage1:
        parser.error("--remesh-only cannot be combined with --disable_stage1")


    input_mesh = load_input_mesh(args.input)
    print("# of input verts : {}".format(len(input_mesh.vertices)))
    print("# of input faces : {}".format(len(input_mesh.faces)))

    points = torch.from_numpy(input_mesh.vertices).float().cuda()
    triangles = torch.from_numpy(input_mesh.faces).int().cuda()

    pamo = PaMO(
        input_mesh,
        use_stage1=True if args.remesh_only else not args.disable_stage1,
        use_stage3=False if args.remesh_only else not args.disable_stage3,
    )
    start = time.time()
    if args.remesh_only:
        verts, faces = pamo.remesh_only(
            points,
            triangles,
            resolution=args.remesh_resolution,
        )
    else:
        verts, faces = pamo.run(
            points,
            triangles,
            min_verts=args.min_vertex,
            ratio=args.ratio,
        )
    end = time.time()
    print("Total time: ", end-start)
    
    output_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    print("# of output verts : {}".format(len(output_mesh.vertices)))
    print("# of output faces : {}".format(len(output_mesh.faces)))
    output_mesh.export(args.output)

    # # test PaSP
    # pasp = PaSP()
    # start = time.time()
    # verts, faces = pasp.run(torch.from_numpy(input_mesh.vertices).float().cuda(), torch.from_numpy(input_mesh.faces).int().cuda(), iter=10000, threshold=0.001)
    # end = time.time()
    # print("PaSP time: ", end-start)
    
    # verts = verts.cpu().numpy()
    # faces = faces.cpu().numpy()
    # output_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # output_mesh.export('./crab_pasp.obj')


if __name__ == '__main__':
    main()
