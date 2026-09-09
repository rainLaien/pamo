"""Extract geometry-aware curved surface patches from an STL mesh.

Each non-planar semantic partition is written as an independent PLY.  A
combined, face-coloured PLY is also emitted so partition boundaries can be
inspected before opening individual patches.
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from simp_cuda.pamo.semantic_partition import build_semantic_partitions


def _patch_color(patch_id):
    hue = (0.08 + int(patch_id) * 0.6180339887498949) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
    return np.asarray((*np.rint(np.asarray(rgb) * 255.0), 255), dtype=np.uint8)


def _compact_patch(vertices, faces):
    vertex_ids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return vertices[vertex_ids], inverse.reshape(-1, 3)


def _connected_curved_patch_labels(faces, semantic_labels, curved_face_mask):
    """Split semantic labels into edge-connected face components."""
    mesh = trimesh.Trimesh(
        vertices=np.zeros((int(faces.max()) + 1, 3), dtype=np.float64),
        faces=faces,
        process=False,
    )
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    usable = adjacency[
        curved_face_mask[adjacency[:, 0]]
        & curved_face_mask[adjacency[:, 1]]
        & (semantic_labels[adjacency[:, 0]] == semantic_labels[adjacency[:, 1]])
    ]
    graph = coo_matrix(
        (
            np.ones(len(usable) * 2, dtype=np.uint8),
            (
                np.concatenate((usable[:, 0], usable[:, 1])),
                np.concatenate((usable[:, 1], usable[:, 0])),
            ),
        ),
        shape=(len(faces), len(faces)),
    ).tocsr()
    _, components = connected_components(graph, directed=False)
    curved_components = components[curved_face_mask]
    _, compact = np.unique(curved_components, return_inverse=True)
    patch_labels = np.full(len(faces), -1, dtype=np.int64)
    patch_labels[curved_face_mask] = compact
    return patch_labels


def _cylinder_membership(
    vertices,
    faces,
    radial_tolerance=1e-2,
    minimum_normal_alignment=0.95,
    maximum_axial_normal=8e-2,
):
    """Return faces satisfying one fitted cylinder equation and normal field."""
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    cross_lengths = np.linalg.norm(crosses, axis=1)
    if len(faces) < 8 or np.any(cross_lengths <= 0.0):
        return None
    normals = crosses / cross_lengths[:, None]

    _, eigenvectors = np.linalg.eigh(normals.T @ normals)
    axis = eigenvectors[:, 0]
    first_basis = normals[0] - axis * float(np.dot(normals[0], axis))
    basis_length = float(np.linalg.norm(first_basis))
    if basis_length <= np.finfo(np.float64).eps:
        return None
    first_basis /= basis_length
    second_basis = np.cross(axis, first_basis)

    vertex_ids = np.unique(faces)
    points = vertices[vertex_ids]
    origin = points.mean(axis=0)
    offsets = points - origin
    projected = np.column_stack(
        (offsets @ first_basis, offsets @ second_basis)
    )
    circle_system = np.column_stack(
        (2.0 * projected[:, 0], 2.0 * projected[:, 1], np.ones(len(projected)))
    )
    circle_rhs = np.einsum("ij,ij->i", projected, projected)
    solution, _, _, _ = np.linalg.lstsq(
        circle_system, circle_rhs, rcond=None
    )
    radius_squared = float(
        solution[2] + solution[0] ** 2 + solution[1] ** 2
    )
    if radius_squared <= 0.0:
        return None
    radius = float(np.sqrt(radius_squared))
    axis_origin = (
        origin
        + solution[0] * first_basis
        + solution[1] * second_basis
    )

    triangle_offsets = triangles - axis_origin
    triangle_axial = np.einsum("fvi,i->fv", triangle_offsets, axis)
    triangle_radial = (
        triangle_offsets - triangle_axial[:, :, None] * axis
    )
    triangle_radii = np.linalg.norm(triangle_radial, axis=2)
    face_radial_residual = np.max(
        np.abs(triangle_radii - radius) / radius, axis=1
    )

    centroid_offsets = triangles.mean(axis=1) - axis_origin
    centroid_axial = centroid_offsets @ axis
    centroid_radial = centroid_offsets - centroid_axial[:, None] * axis
    centroid_radii = np.linalg.norm(centroid_radial, axis=1)
    normal_alignment = np.abs(
        np.einsum("ij,ij->i", normals, centroid_radial)
    ) / np.maximum(centroid_radii, np.finfo(np.float64).eps)
    axial_normal = np.abs(normals @ axis)
    membership = (
        (face_radial_residual <= float(radial_tolerance))
        & (normal_alignment >= float(minimum_normal_alignment))
        & (axial_normal <= float(maximum_axial_normal))
    )
    return membership


def extract_curved_patches(
    input_path,
    output_directory,
    feature_angle_degrees=15.0,
    curvature_gradient_degrees=1.0,
    minimum_region_faces=20,
    cylinder_radial_tolerance=1e-2,
    cylinder_minimum_normal_alignment=0.95,
    cylinder_maximum_axial_normal=8e-2,
):
    input_path = Path(input_path).resolve()
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    loaded = trimesh.load_mesh(input_path, process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Input is not one triangle mesh: {input_path}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Input mesh is empty: {input_path}")

    result = build_semantic_partitions(
        vertices,
        faces,
        feature_angle_degrees=feature_angle_degrees,
        curvature_gradient_degrees=curvature_gradient_degrees,
        minimum_region_faces=minimum_region_faces,
    )
    region_count = int(result.stats["partitions"])
    curved_region_ids = np.asarray(
        [
            region
            for region, region_type in enumerate(result.region_types)
            if region_type != "plane"
        ],
        dtype=np.int64,
    )
    curved_region_mask = np.zeros(region_count, dtype=bool)
    curved_region_mask[curved_region_ids] = True
    curved_face_mask = curved_region_mask[result.labels]

    if not np.any(curved_face_mask):
        raise ValueError("No curved surface patch was detected.")

    patch_labels = _connected_curved_patch_labels(
        faces, result.labels, curved_face_mask
    )
    initial_patch_count = int(patch_labels.max()) + 1
    trimmed_faces = 0
    for patch in range(initial_patch_count):
        face_ids = np.flatnonzero(patch_labels == patch)
        source_region = int(result.labels[face_ids[0]])
        if result.region_types[source_region] != "cylinder":
            continue
        membership = _cylinder_membership(
            vertices,
            faces[face_ids],
            radial_tolerance=cylinder_radial_tolerance,
            minimum_normal_alignment=cylinder_minimum_normal_alignment,
            maximum_axial_normal=cylinder_maximum_axial_normal,
        )
        if membership is None:
            continue
        retained = int(np.count_nonzero(membership))
        # A failed global fit must not erase a patch.  A valid cylindrical core
        # needs enough support to define both its axis and circular section.
        if retained < 8 or retained < len(face_ids) * 0.1:
            continue
        rejected_ids = face_ids[~membership]
        curved_face_mask[rejected_ids] = False
        trimmed_faces += len(rejected_ids)

    patch_labels = _connected_curved_patch_labels(
        faces, result.labels, curved_face_mask
    )
    patch_count = int(patch_labels.max()) + 1
    patch_sizes = np.bincount(
        patch_labels[curved_face_mask], minlength=patch_count
    )
    combined_faces = faces[curved_face_mask]
    combined_labels = patch_labels[curved_face_mask]
    combined_vertices, compact_faces = _compact_patch(vertices, combined_faces)
    palette = np.vstack([_patch_color(patch) for patch in range(patch_count)])
    combined_mesh = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=compact_faces,
        process=False,
    )
    combined_mesh.visual.face_colors = palette[combined_labels]
    combined_path = output_directory / "_all_curved_patches.ply"
    combined_mesh.export(combined_path, file_type="ply")

    written = []
    patch_types = []
    for patch in range(patch_count):
        face_ids = np.flatnonzero(patch_labels == patch)
        patch_vertices, patch_faces = _compact_patch(vertices, faces[face_ids])
        source_region = int(result.labels[face_ids[0]])
        region_type = result.region_types[source_region]
        patch_types.append(region_type)
        patch_mesh = trimesh.Trimesh(
            vertices=patch_vertices,
            faces=patch_faces,
            process=False,
        )
        patch_mesh.visual.face_colors = np.tile(
            _patch_color(patch), (len(patch_faces), 1)
        )
        patch_path = output_directory / (
            f"patch_{patch:04d}_{region_type}_"
            f"{int(patch_sizes[patch])}_faces.ply"
        )
        patch_mesh.export(patch_path, file_type="ply")
        written.append(patch_path)

    type_counts = {
        kind: patch_types.count(kind) for kind in sorted(set(patch_types))
    }
    return {
        "input_faces": len(faces),
        "all_partitions": region_count,
        "curved_patches": len(written),
        "curved_faces": int(np.count_nonzero(curved_face_mask)),
        "cylinder_nonmembers_removed": int(trimmed_faces),
        "types": type_counts,
        "combined_path": combined_path,
        "patch_paths": written,
        "partition_stats": result.stats,
    }


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Remove planar regions from an STL and export every curved feature "
            "patch as a separate PLY."
        )
    )
    parser.add_argument("--input", required=True, help="Input STL path")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory receiving one PLY per curved patch",
    )
    parser.add_argument(
        "--feature-angle",
        type=float,
        default=15.0,
        help="Hard crease angle in degrees (default: 15)",
    )
    parser.add_argument(
        "--gradient-threshold",
        type=float,
        default=1.0,
        help="Curvature-change boundary threshold in degrees (default: 1)",
    )
    parser.add_argument(
        "--minimum-region-faces",
        type=int,
        default=20,
        help="Merge smaller soft-boundary fragments (default: 20)",
    )
    parser.add_argument(
        "--cylinder-radial-tolerance",
        type=float,
        default=1e-2,
        help="Maximum relative vertex-to-cylinder residual (default: 0.01)",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    if not 0.0 <= args.feature_angle < 180.0:
        raise ValueError("--feature-angle must be in [0, 180).")
    if args.gradient_threshold <= 0.0:
        raise ValueError("--gradient-threshold must be positive.")
    if args.minimum_region_faces < 1:
        raise ValueError("--minimum-region-faces must be positive.")
    if args.cylinder_radial_tolerance <= 0.0:
        raise ValueError("--cylinder-radial-tolerance must be positive.")

    stats = extract_curved_patches(
        args.input,
        args.output_dir,
        feature_angle_degrees=args.feature_angle,
        curvature_gradient_degrees=args.gradient_threshold,
        minimum_region_faces=args.minimum_region_faces,
        cylinder_radial_tolerance=args.cylinder_radial_tolerance,
    )
    print(
        "Curved patch extraction: {} / {} partition(s), {} / {} face(s); "
        "{} non-cylinder face(s) removed from fitted cylinders; types {}.".format(
            stats["curved_patches"],
            stats["all_partitions"],
            stats["curved_faces"],
            stats["input_faces"],
            stats["cylinder_nonmembers_removed"],
            stats["types"],
        )
    )
    print(f"Combined coloured PLY: {stats['combined_path']}")
    print(f"Individual patch directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
