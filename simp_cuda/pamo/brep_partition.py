"""Strict model-first recovery of B-rep-like STL surface patches."""

import colorsys
from pathlib import Path

import igl
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _fit_cylindrical_face_set(vertices, faces, radius_tolerance=1e-3):
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(crosses, axis=1)
    if np.any(lengths <= np.finfo(np.float64).eps):
        return None
    normals = crosses / lengths[:, None]
    _, eigenvectors = np.linalg.eigh(normals.T @ normals / len(normals))
    axis = eigenvectors[:, 0]
    normal_error = float(np.sqrt(np.mean((normals @ axis) ** 2)))
    if normal_error > 2e-2:
        return None

    vertex_ids = np.unique(faces)
    points = vertices[vertex_ids]
    origin = points.mean(axis=0)
    first_basis = normals[0] - axis * float(np.dot(normals[0], axis))
    basis_length = float(np.linalg.norm(first_basis))
    if basis_length <= np.finfo(np.float64).eps:
        return None
    first_basis /= basis_length
    second_basis = np.cross(axis, first_basis)
    offsets = points - origin
    projected = np.column_stack((offsets @ first_basis, offsets @ second_basis))
    system = np.column_stack(
        (2.0 * projected[:, 0], 2.0 * projected[:, 1], np.ones(len(projected)))
    )
    solution, _, _, _ = np.linalg.lstsq(
        system, np.sum(projected * projected, axis=1), rcond=None
    )
    center_2d = solution[:2]
    radii = np.linalg.norm(projected - center_2d, axis=1)
    radius = float(radii.mean())
    if radius <= 0.0 or float(radii.std() / radius) > float(radius_tolerance):
        return None
    axis_origin = (
        origin + center_2d[0] * first_basis + center_2d[1] * second_basis
    )
    centroid_offsets = triangles.mean(axis=1) - axis_origin
    axial = centroid_offsets @ axis
    radial = centroid_offsets - axial[:, None] * axis
    radial_lengths = np.linalg.norm(radial, axis=1)
    alignment = np.abs(np.sum(normals * radial, axis=1)) / np.maximum(
        radial_lengths, np.finfo(np.float64).eps
    )
    if float(np.percentile(alignment, 5.0)) < 0.98:
        return None
    return {
        "axis": axis,
        "origin": axis_origin,
        "radius": radius,
        "normal_error": normal_error,
    }


def build_brep_model_partitions(vertices, faces, minimum_cylinder_facets=3):
    """Recover only unambiguous connected planar/cylindrical STL patches."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    facets = [np.asarray(facet, dtype=np.int64) for facet in mesh.facets]
    if not facets:
        return None
    face_to_facet = np.full(len(faces), -1, dtype=np.int64)
    for facet_id, facet_faces in enumerate(facets):
        if np.any(face_to_facet[facet_faces] >= 0):
            return None
        face_to_facet[facet_faces] = facet_id
    if np.any(face_to_facet < 0):
        return None

    facet_areas = np.asarray(
        [float(mesh.area_faces[facet].sum()) for facet in facets],
        dtype=np.float64,
    )
    ordered_areas = np.sort(facet_areas)
    if len(ordered_areas) < 4 or np.any(ordered_areas <= 0.0):
        return None
    ratios = ordered_areas[1:] / ordered_areas[:-1]
    gap_index = int(np.argmax(ratios))
    if float(ratios[gap_index]) < 4.0:
        return None
    area_threshold = float(
        np.sqrt(ordered_areas[gap_index] * ordered_areas[gap_index + 1])
    )
    narrow_facets = facet_areas < area_threshold
    if int(np.count_nonzero(narrow_facets)) < int(minimum_cylinder_facets):
        return None

    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    facet_pairs = np.sort(face_to_facet[adjacency], axis=1)
    facet_pairs = np.unique(
        facet_pairs[facet_pairs[:, 0] != facet_pairs[:, 1]], axis=0
    )
    usable_pairs = facet_pairs[
        narrow_facets[facet_pairs[:, 0]] & narrow_facets[facet_pairs[:, 1]]
    ]
    graph = coo_matrix(
        (
            np.ones(len(usable_pairs) * 2, dtype=np.uint8),
            (
                np.concatenate((usable_pairs[:, 0], usable_pairs[:, 1])),
                np.concatenate((usable_pairs[:, 1], usable_pairs[:, 0])),
            ),
        ),
        shape=(len(facets), len(facets)),
    ).tocsr()
    _, facet_components = connected_components(graph, directed=False)

    cylinder_groups = []
    consumed = np.zeros(len(facets), dtype=bool)
    for component_id in np.unique(facet_components[narrow_facets]):
        facet_ids = np.flatnonzero(
            narrow_facets & (facet_components == int(component_id))
        )
        if len(facet_ids) < int(minimum_cylinder_facets):
            continue
        component_faces = np.concatenate([facets[index] for index in facet_ids])
        fit = _fit_cylindrical_face_set(vertices, faces[component_faces])
        if fit is None:
            continue
        cylinder_groups.append((component_faces, fit))
        consumed[facet_ids] = True
    if not cylinder_groups or np.any(narrow_facets & ~consumed):
        return None

    labels = np.full(len(faces), -1, dtype=np.int64)
    surface_types = []
    for facet_id, facet_faces in enumerate(facets):
        if consumed[facet_id]:
            continue
        labels[facet_faces] = len(surface_types)
        surface_types.append("plane")
    cylinder_models = []
    for component_faces, fit in cylinder_groups:
        labels[component_faces] = len(surface_types)
        surface_types.append("cylinder")
        cylinder_models.append(fit)
    if np.any(labels < 0):
        return None
    sizes = np.bincount(labels)
    return labels, {
        "patches": int(len(sizes)),
        "plane_patches": int(surface_types.count("plane")),
        "cylinder_patches": int(surface_types.count("cylinder")),
        "surface_types": surface_types,
        "cylinder_models": cylinder_models,
        "input_patch_face_counts": sizes.tolist(),
        "facet_area_threshold": area_threshold,
        "partition_method": "brep-model-first",
    }


def patch_boundary_edges(faces, labels):
    """Return mesh-boundary and cross-patch edges, each exactly once."""
    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    edge_records = {}
    for face_id, face in enumerate(faces):
        for first, second in (
            (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_records.setdefault(edge, []).append(int(labels[face_id]))
    boundaries = [
        edge for edge, memberships in edge_records.items()
        if len(memberships) != 2 or memberships[0] != memberships[1]
    ]
    return np.asarray(sorted(boundaries), dtype=np.int64).reshape(-1, 2)


def transfer_patch_labels(
    reference_vertices,
    reference_faces,
    reference_labels,
    output_vertices,
    output_faces,
):
    """Transfer patch ownership by closest source face at face centroids."""
    output_centroids = np.asarray(output_vertices, dtype=np.float64)[
        np.asarray(output_faces, dtype=np.int64)
    ].mean(axis=1)
    _, source_faces, _ = igl.point_mesh_squared_distance(
        output_centroids,
        np.asarray(reference_vertices, dtype=np.float64),
        np.asarray(reference_faces, dtype=np.int64),
    )
    return np.asarray(reference_labels, dtype=np.int64)[
        np.asarray(source_faces, dtype=np.int64)
    ]


def _patch_color(patch_id):
    hue = (0.08 + int(patch_id) * 0.6180339887498949) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
    return np.asarray((*np.rint(np.asarray(rgb) * 255.0), 255), dtype=np.uint8)


def write_patch_id_ply(path, vertices, faces, labels):
    """Write solid per-patch colors and integer IDs on vertices and faces."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    chunks = []
    face_chunks = []
    vertex_ids = []
    face_ids = []
    vertex_colors = []
    face_colors = []
    offset = 0
    for patch_id in np.unique(labels):
        selected = faces[labels == patch_id]
        used, inverse = np.unique(selected.reshape(-1), return_inverse=True)
        local_faces = inverse.reshape(-1, 3)
        color = _patch_color(patch_id)
        chunks.append(vertices[used])
        face_chunks.append(local_faces + offset)
        vertex_ids.append(np.full(len(used), patch_id, dtype=np.int32))
        face_ids.append(np.full(len(local_faces), patch_id, dtype=np.int32))
        vertex_colors.append(np.tile(color, (len(used), 1)))
        face_colors.append(np.tile(color, (len(local_faces), 1)))
        offset += len(used)
    output_vertices = np.vstack(chunks).astype("<f4", copy=False)
    output_faces = np.vstack(face_chunks).astype("<i4", copy=False)
    vertex_ids = np.concatenate(vertex_ids)
    face_ids = np.concatenate(face_ids)
    vertex_colors = np.vstack(vertex_colors)
    face_colors = np.vstack(face_colors)
    vertex_dtype = np.dtype([
        ("xyz", "<f4", (3,)), ("patch_id", "<i4"),
        ("rgba", "u1", (4,)),
    ])
    face_dtype = np.dtype([
        ("count", "u1"), ("indices", "<i4", (3,)),
        ("patch_id", "<i4"), ("rgba", "u1", (4,)),
    ])
    vertex_records = np.empty(len(output_vertices), dtype=vertex_dtype)
    vertex_records["xyz"] = output_vertices
    vertex_records["patch_id"] = vertex_ids
    vertex_records["rgba"] = vertex_colors
    face_records = np.empty(len(output_faces), dtype=face_dtype)
    face_records["count"] = 3
    face_records["indices"] = output_faces
    face_records["patch_id"] = face_ids
    face_records["rgba"] = face_colors
    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        "comment model-first B-rep patch IDs",
        f"element vertex {len(vertex_records)}",
        "property float x", "property float y", "property float z",
        "property int patch_id", "property uchar red",
        "property uchar green", "property uchar blue", "property uchar alpha",
        f"element face {len(face_records)}",
        "property list uchar int vertex_indices", "property int patch_id",
        "property uchar red", "property uchar green",
        "property uchar blue", "property uchar alpha", "end_header", "",
    ]).encode("ascii")
    with Path(path).open("wb") as stream:
        stream.write(header)
        vertex_records.tofile(stream)
        face_records.tofile(stream)
