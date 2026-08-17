import numpy as np

from example import load_input_mesh
from pamo.original_constrained import (
    _build_edge_faces,
    classify_nonplanar_face_regions,
    classify_planar_face_regions,
    detect_hard_constraint_edges,
)
from pamo.feature_optimize import _triangle_quality_values


mesh = load_input_mesh("examples/222.stl")
hard_edges = detect_hard_constraint_edges(mesh, feature_angle_degrees=15.0)
planar_regions, planar_stats = classify_planar_face_regions(
    mesh.vertices,
    mesh.faces,
    protected_edges=hard_edges,
    minimum_faces=20,
    maximum_normal_angle_degrees=0.5,
    maximum_plane_distance_ratio=1e-5,
)
nonplanar_regions, nonplanar_stats = classify_nonplanar_face_regions(
    mesh.vertices,
    mesh.faces,
    planar_regions,
    protected_edges=hard_edges,
    minimum_faces=4,
    axial_normal_tolerance=2e-2,
    radius_tolerance=1e-2,
)
print("PLANAR", planar_stats, flush=True)
vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.faces)
triangles = vertices[faces]
crosses = np.cross(
    triangles[:, 1] - triangles[:, 0],
    triangles[:, 2] - triangles[:, 0],
)
face_areas = 0.5 * np.linalg.norm(crosses, axis=1)
for rank, region in enumerate(
    sorted(planar_regions, key=lambda item: -float(face_areas[item].sum()))[:12]
):
    region_cross = crosses[region].sum(axis=0)
    region_normal = region_cross / max(np.linalg.norm(region_cross), 1e-30)
    region_vertices = np.unique(faces[region].reshape(-1))
    points = vertices[region_vertices]
    print(
        "PLANE_RANK", rank,
        "faces", len(region),
        "area", float(face_areas[region].sum()),
        "normal", region_normal.tolist(),
        "centroid", points.mean(axis=0).tolist(),
        "bounds", points.min(axis=0).tolist(), points.max(axis=0).tolist(),
        flush=True,
    )
selected_pair = sorted(
    planar_regions, key=lambda item: -float(face_areas[item].sum())
)[:2]
selected_labels = np.full(len(faces), -1, dtype=np.int8)
for label, region in enumerate(selected_pair):
    selected_labels[region] = label
edge_faces_for_pair = _build_edge_faces(faces)
hard_edge_set = {tuple(map(int, edge)) for edge in np.asarray(hard_edges)}
pair_boundary = {"hard": [], "smooth": []}
for edge, memberships in edge_faces_for_pair.items():
    labels = {int(selected_labels[index]) for index in memberships}
    if not any(label >= 0 for label in labels) or len(labels) == 1:
        continue
    length = float(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))
    if length <= 10.0:
        continue
    key = "hard" if edge in hard_edge_set else "smooth"
    pair_boundary[key].append(length)
print(
    "PAIR_BOUNDARY",
    {
        key: {
            "count": len(lengths),
            "max": max(lengths) if lengths else 0.0,
            "splits": sum(int(np.ceil(np.log2(length / 10.0))) for length in lengths),
        }
        for key, lengths in pair_boundary.items()
    },
    flush=True,
)
print("NONPLANAR", nonplanar_stats, flush=True)
extrusion_sizes = np.asarray(
    [len(region["faces"]) for region in nonplanar_regions["extrusions"]],
    dtype=np.int64,
)
print(
    "EXTRUSION_SIZES",
    {
        "at_least_20": int(np.count_nonzero(extrusion_sizes >= 20)),
        "at_least_50": int(np.count_nonzero(extrusion_sizes >= 50)),
        "at_least_80": int(np.count_nonzero(extrusion_sizes >= 80)),
        "at_least_100": int(np.count_nonzero(extrusion_sizes >= 100)),
        "p50": float(np.percentile(extrusion_sizes, 50.0)),
        "p95": float(np.percentile(extrusion_sizes, 95.0)),
        "max": int(extrusion_sizes.max()),
    },
    flush=True,
)
eligible_counts = {0.05: [0, 0], 0.01: [0, 0], 0.005: [0, 0], 0.001: [0, 0]}
for region in nonplanar_regions["extrusions"]:
    if len(region["faces"]) < 80:
        continue
    region_faces = np.asarray(mesh.faces)[region["faces"]]
    quality = _triangle_quality_values(np.asarray(mesh.vertices), region_faces)
    region_triangles = np.asarray(mesh.vertices)[region_faces]
    maximum_edge = float(
        np.linalg.norm(
            region_triangles[:, (1, 2, 0)]
            - region_triangles[:, (0, 1, 2)],
            axis=2,
        ).max()
    )
    quality_p5 = float(np.percentile(quality, 5.0))
    for threshold, counts in eligible_counts.items():
        if quality_p5 < threshold and maximum_edge > 10.0:
            counts[0] += 1
            counts[1] += len(region["faces"])
print("ELIGIBLE_80", eligible_counts, flush=True)
labels = np.full(len(mesh.faces), -1, dtype=np.int64)
next_label = 0
for region in planar_regions:
    labels[region] = next_label
    next_label += 1
for region in nonplanar_regions["extrusions"]:
    if len(region["faces"]) < 80:
        continue
    region_faces = np.asarray(mesh.faces)[region["faces"]]
    quality = _triangle_quality_values(np.asarray(mesh.vertices), region_faces)
    region_triangles = np.asarray(mesh.vertices)[region_faces]
    maximum_edge = float(
        np.linalg.norm(
            region_triangles[:, (1, 2, 0)]
            - region_triangles[:, (0, 1, 2)],
            axis=2,
        ).max()
    )
    if float(np.percentile(quality, 5.0)) < 0.005 and maximum_edge > 10.0:
        labels[region["faces"]] = next_label
        next_label += 1
hard_edge_set = {tuple(map(int, edge)) for edge in hard_edges}
boundary_counts = {"both": 0, "one": 0, "open": 0}
boundary_lengths = []
estimated_splits = 0
for edge, memberships in _build_edge_faces(np.asarray(mesh.faces)).items():
    if edge in hard_edge_set:
        continue
    length = float(
        np.linalg.norm(np.asarray(mesh.vertices)[edge[0]] - np.asarray(mesh.vertices)[edge[1]])
    )
    if length <= 10.0 * (1.0 + 1e-8):
        continue
    face_ids = tuple(memberships)
    if len(face_ids) == 1:
        if labels[face_ids[0]] >= 0:
            boundary_counts["open"] += 1
        else:
            continue
    elif len(face_ids) == 2:
        first_label, second_label = labels[list(face_ids)]
        if first_label >= 0 and second_label >= 0 and first_label != second_label:
            boundary_counts["both"] += 1
        elif (first_label >= 0) != (second_label >= 0):
            boundary_counts["one"] += 1
        else:
            continue
    else:
        continue
    boundary_lengths.append(length)
    estimated_splits += 2 ** int(np.ceil(np.log2(length / 10.0))) - 1
print(
    "BOUNDARY_EXPANSION",
    boundary_counts,
    len(boundary_lengths),
    max(boundary_lengths) if boundary_lengths else 0.0,
    estimated_splits,
    flush=True,
)

triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
edge_lengths = np.linalg.norm(
    triangles[:, (1, 2, 0)] - triangles[:, (0, 1, 2)], axis=2
)
worst_face = int(np.argmax(edge_lengths.max(axis=1)))
label = "general"
region_size = 0
for category in ("cylinders", "fillets", "extrusions"):
    for region in nonplanar_regions[category]:
        if worst_face in set(region["faces"].tolist()):
            label = category
            region_size = len(region["faces"])
            break
print(
    "LONGEST_FACE",
    worst_face,
    float(edge_lengths[worst_face].max()),
    label,
    region_size,
    triangles[worst_face].tolist(),
    flush=True,
)
