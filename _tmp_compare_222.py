import numpy as np
import trimesh


def triangle_quality(vertices, faces):
    triangles = vertices[faces]
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 1]
    third = triangles[:, 0] - triangles[:, 2]
    squared_sum = (
        np.einsum("ij,ij->i", first, first)
        + np.einsum("ij,ij->i", second, second)
        + np.einsum("ij,ij->i", third, third)
    )
    twice_area = np.linalg.norm(np.cross(first, -third), axis=1)
    return np.divide(
        2.0 * np.sqrt(3.0) * twice_area,
        squared_sum,
        out=np.zeros(len(faces), dtype=np.float64),
        where=squared_sum > 0.0,
    )


def analyze(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    edge_lengths = np.linalg.norm(
        triangles[:, (1, 2, 0)] - triangles[:, (0, 1, 2)], axis=2
    )
    longest = edge_lengths.max(axis=1)
    quality = triangle_quality(vertices, faces)
    worst_edge_face = int(np.argmax(longest))
    target_mask = (
        (triangles[:, :, 1].min(axis=1) > 123.2)
        & (triangles[:, :, 1].max(axis=1) < 123.5)
        & (triangles[:, :, 2].min(axis=1) > -24.2)
        & (triangles[:, :, 2].max(axis=1) < -23.7)
    )
    target_longest = (
        float(longest[target_mask].max()) if np.any(target_mask) else 0.0
    )
    return {
        "vertices": len(vertices),
        "faces": len(faces),
        "quality_p1": float(np.percentile(quality, 1.0)),
        "quality_p5": float(np.percentile(quality, 5.0)),
        "quality_p25": float(np.percentile(quality, 25.0)),
        "quality_median": float(np.median(quality)),
        "quality_mean": float(quality.mean()),
        "edge_p95": float(np.percentile(longest, 95.0)),
        "edge_p99": float(np.percentile(longest, 99.0)),
        "edge_max": float(longest.max()),
        "faces_over_100": int(np.count_nonzero(longest > 100.0)),
        "faces_over_400": int(np.count_nonzero(longest > 400.0)),
        "target_longest": target_longest,
        "worst_triangle": triangles[worst_edge_face].tolist(),
    }


for label, path in (
    ("input", "examples/222.stl"),
    ("rejected_bisection", "examples/test_outputs/222_thinwall_feature_adaptive.stl"),
    ("new", "examples/test_outputs/222_thinwall_classified_rebuilt.stl"),
):
    print(label, analyze(path), flush=True)
