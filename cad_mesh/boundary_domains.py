"""Separate removable surface labels from shared geometric constraints."""
from __future__ import annotations

from dataclasses import replace
import numpy as np


def model_distance_normal(points, patch):
    """Unsigned analytic distance and oriented model normal; no mesh mutation."""
    p = patch.get("parameters") or {}
    kind = patch["type"]
    if kind == "Plane":
        n = np.array(p["normal"], dtype=float, copy=True)
        n /= np.linalg.norm(n)
        return np.abs((points - p["origin"]) @ n), np.broadcast_to(n, points.shape)
    if kind == "Sphere":
        delta = points - p["center"]
        radius = np.linalg.norm(delta, axis=1)
        return np.abs(radius - p["radius"]), delta / np.maximum(radius[:, None], 1e-30)
    axis = np.array(p["axis_direction"], dtype=float, copy=True)
    axis /= np.linalg.norm(axis)
    delta = points - p["axis_origin"]
    height = delta @ axis
    radial = delta - height[:, None] * axis
    radius = np.linalg.norm(radial, axis=1)
    radial_normal = radial / np.maximum(radius[:, None], 1e-30)
    if kind == "Cylinder":
        return np.abs(radius - p["radius"]), radial_normal
    if kind == "Cone":
        angle = p["semi_angle_radians"]
        distance = np.abs(radius * np.cos(angle) - height * np.sin(angle))
        distance = np.where(height >= 0, distance, np.linalg.norm(delta, axis=1))
        return distance, radial_normal * np.cos(angle) - axis * np.sin(angle)
    if kind == "Torus":
        tube = radial - radial_normal * p["major_radius"] + height[:, None] * axis
        tube_radius = np.linalg.norm(tube, axis=1)
        return np.abs(tube_radius - p["minor_radius"]), tube / np.maximum(tube_radius[:, None], 1e-30)
    raise ValueError(f"No certified analytic model for {kind}")


def prepare_surface_domains(source):
    """Union compatible adjacent labels without crossing any geometric crease.

    Entire accumulated unions must fit both original representative models.
    Freeform labels are never assumed to be the same surface merely by type.
    Original semantic memberships remain available as source_patch_ids.
    """
    patches = source.report["patches"]
    count = len(patches)
    parent = np.arange(count)
    members = {i: [i] for i in range(count)}
    face_groups = [np.asarray(p["triangle_ids"], dtype=np.int64) for p in patches]
    vertices_by_patch = [np.unique(source.faces[f]) for f in face_groups]
    tri = source.vertices[source.faces]
    centers = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    del tri
    tolerance = 3.0 * float(source.report.get("resolution", {}).get("fitting_tolerance", 0))
    tolerance = max(tolerance, np.linalg.norm(np.ptp(source.vertices, axis=0)) * 1e-12)
    cosine = np.cos(0.14)
    hard_set = set(map(tuple, source.hard_edges))
    adjacency, blocked = set(), set()
    for edge in source.report["constraint_edges"]:
        labels = edge["incident_patch_ids"]
        if len(labels) != 2:
            continue
        pair = tuple(sorted(labels))
        adjacency.add(pair)
        if tuple(sorted(edge["vertex_ids"])) in hard_set:
            blocked.add(pair)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def fits(indices, model):
        for index in indices:
            distance, _ = model_distance_normal(source.vertices[vertices_by_patch[index]], model)
            if np.max(distance, initial=0) > tolerance:
                return False
            _, model_normals = model_distance_normal(centers[face_groups[index]], model)
            agreement = np.abs(np.einsum("ij,ij->i", normals[face_groups[index]], model_normals))
            if np.min(agreement, initial=1) < cosine:
                return False
        return True

    merged = 0
    for a, b in sorted(adjacency):
        a, b = root(a), root(b)
        if a == b:
            continue
        pa, pb = patches[a], patches[b]
        if (pa["type"] != pb["type"] or pa["type"] in ("Unknown", "Freeform")
                or pa.get("feature_role", "Ordinary") != pb.get("feature_role", "Ordinary")
                or pa.get("support_patch_ids", []) != pb.get("support_patch_ids", [])):
            continue
        if any(tuple(sorted((x, y))) in blocked for x in members[a] for y in members[b]):
            continue
        union = members[a] + members[b]
        if not fits(union, pa) or not fits(union, pb):
            continue
        parent[b] = a
        members[a] = union
        del members[b]
        merged += 1

    roots = np.array([root(i) for i in range(count)])
    representatives, patch_to_domain = np.unique(roots, return_inverse=True)
    labels = patch_to_domain[source.face_patch_ids]
    face_order = np.argsort(labels, kind="stable")
    face_offsets = np.r_[0, np.cumsum(np.bincount(labels, minlength=len(representatives)))]
    output_patches = []
    for domain, representative in enumerate(representatives):
        patch = dict(patches[representative])
        ids = face_order[face_offsets[domain]:face_offsets[domain + 1]]
        originals = np.flatnonzero(patch_to_domain == domain)
        patch.update(id=domain, triangle_ids=ids, triangle_count=len(ids),
                     source_patch_ids=originals.tolist(),
                     representative_source_patch_id=int(representative))
        patch["support_patch_ids"] = sorted(set(int(patch_to_domain[i]) for i in patch.get("support_patch_ids", [])) - {domain})
        output_patches.append(patch)
    removed = []
    kept_smooth = []
    for record in source.report["constraint_edges"]:
        pair = tuple(sorted(record["vertex_ids"]))
        if pair in hard_set:
            continue
        incident = set(patch_to_domain[record["incident_patch_ids"]])
        (kept_smooth if len(incident) > 1 else removed).append(pair)
    smooth = np.asarray(kept_smooth, dtype=np.int64).reshape(-1, 2)
    removed = np.asarray(removed, dtype=np.int64).reshape(-1, 2)
    required = np.unique(np.vstack((source.hard_edges, smooth)), axis=0)
    corners = np.intersect1d(source.corner_vertex_ids, np.unique(required))
    report = dict(source.report, patches=output_patches)
    prepared = replace(source, report=report, face_patch_ids=labels,
                       smooth_edges=np.unique(smooth, axis=0), corner_vertex_ids=corners)
    diagnostics = {
        "input_patch_count": count, "surface_domain_count": len(output_patches),
        "merged_label_pairs": merged, "released_label_edge_count": len(removed),
        "required_geometric_edge_count": len(required),
        "hard_geometric_edge_count": len(source.hard_edges),
        "smooth_geometric_transition_edge_count": len(smooth),
        "source_patch_to_surface_domain": patch_to_domain.tolist(),
        "released_label_edges": removed.tolist(),
        "union_distance_tolerance": tolerance,
        "semantics": "same-model label boundaries may disappear; geometric interfaces remain shared",
    }
    return prepared, diagnostics
