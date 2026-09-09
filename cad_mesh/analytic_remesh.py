"""Rebuild a complete analytic patch in one constrained parameter chart.

Boundary vertices are immutable. Periodic cylinder/cone charts duplicate only
their parameter seam; both seam copies map to exactly the same 3D vertex IDs.
Geometry-error diagnostics below are sampled distances, not Hausdorff bounds.
"""
from __future__ import annotations

import math

import numpy as np


class _Rejected(ValueError):
    pass


def _reject(reason):
    raise _Rejected(reason)


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        _reject("invalid_analytic_direction")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        _reject("zero_analytic_direction")
    return vector / length


def _basis(axis):
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = _unit(seed - axis * np.dot(seed, axis))
    return first, np.cross(axis, first)


def _edge_topology(faces):
    directed = faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True,
        return_counts=True,
    )
    orientation = np.zeros(len(edges), dtype=np.int64)
    np.add.at(orientation, inverse,
              np.where(directed[:, 0] < directed[:, 1], 1, -1))
    return edges, counts, orientation


def _loops(edges):
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        _reject("boundary_is_not_disjoint_simple_loops")
    remaining = set(adjacency)
    result = []
    while remaining:
        first = min(remaining)
        loop, previous, current = [], None, first
        while True:
            if current not in remaining:
                _reject("boundary_loop_revisits_a_vertex")
            remaining.remove(current)
            loop.append(current)
            neighbors = adjacency[current]
            following = neighbors[0] if neighbors[0] != previous else neighbors[1]
            previous, current = current, following
            if current == first:
                break
        if len(loop) < 3:
            _reject("boundary_loop_has_fewer_than_three_vertices")
        result.append(np.array(loop, dtype=np.int64))
    return result


def _polygon_area(points):
    following = np.roll(points, -1, axis=0)
    return .5 * float(np.sum(points[:, 0] * following[:, 1]
                             - points[:, 1] * following[:, 0]))


def _fan_boundary_loops(faces):
    """Follow oriented face fans at shared boundary vertices, not vertex degree."""
    following = {}
    for a, b, c in faces:
        following[int(a), int(b)] = (int(b), int(c))
        following[int(b), int(c)] = (int(c), int(a))
        following[int(c), int(a)] = (int(a), int(b))
    boundary = {edge for edge in following if edge[::-1] not in following}
    remaining = set(boundary)
    loops = []
    while remaining:
        start = min(remaining)
        edge, walk = start, []
        while True:
            if edge not in remaining:
                _reject('boundary_fan_walk_revisits_an_edge')
            remaining.remove(edge)
            walk.append(edge[0])
            edge = following[edge]
            steps = 0
            while edge not in boundary:
                edge = following[edge[::-1]]
                steps += 1
                if steps > len(following):
                    _reject('boundary_fan_walk_does_not_terminate')
            if edge == start:
                break
        # A boundary walk can touch itself at a physical vertex. Keep the
        # incident loops distinct in parameter space; their 3D alias is shared.
        stack, locations = [], {}
        for vertex in walk + [walk[0]]:
            if vertex in locations:
                position = locations[vertex]
                cycle = stack[position:]
                if len(cycle) < 3:
                    _reject('boundary_fan_loop_is_degenerate')
                loops.append(np.asarray(cycle, dtype=np.int64))
                for old in stack[position:]:
                    del locations[old]
                stack = stack[:position]
            locations[vertex] = len(stack)
            stack.append(vertex)
    return loops


def _adjacent_angles(theta, faces):
    """Propagate angular lifts through mesh edges and detect periodic cycles."""
    adjacency = {}
    for a, b in _edge_topology(faces)[0]:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    angle = np.full(len(theta), np.nan)
    for start in adjacency:
        if np.isfinite(angle[start]):
            continue
        angle[start] = theta[start]
        queue = [start]
        for a in queue:
            for b in adjacency[a]:
                delta = math.atan2(math.sin(theta[b] - theta[a]), math.cos(theta[b] - theta[a]))
                value = angle[a] + delta
                if np.isnan(angle[b]):
                    angle[b] = value
                    queue.append(b)
                elif abs(angle[b] - value) > 1e-7:
                    _reject('surface_requires_a_periodic_parameter_seam')
    return angle


def _inside_polygon(point, polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    selected = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    if not np.any(selected):
        return False
    a, b = a[selected], b[selected]
    x = a[:, 0] + (point[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(x > point[0]) % 2)


def _interior_point(polygon, triangle):
    count = len(polygon)
    result = triangle.triangulate({
        "vertices": polygon,
        "segments": np.column_stack((np.arange(count), np.roll(np.arange(count), -1))),
    }, "pYQ")
    if "triangles" not in result or not len(result["triangles"]):
        _reject("cannot_find_boundary_loop_interior")
    triangles = result["vertices"][result["triangles"]]
    area = np.abs(np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0]))
    point = triangles[int(np.argmax(area))].mean(axis=0)
    if not _inside_polygon(point, polygon):
        _reject("boundary_loop_interior_not_verified")
    return point


class _Surface:
    def __init__(self, patch):
        self.kind = patch.get("type", "Unknown")
        parameters = patch.get("parameters")
        if self.kind not in ("Plane", "Cylinder", "Cone"):
            _reject("unsupported_surface_type")
        if not isinstance(parameters, dict):
            _reject("missing_analytic_parameters")
        if self.kind == "Plane":
            self.origin = np.asarray(parameters["origin"], dtype=np.float64)
            self.axis = _unit(parameters["normal"])
        else:
            self.origin = np.asarray(parameters["axis_origin"], dtype=np.float64)
            self.axis = _unit(parameters["axis_direction"])
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            _reject("invalid_analytic_origin")
        self.first, self.second = _basis(self.axis)
        if self.kind == "Cylinder":
            self.radius = float(parameters["radius"])
            if not np.isfinite(self.radius) or self.radius <= 0.0:
                _reject("invalid_cylinder_radius")
        elif self.kind == "Cone":
            self.angle = float(parameters["semi_angle_radians"])
            if not 0.0 < self.angle < math.pi / 2:
                _reject("invalid_cone_semi_angle")
            self.sine, self.cosine = math.sin(self.angle), math.cos(self.angle)

    def coordinates(self, points):
        offsets = points - self.origin
        height = offsets @ self.axis
        theta = np.arctan2(offsets @ self.second, offsets @ self.first)
        radial = offsets - height[:, None] * self.axis
        return height, theta, np.linalg.norm(radial, axis=1)

    def distance(self, points):
        h, _, r = self.coordinates(points)
        if self.kind == "Plane":
            return np.abs(h)
        if self.kind == "Cylinder":
            return np.abs(r - self.radius)
        return np.where(h >= 0.0, np.abs(r * self.cosine - h * self.sine),
                        np.linalg.norm(points - self.origin, axis=1))

    def normals(self, points):
        if self.kind == "Plane":
            return np.broadcast_to(self.axis, points.shape)
        h, _, r = self.coordinates(points)
        radial = (points - self.origin - h[:, None] * self.axis)
        radial /= np.maximum(r[:, None], np.finfo(np.float64).tiny)
        if self.kind == "Cylinder":
            return radial
        return radial * self.cosine - self.axis * self.sine

    def from_cylindrical(self, height, theta):
        radius = self.radius if self.kind == "Cylinder" else height * math.tan(self.angle)
        return (self.origin + height[:, None] * self.axis
                + np.asarray(radius)[..., None]
                * (np.cos(theta)[:, None] * self.first
                   + np.sin(theta)[:, None] * self.second))


def _sample_points(vertices, faces):
    triangles = vertices[faces]
    return np.vstack((vertices[np.unique(faces)],
                      ((triangles + np.roll(triangles, -1, axis=1)) * .5).reshape(-1, 3),
                      triangles.mean(axis=1)))


def _simple_chart(vertices, faces, loops, surface):
    """One uncut patch chart, including planar/partial-surface holes."""
    if surface.kind == "Plane":
        offsets = vertices - surface.origin
        uv = np.column_stack((offsets @ surface.first, offsets @ surface.second))

        def lift(points):
            return (surface.origin + points[:, 0, None] * surface.first
                    + points[:, 1, None] * surface.second)
    else:
        height, theta, _ = surface.coordinates(vertices)
        used = np.unique(faces)
        if surface.kind == "Cone" and np.any(height[used] <= 0.0):
            _reject("nonperiodic_cone_chart_contains_apex_or_other_nappe")
        angle = _adjacent_angles(theta, faces)
        cut = float(np.nanmin(angle))
        angle -= cut
        if np.any(np.ptp(angle[faces], axis=1) >= math.pi):
            _reject("surface_requires_a_periodic_parameter_seam")
        if surface.kind == "Cylinder":
            uv = np.column_stack((surface.radius * angle, height))

            def lift(points):
                return surface.from_cylindrical(points[:, 1], points[:, 0] / surface.radius + cut)
        else:
            polar = angle * surface.sine
            slant = height / surface.cosine
            uv = np.column_stack((slant * np.cos(polar), slant * np.sin(polar)))
            polar_mid = .5 * (float(polar[used].min()) + float(polar[used].max()))

            def lift(points):
                angle_out = np.arctan2(points[:, 1], points[:, 0])
                angle_out += 2 * math.pi * np.round((polar_mid - angle_out) / (2 * math.pi))
                return surface.from_cylindrical(
                    np.linalg.norm(points, axis=1) * surface.cosine,
                    angle_out / surface.sine + cut,
                )
    uv_triangles = uv[faces]
    signs = np.cross(uv_triangles[:, 1] - uv_triangles[:, 0],
                     uv_triangles[:, 2] - uv_triangles[:, 0])
    if np.any(signs == 0.0) or (np.any(signs > 0.0) and np.any(signs < 0.0)):
        _reject("parameter_chart_folds_or_collapses_source_triangles")
    ids = np.concatenate(loops)
    chart_loops, offset = [], 0
    for loop in loops:
        chart_loops.append(np.arange(offset, offset + len(loop), dtype=np.int64))
        offset += len(loop)
    return {
        "uv": uv[ids], "aliases": ids, "loops": chart_loops, "lift": lift,
        "periodic": False, "seam_vertices": 0,
        "source_chart_area": float(np.abs(signs).sum() * .5),
    }


def _ordered_ring(loop, theta, anchor):
    relative = np.mod(theta[loop] - theta[anchor], 2 * math.pi)
    order = np.argsort(relative)
    ordered = loop[order]
    original_edges = {tuple(sorted((int(a), int(b))))
                      for a, b in zip(loop, np.roll(loop, -1))}
    sorted_edges = {tuple(sorted((int(a), int(b))))
                    for a, b in zip(ordered, np.roll(ordered, -1))}
    if original_edges != sorted_edges or np.max(np.diff(np.append(relative[order], 2 * math.pi))) >= math.pi:
        _reject("periodic_boundary_is_not_a_simple_circumferential_ring")
    return ordered, relative[order]


def _periodic_chart(vertices, faces, loops, surface, target, deviation):
    if surface.kind == "Plane" or len(loops) != 2:
        _reject("unsupported_periodic_boundary_configuration")
    height, theta, _ = surface.coordinates(vertices)
    lower, upper = sorted(loops, key=lambda loop: float(np.mean(height[loop])))
    levels = np.array([float(np.mean(height[lower])), float(np.mean(height[upper]))])
    scale = max(float(np.ptp(height[np.unique(faces)])), 1e-30)
    # Full periodic support currently requires two cross-sectional end rings.
    # Irregular trims use a valid single chart or the caller's 3D fallback.
    axial_tolerance = max(deviation * .1, scale * 1e-8)
    if any(np.ptp(height[loop]) > axial_tolerance for loop in (lower, upper)):
        _reject("periodic_end_boundaries_are_not_cross_sections")
    if levels[1] - levels[0] <= axial_tolerance:
        _reject("periodic_patch_has_no_axial_extent")
    if surface.kind == "Cone" and levels[0] <= 0.0:
        _reject("periodic_cone_chart_requires_positive_truncation_radius")
    low_anchor = int(lower[np.argmin(theta[lower])])
    upper_deltas = np.arctan2(np.sin(theta[upper] - theta[low_anchor]),
                              np.cos(theta[upper] - theta[low_anchor]))
    high_anchor = int(upper[int(np.argmin(np.abs(upper_deltas)))])
    delta = float(np.arctan2(math.sin(theta[high_anchor] - theta[low_anchor]),
                             math.cos(theta[high_anchor] - theta[low_anchor])))
    lower, low_angles = _ordered_ring(lower, theta, low_anchor)
    upper, high_angles = _ordered_ring(upper, theta, high_anchor)
    h0, h1 = float(height[low_anchor]), float(height[high_anchor])

    def cut_angle(h):
        return theta[low_anchor] + (h - h0) / (h1 - h0) * delta

    if surface.kind == "Cylinder":
        def project(h, relative):
            return np.column_stack((surface.radius * relative, h))

        def lift(points):
            return surface.from_cylindrical(
                points[:, 1], points[:, 0] / surface.radius + cut_angle(points[:, 1]))
        seam_length = math.hypot(h1 - h0, surface.radius * delta)
    else:
        polar_min = min(0.0, delta * surface.sine)
        polar_max = 2 * math.pi * surface.sine + max(0.0, delta * surface.sine)
        if polar_max - polar_min >= 2 * math.pi - 1e-7:
            _reject("periodic_cone_seam_overlaps_in_development")
        polar_mid = .5 * (polar_min + polar_max)

        def project(h, relative):
            polar = (relative + cut_angle(h) - theta[low_anchor]) * surface.sine
            slant = h / surface.cosine
            return np.column_stack((slant * np.cos(polar), slant * np.sin(polar)))

        def lift(points):
            polar = np.arctan2(points[:, 1], points[:, 0])
            polar += 2 * math.pi * np.round((polar_mid - polar) / (2 * math.pi))
            return surface.from_cylindrical(
                np.linalg.norm(points, axis=1) * surface.cosine,
                polar / surface.sine + theta[low_anchor],
            )
        seam_length = math.hypot((h1 - h0) / surface.cosine,
                                  levels[1] * math.tan(surface.angle) * delta)
    seam_count = max(1, int(math.ceil(seam_length / (target * .8))))
    if seam_count > 100000:
        _reject("periodic_seam_sample_budget_exceeded")
    seam_heights = np.linspace(h0, h1, seam_count + 1)[1:-1]
    # Negative aliases denote new seam vertices. Each appears once on each
    # parameter side, and is appended exactly once to the 3D vertex table.
    seam_alias = -1 - np.arange(len(seam_heights), dtype=np.int64)
    uv_parts = [project(height[lower], low_angles),
                project(np.array([h0]), np.array([2 * math.pi])),
                project(seam_heights, np.full(len(seam_heights), 2 * math.pi)),
                project(np.array([h1]), np.array([2 * math.pi])),
                project(height[upper[::-1]], high_angles[::-1]),
                project(seam_heights[::-1], np.zeros(len(seam_heights)))]
    aliases = np.concatenate((lower, [low_anchor], seam_alias, [high_anchor],
                               upper[::-1], seam_alias[::-1])).astype(np.int64)
    uv = np.vstack(uv_parts)
    return {
        "uv": uv, "aliases": aliases,
        "loops": [np.arange(len(uv), dtype=np.int64)], "lift": lift,
        "periodic": True, "seam_vertices": len(seam_heights),
        "source_chart_area": None,
    }


def _trimmed_periodic_chart(vertices, faces, surface, target, cut=0., allow_shared_fans=False):
    """Cut a cylinder/cone along source adjacency, retaining only cut/trim edges.

    Each triangle is lifted to the angular universal cover. Vertex copies with
    different windings form the two seam sides; aliases weld them after chart
    triangulation. Source interior diagonals do not enter the PSLG.
    """
    height, theta, _ = surface.coordinates(vertices)
    if surface.kind == 'Cone' and np.any(height[np.unique(faces)] <= 0):
        _reject('trimmed_cone_chart_contains_apex_or_other_nappe')
    period = 2 * math.pi
    angles = np.mod(theta - cut, period)
    face_angles = angles[faces]
    relative = np.arctan2(np.sin(face_angles - face_angles[:, :1]),
                          np.cos(face_angles - face_angles[:, :1]))
    lifted = face_angles[:, :1] + relative
    lifted -= period * np.floor(lifted.mean(axis=1) / period)[:, None]
    if np.any(np.ptp(lifted, axis=1) >= math.pi):
        _reject('trimmed_cylinder_has_ambiguous_angular_triangle')
    winding = np.rint((lifted - face_angles) / period).astype(np.int64)
    records, inverse = np.unique(np.column_stack((faces.reshape(-1), winding.reshape(-1))),
                                 axis=0, return_inverse=True)
    chart_faces = inverse.reshape(-1, 3)
    unwrapped = angles[records[:, 0]] + period * records[:, 1]
    if surface.kind == 'Cylinder':
        uv = np.column_stack((surface.radius * unwrapped, height[records[:, 0]]))

        def lift(points):
            return surface.from_cylindrical(points[:, 1], points[:, 0] / surface.radius + cut)
    else:
        polar = unwrapped * surface.sine
        if np.ptp(polar) >= 2 * math.pi - 1e-7:
            _reject('trimmed_cone_seam_overlaps_in_development')
        polar_mid = .5 * (polar.min() + polar.max())
        slant = height[records[:, 0]] / surface.cosine
        uv = np.column_stack((slant * np.cos(polar), slant * np.sin(polar)))

        def lift(points):
            phi = np.arctan2(points[:, 1], points[:, 0])
            phi += period * np.round((polar_mid - phi) / period)
            return surface.from_cylindrical(np.linalg.norm(points, axis=1) * surface.cosine,
                                            phi / surface.sine + cut)
    triangles = uv[chart_faces]
    signs = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(signs == 0) or (np.any(signs > 0) and np.any(signs < 0)):
        _reject('trimmed_cylinder_chart_folds_or_collapses')
    edges, counts, _ = _edge_topology(chart_faces)
    cut_edges = edges[counts == 1]
    loops = _fan_boundary_loops(chart_faces) if allow_shared_fans else _loops(cut_edges)
    physical_edges = np.sort(records[cut_edges, 0], axis=1)
    unique_edges, edge_counts = np.unique(physical_edges, axis=0, return_counts=True)
    seam_edges = set(map(tuple, unique_edges[edge_counts == 2]))
    seam_sites = {}
    points, aliases, result_loops = [], [], []
    next_alias = -1
    for loop in loops:
        chart_loop = []
        for a, b in zip(loop, np.roll(loop, -1)):
            chart_loop.append(len(points))
            points.append(uv[a])
            first, second = int(records[a, 0]), int(records[b, 0])
            aliases.append(first)
            key = tuple(sorted((first, second)))
            if key not in seam_edges:
                continue
            if key not in seam_sites:
                divisions = max(1, int(math.ceil(np.linalg.norm(uv[b] - uv[a]) / (target * .8))))
                if divisions > 100000:
                    _reject('periodic_seam_sample_budget_exceeded')
                new_aliases = np.arange(next_alias, next_alias - divisions + 1, -1, dtype=np.int64)
                next_alias -= divisions - 1
                seam_sites[key] = new_aliases
            new_aliases = seam_sites[key]
            if first > second:
                new_aliases = new_aliases[::-1]
            for fraction, alias in zip(np.linspace(0, 1, len(new_aliases) + 2)[1:-1], new_aliases):
                chart_loop.append(len(points))
                points.append((1 - fraction) * uv[a] + fraction * uv[b])
                aliases.append(int(alias))
        result_loops.append(np.asarray(chart_loop, dtype=np.int64))

    # PSLG loops may share a point. Supply it once to Triangle while retaining
    # separate loop walks, avoiding zero-length edges or duplicate sites.
    unique_points, unique_aliases, site_ids = [], [], {}
    mapping = []
    for point, alias in zip(points, aliases):
        key = (int(alias), float(point[0]), float(point[1]))
        if key not in site_ids:
            site_ids[key] = len(unique_points)
            unique_points.append(point)
            unique_aliases.append(alias)
        mapping.append(site_ids[key])
    mapping = np.asarray(mapping)
    result_loops = [mapping[loop] for loop in result_loops]
    return {'uv': np.asarray(unique_points), 'aliases': np.asarray(unique_aliases, dtype=np.int64),
            'loops': result_loops, 'lift': lift, 'periodic': True,
            'seam_vertices': -next_alias - 1,
            'source_chart_area': float(np.abs(signs).sum() * .5),
            'trimmed_cylinder': surface.kind == 'Cylinder',
            'trimmed_cone': surface.kind == 'Cone'}


def _routed_periodic_chart(vertices, faces, surface, target):
    # Try alternative meridians before accepting any shared-point cut. Most
    # regular patches use the first route; work is bounded for difficult trims.
    last_error = None
    for shared_fans in (False, True):
        for cut in np.linspace(0., 2 * math.pi, 12, endpoint=False):
            try:
                chart = _trimmed_periodic_chart(vertices, faces, surface, target,
                                               float(cut), shared_fans)
                chart['seam_cut_angle'] = float(cut)
                chart['shared_boundary_fans'] = shared_fans
                return chart
            except _Rejected as error:
                last_error = error
    raise last_error


def _triangulate_chart(chart, target, triangle, original_vertices,
                       fixed_boundary_edges, maximum_edge_length):
    uv, loops = chart["uv"], chart["loops"]
    offset = uv.min(axis=0)
    coordinate_scale = float(np.max(np.ptp(uv, axis=0)))
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0.0:
        _reject("parameter_domain_has_invalid_scale")
    normalized_uv = (uv - offset) / coordinate_scale
    areas = np.array([abs(_polygon_area(normalized_uv[loop])) for loop in loops])
    outer = int(np.argmax(areas))
    if areas[outer] <= 0.0:
        _reject("parameter_domain_has_zero_area")
    holes = []
    for index, loop in enumerate(loops):
        if index == outer:
            continue
        point = _interior_point(normalized_uv[loop], triangle)
        if not _inside_polygon(point, normalized_uv[loops[outer]]):
            _reject("parameter_domain_has_multiple_outer_components")
        if any(_inside_polygon(point, normalized_uv[other])
               for j, other in enumerate(loops) if j not in (outer, index)):
            _reject("nested_hole_loops_are_unsupported")
        holes.append(point)
    domain_area = float(areas[outer] - np.sum(np.delete(areas, outer)))
    if domain_area <= 0.0:
        _reject("invalid_parameter_domain_area")
    source_area = chart["source_chart_area"]
    if source_area is not None:
        source_area /= coordinate_scale * coordinate_scale
        if abs(source_area - domain_area) > max(domain_area * 1e-7, 1e-20):
            _reject("source_chart_does_not_cover_the_boundary_domain_once")
    normalized_target = min(target / coordinate_scale, 2.0)
    maximum_area = math.sqrt(3.0) * .25 * normalized_target * normalized_target * .8
    if maximum_area <= 0.0:
        _reject("analytic_chart_target_is_below_numeric_precision")
    if domain_area / maximum_area > 200000:
        _reject("analytic_chart_face_budget_exceeded")
    segments = np.vstack([np.column_stack((loop, np.roll(loop, -1))) for loop in loops])
    # A long immutable edge can prevent Triangle's circumcenter refinement:
    # the desired center would encroach on a segment that Y forbids splitting.
    # Place a thin row of interior support points beside those edges instead.
    # These are chart-level seeds, unrelated to the source triangulation.
    support_points = []
    seen_points = set(map(tuple, normalized_uv))

    def in_domain(point):
        return (_inside_polygon(point, normalized_uv[loops[outer]])
                and not any(_inside_polygon(point, normalized_uv[loop])
                            for index, loop in enumerate(loops) if index != outer))

    for first, second in normalized_uv[segments]:
        vector = second - first
        length = float(np.linalg.norm(vector))
        if length <= 2 * normalized_target:
            continue
        inward = np.array([-vector[1], vector[0]]) / length
        step = min(normalized_target * .25, maximum_area / length)
        midpoint = (first + second) * .5
        for sign in (1.0, -1.0):
            point = midpoint + sign * step * inward
            if in_domain(point):
                if tuple(point) not in seen_points:
                    support_points.append(point)
                    seen_points.add(tuple(point))
                break
    input_points = (np.vstack((normalized_uv, support_points))
                    if support_points else normalized_uv)
    pslg = {"vertices": input_points, "segments": segments}
    if holes:
        pslg["holes"] = np.asarray(holes)
    # Triangle's maximum area is a density request, not an edge-length bound.
    # Certify actual 3D lengths and add midpoint sites only to the generated
    # chart's long interior edges. This never subdivides source triangles or
    # independently resamples the supplied boundary/seam constraints.
    fixed_edges = set(map(tuple, fixed_boundary_edges))
    edge_bound = maximum_edge_length * (1.0 + 1e-6)
    aliases = chart["aliases"]
    for refinement_pass in range(9):
        result = triangle.triangulate(pslg, f"pYq25a{maximum_area:.17f}Q")
        if "triangles" not in result or not len(result["triangles"]):
            _reject("constrained_chart_triangulation_failed")
        normalized_output = np.asarray(result["vertices"], dtype=np.float64)
        output_faces = np.asarray(result["triangles"], dtype=np.int64)
        if len(output_faces) > 500000:
            _reject("analytic_chart_output_budget_exceeded")
        if (len(normalized_output) < len(uv)
                or not np.array_equal(normalized_output[:len(uv)], normalized_uv)):
            _reject("triangulator_changed_supplied_constraint_vertices")
        output_edges, counts, _ = _edge_topology(output_faces)
        output_uv = normalized_output * coordinate_scale + offset
        output_uv[:len(uv)] = uv
        spatial = chart["lift"](output_uv)
        original_sites = np.flatnonzero(aliases >= 0)
        spatial[original_sites] = original_vertices[aliases[original_sites]]
        seam_sites = {}
        for site in np.flatnonzero(aliases < 0):
            alias = int(aliases[site])
            if alias in seam_sites:
                spatial[site] = spatial[seam_sites[alias]]
            else:
                seam_sites[alias] = site
        fixed = np.zeros(len(output_edges), dtype=bool)
        for index in np.flatnonzero(np.all(output_edges < len(aliases), axis=1)):
            endpoints = aliases[output_edges[index]]
            if np.all(endpoints >= 0):
                fixed[index] = tuple(sorted(map(int, endpoints))) in fixed_edges
        lengths = np.linalg.norm(spatial[output_edges[:, 1]] - spatial[output_edges[:, 0]], axis=1)
        if not np.isfinite(lengths).all():
            _reject("nonfinite_analytic_chart_edge_length")
        long_edges = (~fixed) & (lengths > edge_bound)
        if not np.any(long_edges):
            chart["interior_refinement_passes"] = refinement_pass
            chart["maximum_internal_edge_length"] = float(np.max(lengths[~fixed], initial=0.0))
            break
        if np.any(long_edges & (counts == 1)):
            # A seam needs paired samples. Its construction already targets
            # 0.8 times the edge bound; reject unusual residual-induced excess
            # instead of introducing an unpaired parameter boundary midpoint.
            _reject("periodic_seam_exceeds_target_edge_length")
        if refinement_pass == 8:
            _reject("analytic_chart_internal_edge_limit_not_reached")
        midpoint_sites = np.mean(normalized_output[output_edges[long_edges]], axis=1)
        midpoint_sites = np.unique(midpoint_sites, axis=0)
        existing_sites = set(map(tuple, normalized_output))
        midpoint_sites = np.asarray([point for point in midpoint_sites
                                     if tuple(point) not in existing_sites])
        if not len(midpoint_sites):
            _reject("analytic_chart_edge_refinement_stalled")
        pslg["vertices"] = np.vstack((normalized_output, midpoint_sites))
    boundary = output_edges[counts == 1]
    if set(map(tuple, boundary)) != set(map(tuple, np.sort(segments, axis=1))):
        _reject("triangulator_changed_parameter_boundary_or_seam")
    triangles = normalized_output[output_faces]
    output_area = .5 * np.abs(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0])).sum()
    if abs(output_area - domain_area) > max(domain_area * 1e-7, 1e-20):
        _reject("triangulation_did_not_preserve_parameter_domain_area")
    return output_uv, output_faces, len(holes), domain_area * coordinate_scale * coordinate_scale


def _map_to_shared_geometry(vertices, chart, uv, faces):
    output = [point.copy() for point in vertices]
    chart_map = np.empty(len(uv), dtype=np.int64)
    lifted = chart["lift"](uv)
    seam_ids = {}
    for index, alias in enumerate(chart["aliases"]):
        if alias >= 0:
            chart_map[index] = alias
        elif int(alias) in seam_ids:
            chart_map[index] = seam_ids[int(alias)]
        else:
            seam_ids[int(alias)] = len(output)
            chart_map[index] = len(output)
            output.append(lifted[index])
    for index in range(len(chart["aliases"]), len(uv)):
        chart_map[index] = len(output)
        output.append(lifted[index])
    return np.asarray(output), chart_map[faces]


def _sampled_mesh_distance(points, vertices, faces, igl):
    maximum = 0.0
    for start in range(0, len(points), 32768):
        squared, _, _ = igl.point_mesh_squared_distance(points[start:start + 32768], vertices, faces)
        if not np.isfinite(squared).all():
            _reject("nonfinite_reference_distance")
        maximum = max(maximum, math.sqrt(max(0.0, float(np.max(squared)))))
    return maximum


def remesh_analytic_patch(vertices, faces, patch, boundary_edges,
                          target_edge_length, maximum_deviation):
    """Return whole-chart geometry, or ``(None, None, rejected_diagnostics)``.

    Accepted geometry retains the original vertex table as an unchanged
    prefix. All boundary edges and their indices survive exactly; callers may
    later compact unused original interior vertices across the whole mesh.
    Planes, trimmed periodic cylinders (including holes), partial cones and
    two-ring periodic cones are supported. Unverified charts return to 3D.
    """
    diagnostics = {"accepted": False, "mode": "whole_analytic_chart",
                   "surface_type": patch.get("type", "Unknown")}
    try:
        import triangle
        import igl
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces)
        boundary_edges = np.asarray(boundary_edges)
        target = float(target_edge_length)
        deviation = float(maximum_deviation)
        if not np.isfinite(target) or target <= 0 or not np.isfinite(deviation) or deviation < 0:
            _reject("invalid_target_length_or_deviation")
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or np.any(faces < 0) or np.any(faces >= len(vertices))):
            _reject("invalid_source_mesh")
        faces = np.asarray(faces, dtype=np.int64)
        if (boundary_edges.ndim != 2 or boundary_edges.shape[1] != 2
                or not np.issubdtype(boundary_edges.dtype, np.integer)):
            _reject("invalid_boundary_edge_array")
        boundary_edges = np.sort(np.asarray(boundary_edges, dtype=np.int64), axis=1)
        source_edges, source_counts, source_orientation = _edge_topology(faces)
        if np.any(source_counts > 2) or np.any(source_orientation[source_counts == 2] != 0):
            _reject("source_patch_is_nonmanifold_or_inconsistently_wound")
        actual_boundary = source_edges[source_counts == 1]
        if (len(np.unique(boundary_edges, axis=0)) != len(boundary_edges)
                or set(map(tuple, actual_boundary)) != set(map(tuple, boundary_edges))):
            _reject("constraints_do_not_equal_patch_boundary_loops")
        boundary_lengths = np.linalg.norm(vertices[actual_boundary[:, 1]]
                                           - vertices[actual_boundary[:, 0]], axis=1)
        if np.any(boundary_lengths > target * (1.0 + 1e-6)):
            _reject("fixed_boundary_exceeds_target_edge_length")
        loops = _fan_boundary_loops(faces)
        surface = _Surface(patch)
        source_points = _sample_points(vertices, faces)
        scale = max(float(np.max(np.abs(source_points))), float(np.max(np.ptp(source_points, axis=0))), 1e-30)
        numeric_tolerance = scale * np.finfo(np.float64).eps * 128
        model_deviation = float(np.max(surface.distance(source_points)))
        diagnostics["source_to_analytic_sampled_maximum"] = model_deviation
        if model_deviation > deviation + numeric_tolerance:
            _reject("source_mesh_exceeds_analytic_model_deviation")
        triangles = vertices[faces]
        source_cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        if np.any(np.linalg.norm(source_cross, axis=1) == 0.0):
            _reject("source_mesh_contains_degenerate_faces")
        source_normal_alignment = np.sum(surface.normals(triangles.mean(axis=1)) * source_cross, axis=1)
        if np.any(source_normal_alignment == 0.0) or (np.any(source_normal_alignment > 0) and np.any(source_normal_alignment < 0)):
            _reject("source_mesh_is_not_consistently_oriented_on_analytic_surface")
        orientation = 1.0 if source_normal_alignment[0] > 0.0 else -1.0
        effective_target = target
        if surface.kind != "Plane":
            _, _, radii = surface.coordinates(vertices[np.unique(faces)])
            radius = surface.radius if surface.kind == "Cylinder" else float(np.min(radii))
            if radius <= 0.0 or deviation <= 0.0:
                _reject("curved_chart_requires_positive_radius_and_deviation")
            effective_target = min(target, math.sqrt(8 * radius * deviation) * .7,
                                   radius * math.pi / 3)
        try:
            chart = _simple_chart(vertices, faces, loops, surface)
        except _Rejected as failure:
            if str(failure) != "surface_requires_a_periodic_parameter_seam":
                raise
            chart = _routed_periodic_chart(vertices, faces, surface, effective_target)
        uv, chart_faces, holes, chart_area = _triangulate_chart(
            chart, effective_target, triangle, vertices, actual_boundary, target)
        output_vertices, output_faces = _map_to_shared_geometry(vertices, chart, uv, chart_faces)
        output_triangles = output_vertices[output_faces]
        crosses = np.cross(output_triangles[:, 1] - output_triangles[:, 0],
                            output_triangles[:, 2] - output_triangles[:, 0])
        alignment = orientation * np.sum(surface.normals(output_triangles.mean(axis=1)) * crosses, axis=1)
        if np.any(alignment == 0.0) or not np.isfinite(alignment).all():
            _reject("remeshed_chart_contains_degenerate_triangles")
        # Triangle's planar winding may be the opposite of the source surface,
        # notably for the cone development. Correct only one global reversal.
        if np.all(alignment < 0.0):
            output_faces = output_faces[:, [0, 2, 1]]
        elif np.any(alignment < 0.0):
            _reject("remeshed_chart_folds_on_the_surface")
        output_edges, output_counts, output_orientation = _edge_topology(output_faces)
        if (np.any(output_counts > 2) or np.any(output_orientation[output_counts == 2] != 0)
                or set(map(tuple, output_edges[output_counts == 1])) != set(map(tuple, actual_boundary))):
            _reject("periodic_seam_or_shared_boundary_did_not_stitch")
        original_euler = len(np.unique(faces)) - len(source_edges) + len(faces)
        output_euler = len(np.unique(output_faces)) - len(output_edges) + len(output_faces)
        if original_euler != output_euler:
            _reject("analytic_remeshing_changed_patch_topology")
        maximum_output_edge = float(np.max(np.linalg.norm(
            output_vertices[output_edges[:, 1]] - output_vertices[output_edges[:, 0]], axis=1)))
        if maximum_output_edge > target * (1.0 + 1e-6):
            _reject("analytic_remesh_exceeds_target_edge_length")
        output_points = _sample_points(output_vertices, output_faces)
        outward = _sampled_mesh_distance(output_points, vertices, faces, igl)
        inward = _sampled_mesh_distance(source_points, output_vertices, output_faces, igl)
        diagnostics.update({
            "output_to_reference_sampled_maximum": outward,
            "reference_to_output_sampled_maximum": inward,
            "geometry_sampling": "all_vertices_edge_midpoints_and_face_centroids",
            "hausdorff_upper_bound": False,
        })
        if max(outward, inward) > deviation + numeric_tolerance:
            _reject("remeshed_chart_exceeds_sampled_reference_deviation")
        diagnostics.update({
            "accepted": True, "reason": "accepted", "periodic_seam": chart["periodic"],
            "trimmed_cylinder_chart": bool(chart.get("trimmed_cylinder", False)),
            "trimmed_cone_chart": bool(chart.get("trimmed_cone", False)),
            "seam_cut_angle": chart.get('seam_cut_angle'),
            "shared_boundary_fans": bool(chart.get('shared_boundary_fans', False)),
            "seam_vertex_count": chart["seam_vertices"], "holes": holes,
            "boundary_loops": len(loops), "original_faces": len(faces),
            "output_faces": len(output_faces),
            "new_vertices": len(output_vertices) - len(vertices),
            "target_edge_length": target, "effective_chart_edge_length": effective_target,
            "maximum_internal_edge_length": chart["maximum_internal_edge_length"],
            "maximum_edge_length": maximum_output_edge,
            "interior_refinement_passes": chart["interior_refinement_passes"],
            "chart_area": chart_area, "boundary_vertices_fixed": True,
            "source_interior_triangulation_reused": False,
        })
        return output_vertices, output_faces, diagnostics
    except (ValueError, KeyError, TypeError, OverflowError, ImportError, RuntimeError) as failure:
        diagnostics["reason"] = str(failure) or type(failure).__name__
        return None, None, diagnostics
