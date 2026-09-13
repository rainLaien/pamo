"""Double-precision, patch-scoped reference queries on CUDA via Warp.

The immutable reference forest is built once and shared by all remesh batches.
Bounds, distances and support tests use float64. Queries traverse the complete
patch BVH, without a nearest-k approximation or float32 mesh coordinates.
"""
import numpy as np
import torch
import warp as wp

from .surface_sample import _ReferencePatchProjector

wp.set_module_options({"enable_backward": False, "fast_math": False})
vec3l = wp.types.vector(length=3, dtype=wp.int64)
stack32 = wp.types.vector(length=32, dtype=wp.int64)


class _CudaTriangleValidationCache:
    """Exact, bounded device cache for fixed-coordinate endpoint collapses.

    Ordered vertex IDs and the compact patch ID form a lossless integer key.
    Keys are stored separately rather than packed into one int64: packing
    silently disabled caching on large meshes. Hash collisions only evict
    entries, and all reads finish before the elected writers update slots.
    Uncached rows are packed on the device so BVH traversal runs on occupied
    lanes instead of leaving most lanes idle among scattered cache hits.
    """

    def __init__(self, vertices, projector, maximum_deviation, normal_degrees):
        self.vertices = vertices
        self.projector = projector
        self.maximum_deviation = maximum_deviation
        self.normal_degrees = normal_degrees
        self.version = -1
        self.enabled = len(vertices) > 0
        # Each endpoint direction proposes many cavity triangles per vertex.
        # Leave room for both directions and successive passes to avoid
        # repeatedly evicting unchanged geometry (at most 148 MiB total).
        self.capacity = min(1 << 22, 1 << max(0, (32 * len(vertices) - 1).bit_length()))
        self.keys = torch.empty((self.capacity, 3), device=vertices.device, dtype=torch.long)
        self.labels = torch.full((self.capacity,), -1, device=vertices.device, dtype=torch.long)
        self.values = torch.empty(self.capacity, device=vertices.device, dtype=torch.bool)
        # The bundled Warp 1.0 beta exposes integer atomic_min only for
        # int32 values and int32 indices. Owners are query-row numbers, not
        # geometric IDs; keep the lossless face/patch keys as int64.
        self.owners = torch.full((self.capacity,), 2**31 - 1, device=vertices.device, dtype=torch.int32)

    def check(self, faces, patch_ids):
        if not self.enabled:
            return self.projector.valid_triangles(self.vertices[faces], patch_ids,
                self.maximum_deviation, self.normal_degrees)
        if self.version != self.vertices._version:
            if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                    or self.vertices.dtype != torch.float64
                    or self.vertices.device != self.projector.gpu_vertices.device):
                raise ValueError('CUDA validation cache needs float64 (n,3) reference-device vertices.')
            if not bool(torch.isfinite(self.vertices).all()):
                raise ValueError('Patch projection points must be finite.')
            self.labels.fill_(-1)
            self.version = self.vertices._version
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.long or faces.device != self.vertices.device:
            raise ValueError('CUDA validation cache needs int64 (n,3) reference-device faces.')
        labels = self.projector._labels(patch_ids, len(faces))
        valid = torch.empty(len(faces), device=faces.device, dtype=torch.bool)
        if len(faces):
            # Retain Torch's checked indexing before passing coordinates to
            # Warp, without sorting keys or compacting a GPU miss list on CPU.
            triangles = self.vertices[faces].reshape(-1, 3).contiguous()
            query_faces = wp.from_torch(faces.contiguous(), dtype=vec3l)
            query_labels = wp.from_torch(labels)
            cache_args = [wp.from_torch(self.keys, dtype=vec3l), wp.from_torch(self.labels),
                          wp.from_torch(self.values), wp.from_torch(self.owners), self.capacity - 1]
            misses = torch.empty(len(faces), device=faces.device, dtype=torch.int32)
            miss_count = torch.zeros(1, device=faces.device, dtype=torch.int32)
            miss_args = [wp.from_torch(misses), wp.from_torch(miss_count)]
            cosine, limit = self.projector._validation_limits(self.maximum_deviation, self.normal_degrees)
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(faces.device))):
                wp.launch(_validation_cache_lookup, dim=len(faces), inputs=[
                    query_faces, query_labels, *cache_args, *miss_args, wp.from_torch(valid)])
                # The device count bounds work without a host synchronization;
                # threads beyond the compacted queue return immediately.
                wp.launch(_validate_cache_misses, dim=len(faces), inputs=[
                    wp.from_torch(triangles, dtype=wp.vec3d), query_labels,
                    *self.projector._reference_args, *self.projector._normal_args,
                    limit, cosine, self.projector.normal_tie_tolerance,
                    self.maximum_deviation is not None, self.normal_degrees is not None,
                    *miss_args, wp.from_torch(valid)])
                wp.launch(_store_validation_cache, dim=len(faces), inputs=[
                    query_faces, query_labels, *cache_args, wp.from_torch(valid)])
        return valid


@wp.func
def _edge_point(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d):
    edge = b - a
    t = wp.clamp(wp.dot(p - a, edge) / wp.dot(edge, edge), wp.float64(0.0), wp.float64(1.0))
    return a + t * edge


@wp.func
def _triangle_point(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    # Cross-product half spaces avoid the cancelling Gram determinant of an
    # extremely thin reference triangle. All edge candidates remain available.
    n = wp.cross(b - a, c - a)
    q = p - n * (wp.dot(p - a, n) / wp.dot(n, n))
    inside = (wp.dot(wp.cross(b - a, q - a), n) >= wp.float64(0.0)
              and wp.dot(wp.cross(c - b, q - b), n) >= wp.float64(0.0)
              and wp.dot(wp.cross(a - c, q - c), n) >= wp.float64(0.0))
    closest = _edge_point(p, a, b)
    distance = wp.dot(p - closest, p - closest)
    other = _edge_point(p, b, c)
    squared = wp.dot(p - other, p - other)
    if squared < distance:
        closest = other
        distance = squared
    other = _edge_point(p, c, a)
    squared = wp.dot(p - other, p - other)
    if squared < distance:
        closest = other
        distance = squared
    if inside and wp.dot(p - q, p - q) <= distance:
        closest = q
    return closest


@wp.func
def _box_distance(p: wp.vec3d, lower: wp.vec3d, upper: wp.vec3d):
    delta = wp.max(wp.max(lower - p, p - upper), wp.vec3d(wp.float64(0.0)))
    return wp.dot(delta, delta)


@wp.func
def _nearest(p: wp.vec3d, root: wp.int64, capacity: wp.int64,
             lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
             node_faces: wp.array(dtype=wp.int64), vertices: wp.array(dtype=wp.vec3d),
             faces: wp.array(dtype=vec3l)):
    best = wp.float64(1.7976931348623157e308)
    best_id = wp.int64(-1)
    closest = p
    pending = stack32(wp.int64(0))
    pending[0] = root
    size = int(1)
    while size > 0:
        size -= 1
        node = pending[size]
        local = node - root
        if _box_distance(p, lower[node], upper[node]) <= best:
            if local >= capacity - wp.int64(1):
                face_id = node_faces[node]
                if face_id >= wp.int64(0):
                    f = faces[face_id]
                    q = _triangle_point(p, vertices[f[0]], vertices[f[1]], vertices[f[2]])
                    distance = wp.dot(q - p, q - p)
                    if distance < best or (distance == best and (best_id < wp.int64(0) or face_id < best_id)):
                        best = distance
                        best_id = face_id
                        closest = q
            else:
                left = root + wp.int64(2) * local + wp.int64(1)
                right = left + wp.int64(1)
                left_distance = _box_distance(p, lower[left], upper[left])
                right_distance = _box_distance(p, lower[right], upper[right])
                near = left
                far = right
                near_distance = left_distance
                far_distance = right_distance
                if right_distance < left_distance:
                    near = right
                    far = left
                    near_distance = right_distance
                    far_distance = left_distance
                if far_distance <= best:
                    pending[size] = far
                    size += 1
                if near_distance <= best:
                    pending[size] = near
                    size += 1
    return closest, best, best_id


@wp.func
def _within_distance(p: wp.vec3d, limit: wp.float64, seed: wp.int64,
                     root: wp.int64, capacity: wp.int64,
                     lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
                     node_faces: wp.array(dtype=wp.int64), vertices: wp.array(dtype=wp.vec3d),
                     faces: wp.array(dtype=vec3l)):
    # Boolean distance validation needs an exact witness inside the bound,
    # not the globally nearest face. Try the centroid's support first, then
    # traverse the complete patch BVH until such a witness is found.
    if seed >= wp.int64(0):
        f = faces[seed]
        q = _triangle_point(p, vertices[f[0]], vertices[f[1]], vertices[f[2]])
        if wp.dot(q - p, q - p) <= limit:
            return True
    pending = stack32(wp.int64(0))
    pending[0] = root
    size = int(1)
    while size > 0:
        size -= 1
        node = pending[size]
        local = node - root
        if _box_distance(p, lower[node], upper[node]) <= limit:
            if local >= capacity - wp.int64(1):
                face_id = node_faces[node]
                if face_id >= wp.int64(0):
                    f = faces[face_id]
                    q = _triangle_point(p, vertices[f[0]], vertices[f[1]], vertices[f[2]])
                    if wp.dot(q - p, q - p) <= limit:
                        return True
            else:
                left = root + wp.int64(2) * local + wp.int64(1)
                right = left + wp.int64(1)
                left_distance = _box_distance(p, lower[left], upper[left])
                right_distance = _box_distance(p, lower[right], upper[right])
                near = left
                far = right
                near_distance = left_distance
                far_distance = right_distance
                if right_distance < left_distance:
                    near = right
                    far = left
                    near_distance = right_distance
                    far_distance = left_distance
                if far_distance <= limit:
                    pending[size] = far
                    size += 1
                if near_distance <= limit:
                    pending[size] = near
                    size += 1
    return False


@wp.func
def _normal_support(p: wp.vec3d, normal: wp.vec3d, cosine: wp.float64,
                    tolerance: wp.float64, thin_only: bool, root: wp.int64, capacity: wp.int64,
                    lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
                    node_faces: wp.array(dtype=wp.int64), vertices: wp.array(dtype=wp.vec3d),
                    faces: wp.array(dtype=vec3l), normals: wp.array(dtype=wp.vec3d),
                    thin: wp.array(dtype=wp.bool), first_match: bool):
    found = wp.int64(-1)
    best_plane = wp.float64(1.7976931348623157e308)
    node = root
    previous = wp.int64(-1)
    while node >= wp.int64(0):
        parent = root + (node - root - wp.int64(1)) // wp.int64(2)
        if node == root:
            parent = wp.int64(-1)
        next_node = parent
        local = node - root
        if previous == parent:
            lo = lower[node] - wp.vec3d(tolerance)
            hi = upper[node] + wp.vec3d(tolerance)
            if p[0] >= lo[0] and p[1] >= lo[1] and p[2] >= lo[2] and p[0] <= hi[0] and p[1] <= hi[1] and p[2] <= hi[2]:
                if local < capacity - wp.int64(1):
                    next_node = root + wp.int64(2) * local + wp.int64(1)
                else:
                    face_id = node_faces[node]
                    if face_id >= wp.int64(0):
                        n = normals[face_id]
                        if (not thin_only or thin[face_id]) and wp.dot(n, normal) >= cosine:
                            f = faces[face_id]
                            a = vertices[f[0]]
                            b = vertices[f[1]]
                            c = vertices[f[2]]
                            # The expanded BVH is only a broad phase. Retain
                            # the exact original triangle box and support test.
                            exact_lo = wp.min(wp.min(a, b), c) - wp.vec3d(tolerance)
                            exact_hi = wp.max(wp.max(a, b), c) + wp.vec3d(tolerance)
                            plane = wp.abs(wp.dot(p - a, n))
                            if (p[0] >= exact_lo[0] and p[1] >= exact_lo[1] and p[2] >= exact_lo[2]
                                    and p[0] <= exact_hi[0] and p[1] <= exact_hi[1] and p[2] <= exact_hi[2]
                                    and plane <= tolerance
                                    and wp.dot(wp.cross(b-a, p-a), n) / wp.length(b-a) >= -tolerance
                                    and wp.dot(wp.cross(c-b, p-b), n) / wp.length(c-b) >= -tolerance
                                    and wp.dot(wp.cross(a-c, p-c), n) / wp.length(a-c) >= -tolerance):
                                if first_match:
                                    return face_id
                                if plane < best_plane or (plane == best_plane and (found < wp.int64(0) or face_id < found)):
                                    found = face_id
                                    best_plane = plane
        elif previous == root + wp.int64(2) * local + wp.int64(1):
            next_node = previous + wp.int64(1)
        previous = node
        node = next_node
    return found


@wp.func
def _triangle_valid(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, root: wp.int64, capacity: wp.int64,
                    lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
                    node_faces: wp.array(dtype=wp.int64), vertices: wp.array(dtype=wp.vec3d),
                    faces: wp.array(dtype=vec3l), normals: wp.array(dtype=wp.vec3d),
                    thin: wp.array(dtype=wp.bool), distance_limit_squared: wp.float64,
                    cosine: wp.float64, tolerance: wp.float64, check_distance: bool, check_normal: bool):
    center = (a + b + c) / wp.float64(3.0)
    cross = wp.cross(b - a, c - a)
    area = wp.length(cross)
    if check_normal and not (area > wp.float64(0.0)):
        return False
    q, center_distance, center_id = _nearest(
        center, root, capacity, lower, upper, node_faces, vertices, faces)
    if center_id < wp.int64(0):
        return False
    if check_distance and not (center_distance <= distance_limit_squared):
        return False
    if check_normal:
        unit_normal = cross / area
        if not (wp.dot(unit_normal, normals[center_id]) >= cosine):
            support = _normal_support(
                center, unit_normal, cosine, tolerance, center_distance > tolerance * tolerance,
                root, capacity, lower, upper, node_faces, vertices, faces, normals, thin, True)
            if support < wp.int64(0):
                return False
            center_id = support
    if check_distance:
        for sample in range(6):
            p = (a + b) * wp.float64(0.5)
            if sample == 1:
                p = (b + c) * wp.float64(0.5)
            elif sample == 2:
                p = (c + a) * wp.float64(0.5)
            elif sample == 3:
                p = a
            elif sample == 4:
                p = b
            elif sample == 5:
                p = c
            if not _within_distance(p, distance_limit_squared, center_id,
                    root, capacity, lower, upper, node_faces, vertices, faces):
                return False
    return True


@wp.kernel
def _validate_mask(triangles: wp.array(dtype=wp.vec3d), labels: wp.array(dtype=wp.int64),
                   roots: wp.array(dtype=wp.int64), capacities: wp.array(dtype=wp.int64),
                   lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
                   node_faces: wp.array(dtype=wp.int64), vertices: wp.array(dtype=wp.vec3d),
                   faces: wp.array(dtype=vec3l), normals: wp.array(dtype=wp.vec3d), thin: wp.array(dtype=wp.bool),
                   distance_limit_squared: wp.float64, cosine: wp.float64, tolerance: wp.float64,
                   check_distance: bool, check_normal: bool, valid: wp.array(dtype=wp.bool)):
    i = wp.tid()
    label = labels[i]
    valid[i] = _triangle_valid(triangles[3 * i], triangles[3 * i + 1], triangles[3 * i + 2],
        roots[label], capacities[label], lower, upper, node_faces, vertices, faces, normals, thin,
        distance_limit_squared, cosine, tolerance, check_distance, check_normal)


@wp.func
def _validation_slot(f: vec3l, label: wp.int64, mask: wp.int64):
    # Unsigned multiplication wraps by definition; the complete integer key
    # is checked below, so hashing cannot introduce an incorrect cache hit.
    mixed = ((wp.uint64(f[0]) * wp.uint64(73856093))
             ^ (wp.uint64(f[1]) * wp.uint64(19349663))
             ^ (wp.uint64(f[2]) * wp.uint64(83492791))
             ^ (wp.uint64(label) * wp.uint64(2654435761)))
    mixed = mixed ^ (mixed >> wp.uint64(32))
    # Cache slots are bounded by 2**22, and Warp's atomic index overload is
    # int32 even though ordinary array indexing also accepts int64.
    return int(mixed & wp.uint64(mask))


@wp.kernel
def _validation_cache_lookup(query_faces: wp.array(dtype=vec3l), labels: wp.array(dtype=wp.int64),
                             cached_faces: wp.array(dtype=vec3l), cached_labels: wp.array(dtype=wp.int64),
                             cached_valid: wp.array(dtype=wp.bool), owners: wp.array(dtype=wp.int32),
                             slot_mask: wp.int64, misses: wp.array(dtype=wp.int32),
                             miss_count: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.bool)):
    i = wp.tid()
    f = query_faces[i]
    label = labels[i]
    slot = _validation_slot(f, label, slot_mask)
    if cached_labels[slot] == label:
        old = cached_faces[slot]
        if old[0] == f[0] and old[1] == f[1] and old[2] == f[2]:
            valid[i] = cached_valid[slot]
            return
    row = wp.atomic_add(miss_count, 0, 1)
    misses[row] = i
    wp.atomic_min(owners, slot, i)


@wp.kernel
def _validate_cache_misses(triangles: wp.array(dtype=wp.vec3d),
                     labels: wp.array(dtype=wp.int64), roots: wp.array(dtype=wp.int64),
                     capacities: wp.array(dtype=wp.int64), lower: wp.array(dtype=wp.vec3d),
                     upper: wp.array(dtype=wp.vec3d), node_faces: wp.array(dtype=wp.int64),
                     vertices: wp.array(dtype=wp.vec3d), faces: wp.array(dtype=vec3l),
                     normals: wp.array(dtype=wp.vec3d), thin: wp.array(dtype=wp.bool),
                     distance_limit_squared: wp.float64, cosine: wp.float64, tolerance: wp.float64,
                     check_distance: bool, check_normal: bool,
                     misses: wp.array(dtype=wp.int32), miss_count: wp.array(dtype=wp.int32),
                     valid: wp.array(dtype=wp.bool)):
    row = wp.tid()
    if row >= miss_count[0]:
        return
    i = misses[row]
    label = labels[i]
    valid[i] = _triangle_valid(triangles[3 * i], triangles[3 * i + 1], triangles[3 * i + 2],
        roots[label], capacities[label], lower, upper, node_faces, vertices, faces, normals, thin,
        distance_limit_squared, cosine, tolerance, check_distance, check_normal)


@wp.kernel
def _store_validation_cache(query_faces: wp.array(dtype=vec3l), labels: wp.array(dtype=wp.int64),
                            cached_faces: wp.array(dtype=vec3l), cached_labels: wp.array(dtype=wp.int64),
                            cached_valid: wp.array(dtype=wp.bool), owners: wp.array(dtype=wp.int32),
                            slot_mask: wp.int64, valid: wp.array(dtype=wp.bool)):
    i = wp.tid()
    f = query_faces[i]
    label = labels[i]
    slot = _validation_slot(f, label, slot_mask)
    if owners[slot] == i:
        cached_faces[slot] = f
        cached_labels[slot] = label
        cached_valid[slot] = valid[i]
        owners[slot] = int(2147483647)


@wp.kernel
def _reduce_bounds(parents: wp.array(dtype=wp.int64), left: wp.array(dtype=wp.int64),
                   lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    p = parents[i]
    child = left[i]
    lower[p] = wp.min(lower[child], lower[child + wp.int64(1)])
    upper[p] = wp.max(upper[child], upper[child + wp.int64(1)])


@wp.kernel
def _query(points: wp.array(dtype=wp.vec3d), labels: wp.array(dtype=wp.int64),
           roots: wp.array(dtype=wp.int64), capacities: wp.array(dtype=wp.int64),
           lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
           node_faces: wp.array(dtype=wp.int64), vertices: wp.array(dtype=wp.vec3d), faces: wp.array(dtype=vec3l),
           positions: wp.array(dtype=wp.vec3d), distances: wp.array(dtype=wp.float64), ids: wp.array(dtype=wp.int64)):
    i = wp.tid()
    label = labels[i]
    q, d, f = _nearest(points[i], roots[label], capacities[label], lower, upper, node_faces, vertices, faces)
    positions[i] = q
    distances[i] = d
    ids[i] = f


@wp.kernel
def _validate(triangles: wp.array(dtype=wp.vec3d), labels: wp.array(dtype=wp.int64),
              roots: wp.array(dtype=wp.int64), capacities: wp.array(dtype=wp.int64),
              lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
              node_faces: wp.array(dtype=wp.int64), vertices: wp.array(dtype=wp.vec3d), faces: wp.array(dtype=vec3l),
              normals: wp.array(dtype=wp.vec3d), thin: wp.array(dtype=wp.bool),
              distance_limit_squared: wp.float64, cosine: wp.float64, tolerance: wp.float64,
              check_distance: bool, check_normal: bool,
              valid: wp.array(dtype=wp.bool), distance_valid: wp.array(dtype=wp.bool),
              normal_valid: wp.array(dtype=wp.bool), maximum_squared: wp.array(dtype=wp.float64),
              alignments: wp.array(dtype=wp.float64), areas: wp.array(dtype=wp.float64),
              centroid_ids: wp.array(dtype=wp.int64), resolved: wp.array(dtype=wp.bool)):
    i = wp.tid()
    a = triangles[3*i]
    b = triangles[3*i+1]
    c = triangles[3*i+2]
    center = (a + b + c) / wp.float64(3.0)
    root = roots[labels[i]]
    capacity = capacities[labels[i]]
    worst = wp.float64(0.0)
    center_distance = wp.float64(0.0)
    center_id = wp.int64(-1)
    for sample in range(7):
        p = center
        if sample == 0:
            p = a
        elif sample == 1:
            p = b
        elif sample == 2:
            p = c
        elif sample == 3:
            p = (a+b) * wp.float64(0.5)
        elif sample == 4:
            p = (b+c) * wp.float64(0.5)
        elif sample == 5:
            p = (c+a) * wp.float64(0.5)
        q, d, f = _nearest(p, root, capacity, lower, upper, node_faces, vertices, faces)
        worst = wp.max(worst, d)
        if sample == 6:
            center_id = f
            center_distance = d
    cross = wp.cross(b-a, c-a)
    area = wp.length(cross)
    alignment = wp.float64(1.0)
    normal_ok = True
    tie_resolved = False
    if check_normal:
        alignment = wp.float64(-1.0)
        if area > wp.float64(0.0) and center_id >= wp.int64(0):
            unit_normal = cross / area
            alignment = wp.dot(unit_normal, normals[center_id])
            if alignment < cosine:
                support = _normal_support(center, unit_normal, cosine, tolerance, center_distance > tolerance*tolerance,
                    root, capacity, lower, upper, node_faces, vertices, faces, normals, thin, False)
                if support >= wp.int64(0):
                    center_id = support
                    alignment = wp.dot(unit_normal, normals[support])
                    tie_resolved = True
        normal_ok = alignment >= cosine and area > wp.float64(0.0)
    distance_ok = center_id >= wp.int64(0) and (not check_distance or worst <= distance_limit_squared)
    valid[i] = distance_ok and normal_ok
    distance_valid[i] = distance_ok
    normal_valid[i] = normal_ok
    maximum_squared[i] = worst
    alignments[i] = alignment
    areas[i] = area
    centroid_ids[i] = center_id
    resolved[i] = tie_resolved


@wp.kernel
def _validate_with_ties(triangles: wp.array(dtype=wp.vec3d), labels: wp.array(dtype=wp.int64),
                        roots: wp.array(dtype=wp.int64), capacities: wp.array(dtype=wp.int64),
                        lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
                        node_faces: wp.array(dtype=wp.int64), vertices: wp.array(dtype=wp.vec3d),
                        faces: wp.array(dtype=vec3l), normals: wp.array(dtype=wp.vec3d),
                        thin: wp.array(dtype=wp.bool), distance_limit_squared: wp.float64,
                        cosine: wp.float64, tolerance: wp.float64,
                        check_distance: bool, check_normal: bool,
                        valid: wp.array(dtype=wp.bool), resolved: wp.array(dtype=wp.bool)):
    # Successful final validation needs the same mask and resolved-normal
    # count as _validate, but does not consume seven exact nearest distances.
    i = wp.tid()
    a = triangles[3 * i]
    b = triangles[3 * i + 1]
    c = triangles[3 * i + 2]
    center = (a + b + c) / wp.float64(3.0)
    root = roots[labels[i]]
    capacity = capacities[labels[i]]
    q, center_distance, center_id = _nearest(
        center, root, capacity, lower, upper, node_faces, vertices, faces)
    cross = wp.cross(b - a, c - a)
    area = wp.length(cross)
    normal_ok = True
    tie_resolved = False
    if check_normal:
        alignment = wp.float64(-1.0)
        if area > wp.float64(0.0) and center_id >= wp.int64(0):
            unit_normal = cross / area
            alignment = wp.dot(unit_normal, normals[center_id])
            if alignment < cosine:
                support = _normal_support(
                    center, unit_normal, cosine, tolerance,
                    center_distance > tolerance * tolerance,
                    root, capacity, lower, upper, node_faces, vertices,
                    faces, normals, thin, False)
                if support >= wp.int64(0):
                    center_id = support
                    alignment = wp.dot(unit_normal, normals[support])
                    tie_resolved = True
        normal_ok = alignment >= cosine and area > wp.float64(0.0)
    # Resolve normals even when distance validation fails: the public summary
    # promises the complete details path's tie flags for every input row.
    resolved[i] = tie_resolved
    distance_ok = center_id >= wp.int64(0)
    if check_distance:
        distance_ok = distance_ok and center_distance <= distance_limit_squared
        if distance_ok and normal_ok:
            for sample in range(6):
                p = a
                if sample == 1:
                    p = b
                elif sample == 2:
                    p = c
                elif sample == 3:
                    p = (a + b) * wp.float64(0.5)
                elif sample == 4:
                    p = (b + c) * wp.float64(0.5)
                elif sample == 5:
                    p = (c + a) * wp.float64(0.5)
                if not _within_distance(p, distance_limit_squared, center_id,
                        root, capacity, lower, upper, node_faces, vertices, faces):
                    distance_ok = False
                    break
    valid[i] = distance_ok and normal_ok


class CudaReferencePatchProjector(_ReferencePatchProjector):
    is_cuda_backend = True
    backend_name = 'cuda_float64_bvh'

    def __init__(self, vertices, faces, face_patch_ids, device='cuda'):
        super().__init__(vertices, faces, face_patch_ids)
        if not np.isfinite(self.normals).all():
            raise ValueError('CUDA reference normals must be finite.')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA reference queries require a CUDA device.')
        wp.init()
        self.gpu_vertices = torch.as_tensor(self.vertices, device=self.device, dtype=torch.float64)
        self.gpu_faces = torch.as_tensor(self.faces, device=self.device, dtype=torch.long)
        self.gpu_normals = torch.as_tensor(self.normals, device=self.device, dtype=torch.float64)
        label_values, local_labels, counts = np.unique(self.face_patch_ids, return_inverse=True, return_counts=True)
        self.label_values = torch.as_tensor(label_values, device=self.device, dtype=torch.long)
        labels = torch.as_tensor(local_labels, device=self.device, dtype=torch.long)
        label_count = len(label_values)
        capacities = np.zeros(label_count, dtype=np.int64)
        present = np.flatnonzero(counts)
        for label in present:
            capacities[label] = 1 << (int(counts[label]) - 1).bit_length()
        node_counts = np.where(capacities > 0, 2*capacities-1, 0)
        # Depth-first traversal needs at most depth + 1 pending nodes.
        if int(capacities.max()).bit_length() > 32:
            raise ValueError('Reference patch exceeds the CUDA traversal stack capacity.')
        bases = np.r_[0, np.cumsum(node_counts)[:-1]].astype(np.int64)
        roots = np.where(counts > 0, bases, -1)
        self.roots = torch.as_tensor(roots, device=self.device)
        self.capacities = torch.as_tensor(capacities, device=self.device)
        self.lower = torch.full((int(node_counts.sum()), 3), torch.inf, dtype=torch.float64, device=self.device)
        self.upper = torch.full_like(self.lower, -torch.inf)
        self.node_faces = torch.full((len(self.lower),), -1, dtype=torch.long, device=self.device)
        triangles = self.gpu_vertices[self.gpu_faces]
        centers = triangles.mean(dim=1)
        patch_min = torch.full((label_count,3), torch.inf, dtype=torch.float64, device=self.device)
        patch_max = torch.full_like(patch_min, -torch.inf)
        patch_min.scatter_reduce_(0, labels[:,None].expand(-1,3), centers, reduce='amin', include_self=True)
        patch_max.scatter_reduce_(0, labels[:,None].expand(-1,3), centers, reduce='amax', include_self=True)
        xyz = (((centers-patch_min[labels]) / (patch_max[labels]-patch_min[labels]).clamp_min(1e-300)) * 1023).long()
        morton = torch.zeros(len(labels), dtype=torch.long, device=self.device)
        for bit in range(10):
            for axis in range(3):
                morton |= ((xyz[:,axis] >> bit) & 1) << (3*bit+axis)
        order = torch.argsort(labels*(1 << 30) + morton, stable=True)
        offsets = torch.as_tensor(np.r_[0,np.cumsum(counts)[:-1]], device=self.device)
        local_ids = torch.arange(len(labels), device=self.device) - offsets[labels[order]]
        leaves = self.roots[labels[order]] + self.capacities[labels[order]] - 1 + local_ids
        self.node_faces[leaves] = order
        self.lower[leaves] = triangles[order].amin(dim=1) - self.numeric_tolerance
        self.upper[leaves] = triangles[order].amax(dim=1) + self.numeric_tolerance
        self.thin = torch.zeros(len(labels), dtype=torch.bool, device=self.device)
        thin_ids = np.concatenate(list(self.thin_patch_faces.values()))
        self.thin[torch.as_tensor(thin_ids, device=self.device)] = True
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            for level in range(1, int(capacities.max()).bit_length()):
                parent_parts, left_parts = [], []
                for label in present:
                    capacity = int(capacities[label])
                    if capacity >= 1 << level:
                        local = np.arange((capacity >> level)-1, (capacity >> (level-1))-1, dtype=np.int64)
                        parent_parts.append(local + bases[label])
                        left_parts.append(2*local + bases[label] + 1)
                parents = torch.as_tensor(np.concatenate(parent_parts), device=self.device)
                left = torch.as_tensor(np.concatenate(left_parts), device=self.device)
                wp.launch(_reduce_bounds, dim=len(parents), inputs=[wp.from_torch(parents), wp.from_torch(left),
                          wp.from_torch(self.lower, dtype=wp.vec3d), wp.from_torch(self.upper, dtype=wp.vec3d)])
        self._reference_args = [wp.from_torch(self.roots), wp.from_torch(self.capacities),
            wp.from_torch(self.lower,dtype=wp.vec3d), wp.from_torch(self.upper,dtype=wp.vec3d),
            wp.from_torch(self.node_faces), wp.from_torch(self.gpu_vertices,dtype=wp.vec3d),
            wp.from_torch(self.gpu_faces,dtype=vec3l)]
        self._normal_args = [wp.from_torch(self.gpu_normals, dtype=wp.vec3d), wp.from_torch(self.thin)]

    def make_triangle_cache(self, vertices, maximum_deviation, normal_degrees):
        return _CudaTriangleValidationCache(vertices, self, maximum_deviation, normal_degrees)

    def _validation_limits(self, maximum_deviation, normal_degrees):
        cosine = -1. if normal_degrees is None else float(np.cos(np.deg2rad(normal_degrees))-1e-12)
        limit = 0. if maximum_deviation is None else (float(maximum_deviation)+self.numeric_tolerance)**2
        return cosine, limit

    def _labels(self, labels, size):
        if labels.shape != (size,) or labels.device != self.gpu_vertices.device or labels.dtype != torch.long:
            raise ValueError('CUDA projection requires one CUDA int64 label per query.')
        if size:
            locations = torch.searchsorted(self.label_values, labels.contiguous())
            valid = (locations < len(self.label_values)) & (self.label_values[locations.clamp_max(len(self.label_values)-1)] == labels)
            if not bool(valid.all()):
                raise ValueError('Projection references an unknown reference patch.')
            return locations
        return labels.contiguous()

    def query_torch(self, points, patch_ids):
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype != torch.float64 or points.device != self.gpu_vertices.device:
            raise ValueError('CUDA projection needs float64 (n, 3) points on the reference device.')
        labels = self._labels(patch_ids, len(points))
        if not bool(torch.isfinite(points).all()):
            raise ValueError('Patch projection points must be finite.')
        positions = torch.empty((len(points),3),dtype=points.dtype,device=points.device)
        squared = torch.empty(len(points), dtype=torch.float64, device=points.device)
        ids = torch.empty(len(points), dtype=torch.long, device=points.device)
        if len(points):
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(points.device))):
                wp.launch(_query,dim=len(points),inputs=[wp.from_torch(points.contiguous(),dtype=wp.vec3d),
                    wp.from_torch(labels),*self._reference_args,wp.from_torch(positions,dtype=wp.vec3d),
                    wp.from_torch(squared),wp.from_torch(ids)])
            if not bool((ids >= 0).all()):
                raise ValueError('Projection query exceeds finite float64 distance range.')
        return positions, squared, ids

    def query_numpy(self, points, patch_ids):
        points = torch.as_tensor(np.asarray(points,dtype=np.float64),device=self.device)
        labels = torch.as_tensor(np.asarray(patch_ids,dtype=np.int64),device=self.device)
        return tuple(array.cpu().numpy() for array in self.query_torch(points, labels))

    def project(self, points, patch_ids):
        positions, _, ids = self.query_torch(points,patch_ids)
        return positions, ids

    def _triangle_validation_labels(self, triangles, patch_ids):
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError(
                f"CUDA validation needs (n,3,3) triangles, got shape={tuple(triangles.shape)}."
            )
        if triangles.dtype != torch.float64:
            raise ValueError(
                f"CUDA validation needs float64 triangles, got dtype={triangles.dtype}."
            )
        if triangles.device != self.gpu_vertices.device:
            raise ValueError(
                f"CUDA validation device mismatch: triangles={triangles.device}, "
                f"reference={self.gpu_vertices.device}."
            )
        labels = self._labels(patch_ids,len(triangles))
        if not bool(torch.isfinite(triangles).all()):
            raise ValueError('Patch projection points must be finite.')
        return labels

    def valid_triangles_with_ties(self, triangles, patch_ids, maximum_deviation,
                                  maximum_normal_deviation_degrees):
        """Return exact validity and normal-tie flags without distance metrics.

        Both results stay on the query CUDA device. Call ``valid_triangles``
        with ``return_details=True`` when exact distance/angle metrics are
        required, such as reporting the reason for a validation failure.
        """
        labels = self._triangle_validation_labels(triangles, patch_ids)
        count = len(triangles)
        mask = torch.empty(count, device=self.device, dtype=torch.bool)
        ties = torch.empty_like(mask)
        if count:
            cosine, limit = self._validation_limits(maximum_deviation, maximum_normal_deviation_degrees)
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(triangles.device))):
                wp.launch(_validate_with_ties, dim=count, inputs=[
                    wp.from_torch(triangles.reshape(-1, 3).contiguous(), dtype=wp.vec3d),
                    wp.from_torch(labels), *self._reference_args, *self._normal_args,
                    limit, cosine, self.normal_tie_tolerance,
                    maximum_deviation is not None, maximum_normal_deviation_degrees is not None,
                    wp.from_torch(mask), wp.from_torch(ties)])
        return mask, ties

    def valid_triangles(self, triangles, patch_ids, maximum_deviation,
                        maximum_normal_deviation_degrees, return_details=False):
        labels = self._triangle_validation_labels(triangles, patch_ids)
        count = len(triangles)
        mask = torch.empty(count, device=self.device, dtype=torch.bool)
        cosine, limit = self._validation_limits(maximum_deviation, maximum_normal_deviation_degrees)
        if not return_details:
            if count:
                with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(triangles.device))):
                    wp.launch(_validate_mask, dim=count, inputs=[
                        wp.from_torch(triangles.reshape(-1, 3).contiguous(), dtype=wp.vec3d),
                        wp.from_torch(labels), *self._reference_args, *self._normal_args,
                        limit, cosine, self.normal_tie_tolerance,
                        maximum_deviation is not None, maximum_normal_deviation_degrees is not None,
                        wp.from_torch(mask)])
            return mask
        distance_mask, normal_mask, ties = [torch.empty(count,device=self.device,dtype=torch.bool) for _ in range(3)]
        squared, alignment, area = [torch.empty(count,device=self.device,dtype=torch.float64) for _ in range(3)]
        ids = torch.empty(count,device=self.device,dtype=torch.long)
        if count:
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(triangles.device))):
                wp.launch(_validate,dim=count,inputs=[wp.from_torch(triangles.reshape(-1,3).contiguous(),dtype=wp.vec3d),
                    wp.from_torch(labels),*self._reference_args,*self._normal_args,limit,cosine,self.normal_tie_tolerance,
                    maximum_deviation is not None,maximum_normal_deviation_degrees is not None,
                    wp.from_torch(mask),wp.from_torch(distance_mask),wp.from_torch(normal_mask),
                    wp.from_torch(squared),wp.from_torch(alignment),wp.from_torch(area),wp.from_torch(ids),wp.from_torch(ties)])
        if return_details:
            if not count:
                return mask, {}
            return mask, {'distance_valid':distance_mask.cpu().numpy(),'normal_valid':normal_mask.cpu().numpy(),
                'maximum_sample_distance':squared.clamp_min(0).sqrt().cpu().numpy(),
                'normal_deviation_degrees':torch.rad2deg(torch.acos(alignment.clamp(-1,1))).cpu().numpy(),
                'double_area':area.cpu().numpy(),'centroid_reference_face_ids':ids.cpu().numpy(),
                'normal_reference_ties_resolved':ties.cpu().numpy()}
        return mask
