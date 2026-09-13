"""Index-preserving I/O for the CadMesh partition to PaMO remeshing handoff."""

from dataclasses import dataclass
from contextlib import ExitStack
from itertools import islice
import json
from pathlib import Path
import re

import numpy as np


SURFACE_TYPES = {
    "Unknown": 0, "Plane": 1, "Cylinder": 2, "Cone": 3,
    "Sphere": 4, "Torus": 5, "Freeform": 6,
}
FEATURE_ROLES = {"Ordinary": 0, "Fillet": 1, "FilletCandidate": 2}
SURFACE_COLORS = np.array([
    [74, 79, 86], [74, 144, 226], [53, 183, 121], [239, 155, 56],
    [160, 108, 213], [230, 105, 158], [145, 153, 164],
], dtype=np.int64)
ROLE_COLORS = np.array([[150, 161, 177], [245, 145, 45]], dtype=np.int64)
_PLY_BLOCK_ROWS = 16384
_PLY_INTEGER_TEXT = re.compile(r"[+\-0-9\s]*\Z", re.ASCII)
_PLY_WIDE_INTEGER = re.compile(r"[0-9]{19}")


def _unique_edge_rows(edges):
    """Lexicographic edge order without NumPy's structured-row unique sort."""
    if not len(edges):
        return np.empty((0, 2), dtype=np.int64)
    ordered = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    keep = np.concatenate(([True], np.any(ordered[1:] != ordered[:-1], axis=1)))
    return ordered[keep]


def _edges(values, name):
    array = np.asarray(values)
    if array.size == 0 and array.shape in ((0,), (0, 2)):
        return np.empty((0, 2), dtype=np.int64)
    if (array.ndim != 2 or array.shape[1] != 2
            or not np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"{name} must contain integer vertex pairs.")
    array = np.sort(np.asarray(array, dtype=np.int64), axis=1)
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError(f"{name} contains a self-edge.")
    return array


@dataclass
class PartitionInput:
    source_directory: Path
    vertices: np.ndarray
    faces: np.ndarray
    face_patch_ids: np.ndarray
    report: dict
    hard_edges: np.ndarray
    smooth_edges: np.ndarray
    corner_vertex_ids: np.ndarray

    @property
    def constraint_edges(self):
        return _unique_edge_rows(np.vstack((self.hard_edges, self.smooth_edges)))


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _ids(values, name, upper_bound=None, allow_empty=True):
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be an integer list.")
    values = np.asarray([_integer(value, name) for value in values], dtype=np.int64)
    if values.ndim != 1 or (not allow_empty and not len(values)):
        raise ValueError(f"{name} must be a nonempty one-dimensional list.")
    # Exported memberships and incident IDs are normally already sorted. Avoid
    # sorting hundreds of thousands of IDs again just to establish uniqueness.
    if (len(values) > 1 and not np.all(values[1:] > values[:-1])
            and len(np.unique(values)) != len(values)):
        raise ValueError(f"{name} contains duplicate IDs.")
    if np.any(values < 0) or (upper_bound is not None and np.any(values >= upper_bound)):
        raise ValueError(f"{name} contains an out-of-range ID.")
    return values


def _reject_json_constant(value):
    raise ValueError(f"Report contains a non-finite JSON number: {value}.")


def _validate_json_numbers(value):
    # JSON's parser can also produce infinity from valid exponent syntax such
    # as 1e999, which parse_constant alone does not catch.
    pending = [value]
    while pending:
        current = pending.pop()
        values = current.values() if isinstance(current, dict) else current
        for item in values:
            if isinstance(item, (dict, list)):
                pending.append(item)
            elif isinstance(item, float) and not np.isfinite(item):
                raise ValueError("Partition report contains a non-finite number.")


def _ply_blocks(stream, count, name):
    # Keep record boundaries so a missing/blank line cannot be silently filled
    # using the next element's data by a bulk whitespace parser.
    for start in range(0, count, _PLY_BLOCK_ROWS):
        size = min(_PLY_BLOCK_ROWS, count - start)
        lines = list(islice(stream, size))
        if len(lines) != size:
            raise ValueError(f"Short PLY {name} records starting at {start}.")
        yield start, lines


def _read_vertex_block(lines, properties, columns, start):
    if len(properties) == 3:
        try:
            values = np.loadtxt(lines, dtype=np.float64, comments=None, ndmin=2)
        except ValueError:
            # Python float accepts some spellings NumPy does not. Keep the
            # original parser for these and for files with extra properties.
            pass
        else:
            if values.shape != (len(lines), 3):
                raise ValueError(f"Invalid PLY vertex records starting at {start}.")
            return values[:, columns]
    values = np.empty((len(lines), 3), dtype=np.float64)
    for local_row, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) != len(properties):
            raise ValueError(f"Invalid PLY vertex record {start + local_row}.")
        values[local_row] = [float(tokens[column]) for column in columns]
    return values


def _read_face_block(lines, properties, index_name, start):
    scalar_names = {"patch_id", "primitive_type", "feature_role", "red", "green", "blue"}
    canonical = all((prop == index_name and is_list)
                    or (prop in scalar_names and not is_list)
                    for prop, is_list in properties)
    text = "".join(lines) if canonical else ""
    if (canonical and _PLY_INTEGER_TEXT.fullmatch(text) is not None
            and _PLY_WIDE_INTEGER.search(text) is None):
        # The lexical guard prevents loadtxt's compatibility conversion from
        # accepting floating-point spellings for integer mesh indices.
        try:
            values = np.loadtxt(lines, dtype=np.int64, comments=None, ndmin=2)
        except ValueError:
            pass
        else:
            if values.shape != (len(lines), len(properties) + 3):
                raise ValueError(f"Invalid PLY face records starting at {start}.")
            columns, cursor = {}, 0
            for prop, is_list in properties:
                columns[prop] = cursor
                cursor += 4 if is_list else 1
            index_column = columns[index_name]
            if np.any(values[:, index_column] != 3):
                raise ValueError("Remeshing requires triangle faces; PLY contains a polygon.")
            roles = (values[:, columns["feature_role"]] if "feature_role" in columns
                     else np.zeros(len(lines), dtype=np.int64))
            return (values[:, index_column + 1:index_column + 4],
                    values[:, columns["patch_id"]], values[:, columns["primitive_type"]], roles)
    # Preserve arbitrary scalar properties and additional list properties in
    # external handoff files, including the original per-record validation.
    faces = np.empty((len(lines), 3), dtype=np.int64)
    labels = np.empty(len(lines), dtype=np.int64)
    types = np.empty(len(lines), dtype=np.int64)
    roles = np.empty(len(lines), dtype=np.int64)
    for local_row, line in enumerate(lines):
        row = start + local_row
        tokens = line.split()
        values, cursor = {}, 0
        for prop, is_list in properties:
            if cursor >= len(tokens):
                raise ValueError(f"Short PLY face record {row}.")
            if is_list:
                size = int(tokens[cursor])
                if size < 0 or cursor + size >= len(tokens):
                    raise ValueError(f"Invalid PLY list in face {row}.")
                values[prop] = tokens[cursor + 1:cursor + 1 + size]
                cursor += size + 1
            else:
                values[prop] = tokens[cursor]
                cursor += 1
        if cursor != len(tokens):
            raise ValueError(f"Excess PLY properties in face {row}.")
        if len(values[index_name]) != 3:
            raise ValueError("Remeshing requires triangle faces; PLY contains a polygon.")
        faces[local_row] = [int(value) for value in values[index_name]]
        labels[local_row] = int(values["patch_id"])
        types[local_row] = int(values["primitive_type"])
        roles[local_row] = int(values.get("feature_role", 0))
    return faces, labels, types, roles


def _read_ply(path):
    """Read bounded records into arrays; never weld or reorder any geometry."""
    from .partition_ply import read_binary_partition_ply
    binary = read_binary_partition_ply(path)
    if binary is not None:
        return binary
    with path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("Expected a PLY header.")
        elements = []
        ascii_format = False
        for line in stream:
            tokens = line.split()
            if not tokens or tokens[0] in ("comment", "obj_info"):
                continue
            if tokens[0] == "format":
                ascii_format = tokens[1:] == ["ascii", "1.0"]
            elif tokens[0] == "element" and len(tokens) == 3:
                count = int(tokens[2])
                if count < 0:
                    raise ValueError("PLY element count must be non-negative.")
                elements.append((tokens[1], count, []))
            elif tokens[0] == "property" and elements:
                if len(tokens) == 3:
                    elements[-1][2].append((tokens[-1], False))
                elif len(tokens) == 5 and tokens[1] == "list":
                    elements[-1][2].append((tokens[-1], True))
                else:
                    raise ValueError("Invalid PLY property declaration.")
            elif tokens == ["end_header"]:
                break
            else:
                raise ValueError("Unsupported PLY header declaration.")
        else:
            raise ValueError("Unterminated PLY header.")
        if not ascii_format:
            raise ValueError("Partition input requires ASCII PLY 1.0.")
        if [name for name, _, _ in elements] != ["vertex", "face"]:
            raise ValueError("Partition PLY must contain vertex then face elements.")
        vertex_count, face_count = elements[0][1], elements[1][1]
        # Even an ASCII record needs at least one byte. Reject impossible
        # header counts before allocating arrays from an untrusted header.
        if vertex_count + face_count > path.stat().st_size:
            raise ValueError("PLY element counts exceed the available payload.")
        vertices = np.empty((vertex_count, 3), dtype=np.float64)
        faces = np.empty((face_count, 3), dtype=np.int64)
        labels = np.empty(face_count, dtype=np.int64)
        types = np.empty(face_count, dtype=np.int64)
        roles = np.zeros(face_count, dtype=np.int64)
        has_roles = False
        for name, count, properties in elements:
            property_names = [prop for prop, _ in properties]
            if len(property_names) != len(set(property_names)):
                raise ValueError(f"PLY {name} contains duplicate properties.")
            if name == "vertex":
                if any(is_list for _, is_list in properties):
                    raise ValueError("PLY vertex list properties are unsupported.")
                if not all(axis in property_names for axis in ("x", "y", "z")):
                    raise ValueError("PLY vertices need x, y, and z properties.")
                columns = [property_names.index(axis) for axis in ("x", "y", "z")]
                for start, lines in _ply_blocks(stream, count, name):
                    vertices[start:start + len(lines)] = _read_vertex_block(
                        lines, properties, columns, start)
                continue
            if not all(prop in property_names for prop in ("patch_id", "primitive_type")):
                raise ValueError("PLY faces need patch_id and primitive_type properties.")
            index_name = ("vertex_indices" if "vertex_indices" in property_names
                          else "vertex_index")
            if (index_name, True) not in properties:
                raise ValueError("PLY faces need a vertex_indices list property.")
            has_roles = "feature_role" in property_names
            for start, lines in _ply_blocks(stream, count, name):
                block_faces, block_labels, block_types, block_roles = _read_face_block(
                    lines, properties, index_name, start)
                end = start + len(lines)
                faces[start:end] = block_faces
                labels[start:end] = block_labels
                types[start:end] = block_types
                roles[start:end] = block_roles
        if any(line.strip() for line in stream):
            raise ValueError("PLY contains trailing records beyond its declared counts.")
    return vertices, faces, labels, types, roles, has_roles


def _validate_geometry(vertices, faces, labels):
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Mesh vertices must have nonempty shape (N, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices contain NaN or infinity.")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Mesh triangles must have nonempty shape (M, 3).")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Mesh triangle indices must be integers.")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Mesh triangle index is out of range.")
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])
              | (faces[:, 2] == faces[:, 0])):
        raise ValueError("Mesh contains a triangle with repeated vertex indices.")
    crosses = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    if not np.isfinite(crosses).all() or np.any(np.all(crosses == 0.0, axis=1)):
        raise ValueError("Mesh contains a degenerate or numerically invalid triangle.")
    if labels.shape != (len(faces),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Mesh needs one integer patch label per triangle.")
    if np.any(labels < 0):
        raise ValueError("Mesh contains a negative patch label.")


def _topology(faces, labels, vertex_count):
    # Packed keys permit vectorized incidence and constraint checks without a
    # Python object per mesh edge. Current PLY indices are well below this bound.
    if vertex_count > 3_037_000_499:
        raise ValueError("Mesh exceeds the supported 64-bit packed edge range.")
    occurrences = np.sort(faces[:, ((0, 1), (1, 2), (2, 0))], axis=2).reshape(-1, 2)
    keys = occurrences[:, 0] * vertex_count + occurrences[:, 1]
    # One stable packed-key sort supplies both incidence order and label
    # reductions; unique(return_inverse) followed by argsort sorted twice.
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1])))
    unique = sorted_keys[starts]
    offsets = np.concatenate((starts, [len(keys)]))
    counts = np.diff(offsets)
    edge_labels = labels[order // 3]
    minimum = np.minimum.reduceat(edge_labels, starts)
    maximum = np.maximum.reduceat(edge_labels, starts)
    return {
        "keys": unique, "counts": counts, "minimum_labels": minimum,
        "maximum_labels": maximum, "order": order, "offsets": offsets,
        "required": (counts != 2) | (minimum != maximum),
    }


def _constraint_indices(edges, topology, vertex_count, name):
    if np.any(edges < 0) or np.any(edges >= vertex_count):
        raise ValueError(f"{name} contains an out-of-range vertex ID.")
    keys = edges[:, 0] * vertex_count + edges[:, 1]
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"{name} contains duplicate mesh edges.")
    indices = np.searchsorted(topology["keys"], keys)
    if np.any(indices >= len(topology["keys"])):
        raise ValueError(f"{name} contains an edge absent from the mesh.")
    if not np.array_equal(topology["keys"][indices], keys):
        raise ValueError(f"{name} contains an edge absent from the mesh.")
    return indices


def _incident_faces(topology, index):
    start, end = topology["offsets"][index:index + 2]
    return topology["order"][start:end] // 3


def _validate_constraint_classes(hard, smooth, topology, vertex_count):
    hard_indices = _constraint_indices(hard, topology, vertex_count, "Hard constraints")
    smooth_indices = _constraint_indices(smooth, topology, vertex_count, "Smooth constraints")
    if len(np.intersect1d(hard_indices, smooth_indices)):
        raise ValueError("An edge cannot be both a hard and a smooth constraint.")
    covered = np.zeros(len(topology["keys"]), dtype=bool)
    covered[hard_indices] = True
    covered[smooth_indices] = True
    if np.any(topology["required"] & ~covered):
        raise ValueError("Constraints omit an open, non-manifold, or cross-patch edge.")
    if (np.any(topology["counts"][smooth_indices] != 2)
            or np.any(topology["minimum_labels"][smooth_indices]
                      == topology["maximum_labels"][smooth_indices])):
        raise ValueError("Smooth constraints must be manifold transitions between patches.")


def _validate_surface_parameters(patch):
    target = patch.get("projection_target", "reference_mesh")
    if target not in ("analytic_surface", "reference_mesh"):
        raise ValueError("Unsupported patch projection target.")
    if target != "analytic_surface":
        return
    parameters = patch.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Analytic patches require surface parameters.")

    def vector(name, direction=False):
        value = np.asarray(parameters.get(name), dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"Analytic {name} must be a finite 3-vector.")
        if direction and not np.any(value != 0.0):
            raise ValueError(f"Analytic {name} must have a nonzero direction.")

    def positive(name):
        value = parameters.get(name)
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"Analytic {name} must be finite and positive.")
        return value

    kind = patch["type"]
    if kind == "Plane":
        vector("origin")
        vector("normal", direction=True)
    elif kind == "Sphere":
        vector("center")
        positive("radius")
    elif kind in ("Cylinder", "Cone", "Torus"):
        vector("axis_origin")
        vector("axis_direction", direction=True)
        if kind == "Cylinder":
            positive("radius")
        elif kind == "Cone":
            if positive("semi_angle_radians") >= np.pi / 2:
                raise ValueError("Cone semi-angle must be below pi/2.")
        elif positive("major_radius") <= positive("minor_radius"):
            raise ValueError("Analytic ring torus requires major_radius > minor_radius.")
    else:
        raise ValueError("Unknown and Freeform patches must use the reference mesh.")


def load_partition(path):
    """Load a validated CadMesh PLY/JSON pair without processing mesh indices."""
    path = Path(path).expanduser().resolve()
    directory = path if path.is_dir() else path.parent
    ply_path = directory / "patch_result.ply" if path.is_dir() else path
    if ply_path.name != "patch_result.ply":
        raise ValueError("Partition input must be a directory or its patch_result.ply.")
    with (directory / "patch_report.json").open("r", encoding="utf-8-sig") as stream:
        report = json.load(stream, parse_constant=_reject_json_constant)
    if (not isinstance(report, dict) or report.get("schema") != "cadmesh.remesh_handoff"
            or _integer(report.get("schema_version"), "Schema version") != 1):
        raise ValueError("Unsupported partition report schema; expected cadmesh.remesh_handoff v1.")
    _validate_json_numbers(report)
    if not isinstance(report.get("indexing"), dict):
        raise ValueError("Partition report needs an indexing object.")
    if (report.get("partition_valid") is not True
            or _integer(report.get("indexing", {}).get("base"), "Index base") != 0):
        raise ValueError("Partition report must declare a valid partition with zero-based indices.")
    vertices, faces, labels, types, roles, has_roles = _read_ply(ply_path)
    _validate_geometry(vertices, faces, labels)
    mesh_report = report.get("mesh", {})
    if not isinstance(mesh_report, dict):
        raise ValueError("Partition report needs a mesh object.")
    if (_integer(mesh_report.get("vertex_count"), "Vertex count") != len(vertices)
            or _integer(mesh_report.get("triangle_count"), "Triangle count") != len(faces)):
        raise ValueError("PLY vertex/triangle counts disagree with the partition report.")
    patches = report.get("patches")
    if (not isinstance(patches, list) or not patches
            or any(not isinstance(patch, dict) for patch in patches)):
        raise ValueError("Partition report needs nonempty patches.")
    if [_integer(patch.get("id"), "Patch ID") for patch in patches] != list(range(len(patches))):
        raise ValueError("Partition IDs must be unique, contiguous, and ordered from zero.")
    if np.any(labels >= len(patches)):
        raise ValueError("PLY contains a patch ID absent from the report.")
    seen = np.zeros(len(faces), dtype=bool)
    for patch in patches:
        patch_id = patch["id"]
        members = _ids(patch.get("triangle_ids"), "Patch triangle_ids", len(faces), False)
        if (_integer(patch.get("triangle_count"), "Patch triangle count") != len(members)
                or np.any(seen[members])):
            raise ValueError("Patch triangle counts disagree or membership is duplicated.")
        seen[members] = True
        if np.any(labels[members] != patch_id):
            raise ValueError("PLY patch labels disagree with JSON triangle membership.")
        kind = patch.get("type")
        role = patch.get("feature_role", "Ordinary")
        if kind not in SURFACE_TYPES or role not in FEATURE_ROLES:
            raise ValueError("Partition contains an unsupported surface type or feature role.")
        _validate_surface_parameters(patch)
        if "feature_role" in patch and not has_roles:
            raise ValueError("Semantic partition report requires feature_role in the PLY.")
        if (np.any(types[members] != SURFACE_TYPES[kind])
                or np.any(roles[members] != FEATURE_ROLES[role])):
            raise ValueError("PLY surface type or feature role disagrees with the report.")
        support = _ids(patch.get("support_patch_ids", []), "Support patch IDs", len(patches))
        if patch_id in support:
            raise ValueError("A patch cannot support its own fillet role.")
    if not np.all(seen):
        raise ValueError("JSON patch membership does not cover every PLY face.")
    topology = _topology(faces, labels, len(vertices))
    if mesh_report.get("edge_count", len(topology["keys"])) != len(topology["keys"]):
        raise ValueError("Mesh edge count disagrees with the partition report.")
    constraints = report.get("constraints", {})
    records = report.get("constraint_edges")
    if not isinstance(constraints, dict):
        raise ValueError("Partition report needs a constraints object.")
    if (not isinstance(records, list)
            or any(not isinstance(record, dict) for record in records)):
        raise ValueError("Partition report needs explicit constraint edge records.")
    record_ids = _ids([record.get("id") for record in records], "Constraint edge IDs")
    declared_ids = _ids(constraints.get("edge_ids"), "Declared constraint IDs")
    if set(record_ids) != set(declared_ids):
        raise ValueError("Constraint edge declarations and records disagree.")
    edge_pairs = _edges([record.get("vertex_ids") for record in records], "Constraint edges")
    indices = _constraint_indices(edge_pairs, topology, len(vertices), "Constraint edges")
    hard_ids = set()
    for record, index in zip(records, indices):
        for flag in ("open_boundary", "non_manifold", "constrained_feature", "hard_feature"):
            if flag in record and not isinstance(record[flag], bool):
                raise ValueError(f"Constraint {flag} flag must be a boolean.")
        incident = _incident_faces(topology, index)
        reported_faces = _ids(record.get("incident_triangle_ids"), "Constraint incident faces", len(faces))
        reported_patches = _ids(record.get("incident_patch_ids"), "Constraint incident patches", len(patches))
        if (set(incident) != set(reported_faces)
                or set(labels[incident]) != set(reported_patches)):
            raise ValueError("Constraint edge incidence disagrees with PLY topology.")
        is_open, is_nonmanifold = len(incident) == 1, len(incident) > 2
        if (record.get("open_boundary", is_open) != is_open
                or record.get("non_manifold", is_nonmanifold) != is_nonmanifold):
            raise ValueError("Constraint topology flags disagree with the mesh.")
        inferred_hard = is_open or is_nonmanifold or bool(record.get("constrained_feature", False))
        if "hard_feature" in record and bool(record["hard_feature"]) != inferred_hard:
            raise ValueError("Constraint hard_feature flag disagrees with topology/feature flags.")
        if inferred_hard:
            hard_ids.add(record["id"])
    smooth_ids = set(record_ids) - hard_ids
    has_hard = "hard_feature_edge_ids" in constraints
    has_smooth = "smooth_surface_transition_edge_ids" in constraints
    if has_hard != has_smooth:
        raise ValueError("Hard and smooth constraint ID declarations must appear together.")
    if has_hard:
        if (set(_ids(constraints["hard_feature_edge_ids"], "Hard feature IDs")) != hard_ids
                or set(_ids(constraints["smooth_surface_transition_edge_ids"], "Smooth transition IDs")) != smooth_ids):
            raise ValueError("Declared hard/smooth constraints disagree with edge flags.")
    hard = edge_pairs[np.array([edge_id in hard_ids for edge_id in record_ids], dtype=bool)]
    smooth = edge_pairs[np.array([edge_id in smooth_ids for edge_id in record_ids], dtype=bool)]
    _validate_constraint_classes(hard, smooth, topology, len(vertices))
    corners = _ids(constraints.get("corner_vertex_ids", []), "Corner vertices", len(vertices))
    if len(corners) and not np.all(np.isin(corners, edge_pairs)):
        raise ValueError("Corner vertices must belong to the constraint graph.")
    return PartitionInput(directory, vertices, faces, labels, report,
                          _unique_edge_rows(hard), _unique_edge_rows(smooth), corners)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _ply_palette(type_ids, role_ids, color_mode):
    patch_count = len(type_ids)
    if color_mode == "surface_type":
        palette = SURFACE_COLORS[type_ids]
    elif color_mode == "feature_role":
        palette = ROLE_COLORS[role_ids]
    else:
        values = ((np.arange(patch_count, dtype=np.uint64) + 1) * 2654435761) & 0xffffffff
        values ^= values >> 16
        palette = np.column_stack((64 + (values & 127),
                                   64 + ((values >> 8) & 127),
                                   64 + ((values >> 16) & 127))).astype(np.int64)
    return palette


def _write_ply_files(targets, vertices, faces, labels, type_ids, role_ids):
    """Write ASCII views in blocks and format their shared vertices only once."""
    with ExitStack() as stack:
        streams = []
        for path, color_mode in targets:
            palette = _ply_palette(type_ids, role_ids, color_mode)
            stream = stack.enter_context(path.open(
                "w", encoding="ascii", newline="\n", buffering=1024 * 1024))
            stream.write(
                f"ply\nformat ascii 1.0\ncomment color_by {color_mode}\n"
                "comment surface_type_ids 0=Unknown 1=Plane 2=Cylinder 3=Cone 4=Sphere 5=Torus 6=Freeform\n"
                "comment feature_role_ids 0=Ordinary 1=Fillet 2=FilletCandidate\n"
                f"element vertex {len(vertices)}\nproperty double x\nproperty double y\nproperty double z\n"
                f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
                "property int patch_id\nproperty int primitive_type\nproperty uchar red\n"
                "property uchar green\nproperty uchar blue\nproperty int feature_role\nend_header\n"
            )
            streams.append((stream, palette))
        for start in range(0, len(vertices), _PLY_BLOCK_ROWS):
            local_vertices = vertices[start:start + _PLY_BLOCK_ROWS]
            # NumPy's tolist and Python's complete-block format execute in C;
            # savetxt dispatches a Python formatting/write operation per row.
            text = ("%.17g %.17g %.17g\n" * len(local_vertices)) % tuple(local_vertices.ravel().tolist())
            for stream, _ in streams:
                stream.write(text)
        for start in range(0, len(faces), _PLY_BLOCK_ROWS):
            local_labels = labels[start:start + _PLY_BLOCK_ROWS]
            rows = np.empty((len(local_labels), 9), dtype=np.int64)
            rows[:, :3] = faces[start:start + _PLY_BLOCK_ROWS]
            rows[:, 3] = local_labels
            rows[:, 4] = type_ids[local_labels]
            rows[:, 8] = role_ids[local_labels]
            record_format = "3 %d %d %d %d %d %d %d %d %d\n" * len(local_labels)
            for stream, palette in streams:
                rows[:, 5:8] = palette[local_labels]
                stream.write(record_format % tuple(rows.ravel().tolist()))


def _write_ply(path, vertices, faces, labels, type_ids, role_ids, color_mode):
    _write_ply_files([(path, color_mode)], vertices, faces, labels, type_ids, role_ids)


def write_remesh_result(output_directory, result, source, *, full_output=True):
    """Write remeshed geometry and fresh memberships, preserving source provenance."""
    original_source = source
    prepared = getattr(result, "prepared_source", None)
    if prepared is not None:
        if (prepared.source_directory != source.source_directory
                or not np.array_equal(prepared.vertices, source.vertices)
                or not np.array_equal(prepared.faces, source.faces)):
            raise ValueError("Prepared surface domains must refer to the exact input geometry.")
        source = prepared
    directory = Path(output_directory).expanduser().resolve()
    source_directory = Path(source.source_directory).resolve()
    if directory == source_directory:
        raise ValueError("Remesh output must not overwrite the source partition directory.")
    paths = {
        "remesh_result_ply": directory / "remesh_result.ply",
    }
    if full_output:
        paths.update({
        "surface_types_ply": directory / "surface_types.ply",
        "feature_roles_ply": directory / "feature_roles.ply",
        "remesh_result_stl": directory / "remesh_result.stl",
        "remesh_report_json": directory / "remesh_report.json",
        })
    for path in paths.values():
        resolved = path.resolve()
        if resolved.parent != directory or resolved.parent == source_directory:
            raise ValueError("Remesh output file redirects outside its output directory.")
    vertices = np.asarray(result.vertices, dtype=np.float64)
    faces = np.asarray(result.faces)
    labels = np.asarray(result.face_patch_ids)
    _validate_geometry(vertices, faces, labels)
    faces, labels = faces.astype(np.int64, copy=False), labels.astype(np.int64, copy=False)
    patches = source.report["patches"]
    if set(np.unique(labels)) != set(range(len(patches))):
        raise ValueError("Remesh output must retain all original patch IDs.")
    hard, smooth = _edges(result.hard_edges, "Hard output edges"), _edges(result.smooth_edges, "Smooth output edges")
    topology = _topology(faces, labels, len(vertices))
    _validate_constraint_classes(hard, smooth, topology, len(vertices))
    corners = _ids(result.corner_vertex_ids, "Output corner vertices", len(vertices))
    constraints = _unique_edge_rows(np.vstack((hard, smooth)))
    if len(corners) and not np.all(np.isin(corners, constraints)):
        raise ValueError("Output corners must belong to the constraint graph.")
    original_corners = source.vertices[source.corner_vertex_ids]
    output_corners = vertices[corners]
    if len(original_corners) != len(output_corners):
        raise ValueError("Remesh output must retain every original corner.")
    original_order = np.lexsort(original_corners.T[::-1])
    output_order = np.lexsort(output_corners.T[::-1])
    if not np.array_equal(original_corners[original_order], output_corners[output_order]):
        raise ValueError("Remesh output changed the position of an original corner.")
    lineage = np.asarray(result.source_constraint_edge_ids)
    source_constraints = source.constraint_edges
    if (lineage.shape != (len(constraints),)
            or not np.issubdtype(lineage.dtype, np.integer)
            or np.any(lineage < 0) or np.any(lineage >= len(source_constraints))):
        raise ValueError("Output constraint lineage must align with the sorted constraint edges.")
    _validate_output_lineage(vertices, constraints, hard, lineage, source, source_constraints)
    type_ids = np.array([SURFACE_TYPES[patch["type"]] for patch in patches])
    role_ids = np.array([FEATURE_ROLES[patch.get("feature_role", "Ordinary")] for patch in patches])
    if not full_output:
        # The shared validation above is required even when memberships,
        # diagnostic statistics and alternate mesh views are not exported.
        directory.mkdir(parents=True, exist_ok=True)
        _write_ply_files([(paths["remesh_result_ply"], "surface_instance")],
                         vertices, faces, labels, type_ids, role_ids)
        return paths
    ordered_faces = np.argsort(labels, kind="stable")
    member_offsets = np.concatenate(([0], np.cumsum(np.bincount(labels, minlength=len(patches)))))
    output_patches = []
    for patch in patches:
        patch_id = patch["id"]
        members = ordered_faces[member_offsets[patch_id]:member_offsets[patch_id + 1]]
        output_patches.append({
            "id": patch_id, "type": patch["type"],
            "source_patch_ids": patch.get("source_patch_ids", [patch_id]),
            "representative_source_patch_id": patch.get("representative_source_patch_id", patch_id),
            "feature_role": patch.get("feature_role", "Ordinary"),
            "support_patch_ids": patch.get("support_patch_ids", []),
            "parameters": patch.get("parameters"),
            "projection_target": result.stats.get("patch_projection_targets", {}).get(str(patch_id), "reference_mesh"),
            "source_projection_target": patch.get("projection_target", "reference_mesh"),
            "triangle_count": len(members), "triangle_ids": members,
            "source_triangle_count": patch["triangle_count"],
            "source_fit_diagnostics": {
                key: patch[key] for key in ("rms", "max", "normal_error", "confidence",
                                           "sampled_mesh_deviation", "fitting_error_semantics")
                if key in patch
            },
            "source_fit_diagnostics_semantics": "representative_source_patch_only_not_union_fit",
            "source_fit_diagnostics_by_patch": [
                {"source_patch_id": source_id, **{
                    key: original_source.report["patches"][source_id][key]
                    for key in ("rms", "max", "normal_error", "confidence", "sampled_mesh_deviation")
                    if key in original_source.report["patches"][source_id]
                }}
                for source_id in patch.get("source_patch_ids", [patch_id])
            ],
            "parameter_semantics": "source_partition_model_not_refitted_to_remesh",
            "remesh_error_validated": False,
        })
    constraint_indices = _constraint_indices(constraints, topology, len(vertices), "Output constraints")
    hard_keys = set(map(tuple, hard))
    edge_records = []
    hard_ids, smooth_ids = [], []
    for edge_id, (edge, index) in enumerate(zip(constraints, constraint_indices)):
        incident = _incident_faces(topology, index)
        is_hard = tuple(edge) in hard_keys
        (hard_ids if is_hard else smooth_ids).append(edge_id)
        edge_records.append({
            "id": edge_id, "vertex_ids": edge,
            "incident_triangle_ids": incident,
            "incident_patch_ids": np.unique(labels[incident]),
            "hard_feature": is_hard, "open_boundary": len(incident) == 1,
            "boundary_kind": "hard_geometry" if is_hard else "smooth_geometry_transition",
            "required_geometric_constraint": True,
            "non_manifold": len(incident) > 2,
            "source_constraint_edge_index": lineage[edge_id],
        })
    report = {
        "schema": "cadmesh.partition_remesh", "schema_version": 1,
        "source_directory": str(source_directory),
        "source_schema": source.report.get("schema"),
        "source_original_patch_count": len(original_source.report["patches"]),
        "source_mesh": source.report.get("mesh"),
        "indexing": {"base": 0, "vertices": "remesh_result_ply_vertex_order",
                     "triangles": "remesh_result_ply_face_order",
                     "source_constraint_edges": "lexicographic_vertex_pair_order"},
        "mesh": {"vertex_count": len(vertices), "triangle_count": len(faces),
                 "edge_count": len(topology["keys"]),
                 "open_boundary_edges": int(np.sum(topology["counts"] == 1)),
                 "non_manifold_edges": int(np.sum(topology["counts"] > 2))},
        "patches": output_patches,
        "constraints": {"edge_ids": list(range(len(constraints))),
                        "hard_feature_edge_ids": hard_ids,
                        "smooth_surface_transition_edge_ids": smooth_ids,
                        "corner_vertex_ids": corners},
        "boundary_policy": result.stats.get("boundary_policy", {
            "semantics": "legacy: every source label interface is fixed"}),
        "constraint_edges": edge_records,
        "source_constraint_edges": source_constraints,
        "source_constraint_edge_indices": lineage,
        "fitting_error_semantics": "source_fit_diagnostics_are_not_output_geometry_error",
        "stl_coordinate_precision": "float32; PLY retains float64 shared indexed geometry",
        "stats": result.stats,
    }
    # Validate serialization before creating output files; NaN/Infinity cannot
    # silently become apparently valid mesh quality or fitting measurements.
    # dumps uses Python's C JSON encoder; dump streams millions of tiny Python
    # writes for face IDs and constraint records. Validate the complete report
    # before producing any files, then write the encoded text in one operation.
    encoded_report = json.dumps(report, default=_json_default, allow_nan=False,
                                separators=(",", ":"))
    directory.mkdir(parents=True, exist_ok=True)
    _write_ply_files([
        (paths["remesh_result_ply"], "surface_instance"),
        (paths["surface_types_ply"], "surface_type"),
        (paths["feature_roles_ply"], "feature_role"),
    ], vertices, faces, labels, type_ids, role_ids)
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    paths["remesh_result_stl"].write_bytes(trimesh.exchange.stl.export_stl(mesh))
    with paths["remesh_report_json"].open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded_report)
        stream.write("\n")
    return paths


def _validate_output_lineage(vertices, edges, hard_edges, lineage, source, parents=None):
    """Check every original constraint survives as one continuous child chain."""
    if parents is None:
        parents = source.constraint_edges
    if not np.array_equal(np.unique(lineage), np.arange(len(parents))):
        raise ValueError("Output lineage omits an original constraint edge.")
    if not len(parents):
        return
    source_hard = set(map(tuple, source.hard_edges))
    output_hard = set(map(tuple, hard_edges))
    for edge, parent_index in zip(edges, lineage):
        if (tuple(edge) in output_hard) != (tuple(parents[parent_index]) in source_hard):
            raise ValueError("Output constraint lineage changes hard/smooth classification.")
    starts = source.vertices[parents[lineage, 0]]
    vectors = source.vertices[parents[lineage, 1]] - starts
    length_squared = np.einsum("ij,ij->i", vectors, vectors)
    points = vertices[edges]
    offsets = points - starts[:, None, :]
    parameter = np.einsum("ijk,ik->ij", offsets, vectors) / length_squared[:, None]
    projected = starts[:, None, :] + parameter[:, :, None] * vectors[:, None, :]
    scale = max(float(np.max(np.abs(source.vertices))), float(np.max(np.abs(vertices))),
                np.finfo(np.float64).tiny)
    tolerance = scale * np.finfo(np.float64).eps * 64 + np.sqrt(length_squared) * 1e-10
    if np.any(np.linalg.norm(points - projected, axis=2) > tolerance[:, None]):
        raise ValueError("Output constraint vertices left their original edge segments.")
    parameter_tolerance = tolerance / np.sqrt(length_squared)
    intervals = np.sort(parameter, axis=1)
    if np.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError("Output constraint lineage contains a zero-length child.")
    order = np.lexsort((intervals[:, 0], lineage))
    sorted_parents, sorted_intervals = lineage[order], intervals[order]
    sorted_tolerance = parameter_tolerance[order]
    first = np.concatenate(([True], sorted_parents[1:] != sorted_parents[:-1]))
    last = np.concatenate((first[1:], [True]))
    if (np.any(np.abs(sorted_intervals[first, 0]) > sorted_tolerance[first])
            or np.any(np.abs(sorted_intervals[last, 1] - 1.0) > sorted_tolerance[last])):
        raise ValueError("Output constraint children do not retain original endpoints.")
    interior = ~last[:-1]
    gaps = sorted_intervals[1:, 0] - sorted_intervals[:-1, 1]
    if np.any(np.abs(gaps[interior]) > np.maximum(sorted_tolerance[1:], sorted_tolerance[:-1])[interior]):
        raise ValueError("Output constraint lineage contains a gap or overlapping children.")
