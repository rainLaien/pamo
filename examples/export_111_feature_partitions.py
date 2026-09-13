import colorsys
import importlib.util
import sys
from pathlib import Path

import numpy as np
import trimesh


root = Path(__file__).resolve().parents[1]
module_path = root / "simp_cuda/pamo/stl_feature_classification.py"

spec = importlib.util.spec_from_file_location(
    "stl_feature_classification", module_path
)
features = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = features
spec.loader.exec_module(features)

input_path = root / "examples/111.stl"
output_path = root / "examples/111_feature_partitions.ply"

mesh = trimesh.load(input_path, force="mesh", process=False)
mesh.merge_vertices()
mesh.update_faces(mesh.unique_faces())
mesh.update_faces(mesh.nondegenerate_faces())
mesh.remove_unreferenced_vertices()
mesh.fix_normals(multibody=True)

result = features.classify_mesh_features(mesh)

palette = []
for index in range(len(result.patches)):
    hue = (index * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    palette.append([int(value * 255) for value in rgb] + [255])

mesh.visual.face_colors = np.asarray(
    palette, dtype=np.uint8
)[result.face_patch_ids]

surface_type_codes = {
    surface_type: index
    for index, surface_type in enumerate(features.SurfaceType)
}
transition_type_codes = {
    transition_type: index
    for index, transition_type in enumerate(features.TransitionType)
}
face_surface_types = np.empty(len(mesh.faces), dtype=np.uint8)
face_transition_types = np.empty(len(mesh.faces), dtype=np.uint8)
for patch in result.patches:
    face_surface_types[patch.face_indices] = surface_type_codes[
        patch.surface_type
    ]
    face_transition_types[patch.face_indices] = transition_type_codes[
        patch.transition_type
    ]

# Binary STL's two-byte per-face field is normally zero and does not contain
# CAD semantics. Replace that inherited field with explicit classification
# properties in the exported PLY.
mesh.face_attributes.pop("stl", None)
mesh.face_attributes["patch_id"] = result.face_patch_ids.astype(np.uint32)
mesh.face_attributes["surface_type"] = face_surface_types
mesh.face_attributes["transition_type"] = face_transition_types
mesh.export(output_path)

for patch in result.patches:
    print(
        f"区域 {patch.patch_id:3d}: "
        f"{patch.surface_type.value:10s} "
        f"{patch.transition_type.value:8s} "
        f"三角面={len(patch.face_indices):6d} "
        f"置信度={patch.confidence:.3f}"
    )

print(f"\n共 {len(result.patches)} 个分区")
print(f"彩色结果：{output_path}")
print(
    "surface_type 编码："
    + ", ".join(
        f"{surface_type.value}={code}"
        for surface_type, code in surface_type_codes.items()
    )
)
print(
    "transition_type 编码："
    + ", ".join(
        f"{transition_type.value}={code}"
        for transition_type, code in transition_type_codes.items()
    )
)
