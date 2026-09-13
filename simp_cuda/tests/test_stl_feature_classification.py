import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1] / "pamo" / "stl_feature_classification.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_pamo_stl_feature_classification_test", MODULE_PATH
)
classification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classification
SPEC.loader.exec_module(classification)


def cylinder_wall(radius=1.0, height=2.0, segments=32, z_segments=1):
    vertices = []
    for z_index in range(z_segments + 1):
        z = height * z_index / z_segments
        for angle_index in range(segments):
            angle = 2.0 * np.pi * angle_index / segments
            vertices.append(
                (radius * np.cos(angle), radius * np.sin(angle), z)
            )
    faces = []
    for z_index in range(z_segments):
        for angle_index in range(segments):
            following = (angle_index + 1) % segments
            lower = z_index * segments + angle_index
            lower_following = z_index * segments + following
            upper = (z_index + 1) * segments + angle_index
            upper_following = (z_index + 1) * segments + following
            faces.extend(
                (
                    (lower, lower_following, upper_following),
                    (lower, upper_following, upper),
                )
            )
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def orient_faces_outward(vertices, faces):
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    reverse = np.einsum("ij,ij->i", normals, triangles.mean(axis=1)) < 0.0
    faces = faces.copy()
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    return faces


def triangulated_cube():
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return vertices, orient_faces_outward(vertices, faces)


def uv_sphere(radius=3.0, latitude_segments=10, longitude_segments=24):
    vertices = [[0.0, 0.0, radius]]
    for latitude_index in range(1, latitude_segments):
        polar = np.pi * latitude_index / latitude_segments
        for longitude_index in range(longitude_segments):
            azimuth = 2.0 * np.pi * longitude_index / longitude_segments
            vertices.append(
                [
                    radius * np.sin(polar) * np.cos(azimuth),
                    radius * np.sin(polar) * np.sin(azimuth),
                    radius * np.cos(polar),
                ]
            )
    south = len(vertices)
    vertices.append([0.0, 0.0, -radius])

    faces = []
    first_ring = 1
    for longitude_index in range(longitude_segments):
        following = (longitude_index + 1) % longitude_segments
        faces.append([0, first_ring + longitude_index, first_ring + following])
    for latitude_index in range(latitude_segments - 2):
        first = 1 + latitude_index * longitude_segments
        second = first + longitude_segments
        for longitude_index in range(longitude_segments):
            following = (longitude_index + 1) % longitude_segments
            faces.extend(
                (
                    [first + longitude_index, second + longitude_index,
                     second + following],
                    [first + longitude_index, second + following,
                     first + following],
                )
            )
    last_ring = 1 + (latitude_segments - 2) * longitude_segments
    for longitude_index in range(longitude_segments):
        following = (longitude_index + 1) % longitude_segments
        faces.append([last_ring + longitude_index, south, last_ring + following])
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    return vertices, orient_faces_outward(vertices, faces)


def rounded_quarter_corner(radius=1.0, length=5.0, arc_segments=16):
    vertices = []
    vertex_ids = {}

    def vertex_id(point):
        key = tuple(np.round(point, 12))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(point)
        return vertex_ids[key]

    z_values = np.linspace(0.0, 1.0, 3)
    angle_values = np.linspace(0.0, 0.5 * np.pi, arc_segments + 1)
    cylinder_ids = np.empty((len(z_values), len(angle_values)), dtype=np.int64)
    for z_index, z in enumerate(z_values):
        for angle_index, angle in enumerate(angle_values):
            cylinder_ids[z_index, angle_index] = vertex_id(
                [radius * np.cos(angle), radius * np.sin(angle), z]
            )

    faces = []
    for z_index in range(len(z_values) - 1):
        for angle_index in range(len(angle_values) - 1):
            lower = cylinder_ids[z_index, angle_index]
            following = cylinder_ids[z_index, angle_index + 1]
            upper = cylinder_ids[z_index + 1, angle_index]
            upper_following = cylinder_ids[z_index + 1, angle_index + 1]
            faces.extend(
                ([lower, following, upper_following], [lower, upper_following, upper])
            )

    # Two long tangent planes make the partial cylinder an unambiguous fillet.
    plane_coordinates = np.linspace(-length, 0.0, 3)
    for plane_index in range(2):
        grid = np.empty((len(z_values), len(plane_coordinates)), dtype=np.int64)
        for z_index, z in enumerate(z_values):
            for coordinate_index, coordinate in enumerate(plane_coordinates):
                if plane_index == 0:
                    point = [radius, coordinate, z]
                else:
                    point = [coordinate, radius, z]
                grid[z_index, coordinate_index] = vertex_id(point)
        for z_index in range(len(z_values) - 1):
            for coordinate_index in range(len(plane_coordinates) - 1):
                lower = grid[z_index, coordinate_index]
                following = grid[z_index, coordinate_index + 1]
                upper = grid[z_index + 1, coordinate_index]
                upper_following = grid[z_index + 1, coordinate_index + 1]
                if plane_index == 0:
                    faces.extend(
                        ([lower, following, upper_following],
                         [lower, upper_following, upper])
                    )
                else:
                    faces.extend(
                        ([lower, upper_following, following],
                         [lower, upper, upper_following])
                    )
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def conical_frustum(
    lower_radius=2.0, upper_radius=1.0, height=2.0,
    angle_segments=32, height_segments=4,
):
    vertices = []
    for height_index in range(height_segments + 1):
        fraction = height_index / height_segments
        radius = lower_radius + fraction * (upper_radius - lower_radius)
        z = fraction * height
        for angle_index in range(angle_segments):
            angle = 2.0 * np.pi * angle_index / angle_segments
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), z])
    faces = []
    for height_index in range(height_segments):
        for angle_index in range(angle_segments):
            following = (angle_index + 1) % angle_segments
            lower = height_index * angle_segments + angle_index
            lower_following = height_index * angle_segments + following
            upper = (height_index + 1) * angle_segments + angle_index
            upper_following = (height_index + 1) * angle_segments + following
            faces.extend(
                ([lower, lower_following, upper_following],
                 [lower, upper_following, upper])
            )
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def torus(major_radius=3.0, minor_radius=1.0, major_segments=32, minor_segments=16):
    vertices = []
    for major_index in range(major_segments):
        major_angle = 2.0 * np.pi * major_index / major_segments
        for minor_index in range(minor_segments):
            minor_angle = 2.0 * np.pi * minor_index / minor_segments
            radial = major_radius + minor_radius * np.cos(minor_angle)
            vertices.append(
                [
                    radial * np.cos(major_angle),
                    radial * np.sin(major_angle),
                    minor_radius * np.sin(minor_angle),
                ]
            )
    faces = []
    for major_index in range(major_segments):
        following_major = (major_index + 1) % major_segments
        for minor_index in range(minor_segments):
            following_minor = (minor_index + 1) % minor_segments
            lower = major_index * minor_segments + minor_index
            lower_following = major_index * minor_segments + following_minor
            upper = following_major * minor_segments + minor_index
            upper_following = following_major * minor_segments + following_minor
            faces.extend(
                ([lower, upper, upper_following],
                 [lower, upper_following, lower_following])
            )
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


class StlFeatureClassificationTest(unittest.TestCase):
    def test_classifies_cylinder_wall(self):
        vertices, faces = cylinder_wall()

        result = classification.classify_mesh_features(vertices, faces)

        self.assertEqual(len(result.patches), 1)
        patch = result.patches[0]
        self.assertEqual(patch.surface_type, classification.SurfaceType.CYLINDER)
        self.assertAlmostEqual(patch.parameters["radius"], 1.0, places=6)
        self.assertGreater(patch.parameters["angular_coverage_degrees"], 340.0)
        self.assertEqual(
            patch.transition_type, classification.TransitionType.NONE
        )

    def test_cube_is_six_planar_patches(self):
        vertices, faces = triangulated_cube()
        result = classification.classify_mesh_features(vertices, faces)

        self.assertEqual(len(result.patches), 6)
        self.assertTrue(
            all(
                patch.surface_type == classification.SurfaceType.PLANE
                for patch in result.patches
            )
        )
        np.testing.assert_array_equal(
            np.bincount(result.face_patch_ids), np.full(6, 2)
        )

    def test_classifies_sphere_separately_from_generic_curve(self):
        vertices, faces = uv_sphere()
        result = classification.classify_mesh_features(vertices, faces)

        self.assertEqual(len(result.patches), 1)
        self.assertEqual(
            result.patches[0].surface_type,
            classification.SurfaceType.SPHERE,
        )
        self.assertAlmostEqual(result.patches[0].parameters["radius"], 3.0)

    def test_partial_cylinder_between_planes_is_a_fillet(self):
        vertices, faces = rounded_quarter_corner()

        result = classification.classify_mesh_features(vertices, faces)

        fillets = [
            patch for patch in result.patches
            if patch.transition_type == classification.TransitionType.FILLET
        ]
        self.assertEqual(len(fillets), 1)
        self.assertEqual(
            fillets[0].surface_type, classification.SurfaceType.CYLINDER
        )
        self.assertAlmostEqual(
            fillets[0].parameters["angular_coverage_degrees"], 90.0
        )

    def test_small_planes_do_not_depend_on_global_mesh_span(self):
        vertices, faces = rounded_quarter_corner(length=100.0)
        config = classification.ClassificationConfig(
            minimum_embedded_plane_span_ratio=0.99
        )

        result = classification.classify_mesh_features(
            vertices, faces, config=config
        )

        planes = [
            patch for patch in result.patches
            if patch.surface_type == classification.SurfaceType.PLANE
        ]
        cylinders = [
            patch for patch in result.patches
            if patch.surface_type == classification.SurfaceType.CYLINDER
        ]
        self.assertEqual(len(planes), 2)
        self.assertEqual(sorted(len(patch.face_indices) for patch in planes), [8, 8])
        self.assertEqual(len(cylinders), 1)
        self.assertEqual(len(cylinders[0].face_indices), 64)

    def test_dense_cylinder_is_not_split_into_planar_strips(self):
        vertices, faces = cylinder_wall(z_segments=4)

        result = classification.classify_mesh_features(vertices, faces)

        self.assertEqual(len(result.patches), 1)
        self.assertEqual(
            result.patches[0].surface_type,
            classification.SurfaceType.CYLINDER,
        )
        self.assertEqual(len(result.patches[0].face_indices), len(faces))

    def test_classifies_conical_frustum(self):
        vertices, faces = conical_frustum()

        result = classification.classify_mesh_features(vertices, faces)

        self.assertEqual(len(result.patches), 1)
        self.assertEqual(
            result.patches[0].surface_type, classification.SurfaceType.CONE
        )

    def test_classifies_torus(self):
        vertices, faces = torus()

        result = classification.classify_mesh_features(vertices, faces)

        self.assertEqual(len(result.patches), 1)
        patch = result.patches[0]
        self.assertEqual(patch.surface_type, classification.SurfaceType.TORUS)
        self.assertAlmostEqual(patch.parameters["major_radius"], 3.0, places=6)
        self.assertAlmostEqual(patch.parameters["minor_radius"], 1.0, places=6)

    def test_rejects_degenerate_triangles(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            classification.classify_mesh_features(
                np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
                ),
                np.array([[0, 1, 2]], dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
