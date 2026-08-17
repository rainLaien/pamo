import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


PAMO_PATH = Path(__file__).parents[1] / "pamo"

try:
    import igl  # noqa: F401
    import scipy  # noqa: F401
    import trimesh
except ImportError:
    feature_optimize = None
else:
    package_name = "_pamo_feature_optimize_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PAMO_PATH)]
    sys.modules[package_name] = package

    for module_name in (
        "segment_query",
        "feature_edges",
        "feature_optimize",
        "original_constrained",
    ):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            PAMO_PATH / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)

    feature_optimize = sys.modules[
        f"{package_name}.feature_optimize"
    ]
    original_constrained = sys.modules[
        f"{package_name}.original_constrained"
    ]


@unittest.skipIf(
    feature_optimize is None,
    "Feature optimization dependencies are unavailable",
)
class FeatureOptimizeTest(unittest.TestCase):
    def test_planar_classifier_merges_tolerant_faces_but_not_hard_edges(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1e-7],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        regions, stats = original_constrained.classify_planar_face_regions(
            vertices,
            faces,
            minimum_faces=2,
            maximum_normal_angle_degrees=0.5,
            maximum_plane_distance_ratio=1e-5,
        )
        separated, separated_stats = (
            original_constrained.classify_planar_face_regions(
                vertices,
                faces,
                protected_edges=np.array([[0, 2]], dtype=np.int64),
                minimum_faces=2,
                maximum_normal_angle_degrees=0.5,
                maximum_plane_distance_ratio=1e-5,
            )
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual(stats["planar_faces"], 2)
        self.assertFalse(separated)
        self.assertEqual(separated_stats["planar_faces"], 0)

    def test_selects_largest_opposed_planar_pair_by_area(self):
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (4.0, 4.0, 0.0),
                (0.0, 4.0, 0.0),
                (0.0, 0.0, 1.0),
                (4.0, 0.0, 1.0),
                (4.0, 3.0, 1.0),
                (0.0, 3.0, 1.0),
                (10.0, 0.0, 0.0),
                (10.0, 5.0, 0.0),
                (10.0, 5.0, 3.0),
                (10.0, 0.0, 3.0),
            ),
            dtype=np.float64,
        )
        faces = np.asarray(
            (
                (0, 1, 2), (0, 2, 3),
                (4, 6, 5), (4, 7, 6),
                (8, 9, 10), (8, 10, 11),
            ),
            dtype=np.int64,
        )
        regions = [
            np.asarray((0, 1), dtype=np.int64),
            np.asarray((2, 3), dtype=np.int64),
            np.asarray((4, 5), dtype=np.int64),
        ]

        selected, stats = (
            original_constrained.select_largest_opposed_planar_regions(
                vertices,
                faces,
                regions,
            )
        )

        self.assertEqual([region.tolist() for region in selected], [[0, 1], [2, 3]])
        self.assertAlmostEqual(stats["primary_area"], 16.0)
        self.assertAlmostEqual(stats["opposite_area"], 12.0)
        self.assertAlmostEqual(stats["normal_dot"], -1.0)

    def test_surface_classifier_separates_plane_and_general_extrusion(self):
        vertices = []
        faces = []

        # A 2x2 planar grid is large enough to form one planar region.
        for y in range(3):
            for x in range(3):
                vertices.append((float(x), 10.0 + float(y), 0.0))
        for y in range(2):
            for x in range(2):
                lower_left = y * 3 + x
                lower_right = lower_left + 1
                upper_left = lower_left + 3
                upper_right = upper_left + 1
                faces.extend(
                    (
                        (lower_left, lower_right, upper_right),
                        (lower_left, upper_right, upper_left),
                    )
                )

        # This non-circular profile is extruded along X.  Its old triangles
        # are deliberately very long in the extrusion direction.
        profile = (
            (-3.0, 0.0),
            (-2.0, 2.0),
            (-1.0, 0.0),
            (0.0, 3.0),
            (1.0, 0.0),
            (2.0, 1.0),
            (3.0, 0.0),
        )
        profile_start = len(vertices)
        for y, z in profile:
            vertices.extend(((0.0, y, z), (50.0, y, z)))
        for index in range(len(profile) - 1):
            lower_first = profile_start + index * 2
            upper_first = lower_first + 1
            lower_second = lower_first + 2
            upper_second = lower_first + 3
            faces.extend(
                (
                    (lower_first, upper_first, upper_second),
                    (lower_first, upper_second, lower_second),
                )
            )

        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        planar_regions, planar_stats = (
            original_constrained.classify_planar_face_regions(
                vertices,
                faces,
                minimum_faces=4,
                maximum_normal_angle_degrees=0.1,
                maximum_plane_distance_ratio=1e-8,
            )
        )
        regions, stats = original_constrained.classify_nonplanar_face_regions(
            vertices,
            faces,
            planar_regions,
            minimum_faces=4,
            axial_normal_tolerance=1e-8,
            radius_tolerance=1e-3,
        )

        self.assertEqual(planar_stats["planar_faces"], 8)
        self.assertGreaterEqual(stats["extrusion_regions"], 1, stats)
        self.assertEqual(stats["extrusion_faces"], 12, stats)
        self.assertEqual(stats["general_curved_faces"], 0, stats)
        self.assertFalse(regions["cylinders"])
        self.assertFalse(regions["fillets"])

    def test_developable_region_discards_internal_edges_in_uv_chart(self):
        boundary = []
        for x in range(10):
            boundary.append((float(x), 0.0, 0.0))
        for y in range(10):
            boundary.append((10.0, float(y), 0.0))
        for x in range(10, 0, -1):
            boundary.append((float(x), 10.0, 0.0))
        for y in range(10, 0, -1):
            boundary.append((0.0, float(y), 0.0))
        vertices = np.asarray(boundary + [(1.0, 1.0, 0.0)], dtype=np.float64)
        center = len(vertices) - 1
        faces = np.asarray(
            [
                (center, index, (index + 1) % len(boundary))
                for index in range(len(boundary))
            ],
            dtype=np.int64,
        )
        old_quality = feature_optimize._triangle_quality_values(vertices, faces)

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_developable_regions(
                vertices,
                faces,
                [{"faces": np.arange(len(faces), dtype=np.int64)}],
                minimum_faces=4,
                maximum_target_edge_length=1.0,
                minimum_triangle_angle_degrees=28.0,
            )
        )
        new_quality = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )

        self.assertEqual(stats["regions"], 1, stats)
        self.assertGreater(float(new_quality.mean()), float(old_quality.mean()))
        self.assertGreater(
            float(np.percentile(new_quality, 5.0)),
            float(np.percentile(old_quality, 5.0)),
        )

    def test_solid_planar_fan_uses_uniform_constrained_mesh(self):
        count = 40
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                [0.0, 0.0, 0.0],
                np.column_stack(
                    (10 * np.cos(angles), 10 * np.sin(angles), np.zeros(count))
                ),
            )
        )
        faces = np.asarray(
            [[0, index + 1, (index + 1) % count + 1] for index in range(count)],
            dtype=np.int64,
        )
        boundary = np.asarray(
            [[index + 1, (index + 1) % count + 1] for index in range(count)],
            dtype=np.int64,
        )

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_planar_annuli(
                vertices,
                faces,
                protected_edges=boundary,
                minimum_faces=20,
                minimum_holes=0,
                maximum_holes=0,
                maximum_target_edge_length=5.0,
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )

        self.assertEqual(stats["regions"], 1, stats)
        self.assertGreater(float(qualities.mean()), 0.9)
        self.assertGreater(float(np.percentile(qualities, 5.0)), 0.7)

    def test_half_cylinder_with_irregular_boundary_is_retriangulated(self):
        count = 31
        angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (10 * np.cos(angles), 10 * np.sin(angles), np.zeros(count))
                ),
                np.column_stack(
                    (10 * np.cos(angles), 10 * np.sin(angles), np.full(count, 10.0))
                ),
            )
        )
        faces = []
        for index in range(count - 1):
            faces.extend(
                (
                    [index, index + 1, count + index],
                    [index + 1, count + index + 1, count + index],
                )
            )
        faces = np.asarray(faces, dtype=np.int64)
        boundary = np.asarray(
            [[index, index + 1] for index in range(count - 1)]
            + [[count + index, count + index + 1] for index in range(count - 1)]
            + [[0, count], [count - 1, 2 * count - 1]],
            dtype=np.int64,
        )

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_partial_cylindrical_walls(
                vertices, faces, boundary, minimum_faces=20
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }

        self.assertEqual(stats["patches"], 1)
        self.assertGreater(float(np.percentile(qualities, 5.0)), 0.8)
        self.assertTrue(
            all(tuple(sorted(map(int, edge))) in result_edges for edge in boundary)
        )

    def test_mismatched_cylinder_rings_get_axial_transition_rows(self):
        lower_count = 60
        upper_count = 20
        lower_angles = np.arange(lower_count) * (2.0 * np.pi / lower_count)
        upper_angles = np.arange(upper_count) * (2.0 * np.pi / upper_count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (10 * np.cos(lower_angles), 10 * np.sin(lower_angles),
                     np.zeros(lower_count))
                ),
                np.column_stack(
                    (10 * np.cos(upper_angles), 10 * np.sin(upper_angles),
                     np.full(upper_count, 8.0))
                ),
            )
        )
        faces = []
        lower_index = 0
        upper_index = 0
        while lower_index < lower_count or upper_index < upper_count:
            next_lower = (
                (lower_index + 1) / lower_count
                if lower_index < lower_count else 2.0
            )
            next_upper = (
                (upper_index + 1) / upper_count
                if upper_index < upper_count else 2.0
            )
            if next_lower <= next_upper:
                faces.append(
                    [lower_index % lower_count,
                     (lower_index + 1) % lower_count,
                     lower_count + upper_index % upper_count]
                )
                lower_index += 1
            else:
                faces.append(
                    [lower_index % lower_count,
                     lower_count + (upper_index + 1) % upper_count,
                     lower_count + upper_index % upper_count]
                )
                upper_index += 1
        faces = np.asarray(faces, dtype=np.int64)
        boundaries = np.asarray(
            [[index, (index + 1) % lower_count] for index in range(lower_count)]
            + [
                [lower_count + index,
                 lower_count + (index + 1) % upper_count]
                for index in range(upper_count)
            ],
            dtype=np.int64,
        )

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_cylindrical_walls(
                vertices,
                faces,
                boundaries,
                minimum_faces=20,
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }
        edge_array = np.sort(
            result_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
            axis=1,
        )
        unique_edges, edge_counts = np.unique(
            edge_array, axis=0, return_counts=True
        )
        boundary_set = {
            tuple(sorted(map(int, edge))) for edge in boundaries
        }

        self.assertEqual(stats["walls"], 1)
        self.assertGreater(float(np.percentile(qualities, 5.0)), 0.5)
        self.assertTrue(
            all(tuple(sorted(map(int, edge))) in result_edges for edge in boundaries)
        )
        self.assertTrue(
            all(
                int(count) == (1 if tuple(map(int, edge)) in boundary_set else 2)
                for edge, count in zip(unique_edges, edge_counts)
            )
        )

    def test_planar_annulus_bridges_are_retriangulated_uniformly(self):
        count = 80
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (30.0 * np.cos(angles), 30.0 * np.sin(angles), np.zeros(count))
                ),
                np.column_stack(
                    (12.0 * np.cos(angles), 12.0 * np.sin(angles), np.zeros(count))
                ),
            )
        )
        faces = []
        for index in range(count):
            following = (index + 1) % count
            faces.extend(
                (
                    [index, following, count + index],
                    [following, count + following, count + index],
                )
            )
        faces = np.asarray(faces, dtype=np.int64)

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_planar_annuli(
                vertices, faces, minimum_faces=20
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }

        self.assertEqual(stats["regions"], 1)
        self.assertGreater(float(np.percentile(qualities, 5.0)), 0.7)
        for offset in (0, count):
            self.assertTrue(
                all(
                    tuple(
                        sorted((offset + index, offset + (index + 1) % count))
                    )
                    in result_edges
                    for index in range(count)
                )
            )

    def test_planar_multi_hole_fallback_recovers_every_boundary(self):
        width = 12
        height = 6
        vertices = np.asarray(
            [
                (float(x), float(y), 0.0)
                for y in range(height + 1)
                for x in range(width + 1)
            ],
            dtype=np.float64,
        )

        def vertex_id(x, y):
            return y * (width + 1) + x

        removed_cells = {
            (x, y)
            for x_start in (2, 8)
            for x in range(x_start, x_start + 2)
            for y in range(2, 4)
        }
        faces = []
        for y in range(height):
            for x in range(width):
                if (x, y) in removed_cells:
                    continue
                lower_left = vertex_id(x, y)
                lower_right = vertex_id(x + 1, y)
                upper_left = vertex_id(x, y + 1)
                upper_right = vertex_id(x + 1, y + 1)
                faces.extend(
                    (
                        (lower_left, lower_right, upper_right),
                        (lower_left, upper_right, upper_left),
                    )
                )
        faces = np.asarray(faces, dtype=np.int64)
        input_edges = np.sort(
            faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
            axis=1,
        )
        unique_input_edges, input_counts = np.unique(
            input_edges,
            axis=0,
            return_counts=True,
        )
        boundary_edges = unique_input_edges[input_counts == 1]
        boundary_vertices = set(boundary_edges.reshape(-1).tolist())
        for vertex_index in range(len(vertices)):
            if vertex_index in boundary_vertices:
                continue
            row = vertex_index // (width + 1)
            vertices[vertex_index, 0] += 0.45 if row % 2 else -0.45

        saved_triangle = original_constrained.constrained_triangle
        original_constrained.constrained_triangle = None
        try:
            result_vertices, result_faces, stats = (
                original_constrained.retriangulate_planar_annuli(
                    vertices,
                    faces,
                    minimum_faces=20,
                    minimum_holes=2,
                    maximum_holes=2,
                    maximum_target_edge_length=1.0,
                    minimum_triangle_angle_degrees=28.0,
                )
            )
        finally:
            original_constrained.constrained_triangle = saved_triangle

        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
        }
        maximum_edge_length = max(
            np.linalg.norm(result_vertices[first] - result_vertices[second])
            for first, second in result_edges
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices,
            result_faces,
        )

        self.assertEqual(stats["regions"], 1, stats)
        self.assertTrue(
            all(tuple(map(int, edge)) in result_edges for edge in boundary_edges)
        )
        self.assertLessEqual(maximum_edge_length, 2.0 + 1e-8)
        self.assertGreater(float(np.percentile(qualities, 5.0)), 0.4)

    def test_planar_annulus_with_internal_hard_edge_is_not_replaced(self):
        count = 12
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                np.column_stack(
                    (3 * np.cos(angles), 3 * np.sin(angles), np.zeros(count))
                ),
                np.column_stack(
                    (np.cos(angles), np.sin(angles), np.zeros(count))
                ),
            )
        )
        faces = []
        for index in range(count):
            following = (index + 1) % count
            faces.extend(
                (
                    [index, following, count + index],
                    [following, count + following, count + index],
                )
            )
        faces = np.asarray(faces, dtype=np.int64)
        internal_hard_edge = np.asarray([[0, count]], dtype=np.int64)

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_planar_annuli(
                vertices,
                faces,
                protected_edges=internal_hard_edge,
                minimum_faces=20,
            )
        )

        self.assertEqual(stats["regions"], 0)
        self.assertEqual(stats["constraint_rejections"], 1)
        np.testing.assert_array_equal(result_vertices, vertices)
        np.testing.assert_array_equal(result_faces, faces)

    def test_planar_circle_fan_is_retriangulated_uniformly(self):
        count = 81
        angles = np.arange(count) * (2.0 * np.pi / count)
        vertices = np.vstack(
            (
                [0.0, 0.0, 0.0],
                np.column_stack(
                    (28.0 * np.cos(angles), 28.0 * np.sin(angles), np.zeros(count))
                ),
            )
        )
        faces = np.asarray(
            [[0, index + 1, (index + 1) % count + 1] for index in range(count)],
            dtype=np.int64,
        )
        boundary = np.asarray(
            [[index + 1, (index + 1) % count + 1] for index in range(count)],
            dtype=np.int64,
        )

        result_vertices, result_faces, stats = (
            original_constrained.retriangulate_planar_fans(
                vertices,
                faces,
                boundary,
                minimum_valence=30,
            )
        )
        qualities = feature_optimize._triangle_quality_values(
            result_vertices, result_faces
        )
        result_edges = {
            tuple(sorted((int(first), int(second))))
            for face in result_faces
            for first, second in (
                (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
            )
        }

        self.assertEqual(stats["fans"], 1)
        self.assertEqual(np.count_nonzero(result_faces == 0), 0)
        self.assertGreater(float(qualities.min()), 0.6)
        self.assertTrue(
            all(tuple(sorted(map(int, edge))) in result_edges for edge in boundary)
        )

    def test_automatic_edge_limit_uses_five_percent_diagonal(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]
        )
        edge_lengths = np.array([1.0, 2.0, 3.0])

        limit, diagonal, percentile_95 = (
            original_constrained.automatic_edge_length_limit(
                vertices,
                edge_lengths,
            )
        )

        self.assertEqual(diagonal, 1000.0)
        self.assertAlmostEqual(percentile_95, 2.9)
        self.assertAlmostEqual(limit, 50.0)

    def test_automatic_split_limit_has_conformity_margin(self):
        budget, estimate = original_constrained.automatic_split_limit(
            np.array([5.0, 20.0, 40.0]),
            max_edge_length=10.0,
        )

        self.assertEqual(estimate, 4)
        self.assertEqual(budget, 100000)

    def test_shared_region_boundary_is_sampled_conformingly(self):
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
                (0.0, 2.0, 0.0),
                (10.0, -2.0, 0.0),
            ),
            dtype=np.float64,
        )
        faces = np.asarray(((0, 1, 2), (1, 0, 3)), dtype=np.int64)
        tracked_roots = {(0, 1): 7}
        result_vertices, result_faces, stats = (
            original_constrained.sample_region_boundary_edges(
                vertices,
                faces,
                [(0, 1)],
                target_edge_length=2.0,
                tracked_edge_roots=tracked_roots,
                tracked_face_labels=np.asarray((3, 9), dtype=np.int64),
            )
        )
        edge_faces = original_constrained._build_edge_faces(result_faces)
        sampled_edges = [
            edge
            for edge in edge_faces
            if abs(result_vertices[edge[0], 1]) < 1e-12
            and abs(result_vertices[edge[1], 1]) < 1e-12
        ]

        self.assertEqual(stats["splits"], 7, stats)
        self.assertEqual(len(sampled_edges), 8)
        self.assertTrue(all(len(edge_faces[edge]) == 2 for edge in sampled_edges))
        self.assertLessEqual(stats["final_max_length"], 2.0 + 1e-8)
        self.assertEqual(set(tracked_roots), set(sampled_edges))
        self.assertEqual(set(tracked_roots.values()), {7})
        result_labels = stats["face_labels"]
        self.assertEqual(len(result_labels), len(result_faces))
        self.assertEqual(np.count_nonzero(result_labels == 3), 8)
        self.assertEqual(np.count_nonzero(result_labels == 9), 8)

    def test_feature_target_refines_hard_chains_without_global_refinement(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        reference = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

        refined_vertices, _, stats = (
            original_constrained.refine_original_mesh_by_longest_edge(
                reference,
                max_edge_length=100.0,
                feature_angle_degrees=30.0,
                feature_target_edge_length=2.0,
                flip_passes=0,
            )
        )

        self.assertGreater(stats["feature_driven_splits"], 0)
        self.assertLessEqual(stats["final_feature_max_length"], 2.0 + 1e-8)
        self.assertGreater(stats["final_max_length"], 2.0)
        self.assertGreater(len(refined_vertices), len(vertices))

    def test_feature_snap_backtracks_instead_of_flipping_triangle(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        corner_target = np.array([[2.0, 2.0, 0.0]])
        constraints = feature_optimize.FeatureConstraintMap(
            reference_edges=np.empty((0, 2), dtype=np.int64),
            reference_segments=np.empty((0, 2, 3), dtype=np.float64),
            matched_edges=np.empty((0, 2), dtype=np.int64),
            feature_vertices=np.empty(0, dtype=np.int64),
            feature_segment_ids=np.empty(0, dtype=np.int64),
            corner_vertices=np.array([0], dtype=np.int64),
            corner_targets=corner_target,
            match_tolerance=1.0,
        )

        snapped, relaxed = feature_optimize._snap_to_feature_constraints(
            vertices,
            faces,
            constraints,
            minimum_area_squared=1e-24,
        )

        np.testing.assert_array_equal(relaxed, [0])
        self.assertFalse(np.allclose(snapped[0], corner_target[0]))
        self.assertGreater(np.linalg.norm(snapped[0] - vertices[0]), 0.0)
        self.assertTrue(
            feature_optimize._valid_relocation(
                vertices,
                snapped,
                faces,
                minimum_area_squared=1e-24,
            )
        )

    def test_flips_only_when_local_triangle_quality_improves(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
        before = feature_optimize.mesh_quality_metrics(vertices, faces)

        flipped_faces, flip_count = feature_optimize._flip_quality_edges(
            vertices,
            faces,
            np.empty((0, 2), dtype=np.int64),
            passes=1,
        )
        after = feature_optimize.mesh_quality_metrics(
            vertices,
            flipped_faces,
        )

        self.assertEqual(flip_count, 1)
        self.assertIn(
            (0, 3),
            {
                edge
                for face in flipped_faces
                for edge in feature_optimize._face_edges(face)
            },
        )
        self.assertGreater(
            after["minimum_triangle_quality"],
            before["minimum_triangle_quality"],
        )

    def test_coplanar_flip_filter_preserves_fold_edge(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)

        filtered_faces, flip_count = feature_optimize._flip_quality_edges(
            vertices,
            faces,
            np.empty((0, 2), dtype=np.int64),
            passes=4,
            maximum_dihedral_degrees=0.1,
        )

        self.assertEqual(flip_count, 0)
        np.testing.assert_array_equal(filtered_faces, faces)

    def test_coplanar_flip_respects_maximum_new_edge_length(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)

        filtered_faces, flip_count = feature_optimize._flip_quality_edges(
            vertices,
            faces,
            np.empty((0, 2), dtype=np.int64),
            passes=4,
            maximum_dihedral_degrees=0.1,
            maximum_edge_length=1.0,
        )

        self.assertEqual(flip_count, 0)
        np.testing.assert_array_equal(filtered_faces, faces)

    def test_preferred_coplanar_seam_flips_at_equal_quality(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        flipped_faces, flip_count = feature_optimize._flip_quality_edges(
            vertices,
            faces,
            np.empty((0, 2), dtype=np.int64),
            passes=1,
            maximum_dihedral_degrees=0.1,
            preferred_edges=np.array([[0, 2]], dtype=np.int64),
        )

        self.assertEqual(flip_count, 1)
        output_edges = {
            edge
            for face in flipped_faces
            for edge in feature_optimize._face_edges(face)
        }
        self.assertNotIn((0, 2), output_edges)
        self.assertIn((1, 3), output_edges)

    def test_maps_boundary_features_and_geometric_corners(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        reference = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

        constraints = feature_optimize.build_feature_constraint_map(
            reference,
            vertices,
            faces,
            resolution=64,
            feature_angle_degrees=30.0,
        )

        self.assertEqual(len(constraints.reference_edges), 4)
        self.assertEqual(len(constraints.matched_edges), 4)
        np.testing.assert_array_equal(
            np.sort(constraints.corner_vertices),
            np.arange(4),
        )

    def test_relocates_interior_vertex_but_keeps_feature_corners(self):
        reference_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        reference_faces = np.array(
            [[0, 1, 2], [0, 2, 3]],
            dtype=np.int64,
        )
        reference = trimesh.Trimesh(
            vertices=reference_vertices,
            faces=reference_faces,
            process=False,
        )
        vertices = np.vstack(
            (reference_vertices, np.array([[0.7, 0.4, 0.0]]))
        )
        faces = np.array(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=np.int64,
        )

        optimized, optimized_faces, stats = (
            feature_optimize.optimize_feature_constrained_mesh(
                reference,
                vertices,
                faces,
                resolution=64,
                feature_angle_degrees=30.0,
                iterations=5,
                smoothing_step=0.5,
                flip_passes=0,
            )
        )

        np.testing.assert_allclose(optimized[:4], reference_vertices)
        self.assertLess(
            np.linalg.norm(optimized[4] - [0.5, 0.5, 0.0]),
            np.linalg.norm(vertices[4] - [0.5, 0.5, 0.0]),
        )
        self.assertLess(
            stats["final_metrics"]["energy"],
            stats["initial_metrics"]["energy"],
        )
        self.assertLessEqual(stats["maximum_feature_distance"], 1e-12)
        self.assertLessEqual(stats["maximum_corner_error"], 1e-12)
        self.assertEqual(stats["relaxed_constraint_count"], 0)
        np.testing.assert_array_equal(optimized_faces, faces)


if __name__ == "__main__":
    unittest.main()
