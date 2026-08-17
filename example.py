import argparse
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import trimesh


LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parent / "simp_cuda"
if str(LOCAL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))

from pamo import PaMO, PaSP


SUPPORTED_INPUT_EXTENSIONS = {'.obj', '.stl', '.ply'}


def load_feature_edge_indices(feature_edge_path):
    """Load zero-based input-mesh edge vertex pairs from a text file."""
    feature_edge_path = Path(feature_edge_path)
    if not feature_edge_path.is_file():
        raise FileNotFoundError(
            "Feature-edge file does not exist: {}".format(feature_edge_path)
        )

    edges = []
    for line_number, line in enumerate(
        feature_edge_path.read_text(encoding='utf-8').splitlines(),
        start=1,
    ):
        line = line.split('#', 1)[0].replace(',', ' ').strip()
        if not line:
            continue
        values = line.split()
        if len(values) != 2:
            raise ValueError(
                "Feature-edge file line {} must contain two vertex indices: "
                "{}".format(line_number, feature_edge_path)
            )
        try:
            edge = (int(values[0]), int(values[1]))
        except ValueError as error:
            raise ValueError(
                "Feature-edge file line {} contains a non-integer index: "
                "{}".format(line_number, feature_edge_path)
            ) from error
        edges.append(edge)

    if not edges:
        raise ValueError(
            "Feature-edge file contains no edges: {}".format(feature_edge_path)
        )
    return np.asarray(edges, dtype=np.int64)


def load_input_mesh(input_path):
    """Load an OBJ, STL, or PLY triangle mesh and clean its topology."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError("Input mesh does not exist: {}".format(input_path))
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            "Unsupported input format '{}'. Supported formats: {}".format(
                suffix or "<none>",
                ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS)),
            )
        )

    input_mesh = trimesh.load(input_path, force='mesh', process=False)
    if not isinstance(input_mesh, trimesh.Trimesh) or input_mesh.is_empty:
        if suffix == '.ply':
            raise ValueError(
                "PLY input must contain polygon faces; point-cloud-only PLY files "
                "are not supported: {}".format(input_path)
            )
        raise ValueError("Input does not contain a valid mesh: {}".format(input_path))

    if suffix in {'.stl', '.ply'}:
        verts_before = len(input_mesh.vertices)
        faces_before = len(input_mesh.faces)

        # STL files commonly repeat all three vertices for every triangle.
        # PLY files may also contain unwelded or duplicate geometry.
        # Weld vertices before PaMO builds edge adjacency.
        input_mesh.merge_vertices()
        input_mesh.update_faces(input_mesh.unique_faces())
        input_mesh.update_faces(input_mesh.nondegenerate_faces())
        input_mesh.remove_unreferenced_vertices()
        input_mesh.fix_normals(multibody=True)

        print(
            "{} cleanup: verts {} -> {}, faces {} -> {}".format(
                suffix[1:].upper(),
                verts_before,
                len(input_mesh.vertices),
                faces_before,
                len(input_mesh.faces),
            )
        )

    if len(input_mesh.vertices) == 0 or len(input_mesh.faces) == 0:
        raise ValueError("Input mesh is empty after cleanup: {}".format(input_path))
    if input_mesh.faces.ndim != 2 or input_mesh.faces.shape[1] != 3:
        raise ValueError("PaMO requires a triangular mesh: {}".format(input_path))
    if not np.isfinite(input_mesh.vertices).all():
        raise ValueError("Input mesh contains non-finite vertex coordinates: {}".format(input_path))

    return input_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='./mesh/BirdHouse_B019SXLRJ2_MetalLeafRoofGreenWalls_TU.obj',
        help="Input triangle mesh in OBJ, STL, or PLY format",
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='./BirdHouse_pamo.obj',
        help="Output mesh path; use .ply to export PLY",
    )
    parser.add_argument('-r', '--ratio', type=float, default=0.1)
    parser.add_argument('-mv', '--min-vertex', type=int, default=0)
    parser.add_argument('--disable_stage1', action='store_true', help="Disable remeshing")
    parser.add_argument('--disable_stage3', action='store_true', help="Disable safe projection")
    remesh_mode = parser.add_mutually_exclusive_group()
    remesh_mode.add_argument(
        '--remesh-only',
        action='store_true',
        help="Run only SDF remeshing; skip simplification and safe projection",
    )
    remesh_mode.add_argument(
        '--feature-remesh',
        action='store_true',
        help="Run SDF remeshing plus safe projection; skip simplification",
    )
    remesh_mode.add_argument(
        '--feature-optimize',
        action='store_true',
        help=(
            "Jointly preserve original feature curves and improve triangle "
            "shape using constrained relocation and non-feature edge flips"
        ),
    )
    remesh_mode.add_argument(
        '--surface-sample-remesh',
        action='store_true',
        help=(
            "Sample the original triangles directly on CUDA, locally "
            "retriangulate them, and optimize only non-feature connections"
        ),
    )
    remesh_mode.add_argument(
        '--sdf-optimize',
        action='store_true',
        help=(
            "Run SDF remeshing plus topology-preserving tangential quality "
            "optimization; skip simplification"
        ),
    )
    remesh_mode.add_argument(
        '--original-constrained-remesh',
        action='store_true',
        help=(
            "Strictly retain original connectivity, preserve every input edge "
            "sharper than the feature threshold, and refine by longest edge"
        ),
    )
    parser.add_argument(
        '--remesh-resolution',
        type=int,
        choices=(64, 128, 256),
        default=256,
        help="SDF grid resolution used by remesh modes (default: 256)",
    )
    parser.add_argument(
        '--sdf-mode',
        choices=('auto', 'exact', 'repair'),
        default='auto',
        help=(
            "SDF semantics: auto uses exact SDF=0 for valid closed meshes and "
            "a repair envelope otherwise; exact rejects invalid/open inputs; "
            "repair always extracts a 0.9-voxel unsigned-distance envelope"
        ),
    )
    parser.add_argument(
        '--projection-iterations',
        type=int,
        default=5,
        help=(
            "Safe-projection iterations used by feature remesh/optimize modes "
            "(default: 5)"
        ),
    )
    parser.add_argument(
        '--feature-edges',
        type=str,
        default=None,
        help=(
            "Optional text file of zero-based input vertex-index pairs; "
            "otherwise sharp and boundary edges are detected automatically"
        ),
    )
    parser.add_argument(
        '--feature-edge-angle',
        type=float,
        default=45.0,
        help="Automatic sharp-edge dihedral threshold in degrees (default: 45)",
    )
    parser.add_argument(
        '--feature-edge-target-length',
        type=float,
        default=None,
        help=(
            "Split matched feature edges until no longer than this world-space "
            "length; valid with --feature-remesh or --feature-optimize"
        ),
    )
    parser.add_argument(
        '--feature-edge-match-tolerance',
        type=float,
        default=None,
        help=(
            "Maximum distance for matching remeshed edges to input features; "
            "default is three SDF voxels"
        ),
    )
    parser.add_argument(
        '--feature-edge-max-splits',
        type=int,
        default=100000,
        help="Safety limit for feature-edge splits (default: 100000)",
    )
    parser.add_argument(
        '--sdf-optimize-iterations',
        type=int,
        default=20,
        help="Tangential SDF optimization iterations (default: 20)",
    )
    parser.add_argument(
        '--sdf-smoothing-step',
        type=float,
        default=0.2,
        help="Tangential smoothing step in (0, 1] (default: 0.2)",
    )
    parser.add_argument(
        '--sdf-projection-steps',
        type=int,
        default=3,
        help="SDF zero-set projection steps per iteration (default: 3)",
    )
    parser.add_argument(
        '--sdf-feature-angle',
        type=float,
        default=45.0,
        help="Lock SDF-mesh edges sharper than this angle (default: 45)",
    )
    parser.add_argument(
        '--feature-quality-iterations',
        type=int,
        default=5,
        help=(
            "Original-surface feature-constrained relocation iterations used "
            "by --feature-optimize (default: 5)"
        ),
    )
    parser.add_argument(
        '--feature-quality-step',
        type=float,
        default=0.2,
        help=(
            "Feature-constrained tangential relocation step in (0, 1], "
            "default=0.2"
        ),
    )
    parser.add_argument(
        '--feature-flip-passes',
        type=int,
        default=2,
        help=(
            "Quality-driven non-feature edge-flip passes used by "
            "--feature-optimize (default: 2)"
        ),
    )
    parser.add_argument(
        '--surface-sample-count',
        type=int,
        default=10000,
        help=(
            "Requested original-surface sample count used by "
            "--surface-sample-remesh (default: 10000)"
        ),
    )
    parser.add_argument(
        '--surface-poisson-radius',
        type=float,
        default=None,
        help=(
            "World-space grid-Poisson radius; default is derived from surface "
            "area and requested sample count"
        ),
    )
    parser.add_argument(
        '--surface-sample-oversample',
        type=int,
        default=4,
        help="Area-sampling candidate multiplier before Poisson filtering",
    )
    parser.add_argument(
        '--surface-sample-seed',
        type=int,
        default=0,
        help="CUDA surface-sampling random seed (default: 0)",
    )
    parser.add_argument(
        '--surface-flip-passes',
        type=int,
        default=5,
        help="Conflict-free CUDA quality-flip batches (default: 5)",
    )
    parser.add_argument(
        '--surface-relax-iterations',
        type=int,
        default=3,
        help="Inserted-point CUDA relaxation iterations (default: 3)",
    )
    parser.add_argument(
        '--surface-relax-step',
        type=float,
        default=0.2,
        help="Inserted-point relaxation step in (0, 1] (default: 0.2)",
    )
    parser.add_argument(
        '--surface-barycentric-margin',
        type=float,
        default=0.08,
        help=(
            "Keep random samples this far from source-triangle edges in "
            "barycentric coordinates (default: 0.08)"
        ),
    )
    parser.add_argument(
        '--surface-min-source-quality',
        type=float,
        default=1e-4,
        help=(
            "Do not insert points into numerically thin source triangles "
            "(default quality threshold: 1e-4)"
        ),
    )
    parser.add_argument(
        '--surface-min-source-area-ratio',
        type=float,
        default=0.25,
        help=(
            "Only sample source faces with area at least this value times "
            "Poisson radius squared (default: 0.25)"
        ),
    )
    parser.add_argument(
        '--surface-max-edge-ratio',
        type=float,
        default=2.0,
        help=(
            "Maximum output edge length divided by the Poisson radius; "
            "longer edges are bisected on CUDA (default: 2.0)"
        ),
    )
    parser.add_argument(
        '--surface-min-edge-ratio',
        type=float,
        default=0.5,
        help=(
            "Short-edge collapse threshold divided by the Poisson radius; "
            "collapses stay inside smooth patches (default: 0.5)"
        ),
    )
    parser.add_argument(
        '--surface-split-passes',
        type=int,
        default=64,
        help="Maximum conflict-free CUDA long-edge split batches (default: 64)",
    )
    parser.add_argument(
        '--surface-collapse-passes',
        type=int,
        default=24,
        help="Maximum feature-safe CUDA short-edge collapse batches (default: 24)",
    )
    parser.add_argument(
        '--surface-protect-source-quality',
        type=float,
        default=0.8,
        help=(
            "Make source faces at or above this triangle-quality threshold "
            "refinement-only; use 0 to disable (default: 0.8)"
        ),
    )
    parser.add_argument(
        '--surface-max-normal-deviation',
        type=float,
        default=5.0,
        help=(
            "Maximum source-normal cone half-angle allowed in a collapse "
            "one-ring, in degrees (default: 5)"
        ),
    )
    parser.add_argument(
        '--surface-max-deviation-ratio',
        type=float,
        default=0.05,
        help=(
            "Maximum collapse/flip source-plane deviation divided by the "
            "Poisson radius (default: 0.05)"
        ),
    )
    parser.add_argument(
        '--surface-min-collapse-quality',
        type=float,
        default=0.25,
        help=(
            "Also consider edges adjacent to triangles below this quality "
            "for controlled collapse; use 0 to disable (default: 0.25)"
        ),
    )
    parser.add_argument(
        '--surface-coplanar-angle',
        type=float,
        default=1.0,
        help=(
            "Treat adjacent source faces within this normal angle as a "
            "single coplanar patch for edge refinement (default: 1 degree)"
        ),
    )
    parser.add_argument(
        '--constraint-projection-distance',
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--constraint-allow-open-surface',
        action='store_true',
        help=(
            "Allow consistently wound thin sheets; boundary and non-manifold "
            "edges remain hard constraints"
        ),
    )
    parser.add_argument(
        '--constraint-feature-distance',
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--coplanar-angle-tolerance',
        type=float,
        default=0.1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--coplanar-distance-ratio',
        type=float,
        default=1e-6,
        help=(
            "Maximum point-to-seed-plane distance divided by the model "
            "diagonal during planar patch classification (default: 1e-6)"
        ),
    )
    parser.add_argument(
        '--constraint-max-edge-length',
        '--constraint-edge-target-length',
        dest='constraint_max_edge_length',
        type=float,
        default=None,
        help=(
            "Globally bisect longest edges until every edge is no longer than "
            "this value; no insertion occurs when already satisfied"
        ),
    )
    parser.add_argument(
        '--constraint-feature-angle',
        '--constraint-edge-output-angle',
        dest='constraint_feature_angle',
        type=float,
        default=5.0,
        help=(
            "Input dihedral angle above which an edge is a hard constraint "
            "which may only be split, default=5 degrees"
        ),
    )
    parser.add_argument(
        '--constraint-feature-target-edge-length',
        type=float,
        default=None,
        help=(
            "Maximum edge length on hard feature, boundary, and non-manifold "
            "chains; smooth high-quality regions retain the coarser global "
            "limit"
        ),
    )
    parser.add_argument(
        '--constraint-max-splits',
        '--constraint-edge-max-splits',
        dest='constraint_max_splits',
        type=int,
        default=None,
        help=(
            "Safety limit for longest-edge splits; by default it is "
            "estimated automatically with a 4x conformity margin"
        ),
    )
    parser.add_argument(
        '--constraint-flip-passes',
        type=int,
        default=8,
        help=(
            "Quality-driven coplanar non-feature edge-flip passes "
            "(default: 8)"
        ),
    )
    parser.add_argument(
        '--constraint-flip-minimum-valence',
        type=int,
        default=None,
        help=(
            "Only consider coplanar edges touching a vertex with at least "
            "this valence; useful for fast planar-fan cleanup"
        ),
    )
    parser.add_argument(
        '--constraint-flip-maximum-candidate-quality',
        type=float,
        default=None,
        help=(
            "Also consider edges adjacent to triangles below this normalized "
            "quality; useful after strict longest-edge splitting"
        ),
    )
    parser.add_argument(
        '--constraint-planar-fan-minimum-valence',
        type=int,
        default=None,
        help=(
            "Replace convex planar center fans at or above this valence by "
            "a uniform local triangulation"
        ),
    )
    parser.add_argument(
        '--constraint-planar-annulus-minimum-faces',
        type=int,
        default=None,
        help=(
            "Uniformly retriangulate planar facets with holes when they "
            "contain at least this many faces"
        ),
    )
    parser.add_argument(
        '--constraint-cylinder-minimum-faces',
        type=int,
        default=None,
        help="Remesh detected cylindrical walls at or above this face count",
    )
    parser.add_argument(
        '--constraint-cylinder-radius-tolerance',
        type=float,
        default=1e-3,
        help="Relative circle/cylinder fitting tolerance (default: 0.001)",
    )
    parser.add_argument(
        '--constraint-cylinder-target-edge-ratio',
        type=float,
        default=1.0,
        help="Cylinder target edge length divided by boundary median (default: 1)",
    )
    parser.add_argument(
        '--constraint-partial-cylinder-minimum-faces',
        type=int,
        default=None,
        help="Remesh open/trimmed cylinder patches at or above this face count",
    )
    parser.add_argument(
        '--constraint-partial-cylinder-radius-tolerance',
        type=float,
        default=2e-3,
        help="Relative radius-fit tolerance for partial cylinders",
    )
    parser.add_argument(
        '--constraint-partial-cylinder-normal-tolerance',
        type=float,
        default=2e-2,
        help="Maximum RMS face-normal component along the fitted cylinder axis",
    )
    parser.add_argument(
        '--constraint-partial-cylinder-minimum-angle',
        type=float,
        default=30.0,
        help="Minimum angular coverage in degrees for partial-cylinder cleanup",
    )
    parser.add_argument(
        '--constraint-rounded-fillet-minimum-faces',
        type=int,
        default=None,
        help="Isolate and remesh smooth cylindrical fillet bands",
    )
    parser.add_argument(
        '--constraint-rounded-fillet-minimum-curvature',
        type=float,
        default=0.2,
        help="Minimum nonzero dihedral in degrees used to isolate fillets",
    )
    parser.add_argument(
        '--constraint-extrusion-region-minimum-faces',
        type=int,
        default=None,
        help=(
            "Unwrap and boundary-rebuild classified extrusion/developable "
            "regions at or above this face count"
        ),
    )
    parser.add_argument(
        '--constraint-extrusion-maximum-input-quality',
        type=float,
        default=0.05,
        help=(
            "Only rebuild extrusion/developable regions whose input P5 "
            "triangle quality is below this threshold (default: 0.05)"
        ),
    )
    parser.add_argument(
        '--constraint-planar-region-minimum-faces',
        type=int,
        default=None,
        help="Uniformly retriangulate solid planar regions of at least this size",
    )
    parser.add_argument(
        '--constraint-planar-largest-opposed-pair-only',
        action='store_true',
        help=(
            "Only boundary-rebuild the largest planar patch and its largest "
            "opposite-facing mate (the primary top/bottom plate skins)"
        ),
    )
    parser.add_argument(
        '--constraint-planar-target-edge-length',
        type=float,
        default=None,
        help=(
            "Maximum interior grid spacing used when retriangulating planar "
            "regions with or without holes"
        ),
    )
    parser.add_argument(
        '--constraint-planar-minimum-angle',
        type=float,
        default=None,
        help=(
            "Minimum triangle angle requested from boundary-only planar "
            "or developable reconstruction; SciPy fallback is used when "
            "the triangle package is unavailable"
        ),
    )
    parser.add_argument(
        '--constraint-quality-iterations',
        type=int,
        default=20,
        help="Feature-safe quality relocation iterations (default: 20)",
    )
    parser.add_argument(
        '--constraint-quality-step',
        type=float,
        default=0.4,
        help="Feature-safe quality relocation step (default: 0.4)",
    )
    parser.add_argument(
        '--constraint-quality-flip-passes',
        type=int,
        default=12,
        help="Feature-safe global quality flip passes (default: 12)",
    )
    args = parser.parse_args()

    if (
        args.remesh_only
        or args.feature_remesh
        or args.feature_optimize
        or args.surface_sample_remesh
        or args.sdf_optimize
        or args.original_constrained_remesh
    ) and args.disable_stage1:
        parser.error("Remesh modes cannot be combined with --disable_stage1")
    if (
        args.feature_remesh or args.feature_optimize
    ) and args.disable_stage3:
        parser.error(
            "Feature remesh/optimize modes cannot be combined with "
            "--disable_stage3"
        )
    if args.projection_iterations <= 0:
        parser.error("--projection-iterations must be a positive integer")
    if (
        args.feature_edges is not None
        or args.feature_edge_target_length is not None
        or args.feature_edge_match_tolerance is not None
    ) and not (
        args.feature_remesh
        or args.feature_optimize
        or args.surface_sample_remesh
    ):
        parser.error(
            "Feature-edge options require --feature-remesh or "
            "--feature-optimize"
        )
    if (
        args.feature_remesh
        and args.feature_edges is not None
        and args.feature_edge_target_length is None
    ):
        parser.error("--feature-edges requires --feature-edge-target-length")
    if (
        args.feature_edge_target_length is not None
        and args.feature_edge_target_length <= 0.0
    ):
        parser.error("--feature-edge-target-length must be positive")
    if not 0.0 < args.feature_edge_angle < 180.0:
        parser.error("--feature-edge-angle must be between 0 and 180")
    if (
        args.feature_edge_match_tolerance is not None
        and args.feature_edge_match_tolerance <= 0.0
    ):
        parser.error("--feature-edge-match-tolerance must be positive")
    if args.feature_edge_max_splits <= 0:
        parser.error("--feature-edge-max-splits must be positive")
    if args.sdf_optimize_iterations <= 0:
        parser.error("--sdf-optimize-iterations must be positive")
    if not 0.0 < args.sdf_smoothing_step <= 1.0:
        parser.error("--sdf-smoothing-step must be in (0, 1]")
    if args.sdf_projection_steps <= 0:
        parser.error("--sdf-projection-steps must be positive")
    if not 0.0 < args.sdf_feature_angle < 180.0:
        parser.error("--sdf-feature-angle must be between 0 and 180")
    if args.feature_quality_iterations <= 0:
        parser.error("--feature-quality-iterations must be positive")
    if not 0.0 < args.feature_quality_step <= 1.0:
        parser.error("--feature-quality-step must be in (0, 1]")
    if args.feature_flip_passes < 0:
        parser.error("--feature-flip-passes must be non-negative")
    if args.surface_sample_count <= 0:
        parser.error("--surface-sample-count must be positive")
    if (
        args.surface_poisson_radius is not None
        and args.surface_poisson_radius <= 0.0
    ):
        parser.error("--surface-poisson-radius must be positive")
    if args.surface_sample_oversample <= 0:
        parser.error("--surface-sample-oversample must be positive")
    if args.surface_flip_passes < 0:
        parser.error("--surface-flip-passes must be non-negative")
    if args.surface_relax_iterations < 0:
        parser.error("--surface-relax-iterations must be non-negative")
    if not 0.0 < args.surface_relax_step <= 1.0:
        parser.error("--surface-relax-step must be in (0, 1]")
    if not 0.0 <= args.surface_barycentric_margin < 1.0 / 3.0:
        parser.error("--surface-barycentric-margin must be in [0, 1/3)")
    if not 0.0 <= args.surface_min_source_quality < 1.0:
        parser.error("--surface-min-source-quality must be in [0, 1)")
    if args.surface_min_source_area_ratio < 0.0:
        parser.error("--surface-min-source-area-ratio must be non-negative")
    if args.surface_max_edge_ratio <= 0.0:
        parser.error("--surface-max-edge-ratio must be positive")
    if not (
        0.0
        <= args.surface_min_edge_ratio
        < args.surface_max_edge_ratio
    ):
        parser.error(
            "--surface-min-edge-ratio must be non-negative and smaller "
            "than --surface-max-edge-ratio"
        )
    if args.surface_split_passes <= 0:
        parser.error("--surface-split-passes must be positive")
    if args.surface_collapse_passes < 0:
        parser.error("--surface-collapse-passes must be non-negative")
    if not 0.0 <= args.surface_protect_source_quality <= 1.0:
        parser.error(
            "--surface-protect-source-quality must be in [0, 1]"
        )
    if not 0.0 <= args.surface_max_normal_deviation <= 180.0:
        parser.error(
            "--surface-max-normal-deviation must be in [0, 180]"
        )
    if args.surface_max_deviation_ratio < 0.0:
        parser.error(
            "--surface-max-deviation-ratio must be non-negative"
        )
    if not 0.0 <= args.surface_min_collapse_quality <= 1.0:
        parser.error(
            "--surface-min-collapse-quality must be in [0, 1]"
        )
    if not 0.0 <= args.surface_coplanar_angle < 180.0:
        parser.error(
            "--surface-coplanar-angle must be in [0, 180)"
        )
    constraint_options_used = (
        args.constraint_projection_distance is not None
        or args.constraint_feature_distance is not None
        or args.constraint_max_edge_length is not None
        or args.constraint_feature_target_edge_length is not None
        or args.constraint_planar_minimum_angle is not None
        or args.constraint_extrusion_region_minimum_faces is not None
        or args.constraint_allow_open_surface
    )
    if (
        constraint_options_used
        and not args.original_constrained_remesh
    ):
        parser.error(
            "Original-constraint options require "
            "--original-constrained-remesh"
        )
    if (
        args.constraint_projection_distance is not None
        and args.constraint_projection_distance <= 0.0
    ):
        parser.error("--constraint-projection-distance must be positive")
    if (
        args.constraint_feature_distance is not None
        and args.constraint_feature_distance <= 0.0
    ):
        parser.error("--constraint-feature-distance must be positive")
    if not 0.0 <= args.coplanar_angle_tolerance < 180.0:
        parser.error("--coplanar-angle-tolerance must be in [0, 180)")
    if args.coplanar_distance_ratio < 0.0:
        parser.error("--coplanar-distance-ratio must be non-negative")
    if (
        args.constraint_max_edge_length is not None
        and args.constraint_max_edge_length <= 0.0
    ):
        parser.error("--constraint-max-edge-length must be positive")
    if not 0.0 <= args.constraint_feature_angle < 180.0:
        parser.error("--constraint-feature-angle must be in [0, 180)")
    if (
        args.constraint_feature_target_edge_length is not None
        and args.constraint_feature_target_edge_length <= 0.0
    ):
        parser.error(
            "--constraint-feature-target-edge-length must be positive"
        )
    if (
        args.constraint_max_splits is not None
        and args.constraint_max_splits <= 0
    ):
        parser.error("--constraint-max-splits must be positive")
    if args.constraint_flip_passes < 0:
        parser.error("--constraint-flip-passes must be non-negative")
    if (
        args.constraint_flip_minimum_valence is not None
        and args.constraint_flip_minimum_valence < 4
    ):
        parser.error("--constraint-flip-minimum-valence must be at least 4")
    if (
        args.constraint_flip_maximum_candidate_quality is not None
        and not 0.0 < args.constraint_flip_maximum_candidate_quality <= 1.0
    ):
        parser.error(
            "--constraint-flip-maximum-candidate-quality must be in (0, 1]"
        )
    if (
        args.constraint_planar_fan_minimum_valence is not None
        and args.constraint_planar_fan_minimum_valence < 6
    ):
        parser.error(
            "--constraint-planar-fan-minimum-valence must be at least 6"
        )
    if (
        args.constraint_planar_annulus_minimum_faces is not None
        and args.constraint_planar_annulus_minimum_faces < 4
    ):
        parser.error(
            "--constraint-planar-annulus-minimum-faces must be at least 4"
        )
    if (
        args.constraint_cylinder_minimum_faces is not None
        and args.constraint_cylinder_minimum_faces < 4
    ):
        parser.error("--constraint-cylinder-minimum-faces must be at least 4")
    if args.constraint_cylinder_radius_tolerance <= 0.0:
        parser.error("--constraint-cylinder-radius-tolerance must be positive")
    if args.constraint_cylinder_target_edge_ratio <= 0.0:
        parser.error("--constraint-cylinder-target-edge-ratio must be positive")
    if (
        args.constraint_partial_cylinder_minimum_faces is not None
        and args.constraint_partial_cylinder_minimum_faces < 4
    ):
        parser.error(
            "--constraint-partial-cylinder-minimum-faces must be at least 4"
        )
    if args.constraint_partial_cylinder_radius_tolerance <= 0.0:
        parser.error(
            "--constraint-partial-cylinder-radius-tolerance must be positive"
        )
    if args.constraint_partial_cylinder_normal_tolerance <= 0.0:
        parser.error(
            "--constraint-partial-cylinder-normal-tolerance must be positive"
        )
    if not 0.0 < args.constraint_partial_cylinder_minimum_angle < 360.0:
        parser.error(
            "--constraint-partial-cylinder-minimum-angle must be in (0, 360)"
        )
    if (
        args.constraint_rounded_fillet_minimum_faces is not None
        and args.constraint_rounded_fillet_minimum_faces < 4
    ):
        parser.error(
            "--constraint-rounded-fillet-minimum-faces must be at least 4"
        )
    if args.constraint_rounded_fillet_minimum_curvature <= 0.0:
        parser.error(
            "--constraint-rounded-fillet-minimum-curvature must be positive"
        )
    if (
        args.constraint_extrusion_region_minimum_faces is not None
        and args.constraint_extrusion_region_minimum_faces < 2
    ):
        parser.error(
            "--constraint-extrusion-region-minimum-faces must be at least 2"
        )
    if not 0.0 < args.constraint_extrusion_maximum_input_quality <= 1.0:
        parser.error(
            "--constraint-extrusion-maximum-input-quality must be in (0, 1]"
        )
    if (
        args.constraint_planar_region_minimum_faces is not None
        and args.constraint_planar_region_minimum_faces < 4
    ):
        parser.error(
            "--constraint-planar-region-minimum-faces must be at least 4"
        )
    if (
        args.constraint_planar_target_edge_length is not None
        and args.constraint_planar_target_edge_length <= 0.0
    ):
        parser.error(
            "--constraint-planar-target-edge-length must be positive"
        )
    if (
        args.constraint_planar_largest_opposed_pair_only
        and args.constraint_planar_annulus_minimum_faces is None
        and args.constraint_planar_region_minimum_faces is None
    ):
        parser.error(
            "--constraint-planar-largest-opposed-pair-only requires planar "
            "annulus or planar region reconstruction"
        )
    if (
        args.constraint_planar_minimum_angle is not None
        and not 0.0 < args.constraint_planar_minimum_angle < 34.0
    ):
        parser.error(
            "--constraint-planar-minimum-angle must be in (0, 34)"
        )
    if (
        args.constraint_planar_minimum_angle is not None
        and args.constraint_planar_annulus_minimum_faces is None
        and args.constraint_planar_region_minimum_faces is None
        and args.constraint_extrusion_region_minimum_faces is None
    ):
        parser.error(
            "--constraint-planar-minimum-angle requires planar annulus or "
            "planar region reconstruction"
        )
    if args.constraint_quality_iterations < 0:
        parser.error("--constraint-quality-iterations must be non-negative")
    if not 0.0 < args.constraint_quality_step <= 1.0:
        parser.error("--constraint-quality-step must be in (0, 1]")
    if args.constraint_quality_flip_passes < 0:
        parser.error("--constraint-quality-flip-passes must be non-negative")


    input_mesh = load_input_mesh(args.input)
    feature_edges = (
        load_feature_edge_indices(args.feature_edges)
        if args.feature_edges is not None
        else None
    )
    print("# of input verts : {}".format(len(input_mesh.vertices)))
    print("# of input faces : {}".format(len(input_mesh.faces)))

    points = torch.from_numpy(input_mesh.vertices).float().cuda()
    triangles = torch.from_numpy(input_mesh.faces).int().cuda()

    pamo = PaMO(
        input_mesh,
        use_stage1=(
            True
            if (
                args.remesh_only
                or args.feature_remesh
                or args.feature_optimize
                or args.surface_sample_remesh
                or args.sdf_optimize
                or args.original_constrained_remesh
            )
            else not args.disable_stage1
        ),
        use_stage3=(
            args.feature_remesh
            or args.feature_optimize
            or (
                not args.remesh_only
                and not args.sdf_optimize
                and not args.original_constrained_remesh
                and not args.surface_sample_remesh
                and not args.disable_stage3
            )
        ),
    )
    start = time.time()
    if args.remesh_only:
        verts, faces = pamo.remesh_only(
            points,
            triangles,
            resolution=args.remesh_resolution,
            sdf_mode=args.sdf_mode,
        )
    elif args.surface_sample_remesh:
        verts, faces = pamo.surface_sample_remesh(
            points,
            triangles,
            sample_count=args.surface_sample_count,
            poisson_radius=args.surface_poisson_radius,
            oversample=args.surface_sample_oversample,
            seed=args.surface_sample_seed,
            feature_edges=feature_edges,
            feature_edge_angle=args.feature_edge_angle,
            flip_passes=args.surface_flip_passes,
            relax_iterations=args.surface_relax_iterations,
            smoothing_step=args.surface_relax_step,
            barycentric_margin=args.surface_barycentric_margin,
            minimum_source_quality=args.surface_min_source_quality,
            minimum_source_area_ratio=args.surface_min_source_area_ratio,
            maximum_edge_ratio=args.surface_max_edge_ratio,
            minimum_edge_ratio=args.surface_min_edge_ratio,
            split_passes=args.surface_split_passes,
            collapse_passes=args.surface_collapse_passes,
            protected_source_quality=(
                args.surface_protect_source_quality
            ),
            maximum_normal_deviation_degrees=(
                args.surface_max_normal_deviation
            ),
            maximum_surface_deviation_ratio=(
                args.surface_max_deviation_ratio
            ),
            minimum_collapse_quality=args.surface_min_collapse_quality,
            coplanar_angle_degrees=args.surface_coplanar_angle,
        )
    elif args.feature_remesh:
        verts, faces = pamo.feature_remesh(
            points,
            triangles,
            resolution=args.remesh_resolution,
            projection_iterations=args.projection_iterations,
            feature_edge_target_length=args.feature_edge_target_length,
            feature_edges=feature_edges,
            feature_edge_angle=args.feature_edge_angle,
            feature_edge_match_tolerance=args.feature_edge_match_tolerance,
            feature_edge_max_splits=args.feature_edge_max_splits,
            sdf_mode=args.sdf_mode,
        )
    elif args.feature_optimize:
        verts, faces = pamo.feature_optimize(
            points,
            triangles,
            resolution=args.remesh_resolution,
            projection_iterations=args.projection_iterations,
            feature_edge_target_length=args.feature_edge_target_length,
            feature_edges=feature_edges,
            feature_edge_angle=args.feature_edge_angle,
            feature_edge_match_tolerance=args.feature_edge_match_tolerance,
            feature_edge_max_splits=args.feature_edge_max_splits,
            sdf_iterations=args.sdf_optimize_iterations,
            sdf_smoothing_step=args.sdf_smoothing_step,
            sdf_projection_steps=args.sdf_projection_steps,
            quality_iterations=args.feature_quality_iterations,
            quality_step=args.feature_quality_step,
            flip_passes=args.feature_flip_passes,
            sdf_mode=args.sdf_mode,
        )
    elif args.sdf_optimize:
        verts, faces = pamo.sdf_optimize(
            points,
            triangles,
            resolution=args.remesh_resolution,
            iterations=args.sdf_optimize_iterations,
            smoothing_step=args.sdf_smoothing_step,
            projection_steps=args.sdf_projection_steps,
            feature_angle=args.sdf_feature_angle,
            sdf_mode=args.sdf_mode,
        )
    elif args.original_constrained_remesh:
        verts, faces = pamo.original_constrained_remesh(
            points,
            triangles,
            resolution=args.remesh_resolution,
            projection_distance=args.constraint_projection_distance,
            feature_snap_distance=args.constraint_feature_distance,
            coplanar_angle_tolerance=args.coplanar_angle_tolerance,
            coplanar_distance_ratio=args.coplanar_distance_ratio,
            sdf_mode=args.sdf_mode,
            allow_open_surface=args.constraint_allow_open_surface,
            max_edge_length=args.constraint_max_edge_length,
            feature_angle=args.constraint_feature_angle,
            feature_target_edge_length=(
                args.constraint_feature_target_edge_length
            ),
            max_splits=args.constraint_max_splits,
            coplanar_flip_passes=args.constraint_flip_passes,
            coplanar_flip_minimum_valence=(
                args.constraint_flip_minimum_valence
            ),
            coplanar_flip_maximum_candidate_quality=(
                args.constraint_flip_maximum_candidate_quality
            ),
            planar_fan_minimum_valence=(
                args.constraint_planar_fan_minimum_valence
            ),
            planar_annulus_minimum_faces=(
                args.constraint_planar_annulus_minimum_faces
            ),
            cylinder_minimum_faces=args.constraint_cylinder_minimum_faces,
            cylinder_radius_tolerance=(
                args.constraint_cylinder_radius_tolerance
            ),
            cylinder_target_edge_ratio=(
                args.constraint_cylinder_target_edge_ratio
            ),
            partial_cylinder_minimum_faces=(
                args.constraint_partial_cylinder_minimum_faces
            ),
            partial_cylinder_radius_tolerance=(
                args.constraint_partial_cylinder_radius_tolerance
            ),
            partial_cylinder_normal_tolerance=(
                args.constraint_partial_cylinder_normal_tolerance
            ),
            partial_cylinder_minimum_angle=(
                args.constraint_partial_cylinder_minimum_angle
            ),
            rounded_fillet_minimum_faces=(
                args.constraint_rounded_fillet_minimum_faces
            ),
            rounded_fillet_minimum_curvature=(
                args.constraint_rounded_fillet_minimum_curvature
            ),
            extrusion_region_minimum_faces=(
                args.constraint_extrusion_region_minimum_faces
            ),
            extrusion_maximum_input_quality=(
                args.constraint_extrusion_maximum_input_quality
            ),
            planar_region_minimum_faces=(
                args.constraint_planar_region_minimum_faces
            ),
            planar_target_edge_length=(
                args.constraint_planar_target_edge_length
            ),
            planar_minimum_angle_degrees=(
                args.constraint_planar_minimum_angle
            ),
            planar_largest_opposed_pair_only=(
                args.constraint_planar_largest_opposed_pair_only
            ),
            quality_iterations=args.constraint_quality_iterations,
            quality_step=args.constraint_quality_step,
            quality_flip_passes=args.constraint_quality_flip_passes,
        )
    else:
        verts, faces = pamo.run(
            points,
            triangles,
            min_verts=args.min_vertex,
            ratio=args.ratio,
            sdf_mode=args.sdf_mode,
        )
    end = time.time()
    print("Total time: ", end-start)
    
    output_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    print("# of output verts : {}".format(len(output_mesh.vertices)))
    print("# of output faces : {}".format(len(output_mesh.faces)))
    output_mesh.export(args.output)

    # # test PaSP
    # pasp = PaSP()
    # start = time.time()
    # verts, faces = pasp.run(torch.from_numpy(input_mesh.vertices).float().cuda(), torch.from_numpy(input_mesh.faces).int().cuda(), iter=10000, threshold=0.001)
    # end = time.time()
    # print("PaSP time: ", end-start)
    
    # verts = verts.cpu().numpy()
    # faces = faces.cpu().numpy()
    # output_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # output_mesh.export('./crab_pasp.obj')


if __name__ == '__main__':
    main()
