"""CLI: remesh an existing cad_mesh PLY/JSON handoff with PaMO."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_mesh.remesh_io import load_partition, write_remesh_result
from cad_mesh.remesh_pipeline import DEFAULT_BATCH_FACE_LIMIT, remesh_partition


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Partition directory or patch_result.ply (requires patch_report.json).")
    parser.add_argument("--output", type=Path, help="New output directory; default: INPUT/pamo_remesh.")
    parser.add_argument("--target-edge-length", type=float, help="Maximum target edge in model units; default 5 percent of bbox diagonal (PaMO original-constrained default).")
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--projection-backend", choices=("cuda", "cpu"), default="cuda",
                        help="Whole-patch projection and validation: CUDA float64 BVH (default) or CPU reference implementation.")
    parser.add_argument("--method", choices=("surface", "legacy"), default="surface",
                        help="surface: whole analytic charts and patch projection (default); legacy: source-triangle refinement.")
    parser.add_argument("--split-passes", type=int, default=128)
    parser.add_argument("--collapse-passes", type=int, default=12)
    parser.add_argument("--flip-passes", type=int, default=8)
    parser.add_argument("--relax-iterations", type=int, default=3)
    parser.add_argument("--max-boundary-splits", type=int, default=1000000)
    parser.add_argument("--batch-face-limit", type=int, default=DEFAULT_BATCH_FACE_LIMIT,
                        help="Maximum source faces per CUDA batch before shared-edge refinement (default 75000; comparison values: 50000 or 100000).")
    parser.add_argument("--max-deviation", type=float, help="Independent geometry deviation in model units; surface default: 0.00025 * input bbox diagonal; legacy default: 0.005 * target.")
    parser.add_argument("--max-normal-deviation-degrees", type=float, default=10.0,
                        help="Maximum allowed output normal deviation in degrees (default 10).")
    parser.add_argument("--trace-surface-validation", action="store_true", default=False,
                        help="Run expensive diagnostic surface checks after every stage (disabled by default; final validation always runs).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-output", action="store_true",
                        help="Also collect diagnostic statistics and export JSON, STL and alternate PLY views; default writes only remesh_result.ply.")
    args = parser.parse_args(argv)
    begin = time.perf_counter()
    try:
        print(f"[remesh] Loading and validating {args.input.resolve()}...", flush=True)
        source = load_partition(args.input)
        load_seconds = time.perf_counter() - begin
        import numpy as np
        target = args.target_edge_length
        if target is None:
            target = float(np.linalg.norm(np.ptp(source.vertices, axis=0))) * .05
        output = args.output or source.source_directory / "pamo_remesh"
        if output.resolve() == source.source_directory.resolve():
            raise ValueError("Remesh output must be a separate directory from the input partition.")
        rebuild_begin = time.perf_counter()
        result = remesh_partition(
            source, target_edge_length=target, sample_count=args.sample_count,
            split_passes=args.split_passes, collapse_passes=args.collapse_passes,
            flip_passes=args.flip_passes, relax_iterations=args.relax_iterations,
            seed=args.seed, maximum_boundary_splits=args.max_boundary_splits,
            maximum_deviation=args.max_deviation,
            maximum_normal_deviation_degrees=args.max_normal_deviation_degrees,
            batch_face_limit=args.batch_face_limit,
            method=args.method,
            projection_backend=args.projection_backend,
            trace_surface_validation=args.trace_surface_validation,
            collect_diagnostics=args.full_output or args.trace_surface_validation,
        )
        rebuild_seconds = time.perf_counter() - rebuild_begin
        export_begin = time.perf_counter()
        paths = write_remesh_result(output, result, source, full_output=args.full_output)
        export_seconds = time.perf_counter() - export_begin
        print(f"[remesh] Complete: {len(source.faces):,} -> {len(result.faces):,} faces; "
              f"{result.stats['input_patch_count']:,} input labels -> {result.stats['output_patch_count']:,} output surface domains.")
        print(f"[remesh] Output: {Path(output).resolve()}")
        print(f"[remesh] Timing: load {load_seconds:.2f}s, rebuild {rebuild_seconds:.2f}s, "
              f"export {export_seconds:.2f}s; total {time.perf_counter() - begin:.2f}s")
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"Remesh failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
