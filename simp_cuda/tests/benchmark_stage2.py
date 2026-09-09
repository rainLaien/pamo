"""Stage2-only baseline, without SDF generation or safe projection.

Run each native build in a separate process. Meshes are deterministic icospheres:
--size small (20,480 faces), medium (327,680), large (1,310,720).
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', choices=('small', 'medium', 'large'), default='small')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--profiling', action='store_true')
    parser.add_argument('--threshold', type=float, default=.001)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    spec = importlib.util.spec_from_file_location('_C', args.extension.resolve())
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    mesh = trimesh.creation.icosphere(subdivisions={'small': 5, 'medium': 7, 'large': 8}[args.size])
    vertices = torch.tensor(mesh.vertices, device='cuda', dtype=torch.float32)
    faces = torch.tensor(mesh.faces, device='cuda', dtype=torch.int32)
    undo = torch.empty(0, device='cuda', dtype=torch.int32)
    solver = native.CUDSP_Free()
    if args.profiling:
        solver.set_profiling(True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    epochs = []
    for iteration in range(args.epochs):
        old_faces = len(faces)
        vertices, faces, occupied, mapping, undo = solver.forward(
            vertices, faces, undo, len(undo), 2., args.threshold, False, iteration == 0)
        report = dict(solver.profiling_report()) if args.profiling else {}
        begin = end = None
        if args.profiling:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        vertices = vertices[occupied.view(-1).bool()]
        faces = faces[faces[:, 0] >= 0]
        for axis in range(3):
            faces[:, axis] = mapping[faces[:, axis].long()].view(-1)
        if args.profiling:
            end.record()
            end.synchronize()
            report['PythonCompactMs'] = begin.elapsed_time(end)
        report.update(iteration=iteration, before_faces=old_faces, after_faces=len(faces))
        epochs.append(report)
        if len(faces) <= 10:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'extension': str(args.extension.resolve()), 'size': args.size,
              'profiling': args.profiling, 'stage2_wall_seconds': elapsed,
              'initial_faces': len(mesh.faces), 'final_faces': len(faces),
              'epochs': epochs, 'torch_peak_memory_bytes': torch.cuda.max_memory_allocated(),
              'memory_note': 'PyTorch allocations only; native cudaMalloc is not included.',
              'timing_note': 'CUDA timeline phase intervals include launch gaps; profiling overhead is enabled only on request.'}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez(args.output.with_suffix('.npz'), vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
    print(json.dumps({'seconds': elapsed, 'faces': len(faces), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
