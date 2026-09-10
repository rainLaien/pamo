#pragma once

namespace CadMesh {

// Compiled together with the analytic fitting kernels by NVRTC. Keeping this
// kernel in source form lets the MinGW executable use CUDA without nvcc or an
// MSVC host compiler.
inline constexpr const char *RemeshKernelSource = R"cuda(
extern "C" __global__ void cadmesh_classify_long_edges(
    const double *vertices, const int *edges, int edgeCount,
    double maximumSquared, unsigned char *selected, double *midpoints) {
  const int id = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (id >= edgeCount) return;
  const int a = edges[2 * id], b = edges[2 * id + 1];
  double squared = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const double first = vertices[3 * a + axis];
    const double second = vertices[3 * b + axis];
    const double delta = second - first;
    squared += delta * delta;
    midpoints[3 * id + axis] = .5 * (first + second);
  }
  selected[id] = squared > maximumSquared * (1.0 + 2.0e-6);
}
)cuda";

} // namespace CadMesh
