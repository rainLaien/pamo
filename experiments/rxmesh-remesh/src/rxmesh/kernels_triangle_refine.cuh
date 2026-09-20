#pragma once

#include "device_math.cuh"
#include "rxmesh/query.h"

namespace cad_adaptive::gpu {

enum TriangleRefinePattern : int {
  kRefineNone = 0,
  kRefineOneEdge = 1,
  kRefineTwoEdges = 2,
  kRefineThreeEdges = 3
};

template <uint32_t blockThreads>
__global__ void classify_triangle_refine_kernel(
    rxmesh::Context context,
    rxmesh::VertexAttribute<float> coords,
    rxmesh::VertexAttribute<float> sizes,
    int* patternCounts) {
  using namespace rxmesh;
  auto block = cooperative_groups::this_thread_block();
  ShmemAllocator shmem;
  Query<blockThreads> query(context);
  query.prologue<Op::FV>(block, shmem);
  block.sync();

  for_each_face(query.get_patch_info(), [&](FaceHandle f) {
    const VertexIterator iter =
        query.template get_iterator<VertexIterator>(f.local_id());
    if (iter.size() != 3) return;
    const VertexHandle v[3] = {iter[0], iter[1], iter[2]};
    if (!v[0].is_valid() || !v[1].is_valid() || !v[2].is_valid()) return;

    int longEdges = 0;
    for (int e = 0; e < 3; ++e) {
      const VertexHandle a = v[e], b = v[(e + 1) % 3];
      const float h = 0.5f * (sizes(a) + sizes(b));
      if (!(h > 0.0f)) continue;
      const float dx = coords(a, 0) - coords(b, 0);
      const float dy = coords(a, 1) - coords(b, 1);
      const float dz = coords(a, 2) - coords(b, 2);
      const float limit = (4.0f / 3.0f) * h;
      if (length2(dx, dy, dz) > limit * limit) ++longEdges;
    }
    ::atomicAdd(&patternCounts[longEdges], 1);
  });
}

} // namespace cad_adaptive::gpu
