#pragma once

#include "device_math.cuh"
#include "rxmesh/query.h"

template <uint32_t blockThreads>
__global__ void tag_edge_patch_kernel(rxmesh::Context context,
                                      rxmesh::EdgeAttribute<int> edgePatch,
                                      rxmesh::FaceAttribute<int> fPatch) {
  using namespace rxmesh;
  auto block = cooperative_groups::this_thread_block();
  Query<blockThreads> query(context);
  ShmemAllocator shmem;
  auto tag = [&](const EdgeHandle &e, const FaceIterator &fs) {
    if (fs.size() == 2 && fs[0].is_valid() && fs[1].is_valid()) {
      const int a = fPatch(fs[0]), b = fPatch(fs[1]);
      edgePatch(e) = (a == b) ? a : cad_adaptive::gpu::kEdgePatchBoundary;
    } else {
      edgePatch(e) = cad_adaptive::gpu::kEdgeMeshBoundary;
    }
  };
  query.dispatch<Op::EF>(block, shmem, tag);
}

template <uint32_t blockThreads>
__global__ void promote_seam_verts_kernel(rxmesh::Context context,
                                          rxmesh::EdgeAttribute<int> edgePatch,
                                          rxmesh::VertexAttribute<int> constraint) {
  using namespace rxmesh;
  auto block = cooperative_groups::this_thread_block();
  Query<blockThreads> query(context);
  ShmemAllocator shmem;
  auto promote = [&](const EdgeHandle &e, const VertexIterator &vs) {
    if (edgePatch(e) != cad_adaptive::gpu::kEdgePatchBoundary) return;
    if (vs.size() < 2 || !vs[0].is_valid() || !vs[1].is_valid()) return;
    if (constraint(vs[0]) < cad_adaptive::gpu::kPatchBoundary)
      constraint(vs[0]) = cad_adaptive::gpu::kPatchBoundary;
    if (constraint(vs[1]) < cad_adaptive::gpu::kPatchBoundary)
      constraint(vs[1]) = cad_adaptive::gpu::kPatchBoundary;
  };
  query.dispatch<Op::EV>(block, shmem, promote);
}

template <uint32_t blockThreads>
__global__ void expand_dirty_kernel(rxmesh::Context context,
                                    const rxmesh::VertexAttribute<int> src,
                                    rxmesh::VertexAttribute<int> dst) {
  using namespace rxmesh;
  auto block = cooperative_groups::this_thread_block();
  Query<blockThreads> query(context);
  ShmemAllocator shmem;
  auto expand = [&](const VertexHandle &v, const VertexIterator &iter) {
    int d = src(v);
    if (!d) {
      for (uint32_t i = 0; i < iter.size(); ++i) {
        if (iter[i].is_valid() && src(iter[i])) {
          d = 1;
          break;
        }
      }
    }
    dst(v) = d;
  };
  query.dispatch<Op::VV>(block, shmem, expand, true);
}

template <uint32_t blockThreads>
__global__ void mark_status_from_dirty_kernel(rxmesh::Context context,
                                              const rxmesh::VertexAttribute<int> vDirty,
                                              rxmesh::EdgeAttribute<cad_adaptive::gpu::EdgeStatus> status) {
  using namespace rxmesh;
  auto block = cooperative_groups::this_thread_block();
  Query<blockThreads> query(context);
  ShmemAllocator shmem;
  auto mark = [&](const EdgeHandle &e, const VertexIterator &vs) {
    if (vs.size() < 2 || !vs[0].is_valid() || !vs[1].is_valid()) {
      status(e) = cad_adaptive::gpu::Skip;
      return;
    }
    status(e) = (vDirty(vs[0]) || vDirty(vs[1])) ? cad_adaptive::gpu::Unseen
                                                 : cad_adaptive::gpu::Skip;
  };
  query.dispatch<Op::EV>(block, shmem, mark);
}
