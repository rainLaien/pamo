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
    if (edgePatch(e) == cad_adaptive::gpu::kEdgeSharp) return;
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
__global__ void tag_sharp_edges_kernel(rxmesh::Context context,
                                       const rxmesh::VertexAttribute<float> coords,
                                       rxmesh::EdgeAttribute<int> edgePatch,
                                       rxmesh::VertexAttribute<int> constraint,
                                       float cosThreshold) {
  using namespace rxmesh;
  auto block = cooperative_groups::this_thread_block();
  Query<blockThreads> query(context);
  ShmemAllocator shmem;
  auto tag = [&](const EdgeHandle &e, const VertexIterator &iter) {
    if (edgePatch(e) < 0 || iter.size() != 4) return;
    const VertexHandle a=iter[0], c=iter[1], b=iter[2], d=iter[3];
    if(!a.is_valid() || !b.is_valid() || !c.is_valid() || !d.is_valid()) return;
    using namespace cad_adaptive::gpu;
    const float3 pa=point3(coords,a), pb=point3(coords,b);
    const float3 pc=point3(coords,c), pd=point3(coords,d);
    const float3 n0=normal3(pa,pb,pc), n1=normal3(pb,pa,pd);
    const float l0=dot3(n0,n0), l1=dot3(n1,n1);
    if(!(l0>1e-20f && l1>1e-20f)) return;
    const float cosine=dot3(n0,n1)*rsqrtf(l0*l1);
    if(cosine > cosThreshold) return;
    edgePatch(e)=kEdgeSharp;
    if(constraint(a)<kFeatureEdge) constraint(a)=kFeatureEdge;
    if(constraint(b)<kFeatureEdge) constraint(b)=kFeatureEdge;
  };
  query.dispatch<Op::EVDiamond>(block,shmem,tag);
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
