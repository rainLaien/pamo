#pragma once

#include "rxmesh/bitmask.cuh"
#include "rxmesh/query.h"

// Copied from RXMesh apps/Remesh/link_condition.cuh (device check only).
template <uint32_t blockThreads>
__inline__ __device__ void link_condition(cooperative_groups::thread_block &block,
                                          const rxmesh::PatchInfo &patch_info,
                                          rxmesh::Query<blockThreads> &ev_query,
                                          rxmesh::Bitmask &edge_mask, rxmesh::Bitmask &v0_mask,
                                          rxmesh::Bitmask &v1_mask, const int v0_index_in_iter,
                                          const int v1_index_in_iter,
                                          const bool check_flip_diagonal = false) {
  using namespace rxmesh;
  __shared__ int s_num_shared_one_ring;
  __shared__ int s_existing_diagonal;
  for (uint16_t e = 0; e < edge_mask.size(); ++e) {
    if (edge_mask(e)) {
      const VertexIterator iter = ev_query.template get_iterator<VertexIterator>(e);
      const uint16_t v0 = iter.local(v0_index_in_iter);
      const uint16_t v1 = iter.local(v1_index_in_iter);
      if (threadIdx.x == 0) {
        s_num_shared_one_ring = 0;
        s_existing_diagonal = 0;
      }
      v0_mask.reset(block);
      v1_mask.reset(block);
      block.sync();
      for_each_edge(
          patch_info,
          [&](EdgeHandle eh) {
            if (eh.local_id() == e && eh.patch_id() == patch_info.patch_id) return;
            const VertexIterator v_iter =
                ev_query.template get_iterator<VertexIterator>(eh.local_id());
            const uint16_t vv0 = v_iter.local(v0_index_in_iter);
            const uint16_t vv1 = v_iter.local(v1_index_in_iter);
            // Flipping (iter[0], iter[2]) creates (iter[1], iter[3]).
            // The collapse-style common-neighbor test below does NOT exclude
            // an existing opposite diagonal. Include ribbon edges as well as
            // owned edges, otherwise patch-boundary flips can duplicate it.
            if (check_flip_diagonal) {
              const uint16_t c = iter.local(1), d = iter.local(3);
              if ((vv0 == c && vv1 == d) || (vv0 == d && vv1 == c))
                ::atomicExch(&s_existing_diagonal, 1);
            }
            if (vv0 == v0) v0_mask.set(vv1, true);
            if (vv0 == v1) v1_mask.set(vv1, true);
            if (vv1 == v0) v0_mask.set(vv0, true);
            if (vv1 == v1) v1_mask.set(vv0, true);
          },
          true);
      block.sync();
      for (int v = threadIdx.x; v < v0_mask.size(); v += blockThreads) {
        if (v0_mask(v) && v1_mask(v)) ::atomicAdd(&s_num_shared_one_ring, 1);
      }
      block.sync();
      if (s_num_shared_one_ring > 2 || s_existing_diagonal) edge_mask.reset(e, true);
      block.sync();
    }
  }
}
