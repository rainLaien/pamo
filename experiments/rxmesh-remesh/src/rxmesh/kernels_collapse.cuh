#pragma once

#include "link_condition.cuh"
#include "device_math.cuh"
#include "rxmesh/cavity_manager.cuh"

template <uint32_t blockThreads>
__global__ void edge_collapse_kernel(rxmesh::Context context,
                                     rxmesh::VertexAttribute<float> coords,
                                     rxmesh::VertexAttribute<float> sizes,
                                     rxmesh::VertexAttribute<int> constraint,
                                     rxmesh::VertexAttribute<int> vPatch,
                                     rxmesh::FaceAttribute<int> fPatch,
                                     rxmesh::EdgeAttribute<int> edgePatch,
                                     rxmesh::EdgeAttribute<cad_adaptive::gpu::EdgeStatus> edgeStatus,
                                     rxmesh::VertexAttribute<bool> vBoundary,
                                     rxmesh::VertexAttribute<int> vDirty,
                                     rxmesh::VertexAttribute<int> vTouched,
                                     const float collapseRatio, const float splitRatio,
                                     int *accepted, int *candidates) {
  using namespace rxmesh;
  using cad_adaptive::gpu::Added;
  using cad_adaptive::gpu::isLocked;
  using cad_adaptive::gpu::isSeamConstraint;
  using cad_adaptive::gpu::kEdgeMeshBoundary;
  using cad_adaptive::gpu::kEdgePatchBoundary;
  using cad_adaptive::gpu::Skip;
  using cad_adaptive::gpu::Unseen;
  auto block = cooperative_groups::this_thread_block();
  ShmemAllocator shmem;
  CavityManager<blockThreads, CavityOp::EV> cavity(block, context, shmem, true);
  if (cavity.patch_id() == INVALID32) return;
  Bitmask edgeMask(cavity.patch_info().edges_capacity, shmem);
  edgeMask.reset(block);
  Bitmask v0Mask(cavity.patch_info().num_vertices[0], shmem);
  Bitmask v1Mask(cavity.patch_info().num_vertices[0], shmem);
  const uint32_t before = shmem.get_allocated_size_bytes();
  Query<blockThreads> query(context, cavity.patch_id());
  query.prologue<Op::EVDiamond>(block, shmem);
  block.sync();

  for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
    if (edgeStatus(eh) != Unseen) return;
    const VertexIterator iter = query.template get_iterator<VertexIterator>(eh.local_id());
    if (iter.size() != 4) return;
    const VertexHandle v0 = iter[0], v1 = iter[2], v2 = iter[1], v3 = iter[3];
    if (!v0.is_valid() || !v1.is_valid() || !v2.is_valid() || !v3.is_valid()) return;
    if (!vDirty(v0) && !vDirty(v1)) return;
    if (edgePatch(eh) < 0) return;
    if (vBoundary(v0) || vBoundary(v1)) return;
    if (cad_adaptive::gpu::isImmobile(constraint(v0)) ||
        cad_adaptive::gpu::isImmobile(constraint(v1)))
      return;
    if (isSeamConstraint(constraint(v0)) && isSeamConstraint(constraint(v1))) return;
    if (vPatch(v0) != vPatch(v1) && !isSeamConstraint(constraint(v0)) &&
        !isSeamConstraint(constraint(v1)))
      return;
    const float h = 0.5f * (sizes(v0) + sizes(v1));
    const float dx = coords(v0, 0) - coords(v1, 0);
    const float dy = coords(v0, 1) - coords(v1, 1);
    const float dz = coords(v0, 2) - coords(v1, 2);
    const float low = collapseRatio * h;
    if (h > 0.f && cad_adaptive::gpu::length2(dx, dy, dz) < low * low) {
      if (candidates) ::atomicAdd(candidates, 1);
      edgeMask.set(eh.local_id(), true);
    }
  });
  block.sync();
  link_condition(block, cavity.patch_info(), query, edgeMask, v0Mask, v1Mask, 0, 2);
  block.sync();
  for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
    if (edgeMask(eh.local_id())) cavity.create(eh);
    else edgeStatus(eh) = Skip;
  });
  block.sync();
  shmem.dealloc(shmem.get_allocated_size_bytes() - before);

  if (cavity.prologue(block, shmem, coords, sizes, constraint, vPatch, fPatch, edgePatch, edgeStatus,
                      vBoundary, vDirty, vTouched)) {
    edgeMask.reset(block);
    block.sync();
    cavity.for_each_cavity(block, [&](uint16_t c, uint16_t n) {
      const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);
      VertexHandle va, vb;
      cavity.get_vertices(src, va, vb);
      const bool keepA = isLocked(constraint(va)) || isSeamConstraint(constraint(va));
      const bool keepB = isLocked(constraint(vb)) || isSeamConstraint(constraint(vb));
      float mx, my, mz;
      int keepC = constraint(va), keepP = vPatch(va);
      const int srcPatch = edgePatch(src);
      if (keepA && !keepB) {
        mx = coords(va, 0);
        my = coords(va, 1);
        mz = coords(va, 2);
      } else if (keepB && !keepA) {
        mx = coords(vb, 0);
        my = coords(vb, 1);
        mz = coords(vb, 2);
        keepC = constraint(vb);
        keepP = vPatch(vb);
      } else {
        mx = 0.5f * (coords(va, 0) + coords(vb, 0));
        my = 0.5f * (coords(va, 1) + coords(vb, 1));
        mz = 0.5f * (coords(va, 2) + coords(vb, 2));
      }
      if (srcPatch >= 0) keepP = srcPatch;
      const float h = 0.5f * (sizes(va) + sizes(vb));
      const float high = splitRatio * h;
      bool longEdge = false;
      for (uint16_t i = 0; i < n; ++i) {
        const VertexHandle v = cavity.get_cavity_vertex(c, i);
        const float dx = coords(v, 0) - mx, dy = coords(v, 1) - my, dz = coords(v, 2) - mz;
        if (cad_adaptive::gpu::length2(dx, dy, dz) >= high * high) {
          longEdge = true;
          break;
        }
      }
      if (longEdge || n < 3) {
        cavity.recover(src);
        edgeStatus(src) = Skip;
        return;
      }
      // All new fan triangles must agree with the oriented cavity boundary.
      // Length/link checks alone admit folds on thin CAD triangles.
      using namespace cad_adaptive::gpu;
      float3 reference=make_float3(0,0,0);
      const auto origin=point3(coords,cavity.get_cavity_vertex(c,0));
      for(uint16_t i=0;i<n;++i) {
        auto u=point3(coords,cavity.get_cavity_vertex(c,i));
        auto w=point3(coords,cavity.get_cavity_vertex(c,(i+1)%n));
        auto normal=normal3(origin,u,w);
        reference.x+=normal.x; reference.y+=normal.y; reference.z+=normal.z;
      }
      bool folded=dot3(reference,reference)<=1e-20f;
      for(uint16_t i=0;i<n && !folded;++i) {
        auto u=point3(coords,cavity.get_cavity_vertex(c,i));
        auto w=point3(coords,cavity.get_cavity_vertex(c,(i+1)%n));
        folded=dot3(normal3(make_float3(mx,my,mz),u,w),reference)<=1e-12f*dot3(reference,reference);
      }
      if(folded) { cavity.recover(src); edgeStatus(src)=Skip; return; }
      const VertexHandle nv = cavity.add_vertex();
      if (!nv.is_valid()) return;
      coords(nv, 0) = mx;
      coords(nv, 1) = my;
      coords(nv, 2) = mz;
      sizes(nv) = h;
      constraint(nv) = keepC;
      vPatch(nv) = keepP;
      vBoundary(nv) = vBoundary(va) || vBoundary(vb);
      vDirty(nv) = 1;
      vTouched(nv) = 1;
      vTouched(va) = 1;
      vTouched(vb) = 1;
      DEdgeHandle e0 = cavity.add_edge(nv, cavity.get_cavity_vertex(c, 0));
      const DEdgeHandle first = e0;
      if (!e0.is_valid()) return;
      edgePatch(e0.get_edge_handle()) = keepP;
      edgeMask.set(e0.local_id(), true);
      for (uint16_t i = 0; i < n; ++i) {
        const DEdgeHandle e = cavity.get_cavity_edge(c, i);
        const DEdgeHandle e1 =
            (i == n - 1) ? first.get_flip_dedge()
                         : cavity.add_edge(cavity.get_cavity_vertex(c, i + 1), nv);
        if (!e1.is_valid()) break;
        if (i != n - 1) {
          edgePatch(e1.get_edge_handle()) = keepP;
          edgeMask.set(e1.local_id(), true);
        }
        const FaceHandle f = cavity.add_face(e0, e, e1);
        if (!f.is_valid()) break;
        fPatch(f) = keepP;
        e0 = e1.get_flip_dedge();
      }
      if (accepted) ::atomicAdd(accepted, 1);
    });
  }
  cavity.epilogue(block);
  if (cavity.is_successful()) {
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
      if (edgeMask(eh.local_id()) || cavity.is_recovered(eh)) edgeStatus(eh) = Added;
    });
  }
}
