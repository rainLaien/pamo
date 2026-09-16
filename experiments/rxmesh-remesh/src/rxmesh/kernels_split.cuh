#pragma once

#include "rxmesh/cavity_manager.cuh"
#include "device_math.cuh"

template <uint32_t blockThreads>
__global__ void edge_split_kernel(rxmesh::Context context, rxmesh::VertexAttribute<float> coords,
                                  rxmesh::VertexAttribute<float> sizes,
                                  rxmesh::VertexAttribute<int> constraint,
                                  rxmesh::VertexAttribute<int> vPatch,
                                  rxmesh::FaceAttribute<int> fPatch,
                                  rxmesh::EdgeAttribute<int> edgePatch,
                                  rxmesh::EdgeAttribute<cad_adaptive::gpu::EdgeStatus> edgeStatus,
                                  rxmesh::VertexAttribute<bool> vBoundary,
                                  rxmesh::VertexAttribute<int> vDirty,
                                  rxmesh::VertexAttribute<int> vTouched, const float splitRatio,
                                  int *accepted, int *candidates) {
  using namespace rxmesh;
  using cad_adaptive::gpu::Added;
  using cad_adaptive::gpu::kPatchBoundary;
  using cad_adaptive::gpu::Skip;
  using cad_adaptive::gpu::Unseen;
  auto block = cooperative_groups::this_thread_block();
  ShmemAllocator shmem;
  CavityManager<blockThreads, CavityOp::E> cavity(block, context, shmem, true);
  if (cavity.patch_id() == INVALID32) return;
  Bitmask updated(cavity.patch_info().edges_capacity, shmem);
  updated.reset(block);
  const uint32_t before = shmem.get_allocated_size_bytes();

  auto should_split = [&](const EdgeHandle &eh, const VertexIterator &iter) {
    if (edgeStatus(eh) != Unseen) return;
    if (iter.size() != 4) return;
    const VertexHandle va = iter[0], vb = iter[2], vc = iter[1], vd = iter[3];
    if (!va.is_valid() || !vb.is_valid() || !vc.is_valid() || !vd.is_valid()) {
      edgeStatus(eh) = Skip;
      return;
    }
    // Match RXMesh's own remeshing safety rule: do not open a split cavity
    // when any vertex in the EVDiamond is a mesh/feature boundary vertex.
    // Real CAD partitions contain many locked/seam vertices; allowing a cavity
    // to straddle them can invalidate the local patch allocation and lead to
    // illegal accesses inside CavityManager.
    if (vBoundary(va) || vBoundary(vb) || vBoundary(vc) || vBoundary(vd) ||
        cad_adaptive::gpu::isImmobile(constraint(va)) ||
        cad_adaptive::gpu::isImmobile(constraint(vb)) ||
        cad_adaptive::gpu::isImmobile(constraint(vc)) ||
        cad_adaptive::gpu::isImmobile(constraint(vd))) {
      edgeStatus(eh) = Skip;
      return;
    }
    // An interior diagonal may join two locked feature vertices. Splitting
    // it leaves both endpoints and every feature edge untouched.
    if ((!vDirty(va) && !vDirty(vb)) || edgePatch(eh) < 0) {
      edgeStatus(eh) = Skip;
      return;
    }
    if (va == vb || va == vc || va == vd || vb == vc || vb == vd || vc == vd) {
      edgeStatus(eh) = Skip;
      return;
    }
    const float ax = coords(va, 0), ay = coords(va, 1), az = coords(va, 2);
    const float bx = coords(vb, 0), by = coords(vb, 1), bz = coords(vb, 2);
    const float h = 0.5f * (sizes(va) + sizes(vb));
    const float limit = splitRatio * h;
    const float len2 = cad_adaptive::gpu::length2(ax - bx, ay - by, az - bz);
    if (h > 0.f && len2 > limit * limit) {
      if (candidates) ::atomicAdd(candidates, 1);
      cavity.create(eh);
    } else
      edgeStatus(eh) = Skip;
  };

  Query<blockThreads> query(context, cavity.patch_id());
  query.dispatch<Op::EVDiamond>(block, shmem, should_split);
  block.sync();
  shmem.dealloc(shmem.get_allocated_size_bytes() - before);

  if (cavity.prologue(block, shmem, coords, sizes, constraint, vPatch, fPatch, edgePatch, edgeStatus,
                      vBoundary, vDirty, vTouched)) {
    cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
      if (size != 4) {
        cavity.recover(cavity.template get_creator<EdgeHandle>(c));
        return;
      }
      const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);
      const int origEp = edgePatch(src);
      const VertexHandle v0 = cavity.get_cavity_vertex(c, 0);
      const VertexHandle v1 = cavity.get_cavity_vertex(c, 2);
      const VertexHandle nv = cavity.add_vertex();
      if (!nv.is_valid()) return;
      coords(nv, 0) = 0.5f * (coords(v0, 0) + coords(v1, 0));
      coords(nv, 1) = 0.5f * (coords(v0, 1) + coords(v1, 1));
      coords(nv, 2) = 0.5f * (coords(v0, 2) + coords(v1, 2));
      sizes(nv) = 0.5f * (sizes(v0) + sizes(v1));
      int cNew = 1; // Surface: an interior midpoint is not a feature vertex.
      // Interior splits stay on origEp (same-patch id). Do not take patch from
      // opposite diamond verts: a seam vertex is tagged patch 0 and would
      // relabel a patch-1 triangle that only touches the seam.
      if (origEp < 0) cNew = kPatchBoundary;
      constraint(nv) = cNew;
      vPatch(nv) = origEp >= 0 ? origEp : (vPatch(v0) < vPatch(v1) ? vPatch(v0) : vPatch(v1));
      vBoundary(nv) = false;
      vDirty(nv) = 1;
      vTouched(nv) = 1;
      vTouched(v0) = 1;
      vTouched(v1) = 1;
      DEdgeHandle e0 = cavity.add_edge(nv, cavity.get_cavity_vertex(c, 0));
      const DEdgeHandle first = e0;
      if (!e0.is_valid()) return;
      edgePatch(e0.get_edge_handle()) = origEp;
      updated.set(e0.local_id(), true);
      for (uint16_t i = 0; i < size; ++i) {
        const DEdgeHandle e = cavity.get_cavity_edge(c, i);
        const DEdgeHandle e1 =
            (i == size - 1) ? first.get_flip_dedge()
                            : cavity.add_edge(cavity.get_cavity_vertex(c, i + 1), nv);
        if (!e1.is_valid()) break;
        if (i != size - 1) edgePatch(e1.get_edge_handle()) = origEp;
        updated.set(e1.local_id(), true);
        const FaceHandle f = cavity.add_face(e0, e, e1);
        if (!f.is_valid()) break;
        fPatch(f) = origEp >= 0 ? origEp : vPatch(nv);
        e0 = e1.get_flip_dedge();
      }
      if (accepted) ::atomicAdd(accepted, 1);
    });
  }
  cavity.epilogue(block);
  if (cavity.is_successful()) {
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
      if (updated(eh.local_id())) edgeStatus(eh) = Added;
    });
  }
}
