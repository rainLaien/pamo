#pragma once

#include "link_condition.cuh"
#include "device_math.cuh"
#include "rxmesh/cavity_manager.cuh"

template <uint32_t blockThreads>
__global__ void compute_valence_kernel(rxmesh::Context context,
                                       rxmesh::VertexAttribute<uint8_t> valence) {
  using namespace rxmesh;
  auto block = cooperative_groups::this_thread_block();
  ShmemAllocator shmem;
  Query<blockThreads> query(context);
  query.compute_vertex_valence(block, shmem);
  block.sync();
  for_each_vertex(query.get_patch_info(),
                  [&](VertexHandle vh) { valence(vh) = query.vertex_valence(vh); });
}

template <uint32_t blockThreads>
__global__ void edge_flip_kernel(rxmesh::Context context, rxmesh::VertexAttribute<float> coords,
                                 const rxmesh::VertexAttribute<uint8_t> valence,
                                 rxmesh::VertexAttribute<float> sizes,
                                 rxmesh::VertexAttribute<int> constraint,
                                 rxmesh::VertexAttribute<int> vPatch,
                                 rxmesh::FaceAttribute<int> fPatch,
                                 rxmesh::EdgeAttribute<int> edgePatch,
                                 rxmesh::EdgeAttribute<cad_adaptive::gpu::EdgeStatus> edgeStatus,
                                 rxmesh::VertexAttribute<bool> vBoundary,
                                 rxmesh::VertexAttribute<int> vDirty,
                                 rxmesh::VertexAttribute<int> vTouched, int *accepted,
                                 int *candidates) {
  using namespace rxmesh;
  using cad_adaptive::gpu::Added;
  using cad_adaptive::gpu::isSeamConstraint;
  using cad_adaptive::gpu::Skip;
  using cad_adaptive::gpu::Unseen;
  auto block = cooperative_groups::this_thread_block();
  ShmemAllocator shmem;
  // The source edge's patch attribute is read during fill-in. Preserve the
  // cavity so another fill-in cannot reuse that deleted edge slot first.
  CavityManager<blockThreads, CavityOp::E> cavity(block, context, shmem, true, false);
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
    if (edgePatch(eh) < 0) return;
    const VertexIterator iter = query.template get_iterator<VertexIterator>(eh.local_id());
    if (iter.size() != 4 || !iter[1].is_valid() || !iter[3].is_valid()) return;
    const VertexHandle a = iter[0], b = iter[2], c = iter[1], d = iter[3];
    if (!vDirty(a) && !vDirty(b)) return;
    // Raw STL feature vertices define a protected crease neighborhood.  Do not
    // let a flip replace connectivity adjacent to a classified crease/corner.
    if (constraint(a) >= cad_adaptive::gpu::kFeatureEdge ||
        constraint(b) >= cad_adaptive::gpu::kFeatureEdge ||
        constraint(c) >= cad_adaptive::gpu::kFeatureEdge ||
        constraint(d) >= cad_adaptive::gpu::kFeatureEdge) return;
    // Flipping an interior edge does not move its fixed boundary vertices.
    // Reject non-convex or folded diamonds geometrically instead.
    using namespace cad_adaptive::gpu;
    const auto pa=point3(coords,a),pb=point3(coords,b),pc=point3(coords,c),pd=point3(coords,d);
    const auto n0=normal3(pa,pb,pc),n1=normal3(pa,pd,pb);
    const auto nn0=normal3(pc,pa,pd),nn1=normal3(pc,pd,pb);
    if (dot3(n0,nn0)<=0 || dot3(n0,nn1)<=0 || dot3(n1,nn0)<=0 || dot3(n1,nn1)<=0) return;
    if (a == b || a == c || a == d || b == c || b == d || c == d) return;
    if (isSeamConstraint(constraint(a)) && isSeamConstraint(constraint(b))) return;
    const int va = valence(a), vb = valence(b), vc = valence(c), vd = valence(d);
    const int pre = (va - 6) * (va - 6) + (vb - 6) * (vb - 6) + (vc - 6) * (vc - 6) +
                    (vd - 6) * (vd - 6);
    const int post = (va - 7) * (va - 7) + (vb - 7) * (vb - 7) + (vc - 5) * (vc - 5) +
                     (vd - 5) * (vd - 5);
    const float q0 = cad_adaptive::gpu::triQuality(
        coords(a, 0), coords(a, 1), coords(a, 2), coords(b, 0), coords(b, 1), coords(b, 2),
        coords(c, 0), coords(c, 1), coords(c, 2));
    const float q1 = cad_adaptive::gpu::triQuality(
        coords(a, 0), coords(a, 1), coords(a, 2), coords(d, 0), coords(d, 1), coords(d, 2),
        coords(b, 0), coords(b, 1), coords(b, 2));
    const float qn0 = cad_adaptive::gpu::triQuality(
        coords(c, 0), coords(c, 1), coords(c, 2), coords(a, 0), coords(a, 1), coords(a, 2),
        coords(d, 0), coords(d, 1), coords(d, 2));
    const float qn1 = cad_adaptive::gpu::triQuality(
        coords(c, 0), coords(c, 1), coords(c, 2), coords(d, 0), coords(d, 1), coords(d, 2),
        coords(b, 0), coords(b, 1), coords(b, 2));
    const float oldMin = fminf(q0, q1);
    const float newMin = fminf(qn0, qn1);
    // Never trade a thin triangle for an even thinner one just to improve
    // valence.  This is particularly important for raw STL input, where the
    // source has no analytic patch projection to repair a bad flip later.
    if (newMin + 1e-7f < oldMin || newMin < 1e-5f) return;
    const float eBefore = float(pre) + (1.f - q0) + (1.f - q1);
    const float eAfter = float(post) + (1.f - qn0) + (1.f - qn1);
    if (eAfter < eBefore - 1e-6f) {
      if (candidates) ::atomicAdd(candidates, 1);
      edgeMask.set(eh.local_id(), true);
    }
  });
  block.sync();
  link_condition(block, cavity.patch_info(), query, edgeMask, v0Mask, v1Mask, 0, 2, true);
  block.sync();
  for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
    if (edgeMask(eh.local_id())) cavity.create(eh);
    else edgeStatus(eh) = Skip;
  });
  block.sync();
  shmem.dealloc(shmem.get_allocated_size_bytes() - before);

  if (cavity.prologue(block, shmem, coords, sizes, constraint, vPatch, fPatch, edgePatch, edgeStatus,
                      vBoundary, valence, vDirty, vTouched)) {
    edgeMask.reset(block);
    block.sync();
    cavity.for_each_cavity(block, [&](uint16_t c, uint16_t n) {
      if (n != 4) return;
      const int patch = edgePatch(cavity.template get_creator<EdgeHandle>(c));
      DEdgeHandle diag =
          cavity.add_edge(cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));
      if (!diag.is_valid()) return;
      edgeMask.set(diag.local_id(), true);
      edgePatch(diag.get_edge_handle()) = patch;
      auto f0 = cavity.add_face(cavity.get_cavity_edge(c, 0), diag, cavity.get_cavity_edge(c, 3));
      auto f1 = cavity.add_face(cavity.get_cavity_edge(c, 1), cavity.get_cavity_edge(c, 2),
                                diag.get_flip_dedge());
      if (f0.is_valid()) fPatch(f0) = patch;
      if (f1.is_valid()) fPatch(f1) = patch;
      vTouched(cavity.get_cavity_vertex(c, 0)) = 1;
      vTouched(cavity.get_cavity_vertex(c, 1)) = 1;
      vTouched(cavity.get_cavity_vertex(c, 2)) = 1;
      vTouched(cavity.get_cavity_vertex(c, 3)) = 1;
      if (accepted) ::atomicAdd(accepted, 1);
    });
  }
  cavity.epilogue(block);
  if (cavity.is_successful()) {
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
      if (edgeMask(eh.local_id())) edgeStatus(eh) = Added;
    });
  }
}
