#include "cad_adaptive/GeometryProjector.h"
#include "cad_adaptive/RemeshField.h"
#include "cad_adaptive/RemeshMetrics.h"
#include "cad_adaptive/RxMeshBackend.h"

#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/macros.h"

#include "kernels_collapse.cuh"
#include "kernels_flip.cuh"
#include "kernels_patch.cuh"
#include "kernels_smooth.cuh"
#include "kernels_split.cuh"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace cad_adaptive {
namespace {

struct RawFeatureStats {
  int SharpEdges = 0;
  int FeatureVertices = 0;
  int Corners = 0;
};

uint64_t rawEdgeKey(uint32_t a, uint32_t b) {
  if (a > b) std::swap(a, b);
  return (uint64_t(a) << 32) | uint64_t(b);
}

RawFeatureStats classifyRawSharpFeatures(SemanticMesh &mesh, float angleDegrees) {
  RawFeatureStats stats;
  if (!(angleDegrees > 0.0f && angleDegrees < 180.0f) || mesh.faceCount() == 0) return stats;
  mesh.rebuildTopology();
  const float cosThreshold = std::cos(angleDegrees * 0.01745329251994329577f);
  std::vector<int> degree(size_t(mesh.vertexCount()), 0);
  for (auto &e : mesh.edges) {
    if (e.face0 < 0 || e.face1 < 0) continue;
    const auto f0 = mesh.face(e.face0), f1 = mesh.face(e.face1);
    const Vec3 n0 = triangleNormal(mesh.position(f0[0]), mesh.position(f0[1]), mesh.position(f0[2]));
    const Vec3 n1 = triangleNormal(mesh.position(f1[0]), mesh.position(f1[1]), mesh.position(f1[2]));
    if (dot(n0, n1) > cosThreshold) continue;
    e.flags = uint8_t(e.flags | EdgeSharp | EdgeProtected);
    ++degree[e.v0]; ++degree[e.v1]; ++stats.SharpEdges;
  }
  for (int v = 0; v < mesh.vertexCount(); ++v) {
    if (degree[v] <= 0) continue;
    if (degree[v] != 2) { mesh.vertexConstraint[v] = uint8_t(VertexConstraint::Corner); ++stats.Corners; }
    else { mesh.vertexConstraint[v] = uint8_t(VertexConstraint::FeatureEdge); ++stats.FeatureVertices; }
  }
  return stats;
}

using rxmesh::DEVICE;
using rxmesh::HandleError;
using rxmesh::HOST;
using rxmesh::LOCATION_ALL;

void ensureCuda() {
  static bool once = false;
  if (!once) {
    rxmesh::rx_init(0);
    once = true;
  }
}

void meshToFv(const SemanticMesh &mesh, std::vector<std::vector<uint32_t>> &fv,
              std::vector<std::vector<float>> &verts, std::vector<int> &constraint,
              std::vector<float> &sizes, std::vector<int> &vPatch, std::vector<int> &fPatch) {
  verts.resize(mesh.vertexCount());
  constraint.resize(mesh.vertexCount());
  sizes.resize(mesh.vertexCount());
  vPatch.resize(mesh.vertexCount());
  for (int v = 0; v < mesh.vertexCount(); ++v) {
    verts[v] = {mesh.px[v], mesh.py[v], mesh.pz[v]};
    constraint[v] = int(mesh.vertexConstraint[v]);
    sizes[v] = v < int(mesh.targetLength.size()) ? mesh.targetLength[v] : 0;
    vPatch[v] = int(mesh.vertexPatchId[v]);
  }
  fv.clear();
  fPatch.clear();
  for (int f = 0; f < mesh.faceCount(); ++f) {
    if (!mesh.faceAlive[f]) continue;
    fv.push_back({mesh.i0[f], mesh.i1[f], mesh.i2[f]});
    fPatch.push_back(int(mesh.facePatchId[f]));
  }
}

void exportMesh(rxmesh::RXMeshDynamic &rx, rxmesh::VertexAttribute<float> *coords,
                rxmesh::VertexAttribute<int> *constraint, rxmesh::VertexAttribute<float> *sizes,
                rxmesh::VertexAttribute<int> *vPatch, rxmesh::FaceAttribute<int> *fPatch,
                const SemanticMesh &source, SemanticMesh &out) {
  rx.update_host();
  coords->move(rxmesh::DEVICE, rxmesh::HOST);
  constraint->move(rxmesh::DEVICE, rxmesh::HOST);
  sizes->move(rxmesh::DEVICE, rxmesh::HOST);
  vPatch->move(rxmesh::DEVICE, rxmesh::HOST);
  fPatch->move(rxmesh::DEVICE, rxmesh::HOST);
  const int nv = int(rx.get_num_vertices());
  const int nf = int(rx.get_num_faces());
  const std::vector<PatchRecord> patches = source.patches;
  out.clear();
  out.patches = patches;
  out.resizeVertices(nv);

  // RXMesh linear_id() is a dense export ID only after update_host(). Verify
  // that contract instead of silently corrupting SemanticMesh on dynamic meshes.
  std::vector<uint8_t> vertexSeen(size_t(nv), 0);
  int exportedVertices = 0;
  rx.for_each_vertex(
      rxmesh::HOST,
      [&](rxmesh::VertexHandle v) {
        const int id = int(rx.linear_id(v));
        if (id < 0 || id >= nv)
          throw std::runtime_error("RXMesh export vertex linear_id out of range: " +
                                   std::to_string(id) + " / " + std::to_string(nv));
        if (vertexSeen[size_t(id)])
          throw std::runtime_error("RXMesh export duplicate vertex linear_id: " +
                                   std::to_string(id));
        vertexSeen[size_t(id)] = 1;
        ++exportedVertices;
        out.setPosition(id, {(*coords)(v, 0), (*coords)(v, 1), (*coords)(v, 2)});
        out.vertexConstraint[id] = uint8_t((*constraint)(v));
        out.targetLength[id] = (*sizes)(v);
        out.vertexPatchId[id] = uint32_t((*vPatch)(v));
      },
      nullptr, false);
  if (exportedVertices != nv)
    throw std::runtime_error("RXMesh export vertex count mismatch: " +
                             std::to_string(exportedVertices) + " / " + std::to_string(nv));

  std::vector<uint32_t> raw(3 * nf);
  rx.create_face_list(raw.data(), false);
  std::vector<int> facePatch(nf, 0);
  std::vector<uint8_t> faceSeen(size_t(nf), 0);
  int exportedFaces = 0;
  rx.for_each_face(
      rxmesh::HOST,
      [&](rxmesh::FaceHandle f) {
        const int id = int(rx.linear_id(f));
        if (id < 0 || id >= nf)
          throw std::runtime_error("RXMesh export face linear_id out of range: " +
                                   std::to_string(id) + " / " + std::to_string(nf));
        if (faceSeen[size_t(id)])
          throw std::runtime_error("RXMesh export duplicate face linear_id: " +
                                   std::to_string(id));
        faceSeen[size_t(id)] = 1;
        ++exportedFaces;
        facePatch[size_t(id)] = (*fPatch)(f);
      }, nullptr, false);
  if (exportedFaces != nf)
    throw std::runtime_error("RXMesh export face count mismatch: " +
                             std::to_string(exportedFaces) + " / " + std::to_string(nf));

  int degenerateFaces = 0;
  for (int f = 0; f < nf; ++f) {
    const uint32_t a = raw[3 * f], b = raw[3 * f + 1], c = raw[3 * f + 2];
    if (a >= uint32_t(nv) || b >= uint32_t(nv) || c >= uint32_t(nv))
      throw std::runtime_error("RXMesh export face vertex index out of range at face " +
                               std::to_string(f));
    if (a == b || b == c || c == a) {
      ++degenerateFaces;
      continue;
    }
    const uint32_t patch = uint32_t(facePatch[f]);
    PatchType type = PatchType::Unknown;
    if (patch < out.patches.size()) type = out.patches[patch].type;
    out.addFace(int(a), int(b), int(c), patch, type);
  }
  if (degenerateFaces)
    std::cerr << "[cad_adaptive rxmesh] export skipped degenerate_faces="
              << degenerateFaces << '\n';
  out.rebuildTopology();
}

template <uint32_t Threads>
void sliceAll(rxmesh::RXMeshDynamic &rx, rxmesh::VertexAttribute<float> *coords,
              rxmesh::VertexAttribute<float> *sizes, rxmesh::VertexAttribute<int> *constraint,
              rxmesh::VertexAttribute<int> *vPatch, rxmesh::FaceAttribute<int> *fPatch,
              rxmesh::EdgeAttribute<int> *edgePatch, rxmesh::EdgeAttribute<gpu::EdgeStatus> *status,
              rxmesh::VertexAttribute<bool> *boundary, rxmesh::VertexAttribute<float> *scratch,
              rxmesh::VertexAttribute<uint8_t> *valence, rxmesh::VertexAttribute<int> *vDirty,
              rxmesh::VertexAttribute<int> *vTouched) {
  rx.cleanup();
  CUDA_ERROR(cudaDeviceSynchronize());
  if (2 * rx.get_num_patches(true) > rx.get_max_num_patches()) {
    throw std::runtime_error("RXMesh patch pool exhausted current=" +
        std::to_string(rx.get_num_patches(true)) + " max=" +
        std::to_string(rx.get_max_num_patches()));
  }
  rx.slice_patches(*coords, *sizes, *constraint, *vPatch, *fPatch, *edgePatch, *status, *boundary,
                   *scratch, *valence, *vDirty, *vTouched);
  rx.cleanup();
  CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace

bool rxmeshAvailable() { return true; }

bool RxMeshBackend::roundTrip(const SemanticMesh &in, SemanticMesh &out, std::vector<int> *valence,
                              std::string *error) {
  try {
    ensureCuda();
    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>> verts;
    std::vector<int> constraint, vPatch, fPatch;
    std::vector<float> sizes;
    meshToFv(in, fv, verts, constraint, sizes, vPatch, fPatch);
    if (fv.empty()) throw std::runtime_error("empty mesh");
    const float reserve = std::max(3.0f, 64.0f / std::max(1.0f, float(fv.size()) / 256.0f));
    rxmesh::RXMeshDynamic rx(fv, "", 256, 3.5f, reserve);
    auto coords = rx.add_vertex_attribute<float>(verts, "coords");
    auto cAttr = rx.add_vertex_attribute<int>(constraint, "constraint");
    auto sAttr = rx.add_vertex_attribute<float>(sizes, "target_length");
    auto vp = rx.add_vertex_attribute<int>(vPatch, "vertex_patch");
    std::vector<std::vector<int>> facePatch(fPatch.size());
    for (size_t i = 0; i < fPatch.size(); ++i) facePatch[i] = {fPatch[i]};
    auto fp = rx.add_face_attribute<int>(facePatch, "face_patch");
    if (!rx.validate()) throw std::runtime_error("RXMesh input topology invalid");
    if (valence) {
      auto val = rx.add_vertex_attribute<int>("valence", 1);
      val->reset(0, rxmesh::LOCATION_ALL);
      auto v = *val;
      rx.for_each<rxmesh::Op::VV, 256>([=] __device__(rxmesh::VertexHandle h,
                                                     const rxmesh::VertexIterator &iter) mutable {
        v(h) = int(iter.size());
      });
      CUDA_ERROR(cudaDeviceSynchronize());
      val->move(rxmesh::DEVICE, rxmesh::HOST);
      valence->assign(int(rx.get_num_vertices()), 0);
      rx.for_each_vertex(
          rxmesh::HOST,
          [&](rxmesh::VertexHandle h) { (*valence)[int(rx.linear_id(h))] = (*val)(h); }, nullptr,
          false);
    }
    exportMesh(rx, coords.get(), cAttr.get(), sAttr.get(), vp.get(), fp.get(), in, out);
    return out.validate(error);
  } catch (const std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

bool RxMeshBackend::remesh(SemanticMesh &mesh, const RemeshConfig &config, RemeshReport &report) {
  report = {};
  const auto started = std::chrono::steady_clock::now();
  try {
    ensureCuda();
    GeometryProjector projector;
    mesh.rebuildTopology();
    const bool rawUnknownSinglePatch = mesh.patches.size() == 1 &&
        mesh.patches[0].type == PatchType::Unknown;
    RawFeatureStats rawFeatures;
    if (rawUnknownSinglePatch) {
      rawFeatures = classifyRawSharpFeatures(mesh, config.featureAngleDegrees);
      std::cout << "raw_feature_classification angle_degrees=" << config.featureAngleDegrees
                << " sharp_edges=" << rawFeatures.SharpEdges
                << " feature_vertices=" << rawFeatures.FeatureVertices
                << " corners=" << rawFeatures.Corners << '\n';
    }
    int boundIn = 0;
    for (const auto &e : mesh.edges)
      if (e.flags & EdgeMeshBoundary) ++boundIn;
    projector.build(mesh);
    RemeshField::compute(mesh, config, projector);
    const bool hasReferenceOnlyPatch = std::any_of(
        mesh.patches.begin(), mesh.patches.end(), [](const PatchRecord &p) {
          return p.type != PatchType::Plane && p.type != PatchType::Cylinder;
        });
    const bool enableGpuSmooth = config.enableSmooth && !hasReferenceOnlyPatch;
    if (config.enableSmooth && hasReferenceOnlyPatch)
      std::cerr << "[cad_adaptive rxmesh] smoothing disabled: Unknown/Freeform patches "
                   "require reference-mesh projection (GPU BVH not wired yet)\n";
    std::unordered_set<uint64_t> rawSharpEdges;
    if (rawUnknownSinglePatch && rawFeatures.SharpEdges > 0) {
      rawSharpEdges.reserve(size_t(rawFeatures.SharpEdges) * 2u);
      for (const auto &e : mesh.edges)
        if (e.flags & EdgeSharp) rawSharpEdges.insert(rawEdgeKey(e.v0, e.v1));
    }
    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>> verts;
    std::vector<int> constraint, vPatch, fPatch;
    std::vector<float> sizes;
    meshToFv(mesh, fv, verts, constraint, sizes, vPatch, fPatch);
    if (!config.adaptive && config.constantLength > 0)
      for (float &h : sizes) h = config.constantLength;
    float reserve = std::max(3.0f, 64.0f / std::max(1.0f, float(fv.size()) / 256.0f));
    if (rawUnknownSinglePatch) {
      // A very coarse STL can require thousands of first-order edge segments
      // before it reaches the target length. RXMesh dynamic slicing consumes
      // patch IDs while that refinement proceeds, so size the metadata pool from
      // the actual coarse-edge demand rather than a model-specific constant.
      uint64_t firstOrderExtraSegments = 0;
      for (const auto &e : mesh.edges) {
        const float h = 0.5f * (mesh.targetLength[e.v0] + mesh.targetLength[e.v1]);
        if (!(h > 0.0f)) continue;
        const float limit = config.splitRatio * h;
        const float len = distance(mesh.position(int(e.v0)), mesh.position(int(e.v1)));
        if (len > limit) firstOrderExtraSegments += uint64_t(std::ceil(len / limit)) - 1u;
      }
      reserve = float(std::max<uint64_t>(1024u,
          std::min<uint64_t>(8192u, std::max<uint64_t>(1u, firstOrderExtraSegments) * 2u)));
      std::cout << "raw_patch_pool first_order_extra_segments=" << firstOrderExtraSegments
                << " patch_alloc_factor=" << reserve << '\n';
    }
    // Raw STL can start as one extremely coarse patch and generate many slices
    // while long surface edges are refined. Reserve patch IDs up front; this is
    // metadata capacity, not permission to cross classified sharp features.
    // Keep small inputs in one initial patch. Splitting a tiny closed mesh
    // into uneven patches gives it undersized local capacities and forces
    // repeated migration/slicing before its first remeshing pass completes.
    const uint32_t patchSize = fv.size() <= 512 ? 512 : 256;
    rxmesh::RXMeshDynamic rx(fv, "", patchSize, 3.5f, reserve);
    auto coords = rx.add_vertex_attribute<float>(verts, "coords");
    auto scratch = rx.add_vertex_attribute<float>("scratch", 3);
    auto sAttr = rx.add_vertex_attribute<float>(sizes, "target_length");
    auto cAttr = rx.add_vertex_attribute<int>(constraint, "constraint");
    auto vp = rx.add_vertex_attribute<int>(vPatch, "vertex_patch");
    std::vector<std::vector<int>> facePatch(fPatch.size());
    for (size_t i = 0; i < fPatch.size(); ++i) facePatch[i] = {fPatch[i]};
    auto fp = rx.add_face_attribute<int>(facePatch, "face_patch");
    auto status = rx.add_edge_attribute<gpu::EdgeStatus>("edge_status", 1);
    auto edgePatch = rx.add_edge_attribute<int>("edge_patch", 1);
    edgePatch->reset(0, rxmesh::LOCATION_ALL);
    if (!rawSharpEdges.empty()) {
      std::vector<uint32_t> edgeList(size_t(rx.get_num_edges()) * 2u);
      rx.create_edge_list(edgeList.data(), false);
      rx.for_each_edge(rxmesh::HOST, [&](rxmesh::EdgeHandle e) {
        const uint32_t row = rx.linear_id(e);
        const uint32_t a = edgeList[2u * row], b = edgeList[2u * row + 1u];
        if (rawSharpEdges.count(rawEdgeKey(a, b))) (*edgePatch)(e) = gpu::kEdgeSharp;
      }, nullptr, false);
      edgePatch->move(rxmesh::HOST, rxmesh::DEVICE);
    }
    auto boundary = rx.add_vertex_attribute<bool>("boundary", 1);
    auto valence = rx.add_vertex_attribute<uint8_t>("valence", 1);
    auto vDirty = rx.add_vertex_attribute<int>("v_dirty", 1);
    auto vTouched = rx.add_vertex_attribute<int>("v_touched", 1);
    vDirty->reset(1, rxmesh::LOCATION_ALL);
    vTouched->reset(0, rxmesh::LOCATION_ALL);
    rx.get_boundary_vertices(*boundary);
    {
      auto b = *boundary;
      auto c = *cAttr;
      rx.for_each_vertex(rxmesh::DEVICE, [=] __device__(rxmesh::VertexHandle v) mutable {
        if (c(v) >= gpu::kCorner) b(v) = true;
      });
    }
    auto syncCuda = [](const char *where) {
      const cudaError_t err = cudaDeviceSynchronize();
      if (err != cudaSuccess)
        std::cerr << "[cad_adaptive rxmesh] CUDA sync failed at " << where << '\n';
      CUDA_ERROR(err);
    };
    auto tagPatchInterfaces = [&] {
      // for_each<Op> uses prepare_launch_box (static). After cavity/slice the
      // patches grow and that under-allocates shared memory (illegal access).
      constexpr uint32_t tagThreads = 256;
      rxmesh::LaunchBox<tagThreads> lb;
      rx.update_launch_box({rxmesh::Op::EF}, lb, (void *)tag_edge_patch_kernel<tagThreads>, false);
      tag_edge_patch_kernel<tagThreads>
          <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(rx.get_context(), *edgePatch, *fp);
      syncCuda("tag_edge_patch");
      rx.update_launch_box({rxmesh::Op::EV}, lb, (void *)promote_seam_verts_kernel<tagThreads>,
                           false);
      promote_seam_verts_kernel<tagThreads>
          <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(rx.get_context(), *edgePatch, *cAttr);
      syncCuda("promote_seam_verts");
    };
    tagPatchInterfaces();
    gpu::PatchGpu *dPatches = nullptr;
    const int nPatches = int(mesh.patches.size());
    CUDA_ERROR(cudaMallocManaged(&dPatches, sizeof(gpu::PatchGpu) * std::max(1, nPatches)));
    for (int i = 0; i < nPatches; ++i) {
      const PatchRecord &pr = mesh.patches[i];
      const Vec3 axis = normalize(pr.axis);
      dPatches[i].type = int(pr.type);
      dPatches[i].ox = pr.origin.x;
      dPatches[i].oy = pr.origin.y;
      dPatches[i].oz = pr.origin.z;
      dPatches[i].ax = axis.x;
      dPatches[i].ay = axis.y;
      dPatches[i].az = axis.z;
      dPatches[i].radius = pr.radius;
    }
    syncCuda("upload_patch_table");
    auto projectSurfaceVerts = [&] {
      auto c = *coords;
      auto k = *cAttr;
      auto patchOf = *vp;
      auto touched = *vTouched;
      const gpu::PatchGpu *table = dPatches;
      const int nP = nPatches;
      rx.for_each_vertex(rxmesh::DEVICE, [=] __device__(rxmesh::VertexHandle v) mutable {
        if (!touched(v) || gpu::isImmobile(k(v))) return;
        const int pid = patchOf(v);
        if (pid < 0 || pid >= nP) return;
        float x = c(v, 0), y = c(v, 1), z = c(v, 2);
        if (gpu::projectPatch(table[pid], x, y, z)) {
          c(v, 0) = x;
          c(v, 1) = y;
          c(v, 2) = z;
        }
      });
      syncCuda("project_surface");
    };
    if (!rx.validate()) throw std::runtime_error("RXMesh input topology invalid");

    std::vector<Vec3> lockedBefore;
    for (int v = 0; v < mesh.vertexCount(); ++v)
      if (VertexConstraint(mesh.vertexConstraint[v]) == VertexConstraint::Locked)
        lockedBefore.push_back(mesh.position(v));

    int *accepted = nullptr, *candidates = nullptr;
    CUDA_ERROR(cudaMallocManaged(&accepted, sizeof(int)));
    CUDA_ERROR(cudaMallocManaged(&candidates, sizeof(int)));
    constexpr uint32_t threads = 256;
    const float splitRatio = config.splitRatio;
    const float collapseRatio = config.collapseRatio;
    // Split needs one temporary edge mask. Collapse and flip both allocate
    // an edge mask plus two vertex masks for link_condition(); account for all
    // three or ShmemAllocator writes past the dynamic shared-memory region.
    auto edgeMaskShmem = [&](uint32_t, uint32_t e, uint32_t) {
      return rxmesh::detail::mask_num_bytes(e) + rxmesh::ShmemAllocator::default_alignment;
    };
    auto topologyMaskShmem = [&](uint32_t v, uint32_t e, uint32_t) {
      return rxmesh::detail::mask_num_bytes(e) +
             2u * rxmesh::detail::mask_num_bytes(v) +
             3u * rxmesh::ShmemAllocator::default_alignment;
    };
    auto expandDirty2Ring = [&] {
      constexpr uint32_t t = 256;
      rxmesh::LaunchBox<t> lb;
      rx.update_launch_box({rxmesh::Op::VV}, lb, (void *)expand_dirty_kernel<t>, false, true);
      expand_dirty_kernel<t>
          <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(rx.get_context(), *vTouched, *vDirty);
      syncCuda("expand_dirty_1");
      expand_dirty_kernel<t>
          <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(rx.get_context(), *vDirty, *vTouched);
      syncCuda("expand_dirty_2");
      auto d = *vDirty;
      auto src = *vTouched;
      rx.for_each_vertex(rxmesh::DEVICE, [=] __device__(rxmesh::VertexHandle v) mutable { d(v) = src(v); });
      syncCuda("copy_dirty");
      // The next operator clears touched. Do not launch an asynchronous reset
      // here: Windows cannot read managed counters while a GPU kernel runs.
    };
    auto markStatusFromDirty = [&] {
      constexpr uint32_t t = 256;
      rxmesh::LaunchBox<t> lb;
      rx.update_launch_box({rxmesh::Op::EV}, lb, (void *)mark_status_from_dirty_kernel<t>, false);
      mark_status_from_dirty_kernel<t>
          <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(rx.get_context(), *vDirty, *status);
      syncCuda("mark_status_dirty");
    };

    auto runSplit = [&] {
      *accepted = 0;
      *candidates = 0;
      vTouched->reset(0, DEVICE);
      markStatusFromDirty();
      rx.reset_scheduler();
      int launches = 0;
      const int launchBudget = rawUnknownSinglePatch ? 64 : 2048;
      while (!rx.is_queue_empty()) {
        if (++launches > launchBudget) {
          if (rawUnknownSinglePatch) break; // continue refinement in the next macro iteration
          throw std::runtime_error("RXMesh scheduler stalled");
        }
        rxmesh::LaunchBox<threads> lb;
        // Split's Query<EVDiamond> is used only to select cavity creators.
        // CavityManager already reserves the dynamic workspace; adding a
        // separate query workspace here double-counts it and can push the
        // launch into a bad shared-memory configuration after patch growth.
        rx.update_launch_box({}, lb, (void *)edge_split_kernel<threads>, true,
                             false, false, false, edgeMaskShmem);
        edge_split_kernel<threads><<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *sAttr, *cAttr, *vp, *fp, *edgePatch, *status, *boundary,
            *vDirty, *vTouched, splitRatio, accepted, candidates);
        syncCuda("edge_split");
        sliceAll<threads>(rx, coords.get(), sAttr.get(), cAttr.get(), vp.get(), fp.get(),
                          edgePatch.get(), status.get(), boundary.get(), scratch.get(),
                          valence.get(), vDirty.get(), vTouched.get());
        projectSurfaceVerts();
      }
      report.splitCandidates += *candidates;
      if (*accepted > 0) expandDirty2Ring();
      return *accepted;
    };
    auto runCollapse = [&] {
      *accepted = 0;
      *candidates = 0;
      vTouched->reset(0, DEVICE);
      markStatusFromDirty();
      rx.reset_scheduler();
      int launches = 0;
      const int launchBudget = rawUnknownSinglePatch ? 64 : 2048;
      while (!rx.is_queue_empty()) {
        if (++launches > launchBudget) {
          if (rawUnknownSinglePatch) break;
          throw std::runtime_error("RXMesh collapse scheduler stalled");
        }
        rxmesh::LaunchBox<threads> lb;
        rx.update_launch_box({rxmesh::Op::EVDiamond}, lb, (void *)edge_collapse_kernel<threads>,
                             true, false, false, false, topologyMaskShmem);
        edge_collapse_kernel<threads><<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *sAttr, *cAttr, *vp, *fp, *edgePatch, *status, *boundary,
            *vDirty, *vTouched, collapseRatio, splitRatio, accepted, candidates);
        syncCuda("edge_collapse");
        sliceAll<threads>(rx, coords.get(), sAttr.get(), cAttr.get(), vp.get(), fp.get(),
                          edgePatch.get(), status.get(), boundary.get(), scratch.get(),
                          valence.get(), vDirty.get(), vTouched.get());
        projectSurfaceVerts();
      }
      report.collapseCandidates += *candidates;
      if (*accepted > 0) expandDirty2Ring();
      return *accepted;
    };
    auto runFlip = [&] {
      rxmesh::LaunchBox<threads> valenceBox;
      rx.update_launch_box({rxmesh::Op::VV}, valenceBox, (void *)compute_valence_kernel<threads>,
                           false, true, false, false);
      compute_valence_kernel<threads>
          <<<valenceBox.blocks, valenceBox.num_threads, valenceBox.smem_bytes_dyn>>>(
              rx.get_context(), *valence);
      syncCuda("compute_valence");
      *accepted = 0;
      *candidates = 0;
      vTouched->reset(0, DEVICE);
      markStatusFromDirty();
      rx.reset_scheduler();
      int launches = 0;
      const int launchBudget = rawUnknownSinglePatch ? 64 : 2048;
      while (!rx.is_queue_empty()) {
        if (++launches > launchBudget) {
          if (rawUnknownSinglePatch) break;
          throw std::runtime_error("RXMesh flip scheduler stalled");
        }
        rxmesh::LaunchBox<threads> lb;
        rx.update_launch_box({rxmesh::Op::EVDiamond}, lb, (void *)edge_flip_kernel<threads>, true,
                             false, false, false, topologyMaskShmem);
        edge_flip_kernel<threads><<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *valence, *sAttr, *cAttr, *vp, *fp, *edgePatch, *status,
            *boundary, *vDirty, *vTouched, accepted, candidates);
        syncCuda("edge_flip");
        sliceAll<threads>(rx, coords.get(), sAttr.get(), cAttr.get(), vp.get(), fp.get(),
                          edgePatch.get(), status.get(), boundary.get(), scratch.get(),
                          valence.get(), vDirty.get(), vTouched.get());
      }
      report.flipCandidates += *candidates;
      if (*accepted > 0) expandDirty2Ring();
      return *accepted;
    };

    report.secondsSetup =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    for (int it = 0; it < config.maxIterations; ++it) {
      auto mark = std::chrono::steady_clock::now();
      if (config.enableSplit) {
        report.splits += runSplit();
        rx.update_host();
        if (!rx.validate())
          throw std::runtime_error("RXMesh topology invalid immediately after split");
      }
      report.secondsSplit +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - mark).count();
      mark = std::chrono::steady_clock::now();
      if (config.enableCollapse) {
        report.collapses += runCollapse();
        rx.update_host();
        if (!rx.validate())
          throw std::runtime_error("RXMesh topology invalid immediately after collapse");
      }
      report.secondsCollapse +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - mark).count();
      mark = std::chrono::steady_clock::now();
      if (config.enableFlip) {
        report.flips += runFlip();
        rx.update_host();
        if (!rx.validate())
          throw std::runtime_error("RXMesh topology invalid immediately after flip");
      }
      report.secondsFlip +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - mark).count();
      tagPatchInterfaces();
      mark = std::chrono::steady_clock::now();
      if (enableGpuSmooth) {
        const float lambda = config.smoothLambda;
        rxmesh::LaunchBox<384> lb;
        rx.update_launch_box({rxmesh::Op::VV}, lb, (void *)vertex_smooth_kernel<384>, false, true,
                             false, false);
        for (int s = 0; s < 3; ++s) {
          *accepted = 0;
          vTouched->reset(0, DEVICE);
          vertex_smooth_kernel<384><<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
              rx.get_context(), *coords, *scratch, *cAttr, *boundary,
              *vDirty, *vTouched, lambda, accepted);
          syncCuda("vertex_smooth");
          std::swap(coords, scratch);
          projectSurfaceVerts();
          report.smoothMoves += *accepted;
          if (*accepted > 0) expandDirty2Ring();
        }
      }
      report.secondsSmooth +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - mark).count();
      rx.update_host();
      if (!rx.validate()) throw std::runtime_error("RXMesh topology invalid after iteration");
    }
    CUDA_ERROR(cudaFree(accepted));
    CUDA_ERROR(cudaFree(candidates));
    CUDA_ERROR(cudaFree(dPatches));
    exportMesh(rx, coords.get(), cAttr.get(), sAttr.get(), vp.get(), fp.get(), mesh, mesh);
    std::string err;
    report.topologyValid = mesh.validate(&err);
    if (!report.topologyValid) std::cerr << "[cad_adaptive rxmesh] export: " << err << '\n';
    int boundOut = 0;
    for (const auto &e : mesh.edges)
      if (e.flags & EdgeMeshBoundary) ++boundOut;
    if (boundOut < boundIn) report.missingBoundaryEdges = boundIn - boundOut;
    fillMeshMetrics(mesh, config, report);
    report.movedLockedVertices = 0;
    for (int v = 0; v < mesh.vertexCount(); ++v) {
      if (VertexConstraint(mesh.vertexConstraint[v]) != VertexConstraint::Locked) continue;
      float best = 1e30f;
      const Vec3 p = mesh.position(v);
      for (const Vec3 &q : lockedBefore) best = std::min(best, length2(p - q));
      if (best > 1e-12f) ++report.movedLockedVertices;
    }
    report.constraintsHeld = report.movedLockedVertices == 0;
    report.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return report.topologyValid;
  } catch (const std::exception &e) {
    report.topologyValid = false;
    report.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cerr << "[cad_adaptive rxmesh] " << e.what() << '\n';
    return false;
  }
}

} // namespace cad_adaptive
