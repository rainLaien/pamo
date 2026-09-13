#include "CadMesh/CadMeshPatchSegmenter.h"
#include "CadMesh/DifferentialGeometryEstimator.h"
#include "CadMesh/HeatPatchRegularizer.h"
#include "CadMesh/ModelFirstPartitioner.h"
#include "CadMesh/SegmentationGuards.h"
#include "CadMesh/SurfaceRoles.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace CadMesh {
namespace {
void ApplyFit(MeshPatch &patch, const SurfaceFitResult &fit) {
  patch.SurfaceType = fit.Type;
  patch.RmsFittingError = fit.Rms;
  patch.MaxFittingError = fit.Max;
  patch.NormalError = fit.Normal;
  patch.Parameters = fit.Parameters;
  patch.Confidence = Clamp01(std::exp(-fit.Score * .2));
}
SurfaceFitResult Fit(const MeshTopology &mesh, const MeshPatch &patch,
                     const SegmentationConfig &config) {
  return SurfaceModelSelector::fitBest(mesh, patch.TriangleIds,
                                       mesh.getResolution(),
                                       config.ModelComplexityPenalty);
}
std::vector<int> UnionTriangles(const MeshPatch &a, const MeshPatch &b) {
  std::vector<int> result = a.TriangleIds;
  result.insert(result.end(), b.TriangleIds.begin(), b.TriangleIds.end());
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}
bool AcceptCombined(const MeshTopology &mesh, const MeshPatch &a,
                    const MeshPatch &b, const SurfaceFitResult &fit,
                    const SegmentationConfig &config) {
  const auto &r = mesh.getResolution();
  double old = std::max(a.RmsFittingError, b.RmsFittingError);
  double allowed = std::max(3.0 * r.FittingTolerance,
                            old * config.MergeErrorFactor + r.FittingTolerance);
  double maxAllowed = std::max(8.0 * r.FittingTolerance,
                               std::max(a.MaxFittingError, b.MaxFittingError) *
                                       config.MergeErrorFactor +
                                   2 * r.FittingTolerance);
  double normalAllowed = std::max(8.0 * r.AngularTolerance,
                                  std::max(a.NormalError, b.NormalError) *
                                          config.MergeErrorFactor +
                                      r.AngularTolerance);
  if (fit.Type == PatchSurfaceType::Unknown ||
      fit.Type == PatchSurfaceType::Freeform || fit.Rms > allowed ||
      fit.Max > maxAllowed || fit.Normal > normalAllowed)
    return false;
  // Refit success alone is insufficient: an averaged normal error can hide
  // a tiny but real face. Validate all original samples against the new model.
  MeshPatch fitted;
  ApplyFit(fitted, fit);
  return IsSurfaceCompatible(
             EvaluateSurfaceCompatibility(mesh, a.TriangleIds, fitted), r) &&
         IsSurfaceCompatible(
             EvaluateSurfaceCompatibility(mesh, b.TriangleIds, fitted), r);
}
std::map<std::pair<int, int>, std::vector<int>>
PatchPairs(const MeshTopology &mesh) {
  std::map<std::pair<int, int>, std::vector<int>> pairs;
  for (int id = 0; id < int(mesh.getEdges().size()); ++id) {
    const auto &e = mesh.getEdges()[id];
    if (e.IsBoundary || e.IsNonManifold || e.Triangle0 < 0 || e.Triangle1 < 0)
      continue;
    int a = mesh.getTriangles()[e.Triangle0].PatchId,
        b = mesh.getTriangles()[e.Triangle1].PatchId;
    if (a < 0 || b < 0 || a == b)
      continue;
    if (a > b)
      std::swap(a, b);
    pairs[{a, b}].push_back(id);
  }
  return pairs;
}
double AverageBoundary(const MeshTopology &mesh,
                       const std::vector<int> &edges) {
  double sum = 0;
  for (int e : edges)
    sum += mesh.getEdges()[e].BoundaryScore;
  return edges.empty() ? 1 : sum / edges.size();
}
bool ProtectedInterface(const MeshTopology &mesh, const std::vector<int> &edges,
                        const SegmentationConfig &config) {
  return std::any_of(edges.begin(), edges.end(), [&](int edgeId) {
    return IsHardSegmentationBoundary(mesh.getEdges()[edgeId], config);
  });
}

bool FitsFixedTarget(const MeshTopology &mesh, const MeshPatch &source,
                     const MeshPatch &target) {
  return IsSurfaceCompatible(
      EvaluateSurfaceCompatibility(mesh, source.TriangleIds, target),
      mesh.getResolution());
}

void ResolveCylinderTessellationSeams(MeshTopology &mesh,
                                      const std::vector<MeshPatch> &patches,
                                      const SegmentationConfig &config) {
  // Raw boundary evidence is a candidate, not yet a confirmed CAD feature.
  // A tiny planar facet may be one ordinary strip of an adjacent cylinder.
  // Certify it against that cylinder's observed error envelope before locking
  // constraints. Never relax an already explicit feature or another boundary.
  int resolved = 0;
  for (const auto &item : PatchPairs(mesh)) {
    const auto &a = patches[item.first.first];
    const auto &b = patches[item.first.second];
    if (!IsCylinderTessellationContinuation(mesh, a, b, item.second, config) &&
        !IsCylinderTessellationContinuation(mesh, b, a, item.second, config))
      continue;
    for (int edgeId : item.second) {
      auto &edge = mesh.getEdges()[edgeId];
      if (edge.BoundaryScore >= config.StrongBoundaryThreshold)
        ++resolved;
      edge.BoundaryScore =
          std::min(edge.BoundaryScore, .5 * config.WeakBoundaryThreshold);
      edge.Evidence.FinalScore = edge.BoundaryScore;
    }
  }
  if (config.Verbose && resolved)
    std::clog << "[CadMesh] certified cylinder tessellation seams: " << resolved
              << " edge(s)\n";
}

bool MayCrossBetween(const MeshTopology &mesh, int source, int target,
                     const SegmentationConfig &config) {
  for (int edgeId : mesh.getTriangles()[source].EdgeIds) {
    const auto &edge = mesh.getEdges()[edgeId];
    if (std::find(edge.IncidentTriangleIds.begin(),
                  edge.IncidentTriangleIds.end(),
                  target) != edge.IncidentTriangleIds.end())
      return !IsHardSegmentationBoundary(edge, config);
  }
  return false;
}
bool SimilarAnalytic(const MeshPatch &a, const MeshPatch &b,
                     const MeshResolutionInfo &r) {
  if (a.SurfaceType != b.SurfaceType)
    return false;
  if (a.SurfaceType == PatchSurfaceType::Plane) {
    auto *pa = std::get_if<PlaneParameters>(&a.Parameters);
    auto *pb = std::get_if<PlaneParameters>(&b.Parameters);
    if (!pa || !pb)
      return false;
    double angle = std::acos(
        std::max(-1.0, std::min(1.0, std::abs(Dot(ToVec(pa->Plane.Normal),
                                                  ToVec(pb->Plane.Normal))))));
    double distance =
        std::abs(Dot(Sub(ToVec(pb->Plane.Origin), ToVec(pa->Plane.Origin)),
                     ToVec(pa->Plane.Normal)));
    return angle < 4 * r.AngularTolerance && distance < 3 * r.FittingTolerance;
  }
  if (a.SurfaceType == PatchSurfaceType::Cylinder) {
    auto *pa = std::get_if<CylinderParameters>(&a.Parameters);
    auto *pb = std::get_if<CylinderParameters>(&b.Parameters);
    if (!pa || !pb)
      return false;
    double align =
        std::abs(Dot(ToVec(pa->Axis.Direction), ToVec(pb->Axis.Direction)));
    Vec3 delta = Sub(ToVec(pb->Axis.Origin), ToVec(pa->Axis.Origin)),
         axis = ToVec(pa->Axis.Direction);
    double axisDistance = Norm(Sub(delta, Mul(axis, Dot(delta, axis))));
    return 1 - align < 8 * r.AngularTolerance * r.AngularTolerance &&
           axisDistance < 4 * r.FittingTolerance &&
           std::abs(pa->Radius - pb->Radius) < 4 * r.FittingTolerance;
  }
  if (a.SurfaceType == PatchSurfaceType::Sphere) {
    auto *pa = std::get_if<SphereParameters>(&a.Parameters);
    auto *pb = std::get_if<SphereParameters>(&b.Parameters);
    return pa && pb &&
           Distance(pa->Center, pb->Center) < 4 * r.FittingTolerance &&
           std::abs(pa->Radius - pb->Radius) < 4 * r.FittingTolerance;
  }
  return false;
}
double PatchModelResidual(const MeshTopology &mesh, const MeshPatch &source,
                          const MeshPatch &target) {
  const double tolerance =
      std::max(mesh.getResolution().FittingTolerance, 1e-30);
  double maximum = 0;
  std::set<int> vertices;
  for (int t : source.TriangleIds)
    for (int v : mesh.getTriangles()[t].VertexIds)
      vertices.insert(v);
  if (target.SurfaceType == PatchSurfaceType::Plane) {
    auto *p = std::get_if<PlaneParameters>(&target.Parameters);
    if (!p)
      return 1e100;
    for (int v : vertices)
      maximum = std::max(maximum,
                         std::abs(Dot(Sub(ToVec(mesh.getVertices()[v].Position),
                                          ToVec(p->Plane.Origin)),
                                      ToVec(p->Plane.Normal))) /
                             tolerance);
    return maximum;
  }
  if (target.SurfaceType == PatchSurfaceType::Cylinder) {
    auto *p = std::get_if<CylinderParameters>(&target.Parameters);
    if (!p)
      return 1e100;
    for (int v : vertices) {
      Vec3 d = Sub(ToVec(mesh.getVertices()[v].Position),
                   ToVec(p->Axis.Origin)),
           axis = ToVec(p->Axis.Direction);
      double radius = Norm(Sub(d, Mul(axis, Dot(d, axis))));
      maximum = std::max(maximum, std::abs(radius - p->Radius) / tolerance);
    }
    return maximum;
  }
  if (target.SurfaceType == PatchSurfaceType::Sphere) {
    auto *p = std::get_if<SphereParameters>(&target.Parameters);
    if (!p)
      return 1e100;
    for (int v : vertices)
      maximum = std::max(
          maximum,
          std::abs(Distance(mesh.getVertices()[v].Position, p->Center) -
                   p->Radius) /
              tolerance);
    return maximum;
  }
  return 0;
}
double TriangleQuality(const MeshTopology &mesh, int triangleId) {
  const auto &triangle = mesh.getTriangles()[triangleId];
  double longestSquared = 0;
  for (int edgeId : triangle.EdgeIds) {
    const auto &edge = mesh.getEdges()[edgeId];
    Vec3 delta = Sub(ToVec(mesh.getVertices()[edge.Vertex1].Position),
                     ToVec(mesh.getVertices()[edge.Vertex0].Position));
    longestSquared = std::max(longestSquared, Dot(delta, delta));
  }
  return triangle.Area / std::max(longestSquared, 1e-30);
}
} // namespace

bool CadMeshPatchSegmenter::segment(const TriangleSoup &soup) {
  using Clock = std::chrono::steady_clock;
  auto stage = Clock::now();
  auto log = [&](const char *name) {
    if (mConfig.Verbose) {
      auto now = Clock::now();
      std::clog << "[CadMesh] " << name << ": "
                << std::chrono::duration<double>(now - stage).count() << " s";
      if (!mPatches.empty())
        std::clog << ", patches=" << mPatches.size();
      std::clog << '\n';
      stage = now;
    }
  };
  mPatches.clear();
  mAdjacency.clear();
  mConstraint = {};
  mComputedDiagnostics = false;
  if (!mMesh.build(soup))
    return false;
  log("topology");
  if (mConfig.EnableModelFirst) {
    mPatches = PartitionBySurfaceModels(mMesh, mConfig);
    log("model-first partition");
  } else {
    DifferentialGeometryEstimator(mMesh).compute(mConfig.CurvatureRingCount);
    log("curvature");
    BoundaryScoreCalculator(mMesh, mConfig).compute();
    mComputedDiagnostics = true;
    log("boundary score");
    GenerateInitialRegions();
    FitPatches();
    log("initial regions");
    ResolveCylinderTessellationSeams(mMesh, mPatches, mConfig);
    // After analytic arbitration, hard features cannot be erased by subsequent
    // label operations, even if faces connect around an alternative route.
    for (auto &edge : mMesh.getEdges())
      if (!edge.IsBoundary && !edge.IsNonManifold &&
          edge.BoundaryScore >= mConfig.StrongBoundaryThreshold)
        edge.IsConstrainedFeature = true;
    GrowSurfaceConsistentRegions();
    log("region growing");
    PatchRefiner::refine(*this);
    log("refinement");
  }
  RebuildConnectedPatches();
  AssignPatchIds();
  std::string error;
  if (!validatePartition(&error))
    throw std::logic_error("Invalid CAD partition: " + error);
  if (mConfig.EnableModelFirst) {
    // A small real crease can be below the initial dihedral barrier. Once
    // both surfaces are certified, their analytic normals resolve that edge
    // without turning noisy per-triangle curvature cues into hard constraints.
    const double tangentCosine =
        std::cos(std::max(2 * mMesh.getResolution().AngularTolerance, .02));
    for (auto &edge : mMesh.getEdges()) {
      if (edge.IncidentTriangleIds.size() != 2 || edge.IsConstrainedFeature)
        continue;
      const int a = mMesh.getTriangles()[edge.IncidentTriangleIds[0]].PatchId;
      const int b = mMesh.getTriangles()[edge.IncidentTriangleIds[1]].PatchId;
      if (a == b)
        continue;
      // Chord midpoints need not lie on the two recovered surfaces. Endpoints
      // retain the fitted intersection, including a small-radius torus fillet.
      for (int vertex : {edge.Vertex0, edge.Vertex1}) {
        const Vec3 point = ToVec(mMesh.getVertices()[vertex].Position);
        Vec3 normalA, normalB;
        double distanceA, distanceB;
        if (SegmentationGuardDetail::SurfaceSample(mPatches[a], point,
                                                   distanceA, normalA) &&
            SegmentationGuardDetail::SurfaceSample(mPatches[b], point,
                                                   distanceB, normalB) &&
            std::abs(Dot(normalA, normalB)) < tangentCosine)
          edge.IsConstrainedFeature = true;
      }
    }
  }
  PatchGraphBuilder::build(*this);
  IdentifySurfaceRoles(mMesh, mPatches, mAdjacency);
  log("patch graph");
  if (mConfig.Verbose) {
    std::size_t tiny = 0, tinyPlanes = 0, tinyFaces = 0;
    double totalArea = 0, tinyArea = 0;
    for (const auto &patch : mPatches) {
      const bool isTiny = patch.TriangleIds.size() <= 2;
      if (isTiny) {
        ++tiny;
        tinyPlanes += patch.SurfaceType == PatchSurfaceType::Plane;
        tinyFaces += patch.TriangleIds.size();
      }
      for (int face : patch.TriangleIds) {
        const double area = mMesh.getTriangles()[face].Area;
        totalArea += area;
        if (isTiny) tinyArea += area;
      }
    }
    std::clog << "[CadMesh] small patches (<=2 input triangles): " << tiny
              << '/' << mPatches.size() << ", planar=" << tinyPlanes
              << ", triangles=" << tinyFaces
              << ", surface_area_percent="
              << (totalArea > 0 ? 100 * tinyArea / totalArea : 0) << '\n';
  }
  return !mPatches.empty();
}

void CadMeshPatchSegmenter::GenerateInitialRegions() {
  auto &triangles = mMesh.getTriangles();
  std::vector<char> seen(triangles.size(), 0);
  for (int seed = 0; seed < int(triangles.size()); ++seed)
    if (!seen[seed]) {
      MeshPatch patch;
      std::queue<int> q;
      q.push(seed);
      seen[seed] = 1;
      while (!q.empty()) {
        int t = q.front();
        q.pop();
        patch.TriangleIds.push_back(t);
        for (int edgeId : triangles[t].EdgeIds) {
          const auto &e = mMesh.getEdges()[edgeId];
          if (e.IsBoundary || e.IsNonManifold ||
              e.BoundaryScore > mConfig.StrongBoundaryThreshold)
            continue;
          for (int n : e.IncidentTriangleIds)
            if (!seen[n]) {
              seen[n] = 1;
              q.push(n);
            }
        }
      }
      mPatches.push_back(std::move(patch));
    }
  AssignPatchIds();
}

void CadMeshPatchSegmenter::FitPatches() {
  for (auto &patch : mPatches)
    ApplyFit(patch, Fit(mMesh, patch, mConfig));
}

void CadMeshPatchSegmenter::GrowSurfaceConsistentRegions() {
  bool changed = true;
  int pass = 0;
  while (changed && pass++ < 32) {
    changed = false;
    AssignPatchIds();
    auto pairs = PatchPairs(mMesh);
    struct Candidate {
      int A, B;
      double Score;
      std::vector<int> Edges;
    };
    std::vector<Candidate> candidates;
    for (auto &item : pairs) {
      double score = AverageBoundary(mMesh, item.second);
      if (score < mConfig.StrongBoundaryThreshold &&
          !ProtectedInterface(mMesh, item.second, mConfig))
        candidates.push_back(
            {item.first.first, item.first.second, score, item.second});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) {
                return a.Score < b.Score;
              });
    for (const auto &candidate : candidates) {
      if (candidate.A >= int(mPatches.size()) ||
          candidate.B >= int(mPatches.size()))
        continue;
      auto triangles =
          UnionTriangles(mPatches[candidate.A], mPatches[candidate.B]);
      MeshPatch combined;
      combined.TriangleIds = std::move(triangles);
      auto fit = Fit(mMesh, combined, mConfig);
      bool weakInterface = candidate.Score < mConfig.WeakBoundaryThreshold;
      bool analytic = SimilarAnalytic(
          mPatches[candidate.A], mPatches[candidate.B], mMesh.getResolution());
      if ((weakInterface || analytic) &&
          AcceptCombined(mMesh, mPatches[candidate.A], mPatches[candidate.B],
                         fit, mConfig)) {
        ApplyFit(combined, fit);
        mPatches[candidate.A] = std::move(combined);
        mPatches.erase(mPatches.begin() + candidate.B);
        changed = true;
        break;
      }
    }
  }
}

void CadMeshPatchSegmenter::AssignPatchIds() {
  auto &triangles = mMesh.getTriangles();
  for (auto &triangle : triangles)
    triangle.PatchId = -1;
  for (int id = 0; id < int(mPatches.size()); ++id) {
    mPatches[id].Id = id;
    if (mPatches[id].TriangleIds.empty())
      throw std::logic_error("Empty CAD patch during ownership assignment");
    for (int t : mPatches[id].TriangleIds) {
      if (t < 0 || t >= int(triangles.size()) || triangles[t].PatchId >= 0)
        throw std::logic_error("Invalid or duplicate CAD triangle ownership");
      triangles[t].PatchId = id;
    }
  }
  for (const auto &triangle : triangles)
    if (triangle.PatchId < 0)
      throw std::logic_error("CAD triangle has no patch owner");
}

void CadMeshPatchSegmenter::RebuildConnectedPatches() {
  AssignPatchIds();
  const auto &triangles = mMesh.getTriangles();
  const auto &edges = mMesh.getEdges();
  std::vector<char> seen(triangles.size(), 0);
  std::vector<MeshPatch> rebuilt;
  for (const auto &original : mPatches) {
    for (int seed : original.TriangleIds) {
      if (seen[seed])
        continue;
      MeshPatch patch = original;
      patch.TriangleIds.clear();
      std::queue<int> queue;
      queue.push(seed);
      seen[seed] = 1;
      while (!queue.empty()) {
        int t = queue.front();
        queue.pop();
        patch.TriangleIds.push_back(t);
        for (int edgeId : triangles[t].EdgeIds) {
          const auto &edge = edges[edgeId];
          if (edge.IsBoundary || edge.IsNonManifold ||
              edge.IsConstrainedFeature)
            continue;
          for (int neighbor : edge.IncidentTriangleIds)
            if (!seen[neighbor] && triangles[neighbor].PatchId == original.Id) {
              seen[neighbor] = 1;
              queue.push(neighbor);
            }
        }
      }
      if (!mConfig.ModelCylindersOnly && !mConfig.ModelConesOnly && patch.TriangleIds.size() != original.TriangleIds.size())
        ApplyFit(patch, Fit(mMesh, patch, mConfig));
      rebuilt.push_back(std::move(patch));
    }
  }
  mPatches = std::move(rebuilt);
  AssignPatchIds();
}

bool CadMeshPatchSegmenter::validatePartition(std::string *error) const {
  if (error)
    error->clear();
  auto fail = [&](const std::string &message) {
    if (error)
      *error = message;
    return false;
  };
  const auto &triangles = mMesh.getTriangles();
  std::vector<int> owner(triangles.size(), -1);
  for (int id = 0; id < int(mPatches.size()); ++id) {
    const auto &patch = mPatches[id];
    if (patch.Id != id || patch.TriangleIds.empty())
      return fail("empty patch or inconsistent patch ID");
    for (int t : patch.TriangleIds) {
      if (t < 0 || t >= int(triangles.size()) || owner[t] >= 0)
        return fail("invalid or duplicate triangle ID");
      owner[t] = id;
      if (triangles[t].PatchId != id)
        return fail("triangle label disagrees with patch membership");
    }
  }
  if (std::find(owner.begin(), owner.end(), -1) != owner.end())
    return fail("triangle missing from partition");
  std::vector<char> seen(triangles.size(), 0);
  for (const auto &patch : mPatches) {
    size_t count = 0;
    std::queue<int> queue;
    queue.push(patch.TriangleIds.front());
    seen[queue.front()] = 1;
    while (!queue.empty()) {
      int t = queue.front();
      queue.pop();
      ++count;
      for (int edgeId : triangles[t].EdgeIds) {
        const auto &edge = mMesh.getEdges()[edgeId];
        if (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature)
          continue;
        for (int neighbor : edge.IncidentTriangleIds)
          if (!seen[neighbor] && owner[neighbor] == patch.Id) {
            seen[neighbor] = 1;
            queue.push(neighbor);
          }
      }
    }
    if (count != patch.TriangleIds.size())
      return fail("patch is disconnected across admissible surface edges");
  }
  return !triangles.empty();
}

void PatchRefiner::refine(CadMeshPatchSegmenter &s) {
  // Recover sizeable CAD planes directly from triangle geometry.  This
  // intentionally ignores tessellation boundary scores, but uses one fixed
  // plane for the whole connected region so gradual cylindrical drift can
  // never be merged by transitivity.
  {
    const auto &r = s.mMesh.getResolution();
    const auto &triangles = s.mMesh.getTriangles();
    double angularLimit =
               std::max(4 * r.AngularTolerance, 3.0 * std::acos(-1.0) / 180.0),
           distanceLimit =
               std::max(6 * r.FittingTolerance, r.BoundingBoxDiagonal * 2e-6);
    std::vector<int> state(triangles.size(), -1),
        visitStamp(triangles.size(), 0), seedOrder(triangles.size());
    std::vector<std::vector<int>> planes;
    int stamp = 0;
    // A skinny triangle is a poor plane seed: a tiny perturbation of one
    // vertex can rotate its normal by many degrees.  Process the most
    // equilateral triangles first so stable CAD facets claim their whole
    // supporting plane before tessellation slivers are considered.
    std::iota(seedOrder.begin(), seedOrder.end(), 0);
    auto quality = [&](int triangleId) {
      const auto &t = triangles[triangleId];
      double longestSquared = 0;
      for (int edgeId : t.EdgeIds) {
        const auto &e = s.mMesh.getEdges()[edgeId];
        Vec3 d = Sub(ToVec(s.mMesh.getVertices()[e.Vertex1].Position),
                     ToVec(s.mMesh.getVertices()[e.Vertex0].Position));
        longestSquared = std::max(longestSquared, Dot(d, d));
      }
      return t.Area / std::max(longestSquared, 1e-30);
    };
    std::sort(seedOrder.begin(), seedOrder.end(),
              [&](int a, int b) { return quality(a) > quality(b); });
    for (int seed : seedOrder)
      if (state[seed] < 0) {
        Vec3 normal = Normalize(ToVec(triangles[seed].Normal)),
             origin = ToVec(triangles[seed].Centroid);
        std::vector<int> region;
        std::queue<int> q;
        q.push(seed);
        visitStamp[seed] = ++stamp;
        while (!q.empty()) {
          int t = q.front();
          q.pop();
          region.push_back(t);
          for (int n : s.mMesh.getTriangleNeighbors(t))
            if (state[n] < 0 && visitStamp[n] != stamp) {
              if (!MayCrossBetween(s.mMesh, t, n, s.mConfig))
                continue;
              bool onPlane = true;
              for (int v : triangles[n].VertexIds)
                if (std::abs(Dot(
                        Sub(ToVec(s.mMesh.getVertices()[v].Position), origin),
                        normal)) > distanceLimit) {
                  onPlane = false;
                  break;
                }
              if (!onPlane)
                continue;
              double cosine = std::abs(Dot(normal, ToVec(triangles[n].Normal)));
              double angle = std::acos(std::max(-1.0, std::min(1.0, cosine)));
              double candidateQuality = quality(n);
              if (angle > angularLimit && candidateQuality > .015)
                continue;
              visitStamp[n] = stamp;
              q.push(n);
            }
        }
        if (region.size() >=
            size_t(s.mConfig.MinimumPlanarConsolidationTriangles)) {
          int id = int(planes.size());
          for (int t : region)
            state[t] = id;
          planes.push_back(std::move(region));
        }
      }
    if (!planes.empty()) {
      std::vector<MeshPatch> rebuilt;
      rebuilt.reserve(planes.size() + s.mPatches.size());
      for (auto &ids : planes) {
        MeshPatch patch;
        patch.TriangleIds = std::move(ids);
        ApplyFit(patch, Fit(s.mMesh, patch, s.mConfig));
        rebuilt.push_back(std::move(patch));
      }
      std::vector<char> seen(triangles.size(), 0);
      for (int seed = 0; seed < int(triangles.size()); ++seed)
        if (state[seed] < 0 && !seen[seed]) {
          int oldPatch = triangles[seed].PatchId;
          MeshPatch patch;
          std::queue<int> q;
          q.push(seed);
          seen[seed] = 1;
          while (!q.empty()) {
            int t = q.front();
            q.pop();
            patch.TriangleIds.push_back(t);
            for (int n : s.mMesh.getTriangleNeighbors(t))
              if (state[n] < 0 && !seen[n] &&
                  triangles[n].PatchId == oldPatch) {
                seen[n] = 1;
                q.push(n);
              }
          }
          ApplyFit(patch, Fit(s.mMesh, patch, s.mConfig));
          rebuilt.push_back(std::move(patch));
        }
      s.mPatches = std::move(rebuilt);
      s.AssignPatchIds();
      if (s.mConfig.Verbose)
        std::clog << "[CadMesh] planar consolidation: " << planes.size()
                  << " global plane regions, patches=" << s.mPatches.size()
                  << '\n';
    }
  }
  // Recover a complete cylindrical CAD face from a reliable local cylinder
  // seed.  Large-radius cylinders are nearly planar over a small stencil and
  // otherwise tend to leave alternating Cylinder/Freeform strips.  Growth is
  // performed against one fixed axis/radius model and the complete result is
  // refitted before it is accepted, so adjacency alone can never pull a
  // fillet, plane, or unrelated cylinder into the region.
  {
    const auto &triangles = s.mMesh.getTriangles();
    const auto &vertices = s.mMesh.getVertices();
    const auto &resolution = s.mMesh.getResolution();
    std::vector<int> seeds;
    for (int patchId = 0; patchId < int(s.mPatches.size()); ++patchId) {
      const auto &patch = s.mPatches[patchId];
      if (patch.SurfaceType == PatchSurfaceType::Cylinder &&
          patch.TriangleIds.size() >= 12 && patch.Confidence >= .55 &&
          std::holds_alternative<CylinderParameters>(patch.Parameters))
        seeds.push_back(patchId);
    }
    std::sort(seeds.begin(), seeds.end(), [&](int a, int b) {
      return s.mPatches[a].TriangleIds.size() >
             s.mPatches[b].TriangleIds.size();
    });
    std::vector<char> claimed(triangles.size(), 0);
    std::vector<int> visitStamp(triangles.size(), 0);
    std::vector<MeshPatch> extracted;
    int stamp = 0;
    auto matchesCylinder = [&](int triangleId,
                               const CylinderParameters &parameters) {
      Vec3 axis = Normalize(ToVec(parameters.Axis.Direction));
      Vec3 origin = ToVec(parameters.Axis.Origin);
      double distanceLimit = std::max(10 * resolution.FittingTolerance,
                                      .015 * resolution.MedianEdgeLength);
      for (int vertexId : triangles[triangleId].VertexIds) {
        Vec3 delta = Sub(ToVec(vertices[vertexId].Position), origin);
        Vec3 radial = Sub(delta, Mul(axis, Dot(delta, axis)));
        if (std::abs(Norm(radial) - parameters.Radius) > distanceLimit)
          return false;
      }
      Vec3 delta = Sub(ToVec(triangles[triangleId].Centroid), origin);
      Vec3 radial = Sub(delta, Mul(axis, Dot(delta, axis)));
      if (Norm(radial) <= 1e-20)
        return false;
      double cosine = std::abs(Dot(
          Normalize(radial), Normalize(ToVec(triangles[triangleId].Normal))));
      double normalError = std::acos(std::max(-1.0, std::min(1.0, cosine)));
      return normalError <= std::max(16 * resolution.AngularTolerance, .14);
    };
    for (int seedPatchId : seeds) {
      const auto &seedPatch = s.mPatches[seedPatchId];
      if (claimed[seedPatch.TriangleIds.front()])
        continue;
      const auto *parameters =
          std::get_if<CylinderParameters>(&seedPatch.Parameters);
      if (!parameters)
        continue;
      ++stamp;
      std::queue<int> queue;
      std::vector<int> region;
      for (int triangleId : seedPatch.TriangleIds) {
        if (!claimed[triangleId] && visitStamp[triangleId] != stamp) {
          visitStamp[triangleId] = stamp;
          queue.push(triangleId);
        }
      }
      while (!queue.empty()) {
        int triangleId = queue.front();
        queue.pop();
        region.push_back(triangleId);
        for (int edgeId : triangles[triangleId].EdgeIds) {
          const auto &edge = s.mMesh.getEdges()[edgeId];
          if (IsHardSegmentationBoundary(edge, s.mConfig))
            continue;
          for (int neighbor : edge.IncidentTriangleIds) {
            if (neighbor == triangleId || claimed[neighbor] ||
                visitStamp[neighbor] == stamp)
              continue;
            int owner = triangles[neighbor].PatchId;
            if (owner < 0 || owner >= int(s.mPatches.size()))
              continue;
            auto ownerType = s.mPatches[owner].SurfaceType;
            if (ownerType != PatchSurfaceType::Freeform &&
                ownerType != PatchSurfaceType::Cylinder)
              continue;
            visitStamp[neighbor] = stamp;
            if (matchesCylinder(neighbor, *parameters))
              queue.push(neighbor);
          }
        }
      }
      if (region.size() < seedPatch.TriangleIds.size() + 12)
        continue;
      MeshPatch candidate;
      candidate.TriangleIds = std::move(region);
      auto fit = Fit(s.mMesh, candidate, s.mConfig);
      if (fit.Type != PatchSurfaceType::Cylinder ||
          fit.Rms > 8 * resolution.FittingTolerance ||
          fit.Max > 20 * resolution.FittingTolerance || fit.Normal > .14)
        continue;
      ApplyFit(candidate, fit);
      for (int triangleId : candidate.TriangleIds)
        claimed[triangleId] = 1;
      extracted.push_back(std::move(candidate));
    }
    if (!extracted.empty()) {
      std::vector<MeshPatch> rebuilt;
      rebuilt.reserve(s.mPatches.size() + extracted.size());
      for (auto &patch : extracted)
        rebuilt.push_back(std::move(patch));
      std::vector<char> seen = claimed;
      for (const auto &oldPatch : s.mPatches)
        for (int seed : oldPatch.TriangleIds)
          if (!seen[seed]) {
            MeshPatch remainder;
            std::queue<int> queue;
            queue.push(seed);
            seen[seed] = 1;
            while (!queue.empty()) {
              int triangleId = queue.front();
              queue.pop();
              remainder.TriangleIds.push_back(triangleId);
              for (int neighbor : s.mMesh.getTriangleNeighbors(triangleId))
                if (!seen[neighbor] &&
                    triangles[neighbor].PatchId == oldPatch.Id) {
                  seen[neighbor] = 1;
                  queue.push(neighbor);
                }
            }
            ApplyFit(remainder, Fit(s.mMesh, remainder, s.mConfig));
            rebuilt.push_back(std::move(remainder));
          }
      s.mPatches = std::move(rebuilt);
      s.AssignPatchIds();
      if (s.mConfig.Verbose)
        std::clog << "[CadMesh] global cylinder recovery: extracted "
                  << extracted.size()
                  << " complete cylinder regions, patches=" << s.mPatches.size()
                  << '\n';
    }
  }
  // Batch-assign one/two-triangle fragments.  A fragment must choose an
  // adjacent larger owner when the interface is smooth and the owner's
  // global analytic model also explains all fragment vertices.
  for (int pass = 0; pass < 8; ++pass) {
    s.AssignPatchIds();
    auto pairs = PatchPairs(s.mMesh);
    int count = int(s.mPatches.size());
    std::vector<int> choice(count, -1);
    std::vector<double> best(count, 1e100);
    for (const auto &item : pairs) {
      int a = item.first.first, b = item.first.second;
      double boundary = AverageBoundary(s.mMesh, item.second), angle = 0;
      for (int e : item.second) {
        const auto &edge = s.mMesh.getEdges()[e];
        angle += std::acos(std::max(
            -1.0,
            std::min(
                1.0,
                std::abs(Dot(
                    ToVec(s.mMesh.getTriangles()[edge.Triangle0].Normal),
                    ToVec(s.mMesh.getTriangles()[edge.Triangle1].Normal))))));
      }
      angle /= std::max<size_t>(1, item.second.size());
      auto consider = [&](int source, int target) {
        size_t sourceSize = s.mPatches[source].TriangleIds.size();
        double angleLimit =
            (sourceSize == 1 ? 30.0 : 8.0) * std::acos(-1.0) / 180.0;
        if (sourceSize > size_t(s.mConfig.MinimumPatchTriangles) ||
            s.mPatches[target].TriangleIds.size() < sourceSize ||
            angle > angleLimit || boundary > .98)
          return;
        double residual =
            PatchModelResidual(s.mMesh, s.mPatches[source], s.mPatches[target]);
        if (ProtectedInterface(s.mMesh, item.second, s.mConfig) ||
            !FitsFixedTarget(s.mMesh, s.mPatches[source], s.mPatches[target]))
          return;
        double cost =
            2 * angle / angleLimit + boundary + .15 * residual -
            .05 * std::log1p(double(s.mPatches[target].TriangleIds.size()));
        if (cost < best[source]) {
          best[source] = cost;
          choice[source] = target;
        }
      };
      consider(a, b);
      consider(b, a);
    }
    std::vector<int> parent(count);
    for (int i = 0; i < count; ++i)
      parent[i] = i;
    auto find = [&](int x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    int changes = 0;
    std::vector<int> sources(count);
    for (int i = 0; i < count; ++i)
      sources[i] = i;
    std::sort(sources.begin(), sources.end(), [&](int a, int b) {
      return s.mPatches[a].TriangleIds.size() <
             s.mPatches[b].TriangleIds.size();
    });
    for (int source : sources)
      if (choice[source] >= 0) {
        int a = find(source), b = find(choice[source]);
        if (a != b && (s.mPatches[b].TriangleIds.size() >
                           s.mPatches[a].TriangleIds.size() ||
                       (s.mPatches[b].TriangleIds.size() ==
                            s.mPatches[a].TriangleIds.size() &&
                        b < a))) {
          parent[a] = b;
          ++changes;
        }
      }
    if (!changes)
      break;
    for (int i = 0; i < count; ++i)
      parent[i] = find(i);
    std::map<int, std::vector<int>> groups;
    for (int i = 0; i < count; ++i)
      groups[parent[i]].push_back(i);
    std::vector<MeshPatch> rebuilt;
    rebuilt.reserve(groups.size());
    for (auto &item : groups) {
      int root = item.first;
      if (item.second.size() == 1) {
        rebuilt.push_back(std::move(s.mPatches[root]));
        continue;
      }
      MeshPatch patch = s.mPatches[root];
      for (int member : item.second)
        if (member != root)
          patch.TriangleIds.insert(patch.TriangleIds.end(),
                                   s.mPatches[member].TriangleIds.begin(),
                                   s.mPatches[member].TriangleIds.end());
      std::sort(patch.TriangleIds.begin(), patch.TriangleIds.end());
      patch.TriangleIds.erase(
          std::unique(patch.TriangleIds.begin(), patch.TriangleIds.end()),
          patch.TriangleIds.end());
      if (!FitsFixedTarget(s.mMesh, patch, s.mPatches[root])) {
        for (int member : item.second)
          rebuilt.push_back(std::move(s.mPatches[member]));
        continue;
      }
      if (!(patch.SurfaceType == PatchSurfaceType::Freeform &&
            patch.TriangleIds.size() > 100000))
        ApplyFit(patch, Fit(s.mMesh, patch, s.mConfig));
      if (!FitsFixedTarget(s.mMesh, patch, patch)) {
        for (int member : item.second)
          rebuilt.push_back(std::move(s.mPatches[member]));
        continue;
      }
      rebuilt.push_back(std::move(patch));
    }
    s.mPatches = std::move(rebuilt);
    if (s.mConfig.Verbose)
      std::clog << "[CadMesh] small-patch assignment pass " << pass + 1
                << ": merged=" << changes << ", patches=" << s.mPatches.size()
                << '\n';
  }
  // Absorb enclosed islands and narrow strips into a dominant analytic
  // owner.  Classification of a skinny strip is often ill-conditioned, so
  // ownership is decided by evaluating every source vertex against the
  // neighbor's already established global surface equation.  A real fillet
  // or chamfer cannot pass this test against its supporting plane.
  for (int pass = 0; pass < 4; ++pass) {
    s.AssignPatchIds();
    auto pairs = PatchPairs(s.mMesh);
    int count = int(s.mPatches.size());
    struct Contact {
      int Neighbor = -1;
      double Length = 0, WeightedScore = 0;
    };
    std::vector<std::vector<Contact>> contacts(count);
    std::vector<double> total(count, 0);
    for (const auto &item : pairs) {
      int a = item.first.first, b = item.first.second;
      double length = 0, score = 0;
      for (int edgeId : item.second) {
        const auto &e = s.mMesh.getEdges()[edgeId];
        double edgeLength = Distance(s.mMesh.getVertices()[e.Vertex0].Position,
                                     s.mMesh.getVertices()[e.Vertex1].Position);
        length += edgeLength;
        score += edgeLength * e.BoundaryScore;
      }
      contacts[a].push_back({b, length, score});
      contacts[b].push_back({a, length, score});
      total[a] += length;
      total[b] += length;
    }
    std::vector<int> parent(count);
    for (int i = 0; i < count; ++i)
      parent[i] = i;
    int changes = 0;
    for (int source = 0; source < count; ++source) {
      const auto &patch = s.mPatches[source];
      if (patch.TriangleIds.size() > 5000 || contacts[source].empty())
        continue;
      const Contact *best = nullptr;
      double bestCost = 1e100;
      for (const auto &contact : contacts[source]) {
        int target = contact.Neighbor;
        double dominance = contact.Length / std::max(total[source], 1e-30),
               confidence =
                   contact.WeightedScore / std::max(contact.Length, 1e-30);
        const auto &type = s.mPatches[target].SurfaceType;
        bool analyticTarget = type == PatchSurfaceType::Plane ||
                              type == PatchSurfaceType::Cylinder ||
                              type == PatchSurfaceType::Sphere;
        std::vector<int> sharedEdges;
        auto pairIt =
            pairs.find({std::min(source, target), std::max(source, target)});
        if (pairIt != pairs.end())
          sharedEdges = pairIt->second;
        if (!analyticTarget ||
            ProtectedInterface(s.mMesh, sharedEdges, s.mConfig) ||
            !FitsFixedTarget(s.mMesh, patch, s.mPatches[target]))
          continue;
        if (s.mPatches[target].TriangleIds.size() < patch.TriangleIds.size() ||
            (s.mPatches[target].TriangleIds.size() ==
                 patch.TriangleIds.size() &&
             target > source))
          continue;
        double residual =
            PatchModelResidual(s.mMesh, patch, s.mPatches[target]);
        if (residual > 8)
          continue;
        double requiredDominance = residual <= 2 ? .45 : .60;
        if (dominance < requiredDominance || confidence >= .90)
          continue;
        double cost = residual + 2 * confidence - dominance;
        if (cost < bestCost) {
          bestCost = cost;
          best = &contact;
        }
      }
      if (!best)
        continue;
      parent[source] = best->Neighbor;
      ++changes;
    }
    if (!changes)
      break;
    auto find = [&](int x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    for (int i = 0; i < count; ++i)
      parent[i] = find(i);
    std::map<int, std::vector<int>> groups;
    for (int i = 0; i < count; ++i)
      groups[parent[i]].push_back(i);
    std::vector<MeshPatch> rebuilt;
    rebuilt.reserve(groups.size());
    for (auto &item : groups) {
      int root = item.first;
      MeshPatch patch = s.mPatches[root];
      if (item.second.size() > 1) {
        for (int member : item.second)
          if (member != root)
            patch.TriangleIds.insert(patch.TriangleIds.end(),
                                     s.mPatches[member].TriangleIds.begin(),
                                     s.mPatches[member].TriangleIds.end());
        std::sort(patch.TriangleIds.begin(), patch.TriangleIds.end());
        patch.TriangleIds.erase(
            std::unique(patch.TriangleIds.begin(), patch.TriangleIds.end()),
            patch.TriangleIds.end());
        if (!FitsFixedTarget(s.mMesh, patch, s.mPatches[root])) {
          for (int member : item.second)
            rebuilt.push_back(std::move(s.mPatches[member]));
          continue;
        }
        if (!(patch.SurfaceType == PatchSurfaceType::Freeform &&
              patch.TriangleIds.size() > 100000))
          ApplyFit(patch, Fit(s.mMesh, patch, s.mConfig));
        if (!FitsFixedTarget(s.mMesh, patch, patch)) {
          for (int member : item.second)
            rebuilt.push_back(std::move(s.mPatches[member]));
          continue;
        }
      }
      rebuilt.push_back(std::move(patch));
    }
    s.mPatches = std::move(rebuilt);
    if (s.mConfig.Verbose)
      std::clog << "[CadMesh] analytic orphan assignment pass " << pass + 1
                << ": merged=" << changes << ", patches=" << s.mPatches.size()
                << '\n';
  }
  // Recover feature boundaries that were missed before region growing.  This
  // pass is deliberately O(F + E): all internal candidate edges are collected
  // once, then every affected patch is split in a batch.  Unlike the legacy
  // exhaustive refinement below it remains enabled for very large STL files.
  if (s.mConfig.EnableInternalFeatureSplit) {
    for (int splitPass = 0; splitPass < s.mConfig.InternalSplitMaximumPasses;
         ++splitPass) {
      s.AssignPatchIds();
      const auto &triangles = s.mMesh.getTriangles();
      auto &edges = s.mMesh.getEdges();
      std::vector<std::vector<int>> cutEdges(s.mPatches.size());
      for (int edgeId = 0; edgeId < int(edges.size()); ++edgeId) {
        const auto &edge = edges[edgeId];
        if (edge.IsBoundary || edge.IsNonManifold || edge.Triangle0 < 0 ||
            edge.Triangle1 < 0)
          continue;
        int patch0 = triangles[edge.Triangle0].PatchId;
        int patch1 = triangles[edge.Triangle1].PatchId;
        if (patch0 < 0 || patch0 != patch1)
          continue;
        double effectiveBoundary = edge.BoundaryScore;
        double minimumQuality =
            std::min(TriangleQuality(s.mMesh, edge.Triangle0),
                     TriangleQuality(s.mMesh, edge.Triangle1));
        if (minimumQuality < s.mConfig.ClosureSkinnyTriangleQuality)
          effectiveBoundary =
              std::max(0.0, effectiveBoundary -
                                s.mConfig.NormalWeight *
                                    edge.Evidence.NormalDiscontinuity -
                                s.mConfig.TessellationWeight *
                                    edge.Evidence.TessellationEvidence);
        double geometricEvidence =
            std::max(edge.Evidence.SurfaceFitDiscontinuity,
                     std::max(edge.Evidence.CurvatureDiscontinuity,
                              edge.Evidence.CurvatureGradient));
        bool strong = effectiveBoundary >= s.mConfig.StrongBoundaryThreshold;
        bool probable =
            effectiveBoundary >= s.mConfig.InternalSplitBoundaryThreshold &&
            geometricEvidence >= s.mConfig.WeakBoundaryThreshold;
        // A G1 transition can have a modest final score because its normals
        // are continuous.  The left/right-vs-combined quadric residual is a
        // direct model-change signal, so let a persistent residual ridge
        // create an internal seed boundary even below the score threshold.
        double supportingEvidence =
            std::max(edge.Evidence.CurvatureDiscontinuity,
                     edge.Evidence.CurvatureGradient);
        bool residualRidge =
            edge.Evidence.SurfaceFitDiscontinuity >=
                s.mConfig.InternalSplitResidualRidgeThreshold &&
            supportingEvidence >= s.mConfig.InternalSplitSupportingEvidence;
        if (strong || probable || residualRidge)
          cutEdges[patch0].push_back(edgeId);
      }

      // Promote edges as complete feature chains rather than independent
      // binary decisions. Short, disconnected or rapidly folding chains are
      // typical tessellation noise and cannot seed a split.
      for (int patchId = 0; patchId < int(cutEdges.size()); ++patchId) {
        auto &patchCuts = cutEdges[patchId];
        if (patchCuts.empty())
          continue;
        std::unordered_map<int, std::vector<int>> vertexEdges;
        vertexEdges.reserve(patchCuts.size() * 2);
        for (int edgeId : patchCuts) {
          vertexEdges[edges[edgeId].Vertex0].push_back(edgeId);
          vertexEdges[edges[edgeId].Vertex1].push_back(edgeId);
        }
        std::unordered_set<int> visitedEdges;
        visitedEdges.reserve(patchCuts.size());
        std::vector<int> filtered;
        for (int seedEdge : patchCuts) {
          if (!visitedEdges.insert(seedEdge).second)
            continue;
          std::vector<int> chain;
          std::queue<int> queue;
          queue.push(seedEdge);
          while (!queue.empty()) {
            int edgeId = queue.front();
            queue.pop();
            chain.push_back(edgeId);
            for (int vertexId : {edges[edgeId].Vertex0, edges[edgeId].Vertex1})
              for (int neighborEdge : vertexEdges[vertexId])
                if (visitedEdges.insert(neighborEdge).second)
                  queue.push(neighborEdge);
          }
          if (chain.size() < size_t(s.mConfig.InternalSplitMinimumCutEdges))
            continue;
          double chainLength = 0;
          std::unordered_map<int, int> degree;
          for (int edgeId : chain) {
            chainLength +=
                Distance(s.mMesh.getVertices()[edges[edgeId].Vertex0].Position,
                         s.mMesh.getVertices()[edges[edgeId].Vertex1].Position);
            ++degree[edges[edgeId].Vertex0];
            ++degree[edges[edgeId].Vertex1];
          }
          if (chainLength < s.mConfig.InternalSplitMinimumChainLengthFactor *
                                s.mMesh.getResolution().MedianEdgeLength)
            continue;
          std::vector<int> endpoints;
          int maximumDegree = 0;
          for (const auto &item : degree) {
            maximumDegree = std::max(maximumDegree, item.second);
            if (item.second == 1)
              endpoints.push_back(item.first);
          }
          bool regular = maximumDegree <= 4;
          if (endpoints.size() == 2) {
            double chord =
                Distance(s.mMesh.getVertices()[endpoints[0]].Position,
                         s.mMesh.getVertices()[endpoints[1]].Position);
            regular = regular && chord / std::max(chainLength, 1e-30) >= .15;
          } else if (!endpoints.empty() && endpoints.size() > 4) {
            regular = false;
          }
          if (regular)
            filtered.insert(filtered.end(), chain.begin(), chain.end());
        }
        patchCuts = std::move(filtered);
      }

      std::vector<std::vector<MeshPatch>> accepted(s.mPatches.size());
      std::vector<int> acceptedBoundaryEdges;
      int splitPatchCount = 0;
      int createdPatchCount = 0;
      for (int patchId = 0; patchId < int(s.mPatches.size()); ++patchId) {
        const auto &original = s.mPatches[patchId];
        if (cutEdges[patchId].size() <
                size_t(s.mConfig.InternalSplitMinimumCutEdges) ||
            original.TriangleIds.size() <
                size_t(2 * s.mConfig.InternalSplitMinimumTriangles))
          continue;
        std::unordered_set<int> cut(cutEdges[patchId].begin(),
                                    cutEdges[patchId].end());
        std::unordered_set<int> seen;
        seen.reserve(original.TriangleIds.size());
        std::vector<std::vector<int>> components;
        for (int seed : original.TriangleIds)
          if (seen.insert(seed).second) {
            components.emplace_back();
            std::queue<int> queue;
            queue.push(seed);
            while (!queue.empty()) {
              int triangleId = queue.front();
              queue.pop();
              components.back().push_back(triangleId);
              for (int edgeId : triangles[triangleId].EdgeIds) {
                if (cut.count(edgeId))
                  continue;
                for (int neighbor : edges[edgeId].IncidentTriangleIds)
                  if (triangles[neighbor].PatchId == patchId &&
                      seen.insert(neighbor).second)
                    queue.push(neighbor);
              }
            }
          }
        std::vector<int> largeComponentIds;
        for (int componentId = 0; componentId < int(components.size());
             ++componentId)
          if (components[componentId].size() >=
              size_t(s.mConfig.InternalSplitMinimumTriangles))
            largeComponentIds.push_back(componentId);
        if (largeComponentIds.size() < 2)
          continue;

        // Small components may be real chamfers. Preserve them in the
        // accepted partition; only guarded later merges may change ownership.
        std::vector<MeshPatch> candidates;
        candidates.reserve(components.size());
        double weightedRmsSquared = 0;
        size_t fittedTriangleCount = 0;
        for (int componentId = 0; componentId < int(components.size());
             ++componentId) {
          MeshPatch candidate;
          candidate.TriangleIds = std::move(components[componentId]);
          ApplyFit(candidate, Fit(s.mMesh, candidate, s.mConfig));
          weightedRmsSquared += candidate.TriangleIds.size() *
                                candidate.RmsFittingError *
                                candidate.RmsFittingError;
          fittedTriangleCount += candidate.TriangleIds.size();
          candidates.push_back(std::move(candidate));
        }
        double splitRms = std::sqrt(weightedRmsSquared /
                                    std::max<size_t>(1, fittedTriangleCount));
        bool analyticDifference = false;
        bool typeDifference = false;
        for (int i = 0; i < int(candidates.size()); ++i)
          for (int j = i + 1; j < int(candidates.size()); ++j) {
            bool analyticI =
                candidates[i].SurfaceType == PatchSurfaceType::Plane ||
                candidates[i].SurfaceType == PatchSurfaceType::Cylinder ||
                candidates[i].SurfaceType == PatchSurfaceType::Sphere;
            bool analyticJ =
                candidates[j].SurfaceType == PatchSurfaceType::Plane ||
                candidates[j].SurfaceType == PatchSurfaceType::Cylinder ||
                candidates[j].SurfaceType == PatchSurfaceType::Sphere;
            if (analyticI && analyticJ &&
                !SimilarAnalytic(candidates[i], candidates[j],
                                 s.mMesh.getResolution()))
              analyticDifference = true;
            if (candidates[i].SurfaceType != candidates[j].SurfaceType &&
                candidates[i].SurfaceType != PatchSurfaceType::Freeform &&
                candidates[j].SurfaceType != PatchSurfaceType::Freeform)
              typeDifference = true;
          }
        bool errorImprovement = original.RmsFittingError > 0 &&
                                splitRms <= s.mConfig.InternalSplitErrorFactor *
                                                original.RmsFittingError;
        if (!analyticDifference && !typeDifference && !errorImprovement)
          continue;
        if (s.mConfig.Verbose) {
          std::clog << "[CadMesh] internal split accepted: patch=" << patchId
                    << ", original=" << SurfaceTypeName(original.SurfaceType)
                    << ':' << original.TriangleIds.size()
                    << ", rms=" << original.RmsFittingError << " -> "
                    << splitRms << ", components=";
          for (size_t candidateId = 0; candidateId < candidates.size();
               ++candidateId) {
            if (candidateId)
              std::clog << ',';
            std::clog << SurfaceTypeName(candidates[candidateId].SurfaceType)
                      << ':' << candidates[candidateId].TriangleIds.size();
          }
          std::clog << '\n';
        }
        accepted[patchId] = std::move(candidates);
        acceptedBoundaryEdges.insert(acceptedBoundaryEdges.end(),
                                     cutEdges[patchId].begin(),
                                     cutEdges[patchId].end());
        ++splitPatchCount;
        createdPatchCount += int(candidates.size());
      }
      if (!splitPatchCount)
        break;
      std::vector<MeshPatch> rebuilt;
      rebuilt.reserve(s.mPatches.size() + createdPatchCount);
      for (int patchId = 0; patchId < int(s.mPatches.size()); ++patchId) {
        if (accepted[patchId].empty()) {
          rebuilt.push_back(std::move(s.mPatches[patchId]));
          continue;
        }
        for (auto &patch : accepted[patchId])
          rebuilt.push_back(std::move(patch));
      }
      s.mPatches = std::move(rebuilt);
      s.AssignPatchIds();
      for (int edgeId : acceptedBoundaryEdges) {
        auto &edge = edges[edgeId];
        if (edge.Triangle0 >= 0 && edge.Triangle1 >= 0 &&
            triangles[edge.Triangle0].PatchId !=
                triangles[edge.Triangle1].PatchId) {
          edge.BoundaryScore = std::max(
              edge.BoundaryScore, s.mConfig.InternalSplitAcceptedBoundaryScore);
          edge.Evidence.FinalScore = edge.BoundaryScore;
          edge.IsConstrainedFeature = true;
        }
      }
      if (s.mConfig.Verbose)
        std::clog << "[CadMesh] internal feature split pass " << splitPass + 1
                  << ": split_patches=" << splitPatchCount
                  << ", patches=" << s.mPatches.size() << '\n';
    }
  }
  {
    auto heat = HeatPatchRegularizer::regularize(s);
    if (s.mConfig.Verbose)
      std::clog << "[CadMesh] heat regularization: ambiguous_components="
                << heat.AmbiguousComponents
                << ", solved=" << heat.SolvedComponents
                << ", oversized=" << heat.SkippedOversizedComponents
                << ", failed=" << heat.FailedSolves
                << ", demoted_seeds=" << heat.DemotedGeometricSeeds
                << ", pruned_seeds=" << heat.PrunedThermalSeeds
                << ", resolve_passes=" << heat.ResolvePasses
                << ", reassigned=" << heat.ReassignedTriangles
                << ", patches=" << s.mPatches.size() << '\n';
  }
  // Close one-triangle notches and peel embedded misclassified strips.  Two or
  // three shared edges are sufficient directly.  A one-edge candidate is also
  // allowed when all three of its vertices are already incident to the same
  // target patch; this is the vertex-closure pattern produced by a saw-tooth
  // strip.  Because one-edge removal may cut a source strip, every changed
  // source is split back into connected components after the batch.
  {
    int pass = 0;
    size_t totalMoved = 0;
    while (pass < s.mConfig.MaximumFaceClosurePasses) {
      ++pass;
      s.AssignPatchIds();
      const auto &triangles = s.mMesh.getTriangles();
      const auto &edges = s.mMesh.getEdges();
      const auto &vertices = s.mMesh.getVertices();
      const auto &resolution = s.mMesh.getResolution();
      std::vector<int> destination(triangles.size(), -1);
      size_t moved = 0;
      struct FaceContact {
        int EdgeCount = 0;
        bool HasConstrainedFeature = false;
        double Length = 0;
        double WeightedBoundary = 0;
        double WeightedNormalEvidence = 0;
        double WeightedTessellationEvidence = 0;
        Vec3 WeightedNormal{0, 0, 0};
      };
      for (int triangleId = 0; triangleId < int(triangles.size());
           ++triangleId) {
        int source = triangles[triangleId].PatchId;
        if (source < 0 || source >= int(s.mPatches.size()))
          continue;
        std::map<int, FaceContact> contacts;
        for (int edgeId : triangles[triangleId].EdgeIds) {
          const auto &edge = edges[edgeId];
          if (edge.IsBoundary || edge.IsNonManifold ||
              edge.IncidentTriangleIds.size() != 2)
            continue;
          int neighbor = edge.IncidentTriangleIds[0] == triangleId
                             ? edge.IncidentTriangleIds[1]
                             : edge.IncidentTriangleIds[0];
          int target = triangles[neighbor].PatchId;
          if (target < 0 || target == source ||
              target >= int(s.mPatches.size()))
            continue;
          double length = Distance(vertices[edge.Vertex0].Position,
                                   vertices[edge.Vertex1].Position);
          auto &contact = contacts[target];
          ++contact.EdgeCount;
          contact.HasConstrainedFeature =
              contact.HasConstrainedFeature ||
              IsHardSegmentationBoundary(edge, s.mConfig);
          contact.Length += length;
          contact.WeightedBoundary += length * edge.BoundaryScore;
          contact.WeightedNormalEvidence +=
              length * edge.Evidence.NormalDiscontinuity;
          contact.WeightedTessellationEvidence +=
              length * edge.Evidence.TessellationEvidence;
          Vec3 adjacentNormal = ToVec(triangles[neighbor].Normal);
          if (Dot(adjacentNormal, ToVec(triangles[triangleId].Normal)) < 0)
            adjacentNormal = Mul(adjacentNormal, -1);
          contact.WeightedNormal =
              Add(contact.WeightedNormal,
                  Mul(adjacentNormal, length * triangles[neighbor].Area));
        }
        int bestTarget = -1;
        double bestCost = 1e100;
        for (const auto &item : contacts) {
          int target = item.first;
          const auto &contact = item.second;
          if (contact.HasConstrainedFeature)
            continue;
          int supportedVertexCount = 0;
          if (contact.EdgeCount == 1) {
            for (int vertexId : triangles[triangleId].VertexIds) {
              for (int incident : vertices[vertexId].IncidentTriangleIds)
                if (triangles[incident].PatchId == target) {
                  ++supportedVertexCount;
                  break;
                }
            }
          }
          size_t sourceSize = s.mPatches[source].TriangleIds.size();
          size_t targetSize = s.mPatches[target].TriangleIds.size();
          if (targetSize < sourceSize ||
              (targetSize == sourceSize && target > source))
            continue;
          double longestSquared = 0;
          for (int edgeId : triangles[triangleId].EdgeIds) {
            const auto &edge = edges[edgeId];
            Vec3 delta = Sub(ToVec(vertices[edge.Vertex1].Position),
                             ToVec(vertices[edge.Vertex0].Position));
            longestSquared = std::max(longestSquared, Dot(delta, delta));
          }
          double triangleQuality =
              triangles[triangleId].Area / std::max(longestSquared, 1e-30);
          bool skinnyTriangle =
              triangleQuality < s.mConfig.ClosureSkinnyTriangleQuality;
          bool vertexClosure =
              contact.EdgeCount == 1 && supportedVertexCount == 3;
          bool weakStripClosure =
              contact.EdgeCount == 1 && supportedVertexCount >= 2 &&
              skinnyTriangle &&
              sourceSize <=
                  size_t(s.mConfig.ClosureWeakStripMaximumTriangles) &&
              double(targetSize) >=
                  s.mConfig.ClosureTargetSizeRatio * double(sourceSize);
          if (contact.EdgeCount < 2 && !vertexClosure && !weakStripClosure)
            continue;
          double rawBoundary =
              contact.WeightedBoundary / std::max(contact.Length, 1e-30);
          double boundary = rawBoundary;
          // Long, nearly degenerate CAD tessellation triangles make both the
          // single-face normal and the local quadric/tessellation evidence
          // unreliable.  Remove those two noise-sensitive contributions here;
          // the target's global analytic equation is checked below.
          if (skinnyTriangle) {
            boundary =
                std::max(0.0, boundary -
                                  s.mConfig.NormalWeight *
                                      contact.WeightedNormalEvidence /
                                      std::max(contact.Length, 1e-30) -
                                  s.mConfig.TessellationWeight *
                                      contact.WeightedTessellationEvidence /
                                      std::max(contact.Length, 1e-30));
          }
          Vec3 supportedNormal = contact.WeightedNormal;
          if (vertexClosure || weakStripClosure)
            for (int vertexId : triangles[triangleId].VertexIds)
              for (int incident : vertices[vertexId].IncidentTriangleIds)
                if (triangles[incident].PatchId == target) {
                  Vec3 normal = ToVec(triangles[incident].Normal);
                  if (Dot(normal, ToVec(triangles[triangleId].Normal)) < 0)
                    normal = Mul(normal, -1);
                  supportedNormal = Add(supportedNormal,
                                        Mul(normal, triangles[incident].Area));
                }
          Vec3 neighborNormal = Normalize(supportedNormal);
          double cosine = std::abs(Dot(
              Normalize(ToVec(triangles[triangleId].Normal)), neighborNormal));
          double normalError = std::acos(std::max(-1.0, std::min(1.0, cosine)));
          MeshPatch candidate;
          candidate.TriangleIds.push_back(triangleId);
          double residual =
              PatchModelResidual(s.mMesh, candidate, s.mPatches[target]);
          bool analyticTarget =
              s.mPatches[target].SurfaceType == PatchSurfaceType::Plane ||
              s.mPatches[target].SurfaceType == PatchSurfaceType::Cylinder ||
              s.mPatches[target].SurfaceType == PatchSurfaceType::Sphere;
          if (!analyticTarget ||
              !FitsFixedTarget(s.mMesh, candidate, s.mPatches[target]))
            continue;
          // A triangle whose three vertices satisfy the large neighboring
          // analytic surface is stronger evidence than a noisy score computed
          // from that triangle alone. A true fillet/chamfer triangle has an
          // off-surface third vertex and therefore cannot use this override.
          bool analyticGeometryClosure =
              analyticTarget &&
              residual <= s.mConfig.ClosureAnalyticResidualTolerance;
          if (weakStripClosure && !analyticGeometryClosure)
            continue;
          double boundaryLimit = contact.EdgeCount == 3            ? .78
                                 : contact.EdgeCount == 2          ? .60
                                 : vertexClosure && skinnyTriangle ? 1.01
                                                                   : .75;
          if (analyticGeometryClosure ||
              (skinnyTriangle && contact.EdgeCount >= 2))
            boundaryLimit = 1.01;
          if (boundary >= boundaryLimit)
            continue;
          double normalLimit =
              contact.EdgeCount == 3
                  ? std::max(20 * resolution.AngularTolerance, .17)
              : contact.EdgeCount == 2
                  ? std::max(10 * resolution.AngularTolerance, .09)
              : vertexClosure && skinnyTriangle
                  ? 1.58
                  : std::max(12 * resolution.AngularTolerance, .17);
          if (analyticGeometryClosure && skinnyTriangle)
            normalLimit = 1.58;
          if (normalError > normalLimit)
            continue;
          double residualLimit = contact.EdgeCount == 3   ? 12.0
                                 : contact.EdgeCount == 2 ? 8.0
                                                          : 6.0;
          if (analyticTarget && residual > residualLimit)
            continue;
          double normalWeight =
              analyticGeometryClosure && skinnyTriangle ? .25 : 4.0;
          double cost = (3 - contact.EdgeCount) + 2 * boundary +
                        normalWeight * normalError +
                        (analyticTarget ? .1 * residual : 0);
          if (cost < bestCost) {
            bestCost = cost;
            bestTarget = target;
          }
        }
        if (bestTarget >= 0) {
          destination[triangleId] = bestTarget;
          ++moved;
        }
      }
      if (!moved)
        break;
      totalMoved += moved;
      std::vector<std::vector<int>> patchTriangles(s.mPatches.size());
      std::vector<char> changed(s.mPatches.size(), 0),
          lostTriangles(s.mPatches.size(), 0),
          receivedTriangles(s.mPatches.size(), 0);
      std::vector<int> resultingPatch(triangles.size(), -1);
      for (int triangleId = 0; triangleId < int(triangles.size());
           ++triangleId) {
        int source = triangles[triangleId].PatchId;
        int target =
            destination[triangleId] >= 0 ? destination[triangleId] : source;
        patchTriangles[target].push_back(triangleId);
        resultingPatch[triangleId] = target;
        if (target != source) {
          changed[source] = 1;
          changed[target] = 1;
          lostTriangles[source] = 1;
          receivedTriangles[target] = 1;
        }
      }
      std::vector<MeshPatch> rebuilt;
      rebuilt.reserve(s.mPatches.size());
      std::vector<char> seen(triangles.size(), 0);
      bool acceptedBatch = true;
      for (int patchId = 0; patchId < int(s.mPatches.size()); ++patchId) {
        if (patchTriangles[patchId].empty())
          continue;
        std::vector<std::vector<int>> components;
        if (lostTriangles[patchId]) {
          for (int seed : patchTriangles[patchId])
            if (!seen[seed]) {
              components.emplace_back();
              std::queue<int> queue;
              queue.push(seed);
              seen[seed] = 1;
              while (!queue.empty()) {
                int triangleId = queue.front();
                queue.pop();
                components.back().push_back(triangleId);
                for (int neighbor : s.mMesh.getTriangleNeighbors(triangleId))
                  if (!seen[neighbor] && resultingPatch[neighbor] == patchId) {
                    seen[neighbor] = 1;
                    queue.push(neighbor);
                  }
              }
            }
        } else {
          components.push_back(std::move(patchTriangles[patchId]));
        }
        for (auto &component : components) {
          MeshPatch patch = s.mPatches[patchId];
          patch.TriangleIds = std::move(component);
          if (changed[patchId] &&
              !(patch.SurfaceType == PatchSurfaceType::Freeform &&
                patch.TriangleIds.size() > 100000))
            ApplyFit(patch, Fit(s.mMesh, patch, s.mConfig));
          if (receivedTriangles[patchId] &&
              (!FitsFixedTarget(s.mMesh, patch, s.mPatches[patchId]) ||
               !FitsFixedTarget(s.mMesh, patch, patch)))
            acceptedBatch = false;
          rebuilt.push_back(std::move(patch));
        }
      }
      if (!acceptedBatch) {
        totalMoved -= moved;
        if (s.mConfig.Verbose)
          std::clog << "[CadMesh] face closure batch rejected by final surface "
                       "validation\n";
        break;
      }
      s.mPatches = std::move(rebuilt);
      if (s.mConfig.Verbose)
        std::clog << "[CadMesh] face closure pass " << pass
                  << ": moved=" << moved << ", patches=" << s.mPatches.size()
                  << '\n';
    }
    if (s.mConfig.Verbose && totalMoved)
      std::clog << "[CadMesh] face closure: moved " << totalMoved
                << " triangles in " << pass - 1 << " passes\n";
  }
  // A small connected CAD face is legitimate. Do not force ownership by
  // triangle count: guarded geometric merges below may absorb actual noise.
  if (int(s.mPatches.size()) > s.mConfig.MaximumExhaustiveRefinementPatches ||
      int(s.mMesh.getTriangles().size()) >
          s.mConfig.MaximumExhaustiveRefinementTriangles) {
    if (s.mConfig.Verbose)
      std::clog << "[CadMesh] refinement guard: " << s.mPatches.size()
                << " patches, " << s.mMesh.getTriangles().size()
                << " triangles exceed an exhaustive refinement limit; "
                   "split/merge skipped\n";
    return;
  }
  // Split an under-segmented region only when an internal high-score chain
  // truly divides it.
  bool split = true;
  int splitPass = 0;
  while (split && splitPass++ < 8) {
    split = false;
    s.AssignPatchIds();
    for (int pi = 0; pi < int(s.mPatches.size()) && !split; ++pi) {
      const auto &original = s.mPatches[pi];
      if (original.TriangleIds.size() <
          size_t(2 * s.mConfig.MinimumPatchTriangles))
        continue;
      std::unordered_set<int> membership(original.TriangleIds.begin(),
                                         original.TriangleIds.end());
      std::unordered_set<int> cut;
      int highEdges = 0;
      for (int t : original.TriangleIds)
        for (int e : s.mMesh.getTriangles()[t].EdgeIds) {
          const auto &edge = s.mMesh.getEdges()[e];
          if (edge.Triangle0 >= 0 && edge.Triangle1 >= 0 &&
              membership.count(edge.Triangle0) &&
              membership.count(edge.Triangle1) &&
              edge.BoundaryScore > s.mConfig.StrongBoundaryThreshold) {
            cut.insert(e);
            ++highEdges;
          }
        }
      if (highEdges < 2)
        continue;
      std::unordered_set<int> seen;
      seen.reserve(original.TriangleIds.size());
      std::vector<std::vector<int>> components;
      for (int seed : original.TriangleIds)
        if (!seen.count(seed)) {
          components.emplace_back();
          std::queue<int> q;
          q.push(seed);
          seen.insert(seed);
          while (!q.empty()) {
            int t = q.front();
            q.pop();
            components.back().push_back(t);
            for (int e : s.mMesh.getTriangles()[t].EdgeIds)
              if (!cut.count(e))
                for (int n : s.mMesh.getEdges()[e].IncidentTriangleIds)
                  if (membership.count(n) && seen.insert(n).second)
                    q.push(n);
          }
        }
      // Keep every component. Discarding a small component loses its patch
      // membership while leaving a stale triangle label in the mesh.
      if (components.size() < 2)
        continue;
      s.mPatches.erase(s.mPatches.begin() + pi);
      for (auto &ids : components) {
        MeshPatch p;
        p.TriangleIds = std::move(ids);
        ApplyFit(p, Fit(s.mMesh, p, s.mConfig));
        s.mPatches.push_back(std::move(p));
      }
      split = true;
    }
  }
  // A one/two-triangle sliver is not a geometric patch: attach it to the best
  // adjacent region.
  bool absorbed = true;
  while (absorbed) {
    absorbed = false;
    s.AssignPatchIds();
    auto pairs = PatchPairs(s.mMesh);
    for (int small = 0; small < int(s.mPatches.size()); ++small)
      if (s.mPatches[small].TriangleIds.size() <=
              size_t(s.mConfig.MinimumPatchTriangles) &&
          s.mPatches[small].Confidence < .8) {
        int best = -1;
        double bestCost = 1e100;
        SurfaceFitResult bestFit;
        for (const auto &item : pairs) {
          int other = -1;
          if (item.first.first == small)
            other = item.first.second;
          else if (item.first.second == small)
            other = item.first.first;
          if (other < 0 ||
              ProtectedInterface(s.mMesh, item.second, s.mConfig) ||
              !FitsFixedTarget(s.mMesh, s.mPatches[small], s.mPatches[other]) ||
              AverageBoundary(s.mMesh, item.second) >=
                  s.mConfig.StrongBoundaryThreshold)
            continue;
          MeshPatch combined;
          combined.TriangleIds =
              UnionTriangles(s.mPatches[small], s.mPatches[other]);
          auto fit = Fit(s.mMesh, combined, s.mConfig);
          if (!AcceptCombined(s.mMesh, s.mPatches[small], s.mPatches[other],
                              fit, s.mConfig))
            continue;
          double cost = fit.Score + 2 * AverageBoundary(s.mMesh, item.second);
          if (cost < bestCost) {
            bestCost = cost;
            best = other;
            bestFit = fit;
          }
        }
        if (best >= 0) {
          MeshPatch combined;
          combined.TriangleIds =
              UnionTriangles(s.mPatches[small], s.mPatches[best]);
          ApplyFit(combined, bestFit);
          int erase = small;
          s.mPatches[best] = std::move(combined);
          s.mPatches.erase(s.mPatches.begin() + erase);
          absorbed = true;
          break;
        }
      }
  }
  // Final conservative analytic merge removes tessellation seams on one
  // underlying CAD surface.
  bool merged = true;
  while (merged) {
    merged = false;
    s.AssignPatchIds();
    auto pairMap = PatchPairs(s.mMesh);
    std::vector<std::pair<int, int>> order;
    for (const auto &item : pairMap)
      order.push_back(item.first);
    std::sort(order.begin(), order.end(), [&](const auto &x, const auto &y) {
      return s.mPatches[x.first].TriangleIds.size() +
                 s.mPatches[x.second].TriangleIds.size() >
             s.mPatches[y.first].TriangleIds.size() +
                 s.mPatches[y.second].TriangleIds.size();
    });
    for (const auto &pair : order) {
      int a = pair.first, b = pair.second;
      if (ProtectedInterface(s.mMesh, pairMap.at(pair), s.mConfig))
        continue;
      MeshPatch combined;
      combined.TriangleIds = UnionTriangles(s.mPatches[a], s.mPatches[b]);
      auto fit = Fit(s.mMesh, combined, s.mConfig);
      const auto &r = s.mMesh.getResolution();
      bool sameSurface = SimilarAnalytic(s.mPatches[a], s.mPatches[b], r);
      bool exactContinuation =
          fit.Type != PatchSurfaceType::Freeform &&
          fit.Type != PatchSurfaceType::Unknown &&
          (fit.Type == s.mPatches[a].SurfaceType ||
           fit.Type == s.mPatches[b].SurfaceType) &&
          fit.Rms < 2 * r.FittingTolerance &&
          fit.Max < 5 * r.FittingTolerance &&
          fit.Normal < std::max(4 * r.AngularTolerance, .04);
      if ((sameSurface || exactContinuation) &&
          AcceptCombined(s.mMesh, s.mPatches[a], s.mPatches[b], fit,
                         s.mConfig)) {
        ApplyFit(combined, fit);
        s.mPatches[a] = std::move(combined);
        s.mPatches.erase(s.mPatches.begin() + b);
        merged = true;
        break;
      }
    }
  }
}

} // namespace CadMesh
