#include "CadMesh/ModelFirstPartitioner.h"
#include "CadMesh/CudaAnalyticFitting.h"
#include "CadMesh/CudaAnalyticPrefetch.h"
#include "CadMesh/SegmentationGuards.h"
#include "CadMesh/SurfaceFitting.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>

namespace CadMesh {
namespace {
struct Cell {
  std::vector<int> Faces, Neighbors;
  double Area = 0;
};
class ModelPartition {
  MeshTopology &Mesh;
  const SegmentationConfig &Config;
  std::vector<std::array<int, 3>> Neighbors;
  std::vector<int> Owner, CellId, Visit, CellVisit;
  std::vector<Cell> Cells;
  std::vector<unsigned char> Seeded;
  std::vector<MeshPatch> Patches;
  int Stamp = 0;
  // Larger windows speculate on more neighborhoods that subsequent commits
  // or suppression may invalidate; keep enough work to fill the GPU without
  // changing the sequential proposal schedule.
  static constexpr size_t AnalyticLookahead = 8;
  static constexpr size_t MaximumPrefetchSeeds = 256;

  struct RemainingProbe {
    std::vector<int> OrderedFaces;
  };
  // Negative results only: successful probes immediately acquire ownership.
  // The sorted key identifies the full component, while OrderedFaces preserves
  // the fitter's accumulation order and ResolvedCurvedSupport's sample order.
  std::map<std::vector<int>, RemainingProbe> RemainingFailures;
  size_t RemainingCachedFaces = 0;
  size_t RemainingProbeCalls = 0, RemainingOversized = 0;
  size_t RemainingRepeatedSets = 0, RemainingOrderChanges = 0;
  size_t RemainingExactHits = 0, RemainingFits = 0;
  double RemainingFitSeconds = 0;
  std::vector<size_t> RemainingOversizedAt;
  size_t RemainingProbeEpoch = 1, RemainingProbeBudget = 0;
  size_t RemainingOversizedReuses = 0, RemainingTraversedFaces = 0;
  double RemainingTraversalSeconds = 0;

  void InvalidateRemainingSizeEvidence() {
    if (RemainingProbeEpoch == std::numeric_limits<size_t>::max()) {
      std::fill(RemainingOversizedAt.begin(), RemainingOversizedAt.end(), 0);
      RemainingProbeEpoch = 1;
    } else {
      ++RemainingProbeEpoch;
    }
  }

  void LogRemainingProbes(const char *stage) const {
    if (!Config.Verbose) return;
    std::clog << "[CadMesh] exposed component probes cumulative after " << stage
              << ": probes=" << RemainingProbeCalls
              << ", over_budget=" << RemainingOversized
              << ", over_budget_reuses=" << RemainingOversizedReuses
              << ", traversed_faces=" << RemainingTraversedFaces
              << ", traversal_seconds=" << RemainingTraversalSeconds
              << ", repeated_face_sets=" << RemainingRepeatedSets
              << ", different_order=" << RemainingOrderChanges
              << ", exact_failure_reuses=" << RemainingExactHits
              << ", selector_calls=" << RemainingFits
              << ", selector_seconds=" << RemainingFitSeconds
              << ", cached_components=" << RemainingFailures.size() << '\n';
  }

  // A cache belongs to one immutable surface snapshot and one candidate
  // evaluation. Mesh geometry is unchanged during partitioning. Ownership and
  // threshold decisions are deliberately never cached.
  class CompatibilityCache {
    const MeshTopology &Mesh;
    const MeshPatch Surface;
    std::unordered_map<int, SurfaceCompatibility> Values;
  public:
    CompatibilityCache(const MeshTopology &mesh, const MeshPatch &surface)
        : Mesh(mesh), Surface(surface) {}
    SurfaceCompatibility evaluate(int face) {
      const auto found = Values.find(face);
      if (found != Values.end()) return found->second;
      auto value = EvaluateSurfaceCompatibility(Mesh, face, Surface);
      if (Values.size() < 65536) Values.emplace(face, value);
      return value;
    }
    SurfaceCompatibility evaluate(const std::vector<int> &faces) {
      SurfaceCompatibility result;
      for (int face : faces) {
        const auto value = evaluate(face);
        result.MaxNormalizedDistance = std::max(result.MaxNormalizedDistance,
                                                value.MaxNormalizedDistance);
        result.MaxNormalError = std::max(result.MaxNormalError, value.MaxNormalError);
        result.ReliableNormalSamples += value.ReliableNormalSamples;
        if (!value.Supported) return result;
      }
      result.Supported = !faces.empty();
      return result;
    }
  };

  MeshPatch FromFit(const SurfaceFitResult &fit) const {
    MeshPatch p;
    p.SurfaceType = fit.Type;
    p.Parameters = fit.Parameters;
    p.RmsFittingError = fit.Rms;
    p.MaxFittingError = fit.Max;
    p.NormalError = fit.Normal;
    p.Confidence =
        fit.Type == PatchSurfaceType::Freeform
            ? .35
            : 1.0 /
                  (1 + fit.Rms / std::max(Mesh.getResolution().FittingTolerance,
                                          1e-30));
    return p;
  }
  std::vector<int> Sample(const std::vector<int> &faces,
                          size_t maximum = 256) const {
    if (faces.size() <= maximum)
      return faces;
    // Equal inclusion probability; fitters apply physical triangle area once.
    // A hard budget bounds unsuccessful local proposal work.
    std::vector<int> result;
    for (size_t i = 0; i < maximum; ++i)
      result.push_back(faces[(2 * i + 1) * faces.size() / (2 * maximum)]);
    return result;
  }
  bool Compatible(int f, const MeshPatch &p, double distanceRatio = -1,
                  CompatibilityCache *cache = nullptr) const {
    const auto check = cache ? cache->evaluate(f) : EvaluateSurfaceCompatibility(Mesh, f, p);
    return check.Supported && check.ReliableNormalSamples > 0 &&
           check.MaxNormalizedDistance <= (distanceRatio < 0
                                               ? Config.ModelFitToleranceRatio
                                               : distanceRatio) &&
           check.MaxNormalError <= Config.ModelNormalTolerance;
  }
  bool Certify(const std::vector<int> &faces, MeshPatch &p,
               CompatibilityCache *cache = nullptr) const {
    if (p.SurfaceType == PatchSurfaceType::Unknown ||
        p.SurfaceType == PatchSurfaceType::Freeform)
      return false;
    const auto check = cache ? cache->evaluate(faces) : EvaluateSurfaceCompatibility(Mesh, faces, p);
    if (!check.Supported ||
        check.MaxNormalizedDistance > Config.ModelFitToleranceRatio ||
        check.MaxNormalError > Config.ModelNormalTolerance)
      return false;
    p.MaxFittingError =
        check.MaxNormalizedDistance * Mesh.getResolution().FittingTolerance;
    return true;
  }
  void Store(std::vector<int> faces, MeshPatch patch) {
    if (patch.SurfaceType != PatchSurfaceType::Freeform) {
      double weight = 0, squaredDistance = 0, normalError = 0;
      patch.MaxFittingError = 0;
      patch.MaxSampledSurfaceDeviation = 0;
      for (int f : faces) {
        const auto &t = Mesh.getTriangles()[f];
        double distance = 0;
        Vec3 normal;
        for (int k = 0; k < 3; ++k) {
          const auto a = ToVec(Mesh.getVertices()[t.VertexIds[k]].Position);
          const auto b =
              ToVec(Mesh.getVertices()[t.VertexIds[(k + 1) % 3]].Position);
          if (SegmentationGuardDetail::SurfaceSample(patch, a, distance,
                                                     normal)) {
            squaredDistance += t.Area / 3 * distance * distance;
            patch.MaxFittingError = std::max(patch.MaxFittingError, distance);
            patch.MaxSampledSurfaceDeviation =
                std::max(patch.MaxSampledSurfaceDeviation, distance);
          }
          if (SegmentationGuardDetail::SurfaceSample(patch, Mul(Add(a, b), .5),
                                                     distance, normal))
            patch.MaxSampledSurfaceDeviation =
                std::max(patch.MaxSampledSurfaceDeviation, distance);
        }
        if (SegmentationGuardDetail::SurfaceSample(patch, ToVec(t.Centroid),
                                                   distance, normal)) {
          normalError +=
              t.Area *
              std::acos(std::min(1.0, std::abs(Dot(normal, ToVec(t.Normal)))));
          patch.MaxSampledSurfaceDeviation =
              std::max(patch.MaxSampledSurfaceDeviation, distance);
        }
        weight += t.Area;
      }
      patch.RmsFittingError =
          std::sqrt(squaredDistance / std::max(weight, 1e-30));
      patch.NormalError = normalError / std::max(weight, 1e-30);
      patch.Confidence =
          1 / (1 + patch.RmsFittingError /
                       std::max(Mesh.getResolution().FittingTolerance, 1e-30));
    }
    patch.Id = int(Patches.size());
    patch.TriangleIds = std::move(faces);
    for (int f : patch.TriangleIds)
      Owner[f] = patch.Id;
    Patches.push_back(std::move(patch));
    // Any new ownership can split/shrink a residual component. Its previous
    // size lower bound must not suppress a now-small component.
    InvalidateRemainingSizeEvidence();
  }
  double GrowthTolerance(const MeshPatch &p) const {
    // Exact CAD models must not absorb the first tangent transition strip just
    // because the global mesh tolerance is larger than that strip's sagitta.
    const double measured =
        p.MaxFittingError /
        std::max(Mesh.getResolution().FittingTolerance, 1e-30);
    return std::min(Config.ModelFitToleranceRatio, std::max(.2, 3 * measured));
  }
  std::vector<int> Grow(int seed, const MeshPatch &p, CompatibilityCache *cache = nullptr) {
    ++Stamp;
    std::vector<int> result;
    Visit[seed] = Stamp;
    if (Owner[seed] >= 0 || !Compatible(seed, p, GrowthTolerance(p), cache))
      return result;
    result.push_back(seed);
    for (size_t i = 0; i < result.size(); ++i)
      for (int next : Neighbors[result[i]])
        if (next >= 0 && Owner[next] < 0 && Visit[next] != Stamp) {
          Visit[next] = Stamp;
          if (Compatible(next, p, GrowthTolerance(p), cache))
            result.push_back(next);
        }
    return result;
  }
  MeshPatch Fit(const std::vector<int> &faces) const {
    return FromFit(SurfaceModelSelector::fitBest(
        Mesh, faces, Mesh.getResolution(), Config.ModelComplexityPenalty));
  }
  template <class Fitter>
  MeshPatch FitOnly(const std::vector<int> &faces,
                    bool proposal = false) const {
    CudaAnalyticFitContext context(proposal ? "neighborhood proposal" : nullptr);
    Fitter fitter;
    if (!fitter.fit(Mesh, proposal ? Sample(faces, 128) : faces))
      return {};
    return FromFit({fitter.getType(), fitter.computeRmsError(),
                    fitter.computeMaxError(), fitter.computeNormalError(), 0,
                    fitter.getParameters()});
  }
  MeshPatch RefitType(const std::vector<int> &faces,
                      PatchSurfaceType type) const {
    switch (type) {
    case PatchSurfaceType::Plane:
      return FitOnly<PlaneSurfaceFitter>(faces);
    case PatchSurfaceType::Cylinder:
      return FitOnly<CylinderSurfaceFitter>(faces);
    case PatchSurfaceType::Cone:
      return FitOnly<ConeSurfaceFitter>(faces);
    case PatchSurfaceType::Sphere:
      return FitOnly<SphereSurfaceFitter>(faces);
    case PatchSurfaceType::Torus:
      return FitOnly<TorusSurfaceFitter>(faces);
    default:
      return {};
    }
  }
  bool ResolvedCurvedSupport(const std::vector<int> &faces) const {
    std::vector<Vec3> normals;
    for (int f : Sample(faces, 128)) {
      const Vec3 normal = ToVec(Mesh.getTriangles()[f].Normal);
      bool distinct = true;
      for (const auto &existing : normals)
        if (std::abs(Dot(existing, normal)) > std::cos(.008)) {
          distinct = false;
          break;
        }
      if (distinct)
        normals.push_back(normal);
      if (normals.size() >= 4)
        return true;
    }
    // Two flat panels can exactly interpolate a cylinder at their vertices.
    // That does not establish an observed continuously rotating normal field.
    return false;
  }
  void BuildNeighbors() {
    const auto &triangles = Mesh.getTriangles();
    Neighbors.assign(triangles.size(), {-1, -1, -1});
    for (int edgeId = 0; edgeId < int(Mesh.getEdges().size()); ++edgeId) {
      auto &edge = Mesh.getEdges()[edgeId];
      if (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature ||
          edge.IncidentTriangleIds.size() != 2)
        continue;
      const int a = edge.IncidentTriangleIds[0],
                b = edge.IncidentTriangleIds[1];
      const double cosine =
          std::abs(Dot(ToVec(triangles[a].Normal), ToVec(triangles[b].Normal)));
      if (cosine < std::cos(Config.ModelSharpAngle)) {
        edge.IsConstrainedFeature = true;
        continue;
      }
      for (int k = 0; k < 3; ++k) {
        if (triangles[a].EdgeIds[k] == edgeId)
          Neighbors[a][k] = b;
        if (triangles[b].EdgeIds[k] == edgeId)
          Neighbors[b][k] = a;
      }
    }
  }
  void WholeComponents() {
    std::vector<char> seen(Owner.size(), false);
    int components = 0;
    for (int seed = 0; seed < int(Owner.size()); ++seed) {
      if (seen[seed])
        continue;
      std::vector<int> faces{seed};
      seen[seed] = true;
      for (size_t i = 0; i < faces.size(); ++i)
        for (int n : Neighbors[faces[i]])
          if (n >= 0 && !seen[n]) {
            seen[n] = true;
            faces.push_back(n);
          }
      ++components;
      MeshPatch fit = Fit(faces);
      if ((fit.SurfaceType == PatchSurfaceType::Plane ||
           ResolvedCurvedSupport(faces)) &&
          Certify(faces, fit))
        Store(std::move(faces), fit);
    }
    if (Config.Verbose)
      std::clog << "[CadMesh] smooth components=" << components
                << ", certified=" << Patches.size() << '\n';
  }
  void BuildCells() {
    const auto &triangles = Mesh.getTriangles();
    for (int seed = 0; seed < int(Owner.size()); ++seed) {
      if (Owner[seed] >= 0 || CellId[seed] >= 0)
        continue;
      Cell cell;
      MeshPatch plane;
      plane.SurfaceType = PatchSurfaceType::Plane;
      plane.Parameters =
          PlaneParameters{{triangles[seed].Centroid, triangles[seed].Normal}};
      cell.Faces.push_back(seed);
      CellId[seed] = int(Cells.size());
      for (size_t i = 0; i < cell.Faces.size(); ++i) {
        const int f = cell.Faces[i];
        cell.Area += triangles[f].Area;
        for (int n : Neighbors[f]) {
          if (n < 0 || Owner[n] >= 0 || CellId[n] >= 0)
            continue;
          const double cosine = std::abs(
              Dot(ToVec(triangles[seed].Normal), ToVec(triangles[n].Normal)));
          if (cosine < std::cos(.012))
            continue;
          if (!Compatible(n, plane, 1.0))
            continue;
          CellId[n] = int(Cells.size());
          cell.Faces.push_back(n);
        }
      }
      Cells.push_back(std::move(cell));
    }
    for (int f = 0; f < int(Owner.size()); ++f)
      if (CellId[f] >= 0)
        for (int n : Neighbors[f])
          if (n >= 0 && CellId[n] >= 0 && CellId[n] != CellId[f])
            Cells[CellId[f]].Neighbors.push_back(CellId[n]);
    for (auto &cell : Cells) {
      auto &n = cell.Neighbors;
      std::sort(n.begin(), n.end());
      n.erase(std::unique(n.begin(), n.end()), n.end());
    }
    if (Config.Verbose)
      std::clog << "[CadMesh] provisional planar cells=" << Cells.size()
                << '\n';
  }
  std::vector<int> CellOrder() const {
    std::vector<int> order(Cells.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return Cells[a].Area > Cells[b].Area;
    });
    return order;
  }
  bool HasUnowned(int id) const {
    for (int f : Cells[id].Faces)
      if (Owner[f] < 0)
        return true;
    return false;
  }
  void ExtractPlaneCores() {
    const double areaScale = Mesh.getResolution().MedianEdgeLength *
                             Mesh.getResolution().MedianEdgeLength;
    for (int id : CellOrder()) {
      if (!HasUnowned(id))
        continue;
      const auto &cell = Cells[id];
      double neighborArea = 0;
      for (int n : cell.Neighbors)
        neighborArea += Cells[n].Area;
      double longestSquared = 0;
      for (int f : cell.Faces) {
        const auto &triangle = Mesh.getTriangles()[f];
        for (int k = 0; k < 3; ++k) {
          const double length = Distance(
              Mesh.getVertices()[triangle.VertexIds[k]].Position,
              Mesh.getVertices()[triangle.VertexIds[(k + 1) % 3]].Position);
          longestSquared = std::max(longestSquared, length * length);
        }
      }
      // A genuine mother plane dominates the small tessellation facets around
      // it. A cylindrical facet has peers of comparable area on both sides.
      // Small bevels are left to connected residual certification, not deleted.
      if (cell.Faces.size() < 2 || cell.Area < .02 * longestSquared ||
          cell.Area < 4 * areaScale || cell.Area < 1.5 * neighborArea)
        continue;
      std::vector<int> remaining;
      for (int f : cell.Faces)
        if (Owner[f] < 0)
          remaining.push_back(f);
      auto plane = FitOnly<PlaneSurfaceFitter>(remaining);
      if (!Certify(remaining, plane))
        continue;
      for (int seed : remaining)
        if (Owner[seed] < 0) {
          auto faces = Grow(seed, plane);
          if (!faces.empty())
            Store(std::move(faces), plane);
        }
    }
    if (Config.Verbose)
      std::clog << "[CadMesh] after mother planes=" << Patches.size() << '\n';
  }
  // Probe a remaining connected component without assigning or cutting it.
  // Small components exposed by removing a mother surface often are complete
  // fillets; they deserve one whole-model test before local primitives compete.
  bool TryRemainingComponent(int seed, size_t budget) {
    CudaAnalyticFitContext context("exposed component selector");
    if (Owner[seed] >= 0)
      return false;
    ++RemainingProbeCalls;
    const auto traversalStart = std::chrono::steady_clock::now();
    const auto finishTraversal = [&]() {
      RemainingTraversalSeconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - traversalStart).count();
    };
    if (RemainingProbeBudget != budget) {
      InvalidateRemainingSizeEvidence();
      RemainingProbeBudget = budget;
    }
    if (RemainingOversizedAt.empty()) RemainingOversizedAt.resize(Owner.size(), 0);
    if (RemainingOversizedAt[seed] == RemainingProbeEpoch) {
      ++RemainingOversized;
      ++RemainingOversizedReuses;
      finishTraversal();
      return false;
    }
    ++Stamp;
    std::vector<int> faces{seed};
    ++RemainingTraversedFaces;
    Visit[seed] = Stamp;
    for (size_t i = 0; i < faces.size(); ++i)
      for (int n : Neighbors[faces[i]])
        if (n >= 0 && Owner[n] < 0 && Visit[n] != Stamp) {
          Visit[n] = Stamp;
          faces.push_back(n);
          ++RemainingTraversedFaces;
          const bool knownOversized = RemainingOversizedAt[n] == RemainingProbeEpoch;
          if (knownOversized || faces.size() > budget) {
            // Every discovered face is connected to the seed through current
            // unowned faces. Reaching a marked face proves the same size lower
            // bound even when this traversal has not yet hit the budget.
            for (int face : faces) RemainingOversizedAt[face] = RemainingProbeEpoch;
            ++RemainingOversized;
            RemainingOversizedReuses += knownOversized;
            finishTraversal();
            return false;
          }
        }
    finishTraversal();
    // Always rediscover the entire current component before consulting the
    // cache. Acquiring any of its faces changes this key and forces a refit.
    // An incomplete traversal stopped by the face budget is never cached.
    auto key = faces;
    std::sort(key.begin(), key.end());
    auto previous = RemainingFailures.find(key);
    if (previous != RemainingFailures.end()) {
      ++RemainingRepeatedSets;
      if (previous->second.OrderedFaces == faces) {
        ++RemainingExactHits;
        return false;
      }
      ++RemainingOrderChanges;
    }
    ++RemainingFits;
    const auto fitStart = std::chrono::steady_clock::now();
    MeshPatch model = Fit(faces);
    RemainingFitSeconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fitStart).count();
    if ((model.SurfaceType != PatchSurfaceType::Plane &&
         !ResolvedCurvedSupport(faces)) ||
        !Certify(faces, model)) {
      // Cache only the failed probe, never a patch id or ownership decision.
      // Bound both component count and retained face ids (key + input order).
      if (previous != RemainingFailures.end()) {
        previous->second.OrderedFaces = faces;
      } else {
        constexpr size_t maximumComponents = 1024;
        constexpr size_t maximumFaceIds = 1024 * 1024;
        const size_t needed = 2 * faces.size();
        if (needed <= maximumFaceIds) {
          while (!RemainingFailures.empty() &&
                 (RemainingFailures.size() >= maximumComponents ||
                  RemainingCachedFaces + needed > maximumFaceIds)) {
            const auto oldestKey = RemainingFailures.begin();
            RemainingCachedFaces -= 2 * oldestKey->first.size();
            RemainingFailures.erase(oldestKey);
          }
          RemainingFailures.emplace(std::move(key), RemainingProbe{faces});
          RemainingCachedFaces += needed;
        }
      }
      return false;
    }
    if (previous != RemainingFailures.end()) {
      RemainingCachedFaces -= 2 * previous->first.size();
      RemainingFailures.erase(previous);
    }
    Store(std::move(faces), model);
    return true;
  }
  struct Neighborhood {
    std::vector<int> Cells, Faces;
  };
  Neighborhood SeedNeighborhood(int id, int depth, size_t maximumCells,
                                bool comparableArea = false) {
    Neighborhood out;
    ++Stamp;
    out.Cells.push_back(id);
    CellVisit[id] = Stamp;
    size_t begin = 0;
    const auto center = Mesh.getTriangles()[Cells[id].Faces[0]].Centroid;
    double radius =
        Config.ModelSeedRadiusFactor * Mesh.getResolution().MedianEdgeLength;
    for (int f : Cells[id].Faces)
      for (int v : Mesh.getTriangles()[f].VertexIds)
        radius = std::max(
            radius, 1.5 * Distance(center, Mesh.getVertices()[v].Position));
    radius *= std::max(1.0, depth / 4.0);
    for (int ring = 0; ring < depth; ++ring) {
      const size_t end = out.Cells.size();
      for (size_t k = begin; k < end && out.Cells.size() < maximumCells; ++k)
        for (int n : Cells[out.Cells[k]].Neighbors) {
          if (CellVisit[n] == Stamp || !HasUnowned(n))
            continue;
          CellVisit[n] = Stamp;
          if (comparableArea && Cells[n].Area < .2 * Cells[id].Area)
            continue;
          if (Distance(center,
                       Mesh.getTriangles()[Cells[n].Faces[0]].Centroid) >
              radius)
            continue;
          out.Cells.push_back(n);
          if (out.Cells.size() >= maximumCells)
            break;
        }
      begin = end;
    }
    for (int c : out.Cells)
      for (int f : Cells[c].Faces)
        if (Owner[f] < 0)
          out.Faces.push_back(f);
    return out;
  }
  struct Candidate {
    MeshPatch Model;
    std::vector<int> Faces;
    double Area = 0;
  };
  void CollectProposalModels(const Neighborhood &peers,
                             const Neighborhood &broad) const {
    for (const auto *neighborhood : {&peers, &broad}) {
      const auto &support = neighborhood->Faces;
      if (support.size() < 6)
        continue;
      auto plane = FitOnly<PlaneSurfaceFitter>(support, true);
      if (Certify(support, plane))
        continue;
      // These fits are independent until Consider/CommitCandidate runs. Stop
      // each fitter at its nonlinear solve and submit all initial guesses
      // together. The ordinary path below still evaluates every candidate in
      // its original order, reusing only byte-identical solver inputs.
      const auto collect = [&](auto fitter) {
        try {
          fitter.fit(Mesh, Sample(support, 128));
        } catch (const AnalyticSeedDeferred &) {
        }
      };
      collect(CylinderSurfaceFitter{});
      collect(ConeSurfaceFitter{});
      collect(SphereSurfaceFitter{});
      if (support.size() >= 20)
        collect(TorusSurfaceFitter{});
    }
  }
  void PrepareProposalModels(CudaAnalyticSeedPrefetch &prepared,
                             const Neighborhood &peers,
                             const Neighborhood &broad) const {
    if (!prepared.enabled())
      return;
    prepared.begin();
    CollectProposalModels(peers, broad);
    prepared.flush();
  }
  bool HasGeometricSupport(const std::vector<int> &faces) const {
    std::unordered_map<int, double> areas;
    double total = 0, largest = 0;
    for (int f : faces) {
      const double area = Mesh.getTriangles()[f].Area;
      total += area;
      if (CellId[f] >= 0)
        areas[CellId[f]] += area;
    }
    for (const auto &item : areas)
      largest = std::max(largest, item.second);
    return areas.size() >= 4 && largest <= .8 * total;
  }
  void Consider(int seed, const std::vector<int> &support, MeshPatch model,
                Candidate &best) {
    CompatibilityCache cache(Mesh, model);
    if (model.SurfaceType == PatchSurfaceType::Unknown ||
        !Compatible(seed, model, -1, &cache))
      return;
    // Reject primitives that interpolate just one thin band of the
    // neighborhood. Models compete on a physical support region, not the first
    // fitter's type.
    double supported = 0, total = 0;
    for (int f : support) {
      double area = Mesh.getTriangles()[f].Area;
      total += area;
      if (Compatible(f, model, -1, &cache))
        supported += area;
    }
    if (supported < .8 * total)
      return;
    auto faces = Grow(seed, model, &cache);
    if (faces.size() < 6 || !HasGeometricSupport(faces))
      return;
    double area = 0;
    for (int f : faces)
      area += Mesh.getTriangles()[f].Area;
    if (area > best.Area * (1 + 1e-8))
      best = {std::move(model), std::move(faces), area};
  }
  bool CommitCandidate(int seed, Candidate candidate) {
    CudaAnalyticFitContext context("candidate full refit");
    if (candidate.Faces.empty())
      return false;
    MeshPatch refined;
    switch (candidate.Model.SurfaceType) {
    case PatchSurfaceType::Cylinder:
      refined = FitOnly<CylinderSurfaceFitter>(candidate.Faces);
      break;
    case PatchSurfaceType::Cone:
      refined = FitOnly<ConeSurfaceFitter>(candidate.Faces);
      break;
    case PatchSurfaceType::Sphere:
      refined = FitOnly<SphereSurfaceFitter>(candidate.Faces);
      break;
    case PatchSurfaceType::Torus:
      refined = FitOnly<TorusSurfaceFitter>(candidate.Faces);
      break;
    default:
      return false;
    }
    CompatibilityCache cache(Mesh, refined);
    if (!Certify(candidate.Faces, refined, &cache))
      return false;
    auto expanded = Grow(seed, refined, &cache);
    if (expanded.size() >= candidate.Faces.size() && Certify(expanded, refined, &cache))
      candidate.Faces = std::move(expanded);
    std::vector<int> exposed;
    for (int f : candidate.Faces)
      for (int n : Neighbors[f])
        if (n >= 0 && Owner[n] < 0)
          exposed.push_back(n);
    Store(std::move(candidate.Faces), refined);
    // Store starts a fresh epoch. Share size evidence only among the following
    // three probes; any successful probe calls Store and invalidates it again.
    // Avoid repeatedly scanning a huge residual component from every boundary
    // vertex. At most a few newly exposed components receive this early test.
    int probes = 0;
    for (int n : exposed)
      if (Owner[n] < 0) {
        TryRemainingComponent(n, 4096);
        if (++probes >= 3)
          break;
      }
    return true;
  }
  void ExtractCurves() {
    CudaAnalyticSeedPrefetch prepared("analytic neighborhoods", Config.Verbose);
    CellVisit.assign(Cells.size(), 0);
    Seeded.assign(Cells.size(), 0);
    std::vector<unsigned char> attempted(Cells.size(), 0);
    const size_t maximumSeeds = size_t(std::max(0, Config.ModelMaximumSeeds));
    size_t proposals = 0;
    bool budgetReached = false;
    const auto cellOrder = CellOrder();
    size_t prefetchedThrough = 0;
    for (size_t position = 0; position < cellOrder.size(); ++position) {
      const int id = cellOrder[position];
      int seed = -1;
      for (int f : Cells[id].Faces)
        if (Owner[f] < 0) {
          seed = f;
          break;
        }
      if (seed < 0 || attempted[id] || Cells[id].Neighbors.empty())
        continue;
      if (proposals >= maximumSeeds) {
        budgetReached = true;
        break;
      }
      if (prepared.enabled() && position >= prefetchedThrough) {
        // Only solver inputs are speculative. Ownership, suppression, proposal
        // counts and candidate competition continue in their original order.
        prepared.begin();
        size_t collected = 0, cursor = position;
        for (; cursor < cellOrder.size() && collected < AnalyticLookahead &&
               collected < maximumSeeds - proposals &&
               prepared.pendingSeeds() < MaximumPrefetchSeeds; ++cursor) {
          const int next = cellOrder[cursor];
          if (attempted[next] || Cells[next].Neighbors.empty() || !HasUnowned(next))
            continue;
          auto peers = SeedNeighborhood(next, 4, 48, true);
          auto broad = SeedNeighborhood(next, 10, 256);
          CollectProposalModels(peers, broad);
          ++collected;
        }
        prefetchedThrough = cursor;
        prepared.flush();
      }
      ++proposals;
      Seeded[id] = 1;
      if (Config.Verbose && proposals % 1000 == 0)
        std::clog << "[CadMesh] analytic seeds=" << proposals
                  << ", patches=" << Patches.size() << '\n';
      Candidate best;
      // Peer facets recover ruled mothers without mixing their tiny tangent
      // transition triangles. Broad neighborhoods then allow torus competition.
      auto peers = SeedNeighborhood(id, 4, 48, true);
      auto broad = SeedNeighborhood(id, 10, 256);
      PrepareProposalModels(prepared, peers, broad);
      for (const auto *neighborhood : {&peers, &broad}) {
        const auto &support = neighborhood->Faces;
        if (support.size() < 6)
          continue;
        auto plane = FitOnly<PlaneSurfaceFitter>(support, true);
        if (Certify(support, plane))
          continue;
        Consider(seed, support, FitOnly<CylinderSurfaceFitter>(support, true),
                 best);
        Consider(seed, support, FitOnly<ConeSurfaceFitter>(support, true),
                 best);
        Consider(seed, support, FitOnly<SphereSurfaceFitter>(support, true),
                 best);
        if (support.size() >= 20)
          Consider(seed, support, FitOnly<TorusSurfaceFitter>(support, true),
                   best);
      }
      const bool committed = CommitCandidate(seed, std::move(best));
      attempted[id] = 1;
      // Suppression is a proposal scheduling choice. It never assigns a label
      // or creates a remesh edge; every skipped face remains in the residual.
      if (!committed) {
        const size_t suppress = std::min<size_t>(
            broad.Cells.size(), Cells.size() > 10000 ? 64 : 12);
        for (size_t i = 0; i < suppress; ++i)
          attempted[broad.Cells[i]] = 1;
      }
    }
    if (Config.Verbose)
      std::clog << "[CadMesh] analytic seeds=" << proposals
                << ", after analytic growth=" << Patches.size() << '\n';
    if (Config.Verbose && budgetReached)
      std::clog << "[CadMesh] analytic seed budget reached (" << maximumSeeds
                << "); continuing with spatial residual reserve\n";
  }
  void DiscoverResidualModels() {
    CudaAnalyticSeedPrefetch prepared("residual neighborhoods", Config.Verbose);
    const size_t budget = size_t(std::max(0, Config.ModelResidualSeedBudget));
    if (!budget)
      return;
    struct Component {
      std::vector<int> Faces;
      Vec3 Low{}, High{};
    };
    std::vector<Component> components;
    std::vector<int> componentId(Owner.size(), -1);
    for (int seed = 0; seed < int(Owner.size()); ++seed) {
      if (Owner[seed] >= 0 || componentId[seed] >= 0)
        continue;
      Component component;
      const int id = int(components.size());
      component.Faces = {seed};
      component.Low = component.High =
          ToVec(Mesh.getTriangles()[seed].Centroid);
      componentId[seed] = id;
      for (size_t i = 0; i < component.Faces.size(); ++i) {
        const int f = component.Faces[i];
        const auto center = ToVec(Mesh.getTriangles()[f].Centroid);
        for (int k = 0; k < 3; ++k) {
          component.Low[k] = std::min(component.Low[k], center[k]);
          component.High[k] = std::max(component.High[k], center[k]);
        }
        for (int next : Neighbors[f])
          if (next >= 0 && Owner[next] < 0 && componentId[next] < 0) {
            componentId[next] = id;
            component.Faces.push_back(next);
          }
      }
      components.push_back(std::move(component));
    }
    std::vector<std::vector<int>> componentSamples;
    componentSamples.reserve(components.size());
    for (const auto &component : components)
      componentSamples.push_back(Sample(component.Faces, 256));
    const auto componentFits = SurfaceModelSelector::fitBestBatch(
        Mesh, componentSamples, Mesh.getResolution(), Config.ModelComplexityPenalty);
    for (size_t i = 0; i < components.size(); ++i) {
      auto &component = components[i];
      // A complete exposed surface gets a cheap bounded proposal and then a
      // full refit/certificate before local searches spend their reserve.
      auto model = FromFit(componentFits[i]);
      if ((model.SurfaceType == PatchSurfaceType::Plane ||
           ResolvedCurvedSupport(component.Faces)) &&
          Certify(component.Faces, model)) {
        CudaAnalyticFitContext context("residual component full refit");
        model = RefitType(component.Faces, model.SurfaceType);
        if (Certify(component.Faces, model))
          Store(component.Faces, model);
      }
    }
    using BucketKey = std::array<int, 3>;
    std::vector<std::map<BucketKey, std::vector<int>>> grids(components.size());
    for (int id : CellOrder()) {
      if (Seeded[id] || Cells[id].Neighbors.empty())
        continue;
      int seed = -1;
      for (int f : Cells[id].Faces)
        if (Owner[f] < 0) {
          seed = f;
          break;
        }
      if (seed < 0)
        continue;
      const int c = componentId[seed];
      if (c < 0 || components[c].Faces.size() < 12)
        continue;
      const auto &component = components[c];
      const auto center = ToVec(Mesh.getTriangles()[seed].Centroid);
      const double span =
          std::max({component.High[0] - component.Low[0],
                    component.High[1] - component.Low[1],
                    component.High[2] - component.Low[2], 1e-30});
      BucketKey key;
      for (int k = 0; k < 3; ++k)
        key[k] = std::max(
            0, std::min(7, int(8 * (center[k] - component.Low[k]) / span)));
      grids[c][key].push_back(id);
    }
    struct Schedule {
      std::vector<std::vector<int>> Buckets;
      std::vector<size_t> Cursor;
      size_t Next = 0;
      int Pop(std::vector<size_t> &cursor, size_t &next) const {
        for (size_t trial = 0; trial < Buckets.size(); ++trial) {
          const size_t bucket = next++ % Buckets.size();
          if (cursor[bucket] < Buckets[bucket].size())
            return Buckets[bucket][cursor[bucket]++];
        }
        return -1;
      }
      int Pop() { return Pop(Cursor, Next); }
    };
    std::vector<Schedule> schedules;
    for (auto &grid : grids) {
      if (grid.empty())
        continue;
      Schedule schedule;
      for (auto &bucket : grid)
        schedule.Buckets.push_back(std::move(bucket.second));
      schedule.Cursor.assign(schedule.Buckets.size(), 0);
      schedules.push_back(std::move(schedule));
    }
    size_t proposals = 0, committed = 0;
    size_t prefetchedThrough = 0;
    bool available = true;
    while (proposals < budget && available) {
      available = false;
      for (size_t scheduleIndex = 0; scheduleIndex < schedules.size(); ++scheduleIndex) {
        auto &schedule = schedules[scheduleIndex];
        if (prepared.enabled() && proposals >= prefetchedThrough) {
          // Copy only the cursors, not the large immutable cell buckets. The
          // true schedule is advanced exclusively by the ordinary Pop below.
          std::vector<std::vector<size_t>> cursors;
          std::vector<size_t> next;
          for (const auto &source : schedules) {
            cursors.push_back(source.Cursor);
            next.push_back(source.Next);
          }
          prepared.begin();
          size_t collected = 0, empty = 0, scan = scheduleIndex;
          while (collected < AnalyticLookahead && collected < budget - proposals &&
                 prepared.pendingSeeds() < MaximumPrefetchSeeds && empty < schedules.size()) {
            const size_t index = scan++ % schedules.size();
            int candidate = schedules[index].Pop(cursors[index], next[index]);
            while (candidate >= 0 && !HasUnowned(candidate))
              candidate = schedules[index].Pop(cursors[index], next[index]);
            if (candidate < 0) {
              ++empty;
              continue;
            }
            empty = 0;
            auto peers = SeedNeighborhood(candidate, 4, 48, true);
            auto broad = SeedNeighborhood(candidate, 10, 256);
            CollectProposalModels(peers, broad);
            ++collected;
          }
          prefetchedThrough = proposals + std::max<size_t>(1, collected);
          prepared.flush();
        }
        int id = schedule.Pop();
        while (id >= 0 && !HasUnowned(id))
          id = schedule.Pop();
        if (id < 0)
          continue;
        available = true;
        Seeded[id] = 1;
        int seed = -1;
        for (int f : Cells[id].Faces)
          if (Owner[f] < 0) {
            seed = f;
            break;
          }
        ++proposals;
        Candidate best;
        auto peers = SeedNeighborhood(id, 4, 48, true);
        auto broad = SeedNeighborhood(id, 10, 256);
        PrepareProposalModels(prepared, peers, broad);
        for (const auto *neighborhood : {&peers, &broad}) {
          const auto &support = neighborhood->Faces;
          if (support.size() < 6)
            continue;
          auto plane = FitOnly<PlaneSurfaceFitter>(support, true);
          if (Certify(support, plane))
            continue;
          Consider(seed, support, FitOnly<CylinderSurfaceFitter>(support, true),
                   best);
          Consider(seed, support, FitOnly<ConeSurfaceFitter>(support, true),
                   best);
          Consider(seed, support, FitOnly<SphereSurfaceFitter>(support, true),
                   best);
          if (support.size() >= 20)
            Consider(seed, support, FitOnly<TorusSurfaceFitter>(support, true),
                     best);
        }
        committed += CommitCandidate(seed, std::move(best));
        if (proposals >= budget)
          break;
      }
    }
    if (Config.Verbose)
      std::clog << "[CadMesh] residual reserve seeds=" << proposals << '/'
                << budget << ", spatial components=" << schedules.size()
                << ", committed=" << committed << '\n';
  }
  bool SplitSupportedPlanarComponent(const std::vector<int> &faces) {
    std::map<int, std::vector<int>> groups;
    double totalArea = 0;
    for (int f : faces) {
      groups[CellId[f]].push_back(f);
      totalArea += Mesh.getTriangles()[f].Area;
      if (groups.size() > 3)
        return false;
    }
    if (groups.size() < 2 || groups.begin()->first < 0)
      return false;
    std::vector<MeshPatch> planes;
    for (const auto &entry : groups) {
      if (entry.second.size() < 2)
        return false;
      double area = 0, longestSquared = 0;
      for (int f : entry.second) {
        const auto &triangle = Mesh.getTriangles()[f];
        area += triangle.Area;
        for (int k = 0; k < 3; ++k) {
          const double length = Distance(
              Mesh.getVertices()[triangle.VertexIds[k]].Position,
              Mesh.getVertices()[triangle.VertexIds[(k + 1) % 3]].Position);
          longestSquared = std::max(longestSquared, length * length);
        }
      }
      if (area < .15 * totalArea || area < .03 * longestSquared)
        return false;
      auto plane = FitOnly<PlaneSurfaceFitter>(entry.second);
      if (!Certify(entry.second, plane) ||
          plane.MaxFittingError > .1 * Mesh.getResolution().FittingTolerance)
        return false;
      planes.push_back(std::move(plane));
    }
    for (size_t i = 0; i < planes.size(); ++i)
      for (size_t j = i + 1; j < planes.size(); ++j) {
        const Vec3 a =
            ToVec(std::get<PlaneParameters>(planes[i].Parameters).Plane.Normal);
        const Vec3 b =
            ToVec(std::get<PlaneParameters>(planes[j].Parameters).Plane.Normal);
        if (std::abs(Dot(a, b)) > std::cos(.05))
          return false;
      }
    size_t index = 0;
    for (auto &entry : groups)
      Store(std::move(entry.second), planes[index++]);
    return true;
  }
  void Residuals() {
    // -1 means undiscovered; -2 means already collected by this traversal.
    // Never use Owner < 0 here: deferred commits would collect visited faces
    // again as singleton components and overwrite their previous ownership.
    // Fit and commit each component before moving to the next seed.
    for (int seed = 0; seed < int(Owner.size()); ++seed)
      if (Owner[seed] == -1) {
        std::vector<int> faces{seed};
        Owner[seed] = -2;
        for (size_t k = 0; k < faces.size(); ++k)
          for (int n : Neighbors[faces[k]])
            if (n >= 0 && Owner[n] == -1) {
              Owner[n] = -2;
              faces.push_back(n);
            }
        MeshPatch model = Fit(faces);
        if ((model.SurfaceType != PatchSurfaceType::Plane &&
             !ResolvedCurvedSupport(faces)) ||
            !Certify(faces, model)) {
          if (SplitSupportedPlanarComponent(faces))
            continue;
          model = FitOnly<FreeformSurfaceFitter>(faces);
          model.SurfaceType = PatchSurfaceType::Freeform;
          model.Parameters = {};
        }
        Store(std::move(faces), model);
      }
  }
  bool SameModelIdentity(const MeshPatch &a, const MeshPatch &b) const {
    if (a.SurfaceType != b.SurfaceType ||
        a.SurfaceType == PatchSurfaceType::Freeform ||
        a.SurfaceType == PatchSurfaceType::Unknown)
      return false;
    // For almost exact models use their measured residual envelope instead of
    // the broad global tolerance: two real parallel faces/radii must not be
    // averaged merely because the fitting threshold permits it.
    const double tolerance = Mesh.getResolution().FittingTolerance;
    const double distance = std::min(
        Config.ModelFitToleranceRatio * tolerance,
        std::max(.2 * tolerance, 3 * (a.MaxFittingError + b.MaxFittingError)));
    const double cosine = std::cos(std::min(.008, Config.ModelNormalTolerance));
    auto aligned = [&](const Direction3 &x, const Direction3 &y) {
      return std::abs(Dot(ToVec(x), ToVec(y))) >= cosine;
    };
    switch (a.SurfaceType) {
    case PatchSurfaceType::Plane: {
      const auto &x = std::get<PlaneParameters>(a.Parameters).Plane;
      const auto &y = std::get<PlaneParameters>(b.Parameters).Plane;
      return aligned(x.Normal, y.Normal) &&
             std::abs(Dot(Sub(ToVec(x.Origin), ToVec(y.Origin)),
                          ToVec(y.Normal))) <= distance;
    }
    case PatchSurfaceType::Cylinder: {
      const auto &x = std::get<CylinderParameters>(a.Parameters);
      const auto &y = std::get<CylinderParameters>(b.Parameters);
      const auto delta = Sub(ToVec(x.Axis.Origin), ToVec(y.Axis.Origin));
      return aligned(x.Axis.Direction, y.Axis.Direction) &&
             std::abs(x.Radius - y.Radius) <= distance &&
             Norm(Cross(delta, ToVec(x.Axis.Direction))) <= distance &&
             Norm(Cross(delta, ToVec(y.Axis.Direction))) <= distance;
    }
    case PatchSurfaceType::Cone: {
      const auto &x = std::get<ConeParameters>(a.Parameters);
      const auto &y = std::get<ConeParameters>(b.Parameters);
      return Dot(ToVec(x.Axis.Direction), ToVec(y.Axis.Direction)) >= cosine &&
             Distance(x.Axis.Origin, y.Axis.Origin) <= distance &&
             std::abs(x.SemiAngle - y.SemiAngle) <= .008;
    }
    case PatchSurfaceType::Sphere: {
      const auto &x = std::get<SphereParameters>(a.Parameters);
      const auto &y = std::get<SphereParameters>(b.Parameters);
      return Distance(x.Center, y.Center) <= distance &&
             std::abs(x.Radius - y.Radius) <= distance;
    }
    case PatchSurfaceType::Torus: {
      const auto &x = std::get<TorusParameters>(a.Parameters);
      const auto &y = std::get<TorusParameters>(b.Parameters);
      return aligned(x.Axis.Direction, y.Axis.Direction) &&
             Distance(x.Axis.Origin, y.Axis.Origin) <= distance &&
             std::abs(x.MajorRadius - y.MajorRadius) <= distance &&
             std::abs(x.MinorRadius - y.MinorRadius) <= distance;
    }
    default:
      return false;
    }
  }
  bool AllCompatible(const std::vector<int> &faces, const MeshPatch &model,
                     double limit, double normalLimit = -1) const {
    const auto check = EvaluateSurfaceCompatibility(Mesh, faces, model);
    return check.Supported && check.ReliableNormalSamples > 0 &&
           check.MaxNormalizedDistance <= limit &&
           check.MaxNormalError <=
               (normalLimit < 0 ? Config.ModelNormalTolerance : normalLimit);
  }
  bool SimilarBridgeCurvature(const MeshPatch &a, const MeshPatch &b) const {
    if (a.SurfaceType != b.SurfaceType)
      return false;
    const double cosine = std::cos(.04);
    const double tolerance = std::min(
        Config.ModelFitToleranceRatio * Mesh.getResolution().FittingTolerance,
        std::max(.2 * Mesh.getResolution().FittingTolerance,
                 3 * (a.MaxFittingError + b.MaxFittingError)));
    if (a.SurfaceType == PatchSurfaceType::Cylinder) {
      const auto &x = std::get<CylinderParameters>(a.Parameters);
      const auto &y = std::get<CylinderParameters>(b.Parameters);
      const double radius = std::min(x.Radius, y.Radius);
      const Vec3 delta = Sub(ToVec(x.Axis.Origin), ToVec(y.Axis.Origin));
      const double displacement =
          std::max(Norm(Cross(delta, ToVec(x.Axis.Direction))),
                   Norm(Cross(delta, ToVec(y.Axis.Direction))));
      const double alignment =
          std::abs(Dot(ToVec(x.Axis.Direction), ToVec(y.Axis.Direction)));
      // A radius jump between resolved coaxial cylinders is a real change.
      // In contrast, a small arc's radius and axis position can move together
      // substantially while its observed points still describe one cylinder.
      if (alignment >= std::cos(.008) && displacement <= tolerance &&
          std::abs(x.Radius - y.Radius) > tolerance)
        return false;
      return alignment >= cosine && displacement <= .1 * radius &&
             std::abs(x.Radius - y.Radius) <= .05 * radius;
    }
    if (a.SurfaceType == PatchSurfaceType::Torus) {
      const auto &x = std::get<TorusParameters>(a.Parameters);
      const auto &y = std::get<TorusParameters>(b.Parameters);
      const double alignment =
          std::abs(Dot(ToVec(x.Axis.Direction), ToVec(y.Axis.Direction)));
      const double displacement = Distance(x.Axis.Origin, y.Axis.Origin);
      if (alignment >= std::cos(.008) && displacement <= tolerance &&
          (std::abs(x.MajorRadius - y.MajorRadius) > tolerance ||
           std::abs(x.MinorRadius - y.MinorRadius) > tolerance))
        return false;
      return alignment >= cosine &&
             displacement <= .1 * std::min(x.MajorRadius, y.MajorRadius) &&
             std::abs(x.MajorRadius - y.MajorRadius) <=
                 .05 * std::min(x.MajorRadius, y.MajorRadius) &&
             std::abs(x.MinorRadius - y.MinorRadius) <=
                 .05 * std::min(x.MinorRadius, y.MinorRadius);
    }
    return false;
  }
  void BridgeResidualSeams() {
    if (!Config.EnableModelSeamBridging || Config.ModelSeamMaximumFaces <= 0 ||
        Config.ModelSeamEvaluationBudget <= 0 ||
        Config.ModelSeamMaximumWidthRadiusRatio <= 0)
      return;
    auto curved = [&](int id) {
      const auto type = Patches[id].SurfaceType;
      return (type == PatchSurfaceType::Cylinder ||
              type == PatchSurfaceType::Torus) &&
             Patches[id].TriangleIds.size() >= 4;
    };
    auto residual = [&](int id) {
      return Patches[id].SurfaceType == PatchSurfaceType::Plane ||
             Patches[id].SurfaceType == PatchSurfaceType::Freeform;
    };
    // Seeds are actual mesh interfaces. A nearby surface across a hole is
    // never a candidate, and a large residual need not be consumed as a whole.
    std::map<std::pair<int, int>, std::vector<int>> seeds;
    for (const auto &edge : Mesh.getEdges()) {
      if (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature ||
          edge.IncidentTriangleIds.size() != 2)
        continue;
      for (int side = 0; side < 2; ++side) {
        int a = edge.IncidentTriangleIds[side];
        int b = edge.IncidentTriangleIds[1 - side];
        if (Owner[a] != Owner[b] && curved(Owner[a]) && residual(Owner[b]))
          seeds[{Owner[a], Owner[b]}].push_back(b);
      }
    }
    struct Rail {
      Vec3 A, B, Midpoint, Inward, Tangent;
      double Length = 0;
      int PatchId = -1;
    };
    size_t evaluated = 0, components = 0, accepted = 0, transferred = 0;
    size_t bounded = 0, unsupported = 0, oppositionRejected = 0, wide = 0;
    size_t hard = 0, fitRejected = 0;
    size_t commonModelFaces = 0, commonModels = 0;
    std::map<std::pair<int, int>, bool> commonModelCache;
    auto commonSurface = [&](int a, int b) {
      if (a == b)
        return true;
      const auto key = std::make_pair(std::min(a, b), std::max(a, b));
      const auto prior = commonModelCache.find(key);
      if (prior != commonModelCache.end())
        return prior->second;
      bool compatible = SameModelIdentity(Patches[a], Patches[b]);
      if (!compatible && SimilarBridgeCurvature(Patches[a], Patches[b])) {
        const size_t count =
            Patches[a].TriangleIds.size() + Patches[b].TriangleIds.size();
        if (commonModelFaces + count <=
            size_t(Config.ModelSeamEvaluationBudget)) {
          commonModelFaces += count;
          auto faces = Patches[a].TriangleIds;
          faces.insert(faces.end(), Patches[b].TriangleIds.begin(),
                       Patches[b].TriangleIds.end());
          // Point support, not parameter proximity, adjudicates small arcs.
          // One better resolved end often already explains both fragments.
          compatible =
              AllCompatible(faces, Patches[a], Config.ModelFitToleranceRatio) ||
              AllCompatible(faces, Patches[b], Config.ModelFitToleranceRatio);
          if (!compatible) {
            auto joint = RefitType(faces, Patches[a].SurfaceType);
            compatible = Certify(faces, joint) &&
                         SimilarBridgeCurvature(Patches[a], joint) &&
                         SimilarBridgeCurvature(Patches[b], joint);
          }
          commonModels += compatible;
        } else {
          ++bounded;
        }
      }
      commonModelCache[key] = compatible;
      return compatible;
    };
    std::set<int> changedResiduals;
    std::vector<int> membership(Owner.size(), 0);
    int componentStamp = 0;
    for (const auto &entry : seeds) {
      const int target = entry.first.first, source = entry.first.second;
      if (Patches[target].TriangleIds.empty())
        continue;
      ++Stamp;
      const int searchStamp = Stamp;
      for (int seed : entry.second) {
        if (Owner[seed] != source || Visit[seed] == searchStamp)
          continue;
        if (evaluated >= size_t(Config.ModelSeamEvaluationBudget)) {
          ++bounded;
          break;
        }
        const auto anchor = Patches[target];
        const double radius =
            anchor.SurfaceType == PatchSurfaceType::Cylinder
                ? std::get<CylinderParameters>(anchor.Parameters).Radius
                : std::get<TorusParameters>(anchor.Parameters).MinorRadius;
        const double maximumWidth =
            Config.ModelSeamMaximumWidthRadiusRatio * radius;
        if (!(maximumWidth > 0))
          continue;
        // A noise allowance is needed beyond the discovery envelope; the
        // accepted union is still refitted and checked against every face.
        const double distanceLimit = Config.ModelFitToleranceRatio;
        const double normalLimit = std::min(.07, Config.ModelNormalTolerance);
        auto eligible = [&](int f) {
          ++evaluated;
          const auto check = EvaluateSurfaceCompatibility(Mesh, f, anchor);
          return check.Supported && check.ReliableNormalSamples > 0 &&
                 check.MaxNormalizedDistance <= distanceLimit &&
                 check.MaxNormalError <= normalLimit;
        };
        Visit[seed] = searchStamp;
        if (!eligible(seed))
          continue;
        std::vector<int> corridor{seed};
        bool overflow = false;
        for (size_t i = 0; i < corridor.size() && !overflow; ++i)
          for (int next : Neighbors[corridor[i]]) {
            if (next < 0 || Owner[next] != source || Visit[next] == searchStamp)
              continue;
            Visit[next] = searchStamp;
            if (evaluated >= size_t(Config.ModelSeamEvaluationBudget)) {
              overflow = true;
              break;
            }
            if (eligible(next)) {
              if (corridor.size() >= size_t(Config.ModelSeamMaximumFaces)) {
                overflow = true;
                break;
              }
              corridor.push_back(next);
            }
          }
        if (overflow) {
          ++bounded;
          // An interrupted component has no established outer boundary.
          // Do not retry partial pieces of it from another interface seed.
          break;
        }
        ++components;
        ++componentStamp;
        for (int f : corridor)
          membership[f] = componentStamp;
        std::vector<Rail> rails;
        std::set<int> anchors{target};
        double area = 0, perimeter = 0, supportLength = 0;
        bool hardConflict = false;
        for (int f : corridor) {
          const auto &triangle = Mesh.getTriangles()[f];
          area += triangle.Area;
          for (int edgeId : triangle.EdgeIds) {
            const auto &edge = Mesh.getEdges()[edgeId];
            int other = -1;
            if (edge.IncidentTriangleIds.size() == 2)
              other = edge.IncidentTriangleIds[0] == f
                          ? edge.IncidentTriangleIds[1]
                          : edge.IncidentTriangleIds[0];
            const bool internal =
                other >= 0 && membership[other] == componentStamp;
            const Vec3 a = ToVec(Mesh.getVertices()[edge.Vertex0].Position);
            const Vec3 b = ToVec(Mesh.getVertices()[edge.Vertex1].Position);
            const double length = Norm(Sub(b, a));
            if (internal) {
              hardConflict = hardConflict || edge.IsConstrainedFeature ||
                             edge.IsNonManifold;
              continue;
            }
            perimeter += length;
            if (other < 0)
              continue;
            const int otherPatch = Owner[other];
            const bool sameSurface =
                otherPatch == target ||
                (curved(otherPatch) && commonSurface(target, otherPatch));
            if (!sameSurface)
              continue;
            if (edge.IsConstrainedFeature || edge.IsNonManifold ||
                edge.IsBoundary) {
              hardConflict = true;
              continue;
            }
            const Vec3 midpoint = Mul(Add(a, b), .5);
            Vec3 normal;
            double distance;
            if (!SegmentationGuardDetail::SurfaceSample(anchor, midpoint,
                                                        distance, normal))
              continue;
            const Vec3 tangent = Normalize(Sub(b, a));
            Vec3 inward = Normalize(Cross(normal, tangent));
            if (Dot(inward, Sub(ToVec(triangle.Centroid), midpoint)) < 0)
              inward = Mul(inward, -1);
            rails.push_back(
                {a, b, midpoint, inward, tangent, length, otherPatch});
            anchors.insert(otherPatch);
            supportLength += length;
          }
        }
        // Do not internalize a protected A/B interface by taking an alternate
        // unprotected route around its endpoint.
        for (int id : anchors)
          for (int f : Patches[id].TriangleIds)
            for (int edgeId : Mesh.getTriangles()[f].EdgeIds) {
              const auto &edge = Mesh.getEdges()[edgeId];
              if (!edge.IsConstrainedFeature && !edge.IsNonManifold)
                continue;
              for (int other : edge.IncidentTriangleIds)
                if (Owner[other] != id && anchors.count(Owner[other]))
                  hardConflict = true;
            }
        for (int f : corridor)
          for (int edgeId : Mesh.getTriangles()[f].EdgeIds) {
            const auto &edge = Mesh.getEdges()[edgeId];
            for (int other : edge.IncidentTriangleIds) {
              if (other == f || (membership[other] != componentStamp &&
                                 !anchors.count(Owner[other])))
                continue;
              if (edge.IsConstrainedFeature || edge.IsNonManifold)
                hardConflict = true;
              auto direction = [&](int face) {
                const auto &ids = Mesh.getTriangles()[face].VertexIds;
                for (int k = 0; k < 3; ++k)
                  if (ids[k] == edge.Vertex0)
                    return ids[(k + 1) % 3] == edge.Vertex1 ? 1 : -1;
                return 0;
              };
              if (direction(f) == direction(other))
                hardConflict = true;
            }
          }
        if (hardConflict) {
          ++hard;
          continue;
        }
        if (rails.size() < 2 || supportLength < .35 * perimeter) {
          ++unsupported;
          continue;
        }
        double anchorArea = 0;
        for (int id : anchors)
          for (int f : Patches[id].TriangleIds)
            anchorArea += Mesh.getTriangles()[f].Area;
        const double width = 2 * area / std::max(supportLength, 1e-30);
        if (width > maximumWidth || area > .35 * anchorArea) {
          ++wide;
          continue;
        }
        // Opposing conormals distinguish two rails from one long tangent
        // border, including a slit whose two sides have the same patch ID.
        // Sample by boundary length so dense tessellation gets no extra vote.
        const int sampleCount = std::min<size_t>(128, rails.size());
        size_t sampledRail = 0;
        double cumulativeLength = rails[0].Length;
        int pairedSamples = 0;
        for (int sample = 0; sample < sampleCount; ++sample) {
          const double position = (sample + .5) * supportLength / sampleCount;
          while (sampledRail + 1 < rails.size() && cumulativeLength < position)
            cumulativeLength += rails[++sampledRail].Length;
          const auto &a = rails[sampledRail];
          bool paired = false;
          for (const auto &b : rails) {
            if (Dot(a.Inward, b.Inward) > -.5 ||
                std::abs(Dot(a.Tangent, b.Tangent)) < .7)
              continue;
            const double t =
                std::clamp(Dot(Sub(a.Midpoint, b.A), b.Tangent), 0.0, b.Length);
            const Vec3 delta = Sub(Add(b.A, Mul(b.Tangent, t)), a.Midpoint);
            const double distance = Norm(delta);
            if (distance <= maximumWidth && distance > radius * 1e-10 &&
                Dot(delta, a.Inward) > .4 * distance &&
                Dot(delta, b.Inward) < -.4 * distance) {
              paired = true;
              break;
            }
          }
          pairedSamples += paired;
        }
        if (pairedSamples < .7 * sampleCount) {
          ++oppositionRejected;
          continue;
        }
        auto combined = corridor;
        for (int id : anchors)
          combined.insert(combined.end(), Patches[id].TriangleIds.begin(),
                          Patches[id].TriangleIds.end());
        auto model = RefitType(combined, anchor.SurfaceType);
        if (!ResolvedCurvedSupport(combined) || !Certify(combined, model) ||
            !SimilarBridgeCurvature(anchor, model) ||
            !AllCompatible(corridor, model, Config.ModelFitToleranceRatio,
                           normalLimit)) {
          ++fitRejected;
          continue;
        }
        model.Id = target;
        model.TriangleIds = std::move(combined);
        for (int id : anchors)
          if (id != target)
            Patches[id].TriangleIds.clear();
        Patches[target] = std::move(model);
        for (int f : Patches[target].TriangleIds)
          Owner[f] = target;
        changedResiduals.insert(source);
        commonModelCache.clear();
        ++accepted;
        transferred += corridor.size();
      }
    }
    if (accepted) {
      auto prior = std::move(Patches);
      Patches.clear();
      // Extracting an interior corridor can divide its old residual. Restore
      // connected patch ownership explicitly rather than leaving islands.
      for (int id = 0; id < int(prior.size()); ++id) {
        auto &patch = prior[id];
        if (!changedResiduals.count(id)) {
          if (!patch.TriangleIds.empty()) {
            // Owner is still needed below; defer its compact reindexing.
            Patches.push_back(std::move(patch));
          }
          continue;
        }
        ++Stamp;
        for (int seed : patch.TriangleIds) {
          if (Owner[seed] != id || Visit[seed] == Stamp)
            continue;
          std::vector<int> component{seed};
          Visit[seed] = Stamp;
          for (size_t i = 0; i < component.size(); ++i)
            for (int next : Neighbors[component[i]])
              if (next >= 0 && Owner[next] == id && Visit[next] != Stamp) {
                Visit[next] = Stamp;
                component.push_back(next);
              }
          MeshPatch remainder = patch;
          if (patch.SurfaceType == PatchSurfaceType::Freeform) {
            // These are diagnostics for this remaining connected component,
            // not the larger parent before its corridor was reassigned.
            remainder = FitOnly<FreeformSurfaceFitter>(component);
            if (remainder.SurfaceType != PatchSurfaceType::Freeform)
              throw std::runtime_error(
                  "Could not refresh residual Freeform diagnostics");
            remainder.Parameters = {};
          }
          remainder.TriangleIds = std::move(component);
          Patches.push_back(std::move(remainder));
        }
      }
      prior = std::move(Patches);
      Patches.clear();
      for (auto &patch : prior) {
        auto faces = std::move(patch.TriangleIds);
        Store(std::move(faces), patch);
      }
    }
    if (Config.Verbose)
      std::clog << "[CadMesh] residual seam bridges=" << accepted
                << ", transferred_faces=" << transferred
                << ", components=" << components << ", evaluated=" << evaluated
                << ", common_models=" << commonModels
                << ", common_model_faces=" << commonModelFaces
                << ", rejected(bound=" << bounded << ", support=" << unsupported
                << ", opposition=" << oppositionRejected << ", width=" << wide
                << ", hard=" << hard << ", fit=" << fitRejected
                << "), patches=" << Patches.size() << '\n';
  }
  void MergeAnalyticFragments() {
    size_t accepted = 0, attempted = 0, absorbed = 0;
    std::vector<double> areas(Patches.size(), 0);
    for (size_t p = 0; p < Patches.size(); ++p)
      for (int f : Patches[p].TriangleIds)
        areas[p] += Mesh.getTriangles()[f].Area;
    for (int pass = 0; pass < std::max(0, Config.ModelMaximumMergePasses);
         ++pass) {
      std::set<std::pair<int, int>> adjacent;
      for (const auto &edge : Mesh.getEdges()) {
        if (edge.IsBoundary || edge.IsNonManifold ||
            edge.IsConstrainedFeature || edge.IncidentTriangleIds.size() != 2)
          continue;
        int a = Owner[edge.Triangle0], b = Owner[edge.Triangle1];
        if (a >= 0 && b >= 0 && a != b)
          adjacent.emplace(std::min(a, b), std::max(a, b));
      }
      std::vector<int> parent(Patches.size());
      std::iota(parent.begin(), parent.end(), 0);
      auto root = [&](int id) {
        while (parent[id] != id) {
          parent[id] = parent[parent[id]];
          id = parent[id];
        }
        return id;
      };
      size_t mergedThisPass = 0;
      for (const auto &pair : adjacent) {
        int a = root(pair.first), b = root(pair.second);
        if (a == b)
          continue;
        if (areas[a] < areas[b])
          std::swap(a, b);
        const bool sameModel = SameModelIdentity(Patches[a], Patches[b]);
        const auto targetType = Patches[a].SurfaceType;
        const auto sourceType = Patches[b].SurfaceType;
        const bool analyticTarget = targetType != PatchSurfaceType::Unknown &&
                                    targetType != PatchSurfaceType::Freeform;
        const bool residualSource = sourceType == PatchSurfaceType::Freeform ||
                                    (sourceType == PatchSurfaceType::Plane &&
                                     targetType != PatchSurfaceType::Plane);
        const bool smallResidual =
            analyticTarget && residualSource &&
            Patches[b].TriangleIds.size() <=
                (sourceType == PatchSurfaceType::Freeform ? 64 : 12) &&
            Patches[a].TriangleIds.size() >=
                std::max<size_t>(12, 4 * Patches[b].TriangleIds.size()) &&
            areas[b] <= .1 * areas[a];
        if (!sameModel && !smallResidual)
          continue;
        bool hardConflict = false;
        for (int f : Patches[b].TriangleIds)
          for (int e : Mesh.getTriangles()[f].EdgeIds) {
            const auto &edge = Mesh.getEdges()[e];
            if (!edge.IsConstrainedFeature && !edge.IsNonManifold)
              continue;
            for (int other : edge.IncidentTriangleIds)
              if (Owner[other] == a)
                hardConflict = true;
          }
        // Final consolidation checks the fitting contract, not the stricter
        // discovery envelope. Fitted fragments can have differing local noise
        // even though their full union represents one certified surface.
        const double tolerance = Config.ModelFitToleranceRatio;
        const double sourceNormalLimit =
            sameModel ? Config.ModelNormalTolerance
                      : std::min(Config.ModelNormalTolerance, .035);
        if (hardConflict ||
            !AllCompatible(Patches[b].TriangleIds, Patches[a], tolerance,
                           sourceNormalLimit) ||
            (sameModel &&
             !AllCompatible(Patches[a].TriangleIds, Patches[b], tolerance)))
          continue;
        ++attempted;
        auto combined = Patches[a].TriangleIds;
        combined.insert(combined.end(), Patches[b].TriangleIds.begin(),
                        Patches[b].TriangleIds.end());
        auto model = RefitType(combined, Patches[a].SurfaceType);
        if (!Certify(combined, model) ||
            !AllCompatible(Patches[b].TriangleIds, model, tolerance,
                           sourceNormalLimit))
          continue;
        model.Id = a;
        model.TriangleIds = std::move(combined);
        Patches[a] = std::move(model);
        Patches[b].TriangleIds.clear();
        areas[a] += areas[b];
        areas[b] = 0;
        parent[b] = a;
        for (int f : Patches[a].TriangleIds)
          Owner[f] = a;
        ++mergedThisPass;
        absorbed += !sameModel;
      }
      accepted += mergedThisPass;
      if (!mergedThisPass)
        break;
    }
    if (accepted) {
      auto prior = std::move(Patches);
      Patches.clear();
      for (auto &patch : prior)
        if (!patch.TriangleIds.empty()) {
          auto faces = std::move(patch.TriangleIds);
          Store(std::move(faces), patch);
        }
    }
    if (Config.Verbose)
      std::clog << "[CadMesh] final adjacency merge=" << accepted << '/'
                << attempted
                << " certified unions (residual absorptions=" << absorbed
                << "), patches=" << Patches.size() << '\n';
  }

public:
  ModelPartition(MeshTopology &mesh, const SegmentationConfig &config)
      : Mesh(mesh), Config(config), Owner(mesh.getTriangles().size(), -1),
        CellId(Owner.size(), -1), Visit(Owner.size(), 0) {}
  std::vector<MeshPatch> Run() {
    CudaAnalyticFitSession fitting(Config.ModelAnalyticSeedBackend, Config.Verbose);
    auto stage = [&](const char *name, auto &&work) {
      const auto begin = std::chrono::steady_clock::now();
      work();
      if (Config.Verbose)
        std::clog << "[CadMesh] " << name << ": "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
                  << " s\n";
    };
    stage("model neighbors", [&] { BuildNeighbors(); });
    stage("whole component fits", [&] { WholeComponents(); });
    stage("planar cells", [&] { BuildCells(); });
    stage("mother planes", [&] { ExtractPlaneCores(); });
    stage("analytic seed search", [&] { ExtractCurves(); });
    LogRemainingProbes("analytic seed search");
    stage("residual seed search", [&] { DiscoverResidualModels(); });
    LogRemainingProbes("residual seed search");
    RemainingFailures.clear();
    RemainingCachedFaces = 0;
    stage("residual classification", [&] { Residuals(); });
    stage("analytic adjacency merge", [&] { MergeAnalyticFragments(); });
    stage("residual seam bridging", [&] { BridgeResidualSeams(); });
    return std::move(Patches);
  }
  std::vector<MeshPatch> Consolidate(std::vector<MeshPatch> patches) {
    Patches = std::move(patches);
    for (int id = 0; id < int(Patches.size()); ++id) {
      Patches[id].Id = id;
      for (int face : Patches[id].TriangleIds) {
        if (Owner.at(face) >= 0)
          throw std::invalid_argument(
              "Consolidation input contains duplicate triangle ownership");
        Owner.at(face) = id;
      }
    }
    if (std::any_of(Owner.begin(), Owner.end(),
                    [](int owner) { return owner < 0; }))
      throw std::invalid_argument("Consolidation input omits a triangle");
    BuildNeighbors();
    MergeAnalyticFragments();
    BridgeResidualSeams();
    return std::move(Patches);
  }
};
} // namespace
std::vector<MeshPatch>
PartitionBySurfaceModels(MeshTopology &mesh, const SegmentationConfig &config) {
  return ModelPartition(mesh, config).Run();
}
std::vector<MeshPatch>
MergeAdjacentSurfaceModels(MeshTopology &mesh, std::vector<MeshPatch> patches,
                           const SegmentationConfig &config) {
  return ModelPartition(mesh, config).Consolidate(std::move(patches));
}
} // namespace CadMesh
