#include "CadMesh/HeatPatchRegularizer.h"
#include "CadMesh/CadMeshPatchSegmenter.h"
#include "CadMesh/SegmentationGuards.h"
#include "CadMesh/SurfaceFitting.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace CadMesh {
namespace {
using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

void ApplyFit(MeshPatch &patch, const SurfaceFitResult &fit) {
  patch.SurfaceType = fit.Type;
  patch.RmsFittingError = fit.Rms;
  patch.MaxFittingError = fit.Max;
  patch.NormalError = fit.Normal;
  patch.Parameters = fit.Parameters;
  patch.Confidence = Clamp01(std::exp(-fit.Score * .2));
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

double EdgeLength(const MeshTopology &mesh, int edgeId) {
  const auto &edge = mesh.getEdges()[edgeId];
  return Distance(mesh.getVertices()[edge.Vertex0].Position,
                  mesh.getVertices()[edge.Vertex1].Position);
}

double PatchModelResidual(const MeshTopology &mesh, const MeshPatch &source,
                          const MeshPatch &target) {
  auto compatibility =
      EvaluateSurfaceCompatibility(mesh, source.TriangleIds, target);
  return IsSurfaceCompatible(compatibility, mesh.getResolution(), 1e100)
             ? compatibility.MaxNormalizedDistance
             : 1e100;
}

bool IsAmbiguous(const MeshPatch &patch, const SegmentationConfig &config) {
  if (patch.TriangleIds.size() >
      size_t(config.HeatMaximumAmbiguousPatchTriangles))
    return false;
  bool lowConfidenceSmallPatch =
      patch.TriangleIds.size() <= 10 && patch.Confidence < .95;
  return patch.SurfaceType == PatchSurfaceType::Freeform ||
         patch.Confidence < config.HeatSeedConfidence ||
         lowConfidenceSmallPatch;
}
} // namespace

HeatRegularizationReport
HeatPatchRegularizer::regularize(CadMeshPatchSegmenter &segmenter) {
  HeatRegularizationReport report;
  auto &mesh = segmenter.mMesh;
  auto &patches = segmenter.mPatches;
  const auto &config = segmenter.mConfig;
  if (!config.EnableHeatRegularization || patches.empty())
    return report;
  segmenter.AssignPatchIds();
  const auto &triangles = mesh.getTriangles();
  const auto &edges = mesh.getEdges();
  const auto &vertices = mesh.getVertices();
  const int patchCount = int(patches.size());

  auto canTransferTriangle = [&](int triangleId, int target) {
    if (!IsSurfaceCompatible(
            EvaluateSurfaceCompatibility(mesh, triangleId, patches[target]),
            mesh.getResolution(), config.HeatSeedDemotionResidualTolerance))
      return false;
    for (int edgeId : triangles[triangleId].EdgeIds) {
      const auto &edge = edges[edgeId];
      if (!IsHardSegmentationBoundary(edge, config))
        continue;
      // Boundary and non-manifold neighborhoods have no unambiguous side
      // across which a heat label can be transported.
      if (edge.IsBoundary || edge.IsNonManifold)
        return false;
      for (int neighbor : edge.IncidentTriangleIds)
        if (neighbor != triangleId && triangles[neighbor].PatchId == target)
          return false;
    }
    return true;
  };

  std::vector<char> ambiguous(patchCount, 0);
  for (int patchId = 0; patchId < patchCount; ++patchId)
    ambiguous[patchId] = IsAmbiguous(patches[patchId], config);

  std::vector<int> parent(patchCount);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto unite = [&](int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b)
      parent[b] = a;
  };
  std::vector<std::vector<std::pair<int, int>>> patchContacts(patchCount);
  for (int edgeId = 0; edgeId < int(edges.size()); ++edgeId) {
    const auto &edge = edges[edgeId];
    if (IsHardSegmentationBoundary(edge, config) || edge.Triangle0 < 0 ||
        edge.Triangle1 < 0)
      continue;
    int a = triangles[edge.Triangle0].PatchId;
    int b = triangles[edge.Triangle1].PatchId;
    if (a < 0 || b < 0 || a == b)
      continue;
    patchContacts[a].push_back({b, edgeId});
    patchContacts[b].push_back({a, edgeId});
  }
  // A small patch can fit an accidental primitive extremely well simply
  // because its triangles are long and nearly degenerate.  Such a patch must
  // not become an immutable heat source when all of its vertices also satisfy
  // a much larger neighboring analytic surface.  Demote it to the thermal
  // competition domain; genuine fillets fail the mother-surface residual.
  for (int source = 0; source < patchCount; ++source) {
    if (ambiguous[source] ||
        patches[source].TriangleIds.size() >
            size_t(config.HeatMaximumAmbiguousPatchTriangles))
      continue;
    for (const auto &contact : patchContacts[source]) {
      int target = contact.first;
      if (patches[target].TriangleIds.size() <
          config.HeatSeedDemotionTargetRatio *
              patches[source].TriangleIds.size())
        continue;
      double length = 0, weightedBoundary = 0;
      for (const auto &candidateContact : patchContacts[source])
        if (candidateContact.first == target) {
          double edgeLength = EdgeLength(mesh, candidateContact.second);
          length += edgeLength;
          weightedBoundary +=
              edgeLength * edges[candidateContact.second].BoundaryScore;
        }
      if (length <= 1e-30 ||
          weightedBoundary / length >= config.HeatSeedDemotionBoundaryScore)
        continue;
      if (PatchModelResidual(mesh, patches[source], patches[target]) <=
          config.HeatSeedDemotionResidualTolerance) {
        ambiguous[source] = 1;
        ++report.DemotedGeometricSeeds;
        break;
      }
    }
  }
  for (int patchId = 0; patchId < patchCount; ++patchId)
    if (ambiguous[patchId])
      for (const auto &contact : patchContacts[patchId])
        if (ambiguous[contact.first])
          unite(patchId, contact.first);
  std::map<int, std::vector<int>> components;
  for (int patchId = 0; patchId < patchCount; ++patchId)
    if (ambiguous[patchId])
      components[find(patchId)].push_back(patchId);
  report.AmbiguousComponents = int(components.size());

  std::vector<int> destination(triangles.size(), -1);
  std::vector<char> componentPatch(patchCount, 0);
  for (const auto &componentItem : components) {
    const auto &componentIds = componentItem.second;
    size_t ambiguousTriangleCount = 0;
    for (int patchId : componentIds) {
      componentPatch[patchId] = 1;
      ambiguousTriangleCount += patches[patchId].TriangleIds.size();
    }
    if (ambiguousTriangleCount > size_t(config.HeatMaximumComponentTriangles)) {
      ++report.SkippedOversizedComponents;
      for (int patchId : componentIds)
        componentPatch[patchId] = 0;
      continue;
    }

    std::map<int, double> candidateContactLength;
    std::map<int, std::vector<int>> candidateBoundaryEdges;
    for (int patchId : componentIds)
      for (const auto &contact : patchContacts[patchId]) {
        int neighbor = contact.first;
        if (componentPatch[neighbor])
          continue;
        const auto &candidate = patches[neighbor];
        if (candidate.Confidence < config.HeatSeedConfidence &&
            candidate.TriangleIds.size() <=
                size_t(config.HeatMaximumAmbiguousPatchTriangles))
          continue;
        candidateContactLength[neighbor] += EdgeLength(mesh, contact.second);
        candidateBoundaryEdges[neighbor].push_back(contact.second);
      }
    std::vector<int> labels;
    for (const auto &item : candidateContactLength)
      labels.push_back(item.first);
    std::sort(labels.begin(), labels.end(), [&](int a, int b) {
      return candidateContactLength[a] > candidateContactLength[b];
    });
    if (labels.size() > size_t(config.HeatMaximumLabels))
      labels.resize(config.HeatMaximumLabels);
    if (labels.empty()) {
      for (int patchId : componentIds)
        componentPatch[patchId] = 0;
      continue;
    }
    if (labels.size() == 1) {
      bool compatible = true;
      for (int patchId : componentIds)
        if (PatchModelResidual(mesh, patches[patchId], patches[labels[0]]) >
            config.HeatSeedDemotionResidualTolerance) {
          compatible = false;
          break;
        }
      if (compatible)
        for (int patchId : componentIds)
          for (int triangleId : patches[patchId].TriangleIds)
            if (labels[0] != patchId &&
                canTransferTriangle(triangleId, labels[0])) {
              destination[triangleId] = labels[0];
              ++report.ReassignedTriangles;
            }
      for (int patchId : componentIds)
        componentPatch[patchId] = 0;
      continue;
    }

    std::unordered_set<int> domainTriangles;
    domainTriangles.reserve(ambiguousTriangleCount + labels.size() * 64);
    for (int patchId : componentIds)
      domainTriangles.insert(patches[patchId].TriangleIds.begin(),
                             patches[patchId].TriangleIds.end());
    struct RingItem {
      int TriangleId = -1;
      int Label = -1;
      int Depth = 0;
    };
    std::queue<RingItem> ring;
    for (int label : labels)
      for (int edgeId : candidateBoundaryEdges[label]) {
        const auto &edge = edges[edgeId];
        for (int triangleId : edge.IncidentTriangleIds)
          if (triangles[triangleId].PatchId == label &&
              domainTriangles.insert(triangleId).second)
            ring.push({triangleId, label, 0});
      }
    while (!ring.empty()) {
      RingItem item = ring.front();
      ring.pop();
      if (item.Depth >= 2)
        continue;
      for (int neighbor : mesh.getTriangleNeighbors(item.TriangleId))
        if (triangles[neighbor].PatchId == item.Label &&
            domainTriangles.insert(neighbor).second)
          ring.push({neighbor, item.Label, item.Depth + 1});
    }

    std::vector<int> localVertices;
    std::unordered_map<int, int> localIndex;
    localIndex.reserve(domainTriangles.size() * 2);
    for (int triangleId : domainTriangles)
      for (int vertexId : triangles[triangleId].VertexIds)
        if (localIndex.emplace(vertexId, int(localVertices.size())).second)
          localVertices.push_back(vertexId);
    std::unordered_map<int, double> cotangentWeights;
    cotangentWeights.reserve(domainTriangles.size() * 2);
    for (int triangleId : domainTriangles) {
      const auto &triangle = triangles[triangleId];
      for (int k = 0; k < 3; ++k) {
        int a = triangle.VertexIds[k];
        int b = triangle.VertexIds[(k + 1) % 3];
        int opposite = triangle.VertexIds[(k + 2) % 3];
        Vec3 u = Sub(ToVec(vertices[a].Position),
                     ToVec(vertices[opposite].Position));
        Vec3 v = Sub(ToVec(vertices[b].Position),
                     ToVec(vertices[opposite].Position));
        double denominator = Norm(Cross(u, v));
        if (denominator <= 1e-30)
          continue;
        double cotangent = Dot(u, v) / denominator;
        cotangentWeights[triangle.EdgeIds[k]] +=
            .5 * std::max(0.0, std::min(cotangent, 1e4));
      }
    }

    std::vector<int> componentTriangles;
    componentTriangles.reserve(ambiguousTriangleCount);
    double componentArea = 0;
    for (int patchId : componentIds)
      for (int triangleId : patches[patchId].TriangleIds) {
        componentTriangles.push_back(triangleId);
        componentArea += triangles[triangleId].Area;
      }
    for (int resolvePass = 0;
         resolvePass < config.HeatMaximumResolvePasses && labels.size() >= 2;
         ++resolvePass) {
      std::unordered_map<int, int> labelColumn;
      for (int column = 0; column < int(labels.size()); ++column)
        labelColumn[labels[column]] = column;
      std::vector<int> seedLabel(localVertices.size(), -1);
      std::vector<int> seedCounts(labels.size(), 0);
      for (int localId = 0; localId < int(localVertices.size()); ++localId) {
        int vertexId = localVertices[localId];
        bool touchesAmbiguous = false;
        int uniqueLabel = -1;
        bool conflicting = false;
        for (int triangleId : vertices[vertexId].IncidentTriangleIds) {
          if (!domainTriangles.count(triangleId))
            continue;
          int patchId = triangles[triangleId].PatchId;
          if (componentPatch[patchId]) {
            touchesAmbiguous = true;
            continue;
          }
          auto labelIt = labelColumn.find(patchId);
          if (labelIt == labelColumn.end())
            continue;
          if (uniqueLabel < 0)
            uniqueLabel = labelIt->second;
          else if (uniqueLabel != labelIt->second)
            conflicting = true;
        }
        if (!touchesAmbiguous && !conflicting && uniqueLabel >= 0) {
          seedLabel[localId] = uniqueLabel;
          ++seedCounts[uniqueLabel];
        }
      }
      std::vector<int> activeLabels;
      for (int column = 0; column < int(labels.size()); ++column)
        if (seedCounts[column] > 0)
          activeLabels.push_back(labels[column]);
      if (activeLabels.size() != labels.size()) {
        report.PrunedThermalSeeds +=
            int(labels.size() - activeLabels.size());
        labels = std::move(activeLabels);
        --resolvePass;
        continue;
      }
      if (labels.size() < 2)
        break;

      std::vector<int> unknownIndex(localVertices.size(), -1);
      int unknownCount = 0;
      for (int localId = 0; localId < int(localVertices.size()); ++localId)
        if (seedLabel[localId] < 0)
          unknownIndex[localId] = unknownCount++;
      if (!unknownCount)
        break;
      std::vector<Triplet> coefficients;
      coefficients.reserve(cotangentWeights.size() * 4 + unknownCount);
      std::vector<double> diagonal(unknownCount, 1e-10);
      Eigen::MatrixXd rightHandSide =
          Eigen::MatrixXd::Zero(unknownCount, int(labels.size()));
      for (const auto &weightItem : cotangentWeights) {
        int edgeId = weightItem.first;
        const auto &edge = edges[edgeId];
        auto aIt = localIndex.find(edge.Vertex0);
        auto bIt = localIndex.find(edge.Vertex1);
        if (aIt == localIndex.end() || bIt == localIndex.end())
          continue;
        double effectiveBoundary = edge.BoundaryScore;
        double minimumQuality = 1.0;
        for (int triangleId : edge.IncidentTriangleIds)
          minimumQuality =
              std::min(minimumQuality, TriangleQuality(mesh, triangleId));
        if (minimumQuality < .015)
          effectiveBoundary = std::max(
              0.0,
              effectiveBoundary -
                  config.NormalWeight * edge.Evidence.NormalDiscontinuity -
                  config.TessellationWeight *
                      edge.Evidence.TessellationEvidence);
        if (IsHardSegmentationBoundary(edge, config) ||
          effectiveBoundary >= config.HeatHardBoundaryScore)
          continue;
        double conductance =
            (weightItem.second + .02) *
            std::exp(-config.HeatBoundaryBeta * effectiveBoundary *
                     effectiveBoundary);
        if (conductance <= 1e-14)
          continue;
        int a = aIt->second, b = bIt->second;
        int ua = unknownIndex[a], ub = unknownIndex[b];
        if (ua >= 0) {
          diagonal[ua] += conductance;
          if (ub >= 0)
            coefficients.emplace_back(ua, ub, -conductance);
          else if (seedLabel[b] >= 0)
            rightHandSide(ua, seedLabel[b]) += conductance;
        }
        if (ub >= 0) {
          diagonal[ub] += conductance;
          if (ua >= 0)
            coefficients.emplace_back(ub, ua, -conductance);
          else if (seedLabel[a] >= 0)
            rightHandSide(ub, seedLabel[a]) += conductance;
        }
      }
      for (int i = 0; i < unknownCount; ++i)
        coefficients.emplace_back(i, i, diagonal[i]);
      SparseMatrix system(unknownCount, unknownCount);
      system.setFromTriplets(coefficients.begin(), coefficients.end());
      Eigen::SimplicialLDLT<SparseMatrix> solver;
      solver.compute(system);
      if (solver.info() != Eigen::Success) {
        ++report.FailedSolves;
        break;
      }
      Eigen::MatrixXd solution = solver.solve(rightHandSide);
      ++report.ResolvePasses;
      if (solver.info() != Eigen::Success || !solution.allFinite()) {
        ++report.FailedSolves;
        break;
      }
      std::vector<std::vector<double>> probabilities;
      probabilities.reserve(componentTriangles.size());
      std::vector<int> ownedTriangles(labels.size(), 0);
      std::vector<double> ownedArea(labels.size(), 0);
      for (int triangleId : componentTriangles) {
        std::vector<double> probability(labels.size(), 0);
        for (int vertexId : triangles[triangleId].VertexIds) {
          int localId = localIndex[vertexId];
          if (seedLabel[localId] >= 0)
            probability[seedLabel[localId]] += 1.0;
          else {
            int unknown = unknownIndex[localId];
            double sum = 0;
            for (int column = 0; column < int(labels.size()); ++column)
              sum += std::max(0.0, solution(unknown, column));
            if (sum > 1e-30)
              for (int column = 0; column < int(labels.size()); ++column)
                probability[column] +=
                    std::max(0.0, solution(unknown, column)) / sum;
          }
        }
        for (double &value : probability)
          value /= 3.0;
        int owner = int(std::max_element(probability.begin(),
                                         probability.end()) -
                        probability.begin());
        ++ownedTriangles[owner];
        ownedArea[owner] += triangles[triangleId].Area;
        probabilities.push_back(std::move(probability));
      }
      int pruneColumn = -1;
      double smallestSupport = 1e100;
      if (labels.size() > 2 &&
          resolvePass + 1 < config.HeatMaximumResolvePasses) {
        for (int column = 0; column < int(labels.size()); ++column) {
          double areaRatio = ownedArea[column] /
                             std::max(componentArea, 1e-30);
          if (ownedTriangles[column] >=
                  config.HeatMinimumSeedRegionTriangles &&
              areaRatio >= config.HeatMinimumSeedRegionAreaRatio)
            continue;
          double support = ownedTriangles[column] + areaRatio;
          if (support < smallestSupport) {
            smallestSupport = support;
            pruneColumn = column;
          }
        }
      }
      if (pruneColumn >= 0) {
        labels.erase(labels.begin() + pruneColumn);
        ++report.PrunedThermalSeeds;
        continue;
      }
      ++report.SolvedComponents;
      for (int index = 0; index < int(componentTriangles.size()); ++index) {
        int triangleId = componentTriangles[index];
        int patchId = triangles[triangleId].PatchId;
        const auto &probability = probabilities[index];
        int best = -1, second = -1;
        for (int column = 0; column < int(probability.size()); ++column) {
          if (best < 0 || probability[column] > probability[best]) {
            second = best;
            best = column;
          } else if (second < 0 || probability[column] > probability[second])
            second = column;
        }
        double margin =
            probability[best] - (second < 0 ? 0.0 : probability[second]);
        if (probability[best] >= config.HeatAssignmentConfidence &&
            margin >= config.HeatAssignmentMargin && labels[best] != patchId &&
            canTransferTriangle(triangleId, labels[best])) {
          destination[triangleId] = labels[best];
          ++report.ReassignedTriangles;
        }
      }
      break;
    }
    for (int patchId : componentIds)
      componentPatch[patchId] = 0;
  }

  if (!report.ReassignedTriangles)
    return report;

  // Treat all transfers connected through a donor/recipient as one
  // transaction. Otherwise rolling back a recipient could leave its donor
  // evaluated against a different set of triangles than the one committed.
  std::vector<int> transactionParent(patchCount);
  std::iota(transactionParent.begin(), transactionParent.end(), 0);
  auto transaction = [&](int id) {
    while (transactionParent[id] != id) {
      transactionParent[id] = transactionParent[transactionParent[id]];
      id = transactionParent[id];
    }
    return id;
  };
  std::vector<char> receiving(patchCount, 0), invalid(patchCount, 0);
  for (int triangleId = 0; triangleId < int(triangles.size()); ++triangleId)
    if (destination[triangleId] >= 0) {
      int source = triangles[triangleId].PatchId;
      int target = destination[triangleId];
      transactionParent[transaction(source)] = transaction(target);
      receiving[target] = 1;
    }
  auto proposedOwner = [&](int triangleId) {
    return destination[triangleId] >= 0 ? destination[triangleId]
                                        : triangles[triangleId].PatchId;
  };
  for (const auto &edge : edges)
    if (IsHardSegmentationBoundary(edge, config))
      for (size_t i = 0; i < edge.IncidentTriangleIds.size(); ++i)
        for (size_t j = i + 1; j < edge.IncidentTriangleIds.size(); ++j) {
          int a = edge.IncidentTriangleIds[i], b = edge.IncidentTriangleIds[j];
          if (triangles[a].PatchId != triangles[b].PatchId &&
              proposedOwner(a) == proposedOwner(b))
            invalid[transaction(proposedOwner(a))] = 1;
        }
  std::vector<std::vector<int>> proposedTriangles(patchCount);
  for (int triangleId = 0; triangleId < int(triangles.size()); ++triangleId)
    if (receiving[proposedOwner(triangleId)])
      proposedTriangles[proposedOwner(triangleId)].push_back(triangleId);
  std::vector<SurfaceFitResult> acceptedFits(patchCount);
  for (int target = 0; target < patchCount; ++target)
    if (receiving[target]) {
      if (!IsSurfaceCompatible(EvaluateSurfaceCompatibility(
                                   mesh, proposedTriangles[target], patches[target]),
                               mesh.getResolution(),
                               config.HeatSeedDemotionResidualTolerance)) {
        invalid[transaction(target)] = 1;
        continue;
      }
      auto fit = SurfaceModelSelector::fitBest(mesh, proposedTriangles[target],
                                               mesh.getResolution(),
                                               config.ModelComplexityPenalty);
      MeshPatch fitted = patches[target];
      ApplyFit(fitted, fit);
      if (!IsSurfaceCompatible(
              EvaluateSurfaceCompatibility(mesh, proposedTriangles[target], fitted),
              mesh.getResolution(), config.HeatSeedDemotionResidualTolerance)) {
        invalid[transaction(target)] = 1;
        continue;
      }
      acceptedFits[target] = std::move(fit);
    }
  report.ReassignedTriangles = 0;
  for (int triangleId = 0; triangleId < int(triangles.size()); ++triangleId)
    if (destination[triangleId] >= 0) {
      if (invalid[transaction(destination[triangleId])])
        destination[triangleId] = -1;
      else
        ++report.ReassignedTriangles;
    }
  if (!report.ReassignedTriangles)
    return report;

  std::vector<int> resultingPatch(triangles.size(), -1);
  std::vector<char> changed(patchCount, 0), seen(triangles.size(), 0);
  for (int triangleId = 0; triangleId < int(triangles.size()); ++triangleId) {
    int source = triangles[triangleId].PatchId;
    int target =
        destination[triangleId] >= 0 ? destination[triangleId] : source;
    resultingPatch[triangleId] = target;
    if (source != target) {
      changed[source] = 1;
      changed[target] = 1;
    }
  }
  std::vector<MeshPatch> rebuilt;
  rebuilt.reserve(patches.size());
  for (int seed = 0; seed < int(triangles.size()); ++seed)
    if (!seen[seed]) {
      int owner = resultingPatch[seed];
      MeshPatch patch = patches[owner];
      patch.TriangleIds.clear();
      std::queue<int> queue;
      queue.push(seed);
      seen[seed] = 1;
      while (!queue.empty()) {
        int triangleId = queue.front();
        queue.pop();
        patch.TriangleIds.push_back(triangleId);
        for (int neighbor : mesh.getTriangleNeighbors(triangleId))
          if (!seen[neighbor] && resultingPatch[neighbor] == owner) {
            seen[neighbor] = 1;
            queue.push(neighbor);
          }
      }
      if (changed[owner] && !(patch.SurfaceType == PatchSurfaceType::Freeform &&
                              patch.TriangleIds.size() > 100000)) {
        auto fit = SurfaceModelSelector::fitBest(mesh, patch.TriangleIds,
                                                 mesh.getResolution(),
                                                 config.ModelComplexityPenalty);
        if (receiving[owner]) {
          MeshPatch candidate = patch;
          ApplyFit(candidate, fit);
          if (!IsSurfaceCompatible(EvaluateSurfaceCompatibility(
                                       mesh, patch.TriangleIds, candidate),
                                   mesh.getResolution(),
                                   config.HeatSeedDemotionResidualTolerance))
            fit = acceptedFits[owner];
        }
        ApplyFit(patch, fit);
      }
      rebuilt.push_back(std::move(patch));
    }
  patches = std::move(rebuilt);
  segmenter.AssignPatchIds();
  return report;
}
} // namespace CadMesh
