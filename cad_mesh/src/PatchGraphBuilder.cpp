#include "CadMesh/CadMeshPatchSegmenter.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace CadMesh {
namespace {
int TriangleEdgeDirection(const MeshTriangle &triangle, int a, int b) {
  for (int i = 0; i < 3; ++i) {
    int from = triangle.VertexIds[i], to = triangle.VertexIds[(i + 1) % 3];
    if (from == a && to == b)
      return 1;
    if (from == b && to == a)
      return -1;
  }
  return 0;
}

bool HasAnalyticParameters(const MeshPatch &patch) {
  switch (patch.SurfaceType) {
  case PatchSurfaceType::Plane:
    return std::holds_alternative<PlaneParameters>(patch.Parameters);
  case PatchSurfaceType::Cylinder:
    return std::holds_alternative<CylinderParameters>(patch.Parameters);
  case PatchSurfaceType::Cone:
    return std::holds_alternative<ConeParameters>(patch.Parameters);
  case PatchSurfaceType::Sphere:
    return std::holds_alternative<SphereParameters>(patch.Parameters);
  case PatchSurfaceType::Torus:
    return std::holds_alternative<TorusParameters>(patch.Parameters);
  default:
    return false;
  }
}

int OtherVertex(const MeshEdge &edge, int vertex) {
  return edge.Vertex0 == vertex ? edge.Vertex1 : edge.Vertex0;
}

DirectedBoundaryChain OrientForPatch(const MeshTopology &mesh,
                                     const MeshPatch &patch,
                                     const BoundaryChain &chain) {
  DirectedBoundaryChain result;
  result.ChainId = chain.Id;
  if (chain.IsNonManifold) {
    result.Status = BoundaryDirectionStatus::NonManifold;
    return result;
  }
  if (chain.HasInconsistentWinding || !patch.HasConsistentFaceOrientation) {
    result.Status = BoundaryDirectionStatus::InconsistentWinding;
    return result;
  }
  int direction = 0;
  for (size_t i = 0; i < chain.EdgeIds.size(); ++i) {
    const auto &edge = mesh.getEdges()[chain.EdgeIds[i]];
    int count = 0, current = 0;
    for (int ti : edge.IncidentTriangleIds) {
      const auto &triangle = mesh.getTriangles()[ti];
      if (triangle.PatchId != patch.Id)
        continue;
      ++count;
      current = TriangleEdgeDirection(triangle, chain.VertexIds[i],
                                      chain.VertexIds[i + 1]);
    }
    if (count != 1 || current == 0) {
      result.Status = count > 1 ? BoundaryDirectionStatus::MultiplePatchSides
                                : BoundaryDirectionStatus::MissingIncidence;
      return result;
    }
    if (direction != 0 && direction != current) {
      result.Status = BoundaryDirectionStatus::InconsistentWinding;
      return result;
    }
    direction = current;
  }
  result.Direction = direction;
  result.Status = BoundaryDirectionStatus::Consistent;
  return result;
}
} // namespace

void PatchGraphBuilder::build(CadMeshPatchSegmenter &s) {
  auto &mesh = s.mMesh;
  auto &constraint = s.mConstraint;
  const auto &edges = mesh.getEdges();
  const auto &triangles = mesh.getTriangles();
  for (auto &patch : s.mPatches) {
    patch.BoundaryEdgeIds.clear();
    patch.NeighborPatchIds.clear();
    patch.BoundaryChainRefs.clear();
    patch.HasConsistentFaceOrientation = true;
    patch.ProjectionTarget = HasAnalyticParameters(patch)
                                 ? PatchProjectionTarget::AnalyticSurface
                                 : PatchProjectionTarget::ReferenceMesh;
    patch.NeedsParameterizationSeamAssessment =
        patch.ProjectionTarget == PatchProjectionTarget::AnalyticSurface &&
        patch.SurfaceType != PatchSurfaceType::Plane;
  }
  s.mAdjacency.clear();
  constraint = {};
  std::map<std::pair<int, int>, PatchAdjacency> graph;
  std::vector<std::vector<int>> edgePatches(edges.size());
  std::vector<std::vector<int>> vertexEdges(mesh.getVertices().size());
  std::vector<std::set<int>> vertexPatches(mesh.getVertices().size());
  std::vector<char> inconsistent(edges.size(), false);
  for (const auto &triangle : triangles)
    if (triangle.PatchId >= 0 && triangle.PatchId < int(s.mPatches.size()))
      for (int vertex : triangle.VertexIds)
        vertexPatches[vertex].insert(triangle.PatchId);

  for (int edgeId = 0; edgeId < int(edges.size()); ++edgeId) {
    const auto &edge = edges[edgeId];
    auto &patches = edgePatches[edgeId];
    for (int ti : edge.IncidentTriangleIds) {
      int patchId = triangles[ti].PatchId;
      if (patchId >= 0 && patchId < int(s.mPatches.size()))
        patches.push_back(patchId);
    }
    std::sort(patches.begin(), patches.end());
    patches.erase(std::unique(patches.begin(), patches.end()), patches.end());
    if (edge.IncidentTriangleIds.size() == 2) {
      int a = TriangleEdgeDirection(triangles[edge.IncidentTriangleIds[0]],
                                    edge.Vertex0, edge.Vertex1);
      int b = TriangleEdgeDirection(triangles[edge.IncidentTriangleIds[1]],
                                    edge.Vertex0, edge.Vertex1);
      inconsistent[edgeId] = a == 0 || b == 0 || a == b;
      if (inconsistent[edgeId])
        for (int patchId : patches)
          s.mPatches[patchId].HasConsistentFaceOrientation = false;
    }
    if (!(edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature ||
          patches.size() > 1))
      continue;
    constraint.ConstraintEdgeIds.push_back(edgeId);
    if (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature)
      constraint.HardFeatureEdgeIds.push_back(edgeId);
    else
      constraint.SurfaceTransitionEdgeIds.push_back(edgeId);
    vertexEdges[edge.Vertex0].push_back(edgeId);
    vertexEdges[edge.Vertex1].push_back(edgeId);
    for (int patchId : patches)
      s.mPatches[patchId].BoundaryEdgeIds.push_back(edgeId);
    // A non-manifold edge can have three or more incident patches. Preserve
    // every pair; consumers must use the chain's non-manifold flag to decide
    // whether/how to repair or remesh that incidence.
    for (size_t i = 0; i < patches.size(); ++i)
      for (size_t j = i + 1; j < patches.size(); ++j) {
        auto &adj = graph[{patches[i], patches[j]}];
        adj.Patch0 = patches[i];
        adj.Patch1 = patches[j];
        adj.SharedBoundaryEdges.push_back(edgeId);
      }
  }
  for (auto &item : graph) {
    auto &adj = item.second;
    for (int edgeId : adj.SharedBoundaryEdges)
      adj.BoundaryConfidence += edges[edgeId].BoundaryScore;
    adj.BoundaryConfidence /= double(adj.SharedBoundaryEdges.size());
    s.mPatches[adj.Patch0].NeighborPatchIds.push_back(adj.Patch1);
    s.mPatches[adj.Patch1].NeighborPatchIds.push_back(adj.Patch0);
    s.mAdjacency.push_back(std::move(adj));
  }

  std::vector<char> stop(mesh.getVertices().size(), false);
  const double angleLimit =
      std::max(0.0, std::min(std::acos(-1.0), s.mConfig.BoundaryCornerAngle));
  constraint.CornerAngleThreshold = angleLimit;
  for (int vertex = 0; vertex < int(vertexEdges.size()); ++vertex) {
    const auto &incident = vertexEdges[vertex];
    BoundaryCorner corner;
    corner.VertexId = vertex;
    corner.IsEndpoint = incident.size() == 1;
    corner.IsJunction =
        incident.size() > 2 || vertexPatches[vertex].size() >= 3;
    if (corner.IsJunction)
      constraint.JunctionVertexIds.push_back(vertex);
    if (incident.empty())
      continue;
    if (incident.size() == 2) {
      int a = incident[0], b = incident[1];
      corner.HasIncidenceChange =
          edgePatches[a] != edgePatches[b] ||
          edges[a].IsNonManifold != edges[b].IsNonManifold ||
          edges[a].IsBoundary != edges[b].IsBoundary ||
          edges[a].IsConstrainedFeature != edges[b].IsConstrainedFeature;
      Vec3 position = ToVec(mesh.getVertices()[vertex].Position);
      Vec3 u =
          Sub(ToVec(mesh.getVertices()[OtherVertex(edges[a], vertex)].Position),
              position);
      Vec3 v =
          Sub(ToVec(mesh.getVertices()[OtherVertex(edges[b], vertex)].Position),
              position);
      double product = Norm(u) * Norm(v);
      if (product > 0) {
        double turn =
            std::acos(std::max(-1.0, std::min(1.0, -Dot(u, v) / product)));
        corner.IsSharpCorner = turn > angleLimit;
      }
    }
    stop[vertex] = incident.size() != 2 || corner.IsJunction ||
                   corner.HasIncidenceChange || corner.IsSharpCorner;
    if (stop[vertex]) {
      constraint.CornerVertexIds.push_back(vertex);
      constraint.Corners.push_back(corner);
    }
  }

  std::vector<char> visited(edges.size(), false);
  auto trace = [&](int firstEdge, int firstVertex) {
    BoundaryChain chain;
    chain.Id = int(constraint.BoundaryChains.size());
    chain.IncidentPatchIds = edgePatches[firstEdge];
    chain.IsHardFeature = edges[firstEdge].IsBoundary ||
                          edges[firstEdge].IsNonManifold ||
                          edges[firstEdge].IsConstrainedFeature;
    chain.VertexIds.push_back(firstVertex);
    int edgeId = firstEdge, vertex = firstVertex;
    while (!visited[edgeId]) {
      visited[edgeId] = true;
      chain.EdgeIds.push_back(edgeId);
      chain.IsNonManifold = chain.IsNonManifold || edges[edgeId].IsNonManifold;
      chain.HasInconsistentWinding =
          chain.HasInconsistentWinding || inconsistent[edgeId];
      vertex = OtherVertex(edges[edgeId], vertex);
      chain.VertexIds.push_back(vertex);
      if (vertex == firstVertex) {
        chain.IsClosed = true;
        break;
      }
      if (stop[vertex])
        break;
      const auto &next = vertexEdges[vertex];
      int candidate = next[0] == edgeId ? next[1] : next[0];
      if (visited[candidate])
        break;
      edgeId = candidate;
    }
    chain.InitialSampleVertexIds = chain.VertexIds;
    for (int patchId : chain.IncidentPatchIds)
      s.mPatches[patchId].BoundaryChainRefs.push_back(
          OrientForPatch(mesh, s.mPatches[patchId], chain));
    constraint.BoundaryChains.push_back(std::move(chain));
  };
  // Trace corner-to-corner chains first, then the remaining smooth loops.
  for (int vertex = 0; vertex < int(vertexEdges.size()); ++vertex)
    if (stop[vertex])
      for (int edgeId : vertexEdges[vertex])
        if (!visited[edgeId])
          trace(edgeId, vertex);
  for (int edgeId : constraint.ConstraintEdgeIds)
    if (!visited[edgeId])
      trace(edgeId, edges[edgeId].Vertex0);
  for (auto &patch : s.mPatches)
    std::sort(patch.NeighborPatchIds.begin(), patch.NeighborPatchIds.end());
}
} // namespace CadMesh
