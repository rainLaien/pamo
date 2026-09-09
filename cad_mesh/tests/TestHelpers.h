#pragma once

#include "CadMesh/CadMeshPatchSegmenter.h"
#include <algorithm>
#include <iterator>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace CadMeshTests {

inline void Require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

inline CadMesh::TriangleSoup Cube() {
  CadMesh::TriangleSoup soup;
  soup.Vertices = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                   {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  soup.Triangles = {{{0, 2, 1}}, {{0, 3, 2}}, {{4, 5, 6}}, {{4, 6, 7}},
                    {{0, 1, 5}}, {{0, 5, 4}}, {{1, 2, 6}}, {{1, 6, 5}},
                    {{2, 3, 7}}, {{2, 7, 6}}, {{3, 0, 4}}, {{3, 4, 7}}};
  return soup;
}

// The single triangle at the clipped corner is a real CAD face, not noise.
inline CadMesh::TriangleSoup ChamferedCube() {
  CadMesh::TriangleSoup soup;
  soup.Vertices = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                   {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, .8},
                   {1, .8, 1}, {.8, 1, 1}};
  soup.Triangles = {
      {{0, 3, 2}}, {{0, 2, 1}}, {{0, 1, 5}}, {{0, 5, 4}},
      {{0, 4, 6}}, {{0, 6, 3}}, {{1, 2, 7}}, {{1, 7, 8}},
      {{1, 8, 5}}, {{3, 6, 9}}, {{3, 9, 7}}, {{3, 7, 2}},
      {{4, 5, 8}}, {{4, 8, 9}}, {{4, 9, 6}}, {{7, 9, 8}}};
  return soup;
}

inline bool IsManifoldConnected(const CadMesh::MeshTopology &mesh,
                                const CadMesh::MeshPatch &patch) {
  if (patch.TriangleIds.empty())
    return false;
  std::set<int> members(patch.TriangleIds.begin(), patch.TriangleIds.end());
  std::set<int> visited{patch.TriangleIds.front()};
  std::queue<int> pending;
  pending.push(patch.TriangleIds.front());
  while (!pending.empty()) {
    int triangleId = pending.front();
    pending.pop();
    for (int edgeId : mesh.getTriangles()[triangleId].EdgeIds) {
      const auto &edge = mesh.getEdges()[edgeId];
      // Sharing a non-manifold edge does not make a smooth surface patch.
      if (edge.IsNonManifold || edge.IsBoundary ||
          edge.IncidentTriangleIds.size() != 2)
        continue;
      for (int neighbor : edge.IncidentTriangleIds)
        if (members.count(neighbor) && visited.insert(neighbor).second)
          pending.push(neighbor);
    }
  }
  return visited.size() == members.size();
}

inline std::set<int> UniqueIds(const std::vector<int> &ids, int upperBound,
                               const std::string &context) {
  std::set<int> result;
  for (int id : ids) {
    Require(id >= 0 && id < upperBound, context + ": invalid id");
    Require(result.insert(id).second, context + ": duplicate id");
  }
  return result;
}

// Reconstruct public remesh relationships from edge incidence, independently
// of the exported patch graph and triangle labels.
inline void AssertPartitionInvariants(
    const CadMesh::CadMeshPatchSegmenter &segmenter,
    const std::string &context = "partition") {
  const auto &mesh = segmenter.getMesh();
  const auto &triangles = mesh.getTriangles();
  const auto &edges = mesh.getEdges();
  const auto &patches = segmenter.getPatches();
  const auto &constraints = segmenter.getRemeshConstraint();
  std::vector<int> membershipCount(triangles.size(), 0);
  for (size_t patchIndex = 0; patchIndex < patches.size(); ++patchIndex) {
    const auto &patch = patches[patchIndex];
    std::string label = context + ": patch " + std::to_string(patchIndex);
    Require(patch.Id == int(patchIndex), label + " has a stale id");
    auto members = UniqueIds(patch.TriangleIds, int(triangles.size()), label);
    Require(!members.empty(), label + " is empty");
    for (int triangleId : members) {
      ++membershipCount[triangleId];
      Require(triangles[triangleId].PatchId == int(patchIndex),
              label + " disagrees with triangle " +
                  std::to_string(triangleId) + " label");
    }
    Require(IsManifoldConnected(mesh, patch),
            label + " is disconnected across manifold edges");
  }
  for (size_t triangleId = 0; triangleId < triangles.size(); ++triangleId) {
    Require(membershipCount[triangleId] == 1,
            context + ": triangle " + std::to_string(triangleId) +
                " must belong to exactly one patch (found " +
                std::to_string(membershipCount[triangleId]) + ")");
    Require(triangles[triangleId].PatchId >= 0 &&
                triangles[triangleId].PatchId < int(patches.size()),
            context + ": triangle has invalid patch label");
  }

  std::set<int> expectedConstraints;
  std::set<int> expectedHardFeatures, expectedSurfaceTransitions;
  std::vector<std::set<int>> expectedBoundaries(patches.size());
  std::vector<std::set<int>> expectedNeighbors(patches.size());
  std::map<std::pair<int, int>, std::set<int>> expectedAdjacency;
  std::vector<int> constraintDegree(mesh.getVertices().size(), 0);
  for (size_t edgeId = 0; edgeId < edges.size(); ++edgeId) {
    const auto &edge = edges[edgeId];
    Require(edge.Vertex0 >= 0 && edge.Vertex0 < int(mesh.getVertices().size()) &&
                edge.Vertex1 >= 0 &&
                edge.Vertex1 < int(mesh.getVertices().size()),
            context + ": edge references an invalid vertex");
    auto incident = UniqueIds(edge.IncidentTriangleIds, int(triangles.size()),
                              context + ": edge incidence");
    Require(!incident.empty(), context + ": edge has no incident triangles");
    std::set<int> owners;
    for (int triangleId : incident) {
      const auto &triangle = triangles[triangleId];
      Require(std::find(triangle.EdgeIds.begin(), triangle.EdgeIds.end(),
                        int(edgeId)) != triangle.EdgeIds.end(),
              context + ": edge incidence disagrees with triangle");
      owners.insert(triangle.PatchId);
    }
    if (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature ||
        owners.size() > 1) {
      expectedConstraints.insert(int(edgeId));
      if (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature)
        expectedHardFeatures.insert(int(edgeId));
      else
        expectedSurfaceTransitions.insert(int(edgeId));
      ++constraintDegree[edge.Vertex0];
      ++constraintDegree[edge.Vertex1];
      for (int owner : owners)
        expectedBoundaries[owner].insert(int(edgeId));
    }
    for (auto a = owners.begin(); a != owners.end(); ++a)
      for (auto b = std::next(a); b != owners.end(); ++b) {
        expectedAdjacency[{*a, *b}].insert(int(edgeId));
        expectedNeighbors[*a].insert(*b);
        expectedNeighbors[*b].insert(*a);
      }
  }
  Require(UniqueIds(constraints.ConstraintEdgeIds, int(edges.size()),
                    context + ": constraint edges") == expectedConstraints,
          context + ": remesh constraints omit or invent a protected edge");
  Require(UniqueIds(constraints.HardFeatureEdgeIds, int(edges.size()),
                    context + ": hard feature edges") == expectedHardFeatures,
          context + ": hard feature export disagrees with geometric edge protection");
  Require(UniqueIds(constraints.SurfaceTransitionEdgeIds, int(edges.size()),
                    context + ": smooth surface transitions") == expectedSurfaceTransitions,
          context + ": smooth transitions were confused with hard feature edges");
  for (size_t patchIndex = 0; patchIndex < patches.size(); ++patchIndex) {
    Require(UniqueIds(patches[patchIndex].BoundaryEdgeIds, int(edges.size()),
                      context + ": patch boundaries") ==
                expectedBoundaries[patchIndex],
            context + ": patch boundaries disagree with edge ownership");
    Require(UniqueIds(patches[patchIndex].NeighborPatchIds, int(patches.size()),
                      context + ": patch neighbors") ==
                expectedNeighbors[patchIndex],
            context + ": patch neighbor graph is incomplete or asymmetric");
  }
  std::map<std::pair<int, int>, std::set<int>> actualAdjacency;
  for (const auto &adjacency : segmenter.getAdjacency()) {
    Require(adjacency.Patch0 >= 0 && adjacency.Patch0 < adjacency.Patch1 &&
                adjacency.Patch1 < int(patches.size()),
            context + ": adjacency references invalid patch ids");
    auto key = std::make_pair(adjacency.Patch0, adjacency.Patch1);
    Require(actualAdjacency
                .emplace(key, UniqueIds(adjacency.SharedBoundaryEdges,
                                        int(edges.size()),
                                        context + ": adjacency edges"))
                .second,
            context + ": duplicate patch adjacency");
  }
  Require(actualAdjacency == expectedAdjacency,
          context + ": adjacency disagrees with incident patch pairs");

  std::vector<std::set<int>> vertexOwners(mesh.getVertices().size());
  for (const auto &triangle : triangles)
    for (int vertexId : triangle.VertexIds) {
      Require(vertexId >= 0 && vertexId < int(vertexOwners.size()),
              context + ": triangle references an invalid vertex");
      vertexOwners[vertexId].insert(triangle.PatchId);
    }
  std::set<int> expectedJunctions;
  for (size_t vertexId = 0; vertexId < vertexOwners.size(); ++vertexId)
    if (vertexOwners[vertexId].size() >= 3 || constraintDegree[vertexId] > 2)
      expectedJunctions.insert(int(vertexId));
  Require(UniqueIds(constraints.JunctionVertexIds, int(vertexOwners.size()),
                    context + ": junction vertices") == expectedJunctions,
          context + ": junction vertices disagree with patch incidence");
}

} // namespace CadMeshTests
