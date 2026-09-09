#include "CadMesh/DebugVisualizer.h"
#include "TestHelpers.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace {
using namespace CadMesh;
using CadMeshTests::AssertPartitionInvariants;
using CadMeshTests::Require;

TriangleSoup Rectangle(double scale = 1) {
  TriangleSoup soup;
  soup.Vertices = {
      {0, 0, 0}, {2 * scale, 0, 0}, {2 * scale, scale, 0}, {0, scale, 0}};
  soup.Triangles = {{{0, 1, 2}}, {{0, 2, 3}}};
  return soup;
}
TriangleSoup Annulus() {
  TriangleSoup soup;
  const int count = 32;
  for (double radius : {1.0, 2.0})
    for (int i = 0; i < count; ++i) {
      double angle = 2 * std::acos(-1.0) * i / count;
      soup.Vertices.push_back(
          {radius * std::cos(angle), radius * std::sin(angle), 0});
    }
  for (int i = 0; i < count; ++i) {
    int j = (i + 1) % count;
    soup.Triangles.push_back({{i, i + count, j + count}});
    soup.Triangles.push_back({{i, j + count, j}});
  }
  return soup;
}
TriangleSoup Cylinder() {
  TriangleSoup soup;
  const int count = 32;
  for (double z : {0.0, 1.0, 2.0})
    for (int i = 0; i < count; ++i) {
      double angle = 2 * std::acos(-1.0) * i / count;
      soup.Vertices.push_back({std::cos(angle), std::sin(angle), z});
    }
  for (int layer = 0; layer < 2; ++layer)
    for (int i = 0; i < count; ++i) {
      int j = (i + 1) % count;
      int a = layer * count + i, b = layer * count + j;
      soup.Triangles.push_back({{a, b, b + count}});
      soup.Triangles.push_back({{a, b + count, a + count}});
    }
  return soup;
}

void AssertChains(const CadMeshPatchSegmenter &s) {
  const auto &constraint = s.getRemeshConstraint();
  const auto &mesh = s.getMesh();
  std::set<int> found;
  std::vector<std::set<int>> references(constraint.BoundaryChains.size());
  for (const auto &patch : s.getPatches()) {
    std::set<int> patchChainIds;
    for (const auto &ref : patch.BoundaryChainRefs) {
      Require(ref.ChainId >= 0 && ref.ChainId < int(references.size()),
              "invalid chain reference");
      Require(patchChainIds.insert(ref.ChainId).second,
              "duplicate patch chain reference");
      references[ref.ChainId].insert(patch.Id);
      if (ref.Status == BoundaryDirectionStatus::Consistent)
        Require(ref.Direction == 1 || ref.Direction == -1,
                "consistent direction must be signed");
      else
        Require(ref.Direction == 0,
                "unresolved orientation must not invent a direction");
      const auto &chain = constraint.BoundaryChains[ref.ChainId];
      if (ref.Direction != 0) {
        for (size_t i = 0; i < chain.EdgeIds.size(); ++i) {
          const auto &edge = mesh.getEdges()[chain.EdgeIds[i]];
          int count = 0;
          for (int ti : edge.IncidentTriangleIds) {
            const auto &triangle = mesh.getTriangles()[ti];
            if (triangle.PatchId != patch.Id)
              continue;
            ++count;
            int from = chain.VertexIds[i], to = chain.VertexIds[i + 1];
            if (ref.Direction < 0)
              std::swap(from, to);
            bool followsFace = false;
            for (int k = 0; k < 3; ++k)
              followsFace =
                  followsFace || (triangle.VertexIds[k] == from &&
                                  triangle.VertexIds[(k + 1) % 3] == to);
            Require(followsFace,
                    "directed chain disagrees with incident triangle winding");
          }
          Require(count == 1,
                  "directed chain must have exactly one patch side");
        }
      }
    }
  }
  for (size_t i = 0; i < constraint.BoundaryChains.size(); ++i) {
    const auto &chain = constraint.BoundaryChains[i];
    Require(chain.Id == int(i), "chain ids must be dense");
    Require(!chain.EdgeIds.empty(), "empty boundary chain");
    Require(chain.VertexIds.size() == chain.EdgeIds.size() + 1,
            "chain edge/vertex lengths disagree");
    Require(chain.InitialSampleVertexIds == chain.VertexIds,
            "initial boundary samples must share original vertices");
    Require(chain.IsClosed ==
                (chain.VertexIds.front() == chain.VertexIds.back()),
            "wrong closed chain flag");
    std::set<int> owners(chain.IncidentPatchIds.begin(),
                         chain.IncidentPatchIds.end());
    Require(owners == references[i],
            "boundary chain missing an incident patch reference");
    for (size_t j = 0; j < chain.EdgeIds.size(); ++j) {
      int edgeId = chain.EdgeIds[j];
      Require(found.insert(edgeId).second,
              "shared edge duplicated in boundary chains");
      const auto &edge = mesh.getEdges().at(edgeId);
      std::set<int> endpoints{edge.Vertex0, edge.Vertex1};
      Require(endpoints ==
                  std::set<int>{chain.VertexIds[j], chain.VertexIds[j + 1]},
              "unordered/disconnected chain");
      std::set<int> edgeOwners;
      for (int ti : edge.IncidentTriangleIds)
        edgeOwners.insert(mesh.getTriangles()[ti].PatchId);
      Require(edgeOwners == owners, "chain crosses an incidence change");
    }
  }
  Require(found == std::set<int>(constraint.ConstraintEdgeIds.begin(),
                                 constraint.ConstraintEdgeIds.end()),
          "boundary chains must cover every constraint edge exactly once");
}

void TestCornersAndSharedChains() {
  for (double scale : {.001, 1.0, 1000.0}) {
    CadMeshPatchSegmenter s;
    Require(s.segment(Rectangle(scale)), "open rectangle segmentation failed");
    AssertPartitionInvariants(s, "open rectangle");
    AssertChains(s);
    Require(s.getPatches().size() == 1, "rectangle must have one plane");
    Require(s.getRemeshConstraint().CornerVertexIds.size() == 4,
            "open planar corners must be protected at every scale");
    Require(s.getRemeshConstraint().BoundaryChains.size() == 4,
            "rectangle boundaries must split at its four corners");
  }
  CadMeshPatchSegmenter cube;
  Require(cube.segment(CadMeshTests::Cube()), "cube segmentation failed");
  AssertChains(cube);
  Require(cube.getRemeshConstraint().BoundaryChains.size() == 12,
          "cube must share twelve boundary chains");
  for (const auto &chain : cube.getRemeshConstraint().BoundaryChains) {
    Require(chain.IncidentPatchIds.size() == 2,
            "cube chain must be shared by two patches");
    int sum = 0;
    for (int patchId : chain.IncidentPatchIds)
      for (const auto &ref : cube.getPatches()[patchId].BoundaryChainRefs)
        if (ref.ChainId == chain.Id)
          sum += ref.Direction;
    Require(sum == 0, "two sides of a consistent manifold boundary must have "
                      "opposite direction");
  }
}
void TestSmoothLoopsAndPeriodicAssessment() {
  CadMeshPatchSegmenter annulus;
  Require(annulus.segment(Annulus()), "annulus segmentation failed");
  AssertPartitionInvariants(annulus, "annulus");
  AssertChains(annulus);
  Require(annulus.getPatches().size() == 1,
          "annulus must remain one planar patch with a hole");
  Require(annulus.getRemeshConstraint().BoundaryChains.size() == 2,
          "annulus must have inner and outer loops");
  Require(annulus.getRemeshConstraint().CornerVertexIds.empty(),
          "smooth polygonal loops must not invent corners");
  for (const auto &chain : annulus.getRemeshConstraint().BoundaryChains)
    Require(chain.IsClosed, "annulus boundaries must close");
  CadMeshPatchSegmenter cylinder;
  Require(cylinder.segment(Cylinder()), "cylinder segmentation failed");
  AssertPartitionInvariants(cylinder, "periodic cylinder");
  AssertChains(cylinder);
  bool foundCylinder = false;
  for (const auto &patch : cylinder.getPatches())
    if (patch.SurfaceType == PatchSurfaceType::Cylinder) {
      foundCylinder = true;
      Require(patch.NeedsParameterizationSeamAssessment,
              "periodic cylinder requires seam assessment");
      Require(patch.ProjectionTarget == PatchProjectionTarget::AnalyticSurface,
              "cylinder projection target missing");
    }
  Require(foundCylinder, "cylinder test must exercise an analytic cylinder");
}
void TestNonManifoldIncidence() {
  TriangleSoup soup;
  soup.Vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, -1, -1}};
  soup.Triangles = {{{0, 1, 2}}, {{0, 1, 3}}, {{0, 1, 4}}};
  CadMeshPatchSegmenter s;
  Require(s.segment(soup), "non-manifold fan segmentation failed");
  AssertPartitionInvariants(s, "three-sided non-manifold edge");
  AssertChains(s);
  bool found = false;
  for (const auto &chain : s.getRemeshConstraint().BoundaryChains)
    if (chain.IsNonManifold) {
      found = true;
      Require(chain.IncidentPatchIds.size() == 3,
              "non-manifold chain must include all three patches");
      for (int patchId : chain.IncidentPatchIds)
        for (const auto &ref : s.getPatches()[patchId].BoundaryChainRefs)
          if (ref.ChainId == chain.Id)
            Require(ref.Status == BoundaryDirectionStatus::NonManifold &&
                        ref.Direction == 0,
                    "non-manifold chain must explicitly leave orientation "
                    "unresolved");
    }
  Require(found, "non-manifold test did not produce its common edge");
}
void TestUnresolvedWindingAndInternalFeatures() {
  CadMeshPatchSegmenter s;
  Require(s.segment(Rectangle()), "rectangle segmentation failed");
  auto &mesh = s.getMesh();
  for (auto &edge : mesh.getEdges())
    if (!edge.IsBoundary)
      edge.IsConstrainedFeature = true;
  PatchGraphBuilder::build(s);
  AssertChains(s);
  bool internal = false;
  for (const auto &ref : s.getPatches()[0].BoundaryChainRefs)
    internal =
        internal || ref.Status == BoundaryDirectionStatus::MultiplePatchSides;
  Require(
      internal,
      "same-patch protected edge must survive with multiple sides unresolved");
  Require(s.getRemeshConstraint().ConstraintEdgeIds.size() == 5,
          "internal protected edge omitted");
  Require(s.segment(Rectangle()), "rectangle reset failed");
  auto &triangle = s.getMesh().getTriangles()[1];
  std::swap(triangle.VertexIds[1], triangle.VertexIds[2]);
  PatchGraphBuilder::build(s);
  Require(!s.getPatches()[0].HasConsistentFaceOrientation,
          "inconsistent face winding went undetected");
  for (const auto &ref : s.getPatches()[0].BoundaryChainRefs)
    Require(
        ref.Direction == 0 &&
            ref.Status == BoundaryDirectionStatus::InconsistentWinding,
        "inconsistent patch winding must not produce invented chain direction");
}
void TestInteriorFeatureEndpoints() {
  TriangleSoup soup;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
      soup.Vertices.push_back({double(x), double(y), 0});
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x) {
      int a = y * 4 + x;
      soup.Triangles.push_back({{a, a + 1, a + 5}});
      soup.Triangles.push_back({{a, a + 5, a + 4}});
    }
  CadMeshPatchSegmenter s;
  Require(s.segment(soup), "planar grid segmentation failed");
  Require(s.getPatches().size() == 1, "planar grid must remain one patch");
  for (auto &edge : s.getMesh().getEdges())
    if (edge.Vertex0 == 5 && edge.Vertex1 == 10)
      edge.IsConstrainedFeature = true;
  PatchGraphBuilder::build(s);
  AssertChains(s);
  std::set<int> endpoints;
  for (const auto &corner : s.getRemeshConstraint().Corners)
    if (corner.IsEndpoint)
      endpoints.insert(corner.VertexId);
  Require(endpoints == std::set<int>{5, 10},
          "open internal feature chain must retain both endpoints");
}
void TestHandoffExport() {
  CadMeshPatchSegmenter s;
  Require(s.segment(Rectangle()), "export rectangle segmentation failed");
  auto path = std::filesystem::temp_directory_path() /
              "cadmesh_patch_graph_test_report.json";
  Require(DebugVisualizer::exportReportJson(s, path), "handoff export failed");
  std::ifstream input(path);
  std::string report((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
  for (const char *field :
       {"\"schema_version\": 1", "\"partition_valid\":true", "\"triangle_ids\"",
        "\"parameters\":{\"origin\"", "\"constraint_edges\"",
        "\"initial_sample_vertex_ids\"", "\"orientation_status\"",
        "\"seam_generated\":false"})
    Require(report.find(field) != std::string::npos,
            std::string("handoff export omits ") + field);
  input.close();
  std::error_code error;
  std::filesystem::remove(path, error);
}
} // namespace

void RunPatchGraphTests() {
  TestCornersAndSharedChains();
  TestSmoothLoopsAndPeriodicAssessment();
  TestNonManifoldIncidence();
  TestUnresolvedWindingAndInternalFeatures();
  TestInteriorFeatureEndpoints();
  TestHandoffExport();
}
