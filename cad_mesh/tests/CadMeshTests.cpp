#include "TestHelpers.h"
#include "CadMesh/StlReader.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace CadMesh;
using namespace CadMeshTests;

void RunPatchGraphTests();
void RunGeometryGuardTests();
void RunAnalyticModelTests();
void RunSurfaceRoleTests();
void RunModelFirstTests();
void RunFilletSeamTests();

namespace {

std::vector<std::string>
PartitionSignature(const CadMeshPatchSegmenter &segmenter) {
  std::vector<std::string> signature;
  const auto &mesh = segmenter.getMesh();
  for (const auto &patch : segmenter.getPatches()) {
    std::vector<std::string> faceKeys;
    for (int triangleId : patch.TriangleIds) {
      std::vector<std::string> vertexKeys;
      for (int vertexId : mesh.getTriangles()[triangleId].VertexIds) {
        const auto &p = mesh.getVertices()[vertexId].Position;
        std::ostringstream key;
        key << std::setprecision(17) << p.X() << ',' << p.Y() << ',' << p.Z();
        vertexKeys.push_back(key.str());
      }
      std::sort(vertexKeys.begin(), vertexKeys.end());
      faceKeys.push_back(vertexKeys[0] + ";" + vertexKeys[1] + ";" +
                         vertexKeys[2]);
    }
    std::sort(faceKeys.begin(), faceKeys.end());
    std::string key = SurfaceTypeName(patch.SurfaceType);
    for (const auto &face : faceKeys)
      key += "|" + face;
    signature.push_back(std::move(key));
  }
  std::sort(signature.begin(), signature.end());
  return signature;
}

void AssertPlanes(const CadMeshPatchSegmenter &segmenter, size_t count,
                  const std::string &context) {
  AssertPartitionInvariants(segmenter, context);
  Require(segmenter.getPatches().size() == count,
          context + ": wrong number of geometric planes");
  for (const auto &patch : segmenter.getPatches())
    Require(patch.SurfaceType == PatchSurfaceType::Plane,
            context + ": a geometric plane was misclassified");
}

void TestCube() {
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(Cube()), "cube segmentation failed");
  AssertPlanes(segmenter, 6, "cube");
}

void TestCleanup() {
  auto soup = Cube();
  soup.Triangles.push_back(soup.Triangles.front());
  soup.Triangles.push_back({{0, 0, 1}});
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), "cleanup segmentation failed");
  const auto &report = segmenter.getMesh().getCleanupReport();
  Require(report.DuplicateTriangles == 1, "duplicate triangle was not removed");
  Require(report.DegenerateTriangles == 1,
          "degenerate triangle was not removed");
  Require(report.OutputTriangles == 12, "cleanup removed a valid cube face");
  AssertPlanes(segmenter, 6, "cleaned cube");
}

void TestUnnamedBody() {
  std::filesystem::path source = CADMESH_SOURCE_DIR;
  auto input = source.parent_path() / "examples" / "Unnamed-Body.stl";
  TriangleSoup soup;
  std::string error;
  Require(StlReader::read(input, soup, error),
          "could not read Unnamed-Body.stl: " + error);
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), "Unnamed-Body segmentation failed");
  AssertPartitionInvariants(segmenter, "Unnamed-Body");
  Require(segmenter.getPatches().size() == 8,
          "Unnamed-Body must produce eight patches");
  int planes = 0, cylinders = 0;
  std::vector<size_t> planeSizes;
  for (const auto &patch : segmenter.getPatches()) {
    planes += patch.SurfaceType == PatchSurfaceType::Plane;
    cylinders += patch.SurfaceType == PatchSurfaceType::Cylinder;
    if (patch.SurfaceType == PatchSurfaceType::Plane)
      planeSizes.push_back(patch.TriangleIds.size());
    if (patch.SurfaceType == PatchSurfaceType::Cylinder) {
      Require(patch.TriangleIds.size() == 64,
              "Unnamed-Body fillets must retain all 64 triangles, including tangent strips");
      Require(std::abs(std::get<CylinderParameters>(patch.Parameters).Radius - 3) < 5e-5,
              "Unnamed-Body fillet radius was replaced by a fictitious large cylinder");
    }
  }
  Require(planes == 6, "Unnamed-Body must contain six planes");
  Require(cylinders == 2, "Unnamed-Body must contain two cylindrical fillets");
  std::sort(planeSizes.begin(), planeSizes.end());
  Require(planeSizes == std::vector<size_t>{2, 2, 2, 2, 66, 66},
          "Unnamed-Body planar boundaries consumed part of a tangent fillet");
}

void CheckChamferedCube(SegmentationConfig config, const std::string &context) {
  CadMeshPatchSegmenter segmenter(config);
  Require(segmenter.segment(ChamferedCube()), context + ": segmentation failed");
  AssertPlanes(segmenter, 7, context);
  Require(segmenter.getMesh().getTriangles().size() == 16,
          context + ": a valid triangle was removed");
  const auto &mesh = segmenter.getMesh();
  Require(mesh.getCleanupReport().BoundaryEdges == 0 &&
              mesh.getCleanupReport().NonManifoldEdges == 0,
          context + ": fixture must remain a closed manifold");
  int chamferTriangle = mesh.getOriginalToCleanTriangleMap().at(15);
  Require(chamferTriangle >= 0, context + ": chamfer was removed in cleanup");
  const auto &patch = segmenter.getPatches().at(
      mesh.getTriangles().at(chamferTriangle).PatchId);
  Require(patch.TriangleIds.size() == 1 &&
              patch.TriangleIds.front() == chamferTriangle,
          context + ": single-triangle CAD chamfer must remain independent");
}

void TestChamferedCube() { CheckChamferedCube({}, "chamfered cube"); }

void TestChamferWithoutExhaustiveRefinement() {
  SegmentationConfig config;
  config.MaximumExhaustiveRefinementTriangles = 0;
  CheckChamferedCube(config, "chamfer without exhaustive refinement");
}

void TestOpenTriangle() {
  TriangleSoup soup;
  soup.Vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  soup.Triangles = {{{0, 1, 2}}};
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), "open triangle segmentation failed");
  AssertPlanes(segmenter, 1, "open triangle");
  Require(segmenter.getRemeshConstraint().ConstraintEdgeIds.size() == 3,
          "open triangle must preserve all three boundary edges");
}

void TestVertexTouchingSheets() {
  TriangleSoup soup;
  soup.Vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                   {-1, 0, 0}, {0, -1, 0}};
  soup.Triangles = {{{0, 1, 2}}, {{0, 3, 4}}};
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), "vertex-touching sheets segmentation failed");
  AssertPlanes(segmenter, 2, "vertex-touching sheets");
  Require(segmenter.getAdjacency().empty(),
          "a shared vertex must not invent an edge adjacency");
}

void TestFaceOrderInvariance() {
  auto soup = ChamferedCube();
  CadMeshPatchSegmenter reference;
  Require(reference.segment(soup), "reference chamfer segmentation failed");
  AssertPlanes(reference, 7, "reference face order");
  auto expected = PartitionSignature(reference);
  for (int permutation = 0; permutation < 3; ++permutation) {
    if (permutation == 0)
      std::reverse(soup.Triangles.begin(), soup.Triangles.end());
    else
      std::rotate(soup.Triangles.begin(), soup.Triangles.begin() + 5,
                  soup.Triangles.end());
    for (auto &triangle : soup.Triangles)
      std::rotate(triangle.begin(), triangle.begin() + 1, triangle.end());
    CadMeshPatchSegmenter segmenter;
    Require(segmenter.segment(soup), "reordered chamfer segmentation failed");
    AssertPlanes(segmenter, 7, "reordered chamfer");
    Require(PartitionSignature(segmenter) == expected,
            "triangle order changed geometric patch membership");
  }
}

void TestRepeatedSegmentation() {
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(Cube()), "initial cube segmentation failed");
  AssertPlanes(segmenter, 6, "initial repeated cube");
  auto expected = PartitionSignature(segmenter);
  Require(segmenter.segment(ChamferedCube()), "reused chamfer segmentation failed");
  AssertPlanes(segmenter, 7, "reused chamfer");
  Require(segmenter.segment(Cube()), "reused cube segmentation failed");
  AssertPlanes(segmenter, 6, "reused cube");
  Require(PartitionSignature(segmenter) == expected,
          "reusing the segmenter changed the cube partition");
}

void TestValidationRejectsStaleLabel() {
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(Cube()), "validation fixture segmentation failed");
  std::string error;
  Require(segmenter.validatePartition(&error),
          "valid partition rejected by public validator: " + error);
  segmenter.getMesh().getTriangles().front().PatchId = -1;
  Require(!segmenter.validatePartition(&error) && !error.empty(),
          "public validator must report a stale triangle label");
  Require(segmenter.segment(Cube()), "segmenter did not recover after reuse");
  AssertPlanes(segmenter, 6, "recovered validation fixture");
}

} // namespace

int main() {
  const std::pair<const char *, void (*)()> tests[] = {
      {"cube", TestCube},
      {"cleanup", TestCleanup},
      {"Unnamed-Body", TestUnnamedBody},
      {"single-triangle chamfer", TestChamferedCube},
      {"chamfer without exhaustive refinement",
       TestChamferWithoutExhaustiveRefinement},
      {"open triangle", TestOpenTriangle},
      {"vertex-touching sheets", TestVertexTouchingSheets},
      {"face order invariance", TestFaceOrderInvariance},
      {"repeated segmentation", TestRepeatedSegmentation},
      {"stale label validation", TestValidationRejectsStaleLabel},
      {"patch graph", RunPatchGraphTests},
      {"geometry guards", RunGeometryGuardTests},
      {"analytic models", RunAnalyticModelTests},
      {"surface roles and output", RunSurfaceRoleTests},
      {"model-first segmentation", RunModelFirstTests},
      {"fillet seam consolidation", RunFilletSeamTests}};
  int failures = 0;
  for (const auto &test : tests) {
    try {
      test.second();
      std::cout << "PASS: " << test.first << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
    }
  }
  if (failures) {
    std::cerr << "CadMesh tests failed: " << failures << '\n';
    return 1;
  }
  std::cout << "CadMesh tests passed\n";
  return 0;
}
