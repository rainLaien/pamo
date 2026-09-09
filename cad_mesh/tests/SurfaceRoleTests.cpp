#include "CadMesh/DebugVisualizer.h"
#include "CadMesh/SurfaceRoles.h"
#include "TestHelpers.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>

namespace {
using namespace CadMesh;
using CadMeshTests::Require;

struct RoleFixture {
  MeshTopology Mesh;
  std::vector<MeshPatch> Patches;
  std::vector<PatchAdjacency> Adjacency;
};

void BuildIncidence(RoleFixture &fixture, const std::vector<int> &labels) {
  auto &mesh = fixture.Mesh;
  fixture.Patches.assign(*std::max_element(labels.begin(), labels.end()) + 1, {});
  fixture.Adjacency.clear();
  for (int i = 0; i < int(fixture.Patches.size()); ++i)
    fixture.Patches[i].Id = i;
  for (size_t i = 0; i < labels.size(); ++i) {
    mesh.getTriangles()[i].PatchId = labels[i];
    fixture.Patches[labels[i]].TriangleIds.push_back(int(i));
  }
  std::map<std::pair<int, int>, PatchAdjacency> adjacency;
  for (size_t i = 0; i < mesh.getEdges().size(); ++i) {
    auto &edge = mesh.getEdges()[i];
    std::set<int> owners;
    for (int triangle : edge.IncidentTriangleIds)
      owners.insert(mesh.getTriangles()[triangle].PatchId);
    if (owners.size() > 1 || edge.IsBoundary)
      for (int owner : owners)
        fixture.Patches[owner].BoundaryEdgeIds.push_back(int(i));
    if (owners.size() == 2) {
      int a = *owners.begin(), b = *owners.rbegin();
      auto &adj = adjacency[{a, b}];
      adj.Patch0 = a;
      adj.Patch1 = b;
      adj.SharedBoundaryEdges.push_back(int(i));
    }
  }
  for (const auto &item : adjacency)
    fixture.Adjacency.push_back(item.second);
}

RoleFixture FilletBand(double scale = 1, int axialSegments = 2) {
  TriangleSoup soup;
  const int arc = 24;
  std::vector<Point3> section{{1, -2, 0}};
  for (int i = 0; i <= arc; ++i) {
    double theta = .5 * std::acos(-1.0) * i / arc;
    section.push_back({std::cos(theta), std::sin(theta), 0});
  }
  section.push_back({-2, 1, 0});
  int stride = int(section.size());
  for (int row = 0; row <= axialSegments; ++row)
    for (const auto &point : section)
      soup.Vertices.push_back({scale * point.X(), scale * point.Y(),
                               scale * 8.0 * row / axialSegments});
  std::vector<int> labels;
  for (int row = 0; row < axialSegments; ++row)
    for (int i = 0; i < stride - 1; ++i) {
      int a = row * stride + i, b = a + 1;
      soup.Triangles.push_back({a, b, b + stride});
      soup.Triangles.push_back({a, b + stride, a + stride});
      int label = i == 0 ? 0 : i == stride - 2 ? 2 : 1;
      labels.push_back(label);
      labels.push_back(label);
    }
  RoleFixture fixture;
  Require(fixture.Mesh.build(soup), "fillet band mesh failed");
  BuildIncidence(fixture, labels);
  fixture.Patches[0].SurfaceType = PatchSurfaceType::Plane;
  fixture.Patches[0].Parameters = PlaneParameters{{{scale, 0, 0}, {1, 0, 0}}};
  fixture.Patches[1].SurfaceType = PatchSurfaceType::Cylinder;
  fixture.Patches[1].Parameters = CylinderParameters{{{0, 0, 0}, {0, 0, 1}}, scale};
  fixture.Patches[2].SurfaceType = PatchSurfaceType::Plane;
  fixture.Patches[2].Parameters = PlaneParameters{{{0, scale, 0}, {0, 1, 0}}};
  return fixture;
}

RoleFixture CappedCylinder() {
  TriangleSoup soup;
  const int count = 32;
  for (double z : {0.0, 3.0})
    for (int i = 0; i < count; ++i) {
      double theta = 2 * std::acos(-1.0) * i / count;
      soup.Vertices.push_back({std::cos(theta), std::sin(theta), z});
    }
  soup.Vertices.push_back({0, 0, 0});
  soup.Vertices.push_back({0, 0, 3});
  std::vector<int> labels;
  for (int i = 0; i < count; ++i) {
    int j = (i + 1) % count;
    soup.Triangles.push_back({i, j, j + count});
    soup.Triangles.push_back({i, j + count, i + count});
    soup.Triangles.push_back({2 * count, j, i});
    soup.Triangles.push_back({2 * count + 1, i + count, j + count});
    labels.insert(labels.end(), {1, 1, 0, 2});
  }
  RoleFixture fixture;
  Require(fixture.Mesh.build(soup), "capped cylinder mesh failed");
  BuildIncidence(fixture, labels);
  fixture.Patches[0].SurfaceType = fixture.Patches[2].SurfaceType =
      PatchSurfaceType::Plane;
  fixture.Patches[0].Parameters = PlaneParameters{{{0, 0, 0}, {0, 0, -1}}};
  fixture.Patches[2].Parameters = PlaneParameters{{{0, 0, 3}, {0, 0, 1}}};
  fixture.Patches[1].SurfaceType = PatchSurfaceType::Cylinder;
  fixture.Patches[1].Parameters = CylinderParameters{{{0, 0, 0}, {0, 0, 1}}, 1};
  return fixture;
}

RoleFixture TorusBand() {
  TriangleSoup soup;
  const int count = 64, arc = 16;
  const double major = 3, minor = .3;
  std::vector<std::pair<double, double>> section{{major + minor, -1.2}};
  for (int i = 0; i <= arc; ++i) {
    double phi = .5 * std::acos(-1.0) * i / arc;
    section.push_back({major + minor * std::cos(phi), minor * std::sin(phi)});
  }
  section.push_back({major - 1.2, minor});
  for (const auto &point : section)
    for (int i = 0; i < count; ++i) {
      double theta = 2 * std::acos(-1.0) * i / count;
      soup.Vertices.push_back({point.first * std::cos(theta),
                               point.first * std::sin(theta), point.second});
    }
  std::vector<int> labels;
  for (int row = 0; row < int(section.size()) - 1; ++row)
    for (int i = 0; i < count; ++i) {
      int a = row * count + i, b = row * count + (i + 1) % count;
      soup.Triangles.push_back({a, b, b + count});
      soup.Triangles.push_back({a, b + count, a + count});
      int label = row == 0 ? 0 : row == int(section.size()) - 2 ? 2 : 1;
      labels.insert(labels.end(), {label, label});
    }
  RoleFixture fixture;
  Require(fixture.Mesh.build(soup), "toroidal fillet band mesh failed");
  BuildIncidence(fixture, labels);
  fixture.Patches[0].SurfaceType = PatchSurfaceType::Cylinder;
  fixture.Patches[0].Parameters =
      CylinderParameters{{{0, 0, 0}, {0, 0, 1}}, major + minor};
  fixture.Patches[1].SurfaceType = PatchSurfaceType::Torus;
  fixture.Patches[1].Parameters =
      TorusParameters{{{0, 0, 0}, {0, 0, 1}}, major, minor};
  fixture.Patches[2].SurfaceType = PatchSurfaceType::Plane;
  fixture.Patches[2].Parameters = PlaneParameters{{{0, 0, minor}, {0, 0, 1}}};
  return fixture;
}

void TestFilletRoleAndConservativeRejection() {
  for (double scale : {.001, 1.0, 1000.0}) {
    auto fixture = FilletBand(scale);
    const auto originalTriangles = fixture.Patches[1].TriangleIds;
    IdentifySurfaceRoles(fixture.Mesh, fixture.Patches, fixture.Adjacency);
    Require(fixture.Patches[1].FeatureRole == PatchFeatureRole::Fillet,
            "continuous tangent quarter-cylinder band must be recognized as a fillet");
    Require(fixture.Patches[1].SupportPatchIds == std::vector<int>({0, 2}),
            "fillet must name both mother surfaces");
    Require(fixture.Patches[1].SurfaceType == PatchSurfaceType::Cylinder &&
                fixture.Patches[1].TriangleIds == originalTriangles,
            "role annotation must preserve surface type and ownership");
    Require(fixture.Patches[0].FeatureRole == PatchFeatureRole::Ordinary,
            "a flat mother surface is not a fillet");
    fixture.Patches[1].SurfaceType = PatchSurfaceType::Freeform;
    fixture.Patches[1].Parameters = std::monostate{};
    IdentifySurfaceRoles(fixture.Mesh, fixture.Patches, fixture.Adjacency);
    Require(fixture.Patches[1].FeatureRole == PatchFeatureRole::Fillet &&
                fixture.Patches[1].SurfaceType == PatchSurfaceType::Freeform,
            "freeform transition role must remain independent of analytic type");
    for (int edge : fixture.Adjacency.front().SharedBoundaryEdges)
      fixture.Mesh.getEdges()[edge].IsConstrainedFeature = true;
    IdentifySurfaceRoles(fixture.Mesh, fixture.Patches, fixture.Adjacency);
    Require(fixture.Patches[1].FeatureRole == PatchFeatureRole::Ordinary &&
                fixture.Patches[1].SupportPatchIds.empty(),
            "sharp-side band must not retain a stale fillet role");
  }
  auto cylinder = CappedCylinder();
  IdentifySurfaceRoles(cylinder.Mesh, cylinder.Patches, cylinder.Adjacency);
  Require(cylinder.Patches[1].FeatureRole == PatchFeatureRole::Ordinary,
          "ordinary cylinder wall with planar end caps is not a fillet");
  auto torus = TorusBand();
  IdentifySurfaceRoles(torus.Mesh, torus.Patches, torus.Adjacency);
  Require(torus.Patches[1].FeatureRole == PatchFeatureRole::Fillet &&
              torus.Patches[1].SupportPatchIds == std::vector<int>({0, 2}),
          "closed toroidal blend must use local transverse normals without cancellation");
  auto unresolved = FilletBand();
  unresolved.Patches[0].SurfaceType = PatchSurfaceType::Freeform;
  unresolved.Patches[0].Parameters = std::monostate{};
  IdentifySurfaceRoles(unresolved.Mesh, unresolved.Patches, unresolved.Adjacency);
  Require(unresolved.Patches[1].FeatureRole == PatchFeatureRole::Ordinary,
          "unconfirmed mother surfaces must not invent a fillet role");
}

void TestFragmentedMotherPlanesDoNotSplitFilletEvidence() {
  for (double scale : {.001, 1.0, 1000.0}) {
    auto fixture = FilletBand(scale, 8);
    const auto models = fixture.Patches;
    std::vector<int> labels;
    for (const auto &face : fixture.Mesh.getTriangles()) {
      const double z = face.Centroid.Z() / scale;
      int id = face.PatchId;
      if (id == 0)
        id = z < 4 ? 0 : 3;
      else if (id == 2)
        id = z < 2 ? 2 : z < 4 ? 4 : z < 6 ? 5 : 6;
      labels.push_back(id);
    }
    BuildIncidence(fixture, labels);
    for (int id = 0; id < int(fixture.Patches.size()); ++id) {
      const auto &model = models[id == 1 ? 1 : (id == 0 || id == 3 ? 0 : 2)];
      fixture.Patches[id].SurfaceType = model.SurfaceType;
      fixture.Patches[id].Parameters = model.Parameters;
    }
    const auto originalFaces = fixture.Patches[1].TriangleIds;
    IdentifySurfaceRoles(fixture.Mesh, fixture.Patches, fixture.Adjacency);
    Require(fixture.Patches[1].FeatureRole == PatchFeatureRole::Fillet,
            "two fragments of one mother plane must not occupy both support sides");
    Require(fixture.Patches[1].SupportPatchIds == std::vector<int>({0, 2}),
            "fillet evidence must name one adjacent representative per mother plane");
    Require(fixture.Patches[1].TriangleIds == originalFaces,
            "mother grouping must not change fillet face ownership");
    // With only one resolved physical mother, arbitrarily many labels still
    // cannot establish the two-sided evidence needed for a fillet.
    for (int id : {2, 4, 5, 6}) {
      fixture.Patches[id].SurfaceType = PatchSurfaceType::Freeform;
      fixture.Patches[id].Parameters = std::monostate{};
    }
    IdentifySurfaceRoles(fixture.Mesh, fixture.Patches, fixture.Adjacency);
    Require(fixture.Patches[1].FeatureRole == PatchFeatureRole::Ordinary &&
                fixture.Patches[1].SupportPatchIds.empty(),
            "one fragmented mother surface must not invent a fillet");
  }
}

std::string Read(const std::filesystem::path &path) {
  std::ifstream file(path);
  return {(std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()};
}
void TestRoleAndSurfaceExports() {
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(CadMeshTests::Cube()), "role export cube failed");
  auto path = std::filesystem::temp_directory_path() / "cadmesh_surface_roles_test";
  std::error_code error;
  std::filesystem::create_directories(path, error);
  Require(DebugVisualizer::exportPatchPly(segmenter, path / "patches.ply") &&
              DebugVisualizer::exportSurfaceTypePly(segmenter, path / "types.ply") &&
              DebugVisualizer::exportFeatureRolePly(segmenter, path / "roles.ply") &&
              DebugVisualizer::exportReportJson(segmenter, path / "report.json"),
          "surface/role debug exports failed");
  auto types = Read(path / "types.ply");
  auto roles = Read(path / "roles.ply");
  Require(types.find("comment color_by surface_type") != std::string::npos &&
              roles.find("comment color_by feature_role") != std::string::npos &&
              types.find("property int feature_role") != std::string::npos,
          "debug PLY must distinguish coloring from surface ownership");
  auto json = Read(path / "report.json");
  for (const char *field : {"\"feature_role\":\"Ordinary\"", "\"support_patch_ids\":[]",
                            "\"hard_feature_edge_ids\"", "\"smooth_surface_transition_edge_ids\"",
                            "\"boundary_kind\"", "\"boundary_chain_refs\""})
    Require(json.find(field) != std::string::npos,
            std::string("semantic handoff export omits ") + field);
  for (const char *file : {"patches.ply", "types.ply", "roles.ply", "report.json"})
    std::filesystem::remove(path / file, error);
  std::filesystem::remove(path, error);
}

std::string PatchReportRecord(const std::string &json, int patchId) {
  std::string marker = "    {\"id\":" + std::to_string(patchId) + ",\"type\":\"";
  auto start = json.find(marker);
  Require(start != std::string::npos, "patch missing from error diagnostic report");
  auto end = json.find('\n', start);
  return json.substr(start, end == std::string::npos ? end : end - start);
}

double ReportNumberAfter(const std::string &record, const std::string &key) {
  auto start = record.find(key);
  Require(start != std::string::npos, "numeric diagnostic missing: " + key);
  auto value = std::stod(record.substr(start + key.size()));
  Require(std::isfinite(value), "exported diagnostic must be finite");
  return value;
}

void TestComputedDiagnosticsAndMeshDeviation() {
  auto path = std::filesystem::temp_directory_path() /
              ("cadmesh_diagnostic_semantics_" + std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()));
  CadMeshPatchSegmenter firstExport;
  Require(firstExport.segment(CadMeshTests::Cube()), "fresh diagnostic fixture failed");
  Require(DebugVisualizer::exportAll(firstExport, path),
          "exporting to a fresh directory must not fail when stale curvature files are absent");
  SegmentationConfig legacyConfig;
  legacyConfig.EnableModelFirst = false;
  CadMeshPatchSegmenter legacy(legacyConfig);
  Require(legacy.segment(CadMeshTests::Cube()), "legacy diagnostic fixture failed");
  Require(legacy.hasComputedDifferentialGeometry() && legacy.hasComputedBoundaryScores(),
          "legacy segmentation must report its measured diagnostics");
  Require(DebugVisualizer::exportAll(legacy, path), "legacy diagnostic export failed");
  auto legacyJson = Read(path / "patch_report.json");
  Require(legacyJson.find("\"differential_geometry\":{\"computed\":true}") != std::string::npos &&
              legacyJson.find("\"boundary_scores\":{\"computed\":true}") != std::string::npos,
          "legacy report must mark differential geometry and boundary scores computed");
  Require(legacyJson.find("\"boundary_score\":null") == std::string::npos &&
              legacyJson.find("\"confidence\":null") == std::string::npos,
          "legacy measured edge and adjacency diagnostics must remain available");
  for (const auto &patch : legacy.getPatches()) {
    auto record = PatchReportRecord(legacyJson, patch.Id);
    Require(record.find("\"sampled_mesh_deviation\":{\"computed\":false,\"maximum\":null") != std::string::npos,
            "legacy must not report the uncomputed mesh deviation as zero");
  }
  auto legacyBoundary = Read(path / "boundary_score.vtk");
  Require(legacyBoundary.find("SCALARS boundary_score ") != std::string::npos &&
              legacyBoundary.find("SCALARS curvature_gradient ") != std::string::npos,
          "legacy measured VTK diagnostic arrays are missing");
  for (const char *file : {"mean_curvature.vtk", "gaussian_curvature.vtk", "k1.vtk", "k2.vtk"})
    Require(std::filesystem::exists(path / file), "legacy curvature VTK missing");

  // All wall vertices lie on an exact cylinder. Its flat facets still depart
  // from that analytic surface between vertices, so the two errors must differ.
  auto fixture = CappedCylinder();
  CadMeshPatchSegmenter modelFirst;
  Require(modelFirst.segment(fixture.Mesh.getOriginalSoup()), "cylinder diagnostic fixture failed");
  Require(!modelFirst.hasComputedDifferentialGeometry() && !modelFirst.hasComputedBoundaryScores(),
          "model-first mode must not claim skipped local diagnostics were measured");
  Require(DebugVisualizer::exportAll(modelFirst, path), "model-first diagnostic export failed");
  auto json = Read(path / "patch_report.json");
  Require(json.find("\"differential_geometry\":{\"computed\":false}") != std::string::npos &&
              json.find("\"boundary_scores\":{\"computed\":false}") != std::string::npos,
          "model-first report must declare unavailable local diagnostics");
  Require(json.find("\"boundary_score\":null") != std::string::npos &&
              json.find("\"confidence\":null") != std::string::npos,
          "uncomputed edge and adjacency diagnostics must be null");
  bool foundCylinder = false;
  for (const auto &patch : modelFirst.getPatches()) {
    if (patch.ProjectionTarget != PatchProjectionTarget::AnalyticSurface)
      continue;
    auto record = PatchReportRecord(json, patch.Id);
    const std::string sampledKey = "\"sampled_mesh_deviation\":{\"computed\":true,\"maximum\":";
    const double vertexMaximum = ReportNumberAfter(record, "\"max\":");
    const double sampledMaximum = ReportNumberAfter(record, sampledKey);
    Require(sampledMaximum >= vertexMaximum &&
                std::abs(sampledMaximum - patch.MaxSampledSurfaceDeviation) < 1e-12,
            "exported sampled deviation must contain the vertex maximum and match the final patch");
    Require(record.find("\"fitting_error_semantics\":\"all_patch_vertices_area_weighted_rms_and_vertex_max\"") != std::string::npos &&
                record.find("\"hausdorff_upper_bound\":false") != std::string::npos &&
                record.find("\"sampling\":\"all_triangle_vertices_edge_midpoints_and_centroids\"") != std::string::npos,
            "diagnostics must distinguish full-member vertex fit from sampled mesh deviation");
    if (patch.SurfaceType == PatchSurfaceType::Cylinder) {
      foundCylinder = true;
      Require(vertexMaximum < 1e-8 && sampledMaximum > 1e-3,
              "exact cylinder vertices must not conceal the measurable flat-facet deviation");
    }
  }
  Require(foundCylinder, "diagnostic test must exercise the curved wall");
  auto boundary = Read(path / "boundary_score.vtk");
  Require(boundary.find("scores_computed=0") != std::string::npos &&
              boundary.find("SCALARS hard_feature ") != std::string::npos &&
              boundary.find("SCALARS smooth_surface_transition ") != std::string::npos,
          "uncomputed local scores must not suppress measured boundary topology");
  for (const char *field : {"SCALARS boundary_score ", "SCALARS normal_discontinuity ",
                            "SCALARS curvature_discontinuity ", "SCALARS curvature_gradient ",
                            "SCALARS surface_fit_discontinuity ", "SCALARS tessellation_evidence "})
    Require(boundary.find(field) == std::string::npos,
            std::string("VTK falsely reports an uncomputed diagnostic: ") + field);
  for (const char *file : {"mean_curvature.vtk", "gaussian_curvature.vtk", "k1.vtk", "k2.vtk"})
    Require(!std::filesystem::exists(path / file),
            "refreshing the output directory must remove stale legacy curvature VTKs");
  Require(!DebugVisualizer::exportCurvatureVtk(modelFirst, path / "uncomputed.vtk", "k1", 2) &&
              !std::filesystem::exists(path / "uncomputed.vtk"),
          "direct curvature export must reject unavailable diagnostics without creating a file");
  std::error_code error;
  for (const char *file : {"patch_result.ply", "surface_types.ply", "feature_roles.ply",
                           "boundary_score.vtk", "patch_report.json"})
    std::filesystem::remove(path / file, error);
  std::filesystem::remove(path, error);
}
} // namespace

void RunSurfaceRoleTests() {
  TestFilletRoleAndConservativeRejection();
  TestFragmentedMotherPlanesDoNotSplitFilletEvidence();
  TestRoleAndSurfaceExports();
  TestComputedDiagnosticsAndMeshDeviation();
}
