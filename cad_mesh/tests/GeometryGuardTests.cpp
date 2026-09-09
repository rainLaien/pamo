#include "CadMesh/BoundaryScoreCalculator.h"
#include "CadMesh/CadMeshPatchSegmenter.h"
#include "CadMesh/HeatPatchRegularizer.h"
#include "CadMesh/SegmentationGuards.h"
#include "CadMesh/SurfaceFitting.h"
#include <numeric>
#include <stdexcept>

using namespace CadMesh;
namespace {
void RequireGeometry(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

TriangleSoup Grid(int nx, int ny, double scale = 1.0) {
  TriangleSoup soup;
  for (int y = 0; y <= ny; ++y)
    for (int x = 0; x <= nx; ++x)
      soup.Vertices.push_back({scale * x, scale * y, 0});
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      int a = y * (nx + 1) + x, b = a + 1, c = a + nx + 1, d = c + 1;
      soup.Triangles.push_back({a, b, d});
      soup.Triangles.push_back({a, d, c});
    }
  return soup;
}

MeshPatch Plane() {
  MeshPatch patch;
  patch.SurfaceType = PatchSurfaceType::Plane;
  patch.Parameters = PlaneParameters{{{0, 0, 0}, {0, 0, 1}}};
  patch.Confidence = 1;
  return patch;
}

void TestFixedSurfaceAcceptance() {
  MeshTopology mesh;
  TriangleSoup soup;
  soup.Vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1e-4},
                   {0, 0, .1}, {1, 0, .1}, {0, 1, .1}};
  soup.Triangles = {{0, 1, 2}, {0, 1, 3}, {4, 5, 6}};
  RequireGeometry(mesh.build(soup), "surface guard fixture failed");
  auto plane = Plane();
  RequireGeometry(IsSurfaceCompatible(EvaluateSurfaceCompatibility(mesh, 0, plane),
                                       mesh.getResolution()),
                  "coplanar triangle rejected");
  auto chamfer = EvaluateSurfaceCompatibility(mesh, 1, plane);
  RequireGeometry(chamfer.MaxNormalizedDistance < 3 &&
                      !IsSurfaceCompatible(chamfer, mesh.getResolution()),
                  "thin perpendicular chamfer accepted by distance alone");
  RequireGeometry(!IsSurfaceCompatible(EvaluateSurfaceCompatibility(mesh, 2, plane),
                                        mesh.getResolution()),
                  "offset parallel surface accepted");
  auto freeform = plane;
  freeform.SurfaceType = PatchSurfaceType::Freeform;
  RequireGeometry(!EvaluateSurfaceCompatibility(mesh, 0, freeform).Supported,
                  "Freeform must not certify analytic surface membership");

  SegmentationConfig config;
  MeshEdge edge;
  edge.IncidentTriangleIds = {0, 1};
  RequireGeometry(!IsHardSegmentationBoundary(edge, config), "smooth edge is hard");
  edge.IsConstrainedFeature = true;
  RequireGeometry(IsHardSegmentationBoundary(edge, config), "feature edge is not hard");
  edge.IsConstrainedFeature = false;
  edge.BoundaryScore = config.StrongBoundaryThreshold;
  RequireGeometry(IsHardSegmentationBoundary(edge, config), "strong score is not hard");
  edge.BoundaryScore = 0;
  edge.IsNonManifold = true;
  RequireGeometry(IsHardSegmentationBoundary(edge, config), "non-manifold edge is not hard");
}

void TestCurvatureGradientUnits() {
  std::vector<double> reference;
  for (double scale : {1.0, .001, 1000.0}) {
    MeshTopology mesh;
    RequireGeometry(mesh.build(Grid(6, 5, scale)), "scale fixture failed");
    for (auto &vertex : mesh.getVertices()) {
      double x = vertex.Position.X() / scale;
      vertex.Geometry.K1 = .001 * x * x / scale;
      vertex.Geometry.K2 = 0;
      vertex.Geometry.MeanCurvature = .5 * vertex.Geometry.K1;
      vertex.Geometry.GaussianCurvature = 0;
      vertex.Geometry.Confidence = 1;
    }
    SegmentationConfig config;
    config.NormalWeight = config.CurvatureWeight = config.SurfaceFitWeight =
        config.TessellationWeight = 0;
    config.GradientWeight = 1;
    BoundaryScoreCalculator(mesh, config).compute();
    std::vector<double> values;
    for (const auto &edge : mesh.getEdges())
      if (!edge.IsBoundary)
        values.push_back(edge.Evidence.CurvatureGradient);
    if (reference.empty()) {
      reference = values;
      RequireGeometry(std::any_of(values.begin(), values.end(), [](double value) {
                        return value > 1e-3 && value < .9;
                      }), "scale test has no unsaturated curvature evidence");
    } else {
      RequireGeometry(values.size() == reference.size(), "scaled topology changed");
      for (size_t i = 0; i < values.size(); ++i)
        RequireGeometry(std::abs(values[i] - reference[i]) < 1e-9,
                        "curvature boundary evidence depends on coordinate units");
    }
  }
}

void TestFreeformDiagnostic() {
  auto soup = Grid(4, 4);
  for (auto &vertex : soup.Vertices)
    vertex.Z() = .1 * vertex.X() * vertex.Y();
  MeshTopology mesh;
  RequireGeometry(mesh.build(soup), "freeform fixture failed");
  std::vector<int> ids(mesh.getTriangles().size());
  std::iota(ids.begin(), ids.end(), 0);
  PlaneSurfaceFitter plane;
  FreeformSurfaceFitter freeform;
  RequireGeometry(plane.fit(mesh, ids) && freeform.fit(mesh, ids),
                  "diagnostic fit failed");
  RequireGeometry(std::abs(freeform.computeRmsError() - plane.computeRmsError()) < 1e-12 &&
                      std::abs(freeform.computeMaxError() - plane.computeMaxError()) < 1e-12 &&
                      std::holds_alternative<std::monostate>(freeform.getParameters()),
                  "freeform fallback reports invented reconstruction accuracy");
}

// The regularizer operates on an intermediate partition. Populate that state
// directly so these regressions do not depend on earlier heuristic stages.
std::vector<MeshPatch> &FixturePatches(CadMeshPatchSegmenter &segmenter) {
  return const_cast<std::vector<MeshPatch> &>(segmenter.getPatches());
}

void PrepareHeatStrip(CadMeshPatchSegmenter &segmenter, bool raisedDonor,
                      bool invalidRecipient) {
  auto soup = Grid(10, 6);
  for (auto &vertex : soup.Vertices)
    if (raisedDonor && vertex.X() > 3 && vertex.X() < 7)
      vertex.Z() = .2;
    else if (invalidRecipient && vertex.X() > 8)
      vertex.Z() = .2;
  auto &mesh = segmenter.getMesh();
  RequireGeometry(mesh.build(soup), "heat strip fixture failed");
  auto &patches = FixturePatches(segmenter);
  patches = {Plane(), Plane(), {}};
  patches[2].SurfaceType = PatchSurfaceType::Freeform;
  for (int id = 0; id < int(mesh.getTriangles().size()); ++id) {
    double x = mesh.getTriangles()[id].Centroid.X();
    int owner = x < 3 ? 0 : x > 7 ? 1 : 2;
    patches[owner].TriangleIds.push_back(id);
    mesh.getTriangles()[id].PatchId = owner;
  }
}

void TestHeatGeometryAndRollback() {
  CadMeshPatchSegmenter flat;
  PrepareHeatStrip(flat, false, false);
  auto flatReport = HeatPatchRegularizer::regularize(flat);
  RequireGeometry(flatReport.ReassignedTriangles > 0,
                  "heat geometry guards disable compatible smoothing");

  CadMeshPatchSegmenter fillet;
  PrepareHeatStrip(fillet, true, false);
  auto originalFillet = FixturePatches(fillet)[2].TriangleIds;
  auto filletReport = HeatPatchRegularizer::regularize(fillet);
  RequireGeometry(filletReport.ReassignedTriangles == 0 &&
                      FixturePatches(fillet)[2].TriangleIds == originalFillet,
                  "multi-label heat consumed a non-planar feature");

  CadMeshPatchSegmenter rollback;
  PrepareHeatStrip(rollback, false, true);
  auto originalDonor = FixturePatches(rollback)[2].TriangleIds;
  auto rollbackReport = HeatPatchRegularizer::regularize(rollback);
  RequireGeometry(rollbackReport.ReassignedTriangles == 0 &&
                      FixturePatches(rollback)[2].TriangleIds == originalDonor,
                  "failed whole-recipient check did not roll back related transfers");
}

void TestSingleLabelHardBoundary() {
  for (int protection : {0, 1, 2}) {
    CadMeshPatchSegmenter segmenter;
    auto &mesh = segmenter.getMesh();
    RequireGeometry(mesh.build(Grid(8, 8)), "heat island fixture failed");
    auto &patches = FixturePatches(segmenter);
    patches = {Plane(), {}};
    patches[1].SurfaceType = PatchSurfaceType::Freeform;
    for (int id = 0; id < int(mesh.getTriangles().size()); ++id) {
      const auto &center = mesh.getTriangles()[id].Centroid;
      int owner = center.X() > 3 && center.X() < 5 &&
                          center.Y() > 3 && center.Y() < 5 ? 1 : 0;
      patches[owner].TriangleIds.push_back(id);
      mesh.getTriangles()[id].PatchId = owner;
    }
    for (auto &edge : mesh.getEdges())
      if (edge.Triangle1 >= 0 &&
          mesh.getTriangles()[edge.Triangle0].PatchId !=
              mesh.getTriangles()[edge.Triangle1].PatchId) {
        edge.IsConstrainedFeature = protection == 1;
        edge.BoundaryScore = protection == 2 ? .9 : 0;
      }
    auto report = HeatPatchRegularizer::regularize(segmenter);
    RequireGeometry(protection == 0 ? report.ReassignedTriangles == 8
                                     : report.ReassignedTriangles == 0,
                    "single-label shortcut failed to respect protected boundary");
  }
}

void TestCylinderFacetContinuation() {
  // Each first angular strip is planar. Subdividing it along the cylinder
  // axis also supplies a sixteen-triangle planar/chamfer-face counterexample:
  // a complete face is outside the narrowly scoped seam adjudication.
  for (int axialSegments : {1, 8}) {
    TriangleSoup soup;
    constexpr int angularSegments = 9;
    constexpr double radius = 3, step = .05;
    for (int y = 0; y <= axialSegments; ++y)
      for (int a = 0; a <= angularSegments; ++a)
        soup.Vertices.push_back({radius * std::cos(a * step),
                                  double(y) / axialSegments,
                                  radius * std::sin(a * step)});
    std::vector<int> sourceIds, targetIds;
    for (int y = 0; y < axialSegments; ++y)
      for (int a = 0; a < angularSegments; ++a) {
        int v = y * (angularSegments + 1) + a;
        for (auto triangle : {std::array<int, 3>{v, v + 1, v + angularSegments + 2},
                              std::array<int, 3>{v, v + angularSegments + 2,
                                                 v + angularSegments + 1}}) {
          (a == 0 ? sourceIds : targetIds).push_back(int(soup.Triangles.size()));
          soup.Triangles.push_back(triangle);
        }
      }
    MeshTopology mesh;
    RequireGeometry(mesh.build(soup), "cylinder continuation fixture failed");
    MeshPatch source = Plane(), target;
    source.TriangleIds = sourceIds;
    target.SurfaceType = PatchSurfaceType::Cylinder;
    target.Parameters = CylinderParameters{{{0, 0, 0}, {0, 1, 0}}, radius};
    target.Confidence = 1;
    target.TriangleIds = targetIds;
    std::vector<int> shared;
    for (int id = 0; id < int(mesh.getEdges().size()); ++id) {
      const auto &edge = mesh.getEdges()[id];
      if (edge.Triangle1 < 0)
        continue;
      auto isSource = [&](int triangleId) {
        return std::find(sourceIds.begin(), sourceIds.end(), triangleId) != sourceIds.end();
      };
      if (isSource(edge.Triangle0) != isSource(edge.Triangle1))
        shared.push_back(id);
    }
    SegmentationConfig config;
    RequireGeometry(IsCylinderTessellationContinuation(mesh, source, target, shared, config) ==
                        (axialSegments == 1),
                    "cylinder continuation either rejected a facet or consumed a complete planar face");
    if (axialSegments != 1)
      continue;
    auto &edge = mesh.getEdges()[shared.front()];
    edge.BoundaryScore = .82;
    RequireGeometry(IsCylinderTessellationContinuation(mesh, source, target, shared, config),
                    "raw tessellation evidence cannot be geometrically adjudicated");
    edge.IsConstrainedFeature = true;
    RequireGeometry(!IsCylinderTessellationContinuation(mesh, source, target, shared, config),
                    "cylinder continuation overrides an explicit feature");
    edge.IsConstrainedFeature = false;
    edge.IsNonManifold = true;
    RequireGeometry(!IsCylinderTessellationContinuation(mesh, source, target, shared, config),
                    "cylinder continuation overrides a non-manifold boundary");
    edge.IsNonManifold = false;

    // A nearby shallow bevel can pass the ordinary distance tolerance yet
    // fall well outside the measured noise of the unchanged mother cylinder.
    auto bevelSoup = soup;
    for (int y = 0; y <= axialSegments; ++y)
      bevelSoup.Vertices[y * (angularSegments + 1)].X() +=
          mesh.getResolution().FittingTolerance;
    MeshTopology bevel;
    RequireGeometry(bevel.build(bevelSoup), "shallow bevel fixture failed");
    RequireGeometry(IsSurfaceCompatible(EvaluateSurfaceCompatibility(bevel, sourceIds, target),
                                         bevel.getResolution()),
                    "continuation distance counterexample is outside ordinary tolerance");
    RequireGeometry(!IsCylinderTessellationContinuation(bevel, source, target, shared, config),
                    "shallow bevel incorrectly treated as cylinder tessellation noise");

    auto coarseSoup = soup;
    for (int y = 0; y <= axialSegments; ++y)
      coarseSoup.Vertices[y * (angularSegments + 1)] =
          {radius * std::cos(-step), double(y) / axialSegments, radius * std::sin(-step)};
    MeshTopology coarse;
    RequireGeometry(coarse.build(coarseSoup), "coarse facet fixture failed");
    RequireGeometry(!IsCylinderTessellationContinuation(coarse, source, target, shared, config),
                    "angularly incompatible facet treated as a tessellation continuation");
    // A genuine broad plane is never a recovery target.
    auto planarTarget = Plane();
    planarTarget.TriangleIds = targetIds;
    RequireGeometry(!IsCylinderTessellationContinuation(mesh, source, planarTarget, shared, config),
                    "continuation rule extended to planar targets");
  }
}
} // namespace

void RunGeometryGuardTests() {
  TestFixedSurfaceAcceptance();
  TestCurvatureGradientUnits();
  TestFreeformDiagnostic();
  TestHeatGeometryAndRollback();
  TestSingleLabelHardBoundary();
  TestCylinderFacetContinuation();
}
