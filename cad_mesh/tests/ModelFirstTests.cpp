#include "CadMesh/ModelFirstPartitioner.h"
#include "CadMesh/SurfaceFitting.h"
#include "TestHelpers.h"
#include <cmath>
#include <iostream>
#include <numeric>

using namespace CadMesh;
using namespace CadMeshTests;

namespace {
constexpr double Pi = 3.14159265358979323846;

// Varying angular and axial sample density changes the tessellation, not the
// underlying cylinder. Alternating diagonals also removes an ordering crutch.
TriangleSoup CylinderWall(int around, bool nonuniform = false,
                          double radius = 1.0, double bottom = 0.0,
                          double top = 2.0) {
  TriangleSoup soup;
  const std::vector<double> levels =
      nonuniform ? std::vector<double>{0, .015, .05, .13, .35, .7, 1}
                 : std::vector<double>{0, .25, .5, .75, 1};
  for (double level : levels)
    for (int i = 0; i < around; ++i) {
      double angle = 2 * Pi * i / around;
      if (nonuniform)
        angle += .42 * std::sin(angle);
      soup.Vertices.push_back({radius * std::cos(angle),
                               radius * std::sin(angle),
                               bottom + (top - bottom) * level});
    }
  for (int row = 0; row + 1 < int(levels.size()); ++row)
    for (int i = 0; i < around; ++i) {
      int j = (i + 1) % around;
      int a = row * around + i, b = row * around + j;
      int c = a + around, d = b + around;
      if ((i + row) % 2) {
        soup.Triangles.push_back({{a, b, c}});
        soup.Triangles.push_back({{b, d, c}});
      } else {
        soup.Triangles.push_back({{a, b, d}});
        soup.Triangles.push_back({{a, d, c}});
      }
    }
  return soup;
}

void Append(TriangleSoup &destination, const TriangleSoup &source) {
  const int offset = int(destination.Vertices.size());
  destination.Vertices.insert(destination.Vertices.end(),
                              source.Vertices.begin(), source.Vertices.end());
  for (auto triangle : source.Triangles) {
    for (int &vertex : triangle)
      vertex += offset;
    destination.Triangles.push_back(triangle);
  }
}

void AssertSingleSurface(const TriangleSoup &soup, PatchSurfaceType type,
                         const std::string &context) {
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), context + ": segmentation failed");
  AssertPartitionInvariants(segmenter, context);
  Require(segmenter.getMesh().getTriangles().size() == soup.Triangles.size(),
          context + ": valid input faces disappeared during segmentation");
  const auto &patches = segmenter.getPatches();
  Require(patches.size() == 1, context + ": complete surface fragmented into " +
                                   std::to_string(patches.size()) + " patches");
  Require(patches.front().SurfaceType == type,
          context + ": expected " + SurfaceTypeName(type) + ", received " +
              SurfaceTypeName(patches.front().SurfaceType));
  Require(patches.front().ProjectionTarget ==
              PatchProjectionTarget::AnalyticSurface,
          context + ": analytic geometry must be available to the remesher");
  for (int edgeId : segmenter.getRemeshConstraint().ConstraintEdgeIds)
    Require(segmenter.getMesh().getEdges()[edgeId].IsBoundary,
            context + ": tessellation seam became a remesh constraint");
}

void TestCylinderTessellationIndependence() {
  std::string errors;
  for (int around : {32, 128})
    for (bool nonuniform : {false, true}) {
      try {
        AssertSingleSurface(CylinderWall(around, nonuniform),
                            PatchSurfaceType::Cylinder,
                            "cylinder " + std::to_string(around) +
                                (nonuniform ? " nonuniform" : " uniform"));
      } catch (const std::exception &error) {
        errors += std::string(error.what()) + "; ";
      }
    }
  Require(errors.empty(), errors);
}

void TestPhysicalScaleConsistency() {
  for (double scale : {.001, 1000.0}) {
    auto soup = CylinderWall(48, true);
    for (auto &vertex : soup.Vertices)
      vertex = {(vertex.X() + 100) * scale, (vertex.Y() - 300) * scale,
                (vertex.Z() + 50) * scale};
    AssertSingleSurface(soup, PatchSurfaceType::Cylinder,
                        "scaled cylinder " + std::to_string(scale));
  }
}

void TestTwoShallowPlanes() {
  const double c = std::cos(15 * Pi / 180.0);
  const double s = std::sin(15 * Pi / 180.0);
  TriangleSoup soup;
  soup.Vertices = {{0, 0, 0}, {1, 0, 0}, {1 + c, 0, s},
                   {0, 1, 0}, {1, 1, 0}, {1 + c, 1, s}};
  soup.Triangles = {{{0, 1, 4}}, {{0, 4, 3}}, {{1, 2, 5}}, {{1, 5, 4}}};
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), "two shallow planes segmentation failed");
  AssertPartitionInvariants(segmenter, "two shallow planes");
  Require(segmenter.getPatches().size() == 2,
          "two planar faces at fifteen degrees must not interpolate a "
          "fictitious cylinder");
  for (const auto &patch : segmenter.getPatches())
    Require(patch.SurfaceType == PatchSurfaceType::Plane &&
                patch.TriangleIds.size() == 2,
            "two shallow planes must retain their complete planar faces");
  const auto &mesh = segmenter.getMesh();
  const auto &constraints = segmenter.getRemeshConstraint();
  const std::set<int> hard(constraints.HardFeatureEdgeIds.begin(),
                           constraints.HardFeatureEdgeIds.end());
  int creaseCount = 0;
  for (int id = 0; id < int(mesh.getEdges().size()); ++id) {
    const auto &edge = mesh.getEdges()[id];
    if (edge.IncidentTriangleIds.size() != 2 ||
        mesh.getTriangles()[edge.Triangle0].PatchId ==
            mesh.getTriangles()[edge.Triangle1].PatchId)
      continue;
    ++creaseCount;
    Require(edge.IsConstrainedFeature && hard.count(id),
            "the fifteen-degree planar crease must remain a hard remesh "
            "constraint");
  }
  Require(creaseCount == 1,
          "the two planar faces lost their single shared crease");
}

void TestPlanarFaceWithHole() {
  TriangleSoup soup;
  const int around = 48;
  for (double radius : {1.0, 1.05, 1.8, 3.0})
    for (int i = 0; i < around; ++i) {
      const double angle = 2 * Pi * i / around;
      soup.Vertices.push_back(
          {radius * std::cos(angle), radius * std::sin(angle), .7});
    }
  for (int row = 0; row < 3; ++row)
    for (int i = 0; i < around; ++i) {
      int a = row * around + i, b = row * around + (i + 1) % around;
      soup.Triangles.push_back({{a, a + around, b + around}});
      soup.Triangles.push_back({{a, b + around, b}});
    }
  AssertSingleSurface(soup, PatchSurfaceType::Plane, "plane with hole");
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), "annulus boundary segmentation failed");
  const auto &chains = segmenter.getRemeshConstraint().BoundaryChains;
  Require(chains.size() == 2 && chains[0].IsClosed && chains[1].IsClosed,
          "one planar face must preserve its inner and outer boundary loops");
}

void TestShallowChamfer() {
  TriangleSoup soup;
  // Seven degrees is below the broad model normal tolerance. Its measurable
  // offset and two support planes must still keep this narrow real CAD face.
  const double rise = .1 * std::tan(7 * Pi / 180.0);
  for (int y = 0; y <= 4; ++y)
    for (auto xz :
         {std::pair<double, double>{0, 0}, {2, 0}, {2.1, rise}, {4.1, rise}})
      soup.Vertices.push_back({xz.first, y * .5, xz.second});
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 3; ++column) {
      int a = row * 4 + column;
      soup.Triangles.push_back({{a, a + 1, a + 5}});
      soup.Triangles.push_back({{a, a + 5, a + 4}});
    }
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), "shallow chamfer segmentation failed");
  AssertPartitionInvariants(segmenter, "shallow chamfer");
  Require(segmenter.getPatches().size() == 3,
          "a seven-degree chamfer between two planes must retain three faces");
  std::set<int> owners;
  const auto &map = segmenter.getMesh().getOriginalToCleanTriangleMap();
  for (int column = 0; column < 3; ++column) {
    const int expected =
        segmenter.getMesh().getTriangles()[map[2 * column]].PatchId;
    owners.insert(expected);
    const auto &patch = segmenter.getPatches()[expected];
    Require(patch.SurfaceType == PatchSurfaceType::Plane &&
                patch.TriangleIds.size() == 8,
            "shallow chamfer surface was absorbed or fragmented");
    for (int row = 0; row < 4; ++row)
      for (int diagonal = 0; diagonal < 2; ++diagonal)
        Require(segmenter.getMesh()
                        .getTriangles()[map[6 * row + 2 * column + diagonal]]
                        .PatchId == expected,
                "shallow chamfer boundary moved onto a tessellation edge");
  }
  Require(owners.size() == 3, "shallow chamfer support planes were merged");
  const auto &constraints = segmenter.getRemeshConstraint();
  const std::set<int> hard(constraints.HardFeatureEdgeIds.begin(),
                           constraints.HardFeatureEdgeIds.end());
  const std::set<int> smooth(constraints.SurfaceTransitionEdgeIds.begin(),
                             constraints.SurfaceTransitionEdgeIds.end());
  int internalSeamEdges = 0;
  const auto &mesh = segmenter.getMesh();
  for (int id = 0; id < int(mesh.getEdges().size()); ++id) {
    const auto &edge = mesh.getEdges()[id];
    if (edge.IncidentTriangleIds.size() != 2 ||
        mesh.getTriangles()[edge.Triangle0].PatchId ==
            mesh.getTriangles()[edge.Triangle1].PatchId)
      continue;
    ++internalSeamEdges;
    Require(edge.IsConstrainedFeature && hard.count(id) && !smooth.count(id),
            "both seven-degree chamfer seams must be geometric hard features");
  }
  Require(internalSeamEdges == 8, "the shallow chamfer must retain all four "
                                  "edges on each of its two seams");
}

void TestDistinctCylinderParameters() {
  constexpr int around = 48;
  auto soup = CylinderWall(around, false, 1, 0, 1);
  auto upper = CylinderWall(around, false, 1.5, 1, 2);
  Append(soup, upper);
  TriangleSoup shoulder;
  for (double radius : {1.0, 1.5})
    for (int i = 0; i < around; ++i) {
      const double angle = 2 * Pi * i / around;
      shoulder.Vertices.push_back(
          {radius * std::cos(angle), radius * std::sin(angle), 1});
    }
  for (int i = 0; i < around; ++i) {
    int j = (i + 1) % around;
    shoulder.Triangles.push_back({{i, j + around, i + around}});
    shoulder.Triangles.push_back({{i, j, j + around}});
  }
  Append(soup, shoulder);
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), "stepped cylinder segmentation failed");
  AssertPartitionInvariants(segmenter, "stepped cylinder");
  int planes = 0;
  std::vector<double> radii;
  for (const auto &patch : segmenter.getPatches()) {
    planes += patch.SurfaceType == PatchSurfaceType::Plane;
    if (patch.SurfaceType == PatchSurfaceType::Cylinder)
      radii.push_back(std::get<CylinderParameters>(patch.Parameters).Radius);
  }
  std::sort(radii.begin(), radii.end());
  Require(
      segmenter.getPatches().size() == 3 && planes == 1 && radii.size() == 2,
      "two cylinder radii and their shoulder must remain three complete faces");
  Require(std::abs(radii[0] - 1) < 1e-5 && std::abs(radii[1] - 1.5) < 1e-5,
          "different cylinder radii were averaged into a fictitious surface");
}

void TestDisconnectedSameType() {
  for (bool rotated : {false, true}) {
    auto soup = CylinderWall(48);
    auto other = CylinderWall(48);
    for (auto &vertex : other.Vertices) {
      if (rotated)
        vertex = {vertex.Z() + 4, vertex.Y(), -vertex.X()};
      else
        vertex.X() += 4;
    }
    Append(soup, other);
    CadMeshPatchSegmenter segmenter;
    Require(segmenter.segment(soup),
            "disconnected cylinders segmentation failed");
    AssertPartitionInvariants(segmenter, "disconnected cylinders");
    Require(
        segmenter.getPatches().size() == 2,
        "same-type disconnected cylinders must have separate complete patches");
    for (const auto &patch : segmenter.getPatches())
      Require(patch.SurfaceType == PatchSurfaceType::Cylinder,
              "a disconnected cylinder lost its analytic type");
    Require(
        segmenter.getAdjacency().empty(),
        "disconnected analytic surfaces must not acquire a boundary adjacency");
    const auto &a =
        std::get<CylinderParameters>(segmenter.getPatches()[0].Parameters);
    const auto &b =
        std::get<CylinderParameters>(segmenter.getPatches()[1].Parameters);
    const double aligned =
        std::abs(Dot(ToVec(a.Axis.Direction), ToVec(b.Axis.Direction)));
    Require(rotated ? aligned < 1e-4 : aligned > .9999,
            "cylinder surface classification discarded independent axis "
            "parameters");
  }
}

void TestAdjacentDifferentCylinderAxes() {
  // Perpendicular radius-one cylinders meet on x == z. They share the exact
  // intersection curve (including two tangent points), so connectivity alone
  // cannot prevent a wrong same-type merge.
  constexpr int around = 64, along = 5;
  TriangleSoup soup;
  for (bool horizontal : {false, true}) {
    TriangleSoup half;
    for (int row = 0; row <= along; ++row)
      for (int column = 0; column < around; ++column) {
        const double angle = 2 * Pi * column / around;
        const double c = std::cos(angle), s = std::sin(angle);
        const double t = double(row) / along;
        if (horizontal)
          half.Vertices.push_back({c + t * (2 - c), s, c});
        else
          half.Vertices.push_back({c, s, -2 + t * (2 + c)});
      }
    for (int row = 0; row < along; ++row)
      for (int column = 0; column < around; ++column) {
        const int a = row * around + column;
        const int b = row * around + (column + 1) % around;
        for (auto triangle : {std::array<int, 3>{a, b, b + around},
                              std::array<int, 3>{a, b + around, a + around}}) {
          if (horizontal)
            std::swap(triangle[1], triangle[2]);
          half.Triangles.push_back(triangle);
        }
      }
    Append(soup, half);
  }
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), "joined cylinder axes segmentation failed");
  AssertPartitionInvariants(segmenter, "joined cylinder axes");
  Require(segmenter.getMesh().getCleanupReport().NonManifoldEdges == 0,
          "joined cylinder fixture must have a manifold shared curve");
  Require(segmenter.getPatches().size() == 2 &&
              segmenter.getAdjacency().size() == 1,
          "adjacent cylinders with different axes must remain two complete "
          "surfaces");
  std::vector<Vec3> axes;
  for (const auto &patch : segmenter.getPatches()) {
    Require(patch.SurfaceType == PatchSurfaceType::Cylinder,
            "joined cylinder lost its analytic type");
    const auto &cylinder = std::get<CylinderParameters>(patch.Parameters);
    axes.push_back(ToVec(cylinder.Axis.Direction));
    Require(std::abs(cylinder.Radius - 1) < 1e-5,
            "joined cylinder radius was changed by merging the other axis");
  }
  Require(std::abs(Dot(axes[0], axes[1])) < 1e-4,
          "joined cylinder axes were averaged together");
}

void TestConeAndTorusSurfaces() {
  auto cone = CylinderWall(64, true);
  for (auto &vertex : cone.Vertices) {
    const double radius = .8 + .4 * vertex.Z();
    vertex.X() *= radius;
    vertex.Y() *= radius;
  }
  AssertSingleSurface(cone, PatchSurfaceType::Cone, "conical wall");

  TriangleSoup torus;
  constexpr int along = 40, across = 20;
  for (int row = 0; row <= across; ++row)
    for (int column = 0; column <= along; ++column) {
      const double u = Pi * column / along;
      const double v = .5 * Pi * row / across;
      const double radial = 3 + .5 * std::cos(v);
      torus.Vertices.push_back(
          {radial * std::cos(u), radial * std::sin(u), .5 * std::sin(v)});
    }
  for (int row = 0; row < across; ++row)
    for (int column = 0; column < along; ++column) {
      const int a = row * (along + 1) + column;
      torus.Triangles.push_back({{a, a + 1, a + along + 2}});
      torus.Triangles.push_back({{a, a + along + 2, a + along + 1}});
    }
  AssertSingleSurface(torus, PatchSurfaceType::Torus,
                      "toroidal transition surface");
}

void TestLegacySwitch() {
  SegmentationConfig config;
  config.EnableModelFirst = false;
  CadMeshPatchSegmenter segmenter(config);
  Require(segmenter.segment(ChamferedCube()),
          "legacy mode segmentation failed");
  AssertPartitionInvariants(segmenter, "legacy mode");
  Require(segmenter.getPatches().size() == 7 &&
              segmenter.getMesh().getTriangles().size() == 16,
          "legacy mode must retain the protected single-triangle chamfer");
}

TriangleSoup TorusWithTangentMothers() {
  // A round annular shoulder, not an isolated primitive: the torus is tangent
  // to the upper plane and lower cylinder. Local cylinders/cones/spheres must
  // not claim its small neighborhoods before the complete model is compared.
  constexpr int around = 48;
  TriangleSoup soup;
  auto appendSurface = [&](const std::vector<double> &rows, auto point,
                           bool reverse) {
    const int offset = int(soup.Vertices.size());
    for (double row : rows)
      for (int column = 0; column < around; ++column)
        soup.Vertices.push_back(point(row, 2 * Pi * column / around));
    for (int row = 0; row + 1 < int(rows.size()); ++row)
      for (int column = 0; column < around; ++column) {
        const int a = offset + row * around + column;
        const int b = offset + row * around + (column + 1) % around;
        for (auto triangle : {std::array<int, 3>{a, b, b + around},
                              std::array<int, 3>{a, b + around, a + around}}) {
          if (reverse)
            std::swap(triangle[1], triangle[2]);
          soup.Triangles.push_back(triangle);
        }
      }
  };
  std::vector<double> angles;
  for (int i = 0; i <= 16; ++i)
    angles.push_back(.5 * Pi * i / 16);
  appendSurface(
      angles,
      [](double v, double u) {
        const double radius = 3 + .5 * std::cos(v);
        return Point3(radius * std::cos(u), radius * std::sin(u),
                      .5 * std::sin(v));
      },
      false);
  appendSurface(
      {1, 2, 3},
      [](double radius, double angle) {
        return Point3(radius * std::cos(angle), radius * std::sin(angle), .5);
      },
      true);
  appendSurface(
      {-1, -.75, -.5, -.25, 0},
      [](double z, double angle) {
        return Point3(3.5 * std::cos(angle), 3.5 * std::sin(angle), z);
      },
      false);
  return soup;
}

void TestTorusWithTangentMotherSurfaces() {
  const auto soup = TorusWithTangentMothers();
  constexpr int around = 48;
  CadMeshPatchSegmenter segmenter;
  Require(segmenter.segment(soup), "torus with mothers segmentation failed");
  AssertPartitionInvariants(segmenter, "torus with tangent mother surfaces");
  Require(
      segmenter.getMesh().getTriangles().size() == soup.Triangles.size() &&
          segmenter.getMesh().getCleanupReport().NonManifoldEdges == 0,
      "torus with mother surfaces fixture must preserve its manifold topology");
  Require(segmenter.getPatches().size() == 3,
          "a toroidal fillet and its two tangent mothers must form three "
          "complete surfaces, received " +
              std::to_string(segmenter.getPatches().size()));
  const auto &map = segmenter.getMesh().getOriginalToCleanTriangleMap();
  const std::pair<PatchSurfaceType, int> groups[] = {
      {PatchSurfaceType::Torus, 1536},
      {PatchSurfaceType::Plane, 192},
      {PatchSurfaceType::Cylinder, 384}};
  int first = 0;
  std::set<int> owners;
  for (const auto &group : groups) {
    const int owner = segmenter.getMesh().getTriangles()[map[first]].PatchId;
    owners.insert(owner);
    const auto &patch = segmenter.getPatches()[owner];
    Require(patch.SurfaceType == group.first &&
                int(patch.TriangleIds.size()) == group.second,
            std::string("tangent mother fixture lost the complete ") +
                SurfaceTypeName(group.first));
    for (int original = first; original < first + group.second; ++original)
      Require(segmenter.getMesh().getTriangles()[map[original]].PatchId ==
                  owner,
              "a tangent transition strip was assigned to the wrong mother "
              "surface");
    if (group.first == PatchSurfaceType::Torus)
      Require(patch.FeatureRole == PatchFeatureRole::Fillet,
              "a torus tangent to two confirmed mothers must retain its fillet "
              "role");
    first += group.second;
  }
  Require(owners.size() == 3,
          "a toroidal fillet was conflated with a mother surface");
  const auto &constraints = segmenter.getRemeshConstraint();
  const std::set<int> hard(constraints.HardFeatureEdgeIds.begin(),
                           constraints.HardFeatureEdgeIds.end());
  const std::set<int> smooth(constraints.SurfaceTransitionEdgeIds.begin(),
                             constraints.SurfaceTransitionEdgeIds.end());
  const auto &mesh = segmenter.getMesh();
  int internalSeamEdges = 0;
  for (int id = 0; id < int(mesh.getEdges().size()); ++id) {
    const auto &edge = mesh.getEdges()[id];
    if (edge.IsBoundary) {
      Require(hard.count(id) && !smooth.count(id),
              "open torus fixture boundaries must remain hard constraints");
      continue;
    }
    if (edge.IncidentTriangleIds.size() != 2 ||
        mesh.getTriangles()[edge.Triangle0].PatchId ==
            mesh.getTriangles()[edge.Triangle1].PatchId)
      continue;
    ++internalSeamEdges;
    Require(!edge.IsConstrainedFeature && !hard.count(id) && smooth.count(id),
            "a tangent torus-to-mother boundary must be a shared smooth "
            "transition");
  }
  Require(internalSeamEdges == 2 * around,
          "torus fillet must retain both complete circular mother boundaries");
}

std::vector<MeshPatch> CylinderFragments(MeshTopology &mesh) {
  std::vector<MeshPatch> patches(4);
  for (int f = 0; f < int(mesh.getTriangles().size()); ++f) {
    const double z = mesh.getTriangles()[f].Centroid.Z();
    patches[std::min(3, int(z * 2))].TriangleIds.push_back(f);
  }
  for (int id = 0; id < int(patches.size()); ++id) {
    CylinderSurfaceFitter fitter;
    auto &patch = patches[id];
    Require(fitter.fit(mesh, patch.TriangleIds),
            "fragment cylinder fixture fit failed");
    patch.Id = id;
    patch.SurfaceType = PatchSurfaceType::Cylinder;
    patch.Parameters = fitter.getParameters();
    patch.RmsFittingError = fitter.computeRmsError();
    patch.MaxFittingError = fitter.computeMaxError();
    patch.NormalError = fitter.computeNormalError();
  }
  return patches;
}

void TestFinalAdjacentCylinderUnion() {
  MeshTopology mesh;
  Require(mesh.build(CylinderWall(48)), "fragment cylinder topology failed");
  auto patches = CylinderFragments(mesh);
  const auto before = mesh.getTriangles();
  auto merged = MergeAdjacentSurfaceModels(mesh, patches, {});
  Require(merged.size() == 1 &&
              merged[0].SurfaceType == PatchSurfaceType::Cylinder,
          "four adjacent fragments of one cylinder were not consolidated");
  Require(merged[0].TriangleIds.size() == before.size(),
          "final union lost a triangle");
  std::set<int> members(merged[0].TriangleIds.begin(),
                        merged[0].TriangleIds.end());
  Require(members.size() == before.size(), "final union duplicated ownership");
  Require(merged[0].MaxFittingError < 1e-10,
          "final union was not certified on every face");
  for (size_t f = 0; f < before.size(); ++f)
    Require(before[f].VertexIds == mesh.getTriangles()[f].VertexIds,
            "surface consolidation changed the input tessellation");

  // A manually protected ring must remain even though both sides fit exactly
  // the same cylinder. The geometric model cannot override user constraints.
  for (auto &edge : mesh.getEdges()) {
    const auto &a = mesh.getVertices()[edge.Vertex0].Position;
    const auto &b = mesh.getVertices()[edge.Vertex1].Position;
    if (std::abs(a.Z() - 1) < 1e-12 && std::abs(b.Z() - 1) < 1e-12)
      edge.IsConstrainedFeature = true;
  }
  auto protectedResult = MergeAdjacentSurfaceModels(mesh, patches, {});
  Require(protectedResult.size() == 2,
          "final union erased an explicit cylinder ring constraint");
  for (const auto &patch : protectedResult) {
    const bool lower =
        mesh.getTriangles()[patch.TriangleIds.front()].Centroid.Z() < 1;
    for (int f : patch.TriangleIds)
      Require((mesh.getTriangles()[f].Centroid.Z() < 1) == lower,
              "a merged patch crossed the protected ring");
  }
}

void TestFairResidualReserve() {
  TriangleSoup soup;
  for (int component = 0; component < 3; ++component) {
    auto surface = TorusWithTangentMothers();
    for (auto &point : surface.Vertices)
      point.X() += component * 12;
    Append(soup, surface);
  }
  SegmentationConfig config;
  config.ModelMaximumSeeds = 1;
  config.ModelResidualSeedBudget = 24;
  auto exhaustedConfig = config;
  exhaustedConfig.ModelResidualSeedBudget = 0;
  CadMeshPatchSegmenter exhausted(exhaustedConfig);
  Require(exhausted.segment(soup),
          "exhausted-budget fixture segmentation failed");
  Require(std::any_of(exhausted.getPatches().begin(),
                      exhausted.getPatches().end(),
                      [](const MeshPatch &patch) {
                        return patch.SurfaceType == PatchSurfaceType::Freeform;
                      }),
          "residual reserve fixture must expose a surface missed by the "
          "initial seed budget");
  CadMeshPatchSegmenter segmenter(config);
  Require(segmenter.segment(soup), "fair residual reserve segmentation failed");
  AssertPartitionInvariants(segmenter, "fair residual reserve");
  Require(segmenter.getPatches().size() == 9,
          "bounded reserve did not recognize all three spatially separate "
          "composite surfaces, patches=" +
              std::to_string(segmenter.getPatches().size()));
  std::map<PatchSurfaceType, int> counts;
  for (const auto &patch : segmenter.getPatches())
    ++counts[patch.SurfaceType];
  Require(counts[PatchSurfaceType::Plane] == 3 &&
              counts[PatchSurfaceType::Cylinder] == 3 &&
              counts[PatchSurfaceType::Torus] == 3 &&
              counts[PatchSurfaceType::Freeform] == 0,
          "initial seed exhaustion left an unvisited component as Freeform");
}

void TestCertifiedResidualAbsorption() {
  for (bool planarFragment : {false, true}) {
    MeshTopology mesh;
    Require(mesh.build(CylinderWall(48)), "residual cylinder topology failed");
    std::vector<MeshPatch> patches(2);
    const auto &map = mesh.getOriginalToCleanTriangleMap();
    std::set<int> fragment;
    for (int row = 0; row < 4; ++row)
      for (int diagonal = 0; diagonal < 2; ++diagonal)
        fragment.insert(map[row * 96 + diagonal]);
    for (int face = 0; face < int(mesh.getTriangles().size()); ++face)
      patches[fragment.count(face) ? 1 : 0].TriangleIds.push_back(face);
    CylinderSurfaceFitter cylinder;
    Require(cylinder.fit(mesh, patches[0].TriangleIds),
            "residual target cylinder fit failed");
    patches[0].SurfaceType = PatchSurfaceType::Cylinder;
    patches[0].Parameters = cylinder.getParameters();
    patches[0].MaxFittingError = cylinder.computeMaxError();
    if (planarFragment) {
      PlaneSurfaceFitter plane;
      Require(plane.fit(mesh, patches[1].TriangleIds),
              "residual planar facet fit failed");
      patches[1].SurfaceType = PatchSurfaceType::Plane;
      patches[1].Parameters = plane.getParameters();
      patches[1].MaxFittingError = plane.computeMaxError();
    } else {
      patches[1].SurfaceType = PatchSurfaceType::Freeform;
    }
    auto merged = MergeAdjacentSurfaceModels(mesh, patches, {});
    Require(merged.size() == 1 &&
                merged[0].SurfaceType == PatchSurfaceType::Cylinder &&
                merged[0].TriangleIds.size() == mesh.getTriangles().size(),
            "a certified narrow tessellation fragment was not restored to its "
            "cylinder");
    Require(merged[0].MaxFittingError < 1e-10,
            "residual absorption did not refit and certify the complete union");
  }
}
} // namespace

void RunModelFirstTests() {
  const std::pair<const char *, void (*)()> tests[] = {
      {"cylinder tessellation", TestCylinderTessellationIndependence},
      {"physical scale consistency", TestPhysicalScaleConsistency},
      {"two shallow planes", TestTwoShallowPlanes},
      {"plane with hole", TestPlanarFaceWithHole},
      {"shallow chamfer", TestShallowChamfer},
      {"different cylinder radii", TestDistinctCylinderParameters},
      {"adjacent different cylinder axes", TestAdjacentDifferentCylinderAxes},
      {"disconnected cylinders", TestDisconnectedSameType},
      {"cone and torus surfaces", TestConeAndTorusSurfaces},
      {"torus with tangent mother surfaces",
       TestTorusWithTangentMotherSurfaces},
      {"final adjacent cylinder union", TestFinalAdjacentCylinderUnion},
      {"fair residual reserve", TestFairResidualReserve},
      {"certified residual absorption", TestCertifiedResidualAbsorption},
      {"legacy configuration switch", TestLegacySwitch}};
  std::string failures;
  for (const auto &test : tests) {
    try {
      test.second();
      std::cout << "PASS model-first: " << test.first << '\n';
    } catch (const std::exception &error) {
      const std::string failure = std::string(test.first) + ": " + error.what();
      std::cerr << "FAIL model-first: " << failure << '\n';
      failures += failure + "; ";
    }
  }
  Require(failures.empty(), failures);
}
