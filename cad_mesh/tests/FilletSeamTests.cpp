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
constexpr int AngularCells = 24, AxialCells = 8, StripColumn = 11;
enum class Shape { Cylinder, DifferentRadius, FoldedPlane, BranchedResidual };

struct SeamFixture {
  MeshTopology Mesh;
  std::vector<MeshPatch> Patches;
  std::vector<int> Left, Corridor, Right, Branch;
};

void FitPatch(MeshTopology &mesh, MeshPatch &patch, PatchSurfaceType type) {
  patch.SurfaceType = type;
  if (type == PatchSurfaceType::Freeform)
    return;
  std::unique_ptr<ISurfaceFitter> fitter;
  if (type == PatchSurfaceType::Plane)
    fitter = std::make_unique<PlaneSurfaceFitter>();
  else
    fitter = std::make_unique<CylinderSurfaceFitter>();
  Require(fitter->fit(mesh, patch.TriangleIds), "fillet seam fixture fit failed");
  patch.Parameters = fitter->getParameters();
  patch.RmsFittingError = fitter->computeRmsError();
  patch.MaxFittingError = fitter->computeMaxError();
  patch.NormalError = fitter->computeNormalError();
  patch.ProjectionTarget = PatchProjectionTarget::AnalyticSurface;
}

SeamFixture MakeSeam(Shape shape = Shape::Cylinder, double scale = 1,
                     double height = 2, double angleStep = Pi / 48,
                     bool twoBranches = false, bool wrapAround = false,
                     bool planarBranches = false) {
  TriangleSoup soup;
  std::vector<int> groups;
  const int angularCells = wrapAround ? 96 : AngularCells;
  auto point = [scale](double x, double y, double z) {
    return Point3{(x + 10) * scale, (y - 20) * scale, (z + 30) * scale};
  };
  for (int row = 0; row <= AxialCells; ++row)
    for (int column = 0; column <= angularCells; ++column) {
      const double angle = (column - 12) * angleStep;
      double x = std::cos(angle), y = std::sin(angle);
      if (column > StripColumn + 1 && shape == Shape::DifferentRadius) {
        x = -.5 + 1.5 * std::cos(angle);
        y = 1.5 * std::sin(angle);
      } else if (column > StripColumn + 1 && shape == Shape::FoldedPlane) {
        x = 1 + .6 * angle;
        y = angle;
      }
      soup.Vertices.push_back(point(x, y, height * row / AxialCells));
    }
  for (int row = 0; row < AxialCells; ++row)
    for (int column = 0; column < angularCells; ++column) {
      const int a = row * (angularCells + 1) + column;
      const int b = a + 1, c = a + angularCells + 1, d = c + 1;
      soup.Triangles.push_back({{a, b, d}});
      soup.Triangles.push_back({{a, d, c}});
      const int group = column < StripColumn ? 0 : column == StripColumn ? 1 : 2;
      groups.insert(groups.end(), 2, group);
    }
  if (shape == Shape::BranchedResidual) {
    // Extend the corridor through its top edge into a much larger, bent
    // residual. These faces share one Freeform label with the corridor, but
    // are incompatible with the cylinder and must remain connected outside it.
    for (int branch = 0; branch < (twoBranches ? 2 : 1); ++branch) {
      const int side = branch == 0 ? 1 : -1;
      int left = (side > 0 ? AxialCells : 0) * (angularCells + 1) + StripColumn;
      int right = left + 1;
      for (int row = 1; row <= 24; ++row) {
        const double extension = row / 6.0;
        const double radius = planarBranches ? 1 :
            1 + .4 * extension + .12 * extension * extension;
        const double halfWidth = angleStep / 2 + (planarBranches ? 0 : .15 * extension);
        const double xOffset = planarBranches ? .8 * extension : 0;
        const double center = -angleStep / 2;
        const double z = (side > 0 ? height : 0) + side * extension;
        const int nextLeft = int(soup.Vertices.size()), nextRight = nextLeft + 1;
        soup.Vertices.push_back(point(radius * std::cos(center - halfWidth) + xOffset,
                                      radius * std::sin(center - halfWidth), z));
        soup.Vertices.push_back(point(radius * std::cos(center + halfWidth) + xOffset,
                                      radius * std::sin(center + halfWidth), z));
        if (side > 0) {
          soup.Triangles.push_back({{left, right, nextRight}});
          soup.Triangles.push_back({{left, nextRight, nextLeft}});
        } else {
          soup.Triangles.push_back({{left, nextRight, right}});
          soup.Triangles.push_back({{left, nextLeft, nextRight}});
        }
        groups.insert(groups.end(), 2, 3);
        left = nextLeft;
        right = nextRight;
      }
    }
  }
  SeamFixture result;
  Require(result.Mesh.build(soup), "fillet seam topology failed");
  Require(result.Mesh.getTriangles().size() == soup.Triangles.size(),
          "fillet seam fixture lost a triangle during cleanup");
  result.Patches.resize(3);
  const auto &mapping = result.Mesh.getOriginalToCleanTriangleMap();
  for (size_t original = 0; original < groups.size(); ++original) {
    const int id = mapping[original], group = groups[original];
    result.Patches[group == 3 ? 1 : group].TriangleIds.push_back(id);
    auto &members = group == 0 ? result.Left : group == 1 ? result.Corridor :
                    group == 2 ? result.Right : result.Branch;
    members.push_back(id);
  }
  FitPatch(result.Mesh, result.Patches[0], PatchSurfaceType::Cylinder);
  FitPatch(result.Mesh, result.Patches[1], shape == Shape::BranchedResidual ?
           PatchSurfaceType::Freeform : PatchSurfaceType::Plane);
  FitPatch(result.Mesh, result.Patches[2], shape == Shape::FoldedPlane ?
           PatchSurfaceType::Plane : PatchSurfaceType::Cylinder);
  Require(result.Corridor.size() == 16, "fixture must exceed the old 12-face gate");
  for (const auto &patch : result.Patches)
    Require(IsManifoldConnected(result.Mesh, patch), "input seam patch is disconnected");
  return result;
}

std::vector<int> AssertConsolidatedOwnership(const SeamFixture &fixture,
                                            const std::vector<MeshPatch> &patches) {
  std::vector<int> owners(fixture.Mesh.getTriangles().size(), -1);
  for (size_t id = 0; id < patches.size(); ++id) {
    Require(patches[id].Id == int(id), "seam output has stale patch IDs");
    Require(IsManifoldConnected(fixture.Mesh, patches[id]),
            "seam repair left a disconnected patch");
    for (int face : patches[id].TriangleIds) {
      Require(face >= 0 && face < int(owners.size()) && owners[face] < 0,
              "seam repair invented or duplicated face ownership");
      owners[face] = int(id);
    }
  }
  Require(std::find(owners.begin(), owners.end(), -1) == owners.end(),
          "seam repair omitted an input face");
  return owners;
}

void TestCompatibleStrip(double scale) {
  auto fixture = MakeSeam(Shape::Cylinder, scale);
  const auto before = fixture.Mesh.getTriangles();
  auto merged = MergeAdjacentSurfaceModels(fixture.Mesh, fixture.Patches, {});
  AssertConsolidatedOwnership(fixture, merged);
  Require(merged.size() == 1 && merged[0].SurfaceType == PatchSurfaceType::Cylinder,
          "one cylinder separated by a 16-face planar seam must merge, scale=" +
              std::to_string(scale) + ", patches=" + std::to_string(merged.size()));
  Require(merged[0].MaxFittingError < scale * 1e-8,
          "fillet seam union was not refitted on the complete cylinder");
  for (size_t face = 0; face < before.size(); ++face)
    Require(before[face].VertexIds == fixture.Mesh.getTriangles()[face].VertexIds,
            "fillet seam repair changed tessellation");
}

void TestExplicitHardEdge() {
  auto fixture = MakeSeam();
  const std::set<int> left(fixture.Left.begin(), fixture.Left.end());
  const std::set<int> strip(fixture.Corridor.begin(), fixture.Corridor.end());
  int protectedEdge = -1;
  for (int id = 0; id < int(fixture.Mesh.getEdges().size()); ++id) {
    auto &edge = fixture.Mesh.getEdges()[id];
    if ((left.count(edge.Triangle0) && strip.count(edge.Triangle1)) ||
        (left.count(edge.Triangle1) && strip.count(edge.Triangle0))) {
      edge.IsConstrainedFeature = true;
      protectedEdge = id;
      break;
    }
  }
  Require(protectedEdge >= 0, "hard seam fixture has no shared edge");
  auto merged = MergeAdjacentSurfaceModels(fixture.Mesh, fixture.Patches, {});
  auto owners = AssertConsolidatedOwnership(fixture, merged);
  const auto &edge = fixture.Mesh.getEdges()[protectedEdge];
  Require(edge.IsConstrainedFeature && owners[edge.Triangle0] != owners[edge.Triangle1],
          "fillet seam repair erased an explicitly protected shared edge");
}

void TestDifferentGeometry(Shape shape) {
  auto fixture = MakeSeam(shape);
  auto merged = MergeAdjacentSurfaceModels(fixture.Mesh, fixture.Patches, {});
  auto owners = AssertConsolidatedOwnership(fixture, merged);
  Require(owners[fixture.Left.front()] != owners[fixture.Right.front()],
          "a physically narrow strip merged genuinely different adjacent surfaces");
  if (shape == Shape::DifferentRadius) {
    const auto &a = merged[owners[fixture.Left.front()]];
    const auto &b = merged[owners[fixture.Right.front()]];
    Require(a.SurfaceType == PatchSurfaceType::Cylinder &&
                b.SurfaceType == PatchSurfaceType::Cylinder,
            "different-radius fixture lost a cylindrical mother surface");
    Require(std::abs(std::get<CylinderParameters>(a.Parameters).Radius - 1) < 1e-8 &&
                std::abs(std::get<CylinderParameters>(b.Parameters).Radius - 1.5) < 1e-8,
            "seam repair replaced distinct radii with an interpolated model");
  }
}

void TestBranchedResidualCorridor() {
  auto fixture = MakeSeam(Shape::BranchedResidual);
  auto merged = MergeAdjacentSurfaceModels(fixture.Mesh, fixture.Patches, {});
  auto owners = AssertConsolidatedOwnership(fixture, merged);
  const int cylinder = owners[fixture.Left.front()];
  for (const auto &members : {fixture.Left, fixture.Corridor, fixture.Right})
    for (int face : members)
      Require(owners[face] == cylinder,
              "compatible corridor was not extracted from the larger Freeform patch");
  for (int face : fixture.Branch)
    Require(owners[face] != cylinder,
            "corridor extraction swallowed the incompatible residual branch");
  Require(merged[cylinder].SurfaceType == PatchSurfaceType::Cylinder,
          "corridor union lost the certified cylindrical model");
}

void TestShortArcParameterDrift() {
  auto fixture = MakeSeam(Shape::Cylinder, 1, 2, Pi / 240);
  // Small cylindrical arcs poorly constrain radius and axis offset separately.
  // Equal and opposite perturbations leave their actual observed geometry
  // within tolerance, as in the radius/axis compensation in real fillets.
  for (int id : {0, 2}) {
    auto &patch = fixture.Patches[id];
    auto parameters = std::get<CylinderParameters>(patch.Parameters);
    const double offset = id == 0 ? -.0005 : .0005;
    parameters.Axis.Origin.X() += offset;
    parameters.Radius -= offset;
    patch.Parameters = parameters;
    patch.MaxFittingError = 0;
    for (int face : patch.TriangleIds)
      for (int vertex : fixture.Mesh.getTriangles()[face].VertexIds) {
        const auto delta = Sub(ToVec(fixture.Mesh.getVertices()[vertex].Position),
                               ToVec(parameters.Axis.Origin));
        const double residual = std::abs(Norm(Cross(delta, ToVec(parameters.Axis.Direction))) -
                                         parameters.Radius);
        patch.MaxFittingError = std::max(patch.MaxFittingError, residual);
      }
    Require(patch.MaxFittingError < fixture.Mesh.getResolution().FittingTolerance,
            "parameter drift fixture no longer represents its observed arc");
  }
  const auto &left = std::get<CylinderParameters>(fixture.Patches[0].Parameters);
  const auto &right = std::get<CylinderParameters>(fixture.Patches[2].Parameters);
  const double oldIdentityTolerance = std::min(
      3 * fixture.Mesh.getResolution().FittingTolerance,
      std::max(.2 * fixture.Mesh.getResolution().FittingTolerance,
               3 * (fixture.Patches[0].MaxFittingError + fixture.Patches[2].MaxFittingError)));
  Require(std::abs(left.Radius - right.Radius) > oldIdentityTolerance,
          "parameter drift must exercise the old strict identity rejection");
  auto merged = MergeAdjacentSurfaceModels(fixture.Mesh, fixture.Patches, {});
  AssertConsolidatedOwnership(fixture, merged);
  Require(merged.size() == 1 && merged[0].SurfaceType == PatchSurfaceType::Cylinder,
          "geometrically compatible short arcs were rejected by parameter drift");
  Require(std::abs(std::get<CylinderParameters>(merged[0].Parameters).Radius - 1) < 1e-8 &&
              merged[0].MaxFittingError < 1e-8,
          "short-arc bridge must recover the complete underlying cylinder");
}

void TestShortAxialCorridor() {
  constexpr double height = .045;
  auto fixture = MakeSeam(Shape::Cylinder, 1, height);
  const double width = 2 * std::sin(Pi / 96);
  const double supportedPerimeterFraction = height / (height + width);
  Require(supportedPerimeterFraction > .40 && supportedPerimeterFraction < .42,
          "short corridor must exercise opposing rails at about 41% perimeter support");
  auto merged = MergeAdjacentSurfaceModels(fixture.Mesh, fixture.Patches, {});
  AssertConsolidatedOwnership(fixture, merged);
  Require(merged.size() == 1 && merged[0].SurfaceType == PatchSurfaceType::Cylinder,
          "short axial corridor with two opposing rails was not bridged");
}

void TestOneSidedStripIsNotABridge() {
  auto fixture = MakeSeam();
  fixture.Patches.assign(2, MeshPatch{});
  fixture.Corridor.clear();
  const auto &mapping = fixture.Mesh.getOriginalToCleanTriangleMap();
  std::set<int> strip;
  for (int row = 0; row < AxialCells; ++row)
    for (int triangle = 0; triangle < 2; ++triangle)
      strip.insert(mapping[row * AngularCells * 2 + triangle]);
  for (int face = 0; face < int(fixture.Mesh.getTriangles().size()); ++face)
    fixture.Patches[strip.count(face) ? 1 : 0].TriangleIds.push_back(face);
  fixture.Corridor.assign(strip.begin(), strip.end());
  FitPatch(fixture.Mesh, fixture.Patches[0], PatchSurfaceType::Cylinder);
  FitPatch(fixture.Mesh, fixture.Patches[1], PatchSurfaceType::Plane);
  auto merged = MergeAdjacentSurfaceModels(fixture.Mesh, fixture.Patches, {});
  auto owners = AssertConsolidatedOwnership(fixture, merged);
  Require(merged.size() == 2, "an isolated one-sided strip was treated as a fillet seam");
  for (int face : fixture.Corridor)
    Require(owners[face] != owners[fixture.Patches[0].TriangleIds.front()],
            "narrowness and fitting alone cannot replace opposing-rail evidence");
}

void TestDisabledAndInsufficientBudget() {
  for (int setting = 0; setting < 3; ++setting) {
    auto fixture = MakeSeam();
    SegmentationConfig config;
    if (setting == 0)
      config.EnableModelSeamBridging = false;
    else if (setting == 1)
      config.ModelSeamEvaluationBudget = 1;
    else
      config.ModelSeamMaximumFaces = 8;
    auto merged = MergeAdjacentSurfaceModels(fixture.Mesh, fixture.Patches, config);
    AssertConsolidatedOwnership(fixture, merged);
    Require(merged.size() == fixture.Patches.size(),
            "disabled or unfinished seam proposal partially changed ownership");
    std::set<std::set<int>> before, after;
    for (const auto &patch : fixture.Patches)
      before.emplace(patch.TriangleIds.begin(), patch.TriangleIds.end());
    for (const auto &patch : merged)
      after.emplace(patch.TriangleIds.begin(), patch.TriangleIds.end());
    Require(before == after, "budget exhaustion partially absorbed a corridor");
  }
}

void TestOneCylinderWithTwoResidualBranches(bool planarDiagnostics = false) {
  auto fixture = MakeSeam(Shape::BranchedResidual, 1, 2, Pi / 48, true, true,
                         planarDiagnostics);
  // Around the back of the full cylinder, both sides of the residual strip
  // already belong to the same connected analytic patch. Removing its local
  // corridor disconnects the original Freeform patch into two separate ends.
  fixture.Patches[0].TriangleIds.insert(fixture.Patches[0].TriangleIds.end(),
      fixture.Patches[2].TriangleIds.begin(), fixture.Patches[2].TriangleIds.end());
  fixture.Patches.resize(2);
  FitPatch(fixture.Mesh, fixture.Patches[0], PatchSurfaceType::Cylinder);
  Require(IsManifoldConnected(fixture.Mesh, fixture.Patches[0]),
          "single-target fixture must connect around the cylinder");
  if (planarDiagnostics) {
    // Each branch is planar, but the two branches and intervening corridor
    // are not coplanar. Seed actual parent diagnostics rather than arbitrary
    // sentinel values so stale inherited errors are observable after split.
    FreeformSurfaceFitter parent;
    auto &residual = fixture.Patches[1];
    Require(parent.fit(fixture.Mesh, residual.TriangleIds),
            "parent Freeform diagnostic fixture fit failed");
    residual.RmsFittingError = parent.computeRmsError();
    residual.MaxFittingError = parent.computeMaxError();
    residual.NormalError = parent.computeNormalError();
    Require(residual.RmsFittingError > 1e-4 && residual.MaxFittingError > 1e-4 &&
                residual.NormalError > .01,
            "parent diagnostic fixture must have nonzero nonplanar errors");
  }
  const auto originalFaces = fixture.Mesh.getTriangles();
  auto merged = MergeAdjacentSurfaceModels(fixture.Mesh, fixture.Patches, {});
  auto owners = AssertConsolidatedOwnership(fixture, merged);
  const int cylinder = owners[fixture.Left.front()];
  for (const auto &members : {fixture.Left, fixture.Corridor, fixture.Right})
    for (int face : members)
      Require(owners[face] == cylinder,
              "opposing rails on one existing cylinder did not close its seam");
  std::set<int> residualOwners;
  for (int face : fixture.Branch) {
    Require(owners[face] != cylinder, "seam repair consumed an incompatible end branch");
    residualOwners.insert(owners[face]);
  }
  Require(merged.size() == 3 && residualOwners.size() == 2,
          "corridor removal must split the remaining Freeform into two connected patches");
  if (planarDiagnostics) {
    for (int owner : residualOwners) {
      const auto &residual = merged[owner];
      Require(residual.SurfaceType == PatchSurfaceType::Freeform,
              "refreshing diagnostics must not reclassify residual geometry");
      Require(residual.RmsFittingError < 1e-8 && residual.MaxFittingError < 1e-8 &&
                  residual.NormalError < 1e-6,
              "split planar residual inherited the nonplanar parent's fitting diagnostics");
    }
    for (size_t face = 0; face < originalFaces.size(); ++face)
      Require(originalFaces[face].VertexIds == fixture.Mesh.getTriangles()[face].VertexIds,
              "residual diagnostic refresh changed the input tessellation");
  }
}
} // namespace

void RunFilletSeamTests() {
  const std::pair<const char *, void (*)()> tests[] = {
      {"16-face planar seam", +[] { TestCompatibleStrip(1); }},
      {"small-scale seam", +[] { TestCompatibleStrip(.001); }},
      {"large-scale seam", +[] { TestCompatibleStrip(1000); }},
      {"explicit hard seam", TestExplicitHardEdge},
      {"different cylinder radii", +[] { TestDifferentGeometry(Shape::DifferentRadius); }},
      {"true planar fold", +[] { TestDifferentGeometry(Shape::FoldedPlane); }},
      {"corridor inside branched Freeform", TestBranchedResidualCorridor},
      {"short-arc parameter compensation", TestShortArcParameterDrift},
      {"short axial corridor", TestShortAxialCorridor},
      {"one-sided strip", TestOneSidedStripIsNotABridge},
      {"disabled and bounded proposals", TestDisabledAndInsufficientBudget},
      {"single target and two residual ends", +[] { TestOneCylinderWithTwoResidualBranches(); }},
      {"split residual diagnostic refresh", +[] { TestOneCylinderWithTwoResidualBranches(true); }}};
  std::string failures;
  for (const auto &test : tests) {
    try {
      test.second();
      std::cout << "PASS fillet seam: " << test.first << '\n';
    } catch (const std::exception &error) {
      const std::string failure = std::string(test.first) + ": " + error.what();
      std::cerr << "FAIL fillet seam: " << failure << '\n';
      failures += failure + "; ";
    }
  }
  Require(failures.empty(), failures);
}
