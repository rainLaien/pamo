#include "cad_adaptive/GeometryProjector.h"
#include "cad_adaptive/RxMeshBackend.h"
#include "check.h"
#include <cmath>

using namespace cad_adaptive;

int main() {
  CHECK(rxmeshAvailable());

  {
    SemanticMesh plane = makeGrid(12, 12, 0, 0, 1, 1, 0);
    lockMeshBoundary(plane);
    RemeshConfig cfg;
    cfg.adaptive = false;
    cfg.constantLength = 0.04f;
    cfg.maxGeometryError = 0.05f;
    cfg.maxIterations = 3;
    RxMeshBackend backend;
    RemeshReport report;
    CHECK(backend.remesh(plane, cfg, report));
    CHECK(report.topologyValid);
    CHECK(report.movedLockedVertices == 0);
    CHECK(report.splits + report.collapses > 0);
    const float planeTol = 1e-5f * std::max(plane.bboxDiagonal(), 1e-6f);
    for (int v = 0; v < plane.vertexCount(); ++v) {
      if (VertexConstraint(plane.vertexConstraint[v]) == VertexConstraint::Locked) continue;
      CHECK_NEAR(plane.pz[v], 0, planeTol);
    }
  }

  {
    const float R = 2.0f, z0 = 0.0f, z1 = 2.0f;
    SemanticMesh mesh = makeCylinder(20, 6, R, z0, z1, 0, 1, 2);
    std::vector<Vec3> locked;
    for (int v = 0; v < mesh.vertexCount(); ++v)
      if (VertexConstraint(mesh.vertexConstraint[v]) == VertexConstraint::Locked)
        locked.push_back(mesh.position(v));

    RemeshConfig cfg;
    cfg.adaptive = false;
    cfg.constantLength = 0.4f;
    cfg.maxGeometryError = 0.05f;
    cfg.maxIterations = 3;
    RxMeshBackend backend;
    RemeshReport report;
    if (!backend.remesh(mesh, cfg, report) || !report.topologyValid || !mesh.validate()) {
      std::cerr << "cylinder remesh failed topology=" << report.topologyValid << '\n';
      return test_result("test_rxmesh_project");
    }
    CHECK(report.movedLockedVertices == 0);
    CHECK(report.splits > 0);

    int movedLocked = 0;
    std::vector<Vec3> lockedAfter;
    for (int v = 0; v < mesh.vertexCount(); ++v)
      if (VertexConstraint(mesh.vertexConstraint[v]) == VertexConstraint::Locked)
        lockedAfter.push_back(mesh.position(v));
    CHECK(int(lockedAfter.size()) == int(locked.size()));
    for (const auto &p : locked) {
      float best = 1e30f;
      for (const auto &q : lockedAfter) best = std::min(best, length2(p - q));
      if (best > 1e-10f) ++movedLocked;
    }
    CHECK(movedLocked == 0);

    GeometryProjector proj;
    mesh.rebuildTopology();
    proj.build(mesh);
    const float bodyTol = 1e-5f * R;
    const float planeTol = 1e-5f * std::max(mesh.bboxDiagonal(), 1e-6f);
    int bodyInterior = 0;
    for (int v = 0; v < mesh.vertexCount(); ++v) {
      const Vec3 p = mesh.position(v);
      const auto c = VertexConstraint(mesh.vertexConstraint[v]);
      if (c == VertexConstraint::Locked) {
        const float radial = std::sqrt(p.x * p.x + p.y * p.y);
        CHECK_NEAR(radial, R, 1e-4f);
        CHECK(std::fabs(p.z - z0) < 1e-4f || std::fabs(p.z - z1) < 1e-4f);
        continue;
      }
      if (c == VertexConstraint::PatchBoundary || c == VertexConstraint::Corner) continue;
      const uint32_t patch = mesh.vertexPatchId[v];
      const auto hit = proj.projectSurface(patch, p);
      CHECK(hit.ok);
      if (patch == 0) {
        ++bodyInterior;
        CHECK_NEAR(std::sqrt(p.x * p.x + p.y * p.y), R, bodyTol);
        CHECK(distance(p, hit.position) <= bodyTol);
      } else {
        CHECK_NEAR(p.z, patch == 1 ? z0 : z1, planeTol);
        CHECK(distance(p, hit.position) <= planeTol);
      }
    }
    CHECK(bodyInterior > 0);

    for (const auto &e : mesh.edges) {
      if (!(e.flags & EdgePatchBoundary)) continue;
      const Vec3 a = mesh.position(int(e.v0));
      const Vec3 b = mesh.position(int(e.v1));
      CHECK(std::fabs(a.z - z0) < 1e-3f || std::fabs(a.z - z1) < 1e-3f);
      CHECK(std::fabs(b.z - z0) < 1e-3f || std::fabs(b.z - z1) < 1e-3f);
      CHECK_NEAR(std::sqrt(a.x * a.x + a.y * a.y), R, 1e-3f);
      CHECK_NEAR(std::sqrt(b.x * b.x + b.y * b.y), R, 1e-3f);
    }
  }

  return test_result("test_rxmesh_project");
}
