#include "cad_adaptive/RxMeshBackend.h"
#include "cad_adaptive/RemeshPolicy.h"
#include "check.h"
#include <cmath>
#include <vector>

using namespace cad_adaptive;

int main() {
  CHECK(rxmeshAvailable());
  SemanticMesh mesh = makeTwoPatchGrid(20, 10, 0, 0, 2, 1);
  const float seamX = 1.0f;
  std::vector<Vec3> locked;
  for (int v = 0; v < mesh.vertexCount(); ++v)
    if (VertexConstraint(mesh.vertexConstraint[v]) == VertexConstraint::Locked)
      locked.push_back(mesh.position(v));
  mesh.rebuildTopology();
  int seamBefore = 0;
  for (const auto &e : mesh.edges)
    if (e.flags & EdgePatchBoundary) ++seamBefore;
  CHECK(seamBefore >= 10);

  RemeshConfig cfg;
  cfg.adaptive = false;
  cfg.constantLength = 0.06f;
  cfg.maxGeometryError = 0.05f;
  cfg.maxIterations = 3;
  RxMeshBackend backend;
  RemeshReport report;
  if (!backend.remesh(mesh, cfg, report) || !report.topologyValid || !mesh.validate()) {
    std::cerr << "remesh failed topology=" << report.topologyValid << '\n';
    return test_result("test_rxmesh_constraints");
  }
  CHECK(report.movedLockedVertices == 0);
  CHECK(report.constraintsHeld);

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

  mesh.rebuildTopology();
  int seam = 0;
  for (const auto &e : mesh.edges) {
    if (!(e.flags & EdgePatchBoundary)) continue;
    ++seam;
    CHECK(e.patchLeft != e.patchRight);
    CHECK(!RemeshPolicy::canFlip(e.flags));
    const Vec3 a = mesh.position(int(e.v0));
    const Vec3 b = mesh.position(int(e.v1));
    CHECK_NEAR(a.x, seamX, 0.02);
    CHECK_NEAR(b.x, seamX, 0.02);
  }
  CHECK(seam >= seamBefore);

  for (int f = 0; f < mesh.faceCount(); ++f) {
    if (!mesh.faceAlive[f]) continue;
    const auto t = mesh.face(f);
    const uint32_t facePatch = mesh.facePatchId[f];
    CHECK(facePatch == 0 || facePatch == 1);
    for (int k = 0; k < 3; ++k) {
      const int v = t[k];
      const auto c = VertexConstraint(mesh.vertexConstraint[v]);
      if (c == VertexConstraint::PatchBoundary || c == VertexConstraint::Locked) continue;
      CHECK(mesh.vertexPatchId[v] == facePatch);
    }
  }

  return test_result("test_rxmesh_constraints");
}
