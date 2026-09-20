#include "cad_adaptive/RxMeshBackend.h"
#include "check.h"
#include <string>

using namespace cad_adaptive;

int main() {
  CHECK(rxmeshAvailable());
  SemanticMesh mesh = makeGrid(24, 24, 0, 0, 1, 1, 0);
  lockMeshBoundary(mesh);
  std::vector<Vec3> locked;
  for (int v = 0; v < mesh.vertexCount(); ++v)
    if (VertexConstraint(mesh.vertexConstraint[v]) == VertexConstraint::Locked)
      locked.push_back(mesh.position(v));

  RemeshConfig cfg;
  cfg.adaptive = false;
  cfg.constantLength = 0.08f;
  cfg.maxGeometryError = 0.05f;
  cfg.maxIterations = 3;
  RxMeshBackend backend;
  RemeshReport report;
  CHECK(backend.remesh(mesh, cfg, report));
  CHECK(report.topologyValid);
  CHECK(mesh.validate());
  CHECK(report.splits + report.collapses + report.flips > 0);
  CHECK(report.splitCandidates >= report.splits);
  CHECK(report.collapseCandidates >= report.collapses);
  CHECK(report.flipCandidates >= report.flips);
  CHECK(report.collapseCandidates > 0);
  CHECK(report.flipCandidates > 0);
  CHECK(report.rejectTopology == 0 && report.rejectPatch == 0 && report.rejectFeature == 0);
  CHECK(report.rejectNormal == 0 && report.rejectQuality == 0 && report.rejectError == 0);

  int moved = 0;
  std::vector<Vec3> after;
  for (int v = 0; v < mesh.vertexCount(); ++v)
    if (VertexConstraint(mesh.vertexConstraint[v]) == VertexConstraint::Locked)
      after.push_back(mesh.position(v));
  CHECK(after.size() == locked.size());
  for (size_t i = 0; i < locked.size(); ++i) {
    // Compact reorders vertices; match by nearest leftover locked position.
  }
  for (const auto &p : locked) {
    float best = 1e9f;
    for (const auto &q : after) best = std::min(best, length2(p - q));
    if (best > 1e-10f) ++moved;
  }
  CHECK(moved == 0);
  CHECK(report.geometryErrorMax <= cfg.maxGeometryError + 1e-3f || report.geometryErrorMax == 0);

  SemanticMesh cube = makeGrid(4, 4, 0, 0, 1, 1, 0);
  lockMeshBoundary(cube);
  cfg.constantLength = 0.15f;
  cfg.maxIterations = 2;
  RemeshReport cubeReport;
  CHECK(backend.remesh(cube, cfg, cubeReport));
  CHECK(cubeReport.topologyValid);
  CHECK(cubeReport.splitCandidates > 0);
  CHECK(cubeReport.splitCandidates >= cubeReport.splits);

  auto stationary = makeGrid(8, 8, 0, 0, 1, 1, 0);
  lockMeshBoundary(stationary);
  cfg.enableSplit = cfg.enableCollapse = cfg.enableFlip = false;
  cfg.enableSmooth = true;
  cfg.smoothLambda = 0;
  RemeshReport noMoves;
  CHECK(backend.remesh(stationary, cfg, noMoves));
  CHECK(noMoves.smoothMoves == 0);
  CHECK(noMoves.splitCandidates == 0 && noMoves.collapseCandidates == 0 &&
        noMoves.flipCandidates == 0);

  return test_result("test_rxmesh_isotropic");
}
