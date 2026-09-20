#include "cad_adaptive/RelaxProject.h"
#include <unordered_set>

namespace cad_adaptive {
namespace {

bool AcceptRelocation(const SemanticMesh& mesh, int v, Vec3 candidate) {
  constexpr float kArea2Eps = 1e-12f;
  constexpr float kQualityFloor = 1e-8f;
  for (int f : mesh.incidentFaces[v]) {
    if (f < 0 || f >= mesh.faceCount() || !mesh.faceAlive[f]) continue;
    const auto fv = mesh.face(f);
    Vec3 oldP[3] = {mesh.position(fv[0]), mesh.position(fv[1]), mesh.position(fv[2])};
    Vec3 newP[3] = {oldP[0], oldP[1], oldP[2]};
    for (int k = 0; k < 3; ++k) if (fv[k] == v) newP[k] = candidate;

    const Vec3 oldCross = cross(oldP[1] - oldP[0], oldP[2] - oldP[0]);
    const Vec3 newCross = cross(newP[1] - newP[0], newP[2] - newP[0]);
    const float oldArea2 = length(oldCross);
    const float newArea2 = length(newCross);
    if (!(newArea2 > kArea2Eps)) return false;
    if (oldArea2 > kArea2Eps && dot(oldCross, newCross) <= 0.0f) return false;

    const float oldQ = triangleQuality(oldP[0], oldP[1], oldP[2]);
    const float newQ = triangleQuality(newP[0], newP[1], newP[2]);
    // Never create a near-degenerate face. For already-poor faces allow only
    // a small temporary decrease; otherwise relocation gets trapped before
    // neighboring vertices have a chance to improve the same fan.
    if (!(newQ > kQualityFloor)) return false;
    if (oldQ > kQualityFloor && newQ + 1e-7f < 0.90f * oldQ) return false;
  }
  return true;
}

} // namespace

int relaxAndProject(SemanticMesh& mesh, const GeometryProjector& reference,
                    int iterations, float lambda) {
  if (iterations <= 0 || !(lambda > 0.0f)) return 0;
  lambda = clampf(lambda, 0.0f, 1.0f);
  int moved = 0;
  for (int it = 0; it < iterations; ++it) {
    mesh.rebuildTopology();
    mesh.computeVertexNormals();
    std::vector<Vec3> next(size_t(mesh.vertexCount()));
    std::vector<uint8_t> change(size_t(mesh.vertexCount()), 0);
    for (int v = 0; v < mesh.vertexCount(); ++v) {
      const auto constraint = VertexConstraint(mesh.vertexConstraint[v]);
      const Vec3 p = mesh.position(v);
      next[size_t(v)] = p;
      if (constraint == VertexConstraint::Locked || constraint == VertexConstraint::Corner)
        continue;

      std::unordered_set<int> neighbors;
      for (int f : mesh.incidentFaces[v]) {
        if (f < 0 || f >= mesh.faceCount() || !mesh.faceAlive[f]) continue;
        for (int u : mesh.face(f)) if (u != v) neighbors.insert(u);
      }
      if (neighbors.empty()) continue;

      Vec3 avg{};
      for (int u : neighbors) avg = avg + mesh.position(u);
      avg = avg * (1.0f / float(neighbors.size()));
      Vec3 delta = avg - p;
      if (!isBoundaryConstraint(constraint)) {
        const Vec3 n = normalize(mesh.normal(v));
        delta = delta - n * dot(delta, n);
      }

      // Backtracking relocation: project every trial to the immutable source
      // surface/feature, then accept only if all incident faces remain valid.
      float step = lambda;
      for (int attempt = 0; attempt < 5; ++attempt, step *= 0.5f) {
        const Vec3 trial = p + delta * step;
        const ProjectionResult hit = reference.projectVertex(mesh, v, trial);
        if (!hit.ok) continue;
        if (!AcceptRelocation(mesh, v, hit.position)) continue;
        if (length2(hit.position - p) <= 1e-16f) break;
        next[size_t(v)] = hit.position;
        change[size_t(v)] = 1;
        break;
      }
    }
    for (int v = 0; v < mesh.vertexCount(); ++v) {
      if (!change[size_t(v)]) continue;
      mesh.setPosition(v, next[size_t(v)]);
      ++moved;
    }
  }
  mesh.rebuildTopology();
  mesh.computeVertexNormals();
  return moved;
}

} // namespace cad_adaptive
