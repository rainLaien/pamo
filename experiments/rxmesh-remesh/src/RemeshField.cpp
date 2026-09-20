#include "cad_adaptive/RemeshField.h"
#include "cad_adaptive/BoundarySizingField.h"
#include <limits>

namespace cad_adaptive {

SizingLimits RemeshField::limits(const SemanticMesh &mesh, const RemeshConfig &config) {
  SizingLimits lim;
  lim.epsilon = config.maxGeometryError > 0 ? config.maxGeometryError : 1e-6f;
  lim.hMin = config.hMin > 0 ? config.hMin : lim.epsilon;
  const float derived = mesh.bboxDiagonal() * 0.05f;
  if (config.hMax > 0)
    lim.hMax = config.hMax;
  else if (config.constantLength > 0)
    lim.hMax = config.constantLength;
  else
    lim.hMax = derived > lim.hMin ? derived : lim.hMin * 8;
  if (lim.hMax < lim.hMin) lim.hMax = lim.hMin;
  return lim;
}

float RemeshField::patchLength(const PatchRecord &patch, const SizingLimits &lim, float hConst) {
  if (patch.type == PatchType::Plane) return lim.hMax;
  if (patch.type == PatchType::Cylinder && patch.radius > 0)
    return std::sqrt(8.0f * patch.radius * lim.epsilon);
  if (hConst > 0) return hConst;
  return lim.hMax;
}

float RemeshField::curvatureLength(float kmax, float epsilon, float hMax) {
  if (!(kmax > 0)) return hMax;
  return std::sqrt(8.0f * epsilon / (kmax + 1e-12f));
}

float RemeshField::featureLength(float distance, float band, float hFeatureEdge, float hRegular) {
  if (!(band > 0)) return hRegular;
  const float t = smoothstep(clampf(distance / band, 0, 1));
  return lerp(hFeatureEdge, hRegular, t);
}

float RemeshField::combine(float hCurvature, float hFeature, float hPatch, float hError,
                           const SizingLimits &lim) {
  const float h = std::min(std::min(hCurvature, hFeature), std::min(hPatch, hError));
  return clampf(h, lim.hMin, lim.hMax);
}

void RemeshField::computeFeatureDistance(SemanticMesh &mesh) {
  const int n = mesh.vertexCount();
  mesh.featureDistance.assign(n, std::numeric_limits<float>::max());
  for (const auto &e : mesh.edges) {
    if ((e.flags & (EdgePatchBoundary | EdgeMeshBoundary | EdgeSharp | EdgeProtected)) == 0)
      continue;
    const Vec3 a = mesh.position(int(e.v0));
    const Vec3 b = mesh.position(int(e.v1));
    for (int v = 0; v < n; ++v) {
      const float d = distance(mesh.position(v), closestOnSegment(mesh.position(v), a, b));
      mesh.featureDistance[v] = std::min(mesh.featureDistance[v], d);
    }
  }
  for (int v = 0; v < n; ++v) {
    if (mesh.featureDistance[v] > 1e20f) mesh.featureDistance[v] = 1e20f;
  }
}

void RemeshField::compute(SemanticMesh &mesh, const RemeshConfig &config,
                          const GeometryProjector &projector) {
  if (mesh.LocalSizing) {
    mesh.LocalSizing->apply(mesh);
    return;
  }
  const SizingLimits lim = limits(mesh, config);
  const int n = mesh.vertexCount();
  mesh.targetLength.resize(n);
  mesh.curvature.resize(n);
  if (int(mesh.featureDistance.size()) != n) computeFeatureDistance(mesh);

  for (int v = 0; v < n; ++v) {
    if (!config.adaptive && config.constantLength > 0) {
      mesh.curvature[v] = projector.analyticKmax(mesh.vertexPatchId[v]);
      mesh.targetLength[v] = config.constantLength;
      continue;
    }
    const uint32_t patchId = mesh.vertexPatchId[v];
    const PatchRecord *rec = projector.patch(patchId);
    PatchRecord fallback;
    if (!rec) {
      fallback.type = PatchType::Unknown;
      rec = &fallback;
    }
    const float kmax = projector.analyticKmax(patchId);
    mesh.curvature[v] = kmax;
    const float hConst = config.constantLength;
    const float hPatch = patchLength(*rec, lim, hConst);
    const float hCurv = curvatureLength(kmax, lim.epsilon, lim.hMax);
    const float hRegular = clampf(std::min(hPatch, hCurv), lim.hMin, lim.hMax);
    const float band = config.featureBand > 0 ? config.featureBand : 4.0f * hRegular;
    // Geometry tolerance is not a mesh-size target. Using 2*epsilon here
    // drove raw STL crease neighborhoods orders of magnitude below the global
    // target (e.g. h=1.8 -> hFeature=0.02), causing split explosions and
    // needle triangles. Unless explicitly overridden, keep the feature itself
    // at the regular local size and let featureBand control only the transition.
    const float hFeatEdge =
        config.featureEdgeLength > 0 ? config.featureEdgeLength : hRegular;
    const float hFeat = featureLength(mesh.featureDistance[v], band, hFeatEdge, hRegular);
    mesh.targetLength[v] = combine(hCurv, hFeat, hPatch, lim.hMax, lim);
  }
}

} // namespace cad_adaptive
