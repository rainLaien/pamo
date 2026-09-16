#include "cad_adaptive/GeometryProjector.h"
#include <limits>

namespace cad_adaptive {
namespace {

Vec3 closestOnTriangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
  const Vec3 ab = b - a, ac = c - a, ap = p - a;
  const float d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const Vec3 bp = p - b;
  const float d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
  const Vec3 cp = p - c;
  const float d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
    return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  const float inv = 1.0f / (va + vb + vc);
  return a + ab * (vb * inv) + ac * (vc * inv);
}

} // namespace

ProjectionResult GeometryProjector::projectPlane(Vec3 origin, Vec3 normal, Vec3 p) {
  const Vec3 n = normalize(normal);
  ProjectionResult r;
  r.ok = true;
  r.position = p - n * dot(p - origin, n);
  return r;
}

ProjectionResult GeometryProjector::projectCylinder(Vec3 origin, Vec3 axis, float radius, Vec3 p) {
  ProjectionResult r;
  if (!(radius > 0)) return r;
  const Vec3 a = normalize(axis);
  const Vec3 offset = p - origin;
  const float height = dot(offset, a);
  const Vec3 radial = offset - a * height;
  const float rn = length(radial);
  if (!(rn > 1e-20f)) return r;
  r.ok = true;
  r.position = origin + a * height + radial * (radius / rn);
  return r;
}

void GeometryProjector::build(const SemanticMesh &mesh) {
  patches_ = mesh.patches;
  for (auto &p : patches_) {
    p.axis = normalize(p.axis);
  }
  referenceTriangles_.clear();
  referenceTriangles_.reserve(mesh.faceCount());
  for (int f = 0; f < mesh.faceCount(); ++f) {
    if (!mesh.faceAlive[f]) continue;
    ReferenceTriangle t;
    t.patchId = mesh.facePatchId[f];
    t.a = mesh.facePoint(f, 0);
    t.b = mesh.facePoint(f, 1);
    t.c = mesh.facePoint(f, 2);
    referenceTriangles_.push_back(t);
  }
  segments_.clear();
  for (const auto &e : mesh.edges) {
    if ((e.flags & (EdgePatchBoundary | EdgeMeshBoundary | EdgeSharp | EdgeProtected)) == 0)
      continue;
    Segment s;
    s.feature = e.featureCurveId;
    s.a = mesh.position(int(e.v0));
    s.b = mesh.position(int(e.v1));
    segments_.push_back(s);
  }
}

ProjectionResult GeometryProjector::projectSurface(uint32_t patchId, Vec3 p) const {
  const PatchRecord *rec = patch(patchId);
  if (rec) {
    if (rec->type == PatchType::Plane) return projectPlane(rec->origin, rec->axis, p);
    if (rec->type == PatchType::Cylinder)
      return projectCylinder(rec->origin, rec->axis, rec->radius, p);
  }

  ProjectionResult best;
  float bestD = std::numeric_limits<float>::max();
  for (const auto &t : referenceTriangles_) {
    if (t.patchId != patchId) continue;
    const Vec3 q = closestOnTriangle(p, t.a, t.b, t.c);
    const float d = length2(p - q);
    if (d < bestD) {
      bestD = d;
      best.ok = true;
      best.position = q;
    }
  }
  return best;
}

ProjectionResult GeometryProjector::projectFeature(uint32_t featureCurveId, Vec3 p) const {
  ProjectionResult best;
  float bestD = std::numeric_limits<float>::max();
  for (const auto &s : segments_) {
    if (featureCurveId != 0 && s.feature != featureCurveId) continue;
    const Vec3 q = closestOnSegment(p, s.a, s.b);
    const float d = length2(p - q);
    if (d < bestD) {
      bestD = d;
      best.ok = true;
      best.position = q;
    }
  }
  return best;
}

ProjectionResult GeometryProjector::projectVertex(const SemanticMesh &mesh, int v, Vec3 p) const {
  const auto c = VertexConstraint(mesh.vertexConstraint[v]);
  if (c == VertexConstraint::Locked || c == VertexConstraint::Corner) {
    ProjectionResult r;
    r.ok = true;
    r.position = mesh.position(v);
    return r;
  }
  if (isBoundaryConstraint(c)) {
    uint32_t feature = 0;
    for (const auto &e : mesh.edges) {
      if (e.v0 != uint32_t(v) && e.v1 != uint32_t(v)) continue;
      if (e.featureCurveId != 0) {
        feature = e.featureCurveId;
        break;
      }
    }
    auto r = projectFeature(feature, p);
    if (r.ok) return r;
  }
  return projectSurface(mesh.vertexPatchId[v], p);
}

Vec3 GeometryProjector::analyticNormal(uint32_t patchId, Vec3 p) const {
  const PatchRecord *rec = patch(patchId);
  if (!rec) return {0, 0, 0};
  if (rec->type == PatchType::Plane) return normalize(rec->axis);
  if (rec->type == PatchType::Cylinder) {
    const Vec3 a = normalize(rec->axis);
    const Vec3 radial = (p - rec->origin) - a * dot(p - rec->origin, a);
    return normalize(radial);
  }
  return {0, 0, 0};
}

float GeometryProjector::analyticKmax(uint32_t patchId) const {
  const PatchRecord *rec = patch(patchId);
  if (!rec) return 0;
  if (rec->type == PatchType::Cylinder && rec->radius > 0) return 1.0f / rec->radius;
  return 0;
}

} // namespace cad_adaptive
