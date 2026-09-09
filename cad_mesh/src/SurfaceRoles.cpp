#include "CadMesh/SurfaceRoles.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace CadMesh {
namespace {
struct BoundarySample {
  Vec3 Position, Normal;
};
struct SupportBoundary {
  int PatchId = -1;
  std::vector<int> PatchIds;
  double Length = 0, TangentLength = 0;
  std::vector<BoundarySample> Samples;
};

void LimitSamples(SupportBoundary &boundary) {
  if (boundary.Samples.size() <= 48)
    return;
  std::vector<BoundarySample> samples;
  for (size_t i = 0; i < 48; ++i)
    samples.push_back(boundary.Samples[i * boundary.Samples.size() / 48]);
  boundary.Samples = std::move(samples);
}

bool SameMotherPlane(const MeshPatch &a, const MeshPatch &b,
                     double fittingTolerance) {
  const auto *x = std::get_if<PlaneParameters>(&a.Parameters);
  const auto *y = std::get_if<PlaneParameters>(&b.Parameters);
  if (!x || !y)
    return false;
  const Vec3 nx = Normalize(ToVec(x->Plane.Normal));
  const Vec3 ny = Normalize(ToVec(y->Plane.Normal));
  const Vec3 offset = Sub(ToVec(x->Plane.Origin), ToVec(y->Plane.Origin));
  const double distance = std::min(3 * fittingTolerance,
      std::max(.2 * fittingTolerance, 3 * (a.MaxFittingError + b.MaxFittingError)));
  return std::abs(Dot(nx, ny)) >= 1 - 1e-8 &&
         std::abs(Dot(offset, nx)) <= distance &&
         std::abs(Dot(offset, ny)) <= distance;
}

bool AnalyticNormal(const MeshPatch &patch, const Vec3 &point, Vec3 &normal) {
  if (const auto *p = std::get_if<PlaneParameters>(&patch.Parameters)) {
    normal = ToVec(p->Plane.Normal);
  } else if (const auto *p =
                 std::get_if<CylinderParameters>(&patch.Parameters)) {
    Vec3 a = Normalize(ToVec(p->Axis.Direction));
    Vec3 v = Sub(point, ToVec(p->Axis.Origin));
    normal = Sub(v, Mul(a, Dot(v, a)));
  } else if (const auto *p = std::get_if<ConeParameters>(&patch.Parameters)) {
    Vec3 a = Normalize(ToVec(p->Axis.Direction));
    Vec3 v = Sub(point, ToVec(p->Axis.Origin));
    double h = Dot(v, a);
    Vec3 radial = Normalize(Sub(v, Mul(a, h)));
    normal = Sub(Mul(radial, std::cos(p->SemiAngle)),
                 Mul(a, std::sin(p->SemiAngle) * (h < 0 ? -1 : 1)));
  } else if (const auto *p = std::get_if<SphereParameters>(&patch.Parameters)) {
    normal = Sub(point, ToVec(p->Center));
  } else if (const auto *p = std::get_if<TorusParameters>(&patch.Parameters)) {
    Vec3 a = Normalize(ToVec(p->Axis.Direction));
    Vec3 v = Sub(point, ToVec(p->Axis.Origin));
    Vec3 radial = Normalize(Sub(v, Mul(a, Dot(v, a))));
    normal = Sub(v, Mul(radial, p->MajorRadius));
  } else {
    return false;
  }
  if (Norm(normal) <= 1e-20)
    return false;
  normal = Normalize(normal);
  return true;
}

Vec3 SurfaceNormal(const MeshPatch &patch, const MeshTriangle &face,
                   const Vec3 &point, bool &analytic) {
  Vec3 normal;
  analytic = AnalyticNormal(patch, point, normal);
  if (!analytic)
    return Normalize(ToVec(face.Normal));
  if (Dot(normal, ToVec(face.Normal)) < 0)
    normal = Mul(normal, -1);
  return normal;
}

bool SameCurvedSurface(const MeshPatch &a, const MeshPatch &b, double tolerance) {
  const auto *x = std::get_if<CylinderParameters>(&a.Parameters);
  const auto *y = std::get_if<CylinderParameters>(&b.Parameters);
  if (x && y) {
    Vec3 axis = Normalize(ToVec(x->Axis.Direction));
    Vec3 offset = Sub(ToVec(y->Axis.Origin), ToVec(x->Axis.Origin));
    return std::abs(x->Radius - y->Radius) <= tolerance &&
           std::abs(Dot(axis, Normalize(ToVec(y->Axis.Direction)))) > .9999 &&
           Norm(Sub(offset, Mul(axis, Dot(offset, axis)))) <= tolerance;
  }
  const auto *u = std::get_if<TorusParameters>(&a.Parameters);
  const auto *v = std::get_if<TorusParameters>(&b.Parameters);
  return u && v && std::abs(u->MajorRadius - v->MajorRadius) <= tolerance &&
         std::abs(u->MinorRadius - v->MinorRadius) <= tolerance &&
         Distance(u->Axis.Origin, v->Axis.Origin) <= tolerance &&
         std::abs(Dot(Normalize(ToVec(u->Axis.Direction)),
                       Normalize(ToVec(v->Axis.Direction)))) > .9999;
}

SupportBoundary MeasureBoundary(const MeshTopology &mesh,
                                const std::vector<MeshPatch> &patches,
                                const MeshPatch &candidate,
                                const PatchAdjacency &adj) {
  SupportBoundary result;
  result.PatchId = adj.Patch0 == candidate.Id ? adj.Patch1 : adj.Patch0;
  result.PatchIds = {result.PatchId};
  const auto &support = patches[result.PatchId];
  const double meshCos = std::cos(25.0 * std::acos(-1.0) / 180.0);
  const double analyticCos = std::cos(12.0 * std::acos(-1.0) / 180.0);
  for (int edgeId : adj.SharedBoundaryEdges) {
    const auto &edge = mesh.getEdges()[edgeId];
    double length = Distance(mesh.getVertices()[edge.Vertex0].Position,
                             mesh.getVertices()[edge.Vertex1].Position);
    result.Length += length;
    if (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature ||
        edge.IncidentTriangleIds.size() != 2)
      continue;
    int first = edge.IncidentTriangleIds[0];
    int second = edge.IncidentTriangleIds[1];
    if (mesh.getTriangles()[first].PatchId != candidate.Id)
      std::swap(first, second);
    const auto &a = mesh.getTriangles()[first];
    const auto &b = mesh.getTriangles()[second];
    if (a.PatchId != candidate.Id || b.PatchId != support.Id)
      continue;
    Vec3 point = Mul(Add(ToVec(mesh.getVertices()[edge.Vertex0].Position),
                         ToVec(mesh.getVertices()[edge.Vertex1].Position)), .5);
    bool analyticA = false, analyticB = false;
    Vec3 normalA = SurfaceNormal(candidate, a, point, analyticA);
    Vec3 normalB = SurfaceNormal(support, b, point, analyticB);
    if (Dot(ToVec(a.Normal), ToVec(b.Normal)) < meshCos ||
        Dot(normalA, normalB) < (analyticA && analyticB ? analyticCos : meshCos))
      continue;
    result.TangentLength += length;
    result.Samples.push_back({point, normalA});
  }
  // Bounded sampling keeps role annotation linear on large shared boundaries.
  LimitSamples(result);
  return result;
}

bool HasTransverseTurn(const SupportBoundary &a, const SupportBoundary &b) {
  std::vector<double> angles;
  for (const auto &sample : a.Samples) {
    double best = std::numeric_limits<double>::infinity();
    const BoundarySample *nearest = nullptr;
    for (const auto &other : b.Samples) {
      double distance = Norm(Sub(sample.Position, other.Position));
      if (distance < best) {
        best = distance;
        nearest = &other;
      }
    }
    if (nearest)
      angles.push_back(std::acos(std::clamp(Dot(sample.Normal, nearest->Normal),
                                            -1.0, 1.0)));
  }
  if (angles.empty())
    return false;
  std::sort(angles.begin(), angles.end());
  const double turn = angles[angles.size() / 2];
  return turn > 12 * std::acos(-1.0) / 180 &&
         turn < 168 * std::acos(-1.0) / 180;
}
} // namespace

void IdentifySurfaceRoles(const MeshTopology &mesh,
                          std::vector<MeshPatch> &patches,
                          const std::vector<PatchAdjacency> &adjacency) {
  std::vector<std::vector<const PatchAdjacency *>> incident(patches.size());
  for (const auto &adj : adjacency)
    if (adj.Patch0 >= 0 && adj.Patch1 >= 0 &&
        adj.Patch0 < int(patches.size()) && adj.Patch1 < int(patches.size())) {
      incident[adj.Patch0].push_back(&adj);
      incident[adj.Patch1].push_back(&adj);
    }
  for (auto &patch : patches) {
    patch.FeatureRole = PatchFeatureRole::Ordinary;
    patch.SupportPatchIds.clear();
    if (patch.SurfaceType != PatchSurfaceType::Cylinder &&
        patch.SurfaceType != PatchSurfaceType::Torus &&
        patch.SurfaceType != PatchSurfaceType::Freeform)
      continue;
    if (!patch.HasConsistentFaceOrientation || patch.Id < 0 ||
        patch.Id >= int(incident.size()) || incident[patch.Id].size() < 2)
      continue;
    double area = 0, perimeter = 0;
    bool nonManifold = false;
    for (int triangle : patch.TriangleIds)
      area += mesh.getTriangles()[triangle].Area;
    for (int edgeId : patch.BoundaryEdgeIds) {
      const auto &edge = mesh.getEdges()[edgeId];
      perimeter += Distance(mesh.getVertices()[edge.Vertex0].Position,
                            mesh.getVertices()[edge.Vertex1].Position);
      nonManifold = nonManifold || edge.IsNonManifold;
    }
    if (nonManifold || !(area > 0) || !(perimeter > 0))
      continue;
    std::vector<SupportBoundary> supports;
    for (const auto *adj : incident[patch.Id]) {
      int other = adj->Patch0 == patch.Id ? adj->Patch1 : adj->Patch0;
      // Unclassified mother surfaces and fragments of the same cylinder do
      // not provide enough evidence for a semantic fillet assignment.
      if (patches[other].SurfaceType == PatchSurfaceType::Freeform ||
          patches[other].SurfaceType == PatchSurfaceType::Unknown ||
          !patches[other].HasConsistentFaceOrientation ||
          SameCurvedSurface(patch, patches[other],
                        3 * mesh.getResolution().FittingTolerance))
        continue;
      auto boundary = MeasureBoundary(mesh, patches, patch, *adj);
      if (!(boundary.Length > 0) || boundary.TangentLength < .85 * boundary.Length)
        continue;
      // Fragmented labels on one physical mother plane form one support side.
      // Compare every group member to prevent a chain of near-equal planes
      // from combining distinct supports. This changes role evidence only.
      auto group = std::find_if(supports.begin(), supports.end(),
          [&](const SupportBoundary &existing) {
            return std::all_of(existing.PatchIds.begin(), existing.PatchIds.end(),
                [&](int id) { return SameMotherPlane(patches[id], patches[other],
                    mesh.getResolution().FittingTolerance); });
          });
      if (group == supports.end()) {
        supports.push_back(std::move(boundary));
      } else {
        group->PatchIds.push_back(other);
        group->Length += boundary.Length;
        group->TangentLength += boundary.TangentLength;
        group->Samples.insert(group->Samples.end(), boundary.Samples.begin(),
                              boundary.Samples.end());
      }
    }
    for (auto &support : supports)
      LimitSamples(support);
    std::sort(supports.begin(), supports.end(),
              [](const SupportBoundary &a, const SupportBoundary &b) {
                return a.TangentLength > b.TangentLength;
              });
    if (supports.size() < 2)
      continue;
    // A fillet band has two dominant long, tangent sides and a real normal
    // turn across its width. End trims may remain sharp/open boundaries.
    const auto &a = supports[0];
    const auto &b = supports[1];
    double length = .5 * (a.TangentLength + b.TangentLength);
    double width = area / length;
    if (a.TangentLength + b.TangentLength < .55 * perimeter ||
        std::min(a.TangentLength, b.TangentLength) < .35 * length ||
        length < 1.2 * width || !HasTransverseTurn(a, b))
      continue;
    if (const auto *c = std::get_if<CylinderParameters>(&patch.Parameters))
      if (width > 1.05 * std::acos(-1.0) * c->Radius)
        continue;
    patch.FeatureRole = PatchFeatureRole::Fillet;
    // Keep the two-mother handoff contract: each physical support side names
    // an actually adjacent representative label, even when evidence was
    // accumulated over several equivalent labels.
    patch.SupportPatchIds = {a.PatchId, b.PatchId};
    std::sort(patch.SupportPatchIds.begin(), patch.SupportPatchIds.end());
  }
}
} // namespace CadMesh
