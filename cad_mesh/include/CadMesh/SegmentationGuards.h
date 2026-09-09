#pragma once

#include "CadMesh/MeshTopology.h"
#include <limits>

namespace CadMesh {

// A partition boundary is a remeshing constraint, not merely a weak label cue.
inline bool IsHardSegmentationBoundary(const MeshEdge &edge,
                                       const SegmentationConfig &config) {
  return edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature ||
         edge.IncidentTriangleIds.size() != 2 ||
         edge.BoundaryScore >= config.StrongBoundaryThreshold;
}

struct SurfaceCompatibility {
  bool Supported = false;
  double MaxNormalizedDistance = 0;
  double MaxNormalError = 0;
  int ReliableNormalSamples = 0;
};

namespace SegmentationGuardDetail {
inline bool SurfaceSample(const MeshPatch &target, const Vec3 &point,
                          double &distance, Vec3 &normal) {
  if (target.SurfaceType == PatchSurfaceType::Plane) {
    const auto *plane = std::get_if<PlaneParameters>(&target.Parameters);
    if (!plane || Norm(ToVec(plane->Plane.Normal)) < .5)
      return false;
    normal = Normalize(ToVec(plane->Plane.Normal));
    distance = std::abs(Dot(Sub(point, ToVec(plane->Plane.Origin)), normal));
  } else if (target.SurfaceType == PatchSurfaceType::Cylinder) {
    const auto *cylinder = std::get_if<CylinderParameters>(&target.Parameters);
    if (!cylinder || !std::isfinite(cylinder->Radius) ||
        cylinder->Radius <= 0 || Norm(ToVec(cylinder->Axis.Direction)) < .5)
      return false;
    Vec3 axis = Normalize(ToVec(cylinder->Axis.Direction));
    Vec3 delta = Sub(point, ToVec(cylinder->Axis.Origin));
    Vec3 radial = Sub(delta, Mul(axis, Dot(delta, axis)));
    double radius = Norm(radial);
    if (!std::isfinite(radius) || radius <= 0)
      return false;
    normal = Mul(radial, 1.0 / radius);
    distance = std::abs(radius - cylinder->Radius);
  } else if (target.SurfaceType == PatchSurfaceType::Sphere) {
    const auto *sphere = std::get_if<SphereParameters>(&target.Parameters);
    if (!sphere || !std::isfinite(sphere->Radius) || sphere->Radius <= 0)
      return false;
    Vec3 radial = Sub(point, ToVec(sphere->Center));
    double radius = Norm(radial);
    if (!std::isfinite(radius) || radius <= 0)
      return false;
    normal = Mul(radial, 1.0 / radius);
    distance = std::abs(radius - sphere->Radius);
  } else if (target.SurfaceType == PatchSurfaceType::Cone) {
    const auto *cone = std::get_if<ConeParameters>(&target.Parameters);
    if (!cone || cone->SemiAngle <= 0 || cone->SemiAngle >= 1.5707963267948966)
      return false;
    const Vec3 axis = Normalize(ToVec(cone->Axis.Direction));
    const Vec3 delta = Sub(point, ToVec(cone->Axis.Origin));
    const double height = Dot(delta, axis);
    if (height < -1e-12)
      return false;
    const Vec3 radial = Sub(delta, Mul(axis, height));
    const double radius = Norm(radial);
    if (radius <= 1e-30)
      return false;
    normal = Sub(Mul(radial, std::cos(cone->SemiAngle) / radius),
                 Mul(axis, std::sin(cone->SemiAngle)));
    distance = std::abs(radius * std::cos(cone->SemiAngle) -
                        height * std::sin(cone->SemiAngle));
  } else if (target.SurfaceType == PatchSurfaceType::Torus) {
    const auto *torus = std::get_if<TorusParameters>(&target.Parameters);
    if (!torus || torus->MinorRadius <= 0 ||
        torus->MajorRadius <= torus->MinorRadius)
      return false;
    const Vec3 axis = Normalize(ToVec(torus->Axis.Direction));
    const Vec3 delta = Sub(point, ToVec(torus->Axis.Origin));
    const double height = Dot(delta, axis);
    const Vec3 radial = Sub(delta, Mul(axis, height));
    const double radius = Norm(radial);
    if (radius <= 1e-30)
      return false;
    const Vec3 tube = Sub(delta, Mul(radial, torus->MajorRadius / radius));
    const double length = Norm(tube);
    if (length <= 1e-30)
      return false;
    normal = Mul(tube, 1 / length);
    distance = std::abs(length - torus->MinorRadius);
  } else {
    // Freeform has no fitted surface equation. Its diagnostic errors cannot
    // certify that another triangle belongs to the same geometric surface.
    return false;
  }
  return std::isfinite(distance) && std::isfinite(Norm(normal));
}

inline bool AccumulateTriangle(const MeshTopology &mesh, int triangleId,
                               const MeshPatch &target,
                               SurfaceCompatibility &result) {
  if (triangleId < 0 || triangleId >= int(mesh.getTriangles().size()))
    return false;
  const auto &triangle = mesh.getTriangles()[triangleId];
  const double tolerance =
      std::max(mesh.getResolution().FittingTolerance, 1e-30);
  double coordinateMagnitude = 0, longest = 0;
  for (int k = 0; k < 3; ++k) {
    Vec3 point = ToVec(mesh.getVertices()[triangle.VertexIds[k]].Position);
    Vec3 normal;
    double distance;
    if (!SurfaceSample(target, point, distance, normal))
      return false;
    result.MaxNormalizedDistance =
        std::max(result.MaxNormalizedDistance, distance / tolerance);
    coordinateMagnitude = std::max(coordinateMagnitude, Norm(point));
    longest = std::max(
        longest,
        Distance(mesh.getVertices()[triangle.VertexIds[k]].Position,
                 mesh.getVertices()[triangle.VertexIds[(k + 1) % 3]].Position));
  }
  // Very thin CAD triangles are not intrinsically unreliable. Discard a
  // normal only when the cross product is at the coordinate roundoff scale.
  const double areaRoundoff = 64 * std::numeric_limits<double>::epsilon() *
                              std::max(coordinateMagnitude, longest) * longest;
  if (2 * triangle.Area > areaRoundoff) {
    double distance;
    Vec3 normal;
    if (!SurfaceSample(target, ToVec(triangle.Centroid), distance, normal))
      return false;
    const double cosine =
        std::abs(Dot(Normalize(ToVec(triangle.Normal)), normal));
    result.MaxNormalError =
        std::max(result.MaxNormalError, std::acos(std::min(1.0, cosine)));
    ++result.ReliableNormalSamples;
  }
  return true;
}
} // namespace SegmentationGuardDetail

inline SurfaceCompatibility
EvaluateSurfaceCompatibility(const MeshTopology &mesh,
                             const std::vector<int> &triangleIds,
                             const MeshPatch &target) {
  SurfaceCompatibility result;
  if (triangleIds.empty())
    return result;
  for (int triangleId : triangleIds)
    if (!SegmentationGuardDetail::AccumulateTriangle(mesh, triangleId, target,
                                                     result))
      return result;
  result.Supported = true;
  return result;
}

inline SurfaceCompatibility
EvaluateSurfaceCompatibility(const MeshTopology &mesh, int triangleId,
                             const MeshPatch &target) {
  SurfaceCompatibility result;
  result.Supported = SegmentationGuardDetail::AccumulateTriangle(
      mesh, triangleId, target, result);
  return result;
}

inline bool IsSurfaceCompatible(const SurfaceCompatibility &result,
                                const MeshResolutionInfo &resolution,
                                double maxNormalizedDistance = 3.0) {
  return result.Supported && result.ReliableNormalSamples > 0 &&
         result.MaxNormalizedDistance <= maxNormalizedDistance &&
         result.MaxNormalError <=
             std::max(12 * resolution.AngularTolerance, .12);
}

// Adjudicate a raw segmentation cue before constraints are locked. A small
// planar facet may simply be the last tessellation strip of a cylinder. This
// does not grant permission to remove an already declared feature boundary.
inline bool IsCylinderTessellationContinuation(
    const MeshTopology &mesh, const MeshPatch &source, const MeshPatch &target,
    const std::vector<int> &sharedEdgeIds, const SegmentationConfig &config) {
  if (source.SurfaceType != PatchSurfaceType::Plane ||
      source.TriangleIds.empty() ||
      source.TriangleIds.size() >
          size_t(std::max(0, config.MinimumPlanarConsolidationTriangles)) ||
      target.SurfaceType != PatchSurfaceType::Cylinder ||
      target.Confidence < .8 ||
      target.TriangleIds.size() < 4 * source.TriangleIds.size() ||
      sharedEdgeIds.empty())
    return false;
  const auto *cylinder = std::get_if<CylinderParameters>(&target.Parameters);
  if (!cylinder)
    return false;
  const auto &resolution = mesh.getResolution();
  const auto sourceSupport =
      EvaluateSurfaceCompatibility(mesh, source.TriangleIds, target);
  const auto targetSupport =
      EvaluateSurfaceCompatibility(mesh, target.TriangleIds, target);
  if (!IsSurfaceCompatible(sourceSupport, resolution) ||
      !IsSurfaceCompatible(targetSupport, resolution))
    return false;
  const double distanceLimit = std::max(
      2 * targetSupport.MaxNormalizedDistance * resolution.FittingTolerance,
      resolution.BoundingBoxDiagonal * 1e-8);
  if (sourceSupport.MaxNormalizedDistance * resolution.FittingTolerance >
          distanceLimit ||
      sourceSupport.MaxNormalError >
          std::max(1.5 * targetSupport.MaxNormalError,
                   2 * resolution.AngularTolerance))
    return false;

  const Vec3 axis = Normalize(ToVec(cylinder->Axis.Direction));
  const Vec3 origin = ToVec(cylinder->Axis.Origin);
  auto member = [](const std::vector<int> &ids, int id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
  };
  auto radialDirection = [&](int vertexId) {
    Vec3 delta = Sub(ToVec(mesh.getVertices()[vertexId].Position), origin);
    return Normalize(Sub(delta, Mul(axis, Dot(delta, axis))));
  };
  auto angle = [&](const Vec3 &a, const Vec3 &b) {
    return std::acos(std::max(-1.0, std::min(1.0, Dot(a, b))));
  };
  std::vector<Vec3> sourceRadials;
  for (int triangleId : source.TriangleIds)
    for (int vertexId : mesh.getTriangles()[triangleId].VertexIds)
      sourceRadials.push_back(radialDirection(vertexId));
  double sourceSpan = 0;
  for (size_t i = 0; i < sourceRadials.size(); ++i)
    for (size_t j = i + 1; j < sourceRadials.size(); ++j)
      sourceSpan =
          std::max(sourceSpan, angle(sourceRadials[i], sourceRadials[j]));
  if (sourceSpan <= 1e-8)
    return false;

  for (int edgeId : sharedEdgeIds) {
    if (edgeId < 0 || edgeId >= int(mesh.getEdges().size()))
      return false;
    const auto &edge = mesh.getEdges()[edgeId];
    if (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature ||
        edge.IncidentTriangleIds.size() != 2)
      return false;
    int targetTriangle = -1;
    bool touchesSource = false;
    for (int triangleId : edge.IncidentTriangleIds) {
      touchesSource = touchesSource || member(source.TriangleIds, triangleId);
      if (member(target.TriangleIds, triangleId))
        targetTriangle = triangleId;
    }
    if (!touchesSource || targetTriangle < 0)
      return false;
    Vec3 edgeDirection =
        Normalize(Sub(ToVec(mesh.getVertices()[edge.Vertex1].Position),
                      ToVec(mesh.getVertices()[edge.Vertex0].Position)));
    if (std::abs(Dot(edgeDirection, axis)) <
        std::cos(2 * resolution.AngularTolerance))
      return false;
    const auto &triangle = mesh.getTriangles()[targetTriangle];
    double neighborSpan = 0;
    for (int k = 0; k < 3; ++k)
      neighborSpan =
          std::max(neighborSpan,
                   angle(radialDirection(triangle.VertexIds[k]),
                         radialDirection(triangle.VertexIds[(k + 1) % 3])));
    if (neighborSpan <= 1e-8 || sourceSpan > 1.5 * neighborSpan + 1e-8 ||
        neighborSpan > 1.5 * sourceSpan + 1e-8)
      return false;
  }
  return true;
}

} // namespace CadMesh
