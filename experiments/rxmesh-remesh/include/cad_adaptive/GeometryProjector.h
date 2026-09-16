#pragma once

#include "cad_adaptive/SemanticMesh.h"

namespace cad_adaptive {

struct ProjectionResult {
  bool ok = false;
  Vec3 position{};
};

class GeometryProjector {
public:
  void build(const SemanticMesh &mesh);

  ProjectionResult projectSurface(uint32_t patchId, Vec3 p) const;
  ProjectionResult projectFeature(uint32_t featureCurveId, Vec3 p) const;
  ProjectionResult projectVertex(const SemanticMesh &mesh, int v, Vec3 p) const;

  Vec3 analyticNormal(uint32_t patchId, Vec3 p) const;
  float analyticKmax(uint32_t patchId) const;

  const PatchRecord *patch(uint32_t id) const {
    return id < patches_.size() ? &patches_[id] : nullptr;
  }

  static ProjectionResult projectPlane(Vec3 origin, Vec3 normal, Vec3 p);
  static ProjectionResult projectCylinder(Vec3 origin, Vec3 axis, float radius, Vec3 p);

private:
  std::vector<PatchRecord> patches_;
  struct Segment {
    uint32_t feature = 0;
    Vec3 a{}, b{};
  };
  struct ReferenceTriangle {
    uint32_t patchId = 0;
    Vec3 a{}, b{}, c{};
  };
  std::vector<Segment> segments_;
  std::vector<ReferenceTriangle> referenceTriangles_;
};

} // namespace cad_adaptive
