#pragma once

#include "cad_adaptive/GeometryProjector.h"
#include "cad_adaptive/SemanticMesh.h"

namespace cad_adaptive {

struct SizingLimits {
  float hMin = 0;
  float hMax = 0;
  float epsilon = 0;
};

class RemeshField {
public:
  static SizingLimits limits(const SemanticMesh &mesh, const RemeshConfig &config);
  static float patchLength(const PatchRecord &patch, const SizingLimits &lim, float hConst);
  static float curvatureLength(float kmax, float epsilon, float hMax);
  static float featureLength(float distance, float band, float hFeatureEdge, float hRegular);
  static float combine(float hCurvature, float hFeature, float hPatch, float hError,
                       const SizingLimits &lim);

  static void computeFeatureDistance(SemanticMesh &mesh);
  static void compute(SemanticMesh &mesh, const RemeshConfig &config,
                      const GeometryProjector &projector);
};

} // namespace cad_adaptive
