#pragma once
#include "cad_adaptive/SemanticMesh.h"
#include <ostream>

namespace cad_adaptive {
// Orthogonal diagnostic roles. Partition interfaces are NOT asserted to be
// geometric features or computation-only seams: CADPART1 has a producer hard
// bit, but lacks curve certification and model-versus-compute interface origin.
enum ConstraintRole : uint8_t { PersistentFeature=1, OpenBoundary=2, PartitionInterface=4 };
enum class ConstraintMotion : uint8_t { Surface, Curve, Fixed };
struct EdgeConstraintClassification {
  uint8_t roles=0; // Roles may overlap, unlike the current vertex movement class.
  ConstraintMotion allowedMotion=ConstraintMotion::Surface;
  bool geometryProvenanceKnown=false;
};
EdgeConstraintClassification classifyConstraint(const SemanticMesh&,const EdgeRec&);
struct ConstraintSummary {
  int fixedVertices=0, curveVertices=0, surfaceVertices=0;
  int persistentFeatureEdges=0, openBoundaryEdges=0, partitionInterfaceEdges=0;
};
struct DefectRegion {
  int faces=0;
  double area=0;
  float minQuality=1;
  Vec3 lower{},upper{};
  std::vector<uint32_t> patches;
};
struct RegionQuality {
  float threshold=0, mean=0, p05=0, minimum=0;
  double area=0, areaWeightedMean=0, lowQualityArea=0, largestLowQualityArea=0;
  int lowQualityFaces=0, lowQualityComponents=0;
  int longEdges=0, shortEdges=0, shortConstrainedEdges=0;
  int longEdgeFaceComponents=0,shortEdgeFaceComponents=0;
  double largestLongEdgeFaceArea=0,largestShortEdgeFaceArea=0;
  float maxTargetTransitionRatio=1;
  float edgeRatioMin=0, edgeRatioP05=0, edgeRatioP95=0, edgeRatioMax=0;
  int zeroFaces=0, inconsistentEdges=0;
  ConstraintSummary constraints;
  // All components contribute to totals. JSON contains the largest 32 only.
  std::vector<DefectRegion> regions;
};
RegionQuality evaluateRegionQuality(const SemanticMesh&, const RemeshConfig&, float threshold);
std::vector<RegionQuality> evaluatePatchQuality(const SemanticMesh&,const RemeshConfig&,float threshold);
// Actual assembled targets are authoritative; task reports cannot describe
// the minimum target shared by vertices belonging to different tasks.
std::vector<uint32_t> longEdgePatches(const SemanticMesh&,const RemeshConfig&);
struct SizeDefectSummary {int count=0;float maxRatio=0;double excessSquared=0;};
// Intermediate size progress only. This does not certify endpoint quality.
bool sizeDefectProgress(const SizeDefectSummary&,const SizeDefectSummary&,bool allowSeverityProgress);
// Conservative endpoint guard. No universal absolute quality target is implied.
enum QualityFailure : uint32_t {
  MeanRegression=1, P05Regression=2, AreaMeanRegression=4,
  LowAreaRegression=8, LargestRegionRegression=16, InvalidFaces=32, WindingRegression=64,
  UnresolvedInputDefect=128
};
uint32_t qualityRegressionMask(const RegionQuality& before,const RegionQuality& after);
bool qualityNonRegression(const RegionQuality& before,const RegionQuality& after);
bool qualityProgress(const RegionQuality& before,const RegionQuality& after);
// An already poor region also needs measurable progress; merely keeping it
// unchanged is not endpoint success. Healthy unchanged regions remain eligible.
uint32_t qualityEndpointMask(const RegionQuality& before,const RegionQuality& after);
// Normal endpoint keeps the whole-mesh Pareto guard while allowing reported
// patch-level tradeoffs. Strict endpoint also requires no pending patches.
bool batchQualityEndpointAccepted(const RegionQuality& before,const RegionQuality& after,
                                  size_t pendingPatches,bool globalQualityAcceptance);
void writeRegionQualityJson(std::ostream&,const RegionQuality&);
}
