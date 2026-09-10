#pragma once

#include "CadMesh/GeometryTypes.h"
#include <array>
#include <string>
#include <variant>
#include <vector>

namespace CadMesh {
enum class PatchSurfaceType {
  Unknown,
  Plane,
  Cylinder,
  Cone,
  Sphere,
  Torus,
  Freeform
};
enum class BoundaryConfidence { Weak, Probable, Certain };
enum class PatchFeatureRole { Ordinary, Fillet };
enum class AnalyticSeedBackend { Auto, Cpu, Cuda };

struct MeshResolutionInfo {
  double BoundingBoxDiagonal = 0, MinEdgeLength = 0, MedianEdgeLength = 0,
         MeanEdgeLength = 0;
  double WeldTolerance = 0, FittingTolerance = 0, CurvatureTolerance = 0,
         AngularTolerance = 0;
};
struct DifferentialGeometry {
  double K1 = 0, K2 = 0, MeanCurvature = 0, GaussianCurvature = 0;
  Direction3 PrincipalDirection1{1, 0, 0}, PrincipalDirection2{0, 1, 0};
  double Confidence = 0;
};
struct BoundaryEvidence {
  double NormalDiscontinuity = 0, CurvatureDiscontinuity = 0,
         CurvatureGradient = 0;
  double SurfaceFitDiscontinuity = 0, TessellationEvidence = 0, FinalScore = 0;
};
struct PlaneParameters {
  PlaneEquation Plane;
};
struct CylinderParameters {
  AxisLine Axis;
  double Radius = 0;
};
struct ConeParameters {
  AxisLine Axis;
  double SemiAngle = 0;
};
struct SphereParameters {
  Point3 Center;
  double Radius = 0;
};
struct TorusParameters {
  AxisLine Axis;
  double MajorRadius = 0, MinorRadius = 0;
};
using SurfaceParameters =
    std::variant<std::monostate, PlaneParameters, CylinderParameters,
                 ConeParameters, SphereParameters, TorusParameters>;

enum class BoundaryDirectionStatus {
  Consistent,
  NonManifold,
  InconsistentWinding,
  MultiplePatchSides,
  MissingIncidence
};
enum class PatchProjectionTarget { AnalyticSurface, ReferenceMesh };

struct DirectedBoundaryChain {
  int ChainId = -1;
  // +1 follows the shared vertex order, -1 reverses it, 0 is unresolved.
  int Direction = 0;
  BoundaryDirectionStatus Status = BoundaryDirectionStatus::MissingIncidence;
};
struct BoundaryChain {
  int Id = -1;
  // EdgeIds[i] joins VertexIds[i] to VertexIds[i + 1]. A closed chain
  // repeats its first vertex at the end; a shared edge occurs in one chain.
  std::vector<int> EdgeIds, VertexIds, IncidentPatchIds;
  // These original mesh vertices are the one shared initial sampling of the
  // chain, not a fitted CAD curve or newly generated remesh samples.
  std::vector<int> InitialSampleVertexIds;
  bool IsClosed = false, IsNonManifold = false;
  bool HasInconsistentWinding = false;
  bool IsHardFeature = false;
};
struct BoundaryCorner {
  int VertexId = -1;
  bool IsEndpoint = false, IsJunction = false, IsSharpCorner = false;
  bool HasIncidenceChange = false;
};

struct MeshPatch {
  int Id = -1;
  PatchSurfaceType SurfaceType = PatchSurfaceType::Unknown;
  PatchFeatureRole FeatureRole = PatchFeatureRole::Ordinary;
  std::vector<int> SupportPatchIds;
  std::vector<int> TriangleIds, BoundaryEdgeIds, NeighborPatchIds;
  double RmsFittingError = 0, MaxFittingError = 0, NormalError = 0,
         Confidence = 0;
  // Additional tessellation discrepancy at triangle vertices, edge midpoints
  // and centroids. This sampled value is not a certified Hausdorff bound.
  double MaxSampledSurfaceDeviation = 0;
  SurfaceParameters Parameters;
  std::vector<DirectedBoundaryChain> BoundaryChainRefs;
  PatchProjectionTarget ProjectionTarget = PatchProjectionTarget::ReferenceMesh;
  bool HasConsistentFaceOrientation = true;
  // Periodic analytic surfaces need chart/seam assessment in the remesher.
  // Segmentation does not generate a parameterization seam.
  bool NeedsParameterizationSeamAssessment = false;
};
struct PatchAdjacency {
  int Patch0 = -1, Patch1 = -1;
  std::vector<int> SharedBoundaryEdges;
  double BoundaryConfidence = 0;
};
struct RemeshConstraint {
  // ConstraintEdgeIds is the shared sampling union, not a list of sharp edges.
  std::vector<int> HardFeatureEdgeIds, SurfaceTransitionEdgeIds;
  std::vector<int> ConstraintEdgeIds, JunctionVertexIds;
  std::vector<int> CornerVertexIds;
  std::vector<BoundaryCorner> Corners;
  std::vector<BoundaryChain> BoundaryChains;
  double CornerAngleThreshold = 0;
};
struct SegmentationConfig {
  bool EnableModelFirst = true;
  AnalyticSeedBackend ModelAnalyticSeedBackend = AnalyticSeedBackend::Auto;
  double ModelFitToleranceRatio = 3.0;
  double ModelNormalTolerance = 0.14;
  double ModelSeedRadiusFactor = 8.0;
  int ModelMaximumSeeds = 2500;
  // Separate bounded reserve for spatially distributed unexplored residuals.
  int ModelResidualSeedBudget = 1250;
  int ModelMaximumMergePasses = 4;
  // Only bridge narrow residual corridors with opposing support from one
  // certified cylinder/torus (possibly represented by several patches).
  bool EnableModelSeamBridging = true;
  int ModelSeamMaximumFaces = 4096;
  int ModelSeamEvaluationBudget = 250000;
  double ModelSeamMaximumWidthRadiusRatio = .4;
  // Only verified dihedral creases, topology and explicit features stop growth.
  double ModelSharpAngle = 0.65;
  double StrongBoundaryThreshold = 0.72, WeakBoundaryThreshold = 0.42;
  double NormalWeight = 0.22, CurvatureWeight = 0.23, GradientWeight = 0.16;
  double SurfaceFitWeight = 0.33, TessellationWeight = 0.06;
  int CurvatureRingCount = 2, MinimumPatchTriangles = 2, RefitBatchSize = 16;
  int MinimumPlanarConsolidationTriangles = 12;
  int MaximumExhaustiveRefinementPatches = 1000;
  int MaximumExhaustiveRefinementTriangles = 200000;
  int HeatMaximumAmbiguousPatchTriangles = 500;
  int HeatMaximumComponentTriangles = 12000;
  int HeatMaximumLabels = 8;
  int HeatMaximumResolvePasses = 4;
  int HeatMinimumSeedRegionTriangles = 6;
  int MaximumFaceClosurePasses = 64;
  int ClosureWeakStripMaximumTriangles = 500;
  int InternalSplitMaximumPasses = 2;
  int InternalSplitMinimumTriangles = 12;
  int InternalSplitMinimumCutEdges = 3;
  double ModelComplexityPenalty = 0.025, MergeErrorFactor = 1.75;
  double HeatSeedConfidence = 0.80;
  double HeatBoundaryBeta = 8.0;
  double HeatHardBoundaryScore = 0.95;
  double HeatAssignmentConfidence = 0.55;
  double HeatAssignmentMargin = 0.08;
  double HeatMinimumSeedRegionAreaRatio = 0.005;
  double HeatSeedDemotionBoundaryScore = 0.65;
  double HeatSeedDemotionTargetRatio = 8.0;
  double HeatSeedDemotionResidualTolerance = 3.0;
  double ClosureSkinnyTriangleQuality = 0.04;
  double ClosureAnalyticResidualTolerance = 3.0;
  double ClosureTargetSizeRatio = 8.0;
  double InternalSplitBoundaryThreshold = 0.55;
  double InternalSplitResidualRidgeThreshold = 0.58;
  double InternalSplitSupportingEvidence = 0.22;
  double InternalSplitMinimumChainLengthFactor = 3.0;
  double InternalSplitErrorFactor = 0.80;
  double InternalSplitAcceptedBoundaryScore = 0.98;
  // Turning angle of a constraint polyline, in radians; scale independent.
  double BoundaryCornerAngle = 0.6108652381980153; // 35 degrees
  bool EnableHeatRegularization = true;
  bool EnableInternalFeatureSplit = true;
  bool Verbose = false;
};
inline const char *SurfaceTypeName(PatchSurfaceType type) {
  switch (type) {
  case PatchSurfaceType::Plane:
    return "Plane";
  case PatchSurfaceType::Cylinder:
    return "Cylinder";
  case PatchSurfaceType::Cone:
    return "Cone";
  case PatchSurfaceType::Sphere:
    return "Sphere";
  case PatchSurfaceType::Torus:
    return "Torus";
  case PatchSurfaceType::Freeform:
    return "Freeform";
  default:
    return "Unknown";
  }
}
inline int SurfaceTypeId(PatchSurfaceType type) {
  return static_cast<int>(type);
}
inline const char *BoundaryDirectionStatusName(BoundaryDirectionStatus status) {
  switch (status) {
  case BoundaryDirectionStatus::Consistent:
    return "consistent";
  case BoundaryDirectionStatus::NonManifold:
    return "non_manifold";
  case BoundaryDirectionStatus::InconsistentWinding:
    return "inconsistent_winding";
  case BoundaryDirectionStatus::MultiplePatchSides:
    return "multiple_patch_sides";
  default:
    return "missing_incidence";
  }
}
} // namespace CadMesh
