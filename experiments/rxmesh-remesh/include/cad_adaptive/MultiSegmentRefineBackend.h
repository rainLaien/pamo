#pragma once
#include "cad_adaptive/GeometryProjector.h"
#include "cad_adaptive/SemanticMesh.h"

namespace cad_adaptive {

struct EdgeSubdivision {
  uint32_t V0 = 0;
  uint32_t V1 = 0;
  int SegmentCount = 1;
  // Ordered V0 -> V1, including both endpoints.
  std::vector<int> VertexIds;
};

struct LocalTriangulationAudit {
  int TriangleCount = 0;
  int ZeroAreaCount = 0;
  int FlippedCount = 0;
  int BoundaryMismatchCount = 0;
  int InteriorNonManifoldCount = 0;
  float QualityMin = 1.0f;
  float QualityMean = 0.0f;
};

struct MultiSegmentRefineStats {
  int InputFaces = 0;
  int InputEdges = 0;
  int SubdividedEdges = 0;
  int PlannedSegments = 0;
  int PlannedInsertedEdgeVertices = 0;
  int MaxSegments = 1;
  float MeanSegments = 1.0f;
  float MaxPlannedSegmentLength = 0.0f;
  int InvalidPlans = 0;
  int FacePlanCount = 0;
  int FacePlanMissingChains = 0;
  int FacePlanOrientationErrors = 0;
  int FacePlanOneLong = 0;
  int FacePlanTwoLong = 0;
  int FacePlanThreeLong = 0;
};

// Planning POC: computes one subdivision count per shared edge. It deliberately
// does not mutate topology; this lets us validate growth before triangulation.
bool planMultiSegmentRefine(const SemanticMesh& input, const RemeshConfig& config,
                            MultiSegmentRefineStats& stats,
                            std::string* error = nullptr);

// Pass 2 POC: materialize all globally planned shared-edge vertices without
// changing faces. The output is intentionally not a valid final mesh yet;
// callers use it only to audit shared-edge placement/projection.
bool materializeMultiSegmentEdges(const SemanticMesh& input, const RemeshConfig& config,
                                  const GeometryProjector& referenceProjector,
                                  SemanticMesh& output, MultiSegmentRefineStats& stats,
                                  std::vector<EdgeSubdivision>* subdivisions = nullptr,
                                  std::string* error = nullptr);

bool triangulateTwoLongChains(const SemanticMesh& vertices,
                              const std::array<int,3>& parent,
                              const std::vector<int>& chain0,
                              const std::vector<int>& chain1,
                              std::vector<std::array<int,3>>& triangles,
                              std::string* error = nullptr);

bool auditLocalTriangulation(const SemanticMesh& vertices,
                             const std::array<int,3>& parent,
                             const std::vector<std::array<int,3>>& triangles,
                             const std::vector<std::pair<int,int>>& boundarySegments,
                             LocalTriangulationAudit& audit,
                             std::string* error = nullptr);

} // namespace cad_adaptive
