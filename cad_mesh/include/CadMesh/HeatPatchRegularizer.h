#pragma once

namespace CadMesh {
class CadMeshPatchSegmenter;

struct HeatRegularizationReport {
  int AmbiguousComponents = 0;
  int SolvedComponents = 0;
  int SkippedOversizedComponents = 0;
  int FailedSolves = 0;
  int DemotedGeometricSeeds = 0;
  int PrunedThermalSeeds = 0;
  int ResolvePasses = 0;
  int ReassignedTriangles = 0;
};

class HeatPatchRegularizer {
public:
  static HeatRegularizationReport regularize(CadMeshPatchSegmenter &segmenter);
};
} // namespace CadMesh
