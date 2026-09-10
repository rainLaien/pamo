#pragma once

#include "CadMesh/Types.h"
#include <array>
#include <cstddef>
#include <vector>

namespace CadMesh {

struct AnalyticPatchRemeshReport {
  std::vector<PatchRemeshReason> PatchReasons;
  std::size_t Attempted = 0;
  std::size_t Rebuilt = 0;
  std::size_t Fallback = 0;
  std::size_t RebuiltFaces = 0;
  std::size_t DeviationFallback = 0;
  std::size_t TopologyFallback = 0;
  std::size_t ParameterizationFallback = 0;
  std::size_t QualityFallback = 0;
  std::size_t CollisionFallback = 0;
  double IndexSeconds = 0;
  double ChartSeconds = 0;
  double CommitSeconds = 0;
};

// Rebuild supported analytic patches as complete parameter-domain charts.
// Existing boundary vertex ids remain unchanged so adjacent patches continue
// to share exactly the same feature and transition curves.
bool RebuildAnalyticPatches(
    std::vector<Point3> &vertices,
    std::vector<std::array<int, 3>> &faces,
    std::vector<int> &labels,
    const std::vector<MeshPatch> &patches,
    double targetEdgeLength,
    double maximumDeviation,
    double maximumNormalDeviationDegrees,
    double targetMeanQuality,
    const std::vector<unsigned char> &excludedPatches,
    std::vector<unsigned char> &successfullyRebuilt,
    AnalyticPatchRemeshReport &report, bool verbose = false);

} // namespace CadMesh
