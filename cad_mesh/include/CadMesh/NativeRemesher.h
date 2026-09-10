#pragma once

#include "CadMesh/CadMeshPatchSegmenter.h"
#include <filesystem>
#include <string>
#include <vector>

namespace CadMesh {

struct NativeRemeshConfig {
  double TargetEdgeLength = 0;
  double MaximumDeviation = 0;
  double MaximumNormalDeviationDegrees = 10;
  double TargetMeanTriangleQuality = .8;
  int SplitPasses = 128;
  int CollapsePasses = 12;
  int FlipPasses = 8;
  int RelaxIterations = 3;
  bool RequireCuda = false;
  bool Verbose = true;
};

struct NativeRemeshStatistics {
  std::size_t InputVertices = 0, InputTriangles = 0;
  std::size_t OutputVertices = 0, OutputTriangles = 0;
  std::size_t Splits = 0, Collapses = 0, Flips = 0;
  std::size_t CollisionRejections = 0;
  std::size_t AnalyticPatchesAttempted = 0;
  std::size_t AnalyticPatchesRebuilt = 0;
  std::size_t AnalyticPatchesFallback = 0;
  int AcceptedRelaxations = 0;
  double MaximumEdgeLength = 0;
  double MeanTriangleQuality = 0;
  double MinimumTriangleQuality = 0;
  double Percentile05TriangleQuality = 0;
  double FractionBelow02TriangleQuality = 0;
  double MinimumAngleDegrees = 0;
  bool QualityTargetMet = false;
  bool QualityMeasured = false;
  bool SelfIntersectionFree = false;
  bool UsedCuda = false;
};

struct NativeRemeshResult {
  std::vector<Point3> Vertices;
  std::vector<std::array<int, 3>> Triangles;
  std::vector<int> PatchIds;
  // Indexed by patch ID; set only for accepted analytic reconstruction after
  // collision rollback. Boundary splitting alone does not set this flag.
  std::vector<unsigned char> RemeshedPatches;
  std::vector<PatchRemeshReason> PatchReasons;
  NativeRemeshStatistics Statistics;
};

class NativeRemesher {
public:
  // Samples shared boundaries, rebuilds supported patches, validates and rolls
  // back collisions, then compacts the indexed mesh. Failed/unsupported patches
  // retain their interiors; no global split/collapse/flip/relax follows.
  static bool remesh(const CadMeshPatchSegmenter &, const NativeRemeshConfig &,
                     NativeRemeshResult &, std::string &error);
  static bool writePly(const NativeRemeshResult &,
                       const std::vector<MeshPatch> &,
                       const std::filesystem::path &, std::string &error);
};

} // namespace CadMesh
