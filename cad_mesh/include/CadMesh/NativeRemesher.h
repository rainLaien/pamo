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
  // Secondary Freeform partition only. Zero distance selects 0.0003 * target
  // length, capped by MaximumDeviation; remesh acceptance is unchanged.
  double SecondaryPlaneDistanceTolerance = 0;
  double SecondaryPlaneNormalToleranceDegrees = 1;
  double SecondaryPlaneMinimumAreaRatio = .25; // area / target length squared
  double GenericFeatureAngleDegrees = 10;
  int GenericRemeshIterations = 5;
  int GenericRemeshWorkers = 20;
  // Internal local-operation policy; configured by the isotropic driver.
  double CollapseLengthRatio = .65;
  bool IsotropicOperationRules = false;
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

struct RemeshCollisionWitness {
  int Attempt=0;
  std::array<int,2> PatchIds{},FaceIds{},Rebuilt{};
  std::array<std::array<int,3>,2> VertexIds{};
  std::array<std::array<Point3,3>,2> Points;
  std::array<std::array<Point3,3>,2> SourcePoints;
  std::array<int,2> SourceFaceIds{{-1,-1}};
  std::array<double,2> SourceDistances{};
  std::array<unsigned char,2> UnchangedFromSource{};
  int SourcePairContact=-1; // nearest-centroid source pair only, not proof of inheritance
};

struct NativeRemeshResult {
  std::vector<Point3> Vertices;
  std::vector<std::array<int, 3>> Triangles;
  std::vector<int> PatchIds;
  // Flat output registry; indices match PatchIds. Source IDs refer to input partition.
  std::vector<MeshPatch> OutputPatches;
  std::vector<int> SourcePatchIds;
  // Legacy patch summary; FaceRemeshed is authoritative for mixed subregions.
  // Boundary splitting alone does not set this flag.
  std::vector<unsigned char> RemeshedPatches;
  // Selection is independent of construction eligibility and final rollback.
  std::vector<unsigned char> RequestedPatches;
  // Per output triangle: 0 unselected, 1 failed, 2 rebuilt, 3 preserved good input.
  std::vector<unsigned char> FaceRemeshed;
  std::vector<PatchRemeshReason> FaceReasons;
  std::vector<PatchRemeshReason> PatchReasons;
  // Copies of pre-rollback geometry; IDs refer to that collision attempt.
  std::vector<RemeshCollisionWitness> CollisionWitnesses;
  NativeRemeshStatistics Statistics;
};

class NativeRemesher {
public:
  // Samples shared boundaries, rebuilds analytic and Freeform subregions, then
  // compacts the indexed mesh. No post-remesh deviation/collision rollback;
  // failed construction regions retain input while successful regions survive.
  static bool remesh(const CadMeshPatchSegmenter &, const NativeRemeshConfig &,
                     NativeRemeshResult &, std::string &error);
  static bool writePly(const NativeRemeshResult &,
                       const std::vector<MeshPatch> &,
                       const std::filesystem::path &, std::string &error);
};

} // namespace CadMesh
