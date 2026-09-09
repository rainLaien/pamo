#pragma once
#include "CadMesh/BoundaryScoreCalculator.h"
namespace CadMesh {
class PatchRefiner;
class PatchGraphBuilder;
class HeatPatchRegularizer;
class CadMeshPatchSegmenter {
public:
  explicit CadMeshPatchSegmenter(SegmentationConfig config = {})
      : mConfig(config) {}
  bool segment(const TriangleSoup &soup);
  // Check complete, unique ownership and manifold-edge connectivity.
  bool validatePartition(std::string *error = nullptr) const;
  const MeshTopology &getMesh() const { return mMesh; }
  MeshTopology &getMesh() { return mMesh; }
  const std::vector<MeshPatch> &getPatches() const { return mPatches; }
  const std::vector<PatchAdjacency> &getAdjacency() const { return mAdjacency; }
  const RemeshConstraint &getRemeshConstraint() const { return mConstraint; }
  bool hasComputedDifferentialGeometry() const { return mComputedDiagnostics; }
  bool hasComputedBoundaryScores() const { return mComputedDiagnostics; }
  bool usesModelFirstPartitioning() const { return mConfig.EnableModelFirst; }

private:
  void GenerateInitialRegions();
  void GrowSurfaceConsistentRegions();
  void FitPatches();
  void AssignPatchIds();
  void RebuildConnectedPatches();
  SegmentationConfig mConfig;
  MeshTopology mMesh;
  std::vector<MeshPatch> mPatches;
  std::vector<PatchAdjacency> mAdjacency;
  RemeshConstraint mConstraint;
  bool mComputedDiagnostics = false;
  friend class PatchRefiner;
  friend class PatchGraphBuilder;
  friend class HeatPatchRegularizer;
};
class PatchRefiner {
public:
  static void refine(CadMeshPatchSegmenter &);
};
class PatchGraphBuilder {
public:
  static void build(CadMeshPatchSegmenter &);
};
} // namespace CadMesh
