#pragma once
#include "cad_adaptive/RawCudaRemesher.h"
#include "cad_adaptive/RegionQuality.h"
#include "cad_adaptive/RegionCandidateSelection.h"
#include <cstdint>
#include <vector>
namespace cad_adaptive {
struct RawBatchOptions {
  int workers=2;
  int gpuConcurrency=2; // single-GPU active task cap; partitions may exceed this
  bool stopWhenIdle=true;
  bool autoPartitionSinglePatch=false;
  int patchesPerTask=0; // 0: auto; pack large CAD jobs without merging ownership
  int smoothPasses=12;
  int collapsePasses=8, flipPasses=8;
  bool strictFlipQuality=true;
  bool requireQualityImprovement=false;
  bool legacyCoverageAcceptance=false; // Explicit baseline comparison only.
  bool globalQualityAcceptance=false; // Whole-mesh Pareto endpoint; patch losses remain reported.
  bool exploreProvisionalChildren=false; // Optional bounded candidate comparison.
  bool trackRegionCandidates=false;
  bool sizeFeasibleFinalRefine=false;
  bool selectFinalRegions=false;
  float lowQualityThreshold=0; // 0: derive once from immutable batch input P05.
  size_t memoryBytes=0; // automatic: 60% of currently free device memory
};
struct RawPatchResult {
  bool accepted=false; // Valid for assembly. This is not endpoint quality success.
  bool qualityAccepted=false;
  bool provisional=false; // Valid candidate retained for recovery; not quality success.
  bool unchanged=false; // Valid source geometry retained after quality-gated attempts.
  bool uniformSizing=false; // Curvature sizing failed; uniform interior sizing succeeded.
  bool qualitySplit=false; // Alternate interior edge point and split quality gate succeeded.
  float qualitySplitRatio=0;
  bool meanRecovery=false; // Flip/smooth-only retry improved mean quality with a lower-tail guard.
  bool sizeRecovery=false; // Size-targeted retry reduced overlong edges under bounded quality guards.
  bool retried=false;
  // Source patch IDs whose accepted output is exactly the retained input.
  // A task may contain several patches and only some may be unchanged.
  std::vector<uint32_t> unchangedPatchIds;
  size_t workspaceBytes=0;
  int inputFaces=0,outputFaces=0;
  float inputQualityMean=0,inputQualityP05=0;
  double seconds=0;
  double secondsInitial=0,secondsGentle=0;
  double secondsChildComparison=0;
  bool childCandidateCompared=false,childCandidateSelected=false;
  std::string childComparisonReason;
  bool childCandidateEvaluated=false;
  int childSelectedPatches=0;
  int childGeneratedPatches=0;
  int childBoundaryTargetChanges=0;
  RegionQuality childIncumbentQuality,childCandidateQuality;
  double secondsUniformGentle=0,secondsUniformStrict=0,secondsQualitySplit=0,secondsMeanRecovery=0;
  std::string error;
  std::string retryReason;
  RemeshReport report;
};
// Anchor IDs identify shared source vertices, not coordinate-welded points.
// A local candidate cannot silently replace the sizing contract at a seam.
int boundaryTargetMismatchCount(const SemanticMesh& before,const std::vector<int>& beforeAnchors,
                                const SemanticMesh& after,const std::vector<int>& afterAnchors);
struct RawBatchReport {
  std::vector<RawPatchResult> patches;
  int patchesPerTask=1;
  int accepted=0,unchanged=0,uniformRegions=0,qualitySplitRegions=0,meanRecoveryRegions=0;
  int fallback=0,retried=0,peakActive=0,boundarySplits=0,boundaryEdgesDeferred=0;
  int finalSizeRefineSplits=0,finalSizeRefineLevels=0,sizeRecoveryRegions=0,longEdgesAfterRefine=0;
  int circularPlaneRemeshed=0,circularPlaneSkipped=0,circularPlaneAddedFaces=0;
  int finalSizeUnselectedLongEdges=0;
  int finalSizeRejectedSplits=0;
  double finalSizeExcessBefore=0,finalSizeExcessAfter=0;
  std::string finalSizeRefineStopReason="not_needed";
  RegionCandidateSelection finalRegionSelection;
  double secondsFinalAlternativeRefine=0;
  float maxOutputEdgeRatio=0;
  size_t memoryBudget=0,peakReserved=0;
  double seconds=0;
  double secondsBoundary=0,secondsTasks=0,secondsAssembly=0;
  int seamRepairs=0;
  int seamSplitRepairs=0;
  bool topologyValid=false,boundariesHeld=false;
  RegionQuality sourceQuality,outputQuality;
  std::vector<RegionQuality> sourcePatchQuality,outputPatchQuality;
  std::vector<uint32_t> regressedPatchIds;
  std::vector<uint32_t> pendingPatchIds,unresolvedPatchIds;
  double secondsQualityAudit=0;
  bool qualityAccepted=false;
};
// Source remains immutable. Hard failures retain source geometry. Quality-only
// failures retain a valid provisional candidate and are reported by the final
// regional guard; the caller must distinguish these from accepted quality.
bool remeshRawCudaPatches(const SemanticMesh&,SemanticMesh&,const RemeshConfig&,
                         const RawBatchOptions&,RawBatchReport&,std::string* error=nullptr);
}
