#pragma once
#include "cad_adaptive/RawCudaRemesher.h"
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
  size_t memoryBytes=0; // automatic: 60% of currently free device memory
};
struct RawPatchResult {
  bool accepted=false;
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
  double secondsUniformGentle=0,secondsUniformStrict=0,secondsQualitySplit=0,secondsMeanRecovery=0;
  std::string error;
  std::string retryReason;
  RemeshReport report;
};
struct RawBatchReport {
  std::vector<RawPatchResult> patches;
  int patchesPerTask=1;
  int accepted=0,unchanged=0,uniformRegions=0,qualitySplitRegions=0,meanRecoveryRegions=0;
  int fallback=0,retried=0,peakActive=0,boundarySplits=0;
  int finalSizeRefineSplits=0,finalSizeRefineLevels=0,sizeRecoveryRegions=0,longEdgesAfterRefine=0;
  float maxOutputEdgeRatio=0;
  size_t memoryBudget=0,peakReserved=0;
  double seconds=0;
  double secondsBoundary=0,secondsTasks=0,secondsAssembly=0;
  int seamRepairs=0;
  int seamSplitRepairs=0;
  bool topologyValid=false,boundariesHeld=false;
};
// Source remains immutable. Failed tasks retain their source region; caller
// must distinguish this partial result from all-regions-remeshed success.
bool remeshRawCudaPatches(const SemanticMesh&,SemanticMesh&,const RemeshConfig&,
                         const RawBatchOptions&,RawBatchReport&,std::string* error=nullptr);
}
