#pragma once
#include "cad_adaptive/RawCudaRemesher.h"
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
  bool retried=false;
  size_t workspaceBytes=0;
  int inputFaces=0,outputFaces=0;
  float inputQualityMean=0,inputQualityP05=0;
  double seconds=0;
  std::string error;
  std::string retryReason;
  RemeshReport report;
};
struct RawBatchReport {
  std::vector<RawPatchResult> patches;
  int patchesPerTask=1;
  int accepted=0,unchanged=0,uniformRegions=0,fallback=0,retried=0,peakActive=0,boundarySplits=0;
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
