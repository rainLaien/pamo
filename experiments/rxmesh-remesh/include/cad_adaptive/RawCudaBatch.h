#pragma once
#include "cad_adaptive/RawCudaRemesher.h"
namespace cad_adaptive {
struct RawBatchOptions {
  int workers=2;
  size_t memoryBytes=0; // automatic: 60% of currently free device memory
};
struct RawPatchResult {
  bool accepted=false;
  size_t workspaceBytes=0;
  int inputFaces=0,outputFaces=0;
  double seconds=0;
  std::string error;
  RemeshReport report;
};
struct RawBatchReport {
  std::vector<RawPatchResult> patches;
  int accepted=0,fallback=0,peakActive=0,boundarySplits=0;
  size_t memoryBudget=0,peakReserved=0;
  double seconds=0;
  bool topologyValid=false,boundariesHeld=false;
};
// Source remains immutable. Failed tasks retain their source region; caller
// must distinguish this partial result from all-regions-remeshed success.
bool remeshRawCudaPatches(const SemanticMesh&,SemanticMesh&,const RemeshConfig&,
                         const RawBatchOptions&,RawBatchReport&,std::string* error=nullptr);
}
