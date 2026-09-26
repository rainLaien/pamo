#pragma once
#include "cad_adaptive/IRemeshBackend.h"
namespace cad_adaptive {
// Raw CUDA path. Faces retain CAD patch ownership while the CPU rebuilds
// adjacency; geometric decisions and mesh mutation passes execute on CUDA.
struct RawCudaOptions {
  bool independentStream=false;
  bool quiet=false;
  std::string *error=nullptr;
  bool freezeBoundary=false;
  bool stopWhenIdle=false; // end a batch region after a cycle with no mutations
  int smoothPasses=12;
  int smoothAttempts=3;
  int collapsePasses=8, flipPasses=8;
  bool strictFlipQuality=true;
  // Permit a stronger quality tradeoff during edge collapse while still
  // rejecting inverted/zero-area faces and invalid topology.
  bool allowQualityTradeoff=false;
  bool optimizeSplitPoint=false;
  float splitQualityRatio=0.f;
  float qualityMeanFloor=0,qualityP05Floor=0;
  size_t workspaceBytes=0; // 0: unlimited; includes cached device buffers
};
bool remeshRawCuda(SemanticMesh&,const RemeshConfig&,RemeshReport&,const RawCudaOptions&);
bool remeshRawCuda(SemanticMesh&,const RemeshConfig&,RemeshReport&);
}
