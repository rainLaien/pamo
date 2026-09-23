#pragma once
#include "cad_adaptive/IRemeshBackend.h"
namespace cad_adaptive {
// Raw single-surface CUDA path. CPU owns adjacency upload/compaction; all
// geometric decisions and triangle/vertex mutation passes execute on CUDA.
struct RawCudaOptions {
  bool independentStream=false;
  bool quiet=false;
  std::string *error=nullptr;
  bool freezeBoundary=false;
  size_t workspaceBytes=0; // 0: unlimited; includes cached device buffers
};
bool remeshRawCuda(SemanticMesh&,const RemeshConfig&,RemeshReport&,const RawCudaOptions&);
bool remeshRawCuda(SemanticMesh&,const RemeshConfig&,RemeshReport&);
}
