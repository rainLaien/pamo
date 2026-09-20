#pragma once
#include "cad_adaptive/IRemeshBackend.h"
namespace cad_adaptive {
// Raw single-surface CUDA path. CPU owns adjacency upload/compaction; all
// geometric decisions and triangle/vertex mutation passes execute on CUDA.
bool remeshRawCuda(SemanticMesh&,const RemeshConfig&,RemeshReport&);
}
