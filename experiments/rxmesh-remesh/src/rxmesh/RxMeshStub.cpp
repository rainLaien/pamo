#include "cad_adaptive/RxMeshBackend.h"

namespace cad_adaptive {

bool rxmeshAvailable() { return false; }

bool RxMeshBackend::remesh(SemanticMesh &, const RemeshConfig &, RemeshReport &report) {
  report = {};
  return false;
}

bool RxMeshBackend::roundTrip(const SemanticMesh &, SemanticMesh &, std::vector<int> *,
                              std::string *error) {
  if (error) *error = "RXMesh backend was not compiled (CAD_ADAPTIVE_RXMESH=OFF)";
  return false;
}

} // namespace cad_adaptive
