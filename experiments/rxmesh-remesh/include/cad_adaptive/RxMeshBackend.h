#pragma once

#include "cad_adaptive/IRemeshBackend.h"
#include <string>
#include <vector>

namespace cad_adaptive {

bool rxmeshAvailable();

class RxMeshBackend final : public IRemeshBackend {
public:
  bool remesh(SemanticMesh &mesh, const RemeshConfig &config, RemeshReport &report) override;

  // 001.0: import SoA attributes, optional one-ring valence, export compact mesh.
  static bool roundTrip(const SemanticMesh &in, SemanticMesh &out, std::vector<int> *valence,
                        std::string *error);
};

} // namespace cad_adaptive
