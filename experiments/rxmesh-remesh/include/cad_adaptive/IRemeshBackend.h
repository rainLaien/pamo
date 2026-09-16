#pragma once

#include "cad_adaptive/SemanticMesh.h"

namespace cad_adaptive {

class IRemeshBackend {
public:
  virtual ~IRemeshBackend() = default;
  virtual bool remesh(SemanticMesh &mesh, const RemeshConfig &config, RemeshReport &report) = 0;
};

class CpuRemeshBackend final : public IRemeshBackend {
public:
  bool remesh(SemanticMesh &mesh, const RemeshConfig &config, RemeshReport &report) override;
};

} // namespace cad_adaptive
