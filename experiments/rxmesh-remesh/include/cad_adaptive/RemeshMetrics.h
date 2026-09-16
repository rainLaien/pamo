#pragma once

#include "cad_adaptive/SemanticMesh.h"
#include <string>

namespace cad_adaptive {

void fillMeshMetrics(SemanticMesh &mesh, const RemeshConfig &config, RemeshReport &report);
std::string remeshReportJson(const RemeshReport &report);

} // namespace cad_adaptive
