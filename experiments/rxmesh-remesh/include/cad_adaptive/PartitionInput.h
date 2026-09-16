#pragma once
#include "cad_adaptive/SemanticMesh.h"

namespace cad_adaptive {
// Read the indexed CADPART1 handoff produced by PAMO's snapshot packager.
// Unsupported projection surfaces fail explicitly rather than remeshing freely.
bool loadPartitionInput(const std::string &path, SemanticMesh &mesh, std::string *error);
int refinePartitionBoundary(SemanticMesh &mesh, float maxLength);
bool validatePartitionOutput(const SemanticMesh &source, const SemanticMesh &output,
                             const RemeshConfig &config, RemeshReport &report, std::string *error);
}
