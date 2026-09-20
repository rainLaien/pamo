#pragma once
#include "cad_adaptive/GeometryProjector.h"
#include "cad_adaptive/SemanticMesh.h"

namespace cad_adaptive {
int relaxAndProject(SemanticMesh& mesh, const GeometryProjector& reference,
                    int iterations, float lambda);
}
