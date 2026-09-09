#pragma once

#include "CadMesh/MeshTopology.h"

namespace CadMesh {
// Annotate existing connected surface instances. This never changes ownership,
// analytic parameters or boundary constraints; an unconfirmed role is Ordinary.
// Equivalent mother-plane labels count as one support side; SupportPatchIds
// names one adjacent representative per physical side (two IDs for a fillet).
void IdentifySurfaceRoles(const MeshTopology &mesh,
                          std::vector<MeshPatch> &patches,
                          const std::vector<PatchAdjacency> &adjacency);
} // namespace CadMesh
