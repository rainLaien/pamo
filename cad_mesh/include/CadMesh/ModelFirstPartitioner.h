#pragma once
#include "CadMesh/MeshTopology.h"
namespace CadMesh {
// Semantic surfaces are recovered before any raw differential cue is locked.
std::vector<MeshPatch>
PartitionBySurfaceModels(MeshTopology &mesh, const SegmentationConfig &config);
// Consolidate supplied connected analytic fragments without crossing hard
// edges. Every accepted union is refitted and checked against all its faces.
std::vector<MeshPatch> MergeAdjacentSurfaceModels(
    MeshTopology &mesh, std::vector<MeshPatch> patches,
    const SegmentationConfig &config);
} // namespace CadMesh
