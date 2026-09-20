#pragma once
#include "cad_adaptive/SemanticMesh.h"
#include "cad_adaptive/Types.h"
#include "cad_adaptive/GeometryProjector.h"

namespace cad_adaptive {

struct TriangleRefineStats {
  int InputFaces = 0;
  int OutputFaces = 0;
  int RefinedFaces = 0;
  int InsertedVertices = 0;
  int SplitEdges = 0;
  int MaskCounts[8] = {0,0,0,0,0,0,0,0};
};

// VCGLib RefineMidpoint-style conforming refinement. All candidate edges are
// marked first, each shared edge gets exactly one midpoint, then every face is
// retriangulated from its 3-bit split mask.
bool refineMidpointConforming(const SemanticMesh& input, const RemeshConfig& config,
                              const GeometryProjector& referenceProjector,
                              SemanticMesh& output, TriangleRefineStats& stats,
                              std::string* error = nullptr);

// Multi-level conforming coarse refinement. Each level globally marks long
// shared edges before retriangulation, so adjacent faces always reuse the same
// subdivision vertex and no fan/spoke topology is introduced.
bool refineCoarseConforming(const SemanticMesh& input, const RemeshConfig& config,
                            const GeometryProjector& referenceProjector,
                            SemanticMesh& output, TriangleRefineStats& stats,
                            int maxLevels = 6, std::string* error = nullptr);

} // namespace cad_adaptive
