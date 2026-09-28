#pragma once
#include "cad_adaptive/RegionQuality.h"
#include <string>
namespace cad_adaptive {
struct RegionCandidateSelection {
  int groupsCompared=0,groupsSelected=0;
  std::vector<uint32_t> selectedPatchIds;
  std::string stopReason;
  double seconds=0;
};
// Both refinements must retain the base vertex prefix without movement or
// target changes. New vertices have candidate-local identities, never welded
// by coordinates. New shared seam vertices couple incident patch choices.
bool selectRefinementRegions(const SemanticMesh& base,const SemanticMesh& incumbent,
    const SemanticMesh& candidate,const RemeshConfig&,float threshold,
    SemanticMesh& output,RegionCandidateSelection&,std::string* error=nullptr);
}
