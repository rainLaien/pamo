#include "CadMesh/ConstrainedRefinement.h"
#include "RefinementCandidateMath.h"
#include <cmath>
#include <cstdlib>
#include <string>

namespace CadMesh {
namespace {
CADMESH_CANDIDATE_FUNCTION(inline)
}
bool EvaluateRefinementCuda(const std::vector<RefinementInput>&,
                           std::vector<RefinementCandidate>&,std::string&);
RefinementBatchResult EvaluateRefinementBatch(const std::vector<RefinementInput> &input,bool allowCuda){
  RefinementBatchResult result;
  const auto sanitize=[&]{for(auto &c:result.Candidates)for(double v:c)if(!std::isfinite(v)){c={};break;}};
  // Explicit opt-in until the user has compared the new path. CPU mode never
  // initializes CUDA. Small batches avoid launch and transfer overhead.
  const char *setting=std::getenv("CADMESH_CDT_CUDA");
  if(allowCuda && setting && std::string(setting)=="1" && input.size()>=2048){
    std::string error;
    if(EvaluateRefinementCuda(input,result.Candidates,error)){result.UsedCuda=true;sanitize();return result;}
  }
  result.Candidates.resize(input.size());
  for(std::size_t i=0;i<input.size();++i)cadmesh_refinement_candidate(input[i].data(),result.Candidates[i].data());
  sanitize();
  return result;
}
}
