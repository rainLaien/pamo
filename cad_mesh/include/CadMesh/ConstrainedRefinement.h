#pragma once
#include <array>
#include <vector>

namespace CadMesh {
// Independent geometric candidate stage. Connectivity stays on the CPU.
// Input: ax,ay,bx,by,cx,cy,target,sin(minimum angle).
// Output: circumcenter x/y, offcenter x/y, shortest length, priority (0 invalid).
using RefinementInput = std::array<double,8>;
using RefinementCandidate = std::array<double,6>;
struct RefinementBatchResult {
  std::vector<RefinementCandidate> Candidates;
  bool UsedCuda=false;
};
RefinementBatchResult EvaluateRefinementBatch(const std::vector<RefinementInput>&,bool allowCuda);
// Each patch worker owns its execution policy. Nested calls restore it.
inline thread_local bool RefinementCudaEnabled=false;
class RefinementExecutionScope {
  bool Previous;
public:
  explicit RefinementExecutionScope(bool enabled):Previous(RefinementCudaEnabled){RefinementCudaEnabled=enabled;}
  ~RefinementExecutionScope(){RefinementCudaEnabled=Previous;}
  RefinementExecutionScope(const RefinementExecutionScope&)=delete;
  RefinementExecutionScope& operator=(const RefinementExecutionScope&)=delete;
};
}
