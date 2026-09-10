#pragma once

#include "CadMesh/CudaAnalyticFitting.h"
#include <cstddef>
#include <memory>

namespace CadMesh {

// Internal control flow: stop an existing fitter immediately after its exact
// nonlinear inputs have been collected. Never treat this as a failed fit.
struct AnalyticSeedDeferred {};

// Diagnostic context only; does not affect solver inputs or scheduling.
class CudaAnalyticFitContext {
public:
  explicit CudaAnalyticFitContext(const char *branch);
  ~CudaAnalyticFitContext();
  CudaAnalyticFitContext(const CudaAnalyticFitContext&) = delete;
  CudaAnalyticFitContext& operator=(const CudaAnalyticFitContext&) = delete;
private:
  const char *Previous;
};

// A bounded, thread-local refinement cache. Collection does not save fitted
// patches or mutate ownership; normal execution rebuilds the support and can
// reuse a result only when every nonlinear input is byte-for-byte identical.
class CudaAnalyticSeedPrefetch {
public:
  explicit CudaAnalyticSeedPrefetch(const char* stage, bool verbose);
  ~CudaAnalyticSeedPrefetch();
  CudaAnalyticSeedPrefetch(const CudaAnalyticSeedPrefetch&) = delete;
  CudaAnalyticSeedPrefetch& operator=(const CudaAnalyticSeedPrefetch&) = delete;
  bool enabled() const;
  void begin();
  std::size_t pendingSeeds() const;
  void flush();

private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
  friend bool RefineAnalyticSeedsPrepared(
      PatchSurfaceType, const std::vector<std::array<double, 4>>&,
      std::vector<std::array<double, 8>>&, std::vector<unsigned char>&, int);
};

bool RefineAnalyticSeedsPrepared(
    PatchSurfaceType type, const std::vector<std::array<double, 4>>& samples,
    std::vector<std::array<double, 8>>& parameters,
    std::vector<unsigned char>& valid, int iterations);

} // namespace CadMesh
