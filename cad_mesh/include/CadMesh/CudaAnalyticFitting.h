#pragma once

#include "CadMesh/Types.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace CadMesh {

// One independently normalized support cloud and its model's initial guesses.
// Different clouds and model types can share a single CUDA launch.
struct AnalyticSeedBatch {
  PatchSurfaceType Type = PatchSurfaceType::Unknown;
  std::vector<std::array<double, 4>> Samples;
  std::vector<std::array<double, 8>> Parameters;
  std::vector<unsigned char> Valid;
  int Iterations = 0;
};

// A session enables CUDA nonlinear seed refinement for this thread only.
// Small model initialization solves and full geometric certificates stay in
// the existing fitters. No CUDA SDK is needed to build the C++ library.
class CudaAnalyticFitSession {
public:
  CudaAnalyticFitSession(AnalyticSeedBackend backend, bool verbose);
  ~CudaAnalyticFitSession();
  CudaAnalyticFitSession(const CudaAnalyticFitSession &) = delete;
  CudaAnalyticFitSession &operator=(const CudaAnalyticFitSession &) = delete;

private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
  friend bool RefineAnalyticSeedsCuda(
      PatchSurfaceType, const std::vector<std::array<double, 4>> &,
      std::vector<std::array<double, 8>> &, std::vector<unsigned char> &, int);
  friend bool RefineAnalyticSeedBatchCuda(std::vector<AnalyticSeedBatch> &);
  friend bool CudaAnalyticFitAvailable();
};

// Samples contain local normalized x/y/z and their quadrature weight.
// Parameters use the same order as SurfaceFitting.cpp, padded to eight doubles.
// false means no CUDA work was performed: caller runs the existing CPU solver.
// true returns one validity byte per seed; failed seeds can retry on the CPU.
bool RefineAnalyticSeedsCuda(
    PatchSurfaceType type, const std::vector<std::array<double, 4>> &samples,
    std::vector<std::array<double, 8>> &parameters,
    std::vector<unsigned char> &valid, int iterations);

bool RefineAnalyticSeedBatchCuda(std::vector<AnalyticSeedBatch> &batches);
bool CudaAnalyticFitAvailable();

// Runtime-compiled remesh primitive used by the native C++ pipeline. The
// stable CUDA Driver/NVRTC ABI avoids nvcc, cl.exe and LibTorch.
bool ClassifyLongEdgesCudaRuntime(
    const std::vector<std::array<double, 3>> &vertices,
    const std::vector<std::array<int, 2>> &edges, double maximumEdgeLength,
    std::vector<unsigned char> &selected,
    std::vector<std::array<double, 3>> &midpoints, std::string &error);

} // namespace CadMesh
