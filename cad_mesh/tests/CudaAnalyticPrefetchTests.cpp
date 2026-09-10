#include "CadMesh/CudaAnalyticPrefetch.h"
#include <iostream>
#include <stdexcept>

namespace {
bool Available = true;
int BatchCalls = 0, DirectCalls = 0;
void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}

// A deterministic solver substitute isolates scheduling and cache validity;
// this test runs without a CUDA driver or a device.
namespace CadMesh {
bool CudaAnalyticFitAvailable() { return Available; }
bool RefineAnalyticSeedBatchCuda(std::vector<AnalyticSeedBatch>& batches) {
  ++BatchCalls;
  if (!Available) return false;
  for (auto& batch : batches) {
    batch.Valid.assign(batch.Parameters.size(), 1);
    for (auto& seed : batch.Parameters)
      seed[0] += batch.Samples[0][0] + batch.Iterations + int(batch.Type);
  }
  return true;
}
bool RefineAnalyticSeedsCuda(PatchSurfaceType type,
    const std::vector<std::array<double, 4>>& samples,
    std::vector<std::array<double, 8>>& parameters,
    std::vector<unsigned char>& valid, int iterations) {
  ++DirectCalls;
  if (!Available) return false;
  std::vector<AnalyticSeedBatch> jobs(1);
  jobs[0].Type = type;
  jobs[0].Samples = samples;
  jobs[0].Parameters = parameters;
  jobs[0].Iterations = iterations;
  RefineAnalyticSeedBatchCuda(jobs);
  parameters = jobs[0].Parameters;
  valid = jobs[0].Valid;
  return true;
}
}

int main() {
  using namespace CadMesh;
  try {
    const std::vector<std::array<double, 4>> samples{{1, 2, 3, 1}, {2, 3, 4, 1}};
    const std::vector<std::array<double, 8>> initial{{2, 0, 0, 1, 0, 0, 1, 0}};
    std::vector<unsigned char> valid;
    CudaAnalyticSeedPrefetch prepared("test", false);
    prepared.begin();
    for (auto type : {PatchSurfaceType::Cylinder, PatchSurfaceType::Sphere,
                      PatchSurfaceType::Cylinder}) {
      auto parameters = initial;
      bool deferred = false;
      try { RefineAnalyticSeedsPrepared(type, samples, parameters, valid, 7); }
      catch (const AnalyticSeedDeferred&) { deferred = true; }
      Check(deferred, "collection must stop before consuming unrefined candidates");
      Check(parameters == initial, "collection modified initial seed");
    }
    Check(prepared.pendingSeeds() == 2 && BatchCalls == 0,
          "independent jobs must wait for one combined submission");
    prepared.flush();
    Check(BatchCalls == 1 && DirectCalls == 0, "jobs were not batched");
    auto parameters = initial;
    Check(RefineAnalyticSeedsPrepared(PatchSurfaceType::Cylinder, samples,
                                     parameters, valid, 7), "cache replay failed");
    Check(parameters[0][0] == 12 && valid == std::vector<unsigned char>{1},
          "cache did not return refined parameters and validity");
    Check(DirectCalls == 0, "exact replay repeated the solver");
    prepared.begin();
    parameters = initial;
    bool replayDeferred = false;
    try { RefineAnalyticSeedsPrepared(PatchSurfaceType::Cylinder, samples,
                                     parameters, valid, 7); }
    catch (const AnalyticSeedDeferred&) { replayDeferred = true; }
    Check(replayDeferred && prepared.pendingSeeds() == 0,
          "collecting a ready result must stop without queuing duplicate work");
    prepared.flush();
    Check(BatchCalls == 1, "ready input was unnecessarily resubmitted");
    for (int changed = 0; changed < 7; ++changed) {
      auto points = samples;
      parameters = initial;
      auto type = PatchSurfaceType::Cylinder;
      int iterations = 7;
      if (changed == 0) points[0][0] += 1;
      if (changed == 1) parameters[0][0] += 1;
      if (changed == 2) type = PatchSurfaceType::Cone;
      if (changed == 3) ++iterations;
      if (changed == 4) points.pop_back(); // A neighboring seed claimed a face.
      if (changed == 5) std::swap(points[0], points[1]); // Different BFS support order.
      if (changed == 6) points[0][3] += 1; // Identical points with changed fit weights.
      Check(RefineAnalyticSeedsPrepared(type, points, parameters, valid, iterations),
            "changed inputs should use normal solver");
      Check(DirectCalls == changed + 1,
            "a changed surface, seed, model type or iteration count reused stale output");
    }
    Available = false;
    parameters = initial;
    Check(!RefineAnalyticSeedsPrepared(PatchSurfaceType::Cylinder, samples,
                                      parameters, valid, 7), "CPU fallback was bypassed");
    Check(parameters == initial, "CPU fallback must preserve initial parameters");
    std::cout << "PASS: analytic prefetch batching, exact input guards, CPU fallback\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
