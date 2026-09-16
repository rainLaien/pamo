#pragma once

#include "cad_adaptive/SemanticMesh.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>

namespace cad_adaptive::global {

struct GlobalTopologyValidation {
  uint32_t invalidVertexReference = 0;
  uint32_t invalidFaceReference = 0;
  uint32_t degenerateFace = 0;
  uint32_t zeroAreaFace = 0;
  uint32_t edgeFaceMismatch = 0;
  uint32_t staleCandidate = 0;

  bool ok() const {
    return invalidVertexReference == 0 && invalidFaceReference == 0 &&
           degenerateFace == 0 && zeroAreaFace == 0 && edgeFaceMismatch == 0;
  }
};

struct GlobalSplitReport {
  uint32_t candidateCount = 0;
  uint32_t acceptedCount = 0;
  uint32_t staleRejected = 0;
  uint32_t capacityRejected = 0;
  uint32_t semanticRejected = 0;
  uint32_t projectionApplied = 0;
  uint32_t projectionFailed = 0;
  uint32_t schedulerRounds = 0;
  uint64_t activeItemsScanned = 0;

  double candidateMs = 0;
  double claimMs = 0;
  double executeMs = 0;
  double validateMs = 0;
  double totalMs = 0;

  size_t globalMemoryBytes = 0;
  size_t dynamicSharedMemoryBytes = 0;
};

struct GlobalTriangleRefineReport {
  uint32_t candidateCount = 0;
  uint32_t acceptedCount = 0;
  uint32_t staleRejected = 0;
  uint32_t semanticRejected = 0;
  uint32_t qualityRejected = 0;
  uint32_t projectionApplied = 0;
  uint32_t projectionFailed = 0;
  uint32_t protectedRejected = 0;
  uint32_t neighborRejected = 0;
  uint32_t windingRejected = 0;
  uint32_t belowRatioRejected = 0;
  uint32_t editableFaceHistogram[4] = {0,0,0,0};
  uint32_t schedulerRounds = 0;
  uint64_t activeItemsScanned = 0;

  double candidateMs = 0;
  double claimMs = 0;
  double executeMs = 0;
  double totalMs = 0;

  size_t globalMemoryBytes = 0;
  size_t dynamicSharedMemoryBytes = 0;
};

struct GlobalFlipReport {
  uint32_t candidateCount = 0;
  uint32_t acceptedCount = 0;
  uint32_t staleRejected = 0;
  uint32_t semanticRejected = 0;
  uint32_t schedulerRounds = 0;
  uint64_t activeItemsScanned = 0;

  double candidateMs = 0;
  double claimMs = 0;
  double executeMs = 0;
  double totalMs = 0;

  size_t globalMemoryBytes = 0;
  size_t dynamicSharedMemoryBytes = 0;
};

class GlobalSplitBackend {
public:
  GlobalSplitBackend();
  ~GlobalSplitBackend();
  GlobalSplitBackend(const GlobalSplitBackend &) = delete;
  GlobalSplitBackend &operator=(const GlobalSplitBackend &) = delete;

  bool Initialize(const SemanticMesh &mesh, float capacityFactor, std::string *error = nullptr);
  bool RunTriangleRefinePass(float refineRatio, GlobalTriangleRefineReport &report, std::string *error = nullptr);
  bool RunSplitPass(float splitRatio, GlobalSplitReport &report,
                    std::string *error = nullptr,
                    float maxRatio = std::numeric_limits<float>::infinity());
  bool RunFlipPass(float minQualityGain, GlobalFlipReport &report, std::string *error = nullptr);
  bool Validate(GlobalTopologyValidation &validation, std::string *error = nullptr) const;
  bool Export(SemanticMesh &mesh, std::string *error = nullptr) const;

  uint32_t VertexCount() const;
  uint32_t EdgeCount() const;
  uint32_t FaceCount() const;
  size_t GlobalMemoryBytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
};

} // namespace cad_adaptive::global
