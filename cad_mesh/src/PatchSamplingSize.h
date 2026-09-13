#pragma once
#include <algorithm>

namespace CadMesh {
// The shared-boundary pass and cylinder chart must use the same metric.
// Small-radius strips need axial rows comparable to their angular sampling.
inline double CylinderAxialSamplingSize(double target, double circumferential) {
  return std::min(target, 2.0 * circumferential);
}
}
