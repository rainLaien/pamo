#pragma once

#include "cad_adaptive/Types.h"

namespace cad_adaptive {

struct CollapseQuery {
  VertexConstraint c0 = VertexConstraint::Free;
  VertexConstraint c1 = VertexConstraint::Free;
  uint32_t patch0 = 0, patch1 = 0;
  uint32_t feature0 = 0, feature1 = 0;
};

struct CollapseDecision {
  bool allowed = false;
  CollapseDest dest = CollapseDest::Reject;
};

class RemeshPolicy {
public:
  static CollapseDecision classifyCollapse(const CollapseQuery &q);

  static bool canSplit(uint8_t edgeFlags);
  static bool canFlip(uint8_t edgeFlags);

  static bool lengthSplitCandidate(float length, float h0, float h1, float splitRatio);
  static bool lengthCollapseCandidate(float length, float h0, float h1, float collapseRatio);
  static float edgeTarget(float h0, float h1) { return 0.5f * (h0 + h1); }

  static VertexConstraint inheritSplitConstraint(uint8_t edgeFlags, VertexConstraint c0,
                                                 VertexConstraint c1);
  static uint32_t inheritSplitPatch(uint32_t patchLeft, uint32_t patchRight);
};

} // namespace cad_adaptive
