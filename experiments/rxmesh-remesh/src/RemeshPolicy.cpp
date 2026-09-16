#include "cad_adaptive/RemeshPolicy.h"
#include <algorithm>

namespace cad_adaptive {

CollapseDecision RemeshPolicy::classifyCollapse(const CollapseQuery &q) {
  const int r0 = constraintRank(q.c0);
  const int r1 = constraintRank(q.c1);
  CollapseDecision d;

  const auto sameSurface = [&] {
    return q.c0 == VertexConstraint::Surface && q.c1 == VertexConstraint::Surface &&
           q.patch0 == q.patch1 && q.patch0 != kInvalidId;
  };
  const auto sameFeature = [&] {
    return isBoundaryConstraint(q.c0) && isBoundaryConstraint(q.c1) && q.feature0 != 0 &&
           q.feature0 == q.feature1;
  };

  if (r0 != r1) {
    const bool keep0 = r0 > r1;
    const VertexConstraint high = keep0 ? q.c0 : q.c1;
    const VertexConstraint low = keep0 ? q.c1 : q.c0;
    if (high == VertexConstraint::Locked || high == VertexConstraint::Corner) {
      d.allowed = true;
      d.dest = keep0 ? CollapseDest::KeepV0 : CollapseDest::KeepV1;
      return d;
    }
    if (isBoundaryConstraint(high) &&
        (low == VertexConstraint::Surface || low == VertexConstraint::Free)) {
      d.allowed = true;
      d.dest = keep0 ? CollapseDest::KeepV0 : CollapseDest::KeepV1;
      return d;
    }
    if (high == VertexConstraint::Surface && low == VertexConstraint::Free) {
      d.allowed = true;
      d.dest = keep0 ? CollapseDest::KeepV0 : CollapseDest::KeepV1;
      return d;
    }
    return d;
  }

  if (q.c0 == VertexConstraint::Free && q.c1 == VertexConstraint::Free) {
    d.allowed = true;
    d.dest = CollapseDest::Optimal;
    return d;
  }
  if (sameSurface()) {
    d.allowed = true;
    d.dest = CollapseDest::Optimal;
    return d;
  }
  if (sameFeature()) {
    d.allowed = true;
    d.dest = CollapseDest::Optimal;
    return d;
  }
  return d;
}

bool RemeshPolicy::canSplit(uint8_t) { return true; }

bool RemeshPolicy::canFlip(uint8_t edgeFlags) {
  const uint8_t blocked = EdgePatchBoundary | EdgeSharp | EdgeProtected | EdgeMeshBoundary;
  return (edgeFlags & blocked) == 0;
}

bool RemeshPolicy::lengthSplitCandidate(float length, float h0, float h1, float splitRatio) {
  const float h = edgeTarget(h0, h1);
  return h > 0 && length > splitRatio * h;
}

bool RemeshPolicy::lengthCollapseCandidate(float length, float h0, float h1, float collapseRatio) {
  const float h = edgeTarget(h0, h1);
  return h > 0 && length < collapseRatio * h;
}

VertexConstraint RemeshPolicy::inheritSplitConstraint(uint8_t edgeFlags, VertexConstraint c0,
                                                      VertexConstraint c1) {
  if (edgeFlags & (EdgePatchBoundary | EdgeMeshBoundary | EdgeSharp | EdgeProtected)) {
    if (c0 == VertexConstraint::Locked && c1 == VertexConstraint::Locked)
      return VertexConstraint::PatchBoundary;
    const int r = std::max(constraintRank(c0), constraintRank(c1));
    if (r >= constraintRank(VertexConstraint::PatchBoundary)) {
      if (c0 == VertexConstraint::FeatureEdge || c1 == VertexConstraint::FeatureEdge)
        return VertexConstraint::FeatureEdge;
      return VertexConstraint::PatchBoundary;
    }
    return VertexConstraint::PatchBoundary;
  }
  if (c0 == VertexConstraint::Surface || c1 == VertexConstraint::Surface)
    return VertexConstraint::Surface;
  return VertexConstraint::Free;
}

uint32_t RemeshPolicy::inheritSplitPatch(uint32_t patchLeft, uint32_t patchRight) {
  if (patchLeft == patchRight) return patchLeft;
  if (patchLeft == kInvalidId) return patchRight;
  if (patchRight == kInvalidId) return patchLeft;
  return std::min(patchLeft, patchRight);
}

} // namespace cad_adaptive
