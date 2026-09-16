#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace cad_adaptive {

// Numeric values match CadMesh::PatchSurfaceType.
enum class PatchType : uint8_t {
  Unknown = 0,
  Plane,
  Cylinder,
  Cone,
  Sphere,
  Torus,
  Freeform
};

enum class VertexConstraint : uint8_t {
  Free = 0,
  Surface,
  FeatureEdge,
  PatchBoundary,
  Corner,
  Locked
};

enum class CollapseDest : uint8_t { Reject = 0, KeepV0, KeepV1, Optimal };

enum EdgeFlag : uint8_t {
  EdgeNone = 0,
  EdgePatchBoundary = 1,
  EdgeSharp = 2,
  EdgeProtected = 4,
  EdgeMeshBoundary = 8
};

struct Vec3 {
  float x = 0, y = 0, z = 0;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }
inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length2(Vec3 a) { return dot(a, a); }
inline float length(Vec3 a) { return std::sqrt(length2(a)); }

inline Vec3 normalize(Vec3 a, Vec3 fallback = {1, 0, 0}) {
  const float n = length(a);
  return n > 1e-20f ? a * (1.0f / n) : fallback;
}

inline float distance(Vec3 a, Vec3 b) { return length(a - b); }
inline float clampf(float x, float lo, float hi) { return std::min(hi, std::max(lo, x)); }
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }
inline float smoothstep(float t) {
  t = clampf(t, 0, 1);
  return t * t * (3 - 2 * t);
}

inline Vec3 closestOnSegment(Vec3 p, Vec3 a, Vec3 b) {
  const Vec3 ab = b - a;
  const float d = length2(ab);
  if (d <= 1e-20f) return a;
  const float t = clampf(dot(p - a, ab) / d, 0, 1);
  return a + ab * t;
}

// 0 degenerate, 1 equilateral.
inline float triangleQuality(Vec3 a, Vec3 b, Vec3 c) {
  const Vec3 ab = b - a, ac = c - a, bc = c - b;
  const float area2 = length(cross(ab, ac));
  const float denom = length2(ab) + length2(ac) + length2(bc);
  if (!(denom > 0) || !(area2 > 0)) return 0;
  return clampf(2.0f * std::sqrt(3.0f) * area2 / denom, 0, 1);
}

inline Vec3 triangleNormal(Vec3 a, Vec3 b, Vec3 c) { return normalize(cross(b - a, c - a)); }
inline Vec3 centroid3(Vec3 a, Vec3 b, Vec3 c) { return (a + b + c) * (1.0f / 3.0f); }

struct PatchRecord {
  PatchType type = PatchType::Unknown;
  Vec3 origin{};
  Vec3 axis{0, 0, 1}; // plane normal or cylinder axis
  float radius = 0;
};

struct RemeshConfig {
  float maxGeometryError = 0.01f;
  float constantLength = 0; // 0 = derive hMax from bbox
  float hMin = 0;           // 0 = ε
  float hMax = 0;           // 0 = max(constantLength, 0.05 * bbox)
  float featureBand = 0;    // 0 = 4 * hRegular
  float featureEdgeLength = 0;
  float splitRatio = 4.0f / 3.0f;
  float collapseRatio = 4.0f / 5.0f;
  float minQuality = 1e-5f;
  float normalDegrees = 10;
  int maxIterations = 5;
  bool adaptive = true;
  bool enableSplit = true;
  bool enableCollapse = true;
  bool enableFlip = true;
  bool enableSmooth = true;
  float smoothLambda = 0.5f;
  float valenceWeight = 1;
  float shapeWeight = 1;
  float sizingWeight = 1;
};

struct RemeshReport {
  int splits = 0, collapses = 0, flips = 0, smoothMoves = 0;
  int cavityRefines = 0;
  int splitCandidates = 0, collapseCandidates = 0, flipCandidates = 0;
  int cavityCandidates = 0;
  int rejectTopology = 0, rejectPatch = 0, rejectFeature = 0;
  int rejectNormal = 0, rejectQuality = 0, rejectError = 0;
  float qualityMean = 0, qualityP05 = 0, qualityMin = 0;
  float sizingErrorMean = 0, sizingErrorP95 = 0;
  float geometryErrorMax = 0;
  int movedLockedVertices = 0, missingBoundaryEdges = 0;
  double seconds = 0;
  double secondsSetup = 0, secondsCavity = 0, secondsSplit = 0, secondsCollapse = 0,
         secondsFlip = 0, secondsSmooth = 0;
  bool topologyValid = false;
  bool constraintsHeld = false;
};

inline int constraintRank(VertexConstraint c) {
  switch (c) {
  case VertexConstraint::Locked:
    return 5;
  case VertexConstraint::Corner:
    return 4;
  case VertexConstraint::PatchBoundary:
  case VertexConstraint::FeatureEdge:
    return 3;
  case VertexConstraint::Surface:
    return 2;
  case VertexConstraint::Free:
    return 1;
  }
  return 0;
}

inline bool isBoundaryConstraint(VertexConstraint c) {
  return c == VertexConstraint::FeatureEdge || c == VertexConstraint::PatchBoundary;
}

inline constexpr uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kOpenBoundaryFeature = 1;

} // namespace cad_adaptive
