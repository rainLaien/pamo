#pragma once
#include <cfloat>
#include <cuda_runtime.h>
namespace cad_adaptive::gpu {
__device__ inline float3 sub3(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline float3 normal3(float3 a, float3 b, float3 c) {
  b = sub3(b, a);
  c = sub3(c, a);
  return make_float3(b.y * c.z - b.z * c.y, b.z * c.x - b.x * c.z,
                     b.x * c.y - b.y * c.x);
}

struct ReferenceTriangleGpu {
  float3 a, b, c;
  int patch;
};
struct SizingSegmentGpu {
  float3 a, b;
  float target;
};
struct ReferenceSurfaceGpu {
  const ReferenceTriangleGpu *triangles = nullptr;
  int count = 0;
  float tolerance = 0;
  const SizingSegmentGpu *sizing = nullptr;
  int sizingCount = 0;
  float regularLength = 0, band = 0;
};
__device__ inline float3 add3(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 mul3(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
// Immutable source features define a continuous spatial field. Re-evaluating it
// after every edit prevents collapse/smoothing from erasing the refinement.
__device__ inline float sizingAt(ReferenceSurfaceGpu ref, float3 p) {
  float h = ref.regularLength;
  for (int i = 0; i < ref.sizingCount; ++i) {
    const auto s = ref.sizing[i];
    const auto ab = sub3(s.b, s.a);
    float t = fminf(
        1.f, fmaxf(0.f, dot3(sub3(p, s.a), ab) / fmaxf(dot3(ab, ab), 1.e-30f)));
    const auto d = sub3(p, add3(s.a, mul3(ab, t)));
    h = fminf(h, s.target + (ref.regularLength - s.target) * sqrtf(dot3(d, d)) /
                                ref.band);
  }
  return h;
}
__device__ inline float toleranceAt(ReferenceSurfaceGpu ref, float3 p) {
  return ref.sizingCount ? fminf(ref.tolerance, .08f * sizingAt(ref, p))
                         : ref.tolerance;
}
__device__ inline float qualityVcg(float3 a, float3 b, float3 c) {
  const auto n = normal3(a, b, c), ab = sub3(a, b), bc = sub3(b, c),
             ca = sub3(c, a);
  const float d = fmaxf(dot3(ab, ab), fmaxf(dot3(bc, bc), dot3(ca, ca)));
  return d > 0 ? sqrtf(dot3(n, n)) / d : 0;
}
__device__ inline float3 closestTriangle(float3 p,
                                         const ReferenceTriangleGpu &t) {
  const auto ab = sub3(t.b, t.a), ac = sub3(t.c, t.a), ap = sub3(p, t.a);
  const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
  if (d1 <= 0 && d2 <= 0)
    return t.a;
  const auto bp = sub3(p, t.b);
  const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
  if (d3 >= 0 && d4 <= d3)
    return t.b;
  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0)
    return add3(t.a, mul3(ab, d1 / (d1 - d3)));
  const auto cp = sub3(p, t.c);
  const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
  if (d6 >= 0 && d5 <= d6)
    return t.c;
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0)
    return add3(t.a, mul3(ac, d2 / (d2 - d6)));
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 >= d3 && d5 >= d6)
    return add3(t.b, mul3(sub3(t.c, t.b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
  const float denom = va + vb + vc;
  if (!(denom > 0))
    return t.a;
  return add3(t.a, add3(mul3(ab, vb / denom), mul3(ac, vc / denom)));
}
// Immutable source triangles. Exhaustive search is intentional for the small
// raw-STL reference; it is exact nearest-triangle search, not a moving target.
__device__ __noinline__ bool projectReference(ReferenceSurfaceGpu ref,
                                              int patch, float3 p, float3 &q) {
  float best = FLT_MAX;
  bool found = false;
  for (int i = 0; i < ref.count; ++i) {
    const auto t = ref.triangles[i];
    if (t.patch != patch)
      continue;
    const auto hit = closestTriangle(p, t), d = sub3(hit, p);
    const float d2 = dot3(d, d);
    if (d2 < best) {
      best = d2;
      q = hit;
      found = true;
    }
  }
  return found;
}
__device__ inline bool referenceNear(ReferenceSurfaceGpu ref, int patch,
                                     float3 p) {
  if (!ref.count)
    return true;
  float3 q;
  if (!projectReference(ref, patch, p, q))
    return false;
  const auto d = sub3(q, p);
  const float tolerance = toleranceAt(ref, p);
  return dot3(d, d) <= tolerance * tolerance;
}
__device__ __noinline__ bool referenceFaceSafe(ReferenceSurfaceGpu ref,
                                               int patch, float3 a, float3 b,
                                               float3 c) {
  return referenceNear(ref, patch, mul3(add3(a, b), .5f)) &&
         referenceNear(ref, patch, mul3(add3(b, c), .5f)) &&
         referenceNear(ref, patch, mul3(add3(c, a), .5f)) &&
         referenceNear(ref, patch, mul3(add3(add3(a, b), c), 1.f / 3.f));
}
} // namespace cad_adaptive::gpu
