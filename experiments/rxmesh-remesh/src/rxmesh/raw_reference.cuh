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
struct ReferenceBvhNode {
  float3 lower,upper;
  int first=-1,count=0,escape=0;
};
struct ReferenceSurfaceGpu {
  const ReferenceTriangleGpu *triangles = nullptr;
  int count = 0;
  float tolerance = 0;
  const SizingSegmentGpu *sizing = nullptr;
  int sizingCount = 0;
  float regularLength = 0, band = 0;
  const ReferenceBvhNode *nodes=nullptr;
  const int *triangleIds=nullptr;
  int nodeCount=0;
  const ReferenceBvhNode *sizingNodes=nullptr;
  const int *sizingIds=nullptr;
  int sizingNodeCount=0;
};
__device__ inline float3 add3(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 mul3(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ inline float boxDistance2(float3 p,const ReferenceBvhNode &t) {
  float x=fmaxf(0.f,fmaxf(t.lower.x-p.x,p.x-t.upper.x));
  float y=fmaxf(0.f,fmaxf(t.lower.y-p.y,p.y-t.upper.y));
  float z=fmaxf(0.f,fmaxf(t.lower.z-p.z,p.z-t.upper.z));
  return x*x+y*y+z*z;
}
// Immutable source features define a continuous spatial field. Re-evaluating it
// after every edit prevents collapse/smoothing from erasing the refinement.
__device__ inline float sizingAt(ReferenceSurfaceGpu ref, float3 p) {
  float h = ref.regularLength;
  int node=0;
  while(node<(ref.sizingNodeCount ? ref.sizingNodeCount : 1)) {
    int first=0,count=ref.sizingCount;
    if(ref.sizingNodeCount) {
      const auto n=ref.sizingNodes[node];
      // Beyond the transition band every descendant contributes >= regularLength.
      if(boxDistance2(p,n)>ref.band*ref.band*1.00001f) {node=n.escape;continue;}
      if(n.first<0){++node;continue;}
      first=n.first;count=n.count;
    }
    for(int k=0;k<count;++k) {
      const int i=ref.sizingNodeCount ? ref.sizingIds[first+k] : k;
    const auto s = ref.sizing[i];
    const auto ab = sub3(s.b, s.a);
    float t = fminf(
        1.f, fmaxf(0.f, dot3(sub3(p, s.a), ab) / fmaxf(dot3(ab, ab), 1.e-30f)));
    const auto d = sub3(p, add3(s.a, mul3(ab, t)));
    h = fminf(h, s.target + (ref.regularLength - s.target) * sqrtf(dot3(d, d)) /
                                ref.band);
    }
    ++node;
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
// Stackless immutable BVH. Equal-distance ties retain original source order.
__device__ __noinline__ bool projectReference(ReferenceSurfaceGpu ref,
                                              int patch,float3 p,float3 &q) {
  float best=FLT_MAX; int bestId=2147483647; bool found=false;
  int node=0;
  while(node<(ref.nodeCount ? ref.nodeCount : 1)) {
    int first=0,count=ref.count;
    if(ref.nodeCount) {
      const auto n=ref.nodes[node];
      if(best<FLT_MAX && boxDistance2(p,n)>best+fmaxf(1.e-10f,best*1.e-5f)) {node=n.escape;continue;}
      if(n.first<0) {++node;continue;}
      first=n.first;count=n.count;
    }
    for(int k=0;k<count;++k) {
      const int id=ref.nodeCount ? ref.triangleIds[first+k] : k;
      const auto t=ref.triangles[id];
      if(t.patch!=patch)continue;
      const auto hit=closestTriangle(p,t),d=sub3(hit,p);
      const float d2=dot3(d,d);
      if(d2<best || (d2==best && id<bestId)) {best=d2;bestId=id;q=hit;found=true;}
    }
    ++node;
  }
  return found;
}
__device__ __noinline__ bool referenceNear(ReferenceSurfaceGpu ref, int patch,
                                     float3 p) {
  if (!ref.count)
    return true;
  const float tolerance = toleranceAt(ref, p);
  const float radius2=tolerance*tolerance;
  // Feasibility needs any source point inside the original tolerance, not
  // the exact closest point. Prune with a conservative bound and stop at
  // the first witness. The actual acceptance predicate is unchanged.
  const float bound=radius2+fmaxf(1.e-10f,radius2*1.e-5f);
  int node=0;
  while(node<(ref.nodeCount ? ref.nodeCount : 1)) {
    int first=0,count=ref.count;
    if(ref.nodeCount) {
      const auto n=ref.nodes[node];
      if(boxDistance2(p,n)>bound){node=n.escape;continue;}
      if(n.first<0){++node;continue;}
      first=n.first;count=n.count;
    }
    for(int k=0;k<count;++k) {
      const int id=ref.nodeCount ? ref.triangleIds[first+k] : k;
      const auto t=ref.triangles[id];
      if(t.patch!=patch)continue;
      const auto d=sub3(closestTriangle(p,t),p);
      if(dot3(d,d)<=radius2)return true;
    }
    ++node;
  }
  return false;
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
