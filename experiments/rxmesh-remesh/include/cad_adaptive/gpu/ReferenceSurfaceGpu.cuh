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
  int patch=0;
};
struct ReferenceBvhNode {
  float3 lower,upper;
  int first=-1,count=0,escape=0;
  float minTarget=FLT_MAX; // sizing BVH lower bound; unused by reference-triangle BVH
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
__device__ inline float sizingAt(ReferenceSurfaceGpu ref, float3 p, int patch=0) {
  float h = ref.regularLength;
  int node=0;
  while(node<(ref.sizingNodeCount ? ref.sizingNodeCount : 1)) {
    int first=0,count=ref.sizingCount;
    if(ref.sizingNodeCount) {
      const auto n=ref.sizingNodes[node];
      const float boxD2=boxDistance2(p,n);
      if(boxD2>ref.band*ref.band*1.00001f) {node=n.escape;continue;}
      if(n.minTarget<FLT_MAX) {
        if(h<=n.minTarget) {node=n.escape;continue;}
        const float span=ref.regularLength-n.minTarget;
        if(span>0.f) {
          const float maxDistance=ref.band*(h-n.minTarget)/span;
          if(boxD2>=maxDistance*maxDistance) {node=n.escape;continue;}
        }
      }
      if(n.first<0){++node;continue;}
      first=n.first;count=n.count;
    }
    for(int k=0;k<count;++k) {
      const int i=ref.sizingNodeCount ? ref.sizingIds[first+k] : k;
      const auto seg = ref.sizing[i];
      if(seg.patch!=patch)continue;
      const auto ab = sub3(seg.b, seg.a);
      float t = fminf(1.f, fmaxf(0.f, dot3(sub3(p, seg.a), ab) / fmaxf(dot3(ab, ab), 1.e-30f)));
      const auto d = sub3(p, add3(seg.a, mul3(ab, t)));
      const float d2 = dot3(d, d);
      const float span = ref.regularLength - seg.target;
      if (span > 0.f && h > seg.target) {
        const float maxDistance = ref.band * (h - seg.target) / span;
        if (d2 < maxDistance * maxDistance)
          h = seg.target + span * sqrtf(d2) / ref.band;
      }
    }
    ++node;
  }
  return h;
}__device__ inline float toleranceAt(ReferenceSurfaceGpu ref, float3 p, int patch=0) {
  return ref.sizingCount ? fminf(ref.tolerance, .08f * sizingAt(ref, p, patch))
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
static __device__ __noinline__ bool projectReference(ReferenceSurfaceGpu ref,
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
static __device__ __noinline__ bool projectReferenceHinted(ReferenceSurfaceGpu ref,
                                              int patch,float3 p,float3 &q,int &hintId) {
  float best=FLT_MAX; int bestId=2147483647; bool found=false;
  if(hintId>=0 && hintId<ref.count) {
    const auto t=ref.triangles[hintId];
    if(t.patch==patch) {
      const auto hit=closestTriangle(p,t),d=sub3(hit,p);
      best=dot3(d,d); bestId=hintId; q=hit; found=true;
    }
  }
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
  hintId=found?bestId:-1;
  return found;
}
// Smoothing makes short successive moves. An interior hit on its previous
// source triangle is already on the correct immutable reference surface.
// Boundary hits still use the full BVH so vertices can cross source triangles.
static __device__ __noinline__ bool projectReferenceHintedLocal(ReferenceSurfaceGpu ref,
    int patch,float3 p,float tolerance,float3 &q,int &hintId) {
  if(hintId>=0 && hintId<ref.count) {
    const auto t=ref.triangles[hintId];
    if(t.patch==patch) {
      const auto ab=sub3(t.b,t.a),ac=sub3(t.c,t.a);
      const float aa=dot3(ab,ab),bb=dot3(ab,ac),cc=dot3(ac,ac);
      const float det=aa*cc-bb*bb;
      if(det>1.e-24f) {
        const auto ap=sub3(p,t.a);
        const float x=(dot3(ap,ab)*cc-dot3(ap,ac)*bb)/det;
        const float y=(dot3(ap,ac)*aa-dot3(ap,ab)*bb)/det;
        if(x>1.e-4f && y>1.e-4f && x+y<.9999f) {
          const auto hit=add3(t.a,add3(mul3(ab,x),mul3(ac,y)));
          const auto d=sub3(hit,p);
          if(dot3(d,d)<=tolerance*tolerance) {q=hit;return true;}
        }
      }
    }
  }
  return projectReferenceHinted(ref,patch,p,q,hintId);
}static __device__ __noinline__ bool referenceNearWithTolerance(ReferenceSurfaceGpu ref, int patch,
                                                  float3 p, float tolerance) {
  if (!ref.count) return true;
  const float radius2=tolerance*tolerance;
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
}static __device__ __noinline__ bool referenceNear(ReferenceSurfaceGpu ref, int patch,
                                     float3 p) {
  return referenceNearWithTolerance(ref, patch, p, toleranceAt(ref, p, patch));
}
static __device__ __noinline__ bool referenceFacePointsNear4(
    ReferenceSurfaceGpu ref,int patch,const float3 (&p)[4],const float (&cachedSizing)[4],
    int skipPoint=-1) {
  if(!ref.count)return true;
  float radius2[4],bound[4];
  bool found[4]={skipPoint==0,skipPoint==1,skipPoint==2,skipPoint==3};
#ifdef __CUDACC__
#pragma unroll
#endif
  for(int i=0;i<4;++i) {
    if(i==skipPoint){radius2[i]=0.f;bound[i]=0.f;continue;}
    const float tolerance=(ref.sizingCount && cachedSizing[i]>=0.f)
        ? fminf(ref.tolerance,.08f*cachedSizing[i]) : toleranceAt(ref,p[i],patch);
    radius2[i]=tolerance*tolerance;
    bound[i]=radius2[i]+fmaxf(1.e-10f,radius2[i]*1.e-5f);
  }
  int node=0;
  while(node<(ref.nodeCount ? ref.nodeCount : 1)) {
    int first=0,count=ref.count;
    if(ref.nodeCount) {
      const auto n=ref.nodes[node];
      bool any=false;
#ifdef __CUDACC__
#pragma unroll
#endif
      for(int i=0;i<4;++i) if(!found[i] && boxDistance2(p[i],n)<=bound[i]) {any=true;break;}
      if(!any){node=n.escape;continue;}
      if(n.first<0){++node;continue;}
      first=n.first;count=n.count;
    }
    for(int k=0;k<count;++k) {
      const int id=ref.nodeCount ? ref.triangleIds[first+k] : k;
      const auto t=ref.triangles[id];
      if(t.patch!=patch)continue;
#ifdef __CUDACC__
#pragma unroll
#endif
      for(int i=0;i<4;++i) if(!found[i]) {
        const auto d=sub3(closestTriangle(p[i],t),p[i]);
        if(dot3(d,d)<=radius2[i])found[i]=true;
      }
      if(found[0]&&found[1]&&found[2]&&found[3])return true;
    }
    ++node;
  }
  return found[0]&&found[1]&&found[2]&&found[3];
}static __device__ __noinline__ bool referenceFaceSafeCached(ReferenceSurfaceGpu ref,int patch,
                                                             float3 a,float3 b,float3 c,
                                                             float sizingAB,float sizingBC,float sizingCA) {
  const float3 p[4]={mul3(add3(a,b),.5f),mul3(add3(b,c),.5f),mul3(add3(c,a),.5f),
                     mul3(add3(add3(a,b),c),1.f/3.f)};
  const float cached[4]={sizingAB,sizingBC,sizingCA,-1.f};
  return referenceFacePointsNear4(ref,patch,p,cached);
}
static __device__ __noinline__ bool referenceChangedFaceSafeCached(
    ReferenceSurfaceGpu ref,int patch,float3 a,float3 b,float3 c,int changedIndex,
    float sizingAB,float sizingBC,float sizingCA) {
  const float3 p[4]={mul3(add3(a,b),.5f),mul3(add3(b,c),.5f),mul3(add3(c,a),.5f),
                     mul3(add3(add3(a,b),c),1.f/3.f)};
  const float cached[4]={sizingAB,sizingBC,sizingCA,-1.f};
  const int skipPoint=changedIndex==0?1:(changedIndex==1?2:0);
  return referenceFacePointsNear4(ref,patch,p,cached,skipPoint);
}
static __device__ __noinline__ bool referenceFaceSafe(ReferenceSurfaceGpu ref,
                                               int patch,float3 a,float3 b,float3 c) {
  return referenceFaceSafeCached(ref,patch,a,b,c,-1.f,-1.f,-1.f);
}} // namespace cad_adaptive::gpu
