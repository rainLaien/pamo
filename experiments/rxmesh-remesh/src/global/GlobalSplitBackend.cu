#include "cad_adaptive/global/GlobalSplitBackend.h"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace cad_adaptive::global {
namespace {

constexpr uint32_t kInvalid = 0xffffffffu;
constexpr uint8_t kSurfaceConstraint = uint8_t(VertexConstraint::Surface);

struct GPatch {
  float ox, oy, oz;
  float ax, ay, az;
  float radius;
  float orientationSign;
  uint8_t type;
  uint8_t padding[3]{};
};

struct GVertex {
  float x, y, z, targetLength;
  uint32_t patchId, generation;
  uint8_t constraint, alive;
  uint16_t padding = 0;
};

struct GEdge {
  uint32_t v0, v1;
  int32_t f0, f1;
  uint32_t generation;
  uint8_t flags, alive;
  uint16_t padding = 0;
};

struct GFace {
  uint32_t v0, v1, v2;
  uint32_t e0, e1, e2;
  uint32_t patchId, generation;
  uint8_t alive;
  uint8_t padding[3]{};
};

struct SplitCandidate {
  uint32_t edgeId, edgeGeneration;
  uint32_t a, b, c, d;
  uint32_t f0, f1;
  float priority;
  uint8_t segmentCount;
  uint8_t accepted;
  uint8_t padding[2]{};
};

struct FlipCandidate {
  uint32_t edgeId, edgeGeneration;
  uint32_t a, b, c, d;
  uint32_t f0, f1;
  float oldQuality, newQuality;
  uint8_t accepted;
  uint8_t padding[3]{};
};

struct CollapseCandidate {
  uint32_t edgeId, edgeGeneration;
  uint32_t keep, remove, c, d;
  uint32_t f0, f1;
  float priority;
  uint8_t accepted;
  uint8_t padding[3]{};
};

struct TriangleRefineCandidate {
  uint32_t faceId, faceGeneration;
  uint32_t v0, v1, v2;
  uint32_t e0, e1, e2;
  uint32_t n0, n1, n2;
  uint32_t d0, d1, d2;
  uint32_t ng0, ng1, ng2;
  uint32_t r0, r1, r2, centerReplace;
  float priority;
  uint8_t edgeMask;
  uint8_t layout;
  uint8_t accepted;
  uint8_t padding = 0;
};



__device__ inline GVertex ProjectAnalyticVertex(GVertex v, const GPatch *patches,
                                                  uint32_t patchCount,
                                                  uint32_t *projectionApplied,
                                                  uint32_t *projectionFailed) {
  if (!patches || v.patchId >= patchCount) return v;
  const GPatch p = patches[v.patchId];
  const float3 x = make_float3(v.x, v.y, v.z);
  const float3 o = make_float3(p.ox, p.oy, p.oz);
  const float3 a = make_float3(p.ax, p.ay, p.az);
  const float3 off = make_float3(x.x-o.x, x.y-o.y, x.z-o.z);
  const float axial = off.x*a.x + off.y*a.y + off.z*a.z;
  if (p.type == uint8_t(PatchType::Plane)) {
    const float3 q = make_float3(x.x-a.x*axial, x.y-a.y*axial, x.z-a.z*axial);
    v.x=q.x; v.y=q.y; v.z=q.z;
    if (projectionApplied) atomicAdd(projectionApplied, 1u);
  } else if (p.type == uint8_t(PatchType::Cylinder)) {
    const float3 radial = make_float3(off.x-a.x*axial, off.y-a.y*axial, off.z-a.z*axial);
    const float rn2 = radial.x*radial.x + radial.y*radial.y + radial.z*radial.z;
    if (!(p.radius > 0.0f) || !(rn2 > 1e-20f)) {
      if (projectionFailed) atomicAdd(projectionFailed, 1u);
      return v;
    }
    const float scale = p.radius * rsqrtf(rn2);
    v.x=o.x+a.x*axial+radial.x*scale;
    v.y=o.y+a.y*axial+radial.y*scale;
    v.z=o.z+a.z*axial+radial.z*scale;
    if (projectionApplied) atomicAdd(projectionApplied, 1u);
  }
  return v;
}

struct ValidationCounters {
  uint32_t invalidVertexReference;
  uint32_t invalidFaceReference;
  uint32_t degenerateFace;
  uint32_t zeroAreaFace;
  uint32_t edgeFaceMismatch;
  uint32_t staleCandidate;
};

inline uint64_t EdgeKey(uint32_t a, uint32_t b) {
  if (a > b) std::swap(a, b);
  return (uint64_t(a) << 32) | uint64_t(b);
}

template <class T>
T *AllocManaged(size_t count) {
  T *ptr = nullptr;
  if (cudaMallocManaged(&ptr, sizeof(T) * count) != cudaSuccess)
    throw std::runtime_error("cudaMallocManaged failed");
  return ptr;
}

inline void CheckCuda(cudaError_t err, const char *where) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
}

__device__ inline float Dist2(const GVertex &a, const GVertex &b) {
  const float x = a.x - b.x, y = a.y - b.y, z = a.z - b.z;
  return x * x + y * y + z * z;
}

__device__ inline float3 ToFloat3(const GVertex &v) { return make_float3(v.x, v.y, v.z); }

__device__ inline float3 Sub3(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 Cross3(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ inline float Dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 TriangleCross(const GVertex &a, const GVertex &b, const GVertex &c) {
  return Cross3(Sub3(ToFloat3(b), ToFloat3(a)), Sub3(ToFloat3(c), ToFloat3(a)));
}

__device__ inline float TriangleQualityGpu(const GVertex &a, const GVertex &b, const GVertex &c) {
  const float ab2 = Dist2(a, b), ac2 = Dist2(a, c), bc2 = Dist2(b, c);
  const float denom = ab2 + ac2 + bc2;
  const float3 n = TriangleCross(a, b, c);
  const float area2 = sqrtf(Dot3(n, n));
  if (!(denom > 0.0f) || !(area2 > 0.0f)) return 0.0f;
  return fminf(1.0f, 3.4641016151377544f * area2 / denom);
}

__device__ inline float3 AnalyticNormalGpu(const GPatch &p, const GVertex &a,
                                                   const GVertex &b, const GVertex &c) {
  if (p.type == uint8_t(PatchType::Plane)) return make_float3(p.ax,p.ay,p.az);
  if (p.type == uint8_t(PatchType::Cylinder)) {
    const float x=(a.x+b.x+c.x)/3.0f, y=(a.y+b.y+c.y)/3.0f, z=(a.z+b.z+c.z)/3.0f;
    const float ox=x-p.ox, oy=y-p.oy, oz=z-p.oz;
    const float h=ox*p.ax+oy*p.ay+oz*p.az;
    float rx=ox-p.ax*h, ry=oy-p.ay*h, rz=oz-p.az*h;
    const float n2=rx*rx+ry*ry+rz*rz;
    if (!(n2>1e-20f)) return make_float3(0,0,0);
    const float inv=rsqrtf(n2); return make_float3(rx*inv,ry*inv,rz*inv);
  }
  return make_float3(0,0,0);
}

__device__ inline bool PatchOrientationSafe(const GPatch *patches, uint32_t patchCount,
                                             uint32_t patchId, const GVertex &a,
                                             const GVertex &b, const GVertex &c) {
  if (!patches || patchId>=patchCount) return true;
  const GPatch p=patches[patchId];
  const float3 an=AnalyticNormalGpu(p,a,b,c);
  const float an2=Dot3(an,an);
  if (!(an2>1e-20f)) return true;
  return Dot3(TriangleCross(a,b,c),an)*p.orientationSign > 0.0f;
}

__device__ inline bool FaceHasDirectedEdge(const GFace &f, uint32_t a, uint32_t b) {
  return (f.v0 == a && f.v1 == b) || (f.v1 == a && f.v2 == b) ||
         (f.v2 == a && f.v0 == b);
}

__device__ inline uint32_t ThirdVertex(const GFace &f, uint32_t a, uint32_t b) {
  if (f.v0 != a && f.v0 != b) return f.v0;
  if (f.v1 != a && f.v1 != b) return f.v1;
  if (f.v2 != a && f.v2 != b) return f.v2;
  return kInvalid;
}

__device__ inline bool EdgeMatches(const GEdge &e, uint32_t a, uint32_t b) {
  return (e.v0 == a && e.v1 == b) || (e.v0 == b && e.v1 == a);
}

__device__ inline uint32_t FindFaceEdge(const GFace &f, const GEdge *edges, uint32_t a,
                                        uint32_t b) {
  const uint32_t ids[3] = {f.e0, f.e1, f.e2};
  for (int i = 0; i < 3; ++i) {
    if (ids[i] != kInvalid && EdgeMatches(edges[ids[i]], a, b)) return ids[i];
  }
  return kInvalid;
}

__device__ inline void ReplaceIncidentFace(GEdge &e, int32_t oldFace, int32_t newFace) {
  if (e.f0 == oldFace)
    e.f0 = newFace;
  else if (e.f1 == oldFace)
    e.f1 = newFace;
}

__device__ inline uint64_t DeviceEdgeKey(uint32_t a, uint32_t b) {
  if (a > b) { const uint32_t t = a; a = b; b = t; }
  return ((uint64_t(a) << 32) | uint64_t(b)) + 1ull;
}

__device__ inline uint32_t HashSlot(uint64_t key, uint32_t mask) {
  key ^= key >> 33;
  key *= 0xff51afd7ed558ccdULL;
  key ^= key >> 33;
  return uint32_t(key) & mask;
}

__global__ void BuildEdgeHash(const GEdge *edges, uint32_t edgeCount,
                              unsigned long long *keys, uint32_t hashMask) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= edgeCount || !edges[id].alive) return;
  const uint64_t key = DeviceEdgeKey(edges[id].v0, edges[id].v1);
  uint32_t slot = HashSlot(key, hashMask);
  for (uint32_t probe = 0; probe <= hashMask; ++probe) {
    const unsigned long long old = atomicCAS(keys + slot, 0ull, (unsigned long long)key);
    if (old == 0ull || old == key) return;
    slot = (slot + 1u) & hashMask;
  }
}

__device__ inline bool EdgeHashContains(const unsigned long long *keys,
                                        uint32_t hashMask, uint32_t a, uint32_t b) {
  const uint64_t key = DeviceEdgeKey(a, b);
  uint32_t slot = HashSlot(key, hashMask);
  for (uint32_t probe = 0; probe <= hashMask; ++probe) {
    const unsigned long long found = keys[slot];
    if (found == 0ull) return false;
    if (found == key) return true;
    slot = (slot + 1u) & hashMask;
  }
  return false;
}

__global__ void CountVertexEdges(const GEdge *edges, uint32_t edgeCount,
                                 uint32_t vertexCount, uint32_t *counts) {
  const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=edgeCount) return;
  const GEdge e=edges[id];
  if(!e.alive || e.v0>=vertexCount || e.v1>=vertexCount) return;
  atomicAdd(counts+e.v0,1u); atomicAdd(counts+e.v1,1u);
}

__global__ void FillVertexEdges(const GEdge *edges, uint32_t edgeCount,
                                uint32_t vertexCount, const uint32_t *offsets,
                                uint32_t *cursor, uint32_t *edgeIds) {
  const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=edgeCount) return;
  const GEdge e=edges[id];
  if(!e.alive || e.v0>=vertexCount || e.v1>=vertexCount) return;
  const uint32_t a=offsets[e.v0]+atomicAdd(cursor+e.v0,1u);
  const uint32_t b=offsets[e.v1]+atomicAdd(cursor+e.v1,1u);
  edgeIds[a]=id; edgeIds[b]=id;
}

__device__ inline GVertex LerpSplitVertex(const GVertex &a, const GVertex &b, float t) {
  GVertex out = a;
  out.x = a.x + (b.x - a.x) * t;
  out.y = a.y + (b.y - a.y) * t;
  out.z = a.z + (b.z - a.z) * t;
  out.targetLength = a.targetLength + (b.targetLength - a.targetLength) * t;
  return out;
}

__device__ inline GVertex ProjectCandidateVertex(GVertex v, const GPatch *patches,
                                                       uint32_t patchCount) {
  return ProjectAnalyticVertex(v, patches, patchCount, nullptr, nullptr);
}

__device__ inline bool SameHemisphere(const GVertex &a, const GVertex &b, const GVertex &c,
                                      const GVertex &x, const GVertex &y, const GVertex &z) {
  return Dot3(TriangleCross(a,b,c), TriangleCross(x,y,z)) > 0.0f;
}

__device__ inline bool SegmentedSplitQualitySafe(const GVertex &a, const GVertex &b,
                                                  const GVertex &c, const GVertex &d,
                                                  uint32_t segments, uint32_t patchId,
                                                  const GPatch *patches, uint32_t patchCount) {
  constexpr float qMin = 1.0e-6f;
  for (uint32_t i=0;i<segments;++i) {
    const float t0=float(i)/float(segments), t1=float(i+1u)/float(segments);
    GVertex p0=LerpSplitVertex(a,b,t0); p0.patchId=patchId;
    GVertex p1=LerpSplitVertex(a,b,t1); p1.patchId=patchId;
    p0=ProjectCandidateVertex(p0,patches,patchCount);
    p1=ProjectCandidateVertex(p1,patches,patchCount);
    if (!(TriangleQualityGpu(p0,p1,c)>qMin) || !(TriangleQualityGpu(p1,p0,d)>qMin) ||
        !SameHemisphere(a,b,c,p0,p1,c) || !SameHemisphere(b,a,d,p1,p0,d) ||
        !PatchOrientationSafe(patches,patchCount,patchId,p0,p1,c) ||
        !PatchOrientationSafe(patches,patchCount,patchId,p1,p0,d)) return false;
  }
  return true;
}

__device__ inline void TriangleClaimFaceEdges(const GFace &f, unsigned long long key,
                                                      unsigned long long *winner) {
  atomicMax(winner + f.e0, key); atomicMax(winner + f.e1, key); atomicMax(winner + f.e2, key);
}
__device__ inline bool TriangleOwnFaceEdges(const GFace &f, unsigned long long key,
                                            const unsigned long long *winner) {
  return winner[f.e0] == key && winner[f.e1] == key && winner[f.e2] == key;
}
__device__ inline void TriangleBlockFaceEdges(const GFace &f, uint8_t *blocked) {
  blocked[f.e0] = 1; blocked[f.e1] = 1; blocked[f.e2] = 1;
}
__device__ inline bool TriangleAnyFaceEdgeBlocked(const GFace &f, const uint8_t *blocked) {
  return blocked[f.e0] || blocked[f.e1] || blocked[f.e2];
}

__device__ inline int32_t OtherIncidentFace(const GEdge &e, uint32_t faceId) {
  if (e.f0 == int32_t(faceId)) return e.f1;
  if (e.f1 == int32_t(faceId)) return e.f0;
  return -1;
}

__device__ inline GVertex MidVertex(const GVertex &a, const GVertex &b, uint32_t patchId) {
  return {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f, (a.z + b.z) * 0.5f,
          (a.targetLength + b.targetLength) * 0.5f, patchId, 1u,
          kSurfaceConstraint, 1u, 0u};
}

__device__ inline bool TriangleRefineQualitySafe(const GVertex &a,const GVertex &b,
 const GVertex &c,const GVertex &d0,const GVertex &d1,const GVertex &d2,uint32_t patchId,
 const GPatch *patches,uint32_t patchCount) {
  GVertex m0=ProjectCandidateVertex(MidVertex(a,b,patchId),patches,patchCount);
  GVertex m1=ProjectCandidateVertex(MidVertex(b,c,patchId),patches,patchCount);
  GVertex m2=ProjectCandidateVertex(MidVertex(c,a,patchId),patches,patchCount);
  constexpr float q=1e-6f;
  const bool quality=TriangleQualityGpu(a,m0,m2)>q && TriangleQualityGpu(m0,b,m1)>q &&
    TriangleQualityGpu(m2,m1,c)>q && TriangleQualityGpu(m0,m1,m2)>q &&
    TriangleQualityGpu(b,m0,d0)>q && TriangleQualityGpu(m0,a,d0)>q &&
    TriangleQualityGpu(c,m1,d1)>q && TriangleQualityGpu(m1,b,d1)>q &&
    TriangleQualityGpu(a,m2,d2)>q && TriangleQualityGpu(m2,c,d2)>q;
  if (!quality) return false;
  return SameHemisphere(a,b,c,a,m0,m2) && SameHemisphere(a,b,c,m0,b,m1) &&
    SameHemisphere(a,b,c,m2,m1,c) && SameHemisphere(a,b,c,m0,m1,m2) &&
    SameHemisphere(b,a,d0,b,m0,d0) && SameHemisphere(b,a,d0,m0,a,d0) &&
    SameHemisphere(c,b,d1,c,m1,d1) && SameHemisphere(c,b,d1,m1,b,d1) &&
    SameHemisphere(a,c,d2,a,m2,d2) && SameHemisphere(a,c,d2,m2,c,d2) &&
    PatchOrientationSafe(patches,patchCount,patchId,a,m0,m2) &&
    PatchOrientationSafe(patches,patchCount,patchId,m0,b,m1) &&
    PatchOrientationSafe(patches,patchCount,patchId,m2,m1,c) &&
    PatchOrientationSafe(patches,patchCount,patchId,m0,m1,m2) &&
    PatchOrientationSafe(patches,patchCount,patchId,b,m0,d0) &&
    PatchOrientationSafe(patches,patchCount,patchId,m0,a,d0) &&
    PatchOrientationSafe(patches,patchCount,patchId,c,m1,d1) &&
    PatchOrientationSafe(patches,patchCount,patchId,m1,b,d1) &&
    PatchOrientationSafe(patches,patchCount,patchId,a,m2,d2) &&
    PatchOrientationSafe(patches,patchCount,patchId,m2,c,d2);
}

__device__ inline int TwoEdgeBestLayout(const GVertex&a,const GVertex&b,const GVertex&c,
 const GVertex&d0,const GVertex&d1,uint32_t patchId,const GPatch*patches,uint32_t patchCount) {
  constexpr float q=1e-6f;
  GVertex m0=ProjectCandidateVertex(MidVertex(a,b,patchId),patches,patchCount);
  GVertex m1=ProjectCandidateVertex(MidVertex(b,c,patchId),patches,patchCount);
  float qn0=fminf(TriangleQualityGpu(b,m0,d0),TriangleQualityGpu(m0,a,d0));
  float qn1=fminf(TriangleQualityGpu(c,m1,d1),TriangleQualityGpu(m1,b,d1));
  if (!(qn0>q) || !(qn1>q) || !SameHemisphere(b,a,d0,b,m0,d0) ||
      !SameHemisphere(b,a,d0,m0,a,d0) || !SameHemisphere(c,b,d1,c,m1,d1) ||
      !SameHemisphere(c,b,d1,m1,b,d1) ||
      !PatchOrientationSafe(patches,patchCount,patchId,b,m0,d0) ||
      !PatchOrientationSafe(patches,patchCount,patchId,m0,a,d0) ||
      !PatchOrientationSafe(patches,patchCount,patchId,c,m1,d1) ||
      !PatchOrientationSafe(patches,patchCount,patchId,m1,b,d1)) return -1;
  float q0=fminf(qn0,qn1); q0=fminf(q0,TriangleQualityGpu(a,m0,m1));
  q0=fminf(q0,TriangleQualityGpu(m0,b,m1)); q0=fminf(q0,TriangleQualityGpu(a,m1,c));
  bool o0=SameHemisphere(a,b,c,a,m0,m1)&&SameHemisphere(a,b,c,m0,b,m1)&&SameHemisphere(a,b,c,a,m1,c)&&
    PatchOrientationSafe(patches,patchCount,patchId,a,m0,m1)&&
    PatchOrientationSafe(patches,patchCount,patchId,m0,b,m1)&&
    PatchOrientationSafe(patches,patchCount,patchId,a,m1,c);
  float q1=fminf(qn0,qn1); q1=fminf(q1,TriangleQualityGpu(a,m0,c));
  q1=fminf(q1,TriangleQualityGpu(m0,m1,c)); q1=fminf(q1,TriangleQualityGpu(m0,b,m1));
  bool o1=SameHemisphere(a,b,c,a,m0,c)&&SameHemisphere(a,b,c,m0,m1,c)&&SameHemisphere(a,b,c,m0,b,m1)&&
    PatchOrientationSafe(patches,patchCount,patchId,a,m0,c)&&
    PatchOrientationSafe(patches,patchCount,patchId,m0,m1,c)&&
    PatchOrientationSafe(patches,patchCount,patchId,m0,b,m1);
  if (!o0) q0=0; if (!o1) q1=0;
  const float best=fmaxf(q0,q1); if (!(best>q)) return -1;
  return q1>q0?1:0;
}

__global__ void GenerateTriangleRefineCandidates(
    const GVertex *vertices, const GEdge *edges, const GFace *faces,
    uint32_t faceCount, float refineRatio, const GPatch *patches, uint32_t patchCount,
    TriangleRefineCandidate *candidates, uint32_t candidateCapacity,
    uint32_t *candidateCount, uint32_t *semanticRejected,
    uint32_t *qualityRejected, uint32_t *rejectCounters, uint32_t *editableHistogram) {
  const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;
  if (id>=faceCount) return;
  const GFace f=faces[id];
  if (!f.alive) return;

  const uint32_t rawE[3]={f.e0,f.e1,f.e2};
  const uint32_t rawV[3]={f.v0,f.v1,f.v2};
  bool editable[3]={false,false,false};
  bool refine[3]={false,false,false};
  float ratio[3]={0,0,0};
  uint32_t editableCount=0, refineCount=0;
  for (int i=0;i<3;++i) {
    const GEdge e=edges[rawE[i]];
    editable[i]=e.alive && e.f0>=0 && e.f1>=0 &&
        !(e.flags & uint8_t(EdgeProtected|EdgeMeshBoundary|EdgePatchBoundary));
    editableCount += editable[i] ? 1u : 0u;
    if (!editable[i]) continue;
    const GVertex a=vertices[rawV[i]], b=vertices[rawV[(i+1)%3]];
    const float h=0.5f*(a.targetLength+b.targetLength);
    if (!(h>0.0f)) continue;
    ratio[i]=sqrtf(Dist2(a,b))/h;
    refine[i]=ratio[i]>refineRatio;
    refineCount += refine[i] ? 1u : 0u;
  }
  if (editableHistogram) atomicAdd(editableHistogram+editableCount,1u);
  if (editableCount==0u) { if(rejectCounters) atomicAdd(rejectCounters+0,1u); return; }
  if (refineCount==0u) { if(rejectCounters) atomicAdd(rejectCounters+3,1u); return; }

  if (refineCount==1u) {
    uint32_t a,b,c,eAB,eBC,eCA; float priority;
    if (refine[0]) { a=f.v0;b=f.v1;c=f.v2;eAB=f.e0;eBC=f.e1;eCA=f.e2;priority=ratio[0]; }
    else if (refine[1]) { a=f.v1;b=f.v2;c=f.v0;eAB=f.e1;eBC=f.e2;eCA=f.e0;priority=ratio[1]; }
    else { a=f.v2;b=f.v0;c=f.v1;eAB=f.e2;eBC=f.e0;eCA=f.e1;priority=ratio[2]; }
    const int32_t ni0=OtherIncidentFace(edges[eAB],id);
    if (ni0<0 || uint32_t(ni0)>=faceCount || !faces[ni0].alive) { if(rejectCounters) atomicAdd(rejectCounters+1,1u); return; }
    const uint32_t n0=uint32_t(ni0);
    if (faces[n0].patchId!=f.patchId) { if(semanticRejected) atomicAdd(semanticRejected,1u); return; }
    if (!FaceHasDirectedEdge(faces[n0],b,a)) { if(rejectCounters) atomicAdd(rejectCounters+2,1u); return; }
    const uint32_t d0=ThirdVertex(faces[n0],a,b);
    const uint32_t r0=d0==kInvalid ? kInvalid : FindFaceEdge(faces[n0],edges,a,d0);
    if (d0==kInvalid || r0==kInvalid) { if(rejectCounters) atomicAdd(rejectCounters+1,1u); return; }
    if (!SegmentedSplitQualitySafe(vertices[a],vertices[b],vertices[c],vertices[d0],2u,f.patchId,patches,patchCount)) {
      if(qualityRejected) atomicAdd(qualityRejected,1u); return;
    }
    const uint32_t slot=atomicAdd(candidateCount,1u); if(slot>=candidateCapacity) return;
    candidates[slot]={id,f.generation,a,b,c,eAB,eBC,eCA,n0,kInvalid,kInvalid,d0,kInvalid,kInvalid,
                      faces[n0].generation,0u,0u,r0,kInvalid,kInvalid,eBC,
                      priority,0x1u,0u,0u,0u};
    return;
  }

  if (refineCount==2u) {
    uint32_t a,b,c,eAB,eBC,eCA; float priority;
    if (!refine[2]) { a=f.v0;b=f.v1;c=f.v2;eAB=f.e0;eBC=f.e1;eCA=f.e2;priority=fmaxf(ratio[0],ratio[1]); }
    else if (!refine[0]) { a=f.v1;b=f.v2;c=f.v0;eAB=f.e1;eBC=f.e2;eCA=f.e0;priority=fmaxf(ratio[1],ratio[2]); }
    else { a=f.v2;b=f.v0;c=f.v1;eAB=f.e2;eBC=f.e0;eCA=f.e1;priority=fmaxf(ratio[2],ratio[0]); }
    const int32_t ni0=OtherIncidentFace(edges[eAB],id), ni1=OtherIncidentFace(edges[eBC],id);
    if (ni0<0 || ni1<0) { if(rejectCounters) atomicAdd(rejectCounters+1,1u); return; }
    const uint32_t n0=uint32_t(ni0), n1=uint32_t(ni1);
    if (n0>=faceCount || n1>=faceCount || n0==n1 || !faces[n0].alive || !faces[n1].alive) {
      if(rejectCounters) atomicAdd(rejectCounters+1,1u); return;
    }
    if (faces[n0].patchId!=f.patchId || faces[n1].patchId!=f.patchId) { if(semanticRejected) atomicAdd(semanticRejected,1u); return; }
    if (!FaceHasDirectedEdge(faces[n0],b,a) || !FaceHasDirectedEdge(faces[n1],c,b)) {
      if(rejectCounters) atomicAdd(rejectCounters+2,1u); return;
    }
    const uint32_t d0=ThirdVertex(faces[n0],a,b), d1=ThirdVertex(faces[n1],b,c);
    if (d0==kInvalid || d1==kInvalid) { if(rejectCounters) atomicAdd(rejectCounters+1,1u); return; }
    const uint32_t r0=FindFaceEdge(faces[n0],edges,a,d0), r1=FindFaceEdge(faces[n1],edges,b,d1);
    if (r0==kInvalid || r1==kInvalid) { if(rejectCounters) atomicAdd(rejectCounters+1,1u); return; }
    const int layout=TwoEdgeBestLayout(vertices[a],vertices[b],vertices[c],vertices[d0],vertices[d1],f.patchId,patches,patchCount);
    if (layout<0) { if(qualityRejected) atomicAdd(qualityRejected,1u); return; }
    const uint32_t slot=atomicAdd(candidateCount,1u); if(slot>=candidateCapacity) return;
    candidates[slot]={id,f.generation,a,b,c,eAB,eBC,eCA,n0,n1,kInvalid,d0,d1,kInvalid,
                      faces[n0].generation,faces[n1].generation,0u,
                      r0,r1,kInvalid,(layout==0 ? eCA : kInvalid),
                      priority,0x3u,uint8_t(layout),0u,0u};
    return;
  }

  const int32_t ni0=OtherIncidentFace(edges[f.e0],id), ni1=OtherIncidentFace(edges[f.e1],id), ni2=OtherIncidentFace(edges[f.e2],id);
  if (ni0<0 || ni1<0 || ni2<0) { if(rejectCounters) atomicAdd(rejectCounters+1,1u); return; }
  const uint32_t n0=uint32_t(ni0),n1=uint32_t(ni1),n2=uint32_t(ni2);
  if (n0>=faceCount || n1>=faceCount || n2>=faceCount || n0==n1 || n1==n2 || n0==n2) { if(rejectCounters) atomicAdd(rejectCounters+1,1u); return; }
  const GFace nf0=faces[n0],nf1=faces[n1],nf2=faces[n2];
  if (!nf0.alive || !nf1.alive || !nf2.alive) return;
  if (nf0.patchId!=f.patchId || nf1.patchId!=f.patchId || nf2.patchId!=f.patchId) { if(semanticRejected) atomicAdd(semanticRejected,1u); return; }
  const uint32_t a=f.v0,b=f.v1,c=f.v2;
  if (!FaceHasDirectedEdge(nf0,b,a) || !FaceHasDirectedEdge(nf1,c,b) || !FaceHasDirectedEdge(nf2,a,c)) { if(rejectCounters) atomicAdd(rejectCounters+2,1u); return; }
  const uint32_t d0=ThirdVertex(nf0,a,b),d1=ThirdVertex(nf1,b,c),d2=ThirdVertex(nf2,c,a);
  if (d0==kInvalid || d1==kInvalid || d2==kInvalid) return;
  const uint32_t r0=FindFaceEdge(nf0,edges,a,d0),r1=FindFaceEdge(nf1,edges,b,d1),r2=FindFaceEdge(nf2,edges,c,d2);
  if (r0==kInvalid || r1==kInvalid || r2==kInvalid) return;
  if (!TriangleRefineQualitySafe(vertices[a],vertices[b],vertices[c],vertices[d0],vertices[d1],vertices[d2],f.patchId,patches,patchCount)) {
    if(qualityRejected) atomicAdd(qualityRejected,1u); return;
  }
  const uint32_t slot=atomicAdd(candidateCount,1u); if(slot>=candidateCapacity) return;
  const float priority=fmaxf(ratio[0],fmaxf(ratio[1],ratio[2]));
  candidates[slot]={id,f.generation,a,b,c,f.e0,f.e1,f.e2,n0,n1,n2,d0,d1,d2,
                    nf0.generation,nf1.generation,nf2.generation,r0,r1,r2,kInvalid,
                    priority,0x7u,0u,0u,0u};
}

__global__ void SumAcceptedTriangleCost(const TriangleRefineCandidate *candidates,
                                        uint32_t candidateCount, uint32_t *newVertexCount) {
  const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;
  if (id>=candidateCount || candidates[id].accepted!=1) return;
  const uint8_t mask=candidates[id].edgeMask;
  const uint32_t cost=((mask&1u)?1u:0u)+((mask&2u)?1u:0u)+((mask&4u)?1u:0u);
  atomicAdd(newVertexCount,cost);
}

__device__ inline unsigned long long TriangleWinnerKey(const TriangleRefineCandidate &c,
                                                        uint32_t id) {
  const uint32_t bits = __float_as_uint(c.priority);
  return (unsigned long long(bits) << 32) | unsigned long long(0xffffffffu - id);
}

__device__ inline void ClaimTriangleMutationEdges(const TriangleRefineCandidate &c,
    unsigned long long key, unsigned long long *edgeWinner) {
  if (c.edgeMask & 0x1u) { atomicMax(edgeWinner+c.e0,key); atomicMax(edgeWinner+c.r0,key); }
  if (c.edgeMask & 0x2u) { atomicMax(edgeWinner+c.e1,key); atomicMax(edgeWinner+c.r1,key); }
  if (c.edgeMask & 0x4u) { atomicMax(edgeWinner+c.e2,key); atomicMax(edgeWinner+c.r2,key); }
  if (c.centerReplace!=kInvalid) atomicMax(edgeWinner+c.centerReplace,key);
}
__device__ inline bool OwnTriangleMutationEdges(const TriangleRefineCandidate &c,
    unsigned long long key, const unsigned long long *edgeWinner) {
  bool ok=true;
  if (c.edgeMask & 0x1u) ok=ok && edgeWinner[c.e0]==key && edgeWinner[c.r0]==key;
  if (c.edgeMask & 0x2u) ok=ok && edgeWinner[c.e1]==key && edgeWinner[c.r1]==key;
  if (c.edgeMask & 0x4u) ok=ok && edgeWinner[c.e2]==key && edgeWinner[c.r2]==key;
  if (c.centerReplace!=kInvalid) ok=ok && edgeWinner[c.centerReplace]==key;
  return ok;
}
__device__ inline void BlockTriangleMutationEdges(const TriangleRefineCandidate &c,
    uint8_t *edgeBlocked) {
  if (c.edgeMask & 0x1u) { edgeBlocked[c.e0]=1; edgeBlocked[c.r0]=1; }
  if (c.edgeMask & 0x2u) { edgeBlocked[c.e1]=1; edgeBlocked[c.r1]=1; }
  if (c.edgeMask & 0x4u) { edgeBlocked[c.e2]=1; edgeBlocked[c.r2]=1; }
  if (c.centerReplace!=kInvalid) edgeBlocked[c.centerReplace]=1;
}
__device__ inline bool AnyTriangleMutationEdgeBlocked(const TriangleRefineCandidate &c,
    const uint8_t *edgeBlocked) {
  bool hit=false;
  if (c.edgeMask & 0x1u) hit=hit || edgeBlocked[c.e0] || edgeBlocked[c.r0];
  if (c.edgeMask & 0x2u) hit=hit || edgeBlocked[c.e1] || edgeBlocked[c.r1];
  if (c.edgeMask & 0x4u) hit=hit || edgeBlocked[c.e2] || edgeBlocked[c.r2];
  if (c.centerReplace!=kInvalid) hit=hit || edgeBlocked[c.centerReplace];
  return hit;
}

__global__ void ClaimTriangleCandidates(
    const TriangleRefineCandidate *candidates, const GFace *faces,
    const uint32_t *active, uint32_t activeCount, unsigned long long *edgeWinner,
    unsigned long long *faceWinner, unsigned long long *vertexWinner) {
  const uint32_t slot=blockIdx.x*blockDim.x+threadIdx.x;
  if (slot>=activeCount) return;
  const uint32_t id=active[slot];
  const TriangleRefineCandidate c=candidates[id];
  const unsigned long long key=TriangleWinnerKey(c,id);
  atomicMax(faceWinner+c.faceId,key);
  if (c.edgeMask & 0x1u) atomicMax(faceWinner+c.n0,key);
  if (c.edgeMask & 0x2u) atomicMax(faceWinner+c.n1,key);
  if (c.edgeMask & 0x4u) atomicMax(faceWinner+c.n2,key);
  ClaimTriangleMutationEdges(c,key,edgeWinner);
}

__global__ void ResolveTriangleCandidates(
    TriangleRefineCandidate *candidates, const GFace *faces, const uint32_t *active,
    uint32_t activeCount, const unsigned long long *edgeWinner,
    const unsigned long long *faceWinner, const unsigned long long *vertexWinner,
    uint8_t *edgeBlocked, uint8_t *faceBlocked, uint8_t *vertexBlocked,
    uint32_t *acceptedCount, uint32_t *roundAccepted) {
  const uint32_t slot=blockIdx.x*blockDim.x+threadIdx.x;
  if (slot>=activeCount) return;
  const uint32_t id=active[slot];
  TriangleRefineCandidate &c=candidates[id];
  const unsigned long long key=TriangleWinnerKey(c,id);
  bool won=faceWinner[c.faceId]==key && OwnTriangleMutationEdges(c,key,edgeWinner);
  if (c.edgeMask & 0x1u) won=won && faceWinner[c.n0]==key;
  if (c.edgeMask & 0x2u) won=won && faceWinner[c.n1]==key;
  if (c.edgeMask & 0x4u) won=won && faceWinner[c.n2]==key;
  if (!won) return;
  c.accepted=1;
  faceBlocked[c.faceId]=1;
  if (c.edgeMask & 0x1u) faceBlocked[c.n0]=1;
  if (c.edgeMask & 0x2u) faceBlocked[c.n1]=1;
  if (c.edgeMask & 0x4u) faceBlocked[c.n2]=1;
  BlockTriangleMutationEdges(c,edgeBlocked);
  atomicAdd(acceptedCount,1u);
  atomicAdd(roundAccepted,1u);
}

__global__ void CompactTriangleCandidates(
    TriangleRefineCandidate *candidates, const GFace *faces, const uint32_t *active,
    uint32_t activeCount, const uint8_t *edgeBlocked, const uint8_t *faceBlocked,
    const uint8_t *vertexBlocked, uint32_t *nextActive, uint32_t *nextCount) {
  const uint32_t slot=blockIdx.x*blockDim.x+threadIdx.x;
  if (slot>=activeCount) return;
  const uint32_t id=active[slot];
  TriangleRefineCandidate &c=candidates[id];
  if (c.accepted!=0) return;
  bool blocked=faceBlocked[c.faceId] || AnyTriangleMutationEdgeBlocked(c,edgeBlocked);
  if (c.edgeMask & 0x1u) blocked=blocked || faceBlocked[c.n0];
  if (c.edgeMask & 0x2u) blocked=blocked || faceBlocked[c.n1];
  if (c.edgeMask & 0x4u) blocked=blocked || faceBlocked[c.n2];
  if (blocked) { c.accepted=2; return; }
  const uint32_t out=atomicAdd(nextCount,1u);
  nextActive[out]=id;
}

__device__ inline bool FaceMatchesCycle(const GFace &f, uint32_t a, uint32_t b, uint32_t c) {
  return (f.v0==a && f.v1==b && f.v2==c) ||
         (f.v0==b && f.v1==c && f.v2==a) ||
         (f.v0==c && f.v1==a && f.v2==b);
}

__global__ void ExecuteTriangleRefineCandidates(
    GVertex *vertices, GEdge *edges, GFace *faces,
    TriangleRefineCandidate *candidates, uint32_t candidateCount,
    uint32_t *vertexCount, uint32_t *edgeCount, uint32_t *faceCount,
    uint32_t *staleRejected, const GPatch *patches, uint32_t patchCount,
    uint32_t *projectionApplied, uint32_t *projectionFailed) {
  const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;
  if (id>=candidateCount || candidates[id].accepted!=1) return;
  const TriangleRefineCandidate c=candidates[id];
  GFace &cf=faces[c.faceId];
  if (!cf.alive || cf.generation!=c.faceGeneration ||
      !FaceMatchesCycle(cf,c.v0,c.v1,c.v2)) {
    atomicAdd(staleRejected,1u); return;
  }

  if (c.edgeMask==0x1u) {
    // Normalized 1-edge green template: AB editable; BC and CA protected.
    if (c.n0==kInvalid || c.d0==kInvalid) { atomicAdd(staleRejected,1u); return; }
    GFace &n0=faces[c.n0];
    if (!n0.alive || n0.generation!=c.ng0 || !FaceHasDirectedEdge(n0,c.v1,c.v0)) {
      atomicAdd(staleRejected,1u); return;
    }
    const uint32_t eB_D0=FindFaceEdge(n0,edges,c.v1,c.d0);
    const uint32_t eA_D0=FindFaceEdge(n0,edges,c.v0,c.d0);
    if (eB_D0==kInvalid || eA_D0==kInvalid) { atomicAdd(staleRejected,1u); return; }
    const uint32_t vb=atomicAdd(vertexCount,1u);
    const uint32_t eb=atomicAdd(edgeCount,3u);
    const uint32_t fb=atomicAdd(faceCount,2u);
    const uint32_t m=vb, half=eb, centerSpoke=eb+1u, neighborSpoke=eb+2u;
    const uint32_t centerExtra=fb, neighborExtra=fb+1u;
    const uint32_t patch=cf.patchId;
    vertices[m]=ProjectAnalyticVertex(MidVertex(vertices[c.v0],vertices[c.v1],patch), patches, patchCount, projectionApplied, projectionFailed);
    GEdge &eAB=edges[c.e0]; const uint8_t flagsAB=eAB.flags;
    eAB={c.v0,m,int32_t(c.faceId),int32_t(neighborExtra),eAB.generation+1u,flagsAB,1u,0u};
    edges[half]={m,c.v1,int32_t(centerExtra),int32_t(c.n0),1u,0u,1u,0u};
    edges[centerSpoke]={m,c.v2,int32_t(c.faceId),int32_t(centerExtra),1u,0u,1u,0u};
    edges[neighborSpoke]={m,c.d0,int32_t(c.n0),int32_t(neighborExtra),1u,0u,1u,0u};
    ReplaceIncidentFace(edges[c.e1],int32_t(c.faceId),int32_t(centerExtra));
    ReplaceIncidentFace(edges[eA_D0],int32_t(c.n0),int32_t(neighborExtra));
    const uint32_t cgen=cf.generation, ngen=n0.generation, np=n0.patchId;
    cf={c.v0,m,c.v2,c.e0,centerSpoke,c.e2,patch,cgen+1u,1u,{0,0,0}};
    faces[centerExtra]={m,c.v1,c.v2,half,c.e1,centerSpoke,patch,1u,1u,{0,0,0}};
    n0={c.v1,m,c.d0,half,neighborSpoke,eB_D0,np,ngen+1u,1u,{0,0,0}};
    faces[neighborExtra]={m,c.v0,c.d0,c.e0,eA_D0,neighborSpoke,np,1u,1u,{0,0,0}};
    return;
  }

  if (c.edgeMask==0x3u) {
    // Normalized 2-edge green template: AB, BC editable; CA protected.
    if (c.n0==kInvalid || c.n1==kInvalid || c.d0==kInvalid || c.d1==kInvalid) {
      atomicAdd(staleRejected,1u); return;
    }
    GFace &n0=faces[c.n0]; GFace &n1=faces[c.n1];
    if (!n0.alive || !n1.alive || n0.generation!=c.ng0 || n1.generation!=c.ng1 ||
        !FaceHasDirectedEdge(n0,c.v1,c.v0) || !FaceHasDirectedEdge(n1,c.v2,c.v1)) {
      atomicAdd(staleRejected,1u); return;
    }
    const uint32_t eB_D0=FindFaceEdge(n0,edges,c.v1,c.d0);
    const uint32_t eA_D0=FindFaceEdge(n0,edges,c.v0,c.d0);
    const uint32_t eC_D1=FindFaceEdge(n1,edges,c.v2,c.d1);
    const uint32_t eB_D1=FindFaceEdge(n1,edges,c.v1,c.d1);
    if (eB_D0==kInvalid || eA_D0==kInvalid || eC_D1==kInvalid || eB_D1==kInvalid) {
      atomicAdd(staleRejected,1u); return;
    }

    const uint32_t vb=atomicAdd(vertexCount,2u);
    const uint32_t eb=atomicAdd(edgeCount,6u);
    const uint32_t fb=atomicAdd(faceCount,4u);
    const uint32_t m0=vb, m1=vb+1u;
    const uint32_t half0=eb, half1=eb+1u;
    const uint32_t spoke0=eb+2u, spoke1=eb+3u;
    const uint32_t inner0=eb+4u, inner1=eb+5u;
    const uint32_t c0=c.faceId, c1=fb, c2=fb+1u;
    const uint32_t n0Extra=fb+2u, n1Extra=fb+3u;
    const uint32_t patch=cf.patchId;
    vertices[m0]=ProjectAnalyticVertex(MidVertex(vertices[c.v0],vertices[c.v1],patch), patches, patchCount, projectionApplied, projectionFailed);
    vertices[m1]=ProjectAnalyticVertex(MidVertex(vertices[c.v1],vertices[c.v2],patch), patches, patchCount, projectionApplied, projectionFailed);

    // Split AB and BC; preserve flags on reused original edges.
    GEdge &eAB=edges[c.e0];
    GEdge &eBC=edges[c.e1];
    const uint8_t flagsAB=eAB.flags, flagsBC=eBC.flags;
    eAB={c.v0,m0,int32_t(c0),int32_t(n0Extra),eAB.generation+1u,flagsAB,1u,0u};
    const uint32_t centralOnBC = c.layout==0 ? c1 : c2;
    eBC={c.v1,m1,int32_t(centralOnBC),int32_t(n1Extra),eBC.generation+1u,flagsBC,1u,0u};
    const uint32_t centralHalf0 = c.layout==0 ? c1 : c2;
    const uint32_t centralHalf1 = c.layout==0 ? c2 : c1;
    edges[half0]={m0,c.v1,int32_t(centralHalf0),int32_t(c.n0),1u,0u,1u,0u};
    edges[half1]={m1,c.v2,int32_t(centralHalf1),int32_t(c.n1),1u,0u,1u,0u};
    edges[spoke0]={m0,c.d0,int32_t(c.n0),int32_t(n0Extra),1u,0u,1u,0u};
    edges[spoke1]={m1,c.d1,int32_t(c.n1),int32_t(n1Extra),1u,0u,1u,0u};

    if (c.layout==0) {
      // (A,M0,M1), (M0,B,M1), (A,M1,C)
      edges[inner0]={m0,m1,int32_t(c0),int32_t(c1),1u,0u,1u,0u};
      edges[inner1]={c.v0,m1,int32_t(c0),int32_t(c2),1u,0u,1u,0u};
      ReplaceIncidentFace(edges[c.e2],int32_t(c.faceId),int32_t(c2));
      const uint32_t oldGen=cf.generation;
      cf={c.v0,m0,m1,c.e0,inner0,inner1,patch,oldGen+1u,1u,{0,0,0}};
      faces[c1]={m0,c.v1,m1,half0,c.e1,inner0,patch,1u,1u,{0,0,0}};
      faces[c2]={c.v0,m1,c.v2,inner1,half1,c.e2,patch,1u,1u,{0,0,0}};
    } else {
      // (A,M0,C), (M0,M1,C), (M0,B,M1)
      edges[inner0]={m0,m1,int32_t(c1),int32_t(c2),1u,0u,1u,0u};
      edges[inner1]={m0,c.v2,int32_t(c0),int32_t(c1),1u,0u,1u,0u};
      const uint32_t oldGen=cf.generation;
      cf={c.v0,m0,c.v2,c.e0,inner1,c.e2,patch,oldGen+1u,1u,{0,0,0}};
      faces[c1]={m0,m1,c.v2,inner0,half1,inner1,patch,1u,1u,{0,0,0}};
      faces[c2]={m0,c.v1,m1,half0,c.e1,inner0,patch,1u,1u,{0,0,0}};
    }

    ReplaceIncidentFace(edges[eA_D0],int32_t(c.n0),int32_t(n0Extra));
    ReplaceIncidentFace(edges[eB_D1],int32_t(c.n1),int32_t(n1Extra));
    const uint32_t n0Gen=n0.generation, n1Gen=n1.generation;
    n0={c.v1,m0,c.d0,half0,spoke0,eB_D0,n0.patchId,n0Gen+1u,1u,{0,0,0}};
    faces[n0Extra]={m0,c.v0,c.d0,c.e0,eA_D0,spoke0,n0.patchId,1u,1u,{0,0,0}};
    n1={c.v2,m1,c.d1,half1,spoke1,eC_D1,n1.patchId,n1Gen+1u,1u,{0,0,0}};
    faces[n1Extra]={m1,c.v1,c.d1,c.e1,eB_D1,spoke1,n1.patchId,1u,1u,{0,0,0}};
    return;
  }

  // Existing 3-edge red refinement.
  if (c.edgeMask!=0x7u || c.n0==kInvalid || c.n1==kInvalid || c.n2==kInvalid) {
    atomicAdd(staleRejected,1u); return;
  }
  const uint32_t ns[3]={c.n0,c.n1,c.n2};
  const uint32_t ng[3]={c.ng0,c.ng1,c.ng2};
  const uint32_t ds[3]={c.d0,c.d1,c.d2};
  const uint32_t us[3]={c.v0,c.v1,c.v2};
  const uint32_t vs[3]={c.v1,c.v2,c.v0};
  const uint32_t ces[3]={c.e0,c.e1,c.e2};
  for (int i=0;i<3;++i) {
    if (!faces[ns[i]].alive || faces[ns[i]].generation!=ng[i] ||
        !FaceHasDirectedEdge(faces[ns[i]],vs[i],us[i])) {
      atomicAdd(staleRejected,1u); return;
    }
  }
  uint32_t eVD[3],eUD[3];
  for (int i=0;i<3;++i) {
    eVD[i]=FindFaceEdge(faces[ns[i]],edges,vs[i],ds[i]);
    eUD[i]=FindFaceEdge(faces[ns[i]],edges,us[i],ds[i]);
    if (eVD[i]==kInvalid || eUD[i]==kInvalid) { atomicAdd(staleRejected,1u); return; }
  }
  const uint32_t vb=atomicAdd(vertexCount,3u);
  const uint32_t eb=atomicAdd(edgeCount,9u);
  const uint32_t fb=atomicAdd(faceCount,6u);
  const uint32_t m[3]={vb,vb+1u,vb+2u};
  const uint32_t half[3]={eb,eb+1u,eb+2u};
  const uint32_t inner[3]={eb+3u,eb+4u,eb+5u};
  const uint32_t spoke[3]={eb+6u,eb+7u,eb+8u};
  const uint32_t center[4]={c.faceId,fb,fb+1u,fb+2u};
  const uint32_t nExtra[3]={fb+3u,fb+4u,fb+5u};
  const uint32_t centerReuse[3]={center[0],center[1],center[2]};
  const uint32_t centerHalf[3]={center[1],center[2],center[0]};
  vertices[m[0]]=ProjectAnalyticVertex(MidVertex(vertices[c.v0],vertices[c.v1],cf.patchId), patches, patchCount, projectionApplied, projectionFailed);
  vertices[m[1]]=ProjectAnalyticVertex(MidVertex(vertices[c.v1],vertices[c.v2],cf.patchId), patches, patchCount, projectionApplied, projectionFailed);
  vertices[m[2]]=ProjectAnalyticVertex(MidVertex(vertices[c.v2],vertices[c.v0],cf.patchId), patches, patchCount, projectionApplied, projectionFailed);
  for (int i=0;i<3;++i) {
    GEdge &old=edges[ces[i]];
    const uint8_t flags=old.flags;
    old.v0=us[i]; old.v1=m[i]; old.f0=int32_t(centerReuse[i]); old.f1=int32_t(nExtra[i]);
    ++old.generation; old.flags=flags;
    edges[half[i]]={m[i],vs[i],int32_t(centerHalf[i]),int32_t(ns[i]),1u,0u,1u,0u};
    edges[spoke[i]]={m[i],ds[i],int32_t(ns[i]),int32_t(nExtra[i]),1u,0u,1u,0u};
    ReplaceIncidentFace(edges[eUD[i]],int32_t(ns[i]),int32_t(nExtra[i]));
  }
  edges[inner[0]]={m[0],m[1],int32_t(center[1]),int32_t(center[3]),1u,0u,1u,0u};
  edges[inner[1]]={m[1],m[2],int32_t(center[2]),int32_t(center[3]),1u,0u,1u,0u};
  edges[inner[2]]={m[2],m[0],int32_t(center[0]),int32_t(center[3]),1u,0u,1u,0u};
  const uint32_t patch=cf.patchId, oldGen=cf.generation;
  cf={c.v0,m[0],m[2],c.e0,inner[2],half[2],patch,oldGen+1u,1u,{0,0,0}};
  faces[center[1]]={m[0],c.v1,m[1],half[0],c.e1,inner[0],patch,1u,1u,{0,0,0}};
  faces[center[2]]={m[2],m[1],c.v2,inner[1],half[1],c.e2,patch,1u,1u,{0,0,0}};
  faces[center[3]]={m[0],m[1],m[2],inner[0],inner[1],inner[2],patch,1u,1u,{0,0,0}};
  for (int i=0;i<3;++i) {
    GFace &nf=faces[ns[i]];
    const uint32_t np=nf.patchId, ngen=nf.generation;
    nf={vs[i],m[i],ds[i],half[i],spoke[i],eVD[i],np,ngen+1u,1u,{0,0,0}};
    faces[nExtra[i]]={m[i],us[i],ds[i],ces[i],eUD[i],spoke[i],np,1u,1u,{0,0,0}};
  }
}

__global__ void GenerateSplitCandidates(const GVertex *vertices, const GEdge *edges,
                                        const GFace *faces, uint32_t edgeCount,
                                        uint32_t faceCount, float splitRatio,
                                        float maxRatio, const GPatch *patches, uint32_t patchCount,
                                        SplitCandidate *candidates,
                                        uint32_t candidateCapacity, uint32_t *candidateCount,
                                        uint32_t *semanticRejected,
                                        uint32_t *capacityRejected) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= edgeCount) return;
  const GEdge e = edges[id];
  if (!e.alive || e.f0 < 0 || e.f1 < 0) return;
  if (e.flags & uint8_t(EdgeProtected | EdgeMeshBoundary | EdgePatchBoundary)) return;
  if (uint32_t(e.f0) >= faceCount || uint32_t(e.f1) >= faceCount) return;
  const GFace f0 = faces[e.f0], f1 = faces[e.f1];
  if (!f0.alive || !f1.alive) return;
  if (f0.patchId != f1.patchId) {
    if (semanticRejected) atomicAdd(semanticRejected, 1u);
    return;
  }
  if (e.v0 == e.v1 || e.v0 == kInvalid || e.v1 == kInvalid) return;
  const GVertex a = vertices[e.v0], b = vertices[e.v1];
  if (!a.alive || !b.alive) return;
  const float h = 0.5f * (a.targetLength + b.targetLength);
  if (!(h > 0.0f)) return;
  const float limit = splitRatio * h;
  const float len2 = Dist2(a, b);
  if (!(len2 > limit * limit)) return;
  if (isfinite(maxRatio)) {
    const float maxLimit = maxRatio * h;
    if (len2 > maxLimit * maxLimit) return;
  }
  const uint32_t c = ThirdVertex(f0, e.v0, e.v1);
  const uint32_t d = ThirdVertex(f1, e.v0, e.v1);
  if (c == kInvalid || d == kInvalid || c == d) return;
  const float len = sqrtf(len2);
  // Residual cleanup only: coarse refinement belongs to the triangle/cavity stage.
  constexpr uint32_t segmentCount = 2u;
  const GVertex vc = vertices[c], vd = vertices[d];
  if (!SegmentedSplitQualitySafe(a, b, vc, vd, segmentCount, f0.patchId, patches, patchCount)) return;
  const uint32_t slot = atomicAdd(candidateCount, 1u);
  if (slot >= candidateCapacity) {
    if (capacityRejected) atomicAdd(capacityRejected, 1u);
    return;
  }
  candidates[slot] = {id, e.generation, e.v0, e.v1, c, d,
                      uint32_t(e.f0), uint32_t(e.f1), len / h,
                      uint8_t(segmentCount), 0, {0, 0}};
}

__global__ void InitActiveQueue(uint32_t *active, uint32_t count) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < count) active[id] = id;
}

__global__ void SumAcceptedSplitCost(const SplitCandidate *candidates,
                                     uint32_t candidateCount,
                                     uint32_t *newVertexCount) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= candidateCount || candidates[id].accepted != 1) return;
  atomicAdd(newVertexCount, uint32_t(candidates[id].segmentCount - 1u));
}

__device__ inline void ClaimFaceEdges(const GFace &f, uint32_t candidateId,
                                      uint32_t *edgeOwner) {
  atomicMin(edgeOwner + f.e0, candidateId);
  atomicMin(edgeOwner + f.e1, candidateId);
  atomicMin(edgeOwner + f.e2, candidateId);
}

__device__ inline bool OwnFaceEdges(const GFace &f, uint32_t candidateId,
                                    const uint32_t *edgeOwner) {
  return edgeOwner[f.e0] == candidateId && edgeOwner[f.e1] == candidateId &&
         edgeOwner[f.e2] == candidateId;
}

__device__ inline bool AnyFaceEdgeBlocked(const GFace &f, const uint8_t *edgeBlocked) {
  return edgeBlocked[f.e0] || edgeBlocked[f.e1] || edgeBlocked[f.e2];
}

__device__ inline void BlockFaceEdges(const GFace &f, uint8_t *edgeBlocked) {
  edgeBlocked[f.e0] = 1;
  edgeBlocked[f.e1] = 1;
  edgeBlocked[f.e2] = 1;
}

__device__ inline unsigned long long SplitWinnerKey(const SplitCandidate &c, uint32_t id) {
  const uint32_t priorityBits = __float_as_uint(c.priority);
  return (unsigned long long(priorityBits) << 32) | unsigned long long(0xffffffffu - id);
}

__device__ inline void ClaimFaceEdgesPriority(const GFace &f, unsigned long long key,
                                              unsigned long long *edgeWinner) {
  atomicMax(edgeWinner + f.e0, key);
  atomicMax(edgeWinner + f.e1, key);
  atomicMax(edgeWinner + f.e2, key);
}

__device__ inline bool OwnFaceEdgesPriority(const GFace &f, unsigned long long key,
                                            const unsigned long long *edgeWinner) {
  return edgeWinner[f.e0] == key && edgeWinner[f.e1] == key && edgeWinner[f.e2] == key;
}

__global__ void ClaimCandidatesActive(const SplitCandidate *candidates, const GFace *faces,
                                      const uint32_t *active, uint32_t activeCount,
                                      unsigned long long *edgeWinner,
                                      unsigned long long *faceWinner,
                                      unsigned long long *vertexWinner) {
  const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= activeCount) return;
  const uint32_t id = active[slot];
  const SplitCandidate c = candidates[id];
  const unsigned long long key = SplitWinnerKey(c, id);
  atomicMax(faceWinner + c.f0, key);
  atomicMax(faceWinner + c.f1, key);
  ClaimFaceEdgesPriority(faces[c.f0], key, edgeWinner);
  ClaimFaceEdgesPriority(faces[c.f1], key, edgeWinner);
  atomicMax(vertexWinner + c.a, key);
  atomicMax(vertexWinner + c.b, key);
  atomicMax(vertexWinner + c.c, key);
  atomicMax(vertexWinner + c.d, key);
}

__global__ void ResolveCandidatesActive(SplitCandidate *candidates, const GFace *faces,
                                        const uint32_t *active, uint32_t activeCount,
                                        const unsigned long long *edgeWinner,
                                        const unsigned long long *faceWinner,
                                        const unsigned long long *vertexWinner,
                                        uint8_t *edgeBlocked, uint8_t *faceBlocked,
                                        uint8_t *vertexBlocked, uint32_t *acceptedCount,
                                        uint32_t *roundAccepted) {
  const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= activeCount) return;
  const uint32_t id = active[slot];
  SplitCandidate &c = candidates[id];
  const unsigned long long key = SplitWinnerKey(c, id);
  const bool won = faceWinner[c.f0] == key && faceWinner[c.f1] == key &&
                   OwnFaceEdgesPriority(faces[c.f0], key, edgeWinner) &&
                   OwnFaceEdgesPriority(faces[c.f1], key, edgeWinner) &&
                   vertexWinner[c.a] == key && vertexWinner[c.b] == key &&
                   vertexWinner[c.c] == key && vertexWinner[c.d] == key;
  if (!won) return;
  c.accepted = 1;
  faceBlocked[c.f0] = 1;
  faceBlocked[c.f1] = 1;
  BlockFaceEdges(faces[c.f0], edgeBlocked);
  BlockFaceEdges(faces[c.f1], edgeBlocked);
  vertexBlocked[c.a] = 1; vertexBlocked[c.b] = 1;
  vertexBlocked[c.c] = 1; vertexBlocked[c.d] = 1;
  atomicAdd(acceptedCount, 1u);
  atomicAdd(roundAccepted, 1u);
}

__global__ void CompactActiveCandidates(SplitCandidate *candidates, const GFace *faces,
                                        const uint32_t *active, uint32_t activeCount,
                                        const uint8_t *edgeBlocked, const uint8_t *faceBlocked,
                                        const uint8_t *vertexBlocked, uint32_t *nextActive,
                                        uint32_t *nextCount) {
  const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= activeCount) return;
  const uint32_t id = active[slot];
  SplitCandidate &c = candidates[id];
  if (c.accepted != 0) return;
  if (faceBlocked[c.f0] || faceBlocked[c.f1] ||
      AnyFaceEdgeBlocked(faces[c.f0], edgeBlocked) ||
      AnyFaceEdgeBlocked(faces[c.f1], edgeBlocked) ||
      vertexBlocked[c.a] || vertexBlocked[c.b] ||
      vertexBlocked[c.c] || vertexBlocked[c.d]) {
    c.accepted = 2;
    return;
  }
  const uint32_t out = atomicAdd(nextCount, 1u);
  nextActive[out] = id;
}


__global__ void GenerateFlipCandidates(const GVertex *vertices, const GEdge *edges,
                                       const GFace *faces, uint32_t edgeCount,
                                       uint32_t faceCount, const unsigned long long *edgeHash,
                                       uint32_t edgeHashMask, float minQualityGain,
                                       FlipCandidate *candidates,
                                       uint32_t candidateCapacity, uint32_t *candidateCount,
                                       uint32_t *semanticRejected,
                                       uint32_t *capacityRejected) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= edgeCount) return;
  const GEdge e = edges[id];
  if (!e.alive || e.f0 < 0 || e.f1 < 0) return;
  if (e.flags & uint8_t(EdgeProtected | EdgeMeshBoundary | EdgePatchBoundary)) return;
  if (uint32_t(e.f0) >= faceCount || uint32_t(e.f1) >= faceCount) return;
  const GFace f0 = faces[e.f0], f1 = faces[e.f1];
  if (!f0.alive || !f1.alive) return;
  if (f0.patchId != f1.patchId) {
    if (semanticRejected) atomicAdd(semanticRejected, 1u);
    return;
  }
  uint32_t a = e.v0, b = e.v1;
  if (!FaceHasDirectedEdge(f0, a, b)) {
    if (FaceHasDirectedEdge(f0, b, a)) {
      const uint32_t t = a; a = b; b = t;
    } else {
      return;
    }
  }
  if (!FaceHasDirectedEdge(f1, b, a)) return;
  const uint32_t c = ThirdVertex(f0, a, b);
  const uint32_t d = ThirdVertex(f1, a, b);
  if (c == kInvalid || d == kInvalid || c == d) return;
  if (EdgeHashContains(edgeHash, edgeHashMask, c, d)) return;
  const GVertex va = vertices[a], vb = vertices[b], vc = vertices[c], vd = vertices[d];
  if (!va.alive || !vb.alive || !vc.alive || !vd.alive) return;
  const float oldQ = fminf(TriangleQualityGpu(va, vb, vc), TriangleQualityGpu(vb, va, vd));
  const float newQ = fminf(TriangleQualityGpu(vc, vd, vb), TriangleQualityGpu(vd, vc, va));
  if (!(newQ > oldQ + minQualityGain)) return;
  const float3 oldN0 = TriangleCross(va, vb, vc);
  const float3 oldN1 = TriangleCross(vb, va, vd);
  const float3 sumN = make_float3(oldN0.x + oldN1.x, oldN0.y + oldN1.y, oldN0.z + oldN1.z);
  const float3 newN0 = TriangleCross(vc, vd, vb);
  const float3 newN1 = TriangleCross(vd, vc, va);
  if (!(Dot3(newN0, sumN) > 0.0f) || !(Dot3(newN1, sumN) > 0.0f)) return;
  const uint32_t slot = atomicAdd(candidateCount, 1u);
  if (slot >= candidateCapacity) {
    if (capacityRejected) atomicAdd(capacityRejected, 1u);
    return;
  }
  candidates[slot] = {id, e.generation, a, b, c, d, uint32_t(e.f0), uint32_t(e.f1),
                      oldQ, newQ, 0, {0, 0, 0}};
}

__global__ void ClaimFlipCandidatesActive(const FlipCandidate *candidates, const GFace *faces,
                                          const uint32_t *active, uint32_t activeCount,
                                          uint32_t *edgeOwner, uint32_t *faceOwner) {
  const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= activeCount) return;
  const uint32_t id = active[slot];
  const FlipCandidate c = candidates[id];
  atomicMin(faceOwner + c.f0, id);
  atomicMin(faceOwner + c.f1, id);
  ClaimFaceEdges(faces[c.f0], id, edgeOwner);
  ClaimFaceEdges(faces[c.f1], id, edgeOwner);
}


__global__ void ResolveFlipCandidatesActive(FlipCandidate *candidates, const GFace *faces,
                                            const uint32_t *active, uint32_t activeCount,
                                            const uint32_t *edgeOwner,
                                            const uint32_t *faceOwner,
                                            uint8_t *edgeBlocked, uint8_t *faceBlocked,
                                            uint32_t *acceptedCount,
                                            uint32_t *roundAccepted) {
  const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= activeCount) return;
  const uint32_t id = active[slot];
  FlipCandidate &c = candidates[id];
  const bool won = faceOwner[c.f0] == id && faceOwner[c.f1] == id &&
                   OwnFaceEdges(faces[c.f0], id, edgeOwner) &&
                   OwnFaceEdges(faces[c.f1], id, edgeOwner);
  if (!won) return;
  c.accepted = 1;
  faceBlocked[c.f0] = 1;
  faceBlocked[c.f1] = 1;
  BlockFaceEdges(faces[c.f0], edgeBlocked);
  BlockFaceEdges(faces[c.f1], edgeBlocked);
  atomicAdd(acceptedCount, 1u);
  atomicAdd(roundAccepted, 1u);
}

__global__ void CompactFlipCandidates(FlipCandidate *candidates, const GFace *faces,
                                      const uint32_t *active, uint32_t activeCount,
                                      const uint8_t *edgeBlocked,
                                      const uint8_t *faceBlocked,
                                      uint32_t *nextActive, uint32_t *nextCount) {
  const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= activeCount) return;
  const uint32_t id = active[slot];
  FlipCandidate &c = candidates[id];
  if (c.accepted != 0) return;
  if (faceBlocked[c.f0] || faceBlocked[c.f1] ||
      AnyFaceEdgeBlocked(faces[c.f0], edgeBlocked) ||
      AnyFaceEdgeBlocked(faces[c.f1], edgeBlocked)) {
    c.accepted = 2;
    return;
  }
  const uint32_t out = atomicAdd(nextCount, 1u);
  nextActive[out] = id;
}

__global__ void ExecuteFlipCandidates(GVertex *vertices, GEdge *edges, GFace *faces,
                                      FlipCandidate *candidates, uint32_t candidateCount,
                                      uint32_t *staleRejected) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= candidateCount || candidates[id].accepted != 1) return;
  const FlipCandidate c = candidates[id];
  GEdge &src = edges[c.edgeId];
  if (!src.alive || src.generation != c.edgeGeneration || src.f0 < 0 || src.f1 < 0) {
    atomicAdd(staleRejected, 1u);
    return;
  }
  const uint32_t f0Id = uint32_t(src.f0), f1Id = uint32_t(src.f1);
  GFace &f0 = faces[f0Id];
  GFace &f1 = faces[f1Id];
  if (!f0.alive || !f1.alive || f0.patchId != f1.patchId) {
    atomicAdd(staleRejected, 1u);
    return;
  }
  uint32_t a = src.v0, b = src.v1;
  if (!FaceHasDirectedEdge(f0, a, b)) {
    if (FaceHasDirectedEdge(f0, b, a)) {
      const uint32_t t = a; a = b; b = t;
    } else {
      atomicAdd(staleRejected, 1u);
      return;
    }
  }
  if (!FaceHasDirectedEdge(f1, b, a)) {
    atomicAdd(staleRejected, 1u);
    return;
  }
  const uint32_t cc = ThirdVertex(f0, a, b);
  const uint32_t dd = ThirdVertex(f1, a, b);
  if (cc != c.c || dd != c.d) {
    atomicAdd(staleRejected, 1u);
    return;
  }
  const uint32_t eAC = FindFaceEdge(f0, edges, a, cc);
  const uint32_t eBC = FindFaceEdge(f0, edges, b, cc);
  const uint32_t eAD = FindFaceEdge(f1, edges, a, dd);
  const uint32_t eBD = FindFaceEdge(f1, edges, b, dd);
  if (eAC == kInvalid || eBC == kInvalid || eAD == kInvalid || eBD == kInvalid) {
    atomicAdd(staleRejected, 1u);
    return;
  }
  ReplaceIncidentFace(edges[eBD], int32_t(f1Id), int32_t(f0Id));
  ReplaceIncidentFace(edges[eAC], int32_t(f0Id), int32_t(f1Id));
  src.v0 = cc;
  src.v1 = dd;
  ++src.generation;
  const uint32_t patch = f0.patchId;
  f0 = {cc, dd, b, c.edgeId, eBD, eBC, patch, f0.generation + 1u, 1u, {0, 0, 0}};
  f1 = {dd, cc, a, c.edgeId, eAC, eAD, patch, f1.generation + 1u, 1u, {0, 0, 0}};
}

__device__ inline bool FaceContainsEdge(const GFace &f, uint32_t a, uint32_t b) {
  const bool hasA = f.v0 == a || f.v1 == a || f.v2 == a;
  const bool hasB = f.v0 == b || f.v1 == b || f.v2 == b;
  return hasA && hasB;
}

__global__ void ExecuteSplitCandidates(GVertex *vertices, GEdge *edges, GFace *faces,
                                       SplitCandidate *candidates, uint32_t candidateCount,
                                       uint32_t *vertexCount, uint32_t *edgeCount,
                                       uint32_t *faceCount, uint32_t *staleRejected,
                                       const GPatch *patches, uint32_t patchCount,
                                       uint32_t *projectionApplied, uint32_t *projectionFailed) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= candidateCount || candidates[id].accepted != 1) return;
  const SplitCandidate c = candidates[id];
  const uint32_t segments = max(2u, min(4u, uint32_t(c.segmentCount)));
  const uint32_t added = segments - 1u;
  GEdge &src = edges[c.edgeId];
  if (!src.alive || src.generation != c.edgeGeneration || src.f0 < 0 || src.f1 < 0) {
    atomicAdd(staleRejected, 1u); return;
  }
  const uint32_t f0Id = uint32_t(src.f0), f1Id = uint32_t(src.f1);
  GFace &f0 = faces[f0Id]; GFace &f1 = faces[f1Id];
  if (!f0.alive || !f1.alive || f0.patchId != f1.patchId) {
    atomicAdd(staleRejected, 1u); return;
  }
  uint32_t a = src.v0, b = src.v1;
  if (!FaceHasDirectedEdge(f0, a, b)) {
    if (!FaceHasDirectedEdge(f0, b, a)) { atomicAdd(staleRejected, 1u); return; }
    const uint32_t t = a; a = b; b = t;
  }
  if (!FaceHasDirectedEdge(f1, b, a)) { atomicAdd(staleRejected, 1u); return; }
  const uint32_t c0 = ThirdVertex(f0, a, b), d0 = ThirdVertex(f1, a, b);
  if (c0 == kInvalid || d0 == kInvalid || c0 == d0) { atomicAdd(staleRejected, 1u); return; }
  const uint32_t eAC = FindFaceEdge(f0, edges, a, c0);
  const uint32_t eBC = FindFaceEdge(f0, edges, b, c0);
  const uint32_t eAD = FindFaceEdge(f1, edges, a, d0);
  const uint32_t eBD = FindFaceEdge(f1, edges, b, d0);
  if (eAC == kInvalid || eBC == kInvalid || eAD == kInvalid || eBD == kInvalid) {
    atomicAdd(staleRejected, 1u); return;
  }
  const uint32_t vertexBase = atomicAdd(vertexCount, added);
  const uint32_t edgeBase = atomicAdd(edgeCount, 3u * added);
  const uint32_t faceBase = atomicAdd(faceCount, 2u * added);
  uint32_t chain[5], chainEdge[4], upperFace[4], lowerFace[4];
  uint32_t upperSpoke[4] = {kInvalid,kInvalid,kInvalid,kInvalid};
  uint32_t lowerSpoke[4] = {kInvalid,kInvalid,kInvalid,kInvalid};
  chain[0] = a; chain[segments] = b;
  const GVertex va = vertices[a], vb = vertices[b];
  for (uint32_t i = 1; i < segments; ++i) {
    const float t = float(i) / float(segments);
    const uint32_t v = vertexBase + (i - 1u); chain[i] = v;
    GVertex generated = {va.x + (vb.x-va.x)*t, va.y + (vb.y-va.y)*t,
                         va.z + (vb.z-va.z)*t,
                         va.targetLength + (vb.targetLength-va.targetLength)*t,
                         f0.patchId, 1u, kSurfaceConstraint, 1u, 0u};
    vertices[v] = ProjectAnalyticVertex(generated, patches, patchCount,
                                        projectionApplied, projectionFailed);
  }
  upperFace[0] = f0Id; lowerFace[0] = f1Id;
  for (uint32_t i = 1; i < segments; ++i) {
    upperFace[i] = faceBase + (i - 1u);
    lowerFace[i] = faceBase + added + (i - 1u);
  }
  chainEdge[0] = c.edgeId;
  for (uint32_t j = 1; j < segments; ++j) chainEdge[j] = edgeBase + (j - 1u);
  for (uint32_t i = 1; i < segments; ++i) {
    upperSpoke[i] = edgeBase + added + (i - 1u);
    lowerSpoke[i] = edgeBase + 2u*added + (i - 1u);
  }
  src = {chain[0], chain[1], int32_t(upperFace[0]), int32_t(lowerFace[0]),
         src.generation + 1u, src.flags, 1u, 0u};
  for (uint32_t j = 1; j < segments; ++j)
    edges[chainEdge[j]] = {chain[j], chain[j+1], int32_t(upperFace[j]),
                           int32_t(lowerFace[j]), 1u, 0u, 1u, 0u};
  for (uint32_t i = 1; i < segments; ++i) {
    edges[upperSpoke[i]] = {chain[i], c0, int32_t(upperFace[i-1]),
                            int32_t(upperFace[i]), 1u, 0u, 1u, 0u};
    edges[lowerSpoke[i]] = {chain[i], d0, int32_t(lowerFace[i-1]),
                            int32_t(lowerFace[i]), 1u, 0u, 1u, 0u};
  }
  ReplaceIncidentFace(edges[eBC], int32_t(f0Id), int32_t(upperFace[segments-1]));
  ReplaceIncidentFace(edges[eBD], int32_t(f1Id), int32_t(lowerFace[segments-1]));
  const uint32_t patch = f0.patchId, f0Gen = f0.generation, f1Gen = f1.generation;
  for (uint32_t j = 0; j < segments; ++j) {
    const uint32_t leftC = j == 0 ? eAC : upperSpoke[j];
    const uint32_t rightC = j + 1u == segments ? eBC : upperSpoke[j+1u];
    const uint32_t leftD = j == 0 ? eAD : lowerSpoke[j];
    const uint32_t rightD = j + 1u == segments ? eBD : lowerSpoke[j+1u];
    const uint32_t ug = j == 0 ? f0Gen + 1u : 1u;
    const uint32_t lg = j == 0 ? f1Gen + 1u : 1u;
    faces[upperFace[j]] = {chain[j], chain[j+1], c0, chainEdge[j], rightC, leftC,
                           patch, ug, 1u, {0,0,0}};
    faces[lowerFace[j]] = {chain[j+1], chain[j], d0, chainEdge[j], leftD, rightD,
                           patch, lg, 1u, {0,0,0}};
  }
}


__device__ inline bool FaceContainsVertex(const GFace &f, uint32_t v) {
  return f.v0==v || f.v1==v || f.v2==v;
}

__device__ inline void ReplaceFaceVertex(GFace &f, uint32_t from, uint32_t to) {
  if (f.v0==from) f.v0=to;
  if (f.v1==from) f.v1=to;
  if (f.v2==from) f.v2=to;
}

__device__ inline void ReplaceFaceEdgeId(GFace &f, uint32_t from, uint32_t to) {
  if (f.e0==from) f.e0=to;
  if (f.e1==from) f.e1=to;
  if (f.e2==from) f.e2=to;
}

__global__ void GenerateCollapseCandidates(
    const GVertex *vertices, const GEdge *edges, const GFace *faces,
    uint32_t edgeCount, uint32_t faceCount, float collapseRatio,
    const unsigned long long *edgeHashKeys, uint32_t hashMask,
    const uint32_t *vertexEdgeOffsets, const uint32_t *vertexEdgeIds,
    const GPatch *patches, uint32_t patchCount,
    CollapseCandidate *candidates, uint32_t candidateCapacity,
    uint32_t *candidateCount, uint32_t *topologyRejected,
    uint32_t *semanticRejected, uint32_t *qualityRejected) {
  const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;
  if (id>=edgeCount) return;
  const GEdge e=edges[id];
  if (!e.alive || e.f0<0 || e.f1<0 || e.flags!=0) return;

  if (uint32_t(e.f0)>=faceCount || uint32_t(e.f1)>=faceCount) return;
  const GVertex va=vertices[e.v0], vb=vertices[e.v1];
  if (!va.alive || !vb.alive) return;
  const float h=0.5f*(va.targetLength+vb.targetLength);
  if (!(h>0.0f)) return;
  const float len2=Dist2(va,vb);
  const float limit=collapseRatio*h;
  if (!(len2<limit*limit)) return;
  const bool vaEditable = va.constraint==uint8_t(VertexConstraint::Free) ||
                          va.constraint==kSurfaceConstraint;
  const bool vbEditable = vb.constraint==uint8_t(VertexConstraint::Free) ||
                          vb.constraint==kSurfaceConstraint;
  if (!vaEditable || !vbEditable || va.patchId!=vb.patchId) {
    if (semanticRejected) atomicAdd(semanticRejected,1u);
    return;
  }
  const GFace f0=faces[e.f0], f1=faces[e.f1];
  if (!f0.alive || !f1.alive || f0.patchId!=f1.patchId || f0.patchId!=va.patchId) {
    if (semanticRejected) atomicAdd(semanticRejected,1u);
    return;
  }
  const uint32_t keep=min(e.v0,e.v1), remove=max(e.v0,e.v1);
  const uint32_t c=ThirdVertex(f0,keep,remove), d=ThirdVertex(f1,keep,remove);
  if (c==kInvalid || d==kInvalid || c==d) {
    if (topologyRejected) atomicAdd(topologyRejected,1u);
    return;
  }
  uint32_t common=0;
  for (uint32_t j=vertexEdgeOffsets[remove];j<vertexEdgeOffsets[remove+1u];++j) {
    const GEdge x=edges[vertexEdgeIds[j]];
    if (!x.alive) continue;
    uint32_t other=kInvalid;
    if (x.v0==remove) other=x.v1; else if (x.v1==remove) other=x.v0;
    if (other==kInvalid || other==keep) continue;
    if (EdgeHashContains(edgeHashKeys,hashMask,keep,other)) ++common;
  }
  if (common!=2u) { if(topologyRejected) atomicAdd(topologyRejected,1u); return; }
  // Link-condition extension: any remove-neighbor other than the two opposite
  // vertices must not already be adjacent to keep, otherwise the collapse
  // would create a duplicate edge / non-manifold one-ring.
  for (uint32_t j=vertexEdgeOffsets[remove];j<vertexEdgeOffsets[remove+1u];++j) {
    const GEdge x=edges[vertexEdgeIds[j]];
    if (!x.alive) continue;
    uint32_t other=kInvalid;
    if (x.v0==remove) other=x.v1; else if (x.v1==remove) other=x.v0;
    if (other==kInvalid || other==keep || other==c || other==d) continue;
    if (EdgeHashContains(edgeHashKeys,hashMask,keep,other)) {
      if(topologyRejected) atomicAdd(topologyRejected,1u);
      return;
    }
  }

  constexpr float qMin=1.0e-6f;
  for (uint32_t j=vertexEdgeOffsets[remove];j<vertexEdgeOffsets[remove+1u];++j) {
    const GEdge x=edges[vertexEdgeIds[j]];
    if(!x.alive) continue;
    const int32_t localFaces[2]={x.f0,x.f1};
    for(int k=0;k<2;++k) {
      if(localFaces[k]<0 || uint32_t(localFaces[k])>=faceCount) continue;
      const uint32_t fi=uint32_t(localFaces[k]);
      const GFace oldF=faces[fi];
      if (!oldF.alive || !FaceContainsVertex(oldF,remove)) continue;
      if (FaceContainsVertex(oldF,keep)) continue;
      if (oldF.patchId!=va.patchId) { if(semanticRejected) atomicAdd(semanticRejected,1u); return; }
      GFace nf=oldF;
      ReplaceFaceVertex(nf,remove,keep);
      if (nf.v0==nf.v1 || nf.v1==nf.v2 || nf.v0==nf.v2 ||
          !(TriangleQualityGpu(vertices[nf.v0],vertices[nf.v1],vertices[nf.v2])>qMin) ||
          !SameHemisphere(vertices[oldF.v0],vertices[oldF.v1],vertices[oldF.v2],
                          vertices[nf.v0],vertices[nf.v1],vertices[nf.v2]) ||
          !PatchOrientationSafe(patches, patchCount, nf.patchId,
                                vertices[nf.v0],vertices[nf.v1],vertices[nf.v2])) {
        if (qualityRejected) atomicAdd(qualityRejected,1u);
        return;
      }
    }
  }
  const uint32_t slot=atomicAdd(candidateCount,1u);
  if (slot>=candidateCapacity) return;
  const float len=sqrtf(len2);
  candidates[slot]={id,e.generation,keep,remove,c,d,uint32_t(e.f0),uint32_t(e.f1),
                    h/fmaxf(len,1e-12f),0u,{0,0,0}};
}

__device__ inline unsigned long long CollapseWinnerKey(const CollapseCandidate &c,uint32_t id) {
  const uint32_t bits=__float_as_uint(c.priority);
  return (unsigned long long(bits)<<32)|unsigned long long(0xffffffffu-id);
}

__device__ inline void ClaimCollapseVertexRing(uint32_t v,const GEdge *edges,
    const uint32_t *offsets,const uint32_t *edgeIds,unsigned long long key,
    unsigned long long *vertexWinner) {
  atomicMax(vertexWinner+v,key);
  for(uint32_t i=offsets[v];i<offsets[v+1u];++i) {
    const GEdge e=edges[edgeIds[i]]; if(!e.alive) continue;
    atomicMax(vertexWinner+e.v0,key); atomicMax(vertexWinner+e.v1,key);
  }
}

__device__ inline bool OwnCollapseVertexRing(uint32_t v,const GEdge *edges,
    const uint32_t *offsets,const uint32_t *edgeIds,unsigned long long key,
    const unsigned long long *vertexWinner) {
  if(vertexWinner[v]!=key) return false;
  for(uint32_t i=offsets[v];i<offsets[v+1u];++i) {
    const GEdge e=edges[edgeIds[i]]; if(!e.alive) continue;
    if(vertexWinner[e.v0]!=key||vertexWinner[e.v1]!=key) return false;
  }
  return true;
}

__device__ inline void BlockCollapseVertexRing(uint32_t v,const GEdge *edges,
    const uint32_t *offsets,const uint32_t *edgeIds,uint8_t *vertexBlocked) {
  vertexBlocked[v]=1;
  for(uint32_t i=offsets[v];i<offsets[v+1u];++i) {
    const GEdge e=edges[edgeIds[i]]; if(!e.alive) continue;
    vertexBlocked[e.v0]=1; vertexBlocked[e.v1]=1;
  }
}

__device__ inline bool CollapseVertexRingBlocked(uint32_t v,const GEdge *edges,
    const uint32_t *offsets,const uint32_t *edgeIds,const uint8_t *vertexBlocked) {
  if(vertexBlocked[v]) return true;
  for(uint32_t i=offsets[v];i<offsets[v+1u];++i) {
    const GEdge e=edges[edgeIds[i]]; if(!e.alive) continue;
    if(vertexBlocked[e.v0]||vertexBlocked[e.v1]) return true;
  }
  return false;
}

__global__ void ClaimCollapseCandidates(const CollapseCandidate *candidates,const uint32_t *active,
    uint32_t activeCount,const GEdge *edges,const uint32_t *offsets,const uint32_t *edgeIds,
    unsigned long long *vertexWinner) {
  const uint32_t slot=blockIdx.x*blockDim.x+threadIdx.x; if(slot>=activeCount) return;
  const uint32_t id=active[slot]; const CollapseCandidate c=candidates[id];
  const unsigned long long key=CollapseWinnerKey(c,id);
  ClaimCollapseVertexRing(c.keep,edges,offsets,edgeIds,key,vertexWinner);
  ClaimCollapseVertexRing(c.remove,edges,offsets,edgeIds,key,vertexWinner);
}

__global__ void ResolveCollapseCandidates(CollapseCandidate *candidates,const uint32_t *active,
    uint32_t activeCount,const GEdge *edges,const uint32_t *offsets,const uint32_t *edgeIds,
    const unsigned long long *vertexWinner,uint8_t *vertexBlocked,
    uint32_t *acceptedCount,uint32_t *roundAccepted) {
  const uint32_t slot=blockIdx.x*blockDim.x+threadIdx.x; if(slot>=activeCount) return;
  const uint32_t id=active[slot]; CollapseCandidate &c=candidates[id];
  const unsigned long long key=CollapseWinnerKey(c,id);
  if(!OwnCollapseVertexRing(c.keep,edges,offsets,edgeIds,key,vertexWinner) ||
     !OwnCollapseVertexRing(c.remove,edges,offsets,edgeIds,key,vertexWinner)) return;
  c.accepted=1;
  BlockCollapseVertexRing(c.keep,edges,offsets,edgeIds,vertexBlocked);
  BlockCollapseVertexRing(c.remove,edges,offsets,edgeIds,vertexBlocked);
  atomicAdd(acceptedCount,1u); atomicAdd(roundAccepted,1u);
}

__global__ void CompactCollapseCandidates(CollapseCandidate *candidates,const uint32_t *active,
    uint32_t activeCount,const GEdge *edges,const uint32_t *offsets,const uint32_t *edgeIds,
    const uint8_t *vertexBlocked,uint32_t *nextActive,uint32_t *nextCount) {
  const uint32_t slot=blockIdx.x*blockDim.x+threadIdx.x; if(slot>=activeCount) return;
  const uint32_t id=active[slot]; CollapseCandidate &c=candidates[id];
  if(c.accepted!=0) return;
  if(CollapseVertexRingBlocked(c.keep,edges,offsets,edgeIds,vertexBlocked) ||
     CollapseVertexRingBlocked(c.remove,edges,offsets,edgeIds,vertexBlocked)) {
    c.accepted=2; return;
  }
  nextActive[atomicAdd(nextCount,1u)]=id;
}

__device__ inline int32_t OtherFace(const GEdge &e,uint32_t faceId) {
  if(e.f0==int32_t(faceId)) return e.f1;
  if(e.f1==int32_t(faceId)) return e.f0;
  return -1;
}

__global__ void ExecuteCollapseCandidates(GVertex *vertices,GEdge *edges,GFace *faces,
    CollapseCandidate *candidates,uint32_t candidateCount,uint32_t edgeCount,uint32_t faceCount,
    const uint32_t *vertexEdgeOffsets,const uint32_t *vertexEdgeIds,uint32_t *staleRejected) {
  const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=candidateCount||candidates[id].accepted!=1) return;
  const CollapseCandidate c=candidates[id];
  GEdge &src=edges[c.edgeId];
  if(!src.alive||src.generation!=c.edgeGeneration||
     !vertices[c.keep].alive||!vertices[c.remove].alive){atomicAdd(staleRejected,1u);return;}
  if(c.f0>=faceCount||c.f1>=faceCount||!faces[c.f0].alive||!faces[c.f1].alive){atomicAdd(staleRejected,1u);return;}
  const uint32_t eKC=FindFaceEdge(faces[c.f0],edges,c.keep,c.c);
  const uint32_t eRC=FindFaceEdge(faces[c.f0],edges,c.remove,c.c);
  const uint32_t eKD=FindFaceEdge(faces[c.f1],edges,c.keep,c.d);
  const uint32_t eRD=FindFaceEdge(faces[c.f1],edges,c.remove,c.d);
  if(eKC==kInvalid||eRC==kInvalid||eKD==kInvalid||eRD==kInvalid){atomicAdd(staleRejected,1u);return;}
  const int32_t outKC=OtherFace(edges[eKC],c.f0), outRC=OtherFace(edges[eRC],c.f0);
  const int32_t outKD=OtherFace(edges[eKD],c.f1), outRD=OtherFace(edges[eRD],c.f1);
  if(outKC<0||outRC<0||outKD<0||outRD<0||
     uint32_t(outKC)>=faceCount||uint32_t(outRC)>=faceCount||
     uint32_t(outKD)>=faceCount||uint32_t(outRD)>=faceCount||
     !faces[outKC].alive||!faces[outRC].alive||!faces[outKD].alive||!faces[outRD].alive){
    atomicAdd(staleRejected,1u);return;
  }

  // Update only the remove one-ring faces from the pass-snapshot CSR.
  for(uint32_t i=vertexEdgeOffsets[c.remove];i<vertexEdgeOffsets[c.remove+1u];++i){
    const GEdge e=edges[vertexEdgeIds[i]];
    if(!e.alive) continue;
    const int32_t fs[2]={e.f0,e.f1};
    for(int k=0;k<2;++k){
      if(fs[k]<0||uint32_t(fs[k])>=faceCount) continue;
      const uint32_t fi=uint32_t(fs[k]);
      if(fi==c.f0||fi==c.f1) continue;
      GFace &f=faces[fi];
      if(f.alive&&FaceContainsVertex(f,c.remove)){ReplaceFaceVertex(f,c.remove,c.keep);++f.generation;}
    }
  }

  // Update only edges incident to the removed vertex.
  for(uint32_t i=vertexEdgeOffsets[c.remove];i<vertexEdgeOffsets[c.remove+1u];++i){
    GEdge &e=edges[vertexEdgeIds[i]]; if(!e.alive) continue;
    bool changed=false;
    if(e.v0==c.remove){e.v0=c.keep;changed=true;}
    if(e.v1==c.remove){e.v1=c.keep;changed=true;}
    if(changed) ++e.generation;
  }

  ReplaceFaceEdgeId(faces[outRC],eRC,eKC);
  ReplaceFaceEdgeId(faces[outRD],eRD,eKD);
  edges[eKC].f0=outKC; edges[eKC].f1=outRC; ++edges[eKC].generation;
  edges[eKD].f0=outKD; edges[eKD].f1=outRD; ++edges[eKD].generation;
  edges[eRC].alive=0; edges[eRC].f0=edges[eRC].f1=-1; ++edges[eRC].generation;
  edges[eRD].alive=0; edges[eRD].f0=edges[eRD].f1=-1; ++edges[eRD].generation;
  src.alive=0; src.f0=src.f1=-1; ++src.generation;
  faces[c.f0].alive=0; ++faces[c.f0].generation;
  faces[c.f1].alive=0; ++faces[c.f1].generation;
  vertices[c.remove].alive=0; ++vertices[c.remove].generation;
}

__device__ inline float FaceArea2(const GVertex *vertices, const GFace &f) {
  const GVertex a = vertices[f.v0], b = vertices[f.v1], c = vertices[f.v2];
  const float abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
  const float acx = c.x - a.x, acy = c.y - a.y, acz = c.z - a.z;
  const float nx = aby * acz - abz * acy;
  const float ny = abz * acx - abx * acz;
  const float nz = abx * acy - aby * acx;
  return nx * nx + ny * ny + nz * nz;
}

__global__ void ValidateFacesKernel(const GVertex *vertices, uint32_t vertexCount,
                                    const GEdge *edges, uint32_t edgeCount,
                                    const GFace *faces, uint32_t faceCount,
                                    ValidationCounters *out) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= faceCount) return;
  const GFace f = faces[id];
  if (!f.alive) return;
  if (f.v0 >= vertexCount || f.v1 >= vertexCount || f.v2 >= vertexCount) {
    atomicAdd(&out->invalidVertexReference, 1u);
    return;
  }
  if (f.v0 == f.v1 || f.v1 == f.v2 || f.v0 == f.v2) {
    atomicAdd(&out->degenerateFace, 1u);
    return;
  }
  if (!(FaceArea2(vertices, f) > 1e-20f)) atomicAdd(&out->zeroAreaFace, 1u);
  const uint32_t ev[3] = {f.e0, f.e1, f.e2};
  const uint32_t va[3] = {f.v0, f.v1, f.v2};
  const uint32_t vb[3] = {f.v1, f.v2, f.v0};
  for (int i = 0; i < 3; ++i) {
    if (ev[i] >= edgeCount || !edges[ev[i]].alive) {
      atomicAdd(&out->invalidFaceReference, 1u);
      continue;
    }
    const GEdge e = edges[ev[i]];
    if (!EdgeMatches(e, va[i], vb[i]) ||
        (e.f0 != int32_t(id) && e.f1 != int32_t(id))) {
      atomicAdd(&out->edgeFaceMismatch, 1u);
    }
  }
}

__global__ void ValidateEdgesKernel(const GVertex *vertices, uint32_t vertexCount,
                                    const GEdge *edges, uint32_t edgeCount,
                                    const GFace *faces, uint32_t faceCount,
                                    ValidationCounters *out) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= edgeCount || !edges[id].alive) return;
  const GEdge e = edges[id];
  if (e.v0 >= vertexCount || e.v1 >= vertexCount || e.v0 == e.v1 ||
      !vertices[e.v0].alive || !vertices[e.v1].alive) {
    atomicAdd(&out->invalidVertexReference, 1u);
    return;
  }
  const int32_t fs[2] = {e.f0, e.f1};
  for (int i = 0; i < 2; ++i) {
    if (fs[i] < 0) continue;
    if (uint32_t(fs[i]) >= faceCount || !faces[fs[i]].alive) {
      atomicAdd(&out->invalidFaceReference, 1u);
      continue;
    }
    if (!FaceContainsEdge(faces[fs[i]], e.v0, e.v1))
      atomicAdd(&out->edgeFaceMismatch, 1u);
  }
}

} // namespace

template <class T>
static void GrowManagedArray(T *&ptr, size_t preserveCount, size_t newCapacity,
                             const char *where) {
  T *next = AllocManaged<T>(newCapacity);
  if (ptr && preserveCount > 0) {
    CheckCuda(cudaMemcpy(next, ptr, sizeof(T) * preserveCount, cudaMemcpyDefault), where);
  }
  cudaFree(ptr);
  ptr = next;
}

struct GlobalSplitBackend::Impl {
  GVertex *vertices = nullptr;
  GEdge *edges = nullptr;
  GFace *faces = nullptr;
  GPatch *gpuPatches = nullptr;
  uint32_t gpuPatchCount = 0;
  SplitCandidate *candidates = nullptr;
  FlipCandidate *flipCandidates = nullptr;
  CollapseCandidate *collapseCandidates = nullptr;
  TriangleRefineCandidate *triangleCandidates = nullptr;
  unsigned long long *edgeHashKeys = nullptr;
  uint32_t *edgeOwner = nullptr;
  uint32_t *faceOwner = nullptr;
  uint32_t *vertexOwner = nullptr;
  unsigned long long *splitEdgeWinner = nullptr;
  unsigned long long *splitFaceWinner = nullptr;
  unsigned long long *splitVertexWinner = nullptr;
  uint8_t *edgeBlocked = nullptr;
  uint8_t *faceBlocked = nullptr;
  uint8_t *vertexBlocked = nullptr;
  uint32_t *activeQueueA = nullptr;
  uint32_t *activeQueueB = nullptr;
  ValidationCounters *validation = nullptr;

  uint32_t *vertexCount = nullptr;
  uint32_t *edgeCount = nullptr;
  uint32_t *faceCount = nullptr;
  uint32_t *candidateCount = nullptr;
  uint32_t *acceptedCount = nullptr;
  uint32_t *roundAccepted = nullptr;
  uint32_t *activeCount = nullptr;
  uint32_t *nextActiveCount = nullptr;
  uint32_t *staleRejected = nullptr;
  uint32_t *capacityRejected = nullptr;
  uint32_t *semanticRejected = nullptr;
  uint32_t *qualityRejected = nullptr;
  uint32_t *collapseTopologyRejected = nullptr;
  uint32_t *projectionApplied = nullptr;
  uint32_t *projectionFailed = nullptr;
  uint32_t *triangleRejectCounters = nullptr;
  uint32_t *triangleEditableHistogram = nullptr;
  uint32_t *splitNewVertexCount = nullptr;
  uint32_t *vertexEdgeCounts = nullptr;
  uint32_t *vertexEdgeOffsets = nullptr;
  uint32_t *vertexEdgeCursor = nullptr;
  uint32_t *vertexEdgeIds = nullptr;
  void *adjacencyScanTemp = nullptr;
  size_t adjacencyScanTempBytes = 0;

  uint32_t vertexCapacity = 0;
  uint32_t edgeCapacity = 0;
  uint32_t faceCapacity = 0;
  uint32_t candidateCapacity = 0;
  uint32_t edgeHashCapacity = 0;
  float fallbackTargetLength = 1.0f;
  std::vector<PatchRecord> patches;


  void EnsureAppendCapacity(uint32_t addVertices, uint32_t addEdges, uint32_t addFaces) {
    const uint32_t requiredV = *vertexCount + addVertices;
    const uint32_t requiredE = *edgeCount + addEdges;
    const uint32_t requiredF = *faceCount + addFaces;
    if (requiredV <= vertexCapacity && requiredE <= edgeCapacity && requiredF <= faceCapacity) return;

    CheckCuda(cudaDeviceSynchronize(), "EnsureAppendCapacity synchronize");
    const uint32_t oldV = vertexCapacity, oldE = edgeCapacity, oldF = faceCapacity;
    uint32_t newV = oldV, newE = oldE, newF = oldF;
    while (newV < requiredV) newV = std::max(requiredV, newV * 2u);
    while (newE < requiredE) newE = std::max(requiredE, newE * 2u);
    while (newF < requiredF) newF = std::max(requiredF, newF * 2u);
    const uint32_t preserveCandidates = std::min(*candidateCount, candidateCapacity);

    if (newV != oldV) {
      GrowManagedArray(vertices, *vertexCount, newV, "grow vertices");
      GrowManagedArray(vertexOwner, 0, newV, "grow vertex owners");
      GrowManagedArray(splitVertexWinner, 0, newV, "grow vertex winners");
      GrowManagedArray(vertexBlocked, 0, newV, "grow vertex blocked");
      GrowManagedArray(vertexEdgeCounts, 0, size_t(newV)+1u, "grow vertex edge counts");
      GrowManagedArray(vertexEdgeOffsets, 0, size_t(newV)+1u, "grow vertex edge offsets");
      GrowManagedArray(vertexEdgeCursor, 0, newV, "grow vertex edge cursor");
      vertexCapacity = newV;
    }
    if (newE != oldE) {
      GrowManagedArray(edges, *edgeCount, newE, "grow edges");
      GrowManagedArray(edgeOwner, 0, newE, "grow edge owners");
      GrowManagedArray(splitEdgeWinner, 0, newE, "grow edge winners");
      GrowManagedArray(edgeBlocked, 0, newE, "grow edge blocked");
      GrowManagedArray(candidates, preserveCandidates, newE, "grow split candidates");
      GrowManagedArray(flipCandidates, preserveCandidates, newE, "grow flip candidates");
      GrowManagedArray(collapseCandidates, preserveCandidates, newE, "grow collapse candidates");
      GrowManagedArray(triangleCandidates, preserveCandidates, newE, "grow triangle candidates");
      GrowManagedArray(activeQueueA, 0, newE, "grow active queue A");
      GrowManagedArray(activeQueueB, 0, newE, "grow active queue B");
      GrowManagedArray(vertexEdgeIds, 0, size_t(newE)*2u, "grow vertex edge ids");
      candidateCapacity = newE;
      edgeCapacity = newE;

      uint32_t newHashCapacity = 1u;
      while (newHashCapacity < newE * 2u) newHashCapacity <<= 1u;
      if (newHashCapacity != edgeHashCapacity) {
        GrowManagedArray(edgeHashKeys, 0, newHashCapacity, "grow edge hash");
        edgeHashCapacity = newHashCapacity;
      }
    }
    if (newF != oldF) {
      GrowManagedArray(faces, *faceCount, newF, "grow faces");
      GrowManagedArray(faceOwner, 0, newF, "grow face owners");
      GrowManagedArray(splitFaceWinner, 0, newF, "grow face winners");
      GrowManagedArray(faceBlocked, 0, newF, "grow face blocked");
      faceCapacity = newF;
    }
  }

  void BuildVertexEdgeAdjacency() {
    const uint32_t nv=*vertexCount, ne=*edgeCount;
    if(nv==0) return;
    CheckCuda(cudaMemset(vertexEdgeCounts,0,sizeof(uint32_t)*(size_t(nv)+1u)),
              "reset vertex edge counts");
    constexpr uint32_t threads=256;
    if(ne>0) {
      const uint32_t blocks=(ne+threads-1u)/threads;
      CountVertexEdges<<<blocks,threads>>>(edges,ne,nv,vertexEdgeCounts);
      CheckCuda(cudaDeviceSynchronize(),"CountVertexEdges");
    }
    size_t requiredBytes=0;
    CheckCuda(cub::DeviceScan::ExclusiveSum(nullptr,requiredBytes,
              vertexEdgeCounts,vertexEdgeOffsets,nv+1u),"query vertex edge scan");
    if(requiredBytes>adjacencyScanTempBytes) {
      cudaFree(adjacencyScanTemp); adjacencyScanTemp=nullptr;
      CheckCuda(cudaMalloc(&adjacencyScanTemp,requiredBytes),"alloc vertex edge scan temp");
      adjacencyScanTempBytes=requiredBytes;
    }
    CheckCuda(cub::DeviceScan::ExclusiveSum(adjacencyScanTemp,adjacencyScanTempBytes,
              vertexEdgeCounts,vertexEdgeOffsets,nv+1u),"scan vertex edge offsets");
    CheckCuda(cudaDeviceSynchronize(),"scan vertex edge offsets sync");
    if(vertexEdgeOffsets[nv] > size_t(edgeCapacity)*2u)
      throw std::runtime_error("vertex edge adjacency capacity exhausted");
    CheckCuda(cudaMemset(vertexEdgeCursor,0,sizeof(uint32_t)*nv),"reset vertex edge cursor");
    if(ne>0) {
      const uint32_t blocks=(ne+threads-1u)/threads;
      FillVertexEdges<<<blocks,threads>>>(edges,ne,nv,vertexEdgeOffsets,
                                          vertexEdgeCursor,vertexEdgeIds);
      CheckCuda(cudaDeviceSynchronize(),"FillVertexEdges");
    }
  }

  ~Impl() {
    cudaFree(vertices);
    cudaFree(edges);
    cudaFree(faces);
    cudaFree(gpuPatches);
    cudaFree(candidates);
    cudaFree(flipCandidates);
    cudaFree(collapseCandidates);
    cudaFree(triangleCandidates);
    cudaFree(edgeHashKeys);
    cudaFree(edgeOwner);
    cudaFree(faceOwner);
    cudaFree(vertexOwner);
    cudaFree(splitEdgeWinner);
    cudaFree(splitFaceWinner);
    cudaFree(splitVertexWinner);
    cudaFree(edgeBlocked);
    cudaFree(faceBlocked);
    cudaFree(vertexBlocked);
    cudaFree(activeQueueA);
    cudaFree(activeQueueB);
    cudaFree(validation);
    cudaFree(vertexCount);
    cudaFree(edgeCount);
    cudaFree(faceCount);
    cudaFree(candidateCount);
    cudaFree(acceptedCount);
    cudaFree(roundAccepted);
    cudaFree(activeCount);
    cudaFree(nextActiveCount);
    cudaFree(staleRejected);
    cudaFree(capacityRejected);
    cudaFree(semanticRejected);
    cudaFree(qualityRejected);
    cudaFree(collapseTopologyRejected);
    cudaFree(projectionApplied);
    cudaFree(projectionFailed);
    cudaFree(triangleRejectCounters);
    cudaFree(triangleEditableHistogram);
    cudaFree(splitNewVertexCount);
    cudaFree(vertexEdgeCounts);
    cudaFree(vertexEdgeOffsets);
    cudaFree(vertexEdgeCursor);
    cudaFree(vertexEdgeIds);
    cudaFree(adjacencyScanTemp);
  }

  size_t MemoryBytes() const {
    return size_t(vertexCapacity) * sizeof(GVertex) +
           size_t(edgeCapacity) * (sizeof(GEdge) + sizeof(SplitCandidate) + sizeof(FlipCandidate) + sizeof(CollapseCandidate) + sizeof(TriangleRefineCandidate) + 3 * sizeof(uint32_t) + sizeof(uint8_t)) +
           size_t(faceCapacity) * (sizeof(GFace) + sizeof(uint32_t) + sizeof(uint8_t)) +
           size_t(vertexCapacity) * (sizeof(uint32_t) + sizeof(uint8_t) + sizeof(unsigned long long)) +
           size_t(edgeCapacity) * sizeof(unsigned long long) +
           size_t(faceCapacity) * sizeof(unsigned long long) +
           size_t(gpuPatchCount) * sizeof(GPatch) +
           (size_t(vertexCapacity)+1u) * 2u * sizeof(uint32_t) +
           size_t(vertexCapacity) * sizeof(uint32_t) +
           size_t(edgeCapacity) * 2u * sizeof(uint32_t) +
           adjacencyScanTempBytes +
           sizeof(ValidationCounters) + 13 * sizeof(uint32_t);
  }
};

GlobalSplitBackend::GlobalSplitBackend() : mImpl(std::make_unique<Impl>()) {}
GlobalSplitBackend::~GlobalSplitBackend() = default;

bool GlobalSplitBackend::Initialize(const SemanticMesh &mesh, float capacityFactor,
                                    std::string *error) {
  try {
    mImpl = std::make_unique<Impl>();
    SemanticMesh src = mesh;
    src.rebuildTopology();
    if (src.vertexCount() == 0 || src.faceCount() == 0)
      throw std::runtime_error("empty mesh");
    if (!(capacityFactor >= 1.5f)) capacityFactor = 2.0f;

    const uint32_t nv = uint32_t(src.vertexCount());
    const uint32_t ne = uint32_t(src.edges.size());
    const uint32_t nf = uint32_t(src.faceCount());
    mImpl->vertexCapacity = std::max(nv + 1024u, uint32_t(std::ceil(nv * capacityFactor)));
    mImpl->edgeCapacity = std::max(ne + 3072u, uint32_t(std::ceil(ne * capacityFactor)));
    mImpl->faceCapacity = std::max(nf + 2048u, uint32_t(std::ceil(nf * capacityFactor)));
    mImpl->candidateCapacity = mImpl->edgeCapacity;
    mImpl->edgeHashCapacity = 1u;
    while (mImpl->edgeHashCapacity < mImpl->edgeCapacity * 2u) mImpl->edgeHashCapacity <<= 1u;
    mImpl->fallbackTargetLength = std::max(1e-6f, src.bboxDiagonal() * 0.01f);
    mImpl->patches = src.patches;

    mImpl->vertices = AllocManaged<GVertex>(mImpl->vertexCapacity);
    mImpl->edges = AllocManaged<GEdge>(mImpl->edgeCapacity);
    mImpl->faces = AllocManaged<GFace>(mImpl->faceCapacity);
    mImpl->gpuPatchCount = uint32_t(src.patches.size());
    if (mImpl->gpuPatchCount > 0) mImpl->gpuPatches = AllocManaged<GPatch>(mImpl->gpuPatchCount);
    mImpl->candidates = AllocManaged<SplitCandidate>(mImpl->candidateCapacity);
    mImpl->flipCandidates = AllocManaged<FlipCandidate>(mImpl->candidateCapacity);
    mImpl->collapseCandidates = AllocManaged<CollapseCandidate>(mImpl->candidateCapacity);
    mImpl->triangleCandidates = AllocManaged<TriangleRefineCandidate>(mImpl->candidateCapacity);
    mImpl->edgeHashKeys = AllocManaged<unsigned long long>(mImpl->edgeHashCapacity);
    mImpl->edgeOwner = AllocManaged<uint32_t>(mImpl->edgeCapacity);
    mImpl->faceOwner = AllocManaged<uint32_t>(mImpl->faceCapacity);
    mImpl->vertexOwner = AllocManaged<uint32_t>(mImpl->vertexCapacity);
    mImpl->splitEdgeWinner = AllocManaged<unsigned long long>(mImpl->edgeCapacity);
    mImpl->splitFaceWinner = AllocManaged<unsigned long long>(mImpl->faceCapacity);
    mImpl->splitVertexWinner = AllocManaged<unsigned long long>(mImpl->vertexCapacity);
    mImpl->edgeBlocked = AllocManaged<uint8_t>(mImpl->edgeCapacity);
    mImpl->faceBlocked = AllocManaged<uint8_t>(mImpl->faceCapacity);
    mImpl->vertexBlocked = AllocManaged<uint8_t>(mImpl->vertexCapacity);
    mImpl->activeQueueA = AllocManaged<uint32_t>(mImpl->candidateCapacity);
    mImpl->activeQueueB = AllocManaged<uint32_t>(mImpl->candidateCapacity);
    mImpl->validation = AllocManaged<ValidationCounters>(1);
    mImpl->vertexCount = AllocManaged<uint32_t>(1);
    mImpl->edgeCount = AllocManaged<uint32_t>(1);
    mImpl->faceCount = AllocManaged<uint32_t>(1);
    mImpl->candidateCount = AllocManaged<uint32_t>(1);
    mImpl->acceptedCount = AllocManaged<uint32_t>(1);
    mImpl->roundAccepted = AllocManaged<uint32_t>(1);
    mImpl->activeCount = AllocManaged<uint32_t>(1);
    mImpl->nextActiveCount = AllocManaged<uint32_t>(1);
    mImpl->staleRejected = AllocManaged<uint32_t>(1);
    mImpl->capacityRejected = AllocManaged<uint32_t>(1);
    mImpl->semanticRejected = AllocManaged<uint32_t>(1);
    mImpl->qualityRejected = AllocManaged<uint32_t>(1);
    mImpl->collapseTopologyRejected = AllocManaged<uint32_t>(1);
    mImpl->projectionApplied = AllocManaged<uint32_t>(1);
    mImpl->projectionFailed = AllocManaged<uint32_t>(1);
    mImpl->triangleRejectCounters = AllocManaged<uint32_t>(4);
    mImpl->triangleEditableHistogram = AllocManaged<uint32_t>(4);
    mImpl->splitNewVertexCount = AllocManaged<uint32_t>(1);
    mImpl->vertexEdgeCounts = AllocManaged<uint32_t>(size_t(mImpl->vertexCapacity)+1u);
    mImpl->vertexEdgeOffsets = AllocManaged<uint32_t>(size_t(mImpl->vertexCapacity)+1u);
    mImpl->vertexEdgeCursor = AllocManaged<uint32_t>(mImpl->vertexCapacity);
    mImpl->vertexEdgeIds = AllocManaged<uint32_t>(size_t(mImpl->edgeCapacity)*2u);

    *mImpl->vertexCount = nv;
    *mImpl->edgeCount = ne;
    *mImpl->faceCount = nf;
    *mImpl->candidateCount = 0;
    *mImpl->acceptedCount = 0;
    *mImpl->roundAccepted = 0;
    *mImpl->activeCount = 0;
    *mImpl->nextActiveCount = 0;
    *mImpl->staleRejected = 0;
    *mImpl->capacityRejected = 0;
    *mImpl->semanticRejected = 0;
    *mImpl->qualityRejected = 0;
    *mImpl->collapseTopologyRejected = 0;
    *mImpl->projectionApplied = 0;
    *mImpl->projectionFailed = 0;
    for (int i = 0; i < 4; ++i) { mImpl->triangleRejectCounters[i] = 0; mImpl->triangleEditableHistogram[i] = 0; }
    *mImpl->splitNewVertexCount = 0;

    std::vector<float> patchOrientation(mImpl->gpuPatchCount, 0.0f);
    for (uint32_t f = 0; f < uint32_t(src.faceCount()); ++f) {
      if (!src.faceAlive[f]) continue;
      const uint32_t patchId = src.facePatchId[f];
      if (patchId >= src.patches.size()) continue;
      const PatchRecord &p = src.patches[patchId];
      Vec3 an{};
      const Vec3 a = src.facePoint(int(f), 0);
      const Vec3 b = src.facePoint(int(f), 1);
      const Vec3 c = src.facePoint(int(f), 2);
      const Vec3 mid = centroid3(a, b, c);
      if (p.type == PatchType::Plane) {
        an = normalize(p.axis, {0, 0, 1});
      } else if (p.type == PatchType::Cylinder) {
        const Vec3 axis = normalize(p.axis, {0, 0, 1});
        const Vec3 off = mid - p.origin;
        an = normalize(off - axis * dot(off, axis), {0, 0, 0});
      }
      if (length2(an) > 0.0f)
        patchOrientation[patchId] += dot(cross(b - a, c - a), an);
    }

    for (uint32_t i = 0; i < mImpl->gpuPatchCount; ++i) {
      const PatchRecord &p = src.patches[i];
      const float ax=p.axis.x, ay=p.axis.y, az=p.axis.z;
      const float n=std::sqrt(ax*ax+ay*ay+az*az);
      const float inv=n>1e-20f ? 1.0f/n : 1.0f;
      const float orientationSign = patchOrientation[i] < 0.0f ? -1.0f : 1.0f;
      mImpl->gpuPatches[i] = {p.origin.x,p.origin.y,p.origin.z,
                              n>1e-20f?ax*inv:0.0f,
                              n>1e-20f?ay*inv:0.0f,
                              n>1e-20f?az*inv:1.0f,
                              p.radius,orientationSign,uint8_t(p.type),{0,0,0}};
    }

    for (uint32_t v = 0; v < nv; ++v) {
      const float h = v < src.targetLength.size() && src.targetLength[v] > 0
                          ? src.targetLength[v]
                          : mImpl->fallbackTargetLength;
      mImpl->vertices[v] = {src.px[v], src.py[v], src.pz[v], h,
                            v < src.vertexPatchId.size() ? src.vertexPatchId[v] : 0u,
                            1u,
                            uint8_t(v < src.vertexConstraint.size() ? src.vertexConstraint[v] : 0u),
                            1u,
                            0u};
    }
    std::unordered_map<uint64_t, uint32_t> edgeMap;
    edgeMap.reserve(src.edges.size() * 2);
    for (uint32_t e = 0; e < ne; ++e) {
      const EdgeRec &se = src.edges[e];
      mImpl->edges[e] = {se.v0, se.v1, se.face0, se.face1, 1u, se.flags, 1u, 0u};
      edgeMap.emplace(EdgeKey(se.v0, se.v1), e);
    }

    for (uint32_t f = 0; f < nf; ++f) {
      const uint32_t a = src.i0[f], b = src.i1[f], c = src.i2[f];
      const auto it0 = edgeMap.find(EdgeKey(a, b));
      const auto it1 = edgeMap.find(EdgeKey(b, c));
      const auto it2 = edgeMap.find(EdgeKey(c, a));
      if (it0 == edgeMap.end() || it1 == edgeMap.end() || it2 == edgeMap.end())
        throw std::runtime_error("face edge lookup failed");
      mImpl->faces[f] = {a, b, c, it0->second, it1->second, it2->second,
                         f < src.facePatchId.size() ? src.facePatchId[f] : 0u,
                         1u,
                         uint8_t(f < src.faceAlive.size() ? src.faceAlive[f] : 1u),
                         {0, 0, 0}};
    }

    CheckCuda(cudaDeviceSynchronize(), "Initialize synchronize");
    return true;
  } catch (const std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

bool GlobalSplitBackend::RunTriangleRefinePass(float refineRatio,
                                                GlobalTriangleRefineReport &report,
                                                std::string *error) {
  report = {};
  const auto totalStart = std::chrono::steady_clock::now();
  try {
    if (!mImpl->vertices) throw std::runtime_error("backend not initialized");
    *mImpl->candidateCount = 0;
    *mImpl->acceptedCount = 0;
    *mImpl->roundAccepted = 0;
    *mImpl->activeCount = 0;
    *mImpl->nextActiveCount = 0;
    *mImpl->staleRejected = 0;
    *mImpl->semanticRejected = 0;
    *mImpl->qualityRejected = 0;
    *mImpl->projectionApplied = 0;
    *mImpl->projectionFailed = 0;
    for (int i = 0; i < 4; ++i) { mImpl->triangleRejectCounters[i] = 0; mImpl->triangleEditableHistogram[i] = 0; }

    const uint32_t faceCount = *mImpl->faceCount;
    constexpr uint32_t threads = 256;
    const uint32_t faceBlocks = (faceCount + threads - 1u) / threads;
    auto mark = std::chrono::steady_clock::now();
    GenerateTriangleRefineCandidates<<<faceBlocks, threads>>>(
        mImpl->vertices, mImpl->edges, mImpl->faces, faceCount, refineRatio,
        mImpl->gpuPatches, mImpl->gpuPatchCount, mImpl->triangleCandidates,
        mImpl->candidateCapacity, mImpl->candidateCount,
        mImpl->semanticRejected, mImpl->qualityRejected, mImpl->triangleRejectCounters, mImpl->triangleEditableHistogram);
    CheckCuda(cudaDeviceSynchronize(), "GenerateTriangleRefineCandidates");
    report.candidateMs = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - mark).count();

    const uint32_t candidates = std::min(*mImpl->candidateCount, mImpl->candidateCapacity);
    report.candidateCount = candidates;
    report.semanticRejected = *mImpl->semanticRejected;
    report.qualityRejected = *mImpl->qualityRejected;
    report.protectedRejected = mImpl->triangleRejectCounters[0];
    report.neighborRejected = mImpl->triangleRejectCounters[1];
    report.windingRejected = mImpl->triangleRejectCounters[2];
    report.belowRatioRejected = mImpl->triangleRejectCounters[3];
    for (int i = 0; i < 4; ++i) report.editableFaceHistogram[i] = mImpl->triangleEditableHistogram[i];
    if (candidates == 0) {
      report.globalMemoryBytes = mImpl->MemoryBytes();
      report.totalMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - totalStart).count();
      return true;
    }

    mark = std::chrono::steady_clock::now();
    const uint32_t candidateBlocks = (candidates + threads - 1u) / threads;
    CheckCuda(cudaMemset(mImpl->edgeBlocked, 0, sizeof(uint8_t) * mImpl->edgeCapacity),
              "reset triangle blocked edges");
    CheckCuda(cudaMemset(mImpl->faceBlocked, 0, sizeof(uint8_t) * mImpl->faceCapacity),
              "reset triangle blocked faces");
    CheckCuda(cudaMemset(mImpl->vertexBlocked, 0, sizeof(uint8_t) * mImpl->vertexCapacity),
              "reset triangle blocked vertices");
    InitActiveQueue<<<candidateBlocks, threads>>>(mImpl->activeQueueA, candidates);
    CheckCuda(cudaDeviceSynchronize(), "InitTriangleActiveQueue");
    *mImpl->activeCount = candidates;
    uint32_t *active = mImpl->activeQueueA;
    uint32_t *next = mImpl->activeQueueB;
    constexpr int kMaxClaimRounds = 4;
    int claimRounds = 0;
    for (; claimRounds < kMaxClaimRounds && *mImpl->activeCount > 0; ++claimRounds) {
      const uint32_t activeNow = *mImpl->activeCount;
      report.activeItemsScanned += activeNow;
      const uint32_t activeBlocks = (activeNow + threads - 1u) / threads;
      CheckCuda(cudaMemset(mImpl->splitEdgeWinner, 0,
                           sizeof(unsigned long long) * mImpl->edgeCapacity),
                "reset triangle edge winners");
      CheckCuda(cudaMemset(mImpl->splitFaceWinner, 0,
                           sizeof(unsigned long long) * mImpl->faceCapacity),
                "reset triangle face winners");
      CheckCuda(cudaMemset(mImpl->splitVertexWinner, 0,
                           sizeof(unsigned long long) * mImpl->vertexCapacity),
                "reset triangle vertex winners");
      *mImpl->roundAccepted = 0;
      ClaimTriangleCandidates<<<activeBlocks, threads>>>(
          mImpl->triangleCandidates, mImpl->faces, active, activeNow,
          mImpl->splitEdgeWinner, mImpl->splitFaceWinner, mImpl->splitVertexWinner);
      CheckCuda(cudaDeviceSynchronize(), "ClaimTriangleCandidates");
      ResolveTriangleCandidates<<<activeBlocks, threads>>>(
          mImpl->triangleCandidates, mImpl->faces, active, activeNow,
          mImpl->splitEdgeWinner, mImpl->splitFaceWinner, mImpl->splitVertexWinner,
          mImpl->edgeBlocked, mImpl->faceBlocked, mImpl->vertexBlocked,
          mImpl->acceptedCount, mImpl->roundAccepted);
      CheckCuda(cudaDeviceSynchronize(), "ResolveTriangleCandidates");
      if (*mImpl->roundAccepted == 0) break;
      *mImpl->nextActiveCount = 0;
      CompactTriangleCandidates<<<activeBlocks, threads>>>(
          mImpl->triangleCandidates, mImpl->faces, active, activeNow,
          mImpl->edgeBlocked, mImpl->faceBlocked, mImpl->vertexBlocked,
          next, mImpl->nextActiveCount);
      CheckCuda(cudaDeviceSynchronize(), "CompactTriangleCandidates");
      *mImpl->activeCount = *mImpl->nextActiveCount;
      std::swap(active, next);
    }
    report.schedulerRounds = uint32_t(claimRounds);
    report.claimMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - mark).count();

    const uint32_t accepted = *mImpl->acceptedCount;
    *mImpl->splitNewVertexCount=0;
    SumAcceptedTriangleCost<<<candidateBlocks,threads>>>(mImpl->triangleCandidates,candidates,mImpl->splitNewVertexCount);
    CheckCuda(cudaDeviceSynchronize(),"SumAcceptedTriangleCost");
    const uint32_t newVertices=*mImpl->splitNewVertexCount;
    mImpl->EnsureAppendCapacity(newVertices, newVertices * 3u, newVertices * 2u);
    mark = std::chrono::steady_clock::now();
    ExecuteTriangleRefineCandidates<<<candidateBlocks, threads>>>(
        mImpl->vertices, mImpl->edges, mImpl->faces, mImpl->triangleCandidates,
        candidates, mImpl->vertexCount, mImpl->edgeCount, mImpl->faceCount,
        mImpl->staleRejected, mImpl->gpuPatches, mImpl->gpuPatchCount,
        mImpl->projectionApplied, mImpl->projectionFailed);
    CheckCuda(cudaDeviceSynchronize(), "ExecuteTriangleRefineCandidates");
    report.executeMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - mark).count();
    report.staleRejected = *mImpl->staleRejected;
    report.acceptedCount = accepted >= report.staleRejected ? accepted - report.staleRejected : 0;
    report.projectionApplied = *mImpl->projectionApplied;
    report.projectionFailed = *mImpl->projectionFailed;
    report.globalMemoryBytes = mImpl->MemoryBytes();
    report.dynamicSharedMemoryBytes = 0;
    report.totalMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - totalStart).count();
    return true;
  } catch (const std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

bool GlobalSplitBackend::RunSplitPass(float splitRatio, GlobalSplitReport &report,
                                      std::string *error, float maxRatio) {
  report = {};
  const auto totalStart = std::chrono::steady_clock::now();
  try {
    if (!mImpl->vertices) throw std::runtime_error("backend not initialized");
    *mImpl->candidateCount = 0;
    *mImpl->acceptedCount = 0;
    *mImpl->roundAccepted = 0;
    *mImpl->activeCount = 0;
    *mImpl->nextActiveCount = 0;
    *mImpl->staleRejected = 0;
    *mImpl->capacityRejected = 0;
    *mImpl->semanticRejected = 0;
    *mImpl->projectionApplied = 0;
    *mImpl->projectionFailed = 0;

    const uint32_t edgeCount = *mImpl->edgeCount;
    const uint32_t faceCount = *mImpl->faceCount;
    constexpr uint32_t threads = 256;
    const uint32_t edgeBlocks = (edgeCount + threads - 1u) / threads;

    auto mark = std::chrono::steady_clock::now();
    GenerateSplitCandidates<<<edgeBlocks, threads>>>(
        mImpl->vertices, mImpl->edges, mImpl->faces, edgeCount, faceCount,
        splitRatio, maxRatio, mImpl->gpuPatches, mImpl->gpuPatchCount,
        mImpl->candidates, mImpl->candidateCapacity,
        mImpl->candidateCount, mImpl->semanticRejected, mImpl->capacityRejected);
    CheckCuda(cudaDeviceSynchronize(), "GenerateSplitCandidates");
    report.candidateMs = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - mark)
                             .count();

    const uint32_t candidates = std::min(*mImpl->candidateCount, mImpl->candidateCapacity);
    report.candidateCount = candidates;
    report.capacityRejected = *mImpl->capacityRejected;
    report.semanticRejected = *mImpl->semanticRejected;
    if (candidates == 0) {
      report.globalMemoryBytes = mImpl->MemoryBytes();
      report.totalMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - totalStart)
                           .count();
      return true;
    }

    mark = std::chrono::steady_clock::now();
    const uint32_t candidateBlocks = (candidates + threads - 1u) / threads;
    CheckCuda(cudaMemset(mImpl->edgeBlocked, 0,
                         sizeof(uint8_t) * mImpl->edgeCapacity),
              "reset blocked edges");
    CheckCuda(cudaMemset(mImpl->faceBlocked, 0,
                         sizeof(uint8_t) * mImpl->faceCapacity),
              "reset blocked faces");
    CheckCuda(cudaMemset(mImpl->vertexBlocked, 0,
                         sizeof(uint8_t) * mImpl->vertexCapacity),
              "reset blocked vertices");
    InitActiveQueue<<<candidateBlocks, threads>>>(mImpl->activeQueueA, candidates);
    CheckCuda(cudaDeviceSynchronize(), "InitActiveQueue");
    *mImpl->activeCount = candidates;
    uint32_t *active = mImpl->activeQueueA;
    uint32_t *next = mImpl->activeQueueB;
    constexpr int kMaxClaimRounds = 4;
    int claimRounds = 0;
    for (; claimRounds < kMaxClaimRounds && *mImpl->activeCount > 0; ++claimRounds) {
      const uint32_t activeNow = *mImpl->activeCount;
      report.activeItemsScanned += activeNow;
      const uint32_t activeBlocks = (activeNow + threads - 1u) / threads;
      CheckCuda(cudaMemset(mImpl->splitEdgeWinner, 0,
                           sizeof(unsigned long long) * mImpl->edgeCapacity),
                "reset split edge winners");
      CheckCuda(cudaMemset(mImpl->splitFaceWinner, 0,
                           sizeof(unsigned long long) * mImpl->faceCapacity),
                "reset split face winners");
      CheckCuda(cudaMemset(mImpl->splitVertexWinner, 0,
                           sizeof(unsigned long long) * mImpl->vertexCapacity),
                "reset split vertex winners");
      *mImpl->roundAccepted = 0;
      ClaimCandidatesActive<<<activeBlocks, threads>>>(
          mImpl->candidates, mImpl->faces, active, activeNow,
          mImpl->splitEdgeWinner, mImpl->splitFaceWinner, mImpl->splitVertexWinner);
      CheckCuda(cudaDeviceSynchronize(), "ClaimCandidatesActive");
      ResolveCandidatesActive<<<activeBlocks, threads>>>(
          mImpl->candidates, mImpl->faces, active, activeNow,
          mImpl->splitEdgeWinner, mImpl->splitFaceWinner, mImpl->splitVertexWinner,
          mImpl->edgeBlocked, mImpl->faceBlocked, mImpl->vertexBlocked,
          mImpl->acceptedCount, mImpl->roundAccepted);
      CheckCuda(cudaDeviceSynchronize(), "ResolveCandidatesActive");
      if (*mImpl->roundAccepted == 0) break;
      *mImpl->nextActiveCount = 0;
      CompactActiveCandidates<<<activeBlocks, threads>>>(
          mImpl->candidates, mImpl->faces, active, activeNow,
          mImpl->edgeBlocked, mImpl->faceBlocked, mImpl->vertexBlocked,
          next, mImpl->nextActiveCount);
      CheckCuda(cudaDeviceSynchronize(), "CompactActiveCandidates");
      *mImpl->activeCount = *mImpl->nextActiveCount;
      std::swap(active, next);
    }
    report.schedulerRounds = uint32_t(claimRounds);
    report.claimMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - mark)
                         .count();

    const uint32_t acceptedCandidates = *mImpl->acceptedCount;
    *mImpl->splitNewVertexCount = 0;
    SumAcceptedSplitCost<<<candidateBlocks, threads>>>(
        mImpl->candidates, candidates, mImpl->splitNewVertexCount);
    CheckCuda(cudaDeviceSynchronize(), "SumAcceptedSplitCost");
    const uint32_t newVertices = *mImpl->splitNewVertexCount;
    mImpl->EnsureAppendCapacity(newVertices, newVertices * 3u, newVertices * 2u);
    const uint32_t verticesBeforeExecute = *mImpl->vertexCount;
    mark = std::chrono::steady_clock::now();
    ExecuteSplitCandidates<<<candidateBlocks, threads>>>(
        mImpl->vertices, mImpl->edges, mImpl->faces, mImpl->candidates,
        candidates, mImpl->vertexCount, mImpl->edgeCount, mImpl->faceCount,
        mImpl->staleRejected, mImpl->gpuPatches, mImpl->gpuPatchCount,
        mImpl->projectionApplied, mImpl->projectionFailed);
    CheckCuda(cudaDeviceSynchronize(), "ExecuteSplitCandidates");
    report.executeMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - mark)
                           .count();

    report.staleRejected = *mImpl->staleRejected;
    report.projectionApplied = *mImpl->projectionApplied;
    report.projectionFailed = *mImpl->projectionFailed;
    // Keep report.splits semantics stable: count inserted split vertices.
    // For an N-segment MultiSplit this contributes N-1.
    report.acceptedCount = *mImpl->vertexCount - verticesBeforeExecute;
    report.globalMemoryBytes = mImpl->MemoryBytes();
    report.dynamicSharedMemoryBytes = 0;
    report.totalMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - totalStart)
                         .count();
    return true;
  } catch (const std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

bool GlobalSplitBackend::RunCollapsePass(float collapseRatio, GlobalCollapseReport &report,
                                         std::string *error) {
  report = {};
  const auto totalStart = std::chrono::steady_clock::now();
  try {
    if (!mImpl->vertices) throw std::runtime_error("backend not initialized");
    *mImpl->candidateCount = 0;
    *mImpl->acceptedCount = 0;
    *mImpl->roundAccepted = 0;
    *mImpl->activeCount = 0;
    *mImpl->nextActiveCount = 0;
    *mImpl->staleRejected = 0;
    *mImpl->semanticRejected = 0;
    *mImpl->qualityRejected = 0;
    *mImpl->collapseTopologyRejected = 0;

    const uint32_t edgeCount = *mImpl->edgeCount;
    const uint32_t faceCount = *mImpl->faceCount;
    constexpr uint32_t threads = 256;
    const uint32_t edgeBlocks = (edgeCount + threads - 1u) / threads;
    CheckCuda(cudaMemset(mImpl->edgeHashKeys, 0,
                         sizeof(unsigned long long) * mImpl->edgeHashCapacity),
              "reset collapse edge hash");
    BuildEdgeHash<<<edgeBlocks, threads>>>(mImpl->edges, edgeCount, mImpl->edgeHashKeys,
                                           mImpl->edgeHashCapacity - 1u);
    CheckCuda(cudaDeviceSynchronize(), "BuildEdgeHash collapse");

    auto adjacencyMark = std::chrono::steady_clock::now();
    mImpl->BuildVertexEdgeAdjacency();
    report.adjacencyMs = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - adjacencyMark).count();
    auto mark = std::chrono::steady_clock::now();
    GenerateCollapseCandidates<<<edgeBlocks, threads>>>(
        mImpl->vertices, mImpl->edges, mImpl->faces, edgeCount, faceCount, collapseRatio,
        mImpl->edgeHashKeys, mImpl->edgeHashCapacity - 1u,
        mImpl->vertexEdgeOffsets, mImpl->vertexEdgeIds,
        mImpl->gpuPatches, mImpl->gpuPatchCount, mImpl->collapseCandidates,
        mImpl->candidateCapacity, mImpl->candidateCount, mImpl->collapseTopologyRejected,
        mImpl->semanticRejected, mImpl->qualityRejected);
    CheckCuda(cudaDeviceSynchronize(), "GenerateCollapseCandidates");
    report.candidateMs = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - mark).count();

    const uint32_t candidates = std::min(*mImpl->candidateCount, mImpl->candidateCapacity);
    report.candidateCount = candidates;
    report.topologyRejected = *mImpl->collapseTopologyRejected;
    report.semanticRejected = *mImpl->semanticRejected;
    report.qualityRejected = *mImpl->qualityRejected;
    if (candidates == 0) {
      report.globalMemoryBytes = mImpl->MemoryBytes();
      report.totalMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - totalStart).count();
      return true;
    }

    const uint32_t candidateBlocks = (candidates + threads - 1u) / threads;
    CheckCuda(cudaMemset(mImpl->vertexBlocked, 0,
                         sizeof(uint8_t) * mImpl->vertexCapacity),
              "reset collapse blocked vertices");
    InitActiveQueue<<<candidateBlocks, threads>>>(mImpl->activeQueueA, candidates);
    CheckCuda(cudaDeviceSynchronize(), "InitCollapseActiveQueue");
    *mImpl->activeCount = candidates;
    uint32_t *active = mImpl->activeQueueA;
    uint32_t *next = mImpl->activeQueueB;
    mark = std::chrono::steady_clock::now();
    constexpr int kMaxClaimRounds = 4;
    int claimRounds = 0;
    for (; claimRounds < kMaxClaimRounds && *mImpl->activeCount > 0; ++claimRounds) {
      const uint32_t activeNow = *mImpl->activeCount;
      report.activeItemsScanned += activeNow;
      const uint32_t activeBlocks = (activeNow + threads - 1u) / threads;
      CheckCuda(cudaMemset(mImpl->splitVertexWinner, 0,
                           sizeof(unsigned long long) * mImpl->vertexCapacity),
                "reset collapse vertex winners");
      *mImpl->roundAccepted = 0;
      ClaimCollapseCandidates<<<activeBlocks, threads>>>(
          mImpl->collapseCandidates, active, activeNow, mImpl->edges,
          mImpl->vertexEdgeOffsets, mImpl->vertexEdgeIds,
          mImpl->splitVertexWinner);
      CheckCuda(cudaDeviceSynchronize(), "ClaimCollapseCandidates");
      ResolveCollapseCandidates<<<activeBlocks, threads>>>(
          mImpl->collapseCandidates, active, activeNow, mImpl->edges,
          mImpl->vertexEdgeOffsets, mImpl->vertexEdgeIds,
          mImpl->splitVertexWinner, mImpl->vertexBlocked,
          mImpl->acceptedCount, mImpl->roundAccepted);
      CheckCuda(cudaDeviceSynchronize(), "ResolveCollapseCandidates");
      if (*mImpl->roundAccepted == 0) break;
      *mImpl->nextActiveCount = 0;
      CompactCollapseCandidates<<<activeBlocks, threads>>>(
          mImpl->collapseCandidates, active, activeNow, mImpl->edges,
          mImpl->vertexEdgeOffsets, mImpl->vertexEdgeIds,
          mImpl->vertexBlocked, next, mImpl->nextActiveCount);
      CheckCuda(cudaDeviceSynchronize(), "CompactCollapseCandidates");
      *mImpl->activeCount = *mImpl->nextActiveCount;
      std::swap(active, next);
    }
    report.schedulerRounds = uint32_t(claimRounds);
    report.claimMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - mark).count();

    const uint32_t accepted = *mImpl->acceptedCount;
    mark = std::chrono::steady_clock::now();
    ExecuteCollapseCandidates<<<candidateBlocks, threads>>>(
        mImpl->vertices, mImpl->edges, mImpl->faces, mImpl->collapseCandidates,
        candidates, edgeCount, faceCount, mImpl->vertexEdgeOffsets,
        mImpl->vertexEdgeIds, mImpl->staleRejected);
    CheckCuda(cudaDeviceSynchronize(), "ExecuteCollapseCandidates");
    report.executeMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - mark).count();
    report.staleRejected = *mImpl->staleRejected;
    report.acceptedCount = accepted >= report.staleRejected ? accepted - report.staleRejected : 0;
    report.globalMemoryBytes = mImpl->MemoryBytes();
    report.dynamicSharedMemoryBytes = 0;
    report.totalMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - totalStart).count();
    return true;
  } catch (const std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

bool GlobalSplitBackend::RunFlipPass(float minQualityGain, GlobalFlipReport &report,
                                     std::string *error) {
  report = {};
  const auto totalStart = std::chrono::steady_clock::now();
  try {
    if (!mImpl->vertices) throw std::runtime_error("backend not initialized");
    *mImpl->candidateCount = 0;
    *mImpl->acceptedCount = 0;
    *mImpl->roundAccepted = 0;
    *mImpl->activeCount = 0;
    *mImpl->nextActiveCount = 0;
    *mImpl->staleRejected = 0;
    *mImpl->capacityRejected = 0;
    *mImpl->semanticRejected = 0;

    const uint32_t edgeCount = *mImpl->edgeCount;
    const uint32_t faceCount = *mImpl->faceCount;
    constexpr uint32_t threads = 256;
    const uint32_t edgeBlocks = (edgeCount + threads - 1u) / threads;
    auto mark = std::chrono::steady_clock::now();
    CheckCuda(cudaMemset(mImpl->edgeHashKeys, 0,
                         sizeof(unsigned long long) * mImpl->edgeHashCapacity),
              "reset edge hash");
    BuildEdgeHash<<<edgeBlocks, threads>>>(mImpl->edges, edgeCount,
                                           mImpl->edgeHashKeys,
                                           mImpl->edgeHashCapacity - 1u);
    CheckCuda(cudaDeviceSynchronize(), "BuildEdgeHash");
    GenerateFlipCandidates<<<edgeBlocks, threads>>>(
        mImpl->vertices, mImpl->edges, mImpl->faces, edgeCount, faceCount,
        mImpl->edgeHashKeys, mImpl->edgeHashCapacity - 1u, minQualityGain,
        mImpl->flipCandidates, mImpl->candidateCapacity,
        mImpl->candidateCount, mImpl->semanticRejected, mImpl->capacityRejected);
    CheckCuda(cudaDeviceSynchronize(), "GenerateFlipCandidates");
    report.candidateMs = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - mark).count();

    const uint32_t candidates = std::min(*mImpl->candidateCount, mImpl->candidateCapacity);
    report.candidateCount = candidates;
    report.semanticRejected = *mImpl->semanticRejected;
    if (candidates == 0) {
      report.globalMemoryBytes = mImpl->MemoryBytes();
      report.totalMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - totalStart).count();
      return true;
    }

    mark = std::chrono::steady_clock::now();
    const uint32_t candidateBlocks = (candidates + threads - 1u) / threads;
    CheckCuda(cudaMemset(mImpl->edgeBlocked, 0, sizeof(uint8_t) * mImpl->edgeCapacity),
              "reset flip blocked edges");
    CheckCuda(cudaMemset(mImpl->faceBlocked, 0, sizeof(uint8_t) * mImpl->faceCapacity),
              "reset flip blocked faces");
    InitActiveQueue<<<candidateBlocks, threads>>>(mImpl->activeQueueA, candidates);
    CheckCuda(cudaDeviceSynchronize(), "InitFlipActiveQueue");
    *mImpl->activeCount = candidates;
    uint32_t *active = mImpl->activeQueueA;
    uint32_t *next = mImpl->activeQueueB;
    constexpr int kMaxClaimRounds = 4;
    int claimRounds = 0;
    for (; claimRounds < kMaxClaimRounds && *mImpl->activeCount > 0; ++claimRounds) {
      const uint32_t activeNow = *mImpl->activeCount;
      report.activeItemsScanned += activeNow;
      const uint32_t activeBlocks = (activeNow + threads - 1u) / threads;
      CheckCuda(cudaMemset(mImpl->edgeOwner, 0xff, sizeof(uint32_t) * mImpl->edgeCapacity),
                "reset flip edge owners");
      CheckCuda(cudaMemset(mImpl->faceOwner, 0xff, sizeof(uint32_t) * mImpl->faceCapacity),
                "reset flip face owners");
      *mImpl->roundAccepted = 0;
      ClaimFlipCandidatesActive<<<activeBlocks, threads>>>(
          mImpl->flipCandidates, mImpl->faces, active, activeNow,
          mImpl->edgeOwner, mImpl->faceOwner);
      CheckCuda(cudaDeviceSynchronize(), "ClaimFlipCandidatesActive");
      ResolveFlipCandidatesActive<<<activeBlocks, threads>>>(
          mImpl->flipCandidates, mImpl->faces, active, activeNow,
          mImpl->edgeOwner, mImpl->faceOwner, mImpl->edgeBlocked, mImpl->faceBlocked,
          mImpl->acceptedCount, mImpl->roundAccepted);
      CheckCuda(cudaDeviceSynchronize(), "ResolveFlipCandidatesActive");
      if (*mImpl->roundAccepted == 0) break;
      *mImpl->nextActiveCount = 0;
      CompactFlipCandidates<<<activeBlocks, threads>>>(
          mImpl->flipCandidates, mImpl->faces, active, activeNow,
          mImpl->edgeBlocked, mImpl->faceBlocked, next, mImpl->nextActiveCount);
      CheckCuda(cudaDeviceSynchronize(), "CompactFlipCandidates");
      *mImpl->activeCount = *mImpl->nextActiveCount;
      std::swap(active, next);
    }
    report.schedulerRounds = uint32_t(claimRounds);
    report.claimMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - mark).count();

    mark = std::chrono::steady_clock::now();
    ExecuteFlipCandidates<<<candidateBlocks, threads>>>(
        mImpl->vertices, mImpl->edges, mImpl->faces, mImpl->flipCandidates,
        candidates, mImpl->staleRejected);
    CheckCuda(cudaDeviceSynchronize(), "ExecuteFlipCandidates");
    report.executeMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - mark).count();
    report.staleRejected = *mImpl->staleRejected;
    const uint32_t accepted = *mImpl->acceptedCount;
    report.acceptedCount = accepted >= report.staleRejected ? accepted - report.staleRejected : 0;
    report.globalMemoryBytes = mImpl->MemoryBytes();
    report.dynamicSharedMemoryBytes = 0;
    report.totalMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - totalStart).count();
    return true;
  } catch (const std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

bool GlobalSplitBackend::Validate(GlobalTopologyValidation &validation,
                                  std::string *error) const {
  validation = {};
  try {
    if (!mImpl->vertices) throw std::runtime_error("backend not initialized");
    CheckCuda(cudaMemset(mImpl->validation, 0, sizeof(ValidationCounters)),
              "reset validation");
    constexpr uint32_t threads = 256;
    const uint32_t nv = *mImpl->vertexCount;
    const uint32_t ne = *mImpl->edgeCount;
    const uint32_t nf = *mImpl->faceCount;
    const uint32_t faceBlocks = (nf + threads - 1u) / threads;
    const uint32_t edgeBlocks = (ne + threads - 1u) / threads;

    ValidateFacesKernel<<<faceBlocks, threads>>>(mImpl->vertices, nv, mImpl->edges, ne,
                                                 mImpl->faces, nf, mImpl->validation);
    CheckCuda(cudaDeviceSynchronize(), "ValidateFacesKernel");
    ValidateEdgesKernel<<<edgeBlocks, threads>>>(mImpl->vertices, nv, mImpl->edges, ne,
                                                 mImpl->faces, nf, mImpl->validation);
    CheckCuda(cudaDeviceSynchronize(), "ValidateEdgesKernel");

    const ValidationCounters c = *mImpl->validation;
    validation.invalidVertexReference = c.invalidVertexReference;
    validation.invalidFaceReference = c.invalidFaceReference;
    validation.degenerateFace = c.degenerateFace;
    validation.zeroAreaFace = c.zeroAreaFace;
    validation.edgeFaceMismatch = c.edgeFaceMismatch;
    validation.staleCandidate = c.staleCandidate;
    return validation.ok();
  } catch (const std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

bool GlobalSplitBackend::Export(SemanticMesh &mesh, std::string *error) const {
  try {
    if (!mImpl->vertices) throw std::runtime_error("backend not initialized");
    CheckCuda(cudaDeviceSynchronize(), "Export synchronize");
    const uint32_t nv = *mImpl->vertexCount;
    const uint32_t nf = *mImpl->faceCount;

    SemanticMesh out;
    out.patches = mImpl->patches;
    std::vector<uint32_t> vertexMap(nv, kInvalid);
    for (uint32_t v = 0; v < nv; ++v) {
      const GVertex &gv = mImpl->vertices[v];
      if (!gv.alive) continue;
      const int mapped = out.addVertex({gv.x, gv.y, gv.z}, gv.patchId,
                                       VertexConstraint(gv.constraint));
      out.targetLength[mapped] = gv.targetLength;
      vertexMap[v] = uint32_t(mapped);
    }

    for (uint32_t f = 0; f < nf; ++f) {
      const GFace &gf = mImpl->faces[f];
      if (!gf.alive) continue;
      if (gf.v0 >= nv || gf.v1 >= nv || gf.v2 >= nv ||
          vertexMap[gf.v0] == kInvalid || vertexMap[gf.v1] == kInvalid ||
          vertexMap[gf.v2] == kInvalid)
        throw std::runtime_error("export references a dead vertex");
      PatchType type = PatchType::Unknown;
      if (gf.patchId < out.patches.size()) type = out.patches[gf.patchId].type;
      out.addFace(int(vertexMap[gf.v0]), int(vertexMap[gf.v1]), int(vertexMap[gf.v2]),
                  gf.patchId, type);
    }
    out.rebuildTopology();
    std::string validationError;
    if (!out.validate(&validationError))
      throw std::runtime_error("export validation failed: " + validationError);
    mesh = std::move(out);
    return true;
  } catch (const std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

uint32_t GlobalSplitBackend::VertexCount() const {
  return mImpl && mImpl->vertexCount ? *mImpl->vertexCount : 0;
}

uint32_t GlobalSplitBackend::EdgeCount() const {
  return mImpl && mImpl->edgeCount ? *mImpl->edgeCount : 0;
}

uint32_t GlobalSplitBackend::FaceCount() const {
  return mImpl && mImpl->faceCount ? *mImpl->faceCount : 0;
}

size_t GlobalSplitBackend::GlobalMemoryBytes() const {
  return mImpl ? mImpl->MemoryBytes() : 0;
}

} // namespace cad_adaptive::global
