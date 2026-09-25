// SPDX-License-Identifier: GPL-2.0-or-later
// Refinement templates and isotropic operator rules follow VCGLib:
// https://github.com/cnr-isti-vclab/vcglib (ISTI-CNR Visual Computing Lab).
// CUDA conflict selection, storage and orchestration are implemented here.
#include "cad_adaptive/RawCudaRemesher.h"
#include "cad_adaptive/RemeshMetrics.h"
#include "cad_adaptive/gpu/ReferenceSurfaceGpu.cuh"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifndef CAD_ADAPTIVE_RAW_PROFILE_COLLAPSE_STAGES
#define CAD_ADAPTIVE_RAW_PROFILE_COLLAPSE_STAGES 0
#endif

namespace cad_adaptive {
namespace raw {
using namespace gpu;
void checked(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
thread_local cudaStream_t executionStream=nullptr;
cudaError_t streamCopy(void *dst,const void *src,size_t bytes,cudaMemcpyKind kind) {
  if(!executionStream)return cudaMemcpy(dst,src,bytes,kind);
  checked(cudaMemcpyAsync(dst,src,bytes,kind,executionStream));
  checked(cudaStreamSynchronize(executionStream));
  return cudaSuccess;
}
cudaError_t streamZero(void *dst,int value,size_t bytes) {
  if(!executionStream)return cudaMemset(dst,value,bytes);
  checked(cudaMemsetAsync(dst,value,bytes,executionStream));
  return cudaSuccess;
}
struct StreamScope {
  cudaStream_t previous=executionStream, owned=nullptr;
  explicit StreamScope(bool independent) {if(independent){checked(cudaStreamCreateWithFlags(&owned,cudaStreamNonBlocking));executionStream=owned;}}
  ~StreamScope(){if(owned){cudaStreamSynchronize(owned);cudaStreamDestroy(owned);}executionStream=previous;}
};
void sync() {
  checked(cudaGetLastError());
  checked(cudaStreamSynchronize(executionStream));
}
// Per-call, per-thread scratch cache. CUDA allocation/free synchronization is
// paid once per capacity class rather than once per topology pass.
struct DeviceArena {
  struct Block {void *p; size_t bytes;};
  std::vector<Block> available;
  size_t allocated=0,limit=0;
  static size_t capacity(size_t bytes) {size_t n=256;while(n<bytes)n*=2;return n;}
  void *acquire(size_t bytes) {
    for(size_t i=0;i<available.size();++i) if(available[i].bytes==bytes) {
      void *p=available[i].p;available[i]=available.back();available.pop_back();return p;
    }
    if(limit && allocated+bytes>limit) {
      for(auto b:available){checked(cudaFree(b.p));allocated-=b.bytes;}available.clear();
      if(allocated+bytes>limit)throw std::runtime_error("raw CUDA task workspace budget exceeded");
    }
    void *p=nullptr;checked(cudaMalloc(&p,bytes));allocated+=bytes;return p;
  }
  void release(void *p,size_t bytes) {available.push_back({p,bytes});}
  ~DeviceArena() {for(auto b:available)cudaFree(b.p);}
};
thread_local DeviceArena *activeArena=nullptr;
struct ArenaScope {
  DeviceArena *previous;
  explicit ArenaScope(DeviceArena &a):previous(activeArena){activeArena=&a;}
  ~ArenaScope(){activeArena=previous;}
};
template <class T> struct Buffer {
  T *p = nullptr;
  size_t n = 0;
  size_t bytes = 0;
  DeviceArena *arena=activeArena;
  explicit Buffer(size_t size = 0) : n(size) {
    if (n) {
      bytes=DeviceArena::capacity(n*sizeof(T));
      if(arena)p=static_cast<T*>(arena->acquire(bytes));
      else checked(cudaMalloc(&p,bytes));
    }
  }
  explicit Buffer(const std::vector<T> &v) : Buffer(v.size()) {
    if (n)
      checked(streamCopy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Buffer() {
    if (p) {
      if(arena)arena->release(p,bytes);
      else cudaFree(p);
    }
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  std::vector<T> read() const {
    std::vector<T> v(n);
    if (n)
      checked(streamCopy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
struct Vertex {
  float3 p;
  int constraint;
  float target;
};
struct Triangle {
  int v[3];
  int patch;
};
struct Edge {
  int a, b, f0, f1, feature;
};
struct Candidate {
  int keep = -1, remove = -1, c = -1, d = -1;
  float3 p;
  float destSizing = 0.f;
};
struct MeshView {
  Vertex *v;
  Triangle *f;
  Edge *e;
  int *offsets;
  int *ids;
  int *faceEdges;
  int nv, nf, ne;
  bool freezeBoundary=false;
};
uint64_t key(int a, int b) {
  if (a > b)
    std::swap(a, b);
  return (uint64_t(a) << 32) | uint32_t(b);
}
float3 point(Vec3 p) { return make_float3(p.x, p.y, p.z); }
__global__ void updateTargets(Vertex *v, const int *patchIds, int n, ReferenceSurfaceGpu ref) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    v[i].target = sizingAt(ref, v[i].p,patchIds[i]);
}
thread_local bool freezeBoundary=false;
struct DeviceMesh {
  Buffer<Vertex> vertices;
  Buffer<Triangle> faces;
  Buffer<Edge> edges;
  Buffer<int> offsets, ids, faceEdges;
  static std::vector<Vertex> verts(const SemanticMesh &m) {
    std::vector<Vertex> out;out.reserve(m.vertexCount());
    for (int i = 0; i < m.vertexCount(); ++i)
      out.push_back(
          {point(m.position(i)), m.vertexConstraint[i], m.targetLength[i]});
    return out;
  }
  static std::vector<Triangle> tris(const SemanticMesh &m) {
    std::vector<Triangle> out;out.reserve(m.faceCount());
    for (int f = 0; f < m.faceCount(); ++f)
      out.push_back({{int(m.i0[f]), int(m.i1[f]), int(m.i2[f])},int(m.facePatchId[f])});
    return out;
  }
  static std::vector<Edge> eds(const SemanticMesh &m) {
    std::vector<Edge> out;out.reserve(m.edges.size());
    for (auto e : m.edges)
      out.push_back({int(e.v0), int(e.v1), e.face0, e.face1,
                     bool(e.flags & (EdgeSharp | EdgeMeshBoundary | EdgePatchBoundary))});
    return out;
  }
  struct HostAdjacency {
    std::vector<int> offsets;
    std::vector<int> ids;
  };
  static HostAdjacency adjacencyPack(const SemanticMesh &m) {
    HostAdjacency pack;
    pack.offsets.assign(m.vertexCount()+1,0);
    for(const auto &e:m.edges){++pack.offsets[e.v0+1];++pack.offsets[e.v1+1];}
    for(size_t i=1;i<pack.offsets.size();++i)pack.offsets[i]+=pack.offsets[i-1];
    pack.ids.resize(pack.offsets.back());
    auto cursor=pack.offsets;
    for(int i=0;i<int(m.edges.size());++i) {
      const auto &e=m.edges[i];
      pack.ids[cursor[e.v0]++]=i;
      pack.ids[cursor[e.v1]++]=i;
    }
    return pack;
  }  static std::vector<int> fe(const SemanticMesh &m) {
    if(m.faceEdgeIds.size()==size_t(m.faceCount())*3 &&
       std::all_of(m.faceEdgeIds.begin(),m.faceEdgeIds.end(),[](int id){return id>=0;}))
      return m.faceEdgeIds;
    std::vector<int> out(size_t(m.faceCount())*3,-1);
    for(int edgeId=0;edgeId<int(m.edges.size());++edgeId) {
      const auto &e=m.edges[edgeId];
      const uint64_t edgeKey=key(int(e.v0),int(e.v1));
      const int incident[2]={e.face0,e.face1};
      for(int q=0;q<2;++q) {
        const int f=incident[q];
        if(f<0)continue;
        const auto t=m.face(f);
        for(int k=0;k<3;++k) if(key(t[k],t[(k+1)%3])==edgeKey) {
          out[size_t(f)*3+k]=edgeId;break;
        }
      }
    }
    for(int id:out)if(id<0)throw std::runtime_error("missing face edge");
    return out;
  }  DeviceMesh(const SemanticMesh &m, ReferenceSurfaceGpu, HostAdjacency pack, bool includeFaceEdges)
      : vertices(verts(m)), faces(tris(m)), edges(eds(m)), offsets(pack.offsets),
        ids(pack.ids), faceEdges(includeFaceEdges ? fe(m) : std::vector<int>{}) {}
  explicit DeviceMesh(const SemanticMesh &m, ReferenceSurfaceGpu ref, bool includeFaceEdges=true)
      : DeviceMesh(m,ref,adjacencyPack(m),includeFaceEdges) {}
  MeshView view() {
    return {vertices.p,  faces.p,         edges.p,      offsets.p,   ids.p,
            faceEdges.p, int(vertices.n), int(faces.n), int(edges.n), freezeBoundary};
  }
};
void refreshHostTargets(SemanticMesh &mesh, ReferenceSurfaceGpu ref) {
  auto host=DeviceMesh::verts(mesh);
  Buffer<Vertex> device(host);
  std::vector<int> patchIds(mesh.vertexPatchId.begin(),mesh.vertexPatchId.end());
  Buffer<int> devicePatchIds(patchIds);
  updateTargets<<<(mesh.vertexCount()+127)/128,128,0,executionStream>>>(device.p,devicePatchIds.p,mesh.vertexCount(),ref);
  sync();
  const auto refreshed=device.read();
  for(int i=0;i<mesh.vertexCount();++i)mesh.targetLength[i]=refreshed[i].target;
}
__device__ int other(Edge e, int v) { return e.a == v ? e.b : e.a; }
__device__ int third(Triangle t, int a, int b) {
  for (int k = 0; k < 3; ++k)
    if (t.v[k] != a && t.v[k] != b)
      return t.v[k];
  return -1;
}
__device__ int vertexPatch(MeshView m,int v) {
  for(int i=m.offsets[v];i<m.offsets[v+1];++i) {
    const Edge e=m.e[m.ids[i]];
    if(e.f0>=0)return m.f[e.f0].patch;
    if(e.f1>=0)return m.f[e.f1].patch;
  }
  return 0;
}
__device__ bool neighbor(MeshView m, int a, int b) {
  for (int i = m.offsets[a]; i < m.offsets[a + 1]; ++i)
    if (other(m.e[m.ids[i]], a) == b)
      return true;
  return false;
}
__device__ float dist2(float3 a, float3 b) {
  auto d = sub3(a, b);
  return dot3(d, d);
}
__device__ bool movable(MeshView m, int v, int edge) {
  if (m.v[v].constraint >= 4)
    return false;
  int count = 0;
  float3 directions[2];
  for (int i = m.offsets[v]; i < m.offsets[v + 1]; ++i) {
    auto e = m.e[m.ids[i]];
    if (!e.feature)
      continue;
    if (count == 2)
      return false;
    directions[count++] = sub3(m.v[other(e, v)].p, m.v[v].p);
  }
  if (!count)
    return true;
  if (!m.e[edge].feature || count != 2)
    return false;
  return dot3(directions[0], directions[1]) <=
         -.9f * sqrtf(dot3(directions[0], directions[0]) *
                      dot3(directions[1], directions[1]));
}
// An incident triangle is reached through two edges of a vertex star.
// Assign it to exactly one, retaining the complete safety test once.
__device__ bool ownsFace(Triangle t,Edge e,int v) {
  int smallest=2147483647;
  for(int k=0;k<3;++k)if(t.v[k]!=v)smallest=min(smallest,t.v[k]);
  return other(e,v)==smallest;
}
__device__ float qualityVcgKnownNormal(float3 a,float3 b,float3 c,float3 n) {
  const auto ab=sub3(a,b),bc=sub3(b,c),ca=sub3(c,a);
  const float d=fmaxf(dot3(ab,ab),fmaxf(dot3(bc,bc),dot3(ca,ca)));
  return d>0?sqrtf(dot3(n,n))/d:0;
}
struct ChangedFaceStats {
  unsigned long long calls=0;
  unsigned long long mergedSkip=0;
  unsigned long long qualityReject=0;
  unsigned long long normalReject=0;
  unsigned long long lengthReject=0;
  unsigned long long referenceReject=0;
  unsigned long long safe=0;
};
__device__ bool changedFaceSafe(MeshView m, Triangle t, int a, int b,
                                float3 dest, float high,
                                ReferenceSurfaceGpu ref, bool relaxed, float destSizing,
                                bool checkReference, ChangedFaceStats *stats) {
  if(stats) atomicAdd(&stats->calls,1ull);
  float3 before[3], after[3];
  bool hasA = false, hasB = false;
  int changedIndex=-1;
  for (int k = 0; k < 3; ++k) {
    int v = t.v[k];
    before[k] = m.v[v].p;
    const bool changed=(v==a || v==b);
    after[k] = changed ? dest : before[k];
    if(changed)changedIndex=k;
    hasA |= v == a;
    hasB |= v == b;
  }
  if (a != b && hasA && hasB) {
    if(stats) atomicAdd(&stats->mergedSkip,1ull);
    return true;
  }
  const auto n0 = normal3(before[0], before[1], before[2]),
             n1 = normal3(after[0], after[1], after[2]);
  if (qualityVcgKnownNormal(after[0], after[1], after[2],n1) <=
      fmaxf(1.e-8f, .5f * qualityVcgKnownNormal(before[0], before[1], before[2],n0))) {
    if(stats) atomicAdd(&stats->qualityReject,1ull);
    return false;
  }
  if (dot3(n0, n1) < .7f * sqrtf(dot3(n0, n0) * dot3(n1, n1))) {
    if(stats) atomicAdd(&stats->normalReject,1ull);
    return false;
  }
  float sizingAB=-1.f,sizingBC=-1.f,sizingCA=-1.f;
  if (!relaxed)
    for (int k = 0; k < 3; ++k)
      if (t.v[k] != a && t.v[k] != b) {
        float midSizing=ref.regularLength;
        if(ref.sizingCount) {
          midSizing=sizingAt(ref,mul3(add3(dest,after[k]),.5f),t.patch);
          if((changedIndex==0&&k==1)||(changedIndex==1&&k==0))sizingAB=midSizing;
          else if((changedIndex==1&&k==2)||(changedIndex==2&&k==1))sizingBC=midSizing;
          else if((changedIndex==2&&k==0)||(changedIndex==0&&k==2))sizingCA=midSizing;
        }
        if (dist2(dest, after[k]) >
              high * high *
                  (ref.sizingCount
                       ? powf(fminf(midSizing,
                                    .5f * (destSizing + m.v[t.v[k]].target)) /
                                  ref.regularLength,
                              2.f)
                       : 1.f)) {
          if(stats) atomicAdd(&stats->lengthReject,1ull);
          return false;
        }
      }
  if(!checkReference) { if(stats) atomicAdd(&stats->safe,1ull); return true; }
  const bool refSafe=(a==b)
      ? referenceFaceSafeCached(ref,t.patch,after[0],after[1],after[2],sizingAB,sizingBC,sizingCA)
      : referenceChangedFaceSafeCached(ref,t.patch,after[0],after[1],after[2],changedIndex,
                                       sizingAB,sizingBC,sizingCA);
  if(!refSafe) { if(stats) atomicAdd(&stats->referenceReject,1ull); return false; }
  if(stats) atomicAdd(&stats->safe,1ull);
  return true;
}__device__ bool changedFaceShapeSafe(MeshView m,Triangle t,int a,int b,float3 dest) {
  float3 before[3],after[3];
  bool hasA=false,hasB=false;
  for(int k=0;k<3;++k) {
    const int v=t.v[k];
    before[k]=m.v[v].p;
    after[k]=(v==a || v==b)?dest:before[k];
    hasA|=v==a; hasB|=v==b;
  }
  if(a!=b && hasA && hasB)return true;
  const auto n0=normal3(before[0],before[1],before[2]);
  const auto n1=normal3(after[0],after[1],after[2]);
  if(qualityVcgKnownNormal(after[0],after[1],after[2],n1)<=
     fmaxf(1.e-8f,.5f*qualityVcgKnownNormal(before[0],before[1],before[2],n0)))return false;
  return dot3(n0,n1)>=.7f*sqrtf(dot3(n0,n0)*dot3(n1,n1));
}
__device__ int findEdgeId(MeshView m,int a,int b) {
  for(int i=m.offsets[a];i<m.offsets[a+1];++i) {
    const int edgeId=m.ids[i];
    if(other(m.e[edgeId],a)==b)return edgeId;
  }
  return -1;
}
__device__ bool changedFaceLengthSafe(MeshView m,Triangle t,int a,int b,float3 dest,float high,
                                      ReferenceSurfaceGpu ref,bool relaxed,float destSizing) {
  if(relaxed)return true;
  bool hasA=false,hasB=false;
  for(int k=0;k<3;++k){hasA|=t.v[k]==a;hasB|=t.v[k]==b;}
  if(a!=b && hasA && hasB)return true;
  for(int k=0;k<3;++k) if(t.v[k]!=a && t.v[k]!=b) {
    const float3 otherP=m.v[t.v[k]].p;
    const float edgeDist2=dist2(dest,otherP);
    if(!ref.sizingCount) {
      if(edgeDist2>high*high)return false;
      continue;
    }
    const float avgTarget=.5f*(destSizing+m.v[t.v[k]].target);
    const float maxScale=fminf(ref.regularLength,avgTarget)/ref.regularLength;
    if(edgeDist2>high*high*maxScale*maxScale)return false;
    const float midSizing=sizingAt(ref,mul3(add3(dest,otherP),.5f),t.patch);
    const float scale=fminf(midSizing,avgTarget)/ref.regularLength;
    if(edgeDist2>high*high*scale*scale)return false;
  }
  return true;
}__device__ float splitEdgeQuality(MeshView m,Edge e,float3 splitPoint) {
  float score=FLT_MAX;
  for(int side=0;side<2;++side) {
    const int f=side==0?e.f0:e.f1;
    if(f<0)continue;
    const auto face=m.f[f];
    int opposite=-1;
    for(int k=0;k<3;++k)if(face.v[k]!=e.a && face.v[k]!=e.b)
      opposite=face.v[k];
    if(opposite<0)continue;
    const auto c=m.v[opposite].p;
    score=fminf(score,qualityVcg(m.v[e.a].p,splitPoint,c));
    score=fminf(score,qualityVcg(splitPoint,m.v[e.b].p,c));
  }
  return score;
}
__global__ void splitVertices(MeshView m, Vertex *out, int *marked, float high,
                              ReferenceSurfaceGpu ref,bool optimizePoint) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m.nv)
    out[i] = m.v[i];
  if (i >= m.ne)
    return;
  auto e = m.e[i];
  const auto midpoint = mul3(add3(m.v[e.a].p, m.v[e.b].p), .5f);
  float target = sizingAt(ref, midpoint,m.f[e.f0].patch);
  high *= ref.sizingCount
              ? fminf(target, .5f * (m.v[e.a].target + m.v[e.b].target)) /
                    ref.regularLength
              : 1.f;
  marked[i] = !(m.freezeBoundary && (e.f1<0 || (e.feature && m.v[e.a].constraint>=4 && m.v[e.b].constraint>=4))) && dist2(m.v[e.a].p, m.v[e.b].p) > high * high;
  float3 splitPoint=midpoint;
  if(marked[i] && optimizePoint) {
    float best=splitEdgeQuality(m,e,midpoint);
    const float trials[4]={.35f,.425f,.575f,.65f};
    for(float t:trials) {
      const auto trial=add3(m.v[e.a].p,mul3(sub3(m.v[e.b].p,m.v[e.a].p),t));
      const float score=splitEdgeQuality(m,e,trial);
      if(score>best) {best=score;splitPoint=trial;}
    }
    target=sizingAt(ref,splitPoint,m.f[e.f0].patch);
  }
  out[m.nv + i] = {splitPoint, e.feature ? 2 : 1, target};
}
__global__ void splitFaces(MeshView m, const Vertex *verts, const int *marked,
                           Triangle *out) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= m.nf)
    return;
  auto t = m.f[f];
  int vv[6] = {t.v[0],
               t.v[1],
               t.v[2],
               m.nv + m.faceEdges[3 * f],
               m.nv + m.faceEdges[3 * f + 1],
               m.nv + m.faceEdges[3 * f + 2]};
  int mask = marked[m.faceEdges[3 * f]] + 2 * marked[m.faceEdges[3 * f + 1]] +
             4 * marked[m.faceEdges[3 * f + 2]];
  const int counts[8] = {1, 2, 2, 3, 2, 3, 3, 4};
  const int tab[8][4][3] = {{{0, 1, 2}},
                            {{0, 3, 2}, {3, 1, 2}},
                            {{0, 1, 4}, {0, 4, 2}},
                            {{3, 1, 4}, {0, 3, 2}, {4, 2, 3}},
                            {{0, 1, 5}, {5, 1, 2}},
                            {{0, 3, 5}, {3, 1, 5}, {2, 5, 1}},
                            {{2, 5, 4}, {0, 1, 5}, {4, 5, 1}},
                            {{3, 4, 5}, {0, 3, 5}, {3, 1, 4}, {5, 4, 2}}};
  Triangle local[4];
  for (int q = 0; q < 4; ++q) {
    local[q].patch=t.patch;
    for (int k = 0; k < 3; ++k)
      local[q].v[k] = q < counts[mask] ? vv[tab[mask][q][k]] : -1;
  }
  if (mask == 3 || mask == 5 || mask == 6) {
    int a0 = mask == 5 ? 3 : 0, a1 = mask == 5 ? 2 : 4, b0 = mask == 3 ? 3 : 5,
        b1 = mask == 3 ? 2 : 1;
    if (dist2(verts[vv[a0]].p, verts[vv[a1]].p) <
        dist2(verts[vv[b0]].p, verts[vv[b1]].p)) {
      local[2].v[1] = local[1].v[0];
      local[1].v[1] = local[2].v[0];
    }
  }
  for (int q = 0; q < 4; ++q)
    out[4 * f + q] = local[q];
}
__global__ void collapsePrefilter(MeshView m, Candidate *out, int *flags,
                                  float *localLow, float low,
                                  ReferenceSurfaceGpu ref, bool relaxed) {
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=m.ne)return;
  out[id].keep=-1; flags[id]=0;
  auto e=m.e[id];
  if(e.f0<0)return;
  int a=e.a,b=e.b;
  float edgeLow=low;
  if(ref.sizingCount) {
    edgeLow*=fminf(sizingAt(ref,mul3(add3(m.v[a].p,m.v[b].p),.5f),m.f[e.f0].patch),
                   .5f*(m.v[a].target+m.v[b].target))/ref.regularLength;
  }
  localLow[id]=edgeLow;
  const float edgeDist2=dist2(m.v[a].p,m.v[b].p);
  if(!relaxed && edgeDist2>=edgeLow*edgeLow) {
    auto t0=m.f[e.f0];
    auto n0=normal3(m.v[t0.v[0]].p,m.v[t0.v[1]].p,m.v[t0.v[2]].p);
    float minArea=.5f*sqrtf(dot3(n0,n0));
    if(e.f1>=0) {
      auto t1=m.f[e.f1];
      auto n1=normal3(m.v[t1.v[0]].p,m.v[t1.v[1]].p,m.v[t1.v[2]].p);
      minArea=fminf(minArea,.5f*sqrtf(dot3(n1,n1)));
    }
    if(minArea>=edgeLow*edgeLow/100.f)return;
  }
  int da=m.offsets[a+1]-m.offsets[a],db=m.offsets[b+1]-m.offsets[b];
  if(relaxed && !((da==3||da==4)&&m.v[a].constraint<2) &&
      !((db==3||db==4)&&m.v[b].constraint<2))return;
  flags[id]=1;
}
__global__ void countRawCollapseStages(const Candidate *c,const int *flags,int n,int *counts) {
  const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;
  if(flags[id])atomicAdd(counts,1);
  if(c[id].keep>=0)atomicAdd(counts+1,1);
}struct CollapseStageCounters {
  unsigned long long active=0;
  unsigned long long movableReject=0;
  unsigned long long topologyReject=0;
  unsigned long long destReferenceReject=0;
  unsigned long long changedFaceReject=0;
  unsigned long long accepted=0;
};__global__ void collapseCandidatesShape(MeshView m, Candidate *out,
                                         const int *flags,bool relaxed) {
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=m.ne || !flags[id])return;
  auto e=m.e[id]; int a=e.a,b=e.b;
  int da=m.offsets[a+1]-m.offsets[a],db=m.offsets[b+1]-m.offsets[b];
  bool ma=movable(m,a,id),mb=movable(m,b,id);
  if(!ma && !mb)return;
  int common=0;
  for(int i=m.offsets[a];i<m.offsets[a+1];++i)
    if(neighbor(m,b,other(m.e[m.ids[i]],a)))++common;
  if(common!=(e.f1<0?1:2) || (da==3 && db==3 && e.f1>=0))return;
  int keep=ma && !mb?b:a,remove=keep==a?b:a;
  float3 dest=ma && mb?mul3(add3(m.v[a].p,m.v[b].p),.5f):m.v[keep].p;
  for(int side=0;side<2;++side) {
    int v=side?a:b;
    for(int i=m.offsets[v];i<m.offsets[v+1];++i) {
      auto x=m.e[m.ids[i]];
      if(x.f0>=0 && ownsFace(m.f[x.f0],x,v) && !changedFaceShapeSafe(m,m.f[x.f0],a,b,dest))return;
      if(x.f1>=0 && ownsFace(m.f[x.f1],x,v) && !changedFaceShapeSafe(m,m.f[x.f1],a,b,dest))return;
    }
  }
  out[id]={keep,remove,-1,-1,dest,(ma&&mb)?0.f:m.v[keep].target};
}
__device__ bool faceContainsBoth(Triangle t,int a,int b) {
  bool hasA=false,hasB=false;
  for(int k=0;k<3;++k){hasA|=t.v[k]==a;hasB|=t.v[k]==b;}
  return hasA&&hasB;
}
__device__ bool edgeHasNonMergedFace(MeshView m,int edgeId,int a,int b) {
  const auto edge=m.e[edgeId];
  if(edge.f0>=0 && !faceContainsBoth(m.f[edge.f0],a,b))return true;
  if(edge.f1>=0 && !faceContainsBoth(m.f[edge.f1],a,b))return true;
  return false;
}
__device__ int edgeBetween(MeshView m,int a,int b) {
  for(int i=m.offsets[a];i<m.offsets[a+1];++i) {
    const int edgeId=m.ids[i];
    if(other(m.e[edgeId],a)==b)return edgeId;
  }
  return -1;
}__global__ void collapseCandidatesLength(MeshView m,Candidate *out,float high,
                                         ReferenceSurfaceGpu ref,bool relaxed) {
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=m.ne || out[id].keep<0)return;
  const auto collapseEdge=m.e[id];
  const int a=collapseEdge.a,b=collapseEdge.b;
  const float3 dest=out[id].p;
  const float destSizing=ref.sizingCount
      ? (out[id].destSizing>0.f?out[id].destSizing:sizingAt(ref,dest,m.f[collapseEdge.f0].patch))
      : ref.regularLength;
  out[id].destSizing=destSizing;
  if(relaxed)return;
  const float high2=high*high;
  for(int side=0;side<2;++side) {
    const int v=side?a:b;
    for(int i=m.offsets[v];i<m.offsets[v+1];++i) {
      const auto edge=m.e[m.ids[i]];
      const int u=other(edge,v);
      if(u==a || u==b)continue;
      bool affects=false;
      if(edge.f0>=0 && !faceContainsBoth(m.f[edge.f0],a,b))affects=true;
      if(edge.f1>=0 && !faceContainsBoth(m.f[edge.f1],a,b))affects=true;
      if(!affects)continue;
      // Common neighbors are visible from both endpoint stars; check once.
      if(side==1 && neighbor(m,a,u))continue;
      const float3 otherP=m.v[u].p;
      const float edgeDist2=dist2(dest,otherP);
      if(!ref.sizingCount) {
        if(edgeDist2>high2){out[id].keep=-1;return;}
        continue;
      }
      const float avgTarget=.5f*(destSizing+m.v[u].target);
      const float maxScale=fminf(ref.regularLength,avgTarget)/ref.regularLength;
      if(edgeDist2>high2*maxScale*maxScale){out[id].keep=-1;return;}
      const float midSizing=sizingAt(ref,mul3(add3(dest,otherP),.5f),m.f[collapseEdge.f0].patch);
      const float scale=fminf(midSizing,avgTarget)/ref.regularLength;
      if(edgeDist2>high2*scale*scale){out[id].keep=-1;return;}
    }
  }
}__global__ void flipCandidates(MeshView m, Candidate *out,
                               ReferenceSurfaceGpu ref,bool strictQuality) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= m.ne)
    return;
  out[id].keep = -1;
  auto e = m.e[id];
  if (e.feature || e.f0 < 0 || e.f1 < 0 || m.f[e.f0].patch!=m.f[e.f1].patch)
    return;
  auto f = m.f[e.f0], g = m.f[e.f1];
  int a = e.a, b = e.b, c = third(f, a, b), d = third(g, a, b);
  if (c < 0 || d < 0 || c == d || neighbor(m, c, d))
    return;
  bool directed = false;
  for (int k = 0; k < 3; ++k)
    directed |= f.v[k] == a && f.v[(k + 1) % 3] == b;
  if (!directed) {
    int t = a;
    a = b;
    b = t;
  }
  auto pa = m.v[a].p, pb = m.v[b].p, pc = m.v[c].p, pd = m.v[d].p;
  float oldQ = fminf(qualityVcg(pa, pb, pc), qualityVcg(pb, pa, pd)),
        newQ = fminf(qualityVcg(pc, pd, pb), qualityVcg(pd, pc, pa));
  int before = 0, after = 0, vs[4] = {a, b, c, d};
  for (int k = 0; k < 4; ++k) {
    int v = vs[k], ideal = 6;
    for (int i = m.offsets[v]; i < m.offsets[v + 1]; ++i)
      if (m.e[m.ids[i]].f1 < 0)
        ideal = 4;
    int degree = m.offsets[v + 1] - m.offsets[v];
    before += abs(degree - ideal);
    after += abs(degree + (k < 2 ? -1 : 1) - ideal);
  }
  // The strict policy prevents a valence-improving flip from immediately
  // reversing by requiring the worse triangle to improve on every flip.
  const bool accepted=strictQuality
      ? ((after <= before && newQ > oldQ) ||
         (after > before && newQ > oldQ * 1.5f))
      : ((after < before && newQ >= oldQ * .5f) ||
         (after == before && newQ > oldQ) ||
         (after > before && newQ > oldQ * 1.5f));
  if (!accepted)
    return;
  auto n0 = normal3(pa, pb, pc), n1 = normal3(pb, pa, pd),
       nn0 = normal3(pc, pd, pb), nn1 = normal3(pd, pc, pa);
  constexpr float cos5 = .996194698f;
  if (dot3(n0, nn0) < cos5 * sqrtf(dot3(n0, n0) * dot3(nn0, nn0)) ||
      dot3(n0, nn1) < cos5 * sqrtf(dot3(n0, n0) * dot3(nn1, nn1)) ||
      dot3(n1, nn0) < cos5 * sqrtf(dot3(n1, n1) * dot3(nn0, nn0)) ||
      dot3(n1, nn1) < cos5 * sqrtf(dot3(n1, n1) * dot3(nn1, nn1)))
    return;
  if (!referenceFaceSafe(ref, f.patch, pc, pd, pb) ||
      !referenceFaceSafe(ref, f.patch, pd, pc, pa))
    return;
  out[id] = {a, b, c, d, {}};
}
__device__ unsigned int priority(int id, unsigned int seed) {
  unsigned x = unsigned(id) ^ seed * 2654435761u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
__device__ unsigned long long rank(int id, unsigned seed) {
  return (static_cast<unsigned long long>(priority(id, seed)) << 32) |
         unsigned(id);
}
__global__ void initClaims(unsigned long long *claim, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    claim[i] = ~0ull;
}
__global__ void claimCandidates(MeshView m, const Candidate *c,
                                unsigned long long *claims, bool flip,
                                unsigned seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= m.ne || c[id].keep < 0)
    return;
  auto x = c[id];
  auto r = rank(id, seed);
  int vs[4] = {x.keep, x.remove, x.c, x.d};
  for (int k = 0; k < (flip ? 4 : 2); ++k) {
    int v = vs[k];
    atomicMin(claims + v, r);
    if (!flip)
      for (int i = m.offsets[v]; i < m.offsets[v + 1]; ++i)
        atomicMin(claims + other(m.e[m.ids[i]], v), r);
  }
}
__global__ void resolveCandidates(MeshView m, const Candidate *c,
                                  const unsigned long long *claims, int *chosen,
                                  bool flip, unsigned seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= m.ne)
    return;
  chosen[id] = 0;
  if (c[id].keep < 0)
    return;
  auto x = c[id];
  auto r = rank(id, seed);
  int vs[4] = {x.keep, x.remove, x.c, x.d};
  for (int k = 0; k < (flip ? 4 : 2); ++k) {
    int v = vs[k];
    if (claims[v] != r)
      return;
    if (!flip)
      for (int i = m.offsets[v]; i < m.offsets[v + 1]; ++i)
        if (claims[other(m.e[m.ids[i]], v)] != r)
          return;
  }
  chosen[id] = 1;
}


__device__ bool changedFaceReferenceSafeOnly(MeshView m,Triangle t,int a,int b,
                                             float3 dest,ReferenceSurfaceGpu ref) {
  float3 after[3];
  bool hasA=false,hasB=false;
  int changedIndex=-1;
  for(int k=0;k<3;++k) {
    const int v=t.v[k];
    const bool changed=(v==a || v==b);
    after[k]=changed?dest:m.v[v].p;
    if(changed)changedIndex=k;
    hasA|=v==a; hasB|=v==b;
  }
  if(a!=b && hasA && hasB)return true;
  return referenceChangedFaceSafeCached(ref,t.patch,after[0],after[1],after[2],changedIndex,
                                        -1.f,-1.f,-1.f);
}
__device__ bool collapseChosenSafe(MeshView m,Candidate &x,int id,float high,
                                   ReferenceSurfaceGpu ref,bool relaxed) {
  const auto e=m.e[id];
  const int a=e.a,b=e.b;
  const float3 dest=x.p;
  const float destSizing=x.destSizing>0.f?x.destSizing:(ref.sizingCount?sizingAt(ref,dest,m.f[e.f0].patch):ref.regularLength);
  x.destSizing=destSizing;
  for(int side=0;side<2;++side) {
    const int v=side?a:b;
    for(int i=m.offsets[v];i<m.offsets[v+1];++i) {
      const auto edge=m.e[m.ids[i]];
      if(edge.f0>=0 && ownsFace(m.f[edge.f0],edge,v) &&
         !changedFaceLengthSafe(m,m.f[edge.f0],a,b,dest,high,ref,relaxed,destSizing))return false;
      if(edge.f1>=0 && ownsFace(m.f[edge.f1],edge,v) &&
         !changedFaceLengthSafe(m,m.f[edge.f1],a,b,dest,high,ref,relaxed,destSizing))return false;
    }
  }
  const float destTolerance=ref.sizingCount?fminf(ref.tolerance,.08f*destSizing):ref.tolerance;
  if(!referenceNearWithTolerance(ref,m.f[e.f0].patch,dest,destTolerance))return false;
  for(int side=0;side<2;++side) {
    const int v=side?a:b;
    for(int i=m.offsets[v];i<m.offsets[v+1];++i) {
      const auto edge=m.e[m.ids[i]];
      if(edge.f0>=0 && ownsFace(m.f[edge.f0],edge,v) &&
         !changedFaceReferenceSafeOnly(m,m.f[edge.f0],a,b,dest,ref))return false;
      if(edge.f1>=0 && ownsFace(m.f[edge.f1],edge,v) &&
         !changedFaceReferenceSafeOnly(m,m.f[edge.f1],a,b,dest,ref))return false;
    }
  }
  return true;
}__device__ bool collapseReferenceSafe(MeshView m,const Candidate &x,int id,float high,
                                      ReferenceSurfaceGpu ref,bool relaxed) {
  const auto e=m.e[id];
  const int a=e.a,b=e.b;
  const float3 dest=x.p;
  const float destSizing=x.destSizing>0.f?x.destSizing:(ref.sizingCount?sizingAt(ref,dest,m.f[e.f0].patch):ref.regularLength);
  const float destTolerance=ref.sizingCount?fminf(ref.tolerance,.08f*destSizing):ref.tolerance;
  if(!referenceNearWithTolerance(ref,m.f[e.f0].patch,dest,destTolerance))return false;
  for(int side=0;side<2;++side) {
    const int v=side?a:b;
    for(int i=m.offsets[v];i<m.offsets[v+1];++i) {
      const auto edge=m.e[m.ids[i]];
      if(edge.f0>=0 && ownsFace(m.f[edge.f0],edge,v) &&
         !changedFaceReferenceSafeOnly(m,m.f[edge.f0],a,b,dest,ref))return false;
      if(edge.f1>=0 && ownsFace(m.f[edge.f1],edge,v) &&
         !changedFaceReferenceSafeOnly(m,m.f[edge.f1],a,b,dest,ref))return false;
    }
  }
  return true;
}
__global__ void validateCollapseBlockers(MeshView m,Candidate *c,
                                         const unsigned long long *claims,
                                         unsigned char *validated,
                                         unsigned seed,float high,ReferenceSurfaceGpu ref,
                                         bool relaxed,int *stats) {
  const int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=m.ne || c[id].keep<0)return;
  const auto x=c[id];
  const auto r=rank(id,seed);
  bool blocker=false;
  const int vs[2]={x.keep,x.remove};
  for(int k=0;k<2 && !blocker;++k) {
    const int v=vs[k];
    blocker=claims[v]==r;
    for(int i=m.offsets[v];i<m.offsets[v+1] && !blocker;++i)
      blocker=claims[other(m.e[m.ids[i]],v)]==r;
  }
  if(!blocker)return;
  atomicAdd(stats,1);
  if(!collapseReferenceSafe(m,x,id,high,ref,relaxed)) {
    c[id].keep=-1;
    atomicAdd(stats+1,1);
  }
}
__global__ void validateChosenCollapse(MeshView m,Candidate *c,int *chosen,float high,
                                       ReferenceSurfaceGpu ref,bool relaxed,int *stats) {
  const int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=m.ne || !chosen[id] || c[id].keep<0)return;
  atomicAdd(stats,1);
  if(!collapseReferenceSafe(m,c[id],id,high,ref,relaxed)) {
    chosen[id]=0;
    c[id].keep=-1;
    atomicAdd(stats+1,1);
  }
}__global__ void initMap(int *map, int n) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n)
    map[v] = v;
}
__global__ void applyCandidates(MeshView m, const Candidate *c,
                                const int *chosen, int *map, bool flip) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= m.ne || !chosen[id])
    return;
  auto x = c[id];
  if (flip) {
    auto e = m.e[id];
    const int patch=m.f[e.f0].patch;
    m.f[e.f0] = {{x.c, x.d, x.remove},patch};
    m.f[e.f1] = {{x.d, x.c, x.keep},patch};
  } else {
    map[x.remove] = x.keep;
    m.v[x.keep].p = x.p;
    if(x.destSizing>0.f) m.v[x.keep].target=x.destSizing;
  }
}
__global__ void remapFaces(MeshView m, const int *map) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= m.nf)
    return;
  for (int k = 0; k < 3; ++k)
    m.f[f].v[k] = map[m.f[f].v[k]];
}
__global__ void smoothVertices(MeshView m, Vertex *next,
                               ReferenceSurfaceGpu ref, float lambda,
                               unsigned seed, int *moved, int *projectionHint, int *dirtyQueue, int *dirtyCount) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= m.nv)
    return;
  auto p = m.v[v].p;
  next[v] = m.v[v];
  if (m.v[v].constraint >= 2 || !(lambda > 0))
    return;
  int degree = m.offsets[v + 1] - m.offsets[v];
  if (degree < 3)
    return;
  const auto r = rank(v, seed);
  float3 sum = make_float3(0, 0, 0);
  for (int i = m.offsets[v]; i < m.offsets[v + 1]; ++i) {
    auto e = m.e[m.ids[i]];
    if (e.f1 < 0)
      return;
    int u = other(e, v);
    if (m.v[u].constraint < 2 && rank(u, seed) < r)
      return;
    sum = add3(sum, m.v[u].p);
  }
  auto avg = mul3(add3(p, mul3(sum, 2.f)), 1.f / (2 * degree + 1));
  float step = lambda;
  const int patch=vertexPatch(m,v);
  for (int attempt = 0; attempt < 8; ++attempt) {
    auto trial = add3(p, mul3(sub3(avg, p), step));
    float3 dest;
    int hintId=projectionHint[v];
    const float tolerance=toleranceAt(ref,trial,patch);
    bool ok=projectReferenceHintedLocal(ref,patch,trial,tolerance,dest,hintId);
    projectionHint[v]=hintId;
    ok=ok && dist2(dest,trial)<=tolerance*tolerance;
    for (int i = m.offsets[v]; i < m.offsets[v + 1] && ok; ++i) {
      auto e = m.e[m.ids[i]];
      if (e.f0 >= 0 && ownsFace(m.f[e.f0],e,v))
        ok = changedFaceSafe(m, m.f[e.f0], v, v, dest, 0, ref, true, 0.f, false, nullptr);
      if (ok && e.f1 >= 0 && ownsFace(m.f[e.f1],e,v))
        ok = changedFaceSafe(m, m.f[e.f1], v, v, dest, 0, ref, true, 0.f, false, nullptr);
    }
    if (ok) {
      if (dist2(dest, p) > 1.e-16f) {
        next[v].p = dest;
        atomicAdd(moved, 1);
        if(dirtyQueue && dirtyCount) {
          for(int j=m.offsets[v];j<m.offsets[v+1];++j) {
            const auto de=m.e[m.ids[j]];
            if(de.f0>=0 && ownsFace(m.f[de.f0],de,v))
              dirtyQueue[atomicAdd(dirtyCount,1)]=de.f0;
            if(de.f1>=0 && ownsFace(m.f[de.f1],de,v))
              dirtyQueue[atomicAdd(dirtyCount,1)]=de.f1;
          }
        }
      }
      return;
    }
    step *= .5f;
  }
}
__global__ void proposeProjection(MeshView m,Vertex *proposal,ReferenceSurfaceGpu ref,int *projectionHint) {
  int v=blockIdx.x*blockDim.x+threadIdx.x;
  if(v>=m.nv)return;
  proposal[v]=m.v[v];
  if(m.v[v].constraint>=4)return;
  float3 q;
  bool ok=false;
  if(projectionHint) {
    int hintId=projectionHint[v];
    ok=projectReferenceHinted(ref,vertexPatch(m,v),m.v[v].p,q,hintId);
    projectionHint[v]=hintId;
  } else {
    ok=projectReference(ref,vertexPatch(m,v),m.v[v].p,q);
  }
  if(ok)proposal[v].p=q;
}__global__ void checkProjectionFaces(MeshView m,const Vertex *proposal,ReferenceSurfaceGpu ref,int *reject,int *badFaces,const int *dirtyFaces=nullptr) {
  int f=blockIdx.x*blockDim.x+threadIdx.x;
  if(f>=m.nf)return;
  if(dirtyFaces && !dirtyFaces[f])return;
  auto t=m.f[f];
  const auto a=m.v[t.v[0]].p,b=m.v[t.v[1]].p,c=m.v[t.v[2]].p;
  const auto x=proposal[t.v[0]].p,y=proposal[t.v[1]].p,z=proposal[t.v[2]].p;
  const auto oldN=normal3(a,b,c),newN=normal3(x,y,z);
  const float area2=dot3(newN,newN),oldArea2=dot3(oldN,oldN);
  const float q=qualityVcgKnownNormal(x,y,z,newN);
  const float oldQ=qualityVcgKnownNormal(a,b,c,oldN);
  const bool qualityBad=!(area2>0) || !isfinite(area2) || !(q>0) || !isfinite(q) ||
      q<.5f*oldQ || dot3(oldN,newN)<.7f*sqrtf(oldArea2*area2);
  const bool moved0=dist2(a,x)>0,moved1=dist2(b,y)>0,moved2=dist2(c,z)>0;
  const bool moved=moved0||moved1||moved2;
  const int movedCount=int(moved0)+int(moved1)+int(moved2);
  bool referenceSafe=true;
  if(!qualityBad && moved) {
    if(movedCount==1) {
      const int changedIndex=moved0?0:(moved1?1:2);
      referenceSafe=referenceChangedFaceSafeCached(ref,t.patch,x,y,z,changedIndex,-1.f,-1.f,-1.f);
    } else {
      referenceSafe=referenceFaceSafe(ref,t.patch,x,y,z);
    }
  }
  const bool errorBad=!qualityBad && moved && !referenceSafe;
  if(qualityBad || errorBad) {
    atomicAdd(badFaces,1);
    if(errorBad)atomicAdd(badFaces+1,1);
    for(int k=0;k<3;++k)atomicExch(reject+t.v[k],1);
  }
}
__global__ void checkProjectionFaceQualityQueue(MeshView m,const Vertex *proposal,int *reject,
                                                 int *badFaces,const int *dirtyQueue,const int *dirtyCount,
                                                 int *qualityBad) {
  const int count=*dirtyCount;
  for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<count;idx+=blockDim.x*gridDim.x) {
    const int f=dirtyQueue[idx];
    qualityBad[f]=0;
    const auto t=m.f[f];
    const auto a=m.v[t.v[0]].p,b=m.v[t.v[1]].p,c=m.v[t.v[2]].p;
    const auto x=proposal[t.v[0]].p,y=proposal[t.v[1]].p,z=proposal[t.v[2]].p;
    const auto oldN=normal3(a,b,c),newN=normal3(x,y,z);
    const float area2=dot3(newN,newN),oldArea2=dot3(oldN,oldN);
    const float q=qualityVcgKnownNormal(x,y,z,newN);
    const float oldQ=qualityVcgKnownNormal(a,b,c,oldN);
    const bool bad=!(area2>0) || !isfinite(area2) || !(q>0) || !isfinite(q) ||
        q<.5f*oldQ || dot3(oldN,newN)<.7f*sqrtf(oldArea2*area2);
    if(!bad)continue;
    qualityBad[f]=1;
    atomicAdd(badFaces,1);
    for(int k=0;k<3;++k)atomicExch(reject+t.v[k],1);
  }
}
__global__ void checkProjectionFaceReferenceExactQueue(MeshView m,const Vertex *proposal,ReferenceSurfaceGpu ref,
                                                        int *reject,int *badFaces,const int *dirtyQueue,
                                                        const int *dirtyCount) {
  const int count=*dirtyCount;
  for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<count;idx+=blockDim.x*gridDim.x) {
    const int f=dirtyQueue[idx];
    const auto t=m.f[f];
    const auto a=m.v[t.v[0]].p,b=m.v[t.v[1]].p,c=m.v[t.v[2]].p;
    const auto x=proposal[t.v[0]].p,y=proposal[t.v[1]].p,z=proposal[t.v[2]].p;
    const bool moved0=dist2(a,x)>0,moved1=dist2(b,y)>0,moved2=dist2(c,z)>0;
    const int movedCount=int(moved0)+int(moved1)+int(moved2);
    if(!movedCount)continue;
    bool safe=false;
    if(movedCount==1) {
      const int changedIndex=moved0?0:(moved1?1:2);
      safe=referenceChangedFaceSafeCached(ref,t.patch,x,y,z,changedIndex,-1.f,-1.f,-1.f);
    } else safe=referenceFaceSafe(ref,t.patch,x,y,z);
    if(safe)continue;
    atomicAdd(badFaces,1);atomicAdd(badFaces+1,1);
    for(int k=0;k<3;++k)atomicExch(reject+t.v[k],1);
  }
}__global__ void checkProjectionFaceReferenceQueue(MeshView m,const Vertex *proposal,ReferenceSurfaceGpu ref,
                                                   int *reject,int *badFaces,const int *dirtyQueue,
                                                   const int *dirtyCount,const int *qualityBad) {
  const int count=*dirtyCount;
  for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<count;idx+=blockDim.x*gridDim.x) {
    const int f=dirtyQueue[idx];
    if(qualityBad[f])continue;
    const auto t=m.f[f];
    const auto a=m.v[t.v[0]].p,b=m.v[t.v[1]].p,c=m.v[t.v[2]].p;
    const auto x=proposal[t.v[0]].p,y=proposal[t.v[1]].p,z=proposal[t.v[2]].p;
    const bool moved=dist2(a,x)>0 || dist2(b,y)>0 || dist2(c,z)>0;
    if(!moved || referenceFaceSafe(ref,t.patch,x,y,z))continue;
    atomicAdd(badFaces,1);atomicAdd(badFaces+1,1);
    for(int k=0;k<3;++k)atomicExch(reject+t.v[k],1);
  }
}__global__ void checkProjectionFaceQuality(MeshView m,const Vertex *proposal,int *reject,
                                            int *badFaces,const int *dirtyFaces,int *qualityBad) {
  int f=blockIdx.x*blockDim.x+threadIdx.x;
  if(f>=m.nf)return;
  qualityBad[f]=0;
  if(dirtyFaces && !dirtyFaces[f])return;
  const auto t=m.f[f];
  const auto a=m.v[t.v[0]].p,b=m.v[t.v[1]].p,c=m.v[t.v[2]].p;
  const auto x=proposal[t.v[0]].p,y=proposal[t.v[1]].p,z=proposal[t.v[2]].p;
  const auto oldN=normal3(a,b,c),newN=normal3(x,y,z);
  const float area2=dot3(newN,newN),oldArea2=dot3(oldN,oldN);
  const float q=qualityVcgKnownNormal(x,y,z,newN);
  const float oldQ=qualityVcgKnownNormal(a,b,c,oldN);
  const bool bad=!(area2>0) || !isfinite(area2) || !(q>0) || !isfinite(q) ||
      q<.5f*oldQ || dot3(oldN,newN)<.7f*sqrtf(oldArea2*area2);
  if(!bad)return;
  qualityBad[f]=1;
  atomicAdd(badFaces,1);
  for(int k=0;k<3;++k)atomicExch(reject+t.v[k],1);
}
__global__ void checkProjectionFaceReference(MeshView m,const Vertex *proposal,ReferenceSurfaceGpu ref,
                                              int *reject,int *badFaces,const int *dirtyFaces) {
  const int f=blockIdx.x*blockDim.x+threadIdx.x;
  if(f>=m.nf || (dirtyFaces && !dirtyFaces[f]))return;
  const auto t=m.f[f];
  const auto a=m.v[t.v[0]].p,b=m.v[t.v[1]].p,c=m.v[t.v[2]].p;
  const auto x=proposal[t.v[0]].p,y=proposal[t.v[1]].p,z=proposal[t.v[2]].p;
  const bool moved0=dist2(a,x)>0,moved1=dist2(b,y)>0,moved2=dist2(c,z)>0;
  const int movedCount=int(moved0)+int(moved1)+int(moved2);
  if(!movedCount)return;
  bool safe=false;
  if(movedCount==1) {
    const int changedIndex=moved0?0:(moved1?1:2);
    safe=referenceChangedFaceSafeCached(ref,t.patch,x,y,z,changedIndex,-1.f,-1.f,-1.f);
  } else {
    safe=referenceFaceSafe(ref,t.patch,x,y,z);
  }
  if(safe)return;
  atomicAdd(badFaces,1);atomicAdd(badFaces+1,1);
  for(int k=0;k<3;++k)atomicExch(reject+t.v[k],1);
}__global__ void revertProjection(MeshView m,Vertex *proposal,const int *reject,int *rejected) {
  int v=blockIdx.x*blockDim.x+threadIdx.x;
  if(v>=m.nv || !reject[v])return;
  if(dist2(proposal[v].p,m.v[v].p)>0)atomicAdd(rejected,1);
  proposal[v]=m.v[v];
}
// Validate the complete proposed face, not each vertex in isolation. Partial
// rollback can invalidate a neighboring face, so recheck until stable. A bounded
// failure rolls back the entire projection; topology checks are never bypassed.
int safeProject(MeshView m,ReferenceSurfaceGpu ref,int *errorRejects=nullptr,int *projectionHint=nullptr) {
  Buffer<Vertex> proposal(m.nv);
  Buffer<int> reject(m.nv),bad(2),rejected(1);
  checked(streamZero(rejected.p,0,sizeof(int)));
  proposeProjection<<<(m.nv+127)/128,128,0,executionStream>>>(m,proposal.p,ref,projectionHint);
  for(int pass=0;pass<16;++pass) {
    checked(streamZero(reject.p,0,m.nv*sizeof(int)));
    checked(streamZero(bad.p,0,2*sizeof(int)));
    checkProjectionFaces<<<(m.nf+127)/128,128,0,executionStream>>>(m,proposal.p,ref,reject.p,bad.p);
    sync();
    const auto counts=bad.read();
    if(errorRejects)*errorRejects+=counts[1];
    if(counts[0]==0) {
      checked(streamCopy(m.v,proposal.p,m.nv*sizeof(Vertex),cudaMemcpyDeviceToDevice));
      return rejected.read()[0];
    }
    revertProjection<<<(m.nv+127)/128,128,0,executionStream>>>(m,proposal.p,reject.p,rejected.p);
  }
  sync();
  // No proposal is committed. The original positions remain valid.
  return m.nv;
}

__global__ void auditGeometry(MeshView m, ReferenceSurfaceGpu ref,
                              unsigned int *maximum) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= m.nf)
    return;
  auto t = m.f[f];
  const auto a = m.v[t.v[0]].p, b = m.v[t.v[1]].p, c = m.v[t.v[2]].p;
  const float3 points[7] = {a,
                            b,
                            c,
                            mul3(add3(a, b), .5f),
                            mul3(add3(b, c), .5f),
                            mul3(add3(c, a), .5f),
                            mul3(add3(add3(a, b), c), 1.f / 3.f)};
  for (auto p : points) {
    float3 q;
    const float error =
        projectReference(ref, t.patch, p, q) ? sqrtf(dist2(p, q)) : FLT_MAX;
    atomicMax(maximum, __float_as_uint(error));
    atomicMax(maximum+1, __float_as_uint(error / fmaxf(toleranceAt(ref,p,t.patch),1.e-20f)));
  }
}


// Preserve the legacy seven-per-face sample set, evaluating shared samples once.
// Isolated vertices are excluded: the legacy face traversal never visits them.
constexpr int stGeometryAuditBlockSize = 128;
__global__ void AuditGeometryUnique(MeshView m, ReferenceSurfaceGpu ref,
                                   unsigned int *maximum) {
  const size_t nv = size_t(m.nv), ne = size_t(m.ne);
  const size_t count = nv + ne + size_t(m.nf);
  unsigned int localError = 0, localRatio = 0;
  for (size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       id < count; id += size_t(blockDim.x) * gridDim.x) {
    float3 p;
    int patch=0;
    if (id < nv) {
      if (m.offsets[id] == m.offsets[id + 1]) continue;
      p = m.v[id].p;
      patch=vertexPatch(m,int(id));
    } else if (id < nv + ne) {
      const auto e = m.e[id - nv];
      if (e.f0 < 0 && e.f1 < 0) continue;
      p = mul3(add3(m.v[e.a].p, m.v[e.b].p), .5f);
      patch=m.f[e.f0>=0?e.f0:e.f1].patch;
    } else {
      const auto t = m.f[id - nv - ne];
      const auto a = m.v[t.v[0]].p, b = m.v[t.v[1]].p, c = m.v[t.v[2]].p;
      p = mul3(add3(add3(a, b), c), 1.f / 3.f);
      patch=t.patch;
    }
    float3 q;
    const float error =
        projectReference(ref, patch, p, q) ? sqrtf(dist2(p, q)) : FLT_MAX;
    localError = max(localError, __float_as_uint(error));
    localRatio = max(localRatio,
        __float_as_uint(error / fmaxf(toleranceAt(ref, p,patch), 1.e-20f)));
  }
  // Max reduction on the same unsigned bit patterns as the legacy atomicMax.
  // No floating-point sum, reordered arithmetic, or relaxed tolerance is used.
  __shared__ unsigned int errors[stGeometryAuditBlockSize];
  __shared__ unsigned int ratios[stGeometryAuditBlockSize];
  const int lane = threadIdx.x;
  errors[lane] = localError; ratios[lane] = localRatio;
  __syncthreads();
  for (int step = stGeometryAuditBlockSize / 2; step > 0; step /= 2) {
    if (lane < step) {
      errors[lane] = max(errors[lane], errors[lane + step]);
      ratios[lane] = max(ratios[lane], ratios[lane + step]);
    }
    __syncthreads();
  }
  if (lane == 0) {
    atomicMax(maximum, errors[0]); atomicMax(maximum + 1, ratios[0]);
  }
}
enum class GeometryAuditMode { Legacy, Unique, Verify };
GeometryAuditMode ReadGeometryAuditMode() {
  const char *value = std::getenv("CAD_ADAPTIVE_RAW_AUDIT_MODE");
  if (!value || !*value || std::strcmp(value, "unique") == 0)
    return GeometryAuditMode::Unique;
  if (std::strcmp(value, "legacy") == 0) return GeometryAuditMode::Legacy;
  if (std::strcmp(value, "verify") == 0) return GeometryAuditMode::Verify;
  throw std::runtime_error("CAD_ADAPTIVE_RAW_AUDIT_MODE must be legacy, unique or verify");
}
struct GeometryAuditTiming {
  double LegacySeconds = 0, UniqueSeconds = 0;
  unsigned long long LegacySampleVisits = 0, UniqueSampleSlots = 0;
  int Calls = 0, VerifiedCalls = 0;
};
thread_local GeometryAuditTiming stGeometryAuditTiming;
void RunGeometryAudit(MeshView m, ReferenceSurfaceGpu ref, unsigned int *maximum) {
  const auto mode = ReadGeometryAuditMode();
  auto &timing = stGeometryAuditTiming;
  ++timing.Calls;
  timing.LegacySampleVisits += 7ull * m.nf;
  const size_t uniqueSlots = size_t(m.nv) + size_t(m.ne) + size_t(m.nf);
  timing.UniqueSampleSlots += uniqueSlots;
  checked(streamZero(maximum, 0, 2 * sizeof(unsigned int)));
  const auto start = std::chrono::steady_clock::now();
  if (mode == GeometryAuditMode::Legacy) {
    if (m.nf > 0)
      auditGeometry<<<(m.nf + 127) / 128, 128, 0, executionStream>>>(m, ref, maximum);
    sync();
    timing.LegacySeconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return;
  }
  if (uniqueSlots > 0) {
    const int blocks = int(std::min(size_t(4096),
        (uniqueSlots + stGeometryAuditBlockSize - 1) / stGeometryAuditBlockSize));
    AuditGeometryUnique<<<blocks, stGeometryAuditBlockSize, 0, executionStream>>>(m, ref, maximum);
  }
  sync();
  timing.UniqueSeconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  if (mode != GeometryAuditMode::Verify) return;
  Buffer<unsigned int> legacy(2);
  checked(streamZero(legacy.p, 0, 2 * sizeof(unsigned int)));
  const auto legacyStart = std::chrono::steady_clock::now();
  if (m.nf > 0)
    auditGeometry<<<(m.nf + 127) / 128, 128, 0, executionStream>>>(m, ref, legacy.p);
  sync();
  timing.LegacySeconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - legacyStart).count();
  const auto expected = legacy.read();
  unsigned int actual[2] = {};
  checked(streamCopy(actual, maximum, sizeof(actual), cudaMemcpyDeviceToHost));
  if (actual[0] != expected[0] || actual[1] != expected[1])
    throw std::runtime_error("geometry audit mismatch: unique=" +
        std::to_string(actual[0]) + "," + std::to_string(actual[1]) +
        " legacy=" + std::to_string(expected[0]) + "," + std::to_string(expected[1]));
  ++timing.VerifiedCalls;
}

// Raw single-patch CUDA path only needs the edge table between GPU passes.
// Avoid rebuilding per-vertex incidentFaces: that structure is unused by the
// raw CUDA operators and caused hundreds of thousands of small host allocations.
void rebuildRawTopology(SemanticMesh &mesh, bool buildFaceEdges=true) {
  mesh.incidentFaces.clear();
  mesh.edges.clear();
  if(buildFaceEdges) mesh.faceEdgeIds.assign(size_t(mesh.faceCount())*3,-1); else mesh.faceEdgeIds.clear();
  mesh.edges.reserve(size_t(mesh.faceCount()) * 3 / 2 + 16);

  static thread_local std::vector<int> hashIndex;
  static thread_local std::vector<uint32_t> hashEpoch;
  static thread_local uint32_t epoch=0;
  size_t required=8;
  while(required<size_t(mesh.faceCount())*6) required*=2;
  if(hashIndex.size()<required) {
    hashIndex.resize(required);
    hashEpoch.resize(required,0u);
  }
  const size_t capacity=hashIndex.size();
  if(++epoch==0u) {std::fill(hashEpoch.begin(),hashEpoch.end(),0u);epoch=1u;}
  const auto slotHash=[&](uint64_t x) {
    x^=x>>33;x*=0xff51afd7ed558ccdULL;x^=x>>33;
    return size_t(x)&(capacity-1);
  };
  const auto findEdgeSlot=[&](uint64_t edgeKey) {
    size_t slot=slotHash(edgeKey);
    while(hashEpoch[slot]==epoch &&
          key(int(mesh.edges[hashIndex[slot]].v0),int(mesh.edges[hashIndex[slot]].v1))!=edgeKey)
      slot=(slot+1)&(capacity-1);
    return slot;
  };
  for(int f=0;f<mesh.faceCount();++f) {
    if(!mesh.faceAlive[f]) continue;
    const uint32_t v[3]={mesh.i0[f],mesh.i1[f],mesh.i2[f]};
    for(int k=0;k<3;++k) {
      const uint32_t va=v[k],vb=v[(k+1)%3];
      const uint64_t edgeKey=key(int(va),int(vb));
      const size_t slot=findEdgeSlot(edgeKey);
      if(hashEpoch[slot]!=epoch) {
        EdgeRec e;
        e.v0=std::min(va,vb);e.v1=std::max(va,vb);
        e.face0=f;e.patchLeft=mesh.facePatchId[f];
        mesh.edges.push_back(e);
        hashIndex[slot]=int(mesh.edges.size())-1;
        hashEpoch[slot]=epoch;
      } else {
        auto &e=mesh.edges[hashIndex[slot]];
        if(e.face1<0) {e.face1=f;e.patchRight=mesh.facePatchId[f];}
        else e.flags=uint8_t(e.flags|EdgeProtected);
      }
      if(buildFaceEdges) mesh.faceEdgeIds[size_t(f)*3+k]=hashIndex[slot];
    }
  }
  // Only ~1% of edges are features on the large STL. Look up those feature keys
  // in the edge hash instead of performing an unordered_map lookup for every edge.
  for(const auto &kv:mesh.featureEdges) {
    const size_t slot=findEdgeSlot(kv.first);
    if(hashEpoch[slot]!=epoch) continue;
    auto &e=mesh.edges[hashIndex[slot]];
    e.flags=uint8_t(e.flags|EdgeSharp|EdgeProtected);
    e.featureCurveId=kv.second;
  }
  for(auto &e:mesh.edges) if(e.face1>=0 && e.patchLeft!=e.patchRight)
    e.flags=uint8_t(e.flags|EdgePatchBoundary|EdgeProtected);
  for(auto &e:mesh.edges) if(e.face1<0) {
    e.flags=uint8_t(e.flags|EdgeMeshBoundary|EdgeProtected);
    e.featureCurveId=kOpenBoundaryFeature;
  }
}
void rebuildSparse(SemanticMesh &mesh,const std::vector<Vertex> &vertices,
                   const std::vector<Triangle> &faces,
                   const std::unordered_map<uint64_t,uint32_t> &features,
                   bool buildFaceEdges=true) {
  if(int(vertices.size())!=mesh.vertexCount())throw std::runtime_error("sparse rebuild vertex count mismatch");
  for(int v=0;v<mesh.vertexCount();++v) {
    const auto p=vertices[v].p;
    mesh.setPosition(v,{p.x,p.y,p.z});
    mesh.vertexConstraint[v]=uint8_t(vertices[v].constraint);
    mesh.targetLength[v]=vertices[v].target;
  }
  size_t validFaceCount=0;
  for(const auto &t:faces)
    validFaceCount+=size_t(t.v[0]>=0 && t.v[0]!=t.v[1] && t.v[1]!=t.v[2] && t.v[2]!=t.v[0]);
  mesh.i0.resize(validFaceCount);mesh.i1.resize(validFaceCount);mesh.i2.resize(validFaceCount);
  mesh.facePatchId.resize(validFaceCount);
  mesh.facePatchType.resize(validFaceCount);
  mesh.faceAlive.assign(validFaceCount,1u);
  size_t packedFace=0;
  for(const auto &t:faces) if(t.v[0]>=0 && t.v[0]!=t.v[1] && t.v[1]!=t.v[2] && t.v[2]!=t.v[0]) {
    mesh.i0[packedFace]=uint32_t(t.v[0]);
    mesh.i1[packedFace]=uint32_t(t.v[1]);
    mesh.i2[packedFace]=uint32_t(t.v[2]);
    mesh.facePatchId[packedFace]=uint32_t(t.patch);
    mesh.facePatchType[packedFace]=uint8_t(mesh.patches[t.patch].type);
    ++packedFace;
  }
  mesh.featureEdges=features;
  rebuildRawTopology(mesh,buildFaceEdges);
}
void compactRawVertices(SemanticMesh &mesh) {
  std::vector<int> map(mesh.vertexCount(),-1);
  std::vector<int> owner(mesh.vertexCount(),-1);
  for(int f=0;f<mesh.faceCount();++f) {
    map[mesh.i0[f]]=0;map[mesh.i1[f]]=0;map[mesh.i2[f]]=0;
    for(int v:mesh.face(f))if(owner[v]<0)owner[v]=int(mesh.facePatchId[f]);
  }
  SemanticMesh out;out.patches=mesh.patches;
  for(int v=0;v<mesh.vertexCount();++v) if(map[v]==0) {
    map[v]=out.addVertex(mesh.position(v),owner[v],VertexConstraint(mesh.vertexConstraint[v]));
    out.targetLength[map[v]]=mesh.targetLength[v];
    if(v<int(mesh.nx.size()))out.setNormal(map[v],mesh.normal(v));
    if(v<int(mesh.curvature.size()))out.curvature[map[v]]=mesh.curvature[v];
    if(v<int(mesh.featureDistance.size()))out.featureDistance[map[v]]=mesh.featureDistance[v];
  }
  for(int f=0;f<mesh.faceCount();++f)
    out.addFace(map[mesh.i0[f]],map[mesh.i1[f]],map[mesh.i2[f]],mesh.facePatchId[f],mesh.patches[mesh.facePatchId[f]].type);
  for(const auto &kv:mesh.featureEdges) {
    const int a=int(kv.first>>32),b=int(uint32_t(kv.first));
    if(a<int(map.size()) && b<int(map.size()) && map[a]>=0 && map[b]>=0 && map[a]!=map[b])
      out.featureEdges[key(map[a],map[b])]=kv.second;
  }
  rebuildRawTopology(out);
  mesh=std::move(out);
}void rebuild(SemanticMesh &mesh, const std::vector<Vertex> &vertices,
             const std::vector<Triangle> &faces,
             const std::unordered_map<uint64_t, uint32_t> &features, float h) {
  std::vector<int> map(vertices.size(), -1);
  std::vector<int> owner(vertices.size(),-1);
  for (auto t : faces)
    if (t.v[0] >= 0 && t.v[0] != t.v[1] && t.v[1] != t.v[2] && t.v[2] != t.v[0])
      for (int v : t.v) {
        map[v] = 0;
        if(owner[v]<0)owner[v]=t.patch;
      }
  SemanticMesh out;
  out.patches = mesh.patches;
  for (int v = 0; v < int(vertices.size()); ++v)
    if (map[v] == 0) {
      auto p = vertices[v].p;
      map[v] = out.addVertex({p.x, p.y, p.z}, owner[v],
                             VertexConstraint(vertices[v].constraint));
      out.targetLength[map[v]] = vertices[v].target;
    }
  for (auto t : faces)
    if (t.v[0] >= 0 && t.v[0] != t.v[1] && t.v[1] != t.v[2] && t.v[2] != t.v[0])
      out.addFace(map[t.v[0]], map[t.v[1]], map[t.v[2]], t.patch, out.patches[t.patch].type);
  for (auto kv : features) {
    int a = int(kv.first >> 32), b = int(uint32_t(kv.first));
    if (a < int(map.size()) && b < int(map.size()) && map[a] >= 0 &&
        map[b] >= 0 && map[a] != map[b])
      out.featureEdges[key(map[a], map[b])] = kv.second;
  }
  rebuildRawTopology(out);
  mesh = std::move(out);
}
__global__ void rejectSplitFaces(MeshView m,const Vertex *v,const Triangle *faces,
                                  int *marked,ReferenceSurfaceGpu ref,int *bad,
                                  float qualityRatio) {
  int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=m.nf)return;
  if(faces[4*f+1].v[0]<0)return;
  const auto parent=m.f[f];
  const float parentQuality=qualityVcg(m.v[parent.v[0]].p,m.v[parent.v[1]].p,
                                        m.v[parent.v[2]].p);
  for(int q=0;q<4;++q) {
    auto t=faces[4*f+q];if(t.v[0]<0)continue;
    auto a=v[t.v[0]].p,b=v[t.v[1]].p,c=v[t.v[2]].p;
    const float childQuality=qualityVcg(a,b,c);
    if(!(childQuality>0) ||
       (qualityRatio>0 && childQuality<qualityRatio*parentQuality) ||
       !referenceFaceSafe(ref,t.patch,a,b,c)) {
      atomicAdd(bad,1);
      for(int k=0;k<3;++k)atomicExch(marked+m.faceEdges[3*f+k],0);
      return;
    }
  }
}
int split(SemanticMesh &mesh, float h, float high, ReferenceSurfaceGpu ref,
          RemeshReport &report,bool optimizePoint,float splitQualityRatio) {
  DeviceMesh d(mesh, ref);
  auto m = d.view();
  Buffer<Vertex> v(m.nv + m.ne);
  Buffer<Triangle> f(4 * m.nf);
  Buffer<int> marked(m.ne);
  splitVertices<<<(std::max(m.nv, m.ne) + 127) / 128, 128,0,executionStream>>>(
      m, v.p, marked.p,high, ref,optimizePoint);
  Buffer<int> bad(1);
  // Read/write split masks are separate: neighboring threads may cancel the
  // same edge, but never read a mask while another thread writes it.
  Buffer<int> checkedMask(m.ne);
  bool stable=false;
  for(int pass=0;pass<16;++pass) {
    splitFaces<<<(m.nf+127)/128,128,0,executionStream>>>(m,v.p,marked.p,f.p);
    checked(streamCopy(checkedMask.p,marked.p,m.ne*sizeof(int),cudaMemcpyDeviceToDevice));
    checked(streamZero(bad.p,0,sizeof(int)));
    rejectSplitFaces<<<(m.nf+127)/128,128,0,executionStream>>>(
        m,v.p,f.p,checkedMask.p,ref,bad.p,splitQualityRatio);
    sync();const int rejected=bad.read()[0];report.rejectError+=rejected;
    if(!rejected){stable=true;break;}
    checked(streamCopy(marked.p,checkedMask.p,m.ne*sizeof(int),cudaMemcpyDeviceToDevice));
  }
  if(!stable)return 0;
  auto marks = marked.read();
  int count = std::count(marks.begin(), marks.end(), 1);
  if (!count)
    return 0;
  auto features = mesh.featureEdges;
  for (int e = 0; e < m.ne; ++e)
    if (marks[e] && (mesh.edges[e].flags & (EdgeSharp | EdgeMeshBoundary))) {
      auto x = mesh.edges[e];
      features.erase(key(x.v0, x.v1));
      features[key(x.v0, m.nv + e)] = x.featureCurveId;
      features[key(m.nv + e, x.v1)] = x.featureCurveId;
    }
  rebuild(mesh, v.read(), f.read(), features, h);
  return count;
}
__global__ void countCandidates(const Candidate *c,const int *chosen,int n,int *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) {if(chosen[i])atomicAdd(counts,1);if(c[i].keep>=0)atomicAdd(counts+1,1);}
}
struct TopologyTiming {
  double deviceMesh = 0;
  double candidate = 0;
  double schedule = 0;
  double apply = 0;
  double mapDownload = 0;
  double meshDownload = 0;
  double hostFeatureRemap = 0;
  double hostRebuild = 0;
  double downloadRebuild = 0;
  int calls = 0;
};
thread_local TopologyTiming collapseTiming;
thread_local TopologyTiming flipTiming;

int topology(SemanticMesh &mesh, float h, float low, float high,
             ReferenceSurfaceGpu ref, bool flip, bool relaxed, unsigned seed,
             RemeshReport &report,bool strictFlipQuality) {
  using TopologyClock = std::chrono::steady_clock;
  auto &timing = flip ? flipTiming : collapseTiming;
  ++timing.calls;
  auto timingMark = TopologyClock::now();
  DeviceMesh d(mesh, ref, false);
  auto m = d.view();
  timing.deviceMesh += std::chrono::duration<double>(TopologyClock::now()-timingMark).count();
  timingMark = TopologyClock::now();
  Buffer<Candidate> candidates(m.ne);
  Buffer<unsigned long long> claims(m.nv);
  Buffer<int> chosen(m.ne), mapping(m.nv);
  Buffer<int> flags(flip ? 0 : m.ne);
  Buffer<float> localLow(flip ? 0 : m.ne);
  if (flip) {
    flipCandidates<<<(m.ne + 127) / 128, 128,0,executionStream>>>(m, candidates.p, ref,strictFlipQuality);
  } else {
    const bool profileCandidateParts=(timing.calls==1 && !freezeBoundary);
    const auto prefilterMark=TopologyClock::now();
    collapsePrefilter<<<(m.ne + 127) / 128, 128,0,executionStream>>>(
        m,candidates.p,flags.p,localLow.p,low,ref,relaxed);
    double prefilterSeconds=0.0;
    if(profileCandidateParts) {
      sync();
      prefilterSeconds=std::chrono::duration<double>(TopologyClock::now()-prefilterMark).count();
    }
    const auto shapeMark=TopologyClock::now();
    collapseCandidatesShape<<<(m.ne + 127) / 128, 128,0,executionStream>>>(
        m,candidates.p,flags.p,relaxed);
    double shapeSeconds=0.0;
    if(profileCandidateParts) {
      sync();
      shapeSeconds=std::chrono::duration<double>(TopologyClock::now()-shapeMark).count();
    }
    const auto lengthMark=TopologyClock::now();
    collapseCandidatesLength<<<(m.ne + 127) / 128, 128,0,executionStream>>>(
        m,candidates.p,high,ref,relaxed);
    if(profileCandidateParts) {
      sync();
      const double lengthSeconds=std::chrono::duration<double>(TopologyClock::now()-lengthMark).count();
      std::cout << "collapse_candidate_parts prefilter_ms=" << prefilterSeconds*1000.0
                << " shape_ms=" << shapeSeconds*1000.0
                << " length_ms=" << lengthSeconds*1000.0 << std::endl;
    }  }
  sync();
  double candidateSeconds = std::chrono::duration<double>(TopologyClock::now()-timingMark).count();
  timing.candidate += candidateSeconds;
  timingMark = TopologyClock::now();
  int referenceBlockers=0, invalidReferenceBlockers=0, referenceRounds=0;
  if(flip) {
    initClaims<<<(m.nv + 127) / 128, 128,0,executionStream>>>(claims.p, m.nv);
    claimCandidates<<<(m.ne + 127) / 128, 128,0,executionStream>>>(m,candidates.p,claims.p,true,seed);
    resolveCandidates<<<(m.ne + 127) / 128, 128,0,executionStream>>>(m,candidates.p,claims.p,chosen.p,true,seed);
  } else {
    Buffer<int> referenceStats(2);
    referenceRounds=1;
    initClaims<<<(m.nv + 127) / 128, 128,0,executionStream>>>(claims.p, m.nv);
    claimCandidates<<<(m.ne + 127) / 128, 128,0,executionStream>>>(m,candidates.p,claims.p,false,seed);
    resolveCandidates<<<(m.ne + 127) / 128, 128,0,executionStream>>>(m,candidates.p,claims.p,chosen.p,false,seed);
    checked(streamZero(referenceStats.p,0,2*sizeof(int)));
    validateChosenCollapse<<<(m.ne+127)/128,128,0,executionStream>>>(
        m,candidates.p,chosen.p,high,ref,relaxed,referenceStats.p);
    sync();
    const auto rs=referenceStats.read();
    referenceBlockers=rs[0];
    invalidReferenceBlockers=rs[1];
  }  Buffer<int> counts(2);
  checked(streamZero(counts.p,0,2*sizeof(int)));
  countCandidates<<<(m.ne+127)/128,128,0,executionStream>>>(candidates.p,chosen.p,m.ne,counts.p);
  sync();
  const auto totals=counts.read();
  timing.schedule += std::chrono::duration<double>(TopologyClock::now()-timingMark).count();
  const int count=totals[0];
  if(flip && !freezeBoundary) {
    std::cout << "raw_flip_pass edges=" << m.ne
              << " candidates=" << totals[1]
              << " accepted=" << count
              << " candidate_ms=" << candidateSeconds*1000.0 << std::endl;
  }
  if(!flip && !freezeBoundary) {
    std::cout << "raw_collapse_pass edges=" << m.ne
              << " candidates=" << totals[1]
              << " accepted=" << count
              << " candidate_ms=" << candidateSeconds*1000.0
              << " ref_blockers=" << referenceBlockers
              << " ref_invalid_blockers=" << invalidReferenceBlockers
              << " ref_rounds=" << referenceRounds
              << " relaxed=" << (relaxed?1:0) << std::endl;
  }
  if(flip)report.flipCandidates+=totals[1];else report.collapseCandidates+=totals[1];
  if (!count)
    return 0;
  timingMark = TopologyClock::now();
  if(!flip) initMap<<<(m.nv + 127) / 128, 128,0,executionStream>>>(mapping.p, m.nv);
  applyCandidates<<<(m.ne + 127) / 128, 128,0,executionStream>>>(m, candidates.p, chosen.p,
                                               mapping.p, flip);
  if (!flip)
    remapFaces<<<(m.nf + 127) / 128, 128,0,executionStream>>>(m, mapping.p);
  sync();
  timing.apply += std::chrono::duration<double>(TopologyClock::now()-timingMark).count();
  timingMark = TopologyClock::now();
  if(flip) {
    // Flips neither delete vertices nor change coordinates/feature identities.
    const auto faces=d.faces.read();
    for(int f=0;f<m.nf;++f) {mesh.i0[f]=faces[f].v[0];mesh.i1[f]=faces[f].v[1];mesh.i2[f]=faces[f].v[2];mesh.facePatchId[f]=uint32_t(faces[f].patch);}
    rebuildRawTopology(mesh,false);
    timing.downloadRebuild += std::chrono::duration<double>(TopologyClock::now()-timingMark).count();
    return count;
  }
  auto features = mesh.featureEdges;
  if (!flip) {
    auto subMark=TopologyClock::now();
    auto map = mapping.read();
    timing.mapDownload += std::chrono::duration<double>(TopologyClock::now()-subMark).count();
    subMark=TopologyClock::now();
    features.clear();
    for (auto kv : mesh.featureEdges) {
      int a = map[kv.first >> 32], b = map[uint32_t(kv.first)];
      if (a != b)
        features[key(a, b)] = kv.second;
    }
    timing.hostFeatureRemap += std::chrono::duration<double>(TopologyClock::now()-subMark).count();
  }
  auto subMark=TopologyClock::now();
  auto hostVertices=d.vertices.read();
  auto hostFaces=d.faces.read();
  timing.meshDownload += std::chrono::duration<double>(TopologyClock::now()-subMark).count();
  subMark=TopologyClock::now();
  rebuildSparse(mesh, hostVertices, hostFaces, features, false);
  timing.hostRebuild += std::chrono::duration<double>(TopologyClock::now()-subMark).count();
  timing.downloadRebuild += std::chrono::duration<double>(TopologyClock::now()-timingMark).count();
  return count;
}
int smooth(SemanticMesh &mesh, ReferenceSurfaceGpu ref, float lambda,
           unsigned seed, int smoothPassCount, int smoothAttempts, RemeshReport &report) {
  DeviceMesh d(mesh, ref);
  auto baseView=d.view();
  Buffer<Vertex> original(baseView.nv);
  checked(streamCopy(original.p,baseView.v,baseView.nv*sizeof(Vertex),cudaMemcpyDeviceToDevice));
  Buffer<int> projectionHint(baseView.nv);
  checked(streamZero(projectionHint.p,0xff,baseView.nv*sizeof(int)));
  int acceptedMoved=0;
  double acceptedKernelSeconds=0.0,acceptedSafeProjectSeconds=0.0;
  float acceptedRatio=0.f,acceptedLambda=0.f;
  std::vector<int> acceptedPasses(smoothPassCount,0);
  bool accepted=false;
  for(int attempt=0;attempt<smoothAttempts && !accepted;++attempt) {
    auto m=d.view();
    checked(streamCopy(m.v,original.p,m.nv*sizeof(Vertex),cudaMemcpyDeviceToDevice));
    Buffer<Vertex> next(m.nv);
    Buffer<int> moved(smoothPassCount);
    Buffer<int> passReject(m.nv),passBad(2),passRejected(1),dirtyQueue(m.nf),dirtyCount(1);
    checked(streamZero(moved.p,0,smoothPassCount*sizeof(int)));
    const float trialLambda=lambda*powf(.5f,float(attempt));
    const auto kernelStart=std::chrono::steady_clock::now();
    for(int pass=0;pass<smoothPassCount;++pass) {
      const bool profileSmoothParts=(!freezeBoundary && attempt==0 && pass==0);
      auto smoothPartMark=std::chrono::steady_clock::now();
      checked(streamZero(dirtyCount.p,0,sizeof(int)));
      smoothVertices<<<(m.nv+127)/128,128,0,executionStream>>>(m,next.p,ref,trialLambda,
                                                   seed+pass,moved.p+pass,projectionHint.p,dirtyQueue.p,dirtyCount.p);
      if(profileSmoothParts){sync();std::cout << " smooth_move_ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-smoothPartMark).count();smoothPartMark=std::chrono::steady_clock::now();}
      checked(streamZero(passReject.p,0,m.nv*sizeof(int)));
      checked(streamZero(passBad.p,0,2*sizeof(int)));
      checked(streamZero(passRejected.p,0,sizeof(int)));
      checkProjectionFaceReferenceExactQueue<<<256,128,0,executionStream>>>(m,next.p,ref,passReject.p,passBad.p,dirtyQueue.p,dirtyCount.p);
      if(profileSmoothParts){sync();std::cout << " reference_ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-smoothPartMark).count();smoothPartMark=std::chrono::steady_clock::now();}
      revertProjection<<<(m.nv+127)/128,128,0,executionStream>>>(m,next.p,passReject.p,passRejected.p);
      if(profileSmoothParts){sync();std::cout << " revert_ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-smoothPartMark).count() << std::endl;}
      std::swap(m.v,next.p);
    }
    sync();
    const double kernelSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-kernelStart).count();
    int localRejectError=0,localRejectQuality=0;
    double safeSeconds=0.0;
    Buffer<unsigned int> maximum(2);
    auto auditRatio=[&]() {
      checked(streamZero(maximum.p,0,2*sizeof(unsigned int)));
      RunGeometryAudit(m, ref, maximum.p);
      sync();
      const auto audit=maximum.read();
      float value=0.f;std::memcpy(&value,&audit[1],sizeof(float));
      return value;
    };
    float ratio=auditRatio();
    if(ratio>1.000001f) {
      const auto safeStart=std::chrono::steady_clock::now();
      localRejectQuality=safeProject(m,ref,&localRejectError,projectionHint.p);
      sync();
      safeSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-safeStart).count();
      ratio=auditRatio();
    }    if(ratio<=1.000001f) {
      const auto movedPasses=moved.read();
      for(int pass=0;pass<smoothPassCount;++pass){acceptedPasses[pass]=movedPasses[pass];acceptedMoved+=movedPasses[pass];}
      acceptedKernelSeconds=kernelSeconds;acceptedSafeProjectSeconds=safeSeconds;
      acceptedRatio=ratio;acceptedLambda=trialLambda;
      report.rejectQuality+=localRejectQuality;report.rejectError+=localRejectError;
      accepted=true;
    } else if(!freezeBoundary) {
      std::cout << "raw_smooth_retry attempt=" << attempt
                << " lambda=" << trialLambda << " ratio=" << ratio << std::endl;
    }
  }
  auto finalView=d.view();
  if(!accepted) {
    checked(streamCopy(finalView.v,original.p,finalView.nv*sizeof(Vertex),cudaMemcpyDeviceToDevice));
    sync();acceptedRatio=0.f;acceptedLambda=0.f;acceptedMoved=0;
  }
  const auto downloadStart=std::chrono::steady_clock::now();
  std::vector<Vertex> verts(finalView.nv);
  checked(streamCopy(verts.data(),finalView.v,finalView.nv*sizeof(Vertex),cudaMemcpyDeviceToHost));
  for(int i=0;i<finalView.nv;++i){auto p=verts[i].p;mesh.setPosition(i,{p.x,p.y,p.z});mesh.targetLength[i]=verts[i].target;}
  const double downloadSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-downloadStart).count();
  if(!freezeBoundary) {
    std::cout << "raw_smooth_pass_moves";
    for(int pass=0;pass<smoothPassCount;++pass)std::cout << " p" << pass << "=" << acceptedPasses[pass];
    std::cout << " lambda=" << acceptedLambda << " ratio=" << acceptedRatio
              << " kernel_s=" << acceptedKernelSeconds
              << " safe_project_s=" << acceptedSafeProjectSeconds
              << " download_s=" << downloadSeconds << std::endl;
  }
  return acceptedMoved;
}void buildReferenceBvh(const std::vector<ReferenceTriangleGpu> &triangles,float boxPad,
                       std::vector<ReferenceBvhNode> &tree,std::vector<int> &sourceIds,
                       const std::vector<float> *primitiveTargets=nullptr) {
    tree.clear();sourceIds.resize(triangles.size());
    std::iota(sourceIds.begin(),sourceIds.end(),0);
    auto coord=[](float3 p,int axis){return axis==0?p.x:(axis==1?p.y:p.z);};
    std::function<void(int,int)> buildTree=[&](int begin,int end) {
      const int id=int(tree.size());tree.emplace_back();
      float3 lo=make_float3(FLT_MAX,FLT_MAX,FLT_MAX),hi=make_float3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
      float minTarget=FLT_MAX;
      for(int i=begin;i<end;++i) {
        const int primitiveId=sourceIds[i];
        if(primitiveTargets)minTarget=std::min(minTarget,(*primitiveTargets)[primitiveId]);
        const auto t=triangles[primitiveId];for(auto p:{t.a,t.b,t.c}) {
        lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);
        hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);
      }}
      tree[id].lower=make_float3(lo.x-boxPad,lo.y-boxPad,lo.z-boxPad);
      tree[id].upper=make_float3(hi.x+boxPad,hi.y+boxPad,hi.z+boxPad);
      tree[id].minTarget=minTarget;
      if(end-begin<=4) {tree[id].first=begin;tree[id].count=end-begin;}
      else {
        int axis=0;if(hi.y-lo.y>hi.x-lo.x)axis=1;if(hi.z-lo.z>coord(hi,axis)-coord(lo,axis))axis=2;
        int mid=(begin+end)/2;
        std::nth_element(sourceIds.begin()+begin,sourceIds.begin()+mid,sourceIds.begin()+end,[&](int a,int b) {
          const auto ta=triangles[a],tb=triangles[b];
          const float ca=coord(ta.a,axis)+coord(ta.b,axis)+coord(ta.c,axis);
          const float cb=coord(tb.a,axis)+coord(tb.b,axis)+coord(tb.c,axis);
          return ca<cb || (ca==cb && a<b);
        });
        buildTree(begin,mid);buildTree(mid,end);
      }
      tree[id].escape=int(tree.size());
    };
    if(!triangles.empty())buildTree(0,int(triangles.size()));
}
} // namespace raw

bool remeshRawCuda(SemanticMesh &m,const RemeshConfig &c,RemeshReport &r) {
  return remeshRawCuda(m,c,r,{});
}

bool remeshRawCuda(SemanticMesh &mesh, const RemeshConfig &cfg,
                   RemeshReport &report, const RawCudaOptions &options) {
  using namespace raw;
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  StreamScope streamScope(options.independentStream);
  struct BoundaryScope {bool old=freezeBoundary;~BoundaryScope(){freezeBoundary=old;}} boundaryScope;
  freezeBoundary=options.freezeBoundary;
  DeviceArena arena;arena.limit=options.workspaceBytes;
  ArenaScope arenaScope(arena);
  stGeometryAuditTiming = {};
  report = {};
  collapseTiming = {};
  flipTiming = {};
  try {
    if (mesh.LocalSizing || (cfg.adaptive && !(cfg.featureEdgeLength > 0)))
      throw std::runtime_error(
          "raw CUDA supports uniform sizing or explicit featureEdgeLength; CAD "
          "LocalSizing is unsupported");
    mesh.rebuildTopology();
    std::string error;
    if (!mesh.validate(&error))
      throw std::runtime_error(error);
    std::vector<Vec3> lockedBefore;
    for (int v = 0; v < mesh.vertexCount(); ++v)
      if (mesh.vertexConstraint[v] >= uint8_t(VertexConstraint::Corner))
        lockedBefore.push_back(mesh.position(v));
    const float h = cfg.constantLength > 0 ? cfg.constantLength
                                           : mesh.bboxDiagonal() * .01f;
    if (!(h > 0) || !std::isfinite(h) || !(cfg.maxGeometryError > 0) ||
        !std::isfinite(cfg.maxGeometryError) || cfg.maxIterations < 1 ||
        options.smoothPasses < 1 || options.smoothPasses > 12 ||
        options.smoothAttempts < 1 || options.smoothAttempts > 3 ||
        options.collapsePasses < 1 || options.collapsePasses > 8 ||
        options.flipPasses < 1 || options.flipPasses > 8 ||
        !(options.splitQualityRatio>=0.f && options.splitQualityRatio<=1.f))
      throw std::runtime_error(
          "invalid raw CUDA sizing, error budget or iteration count");
    const auto setupFeatureStart=Clock::now();
    std::vector<Vec3> normals(mesh.faceCount());
    for(int f=0;f<mesh.faceCount();++f)
      normals[f]=triangleNormal(mesh.facePoint(f,0),mesh.facePoint(f,1),mesh.facePoint(f,2));
    const float cosAngle =
        std::cos(cfg.featureAngleDegrees * .01745329251994329577f);
    for (auto e : mesh.edges) {
      bool feature = e.face1 < 0;
      if (e.face1 >= 0)
        feature = dot(normals[e.face0],normals[e.face1]) <= cosAngle;
      if (feature)
        mesh.featureEdges[key(e.v0, e.v1)] = 0;
    }
    rebuildRawTopology(mesh);
    std::vector<int> featureDegree(mesh.vertexCount());
    for (auto e : mesh.edges)
      if (e.flags & (EdgeSharp | EdgeMeshBoundary)) {
        ++featureDegree[e.v0];
        ++featureDegree[e.v1];
      }
    for (int v = 0; v < mesh.vertexCount(); ++v) {
      if (mesh.vertexConstraint[v] >= 4)
        continue;
      mesh.vertexConstraint[v] =
          featureDegree[v]
              ? uint8_t(featureDegree[v] == 2 ? VertexConstraint::FeatureEdge
                                              : VertexConstraint::Corner)
              : uint8_t(VertexConstraint::Surface);
    }
    const auto setupFeatureSeconds=std::chrono::duration<double>(Clock::now()-setupFeatureStart).count();
    const auto setupReferenceStart=Clock::now();
    std::vector<ReferenceTriangleGpu> triangles;
    for (int f = 0; f < mesh.faceCount(); ++f)
      triangles.push_back({point(mesh.facePoint(f, 0)),
                           point(mesh.facePoint(f, 1)),
                           point(mesh.facePoint(f, 2)), int(mesh.facePatchId[f])});
    std::vector<ReferenceBvhNode> tree;
    std::vector<int> sourceIds;
    buildReferenceBvh(triangles,mesh.bboxDiagonal()*1.e-6f,tree,sourceIds);
    Buffer<ReferenceBvhNode> bvh(tree);
    Buffer<int> triangleIds(sourceIds);
    Buffer<ReferenceTriangleGpu> reference(triangles);
    std::vector<SizingSegmentGpu> seeds;
    const auto setupReferenceSeconds=std::chrono::duration<double>(Clock::now()-setupReferenceStart).count();
    const auto setupSizingStart=Clock::now();
    const bool refine = cfg.featureEdgeLength > 0;
    const float fine = cfg.featureEdgeLength;
    const float band = cfg.featureBand > 0 ? cfg.featureBand : .75f * h;
    if (refine && (!(fine < h) || !std::isfinite(fine) || !std::isfinite(band)))
      throw std::runtime_error("feature length must be positive and below "
                               "regular length; band must be finite");
    // Join coplanar triangles before interpreting dihedrals. An isolated
    // shallow plane/plane junction is not evidence of a curved surface; a
    // sampled curve has a chain of changing normals through multiple regions.
    std::vector<int> region(mesh.faceCount());
    for(int f=0;f<mesh.faceCount();++f) region[f]=f;
    auto root=[&](int f) {while(region[f]!=f) {region[f]=region[region[f]];f=region[f];}return f;};
    const float coplanarCos=std::cos(.01745329252f);
    if(refine) for(auto e:mesh.edges) if(e.face0>=0 && e.face1>=0 &&
        e.patchLeft==e.patchRight &&
        dot(normals[e.face0],normals[e.face1])>=coplanarCos)
      region[root(e.face1)]=root(e.face0);
    std::vector<int> regionRoot(mesh.faceCount());
    for(int f=0;f<mesh.faceCount();++f)regionRoot[f]=root(f);
    std::vector<int> smoothNeighbor0(mesh.faceCount(),-1),smoothNeighbor1(mesh.faceCount(),-1);
    auto addSmoothNeighbor=[&](int r,int n) {
      if(r<0 || n<0 || r==n || smoothNeighbor0[r]==n || smoothNeighbor1[r]==n)return;
      if(smoothNeighbor0[r]<0)smoothNeighbor0[r]=n;
      else if(smoothNeighbor1[r]<0)smoothNeighbor1[r]=n;
    };
    if(refine) for(auto e:mesh.edges) if(e.face0>=0 && e.face1>=0 &&
        !(e.flags&(EdgeSharp|EdgeMeshBoundary|EdgePatchBoundary))) {
      const int a=regionRoot[e.face0], b=regionRoot[e.face1];
      if(a!=b) {addSmoothNeighbor(a,b);addSmoothNeighbor(b,a);}
    }    if (refine)
      for (auto e : mesh.edges) {
        const auto a = mesh.position(e.v0), b = mesh.position(e.v1);
        float local = h;
        // A crease is a geometric constraint, not a curvature sample. In
        // particular, plane/plane intersections must retain the regular size.
        if (!(e.flags & (EdgeSharp | EdgeMeshBoundary | EdgePatchBoundary)) && e.face0 >= 0 && e.face1 >= 0) {
          const auto n0 = normals[e.face0];
          const auto n1 = normals[e.face1];
          const float angle = std::acos(clampf(dot(n0, n1), -1.f, 1.f));
          if (angle > .0174533f &&
              (smoothNeighbor1[regionRoot[e.face0]]>=0 || smoothNeighbor1[regionRoot[e.face1]]>=0)) {
            // Dual width across the edge, rather than its possibly very long
            // axial length.
            float width = 0;
            for (int f : {e.face0, e.face1})
              for (auto v : mesh.face(f))
                if (v != e.v0 && v != e.v1)
                  width += .5f * length(cross(b - a, mesh.position(v) - a)) /
                           std::max(distance(a, b), 1.e-20f);
            float curvature = angle / std::max(width, 1.e-12f);
            local = clampf(
                std::min(std::sqrt(8.f * std::min(cfg.maxGeometryError, .02f * h) /
                          curvature), cfg.normalDegrees * .01745329252f / curvature),
                fine, h);
          }
        }
        if (local < h)
          seeds.push_back({point(a), point(b), local,int(e.patchLeft)});
      }
    const auto setupSizingFieldSeconds=std::chrono::duration<double>(Clock::now()-setupSizingStart).count();
    const auto setupSizingBvhStart=Clock::now();
    std::vector<ReferenceTriangleGpu> seedBounds;
    std::vector<float> seedTargets;seedTargets.reserve(seeds.size());
    for(auto seed:seeds){seedBounds.push_back({seed.a,seed.b,seed.a,seed.patch});seedTargets.push_back(seed.target);}
    std::vector<ReferenceBvhNode> sizeTree;std::vector<int> sizeIds;
    buildReferenceBvh(seedBounds,mesh.bboxDiagonal()*1.e-6f,sizeTree,sizeIds,&seedTargets);
    Buffer<ReferenceBvhNode> sizingBvh(sizeTree);Buffer<int> sizingIds(sizeIds);
    Buffer<SizingSegmentGpu> sizing(seeds);
    ReferenceSurfaceGpu ref{reference.p, int(reference.n), cfg.maxGeometryError,
                            sizing.p,    int(sizing.n),    h,
                            band};
    ref.sizingNodes=sizingBvh.p;ref.sizingNodeCount=int(sizeTree.size());ref.sizingIds=sizingIds.p;
    ref.nodes=bvh.p;ref.nodeCount=int(tree.size());ref.triangleIds=triangleIds.p;
    const auto setupSizingBvhSeconds=std::chrono::duration<double>(Clock::now()-setupSizingBvhStart).count();
    const auto setupTargetStart=Clock::now();
    refreshHostTargets(mesh,ref);
    const auto setupTargetSeconds=std::chrono::duration<double>(Clock::now()-setupTargetStart).count();
    if(!options.quiet) std::cout<<"reference_query=bvh nodes="<<ref.nodeCount<<std::endl;
    if(!options.quiet) std::cout << "raw_sizing="
              << (refine ? "curvature_only" : "uniform")
              << " seeds=" << seeds.size()
              << " feature_length=" << (refine ? fine : h)
              << " transition_band=" << band << std::endl;
    report.secondsSetup =
        std::chrono::duration<double>(Clock::now() - start).count();
    if(!options.quiet) std::cout << "raw_setup_parts feature=" << setupFeatureSeconds
              << " reference=" << setupReferenceSeconds
              << " sizing_field=" << setupSizingFieldSeconds
              << " sizing_bvh=" << setupSizingBvhSeconds
              << " target_refresh=" << setupTargetSeconds << std::endl;
    if(!options.quiet) std::cout << "raw_gpu_policy=triangle_templates reference_faces="
              << triangles.size()
              << " feature_edges=" << mesh.featureEdges.size()
              << " host_adjacency=true" << std::endl;
    int currentCycle=-1;
    const char *stage="setup";
    auto validate = [&]() {
      const auto validationStart=Clock::now();
      if (!mesh.validate(&error))
        throw std::runtime_error("raw CUDA topology: cycle="+std::to_string(currentCycle)+" stage="+stage+": " + error);
      report.secondsValidate+=std::chrono::duration<double>(Clock::now()-validationStart).count();
    };
    auto qualitySummary=[&](const SemanticMesh &candidate) {
      std::vector<float> values;values.reserve(candidate.faceCount());
      double sum=0;
      for(int f=0;f<candidate.faceCount();++f)if(candidate.faceAlive[f]) {
        const auto t=candidate.face(f);
        const float q=triangleQuality(candidate.position(t[0]),candidate.position(t[1]),candidate.position(t[2]));
        values.push_back(q);sum+=q;
      }
      if(values.empty())return std::pair<float,float>{0.f,0.f};
      const size_t p05=values.size()/20;
      std::nth_element(values.begin(),values.begin()+p05,values.end());
      return std::pair<float,float>{float(sum/double(values.size())),values[p05]};
    };
    SemanticMesh bestMesh;
    float bestMean=-1,bestP05=-1;
    auto considerBest=[&](int selected) {
      const auto [mean,p05]=qualitySummary(mesh);
      if(mean+1.e-6f<options.qualityMeanFloor || p05+1.e-6f<options.qualityP05Floor)
        return std::pair<float,float>{mean,p05};
      if(bestP05>=0 && (p05<bestP05-1.e-6f ||
                         (std::abs(p05-bestP05)<=1.e-6f && mean<=bestMean)))
        return std::pair<float,float>{mean,p05};
      bestMesh=mesh;bestMean=mean;bestP05=p05;report.selectedCycle=selected;
      return std::pair<float,float>{mean,p05};
    };
    const bool trackQuality=options.freezeBoundary &&
        (options.qualityMeanFloor>0.f || options.qualityP05Floor>0.f);
    if(trackQuality)considerBest(-1);
    int severeQualityStallCycles=0;
    for (int cycle = 0; cycle < cfg.maxIterations; ++cycle) {
      // A late topology failure must not discard all earlier valid cycles of
      // a frozen-boundary region. The source boundary identity is unchanged.
      SemanticMesh checkpoint;
      if(options.freezeBoundary && cycle>0)checkpoint=mesh;
      const int priorSplits=report.splits,priorCollapses=report.collapses;
      const int priorFlips=report.flips,priorSmoothMoves=report.smoothMoves;
      try {
      currentCycle=cycle;
      report.cyclesExecuted=cycle+1;
      int ns = 0, nc = 0, nf = 0, nm = 0;
      auto mark = Clock::now();
      stage="split";
      if (cfg.enableSplit) {
        ns = split(mesh, h, cfg.splitRatio * h, ref,report,
                   options.optimizeSplitPoint,options.splitQualityRatio);
        report.splits += ns;
        report.splitCandidates += ns;
        validate();
      }
      report.secondsSplit +=
          std::chrono::duration<double>(Clock::now() - mark).count();
      mark = Clock::now();
      stage="collapse";
      if (cfg.enableCollapse) {
        for (int pass = 0; pass < options.collapsePasses; ++pass) {
          const size_t edgesBefore=mesh.edges.size();
          const auto passStart=Clock::now();
          int n =
              topology(mesh, h, cfg.collapseRatio * h, cfg.splitRatio * h, ref,
                       false, false, unsigned(cycle * 31 + pass + 1), report,options.strictFlipQuality);
          ++report.collapsePassCalls[pass];
          report.collapsePassAccepted[pass]+=n;
          report.collapsePassSeconds[pass]+=std::chrono::duration<double>(Clock::now()-passStart).count();
          nc += n;
          const double acceptRatio=edgesBefore?double(n)/double(edgesBefore):0.0;
          if (!n || (pass>=2 && acceptRatio<5.0e-4))
            break;
        }
        int n = topology(mesh, h, cfg.collapseRatio * h, cfg.splitRatio * h,
                         ref, false, true, unsigned(cycle + 717), report,options.strictFlipQuality);
        nc += n;
        report.collapses += nc;
        const auto compactStart=Clock::now();
        compactRawVertices(mesh);
        report.secondsCompact+=std::chrono::duration<double>(Clock::now()-compactStart).count();
        validate();
      }
      report.secondsCollapse +=
          std::chrono::duration<double>(Clock::now() - mark).count();
      mark = Clock::now();
      stage="flip";
      if (cfg.enableFlip) {
        for (int pass = 0; pass < options.flipPasses; ++pass) {
          const auto passStart=Clock::now();
          int n =
              topology(mesh, h, cfg.collapseRatio * h, cfg.splitRatio * h, ref,
                       true, false, unsigned(cycle * 37 + pass + 1), report,options.strictFlipQuality);
          ++report.flipPassCalls[pass];
          report.flipPassAccepted[pass]+=n;
          report.flipPassSeconds[pass]+=std::chrono::duration<double>(Clock::now()-passStart).count();
          nf += n;
          if (!n)
            break;
        }
        report.flips += nf;
        validate();
      }
      report.secondsFlip +=
          std::chrono::duration<double>(Clock::now() - mark).count();
      mark = Clock::now();
      stage="smooth_projection";
      // Frozen batch boundaries cannot move. A region whose current vertices
      // are all constrained has no smoothing candidates, so avoid the GPU
      // smoothing retries and host/device copies for that region.
      const bool hasMovableVertex = !options.freezeBoundary ||
          std::any_of(mesh.vertexConstraint.begin(), mesh.vertexConstraint.end(),
                      [](uint8_t constraint) { return constraint < 2u; });
      if (cfg.enableSmooth && hasMovableVertex) {
        nm = smooth(mesh, ref, cfg.smoothLambda, unsigned(cycle * 12 + 1),options.smoothPasses,options.smoothAttempts,report);
        report.smoothMoves += nm;
      } else if (!cfg.enableSmooth) {
        DeviceMesh d(mesh, ref);
        auto m = d.view();
        report.rejectQuality += safeProject(m,ref,&report.rejectError);
        sync();
        auto v = d.vertices.read();
        for (int i = 0; i < m.nv; ++i) {
          auto p = v[i].p;
          mesh.setPosition(i, {p.x, p.y, p.z});
        }
      }
      validate();
      report.secondsSmooth +=
          std::chrono::duration<double>(Clock::now() - mark).count();
      if(trackQuality) {
        const auto [cycleMean,cycleP05]=considerBest(cycle);
        if(cycleMean<.75f*options.qualityMeanFloor &&
           cycleP05<.01f*options.qualityP05Floor)
          ++severeQualityStallCycles;
        else
          severeQualityStallCycles=0;
        // A region with both metrics far below its input for several full
        // cycles is not recovering. Preserve the best accepted mesh and let
        // the batch controller try its next sizing/split strategy.
        if(severeQualityStallCycles>=3) {
          if(!options.quiet)std::cout<<"raw_quality_stall_stop cycle="<<cycle
                                     <<" mean="<<cycleMean<<" p05="<<cycleP05<<std::endl;
          break;
        }
      }
      if(!options.quiet) std::cout << "raw_gpu_cycle=" << cycle << " faces=" << mesh.faceCount()
                 << " split=" << ns << " collapse=" << nc << " flip=" << nf
                 << " smooth=" << nm << std::endl;
      // With unchanged topology and positions the next cycle sees the same
      // reference, sizing field and constraints. Avoid repeating the full
      // CUDA setup and validation on converged regions of a partition batch.
      if (options.stopWhenIdle && ns==0 && nc==0 && nf==0 && nm==0) break;
      if(trackQuality && report.selectedCycle>=0 && cycle>=10 &&
         cycle-report.selectedCycle>=5)break;
      } catch(const std::exception &cycleError) {
        const std::string message=cycleError.what();
        if(!options.freezeBoundary || cycle==0 ||
           message.rfind("raw CUDA topology:",0)!=0)throw;
        mesh=std::move(checkpoint);
        mesh.rebuildTopology();
        report.splits=priorSplits;report.collapses=priorCollapses;
        report.flips=priorFlips;report.smoothMoves=priorSmoothMoves;
        report.cyclesExecuted=cycle;
        ++report.recoveredCycleFailures;
        if(!options.quiet)std::cout<<"raw_recovered_cycle="<<cycle
                                    <<" reason="<<message<<std::endl;
        break;
      }
    }
    if(trackQuality && bestP05>=0)mesh=std::move(bestMesh);
    validate();
    float localErrorRatio=0;
    {
      DeviceMesh d(mesh, ref);
      auto m = d.view();
      Buffer<unsigned int> maximum(2);
      checked(streamZero(maximum.p, 0, 2*sizeof(unsigned int)));
      RunGeometryAudit(m, ref, maximum.p);
      sync();
      auto audit = maximum.read();
      auto bits = audit[0];
      std::memcpy(&localErrorRatio,&audit[1],sizeof(float));
      if(refine && !options.quiet) std::cout<<"raw_local_error_ratio="<<localErrorRatio<<std::endl;
      static_assert(sizeof(bits) == sizeof(report.geometryErrorMax));
      std::memcpy(&report.geometryErrorMax, &bits, sizeof(bits));
    }
    if (refine) {
      DeviceMesh d(mesh, ref);
      sync();
      const auto vertices = d.vertices.read();
      for (int v = 0; v < mesh.vertexCount(); ++v)
        mesh.targetLength[v] = vertices[v].target;
    }
    const float referenceAuditError=report.geometryErrorMax;
    fillMeshMetrics(mesh, cfg, report, refine);
    report.topologyValid = true;
    for (auto p : lockedBefore) {
      bool found = false;
      for (int v = 0; v < mesh.vertexCount(); ++v)
        if (distance(p, mesh.position(v)) <= 1.e-6f) {
          found = true;
          break;
        }
      if (!found)
        ++report.movedLockedVertices;
    }
    report.constraintsHeld =
        report.movedLockedVertices == 0 && localErrorRatio <= 1.001f &&
        report.geometryErrorMax <=
            cfg.maxGeometryError + mesh.bboxDiagonal() * 1.e-6f;
    report.seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    if(!options.quiet) {
      const auto printTopologyTiming=[](const char *name,const TopologyTiming &t) {
        std::cout << "raw_gpu_topology_timing stage=" << name
                  << " calls=" << t.calls
                  << " device_mesh=" << t.deviceMesh
                  << " candidate=" << t.candidate
                  << " schedule=" << t.schedule
                  << " apply=" << t.apply
                  << " map_download=" << t.mapDownload
                  << " mesh_download=" << t.meshDownload
                  << " host_feature_remap=" << t.hostFeatureRemap
                  << " host_rebuild=" << t.hostRebuild
                  << " download_rebuild=" << t.downloadRebuild
                  << " total=" << (t.deviceMesh+t.candidate+t.schedule+t.apply+t.downloadRebuild)
                  << std::endl;
      };
      printTopologyTiming("collapse",collapseTiming);
      printTopologyTiming("flip",flipTiming);
      const auto &auditTiming = stGeometryAuditTiming;
      std::cout << "raw_geometry_audit calls=" << auditTiming.Calls
                << " verified_calls=" << auditTiming.VerifiedCalls
                << " legacy_sample_visits=" << auditTiming.LegacySampleVisits
                << " unique_sample_slots=" << auditTiming.UniqueSampleSlots
                << " legacy_wall_s=" << auditTiming.LegacySeconds
                << " unique_wall_s=" << auditTiming.UniqueSeconds << std::endl;

    }
    if(!report.constraintsHeld && options.error)
      *options.error="final raw CUDA geometry or locked-vertex constraints failed: reference="+
        std::to_string(referenceAuditError)+" analytic="+std::to_string(report.geometryErrorMax)+
        " locked="+std::to_string(report.movedLockedVertices)+
        " ratio="+std::to_string(localErrorRatio);
    return report.constraintsHeld;
  } catch (const std::exception &e) {
    if(options.error)*options.error=e.what();
    if(!options.quiet) std::cerr << "[raw CUDA] " << e.what() << '\n';
    report.seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    return false;
  }
}
} // namespace cad_adaptive
