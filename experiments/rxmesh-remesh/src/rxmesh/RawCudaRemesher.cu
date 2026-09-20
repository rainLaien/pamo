// SPDX-License-Identifier: GPL-2.0-or-later
// Refinement templates and isotropic operator rules follow VCGLib:
// https://github.com/cnr-isti-vclab/vcglib (ISTI-CNR Visual Computing Lab).
// CUDA conflict selection, storage and orchestration are implemented here.
#include "cad_adaptive/RawCudaRemesher.h"
#include "cad_adaptive/RemeshMetrics.h"
#include "raw_reference.cuh"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace cad_adaptive {
namespace raw {
using namespace gpu;
void checked(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
void sync() {
  checked(cudaGetLastError());
  checked(cudaDeviceSynchronize());
}
// Per-call, per-thread scratch cache. CUDA allocation/free synchronization is
// paid once per capacity class rather than once per topology pass.
struct DeviceArena {
  struct Block {void *p; size_t bytes;};
  std::vector<Block> available;
  static size_t capacity(size_t bytes) {size_t n=256;while(n<bytes)n*=2;return n;}
  void *acquire(size_t bytes) {
    for(size_t i=0;i<available.size();++i) if(available[i].bytes==bytes) {
      void *p=available[i].p;available[i]=available.back();available.pop_back();return p;
    }
    void *p=nullptr;checked(cudaMalloc(&p,bytes));return p;
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
      checked(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
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
      checked(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
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
};
struct Edge {
  int a, b, f0, f1, feature;
};
struct Candidate {
  int keep = -1, remove = -1, c = -1, d = -1;
  float3 p;
};
struct MeshView {
  Vertex *v;
  Triangle *f;
  Edge *e;
  int *offsets;
  int *ids;
  int *faceEdges;
  int nv, nf, ne;
};
uint64_t key(int a, int b) {
  if (a > b)
    std::swap(a, b);
  return (uint64_t(a) << 32) | uint32_t(b);
}
float3 point(Vec3 p) { return make_float3(p.x, p.y, p.z); }
__global__ void updateTargets(Vertex *v, int n, ReferenceSurfaceGpu ref) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    v[i].target = sizingAt(ref, v[i].p);
}
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
      out.push_back({{int(m.i0[f]), int(m.i1[f]), int(m.i2[f])}});
    return out;
  }
  static std::vector<Edge> eds(const SemanticMesh &m) {
    std::vector<Edge> out;out.reserve(m.edges.size());
    for (auto e : m.edges)
      out.push_back({int(e.v0), int(e.v1), e.face0, e.face1,
                     bool(e.flags & (EdgeSharp | EdgeMeshBoundary))});
    return out;
  }
  static std::vector<int> offs(const SemanticMesh &m) {
    std::vector<int> o(m.vertexCount() + 1);
    for (auto e : m.edges) {
      ++o[e.v0 + 1];
      ++o[e.v1 + 1];
    }
    for (size_t i = 1; i < o.size(); ++i)
      o[i] += o[i - 1];
    return o;
  }
  static std::vector<int> adjacency(const SemanticMesh &m) {
    auto o = offs(m);
    std::vector<int> a(o.back());
    for (int i = 0; i < int(m.edges.size()); ++i) {
      a[o[m.edges[i].v0]++] = i;
      a[o[m.edges[i].v1]++] = i;
    }
    return a;
  }
  static std::vector<int> fe(const SemanticMesh &m) {
    size_t capacity=8;while(capacity<2*m.edges.size())capacity*=2;
    std::vector<int> slots(capacity,-1);
    auto hash=[&](uint64_t x){x^=x>>33;x*=0xff51afd7ed558ccdULL;x^=x>>33;return size_t(x)&(capacity-1);};
    for(int i=0;i<int(m.edges.size());++i) {
      auto slot=hash(key(m.edges[i].v0,m.edges[i].v1));
      while(slots[slot]>=0)slot=(slot+1)&(capacity-1);
      slots[slot]=i;
    }
    std::vector<int> out;out.reserve(3*m.faceCount());
    for(int f=0;f<m.faceCount();++f) {
      auto t=m.face(f);
      for(int k=0;k<3;++k) {
        const auto edgeKey=key(t[k],t[(k+1)%3]);auto slot=hash(edgeKey);
        while(slots[slot]>=0 && key(m.edges[slots[slot]].v0,m.edges[slots[slot]].v1)!=edgeKey)
          slot=(slot+1)&(capacity-1);
        if(slots[slot]<0)throw std::runtime_error("missing face edge");
        out.push_back(slots[slot]);
      }
    }
    return out;
  }
  explicit DeviceMesh(const SemanticMesh &m, ReferenceSurfaceGpu ref)
      : vertices(verts(m)), faces(tris(m)), edges(eds(m)), offsets(offs(m)),
        ids(adjacency(m)), faceEdges(fe(m)) {
    updateTargets<<<(m.vertexCount() + 127) / 128, 128>>>(vertices.p,
                                                          m.vertexCount(), ref);
  }
  MeshView view() {
    return {vertices.p,  faces.p,         edges.p,      offsets.p,   ids.p,
            faceEdges.p, int(vertices.n), int(faces.n), int(edges.n)};
  }
};
__device__ int other(Edge e, int v) { return e.a == v ? e.b : e.a; }
__device__ int third(Triangle t, int a, int b) {
  for (int k = 0; k < 3; ++k)
    if (t.v[k] != a && t.v[k] != b)
      return t.v[k];
  return -1;
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
__device__ bool changedFaceSafe(MeshView m, Triangle t, int a, int b,
                                float3 dest, float high,
                                ReferenceSurfaceGpu ref, bool relaxed) {
  float3 before[3], after[3];
  bool hasA = false, hasB = false;
  for (int k = 0; k < 3; ++k) {
    int v = t.v[k];
    before[k] = m.v[v].p;
    after[k] = (v == a || v == b) ? dest : before[k];
    hasA |= v == a;
    hasB |= v == b;
  }
  if (a != b && hasA && hasB)
    return true;
  const auto n0 = normal3(before[0], before[1], before[2]),
             n1 = normal3(after[0], after[1], after[2]);
  if (qualityVcg(after[0], after[1], after[2]) <=
      fmaxf(1.e-8f, .5f * qualityVcg(before[0], before[1], before[2])))
    return false;
  if (dot3(n0, n1) < .7f * sqrtf(dot3(n0, n0) * dot3(n1, n1)))
    return false;
  if (!relaxed)
    for (int k = 0; k < 3; ++k)
      if (t.v[k] != a && t.v[k] != b &&
          dist2(dest, after[k]) >
              high * high *
                  (ref.sizingCount
                       ? powf(fminf(sizingAt(ref,
                                             mul3(add3(dest, after[k]), .5f)),
                                    .5f * (sizingAt(ref, dest) +
                                           m.v[t.v[k]].target)) /
                                  ref.regularLength,
                              2.f)
                       : 1.f))
        return false;
  return referenceFaceSafe(ref, 0, after[0], after[1], after[2]);
}
__global__ void splitVertices(MeshView m, Vertex *out, int *marked, float high,
                              ReferenceSurfaceGpu ref) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m.nv)
    out[i] = m.v[i];
  if (i >= m.ne)
    return;
  auto e = m.e[i];
  const auto midpoint = mul3(add3(m.v[e.a].p, m.v[e.b].p), .5f);
  const float target = sizingAt(ref, midpoint);
  high *= ref.sizingCount
              ? fminf(target, .5f * (m.v[e.a].target + m.v[e.b].target)) /
                    ref.regularLength
              : 1.f;
  marked[i] = dist2(m.v[e.a].p, m.v[e.b].p) > high * high;
  out[m.nv + i] = {midpoint, e.feature ? 2 : 1, target};
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
  for (int q = 0; q < 4; ++q)
    for (int k = 0; k < 3; ++k)
      local[q].v[k] = q < counts[mask] ? vv[tab[mask][q][k]] : -1;
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
__global__ void collapseCandidates(MeshView m, Candidate *out, float low,
                                   float high, ReferenceSurfaceGpu ref,
                                   bool relaxed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= m.ne)
    return;
  out[id].keep = -1;
  auto e = m.e[id];
  if (e.f0 < 0)
    return;
  int a = e.a, b = e.b;
  if (ref.sizingCount) {
    low *= fminf(sizingAt(ref, mul3(add3(m.v[a].p, m.v[b].p), .5f)),
                 .5f * (m.v[a].target + m.v[b].target)) /
           ref.regularLength;
  }
  bool ma = movable(m, a, id), mb = movable(m, b, id);
  if (!ma && !mb)
    return;
  int da = m.offsets[a + 1] - m.offsets[a],
      db = m.offsets[b + 1] - m.offsets[b];
  if (relaxed && !((da == 3 || da == 4) && m.v[a].constraint < 2) &&
      !((db == 3 || db == 4) && m.v[b].constraint < 2))
    return;
  int common = 0;
  for (int i = m.offsets[a]; i < m.offsets[a + 1]; ++i)
    if (neighbor(m, b, other(m.e[m.ids[i]], a)))
      ++common;
  if (common != (e.f1 < 0 ? 1 : 2) || (da == 3 && db == 3 && e.f1 >= 0))
    return;
  auto tri = m.f[e.f0];
  auto n = normal3(m.v[tri.v[0]].p, m.v[tri.v[1]].p, m.v[tri.v[2]].p);
  float area = .5f * sqrtf(dot3(n, n));
  if (e.f1 >= 0) {
    tri = m.f[e.f1];
    n = normal3(m.v[tri.v[0]].p, m.v[tri.v[1]].p, m.v[tri.v[2]].p);
    area = fminf(area, .5f * sqrtf(dot3(n, n)));
  }
  if (!relaxed && dist2(m.v[a].p, m.v[b].p) >= low * low &&
      area >= low * low / 100.f)
    return;
  int keep = ma && !mb ? b : a, remove = keep == a ? b : a;
  float3 dest = ma && mb ? mul3(add3(m.v[a].p, m.v[b].p), .5f) : m.v[keep].p;
  if (!referenceNear(ref, 0, dest))
    return;
  // Midpoints stay on the source edge until the cycle's reprojection, matching
  // VCGLib's collapse placement. Test the unprojected surviving triangles.
  for (int side = 0; side < 2; ++side) {
    int v = side ? a : b;
    for (int i = m.offsets[v]; i < m.offsets[v + 1]; ++i) {
      auto x = m.e[m.ids[i]];
      if (x.f0 >= 0 && ownsFace(m.f[x.f0],x,v) &&
          !changedFaceSafe(m, m.f[x.f0], a, b, dest, high, ref, relaxed))
        return;
      if (x.f1 >= 0 && ownsFace(m.f[x.f1],x,v) &&
          !changedFaceSafe(m, m.f[x.f1], a, b, dest, high, ref, relaxed))
        return;
    }
  }
  out[id] = {keep, remove, -1, -1, dest};
}
__global__ void flipCandidates(MeshView m, Candidate *out,
                               ReferenceSurfaceGpu ref) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= m.ne)
    return;
  out[id].keep = -1;
  auto e = m.e[id];
  if (e.feature || e.f0 < 0 || e.f1 < 0)
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
  if (!((after < before && newQ >= oldQ * .5f) ||
        (after == before && newQ > oldQ) ||
        (after > before && newQ > oldQ * 1.5f)))
    return;
  auto n0 = normal3(pa, pb, pc), n1 = normal3(pb, pa, pd),
       nn0 = normal3(pc, pd, pb), nn1 = normal3(pd, pc, pa);
  constexpr float cos5 = .996194698f;
  if (dot3(n0, nn0) < cos5 * sqrtf(dot3(n0, n0) * dot3(nn0, nn0)) ||
      dot3(n0, nn1) < cos5 * sqrtf(dot3(n0, n0) * dot3(nn1, nn1)) ||
      dot3(n1, nn0) < cos5 * sqrtf(dot3(n1, n1) * dot3(nn0, nn0)) ||
      dot3(n1, nn1) < cos5 * sqrtf(dot3(n1, n1) * dot3(nn1, nn1)))
    return;
  if (!referenceFaceSafe(ref, 0, pc, pd, pb) ||
      !referenceFaceSafe(ref, 0, pd, pc, pa))
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
__global__ void initMap(int *map, int n) {
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
    m.f[e.f0] = {{x.c, x.d, x.remove}};
    m.f[e.f1] = {{x.d, x.c, x.keep}};
  } else {
    map[x.remove] = x.keep;
    m.v[x.keep].p = x.p;
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
                               unsigned seed, int *moved) {
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
  for (int attempt = 0; attempt < 8; ++attempt) {
    auto trial = add3(p, mul3(sub3(avg, p), step));
    float3 dest;
    bool ok=projectReference(ref,0,trial,dest);
    const float tolerance=toleranceAt(ref,trial);
    ok=ok && dist2(dest,trial)<=tolerance*tolerance;
    for (int i = m.offsets[v]; i < m.offsets[v + 1] && ok; ++i) {
      auto e = m.e[m.ids[i]];
      if (e.f0 >= 0 && ownsFace(m.f[e.f0],e,v))
        ok = changedFaceSafe(m, m.f[e.f0], v, v, dest, 0, ref, true);
      if (ok && e.f1 >= 0 && ownsFace(m.f[e.f1],e,v))
        ok = changedFaceSafe(m, m.f[e.f1], v, v, dest, 0, ref, true);
    }
    if (ok) {
      if (dist2(dest, p) > 1.e-16f) {
        next[v].p = dest;
        atomicAdd(moved, 1);
      }
      return;
    }
    step *= .5f;
  }
}
__global__ void proposeProjection(MeshView m,Vertex *proposal,ReferenceSurfaceGpu ref) {
  int v=blockIdx.x*blockDim.x+threadIdx.x;
  if(v>=m.nv)return;
  proposal[v]=m.v[v];
  if(m.v[v].constraint>=4)return;
  float3 q;
  if(projectReference(ref,0,m.v[v].p,q))proposal[v].p=q;
}
__global__ void checkProjectionFaces(MeshView m,const Vertex *proposal,ReferenceSurfaceGpu ref,int *reject,int *badFaces) {
  int f=blockIdx.x*blockDim.x+threadIdx.x;
  if(f>=m.nf)return;
  auto t=m.f[f];
  const auto a=m.v[t.v[0]].p,b=m.v[t.v[1]].p,c=m.v[t.v[2]].p;
  const auto x=proposal[t.v[0]].p,y=proposal[t.v[1]].p,z=proposal[t.v[2]].p;
  const auto oldN=normal3(a,b,c),newN=normal3(x,y,z);
  const float area2=dot3(newN,newN),oldArea2=dot3(oldN,oldN);
  const float q=qualityVcg(x,y,z);
  const bool qualityBad=!(area2>0) || !isfinite(area2) || !(q>0) || !isfinite(q) ||
      q<.5f*qualityVcg(a,b,c) || dot3(oldN,newN)<.7f*sqrtf(oldArea2*area2);
  const bool moved=dist2(a,x)>0 || dist2(b,y)>0 || dist2(c,z)>0;
  const bool errorBad=!qualityBad && moved && !referenceFaceSafe(ref,0,x,y,z);
  if(qualityBad || errorBad) {
    atomicAdd(badFaces,1);
    if(errorBad)atomicAdd(badFaces+1,1);
    for(int k=0;k<3;++k)atomicExch(reject+t.v[k],1);
  }
}
__global__ void revertProjection(MeshView m,Vertex *proposal,const int *reject,int *rejected) {
  int v=blockIdx.x*blockDim.x+threadIdx.x;
  if(v>=m.nv || !reject[v])return;
  if(dist2(proposal[v].p,m.v[v].p)>0)atomicAdd(rejected,1);
  proposal[v]=m.v[v];
}
// Validate the complete proposed face, not each vertex in isolation. Partial
// rollback can invalidate a neighboring face, so recheck until stable. A bounded
// failure rolls back the entire projection; topology checks are never bypassed.
int safeProject(MeshView m,ReferenceSurfaceGpu ref,int *errorRejects=nullptr) {
  Buffer<Vertex> proposal(m.nv);
  Buffer<int> reject(m.nv),bad(2),rejected(1);
  checked(cudaMemset(rejected.p,0,sizeof(int)));
  proposeProjection<<<(m.nv+127)/128,128>>>(m,proposal.p,ref);
  for(int pass=0;pass<16;++pass) {
    checked(cudaMemset(reject.p,0,m.nv*sizeof(int)));
    checked(cudaMemset(bad.p,0,2*sizeof(int)));
    checkProjectionFaces<<<(m.nf+127)/128,128>>>(m,proposal.p,ref,reject.p,bad.p);
    sync();
    const auto counts=bad.read();
    if(errorRejects)*errorRejects+=counts[1];
    if(counts[0]==0) {
      checked(cudaMemcpy(m.v,proposal.p,m.nv*sizeof(Vertex),cudaMemcpyDeviceToDevice));
      return rejected.read()[0];
    }
    revertProjection<<<(m.nv+127)/128,128>>>(m,proposal.p,reject.p,rejected.p);
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
        projectReference(ref, 0, p, q) ? sqrtf(dist2(p, q)) : FLT_MAX;
    atomicMax(maximum, __float_as_uint(error));
    atomicMax(maximum+1, __float_as_uint(error / fmaxf(toleranceAt(ref,p),1.e-20f)));
  }
}

void rebuild(SemanticMesh &mesh, const std::vector<Vertex> &vertices,
             const std::vector<Triangle> &faces,
             const std::unordered_map<uint64_t, uint32_t> &features, float h) {
  std::vector<int> map(vertices.size(), -1);
  for (auto t : faces)
    if (t.v[0] >= 0 && t.v[0] != t.v[1] && t.v[1] != t.v[2] && t.v[2] != t.v[0])
      for (int v : t.v)
        map[v] = 0;
  SemanticMesh out;
  out.patches = mesh.patches;
  for (int v = 0; v < int(vertices.size()); ++v)
    if (map[v] == 0) {
      auto p = vertices[v].p;
      map[v] = out.addVertex({p.x, p.y, p.z}, 0,
                             VertexConstraint(vertices[v].constraint));
      out.targetLength[map[v]] = vertices[v].target;
    }
  for (auto t : faces)
    if (t.v[0] >= 0 && t.v[0] != t.v[1] && t.v[1] != t.v[2] && t.v[2] != t.v[0])
      out.addFace(map[t.v[0]], map[t.v[1]], map[t.v[2]], 0, PatchType::Unknown);
  for (auto kv : features) {
    int a = int(kv.first >> 32), b = int(uint32_t(kv.first));
    if (a < int(map.size()) && b < int(map.size()) && map[a] >= 0 &&
        map[b] >= 0 && map[a] != map[b])
      out.featureEdges[key(map[a], map[b])] = kv.second;
  }
  out.rebuildTopology();
  mesh = std::move(out);
}
__global__ void rejectSplitFaces(MeshView m,const Vertex *v,const Triangle *faces,
                                  int *marked,ReferenceSurfaceGpu ref,int *bad) {
  int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=m.nf)return;
  if(faces[4*f+1].v[0]<0)return;
  for(int q=0;q<4;++q) {
    auto t=faces[4*f+q];if(t.v[0]<0)continue;
    auto a=v[t.v[0]].p,b=v[t.v[1]].p,c=v[t.v[2]].p;
    if(!(qualityVcg(a,b,c)>0) || !referenceFaceSafe(ref,0,a,b,c)) {
      atomicAdd(bad,1);
      for(int k=0;k<3;++k)atomicExch(marked+m.faceEdges[3*f+k],0);
      return;
    }
  }
}
int split(SemanticMesh &mesh, float h, float high, ReferenceSurfaceGpu ref,RemeshReport &report) {
  DeviceMesh d(mesh, ref);
  auto m = d.view();
  Buffer<Vertex> v(m.nv + m.ne);
  Buffer<Triangle> f(4 * m.nf);
  Buffer<int> marked(m.ne);
  splitVertices<<<(std::max(m.nv, m.ne) + 127) / 128, 128>>>(m, v.p, marked.p,
                                                             high, ref);
  Buffer<int> bad(1);
  // Read/write split masks are separate: neighboring threads may cancel the
  // same edge, but never read a mask while another thread writes it.
  Buffer<int> checkedMask(m.ne);
  bool stable=false;
  for(int pass=0;pass<16;++pass) {
    splitFaces<<<(m.nf+127)/128,128>>>(m,v.p,marked.p,f.p);
    checked(cudaMemcpy(checkedMask.p,marked.p,m.ne*sizeof(int),cudaMemcpyDeviceToDevice));
    checked(cudaMemset(bad.p,0,sizeof(int)));
    rejectSplitFaces<<<(m.nf+127)/128,128>>>(m,v.p,f.p,checkedMask.p,ref,bad.p);
    sync();const int rejected=bad.read()[0];report.rejectError+=rejected;
    if(!rejected){stable=true;break;}
    checked(cudaMemcpy(marked.p,checkedMask.p,m.ne*sizeof(int),cudaMemcpyDeviceToDevice));
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
int topology(SemanticMesh &mesh, float h, float low, float high,
             ReferenceSurfaceGpu ref, bool flip, bool relaxed, unsigned seed,
             RemeshReport &report) {
  DeviceMesh d(mesh, ref);
  auto m = d.view();
  Buffer<Candidate> candidates(m.ne);
  Buffer<unsigned long long> claims(m.nv);
  Buffer<int> chosen(m.ne), mapping(m.nv);
  if (flip)
    flipCandidates<<<(m.ne + 127) / 128, 128>>>(m, candidates.p, ref);
  else
    collapseCandidates<<<(m.ne + 127) / 128, 128>>>(m, candidates.p, low, high,
                                                    ref, relaxed);
  initClaims<<<(m.nv + 127) / 128, 128>>>(claims.p, m.nv);
  claimCandidates<<<(m.ne + 127) / 128, 128>>>(m, candidates.p, claims.p, flip,
                                               seed);
  resolveCandidates<<<(m.ne + 127) / 128, 128>>>(m, candidates.p, claims.p,
                                                 chosen.p, flip, seed);
  Buffer<int> counts(2);
  checked(cudaMemset(counts.p,0,2*sizeof(int)));
  countCandidates<<<(m.ne+127)/128,128>>>(candidates.p,chosen.p,m.ne,counts.p);
  sync();
  const auto totals=counts.read();
  const int count=totals[0];
  if(flip)report.flipCandidates+=totals[1];else report.collapseCandidates+=totals[1];
  if (!count)
    return 0;
  if(!flip) initMap<<<(m.nv + 127) / 128, 128>>>(mapping.p, m.nv);
  applyCandidates<<<(m.ne + 127) / 128, 128>>>(m, candidates.p, chosen.p,
                                               mapping.p, flip);
  if (!flip)
    remapFaces<<<(m.nf + 127) / 128, 128>>>(m, mapping.p);
  sync();
  if(flip) {
    // Flips neither delete vertices nor change coordinates/feature identities.
    const auto faces=d.faces.read();
    for(int f=0;f<m.nf;++f) {mesh.i0[f]=faces[f].v[0];mesh.i1[f]=faces[f].v[1];mesh.i2[f]=faces[f].v[2];}
    mesh.rebuildTopology();
    return count;
  }
  auto features = mesh.featureEdges;
  if (!flip) {
    auto map = mapping.read();
    features.clear();
    for (auto kv : mesh.featureEdges) {
      int a = map[kv.first >> 32], b = map[uint32_t(kv.first)];
      if (a != b)
        features[key(a, b)] = kv.second;
    }
  }
  rebuild(mesh, d.vertices.read(), d.faces.read(), features, h);
  return count;
}
int smooth(SemanticMesh &mesh, ReferenceSurfaceGpu ref, float lambda,
           unsigned seed, RemeshReport &report) {
  DeviceMesh d(mesh, ref);
  auto m = d.view();
  Buffer<Vertex> next(m.nv);
  Buffer<int> moved(1);
  checked(cudaMemset(moved.p, 0, sizeof(int)));
  for (int pass = 0; pass < 12; ++pass) {
    smoothVertices<<<(m.nv + 127) / 128, 128>>>(m, next.p, ref, lambda,
                                                seed + pass, moved.p);
    std::swap(m.v, next.p);
  }
  report.rejectQuality += safeProject(m,ref,&report.rejectError);
  sync();
  std::vector<Vertex> verts(m.nv);
  checked(cudaMemcpy(verts.data(), m.v, m.nv * sizeof(Vertex),
                     cudaMemcpyDeviceToHost));
  for (int i = 0; i < m.nv; ++i) {
    auto p = verts[i].p;
    mesh.setPosition(i, {p.x, p.y, p.z});
  }
  return moved.read()[0];
}
void buildReferenceBvh(const std::vector<ReferenceTriangleGpu> &triangles,float boxPad,
                       std::vector<ReferenceBvhNode> &tree,std::vector<int> &sourceIds) {
    tree.clear();sourceIds.resize(triangles.size());
    std::iota(sourceIds.begin(),sourceIds.end(),0);
    auto coord=[](float3 p,int axis){return axis==0?p.x:(axis==1?p.y:p.z);};
    std::function<void(int,int)> buildTree=[&](int begin,int end) {
      const int id=int(tree.size());tree.emplace_back();
      float3 lo=make_float3(FLT_MAX,FLT_MAX,FLT_MAX),hi=make_float3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
      for(int i=begin;i<end;++i) {const auto t=triangles[sourceIds[i]];for(auto p:{t.a,t.b,t.c}) {
        lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);
        hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);
      }}
      tree[id].lower=make_float3(lo.x-boxPad,lo.y-boxPad,lo.z-boxPad);
      tree[id].upper=make_float3(hi.x+boxPad,hi.y+boxPad,hi.z+boxPad);
      if(end-begin<=4) {tree[id].first=begin;tree[id].count=end-begin;}
      else {
        int axis=0;if(hi.y-lo.y>hi.x-lo.x)axis=1;if(hi.z-lo.z>coord(hi,axis)-coord(lo,axis))axis=2;
        std::stable_sort(sourceIds.begin()+begin,sourceIds.begin()+end,[&](int a,int b) {
          const auto ta=triangles[a],tb=triangles[b];
          return coord(ta.a,axis)+coord(ta.b,axis)+coord(ta.c,axis)<coord(tb.a,axis)+coord(tb.b,axis)+coord(tb.c,axis);
        });
        int mid=(begin+end)/2;buildTree(begin,mid);buildTree(mid,end);
      }
      tree[id].escape=int(tree.size());
    };
    if(!triangles.empty())buildTree(0,int(triangles.size()));
}
} // namespace raw

bool remeshRawCuda(SemanticMesh &mesh, const RemeshConfig &cfg,
                   RemeshReport &report) {
  using namespace raw;
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  DeviceArena arena;
  ArenaScope arenaScope(arena);
  report = {};
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
        !std::isfinite(cfg.maxGeometryError) || cfg.maxIterations < 1)
      throw std::runtime_error(
          "invalid raw CUDA sizing, error budget or iteration count");
    const float cosAngle =
        std::cos(cfg.featureAngleDegrees * .01745329251994329577f);
    for (auto e : mesh.edges) {
      bool feature = e.face1 < 0;
      if (e.face1 >= 0) {
        auto a = mesh.face(e.face0), b = mesh.face(e.face1);
        feature = dot(triangleNormal(mesh.position(a[0]), mesh.position(a[1]),
                                     mesh.position(a[2])),
                      triangleNormal(mesh.position(b[0]), mesh.position(b[1]),
                                     mesh.position(b[2]))) <= cosAngle;
      }
      if (feature)
        mesh.featureEdges[key(e.v0, e.v1)] = 0;
    }
    mesh.rebuildTopology();
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
    std::vector<ReferenceTriangleGpu> triangles;
    for (int f = 0; f < mesh.faceCount(); ++f)
      triangles.push_back({point(mesh.facePoint(f, 0)),
                           point(mesh.facePoint(f, 1)),
                           point(mesh.facePoint(f, 2)), 0});
    std::vector<ReferenceBvhNode> tree;
    std::vector<int> sourceIds;
    buildReferenceBvh(triangles,mesh.bboxDiagonal()*1.e-6f,tree,sourceIds);
    Buffer<ReferenceBvhNode> bvh(tree);
    Buffer<int> triangleIds(sourceIds);
    Buffer<ReferenceTriangleGpu> reference(triangles);
    std::vector<SizingSegmentGpu> seeds;
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
    std::vector<Vec3> normals(mesh.faceCount());
    for(int f=0;f<mesh.faceCount();++f) {
      region[f]=f;
      normals[f]=triangleNormal(mesh.facePoint(f,0),mesh.facePoint(f,1),mesh.facePoint(f,2));
    }
    auto root=[&](int f) {while(region[f]!=f) {region[f]=region[region[f]];f=region[f];}return f;};
    const float coplanarCos=std::cos(.01745329252f);
    if(refine) for(auto e:mesh.edges) if(e.face0>=0 && e.face1>=0 &&
        dot(normals[e.face0],normals[e.face1])>=coplanarCos)
      region[root(e.face1)]=root(e.face0);
    std::vector<std::unordered_set<int>> smoothNeighbors(mesh.faceCount());
    if(refine) for(auto e:mesh.edges) if(e.face0>=0 && e.face1>=0 &&
        !(e.flags&(EdgeSharp|EdgeMeshBoundary))) {
      int a=root(e.face0), b=root(e.face1);
      if(a!=b) {smoothNeighbors[a].insert(b);smoothNeighbors[b].insert(a);}
    }
    if (refine)
      for (auto e : mesh.edges) {
        const auto a = mesh.position(e.v0), b = mesh.position(e.v1);
        float local = h;
        // A crease is a geometric constraint, not a curvature sample. In
        // particular, plane/plane intersections must retain the regular size.
        if (!(e.flags & (EdgeSharp | EdgeMeshBoundary)) && e.face0 >= 0 && e.face1 >= 0) {
          const auto n0 = triangleNormal(mesh.facePoint(e.face0, 0),
                                         mesh.facePoint(e.face0, 1),
                                         mesh.facePoint(e.face0, 2));
          const auto n1 = triangleNormal(mesh.facePoint(e.face1, 0),
                                         mesh.facePoint(e.face1, 1),
                                         mesh.facePoint(e.face1, 2));
          const float angle = std::acos(clampf(dot(n0, n1), -1.f, 1.f));
          if (angle > .0174533f &&
              (smoothNeighbors[root(e.face0)].size()>=2 || smoothNeighbors[root(e.face1)].size()>=2)) {
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
          seeds.push_back({point(a), point(b), local});
      }
    std::vector<ReferenceTriangleGpu> seedBounds;
    for(auto seed:seeds)seedBounds.push_back({seed.a,seed.b,seed.a,0});
    std::vector<ReferenceBvhNode> sizeTree;std::vector<int> sizeIds;
    buildReferenceBvh(seedBounds,mesh.bboxDiagonal()*1.e-6f,sizeTree,sizeIds);
    Buffer<ReferenceBvhNode> sizingBvh(sizeTree);Buffer<int> sizingIds(sizeIds);
    Buffer<SizingSegmentGpu> sizing(seeds);
    ReferenceSurfaceGpu ref{reference.p, int(reference.n), cfg.maxGeometryError,
                            sizing.p,    int(sizing.n),    h,
                            band};
    ref.sizingNodes=sizingBvh.p;ref.sizingNodeCount=int(sizeTree.size());ref.sizingIds=sizingIds.p;
    ref.nodes=bvh.p;ref.nodeCount=int(tree.size());ref.triangleIds=triangleIds.p;
    std::cout<<"reference_query=bvh nodes="<<ref.nodeCount<<std::endl;
    std::cout << "raw_sizing="
              << (refine ? "curvature_only" : "uniform")
              << " seeds=" << seeds.size()
              << " feature_length=" << (refine ? fine : h)
              << " transition_band=" << band << std::endl;
    report.secondsSetup =
        std::chrono::duration<double>(Clock::now() - start).count();
    std::cout << "raw_gpu_policy=triangle_templates reference_faces="
              << triangles.size()
              << " feature_edges=" << mesh.featureEdges.size()
              << " host_adjacency=true" << std::endl;
    int currentCycle=-1;
    const char *stage="setup";
    auto validate = [&]() {
      if (!mesh.validate(&error))
        throw std::runtime_error("raw CUDA topology: cycle="+std::to_string(currentCycle)+" stage="+stage+": " + error);
    };
    for (int cycle = 0; cycle < cfg.maxIterations; ++cycle) {
      currentCycle=cycle;
      int ns = 0, nc = 0, nf = 0, nm = 0;
      auto mark = Clock::now();
      stage="split";
      if (cfg.enableSplit) {
        ns = split(mesh, h, cfg.splitRatio * h, ref,report);
        report.splits += ns;
        report.splitCandidates += ns;
        validate();
      }
      report.secondsSplit +=
          std::chrono::duration<double>(Clock::now() - mark).count();
      mark = Clock::now();
      stage="collapse";
      if (cfg.enableCollapse) {
        for (int pass = 0; pass < 8; ++pass) {
          int n =
              topology(mesh, h, cfg.collapseRatio * h, cfg.splitRatio * h, ref,
                       false, false, unsigned(cycle * 31 + pass + 1), report);
          nc += n;
          if (!n)
            break;
        }
        int n = topology(mesh, h, cfg.collapseRatio * h, cfg.splitRatio * h,
                         ref, false, true, unsigned(cycle + 717), report);
        nc += n;
        report.collapses += nc;
        validate();
      }
      report.secondsCollapse +=
          std::chrono::duration<double>(Clock::now() - mark).count();
      mark = Clock::now();
      stage="flip";
      if (cfg.enableFlip) {
        for (int pass = 0; pass < 8; ++pass) {
          int n =
              topology(mesh, h, cfg.collapseRatio * h, cfg.splitRatio * h, ref,
                       true, false, unsigned(cycle * 37 + pass + 1), report);
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
      if (cfg.enableSmooth) {
        nm = smooth(mesh, ref, cfg.smoothLambda, unsigned(cycle * 12 + 1),report);
        report.smoothMoves += nm;
      } else {
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
      std::cout << "raw_gpu_cycle=" << cycle << " faces=" << mesh.faceCount()
                << " split=" << ns << " collapse=" << nc << " flip=" << nf
                << " smooth=" << nm << std::endl;
    }
    validate();
    float localErrorRatio=0;
    {
      DeviceMesh d(mesh, ref);
      auto m = d.view();
      Buffer<unsigned int> maximum(2);
      checked(cudaMemset(maximum.p, 0, 2*sizeof(unsigned int)));
      auditGeometry<<<(m.nf + 127) / 128, 128>>>(m, ref, maximum.p);
      sync();
      auto audit = maximum.read();
      auto bits = audit[0];
      std::memcpy(&localErrorRatio,&audit[1],sizeof(float));
      if(refine) std::cout<<"raw_local_error_ratio="<<localErrorRatio<<std::endl;
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
    return report.constraintsHeld;
  } catch (const std::exception &e) {
    std::cerr << "[raw CUDA] " << e.what() << '\n';
    report.seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    return false;
  }
}
} // namespace cad_adaptive
