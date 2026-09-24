#include "cad_adaptive/RemeshField.h"
#include "cad_adaptive/BoundarySizingField.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

namespace cad_adaptive {

SizingLimits RemeshField::limits(const SemanticMesh &mesh, const RemeshConfig &config) {
  SizingLimits lim;
  lim.epsilon = config.maxGeometryError > 0 ? config.maxGeometryError : 1e-6f;
  lim.hMin = config.hMin > 0 ? config.hMin : lim.epsilon;
  const float derived = mesh.bboxDiagonal() * 0.05f;
  if (config.hMax > 0)
    lim.hMax = config.hMax;
  else if (config.constantLength > 0)
    lim.hMax = config.constantLength;
  else
    lim.hMax = derived > lim.hMin ? derived : lim.hMin * 8;
  if (lim.hMax < lim.hMin) lim.hMax = lim.hMin;
  return lim;
}

float RemeshField::patchLength(const PatchRecord &patch, const SizingLimits &lim, float hConst) {
  if (patch.type == PatchType::Plane) return lim.hMax;
  if (patch.type == PatchType::Cylinder && patch.radius > 0)
    return std::sqrt(8.0f * patch.radius * lim.epsilon);
  if (hConst > 0) return hConst;
  return lim.hMax;
}

float RemeshField::curvatureLength(float kmax, float epsilon, float hMax) {
  if (!(kmax > 0)) return hMax;
  return std::sqrt(8.0f * epsilon / (kmax + 1e-12f));
}

float RemeshField::featureLength(float distance, float band, float hFeatureEdge, float hRegular) {
  if (!(band > 0)) return hRegular;
  const float t = smoothstep(clampf(distance / band, 0, 1));
  return lerp(hFeatureEdge, hRegular, t);
}

float RemeshField::combine(float hCurvature, float hFeature, float hPatch, float hError,
                           const SizingLimits &lim) {
  const float h = std::min(std::min(hCurvature, hFeature), std::min(hPatch, hError));
  return clampf(h, lim.hMin, lim.hMax);
}

void RemeshField::computeFeatureDistance(SemanticMesh &mesh) {
  const int n = mesh.vertexCount();
  mesh.featureDistance.assign(n, std::numeric_limits<float>::max());
  struct Segment { Vec3 a, b, lo, hi; };
  std::vector<Segment> segments;
  segments.reserve(mesh.edges.size());
  for (const auto &e : mesh.edges) {
    if ((e.flags & (EdgePatchBoundary | EdgeMeshBoundary | EdgeSharp | EdgeProtected)) == 0)
      continue;
    const Vec3 a = mesh.position(int(e.v0));
    const Vec3 b = mesh.position(int(e.v1));
    segments.push_back({a, b,
        {std::min(a.x,b.x),std::min(a.y,b.y),std::min(a.z,b.z)},
        {std::max(a.x,b.x),std::max(a.y,b.y),std::max(a.z,b.z)}});
  }
  if (segments.empty()) {
    std::fill(mesh.featureDistance.begin(),mesh.featureDistance.end(),1e20f);
    return;
  }
  struct Node { Vec3 lo, hi; int left=-1, right=-1; size_t begin=0, end=0; };
  std::vector<size_t> ids(segments.size());
  std::iota(ids.begin(),ids.end(),size_t(0));
  std::vector<Node> nodes;
  nodes.reserve(segments.size()*2);
  const auto coord=[](Vec3 p,int axis) {return axis==0?p.x:(axis==1?p.y:p.z);};
  std::function<int(size_t,size_t)> build=[&](size_t begin,size_t end) {
    const int index=int(nodes.size());
    nodes.emplace_back();
    Vec3 lo=segments[ids[begin]].lo,hi=segments[ids[begin]].hi;
    for(size_t i=begin+1;i<end;++i) {
      const auto &s=segments[ids[i]];
      lo.x=std::min(lo.x,s.lo.x);lo.y=std::min(lo.y,s.lo.y);lo.z=std::min(lo.z,s.lo.z);
      hi.x=std::max(hi.x,s.hi.x);hi.y=std::max(hi.y,s.hi.y);hi.z=std::max(hi.z,s.hi.z);
    }
    nodes[index].lo=lo;nodes[index].hi=hi;
    nodes[index].begin=begin;nodes[index].end=end;
    if(end-begin<=8) return index;
    const Vec3 span=hi-lo;
    const int axis=span.y>span.x ? (span.z>span.y?2:1) : (span.z>span.x?2:0);
    const size_t middle=begin+(end-begin)/2;
    std::nth_element(ids.begin()+begin,ids.begin()+middle,ids.begin()+end,
        [&](size_t a,size_t b) {
          const float ca=coord(segments[a].lo,axis)+coord(segments[a].hi,axis);
          const float cb=coord(segments[b].lo,axis)+coord(segments[b].hi,axis);
          return ca<cb || (ca==cb && a<b);
        });
    const int left=build(begin,middle),right=build(middle,end);
    nodes[index].left=left;nodes[index].right=right;
    return index;
  };
  build(0,ids.size());
  const auto boxDistance2=[](Vec3 p,const Node &node) {
    const float dx=std::max(std::max(node.lo.x-p.x,0.0f),p.x-node.hi.x);
    const float dy=std::max(std::max(node.lo.y-p.y,0.0f),p.y-node.hi.y);
    const float dz=std::max(std::max(node.lo.z-p.z,0.0f),p.z-node.hi.z);
    return dx*dx+dy*dy+dz*dz;
  };
  std::vector<int> stack;stack.reserve(64);
  for(int v=0;v<n;++v) {
    const Vec3 p=mesh.position(v);
    float best=std::numeric_limits<float>::max();
    stack.clear();stack.push_back(0);
    while(!stack.empty()) {
      const int index=stack.back();stack.pop_back();
      const Node &node=nodes[index];
      if(boxDistance2(p,node)>=best) continue;
      if(node.left<0) {
        for(size_t i=node.begin;i<node.end;++i) {
          const auto &s=segments[ids[i]];
          const Vec3 q=closestOnSegment(p,s.a,s.b);
          best=std::min(best,length2(p-q));
        }
      } else {
        const float dl=boxDistance2(p,nodes[node.left]);
        const float dr=boxDistance2(p,nodes[node.right]);
        if(dl<dr) {
          if(dr<best)stack.push_back(node.right);
          if(dl<best)stack.push_back(node.left);
        } else {
          if(dl<best)stack.push_back(node.left);
          if(dr<best)stack.push_back(node.right);
        }
      }
    }
    mesh.featureDistance[v]=std::sqrt(best);
  }
}

void RemeshField::compute(SemanticMesh &mesh, const RemeshConfig &config,
                          const GeometryProjector &projector) {
  if (mesh.LocalSizing) {
    mesh.LocalSizing->apply(mesh);
    return;
  }
  const SizingLimits lim = limits(mesh, config);
  const int n = mesh.vertexCount();
  mesh.targetLength.resize(n);
  mesh.curvature.resize(n);
  if (int(mesh.featureDistance.size()) != n) computeFeatureDistance(mesh);

  for (int v = 0; v < n; ++v) {
    if (!config.adaptive && config.constantLength > 0) {
      mesh.curvature[v] = projector.analyticKmax(mesh.vertexPatchId[v]);
      mesh.targetLength[v] = config.constantLength;
      continue;
    }
    const uint32_t patchId = mesh.vertexPatchId[v];
    const PatchRecord *rec = projector.patch(patchId);
    PatchRecord fallback;
    if (!rec) {
      fallback.type = PatchType::Unknown;
      rec = &fallback;
    }
    const float kmax = projector.analyticKmax(patchId);
    mesh.curvature[v] = kmax;
    const float hConst = config.constantLength;
    const float hPatch = patchLength(*rec, lim, hConst);
    const float hCurv = curvatureLength(kmax, lim.epsilon, lim.hMax);
    const float hRegular = clampf(std::min(hPatch, hCurv), lim.hMin, lim.hMax);
    const float band = config.featureBand > 0 ? config.featureBand : 4.0f * hRegular;
    // Geometry tolerance is not a mesh-size target. Using 2*epsilon here
    // drove raw STL crease neighborhoods orders of magnitude below the global
    // target (e.g. h=1.8 -> hFeature=0.02), causing split explosions and
    // needle triangles. Unless explicitly overridden, keep the feature itself
    // at the regular local size and let featureBand control only the transition.
    const float hFeatEdge =
        config.featureEdgeLength > 0 ? config.featureEdgeLength : hRegular;
    const float hFeat = featureLength(mesh.featureDistance[v], band, hFeatEdge, hRegular);
    mesh.targetLength[v] = combine(hCurv, hFeat, hPatch, lim.hMax, lim);
  }
}

} // namespace cad_adaptive
