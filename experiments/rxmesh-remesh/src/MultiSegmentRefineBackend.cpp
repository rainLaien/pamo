#include "cad_adaptive/MultiSegmentRefineBackend.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace cad_adaptive {

bool planMultiSegmentRefine(const SemanticMesh& input,const RemeshConfig& config,
                            MultiSegmentRefineStats& stats,std::string* error){
  stats={};
  stats.InputFaces=input.faceCount();
  stats.InputEdges=int(input.edges.size());
  if(input.targetLength.size()!=size_t(input.vertexCount())){
    if(error)*error="targetLength size mismatch";
    return false;
  }
  long long totalSegments=0;
  for(const auto&e:input.edges){
    if(e.v0>=uint32_t(input.vertexCount())||e.v1>=uint32_t(input.vertexCount())){
      if(error)*error="edge vertex index out of range";
      return false;
    }
    const float h=0.5f*(input.targetLength[e.v0]+input.targetLength[e.v1]);
    if(!(h>0.0f)||!std::isfinite(h)){
      if(error)*error="invalid edge target length";
      return false;
    }
    const float len=distance(input.position(int(e.v0)),input.position(int(e.v1)));
    if(!std::isfinite(len)){
      if(error)*error="invalid edge length";
      return false;
    }
    // Keep every generated segment at or below the split threshold while
    // avoiding needless over-refinement toward h/2.
    const float maxSegment=config.splitRatio*h;
    const int n=std::max(1,int(std::ceil(len/maxSegment)));
    if(n>1) {
      ++stats.SubdividedEdges;
      stats.PlannedInsertedEdgeVertices += n - 1;
    }
    stats.MaxSegments=std::max(stats.MaxSegments,n);
    stats.MaxPlannedSegmentLength=std::max(stats.MaxPlannedSegmentLength,
                                           n>0?len/float(n):len);
    if(len/float(n)>maxSegment*(1.0f+1e-5f)) ++stats.InvalidPlans;
    totalSegments+=n;
  }
  stats.PlannedSegments=int(std::min<long long>(totalSegments,2147483647LL));
  stats.MeanSegments=stats.InputEdges>0?float(double(totalSegments)/stats.InputEdges):1.0f;
  return true;
}

bool materializeMultiSegmentEdges(const SemanticMesh& input,const RemeshConfig& config,
                                  const GeometryProjector& referenceProjector,
                                  SemanticMesh& output,MultiSegmentRefineStats& stats,
                                  std::vector<EdgeSubdivision>* subdivisions,
                                  std::string* error){
  if(!planMultiSegmentRefine(input,config,stats,error)) return false;
  output=input;
  if(subdivisions) {
    subdivisions->clear();
    subdivisions->reserve(input.edges.size());
  }
  for(const auto&e:input.edges){
    EdgeSubdivision chain;
    chain.V0=e.v0;
    chain.V1=e.v1;
    chain.VertexIds.push_back(int(e.v0));
    const float h=0.5f*(input.targetLength[e.v0]+input.targetLength[e.v1]);
    const float len=distance(input.position(int(e.v0)),input.position(int(e.v1)));
    const int n=std::max(1,int(std::ceil(len/(config.splitRatio*h))));
    chain.SegmentCount=n;
    if(n<=1) {
      chain.VertexIds.push_back(int(e.v1));
      if(subdivisions) subdivisions->push_back(std::move(chain));
      continue;
    }
    const Vec3 a=input.position(int(e.v0)),b=input.position(int(e.v1));
    const bool feature=(e.flags&(EdgeSharp|EdgePatchBoundary|EdgeMeshBoundary))!=0;
    const uint32_t patch=e.patchLeft!=kInvalidId?e.patchLeft:
                         (e.patchRight!=kInvalidId?e.patchRight:input.vertexPatchId[e.v0]);
    for(int i=1;i<n;++i){
      const float t=float(i)/float(n);
      Vec3 p=a+(b-a)*t;
      ProjectionResult hit=feature
          ? referenceProjector.projectFeature(e.featureCurveId,p)
          : referenceProjector.projectSurface(patch,p);
      if(hit.ok) p=hit.position;
      const VertexConstraint constraint=feature
          ? ((e.flags&EdgeSharp)?VertexConstraint::FeatureEdge:VertexConstraint::PatchBoundary)
          : VertexConstraint::Surface;
      const int v=output.addVertex(p,patch,constraint);
      output.targetLength[v]=(1.0f-t)*input.targetLength[e.v0]+t*input.targetLength[e.v1];
      chain.VertexIds.push_back(v);
    }
    chain.VertexIds.push_back(int(e.v1));
    if(int(chain.VertexIds.size())!=n+1){
      if(error)*error="edge subdivision chain size mismatch";
      return false;
    }
    if(subdivisions) subdivisions->push_back(std::move(chain));
  }
  if(subdivisions && subdivisions->size()!=input.edges.size()){
    if(error)*error="edge subdivision table size mismatch";
    return false;
  }
  if(subdivisions){
    std::unordered_map<uint64_t,size_t> chainByEdge;
    chainByEdge.reserve(subdivisions->size()*2);
    for(size_t i=0;i<subdivisions->size();++i){
      const auto& chain=(*subdivisions)[i];
      chainByEdge.emplace((uint64_t(std::min(chain.V0,chain.V1))<<32) |
                          uint64_t(std::max(chain.V0,chain.V1)),i);
    }
    for(int f=0;f<input.faceCount();++f){
      if(!input.faceAlive[f]) continue;
      ++stats.FacePlanCount;
      const auto fv=input.face(f);
      int longCount=0;
      for(int e=0;e<3;++e){
        const uint32_t a=uint32_t(fv[e]), b=uint32_t(fv[(e+1)%3]);
        const uint64_t key=(uint64_t(std::min(a,b))<<32)|uint64_t(std::max(a,b));
        auto it=chainByEdge.find(key);
        if(it==chainByEdge.end()){
          ++stats.FacePlanMissingChains;
          continue;
        }
        const auto& chain=(*subdivisions)[it->second];
        if(chain.SegmentCount>1) ++longCount;
        const bool forward=chain.V0==a && chain.V1==b;
        const bool reverse=chain.V0==b && chain.V1==a;
        if(!forward && !reverse) ++stats.FacePlanOrientationErrors;
        if(chain.VertexIds.empty() ||
           (forward && (chain.VertexIds.front()!=int(a)||chain.VertexIds.back()!=int(b))) ||
           (reverse && (chain.VertexIds.front()!=int(b)||chain.VertexIds.back()!=int(a))))
          ++stats.FacePlanOrientationErrors;
      }
      if(longCount==1) ++stats.FacePlanOneLong;
      else if(longCount==2) ++stats.FacePlanTwoLong;
      else if(longCount==3) ++stats.FacePlanThreeLong;
    }
  }
  const int inserted=output.vertexCount()-input.vertexCount();
  if(inserted!=stats.PlannedInsertedEdgeVertices){
    if(error)*error="materialized edge vertex count mismatch";
    return false;
  }
  return true;
}

bool triangulateTwoLongChains(const SemanticMesh& vertices,
                              const std::array<int,3>& parent,
                              const std::vector<int>& chain0,
                              const std::vector<int>& chain1,
                              std::vector<std::array<int,3>>& triangles,
                              std::string* error){
  triangles.clear();
  if(chain0.size()<2||chain1.size()<2){if(error)*error="two-long chain too short";return false;}
  if(chain0.front()!=chain1.front()){if(error)*error="two-long chains do not share apex";return false;}
  const int apex=chain0.front();
  if(apex!=parent[0]&&apex!=parent[1]&&apex!=parent[2]){if(error)*error="two-long apex not in parent";return false;}
  const int n0=int(chain0.size())-1,n1=int(chain1.size())-1;
  int i=1,j=1;
  triangles.push_back({apex,chain0[1],chain1[1]});
  while(i<n0||j<n1){
    const double t0=i<n0?double(i+1)/double(n0):2.0;
    const double t1=j<n1?double(j+1)/double(n1):2.0;
    if(t0<t1-1e-12){
      triangles.push_back({chain0[i],chain0[i+1],chain1[j]});
      ++i;
    }else if(t1<t0-1e-12){
      triangles.push_back({chain0[i],chain1[j+1],chain1[j]});
      ++j;
    }else{
      // Simultaneous advance forms a quad. Pick the diagonal maximizing the
      // weaker of its two triangle qualities.
      const int a=chain0[i],b=chain0[i+1],c=chain1[j+1],e=chain1[j];
      const float qa=std::min(triangleQuality(vertices.position(a),vertices.position(b),vertices.position(c)),
                              triangleQuality(vertices.position(a),vertices.position(c),vertices.position(e)));
      const float qb=std::min(triangleQuality(vertices.position(a),vertices.position(b),vertices.position(e)),
                              triangleQuality(vertices.position(b),vertices.position(c),vertices.position(e)));
      if(qa>=qb){triangles.push_back({a,b,c});triangles.push_back({a,c,e});}
      else {triangles.push_back({a,b,e});triangles.push_back({b,c,e});}
      ++i;++j;
    }
  }
  return true;
}

bool auditLocalTriangulation(const SemanticMesh& vertices,
                             const std::array<int,3>& parent,
                             const std::vector<std::array<int,3>>& triangles,
                             const std::vector<std::pair<int,int>>& boundarySegments,
                             LocalTriangulationAudit& audit,std::string* error){
  audit={};
  audit.QualityMin=1.0f;
  const Vec3 pa=vertices.position(parent[0]),pb=vertices.position(parent[1]),pc=vertices.position(parent[2]);
  const Vec3 parentN=cross(pb-pa,pc-pa);
  if(length2(parentN)<=1e-20f){if(error)*error="degenerate parent triangle";return false;}
  std::unordered_map<uint64_t,int> use;
  auto key=[](int a,int b){if(a>b)std::swap(a,b);return(uint64_t(uint32_t(a))<<32)|uint32_t(b);};
  double qsum=0.0;
  for(const auto&t:triangles){
    ++audit.TriangleCount;
    if(t[0]<0||t[1]<0||t[2]<0||t[0]>=vertices.vertexCount()||t[1]>=vertices.vertexCount()||t[2]>=vertices.vertexCount()){
      if(error)*error="local triangle vertex index out of range";return false;
    }
    const Vec3 a=vertices.position(t[0]),b=vertices.position(t[1]),c=vertices.position(t[2]);
    const Vec3 n=cross(b-a,c-a);
    if(length2(n)<=1e-20f) ++audit.ZeroAreaCount;
    if(dot(parentN,n)<=0.0f) ++audit.FlippedCount;
    const float q=triangleQuality(a,b,c);
    audit.QualityMin=std::min(audit.QualityMin,q);
    qsum+=q;
    ++use[key(t[0],t[1])];++use[key(t[1],t[2])];++use[key(t[2],t[0])];
  }
  std::unordered_set<uint64_t> boundary;
  boundary.reserve(boundarySegments.size()*2);
  for(const auto&e:boundarySegments) boundary.insert(key(e.first,e.second));
  for(const auto&e:boundary){
    auto it=use.find(e);
    if(it==use.end()||it->second!=1) ++audit.BoundaryMismatchCount;
  }
  for(const auto&kv:use){
    if(boundary.count(kv.first)){if(kv.second!=1)++audit.BoundaryMismatchCount;}
    else if(kv.second!=2) ++audit.InteriorNonManifoldCount;
  }
  audit.QualityMean=audit.TriangleCount?float(qsum/double(audit.TriangleCount)):0.0f;
  if(audit.TriangleCount==0) audit.QualityMin=0.0f;
  return audit.ZeroAreaCount==0&&audit.FlippedCount==0&&
         audit.BoundaryMismatchCount==0&&audit.InteriorNonManifoldCount==0;
}

} // namespace cad_adaptive
