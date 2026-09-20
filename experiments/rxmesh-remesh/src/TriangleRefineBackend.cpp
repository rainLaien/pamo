#include "cad_adaptive/TriangleRefineBackend.h"
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace cad_adaptive {
namespace {
uint64_t EdgeKey(uint32_t a,uint32_t b){if(a>b)std::swap(a,b);return(uint64_t(a)<<32)|b;}
}

bool refineMidpointConforming(const SemanticMesh& in,const RemeshConfig& cfg,
                              const GeometryProjector& referenceProjector,
                              SemanticMesh& out,TriangleRefineStats& st,std::string* error){
  st={}; st.InputFaces=in.faceCount();
  std::unordered_set<uint64_t> marked;
  for(const auto&e:in.edges){
    const float h=0.5f*(in.targetLength[e.v0]+in.targetLength[e.v1]);
    if(h>0&&distance(in.position(int(e.v0)),in.position(int(e.v1)))>cfg.splitRatio*h)
      marked.insert(EdgeKey(e.v0,e.v1));
  }
  st.SplitEdges=int(marked.size());
  if(marked.empty()){out=in;st.OutputFaces=in.faceCount();return true;}

  out=in;
  out.i0.clear();out.i1.clear();out.i2.clear();out.facePatchId.clear();out.facePatchType.clear();out.faceAlive.clear();
  out.edges.clear();out.incidentFaces.assign(out.vertexCount(),{});
  std::unordered_map<uint64_t,int> mids;
  mids.reserve(marked.size()*2);
  auto mid=[&](int a,int b,uint32_t patch){
    const uint64_t k=EdgeKey(uint32_t(a),uint32_t(b));
    auto it=mids.find(k);if(it!=mids.end())return it->second;
    Vec3 p=(in.position(a)+in.position(b))*0.5f;
    const auto ca=VertexConstraint(in.vertexConstraint[a]),cb=VertexConstraint(in.vertexConstraint[b]);
    VertexConstraint cc=VertexConstraint::Surface;
    uint8_t parentFlags=0;
    uint32_t feature=0;
    for(const auto&e:in.edges)if(EdgeKey(e.v0,e.v1)==k){
      parentFlags=e.flags;
      feature=e.featureCurveId;
      // EdgeProtected is a generic topology guard, not a geometric feature.
      // Only true geometric/boundary edge classes may promote a midpoint to a
      // 1-D constraint.
      if(e.flags&(EdgeSharp|EdgePatchBoundary|EdgeMeshBoundary))
        cc=(e.flags&EdgeSharp)?VertexConstraint::FeatureEdge:VertexConstraint::PatchBoundary;
      break;
    }
    ProjectionResult hit;
    if(parentFlags&(EdgeSharp|EdgePatchBoundary|EdgeMeshBoundary))
      hit=referenceProjector.projectFeature(feature,p);
    else
      hit=referenceProjector.projectSurface(patch,p);
    if(hit.ok)p=hit.position;
    const int v=out.addVertex(p,patch,cc);
    out.targetLength[v]=0.5f*(in.targetLength[a]+in.targetLength[b]);
    if(parentFlags&EdgeSharp) {
      out.featureEdges[EdgeKey(uint32_t(a),uint32_t(v))]=feature;
      out.featureEdges[EdgeKey(uint32_t(v),uint32_t(b))]=feature;
      out.featureEdges.erase(k);
    }
    mids.emplace(k,v);++st.InsertedVertices;return v;
  };
  auto add=[&](int a,int b,int c,uint32_t p,PatchType t){out.addFace(a,b,c,p,t);};

  for(int f=0;f<in.faceCount();++f){
    if(!in.faceAlive[f])continue;
    const int v[3]={int(in.i0[f]),int(in.i1[f]),int(in.i2[f])};
    const uint64_t k[3]={EdgeKey(v[0],v[1]),EdgeKey(v[1],v[2]),EdgeKey(v[2],v[0])};
    const bool s[3]={marked.count(k[0])!=0,marked.count(k[1])!=0,marked.count(k[2])!=0};
    const int mask=(s[0]?1:0)|(s[1]?2:0)|(s[2]?4:0);
    ++st.MaskCounts[mask];
    const uint32_t p=in.facePatchId[f];const PatchType t=PatchType(in.facePatchType[f]);
    int m[3]={-1,-1,-1};for(int e=0;e<3;++e)if(s[e])m[e]=mid(v[e],v[(e+1)%3],p);
    // Exact VCGLib SplitTab convention:
    // vv 0..2 = original vertices, 3=m01, 4=m12, 5=m20.
    const int vv[6]={v[0],v[1],v[2],m[0],m[1],m[2]};
    static constexpr int triNum[8]={1,2,2,3,2,3,3,4};
    static const int tv[8][4][3]={
      {{0,1,2},{0,0,0},{0,0,0},{0,0,0}},
      {{0,3,2},{3,1,2},{0,0,0},{0,0,0}},
      {{0,1,4},{0,4,2},{0,0,0},{0,0,0}},
      {{3,1,4},{0,3,2},{4,2,3},{0,0,0}},
      {{0,1,5},{5,1,2},{0,0,0},{0,0,0}},
      {{0,3,5},{3,1,5},{2,5,1},{0,0,0}},
      {{2,5,4},{0,1,5},{4,5,1},{0,0,0}},
      {{3,4,5},{0,3,5},{3,1,4},{5,4,2}}
    };
    int local[4][3]{};
    for(int q=0;q<triNum[mask];++q)
      for(int j=0;j<3;++j)local[q][j]=vv[tv[mask][q][j]];
    if(mask==3||mask==5||mask==6){
      int a0,a1,b0,b1;
      if(mask==3){a0=vv[0];a1=vv[4];b0=vv[3];b1=vv[2];}
      else if(mask==5){a0=vv[3];a1=vv[2];b0=vv[5];b1=vv[1];}
      else {a0=vv[0];a1=vv[4];b0=vv[5];b1=vv[1];}
      if(distance(out.position(a0),out.position(a1)) <
         distance(out.position(b0),out.position(b1))){
        local[2][1]=local[1][0];
        local[1][1]=local[2][0];
      }
    }
    if(mask!=0)++st.RefinedFaces;
    for(int q=0;q<triNum[mask];++q)add(local[q][0],local[q][1],local[q][2],p,t);
  }
  out.rebuildTopology();out.computeVertexNormals();st.OutputFaces=out.faceCount();
  return out.validate(error);
}

bool refineCoarseConforming(const SemanticMesh& input,const RemeshConfig& cfg,
                            const GeometryProjector& referenceProjector,
                            SemanticMesh& output,TriangleRefineStats& total,
                            int maxLevels,std::string* error){
  total={};
  total.InputFaces=input.faceCount();
  SemanticMesh current=input;
  maxLevels=std::max(1,std::min(maxLevels,8));
  for(int level=0;level<maxLevels;++level){
    TriangleRefineStats step;
    SemanticMesh next;
    if(!refineMidpointConforming(current,cfg,referenceProjector,next,step,error))
      return false;
    total.RefinedFaces+=step.RefinedFaces;
    total.InsertedVertices+=step.InsertedVertices;
    total.SplitEdges+=step.SplitEdges;
    for(int i=0;i<8;++i) total.MaskCounts[i]+=step.MaskCounts[i];
    current=std::move(next);
    std::cout<<"coarse_refine_level="<<level
             <<" split_edges="<<step.SplitEdges
             <<" inserted_vertices="<<step.InsertedVertices
             <<" faces="<<current.faceCount()<<'\n';
    if(step.SplitEdges==0) break;
  }
  output=std::move(current);
  total.OutputFaces=output.faceCount();
  return output.validate(error);
}
} // namespace cad_adaptive
