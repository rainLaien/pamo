#include "cad_adaptive/TriangleRefineBackend.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace cad_adaptive {
namespace {
uint64_t EdgeKey(uint32_t a,uint32_t b){if(a>b)std::swap(a,b);return(uint64_t(a)<<32)|b;}
bool SameOrientation(Vec3 a,Vec3 b,Vec3 c,Vec3 x,Vec3 y,Vec3 z){
  const auto normal=[](Vec3 p,Vec3 q,Vec3 r){
    const double u[3]={double(q.x)-p.x,double(q.y)-p.y,double(q.z)-p.z};
    const double v[3]={double(r.x)-p.x,double(r.y)-p.y,double(r.z)-p.z};
    return std::array<double,3>{u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
  };
  const auto n=normal(a,b,c),m=normal(x,y,z);
  return n[0]*m[0]+n[1]*m[1]+n[2]*m[2]>0.;
}
}

bool refineMidpointConforming(const SemanticMesh& in,const RemeshConfig& cfg,
                              const GeometryProjector& referenceProjector,
                              SemanticMesh& out,TriangleRefineStats& st,std::string* error,
                              const std::unordered_set<uint32_t>* onlyPatches,
                              bool projectNewVertices,
                              const std::unordered_set<uint32_t>* forcePatches,
                              bool sizeFeasibleSplits){
  st={}; st.InputFaces=in.faceCount();
  std::unordered_set<uint64_t> marked;
  for(const auto&e:in.edges){
    const bool selected=!onlyPatches ||
        (e.face0>=0 && onlyPatches->count(in.facePatchId[e.face0])) ||
        (e.face1>=0 && onlyPatches->count(in.facePatchId[e.face1]));
    if(!selected)continue;
    const float h=0.5f*(in.targetLength[e.v0]+in.targetLength[e.v1]);
    if(h>0&&distance(in.position(int(e.v0)),in.position(int(e.v1)))>cfg.splitRatio*h) {
      const Vec3 a=in.position(int(e.v0)),b=in.position(int(e.v1));
      const Vec3 midpoint=(a+b)*.5f;
      // At large coordinate magnitudes a representable edge can still be
      // shorter than one float ULP at its midpoint. Splitting it would create
      // duplicate vertices and zero-area children, so leave that edge intact.
      if((midpoint.x!=a.x || midpoint.y!=a.y || midpoint.z!=a.z) &&
         (midpoint.x!=b.x || midpoint.y!=b.y || midpoint.z!=b.z))
        marked.insert(EdgeKey(e.v0,e.v1));
    }
  }
  std::unordered_map<uint64_t,Vec3> splitPoints;
  splitPoints.reserve(marked.size()*2);
  // Choose a split point on each long edge by maximizing the worst child-face
  // quality across both incident triangles. Include off-center candidates so
  // sliver faces are not forced to use a midpoint that collapses numerically.
  static constexpr float splitFractions[]={.1f,.2f,.3f,.4f,.5f,.6f,.7f,.8f,.9f};
  for(const auto &edge:in.edges) {
    const uint64_t key=EdgeKey(edge.v0,edge.v1);
    if(!marked.count(key))continue;
    const Vec3 a=in.position(int(edge.v0)),b=in.position(int(edge.v1));
    float bestScore=-1.f;
    float bestChildRatio=std::numeric_limits<float>::max();
    bool bestFeasible=false;
    Vec3 bestPoint{};
    std::vector<float> fractions(std::begin(splitFractions),std::end(splitFractions));
    const float h0=in.targetLength[edge.v0],h1=in.targetLength[edge.v1],hm=.5f*(h0+h1);
    if(sizeFeasibleSplits && h0+h1>0)
      fractions.push_back((h0+hm)/(h0+h1+2.f*hm));
    auto considerFraction=[&](float t) {
      const Vec3 point=a+(b-a)*t;
      if((point.x==a.x&&point.y==a.y&&point.z==a.z) ||
         (point.x==b.x&&point.y==b.y&&point.z==b.z))return;
      float score=std::numeric_limits<float>::max();
      for(int f:{edge.face0,edge.face1}) {
        if(f<0)continue;
        const auto tri=in.face(f);
        int opposite=-1;
        for(int v:tri)if(v!=int(edge.v0)&&v!=int(edge.v1)){opposite=v;break;}
        if(opposite<0)continue;
        const Vec3 c=in.position(opposite);
        if(sizeFeasibleSplits && (!SameOrientation(a,b,c,a,point,c) ||
                                  !SameOrientation(a,b,c,point,b,c))) {score=-1.f;break;}
        score=std::min(score,triangleQuality(a,point,c));
        score=std::min(score,triangleQuality(point,b,c));
      }
      // Slightly prefer a balanced split when child quality is comparable.
      score*=1.f-.05f*std::abs(t-.5f);
      if(score<=0.f)return;
      const float leftTarget=.5f*(h0+hm),rightTarget=.5f*(h1+hm);
      const float childRatio=leftTarget>0 && rightTarget>0?
          std::max(distance(a,point)/leftTarget,distance(point,b)/rightTarget):
          std::numeric_limits<float>::max();
      const bool feasible=childRatio<=cfg.splitRatio*(1.f+1.e-4f);
      // Avoid repeatedly splitting a residual long child when a one-level
      // solution exists. For a genuinely very long edge, first minimize the
      // largest child ratio. Quality breaks ties and ranks feasible points.
      const bool better=!sizeFeasibleSplits?score>bestScore:
          (bestScore<0 || (feasible && !bestFeasible) ||
           (feasible==bestFeasible && (feasible?score>bestScore:
             childRatio<bestChildRatio-1.e-5f ||
             (std::abs(childRatio-bestChildRatio)<=1.e-5f && score>bestScore))));
      if(better){bestScore=score;bestPoint=point;bestChildRatio=childRatio;bestFeasible=feasible;}
    };
    for(float t:fractions)considerFraction(t);
    if(sizeFeasibleSplits && !(bestScore>0.f)) {
      // Float-coordinate slivers can have disconnected intervals of safe
      // split positions. Exhausting a coarse grid is not a geometric proof
      // that no split exists. Search a bounded finer dyadic grid before
      // retaining this edge for another regional strategy.
      for(int sample=1;sample<64;++sample)considerFraction(float(sample)/64.f);
    }
    if(!(bestScore>0.f))marked.erase(key);
    else splitPoints.emplace(key,bestPoint);
  }
  static constexpr int refineTriNum[8]={1,2,2,3,2,3,3,4};
  static constexpr int refineTv[8][4][3]={
    {{0,1,2},{0,0,0},{0,0,0},{0,0,0}},
    {{0,3,2},{3,1,2},{0,0,0},{0,0,0}},
    {{0,1,4},{0,4,2},{0,0,0},{0,0,0}},
    {{3,1,4},{0,3,2},{4,2,3},{0,0,0}},
    {{0,1,5},{5,1,2},{0,0,0},{0,0,0}},
    {{0,3,5},{3,1,5},{2,5,1},{0,0,0}},
    {{2,5,4},{0,1,5},{4,5,1},{0,0,0}},
    {{3,4,5},{0,3,5},{3,1,4},{5,4,2}}
  };
  auto maskIsSafe=[&](int mask,const Vec3 *p) {
    if(mask==0)return true;
    const int vv[6]={0,1,2,3,4,5};
    int local[4][3]{};
    for(int q=0;q<refineTriNum[mask];++q)
      for(int j=0;j<3;++j)local[q][j]=vv[refineTv[mask][q][j]];
    if(mask==3||mask==5||mask==6) {
      int a0,a1,b0,b1;
      if(mask==3){a0=0;a1=4;b0=3;b1=2;}
      else if(mask==5){a0=3;a1=2;b0=5;b1=1;}
      else {a0=0;a1=4;b0=5;b1=1;}
      const auto d2=[&](int x,int y){const Vec3 d=p[x]-p[y];return length2(d);};
      if(d2(a0,a1)<d2(b0,b1)) {
        local[2][1]=local[1][0];
        local[1][1]=local[2][0];
      }
    }
    for(int q=0;q<refineTriNum[mask];++q) {
      if(!(triangleQuality(p[local[q][0]],p[local[q][1]],p[local[q][2]])>0.f))return false;
      if(sizeFeasibleSplits && !SameOrientation(p[0],p[1],p[2],
          p[local[q][0]],p[local[q][1]],p[local[q][2]]))return false;
    }
    return true;
  };
  bool removedUnsafe=true;
  while(removedUnsafe) {
    removedUnsafe=false;
    for(int f=0;f<in.faceCount();++f) {
      const int v[3]={int(in.i0[f]),int(in.i1[f]),int(in.i2[f])};
      const uint64_t k[3]={EdgeKey(v[0],v[1]),EdgeKey(v[1],v[2]),EdgeKey(v[2],v[0])};
      const bool s[3]={marked.count(k[0])!=0,marked.count(k[1])!=0,marked.count(k[2])!=0};
      const int mask=(s[0]?1:0)|(s[1]?2:0)|(s[2]?4:0);
      if(mask==0)continue;
      const Vec3 p[6]={in.position(v[0]),in.position(v[1]),in.position(v[2]),
          s[0]?splitPoints[k[0]]:Vec3{},
          s[1]?splitPoints[k[1]]:Vec3{},
          s[2]?splitPoints[k[2]]:Vec3{}};
      if(maskIsSafe(mask,p))continue;
      bool reduced=false;
      for(int e=0;e<3;++e)if(s[e] && maskIsSafe(mask&~(1<<e),p)) {
        removedUnsafe|=marked.erase(k[e])!=0;
        splitPoints.erase(k[e]);
        reduced=true;
        break;
      }
      if(!reduced)for(int e=0;e<3;++e)if(s[e]) {
        removedUnsafe|=marked.erase(k[e])!=0;
        splitPoints.erase(k[e]);
      }
    }
  }
  st.SplitEdges=int(marked.size());
  if(marked.empty()){out=in;st.OutputFaces=in.faceCount();return true;}

  out=in;
  out.i0.clear();out.i1.clear();out.i2.clear();out.facePatchId.clear();out.facePatchType.clear();out.faceAlive.clear();
  out.edges.clear();out.incidentFaces.assign(out.vertexCount(),{});
  std::unordered_map<uint64_t,int> mids;
  mids.reserve(marked.size()*2);
  std::unordered_map<uint64_t,const EdgeRec*> edgeByKey;
  edgeByKey.reserve(in.edges.size()*2);
  for(const auto &e:in.edges)edgeByKey.emplace(EdgeKey(e.v0,e.v1),&e);
  auto mid=[&](int a,int b,uint32_t patch){
    const uint64_t k=EdgeKey(uint32_t(a),uint32_t(b));
    auto it=mids.find(k);if(it!=mids.end())return it->second;
    Vec3 p=splitPoints.at(k);
    const auto ca=VertexConstraint(in.vertexConstraint[a]),cb=VertexConstraint(in.vertexConstraint[b]);
    VertexConstraint cc=VertexConstraint::Surface;
    uint8_t parentFlags=0;
    uint32_t feature=0;
    if(auto edge=edgeByKey.find(k);edge!=edgeByKey.end()){
      const auto &e=*edge->second;
      parentFlags=e.flags;
      feature=e.featureCurveId;
      // EdgeProtected is a generic topology guard, not a geometric feature.
      // Only true geometric/boundary edge classes may promote a midpoint to a
      // 1-D constraint.
      if(e.flags&(EdgeSharp|EdgePatchBoundary|EdgeMeshBoundary))
        cc=(e.flags&EdgeSharp)?VertexConstraint::FeatureEdge:VertexConstraint::PatchBoundary;
    }
    if(projectNewVertices) {
      ProjectionResult hit;
      if(parentFlags&(EdgeSharp|EdgePatchBoundary|EdgeMeshBoundary))
        hit=referenceProjector.projectFeature(feature,p);
      else
        hit=referenceProjector.projectSurface(patch,p);
      if(hit.ok)p=hit.position;
    }
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
    if(mask==0 && forcePatches && forcePatches->count(p)) {
      // Coverage-only pass: split the face around an interior centroid. This
      // changes every still-untouched patch without inserting seam vertices,
      // so neighboring patches remain conforming and boundary identities stay
      // intact. Skip a centroid that collapses under float precision.
      const Vec3 center=(in.position(v[0])+in.position(v[1])+in.position(v[2]))*(1.f/3.f);
      if(triangleQuality(in.position(v[0]),in.position(v[1]),center)>0.f &&
         triangleQuality(in.position(v[1]),in.position(v[2]),center)>0.f &&
         triangleQuality(in.position(v[2]),in.position(v[0]),center)>0.f) {
        const int c=out.addVertex(center,p,VertexConstraint::Surface);
        out.targetLength[c]=(in.targetLength[v[0]]+in.targetLength[v[1]]+in.targetLength[v[2]])/3.f;
        add(v[0],v[1],c,p,t);add(v[1],v[2],c,p,t);add(v[2],v[0],c,p,t);
        ++st.InsertedVertices;++st.RefinedFaces;
        st.touchedPatches.insert(p);
        continue;
      }
    }
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
    if(mask!=0){++st.RefinedFaces;st.touchedPatches.insert(p);}
    for(int q=0;q<triNum[mask];++q)add(local[q][0],local[q][1],local[q][2],p,t);
  }
  out.rebuildTopology();out.computeVertexNormals();st.OutputFaces=out.faceCount();
  return out.validate(error);
}

bool refineCoarseConforming(const SemanticMesh& input,const RemeshConfig& cfg,
                            const GeometryProjector& referenceProjector,
                            SemanticMesh& output,TriangleRefineStats& total,
                            int maxLevels,std::string* error,
                            const std::unordered_set<uint32_t>* onlyPatches,
                            bool projectNewVertices){
  total={};
  total.InputFaces=input.faceCount();
  SemanticMesh current=input;
  maxLevels=std::max(1,std::min(maxLevels,8));
  for(int level=0;level<maxLevels;++level){
    TriangleRefineStats step;
    SemanticMesh next;
    if(!refineMidpointConforming(current,cfg,referenceProjector,next,step,error,
                                 onlyPatches,projectNewVertices))
      return false;
    total.RefinedFaces+=step.RefinedFaces;
    total.InsertedVertices+=step.InsertedVertices;
    total.SplitEdges+=step.SplitEdges;
    ++total.Levels;
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
