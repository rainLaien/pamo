#include "cad_adaptive/RawCudaBatch.h"
#include "cad_adaptive/PartitionInput.h"
#include "cad_adaptive/TriangleRefineBackend.h"
#include "cad_adaptive/BoundarySizingField.h"
#include "RemeshWorkerPool.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_set>

namespace cad_adaptive {
int boundaryTargetMismatchCount(const SemanticMesh& before,const std::vector<int>& beforeAnchors,
                                const SemanticMesh& after,const std::vector<int>& afterAnchors) {
  if(beforeAnchors.size()!=size_t(before.vertexCount()) || afterAnchors.size()!=size_t(after.vertexCount()))
    throw std::runtime_error("invalid boundary sizing anchor map");
  std::map<int,float> targets;
  for(int v=0;v<before.vertexCount();++v)if(beforeAnchors[v]>=0)targets.emplace(beforeAnchors[v],before.targetLength[v]);
  std::set<int> boundary;
  for(const auto &edge:after.edges)if(edge.flags&EdgeMeshBoundary) {boundary.insert(int(edge.v0));boundary.insert(int(edge.v1));}
  int mismatch=0;
  for(int v:boundary) {
    const auto found=targets.find(afterAnchors[v]);
    const float h=after.targetLength[v];
    if(afterAnchors[v]<0 || found==targets.end() || !(h>0) || !(found->second>0) ||
       std::abs(h-found->second)>std::max(1.e-6f,std::abs(found->second)*1.e-6f))++mismatch;
  }
  return mismatch;
}
namespace {
using Clock=std::chrono::steady_clock;
using Position=std::tuple<float,float,float>;
Position positionKey(Vec3 p){return {p.x,p.y,p.z};}
uint64_t edgeKey(int a,int b){return (uint64_t(std::min(a,b))<<32)|uint32_t(std::max(a,b));}
void remeshCircularPlaneInteriors(SemanticMesh &mesh,const RemeshConfig &cfg,RawBatchReport &report) {
  if(!(cfg.featureEdgeLength>0.f))return;
  const int originalFaces=mesh.faceCount();
  bool insertedVertices=false;
  std::set<uint32_t> remeshedPatches;
  std::vector<std::vector<int>> patchFaces(mesh.patches.size());
  std::vector<std::vector<std::pair<int,int>>> patchBoundary(mesh.patches.size());
  for(int f=0;f<originalFaces;++f)if(mesh.faceAlive[f] && mesh.facePatchId[f]<patchFaces.size())
    patchFaces[mesh.facePatchId[f]].push_back(f);
  for(const auto &edge:mesh.edges) {
    const int a=edge.face0>=0?int(mesh.facePatchId[edge.face0]):-1;
    const int b=edge.face1>=0?int(mesh.facePatchId[edge.face1]):-1;
    if(a==b)continue;
    if(a>=0 && a<int(patchBoundary.size()))patchBoundary[a].push_back({int(edge.v0),int(edge.v1)});
    if(b>=0 && b<int(patchBoundary.size()))patchBoundary[b].push_back({int(edge.v0),int(edge.v1)});
  }
  constexpr double pi=3.14159265358979323846;
  for(size_t patch=0;patch<mesh.patches.size();++patch) {
    const auto &faces=patchFaces[patch];
    const auto &edges=patchBoundary[patch];
    if(mesh.patches[patch].type!=PatchType::Plane || faces.size()<12 ||
       edges.size()<10 || edges.size()>128)continue;
    std::map<int,std::vector<int>> neighbors;
    for(auto [a,b]:edges) {neighbors[a].push_back(b);neighbors[b].push_back(a);}
    if(neighbors.size()!=edges.size() ||
       std::any_of(neighbors.begin(),neighbors.end(),[](const auto &item){return item.second.size()!=2;}))continue;
    std::vector<int> outer;outer.reserve(edges.size());
    int current=edges.front().first,previous=-1;
    for(size_t k=0;k<edges.size();++k) {
      outer.push_back(current);
      const auto &adjacent=neighbors[current];
      const int next=adjacent[0]==previous?adjacent[1]:adjacent[0];
      previous=current;current=next;
      if(current==outer.front() && k+1<edges.size())break;
    }
    if(outer.size()!=edges.size() || current!=outer.front())continue;
    const auto first=mesh.face(faces.front());
    const Vec3 normal=triangleNormal(mesh.position(first[0]),mesh.position(first[1]),mesh.position(first[2]));
    if(length2(normal)<.5f)continue;
    Vec3 center{};
    for(int v:outer)center=center+mesh.position(v);
    center=center*(1.f/float(outer.size()));
    double area=0,perimeter=0,signedArea=0;
    for(int f:faces) {
      auto t=mesh.face(f);
      area+=.5*length(cross(mesh.position(t[1])-mesh.position(t[0]),
                              mesh.position(t[2])-mesh.position(t[0])));
    }
    for(size_t i=0;i<outer.size();++i) {
      const Vec3 a=mesh.position(outer[i]),b=mesh.position(outer[(i+1)%outer.size()]);
      perimeter+=distance(a,b);
      signedArea+=dot(cross(a-center,b-center),normal);
    }
    if(!(perimeter>0) || signedArea==0)continue;
    if(signedArea<0)std::reverse(outer.begin(),outer.end());
    const double circularity=4*pi*area/(perimeter*perimeter);
    if(circularity<.9 || circularity>1.02)continue;
    ++report.circularPlaneSkipped;
    float inradius=std::numeric_limits<float>::max();
    bool convex=true;
    for(size_t i=0;i<outer.size();++i) {
      const Vec3 a=mesh.position(outer[i]),b=mesh.position(outer[(i+1)%outer.size()]);
      const Vec3 c=mesh.position(outer[(i+2)%outer.size()]);
      const Vec3 ab=b-a;
      if(dot(cross(ab,c-b),normal)<-1.e-5f*length2(ab)){convex=false;break;}
      inradius=std::min(inradius,length(cross(ab,center-a))/std::max(length(ab),1.e-12f));
    }
    if(!convex || !(inradius>0))continue;
    const Vec3 u=normalize(mesh.position(outer.front())-center),v=normalize(cross(normal,u));
    if(length2(u)<.5f || length2(v)<.5f)continue;
    std::vector<double> outerPhase;outerPhase.reserve(outer.size()+1);
    for(int id:outer) {
      const Vec3 d=mesh.position(id)-center;
      double angle=std::atan2(double(dot(d,v)),double(dot(d,u)));
      if(angle<0)angle+=2*pi;
      outerPhase.push_back(angle/(2*pi));
    }
    outerPhase.front()=0;
    bool monotone=true;
    for(size_t i=1;i<outerPhase.size();++i)
      if(outerPhase[i]<=outerPhase[i-1]+1.e-7){monotone=false;break;}
    if(!monotone)continue;
    outerPhase.push_back(1.);
    const int n0=std::min(int(outer.size())-1,std::max(8,int(std::lround(outer.size()*.72))));
    const int n1=std::min(n0-1,std::max(6,int(std::lround(outer.size()*.36))));
    std::vector<int> ring0,ring1;
    auto makeRing=[&](int count,float radius,std::vector<int> &ids) {
      ids.reserve(count);
      for(int j=0;j<count;++j) {
        const float angle=float(2*pi*j/count);
        const Vec3 p=center+(u*std::cos(angle)+v*std::sin(angle))*radius;
        const int id=mesh.addVertex(p,uint32_t(patch),VertexConstraint::Surface);
        const float distanceFromBoundary=std::max(0.f,inradius-radius);
        mesh.targetLength[id]=std::min(cfg.constantLength,cfg.featureEdgeLength+
            (cfg.constantLength-cfg.featureEdgeLength)*distanceFromBoundary/cfg.featureBand);
        ids.push_back(id);
      }
    };
    makeRing(n0,.7f*inradius,ring0);
    makeRing(n1,.35f*inradius,ring1);
    const int middle=mesh.addVertex(center,uint32_t(patch),VertexConstraint::Surface);
    insertedVertices=true;
    mesh.targetLength[middle]=std::min(cfg.constantLength,cfg.featureEdgeLength+
        (cfg.constantLength-cfg.featureEdgeLength)*inradius/cfg.featureBand);
    std::vector<std::array<int,3>> triangles;
    auto bridge=[&](const std::vector<int> &outside,const std::vector<double> &phase,
                    const std::vector<int> &inside) {
      const int no=int(outside.size()),ni=int(inside.size());
      int i=0,j=0;
      while(i<no || j<ni) {
        const double nextO=i<no?phase[i+1]:2.;
        const double nextI=j<ni?double(j+1)/ni:2.;
        if(nextO<=nextI) {
          triangles.push_back({outside[i%no],outside[(i+1)%no],inside[j%ni]});++i;
        } else {
          triangles.push_back({outside[i%no],inside[(j+1)%ni],inside[j%ni]});++j;
        }
      }
    };
    bridge(outer,outerPhase,ring0);
    std::vector<double> ringPhase(n0+1);
    for(int j=0;j<=n0;++j)ringPhase[j]=double(j)/n0;
    bridge(ring0,ringPhase,ring1);
    for(int j=0;j<n1;++j)triangles.push_back({ring1[j],ring1[(j+1)%n1],middle});
    std::vector<float> oldQuality,newQuality;oldQuality.reserve(faces.size());newQuality.reserve(triangles.size());
    for(int f:faces) {
      auto t=mesh.face(f);
      oldQuality.push_back(triangleQuality(mesh.position(t[0]),mesh.position(t[1]),mesh.position(t[2])));
    }
    bool valid=true;
    for(const auto &t:triangles) {
      const Vec3 a=mesh.position(t[0]),b=mesh.position(t[1]),c=mesh.position(t[2]);
      if(dot(cross(b-a,c-a),normal)<=1.e-8f || !std::isfinite(a.x+b.x+c.x)){valid=false;break;}
      newQuality.push_back(triangleQuality(a,b,c));
    }
    if(!valid)continue;
    const float oldMean=std::accumulate(oldQuality.begin(),oldQuality.end(),0.f)/oldQuality.size();
    const float newMean=std::accumulate(newQuality.begin(),newQuality.end(),0.f)/newQuality.size();
    std::sort(oldQuality.begin(),oldQuality.end());std::sort(newQuality.begin(),newQuality.end());
    if(newMean+1.e-5f<oldMean || newQuality[newQuality.size()/20]+1.e-5f<oldQuality[oldQuality.size()/20])continue;
    for(int f:faces)mesh.killFace(f);
    for(const auto &t:triangles)mesh.addFace(t[0],t[1],t[2],uint32_t(patch),PatchType::Plane);
    ++report.circularPlaneRemeshed;--report.circularPlaneSkipped;
    remeshedPatches.insert(uint32_t(patch));
    report.circularPlaneAddedFaces+=int(triangles.size())-int(faces.size());
  }
  if(insertedVertices) {
    mesh.compact();mesh.rebuildTopology();mesh.computeVertexNormals();
  }
  for(auto &task:report.patches) {
    auto &ids=task.unchangedPatchIds;
    ids.erase(std::remove_if(ids.begin(),ids.end(),[&](uint32_t id){return remeshedPatches.count(id)!=0;}),ids.end());
    if(!remeshedPatches.empty() && task.unchanged && ids.empty())task.unchanged=false;
  }
  report.unchanged=int(std::count_if(report.patches.begin(),report.patches.end(),
      [](const RawPatchResult &task){return task.accepted && task.unchanged;}));
}
bool featureSegmentCovered(const SemanticMesh &source,const SemanticMesh &output,
                           uint64_t sourceEdge,uint32_t featureId) {
  const uint32_t a=uint32_t(sourceEdge>>32),b=uint32_t(sourceEdge);
  if(a>=uint32_t(source.vertexCount())||b>=uint32_t(source.vertexCount()))return false;
  const Vec3 p0=source.position(int(a)),p1=source.position(int(b)),d=p1-p0;
  const float len2=length2(d),len=std::sqrt(len2);
  if(!(len2>0.f))return false;
  const float tolerance=std::max(1.e-6f,source.bboxDiagonal()*1.e-6f);
  std::vector<std::pair<float,float>> intervals;
  for(const auto &feature:output.featureEdges) {
    if(feature.second!=featureId)continue;
    const uint32_t u=uint32_t(feature.first>>32),v=uint32_t(feature.first);
    if(u>=uint32_t(output.vertexCount())||v>=uint32_t(output.vertexCount()))continue;
    const Vec3 x=output.position(int(u)),y=output.position(int(v));
    const float tx=dot(x-p0,d)/len2,ty=dot(y-p0,d)/len2;
    if(tx < -tolerance/len || tx > 1.f+tolerance/len ||
       ty < -tolerance/len || ty > 1.f+tolerance/len)continue;
    if(distance(x,p0+d*tx)>tolerance||distance(y,p0+d*ty)>tolerance)continue;
    intervals.emplace_back(std::min(tx,ty),std::max(tx,ty));
  }
  std::sort(intervals.begin(),intervals.end());
  float covered=0.f;
  for(const auto &interval:intervals) {
    if(interval.first>covered+tolerance/len)continue;
    covered=std::max(covered,interval.second);
    if(covered>=1.f-tolerance/len)return true;
  }
  return false;
}
std::pair<float,float> qualitySummary(const SemanticMesh &mesh) {
  std::vector<float> values;values.reserve(mesh.faceCount());
  double sum=0;
  for(int f=0;f<mesh.faceCount();++f)if(mesh.faceAlive[f]) {
    const auto t=mesh.face(f);
    const float q=triangleQuality(mesh.position(t[0]),mesh.position(t[1]),mesh.position(t[2]));
    values.push_back(q);sum+=q;
  }
  if(values.empty())return {0.f,0.f};
  const size_t p05=values.size()/20;
  std::nth_element(values.begin(),values.begin()+p05,values.end());
  return {float(sum/double(values.size())),values[p05]};
}
bool meshChanged(const SemanticMesh &a,const SemanticMesh &b) {
  if(a.vertexCount()!=b.vertexCount() || a.faceCount()!=b.faceCount() ||
     a.i0!=b.i0 || a.i1!=b.i1 || a.i2!=b.i2)return true;
  for(int v=0;v<a.vertexCount();++v) {
    const Vec3 p=a.position(v),q=b.position(v);
    if(p.x!=q.x || p.y!=q.y || p.z!=q.z)return true;
  }
  return false;
}
void cudaCheck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct FacePartitionItem {
  int Face=0;
  Vec3 Centroid{};
};
void PartitionFacesRecursive(std::vector<FacePartitionItem> &items,size_t begin,size_t end,
                             int firstPartition,int partitionCount,std::vector<uint32_t> &facePatchId) {
  if(partitionCount<=1 || end-begin<=1) {
    for(size_t i=begin;i<end;++i) facePatchId[items[i].Face]=uint32_t(firstPartition);
    return;
  }
  Vec3 lo=items[begin].Centroid,hi=lo;
  for(size_t i=begin+1;i<end;++i) {
    const Vec3 c=items[i].Centroid;
    lo.x=std::min(lo.x,c.x);lo.y=std::min(lo.y,c.y);lo.z=std::min(lo.z,c.z);
    hi.x=std::max(hi.x,c.x);hi.y=std::max(hi.y,c.y);hi.z=std::max(hi.z,c.z);
  }
  const Vec3 span=hi-lo;
  const int axis=span.y>span.x ? (span.z>span.y?2:1) : (span.z>span.x?2:0);
  auto coord=[axis](const FacePartitionItem &x){return axis==0?x.Centroid.x:(axis==1?x.Centroid.y:x.Centroid.z);};
  const int leftPartitions=partitionCount/2;
  const int rightPartitions=partitionCount-leftPartitions;
  const size_t leftCount=(end-begin)*size_t(leftPartitions)/size_t(partitionCount);
  const size_t mid=begin+std::max<size_t>(1,std::min(end-begin-1,leftCount));
  std::nth_element(items.begin()+begin,items.begin()+mid,items.begin()+end,
                   [&](const FacePartitionItem &a,const FacePartitionItem &b){
                     const float ca=coord(a),cb=coord(b);
                     return ca<cb || (ca==cb && a.Face<b.Face);
                   });
  PartitionFacesRecursive(items,begin,mid,firstPartition,leftPartitions,facePatchId);
  PartitionFacesRecursive(items,mid,end,firstPartition+leftPartitions,rightPartitions,facePatchId);
}
int AutoPartitionSinglePatch(SemanticMesh &mesh,int requestedPartitions) {
  if(requestedPartitions<=1 || mesh.faceCount()<2 || mesh.patches.size()!=1) return int(mesh.patches.size());
  const int partitions=std::max(1,std::min(requestedPartitions,mesh.faceCount()));
  std::vector<FacePartitionItem> items;items.reserve(mesh.faceCount());
  for(int f=0;f<mesh.faceCount();++f) if(mesh.faceAlive.empty() || mesh.faceAlive[f])
    items.push_back({f,centroid3(mesh.facePoint(f,0),mesh.facePoint(f,1),mesh.facePoint(f,2))});
  if(items.size()<2)return 1;
  mesh.facePatchId.assign(mesh.faceCount(),0u);
  PartitionFacesRecursive(items,0,items.size(),0,partitions,mesh.facePatchId);
  const PatchRecord base=mesh.patches.front();
  mesh.patches.assign(partitions,base);
  mesh.facePatchType.resize(mesh.faceCount(),uint8_t(base.type));
  mesh.rebuildTopology();
  return partitions;
}
struct Job {
  int id=0,firstPatch=0,patchCount=0; double cost=0; size_t bytes=0;
  bool succeeded=false,unchanged=false,uniformSizing=false,qualitySplit=false,meanRecovery=false,sizeRecovery=false;
  float qualityMeanFloor=0,qualityP05Floor=0;
  std::vector<int> faces;
  SemanticMesh output;
  std::vector<int> aliases;
  std::vector<Job> children;
};
Job extractIncumbentPatch(const Job &parent,int patch) {
  Job part;part.firstPatch=parent.firstPatch+patch;part.patchCount=1;
  part.bytes=parent.bytes;part.succeeded=true;part.unchanged=parent.unchanged;
  part.output.patches.push_back(parent.output.patches[patch]);
  if(parent.output.LocalSizing)part.output.LocalSizing=parent.output.LocalSizing->subset(patch,1);
  std::map<int,int> vertices;
  for(int f=0;f<parent.output.faceCount();++f)if(parent.output.faceAlive[f] &&
       parent.output.facePatchId[f]==uint32_t(patch)) {
    auto t=parent.output.face(f);
    for(int &v:t) {
      const int old=v;auto found=vertices.find(old);
      if(found==vertices.end()) {
        v=part.output.addVertex(parent.output.position(old),0,VertexConstraint(parent.output.vertexConstraint[old]));
        part.output.targetLength[v]=parent.output.targetLength[old];
        part.aliases.push_back(parent.aliases[old]);vertices.emplace(old,v);
      } else v=found->second;
    }
    part.output.addFace(t[0],t[1],t[2],0,part.output.patches[0].type);
  }
  for(auto feature:parent.output.featureEdges) {
    auto a=vertices.find(int(feature.first>>32)),b=vertices.find(int(uint32_t(feature.first)));
    if(a!=vertices.end() && b!=vertices.end())part.output.featureEdges[edgeKey(a->second,b->second)]=feature.second;
  }
  part.output.rebuildTopology();
  return part;
}
bool splitConflictingChord(Job &job,int sourceA,int sourceB,std::string &reason) {
  if(!job.succeeded){reason="job not accepted";return false;}
  int a=-1,b=-1;
  for(int v=0;v<int(job.aliases.size());++v) {
    if(job.aliases[v]==sourceA)a=v;
    if(job.aliases[v]==sourceB)b=v;
  }
  if(a<0 || b<0){reason="chord endpoint absent";return false;}
  SemanticMesh candidate=job.output;
  candidate.rebuildTopology();
  const uint64_t chord=edgeKey(a,b);
  for(const auto &edge:candidate.edges) {
    if(edgeKey(edge.v0,edge.v1)!=chord)continue;
    if(edge.face0<0 || edge.face1<0 ||
       (edge.flags&(EdgePatchBoundary|EdgeMeshBoundary))){reason="chord is a boundary edge";return false;}
    const auto feature=candidate.featureEdges.find(chord);
    const bool sharp=feature!=candidate.featureEdges.end();
    const uint32_t featureId=sharp?feature->second:0;
    const int mid=candidate.addVertex((candidate.position(a)+candidate.position(b))*.5f,
                                      edge.patchLeft,sharp?VertexConstraint::FeatureEdge:
                                                           VertexConstraint::Surface);
    candidate.targetLength[mid]=.5f*(candidate.targetLength[a]+candidate.targetLength[b]);
    if(sharp) {
      candidate.featureEdges.erase(chord);
      candidate.featureEdges[edgeKey(a,mid)]=featureId;
      candidate.featureEdges[edgeKey(mid,b)]=featureId;
    }
    for(int f:{edge.face0,edge.face1}) {
      const auto t=candidate.face(f);
      const uint32_t patch=candidate.facePatchId[f];
      bool found=false;
      for(int k=0;k<3;++k) {
        const int u=t[k],v=t[(k+1)%3],w=t[(k+2)%3];
        if((u==a && v==b) || (u==b && v==a)) {
          candidate.killFace(f);
          candidate.addFace(u,mid,w,patch,PatchType(candidate.facePatchType[f]));
          candidate.addFace(mid,v,w,patch,PatchType(candidate.facePatchType[f]));
          found=true;break;
        }
      }
      if(!found){reason="incident triangle does not contain chord";return false;}
    }
    candidate.rebuildTopology();
    std::string problem;
    if(!candidate.validate(&problem)){reason="split topology: "+problem;return false;}
    // Seam repair is required to make the assembled topology conforming.
    // A quality-only rejection here used to discard the entire remeshed job
    // and restore its source faces, creating large untouched regions.
    job.aliases.resize(candidate.vertexCount(),-1);
    job.output=std::move(candidate);
    return true;
  }
  reason="chord absent from job";
  return false;
}
void execute(Job &job,const SemanticMesh &source,const RemeshConfig &config,
             bool stopWhenIdle,int smoothPasses,int collapsePasses,int flipPasses,bool strictFlipQuality,
             bool requireQualityImprovement,bool legacyCoverageAcceptance,bool exploreProvisionalChildren,bool trackRegionCandidates,
             float lowQualityThreshold,RawPatchResult &result) {
  const auto start=Clock::now();
  result.inputFaces=int(job.faces.size());result.workspaceBytes=job.bytes;
  std::vector<RegionQuality> originalPatchQuality;
  try {
    SemanticMesh local;
    local.patches.assign(source.patches.begin()+job.firstPatch,
                         source.patches.begin()+job.firstPatch+job.patchCount);
    if(source.LocalSizing) {
      const auto field=source.LocalSizing->subset(job.firstPatch,job.patchCount);
      if(!field->nodes().empty() || std::any_of(field->patches().begin(),field->patches().end(),
           [&](const BoundarySizingPatch &p){return p.BaseLength!=config.constantLength;}))
        local.LocalSizing=field;
    }
    std::map<int,int> indices;
    std::map<Position,int> anchors;
    std::vector<int> originals;
    for(int f:job.faces) {
      auto t=source.face(f);
      for(int &v:t) {
        auto found=indices.find(v);
        if(found==indices.end()) {
          const int original=v;
          const auto constraint=VertexConstraint(source.vertexConstraint[v]);
          const auto p=source.position(v);
          v=local.addVertex(p,source.facePatchId[f]-job.firstPatch,constraint);originals.push_back(original);
          indices.emplace(original,v);
          if(constraint==VertexConstraint::Locked && !anchors.emplace(positionKey(p),original).second)
            throw std::runtime_error("ambiguous coincident boundary identities");
        } else v=found->second;
      }
      const auto patch=source.facePatchId[f]-job.firstPatch;
      local.addFace(t[0],t[1],t[2],patch,source.patches[source.facePatchId[f]].type);
    }
    for(auto feature:source.featureEdges) {
      auto a=indices.find(int(feature.first>>32)),b=indices.find(int(uint32_t(feature.first)));
      if(a!=indices.end() && b!=indices.end())local.featureEdges[edgeKey(a->second,b->second)]=feature.second;
    }
    local.rebuildTopology();
    if(local.LocalSizing)local.LocalSizing->apply(local);
    const auto sourceRegionQuality=evaluateRegionQuality(local,config,lowQualityThreshold);
    const auto endpointQualitySafe=[&](const SemanticMesh& candidate) {
      return legacyCoverageAcceptance || !requireQualityImprovement ||
        qualityEndpointMask(sourceRegionQuality,evaluateRegionQuality(candidate,config,lowQualityThreshold))==0;
    };
    const auto localQualityFloor=requireQualityImprovement?qualitySummary(local):
        std::pair<float,float>{0.f,0.f};
    result.inputQualityMean=localQualityFloor.first;
    result.inputQualityP05=localQualityFloor.second;
    job.qualityMeanFloor=localQualityFloor.first;
    job.qualityP05Floor=localQualityFloor.second;
    std::set<uint64_t> required;
    for(auto e:local.edges) if(e.flags&EdgeMeshBoundary)
      required.insert(edgeKey(originals[e.v0],originals[e.v1]));
    const bool hasSharedOrOpenBoundary=source.patches.size()>1 ||
        std::any_of(source.edges.begin(),source.edges.end(),[](const EdgeRec &e){return (e.flags&EdgeMeshBoundary)!=0;});
    // The initial pass uses the same shape guards with or without a spatial
    // sizing field. Previously zero-initialized uniform targets disabled this
    // relaxation accidentally; filling valid local targets must not enable
    // aggressive collapse across every unrelated patch in the packed task.
    RawCudaOptions options;options.independentStream=true;options.quiet=true;options.error=&result.error;options.freezeBoundary=hasSharedOrOpenBoundary;options.stopWhenIdle=stopWhenIdle;options.smoothPasses=smoothPasses;options.collapsePasses=collapsePasses;options.flipPasses=flipPasses;options.strictFlipQuality=strictFlipQuality;options.allowQualityTradeoff=false;options.workspaceBytes=job.bytes;
    // Let the GPU finish its best valid cycle even when its quality metrics
    // trade off against the source. Quality floors here previously caused
    // remeshRawCuda to restore the input mesh wholesale for difficult patches.
    options.qualityMeanFloor=0.f;options.qualityP05Floor=0.f;
    options.trackRegionCandidates=trackRegionCandidates;options.lowQualityThreshold=lowQualityThreshold;
    SemanticMesh initialLocal;
    const SemanticMesh originalLocal=local;
    if(exploreProvisionalChildren && job.patchCount>1)
      originalPatchQuality=evaluatePatchQuality(originalLocal,config,lowQualityThreshold);
    options.referenceMesh=&originalLocal;
    if(requireQualityImprovement)initialLocal=local;
    auto phaseStart=Clock::now();
    const bool initialOk=remeshRawCuda(local,config,result.report,options);
    result.secondsInitial=std::chrono::duration<double>(Clock::now()-phaseStart).count();
    const float initialSizingP95=result.report.sizingErrorP95;
    const auto initialQuality=initialOk?qualitySummary(local):std::pair<float,float>{0.f,0.f};
    bool improved=initialOk && (!requireQualityImprovement ||
        (legacyCoverageAcceptance ? (result.report.selectedCycle>=0 &&
         initialQuality.first+1.e-6f>=localQualityFloor.first &&
         initialQuality.second+1.e-6f>=localQualityFloor.second) :
         endpointQualitySafe(local)));
    // Preserve a geometrically valid incumbent while trying recovery. A quality
    // failure is not an exception and must not restore the whole source region.
    SemanticMesh provisionalMesh;
    RemeshReport provisionalReport=result.report;
    bool provisionalValid=initialOk;
    if(!legacyCoverageAcceptance && initialOk && !improved)provisionalMesh=local;
    if(legacyCoverageAcceptance && !improved && initialOk && requireQualityImprovement && meshChanged(initialLocal,local)) {
      // A changed, valid local result is useful coverage even when no cycle
      // satisfies the old shape gate. Keep it and report the quality tradeoff
      // at the assembled-mesh level instead of silently restoring this patch.
      improved=true;
      result.qualitySplit=true;
      result.error.clear();
    }
    if(!improved && requireQualityImprovement && collapsePasses>1) {
      result.retried=true;
      result.retryReason=initialOk?"no cycle improved both local quality metrics":result.error;
      std::string retryError;
      RawCudaOptions retryOptions=options;
      retryOptions.error=&retryError;
      retryOptions.collapsePasses=1;
      // A small lower-tail tradeoff is allowed only with a substantial mean
      // gain. The full output still has to pass its global quality gate.
      retryOptions.qualityMeanFloor=localQualityFloor.first*1.01f;
      retryOptions.qualityP05Floor=localQualityFloor.second*.99f;
      if(!initialOk) {
        // Recover legal geometry first without selecting the unchanged source
        // solely because no intermediate cycle reaches a quality floor.
        retryOptions.qualityMeanFloor=retryOptions.qualityP05Floor=0.f;
      }
      RemeshConfig retryConfig=config;
      retryConfig.maxIterations=std::min(config.maxIterations,8);
      RemeshReport retryReport;
      phaseStart=Clock::now();
      if(remeshRawCuda(initialLocal,retryConfig,retryReport,retryOptions)) {
        if(!provisionalValid && !legacyCoverageAcceptance) {
          provisionalMesh=initialLocal;provisionalReport=retryReport;
          provisionalValid=true;
        }
        const auto retryQuality=qualitySummary(initialLocal);
        if(retryReport.selectedCycle>=0 &&
           retryQuality.first+1.e-6f>=retryOptions.qualityMeanFloor &&
           retryQuality.second+1.e-6f>=retryOptions.qualityP05Floor && endpointQualitySafe(initialLocal)) {
          local=std::move(initialLocal);
          result.report=retryReport;
          result.error.clear();
          improved=true;
        }
      }
      result.secondsGentle=std::chrono::duration<double>(Clock::now()-phaseStart).count();
      if(!improved && !retryError.empty())result.error=retryError;
    }
    const bool poorTailWithLargeSizingError=
        initialQuality.second<.02f && initialSizingP95>2.f;
    if(!improved && initialOk && requireQualityImprovement &&
       config.featureEdgeLength>0.f && !poorTailWithLargeSizingError) {
      // Crease-rich regions can accumulate slivers under curvature sizing.
      // The source's hard feature edges and frozen seam stay unchanged while
      // only the interior sizing field is switched to uniform.
      result.retried=true;
      RemeshConfig uniformConfig=config;
      uniformConfig.featureEdgeLength=0.f;
      uniformConfig.featureBand=0.f;
      uniformConfig.maxIterations=std::min(config.maxIterations,8);
      SemanticMesh uniformSource=local;
      RawCudaOptions uniformOptions=options;
      std::string uniformError;
      uniformOptions.error=&uniformError;
      uniformOptions.collapsePasses=1;
      uniformOptions.qualityMeanFloor=localQualityFloor.first*1.01f;
      uniformOptions.qualityP05Floor=localQualityFloor.second*.99f;
      RemeshReport uniformReport;
      SemanticMesh uniformMesh=uniformSource;
      phaseStart=Clock::now();
      bool uniformOk=remeshRawCuda(uniformMesh,uniformConfig,uniformReport,uniformOptions);
      result.secondsUniformGentle=std::chrono::duration<double>(Clock::now()-phaseStart).count();
      auto uniformQuality=uniformOk?qualitySummary(uniformMesh):std::pair<float,float>{0.f,0.f};
      if(!(uniformOk && uniformReport.selectedCycle>=0 &&
           uniformQuality.first+1.e-6f>=uniformOptions.qualityMeanFloor &&
           uniformQuality.second+1.e-6f>=uniformOptions.qualityP05Floor && endpointQualitySafe(uniformMesh))) {
        uniformMesh=std::move(uniformSource);
        uniformOptions=options;
        uniformOptions.error=&uniformError;
        phaseStart=Clock::now();
        uniformOk=remeshRawCuda(uniformMesh,uniformConfig,uniformReport,uniformOptions);
        result.secondsUniformStrict=std::chrono::duration<double>(Clock::now()-phaseStart).count();
        uniformQuality=uniformOk?qualitySummary(uniformMesh):std::pair<float,float>{0.f,0.f};
      }
      if(uniformOk && uniformReport.selectedCycle>=0 &&
         uniformQuality.first+1.e-6f>=uniformOptions.qualityMeanFloor &&
         uniformQuality.second+1.e-6f>=uniformOptions.qualityP05Floor && endpointQualitySafe(uniformMesh)) {
        local=std::move(uniformMesh);
        result.report=uniformReport;
        result.uniformSizing=true;
        result.error.clear();
        improved=true;
      }
    }
    if(!improved && initialOk && requireQualityImprovement) {
      result.retried=true;
      RemeshConfig splitConfig=config;
      splitConfig.maxIterations=std::min(config.maxIterations,8);
      SemanticMesh splitMesh=local;
      RawCudaOptions splitOptions=options;
      splitOptions.optimizeSplitPoint=true;
      // A region with a stronger input lower tail can tolerate a more
      // permissive intermediate split; the final region and sizing gates
      // still require non-regression before accepting the result.
      splitOptions.splitQualityRatio=localQualityFloor.second>.04f?.6f:.65f;
      std::string splitError;
      splitOptions.error=&splitError;
      RemeshReport splitReport;
      phaseStart=Clock::now();
      const bool splitOk=remeshRawCuda(splitMesh,splitConfig,splitReport,splitOptions);
      result.secondsQualitySplit+=std::chrono::duration<double>(Clock::now()-phaseStart).count();
      if(splitOk) {
        const auto splitQuality=qualitySummary(splitMesh);
        if(splitReport.selectedCycle>=0 &&
           splitQuality.first+1.e-6f>=localQualityFloor.first &&
           splitQuality.second+1.e-6f>=localQualityFloor.second &&
           splitReport.sizingErrorP95<=initialSizingP95+1.e-5f && endpointQualitySafe(splitMesh)) {
          local=std::move(splitMesh);
          result.report=splitReport;
          result.qualitySplit=true;
          result.qualitySplitRatio=splitOptions.splitQualityRatio;
          result.error.clear();
          improved=true;
        }
      }
    }
    // Low mean quality with a healthy lower tail and already-correct sizing
    // can sometimes be recovered by tangential/topology-only operations.
    // Avoid splitting/collapsing so this retry cannot drift from target size.
    if(!improved && initialOk && requireQualityImprovement &&
       localQualityFloor.first<.5f && localQualityFloor.second>.06f &&
       initialSizingP95<=1.05f) {
      SemanticMesh recoveryMesh=local;
      RemeshConfig recoveryConfig=config;
      recoveryConfig.enableSplit=false;
      recoveryConfig.enableCollapse=false;
      recoveryConfig.maxIterations=std::min(config.maxIterations,8);
      RawCudaOptions recoveryOptions=options;
      recoveryOptions.qualityMeanFloor=localQualityFloor.first*1.02f;
      recoveryOptions.qualityP05Floor=localQualityFloor.second*.90f;
      std::string recoveryError;
      recoveryOptions.error=&recoveryError;
      RemeshReport recoveryReport;
      phaseStart=Clock::now();
      const bool recoveryOk=remeshRawCuda(recoveryMesh,recoveryConfig,recoveryReport,recoveryOptions);
      result.secondsMeanRecovery=std::chrono::duration<double>(Clock::now()-phaseStart).count();
      if(recoveryOk) {
        const auto recoveryQuality=qualitySummary(recoveryMesh);
        if(recoveryReport.selectedCycle>=0 &&
           recoveryQuality.first+1.e-6f>=recoveryOptions.qualityMeanFloor &&
           recoveryQuality.second+1.e-6f>=recoveryOptions.qualityP05Floor &&
           recoveryReport.sizingErrorP95<=initialSizingP95+1.e-5f && endpointQualitySafe(recoveryMesh)) {
          local=std::move(recoveryMesh);
          result.report=recoveryReport;
          result.meanRecovery=true;
          result.error.clear();
          improved=true;
        }
      }
    }
    const bool geometryRetry=!initialOk && result.report.overlongEdges>0 &&
        result.error.find("final raw CUDA geometry or locked-vertex constraints failed")!=std::string::npos;
    if(!improved && requireQualityImprovement &&
       ((initialOk && result.report.maxEdgeLengthRatio>config.splitRatio*(1.f+1.e-4f)) || geometryRetry)) {
      // A shape-only gate can reject every useful cycle even when a region
      // still contains extreme long edges. Retry from the untouched source,
      // prioritize edge sizing, and allow only a bounded local quality tradeoff.
      result.retried=true;
      RemeshConfig sizeConfig=config;
      if(geometryRetry) {
        sizeConfig.constantLength=config.constantLength*.9f;
        if(sizeConfig.featureEdgeLength>0.f)sizeConfig.featureEdgeLength*=.9f;
      }
      SemanticMesh sizeMesh=initialLocal;
      RawCudaOptions sizeOptions=options;
      sizeOptions.optimizeSplitPoint=true;
      sizeOptions.collapsePasses=2;
      sizeOptions.splitQualityRatio=.20f;
      sizeOptions.qualityMeanFloor=localQualityFloor.first*.90f;
      sizeOptions.qualityP05Floor=localQualityFloor.second*.20f;
      std::string sizeError;
      sizeOptions.error=&sizeError;
      RemeshReport sizeReport;
      phaseStart=Clock::now();
      const bool sizeOk=remeshRawCuda(sizeMesh,sizeConfig,sizeReport,sizeOptions);
      result.secondsQualitySplit+=std::chrono::duration<double>(Clock::now()-phaseStart).count();
      const auto sizeQuality=sizeOk?qualitySummary(sizeMesh):std::pair<float,float>{0.f,0.f};
      const bool reducedLongEdges=geometryRetry ? sizeReport.splits>0 :
          (sizeReport.overlongEdges<int(result.report.overlongEdges*.8f) ||
           sizeReport.maxEdgeLengthRatio<result.report.maxEdgeLengthRatio*.8f);
      if(sizeOk && sizeReport.selectedCycle>=0 && reducedLongEdges &&
         sizeQuality.first+1.e-6f>=sizeOptions.qualityMeanFloor &&
         sizeQuality.second+1.e-6f>=sizeOptions.qualityP05Floor && endpointQualitySafe(sizeMesh)) {
        local=std::move(sizeMesh);
        result.report=sizeReport;
        result.sizeRecovery=true;
        result.error.clear();
        improved=true;
      } else if(!sizeError.empty()) {
        result.retryReason=sizeError;
      } else {
        result.retryReason="size-only retry rejected: selected="+std::to_string(sizeReport.selectedCycle)+
            " overlong="+std::to_string(sizeReport.overlongEdges)+"/"+
            std::to_string(result.report.overlongEdges)+" max_ratio="+
            std::to_string(sizeReport.maxEdgeLengthRatio)+"/"+
            std::to_string(result.report.maxEdgeLengthRatio)+" quality="+
            std::to_string(sizeQuality.first)+","+std::to_string(sizeQuality.second);
      }
    }
    if(!improved && initialOk && requireQualityImprovement &&
       result.report.maxEdgeLengthRatio>config.splitRatio*(1.f+1.e-4f)) {
      // If the permissive size retry would make a severe lower-tail sliver,
      // try again with stronger per-child quality protection. This often
      // handles a few isolated giant edges without changing the rest of a patch.
      RemeshConfig strictSizeConfig=config;
      SemanticMesh strictSizeMesh=initialLocal;
      RawCudaOptions strictSizeOptions=options;
      strictSizeOptions.optimizeSplitPoint=true;
      strictSizeOptions.splitQualityRatio=.60f;
      strictSizeOptions.collapsePasses=2;
      strictSizeOptions.qualityMeanFloor=localQualityFloor.first*.75f;
      strictSizeOptions.qualityP05Floor=localQualityFloor.second*.10f;
      std::string strictSizeError;
      strictSizeOptions.error=&strictSizeError;
      RemeshReport strictSizeReport;
      phaseStart=Clock::now();
      const bool strictSizeOk=remeshRawCuda(strictSizeMesh,strictSizeConfig,strictSizeReport,strictSizeOptions);
      result.secondsQualitySplit+=std::chrono::duration<double>(Clock::now()-phaseStart).count();
      const auto strictQuality=strictSizeOk?qualitySummary(strictSizeMesh):std::pair<float,float>{0.f,0.f};
      const bool strictSizeReduced=strictSizeReport.overlongEdges<int(result.report.overlongEdges*.8f) ||
          strictSizeReport.maxEdgeLengthRatio<result.report.maxEdgeLengthRatio*.8f;
      if(strictSizeOk && strictSizeReport.selectedCycle>=0 && strictSizeReduced &&
         strictQuality.first+1.e-6f>=strictSizeOptions.qualityMeanFloor &&
         strictQuality.second+1.e-6f>=strictSizeOptions.qualityP05Floor && endpointQualitySafe(strictSizeMesh)) {
        local=std::move(strictSizeMesh);
        result.report=strictSizeReport;
        result.sizeRecovery=true;
        result.error.clear();
        improved=true;
      } else if(!strictSizeError.empty()) {
        result.retryReason=strictSizeError;
      }
    }
    if(config.enableCollapse && initialOk && result.report.maxEdgeLengthRatio>16.f) {
      // For genuinely extreme slivers, run a collapse-first local pass before
      // the conforming midpoint fallback. It is scoped to this packed task,
      // preserves frozen patch boundaries, and accepts quality loss only if
      // the resulting local sizing actually improves.
      RemeshConfig collapseConfig=config;
      collapseConfig.enableSplit=false;
      collapseConfig.enableCollapse=true;
      collapseConfig.enableSmooth=false;
      collapseConfig.maxIterations=3;
      collapseConfig.collapseRatio=1.15f;
      SemanticMesh collapseMesh=local;
      RawCudaOptions collapseOptions=options;
      collapseOptions.allowQualityTradeoff=true;
      collapseOptions.qualityMeanFloor=0.f;
      collapseOptions.qualityP05Floor=0.f;
      collapseOptions.collapsePasses=8;
      collapseOptions.flipPasses=2;
      collapseOptions.strictFlipQuality=false;
      std::string collapseError;
      collapseOptions.error=&collapseError;
      RemeshReport collapseReport;
      auto sizing=[&](const SemanticMesh &mesh) {
        int overlong=0;float maxRatio=0.f;
        for(const auto &edge:mesh.edges) {
          const float h=.5f*(mesh.targetLength[edge.v0]+mesh.targetLength[edge.v1]);
          if(!(h>0.f))continue;
          const float ratio=distance(mesh.position(int(edge.v0)),mesh.position(int(edge.v1)))/h;
          maxRatio=std::max(maxRatio,ratio);
          if(ratio>config.splitRatio*(1.f+1.e-4f))++overlong;
        }
        return std::pair<int,float>{overlong,maxRatio};
      };
      const auto beforeCollapse=sizing(local);
      const auto qualityBeforeCollapse=qualitySummary(local);
      phaseStart=Clock::now();
      const bool collapseOk=remeshRawCuda(collapseMesh,collapseConfig,collapseReport,collapseOptions);
      result.secondsQualitySplit+=std::chrono::duration<double>(Clock::now()-phaseStart).count();
      const auto afterCollapse=collapseOk?sizing(collapseMesh):beforeCollapse;
      const auto qualityAfterCollapse=collapseOk?qualitySummary(collapseMesh):qualityBeforeCollapse;
      if(collapseOk && collapseReport.collapses>0 &&
         (afterCollapse.first<beforeCollapse.first || afterCollapse.second<beforeCollapse.second-1.e-4f ||
          qualityAfterCollapse.first>qualityBeforeCollapse.first+1.e-5f ||
          qualityAfterCollapse.second>qualityBeforeCollapse.second+1.e-5f) && endpointQualitySafe(collapseMesh)) {
        local=std::move(collapseMesh);
        result.report.collapses+=collapseReport.collapses;
        result.report.flips+=collapseReport.flips;
        result.report.smoothMoves+=collapseReport.smoothMoves;
        result.report.overlongEdges=afterCollapse.first;
        result.report.maxEdgeLengthRatio=afterCollapse.second;
        result.sizeRecovery=true;
        result.error.clear();
        improved=true;
      }
    }
    if(!legacyCoverageAcceptance && !improved && provisionalValid) {
      local=std::move(provisionalMesh);
      result.report=provisionalReport;
      result.provisional=true;
      result.error.clear();
      improved=true; // Assembly eligibility only; qualityAccepted is separate.
    }
    if(!improved && initialOk &&
       initialQuality.first+1.e-6f>=localQualityFloor.first &&
       initialQuality.second+1.e-6f>=localQualityFloor.second) {
      // The input itself passed the quality and geometry gates. Keep this
      // outcome visible as unchanged rather than calling it an improvement.
      result.unchanged=true;
      result.error.clear();
      improved=true;
    }
    if(!improved)throw std::runtime_error(result.error.empty()?
        "no GPU cycle met the local quality gate":result.error);
    std::string topologyError;
    if(!local.validate(&topologyError))throw std::runtime_error("task topology: "+topologyError);
    job.aliases.assign(local.vertexCount(),-1);
    std::set<int> retained;
    for(int v=0;v<local.vertexCount();++v) {
      if(local.vertexConstraint[v]!=uint8_t(VertexConstraint::Locked))continue;
      auto found=anchors.find(positionKey(local.position(v)));
      if(found==anchors.end() || !retained.insert(found->second).second)
        throw std::runtime_error("boundary identity changed or duplicated");
      job.aliases[v]=found->second;
    }
    if(retained.size()!=anchors.size())throw std::runtime_error("missing locked vertex");
    for(auto e:local.edges) if(e.flags&EdgeMeshBoundary) {
      int a=job.aliases[e.v0],b=job.aliases[e.v1];
      if(a<0 || b<0 || !required.erase(edgeKey(a,b)))throw std::runtime_error("unexpected patch boundary");
    }
    if(!required.empty())throw std::runtime_error("missing patch boundary edge");
    result.qualityAccepted=qualityEndpointMask(sourceRegionQuality,evaluateRegionQuality(local,config,lowQualityThreshold))==0;
    result.provisional=!result.qualityAccepted;
    if(!legacyCoverageAcceptance && requireQualityImprovement)result.unchanged=!meshChanged(originalLocal,local);
    result.outputFaces=local.faceCount();result.accepted=true;job.succeeded=true;
    if(result.unchanged) {
      for(int patch=job.firstPatch;patch<job.firstPatch+job.patchCount;++patch)
        result.unchangedPatchIds.push_back(uint32_t(patch));
    }
    job.unchanged=result.unchanged;job.uniformSizing=result.uniformSizing;
    job.qualitySplit=result.qualitySplit;
    job.meanRecovery=result.meanRecovery;
    job.sizeRecovery=result.sizeRecovery;
    job.output=std::move(local);
  } catch(const std::exception &e) {result.error=e.what();result.outputFaces=result.inputFaces;}
  const bool compareChildren=result.accepted && result.provisional && job.patchCount>1 &&
      exploreProvisionalChildren && !legacyCoverageAcceptance;
  Job incumbent;
  RawPatchResult incumbentResult;
  std::vector<RemeshReport> comparisonChildReports;
  std::vector<RegionQuality> comparisonIncumbentPatches;
  std::vector<uint8_t> comparisonGenerated;
  const auto comparisonStart=Clock::now();
  if(compareChildren) {
    incumbent=job;incumbentResult=result;comparisonChildReports.resize(job.patchCount);
    comparisonIncumbentPatches=evaluatePatchQuality(incumbent.output,config,lowQualityThreshold);
    comparisonGenerated.resize(job.patchCount,0);
  }
  if((!result.accepted || compareChildren) && job.patchCount>1) {
    result.retried=true;
    result.retryReason=result.error;
    job.children.resize(job.patchCount);
    bool allAccepted=true;
    bool allQualityAccepted=true;
    int outputFaces=0;
    bool anySizeRecovery=false;
    std::string failures;
    RemeshReport combined;
    for(int k=0;k<job.patchCount;++k) {
      auto &child=job.children[k];
      child.firstPatch=job.firstPatch+k;child.patchCount=1;child.bytes=job.bytes;
      for(int f:job.faces)if(source.facePatchId[f]==uint32_t(child.firstPatch))child.faces.push_back(f);
      RawPatchResult childResult;
      const bool healthy=compareChildren && comparisonIncumbentPatches[k].lowQualityFaces==0 &&
          qualityEndpointMask(originalPatchQuality[k],comparisonIncumbentPatches[k])==0;
      if(healthy) {
        auto retained=extractIncumbentPatch(incumbent,k);retained.faces=child.faces;child=std::move(retained);
        childResult.accepted=true;childResult.qualityAccepted=true;
        childResult.outputFaces=child.output.faceCount();
      } else {
        execute(child,source,config,stopWhenIdle,smoothPasses,collapsePasses,flipPasses,
                strictFlipQuality,requireQualityImprovement,legacyCoverageAcceptance,false,trackRegionCandidates,lowQualityThreshold,childResult);
        if(compareChildren)comparisonGenerated[k]=1;
      }
      if(compareChildren)comparisonChildReports[k]=childResult.report;
      allAccepted &= childResult.accepted;
      allQualityAccepted &= childResult.qualityAccepted;
      outputFaces+=childResult.outputFaces;
      anySizeRecovery|=childResult.sizeRecovery;
      result.unchangedPatchIds.insert(result.unchangedPatchIds.end(),
          childResult.unchangedPatchIds.begin(),childResult.unchangedPatchIds.end());
      if(!childResult.accepted)failures+=" patch "+std::to_string(child.firstPatch)+": "+childResult.error;
      const auto &r=childResult.report;
      combined.cyclesExecuted=std::max(combined.cyclesExecuted,r.cyclesExecuted);
      combined.recoveredCycleFailures+=r.recoveredCycleFailures;
      combined.splits+=r.splits;combined.collapses+=r.collapses;
      combined.flips+=r.flips;combined.smoothMoves+=r.smoothMoves;
      combined.geometryErrorMax=std::max(combined.geometryErrorMax,r.geometryErrorMax);
      combined.geometryErrorReverseMax=std::max(combined.geometryErrorReverseMax,r.geometryErrorReverseMax);
      combined.maxEdgeLengthRatio=std::max(combined.maxEdgeLengthRatio,r.maxEdgeLengthRatio);
      combined.overlongEdges+=r.overlongEdges;
      combined.secondsSetup+=r.secondsSetup;combined.secondsSplit+=r.secondsSplit;
      combined.secondsCollapse+=r.secondsCollapse;combined.secondsCompact+=r.secondsCompact;
      combined.secondsValidate+=r.secondsValidate;combined.secondsFlip+=r.secondsFlip;
      combined.secondsSmooth+=r.secondsSmooth;
      for(int pass=0;pass<8;++pass) {
        combined.collapsePassCalls[pass]+=r.collapsePassCalls[pass];
        combined.collapsePassAccepted[pass]+=r.collapsePassAccepted[pass];
        combined.collapsePassSeconds[pass]+=r.collapsePassSeconds[pass];
        combined.flipPassCalls[pass]+=r.flipPassCalls[pass];
        combined.flipPassAccepted[pass]+=r.flipPassAccepted[pass];
        combined.flipPassSeconds[pass]+=r.flipPassSeconds[pass];
      }
    }
    result.accepted=allAccepted;
    result.qualityAccepted=allAccepted && allQualityAccepted;
    result.provisional=allAccepted && !allQualityAccepted;
    std::sort(result.unchangedPatchIds.begin(),result.unchangedPatchIds.end());
    result.unchangedPatchIds.erase(std::unique(result.unchangedPatchIds.begin(),
        result.unchangedPatchIds.end()),result.unchangedPatchIds.end());
    result.unchanged=allAccepted &&
        result.unchangedPatchIds.size()==size_t(job.patchCount);
    result.outputFaces=outputFaces;
    result.error=allAccepted?std::string{}:failures;
    result.report=combined;
    result.sizeRecovery=anySizeRecovery;
  }
  if(compareChildren) {
    bool selected=false;
    int selectedPatches=0;
    int boundaryTargetChanges=0;
    std::vector<int> chosenPatches;
    std::string comparisonReason="child_geometry_or_boundary_failure";
    bool evaluated=false;
    RegionQuality comparisonBefore,comparisonAfter;
    float comparisonSizingMean=0,comparisonSizingP95=0;
    // Construct a bounded candidate without changing the live assembly aliases.
    // Only immutable source anchor IDs are welded; coincident interior points
    // are kept distinct. Each child already passed geometry and boundary checks.
    try { if(job.children.size()==size_t(incumbent.patchCount)) {
      // Select ownership regions independently, then check their connected
      // union. Failed exploration retains that region's legal incumbent.
      const auto &incumbentPatches=comparisonIncumbentPatches;
      for(int p=0;p<incumbent.patchCount;++p) {
        auto &child=job.children[p];bool choose=false;
        if(child.succeeded && comparisonGenerated[p]) {
          const int targetChanges=boundaryTargetMismatchCount(incumbent.output,incumbent.aliases,child.output,child.aliases);
          boundaryTargetChanges+=targetChanges;
          const auto after=evaluateRegionQuality(child.output,config,lowQualityThreshold);
          const auto &before=incumbentPatches[p];
          choose=targetChanges==0 && qualityEndpointMask(before,after)==0 &&
              after.longEdges<=before.longEdges && after.shortEdges<=before.shortEdges;
        }
        if(choose) {++selectedPatches;chosenPatches.push_back(p);}
        else {
          auto retained=extractIncumbentPatch(incumbent,p);
          retained.faces=child.faces;child=std::move(retained);
        }
      }
      SemanticMesh candidate;candidate.patches=incumbent.output.patches;
      candidate.LocalSizing=incumbent.output.LocalSizing;
      std::map<int,int> anchors;
      for(const auto &child:job.children) {
        std::vector<int> ids(child.output.vertexCount());
        for(int v=0;v<child.output.vertexCount();++v) {
          const int anchor=child.aliases[v];
          auto found=anchor>=0?anchors.find(anchor):anchors.end();
          if(found!=anchors.end())ids[v]=found->second;
          else {
            ids[v]=candidate.addVertex(child.output.position(v),
                child.firstPatch-job.firstPatch+child.output.vertexPatchId[v],
                VertexConstraint(child.output.vertexConstraint[v]));
            if(anchor>=0)anchors.emplace(anchor,ids[v]);
          }
          float &target=candidate.targetLength[ids[v]];
          const float childTarget=child.output.targetLength[v];
          target=target>0?std::min(target,childTarget):childTarget;
        }
        for(int f=0;f<child.output.faceCount();++f)if(child.output.faceAlive[f]) {
          auto t=child.output.face(f);
          const auto patch=child.firstPatch-job.firstPatch+child.output.facePatchId[f];
          candidate.addFace(ids[t[0]],ids[t[1]],ids[t[2]],patch,candidate.patches[patch].type);
        }
        for(auto feature:child.output.featureEdges)
          candidate.featureEdges[edgeKey(ids[int(feature.first>>32)],ids[int(uint32_t(feature.first))])]=feature.second;
      }
      candidate.rebuildTopology();
      if(candidate.LocalSizing)candidate.LocalSizing->apply(candidate,true);
      std::string why;
      if(candidate.validate(&why)) {
        const auto before=evaluateRegionQuality(incumbent.output,config,lowQualityThreshold);
        const auto after=evaluateRegionQuality(candidate,config,lowQualityThreshold);
        comparisonBefore=before;comparisonAfter=after;evaluated=true;
        std::vector<float> sizing;
        double sizingSum=0;
        for(const auto &edge:candidate.edges) {
          const float h=.5f*(candidate.targetLength[edge.v0]+candidate.targetLength[edge.v1]);
          if(!(h>0))continue;
          const float error=std::abs(distance(candidate.position(edge.v0),candidate.position(edge.v1))/h-1.f);
          sizing.push_back(error);sizingSum+=error;
        }
        if(!sizing.empty()) {
          comparisonSizingMean=float(sizingSum/sizing.size());
          const size_t p95=sizing.size()*95/100;
          std::nth_element(sizing.begin(),sizing.begin()+p95,sizing.end());comparisonSizingP95=sizing[p95];
        }
        selected=selectedPatches>0 && qualityEndpointMask(before,after)==0 &&
            after.longEdges<=before.longEdges && after.shortEdges<=before.shortEdges;
        comparisonReason="global_quality_mask_"+std::to_string(qualityEndpointMask(before,after));
        if(after.longEdges>before.longEdges || after.shortEdges>before.shortEdges)
          comparisonReason+="_size_regression";
        const auto beforePatches=evaluatePatchQuality(incumbent.output,config,lowQualityThreshold);
        const auto afterPatches=evaluatePatchQuality(candidate,config,lowQualityThreshold);
        for(size_t p=0;selected && p<beforePatches.size();++p) {
          selected=qualityNonRegression(beforePatches[p],afterPatches[p]) &&
              afterPatches[p].longEdges<=beforePatches[p].longEdges &&
              afterPatches[p].shortEdges<=beforePatches[p].shortEdges;
          if(!selected)comparisonReason="patch_"+std::to_string(job.firstPatch+p)+
              "_quality_mask_"+std::to_string(qualityRegressionMask(beforePatches[p],afterPatches[p]))+"_or_size_regression";
        }
        if(selected)comparisonReason="selected";
      } else comparisonReason="child_topology_failure";
    }} catch(const std::exception &) {selected=false;comparisonReason="child_evaluation_failure";}
    // A failed or regressing exploration restores the legal incumbent, never
    // the poor original input. Endpoint success remains a separate report.
    if(!selected) {job=std::move(incumbent);result=std::move(incumbentResult);}
    else {
      // Endpoint status is decided by the assembled regional audit. Attempted
      // operation counts do not certify the selected mixed geometry.
      result=incumbentResult;result.accepted=true;result.qualityAccepted=false;result.provisional=true;
      result.outputFaces=0;result.unchangedPatchIds.clear();
      for(const auto &child:job.children) {
        result.outputFaces+=child.output.faceCount();
        if(child.unchanged)result.unchangedPatchIds.push_back(uint32_t(child.firstPatch));
      }
      result.unchanged=result.unchangedPatchIds.size()==size_t(job.patchCount);
      result.retried=true;
      result.report.qualityMean=comparisonAfter.mean;result.report.qualityP05=comparisonAfter.p05;
      result.report.qualityMin=comparisonAfter.minimum;
      result.report.overlongEdges=comparisonAfter.longEdges;
      result.report.maxEdgeLengthRatio=comparisonAfter.edgeRatioMax;
      result.report.sizingErrorMean=comparisonSizingMean;result.report.sizingErrorP95=comparisonSizingP95;
      for(int p:chosenPatches) {
        result.report.splits+=comparisonChildReports[p].splits;
        result.report.collapses+=comparisonChildReports[p].collapses;
        result.report.flips+=comparisonChildReports[p].flips;
        result.report.smoothMoves+=comparisonChildReports[p].smoothMoves;
        result.report.geometryErrorMax=std::max(result.report.geometryErrorMax,comparisonChildReports[p].geometryErrorMax);
        result.report.geometryErrorReverseMax=std::max(result.report.geometryErrorReverseMax,comparisonChildReports[p].geometryErrorReverseMax);
      }
    }
    result.childCandidateCompared=true;result.childCandidateSelected=selected;
    result.childComparisonReason=comparisonReason;
    result.childCandidateEvaluated=evaluated;
    result.childGeneratedPatches=int(std::count(comparisonGenerated.begin(),comparisonGenerated.end(),uint8_t(1)));
    result.childBoundaryTargetChanges=boundaryTargetChanges;
    result.childSelectedPatches=selected?selectedPatches:0;
    result.childIncumbentQuality=std::move(comparisonBefore);
    result.childCandidateQuality=std::move(comparisonAfter);
    result.secondsChildComparison=std::chrono::duration<double>(Clock::now()-comparisonStart).count();
  }
  result.seconds=std::chrono::duration<double>(Clock::now()-start).count();
  result.report.seconds=result.seconds;
}
}

bool remeshRawCudaPatches(const SemanticMesh &input,SemanticMesh &output,const RemeshConfig &cfg,
                         const RawBatchOptions &options,RawBatchReport &report,std::string *error) {
  report={};const auto start=Clock::now();
  try {
    if(options.workers<1 || options.workers>32 || options.gpuConcurrency<1 || options.gpuConcurrency>8 || options.patchesPerTask<0 || options.patchesPerTask>32 || options.smoothPasses<1 || options.smoothPasses>12 || options.collapsePasses<1 || options.collapsePasses>8 || options.flipPasses<1 || options.flipPasses>8 || !(cfg.constantLength>0) || !std::isfinite(cfg.constantLength) || !(cfg.maxGeometryError>0) || !std::isfinite(cfg.maxGeometryError) || cfg.maxIterations<1 || cfg.adaptive)
      throw std::runtime_error("invalid batch worker count or sizing configuration");
    if(options.selectFinalRegions && options.legacyCoverageAcceptance)
      throw std::runtime_error("regional refinement selection requires quality acceptance");
    SemanticMesh source=input;source.rebuildTopology();
    if(!std::isfinite(options.lowQualityThreshold) || options.lowQualityThreshold<0 || options.lowQualityThreshold>1)
      throw std::runtime_error("low quality threshold must be in [0,1]");
    const float lowQualityThreshold=options.lowQualityThreshold>0?options.lowQualityThreshold:qualitySummary(source).second;
    report.sourceQuality=evaluateRegionQuality(source,cfg,lowQualityThreshold);
    if(options.autoPartitionSinglePatch)AutoPartitionSinglePatch(source,options.workers);
    const auto sourceQualityAuditStart=Clock::now();
    report.sourcePatchQuality=evaluatePatchQuality(source,cfg,lowQualityThreshold);
    report.secondsQualityAudit=std::chrono::duration<double>(Clock::now()-sourceQualityAuditStart).count();
    const bool semanticPatches=std::any_of(source.patches.begin(),source.patches.end(),
        [](const PatchRecord &patch){return patch.type!=PatchType::Unknown;});
    const int patchesPerTask=options.patchesPerTask?options.patchesPerTask:
        (semanticPatches && source.patches.size()>=128?16:1);
    report.patchesPerTask=patchesPerTask;
    std::string problem;if(!source.validate(&problem))throw std::runtime_error(problem);
    for(auto e:source.edges)if(e.flags&(EdgePatchBoundary|EdgeMeshBoundary)) {
      source.vertexConstraint[e.v0]=source.vertexConstraint[e.v1]=uint8_t(VertexConstraint::Locked);
    }
    // Both incident patches receive exactly the same new vertices. This runs
    // once before creating jobs, never independently inside a patch worker.
    const auto boundaryStart=Clock::now();
    report.boundarySplits=refinePartitionBoundary(source,cfg,&report.boundaryEdgesDeferred);
    if(report.boundaryEdgesDeferred>0) {
      // A partially subdivided boundary can create zero-area children on
      // extremely short or skinny source triangles. Roll back the whole
      // global subdivision stage to the known-valid input, then let the
      // patch workers operate with the original shared edges locked.
      source=input;
      source.rebuildTopology();
      for(const auto &edge:source.edges) {
        if(edge.flags&(EdgePatchBoundary|EdgeMeshBoundary))
          source.vertexConstraint[edge.v0]=source.vertexConstraint[edge.v1]=uint8_t(VertexConstraint::Locked);
      }
      report.boundarySplits=0;
      report.boundaryEdgesDeferred=0;
      for(const auto &edge:source.edges) {
        if(!(edge.flags&(EdgePatchBoundary|EdgeMeshBoundary|EdgeSharp)))continue;
        const Vec3 a=source.position(edge.v0),b=source.position(edge.v1),mid=(a+b)*0.5f;
        float target=cfg.constantLength;
        if((edge.flags&EdgeSharp) && cfg.featureEdgeLength>0.f)
          target=std::min(target,cfg.featureEdgeLength);
        for(uint32_t patch:{edge.patchLeft,edge.patchRight}) if(source.LocalSizing && patch<source.patches.size()) {
          target=std::min(target,source.LocalSizing->evaluate(patch,a));
          target=std::min(target,source.LocalSizing->evaluate(patch,b));
          target=std::min(target,source.LocalSizing->evaluate(patch,mid));
        }
        if(target>0.f && distance(a,b)>cfg.splitRatio*target*(1.f+1.e-5f))
          ++report.boundaryEdgesDeferred;
      }
    }
    report.secondsBoundary=std::chrono::duration<double>(Clock::now()-boundaryStart).count();
    if(!source.validate(&problem))throw std::runtime_error("refined source: "+problem);
    if(report.boundaryEdgesDeferred>0) {
      // A boundary that exceeds the safety budget signals that local sizing
      // would require an unusually large job. Preserve the valid input as a
      // reviewable candidate instead of launching a GPU batch with unbounded
      // output growth. The CLI reports this as a partial result (exit 3).
      output=source;
      report.outputQuality=evaluateRegionQuality(output,cfg,lowQualityThreshold);
      report.outputPatchQuality=report.sourcePatchQuality;
      report.topologyValid=true;
      report.boundariesHeld=true;
      report.seconds=std::chrono::duration<double>(Clock::now()-start).count();
      return true;
    }
    std::vector<Job> jobs((source.patches.size()+patchesPerTask-1)/patchesPerTask);
    for(size_t i=0;i<jobs.size();++i) {
      jobs[i].id=int(i);
      jobs[i].firstPatch=int(i)*patchesPerTask;
      jobs[i].patchCount=std::min(patchesPerTask,int(source.patches.size())-jobs[i].firstPatch);
    }
    for(int f=0;f<source.faceCount();++f) {
      if(source.facePatchId[f]>=source.patches.size())throw std::runtime_error("invalid patch id");
      auto &job=jobs[source.facePatchId[f]/patchesPerTask];job.faces.push_back(f);
      job.cost+=.5*length(cross(source.facePoint(f,1)-source.facePoint(f,0),source.facePoint(f,2)-source.facePoint(f,0)));
    }
    int device=0;cudaCheck(cudaGetDevice(&device));cudaCheck(cudaFree(nullptr));
    size_t freeBytes=0,totalBytes=0;cudaCheck(cudaMemGetInfo(&freeBytes,&totalBytes));
    report.memoryBudget=options.memoryBytes?std::min(options.memoryBytes,size_t(freeBytes*.8)):size_t(freeBytes*.6);
    if(report.memoryBudget<16*1024*1024)throw std::runtime_error("insufficient CUDA task memory budget");
    const double h=cfg.featureEdgeLength>0?cfg.featureEdgeLength:cfg.constantLength;
    for(auto &job:jobs) {
      const double estimatedFaces=std::max(4.*job.faces.size(),4.*job.cost/(.433*h*h));
      job.cost=estimatedFaces;
      // Conservative capacity estimate, not a promise: DeviceArena enforces
      // this task's actual allocation cap and failures keep original geometry.
      job.bytes=size_t(std::min(double(report.memoryBudget),64.*1024*1024+estimatedFaces*1024));
    }
    std::vector<int> pending(jobs.size());std::iota(pending.begin(),pending.end(),0);
    std::stable_sort(pending.begin(),pending.end(),[&](int a,int b){return jobs[a].cost>jobs[b].cost;});
    report.patches.resize(jobs.size());
    std::mutex mutex;std::condition_variable changed;size_t reserved=0;int active=0;
    const int count=std::min({options.workers,options.gpuConcurrency,int(jobs.size())});
    CadMesh::RemeshWorkerPool pool(count);std::vector<std::future<void>> workers;
    for(int w=0;w<count;++w)workers.push_back(pool.submit([&]{
      cudaCheck(cudaSetDevice(device));
      for(;;) {
        int id=-1;
        {
          std::unique_lock<std::mutex> lock(mutex);
          changed.wait(lock,[&]{
            if(pending.empty())return true;
            for(int candidate:pending)if(jobs[candidate].bytes<=report.memoryBudget-reserved)return true;
            return false;
          });
          if(pending.empty())return;
          auto it=std::find_if(pending.begin(),pending.end(),[&](int candidate){return jobs[candidate].bytes<=report.memoryBudget-reserved;});
          id=*it;pending.erase(it);reserved+=jobs[id].bytes;++active;
          report.peakActive=std::max(report.peakActive,active);report.peakReserved=std::max(report.peakReserved,reserved);
        }
        execute(jobs[id],source,cfg,options.stopWhenIdle,options.smoothPasses,options.collapsePasses,options.flipPasses,options.strictFlipQuality,
                options.requireQualityImprovement,options.legacyCoverageAcceptance,options.exploreProvisionalChildren,
                options.trackRegionCandidates,lowQualityThreshold,report.patches[id]);
        {std::lock_guard<std::mutex> lock(mutex);reserved-=jobs[id].bytes;--active;}
        changed.notify_all();
      }
    }));
    for(auto &worker:workers)worker.get();
    report.secondsTasks=std::chrono::duration<double>(Clock::now()-boundaryStart).count()-report.secondsBoundary;
    for(const auto &task:report.patches)report.retried+=int(task.retried);
    // Deterministic assembly, independent of which worker finishes first.
    SemanticMesh merged;
    std::vector<Job*> faceOwner;
    std::function<void(Job&)> appendJob=[&](Job &job) {
      if(!job.children.empty()) {
        for(auto &child:job.children)appendJob(child);
        return;
      }
      if(!job.succeeded) {
        ++report.fallback;
        for(int f:job.faces){auto t=source.face(f);const auto patch=source.facePatchId[f];merged.addFace(t[0],t[1],t[2],patch,source.patches[patch].type);faceOwner.push_back(nullptr);}
        return;
      }
      ++report.accepted;
      report.unchanged+=int(job.unchanged);
      report.uniformRegions+=int(job.uniformSizing);
      report.qualitySplitRegions+=int(job.qualitySplit);
      report.meanRecoveryRegions+=int(job.meanRecovery);
      for(int &alias:job.aliases)if(alias>=source.vertexCount())alias=-1;
      for(int v=0;v<job.output.vertexCount();++v)if(job.aliases[v]<0)
        job.aliases[v]=merged.addVertex(job.output.position(v),job.firstPatch+job.output.vertexPatchId[v],VertexConstraint(job.output.vertexConstraint[v]));
      for(int v=0;v<job.output.vertexCount();++v) {
        float &h=merged.targetLength[job.aliases[v]];
        h=h>0?std::min(h,job.output.targetLength[v]):job.output.targetLength[v];
      }
      for(int f=0;f<job.output.faceCount();++f)if(job.output.faceAlive[f]){auto t=job.output.face(f);const auto patch=job.firstPatch+job.output.facePatchId[f];merged.addFace(job.aliases[t[0]],job.aliases[t[1]],job.aliases[t[2]],patch,source.patches[patch].type);faceOwner.push_back(&job);}
      for(auto feature:job.output.featureEdges)merged.featureEdges[edgeKey(job.aliases[int(feature.first>>32)],job.aliases[int(uint32_t(feature.first))])]=feature.second;
    };
    for(int repair=0;repair<=int(jobs.size());++repair) {
      merged=SemanticMesh{};merged.patches=source.patches;merged.featureEdges=source.featureEdges;
      merged.LocalSizing=source.LocalSizing;
      for(int v=0;v<source.vertexCount();++v)
        merged.addVertex(source.position(v),source.vertexPatchId[v],VertexConstraint(source.vertexConstraint[v]));
      for(auto &h:merged.targetLength)h=cfg.constantLength;
      faceOwner.clear();report.accepted=report.unchanged=report.uniformRegions=
          report.qualitySplitRegions=report.meanRecoveryRegions=report.fallback=0;
      for(auto &job:jobs)appendJob(job);
      merged.rebuildTopology();
      if(merged.LocalSizing)merged.LocalSizing->apply(merged,true);
      std::set<uint64_t> outputSeams;
      for(auto e:merged.edges)if(e.flags&(EdgePatchBoundary|EdgeMeshBoundary))outputSeams.insert(edgeKey(e.v0,e.v1));
      for(auto e:source.edges)if(e.flags&(EdgePatchBoundary|EdgeMeshBoundary))
        if(!outputSeams.erase(edgeKey(e.v0,e.v1)))throw std::runtime_error("global shared boundary missing");
      if(!outputSeams.empty())throw std::runtime_error("unexpected global seam or hole");
      if(merged.validate(&problem))break;
      uint32_t a=0,b=0;
      if(std::sscanf(problem.c_str(),"non-manifold edge %u,%u",&a,&b)!=2)
        throw std::runtime_error("assembled before compaction: "+problem);
      std::set<Job*> owners;
      for(int f=0;f<merged.faceCount();++f) {
        const auto t=merged.face(f);
        if(faceOwner[f] && std::find(t.begin(),t.end(),int(a))!=t.end() &&
                           std::find(t.begin(),t.end(),int(b))!=t.end())owners.insert(faceOwner[f]);
      }
      std::vector<Job*> repairCandidates;
      for(Job *candidate:owners) {
        bool sourceEdge=false;
        for(int f:candidate->faces) {
          const auto t=source.face(f);
          if(std::find(t.begin(),t.end(),int(a))!=t.end() &&
             std::find(t.begin(),t.end(),int(b))!=t.end()) {sourceEdge=true;break;}
        }
        if(!sourceEdge)repairCandidates.push_back(candidate);
      }
      if(repairCandidates.empty() || repair==int(jobs.size()))
        throw std::runtime_error("assembled before compaction: "+problem);
      std::sort(repairCandidates.begin(),repairCandidates.end(),
                [](const Job *x,const Job *y){return x->faces.size()<y->faces.size();});
      std::string repairReason;
      bool split=false;
      for(Job *candidate:repairCandidates)
        if(splitConflictingChord(*candidate,int(a),int(b),repairReason)){split=true;break;}
      if(split) {
        ++report.seamRepairs;
        ++report.seamSplitRepairs;
        continue;
      }
      Job *fallbackJob=repairCandidates.front();
      auto &taskReport=report.patches[fallbackJob->firstPatch/patchesPerTask];
      taskReport.accepted=false;
      if(!taskReport.error.empty())taskReport.error+="; ";
      taskReport.error+="assembly seam conflict at "+std::to_string(a)+","+std::to_string(b)+"; split rejected: "+repairReason;
      fallbackJob->succeeded=false;
      fallbackJob->output=SemanticMesh{};
      ++report.seamRepairs;
    }
    std::unordered_set<uint64_t> outputEdges;
    outputEdges.reserve(merged.edges.size());
    for(const auto &e:merged.edges)outputEdges.insert(edgeKey(e.v0,e.v1));
    for(const auto &feature:source.featureEdges)
      if(!outputEdges.count(feature.first) &&
         !featureSegmentCovered(source,merged,feature.first,feature.second))
        throw std::runtime_error("hard feature edge missing after assembly");
    std::unordered_set<uint32_t> coarsePatches,coveragePatches;
    for(size_t task=0;task<report.patches.size();++task) {
      const auto &patch=report.patches[task];
      // Include every patch returned unchanged by the local GPU pass. Some
      // have acceptable edge lengths but were still never modified because
      // no cycle passed the old quality gate.
      if(options.legacyCoverageAcceptance) {
        coarsePatches.insert(patch.unchangedPatchIds.begin(),patch.unchangedPatchIds.end());
        coveragePatches.insert(patch.unchangedPatchIds.begin(),patch.unchangedPatchIds.end());
      }
    }
    // Assembly combines endpoint targets using their minimum. Its defects can
    // therefore be absent from every local task report. Select actual incident
    // patches, rather than whole packed tasks selected by stale local ratios.
    const auto assembledCoarse=longEdgePatches(merged,cfg);
    coarsePatches.insert(assembledCoarse.begin(),assembledCoarse.end());
    if(!coarsePatches.empty()) {
      auto selectedSize=[&](const SemanticMesh &candidate) {
        int count=0;float maxRatio=0.f;double excessSquared=0;
        for(const auto &edge:candidate.edges) {
          const bool selected=(edge.face0>=0&&coarsePatches.count(candidate.facePatchId[edge.face0])) ||
                              (edge.face1>=0&&coarsePatches.count(candidate.facePatchId[edge.face1]));
          if(!selected)continue;
          const float h=.5f*(candidate.targetLength[edge.v0]+candidate.targetLength[edge.v1]);
          if(!(h>0.f))continue;
          const float ratio=distance(candidate.position(int(edge.v0)),candidate.position(int(edge.v1)))/h;
          maxRatio=std::max(maxRatio,ratio);
          if(ratio>cfg.splitRatio*(1.f+1.e-4f)) {
            ++count;const double excess=double(ratio)-cfg.splitRatio;
            excessSquared+=excess*excess;
          }
        }
        return SizeDefectSummary{count,maxRatio,excessSquared};
      };
      GeometryProjector refineProjector;
      SemanticMesh refinementBase;
      if(options.selectFinalRegions)refinementBase=merged;
      auto refineCandidate=[&](bool feasibleSplits) {
      auto before=selectedSize(merged);
      // Evaluate and commit one subdivision level at a time. The old code
      // built up to eight levels and rejected the entire result if the final
      // tail of refinement crossed a quality floor, discarding even useful
      // early levels and leaving the original oversized patches untouched.
      TriangleRefineStats refineTotal;
      refineTotal.InputFaces=merged.faceCount();
      report.finalSizeRefineStopReason="level_limit";
      for(int level=0;level<8;++level) {
        SemanticMesh refined;
        TriangleRefineStats refineStep;
        std::string refineError;
        const bool refineOk=refineMidpointConforming(merged,cfg,refineProjector,refined,refineStep,
                                     &refineError,&coarsePatches,false,
                                     options.legacyCoverageAcceptance && level==0?&coveragePatches:nullptr,
                                     feasibleSplits);
        if(!refineOk) {
          report.finalSizeRefineStopReason="refine_error: "+refineError;
          break;
        }
        if(refined.LocalSizing)refined.LocalSizing->apply(refined,true);
        if(refineStep.SplitEdges==0 && refineStep.InsertedVertices==0) {
          report.finalSizeRefineStopReason=before.count>0?"no_safe_selected_split":"selected_size_satisfied";
          break;
        }
        const auto after=selectedSize(refined);
        report.finalSizeExcessBefore=before.excessSquared;report.finalSizeExcessAfter=after.excessSquared;
        const bool sizeImproved=sizeDefectProgress(before,after,feasibleSplits);
        // The first pass deliberately subdivides each still-unchanged patch
        // once, even when its sizing field was already within the edge limit.
        // Other selected patches are split only when overlong. Further passes
        // are size-driven; topology and degeneracy remain validated.
        if(!sizeImproved && !(options.legacyCoverageAcceptance && level==0)) {
          report.finalSizeRejectedSplits=refineStep.SplitEdges;
          report.finalSizeRefineStopReason="no_selected_size_progress";break;
        }
        merged=std::move(refined);
        before=after;
        refineTotal.SplitEdges+=refineStep.SplitEdges;
        refineTotal.InsertedVertices+=refineStep.InsertedVertices;
        refineTotal.RefinedFaces+=refineStep.RefinedFaces;
        refineTotal.OutputFaces=merged.faceCount();
        ++refineTotal.Levels;
        for(int i=0;i<8;++i)refineTotal.MaskCounts[i]+=refineStep.MaskCounts[i];
        refineTotal.touchedPatches.insert(refineStep.touchedPatches.begin(),
                                          refineStep.touchedPatches.end());
      }
      return refineTotal;
      };
      auto refineTotal=refineCandidate(options.selectFinalRegions?false:options.sizeFeasibleFinalRefine);
      if(options.selectFinalRegions) {
        SemanticMesh incumbent=std::move(merged);const auto incumbentStats=refineTotal;
        merged=refinementBase;const auto alternativeStart=Clock::now();
        const auto alternativeStats=refineCandidate(true);SemanticMesh alternative=std::move(merged);
        report.secondsFinalAlternativeRefine=std::chrono::duration<double>(Clock::now()-alternativeStart).count();
        std::string selectionError;
        if(!selectRefinementRegions(refinementBase,incumbent,alternative,cfg,lowQualityThreshold,
              merged,report.finalRegionSelection,&selectionError)) {
          merged=std::move(incumbent);refineTotal=incumbentStats;
        } else {
          refineTotal=incumbentStats;
          refineTotal.OutputFaces=merged.faceCount();
          refineTotal.InsertedVertices=merged.vertexCount()-refinementBase.vertexCount();
          refineTotal.SplitEdges=refineTotal.InsertedVertices;
          refineTotal.Levels=std::max(incumbentStats.Levels,alternativeStats.Levels);
          refineTotal.touchedPatches.clear();
          for(int f=0;f<merged.faceCount();++f)if(merged.faceAlive[f])
            for(int v:merged.face(f))if(v>=refinementBase.vertexCount()) {
              refineTotal.touchedPatches.insert(merged.facePatchId[f]);break;
            }
        }
      }
      if(refineTotal.Levels>0) {
          report.finalSizeRefineSplits=refineTotal.SplitEdges;
          report.finalSizeRefineLevels=refineTotal.Levels;
          report.sizeRecoveryRegions=int(refineTotal.touchedPatches.size());
          for(auto &patch:report.patches) {
            const int first=int(&patch-report.patches.data())*patchesPerTask;
            const int last=std::min(int(source.patches.size()),first+patchesPerTask);
            bool touched=false;
            for(int id=first;id<last;++id)touched|=refineTotal.touchedPatches.count(uint32_t(id))!=0;
            if(touched) {
              patch.unchangedPatchIds.erase(std::remove_if(patch.unchangedPatchIds.begin(),
                  patch.unchangedPatchIds.end(),[&](uint32_t id){return refineTotal.touchedPatches.count(id)!=0;}),
                  patch.unchangedPatchIds.end());
              patch.unchanged=patch.unchangedPatchIds.size()==size_t(last-first);
              patch.sizeRecovery|=std::any_of(refineTotal.touchedPatches.begin(),
                  refineTotal.touchedPatches.end(),[&](uint32_t id){return id>=uint32_t(first)&&id<uint32_t(last);});
            }
          }
          report.unchanged=int(std::count_if(report.patches.begin(),report.patches.end(),
              [](const RawPatchResult &patch){return patch.accepted && patch.unchanged;}));
      }
    }
    if(cfg.featureEdgeLength>0.f) {
      SemanticMesh beforeCircularPlanes=merged;
      const auto beforeCircularTaskReports=report.patches;
      const int beforeCircularUnchanged=report.unchanged;
      remeshCircularPlaneInteriors(merged,cfg,report);
      std::string circularError;
      if(!merged.validate(&circularError)) {
        merged=std::move(beforeCircularPlanes);
        report.patches=beforeCircularTaskReports;
        report.unchanged=beforeCircularUnchanged;
        report.circularPlaneRemeshed=report.circularPlaneAddedFaces=0;
        report.circularPlaneSkipped=0;
      }
    }
    merged.rebuildTopology();
    report.longEdgesAfterRefine=0;report.maxOutputEdgeRatio=0.f;
    if(merged.LocalSizing)merged.LocalSizing->apply(merged,true);
    for(const auto &edge:merged.edges) {
      const float h=.5f*(merged.targetLength[edge.v0]+merged.targetLength[edge.v1]);
      if(!(h>0.f))continue;
      const float ratio=distance(merged.position(int(edge.v0)),merged.position(int(edge.v1)))/h;
      report.maxOutputEdgeRatio=std::max(report.maxOutputEdgeRatio,ratio);
      if(ratio>cfg.splitRatio*(1.f+1.e-4f)) {
        ++report.longEdgesAfterRefine;
        const bool selected=(edge.face0>=0&&coarsePatches.count(merged.facePatchId[edge.face0])) ||
                            (edge.face1>=0&&coarsePatches.count(merged.facePatchId[edge.face1]));
        if(!selected)++report.finalSizeUnselectedLongEdges;
      }
    }
    merged.compact();merged.rebuildTopology();merged.computeVertexNormals();
    if(!merged.validate(&problem))throw std::runtime_error("assembled mesh: "+problem);
    report.topologyValid=report.boundariesHeld=true;
    report.outputQuality=evaluateRegionQuality(merged,cfg,lowQualityThreshold);
    report.qualityAccepted=qualityEndpointMask(report.sourceQuality,report.outputQuality)==0;
    const auto qualityAuditStart=Clock::now();
    report.outputPatchQuality=evaluatePatchQuality(merged,cfg,lowQualityThreshold);
    for(size_t p=0;p<report.sourcePatchQuality.size();++p)if(report.sourcePatchQuality[p].area>0) {
      const auto mask=qualityEndpointMask(report.sourcePatchQuality[p],report.outputPatchQuality[p]);
      if(mask)report.pendingPatchIds.push_back(uint32_t(p));
      if(mask&UnresolvedInputDefect)report.unresolvedPatchIds.push_back(uint32_t(p));
      if(mask&~uint32_t(UnresolvedInputDefect))report.regressedPatchIds.push_back(uint32_t(p));
    }
    // A changed healthy patch may lose a small amount of mean/P05 while the
    // complete mesh makes genuine progress. The opt-in policy uses the same
    // global non-regression and connected-defect gate above, and still reports
    // every pending patch. The default strict gate is unchanged.
    report.qualityAccepted=batchQualityEndpointAccepted(report.sourceQuality,report.outputQuality,
                                                         report.pendingPatchIds.size(),options.globalQualityAcceptance);
    report.secondsQualityAudit+=std::chrono::duration<double>(Clock::now()-qualityAuditStart).count();
    report.secondsAssembly=std::chrono::duration<double>(Clock::now()-boundaryStart).count()-
        report.secondsBoundary-report.secondsTasks;
    report.seconds=std::chrono::duration<double>(Clock::now()-start).count();
    output=std::move(merged);return true;
  } catch(const std::exception &e) {
    report.seconds=std::chrono::duration<double>(Clock::now()-start).count();
    if(report.secondsTasks>0)
      report.secondsAssembly=std::max(0.,report.seconds-report.secondsBoundary-report.secondsTasks);
    if(error)*error=e.what();return false;
  }
}
}
