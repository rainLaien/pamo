#include "cad_adaptive/RegionCandidateSelection.h"
#include <algorithm>
#include <chrono>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_set>
namespace cad_adaptive {
namespace {
uint64_t key(int a,int b){if(a>b)std::swap(a,b);return(uint64_t(uint32_t(a))<<32)|uint32_t(b);}
struct Groups {
  std::vector<int> parent;
  explicit Groups(int n):parent(n){std::iota(parent.begin(),parent.end(),0);}
  int find(int p){while(parent[p]!=p){parent[p]=parent[parent[p]];p=parent[p];}return p;}
  void join(int a,int b){a=find(a);b=find(b);if(a!=b)parent[std::max(a,b)]=std::min(a,b);}
};
using Faces=std::vector<std::vector<int>>;
Faces patchFaces(const SemanticMesh&m){
  Faces out(m.patches.size());
  for(int f=0;f<m.faceCount();++f)if(m.faceAlive[f])out.at(m.facePatchId[f]).push_back(f);
  return out;
}
SemanticMesh extract(const SemanticMesh&m,const Faces&faces,const std::vector<int>&patches){
  SemanticMesh out;std::map<int,int> vertices;
  for(int p:patches){const int owner=out.patches.size();out.patches.push_back(m.patches[p]);
    for(int f:faces[p]){auto t=m.face(f);for(auto&v:t){auto it=vertices.find(v);
      if(it==vertices.end()){const int old=v;v=out.addVertex(m.position(old),owner,VertexConstraint(m.vertexConstraint[old]));
        out.targetLength[v]=m.targetLength[old];vertices.emplace(old,v);}else v=it->second;}
      out.addFace(t[0],t[1],t[2],owner,m.patches[p].type);}}
  out.rebuildTopology();return out;
}
bool different(const RegionQuality&a,const RegionQuality&b){
  return a.mean!=b.mean || a.p05!=b.p05 || a.areaWeightedMean!=b.areaWeightedMean ||
      a.lowQualityArea!=b.lowQualityArea || a.largestLowQualityArea!=b.largestLowQualityArea ||
      a.longEdges!=b.longEdges || a.shortEdges!=b.shortEdges;
}
}
bool selectRefinementRegions(const SemanticMesh&base,const SemanticMesh&incumbent,
    const SemanticMesh&candidate,const RemeshConfig&cfg,float threshold,
    SemanticMesh&output,RegionCandidateSelection&report,std::string*error){
  const auto start=std::chrono::steady_clock::now();report={};
  try {
    const int prefix=base.vertexCount(),np=base.patches.size();
    for(const auto*m:{&incumbent,&candidate}){
      if(m->vertexCount()<prefix || m->patches.size()!=base.patches.size())throw std::runtime_error("refinement prefix/ownership mismatch");
      for(int v=0;v<prefix;++v)if(m->px[v]!=base.px[v] || m->py[v]!=base.py[v] || m->pz[v]!=base.pz[v] ||
          m->targetLength[v]!=base.targetLength[v] || m->vertexConstraint[v]!=base.vertexConstraint[v])
        throw std::runtime_error("refinement changed common anchor contract");
    }
    const auto before=evaluatePatchQuality(incumbent,cfg,threshold),after=evaluatePatchQuality(candidate,cfg,threshold);
    const auto oldFaces=patchFaces(incumbent),newFaces=patchFaces(candidate);
    Groups groups(np);
    for(const auto*m:{&incumbent,&candidate})for(int v=prefix;v<m->vertexCount();++v){
      int first=-1;for(int f:m->incidentFaces[v]){const int p=m->facePatchId[f];
        if(first<0)first=p;else groups.join(first,p);}}
    std::map<int,std::vector<int>> members;
    for(int p=0;p<np;++p)members[groups.find(p)].push_back(p);
    std::vector<bool> selected(np,false);
    for(const auto&entry:members){const auto&ids=entry.second;bool changed=false,safe=true,progress=false;
      for(int p:ids){changed|=different(before[p],after[p]);
        safe&=qualityNonRegression(before[p],after[p]) && after[p].longEdges<=before[p].longEdges && after[p].shortEdges<=before[p].shortEdges;
        progress|=qualityProgress(before[p],after[p]) || after[p].longEdges<before[p].longEdges || after[p].shortEdges<before[p].shortEdges;}
      if(!changed)continue;++report.groupsCompared;
      if(!safe || !progress)continue;
      if(ids.size()>1){const auto a=evaluateRegionQuality(extract(incumbent,oldFaces,ids),cfg,threshold);
        const auto b=evaluateRegionQuality(extract(candidate,newFaces,ids),cfg,threshold);
        if(!qualityNonRegression(a,b))continue;}
      ++report.groupsSelected;for(int p:ids){selected[p]=true;report.selectedPatchIds.push_back(p);}
    }
    if(report.selectedPatchIds.empty()){
      output=incumbent;report.stopReason="no_nonregressing_region";
    }else{
      SemanticMesh joined=base;joined.i0.clear();joined.i1.clear();joined.i2.clear();
      joined.facePatchId.clear();joined.facePatchType.clear();joined.faceAlive.clear();joined.featureEdges.clear();
      joined.edges.clear();joined.incidentFaces.assign(prefix,{});
      std::vector<int> maps[2]={std::vector<int>(incumbent.vertexCount(),-1),std::vector<int>(candidate.vertexCount(),-1)};
      for(auto&map:maps)for(int v=0;v<prefix;++v)map[v]=v;
      const SemanticMesh*inputs[2]={&incumbent,&candidate};const Faces*faces[2]={&oldFaces,&newFaces};
      for(int p=0;p<np;++p){const int choice=selected[p]?1:0;const auto&m=*inputs[choice];auto&map=maps[choice];
        for(int f:(*faces[choice])[p]){auto t=m.face(f);for(int&v:t){const int old=v;
          if(map[old]<0){const int id=joined.addVertex(m.position(old),m.vertexPatchId[old],VertexConstraint(m.vertexConstraint[old]));
            joined.targetLength[id]=m.targetLength[old];joined.curvature[id]=m.curvature[old];joined.featureDistance[id]=m.featureDistance[old];map[old]=id;}
          v=map[old];}joined.addFace(t[0],t[1],t[2],p,m.patches[p].type);}}
      joined.rebuildTopology();std::unordered_set<uint64_t> edges;
      for(const auto&e:joined.edges)edges.insert(key(e.v0,e.v1));
      for(int choice=0;choice<2;++choice)for(const auto&e:inputs[choice]->featureEdges){
        const int a=maps[choice][uint32_t(e.first>>32)],b=maps[choice][uint32_t(e.first)];
        if(a>=0 && b>=0 && edges.count(key(a,b))){const auto inserted=joined.featureEdges.emplace(key(a,b),e.second);
          if(!inserted.second && inserted.first->second!=e.second)throw std::runtime_error("feature identity conflict");}}
      joined.rebuildTopology();std::string problem;
      if(!joined.validate(&problem))throw std::runtime_error("regional join: "+problem);
      const auto a=evaluateRegionQuality(incumbent,cfg,threshold),b=evaluateRegionQuality(joined,cfg,threshold);
      if(!qualityNonRegression(a,b) || b.longEdges>a.longEdges || b.shortEdges>a.shortEdges){
        output=incumbent;report.groupsSelected=0;report.selectedPatchIds.clear();report.stopReason="connected_global_guard_rejected";
      }else{output=std::move(joined);report.stopReason="regional_progress_retained";}
    }
    report.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return true;
  }catch(const std::exception&e){if(error)*error=e.what();report.stopReason=e.what();
    report.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return false;}
}
}
