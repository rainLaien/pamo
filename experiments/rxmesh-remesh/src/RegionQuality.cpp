#include "cad_adaptive/RegionQuality.h"
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <unordered_map>

namespace cad_adaptive {
namespace {
struct DisjointSet {
  std::vector<int> p;
  explicit DisjointSet(int n):p(n){std::iota(p.begin(),p.end(),0);}
  int find(int i){while(p[i]!=i){p[i]=p[p[i]];i=p[i];}return i;}
  void join(int a,int b){a=find(a);b=find(b);if(a!=b)p[std::max(a,b)]=std::min(a,b);}
};
template<class T>T quantile(std::vector<T>& v,size_t n){if(v.empty())return 0;std::nth_element(v.begin(),v.begin()+n,v.end());return v[n];}
double defectAreaTolerance(double area){return std::max(1.e-12,area*1.e-6);}
}
EdgeConstraintClassification classifyConstraint(const SemanticMesh& mesh,const EdgeRec&e){
  EdgeConstraintClassification c;
  if(mesh.featureEdges.count((uint64_t(e.v0)<<32)|e.v1))c.roles|=PersistentFeature;
  if(e.flags&EdgeMeshBoundary)c.roles|=OpenBoundary;
  if(e.flags&EdgePatchBoundary)c.roles|=PartitionInterface;
  c.allowedMotion=(c.roles&(OpenBoundary|PartitionInterface))?ConstraintMotion::Fixed:
    ((c.roles&PersistentFeature)?ConstraintMotion::Curve:ConstraintMotion::Surface);
  if(mesh.vertexConstraint[e.v0]>=uint8_t(VertexConstraint::Corner) &&
     mesh.vertexConstraint[e.v1]>=uint8_t(VertexConstraint::Corner))c.allowedMotion=ConstraintMotion::Fixed;
  // Only the open mesh boundary is intrinsic. Sharp-edge IDs can also be inferred
  // from STL tessellation and partition interfaces have no reliable provenance.
  c.geometryProvenanceKnown=(c.roles&OpenBoundary)!=0;
  return c;
}
RegionQuality evaluateRegionQuality(const SemanticMesh& mesh,const RemeshConfig& cfg,float threshold){
  RegionQuality s;s.threshold=threshold;
  std::vector<double> q(mesh.faceCount(),0),values;
  std::vector<float> ratios;
  std::vector<double> areas(mesh.faceCount(),0);
  std::vector<uint8_t> longAffected(mesh.faceCount(),0),shortAffected(mesh.faceCount(),0);
  DisjointSet components(mesh.faceCount());double sum=0,weighted=0;
  for(int f=0;f<mesh.faceCount();++f)if(mesh.faceAlive[f]){
    const auto a=mesh.facePoint(f,0),b=mesh.facePoint(f,1),c=mesh.facePoint(f,2);
    // Evaluate the declared float coordinates in double precision, matching
    // the independent audit without changing the shape-quality formula.
    const double u[3]={double(b.x)-a.x,double(b.y)-a.y,double(b.z)-a.z};
    const double v[3]={double(c.x)-a.x,double(c.y)-a.y,double(c.z)-a.z};
    const double w[3]={double(c.x)-b.x,double(c.y)-b.y,double(c.z)-b.z};
    const double n[3]={u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
    const double area2=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    const double denom=u[0]*u[0]+u[1]*u[1]+u[2]*u[2]+v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+w[0]*w[0]+w[1]*w[1]+w[2]*w[2];
    q[f]=denom>0?std::min(1.,2.*std::sqrt(3.)*area2/denom):0.;areas[f]=.5*area2;
    values.push_back(q[f]);sum+=q[f];weighted+=areas[f]*q[f];s.area+=areas[f];
    if(!(q[f]>0) || !std::isfinite(q[f]))++s.zeroFaces;
    if(q[f]<threshold){++s.lowQualityFaces;s.lowQualityArea+=areas[f];}
  }
  if(!values.empty()){
    s.mean=float(sum/values.size());s.minimum=*std::min_element(values.begin(),values.end());
    s.p05=quantile(values,values.size()/20);
  }
  s.areaWeightedMean=s.area>0?weighted/s.area:0;
  for(const auto&e:mesh.edges){
    if(e.face0>=0&&e.face1>=0){
      // Cross patch IDs: an artificial cut must not hide a connected defect.
      if(q[e.face0]<threshold&&q[e.face1]<threshold)components.join(e.face0,e.face1);
      auto direction=[&](int f){const auto t=mesh.face(f);for(int k=0;k<3;++k)
        if(uint32_t(t[k])==e.v0&&uint32_t(t[(k+1)%3])==e.v1)return 1;return -1;};
      if(direction(e.face0)==direction(e.face1))++s.inconsistentEdges;
    }
    const auto classification=classifyConstraint(mesh,e);
    const bool feature=(classification.roles&PersistentFeature)!=0;
    s.constraints.persistentFeatureEdges+=int(feature);
    s.constraints.openBoundaryEdges+=int((e.flags&EdgeMeshBoundary)!=0);
    s.constraints.partitionInterfaceEdges+=int((e.flags&EdgePatchBoundary)!=0);
    const float h=cfg.featureEdgeLength>0&&feature?cfg.featureEdgeLength:cfg.constantLength;
    const float stored=.5f*(mesh.targetLength[e.v0]+mesh.targetLength[e.v1]);
    const float target=stored>0?stored:h;
    if(!(target>0))continue;
    const float r=distance(mesh.position(e.v0),mesh.position(e.v1))/target;ratios.push_back(r);
    if(r>cfg.splitRatio*(1.f+1.e-4f)){++s.longEdges;
      for(int f:{e.face0,e.face1})if(f>=0)longAffected[f]=1;}
    if(r<cfg.collapseRatio){++s.shortEdges;if(feature||(e.flags&(EdgePatchBoundary|EdgeMeshBoundary)))++s.shortConstrainedEdges;
      for(int f:{e.face0,e.face1})if(f>=0)shortAffected[f]=1;}
    const float ha=mesh.targetLength[e.v0]>0?mesh.targetLength[e.v0]:h;
    const float hb=mesh.targetLength[e.v1]>0?mesh.targetLength[e.v1]:h;
    if(std::min(ha,hb)>0)s.maxTargetTransitionRatio=std::max(s.maxTargetTransitionRatio,std::max(ha,hb)/std::min(ha,hb));
  }
  if(!ratios.empty()){
    s.edgeRatioMin=*std::min_element(ratios.begin(),ratios.end());s.edgeRatioMax=*std::max_element(ratios.begin(),ratios.end());
    s.edgeRatioP05=quantile(ratios,ratios.size()/20);s.edgeRatioP95=quantile(ratios,ratios.size()*95/100);
  }
  for(auto c:mesh.vertexConstraint){if(c>=uint8_t(VertexConstraint::Corner))++s.constraints.fixedVertices;
    else if(c>=uint8_t(VertexConstraint::FeatureEdge))++s.constraints.curveVertices;else ++s.constraints.surfaceVertices;}
  std::map<int,DefectRegion> groups;std::map<int,std::set<uint32_t>> patches;
  for(int f=0;f<mesh.faceCount();++f)if(mesh.faceAlive[f]&&q[f]<threshold){
    const int root=components.find(f);auto&g=groups[root];
    if(g.faces==0)g.lower=g.upper=mesh.facePoint(f,0);
    ++g.faces;g.area+=areas[f];g.minQuality=std::min(g.minQuality,float(q[f]));patches[root].insert(mesh.facePatchId[f]);
    for(int k=0;k<3;++k){const auto p=mesh.facePoint(f,k);
      g.lower={std::min(g.lower.x,p.x),std::min(g.lower.y,p.y),std::min(g.lower.z,p.z)};
      g.upper={std::max(g.upper.x,p.x),std::max(g.upper.y,p.y),std::max(g.upper.z,p.z)};}
  }
  s.lowQualityComponents=int(groups.size());
  for(auto&entry:groups){auto&g=entry.second;g.patches.assign(patches[entry.first].begin(),patches[entry.first].end());s.regions.push_back(std::move(g));}
  std::sort(s.regions.begin(),s.regions.end(),[](const auto&a,const auto&b){return a.area>b.area;});
  if(!s.regions.empty())s.largestLowQualityArea=s.regions[0].area;
  auto sizeComponents=[&](const std::vector<uint8_t>& affected,int& count,double& maximum){
    DisjointSet group(mesh.faceCount());
    for(const auto&e:mesh.edges)if(e.face0>=0&&e.face1>=0&&affected[e.face0]&&affected[e.face1])group.join(e.face0,e.face1);
    std::vector<double> sums(mesh.faceCount(),0);
    for(int f=0;f<mesh.faceCount();++f)if(mesh.faceAlive[f]&&affected[f])sums[group.find(f)]+=areas[f];
    for(double area:sums)if(area>0){++count;maximum=std::max(maximum,area);}
  };
  sizeComponents(longAffected,s.longEdgeFaceComponents,s.largestLongEdgeFaceArea);
  sizeComponents(shortAffected,s.shortEdgeFaceComponents,s.largestShortEdgeFaceArea);
  return s;
}
uint32_t qualityRegressionMask(const RegionQuality&a,const RegionQuality&b){
  uint32_t flags=0;
  if(b.mean+1.e-6f<a.mean)flags|=MeanRegression;
  if(b.p05+1.e-6f<a.p05)flags|=P05Regression;
  if(b.areaWeightedMean+1.e-6<a.areaWeightedMean)flags|=AreaMeanRegression;
  if(b.lowQualityArea>a.lowQualityArea+defectAreaTolerance(a.lowQualityArea))flags|=LowAreaRegression;
  if(b.largestLowQualityArea>a.largestLowQualityArea+defectAreaTolerance(a.largestLowQualityArea))flags|=LargestRegionRegression;
  if(b.zeroFaces)flags|=InvalidFaces;
  if(b.inconsistentEdges>a.inconsistentEdges)flags|=WindingRegression;
  return flags;
}
bool sizeDefectProgress(const SizeDefectSummary&a,const SizeDefectSummary&b,bool allowSeverityProgress){
  const bool severity=allowSeverityProgress && b.maxRatio<=a.maxRatio*(1.f+1.e-4f) &&
      b.excessSquared<a.excessSquared-std::max(1.e-12,a.excessSquared*1.e-6);
  return b.count<a.count || b.maxRatio<a.maxRatio-1.e-4f || severity;
}
bool qualityNonRegression(const RegionQuality&a,const RegionQuality&b){return qualityRegressionMask(a,b)==0;}
std::vector<uint32_t> longEdgePatches(const SemanticMesh& mesh,const RemeshConfig& cfg) {
  std::set<uint32_t> patches;
  for(const auto &edge:mesh.edges) {
    const float stored=.5f*(mesh.targetLength[edge.v0]+mesh.targetLength[edge.v1]);
    const float target=stored>0?stored:cfg.constantLength;
    if(!(target>0) || distance(mesh.position(edge.v0),mesh.position(edge.v1))/target<=cfg.splitRatio*(1.f+1.e-4f))continue;
    for(int face:{edge.face0,edge.face1})if(face>=0)patches.insert(mesh.facePatchId[face]);
  }
  return {patches.begin(),patches.end()};
}
uint32_t qualityEndpointMask(const RegionQuality&a,const RegionQuality&b){
  uint32_t flags=qualityRegressionMask(a,b);
  if(a.lowQualityFaces>0 && !qualityProgress(a,b))flags|=UnresolvedInputDefect;
  return flags;
}
bool batchQualityEndpointAccepted(const RegionQuality&a,const RegionQuality&b,
                                  size_t pending,bool globalQualityAcceptance){
  return qualityEndpointMask(a,b)==0 && (globalQualityAcceptance || pending==0);
}
bool qualityProgress(const RegionQuality&a,const RegionQuality&b){
  return b.mean>a.mean+1.e-6f || b.p05>a.p05+1.e-6f ||
    b.areaWeightedMean>a.areaWeightedMean+1.e-6 || b.lowQualityArea<a.lowQualityArea-defectAreaTolerance(a.lowQualityArea) ||
    b.largestLowQualityArea<a.largestLowQualityArea-defectAreaTolerance(a.largestLowQualityArea);
}
std::vector<RegionQuality> evaluatePatchQuality(const SemanticMesh&mesh,const RemeshConfig&cfg,float threshold){
  std::vector<std::vector<int>> faces(mesh.patches.size());
  for(int f=0;f<mesh.faceCount();++f)if(mesh.faceAlive[f]){
    if(mesh.facePatchId[f]>=faces.size())throw std::runtime_error("invalid patch ownership in regional quality audit");
    faces[mesh.facePatchId[f]].push_back(f);
  }
  std::vector<RegionQuality> out(mesh.patches.size());
  for(size_t p=0;p<faces.size();++p){
    if(faces[p].empty())continue;
    SemanticMesh part;part.patches.push_back(mesh.patches[p]);
    std::unordered_map<int,int> indices;
    for(int f:faces[p]){auto t=mesh.face(f);for(int&v:t){const int old=v;auto it=indices.find(old);
      if(it==indices.end()){v=part.addVertex(mesh.position(old),0,VertexConstraint(mesh.vertexConstraint[old]));
        part.targetLength[v]=mesh.targetLength[old];indices.emplace(old,v);}else v=it->second;}
      part.addFace(t[0],t[1],t[2],0,mesh.patches[p].type);}
    part.rebuildTopology();out[p]=evaluateRegionQuality(part,cfg,threshold);
    for(auto&region:out[p].regions)region.patches={uint32_t(p)};
  }
  return out;
}
void writeRegionQualityJson(std::ostream&o,const RegionQuality&s){
  o<<"{\"threshold\":"<<s.threshold<<",\"mean\":"<<s.mean<<",\"p05\":"<<s.p05<<",\"minimum\":"<<s.minimum
   <<",\"area\":"<<s.area<<",\"area_weighted_mean\":"<<s.areaWeightedMean<<",\"low_quality_faces\":"<<s.lowQualityFaces
   <<",\"low_quality_area\":"<<s.lowQualityArea<<",\"low_quality_components\":"<<s.lowQualityComponents
   <<",\"largest_low_quality_area\":"<<s.largestLowQualityArea<<",\"long_edges\":"<<s.longEdges<<",\"short_edges\":"<<s.shortEdges
   <<",\"short_constrained_edges\":"<<s.shortConstrainedEdges<<",\"edge_ratio_min\":"<<s.edgeRatioMin
   <<",\"long_edge_face_components\":"<<s.longEdgeFaceComponents<<",\"largest_long_edge_face_area\":"<<s.largestLongEdgeFaceArea
   <<",\"short_edge_face_components\":"<<s.shortEdgeFaceComponents<<",\"largest_short_edge_face_area\":"<<s.largestShortEdgeFaceArea
   <<",\"max_target_transition_ratio\":"<<s.maxTargetTransitionRatio
   <<",\"edge_ratio_p05\":"<<s.edgeRatioP05<<",\"edge_ratio_p95\":"<<s.edgeRatioP95<<",\"edge_ratio_max\":"<<s.edgeRatioMax
   <<",\"zero_faces\":"<<s.zeroFaces<<",\"inconsistent_edges\":"<<s.inconsistentEdges
   <<",\"constraints\":{\"fixed_vertices\":"<<s.constraints.fixedVertices<<",\"curve_vertices\":"<<s.constraints.curveVertices
   <<",\"surface_vertices\":"<<s.constraints.surfaceVertices<<",\"persistent_feature_edges\":"<<s.constraints.persistentFeatureEdges
   <<",\"open_boundary_edges\":"<<s.constraints.openBoundaryEdges<<",\"partition_interfaces_unknown_provenance\":"<<s.constraints.partitionInterfaceEdges<<"}"
   <<",\"reported_region_limit\":32,\"regions\":[";
  for(size_t i=0;i<std::min<size_t>(32,s.regions.size());++i){const auto&g=s.regions[i];o<<(i?",":"")<<"{\"faces\":"<<g.faces<<",\"area\":"<<g.area<<",\"minimum\":"<<g.minQuality
    <<",\"bbox\":["<<g.lower.x<<','<<g.lower.y<<','<<g.lower.z<<','<<g.upper.x<<','<<g.upper.y<<','<<g.upper.z<<"],\"patch_ids\":[";
    for(size_t j=0;j<g.patches.size();++j)o<<(j?",":"")<<g.patches[j];o<<"]}";}
  o<<"]}";
}
}
