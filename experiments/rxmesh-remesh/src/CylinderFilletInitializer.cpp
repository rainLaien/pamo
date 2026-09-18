#include "cad_adaptive/CylinderFilletInitializer.h"
#include "cad_adaptive/PartitionInput.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace cad_adaptive {
namespace {
constexpr double stPi = 3.14159265358979323846;
struct PointD { double X=0, Y=0, Z=0; };
PointD toDouble(Vec3 p) { return {p.x,p.y,p.z}; }
Vec3 toFloat(PointD p) { return {float(p.X),float(p.Y),float(p.Z)}; }
PointD operator+(PointD a,PointD b) { return {a.X+b.X,a.Y+b.Y,a.Z+b.Z}; }
PointD operator-(PointD a,PointD b) { return {a.X-b.X,a.Y-b.Y,a.Z-b.Z}; }
PointD operator*(PointD a,double s) { return {a.X*s,a.Y*s,a.Z*s}; }
double dotD(PointD a,PointD b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
PointD crossD(PointD a,PointD b) { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
double normD(PointD p) { return std::sqrt(dotD(p,p)); }
PointD unitD(PointD p) { const double n=normD(p); if(!(n>1e-20)) throw std::runtime_error("zero fillet axis"); return p*(1.0/n); }
uint64_t edgeKey(int a,int b) { if(a>b) std::swap(a,b); return (uint64_t(uint32_t(a))<<32)|uint32_t(b); }
struct SidePoint { double Parameter=0; int Vertex=-1; };
struct StripPlan {
  uint32_t Patch=0;
  PointD Origin, Axis, X, Y;
  double Radius=0, AngleStart=0, Width=0, LowZ=0, HighZ=0, Tolerance=0;
  double Target=0, Orientation=1;
  // 0,1 are constant-angle axial rails; 2,3 are constant-z end arcs.
  std::array<std::vector<SidePoint>,4> Sides;
  std::array<std::set<uint64_t>,4> SideEdges;
  std::vector<double> Angles, Heights;
};
double angleOf(const StripPlan &p,PointD x) {
  const PointD d=x-p.Origin;
  double a=std::atan2(dotD(d,p.Y),dotD(d,p.X))-p.AngleStart;
  while(a < -p.Tolerance/p.Radius) a+=2*stPi;
  while(a >= 2*stPi-p.Tolerance/p.Radius) a-=2*stPi;
  return a;
}
std::vector<double> sampleUnion(const std::vector<SidePoint> &a,const std::vector<SidePoint> &b,
                               double step,double tolerance) {
  std::vector<double> knots;
  for(const auto &x:a) knots.push_back(x.Parameter);
  for(const auto &x:b) knots.push_back(x.Parameter);
  std::sort(knots.begin(),knots.end());
  std::vector<double> unique;
  for(double x:knots) if(unique.empty() || x-unique.back()>tolerance) unique.push_back(x);
  if(unique.size()<2) throw std::runtime_error("collapsed fillet parameter range");
  std::vector<double> samples{unique.front()};
  for(size_t i=1;i<unique.size();++i) {
    const double length=unique[i]-unique[i-1];
    const double raw=std::ceil(length/step-1e-7);
    if(!std::isfinite(raw) || raw>100000) throw std::runtime_error("fillet segment budget exceeded");
    const int count=std::max(1,int(raw));
    for(int k=1;k<=count;++k) samples.push_back(unique[i-1]+length*double(k)/count);
  }
  return samples;
}

bool buildPlan(const SemanticMesh &mesh,uint32_t pid,const RemeshConfig &cfg,
               StripPlan &p,std::string &why) {
  const auto reject=[&](const char *reason) { why=reason; return false; };
  const PatchRecord &patch=mesh.patches[pid];
  p.Patch=pid; p.Origin=toDouble(patch.origin); p.Axis=unitD(toDouble(patch.axis));
  p.Radius=patch.radius; p.Tolerance=std::max(1e-6,double(mesh.bboxDiagonal())*1e-7);
  if(!(p.Radius>p.Tolerance)) return reject("invalid/small radius");
  std::vector<const EdgeRec*> border;
  std::map<int,int> boundaryDegree;
  std::set<int> used, boundaryVertices;
  double patchArea=0;
  for(int f=0;f<mesh.faceCount();++f) if(mesh.faceAlive[f] && mesh.facePatchId[f]==pid)
    for(int id:mesh.face(f)) used.insert(id);
  if(used.empty()) return reject("empty patch");
  PointD radial=toDouble(mesh.position(*used.begin()))-p.Origin;
  radial=radial-p.Axis*dotD(radial,p.Axis);
  p.X=unitD(radial); p.Y=crossD(p.Axis,p.X);
  for(const auto &e:mesh.edges) if((e.patchLeft==pid)!=(e.patchRight==pid)) {
    if(e.face0<0 || e.face1<0) return reject("open cylinder boundary");
    border.push_back(&e); ++boundaryDegree[e.v0]; ++boundaryDegree[e.v1];
    boundaryVertices.insert(e.v0); boundaryVertices.insert(e.v1);
  }
  if(border.size()<4) return reject("no rectangular boundary");
  for(const auto &kv:boundaryDegree) if(kv.second!=2) return reject("branched boundary");
  for(int id:used) if(!boundaryVertices.count(id) &&
      mesh.vertexConstraint[id]!=uint8_t(VertexConstraint::Surface) &&
      mesh.vertexConstraint[id]!=uint8_t(VertexConstraint::Free))
    return reject("constrained interior vertex");
  std::vector<double> angles;
  p.LowZ=1e300; p.HighZ=-1e300;
  for(int id:used) {
    const PointD q=toDouble(mesh.position(id))-p.Origin;
    double a=std::atan2(dotD(q,p.Y),dotD(q,p.X)); if(a<0) a+=2*stPi;
    angles.push_back(a);
    const double z=dotD(q,p.Axis); p.LowZ=std::min(p.LowZ,z); p.HighZ=std::max(p.HighZ,z);
  }
  std::sort(angles.begin(),angles.end()); double largestGap=-1;
  for(size_t i=0;i<angles.size();++i) {
    const double next=i+1<angles.size()?angles[i+1]:angles[0]+2*stPi;
    if(next-angles[i]>largestGap) { largestGap=next-angles[i]; p.AngleStart=std::fmod(next,2*stPi); }
  }
  p.Width=2*stPi-largestGap;
  if(!(p.Width>1e-5 && p.Width<stPi-1e-5 && p.HighZ-p.LowZ>p.Tolerance))
    return reject("not a partial rectangular fillet");
  const double angleTol=p.Tolerance/p.Radius, tangentCos=std::cos(stPi/180.0);
  std::array<std::set<uint32_t>,2> railSupports;
  for(const EdgeRec *ep:border) {
    const auto &e=*ep;
    const PointD a=toDouble(mesh.position(e.v0)),b=toDouble(mesh.position(e.v1));
    const double ua=angleOf(p,a),ub=angleOf(p,b);
    const double za=dotD(a-p.Origin,p.Axis),zb=dotD(b-p.Origin,p.Axis);
    int side=-1;
    if(std::abs(ua)<angleTol && std::abs(ub)<angleTol) side=0;
    else if(std::abs(ua-p.Width)<angleTol && std::abs(ub-p.Width)<angleTol) side=1;
    else if(std::abs(za-p.LowZ)<p.Tolerance && std::abs(zb-p.LowZ)<p.Tolerance) side=2;
    else if(std::abs(za-p.HighZ)<p.Tolerance && std::abs(zb-p.HighZ)<p.Tolerance) side=3;
    if(side<0) return reject("trim is not a four-sided strip");
    const uint32_t neighbor=e.patchLeft==pid?e.patchRight:e.patchLeft;
    if(neighbor>=mesh.patches.size() || mesh.patches[neighbor].type!=PatchType::Plane)
      return reject("nonplanar supporting face");
    const PointD n=unitD(toDouble(mesh.patches[neighbor].axis));
    if(side<2) {
      PointD r=(a+b)*0.5-p.Origin; r=unitD(r-p.Axis*dotD(r,p.Axis));
      if(std::abs(dotD(n,r))<tangentCos || std::abs(dotD(n,p.Axis))>0.02)
        return reject("axial neighbor is not tangent");
      railSupports[side].insert(neighbor);
    } else if(std::abs(dotD(n,p.Axis))<tangentCos) return reject("end trim not perpendicular to axis");
    p.Sides[side].push_back({side<2?za:ua,int(e.v0)});
    p.Sides[side].push_back({side<2?zb:ub,int(e.v1)});
    p.SideEdges[side].insert(edgeKey(e.v0,e.v1));
  }
  if(railSupports[0].size()!=1 || railSupports[1].size()!=1 || railSupports[0]==railSupports[1])
    return reject("requires two distinct tangent supports");
  for(int s=0;s<4;++s) {
    auto &chain=p.Sides[s]; std::sort(chain.begin(),chain.end(),[](auto a,auto b){return a.Parameter<b.Parameter;});
    chain.erase(std::unique(chain.begin(),chain.end(),[](auto a,auto b){return a.Vertex==b.Vertex;}),chain.end());
    const double tolerance=s<2?p.Tolerance:angleTol;
    const double low=s<2?p.LowZ:0, high=s<2?p.HighZ:p.Width;
    if(chain.size()<2 || std::abs(chain.front().Parameter-low)>tolerance ||
       std::abs(chain.back().Parameter-high)>tolerance) return reject("incomplete side chain");
    for(size_t j=1;j<chain.size();++j) if(chain[j].Parameter-chain[j-1].Parameter<=tolerance ||
        !p.SideEdges[s].count(edgeKey(chain[j-1].Vertex,chain[j].Vertex)))
      return reject("disconnected/repeated side chain");
  }
  for(int f=0;f<mesh.faceCount();++f) if(mesh.faceAlive[f] && mesh.facePatchId[f]==pid) {
    const PointD a=toDouble(mesh.facePoint(f,0)),b=toDouble(mesh.facePoint(f,1)),c=toDouble(mesh.facePoint(f,2));
    const double au=angleOf(p,a),bu=angleOf(p,b),cu=angleOf(p,c);
    const double az=dotD(a-p.Origin,p.Axis),bz=dotD(b-p.Origin,p.Axis),cz=dotD(c-p.Origin,p.Axis);
    patchArea+=0.5*((bu-au)*(cz-az)-(cu-au)*(bz-az));
  }
  const double expected=p.Width*(p.HighZ-p.LowZ);
  if(std::abs(std::abs(patchArea)-expected)>expected*1e-4) return reject("patch has holes or overlapping parameter coverage");
  p.Orientation=patchArea>0?1:-1;
  const double eps=std::min(double(cfg.maxGeometryError),p.Radius);
  const double chord=2*std::sqrt(2*p.Radius*eps-eps*eps);
  const double normalChord=2*p.Radius*std::sin(cfg.normalDegrees*stPi/180);
  p.Target=std::min(double(cfg.constantLength),0.95*std::min(chord,normalChord)/cfg.splitRatio);
  if(!(p.Target>p.Tolerance*10)) return reject("target below stable coordinate resolution");
  p.Angles=sampleUnion(p.Sides[2],p.Sides[3],p.Target/p.Radius,angleTol);
  const double meanArc=p.Radius*p.Width/double(p.Angles.size()-1);
  const double axialStep=std::min(p.Target,0.8660254037844386*meanArc);
  p.Heights=sampleUnion(p.Sides[0],p.Sides[1],axialStep,p.Tolerance);
  if(p.Angles.size()>1000000/p.Heights.size()) throw std::runtime_error("fillet initialization exceeds one million vertices");
  return true;
}

using EdgeInsertions=std::map<uint64_t,std::vector<std::pair<double,int>>>;
int boundarySample(SemanticMesh &mesh,const StripPlan &p,int side,double t,
                   EdgeInsertions &insertions,uint32_t &added) {
  const auto &chain=p.Sides[side]; const double tol=side<2?p.Tolerance:p.Tolerance/p.Radius;
  auto next=std::lower_bound(chain.begin(),chain.end(),t,[](const SidePoint &x,double value){return x.Parameter<value;});
  if(next!=chain.end() && std::abs(next->Parameter-t)<=tol) return next->Vertex;
  if(next!=chain.begin() && std::abs((next-1)->Parameter-t)<=tol) return (next-1)->Vertex;
  if(next==chain.begin() || next==chain.end()) throw std::runtime_error("boundary sample outside source chain");
  const auto &b=*next,&a=*(next-1);
  const double fraction=(t-a.Parameter)/(b.Parameter-a.Parameter);
  const uint64_t key=edgeKey(a.Vertex,b.Vertex);
  const double canonicalFraction=a.Vertex<b.Vertex?fraction:1-fraction;
  auto &row=insertions[key];
  for(const auto &old:row) if(std::abs(old.first-canonicalFraction)<1e-10) return old.second;
  const Vec3 point=toFloat(toDouble(mesh.position(a.Vertex))*(1-fraction)+toDouble(mesh.position(b.Vertex))*fraction);
  const int id=mesh.addVertex(point,p.Patch,VertexConstraint::Locked);
  row.push_back({canonicalFraction,id}); ++added;
  return id;
}

void triangulateConvexPolygon(SemanticMesh &mesh,std::vector<int> polygon,uint32_t patch,PointD normal) {
  // Ear clipping preserves every collinear boundary sample. Never fan an entire
  // new cylinder boundary into one old cylinder vertex; this routine only
  // subdivides neighboring source triangles and the validation reference.
  normal=unitD(normal);
  const auto side=[&](PointD a,PointD b,PointD c){return dotD(crossD(b-a,c-a),normal);};
  while(polygon.size()>3) {
    size_t best=polygon.size(); float bestQuality=-1;
    for(size_t i=0;i<polygon.size();++i) {
      const int ai=polygon[(i+polygon.size()-1)%polygon.size()],bi=polygon[i],ci=polygon[(i+1)%polygon.size()];
      const PointD a=toDouble(mesh.position(ai)),b=toDouble(mesh.position(bi)),c=toDouble(mesh.position(ci));
      const double area=side(a,b,c); if(!(area>1e-16)) continue;
      bool contains=false; const double tol=area*1e-10;
      for(int id:polygon) if(id!=ai && id!=bi && id!=ci) {
        const PointD x=toDouble(mesh.position(id));
        if(side(a,b,x)>=-tol && side(b,c,x)>=-tol && side(c,a,x)>=-tol) {contains=true;break;}
      }
      if(contains) continue;
      const float q=triangleQuality(mesh.position(ai),mesh.position(bi),mesh.position(ci));
      if(q>bestQuality) {bestQuality=q;best=i;}
    }
    if(best==polygon.size()) throw std::runtime_error("cannot triangulate shared boundary polygon");
    const size_t n=polygon.size();
    mesh.addFace(polygon[(best+n-1)%n],polygon[best],polygon[(best+1)%n],patch,mesh.patches[patch].type);
    polygon.erase(polygon.begin()+best);
  }
  if(polygon.size()!=3 || side(toDouble(mesh.position(polygon[0])),toDouble(mesh.position(polygon[1])),
      toDouble(mesh.position(polygon[2])))<=0) throw std::runtime_error("degenerate boundary retriangulation");
  mesh.addFace(polygon[0],polygon[1],polygon[2],patch,mesh.patches[patch].type);
}

SemanticMesh refineSourceBoundaryFaces(const SemanticMesh &source,const SemanticMesh &vertices,
                                       EdgeInsertions &insertions) {
  SemanticMesh result=vertices;
  result.i0.clear();result.i1.clear();result.i2.clear();result.facePatchId.clear();
  result.facePatchType.clear();result.faceAlive.clear();
  for(auto &kv:insertions) std::sort(kv.second.begin(),kv.second.end());
  for(int f=0;f<source.faceCount();++f) if(source.faceAlive[f]) {
    const auto tri=source.face(f); std::vector<int> polygon;
    for(int k=0;k<3;++k) {
      const int a=tri[k],b=tri[(k+1)%3]; polygon.push_back(a);
      const auto it=insertions.find(edgeKey(a,b));
      if(it==insertions.end()) continue;
      if(a<b) for(const auto &sample:it->second) polygon.push_back(sample.second);
      else for(auto sample=it->second.rbegin();sample!=it->second.rend();++sample) polygon.push_back(sample->second);
    }
    if(polygon.size()==3) result.addFace(tri[0],tri[1],tri[2],source.facePatchId[f],source.patches[source.facePatchId[f]].type);
    else triangulateConvexPolygon(result,std::move(polygon),source.facePatchId[f],
        crossD(toDouble(source.position(tri[1]))-toDouble(source.position(tri[0])),
               toDouble(source.position(tri[2]))-toDouble(source.position(tri[0]))));
  }
  result.rebuildTopology(); return result;
}
} // namespace

bool initializeCylinderFillets(SemanticMesh &mesh,const RemeshConfig &config,
    CylinderFilletInitReport &report,std::string *error) {
  report={};
  try {
    if(!(config.constantLength>0) || !std::isfinite(config.constantLength) ||
       !(config.maxGeometryError>0) || !std::isfinite(config.maxGeometryError) ||
       !(config.splitRatio>1) || !std::isfinite(config.splitRatio) ||
       !(config.normalDegrees>0 && config.normalDegrees<90))
      throw std::runtime_error("invalid fillet initialization settings");
    SemanticMesh source=mesh; source.rebuildTopology();
    report.PatchTargetLengths.assign(source.patches.size(),config.constantLength);
    std::vector<StripPlan> plans;
    for(uint32_t p=0;p<source.patches.size();++p) if(source.patches[p].type==PatchType::Cylinder) {
      ++report.Detected; StripPlan plan; std::string reason;
      if(buildPlan(source,p,config,plan,reason)) plans.push_back(std::move(plan));
      else report.Skipped.push_back("patch="+std::to_string(p)+" reason="+reason);
    }
    if(plans.empty()) return true;
    size_t totalGridVertices=0;
    for(const auto &p:plans) totalGridVertices+=p.Angles.size()*p.Heights.size();
    if(totalGridVertices>1000000) throw std::runtime_error("combined fillet initialization budget exceeded");
    SemanticMesh staged=source; EdgeInsertions insertions;
    std::vector<std::vector<int>> grids;
    for(const auto &p:plans) {
      const size_t nu=p.Angles.size(),nz=p.Heights.size();
      std::vector<int> ids(nu*nz,-1);
      for(size_t j=0;j<nz;++j) for(size_t i=0;i<nu;++i) {
        int id=-1;
        if(i==0 || i+1==nu) id=boundarySample(staged,p,i==0?0:1,p.Heights[j],insertions,report.BoundaryVerticesAdded);
        else if(j==0 || j+1==nz) id=boundarySample(staged,p,j==0?2:3,p.Angles[i],insertions,report.BoundaryVerticesAdded);
        else {
          const double angle=p.AngleStart+p.Angles[i];
          const PointD position=p.Origin+p.Axis*p.Heights[j]+p.X*(p.Radius*std::cos(angle))+p.Y*(p.Radius*std::sin(angle));
          id=staged.addVertex(toFloat(position),p.Patch,VertexConstraint::Surface);
        }
        staged.targetLength[id]=float(p.Target); ids[j*nu+i]=id;
      }
      grids.push_back(std::move(ids));
    }
    // The reference contains exactly the same subdivided shared polylines, but
    // retains original surface triangles (only their boundaries are split).
    // This preserves the existing CAD validator instead of weakening it.
    SemanticMesh refinedReference=refineSourceBoundaryFaces(source,staged,insertions);
    SemanticMesh output=refinedReference;
    for(int f=0;f<output.faceCount();++f)
      for(const auto &p:plans) if(output.facePatchId[f]==p.Patch) {output.killFace(f);break;}
    for(size_t k=0;k<plans.size();++k) {
      const auto &p=plans[k];const auto &ids=grids[k];
      const size_t nu=p.Angles.size(),nz=p.Heights.size();
      CylinderFilletPatchReport r;
      r.PatchId=p.Patch;r.TargetLength=float(p.Target);r.ArcSegments=uint32_t(nu-1);r.AxialSegments=uint32_t(nz-1);
      r.ArcStep=float(p.Radius*p.Width/(nu-1));r.AxialStep=float((p.HighZ-p.LowZ)/(nz-1));r.QualityMin=1;
      double qualitySum=0;
      const auto add=[&](int a,int b,int c) {
        if(p.Orientation<0) std::swap(b,c);
        const float q=triangleQuality(output.position(a),output.position(b),output.position(c));
        if(!(q>0.1f)) throw std::runtime_error("fillet sampling incompatible with preserved boundary knots");
        output.addFace(a,b,c,p.Patch,PatchType::Cylinder);r.QualityMin=std::min(r.QualityMin,q);qualitySum+=q;++r.InitialFaces;
      };
      for(size_t j=0;j+1<nz;++j) for(size_t i=0;i+1<nu;++i) {
        const int a=ids[j*nu+i],b=ids[j*nu+i+1],c=ids[(j+1)*nu+i+1],d=ids[(j+1)*nu+i];
        if((i+j)&1) {add(a,b,d);add(b,c,d);} else {add(a,b,c);add(a,c,d);}
      }
      r.QualityMean=float(qualitySum/r.InitialFaces);report.Patches.push_back(r);
      report.PatchTargetLengths[p.Patch]=float(p.Target);
    }
    output.rebuildTopology();output.compact();
    std::string issue;
    if(!output.validate(&issue)) throw std::runtime_error("fillet initial topology: "+issue);
    RemeshReport validation;
    if(!validatePartitionOutput(refinedReference,output,config,validation,&issue))
      throw std::runtime_error("fillet initialization violates CAD constraints: "+issue);
    report.Initialized=uint32_t(plans.size());mesh=std::move(output);return true;
  } catch(const std::exception &e) { if(error) *error=e.what(); return false; }
}
} // namespace cad_adaptive
