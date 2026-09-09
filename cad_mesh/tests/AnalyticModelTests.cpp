#include "CadMesh/SurfaceFitting.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
using namespace CadMesh;
constexpr double Pi = 3.14159265358979323846;
void Check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
const Vec3 axis = Normalize({1, 2, 3});
const Vec3 u = Normalize(Cross(axis, {0, 1, 0}));
const Vec3 v = Normalize(Cross(axis, u));
Point3 Transform(Vec3 p, double scale, Vec3 offset) {
  return ToPoint(Add(offset, Mul(Add(Add(Mul(u,p[0]),Mul(v,p[1])),Mul(axis,p[2])),scale)));
}
template<typename Sample>
TriangleSoup Grid(Sample sample, int nu, int nv, double scale, Vec3 offset, bool uneven) {
  TriangleSoup soup;
  for (int j=0;j<=nv;++j)
    for (int i=0;i<=nu;++i) {
      double a=double(i)/nu,b=double(j)/nv;
      if(uneven) {a=std::pow(a,1.7);b=std::pow(b,1.4);}
      soup.Vertices.push_back(Transform(sample(a,b),scale,offset));
    }
  for (int j=0;j<nv;++j)
    for (int i=0;i<nu;++i) {
      int a=j*(nu+1)+i,b=a+1,c=a+nu+1,d=c+1;
      if((i+j)%2) {soup.Triangles.push_back({{a,b,c}});soup.Triangles.push_back({{b,d,c}});}
      else {soup.Triangles.push_back({{a,b,d}});soup.Triangles.push_back({{a,d,c}});}
    }
  return soup;
}
std::vector<int> All(const MeshTopology &mesh) {
  std::vector<int> ids(mesh.getTriangles().size());std::iota(ids.begin(),ids.end(),0);return ids;
}
void CheckCone(double scale, Vec3 offset, bool uneven) {
  auto soup=Grid([](double a,double b)->Vec3 {
    double phi=-1.2+3.1*a,h=1+2*b,r=h*std::tan(.46);
    return {r*std::cos(phi),r*std::sin(phi),h};
  },32,10,scale,offset,uneven);
  MeshTopology mesh;Check(mesh.build(soup),"cone fixture build");
  ConeSurfaceFitter fitter;Check(fitter.fit(mesh,All(mesh)),"truncated cone must fit");
  auto p=std::get<ConeParameters>(fitter.getParameters());
  Check(std::abs(p.SemiAngle-.46)<2e-5,"cone angle inaccurate");
  Check(Distance(p.Axis.Origin,ToPoint(offset))<2e-4*scale,"cone apex inaccurate");
  Check(Dot(ToVec(p.Axis.Direction),axis)>.99999,"cone nappe axis incorrect");
  Check(fitter.computeMaxError()<2e-5*scale,"cone max residual inaccurate");
  auto selected=SurfaceModelSelector::fitBest(mesh,All(mesh),mesh.getResolution());
  Check(selected.Type==PatchSurfaceType::Cone,"truncated cone model selection");
}
void CheckTorus(double scale, Vec3 offset, bool uneven) {
  auto soup=Grid([](double a,double b)->Vec3 {
    double phi=-1.15+2.3*a,theta=-.8+2.1*b;
    double rho=3+.65*std::cos(theta);
    return {rho*std::cos(phi),rho*std::sin(phi),.65*std::sin(theta)};
  },28,16,scale,offset,uneven);
  MeshTopology mesh;Check(mesh.build(soup),"torus fixture build");
  TorusSurfaceFitter fitter;Check(fitter.fit(mesh,All(mesh)),"partial torus must fit");
  auto p=std::get<TorusParameters>(fitter.getParameters());
  Check(std::abs(p.MajorRadius-3*scale)<3e-4*scale,"torus major radius inaccurate");
  Check(std::abs(p.MinorRadius-.65*scale)<3e-4*scale,"torus minor radius inaccurate");
  Check(Distance(p.Axis.Origin,ToPoint(offset))<4e-4*scale,"torus center inaccurate");
  Check(std::abs(Dot(ToVec(p.Axis.Direction),axis))>.99999,"torus axis inaccurate");
  Check(fitter.computeMaxError()<2e-5*scale,"torus max residual inaccurate");
  auto selected=SurfaceModelSelector::fitBest(mesh,All(mesh),mesh.getResolution());
  Check(selected.Type==PatchSurfaceType::Torus,"partial torus model selection");
}
void CheckOtherModels(double scale, Vec3 offset) {
  auto soup=Grid([](double a,double b)->Vec3 {
    double phi=-1.2+2.4*a,h=-1+2*b;
    return {2*std::cos(phi),2*std::sin(phi),h};
  },24,7,scale,offset,true);
  MeshTopology mesh;Check(mesh.build(soup),"cylinder fixture build");
  CylinderSurfaceFitter cylinder;Check(cylinder.fit(mesh,All(mesh)),"weighted cylinder fit");
  auto p=std::get<CylinderParameters>(cylinder.getParameters());
  Check(std::abs(p.Radius-2*scale)<2e-5*scale,"weighted cylinder radius");
  Check(cylinder.computeMaxError()<2e-5*scale,"weighted cylinder max error");
  ConeSurfaceFitter cone;Check(!cone.fit(mesh,All(mesh)),"cylinder must not invent cone apex");
  soup=Grid([](double a,double b)->Vec3 {
    double phi=-1+2*a,theta=.6+1.2*b;
    return {2*std::sin(theta)*std::cos(phi),2*std::sin(theta)*std::sin(phi),2*std::cos(theta)};
  },24,12,scale,offset,true);
  Check(mesh.build(soup),"sphere fixture build");
  SphereSurfaceFitter sphere;Check(sphere.fit(mesh,All(mesh)),"weighted sphere fit");
  auto s=std::get<SphereParameters>(sphere.getParameters());
  Check(std::abs(s.Radius-2*scale)<2e-5*scale,"normalized sphere radius");
  Check(Distance(s.Center,ToPoint(offset))<2e-5*scale,"normalized sphere center");
  Check(sphere.computeMaxError()<2e-5*scale,"sphere max error");
}
void CheckDegenerateAndWeights() {
  auto soup=Grid([](double a,double b)->Vec3{return {3*a,b,0};},24,8,1,{0,0,0},true);
  MeshTopology mesh;Check(mesh.build(soup),"plane fixture build");
  ConeSurfaceFitter cone;TorusSurfaceFitter torus;
  Check(!cone.fit(mesh,All(mesh)),"plane must not fit cone");
  Check(!torus.fit(mesh,All(mesh)),"plane must not fit torus");
  auto noisy = Grid([](double a,double b)->Vec3 {
    return {3*a,b,1e-8*std::sin(20*a)*std::cos(13*b)};
  },24,8,1,{0,0,0},false);
  MeshTopology nearPlane;
  Check(nearPlane.build(noisy),"near-plane fixture build");
  Check(!cone.fit(nearPlane,All(nearPlane)),"near-planar noise must not establish a cone");
  Check(!torus.fit(nearPlane,All(nearPlane)),"near-planar noise must not establish a torus");
  // Two disconnected, differently triangulated parallel rectangles have equal
  // area. The area-weighted plane must stay halfway between them.
  auto upper=Grid([](double a,double b)->Vec3{return {3*a,b,1};},2,2,1,{0,0,0},false);
  int start=int(soup.Vertices.size());
  soup.Vertices.insert(soup.Vertices.end(),upper.Vertices.begin(),upper.Vertices.end());
  for(auto t:upper.Triangles){for(int &id:t)id+=start;soup.Triangles.push_back(t);}
  Check(mesh.build(soup),"area weighting fixture build");
  PlaneSurfaceFitter plane;Check(plane.fit(mesh,All(mesh)),"weighted plane fit");
  auto p=std::get<PlaneParameters>(plane.getParameters());
  // Here the smaller in-plane variance belongs to y; use weighted centroid,
  // which is defined independently of the orientation of the best plane.
  Check(Distance(p.Plane.Origin,Transform({1.5,.5,.5},1,{0,0,0}))<1e-12,"dense tessellation must not bias plane centroid: " + std::to_string(p.Plane.Origin.Z()));
  double actual=0;
  for(const auto &point:mesh.getVertices())
    actual=std::max(actual,std::abs(Dot(Sub(ToVec(point.Position),ToVec(p.Plane.Origin)),ToVec(p.Plane.Normal))));
  Check(std::abs(actual-plane.computeMaxError())<1e-12,"max error must include every vertex");
}

void CheckUnsampledResiduals() {
  auto soup=Grid([](double a,double b)->Vec3 {
    const double phi=-1.2+2.4*a,theta=.5+1.6*b;
    return {2*std::sin(theta)*std::cos(phi),2*std::sin(theta)*std::sin(phi),2*std::cos(theta)};
  },42,30,1,{0,0,0},false);
  // More vertices than the bounded nonlinear sample budget, including a real
  // geometric outlier on a small-area corner that must not escape Max.
  soup.Vertices.back() = ToPoint(Mul(ToVec(soup.Vertices.back()),1.015));
  MeshTopology mesh;Check(mesh.build(soup),"unsampled residual fixture build");
  Check(mesh.getVertices().size()>768,"residual fixture must exceed solver sample budget");
  SphereSurfaceFitter sphere;Check(sphere.fit(mesh,All(mesh)),"outlier sphere fit");
  const auto parameters=std::get<SphereParameters>(sphere.getParameters());
  double maximum=0,weightedSquared=0,area=0;
  for(const auto &triangle:mesh.getTriangles()) {
    area+=triangle.Area;
    for(int id:triangle.VertexIds) {
      const double residual=std::abs(Distance(mesh.getVertices()[id].Position,parameters.Center)-parameters.Radius);
      maximum=std::max(maximum,residual);
      weightedSquared+=triangle.Area/3*residual*residual;
    }
  }
  Check(maximum>.02,"outlier must remain measurable after fitting");
  Check(std::abs(maximum-sphere.computeMaxError())<1e-12,"reported Max must certify unsampled vertices");
  Check(std::abs(std::sqrt(weightedSquared/area)-sphere.computeRmsError())<1e-12,
        "reported RMS must use full area-weighted surface support");
}
} // namespace

void RunAnalyticModelTests() {
  CheckCone(1,{0,0,0},false);
  CheckCone(1e-4,{100,-200,300},true);
  CheckCone(1e4,{1e10,-2e10,3e10},true);
  CheckTorus(1,{0,0,0},false);
  CheckTorus(1e-4,{100,-200,300},true);
  CheckTorus(1e4,{1e10,-2e10,3e10},true);
  CheckOtherModels(1e-4,{100,-200,300});
  CheckOtherModels(1e4,{1e10,-2e10,3e10});
  CheckDegenerateAndWeights();
  CheckUnsampledResiduals();
}
