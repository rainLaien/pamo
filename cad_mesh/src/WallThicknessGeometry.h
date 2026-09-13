#pragma once
// Private geometry for the wall-thickness solver. Standard C++ only.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace CadMesh { namespace ThicknessGeometry {
struct Vector {
  double v[3]{};
  Vector()=default;
  Vector(double x,double y,double z):v{x,y,z}{}
  double& operator[](int i){return v[i];}
  double operator[](int i)const{return v[i];}
  double X()const{return v[0];} double Y()const{return v[1];} double Z()const{return v[2];}
  Vector operator+(const Vector& b)const{return {v[0]+b[0],v[1]+b[1],v[2]+b[2]};}
  Vector operator-(const Vector& b)const{return {v[0]-b[0],v[1]-b[1],v[2]-b[2]};}
  Vector operator-()const{return {-v[0],-v[1],-v[2]};}
  Vector operator*(double s)const{return {v[0]*s,v[1]*s,v[2]*s};}
  Vector operator/(double s)const{return *this*(1/s);}
  Vector& operator+=(const Vector& b){*this=*this+b;return *this;}
  Vector& operator-=(const Vector& b){*this=*this-b;return *this;}
  Vector& operator*=(double s){*this=*this*s;return *this;}
  Vector& operator/=(double s){*this=*this/s;return *this;}
  double dot(const Vector& b)const{return v[0]*b[0]+v[1]*b[1]+v[2]*b[2];}
  double operator*(const Vector& b)const{return dot(b);}
  Vector operator^(const Vector& b)const{return {v[1]*b[2]-v[2]*b[1],v[2]*b[0]-v[0]*b[2],v[0]*b[1]-v[1]*b[0]};}
  double SquaredNorm()const{return dot(*this);}
  double Norm()const{return std::sqrt(SquaredNorm());}
  Vector normalized()const{const double n=Norm();return n>0?*this/n:Vector{};}
  Vector& Normalize(){*this=normalized();return *this;}
};
inline Vector operator*(double s,const Vector& v){return v*s;}
struct Line {
  Vector origin,direction;
  Line(Vector p,Vector d):origin(p),direction(d){}
  const Vector& Origin()const{return origin;}
  const Vector& Direction()const{return direction;}
  Vector P(double t)const{return origin+direction*t;}
};
struct Box {
  Vector min{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
  Vector max{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()};
  void add(Vector p){for(int k=0;k<3;++k){min[k]=std::min(min[k],p[k]);max[k]=std::max(max[k],p[k]);}}
  double DimX()const{return max[0]-min[0];}
  double DimY()const{return max[1]-min[1];}
  double DimZ()const{return max[2]-min[2];}
  double Diag()const{return (max-min).Norm();}
  bool IsIn(Vector p)const{
    for(int k=0;k<3;++k)if(p[k]<min[k]||p[k]>max[k])return false;
    return true;
  }
};
struct Face {
  std::array<Vector,3> points;
  Vector normal;
  int id=-1;
  const Vector& cP(int i)const{return points[i];}
  const Vector& cN()const{return normal;}
  const Vector& N()const{return normal;}
};
struct Mesh {std::vector<Face> face;Box bbox;};
struct Node {
  Box box;
  Node* children[2]{};
  Face** oBegin=nullptr;Face** oEnd=nullptr;
  bool IsLeaf()const{return children[0]==nullptr;}
};
class Bvh {
  std::vector<Node> nodes;
  std::vector<Face*> ordered;
  Node* build(std::size_t first,std::size_t last){
    const auto index=nodes.size();nodes.emplace_back();
    Box bounds,centers;
    for(auto i=first;i<last;++i){
      const auto& f=*ordered[i];for(const auto& p:f.points)bounds.add(p);
      centers.add((f.points[0]+f.points[1]+f.points[2])/3);
    }
    nodes[index].box=bounds;
    if(last-first<=2){nodes[index].oBegin=ordered.data()+first;nodes[index].oEnd=ordered.data()+last;return &nodes[index];}
    int axis=0;for(int k=1;k<3;++k)if(centers.max[k]-centers.min[k]>centers.max[axis]-centers.min[axis])axis=k;
    const auto middle=first+(last-first)/2;
    std::nth_element(ordered.begin()+first,ordered.begin()+middle,ordered.begin()+last,[axis](const Face* a,const Face* b){
      const double x=(a->points[0][axis]+a->points[1][axis]+a->points[2][axis])/3;
      const double y=(b->points[0][axis]+b->points[1][axis]+b->points[2][axis])/3;
      return x==y?a->id<b->id:x<y;
    });
    nodes[index].children[0]=build(first,middle);nodes[index].children[1]=build(middle,last);
    return &nodes[index];
  }
public:
  Node* pRoot=nullptr;
  Bvh()=default;Bvh(const Bvh&)=delete;Bvh& operator=(const Bvh&)=delete;
  void initialize(Mesh& mesh){
    ordered.clear();nodes.clear();pRoot=nullptr;
    for(auto& face:mesh.face)ordered.push_back(&face);
    nodes.reserve(ordered.size()*2);
    if(!ordered.empty())pRoot=build(0,ordered.size());
  }
  bool Empty()const{return pRoot==nullptr;}
  const Bvh& Tree()const{return *this;}
};
inline bool rayBoxEntry(const Box& box,const Line& ray,double& entry){
  double lo=0,hi=std::numeric_limits<double>::infinity();
  for(int k=0;k<3;++k){
    const double o=ray.origin[k],d=ray.direction[k];
    if(d==0){if(o<box.min[k]||o>box.max[k])return false;continue;}
    double a=(box.min[k]-o)/d,b=(box.max[k]-o)/d;if(a>b)std::swap(a,b);
    lo=std::max(lo,a);hi=std::min(hi,b);if(lo>hi)return false;
  }
  entry=lo;return true;
}
} }
