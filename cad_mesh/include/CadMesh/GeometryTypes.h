#pragma once
#include <vcg/space/point3.h>
#include <algorithm>
#include <array>
#include <cmath>

namespace CadMesh {
using Point3=vcg::Point3d; using Direction3=vcg::Point3d; using Vec3=std::array<double,3>;
struct AxisLine{Point3 Origin;Direction3 Direction{1,0,0};};
struct PlaneEquation{Point3 Origin;Direction3 Normal{0,0,1};};
inline Vec3 ToVec(const Point3&p){return{p.X(),p.Y(),p.Z()};}
inline Point3 ToPoint(const Vec3&p){return{p[0],p[1],p[2]};}
inline Direction3 ToDirection(const Vec3&p){const double n=std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);return n>1e-30?Direction3(p[0]/n,p[1]/n,p[2]/n):Direction3(1,0,0);}
inline Vec3 Add(const Vec3&a,const Vec3&b){return{a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
inline Vec3 Sub(const Vec3&a,const Vec3&b){return{a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline Vec3 Mul(const Vec3&a,double s){return{a[0]*s,a[1]*s,a[2]*s};}
inline double Dot(const Vec3&a,const Vec3&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline Vec3 Cross(const Vec3&a,const Vec3&b){return{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline double Norm(const Vec3&a){return std::sqrt(Dot(a,a));}
inline Vec3 Normalize(const Vec3&a,const Vec3&fallback={1,0,0}){const double n=Norm(a);return n>1e-30?Mul(a,1.0/n):fallback;}
inline double Clamp01(double x){return std::max(0.0,std::min(1.0,x));}
inline double Distance(const Point3&a,const Point3&b){return Norm(Sub(ToVec(a),ToVec(b)));}
}
