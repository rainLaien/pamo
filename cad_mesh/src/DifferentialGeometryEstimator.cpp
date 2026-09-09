#include "CadMesh/DifferentialGeometryEstimator.h"
#include <cmath>
#include <queue>
#include <set>

namespace CadMesh { namespace {
bool Solve6(double a[6][7],double x[6]){for(int c=0;c<6;++c){int pivot=c;for(int r=c+1;r<6;++r)if(std::abs(a[r][c])>std::abs(a[pivot][c]))pivot=r;if(std::abs(a[pivot][c])<1e-18)return false;for(int j=c;j<7;++j)std::swap(a[c][j],a[pivot][j]);double d=a[c][c];for(int j=c;j<7;++j)a[c][j]/=d;for(int r=0;r<6;++r)if(r!=c){double f=a[r][c];for(int j=c;j<7;++j)a[r][j]-=f*a[c][j];}}for(int i=0;i<6;++i)x[i]=a[i][6];return true;}
}
void DifferentialGeometryEstimator::compute(int rings){for(int i=0;i<int(mMesh.getVertices().size());++i)mMesh.getVertices()[i].Geometry=FitVertex(i,rings);}
DifferentialGeometry DifferentialGeometryEstimator::FitVertex(int id,int rings)const{
    DifferentialGeometry result;const auto&vertices=mMesh.getVertices();const Vec3 origin=ToVec(vertices[id].Position),n=Normalize(ToVec(vertices[id].Normal));Vec3 seed=std::abs(n[0])<0.8?Vec3{1,0,0}:Vec3{0,1,0};Vec3 u=Normalize(Cross(n,seed)),v=Normalize(Cross(n,u));
    std::set<int> neighborhood{id},frontier{id};for(int ring=0;ring<rings;++ring){std::set<int>next;for(int p:frontier)for(int q:vertices[p].NeighborVertexIds)if(neighborhood.insert(q).second)next.insert(q);frontier.swap(next);}if(neighborhood.size()<6)return result;
    double normal[6][6]{};double rhs[6]{};double radius=0;std::vector<std::array<double,3>> samples;
    for(int p:neighborhood){Vec3 d=Sub(ToVec(vertices[p].Position),origin);double x=Dot(d,u),y=Dot(d,v),z=Dot(d,n);double row[6]{x*x,x*y,y*y,x,y,1};for(int i=0;i<6;++i){rhs[i]+=row[i]*z;for(int j=0;j<6;++j)normal[i][j]+=row[i]*row[j];}samples.push_back({x,y,z});radius=std::max(radius,std::sqrt(x*x+y*y));}
    double augmented[6][7]{};for(int i=0;i<6;++i){for(int j=0;j<6;++j)augmented[i][j]=normal[i][j];augmented[i][6]=rhs[i];}double c[6]{};if(!Solve6(augmented,c))return result;
    const double h00=2*c[0],h01=c[1],h11=2*c[2],trace=h00+h11,disc=std::sqrt(std::max(0.0,(h00-h11)*(h00-h11)+4*h01*h01));result.K1=0.5*(trace-disc);result.K2=0.5*(trace+disc);result.MeanCurvature=0.5*(result.K1+result.K2);result.GaussianCurvature=result.K1*result.K2;
    Vec3 d1=u,d2=v;if(std::abs(h01)>1e-14){double ex=h01,ey=result.K1-h00;double len=std::sqrt(ex*ex+ey*ey);ex/=len;ey/=len;d1=Normalize(Add(Mul(u,ex),Mul(v,ey)));d2=Normalize(Cross(n,d1));}result.PrincipalDirection1=ToDirection(d1);result.PrincipalDirection2=ToDirection(d2);
    double squared=0;for(const auto&s:samples){double predicted=c[0]*s[0]*s[0]+c[1]*s[0]*s[1]+c[2]*s[1]*s[1]+c[3]*s[0]+c[4]*s[1]+c[5];double e=predicted-s[2];squared+=e*e;}double rms=std::sqrt(squared/samples.size());double tol=mMesh.getResolution().FittingTolerance;result.Confidence=Clamp01((samples.size()-5)/15.0)*std::exp(-rms/std::max(tol,1e-30))*Clamp01(radius/std::max(mMesh.getResolution().MedianEdgeLength,1e-30));return result;
}
}
