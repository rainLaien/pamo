#include "CadMesh/BoundaryScoreCalculator.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <unordered_set>

namespace CadMesh { namespace {
double Angle(const Direction3& a,const Direction3& b){return std::acos(std::max(-1.0,std::min(1.0,Dot(ToVec(a),ToVec(b)))));}
double Median(std::vector<double> values){if(values.empty())return 0;std::sort(values.begin(),values.end());return values[values.size()/2];}
bool Solve6(double a[6][7],double x[6]){for(int c=0;c<6;++c){int p=c;for(int r=c+1;r<6;++r)if(std::abs(a[r][c])>std::abs(a[p][c]))p=r;if(std::abs(a[p][c])<1e-18)return false;for(int j=c;j<7;++j)std::swap(a[c][j],a[p][j]);double d=a[c][c];for(int j=c;j<7;++j)a[c][j]/=d;for(int r=0;r<6;++r)if(r!=c){double f=a[r][c];for(int j=c;j<7;++j)a[r][j]-=f*a[c][j];}}for(int i=0;i<6;++i)x[i]=a[i][6];return true;}
struct CurvatureDescriptor{double K1=0,K2=0,H=0,G=0,Confidence=0;};
CurvatureDescriptor TriangleCurvature(const MeshTopology& mesh,int triangleId){CurvatureDescriptor d;const auto&t=mesh.getTriangles()[triangleId];for(int v:t.VertexIds){const auto&g=mesh.getVertices()[v].Geometry;d.K1+=g.K1;d.K2+=g.K2;d.H+=g.MeanCurvature;d.G+=g.GaussianCurvature;d.Confidence+=g.Confidence;}d.K1/=3;d.K2/=3;d.H/=3;d.G/=3;d.Confidence/=3;return d;}
double DescriptorDistance(const CurvatureDescriptor&a,const CurvatureDescriptor&b,double scale){double linear=std::abs(a.K1-b.K1)+std::abs(a.K2-b.K2)+2*std::abs(a.H-b.H);double gaussian=std::sqrt(std::abs(a.G-b.G));return Clamp01((.2*linear+.15*gaussian)/std::max(scale,1e-30))*std::sqrt(Clamp01(a.Confidence*b.Confidence));}
double TriangleAspect(const MeshTopology&mesh,int id){const auto&t=mesh.getTriangles()[id];double sum=0,longest=0;for(int k=0;k<3;++k){double l=Distance(mesh.getVertices()[t.VertexIds[k]].Position,mesh.getVertices()[t.VertexIds[(k+1)%3]].Position);sum+=l*l;longest=std::max(longest,l);}if(t.Area<=1e-30)return 1e6;return longest*longest/(2.0*std::sqrt(3.0)*t.Area)+sum/(12.0*std::sqrt(3.0)*t.Area);}
std::vector<int> SideRing(const MeshTopology &mesh, int seed, int blocked,
                          const MeshEdge &candidate) {
    const auto &triangles = mesh.getTriangles();
    Vec3 a = ToVec(mesh.getVertices()[candidate.Vertex0].Position);
    Vec3 b = ToVec(mesh.getVertices()[candidate.Vertex1].Position);
    Vec3 midpoint = Mul(Add(a, b), .5), tangent = Normalize(Sub(b, a));
    Vec3 bisector = Normalize(Add(ToVec(triangles[seed].Normal),
                                  ToVec(triangles[blocked].Normal)),
                               ToVec(triangles[seed].Normal));
    Vec3 side = Normalize(Cross(tangent, bisector));
    if (Dot(Sub(ToVec(triangles[seed].Centroid), midpoint), side) < 0)
        side = Mul(side, -1);
    const double tolerance = 1e-10 * std::max(Norm(Sub(b, a)),
                                              mesh.getResolution().MedianEdgeLength);
    // Clip the neighborhood to the local half-surface separated by the
    // candidate edge and the normal bisector. Blocking only the opposite
    // triangle after a BFS still includes faces reached through that triangle
    // (or by walking around the edge endpoints).
    auto onSide = [&](int triangleId) {
        for (int vertexId : triangles[triangleId].VertexIds)
            if (Dot(Sub(ToVec(mesh.getVertices()[vertexId].Position), midpoint),
                    side) < -tolerance)
                return false;
        return true;
    };
    std::vector<int> result;
    std::unordered_set<int> seen{seed};
    std::queue<std::pair<int, int>> frontier;
    frontier.push({seed, 0});
    while (!frontier.empty()) {
        auto current = frontier.front();
        frontier.pop();
        result.push_back(current.first);
        if (current.second >= 2)
            continue;
        for (int edgeId : triangles[current.first].EdgeIds) {
            const auto &edge = mesh.getEdges()[edgeId];
            if (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature)
                continue;
            for (int neighbor : edge.IncidentTriangleIds)
                if (neighbor != blocked && !seen.count(neighbor) && onSide(neighbor)) {
                    seen.insert(neighbor);
                    frontier.push({neighbor, current.second + 1});
                }
        }
    }
    return result;
}
}

double BoundaryScoreCalculator::LocalQuadricRms(const std::vector<int>&triangleIds)const{
    std::set<int> unique;Vec3 origin{0,0,0},normal{0,0,0};double area=0;
    for(int ti:triangleIds){if(ti<0||ti>=int(mMesh.getTriangles().size()))continue;const auto&t=mMesh.getTriangles()[ti];for(int v:t.VertexIds)unique.insert(v);origin=Add(origin,Mul(ToVec(t.Centroid),t.Area));normal=Add(normal,Mul(ToVec(t.Normal),t.Area));area+=t.Area;}
    if(unique.size()<6||area<=1e-30)return mMesh.getResolution().FittingTolerance*10;
    origin=Mul(origin,1.0/area);normal=Normalize(normal);Vec3 seed=std::abs(normal[0])<.8?Vec3{1,0,0}:Vec3{0,1,0},u=Normalize(Cross(normal,seed)),v=Normalize(Cross(normal,u));
    double a[6][7]{};std::vector<std::array<double,3>>samples;samples.reserve(unique.size());
    for(int id:unique){Vec3 d=Sub(ToVec(mMesh.getVertices()[id].Position),origin);double x=Dot(d,u),y=Dot(d,v),z=Dot(d,normal),row[6]{x*x,x*y,y*y,x,y,1};for(int i=0;i<6;++i){a[i][6]+=row[i]*z;for(int j=0;j<6;++j)a[i][j]+=row[i]*row[j];}samples.push_back({x,y,z});}
    double c[6]{};if(!Solve6(a,c))return mMesh.getResolution().FittingTolerance*10;double sq=0;for(const auto&s:samples){double z=c[0]*s[0]*s[0]+c[1]*s[0]*s[1]+c[2]*s[1]*s[1]+c[3]*s[0]+c[4]*s[1]+c[5];sq+=(z-s[2])*(z-s[2]);}return std::sqrt(sq/samples.size());
}

void BoundaryScoreCalculator::compute(){
    const auto&r=mMesh.getResolution();auto&edges=mMesh.getEdges();const double pi=std::acos(-1.0),smoothLimit=15*pi/180.0;
    // A CAD fillet is often tessellated as a sequence of long two-triangle
    // strips.  The normal rotation is constant inside the fillet and nearly
    // zero on a plane.  Comparing the per-face low-angle rotation is much
    // more stable here than differentiating noisy point curvatures.
    std::vector<int>parent(mMesh.getTriangles().size());for(int i=0;i<int(parent.size());++i)parent[i]=i;auto find=[&](int x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;};auto unite=[&](int a,int b){a=find(a);b=find(b);if(a!=b)parent[b]=a;};const double coplanarTolerance=std::max(.1*r.AngularTolerance,.02*pi/180.0);
    for(const auto&e:edges)if(e.Triangle0>=0&&e.Triangle1>=0&&!e.IsNonManifold&&Angle(mMesh.getTriangles()[e.Triangle0].Normal,mMesh.getTriangles()[e.Triangle1].Normal)<coplanarTolerance)unite(e.Triangle0,e.Triangle1);
    for(int i=0;i<int(parent.size());++i)parent[i]=find(i);
    std::vector<double>cellRotation(parent.size(),0);std::vector<int>cellRotationCount(parent.size(),0);
    for(const auto&e:edges)if(e.Triangle0>=0&&e.Triangle1>=0&&!e.IsNonManifold){double angle=Angle(mMesh.getTriangles()[e.Triangle0].Normal,mMesh.getTriangles()[e.Triangle1].Normal);int a=parent[e.Triangle0],b=parent[e.Triangle1];if(a!=b&&angle<smoothLimit){cellRotation[a]+=angle;cellRotation[b]+=angle;++cellRotationCount[a];++cellRotationCount[b];}}
    for(size_t i=0;i<cellRotation.size();++i)if(cellRotationCount[i])cellRotation[i]/=cellRotationCount[i];
    for(auto&e:edges){BoundaryEvidence b;if(e.IsBoundary||e.IsNonManifold||e.Triangle0<0||e.Triangle1<0){b.NormalDiscontinuity=1;b.SurfaceFitDiscontinuity=1;b.FinalScore=1;e.Evidence=b;e.BoundaryScore=1;continue;}
        const auto&t0=mMesh.getTriangles()[e.Triangle0];const auto&t1=mMesh.getTriangles()[e.Triangle1];double angle=Angle(t0.Normal,t1.Normal);b.NormalDiscontinuity=Clamp01(angle/std::max(6*r.AngularTolerance,1e-9));
        auto c0=TriangleCurvature(mMesh,e.Triangle0),c1=TriangleCurvature(mMesh,e.Triangle1);b.CurvatureDiscontinuity=DescriptorDistance(c0,c1,std::max(r.CurvatureTolerance,.1*(std::abs(c0.H)+std::abs(c1.H))));
        auto gradient=[&](int triangle,int opposite,const CurvatureDescriptor&center){double sum=0;int count=0;for(int n:mMesh.getTriangleNeighbors(triangle))if(n!=opposite){sum+=std::abs(center.H-TriangleCurvature(mMesh,n).H);++count;}return count?sum/count:0;};
        double g0=gradient(e.Triangle0,e.Triangle1,c0),g1=gradient(e.Triangle1,e.Triangle0,c1);double edgeLength=Distance(mMesh.getVertices()[e.Vertex0].Position,mMesh.getVertices()[e.Vertex1].Position);
        // Both neighborhood curvature differences and CurvatureTolerance
        // have units 1/length. Their ratio is invariant under unit changes.
        double quadricGradient=Clamp01(std::abs(g0-g1)/std::max(r.CurvatureTolerance,1e-30))*std::sqrt(Clamp01(c0.Confidence*c1.Confidence));double rotationJump=parent[e.Triangle0]==parent[e.Triangle1]?0:std::abs(cellRotation[parent[e.Triangle0]]-cellRotation[parent[e.Triangle1]]);double discreteGradient=Clamp01(rotationJump/std::max(r.AngularTolerance,.25*pi/180.0));b.CurvatureGradient=std::max(quadricGradient,discreteGradient);
        auto localLengths=[&](int ti){std::vector<double>x;for(int edgeId:mMesh.getTriangles()[ti].EdgeIds){const auto&edge=edges[edgeId];x.push_back(Distance(mMesh.getVertices()[edge.Vertex0].Position,mMesh.getVertices()[edge.Vertex1].Position));}return Median(x);};double l0=localLengths(e.Triangle0),l1=localLengths(e.Triangle1),density=std::abs(std::log(std::max(l0,1e-30)/std::max(l1,1e-30))),aspect=std::abs(std::log(std::max(TriangleAspect(mMesh,e.Triangle0),1e-30)/std::max(TriangleAspect(mMesh,e.Triangle1),1e-30)));b.TessellationEvidence=Clamp01(.55*density+.25*aspect);
        // The three local quadric solves dominate runtime.  Evaluate them on
        // edges that already have another weak cue; completely uniform
        // interiors cannot become a boundary from a combined-fit test alone.
        double preliminary=mConfig.NormalWeight*b.NormalDiscontinuity+mConfig.CurvatureWeight*b.CurvatureDiscontinuity+mConfig.GradientWeight*b.CurvatureGradient+mConfig.TessellationWeight*b.TessellationEvidence;
        if(preliminary>.18){auto left=SideRing(mMesh,e.Triangle0,e.Triangle1,e),right=SideRing(mMesh,e.Triangle1,e.Triangle0,e),combined=left;combined.insert(combined.end(),right.begin(),right.end());std::sort(combined.begin(),combined.end());combined.erase(std::unique(combined.begin(),combined.end()),combined.end());double le=LocalQuadricRms(left),re=LocalQuadricRms(right),ce=LocalQuadricRms(combined),base=std::max({le,re,.25*r.FittingTolerance});b.SurfaceFitDiscontinuity=Clamp01((ce-base)/(base+2*r.FittingTolerance));}
        b.FinalScore=Clamp01(mConfig.NormalWeight*b.NormalDiscontinuity+mConfig.CurvatureWeight*b.CurvatureDiscontinuity+mConfig.GradientWeight*b.CurvatureGradient+mConfig.SurfaceFitWeight*b.SurfaceFitDiscontinuity+mConfig.TessellationWeight*b.TessellationEvidence);if(angle>std::max(12*r.AngularTolerance,30*pi/180.0))b.FinalScore=std::max(b.FinalScore,.92);if(rotationJump>.7*r.AngularTolerance&&edgeLength>4*r.MedianEdgeLength)b.FinalScore=std::max(b.FinalScore,.82);e.Evidence=b;e.BoundaryScore=b.FinalScore;
    }
}
}
