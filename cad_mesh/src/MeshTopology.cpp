#include "CadMesh/MeshTopology.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <unordered_set>

namespace CadMesh { namespace {
struct Cell { long long X,Y,Z; bool operator==(const Cell&o)const{return X==o.X&&Y==o.Y&&Z==o.Z;} };
struct CellHash { size_t operator()(const Cell&c)const{size_t h=std::hash<long long>{}(c.X);h^=std::hash<long long>{}(c.Y)+0x9e3779b9+(h<<6)+(h>>2);h^=std::hash<long long>{}(c.Z)+0x9e3779b9+(h<<6)+(h>>2);return h;} };
using EdgeKey=std::uint64_t;
EdgeKey Key(int a,int b){if(a>b)std::swap(a,b);return(std::uint64_t(std::uint32_t(a))<<32)|std::uint32_t(b);}
std::array<int,3> Sorted(std::array<int,3>a){std::sort(a.begin(),a.end());return a;}
}

bool MeshTopology::build(const TriangleSoup&soup){
    mOriginalSoup=soup;mVertices.clear();mEdges.clear();mTriangles.clear();mCleanup={};
    mCleanup.InputTriangles=int(soup.Triangles.size());
    if(soup.Vertices.empty()||soup.Triangles.empty())return false;
    EstimateRawResolution();WeldVertices();BuildTriangles();BuildEdges();ComputeNormals();
    mCleanup.OutputTriangles=int(mTriangles.size());return !mTriangles.empty();
}
bool MeshTopology::buildIndexed(const TriangleSoup &soup, const MeshResolutionInfo &resolution) {
    if (soup.Vertices.empty() || soup.Triangles.empty() ||
        soup.Vertices.size() > size_t(std::numeric_limits<int>::max()) ||
        soup.Triangles.size() > size_t(std::numeric_limits<int>::max())) return false;
    mOriginalSoup = soup; mResolution = resolution; mCleanup = {};
    mVertices.clear(); mTriangles.clear(); mEdges.clear();
    mCleanup.InputTriangles = int(soup.Triangles.size());
    mVertices.resize(soup.Vertices.size());
    mOriginalToWelded.resize(soup.Vertices.size());
    for (size_t i = 0; i < soup.Vertices.size(); ++i) {
        for (int k = 0; k < 3; ++k) if (!std::isfinite(soup.Vertices[i][k])) return false;
        mVertices[i].Position = soup.Vertices[i];
        mVertices[i].OriginalVertexIds.push_back(int(i));
        mOriginalToWelded[i] = int(i);
    }
    BuildTriangles();
    // Reject damaged snapshots rather than silently changing face ids.
    if (mTriangles.size() != soup.Triangles.size()) return false;
    for (size_t i = 0; i < mOriginalToTriangle.size(); ++i)
        if (mOriginalToTriangle[i] != int(i)) return false;
    BuildEdges(); ComputeNormals();
    mCleanup.OutputTriangles = int(mTriangles.size());
    return true;
}
void MeshTopology::EstimateRawResolution(){
    Vec3 lo=ToVec(mOriginalSoup.Vertices.front()),hi=lo;std::vector<double> lengths;
    for(const auto&p:mOriginalSoup.Vertices)for(int k=0;k<3;++k){lo[k]=std::min(lo[k],p[k]);hi[k]=std::max(hi[k],p[k]);}
    for(const auto&t:mOriginalSoup.Triangles)for(int k=0;k<3;++k){int a=t[k],b=t[(k+1)%3];if(a>=0&&b>=0&&a<int(mOriginalSoup.Vertices.size())&&b<int(mOriginalSoup.Vertices.size())){double l=Distance(mOriginalSoup.Vertices[a],mOriginalSoup.Vertices[b]);if(l>0&&std::isfinite(l))lengths.push_back(l);}}
    std::sort(lengths.begin(),lengths.end());mResolution.BoundingBoxDiagonal=Norm(Sub(hi,lo));
    if(lengths.empty())return;
    mResolution.MinEdgeLength=lengths.front();mResolution.MedianEdgeLength=lengths[lengths.size()/2];
    double sum=0;for(double l:lengths)sum+=l;mResolution.MeanEdgeLength=sum/lengths.size();
    mResolution.WeldTolerance=std::max(mResolution.BoundingBoxDiagonal*1e-10,mResolution.MedianEdgeLength*1e-6);
    mResolution.FittingTolerance=std::max(mResolution.BoundingBoxDiagonal*1e-7,mResolution.MedianEdgeLength*2e-3);
    mResolution.CurvatureTolerance=1.0/std::max(mResolution.BoundingBoxDiagonal,mResolution.MedianEdgeLength*100.0);
    const double pi=std::acos(-1.0);
    mResolution.AngularTolerance=std::max(0.5*pi/180.0,std::min(5.0*pi/180.0,mResolution.FittingTolerance/std::max(mResolution.MedianEdgeLength,1e-30)));
}
void MeshTopology::WeldVertices(){
    const double tol=std::max(mResolution.WeldTolerance,1e-15),inv=1.0/tol;mOriginalToWelded.assign(mOriginalSoup.Vertices.size(),-1);
    std::unordered_map<Cell,std::vector<int>,CellHash> grid;
    for(int original=0;original<int(mOriginalSoup.Vertices.size());++original){const auto&p=mOriginalSoup.Vertices[original];Cell c{static_cast<long long>(std::floor(p.X()*inv)),static_cast<long long>(std::floor(p.Y()*inv)),static_cast<long long>(std::floor(p.Z()*inv))};int found=-1;
        for(int dx=-1;dx<=1&&found<0;++dx)for(int dy=-1;dy<=1&&found<0;++dy)for(int dz=-1;dz<=1&&found<0;++dz){auto it=grid.find({c.X+dx,c.Y+dy,c.Z+dz});if(it==grid.end())continue;for(int id:it->second)if(Distance(mVertices[id].Position,p)<=tol){found=id;break;}}
        if(found<0){found=int(mVertices.size());MeshVertex v;v.Position=p;mVertices.push_back(v);grid[c].push_back(found);}mVertices[found].OriginalVertexIds.push_back(original);mOriginalToWelded[original]=found;
    }
}
void MeshTopology::BuildTriangles(){
    mOriginalToTriangle.assign(mOriginalSoup.Triangles.size(),-1);std::set<std::array<int,3>> seen;
    const double areaTol=std::max(mResolution.BoundingBoxDiagonal*mResolution.BoundingBoxDiagonal*1e-24,1e-30);
    for(int original=0;original<int(mOriginalSoup.Triangles.size());++original){auto raw=mOriginalSoup.Triangles[original];if(*std::min_element(raw.begin(),raw.end())<0||*std::max_element(raw.begin(),raw.end())>=int(mOriginalToWelded.size())){++mCleanup.DegenerateTriangles;continue;}std::array<int,3>w{mOriginalToWelded[raw[0]],mOriginalToWelded[raw[1]],mOriginalToWelded[raw[2]]};if(w[0]==w[1]||w[1]==w[2]||w[2]==w[0]){++mCleanup.DegenerateTriangles;continue;}if(!seen.insert(Sorted(w)).second){++mCleanup.DuplicateTriangles;continue;}
        Vec3 a=ToVec(mVertices[w[0]].Position),b=ToVec(mVertices[w[1]].Position),c=ToVec(mVertices[w[2]].Position),cross=Cross(Sub(b,a),Sub(c,a));double twice=Norm(cross);if(twice<=areaTol){++mCleanup.DegenerateTriangles;continue;}MeshTriangle t;t.VertexIds=w;t.Area=0.5*twice;t.Normal=ToDirection(cross);t.Centroid=ToPoint(Mul(Add(Add(a,b),c),1.0/3.0));int id=int(mTriangles.size());mTriangles.push_back(t);mOriginalToTriangle[original]=id;for(int v:w)mVertices[v].IncidentTriangleIds.push_back(id);
    }
}
void MeshTopology::BuildEdges(){
    std::unordered_map<EdgeKey,int> lookup;
    for(int ti=0;ti<int(mTriangles.size());++ti)for(int k=0;k<3;++k){int a=mTriangles[ti].VertexIds[k],b=mTriangles[ti].VertexIds[(k+1)%3];EdgeKey key=Key(a,b);auto it=lookup.find(key);int id;if(it==lookup.end()){id=int(mEdges.size());MeshEdge e;e.Vertex0=std::min(a,b);e.Vertex1=std::max(a,b);mEdges.push_back(e);lookup[key]=id;mVertices[a].NeighborVertexIds.push_back(b);mVertices[b].NeighborVertexIds.push_back(a);}else id=it->second;mEdges[id].IncidentTriangleIds.push_back(ti);mTriangles[ti].EdgeIds[k]=id;}
    for(auto&e:mEdges){std::sort(e.IncidentTriangleIds.begin(),e.IncidentTriangleIds.end());e.Triangle0=e.IncidentTriangleIds.empty()?-1:e.IncidentTriangleIds[0];e.Triangle1=e.IncidentTriangleIds.size()<2?-1:e.IncidentTriangleIds[1];e.IsBoundary=e.IncidentTriangleIds.size()==1;e.IsNonManifold=e.IncidentTriangleIds.size()>2;if(e.IsBoundary)++mCleanup.BoundaryEdges;if(e.IsNonManifold)++mCleanup.NonManifoldEdges;}
    for(auto&v:mVertices){std::sort(v.NeighborVertexIds.begin(),v.NeighborVertexIds.end());v.NeighborVertexIds.erase(std::unique(v.NeighborVertexIds.begin(),v.NeighborVertexIds.end()),v.NeighborVertexIds.end());}
    if(mCleanup.NonManifoldEdges)mCleanup.Warnings.push_back("non-manifold edges were retained as certain boundaries");
}
void MeshTopology::ComputeNormals(){
    std::vector<Vec3>sums(mVertices.size(),{0,0,0});for(const auto&t:mTriangles){Vec3 weighted=Mul(ToVec(t.Normal),t.Area);for(int v:t.VertexIds)sums[v]=Add(sums[v],weighted);}for(size_t i=0;i<mVertices.size();++i)mVertices[i].Normal=ToDirection(sums[i]);
}
std::vector<int> MeshTopology::getTriangleNeighbors(int id)const{std::vector<int>out;if(id<0||id>=int(mTriangles.size()))return out;for(int edge:mTriangles[id].EdgeIds)for(int t:mEdges[edge].IncidentTriangleIds)if(t!=id)out.push_back(t);std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;}
std::vector<int> MeshTopology::collectTriangleRing(int id,int rings)const{std::vector<int>out;if(id<0||id>=int(mTriangles.size()))return out;std::unordered_set<int>seen;seen.reserve(size_t(std::max(8,3*rings*rings)));std::queue<std::pair<int,int>>q;q.push({id,0});seen.insert(id);while(!q.empty()){auto[t,d]=q.front();q.pop();out.push_back(t);if(d>=rings)continue;for(int n:getTriangleNeighbors(t))if(seen.insert(n).second)q.push({n,d+1});}return out;}
}
