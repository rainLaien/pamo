#pragma once

#include "CadMesh/Types.h"
#include <cstdint>
#include <unordered_map>

namespace CadMesh {
struct TriangleSoup { std::vector<Point3> Vertices; std::vector<std::array<int,3>> Triangles; };
struct MeshVertex { Point3 Position; std::vector<int> IncidentTriangleIds, OriginalVertexIds, NeighborVertexIds; Direction3 Normal{1,0,0}; DifferentialGeometry Geometry; };
struct MeshEdge { int Vertex0=-1,Vertex1=-1,Triangle0=-1,Triangle1=-1; std::vector<int> IncidentTriangleIds; double BoundaryScore=0; BoundaryEvidence Evidence; bool IsBoundary=false,IsNonManifold=false,IsConstrainedFeature=false; };
struct MeshTriangle { std::array<int,3> VertexIds{}; Direction3 Normal{1,0,0}; Point3 Centroid; int PatchId=-1; double Area=0; std::array<int,3> EdgeIds{{-1,-1,-1}}; };
struct CleanupReport { int InputTriangles=0,OutputTriangles=0,DegenerateTriangles=0,DuplicateTriangles=0,NonManifoldEdges=0,BoundaryEdges=0; std::vector<std::string> Warnings; };

class MeshTopology {
public:
    bool build(const TriangleSoup& soup);
    const std::vector<MeshVertex>& getVertices() const{return mVertices;}
    std::vector<MeshVertex>& getVertices(){return mVertices;}
    const std::vector<MeshEdge>& getEdges() const{return mEdges;}
    std::vector<MeshEdge>& getEdges(){return mEdges;}
    const std::vector<MeshTriangle>& getTriangles() const{return mTriangles;}
    std::vector<MeshTriangle>& getTriangles(){return mTriangles;}
    const TriangleSoup& getOriginalSoup() const{return mOriginalSoup;}
    const std::vector<int>& getOriginalToWeldedVertexMap() const{return mOriginalToWelded;}
    const std::vector<int>& getOriginalToCleanTriangleMap() const{return mOriginalToTriangle;}
    const MeshResolutionInfo& getResolution() const{return mResolution;}
    const CleanupReport& getCleanupReport() const{return mCleanup;}
    std::vector<int> getTriangleNeighbors(int triangleId) const;
    std::vector<int> collectTriangleRing(int triangleId,int rings) const;
private:
    void EstimateRawResolution(); void WeldVertices(); void BuildTriangles(); void BuildEdges(); void ComputeNormals();
    TriangleSoup mOriginalSoup; std::vector<MeshVertex> mVertices; std::vector<MeshEdge> mEdges; std::vector<MeshTriangle> mTriangles;
    std::vector<int> mOriginalToWelded,mOriginalToTriangle; MeshResolutionInfo mResolution; CleanupReport mCleanup;
};
}
