#pragma once
#include "CadMesh/MeshTopology.h"
namespace CadMesh {
class DifferentialGeometryEstimator {
public: explicit DifferentialGeometryEstimator(MeshTopology& mesh):mMesh(mesh){}
    void compute(int ringCount=2);
private: DifferentialGeometry FitVertex(int vertexId,int ringCount) const; MeshTopology& mMesh;
};
}
