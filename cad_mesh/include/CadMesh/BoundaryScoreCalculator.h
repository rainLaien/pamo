#pragma once
#include "CadMesh/SurfaceFitting.h"
namespace CadMesh {
class BoundaryScoreCalculator {
public: BoundaryScoreCalculator(MeshTopology& mesh,const SegmentationConfig& config):mMesh(mesh),mConfig(config){}
    void compute();
private: double LocalQuadricRms(const std::vector<int>& triangleIds) const; MeshTopology& mMesh; SegmentationConfig mConfig;
};
}
