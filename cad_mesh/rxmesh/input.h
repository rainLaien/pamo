#pragma once
#include "geometry.h"
#include <array>
#include <string>
#include <vector>
namespace pamo_rx {
struct Options {
    std::string input,output;bool snapshot=false,segment=false,skipAudit=false,skipRelocation=false,skipCollapse=false;
    double ratio=.01,target=0,deviation=.03,normalDegrees=10,featureDegrees=45;
    int iterations=5,maxFaces=4000000;
};
struct Input {
    std::vector<Vec> points;
    std::vector<std::vector<uint32_t>> faces;
    std::vector<std::vector<int>> labels,fixed,vertexRegion;
    std::vector<std::vector<double>> sizes,coordinates;
    std::vector<RefTriangle> reference;
    std::vector<RegionPlane> planes;
    std::vector<BvhNode> nodes;std::vector<int> roots;
    Vec origin;double diagonal=1,target=0,deviation=0;
    size_t sourceFaces=0,boundarySplits=0;
    size_t inputNonmanifoldEdges=0,manifoldVertexCopies=0;
    std::vector<std::array<int,2>> constraints;
};
Input prepare(const Options&);
void buildReference(std::vector<RefTriangle>&,std::vector<BvhNode>&,std::vector<int>&);
void writeResult(const Options&,const Input&,const std::vector<Vec>&,
                 const std::vector<std::array<int,3>>&,const std::vector<int>&,const std::vector<int>&,
                 double,double,double,bool);
}
