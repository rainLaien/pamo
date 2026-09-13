#include "CadMesh/NativeRemesher.h"
#include "CadMesh/StlReader.h"
#include <iostream>
#include <chrono>
#include <cmath>
int main(int argc,char **argv){
  try{
    if(argc<3){std::cerr<<"usage: cad_mesh_generic input.stl output_directory [--target 6] [--deviation 0.03] [--normal 10] [--feature-angle 45] [--iterations 5] [--analytic-guides 1]\n";return 2;}
    CadMesh::NativeRemeshConfig config;config.TargetEdgeLength=6;config.MaximumDeviation=.03;
    config.GenericFeatureAngleDegrees=45;config.GenericRemeshIterations=5;config.DisableCuda=true;
    for(int i=3;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("missing option value");
      const std::string option=argv[i];const double value=std::stod(argv[i+1]);
      if(!std::isfinite(value)||value<=0)throw std::runtime_error("positive finite value required");
      if(option=="--target")config.TargetEdgeLength=value;
      else if(option=="--deviation")config.MaximumDeviation=value;
      else if(option=="--normal" && value<90)config.MaximumNormalDeviationDegrees=value;
      else if(option=="--feature-angle" && value<180)config.GenericFeatureAngleDegrees=value;
      else if(option=="--iterations" && value<=30 && std::floor(value)==value)config.GenericRemeshIterations=int(value);
      else if(option=="--analytic-guides" && value==1)config.GenericAnalyticGuides=true;
      else throw std::runtime_error("invalid option: "+option);
    }
    const auto start=std::chrono::steady_clock::now();CadMesh::TriangleSoup soup;std::string error;
    if(!CadMesh::StlReader::read(argv[1],soup,error)||!CadMesh::NativeRemesher::genericRemesh(soup,config,argv[2],error)){
      std::cerr<<error<<'\n';return 1;}
    std::clog<<"[Generic] total including STL read: "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" s\n";
    return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
