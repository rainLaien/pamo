#include "CadMesh/WallThicknessCalculator.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Property { bool list=false; std::string name; };
struct Ply {
  std::vector<std::string> header, vertices, faces;
  std::vector<std::array<double,3>> positions;
  std::vector<std::array<int,3>> triangles;
  std::size_t vertexCount=0, faceCount=0;
  std::vector<Property> vertexProperties, faceProperties;
  std::size_t x=SIZE_MAX,y=SIZE_MAX,z=SIZE_MAX,indices=SIZE_MAX;
};

bool readPly(const std::string& path, Ply& ply, std::string& error) {
  std::ifstream in(path);
  if(!in){error="cannot open input PLY";return false;}
  std::string line, element;
  bool ended=false, ascii=false;
  while(std::getline(in,line)){
    ply.header.push_back(line);
    std::istringstream row(line); std::string first; row>>first;
    if(first=="format") {std::string format,version;row>>format>>version;ascii=format=="ascii"&&version=="1.0";}
    else if(first=="element"){
      row>>element;
      if(element=="vertex") row>>ply.vertexCount;
      else if(element=="face") row>>ply.faceCount;
      else if(!element.empty()){error="only vertex and face elements are supported";return false;}
    } else if(first=="property"){
      std::string type,name;row>>type;
      if(type=="list"){
        std::string countType,itemType;row>>countType>>itemType>>name;
        Property p{true,name};
        if(element=="vertex") ply.vertexProperties.push_back(p);
        else if(element=="face"){
          if(name=="vertex_indices"||name=="vertex_index")ply.indices=ply.faceProperties.size();
          ply.faceProperties.push_back(p);
        }
      } else {
        row>>name;
        if(element=="vertex"){
          const auto index=ply.vertexProperties.size();
          if(name=="x")ply.x=index;if(name=="y")ply.y=index;if(name=="z")ply.z=index;
          ply.vertexProperties.push_back({false,name});
        } else if(element=="face")ply.faceProperties.push_back({false,name});
      }
    } else if(first=="end_header"){ended=true;break;}
  }
  if(!ended||!ascii||ply.vertexCount==0||ply.faceCount==0||
     ply.x==SIZE_MAX||ply.y==SIZE_MAX||ply.z==SIZE_MAX||ply.indices==SIZE_MAX){
    error="input must be an ASCII PLY 1.0 with xyz vertices and vertex_indices faces";return false;
  }
  ply.positions.reserve(ply.vertexCount);ply.vertices.reserve(ply.vertexCount);
  for(std::size_t i=0;i<ply.vertexCount;++i){
    if(!std::getline(in,line)){error="truncated vertex data";return false;}
    std::istringstream row(line);std::vector<double> values;double value;
    while(row>>value)values.push_back(value);
    if(values.size()<ply.vertexProperties.size()||!std::isfinite(values[ply.x])||
       !std::isfinite(values[ply.y])||!std::isfinite(values[ply.z])){
      error="invalid vertex row "+std::to_string(i);return false;
    }
    ply.positions.push_back({values[ply.x],values[ply.y],values[ply.z]});ply.vertices.push_back(line);
  }
  ply.triangles.reserve(ply.faceCount);ply.faces.reserve(ply.faceCount);
  for(std::size_t i=0;i<ply.faceCount;++i){
    if(!std::getline(in,line)){error="truncated face data";return false;}
    std::istringstream row(line);std::vector<std::string> tokens;std::string token;
    while(row>>token)tokens.push_back(token);
    std::size_t cursor=0;std::array<int,3> triangle{};bool got=false;
    for(std::size_t p=0;p<ply.faceProperties.size();++p){
      if(cursor>=tokens.size()){error="invalid face row "+std::to_string(i);return false;}
      if(ply.faceProperties[p].list){
        std::size_t n=0;try{n=std::stoul(tokens[cursor++]);}catch(...){error="invalid face list";return false;}
        if(cursor+n>tokens.size()){error="truncated face index list";return false;}
        if(p==ply.indices){if(n!=3){error="wall thickness requires triangular faces";return false;}
          try{for(int k=0;k<3;++k)triangle[k]=std::stoi(tokens[cursor+k]);}catch(...){error="invalid triangle index";return false;}got=true;}
        cursor+=n;
      } else ++cursor;
    }
    if(!got){error="missing triangle indices";return false;}
    for(int index:triangle)if(index<0||std::size_t(index)>=ply.vertexCount){error="face index out of range";return false;}
    ply.triangles.push_back(triangle);ply.faces.push_back(line);
  }
  return true;
}

bool writePly(const std::string& path,const Ply& ply,const CadMesh::WallThicknessResult& thickness,
              std::string& error){
  std::ofstream out(path);if(!out){error="cannot open output PLY";return false;}
  for(const auto& line:ply.header){
    if(line=="end_header"){
      out<<"comment wall_thickness is rolling_ball_diameter_in_model_units; invalid=nan\n"
            "comment thickness_status valid=0_measured_or_9_feature_sample; see WALL_THICKNESS.md\n"
            "property float wall_thickness\nproperty uchar thickness_valid\nproperty uchar thickness_status\n";
    }
    out<<line<<'\n';
  }
  for(const auto& line:ply.vertices)out<<line<<'\n';
  out<<std::setprecision(9);
  for(std::size_t i=0;i<ply.faces.size();++i){
    const auto status=thickness.Status[i];
    const bool valid=status==CadMesh::ThicknessStatus::Measured||status==CadMesh::ThicknessStatus::FeatureSample;
    out<<ply.faces[i]<<' ';
    if(valid)out<<thickness.Values[i];else out<<"nan";
    out<<' '<<int(valid)<<' '<<int(status)<<'\n';
  }
  out.flush();if(!out){error="writing output PLY failed";return false;}return true;
}
}

int main(int argc,char** argv){
  if(argc<3){std::cerr<<"Usage: cad_mesh_thickness input.ply output.ply [--workers count] [--minimum value] [--contact-angle-deg value]\n";return 2;}
  try{
    CadMesh::WallThicknessOptions options;
    for(int i=3;i<argc;++i){
      if(i+1>=argc)throw std::invalid_argument("missing value for "+std::string(argv[i]));
      const std::string key=argv[i++];const double value=std::stod(argv[i]);
      if(!std::isfinite(value))throw std::invalid_argument("nonfinite option value");
      if(key=="--workers"&&value>=1&&value<=128&&std::floor(value)==value)options.Workers=int(value);
      else if(key=="--minimum"&&value>=0)options.MinimumThickness=value;
      else if(key=="--contact-angle-deg"&&value>=0&&value<=180)options.MinimumContactAngleDegrees=value;
      else throw std::invalid_argument("invalid option or value: "+key);
    }
    using Clock=std::chrono::steady_clock;const auto start=Clock::now();
    Ply ply;std::string error;if(!readPly(argv[1],ply,error))throw std::runtime_error(error);
    CadMesh::WallThicknessResult result;
    if(!CadMesh::WallThicknessCalculator::compute(ply.positions,ply.triangles,options,result,error))throw std::runtime_error(error);
    if(!writePly(argv[2],ply,result,error))throw std::runtime_error(error);
    const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
    std::cout<<std::setprecision(9)<<"[CadMesh] wall thickness only: faces="<<ply.triangles.size()
      <<", valid="<<result.ValidFaces<<", valid_area_fraction="<<result.ValidAreaFraction
      <<", min="<<result.Minimum<<", max="<<result.Maximum
      <<", area_weighted_mean="<<result.AreaWeightedAverage
      <<", preparation_s="<<result.PreparationSeconds<<", sampling_s="<<result.SamplingSeconds
      <<", read_compute_write_s="<<elapsed<<"\nOutput: "<<argv[2]<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<"[CadMesh] thickness-only failed: "<<e.what()<<'\n';return 1;}
}
