#include "cad_adaptive/RawCudaBatch.h"
#include "cad_adaptive/PartitionInput.h"
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace cad_adaptive;
int main(int argc,char **argv) {
  if(argc<3){std::cerr<<"Usage: cad_raw_partition_cli INPUT.cadpart OUTPUT.ply [--target h] [--iters n] [--workers n] [--memory-mb n] [--max-error e] [--feature-refine]\n";return 2;}
  try {
    RawBatchOptions options;RemeshConfig cfg;cfg.adaptive=false;cfg.maxIterations=20;
    bool refine=false;float errorBudget=0;
    for(int i=3;i<argc;++i) {
      std::string flag=argv[i];if(flag=="--feature-refine"){refine=true;continue;}
      if(i+1>=argc)throw std::runtime_error("missing option value");
      std::string value=argv[++i];size_t used=0;double number=std::stod(value,&used);
      if(used!=value.size() || !std::isfinite(number) || number<=0)throw std::runtime_error("invalid positive option value");
      if(flag=="--target")cfg.constantLength=float(number);
      else if(flag=="--max-error")errorBudget=float(number);
      else if(flag=="--iters" || flag=="--workers") {
        if(number!=std::floor(number) || number>1000)throw std::runtime_error("invalid integer option");
        if(flag=="--iters")cfg.maxIterations=int(number);else options.workers=int(number);
      } else if(flag=="--memory-mb") {
        if(number>1024*1024)throw std::runtime_error("memory budget too large");
        options.memoryBytes=size_t(number*1024*1024);
      } else throw std::runtime_error("unknown option: "+flag);
    }
    SemanticMesh source,output;std::string error;
    if(!loadPartitionInput(argv[1],source,&error,true))throw std::runtime_error(error);
    if(!(cfg.constantLength>0))cfg.constantLength=source.bboxDiagonal()*.01f;
    cfg.maxGeometryError=errorBudget>0?errorBudget:.2f*cfg.constantLength;
    if(refine){cfg.featureEdgeLength=.25f*cfg.constantLength;cfg.featureBand=.75f*cfg.constantLength;}
    RawBatchReport report;
    const bool valid=remeshRawCudaPatches(source,output,cfg,options,report,&error);
    std::ofstream out(std::string(argv[2])+".json");
    out<<"{\n  \"backend\": \"gpu-raw-cuda-patches\",\n  \"workers\": "<<options.workers
       <<",\n  \"seconds\": "<<report.seconds<<",\n  \"memory_budget\": "<<report.memoryBudget
       <<",\n  \"peak_reserved\": "<<report.peakReserved<<",\n  \"peak_active\": "<<report.peakActive
       <<",\n  \"accepted\": "<<report.accepted<<",\n  \"fallback\": "<<report.fallback
       <<",\n  \"boundary_splits\": "<<report.boundarySplits
       <<",\n  \"topology_valid\": "<<(report.topologyValid?"true":"false")
       <<",\n  \"boundaries_held\": "<<(report.boundariesHeld?"true":"false")<<",\n  \"patches\": [";
    for(size_t i=0;i<report.patches.size();++i){const auto &p=report.patches[i];
      out<<(i?",\n":"\n")<<"    {\"id\": "<<i<<", \"accepted\": "<<(p.accepted?"true":"false")
         <<", \"input_faces\": "<<p.inputFaces<<", \"output_faces\": "<<p.outputFaces
         <<", \"seconds\": "<<p.seconds<<", \"workspace_bytes\": "<<p.workspaceBytes
         <<", \"geometry_error_max\": "<<p.report.geometryErrorMax<<", \"error\": "<<std::quoted(p.error)<<"}";
    }
    out<<"\n  ]\n}\n";out.close();if(!out)throw std::runtime_error("cannot write batch report");
    if(!valid)throw std::runtime_error(error);
    if(!output.save(argv[2],&error))throw std::runtime_error(error);
    std::cout<<"raw_batch seconds="<<report.seconds<<" accepted="<<report.accepted<<" fallback="<<report.fallback
             <<" peak_active="<<report.peakActive<<" vertices="<<output.vertexCount()<<" faces="<<output.faceCount()<<'\n';
    return report.fallback?3:0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
