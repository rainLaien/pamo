#include "cad_adaptive/RawCudaBatch.h"
#include "cad_adaptive/PartitionInput.h"
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace cad_adaptive;
namespace {
std::pair<double,double> qualitySummary(const SemanticMesh &mesh) {
  std::vector<float> quality;quality.reserve(mesh.faceCount());
  double sum=0;
  for(int f=0;f<mesh.faceCount();++f)if(mesh.faceAlive[f]) {
    const auto t=mesh.face(f);
    const float q=triangleQuality(mesh.position(t[0]),mesh.position(t[1]),mesh.position(t[2]));
    quality.push_back(q);sum+=q;
  }
  if(quality.empty())return {0,0};
  const size_t index=quality.size()/20;
  std::nth_element(quality.begin(),quality.begin()+index,quality.end());
  return {sum/double(quality.size()),quality[index]};
}
}
int main(int argc,char **argv) {
  if(argc<3){std::cerr<<"Usage: cad_raw_partition_cli INPUT.(cadpart|stl|obj|ply) OUTPUT.ply [--target h] [--iters n] [--workers n] [--gpu-concurrency n] [--patches-per-task n] [--smooth-passes n] [--collapse-passes n] [--flip-passes n] [--auto-partition] [--legacy-flip] [--no-idle-stop] [--memory-mb n] [--max-error e] [--feature-refine]\n";return 2;}
  try {
    RawBatchOptions options;options.requireQualityImprovement=true;
    RemeshConfig cfg;cfg.adaptive=false;cfg.maxIterations=20;
    bool refine=false;float errorBudget=0;
    for(int i=3;i<argc;++i) {
      std::string flag=argv[i];if(flag=="--feature-refine"){refine=true;continue;}
      if(flag=="--no-idle-stop"){options.stopWhenIdle=false;continue;}
      if(flag=="--legacy-flip"){options.strictFlipQuality=false;continue;}
      if(flag=="--auto-partition"){options.autoPartitionSinglePatch=true;continue;}
      if(i+1>=argc)throw std::runtime_error("missing option value");
      std::string value=argv[++i];size_t used=0;double number=std::stod(value,&used);
      if(used!=value.size() || !std::isfinite(number) || number<=0)throw std::runtime_error("invalid positive option value");
      if(flag=="--target")cfg.constantLength=float(number);
      else if(flag=="--max-error")errorBudget=float(number);
      else if(flag=="--iters" || flag=="--workers" || flag=="--gpu-concurrency" || flag=="--patches-per-task" || flag=="--smooth-passes" || flag=="--collapse-passes" || flag=="--flip-passes") {
        if(number!=std::floor(number) || number>1000)throw std::runtime_error("invalid integer option");
        if(flag=="--iters")cfg.maxIterations=int(number);
        else if(flag=="--workers")options.workers=int(number);
        else if(flag=="--gpu-concurrency")options.gpuConcurrency=int(number);
        else if(flag=="--patches-per-task")options.patchesPerTask=int(number);
        else if(flag=="--smooth-passes")options.smoothPasses=int(number);
        else if(flag=="--collapse-passes")options.collapsePasses=int(number);
        else options.flipPasses=int(number);
      } else if(flag=="--memory-mb") {
        if(number>1024*1024)throw std::runtime_error("memory budget too large");
        options.memoryBytes=size_t(number*1024*1024);
      } else throw std::runtime_error("unknown option: "+flag);
    }
    const auto clockStart=std::chrono::steady_clock::now();
    const auto elapsed=[&] {return std::chrono::duration<double>(std::chrono::steady_clock::now()-clockStart).count();};
    SemanticMesh source,output;std::string error;
    const std::string inputPath=argv[1];
    const bool isCadPart=inputPath.size()>=8 && inputPath.substr(inputPath.size()-8)==".cadpart";
    if(isCadPart) {
      if(!loadPartitionInput(inputPath,source,&error,true))throw std::runtime_error(error);
    } else {
      if(!source.load(inputPath,&error))throw std::runtime_error(error);
      source.rebuildTopology();
    }
    const double loadSeconds=elapsed();
    if(!(cfg.constantLength>0))cfg.constantLength=source.bboxDiagonal()*.01f;
    cfg.maxGeometryError=errorBudget>0?errorBudget:.2f*cfg.constantLength;
    if(refine){cfg.featureEdgeLength=.25f*cfg.constantLength;cfg.featureBand=.75f*cfg.constantLength;}
    RawBatchReport report;
    const bool valid=remeshRawCudaPatches(source,output,cfg,options,report,&error);
    const double remeshEnd=elapsed();
    const auto sourceQuality=qualitySummary(source);
    const auto outputQuality=valid?qualitySummary(output):std::pair<double,double>{0,0};
    const bool qualityImproved=valid && outputQuality.first>=sourceQuality.first &&
        outputQuality.second>=sourceQuality.second;
    std::ofstream out(std::string(argv[2])+".json");
    auto writeArray=[&](const auto &values) {
      out<<'[';
      for(size_t j=0;j<values.size();++j)out<<(j?", ":"")<<values[j];
      out<<']';
    };
    out<<"{\n  \"backend\": \"gpu-raw-cuda-patches\",\n  \"workers\": "<<options.workers
       <<",\n  \"gpu_concurrency\": "<<options.gpuConcurrency
       <<",\n  \"patches_per_task\": "<<report.patchesPerTask
       <<",\n  \"smooth_passes\": "<<options.smoothPasses
       <<",\n  \"collapse_passes\": "<<options.collapsePasses
       <<",\n  \"flip_passes\": "<<options.flipPasses
       <<",\n  \"strict_flip_quality\": "<<(options.strictFlipQuality?"true":"false")
       <<",\n  \"auto_partition_single_patch\": "<<(options.autoPartitionSinglePatch?"true":"false")
       <<",\n  \"stop_when_idle\": "<<(options.stopWhenIdle?"true":"false")
       <<",\n  \"seconds\": "<<report.seconds<<",\n  \"memory_budget\": "<<report.memoryBudget
       <<",\n  \"source_quality_mean\": "<<sourceQuality.first
       <<",\n  \"source_quality_p05\": "<<sourceQuality.second
       <<",\n  \"output_quality_mean\": "<<outputQuality.first
       <<",\n  \"output_quality_p05\": "<<outputQuality.second
       <<",\n  \"quality_improved\": "<<(qualityImproved?"true":"false")
       <<",\n  \"seconds_boundary\": "<<report.secondsBoundary
       <<",\n  \"seconds_tasks\": "<<report.secondsTasks
       <<",\n  \"seconds_assembly\": "<<report.secondsAssembly
       <<",\n  \"peak_reserved\": "<<report.peakReserved<<",\n  \"peak_active\": "<<report.peakActive
       <<",\n  \"initial_tasks\": "<<report.patches.size()
       <<",\n  \"accepted\": "<<report.accepted<<",\n  \"fallback\": "<<report.fallback
       <<",\n  \"retried\": "<<report.retried
       <<",\n  \"seam_repairs\": "<<report.seamRepairs
       <<",\n  \"seam_split_repairs\": "<<report.seamSplitRepairs
       <<",\n  \"boundary_splits\": "<<report.boundarySplits
       <<",\n  \"topology_valid\": "<<(report.topologyValid?"true":"false")
       <<",\n  \"boundaries_held\": "<<(report.boundariesHeld?"true":"false")<<",\n  \"patches\": [";
    for(size_t i=0;i<report.patches.size();++i){const auto &p=report.patches[i];
      out<<(i?",\n":"\n")<<"    {\"id\": "<<i<<", \"accepted\": "<<(p.accepted?"true":"false")
         <<", \"retried\": "<<(p.retried?"true":"false")
         <<", \"input_faces\": "<<p.inputFaces<<", \"output_faces\": "<<p.outputFaces
         <<", \"cycles_executed\": "<<p.report.cyclesExecuted
         <<", \"recovered_cycle_failures\": "<<p.report.recoveredCycleFailures
         <<", \"selected_cycle\": "<<p.report.selectedCycle
         <<", \"seconds\": "<<p.seconds<<", \"workspace_bytes\": "<<p.workspaceBytes
         <<", \"seconds_setup\": "<<p.report.secondsSetup
         <<", \"seconds_split\": "<<p.report.secondsSplit
         <<", \"seconds_collapse\": "<<p.report.secondsCollapse
         <<", \"seconds_compact\": "<<p.report.secondsCompact
         <<", \"seconds_validate\": "<<p.report.secondsValidate
         <<", \"seconds_flip\": "<<p.report.secondsFlip
         <<", \"seconds_smooth\": "<<p.report.secondsSmooth
         <<", \"splits\": "<<p.report.splits<<", \"collapses\": "<<p.report.collapses
         <<", \"flips\": "<<p.report.flips<<", \"smooth_moves\": "<<p.report.smoothMoves
         <<", \"geometry_error_max\": "<<p.report.geometryErrorMax<<", \"error\": "<<std::quoted(p.error)
         <<", \"retry_reason\": "<<std::quoted(p.retryReason);
      out<<", \"collapse_pass_calls\": ";writeArray(p.report.collapsePassCalls);
      out<<", \"collapse_pass_accepted\": ";writeArray(p.report.collapsePassAccepted);
      out<<", \"collapse_pass_seconds\": ";writeArray(p.report.collapsePassSeconds);
      out<<", \"flip_pass_calls\": ";writeArray(p.report.flipPassCalls);
      out<<", \"flip_pass_accepted\": ";writeArray(p.report.flipPassAccepted);
      out<<", \"flip_pass_seconds\": ";writeArray(p.report.flipPassSeconds);
      out<<'}';
    }
    out<<"\n  ]\n}\n";out.close();if(!out)throw std::runtime_error("cannot write batch report");
    if(!valid)throw std::runtime_error(error);
    if(!output.save(argv[2],&error))throw std::runtime_error(error);
    std::cout<<"raw_batch_timing load_s="<<loadSeconds<<" batch_s="<<(remeshEnd-loadSeconds)
             <<" save_and_report_s="<<(elapsed()-remeshEnd)<<'\n';
    std::cout<<"raw_batch seconds="<<report.seconds<<" patches_per_task="<<report.patchesPerTask<<" accepted="<<report.accepted<<" fallback="<<report.fallback<<" retried="<<report.retried
             <<" peak_active="<<report.peakActive<<" vertices="<<output.vertexCount()<<" faces="<<output.faceCount()<<'\n';
    if(report.fallback)return 3;
    if(!qualityImproved){std::cerr<<"output mesh quality regressed\n";return 4;}
    return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
