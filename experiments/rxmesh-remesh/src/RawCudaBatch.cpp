#include "cad_adaptive/RawCudaBatch.h"
#include "cad_adaptive/PartitionInput.h"
#include "RemeshWorkerPool.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_set>

namespace cad_adaptive {
namespace {
using Clock=std::chrono::steady_clock;
using Position=std::tuple<float,float,float>;
Position positionKey(Vec3 p){return {p.x,p.y,p.z};}
uint64_t edgeKey(int a,int b){return (uint64_t(std::min(a,b))<<32)|uint32_t(std::max(a,b));}
std::pair<float,float> qualitySummary(const SemanticMesh &mesh) {
  std::vector<float> values;values.reserve(mesh.faceCount());
  double sum=0;
  for(int f=0;f<mesh.faceCount();++f)if(mesh.faceAlive[f]) {
    const auto t=mesh.face(f);
    const float q=triangleQuality(mesh.position(t[0]),mesh.position(t[1]),mesh.position(t[2]));
    values.push_back(q);sum+=q;
  }
  if(values.empty())return {0.f,0.f};
  const size_t p05=values.size()/20;
  std::nth_element(values.begin(),values.begin()+p05,values.end());
  return {float(sum/double(values.size())),values[p05]};
}
void cudaCheck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct FacePartitionItem {
  int Face=0;
  Vec3 Centroid{};
};
void PartitionFacesRecursive(std::vector<FacePartitionItem> &items,size_t begin,size_t end,
                             int firstPartition,int partitionCount,std::vector<uint32_t> &facePatchId) {
  if(partitionCount<=1 || end-begin<=1) {
    for(size_t i=begin;i<end;++i) facePatchId[items[i].Face]=uint32_t(firstPartition);
    return;
  }
  Vec3 lo=items[begin].Centroid,hi=lo;
  for(size_t i=begin+1;i<end;++i) {
    const Vec3 c=items[i].Centroid;
    lo.x=std::min(lo.x,c.x);lo.y=std::min(lo.y,c.y);lo.z=std::min(lo.z,c.z);
    hi.x=std::max(hi.x,c.x);hi.y=std::max(hi.y,c.y);hi.z=std::max(hi.z,c.z);
  }
  const Vec3 span=hi-lo;
  const int axis=span.y>span.x ? (span.z>span.y?2:1) : (span.z>span.x?2:0);
  auto coord=[axis](const FacePartitionItem &x){return axis==0?x.Centroid.x:(axis==1?x.Centroid.y:x.Centroid.z);};
  const int leftPartitions=partitionCount/2;
  const int rightPartitions=partitionCount-leftPartitions;
  const size_t leftCount=(end-begin)*size_t(leftPartitions)/size_t(partitionCount);
  const size_t mid=begin+std::max<size_t>(1,std::min(end-begin-1,leftCount));
  std::nth_element(items.begin()+begin,items.begin()+mid,items.begin()+end,
                   [&](const FacePartitionItem &a,const FacePartitionItem &b){
                     const float ca=coord(a),cb=coord(b);
                     return ca<cb || (ca==cb && a.Face<b.Face);
                   });
  PartitionFacesRecursive(items,begin,mid,firstPartition,leftPartitions,facePatchId);
  PartitionFacesRecursive(items,mid,end,firstPartition+leftPartitions,rightPartitions,facePatchId);
}
int AutoPartitionSinglePatch(SemanticMesh &mesh,int requestedPartitions) {
  if(requestedPartitions<=1 || mesh.faceCount()<2 || mesh.patches.size()!=1) return int(mesh.patches.size());
  const int partitions=std::max(1,std::min(requestedPartitions,mesh.faceCount()));
  std::vector<FacePartitionItem> items;items.reserve(mesh.faceCount());
  for(int f=0;f<mesh.faceCount();++f) if(mesh.faceAlive.empty() || mesh.faceAlive[f])
    items.push_back({f,centroid3(mesh.facePoint(f,0),mesh.facePoint(f,1),mesh.facePoint(f,2))});
  if(items.size()<2)return 1;
  mesh.facePatchId.assign(mesh.faceCount(),0u);
  PartitionFacesRecursive(items,0,items.size(),0,partitions,mesh.facePatchId);
  const PatchRecord base=mesh.patches.front();
  mesh.patches.assign(partitions,base);
  mesh.facePatchType.resize(mesh.faceCount(),uint8_t(base.type));
  mesh.rebuildTopology();
  return partitions;
}
struct Job {
  int id=0,firstPatch=0,patchCount=0; double cost=0; size_t bytes=0;
  bool succeeded=false,unchanged=false,uniformSizing=false,qualitySplit=false;
  float qualityMeanFloor=0,qualityP05Floor=0;
  std::vector<int> faces;
  SemanticMesh output;
  std::vector<int> aliases;
  std::vector<Job> children;
};
bool splitConflictingChord(Job &job,int sourceA,int sourceB,std::string &reason) {
  if(!job.succeeded){reason="job not accepted";return false;}
  int a=-1,b=-1;
  for(int v=0;v<int(job.aliases.size());++v) {
    if(job.aliases[v]==sourceA)a=v;
    if(job.aliases[v]==sourceB)b=v;
  }
  if(a<0 || b<0){reason="chord endpoint absent";return false;}
  SemanticMesh candidate=job.output;
  candidate.rebuildTopology();
  const uint64_t chord=edgeKey(a,b);
  for(const auto &edge:candidate.edges) {
    if(edgeKey(edge.v0,edge.v1)!=chord)continue;
    if(edge.face0<0 || edge.face1<0 ||
       (edge.flags&(EdgePatchBoundary|EdgeMeshBoundary))){reason="chord is a boundary edge";return false;}
    const auto feature=candidate.featureEdges.find(chord);
    const bool sharp=feature!=candidate.featureEdges.end();
    const uint32_t featureId=sharp?feature->second:0;
    const int mid=candidate.addVertex((candidate.position(a)+candidate.position(b))*.5f,
                                      edge.patchLeft,sharp?VertexConstraint::FeatureEdge:
                                                           VertexConstraint::Surface);
    candidate.targetLength[mid]=.5f*(candidate.targetLength[a]+candidate.targetLength[b]);
    if(sharp) {
      candidate.featureEdges.erase(chord);
      candidate.featureEdges[edgeKey(a,mid)]=featureId;
      candidate.featureEdges[edgeKey(mid,b)]=featureId;
    }
    for(int f:{edge.face0,edge.face1}) {
      const auto t=candidate.face(f);
      const uint32_t patch=candidate.facePatchId[f];
      bool found=false;
      for(int k=0;k<3;++k) {
        const int u=t[k],v=t[(k+1)%3],w=t[(k+2)%3];
        if((u==a && v==b) || (u==b && v==a)) {
          candidate.killFace(f);
          candidate.addFace(u,mid,w,patch,PatchType(candidate.facePatchType[f]));
          candidate.addFace(mid,v,w,patch,PatchType(candidate.facePatchType[f]));
          found=true;break;
        }
      }
      if(!found){reason="incident triangle does not contain chord";return false;}
    }
    candidate.rebuildTopology();
    std::string problem;
    if(!candidate.validate(&problem)){reason="split topology: "+problem;return false;}
    const auto newQuality=qualitySummary(candidate);
    if(newQuality.first+1.e-6f<job.qualityMeanFloor ||
       newQuality.second+1.e-6f<job.qualityP05Floor){reason="split quality below local input";return false;}
    job.aliases.resize(candidate.vertexCount(),-1);
    job.output=std::move(candidate);
    return true;
  }
  reason="chord absent from job";
  return false;
}
void execute(Job &job,const SemanticMesh &source,const RemeshConfig &config,
             bool stopWhenIdle,int smoothPasses,int collapsePasses,int flipPasses,bool strictFlipQuality,
             bool requireQualityImprovement,RawPatchResult &result) {
  const auto start=Clock::now();
  result.inputFaces=int(job.faces.size());result.workspaceBytes=job.bytes;
  try {
    SemanticMesh local;
    local.patches.assign(source.patches.begin()+job.firstPatch,
                         source.patches.begin()+job.firstPatch+job.patchCount);
    std::map<int,int> indices;
    std::map<Position,int> anchors;
    std::vector<int> originals;
    for(int f:job.faces) {
      auto t=source.face(f);
      for(int &v:t) {
        auto found=indices.find(v);
        if(found==indices.end()) {
          const int original=v;
          const auto constraint=VertexConstraint(source.vertexConstraint[v]);
          const auto p=source.position(v);
          v=local.addVertex(p,source.facePatchId[f]-job.firstPatch,constraint);originals.push_back(original);
          indices.emplace(original,v);
          if(constraint==VertexConstraint::Locked && !anchors.emplace(positionKey(p),original).second)
            throw std::runtime_error("ambiguous coincident boundary identities");
        } else v=found->second;
      }
      const auto patch=source.facePatchId[f]-job.firstPatch;
      local.addFace(t[0],t[1],t[2],patch,source.patches[source.facePatchId[f]].type);
    }
    for(auto feature:source.featureEdges) {
      auto a=indices.find(int(feature.first>>32)),b=indices.find(int(uint32_t(feature.first)));
      if(a!=indices.end() && b!=indices.end())local.featureEdges[edgeKey(a->second,b->second)]=feature.second;
    }
    local.rebuildTopology();
    const auto localQualityFloor=requireQualityImprovement?qualitySummary(local):
        std::pair<float,float>{0.f,0.f};
    result.inputQualityMean=localQualityFloor.first;
    result.inputQualityP05=localQualityFloor.second;
    job.qualityMeanFloor=localQualityFloor.first;
    job.qualityP05Floor=localQualityFloor.second;
    std::set<uint64_t> required;
    for(auto e:local.edges) if(e.flags&EdgeMeshBoundary)
      required.insert(edgeKey(originals[e.v0],originals[e.v1]));
    const bool hasSharedOrOpenBoundary=source.patches.size()>1 ||
        std::any_of(source.edges.begin(),source.edges.end(),[](const EdgeRec &e){return (e.flags&EdgeMeshBoundary)!=0;});
    RawCudaOptions options;options.independentStream=true;options.quiet=true;options.error=&result.error;options.freezeBoundary=hasSharedOrOpenBoundary;options.stopWhenIdle=stopWhenIdle;options.smoothPasses=smoothPasses;options.collapsePasses=collapsePasses;options.flipPasses=flipPasses;options.strictFlipQuality=strictFlipQuality;options.workspaceBytes=job.bytes;options.qualityMeanFloor=localQualityFloor.first;options.qualityP05Floor=localQualityFloor.second;
    SemanticMesh initialLocal;
    if(requireQualityImprovement && collapsePasses>1)initialLocal=local;
    const bool initialOk=remeshRawCuda(local,config,result.report,options);
    const auto initialQuality=initialOk?qualitySummary(local):std::pair<float,float>{0.f,0.f};
    bool improved=initialOk && (!requireQualityImprovement ||
        (result.report.selectedCycle>=0 &&
         initialQuality.first+1.e-6f>=localQualityFloor.first &&
         initialQuality.second+1.e-6f>=localQualityFloor.second));
    if(!improved && requireQualityImprovement && collapsePasses>1) {
      result.retried=true;
      result.retryReason=initialOk?"no cycle improved both local quality metrics":result.error;
      std::string retryError;
      RawCudaOptions retryOptions=options;
      retryOptions.error=&retryError;
      retryOptions.collapsePasses=1;
      // A small lower-tail tradeoff is allowed only with a substantial mean
      // gain. The full output still has to pass its global quality gate.
      retryOptions.qualityMeanFloor=localQualityFloor.first*1.01f;
      retryOptions.qualityP05Floor=localQualityFloor.second*.99f;
      RemeshConfig retryConfig=config;
      retryConfig.maxIterations=std::min(config.maxIterations,8);
      RemeshReport retryReport;
      if(remeshRawCuda(initialLocal,retryConfig,retryReport,retryOptions)) {
        const auto retryQuality=qualitySummary(initialLocal);
        if(retryReport.selectedCycle>=0 &&
           retryQuality.first+1.e-6f>=retryOptions.qualityMeanFloor &&
           retryQuality.second+1.e-6f>=retryOptions.qualityP05Floor) {
          local=std::move(initialLocal);
          result.report=retryReport;
          result.error.clear();
          improved=true;
        }
      }
      if(!improved && !retryError.empty())result.error=retryError;
    }
    if(!improved && initialOk && requireQualityImprovement &&
       config.featureEdgeLength>0.f) {
      // Crease-rich regions can accumulate slivers under curvature sizing.
      // The source's hard feature edges and frozen seam stay unchanged while
      // only the interior sizing field is switched to uniform.
      result.retried=true;
      RemeshConfig uniformConfig=config;
      uniformConfig.featureEdgeLength=0.f;
      uniformConfig.featureBand=0.f;
      uniformConfig.maxIterations=std::min(config.maxIterations,8);
      SemanticMesh uniformSource=local;
      RawCudaOptions uniformOptions=options;
      std::string uniformError;
      uniformOptions.error=&uniformError;
      uniformOptions.collapsePasses=1;
      uniformOptions.qualityMeanFloor=localQualityFloor.first*1.01f;
      uniformOptions.qualityP05Floor=localQualityFloor.second*.99f;
      RemeshReport uniformReport;
      SemanticMesh uniformMesh=uniformSource;
      bool uniformOk=remeshRawCuda(uniformMesh,uniformConfig,uniformReport,uniformOptions);
      auto uniformQuality=uniformOk?qualitySummary(uniformMesh):std::pair<float,float>{0.f,0.f};
      if(!(uniformOk && uniformReport.selectedCycle>=0 &&
           uniformQuality.first+1.e-6f>=uniformOptions.qualityMeanFloor &&
           uniformQuality.second+1.e-6f>=uniformOptions.qualityP05Floor)) {
        uniformMesh=std::move(uniformSource);
        uniformOptions=options;
        uniformOptions.error=&uniformError;
        uniformOk=remeshRawCuda(uniformMesh,uniformConfig,uniformReport,uniformOptions);
        uniformQuality=uniformOk?qualitySummary(uniformMesh):std::pair<float,float>{0.f,0.f};
      }
      if(uniformOk && uniformReport.selectedCycle>=0 &&
         uniformQuality.first+1.e-6f>=uniformOptions.qualityMeanFloor &&
         uniformQuality.second+1.e-6f>=uniformOptions.qualityP05Floor) {
        local=std::move(uniformMesh);
        result.report=uniformReport;
        result.uniformSizing=true;
        result.error.clear();
        improved=true;
      }
    }
    if(!improved && initialOk && requireQualityImprovement) {
      result.retried=true;
      SemanticMesh splitMesh=local;
      RemeshConfig splitConfig=config;
      splitConfig.maxIterations=std::min(config.maxIterations,8);
      RawCudaOptions splitOptions=options;
      splitOptions.optimizeSplitPoint=true;
      splitOptions.splitQualityRatio=.5f;
      std::string splitError;
      splitOptions.error=&splitError;
      RemeshReport splitReport;
      if(remeshRawCuda(splitMesh,splitConfig,splitReport,splitOptions)) {
        const auto splitQuality=qualitySummary(splitMesh);
        if(splitReport.selectedCycle>=0 &&
           splitQuality.first+1.e-6f>=localQualityFloor.first &&
           splitQuality.second+1.e-6f>=localQualityFloor.second) {
          local=std::move(splitMesh);
          result.report=splitReport;
          result.qualitySplit=true;
          result.error.clear();
          improved=true;
        }
      }
    }
    if(!improved && initialOk &&
       initialQuality.first+1.e-6f>=localQualityFloor.first &&
       initialQuality.second+1.e-6f>=localQualityFloor.second) {
      // The input itself passed the quality and geometry gates. Keep this
      // outcome visible as unchanged rather than calling it an improvement.
      result.unchanged=true;
      result.error.clear();
      improved=true;
    }
    if(!improved)throw std::runtime_error(result.error.empty()?
        "no GPU cycle met the local quality gate":result.error);
    std::string topologyError;
    if(!local.validate(&topologyError))throw std::runtime_error("task topology: "+topologyError);
    job.aliases.assign(local.vertexCount(),-1);
    std::set<int> retained;
    for(int v=0;v<local.vertexCount();++v) {
      if(local.vertexConstraint[v]!=uint8_t(VertexConstraint::Locked))continue;
      auto found=anchors.find(positionKey(local.position(v)));
      if(found==anchors.end() || !retained.insert(found->second).second)
        throw std::runtime_error("boundary identity changed or duplicated");
      job.aliases[v]=found->second;
    }
    if(retained.size()!=anchors.size())throw std::runtime_error("missing locked vertex");
    for(auto e:local.edges) if(e.flags&EdgeMeshBoundary) {
      int a=job.aliases[e.v0],b=job.aliases[e.v1];
      if(a<0 || b<0 || !required.erase(edgeKey(a,b)))throw std::runtime_error("unexpected patch boundary");
    }
    if(!required.empty())throw std::runtime_error("missing patch boundary edge");
    result.outputFaces=local.faceCount();result.accepted=true;job.succeeded=true;
    job.unchanged=result.unchanged;job.uniformSizing=result.uniformSizing;
    job.qualitySplit=result.qualitySplit;
    job.output=std::move(local);
  } catch(const std::exception &e) {result.error=e.what();result.outputFaces=result.inputFaces;}
  if(!result.accepted && job.patchCount>1) {
    result.retried=true;
    result.retryReason=result.error;
    job.children.resize(job.patchCount);
    bool allAccepted=true;
    int outputFaces=0;
    std::string failures;
    RemeshReport combined;
    for(int k=0;k<job.patchCount;++k) {
      auto &child=job.children[k];
      child.firstPatch=job.firstPatch+k;child.patchCount=1;child.bytes=job.bytes;
      for(int f:job.faces)if(source.facePatchId[f]==uint32_t(child.firstPatch))child.faces.push_back(f);
      RawPatchResult childResult;
      execute(child,source,config,stopWhenIdle,smoothPasses,collapsePasses,flipPasses,
              strictFlipQuality,requireQualityImprovement,childResult);
      allAccepted &= childResult.accepted;
      outputFaces+=childResult.outputFaces;
      if(!childResult.accepted)failures+=" patch "+std::to_string(child.firstPatch)+": "+childResult.error;
      const auto &r=childResult.report;
      combined.cyclesExecuted=std::max(combined.cyclesExecuted,r.cyclesExecuted);
      combined.recoveredCycleFailures+=r.recoveredCycleFailures;
      combined.splits+=r.splits;combined.collapses+=r.collapses;
      combined.flips+=r.flips;combined.smoothMoves+=r.smoothMoves;
      combined.geometryErrorMax=std::max(combined.geometryErrorMax,r.geometryErrorMax);
      combined.secondsSetup+=r.secondsSetup;combined.secondsSplit+=r.secondsSplit;
      combined.secondsCollapse+=r.secondsCollapse;combined.secondsCompact+=r.secondsCompact;
      combined.secondsValidate+=r.secondsValidate;combined.secondsFlip+=r.secondsFlip;
      combined.secondsSmooth+=r.secondsSmooth;
      for(int pass=0;pass<8;++pass) {
        combined.collapsePassCalls[pass]+=r.collapsePassCalls[pass];
        combined.collapsePassAccepted[pass]+=r.collapsePassAccepted[pass];
        combined.collapsePassSeconds[pass]+=r.collapsePassSeconds[pass];
        combined.flipPassCalls[pass]+=r.flipPassCalls[pass];
        combined.flipPassAccepted[pass]+=r.flipPassAccepted[pass];
        combined.flipPassSeconds[pass]+=r.flipPassSeconds[pass];
      }
    }
    result.accepted=allAccepted;
    result.outputFaces=outputFaces;
    result.error=allAccepted?std::string{}:failures;
    result.report=combined;
  }
  result.seconds=std::chrono::duration<double>(Clock::now()-start).count();
  result.report.seconds=result.seconds;
}
}

bool remeshRawCudaPatches(const SemanticMesh &input,SemanticMesh &output,const RemeshConfig &cfg,
                         const RawBatchOptions &options,RawBatchReport &report,std::string *error) {
  report={};const auto start=Clock::now();
  try {
    if(options.workers<1 || options.workers>32 || options.gpuConcurrency<1 || options.gpuConcurrency>8 || options.patchesPerTask<0 || options.patchesPerTask>32 || options.smoothPasses<1 || options.smoothPasses>12 || options.collapsePasses<1 || options.collapsePasses>8 || options.flipPasses<1 || options.flipPasses>8 || !(cfg.constantLength>0) || !std::isfinite(cfg.constantLength) || !(cfg.maxGeometryError>0) || !std::isfinite(cfg.maxGeometryError) || cfg.maxIterations<1 || cfg.adaptive)
      throw std::runtime_error("invalid batch worker count or sizing configuration");
    SemanticMesh source=input;source.rebuildTopology();
    if(options.autoPartitionSinglePatch)AutoPartitionSinglePatch(source,options.workers);
    const bool semanticPatches=std::any_of(source.patches.begin(),source.patches.end(),
        [](const PatchRecord &patch){return patch.type!=PatchType::Unknown;});
    const int patchesPerTask=options.patchesPerTask?options.patchesPerTask:
        (semanticPatches && source.patches.size()>=128?16:1);
    report.patchesPerTask=patchesPerTask;
    std::string problem;if(!source.validate(&problem))throw std::runtime_error(problem);
    for(auto e:source.edges)if(e.flags&(EdgePatchBoundary|EdgeMeshBoundary)) {
      source.vertexConstraint[e.v0]=source.vertexConstraint[e.v1]=uint8_t(VertexConstraint::Locked);
    }
    // Both incident patches receive exactly the same new vertices. This runs
    // once before creating jobs, never independently inside a patch worker.
    const auto boundaryStart=Clock::now();
    report.boundarySplits=refinePartitionBoundary(source,cfg);
    report.secondsBoundary=std::chrono::duration<double>(Clock::now()-boundaryStart).count();
    if(!source.validate(&problem))throw std::runtime_error("refined source: "+problem);
    std::vector<Job> jobs((source.patches.size()+patchesPerTask-1)/patchesPerTask);
    for(size_t i=0;i<jobs.size();++i) {
      jobs[i].id=int(i);
      jobs[i].firstPatch=int(i)*patchesPerTask;
      jobs[i].patchCount=std::min(patchesPerTask,int(source.patches.size())-jobs[i].firstPatch);
    }
    for(int f=0;f<source.faceCount();++f) {
      if(source.facePatchId[f]>=source.patches.size())throw std::runtime_error("invalid patch id");
      auto &job=jobs[source.facePatchId[f]/patchesPerTask];job.faces.push_back(f);
      job.cost+=.5*length(cross(source.facePoint(f,1)-source.facePoint(f,0),source.facePoint(f,2)-source.facePoint(f,0)));
    }
    int device=0;cudaCheck(cudaGetDevice(&device));cudaCheck(cudaFree(nullptr));
    size_t freeBytes=0,totalBytes=0;cudaCheck(cudaMemGetInfo(&freeBytes,&totalBytes));
    report.memoryBudget=options.memoryBytes?std::min(options.memoryBytes,size_t(freeBytes*.8)):size_t(freeBytes*.6);
    if(report.memoryBudget<16*1024*1024)throw std::runtime_error("insufficient CUDA task memory budget");
    const double h=cfg.featureEdgeLength>0?cfg.featureEdgeLength:cfg.constantLength;
    for(auto &job:jobs) {
      const double estimatedFaces=std::max(4.*job.faces.size(),4.*job.cost/(.433*h*h));
      job.cost=estimatedFaces;
      // Conservative capacity estimate, not a promise: DeviceArena enforces
      // this task's actual allocation cap and failures keep original geometry.
      job.bytes=size_t(std::min(double(report.memoryBudget),64.*1024*1024+estimatedFaces*1024));
    }
    std::vector<int> pending(jobs.size());std::iota(pending.begin(),pending.end(),0);
    std::stable_sort(pending.begin(),pending.end(),[&](int a,int b){return jobs[a].cost>jobs[b].cost;});
    report.patches.resize(jobs.size());
    std::mutex mutex;std::condition_variable changed;size_t reserved=0;int active=0;
    const int count=std::min({options.workers,options.gpuConcurrency,int(jobs.size())});
    CadMesh::RemeshWorkerPool pool(count);std::vector<std::future<void>> workers;
    for(int w=0;w<count;++w)workers.push_back(pool.submit([&]{
      cudaCheck(cudaSetDevice(device));
      for(;;) {
        int id=-1;
        {
          std::unique_lock<std::mutex> lock(mutex);
          changed.wait(lock,[&]{
            if(pending.empty())return true;
            for(int candidate:pending)if(jobs[candidate].bytes<=report.memoryBudget-reserved)return true;
            return false;
          });
          if(pending.empty())return;
          auto it=std::find_if(pending.begin(),pending.end(),[&](int candidate){return jobs[candidate].bytes<=report.memoryBudget-reserved;});
          id=*it;pending.erase(it);reserved+=jobs[id].bytes;++active;
          report.peakActive=std::max(report.peakActive,active);report.peakReserved=std::max(report.peakReserved,reserved);
        }
        execute(jobs[id],source,cfg,options.stopWhenIdle,options.smoothPasses,options.collapsePasses,options.flipPasses,options.strictFlipQuality,
                options.requireQualityImprovement,report.patches[id]);
        {std::lock_guard<std::mutex> lock(mutex);reserved-=jobs[id].bytes;--active;}
        changed.notify_all();
      }
    }));
    for(auto &worker:workers)worker.get();
    report.secondsTasks=std::chrono::duration<double>(Clock::now()-boundaryStart).count()-report.secondsBoundary;
    for(const auto &task:report.patches)report.retried+=int(task.retried);
    // Deterministic assembly, independent of which worker finishes first.
    SemanticMesh merged;
    std::vector<Job*> faceOwner;
    std::function<void(Job&)> appendJob=[&](Job &job) {
      if(!job.children.empty()) {
        for(auto &child:job.children)appendJob(child);
        return;
      }
      if(!job.succeeded) {
        ++report.fallback;
        for(int f:job.faces){auto t=source.face(f);const auto patch=source.facePatchId[f];merged.addFace(t[0],t[1],t[2],patch,source.patches[patch].type);faceOwner.push_back(nullptr);}
        return;
      }
      ++report.accepted;
      report.unchanged+=int(job.unchanged);
      report.uniformRegions+=int(job.uniformSizing);
      report.qualitySplitRegions+=int(job.qualitySplit);
      for(int &alias:job.aliases)if(alias>=source.vertexCount())alias=-1;
      for(int v=0;v<job.output.vertexCount();++v)if(job.aliases[v]<0)
        job.aliases[v]=merged.addVertex(job.output.position(v),job.firstPatch+job.output.vertexPatchId[v],VertexConstraint(job.output.vertexConstraint[v]));
      for(int v=0;v<job.output.vertexCount();++v) {
        float &h=merged.targetLength[job.aliases[v]];
        h=h>0?std::min(h,job.output.targetLength[v]):job.output.targetLength[v];
      }
      for(int f=0;f<job.output.faceCount();++f)if(job.output.faceAlive[f]){auto t=job.output.face(f);const auto patch=job.firstPatch+job.output.facePatchId[f];merged.addFace(job.aliases[t[0]],job.aliases[t[1]],job.aliases[t[2]],patch,source.patches[patch].type);faceOwner.push_back(&job);}
      for(auto feature:job.output.featureEdges)merged.featureEdges[edgeKey(job.aliases[int(feature.first>>32)],job.aliases[int(uint32_t(feature.first))])]=feature.second;
    };
    for(int repair=0;repair<=int(jobs.size());++repair) {
      merged=SemanticMesh{};merged.patches=source.patches;merged.featureEdges=source.featureEdges;
      for(int v=0;v<source.vertexCount();++v)
        merged.addVertex(source.position(v),source.vertexPatchId[v],VertexConstraint(source.vertexConstraint[v]));
      for(auto &h:merged.targetLength)h=cfg.constantLength;
      faceOwner.clear();report.accepted=report.unchanged=report.uniformRegions=
          report.qualitySplitRegions=report.fallback=0;
      for(auto &job:jobs)appendJob(job);
      merged.rebuildTopology();
      std::set<uint64_t> outputSeams;
      for(auto e:merged.edges)if(e.flags&(EdgePatchBoundary|EdgeMeshBoundary))outputSeams.insert(edgeKey(e.v0,e.v1));
      for(auto e:source.edges)if(e.flags&(EdgePatchBoundary|EdgeMeshBoundary))
        if(!outputSeams.erase(edgeKey(e.v0,e.v1)))throw std::runtime_error("global shared boundary missing");
      if(!outputSeams.empty())throw std::runtime_error("unexpected global seam or hole");
      if(merged.validate(&problem))break;
      uint32_t a=0,b=0;
      if(std::sscanf(problem.c_str(),"non-manifold edge %u,%u",&a,&b)!=2)
        throw std::runtime_error("assembled before compaction: "+problem);
      std::set<Job*> owners;
      for(int f=0;f<merged.faceCount();++f) {
        const auto t=merged.face(f);
        if(faceOwner[f] && std::find(t.begin(),t.end(),int(a))!=t.end() &&
                           std::find(t.begin(),t.end(),int(b))!=t.end())owners.insert(faceOwner[f]);
      }
      std::vector<Job*> repairCandidates;
      for(Job *candidate:owners) {
        bool sourceEdge=false;
        for(int f:candidate->faces) {
          const auto t=source.face(f);
          if(std::find(t.begin(),t.end(),int(a))!=t.end() &&
             std::find(t.begin(),t.end(),int(b))!=t.end()) {sourceEdge=true;break;}
        }
        if(!sourceEdge)repairCandidates.push_back(candidate);
      }
      if(repairCandidates.empty() || repair==int(jobs.size()))
        throw std::runtime_error("assembled before compaction: "+problem);
      std::sort(repairCandidates.begin(),repairCandidates.end(),
                [](const Job *x,const Job *y){return x->faces.size()<y->faces.size();});
      std::string repairReason;
      bool split=false;
      for(Job *candidate:repairCandidates)
        if(splitConflictingChord(*candidate,int(a),int(b),repairReason)){split=true;break;}
      if(split) {
        ++report.seamRepairs;
        ++report.seamSplitRepairs;
        continue;
      }
      Job *fallbackJob=repairCandidates.front();
      auto &taskReport=report.patches[fallbackJob->firstPatch/patchesPerTask];
      taskReport.accepted=false;
      if(!taskReport.error.empty())taskReport.error+="; ";
      taskReport.error+="assembly seam conflict at "+std::to_string(a)+","+std::to_string(b)+"; split rejected: "+repairReason;
      fallbackJob->succeeded=false;
      fallbackJob->output=SemanticMesh{};
      ++report.seamRepairs;
    }
    std::unordered_set<uint64_t> outputEdges;
    outputEdges.reserve(merged.edges.size());
    for(const auto &e:merged.edges)outputEdges.insert(edgeKey(e.v0,e.v1));
    for(const auto &feature:source.featureEdges)
      if(!outputEdges.count(feature.first))
        throw std::runtime_error("hard feature edge missing after assembly");
    merged.compact();merged.rebuildTopology();merged.computeVertexNormals();
    if(!merged.validate(&problem))throw std::runtime_error("assembled mesh: "+problem);
    report.topologyValid=report.boundariesHeld=true;
    report.secondsAssembly=std::chrono::duration<double>(Clock::now()-boundaryStart).count()-
        report.secondsBoundary-report.secondsTasks;
    report.seconds=std::chrono::duration<double>(Clock::now()-start).count();
    output=std::move(merged);return true;
  } catch(const std::exception &e) {
    report.seconds=std::chrono::duration<double>(Clock::now()-start).count();
    if(report.secondsTasks>0)
      report.secondsAssembly=std::max(0.,report.seconds-report.secondsBoundary-report.secondsTasks);
    if(error)*error=e.what();return false;
  }
}
}
