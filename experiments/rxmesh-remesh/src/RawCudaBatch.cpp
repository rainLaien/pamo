#include "cad_adaptive/RawCudaBatch.h"
#include "cad_adaptive/PartitionInput.h"
#include "RemeshWorkerPool.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>

namespace cad_adaptive {
namespace {
using Clock=std::chrono::steady_clock;
using Position=std::tuple<float,float,float>;
Position positionKey(Vec3 p){return {p.x,p.y,p.z};}
uint64_t edgeKey(int a,int b){return (uint64_t(std::min(a,b))<<32)|uint32_t(std::max(a,b));}
void cudaCheck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Job {
  int id=0; double cost=0; size_t bytes=0;
  std::vector<int> faces;
  SemanticMesh output;
  std::vector<int> aliases;
};
void execute(Job &job,const SemanticMesh &source,const RemeshConfig &config,RawPatchResult &result) {
  const auto start=Clock::now();
  result.inputFaces=int(job.faces.size());result.workspaceBytes=job.bytes;
  try {
    SemanticMesh local;local.patches.resize(1);
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
          v=local.addVertex(p,0,constraint);originals.push_back(original);
          indices.emplace(original,v);
          if(constraint==VertexConstraint::Locked && !anchors.emplace(positionKey(p),original).second)
            throw std::runtime_error("ambiguous coincident boundary identities");
        } else v=found->second;
      }
      local.addFace(t[0],t[1],t[2],0,PatchType::Unknown);
    }
    for(auto feature:source.featureEdges) {
      auto a=indices.find(int(feature.first>>32)),b=indices.find(int(uint32_t(feature.first)));
      if(a!=indices.end() && b!=indices.end())local.featureEdges[edgeKey(a->second,b->second)]=feature.second;
    }
    local.rebuildTopology();
    std::set<uint64_t> required;
    for(auto e:local.edges) if(e.flags&EdgeMeshBoundary)
      required.insert(edgeKey(originals[e.v0],originals[e.v1]));
    RawCudaOptions options;options.independentStream=true;options.quiet=true;options.error=&result.error;options.freezeBoundary=true;options.workspaceBytes=job.bytes;
    if(!remeshRawCuda(local,config,result.report,options))throw std::runtime_error(result.error.empty()?"raw CUDA task failed":result.error);
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
    result.outputFaces=local.faceCount();result.accepted=true;job.output=std::move(local);
  } catch(const std::exception &e) {result.error=e.what();result.outputFaces=result.inputFaces;}
  result.seconds=std::chrono::duration<double>(Clock::now()-start).count();
}
}

bool remeshRawCudaPatches(const SemanticMesh &input,SemanticMesh &output,const RemeshConfig &cfg,
                         const RawBatchOptions &options,RawBatchReport &report,std::string *error) {
  report={};const auto start=Clock::now();
  try {
    if(options.workers<1 || options.workers>32 || !(cfg.constantLength>0) || !std::isfinite(cfg.constantLength) || !(cfg.maxGeometryError>0) || !std::isfinite(cfg.maxGeometryError) || cfg.maxIterations<1 || cfg.adaptive)
      throw std::runtime_error("invalid batch worker count or sizing configuration");
    SemanticMesh source=input;source.rebuildTopology();
    std::string problem;if(!source.validate(&problem))throw std::runtime_error(problem);
    for(auto e:source.edges)if(e.flags&(EdgePatchBoundary|EdgeMeshBoundary)) {
      source.vertexConstraint[e.v0]=source.vertexConstraint[e.v1]=uint8_t(VertexConstraint::Locked);
    }
    // Both incident patches receive exactly the same new vertices. This runs
    // once before creating jobs, never independently inside a patch worker.
    report.boundarySplits=refinePartitionBoundary(source,cfg);
    std::vector<Job> jobs(source.patches.size());
    for(size_t i=0;i<jobs.size();++i)jobs[i].id=int(i);
    for(int f=0;f<source.faceCount();++f) {
      if(source.facePatchId[f]>=jobs.size())throw std::runtime_error("invalid patch id");
      auto &job=jobs[source.facePatchId[f]];job.faces.push_back(f);
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
    const int count=std::min(options.workers,int(jobs.size()));
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
        execute(jobs[id],source,cfg,report.patches[id]);
        {std::lock_guard<std::mutex> lock(mutex);reserved-=jobs[id].bytes;--active;}
        changed.notify_all();
      }
    }));
    for(auto &worker:workers)worker.get();
    // Deterministic assembly, independent of which worker finishes first.
    SemanticMesh merged;merged.patches=source.patches;
    merged.featureEdges=source.featureEdges;
    for(int v=0;v<source.vertexCount();++v)merged.addVertex(source.position(v),source.vertexPatchId[v],VertexConstraint(source.vertexConstraint[v]));
    for(auto &h:merged.targetLength)h=cfg.constantLength;
    for(auto &job:jobs) {
      if(!report.patches[job.id].accepted) {
        ++report.fallback;
        for(int f:job.faces){auto t=source.face(f);merged.addFace(t[0],t[1],t[2],job.id,source.patches[job.id].type);}
        continue;
      }
      ++report.accepted;
      for(int v=0;v<job.output.vertexCount();++v)if(job.aliases[v]<0)
        job.aliases[v]=merged.addVertex(job.output.position(v),job.id,VertexConstraint(job.output.vertexConstraint[v]));
      for(int v=0;v<job.output.vertexCount();++v) {
        float &h=merged.targetLength[job.aliases[v]];
        h=h>0?std::min(h,job.output.targetLength[v]):job.output.targetLength[v];
      }
      for(int f=0;f<job.output.faceCount();++f){auto t=job.output.face(f);merged.addFace(job.aliases[t[0]],job.aliases[t[1]],job.aliases[t[2]],job.id,source.patches[job.id].type);}
      for(auto feature:job.output.featureEdges)merged.featureEdges[edgeKey(job.aliases[int(feature.first>>32)],job.aliases[int(uint32_t(feature.first))])]=feature.second;
    }
    merged.rebuildTopology();
    std::set<uint64_t> outputSeams;
    for(auto e:merged.edges)if(e.flags&(EdgePatchBoundary|EdgeMeshBoundary))outputSeams.insert(edgeKey(e.v0,e.v1));
    for(auto e:source.edges)if(e.flags&(EdgePatchBoundary|EdgeMeshBoundary))
      if(!outputSeams.erase(edgeKey(e.v0,e.v1)))throw std::runtime_error("global shared boundary missing");
    if(!outputSeams.empty())throw std::runtime_error("unexpected global seam or hole");
    merged.compact();merged.rebuildTopology();merged.computeVertexNormals();
    if(!merged.validate(&problem))throw std::runtime_error("assembled mesh: "+problem);
    report.topologyValid=report.boundariesHeld=true;
    report.seconds=std::chrono::duration<double>(Clock::now()-start).count();
    output=std::move(merged);return true;
  } catch(const std::exception &e) {
    report.seconds=std::chrono::duration<double>(Clock::now()-start).count();
    if(error)*error=e.what();return false;
  }
}
}
