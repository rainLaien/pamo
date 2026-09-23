#include "cad_adaptive/RawCudaBatch.h"
#include "check.h"
using namespace cad_adaptive;
int main(){
  // Unequal adjacent regions with a shared seam and a source crease.
  auto source=makeGrid(18,12,0,0,6,4,0);source.patches.resize(3);
  for(int f=0;f<source.faceCount();++f){auto p=centroid3(source.facePoint(f,0),source.facePoint(f,1),source.facePoint(f,2));source.facePatchId[f]=p.x<1?0:p.x<2?1:2;}
  source.rebuildTopology();
  RemeshConfig cfg;cfg.adaptive=false;cfg.constantLength=.5f;cfg.maxGeometryError=.1f;cfg.maxIterations=3;
  RawBatchOptions options;options.workers=1;options.memoryBytes=512ull*1024*1024;
  SemanticMesh serial,parallel;RawBatchReport a,b;std::string error;
  CHECK(remeshRawCudaPatches(source,serial,cfg,options,a,&error));CHECK(a.fallback==0);CHECK(a.accepted==3);
  options.workers=3;
  CHECK(remeshRawCudaPatches(source,parallel,cfg,options,b,&error));CHECK(b.fallback==0);CHECK(b.peakActive>1);
  CHECK(b.peakReserved<=b.memoryBudget);CHECK(b.topologyValid && b.boundariesHeld);
  CHECK(serial.i0==parallel.i0 && serial.i1==parallel.i1 && serial.i2==parallel.i2);
  CHECK(serial.px==parallel.px && serial.py==parallel.py && serial.pz==parallel.pz);
  CHECK(serial.facePatchId==parallel.facePatchId);
  // Globally subdivide long shared edges; independent workers must retain them.
  source=makeTwoPatchGrid(2,2,0,0,4,4);cfg.constantLength=.7f;cfg.maxIterations=2;
  CHECK(remeshRawCudaPatches(source,parallel,cfg,options,b,&error));
  CHECK(b.boundarySplits>0 && b.fallback==0 && b.boundariesHeld);
  // A task exceeding its allocation cap retains valid source geometry and
  // reports fallback, rather than silently claiming it was remeshed.
  source=makeGrid(256,256,0,0,4,4,0);options.memoryBytes=16ull*1024*1024;
  CHECK(remeshRawCudaPatches(source,parallel,cfg,options,b,&error));
  CHECK(b.fallback==1 && b.accepted==0 && b.boundariesHeld);
  CHECK(parallel.faceCount()==source.faceCount());
  CHECK(!b.patches[0].error.empty());
  // Bad configuration leaves caller output untouched.
  cfg.adaptive=true;
  CHECK(!remeshRawCudaPatches(source,parallel,cfg,options,b,&error));
  return test_result("raw_batch_safety");
}
