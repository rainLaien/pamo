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
  // Two distinct reference planes share one GPU task. Every output face must
  // retain its CAD owner and remain on that owner's plane after all operators.
  source=makeTwoPatchGrid(8,6,0,0,4,3);
  source.patches[1].axis={-0.70710678f,0,0.70710678f};
  for(int v=0;v<source.vertexCount();++v) {
    auto p=source.position(v);
    if(p.x>2) {p.z=p.x-2;source.setPosition(v,p);}
  }
  source.rebuildTopology();
  options.patchesPerTask=2;cfg.constantLength=.55f;cfg.maxGeometryError=.05f;cfg.maxIterations=3;
  CHECK(remeshRawCudaPatches(source,parallel,cfg,options,b,&error));
  if(b.fallback)std::cerr<<"packed fallback: "<<b.patches[0].error
    <<" geom="<<b.patches[0].report.geometryErrorMax
    <<" locked="<<b.patches[0].report.movedLockedVertices<<'\n';
  CHECK(b.patches.size()==1 && b.accepted==1 && b.fallback==0 && b.boundariesHeld);
  int owner0=0,owner1=0;
  for(int f=0;f<parallel.faceCount();++f) {
    const auto patch=parallel.facePatchId[f];
    CHECK(patch<2);
    if(patch==0)++owner0;else ++owner1;
    for(int k=0;k<3;++k) {
      const auto p=parallel.facePoint(f,k);
      CHECK_NEAR(p.z,patch==0?0.f:p.x-2.f,.051f);
    }
  }
  CHECK(owner0>0 && owner1>0);
  options.patchesPerTask=1;
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
