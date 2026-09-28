#include "cad_adaptive/RawCudaBatch.h"
#include "cad_adaptive/BoundarySizingField.h"
#include "cad_adaptive/TriangleRefineBackend.h"
#include "check.h"
#include <numeric>
using namespace cad_adaptive;
int main(){
  // Off-center quality maximization must not leave a long child if both
  // children can satisfy the unchanged endpoint sizing in one level.
  SemanticMesh thin;thin.patches.resize(1);
  thin.addVertex({0,0,0},0,VertexConstraint::Locked);
  thin.addVertex({2,0,0},0,VertexConstraint::Locked);
  thin.addVertex({0,.01f,0},0,VertexConstraint::Locked);
  thin.addFace(0,1,2,0,PatchType::Unknown);thin.rebuildTopology();
  thin.targetLength={1,1,10};
  RemeshConfig thinCfg;thinCfg.adaptive=false;thinCfg.constantLength=1;
  GeometryProjector unused;SemanticMesh biased,feasible;TriangleRefineStats oldStats,newStats;
  CHECK(refineMidpointConforming(thin,thinCfg,unused,biased,oldStats,nullptr,nullptr,false));
  CHECK(refineMidpointConforming(thin,thinCfg,unused,feasible,newStats,nullptr,nullptr,false,nullptr,true));
  CHECK(oldStats.SplitEdges==1 && newStats.SplitEdges==1);
  CHECK(evaluateRegionQuality(biased,thinCfg,.01f).longEdges>0);
  CHECK(evaluateRegionQuality(feasible,thinCfg,.01f).longEdges==0);
  CHECK(feasible.position(0).x==thin.position(0).x && feasible.position(1).x==thin.position(1).x);
  CHECK(feasible.faceCount()==2 && feasible.validate());
  // Large-coordinate narrow triangles admit safe intervals missed by the
  // coarse decimal fractions. Use the same bounded fallback for any patch.
  SemanticMesh quantized;quantized.patches.resize(1);
  for(Vec3 p:std::array<Vec3,4>{{{183.341796875f,42.81513977050781f,391.56256103515625f},
      {183.34043884277344f,42.85436248779297f,405.9156494140625f},
      {183.3409423828125f,42.83970260620117f,400.5509948730469f},
      {183.34112548828125f,42.83464813232422f,398.70135498046875f}}})
    quantized.addVertex(p,0,VertexConstraint::Surface);
  quantized.addFace(2,1,0,0,PatchType::Unknown);quantized.addFace(1,3,0,0,PatchType::Unknown);
  quantized.rebuildTopology();for(auto &h:quantized.targetLength)h=6.861335754f;
  thinCfg.constantLength=6.861335754f;
  CHECK(refineMidpointConforming(quantized,thinCfg,unused,feasible,newStats,nullptr,nullptr,false,nullptr,true));
  CHECK(newStats.SplitEdges==1 && feasible.faceCount()==4);
  CHECK(evaluateRegionQuality(feasible,thinCfg,.01f).longEdges==0);
  for(int parent=0;parent<2;++parent)for(int child=0;child<2;++child)
    CHECK(dot(triangleNormal(quantized.facePoint(parent,0),quantized.facePoint(parent,1),quantized.facePoint(parent,2)),
              triangleNormal(feasible.facePoint(parent*2+child,0),feasible.facePoint(parent*2+child,1),feasible.facePoint(parent*2+child,2)))>0);
  auto contract=makeGrid(2,2,0,0,1,1,0);contract.rebuildTopology();
  std::vector<int> anchors(contract.vertexCount());std::iota(anchors.begin(),anchors.end(),0);
  for(auto &h:contract.targetLength)h=1.f;
  auto changedTargets=contract;changedTargets.targetLength[0]=.5f;
  CHECK(boundaryTargetMismatchCount(contract,anchors,contract,anchors)==0);
  CHECK(boundaryTargetMismatchCount(contract,anchors,changedTargets,anchors)==1);
  changedTargets.targetLength[0]=1.5f;
  CHECK(boundaryTargetMismatchCount(contract,anchors,changedTargets,anchors)==1);
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
  // A region already within its size limits can remain unchanged. It is not
  // classified as failed coverage or forced to manufacture a topology change.
  source=makeGrid(1,1,0,0,1,1,0);cfg.constantLength=5;cfg.maxIterations=1;
  options.requireQualityImprovement=true;
  CHECK(remeshRawCudaPatches(source,parallel,cfg,options,b,&error));
  CHECK(parallel.faceCount()==source.faceCount());CHECK(b.unchanged==1);
  CHECK(b.qualityAccepted && b.patches[0].qualityAccepted && !b.patches[0].provisional);
  CHECK(b.finalSizeRefineSplits==0 && b.fallback==0);
  options.requireQualityImprovement=false;cfg.constantLength=.55f;cfg.maxIterations=3;
  source=makeTwoPatchGrid(8,6,0,0,4,3);
  const auto field=BoundarySizingField::create(source,cfg,.5f,false,{.25f,.55f},false);
  source.LocalSizing=field;field->apply(source);
  options.patchesPerTask=1;
  CHECK(remeshRawCudaPatches(source,parallel,cfg,options,b,&error));
  CHECK(b.fallback==0 && b.boundariesHeld && parallel.LocalSizing==field);
  auto expected=parallel;field->apply(expected);
  for(int v=0;v<parallel.vertexCount();++v)CHECK_NEAR(parallel.targetLength[v],expected.targetLength[v],1.e-6f);
  CHECK(*std::min_element(parallel.targetLength.begin(),parallel.targetLength.end())<.3f);
  CHECK(*std::max_element(parallel.targetLength.begin(),parallel.targetLength.end())>.5f);
  // Assembly must not enlarge a finer target returned by geometric recovery
  // in a different region when applying the explicit regional upper envelope.
  auto recovered=source;field->apply(recovered);
  recovered.targetLength[0]=.1f;
  field->apply(recovered,true);
  CHECK_NEAR(recovered.targetLength[0],.1f,1.e-7f);
  // Bounded exploration cannot replace a legal incumbent with a regional
  // quality or size regression, or turn exploration failure into input fallback.
  source=makeTwoPatchGrid(8,6,0,0,4,3);
  options.requireQualityImprovement=true;options.patchesPerTask=2;
  options.lowQualityThreshold=.99f;cfg.maxIterations=1;
  SemanticMesh incumbent,explored;RawBatchReport incumbentReport,exploredReport;
  CHECK(remeshRawCudaPatches(source,incumbent,cfg,options,incumbentReport,&error));
  options.exploreProvisionalChildren=true;
  CHECK(remeshRawCudaPatches(source,explored,cfg,options,exploredReport,&error));
  CHECK(exploredReport.fallback==0 && exploredReport.boundariesHeld);
  CHECK(exploredReport.patches[0].childCandidateCompared);
  CHECK(exploredReport.patches[0].childGeneratedPatches==2);
  CHECK(!exploredReport.patches[0].childComparisonReason.empty());
  CHECK(qualityNonRegression(incumbentReport.outputQuality,exploredReport.outputQuality));
  CHECK(exploredReport.outputQuality.longEdges<=incumbentReport.outputQuality.longEdges);
  CHECK(exploredReport.outputQuality.shortEdges<=incumbentReport.outputQuality.shortEdges);
  for(size_t p=0;p<incumbentReport.outputPatchQuality.size();++p)
    CHECK(qualityNonRegression(incumbentReport.outputPatchQuality[p],exploredReport.outputPatchQuality[p]));
  options.exploreProvisionalChildren=false;options.requireQualityImprovement=false;
  options.lowQualityThreshold=0;options.patchesPerTask=1;
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
