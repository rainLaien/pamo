#include "cad_adaptive/RegionQuality.h"
#include "check.h"
#include <sstream>
using namespace cad_adaptive;
int main(){
  RegionQuality poor,better;
  poor.mean=.4f;poor.p05=.01f;poor.areaWeightedMean=.4;
  poor.lowQualityFaces=10;poor.lowQualityArea=100;poor.largestLowQualityArea=50;
  better=poor;better.mean=.5f;better.p05=.02f;better.areaWeightedMean=.5;
  better.lowQualityFaces=2;better.lowQualityArea=10;better.largestLowQualityArea=5;
  CHECK(!batchQualityEndpointAccepted(poor,better,2,false));
  CHECK(batchQualityEndpointAccepted(poor,better,2,true));
  CHECK(batchQualityEndpointAccepted(poor,better,0,false));
  CHECK(!batchQualityEndpointAccepted(poor,poor,2,true));
  auto regressedQuality=better;regressedQuality.areaWeightedMean=.3;
  CHECK(!batchQualityEndpointAccepted(poor,regressedQuality,2,true));
  auto resolvedQuality=better;resolvedQuality.lowQualityFaces=0;resolvedQuality.lowQualityArea=0;
  CHECK(batchQualityEndpointAccepted(resolvedQuality,resolvedQuality,1,true));
  const SizeDefectSummary plateau{4,2.1f,2.};
  CHECK(!sizeDefectProgress(plateau,{4,2.1f,1.5},false));
  CHECK(sizeDefectProgress(plateau,{4,2.1f,1.5},true));
  CHECK(!sizeDefectProgress(plateau,plateau,true));
  CHECK(!sizeDefectProgress(plateau,{4,2.1f,2.5},true));
  CHECK(!sizeDefectProgress(plateau,{4,2.2f,1.5},true));
  RemeshConfig cfg;cfg.adaptive=false;cfg.constantLength=1;
  auto mesh=makeTwoPatchGrid(4,2,0,0,4,2);mesh.rebuildTopology();
  auto a=evaluateRegionQuality(mesh,cfg,.99f);
  CHECK(a.lowQualityComponents==1); // Must not fragment at the artificial cut.
  CHECK(a.regions.size()==1 && a.regions[0].patches.size()==2);
  CHECK_NEAR(a.lowQualityArea,8,1.e-5);CHECK(a.constraints.partitionInterfaceEdges>0);
  CHECK(a.constraints.openBoundaryEdges>0);CHECK(a.inconsistentEdges==0);
  auto sized=mesh;for(auto &h:sized.targetLength)h=2.f;
  CHECK(longEdgePatches(sized,cfg).empty());
  for(auto &h:sized.targetLength)h=.1f;
  const auto sizePatches=longEdgePatches(sized,cfg);
  CHECK(sizePatches==std::vector<uint32_t>({0,1}));
  CHECK(qualityNonRegression(a,a));
  CHECK(qualityEndpointMask(a,a)&UnresolvedInputDefect);
  const auto healthy=evaluateRegionQuality(mesh,cfg,.1f);
  CHECK(qualityEndpointMask(healthy,healthy)==0);
  for(const auto&e:mesh.edges)if(e.flags&EdgePatchBoundary){
    auto c=classifyConstraint(mesh,e);CHECK(c.roles&PartitionInterface);
    CHECK(c.allowedMotion==ConstraintMotion::Fixed);CHECK(!c.geometryProvenanceKnown);
  }
  auto damaged=mesh;for(int v=0;v<damaged.vertexCount();++v)damaged.py[v]*=.001f;
  auto b=evaluateRegionQuality(damaged,cfg,.99f);
  CHECK(damaged.vertexCount()==mesh.vertexCount());CHECK(b.zeroFaces==0);
  CHECK(!qualityNonRegression(a,b)); // Legal and moved is insufficient.
  CHECK(b.shortEdges>0 && b.shortConstrainedEdges>0);
  auto splitRegion=mesh;
  // A winding error must be visible even when face quality is unchanged.
  std::swap(splitRegion.i0[0],splitRegion.i1[0]);splitRegion.rebuildTopology();
  auto c=evaluateRegionQuality(splitRegion,cfg,.99f);CHECK(c.inconsistentEdges>0);
  CHECK(!qualityNonRegression(a,c));
  auto smallDefect=a;smallDefect.area=1.e6;smallDefect.lowQualityArea=.001;smallDefect.largestLowQualityArea=.001;
  auto smallRegression=smallDefect;smallRegression.lowQualityArea+=1.e-6;smallRegression.largestLowQualityArea+=1.e-6;
  CHECK(qualityRegressionMask(smallDefect,smallRegression)&LowAreaRegression);
  CHECK(qualityRegressionMask(smallDefect,smallRegression)&LargestRegionRegression);
  auto smallProgress=smallDefect;smallProgress.lowQualityArea-=1.e-5;
  CHECK(!(qualityEndpointMask(smallDefect,smallProgress)&UnresolvedInputDefect));
  std::ostringstream json;writeRegionQualityJson(json,a);
  CHECK(json.str().find("partition_interfaces_unknown_provenance")!=std::string::npos);
  return test_result("region_quality");
}
