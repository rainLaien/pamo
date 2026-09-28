#include "../src/rxmesh/RawCudaRemesher.cu"
#include "check.h"
using namespace cad_adaptive;
using namespace cad_adaptive::raw;

__global__ void compareSizing(ReferenceSurfaceGpu indexed,int *failures) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=4096)return;
  auto linear=indexed;linear.sizingNodeCount=0;
  const auto p=make_float3(float(i%32)*.23f-3.5f,float((i/32)%32)*.21f-3.2f,float(i/1024)*.3f);
  for(int patch=0;patch<2;++patch)
    if(sizingAt(indexed,p,patch)!=sizingAt(linear,p,patch))atomicAdd(failures,1);
}

__global__ void compareBoundaryField(ReferenceSurfaceGpu ref,const float *expected,int *failures) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=1024)return;
  const auto p=make_float3(float(i%32)/8.f,float(i/32)/8.f,0);
  if(fabsf(sizingAt(ref,p,0)-expected[i])>2.e-6f)atomicAdd(failures,1);
  if(toleranceAt(ref,p,0)!=ref.tolerance)atomicAdd(failures,1);
}

__global__ void compareReferenceNear(ReferenceSurfaceGpu ref,int *failures) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=8192)return;
  const auto p=make_float3(float(i%32)*.125f-2.f,
                          float((i/32)%32)*.125f-2.f,
                          float(i/1024-4)*.03125f);
  for(int patch=0;patch<3;++patch) {
    auto linear=ref;linear.nodeCount=0;
    float3 q;
    const float tolerance=toleranceAt(ref,p,patch);
    const bool expected=projectReference(linear,patch,p,q) &&
                        dist2(p,q)<=tolerance*tolerance;
    if(referenceNear(ref,patch,p)!=expected)atomicAdd(failures,1);
    if(referenceNear(linear,patch,p)!=expected)atomicAdd(failures,1);
  }
}

int main() {
  DeviceArena arena;ArenaScope scope(arena);
  SemanticMesh mesh;mesh.patches.resize(1);
  mesh.addVertex({0,0,0},0,VertexConstraint::Surface);
  mesh.addVertex({1,0,0},0,VertexConstraint::Surface);
  mesh.addVertex({0,1,0},0,VertexConstraint::Surface);
  mesh.addFace(0,1,2,0,PatchType::Unknown);mesh.rebuildTopology();
  // Both vertices 0 and 2 project to the same source corner. Independent
  // nearest-point writes create a zero-area face despite distinct vertex IDs.
  std::vector<ReferenceTriangleGpu> source{{{0,-1,0},{1,-1,0},{0,0,0},0}};
  Buffer<ReferenceTriangleGpu> reference(source);
  ReferenceSurfaceGpu ref{reference.p,1,2.f};ref.regularLength=1.f;
  DeviceMesh d(mesh,ref);auto m=d.view();Buffer<Vertex> proposed(3);
  proposeProjection<<<1,128>>>(m,proposed.p,ref,nullptr);sync();
  const auto unsafe=proposed.read();
  auto v=[](float3 p){return Vec3{p.x,p.y,p.z};};
  CHECK(triangleQuality(v(unsafe[0].p),v(unsafe[1].p),v(unsafe[2].p))==0);
  CHECK(safeProject(m,ref)>0);
  auto safe=d.vertices.read();
  CHECK(triangleQuality(v(safe[0].p),v(safe[1].p),v(safe[2].p))>0);
  for(int i=0;i<3;++i)CHECK(distance(v(safe[i].p),mesh.position(i))==0);
  // A harmless common translation remains accepted rather than disabling projection.
  source[0]={{-10,-10,1},{10,-10,1},{0,10,1},0};
  checked(cudaMemcpy(reference.p,source.data(),sizeof(ReferenceTriangleGpu),cudaMemcpyHostToDevice));
  CHECK(safeProject(m,ref)==0);safe=d.vertices.read();
  for(auto x:safe)CHECK_NEAR(x.p.z,1.f,1.e-6f);
  // Nearest projections onto separate small surface islands preserve triangle
  // quality but move its interior beyond the error budget. The old triangle
  // lies 0.1 above the broad reference plane and is within the 0.11 budget.
  std::vector<ReferenceTriangleGpu> islands{{{-10,-10,.9f},{10,-10,.9f},{0,10,.9f},0}};
  for(auto x:safe) {
    auto p=x.p;
    islands.push_back({{p.x-.03f,p.y-.03f,1.05f},
                       {p.x+.06f,p.y-.03f,1.05f},
                       {p.x-.03f,p.y+.06f,1.05f},0});
  }
  Buffer<ReferenceTriangleGpu> islandReference(islands);
  ref.triangles=islandReference.p;ref.count=int(islands.size());ref.tolerance=.11f;
  int errorRejects=0;
  CHECK(safeProject(m,ref,&errorRejects)>0);
  CHECK(errorRejects>0);
  safe=d.vertices.read();
  for(auto x:safe)CHECK_NEAR(x.p.z,1.f,1.e-6f);
  // Spatial pruning must reproduce the original sizing field exactly.
  std::vector<SizingSegmentGpu> seeds;
  std::vector<ReferenceTriangleGpu> bounds;
  for(int i=0;i<128;++i) {
    float3 a=make_float3(float(i%16)*.3f-2.f,float(i/16)*.4f-1.5f,0);
    float3 b=make_float3(a.x+.17f,a.y+.2f,.4f);
    seeds.push_back({a,b,.2f+.001f*i,i%2});bounds.push_back({a,b,a,i%2});
  }
  std::vector<ReferenceBvhNode> nodes;std::vector<int> ids;
  buildReferenceBvh(bounds,1.e-5f,nodes,ids);
  Buffer<SizingSegmentGpu> sizes(seeds);Buffer<ReferenceBvhNode> tree(nodes);Buffer<int> order(ids),failures(1);
  ref.sizing=sizes.p;ref.sizingCount=int(seeds.size());ref.sizingNodes=tree.p;ref.sizingNodeCount=int(nodes.size());ref.sizingIds=order.p;ref.band=.75f;
  checked(cudaMemset(failures.p,0,sizeof(int)));
  compareSizing<<<32,128>>>(ref,failures.p);sync();CHECK(failures.read()[0]==0);
  // Compare radius witnesses to brute-force nearest-point acceptance, including
  // missing patches, exact tolerance boundaries and the spatial sizing field.
  std::vector<ReferenceTriangleGpu> surface;
  for(int i=0;i<64;++i) {
    float x=float(i%8)*.5f-2.f,y=float(i/8)*.5f-2.f;
    surface.push_back({{x,y,0},{x+.5f,y,0},{x,y+.5f,0},i%2});
  }
  buildReferenceBvh(surface,1.e-5f,nodes,ids);
  Buffer<ReferenceTriangleGpu> surfaceBuffer(surface);
  Buffer<ReferenceBvhNode> surfaceTree(nodes);Buffer<int> surfaceOrder(ids);
  ref.triangles=surfaceBuffer.p;ref.count=int(surface.size());
  ref.nodes=surfaceTree.p;ref.nodeCount=int(nodes.size());ref.triangleIds=surfaceOrder.p;
  for(float tolerance:{0.f,.03125f,.125f,.25f}) {
    ref.tolerance=tolerance;
    for(bool sized:{false,true}) {
      auto query=ref;if(!sized)query.sizingCount=0;
      compareReferenceNear<<<64,128>>>(query,failures.p);sync();
      CHECK(failures.read()[0]==0);
    }
  }
  // A retry cannot reset its error budget by treating displaced intermediate
  // geometry as a new reference. Locked vertices make the expected failure
  // independent of smoothing or nearest-point motion.
  SemanticMesh immutable;
  immutable.patches.resize(1);
  immutable.addVertex({0,0,0},0,VertexConstraint::Locked);
  immutable.addVertex({1,0,0},0,VertexConstraint::Locked);
  immutable.addVertex({0,1,0},0,VertexConstraint::Locked);
  immutable.addFace(0,1,2,0,PatchType::Unknown);
  immutable.rebuildTopology();
  SemanticMesh displaced=immutable;
  for(int i=0;i<3;++i)displaced.pz[i]=.2f;
  RemeshConfig cfg;cfg.adaptive=false;cfg.constantLength=2.f;
  cfg.maxGeometryError=.1f;cfg.maxIterations=1;
  cfg.enableSplit=cfg.enableCollapse=cfg.enableFlip=cfg.enableSmooth=false;
  RawCudaOptions options;options.quiet=true;options.freezeBoundary=true;
  RemeshReport audit;
  auto rebased=displaced;
  CHECK(remeshRawCuda(rebased,cfg,audit,options));
  options.referenceMesh=&immutable;
  CHECK(!remeshRawCuda(displaced,cfg,audit,options));
  CHECK(audit.geometryErrorMax>.19f);
  auto cropped=immutable;
  for(int i=0;i<3;++i) {cropped.px[i]*=.5f;cropped.py[i]*=.5f;}
  CHECK(!remeshRawCuda(cropped,cfg,audit,options));
  CHECK(audit.geometryErrorMax<1.e-5f);
  CHECK(audit.geometryErrorReverseMax>.49f);
  // Zero endpoint floors must not disable optional legal-cycle tracking.
  // A poor source is not selected as success just because it is immutable.
  auto trackSource=makeGrid(6,6,0,0,3,3,0);auto tracked=trackSource;
  cfg=RemeshConfig{};cfg.adaptive=false;cfg.constantLength=.8f;cfg.maxGeometryError=.1f;cfg.maxIterations=3;
  options=RawCudaOptions{};options.quiet=true;options.freezeBoundary=true;options.smoothPasses=1;
  options.referenceMesh=&trackSource;options.trackRegionCandidates=true;options.lowQualityThreshold=.99f;
  CHECK(remeshRawCuda(tracked,cfg,audit,options));
  CHECK(audit.candidateGeometryChecks>0 && audit.selectedCycle>=0);
  CHECK(audit.geometryErrorMax<=cfg.maxGeometryError && audit.geometryErrorReverseMax<=cfg.maxGeometryError);
  CHECK(tracked.faceCount()!=trackSource.faceCount());
  auto completed=trackSource;auto completedOptions=options;completedOptions.trackRegionCandidates=false;
  RemeshReport completedReport;
  CHECK(remeshRawCuda(completed,cfg,completedReport,completedOptions));
  const auto completedQuality=evaluateRegionQuality(completed,cfg,options.lowQualityThreshold);
  const auto trackedQuality=evaluateRegionQuality(tracked,cfg,options.lowQualityThreshold);
  CHECK(qualityNonRegression(completedQuality,trackedQuality));
  CHECK(trackedQuality.longEdges<=completedQuality.longEdges);
  CHECK(trackedQuality.shortEdges<=completedQuality.shortEdges);
  // A task slice must reproduce the parent field and the GPU must evaluate
  // the same gradation. Explicit sizing does not silently change geometry tolerance.
  auto sizingMesh=makeTwoPatchGrid(8,8,0,0,4,4);
  cfg.constantLength=1.f;
  const auto fullField=BoundarySizingField::create(sizingMesh,cfg,.5f,false,{.25f,1.f},false);
  const auto slice=fullField->subset(1,1);
  std::vector<float> expected(1024);
  for(int i=0;i<1024;++i) {
    const Vec3 p{float(i%32)/8.f,float(i/32)/8.f,0};
    expected[i]=fullField->evaluate(1,p);
    CHECK_NEAR(expected[i],slice->evaluate(0,p),1.e-7f);
  }
  CHECK_NEAR(fullField->evaluate(0,{.5f,2,0}),.25f,1.e-7f);
  CHECK_NEAR(fullField->evaluate(1,{2.5f,2,0}),.5f,1.e-7f);
  CHECK_NEAR(fullField->evaluate(1,{4,2,0}),1.f,1.e-7f);
  Buffer<BoundarySizingPatch> bp(slice->patches());
  Buffer<BoundarySizingNode> bn(slice->nodes());Buffer<BoundarySizingSeed> bs(slice->seeds());
  Buffer<float> expectedBuffer(expected);
  ReferenceSurfaceGpu shared;shared.regularLength=1.f;shared.tolerance=.1f;
  shared.boundaryPatches=bp.p;shared.boundaryNodes=bn.p;shared.boundarySeeds=bs.p;
  shared.boundaryPatchCount=1;shared.boundaryGradation=.5f;
  compareBoundaryField<<<8,128>>>(shared,expectedBuffer.p,failures.p);sync();
  CHECK(failures.read()[0]==0);
  // Cone sizing uses the smallest sampled radial distance and the cone's
  // circumferential principal curvature, while the plane stays at global h.
  sizingMesh.patches[1].type=PatchType::Cone;
  sizingMesh.patches[1].origin={0,0,0};
  sizingMesh.patches[1].axis={0,0,1};
  sizingMesh.patches[1].radius=.34906585f; // semi-angle, 20 degrees
  cfg.constantLength=8.f;
  const auto coneField=BoundarySizingField::create(sizingMesh,cfg,.5f,true,{},false);
  const double curvatureRadius=2.0/std::cos(double(sizingMesh.patches[1].radius));
  const double epsilon=std::min(double(cfg.maxGeometryError),curvatureRadius);
  const double chord=2.0*std::sqrt(2.0*curvatureRadius*epsilon-epsilon*epsilon);
  const double normalChord=2.0*curvatureRadius*std::sin(double(cfg.normalDegrees)*.017453292519943295);
  const float coneTarget=float(.95*std::min(chord,normalChord)/cfg.splitRatio);
  CHECK_NEAR(coneField->patches()[0].BaseLength,8.f,1.e-6f);
  CHECK_NEAR(coneField->patches()[1].BaseLength,coneTarget,1.e-5f);
  return test_result("raw_projection_safety");
}
