#include "../src/rxmesh/RawCudaRemesher.cu"
#include "check.h"
using namespace cad_adaptive;
using namespace cad_adaptive::raw;

__global__ void compareSizing(ReferenceSurfaceGpu indexed,int *failures) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=4096)return;
  auto linear=indexed;linear.sizingNodeCount=0;
  const auto p=make_float3(float(i%32)*.23f-3.5f,float((i/32)%32)*.21f-3.2f,float(i/1024)*.3f);
  if(sizingAt(indexed,p)!=sizingAt(linear,p))atomicAdd(failures,1);
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
    const float tolerance=toleranceAt(ref,p);
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
  proposeProjection<<<1,128>>>(m,proposed.p,ref);sync();
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
    seeds.push_back({a,b,.2f+.001f*i});bounds.push_back({a,b,a,0});
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
  return test_result("raw_projection_safety");
}
