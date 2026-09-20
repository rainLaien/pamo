#include "cad_adaptive/GeometryProjector.h"
#include "cad_adaptive/RxMeshBackend.h"
#include "check.h"
using namespace cad_adaptive;

int main() {
  // An unlabelled distorted plane exercises the raw reference path, rather
  // than the analytic plane projector used by the older smoothing tests.
  auto source = makeGrid(8, 8, 0, 0, 1, 1, 0);
  source.patches[0].type = PatchType::Unknown;
  for (auto &t : source.facePatchType)
    t = uint8_t(PatchType::Unknown);
  lockMeshBoundary(source);
  source.setPosition(40, {.44f, .56f, 0});
  GeometryProjector reference;
  reference.build(source);
  RxMeshBackend backend;
  RemeshConfig c;
  c.adaptive = false;
  c.constantLength = .125f;
  c.maxGeometryError = .01f;
  c.maxIterations = 4;
  c.enableSplit = c.enableCollapse = c.enableFlip = false;
  auto mesh = source;
  RemeshReport report;
  CHECK(backend.remesh(mesh, c, report));
  CHECK(report.smoothMoves > 0);
  CHECK(mesh.validate());
  CHECK(mesh.vertexCount() == source.vertexCount());
  CHECK(mesh.faceCount() == source.faceCount());
  for (int v = 0; v < mesh.vertexCount(); ++v) {
    const auto p = mesh.position(v);
    CHECK_NEAR(p.z, 0, 1.e-6f);
    if (VertexConstraint(mesh.vertexConstraint[v]) == VertexConstraint::Locked) {
      float nearest = 1.e20f;
      for (int u = 0; u < source.vertexCount(); ++u)
        nearest = std::min(nearest, distance(p, source.position(u)));
      CHECK(nearest < 1.e-6f);
    }
  }
  for (int f = 0; f < mesh.faceCount(); ++f) {
    auto a = mesh.facePoint(f, 0), b = mesh.facePoint(f, 1), d = mesh.facePoint(f, 2);
    CHECK(cross(b - a, d - a).z > 0);
    CHECK(distance(centroid3(a, b, d), reference.projectSurface(0, centroid3(a, b, d)).position) <
          1.e-6f);
  }
  mesh = source;
  c.smoothLambda = 0;
  CHECK(backend.remesh(mesh, c, report));
  CHECK(report.smoothMoves == 0);
  // The user's coarse, closed STL is the acceptance regression. These are
  // quality/geometry contracts, not a byte-for-byte topology golden file.
  std::string error;
  CHECK(source.load(RAW_BODY_FIXTURE, &error));
  reference.build(source);
  mesh = source;
  c = RemeshConfig{};
  c.adaptive = false;
  c.constantLength = source.bboxDiagonal() * .01f;
  c.maxGeometryError = c.constantLength * .2f;
  c.maxIterations = 20;
  CHECK(backend.remesh(mesh, c, report));
  CHECK(report.qualityMean > .95f);
  CHECK(report.qualityP05 > .85f);
  CHECK(report.qualityMin > .4f);
  CHECK(report.splits > 0 && report.collapses > 0 && report.flips > 0 && report.smoothMoves > 0);
  int within = 0;
  for (auto e : mesh.edges) {
    CHECK(e.face0 >= 0 && e.face1 >= 0);
    float len = distance(mesh.position(e.v0), mesh.position(e.v1));
    within += len >= .8f * c.constantLength && len <= 4.f / 3.f * c.constantLength;
  }
  CHECK(float(within) / mesh.edges.size() > .94f);
  for (int f = 0; f < mesh.faceCount(); ++f) {
    auto a = mesh.facePoint(f, 0), b = mesh.facePoint(f, 1), d = mesh.facePoint(f, 2);
    for (auto p : {a, b, d, (a + b) * .5f, (b + d) * .5f, (d + a) * .5f, centroid3(a, b, d)}) {
      auto hit = reference.projectSurface(0, p);
      CHECK(hit.ok);
      CHECK(distance(p, hit.position) <= c.maxGeometryError + 1.e-4f);
    }
  }
  const float uniformError=report.geometryErrorMax;
  const int uniformFaces=mesh.faceCount();
  mesh=source;
  c.featureEdgeLength=.25f*c.constantLength;
  c.featureBand=0;
  CHECK(backend.remesh(mesh,c,report));
  CHECK(mesh.faceCount()>uniformFaces);
  CHECK(report.qualityMean>.90f && report.qualityP05>.75f && report.qualityMin>.3f);
  CHECK(report.geometryErrorMax<.5f*uniformError);
  CHECK(report.constraintsHeld && report.topologyValid);
  int fineVertices=0,regularVertices=0;
  double sharpLength=0; int sharpCount=0;
  for (float h:mesh.targetLength) {
    CHECK(h>=c.featureEdgeLength-1.e-5f && h<=c.constantLength+1.e-5f);
    fineVertices+=h<.4f*c.constantLength;
    regularVertices+=h>.99f*c.constantLength;
  }
  CHECK(fineVertices>100 && regularVertices>100);
  for (auto e:mesh.edges) {
    CHECK(e.face0>=0 && e.face1>=0);
    if(e.flags&EdgeSharp) {sharpLength+=distance(mesh.position(e.v0),mesh.position(e.v1));++sharpCount;}
  }
  // Most sharp edges here join planes. They must no longer be over-refined.
  CHECK(sharpCount>0 && sharpLength/sharpCount>.65f*c.constantLength);
  int flatEdges=0, curveEdges=0; double flatSum=0,curveSum=0;
  for(auto e:mesh.edges) {
    const auto p=(mesh.position(e.v0)+mesh.position(e.v1))*.5f;
    const float len=distance(mesh.position(e.v0),mesh.position(e.v1));
    if(p.x>-50 && p.x<95 && (p.z<.001f || p.z>9.999f)) {flatSum+=len;++flatEdges;}
    if(p.x<-63.5f && p.z>7.2f) {curveSum+=len;++curveEdges;}
  }
  CHECK(flatEdges>100 && curveEdges>100);
  CHECK(flatSum/flatEdges>.85f*c.constantLength);
  CHECK(curveSum/curveEdges<.5f*c.constantLength);
  // Smooth curvature below the sharp-edge angle must trigger sizing even far
  // from the end creases. A long polygonal cylinder isolates that condition.
  mesh=makeCylinder(24,8,1.f,0.f,8.f,0,1,2);
  mesh.patches.resize(1); mesh.patches[0].type=PatchType::Unknown;
  for(auto &p:mesh.facePatchId)p=0;
  for(auto &p:mesh.vertexPatchId)p=0;
  for(auto &p:mesh.facePatchType)p=uint8_t(PatchType::Unknown);
  c.constantLength=.8f; c.featureEdgeLength=.2f; c.featureBand=1.6f;
  c.maxGeometryError=.1f; c.maxIterations=1;
  c.enableSplit=c.enableCollapse=c.enableFlip=c.enableSmooth=false;
  CHECK(backend.remesh(mesh,c,report));
  int curvedMiddle=0;
  for(int v=0;v<mesh.vertexCount();++v) if(mesh.position(v).z>3 && mesh.position(v).z<5) {
    CHECK(mesh.targetLength[v]<.6f*c.constantLength);
    ++curvedMiddle;
  }
  CHECK(curvedMiddle>0);
  // A square prism has sharp plane/plane intersections but no curved region.
  // Enabling refinement must leave its entire size field at the regular target.
  mesh=makeCylinder(4,2,2.f,0.f,4.f,0,1,2);
  mesh.patches.resize(1); mesh.patches[0].type=PatchType::Unknown;
  for(auto &p:mesh.facePatchId)p=0;
  for(auto &p:mesh.vertexPatchId)p=0;
  for(auto &p:mesh.facePatchType)p=uint8_t(PatchType::Unknown);
  CHECK(backend.remesh(mesh,c,report));
  CHECK(!mesh.featureEdges.empty());
  for(float h:mesh.targetLength) CHECK_NEAR(h,c.constantLength,1.e-6f);
  // Two planes meeting at only 15 degrees are still planes, not a fillet.
  mesh=makeGrid(4,4,-2,-2,2,2,0);
  mesh.patches[0].type=PatchType::Unknown;
  for(auto &p:mesh.facePatchType)p=uint8_t(PatchType::Unknown);
  for(int v=0;v<mesh.vertexCount();++v) {auto p=mesh.position(v);if(p.x>0)p.z=p.x*.2679492f;mesh.setPosition(v,p);}
  CHECK(backend.remesh(mesh,c,report));
  for(float h:mesh.targetLength) CHECK_NEAR(h,c.constantLength,1.e-6f);
  return test_result("test_rxmesh_raw_reference");
}
