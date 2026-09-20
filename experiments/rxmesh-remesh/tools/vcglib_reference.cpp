// Local quality reference only. Calls the existing VCGLib installation;
// no VCGLib implementation is copied into the production CUDA backend.
#include "cad_adaptive/GeometryProjector.h"
#include "cad_adaptive/PartitionInput.h"
#include "cad_adaptive/RemeshMetrics.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vcg/complex/algorithms/isotropic_remeshing.h>
#include <vcg/complex/complex.h>
class VcgVertex;
class VcgFace;
struct VcgUsed : vcg::UsedTypes<vcg::Use<VcgVertex>::AsVertexType, vcg::Use<VcgFace>::AsFaceType> {
};
class VcgVertex : public vcg::Vertex<VcgUsed, vcg::vertex::Coord3f, vcg::vertex::Normal3f,
                                     vcg::vertex::VFAdj, vcg::vertex::Qualityf, vcg::vertex::Mark,
                                     vcg::vertex::Color4b, vcg::vertex::BitFlags> {};
class VcgFace : public vcg::Face<VcgUsed, vcg::face::VertexRef, vcg::face::Normal3f,
                                 vcg::face::FFAdj, vcg::face::VFAdj, vcg::face::Qualityf,
                                 vcg::face::Mark, vcg::face::BitFlags> {};
class VcgMesh : public vcg::tri::TriMesh<std::vector<VcgVertex>, std::vector<VcgFace>> {};
int main(int argc, char **argv) {
  using namespace cad_adaptive;
  if (argc < 3) {
    std::cerr << "usage: vcglib_reference INPUT.stl OUTPUT.obj [length] [iterations] "
                 "[feature_angle] [max_error]\n";
    return 2;
  }
  SemanticMesh input;
  std::string error;
  if (!input.load(argv[1], &error)) {
    std::cerr << error;
    return 1;
  }
  RemeshConfig cfg;
  cfg.adaptive = false;
  cfg.constantLength = argc > 3 ? std::stof(argv[3]) : input.bboxDiagonal() * .01f;
  cfg.maxIterations = argc > 4 ? std::stoi(argv[4]) : 20;
  cfg.featureAngleDegrees = argc > 5 ? std::stof(argv[5]) : 30.f;
  cfg.maxGeometryError = argc > 6 ? std::stof(argv[6]) : cfg.constantLength * .2f;
  GeometryProjector reference;
  reference.build(input);
  VcgMesh m;
  vcg::tri::Allocator<VcgMesh>::AddVertices(m, input.vertexCount());
  vcg::tri::Allocator<VcgMesh>::AddFaces(m, input.faceCount());
  for (int v = 0; v < input.vertexCount(); ++v) {
    auto p = input.position(v);
    m.vert[v].P() = {p.x, p.y, p.z};
    m.vert[v].Q() = 0;
  }
  for (int f = 0; f < input.faceCount(); ++f)
    for (int k = 0; k < 3; ++k)
      m.face[f].V(k) = &m.vert[input.face(f)[k]];
  vcg::tri::IsotropicRemeshing<VcgMesh>::Params params;
  params.SetTargetLen(cfg.constantLength);
  params.iter = cfg.maxIterations;
  params.SetFeatureAngleDeg(cfg.featureAngleDegrees);
  params.maxSurfDist = cfg.maxGeometryError;
  params.surfDistCheck = true;
  params.projectFlag = true;
  const auto start = std::chrono::steady_clock::now();
  vcg::tri::IsotropicRemeshing<VcgMesh>::Do(m, params);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  vcg::tri::Allocator<VcgMesh>::CompactEveryVector(m);
  SemanticMesh output;
  output.patches = input.patches;
  for (auto &v : m.vert)
    output.addVertex({v.P().X(), v.P().Y(), v.P().Z()}, 0, VertexConstraint::Surface);
  for (auto &f : m.face)
    output.addFace(int(f.V(0) - m.vert.data()), int(f.V(1) - m.vert.data()),
                   int(f.V(2) - m.vert.data()), 0, PatchType::Unknown);
  output.rebuildTopology();
  if (!output.save(argv[2], &error)) {
    std::cerr << error;
    return 1;
  }
  RemeshReport report;
  fillMeshMetrics(output, cfg, report);
  report.seconds = seconds;
  report.topologyValid = output.validate(&error);
  for (int f = 0; f < output.faceCount(); ++f) {
    const auto a = output.facePoint(f, 0), b = output.facePoint(f, 1), c = output.facePoint(f, 2);
    for (auto p :
         {a, b, c, (a + b) * .5f, (b + c) * .5f, (c + a) * .5f, (a + b + c) * (1.f / 3.f)}) {
      auto hit = reference.projectSurface(0, p);
      if (hit.ok)
        report.geometryErrorMax = std::max(report.geometryErrorMax, distance(p, hit.position));
    }
  }
  std::string jsonPath = argv[2];
  auto dot = jsonPath.find_last_of('.');
  if (dot != std::string::npos)
    jsonPath.resize(dot);
  jsonPath += ".json";
  std::ofstream(jsonPath) << remeshReportJson(report);
  std::cout << "vcglib_reference vertices=" << output.vertexCount()
            << " faces=" << output.faceCount() << " h=" << cfg.constantLength
            << " iterations=" << cfg.maxIterations << " feature_angle=" << cfg.featureAngleDegrees
            << " error_budget=" << cfg.maxGeometryError << "\n"
            << remeshReportJson(report);
  return report.topologyValid ? 0 : 1;
}
