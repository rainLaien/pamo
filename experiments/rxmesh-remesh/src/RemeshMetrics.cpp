#include "cad_adaptive/RemeshMetrics.h"
#include "cad_adaptive/GeometryProjector.h"
#include "cad_adaptive/RemeshField.h"
#include "cad_adaptive/RemeshPolicy.h"

#include <algorithm>
#include <sstream>
#include <vector>

namespace cad_adaptive {

void fillMeshMetrics(SemanticMesh &mesh, const RemeshConfig &config, RemeshReport &report) {
  mesh.rebuildTopology();
  GeometryProjector projector;
  projector.build(mesh);
  RemeshField::compute(mesh, config, projector);

  std::vector<float> qualities, sizing;
  float qmin = 1, geom = 0;
  for (int f = 0; f < mesh.faceCount(); ++f) {
    if (!mesh.faceAlive[f]) continue;
    const Vec3 a = mesh.position(int(mesh.i0[f]));
    const Vec3 b = mesh.position(int(mesh.i1[f]));
    const Vec3 c = mesh.position(int(mesh.i2[f]));
    const float q = triangleQuality(a, b, c);
    qualities.push_back(q);
    qmin = std::min(qmin, q);
    const Vec3 mid = centroid3(a, b, c);
    const auto hit = projector.projectSurface(mesh.facePatchId[f], mid);
    if (hit.ok) geom = std::max(geom, distance(mid, hit.position));
  }
  for (const auto &e : mesh.edges) {
    const float h = RemeshPolicy::edgeTarget(mesh.targetLength[e.v0], mesh.targetLength[e.v1]);
    const float len = distance(mesh.position(int(e.v0)), mesh.position(int(e.v1)));
    if (h > 0) sizing.push_back(std::abs(len / h - 1));
  }
  std::sort(qualities.begin(), qualities.end());
  std::sort(sizing.begin(), sizing.end());
  if (!qualities.empty()) {
    report.qualityMin = qmin;
    float s = 0;
    for (float q : qualities) s += q;
    report.qualityMean = s / float(qualities.size());
    report.qualityP05 = qualities[std::min(qualities.size() - 1, qualities.size() / 20)];
  }
  if (!sizing.empty()) {
    float s = 0;
    for (float x : sizing) s += x;
    report.sizingErrorMean = s / float(sizing.size());
    report.sizingErrorP95 = sizing[std::min(sizing.size() - 1, sizing.size() * 95 / 100)];
  }
  report.geometryErrorMax = geom;
}

std::string remeshReportJson(const RemeshReport &r) {
  std::ostringstream o;
  o << "{\n"
    << "  \"splits\": " << r.splits << ",\n"
    << "  \"cavity_refines\": " << r.cavityRefines << ",\n"
    << "  \"collapses\": " << r.collapses << ",\n"
    << "  \"flips\": " << r.flips << ",\n"
    << "  \"smooth_moves\": " << r.smoothMoves << ",\n"
    << "  \"split_candidates\": " << r.splitCandidates << ",\n"
    << "  \"cavity_candidates\": " << r.cavityCandidates << ",\n"
    << "  \"collapse_candidates\": " << r.collapseCandidates << ",\n"
    << "  \"flip_candidates\": " << r.flipCandidates << ",\n"
    << "  \"reject_topology\": " << r.rejectTopology << ",\n"
    << "  \"reject_patch\": " << r.rejectPatch << ",\n"
    << "  \"reject_feature\": " << r.rejectFeature << ",\n"
    << "  \"reject_normal\": " << r.rejectNormal << ",\n"
    << "  \"reject_quality\": " << r.rejectQuality << ",\n"
    << "  \"reject_error\": " << r.rejectError << ",\n"
    << "  \"quality_mean\": " << r.qualityMean << ",\n"
    << "  \"quality_p05\": " << r.qualityP05 << ",\n"
    << "  \"quality_min\": " << r.qualityMin << ",\n"
    << "  \"sizing_error_mean\": " << r.sizingErrorMean << ",\n"
    << "  \"sizing_error_p95\": " << r.sizingErrorP95 << ",\n"
    << "  \"geometry_error_max\": " << r.geometryErrorMax << ",\n"
    << "  \"moved_locked_vertices\": " << r.movedLockedVertices << ",\n"
    << "  \"missing_boundary_edges\": " << r.missingBoundaryEdges << ",\n"
    << "  \"topology_valid\": " << (r.topologyValid ? "true" : "false") << ",\n"
    << "  \"constraints_held\": " << (r.constraintsHeld ? "true" : "false") << ",\n"
    << "  \"seconds_setup\": " << r.secondsSetup << ",\n"
    << "  \"seconds_cavity\": " << r.secondsCavity << ",\n"
    << "  \"seconds_split\": " << r.secondsSplit << ",\n"
    << "  \"seconds_collapse\": " << r.secondsCollapse << ",\n"
    << "  \"seconds_flip\": " << r.secondsFlip << ",\n"
    << "  \"seconds_smooth\": " << r.secondsSmooth << ",\n"
    << "  \"seconds\": " << r.seconds << "\n"
    << "}\n";
  return o.str();
}

} // namespace cad_adaptive
