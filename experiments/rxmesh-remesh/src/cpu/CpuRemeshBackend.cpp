#include "dirty_ring.h"
#include "cad_adaptive/GeometryProjector.h"
#include "cad_adaptive/IRemeshBackend.h"
#include "cad_adaptive/RemeshMetrics.h"
#include "cad_adaptive/RemeshField.h"
#include "cad_adaptive/RemeshPolicy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_set>
#include <vector>

namespace cad_adaptive {
namespace {

int oppositeVertex(const SemanticMesh &m, int f, int a, int b) {
  const int v[3] = {int(m.i0[f]), int(m.i1[f]), int(m.i2[f])};
  for (int k = 0; k < 3; ++k)
    if (v[k] != a && v[k] != b) return v[k];
  return -1;
}

bool directedEdge(const SemanticMesh &m, int f, int a, int b) {
  const int v[3] = {int(m.i0[f]), int(m.i1[f]), int(m.i2[f])};
  for (int k = 0; k < 3; ++k)
    if (v[k] == a && v[(k + 1) % 3] == b) return true;
  return false;
}


int valence(const SemanticMesh &m, int v) {
  std::unordered_set<int> nbr;
  for (int f : m.incidentFaces[v]) {
    if (f < 0 || f >= m.faceCount() || !m.faceAlive[f]) continue;
    const auto t = m.face(f);
    for (int k = 0; k < 3; ++k)
      if (t[k] != v) nbr.insert(t[k]);
  }
  return int(nbr.size());
}

std::unordered_set<int> neighbors(const SemanticMesh &m, int v) {
  std::unordered_set<int> nbr;
  for (int f : m.incidentFaces[v]) {
    if (f < 0 || f >= m.faceCount() || !m.faceAlive[f]) continue;
    const auto t = m.face(f);
    for (int k = 0; k < 3; ++k)
      if (t[k] != v) nbr.insert(t[k]);
  }
  return nbr;
}

enum class Reject { None, Topology, Patch, Feature, Normal, Quality, Error };

void countReject(RemeshReport &r, Reject k) {
  if (k == Reject::Topology) ++r.rejectTopology;
  else if (k == Reject::Patch) ++r.rejectPatch;
  else if (k == Reject::Feature) ++r.rejectFeature;
  else if (k == Reject::Normal) ++r.rejectNormal;
  else if (k == Reject::Quality) ++r.rejectQuality;
  else if (k == Reject::Error) ++r.rejectError;
}

Reject checkTriangle(const SemanticMesh &mesh, Vec3 a, Vec3 b, Vec3 c, uint32_t patchId,
                     Vec3 oldNormal, const RemeshConfig &cfg, const GeometryProjector &proj) {
  const float q = triangleQuality(a, b, c);
  if (q < cfg.minQuality) return Reject::Quality;
  const Vec3 n = triangleNormal(a, b, c);
  if (length2(oldNormal) > 0 && dot(n, oldNormal) < 0) return Reject::Normal;
  const float cosine = std::cos(cfg.normalDegrees * 3.14159265f / 180.0f);
  const Vec3 an = proj.analyticNormal(patchId, centroid3(a, b, c));
  if (length2(an) > 0 && dot(n, an) < cosine) return Reject::Normal;
  const float eps = cfg.maxGeometryError > 0 ? cfg.maxGeometryError : 1e-4f;
  const Vec3 pts[3] = {a, b, c};
  for (Vec3 p : pts) {
    const auto hit = proj.projectSurface(patchId, p);
    if (hit.ok && length2(p - hit.position) > eps * eps) return Reject::Error;
  }
  return Reject::None;
}

bool replaceVertexInFace(SemanticMesh &mesh, int f, int from, int to) {
  if (int(mesh.i0[f]) == from) mesh.i0[f] = uint32_t(to);
  if (int(mesh.i1[f]) == from) mesh.i1[f] = uint32_t(to);
  if (int(mesh.i2[f]) == from) mesh.i2[f] = uint32_t(to);
  const int a = int(mesh.i0[f]), b = int(mesh.i1[f]), c = int(mesh.i2[f]);
  return a != b && b != c && a != c;
}

void splitEdge(SemanticMesh &mesh, const EdgeRec &e, int mid) {
  const int faces[2] = {e.face0, e.face1};
  for (int fi : faces) {
    if (fi < 0) continue;
    int u = int(e.v0), w = int(e.v1);
    if (directedEdge(mesh, fi, int(e.v1), int(e.v0))) {
      u = int(e.v1);
      w = int(e.v0);
    } else if (!directedEdge(mesh, fi, int(e.v0), int(e.v1))) {
      continue;
    }
    const int opp = oppositeVertex(mesh, fi, u, w);
    const uint32_t patch = mesh.facePatchId[fi];
    const auto type = PatchType(mesh.facePatchType[fi]);
    mesh.killFace(fi);
    mesh.addFace(u, mid, opp, patch, type);
    mesh.addFace(mid, w, opp, patch, type);
  }
}

} // namespace

bool CpuRemeshBackend::remesh(SemanticMesh &mesh, const RemeshConfig &config, RemeshReport &report) {
  report = {};
  const auto started = std::chrono::steady_clock::now();
  std::string err;
  if (!mesh.validate(&err)) {
    report.topologyValid = false;
    return false;
  }
  mesh.rebuildTopology();
  const SemanticMesh referenceMesh = mesh;
  int boundIn = 0;
  for (const auto &e : mesh.edges)
    if (e.flags & EdgeMeshBoundary) ++boundIn;
  GeometryProjector projector;
  projector.build(referenceMesh);
  RemeshField::compute(mesh, config, projector);

  std::vector<Vec3> lockedPos(mesh.vertexCount());
  std::vector<uint8_t> wasLocked(mesh.vertexCount(), 0);
  for (int v = 0; v < mesh.vertexCount(); ++v) {
    if (VertexConstraint(mesh.vertexConstraint[v]) == VertexConstraint::Locked) {
      wasLocked[v] = 1;
      lockedPos[v] = mesh.position(v);
    }
  }

  auto refresh = [&] {
    mesh.rebuildTopology();
    projector.build(referenceMesh);
    RemeshField::compute(mesh, config, projector);
    mesh.computeVertexNormals();
    for (int v = 0; v < mesh.vertexCount(); ++v) {
      const auto n = projector.analyticNormal(mesh.vertexPatchId[v], mesh.position(v));
      if (length2(n) > 0) mesh.setNormal(v, n);
    }
  };
  refresh();
  std::vector<uint8_t> dirty(mesh.vertexCount(), 1);
  auto edgeDirty = [&](const EdgeRec &e) -> bool {
    if (dirty.empty()) return true;
    const bool d0 = e.v0 < dirty.size() && dirty[e.v0];
    const bool d1 = e.v1 < dirty.size() && dirty[e.v1];
    return d0 || d1;
  };
  auto stampDirty = [&](const std::vector<uint8_t> &usedVerts) {
    updateDirty2Ring(mesh, usedVerts, dirty);
  };

  for (int it = 0; it < config.maxIterations; ++it) {
    refresh();
    if (int(dirty.size()) < mesh.vertexCount()) dirty.resize(mesh.vertexCount(), 1);
    const int nv = mesh.vertexCount();
    std::vector<uint8_t> used(nv, 0);

    if (config.enableSplit) {
      struct Cand {
        int e;
        float score;
      };
      std::vector<Cand> cands;
      for (int ei = 0; ei < int(mesh.edges.size()); ++ei) {
        const auto &e = mesh.edges[ei];
        if (!edgeDirty(e)) continue;
        if (!RemeshPolicy::canSplit(e.flags)) continue;
        const float h0 = mesh.targetLength[e.v0], h1 = mesh.targetLength[e.v1];
        const float len = distance(mesh.position(int(e.v0)), mesh.position(int(e.v1)));
        if (!RemeshPolicy::lengthSplitCandidate(len, h0, h1, config.splitRatio)) continue;
        ++report.splitCandidates;
        cands.push_back({ei, len / std::max(RemeshPolicy::edgeTarget(h0, h1), 1e-12f)});
      }
      std::sort(cands.begin(), cands.end(),
                [](const Cand &a, const Cand &b) { return a.score > b.score; });
      for (const auto &c : cands) {
        const auto e = mesh.edges[c.e];
        if (e.v0 >= uint32_t(used.size()) || e.v1 >= uint32_t(used.size())) continue;
        if (used[e.v0] || used[e.v1]) continue;
        if (e.face0 < 0) continue;
        Vec3 mid = (mesh.position(int(e.v0)) + mesh.position(int(e.v1))) * 0.5f;
        const auto constraint = RemeshPolicy::inheritSplitConstraint(
            e.flags, VertexConstraint(mesh.vertexConstraint[e.v0]),
            VertexConstraint(mesh.vertexConstraint[e.v1]));
        const uint32_t patch = RemeshPolicy::inheritSplitPatch(e.patchLeft, e.patchRight);
        ProjectionResult hit;
        if (isBoundaryConstraint(constraint) || (e.flags & (EdgePatchBoundary | EdgeMeshBoundary)))
          hit = projector.projectFeature(e.featureCurveId, mid);
        else
          hit = projector.projectSurface(patch, mid);
        if (!hit.ok) {
          ++report.rejectError;
          continue;
        }
        mid = hit.position;
        const Vec3 oldN = triangleNormal(mesh.facePoint(e.face0, 0), mesh.facePoint(e.face0, 1),
                                         mesh.facePoint(e.face0, 2));
        const int opp = oppositeVertex(mesh, e.face0, int(e.v0), int(e.v1));
        if (opp < 0) {
          ++report.rejectTopology;
          continue;
        }
        const Reject chk =
            checkTriangle(mesh, mesh.position(int(e.v0)), mid, mesh.position(opp),
                          mesh.facePatchId[e.face0], oldN, config, projector);
        if (chk != Reject::None) {
          countReject(report, chk);
          continue;
        }
        used[e.v0] = used[e.v1] = 1;
        const int m = mesh.addVertex(mid, patch, constraint);
        mesh.targetLength[m] =
            RemeshPolicy::edgeTarget(mesh.targetLength[e.v0], mesh.targetLength[e.v1]);
        if (int(used.size()) <= m) used.resize(m + 1, 0);
        used[m] = 1;
        splitEdge(mesh, e, m);
        ++report.splits;
      }
      refresh();
      stampDirty(used);
      used.assign(mesh.vertexCount(), 0);
    }

    if (config.enableCollapse) {
      struct Cand {
        int e;
        float score;
      };
      std::vector<Cand> cands;
      for (int ei = 0; ei < int(mesh.edges.size()); ++ei) {
        const auto &e = mesh.edges[ei];
        if (!edgeDirty(e)) continue;
        const float h0 = mesh.targetLength[e.v0], h1 = mesh.targetLength[e.v1];
        const float len = distance(mesh.position(int(e.v0)), mesh.position(int(e.v1)));
        if (!RemeshPolicy::lengthCollapseCandidate(len, h0, h1, config.collapseRatio)) continue;
        ++report.collapseCandidates;
        CollapseQuery q;
        q.c0 = VertexConstraint(mesh.vertexConstraint[e.v0]);
        q.c1 = VertexConstraint(mesh.vertexConstraint[e.v1]);
        q.patch0 = mesh.vertexPatchId[e.v0];
        q.patch1 = mesh.vertexPatchId[e.v1];
        q.feature0 = e.featureCurveId;
        q.feature1 = e.featureCurveId;
        if (e.featureCurveId == 0) {
          q.feature0 = 0;
          q.feature1 = 0;
        }
        const auto dec = RemeshPolicy::classifyCollapse(q);
        if (!dec.allowed) {
          if (q.patch0 != q.patch1)
            ++report.rejectPatch;
          else
            ++report.rejectFeature;
          continue;
        }
        cands.push_back({ei, RemeshPolicy::edgeTarget(h0, h1) / std::max(len, 1e-12f)});
      }
      std::sort(cands.begin(), cands.end(),
                [](const Cand &a, const Cand &b) { return a.score > b.score; });
      for (const auto &c : cands) {
        const auto e = mesh.edges[c.e];
        if (used[e.v0] || used[e.v1]) continue;
        CollapseQuery q;
        q.c0 = VertexConstraint(mesh.vertexConstraint[e.v0]);
        q.c1 = VertexConstraint(mesh.vertexConstraint[e.v1]);
        q.patch0 = mesh.vertexPatchId[e.v0];
        q.patch1 = mesh.vertexPatchId[e.v1];
        q.feature0 = q.feature1 = e.featureCurveId;
        const auto dec = RemeshPolicy::classifyCollapse(q);
        if (!dec.allowed) continue;
        int keep = int(e.v0), remove = int(e.v1);
        if (dec.dest == CollapseDest::KeepV1) {
          keep = int(e.v1);
          remove = int(e.v0);
        }
        const auto nKeep = neighbors(mesh, keep);
        const auto nRemove = neighbors(mesh, remove);
        int common = 0;
        for (int x : nKeep)
          if (nRemove.count(x)) ++common;
        const int expected = (e.flags & EdgeMeshBoundary) ? 1 : 2;
        if (common != expected) {
          ++report.rejectTopology;
          continue;
        }
        Vec3 newP = mesh.position(keep);
        if (dec.dest == CollapseDest::Optimal)
          newP = (mesh.position(keep) + mesh.position(remove)) * 0.5f;
        ProjectionResult hit;
        const auto keepC = VertexConstraint(mesh.vertexConstraint[keep]);
        if (isBoundaryConstraint(keepC) || keepC == VertexConstraint::Locked ||
            keepC == VertexConstraint::Corner)
          hit = projector.projectVertex(mesh, keep, newP);
        else
          hit = projector.projectSurface(mesh.vertexPatchId[keep], newP);
        if (!hit.ok) {
          ++report.rejectError;
          continue;
        }
        newP = hit.position;
        const Vec3 oldKeep = mesh.position(keep);
        mesh.setPosition(keep, newP);
        bool ok = true;
        Reject why = Reject::None;
        for (int f : mesh.incidentFaces[remove]) {
          if (f < 0 || f >= mesh.faceCount() || !mesh.faceAlive[f]) continue;
          const auto t = mesh.face(f);
          const bool usesKeep = t[0] == keep || t[1] == keep || t[2] == keep;
          const Vec3 oldN = triangleNormal(mesh.position(t[0]), mesh.position(t[1]), mesh.position(t[2]));
          if (usesKeep) continue;
          Vec3 pa = mesh.position(t[0] == remove ? keep : t[0]);
          Vec3 pb = mesh.position(t[1] == remove ? keep : t[1]);
          Vec3 pc = mesh.position(t[2] == remove ? keep : t[2]);
          why = checkTriangle(mesh, pa, pb, pc, mesh.facePatchId[f], oldN, config, projector);
          if (why != Reject::None) {
            ok = false;
            break;
          }
        }
        if (!ok) {
          mesh.setPosition(keep, oldKeep);
          countReject(report, why);
          continue;
        }
        for (int f : mesh.incidentFaces[remove]) {
          if (f < 0 || f >= mesh.faceCount() || !mesh.faceAlive[f]) continue;
          const auto t = mesh.face(f);
          const bool usesKeep = t[0] == keep || t[1] == keep || t[2] == keep;
          if (usesKeep)
            mesh.killFace(f);
          else if (!replaceVertexInFace(mesh, f, remove, keep))
            mesh.killFace(f);
        }
        used[keep] = used[remove] = 1;
        ++report.collapses;
      }
      refresh();
      stampDirty(used);
      used.assign(mesh.vertexCount(), 0);
    }

    if (config.enableFlip) {
      for (int ei = 0; ei < int(mesh.edges.size()); ++ei) {
        const auto &e = mesh.edges[ei];
        if (!edgeDirty(e)) continue;
        if (!RemeshPolicy::canFlip(e.flags) || e.face0 < 0 || e.face1 < 0) continue;
        if (used[e.v0] || used[e.v1]) continue;
        const int a = int(e.v0), b = int(e.v1);
        if (!directedEdge(mesh, e.face0, a, b) && !directedEdge(mesh, e.face0, b, a)) continue;
        const int f0 = e.face0, f1 = e.face1;
        int u = a, w = b;
        if (directedEdge(mesh, f0, b, a)) {
          u = b;
          w = a;
        }
        const int opp0 = oppositeVertex(mesh, f0, u, w);
        const int opp1 = oppositeVertex(mesh, f1, u, w);
        if (opp0 < 0 || opp1 < 0 || opp0 == opp1) continue;
        if (used[uint32_t(opp0)] || used[uint32_t(opp1)]) continue;
        const auto n0 = neighbors(mesh, opp0);
        if (n0.count(opp1)) continue;
        ++report.flipCandidates;
        const Vec3 pa = mesh.position(u), pb = mesh.position(w);
        const Vec3 pc = mesh.position(opp0), pd = mesh.position(opp1);
        const float hDiag = RemeshPolicy::edgeTarget(mesh.targetLength[opp0], mesh.targetLength[opp1]);
        const float eBefore =
            config.valenceWeight * float(std::abs(valence(mesh, u) - 6) + std::abs(valence(mesh, w) - 6) +
                                         std::abs(valence(mesh, opp0) - 6) + std::abs(valence(mesh, opp1) - 6)) +
            config.shapeWeight * ((1 - triangleQuality(pa, pb, pc)) + (1 - triangleQuality(pa, pb, pd))) +
            config.sizingWeight * std::abs(distance(pa, pb) / std::max(RemeshPolicy::edgeTarget(
                                                                          mesh.targetLength[u], mesh.targetLength[w]),
                                                                      1e-12f) -
                                           1);
        const float eAfter =
            config.valenceWeight * float(std::abs(valence(mesh, u) - 7) + std::abs(valence(mesh, w) - 7) +
                                         std::abs(valence(mesh, opp0) - 5) + std::abs(valence(mesh, opp1) - 5)) +
            config.shapeWeight * ((1 - triangleQuality(pc, pa, pd)) + (1 - triangleQuality(pc, pd, pb))) +
            config.sizingWeight * std::abs(distance(pc, pd) / std::max(hDiag, 1e-12f) - 1);
        if (!(eAfter < eBefore - 1e-8f)) continue;
        const uint32_t patch = mesh.facePatchId[f0];
        const Vec3 oldN = triangleNormal(pa, pb, pc);
        if (checkTriangle(mesh, pc, pa, pd, patch, oldN, config, projector) != Reject::None ||
            checkTriangle(mesh, pc, pd, pb, patch, oldN, config, projector) != Reject::None) {
          ++report.rejectNormal;
          continue;
        }
        const PatchType type = PatchType(mesh.facePatchType[f0]);
        mesh.killFace(f0);
        mesh.killFace(f1);
        mesh.addFace(opp0, u, opp1, patch, type);
        mesh.addFace(opp0, opp1, w, patch, type);
        used[u] = used[w] = used[uint32_t(opp0)] = used[uint32_t(opp1)] = 1;
        ++report.flips;
      }
      refresh();
      stampDirty(used);
      used.assign(mesh.vertexCount(), 0);
    }

    if (config.enableSmooth) {
      refresh();
      for (int v = 0; v < mesh.vertexCount(); ++v) {
        if (v < int(dirty.size()) && !dirty[v]) continue;
        const auto c = VertexConstraint(mesh.vertexConstraint[v]);
        if (c == VertexConstraint::Locked || c == VertexConstraint::Corner) continue;
        Vec3 sum{};
        int count = 0;
        for (int n : neighbors(mesh, v)) {
          sum = sum + mesh.position(n);
          ++count;
        }
        if (count < 2) continue;
        const Vec3 p = mesh.position(v);
        Vec3 d = sum * (1.0f / float(count)) - p;
        if (isBoundaryConstraint(c)) {
          Vec3 t{};
          int tc = 0;
          for (const auto &e : mesh.edges) {
            if (e.v0 != uint32_t(v) && e.v1 != uint32_t(v)) continue;
            if ((e.flags & (EdgePatchBoundary | EdgeMeshBoundary | EdgeProtected)) == 0) continue;
            const int o = e.v0 == uint32_t(v) ? int(e.v1) : int(e.v0);
            t = t + (mesh.position(o) - p);
            ++tc;
          }
          if (tc == 0) continue;
          t = normalize(t);
          d = t * dot(d, t);
        } else {
          const Vec3 n = mesh.normal(v);
          d = d - n * dot(d, n);
        }
        Vec3 q = p + d * config.smoothLambda;
        const auto hit = projector.projectVertex(mesh, v, q);
        if (!hit.ok) {
          ++report.rejectError;
          continue;
        }
        const Vec3 dest = hit.position;
        if (dest.x == p.x && dest.y == p.y && dest.z == p.z) continue;
        bool ok = true;
        const Vec3 old = p;
        mesh.setPosition(v, dest);
        for (int f : mesh.incidentFaces[v]) {
          if (f < 0 || f >= mesh.faceCount() || !mesh.faceAlive[f]) continue;
          const auto t = mesh.face(f);
          const Vec3 oa = t[0] == v ? old : mesh.position(t[0]);
          const Vec3 ob = t[1] == v ? old : mesh.position(t[1]);
          const Vec3 oc = t[2] == v ? old : mesh.position(t[2]);
          if (checkTriangle(mesh, mesh.position(t[0]), mesh.position(t[1]), mesh.position(t[2]),
                            mesh.facePatchId[f], triangleNormal(oa, ob, oc), config,
                            projector) != Reject::None) {
            ok = false;
            break;
          }
        }
        if (!ok)
          mesh.setPosition(v, old);
        else {
          if (v >= int(used.size())) used.resize(v + 1, 0);
          used[v] = 1;
          ++report.smoothMoves;
        }
      }
      stampDirty(used);
    }
  }

  const int nLock = int(wasLocked.size());
  for (int v = 0; v < nLock && v < mesh.vertexCount(); ++v) {
    if (!wasLocked[v]) continue;
    if (VertexConstraint(mesh.vertexConstraint[v]) == VertexConstraint::Locked &&
        length2(mesh.position(v) - lockedPos[v]) > 1e-18f)
      ++report.movedLockedVertices;
  }

  mesh.compact();
  mesh.rebuildTopology();
  int boundOut = 0;
  for (const auto &e : mesh.edges)
    if (e.flags & EdgeMeshBoundary) ++boundOut;
  if (boundOut < boundIn) report.missingBoundaryEdges = boundIn - boundOut;
  fillMeshMetrics(mesh, config, report);
  report.topologyValid = mesh.validate();

  report.constraintsHeld = report.movedLockedVertices == 0 && report.missingBoundaryEdges == 0;
  report.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return report.topologyValid && report.constraintsHeld;
}

} // namespace cad_adaptive
