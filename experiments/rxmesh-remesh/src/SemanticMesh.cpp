#include "cad_adaptive/SemanticMesh.h"
#include "cad_adaptive/BoundarySizingField.h"

#include <algorithm>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <tuple>
#include <unordered_map>

namespace cad_adaptive {
namespace {

uint64_t edgeKey(uint32_t a, uint32_t b) {
  if (a > b) std::swap(a, b);
  return (uint64_t(a) << 32) | uint64_t(b);
}

uint32_t featureForPatches(uint32_t a, uint32_t b) {
  if (a == kInvalidId || b == kInvalidId || a == b) return kOpenBoundaryFeature;
  if (a > b) std::swap(a, b);
  return 2u + a * 100003u + b;
}

} // namespace

int SemanticMesh::addVertex(Vec3 p, uint32_t patchId, VertexConstraint constraint) {
  px.push_back(p.x);
  py.push_back(p.y);
  pz.push_back(p.z);
  nx.push_back(0);
  ny.push_back(0);
  nz.push_back(1);
  vertexPatchId.push_back(patchId);
  vertexConstraint.push_back(uint8_t(constraint));
  targetLength.push_back(LocalSizing ? LocalSizing->evaluate(patchId,p) : 0.0f);
  curvature.push_back(0);
  featureDistance.push_back(1e20f);
  incidentFaces.emplace_back();
  return vertexCount() - 1;
}

int SemanticMesh::addFace(int a, int b, int c, uint32_t patchId, PatchType type) {
  if (a == b || b == c || a == c) return -1;
  i0.push_back(uint32_t(a));
  i1.push_back(uint32_t(b));
  i2.push_back(uint32_t(c));
  facePatchId.push_back(patchId);
  facePatchType.push_back(uint8_t(type));
  faceAlive.push_back(1);
  return faceCount() - 1;
}

void SemanticMesh::killFace(int f) {
  if (f >= 0 && f < faceCount()) faceAlive[f] = 0;
}

void SemanticMesh::resizeVertices(int n) {
  px.assign(n, 0);
  py.assign(n, 0);
  pz.assign(n, 0);
  nx.assign(n, 0);
  ny.assign(n, 0);
  nz.assign(n, 1);
  vertexPatchId.assign(n, 0);
  vertexConstraint.assign(n, uint8_t(VertexConstraint::Surface));
  targetLength.assign(n, 0);
  curvature.assign(n, 0);
  featureDistance.assign(n, 1e20f);
  incidentFaces.assign(n, {});
}

void SemanticMesh::clear() {
  *this = SemanticMesh{};
}

void SemanticMesh::bbox(Vec3 &bmin, Vec3 &bmax) const {
  bmin = {1e20f, 1e20f, 1e20f};
  bmax = {-1e20f, -1e20f, -1e20f};
  for (int v = 0; v < vertexCount(); ++v) {
    bmin.x = std::min(bmin.x, px[v]);
    bmin.y = std::min(bmin.y, py[v]);
    bmin.z = std::min(bmin.z, pz[v]);
    bmax.x = std::max(bmax.x, px[v]);
    bmax.y = std::max(bmax.y, py[v]);
    bmax.z = std::max(bmax.z, pz[v]);
  }
}

float SemanticMesh::bboxDiagonal() const {
  Vec3 a, b;
  bbox(a, b);
  return std::max(length(b - a), 1e-6f);
}


EdgeLengthAudit SemanticMesh::edgeLengthAudit(float target, float splitRatio,
                                               float collapseRatio) const {
  EdgeLengthAudit out;
  out.targetLength = target;
  out.splitThreshold = target * splitRatio;
  out.collapseThreshold = target * collapseRatio;
  std::vector<float> lengths;
  lengths.reserve(edges.size());
  for (const EdgeRec &e : edges) {
    if (e.v0 >= uint32_t(vertexCount()) || e.v1 >= uint32_t(vertexCount())) continue;
    const float l = distance(position(int(e.v0)), position(int(e.v1)));
    const float edgeTarget = LocalSizing && targetLength.size()==px.size()
        ? 0.5f*(targetLength[e.v0]+targetLength[e.v1]) : target;
    const float upper=edgeTarget*splitRatio, lower=edgeTarget*collapseRatio;
    lengths.push_back(l);
    ++out.edgeCount;
    const bool isProtected = (e.flags & EdgeProtected) != 0;
    if (isProtected) {
      ++out.protectedCount;
      out.protectedMax = std::max(out.protectedMax, l);
      if (l > upper) ++out.protectedAboveSplit;
    } else {
      ++out.editableCount;
      out.editableMax = std::max(out.editableMax, l);
      if (l > upper) ++out.editableAboveSplit;
      if (l < lower) ++out.editableBelowCollapse;
    }
  }
  if (lengths.empty()) return out;
  std::sort(lengths.begin(), lengths.end());
  const auto percentile = [&](double p) {
    if (lengths.size() == 1) return lengths.front();
    const double x = p * double(lengths.size() - 1);
    const size_t lo = size_t(x);
    const size_t hi = std::min(lo + 1, lengths.size() - 1);
    const float t = float(x - double(lo));
    return lerp(lengths[lo], lengths[hi], t);
  };
  out.min = lengths.front();
  out.p05 = percentile(0.05);
  out.median = percentile(0.50);
  out.p95 = percentile(0.95);
  out.p99 = percentile(0.99);
  out.max = lengths.back();
  return out;
}

AnalyticGeometryAudit SemanticMesh::analyticGeometryAudit() const {
  AnalyticGeometryAudit out;
  std::vector<float> errors;
  errors.reserve(vertexCount());
  double sum = 0.0;
  for (int v = 0; v < vertexCount(); ++v) {
    const uint32_t patchId = v < int(vertexPatchId.size()) ? vertexPatchId[v] : kInvalidId;
    if (patchId >= patches.size()) {
      ++out.unsupportedVertexCount;
      continue;
    }
    const PatchRecord &patch = patches[patchId];
    const Vec3 p = position(v);
    float err = 0.0f;
    bool supported = false;
    if (patch.type == PatchType::Plane) {
      const Vec3 n = normalize(patch.axis, {0, 0, 1});
      err = std::fabs(dot(p - patch.origin, n));
      supported = true;
    } else if (patch.type == PatchType::Cylinder && patch.radius > 0.0f) {
      const Vec3 a = normalize(patch.axis, {0, 0, 1});
      const Vec3 off = p - patch.origin;
      const Vec3 radial = off - a * dot(off, a);
      err = std::fabs(length(radial) - patch.radius);
      supported = true;
    }
    if (!supported) {
      ++out.unsupportedVertexCount;
      continue;
    }
    ++out.supportedVertexCount;
    errors.push_back(err);
    sum += err;
    out.max = std::max(out.max, err);
  }
  if (errors.empty()) return out;
  std::sort(errors.begin(), errors.end());
  const double x = 0.95 * double(errors.size() - 1);
  const size_t lo = size_t(x);
  const size_t hi = std::min(lo + 1, errors.size() - 1);
  out.p95 = lerp(errors[lo], errors[hi], float(x - double(lo)));
  out.mean = float(sum / double(errors.size()));
  return out;
}

void SemanticMesh::rebuildTopology() {
  incidentFaces.assign(vertexCount(), {});
  edges.clear();
  std::unordered_map<uint64_t, int> index;
  index.reserve(size_t(faceCount()) * 2);
  for (int f = 0; f < faceCount(); ++f) {
    if (!faceAlive[f]) continue;
    const int v[3] = {int(i0[f]), int(i1[f]), int(i2[f])};
    for (int k = 0; k < 3; ++k) {
      if (v[k] >= 0 && v[k] < vertexCount()) incidentFaces[v[k]].push_back(f);
      const uint32_t a = uint32_t(v[k]), b = uint32_t(v[(k + 1) % 3]);
      const uint64_t key = edgeKey(a, b);
      auto it = index.find(key);
      if (it == index.end()) {
        EdgeRec e;
        e.v0 = std::min(a, b);
        e.v1 = std::max(a, b);
        e.face0 = f;
        e.patchLeft = facePatchId[f];
        edges.push_back(e);
        index[key] = int(edges.size()) - 1;
      } else {
        EdgeRec &e = edges[it->second];
        if (e.face1 < 0) {
          e.face1 = f;
          e.patchRight = facePatchId[f];
        } else {
          e.flags = uint8_t(e.flags | EdgeProtected);
        }
      }
    }
  }
  for (auto &e : edges) {
    const auto fit = featureEdges.find(edgeKey(e.v0, e.v1));
    if (fit != featureEdges.end()) {
      e.flags = uint8_t(e.flags | EdgeSharp | EdgeProtected);
      e.featureCurveId = fit->second;
    }
    if (e.face1 < 0) {
      e.flags = uint8_t(e.flags | EdgeMeshBoundary | EdgeProtected);
      e.featureCurveId = kOpenBoundaryFeature;
      continue;
    }
    if (e.patchLeft != e.patchRight) {
      e.flags = uint8_t(e.flags | EdgePatchBoundary | EdgeProtected);
      e.featureCurveId = featureForPatches(e.patchLeft, e.patchRight);
    }
    // Do not infer sharp topology from endpoint constraints. A feature
    // vertex can have ordinary interior spokes after refinement. Geometric
    // sharp edges are represented explicitly and persistently by featureEdges.
  }
}

void SemanticMesh::computeVertexNormals() {
  nx.assign(vertexCount(), 0);
  ny.assign(vertexCount(), 0);
  nz.assign(vertexCount(), 0);
  for (int f = 0; f < faceCount(); ++f) {
    if (!faceAlive[f]) continue;
    const Vec3 n = triangleNormal(position(int(i0[f])), position(int(i1[f])), position(int(i2[f])));
    const int v[3] = {int(i0[f]), int(i1[f]), int(i2[f])};
    for (int k = 0; k < 3; ++k) {
      nx[v[k]] += n.x;
      ny[v[k]] += n.y;
      nz[v[k]] += n.z;
    }
  }
  for (int v = 0; v < vertexCount(); ++v) setNormal(v, normalize(normal(v)));
}

bool SemanticMesh::validate(std::string *error) const {
  const auto fail = [&](const char *msg) {
    if (error) *error = msg;
    return false;
  };
  if (px.size() != py.size() || px.size() != pz.size()) return fail("vertex soa mismatch");
  if (i0.size() != i1.size() || i0.size() != i2.size()) return fail("face soa mismatch");
  if (faceAlive.size() != i0.size()) return fail("faceAlive size");
  int live = 0;
  for (int f = 0; f < faceCount(); ++f) {
    if (!faceAlive[f]) continue;
    ++live;
    const int a = int(i0[f]), b = int(i1[f]), c = int(i2[f]);
    if (a < 0 || b < 0 || c < 0 || a >= vertexCount() || b >= vertexCount() || c >= vertexCount())
      return fail("face index out of range");
    if (a == b || b == c || a == c) return fail("degenerate face");
    if (!(triangleQuality(position(a), position(b), position(c)) > 0))
      return fail("zero-area face");
  }
  if (live == 0) return fail("no live faces");
  std::unordered_map<uint64_t, int> count;
  for (int f = 0; f < faceCount(); ++f) {
    if (!faceAlive[f]) continue;
    const int v[3] = {int(i0[f]), int(i1[f]), int(i2[f])};
    for (int k = 0; k < 3; ++k) count[edgeKey(uint32_t(v[k]), uint32_t(v[(k + 1) % 3]))]++;
  }
  for (const auto &kv : count) {
    if (kv.second > 2) return fail("non-manifold edge");
  }
  return true;
}

void SemanticMesh::compact() {
  std::vector<int> map(vertexCount(), -1);
  int nv = 0;
  SemanticMesh out;
  out.patches = patches;
  out.LocalSizing = LocalSizing;
  for (int v = 0; v < vertexCount(); ++v) {
    bool used = false;
    for (int f : incidentFaces.empty() ? std::vector<int>{} : incidentFaces[v]) {
      if (f >= 0 && f < faceCount() && faceAlive[f]) {
        used = true;
        break;
      }
    }
    if (!used) {
      for (int f = 0; f < faceCount() && !used; ++f) {
        if (!faceAlive[f]) continue;
        if (int(i0[f]) == v || int(i1[f]) == v || int(i2[f]) == v) used = true;
      }
    }
    if (!used) continue;
    map[v] = nv++;
    out.addVertex(position(v), vertexPatchId[v], VertexConstraint(vertexConstraint[v]));
    out.setNormal(map[v], normal(v));
    out.targetLength[map[v]] = v < int(targetLength.size()) ? targetLength[v] : 0;
    out.curvature[map[v]] = v < int(curvature.size()) ? curvature[v] : 0;
    out.featureDistance[map[v]] = v < int(featureDistance.size()) ? featureDistance[v] : 1e20f;
  }
  for (int f = 0; f < faceCount(); ++f) {
    if (!faceAlive[f]) continue;
    const int a = map[int(i0[f])], b = map[int(i1[f])], c = map[int(i2[f])];
    if (a < 0 || b < 0 || c < 0) continue;
    out.addFace(a, b, c, facePatchId[f], PatchType(facePatchType[f]));
  }
  for (const auto &kv : featureEdges) {
    const uint32_t oldA = uint32_t(kv.first >> 32);
    const uint32_t oldB = uint32_t(kv.first & 0xffffffffu);
    if (oldA >= map.size() || oldB >= map.size()) continue;
    const int a = map[oldA], b = map[oldB];
    if (a < 0 || b < 0 || a == b) continue;
    const uint32_t lo = uint32_t(std::min(a, b)), hi = uint32_t(std::max(a, b));
    out.featureEdges[(uint64_t(lo) << 32) | uint64_t(hi)] = kv.second;
  }
  out.rebuildTopology();
  *this = std::move(out);
}

bool SemanticMesh::load(const std::string &path, std::string *error) {
  auto ends = [&](const char *ext) {
    if (path.size() < std::strlen(ext)) return false;
    const char *t = path.c_str() + path.size() - std::strlen(ext);
    for (size_t i = 0; ext[i]; ++i) {
      const char a = t[i], b = ext[i];
      if ((a | 32) != (b | 32)) return false;
    }
    return true;
  };
  if (ends(".stl")) return loadStl(path, error);
  return loadObj(path, error);
}

bool SemanticMesh::loadStl(const std::string &path, std::string *error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamoff sz = in.tellg();
  in.seekg(0, std::ios::beg);
  char header[80] = {};
  in.read(header, 80);
  uint32_t ntri = 0;
  in.read(reinterpret_cast<char *>(&ntri), 4);
  const bool binary = in && sz == std::streamoff(84 + uint64_t(ntri) * 50);
  clear();
  patches.resize(1);
  patches[0].type = PatchType::Unknown;
  std::map<std::tuple<int, int, int>, int> weld;
  auto quant = [](float v) { return int(std::lround(double(v) * 1.0e5)); };
  auto addWelded = [&](float x, float y, float z) {
    const auto key = std::make_tuple(quant(x), quant(y), quant(z));
    auto it = weld.find(key);
    if (it != weld.end()) return it->second;
    const int id = addVertex({x, y, z}, 0, VertexConstraint::Surface);
    weld.emplace(key, id);
    return id;
  };
  auto addTri = [&](float ax, float ay, float az, float bx, float by, float bz, float cx, float cy,
                    float cz) {
    const int a = addWelded(ax, ay, az), b = addWelded(bx, by, bz), c = addWelded(cx, cy, cz);
    if (a == b || b == c || a == c) return;
    addFace(a, b, c, 0, PatchType::Unknown);
  };
  if (binary) {
    for (uint32_t t = 0; t < ntri; ++t) {
      float buf[12];
      uint16_t attr = 0;
      in.read(reinterpret_cast<char *>(buf), 48);
      in.read(reinterpret_cast<char *>(&attr), 2);
      if (!in) {
        if (error) *error = "truncated STL " + path;
        return false;
      }
      addTri(buf[3], buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11]);
    }
  } else {
    in.clear();
    in.seekg(0);
    std::string line;
    float v[9];
    int nv = 0;
    while (std::getline(in, line)) {
      std::istringstream ss(line);
      std::string tag;
      ss >> tag;
      if (tag != "vertex") continue;
      if (!(ss >> v[nv] >> v[nv + 1] >> v[nv + 2])) continue;
      nv += 3;
      if (nv == 9) {
        addTri(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
        nv = 0;
      }
    }
  }
  if (faceCount() == 0) {
    if (error) *error = "empty STL " + path;
    return false;
  }
  rebuildTopology();
  return validate(error);
}

bool SemanticMesh::loadObj(const std::string &path, std::string *error) {
  std::ifstream in(path);
  if (!in) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.size() < 2 || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag == "v") {
      float x, y, z;
      ss >> x >> y >> z;
      addVertex({x, y, z}, 0, VertexConstraint::Surface);
    } else if (tag == "f") {
      int a, b, c;
      ss >> a >> b >> c;
      if (a == 0 || b == 0 || c == 0) continue;
      addFace(a - 1, b - 1, c - 1, 0, PatchType::Unknown);
    }
  }
  rebuildTopology();
  return validate(error);
}

bool SemanticMesh::save(const std::string &path, std::string *error) const {
  auto ends = [&](const char *ext) {
    if (path.size() < std::strlen(ext)) return false;
    const char *t = path.c_str() + path.size() - std::strlen(ext);
    for (size_t i = 0; ext[i]; ++i) {
      const char a = t[i], b = ext[i];
      if ((a | 32) != (b | 32)) return false;
    }
    return true;
  };
  if (ends(".ply")) return savePly(path, error);
  if (ends(".obj")) return saveObj(path, error);
  if (error) *error = "unsupported output format: " + path + " (use .ply or .obj)";
  return false;
}

bool SemanticMesh::saveObj(const std::string &path, std::string *error) const {
  std::ofstream out(path);
  if (!out) {
    if (error) *error = "cannot write " + path;
    return false;
  }
  out << std::setprecision(9);
  for (int v = 0; v < vertexCount(); ++v)
    out << "v " << px[v] << ' ' << py[v] << ' ' << pz[v] << '\n';
  uint32_t previousPatch = kInvalidId;
  for (int f = 0; f < faceCount(); ++f) {
    if (!faceAlive[f]) continue;
    if (facePatchId[f] != previousPatch) {
      previousPatch = facePatchId[f];
      out << "g patch_" << previousPatch << '\n';
    }
    out << "f " << i0[f] + 1 << ' ' << i1[f] + 1 << ' ' << i2[f] + 1 << '\n';
  }
  return true;
}

bool SemanticMesh::savePly(const std::string &path, std::string *error) const {
  std::ofstream out(path);
  if (!out) {
    if (error) *error = "cannot write " + path;
    return false;
  }
  int liveFaces = 0;
  for (int f = 0; f < faceCount(); ++f)
    if (faceAlive[f]) ++liveFaces;

  out << "ply\n"
      << "format ascii 1.0\n"
      << "comment coordinate_system raw_xyz_no_transform\n"
      << "element vertex " << vertexCount() << "\n"
      << "property float x\n"
      << "property float y\n"
      << "property float z\n"
      << "element face " << liveFaces << "\n"
      << "property list uchar int vertex_indices\n"
      << "property uint patch_id\n"
      << "end_header\n";
  out << std::setprecision(9);
  for (int v = 0; v < vertexCount(); ++v)
    out << px[v] << ' ' << py[v] << ' ' << pz[v] << '\n';
  for (int f = 0; f < faceCount(); ++f) {
    if (!faceAlive[f]) continue;
    out << "3 " << i0[f] << ' ' << i1[f] << ' ' << i2[f] << ' ' << facePatchId[f] << '\n';
  }
  return true;
}

SemanticMesh makeGrid(int nx, int ny, float x0, float y0, float x1, float y1, uint32_t patchId) {
  SemanticMesh mesh;
  PatchRecord plane;
  plane.type = PatchType::Plane;
  plane.origin = {x0, y0, 0};
  plane.axis = {0, 0, 1};
  if (int(mesh.patches.size()) <= int(patchId)) mesh.patches.resize(patchId + 1);
  mesh.patches[patchId] = plane;
  for (int y = 0; y <= ny; ++y) {
    for (int x = 0; x <= nx; ++x) {
      const float u = float(x) / float(nx);
      const float v = float(y) / float(ny);
      const int id = mesh.addVertex({lerp(x0, x1, u), lerp(y0, y1, v), 0}, patchId,
                                    VertexConstraint::Surface);
      mesh.setNormal(id, {0, 0, 1});
    }
  }
  const int row = nx + 1;
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const int a = y * row + x;
      const int b = a + 1;
      const int c = a + row;
      const int d = c + 1;
      mesh.addFace(a, b, d, patchId, PatchType::Plane);
      mesh.addFace(a, d, c, patchId, PatchType::Plane);
    }
  }
  mesh.rebuildTopology();
  return mesh;
}

SemanticMesh makeTwoPatchGrid(int nx, int ny, float x0, float y0, float x1, float y1) {
  if (nx < 2) nx = 2;
  const int seamX = nx / 2;
  SemanticMesh mesh;
  mesh.patches.resize(2);
  PatchRecord plane;
  plane.type = PatchType::Plane;
  plane.origin = {x0, y0, 0};
  plane.axis = {0, 0, 1};
  mesh.patches[0] = plane;
  mesh.patches[1] = plane;
  mesh.patches[1].origin = {lerp(x0, x1, float(seamX) / float(nx)), y0, 0};
  const int row = nx + 1;
  for (int y = 0; y <= ny; ++y) {
    for (int x = 0; x <= nx; ++x) {
      const float u = float(x) / float(nx);
      const float v = float(y) / float(ny);
      uint32_t patch = x < seamX ? 0u : 1u;
      if (x == seamX) patch = 0;
      VertexConstraint c = VertexConstraint::Surface;
      if (x == 0 || x == nx || y == 0 || y == ny) c = VertexConstraint::Locked;
      else if (x == seamX) c = VertexConstraint::PatchBoundary;
      const int id = mesh.addVertex({lerp(x0, x1, u), lerp(y0, y1, v), 0}, patch, c);
      mesh.setNormal(id, {0, 0, 1});
    }
  }
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const int a = y * row + x;
      const int b = a + 1;
      const int c = a + row;
      const int d = c + 1;
      const uint32_t patch = x < seamX ? 0u : 1u;
      mesh.addFace(a, b, d, patch, PatchType::Plane);
      mesh.addFace(a, d, c, patch, PatchType::Plane);
    }
  }
  mesh.rebuildTopology();
  return mesh;
}

SemanticMesh makeCylinder(int slices, int stacks, float radius, float z0, float z1,
                          uint32_t bodyPatch, uint32_t bottomPatch, uint32_t topPatch) {
  SemanticMesh mesh;
  mesh.patches.resize(std::max(bodyPatch, std::max(bottomPatch, topPatch)) + 1);
  PatchRecord body;
  body.type = PatchType::Cylinder;
  body.origin = {0, 0, z0};
  body.axis = {0, 0, 1};
  body.radius = radius;
  PatchRecord bottom;
  bottom.type = PatchType::Plane;
  bottom.origin = {0, 0, z0};
  bottom.axis = {0, 0, -1};
  PatchRecord top;
  top.type = PatchType::Plane;
  top.origin = {0, 0, z1};
  top.axis = {0, 0, 1};
  mesh.patches[bodyPatch] = body;
  mesh.patches[bottomPatch] = bottom;
  mesh.patches[topPatch] = top;

  const int rings = stacks + 1;
  std::vector<int> vid(rings * slices);
  for (int s = 0; s < rings; ++s) {
    const float z = lerp(z0, z1, float(s) / float(stacks));
    for (int i = 0; i < slices; ++i) {
      const float ang = 2.0f * 3.14159265f * float(i) / float(slices);
      const Vec3 p{radius * std::cos(ang), radius * std::sin(ang), z};
      VertexConstraint c = VertexConstraint::Surface;
      uint32_t patch = bodyPatch;
      if (s == 0 || s == stacks) c = VertexConstraint::Locked;
      const int id = mesh.addVertex(p, patch, c);
      mesh.setNormal(id, normalize({p.x, p.y, 0}));
      vid[s * slices + i] = id;
    }
  }
  for (int s = 0; s < stacks; ++s) {
    for (int i = 0; i < slices; ++i) {
      const int j = (i + 1) % slices;
      const int a = vid[s * slices + i];
      const int b = vid[s * slices + j];
      const int c = vid[(s + 1) * slices + i];
      const int d = vid[(s + 1) * slices + j];
      mesh.addFace(a, b, d, bodyPatch, PatchType::Cylinder);
      mesh.addFace(a, d, c, bodyPatch, PatchType::Cylinder);
    }
  }
  const int bottomCenter =
      mesh.addVertex({0, 0, z0}, bottomPatch, VertexConstraint::Surface);
  mesh.setNormal(bottomCenter, {0, 0, -1});
  const int topCenter = mesh.addVertex({0, 0, z1}, topPatch, VertexConstraint::Surface);
  mesh.setNormal(topCenter, {0, 0, 1});
  for (int i = 0; i < slices; ++i) {
    const int j = (i + 1) % slices;
    mesh.addFace(bottomCenter, vid[j], vid[i], bottomPatch, PatchType::Plane);
    mesh.addFace(topCenter, vid[stacks * slices + i], vid[stacks * slices + j], topPatch,
                 PatchType::Plane);
  }
  mesh.rebuildTopology();
  return mesh;
}

void lockMeshBoundary(SemanticMesh &mesh) {
  mesh.rebuildTopology();
  for (const auto &e : mesh.edges) {
    if (e.flags & EdgeMeshBoundary) {
      mesh.vertexConstraint[e.v0] = uint8_t(VertexConstraint::Locked);
      mesh.vertexConstraint[e.v1] = uint8_t(VertexConstraint::Locked);
    }
  }
}

} // namespace cad_adaptive
