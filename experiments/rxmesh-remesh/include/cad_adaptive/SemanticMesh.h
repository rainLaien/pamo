#pragma once

#include "cad_adaptive/Types.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace cad_adaptive {

class BoundarySizingField;

struct EdgeRec {
  uint32_t v0 = 0, v1 = 0;
  int face0 = -1, face1 = -1;
  uint32_t patchLeft = kInvalidId, patchRight = kInvalidId;
  uint32_t featureCurveId = 0;
  uint8_t flags = 0;
};


struct EdgeLengthAudit {
  float targetLength = 0;
  float splitThreshold = 0;
  float collapseThreshold = 0;
  float min = 0, p05 = 0, median = 0, p95 = 0, p99 = 0, max = 0;
  float editableMax = 0, protectedMax = 0;
  uint32_t edgeCount = 0, editableCount = 0, protectedCount = 0;
  uint32_t editableAboveSplit = 0, protectedAboveSplit = 0;
  uint32_t editableBelowCollapse = 0;
};

struct AnalyticGeometryAudit {
  float mean = 0;
  float p95 = 0;
  float max = 0;
  uint32_t supportedVertexCount = 0;
  uint32_t unsupportedVertexCount = 0;
};

struct SemanticMesh {
  std::shared_ptr<const BoundarySizingField> LocalSizing;
  std::vector<float> px, py, pz;
  std::vector<float> nx, ny, nz;
  std::vector<uint32_t> vertexPatchId;
  std::vector<uint8_t> vertexConstraint;
  std::vector<float> targetLength;
  std::vector<float> curvature;
  std::vector<float> featureDistance;

  std::vector<uint32_t> i0, i1, i2;
  std::vector<uint32_t> facePatchId;
  std::vector<uint8_t> facePatchType;
  std::vector<uint8_t> faceAlive;

  std::vector<PatchRecord> patches;
  std::vector<EdgeRec> edges;
  std::vector<std::vector<int>> incidentFaces;

  int vertexCount() const { return int(px.size()); }
  int faceCount() const { return int(i0.size()); }

  Vec3 position(int v) const { return {px[v], py[v], pz[v]}; }
  void setPosition(int v, Vec3 p) {
    px[v] = p.x;
    py[v] = p.y;
    pz[v] = p.z;
  }
  Vec3 normal(int v) const { return {nx[v], ny[v], nz[v]}; }
  void setNormal(int v, Vec3 n) {
    nx[v] = n.x;
    ny[v] = n.y;
    nz[v] = n.z;
  }
  std::array<int, 3> face(int f) const { return {int(i0[f]), int(i1[f]), int(i2[f])}; }
  Vec3 facePoint(int f, int k) const {
    const int v = k == 0 ? int(i0[f]) : k == 1 ? int(i1[f]) : int(i2[f]);
    return position(v);
  }

  int addVertex(Vec3 p, uint32_t patchId, VertexConstraint constraint);
  int addFace(int a, int b, int c, uint32_t patchId, PatchType type);
  void killFace(int f);

  void resizeVertices(int n);
  void clear();
  void bbox(Vec3 &bmin, Vec3 &bmax) const;
  float bboxDiagonal() const;
  EdgeLengthAudit edgeLengthAudit(float targetLength, float splitRatio, float collapseRatio) const;
  AnalyticGeometryAudit analyticGeometryAudit() const;

  void rebuildTopology();
  void computeVertexNormals();
  bool validate(std::string *error = nullptr) const;
  void compact();

  bool load(const std::string &path, std::string *error = nullptr);
  bool loadObj(const std::string &path, std::string *error = nullptr);
  bool loadStl(const std::string &path, std::string *error = nullptr);
  bool save(const std::string &path, std::string *error = nullptr) const;
  bool saveObj(const std::string &path, std::string *error = nullptr) const;
  bool savePly(const std::string &path, std::string *error = nullptr) const;
};

SemanticMesh makeGrid(int nx, int ny, float x0, float y0, float x1, float y1, uint32_t patchId);
SemanticMesh makeTwoPatchGrid(int nx, int ny, float x0, float y0, float x1, float y1);
SemanticMesh makeCylinder(int slices, int stacks, float radius, float z0, float z1,
                          uint32_t bodyPatch, uint32_t bottomPatch, uint32_t topPatch);
void lockMeshBoundary(SemanticMesh &mesh);

} // namespace cad_adaptive
