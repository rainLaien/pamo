#include "CadMesh/CudaAnalyticFitting.h"
#include "CadMesh/AnalyticPatchRemesher.h"
#include "CadMesh/NativeRemesher.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace CadMesh {
namespace {

using EdgeKey = std::uint64_t;
EdgeKey Key(int a, int b) {
  if (a > b) std::swap(a, b);
  return (std::uint64_t(std::uint32_t(a)) << 32) | std::uint32_t(b);
}
int KeyFirst(EdgeKey key) { return int(std::uint32_t(key >> 32)); }
int KeySecond(EdgeKey key) { return int(std::uint32_t(key)); }

struct EdgeRecord {
  int A = -1, B = -1;
  std::array<int, 2> Faces{-1, -1};
  int FaceCount = 0;
  bool NonManifold = false;

  void addFace(int face) {
    if (FaceCount < int(Faces.size()))
      Faces[FaceCount] = face;
    else
      NonManifold = true;
    ++FaceCount;
  }
};

std::vector<EdgeRecord>
BuildEdges(const std::vector<std::array<int, 3>> &faces,
           const std::vector<int> *labels = nullptr,
           const std::vector<unsigned char> *skippedPatches = nullptr) {
  std::unordered_map<EdgeKey, int> lookup;
  std::vector<EdgeRecord> edges;
  edges.reserve(faces.size() * 3 / 2 + 1);
  for (int face = 0; face < int(faces.size()); ++face) {
    if (labels && skippedPatches) {
      const int patch = (*labels)[face];
      if (patch >= 0 && patch < int(skippedPatches->size()) &&
          (*skippedPatches)[patch])
        continue;
    }
    for (int side = 0; side < 3; ++side) {
      int a = faces[face][side], b = faces[face][(side + 1) % 3];
      EdgeKey key = Key(a, b);
      auto found = lookup.find(key);
      int id;
      if (found == lookup.end()) {
        id = int(edges.size());
        lookup.emplace(key, id);
        edges.push_back({std::min(a, b), std::max(a, b)});
      } else {
        id = found->second;
      }
      edges[id].addFace(face);
    }
  }
  return edges;
}

double TriangleQuality(const Point3 &a, const Point3 &b, const Point3 &c) {
  const Vec3 ab = Sub(ToVec(b), ToVec(a));
  const Vec3 bc = Sub(ToVec(c), ToVec(b));
  const Vec3 ca = Sub(ToVec(a), ToVec(c));
  const double denominator = Dot(ab, ab) + Dot(bc, bc) + Dot(ca, ca);
  return denominator > 0 ? 2 * std::sqrt(3.0) * Norm(Cross(ab, ca)) / denominator : 0;
}

Vec3 TriangleNormal(const std::vector<Point3> &vertices,
                    const std::array<int, 3> &face) {
  return Normalize(Cross(Sub(ToVec(vertices[face[1]]), ToVec(vertices[face[0]])),
                         Sub(ToVec(vertices[face[2]]), ToVec(vertices[face[0]]))));
}

struct Box3 {
  Vec3 Lower{std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
  Vec3 Upper{-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()};
  void include(const Point3 &point) {
    const Vec3 p = ToVec(point);
    for (int axis = 0; axis < 3; ++axis) {
      Lower[axis] = std::min(Lower[axis], p[axis]);
      Upper[axis] = std::max(Upper[axis], p[axis]);
    }
  }
  void include(const Box3 &box) {
    for (int axis = 0; axis < 3; ++axis) {
      Lower[axis] = std::min(Lower[axis], box.Lower[axis]);
      Upper[axis] = std::max(Upper[axis], box.Upper[axis]);
    }
  }
};

double BoxDistanceSquared(const Box3 &box, const Point3 &point) {
  const Vec3 p = ToVec(point);
  double squared = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const double delta = p[axis] < box.Lower[axis]
                             ? box.Lower[axis] - p[axis]
                         : p[axis] > box.Upper[axis]
                             ? p[axis] - box.Upper[axis]
                             : 0;
    squared += delta * delta;
  }
  return squared;
}

bool BoxesOverlap(const Box3 &a, const Box3 &b, double epsilon = 0) {
  for (int axis = 0; axis < 3; ++axis)
    if (a.Upper[axis] + epsilon < b.Lower[axis] ||
        b.Upper[axis] + epsilon < a.Lower[axis])
      return false;
  return true;
}

Point3 ClosestPointOnTriangle(const Point3 &point, const Point3 &a,
                              const Point3 &b, const Point3 &c) {
  const Vec3 p = ToVec(point), av = ToVec(a), ab = Sub(ToVec(b), av),
             ac = Sub(ToVec(c), av), ap = Sub(p, av);
  const double d1 = Dot(ab, ap), d2 = Dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const Vec3 bp = Sub(p, ToVec(b));
  const double d3 = Dot(ab, bp), d4 = Dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0)
    return ToPoint(Add(av, Mul(ab, d1 / (d1 - d3))));
  const Vec3 cp = Sub(p, ToVec(c));
  const double d5 = Dot(ab, cp), d6 = Dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0)
    return ToPoint(Add(av, Mul(ac, d2 / (d2 - d6))));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const Vec3 bc = Sub(ToVec(c), ToVec(b));
    return ToPoint(Add(ToVec(b), Mul(bc, (d4 - d3) /
                                            ((d4 - d3) + (d5 - d6)))));
  }
  const double denominator = 1.0 / (va + vb + vc);
  return ToPoint(Add(av, Add(Mul(ab, vb * denominator),
                             Mul(ac, vc * denominator))));
}

struct SpatialTriangle {
  std::array<Point3, 3> Points;
  std::array<int, 3> VertexIds;
  int FaceId = -1;
  int PatchId = -1;
  Box3 Box;
};

struct BvhNode {
  Box3 Box;
  int Left = -1, Right = -1, Begin = 0, Count = 0;
};

double ProjectionMinimum(const std::array<Point3, 3> &triangle,
                         const Vec3 &axis, double &maximum) {
  double minimum = Dot(ToVec(triangle[0]), axis);
  maximum = minimum;
  for (int i = 1; i < 3; ++i) {
    const double value = Dot(ToVec(triangle[i]), axis);
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  return minimum;
}

bool SeparatedOnAxis(const std::array<Point3, 3> &a,
                     const std::array<Point3, 3> &b, const Vec3 &axis,
                     double epsilon) {
  const double magnitude = Norm(axis);
  if (magnitude <= 1e-20) return false;
  const Vec3 normalized = Mul(axis, 1.0 / magnitude);
  double maximumA, maximumB;
  const double minimumA = ProjectionMinimum(a, normalized, maximumA);
  const double minimumB = ProjectionMinimum(b, normalized, maximumB);
  return maximumA < minimumB - epsilon || maximumB < minimumA - epsilon;
}

bool TrianglesIntersect(const std::array<Point3, 3> &a,
                        const std::array<Point3, 3> &b, double epsilon) {
  Vec3 edgeA[3], edgeB[3];
  for (int i = 0; i < 3; ++i) {
    edgeA[i] = Sub(ToVec(a[(i + 1) % 3]), ToVec(a[i]));
    edgeB[i] = Sub(ToVec(b[(i + 1) % 3]), ToVec(b[i]));
  }
  const Vec3 normalA = Cross(edgeA[0], edgeA[1]);
  const Vec3 normalB = Cross(edgeB[0], edgeB[1]);
  if (SeparatedOnAxis(a, b, normalA, epsilon) ||
      SeparatedOnAxis(a, b, normalB, epsilon))
    return false;
  for (const Vec3 &first : edgeA)
    for (const Vec3 &second : edgeB)
      if (SeparatedOnAxis(a, b, Cross(first, second), epsilon))
        return false;
  if (Norm(Cross(normalA, normalB)) <=
      1e-10 * std::max(Norm(normalA) * Norm(normalB), 1e-30)) {
    for (const Vec3 &edge : edgeA)
      if (SeparatedOnAxis(a, b, Cross(normalA, edge), epsilon)) return false;
    for (const Vec3 &edge : edgeB)
      if (SeparatedOnAxis(a, b, Cross(normalA, edge), epsilon)) return false;
  }
  return true;
}

class SurfaceIndex {
public:
  SurfaceIndex(const std::vector<Point3> &vertices,
               const std::vector<std::array<int, 3>> &faces,
               const std::vector<int> &labels) {
    mTriangles.reserve(faces.size());
    for (int faceId = 0; faceId < int(faces.size()); ++faceId) {
      SpatialTriangle triangle;
      triangle.FaceId = faceId;
      triangle.PatchId = labels[faceId];
      triangle.VertexIds = faces[faceId];
      for (int i = 0; i < 3; ++i) {
        triangle.Points[i] = vertices[faces[faceId][i]];
        triangle.Box.include(triangle.Points[i]);
      }
      mTriangles.push_back(triangle);
    }
    mOrder.resize(mTriangles.size());
    for (int i = 0; i < int(mOrder.size()); ++i) mOrder[i] = i;
    if (!mOrder.empty()) build(0, int(mOrder.size()));
  }

  bool closest(const Point3 &query, int patch, Point3 &point,
               double &distance) const {
    if (mNodes.empty()) return false;
    double best = std::numeric_limits<double>::infinity();
    nearest(0, query, patch, best, point);
    if (!std::isfinite(best)) return false;
    distance = std::sqrt(best);
    return true;
  }

  bool intersects(const std::array<Point3, 3> &candidate,
                  const std::array<int, 3> &vertexIds,
                  const std::unordered_set<int> &excludedFaces,
                  double epsilon, int *hitFace = nullptr) const {
    if (mNodes.empty()) return false;
    Box3 box;
    for (const auto &point : candidate) box.include(point);
    return intersectsNode(0, box, candidate, vertexIds, excludedFaces,
                          epsilon, hitFace);
  }

  bool firstSelfIntersection(int &first, int &second,
                             double epsilon) const {
    for (const auto &triangle : mTriangles) {
      const std::unordered_set<int> excluded{triangle.FaceId};
      int hit = -1;
      if (intersects(triangle.Points, triangle.VertexIds, excluded,
                     epsilon, &hit)) {
        first = triangle.FaceId;
        second = hit;
        return true;
      }
    }
    return false;
  }

  std::size_t collectIntersectingRebuiltPatches(
      const std::vector<unsigned char> &rebuilt,
      std::vector<unsigned char> &rejected, double epsilon) const {
    rejected.assign(rebuilt.size(), 0);
    for (const auto &triangle : mTriangles) {
      const std::unordered_set<int> excluded{triangle.FaceId};
      int hit = -1;
      if (!intersects(triangle.Points, triangle.VertexIds, excluded,
                      epsilon, &hit) || hit < 0 || hit >= int(mTriangles.size()))
        continue;
      const int patches[2] = {triangle.PatchId, mTriangles[hit].PatchId};
      for (int patch : patches)
        if (patch >= 0 && patch < int(rebuilt.size()) && rebuilt[patch])
          rejected[patch] = 1;
    }
    return std::count(rejected.begin(), rejected.end(),
                      static_cast<unsigned char>(1));
  }

private:
  int build(int begin, int end) {
    const int nodeId = int(mNodes.size());
    mNodes.push_back({});
    Box3 box, centers;
    for (int i = begin; i < end; ++i) {
      const auto &triangle = mTriangles[mOrder[i]];
      box.include(triangle.Box);
      centers.include(ToPoint(Mul(Add(triangle.Box.Lower,
                                     triangle.Box.Upper), .5)));
    }
    mNodes[nodeId].Box = box;
    if (end - begin <= 8) {
      mNodes[nodeId].Begin = begin;
      mNodes[nodeId].Count = end - begin;
      return nodeId;
    }
    int axis = 0;
    for (int i = 1; i < 3; ++i)
      if (centers.Upper[i] - centers.Lower[i] >
          centers.Upper[axis] - centers.Lower[axis])
        axis = i;
    const int middle = (begin + end) / 2;
    std::nth_element(mOrder.begin() + begin, mOrder.begin() + middle,
                     mOrder.begin() + end, [&](int a, int b) {
      const auto &first = mTriangles[a].Box;
      const auto &second = mTriangles[b].Box;
      return first.Lower[axis] + first.Upper[axis] <
             second.Lower[axis] + second.Upper[axis];
    });
    const int left = build(begin, middle), right = build(middle, end);
    mNodes[nodeId].Left = left;
    mNodes[nodeId].Right = right;
    return nodeId;
  }

  void nearest(int nodeId, const Point3 &query, int patch, double &best,
               Point3 &point) const {
    const BvhNode &node = mNodes[nodeId];
    if (BoxDistanceSquared(node.Box, query) > best) return;
    if (node.Count) {
      for (int i = node.Begin; i < node.Begin + node.Count; ++i) {
        const auto &triangle = mTriangles[mOrder[i]];
        if (patch >= 0 && triangle.PatchId != patch) continue;
        const Point3 candidate = ClosestPointOnTriangle(
            query, triangle.Points[0], triangle.Points[1],
            triangle.Points[2]);
        const Vec3 delta = Sub(ToVec(candidate), ToVec(query));
        const double squared = Dot(delta, delta);
        if (squared < best) {
          best = squared;
          point = candidate;
        }
      }
      return;
    }
    const double leftDistance = BoxDistanceSquared(mNodes[node.Left].Box, query);
    const double rightDistance = BoxDistanceSquared(mNodes[node.Right].Box, query);
    if (leftDistance < rightDistance) {
      nearest(node.Left, query, patch, best, point);
      nearest(node.Right, query, patch, best, point);
    } else {
      nearest(node.Right, query, patch, best, point);
      nearest(node.Left, query, patch, best, point);
    }
  }

  bool intersectsNode(int nodeId, const Box3 &box,
                      const std::array<Point3, 3> &candidate,
                      const std::array<int, 3> &vertexIds,
                      const std::unordered_set<int> &excludedFaces,
                      double epsilon, int *hitFace) const {
    const BvhNode &node = mNodes[nodeId];
    if (!BoxesOverlap(node.Box, box, epsilon)) return false;
    if (node.Count) {
      for (int i = node.Begin; i < node.Begin + node.Count; ++i) {
        const auto &triangle = mTriangles[mOrder[i]];
        if (excludedFaces.count(triangle.FaceId) ||
            !BoxesOverlap(triangle.Box, box, epsilon))
          continue;
        bool sharesVertex = false;
        for (int a : vertexIds)
          for (int b : triangle.VertexIds)
            sharesVertex = sharesVertex || (a >= 0 && a == b);
        if (sharesVertex) continue;
        if (TrianglesIntersect(candidate, triangle.Points, epsilon)) {
          if (hitFace) *hitFace = triangle.FaceId;
          return true;
        }
      }
      return false;
    }
    return intersectsNode(node.Left, box, candidate, vertexIds,
                          excludedFaces, epsilon, hitFace) ||
           intersectsNode(node.Right, box, candidate, vertexIds,
                          excludedFaces, epsilon, hitFace);
  }

  std::vector<SpatialTriangle> mTriangles;
  std::vector<int> mOrder;
  std::vector<BvhNode> mNodes;
};

struct CandidateTriangle {
  std::array<Point3, 3> Points;
  std::array<int, 3> VertexIds;
  Box3 Box;
};

CandidateTriangle MakeCandidate(const std::array<Point3, 3> &points,
                                const std::array<int, 3> &vertexIds) {
  CandidateTriangle result{points, vertexIds, {}};
  for (const Point3 &point : points) result.Box.include(point);
  return result;
}

struct CellKey {
  std::int64_t X = 0, Y = 0, Z = 0;
  bool operator==(const CellKey &other) const {
    return X == other.X && Y == other.Y && Z == other.Z;
  }
};

struct CellHash {
  std::size_t operator()(const CellKey &cell) const {
    std::size_t value = std::hash<std::int64_t>{}(cell.X);
    value ^= std::hash<std::int64_t>{}(cell.Y) + 0x9e3779b9 +
             (value << 6) + (value >> 2);
    value ^= std::hash<std::int64_t>{}(cell.Z) + 0x9e3779b9 +
             (value << 6) + (value >> 2);
    return value;
  }
};

class DynamicCollisionSet {
public:
  explicit DynamicCollisionSet(double cellSize)
      : mCellSize(std::max(cellSize, 1e-12)) {}

  bool intersects(const CandidateTriangle &candidate, double epsilon) const {
    const auto lower = cell(candidate.Box.Lower);
    const auto upper = cell(candidate.Box.Upper);
    std::unordered_set<int> visited;
    for (std::int64_t x = lower.X; x <= upper.X; ++x)
      for (std::int64_t y = lower.Y; y <= upper.Y; ++y)
        for (std::int64_t z = lower.Z; z <= upper.Z; ++z) {
          auto found = mCells.find({x, y, z});
          if (found == mCells.end()) continue;
          for (int id : found->second) {
            if (!visited.insert(id).second) continue;
            const auto &other = mTriangles[id];
            if (!BoxesOverlap(candidate.Box, other.Box, epsilon)) continue;
            bool sharesVertex = false;
            for (int a : candidate.VertexIds)
              for (int b : other.VertexIds)
                sharesVertex = sharesVertex || (a >= 0 && a == b);
            if (!sharesVertex && TrianglesIntersect(
                                     candidate.Points, other.Points, epsilon))
              return true;
          }
        }
    return false;
  }

  void insert(const CandidateTriangle &triangle) {
    const int id = int(mTriangles.size());
    mTriangles.push_back(triangle);
    const auto lower = cell(triangle.Box.Lower);
    const auto upper = cell(triangle.Box.Upper);
    for (std::int64_t x = lower.X; x <= upper.X; ++x)
      for (std::int64_t y = lower.Y; y <= upper.Y; ++y)
        for (std::int64_t z = lower.Z; z <= upper.Z; ++z)
          mCells[{x, y, z}].push_back(id);
  }

private:
  CellKey cell(const Vec3 &point) const {
    return {std::int64_t(std::floor(point[0] / mCellSize)),
            std::int64_t(std::floor(point[1] / mCellSize)),
            std::int64_t(std::floor(point[2] / mCellSize))};
  }
  double mCellSize;
  std::vector<CandidateTriangle> mTriangles;
  std::unordered_map<CellKey, std::vector<int>, CellHash> mCells;
};

bool ProjectAnalytic(const MeshPatch &patch, Point3 &point) {
  Vec3 p = ToVec(point);
  if (const auto *parameters = std::get_if<PlaneParameters>(&patch.Parameters)) {
    const Vec3 origin = ToVec(parameters->Plane.Origin);
    const Vec3 normal = Normalize(ToVec(parameters->Plane.Normal));
    p = Sub(p, Mul(normal, Dot(Sub(p, origin), normal)));
  } else if (const auto *parameters = std::get_if<CylinderParameters>(&patch.Parameters)) {
    const Vec3 origin = ToVec(parameters->Axis.Origin);
    const Vec3 axis = Normalize(ToVec(parameters->Axis.Direction));
    const Vec3 offset = Sub(p, origin);
    const double height = Dot(offset, axis);
    const Vec3 radial = Sub(offset, Mul(axis, height));
    if (Norm(radial) <= 1e-20 || !(parameters->Radius > 0)) return false;
    p = Add(origin, Add(Mul(axis, height), Mul(Normalize(radial), parameters->Radius)));
  } else if (const auto *parameters = std::get_if<ConeParameters>(&patch.Parameters)) {
    const Vec3 origin = ToVec(parameters->Axis.Origin);
    const Vec3 axis = Normalize(ToVec(parameters->Axis.Direction));
    const Vec3 offset = Sub(p, origin);
    double height = Dot(offset, axis);
    const Vec3 radial = Sub(offset, Mul(axis, height));
    if (height <= 0 || Norm(radial) <= 1e-20 || !(parameters->SemiAngle > 0)) return false;
    // Orthogonal projection in the axial/radial section.
    const double radius = Norm(radial), tangent = std::tan(parameters->SemiAngle);
    height = std::max(0.0, (height + tangent * radius) / (1 + tangent * tangent));
    p = Add(origin, Add(Mul(axis, height), Mul(Normalize(radial), height * tangent)));
  } else if (const auto *parameters = std::get_if<SphereParameters>(&patch.Parameters)) {
    const Vec3 center = ToVec(parameters->Center);
    const Vec3 radial = Sub(p, center);
    if (Norm(radial) <= 1e-20 || !(parameters->Radius > 0)) return false;
    p = Add(center, Mul(Normalize(radial), parameters->Radius));
  } else if (const auto *parameters = std::get_if<TorusParameters>(&patch.Parameters)) {
    const Vec3 origin = ToVec(parameters->Axis.Origin);
    const Vec3 axis = Normalize(ToVec(parameters->Axis.Direction));
    const Vec3 offset = Sub(p, origin);
    const double height = Dot(offset, axis);
    const Vec3 radial = Sub(offset, Mul(axis, height));
    if (Norm(radial) <= 1e-20 || !(parameters->MajorRadius > 0) ||
        !(parameters->MinorRadius > 0)) return false;
    const Vec3 ring = Add(origin, Mul(Normalize(radial), parameters->MajorRadius));
    const Vec3 tube = Sub(p, ring);
    if (Norm(tube) <= 1e-20) return false;
    p = Add(ring, Mul(Normalize(tube), parameters->MinorRadius));
  } else {
    return false;
  }
  point = ToPoint(p);
  return true;
}

class CudaEdgeClassifier {
public:
  bool available() const {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
  }
  bool classify(const std::vector<Point3> &vertices,
                const std::vector<EdgeRecord> &edges, double maximum,
                std::vector<unsigned char> &selected,
                std::vector<Point3> &midpoints, std::string &error) const {
    std::vector<std::array<double, 3>> packedVertices(vertices.size());
    std::vector<std::array<int, 2>> packedEdges(edges.size());
    for (std::size_t i = 0; i < vertices.size(); ++i)
      packedVertices[i] = {vertices[i][0], vertices[i][1], vertices[i][2]};
    for (std::size_t i = 0; i < edges.size(); ++i) {
      packedEdges[i] = {edges[i].A, edges[i].B};
    }
    std::vector<std::array<double, 3>> packedMidpoints;
    if (!ClassifyLongEdgesCudaRuntime(packedVertices, packedEdges, maximum,
                                      selected, packedMidpoints, error))
      return false;
    midpoints.resize(edges.size());
    for (std::size_t i = 0; i < edges.size(); ++i)
      midpoints[i] = Point3(packedMidpoints[i][0], packedMidpoints[i][1],
                            packedMidpoints[i][2]);
    return true;
  }
};

void CpuClassify(const std::vector<Point3> &vertices,
                 const std::vector<EdgeRecord> &edges, double maximum,
                 std::vector<unsigned char> &selected,
                 std::vector<Point3> &midpoints) {
  selected.assign(edges.size(), 0); midpoints.resize(edges.size());
  for (std::size_t i = 0; i < edges.size(); ++i) {
    const auto &edge = edges[i];
    selected[i] = Distance(vertices[edge.A], vertices[edge.B]) > maximum * (1 + 1e-6);
    midpoints[i] = ToPoint(Mul(Add(ToVec(vertices[edge.A]), ToVec(vertices[edge.B])), .5));
  }
}

void AppendTriangle(std::vector<std::array<int, 3>> &faces,
                    std::vector<int> &labels, int label, int a, int b, int c) {
  faces.push_back({a, b, c}); labels.push_back(label);
}

std::size_t SplitLongEdges(std::vector<Point3> &vertices,
                           std::vector<std::array<int, 3>> &faces,
                           std::vector<int> &labels,
                           std::unordered_set<EdgeKey> &constraints,
                           const std::vector<MeshPatch> &patches,
                           const NativeRemeshConfig &config,
                           const CudaEdgeClassifier &cuda, bool &usedCuda,
                           const std::vector<unsigned char> &rebuiltPatches,
                           bool onlyConstraints,
                           std::string &error) {
  (void)patches;
  std::size_t total = 0;
  for (int pass = 0; pass < config.SplitPasses; ++pass) {
    auto edges = BuildEdges(faces, &labels, &rebuiltPatches);
    if (onlyConstraints) {
      edges.erase(std::remove_if(edges.begin(), edges.end(), [&](const EdgeRecord &edge) {
        return !constraints.count(Key(edge.A, edge.B));
      }), edges.end());
    }
    if (edges.empty()) break;
    if (config.Verbose)
      std::clog << "[CadMesh] " << (onlyConstraints ? "boundary" : "local")
                << " split pass " << pass + 1 << ": faces=" << faces.size()
                << ", candidate_edges=" << edges.size() << std::endl;
    std::vector<unsigned char> selected;
    std::vector<Point3> midpoints;
#ifdef _WIN32
    if (cuda.available()) {
      if (cuda.classify(vertices, edges, config.TargetEdgeLength,
                        selected, midpoints, error)) {
        usedCuda = true;
      } else if (config.RequireCuda) {
        return total;
      } else {
        error.clear();
        CpuClassify(vertices, edges, config.TargetEdgeLength, selected, midpoints);
      }
    } else
#endif
      CpuClassify(vertices, edges, config.TargetEdgeLength, selected, midpoints);
    if (std::none_of(selected.begin(), selected.end(), [](unsigned char value) { return value != 0; }))
      break;
    std::unordered_map<EdgeKey, int> midpointIds;
    midpointIds.reserve(edges.size());
    std::vector<std::pair<EdgeKey, int>> splitConstraints;
    // Split the longest eligible edge first, with at most one split per
    // incident face in this batch. Splitting both long sides of a sliver
    // simultaneously reproduces its aspect ratio at every refinement level.
    std::vector<std::size_t> splitOrder;
    std::vector<double> splitLengths(edges.size(), 0);
    for (std::size_t i = 0; i < edges.size(); ++i) {
      if (!selected[i]) continue;
      splitOrder.push_back(i);
      splitLengths[i] = Distance(vertices[edges[i].A], vertices[edges[i].B]);
    }
    // Linear-time local maxima replace the global O(E log E) sort. Boundary
    // sampling can split all its edges together; it does not seed interiors.
    const std::size_t noEdge = edges.size();
    std::vector<std::size_t> longest(onlyConstraints ? 0 : faces.size(), noEdge);
    if (!onlyConstraints) for (std::size_t edgeId : splitOrder)
      for (int faceId : edges[edgeId].Faces) {
        if (faceId < 0 || faceId >= int(faces.size())) continue;
        auto &best = longest[faceId];
        if (best == noEdge || splitLengths[edgeId] > splitLengths[best] ||
            (splitLengths[edgeId] == splitLengths[best] && edgeId < best)) best = edgeId;
      }
    for (std::size_t edgeId : splitOrder) {
      if (!selected[edgeId]) continue;
      const auto &edge = edges[edgeId];
      if (onlyConstraints && !constraints.count(Key(edge.A,edge.B))) continue;
      bool frozen = false;
      for (int faceId : edge.Faces)
        if (faceId >= 0 && faceId < int(labels.size())) {
          const int patch = labels[faceId];
          frozen = frozen || (patch >= 0 && patch < int(rebuiltPatches.size()) &&
                              rebuiltPatches[patch]);
        }
      if (frozen) continue;
      bool longestOnBothSides = true;
      if (!onlyConstraints) for (int faceId : edge.Faces)
        if (faceId >= 0 && faceId < int(faces.size()) && longest[faceId] != edgeId)
          longestOnBothSides = false;
      if (!longestOnBothSides) continue;
      Point3 midpoint = midpoints[edgeId];
      const EdgeKey key = Key(edge.A, edge.B);
      const bool constrained = constraints.count(key) != 0;
      // A split on the current edge is geometrically contained in the two
      // incident source triangles. Moving it to an infinite analytic surface
      // can cross a nearby sheet, so projection is reserved for guarded
      // collapse and relaxation candidates.
      const int id = int(vertices.size());
      vertices.push_back(midpoint); midpointIds.emplace(key, id);
      if (constrained) splitConstraints.emplace_back(key, id);
      ++total;
    }
    if (midpointIds.empty()) break;
    std::vector<std::array<int, 3>> outputFaces;
    std::vector<int> outputLabels;
    outputFaces.reserve(faces.size() * 2); outputLabels.reserve(faces.size() * 2);
    for (std::size_t faceId = 0; faceId < faces.size(); ++faceId) {
      const auto f = faces[faceId];
      const int a = f[0], b = f[1], c = f[2];
      auto find = [&](int x, int y) {
        auto it = midpointIds.find(Key(x, y)); return it == midpointIds.end() ? -1 : it->second;
      };
      const int ab = find(a, b), bc = find(b, c), ca = find(c, a);
      const int mask = (ab >= 0 ? 1 : 0) | (bc >= 0 ? 2 : 0) | (ca >= 0 ? 4 : 0);
      const int label = labels[faceId];
      switch (mask) {
      case 0: AppendTriangle(outputFaces, outputLabels, label, a, b, c); break;
      case 1: AppendTriangle(outputFaces, outputLabels, label, a, ab, c);
              AppendTriangle(outputFaces, outputLabels, label, ab, b, c); break;
      case 2: AppendTriangle(outputFaces, outputLabels, label, b, bc, a);
              AppendTriangle(outputFaces, outputLabels, label, bc, c, a); break;
      case 4: AppendTriangle(outputFaces, outputLabels, label, c, ca, b);
              AppendTriangle(outputFaces, outputLabels, label, ca, a, b); break;
      case 3: AppendTriangle(outputFaces, outputLabels, label, b, bc, ab);
              AppendTriangle(outputFaces, outputLabels, label, a, ab, c);
              AppendTriangle(outputFaces, outputLabels, label, ab, bc, c); break;
      case 6: AppendTriangle(outputFaces, outputLabels, label, c, ca, bc);
              AppendTriangle(outputFaces, outputLabels, label, b, bc, a);
              AppendTriangle(outputFaces, outputLabels, label, bc, ca, a); break;
      case 5: AppendTriangle(outputFaces, outputLabels, label, a, ab, ca);
              AppendTriangle(outputFaces, outputLabels, label, c, ca, b);
              AppendTriangle(outputFaces, outputLabels, label, ca, ab, b); break;
      default: AppendTriangle(outputFaces, outputLabels, label, a, ab, ca);
               AppendTriangle(outputFaces, outputLabels, label, ab, b, bc);
               AppendTriangle(outputFaces, outputLabels, label, ca, bc, c);
               AppendTriangle(outputFaces, outputLabels, label, ab, bc, ca); break;
      }
    }
    faces.swap(outputFaces); labels.swap(outputLabels);
    for (const auto &[key, midpoint] : splitConstraints) {
      constraints.erase(key);
      constraints.insert(Key(KeyFirst(key), midpoint));
      constraints.insert(Key(midpoint, KeySecond(key)));
    }
  }
  return total;
}

std::size_t CollapseShortEdges(std::vector<Point3> &vertices,
                               std::vector<std::array<int, 3>> &faces,
                               std::vector<int> &labels,
                               const std::unordered_set<EdgeKey> &constraints,
                               const std::vector<MeshPatch> &patches,
                               const SurfaceIndex &reference,
                               std::size_t &collisionRejections,
                               const NativeRemeshConfig &config,
                               const std::vector<unsigned char> &rebuiltPatches) {
  std::size_t total = 0;
  const double cosine = std::cos(config.MaximumNormalDeviationDegrees *
                                 std::acos(-1.0) / 180.0);
  std::unordered_set<int> locked;
  for (EdgeKey key : constraints) { locked.insert(KeyFirst(key)); locked.insert(KeySecond(key)); }
  for (int pass = 0; pass < config.CollapsePasses; ++pass) {
    const auto edges = BuildEdges(faces, &labels, &rebuiltPatches);
    bool hasCandidate=false;
    for(const auto&edge:edges)
      if(edge.FaceCount==2&&!edge.NonManifold&&!locked.count(edge.A)&&
         !locked.count(edge.B)&&!constraints.count(Key(edge.A,edge.B))&&
         Distance(vertices[edge.A],vertices[edge.B])<config.TargetEdgeLength*.65&&
         labels[edge.Faces[0]]==labels[edge.Faces[1]]){hasCandidate=true;break;}
    if(!hasCandidate)break;
    std::vector<std::unordered_set<int>> neighbors(vertices.size());
    for (const auto &edge : edges) { neighbors[edge.A].insert(edge.B); neighbors[edge.B].insert(edge.A); }
    std::vector<std::vector<int>> incident(vertices.size());
    for (int faceId = 0; faceId < int(faces.size()); ++faceId) {
      const int owner=labels[faceId];
      if(owner>=0&&owner<int(rebuiltPatches.size())&&rebuiltPatches[owner])continue;
      for (int vertex : faces[faceId])
        incident[vertex].push_back(faceId);
    }
    const SurfaceIndex collision(vertices, faces, labels);
    std::vector<int> replacement(vertices.size());
    for (int i = 0; i < int(vertices.size()); ++i) replacement[i] = i;
    std::vector<unsigned char> claimed(vertices.size(), 0);
    std::vector<Point3> proposed = vertices;
    DynamicCollisionSet acceptedGeometry(config.TargetEdgeLength);
    std::size_t accepted = 0;
    for (const auto &edge : edges) {
      if (edge.FaceCount != 2 || edge.NonManifold || locked.count(edge.A) || locked.count(edge.B) ||
          claimed[edge.A] || claimed[edge.B] || constraints.count(Key(edge.A, edge.B)) ||
          Distance(vertices[edge.A], vertices[edge.B]) >= config.TargetEdgeLength * .65)
        continue;
      if (labels[edge.Faces[0]] != labels[edge.Faces[1]]) continue;
      const int patch = labels[edge.Faces[0]];
      if (patch < 0 || patch >= int(patches.size())) continue;
      if (patch < int(rebuiltPatches.size()) && rebuiltPatches[patch]) continue;
      std::vector<int> common;
      for (int vertex : neighbors[edge.A])
        if (neighbors[edge.B].count(vertex)) common.push_back(vertex);
      if (common.size() != 2) continue; // manifold link condition
      Point3 point = ToPoint(Mul(Add(ToVec(vertices[edge.A]), ToVec(vertices[edge.B])), .5));
      Point3 nearest;
      double sourceDistance = 0;
      bool acceptedAnalytic = false;
      if (patches[patch].ProjectionTarget ==
          PatchProjectionTarget::AnalyticSurface) {
        Point3 analytic = point;
        Point3 sourcePoint;
        double analyticDeviation = 0;
        if (ProjectAnalytic(patches[patch], analytic) &&
            reference.closest(analytic, patch, sourcePoint,
                              analyticDeviation) &&
            analyticDeviation <= config.MaximumDeviation) {
          point = analytic;
          acceptedAnalytic = true;
        }
      }
      if (!acceptedAnalytic) {
        if (!reference.closest(point, patch, nearest, sourceDistance)) continue;
        point = nearest;
      }
      bool valid = true;
      std::vector<int> affected = incident[edge.A];
      affected.insert(affected.end(), incident[edge.B].begin(), incident[edge.B].end());
      std::sort(affected.begin(), affected.end());
      affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
      auto position = [&](int vertex) -> const Point3 & {
        return vertex == edge.A ? point : vertices[vertex];
      };
      double oldMinimum = 1.0, newMinimum = 1.0;
      double oldSum = 0.0, newSum = 0.0;
      for (int faceId : affected) {
        if (!valid) break;
        const auto &face = faces[faceId];
        std::array<int, 3> candidate = face;
        for (int &vertex : candidate) if (vertex == edge.B) vertex = edge.A;
        if (candidate[0] == candidate[1] || candidate[1] == candidate[2] || candidate[2] == candidate[0])
          continue;
        const double before = TriangleQuality(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
        const double after = TriangleQuality(position(candidate[0]), position(candidate[1]),
                                             position(candidate[2]));
        oldMinimum = std::min(oldMinimum, before);
        newMinimum = std::min(newMinimum, after);
        oldSum += before;
        newSum += after;
        const Vec3 oldNormal = TriangleNormal(vertices, face);
        const Vec3 newCross = Cross(Sub(ToVec(position(candidate[1])),
                                        ToVec(position(candidate[0]))),
                                    Sub(ToVec(position(candidate[2])),
                                        ToVec(position(candidate[0]))));
        const double magnitude = Norm(newCross);
        if (magnitude <= 1e-20 ||
            Dot(Mul(newCross, 1.0 / magnitude), oldNormal) < cosine)
          valid = false;
        for (int side = 0; side < 3 && valid; ++side)
          if (Distance(position(candidate[side]),
                       position(candidate[(side + 1) % 3])) >
              config.TargetEdgeLength * (1 + 1e-6))
            valid = false;
      }
      if (newMinimum + 1e-10 < oldMinimum || newSum + 1e-10 < oldSum)
        valid = false;
      if (valid) {
        const std::unordered_set<int> excluded(affected.begin(), affected.end());
        std::vector<CandidateTriangle> localGeometry;
        for (int faceId : affected) {
          std::array<int, 3> candidate = faces[faceId];
          for (int &vertex : candidate)
            if (vertex == edge.B) vertex = edge.A;
          if (candidate[0] == candidate[1] ||
              candidate[1] == candidate[2] ||
              candidate[2] == candidate[0])
            continue;
          std::array<Point3, 3> points{position(candidate[0]),
                                       position(candidate[1]),
                                       position(candidate[2])};
          const CandidateTriangle geometry = MakeCandidate(points, candidate);
          if (collision.intersects(points, candidate, excluded,
                                   config.TargetEdgeLength * 1e-10) ||
              acceptedGeometry.intersects(
                  geometry, config.TargetEdgeLength * 1e-10)) {
            ++collisionRejections;
            valid = false;
            break;
          }
          localGeometry.push_back(geometry);
        }
        if (valid)
          for (const auto &geometry : localGeometry)
            acceptedGeometry.insert(geometry);
      }
      if (!valid) continue;
      replacement[edge.B] = edge.A; proposed[edge.A] = point;
      for (int faceId : affected)
        for (int vertex : faces[faceId]) claimed[vertex] = 1;
      ++accepted;
    }
    if (!accepted) break;
    std::vector<std::array<int, 3>> outputFaces;
    std::vector<int> outputLabels;
    for (std::size_t faceId = 0; faceId < faces.size(); ++faceId) {
      auto face = faces[faceId];
      for (int &vertex : face) vertex = replacement[vertex];
      if (face[0] == face[1] || face[1] == face[2] || face[2] == face[0]) continue;
      outputFaces.push_back(face); outputLabels.push_back(labels[faceId]);
    }
    vertices.swap(proposed); faces.swap(outputFaces); labels.swap(outputLabels);
    total += accepted;
  }
  return total;
}

std::size_t FlipEdges(std::vector<Point3> &vertices,
                      std::vector<std::array<int, 3>> &faces,
                      const std::vector<int> &labels,
                      const std::unordered_set<EdgeKey> &constraints,
                      std::size_t &collisionRejections,
                      const NativeRemeshConfig &config,
                      const std::vector<unsigned char> &rebuiltPatches) {
  std::size_t total = 0;
  const double cosine = std::cos(config.MaximumNormalDeviationDegrees *
                                 std::acos(-1.0) / 180.0);
  for (int pass = 0; pass < config.FlipPasses; ++pass) {
    const auto edges = BuildEdges(faces, &labels, &rebuiltPatches);
    const SurfaceIndex collision(vertices, faces, labels);
    std::unordered_set<EdgeKey> existing;
    for (const auto &edge : edges) existing.insert(Key(edge.A, edge.B));
    std::vector<unsigned char> usedFace(faces.size(), 0);
    std::vector<unsigned char> claimedVertex(vertices.size(), 0);
    DynamicCollisionSet acceptedGeometry(config.TargetEdgeLength);
    std::size_t accepted = 0;
    for (const auto &edge : edges) {
      if (edge.FaceCount != 2 || edge.NonManifold ||
          constraints.count(Key(edge.A, edge.B)))
        continue;
      const int first = edge.Faces[0], second = edge.Faces[1];
      if (usedFace[first] || usedFace[second] || labels[first] != labels[second]) continue;
      const int patch = labels[first];
      if (patch >= 0 && patch < int(rebuiltPatches.size()) && rebuiltPatches[patch]) continue;
      int c = -1, d = -1;
      for (int vertex : faces[first]) if (vertex != edge.A && vertex != edge.B) c = vertex;
      for (int vertex : faces[second]) if (vertex != edge.A && vertex != edge.B) d = vertex;
      if (c < 0 || d < 0 || c == d || existing.count(Key(c, d))) continue;
      if (claimedVertex[edge.A] || claimedVertex[edge.B] ||
          claimedVertex[c] || claimedVertex[d])
        continue;
      const Vec3 old0 = TriangleNormal(vertices, faces[first]);
      const Vec3 old1 = TriangleNormal(vertices, faces[second]);
      const Vec3 reference = Normalize(Add(old0, old1), old0);
      std::array<int, 3> candidate0{c, d, edge.B}, candidate1{d, c, edge.A};
      if (Dot(TriangleNormal(vertices, candidate0), reference) < 0) std::swap(candidate0[1], candidate0[2]);
      if (Dot(TriangleNormal(vertices, candidate1), reference) < 0) std::swap(candidate1[1], candidate1[2]);
      const double oldMinimum = std::min(
          TriangleQuality(vertices[faces[first][0]], vertices[faces[first][1]], vertices[faces[first][2]]),
          TriangleQuality(vertices[faces[second][0]], vertices[faces[second][1]], vertices[faces[second][2]]));
      const double newMinimum = std::min(
          TriangleQuality(vertices[candidate0[0]], vertices[candidate0[1]], vertices[candidate0[2]]),
          TriangleQuality(vertices[candidate1[0]], vertices[candidate1[1]], vertices[candidate1[2]]));
      if (newMinimum <= oldMinimum + 1e-8 ||
          Dot(TriangleNormal(vertices, candidate0), reference) < cosine ||
          Dot(TriangleNormal(vertices, candidate1), reference) < cosine ||
          Distance(vertices[c], vertices[d]) > config.TargetEdgeLength * (1 + 1e-6)) continue;
      const std::unordered_set<int> excluded{first, second};
      const std::array<Point3, 3> points0{vertices[candidate0[0]],
                                          vertices[candidate0[1]],
                                          vertices[candidate0[2]]};
      const std::array<Point3, 3> points1{vertices[candidate1[0]],
                                          vertices[candidate1[1]],
                                          vertices[candidate1[2]]};
      const CandidateTriangle geometry0 = MakeCandidate(points0, candidate0);
      const CandidateTriangle geometry1 = MakeCandidate(points1, candidate1);
      if (collision.intersects(points0, candidate0, excluded,
                               config.TargetEdgeLength * 1e-10) ||
          collision.intersects(points1, candidate1, excluded,
                               config.TargetEdgeLength * 1e-10) ||
          acceptedGeometry.intersects(
              geometry0, config.TargetEdgeLength * 1e-10) ||
          acceptedGeometry.intersects(
              geometry1, config.TargetEdgeLength * 1e-10))
      {
        ++collisionRejections;
        continue;
      }
      faces[first] = candidate0; faces[second] = candidate1;
      usedFace[first] = usedFace[second] = 1; existing.insert(Key(c, d)); ++accepted;
      claimedVertex[edge.A] = claimedVertex[edge.B] = 1;
      claimedVertex[c] = claimedVertex[d] = 1;
      acceptedGeometry.insert(geometry0);
      acceptedGeometry.insert(geometry1);
    }
    total += accepted;
    if (!accepted) break;
  }
  return total;
}

int RelaxVertices(std::vector<Point3> &vertices,
                  const std::vector<std::array<int, 3>> &faces,
                  const std::vector<int> &labels,
                  const std::unordered_set<EdgeKey> &constraints,
                  const std::vector<MeshPatch> &patches,
                  const SurfaceIndex &reference,
                  std::size_t &collisionRejections,
                  const NativeRemeshConfig &config,
                  const std::vector<unsigned char> &rebuiltPatches) {
  std::unordered_set<int> locked;
  const double cosine = std::cos(config.MaximumNormalDeviationDegrees *
                                 std::acos(-1.0) / 180.0);
  for (EdgeKey key : constraints) { locked.insert(KeyFirst(key)); locked.insert(KeySecond(key)); }
  int acceptedIterations = 0;
  for (int iteration = 0; iteration < config.RelaxIterations; ++iteration) {
    auto edges = BuildEdges(faces, &labels, &rebuiltPatches);
    if(edges.empty())break;
    const SurfaceIndex collision(vertices, faces, labels);
    std::vector<Vec3> sums(vertices.size(), {0, 0, 0});
    std::vector<int> counts(vertices.size(), 0), patch(vertices.size(), -2);
    std::vector<std::vector<int>> incident(vertices.size());
    std::vector<std::vector<int>> neighbors(vertices.size());
    for (const auto &edge : edges) {
      sums[edge.A] = Add(sums[edge.A], ToVec(vertices[edge.B])); ++counts[edge.A];
      sums[edge.B] = Add(sums[edge.B], ToVec(vertices[edge.A])); ++counts[edge.B];
      neighbors[edge.A].push_back(edge.B);
      neighbors[edge.B].push_back(edge.A);
    }
    for (int faceId = 0; faceId < int(faces.size()); ++faceId) {
      const int owner=labels[faceId];
      if(owner>=0&&owner<int(rebuiltPatches.size())&&rebuiltPatches[owner])continue;
      for (int vertex : faces[faceId]) {
        incident[vertex].push_back(faceId);
        if (patch[vertex] == -2) patch[vertex] = labels[faceId];
        else if (patch[vertex] != labels[faceId]) patch[vertex] = -1;
      }
    }
    auto proposed = vertices;
    std::vector<unsigned char> claimed(vertices.size(), 0);
    DynamicCollisionSet acceptedGeometry(config.TargetEdgeLength);
    bool moved = false;
    std::vector<int> activeVertices;
    activeVertices.reserve(edges.size());
    for(int vertex=0;vertex<int(counts.size());++vertex)
      if(counts[vertex])activeVertices.push_back(vertex);
    for (int vertex : activeVertices) {
      if (locked.count(vertex) || claimed[vertex] || !counts[vertex] ||
          patch[vertex] < 0 || patch[vertex] >= int(patches.size()))
        continue;
      if (patch[vertex] < int(rebuiltPatches.size()) && rebuiltPatches[patch[vertex]])
        continue;
      Point3 point = ToPoint(Add(Mul(ToVec(vertices[vertex]), .8),
                                 Mul(sums[vertex], .2 / counts[vertex])));
      Point3 nearest;
      double sourceDistance = 0;
      bool acceptedAnalytic = false;
      if (patches[patch[vertex]].ProjectionTarget ==
          PatchProjectionTarget::AnalyticSurface) {
        Point3 analytic = point, sourcePoint;
        double analyticDeviation = 0;
        if (ProjectAnalytic(patches[patch[vertex]], analytic) &&
            reference.closest(analytic, patch[vertex], sourcePoint,
                              analyticDeviation) &&
            analyticDeviation <= config.MaximumDeviation) {
          point = analytic;
          acceptedAnalytic = true;
        }
      }
      if (!acceptedAnalytic) {
        if (!reference.closest(point, patch[vertex], nearest,
                               sourceDistance))
          continue;
        point = nearest;
      }
      double before = 0, after = 0;
      double beforeMinimum = 1, afterMinimum = 1;
      bool valid = true;
      proposed[vertex] = point;
      for (int faceId : incident[vertex]) {
        const auto &face = faces[faceId];
        const double oldQuality = TriangleQuality(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
        const double newQuality = TriangleQuality(proposed[face[0]], proposed[face[1]], proposed[face[2]]);
        before += oldQuality; after += newQuality;
        beforeMinimum = std::min(beforeMinimum, oldQuality);
        afterMinimum = std::min(afterMinimum, newQuality);
        const Vec3 oldNormal = TriangleNormal(vertices, face);
        const Vec3 newNormal = TriangleNormal(proposed, face);
        if (Distance(proposed[face[0]], proposed[face[1]]) > config.TargetEdgeLength * (1 + 1e-6) ||
            Distance(proposed[face[1]], proposed[face[2]]) > config.TargetEdgeLength * (1 + 1e-6) ||
            Distance(proposed[face[2]], proposed[face[0]]) > config.TargetEdgeLength * (1 + 1e-6) ||
            Dot(newNormal, oldNormal) < cosine)
          valid = false;
      }
      std::vector<CandidateTriangle> localGeometry;
      if (valid) {
        const std::unordered_set<int> excluded(incident[vertex].begin(),
                                                incident[vertex].end());
        for (int faceId : incident[vertex]) {
          const auto &face = faces[faceId];
          const std::array<Point3, 3> points{proposed[face[0]],
                                             proposed[face[1]],
                                             proposed[face[2]]};
          const CandidateTriangle geometry = MakeCandidate(points, face);
          if (collision.intersects(points, face, excluded,
                                   config.TargetEdgeLength * 1e-10) ||
              acceptedGeometry.intersects(
                  geometry, config.TargetEdgeLength * 1e-10)) {
            ++collisionRejections;
            valid = false;
            break;
          }
          localGeometry.push_back(geometry);
        }
      }
      if (!valid || after + 1e-10 < before ||
          afterMinimum + 1e-10 < beforeMinimum)
        proposed[vertex] = vertices[vertex];
      else if (Distance(proposed[vertex], vertices[vertex]) > 1e-15) {
        for (const auto &geometry : localGeometry)
          acceptedGeometry.insert(geometry);
        moved = true;
        claimed[vertex] = 1;
        for (int neighbor : neighbors[vertex]) claimed[neighbor] = 1;
      }
    }
    if (!moved) break;
    vertices.swap(proposed); ++acceptedIterations;
  }
  return acceptedIterations;
}

void Compact(NativeRemeshResult &result, std::unordered_set<EdgeKey> &constraints) {
  std::vector<unsigned char> used(result.Vertices.size(), 0);
  for (const auto &face : result.Triangles) for (int vertex : face) used[vertex] = 1;
  std::vector<int> map(result.Vertices.size(), -1);
  std::vector<Point3> vertices;
  for (int i = 0; i < int(result.Vertices.size()); ++i)
    if (used[i]) { map[i] = int(vertices.size()); vertices.push_back(result.Vertices[i]); }
  for (auto &face : result.Triangles) for (int &vertex : face) vertex = map[vertex];
  std::unordered_set<EdgeKey> remapped;
  for (EdgeKey key : constraints)
    if (map[KeyFirst(key)] >= 0 && map[KeySecond(key)] >= 0)
      remapped.insert(Key(map[KeyFirst(key)], map[KeySecond(key)]));
  result.Vertices.swap(vertices); constraints.swap(remapped);
}

void Measure(NativeRemeshResult &result, double target) {
  auto &stats = result.Statistics;
  double sum = 0, minimum = 1, maximumEdge = 0, minimumAngle = 180;
  std::vector<double> qualities;
  qualities.reserve(result.Triangles.size());
  std::size_t below02 = 0;
  for (const auto &face : result.Triangles) {
    const auto &a = result.Vertices[face[0]], &b = result.Vertices[face[1]], &c = result.Vertices[face[2]];
    const double quality = TriangleQuality(a, b, c);
    sum += quality; minimum = std::min(minimum, quality);
    qualities.push_back(quality);
    below02 += quality < .2;
    const double lengths[3] = {Distance(a, b), Distance(b, c), Distance(c, a)};
    maximumEdge = std::max(maximumEdge, *std::max_element(lengths, lengths + 3));
    for (int corner = 0; corner < 3; ++corner) {
      const double adjacent0 = lengths[corner], adjacent1 = lengths[(corner + 2) % 3];
      const double opposite = lengths[(corner + 1) % 3];
      if (adjacent0 > 0 && adjacent1 > 0) {
        const double cosine = std::clamp((adjacent0 * adjacent0 + adjacent1 * adjacent1 - opposite * opposite) /
                                         (2 * adjacent0 * adjacent1), -1.0, 1.0);
        minimumAngle = std::min(minimumAngle, std::acos(cosine) * 180 / std::acos(-1.0));
      }
    }
  }
  stats.OutputVertices = result.Vertices.size(); stats.OutputTriangles = result.Triangles.size();
  stats.MaximumEdgeLength = maximumEdge;
  stats.MeanTriangleQuality = result.Triangles.empty() ? 0 : sum / result.Triangles.size();
  stats.MinimumTriangleQuality = result.Triangles.empty() ? 0 : minimum;
  if (!qualities.empty()) {
    const std::size_t percentile = std::min(qualities.size() - 1,
                                            qualities.size() / 20);
    std::nth_element(qualities.begin(), qualities.begin() + percentile,
                     qualities.end());
    stats.Percentile05TriangleQuality = qualities[percentile];
    stats.FractionBelow02TriangleQuality =
        double(below02) / qualities.size();
  }
  stats.MinimumAngleDegrees = result.Triangles.empty() ? 0 : minimumAngle;
  stats.QualityTargetMet = stats.MeanTriangleQuality + 1e-12 >= target;
}

std::array<int, 3> Color(int id) {
  unsigned value = unsigned(id + 1) * 2654435761u; value ^= value >> 16;
  return {64 + int(value & 127), 64 + int((value >> 8) & 127), 64 + int((value >> 16) & 127)};
}
} // namespace

bool NativeRemesher::remesh(const CadMeshPatchSegmenter &segmenter,
                            const NativeRemeshConfig &config,
                            NativeRemeshResult &result, std::string &error) {
  if (!(config.TargetEdgeLength > 0) || !(config.MaximumDeviation >= 0) ||
      !(config.MaximumNormalDeviationDegrees >= 0 && config.MaximumNormalDeviationDegrees <= 180) ||
      !(config.TargetMeanTriangleQuality >= 0 && config.TargetMeanTriangleQuality <= 1)) {
    error = "invalid native remesh configuration"; return false;
  }
  const auto &mesh = segmenter.getMesh();
  using Clock=std::chrono::steady_clock;
  Clock::time_point phaseStart;
  double boundarySplitSeconds=0,analyticSeconds=0,localSplitSeconds=0,
         collapseSeconds=0,flipSeconds=0,relaxSeconds=0,validationSeconds=0;
  result = {};
  result.Vertices.reserve(mesh.getVertices().size());
  for (const auto &vertex : mesh.getVertices()) result.Vertices.push_back(vertex.Position);
  result.Triangles.reserve(mesh.getTriangles().size()); result.PatchIds.reserve(mesh.getTriangles().size());
  for (const auto &triangle : mesh.getTriangles()) {
    result.Triangles.push_back(triangle.VertexIds); result.PatchIds.push_back(triangle.PatchId);
  }
  auto &stats = result.Statistics;
  stats.InputVertices = result.Vertices.size(); stats.InputTriangles = result.Triangles.size();
  if (config.Verbose) std::clog << "[CadMesh] native remesh: building reference index" << std::endl;
  const SurfaceIndex reference(result.Vertices, result.Triangles,
                               result.PatchIds);
  std::unordered_set<EdgeKey> constraints;
  for (int edgeId : segmenter.getRemeshConstraint().ConstraintEdgeIds) {
    if (edgeId < 0 || edgeId >= int(mesh.getEdges().size())) continue;
    const auto &edge = mesh.getEdges()[edgeId]; constraints.insert(Key(edge.Vertex0, edge.Vertex1));
  }
  if (config.Verbose) std::clog << "[CadMesh] native remesh: initializing CUDA" << std::endl;
  CudaEdgeClassifier cuda;
  if (config.RequireCuda && !cuda.available()) {
    error = "native remesh CUDA Driver/NVRTC runtime is unavailable"; return false;
  }
  std::vector<unsigned char> rebuiltPatches;
  // Sample every shared boundary before chart construction. Analytic charts
  // then consume the same global boundary ids on both incident patches.
  phaseStart=Clock::now();
  stats.Splits = SplitLongEdges(result.Vertices, result.Triangles,
                                result.PatchIds, constraints,
                                segmenter.getPatches(), config, cuda,
                                stats.UsedCuda, rebuiltPatches, true, error);
  if (!error.empty()) return false;
  boundarySplitSeconds=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  if (config.Verbose)
    std::clog << "[CadMesh] boundary split complete: " << boundarySplitSeconds
              << " s; starting analytic rebuild" << std::endl;
  phaseStart=Clock::now();
  const auto analyticBaseTriangles = result.Triangles;
  const auto analyticBasePatchIds = result.PatchIds;
  std::vector<unsigned char> excludedAnalyticPatches(
      segmenter.getPatches().size(), 0);
  AnalyticPatchRemeshReport analyticReport;
  RebuildAnalyticPatches(result.Vertices, result.Triangles, result.PatchIds,
                         segmenter.getPatches(), config.TargetEdgeLength,
                         config.MaximumDeviation,
                         config.MaximumNormalDeviationDegrees,
                         config.TargetMeanTriangleQuality,
                         excludedAnalyticPatches, rebuiltPatches,
                         analyticReport, config.Verbose);
  // Chart boundaries retain their original global vertex ids. Restore only
  // rejected patches using those ids; unrelated charts need no retriangulation.
  for (int collisionAttempt = 0; collisionAttempt < 4; ++collisionAttempt) {
    if (std::none_of(rebuiltPatches.begin(), rebuiltPatches.end(),
                     [](unsigned char rebuilt) { return rebuilt != 0; })) break;
    if (config.Verbose)
      std::clog << "[CadMesh] analytic rebuild complete; collision guard attempt "
                << collisionAttempt + 1 << std::endl;
    const SurfaceIndex analyticCollision(result.Vertices, result.Triangles,
                                         result.PatchIds);
    std::vector<unsigned char> rejected;
    const std::size_t collisionPatches =
        analyticCollision.collectIntersectingRebuiltPatches(
            rebuiltPatches, rejected, config.TargetEdgeLength * 1e-10);
    if (!collisionPatches) break;
    if (config.Verbose)
      std::clog << "[CadMesh] analytic collision guard: rejecting "
                << collisionPatches << " rebuilt patches (attempt "
                << collisionAttempt + 1 << ")\n";
    if (collisionAttempt >= 2) {
      // A collision chain can hide later contacts behind the first BVH hit.
      // The bounded final retry restores every remaining analytic patch to its
      // original surface triangulation instead of allowing a late hard fail.
      rejected = rebuiltPatches;
    }
    const auto restorePatch = [&](int patch) {
      return patch >= 0 && patch < int(rejected.size()) && rejected[patch] &&
             rebuiltPatches[patch];
    };
    std::vector<std::array<int, 3>> retainedFaces;
    std::vector<int> retainedLabels;
    retainedFaces.reserve(result.Triangles.size());
    retainedLabels.reserve(result.PatchIds.size());
    for (std::size_t face = 0; face < result.Triangles.size(); ++face) {
      if (restorePatch(result.PatchIds[face])) {
        --analyticReport.RebuiltFaces;
        continue;
      }
      retainedFaces.push_back(result.Triangles[face]);
      retainedLabels.push_back(result.PatchIds[face]);
    }
    for (std::size_t face = 0; face < analyticBaseTriangles.size(); ++face) {
      if (!restorePatch(analyticBasePatchIds[face])) continue;
      retainedFaces.push_back(analyticBaseTriangles[face]);
      retainedLabels.push_back(analyticBasePatchIds[face]);
    }
    result.Triangles.swap(retainedFaces);
    result.PatchIds.swap(retainedLabels);
    for (int patch = 0; patch < int(rebuiltPatches.size()); ++patch) {
      if (!restorePatch(patch)) continue;
      rebuiltPatches[patch] = 0;
      --analyticReport.Rebuilt;
      ++analyticReport.Fallback;
      ++analyticReport.CollisionFallback;
    }
  }
  analyticSeconds=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  stats.AnalyticPatchesAttempted = analyticReport.Attempted;
  stats.AnalyticPatchesRebuilt = analyticReport.Rebuilt;
  stats.AnalyticPatchesFallback = analyticReport.Fallback;
  if (config.Verbose)
    std::clog << "[CadMesh] analytic parameter-domain rebuild: "
              << analyticReport.Rebuilt << " / " << analyticReport.Attempted
              << " patches, fallback=" << analyticReport.Fallback
              << " (deviation=" << analyticReport.DeviationFallback
              << ", topology=" << analyticReport.TopologyFallback
              << ", parameterization=" << analyticReport.ParameterizationFallback
              << ", quality=" << analyticReport.QualityFallback
              << ", collision=" << analyticReport.CollisionFallback << ")"
              << ", rebuilt_faces=" << analyticReport.RebuiltFaces
              << ", time(index=" << analyticReport.IndexSeconds
              << "s, charts=" << analyticReport.ChartSeconds
              << "s, commit=" << analyticReport.CommitSeconds << "s)\n";
  // Drop superseded analytic interior vertices before any CUDA transfer or
  // fallback adjacency allocation. Boundary constraints are remapped here.
  Compact(result,constraints);
  phaseStart=Clock::now();
  stats.Splits += SplitLongEdges(result.Vertices, result.Triangles,
                                 result.PatchIds, constraints,
                                 segmenter.getPatches(), config, cuda,
                                 stats.UsedCuda, rebuiltPatches, false, error);
  if (!error.empty()) return false;
  localSplitSeconds+=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  phaseStart=Clock::now();
  stats.Collapses = CollapseShortEdges(result.Vertices, result.Triangles, result.PatchIds,
                                       constraints, segmenter.getPatches(),
                                       reference, stats.CollisionRejections,
                                       config, rebuiltPatches);
  collapseSeconds=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  // A parallel collapse batch validates each candidate against the mesh at the
  // start of the pass. Two disjoint accepted collapses can still become the
  // endpoints of the same new edge, so restore the hard edge-length invariant
  // before quality optimization.
  phaseStart=Clock::now();
  stats.Splits += SplitLongEdges(result.Vertices, result.Triangles,
                                 result.PatchIds, constraints,
                                 segmenter.getPatches(), config, cuda,
                                 stats.UsedCuda, rebuiltPatches, false, error);
  if (!error.empty()) return false;
  localSplitSeconds+=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  phaseStart=Clock::now();
  stats.Flips = FlipEdges(result.Vertices, result.Triangles,
                          result.PatchIds, constraints,
                          stats.CollisionRejections, config, rebuiltPatches);
  flipSeconds=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  phaseStart=Clock::now();
  stats.AcceptedRelaxations = RelaxVertices(result.Vertices, result.Triangles, result.PatchIds,
                                             constraints, segmenter.getPatches(),
                                             reference,
                                             stats.CollisionRejections, config,
                                             rebuiltPatches);
  relaxSeconds=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  // Keep the file-producing path self-contained: if a numerical edge case in
  // an optimization pass creates an overlong edge, repair it here instead of
  // rejecting an otherwise usable remesh at final validation.
  phaseStart=Clock::now();
  const std::size_t finalSplits = SplitLongEdges(
      result.Vertices, result.Triangles, result.PatchIds, constraints,
      segmenter.getPatches(), config, cuda, stats.UsedCuda, rebuiltPatches,
      false, error);
  localSplitSeconds+=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  stats.Splits += finalSplits;
  if (!error.empty()) return false;
  if (finalSplits) {
    phaseStart=Clock::now();
    stats.Flips += FlipEdges(result.Vertices, result.Triangles,
                             result.PatchIds, constraints,
                             stats.CollisionRejections, config,
                             rebuiltPatches);
    flipSeconds+=std::chrono::duration<double>(Clock::now()-phaseStart).count();
    phaseStart=Clock::now();
    stats.AcceptedRelaxations += RelaxVertices(
        result.Vertices, result.Triangles, result.PatchIds, constraints,
        segmenter.getPatches(), reference, stats.CollisionRejections,
        config, rebuiltPatches);
    relaxSeconds+=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  }
  phaseStart=Clock::now();
  Compact(result, constraints); Measure(result, config.TargetMeanTriangleQuality);
  const SurfaceIndex finalCollision(result.Vertices, result.Triangles,
                                    result.PatchIds);
  int intersection0 = -1, intersection1 = -1;
  const bool hasSelfIntersection = finalCollision.firstSelfIntersection(
      intersection0, intersection1, config.TargetEdgeLength * 1e-10);
  stats.SelfIntersectionFree = !hasSelfIntersection;
  if (hasSelfIntersection) {
    std::clog << "[CadMesh] warning: output self-intersection between faces "
              << intersection0 << " (patch " << result.PatchIds[intersection0]
              << ") and " << intersection1 << " (patch "
              << result.PatchIds[intersection1] << ")";
    int sourceIntersection0 = -1, sourceIntersection1 = -1;
    if (reference.firstSelfIntersection(
            sourceIntersection0, sourceIntersection1,
            config.TargetEdgeLength * 1e-10)) {
      std::clog << "; input mesh already self-intersects between source faces "
                << sourceIntersection0 << " (patch "
                << mesh.getTriangles()[sourceIntersection0].PatchId << ") and "
                << sourceIntersection1 << " (patch "
                << mesh.getTriangles()[sourceIntersection1].PatchId << ")";
    } else {
      std::clog << "; input mesh is intersection-free, so this intersection "
                   "was introduced during remesh";
    }
    std::clog << "; continuing because self-intersection is diagnostic only\n";
  }
  validationSeconds=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  if (stats.MaximumEdgeLength > config.TargetEdgeLength * (1 + 1e-6)) {
    std::ostringstream message;
    message << std::setprecision(17)
            << "native remesh maximum edge " << stats.MaximumEdgeLength
            << " exceeds target " << config.TargetEdgeLength;
    error = message.str();
    return false;
  }
  if (config.Verbose) {
    std::clog << "[CadMesh] native remesh: " << stats.InputTriangles << " -> "
              << stats.OutputTriangles << " triangles; splits=" << stats.Splits
              << ", collapses=" << stats.Collapses << ", flips=" << stats.Flips
              << ", relax=" << stats.AcceptedRelaxations
              << ", collision_rejections=" << stats.CollisionRejections
              << ", analytic=" << stats.AnalyticPatchesRebuilt << "/"
              << stats.AnalyticPatchesAttempted
              << ", analytic_fallback=" << stats.AnalyticPatchesFallback
              << ", CUDA=" << (stats.UsedCuda ? "yes" : "no")
              << ", mean quality=" << stats.MeanTriangleQuality
              << " (target " << config.TargetMeanTriangleQuality << ")"
              << ", q05=" << stats.Percentile05TriangleQuality
              << ", below_q0.2="
              << 100.0 * stats.FractionBelow02TriangleQuality << "%\n";
    std::clog << "[CadMesh] native remesh timing: boundary_split="
              << boundarySplitSeconds << "s, analytic=" << analyticSeconds
              << "s, local_split=" << localSplitSeconds
              << "s, collapse=" << collapseSeconds
              << "s, flip=" << flipSeconds << "s, relax=" << relaxSeconds
              << "s, validation=" << validationSeconds << "s\n";
  }
  return true;
}

bool NativeRemesher::writePly(const NativeRemeshResult &result,
                              const std::vector<MeshPatch> &patches,
                              const std::filesystem::path &path,
                              std::string &error) {
  std::error_code filesystemError;
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), filesystemError);
  if (filesystemError) { error = filesystemError.message(); return false; }
  std::ofstream out(path);
  if (!out) { error = "cannot open output PLY"; return false; }
  out << "ply\nformat ascii 1.0\ncomment color_by surface_instance"
         "\ncomment surface_type_ids 0=Unknown 1=Plane 2=Cylinder 3=Cone 4=Sphere 5=Torus 6=Freeform"
         "\ncomment feature_role_ids 0=Ordinary 1=Fillet\nelement vertex "
      << result.Vertices.size()
      << "\nproperty double x\nproperty double y\nproperty double z\nelement face "
      << result.Triangles.size()
      << "\nproperty list uchar int vertex_indices\nproperty int patch_id"
         "\nproperty int primitive_type\nproperty uchar red\nproperty uchar green"
         "\nproperty uchar blue\nproperty int feature_role\nend_header\n"
      << std::setprecision(17);
  for (const auto &vertex : result.Vertices)
    out << vertex.X() << ' ' << vertex.Y() << ' ' << vertex.Z() << '\n';
  for (std::size_t i = 0; i < result.Triangles.size(); ++i) {
    const int patchId = result.PatchIds[i];
    const MeshPatch *patch = patchId >= 0 && patchId < int(patches.size()) ? &patches[patchId] : nullptr;
    const auto color = Color(patchId);
    const auto &face = result.Triangles[i];
    out << "3 " << face[0] << ' ' << face[1] << ' ' << face[2] << ' ' << patchId << ' '
        << SurfaceTypeId(patch ? patch->SurfaceType : PatchSurfaceType::Unknown) << ' '
        << color[0] << ' ' << color[1] << ' ' << color[2] << ' '
        << (patch && patch->FeatureRole == PatchFeatureRole::Fillet ? 1 : 0) << '\n';
  }
  out.flush();
  if (!out) { error = "writing output PLY failed"; return false; }
  return true;
}

} // namespace CadMesh
