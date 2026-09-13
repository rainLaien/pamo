// Adapted from Apollo algorithm/thickness/wall_thickness_calculator.cpp.
// Face rolling-ball search, containment and representative refinement retained.
// Native geometry/BVH backend; no VCGLib dependency.
// Apollo model/UI/export and vertex display reconstruction are not included.
#include "CadMesh/WallThicknessCalculator.h"
#include "RemeshWorkerPool.h"
#include "WallThicknessGeometry.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <unordered_set>
namespace CadMesh { namespace {
using ThicknessGeometry::Vector;
using ThicknessGeometry::Line;
using ThicknessGeometry::Mesh;
using ThicknessGeometry::Face;
using ThicknessGeometry::Bvh;
using ThicknessGeometry::Node;
using std::vector;
constexpr float kThicknessSampleConeAngleRad=0.1308996939f;
constexpr int kThicknessDirectionCount=9;
// C++17 view for Apollo's span parameters; preserves const element access.
template<class T> class ArrayView {
  T* pointer; std::size_t count;
public:
  template<class C> ArrayView(C& c):pointer(c.data()),count(c.size()){}
  T* begin() const{return pointer;} T* end() const{return pointer+count;}
  T* data() const{return pointer;}
  std::size_t size() const{return count;}
  bool empty() const{return count==0;}
  ArrayView(T* p,std::size_t n):pointer(p),count(n){}
  ArrayView first(std::size_t n)const{return {pointer,n};}
  T& operator[](std::size_t i) const{return pointer[i];}
};
struct PointOnFace {
  // int face;
  Face* face = nullptr;
  Vector point;
  // Eigen::Vector3 point;
};
using EdgeId              = int;
static constexpr auto eps = 10 * std::numeric_limits<double>::epsilon();

struct TriPointf {
  /// barycentric coordinates:
  /// a+b in [0,1], a+b=0 => point is in v0, a+b=1 => point is on [v1,v2] edge
  /// a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
  double a = 0.0;
  /// b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
  double b = 0.0;
  int inVertex() const {
    if (a <= eps && b <= eps) return 0;
    if (1 - a - b <= eps) {
      if (b <= eps) return 1;
      if (a <= eps) return 2;
    }
    return -1;
  }
  int onEdge() const {
    if (1 - a - b <= eps) return 0;
    if (a <= eps) return 1;
    if (b <= eps) return 2;
    return -1;
  }
};

struct MeshTriPoint { TriPointf bary; };
struct MeshIntersectionResult {
  /// stores intersected face and global coordinates
  PointOnFace proj;

  /// stores barycentric coordinates
  MeshTriPoint mtp;

  /// stores the distance from ray origin to the intersection point in direction units
  double distanceAlongLine = 0;
  explicit operator bool() const { return proj.face; }
};
struct MeshProjectionResult {
  /// the closest point on mesh, transformed by xf if it is given
  PointOnFace proj;
  /// its barycentric representation
  MeshTriPoint mtp;
  /// squared distance from pt to proj
  double distSq = 0;
  explicit operator bool() const { return proj.face; }
};
struct InSphereSearchSettings {
  /// if false then searches for the maximal inscribed sphere in mesh;
  /// if true then searches for both a) maximal inscribed sphere, and b) maximal sphere outside the
  /// mesh touching it at two points;
  ///              and returns the smaller of two, and if it is b) then with minus sign
  bool insideAndOutside = false;

  /// maximum allowed radius of the sphere;
  /// for almost closed meshes the article recommends maxRadius = 0.5 * std::min( { boxSize.x,
  /// boxSize.y, boxSize.z } )
  double maxRadius       = 1;

  /// maximum number of shrinking iterations for one triangle
  int maxIters          = 64;

  /// iterations stop if next radius is larger than minShrinkage times previous radius
  double minShrinkage    = 0.9999999;

  /// minimum cosine of the angle between two unit directions:
  /// 1) search unit direction (m.inDir),
  /// 2) unit direction from sphere's center to the other found touch point;
  /// -1 includes all contacts. Used only AFTER containment and convergence;
  /// it classifies contact separation and never removes geometric constraints.
  double minAngleCos     = -1.0;  // disabled unless contact classification is requested

  /// Minimum cosine between the inward search direction and the normal of the
  /// opposite surface. A positive value rejects side contacts and surfaces
  /// facing the wrong way.
  double minOppositeNormalCos = -1;
};
struct ContainmentReport {
  unsigned reason = 0;
  double depth = 0;
  double radius = 0;
  double tolerance = 0;
  int face = -1;
  unsigned faceKind = 0;
};

struct InSphere {
  Vector center;
  Vector referenceCenter;
  double radius = 0;
  bool contained = true;
  bool converged = false;
  bool wallContact = false;
  RollingBallSample sample;
  bool sourceRelocated = false;
  double sourceDisplacement = 0;
  ContainmentReport rejection;
  MeshProjectionResult oppositeTouchPoint;  ///< excluding input point and incident triangles,
                                            ///< distSq - squared distance to sphere's center

  bool isValid() const { return contained && converged && wallContact && oppositeTouchPoint.proj.face != nullptr && std::isfinite(radius) && radius != 0.0; }
};
using FacePredicate = std::function<bool(Face*)>;

void BuildThicknessAabbTree(Mesh& mesh,Bvh& tree){tree.initialize(mesh);}

auto BuildThicknessSampleDirections(const Vector& inwardDirection)
    -> std::array<Vector, kThicknessDirectionCount> {
  std::array<Vector, kThicknessDirectionCount> directions;
  const auto inward = inwardDirection.normalized();
  const Vector helperAxis = std::abs(inward.X()) < 0.8f ? Vector{1.0f, 0.0f, 0.0f}
                                                              : Vector{0.0f, 1.0f, 0.0f};
  const auto firstTangent  = (inward ^ helperAxis).normalized();
  const auto secondTangent = (inward ^ firstTangent).normalized();
  directions[0]            = inward;

  constexpr float fullTurn = 6.2831853072f;
  const float axialWeight  = std::cos(kThicknessSampleConeAngleRad);
  const float radialWeight = std::sin(kThicknessSampleConeAngleRad);
  for (int index = 1; index < kThicknessDirectionCount; ++index) {
    const float azimuth = fullTurn * static_cast<float>(index - 1) /
                          static_cast<float>(kThicknessDirectionCount - 1);
    const auto radialDirection = firstTangent * std::cos(azimuth) + secondTangent * std::sin(azimuth);
    directions[index]          = (inward * axialWeight + radialDirection * radialWeight).normalized();
  }
  return directions;
}

struct MeshPoint {
  Face* sourceFace = nullptr;
  MeshTriPoint triPoint;           ///< relative position on mesh
  Vector pt;                 ///< 3d coordinates
  Vector inDir;              ///< sphere-center direction inside the mesh = minus surface normal
  FacePredicate notIncidentFaces;  ///< predicate that returns true for mesh faces not-incident to the point
  FacePredicate sphereConstraintFaces;  ///< also admits incident faces at sharp features
  FacePredicate sharpIncidentFaces;  ///< sharp incident faces bypass smooth-sheet direction filtering

};

void findMaxVectorDim(int& dimX, int& dimY, int& dimZ, const Vector& dir) {
  if (dir.X() > dir.Y()) {
    if (dir.X() > dir.Z()) {
      if (dir.Y() > dir.Z()) {
        // X>Y>Z
        if (-dir.Z() > dir.X()) {
          dimZ = 2;
          dimX = 1;
          dimY = 0;
        } else {
          dimZ = 0;
          dimX = 1;
          dimY = 2;
        }
      } else {
        // X>Z>Y
        if (-dir.Y() > dir.X()) {
          dimZ = 1;
          dimX = 0;
          dimY = 2;
        } else {
          dimZ = 0;
          dimX = 1;
          dimY = 2;
        }
      }
    } else {
      // Z>X>Y
      if (-dir.Y() > dir.Z()) {
        dimZ = 1;
        dimX = 0;
        dimY = 2;
      } else {
        dimZ = 2;
        dimX = 0;
        dimY = 1;
      }
    }
  } else {
    if (dir.Y() > dir.Z()) {
      if (dir.X() < dir.Z()) {
        // Y>Z>X
        if (-dir.X() > dir.Y()) {
          dimZ = 0;
          dimX = 2;
          dimY = 1;
        } else {
          dimZ = 1;
          dimX = 2;
          dimY = 0;
        }
      } else {
        // Y>X>Z
        if (-dir.Z() > dir.Y()) {
          dimZ = 2;
          dimX = 1;
          dimY = 0;
        } else {
          dimZ = 1;
          dimX = 2;
          dimY = 0;
        }
      }
    } else {
      // Z>Y>X
      if (-dir.X() > dir.Z()) {
        dimZ = 0;
        dimX = 2;
        dimY = 1;
      } else {
        dimZ = 2;
        dimX = 0;
        dimY = 1;
      }
    }
  }
}
struct IntersectionPrecomputes {
  // {1 / dir}
  Vector invDir;
  // [0]max, [1]next, [2]next-next
  // f.e. {1,2,-3} => {2,1,0}
  int maxDimIdxZ = 2;
  int idxX       = 0;
  int idxY       = 1;

  /// stores signs of direction vector;
  std::array<int,3> sign;

  /// precomputed factors
  double Sx, Sy, Sz;
  IntersectionPrecomputes() = default;
  IntersectionPrecomputes(const Vector& dir) {
    findMaxVectorDim(idxX, idxY, maxDimIdxZ, dir);
    sign[0]   = dir.X() >= 0 ? 1 : 0;
    sign[1]   = dir.Y() >= 0 ? 1 : 0;
    sign[2]   = dir.Z() >= 0 ? 1 : 0;

    Sx        = dir[idxX] / dir[maxDimIdxZ];
    Sy        = dir[idxY] / dir[maxDimIdxZ];
    Sz        = 1 / dir[maxDimIdxZ];

    invDir[0] = (dir.X() == 0) ? std::numeric_limits<double>::max() : 1 / dir.X();
    invDir[1] = (dir.Y() == 0) ? std::numeric_limits<double>::max() : 1 / dir.Y();
    invDir[2] = (dir.Z() == 0) ? std::numeric_limits<double>::max() : 1 / dir.Z();
  }
};

std::pair<Vector, TriPointf> closestPointInTriangle(const Vector& p, const Vector& a,
                                                          const Vector& b, const Vector& c) {
  const Vector ab = b - a;
  const Vector ac = c - a;
  const double edgeScale = std::max({ab.SquaredNorm(), ac.SquaredNorm(), (c-b).SquaredNorm()});
  const double roundoff = 64.0 * std::numeric_limits<double>::epsilon() * std::numeric_limits<double>::epsilon();
  if (!(edgeScale > 0) || (ab ^ ac).SquaredNorm() <= roundoff * edgeScale * edgeScale) {
    auto segment = [&](const Vector& start, const Vector& end, TriPointf first, TriPointf last) {
      const auto direction = end - start;
      const double lengthSq = direction.SquaredNorm();
      const double t = lengthSq > 0 ? std::clamp((p-start).dot(direction)/lengthSq, 0.0, 1.0) : 0.0;
      return std::pair{start + direction*t, TriPointf{first.a*(1-t)+last.a*t, first.b*(1-t)+last.b*t}};
    };
    auto result = segment(a,b,{0,0},{1,0});
    for (const auto& candidate : {segment(a,c,{0,0},{0,1}), segment(b,c,{1,0},{0,1})})
      if ((candidate.first-p).SquaredNorm() < (result.first-p).SquaredNorm()) result = candidate;
    return result;
  }
  const Vector ap = p - a;

  const double d1       = ab.dot(ap);
  const double d2       = ac.dot(ap);
  if (d1 <= 0 && d2 <= 0) return {a, {0, 0}};  // #1

  const Vector bp = p - b;
  const double d3       = ab.dot(bp);
  const double d4       = ac.dot(bp);

  if (d3 >= 0 && d4 <= d3) return {b, {1, 0}};  // #2

  const Vector cp = p - c;
  const double d5       = ab.dot(cp);
  const double d6       = ac.dot(cp);

  if (d6 >= 0 && d5 <= d6) return {c, {0, 1}};  // #3

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const double v = d1 / (d1 - d3);
    return {a + v * ab, {static_cast<double>(v), 0}};  // #4
  }

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d6 <= 0) {
    const double v = d2 / (d2 - d6);
    return {a + v * ac, {0, static_cast<double>(v)}};  // #5
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0) {
    if (d4 < d3)           // floating-point rounding errors
      return {b, {1, 0}};  // #2
    if (d5 < d6)           // floating-point rounding errors
      return {c, {0, 1}};  // #3

    const double v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return {b + v * (c - b), {static_cast<double>(1 - v), static_cast<double>(v)}};  // #6
  }

  const double denom = 1 / (va + vb + vc);
  const double v     = vb * denom;
  const double w     = vc * denom;
  return {a + v * ab + w * ac, {static_cast<double>(v), static_cast<double>(w)}};  // #0
}

bool SPIntersectionRayTriangle(const Line& line, const Vector& vert0, const Vector& vert1, const Vector& vert2,
                               const IntersectionPrecomputes& prec, double& hitDistance, double& bar1, double& bar2) {
  const Vector a = vert0 - line.Origin();
  const Vector b = vert1 - line.Origin();
  const Vector c = vert2 - line.Origin();

  const double ax  = a[prec.idxX] - prec.Sx * a[prec.maxDimIdxZ];
  const double ay  = a[prec.idxY] - prec.Sy * a[prec.maxDimIdxZ];
  const double bx  = b[prec.idxX] - prec.Sx * b[prec.maxDimIdxZ];
  const double by  = b[prec.idxY] - prec.Sy * b[prec.maxDimIdxZ];
  const double cx  = c[prec.idxX] - prec.Sx * c[prec.maxDimIdxZ];
  const double cy  = c[prec.idxY] - prec.Sy * c[prec.maxDimIdxZ];

  const double coordinateScale =
      std::max({std::abs(ax), std::abs(ay), std::abs(bx), std::abs(by), std::abs(cx), std::abs(cy)});
  const double tolerance = 8.0 * std::numeric_limits<double>::epsilon() * coordinateScale * coordinateScale;

  const double u         = cx * by - cy * bx;
  const double v         = ax * cy - ay * cx;
  const double w         = bx * ay - by * ax;
  if ((u < -tolerance || v < -tolerance || w < -tolerance) && (u > tolerance || v > tolerance || w > tolerance)) {
    return false;
  }

  const double determinant = u + v + w;
  if (determinant == 0.0 || !std::isfinite(determinant)) return false;

  const double az                 = prec.Sz * a[prec.maxDimIdxZ];
  const double bz                 = prec.Sz * b[prec.maxDimIdxZ];
  const double cz                 = prec.Sz * c[prec.maxDimIdxZ];
  const double inverseDeterminant = 1.0 / determinant;
  hitDistance                    = (u * az + v * bz + w * cz) * inverseDeterminant;
  bar1                           = v * inverseDeterminant;
  bar2                           = w * inverseDeterminant;
  return std::isfinite(hitDistance) && std::isfinite(bar1) && std::isfinite(bar2);
}

// Typical BVH depth fits locally. Retain dynamic overflow for arbitrarily deep
// trees; each query owns its stack, including queries made from callbacks.
class ThicknessTraversalStack {
 public:
  bool empty() const { return mSize == 0; }
  void push_back(Node* node) {
    if (mSize < mLocal.size()) mLocal[mSize] = node;
    else mOverflow.push_back(node);
    ++mSize;
  }
  Node* back() const {
    return mSize <= mLocal.size() ? mLocal[mSize-1] : mOverflow.back();
  }
  void pop_back() {
    if (mSize > mLocal.size()) mOverflow.pop_back();
    --mSize;
  }
 private:
  std::array<Node*,64> mLocal;
  std::vector<Node*> mOverflow;
  std::size_t mSize = 0;
};

MeshIntersectionResult meshRayIntersect_(Mesh& m, Bvh& tS2, const Line& line, double rayStart, double rayEnd,
                                         const IntersectionPrecomputes& prec, bool closestIntersect,
                                         const FacePredicate& validFaces) {
  MeshIntersectionResult res;
  if (tS2.Empty()) return res;
  (void)m;
  (void)closestIntersect;

  auto& tree = tS2.Tree();
  Line ray(line.Origin(), line.Direction());
  ThicknessTraversalStack nodesStack;
  nodesStack.push_back(tree.pRoot);

  Face* faceId = nullptr;
  TriPointf triP;
  while (!nodesStack.empty()) {
    Node* node = nodesStack.back();
    nodesStack.pop_back();

    double boxEntry=0;
    if (!ThicknessGeometry::rayBoxEntry(node->box,ray,boxEntry)) continue;
    if (boxEntry > rayEnd) continue;

    if (!node->IsLeaf()) {
      nodesStack.push_back(node->children[1]);
      nodesStack.push_back(node->children[0]);
      continue;
    }

    for (auto iter = node->oBegin; iter != node->oEnd; ++iter) {
      Face* face = *iter;
      if (!face) continue;
      if (validFaces && !validFaces(face)) continue;

      double t = 0;
      double u = 0;
      double v = 0;
      if (SPIntersectionRayTriangle(line, face->cP(0), face->cP(1), face->cP(2), prec, t, u, v) &&
          t > rayStart && t <= rayEnd) {
        faceId = face;
        triP   = {u, v};
        rayEnd = t;
      }
    }
  }

  if (faceId) {
    res.proj.face         = faceId;
    res.proj.point        = line.P(rayEnd);

    res.mtp.bary          = triP;
    res.distanceAlongLine = rayEnd;
  }
  return res;
}

MeshIntersectionResult rayMeshIntersect(Mesh& meshPart, Bvh& tree, const Line& line, double rayStart,
                                        double rayEnd, const IntersectionPrecomputes* prec, bool closestIntersect,
                                        const FacePredicate& validFaces) {
  if (prec) {
    return meshRayIntersect_(meshPart, tree, line, rayStart, rayEnd, *prec, closestIntersect, validFaces);
  } else {
    const IntersectionPrecomputes precNew(line.Direction());
    return meshRayIntersect_(meshPart, tree, line, rayStart, rayEnd, precNew, closestIntersect, validFaces);
  }
}
MeshIntersectionResult rayInsideIntersect(Mesh& mesh, Bvh& tree, const MeshPoint& m,
                                          const Vector& rayDirection, double rayEnd,
                                          const FacePredicate& validFaces) {
  return rayMeshIntersect(mesh, tree, {m.pt, rayDirection}, 0.0, rayEnd, nullptr, true, validFaces);
}

enum class PointInsideResult { Inside, Outside, Undetermined };

PointInsideResult isInsideClosedMesh(Mesh& mesh, Bvh& tree, const Vector& point, double tolerance) {
  // A non-intersecting ball can also be outside the material (for example at
  // a flipped/folded source triangle). Odd/even crossings do not rely on face
  // orientation. The oblique ray avoids systematic grid-edge intersections.
  if (!mesh.bbox.IsIn(point)) return PointInsideResult::Outside;
  const Vector direction = Vector(0.723, 0.439, 0.535).normalized();
  const double rayEnd = 2.0 * mesh.bbox.Diag();
  double rayStart = 0.0;
  int crossings = 0;
  while (crossings < 128) {
    const auto hit = rayMeshIntersect(mesh, tree, {point, direction}, rayStart, rayEnd, nullptr, true, {});
    if (!hit) return crossings % 2 == 1 ? PointInsideResult::Inside : PointInsideResult::Outside;
    ++crossings;
    // Deduplicate shared-edge hits and guarantee progress in double arithmetic.
    rayStart = std::max(hit.distanceAlongLine + tolerance,
                        std::nextafter(hit.distanceAlongLine, std::numeric_limits<double>::infinity()));
  }
  return PointInsideResult::Undetermined;
}
enum class Processing : bool { Continue, Stop };

struct SearchBall {
  Vector center;
  double radiusSq = 0;
};

template <typename Callback>
void findTrisInBall(Mesh& m, Bvh& tree, SearchBall ball, const Callback& foundCallback,
                    const FacePredicate& validFaces) {
  if (tree.Empty()) return;
  (void)m;
  auto& aabbTree = tree.Tree();

  auto boxDistSq = [&](Node* node) {
    Vector pMin     = node->box.min;
    Vector pMax     = node->box.max;
    double res             = 0;
    const Vector& p = ball.center;
    for (int i = 0; i < 3; ++i) {
      if (p[i] < pMin[i])
        res += (p[i] - pMin[i]) * (p[i] - pMin[i]);
      else if (p[i] > pMax[i])
        res += (p[i] - pMax[i]) * (p[i] - pMax[i]);
    }
    return res;
  };

  ThicknessTraversalStack subtasks;
  auto addSubTask = [&](Node* node, double distanceSq) {
    if (distanceSq < ball.radiusSq) subtasks.push_back(node);
  };

  addSubTask(aabbTree.pRoot, boxDistSq(aabbTree.pRoot));

  while (!subtasks.empty()) {
    Node* node = subtasks.back();
    subtasks.pop_back();
    if (!(boxDistSq(node) < ball.radiusSq)) continue;

    if (node->IsLeaf()) {
      for (auto si = (node)->oBegin; si != (node)->oEnd; ++si) {
        Face* face = *(si);
        if (!face) continue;
        if (validFaces && !validFaces(face)) continue;

        const auto [projD, bary] = closestPointInTriangle(Vector(ball.center), Vector(face->cP(0)),
                                                          Vector(face->cP(1)), Vector(face->cP(2)));
        const Vector projection(projD);

        MeshProjectionResult candidate;
        candidate.proj.face  = face;
        candidate.proj.point = projection;

        candidate.mtp.bary   = bary;
        candidate.distSq     = (projection - ball.center).SquaredNorm();
        if (candidate.distSq < ball.radiusSq && foundCallback(candidate, ball) == Processing::Stop) {
          return;
        }
      }
      continue;
    }

    auto lDistSq = boxDistSq(node->children[0]);
    auto rDistSq = boxDistSq(node->children[1]);
    /// first go in the node located closer to ball's center (in case the ball will shrink and the
    /// other node will be away)
    if (lDistSq <= rDistSq) {
      addSubTask(node->children[1], rDistSq);
      addSubTask(node->children[0], lDistSq);
    } else {
      addSubTask(node->children[0], lDistSq);
      addSubTask(node->children[1], rDistSq);
    }
  }
}

InSphere findInSphereImpl(Mesh& mesh, Bvh& tree, const MeshPoint& m,
                          ArrayView<const Vector> rayDirections,
                          const InSphereSearchSettings& settings) {
  assert(settings.maxRadius > 0);
  assert(settings.maxIters > 0);
  assert(settings.minShrinkage > 0);
  assert(settings.minShrinkage < 1);

  // initial assessment - sphere with maximal radius
  InSphere res;
  res.center                    = m.pt + m.inDir * settings.maxRadius;
  res.radius                    = settings.maxRadius;
  res.oppositeTouchPoint.distSq = res.radius * res.radius;
  const double coordinateScale = std::max({std::abs(m.pt[0]), std::abs(m.pt[1]), std::abs(m.pt[2]),
                                           settings.maxRadius, 1.0});
  auto sphereTolerance = [&](double radius) {
    return 8.0 * std::numeric_limits<double>::epsilon() * coordinateScale +
            2.0 * (1.0 - settings.minShrinkage) * radius;
  };

  auto isNonIncidentFace = [&](Face* face) -> bool {
    if (!face) return false;
    if (m.notIncidentFaces && !m.notIncidentFaces(face)) return false;
    return true;
  };

  // The normal-angle requirement is only meaningful when selecting the
  // first surface hit along the sampled ray. Once that hit has established
  // an upper bound, lateral surfaces can also constrain the inscribed sphere.
  auto isSuitableRayOppositeFace = [&](Face* face, const Vector& rayDirection) -> bool {
    if (!isNonIncidentFace(face)) return false;
    if (settings.minOppositeNormalCos <= -1) return true;
    const auto& faceNormal = face->cN();
    if (!(faceNormal.SquaredNorm() > std::numeric_limits<double>::epsilon())) return false;
    return rayDirection.dot(faceNormal.normalized()) >= settings.minOppositeNormalCos;
  };

  auto isSuitableSphereConstraintFace = [&](Face* face) -> bool {
    if (!face) return false;
    if (m.sphereConstraintFaces && !m.sphereConstraintFaces(face)) return false;
    // Only the faces containing the source point are excluded here. A drafted
    // side wall may point slightly toward the source and still limit the ball.
    // The opposite-normal test belongs to ray initialization, not containment.
    return true;
  };

  // check candidate point, and if the sphere though it is smaller than current res, then replaces
  // res; returns true if res was updated
  auto processCandidate         = [&](const MeshProjectionResult& candidate) -> bool {
    ++res.sample.mCandidates;
    if (!isSuitableSphereConstraintFace(candidate.proj.face)) return false;
    // Coplanar source triangles can differ by a few double ULPs. Do not turn
    // that numerical contact into a tiny ball through division by a near-zero
    // normal displacement. Use the same tolerance as the final verification.
    if (res.radius - (candidate.proj.point - res.center).Norm() <= sphereTolerance(res.radius)) return false;
    const auto d  = candidate.proj.point - m.pt;
    const auto dn = m.inDir.dot(d);
    if (!(dn > 0) || !std::isfinite(dn)) return false;  // avoid circle inversion
    const auto x   = d.dot(d) / (2 * dn);
    const auto xSq = x * x;
    if (!(x > 0) || !std::isfinite(xSq)) return false;
    if (!(xSq < res.oppositeTouchPoint.distSq)) return false;  // no reduction of circle
    const auto candidateSphereCenter = m.pt + m.inDir * x;
    // Every boundary still limits the ball. Contact angle classifies the
    // final measurement; it must never remove a containment constraint.
    ++res.sample.mIterations;
    res.center                    = candidateSphereCenter;
    res.radius                    = x;
    res.oppositeTouchPoint        = candidate;
    res.oppositeTouchPoint.distSq = xSq;
    return true;
  };

  // Collect the reliable first hit from every sampled ray, convert each hit
  // to the radius of a sphere whose center remains on the true normal line,
  // and initialize from the median candidate. The expensive geometric sphere
  // reduction below is then performed only once.
  struct RayBound {
    double radius = 0.0;
    MeshProjectionResult candidate;
  };
  std::array<RayBound,1> singleRayBound;
  std::vector<RayBound> multipleRayBounds;
  if (rayDirections.size()>1) multipleRayBounds.resize(rayDirections.size());
  ArrayView<RayBound> rayStorage = rayDirections.size()>1
      ? ArrayView<RayBound>(multipleRayBounds) : ArrayView<RayBound>(singleRayBound);
  std::size_t rayBoundCount=0;
  constexpr double boxDiagonalToMinimumDimensionLimit = 1.7320508076;
  const double raySearchLength = 2.0 * res.radius * boxDiagonalToMinimumDimensionLimit;
  for (const auto& rayDirection : rayDirections) {
    const FacePredicate reliableOppositeFaces = [&](Face* face) {
      return isSuitableRayOppositeFace(face, rayDirection);
    };
    auto isec = rayInsideIntersect(mesh, tree, m, rayDirection, raySearchLength, reliableOppositeFaces);
    if (!isec) continue;

    MeshProjectionResult rayCandidate;
    rayCandidate.proj   = isec.proj;
    rayCandidate.mtp    = isec.mtp;
    rayCandidate.distSq = isec.distanceAlongLine * isec.distanceAlongLine;
    if (!isSuitableSphereConstraintFace(rayCandidate.proj.face)) continue;
    const auto d  = rayCandidate.proj.point - m.pt;
    const auto dn = m.inDir.dot(d);
    if (!(dn > 0.0) || !std::isfinite(dn)) continue;
    const auto rayBoundRadius = d.dot(d) / (2.0 * dn);
    if (!(rayBoundRadius > 0.0) || !std::isfinite(rayBoundRadius)) continue;
    rayStorage[rayBoundCount++]={rayBoundRadius, rayCandidate};
  }
  auto rayBounds=rayStorage.first(rayBoundCount);
  if (!rayBounds.empty()) {
    auto middle = rayBounds.begin() + static_cast<std::ptrdiff_t>(rayBounds.size() / 2);
    std::nth_element(rayBounds.begin(), middle, rayBounds.end(),
                     [](const RayBound& lhs, const RayBound& rhs) { return lhs.radius < rhs.radius; });
    const auto& medianBound = *middle;
    res.radius              = std::min(res.radius, medianBound.radius);
    res.center              = m.pt + m.inDir * res.radius;
    res.oppositeTouchPoint  = medianBound.candidate;
    res.oppositeTouchPoint.distSq = res.radius * res.radius;
  }

  // Every constraint is queried through the same BVH path. The former
  // vertex/edge shortcut dereferenced incomplete edge-face adjacency and
  // associated neighbor vertices with unrelated faces.
  res.referenceCenter = res.center;
  findTrisInBall(
      mesh, tree, SearchBall{res.center, res.oppositeTouchPoint.distSq},
      [&](MeshProjectionResult candidate, SearchBall& ball) {
        auto preRadius = res.radius;
        if (!processCandidate(candidate)) return Processing::Continue;
        if (res.radius <= preRadius * settings.minShrinkage) {
          // since triangle's closest point to old sphere center is not the closest point for updated
          // sphere center, repeat several times for the same triangle
          Vector a, b, c;
          a = candidate.proj.face->cP(0);
          b = candidate.proj.face->cP(1);
          c = candidate.proj.face->cP(2);
          //  start from 1 because 1 iteration was already done
          for (int subIt = 1; subIt < settings.maxIters; ++subIt) {
            preRadius = res.radius;
            const auto [projD, bary] =
                closestPointInTriangle(Vector(res.center), Vector(a), Vector(b), Vector(c));
            candidate.proj.point = Vector(projD);
            candidate.mtp.bary   = bary;
            candidate.distSq     = (candidate.proj.point - res.center).SquaredNorm();
            if (!processCandidate(candidate)) break;
            if (res.radius > preRadius * settings.minShrinkage) break;
          }
        }
        ball = SearchBall{res.center, res.oppositeTouchPoint.distSq};
        return Processing::Continue;
      },
      m.sphereConstraintFaces);

  // Verify against ALL triangles, including the source fan. An averaged
  // vertex normal can enter a faceted source sheet; such a ball is not valid.
  // Tolerance covers double coordinates and the shrinking convergence only,
  // and is independent of the user's minimum accepted wall thickness.
  const double tolerance = sphereTolerance(res.radius);
  const double interiorRadius = std::max(0.0, res.radius - tolerance);
  findTrisInBall(mesh, tree, SearchBall{res.center, interiorRadius * interiorRadius},
                 [&](const MeshProjectionResult& candidate, SearchBall&) {
                   res.contained = false;
                   const bool source = m.notIncidentFaces && !m.notIncidentFaces(candidate.proj.face);
                   const unsigned kind = source ? 1U : 2U;
                   res.rejection.reason |= kind;
                   const double depth = res.radius - std::sqrt(std::max(candidate.distSq, 0.0));
                   const int face = static_cast<int>(candidate.proj.face - mesh.face.data());
                   if (depth > res.rejection.depth || (depth == res.rejection.depth && face < res.rejection.face)) {
                     res.rejection.depth = depth;
                     res.rejection.radius = res.radius;
                     res.rejection.tolerance = tolerance;
                     res.rejection.face = face;
                     res.rejection.faceKind = kind;
                   }
                   // Visit every penetrating triangle: first-hit depth is
                   // traversal-dependent and can hide a deeper side contact.
                   return Processing::Continue;
                 }, {});
  // The last shrinking triangle need not be the opposite contact. A cylinder
  // can have an entire ring of active contacts. Inspect the complete contact
  // shell and choose the most separated contact before classifying the ball.
  if (res.contained && res.radius > 0) {
    double bestRadialCos = -std::numeric_limits<double>::infinity();
    const double shellRadius = res.radius + tolerance;
    findTrisInBall(mesh, tree, {res.center, shellRadius*shellRadius},
      [&](const MeshProjectionResult& candidate, SearchBall&) {
        if (std::abs(std::sqrt(candidate.distSq)-res.radius) > tolerance) return Processing::Continue;
        const double radialCos = m.inDir.dot(candidate.proj.point-res.center)/res.radius;
        if (radialCos > bestRadialCos) {
          bestRadialCos = radialCos;
          res.oppositeTouchPoint = candidate;
        }
        return Processing::Continue;
      }, m.notIncidentFaces);
  }
  // A capped ray bound is not proof of a second tangency. Re-evaluate at the final center.
  res.sample.mSource = {m.pt.X(), m.pt.Y(), m.pt.Z()};
  res.sample.mCenter = {res.center.X(), res.center.Y(), res.center.Z()};
  res.sample.mInward = {m.inDir.X(), m.inDir.Y(), m.inDir.Z()};
  res.sample.mRadius = res.radius;
  res.sample.mTolerance = tolerance;
  auto* sourceFace = m.sourceFace;
  if (sourceFace) res.sample.mSourceTriangle = static_cast<int>(sourceFace - mesh.face.data());
  if (auto* face = res.oppositeTouchPoint.proj.face) {
    const auto [q, bary] = closestPointInTriangle(res.center, face->cP(0), face->cP(1), face->cP(2));
    res.oppositeTouchPoint.proj.point = q;
    res.oppositeTouchPoint.mtp.bary = bary;
    res.oppositeTouchPoint.distSq = (q - res.center).SquaredNorm();
    res.sample.mContact = {q.X(), q.Y(), q.Z()};
    res.sample.mContactTriangle = static_cast<int>(face - mesh.face.data());
    res.sample.mContactResidual = std::abs(std::sqrt(res.oppositeTouchPoint.distSq) - res.radius);
    res.sample.mFeature = bary.inVertex() >= 0 ? ContactFeature::Vertex :
                         bary.onEdge() >= 0 ? ContactFeature::Edge : ContactFeature::Interior;
    res.converged = res.sample.mContactResidual <= tolerance;
    const double radialCos = res.radius > 0 ? m.inDir.dot(q-res.center)/res.radius : -1.0;
    res.wallContact = settings.minAngleCos <= -1.0 || radialCos >= settings.minAngleCos;

  }
  if (res.contained && res.converged && res.oppositeTouchPoint.proj.face) {
    const double rayTolerance = 8.0 * std::numeric_limits<double>::epsilon() * coordinateScale;
    const auto inside = isInsideClosedMesh(mesh, tree, res.center, rayTolerance);
    res.contained = inside == PointInsideResult::Inside;
    if (inside == PointInsideResult::Outside) res.rejection.reason |= 4U;
    if (inside == PointInsideResult::Undetermined) res.rejection.reason |= 8U;
  }

  res.sample.mConverged = res.converged && res.contained;
  return res;
}

InSphere findInSphere(Mesh& mesh, Bvh& tree, const MeshPoint& point,
                      ArrayView<const Vector> rayDirections,
                      const InSphereSearchSettings& settings) {
  InSphere result = findInSphereImpl(mesh, tree, point, rayDirections, settings);
  if (!settings.insideAndOutside) return result;

  MeshPoint oppositePoint = point;
  oppositePoint.inDir     = -oppositePoint.inDir;
  std::vector<Vector> oppositeRayDirections;
  oppositeRayDirections.reserve(rayDirections.size());
  for (const auto& direction : rayDirections) oppositeRayDirections.push_back(-direction);
  InSphere oppositeResult = findInSphereImpl(mesh, tree, oppositePoint, oppositeRayDirections, settings);
  if (oppositeResult.isValid() && (!result.isValid() || oppositeResult.radius < std::abs(result.radius))) {
    result        = oppositeResult;
    result.radius = -result.radius;
  }
  return result;
}

InSphere findInSphere(Mesh& mesh, Bvh& tree, const MeshPoint& point,
                      const InSphereSearchSettings& settings) {
  const std::array<Vector, 1> rayDirections{point.inDir};
  return findInSphere(mesh, tree, point, rayDirections, settings);
}

// A face centroid is a query location, not an immutable tangency constraint.
// Follow a short tessellation branch within a radius-scaled neighborhood. All
// proposals go through the same complete containment and contact certification.
InSphere refineFaceRepresentative(Mesh& mesh,Bvh& tree,const MeshPoint& query,
                                  InSphere initial,const InSphereSearchSettings& settings,
                                  int& attempts,double& referenceRadius) {
  auto separated=[](const InSphere& sphere) {
    if (!sphere.isValid() || !(sphere.radius>0)) return false;
    const auto& a=sphere.sample.mSource;const auto& b=sphere.sample.mContact;
    return (Vector(a[0],a[1],a[2])-sphere.center).dot(
             Vector(b[0],b[1],b[2])-sphere.center)<=0.5*sphere.radius*sphere.radius;
  };
  const auto reference=initial.referenceCenter;
  referenceRadius=(reference-query.pt).Norm();
  if (separated(initial)) return initial;
  if (!(referenceRadius>0) || !std::isfinite(referenceRadius)) {
    initial.wallContact=false;return initial;
  }
  const double neighborhood=0.5*referenceRadius;
  const double contactNeighborhood=2.5*referenceRadius;
  const double roundoff=64*std::numeric_limits<double>::epsilon()*
      std::max({query.pt.Norm(),referenceRadius,1.0});
  auto bounded=[&](const InSphere& sphere) {
    if (!sphere.contained || !sphere.converged || !(sphere.radius>0)) return false;
    const auto& s=sphere.sample.mSource;
    return (Vector(s[0],s[1],s[2])-query.pt).Norm()<=contactNeighborhood+roundoff &&
           (sphere.center-reference).Norm()<=neighborhood+roundoff &&
           (sphere.center-reference).Norm()<=sphere.radius+roundoff;
  };
  auto atCenter=[&](const Vector& seed) {
    ++attempts;
    InSphere result;
    if ((seed-reference).Norm()>neighborhood+roundoff) return result;
    MeshProjectionResult closest;
    double nearest=4*referenceRadius*referenceRadius;
    findTrisInBall(mesh,tree,{seed,nearest},
      [&](const MeshProjectionResult& candidate,SearchBall& ball) {
        if ((candidate.proj.point-query.pt).Norm()>contactNeighborhood+roundoff) return Processing::Continue;
        if (candidate.distSq<nearest) {
          closest=candidate;nearest=candidate.distSq;ball.radiusSq=nearest;
        }
        return Processing::Continue;
      },{});
    if (!closest.proj.face) return result;
    MeshPoint active;
    active.sourceFace=closest.proj.face;active.pt=closest.proj.point;
    active.inDir=seed-active.pt;
    if (!(active.inDir.SquaredNorm()>0)) return result;
    active.inDir.Normalize();
    active.triPoint={closest.mtp.bary};
    active.notIncidentFaces=[face=active.sourceFace](Face* other) {return other && other!=face;};
    active.sphereConstraintFaces=active.notIncidentFaces;
    auto localSettings=settings;
    localSettings.maxRadius=std::min(settings.maxRadius,1.5*referenceRadius);
    result=findInSphere(mesh,tree,active,localSettings);
    if (!bounded(result)) {result.contained=false;return result;}
    return result;
  };
  auto best=atCenter(reference);
  if (bounded(best)) {
    double step=0.125*best.radius;
    for (int iteration=0;iteration<96 && !separated(best);++iteration) {
      const auto& a=best.sample.mSource;const auto& b=best.sample.mContact;
      auto first=(Vector(a[0],a[1],a[2])-best.center).normalized();
      auto second=(Vector(b[0],b[1],b[2])-best.center).normalized();
      auto inward=-(first+second);
      if (!(inward.SquaredNorm()>0) || step<=std::max(roundoff,1e-10*best.radius)) break;
      inward.Normalize();
      auto candidate=atCenter(best.center+inward*step);
      if (bounded(candidate) &&
          (candidate.radius>best.radius*(1+1e-10) ||
           (separated(candidate) && candidate.radius>=best.radius*(1-1e-10)))) {
        best=std::move(candidate);step=std::min(step*1.5,0.125*best.radius);
      } else step*=0.5;
    }
    if (separated(best) && (!initial.isValid() || best.radius>=initial.radius*(1-1e-10))) {
      best.sourceRelocated=true;
      best.sourceDisplacement=(Vector(best.sample.mSource[0],best.sample.mSource[1],best.sample.mSource[2])-query.pt).Norm();
      best.referenceCenter=reference;
      return best;
    }
  }
  // A failed representative search is not permission to publish its short
  // same-side predecessor as a measured engineering thickness.
  initial.wallContact=false;
  return initial;
}


} // private namespace

bool WallThicknessCalculator::compute(const std::vector<std::array<double,3>>& vertices,
    const std::vector<std::array<int,3>>& faces,const WallThicknessOptions& options,
    WallThicknessResult& result,std::string& error) {
  using Clock=std::chrono::steady_clock;
  const auto start=Clock::now();result={};error.clear();
  if(vertices.empty()||faces.empty()||faces.size()>std::size_t(std::numeric_limits<int>::max())||
     vertices.size()>std::size_t(std::numeric_limits<int>::max())){
    error="wall thickness requires a nonempty indexed triangle mesh";return false;
  }
  if(options.Workers<1||options.Workers>128||!std::isfinite(options.MinimumThickness)||options.MinimumThickness<0||
     !std::isfinite(options.MinimumContactAngleDegrees)||options.MinimumContactAngleDegrees<0||options.MinimumContactAngleDegrees>180){
    error="invalid wall thickness options";return false;
  }
  try {
    Mesh mesh;
    mesh.face.resize(faces.size());
    for(const auto& vertex:vertices)for(double value:vertex)if(!std::isfinite(value)){
      error="wall thickness input has a nonfinite vertex";return false;
    }
    for(std::size_t i=0;i<faces.size();++i){
      auto& face=mesh.face[i];face.id=int(i);
      for(int k=0;k<3;++k){
        const int index=faces[i][k];if(index<0||std::size_t(index)>=vertices.size()){
          error="wall thickness input has an invalid face index";return false;
        }
        const auto& p=vertices[index];face.points[k]=Vector(p[0],p[1],p[2]);mesh.bbox.add(face.points[k]);
      }
      const auto cross=(face.points[1]-face.points[0])^(face.points[2]-face.points[0]);
      const double twiceArea=cross.Norm();
      if(!(twiceArea>0)||!std::isfinite(twiceArea)){
        error="wall thickness input has a degenerate face: "+std::to_string(i);return false;
      }
      face.normal=cross/twiceArea;
    }
    const double minDimension=std::min({mesh.bbox.DimX(),mesh.bbox.DimY(),mesh.bbox.DimZ()});
    if(!(minDimension>0)||!std::isfinite(minDimension)){
      error="wall thickness requires positive extent in all three dimensions";return false;
    }
    InSphereSearchSettings settings;
    settings.minAngleCos=-std::cos(options.MinimumContactAngleDegrees*3.14159265358979323846/180.0);
    settings.maxRadius=std::nextafter(.5*minDimension,std::numeric_limits<double>::max());
    const double minimum=options.MinimumThickness>0?options.MinimumThickness:minDimension*.0075;
    const double maximum=2*settings.maxRadius;
    Bvh tree;BuildThicknessAabbTree(mesh,tree);
    const std::size_t count=faces.size();
    result.Values.assign(count,std::numeric_limits<float>::quiet_NaN());
    result.Status.assign(count,ThicknessStatus::InvalidNormal);result.Samples.resize(count);
    std::vector<double> areas(count,0);
    const auto samplingStart=Clock::now();
    result.PreparationSeconds=std::chrono::duration<double>(samplingStart-start).count();
    auto sample=[&](std::size_t i){
      auto& f=mesh.face[i];
      areas[i]=((f.cP(1)-f.cP(0))^(f.cP(2)-f.cP(0))).Norm()*.5;
      if(!(f.N().SquaredNorm()>std::numeric_limits<float>::epsilon()))return;
      MeshPoint point;
      point.triPoint=MeshTriPoint{{1.0/3.0,1.0/3.0}};
      point.pt=f.cP(0)+((f.cP(1)-f.cP(0))+(f.cP(2)-f.cP(0)))/3.0;
      point.sourceFace=&f;point.inDir=-f.cN();
      point.notIncidentFaces=[&f](Face* other){return other && other!=&f;};
      point.sphereConstraintFaces=point.notIncidentFaces;
      auto sphere=findInSphere(mesh,tree,point,settings);
      int attempts=0;double referenceRadius=0;
      sphere=refineFaceRepresentative(mesh,tree,point,std::move(sphere),settings,attempts,referenceRadius);
      const double thickness=2*sphere.radius;auto& status=result.Status[i];
      if(!sphere.contained){status=ThicknessStatus::PenetratingSphere;return;}
      if(sphere.converged&&!sphere.wallContact){status=ThicknessStatus::LocalSurfaceContact;return;}
      if(!sphere.isValid()){status=ThicknessStatus::InvalidSphere;return;}
      if(!std::isfinite(thickness)||thickness>std::numeric_limits<float>::max()){
        status=ThicknessStatus::NonFinite;return;
      }
      if(thickness<=minimum){status=ThicknessStatus::BelowMinimum;return;}
      if(thickness>maximum*(1+1e-5)){status=ThicknessStatus::AboveMaximum;return;}
      result.Values[i]=float(thickness);
      status=sphere.sourceRelocated?ThicknessStatus::FeatureSample:ThicknessStatus::Measured;
      result.Samples[i]=sphere.sample;
    };
    // Bounded queue: one task per worker; each claims 32 faces at a time.
    std::atomic<std::size_t> next{0};
    const auto workerCount=std::min(std::size_t(options.Workers),(count+31)/32);
    RemeshWorkerPool pool(workerCount);
    std::vector<std::future<void>> pending;
    for(std::size_t worker=0;worker<workerCount;++worker)pending.push_back(pool.submit([&]{
      for(;;){const auto first=next.fetch_add(32);if(first>=count)break;
        for(auto i=first;i<std::min(first+32,count);++i)sample(i);
      }
    }));
    for(auto& future:pending)future.get();
    result.SamplingSeconds=std::chrono::duration<double>(Clock::now()-samplingStart).count();
    double totalArea=0,validArea=0,sum=0;
    for(std::size_t i=0;i<count;++i){
      totalArea+=areas[i];if(!std::isfinite(result.Values[i]))continue;
      const double value=result.Values[i];
      if(result.ValidFaces==0)result.Minimum=result.Maximum=value;
      else {result.Minimum=std::min(result.Minimum,value);result.Maximum=std::max(result.Maximum,value);}
      ++result.ValidFaces;validArea+=areas[i];sum+=value*areas[i];
    }
    result.ValidAreaFraction=totalArea>0?validArea/totalArea:0;
    result.AreaWeightedAverage=validArea>0?sum/validArea:0;
    return true;
  }catch(const std::exception& exception){
    result={};error=std::string("wall thickness: ")+exception.what();return false;
  }
}
} // namespace CadMesh

