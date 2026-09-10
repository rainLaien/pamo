#include "CadMesh/AnalyticPatchRemesher.h"
#include "CadMesh/CudaChartRemesher.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iostream>
#include <numeric>
#include <set>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace CadMesh {
namespace {

using EdgeKey = std::uint64_t;
constexpr double Pi = 3.1415926535897932384626433832795;

EdgeKey Key(int a, int b) {
  if (a > b) std::swap(a, b);
  return (std::uint64_t(std::uint32_t(a)) << 32) | std::uint32_t(b);
}

struct UV { double X = 0, Y = 0; };
UV Add2(UV a, UV b) { return {a.X + b.X, a.Y + b.Y}; }
UV Mul2(UV a, double s) { return {a.X * s, a.Y * s}; }
double Cross2(UV a, UV b, UV c) {
  return (b.X - a.X) * (c.Y - a.Y) -
         (b.Y - a.Y) * (c.X - a.X);
}

double Quality(const Point3 &a, const Point3 &b, const Point3 &c) {
  const Vec3 ab = Sub(ToVec(b), ToVec(a));
  const Vec3 bc = Sub(ToVec(c), ToVec(b));
  const Vec3 ca = Sub(ToVec(a), ToVec(c));
  const double denominator = Dot(ab, ab) + Dot(bc, bc) + Dot(ca, ca);
  return denominator > 0
             ? 2 * std::sqrt(3.0) * Norm(Cross(ab, ca)) / denominator
             : 0;
}

struct Frame {
  PatchSurfaceType Type = PatchSurfaceType::Unknown;
  Vec3 Origin{}, Axis{}, First{}, Second{};
  double Radius = 0, Major = 0, Minor = 0, Angle = 0;
  double UPeriod = 0, VPeriod = 0;

  bool parameter(const Point3 &point, UV &uv) const {
    const Vec3 offset = Sub(ToVec(point), Origin);
    if (Type == PatchSurfaceType::Plane) {
      uv = {Dot(offset, First), Dot(offset, Second)};
      return true;
    }
    if (Type == PatchSurfaceType::Sphere) {
      const double length = Norm(offset);
      if (length <= 1e-20) return false;
      const double theta = std::atan2(Dot(offset, Second), Dot(offset, First));
      const double latitude = std::asin(std::clamp(Dot(offset, Axis) / length, -1.0, 1.0));
      uv = {Radius * theta, Radius * latitude};
      return true;
    }
    const double height = Dot(offset, Axis);
    const Vec3 radial = Sub(offset, Mul(Axis, height));
    if (Norm(radial) <= 1e-20) return false;
    const double theta = std::atan2(Dot(radial, Second), Dot(radial, First));
    if (Type == PatchSurfaceType::Cylinder) {
      uv = {Radius * theta, height};
      return true;
    }
    if (Type == PatchSurfaceType::Cone) {
      uv = {Radius * theta, height / std::cos(Angle)};
      return true;
    }
    if (Type == PatchSurfaceType::Torus) {
      const double radialLength = Norm(radial);
      const double phi = std::atan2(height, radialLength - Major);
      uv = {Major * theta, Minor * phi};
      return true;
    }
    return false;
  }

  Point3 lift(UV uv) const {
    if (Type == PatchSurfaceType::Plane)
      return ToPoint(Add(Origin, Add(Mul(First, uv.X), Mul(Second, uv.Y))));
    if (Type == PatchSurfaceType::Sphere) {
      const double theta = uv.X / Radius, latitude = uv.Y / Radius;
      const Vec3 radial = Add(Mul(First, std::cos(theta)),
                              Mul(Second, std::sin(theta)));
      return ToPoint(Add(Origin, Add(Mul(radial, Radius * std::cos(latitude)),
                                     Mul(Axis, Radius * std::sin(latitude)))));
    }
    const double theta = uv.X / (Type == PatchSurfaceType::Torus ? Major : Radius);
    const Vec3 radial = Add(Mul(First, std::cos(theta)),
                            Mul(Second, std::sin(theta)));
    if (Type == PatchSurfaceType::Cylinder)
      return ToPoint(Add(Origin, Add(Mul(Axis, uv.Y), Mul(radial, Radius))));
    if (Type == PatchSurfaceType::Cone) {
      const double height = uv.Y * std::cos(Angle);
      return ToPoint(Add(Origin, Add(Mul(Axis, height),
                                     Mul(radial, height * std::tan(Angle)))));
    }
    const double phi = uv.Y / Minor;
    const Vec3 ring = Add(Origin, Mul(radial, Major + Minor * std::cos(phi)));
    return ToPoint(Add(ring, Mul(Axis, Minor * std::sin(phi))));
  }

  Vec3 normalAt(const Point3 &point) const {
    if (Type == PatchSurfaceType::Plane) return Axis;
    const Vec3 offset = Sub(ToVec(point), Origin);
    if (Type == PatchSurfaceType::Sphere) return Normalize(offset);
    const double height = Dot(offset, Axis);
    const Vec3 radial = Normalize(Sub(offset, Mul(Axis, height)));
    if (Type == PatchSurfaceType::Cylinder) return radial;
    if (Type == PatchSurfaceType::Cone)
      return Normalize(Sub(Mul(radial, std::cos(Angle)),
                           Mul(Axis, std::sin(Angle))));
    const Vec3 ringCenter = Add(Origin, Mul(radial, Major));
    return Normalize(Sub(ToVec(point), ringCenter));
  }
};

Vec3 StablePerpendicular(const Vec3 &axis) {
  const Vec3 reference = std::abs(axis[0]) < .8 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
  return Normalize(Cross(axis, reference));
}

bool MakeFrame(const MeshPatch &patch,
               const std::vector<Point3> &vertices,
               const std::vector<int> &boundary,
               Frame &frame) {
  frame.Type = patch.SurfaceType;
  if (const auto *p = std::get_if<PlaneParameters>(&patch.Parameters)) {
    frame.Origin = ToVec(p->Plane.Origin);
    frame.Axis = Normalize(ToVec(p->Plane.Normal));
  } else if (const auto *p = std::get_if<CylinderParameters>(&patch.Parameters)) {
    if (!(p->Radius > 0)) return false;
    frame.Origin = ToVec(p->Axis.Origin); frame.Axis = Normalize(ToVec(p->Axis.Direction));
    frame.Radius = p->Radius; frame.UPeriod = 2 * Pi * frame.Radius;
  } else if (const auto *p = std::get_if<ConeParameters>(&patch.Parameters)) {
    if (!(p->SemiAngle > 0 && p->SemiAngle < Pi / 2)) return false;
    frame.Origin = ToVec(p->Axis.Origin); frame.Axis = Normalize(ToVec(p->Axis.Direction));
    frame.Angle = p->SemiAngle;
    double radius = 0;
    for (int id : boundary) {
      const Vec3 d = Sub(ToVec(vertices[id]), frame.Origin);
      radius += Norm(Sub(d, Mul(frame.Axis, Dot(d, frame.Axis))));
    }
    frame.Radius = boundary.empty() ? 0 : radius / boundary.size();
    if (!(frame.Radius > 0)) return false;
    frame.UPeriod = 2 * Pi * frame.Radius;
  } else if (const auto *p = std::get_if<SphereParameters>(&patch.Parameters)) {
    if (!(p->Radius > 0)) return false;
    frame.Origin = ToVec(p->Center); frame.Axis = {0, 0, 1};
    frame.Radius = p->Radius; frame.UPeriod = 2 * Pi * frame.Radius;
  } else if (const auto *p = std::get_if<TorusParameters>(&patch.Parameters)) {
    if (!(p->MajorRadius > 0 && p->MinorRadius > 0)) return false;
    frame.Origin = ToVec(p->Axis.Origin); frame.Axis = Normalize(ToVec(p->Axis.Direction));
    frame.Major = p->MajorRadius; frame.Minor = p->MinorRadius;
    frame.UPeriod = 2 * Pi * frame.Major; frame.VPeriod = 2 * Pi * frame.Minor;
  } else return false;
  frame.First = StablePerpendicular(frame.Axis);
  frame.Second = Normalize(Cross(frame.Axis, frame.First));
  return true;
}

struct BoundaryData {
  std::vector<std::vector<int>> Loops;
  std::unordered_set<EdgeKey> Edges;
};

bool TraceBoundary(const std::unordered_map<EdgeKey, int> &counts,
                   BoundaryData &output) {
  std::unordered_map<int, std::vector<int>> adjacency;
  for (const auto &[edge, count] : counts) if (count == 1) {
    const int a = int(std::uint32_t(edge >> 32)), b = int(std::uint32_t(edge));
    output.Edges.insert(edge); adjacency[a].push_back(b); adjacency[b].push_back(a);
  }
  if (adjacency.empty()) return !counts.empty();
  for (const auto &[vertex, neighbors] : adjacency)
    if (neighbors.size() != 2) return false;
  std::unordered_set<EdgeKey> visited;
  for (const auto &[start, unused] : adjacency) {
    if (visited.count(Key(start, unused[0])) && visited.count(Key(start, unused[1]))) continue;
    std::vector<int> loop;
    int previous = -1, current = start;
    do {
      loop.push_back(current);
      const auto &neighbors = adjacency[current];
      const int next = neighbors[0] == previous ? neighbors[1] : neighbors[0];
      if (!visited.insert(Key(current, next)).second && next != start) return false;
      previous = current; current = next;
      if (loop.size() > adjacency.size() + 1) return false;
    } while (current != start);
    if (loop.size() < 3) return false;
    output.Loops.push_back(std::move(loop));
  }
  return !output.Loops.empty();
}

void BuildPatchData(
    const std::vector<std::array<int,3>> &faces,
    const std::vector<int> &labels, std::size_t patchCount,
    std::vector<std::vector<std::array<int,3>>> &patchFaces,
    std::vector<BoundaryData> &boundaries,
    std::vector<unsigned char> &validBoundaries) {
  patchFaces.resize(patchCount); boundaries.resize(patchCount);
  validBoundaries.assign(patchCount,0);
  std::vector<std::unordered_map<EdgeKey,int>> counts(patchCount);
  for(std::size_t faceId=0;faceId<faces.size();++faceId){
    const int patch=labels[faceId];
    if(patch<0||patch>=int(patchCount))continue;
    patchFaces[patch].push_back(faces[faceId]);
    for(int side=0;side<3;++side)
      ++counts[patch][Key(faces[faceId][side],faces[faceId][(side+1)%3])];
  }
  for(std::size_t patch=0;patch<patchCount;++patch)
    validBoundaries[patch]=TraceBoundary(counts[patch],boundaries[patch]);
}

void Unwrap(std::vector<UV> &uv, double uPeriod, double vPeriod) {
  for (std::size_t i = 1; i < uv.size(); ++i) {
    if (uPeriod > 0)
      uv[i].X += uPeriod * std::round((uv[i - 1].X - uv[i].X) / uPeriod);
    if (vPeriod > 0)
      uv[i].Y += vPeriod * std::round((uv[i - 1].Y - uv[i].Y) / vPeriod);
  }
}

double PolygonArea(const std::vector<UV> &points) {
  double area = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const UV a = points[i], b = points[(i + 1) % points.size()];
    area += a.X * b.Y - a.Y * b.X;
  }
  return area * .5;
}

bool PointInTriangle(UV p, UV a, UV b, UV c, double epsilon) {
  return Cross2(a, b, p) >= -epsilon && Cross2(b, c, p) >= -epsilon &&
         Cross2(c, a, p) >= -epsilon;
}

bool Same2(UV a, UV b, double epsilon) {
  return std::abs(a.X-b.X)<=epsilon && std::abs(a.Y-b.Y)<=epsilon;
}

bool EarClip(const std::vector<UV> &points,
             std::vector<std::array<int, 3>> &triangles,
             std::size_t maximumPredicateWork = 100000000) {
  if (points.size() < 3) return false;
  std::vector<int> polygon(points.size());
  std::iota(polygon.begin(), polygon.end(), 0);
  if (PolygonArea(points) < 0) std::reverse(polygon.begin(), polygon.end());
  double scale = 1;
  for (UV p : points) scale = std::max(scale, std::max(std::abs(p.X), std::abs(p.Y)));
  const double epsilon = scale * scale * 1e-13;
  // Degenerate or heavily sampled trims can make ear clipping cubic. Bound
  // predicate work so one chart cannot stall all remaining patches.
  std::size_t predicateWork = 0;
  std::size_t guard = 0;
  while (polygon.size() > 3 && guard++ < points.size() * points.size()) {
    bool clipped = false;
    for (std::size_t i = 0; i < polygon.size(); ++i) {
      if (++predicateWork > maximumPredicateWork) return false;
      const int a = polygon[(i + polygon.size() - 1) % polygon.size()];
      const int b = polygon[i], c = polygon[(i + 1) % polygon.size()];
      if (Cross2(points[a], points[b], points[c]) <= epsilon) continue;
      bool contains = false;
      for (int id : polygon) {
        if (++predicateWork > maximumPredicateWork) return false;
        if (id != a && id != b && id != c &&
          !Same2(points[id],points[a],epsilon) &&
          !Same2(points[id],points[b],epsilon) &&
          !Same2(points[id],points[c],epsilon) &&
          PointInTriangle(points[id], points[a], points[b], points[c], epsilon)) {
        contains = true; break;
      }
      }
      if (contains) continue;
      triangles.push_back({a, b, c});
      polygon.erase(polygon.begin() + i); clipped = true; break;
    }
    if (!clipped) return false;
  }
  if (polygon.size() != 3) return false;
  triangles.push_back({polygon[0], polygon[1], polygon[2]});
  return true;
}

struct Chart {
  Frame Surface;
  std::vector<UV> Points;
  std::vector<int> Aliases;
  std::vector<unsigned char> Fixed;
  std::vector<std::array<int, 3>> Faces;
  bool Closed = false;
};

bool MakeSimpleChart(const Frame &frame, const std::vector<int> &loop,
                     const std::vector<Point3> &vertices, Chart &chart) {
  chart.Surface = frame;
  for (int id : loop) {
    UV uv;
    if (!frame.parameter(vertices[id], uv)) return false;
    chart.Points.push_back(uv); chart.Aliases.push_back(id); chart.Fixed.push_back(1);
  }
  Unwrap(chart.Points, frame.UPeriod, frame.VPeriod);
  return EarClip(chart.Points, chart.Faces);
}

bool MakePlanarChartWithHoles(const Frame &frame,
                              const std::vector<std::vector<int>> &loops,
                              const std::vector<Point3> &vertices,
                              Chart &chart) {
  if (frame.Type != PatchSurfaceType::Plane || loops.size() < 2) return false;
  std::vector<std::vector<UV>> points(loops.size());
  for (std::size_t k=0;k<loops.size();++k)
    for(int id:loops[k]){UV uv;if(!frame.parameter(vertices[id],uv))return false;points[k].push_back(uv);}
  std::size_t outer=0;
  for(std::size_t k=1;k<points.size();++k)
    if(std::abs(PolygonArea(points[k]))>std::abs(PolygonArea(points[outer])))outer=k;
  std::vector<UV> polygon=points[outer];std::vector<int> aliases=loops[outer];
  if(PolygonArea(polygon)<0){std::reverse(polygon.begin(),polygon.end());std::reverse(aliases.begin(),aliases.end());}
  for(std::size_t k=0;k<points.size();++k)if(k!=outer){
    auto hole=points[k];auto holeAliases=loops[k];
    if(PolygonArea(hole)>0){std::reverse(hole.begin(),hole.end());std::reverse(holeAliases.begin(),holeAliases.end());}
    int h=0;for(int i=1;i<int(hole.size());++i)if(hole[i].X>hole[h].X)h=i;
    int o=0;double best=std::numeric_limits<double>::infinity();
    for(int i=0;i<int(polygon.size());++i){const double dx=polygon[i].X-hole[h].X,dy=polygon[i].Y-hole[h].Y,d=dx*dx+dy*dy;if(d<best){best=d;o=i;}}
    std::vector<UV> merged;std::vector<int> mergedAliases;
    auto append=[&](UV uv,int id){merged.push_back(uv);mergedAliases.push_back(id);};
    for(int i=0;i<=o;++i)append(polygon[i],aliases[i]);
    for(int j=0;j<int(hole.size());++j){int i=(h+j)%hole.size();append(hole[i],holeAliases[i]);}
    append(hole[h],holeAliases[h]);append(polygon[o],aliases[o]);
    for(int i=o+1;i<int(polygon.size());++i)append(polygon[i],aliases[i]);
    polygon.swap(merged);aliases.swap(mergedAliases);
  }
  chart.Surface=frame;chart.Points=std::move(polygon);chart.Aliases=std::move(aliases);
  chart.Fixed.assign(chart.Points.size(),1);
  return EarClip(chart.Points,chart.Faces);
}

bool MakeClosedChart(const Frame &frame, double target, Chart &chart) {
  chart.Surface = frame; chart.Closed = true;
  if (frame.Type == PatchSurfaceType::Torus) {
    const int nu = std::max(8, int(std::ceil(2 * Pi * (frame.Major + frame.Minor) /
                                                     std::max(target * .62, 1e-12))));
    const int nv = std::max(6, int(std::ceil(2 * Pi * frame.Minor /
                                             std::max(target * .62, 1e-12))));
    if (std::int64_t(nu) * nv > 1000000) return false;
    for (int v = 0; v < nv; ++v) for (int u = 0; u < nu; ++u) {
      chart.Points.push_back({frame.UPeriod * u / nu, frame.VPeriod * v / nv});
      chart.Aliases.push_back(-1); chart.Fixed.push_back(1);
    }
    auto id = [nu,nv](int u,int v){u=(u%nu+nu)%nu;v=(v%nv+nv)%nv;return v*nu+u;};
    for (int v = 0; v < nv; ++v) for (int u = 0; u < nu; ++u) {
      const int a=id(u,v),b=id(u+1,v),c=id(u+1,v+1),d=id(u,v+1);
      if ((u+v)&1) { chart.Faces.push_back({a,b,d}); chart.Faces.push_back({b,c,d}); }
      else { chart.Faces.push_back({a,b,c}); chart.Faces.push_back({a,c,d}); }
    }
    return true;
  }
  if (frame.Type != PatchSurfaceType::Sphere) return false;
  const double t = (1.0 + std::sqrt(5.0)) * .5;
  std::vector<Vec3> directions{{-1,t,0},{1,t,0},{-1,-t,0},{1,-t,0},
      {0,-1,t},{0,1,t},{0,-1,-t},{0,1,-t},{t,0,-1},{t,0,1},{-t,0,-1},{-t,0,1}};
  for (Vec3 &d : directions) d = Normalize(d);
  chart.Faces = {{0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},{1,5,9},
      {5,11,4},{11,10,2},{10,7,6},{7,1,8},{3,9,4},{3,4,2},{3,2,6},
      {3,6,8},{3,8,9},{4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}};
  for (int pass = 0; pass < 12; ++pass) {
    bool longEdge = false;
    for (const auto &f : chart.Faces) for (int side=0;side<3;++side) {
      const Point3 a=ToPoint(Add(frame.Origin,Mul(directions[f[side]],frame.Radius)));
      const Point3 b=ToPoint(Add(frame.Origin,Mul(directions[f[(side+1)%3]],frame.Radius)));
      longEdge = longEdge || Distance(a,b) > target * (1 + 1e-6);
    }
    if (!longEdge) break;
    if (pass == 11 || chart.Faces.size() > 500000) return false;
    std::unordered_map<EdgeKey,int> mids;
    auto midpoint=[&](int a,int b){const EdgeKey key=Key(a,b);auto found=mids.find(key);if(found!=mids.end())return found->second;int id=int(directions.size());directions.push_back(Normalize(Add(directions[a],directions[b])));mids[key]=id;return id;};
    std::vector<std::array<int,3>> next;next.reserve(chart.Faces.size()*4);
    for(const auto&f:chart.Faces){int ab=midpoint(f[0],f[1]),bc=midpoint(f[1],f[2]),ca=midpoint(f[2],f[0]);next.push_back({f[0],ab,ca});next.push_back({f[1],bc,ab});next.push_back({f[2],ca,bc});next.push_back({ab,bc,ca});}
    chart.Faces.swap(next);
  }
  for(const Vec3 &d:directions){UV uv;if(!frame.parameter(ToPoint(Add(frame.Origin,Mul(d,frame.Radius))),uv))return false;chart.Points.push_back(uv);chart.Aliases.push_back(-1);chart.Fixed.push_back(1);}
  return true;
}

bool MakePeriodicRingChart(const Frame &frame,
                           const std::vector<std::vector<int>> &loops,
                           const std::vector<Point3> &vertices, double target,
                           Chart &chart, std::size_t maximumPredicateWork=100000000) {
  if (loops.size() != 2 || !(frame.UPeriod > 0)) return false;
  std::vector<std::vector<int>> loopIds = loops;
  std::vector<std::vector<UV>> coordinates(2);
  for (int k = 0; k < 2; ++k) {
    for (int id : loopIds[k]) {
      UV uv; if (!frame.parameter(vertices[id], uv)) return false;
      coordinates[k].push_back(uv);
    }
    double winding = 0, minimumY = coordinates[k][0].Y,
           maximumY = coordinates[k][0].Y;
    for (std::size_t i=0;i<coordinates[k].size();++i) {
      winding += std::remainder(
          coordinates[k][(i+1)%coordinates[k].size()].X-coordinates[k][i].X,
          frame.UPeriod);
      minimumY=std::min(minimumY,coordinates[k][i].Y);
      maximumY=std::max(maximumY,coordinates[k][i].Y);
    }
    if (std::abs(std::abs(winding)-frame.UPeriod)>frame.UPeriod*.05 ||
        maximumY-minimumY>std::max(target*.1,frame.UPeriod*1e-8)) return false;
    if (winding < 0) {
      std::reverse(coordinates[k].begin(), coordinates[k].end());
      std::reverse(loopIds[k].begin(), loopIds[k].end());
    }
    Unwrap(coordinates[k], frame.UPeriod, 0);
  }
  int low = 0;
  const auto meanY = [&](int k) { double s = 0; for (UV p : coordinates[k]) s += p.Y; return s / coordinates[k].size(); };
  if (meanY(1) < meanY(0)) low = 1;
  const int high = 1 - low;
  auto lowerIds = loopIds[low], upperIds = loopIds[high];
  auto lower = coordinates[low], upper = coordinates[high];
  const double base = lower.front().X;
  for (UV &p : lower) p.X += frame.UPeriod * std::round((base - p.X) / frame.UPeriod);
  for (UV &p : upper) p.X += frame.UPeriod * std::round((base - p.X) / frame.UPeriod);
  Unwrap(lower, frame.UPeriod, 0); Unwrap(upper, frame.UPeriod, 0);
  if (lower.back().X < lower.front().X) { std::reverse(lower.begin(), lower.end()); std::reverse(lowerIds.begin(), lowerIds.end()); }
  if (upper.back().X < upper.front().X) { std::reverse(upper.begin(), upper.end()); std::reverse(upperIds.begin(), upperIds.end()); }
  const double start = lower.front().X;
  for (UV &p : upper) p.X += frame.UPeriod * std::round((start - p.X) / frame.UPeriod);
  Unwrap(upper, frame.UPeriod, 0);
  chart.Surface = frame;
  auto append = [&](UV uv, int alias) { chart.Points.push_back(uv); chart.Aliases.push_back(alias); chart.Fixed.push_back(1); };
  for (std::size_t i = 0; i < lower.size(); ++i) append(lower[i], lowerIds[i]);
  const UV seamLower{lower.front().X + frame.UPeriod, lower.front().Y};
  const UV seamUpper{upper.front().X + frame.UPeriod, upper.front().Y};
  append(seamLower, lowerIds.front());
  const double seamLength = Distance(frame.lift(seamLower), frame.lift(seamUpper));
  const int seamSegments = std::max(1, int(std::ceil(seamLength /
                                                     std::max(target * .9, 1e-12))));
  for (int i = 1; i < seamSegments; ++i) {
    const double t = double(i) / seamSegments;
    append(Add2(Mul2(seamLower, 1 - t), Mul2(seamUpper, t)), -1 - i);
  }
  append(seamUpper, upperIds.front());
  for (std::size_t i = upper.size(); i-- > 1;) append(upper[i], upperIds[i]);
  append(upper.front(), upperIds.front());
  for (int i = seamSegments - 1; i >= 1; --i) {
    const double t = double(i) / seamSegments;
    UV point = Add2(Mul2(seamLower, 1 - t), Mul2(seamUpper, t));
    point.X -= frame.UPeriod;
    append(point, -1 - i);
  }
  return EarClip(chart.Points, chart.Faces,maximumPredicateWork);
}

struct LocalEdge { int A=-1, B=-1; std::vector<int> Faces; };
std::vector<LocalEdge> Edges(const std::vector<std::array<int, 3>> &faces) {
  std::unordered_map<EdgeKey, int> lookup;
  std::vector<LocalEdge> edges;
  for (int face = 0; face < int(faces.size()); ++face) for (int side = 0; side < 3; ++side) {
    int a = faces[face][side], b = faces[face][(side + 1) % 3];
    const EdgeKey key = Key(a,b); auto found = lookup.find(key);
    if (found == lookup.end()) { lookup[key] = int(edges.size()); edges.push_back({std::min(a,b),std::max(a,b),{face}}); }
    else edges[found->second].Faces.push_back(face);
  }
  return edges;
}

void Append(std::vector<std::array<int,3>> &out,int a,int b,int c){out.push_back({a,b,c});}

void AppendBestQuad(const Chart &chart,
                    std::vector<std::array<int,3>> &out,
                    int q0,int q1,int q2,int q3) {
  const auto point=[&](int id){return chart.Surface.lift(chart.Points[id]);};
  const Point3 p0=point(q0),p1=point(q1),p2=point(q2),p3=point(q3);
  const double diagonal02=std::min(Quality(p0,p1,p2),Quality(p0,p2,p3));
  const double diagonal13=std::min(Quality(p0,p1,p3),Quality(p1,p2,p3));
  if(diagonal02>=diagonal13){Append(out,q0,q1,q2);Append(out,q0,q2,q3);}
  else{Append(out,q0,q1,q3);Append(out,q1,q2,q3);}
}

bool FlipChartEdges(Chart &chart, double target, int maximumPasses) {
  bool changedAny = false;
  for (int pass = 0; pass < maximumPasses; ++pass) {
    bool changed = false;
    const auto edges = Edges(chart.Faces);
    std::vector<unsigned char> usedFace(chart.Faces.size(), 0);
    std::unordered_set<EdgeKey> existing;
    for (const auto &edge : edges) existing.insert(Key(edge.A, edge.B));
    for (const auto &edge : edges) {
      // Only a one-sided edge is a chart boundary. An interior diagonal whose
      // endpoints happen to lie on the boundary must remain flippable.
      if (edge.Faces.size() != 2) continue;
      const int face0 = edge.Faces[0], face1 = edge.Faces[1];
      if (usedFace[face0] || usedFace[face1]) continue;
      int c = -1, d = -1;
      auto &f0 = chart.Faces[face0];
      auto &f1 = chart.Faces[face1];
      for (int x : f0) if (x != edge.A && x != edge.B) c = x;
      for (int x : f1) if (x != edge.A && x != edge.B) d = x;
      if (c < 0 || d < 0 || c == d || existing.count(Key(c, d))) continue;
      const Point3 a = chart.Surface.lift(chart.Points[edge.A]);
      const Point3 b = chart.Surface.lift(chart.Points[edge.B]);
      const Point3 cp = chart.Surface.lift(chart.Points[c]);
      const Point3 dp = chart.Surface.lift(chart.Points[d]);
      const double sideA=Cross2(chart.Points[c],chart.Points[d],chart.Points[edge.A]);
      const double sideB=Cross2(chart.Points[c],chart.Points[d],chart.Points[edge.B]);
      if(sideA*sideB>=-1e-20)continue;
      // During refinement an inherited diagonal can still exceed the final
      // target. Permit a shorter replacement so the next split pass sees a
      // balanced triangulation instead of recursively copying the same fan.
      const double oldLength = Distance(a, b);
      const double newLength = Distance(cp, dp);
      if (newLength > std::max(target, oldLength) * (1 + 1e-6)) continue;
      const double oldQ = std::min(Quality(a, b, cp), Quality(b, a, dp));
      const double newQ = std::min(Quality(cp, dp, a), Quality(dp, cp, b));
      if (newQ <= oldQ + 1e-8) continue;
      std::array<int,3> n0{c,d,edge.A}, n1{d,c,edge.B};
      if (Cross2(chart.Points[n0[0]],chart.Points[n0[1]],chart.Points[n0[2]]) < 0)
        std::swap(n0[1],n0[2]);
      if (Cross2(chart.Points[n1[0]],chart.Points[n1[1]],chart.Points[n1[2]]) < 0)
        std::swap(n1[1],n1[2]);
      f0 = n0; f1 = n1;
      usedFace[face0] = usedFace[face1] = 1;
      existing.insert(Key(c,d));
      changed = changedAny = true;
    }
    if (!changed) break;
  }
  return changedAny;
}

bool Refine(Chart &chart, double target) {
  for (int pass = 0; pass < 64; ++pass) {
    // Match the Python chart budget; retain the source patch on exhaustion.
    if (chart.Faces.size() > 200000 || chart.Points.size() > 200000) return false;
    // Ear clipping is only a topologically valid seed. Regularize it before
    // selecting the first midpoints, otherwise every child inherits the same
    // long fan diagonal and the defect multiplies on each refinement level.
    FlipChartEdges(chart, target, pass == 0 ? 32 : 8);
    const auto edges = Edges(chart.Faces);
    std::unordered_map<EdgeKey,int> mids;
    std::vector<double> lengths(edges.size());
    const std::size_t noEdge = edges.size();
    std::vector<std::size_t> longest(chart.Faces.size(), noEdge);
    for (std::size_t i = 0; i < edges.size(); ++i) {
      lengths[i] = Distance(chart.Surface.lift(chart.Points[edges[i].A]),
                            chart.Surface.lift(chart.Points[edges[i].B]));
      if (lengths[i] <= target * (1 + 1e-6)) continue;
      if (edges[i].Faces.size() == 1 && chart.Fixed[edges[i].A] && chart.Fixed[edges[i].B]) continue;
      for (int face : edges[i].Faces) {
        auto &best = longest[face];
        if (best == noEdge || lengths[i] > lengths[best]) best = i;
      }
    }
    for (std::size_t edgeId = 0; edgeId < edges.size(); ++edgeId) {
      const auto &edge = edges[edgeId];
      if (lengths[edgeId] <= target * (1 + 1e-6)) continue;
      // The shared boundary has already been sampled globally. It remains
      // immutable here even when curvature asks for a smaller interior size.
      if (edge.Faces.size() == 1 && chart.Fixed[edge.A] && chart.Fixed[edge.B]) continue;
      // Avoid copying a skinny parent into multiple similar skinny children.
      // The next pass flips and reselects edges on the updated triangulation.
      bool longestOnBothSides = true;
      for (int face : edge.Faces)
        longestOnBothSides = longestOnBothSides && longest[face] == edgeId;
      if (!longestOnBothSides) continue;
      const int id = int(chart.Points.size());
      chart.Points.push_back(Mul2(Add2(chart.Points[edge.A],chart.Points[edge.B]),.5));
      chart.Aliases.push_back(-1); chart.Fixed.push_back(0); mids[Key(edge.A,edge.B)] = id;
    }
    if (mids.empty()) return true;
    std::vector<std::array<int,3>> output; output.reserve(chart.Faces.size()*2);
    for (const auto &f : chart.Faces) {
      auto mid=[&](int a,int b){auto it=mids.find(Key(a,b));return it==mids.end()?-1:it->second;};
      int ab=mid(f[0],f[1]),bc=mid(f[1],f[2]),ca=mid(f[2],f[0]);
      int mask=(ab>=0?1:0)|(bc>=0?2:0)|(ca>=0?4:0);
      int a=f[0],b=f[1],c=f[2];
      switch(mask){
      case 0:Append(output,a,b,c);break;
      case 1:Append(output,a,ab,c);Append(output,ab,b,c);break;
      case 2:Append(output,b,bc,a);Append(output,bc,c,a);break;
      case 4:Append(output,c,ca,b);Append(output,ca,a,b);break;
      case 3:Append(output,b,bc,ab);AppendBestQuad(chart,output,a,ab,bc,c);break;
      case 6:Append(output,c,ca,bc);AppendBestQuad(chart,output,b,bc,ca,a);break;
      case 5:Append(output,a,ab,ca);AppendBestQuad(chart,output,c,ca,ab,b);break;
      default:Append(output,a,ab,ca);Append(output,ab,b,bc);Append(output,ca,bc,c);Append(output,ab,bc,ca);break;
      }
    }
    if (output.size() > 200000) return false;
    chart.Faces.swap(output);
  }
  return false;
}

void Improve(Chart &chart, double target) {
  FlipChartEdges(chart, target, 32);
  for(int iteration=0;iteration<8;++iteration){
    std::vector<UV>sums(chart.Points.size());std::vector<int>counts(chart.Points.size());
    std::vector<std::vector<int>> incident(chart.Points.size());
    for(int faceId=0;faceId<int(chart.Faces.size());++faceId)
      for(int v:chart.Faces[faceId])incident[v].push_back(faceId);
    for(const auto&e:Edges(chart.Faces)){sums[e.A]=Add2(sums[e.A],chart.Points[e.B]);++counts[e.A];sums[e.B]=Add2(sums[e.B],chart.Points[e.A]);++counts[e.B];}
    bool moved=false;
    for(int v=0;v<int(chart.Points.size());++v)if(!chart.Fixed[v]&&counts[v]){
      UV candidate=Add2(Mul2(chart.Points[v],.5),Mul2(sums[v],.5/counts[v]));
      double before=1,after=1;
      bool lengthValid=true;
      for(int faceId:incident[v]){const auto&f=chart.Faces[faceId];
        const auto uv = [&](int id) { return id == v ? candidate : chart.Points[id]; };
        const double oldArea = Cross2(chart.Points[f[0]], chart.Points[f[1]], chart.Points[f[2]]);
        const double newArea = Cross2(uv(f[0]), uv(f[1]), uv(f[2]));
        if (oldArea * newArea <= 0) { lengthValid = false; break; }
        before=std::min(before,Quality(chart.Surface.lift(chart.Points[f[0]]),chart.Surface.lift(chart.Points[f[1]]),chart.Surface.lift(chart.Points[f[2]])));
        const auto point=[&](int id){return chart.Surface.lift(id==v?candidate:chart.Points[id]);};
        after=std::min(after,Quality(point(f[0]),point(f[1]),point(f[2])));
        lengthValid=lengthValid&&Distance(point(f[0]),point(f[1]))<=target*(1+1e-6)&&Distance(point(f[1]),point(f[2]))<=target*(1+1e-6)&&Distance(point(f[2]),point(f[0]))<=target*(1+1e-6);
      }
      if(lengthValid&&after>before+1e-8){chart.Points[v]=candidate;moved=true;}
    }
    if(moved)FlipChartEdges(chart,target,4);
    if(!moved)break;
  }
}

#include "PlanarDomain.h"

bool ValidatePatch(const std::vector<Point3>&vertices,
                   const std::vector<std::array<int,3>>&faces,
                   const std::unordered_set<EdgeKey>&expectedBoundary,
                   double target) {
  std::unordered_map<EdgeKey,int> counts;
  std::set<std::array<int,3>> uniqueFaces;
  for(const auto&face:faces){
    auto canonical=face;std::sort(canonical.begin(),canonical.end());
    if(canonical[0]==canonical[1]||canonical[1]==canonical[2]||
       !uniqueFaces.insert(canonical).second)return false;
    const Vec3 cross=Cross(Sub(ToVec(vertices[face[1]]),ToVec(vertices[face[0]])),
                           Sub(ToVec(vertices[face[2]]),ToVec(vertices[face[0]])));
    const double areaSquared=Dot(cross,cross);
    if(!std::isfinite(areaSquared)||areaSquared<=0)return false;
    for(int side=0;side<3;++side){
      if(Distance(vertices[face[side]],vertices[face[(side+1)%3]])>
         target*(1+1e-6))return false;
      if(++counts[Key(face[side],face[(side+1)%3])]>2)return false;
    }
  }
  std::unordered_set<EdgeKey> actualBoundary;
  for(const auto&entry:counts)if(entry.second==1)actualBoundary.insert(entry.first);
  return actualBoundary==expectedBoundary;
}

struct PreparedChart {
  enum Status { Skip, Collision, Deviation, Topology, Parameterization, GpuPending, Ready } State = Skip;
  Chart Value;
  double Target = 0;
  bool UsedCuda = false;
  bool SeededPlanar = false;
  bool SeededCylinder = false;
  ChartBuildDiagnostics CylinderBuild;
  double MakeSeconds = 0, RefineSeconds = 0, ImproveSeconds = 0;
};

// A fixed worker pool. Queued tasks contain only captures; each completed
// chart is retained in its future until the caller commits that patch.
// Workers only construct local charts from an immutable vertex snapshot.
class ChartWorkers {
  std::mutex Mutex;
  std::condition_variable Wake;
  std::deque<std::packaged_task<PreparedChart()>> Tasks;
  std::vector<std::thread> Threads;
  bool Stop = false;
  void finish() {
    { std::lock_guard<std::mutex> lock(Mutex); Stop = true; }
    Wake.notify_all();
    for (auto &thread : Threads) if (thread.joinable()) thread.join();
  }
public:
  explicit ChartWorkers(int count) {
    try {
      for (int i = 0; i < count; ++i) Threads.emplace_back([this] {
        for (;;) {
          std::packaged_task<PreparedChart()> task;
          {
            std::unique_lock<std::mutex> lock(Mutex);
            Wake.wait(lock, [&] { return Stop || !Tasks.empty(); });
            if (Tasks.empty()) return;
            task = std::move(Tasks.front()); Tasks.pop_front();
          }
          task(); // Exceptions are delivered to the owner through the future.
        }
      });
    } catch (...) { finish(); throw; }
  }
  ~ChartWorkers() { finish(); }
  std::future<PreparedChart> submit(std::function<PreparedChart()> work) {
    std::packaged_task<PreparedChart()> task(std::move(work));
    auto result = task.get_future();
    { std::lock_guard<std::mutex> lock(Mutex); Tasks.push_back(std::move(task)); }
    Wake.notify_one();
    return result;
  }
};
} // namespace

bool RebuildAnalyticPatches(
    std::vector<Point3> &vertices, std::vector<std::array<int, 3>> &faces,
    std::vector<int> &labels, const std::vector<MeshPatch> &patches,
    double targetEdgeLength, double maximumDeviation,
    double maximumNormalDeviationDegrees, double targetMeanQuality,
    const std::vector<unsigned char> &excludedPatches,
    std::vector<unsigned char> &successfullyRebuilt,
    AnalyticPatchRemeshReport &report, bool verbose) {
  using Clock=std::chrono::steady_clock;
  successfullyRebuilt.assign(patches.size(), 0); report = {};
  report.PatchReasons.assign(patches.size(),PatchRemeshReason::Unknown);
  (void)targetMeanQuality; // Patch acceptance no longer uses quality thresholds.
  std::vector<std::vector<std::array<int,3>>> patchFaces;
  std::vector<BoundaryData> boundaries;
  std::vector<unsigned char> validBoundaries;
  auto phaseStart=Clock::now();
  if (verbose) std::clog << "[CadMesh] analytic rebuild: indexing patch boundaries" << std::endl;
  BuildPatchData(faces,labels,patches.size(),patchFaces,boundaries,
                 validBoundaries);
  report.IndexSeconds=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  std::vector<std::vector<std::array<int,3>>> replacements(patches.size());
  // Main-thread commits append/reallocate vertices. Workers must never read
  // that vector concurrently, even though existing coordinates stay unchanged.
  const auto sourceVertices = vertices;
  const char *planeSetting=std::getenv("CADMESH_REMESH_SIMPLE_PLANES_ONLY");
  const bool simplePlanesOnly=planeSetting && std::string(planeSetting)=="1";
  const char *cylinderSetting=std::getenv("CADMESH_REMESH_CYLINDERS_ONLY");
  const bool cylindersOnly=cylinderSetting && std::string(cylinderSetting)=="1";
  const char *coneSetting=std::getenv("CADMESH_REMESH_CONES_ONLY");
  const bool conesOnly=coneSetting && std::string(coneSetting)=="1";
  const char *otherSetting=std::getenv("CADMESH_REMESH_OTHER_FEATURES_ONLY");
  const bool otherFeaturesOnly=otherSetting && std::string(otherSetting)=="1";
  if(int(simplePlanesOnly)+int(cylindersOnly)+int(conesOnly)+int(otherFeaturesOnly)>1)throw std::invalid_argument("select only one surface-only mode");
  const auto selectedSurface=[&](int id){
    if(otherFeaturesOnly)return patches[id].SurfaceType!=PatchSurfaceType::Plane && patches[id].SurfaceType!=PatchSurfaceType::Cylinder;
    return patches[id].SurfaceType==(conesOnly?PatchSurfaceType::Cone:cylindersOnly?PatchSurfaceType::Cylinder:PatchSurfaceType::Plane);
  };
  const char *cudaSetting = std::getenv("CADMESH_REMESH_CHART_CUDA");
  const bool useChartCuda = !cudaSetting || std::string(cudaSetting) != "0";
  const auto prepare = [&](int patchId) {
    PreparedChart result;
    const auto &patch = patches[patchId];
    if((simplePlanesOnly||cylindersOnly||conesOnly||otherFeaturesOnly) && !selectedSurface(patchId))return result;
    if (patch.ProjectionTarget != PatchProjectionTarget::AnalyticSurface ||
        patch.SurfaceType == PatchSurfaceType::Freeform ||
        patch.SurfaceType == PatchSurfaceType::Unknown) return result;
    if (patchId < int(excludedPatches.size()) && excludedPatches[patchId]) {
      result.State = PreparedChart::Collision; return result;
    }
    const double tolerance = std::max(1.0, targetEdgeLength) * 1e-12;
    if (patch.MaxSampledSurfaceDeviation > maximumDeviation + tolerance) {
      result.State = PreparedChart::Deviation; return result;
    }
    const BoundaryData &boundary=boundaries[patchId];
    if (!validBoundaries[patchId]) {
      result.State = PreparedChart::Topology; return result;
    }
    std::vector<int> allBoundary;
    for (const auto &loop : boundary.Loops) allBoundary.insert(allBoundary.end(),loop.begin(),loop.end());
    Frame frame;
    if (!MakeFrame(patch, sourceVertices, allBoundary, frame)) {
      result.State = PreparedChart::Parameterization; return result;
    }
    double chartTarget = targetEdgeLength;
    double curvatureRadius = 0;
    if (frame.Type == PatchSurfaceType::Cylinder || frame.Type == PatchSurfaceType::Cone ||
        frame.Type == PatchSurfaceType::Sphere) curvatureRadius = frame.Radius;
    else if (frame.Type == PatchSurfaceType::Torus) curvatureRadius = frame.Minor;
    if (curvatureRadius > 0 && maximumNormalDeviationDegrees > 0 &&
        maximumNormalDeviationDegrees < 180)
      chartTarget = std::min(chartTarget, 1.8 * curvatureRadius *
          std::sin(maximumNormalDeviationDegrees * Pi / 360.0));
    Chart &chart = result.Value;
    auto start = Clock::now();
    if(frame.Type==PatchSurfaceType::Plane){
      result.SeededPlanar=true;
      const bool made=MakeSeededPlanarChart(frame,boundary.Loops,patchFaces[patchId],sourceVertices,chartTarget,chart);
      result.MakeSeconds=std::chrono::duration<double>(Clock::now()-start).count();
      result.Target=chartTarget;
      result.State=made?PreparedChart::Ready:PreparedChart::Parameterization;
      if(!made)result.Value={};
      return result;
    }
    if(frame.Type==PatchSurfaceType::Cylinder){
      result.SeededCylinder=true;
      const bool made=MakeSeededCylinderChart(frame,boundary.Loops,sourceVertices,chartTarget,chart,
                                             targetEdgeLength,result.CylinderBuild);
      result.CylinderBuild.Points=chart.Points.size();result.CylinderBuild.Faces=chart.Faces.size();
      if(!made && result.CylinderBuild.Failure.empty())result.CylinderBuild.Failure=result.CylinderBuild.Stage+"_failed";
      result.MakeSeconds=std::chrono::duration<double>(Clock::now()-start).count();
      result.Target=chartTarget;
      result.State=made?PreparedChart::Ready:PreparedChart::Parameterization;
      if(!made)result.Value={};
      return result;
    }
    bool made = boundary.Loops.empty()
                    ? MakeClosedChart(frame,chartTarget,chart)
                    : boundary.Loops.size()==1
                          ? MakeSimpleChart(frame,boundary.Loops[0],sourceVertices,chart)
                          : frame.Type==PatchSurfaceType::Plane
                                ? MakePlanarChartWithHoles(frame,boundary.Loops,
                                                           sourceVertices,chart)
                                : MakePeriodicRingChart(frame,boundary.Loops,
                                                        sourceVertices,chartTarget,chart);
    result.MakeSeconds = std::chrono::duration<double>(Clock::now()-start).count();
    result.Target = chartTarget;
    // A single triangle has only fixed boundary edges and no movable vertex.
    // Refine/Improve cannot change it; avoid a GPU launch (or CPU iterations).
    if(made && boundary.Loops.size()==1 && chart.Faces.size()==1 && chart.Points.size()==3){
      result.State=PreparedChart::Ready;return result;
    }
    if (made && useChartCuda && boundary.Loops.size()==1 &&
        frame.Type==PatchSurfaceType::Plane && frame.UPeriod==0 && frame.VPeriod==0 &&
        chart.Points.size()<=200000 && chart.Faces.size()<=200000) {
      result.State = PreparedChart::GpuPending;
      return result;
    }
    start = Clock::now();
    const bool refined = made && (chart.Closed || Refine(chart,chartTarget));
    result.RefineSeconds = std::chrono::duration<double>(Clock::now()-start).count();
    if (!refined) {
      result.State = PreparedChart::Parameterization;
      result.Value = {}; return result;
    }
    start = Clock::now();
    if (!chart.Closed) Improve(chart,chartTarget);
    result.ImproveSeconds = std::chrono::duration<double>(Clock::now()-start).count();
    result.State = PreparedChart::Ready;
    return result;
  };
  int workerCount = int(std::min(4u, std::max(1u, std::thread::hardware_concurrency())));
  if (const char *setting = std::getenv("CADMESH_REMESH_THREADS")) {
    char *end = nullptr;
    const long value = std::strtol(setting, &end, 10);
    if (end == setting || *end || value < 1 || value > 16)
      throw std::invalid_argument("CADMESH_REMESH_THREADS must be 1..16");
    workerCount = int(value);
  }
  const auto rebuildStart = Clock::now();
  struct WorkItem { int Id=0,Type=0,Topology=0,Bucket=0; double Cost=0; };
  std::vector<WorkItem> order;
  order.reserve(patches.size());
  for(int id=0;id<int(patches.size());++id){
    if((simplePlanesOnly||cylindersOnly||conesOnly||otherFeaturesOnly) && !selectedSurface(id)){
      order.push_back({id,int(patches[id].SurfaceType),0,0,1.0});continue;
    }
    std::size_t boundaryCount=0;
    for(const auto &loop:boundaries[id].Loops)boundaryCount+=loop.size();
    double area=0;
    for(const auto &f:patchFaces[id]){
      const Vec3 a=ToVec(sourceVertices[f[0]]),b=ToVec(sourceVertices[f[1]]),c=ToVec(sourceVertices[f[2]]);
      area+=.5*Norm(Cross(Sub(b,a),Sub(c,a)));
    }
    const int type=int(patches[id].SurfaceType);
    const int topology=boundaries[id].Loops.empty()?2:boundaries[id].Loops.size()==1?0:1;
    double target=targetEdgeLength;
    double radius=0;
    if(const auto *p=std::get_if<CylinderParameters>(&patches[id].Parameters))radius=p->Radius;
    else if(const auto *p=std::get_if<SphereParameters>(&patches[id].Parameters))radius=p->Radius;
    else if(const auto *p=std::get_if<TorusParameters>(&patches[id].Parameters))radius=p->MinorRadius;
    if(radius>0 && maximumNormalDeviationDegrees>0 && maximumNormalDeviationDegrees<180)
      target=std::min(target,1.8*radius*std::sin(maximumNormalDeviationDegrees*Pi/360));
    double estimate=area/std::max(target*target,1e-24)+double(boundaryCount);
    if(!std::isfinite(estimate))estimate=1e12;
    estimate=std::clamp(estimate,1.0,1e12);
    const int bucket=int(std::log2(estimate));
    // Large legacy curved/holed jobs tend to dominate the tail. The factor
    // only changes scheduling, never geometry or acceptance tolerances.
    const double factor=patches[id].SurfaceType==PatchSurfaceType::Plane?(topology==0?1.0:2.0):4.0;
    order.push_back({id,type,topology,bucket,estimate*factor});
  }
  // Classify by surface/topology and powers-of-two size bands. Prioritize
  // expensive jobs globally so one type cannot hide a long job until the end.
  std::stable_sort(order.begin(),order.end(),[](const WorkItem&a,const WorkItem&b){
    const int bandA=int(std::log2(a.Cost)),bandB=int(std::log2(b.Cost));
    if(bandA!=bandB)return bandA>bandB;
    if(a.Type!=b.Type)return a.Type<b.Type;
    if(a.Topology!=b.Topology)return a.Topology<b.Topology;
    if(a.Bucket!=b.Bucket)return a.Bucket>b.Bucket;
    if(a.Cost!=b.Cost)return a.Cost>b.Cost;
    return a.Id<b.Id;
  });
  std::atomic<std::size_t> completed{0};
  std::mutex progressMutex;
  std::unordered_map<int,Clock::time_point> running;
  std::vector<double> taskSeconds(patches.size(),0);
  // These shared objects precede the pool so its destructor joins workers
  // before any captured state is destroyed, including exceptional exits.
  ChartWorkers workers(workerCount);
  std::vector<std::future<PreparedChart>> pending(patches.size());
  if(verbose)std::clog << "[CadMesh] analytic scheduler: workers=" << workerCount
      << ", tasks=" << order.size() << ", type/topology/size buckets, largest first"
      << "; dynamic work queue, results keyed by patch ID, ordered commits" << std::endl;
  if(verbose && (simplePlanesOnly||cylindersOnly||conesOnly||otherFeaturesOnly)){
    std::size_t selected=0;for(int id=0;id<int(patches.size());++id)if(selectedSurface(id))++selected;
    std::clog << (otherFeaturesOnly?"[CadMesh] other-features-only: selected=":conesOnly?"[CadMesh] cones-only (legacy chart refinement): selected=":cylindersOnly?"[CadMesh] cylinders-only: selected=":"[CadMesh] planes-only (including holes): selected=") << selected
        << ", skipped=" << patches.size()-selected << "; other patch interiors retained" << std::endl;
  }
  for(const auto &item:order){const int id=item.Id;
    pending[id]=workers.submit([&,id]{
      const auto start=Clock::now();
      {std::lock_guard<std::mutex> lock(progressMutex);running[id]=start;}
      const auto finish=[&]{
        {std::lock_guard<std::mutex> lock(progressMutex);running.erase(id);
          taskSeconds[id]=std::chrono::duration<double>(Clock::now()-start).count();}
        ++completed;
      };
      try{auto result=prepare(id);finish();return result;}
      catch(...){finish();throw;}
    });
  }
  double makeSeconds = 0, refineSeconds = 0, improveSeconds = 0;
  double waitSeconds=0,detailLogSeconds=0,liftSeconds=0,validationSeconds=0;
  double cudaMilliseconds=0;
  std::size_t cudaBatches=0,cudaSubmitted=0,cudaCompleted=0,cudaAccepted=0,cudaFallback=0;
  std::size_t seededPlanar=0,seededPlanarFailed=0,seededPlanarAccepted=0;
  std::size_t seededCylinder=0,seededCylinderFailed=0,seededCylinderAccepted=0;
  std::unordered_map<std::string,std::pair<std::size_t,double>> cylinderFailures;
  bool cudaDisabled=false;
  std::deque<PreparedChart> ready;
  if(verbose)std::clog << "[CadMesh] analytic chart backend: "
      << "shared boundaries + interior lattice + constrained Delaunay (planes and unwrapped cylinders); legacy other curved surfaces" << std::endl;
  for (int patchId = 0; patchId < int(patches.size()); ++patchId) {
    if (verbose && patchId % 64 == 0)
      std::clog << "[CadMesh] analytic progress: " << patchId << '/' << patches.size()
                << ", computed=" << completed.load()
                << ", rebuilt=" << report.Rebuilt << ", fallback=" << report.Fallback
                << ", elapsed_s=" << std::chrono::duration<double>(Clock::now()-rebuildStart).count() << std::endl;
    if(ready.empty()) {
      const int count=std::min(16,int(patches.size())-patchId);
      std::vector<CudaPlanarChart> gpu;
      std::vector<int> gpuIndices;
      for(int i=0;i<count;++i){
        const auto waitStart=Clock::now();
        while(pending[patchId+i].wait_for(std::chrono::seconds(5))!=std::future_status::ready)
          if(verbose){
            std::lock_guard<std::mutex> lock(progressMutex);
            std::clog << "[CadMesh] analytic commit waiting: patch=" << patchId+i+1
                << ", computed=" << completed.load() << '/' << patches.size() << ", running=" << running.size();
            for(const auto &task:running)std::clog << " [patch=" << task.first+1
                << ", seconds=" << std::chrono::duration<double>(Clock::now()-task.second).count() << ']';
            std::clog << std::endl;
          }
        ready.push_back(pending[patchId+i].get());
        waitSeconds+=std::chrono::duration<double>(Clock::now()-waitStart).count();
        auto &r=ready.back();
        if(r.State==PreparedChart::GpuPending&&!cudaDisabled){
          CudaPlanarChart input;input.Target=r.Target;input.Faces=r.Value.Faces;
          for(auto uv:r.Value.Points)input.Points.push_back({uv.X,uv.Y});
          gpu.push_back(std::move(input));gpuIndices.push_back(i);
        }
      }
      if(!gpu.empty()){
        ++cudaBatches;cudaSubmitted+=gpu.size();
        if(verbose)std::clog << "[CadMesh] analytic CUDA chart batch: first_patch=" << patchId+1
                            << ", jobs=" << gpu.size() << std::endl;
        std::string error;
        bool ran=RebuildPlanarChartsCuda(gpu,cudaMilliseconds,error);
        // Retry capacity exhaustion on the GPU from the original input, with
        // geometrically increasing bounds. Never return partial GPU geometry.
        for(int retry=0;ran&&retry<10;++retry){
          std::vector<CudaPlanarChart> retryJobs;std::vector<std::size_t> indices;
          for(std::size_t i=0;i<gpu.size();++i)if(gpu[i].Status==2&&gpu[i].CapacityHint<200000){
            auto input=gpu[i];input.CapacityHint=std::min(200000,2*input.CapacityHint);
            retryJobs.push_back(std::move(input));indices.push_back(i);
          }
          if(retryJobs.empty())break;
          if(verbose)std::clog << "[CadMesh] analytic CUDA chart capacity retry: jobs=" << retryJobs.size() << std::endl;
          ran=RebuildPlanarChartsCuda(retryJobs,cudaMilliseconds,error);
          if(ran)for(std::size_t i=0;i<indices.size();++i)gpu[indices[i]]=std::move(retryJobs[i]);
        }
        if(!ran){
          cudaDisabled=true;
          std::clog << "[CadMesh] analytic chart CUDA unavailable; CPU fallback: " << error << std::endl;
        }else for(std::size_t i=0;i<gpu.size();++i){
          auto &output=gpu[i];auto &r=ready[gpuIndices[i]];
          bool valid=output.Complete&&output.Points.size()>=r.Value.Points.size();
          if(valid){
            for(const auto &uv:output.Points)valid=valid&&std::isfinite(uv[0])&&std::isfinite(uv[1]);
            for(const auto &f:output.Faces)for(int v:f)valid=valid&&v>=0&&std::size_t(v)<output.Points.size();
            for(std::size_t v=0;v<r.Value.Points.size();++v)
              valid=valid&&output.Points[v][0]==r.Value.Points[v].X&&output.Points[v][1]==r.Value.Points[v].Y;
          }
          if(!valid){
            if(verbose)std::clog << "[CadMesh] analytic CUDA chart CPU retry: patch=" << patchId+gpuIndices[i]+1
                                << ", status=" << output.Status << std::endl;
            continue;
          }
          auto &chart=r.Value;chart.Points.clear();
          for(auto uv:output.Points)chart.Points.push_back({uv[0],uv[1]});
          chart.Aliases.resize(chart.Points.size(),-1);chart.Fixed.resize(chart.Points.size(),0);
          chart.Faces=std::move(output.Faces);r.State=PreparedChart::Ready;r.UsedCuda=true;++cudaCompleted;
        }
      }
      // Failed/unsupported runtime jobs use the existing worker pool. All
      // validation and global vertex IDs still commit in patch order.
      std::vector<std::pair<int,std::future<PreparedChart>>> retries;
      for(int i=0;i<count;++i)if(ready[i].State==PreparedChart::GpuPending){
        ++cudaFallback;
        retries.emplace_back(i,workers.submit([r=std::move(ready[i])]() mutable {
          auto start=Clock::now();bool ok=Refine(r.Value,r.Target);
          r.RefineSeconds=std::chrono::duration<double>(Clock::now()-start).count();
          if(ok){start=Clock::now();Improve(r.Value,r.Target);
            r.ImproveSeconds=std::chrono::duration<double>(Clock::now()-start).count();r.State=PreparedChart::Ready;
          }else {r.State=PreparedChart::Parameterization;r.Value={};}
          return r;
        }));
      }
      for(auto &retry:retries){
        while(retry.second.wait_for(std::chrono::seconds(5))!=std::future_status::ready)
          if(verbose)std::clog << "[CadMesh] analytic CPU fallback waiting: patch=" << patchId+retry.first+1 << std::endl;
        ready[retry.first]=retry.second.get();
      }
    }
    auto prepared=std::move(ready.front());ready.pop_front();
    const auto detailLogStart=Clock::now();
    if(prepared.SeededCylinder){
      ++seededCylinder;if(prepared.State!=PreparedChart::Ready)++seededCylinderFailed;
      const auto &detail=prepared.CylinderBuild;
      if(!detail.Failure.empty()){
        auto &group=cylinderFailures[detail.Failure];++group.first;group.second+=prepared.MakeSeconds;
      }
      if(verbose)std::clog << "[CadMesh] cylinder constrained mesh: patch=" << patchId+1
          << ", method=" << (boundaries[patchId].Loops.size()==2?"direct_periodic_strip":"unwrapped_cdt")
          << ", boundary_loops=" << boundaries[patchId].Loops.size()
          << ", points=" << prepared.Value.Points.size() << ", faces=" << prepared.Value.Faces.size()
          << ", constructed=" << (prepared.State==PreparedChart::Ready?"yes":"no")
          << ", circumferential_target=" << prepared.Target << ", axial_target=" << targetEdgeLength
          << ", ring_rows=" << detail.RingRows << ", ring_columns=" << detail.RingColumns
          << ", ring_interior_spacing=" << detail.RingInteriorSpacing
          << ", stage=" << detail.Stage << ", failure=" << (detail.Failure.empty()?"none":detail.Failure)
          << ", generated_points=" << detail.Points << ", generated_faces=" << detail.Faces
          << ", problem_edge_global=" << detail.GlobalEdgeA << ':' << detail.GlobalEdgeB
          << ", problem_edge_local=" << detail.EdgeA << ':' << detail.EdgeB
          << ", problem_edge_metric_length=" << detail.EdgeLength << ", problem_edge_metric_target=" << detail.EdgeTarget
          << ", work=" << detail.Work << ", legalize_s=" << detail.LegalizeSeconds
          << ", lattice_s=" << detail.LatticeSeconds << ", repair_s=" << detail.RepairSeconds
          << ", seconds=" << prepared.MakeSeconds << std::endl;
    }
    if(prepared.SeededPlanar){
      ++seededPlanar;
      if(prepared.State!=PreparedChart::Ready)++seededPlanarFailed;
      if(verbose)std::clog << "[CadMesh] planar constrained mesh: patch=" << patchId+1
          << ", boundary_loops=" << boundaries[patchId].Loops.size()
          << ", points=" << prepared.Value.Points.size() << ", faces=" << prepared.Value.Faces.size()
          << ", constructed=" << (prepared.State==PreparedChart::Ready?"yes":"no")
          << ", seconds=" << prepared.MakeSeconds << std::endl;
    }
    detailLogSeconds+=std::chrono::duration<double>(Clock::now()-detailLogStart).count();
    makeSeconds += prepared.MakeSeconds;
    refineSeconds += prepared.RefineSeconds;
    improveSeconds += prepared.ImproveSeconds;
    auto &reason=report.PatchReasons[patchId];
    if (prepared.State == PreparedChart::Skip) {
      if((simplePlanesOnly||cylindersOnly||conesOnly||otherFeaturesOnly) && !selectedSurface(patchId))
        reason=PatchRemeshReason::Unselected;
      else if(patches[patchId].SurfaceType==PatchSurfaceType::Freeform ||
              patches[patchId].SurfaceType==PatchSurfaceType::Unknown)
        reason=PatchRemeshReason::Unsupported;
      else reason=PatchRemeshReason::Nonanalytic;
      continue;
    }
    ++report.Attempted;
    if (prepared.State != PreparedChart::Ready) {
      reason=prepared.State==PreparedChart::Collision?PatchRemeshReason::Collision:
          prepared.State==PreparedChart::Deviation?PatchRemeshReason::Deviation:
          prepared.State==PreparedChart::Topology?PatchRemeshReason::Topology:
          PatchRemeshReason::Parameterization;
      ++report.Fallback;
      switch (prepared.State) {
      case PreparedChart::Collision: ++report.CollisionFallback; break;
      case PreparedChart::Deviation: ++report.DeviationFallback; break;
      case PreparedChart::Topology: ++report.TopologyFallback; break;
      default: ++report.ParameterizationFallback; break;
      }
      continue;
    }
    const auto liftStart=Clock::now();
    Chart &chart = prepared.Value;
    const Frame &frame = chart.Surface;
    const BoundaryData &boundary = boundaries[patchId];
    const std::size_t originalVertexCount=vertices.size();
    std::vector<int> mapping(chart.Points.size(),-1);
    std::unordered_map<int,int> periodicAliases;
    for(int i=0;i<int(chart.Points.size());++i){
      if(chart.Aliases[i]>=0)mapping[i]=chart.Aliases[i];
      else if(chart.Aliases[i] < -1) {
        auto found = periodicAliases.find(chart.Aliases[i]);
        if(found != periodicAliases.end()) mapping[i] = found->second;
        else { mapping[i]=int(vertices.size()); periodicAliases[chart.Aliases[i]]=mapping[i]; vertices.push_back(frame.lift(chart.Points[i])); }
      } else {mapping[i]=int(vertices.size());vertices.push_back(frame.lift(chart.Points[i]));}
    }
    std::vector<std::array<int,3>> candidateFaces;candidateFaces.reserve(chart.Faces.size());
    double orientation = 0;const auto &oldFaces=patchFaces[patchId];
    for(const auto &f:oldFaces){
      const Vec3 n=Cross(Sub(ToVec(vertices[f[1]]),ToVec(vertices[f[0]])),Sub(ToVec(vertices[f[2]]),ToVec(vertices[f[0]])));
      const Point3 center=ToPoint(Mul(Add(Add(ToVec(vertices[f[0]]),ToVec(vertices[f[1]])),ToVec(vertices[f[2]])),1.0/3));
      orientation += Dot(n,frame.normalAt(center));
    }
    orientation = orientation < 0 ? -1 : 1;
    for(auto f:chart.Faces){for(int&i:f)i=mapping[i];Vec3 n=Cross(Sub(ToVec(vertices[f[1]]),ToVec(vertices[f[0]])),Sub(ToVec(vertices[f[2]]),ToVec(vertices[f[0]])));const Point3 center=ToPoint(Mul(Add(Add(ToVec(vertices[f[0]]),ToVec(vertices[f[1]])),ToVec(vertices[f[2]])),1.0/3));if(orientation*Dot(n,frame.normalAt(center))<0)std::swap(f[1],f[2]);if(f[0]!=f[1]&&f[1]!=f[2]&&f[2]!=f[0])candidateFaces.push_back(f);}
    liftSeconds+=std::chrono::duration<double>(Clock::now()-liftStart).count();
    const auto validationStart=Clock::now();
    const bool valid=ValidatePatch(vertices,candidateFaces,boundary.Edges,targetEdgeLength);
    validationSeconds+=std::chrono::duration<double>(Clock::now()-validationStart).count();
    if(!valid){reason=PatchRemeshReason::Topology;vertices.resize(originalVertexCount);++report.Fallback;++report.TopologyFallback;continue;}
    report.RebuiltFaces+=candidateFaces.size();
    replacements[patchId]=std::move(candidateFaces);
    successfullyRebuilt[patchId]=1;++report.Rebuilt;
    reason=PatchRemeshReason::Rebuilt;
    if(prepared.UsedCuda)++cudaAccepted;
    if(prepared.SeededPlanar)++seededPlanarAccepted;
    if(prepared.SeededCylinder)++seededCylinderAccepted;
  }
  report.ChartSeconds = std::chrono::duration<double>(Clock::now()-rebuildStart).count();
  if(verbose){
    std::vector<int> slowest(patches.size());std::iota(slowest.begin(),slowest.end(),0);
    std::stable_sort(slowest.begin(),slowest.end(),[&](int a,int b){return taskSeconds[a]>taskSeconds[b];});
    for(std::size_t i=0;i<std::min<std::size_t>(8,slowest.size());++i){
      const int id=slowest[i];std::size_t boundaryCount=0;
      for(const auto &loop:boundaries[id].Loops)boundaryCount+=loop.size();
      std::clog << "[CadMesh] analytic slow task: patch=" << id+1
          << ", surface_type=" << int(patches[id].SurfaceType)
          << ", loops=" << boundaries[id].Loops.size() << ", boundary_vertices=" << boundaryCount
          << ", source_faces=" << patchFaces[id].size() << ", seconds=" << taskSeconds[id] << std::endl;
    }
  }
  if(verbose)std::clog << "[CadMesh] planar constrained mesh complete: attempted=" << seededPlanar
      << ", construction_failed=" << seededPlanarFailed << ", accepted=" << seededPlanarAccepted << std::endl;
  if(verbose)std::clog << "[CadMesh] cylinder constrained mesh complete: attempted=" << seededCylinder
      << ", construction_failed=" << seededCylinderFailed << ", accepted=" << seededCylinderAccepted << std::endl;
  if(verbose)for(const auto &group:cylinderFailures)
    std::clog << "[CadMesh] cylinder construction failure: reason=" << group.first
        << ", count=" << group.second.first << ", worker_seconds=" << group.second.second << std::endl;
  if(verbose&&cudaSubmitted)std::clog << "[CadMesh] analytic chart CUDA: batches=" << cudaBatches
      << ", submitted=" << cudaSubmitted << ", completed=" << cudaCompleted
      << ", accepted=" << cudaAccepted << ", cpu_fallback=" << cudaFallback
      << ", gpu_pipeline_ms=" << cudaMilliseconds << std::endl;
  if (verbose) std::clog << "[CadMesh] analytic charts complete: wall_s=" << report.ChartSeconds
                        << ", worker_time_sums(make=" << makeSeconds << ", refine=" << refineSeconds
                        << ", improve=" << improveSeconds << ")" << std::endl;
  if(verbose)std::clog << "[CadMesh] analytic commit timing: future_wait_s=" << waitSeconds
      << ", patch_detail_logging_s=" << detailLogSeconds << ", lift_and_orientation_s=" << liftSeconds
      << ", topology_validation_s=" << validationSeconds << std::endl;
  phaseStart=Clock::now();
  if(report.Rebuilt){
    std::size_t faceCount=faces.size();
    for(std::size_t patch=0;patch<patches.size();++patch)if(successfullyRebuilt[patch])
      faceCount=faceCount-patchFaces[patch].size()+replacements[patch].size();
    std::vector<std::array<int,3>> outputFaces;std::vector<int>outputLabels;
    outputFaces.reserve(faceCount);outputLabels.reserve(faceCount);
    for(std::size_t i=0;i<faces.size();++i){const int patch=labels[i];if(patch>=0&&patch<int(successfullyRebuilt.size())&&successfullyRebuilt[patch])continue;outputFaces.push_back(faces[i]);outputLabels.push_back(patch);}
    for(std::size_t patch=0;patch<replacements.size();++patch)for(const auto&face:replacements[patch]){outputFaces.push_back(face);outputLabels.push_back(int(patch));}
    faces.swap(outputFaces);labels.swap(outputLabels);
  }
  report.CommitSeconds=std::chrono::duration<double>(Clock::now()-phaseStart).count();
  return true;
}

} // namespace CadMesh
