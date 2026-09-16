#pragma once

#include "rxmesh/attribute.h"

namespace cad_adaptive {
namespace gpu {

using EdgeStatus = int8_t;
enum : EdgeStatus { Unseen = 0, Skip = 1, Added = 3 };

constexpr int kFeatureEdge = 2;
constexpr int kPatchBoundary = 3;
constexpr int kCorner = 4;
constexpr int kLocked = 5;
constexpr int kEdgeMeshBoundary = -1;
constexpr int kEdgePatchBoundary = -2;
constexpr int kPatchPlane = 1;
constexpr int kPatchCylinder = 2;

struct PatchGpu {
  int type;
  float ox, oy, oz;
  float ax, ay, az;
  float radius;
};

__device__ inline float length2(float ax, float ay, float az) { return ax * ax + ay * ay + az * az; }
__device__ inline float3 sub3(float3 a, float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ inline float dot3(float3 a,float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ inline float3 normal3(float3 a,float3 b,float3 c) {
  b=sub3(b,a); c=sub3(c,a);
  return make_float3(b.y*c.z-b.z*c.y,b.z*c.x-b.x*c.z,b.x*c.y-b.y*c.x);
}
template<class Attribute, class Handle>
__device__ inline float3 point3(const Attribute &coords,Handle v) {
  return make_float3(coords(v,0),coords(v,1),coords(v,2));
}

__device__ inline bool projectPatch(const PatchGpu &p, float &x, float &y, float &z) {
  if (p.type == kPatchPlane) {
    const float n2 = length2(p.ax, p.ay, p.az);
    if (!(n2 > 0.f)) return false;
    const float inv = rsqrtf(n2);
    const float ax = p.ax * inv, ay = p.ay * inv, az = p.az * inv;
    const float d = (x - p.ox) * ax + (y - p.oy) * ay + (z - p.oz) * az;
    x -= ax * d;
    y -= ay * d;
    z -= az * d;
    return true;
  }
  if (p.type == kPatchCylinder) {
    if (!(p.radius > 0.f)) return false;
    const float n2 = length2(p.ax, p.ay, p.az);
    if (!(n2 > 0.f)) return false;
    const float inv = rsqrtf(n2);
    const float ax = p.ax * inv, ay = p.ay * inv, az = p.az * inv;
    const float dx = x - p.ox, dy = y - p.oy, dz = z - p.oz;
    const float h = dx * ax + dy * ay + dz * az;
    const float rx = dx - ax * h, ry = dy - ay * h, rz = dz - az * h;
    const float rn2 = length2(rx, ry, rz);
    if (!(rn2 > 1e-20f)) return false;
    const float s = p.radius * rsqrtf(rn2);
    x = p.ox + ax * h + rx * s;
    y = p.oy + ay * h + ry * s;
    z = p.oz + az * h + rz * s;
    return true;
  }
  return false;
}

__device__ inline bool isLocked(int constraint) { return constraint >= kCorner; }
__device__ inline bool isSeamConstraint(int constraint) {
  return constraint == kFeatureEdge || constraint == kPatchBoundary;
}
__device__ inline bool isImmobile(int constraint) { return constraint >= kPatchBoundary; }

__device__ inline float triQuality(float ax, float ay, float az, float bx, float by, float bz,
                                   float cx, float cy, float cz) {
  const float abx = bx - ax, aby = by - ay, abz = bz - az;
  const float acx = cx - ax, acy = cy - ay, acz = cz - az;
  const float bcx = cx - bx, bcy = cy - by, bcz = cz - bz;
  const float nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
  const float area2 = sqrtf(length2(nx, ny, nz));
  const float denom = length2(abx, aby, abz) + length2(acx, acy, acz) + length2(bcx, bcy, bcz);
  if (!(denom > 0.f) || !(area2 > 0.f)) return 0.f;
  const float q = 2.f * 1.7320508f * area2 / denom;
  return q < 0.f ? 0.f : (q > 1.f ? 1.f : q);
}

} // namespace gpu
} // namespace cad_adaptive
