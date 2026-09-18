#pragma once

#include "device_math.cuh"
#include "rxmesh/query.h"

template <uint32_t blockThreads>
__global__ void vertex_smooth_kernel(rxmesh::Context context,
                                     const rxmesh::VertexAttribute<float> coords,
                                     rxmesh::VertexAttribute<float> next,
                                     const rxmesh::VertexAttribute<int> constraint,
                                     const rxmesh::VertexAttribute<bool> vBoundary,
                                     rxmesh::VertexAttribute<int> vDirty,
                                     rxmesh::VertexAttribute<int> vTouched, const float lambda,
                                     int *accepted) {
  using namespace rxmesh;
  auto block = cooperative_groups::this_thread_block();
  auto smooth = [&](VertexHandle v, VertexIterator &iter) {
    next(v, 0) = coords(v, 0);
    next(v, 1) = coords(v, 1);
    next(v, 2) = coords(v, 2);
    if (iter.size() == 0 || vBoundary(v) || cad_adaptive::gpu::isImmobile(constraint(v)) ||
        cad_adaptive::gpu::isSeamConstraint(constraint(v)) || !vDirty(v))
      return;
    float sx = 0, sy = 0, sz = 0;
    float nx = 0, ny = 0, nz = 0;
    float px = coords(v, 0), py = coords(v, 1), pz = coords(v, 2);
    VertexHandle prev = iter.back();
    float qx = coords(prev, 0), qy = coords(prev, 1), qz = coords(prev, 2);
    for (uint32_t i = 0; i < iter.size(); ++i) {
      const VertexHandle r = iter[i];
      const float rx = coords(r, 0), ry = coords(r, 1), rz = coords(r, 2);
      const float ax = qx - px, ay = qy - py, az = qz - pz;
      const float bx = rx - px, by = ry - py, bz = rz - pz;
      nx += ay * bz - az * by;
      ny += az * bx - ax * bz;
      nz += ax * by - ay * bx;
      sx += rx;
      sy += ry;
      sz += rz;
      qx = rx;
      qy = ry;
      qz = rz;
    }
    const float inv = 1.f / float(iter.size());
    float cx = sx * inv, cy = sy * inv, cz = sz * inv;
    const float n2 = cad_adaptive::gpu::length2(nx, ny, nz);
    if (n2 < 1e-12f) return;
    const float invn = rsqrtf(n2);
    nx *= invn;
    ny *= invn;
    nz *= invn;
    float dx = cx - px, dy = cy - py, dz = cz - pz;
    const float dn = dx * nx + dy * ny + dz * nz;
    dx -= nx * dn;
    dy -= ny * dn;
    dz -= nz * dn;
    using namespace cad_adaptive::gpu;
    float step=lambda;
    bool valid=false;
    for(int attempt=0;attempt<8 && !valid;++attempt) {
      valid=true;
      auto proposed=make_float3(px+step*dx,py+step*dy,pz+step*dz);
      auto old=make_float3(px,py,pz);
      for(uint32_t i=0;i<iter.size();++i) {
        auto a=point3(coords,iter[i]),b=point3(coords,iter[(i+1)%iter.size()]);
        auto before=normal3(old,a,b),after=normal3(proposed,a,b);
        if(dot3(before,after)<=1e-6f*dot3(before,before)) { valid=false; break; }
      }
      if(!valid) step*=0.5f;
    }
    if(!valid) return;
    next(v, 0) = px + step * dx;
    next(v, 1) = py + step * dy;
    next(v, 2) = pz + step * dz;
    if (next(v, 0) != px || next(v, 1) != py || next(v, 2) != pz) {
      vTouched(v) = 1;
      ::atomicAdd(accepted, 1);
    }
  };
  Query<blockThreads> query(context);
  ShmemAllocator shmem;
  query.dispatch<Op::VV>(block, shmem, smooth, true);
}
