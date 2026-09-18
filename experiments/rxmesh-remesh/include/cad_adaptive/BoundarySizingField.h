#pragma once
#include "cad_adaptive/Types.h"
#include <memory>
#include <vector>

#ifdef __CUDACC__
#define CAD_SIZING_HD __host__ __device__
#else
#define CAD_SIZING_HD
#endif

namespace cad_adaptive {
struct SemanticMesh;

struct BoundarySizingSeed {
  float Ax, Ay, Az, Bx, By, Bz, Size;
};
struct BoundarySizingNode {
  float MinX, MinY, MinZ, MaxX, MaxY, MaxZ, MinSize;
  uint32_t Skip = 0; // preorder index immediately after this subtree
  uint32_t Seed = kInvalidId;
};
struct BoundarySizingPatch {
  float BaseLength = 0;
  uint32_t Begin = 0, End = 0;
};

// Immutable local lower envelope. The acceleration structure changes neither
// the seed sizes nor their support: a short edge affects only its own patches.
CAD_SIZING_HD inline float evaluateBoundarySizing(
    float x, float y, float z, const BoundarySizingPatch &patch,
    const BoundarySizingNode *nodes, const BoundarySizingSeed *seeds, float gradation) {
  float best = patch.BaseLength;
  uint32_t index = patch.Begin;
  while (index < patch.End) {
    const BoundarySizingNode &node = nodes[index];
    const float dx = fmaxf(fmaxf(node.MinX-x, 0.0f), x-node.MaxX);
    const float dy = fmaxf(fmaxf(node.MinY-y, 0.0f), y-node.MaxY);
    const float dz = fmaxf(fmaxf(node.MinZ-z, 0.0f), z-node.MaxZ);
    const float radius = (best-node.MinSize)/gradation;
    if (!(radius > 0.0f) || dx*dx+dy*dy+dz*dz >= radius*radius) {
      index = node.Skip;
      continue;
    }
    if (node.Seed != kInvalidId) {
      const BoundarySizingSeed &seed = seeds[node.Seed];
      const float ex=seed.Bx-seed.Ax, ey=seed.By-seed.Ay, ez=seed.Bz-seed.Az;
      const float e2=ex*ex+ey*ey+ez*ez;
      const float t=e2>0.0f ? fminf(1.0f,fmaxf(0.0f,
          ((x-seed.Ax)*ex+(y-seed.Ay)*ey+(z-seed.Az)*ez)/e2)) : 0.0f;
      const float qx=x-seed.Ax-t*ex, qy=y-seed.Ay-t*ey, qz=z-seed.Az-t*ez;
      best=fminf(best,seed.Size+gradation*sqrtf(qx*qx+qy*qy+qz*qz));
    }
    ++index;
  }
  return best;
}

class BoundarySizingField {
public:
  static std::shared_ptr<const BoundarySizingField> create(
      const SemanticMesh &reference, const RemeshConfig &config,
      float gradation = 0.35f, bool curvatureSizing = false,
      const std::vector<float> &patchTargetLengths = {});
  float evaluate(uint32_t patchId, Vec3 point) const;
  void apply(SemanticMesh &mesh) const;
  float gradation() const { return mGradation; }
  const std::vector<BoundarySizingSeed> &seeds() const { return mSeeds; }
  const std::vector<BoundarySizingNode> &nodes() const { return mNodes; }
  const std::vector<BoundarySizingPatch> &patches() const { return mPatches; }
private:
  uint32_t BuildTree(std::vector<uint32_t> &ids, size_t begin, size_t end);
  float mGradation = 0.35f;
  float mFallbackLength = 1.0f;
  std::vector<BoundarySizingSeed> mSeeds;
  std::vector<BoundarySizingNode> mNodes;
  std::vector<BoundarySizingPatch> mPatches;
};
} // namespace cad_adaptive
#undef CAD_SIZING_HD
