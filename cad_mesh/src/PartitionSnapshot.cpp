#include "CadMesh/CadMeshPatchSegmenter.h"
#include "CadMesh/SegmentationGuards.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace CadMesh {
namespace {
struct Reader {
  std::ifstream Stream;
  explicit Reader(const std::filesystem::path &path) : Stream(path, std::ios::binary) {
    char magic[8]{};
    if (!Stream.read(magic, 8) || std::memcmp(magic, "CADPART1", 8))
      throw std::runtime_error("Invalid partition snapshot header");
  }
  std::uint64_t integer(int bytes) {
    unsigned char data[8]{};
    if (!Stream.read(reinterpret_cast<char*>(data), bytes))
      throw std::runtime_error("Truncated partition snapshot");
    std::uint64_t value = 0;
    for (int i = 0; i < bytes; ++i) value |= std::uint64_t(data[i]) << (8 * i);
    return value;
  }
  int count() {
    const auto value = integer(4);
    if (value > std::uint64_t(std::numeric_limits<int>::max()))
      throw std::runtime_error("Partition index exceeds int32");
    return int(value);
  }
  double number() {
    const auto bits = integer(8);
    double value;
    std::memcpy(&value, &bits, 8);
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite snapshot value");
    return value;
  }
};
void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
}

bool CadMeshPatchSegmenter::loadRemeshSnapshot(const std::filesystem::path &path,
                                              std::string &error) {
  try {
    Reader in(path);
    const int vertexCount = in.count(), faceCount = in.count();
    const int patchCount = in.count(), edgeCount = in.count();
    Require(vertexCount > 0 && faceCount > 0 && patchCount > 0 && patchCount <= faceCount,
            "Invalid snapshot counts");
    const auto bytes = std::filesystem::file_size(path);
    Require(std::uint64_t(vertexCount) * 24 + std::uint64_t(faceCount) * 16 +
                std::uint64_t(patchCount) * 92 + std::uint64_t(edgeCount) * 16 <= bytes,
            "Snapshot counts exceed file size");
    MeshResolutionInfo resolution;
    resolution.BoundingBoxDiagonal = in.number(); resolution.MedianEdgeLength = in.number();
    resolution.WeldTolerance = in.number(); resolution.FittingTolerance = in.number();
    resolution.CurvatureTolerance = in.number(); resolution.AngularTolerance = in.number();
    Require(resolution.BoundingBoxDiagonal > 0 && resolution.MedianEdgeLength > 0 &&
            resolution.WeldTolerance >= 0 && resolution.FittingTolerance > 0 &&
            resolution.AngularTolerance > 0, "Invalid saved resolution");
    TriangleSoup soup;
    soup.Vertices.resize(vertexCount); soup.Triangles.resize(faceCount);
    for (auto &point : soup.Vertices) for (int k = 0; k < 3; ++k) point[k] = in.number();
    std::vector<int> labels(faceCount);
    for (int i = 0; i < faceCount; ++i) {
      for (int &v : soup.Triangles[i]) { v = in.count(); Require(v < vertexCount, "Invalid vertex id"); }
      labels[i] = in.count(); Require(labels[i] < patchCount, "Invalid patch id");
    }
    CadMeshPatchSegmenter loaded(mConfig);
    Require(loaded.mMesh.buildIndexed(soup, resolution), "Snapshot mesh is not clean/index-preserving");
    loaded.mPatches.resize(patchCount);
    std::vector<int> expectedFaces(patchCount);
    for (int id = 0; id < patchCount; ++id) {
      auto &patch = loaded.mPatches[id]; patch.Id = id;
      const int type = in.count(), analytic = in.count(), role = in.count();
      expectedFaces[id] = in.count();
      const int supportCount = in.count();
      Require(type <= 6 && analytic <= 1 && role <= 1 && supportCount <= patchCount,
              "Invalid patch metadata");
      patch.SurfaceType = static_cast<PatchSurfaceType>(type);
      patch.ProjectionTarget = analytic ? PatchProjectionTarget::AnalyticSurface : PatchProjectionTarget::ReferenceMesh;
      patch.FeatureRole = role ? PatchFeatureRole::Fillet : PatchFeatureRole::Ordinary;
      for (int i = 0; i < supportCount; ++i) {
        int support = in.count(); Require(support < patchCount, "Invalid support patch");
        patch.SupportPatchIds.push_back(support);
      }
      double p[8]; for (double &value : p) value = in.number();
      patch.MaxSampledSurfaceDeviation = in.number(); // -1: legacy handoff omitted this statistic.
      if (analytic) {
        const Point3 origin(p[0], p[1], p[2]);
        const Direction3 direction(p[3], p[4], p[5]);
        if (type != 4) Require(Norm(ToVec(direction)) > .5, "Invalid surface direction");
        switch (patch.SurfaceType) {
        case PatchSurfaceType::Plane: patch.Parameters = PlaneParameters{{origin, direction}}; break;
        case PatchSurfaceType::Cylinder:
          Require(p[6] > 0, "Invalid cylinder radius");
          patch.Parameters = CylinderParameters{{origin, direction}, p[6]}; break;
        case PatchSurfaceType::Cone:
          Require(p[6] > 0 && p[6] < std::acos(-1.0) / 2, "Invalid cone angle");
          patch.Parameters = ConeParameters{{origin, direction}, p[6]}; break;
        case PatchSurfaceType::Sphere:
          Require(p[6] > 0, "Invalid sphere radius");
          patch.Parameters = SphereParameters{origin, p[6]}; break;
        case PatchSurfaceType::Torus:
          Require(p[6] > p[7] && p[7] > 0, "Invalid torus radii");
          patch.Parameters = TorusParameters{{origin, direction}, p[6], p[7]}; break;
        default: throw std::runtime_error("Unsupported analytic surface type");
        }
      }
    }
    for (int i = 0; i < faceCount; ++i) {
      loaded.mMesh.getTriangles()[i].PatchId = labels[i];
      loaded.mPatches[labels[i]].TriangleIds.push_back(i);
    }
    std::vector<int> constraintIds;
    for (int i = 0; i < edgeCount; ++i) {
      const int id = in.count(), a = in.count(), b = in.count(), hard = in.count();
      Require(id < int(loaded.mMesh.getEdges().size()) && hard <= 1, "Invalid saved constraint");
      auto &edge = loaded.mMesh.getEdges()[id];
      Require(edge.Vertex0 == std::min(a,b) && edge.Vertex1 == std::max(a,b),
              "Constraint edge ids disagree with saved geometry");
      edge.IsConstrainedFeature = hard != 0;
      constraintIds.push_back(id);
    }
    Require(in.Stream.peek() == std::char_traits<char>::eof(), "Trailing snapshot data");
    for (auto &patch : loaded.mPatches) {
      Require(int(patch.TriangleIds.size()) == expectedFaces[patch.Id], "Patch triangle count mismatch");
      if (patch.ProjectionTarget != PatchProjectionTarget::AnalyticSurface || patch.MaxSampledSurfaceDeviation >= 0)
        continue;
      // Older JSON exports omitted this value. Re-evaluate the saved surface
      // on its saved triangles; no fitting, growing or label changes occur.
      patch.MaxSampledSurfaceDeviation = 0;
      const auto sample = [&](const Vec3 &point) {
        double distance; Vec3 normal;
        Require(SegmentationGuardDetail::SurfaceSample(patch, point, distance, normal),
                "Cannot evaluate saved analytic surface");
        patch.MaxSampledSurfaceDeviation = std::max(patch.MaxSampledSurfaceDeviation, distance);
      };
      for (int id : patch.TriangleIds) {
        const auto &face = loaded.mMesh.getTriangles()[id];
        for (int k = 0; k < 3; ++k) {
          const auto a = ToVec(soup.Vertices[face.VertexIds[k]]);
          const auto b = ToVec(soup.Vertices[face.VertexIds[(k+1)%3]]);
          sample(a); sample(Mul(Add(a,b), .5));
        }
        sample(ToVec(face.Centroid));
      }
    }
    PatchGraphBuilder::build(loaded);
    std::sort(constraintIds.begin(), constraintIds.end());
    auto restored = loaded.mConstraint.ConstraintEdgeIds;
    std::sort(restored.begin(), restored.end());
    Require(constraintIds == restored, "Saved constraints disagree with patch boundaries");
    Require(loaded.validatePartition(&error), "Invalid saved partition ownership/connectivity");
    *this = std::move(loaded);
    error.clear(); return true;
  } catch (const std::exception &failure) {
    error = failure.what(); return false;
  }
}
}
