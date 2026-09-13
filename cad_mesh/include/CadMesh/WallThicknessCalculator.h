#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace CadMesh {
// Values retain Apollo's diagnostic status numbering. Only Measured and
// FeatureSample represent accepted face measurements in this implementation.
enum class ThicknessStatus : std::uint8_t {
  Measured, DeletedVertex, InvalidNormal, MissingEdge, InvalidEdge,
  InvalidSphere, NonFinite, BelowMinimum, AboveMaximum, FeatureSample,
  Reconstructed, PenetratingSphere, LocalSurfaceContact, FaceMapped, NoFaceSupport
};
enum class ContactFeature : std::uint8_t { Unknown, Interior, Edge, Vertex };
struct RollingBallSample {
  std::array<double,3> mSource{},mCenter{},mContact{},mInward{};
  double mRadius=0,mContactResidual=0,mTolerance=0;
  int mSourceTriangle=-1,mContactTriangle=-1,mIterations=0,mCandidates=0;
  ContactFeature mFeature=ContactFeature::Unknown;
  bool mConverged=false;
};
struct WallThicknessOptions {
  int Workers=20;
  // Model units. Zero selects Apollo's 0.0075 * minimum bbox dimension.
  // This is a minimum accepted thickness, NOT solver accuracy.
  double MinimumThickness=0.01;
  double MinimumContactAngleDegrees=0;
};
struct WallThicknessResult {
  // One entry per input face. Invalid measurements are NaN, never filled.
  std::vector<float> Values;
  std::vector<ThicknessStatus> Status;
  std::vector<RollingBallSample> Samples;
  std::size_t ValidFaces=0;
  double Minimum=0,Maximum=0,AreaWeightedAverage=0,ValidAreaFraction=0;
  double PreparationSeconds=0,SamplingSeconds=0;
};
class WallThicknessCalculator {
public:
  // Apollo face rolling-ball algorithm, on a private mesh and shared read-only
  // AABB tree. Does not alter the input or restrict contacts to a single patch.
  // A successful computation can contain zero valid samples; inspect ValidFaces.
  static bool compute(const std::vector<std::array<double,3>>& vertices,
                      const std::vector<std::array<int,3>>& faces,
                      const WallThicknessOptions& options,
                      WallThicknessResult& result,std::string& error);
};
}
