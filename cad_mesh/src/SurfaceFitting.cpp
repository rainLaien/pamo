#include "CadMesh/SurfaceFitting.h"
#include "CadMesh/CudaAnalyticFitting.h"
#include "CadMesh/CudaAnalyticPrefetch.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

namespace CadMesh {
namespace {
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;
V E(const Point3 &p) { return {p.X(), p.Y(), p.Z()}; }
Point3 P(const V &p) { return {p.x(), p.y(), p.z()}; }
constexpr double Pi = 3.14159265358979323846;
struct WeightedPoint { V Point; double Weight; };
struct WeightedFace { V Point, Normal; double Weight; int Id; };
// Incident area / 3 is a vertex quadrature weight. In particular a heavily
// tessellated part of a surface no longer receives one vote per vertex.
struct Cloud {
  V Origin = V::Zero();
  double Scale = 0;
  std::vector<WeightedPoint> Points, Samples;
  std::vector<WeightedFace> Faces;
  M Covariance = M::Zero(), NormalCovariance = M::Zero();
  M CenteredNormalCovariance = M::Zero();
  bool build(const MeshTopology &mesh, const std::vector<int> &triangles) {
    std::map<int, double> weights;
    double area = 0;
    for (int id : triangles) {
      if (id < 0 || id >= int(mesh.getTriangles().size())) return false;
      const auto &t = mesh.getTriangles()[id];
      if (!std::isfinite(t.Area) || t.Area <= 0) continue;
      area += t.Area;
      for (int v : t.VertexIds) weights[v] += t.Area / 3;
    }
    if (!(area > 0) || !std::isfinite(area) || weights.size() < 3) return false;
    // Subtract a local anchor before accumulating, preserving small detail on
    // models that are positioned far from the global origin.
    const V anchor = E(mesh.getVertices()[weights.begin()->first].Position);
    for (const auto &entry : weights)
      Origin += (entry.second / area) *
                (E(mesh.getVertices()[entry.first].Position) - anchor);
    Origin += anchor;
    for (const auto &entry : weights) {
      const V point = E(mesh.getVertices()[entry.first].Position) - Origin;
      if (!point.allFinite()) return false;
      Scale += entry.second / area * point.squaredNorm();
    }
    Scale = std::sqrt(Scale);
    if (!(Scale > 1e-100) || !std::isfinite(Scale)) return false;
    for (const auto &entry : weights) {
      const V point = (E(mesh.getVertices()[entry.first].Position) - Origin) / Scale;
      const double weight = entry.second / area;
      Points.push_back({point, weight});
      Covariance.noalias() += weight * point * point.transpose();
    }
    V averageNormal = V::Zero();
    for (int id : triangles) {
      const auto &t = mesh.getTriangles()[id];
      if (!(t.Area > 0)) continue;
      const V normal = E(t.Normal);
      if (!normal.allFinite()) return false;
      const double weight = t.Area / area;
      Faces.push_back({(E(t.Centroid) - Origin) / Scale, normal, weight, id});
      NormalCovariance.noalias() += weight * normal * normal.transpose();
      averageNormal += weight * normal;
    }
    CenteredNormalCovariance = NormalCovariance - averageNormal * averageNormal.transpose();
    // Bound nonlinear work. Area-stratified deterministic samples propose a
    // model; the reported errors and acceptance always visit ALL vertices.
    if (Points.size() <= 768) Samples = Points;
    else {
      size_t index = 0;
      double cumulative = Points.front().Weight;
      for (int i = 0; i < 768; ++i) {
        const double target = (i + .5) / 768.0;
        while (index + 1 < Points.size() && cumulative < target)
          cumulative += Points[++index].Weight;
        Samples.push_back({Points[index].Point, 1.0 / 768});
      }
    }
    return true;
  }
};
double Angle(const V &a, const V &b) {
  if (a.norm() < 1e-14 || b.norm() < 1e-14) return Pi / 2;
  return std::acos(std::clamp(std::abs(a.dot(b) / (a.norm() * b.norm())), 0.0, 1.0));
}
template <typename Distance, typename Normal>
void Errors(const Cloud &c, Distance distance, Normal normal,
            double &rms, double &maximum, double &normalError) {
  double squared = 0;
  maximum = normalError = 0;
  for (const auto &p : c.Points) {
    const double error = std::abs(distance(p.Point)) * c.Scale;
    squared += p.Weight * error * error;
    maximum = std::max(maximum, error);
  }
  for (const auto &f : c.Faces)
    normalError += f.Weight * Angle(f.Normal, normal(f.Point));
  rms = std::sqrt(squared);
}
bool Eigenvectors(const M &matrix, Eigen::Vector3d &values, M &vectors) {
  Eigen::SelfAdjointEigenSolver<M> solver(matrix);
  if (solver.info() != Eigen::Success) return false;
  values = solver.eigenvalues();
  vectors = solver.eigenvectors();
  return values.allFinite() && vectors.allFinite();
}
void Basis(const V &axis, V &u, V &v) {
  u = axis.unitOrthogonal();
  v = axis.cross(u).normalized();
}
bool Circle(const std::vector<WeightedPoint> &points, const V &axis,
            V &center, double &radius) {
  V u, v;
  Basis(axis, u, v);
  M normal = M::Zero();
  V rhs = V::Zero();
  double h = 0;
  for (const auto &p : points) {
    double x = p.Point.dot(u), y = p.Point.dot(v);
    V row(2 * x, 2 * y, 1);
    normal.noalias() += p.Weight * row * row.transpose();
    rhs += p.Weight * (x * x + y * y) * row;
    h += p.Weight * p.Point.dot(axis);
  }
  Eigen::SelfAdjointEigenSolver<M> spectrum(normal);
  if (spectrum.info() != Eigen::Success ||
      spectrum.eigenvalues()[0] < 1e-11 * spectrum.eigenvalues()[2]) return false;
  V answer = normal.ldlt().solve(rhs);
  if (!answer.allFinite()) return false;
  center = answer[0] * u + answer[1] * v + h * axis;
  radius = 0;
  for (const auto &p : points) {
    V d = p.Point - center;
    radius += p.Weight * (d - d.dot(axis) * axis).norm();
  }
  return std::isfinite(radius) && radius > 1e-8 && radius < 1e4;
}
// Small damped least-squares solver in local, dimensionless coordinates.
// Axis vectors are normalized in the residual, so damping removes their
// redundant radial degree of freedom without an angle-pole singularity.
template <typename Residual, typename Valid>
bool Refine(const Cloud &cloud, Eigen::VectorXd &parameters,
            Residual residual, Valid valid, int iterations = 32) {
  if (!valid(parameters)) return false;
  const int count = int(cloud.Samples.size()), dimension = int(parameters.size());
  auto evaluate = [&](const Eigen::VectorXd &p, Eigen::VectorXd &out) {
    out.resize(count);
    for (int i = 0; i < count; ++i)
      out[i] = std::sqrt(cloud.Samples[i].Weight) * residual(cloud.Samples[i].Point, p);
    return out.allFinite();
  };
  Eigen::VectorXd errors;
  if (!evaluate(parameters, errors)) return false;
  double cost = errors.squaredNorm(), damping = 1e-5;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    Eigen::MatrixXd jacobian(count, dimension);
    for (int j = 0; j < dimension; ++j) {
      Eigen::VectorXd probe = parameters, other;
      const double delta = 1e-6 * std::max(1.0, std::abs(parameters[j]));
      probe[j] += delta;
      if (!evaluate(probe, other)) return false;
      jacobian.col(j) = (other - errors) / delta;
    }
    const Eigen::MatrixXd hessian = jacobian.transpose() * jacobian;
    const Eigen::VectorXd gradient = jacobian.transpose() * errors;
    if (gradient.lpNorm<Eigen::Infinity>() < 1e-13 || cost < 1e-24) break;
    bool accepted = false;
    for (int trial = 0; trial < 7; ++trial) {
      Eigen::MatrixXd system = hessian;
      for (int j = 0; j < dimension; ++j)
        system(j,j) += damping * std::max(1e-4, hessian(j,j));
      Eigen::VectorXd step = system.ldlt().solve(-gradient);
      if (!step.allFinite()) return false;
      Eigen::VectorXd proposal = parameters + step, nextErrors;
      if (valid(proposal) && evaluate(proposal, nextErrors) &&
          nextErrors.squaredNorm() < cost) {
        const double nextCost = nextErrors.squaredNorm();
        parameters = proposal;
        errors = nextErrors;
        const double improvement = cost - nextCost;
        cost = nextCost;
        damping = std::max(1e-12, damping * .25);
        accepted = true;
        if (step.norm() < 1e-9 || improvement < 1e-15 * std::max(1.0, cost))
          return true;
        break;
      }
      damping *= 10;
    }
    if (!accepted) break;
  }
  return parameters.allFinite();
}

// Upload one normalized cloud and all of a model's initial guesses together.
// Consume results in the original seed order: CPU fallback remains lazy, so a
// torus's early success still avoids refining its unused seeds on the CPU.
class RefineCandidates {
public:
  RefineCandidates(const Cloud &cloud, PatchSurfaceType type,
                   const std::vector<Eigen::VectorXd> &initial, int iterations)
      : mCloud(cloud), mInitial(initial), mIterations(iterations) {
    if (initial.empty()) return;
    mParameters.resize(initial.size());
    for (size_t i = 0; i < initial.size(); ++i) {
      if (initial[i].size() > 8) return;
      mParameters[i].fill(0);
      for (Eigen::Index j = 0; j < initial[i].size(); ++j)
        mParameters[i][size_t(j)] = initial[i][j];
    }
    std::vector<std::array<double, 4>> samples;
    samples.reserve(cloud.Samples.size());
    for (const auto &sample : cloud.Samples)
      samples.push_back({sample.Point.x(), sample.Point.y(), sample.Point.z(),
                         sample.Weight});
    mOnCuda = RefineAnalyticSeedsPrepared(type, samples, mParameters, mValid,
                                     iterations) &&
              mParameters.size() == initial.size() &&
              mValid.size() == initial.size();
  }

  template <typename Residual, typename Valid>
  bool refine(size_t index, Eigen::VectorXd &parameters,
              Residual residual, Valid valid) const {
    parameters = mInitial[index];
    if (!valid(parameters)) return false;
    if (mOnCuda && mValid[index]) {
      Eigen::VectorXd proposal = parameters;
      for (Eigen::Index j = 0; j < proposal.size(); ++j)
        proposal[j] = mParameters[index][size_t(j)];
      // Keep the fitter's original bounds as the authority even if a device
      // solver reports success. Never start CPU fallback from a failed device
      // iterate; it must receive the same initial guess as the old path.
      if (valid(proposal)) {
        parameters = std::move(proposal);
        return true;
      }
    }
    return Refine(mCloud, parameters, residual, valid, mIterations);
  }

private:
  const Cloud &mCloud;
  const std::vector<Eigen::VectorXd> &mInitial;
  int mIterations;
  bool mOnCuda = false;
  std::vector<std::array<double, 8>> mParameters;
  std::vector<unsigned char> mValid;
};

V Axis(const Eigen::VectorXd &p) { return p.segment<3>(3).normalized(); }
bool AxisValid(const Eigen::VectorXd &p) {
  return p.allFinite() && p.segment<3>(3).norm() > .1 &&
         p.segment<3>(3).norm() < 10 && p.head<3>().norm() < 1e4;
}
} // namespace

bool PlaneSurfaceFitter::fit(const MeshTopology &mesh, const std::vector<int> &triangles) {
  Cloud c;
  if (!c.build(mesh, triangles)) return false;
  V values; M vectors;
  if (!Eigenvectors(c.Covariance, values, vectors) || values[1] < 1e-14) return false;
  const V n = vectors.col(0);
  Errors(c, [&](const V &p) { return p.dot(n); }, [&](const V &) { return n; },
         mRms, mMax, mNormal);
  mPlane = {P(c.Origin), P(n)};
  return true;
}

bool CylinderSurfaceFitter::fit(const MeshTopology &mesh, const std::vector<int> &triangles) {
  Cloud c;
  if (!c.build(mesh, triangles) || c.Points.size() < 6) return false;
  V values; M vectors;
  if (!Eigenvectors(c.NormalCovariance, values, vectors) || values[1] < 1e-8) return false;
  std::vector<V> candidates{vectors.col(0)};
  if (Eigenvectors(c.Covariance, values, vectors)) candidates.push_back(vectors.col(2));
  V curvatureAxis = V::Zero();
  for (int id : triangles)
    for (int vertex : mesh.getTriangles()[id].VertexIds) {
      const auto &geometry = mesh.getVertices()[vertex].Geometry;
      if (geometry.Confidence <= .1) continue;
      V direction = E(std::abs(geometry.K1) <= std::abs(geometry.K2)
                          ? geometry.PrincipalDirection1 : geometry.PrincipalDirection2);
      if (curvatureAxis.dot(direction) < 0) direction = -direction;
      curvatureAxis += mesh.getTriangles()[id].Area * geometry.Confidence * direction;
    }
  if (curvatureAxis.norm() > 1e-20) candidates.push_back(curvatureAxis.normalized());
  std::vector<Eigen::VectorXd> seeds;
  seeds.reserve(candidates.size());
  for (const V &axis : candidates) {
    V center; double radius;
    if (!Circle(c.Points, axis, center, radius)) continue;
    Eigen::VectorXd parameters(7);
    parameters << center, axis, radius;
    seeds.push_back(std::move(parameters));
  }
  auto residual = [](const V &p, const Eigen::VectorXd &x) {
    const V d = p - x.head<3>(), a = Axis(x);
    return (d - d.dot(a) * a).norm() - x[6];
  };
  auto valid = [](const Eigen::VectorXd &x) { return AxisValid(x) && x[6] > 1e-7 && x[6] < 1e4; };
  RefineCandidates refinements(c, PatchSurfaceType::Cylinder, seeds, 18);
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < seeds.size(); ++i) {
    Eigen::VectorXd parameters;
    if (!refinements.refine(i, parameters, residual, valid)) continue;
    const V a = Axis(parameters), o = parameters.head<3>();
    double rms, maximum, normal;
    Errors(c, [&](const V &p) { return residual(p, parameters); }, [&](const V &p) -> V {
      const V d = p - o; return d - d.dot(a) * a;
    }, rms, maximum, normal);
    double score = rms + .25 * maximum + .25 * mesh.getResolution().MedianEdgeLength * normal;
    if (score < best) {
      best = score;
      mRms = rms; mMax = maximum; mNormal = normal;
      mAxis = {P(c.Origin + c.Scale * o), P(a)};
      mRadius = parameters[6] * c.Scale;
    }
  }
  return std::isfinite(best);
}

bool SphereSurfaceFitter::fit(const MeshTopology &mesh, const std::vector<int> &triangles) {
  Cloud c;
  if (!c.build(mesh, triangles) || c.Points.size() < 5) return false;
  Eigen::Matrix4d matrix = Eigen::Matrix4d::Zero();
  Eigen::Vector4d rhs = Eigen::Vector4d::Zero();
  for (const auto &p : c.Points) {
    Eigen::Vector4d row; row << 2 * p.Point, 1;
    matrix.noalias() += p.Weight * row * row.transpose();
    rhs += p.Weight * p.Point.squaredNorm() * row;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> spectrum(matrix);
  if (spectrum.info() != Eigen::Success ||
      spectrum.eigenvalues()[0] < 1e-11 * spectrum.eigenvalues()[3]) return false;
  Eigen::VectorXd parameters = matrix.ldlt().solve(rhs);
  parameters[3] = 0;
  for (const auto &p : c.Points) parameters[3] += p.Weight * (p.Point - parameters.head<3>()).norm();
  auto residual = [](const V &p, const Eigen::VectorXd &x) { return (p - x.head<3>()).norm() - x[3]; };
  auto valid = [](const Eigen::VectorXd &x) { return x.allFinite() && x.head<3>().norm() < 1e4 && x[3] > 1e-7 && x[3] < 1e4; };
  const std::vector<Eigen::VectorXd> seeds{parameters};
  RefineCandidates refinements(c, PatchSurfaceType::Sphere, seeds, 20);
  if (!refinements.refine(0, parameters, residual, valid)) return false;
  const V center = parameters.head<3>();
  Errors(c, [&](const V &p) { return residual(p, parameters); },
         [&](const V &p) -> V { return p - center; }, mRms, mMax, mNormal);
  mCenter = P(c.Origin + c.Scale * center);
  mRadius = c.Scale * parameters[3];
  return true;
}

bool ConeSurfaceFitter::fit(const MeshTopology &mesh, const std::vector<int> &triangles) {
  Cloud c;
  if (!c.build(mesh, triangles) || c.Points.size() < 9 || c.Faces.size() < 6) return false;
  V values; M vectors;
  if (!Eigenvectors(c.CenteredNormalCovariance, values, vectors) || values[1] < 1e-6) return false;
  V axis = vectors.col(0);
  // Tangent planes of a cone meet at its apex. A cylinder or a flat strip has
  // an unobservable apex and is rejected instead of inventing a remote one.
  V normalValues; M unused;
  if (!Eigenvectors(c.NormalCovariance, normalValues, unused) ||
      normalValues[0] < 1e-7 * normalValues[2]) return false;
  V rhs = V::Zero();
  for (const auto &f : c.Faces) rhs += f.Weight * f.Normal.dot(f.Point) * f.Normal;
  V apex = c.NormalCovariance.ldlt().solve(rhs);
  if (!apex.allFinite() || apex.norm() > 1e3) return false;
  double height = 0, radial = 0;
  for (const auto &p : c.Points) {
    V d = p.Point - apex;
    height += p.Weight * d.dot(axis);
    radial += p.Weight * (d - d.dot(axis) * axis).norm();
  }
  if (height < 0) { axis = -axis; height = -height; }
  Eigen::VectorXd parameters(7);
  parameters << apex, axis, std::atan2(radial, height);
  auto residual = [](const V &p, const Eigen::VectorXd &x) {
    V d = p - x.head<3>(), a = Axis(x);
    const double z = d.dot(a), rho = (d - z * a).norm();
    // Signed perpendicular distance to the selected nappe, not r - z*tan(a).
    // Behind the apex the nearest point may instead be the apex itself.
    if (z * std::cos(x[6]) + rho * std::sin(x[6]) <= 0) return d.norm();
    return rho * std::cos(x[6]) - z * std::sin(x[6]);
  };
  auto valid = [](const Eigen::VectorXd &x) {
    return AxisValid(x) && x[6] > .00872664626 && x[6] < Pi / 2 - .00872664626;
  };
  const std::vector<Eigen::VectorXd> seeds{parameters};
  RefineCandidates refinements(c, PatchSurfaceType::Cone, seeds, 32);
  if (!refinements.refine(0, parameters, residual, valid)) return false;
  const V a = Axis(parameters), o = parameters.head<3>();
  double minZ = 1e100, maxZ = -1e100;
  for (const auto &p : c.Points) {
    const double z = (p.Point - o).dot(a);
    minZ = std::min(minZ, z); maxZ = std::max(maxZ, z);
  }
  if (minZ < -1e-5 || maxZ - minZ < .02) return false;
  Errors(c, [&](const V &p) { return residual(p, parameters); }, [&](const V &p) -> V {
    const V d = p - o, r = d - d.dot(a) * a;
    if (r.norm() < 1e-12) return V::Zero();
    return std::cos(parameters[6]) * r.normalized() - std::sin(parameters[6]) * a;
  }, mRms, mMax, mNormal);
  mAxis = {P(c.Origin + c.Scale * o), P(a)};
  mAngle = parameters[6];
  return std::isfinite(mRms) && std::isfinite(mNormal);
}

bool TorusSurfaceFitter::fit(const MeshTopology &mesh, const std::vector<int> &triangles) {
  Cloud c;
  if (!c.build(mesh, triangles) || c.Points.size() < 20 || c.Faces.size() < 20) return false;
  V values; M vectors;
  if (!Eigenvectors(c.CenteredNormalCovariance, values, vectors) || values[1] < 2e-5) return false;
  // A two-dimensional normal span is required. A planar patch or a strip with
  // only one resolvable curvature cannot establish seven torus parameters.
  if (values[0] < 1e-7) return false;
  std::vector<Eigen::VectorXd> seeds;
  auto addSeed = [&](const V &center, const V &axis, double major, double minor) {
    if (!(major > 1.005 * minor && minor > 1e-4)) return;
    Eigen::VectorXd p(8); p << center, axis.normalized(), major, minor;
    seeds.push_back(p);
  };
  // First seeds: projected circle centers for all point-covariance axes.
  if (Eigenvectors(c.Covariance, values, vectors))
    for (int i = 0; i < 3; ++i) {
      const V axis = vectors.col(i);
      V center; double major;
      if (!Circle(c.Points, axis, center, major)) continue;
      double minor = 0;
      for (const auto &p : c.Points) {
        V d = p.Point - center;
        double z = d.dot(axis), rho = (d - z * axis).norm();
        minor += p.Weight * std::hypot(rho - major, z);
      }
      addSeed(center, axis, major, minor);
    }
  // Normal-offset centers lie on the torus central circle when the offset
  // equals the tube radius. Estimate its scale from neighboring face normals,
  // then try both signs (input winding may be inward).
  std::map<int, size_t> faceIndex;
  for (size_t i = 0; i < c.Faces.size(); ++i) faceIndex[c.Faces[i].Id] = i;
  std::vector<double> radii;
  for (const auto &f : c.Faces)
    for (int neighbor : mesh.getTriangleNeighbors(f.Id)) {
      auto found = faceIndex.find(neighbor);
      if (neighbor <= f.Id || found == faceIndex.end()) continue;
      const auto &g = c.Faces[found->second];
      const V dn = g.Normal - f.Normal, dp = g.Point - f.Point;
      if (dn.squaredNorm() < 1e-8) continue;
      double radius = std::abs(dp.dot(dn)) / dn.squaredNorm();
      if (radius > 1e-4 && radius < 20) radii.push_back(radius);
    }
  if (!radii.empty()) {
    std::sort(radii.begin(), radii.end());
    double estimate = radii[radii.size() / 5];
    for (double multiplier : {.7, 1.0, 1.4})
      for (double sign : {-1.0, 1.0}) {
        const double minor = estimate * multiplier;
        std::vector<WeightedPoint> centers;
        V mean = V::Zero();
        for (const auto &f : c.Faces) {
          V q = f.Point - sign * minor * f.Normal;
          centers.push_back({q, f.Weight}); mean += f.Weight * q;
        }
        M covariance = M::Zero();
        for (const auto &q : centers) {
          const V d = q.Point - mean;
          covariance.noalias() += q.Weight * d * d.transpose();
        }
        V eigenvalues; M eigenvectors;
        if (!Eigenvectors(covariance, eigenvalues, eigenvectors)) continue;
        const V axis = eigenvectors.col(0);
        V center; double major;
        if (Circle(centers, axis, center, major)) addSeed(center, axis, major, minor);
      }
  }
  auto residual = [](const V &p, const Eigen::VectorXd &x) {
    const V d = p - x.head<3>(), a = Axis(x);
    const double z = d.dot(a), rho = (d - z * a).norm();
    return std::hypot(rho - x[6], z) - x[7];
  };
  auto valid = [](const Eigen::VectorXd &x) {
    return AxisValid(x) && x[7] > 1e-4 && x[6] > 1.005 * x[7] && x[6] < 100;
  };
  RefineCandidates refinements(c, PatchSurfaceType::Torus, seeds, 45);
  double best = 1e100;
  for (size_t i = 0; i < seeds.size(); ++i) {
    Eigen::VectorXd parameters;
    if (!refinements.refine(i, parameters, residual, valid)) continue;
    const V a = Axis(parameters), o = parameters.head<3>();
    double rms, maximum, normal;
    Errors(c, [&](const V &p) { return residual(p, parameters); }, [&](const V &p) -> V {
      const V d = p - o;
      const double z = d.dot(a);
      const V radial = d - z * a;
      if (radial.norm() < 1e-12) return V::Zero();
      return (radial.norm() - parameters[6]) * radial.normalized() + z * a;
    }, rms, maximum, normal);
    double score = rms + .25 * maximum + .25 * mesh.getResolution().MedianEdgeLength * normal;
    if (score < best) {
      best = score;
      mRms = rms; mMax = maximum; mNormal = normal;
      mAxis = {P(c.Origin + c.Scale * o), P(a)};
      mMajor = c.Scale * parameters[6]; mMinor = c.Scale * parameters[7];
    }
    if (rms / c.Scale < 1e-9 && maximum / c.Scale < 1e-8) break;
  }
  return best < 1e100;
}

bool FreeformSurfaceFitter::fit(const MeshTopology &mesh, const std::vector<int> &triangles) {
  PlaneSurfaceFitter plane;
  if (!plane.fit(mesh, triangles)) return false;
  // Diagnostic only; no fitted freeform surface is manufactured here.
  mRms = plane.computeRmsError(); mMax = plane.computeMaxError(); mNormal = plane.computeNormalError();
  return true;
}
SurfaceFitResult SurfaceModelSelector::fitBest(const MeshTopology &mesh,
                                              const std::vector<int> &triangles,
                                              const MeshResolutionInfo &r,
                                              double penalty) {
  SurfaceFitResult best;
  std::vector<std::unique_ptr<ISurfaceFitter>> fitters;
  fitters.push_back(std::make_unique<PlaneSurfaceFitter>());
  fitters.push_back(std::make_unique<CylinderSurfaceFitter>());
  fitters.push_back(std::make_unique<ConeSurfaceFitter>());
  fitters.push_back(std::make_unique<SphereSurfaceFitter>());
  fitters.push_back(std::make_unique<TorusSurfaceFitter>());
  int complexity = 0;
  const double rmsLimit = std::max(8 * r.FittingTolerance, .01 * r.MedianEdgeLength),
      maxLimit = std::max(20 * r.FittingTolerance, .03 * r.MedianEdgeLength),
      normalLimit = std::max(12 * r.AngularTolerance, .12);
  for (auto &f : fitters) {
    ++complexity;
    if (!f->fit(mesh, triangles)) continue;
    if (!std::isfinite(f->computeRmsError()) || !std::isfinite(f->computeMaxError()) ||
        !std::isfinite(f->computeNormalError()) || f->computeRmsError() > rmsLimit ||
        f->computeMaxError() > maxLimit || f->computeNormalError() > normalLimit) continue;
    const double normalized = f->computeRmsError() / std::max(r.FittingTolerance, 1e-30) +
        .35 * f->computeMaxError() / std::max(r.FittingTolerance, 1e-30) +
        .25 * f->computeNormalError() / std::max(r.AngularTolerance, 1e-9);
    const double score = normalized + penalty * complexity;
    if (score < best.Score)
      best = {f->getType(), f->computeRmsError(), f->computeMaxError(),
              f->computeNormalError(), score, f->getParameters()};
    // Exact lower-complexity models need no expensive torus search. This is
    // intentionally much tighter than the ordinary acceptance tolerance.
    if (f->computeRmsError() < .02 * r.FittingTolerance &&
        f->computeMaxError() < .05 * r.FittingTolerance && f->computeNormalError() < .06)
      return best;
  }
  if (best.Type != PatchSurfaceType::Unknown) return best;
  FreeformSurfaceFitter freeform;
  if (!freeform.fit(mesh, triangles)) return best;
  return {PatchSurfaceType::Freeform, freeform.computeRmsError(), freeform.computeMaxError(),
          freeform.computeNormalError(),
          10.0 + freeform.computeRmsError() / std::max(r.FittingTolerance, 1e-30), {}};
}
std::vector<SurfaceFitResult> SurfaceModelSelector::fitBestBatch(
    const MeshTopology &mesh, const std::vector<std::vector<int>> &supports,
    const MeshResolutionInfo &r, double penalty) {
  std::vector<SurfaceFitResult> results(supports.size());
  if (!CudaAnalyticFitAvailable()) {
    for (size_t i = 0; i < supports.size(); ++i)
      results[i] = fitBest(mesh, supports[i], r, penalty);
    return results;
  }
  const double rmsLimit = std::max(8 * r.FittingTolerance, .01 * r.MedianEdgeLength);
  const double maxLimit = std::max(20 * r.FittingTolerance, .03 * r.MedianEdgeLength);
  const double normalLimit = std::max(12 * r.AngularTolerance, .12);
  const auto makeFitter = [](int stage) -> std::unique_ptr<ISurfaceFitter> {
    switch (stage) {
    case 0: return std::make_unique<PlaneSurfaceFitter>();
    case 1: return std::make_unique<CylinderSurfaceFitter>();
    case 2: return std::make_unique<ConeSurfaceFitter>();
    case 3: return std::make_unique<SphereSurfaceFitter>();
    default: return std::make_unique<TorusSurfaceFitter>();
    }
  };
  std::vector<unsigned char> finished(supports.size(), 0);
  // Model order, scoring, strict tie-breaking and early exit match fitBest.
  // No speculative higher-complexity model is run after an exact fit.
  CudaAnalyticSeedPrefetch prepared("independent component fits", false);
  for (size_t begin = 0; begin < supports.size(); begin += 8) {
    const size_t end = std::min(supports.size(), begin + 8);
    for (int stage = 0; stage < 5; ++stage) {
      std::vector<std::unique_ptr<ISurfaceFitter>> fitters(end - begin);
      std::vector<unsigned char> deferred(end - begin, 0), fitted(end - begin, 0);
      const bool collect = stage > 0 && prepared.enabled();
      if (collect) prepared.begin();
      for (size_t i = begin; i < end; ++i) {
        if (finished[i]) continue;
        auto &f = fitters[i - begin];
        f = makeFitter(stage);
        try { fitted[i - begin] = f->fit(mesh, supports[i]); }
        catch (const AnalyticSeedDeferred &) { deferred[i - begin] = 1; }
      }
      if (collect) prepared.flush();
      for (size_t i = begin; i < end; ++i) {
        if (finished[i]) continue;
        auto &f = fitters[i - begin];
        if (deferred[i - begin]) fitted[i - begin] = f->fit(mesh, supports[i]);
        if (!fitted[i - begin]) continue;
        const double rms = f->computeRmsError(), maximum = f->computeMaxError();
        const double normal = f->computeNormalError();
        if (!std::isfinite(rms) || !std::isfinite(maximum) || !std::isfinite(normal) ||
            rms > rmsLimit || maximum > maxLimit || normal > normalLimit) continue;
        const double score = rms / std::max(r.FittingTolerance, 1e-30) +
            .35 * maximum / std::max(r.FittingTolerance, 1e-30) +
            .25 * normal / std::max(r.AngularTolerance, 1e-9) + penalty * (stage + 1);
        if (score < results[i].Score)
          results[i] = {f->getType(), rms, maximum, normal, score, f->getParameters()};
        finished[i] = rms < .02 * r.FittingTolerance &&
                      maximum < .05 * r.FittingTolerance && normal < .06;
      }
    }
    for (size_t i = begin; i < end; ++i) {
      if (results[i].Type != PatchSurfaceType::Unknown) continue;
      FreeformSurfaceFitter freeform;
      if (freeform.fit(mesh, supports[i]))
        results[i] = {PatchSurfaceType::Freeform, freeform.computeRmsError(),
            freeform.computeMaxError(), freeform.computeNormalError(),
            10.0 + freeform.computeRmsError() / std::max(r.FittingTolerance, 1e-30), {}};
    }
  }
  return results;
}
} // namespace CadMesh
