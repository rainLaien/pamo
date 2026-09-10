#pragma once

namespace CadMesh {

// Compiled by NVRTC as CUDA C++14 with precise double arithmetic and
// --fmad=false. This source deliberately has no Eigen, Warp, or host headers.
// Separate literals avoid MSVC C2026. The host joins these at runtime;
// adjacent literals would still be concatenated by the C++ compiler.
inline constexpr const char* AnalyticSeedKernelSourceParts[] = {R"cuda(
namespace cadmesh_seed_detail {

constexpr int Threads = 128;
constexpr int Warps = Threads / 32;
constexpr int MaxSamples = 768;
constexpr int MaxDimension = 8;
constexpr int MaxStatistics = 36 + MaxDimension;
constexpr double Pi = 3.14159265358979323846;

struct Model {
  double origin[3];
  double axis[3];
  double primary, secondary, sine, cosine;
};

struct SeedState {
  // Cache the identical weighted sample cloud once per seed block. Slot 3 is
  // sqrt(weight), matching Refine's evaluate lambda rather than raw weight.
  double samples[MaxSamples * 4];
  double errors[MaxSamples];
  double nextErrors[MaxSamples];
  double parameters[MaxDimension];
  double proposal[MaxDimension];
  double step[MaxDimension];
  double delta[MaxDimension];
  double hessian[MaxDimension * MaxDimension];
  double gradient[MaxDimension];
  Model models[MaxDimension + 1];
  double partial[MaxStatistics * Warps];
  double cost, evaluatedCost, damping, stepNorm;
  int failed, done, accepted, proposalValid, evaluationValid;
};

// Around 39 KiB, including the full 768-sample cloud and both residual
// vectors; no dynamic shared memory or double-precision atomics are needed.
static_assert(sizeof(SeedState) < 48 * 1024, "Seed block exceeds shared memory budget");

__device__ __forceinline__ double maximum(double first, double second) {
  return first < second ? second : first;
}

__device__ __forceinline__ double norm3(double x, double y, double z) {
  return sqrt(x * x + y * y + z * z);
}

__device__ __forceinline__ bool parameters_finite(const double* p, int dimension) {
  for (int j = 0; j < dimension; ++j)
    if (!isfinite(p[j])) return false;
  return true;
}

__device__ __forceinline__ bool parameters_valid(const double* p, int type,
                                                int dimension) {
  if (!parameters_finite(p, dimension) ||
      !(norm3(p[0], p[1], p[2]) < 1e4)) return false;
  if (type == 4)
    return p[3] > 1e-7 && p[3] < 1e4;
  const double axisLength = norm3(p[3], p[4], p[5]);
  if (!(axisLength > .1 && axisLength < 10)) return false;
  if (type == 2)
    return p[6] > 1e-7 && p[6] < 1e4;
  if (type == 3)
    return p[6] > .00872664626 && p[6] < Pi / 2 - .00872664626;
  if (type == 5)
    return p[7] > 1e-4 && p[6] > 1.005 * p[7] && p[6] < 100;
  return false;
}

__device__ __forceinline__ void prepare_model(const double* p, int type,
                                             Model& model) {
  for (int j = 0; j < 3; ++j) model.origin[j] = p[j];
  model.secondary = model.sine = model.cosine = 0;
  if (type == 4) {
    model.axis[0] = model.axis[1] = model.axis[2] = 0;
    model.primary = p[3];
    return;
  }
  const double axisLength = norm3(p[3], p[4], p[5]);
  for (int j = 0; j < 3; ++j) model.axis[j] = p[j + 3] / axisLength;
  model.primary = p[6];
  if (type == 3) {
    model.sine = sin(p[6]);
    model.cosine = cos(p[6]);
  } else if (type == 5) {
    model.secondary = p[7];
  }
}

__device__ __forceinline__ double residual(const double* point, int type,
                                          const Model& model) {
  const double dx = point[0] - model.origin[0];
  const double dy = point[1] - model.origin[1];
  const double dz = point[2] - model.origin[2];
  if (type == 4) return norm3(dx, dy, dz) - model.primary;
  const double z = dx * model.axis[0] + dy * model.axis[1] + dz * model.axis[2];
  const double rx = dx - z * model.axis[0];
  const double ry = dy - z * model.axis[1];
  const double rz = dz - z * model.axis[2];
  const double rho = norm3(rx, ry, rz);
  if (type == 2) return rho - model.primary;
  if (type == 3) {
    // The same selected-nappe/apex residual as SurfaceFitting.cpp, including
    // the <= 0 branch. This is not the segmentation distance predicate.
    if (z * model.cosine + rho * model.sine <= 0) return norm3(dx, dy, dz);
    return rho * model.cosine - z * model.sine;
  }
  return hypot(rho - model.primary, z) - model.secondary;
}

__device__ __forceinline__ double warp_sum(double value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// Every thread in the 128-thread block must enter this function. Each sample
// belongs to exactly one thread; the reduction changes addition order only.
template<int Type>
__device__ __forceinline__ void evaluate(SeedState& state, int sampleCount,
                         double* errors) {
  const int thread = int(threadIdx.x);
  double squared = 0;
  int invalid = 0;
  for (int i = thread; i < sampleCount; i += Threads) {
    const double* point = state.samples + 4 * i;
    const double error = point[3] * residual(point, Type, state.models[MaxDimension]);
    errors[i] = error;
    if (!isfinite(error)) invalid = 1;
    squared += error * error;
  }
  squared = warp_sum(squared);
  if ((thread & 31) == 0) state.partial[thread >> 5] = squared;
  const int anyInvalid = __syncthreads_or(invalid);
  if (thread == 0) {
    double cost = 0;
    for (int warp = 0; warp < Warps; ++warp) cost += state.partial[warp];
    state.evaluatedCost = cost;
    state.evaluationValid = !anyInvalid && isfinite(cost);
  }
  __syncthreads();
}

// The damped normal system is at most 8 x 8. Cholesky failure requests the
// original CPU LDLT solver, rather than accepting an unreliable GPU update.
template<int dimension>
__device__ __forceinline__ bool solve_damped(SeedState& state) {
  double lower[MaxDimension * MaxDimension];
  double forward[MaxDimension];
  for (int row = 0; row < dimension; ++row) {
    for (int column = 0; column <= row; ++column) {
      double value = state.hessian[row * MaxDimension + column];
      if (row == column)
        value += state.damping * maximum(1e-4, value);
      for (int k = 0; k < column; ++k)
        value -= lower[row * MaxDimension + k] * lower[column * MaxDimension + k];
      if (row == column) {
        if (!(value > 0) || !isfinite(value)) return false;
        lower[row * MaxDimension + column] = sqrt(value);
      } else {
        value /= lower[column * MaxDimension + column];
        if (!isfinite(value)) return false;
        lower[row * MaxDimension + column] = value;
      }
    }
    double value = -state.gradient[row];
    for (int k = 0; k < row; ++k)
      value -= lower[row * MaxDimension + k] * forward[k];
    forward[row] = value / lower[row * MaxDimension + row];
    if (!isfinite(forward[row])) return false;
  }
  for (int row = dimension - 1; row >= 0; --row) {
    double value = forward[row];
    for (int k = row + 1; k < dimension; ++k)
      value -= lower[k * MaxDimension + row] * state.step[k];
    state.step[row] = value / lower[row * MaxDimension + row];
    if (!isfinite(state.step[row])) return false;
  }
  return true;
}

} // namespace cadmesh_seed_detail
)cuda", R"cuda(

// Compile each model/parameter dimension separately, but dispatch per block
// within one launch so mixed-model batches retain their original concurrency.
template<int type, int dimension>
__device__ __forceinline__ void refine_analytic_seed(
    const double* samples, const int* descriptors, double* parameters, int* valid,
    cadmesh_seed_detail::SeedState& state) {
  using namespace cadmesh_seed_detail;
  const int seed = int(blockIdx.x);
  const int thread = int(threadIdx.x);
  // Each block can fit a different model on a different normalized cloud.
  // Several initial guesses for one cloud share its uploaded sample range.
  const int* descriptor = descriptors + seed * 5;
  samples += descriptor[0] * 4;
  const int sampleCount = descriptor[1];
  const int iterations = descriptor[4];
  if (blockDim.x != Threads || blockDim.y != 1 || blockDim.z != 1 ||
      sampleCount <= 0 || sampleCount > MaxSamples ||
      descriptor[2] != type || descriptor[3] != dimension) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) valid[seed] = 0;
    return;
  }
  for (int i = thread; i < sampleCount; i += Threads) {
    for (int axis = 0; axis < 3; ++axis)
      state.samples[4 * i + axis] = samples[4 * i + axis];
    state.samples[4 * i + 3] = sqrt(samples[4 * i + 3]);
  }
  if (thread == 0) {
    // Never overwrite the input seed until the complete solve succeeds, so
    // valid == 0 leaves an exact original seed for the CPU fallback.
    valid[seed] = 0;
    for (int j = 0; j < MaxDimension; ++j)
      state.parameters[j] = j < dimension ? parameters[seed * MaxDimension + j] : 0;
    state.failed = !parameters_valid(state.parameters, type, dimension);
    state.done = state.accepted = 0;
    if (!state.failed)
      prepare_model(state.parameters, type, state.models[MaxDimension]);
  }
  __syncthreads();
  if (state.failed) return;
  evaluate<type>(state, sampleCount, state.errors);
  if (thread == 0) {
    state.failed = !state.evaluationValid;
    state.cost = state.evaluatedCost;
    state.damping = 1e-5;
  }
  __syncthreads();
  if (state.failed) return;

  constexpr int upperCount = dimension * (dimension + 1) / 2;
  constexpr int statisticCount = upperCount + dimension;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (thread < dimension) {
        const int j = thread;
        double probe[dimension];
        for (int k = 0; k < dimension; ++k) probe[k] = state.parameters[k];
        state.delta[j] = 1e-6 * maximum(1.0, fabs(state.parameters[j]));
        probe[j] += state.delta[j];
        // CPU Refine evaluates finite-difference probes without invoking
        // valid(probe); keep that behavior at radius/angle bounds.
        prepare_model(probe, type, state.models[j]);
    }
    if (thread == 0) state.accepted = 0;
    __syncthreads();

)cuda", R"cuda(    double statistics[statisticCount];
    #pragma unroll
    for (int j = 0; j < statisticCount; ++j) statistics[j] = 0;
    int invalid = 0;
    for (int i = thread; i < sampleCount; i += Threads) {
      const double* point = state.samples + 4 * i;
      const double error = state.errors[i];
      double jacobian[dimension];
      double torusDx = 0, torusDy = 0, torusDz = 0;
      double torusZ = 0, torusRho = 0;
      if (type == 5) {
        // Radius probes 6 and 7 have identical center and normalized axis.
        // Use probe 6, not models[MaxDimension]: the latter may still contain
        // a rejected trial from the previous iteration.
        const Model& radiusProbe = state.models[6];
        torusDx = point[0] - radiusProbe.origin[0];
        torusDy = point[1] - radiusProbe.origin[1];
        torusDz = point[2] - radiusProbe.origin[2];
        torusZ = torusDx * radiusProbe.axis[0] + torusDy * radiusProbe.axis[1] +
                 torusDz * radiusProbe.axis[2];
        const double rx = torusDx - torusZ * radiusProbe.axis[0];
        const double ry = torusDy - torusZ * radiusProbe.axis[1];
        const double rz = torusDz - torusZ * radiusProbe.axis[2];
        torusRho = norm3(rx, ry, rz);
      }
      #pragma unroll
      for (int j = 0; j < dimension; ++j) {
        double raw;
        if (type == 5) {
          const Model& model = state.models[j];
          double z = torusZ, rho = torusRho;
          if (j < 6) {
            // Center probes alter exactly one coordinate; axis probes leave
            // all three offsets unchanged. Preserve the original subtraction,
            // dot-product and norm expressions instead of algebraic rewrites.
            const double dx = j == 0 ? point[0] - model.origin[0] : torusDx;
            const double dy = j == 1 ? point[1] - model.origin[1] : torusDy;
            const double dz = j == 2 ? point[2] - model.origin[2] : torusDz;
            z = dx * model.axis[0] + dy * model.axis[1] + dz * model.axis[2];
            const double rx = dx - z * model.axis[0];
            const double ry = dy - z * model.axis[1];
            const double rz = dz - z * model.axis[2];
            rho = norm3(rx, ry, rz);
          }
          raw = hypot(rho - model.primary, z) - model.secondary;
        } else {
          raw = residual(point, type, state.models[j]);
        }
        const double other = point[3] * raw;
        const double derivative = (other - error) / state.delta[j];
        if (!isfinite(other) || !isfinite(derivative)) {
          invalid = 1;
          jacobian[j] = 0;
        } else {
          jacobian[j] = derivative;
        }
      }
      int entry = 0;
      #pragma unroll
      for (int row = 0; row < dimension; ++row) {
        #pragma unroll
        for (int column = row; column < dimension; ++column)
          statistics[entry++] += jacobian[row] * jacobian[column];
        statistics[upperCount + row] += jacobian[row] * error;
      }
    }
    #pragma unroll
    for (int j = 0; j < statisticCount; ++j) {
      const double value = warp_sum(statistics[j]);
      if ((thread & 31) == 0)
        state.partial[j * Warps + (thread >> 5)] = value;
    }
    __syncthreads();
    // Assign one statistic per thread. Each still sums the four warp results
    // in exactly the original order; only independent entries run in parallel.
    if (thread < statisticCount) {
      double value = 0;
      for (int warp = 0; warp < Warps; ++warp)
        value += state.partial[thread * Warps + warp];
      if (!isfinite(value)) invalid = 1;
      if (thread < upperCount) {
        int row = 0, columnOffset = thread;
        while (columnOffset >= dimension - row) {
          columnOffset -= dimension - row;
          ++row;
        }
        const int column = row + columnOffset;
        state.hessian[row * MaxDimension + column] = value;
        state.hessian[column * MaxDimension + row] = value;
      } else {
        state.gradient[thread - upperCount] = value;
      }
    }
    const int anyInvalid = __syncthreads_or(invalid);
    if (thread == 0) {
      state.failed = anyInvalid;
      double maxGradient = 0;
      for (int j = 0; j < dimension; ++j)
        maxGradient = maximum(maxGradient, fabs(state.gradient[j]));
      state.done = maxGradient < 1e-13 || state.cost < 1e-24;
    }
    __syncthreads();
    if (state.failed || state.done) break;
    // A failing first solve may change failed. Do not let that update race
    // another warp still reading the Jacobian phase's completion decision.
    __syncthreads();

    for (int trial = 0; trial < 7; ++trial) {
      if (thread == 0) {
        state.failed = !solve_damped<dimension>(state);
        state.proposalValid = 0;
        if (!state.failed) {
          double stepSquared = 0;
          for (int j = 0; j < dimension; ++j) {
            state.proposal[j] = state.parameters[j] + state.step[j];
            stepSquared += state.step[j] * state.step[j];
          }
          state.stepNorm = sqrt(stepSquared);
          state.proposalValid = parameters_valid(state.proposal, type, dimension);
          if (state.proposalValid)
            prepare_model(state.proposal, type, state.models[MaxDimension]);
          else
            state.damping *= 10;
        }
      }
      __syncthreads();
      if (state.failed) break;
      if (!state.proposalValid) {
        // All warps must consume this decision before thread 0 overwrites it
        // while constructing the next trial's proposal.
        __syncthreads();
        continue;
      }

      evaluate<type>(state, sampleCount, state.nextErrors);
      if (thread == 0) {
        if (state.evaluationValid && state.evaluatedCost < state.cost) {
          const double improvement = state.cost - state.evaluatedCost;
          for (int j = 0; j < dimension; ++j)
            state.parameters[j] = state.proposal[j];
          state.cost = state.evaluatedCost;
          state.damping = maximum(1e-12, state.damping * .25);
          state.accepted = 1;
          state.done = state.stepNorm < 1e-9 ||
                       improvement < 1e-15 * maximum(1.0, state.cost);
        } else {
          state.damping *= 10;
        }
      }
      __syncthreads();
      if (state.accepted) {
        for (int i = thread; i < sampleCount; i += Threads)
          state.errors[i] = state.nextErrors[i];
        __syncthreads();
        break;
      }
    }
    if (state.failed || state.done || !state.accepted) break;
    // The next iteration resets accepted. Keep that write after every warp
    // has consumed the current iteration's termination decision.
    __syncthreads();
  }
  if (thread == 0 && !state.failed && parameters_finite(state.parameters, dimension)) {
    for (int j = 0; j < dimension; ++j)
      parameters[seed * MaxDimension + j] = state.parameters[j];
    valid[seed] = 1;
  }
}
extern "C" __global__ void refine_analytic_seeds(
    const double* samples, const int* descriptors, double* parameters, int* valid) {
  __shared__ cadmesh_seed_detail::SeedState state;
  // The descriptor is block-uniform; all threads take the same branch and
  // participate in the same barriers. No host-side reordering is required.
  switch (descriptors[int(blockIdx.x) * 5 + 2]) {
  case 2: refine_analytic_seed<2, 7>(samples, descriptors, parameters, valid, state); break;
  case 3: refine_analytic_seed<3, 7>(samples, descriptors, parameters, valid, state); break;
  case 4: refine_analytic_seed<4, 4>(samples, descriptors, parameters, valid, state); break;
  case 5: refine_analytic_seed<5, 8>(samples, descriptors, parameters, valid, state); break;
  default:
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) valid[blockIdx.x] = 0;
  }
}
)cuda"};

} // namespace CadMesh
