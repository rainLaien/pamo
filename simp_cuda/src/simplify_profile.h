#pragma once
#include <cuda_runtime.h>
#include <array>
#include <stdexcept>
#include <vector>

namespace cusimp_free {
enum class ProfileStage { InputTransfer, BuildVertexFace, BuildEdge,
    ComputeFaceQuadric, ComputeVertexQuadric, ComputeEdgeCost, PropagateCost,
    Collapse, BuildBvh, Intersection, Undo, Compact, Export, Count };

// Host-owned event recorder. CUSimp_Free carries only a pointer to this object
// so copying the solver into kernel arguments never copies a host container.
struct SimplifyProfile {
    struct Interval { ProfileStage stage; cudaEvent_t start{}, end{}; };
    bool enabled = false;
    std::vector<Interval> intervals;
    std::array<float, static_cast<size_t>(ProfileStage::Count)> milliseconds{};
    int inputVertices = 0, inputFaces = 0, edges = 0, collapsed = 0, undone = 0, undoRounds = 0;
    bool open = false;

    static void check(cudaError_t error) {
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
    void clear() {
        for (auto& interval : intervals) {
            cudaEventDestroy(interval.start);
            cudaEventDestroy(interval.end);
        }
        intervals.clear();
        milliseconds.fill(0.f);
        open = false;
        inputVertices = inputFaces = edges = collapsed = undone = undoRounds = 0;
    }
    ~SimplifyProfile() { clear(); }
    void mark(ProfileStage stage, cudaStream_t stream = nullptr) {
        if (!enabled) return;
        if (open) check(cudaEventRecord(intervals.back().end, stream));
        Interval interval{stage};
        check(cudaEventCreate(&interval.start));
        auto error = cudaEventCreate(&interval.end);
        if (error != cudaSuccess) { cudaEventDestroy(interval.start); check(error); }
        intervals.push_back(interval);
        check(cudaEventRecord(intervals.back().start, stream));
        open = true;
    }
    void finish(cudaStream_t stream = nullptr) {
        if (!enabled || !open) return;
        check(cudaEventRecord(intervals.back().end, stream));
        open = false;
        // Profiling only: no new synchronization is introduced when disabled.
        for (const auto& interval : intervals) {
            check(cudaEventSynchronize(interval.end));
            float elapsed = 0.f;
            check(cudaEventElapsedTime(&elapsed, interval.start, interval.end));
            milliseconds[static_cast<size_t>(interval.stage)] += elapsed;
        }
    }
};
}
