#include "CadMesh/CudaAnalyticPrefetch.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace CadMesh {
namespace {
using Samples = std::vector<std::array<double, 4>>;
using Parameters = std::vector<std::array<double, 8>>;
thread_local const char *FitBranch = "unscoped";

std::size_t InputHash(PatchSurfaceType type, const Samples& samples,
                      const Parameters& parameters, int iterations) {
  std::uint64_t hash = 1469598103934665603ull;
  const auto add = [&](std::uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
  };
  add(static_cast<std::uint64_t>(type));
  add(static_cast<std::uint64_t>(iterations));
  add(samples.size());
  add(parameters.size());
  const auto values = [&](const auto& records) {
    for (const auto& record : records)
      for (double value : record) {
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        add(bits);
      }
  };
  values(samples);
  values(parameters);
  return static_cast<std::size_t>(hash);
}

template <typename Record>
bool ExactRecords(const std::vector<Record>& first, const std::vector<Record>& second) {
  return first.size() == second.size() &&
         (first.empty() || std::memcmp(first.data(), second.data(),
                                      first.size() * sizeof(Record)) == 0);
}

struct Entry {
  std::size_t Hash = 0;
  PatchSurfaceType Type = PatchSurfaceType::Unknown;
  int Iterations = 0;
  Samples Points;
  Parameters Initial, Refined;
  std::vector<unsigned char> Valid;
  bool Ready = false, Used = false;
  std::size_t bytes() const {
    return Points.size() * sizeof(Points[0]) +
           2 * Initial.size() * sizeof(Initial[0]) + Initial.size();
  }
};
} // namespace

struct CudaAnalyticSeedPrefetch::Impl {
  static thread_local Impl* Active;
  Impl* Previous = nullptr;
  std::string Stage;
  bool Verbose = false, Collecting = false;
  std::size_t PendingSeeds = 0, CachedSeeds = 0, CachedBytes = 0;
  std::size_t QueuedJobs = 0, QueuedSeeds = 0, CompletedJobs = 0;
  std::size_t ReusedJobs = 0, Replays = 0, DirectCalls = 0, Flushes = 0;
  std::size_t ReusedSeeds = 0, MaximumBatchSeeds = 0;
  std::unordered_map<std::string, std::size_t> DirectBranches;
  std::unordered_multimap<std::size_t, std::shared_ptr<Entry>> Entries;
  std::vector<std::shared_ptr<Entry>> Pending;
  std::deque<std::shared_ptr<Entry>> Oldest;

  Impl(const char* stage, bool verbose) : Previous(Active), Stage(stage), Verbose(verbose) {
    Active = this;
  }
  ~Impl() {
    Active = Previous;
    if (Verbose && (QueuedJobs || DirectCalls))
      std::clog << "[CadMesh] analytic CUDA prefetch " << Stage
                << ": flushes=" << Flushes << ", queued seeds=" << QueuedSeeds
                << ", queued jobs=" << QueuedJobs << ", reused jobs=" << ReusedJobs
                << ", exact replays=" << Replays << ", direct misses=" << DirectCalls
                << ", unused jobs=" << (CompletedJobs - ReusedJobs)
                << ", used seeds=" << ReusedSeeds
                << ", mean batch seeds=" << (Flushes ? double(QueuedSeeds) / Flushes : 0)
                << ", max batch seeds=" << MaximumBatchSeeds << '\n';
    if (Verbose)
      for (const auto &branch : DirectBranches)
        std::clog << "[CadMesh] analytic CUDA direct misses: stage=" << Stage
                  << ", branch=" << branch.first << ", calls=" << branch.second << '\n';
  }
  std::shared_ptr<Entry> find(std::size_t hash, PatchSurfaceType type,
                              const Samples& samples, const Parameters& initial,
                              int iterations) const {
    const auto range = Entries.equal_range(hash);
    for (auto it = range.first; it != range.second; ++it) {
      const auto& entry = it->second;
      if (entry->Type == type && entry->Iterations == iterations &&
          ExactRecords(entry->Points, samples) && ExactRecords(entry->Initial, initial))
        return entry;
    }
    return {};
  }
  void erase(const std::shared_ptr<Entry>& entry) {
    const auto range = Entries.equal_range(entry->Hash);
    for (auto it = range.first; it != range.second; ++it)
      if (it->second == entry) {
        Entries.erase(it);
        return;
      }
  }
  void trim() {
    constexpr std::size_t maximumSeeds = 4096;
    constexpr std::size_t maximumBytes = 64 * 1024 * 1024;
    while (!Oldest.empty() && (CachedSeeds > maximumSeeds || CachedBytes > maximumBytes)) {
      auto entry = Oldest.front();
      Oldest.pop_front();
      CachedSeeds -= entry->Initial.size();
      CachedBytes -= entry->bytes();
      erase(entry);
    }
  }
};

thread_local CudaAnalyticSeedPrefetch::Impl* CudaAnalyticSeedPrefetch::Impl::Active = nullptr;

CudaAnalyticFitContext::CudaAnalyticFitContext(const char *branch)
    : Previous(FitBranch) { if (branch) FitBranch = branch; }
CudaAnalyticFitContext::~CudaAnalyticFitContext() { FitBranch = Previous; }

CudaAnalyticSeedPrefetch::CudaAnalyticSeedPrefetch(const char* stage, bool verbose)
    : mImpl(std::make_unique<Impl>(stage, verbose)) {}
CudaAnalyticSeedPrefetch::~CudaAnalyticSeedPrefetch() = default;

bool CudaAnalyticSeedPrefetch::enabled() const { return CudaAnalyticFitAvailable(); }

void CudaAnalyticSeedPrefetch::begin() {
  if (mImpl->Collecting || !mImpl->Pending.empty())
    throw std::logic_error("Nested analytic seed collection is unsupported");
  mImpl->Collecting = enabled();
}

std::size_t CudaAnalyticSeedPrefetch::pendingSeeds() const { return mImpl->PendingSeeds; }

void CudaAnalyticSeedPrefetch::flush() {
  mImpl->Collecting = false;
  if (mImpl->Pending.empty()) return;
  std::vector<AnalyticSeedBatch> batches;
  batches.reserve(mImpl->Pending.size());
  for (const auto& entry : mImpl->Pending) {
    AnalyticSeedBatch batch;
    batch.Type = entry->Type;
    batch.Samples = entry->Points;
    batch.Parameters = entry->Initial;
    batch.Iterations = entry->Iterations;
    batches.push_back(std::move(batch));
  }
  const bool onCuda = RefineAnalyticSeedBatchCuda(batches);
  mImpl->MaximumBatchSeeds = std::max(mImpl->MaximumBatchSeeds, mImpl->PendingSeeds);
  ++mImpl->Flushes;
  for (std::size_t i = 0; i < mImpl->Pending.size(); ++i) {
    const auto& entry = mImpl->Pending[i];
    if (onCuda && batches[i].Parameters.size() == entry->Initial.size() &&
        batches[i].Valid.size() == entry->Initial.size()) {
      entry->Refined = std::move(batches[i].Parameters);
      entry->Valid = std::move(batches[i].Valid);
      entry->Ready = true;
      mImpl->Oldest.push_back(entry);
      mImpl->CachedSeeds += entry->Initial.size();
      mImpl->CachedBytes += entry->bytes();
      ++mImpl->CompletedJobs;
    } else {
      mImpl->erase(entry);
    }
  }
  mImpl->Pending.clear();
  mImpl->PendingSeeds = 0;
  mImpl->trim();
}

bool RefineAnalyticSeedsPrepared(PatchSurfaceType type, const Samples& samples,
    Parameters& parameters, std::vector<unsigned char>& valid, int iterations) {
  auto* cache = CudaAnalyticSeedPrefetch::Impl::Active;
  if (!cache || !CudaAnalyticFitAvailable() || parameters.empty())
    return RefineAnalyticSeedsCuda(type, samples, parameters, valid, iterations);
  const auto hash = InputHash(type, samples, parameters, iterations);
  auto entry = cache->find(hash, type, samples, parameters, iterations);
  if (cache->Collecting) {
    if (!entry) {
      entry = std::make_shared<Entry>();
      entry->Hash = hash;
      entry->Type = type;
      entry->Iterations = iterations;
      entry->Points = samples;
      entry->Initial = parameters;
      cache->Pending.push_back(entry);
      cache->Entries.emplace(hash, entry);
      cache->PendingSeeds += parameters.size();
      cache->QueuedSeeds += parameters.size();
      ++cache->QueuedJobs;
    }
    // Even a cache hit stops collection here: Errors and Certify belong to
    // normal execution with the freshly regenerated, current-owner support.
    throw AnalyticSeedDeferred{};
  }
  if (entry && entry->Ready) {
    parameters = entry->Refined;
    valid = entry->Valid;
    ++cache->Replays;
    if (!entry->Used) {
      entry->Used = true;
      ++cache->ReusedJobs;
      cache->ReusedSeeds += entry->Initial.size();
    }
    return true;
  }
  ++cache->DirectCalls;
  ++cache->DirectBranches[FitBranch];
  return RefineAnalyticSeedsCuda(type, samples, parameters, valid, iterations);
}

} // namespace CadMesh
