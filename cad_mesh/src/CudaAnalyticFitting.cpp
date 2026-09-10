#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "CadMesh/CudaAnalyticFitting.h"
#include "CudaAnalyticKernels.h"
#include "CudaRemeshKernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace CadMesh {
namespace {
using Clock = std::chrono::steady_clock;
double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

int ParameterDimension(PatchSurfaceType type) {
  switch (type) {
  case PatchSurfaceType::Cylinder:
  case PatchSurfaceType::Cone: return 7;
  case PatchSurfaceType::Sphere: return 4;
  case PatchSurfaceType::Torus: return 8;
  default: throw std::runtime_error("CUDA analytic refinement received an unsupported surface type");
  }
}

struct RuntimeTelemetry {
  std::uint64_t HostToDeviceBytes = 0, DeviceToHostBytes = 0;
  std::size_t SampleUploadHits = 0, SampleUploadMisses = 0;
  double HostPackingSeconds = 0, SampleCacheSeconds = 0;
  double HostToDeviceSeconds = 0, ReadbackWaitSeconds = 0;
  double KernelMilliseconds = 0;
  std::array<double, 6> TypeKernelMilliseconds{};
  std::array<std::size_t, 6> TypeLaunches{}, TypeSeeds{};
  std::array<std::uint64_t, 6> TypeSamples{};
};

#ifdef _WIN32
// The stable CUDA C ABI is loaded at runtime. In particular this translation
// unit does not require nvcc, CUDA SDK headers, or an MSVC import library.
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;
using CUevent = void *;
using NvrtcProgram = void *;
static_assert(sizeof(void *) == 8, "CUDA analytic fitting requires a 64-bit build");
static_assert(sizeof(int) == 4, "CUDA analytic validity uses 32-bit integers");
static_assert(sizeof(std::array<double, 4>) == 4 * sizeof(double), "CUDA sample packing mismatch");
static_assert(sizeof(std::array<double, 8>) == 8 * sizeof(double), "CUDA seed packing mismatch");

std::wstring EnvironmentValue(const wchar_t *name) {
  const DWORD length = GetEnvironmentVariableW(name, nullptr, 0);
  if (!length) return {};
  std::vector<wchar_t> value(length);
  const DWORD written = GetEnvironmentVariableW(name, value.data(), length);
  return written && written < length ? std::wstring(value.data(), written) : std::wstring();
}

std::filesystem::path ExecutableDirectory() {
  std::vector<wchar_t> path(32768);
  const DWORD count = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
  if (!count || count >= path.size()) return {};
  return std::filesystem::path(std::wstring(path.data(), count)).parent_path();
}

class DynamicLibrary {
public:
  DynamicLibrary() = default;
  explicit DynamicLibrary(const std::filesystem::path &path) { load(path); }
  ~DynamicLibrary() { if (mHandle) FreeLibrary(mHandle); }
  DynamicLibrary(const DynamicLibrary &) = delete;
  DynamicLibrary &operator=(const DynamicLibrary &) = delete;
  DynamicLibrary(DynamicLibrary &&other) noexcept : mHandle(other.mHandle) {
    other.mHandle = nullptr;
  }
  DynamicLibrary &operator=(DynamicLibrary &&other) noexcept {
    if (this != &other) {
      if (mHandle) FreeLibrary(mHandle);
      mHandle = other.mHandle;
      other.mHandle = nullptr;
    }
    return *this;
  }
  void load(const std::filesystem::path &path) {
    // Resolve dependencies beside this DLL and in the normal trusted DLL
    // directories, without changing the process working directory or PATH.
    constexpr DWORD searchFlags = 0x00000100 | 0x00001000;
    const auto absolute = std::filesystem::absolute(path);
    HMODULE handle = LoadLibraryExW(absolute.c_str(), nullptr, searchFlags);
    if (!handle) {
      std::ostringstream message;
      message << "cannot load CUDA library " << path.u8string()
              << " (Windows error " << GetLastError() << ')';
      throw std::runtime_error(message.str());
    }
    if (mHandle) FreeLibrary(mHandle);
    mHandle = handle;
  }
  template <typename Function> Function find(const char *name, bool required = true) const {
    const auto address = GetProcAddress(mHandle, name);
    if (!address && required)
      throw std::runtime_error(std::string("CUDA library is missing entry point ") + name);
    // GetProcAddress returns FARPROC, whose signature is intentionally generic.
    // Copy the representation to avoid GCC's incompatible-function-cast warning.
    Function result = nullptr;
    static_assert(sizeof(result) == sizeof(address), "Windows function pointer size mismatch");
    std::memcpy(&result, &address, sizeof(result));
    return result;
  }
private:
  HMODULE mHandle = nullptr;
};

std::vector<std::filesystem::path> FilesMatching(
    const std::filesystem::path &directory, const std::wstring &prefix) {
  std::vector<std::filesystem::path> paths;
  // Ask Windows to filter names before returning directory entries. PATH
  // can contain large system directories unrelated to CUDA.
  WIN32_FIND_DATAW entry{};
  const auto pattern = directory / (prefix + L"*.dll");
  const HANDLE search = FindFirstFileW(pattern.c_str(), &entry);
  if (search == INVALID_HANDLE_VALUE) return paths;
  struct SearchCleanup {
    HANDLE Handle;
    ~SearchCleanup() { FindClose(Handle); }
  } cleanup{search};
  do {
    const std::wstring name = entry.cFileName;
    if (name.size() > prefix.size() + 4 && name.compare(0, prefix.size(), prefix) == 0
        && name.compare(name.size() - 4, 4, L".dll") == 0
        && !(entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
      // CUDA installations also contain nvrtc64_*.alt.dll. It is not the
      // compiler entry point and must never win a wildcard search.
      const auto middle = name.substr(prefix.size(), name.size() - prefix.size() - 4);
      if (std::all_of(middle.begin(), middle.end(), [](wchar_t c) {
            return (c >= L'0' && c <= L'9') || c == L'_';
          })) paths.push_back(directory / name);
    }
  } while (FindNextFileW(search, &entry));
  std::sort(paths.rbegin(), paths.rend());
  return paths;
}

std::vector<std::filesystem::path> NvrtcCandidates() {
  std::vector<std::filesystem::path> candidates, directories;
  const auto explicitLibrary = EnvironmentValue(L"CADMESH_NVRTC_DLL");
  if (!explicitLibrary.empty()) candidates.emplace_back(explicitLibrary);
  const auto addCudaRoot = [&](const std::wstring &root) {
    if (!root.empty()) {
      directories.push_back(std::filesystem::path(root) / L"bin");
      directories.push_back(std::filesystem::path(root) / L"bin" / L"x64");
    }
  };
  addCudaRoot(EnvironmentValue(L"CUDA_PATH"));
  if (LPWCH environment = GetEnvironmentStringsW()) {
    for (const wchar_t *entry = environment; *entry; entry += std::wcslen(entry) + 1) {
      const std::wstring item(entry);
      if (item.compare(0, 11, L"CUDA_PATH_V") == 0) {
        const auto split = item.find(L'=');
        if (split != std::wstring::npos) addCudaRoot(item.substr(split + 1));
      }
    }
    FreeEnvironmentStringsW(environment);
  }
  auto ancestor = ExecutableDirectory();
  for (int level = 0; level < 5 && !ancestor.empty(); ++level) {
    directories.push_back(ancestor / L".venv" / L"Lib" / L"site-packages" / L"torch" / L"lib");
    const auto parent = ancestor.parent_path();
    if (parent == ancestor) break;
    ancestor = parent;
  }
  const std::wstring searchPath = EnvironmentValue(L"PATH");
  std::size_t start = 0;
  while (start < searchPath.size()) {
    const auto stop = searchPath.find(L';', start);
    auto part = searchPath.substr(start, stop == std::wstring::npos ? stop : stop - start);
    if (part.size() >= 2 && part.front() == L'"' && part.back() == L'"')
      part = part.substr(1, part.size() - 2);
    if (!part.empty()) directories.emplace_back(part);
    if (stop == std::wstring::npos) break;
    start = stop + 1;
  }
  const auto programFiles = EnvironmentValue(L"ProgramFiles");
  if (!programFiles.empty()) {
    const auto root = std::filesystem::path(programFiles) / L"NVIDIA GPU Computing Toolkit" / L"CUDA";
    std::error_code error;
    std::filesystem::directory_iterator it(root, error), end;
    while (!error && it != end) {
      directories.push_back(it->path() / L"bin");
      directories.push_back(it->path() / L"bin" / L"x64");
      it.increment(error);
    }
  }
  for (const auto &directory : directories) {
    for (const auto &path : FilesMatching(directory, L"nvrtc64_")) {
      if (std::find(candidates.begin(), candidates.end(), path) == candidates.end())
        candidates.push_back(path);
    }
  }
  return candidates;
}

struct DriverApi {
  CUresult (__stdcall *Init)(unsigned int) = nullptr;
  CUresult (__stdcall *DeviceGet)(CUdevice *, int) = nullptr;
  CUresult (__stdcall *DeviceGetName)(char *, int, CUdevice) = nullptr;
  CUresult (__stdcall *DeviceGetAttribute)(int *, int, CUdevice) = nullptr;
  CUresult (__stdcall *PrimaryRetain)(CUcontext *, CUdevice) = nullptr;
  CUresult (__stdcall *PrimaryRelease)(CUdevice) = nullptr;
  CUresult (__stdcall *PushCurrent)(CUcontext) = nullptr;
  CUresult (__stdcall *PopCurrent)(CUcontext *) = nullptr;
  CUresult (__stdcall *MemAlloc)(CUdeviceptr *, std::size_t) = nullptr;
  CUresult (__stdcall *MemFree)(CUdeviceptr) = nullptr;
  CUresult (__stdcall *CopyToDevice)(CUdeviceptr, const void *, std::size_t) = nullptr;
  CUresult (__stdcall *CopyToHost)(void *, CUdeviceptr, std::size_t) = nullptr;
  CUresult (__stdcall *ModuleLoad)(CUmodule *, const void *, unsigned int, int *, void **) = nullptr;
  CUresult (__stdcall *ModuleUnload)(CUmodule) = nullptr;
  CUresult (__stdcall *ModuleFunction)(CUfunction *, CUmodule, const char *) = nullptr;
  CUresult (__stdcall *Launch)(CUfunction, unsigned int, unsigned int, unsigned int,
                             unsigned int, unsigned int, unsigned int,
                             unsigned int, CUstream, void **, void **) = nullptr;
  CUresult (__stdcall *Synchronize)() = nullptr;
  CUresult (__stdcall *EventCreate)(CUevent *, unsigned int) = nullptr;
  CUresult (__stdcall *EventRecord)(CUevent, CUstream) = nullptr;
  CUresult (__stdcall *EventElapsedTime)(float *, CUevent, CUevent) = nullptr;
  CUresult (__stdcall *EventDestroy)(CUevent) = nullptr;
  CUresult (__stdcall *ErrorString)(CUresult, const char **) = nullptr;

  void load(const DynamicLibrary &library) {
#define CADMESH_DRIVER(Member, Name) Member = library.find<decltype(Member)>(Name)
    CADMESH_DRIVER(Init, "cuInit");
    CADMESH_DRIVER(DeviceGet, "cuDeviceGet");
    CADMESH_DRIVER(DeviceGetName, "cuDeviceGetName");
    CADMESH_DRIVER(DeviceGetAttribute, "cuDeviceGetAttribute");
    CADMESH_DRIVER(PrimaryRetain, "cuDevicePrimaryCtxRetain");
    CADMESH_DRIVER(PrimaryRelease, "cuDevicePrimaryCtxRelease_v2");
    CADMESH_DRIVER(PushCurrent, "cuCtxPushCurrent_v2");
    CADMESH_DRIVER(PopCurrent, "cuCtxPopCurrent_v2");
    CADMESH_DRIVER(MemAlloc, "cuMemAlloc_v2");
    CADMESH_DRIVER(MemFree, "cuMemFree_v2");
    CADMESH_DRIVER(CopyToDevice, "cuMemcpyHtoD_v2");
    CADMESH_DRIVER(CopyToHost, "cuMemcpyDtoH_v2");
    CADMESH_DRIVER(ModuleLoad, "cuModuleLoadDataEx");
    CADMESH_DRIVER(ModuleUnload, "cuModuleUnload");
    CADMESH_DRIVER(ModuleFunction, "cuModuleGetFunction");
    CADMESH_DRIVER(Launch, "cuLaunchKernel");
    CADMESH_DRIVER(Synchronize, "cuCtxSynchronize");
    CADMESH_DRIVER(EventCreate, "cuEventCreate");
    CADMESH_DRIVER(EventRecord, "cuEventRecord");
    CADMESH_DRIVER(EventElapsedTime, "cuEventElapsedTime");
    CADMESH_DRIVER(EventDestroy, "cuEventDestroy_v2");
    CADMESH_DRIVER(ErrorString, "cuGetErrorString");
#undef CADMESH_DRIVER
  }
  void check(CUresult result, const char *operation) const {
    if (result == 0) return;
    const char *description = nullptr;
    if (ErrorString) ErrorString(result, &description);
    throw std::runtime_error(std::string(operation) + " failed (CUDA "
        + std::to_string(result) + "): " + (description ? description : "unknown driver error"));
  }
};

class CurrentContext {
public:
  CurrentContext(DriverApi &driver, CUcontext context) : mDriver(driver) {
    mDriver.check(mDriver.PushCurrent(context), "cuCtxPushCurrent_v2");
  }
  ~CurrentContext() {
    CUcontext previous = nullptr;
    mDriver.PopCurrent(&previous);
  }
private:
  DriverApi &mDriver;
};

struct NvrtcApi {
  int (*Version)(int *, int *) = nullptr;
  int (*Create)(NvrtcProgram *, const char *, const char *, int,
                const char *const *, const char *const *) = nullptr;
  int (*Compile)(NvrtcProgram, int, const char *const *) = nullptr;
  int (*PtxSize)(NvrtcProgram, std::size_t *) = nullptr;
  int (*Ptx)(NvrtcProgram, char *) = nullptr;
  int (*LogSize)(NvrtcProgram, std::size_t *) = nullptr;
  int (*Log)(NvrtcProgram, char *) = nullptr;
  int (*Destroy)(NvrtcProgram *) = nullptr;
  const char *(*ErrorString)(int) = nullptr;
  int (*SupportedCount)(int *) = nullptr;
  int (*SupportedArchs)(int *) = nullptr;
  void load(const DynamicLibrary &library) {
#define CADMESH_NVRTC(Member, Name) Member = library.find<decltype(Member)>(Name)
    CADMESH_NVRTC(Version, "nvrtcVersion");
    CADMESH_NVRTC(Create, "nvrtcCreateProgram");
    CADMESH_NVRTC(Compile, "nvrtcCompileProgram");
    CADMESH_NVRTC(PtxSize, "nvrtcGetPTXSize");
    CADMESH_NVRTC(Ptx, "nvrtcGetPTX");
    CADMESH_NVRTC(LogSize, "nvrtcGetProgramLogSize");
    CADMESH_NVRTC(Log, "nvrtcGetProgramLog");
    CADMESH_NVRTC(Destroy, "nvrtcDestroyProgram");
    CADMESH_NVRTC(ErrorString, "nvrtcGetErrorString");
#undef CADMESH_NVRTC
    SupportedCount = library.find<decltype(SupportedCount)>("nvrtcGetNumSupportedArchs", false);
    SupportedArchs = library.find<decltype(SupportedArchs)>("nvrtcGetSupportedArchs", false);
  }
  void check(int result, const char *operation) const {
    if (result != 0)
      throw std::runtime_error(std::string(operation) + " failed: " + ErrorString(result));
  }
};

class CudaRuntime {
public:
  ~CudaRuntime() {
    if (mContext) {
      if (mDriver.PushCurrent && mDriver.PushCurrent(mContext) == 0) {
        if (mSamples) mDriver.MemFree(mSamples);
        if (mParameters) mDriver.MemFree(mParameters);
        if (mValid) mDriver.MemFree(mValid);
        if (mDescriptors) mDriver.MemFree(mDescriptors);
        if (mRemeshVertices) mDriver.MemFree(mRemeshVertices);
        if (mRemeshEdges) mDriver.MemFree(mRemeshEdges);
        if (mRemeshSelected) mDriver.MemFree(mRemeshSelected);
        if (mRemeshMidpoints) mDriver.MemFree(mRemeshMidpoints);
        if (mKernelStart) mDriver.EventDestroy(mKernelStart);
        if (mKernelEnd) mDriver.EventDestroy(mKernelEnd);
        for (auto event : mTypeStart) if (event) mDriver.EventDestroy(event);
        for (auto event : mTypeEnd) if (event) mDriver.EventDestroy(event);
        if (mModule) mDriver.ModuleUnload(mModule);
        CUcontext context = nullptr;
        mDriver.PopCurrent(&context);
      }
      if (mDriver.PrimaryRelease) mDriver.PrimaryRelease(mDevice);
    }
  }
  void initialize(bool verbose = false) {
    mVerbose = verbose;
    std::vector<wchar_t> system(32768);
    const UINT count = GetSystemDirectoryW(system.data(), UINT(system.size()));
    if (!count || count >= system.size()) throw std::runtime_error("cannot locate the Windows CUDA driver directory");
    mDriverLibrary.load(std::filesystem::path(std::wstring(system.data(), count)) / L"nvcuda.dll");
    mDriver.load(mDriverLibrary);
    mDriver.check(mDriver.Init(0), "cuInit");
    mDriver.check(mDriver.DeviceGet(&mDevice, 0), "cuDeviceGet");
    char deviceName[256] = {};
    mDriver.check(mDriver.DeviceGetName(deviceName, int(sizeof(deviceName)), mDevice), "cuDeviceGetName");
    deviceName[sizeof(deviceName) - 1] = '\0';
    mDeviceName = deviceName;
    mDriver.check(mDriver.PrimaryRetain(&mContext, mDevice), "cuDevicePrimaryCtxRetain");
    CurrentContext current(mDriver, mContext);

    const auto candidates = NvrtcCandidates();
    std::string lastError;
    bool loaded = false;
    for (const auto &path : candidates) {
      try {
        std::vector<DynamicLibrary> builtins;
        for (const auto &builtin : FilesMatching(path.parent_path(), L"nvrtc-builtins64_"))
          builtins.emplace_back(builtin);
        DynamicLibrary library(path);
        NvrtcApi api;
        api.load(library);
        mNvrtcLibrary = std::move(library);
        mBuiltins = std::move(builtins);
        mNvrtc = api;
        std::error_code metadataError;
        const auto size = std::filesystem::file_size(path, metadataError);
        mCompilerIdentity = path.u8string() + "|" + (metadataError ? "unknown" : std::to_string(size));
        const auto modified = std::filesystem::last_write_time(path, metadataError);
        if (!metadataError) mCompilerIdentity += "|" + std::to_string(modified.time_since_epoch().count());
        loaded = true;
        break;
      } catch (const std::exception &error) {
        lastError = error.what();
      }
    }
    if (!loaded)
      throw std::runtime_error("NVRTC could not be loaded; set CUDA_PATH or CADMESH_NVRTC_DLL, "
          "or provide the project's .venv/Lib/site-packages/torch/lib CUDA libraries"
          + (lastError.empty() ? std::string() : std::string("; ") + lastError));

    int major = 0, minor = 0;
    mDriver.check(mDriver.DeviceGetAttribute(&major, 75, mDevice), "CUDA compute capability major");
    mDriver.check(mDriver.DeviceGetAttribute(&minor, 76, mDevice), "CUDA compute capability minor");
    int architecture = 10 * major + minor;
    if (mNvrtc.SupportedCount && mNvrtc.SupportedArchs) {
      int countSupported = 0;
      mNvrtc.check(mNvrtc.SupportedCount(&countSupported), "nvrtcGetNumSupportedArchs");
      if (countSupported <= 0) throw std::runtime_error("NVRTC reports no supported GPU architectures");
      std::vector<int> supported(std::size_t(countSupported), 0);
      mNvrtc.check(mNvrtc.SupportedArchs(supported.data()), "nvrtcGetSupportedArchs");
      int selected = 0;
      for (int candidate : supported)
        if (candidate <= architecture) selected = std::max(selected, candidate);
      if (!selected) throw std::runtime_error("NVRTC does not support this GPU's compute capability");
      architecture = selected;
    }
    compile(architecture);
    mComputeArchitecture = architecture;
  }

  std::string description() const {
    return "device 0 " + mDeviceName + ", compute_" + std::to_string(mComputeArchitecture);
  }

  void refine(std::vector<AnalyticSeedBatch> &batches, RuntimeTelemetry &telemetry,
              bool profileKernel, bool profileByType) {
    CurrentContext current(mDriver, mContext);
    const auto packingStart = Clock::now();
    std::size_t sampleCount = 0, seedCount = 0;
    for (const auto &batch : batches) {
      if (batch.Parameters.empty()) continue;
      sampleCount += batch.Samples.size();
      seedCount += batch.Parameters.size();
    }
    std::vector<std::array<double, 4>> samples;
    std::vector<std::array<double, 8>> parameters;
    std::vector<std::array<int, 5>> descriptors;
    samples.reserve(sampleCount);
    parameters.reserve(seedCount);
    descriptors.reserve(seedCount);
    for (const auto &batch : batches) {
      if (batch.Parameters.empty()) continue;
      const std::array<int, 5> descriptor = {
          int(samples.size()), int(batch.Samples.size()), int(batch.Type),
          ParameterDimension(batch.Type), batch.Iterations};
      samples.insert(samples.end(), batch.Samples.begin(), batch.Samples.end());
      parameters.insert(parameters.end(), batch.Parameters.begin(), batch.Parameters.end());
      descriptors.insert(descriptors.end(), batch.Parameters.size(), descriptor);
    }
    std::array<std::size_t, 6> typeBegin{}, typeCount{};
    std::vector<std::size_t> originalToPacked;
    if (profileByType) {
      // Diagnostic scheduling only. Stable grouping never changes a seed's
      // samples, parameters or iteration budget. Restore original result order
      // below before the model selector makes any decisions.
      auto originalParameters = std::move(parameters);
      auto originalDescriptors = std::move(descriptors);
      parameters.clear(); descriptors.clear();
      parameters.reserve(seedCount); descriptors.reserve(seedCount);
      originalToPacked.resize(seedCount);
      for (int type = 2; type <= 5; ++type) {
        typeBegin[type] = parameters.size();
        for (std::size_t i = 0; i < originalParameters.size(); ++i) {
          if (originalDescriptors[i][2] != type) continue;
          originalToPacked[i] = parameters.size();
          parameters.push_back(originalParameters[i]);
          descriptors.push_back(originalDescriptors[i]);
          telemetry.TypeSamples[type] += std::uint64_t(originalDescriptors[i][1]);
        }
        typeCount[type] = parameters.size() - typeBegin[type];
        telemetry.TypeSeeds[type] += typeCount[type];
      }
    }
    telemetry.HostPackingSeconds += SecondsSince(packingStart);
    const auto sampleBytes = samples.size() * sizeof(samples[0]);
    const auto parameterBytes = parameters.size() * sizeof(parameters[0]);
    const auto validBytes = parameters.size() * sizeof(int);
    const auto descriptorBytes = descriptors.size() * sizeof(descriptors[0]);
    reserve(mSamples, mSampleCapacity, sampleBytes);
    reserve(mParameters, mParameterCapacity, parameterBytes);
    reserve(mValid, mValidCapacity, validBytes);
    reserve(mDescriptors, mDescriptorCapacity, descriptorBytes);
    const auto comparisonStart = Clock::now();
    const bool reuseSamples = mLastSamples == samples;
    telemetry.SampleCacheSeconds += SecondsSince(comparisonStart);
    if (!reuseSamples) {
      const auto uploadStart = Clock::now();
      mDriver.check(mDriver.CopyToDevice(mSamples, samples.data(), sampleBytes), "copy analytic samples to CUDA");
      telemetry.HostToDeviceSeconds += SecondsSince(uploadStart);
      telemetry.HostToDeviceBytes += sampleBytes;
      ++telemetry.SampleUploadMisses;
      const auto cacheCopyStart = Clock::now();
      mLastSamples = samples;
      telemetry.SampleCacheSeconds += SecondsSince(cacheCopyStart);
    } else {
      ++telemetry.SampleUploadHits;
    }
    const auto uploadStart = Clock::now();
    mDriver.check(mDriver.CopyToDevice(mParameters, parameters.data(), parameterBytes), "copy analytic seeds to CUDA");
    mDriver.check(mDriver.CopyToDevice(mDescriptors, descriptors.data(), descriptorBytes), "copy analytic seed descriptors");
    telemetry.HostToDeviceSeconds += SecondsSince(uploadStart);
    telemetry.HostToDeviceBytes += parameterBytes + descriptorBytes;
    void *arguments[] = {&mSamples, &mDescriptors, &mParameters, &mValid};
    if (profileKernel) {
      if (!mKernelStart) mDriver.check(mDriver.EventCreate(&mKernelStart, 0), "create analytic start event");
      if (!mKernelEnd) mDriver.check(mDriver.EventCreate(&mKernelEnd, 0), "create analytic end event");
      mDriver.check(mDriver.EventRecord(mKernelStart, nullptr), "record analytic kernel start");
    }
    if (profileByType) {
      for (int type = 2; type <= 5; ++type) {
        if (!typeCount[type]) continue;
        if (!mTypeStart[type]) mDriver.check(mDriver.EventCreate(&mTypeStart[type], 0), "create type start event");
        if (!mTypeEnd[type]) mDriver.check(mDriver.EventCreate(&mTypeEnd[type], 0), "create type end event");
        CUdeviceptr typeDescriptors = mDescriptors + typeBegin[type] * sizeof(descriptors[0]);
        CUdeviceptr typeParameters = mParameters + typeBegin[type] * sizeof(parameters[0]);
        CUdeviceptr typeValid = mValid + typeBegin[type] * sizeof(int);
        void *typeArguments[] = {&mSamples, &typeDescriptors, &typeParameters, &typeValid};
        mDriver.check(mDriver.EventRecord(mTypeStart[type], nullptr), "record type kernel start");
        mDriver.check(mDriver.Launch(mFunction, unsigned(typeCount[type]), 1, 1, 128, 1, 1,
                                    0, nullptr, typeArguments, nullptr), "launch analytic type profile");
        mDriver.check(mDriver.EventRecord(mTypeEnd[type], nullptr), "record type kernel end");
        ++telemetry.TypeLaunches[type];
      }
    } else {
      mDriver.check(mDriver.Launch(mFunction, unsigned(parameters.size()), 1, 1, 128, 1, 1,
                                  0, nullptr, arguments, nullptr), "launch refine_analytic_seeds");
    }
    if (profileKernel)
      mDriver.check(mDriver.EventRecord(mKernelEnd, nullptr), "record analytic kernel end");
    // The first blocking copy waits for this stream's kernel and surfaces
    // execution errors. Do not add an extra device-wide synchronization.
    std::vector<std::array<double, 8>> output(parameters.size());
    std::vector<int> outputValid(parameters.size());
    const auto readbackStart = Clock::now();
    mDriver.check(mDriver.CopyToHost(output.data(), mParameters, parameterBytes), "copy refined CUDA seeds");
    mDriver.check(mDriver.CopyToHost(outputValid.data(), mValid, validBytes), "copy CUDA seed validity");
    telemetry.ReadbackWaitSeconds += SecondsSince(readbackStart);
    telemetry.DeviceToHostBytes += parameterBytes + validBytes;
    if (profileKernel) {
      // The existing blocking readback completes the recorded end event; no
      // extra stream/device synchronization is needed to obtain GPU timing.
      float milliseconds = 0;
      mDriver.check(mDriver.EventElapsedTime(&milliseconds, mKernelStart, mKernelEnd),
                    "read analytic kernel elapsed time");
      telemetry.KernelMilliseconds += milliseconds;
    }
    if (profileByType) {
      for (int type = 2; type <= 5; ++type) {
        if (!typeCount[type]) continue;
        float milliseconds = 0;
        mDriver.check(mDriver.EventElapsedTime(&milliseconds, mTypeStart[type], mTypeEnd[type]),
                      "read type kernel elapsed time");
        telemetry.TypeKernelMilliseconds[type] += milliseconds;
      }
    }
    std::size_t offset = 0;
    for (auto &batch : batches) {
      batch.Valid.assign(batch.Parameters.size(), 0);
      if (batch.Parameters.empty()) continue;
      const int dimension = ParameterDimension(batch.Type);
      for (std::size_t i = 0; i < batch.Parameters.size(); ++i) {
        const std::size_t index = profileByType ? originalToPacked[offset + i] : offset + i;
        bool finite = outputValid[index] == 1;
        for (int j = 0; j < dimension; ++j)
          finite = finite && std::isfinite(output[index][std::size_t(j)]);
        if (finite) {
          batch.Parameters[i] = output[index];
          batch.Valid[i] = 1;
        }
      }
      offset += batch.Parameters.size();
    }
  }

  void classifyLongEdges(
      const std::vector<std::array<double, 3>> &vertices,
      const std::vector<std::array<int, 2>> &edges, double maximum,
      std::vector<unsigned char> &selected,
      std::vector<std::array<double, 3>> &midpoints) {
    CurrentContext current(mDriver, mContext);
    const std::size_t vertexBytes = vertices.size() * sizeof(vertices[0]);
    const std::size_t edgeBytes = edges.size() * sizeof(edges[0]);
    const std::size_t selectedBytes = edges.size() * sizeof(unsigned char);
    const std::size_t midpointBytes = edges.size() * sizeof(midpoints[0]);
    reserve(mRemeshVertices, mRemeshVertexCapacity, vertexBytes);
    reserve(mRemeshEdges, mRemeshEdgeCapacity, edgeBytes);
    reserve(mRemeshSelected, mRemeshSelectedCapacity, selectedBytes);
    reserve(mRemeshMidpoints, mRemeshMidpointCapacity, midpointBytes);
    selected.resize(edges.size());
    midpoints.resize(edges.size());
    if (edges.empty()) return;
    mDriver.check(mDriver.CopyToDevice(mRemeshVertices, vertices.data(),
                                      vertexBytes),
                  "copy remesh vertices to CUDA");
    mDriver.check(mDriver.CopyToDevice(mRemeshEdges, edges.data(), edgeBytes),
                  "copy remesh edges to CUDA");
    int edgeCount = int(edges.size());
    double maximumSquared = maximum * maximum;
    void *arguments[] = {&mRemeshVertices, &mRemeshEdges,
                         &edgeCount, &maximumSquared,
                         &mRemeshSelected, &mRemeshMidpoints};
    mDriver.check(mDriver.Launch(
                      mRemeshFunction, unsigned((edges.size() + 255) / 256),
                      1, 1, 256, 1, 1, 0, nullptr, arguments, nullptr),
                  "launch cadmesh_classify_long_edges");
    mDriver.check(mDriver.CopyToHost(selected.data(), mRemeshSelected,
                                    selectedBytes),
                  "copy remesh edge selection from CUDA");
    mDriver.check(mDriver.CopyToHost(midpoints.data(), mRemeshMidpoints,
                                    midpointBytes),
                  "copy remesh midpoints from CUDA");
  }

private:
  static std::uint64_t CacheHash(const char *data, std::size_t size) {
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < size; ++i) {
      hash ^= static_cast<unsigned char>(data[i]);
      hash *= 1099511628211ull;
    }
    return hash;
  }

  static bool ReadPtxCache(const std::filesystem::path &path,
                           const std::string &key, std::vector<char> &ptx) {
    if (path.empty()) return false;
    std::ifstream input(path, std::ios::binary);
    std::uint64_t header[4] = {};
    if (!input.read(reinterpret_cast<char *>(header), sizeof(header)) ||
        header[0] != 0x4341445054583031ull || header[1] != key.size() ||
        header[2] == 0 || header[2] > 64 * 1024 * 1024) return false;
    std::string storedKey(key.size(), '\0');
    if (!input.read(storedKey.data(), std::streamsize(storedKey.size())) || storedKey != key)
      return false;
    std::vector<char> candidate(static_cast<std::size_t>(header[2]));
    if (!input.read(candidate.data(), std::streamsize(candidate.size())) ||
        candidate.back() != '\0' ||
        CacheHash(candidate.data(), candidate.size()) != header[3]) return false;
    ptx.swap(candidate);
    return true;
  }

  static bool WritePtxCache(const std::filesystem::path &path,
                            const std::string &key, const std::vector<char> &ptx) {
    if (path.empty()) return false;
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) return false;
    auto temporary = path;
    temporary += ".tmp." + std::to_string(GetCurrentProcessId()) + "." +
                 std::to_string(GetCurrentThreadId()) + "." +
                 std::to_string(Clock::now().time_since_epoch().count());
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      const std::uint64_t header[] = {0x4341445054583031ull,
          std::uint64_t(key.size()), std::uint64_t(ptx.size()), CacheHash(ptx.data(), ptx.size())};
      output.write(reinterpret_cast<const char *>(header), sizeof(header));
      output.write(key.data(), std::streamsize(key.size()));
      output.write(ptx.data(), std::streamsize(ptx.size()));
      output.close();
      if (!output) {
        std::filesystem::remove(temporary, error);
        return false;
      }
    }
    // Publish a complete entry atomically; concurrent processes never read a
    // half-written PTX. Cache failure must not fail a successful compilation.
    if (MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) return true;
    std::filesystem::remove(temporary, error);
    return false;
  }

  void reserve(CUdeviceptr &buffer, std::size_t &capacity, std::size_t bytes) {
    if (capacity >= bytes) return;
    std::size_t next = std::max<std::size_t>(4096, capacity);
    while (next < bytes) {
      if (next > std::numeric_limits<std::size_t>::max() / 2)
        throw std::runtime_error("CUDA analytic buffer size overflow");
      next *= 2;
    }
    CUdeviceptr replacement = 0;
    mDriver.check(mDriver.MemAlloc(&replacement, next), "allocate CUDA analytic workspace");
    if (buffer) mDriver.MemFree(buffer);
    buffer = replacement;
    capacity = next;
  }

  void compile(int architecture) {
    std::string source;
    for (const char *part : AnalyticSeedKernelSourceParts) source += part;
    source += std::string("\n") + RemeshKernelSource;
    const std::string target = "--gpu-architecture=compute_" + std::to_string(architecture);
    const char *options[] = {target.c_str(), "--std=c++14", "--fmad=false", "--ftz=false",
                             "--prec-div=true", "--prec-sqrt=true"};
    int major = 0, minor = 0;
    mNvrtc.check(mNvrtc.Version(&major, &minor), "nvrtcVersion");
    std::string key = "cadmesh-ptx-v1|" + mCompilerIdentity + "|nvrtc=" +
                      std::to_string(major) + "." + std::to_string(minor);
    for (const char *option : options) key += std::string("\n") + option;
    key += "\n" + source;
    std::filesystem::path cachePath;
    if (EnvironmentValue(L"CADMESH_CUDA_CACHE") != L"0") {
      auto root = EnvironmentValue(L"CADMESH_CUDA_CACHE_DIR");
      if (root.empty()) {
        const auto local = EnvironmentValue(L"LOCALAPPDATA");
        if (!local.empty()) root = (std::filesystem::path(local) / "CadMesh" / "cuda-cache").wstring();
      }
      if (!root.empty()) {
        std::ostringstream name;
        name << std::hex << CacheHash(key.data(), key.size()) << ".ptxcache";
        cachePath = std::filesystem::path(root) / name.str();
      }
    }
    std::vector<char> ptx;
    const auto cacheStart = Clock::now();
    bool cacheHit = ReadPtxCache(cachePath, key, ptx);
    const double cacheReadSeconds = SecondsSince(cacheStart);
    double compileSeconds = 0, loadSeconds = 0, cacheWriteSeconds = 0;
    const auto compilePtx = [&]() {
    const auto compileStart = Clock::now();
    NvrtcProgram program = nullptr;
    mNvrtc.check(mNvrtc.Create(&program, source.c_str(), "cadmesh_cuda_kernels.cu",
                               0, nullptr, nullptr), "nvrtcCreateProgram");
    struct ProgramCleanup {
      NvrtcApi &Api;
      NvrtcProgram &Program;
      ~ProgramCleanup() { Api.Destroy(&Program); }
    } cleanup{mNvrtc, program};
    const int result = mNvrtc.Compile(program, int(sizeof(options) / sizeof(options[0])), options);
    if (result != 0) {
      std::size_t bytes = 0;
      mNvrtc.LogSize(program, &bytes);
      std::vector<char> log(std::max<std::size_t>(bytes, 1), '\0');
      if (bytes) mNvrtc.Log(program, log.data());
      throw std::runtime_error(std::string("CUDA analytic kernel compilation failed: ")
          + mNvrtc.ErrorString(result) + "\n" + log.data());
    }
    std::size_t bytes = 0;
    mNvrtc.check(mNvrtc.PtxSize(program, &bytes), "nvrtcGetPTXSize");
    if (!bytes) throw std::runtime_error("NVRTC produced an empty analytic kernel");
    ptx.resize(bytes);
    mNvrtc.check(mNvrtc.Ptx(program, ptx.data()), "nvrtcGetPTX");
    compileSeconds += SecondsSince(compileStart);
    };
    if (!cacheHit) compilePtx();
    bool cacheRejected = false;
    for (;;) {
    const auto loadStart = Clock::now();
    char errorLog[4096] = {};
    int jitOptions[] = {5, 6}; // CU_JIT_ERROR_LOG_BUFFER and its size.
    void *jitValues[] = {errorLog, reinterpret_cast<void *>(std::uintptr_t(sizeof(errorLog)))};
    const CUresult loaded = mDriver.ModuleLoad(&mModule, ptx.data(), 2, jitOptions, jitValues);
    loadSeconds += SecondsSince(loadStart);
    if (loaded != 0 && cacheHit) {
      if (mModule) { mDriver.ModuleUnload(mModule); mModule = nullptr; }
      cacheHit = false;
      cacheRejected = true;
      compilePtx();
      continue;
    }
    if (loaded != 0) {
      errorLog[sizeof(errorLog) - 1] = '\0';
      try { mDriver.check(loaded, "load CUDA analytic PTX"); }
      catch (const std::exception &error) {
        throw std::runtime_error(std::string(error.what()) + "\n" + errorLog);
      }
    }
    mDriver.check(mDriver.ModuleFunction(&mFunction, mModule, "refine_analytic_seeds"),
                   "resolve refine_analytic_seeds");
    mDriver.check(mDriver.ModuleFunction(&mRemeshFunction, mModule,
                                         "cadmesh_classify_long_edges"),
                  "resolve cadmesh_classify_long_edges");
    break;
    }
    bool cacheWritten = false;
    if (!cacheHit && !cachePath.empty()) {
      const auto writeStart = Clock::now();
      cacheWritten = WritePtxCache(cachePath, key, ptx);
      cacheWriteSeconds = SecondsSince(writeStart);
    }
    if (mVerbose)
      std::clog << "[CadMesh] CUDA startup: ptx_cache="
                << (cachePath.empty() ? "disabled" : cacheHit ? "hit" : cacheRejected ? "rejected" : "miss")
                << ", nvrtc_compile_s=" << compileSeconds << ", module_load_s=" << loadSeconds
                << ", cache_read_s=" << cacheReadSeconds << ", cache_write_s=" << cacheWriteSeconds
                << ", cache_saved=" << cacheWritten << '\n';
  }

  DynamicLibrary mDriverLibrary;
  bool mVerbose = false;
  std::string mCompilerIdentity;
  std::vector<DynamicLibrary> mBuiltins;
  DynamicLibrary mNvrtcLibrary;
  DriverApi mDriver;
  NvrtcApi mNvrtc;
  CUdevice mDevice = 0;
  std::string mDeviceName;
  int mComputeArchitecture = 0;
  CUcontext mContext = nullptr;
  CUmodule mModule = nullptr;
  CUfunction mFunction = nullptr;
  CUfunction mRemeshFunction = nullptr;
  CUevent mKernelStart = nullptr, mKernelEnd = nullptr;
  std::array<CUevent, 6> mTypeStart{}, mTypeEnd{};
  CUdeviceptr mSamples = 0, mParameters = 0, mValid = 0;
  CUdeviceptr mDescriptors = 0;
  CUdeviceptr mRemeshVertices = 0, mRemeshEdges = 0;
  CUdeviceptr mRemeshSelected = 0, mRemeshMidpoints = 0;
  std::size_t mSampleCapacity = 0, mParameterCapacity = 0, mValidCapacity = 0;
  std::size_t mDescriptorCapacity = 0;
  std::size_t mRemeshVertexCapacity = 0, mRemeshEdgeCapacity = 0;
  std::size_t mRemeshSelectedCapacity = 0, mRemeshMidpointCapacity = 0;
  std::vector<std::array<double, 4>> mLastSamples;
};
#else
class CudaRuntime {
public:
  void initialize(bool = false) {
    throw std::runtime_error("CUDA analytic runtime loading is currently implemented for Windows x64");
  }
  void refine(std::vector<AnalyticSeedBatch> &, RuntimeTelemetry &, bool, bool) {
    throw std::runtime_error("CUDA analytic runtime is unavailable on this platform");
  }
  void classifyLongEdges(const std::vector<std::array<double, 3>> &,
                         const std::vector<std::array<int, 2>> &, double,
                         std::vector<unsigned char> &,
                         std::vector<std::array<double, 3>> &) {
    throw std::runtime_error("CUDA remesh runtime is unavailable on this platform");
  }
  std::string description() const { return "unavailable"; }
};
#endif

// Keep compiled code, its primary context and reusable workspaces across
// consecutive segmentations on the same thread. A failed runtime is discarded.
thread_local std::unique_ptr<CudaRuntime> Runtime;
} // namespace

struct CudaAnalyticFitSession::Impl {
  static thread_local Impl *Active;
  Impl *Previous = nullptr;
  AnalyticSeedBackend Backend;
  bool Verbose = false, Disabled = false, Attempted = false;
  bool ProfileKernel = false;
  bool ProfileByType = false;
  RuntimeTelemetry Telemetry;
  std::size_t Batches = 0, Seeds = 0, Invalid = 0;
  double InitializationSeconds = 0, CallSeconds = 0;
  Impl(AnalyticSeedBackend backend, bool verbose) : Previous(Active), Backend(backend), Verbose(verbose) {
    const char *profile = std::getenv("CADMESH_CUDA_PROFILE");
    ProfileKernel = verbose && profile && std::string(profile) == "1";
    const char *byType = std::getenv("CADMESH_CUDA_PROFILE_BY_TYPE");
    ProfileByType = verbose && byType && std::string(byType) == "1";
    ProfileKernel = ProfileKernel || ProfileByType;
    if (ProfileByType)
      std::clog << "[CadMesh] CUDA type profiling enabled: mixed batches run as separate type launches; "
                   "timings are diagnostic, not the normal mixed-batch baseline\n";
    Active = this;
  }
  ~Impl() {
    Active = Previous;
    if (Verbose && Attempted) {
      std::cerr << "[CadMesh] analytic CUDA: batches=" << Batches << ", seeds=" << Seeds
                << ", average seeds/launch=" << (Batches ? double(Seeds) / Batches : 0.0)
                << ", invalid=" << Invalid << ", initialization=" << InitializationSeconds
                << " s, transfer+kernel+wait=" << CallSeconds << " s"
                << (Disabled ? ", CPU fallback active" : "") << '\n';
      std::cerr << "[CadMesh] analytic CUDA telemetry: h2d_bytes=" << Telemetry.HostToDeviceBytes
                << ", d2h_bytes=" << Telemetry.DeviceToHostBytes
                << ", sample_upload_hits=" << Telemetry.SampleUploadHits
                << ", sample_upload_misses=" << Telemetry.SampleUploadMisses
                << ", host_packing_s=" << Telemetry.HostPackingSeconds
                << ", sample_cache_cpu_s=" << Telemetry.SampleCacheSeconds
                << ", h2d_s=" << Telemetry.HostToDeviceSeconds
                << ", readback_and_wait_s=" << Telemetry.ReadbackWaitSeconds;
      if (ProfileKernel) std::cerr << ", gpu_kernel_ms=" << Telemetry.KernelMilliseconds;
      else std::cerr << ", gpu_kernel_ms=disabled (CADMESH_CUDA_PROFILE=1 enables event timing)";
      std::cerr << '\n';
      if (ProfileByType) {
        const char *names[] = {"", "", "cylinder", "cone", "sphere", "torus"};
        double total = 0;
        for (int type = 2; type <= 5; ++type) total += Telemetry.TypeKernelMilliseconds[type];
        for (int type = 2; type <= 5; ++type)
          std::cerr << "[CadMesh] analytic CUDA type: " << names[type]
                    << ", launches=" << Telemetry.TypeLaunches[type]
                    << ", seeds=" << Telemetry.TypeSeeds[type]
                    << ", mean_samples_per_seed="
                    << (Telemetry.TypeSeeds[type] ? double(Telemetry.TypeSamples[type]) / Telemetry.TypeSeeds[type] : 0)
                    << ", gpu_kernel_ms=" << Telemetry.TypeKernelMilliseconds[type]
                    << ", share_percent=" << (total > 0 ? 100 * Telemetry.TypeKernelMilliseconds[type] / total : 0)
                    << '\n';
      }
    }
  }
};

thread_local CudaAnalyticFitSession::Impl *CudaAnalyticFitSession::Impl::Active = nullptr;

CudaAnalyticFitSession::CudaAnalyticFitSession(AnalyticSeedBackend backend, bool verbose)
    : mImpl(std::make_unique<Impl>(backend, verbose)) {}
CudaAnalyticFitSession::~CudaAnalyticFitSession() = default;

bool RefineAnalyticSeedsCuda(PatchSurfaceType type,
    const std::vector<std::array<double, 4>> &samples,
    std::vector<std::array<double, 8>> &parameters,
    std::vector<unsigned char> &valid, int iterations) {
  if (!CudaAnalyticFitAvailable()) return false;
  std::vector<AnalyticSeedBatch> batches(1);
  batches[0].Type = type;
  batches[0].Samples = samples;
  batches[0].Parameters = parameters;
  batches[0].Iterations = iterations;
  if (!RefineAnalyticSeedBatchCuda(batches)) return false;
  parameters = std::move(batches[0].Parameters);
  valid = std::move(batches[0].Valid);
  return true;
}

bool CudaAnalyticFitAvailable() {
  auto *session = CudaAnalyticFitSession::Impl::Active;
  return session && session->Backend != AnalyticSeedBackend::Cpu && !session->Disabled;
}

bool RefineAnalyticSeedBatchCuda(std::vector<AnalyticSeedBatch> &batches) {
  if (!CudaAnalyticFitAvailable()) return false;
  auto *session = CudaAnalyticFitSession::Impl::Active;
  std::size_t seedCount = 0, sampleCount = 0;
  for (auto &batch : batches) {
    batch.Valid.clear();
    seedCount += batch.Parameters.size();
    if (!batch.Parameters.empty()) sampleCount += batch.Samples.size();
  }
  if (seedCount == 0) return true;
  session->Attempted = true;
  try {
    if (seedCount > std::size_t(std::numeric_limits<int>::max() / 8)
        || sampleCount > std::size_t(std::numeric_limits<int>::max() / 4))
      throw std::runtime_error("CUDA analytic batch exceeds descriptor indexing limits");
    for (const auto &batch : batches) {
      if (batch.Parameters.empty()) continue;
      ParameterDimension(batch.Type);
      if (batch.Samples.empty() || batch.Samples.size() > 768 || batch.Iterations < 0)
        throw std::runtime_error("invalid CUDA analytic job dimensions (samples must be 1..768)");
      for (const auto &sample : batch.Samples) {
        for (double value : sample)
          if (!std::isfinite(value)) throw std::runtime_error("CUDA analytic samples must be finite");
        if (sample[3] < 0) throw std::runtime_error("CUDA analytic sample weights must be nonnegative");
      }
    }
    // Invalid initial candidates are individual failed fits, as in the CPU
    // solver. The kernel marks only those seeds invalid and preserves their
    // parameters; they must not disable CUDA or abort the whole partition.
    if (!Runtime) {
      const auto start = Clock::now();
      try {
        auto runtime = std::make_unique<CudaRuntime>();
        runtime->initialize(session->Verbose);
        Runtime = std::move(runtime);
      } catch (...) {
        session->InitializationSeconds += SecondsSince(start);
        throw;
      }
      session->InitializationSeconds += SecondsSince(start);
    }
    if (session->Verbose && session->Batches == 0)
      std::cerr << "[CadMesh] analytic CUDA ready: " << Runtime->description()
                << "; float64 batched seed refinement (specialized models, parallel statistics)\n";
    const auto start = Clock::now();
    try { Runtime->refine(batches, session->Telemetry, session->ProfileKernel, session->ProfileByType); }
    catch (...) {
      session->CallSeconds += SecondsSince(start);
      throw;
    }
    session->CallSeconds += SecondsSince(start);
    ++session->Batches;
    session->Seeds += seedCount;
    for (const auto &batch : batches)
      session->Invalid += std::size_t(std::count(batch.Valid.begin(), batch.Valid.end(),
                                                static_cast<unsigned char>(0)));
    return true;
  } catch (const std::exception &error) {
    Runtime.reset();
    if (session->Backend == AnalyticSeedBackend::Cuda)
      throw std::runtime_error(std::string("CUDA analytic seeds were explicitly requested: ") + error.what());
    session->Disabled = true;
    std::cerr << "[CadMesh] analytic CUDA unavailable; using CPU seed refinement: " << error.what() << '\n';
    return false;
  }
}

bool ClassifyLongEdgesCudaRuntime(
    const std::vector<std::array<double, 3>> &vertices,
    const std::vector<std::array<int, 2>> &edges, double maximumEdgeLength,
    std::vector<unsigned char> &selected,
    std::vector<std::array<double, 3>> &midpoints, std::string &error) {
  try {
    if (!(maximumEdgeLength > 0) ||
        edges.size() > std::size_t(std::numeric_limits<int>::max()))
      throw std::runtime_error("invalid CUDA remesh edge batch");
    if (!Runtime) {
      auto runtime = std::make_unique<CudaRuntime>();
      runtime->initialize();
      Runtime = std::move(runtime);
    }
    Runtime->classifyLongEdges(vertices, edges, maximumEdgeLength, selected,
                               midpoints);
    return true;
  } catch (const std::exception &failure) {
    Runtime.reset();
    error = failure.what();
    return false;
  }
}
} // namespace CadMesh
