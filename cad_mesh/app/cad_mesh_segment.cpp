#include "CadMesh/CadMeshPatchSegmenter.h"
#include "CadMesh/DebugVisualizer.h"
#include "CadMesh/NativeRemesher.h"
#include "CadMesh/StlReader.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int Run(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: cad_mesh_segment input.stl output_directory "
                 "[--fit-tolerance-ratio value] [--normal-angle-deg value] "
                 "[--sharp-angle-deg value] [--seed-radius-factor value] "
                 "[--max-model-seeds count] "
                 "[--analytic-seed-backend auto|cpu|cuda] "
                 "[--remesh --target-edge-length value --max-deviation value] "
                 "[--max-normal-deviation-deg value] [--target-mean-quality value] "
                 "[--split-passes count] [--collapse-passes count] "
                 "[--flip-passes count] [--relax-iterations count] "
                 "[--require-remesh-cuda | --cpu] "
                 "[--remesh-handoff] "
                 "[--stop-after-partition] "
                 "[--partition-snapshot (input is a prepared native snapshot)] "
                 "[--legacy] [--strong value] [--weak value] [--rings count]\n";
    return 2;
  }
  CadMesh::SegmentationConfig config;
  CadMesh::NativeRemeshConfig remeshConfig;
  bool nativeRemesh = false;
  bool remeshHandoff = false;
  bool maximumDeviationSpecified = false;
  bool stopAfterPartition = false;
  bool partitionSnapshot = false;
  config.Verbose = true;
  std::clog.setf(std::ios::unitbuf);
  for (int i = 3; i < argc; ++i) {
    std::string option = argv[i];
    if (option == "--cpu") {
      remeshConfig.DisableCuda = true;
      continue;
    }
    if (option == "--partition-snapshot") {
      partitionSnapshot = true;
      nativeRemesh = true;
      continue;
    }
    if (option == "--stop-after-partition") {
      stopAfterPartition = true;
      continue;
    }
    if (option == "--remesh") {
      nativeRemesh = true;
      continue;
    }
    if (option == "--require-remesh-cuda") {
      remeshConfig.RequireCuda = true;
      continue;
    }
    if (option == "--remesh-handoff") {
      remeshHandoff = true;
      continue;
    }
    if (option == "--legacy") {
      config.EnableModelFirst = false;
      continue;
    }
    if (i + 1 >= argc)
      throw std::invalid_argument("Missing value for " + option);
    const std::string argument = argv[++i];
    if (option == "--analytic-seed-backend") {
      if (argument == "auto")
        config.ModelAnalyticSeedBackend = CadMesh::AnalyticSeedBackend::Auto;
      else if (argument == "cpu")
        config.ModelAnalyticSeedBackend = CadMesh::AnalyticSeedBackend::Cpu;
      else if (argument == "cuda")
        config.ModelAnalyticSeedBackend = CadMesh::AnalyticSeedBackend::Cuda;
      else
        throw std::invalid_argument("--analytic-seed-backend must be auto, cpu or cuda");
      continue;
    }
    size_t consumed = 0;
    const double value = std::stod(argument, &consumed);
    if (consumed != argument.size() || !std::isfinite(value) || value < 0)
      throw std::invalid_argument("Expected a finite nonnegative value for " +
                                  option);
    constexpr double radians = 3.14159265358979323846 / 180;
    if (option == "--fit-tolerance-ratio" && value > 0)
      config.ModelFitToleranceRatio = value;
    else if (option == "--normal-angle-deg" && value > 0 && value < 90)
      config.ModelNormalTolerance = value * radians;
    else if (option == "--sharp-angle-deg" && value > 0 && value < 90)
      config.ModelSharpAngle = value * radians;
    else if (option == "--seed-radius-factor" && value > 0)
      config.ModelSeedRadiusFactor = value;
    else if (option == "--max-model-seeds" && value >= 1 && value <= 1000000 &&
             std::floor(value) == value)
      config.ModelMaximumSeeds = int(value);
    else if (option == "--strong" && value <= 1)
      config.StrongBoundaryThreshold = value;
    else if (option == "--weak" && value <= 1)
      config.WeakBoundaryThreshold = value;
    else if (option == "--rings" && value >= 1 && value <= 16 &&
             std::floor(value) == value)
      config.CurvatureRingCount = int(value);
    else if (option == "--target-edge-length" && value > 0)
      remeshConfig.TargetEdgeLength = value;
    else if (option == "--generic-feature-angle-deg" && value >= 0 && value <= 180)
      remeshConfig.GenericFeatureAngleDegrees = value;
    else if (option == "--generic-remesh-iterations" && value >= 1 && value <= 100 && std::floor(value)==value)
      remeshConfig.GenericRemeshIterations = int(value);
    else if (option == "--generic-remesh-workers" && value >= 1 && value <= 128 && std::floor(value)==value)
      remeshConfig.GenericRemeshWorkers = int(value);
    else if (option == "--max-deviation") {
      remeshConfig.MaximumDeviation = value;
      maximumDeviationSpecified = true;
    }
    else if ((option == "--max-normal-deviation-deg" ||
              option == "--max-normal-deviation-degrees") &&
             value > 0 && value <= 180)
      remeshConfig.MaximumNormalDeviationDegrees = value;
    else if (option == "--target-mean-quality" && value > 0 && value <= 1)
      remeshConfig.TargetMeanTriangleQuality = value;
    else if (option == "--split-passes" && value >= 1 && value <= 10000 &&
             std::floor(value) == value)
      remeshConfig.SplitPasses = int(value);
    else if (option == "--collapse-passes" && value >= 0 && value <= 10000 &&
             std::floor(value) == value)
      remeshConfig.CollapsePasses = int(value);
    else if (option == "--flip-passes" && value >= 0 && value <= 10000 &&
             std::floor(value) == value)
      remeshConfig.FlipPasses = int(value);
    else if (option == "--relax-iterations" && value >= 0 && value <= 10000 &&
             std::floor(value) == value)
      remeshConfig.RelaxIterations = int(value);
    else {
      std::cerr << "Unknown option or value out of range: " << option << '\n';
      return 2;
    }
  }
  if (remeshConfig.DisableCuda) {
    if (remeshConfig.RequireCuda)
      throw std::invalid_argument("--cpu cannot be combined with --require-remesh-cuda");
    config.ModelAnalyticSeedBackend = CadMesh::AnalyticSeedBackend::Cpu;
    std::clog << "[CadMesh] CPU mode: CUDA disabled for fitting, boundaries and charts\n";
  }
  CadMesh::TriangleSoup soup;
  std::string error;
  using Clock = std::chrono::steady_clock;
  const auto importStart = Clock::now();
  if (!partitionSnapshot && !CadMesh::StlReader::read(argv[1], soup, error)) {
    std::cerr << "STL read failed: " << error << '\n';
    return 1;
  }
  if (nativeRemesh && remeshHandoff)
    throw std::invalid_argument("--remesh and --remesh-handoff are mutually exclusive");
  if (partitionSnapshot && stopAfterPartition)
    throw std::invalid_argument("--partition-snapshot cannot be combined with --stop-after-partition");
  if (!partitionSnapshot) std::clog << "[CadMesh] STL import: "
            << std::chrono::duration<double>(Clock::now() - importStart).count() << " s\n";
  CadMesh::CadMeshPatchSegmenter segmenter(config);
  const auto partitionStart = Clock::now();
  if (partitionSnapshot ? !segmenter.loadRemeshSnapshot(argv[1], error) : !segmenter.segment(soup)) {
    std::cerr << (partitionSnapshot ? "Partition snapshot load failed: " : "Segmentation failed: ")
              << error << '\n';
    return 1;
  }
  if (partitionSnapshot)
    std::clog << "[CadMesh] saved partition load: "
              << std::chrono::duration<double>(Clock::now() - partitionStart).count()
              << " s, patches=" << segmenter.getPatches().size()
              << "; surface fitting and segmentation skipped\n";
  if (stopAfterPartition) {
    std::clog << "[CadMesh] stop-after-partition: "
              << std::chrono::duration<double>(Clock::now() - partitionStart).count()
              << " s (topology + partition + patch graph), patches="
              << segmenter.getPatches().size()
              << (remeshHandoff ? "; exporting partition snapshot\n"
                                : "; remesh and export skipped\n");
    if (remeshHandoff) {
      const auto snapshotStart = Clock::now();
      const auto directory = std::filesystem::absolute(argv[2]);
      if (!CadMesh::DebugVisualizer::exportRemeshHandoff(segmenter, directory)) {
        std::cerr << "Partition snapshot export failed: " << directory << '\n';
        return 1;
      }
      std::clog << "[CadMesh] partition snapshot export: "
                << std::chrono::duration<double>(Clock::now() - snapshotStart).count()
                << " s\n[CadMesh] patch PLY: " << (directory / "patch_result.ply")
                << "\n[CadMesh] surface models and constraints: "
                << (directory / "patch_report.json") << '\n';
    }
    return 0;
  }
  const auto exportStart = Clock::now();
  std::filesystem::path finalOutput;
  if (nativeRemesh) {
    const auto &resolution = segmenter.getMesh().getResolution();
    if (!(remeshConfig.TargetEdgeLength > 0))
      remeshConfig.TargetEdgeLength = resolution.BoundingBoxDiagonal * .05;
    if (!maximumDeviationSpecified)
      remeshConfig.MaximumDeviation = remeshConfig.TargetEdgeLength * .005;
    CadMesh::NativeRemeshResult remeshed;
    if (!CadMesh::NativeRemesher::remesh(segmenter, remeshConfig, remeshed, error)) {
      std::cerr << "Native remesh failed: " << error << '\n';
      return 1;
    }
    finalOutput = std::filesystem::path(argv[2]);
    if (finalOutput.extension() != ".ply")
      finalOutput /= "remesh_result.ply";
    if (!CadMesh::NativeRemesher::writePly(
            remeshed, segmenter.getPatches(), finalOutput, error)) {
      std::cerr << "PLY export failed: " << error << '\n';
      return 1;
    }
    const auto &stats = remeshed.Statistics;
    std::cout << "remesh: vertices=" << stats.InputVertices << " -> "
              << stats.OutputVertices << ", triangles=" << stats.InputTriangles
              << " -> " << stats.OutputTriangles;
    if(stats.QualityMeasured)std::cout << ", max_edge=" << stats.MaximumEdgeLength
              << ", mean_quality=" << stats.MeanTriangleQuality
              << ", min_quality=" << stats.MinimumTriangleQuality
              << ", q05_quality=" << stats.Percentile05TriangleQuality
              << ", below_q0.2="
              << 100.0 * stats.FractionBelow02TriangleQuality << "%"
              << ", min_angle_deg=" << stats.MinimumAngleDegrees
              << ", quality_target="
              << (stats.QualityTargetMet ? "met" : "not_met");
    else std::cout << ", quality=not_measured";
    std::cout << ", collision_rejections=" << stats.CollisionRejections
              << ", analytic_patches=" << stats.AnalyticPatchesRebuilt
              << "/" << stats.AnalyticPatchesAttempted
              << ", analytic_fallback=" << stats.AnalyticPatchesFallback
              << ", self_intersection_free="
              << (stats.SelfIntersectionFree ? "yes" : "no")
              << ", cuda=" << (stats.UsedCuda ? "yes" : "no") << '\n';
  } else if (!(remeshHandoff
                   ? CadMesh::DebugVisualizer::exportRemeshHandoff(segmenter, argv[2])
                   : CadMesh::DebugVisualizer::exportAll(segmenter, argv[2]))) {
      std::cerr << "Debug export failed\n";
      return 1;
  }
  std::clog << "[CadMesh] " << (nativeRemesh ? "native remesh and PLY export" :
            (remeshHandoff ? "remesh handoff export" : "full diagnostic export"))
            << ": " << std::chrono::duration<double>(Clock::now() - exportStart).count() << " s\n";
  const auto &r = segmenter.getMesh().getResolution();
  const auto &c = segmenter.getMesh().getCleanupReport();
  std::cout << "partition mode: "
            << (config.EnableModelFirst ? "model-first" : "legacy")
            << "\ntriangles: " << c.InputTriangles << " -> "
            << c.OutputTriangles
            << ", patches: " << segmenter.getPatches().size()
            << "\nmedian edge: " << r.MedianEdgeLength
            << ", fit tolerance: " << r.FittingTolerance << "\n";
  if (segmenter.getPatches().size() <= 100) {
    for (const auto &p : segmenter.getPatches())
      std::cout << "patch " << p.Id << ": "
                << CadMesh::SurfaceTypeName(p.SurfaceType)
                << ", triangles=" << p.TriangleIds.size()
                << ", rms=" << p.RmsFittingError
                << ", neighbors=" << p.NeighborPatchIds.size() << '\n';
  } else {
    int counts[7]{};
    for (const auto &p : segmenter.getPatches())
      ++counts[CadMesh::SurfaceTypeId(p.SurfaceType)];
    std::cout << "types: Plane=" << counts[1] << ", Cylinder=" << counts[2]
              << ", Cone=" << counts[3] << ", Sphere=" << counts[4]
              << ", Torus=" << counts[5] << ", Freeform=" << counts[6] << '\n';
  }
  for (const auto &w : c.Warnings)
    std::cerr << "warning: " << w << '\n';
  if (nativeRemesh)
    std::cout << "remesh output: " << std::filesystem::absolute(finalOutput).string() << '\n';
  else
    std::cout << "debug output: " << std::filesystem::absolute(argv[2]).string() << '\n';
  return 0;
}

int main(int argc, char **argv) {
  using Clock = std::chrono::steady_clock;
  const auto programStart = Clock::now();
  int exitCode = 1;
  try {
    exitCode = Run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "Segmentation failed: " << error.what() << '\n';
  }
  std::clog << "[CadMesh] total wall time: "
            << std::chrono::duration<double>(Clock::now() - programStart).count()
            << " s (exit " << exitCode << ")\n";
  return exitCode;
}
