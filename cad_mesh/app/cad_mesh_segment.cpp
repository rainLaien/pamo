#include "CadMesh/CadMeshPatchSegmenter.h"
#include "CadMesh/DebugVisualizer.h"
#include "CadMesh/StlReader.h"
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
                 "[--legacy] [--strong value] [--weak value] [--rings count]\n";
    return 2;
  }
  CadMesh::SegmentationConfig config;
  config.Verbose = true;
  std::clog.setf(std::ios::unitbuf);
  for (int i = 3; i < argc; ++i) {
    std::string option = argv[i];
    if (option == "--legacy") {
      config.EnableModelFirst = false;
      continue;
    }
    if (i + 1 >= argc)
      throw std::invalid_argument("Missing value for " + option);
    const std::string argument = argv[++i];
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
    else {
      std::cerr << "Unknown option or value out of range: " << option << '\n';
      return 2;
    }
  }
  CadMesh::TriangleSoup soup;
  std::string error;
  if (!CadMesh::StlReader::read(argv[1], soup, error)) {
    std::cerr << "STL read failed: " << error << '\n';
    return 1;
  }
  CadMesh::CadMeshPatchSegmenter segmenter(config);
  if (!segmenter.segment(soup)) {
    std::cerr << "Segmentation failed\n";
    return 1;
  }
  if (!CadMesh::DebugVisualizer::exportAll(segmenter, argv[2])) {
    std::cerr << "Debug export failed\n";
    return 1;
  }
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
  std::cout << "debug output: " << std::filesystem::absolute(argv[2]).string()
            << '\n';
  return 0;
}

int main(int argc, char **argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "Segmentation failed: " << error.what() << '\n';
    return 1;
  }
}
