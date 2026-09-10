#pragma once
#include <array>
#include <string>
#include <vector>

namespace CadMesh {
// Nonperiodic planar charts only. Original vertices are immutable boundary
// vertices; new vertices follow them in the returned arrays.
struct CudaPlanarChart {
  std::vector<std::array<double, 2>> Points;
  std::vector<std::array<int, 3>> Faces;
  double Target = 0;
  bool Complete = false;
  int Status = 0; // 1 complete, 2 capacity, 3 topology, 4 iteration budget
  int CapacityHint = 0;
};
// A failed job leaves its input untouched. Runtime failures return false.
bool RebuildPlanarChartsCuda(std::vector<CudaPlanarChart> &charts,
                            double &kernelMilliseconds, std::string &error);
}
