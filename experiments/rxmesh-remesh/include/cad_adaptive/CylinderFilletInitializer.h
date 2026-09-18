#pragma once
#include "cad_adaptive/SemanticMesh.h"
#include <string>
#include <vector>

namespace cad_adaptive {
struct CylinderFilletPatchReport {
  uint32_t PatchId = kInvalidId;
  uint32_t AxialSegments = 0, ArcSegments = 0, InitialFaces = 0;
  float TargetLength = 0, AxialStep = 0, ArcStep = 0;
  float QualityMean = 0, QualityMin = 0;
};
struct CylinderFilletInitReport {
  uint32_t Detected = 0, Initialized = 0, BoundaryVerticesAdded = 0;
  std::vector<float> PatchTargetLengths;
  std::vector<CylinderFilletPatchReport> Patches;
  std::vector<std::string> Skipped;
};
// Transactional initializer for rectangular, constant-radius cylindrical fillets
// between two tangent planes. Original boundary points and polylines are kept;
// unsupported shapes are reported and left on the existing remesh path.
bool initializeCylinderFillets(SemanticMesh &mesh, const RemeshConfig &config,
    CylinderFilletInitReport &report, std::string *error = nullptr);
}
