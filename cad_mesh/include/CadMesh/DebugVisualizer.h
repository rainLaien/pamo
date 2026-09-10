#pragma once
#include "CadMesh/CadMeshPatchSegmenter.h"
#include <filesystem>
namespace CadMesh {
class DebugVisualizer {
public: static bool exportAll(const CadMeshPatchSegmenter&,const std::filesystem::path&);
    // Compact internal handoff: indexed float64 binary PLY plus the model and
    // constraint records required by remeshing; diagnostic exports stay opt-in.
    static bool exportRemeshHandoff(const CadMeshPatchSegmenter&,const std::filesystem::path&);
    static bool exportPatchPly(const CadMeshPatchSegmenter&,const std::filesystem::path&);
    static bool exportSurfaceTypePly(const CadMeshPatchSegmenter&,const std::filesystem::path&);
    static bool exportFeatureRolePly(const CadMeshPatchSegmenter&,const std::filesystem::path&);
    static bool exportBoundaryVtk(const CadMeshPatchSegmenter&,const std::filesystem::path&);
    static bool exportCurvatureVtk(const CadMeshPatchSegmenter&,const std::filesystem::path&,const std::string&,int);
    static bool exportReportJson(const CadMeshPatchSegmenter&,const std::filesystem::path&);
};
}
