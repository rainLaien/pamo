#pragma once
#include "CadMesh/MeshTopology.h"
#include <filesystem>
namespace CadMesh { class StlReader { public: static bool read(const std::filesystem::path&,TriangleSoup&,std::string&); }; }
