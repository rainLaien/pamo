cmake_minimum_required(VERSION 3.25)
enable_language(C)
enable_language(CUDA)
include(FetchContent)
set(RX_BUILD_APPS ON CACHE BOOL "" FORCE)
set(RX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RX_USE_POLYSCOPE OFF CACHE BOOL "" FORCE)
option(CADMESH_RXMESH_DOUBLE "Use double precision in the custom CadMesh backend" ON)
set(RX_USE_DOUBLE ${CADMESH_RXMESH_DOUBLE} CACHE BOOL "" FORCE)
FetchContent_Declare(rxmesh
    GIT_REPOSITORY https://github.com/owensgroup/RXMesh.git
    GIT_TAG e468c34ffabc70cd207309bce662d4821a9ed3b7)
FetchContent_MakeAvailable(rxmesh)
# Keep VCGLib's legacy Eigen headers out of the CUDA translation unit.
# The adapter boundary exposes only standard containers and plain geometry.
add_library(cad_mesh_rxmesh_input STATIC input.cpp)
target_link_libraries(cad_mesh_rxmesh_input PRIVATE CadMesh OpenMP::OpenMP_CXX)
target_compile_features(cad_mesh_rxmesh_input PRIVATE cxx_std_17)
add_executable(cad_mesh_rxmesh main.cu)
target_link_libraries(cad_mesh_rxmesh PRIVATE RXMesh cad_mesh_rxmesh_input)
target_compile_features(cad_mesh_rxmesh PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(cad_mesh_rxmesh PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(cad_mesh_rxmesh PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-lineinfo>)
# Deliberately do not enable fast-math in geometric predicates.
