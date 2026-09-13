cmake_minimum_required(VERSION 3.16)
project(CadMeshPatchSegmentation VERSION 0.1 LANGUAGES CXX)
find_package(Threads REQUIRED)

option(CADMESH_BUILD_TESTS "Build CadMesh tests" ON)
set(VCGLIB_ROOT "d:/openSourceInstall/vcglib" CACHE PATH "VCGLib root")

if(NOT EXISTS "${VCGLIB_ROOT}/vcg/complex/complex.h")
    message(FATAL_ERROR "VCGLib not found: ${VCGLIB_ROOT}")
endif()

add_library(CadMesh
    src/MeshTopology.cpp
    src/DifferentialGeometryEstimator.cpp
    src/BoundaryScoreCalculator.cpp
    src/SurfaceFitting.cpp
    src/CudaAnalyticFitting.cpp
    src/CudaAnalyticPrefetch.cpp
    src/AnalyticPatchRemesher.cpp
    src/HeatPatchRegularizer.cpp
    src/CadMeshPatchSegmenter.cpp
    src/PartitionSnapshot.cpp
    src/ModelFirstPartitioner.cpp
    src/SurfaceRoles.cpp
    src/PatchGraphBuilder.cpp
    src/NativeRemesher.cpp
    src/DebugVisualizer.cpp
    src/StlReader.cpp)
target_include_directories(CadMesh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${VCGLIB_ROOT} ${VCGLIB_ROOT}/eigenlib)
target_compile_features(CadMesh PUBLIC cxx_std_17)
target_link_libraries(CadMesh PRIVATE Threads::Threads)
if(CMAKE_DL_LIBS)
    target_link_libraries(CadMesh PRIVATE ${CMAKE_DL_LIBS})
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(CadMesh PUBLIC stdc++fs)
endif()

if(MSVC)
    target_compile_options(CadMesh PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(CadMesh PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(cad_mesh_segment app/cad_mesh_segment.cpp)
target_link_libraries(cad_mesh_segment PRIVATE CadMesh)
add_executable(cad_mesh_generic app/cad_mesh_generic.cpp)
target_link_libraries(cad_mesh_generic PRIVATE CadMesh)
add_executable(cad_mesh_quality app/cad_mesh_quality.cpp)
target_compile_features(cad_mesh_quality PRIVATE cxx_std_17)

if(CADMESH_BUILD_TESTS)
    enable_testing()
    add_executable(cad_mesh_tests
        tests/CadMeshTests.cpp
        tests/PatchGraphTests.cpp
        tests/GeometryGuardTests.cpp
        tests/ModelFirstTests.cpp
        tests/FilletSeamTests.cpp
        tests/AnalyticModelTests.cpp
        tests/SurfaceRoleTests.cpp)
    target_link_libraries(cad_mesh_tests PRIVATE CadMesh)
    target_compile_definitions(cad_mesh_tests PRIVATE CADMESH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME cad_mesh_tests COMMAND cad_mesh_tests)
    add_executable(cad_mesh_prefetch_tests
        tests/CudaAnalyticPrefetchTests.cpp
        src/CudaAnalyticPrefetch.cpp)
    target_include_directories(cad_mesh_prefetch_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${VCGLIB_ROOT} ${VCGLIB_ROOT}/eigenlib)
    target_compile_features(cad_mesh_prefetch_tests PRIVATE cxx_std_17)
    add_test(NAME cad_mesh_prefetch_tests COMMAND cad_mesh_prefetch_tests)
endif()
