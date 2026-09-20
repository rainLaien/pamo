cmake_minimum_required(VERSION 3.16)
project(cad_adaptive_remesh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cad_adaptive
    src/SemanticMesh.cpp
    src/PartitionInput.cpp
    src/CylinderFilletInitializer.cpp
    src/RemeshField.cpp
    src/BoundarySizingField.cpp
    src/RemeshPolicy.cpp
    src/AnalyticProjector.cpp
    src/RemeshMetrics.cpp
    src/RelaxProject.cpp
    src/MultiSegmentRefineBackend.cpp
    src/TriangleRefineBackend.cpp
    src/cpu/CpuRemeshBackend.cpp)
target_include_directories(cad_adaptive PUBLIC include)

add_executable(cad_adaptive_cli tools/remesh_cli.cpp)
target_link_libraries(cad_adaptive_cli PRIVATE cad_adaptive)

option(CAD_ADAPTIVE_VCGLIB_REFERENCE "Build a native VCGLib raw-mesh comparison executable" OFF)
set(VCGLIB_ROOT "D:/openSourceInstall/vcglib" CACHE PATH "VCGLib headers for the optional reference")
if(CAD_ADAPTIVE_VCGLIB_REFERENCE)
    add_executable(vcglib_reference tools/vcglib_reference.cpp)
    target_include_directories(vcglib_reference PRIVATE "${VCGLIB_ROOT}" "${VCGLIB_ROOT}/eigenlib")
    target_link_libraries(vcglib_reference PRIVATE cad_adaptive)
endif()

option(CAD_ADAPTIVE_RXMESH "Build the isolated RXMesh GPU backend" OFF)
include(CTest)
set(CAD_ADAPTIVE_RXMESH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../RXMesh"
    CACHE PATH "Pinned RXMesh source (e468c34)")

if(CAD_ADAPTIVE_RXMESH)
    cmake_minimum_required(VERSION 3.25)
    enable_language(C)
    enable_language(CUDA)

    # RXMesh experiment is developed and benchmarked on RTX 3060 (SM 86).
    # Keep a single architecture here: every extra SM multiplies CUDA compile
    # and device-link work. Override this cache value explicitly when a wider
    # deployment build is required.
    set(CAD_ADAPTIVE_RXMESH_ARCHITECTURES "86" CACHE STRING
        "CUDA architectures used by the isolated RXMesh backend")
    set(CMAKE_CUDA_ARCHITECTURES "${CAD_ADAPTIVE_RXMESH_ARCHITECTURES}"
        CACHE STRING "CUDA architectures" FORCE)
    set(RX_BUILD_APPS OFF CACHE BOOL "" FORCE)
    set(RX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(RX_USE_POLYSCOPE OFF CACHE BOOL "" FORCE)
    set(RX_USE_DOUBLE OFF CACHE BOOL "" FORCE)
    if(NOT EXISTS "${CAD_ADAPTIVE_RXMESH_SOURCE}/CMakeLists.txt")
        message(FATAL_ERROR "RXMesh source not found: ${CAD_ADAPTIVE_RXMESH_SOURCE}")
    endif()
    set(_saved_runtime "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    add_subdirectory("${CAD_ADAPTIVE_RXMESH_SOURCE}" ${CMAKE_BINARY_DIR}/rxmesh_dep)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${_saved_runtime}")

    # RXMesh itself contains multiple CUDA translation units with relocatable
    # device code. The wrapper therefore still requires one final CUDA device
    # link; disabling it produces __cudaRegisterLinkedBinary_* unresolved symbols.
    add_library(cad_adaptive_rx STATIC src/rxmesh/RxMeshBackend.cu)
    add_library(cad_adaptive_raw_cuda STATIC src/rxmesh/RawCudaRemesher.cu)
    target_link_libraries(cad_adaptive_raw_cuda PUBLIC cad_adaptive)
    target_compile_features(cad_adaptive_raw_cuda PUBLIC cxx_std_17 cuda_std_17)
    target_link_libraries(cad_adaptive_rx PUBLIC cad_adaptive RXMesh cad_adaptive_raw_cuda)
    target_compile_features(cad_adaptive_rx PUBLIC cxx_std_17 cuda_std_17)
    target_compile_definitions(cad_adaptive_rx PUBLIC CAD_ADAPTIVE_RXMESH=1)
    set_target_properties(cad_adaptive_rx PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)
    target_compile_options(cad_adaptive_rx PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-lineinfo>)

    # Patch-independent global-topology POC. This target intentionally does not
    # link RXMesh: topology lives in global memory and split operators gather
    # only their local EVDiamond working set.
    add_library(cad_adaptive_global STATIC src/global/GlobalSplitBackend.cu)
    if(MSVC AND CMAKE_CUDA_COMPILER)
        get_filename_component(_cuda_bin_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        get_filename_component(_cuda_root "${_cuda_bin_dir}" DIRECTORY)
        set_property(TARGET cad_adaptive_global PROPERTY VS_GLOBAL_CudaToolkitDir "${_cuda_root}/")
    endif()
    target_link_libraries(cad_adaptive_global PUBLIC cad_adaptive)
    target_include_directories(cad_adaptive_global PUBLIC include)
    target_compile_features(cad_adaptive_global PUBLIC cxx_std_17 cuda_std_17)
    target_compile_definitions(cad_adaptive_global PUBLIC CAD_ADAPTIVE_GLOBAL_TOPOLOGY=1)
    set_target_properties(cad_adaptive_global PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(cad_adaptive_global PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

    # VS CudaLink can leave cad_adaptive_rx.device-link.obj stale when
    # RxMeshBackend.cu or an included RXMesh kernel changes. Force that
    # intermediate to be regenerated so the fatbin registration hashes match.
    set(_rx_dlink_stamp "${CMAKE_CURRENT_BINARY_DIR}/rxmesh_backend.dlink.stamp")
    file(GLOB _rx_kernel_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/rxmesh/*.cuh")
    add_custom_command(
        OUTPUT "${_rx_dlink_stamp}"
        COMMAND ${CMAKE_COMMAND} -E rm -f
            "${CMAKE_CURRENT_BINARY_DIR}/cad_adaptive_rx.dir/Release/cad_adaptive_rx.device-link.obj"
            "${CMAKE_CURRENT_BINARY_DIR}/cad_adaptive_rx.dir/Debug/cad_adaptive_rx.device-link.obj"
            "${CMAKE_CURRENT_BINARY_DIR}/cad_adaptive_rx.dir/RelWithDebInfo/cad_adaptive_rx.device-link.obj"
            "${CMAKE_CURRENT_BINARY_DIR}/Release/cad_adaptive_rx.device-link.obj"
        COMMAND ${CMAKE_COMMAND} -E touch "${_rx_dlink_stamp}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/rxmesh/RxMeshBackend.cu"
            ${_rx_kernel_headers}
            "${CAD_ADAPTIVE_RXMESH_SOURCE}/include/rxmesh/rxmesh_dynamic.cu"
            "${CAD_ADAPTIVE_RXMESH_SOURCE}/include/rxmesh/rxmesh_static.cu"
        COMMENT "Drop stale cad_adaptive_rx CUDA device-link object")
    target_sources(cad_adaptive_rx PRIVATE "${_rx_dlink_stamp}")

    # Same CLI and same global remesher, without unrelated RXMesh solver DLLs.
    # Keep the original multi-backend executable for explicit --gpu users.
    add_executable(cad_adaptive_global_cli tools/remesh_cli.cpp)
    target_link_libraries(cad_adaptive_global_cli PRIVATE cad_adaptive_global)


    target_link_libraries(cad_adaptive_cli PRIVATE cad_adaptive_rx cad_adaptive_global)
    target_compile_definitions(cad_adaptive_cli PRIVATE
        CAD_ADAPTIVE_RXMESH=1
        CAD_ADAPTIVE_GLOBAL_TOPOLOGY=1)
    if(BUILD_TESTING)
        # Compile internal CUDA helpers in a standalone regression executable.
        add_executable(test_raw_projection tests/test_raw_projection.cu)
        target_link_libraries(test_raw_projection PRIVATE cad_adaptive)
        target_compile_features(test_raw_projection PRIVATE cuda_std_17)
        if(MSVC)
            set_property(TARGET test_raw_projection PROPERTY VS_GLOBAL_CudaToolkitDir "${_cuda_root}/")
        endif()
        add_test(NAME raw_projection_safety COMMAND test_raw_projection)
        set_tests_properties(raw_projection_safety PROPERTIES TIMEOUT 120)
        foreach(case raw_reference isotropic constraints project)
            add_executable(test_rxmesh_${case} tests/test_rxmesh_${case}.cpp)
            target_link_libraries(test_rxmesh_${case} PRIVATE cad_adaptive_rx)
            add_test(NAME rxmesh_${case} COMMAND test_rxmesh_${case})
            set_tests_properties(rxmesh_${case} PROPERTIES TIMEOUT 180)
        endforeach()
        target_compile_definitions(test_rxmesh_raw_reference PRIVATE
            RAW_BODY_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/../../examples/Unnamed-Body.stl")
    endif()
else()
    target_sources(cad_adaptive PRIVATE src/rxmesh/RxMeshStub.cpp)
endif()
