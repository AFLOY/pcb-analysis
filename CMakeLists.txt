# Builds the optional pybind11 extensions of pcb-analysis in place, next to the
# Python package that loads them.  The per-package ``native/build.py`` scripts
# remain as the single-file fallback; this file is the one place that knows
# every native target, so a new kernel is added here and nowhere else.
#
#   cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/native
#
# ``-DPCB_NATIVE_OPENMP=OFF`` drops OpenMP, ``-DPCB_NATIVE_MARCH=`` overrides
# the ``-march=native`` default (set it to a fixed ISA for a shared wheel).
# ``Python_EXECUTABLE`` selects the interpreter whose ABI tag the modules get;
# pybind11 is found through that interpreter's ``python -m pybind11 --cmakedir``.

cmake_minimum_required(VERSION 3.18)
project(pcb_analysis_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PCB_NATIVE_OPENMP "Compile the native kernels with OpenMP" ON)
set(PCB_NATIVE_MARCH "native" CACHE STRING "Value for -march (empty disables)")

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT pybind11_DIR AND NOT pybind11_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_result)
  if(_pybind11_result EQUAL 0)
    set(pybind11_DIR "${_pybind11_cmakedir}")
  endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

if(PCB_NATIVE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)


# The kernels that were separate modules (_layered_dc_native,
# _scalar_maxwell_native, _thermal_native, _dipole_native, _voxelize_native,
# _sheet_pfft_native) are submodules of electrical._pcbcore now, built with
# the flags they always had: their OpenMP SIMD reductions follow the vector
# width, so the same flags keep the same bits.  native/build.py still builds
# each one on its own (PCB_STANDALONE_MODULE) where CMake is not available.
add_library(pcbcore_kernels STATIC
  src/electrical/matrix_free_mpir_fem/native/scalar_maxwell_q1.cpp
  src/electrical/matrix_free_mpir_fem/native/layered_dc_q1.cpp
  src/thermal/matrix_free_mpir_fem/native/hex_q1.cpp
  src/emc/tiled_dipole_superposition/native/dipole_fields.cpp
  src/geometry/cad_import/native/point_in_mesh.cpp
  src/electrical/sheet_peec/native/pfft_near.cpp)
target_compile_options(pcbcore_kernels PRIVATE -O3 -fcx-limited-range -fno-math-errno -DNDEBUG)
if(PCB_NATIVE_MARCH)
  target_compile_options(pcbcore_kernels PRIVATE "-march=${PCB_NATIVE_MARCH}" -mprefer-vector-width=512)
endif()
target_compile_definitions(pcbcore_kernels PRIVATE EIGEN_DONT_PARALLELIZE EIGEN_MPL2_ONLY)
target_include_directories(pcbcore_kernels PRIVATE cpp/include)
target_link_libraries(pcbcore_kernels PRIVATE Eigen3::Eigen pybind11::module)
if(PCB_NATIVE_OPENMP)
  target_link_libraries(pcbcore_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# pcbcore: the C++ solver core (cpp/) and its one extension module,
# ``electrical._pcbcore``.  New solver code goes here rather than into another
# per-package module.
#
# SuperLU is built from its release source with its bundled CBLAS, as a static
# library inside the module: the factorisations then run on one thread with
# the same BLAS on every machine, never on whatever multithreaded BLAS the
# system's libblas happens to be.  SuperLU is BSD-licensed.
include(FetchContent)
set(enable_internal_blaslib ON CACHE BOOL "" FORCE)
set(enable_single OFF CACHE BOOL "" FORCE)
set(enable_complex OFF CACHE BOOL "" FORCE)
set(enable_double ON CACHE BOOL "" FORCE)
set(enable_complex16 ON CACHE BOOL "" FORCE)
set(enable_tests OFF CACHE BOOL "" FORCE)
set(enable_examples OFF CACHE BOOL "" FORCE)
set(enable_fortran OFF CACHE BOOL "" FORCE)
set(enable_matlabmex OFF CACHE BOOL "" FORCE)
set(enable_doc OFF CACHE BOOL "" FORCE)
set(TPL_ENABLE_METISLIB OFF CACHE BOOL "" FORCE)
set(_pcb_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(superlu
  URL https://github.com/xiaoyeli/superlu/archive/refs/tags/v7.0.1.tar.gz
  URL_HASH SHA256=86dcca1e086f8b8079990d07f00eb707fc9ef412cf3b2ce808b37956f0de2cb8)
FetchContent_MakeAvailable(superlu)
set(BUILD_SHARED_LIBS ${_pcb_shared_libs})

# pocketfft (BSD), header only: the FFT NumPy and SciPy use.  Compiled without
# its own threads; the operators run independent transforms on OpenMP teams.
FetchContent_Declare(pocketfft
  URL https://github.com/mreineck/pocketfft/archive/c90e55b3d529f8efa40ed01a20de22405f45fc65.tar.gz
  URL_HASH SHA256=2d90016da31a8cc1dd0ac0c89c1f2fb55df8f4ccc3f0386dcfbad09ac6d1c3ba)
FetchContent_MakeAvailable(pocketfft)

add_library(pcbcore STATIC
  cpp/src/fft.cpp
  cpp/src/linalg/sparse_lu.cpp
  cpp/src/network/dc_network.cpp
  cpp/src/sheet/convolution_operator.cpp
  cpp/src/sheet/hoer_love.cpp
  cpp/src/sheet/near_field.cpp
  cpp/src/sheet/pfft_operator.cpp
  cpp/src/sheet/sheet_solve.cpp
  cpp/src/sheet/mesh.cpp
  cpp/src/sheet/pfft_build.cpp
  cpp/src/fem/layered_dc_system.cpp
  cpp/src/fem/dc_ports.cpp)
target_include_directories(pcbcore PUBLIC cpp/include)
target_include_directories(pcbcore PRIVATE ${pocketfft_SOURCE_DIR})
target_compile_definitions(pcbcore PRIVATE POCKETFFT_NO_MULTITHREADING POCKETFFT_CACHE_SIZE=16)
# No FMA contraction: a*b + c*d rounds twice, as in NumPy, whatever the ISA,
# so results do not change with -march (or between wheels for different CPUs).
target_compile_options(pcbcore PRIVATE -O3 -ffp-contract=off -fcx-limited-range -fno-math-errno -DNDEBUG)
if(PCB_NATIVE_MARCH)
  target_compile_options(pcbcore PRIVATE "-march=${PCB_NATIVE_MARCH}")
endif()
target_compile_definitions(pcbcore PRIVATE EIGEN_DONT_PARALLELIZE EIGEN_MPL2_ONLY)
find_package(Threads REQUIRED)
target_link_libraries(pcbcore PRIVATE superlu Eigen3::Eigen)
# std::thread: the N-port basis runs its unit solves on a pool of its own.
target_link_libraries(pcbcore PUBLIC Threads::Threads)
if(PCB_NATIVE_OPENMP)
  target_link_libraries(pcbcore PUBLIC OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(_pcbcore MODULE cpp/bindings/module.cpp cpp/bindings/fem.cpp)
target_compile_options(_pcbcore PRIVATE -O3 -DNDEBUG)
target_link_libraries(_pcbcore PRIVATE pcbcore pcbcore_kernels)
set_target_properties(_pcbcore PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/electrical")
