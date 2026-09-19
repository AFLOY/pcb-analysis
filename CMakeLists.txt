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
project(pcb_analysis_native LANGUAGES CXX)

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

# One native module: ``name`` is the importable module, ``package_dir`` the
# package directory the module is written into (relative to src/).
function(pcb_native_module name package_dir)
  set(sources ${ARGN})
  pybind11_add_module(${name} MODULE ${sources})
  target_compile_options(${name} PRIVATE
    -O3 -fcx-limited-range -fno-math-errno -DNDEBUG)
  if(PCB_NATIVE_MARCH)
    target_compile_options(${name} PRIVATE
      "-march=${PCB_NATIVE_MARCH}" -mprefer-vector-width=512)
  endif()
  if(PCB_NATIVE_OPENMP)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  set_target_properties(${name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/${package_dir}")
endfunction()

pcb_native_module(_scalar_maxwell_native electrical/matrix_free_mpir_fem
  src/electrical/matrix_free_mpir_fem/native/scalar_maxwell_q1.cpp)
pcb_native_module(_thermal_native thermal/matrix_free_mpir_fem
  src/thermal/matrix_free_mpir_fem/native/hex_q1.cpp)
pcb_native_module(_dipole_native emc/tiled_dipole_superposition
  src/emc/tiled_dipole_superposition/native/dipole_fields.cpp)
pcb_native_module(_voxelize_native geometry/cad_import
  src/geometry/cad_import/native/point_in_mesh.cpp)
