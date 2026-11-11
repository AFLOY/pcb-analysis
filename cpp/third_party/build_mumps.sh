#!/usr/bin/env bash
# Build MUMPS (sequential, complex double, OpenMP) with METIS and a
# single-threaded OpenBLAS into PREFIX, for -DPCB_NATIVE_MUMPS=ON.
#
#   CC=clang CXX=clang++ FC=flang PREFIX=~/opt/mumps-aocc cpp/third_party/build_mumps.sh
#   cmake -S . -B build/native -DPCB_NATIVE_MUMPS=ON -DMUMPS_ROOT=$PREFIX \
#         -DCMAKE_Fortran_COMPILER=$FC ...
#
# Use the compilers the C++ core is built with, so one OpenMP runtime serves
# the whole process (AOCC: LLVM libomp; GCC: libgomp).  OpenBLAS is built
# with USE_THREAD=0: it never starts a thread, and MUMPS's OpenMP team
# (ICNTL(16), set per call from the thread budget) is the only parallelism.
#
# Environment:
#   CC, CXX, FC        compilers (default gcc, g++, gfortran)
#   PREFIX             install prefix (default $HOME/opt/mumps)
#   SRC                download and build directory (default $PREFIX/src)
#   OPENBLAS_TARGET    OpenBLAS TARGET (default: detected by OpenBLAS)
#   JOBS               parallel build jobs (default 8)
set -euo pipefail

CC=${CC:-gcc}
CXX=${CXX:-g++}
FC=${FC:-gfortran}
PREFIX=${PREFIX:-$HOME/opt/mumps}
SRC=${SRC:-$PREFIX/src}
JOBS=${JOBS:-8}
OPENBLAS_VERSION=0.3.34
METIS_COMMIT=456244e99d7f0027706033d18bc1b3424d3507cc   # scivision/metis: METIS 5.1 with a CMake build
MUMPS_CMAKE_TAG=v5.9.1.2                                  # scivision/mumps: CMake build of MUMPS 5.9.1

mkdir -p "$PREFIX" "$SRC"
cd "$SRC"

fetch() {  # url, directory
    if [ ! -d "$2" ]; then
        curl -sSL -o "$2.tar.gz" "$1"
        mkdir -p "$2"
        tar -xzf "$2.tar.gz" -C "$2" --strip-components=1
    fi
}

# --- METIS (static, position-independent: it ends up inside a Python module)
fetch "https://github.com/scivision/metis/archive/${METIS_COMMIT}.tar.gz" metis-src
cmake -S metis-src -B metis-build -DCMAKE_C_COMPILER="$CC" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build metis-build -j "$JOBS"
cmake --install metis-build

# --- OpenBLAS with its LAPACK, single-threaded, static
fetch "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" openblas-src
# USE_LOCKING=1: a single-threaded OpenBLAS shares its work buffers between
# callers unless it locks them, and MUMPS calls BLAS from every thread of its
# OpenMP team.  Without it, MUMPS on 8-48 threads returned factors whose
# solves had residuals of 1e25 to NaN (Flow2, GCC 15, 2026-11-11); one
# thread was right.
openblas_options=(CC="$CC" FC="$FC" USE_THREAD=0 USE_OPENMP=0 USE_LOCKING=1 NUM_THREADS=1 NO_SHARED=1 DYNAMIC_ARCH=0)
if [ -n "${OPENBLAS_TARGET:-}" ]; then
    openblas_options+=(TARGET="$OPENBLAS_TARGET")
fi
# libs in parallel, then the LAPACK archive serially: a parallel netlib step
# raced on the archive and truncated it.
# A build left by other options is cleaned: make does not track them.
stamp="USE_LOCKING=1"
if [ "$(cat openblas-src/.pcb_options 2>/dev/null)" != "$stamp" ]; then
    make -C openblas-src clean > /dev/null
    echo "$stamp" > openblas-src/.pcb_options
fi
make -C openblas-src -j "$JOBS" "${openblas_options[@]}" libs
make -C openblas-src -j 1 "${openblas_options[@]}" netlib
make -C openblas-src "${openblas_options[@]}" PREFIX="$PREFIX" install

# --- MUMPS: sequential (MPI stub), complex double only, METIS ordering, OpenMP
if [ ! -d mumps-cmake ]; then
    git clone -q --depth 1 --branch "$MUMPS_CMAKE_TAG" https://github.com/scivision/mumps.git mumps-cmake
fi
cmake -S mumps-cmake -B mumps-build \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_Fortran_COMPILER="$FC" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DMUMPS_parallel=OFF -DMUMPS_openmp=ON -DMUMPS_metis=ON -DMUMPS_scotch=OFF \
    -DBUILD_SINGLE=OFF -DBUILD_DOUBLE=OFF -DBUILD_COMPLEX=OFF -DBUILD_COMPLEX16=ON \
    -DMUMPS_BUILD_TESTING=OFF -DLAPACK_VENDOR=OpenBLAS -DLAPACK_ROOT="$PREFIX" -DMETIS_ROOT="$PREFIX"
cmake --build mumps-build -j "$JOBS"
cmake --install mumps-build
echo "MUMPS installed in $PREFIX"
