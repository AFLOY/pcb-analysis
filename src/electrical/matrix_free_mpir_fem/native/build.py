"""Build the optional C++ extensions in place.

Usage: ``python -m electrical.matrix_free_mpir_fem.native.build [--no-openmp]``

The compiled module is written next to the package and is not tracked by Git.
``compile_extension`` is shared by the thermal and emc packages, whose build
modules pass their own source file and module name.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path


def compile_extension(
    source: Path,
    module_name: str,
    output_dir: Path,
    *,
    openmp: bool = True,
    verbose: bool = True,
) -> Path:
    """Compile one pybind11 source into ``output_dir/<module_name><EXT_SUFFIX>``."""

    import pybind11

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    output = Path(output_dir) / f"{module_name}{suffix}"
    command = [
        "g++",
        "-O3",
        "-march=native",
        "-mprefer-vector-width=512",
        "-fcx-limited-range",
        "-fno-math-errno",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        "-DNDEBUG",
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_paths()['include']}",
        str(source),
        "-o",
        str(output),
    ]
    if openmp:
        command.insert(1, "-fopenmp")
    if verbose:
        print(" ".join(command))
    subprocess.run(command, check=True)
    return output


def build(*, openmp: bool = True, verbose: bool = True) -> Path:
    """Compile the scalar Maxwell kernel; returns its module path."""

    here = Path(__file__).resolve().parent
    return compile_extension(
        here / "scalar_maxwell_q1.cpp",
        "_scalar_maxwell_native",
        here.parent,
        openmp=openmp,
        verbose=verbose,
    )


def build_layered_dc(*, openmp: bool = True, verbose: bool = True) -> Path:
    """Compile the layered-PCB DC conduction kernel; returns its module path."""

    here = Path(__file__).resolve().parent
    return compile_extension(
        here / "layered_dc_q1.cpp",
        "_layered_dc_native",
        here.parent,
        openmp=openmp,
        verbose=verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-openmp", action="store_true")
    args = parser.parse_args()
    for output in (build(openmp=not args.no_openmp), build_layered_dc(openmp=not args.no_openmp)):
        print(f"built {output}")


if __name__ == "__main__":
    sys.exit(main())
