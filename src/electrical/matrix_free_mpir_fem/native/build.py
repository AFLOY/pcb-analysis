"""Build the optional C++ extension in place.

Usage: ``python -m electrical.matrix_free_mpir_fem.native.build [--no-openmp]``

The compiled module is written next to this file and is not tracked by Git.
It is an experiment-branch build step; packaging it as a wheel extension is a
separate decision.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path


def build(*, openmp: bool = True, verbose: bool = True) -> Path:
    import pybind11

    here = Path(__file__).resolve().parent
    source = here / "scalar_maxwell_q1.cpp"
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    output = here.parent / f"_scalar_maxwell_native{suffix}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-openmp", action="store_true")
    args = parser.parse_args()
    output = build(openmp=not args.no_openmp)
    print(f"built {output}")


if __name__ == "__main__":
    sys.exit(main())
