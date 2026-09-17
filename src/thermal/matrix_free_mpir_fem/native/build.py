"""Build the thermal C++ extension in place.

Usage: ``python -m thermal.matrix_free_mpir_fem.native.build [--no-openmp]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from electrical.matrix_free_mpir_fem.native.build import compile_extension


def build(*, openmp: bool = True, verbose: bool = True) -> Path:
    here = Path(__file__).resolve().parent
    return compile_extension(
        here / "hex_q1.cpp", "_thermal_native", here.parent, openmp=openmp, verbose=verbose
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-openmp", action="store_true")
    args = parser.parse_args()
    print(f"built {build(openmp=not args.no_openmp)}")


if __name__ == "__main__":
    sys.exit(main())
