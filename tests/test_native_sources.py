"""Source rules the native kernels keep so that the thread budget holds.

The budget is the only threading control: every OpenMP region names its team
(``num_threads``) from the budget the Python side passes in, nothing changes
the runtime's process defaults, and Eigen never starts a team of its own.  A
region without ``num_threads`` would run on whatever the runtime defaults to
(every logical CPU unless something else set it), which is how a solve ends up
on hyper-threads or oversubscribing the machine.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    path
    for pattern in ("*.cpp", "*.hpp", "*.cu", "*.cuh")
    for root in (ROOT / "src", ROOT / "cpp")
    for path in root.rglob(pattern)
)


def _pragmas(text: str) -> list[str]:
    """Each ``#pragma omp`` directive with its backslash continuations joined."""

    joined = re.sub(r"\\\n", " ", text)
    return [line.strip() for line in joined.splitlines() if re.match(r"\s*#\s*pragma\s+omp\b", line)]


def test_there_are_native_sources_to_check() -> None:
    assert SOURCES


def test_every_parallel_region_names_its_team() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}: {pragma}"
        for path in SOURCES
        for pragma in _pragmas(path.read_text())
        if re.search(r"\bparallel\b", pragma) and "num_threads(" not in pragma
    ]
    assert not offenders, "OpenMP regions without num_threads:\n" + "\n".join(offenders)


def test_nothing_changes_the_runtime_defaults() -> None:
    forbidden = re.compile(r"\b(omp_set_num_threads|omp_set_dynamic|omp_set_max_active_levels|setNbThreads)\s*\(")
    offenders = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in SOURCES
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if forbidden.search(line)
    ]
    assert not offenders, "calls that change OpenMP or Eigen process defaults:\n" + "\n".join(offenders)


def test_no_reduction_is_left_to_the_vectoriser() -> None:
    # An ``omp simd reduction`` splits its terms by the alignment of the first
    # one, which changes with every allocation: the same solve rounds
    # differently from run to run.  Sums go through pcbcore::lane_sum.
    offenders = [
        f"{path.relative_to(ROOT)}: {pragma}"
        for path in SOURCES
        for pragma in _pragmas(path.read_text())
        if re.search(r"\bsimd\b", pragma) and "reduction" in pragma
    ]
    assert not offenders, "vectoriser reductions:\n" + "\n".join(offenders)


def test_eigen_is_built_without_its_own_parallelism() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "EIGEN_DONT_PARALLELIZE" in cmake
