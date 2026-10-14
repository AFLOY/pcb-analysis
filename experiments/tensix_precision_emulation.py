"""CPU emulation of Tenstorrent Tensix number formats in the FP32 inner PCG.

Before any Tensix kernel exists, this asks which part of the matrix-free MPIR
inner solve tolerates the formats the Tensix matrix engine (FPU) consumes.  The
FPU reads its operands from 19-bit source registers, so FP32 inputs arrive as
TF32 (10 mantissa bits); BF16 keeps 7, BFP8_B/BFP4_B share one exponent per 16
values of a row and keep 7/3 magnitude bits.  Accumulation is FP32, as with
``fp32_dest_acc_en``.  Vector updates, Jacobi scaling and dot products stay FP32
(SFPU work), exactly as in the NumPy FP32 runtime.

Each variant wraps the portable NumPy operator (``native=False``) and rounds
only the FPU inputs of one component:

* ``coarse-*``: the dense ``(Z^T A Z)^-1`` GEMV of the two-level preconditioner.
  The matrix is rounded to the format, the restricted residual to the matching
  activation format, and the product is accumulated in FP32.
* ``stencil-*``: the operator input vector is rounded before the operator's
  own FP32 action (FP32 coefficients, FP32 accumulation).  Coefficient
  rounding is not emulated, so this is a lower bound on the damage of running
  the stencil on the FPU.  ``bf16xK`` splits the vector into K BF16 pieces,
  applies the FP32 action to each (K FPU passes) and sums them in FP32.
* ``coarse-factor-*``: the coarse inverse kept as its lower Cholesky factor
  ``R`` (``(Z^T A Z)^-1 = R R^T``) in the format, applied as ``R (r^T R)^T``
  from one stored matrix.  Rounding then leaves the correction ``R~ R~^T``
  symmetric positive semidefinite, which a rounded inverse does not guarantee.

Rounding is round-to-nearest-even; the device's own conversion is measured
separately on hardware.  Fixtures are the DC board of dc_native_benchmark.py
and the thermal stack of thermal_native_benchmark.py with their default
MPIRConfig.  The reference is the unmodified portable FP32 solve.  Wall times
are CPU emulation times and are not a Tensix performance estimate.

    .venv/bin/python experiments/tensix_precision_emulation.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_native_benchmark import CONFIG as DC_CONFIG, board  # noqa: E402
from electrical.matrix_free_mpir_fem import MatrixFreePCBOperator, MPIRConfig, solve_mpir  # noqa: E402
from thermal.matrix_free_mpir_fem import MatrixFreeThermalOperator  # noqa: E402
from thermal_native_benchmark import CONFIG as THERMAL_CONFIG, stack  # noqa: E402
from tensix_formats import MATRIX_FORMATS, STENCIL_FORMATS  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    coarse: str | None  # None: the operator's own preconditioner
    stencil: str | None  # None: the operator's own FP32 action
    factored: bool = False  # coarse inverse kept as its Cholesky factor


VARIANTS = (
    Variant("fp32-stock", None, None),
    Variant("coarse-fp32", "fp32", None),
    Variant("coarse-tf32", "tf32", None),
    Variant("coarse-bf16", "bf16", None),
    Variant("coarse-bfp8b", "bfp8b", None),
    Variant("coarse-bfp4b", "bfp4b", None),
    Variant("coarse-factor-bf16", "bf16", None, True),
    Variant("coarse-factor-bfp8b", "bfp8b", None, True),
    Variant("coarse-factor-bfp4b", "bfp4b", None, True),
    Variant("stencil-tf32", "fp32", "tf32"),
    Variant("stencil-bf16", "fp32", "bf16"),
    Variant("stencil-bf16x2", "fp32", "bf16x2"),
    Variant("stencil-bf16x3", "fp32", "bf16x3"),
    Variant("stencil-bf16x3+coarse-factor-bfp8b", "bfp8b", "bf16x3", True),
)


# --------------------------------------------------------- emulated system
class EmulatedSystem:
    """The portable operator with one component's FPU inputs rounded."""

    inner_solver = "pcg"
    high_dtype = np.float64

    def __init__(self, operator: Any, variant: Variant) -> None:
        self.operator = operator
        self.variant = variant
        self.size = operator.size
        self.runtime = operator.runtime
        correction = operator.coarse_correction
        self.node_shape = tuple(int(axis) for axis in correction.node_shape)
        self.coarse_shape = tuple(int(axis) for axis in correction.coarse_shape)
        self.block = int(correction.block)
        self.free = np.asarray(operator.free_nodes, dtype=bool).reshape(self.node_shape)
        self.diagonal = np.asarray(operator.diagonal_low(), dtype=np.float32)
        self.format = None if variant.coarse is None else MATRIX_FORMATS[variant.coarse]
        if self.format is not None:
            inverse = np.linalg.inv(correction.coarse_matrix)
            inverse = 0.5 * (inverse + inverse.T)
            self.coarse_inverse_fp32 = inverse.astype(np.float32)
            if variant.factored:
                factor = np.linalg.cholesky(inverse)
                self.coarse_factor = self.format.matrix(factor)
                self.coarse_inverse = self.coarse_factor.astype(np.float64) @ self.coarse_factor.T.astype(np.float64)
            else:
                self.coarse_inverse = self.format.matrix(inverse)
        self.stencil = None if variant.stencil is None else STENCIL_FORMATS[variant.stencil]
        self.free_flat = self.free.reshape(-1)

    def apply_high(self, vector: np.ndarray) -> np.ndarray:
        return self.operator.apply_high(vector)

    def diagonal_low(self) -> Any:
        return self.operator.diagonal_low()

    def apply_low(self, vector: np.ndarray) -> np.ndarray:
        if self.stencil is None:
            return self.operator.apply_low(vector)
        pieces = self.stencil(vector)
        action = self.operator.apply_low(pieces[0])
        for piece in pieces[1:]:
            action = action + self.operator.apply_low(piece)
        return np.where(self.free_flat, action, vector).astype(np.float32)

    def _restrict(self, grid: np.ndarray) -> np.ndarray:
        layers, rows, cols = self.node_shape
        padded = np.zeros((layers, self.coarse_shape[1] * self.block, self.coarse_shape[2] * self.block), dtype=grid.dtype)
        padded[:, :rows, :cols] = grid
        return padded.reshape(layers, self.coarse_shape[1], self.block, self.coarse_shape[2], self.block).sum(axis=(2, 4))

    def _prolong(self, coarse: np.ndarray) -> np.ndarray:
        _, rows, cols = self.node_shape
        grid = coarse.reshape(self.coarse_shape)
        return np.repeat(np.repeat(grid, self.block, axis=1), self.block, axis=2)[:, :rows, :cols]

    def precondition_low(self, vector: np.ndarray) -> np.ndarray:
        if self.format is None:
            return self.operator.precondition_low(vector)
        flat = np.asarray(vector, dtype=np.float32).reshape(-1)
        smoothed = flat / self.diagonal
        grid = np.where(self.free, flat.reshape(self.node_shape), np.float32(0.0))
        restricted = self.format.vector(self._restrict(grid).reshape(-1))
        if self.variant.factored:
            # r^T R then R t, both from the one stored factor; the intermediate
            # leaves the FPU as FP32 and re-enters in the activation format.
            inner = self.format.vector(np.matmul(restricted, self.coarse_factor).astype(np.float32))
            coarse = np.matmul(self.coarse_factor, inner).astype(np.float32)
        else:
            coarse = np.matmul(self.coarse_inverse, restricted).astype(np.float32)
        correction = np.where(self.free, self._prolong(coarse), np.float32(0.0))
        return (smoothed + correction.reshape(-1)).astype(np.float32)

    def coarse_matrix_error(self) -> float | None:
        if self.format is None:
            return None
        reference = self.coarse_inverse_fp32.astype(np.float64)
        return float(np.linalg.norm(self.coarse_inverse - reference) / np.linalg.norm(reference))

    def coarse_asymmetry(self) -> float | None:
        if self.format is None:
            return None
        matrix = self.coarse_inverse.astype(np.float64)
        return float(np.linalg.norm(matrix - matrix.T) / np.linalg.norm(matrix))


# ------------------------------------------------------------------ cases
def _solve(system: EmulatedSystem, rhs: np.ndarray, config: MPIRConfig) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = solve_mpir(system, rhs, config=config)
    except RuntimeError as exc:  # the inner PCG rejects a non-SPD curvature
        return {"failed": str(exc), "solve_ms": (time.perf_counter() - start) * 1e3}
    return {
        "solve_ms": (time.perf_counter() - start) * 1e3,
        "converged": bool(result.converged),
        "relative_residual": float(result.relative_residual),
        "outer_iterations": int(result.outer_iterations),
        "inner_iterations": int(result.inner_iterations),
        "low_operator_applications": int(result.low_operator_applications),
        "high_operator_applications": int(result.high_operator_applications),
        "history": [[step.outer_iteration, step.high_relative_residual, step.inner_iterations, step.inner_relative_residual]
                    for step in result.history],
        "_solution": result.solution,
    }


def _case(physics: str, elements: int, operator: Any, rhs: np.ndarray, config: MPIRConfig, variants: list[Variant]) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    reference: np.ndarray | None = None
    for variant in variants:
        system = EmulatedSystem(operator, variant)
        run = _solve(system, rhs, config)
        run["coarse_format"] = variant.coarse
        run["stencil_format"] = variant.stencil
        run["coarse_matrix_relative_error_vs_fp32"] = system.coarse_matrix_error()
        run["coarse_matrix_asymmetry"] = system.coarse_asymmetry()
        if system.format is not None:
            run["coarse_inverse_bytes"] = int(round(system.format.bytes_per_value * system.coarse_inverse.size))
        solution = run.pop("_solution", None)
        if variant.name == "fp32-stock":
            reference = solution
        if solution is not None and reference is not None:
            difference = solution - reference
            run["relative_solution_error_vs_stock"] = float(np.linalg.norm(difference) / np.linalg.norm(reference))
            run["relative_max_error_vs_stock"] = float(np.abs(difference).max() / np.abs(reference).max())
        print(f"{physics:7s} {elements:4d} {variant.name:28s} "
              + (f"failed: {run['failed']}" if "failed" in run else
                 f"conv={run['converged']} outer={run['outer_iterations']:2d} inner={run['inner_iterations']:5d} "
                 f"res={run['relative_residual']:.2e} err={run.get('relative_solution_error_vs_stock', 0.0):.2e} "
                 f"{run['solve_ms'] / 1e3:7.2f} s"), flush=True)
        runs[variant.name] = run
    correction = operator.coarse_correction
    return {
        "physics": physics,
        "elements": [elements, elements],
        "nodes": int(operator.size),
        "coarse_size": int(correction.coarse_size),
        "coarse_block": int(correction.block),
        "config": {"relative_tolerance": config.relative_tolerance, "inner_relative_tolerance": config.inner_relative_tolerance,
                   "max_outer_iterations": config.max_outer_iterations, "max_inner_iterations": config.max_inner_iterations},
        "runs": runs,
    }


def verdict(run: dict[str, Any], stock: dict[str, Any], config: MPIRConfig) -> bool:
    if "failed" in run or not run["converged"] or run["relative_residual"] > config.relative_tolerance:
        return False
    return (run["outer_iterations"] <= stock["outer_iterations"] + 1
            and run["inner_iterations"] <= 1.10 * stock["inner_iterations"]
            and run["relative_solution_error_vs_stock"] <= 100.0 * config.relative_tolerance)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def run(dc_sizes: list[int], thermal_sizes: list[int], variants: list[Variant]) -> dict[str, Any]:
    cases = []
    for elements in dc_sizes:
        problem = board(elements)
        operator = MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias, native=False)
        cases.append(_case("dc", elements, operator, operator.build_rhs(problem.terminals), DC_CONFIG, variants))
    for elements in thermal_sizes:
        operator = MatrixFreeThermalOperator(stack(elements), native=False)
        rhs = operator.build_rhs(operator.default_reference_temperature())
        cases.append(_case("thermal", elements, operator, rhs, THERMAL_CONFIG, variants))

    verdicts: dict[str, Any] = {}
    for variant in variants[1:]:
        per_case = [verdict(case["runs"][variant.name], case["runs"]["fp32-stock"],
                             DC_CONFIG if case["physics"] == "dc" else THERMAL_CONFIG) for case in cases]
        verdicts[variant.name] = all(per_case)
    candidates = [name for name, ok in verdicts.items() if ok and name != "coarse-fp32"]
    return {
        "schema": "tensix-precision-emulation/v1",
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "device": "cpu (Tensix formats emulated; no Tenstorrent device used)",
        },
        "definitions": {
            "reference": "fp32-stock: unmodified portable NumPy FP32 inner PCG with the two-level preconditioner",
            "coarse_format": "format of (Z^T A Z)^-1 on the FPU; the restricted residual uses fp32/tf32/bf16 to match, bf16 for block floats; FP32 accumulation",
            "coarse_factor": "factored variants store the lower Cholesky factor R of (Z^T A Z)^-1 once and apply R (r^T R)^T; coarse_inverse_bytes counts the dense n x n factor; coarse_matrix_relative_error_vs_fp32 compares R~ R~^T",
            "stencil_format": "format of the operator input vector fed to the portable FP32 action; bf16xK sums K FP32 actions of BF16 pieces in FP32; coefficient rounding not emulated",
            "rounding": "round-to-nearest-even; block floats share the largest exponent of 16 consecutive row values",
            "solve_ms": "CPU emulation wall time, not a Tensix estimate",
            "viable": "every case converges to the outer tolerance with at most one extra outer step, at most 10% more inner iterations, and a solution within 100 x relative_tolerance of fp32-stock",
        },
        "cases": cases,
        "decision": {
            "adopted": False,
            "scope": "precision feasibility only; nothing in src/ changes. Viable variants are carried to the Blackhole microbenchmark",
            "viable_by_variant": verdicts,
            "candidates_for_device_measurement": candidates,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dc-sizes", default="100,200,320", help="comma separated in-plane element counts of the DC board")
    parser.add_argument("--thermal-sizes", default="50,100,200", help="comma separated in-plane element counts of the thermal stack")
    parser.add_argument("--variants", default="", help="comma separated variant names; empty runs all (fp32-stock is always first)")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "tensix_precision_emulation.json")
    args = parser.parse_args()
    sizes = lambda text: [int(item) for item in text.split(",") if item]  # noqa: E731
    wanted = {name for name in args.variants.split(",") if name}
    variants = [variant for variant in VARIANTS if variant.name == "fp32-stock" or not wanted or variant.name in wanted]
    report = run(sizes(args.dc_sizes), sizes(args.thermal_sizes), variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
