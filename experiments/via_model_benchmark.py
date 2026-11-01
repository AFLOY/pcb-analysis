"""The via model, current against candidate: barrel self term, filament far form, pFFT line kernels.

Each run measures one source tree (``--src``) on the same fixtures and
writes its numbers; ``combine`` puts a current and a candidate run into one
JSON with ``environment`` and ``decision``.

Fixtures (1.6 mm vias, a 0.3 mm drill with 25 um plating):

- ``pair``: a go/return pair 1 mm apart joined by a one-cell trace, at 0.10,
  0.20 and 0.25 mm cells; the vias' share of the loop (loop L less the same
  loop with no vertical inductance) against the barrels' closed form
  ``2 (L_barrel - M)``.
- ``array``: a 3 x 3 array carrying one current (each via its own island),
  against the barrels' closed form.
- ``pfft``: via structures read off the pFFT operator (near reach 4, the
  current-field contract's) against the uniform operator, no solve.
- ``plane``: the two-layer plane of ``sheet_preconditioner_benchmark.py``
  (33 x 30.3 mm, 0.25 mm cells, a via array) at 30 MHz: solve time,
  iterations, residual and the loop impedance, columns and (where the tree
  has them) barrels.

    PYTHONPATH= .venv/bin/python experiments/via_model_benchmark.py measure --src ../main/src \\
        --label current --out benchmark-results/via_current.json
    .venv/bin/python experiments/via_model_benchmark.py measure --src src --label candidate \\
        --out benchmark-results/via_candidate.json
    .venv/bin/python experiments/via_model_benchmark.py combine benchmark-results/via_current.json \\
        benchmark-results/via_candidate.json --out docs/SHEET_VIA_RESULTS.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SPAN, CU, RHO, FREQUENCY = 1.6e-3, 35e-6, 1.724e-8, 1e6
OUTER, BORE = 0.15e-3, 0.125e-3
MU0_OVER_4PI = 1e-7


def _filament(length: float, d: float) -> float:
    ratio = length / d
    return 2 * MU0_OVER_4PI * length * (math.log(ratio + math.sqrt(1 + ratio * ratio)) - math.sqrt(1 + 1 / ratio**2) + 1 / ratio)


def _barrel_self() -> float:
    r2, r1 = OUTER, BORE
    area = r2 * r2 - r1 * r1
    gmd = math.exp(math.log(r2) - r1**4 * math.log(r2 / r1) / area**2 + (3 * r1 * r1 - r2 * r2) / (4 * area))
    return _filament(SPAN, gmd)


class _NoVertical:
    """The operator with the vertical branches' inductance removed."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    def apply(self, x, y, z=None):
        if z is None:
            return self._base.apply(x, y)
        fx, fy, _ = self._base.apply(x, y, z)
        return fx, fy, np.zeros_like(z)

    def apply_many(self, currents):
        return [self.apply(x, y, z) if z is not None else self.apply(x, y) for x, y, z in currents]

    def near_inductance(self, mesh, *args, **kwargs):
        import scipy.sparse as sp

        near = self._base.near_inductance(mesh, *args, **kwargs).tocoo()
        first = len(mesh.branch_x) + len(mesh.branch_y)
        keep = (near.row < first) & (near.col < first)
        return sp.csr_matrix((near.data[keep], (near.row[keep], near.col[keep])), shape=near.shape)

    def vertical_self_inductance(self, mesh):
        return np.zeros(len(mesh.via_branches))


def _has_barrels() -> bool:
    from electrical.sheet_peec.sheet_peec import ViaBranch

    return "barrel_radius_m" in ViaBranch.__dataclass_fields__


def _via(row, col, lower, upper, barrel):
    from electrical.sheet_peec.sheet_peec import ViaBranch

    extra = {"barrel_radius_m": OUTER, "barrel_bore_radius_m": BORE} if barrel else {}
    return ViaBranch(row, col, lower, upper, 1e-6, **extra)


def pair(pitch: float, barrel: bool) -> dict:
    from electrical.sheet_peec import SheetInductanceOperator, SheetLayer, SheetMesh, SheetStackup, Terminal
    from electrical.sheet_peec import solve_sheet_case

    cells = round(1e-3 / pitch)
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, CU, RHO), SheetLayer("B.Cu", -SPAN, CU, RHO)))
    rows, cols = 5, cells + 5
    occupancy = np.zeros((2, rows, cols), bool)
    go, back = (2, 2), (2, 2 + cells)
    occupancy[0, 2, 2 : 3 + cells] = True
    occupancy[1, go[0], go[1]] = occupancy[1, back[0], back[1]] = True
    terminals = [Terminal("go", 1, (go,), 1.0), Terminal("back", 1, (back,), -1.0)]

    def loop(with_barrel: bool, copper_only: bool) -> float:
        mesh = SheetMesh((rows, cols), pitch, stackup, occupancy,
                         vias=(_via(*go, 1, 0, with_barrel), _via(*back, 1, 0, with_barrel)))
        operator = SheetInductanceOperator((rows, cols), pitch, stackup, vertical_levels=mesh.vertical_levels)
        solution = solve_sheet_case(mesh, _NoVertical(operator) if copper_only else operator, terminals,
                                    frequency_hz=FREQUENCY, tolerance=1e-10)
        v = solution.node_voltage[mesh.node_index[(1, *go)]] - solution.node_voltage[mesh.node_index[(1, *back)]]
        return v.imag / (2 * math.pi * FREQUENCY)

    copper = loop(False, True)
    reference = 2 * (_barrel_self() - _filament(SPAN, 1e-3))
    row = {"pitch_mm": pitch * 1e3, "reference_via_share_nH": reference * 1e9,
           "column_via_share_nH": (loop(False, False) - copper) * 1e9}
    row["column_error"] = row["column_via_share_nH"] / row["reference_via_share_nH"] - 1
    if barrel:
        row["barrel_via_share_nH"] = (loop(True, False) - copper) * 1e9
        row["barrel_error"] = row["barrel_via_share_nH"] / row["reference_via_share_nH"] - 1
    return row


def array(pitch: float, step: int, barrel: bool) -> dict:
    from electrical.sheet_peec import SheetInductanceOperator, SheetLayer, SheetMesh, SheetStackup, Terminal
    from electrical.sheet_peec import solve_sheet_case

    n = 3
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, CU, RHO), SheetLayer("B.Cu", -SPAN, CU, RHO)))
    size = (n - 1) * step + 3
    occupancy = np.zeros((2, size, size), bool)
    where = [(1 + i * step, 1 + j * step) for i in range(n) for j in range(n)]
    for cell in where:
        occupancy[:, cell[0], cell[1]] = True
    xy = [(r * pitch, c * pitch) for r, c in where]
    matrix = np.array([[_barrel_self() if i == j else _filament(SPAN, math.dist(xy[i], xy[j])) for j in range(9)]
                       for i in range(9)])
    row = {"pitch_mm": pitch * 1e3, "via_pitch_mm": pitch * step * 1e3, "reference_nH": matrix.sum() / 81 * 1e9}
    for name, with_barrel in (("column", False), ("barrel", True)):
        if with_barrel and not barrel:
            continue
        mesh = SheetMesh((size, size), pitch, stackup, occupancy, vias=tuple(_via(*c, 1, 0, with_barrel) for c in where))
        operator = SheetInductanceOperator((size, size), pitch, stackup, vertical_levels=mesh.vertical_levels)
        solution = solve_sheet_case(mesh, operator, [Terminal("t", 0, tuple(where), 1.0), Terminal("b", 1, tuple(where), -1.0)],
                                    frequency_hz=FREQUENCY, tolerance=1e-10)
        v = np.mean([solution.node_voltage[mesh.node_index[(0, *c)]] - solution.node_voltage[mesh.node_index[(1, *c)]]
                     for c in where])
        row[f"{name}_nH"] = v.imag / (2 * math.pi * FREQUENCY) * 1e9
        row[f"{name}_error"] = row[f"{name}_nH"] / row["reference_nH"] - 1
    return row


def pfft() -> dict:
    from electrical.matrix_free_mpir_fem.grid import TensorGrid
    from electrical.sheet_peec import SheetInductanceOperator, SheetLayer, SheetMesh, SheetStackup, ViaBranch
    from electrical.sheet_peec.sheet_pfft import PfftSheetInductanceOperator

    h, n = 0.1e-3, 40
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, CU, RHO), SheetLayer("B.Cu", -SPAN, CU, RHO)))
    mesh = SheetMesh((n, n), None, stackup, np.ones((2, n, n), bool), vias=(ViaBranch(5, 5, 1, 0, 1e-6),),
                     grid=TensorGrid.uniform(h, (n, n)))
    uniform = SheetInductanceOperator((n, n), h, stackup, vertical_levels=mesh.vertical_levels)
    graded = PfftSheetInductanceOperator(mesh, near_radius_cells=4)
    structures = {
        "3x3 at 1.0 mm, one current": [((5 + 10 * i, 5 + 10 * j), 1 / 9) for i in range(3) for j in range(3)],
        "3x3 at 0.5 mm, one current": [((5 + 5 * i, 5 + 5 * j), 1 / 9) for i in range(3) for j in range(3)],
        "go/return 1.0 mm": [((20, 5), 1.0), ((20, 15), -1.0)],
        "go/return 2.0 mm": [((20, 5), 1.0), ((20, 25), -1.0)],
        "2 go + 2 return interleaved 1.0 mm": [((10, 5), 1.0), ((10, 15), -1.0), ((20, 5), -1.0), ((20, 15), 1.0)],
    }
    zero = np.zeros((2, n, n))
    out = {}
    for name, sources in structures.items():
        currents = np.zeros((1, n, n))
        for (r, c), value in sources:
            currents[0, r, c] = value
        ref = float(np.sum(uniform.apply(zero, zero, currents)[2] * currents))
        approx = float(np.sum(graded.apply(zero, zero, currents)[2] * currents))
        out[name] = {"uniform_nH": ref * 1e9, "pfft_error": approx / ref - 1}
    return out


def plane(barrel: bool) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sheet_preconditioner_benchmark import build

    from electrical.sheet_peec import SheetInductanceOperator, SheetMesh
    from electrical.sheet_peec import solve_sheet_case

    out = {}
    for name, with_barrel in (("column", False), ("barrel", True)):
        if with_barrel and not barrel:
            continue
        mesh, operator, terminals = build(0.25)
        if with_barrel:
            vias = tuple(_via(v.row, v.col, v.lower_layer, v.upper_layer, True) for v in mesh.vias)
            mesh = SheetMesh(mesh.shape, mesh.pitch_m, mesh.stackup, mesh.occupancy, vias=vias)
            operator = SheetInductanceOperator(mesh.shape, mesh.pitch_m, mesh.stackup, vertical_levels=mesh.vertical_levels)
        started = time.perf_counter()
        solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=30e6, preconditioner="near",
                                    tolerance=1e-9, restart=120, max_iterations=400)
        wall = time.perf_counter() - started

        def mean(terminal):
            return np.mean([solution.node_voltage[mesh.node_index[(terminal.layer, r, c)]] for r, c in terminal.cells])

        z = (mean(terminals[0]) - mean(terminals[1])) / terminals[0].current_a
        out[name] = {"nodes": int(mesh.node_count), "branches": int(mesh.branch_count), "vias": len(mesh.via_branches),
                     "solve_s": round(wall, 3), "iterations": int(solution.iterations), "converged": bool(solution.converged),
                     "residual": float(solution.residual), "z_ohm": [float(z.real), float(z.imag)]}
    return out


def measure(args) -> None:
    sys.path.insert(0, str(Path(args.src).resolve()))
    import electrical

    try:
        from electrical import backend

        native = bool(backend.native_available())
    except ImportError:
        native = False
    barrel = _has_barrels()
    result = {
        "label": args.label,
        "source": str(Path(electrical.__file__).resolve().parent.parent),
        "capabilities": {"cpp_core": native, "barrel_vias": barrel},
        "pair": [pair(p, barrel) for p in (0.1e-3, 0.2e-3, 0.25e-3)],
        "array": [array(0.1e-3, 5, barrel), array(0.2e-3, 5, barrel)],
        "pfft": pfft(),
        "plane": plane(barrel),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def combine(args) -> None:
    current = json.loads(Path(args.current).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    pair_ok = all(abs(row["barrel_error"]) <= 0.01 for row in candidate["pair"])
    shares = [row["barrel_via_share_nH"] for row in candidate["pair"]]
    pitch_ok = (max(shares) - min(shares)) / min(shares) <= 0.02
    array_ok = all(abs(row["barrel_error"]) <= 0.01 for row in candidate["array"])
    pfft_ok = all(abs(v["pfft_error"]) <= 1e-3 for v in candidate["pfft"].values())
    plane_ok = candidate["plane"]["barrel"]["converged"] and candidate["plane"]["column"]["converged"]
    adopted = pair_ok and pitch_ok and array_ok and pfft_ok and plane_ok
    import scipy

    result = {
        "benchmark": "via model: barrel self term, filament far form, pFFT line kernels",
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "device": "CPU (no CUDA device on this host; CUDA not measured)",
        },
        "settings": {"frequency_hz": FREQUENCY, "via_span_mm": SPAN * 1e3, "barrel_outer_diameter_mm": 2 * OUTER * 1e3,
                     "barrel_drill_diameter_mm": 2 * BORE * 1e3, "tolerance": 1e-10,
                     "plane": {"frequency_hz": 30e6, "pitch_mm": 0.25, "preconditioner": "near", "tolerance": 1e-9}},
        "current": current,
        "candidate": candidate,
        "decision": {
            "adopted": adopted,
            "criteria": ("candidate: go/return via share within 1 % of the barrels' closed form at 0.10/0.20/0.25 mm "
                         "and within 2 % of itself across them; 3 x 3 arrays within 1 %; pFFT within 0.1 % of the "
                         "uniform operator on every via structure; the plane converges with columns and barrels"),
            "checks": {"pair": pair_ok, "pitch_independence": pitch_ok, "array": array_ok, "pfft": pfft_ok,
                       "plane_converged": plane_ok},
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["decision"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--src", required=True)
    m.add_argument("--label", required=True)
    m.add_argument("--out", required=True)
    c = sub.add_parser("combine")
    c.add_argument("current")
    c.add_argument("candidate")
    c.add_argument("--out", required=True)
    args = parser.parse_args()
    (measure if args.command == "measure" else combine)(args)


if __name__ == "__main__":
    main()
