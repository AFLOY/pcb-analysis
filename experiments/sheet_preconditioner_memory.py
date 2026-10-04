"""Memory of the sheet-PEEC near preconditioner: whole saddle-point LU against alternatives.

Builds a two-layer plane of the power_module footprint (33 x 30 mm) at a given
pitch with a via field, forms the pieces solve_sheet_case forms, and measures,
each in its own subprocess (peak RSS from getrusage, SuperLU nnz):
  saddle         : splu([[Z_near, -A], [A^T, 0]])            (current "near")
  saddle_mmd     : the same with permc_spec="MMD_AT_PLUS_A"
  block          : splu(Z_near) + splu(A^T diag(Z)^-1 A)      (constraint-type block preconditioner)
  diagonal       : splu(A^T diag(Z)^-1 A) only                 (current "diagonal")

    .venv/bin/python experiments/sheet_preconditioner_memory.py --pitch-mm 0.2
"""
from __future__ import annotations

import argparse, json, resource, subprocess, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build(pitch_mm: float):
    from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
    from electrical.sheet_peec.sheet_peec import SheetMesh, ViaBranch
    rows, cols = int(round(30.3 / pitch_mm)), int(round(33.0 / pitch_mm))
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, 35e-6, 1.724e-8), SheetLayer("B.Cu", -1.6e-3, 35e-6, 1.724e-8)))
    step = max(1, rows // 8)
    vias = tuple(ViaBranch(r, c, 0, 1, resistance_ohm=1e-3) for r in range(step, rows, step) for c in range(step, cols, step))
    mesh = SheetMesh((rows, cols), pitch_mm * 1e-3, stackup, np.ones((2, rows, cols), dtype=bool), vias=vias)
    operator = SheetInductanceOperator((rows, cols), pitch_mm * 1e-3, stackup, vertical_levels=mesh.vertical_levels)
    return mesh, operator


def pieces(pitch_mm: float, frequency_hz: float):
    import scipy.sparse as sp
    from electrical.sheet_peec.sheet_peec import _near_impedance, _inline_self_inductance, _vertical_self_inductance
    mesh, operator = build(pitch_mm)
    incidence = mesh.incidence()
    resistance = mesh.resistances()
    omega = 2 * np.pi * frequency_hz
    active = np.ones(mesh.branch_count, dtype=bool)
    keep = np.ones(mesh.node_count, dtype=bool); keep[0] = False
    reduced = incidence[:, keep].tocsr()
    near = _near_impedance(mesh, operator, omega, resistance, active)
    diag = resistance + 1j * omega * np.concatenate([_inline_self_inductance(mesh, operator), _vertical_self_inductance(mesh, operator)])
    return mesh, reduced, near, diag


def run_variant(variant: str, pitch_mm: float, frequency_hz: float) -> dict:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    t0 = time.perf_counter()
    mesh, reduced, near, diag = pieces(pitch_mm, frequency_hz)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    nnz = {}
    if variant in ("saddle", "saddle_mmd"):
        saddle = sp.bmat([[near, -reduced], [reduced.T, None]], format="csc")
        lu = spla.splu(saddle, permc_spec="MMD_AT_PLUS_A" if variant == "saddle_mmd" else "COLAMD")
        nnz = {"saddle": int(saddle.nnz), "LU": int(lu.L.nnz + lu.U.nnz)}
    elif variant == "block":
        lu_z = spla.splu(near.tocsc(), permc_spec="MMD_AT_PLUS_A")
        schur = (reduced.T @ sp.diags(1.0 / diag) @ reduced).tocsc()
        lu_s = spla.splu(schur, permc_spec="MMD_AT_PLUS_A")
        nnz = {"Z_near": int(near.nnz), "LU_Z": int(lu_z.L.nnz + lu_z.U.nnz), "schur": int(schur.nnz), "LU_S": int(lu_s.L.nnz + lu_s.U.nnz)}
    elif variant == "diagonal":
        schur = (reduced.T @ sp.diags(1.0 / diag) @ reduced).tocsc()
        lu_s = spla.splu(schur)
        nnz = {"schur": int(schur.nnz), "LU_S": int(lu_s.L.nnz + lu_s.U.nnz)}
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {"variant": variant, "branches": int(mesh.branch_count), "nodes": int(mesh.node_count), "nnz": nnz,
            "peak_rss_mb_before_factor": round(rss_before), "peak_rss_mb": round(rss_after), "factor_mb": round(rss_after - rss_before), "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitch-mm", type=float, default=0.2)
    parser.add_argument("--frequency-hz", type=float, default=300e3)
    parser.add_argument("--variants", default="diagonal,block,saddle_mmd,saddle")
    parser.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.variant:
        print(json.dumps(run_variant(args.variant, args.pitch_mm, args.frequency_hz)))
        return
    for variant in args.variants.split(","):
        done = subprocess.run([sys.executable, __file__, "--pitch-mm", str(args.pitch_mm), "--frequency-hz", str(args.frequency_hz), "--variant", variant],
                              capture_output=True, text=True, timeout=3600)
        if done.returncode != 0:
            print(variant, "FAILED:", done.stderr.strip().splitlines()[-1][:200]); continue
        r = json.loads(done.stdout.strip().splitlines()[-1])
        print(f"{variant:11s} branches {r['branches']:>8,} nodes {r['nodes']:>8,}  factor {r['factor_mb']:>6} MB  peak {r['peak_rss_mb']:>6} MB  {r['wall_s']:6.1f} s  nnz {r['nnz']}")


if __name__ == "__main__":
    main()
