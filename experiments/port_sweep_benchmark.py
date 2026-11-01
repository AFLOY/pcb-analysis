"""Interpolating the copper's Z(f) between solves: which forms, how many solves, how close between the checks.

For each pair of forms (R, L) and each tolerance, ``refine_frequencies`` runs
over a band from its two ends; the sweep is then compared, not at its own
checks, but at validation frequencies solved directly.  Two coppers:

- ``synthetic``: two ports, R from its DC value into sqrt(f) at 200 kHz, L
  falling from 12 to 9 nH around 1 MHz, 10 kHz..100 MHz, 80 validation points;
- ``two strips``: the port-impedance contract fixture (5 x 40 cell strips,
  F.Cu over In1.Cu, 0.2 mm cells) with up to 6 skin filaments per layer,
  0.1..100 MHz, 30 validation points solved by the contract.

Writes a JSON with ``environment`` and ``decision``.

    .venv/bin/python experiments/port_sweep_benchmark.py --out docs/PORT_SWEEP_RESULTS.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

FORMS = [("linear-sqrt", "pchip-log"), ("pchip-sqrt", "pchip-log"), ("pchip-log", "pchip-log"), ("linear-log", "linear-log")]
TOLERANCES = (1e-2, 2e-3)
CANDIDATE = ("pchip-sqrt", "pchip-log")


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--maximum-solves", type=int, default=80)
    args = parser.parse_args()

    import scipy

    from electrical.port_network import refine_frequencies
    from electrical.sheet_peec import solve_port_impedance_problem
    from test_port_impedance_contract import _problem
    from test_port_sweep import _synthetic

    settings = {"maximum_filaments": 6, "relative_tolerance": 1e-10}
    cache: dict[float, object] = {}
    solve_seconds: list[float] = []

    def strips(f: float):
        f = float(f)
        if f not in cache:
            started = time.perf_counter()
            cache[f] = solve_port_impedance_problem(_problem(frequencies=(f,)), settings).impedances[0]
            solve_seconds.append(time.perf_counter() - started)
        return cache[f]

    coppers = {
        "synthetic": (_synthetic, (1e4, 1e8), np.logspace(4.01, 7.99, 80)),
        "two strips": (strips, (1e5, 1e8), np.logspace(5.02, 7.98, 30)),
    }
    rows = []
    for name, (solve, band, validation) in coppers.items():
        exact = {float(f): np.asarray(solve(f).z_ohm) for f in validation}
        for forms in FORMS:
            for tolerance in TOLERANCES:
                sweep = refine_frequencies(solve, band, tolerance=tolerance, maximum_solves=args.maximum_solves,
                                           forms=forms)
                worst_r = worst_l = 0.0
                for f, z in exact.items():
                    guess = sweep.at(f).z_ohm
                    worst_r = max(worst_r, float(np.abs(guess.real - z.real).max() / np.abs(z.real).max()))
                    worst_l = max(worst_l, float(np.abs(guess.imag - z.imag).max() / np.abs(z.imag).max()))
                rows.append({
                    "copper": name, "band_hz": list(band), "resistance_form": forms[0], "inductance_form": forms[1],
                    "tolerance": tolerance, "solves": len(sweep.points), "converged": sweep.converged,
                    "validation_points": len(exact), "worst_validation_resistance_error": worst_r,
                    "worst_validation_inductance_error": worst_l,
                })
                print(json.dumps(rows[-1]))

    def row(copper, forms, tolerance):
        return next(r for r in rows if r["copper"] == copper and (r["resistance_form"], r["inductance_form"]) == forms
                    and r["tolerance"] == tolerance)

    # adopted when, at 1 %, the candidate converges, holds 1 % between its checks on both coppers,
    # and spends no more solves than any other pair that also holds it
    holding = {}
    for forms in FORMS:
        ok = all(row(c, forms, 1e-2)["converged"] and row(c, forms, 1e-2)["worst_validation_resistance_error"] <= 1e-2
                 and row(c, forms, 1e-2)["worst_validation_inductance_error"] <= 1e-2 for c in coppers)
        holding[forms] = ok
    cost = {forms: sum(row(c, forms, 1e-2)["solves"] for c in coppers) for forms in FORMS}
    adopted = holding[CANDIDATE] and all(cost[CANDIDATE] <= cost[f] for f in FORMS if holding[f])
    result = {
        "benchmark": "port-impedance sweep: interpolation forms and frequency refinement",
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"cpu": _cpu_model(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
                        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "device": "CPU"},
        "settings": {"two_strips": settings, "minimum_ratio": 1.05, "maximum_solves": args.maximum_solves,
                     "two_strips_mean_solve_s": float(np.mean(solve_seconds)) if solve_seconds else None},
        "rows": rows,
        "decision": {
            "adopted": adopted,
            "candidate": list(CANDIDATE),
            "criteria": ("at a 1 % tolerance the candidate converges and its sweep is within 1 % in R and L at every "
                         "validation point of both coppers, with no more solves than any other pair that also does"),
            "holding_at_1_percent": {f"{r}/{l}": ok for (r, l), ok in holding.items()},
            "solves_at_1_percent": {f"{r}/{l}": n for (r, l), n in cost.items()},
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["decision"], indent=1))


if __name__ == "__main__":
    main()
