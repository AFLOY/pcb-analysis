"""Same input, same bits: across processes, hash seeds and thread budgets.

Python randomises the hash of every string in each process, so a set of
names, or of cells keyed by a layer's name, iterates in another order in the
next process; a sum taken in that order moves in its last bits.  The
contracts take named layers, pads and parts, so each is solved here in three
processes, each with its own ``PYTHONHASHSEED`` and thread budget, and every
float it returns, in the order it returns them, must be the same.

On one machine and one build only: another CPU (other BLAS kernels), other
NumPy or SciPy versions, or the CUDA path may differ within the solver's
tolerance (``docs/DESIGN.md``, "Reproducibility").
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Runs in a fresh interpreter: argv[1] is the thread budget, argv[2] the case.
SCRIPT = r"""
import hashlib, sys
import numpy as np
from electrical.threads import set_thread_budget
set_thread_budget(int(sys.argv[1]))
case = sys.argv[2]
items = []

def add(value):
    if isinstance(value, (bool, int, str)) or value is None:
        items.append(repr(value))
    elif isinstance(value, float):
        items.append(value.hex())
    elif isinstance(value, complex):
        items.append(value.real.hex() + value.imag.hex())
    elif isinstance(value, np.ndarray):
        items.append(value.dtype.str + value.tobytes().hex())
    elif isinstance(value, dict):
        for key, item in value.items():  # the order returned, not sorted
            items.append(repr(key))
            add(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            add(item)
    else:
        items.append(repr(value))

if case == "current_field":
    from electrical.sheet_peec.current_field_contract import solve_current_field_problems
    from tests.test_current_field_contract import _problem
    first = _problem(frequency_hz=1000.0)
    second = _problem(frequency_hz=1000.0)
    second["name"] = "three_layer_other_way"
    second["terminals"][0]["cells"] = [{"layer": "In1.Cu", "x": 1, "y": 0}]
    for result in solve_current_field_problems([first, second, _problem(frequency_hz=0.0)]):
        add(result.voltage_phasor)
        add(result.current_density_phasor)
        add({k: v for k, v in result.metrics.items() if "seconds" not in k and not k.endswith("_ms")})
elif case == "port_impedance":
    from electrical.sheet_peec.port_impedance_contract import solve_port_impedance_problem
    from tests.test_port_impedance_contract import ONE_FILAMENT, _problem
    result = solve_port_impedance_problem(_problem(frequencies=(1e6, 1e8, 1e7), vias=True), ONE_FILAMENT)
    add(list(result.ports))
    for impedance in result.impedances:
        add(impedance.z_ohm)
else:
    raise SystemExit(f"unknown case {case}")
print(hashlib.sha256("|".join(items).encode()).hexdigest(), len(items))
"""


def _digest(case: str, seed: int, threads: int) -> str:
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(threads), case], cwd=ROOT, env=env, capture_output=True, text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


@pytest.mark.parametrize("case", ["current_field", "port_impedance"])
def test_a_contract_gives_the_same_bits_in_every_process(case: str) -> None:
    runs = {(seed, threads): _digest(case, seed, threads) for seed, threads in ((0, 1), (1, 2), (2, 4))}
    assert len(set(runs.values())) == 1, runs
