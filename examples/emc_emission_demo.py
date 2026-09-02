"""Radiated-emission demo: a DC current pattern read as a phasor at each frequency.

A two-layer board carries 1 A out along a top-layer trace, down through a via
field at the far end, and back along the bottom layer to a pad under the
source pad, so the current closes on the board in a loop the layer separation
sets. The DC conduction solve gives the current density; every element becomes
a Hertzian dipole. The demo then prints, per frequency:

* the magnetic near field 5 mm above the board, as a scan probe would read;
* the far-zone maximum at 10 m and 3 m in dBµV/m against CISPR 32 Class B and
  FCC Part 15 Class B;
* the electric and magnetic dipole moments and how much of the radiated power
  those two lowest moments explain.

The quasi-static assumption is stated in the output: the DC pattern is used
unchanged at each frequency, which holds while the board is small compared
with a wavelength. The electric dipole moment |P| is the check that the loop
really closes: source and sink pads at different places leave a net current
moment I·L whose radiation belongs to the unmodelled external return, not to
the board. By default the demo closes the terminals with a straight element
standing for the component between the pads (``--open-terminals`` omits it).

    python examples/emc_emission_demo.py [--backend cpu|cuda|auto] [--ground-plane]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    PCBConductionProblem,
    ViaConnection,
    solve_pcb_dc,
)
from emc.tiled_dipole_superposition import (
    CISPR32_CLASS_B,
    FCC_PART15_CLASS_B,
    db_microvolt_per_m,
    dipole_moments,
    dipoles_from_pcb_dc,
    emission_margin,
    evaluate_fields,
    far_field_pattern,
    scan_plane,
)


def build_board(rows: int, cols: int, pitch_m: float) -> PCBConductionProblem:
    active = np.zeros((2, rows, cols), dtype=bool)
    trace = slice(rows // 2 - 2, rows // 2 + 2)
    active[0, trace, :] = True          # return on the bottom layer
    active[1, trace, :] = True          # outgoing trace on the top layer
    mesh = LayeredPCBMesh(
        element_active=active,
        layer_thickness_m=(35.0e-6, 35.0e-6),
        pitch_x_m=pitch_m,
        pitch_y_m=pitch_m,
    )
    node_rows = tuple(range(rows // 2 - 2, rows // 2 + 3))
    vias = tuple(
        ViaConnection((0, r, c), (1, r, c), 1.5e-3)
        for r in node_rows[1:-1]
        for c in (cols - 2, cols)
    )
    return PCBConductionProblem(
        mesh=mesh,
        terminals=(
            CurrentTerminal(tuple((1, r, 0) for r in node_rows), 1.0, "source pad, top"),
            CurrentTerminal(tuple((0, r, 0) for r in node_rows), -1.0, "return pad, bottom"),
        ),
        reference_node=(0, node_rows[0], 0),
        vias=vias,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--pitch-mm", type=float, default=0.5)
    parser.add_argument("--separation-mm", type=float, default=1.6)
    parser.add_argument("--frequencies-mhz", type=float, nargs="+", default=(30.0, 100.0, 300.0))
    parser.add_argument("--ground-plane", action="store_true",
                        help="add images in an infinite PEC 10 mm below the bottom layer")
    parser.add_argument("--open-terminals", action="store_true",
                        help="omit the element that closes the terminal current through the component")
    args = parser.parse_args()

    problem = build_board(args.rows, args.cols, args.pitch_mm * 1e-3)
    solution = solve_pcb_dc(problem, backend=args.backend)
    heights = (0.0, args.separation_mm * 1e-3)
    dipoles = dipoles_from_pcb_dc(
        problem, solution, heights, close_terminals=not args.open_terminals
    )
    if args.ground_plane:
        dipoles = dipoles.with_ground_plane_images(-10.0e-3)
    print(f"electrical: converged={solution.solve.converged}, "
          f"{dipoles.count} current elements, loss {solution.joule_loss_w * 1e3:.2f} mW")
    print("assumption: the DC current pattern is used unchanged as a phasor at every frequency")

    board_x = args.cols * args.pitch_mm * 1e-3
    board_y = args.rows * args.pitch_mm * 1e-3
    probe = scan_plane(
        np.linspace(-0.01, board_x + 0.01, 60),
        np.linspace(-0.01, board_y + 0.01, 30),
        heights[1] + 5.0e-3,
    )

    for frequency_mhz in args.frequencies_mhz:
        frequency = frequency_mhz * 1e6
        started = time.perf_counter()
        near = evaluate_fields(dipoles, probe, frequency, backend=args.backend)
        pattern = far_field_pattern(dipoles, frequency, distance_m=10.0, backend=args.backend)
        elapsed = time.perf_counter() - started
        moments = dipole_moments(dipoles, frequency)
        margin_cispr = emission_margin(pattern.max_polarised_field_v_per_m, frequency, CISPR32_CLASS_B, distance_m=10.0)
        at_3m = pattern.scaled_to_distance(3.0)
        margin_fcc = emission_margin(at_3m.max_polarised_field_v_per_m, frequency, FCC_PART15_CLASS_B, distance_m=3.0)
        wavelength = 299_792_458.0 / frequency

        print(f"\n{frequency_mhz:8.1f} MHz  (board / wavelength = {max(board_x, board_y) / wavelength:.3f}, {elapsed:.2f} s)")
        print(f"  near field 5 mm above top copper: max |H| {np.max(near.magnetic_magnitude_a_per_m) * 1e3:.3f} mA/m, "
              f"max |E| {np.max(near.electric_magnitude_v_per_m):.3f} V/m")
        print(f"  far field: {pattern.radiated_power_w * 1e9:.3f} nW radiated, "
              f"max {pattern.max_field_dbuv_per_m:.1f} dBµV/m at 10 m, "
              f"{db_microvolt_per_m(at_3m.max_field_v_per_m):.1f} dBµV/m at 3 m, "
              f"directivity {pattern.directivity_dbi:.2f} dBi")
        for margin in (margin_cispr, margin_fcc):
            verdict = "pass" if margin.compliant else "FAIL"
            print(f"  {margin.limit_name:22s} @ {margin.distance_m:4.0f} m: predicted {margin.predicted_dbuv_per_m:6.1f}, "
                  f"limit {margin.limit_dbuv_per_m:5.1f} dBµV/m, margin {margin.margin_db:+6.1f} dB  {verdict}")
        explained = moments.total_radiated_power_w / pattern.radiated_power_w if pattern.radiated_power_w else float("nan")
        print(f"  moments: |P| {np.linalg.norm(moments.electric_a_m) * 1e3:.3f} mA·m, "
              f"|M| {np.linalg.norm(moments.magnetic_a_m2) * 1e6:.3f} µA·m²; "
              f"dipole terms explain {explained * 100:.1f}% of the radiated power")


if __name__ == "__main__":
    main()
