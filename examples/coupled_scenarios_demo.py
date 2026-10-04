"""Coupled scenarios on one board: electro-thermal iteration, then emission.

A two-layer loop (outgoing trace on top, return on the bottom layer, vias at
the far end) is held at a fixed source voltage. Heating raises the copper
resistance, so the current and the loss fall until the board settles; a
fixed current would instead raise the loss and can run away, which is why
current-driven coupling is deprecated. The demo runs, through `run_scenario`:

1. the electro-thermal iteration with temperature-dependent copper, printing
   each staggered step and the loss and current it converges to;
2. the emission of the ρ(T)-converged current at several frequencies against
   CISPR 32 Class B at 10 m, next to the emission of the cold current.

Without ``--voltage`` the source voltage is the one that drives
``--cold-current`` through the cold loop.

    python examples/coupled_scenarios_demo.py [--cold-current 3 | --voltage 0.02] [--backend cpu|cuda|auto]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from electrical.matrix_free_mpir_fem import (
    LayeredPCBMesh,
    PCBConductionProblem,
    ViaConnection,
    VoltageTerminal,
    solve_pcb_dc,
)
from multiphysics.staggered_coupling import (
    CouplingConfig,
    ElectroThermalEmissionScenario,
    ElectroThermalScenario,
    EmissionScenario,
    ScanPlane,
    run_electro_thermal,
    run_scenario,
)
from thermal.matrix_free_mpir_fem import (
    COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K,
    ConvectionBoundary,
    LayeredThermalMesh,
)


def loop_problem(rows: int, cols: int, pitch_m: float, voltage_v: float) -> PCBConductionProblem:
    active = np.zeros((2, rows, cols), dtype=bool)
    trace = slice(rows // 2 - 2, rows // 2 + 2)
    active[:, trace, :] = True
    mesh = LayeredPCBMesh(active, (35e-6, 35e-6), pitch_m, pitch_m)
    node_rows = tuple(range(rows // 2 - 2, rows // 2 + 3))
    vias = tuple(
        ViaConnection((0, r, c), (1, r, c), 1.5e-3) for r in node_rows[1:-1] for c in (cols - 2, cols)
    )
    return PCBConductionProblem(
        mesh,
        voltage_terminals=(
            VoltageTerminal(tuple((1, r, 0) for r in node_rows), voltage_v, "source pad, top"),
            VoltageTerminal(tuple((0, r, 0) for r in node_rows), 0.0, "return pad, bottom"),
        ),
        vias=vias,
    )


def build_scenario(
    rows: int, cols: int, pitch_m: float, voltage_v: float, film: float, ambient: float
) -> ElectroThermalScenario:
    electrical = loop_problem(rows, cols, pitch_m, voltage_v)
    active = electrical.mesh.element_active
    copper = COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K
    in_plane = np.full((3, rows, cols), FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K)
    through = np.full((3, rows, cols), FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K)
    for layer, slab in enumerate((0, 2)):
        in_plane[slab] = np.where(active[layer], copper, in_plane[slab])
        through[slab] = np.where(active[layer], copper, through[slab])
    thermal = LayeredThermalMesh(
        (35e-6, 1.5e-3, 35e-6), pitch_m, pitch_m, in_plane, through_plane_conductivity_w_per_m_k=through
    )
    return ElectroThermalScenario(
        electrical,
        thermal,
        (0, 2),
        convection=(ConvectionBoundary("top", film, ambient), ConvectionBoundary("bottom", film, ambient)),
        conductivity_reference_temperature_k=293.15,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default=None, choices=(None, "cpu", "cuda", "auto"))
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--cols", type=int, default=80)
    parser.add_argument("--pitch-mm", type=float, default=0.5)
    drive = parser.add_mutually_exclusive_group()
    drive.add_argument("--voltage", type=float, default=None, help="source voltage in V")
    drive.add_argument("--cold-current", type=float, default=3.0,
                       help="pick the voltage that drives this current (A) through cold copper")
    parser.add_argument("--film", type=float, default=10.0, help="W/m²K on both faces")
    parser.add_argument("--ambient", type=float, default=298.15)
    parser.add_argument("--no-aitken", action="store_true")
    parser.add_argument("--frequencies-mhz", type=float, nargs="+", default=(30.0, 100.0, 300.0))
    args = parser.parse_args()

    pitch_m = args.pitch_mm * 1e-3
    voltage = args.voltage
    if voltage is None:
        # The cold problem is linear in V: one probe solve at 1 V scales it.
        probe = solve_pcb_dc(loop_problem(args.rows, args.cols, pitch_m, 1.0))
        voltage = args.cold_current / float(probe.voltage_terminal_current_a[0])
    scenario = build_scenario(args.rows, args.cols, pitch_m, voltage, args.film, args.ambient)
    config = CouplingConfig(aitken=not args.no_aitken)

    started = time.perf_counter()
    coupled = run_electro_thermal(scenario, config=config, backend=args.backend)
    elapsed = time.perf_counter() - started
    print(f"electro-thermal: converged={coupled.converged} in {coupled.iterations} iterations, {elapsed:.1f} s")
    print("  it   loss [W]    Tmax [K]   dT [K]     omega   e-inner  t-inner")
    for step in coupled.history:
        print(f"  {step.iteration:2d}  {step.joule_loss_w:9.5f}  {step.max_temperature_k:9.3f}  "
              f"{step.temperature_change_k:9.2e}  {step.relaxation:5.2f}  {step.electrical_inner_iterations:7d}  "
              f"{step.thermal_inner_iterations:7d}")
    print(f"  loss cold {coupled.cold_joule_loss_w * 1e3:.2f} mW -> hot {coupled.electrical.joule_loss_w * 1e3:.2f} mW "
          f"(x{coupled.loss_increase_ratio:.3f}); copper {coupled.element_temperature_k[coupled.element_temperature_k > 0].max() - args.ambient:.1f} K above ambient")
    cold_current = coupled.cold_joule_loss_w / voltage
    hot_current = float(coupled.electrical.voltage_terminal_current_a[0])
    print(f"  source {voltage * 1e3:.3f} mV: current cold {cold_current:.3f} A -> hot {hot_current:.3f} A")
    print(f"  via resistance {coupled.via_resistance_ohm.min() * 1e3:.3f}-{coupled.via_resistance_ohm.max() * 1e3:.3f} mΩ (cold 1.500)")
    print(f"  thermal heat budget error {coupled.thermal.heat_balance_error_w:.1e} W")

    board_x, board_y = args.cols * args.pitch_mm * 1e-3, args.rows * args.pitch_mm * 1e-3
    emission = EmissionScenario(
        tuple(f * 1e6 for f in args.frequencies_mhz),
        scan=ScanPlane(np.linspace(0, board_x, 40), np.linspace(0, board_y, 12), 1.6e-3 + 5e-3),
    )
    started = time.perf_counter()
    chained = run_scenario(
        ElectroThermalEmissionScenario(scenario, (0.0, 1.6e-3), emission, coupling=config),
        backend=args.backend,
    )
    elapsed = time.perf_counter() - started
    print(f"\nemission (ρ(T)-converged current), {elapsed:.1f} s including the coupled solve:")
    print("   f [MHz]  cold dBµV/m   hot dBµV/m   limit   margin [dB]   shift [dB]   |H| 5 mm [A/m]")
    for point, cold, shift in zip(chained.emission.points, chained.cold_emission.points, chained.heating_shift_db):
        print(f"  {point.frequency_hz / 1e6:8.1f}  {cold.margin.predicted_dbuv_per_m:11.1f}  {point.margin.predicted_dbuv_per_m:11.1f}  "
              f"{point.margin.limit_dbuv_per_m:6.1f}  {point.margin.margin_db:+11.1f}  {shift:+10.3f}  {point.max_near_magnetic_a_per_m:14.3f}")
    print(f"  worst margin {chained.emission.worst_margin_db:+.1f} dB -> {'compliant' if chained.emission.compliant else 'over the limit'}")


if __name__ == "__main__":
    main()
