"""Electrothermal demo: DC copper loss feeding a steady thermal solve.

A two-layer board carries 8 A from a pad on the bottom layer to a pad on the
top layer through a via field. The electrical solve reports the Joule loss
of every copper element and via; the thermal solve spreads that heat through
the copper/FR-4 stack and cools both faces by natural convection.

Run from the repository root:

    python examples/electrothermal_demo.py [--backend cpu|cuda|auto]
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
from thermal.matrix_free_mpir_fem import (
    COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K,
    ConvectionBoundary,
    LayeredThermalMesh,
    ThermalConductionProblem,
    element_joule_heat_w,
    solve_thermal_conduction,
    via_joule_heat_sources,
)


def build_electrical(rows: int, cols: int, pitch_m: float) -> PCBConductionProblem:
    active = np.zeros((2, rows, cols), dtype=bool)
    trace = slice(rows // 2 - 4, rows // 2 + 4)
    active[0, trace, : cols // 2 + 6] = True          # bottom trace to the via field
    active[1, trace, cols // 2 - 6 :] = True          # top trace from the via field
    mesh = LayeredPCBMesh(
        element_active=active,
        layer_thickness_m=(35.0e-6, 35.0e-6),
        pitch_x_m=pitch_m,
        pitch_y_m=pitch_m,
    )
    node_rows = tuple(range(rows // 2 - 4, rows // 2 + 5))
    vias = tuple(
        ViaConnection((0, r, c), (1, r, c), 1.5e-3)
        for r in node_rows[1:-1:2]
        for c in range(cols // 2 - 3, cols // 2 + 4, 3)
    )
    return PCBConductionProblem(
        mesh=mesh,
        terminals=(
            CurrentTerminal(tuple((0, r, 0) for r in node_rows), 8.0, "bottom pad"),
            CurrentTerminal(tuple((1, r, cols) for r in node_rows), -8.0, "top pad"),
        ),
        reference_node=(1, node_rows[0], cols),
        vias=vias,
    )


def build_thermal(problem: PCBConductionProblem) -> tuple[LayeredThermalMesh, tuple[int, int]]:
    active = problem.mesh.element_active
    _, rows, cols = active.shape
    copper = COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K
    in_plane = np.full((3, rows, cols), FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K)
    through = np.full((3, rows, cols), FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K)
    for layer, slab in enumerate((0, 2)):
        in_plane[slab] = np.where(active[layer], copper, in_plane[slab])
        through[slab] = np.where(active[layer], copper, through[slab])
    # Plated vias are copper columns through the laminate slab.
    for via in problem.vias:
        _, r, c = via.lower
        in_plane[1, r - 1 : r + 1, c - 1 : c + 1] = copper
        through[1, r - 1 : r + 1, c - 1 : c + 1] = copper
    mesh = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 1.5e-3, 35.0e-6),
        pitch_x_m=problem.mesh.pitch_x_m,
        pitch_y_m=problem.mesh.pitch_y_m,
        conductivity_w_per_m_k=in_plane,
        through_plane_conductivity_w_per_m_k=through,
    )
    return mesh, (0, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument("--cols", type=int, default=120)
    parser.add_argument("--pitch-mm", type=float, default=0.5)
    parser.add_argument("--film", type=float, default=10.0, help="W/m²K on both faces")
    parser.add_argument("--ambient", type=float, default=298.15)
    args = parser.parse_args()

    electrical_problem = build_electrical(args.rows, args.cols, args.pitch_mm * 1e-3)
    started = time.perf_counter()
    electrical = solve_pcb_dc(electrical_problem, backend=args.backend)
    electrical_seconds = time.perf_counter() - started

    mesh, layer_slabs = build_thermal(electrical_problem)
    thermal_problem = ThermalConductionProblem(
        mesh,
        convection=(
            ConvectionBoundary("top", args.film, args.ambient),
            ConvectionBoundary("bottom", args.film, args.ambient),
        ),
        element_heat_w=element_joule_heat_w(electrical, mesh, layer_slabs),
        heat_sources=via_joule_heat_sources(
            electrical_problem, electrical, mesh, layer_slabs
        ),
    )
    started = time.perf_counter()
    thermal = solve_thermal_conduction(thermal_problem, backend=args.backend)
    thermal_seconds = time.perf_counter() - started

    print(f"electrical: converged={electrical.solve.converged} "
          f"outer={electrical.solve.outer_iterations} inner={electrical.solve.inner_iterations} "
          f"{electrical_seconds:.2f}s runtime={electrical.solve.low_runtime}")
    print(f"  joule loss          {electrical.joule_loss_w * 1e3:.2f} mW "
          f"(vias {np.sum(electrical.via_joule_loss_w) * 1e3:.2f} mW)")
    print(f"thermal:    converged={thermal.solve.converged} "
          f"outer={thermal.solve.outer_iterations} inner={thermal.solve.inner_iterations} "
          f"{thermal_seconds:.2f}s runtime={thermal.solve.low_runtime}")
    print(f"  nodes               {mesh.size}")
    print(f"  heat input          {thermal.total_heat_input_w * 1e3:.2f} mW")
    print(f"  convective removal  top {thermal.convective_heat_w[0] * 1e3:.2f} mW, "
          f"bottom {thermal.convective_heat_w[1] * 1e3:.2f} mW")
    print(f"  balance error       {thermal.heat_balance_error_w:.2e} W")
    print(f"  temperature rise    max {thermal.max_temperature_k - args.ambient:.2f} K, "
          f"min {thermal.min_temperature_k - args.ambient:.2f} K")
    hottest = np.unravel_index(np.argmax(thermal.temperature_k), mesh.node_shape)
    print(f"  hottest node        (face, row, col) = {tuple(int(i) for i in hottest)}")


if __name__ == "__main__":
    main()
