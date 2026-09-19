"""3D voxel PEEC through PyPEEC: a copper busbar against its DC resistance."""

from __future__ import annotations

import numpy as np
import pytest

pypeec = pytest.importorskip("pypeec")

from electrical.voxel_peec import (  # noqa: E402
    VoxelConductorProblem,
    VoxelTerminal,
    build_pypeec_inputs,
    solve_voxel_peec,
)
from electrical.sheet_peec.skin_filaments import skin_depth_m  # noqa: E402

RHO = 1.68e-8
PITCH = 0.5e-3
NX, NY, NZ = 40, 8, 4  # 20 x 4 x 2 mm bar


def _busbar(frequency_hz: float = 0.0, current: complex = 10.0) -> VoxelConductorProblem:
    conductor = np.ones((NZ, NY, NX), dtype=bool)
    source = np.zeros_like(conductor)
    source[:, :, :2] = True
    reference = np.zeros_like(conductor)
    reference[:, :, -2:] = True
    return VoxelConductorProblem(
        conductor,
        (PITCH, PITCH, PITCH),
        (VoxelTerminal("src", source, current), VoxelTerminal("ref", reference)),
        frequency_hz=frequency_hz,
        resistivity_ohm_m=RHO,
        origin_m=(0.01, 0.02, 0.0),
        name="busbar",
    )


def test_pypeec_inputs_follow_the_voxel_contract() -> None:
    problem = _busbar()
    geometry, pypeec_problem, tolerance = build_pypeec_inputs(problem)
    param = geometry["data_voxelize"]["param"]
    assert param["n"] == [NX, NY, NZ] and param["d"] == [PITCH] * 3
    np.testing.assert_allclose(param["c"], [0.01 + 0.5 * NX * PITCH, 0.02 + 0.5 * NY * PITCH, 0.5 * NZ * PITCH])
    domains = geometry["data_voxelize"]["domain_index"]
    assert set(domains) == {"terminal_src", "terminal_ref", "body_1"}
    assert len(domains["terminal_src"]) == 2 * NY * NZ and domains["terminal_src"][:3] == [0, 1, NX]
    assert sum(len(v) for v in domains.values()) == NX * NY * NZ
    assert pypeec_problem["source_def"]["terminal_ref"]["source_type"] == "voltage"
    assert pypeec_problem["sweep_solver"]["target"]["param"]["freq"] == 0.0 and "dc" not in pypeec_problem["sweep_solver"]
    assert build_pypeec_inputs(_busbar(1.0e5))[1]["sweep_solver"]["target"]["init"] == "dc"
    assert tolerance["dense_options"]["fft_options"]["library"] == "SciPy"
    with pytest.raises(ValueError, match="exactly one terminal"):
        VoxelConductorProblem(problem.conductor, problem.pitch_m, (problem.terminals[0],) + (VoxelTerminal("b", problem.terminals[1].voxels, 1.0),))
    with pytest.raises(ValueError, match="overlaps"):
        VoxelConductorProblem(problem.conductor, problem.pitch_m, (problem.terminals[0], VoxelTerminal("dup", problem.terminals[0].voxels)))


def test_dc_busbar_matches_the_analytic_resistance() -> None:
    problem = _busbar()
    solution = solve_voxel_peec(problem)
    assert solution.converged and solution.backend == "cpu"
    src = next(t for t in solution.terminals if t.name == "src")
    assert src.current_a == pytest.approx(10.0, rel=1.0e-6)
    # The lumped terminals sit at the bar ends; the resistance seen between
    # their centres is that of the bar between the terminal midplanes.
    area = NY * PITCH * NZ * PITCH
    length = (NX - 2) * PITCH
    analytic = RHO * length / area
    assert solution.impedance_ohm["src"].real == pytest.approx(analytic, rel=2.0e-2)
    assert abs(solution.impedance_ohm["src"].imag) < 1.0e-9
    # Joule loss equals I^2 R and is distributed over the conductor voxels.
    assert solution.joule_loss_w == pytest.approx(100.0 * solution.impedance_ohm["src"].real, rel=1.0e-3)
    heat = solution.element_heat_w()
    assert heat.shape == (NZ, NY, NX) and heat.sum() == pytest.approx(solution.joule_loss_w)
    assert np.all(heat[problem.conductor] > 0.0)
    inner = solution.current_density_a_per_m2[:, :, 10:30, 0]
    np.testing.assert_allclose(np.abs(inner), 10.0 / area, rtol=2.0e-2)
    potential = solution.potential_v[0, 0]
    assert np.all(np.isfinite(potential)) and potential[0].real > potential[-1].real


def test_ac_busbar_shows_skin_effect_and_reactance() -> None:
    dc = solve_voxel_peec(_busbar())
    ac = solve_voxel_peec(_busbar(1.0e5))
    assert ac.converged and ac.frequency_hz == 1.0e5
    delta = skin_depth_m(1.0e5, RHO)
    assert delta < NZ * PITCH / 2  # the bar is several skin depths thick
    z_dc = dc.impedance_ohm["src"]
    z_ac = ac.impedance_ohm["src"]
    assert z_ac.real > 1.5 * z_dc.real
    assert z_ac.imag > 0.0
    # Time-averaged loss for a 10 A peak phasor: 0.5 |I|^2 R_ac.
    assert ac.joule_loss_w == pytest.approx(0.5 * 100.0 * z_ac.real, rel=2.0e-2)
    # Current crowds to the surface: edge voxels carry more than the core.
    magnitude = np.abs(ac.current_density_a_per_m2[:, :, NX // 2, 0])
    assert magnitude[0, 0] > 1.5 * magnitude[NZ // 2, NY // 2]


def test_cuda_backend_matches_the_cpu_solution() -> None:
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - no device
        pytest.skip(str(exc))
    problem = _busbar(1.0e4)
    cpu = solve_voxel_peec(problem)
    gpu = solve_voxel_peec(problem, backend="cuda")
    assert gpu.backend == "cuda" and gpu.converged
    assert gpu.impedance_ohm["src"] == pytest.approx(cpu.impedance_ohm["src"], rel=1.0e-4)
    assert gpu.joule_loss_w == pytest.approx(cpu.joule_loss_w, rel=1.0e-4)
