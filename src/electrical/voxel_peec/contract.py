"""3D voxel PEEC for conductors too thick for a 2.5D sheet.

A busbar, a terminal block or heavy copper carries current through its
thickness, and above a few skin depths a sheet cannot represent it.  Such a
conductor is voxelised (by ``geometry.cad_import`` from CAD, or directly
as arrays) and solved with PyPEEC's voxel PEEC, which this module wraps in
the same way :mod:`electrical.sheet_peec.plane_opt_contract` wraps the sheet solve: the caller
gives arrays, this module owns the PyPEEC geometry/problem/tolerance
mappings, runs the solve on the CPU or through :class:`.cuda_pypeec.CudaPyPeecExecutor`,
and returns arrays.  PyPEEC keeps ownership of the assembly and the solve.

Voxel arrays follow the thermal convention ``(nz, ny, nx)`` with ``x`` the
last axis; PyPEEC's linear voxel index is ``x + nx (y + ny z)``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..sheet_peec.skin_filaments import COPPER_RESISTIVITY_OHM_M

_SOLVER_LOCK = threading.Lock()


@dataclass(frozen=True)
class VoxelTerminal:
    """A lumped port on a set of conductor voxels.

    One terminal per problem is the reference (``current_a is None``): it is
    held at zero volts through a voltage source.  Every other terminal
    injects ``current_a`` (a phasor for AC).
    """

    name: str
    voxels: np.ndarray
    current_a: complex | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("terminal name must be a non-empty alphanumeric/underscore string")
        mask = np.asarray(self.voxels, dtype=bool)
        if mask.ndim != 3 or not np.any(mask):
            raise ValueError(f"terminal {self.name!r} needs a non-empty (nz, ny, nx) voxel mask")
        object.__setattr__(self, "voxels", mask.copy())
        if self.current_a is not None:
            value = complex(self.current_a)
            if not np.isfinite(value.real) or not np.isfinite(value.imag):
                raise ValueError(f"terminal {self.name!r} current must be finite")
            object.__setattr__(self, "current_a", value)

    @property
    def is_reference(self) -> bool:
        return self.current_a is None


@dataclass(frozen=True)
class VoxelConductorProblem:
    """Conductor voxels, their resistivity, the ports and the frequency."""

    conductor: np.ndarray
    pitch_m: tuple[float, float, float]
    terminals: tuple[VoxelTerminal, ...]
    frequency_hz: float = 0.0
    resistivity_ohm_m: float | np.ndarray = COPPER_RESISTIVITY_OHM_M
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "conductor"
    material_id: np.ndarray | None = None
    material_resistivity_ohm_m: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        conductor = np.asarray(self.conductor, dtype=bool)
        if conductor.ndim != 3 or not np.any(conductor):
            raise ValueError("conductor must be a non-empty (nz, ny, nx) mask")
        pitch = tuple(float(v) for v in self.pitch_m)
        if len(pitch) != 3 or any(not np.isfinite(p) or p <= 0.0 for p in pitch):
            raise ValueError("pitch_m must be three positive lengths (hx, hy, hz)")
        origin = tuple(float(v) for v in self.origin_m)
        if len(origin) != 3 or any(not np.isfinite(o) for o in origin):
            raise ValueError("origin_m must be three finite coordinates")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz < 0.0:
            raise ValueError("frequency_hz must be non-negative")
        terminals = tuple(self.terminals)
        if len(terminals) < 2:
            raise ValueError("at least two terminals (one reference) are required")
        names = [t.name for t in terminals]
        if len(set(names)) != len(names):
            raise ValueError("terminal names must be unique")
        references = [t for t in terminals if t.is_reference]
        if len(references) != 1:
            raise ValueError("exactly one terminal must be the reference (current_a=None)")
        claimed = np.zeros(conductor.shape, dtype=bool)
        for terminal in terminals:
            if terminal.voxels.shape != conductor.shape:
                raise ValueError(f"terminal {terminal.name!r} mask must match the conductor shape")
            if not np.all(conductor[terminal.voxels]):
                raise ValueError(f"terminal {terminal.name!r} lies partly outside the conductor")
            if np.any(claimed & terminal.voxels):
                raise ValueError(f"terminal {terminal.name!r} overlaps another terminal")
            claimed |= terminal.voxels
        if self.material_id is None:
            resistivity = np.asarray(self.resistivity_ohm_m, dtype=np.float64)
            if resistivity.ndim != 0:
                raise ValueError("resistivity_ohm_m must be a scalar unless material_id is given")
            if not np.isfinite(resistivity) or resistivity <= 0.0:
                raise ValueError("resistivity must be finite and positive")
            material_id = np.where(conductor, 1, 0).astype(np.int64)
            table: dict[int, float] = {1: float(resistivity)}
        else:
            material_id = np.asarray(self.material_id, dtype=np.int64)
            if material_id.shape != conductor.shape:
                raise ValueError("material_id must match the conductor shape")
            table = {int(k): float(v) for k, v in self.material_resistivity_ohm_m.items()}
            used = set(np.unique(material_id[conductor]).tolist())
            if 0 in used or used - set(table):
                raise ValueError("every conductor voxel needs a material id with a resistivity")
            if any(not np.isfinite(v) or v <= 0.0 for v in table.values()):
                raise ValueError("material resistivities must be finite and positive")
            material_id = np.where(conductor, material_id, 0)
        object.__setattr__(self, "conductor", conductor.copy())
        object.__setattr__(self, "pitch_m", pitch)
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "terminals", terminals)
        object.__setattr__(self, "material_id", material_id)
        object.__setattr__(self, "material_resistivity_ohm_m", table)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(a) for a in self.conductor.shape)  # type: ignore[return-value]

    @property
    def reference(self) -> VoxelTerminal:
        return next(t for t in self.terminals if t.is_reference)

    @property
    def voxel_volume_m3(self) -> float:
        hx, hy, hz = self.pitch_m
        return hx * hy * hz

    def resistivity_field(self) -> np.ndarray:
        table = np.zeros(max(self.material_resistivity_ohm_m) + 1)
        for key, value in self.material_resistivity_ohm_m.items():
            table[key] = value
        return np.where(self.conductor, table[self.material_id], 0.0)


def linear_indices(mask: np.ndarray) -> list[int]:
    """PyPEEC voxel indices ``x + nx (y + ny z)`` of a ``(nz, ny, nx)`` mask."""

    nz, ny, nx = mask.shape
    z, y, x = np.nonzero(mask)
    return sorted((x + nx * (y + ny * z)).tolist())


def default_tolerance(settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """PyPEEC 5.8 solver settings: SciPy FFT dense operator, GMRES, direct coupling."""

    settings = dict(settings or {})
    iterative = {
        "solver": str(settings.get("iterative_solver", "gmres")),
        "rel_tol": float(settings.get("relative_tolerance", 1e-6)),
        "abs_tol": float(settings.get("absolute_tolerance", 1e-12)),
        "n_inner": int(settings.get("iteration_inner", 30)),
        "n_outer": int(settings.get("iteration_outer", 30)),
    }
    return {
        "parallel_sweep": {"n_jobs": 0, "n_threads": None},
        "integral_simplify": float(settings.get("integral_simplify", 20.0)),
        "biot_savart": "face",
        "dense_options": {
            "method": "fft",
            "split": bool(settings.get("split_fft", True)),
            "fft_options": {
                "library": str(settings.get("fft_library", "SciPy")),
                "scipy_worker": int(settings.get("scipy_workers", -1)),
                "fftw_thread": 0,
                "fftw_cache": False,
                "fftw_timeout": 100.0,
                "fftw_byte_align": 16,
            },
        },
        "factorization_options": {
            "schur": True,
            "library": str(settings.get("factorization_library", "SuperLU")),
            "pyamg_options": {"tol": 1e-6, "solver": "root", "krylov": None},
            "pardiso_options": {"thread_pardiso": 0, "thread_mkl": 0},
        },
        "solver_options": {
            "coupling": "direct",
            "status_options": {"ignore_status": False, "ignore_res": False, "rel_tol": float(settings.get("status_relative_tolerance", 1e-3)), "abs_tol": 1e-9},
            "power_options": {"stop": True, "n_min": 4, "n_cmp": 3, "rel_tol": 1e-4, "abs_tol": 1e-10},
            "direct_options": dict(iterative),
            "segregated_options": {
                "rel_tol": iterative["rel_tol"], "abs_tol": iterative["abs_tol"],
                "relax_electric": 1.0, "relax_magnetic": 1.0, "n_min": 2, "n_max": 20,
                "iter_electric_options": dict(iterative), "iter_magnetic_options": dict(iterative),
            },
        },
        "condition_options": {
            "check": bool(settings.get("check_condition", False)),
            "tolerance_electric": 1e15, "tolerance_magnetic": 1e15,
            "norm_options": {"t_accuracy": 2, "n_iter_max": 25},
        },
    }


def build_pypeec_inputs(
    problem: VoxelConductorProblem, *, settings: Mapping[str, Any] | None = None, dc_initialization: bool = True
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """PyPEEC ``geometry``, ``problem`` and ``tolerance`` mappings for the conductor."""

    nz, ny, nx = problem.shape
    hx, hy, hz = problem.pitch_m
    ox, oy, oz = problem.origin_m
    claimed = np.zeros(problem.shape, dtype=bool)
    domain_index: dict[str, list[int]] = {}
    for terminal in problem.terminals:
        domain_index[f"terminal_{terminal.name}"] = linear_indices(terminal.voxels)
        claimed |= terminal.voxels
    material_domains: dict[int, list[str]] = {}
    for key in sorted(problem.material_resistivity_ohm_m):
        body = problem.conductor & ~claimed & (problem.material_id == key)
        if np.any(body):
            tag = f"body_{key}"
            domain_index[tag] = linear_indices(body)
            material_domains.setdefault(key, []).append(tag)
    # A terminal takes the material of its voxels; PyPEEC wants every
    # conductor domain in exactly one material.
    for terminal in problem.terminals:
        keys = np.unique(problem.material_id[terminal.voxels])
        if keys.size != 1:
            raise ValueError(f"terminal {terminal.name!r} spans more than one material")
        material_domains.setdefault(int(keys[0]), []).append(f"terminal_{terminal.name}")
    geometry = {
        "mesh_type": "voxel",
        "data_voxelize": {
            "param": {
                "n": [nx, ny, nz],
                "d": [hx, hy, hz],
                "c": [ox + 0.5 * nx * hx, oy + 0.5 * ny * hy, oz + 0.5 * nz * hz],
            },
            "domain_index": domain_index,
        },
        "data_point": {"check_cloud": False, "filter_cloud": True, "pts_cloud": []},
        "data_resampling": {"use_reduce": False, "use_resample": False, "resampling_factor": [1, 1, 1]},
        "data_conflict": {"resolve_rules": True, "resolve_random": False, "conflict_rules": []},
        "data_integrity": {"check_integrity": False, "domain_connected": {}, "domain_adjacent": {}},
    }
    material_def = {
        f"material_{key}": {"domain_list": domains, "material_type": "electric", "orientation_type": "isotropic", "var_type": "lumped"}
        for key, domains in material_domains.items()
    }
    material_val = {f"material_{key}": {"rho_re": problem.material_resistivity_ohm_m[key], "rho_im": 0.0} for key in material_domains}
    source_def: dict[str, Any] = {}
    source_val: dict[str, Any] = {}
    for terminal in problem.terminals:
        tag = f"terminal_{terminal.name}"
        if terminal.is_reference:
            source_def[tag] = {"domain_list": [tag], "source_type": "voltage", "var_type": "lumped"}
            source_val[tag] = {"V_re": 0.0, "V_im": 0.0, "Z_re": 0.0, "Z_im": 0.0}
        else:
            current = terminal.current_a or 0j
            source_def[tag] = {"domain_list": [tag], "source_type": "current", "var_type": "lumped"}
            source_val[tag] = {"I_re": float(current.real), "I_im": float(current.imag), "Y_re": 0.0, "Y_im": 0.0}
    sweeps: dict[str, Any] = {}
    init = None
    if problem.frequency_hz > 0.0 and dc_initialization:
        sweeps["dc"] = {"init": None, "param": {"freq": 0.0, "material_val": material_val, "source_val": source_val}}
        init = "dc"
    sweeps["target"] = {"init": init, "param": {"freq": float(problem.frequency_hz), "material_val": material_val, "source_val": source_val}}
    pypeec_problem = {"material_def": material_def, "source_def": source_def, "sweep_solver": sweeps}
    return geometry, pypeec_problem, default_tolerance(settings)


@dataclass(frozen=True)
class VoxelTerminalResult:
    name: str
    current_a: complex
    voltage_v: complex  # relative to the reference terminal


@dataclass(frozen=True)
class VoxelPeecSolution:
    """Fields on the conductor voxels and the port quantities of one solve."""

    frequency_hz: float
    current_density_a_per_m2: np.ndarray  # (nz, ny, nx, 3) complex, zero outside the conductor
    potential_v: np.ndarray  # (nz, ny, nx) complex, nan outside
    loss_density_w_per_m3: np.ndarray  # (nz, ny, nx) real, time-averaged for AC
    terminals: tuple[VoxelTerminalResult, ...]
    joule_loss_w: float
    converged: bool
    iterations: int
    residual: float
    backend: str
    pypeec_version: str

    @property
    def impedance_ohm(self) -> dict[str, complex]:
        """``V / I`` of every driven terminal against the reference."""

        return {t.name: t.voltage_v / t.current_a for t in self.terminals if t.current_a != 0}

    def element_heat_w(self) -> np.ndarray:
        """Joule heat per voxel in W, ready for the thermal solve's ``element_heat_w``."""

        return self.loss_density_w_per_m3 * self._voxel_volume_m3

    _voxel_volume_m3: float = 0.0


def _reset_fft_backend() -> None:
    # PyPEEC 5.8 caches its FFT library choice in module globals after the
    # first solve; reset so this run honours its own tolerance.
    try:
        from pypeec.lib_matrix import multiply_fft

        multiply_fft.SET = False
    except (ImportError, AttributeError):  # pragma: no cover - PyPEEC internals
        pass


def solution_from_pypeec(problem: VoxelConductorProblem, solution: Mapping[str, Any], *, backend: str, pypeec_version: str) -> VoxelPeecSolution:
    if not bool(solution.get("status")):
        raise RuntimeError(f"{problem.name}: PyPEEC returned an invalid solution")
    sweep = solution["data_sweep"]["target"]
    idx = np.asarray(solution["data_init"]["idx_vc"], dtype=np.int64)
    fields = sweep["field_values"]
    current = np.asarray(fields["J_c"]["var"], dtype=np.complex128)
    potential = np.asarray(fields["V_c"]["var"], dtype=np.complex128)
    loss = np.asarray(fields["P_c"]["var"], dtype=np.float64).real
    if not (len(idx) == len(current) == len(potential) == len(loss)):
        raise RuntimeError(f"{problem.name}: inconsistent PyPEEC field array lengths")
    nz, ny, nx = problem.shape
    z, rem = np.divmod(idx, nx * ny)
    y, x = np.divmod(rem, nx)
    density = np.zeros((nz, ny, nx, 3), dtype=np.complex128)
    density[z, y, x] = current
    volt = np.full((nz, ny, nx), np.nan + 0j, dtype=np.complex128)
    volt[z, y, x] = potential
    loss_density = np.zeros((nz, ny, nx))
    # PyPEEC's P_c is the loss density of each voxel in W/m^3.
    loss_density[z, y, x] = loss
    sources = sweep["source_values"]
    reference_v = complex(sources[f"terminal_{problem.reference.name}"]["V"])
    terminals = tuple(
        VoxelTerminalResult(t.name, complex(sources[f"terminal_{t.name}"]["I"]), complex(sources[f"terminal_{t.name}"]["V"]) - reference_v)
        for t in problem.terminals
    )
    status = sweep.get("solver_status", {})
    return VoxelPeecSolution(
        frequency_hz=float(sweep["freq"]),
        current_density_a_per_m2=density,
        potential_v=volt,
        loss_density_w_per_m3=loss_density,
        terminals=terminals,
        joule_loss_w=float(np.sum(loss)) * problem.voxel_volume_m3,
        converged=bool(sweep.get("solution_ok")),
        iterations=int(status.get("n_iter", 0) or 0),
        residual=float(status.get("residuum_val", float("nan")) or float("nan")),
        backend=backend,
        pypeec_version=pypeec_version,
        _voxel_volume_m3=problem.voxel_volume_m3,
    )


def solve_voxel_peec(
    problem: VoxelConductorProblem,
    *,
    backend: str = "cpu",
    settings: Mapping[str, Any] | None = None,
    cuda_config: Mapping[str, Any] | None = None,
    quiet: bool = True,
) -> VoxelPeecSolution:
    """Solve the conductor with PyPEEC on the CPU or through the CUDA executor."""

    try:
        import pypeec
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyPEEC is required for the 3D voxel PEEC solve: pip install 'pcb-analysis[cuda]' or pypeec") from exc
    geometry, pypeec_problem, tolerance = build_pypeec_inputs(problem, settings=settings)
    if backend == "cuda":
        from .cuda_pypeec import CudaPyPeecExecutor

        with CudaPyPeecExecutor(dict(cuda_config or {})) as executor:
            result = executor.execute(geometry, pypeec_problem, tolerance)
        return solution_from_pypeec(problem, result.solution, backend="cuda", pypeec_version=str(pypeec.__version__))
    if backend != "cpu":
        raise ValueError("backend must be 'cpu' or 'cuda'")
    try:
        import scilogger
    except ModuleNotFoundError:  # pragma: no cover
        scilogger = None
    with _SOLVER_LOCK:
        if quiet and scilogger is not None:
            scilogger.disable()
        try:
            _reset_fft_backend()
            voxel = pypeec.run_mesher_data(geometry)
            solution = pypeec.run_solver_data(voxel, pypeec_problem, tolerance)
        finally:
            if quiet and scilogger is not None:
                scilogger.enable()
    return solution_from_pypeec(problem, solution, backend="cpu", pypeec_version=str(pypeec.__version__))


__all__ = [
    "VoxelConductorProblem",
    "VoxelPeecSolution",
    "VoxelTerminal",
    "VoxelTerminalResult",
    "build_pypeec_inputs",
    "default_tolerance",
    "linear_indices",
    "solution_from_pypeec",
    "solve_voxel_peec",
]
