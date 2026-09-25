# Coupled analysis scenarios

## Scope

`src/multiphysics/staggered_coupling` runs the electrical, thermal, and EMC
solvers of this repository as one analysis. It owns no field solve; it carries
fields between the solvers and iterates where a field feeds back.

| Scenario | Solves | Coupling |
|---|---|---|
| `ElectricalScenario` | DC conduction | — |
| `ThermalScenario` | steady heat conduction | — |
| `ThermalTransientScenario` | backward-Euler heat conduction over a `TimeSchedule` | — |
| `ElectroThermalScenario` | both, iterated | Joule heat → `T` → `σ(T)`, via `R(T)` |
| `ElectroEmissionScenario` | DC conduction, dipole superposition | `J` → near and far field, limit margin |
| `ElectroThermalEmissionScenario` | all three | `σ(T)`-converged `J` → field; cold `J` → field for comparison |
| `SheetPeecEmissionScenario` | sheet PEEC at each frequency, dipole superposition | `J(f)` → field |
| `BoardEnclosureThermalScenario` | board conduction and one conduction solve per body, iterated | contact heat → body `T` → board Robin ambient |
| `ElectroThermalEnclosureScenario` | DC conduction, then board and bodies, iterated | Joule heat → board/body `T` (contact exchange, radiation) → `σ(T)`, via `R(T)` |

`run_scenario` dispatches on the dataclass type; `run_scenarios` runs a list.
The `backend` argument reaches every solver (`"cpu"`, `"cuda"`, `"auto"`).

## Electro-thermal coupling

Copper resistivity rises with temperature, `σ(T) = σ_ref / (1 + α (T − T_ref))`
with `α = 3.93e-3 /K` by default. Under constant current the loss therefore
rises with temperature, which raises the temperature: a positive feedback with
gain `s = α I² R₀ R_th`. For `s < 1` a steady state exists; for `s ≥ 1` the
board runs away and no iteration converges.

The iteration is partitioned (staggered):

1. electrical solve with the current per-element conductivity and via
   resistances, warm-started from the previous potential;
2. thermal solve with the element Joule heat and via heat, warm-started from
   the previous temperature;
3. relaxed update `T ← T + ω ΔT`; Aitken's Δ² estimate rescales `ω` from two
   successive increments, clipped to `[0.05, max_relaxation]`;
4. new conductivity from the mean corner temperature of each copper element,
   new via resistance from the mean temperature of the via endpoints;
5. stop when the relaxed temperature change and the relative loss change are
   below tolerance and both inner solves converged.

Aitken over-relaxes a slowly contracting iteration: for a linear feedback with
gain `s` the plain iteration converges as `sⁿ`, Aitken jumps to the fixed point
in one step. On the test loop (2 A, `s ≈ 0.1`) the accelerated run converges in
5 iterations against 9 for the plain one; on a near-runaway case (`s ≈ 0.84`)
the plain iteration is still 35 K from the fixed point after 20 steps while the
accelerated one is within 0.2 K after 13. Over-relaxation is refused when it
would push a node below the temperature where the linear model breaks down.

The result carries both solutions, the conductivity and via resistances the
final electrical solve used, the element temperatures of each copper layer,
the cold loss, and a per-iteration history with the relaxation factor.

### Verification

- A board whose thermal nodes are all fixed at `T_hot` is isothermal, so the
  converged loss has to equal `I² R(T_hot)` exactly: the test checks
  `loss / loss_cold = 1 + α (T_hot − T_ref)` to `1e-9`, including the vias.
- `α = 0` completes in one pass with the reference conductivity untouched.
- The converged conductivity equals the law evaluated at the converged
  temperature field to `1e-6`; the thermal heat input equals the electrical
  loss to `1e-12`; the accelerated and plain iterations agree to `1e-6`.
- An iteration cap reports `converged=False` and keeps the last iterate.

## Board and separately meshed bodies

A heat sink, enclosure or component body keeps its own `LayeredThermalMesh`
(a `VoxelThermalMesh` from CAD) with its own pitch and origin. A `ContactMap`
from `thermal.matrix_free_mpir_fem` names which board face touches which body
face, the overlap area, and the joint conductance `G` (from a TIM's `k / t`
or a contact coefficient); `planar_contact_map` builds it for a body standing
on the board's top or hanging below its bottom, intersecting the two cell
footprints in a common frame. The iteration is Robin on the board, Neumann on
the body:

1. the board is solved with an extra `ConvectionBoundary` on the contact
   faces, film coefficient `G / A` and a per-face ambient equal to the body's
   current contact temperature (several body faces on one board face add
   their conductances and average their temperatures with `G` weights);
2. the heat that crossed each pair, `G (T_board − T_body)`, is lumped onto the
   body's face nodes as `nodal_heat_w` and the body is solved with its own
   boundaries;
3. the contact temperatures are relaxed (Aitken by default) and step 1
   repeats until they move less than `temperature_tolerance_k` and the
   interface heat is steady.

A hard contact on a stiff body has a fixed-point gain near one, so the
unrelaxed exchange oscillates with growing amplitude; the run stops when the
contact temperature change exceeds `divergence_temperature_k` and reports
`converged = False` rather than overflowing the solves.

### Verification

`tests/test_board_enclosure_coupling.py` solves a three-layer board with a
6 × 6 × 4 mm aluminium block on a 1 µm, 0.01 W/mK interface (`G / A = 1e4
W/m²K`) both as one masked layered mesh and as two meshes through the contact
map. With a 97 K rise the partitioned board agrees with the monolithic mesh to
0.036 K and the block to 0.011 K (both below `2e-3` of the rise); the residual
is the monolithic mesh's cooled interface sliver and its in-plane conduction,
which the contact model omits. The interface heat equals the board's
convective heat on the contact boundary to `1e-9` and the block's rejected heat
to `1e-6`. Aitken converges in 6 interface iterations, a fixed relaxation of
0.2 in 20, and the unrelaxed exchange is stopped as diverging after 9.

The adopted measurement is `BOARD_ENCLOSURE_ACCEPTANCE_RESULTS.json`
(`experiments/board_enclosure_acceptance.py`; Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz):

| Run | Converged | Interface iterations | Wall time (ms) | Board diff (K) | Sink diff (K) |
|---|---|---|---|---|---|
| monolithic masked mesh (1989 nodes) | True | — | 434 | — | — |
| partitioned, Aitken | True | 6 | 1244 | 3.6e-02 | 1.1e-02 |
| partitioned, fixed ω = 0.2 | True | 20 | 3723 | 3.6e-02 | 1.1e-02 |
| partitioned, unrelaxed | False | 9 (stopped) | 2209 | — | — |

On this small case the monolithic mesh is about 3× faster than the
partitioned iteration; the partitioned form is adopted for what the single
grid cannot express (a body at another pitch or origin, several bodies, a
body far larger than the board), not for speed.

## σ(T) around the board and its bodies

`ElectroThermalEnclosureScenario(electro_thermal, bodies)` wraps the interface
iteration above in the copper resistivity loop: an electrical solve gives the
Joule heat, `run_board_enclosure_thermal` solves the board *and* the bodies to
a common contact temperature, the copper conductivity and via resistances
follow the board temperature (the same `TemperatureFixedPoint` with Aitken
relaxation as `ElectroThermalScenario`), repeat. Each interface iteration is
warm-started from the previous outer step (`initial=`), so after the first
pass it costs two to four board/body solves. Radiating faces on the board or
on a body (`radiation=` on the scenario and on the body problems) are solved
inside every thermal solve by the Newton loop of `solve_thermal_conduction`,
so one outer loop closes `σ(T)`, the contact exchange and radiation together.

### Verification

`tests/test_electro_thermal_enclosure.py` and
`experiments/electrothermal_enclosure_acceptance.py` drive a two-layer 6 A
loop (0.5 mm grid, 35 µm copper) with a 6 × 2 × 3 mm aluminium block on a
1 µm, 0.01 W/mK interface, once as one masked electro-thermal mesh
(`ElectroThermalScenario`, the reference) and once through the contact map,
with and without `ε = 0.9` radiation from the board top, the board edges and
the block. Adopted measurement `ELECTROTHERMAL_ENCLOSURE_RESULTS.json`
(Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz):

| Radiation | Rise (K) | Outer iterations ref / part | Interface iterations per outer step | Loss ratio ref / part | Board diff (K) | Sink diff (K) | Wall ref / part (ms) |
|---|---|---|---|---|---|---|---|
| no | 199.0 | 6 / 6 | 6, 6, 4, 4, 2, 2 | 1.7858 / 1.7859 | 4.1e-02 | 3.6e-02 | 1604 / 4370 |
| yes | 104.7 | 7 / 7 | 9, 8, 6, 5, 4, 3, 2 | 1.4138 / 1.4139 | 1.9e-02 | 1.5e-02 | 3917 / 13465 |

Decision: adopted; every difference is below `3e-3` of the rise and the loss
ratios agree to `2e-3`. Radiation lowers the rise of this small, mostly
radiating assembly from 199 K to 105 K and the loss increase from 79 % to 41 %.
The partitioned route is 2.7 to 3.4 times slower than the single mesh here, as
for the thermal-only case; it is adopted for bodies the single grid cannot
express, not for speed. The residual difference is the monolithic mesh's cooled
interface sliver, which the contact model omits.

## Emission scenarios

`ElectroEmissionScenario` uses one DC solve as a phasor at every frequency of
the sweep (quasi-static), with the terminals closed through the component by
default. `ElectroThermalEmissionScenario` runs the electro-thermal iteration
first, evaluates the emission of the `σ(T)`-converged current, and also of the
cold current; `heating_shift_db` is the difference. For a loop carrying a fixed
current the heating redistributes the current only slightly, and the test
requires the shift to stay below 0.5 dB. `SheetPeecEmissionScenario` solves the
sheet-PEEC case at each frequency, so the current distribution itself is
frequency-resolved.

Every emission result holds, per frequency, the far-field pattern at the
limit's distance, the margin to the limit line, the dipole moments, and, if a
`ScanPlane` is given, the near-field samples. `worst_margin_db` and `compliant`
summarise the sweep. The test checks that loop radiation grows 40 dB per
decade in field and that the FCC Class B limits at 3 m are read correctly.

## Solver changes made for the coupling

- `solve_pcb_dc` accepts `initial_potential_v` to warm-start the outer MPIR
  refinement.
- `solve_thermal_conduction` defaults to 16 outer iterations, and, when the
  requested residual is not reached, re-references the unknown to the mean
  free-node temperature and continues from the current iterate. The FP64
  residual of `K θ` floors at `eps · ‖K‖ · ‖θ‖`; a copper plate 120 K above
  ambient, almost uniformly, hits that floor at a relative residual near
  `3e-10` before the requested `1e-10`. After the shift the remaining unknown
  is the small in-plane variation, whose residual is accurate, and the solve
  finishes in a few more outer steps.

## Fused C++ host paths in the coupled solve (measured on `exp/cpp-multiphysics-dc-emc`)

`run_electro_thermal(native=, native_threads=)` hands one selection to both
solvers, `solve_pcb_dc` and `solve_thermal_conduction`; `None` (the default)
leaves each to its own flag, `PCB_NATIVE_Q1` and `PCB_NATIVE_THERMAL`. The
coupling itself stays NumPy: the Joule-heat mapping, the ρ(T) update of the
element conductivities and the Aitken fixed point are O(elements) array
passes and do not show in a profile.

### Where a coupled solve spends its time

One case of the Kicad_PowerOpt thermal coupling (power_module, full-domain
PGND copper on both layers, 201,248 electrical nodes, 0.1 mm grid, four
staggered iterations, one thread) took 126.7 s on the NumPy paths, of which
103.3 s (82 %) was the thermal `apply_low` (`_stiffness_action`), 14.7 s the
two two-level preconditioner constructions per iteration (27 FP64 probes and
the dense coarse inverse, electrical and thermal), 8.5 s the electrical DC
solves and 4.4 s the dense coarse solves. The thermal action already had its
opt-in C++ path; the DC operator did not (see `MATRIX_FREE_MPIR_FEM.md`).

### pcb-analysis benchmark

`experiments/electrothermal_native_benchmark.py --sizes 100,200,320
--threads 1,4,16 --repeats 3` on the Xeon Platinum 8581C (GCC 14.2.1, NumPy
2.3.5, `OPENBLAS_NUM_THREADS=1`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`;
`ELECTROTHERMAL_NATIVE_XEON_8581C_RESULTS.json`). The board of the DC
benchmark on a three-slab thermal stack (35 µm copper, 1.5 mm laminate,
35 µm copper), 10 W/m²/K on both faces, ρ(T) copper, 10 A. One whole
`run_electro_thermal`, every solve and every operator construction included,
both solvers portable against both native:

| Electrical / thermal nodes | Portable | Native 1 / 4 / 16 threads | Coupling iterations | Temperature difference (of the rise) |
|---:|---:|---:|---:|---:|
| 20,402 / 40,804 | 12.3 s | 7.7 (1.6×) / 4.9 (2.5×) / 4.2 s (2.9×) | 4 | 2.5e-11 |
| 80,802 / 161,604 | 42.3 s | 11.9 (3.6×) / 7.1 (6.0×) / 5.7 s (7.5×) | 4 | 1.1e-10 |
| 206,082 / 412,164 | 118.0 s | 19.9 (5.9×) / 10.6 (11.2×) / 8.0 s (14.7×) | 4 | 1.1e-10 |

The Joule losses agree to 8e-13 and the temperatures to 1.1e-10 of the rise.
At sixteen threads the remaining 4 to 8 s is dominated by the preconditioner
constructions (eight per coupled solve, each with a dense inverse of a 1,352
to 1,800 square coarse matrix) and by the FP64 residual work the outer MPIR
loops do in NumPy. The benchmark's own criterion (at least 2× on every case
at one thread) misses on the smallest case (1.6×) for the same reason; from
four threads on every case gains at least 2.5×.

### Kicad_PowerOpt system benchmark

The consumer of this path is Kicad_PowerOpt's thermal coupling
(`plane_opt.physics.electro_thermal`, `thermal_coupling.enabled`). The whole
adopted pipeline was run twice on the same host and the same inputs
(`KICAD_POWEROPT_SYSTEM_BENCHMARK_XEON_8581C_RESULTS.json`): Kicad_PowerOpt
`origin/main` `d25c11e` (pcb-analysis 0.9 names), board `power_module`, the
board's settings plus `thermal_coupling.enabled`, `geometry_interface.
authoritative = "grid"` (the STEP default needs `kicad-cli`, absent here),
`sheet_peec` current field, one bootstrap order, 03 skipped, physics and
contour workers in parallel, `OPENBLAS_NUM_THREADS=1`.

| pcb-analysis | Native flags | Wall | User CPU | Thermal coupling, 9 cases standalone |
|---|---|---:|---:|---:|
| `main` `00e4491` (0.9.0), no extension built | none | 1,895 s | 4,275 s | 177.1 s |
| `exp/cpp-multiphysics-dc-emc` `836b144`, all extensions built | `PCB_NATIVE_Q1=1 PCB_NATIVE_THERMAL=1 PCB_NATIVE_EMC=1 PCB_NATIVE_THREADS=3` | 1,551 s | 3,132 s | 21.7 s |
| same, thermal kernel only | `PCB_NATIVE_THERMAL=1 PCB_NATIVE_THREADS=3` | — | — | 27.2 s |

The 9-case column is `evaluate_thermal_coupling` on the full-domain copper of
every role with the run's own thread pool (nine scenario threads, three
OpenMP threads each on the 32-core host); the pipeline evaluates it more than
once per candidate, which accounts for the 344 s the run saved. The two runs
reach the same candidate: the same masks and areas, the same 9-case
authoritative ratios (maximum 4.2668, voltage 3.4186), the same coupling
iteration counts, and a hottest rise of 0.67018 K in both to 1e-10. Kicad_
PowerOpt does not call the emc package, so `PCB_NATIVE_EMC` had no effect
there. The run before the FMA fix of the DC kernel took 1,546 s with 10 to
25 % more electrical inner iterations; it is kept out of the record because
its DC path was not the one shipped.

**Decision:** the coupled path is 2.5 to 14.7× faster from four threads on
and the consumer's thermal coupling 8.2× faster at its own thread layout,
with the same results. Both kernels stay opt-in through `native=True` or
their environment flags; adoption into `main` is a separate `feature/`
step.

## Limitations

- The thermal feedback is through copper resistivity only. Laminate
  conductivity, film coefficients, and component power are held fixed.
- The electrical solve is DC; the emission sweep therefore assumes the
  current pattern is frequency-independent. Use `SheetPeecEmissionScenario`
  where skin and proximity effects matter, at the cost of one sheet solve
  per frequency.
- Electromagnetic radiation does not feed back into the currents, and no
  susceptibility (immunity) scenario is provided.
- The scenarios share one in-plane element grid; the thermal stack may add
  laminate slabs but not refine the footprint.
- Thermal radiation is surface-to-ambient (see `THERMAL_MPIR_FEM.md`): a
  board inside a case radiates to a given case temperature, not to the case's
  computed field.
- The coupled scenarios are steady. `ThermalTransientScenario` marches a
  thermal problem alone; a transient electro-thermal or board/body march
  would time-step the same loops with `solve_thermal_transient` and is not
  implemented.
- The contact model carries no in-plane conduction inside the joint and no
  cooling of the joint's edge; a thick or conductive interface material
  belongs in the body mesh instead.

`examples/coupled_scenarios_demo.py` runs the electro-thermal iteration on a
two-layer loop and then evaluates its emission, printing the iteration history,
the loss increase, the margins, and the heating shift.
