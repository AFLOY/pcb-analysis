# Coupled analysis scenarios

## Scope

`src/multiphysics/staggered_coupling` runs the electrical, thermal, and EMC
solvers of this repository as one analysis. It owns no field solve; it carries
fields between the solvers and iterates where a field feeds back.

| Scenario | Solves | Coupling |
|---|---|---|
| `ElectricalScenario` | DC conduction | — |
| `ThermalScenario` | steady heat conduction | — |
| `ElectroThermalScenario` | both, iterated | Joule heat → `T` → `σ(T)`, via `R(T)` |
| `ElectroEmissionScenario` | DC conduction, dipole superposition | `J` → near and far field, limit margin |
| `ElectroThermalEmissionScenario` | all three | `σ(T)`-converged `J` → field; cold `J` → field for comparison |
| `SheetPeecEmissionScenario` | sheet PEEC at each frequency, dipole superposition | `J(f)` → field |

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

## Limitations

- The thermal feedback is through copper resistivity only. Laminate
  conductivity, film coefficients, and component power are held fixed.
- The electrical solve is DC; the emission sweep therefore assumes the
  current pattern is frequency-independent. Use `SheetPeecEmissionScenario`
  where skin and proximity effects matter, at the cost of one sheet solve
  per frequency.
- Radiation does not feed back into the currents, and no susceptibility
  (immunity) scenario is provided.
- The scenarios share one in-plane element grid; the thermal stack may add
  laminate slabs but not refine the footprint.

`examples/coupled_scenarios_demo.py` runs the electro-thermal iteration on a
two-layer loop and then evaluates its emission, printing the iteration history,
the loss increase, the margins, and the heating shift.
