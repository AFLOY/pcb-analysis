# Radiated-emission evaluation by tiled dipole superposition

## Scope

`src/emc/tiled_dipole_superposition` turns a solved current distribution into
the quantities an EMC engineer reads:

- the magnetic and electric near field on a scan plane a few millimetres
  above the board, as a probe scan would see it;
- the far-zone pattern on a sphere at the test-site distance, the radiated
  power, directivity, and the maximum field per linear polarisation;
- the margin of that maximum to the radiated limits of CISPR 32 Class A/B
  and FCC Part 15 Class A/B, rescaled between 3 m and 10 m;
- the net electric and magnetic dipole moments as cheap, quadratic proxies
  for an optimizer.

It owns no field solve. The current comes from `solve_pcb_dc` (a DC pattern
read as a phasor at each frequency) or from the frequency-resolved sheet-PEEC
solve; adapters put both into one `CurrentDipoles` container of positions and
complex moments `I dl` in A·m.

Not implemented: dielectric substrate effects on radiation, finite ground
planes and their edge currents (only an infinite PEC plane by images),
cables, enclosures, common-mode currents driven by voltages the electrical
solves do not compute, conducted emissions, and susceptibility.

## Method

Every element carries moment `p = I dl`. With `n = r/|r|`, `k = ω/c`, and
the `e^{jωt}` convention, its fields are exact at every distance:

```text
H = (-jk / 4π) (n × p) (1 + 1/(jkr)) e^{-jkr} / r
E = (η / (4π jk)) { k² (n × p) × n / r + [3n(n·p) - p] (1/r³ + jk/r²) } e^{-jkr}
```

`evaluate_fields` sums these over all sources for each observation point,
in tiles of points against all sources, on NumPy or CuPy. At `k = 0` the
magnetic term is Biot-Savart and the electric field is refused: a current
distribution alone does not fix the static charge.

In the far zone the sum collapses to one phase-weighted vector per direction,

```text
F(n) = Σ_i [(n × p_i) × n] e^{jk n·r_i},   E = (jηk/4π) e^{-jkr} F(n) / r,
P_rad = (η k² / 32π²) ∮ |F|² dΩ,
```

which `far_field_pattern` evaluates as one matmul per direction tile and
integrates with Gauss-Legendre nodes in `cos θ` and uniform `φ`.

An infinite perfectly conducting plane is handled by `with_ground_plane_images`:
horizontal elements get inverted images, vertical ones upright images, and the
fields are valid on the source side only.

## Sources and the closure of the current

`dipoles_from_pcb_dc` places `J · (t px py)` at element centres on the layer
heights the caller supplies and `I · Δz` at via nodes. `dipoles_from_sheet_peec`
places `I · pitch` midway along each in-plane branch on the layer's `z_m` and
`I · Δz` on via branches; branch currents are positive along `+x`, `+y`, and
lower-to-upper.

A conduction solve injects current at some pads and removes it at others.
The board's current alone then has a net moment `Σ I_t c_t` that radiates
like an open wire and whose end charges dominate the electric near field.
`dipole_moments` reports that net electric moment `|P|`; for a board whose
terminal currents balance, a non-zero `|P|` means the path through the
component is missing. `close_terminals=True` (or `terminal_closure_dipoles`)
appends one straight element per terminal leg through a star point so the
whole distribution is divergence-free; its moment is exact, the geometry of
the component is not represented.

## Verification

The tests in `tests/test_emc_dipole_superposition.py` and
`tests/test_emc_sources.py` hold the implementation to closed forms:

| Check | Result |
|---|---|
| Single dipole radiated power `η k² p² / 12π` | relative error `1e-15` |
| Single dipole peak field `η k p / 4π r` and directivity 1.76 dBi | `1e-9` |
| Exact field at `kr ≈ 126` vs far-field magnitude and `E_θ / H_φ = η` | within `2/(kr)²` |
| Straight wire at `k = 0`: `H = I / 2πρ` and the right-hand rule | `1e-4` |
| 1 cm square loop vs magnetic dipole `η k⁴ m² / 12π`; `E_φ` in plane, `sin θ` on axis | `1e-3` |
| Tangential `E` on an imaged PEC plane | `1e-12` of the field scale |
| Tiling, `complex64`, and CUDA against the NumPy `complex128` reference | `1e-12`, `1e-4`, `1e-10` |
| DC strip: `Σ J V = I L`; closure drives `Σ p` to zero and the far field down by `> 30 dB` | exact / `1e-9` |
| Sheet PEEC: one via carries the whole current over the layer separation | `1e-6` |
| Limit tables against the published values | exact |

## What the demo shows

`examples/emc_emission_demo.py` solves a 50 mm two-layer loop (1 A out on
the top trace, back on the bottom layer, 1.6 mm apart) and evaluates it at
30, 100, and 300 MHz. With the terminals closed the net electric moment is
zero and the magnetic moment is `I L d = 79 µA·m²`; the pattern's integrated
power equals the magnetic-dipole formula to 0.3 %. The 1 A loop exceeds
CISPR 32 Class B by 9 dB at 30 MHz and by 42 dB at 300 MHz, the `k⁴` growth
of loop radiation. Without closure the same board reports 70 dB over the
limit: that is the open-wire moment of the unmodelled return, which is why
`|P|` is printed next to the margin.

## Accuracy limits to keep in mind

- **Quasi-static current.** A DC pattern used at `f` assumes the distribution
  does not change with frequency. The demo prints `board / λ`; above about
  0.1 the sheet-PEEC solve at that frequency is the right source.
- **Electric near field.** Each element carries end charges `±I/(jω)`.
  Where the element chain is not perfectly continuous (element centres,
  via nodes, the closure leg), the residual charges set the electric near
  field, so it is discretisation-sensitive at low frequency. The magnetic
  near field and the far field, which depend on the moments and not on the
  local charge, are the reliable outputs.
- **Limit comparison.** The tables are the published quasi-peak (and
  above 1 GHz average/peak) limits at their stated distances. Rescaling by
  `20 log10(d₁/d₂)` is the far-zone rule; at 30 MHz a 3 m site is not in
  the far zone of a 50 mm board, and a real measurement adds antenna
  factors, a height scan, and uncertainty. The margin is an estimate to
  rank layouts by, not a certification.
- **Environment.** Free space or one infinite PEC plane. A finite ground
  plane, a chassis, or cables change the result by tens of dB; those
  currents have to be part of the source distribution to be counted.

## Cost and acceleration

Near-field cost is `points × sources` pair evaluations, tiled; far-field
cost is `directions × sources` per frequency as a complex matmul. Both run
on CuPy with `backend="cuda"`, where the 300-source, 300-point regression
test agrees with NumPy to `1e-10`. The complex128 default is right for the
far field, which relies on cancellation between elements; `dtype=complex64`
is adequate for the magnetic near field on a GPU without fast FP64.

## Software boundary

| Module | Responsibility |
|---|---|
| `fields.py` | `CurrentDipoles`, exact dipole fields in tiles, ground-plane images, scan planes |
| `far_field.py` | sphere sampling, pattern, radiated power, directivity, dBµV/m |
| `limits.py` | CISPR 32 and FCC Part 15 tables, distance rescaling, margins |
| `moments.py` | electric and magnetic dipole moments and their radiated powers |
| `sources.py` | adapters from `PCBConductionSolution` and `SheetSolution`, terminal closure |
