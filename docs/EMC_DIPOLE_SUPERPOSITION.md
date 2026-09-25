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

### C++ direct summation (measured on `exp/cpp-thermal-emc`)

The tiled array path materialises `(tile × sources × 3)` complex temporaries
about thirty times per tile, so it is memory bound.  `native=True` on
`evaluate_fields` and `far_field_pattern` (or `PCB_NATIVE_EMC=1`, after
`python -m emc.tiled_dipole_superposition.native.build`) sums the pairs point
by point in C++ with the same complex128 formulas; OpenMP threads own disjoint
observation points (`native_threads`, default `PCB_NATIVE_THREADS` or one), so
the result is the array path's sum in another order and does not depend on the
thread count.  CPU and complex128 only; CuPy and complex64 keep the array path.

Measured on an Intel Xeon Platinum 8581C (16 cores / 32 threads), GCC 14.2.1,
NumPy 2.3.5, `OPENBLAS_NUM_THREADS=1`, random elements in a 4 cm cube with
complex moments at 300 MHz, a scan plane 5 cm above them, the default
2,048-direction sphere; medians of three runs
(`experiments/emc_native_benchmark.py`, `EMC_NATIVE_XEON_8581C_RESULTS.json`):

| Sources | Points | Array near field | C++ 1 thread | 4 | 8 | 16 | Array far field | C++ 1 thread | 16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 1,764 | 1,383 ms | 102 ms (13.6×) | 30 ms | 14 ms | 7.3 ms (189×) | 79 ms | 44 ms (1.8×) | 4.1 ms (19.4×) |
| 8,000 | 4,096 | 14,023 ms | 932 ms (15.1×) | 234 ms | 121 ms | 67 ms (209×) | 406 ms | 180 ms (2.3×) | 16 ms (24.9×) |

Near-field results agree with the array path to 3.1e-15 (relative, E and H),
far-field patterns and radiated power to the same level.  The near field
gains most because the array path is memory bound while the direct sum is
compute bound (one `sincos` and a few dozen flops per pair) and scales
linearly to the sixteen cores.  The far field was already one complex matmul
per tile, so the single-thread gain is small; the threaded sum still wins
because the matmul in the array path runs on one BLAS thread here.

Decision recorded in the JSON: criteria met (near field at least 2× on every
case, far field not slower, results within 1e-10).  The default stays the
array path.

### Current elements from a sheet-PEEC solve (measured on `exp/cpp-multiphysics-dc-emc`)

`dipoles_from_sheet_peec` assembled one Python tuple per branch. A plane at
the acceptance grid has 10^5 to 10^6 branches, so the loop was tried two
ways: a C++ kernel `sheet_branch_dipoles` in `_dipole_native` (OpenMP over
branches, `native=True` or `PCB_NATIVE_EMC=1`), and the same assembly with
NumPy indexing. Measured with `experiments/emc_sources_benchmark.py --shapes
100x100,300x300,600x600 --threads 1,4,16` on the Xeon Platinum 8581C (GCC
14.2.1, NumPy 2.3.5, `OPENBLAS_NUM_THREADS=1`;
`EMC_SOURCES_XEON_8581C_RESULTS.json`), three layers, fully occupied, a via
bank, random complex branch currents:

| Branches | Python loop | NumPy indexing | C++ 1 / 4 / 16 threads | C++ over NumPy |
|---:|---:|---:|---:|---:|
| 59,761 | 34.4 ms | 9.0 ms | 8.5 / 8.5 / 9.1 ms | 1.00 to 1.07× |
| 538,561 | 375.7 ms | 86.0 ms | 79.7 / 76.4 / 80.7 ms | 1.07 to 1.13× |
| 2,156,761 | 1,665.0 ms | 436.5 ms | 402.4 / 380.1 / 377.7 ms | 1.08 to 1.16× |

All three produce bit-identical positions and moments. The assembly is a
gather of a few index arrays into two output arrays and is bound by memory
traffic and by the `np.asarray` conversion of the mesh's branch lists, which
the kernel needs as well; threads do not help it. **Decision:** the NumPy
indexing form is the shipped path (`perf:` on the same branch, 3.8 to 4.4×
over the loop); the C++ kernel is measured, not adopted, and exists only on
the `exp/cpp-multiphysics-dc-emc` branch (tag `work/cpp-multiphysics-dc-emc`),
together with its test and `experiments/emc_sources_benchmark.py`. The near
and far fields above remain the parts of this package where C++ pays.

## Software boundary

| Module | Responsibility |
|---|---|
| `fields.py` | `CurrentDipoles`, exact dipole fields in tiles, ground-plane images, scan planes |
| `far_field.py` | sphere sampling, pattern, radiated power, directivity, dBµV/m |
| `limits.py` | CISPR 32 and FCC Part 15 tables, distance rescaling, margins |
| `moments.py` | electric and magnetic dipole moments and their radiated powers |
| `sources.py` | adapters from `PCBConductionSolution` and `SheetSolution`, terminal closure |
| `native_dipole.py`, `native/` | opt-in C++ direct summation for both field evaluations (built in place) |
