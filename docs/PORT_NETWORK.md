# Port networks: copper seen from its pads, closed by lumped parts

`src/electrical/port_network/` is the part of the library that treats copper
as a circuit element.  A solver in this repository answers one conductor
shape at a time (REQUIREMENTS.md).  This package takes what such solves
measure between pads and carries it into circuit form:

- realizes a pad-port impedance as resistors and coupled inductors
  (`realize_port_network`), with a reference per conductor when the copper
  is in several;
- writes the result as a SPICE subcircuit (`subckt_text`);
- closes copper with lumped parts in the frequency domain
  (`nodal_voltages`, `loop_impedance`);
- reduces a circuit simulator's port-current waveforms to the time averages
  a linear copper solve needs (`window_moments`, `close_kcl`).

It holds no field solver and runs no circuit simulator.  Choosing which
pads, which frequency, which parts, and running SPICE belong to the caller
(Kicad_PowerOpt keeps its deck contract and its LTspice/ngspice adapters).

The code came from Kicad_PowerOpt (`physics/port_impedance.py`,
`electrical/hot_loop.py`, `electrical/decoupling.py` and
`electrical/port_moments.py`) on 2026-11-01.  It was moved with the same
arithmetic: Kicad_PowerOpt's hot-loop, ripple-transfer, network and
conductance-block outputs on its test fixtures are identical before and
after the move, with single-threaded BLAS.

## Realization (`realization.py`)

Copper seen from `n + 1` pads, one of them the reference, is an n-port:
`z[j, k]` = V(pad j) − V(reference) for 1 A into pad k and out of the
reference.  `realize_port_network(z, pads, reference, f)` keeps resistance
and inductance apart:

- every non-reference pad enters through its own inductor to an inner node;
- the inductors carry `Im Z / ω` as self terms and couplings `k`;
- a resistor network between the inner nodes and the reference carries
  `(Re Z)^-1`, with branch conductances from `conductance_branches`.

The network reproduces `Z` at the extraction frequency and is a
positive-definite resistor network at DC, which an operating-point solve
needs.  A series R-L branch network would match only at one frequency and
be singular at DC.

Flags, not silent fixes:

| flag | meaning |
|---|---|
| `negative_resistance` (on a resistor) | two pads share most of their copper path; the network as a whole still equals `Re Z`, so the branch is kept |
| `resistance_matrix_not_positive_definite` | `Re Z` is not a passive resistance |
| `inductance_matrix_not_positive_definite` | the self terms are kept and every coupling dropped |
| `coupling_clipped` | a coupling of magnitude 1 or more was set to ±0.999 |

`PortNetwork.impedance_matrix(f)` rebuilds `Z` from the elements for checking.

### Several conductors

`realize_port_network(z, pads, references, f)` takes either one reference
pad or each pad's reference pad (`SheetPortImpedance.references` from
`electrical.sheet_peec.solve_port_impedance`).  With several conductors
(a loop's go and return):

- the resistor network is built per conductor from its own block of `Re Z`,
  because a resistor between two conductors would join them at DC;
- the inductors carry the whole `Im Z / ω`, couplings across conductors
  included: that is what the cross-conductor matrix is for;
- the resistance between conductors is left out, and its largest magnitude
  is kept in `dropped_cross_resistance_ohm`.

That resistance is not always small.  It is the loss the driven conductor's
field causes by eddy currents in the other.  On two 5 x 40 cell strips
(F.Cu over In1.Cu, 0.2 mm apart, 1 MHz) it is 0.34 mOhm against 4.21 mOhm
self (`tests/test_sheet_ports.py`).  The realized loop keeps `L` exactly
but reads `R` higher by twice the mutual resistance, 9 % here.  A mutual
resistance that vanishes at DC needs a frequency-dependent element (a lossy
coupled loop, or a fitted macromodel), which is not implemented.  The
subcircuit lists every reference pad first among its ports.

## SPICE export (`subckt.py`)

`subckt_text(network, name)` writes `.subckt <name> <reference> <pads...>`.
Pad names become nodes with every character other than a letter, digit or
`_` replaced by `_`; inner nodes are `<pad>_m`; elements are `L_<pad>`,
`K_<a>__<b>` and `R_<a>__<b>`.  Values carry seven significant digits and
couplings six decimals.  Any SPICE that reads `L`, `K` and `R` cards takes it.

## Lumped closure (`lumped.py`, `nodal.py`, `loops.py`)

`SeriesRLC(R, L, C)` is a part over the band a layout cares about: a capacitor
is its capacitance at the operating bias, its ESR, and the ESL of the part and
its mounting (`capacitor(c, esr, esl)`).  `C = 0` means no capacitor.

- `nodal_voltages(node_count, branches, injection, reference, f)` assembles
  the admittance matrix of series branches and solves the node voltages
  directly.  A branch of zero impedance is taken as `SHORT_OHM` (1e-9 Ω).
  A singular network raises `numpy.linalg.LinAlgError`.
- `loop_impedance(R, L, parts, f)` closes parallel meshes through shared
  copper: mesh `k` runs from a switch through part `k`.  With the copper's
  mesh matrices `R + jωL` and each part on the diagonal, the mesh currents
  for 1 V are `(R + jωL + Zc)^-1 1`.  Their sum is the switch current, and its
  inverse the loop impedance.  `mesh_matrices(Zm, f)` symmetrises a solved
  mesh matrix and splits it into `R` and `L`.  The copper is taken as
  `R + jωL` from one solve, which holds where one filament through the foil
  describes it.

## Copper and parts solved together (`parts.py`, `assembly.py`)

`BoardNetwork(copper, parts, ground, current_sources=, voltage_sources=)`
solves the copper with every part that joins its nets, at each frequency
the copper has a matrix for.  Every part stays its own element between its
own pads, so its current is a result at every frequency.  The share of the
ripple each decoupling capacitor shunts is what shows its filtering, and a
reduction that merged the parts would lose it.

- Nodes are pads.  The copper enters as `Bᵀ Z(f)⁻¹ B`: `B` is the port
  incidence, `+1` at a port pad and `-1` at its conductor's reference pad.
  Copper in several conductors (from `solve_port_impedance`) couples
  through the cross terms of `Z`.
- A pad that is neither a copper port nor a reference is a node of the
  parts alone (between a source and its series resistor, say).  Joined by
  one element only it is a dead end whose element carries no current, and
  the network refuses it, naming the copper's pads of the same part: with
  `merge_pins` the copper's pad is `C2:VIN`, and a part stamped on `C2.1`
  would otherwise carry 1e-17 A without a word.
- `Part(name, a, b, model)` is stamped between pads `a` and `b`.  The model
  is anything with `impedance(omega)`:
  - `SeriesRLC`
  - `TabulatedImpedance`: a measured or simulated curve, such as a capacitor
    at its DC bias or a power inductor, interpolated as `log|Z|` and phase
    in `log f`; it refuses outside its table
  - `Tie`: an ideal connection, solved as a 0 V branch so its current is
    reported
- `CurrentSource` (for example a switch's current spectrum) and
  `VoltageSource` are given as phasors per frequency.  A voltage source's
  current is a result.
- The solve is modified nodal analysis, dense, one complex solve per
  frequency.  The result `NetworkSolution` holds:
  - pad voltages against the ground pad
  - each part's current (from `a` to `b`)
  - each source's current from `a` to `b` (a voltage source's solved, a
    current source's own, per case)
  - the current into each copper port
- Each frequency is factorised once (`transfer(f)`), with one right-hand
  side per source at unit strength (current sources at 1 A, then voltage
  sources at 1 V).  `NetworkTransfer` holds every response per unit source:
  pad voltages, part currents, voltage-source currents and copper port
  currents, one column per source, kept per frequency.  `solve(f)` applies
  the sources' own spectra to it.
- `solve_cases({case: {source: {f: phasor}}})` solves many excitations at
  once, each frequency with one factorisation and one matrix product for all
  cases.  A case is an operating point of the circuit, its spectra the
  harmonics of each source; a source or a frequency a case does not list
  carries nothing in it.  Each case comes back as a `NetworkSpectrum`, so
  every part's current spectrum is a result per case.  What the cases mean
  (the share a capacitor takes, the dB of a transfer, a hot loop's value) is
  the caller's to define.
- A frequency without a copper matrix is refused, not interpolated, unless
  the copper is a `PortImpedanceSweep` (below).  A pad
  or a conductor with no path to the ground pad is reported as singular.

Checked (`tests/test_board_network.py`): on the strips below with a 1 uF
and a 10 uF capacitor, a switch current source and an input voltage source,
three cases at 1 and 30 MHz from `solve_cases` agree with each case solved by
its own network within 1e-12 relative, and 49 cases reuse one factorisation.
`solve` through the transfer differs from the earlier direct solve by
rounding only (2.3e-16 relative in the capacitor currents).

On two strips (VIN over PGND, a switch at one end, a capacitor of 1 uF /
2 mOhm / 0.5 nH at the middle and one at the far end;
`tests/test_board_network.py`), the switch current closes through the two
capacitors.  The nearer one takes 0.50 of it at 100 kHz and 0.81 at 10 MHz.

## Copper over a band (`sweep.py`)

A board network is solved at many frequencies -- every harmonic of every
operating case -- and the copper's field solve is the expensive part.
`PortImpedanceSweep(points)` holds the matrices solved at a few frequencies
and answers any frequency inside their band by interpolation, entry by entry
(couplings across conductors included):

- `R = Re Z`: monotone cubic (PCHIP) in `sqrt(f)`, where the skin-effect law
  is a straight line and the bend from the DC value is followed;
- `L = Im Z / omega`: PCHIP in `log f`.

`sweep.at(f)` returns the solved matrix at a solved frequency and an
interpolated one (`interpolated=True`) inside the band; outside it refuses.
`SweepPoint.converged` carries the port solves' flags (for an interpolated
matrix, those of every point solved).

Frequencies are matched within `FREQUENCY_RTOL` (1e-9 relative,
`match_frequency`), not to the bit: a geometric midpoint, a harmonic
`k * f0` or a JSON round trip can bring a frequency back with its last bits
changed.  This holds for the sweep's solved points, `BoardNetwork`'s copper,
the sources' spectra (where a missed key used to carry 0 A without a
word), `PortImpedanceResult.at` and `network_emission`.

`refine_frequencies(solve, frequencies, tolerance=0.01, maximum_solves=24,
minimum_ratio=1.05)` chooses the solves.  Each interval between solved points
is checked at its geometric midpoint: the copper is solved there and compared
with what the interpolation over the points solved so far predicted, R and L
each against the largest entry of its own matrix.  An interval that misses is
split and its halves checked in turn.  The checks (`FrequencyCheck`) are kept
with the sweep, and `converged` says whether every interval met the tolerance before a split got narrower than
`minimum_ratio` or the solves ran out.  An interval between two frequencies
asked for that is already that narrow is left unchecked and does not count
against `converged`.
The port-impedance contract runs it with
`settings["frequency_refinement"]` (`docs/SHEET_PEEC.md`).

`BoardNetwork(sweep, ...)` takes a sweep as its copper: a frequency the
copper was not solved at is served by the interpolation and the solution
says so (`NetworkSolution.copper_interpolated`, also on `NetworkTransfer`).
`copper_converged` says likewise whether the port solves behind the copper's
matrix converged (`None` when the copper carries no flags), and a matrix
from solves that did not converge raises a `RuntimeWarning` once per
frequency.
Copper given as matrices still refuses a frequency it has no matrix for.
`PortImpedanceSweep.to_mapping()` / `from_mapping()` serialise a sweep with
its checks and forms (`port-impedance-sweep/v1`).

### The forms (measured, `PORT_SWEEP_RESULTS.json`)

Each pair of forms refined over the band from its two ends, then compared
not at its own checks but at validation frequencies solved directly: a
synthetic two-port (R from its DC value into `sqrt f` at 200 kHz, L falling
from 12 to 9 nH around 1 MHz; 80 points over 10 kHz-100 MHz) and the
contract's two strips with up to six skin filaments per layer (30 points over
0.1-100 MHz; one solve 1.22 s on AMD Ryzen 7 9700X 8-Core Processor):

| copper | R / L form | tolerance | solves | converged | worst R between checks | worst L between checks |
|---|---|---:|---:|---|---:|---:|
| synthetic | linear-sqrt / pchip-log | 0.01 | 15 | yes | 0.38 % | 0.114 % |
| synthetic | linear-sqrt / pchip-log | 0.002 | 33 | yes | 0.05 % | 0.043 % |
| synthetic | pchip-sqrt / pchip-log | 0.01 | 9 | yes | 0.15 % | 0.211 % |
| synthetic | pchip-sqrt / pchip-log | 0.002 | 15 | yes | 0.02 % | 0.043 % |
| synthetic | pchip-log / pchip-log | 0.01 | 15 | yes | 0.32 % | 0.028 % |
| synthetic | pchip-log / pchip-log | 0.002 | 33 | yes | 0.01 % | 0.003 % |
| synthetic | linear-log / linear-log | 0.01 | 47 | yes | 0.24 % | 0.044 % |
| synthetic | linear-log / linear-log | 0.002 | 80 | no | 0.06 % | 0.006 % |
| two strips | linear-sqrt / pchip-log | 0.01 | 17 | yes | 1.92 % | 0.159 % |
| two strips | linear-sqrt / pchip-log | 0.002 | 43 | yes | 0.12 % | 0.003 % |
| two strips | pchip-sqrt / pchip-log | 0.01 | 15 | yes | 0.58 % | 0.019 % |
| two strips | pchip-sqrt / pchip-log | 0.002 | 31 | no | 0.12 % | 0.002 % |
| two strips | pchip-log / pchip-log | 0.01 | 15 | yes | 0.84 % | 0.019 % |
| two strips | pchip-log / pchip-log | 0.002 | 33 | no | 0.11 % | 0.002 % |
| two strips | linear-log / linear-log | 0.01 | 27 | yes | 0.32 % | 0.062 % |
| two strips | linear-log / linear-log | 0.002 | 71 | no | 0.05 % | 0.008 % |

`a + b sqrt(f)` between neighbours (linear in `sqrt f`) passed its 1 % checks
on the strips yet missed R by 1.9 % between them.  PCHIP in `sqrt f` held
1 % everywhere with the fewest solves (24 over both coppers at 1 %, against
30 and 74 for the other pairs that also held it) and is the default; `forms=`
selects another pair.  At 0.2 % no pair converges on the strips: their R
steps where the number of skin filaments changes with frequency, and the
refinement stops at `minimum_ratio` and says so.

## Emission of a network

`emc.tiled_dipole_superposition.network_dipoles(network, solution, copper,
mesh, pad_positions)` turns a `NetworkSolution` into radiating current
elements: the copper's current is the sum of the port solves kept with its
matrix (`solve_port_impedance(keep_solutions=True)`,
`SheetPortImpedance.superpose`) weighted by the solution's port currents,
and each part and source carries its current along the line between its
pads.  `network_emission` gives each frequency's far field, margin to a limit
line and near-field |H| (docs/EMC_DIPOLE_SUPERPOSITION.md).

## Port moments (`moments.py`)

The copper is linear, so for any port-current waveforms the time-averaged
loss of every copper element follows from `<I_k I_l>` alone.  This is the
matrix `multiphysics.staggered_coupling.PortExcitation` takes.
`window_moments` integrates the simulator's own (non-uniform) time points
with the trapezoid rule:

- means and second moments over the window;
- harmonic phasors `(2/T) ∫ I e^{-jωt} dt`, only when the window spans a
  whole number of periods.

`close_kcl(moments, groups_by_port)` removes each group's residual (a group
is the ports of one conductor) with one projection, applied to the means, the
second moments and the harmonics alike.  A residual above
`maximum_relative_residual` of the carried current is refused as a circuit
error.

## Not done

- The mutual resistance between conductors is dropped (see above).
- The cross-conductor matrix is measured by
  `electrical.sheet_peec.solve_port_impedance` on the CPU path only.
- Across a band the copper's matrices are interpolated between solves
  (`PortImpedanceSweep`), not fitted: a wideband macromodel for time-domain
  export (vector fitting with passivity) is not implemented.
