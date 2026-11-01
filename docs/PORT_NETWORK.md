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
- Frequency dependence beyond one `R + jωL` (skin effect across the band) is
  not fitted; a wideband macromodel (vector fitting with passivity) is not
  implemented.
