"""Copper as a network of ports, closed by lumped parts.

A board's copper, seen from its pads, is an n-port.  This package turns a
measured pad-port impedance into resistors and coupled inductors
(``realize_port_network``) and writes it as a SPICE subcircuit
(``subckt_text``).  It closes copper with lumped parts in the frequency
domain, either as a nodal network of series R-L-C branches
(``nodal_voltages``) or as parallel loops through shared copper
(``loop_impedance``).  ``BoardNetwork`` solves copper and every part that joins its nets
together, frequency by frequency, and keeps each part's own current: the
share of the ripple each capacitor shunts is a result.  It also reduces
simulated port-current waveforms to the time averages a linear copper
solve needs (``window_moments``).  ``PortImpedanceSweep`` interpolates the
copper's matrix over a band from a few solves, and ``refine_frequencies``
chooses those solves by checking each interval at its midpoint.

No solver of the copper lives here, and no circuit simulator is run.
"""

from .assembly import BoardNetwork, CopperPorts, NetworkSolution, NetworkSpectrum, NetworkTransfer
from .frequency import FREQUENCY_RTOL, distinct_frequencies, match_frequency
from .loops import loop_impedance, mesh_matrices
from .lumped import SeriesRLC, capacitor
from .moments import WindowMoments, close_kcl, port_block, window_moments
from .nodal import SHORT_OHM, Branch, nodal_voltages
from .parts import CurrentSource, ImpedanceModel, Part, TabulatedImpedance, Tie, VoltageSource
from .realization import (
    PortNetwork,
    Resistor,
    conductance_branches,
    realize_port_network,
    resistor_network,
)
from .subckt import node_token, subckt_text
from .sweep import (
    PORT_IMPEDANCE_SWEEP_SCHEMA,
    FrequencyCheck,
    PortImpedanceSweep,
    SweepPoint,
    refine_frequencies,
    relative_errors,
)

__all__ = [
    "FREQUENCY_RTOL",
    "distinct_frequencies",
    "match_frequency",
    "PORT_IMPEDANCE_SWEEP_SCHEMA",
    "FrequencyCheck",
    "PortImpedanceSweep",
    "SweepPoint",
    "refine_frequencies",
    "relative_errors",
    "VoltageSource",
    "Tie",
    "TabulatedImpedance",
    "Part",
    "NetworkSpectrum",
    "NetworkTransfer",
    "NetworkSolution",
    "ImpedanceModel",
    "CurrentSource",
    "CopperPorts",
    "BoardNetwork",
    "Branch",
    "PortNetwork",
    "Resistor",
    "SHORT_OHM",
    "SeriesRLC",
    "WindowMoments",
    "capacitor",
    "close_kcl",
    "conductance_branches",
    "loop_impedance",
    "mesh_matrices",
    "nodal_voltages",
    "node_token",
    "port_block",
    "realize_port_network",
    "resistor_network",
    "subckt_text",
    "window_moments",
]
