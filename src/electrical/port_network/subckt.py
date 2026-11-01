"""A port network as a SPICE ``.subckt``.

The subcircuit's ports are the reference pads first (one per conductor),
then the other pads in matrix order; its elements are ``L``/``K``/``R`` named after the pads.  Pad
names become node names with every character other than a letter, a digit
or ``_`` replaced by ``_``.  Values are written with seven significant
digits, couplings with six decimals.  Nothing here knows a simulator; any
SPICE that reads ``L``, ``K`` and ``R`` cards takes it.
"""

from __future__ import annotations

import re

from .realization import PortNetwork

_UNSAFE = re.compile(r"[^A-Za-z0-9_]")


def node_token(pad: str) -> str:
    """``C1.2`` -> ``C1_2``."""
    return _UNSAFE.sub("_", pad)


def subckt_text(network: PortNetwork, name: str) -> str:
    """The network as ``.subckt <name> <references...> <pads...> ... .ends``."""
    if not network.pads:
        raise ValueError("a port network needs at least one non-reference pad")
    references = set(network.reference_pads)
    inductive = bool(network.self_inductance_h)

    def outer(pad: str) -> str:
        return node_token(pad)

    def inner(pad: str) -> str:
        return f"{node_token(pad)}_m"

    def resistor_node(pad: str) -> str:
        if pad in references:
            return node_token(pad)
        return inner(pad) if inductive else outer(pad)

    ports = [*(node_token(r) for r in network.reference_pads), *(outer(p) for p in network.pads)]
    lines = [f"* {len(network.resistors)} resistors, {len(network.self_inductance_h)} inductors, "
             f"{len(network.couplings)} couplings at {network.frequency_hz:g} Hz"
             + (f"; {' '.join(network.flags)}" if network.flags else ""),
             f".subckt {name} {' '.join(ports)}"]
    for pad in network.pads:
        if pad in network.self_inductance_h:
            lines.append(f"L_{node_token(pad)} {outer(pad)} {inner(pad)} {network.self_inductance_h[pad]:.6e}")
    for pad_a, pad_b, k in network.couplings:
        lines.append(f"K_{node_token(pad_a)}__{node_token(pad_b)} L_{node_token(pad_a)} L_{node_token(pad_b)} {k:.6f}")
    for r in network.resistors:
        comment = f"  ; {' '.join(r.flags)}" if r.flags else ""
        lines.append(f"R_{node_token(r.pad_a)}__{node_token(r.pad_b)} {resistor_node(r.pad_a)} "
                     f"{resistor_node(r.pad_b)} {r.resistance_ohm:.6e}{comment}")
    lines.append(f".ends {name}")
    return "\n".join(lines) + "\n"


__all__ = ["node_token", "subckt_text"]
