"""The go/return strip pair on the hybrid triangle/square mesh, against the 2D FEM.

Two strips (width w, 71.12 um copper, centres 1.48336 mm apart, LM5145EVM-2L)
turned by ``--angle``: a port across the start of each, the far ends shorted.
R and L per length come from two lengths, so the ends cancel.

    python experiments/hybrid_strip_pair.py --width-mm 1.2 --freq 1e6 3e7 --angle 45
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

from electrical.sheet_peec.hybrid_mesh import build_hybrid_mesh, graded_offsets
from electrical.sheet_peec.hybrid_fast import HybridOperator, solve_ports_fast
from electrical.sheet_peec.hybrid_rt0 import Sheet, assemble, solve_driven
from electrical.sheet_peec.skin_filaments import graded_filaments, skin_depth_m

T, D, RHO = 71.12e-6, 1.48336e-3, 1.724e-8


def loop_impedance(length, width, angle, freq, args):
    u = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    poly = affinity.rotate(Polygon([(0, -width / 2), (length, -width / 2), (length, width / 2), (0, width / 2)]),
                           angle, origin=(0, 0))
    delta = skin_depth_m(max(freq, 1.0), RHO)
    first = args.first_fraction * delta if freq > 0 else args.band_mm * 1e-3 / 8
    mesh = build_hybrid_mesh(poly, pitch=args.pitch_mm * 1e-3, offsets=graded_offsets(first, 2.0, args.band_mm * 1e-3),
                             along=args.along_mm * 1e-3)
    tol = 1e-9
    if args.fast:
        # the far end shorted by links between the layers' facing filaments
        space = mesh.space.with_ports(lambda c: np.abs(c @ u) < tol)
        ids = lambda p: (p,)  # noqa: E731
    else:
        space = mesh.space.with_ports(lambda c: np.abs(c @ u) < tol, lambda c: np.abs(c @ u - length) < tol)
        ids = lambda p: (p, 2)  # noqa: E731
    sheets, links = [], []
    for name, zc, start_port in (("F", D / 2, 0), ("B", -D / 2, 1)):
        stack = graded_filaments(T, zc, max(freq, 1.0), maximum_filaments=args.max_filaments)
        first_sheet = len(sheets)
        for t, z in zip(stack.thicknesses_m, stack.centers_m):
            sheets.append(Sheet(space, z, t, RHO, name, port_ids=ids(start_port)))
        allel = np.arange(len(space.kind))
        for k in range(first_sheet, len(sheets) - 1):
            dz = abs(sheets[k].z - sheets[k + 1].z)
            links.append((k, k + 1, allel, allel, RHO * dz / space.area))
    t0 = time.time()
    if args.fast:
        end = np.flatnonzero(space.centroid @ u > length - 0.3e-3)
        f_sheets = [i for i, s_ in enumerate(sheets) if s_.layer == "F"]
        b_sheets = [i for i, s_ in enumerate(sheets) if s_.layer == "B"]
        links.append((f_sheets[-1], b_sheets[0], end, end, np.full(len(end), 1e-7)))
        op = HybridOperator(sheets, links, freq, grid_pitch=args.fast * 1e-3, log=lambda *_: None)
        t1 = time.time()
        _, _, driven, iterations = solve_ports_fast(op, {0: 1.0, 1: 0.0}, tolerance=1e-9, log=lambda *_: None)
        t2 = time.time()
        info = dict(elements=len(space.kind), dofs=op.m, sheets=len(sheets), squares=len(mesh.square_elements),
                    band=len(mesh.band_elements), gap=len(mesh.gap_elements), assemble_s=round(t1 - t0, 1),
                    solve_s=round(t2 - t1, 1), iterations=iterations, grid=list(op.grid.shape), **op.timings)
        return 1.0 / driven[0], info
    z, dmat, ports, _, _, y = assemble(sheets, links, freq)
    t1 = time.time()
    _, _, driven = solve_driven(z, dmat, ports, {0: 1.0, 1: 0.0}, floating=[2], y=y)
    t2 = time.time()
    info = dict(elements=len(space.kind), dofs=z.shape[0], sheets=len(sheets), squares=len(mesh.square_elements),
                band=len(mesh.band_elements), gap=len(mesh.gap_elements), assemble_s=round(t1 - t0, 1),
                solve_s=round(t2 - t1, 1))
    return 1.0 / driven[0], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-mm", type=float, default=1.2)
    ap.add_argument("--lengths-mm", type=float, nargs=2, default=[3.0, 6.0])
    ap.add_argument("--freq", type=float, nargs="+", default=[1e6])
    ap.add_argument("--angle", type=float, default=45.0)
    ap.add_argument("--pitch-mm", type=float, default=0.4)
    ap.add_argument("--band-mm", type=float, default=0.3)
    ap.add_argument("--along-mm", type=float, default=0.25)
    ap.add_argument("--first-fraction", type=float, default=0.25)
    ap.add_argument("--max-filaments", type=int, default=64)
    ap.add_argument("--fast", type=float, default=0.0, help="grid pitch (mm) of the FFT far field; 0 solves densely")
    args = ap.parse_args()
    for freq in args.freq:
        (l1, l2) = (x * 1e-3 for x in args.lengths_mm)
        z1, i1 = loop_impedance(l1, args.width_mm * 1e-3, args.angle, freq, args)
        z2, i2 = loop_impedance(l2, args.width_mm * 1e-3, args.angle, freq, args)
        per = (z2 - z1) / (l2 - l1) * 1e-3  # ohm per mm
        omega = 2 * math.pi * freq
        out = dict(width_mm=args.width_mm, angle=args.angle, freq_hz=freq, pitch_mm=args.pitch_mm, band_mm=args.band_mm,
                   along_mm=args.along_mm, R_mohm_per_mm=float(np.real(per)) * 1e3,
                   L_nh_per_mm=float(np.imag(per)) / omega * 1e9 if freq > 0 else None, short=i1, long=i2)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
