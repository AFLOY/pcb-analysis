"""One current-field case of a board solved on the hybrid mesh, against the uniform sheet solve.

Takes a dumped case (``{"problem", "metrics", ...}``: the current-field
problem the uniform grid solved and its result) and the net's copper STEP;
meshes each layer's section, feeds every terminal's current into the
elements inside its cells, joins the layers through each via (its barrel's
resistance and self inductance, between the elements around its hole) and
compares the loss.

    python experiments/hybrid_board_case.py CASE.json copper_NET.step [--grid-mm 0.2]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from electrical.sheet_peec.hybrid_fast import HybridOperator, solve_injection_fast
from electrical.sheet_peec.hybrid_mesh import graded_offsets, mesh_region
from electrical.sheet_peec.hybrid_rt0 import Sheet
from electrical.sheet_peec.sheet_inductance import barrel_self_inductance
from electrical.sheet_peec.skin_filaments import skin_depth_m, tube_resistance_ratio
from geometry.cad_import.reader import load_step

MM = 1e-3
import os
VIA_L_SCALE = float(os.environ.get("HYBRID_VIA_L_SCALE", "1"))


def chain_loops(seg, tol=1e-9):
    def key(p):
        return (round(p[0] / tol), round(p[1] / tol))

    nxt = {}
    for a, b in seg:
        nxt.setdefault(key(a), []).append((key(b), b))
    loops, used = [], set()
    for st in list(nxt):
        if st in used:
            continue
        loop, cur = [], st
        while cur not in used and cur in nxt:
            used.add(cur)
            k, pt = nxt[cur][0]
            loop.append(pt)
            cur = k
        if len(loop) >= 3:
            loops.append(np.array(loop))
    return loops


def section_region(solids, z):
    pieces = []
    for s_ in solids:
        if not (s_.bounds_m[0][2] <= z <= s_.bounds_m[1][2]):
            continue
        loops = chain_loops(s_.tessellate().section_segments(z))
        area = [0.5 * np.sum(l[:, 0] * np.roll(l[:, 1], -1) - np.roll(l[:, 0], -1) * l[:, 1]) for l in loops]
        outers = [l for l, a in zip(loops, area) if a > 0]
        holes = [l for l, a in zip(loops, area) if a < 0]
        for o in outers:
            po = Polygon(o)
            hs = [h for h in holes if po.contains(Polygon(h).representative_point())]
            pieces.append(shapely.make_valid(Polygon(o, hs)))
    return unary_union(pieces).buffer(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("step")
    ap.add_argument("--grid-mm", type=float, default=0.2)
    ap.add_argument("--pitch-mm", type=float, default=0.4)
    ap.add_argument("--along-mm", type=float, default=0.5)
    ap.add_argument("--band-mm", type=float, default=0.3)
    ap.add_argument("--frequency", type=float, default=None, help="solve at this frequency instead of the case's")
    ap.add_argument("--uniform", action="store_true", help="also solve the case on the uniform grid here, at that frequency")
    args = ap.parse_args()
    case = json.load(open(args.case))
    prob, ref = case["problem"], case["metrics"]
    freq = float(prob["frequency_hz"]) if args.frequency is None else float(args.frequency)
    uniform = None
    if args.uniform:
        from electrical.sheet_peec.current_field_contract import solve_current_field_problem

        tu = time.time()
        res = solve_current_field_problem(dict(prob, frequency_hz=freq), {"maximum_iterations": 400})
        uniform = dict(loss_w=res.metrics["i2r_loss_w"], seconds=round(time.time() - tu, 1),
                       iterations=res.metrics.get("solver_iterations"))
        print("uniform", uniform, flush=True)
    g = prob["grid"]
    pitch = g["pitch_mm"] * MM

    solids = [s_ for s_ in load_step(args.step).solids if (s_.bounds_m[1][2] - s_.bounds_m[0][2]) < 0.2 * MM]
    mids = sorted({round(0.5 * (s_.bounds_m[0][2] + s_.bounds_m[1][2]), 7) for s_ in solids})
    layers = sorted(prob["layers"], key=lambda L: -L["center_z_mm"])  # front first
    step_z = {L["name"]: z for L, z in zip(layers, sorted(mids, reverse=True))}

    # KiCad grid -> STEP frame: x as is, y negated (KiCad y grows downwards).
    def cell_box(x, y):
        x0 = (g["min_x_mm"] + x * g["pitch_mm"]) * MM
        y0 = (g["min_y_mm"] + y * g["pitch_mm"]) * MM
        return box(x0, -(y0 + pitch), x0 + pitch, -y0)

    t0 = time.time()
    delta = skin_depth_m(max(freq, 1.0), 1.724e-8)
    first = min(0.25 * delta, args.band_mm * MM / 8)
    sheets, meshes = [], {}
    for L in layers:
        region = section_region(solids, step_z[L["name"]])
        copper = unary_union([cell_box(c["x"], c["y"]) for c in prob["copper_by_layer"][L["name"]]])
        overlap = region.intersection(copper).area / max(region.area, copper.area)
        mesh = mesh_region(region, pitch=args.pitch_mm * MM, offsets=graded_offsets(first, 2.0, args.band_mm * MM),
                           along=args.along_mm * MM, origin=(0.0, 0.0))
        meshes[L["name"]] = mesh
        sheets.append(Sheet(mesh.space, L["center_z_mm"] * MM, L["thickness_mm"] * MM, L["resistivity_ohm_m"], L["name"]))
        print(f"{L['name']}: section {region.area / MM**2:.1f} mm2, grid copper overlap {overlap:.3f}, "
              f"{len(mesh.space.kind)} elements, {mesh.space.size} dofs", flush=True)
    index = {L["name"]: i for i, L in enumerate(layers)}
    t_mesh = time.time() - t0

    # Terminals: current over the elements inside the terminal's cells, by area.
    injected = [np.zeros(len(s_.space.kind), dtype=np.complex128) for s_ in sheets]
    for term in prob["terminals"]:
        current = term["current_a"]["real"] + 1j * term["current_a"]["imag"]
        by_layer = {}
        for c in term["cells"]:
            by_layer.setdefault(c["layer"], []).append(cell_box(c["x"], c["y"]))
        picks = []
        for name, boxes in by_layer.items():
            area = unary_union(boxes)
            sp_ = sheets[index[name]].space
            inside = np.flatnonzero(shapely.contains_xy(area, sp_.centroid[:, 0], sp_.centroid[:, 1]))
            picks.append((index[name], inside, sp_.area[inside]))
        total = sum(a.sum() for _, _, a in picks)
        if total == 0:
            raise ValueError(f"{term['name']}: no element inside its cells")
        for k, inside, a in picks:
            injected[k][inside] += current * a / total

    # Vias: the elements around each hole on both layers, paired by angle.
    links = []
    for via in prob["vertical_connections"]:
        cx = (g["min_x_mm"] + (via["cell"]["x"] + 0.5) * g["pitch_mm"]) * MM
        cy = -(g["min_y_mm"] + (via["cell"]["y"] + 0.5) * g["pitch_mm"]) * MM
        barrel = via.get("resistance_barrel") or via.get("barrel") or {}
        outer = barrel.get("outer_diameter_mm", 0.3) * MM
        drill = barrel.get("drill_diameter_mm", 0.0) * MM
        r_dc = via["resistance_ohm"]
        ratio = tube_resistance_ratio(outer / 2, 0.0 if barrel.get("solid_pin") else drill / 2, freq, 1.724e-8) if barrel and freq > 0 else 1.0
        inductance = VIA_L_SCALE * barrel_self_inductance(via["length_mm"] * MM, outer / 2, drill / 2)
        a, b = index[via["layers"][0]], index[via["layers"][-1]]
        sa, sb = sheets[a].space, sheets[b].space
        reach = outer / 2 + 0.15 * MM
        ea = np.flatnonzero(np.hypot(sa.centroid[:, 0] - cx, sa.centroid[:, 1] - cy) < reach)
        eb = np.flatnonzero(np.hypot(sb.centroid[:, 0] - cx, sb.centroid[:, 1] - cy) < reach)
        if len(ea) == 0 or len(eb) == 0:
            raise ValueError(f"{via['name']}: no copper around the hole on a layer")
        ang_a = np.arctan2(sa.centroid[ea, 1] - cy, sa.centroid[ea, 0] - cx)
        ang_b = np.arctan2(sb.centroid[eb, 1] - cy, sb.centroid[eb, 0] - cx)
        pair_b = eb[np.argmin(np.abs(np.angle(np.exp(1j * (ang_a[:, None] - ang_b[None, :])))), axis=1)]
        n = len(ea)
        links.append((a, b, ea, pair_b, (np.full(n, r_dc * ratio * n), np.full(n, inductance * n))))

    t1 = time.time()
    op = HybridOperator(sheets, links, freq, grid_pitch=args.grid_mm * MM)
    t2 = time.time()
    s = np.concatenate(injected)
    current, phi, iterations = solve_injection_fast(op, s)
    t3 = time.time()
    power = np.vdot(s, phi)  # sum phi s*: the complex power for phasor amplitudes, times two
    out = dict(case=prob["name"], frequency_hz=freq, hybrid_dofs=op.m, hybrid_elements=op.n,
               uniform_branches=ref.get("branch_count"), loss_half_w=0.5 * power.real, loss_full_w=power.real,
               reference_loss_w=ref["i2r_loss_w"], reference_frequency_hz=prob["frequency_hz"], uniform_here=uniform,
               iterations=iterations,
               reference_iterations=ref.get("solver_iterations"),
               mesh_s=round(t_mesh, 1), operator_s=round(t2 - t1, 1), solve_s=round(t3 - t2, 1), **op.timings)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
