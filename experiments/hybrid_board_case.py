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
from electrical.sheet_peec.hybrid_rt0 import RT0Space, Sheet
from electrical.sheet_peec.sheet_inductance import barrel_self_inductance
from electrical.sheet_peec.skin_filaments import graded_filaments, skin_depth_m, tube_resistance_ratio
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


def solve_case(prob, step_path, args, reference=None):
    """The hybrid solve of one current-field problem (and the uniform grid's, with --uniform)."""
    freq = float(prob["frequency_hz"]) if args.frequency is None else float(args.frequency)
    g = prob["grid"]
    pitch = g["pitch_mm"] * MM
    out = dict(case=prob["name"], role=prob.get("role"), frequency_hz=freq)
    if args.uniform:
        from electrical.sheet_peec.current_field_contract import solve_current_field_problem

        tu = time.time()
        settings = {"maximum_iterations": 400, "relative_tolerance": 1e-9}
        if args.factor_threads is not None:
            settings["near_factor_threads"] = args.factor_threads
        res = solve_current_field_problem(dict(prob, frequency_hz=freq), settings)
        out["uniform"] = dict(loss_w=res.metrics["i2r_loss_w"], seconds=round(time.time() - tu, 1),
                              iterations=res.metrics.get("solver_iterations"), branches=res.metrics.get("branch_count"),
                              filaments=res.metrics.get("filament_counts"))
        print("uniform", out["uniform"], flush=True)

    # A barrel the uniform grid laid on several cells comes as several
    # vertical connections of one name, each with its share of the barrel's
    # resistance: here it is one barrel again -- the shares in parallel, one
    # self inductance, its centre the mean of its cells.
    merged = {}
    for via in prob["vertical_connections"]:
        key = (via["name"], tuple(via["layers"]))
        merged.setdefault(key, []).append(via)
    vias = []
    for (name, layers_), group in merged.items():
        first = dict(group[0])
        first["resistance_ohm"] = 1.0 / sum(1.0 / v["resistance_ohm"] for v in group)
        cx = np.mean([(g["min_x_mm"] + (v["cell"]["x"] + 0.5) * g["pitch_mm"]) * MM for v in group])
        cy = np.mean([-(g["min_y_mm"] + (v["cell"]["y"] + 0.5) * g["pitch_mm"]) * MM for v in group])
        first["centre"] = (float(cx), float(cy))
        first["cells_merged"] = len(group)
        vias.append(first)
    prob = dict(prob, vertical_connections=vias)
    out["vias"] = dict(connections=sum(len(v) for v in merged.values()), barrels=len(vias))

    solids = [s_ for s_ in load_step(step_path).solids if (s_.bounds_m[1][2] - s_.bounds_m[0][2]) < 0.2 * MM]
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
    sheets, filaments_of, meshes, regions = [], {}, {}, {}
    for L in layers:
        region = section_region(solids, step_z[L["name"]])
        regions[L["name"]] = region
        mesh = mesh_region(region, pitch=args.pitch_mm * MM, offsets=graded_offsets(first, 2.0, args.band_mm * MM),
                           along=args.along_mm * MM, origin=(0.0, 0.0))
        # The layer's filaments, front to back as the uniform contract cuts them.
        stack = graded_filaments(L["thickness_mm"] * MM, L["center_z_mm"] * MM, max(freq, 1.0),
                                 resistivity_ohm_m=L["resistivity_ohm_m"], maximum_filaments=args.max_filaments) \
            if freq > 0 else None
        cuts = list(zip(stack.thicknesses_m, stack.centers_m)) if stack is not None else [(L["thickness_mm"] * MM, L["center_z_mm"] * MM)]
        filaments_of[L["name"]] = list(range(len(sheets), len(sheets) + len(cuts)))
        meshes[L["name"]] = (mesh, cuts)
        for t, z in cuts:
            sheets.append(Sheet(mesh.space, z, t, L["resistivity_ohm_m"], L["name"]))
        print(f"{L['name']}: section {region.area / MM**2:.1f} mm2, {len(mesh.space.kind)} elements, "
              f"{mesh.space.size} dofs, {len(cuts)} filaments", flush=True)
    t_mesh = time.time() - t0

    # Each via's hole wall on each layer: the boundary edges around its centre.
    from scipy.spatial import cKDTree

    def via_centre(via):
        return via["centre"]

    walls = {}  # (via index, layer) -> boundary edge rows, for the via that owns the hole
    owner = {}  # (via index, layer) -> the via whose node that hole is (vias sharing a hole share it)

    def pid(v, layer):
        return 2 * v + (0 if prob["vertical_connections"][v]["layers"][0] == layer else 1)

    def node(v, layer):
        return pid(owner.get((v, layer), v), layer)

    if args.via_model == "wall":
        from shapely.geometry import Point as _Point

        for name, (mesh, cuts) in meshes.items():
            sp_ = mesh.space
            region = regions[name]
            holes = [h for part in (region.geoms if hasattr(region, "geoms") else [region]) for h in part.interiors]
            hole_polys = [Polygon(h) for h in holes]
            tree = shapely.STRtree(hole_polys) if hole_polys else None
            mid = 0.5 * (sp_.vertices[sp_.boundary_edges[:, 0]] + sp_.vertices[sp_.boundary_edges[:, 1]])
            ring_owner = {}
            for v, via in enumerate(prob["vertical_connections"]):
                if name not in via["layers"] or tree is None:
                    continue
                barrel = via.get("resistance_barrel") or via.get("barrel") or {}
                limit = math.pi * (0.5 * barrel.get("outer_diameter_mm", 0.3) * MM + 0.1 * MM) ** 2
                centre = _Point(via_centre(via))
                found = [k for k in tree.query(centre) if hole_polys[k].contains(centre) and hole_polys[k].area < limit]
                if not found:
                    continue  # no hole of this via here: the barrel lands on copper
                if found[0] in ring_owner:
                    owner[v, name] = ring_owner[found[0]]
                    continue
                ring = holes[found[0]]
                near = shapely.distance(ring, shapely.points(mid)) < 5e-6
                rows = np.flatnonzero(near)
                if len(rows) >= 3:
                    walls[v, name] = list(rows)
                    owner[v, name] = v
                    ring_owner[found[0]] = v
        # rebuild each layer's space with its walls as ports, ids 2v (upper) / 2v + 1 (lower)
        for name, (mesh, cuts) in meshes.items():
            sp_ = mesh.space
            mine = [(v, rows) for (v, n), rows in walls.items() if n == name]
            space = RT0Space(sp_.vertices, sp_.corners, sp_.kind,
                             port_edges=tuple(sp_.boundary_edges[rows, :2] for _v, rows in mine))
            ids = tuple(pid(v, name) for v, _rows in mine)
            for k in filaments_of[name]:
                sheets[k] = Sheet(space, sheets[k].z, sheets[k].thickness, sheets[k].resistivity, name, port_ids=ids)

    # Terminals: each layer of a terminal takes an equal share (as the uniform
    # contract does), fed into the layer's first filament over the elements
    # inside its cells, by area.
    injected = [np.zeros(len(s_.space.kind), dtype=np.complex128) for s_ in sheets]
    for term in prob["terminals"]:
        current = term["current_a"]["real"] + 1j * term["current_a"]["imag"]
        by_layer = {}
        for c in term["cells"]:
            by_layer.setdefault(c["layer"], []).append(cell_box(c["x"], c["y"]))
        share = current / len(by_layer)
        for name, boxes in by_layer.items():
            area = unary_union(boxes)
            k = filaments_of[name][0]
            sp_ = sheets[k].space
            inside = np.flatnonzero(shapely.contains_xy(area, sp_.centroid[:, 0], sp_.centroid[:, 1]))
            if len(inside) == 0:
                raise ValueError(f"{term['name']}: no element inside its cells on {name}")
            injected[k][inside] += share * sp_.area[inside] / sp_.area[inside].sum()

    links = []
    # Filaments of a layer joined element by element.
    for name, ks in filaments_of.items():
        for a, b in zip(ks[:-1], ks[1:]):
            sp_ = sheets[a].space
            allel = np.arange(len(sp_.kind))
            links.append((a, b, allel, allel, sheets[a].resistivity * abs(sheets[a].z - sheets[b].z) / sp_.area))
    def _element_via(v, via):
            cx, cy = via["centre"]
            barrel = via.get("resistance_barrel") or via.get("barrel") or {}
            outer = barrel.get("outer_diameter_mm", 0.3) * MM
            drill = barrel.get("drill_diameter_mm", 0.0) * MM
            r_dc = via["resistance_ohm"]
            ratio = tube_resistance_ratio(outer / 2, 0.0 if barrel.get("solid_pin") else drill / 2, freq, 1.724e-8) if barrel and freq > 0 else 1.0
            inductance = VIA_L_SCALE * barrel_self_inductance(via["length_mm"] * MM, outer / 2, drill / 2)
            upper, lower = via["layers"][0], via["layers"][-1]
            z_up = next(L["center_z_mm"] for L in layers if L["name"] == upper) * MM
            z_lo = next(L["center_z_mm"] for L in layers if L["name"] == lower) * MM
            a = min(filaments_of[upper], key=lambda k: abs(sheets[k].z - z_lo))
            b = min(filaments_of[lower], key=lambda k: abs(sheets[k].z - z_up))
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


    # Vias: a barrel between its two hole walls (node ports), or, where a
    # layer has no hole at the via, the elements around it on the facing
    # filaments, paired by angle.
    import os
    _os_env = os.environ
    node_ports, node_links, node_element_links, wall_vias = [], [], [], 0
    one_sided = 0
    for v, via in enumerate(prob["vertical_connections"]):
        upper, lower = via["layers"][0], via["layers"][-1]
        if (v, upper) in owner and (v, lower) in owner:
            barrel = via.get("resistance_barrel") or via.get("barrel") or {}
            outer = barrel.get("outer_diameter_mm", 0.3) * MM
            drill = barrel.get("drill_diameter_mm", 0.0) * MM
            ratio = tube_resistance_ratio(outer / 2, 0.0 if barrel.get("solid_pin") else drill / 2, freq, 1.724e-8) \
                if barrel and freq > 0 else 1.0
            inductance = VIA_L_SCALE * barrel_self_inductance(via["length_mm"] * MM, outer / 2, drill / 2)
            node_ports += [node(v, upper), node(v, lower)]
            node_links.append((node(v, upper), node(v, lower), (via["resistance_ohm"] * ratio, inductance)))
            wall_vias += 1
            continue
        if any((v, layer) in owner for layer in (upper, lower)) and not _os_env.get("HYBRID_NO_LANDINGS"):
            # a hole on one layer only: the other end lands on copper; its node
            # is joined to the elements over the via on every filament there
            barrel = via.get("resistance_barrel") or via.get("barrel") or {}
            outer = barrel.get("outer_diameter_mm", 0.3) * MM
            drill = barrel.get("drill_diameter_mm", 0.0) * MM
            ratio = tube_resistance_ratio(outer / 2, 0.0 if barrel.get("solid_pin") else drill / 2, freq, 1.724e-8) \
                if barrel and freq > 0 else 1.0
            inductance = VIA_L_SCALE * barrel_self_inductance(via["length_mm"] * MM, outer / 2, drill / 2)
            node_ports += [node(v, upper), node(v, lower)]
            node_links.append((node(v, upper), node(v, lower), (via["resistance_ohm"] * ratio, inductance)))
            one_sided += 1
            solid = lower if (v, upper) in owner else upper
            port = node(v, solid)
            cx, cy = via_centre(via)
            for k in filaments_of[solid]:
                sp_ = sheets[k].space
                near = np.flatnonzero(np.hypot(sp_.centroid[:, 0] - cx, sp_.centroid[:, 1] - cy) < outer / 2)
                if len(near) == 0:
                    near = np.array([int(np.argmin(np.hypot(sp_.centroid[:, 0] - cx, sp_.centroid[:, 1] - cy)))])
                # a short to the copper: resistance of the land's thickness over its area
                r_land = sheets[k].resistivity * sheets[k].thickness / max(sp_.area[near].sum(), 1e-12)
                node_element_links.append((port, k, near, np.full(len(near), r_land * len(near))))
            wall_vias += 1
            continue
        _element_via(v, via)


    t1 = time.time()
    node_ports = sorted(set(node_ports))
    op = HybridOperator(sheets, links, freq, grid_pitch=args.grid_mm * MM, taper_radius=args.taper_mm * MM,
                        node_ports=node_ports, node_links=node_links,
                        node_element_links=node_element_links, log=lambda *_: None)
    t2 = time.time()
    s = np.concatenate(injected + [np.zeros(len(node_ports), dtype=np.complex128)])
    current, phi, iterations = solve_injection_fast(op, s, factor_threads=args.factor_threads, log=lambda *_: None)
    t3 = time.time()
    power = np.vdot(s, phi)
    r_real = op.resistance.copy()
    r_real.data = r_real.data.real
    dissipated = float(np.real(np.vdot(current, r_real @ current)))
    out["energy_check"] = dict(injected_w=float(power.real), dissipated_w=dissipated)
    import os as _os
    if _os.environ.get("HYBRID_VIA_DEBUG"):
        rows = []
        base = op.m_plane + sum(len(l[2]) for l in links)
        for k, (pa, pb, _z) in enumerate(node_links):
            rows.append((pa // 2, "wall", abs(current[base + k])))
        row0 = op.m_plane
        for (a_, b_, ea, eb, _r) in links:
            n = len(ea)
            if isinstance(_r, tuple):  # an element via
                rows.append((-1, "elements", abs(current[row0:row0 + n].sum())))
            row0 += n
        for (v, name), wrows in walls.items():
            sp_ = meshes[name][0].space
            mid = 0.5 * (sp_.vertices[sp_.boundary_edges[wrows, 0]] + sp_.vertices[sp_.boundary_edges[wrows, 1]])
            d = np.hypot(mid[:, 0] - via_centre(prob["vertical_connections"][v])[0], mid[:, 1] - via_centre(prob["vertical_connections"][v])[1])
            if d.max() > 0.16e-3 or len(wrows) < 8:
                print("wall", v, name, "edges", len(wrows), "dist mm %.3f..%.3f" % (d.min() * 1e3, d.max() * 1e3))
        rd = r_real.diagonal()
        cut = [0, op.m_plane, op.m_plane + sum(len(l[2]) for l in links), base + len(node_links), op.m]
        names = ["plane", "links", "barrels", "landings"]
        parts = {}
        for nm, (lo, hi) in zip(names, zip(cut[:-1], cut[1:])):
            if nm == "plane":
                parts[nm] = float(np.real(np.vdot(current[lo:hi], (r_real[lo:hi, lo:hi] @ current[lo:hi]))))
            else:
                parts[nm] = float(np.sum(rd[lo:hi] * np.abs(current[lo:hi]) ** 2))
        # plane loss per sheet
        per_sheet = [float(np.real(np.vdot(current[op.dof0[k]:op.dof0[k + 1]], r_real[op.dof0[k]:op.dof0[k + 1], op.dof0[k]:op.dof0[k + 1]] @ current[op.dof0[k]:op.dof0[k + 1]]))) for k in range(len(sheets))]
        print("loss parts", {k: round(v, 5) for k, v in parts.items()}, "per sheet", [round(x, 5) for x in per_sheet])
        per_via = {}
        for k, (pa, pb, _z) in enumerate(node_links):
            per_via[pa // 2] = complex(current[base + k])
        row0 = op.m_plane
        via_order = [v for v in range(len(prob["vertical_connections"]))]
        element_rows = []
        for (a_, b_, ea, eb, _r) in links:
            n = len(ea)
            if isinstance(_r, tuple):
                element_rows.append((row0, n))
            row0 += n
        ev = [v for v in via_order if v not in per_via and not any((v, L) in owner for L in prob["vertical_connections"][v]["layers"])]
        for v, (r0, n) in zip(ev, element_rows):
            per_via[v] = complex(current[r0:r0 + n].sum())
        import json as _json
        _json.dump({str(k): [c.real, c.imag] for k, c in per_via.items()}, open(_os.environ["HYBRID_VIA_DEBUG"], "w"))
        currents = sorted((c for _v, _m, c in rows), reverse=True)
        print("via currents (A): largest", [round(c, 3) for c in currents[:8]], "sum", round(sum(currents), 3), "count", len(currents))
    print("energy", out["energy_check"], flush=True)
    out.update(hybrid=dict(loss_w=power.real, dofs=op.m, elements=op.n, sheets=len(sheets), iterations=iterations,
                           wall_vias=wall_vias, element_vias=len(prob["vertical_connections"]) - wall_vias,
                           one_sided_vias=one_sided,
                           mesh_s=round(t_mesh, 1), operator_s=round(t2 - t1, 1), solve_s=round(t3 - t2, 1),
                           **op.timings))
    if reference is not None:
        out["reference"] = dict(loss_w=reference.get("i2r_loss_w"), frequency_hz=reference.get("frequency_hz"))
    if "uniform" in out:
        out["hybrid_over_uniform"] = out["hybrid"]["loss_w"] / out["uniform"]["loss_w"] - 1.0
    print(json.dumps(out, default=float), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", help="a dumped case ({problem, metrics}) or a recorded call ({problems: [...]})")
    ap.add_argument("step", help="the net's copper STEP, or a directory of copper_<ROLE>.step")
    ap.add_argument("--only", nargs="*", default=None, help="problem names to solve (a call)")
    ap.add_argument("--grid-mm", type=float, default=0.2)
    ap.add_argument("--pitch-mm", type=float, default=0.4)
    ap.add_argument("--along-mm", type=float, default=0.5)
    ap.add_argument("--band-mm", type=float, default=0.3)
    ap.add_argument("--taper-mm", type=float, default=2.0)
    ap.add_argument("--max-filaments", type=int, default=64)
    ap.add_argument("--via-model", choices=["wall", "elements"], default="wall")
    ap.add_argument("--factor-threads", default=None)
    ap.add_argument("--frequency", type=float, default=None, help="solve at this frequency instead of the case's")
    ap.add_argument("--uniform", action="store_true", help="also solve the case on the uniform grid, same frequency")
    args = ap.parse_args()
    data = json.load(open(args.case))
    cases = [(data["problem"], data.get("metrics"))] if "problem" in data else [(p, None) for p in data["problems"]]
    import os

    for prob, ref in cases:
        if args.only and prob["name"] not in args.only:
            continue
        step = args.step
        if os.path.isdir(step):
            step = os.path.join(step, f"copper_{prob['role']}.step")
        solve_case(prob, step, args, ref)


if __name__ == "__main__":
    main()
