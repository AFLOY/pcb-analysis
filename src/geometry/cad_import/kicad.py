"""KiCad boards: STEP export through ``kicad-cli`` and a z-based body map.

KiCad (8 and later) exports tracks, pads, zones and vias as solids when asked,
but names them after the assembly node (``board 1/=>[0:1:1:3]``), so the body
map cannot use names.  In a KiCad export the z coordinate is unambiguous: the
board body spans the laminate from the bottom of the lowest core to the top
of the highest, each copper layer is its own 35 µm sheet just outside or
inside it, and a via barrel runs through the stack.  The map built here
selects by those z windows.  Export with ``--no-extra-pad-thickness`` so pads
have the same thickness as tracks and fall in the same window.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .bodymap import BoardSpec, BodyMap, CopperSpec, LayerSpec, ViaSpec
from .reader import StepModel, StepSolid

KICAD_STEP_COPPER_FLAGS: tuple[str, ...] = (
    "--include-tracks",
    "--include-pads",
    "--include-zones",
    "--include-inner-copper",
    "--no-extra-pad-thickness",
)


def kicad_cli_available(kicad_cli: str = "kicad-cli") -> bool:
    return shutil.which(kicad_cli) is not None


def kicad_cli_version(kicad_cli: str = "kicad-cli") -> str | None:
    try:
        result = subprocess.run([kicad_cli, "version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None


def export_kicad_step(
    board: str | Path,
    output: str | Path,
    *,
    kicad_cli: str = "kicad-cli",
    components: bool = True,
    substitute_models: bool = True,
    model_dir: str | Path | None = None,
    define_vars: Mapping[str, str] | None = None,
    extra_args: Sequence[str] = (),
) -> Path:
    """Export a ``.kicad_pcb`` to STEP with copper as solids.

    ``kicad-cli`` does not read the GUI's path configuration, so footprint
    models referenced through ``${KICAD<major>_3DMODEL_DIR}`` are skipped
    (with a warning per footprint) unless the variable is defined here:
    ``model_dir`` defines it for the running KiCad major version, and
    ``define_vars`` passes any further ``--define-var`` pairs.
    """

    board_path = Path(board)
    if not board_path.is_file():
        raise FileNotFoundError(board_path)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [kicad_cli, "pcb", "export", "step", *KICAD_STEP_COPPER_FLAGS, "--force"]
    if not components:
        command.append("--no-components")
    if substitute_models:
        command.append("--subst-models")
    variables = dict(define_vars or {})
    if model_dir is not None:
        variables.setdefault(kicad_model_dir_variable(kicad_cli), str(Path(model_dir)))
    for key, value in variables.items():
        command += ["--define-var", f"{key}={value}"]
    command += [*extra_args, "--output", str(out), str(board_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not out.is_file():
        raise RuntimeError(f"kicad-cli failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
    return out


def kicad_model_dir_variable(kicad_cli: str = "kicad-cli") -> str:
    """``KICAD<major>_3DMODEL_DIR`` for the installed ``kicad-cli`` (default major 10)."""

    version = kicad_cli_version(kicad_cli) or ""
    match = re.match(r"\s*(\d+)", version)
    major = match.group(1) if match else "10"
    return f"KICAD{major}_3DMODEL_DIR"


def default_kicad_model_dir(kicad_cli: str = "kicad-cli") -> Path | None:
    """The user or system ``3dmodels`` directory of the installed KiCad, if present."""

    variable = kicad_model_dir_variable(kicad_cli)
    major = variable[len("KICAD") : -len("_3DMODEL_DIR")]
    candidates = [
        Path.home() / ".local" / "share" / "kicad" / f"{major}.0" / "3dmodels",
        Path("/usr/share/kicad/3dmodels"),
        Path(f"/usr/share/kicad-{major}/3dmodels"),
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


# ------------------------------------------------------------------ footprints
_FOOTPRINT = re.compile(r'\(footprint\s+"(?P<name>[^"]*)"(?P<body>.*?)\n\t\)\n', re.S)
_FOOTPRINT_LAYER = re.compile(r'^\s*\(layer\s+"(?P<layer>[^"]+)"\)', re.M)
_FOOTPRINT_AT = re.compile(r"^\s*\(at\s+(?P<x>[-+0-9.eE]+)\s+(?P<y>[-+0-9.eE]+)(?:\s+(?P<rot>[-+0-9.eE]+))?\s*\)", re.M)
_FOOTPRINT_PROPERTY = re.compile(r'\(property\s+"(?P<key>[^"]+)"\s+"(?P<value>[^"]*)"')


@dataclass(frozen=True)
class KicadFootprint:
    """One placed footprint of a ``.kicad_pcb``: where KiCad puts its 3D model."""

    reference: str
    footprint: str
    x_mm: float
    y_mm: float
    rotation_deg: float
    layer: str
    properties: Mapping[str, str]

    @property
    def step_xy_mm(self) -> tuple[float, float]:
        """The footprint origin in the STEP frame (KiCad negates y on export)."""

        return (self.x_mm, -self.y_mm)


def read_kicad_footprints(board: str | Path) -> dict[str, KicadFootprint]:
    """Footprints of a ``.kicad_pcb`` keyed by reference designator.

    The ``(property "Key" "Value")`` fields of each footprint come along, so a
    design can carry per-part data (a thermal model name, junction-to-case and
    junction-to-board resistances, a power) as custom fields.
    """

    text = Path(board).read_text(encoding="utf-8")
    out: dict[str, KicadFootprint] = {}
    for match in _FOOTPRINT.finditer(text):
        body = match.group("body")
        properties = {m.group("key"): m.group("value") for m in _FOOTPRINT_PROPERTY.finditer(body)}
        reference = properties.get("Reference")
        layer = _FOOTPRINT_LAYER.search(body)
        at = _FOOTPRINT_AT.search(body)
        if reference is None or at is None:
            continue
        if reference in out:
            raise ValueError(f"reference designator {reference!r} is used by two footprints")
        out[reference] = KicadFootprint(
            reference=reference,
            footprint=match.group("name"),
            x_mm=float(at.group("x")),
            y_mm=float(at.group("y")),
            rotation_deg=float(at.group("rot") or 0.0),
            layer=layer.group("layer") if layer else "",
            properties=properties,
        )
    return out


def kicad_component_solids(model: StepModel) -> dict[str, tuple[StepSolid, ...]]:
    """Solids of a KiCad STEP export grouped by reference designator.

    KiCad places each footprint's 3D model as an assembly component named by
    the reference designator, so after flattening its solids are
    ``<board>/<refdes>/<model path>``. Copper and the board body are direct
    children (``<board>/=>[...]``) and are not components.
    """

    out: dict[str, list[StepSolid]] = {}
    for solid in model.solids:
        parts = solid.name.split("/")
        if len(parts) >= 3 and not parts[1].startswith("=>"):
            out.setdefault(parts[1], []).append(solid)
    return {reference: tuple(solids) for reference, solids in out.items()}


_STACKUP_LAYER = re.compile(
    r'\(layer\s+"(?P<name>[^"]+)"\s*\(type\s+"(?P<type>[^"]+)"\)(?P<rest>(?:\s*\((?!layer)[^()]*(?:\([^()]*\)[^()]*)*\))*)',
    re.S,
)
_THICKNESS = re.compile(r"\(thickness\s+([-+0-9.eE]+)")


@dataclass(frozen=True)
class KicadStackupEntry:
    name: str
    type: str
    thickness_mm: float | None


def read_kicad_stackup(board: str | Path) -> tuple[KicadStackupEntry, ...]:
    """The ``(stackup ...)`` entries of a ``.kicad_pcb``, top to bottom.

    Only the layer name, type and thickness are read, which is what placing
    the copper sheets in z needs; a board without a stackup section raises.
    """

    text = Path(board).read_text(encoding="utf-8")
    start = text.find("(stackup")
    if start < 0:
        raise ValueError(f"{board}: no (stackup ...) section; set the board stackup in KiCad first")
    # The stackup block ends where the next top-level setup key begins; scan
    # to its matching parenthesis.
    depth = 0
    end = start
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    entries = []
    for match in _STACKUP_LAYER.finditer(text[start:end]):
        thickness = _THICKNESS.search(match.group("rest") or "")
        entries.append(
            KicadStackupEntry(match.group("name"), match.group("type"), float(thickness.group(1)) if thickness else None)
        )
    if not any(entry.type == "copper" for entry in entries):
        raise ValueError(f"{board}: the stackup lists no copper layer")
    return tuple(entries)


def layers_from_kicad_stackup(
    entries: Sequence[KicadStackupEntry], *, resistivity_ohm_m: float = 1.68e-8
) -> tuple[tuple[LayerSpec, ...], float]:
    """Layer specs (bottom first) and the board top ``z`` from a KiCad stackup.

    Dielectrics between two copper layers add up (a core and its prepregs);
    silkscreen, mask and paste entries are ignored.
    """

    copper = [entry for entry in entries if entry.type == "copper"]
    if len(copper) < 1:
        raise ValueError("no copper layer in the stackup")
    if any(entry.thickness_mm is None for entry in copper):
        raise ValueError("every copper layer needs a thickness")
    dielectrics: list[float] = []
    between = 0.0
    seen_copper = False
    for entry in entries:
        if entry.type == "copper":
            if seen_copper:
                dielectrics.append(between)
                between = 0.0
            seen_copper = True
        elif seen_copper and entry.type in ("core", "prepreg", "dielectric") and entry.thickness_mm:
            between += entry.thickness_mm
    layers = kicad_step_layers(
        [entry.name for entry in copper],
        copper_thickness_mm=[float(entry.thickness_mm) for entry in copper],  # type: ignore[arg-type]
        dielectric_thickness_mm=dielectrics,
        resistivity_ohm_m=resistivity_ohm_m,
    )
    board_top = max(layer.bottom_z_mm for layer in layers)
    return layers, board_top


def kicad_step_layers(
    copper_layers: Sequence[str],
    *,
    copper_thickness_mm: float | Sequence[float] = 0.035,
    dielectric_thickness_mm: Sequence[float],
    resistivity_ohm_m: float = 1.68e-8,
) -> tuple[LayerSpec, ...]:
    """Layer specs for a KiCad export, bottom layer first.

    KiCad puts the top of the bottom copper at ``z = 0``: the bottom layer
    occupies ``[-t, 0]``, the first dielectric ``[0, d1]``, the next copper
    ``[d1, d1 + t]`` and so on.  ``copper_layers`` are named top to bottom as
    KiCad lists them (``F.Cu``, inner layers, ``B.Cu``); ``dielectric_thickness_mm``
    lists the ``len(copper_layers) - 1`` dielectrics from top to bottom.
    """

    names = list(copper_layers)
    if len(names) < 1:
        raise ValueError("at least one copper layer is required")
    dielectrics = list(dielectric_thickness_mm)
    if len(dielectrics) != len(names) - 1:
        raise ValueError("one dielectric thickness per gap between copper layers is required")
    if isinstance(copper_thickness_mm, (int, float)):
        thickness = [float(copper_thickness_mm)] * len(names)
    else:
        thickness = [float(value) for value in copper_thickness_mm]
        if len(thickness) != len(names):
            raise ValueError("one copper thickness per layer is required")
    # Build from the bottom: B.Cu is names[-1].
    layers: list[LayerSpec] = []
    z = 0.0  # top of the bottom copper
    bottom = names[-1]
    layers.append(LayerSpec(bottom, center_z_mm=-thickness[-1] / 2.0, thickness_mm=thickness[-1], resistivity_ohm_m=resistivity_ohm_m))
    for index in range(len(names) - 2, -1, -1):
        z += dielectrics[index]
        # Inner layers sit inside the laminate; KiCad models them as sheets
        # whose bottom is at the dielectric boundary.
        layers.append(LayerSpec(names[index], center_z_mm=z + thickness[index] / 2.0, thickness_mm=thickness[index], resistivity_ohm_m=resistivity_ohm_m))
        if index > 0:
            z += thickness[index]
    return tuple(layers)


def kicad_grid_origin_mm(
    min_x_mm: float, min_y_mm: float, rows: int, pitch_mm: float
) -> tuple[float, float]:
    """STEP-frame grid origin for a KiCad y-down grid.

    KiCad exports STEP with ``y`` negated. A KiCad grid whose cell ``(x, y)``
    has its corner at ``(min_x + x p, min_y + y p)`` maps onto a STEP grid
    with origin ``(min_x, -(min_y + rows p))`` when rows are read top-down
    (``rasterize_board(..., shape=(rows, cols), y_down=True)``); a board
    height that is not a multiple of the pitch would otherwise shift every
    row by the remainder.
    """

    return (float(min_x_mm), -(float(min_y_mm) + rows * float(pitch_mm)))


def kicad_step_body_map(
    layers: Sequence[LayerSpec],
    *,
    board_top_z_mm: float,
    tolerance_mm: float = 0.002,
    ignore: Sequence[str] = (),
) -> BodyMap:
    """A body map that tells board, copper layers and vias apart by z windows."""

    ordered = tuple(sorted(layers, key=lambda layer: layer.center_z_mm))
    # The body spans the whole laminate; inner copper sheets lie inside that
    # window but do not span it, so they are not mistaken for the board.
    board = BoardSpec(
        ".*",
        ordered,
        z_range_mm=(-tolerance_mm, board_top_z_mm + tolerance_mm),
        z_span_mm=(0.0, board_top_z_mm),
    )
    copper = tuple(
        CopperSpec(".*", layer.name, z_range_mm=(layer.bottom_z_mm - tolerance_mm, layer.top_z_mm + tolerance_mm))
        for layer in ordered
    )
    lowest, highest = ordered[0], ordered[-1]
    # A barrel reaches from inside the lowest layer to inside the highest; the
    # board body starts at the top of the lowest layer, so it does not.
    vias = (
        ViaSpec(
            ".*",
            z_range_mm=(lowest.bottom_z_mm - tolerance_mm, highest.top_z_mm + tolerance_mm),
            z_span_mm=(lowest.center_z_mm, highest.center_z_mm),
        ),
    )
    return BodyMap(board=board, copper=copper, vias=vias, ignore=tuple(ignore))


__all__ = [
    "KICAD_STEP_COPPER_FLAGS",
    "KicadFootprint",
    "KicadStackupEntry",
    "default_kicad_model_dir",
    "export_kicad_step",
    "kicad_component_solids",
    "kicad_model_dir_variable",
    "read_kicad_footprints",
    "layers_from_kicad_stackup",
    "read_kicad_stackup",
    "kicad_cli_available",
    "kicad_cli_version",
    "kicad_grid_origin_mm",
    "kicad_step_body_map",
    "kicad_step_layers",
]
