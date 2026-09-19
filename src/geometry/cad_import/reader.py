"""STEP input through OpenCASCADE (``OCP``); the only module that imports it.

A STEP file is flattened into named solids in metres.  Every public result is
a plain dataclass or NumPy array; the OpenCASCADE shape stays private to the
``StepSolid`` that owns it and is used only through ``contains``.  Synthetic
boxes and cylinders are provided so fixtures and demos need no CAD tool, and
``write_step`` round-trips them through a real STEP file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .mesh import ClassifyMethod, TriangleMesh, default_method

MM = 1.0e-3
DEFAULT_DEFLECTION_M = 5.0e-6


def _ocp() -> Any:
    try:
        import OCP  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "STEP input needs the OpenCASCADE binding: pip install 'pcb-analysis[cad]'"
        ) from exc
    from OCP.Message import Message, Message_Gravity

    printers = Message.DefaultMessenger_s().Printers()
    for index in range(1, printers.Length() + 1):
        printers.Value(index).SetTraceLevel(Message_Gravity.Message_Fail)
    return OCP


def ocp_available() -> bool:
    try:
        import OCP  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class StepSolid:
    """One solid of the model: its name, bounds and volume in metres."""

    name: str
    index: int
    bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    volume_m3: float
    _shape: Any = field(repr=False, compare=False)
    _meshes: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def size_m(self) -> tuple[float, float, float]:
        lo, hi = self.bounds_m
        return (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])

    @property
    def centre_m(self) -> tuple[float, float, float]:
        lo, hi = self.bounds_m
        return ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0)

    def tessellate(self, deflection_m: float = DEFAULT_DEFLECTION_M) -> TriangleMesh:
        """Outward-oriented triangles of the solid, cached per deflection.

        Planar faces tessellate exactly; curved faces deviate by at most
        ``deflection_m`` (default 5 µm), with a 0.2 rad angular deflection.
        """

        key = float(deflection_m)
        cached = self._meshes.get(key)
        if cached is not None:
            return cached
        _ocp()
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS

        BRepMesh_IncrementalMesh(self._shape, key / MM, False, 0.2, True)
        triangles: list[np.ndarray] = []
        explorer = TopExp_Explorer(self._shape, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face(explorer.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            explorer.Next()
            if triangulation is None:
                continue
            transform = location.Transformation()
            nodes = np.array(
                [
                    (point.X(), point.Y(), point.Z())
                    for point in (
                        triangulation.Node(index).Transformed(transform)
                        for index in range(1, triangulation.NbNodes() + 1)
                    )
                ],
                dtype=np.float64,
            )
            reversed_face = face.Orientation() == TopAbs_REVERSED
            for index in range(1, triangulation.NbTriangles() + 1):
                a, b, c = triangulation.Triangle(index).Get()
                if reversed_face:
                    a, c = c, a
                triangles.append(nodes[[a - 1, b - 1, c - 1]])
        if not triangles:
            raise ValueError(f"solid {self.name!r} produced no triangles")
        mesh = TriangleMesh(np.asarray(triangles) * MM, key)
        if mesh.signed_volume_m3() <= 0.0:
            raise ValueError(f"solid {self.name!r} tessellated with inward orientation")
        self._meshes[key] = mesh
        return mesh

    def contains(
        self,
        points_m: np.ndarray,
        *,
        tolerance_m: float = 1.0e-9,
        method: ClassifyMethod = "auto",
        deflection_m: float = DEFAULT_DEFLECTION_M,
        threads: int | None = None,
    ) -> np.ndarray:
        """Point-in-solid test for ``(n, 3)`` points in metres.

        ``method`` ``"occ"`` classifies each point with OpenCASCADE (exact on
        the B-rep, one call per point); ``"native"`` and ``"numpy"`` use the
        winding number over the tessellation; ``"auto"`` follows
        ``mesh.default_method`` (native when built, else NumPy).
        """

        chosen = default_method() if method == "auto" else method
        if chosen != "occ":
            return self.tessellate(deflection_m).contains(points_m, method=chosen, threads=threads)
        _ocp()
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON

        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_m must have shape (n, 3)")
        lo = np.asarray(self.bounds_m[0]) - tolerance_m
        hi = np.asarray(self.bounds_m[1]) + tolerance_m
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        candidates = np.nonzero(inside)[0]
        if candidates.size == 0:
            return inside
        classifier = BRepClass3d_SolidClassifier(self._shape)
        tolerance_mm = tolerance_m / MM
        result = np.zeros(candidates.size, dtype=bool)
        for slot, index in enumerate(candidates.tolist()):
            x, y, z = points[index] / MM
            classifier.Perform(gp_Pnt(float(x), float(y), float(z)), tolerance_mm)
            state = classifier.State()
            result[slot] = state == TopAbs_IN or state == TopAbs_ON
        inside[candidates] = result
        return inside


@dataclass(frozen=True)
class StepModel:
    """Flattened solids of one STEP file (or of synthetic shapes)."""

    solids: tuple[StepSolid, ...]
    source: str = ""

    def __post_init__(self) -> None:
        solids = tuple(self.solids)
        if not solids:
            raise ValueError("the model has no solid")
        names = [solid.name for solid in solids]
        if len(set(names)) != len(names):
            raise ValueError("solid names must be unique after flattening")
        object.__setattr__(self, "solids", solids)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(solid.name for solid in self.solids)

    def solid(self, name: str) -> StepSolid:
        for solid in self.solids:
            if solid.name == name:
                return solid
        raise KeyError(f"no solid named {name!r}; the model has {self.names}")

    def matching(self, pattern: str) -> tuple[StepSolid, ...]:
        """Solids whose name fully matches the regular expression."""

        regex = re.compile(pattern)
        return tuple(solid for solid in self.solids if regex.fullmatch(solid.name))

    def bounds_m(self, solids: Iterable[StepSolid] | None = None) -> tuple[np.ndarray, np.ndarray]:
        chosen = tuple(solids) if solids is not None else self.solids
        lo = np.min([solid.bounds_m[0] for solid in chosen], axis=0)
        hi = np.max([solid.bounds_m[1] for solid in chosen], axis=0)
        return lo, hi


# --------------------------------------------------------------------- shapes
def _solid_from_shape(name: str, index: int, shape: Any) -> StepSolid:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    # Optimal bounds without the shape tolerance, so a 20 mm box reports
    # 20 mm and not 20.0000002 mm (which would cost the grid a whole cell).
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    lo, hi = box.CornerMin(), box.CornerMax()
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return StepSolid(
        name=name,
        index=index,
        bounds_m=(
            (lo.X() * MM, lo.Y() * MM, lo.Z() * MM),
            (hi.X() * MM, hi.Y() * MM, hi.Z() * MM),
        ),
        volume_m3=float(properties.Mass()) * MM**3,
        _shape=shape,
    )


def _explode_solids(shape: Any) -> list[Any]:
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    return solids


def _label_name(label: Any) -> str:
    from OCP.TCollection import TCollection_AsciiString
    from OCP.TDataStd import TDataStd_Name

    attribute = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
        return TCollection_AsciiString(attribute.Get()).ToCString()
    return ""


def _flatten(
    shape_tool: Any,
    label: Any,
    prefix: str,
    out: list[tuple[str, Any]],
    location: Any = None,
) -> None:
    """Walk an XCAF assembly tree and emit ``(path, placed shape)`` leaves.

    ``location`` is the accumulated placement of the enclosing components.
    ``GetShape_s`` on a component label applies only that component's own
    transform, so a solid two levels down (KiCad places every footprint model
    as ``board/<refdes>/<model>``) would otherwise land at the model's origin.
    """

    from OCP.collections import Sequence_TDF_Label

    def placed(shape: Any) -> Any:
        return shape if location is None else shape.Moved(location)

    name = _label_name(label)
    if shape_tool.IsReference_s(label):
        from OCP.TDF import TDF_Label

        referred = TDF_Label()
        if not shape_tool.GetReferredShape_s(label, referred):
            out.append((f"{prefix}{name}", placed(shape_tool.GetShape_s(label))))
            return
        # A component label carries the placement; its name is the instance
        # name, else the referred prototype's name.
        name = name or _label_name(referred)
        if shape_tool.IsAssembly_s(referred):
            own = shape_tool.GetLocation_s(label)
            below = own if location is None else location.Multiplied(own)
            children = Sequence_TDF_Label()
            shape_tool.GetComponents_s(referred, children, False)
            joined = f"{prefix}{name}/" if name else prefix
            for index in range(1, children.Length() + 1):
                _flatten(shape_tool, children.Value(index), joined, out, below)
            return
        out.append((f"{prefix}{name}", placed(shape_tool.GetShape_s(label))))
        return
    if shape_tool.IsAssembly_s(label):
        children = Sequence_TDF_Label()
        shape_tool.GetComponents_s(label, children, False)
        joined = f"{prefix}{name}/" if name else prefix
        for index in range(1, children.Length() + 1):
            _flatten(shape_tool, children.Value(index), joined, out, location)
        return
    out.append((f"{prefix}{name}", placed(shape_tool.GetShape_s(label))))


def _model_from_named_shapes(named: Sequence[tuple[str, Any]], source: str) -> StepModel:
    solids: list[StepSolid] = []
    seen: dict[str, int] = {}
    for name, shape in named:
        pieces = _explode_solids(shape)
        for piece_index, piece in enumerate(pieces):
            base = name or f"solid{len(solids)}"
            label = base if len(pieces) == 1 else f"{base}#{piece_index}"
            count = seen.get(label, 0)
            seen[label] = count + 1
            if count:
                label = f"{label}@{count}"
            solids.append(_solid_from_shape(label, len(solids), piece))
    return StepModel(tuple(solids), source=source)


def load_step(path: str | Path) -> StepModel:
    """Read a STEP file and flatten its assembly into named solids."""

    _ocp()
    from OCP.collections import Sequence_TDF_Label
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(file)
    app = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(file)) != IFSelect_RetDone:
        raise ValueError(f"OpenCASCADE could not read {file}")
    if not reader.Transfer(document):
        raise ValueError(f"OpenCASCADE could not transfer {file}")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    labels = Sequence_TDF_Label()
    shape_tool.GetFreeShapes(labels)
    named: list[tuple[str, Any]] = []
    for index in range(1, labels.Length() + 1):
        _flatten(shape_tool, labels.Value(index), "", named)
    return _model_from_named_shapes(named, source=str(file))


def write_step(path: str | Path, solids: Sequence[StepSolid]) -> Path:
    """Write named solids to a STEP file (names survive as product names)."""

    _ocp()
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    app = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    for solid in solids:
        label = shape_tool.AddShape(solid._shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(solid.name))
    writer = STEPCAFControl_Writer()
    writer.Transfer(document, STEPControl_AsIs)
    file = Path(path)
    if writer.Write(str(file)) != IFSelect_RetDone:
        raise ValueError(f"OpenCASCADE could not write {file}")
    return file


# ------------------------------------------------------------ synthetic shapes
def box_solid(
    name: str,
    corner_m: tuple[float, float, float],
    size_m: tuple[float, float, float],
    *,
    index: int = 0,
) -> StepSolid:
    """An axis-aligned box from its minimum corner and size, in metres."""

    _ocp()
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    if any(value <= 0.0 for value in size_m):
        raise ValueError("box sizes must be positive")
    x, y, z = (value / MM for value in corner_m)
    dx, dy, dz = (value / MM for value in size_m)
    shape = BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()
    return _solid_from_shape(name, index, shape)


def cylinder_solid(
    name: str,
    centre_xy_m: tuple[float, float],
    z_m: float,
    height_m: float,
    radius_m: float,
    *,
    index: int = 0,
) -> StepSolid:
    """A vertical cylinder (a via barrel, a standoff), in metres."""

    _ocp()
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    if height_m <= 0.0 or radius_m <= 0.0:
        raise ValueError("cylinder height and radius must be positive")
    axis = gp_Ax2(gp_Pnt(centre_xy_m[0] / MM, centre_xy_m[1] / MM, z_m / MM), gp_Dir(0.0, 0.0, 1.0))
    shape = BRepPrimAPI_MakeCylinder(axis, radius_m / MM, height_m / MM).Shape()
    return _solid_from_shape(name, index, shape)


def synthetic_model(solids: Sequence[StepSolid], source: str = "synthetic") -> StepModel:
    """A model from synthetic solids, re-indexed in order."""

    return StepModel(
        tuple(
            StepSolid(solid.name, index, solid.bounds_m, solid.volume_m3, solid._shape)
            for index, solid in enumerate(solids)
        ),
        source=source,
    )


__all__ = [
    "StepModel",
    "StepSolid",
    "box_solid",
    "cylinder_solid",
    "load_step",
    "ocp_available",
    "synthetic_model",
    "write_step",
]
