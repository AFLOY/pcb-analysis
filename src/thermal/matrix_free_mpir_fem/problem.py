"""The steady conduction problem: mesh, boundaries, fixed nodes and loads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundaries import Convection, ConvectionBoundary, ExposedFaceConvection, HeatSource
from .mesh import LayeredThermalMesh, _flat_index


@dataclass(frozen=True)
class ThermalConductionProblem:
    """Steady conduction with convective faces and fixed-temperature nodes.

    ``element_heat_w`` is the total power released in every element; it is
    lumped equally onto the element's eight corner nodes.  ``nodal_heat_w``
    is power already lumped onto nodes, one value per node (the heat a
    contact hands over through an interface).  At least one positive film
    coefficient or one fixed node is required, otherwise the temperature
    level is undetermined.
    """

    mesh: LayeredThermalMesh
    convection: tuple[Convection, ...] = ()
    fixed_temperature_mask: np.ndarray | None = None
    fixed_temperature_k: float | np.ndarray | None = None
    heat_sources: tuple[HeatSource, ...] = ()
    element_heat_w: np.ndarray | None = None
    nodal_heat_w: np.ndarray | None = None

    def __post_init__(self) -> None:
        node_shape = self.mesh.node_shape
        element_shape = self.mesh.element_grid_shape

        convection = tuple(self.convection)
        for boundary in convection:
            if not isinstance(boundary, (ConvectionBoundary, ExposedFaceConvection)):
                raise TypeError(
                    "convection entries must be ConvectionBoundary or "
                    "ExposedFaceConvection"
                )
            boundary.check_shape(self.mesh)
        has_convection = any(boundary.cools for boundary in convection)

        if self.fixed_temperature_mask is None:
            mask = np.zeros(node_shape, dtype=bool)
            values = np.zeros(node_shape, dtype=np.float64)
            if self.fixed_temperature_k is not None:
                raise ValueError(
                    "fixed_temperature_k needs a fixed_temperature_mask"
                )
        else:
            mask = np.asarray(self.fixed_temperature_mask, dtype=bool)
            if mask.shape != node_shape:
                raise ValueError("fixed_temperature_mask must match mesh.node_shape")
            if self.fixed_temperature_k is None:
                raise ValueError("fixed_temperature_mask needs fixed_temperature_k")
            values = np.asarray(self.fixed_temperature_k, dtype=np.float64)
            if values.ndim == 0:
                values = np.full(node_shape, float(values), dtype=np.float64)
            elif values.shape != node_shape:
                raise ValueError(
                    "fixed_temperature_k must be a scalar or match mesh.node_shape"
                )
            if not np.all(np.isfinite(values[mask])):
                raise ValueError("fixed temperatures must be finite")
            values = np.where(mask, values, 0.0)
        active_nodes = self.mesh.active_nodes
        if not has_convection and not np.any(mask & active_nodes):
            raise ValueError(
                "the problem needs a positive film coefficient or a fixed node"
            )

        if self.element_heat_w is None:
            element_heat = np.zeros(element_shape, dtype=np.float64)
        else:
            element_heat = np.asarray(self.element_heat_w, dtype=np.float64)
            if element_heat.shape != element_shape:
                raise ValueError("element_heat_w must match (slabs, rows, cols)")
            if not np.all(np.isfinite(element_heat)):
                raise ValueError("element heat must be finite")
            if np.any(element_heat[~self.mesh.active] != 0.0):  # type: ignore[index]
                raise ValueError("element heat must be zero on inactive elements")

        if self.nodal_heat_w is None:
            nodal_heat = np.zeros(node_shape, dtype=np.float64)
        else:
            nodal_heat = np.asarray(self.nodal_heat_w, dtype=np.float64)
            if nodal_heat.shape != node_shape:
                raise ValueError("nodal_heat_w must match mesh.node_shape")
            if not np.all(np.isfinite(nodal_heat)):
                raise ValueError("nodal heat must be finite")
            if np.any(nodal_heat[~active_nodes] != 0.0):
                raise ValueError("nodal heat must be zero on inactive nodes")

        sources = tuple(self.heat_sources)
        for source in sources:
            for node in source.nodes:
                if not active_nodes.flat[_flat_index(node, node_shape)]:
                    raise ValueError(
                        f"heat source {source.name!r} touches inactive node {node!r}"
                    )

        object.__setattr__(self, "convection", convection)
        object.__setattr__(self, "fixed_temperature_mask", mask.copy())
        object.__setattr__(self, "fixed_temperature_k", values.copy())
        object.__setattr__(self, "heat_sources", sources)
        object.__setattr__(self, "element_heat_w", element_heat.copy())
        object.__setattr__(self, "nodal_heat_w", nodal_heat.copy())
