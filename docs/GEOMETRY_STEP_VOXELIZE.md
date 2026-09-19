# STEP geometry front end: 2.5D board, 3D enclosure and heat sink

## Scope

`src/geometry/step_voxelize` reads mechanical CAD (STEP, via OpenCASCADE) and
turns it into the array inputs the solvers of this repository already accept.
It owns no field solve. Two decisions fix its shape:

1. **OpenCASCADE binding**: `OCP` (PyPI `cadquery-ocp`), the pybind11 build
   of OpenCASCADE that ships as a wheel and updates with `pip`. It is an
   optional extra (`pcb-analysis[cad]`); the core distribution stays on
   NumPy and SciPy. No OCC object crosses the package boundary: every public
   result is a NumPy array or a frozen dataclass of arrays.
2. **Two meshes, coupled at the interface**: the board is a 2.5D layered
   model on the electrical grid; the enclosure, heat sink and other 3D
   bodies are a separate masked voxel mesh. The two are joined by a
   staggered interface iteration in `multiphysics`. Neither solver knows
   about the other.

Rasterisation here means fixed-pitch sampling of B-rep solids. There is no
unstructured mesher and no CFD. Convection is a film coefficient on exposed
faces, radiation is not modelled.

## Input model

A STEP file arrives as an assembly tree of solids in millimetres. The front
end flattens the tree with `XCAFDoc` (labels, colours, product names) and
classifies each solid by a **body map** supplied by the caller:

```text
BodyMap
  board:      {solid: <name or label>, stackup: Stackup spec}
  copper:     [{solid, layer}]               # optional; see below
  components: [{solid, power_w or None, k}]  # bodies sitting on the board
  bodies:     [{solid, material, role}]      # heat sink, case, TIM, standoffs
  ignore:     [solid, ...]
```

Names match the STEP product/instance names; a regex is accepted. Solids not
covered by the map fail loudly rather than being guessed. Colour-based
classification is accepted only as a hint that the caller confirms in the
map, because EDA exporters do not agree on colours.

Copper is the weak point of mechanical STEP. Most exporters emit the board
outline and component bodies only; KiCad 8 can include tracks, pads, vias and
zones as solids when asked. The front end therefore accepts copper from three
places, in this precedence: the `plane-opt-current-field-problem/v1` mapping,
an occupancy array given directly, or copper solids in the STEP. Mixing is
allowed per layer. Without any copper, only the thermal path runs, with the
board as bare laminate.

## Board path (2.5D)

The board solid gives the outline and the total thickness; the stackup spec
gives layer names, centre `z` and thickness. For each layer the front end
takes the planar section of the copper solids at the layer's centre `z`
(`BRepAlgoAPI_Section` against a `gp_Pln`), closes the resulting wires into
faces, and rasterises them onto the routing grid:

- `pitch_mm` and the grid origin are caller inputs, fixed for the whole
  board; every solver of this repository assumes one in-plane grid.
- A cell is copper when the sampled fill exceeds a threshold (default 0.5).
  The fill is computed by `n × n` supersampling of point-in-face tests
  (`BRepClass_FaceClassifier`) with `n = 3` by default; the fill fraction is
  also kept as a float array for the thermal conductivity blend.
- Vias are solids whose section is a disc on two or more consecutive layer
  planes with the same centre; they become `ViaSpec` entries of a `ViaSet`.
- Terminals are named cells: either from the plane-opt mapping or from
  component pins the body map names.

Outputs are exactly the objects `electrical.dice_peec` and
`thermal.matrix_free_mpir_fem` consume today: `Stackup`, occupancy `x0` of
shape `(n_layers, ny, nx)`, `ViaSet`, and a `LayeredThermalMesh` whose slab
conductivities are the copper/laminate blend by fill fraction. The electrical
solvers are unchanged.

## Enclosure path (3D)

Bodies listed in `bodies` and `components` are voxelised on their own
Cartesian grid with an independent pitch `(hx, hy, hz)` and bounding box,
usually coarser than the board grid. For each voxel centre a
`BRepClass3d_SolidClassifier` test against each body assigns a material id;
`0` is void. Supersampling gives a fill fraction that scales the voxel
conductivity for partially filled voxels, which keeps thin fins from
disappearing when the pitch is coarse. Bodies are tested in map order so a
TIM between heat sink and component wins over the two it touches.

Output is a `VoxelSolidModel`:

```text
material_id      uint16 (nz, ny, nx)       0 = void
fill             float64 (nz, ny, nx)      0..1
materials        table id -> (k_w_per_m_k, name, role)
origin_m, pitch_m
exposed_faces    boolean (6, nz, ny, nx)   active voxel face touching void
```

The thermal package turns this into a `VoxelThermalMesh`: the same hexahedral
Q1 discretisation as `LayeredThermalMesh`, generalised with an **active
element mask** and with Newton cooling on any exposed face, not only the top
and bottom of a stack. The matrix-free action skips inactive elements; the
two-level preconditioner aggregates active nodes only. This is a mesh
generalisation of the existing method, so it lives in
`thermal.matrix_free_mpir_fem` and reuses its operator, kernels and the
electrical package's MPIR solver; it is not a new solver.

Inside the enclosure, air is void by default. Where the caller wants the
air gap to conduct, a body of low conductivity is added to the map; the
solver does not model buoyancy.

## Interface between the two meshes

The front end also emits the **contact map** between the board and the 3D
bodies (the dataclass lives in `thermal.matrix_free_mpir_fem.contact`, so
`multiphysics` needs no geometry import to use it). For every board element face on the top or bottom surface that lies
under an active voxel, it records

```text
ContactMap
  board_cells    (m, 3)   side, row, col on the board grid
  voxels         (m, 3)   k, j, i on the voxel grid
  area_m2        (m,)     overlap of the two footprints
  conductance    (m,)     W/K, from the TIM or contact spec of the body
```

Interface conductance comes from the body map (`tim: {k, thickness}` or a
direct `h_contact`); a hard contact uses a large finite value rather than
merging nodes, so both meshes stay independent.

Coupling is staggered and lives in `multiphysics.staggered_coupling` as a
`BoardEnclosureThermalScenario`:

1. Board solve with a Robin boundary on the contact cells: film coefficient
   `G / A` and ambient temperature equal to the current voxel temperature at
   the matching contact. This needs `ConvectionBoundary` to accept a
   per-element ambient temperature array; today it takes one scalar.
2. Enclosure solve with the resulting interface heat flux applied as voxel
   heat sources at the matching voxels, plus its own convection to ambient.
3. Repeat until the interface flux changes by less than the tolerance.
   Aitken relaxation on the interface temperature is available for stiff
   contacts.

The board's Joule loss enters step 1 through the existing
`element_joule_heat_w` mapping; the copper `σ(T)` loop of
`ElectroThermalScenario` wraps the whole interface iteration when both are
requested.

## Software boundary

| Module | Responsibility |
|---|---|
| `geometry/step_voxelize/reader.py` | STEP load, assembly flattening, body map resolution (only module importing `OCP`) |
| `geometry/step_voxelize/section.py` | planar sections and 2D rasterisation to the routing grid |
| `geometry/step_voxelize/voxelize.py` | 3D classification, fill fraction, exposed faces |
| `geometry/step_voxelize/contact.py` | board/voxel contact placement (origins, TIM spec) feeding `thermal.matrix_free_mpir_fem.planar_contact_map` |
| `geometry/step_voxelize/adapters.py` | build `Stackup`, occupancy, `ViaSet`, `LayeredThermalMesh`, plane-opt mapping |
| `thermal/matrix_free_mpir_fem/voxel.py`, `conduction.py` | `VoxelThermalMesh`, active-element mask, exposed-face convection |
| `thermal/matrix_free_mpir_fem/contact.py` | `ContactMap`, `planar_contact_map` |
| `multiphysics/staggered_coupling/board_enclosure.py` | interface iteration |

Import direction: `electrical ← geometry`, `thermal ← geometry`,
`geometry ← multiphysics`. `geometry` imports nothing from `emc` or
`multiphysics`, and `electrical` keeps importing nothing from anyone. `OCP`
is imported lazily inside `reader.py`; every other module, and every test
that does not need a STEP file, runs on NumPy alone. Tests that need `OCP`
are skipped where it is missing, and a skipped test is not recorded as a
pass: CI installs the `cad` extra in one job so those tests run.

Packaging: `cad = ["cadquery-ocp>=7.7"]` in `[project.optional-dependencies]`,
a `geometry` top-level package with `py.typed`, registered in
`[tool.setuptools.package-data]`, and an import check in the distribution
job.

## Acceptance

Adoption follows the repository rule: the same fixture, tolerance and device
for the current path and the candidate, with the numbers written to
`docs/*_RESULTS.json` carrying `decision` and `environment`.

1. **Rasteriser**: rectangles and discs built in OCC, rasterised at several
   pitches; copper area error and via centre error against the analytic
   values, as a function of pitch and supersampling.
2. **Voxel mesh against the layered mesh**: one solid block modelled both as
   `LayeredThermalMesh` slabs and as `VoxelThermalMesh` on the same grid.
   Temperatures must agree to the solver tolerance; record inner/outer
   iterations, residual and time on both paths.
3. **Fin**: a single rectangular fin with convection on its faces against the
   1D fin solution; error against pitch.
4. **Interface iteration**: board plus block heat sink solved (a) as one
   monolithic layered mesh with the block as extra slabs and (b) as two
   meshes through the contact map. (b) must converge to (a) within the
   contact tolerance; record the number of interface iterations, the total
   solve time, and the interface flux history.
5. **STEP round trip**: a small KiCad export with copper enabled, compared
   with the same board's `.kicad_pcb`-derived occupancy from the plane-opt
   mapping.

## Increments

Each step is one branch and passes the full test suite before the next.

1. `docs/`: this document, the pointers in the thermal and multiphysics
   documents (this branch).
2. `feature/thermal-voxel-mesh`: `VoxelThermalMesh`, active mask in the
   operator and preconditioner, exposed-face convection, per-element ambient
   in `ConvectionBoundary`; acceptance items 2 and 3.
3. `feature/board-enclosure-coupling`: contact map dataclass and the
   interface iteration; acceptance item 4 on synthetic arrays, no STEP yet.
4. `feature/geometry-step-voxelize`: the `geometry` package, `cad` extra,
   packaging and CI; acceptance items 1 and 5.
5. CUDA kernels for the masked action once the array path is adopted, on
   the GPU shell with the device and CuPy version recorded.

## Not in scope

CFD and buoyant air, radiation, transient conduction, temperature-dependent
conductivity, unstructured meshes, and automatic body classification without
a body map.
