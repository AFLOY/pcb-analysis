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

Rasterisation here means fixed-pitch sampling of B-rep solids: every cell or
voxel is covered by `n × n` (or `n³`) sample points classified against the
solids (see "Point classification"), and the inside fraction is the
fill. The 2.5D layers are sampled at their centre `z` the same way rather
than through section faces, so one code path serves both parts. Bounds come
from `BRepBndLib::AddOptimal` without the shape tolerance, otherwise a 20 mm
board reports 20.0000002 mm and the grid gains a cell. There is no
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

## Point classification

Three paths answer "is this point inside this solid", selectable per call
(`method=`) or per process (`PCB_GEOMETRY_CLASSIFY`):

- `occ`: `BRepClass3d_SolidClassifier` per point after a bounding-box
  prefilter. Exact on the B-rep; one Python call per point.
- `numpy` and `native`: the solid is tessellated once (`BRepMesh`, 5 µm
  linear deflection, cached on the `StepSolid`) into outward-oriented
  triangles, and the point's generalized winding number, the sum of the
  signed solid angles of the triangles over `4π`, is 1 inside and 0 outside.
  Unlike ray parity it has no degenerate ray/edge cases. `numpy` evaluates
  it in chunks; `native` is the same sum in C++ (`native/point_in_mesh.cpp`,
  OpenMP over points, module `_voxelize_native`), built by the root
  `CMakeLists.txt` with the other kernels. `auto` picks `native` when built,
  else `numpy`.

Planar solids tessellate exactly, so the two winding-number paths reproduce
`occ` cell for cell on boxes; a cylinder deviates by the deflection (a 0.3 mm
via at 5 µm carries 248 triangles and 0.17 % less volume). `StepSolid.tessellate`
rejects a mesh with inward orientation and `TriangleMesh.is_closed` checks
watertightness. Measured by `experiments/geometry_classify_benchmark.py`
(Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz, 12 logical CPUs, OCP 8.0.1.0.0, NumPy 2.3.5,
g++ (GCC) 14.3.1 20251022 (Red Hat 14.3.1-4)); the adopted run is `GEOMETRY_CLASSIFY_RESULTS.json`:

| Case | Path | Time (ms) | Speed-up vs occ | Max fill difference vs occ |
|---|---|---|---|---|
| copper layer at 0.25 mm, 3 samples per axis | occ | 4294.6 | — | — |
| copper layer at 0.25 mm, 3 samples per axis | numpy | 419.7 | 10× | 0.000 |
| copper layer at 0.25 mm, 3 samples per axis | native_threads1 | 55.7 | 77× | 0.000 |
| copper layer at 0.25 mm, 3 samples per axis | native_threads4 | 16.8 | 256× | 0.000 |
| copper layer at 0.25 mm, 3 samples per axis | native_threads6 | 12.0 | 359× | 0.000 |
| heat sink voxels (0.5, 0.5, 1.0) mm, 2 samples per axis | occ | 2062.0 | — | — |
| heat sink voxels (0.5, 0.5, 1.0) mm, 2 samples per axis | numpy | 151.6 | 14× | 0.000 |
| heat sink voxels (0.5, 0.5, 1.0) mm, 2 samples per axis | native_threads1 | 26.5 | 78× | 0.000 |
| heat sink voxels (0.5, 0.5, 1.0) mm, 2 samples per axis | native_threads4 | 8.1 | 254× | 0.000 |
| heat sink voxels (0.5, 0.5, 1.0) mm, 2 samples per axis | native_threads6 | 6.1 | 340× | 0.000 |

Decision: adopted (`native` is the default when built); the `occ` path stays
as the exact reference for curved solids and for tests.

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
| `geometry/step_voxelize/reader.py` | STEP load through `STEPCAFControl`, assembly flattening into named solids, point-in-solid tests, synthetic boxes and cylinders, `write_step` (the only module importing `OCP`) |
| `geometry/step_voxelize/bodymap.py` | `BodyMap` (board stackup, copper, vias, bodies, ignore) and its exhaustive resolution against the model |
| `geometry/step_voxelize/mesh.py` | `TriangleMesh`, NumPy winding number, path selection |
| `geometry/step_voxelize/native/point_in_mesh.cpp` | C++ winding number over points (OpenMP), module `_voxelize_native` |
| `geometry/step_voxelize/section.py` | per-layer sampling of the board outline and copper onto the routing grid (`BoardRaster`) |
| `geometry/step_voxelize/voxelize.py` | 3D sampling of bodies onto a voxel grid, fill fraction, material precedence (`VoxelSolidModel`) |
| `geometry/step_voxelize/contact.py` | board/voxel contact placement (origins, contact spec) feeding `thermal.matrix_free_mpir_fem.planar_contact_map` |
| `geometry/step_voxelize/adapters.py` | build `Stackup`, occupancy, `ViaSet`, the board's `LayeredThermalMesh`, body meshes and heat sources, plane-opt mapping |
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

## Status

| Increment | Branch | State |
|---|---|---|
| this document and the pointers in the thermal and multiphysics documents | `docs/step-geometry-3d-thermal` | done |
| active-element mask, `VoxelThermalMesh`, exposed-face convection, per-face ambient (acceptance 2, 3) | `feature/thermal-voxel-mesh` | done; `tests/test_thermal_voxel.py`, array, C++ and CUDA paths |
| `ContactMap`, `nodal_heat_w`, `BoardEnclosureThermalScenario` (acceptance 4) | `feature/board-enclosure-coupling` | done; `tests/test_board_enclosure_coupling.py`, numbers in `MULTIPHYSICS_SCENARIOS.md` and `BOARD_ENCLOSURE_ACCEPTANCE_RESULTS.json` |
| `geometry.step_voxelize`, `cad` extra, packaging and CI (acceptance 1, 5 on a synthetic STEP) | `feature/geometry-step-voxelize` | done; `tests/test_geometry_step.py` |
| acceptance 5 on a KiCad export with copper enabled | — | not started; needs a real export as fixture |
| electro-thermal `σ(T)` loop around the interface iteration | — | not started |
| tessellation and C++ winding-number classification | `feature/geometry-native-classify` | done; `tests/test_native_geometry.py`, `GEOMETRY_CLASSIFY_RESULTS.json` |

Measured on the synthetic fixture of `tests/test_geometry_step.py` (a 20 × 12 ×
1.6 mm board, copper on both faces, one via, a 6 × 6 × 4 mm sink, one
component), OCP 8.0.1, this host: rectangles aligned with the grid rasterise
to their exact area at 1.0 and 0.5 mm; at 0.4 mm with three samples per axis
the area is 5 % high because a boundary sample counts as inside and a
half-covered cell quantises to 2/3. A 0.3 mm via disc sampled at 0.1 mm is
13 % off with one sample per cell and 1 % with two or four. The sink
voxelises to its exact volume at (0.5, 0.5, 1.0) mm, its 144-pair contact
map covers 36 mm², and the coupled board/sink solve converges in 6 interface
iterations. These are checks that the pipeline is wired correctly; the
adoption benchmark against a real export is still to be measured.

## Not in scope

CFD and buoyant air, radiation, transient conduction, temperature-dependent
conductivity, unstructured meshes, and automatic body classification without
a body map.
