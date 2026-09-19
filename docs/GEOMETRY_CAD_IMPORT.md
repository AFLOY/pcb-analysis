# STEP geometry front end (`geometry.cad_import`): 2.5D board, 3D enclosure and heat sink

## Scope

`src/geometry/cad_import` reads mechanical CAD (STEP, via OpenCASCADE) and
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
outline and component bodies only; KiCad 8 and later include tracks, pads,
vias and zones as solids when asked (see "KiCad exports"). The front end therefore accepts copper from three
places, in this precedence: the `plane-opt-current-field-problem/v1` mapping,
an occupancy array given directly, or copper solids in the STEP. Mixing is
allowed per layer. Without any copper, only the thermal path runs, with the
board as bare laminate.

## Layer sections (2.5D)

A conductor layer is the section of the copper solids at the layer's centre
`z`. With the native extension that section is rasterised exactly
(`method="section"`, the default for `sample_plane_fill` and
`rasterize_board` when built): every triangle of a solid's tessellation that
crosses the plane yields one oriented segment (direction `ẑ × n` for the
outward normal `n`, so the loops run counter-clockwise around copper and
need no stitching), and the segments are accumulated onto the cell grid with
the signed-area scheme of font rasterisers (each segment adds area and cover
terms, a prefix sum along `x` gives the fraction of every cell inside the
loops). The result is the exact covered area per cell, so `supersample` does
not apply and the quantisation of a 0.25 mm track on 0.25 mm cells
disappears; solids of one layer touch but do not overlap, so their coverages
add and are clamped to one. Cost is linear in the crossing triangles and the
cells they touch, independent of a fused zone's bounding box, which is what
made the point path slow on a real board (see `KICAD_STEP_RESULTS.json`).
Components, heat sinks and enclosures stay on the 3D voxel path below.

## Layer thickness and skin effect

The thickness a layer is given matters more than its section: the sheet PEEC
cuts each layer's thickness into graded filaments to resolve the skin effect,
so the 2.5D model is only as right as the thickness it receives. Two checks
guard it:

- `rasterize_board` measures each layer's copper thickness from the copper
  solids (volume-weighted mean of their z extents, barrels excluded) and
  warns when it differs from the stackup by more than `thickness_tolerance`
  (5 %); `BoardRaster.thickness_mismatches` lists the layers, and
  `thickness_source="measured"` on `plane_opt_problem_mapping` and
  `board_thermal_mesh` uses the solids' value instead of the stackup's.
- `skin_report(raster, frequency_hz)` classifies each layer by thickness over
  skin depth at the analysis frequency: `uniform` below one skin depth,
  `filaments` where the solver's graded filaments resolve the profile, and
  `3d` above twenty skin depths or above 1 mm, where a sheet cannot represent
  the conductor at all. `plane_opt_problem_mapping` runs the report when a
  frequency is given and warns on `3d` layers; such conductors (busbars,
  terminal blocks) belong on the voxel path with a 3D PEEC solve. Splitting
  a thick layer into two stackup sheets was considered and not adopted: the
  contract's vertical connections are per-cell lumped segments, two equal
  sheets resolve the profile only up to about two skin depths, and the
  solver's filaments already do the same job with a frequency-dependent
  count and grading.

## Thick conductors: the 3D path

A layer or body that `skin_report` classifies as `3d` leaves the stackup and
goes to the voxel PEEC: `conductor_problem_from_solids` voxelises its solids,
marks the terminal voxels from solids or boxes, and `electrical.voxel_peec.solve_voxel_peec`
solves it with PyPEEC (CPU or CUDA); `conductor_heat_w` returns the Joule
loss on the same voxel grid for the thermal solve. The contract, the
acceptance on a busbar and the pitch-versus-skin-depth caveat are in
`VOXEL_PEEC.md`.

## Point classification

Three paths answer "is this point inside this solid", selectable per call
(`method=`) or per process (`PCB_GEOMETRY_CLASSIFY`); they serve the 3D
voxelisation and remain available for layers (`"occ"`, `"numpy"`, `"native"`):

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

## KiCad exports

`kicad-cli pcb export step --include-tracks --include-pads --include-zones
--include-inner-copper --no-extra-pad-thickness` (KiCad 8 and later) writes
copper as solids, but names every solid after its assembly node
(`board 1/=>[0:1:1:3]`), so a body map by name cannot sort them. In a KiCad
export z is unambiguous instead: the board body spans the laminate from the
top of the bottom copper (`z = 0`) to the bottom of the top copper, each
copper layer is its own sheet (`B.Cu` at `[-t, 0]`, inner layers at the
dielectric boundaries, `F.Cu` on top), and a via or through-hole barrel runs
from inside the lowest layer to inside the highest. `geometry.cad_import.kicad`
therefore

- runs the export (`export_kicad_step`), reads the `(stackup ...)` section of
  the `.kicad_pcb` (`read_kicad_stackup`, `layers_from_kicad_stackup`) and
  builds a body map whose selectors are z windows: the body must *span* the
  laminate (`z_span_mm`), copper must lie *within* its layer's window
  (`z_range_mm`), a barrel must span from the lowest to the highest layer
  centre; every selector is available on the ordinary `BodyMap` entries too;
- maps a KiCad y-down grid onto the STEP frame, where KiCad negates y
  (`kicad_grid_origin_mm`, `rasterize_board(..., shape=, y_down=True)`). The
  origin is `(min_x, -(min_y + rows·pitch))`: a board height that is not a
  multiple of the pitch would otherwise shift every row by the remainder.

Through-hole pads come out physically: an annulus on each outer layer, a
plating barrel through the hole, and the hole itself open. Pads with the
extra thickness KiCad adds by default would sit in a different z window, hence
the `--no-extra-pad-thickness` flag.

### Component models

With components enabled (the default) KiCad places each footprint's 3D model
as an assembly component *named by the reference designator* at the footprint
origin, so after flattening a part's solids are `board/<refdes>/<model path>`
(`power_module 1/C7/=>[0:1:1:3]`) while copper and the laminate stay direct
children (`board/=>[...]`). `kicad_component_solids(model)` groups the solids
by that second path segment, `read_kicad_footprints(board)` reads every
footprint's reference, footprint name, position, rotation, layer and
`(property ...)` fields from the `.kicad_pcb`, and `KicadFootprint.step_xy_mm`
gives the origin in the STEP frame (y negated). Custom footprint fields are
where a design carries per-part data that the STEP does not, such as a
thermal network model and its resistances; AP242 user-defined properties
(`STEPCAFControl_Writer.SetMetadataMode`, read back as `TDataStd_NamedData`)
round-trip through OCP 8.0.1 and are the planned carrier once the parameters
live in the STEP itself.

`kicad-cli` does not read the GUI path table, so library models referenced as
`${KICAD10_3DMODEL_DIR}/...` are silently dropped ("Could not add 3D model for
C1") unless the variable is passed: `export_kicad_step(..., model_dir=...)`
defines it for the installed major version (`kicad_model_dir_variable`,
`default_kicad_model_dir`). Placement needs the flattener to accumulate the
transforms of nested components; before that fix every part landed at its
model origin. Verified with KiCad 10.0.5 on `power_module`: the eleven 0603
parts whose library model was present centre within 0.05 mm of their
footprint origin and stand on the board (`tests/test_kicad_step.py`, skipped
without the library models; `tests/test_geometry_step.py` covers the nested
placement with a synthetic assembly).

### Graded grids from the components

`rasterize_board(..., grid=)` takes a `TensorGrid`
(`electrical.matrix_free_mpir_fem.grid`) instead of a pitch: the section
rasteriser has a graded-grid entry (`plane_section_coverage_graded`) that
splits every section segment at the grid lines it crosses and maps each piece
onto cell index space, where the map is affine, so the per-cell area fractions
stay exact (a box and a cylinder come out to `1e-15`, uniform lines reproduce
the uniform path bitwise). `BoardRaster.grid` holds the lines; `pitch_x_m` /
`pitch_y_m` / `cell_area_m2` follow the array row order (reversed with
`y_down`), `pitch_mm` is `None` on a graded grid, and `board_thermal_mesh`
hands the per-cell pitches to the thermal mesh.

`plane_opt_problem_mapping` writes schema `v2` (grid lines in `x_edges_mm` /
`y_edges_mm`, top-down with `y_down`) for a graded raster and `v1` for a
uniform one; the sheet PEEC solves the former with its pFFT operator
(`SHEET_PEEC.md`).

`geometry.cad_import.refinement` makes the grid from the parts:
`component_boxes(kicad_component_solids(model), min_size_m=)` is the in-plane
box of every component's solids, `board_refined_grid(board, coarse_pitch_mm=,
fine_pitch_mm=, boxes=, margin_mm=, growth=)` is the graded grid over the
board's bounding box, fine over the boxes widened by the margin and growing
geometrically to the coarse pitch, and `refinement_summary` reports the cell
counts against the uniform alternatives. Refining a box refines its whole row
and column strips (tensor grid).

KiCad writes pads, tracks and barrels as separate, overlapping solids; the
rasteriser sums overlapping coverages per cell and clamps at one, so the
union area then depends slightly (about `1e-3`) on the cell size. Export with
`export_kicad_step(..., fuse_shapes=True)` (`--fuse-shapes`) to get united
copper per layer, whose exact coverage is the same on every grid.

Measured on `power_module` with fused copper and the eleven 0603 parts that
have library models (`KICAD_REFINEMENT_RESULTS.json`,
`experiments/kicad_refinement_acceptance.py`, Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz, KiCad
10.0.5): 1 W in C1's pads (its box overlap weighted by copper
fill), `h = 10 W/m²K` both faces, rise 104.5 K on the uniform 0.1 mm grid;
the steep grading has 32200 cells against 121104 uniform fine and
4761 uniform coarse.

| Grid | Cells | Thermal nodes | Raster (ms) | Solve (ms, C++) | Inner iterations | Peak error vs 0.1 mm (K) | Copper area rel. diff |
|---|---|---|---|---|---|---|---|
| uniform 0.5 mm | 4900 | 20164 | 205 | 946 | 975 | -9.04 | 5.5e-14 |
| uniform 0.1 mm | 120409 | 484416 | 7 | 9835 | 1200 | +0.00 | 0.0e+00 |
| graded 0.1 mm under components, margin 1.0 mm, growth 1.4 | 32200 | 130248 | 7 | 4233 | 2400 | -2.80 | 5.7e-14 |
| graded 0.1 mm under components, margin 2.0 mm, growth 1.20 | 43621 | 176176 | 7 | 4448 | 1600 | -2.89 | 5.7e-14 |

Decision: adopted. The graded grids cut the peak error to a third of the
coarse grid's with a quarter of the fine grid's cells and 2.3× less solve
time; the copper areas are exact on every grid. The remaining 2.8 K is not
the grading (the gentler one does not remove it) but the blended-fill
homogenisation of 0.25 mm traces in 0.5 mm cells away from the part, which
the uniform coarse grid shares; a smaller coarse pitch or refinement boxes
along the hot traces are the knobs. The two-level preconditioner needs 1.3 to
2× more inner iterations on the graded grids than on the fine uniform one.

### Acceptance 5 against plane_opt

`experiments/kicad_step_acceptance.py` exports the three boards of the
`plane_opt_refactor` checkout (`power_module` and `bldc_driver`, two layers;
`drone`, four layers), rasterises each layer onto plane_opt's own grid at
0.25 mm and compares with plane_opt's `CopperGrid` built from the
`.kicad_pcb` with every net in one role. plane_opt samples the cell centre
against the primitives with an inclusive edge rule and paints drill holes as
copper, so the cells are split three ways: interior (no differing
4-neighbour in either mask), drill hole (within a drill radius plus half a
cell diagonal), boundary (the rest). The adopted run is
`KICAD_STEP_RESULTS.json`:

| Board | Layers | Solids / triangles | Grid | Export / load / tessellate / raster (s) | Layer | Interior step-only | Interior plane_opt-only (% of copper) | Hole cells step / plane_opt only | Boundary disagreement |
|---|---|---|---|---|---|---|---|---|---|
| `power_module` | 2 | 82 / 36656 | 139×139 @ 0.25 mm | 0.5 / 1.1 / 0.5 / 0.0 | B.Cu | 0 | 0 (0.00) | 0 / 104 | 48 / 796 |
| | | | | | F.Cu | 0 | 0 (0.00) | 0 / 99 | 49 / 2234 |
| `bldc_driver` | 2 | 171 / 49340 | 268×189 @ 0.25 mm | 0.7 / 0.8 / 0.7 / 0.1 | B.Cu | 0 | 0 (0.00) | 1 / 291 | 332 / 2013 |
| | | | | | F.Cu | 0 | 0 (0.00) | 0 / 291 | 63 / 4779 |
| `drone` | 4 | 6198 / 3936108 | 441×389 @ 0.25 mm | 21.0 / 41.7 / 49.9 / 1.3 | B.Cu | 0 | 0 (0.00) | 11 / 2393 | 250 / 25201 |
| | | | | | In2.Cu | 0 | 0 (0.00) | 86 / 1601 | 146 / 9965 |
| | | | | | In1.Cu | 0 | 0 (0.00) | 70 / 1785 | 188 / 14018 |
| | | | | | F.Cu | 0 | 169 (0.23) | 64 / 2240 | 2083 / 39787 |
Measured with KiCad 10.0.5, OCP 8.0.1.0.0, the exact section rasteriser (`method="section"`). The point path measured before it (winding number at one sample per cell, 6 threads) took 165.5 s for the four `drone` layers against 1.3 s here, with the same interior agreement; it is kept for the 3D bodies and as a cross-check, not for layers. Decision: adopted; the largest interior plane_opt-only share is 0.23 % and the largest boundary disagreement 16 % of boundary cells.

Interior cells agree exactly on every layer of every board except 0.23 % of
`drone`'s F.Cu, where plane_opt paints zone clearance cut-outs and an
off-board track stub. Drill-hole cells differ in both directions: plane_opt
paints drills and non-plated holes as copper, and it omits the inner-layer
annular rings of through vias that the export carries.
Boundary disagreements are largest where a 0.25 mm track runs exactly along
a cell edge: plane_opt's `distance <= radius` rule paints both rows, the
sampler one, and neither is wrong at that resolution. `tests/test_kicad_step.py`
checks the same on `power_module` and is skipped where `kicad-cli`, `OCP` or
the checkout is missing.

## Board path (2.5D)

The board solid gives the outline and the total thickness; the stackup spec
gives layer names, centre `z` and thickness. For each layer the front end
takes the planar section of the copper solids at the layer's centre `z`
(`BRepAlgoAPI_Section` against a `gp_Pln`), closes the resulting wires into
faces, and rasterises them onto the routing grid:

- `pitch_mm` and the grid origin are caller inputs, fixed for the whole
  board; every solver of this repository assumes one in-plane grid.
- A cell is copper when its fill exceeds a threshold (default 0.5). The
  fill is the exact covered area on the section path, or the `n × n`
  supersampled inside fraction on the point paths (`n = 3` by default); it
  is also kept as a float array for the thermal conductivity blend.
- Vias are solids whose section is a disc on two or more consecutive layer
  planes with the same centre; they become `ViaSpec` entries of a `ViaSet`.
- Terminals are named cells: either from the plane-opt mapping or from
  component pins the body map names.

Outputs are exactly the objects `electrical.dice_peec`, `electrical.sheet_peec` and
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
| `geometry/cad_import/reader.py` | STEP load through `STEPCAFControl`, assembly flattening into named solids, point-in-solid tests, synthetic boxes and cylinders, `write_step` (the only module importing `OCP`) |
| `geometry/cad_import/bodymap.py` | `BodyMap` (board stackup, copper, vias, bodies, ignore) and its exhaustive resolution against the model |
| `geometry/cad_import/kicad.py` | `kicad-cli` STEP export, stackup and footprint reading, refdes to solid binding, z-window body map, y-down grid origin |
| `geometry/cad_import/mesh.py` | `TriangleMesh`, NumPy winding number, path selection, uniform and graded plane-section coverage wrappers |
| `geometry/cad_import/skin.py` | thickness over skin depth per layer, `uniform` / `filaments` / `3d` |
| `geometry/cad_import/native/point_in_mesh.cpp` | C++ winding number over points (OpenMP) and the exact plane-section coverage rasteriser on uniform and graded grids, module `_voxelize_native` |
| `geometry/cad_import/section.py` | per-layer sampling of the board outline and copper onto the routing grid, uniform or graded (`BoardRaster` with its `TensorGrid`) |
| `geometry/cad_import/refinement.py` | refinement boxes from component solids, graded board grid, cell-count summary |
| `geometry/cad_import/voxelize.py` | 3D sampling of bodies onto a voxel grid, fill fraction, material precedence (`VoxelSolidModel`) |
| `geometry/cad_import/contact.py` | board/voxel contact placement (origins, contact spec) feeding `thermal.matrix_free_mpir_fem.planar_contact_map` |
| `geometry/cad_import/conductors.py` | thick conductor solids to `VoxelConductorProblem`, terminal regions, Joule loss to the thermal grid |
| `geometry/cad_import/adapters.py` | build `Stackup`, occupancy, `ViaSet`, the board's `LayeredThermalMesh`, body meshes and heat sources, plane-opt mapping |
| `thermal/matrix_free_mpir_fem/voxel.py`, `mesh.py`, `boundaries.py` | `VoxelThermalMesh`, active-element mask, exposed-face convection |
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
| `geometry.cad_import`, `cad` extra, packaging and CI (acceptance 1, 5 on a synthetic STEP) | `feature/geometry-step-voxelize` | done; `tests/test_geometry_step.py` |
| acceptance 5 on KiCad exports with copper (`power_module`, `bldc_driver`, `drone`) | `feature/kicad-step-acceptance` | done; `tests/test_kicad_step.py`, `KICAD_STEP_RESULTS.json` |
| thick conductors to the 3D voxel PEEC (PyPEEC) with Joule loss to the voxel thermal mesh | `feature/voxel-peec-3d` | done; `tests/test_voxel_peec.py`, `VOXEL_PEEC_RESULTS.json` |
| electro-thermal `σ(T)` loop around the interface iteration | `feature/enclosure-electrothermal-radiation` | done; `tests/test_electro_thermal_enclosure.py`, `ELECTROTHERMAL_ENCLOSURE_RESULTS.json` |
| graded tensor grids in the rasteriser, component-driven refinement | `feature/tensor-grid-geometry` | done; `tests/test_geometry_step.py`, `tests/test_kicad_step.py`, `KICAD_REFINEMENT_RESULTS.json` |
| sheet PEEC on graded grids (pFFT) and the plane-opt grid contract v2 | `feature/sheet-peec-pfft` | done; `tests/test_sheet_pfft.py`, `SHEET_PFFT_RESULTS.json`; `plane_opt_problem_mapping` emits v2 grid lines for graded rasters |
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
