# femtic_mod_plot.py

Read and plot axis-parallel slice panels of a FEMTIC resistivity model,
with optional site-position overlay from `observe.dat`.

---

## Purpose

`femtic_mod_plot.py` reads a FEMTIC `resistivity_block_iterX.dat` together
with its `mesh.dat` and renders a configurable set of horizontal map slices,
N-S curtains, E-W curtains, and arbitrary strike/dip planes as a multi-panel
Matplotlib figure.  No model modification is performed — the script is a
pure visualisation tool.

Typical use cases:

- inspecting a starting model or inversion result without running `femtic_mod_edit.py`,
- checking the position of a specific MT site relative to model structure,
- producing publication figures with UTM-labelled axes directly from the
  FEMTIC model files.

---

## Workflow

```
UTM_ORIGIN_LAT / LON  →  UTM zone (auto)
                                          [optional: ESTIMATE_ORIGIN = True]
        CALIBRATION_SITES  +  observe.dat  →  estimate_utm_origin()
                                               UTM_ORIGIN_E / N  (printed)

resistivity_block_iterX.dat  +  mesh.dat
        |
        v  fviz.read_femtic_mesh / read_resistivity_block
   nodes, conn, rho (per element)
        |
        v  resolve_slices(PLOT_SLICES)       [optional CRS conversion]
   slice positions in model-local metres
        |                                    [optional]
        v  read_site_position(OBSERVE_FILE)
   (x_m, y_m) site overlay
        |
        v  plot_model_slices(...)            [exact tet-plane intersection]
figure file / interactive window
        |                                    [optional, PLOT3D = True]
        v  fviz.plot_model_3d(...)           [PyVista — requires pyvista]
interactive HTML / static screenshot
        |                                    [optional, PLOT_ENS = True]
        v  plot_ensemble_slices(...)         [n_members × n_slices figure]
ensemble PDF / per-member PDFs
        |                                    [optional, PLOT_BOREHOLE = True]
        v  plot_borehole_logs(...)           [1-D ρ(z) traces — point-in-element]
borehole PDF / interactive window
```

---

## Configuration

All user-editable settings live in the **Configuration** block near the top
of the script.  No command-line arguments are used; edit the script directly.

### Paths

| Variable | Default | Description |
|---|---|---|
| `WORK_DIR` | `/home/vrath/Py4MTX/work/` | Working directory |
| `MODEL_FILE` | `resistivity_block_iter0.dat` | Resistivity block to display (any iteration) |
| `MESH_FILE` | `mesh.dat` | Mesh file — always required |
| `OBSERVE_FILE` | `observe.dat` | Site data file — required only when `SITE_NUMBER` is set |

### Ocean / air handling

| Variable | Default | Description |
|---|---|---|
| `OCEAN` | `None` | `None` = auto-infer; `True` / `False` = force |
| `AIR_RHO` | `1e9` Ω·m | Sentinel value for region 0 (air) |
| `OCEAN_RHO` | `0.25` Ω·m | Sentinel value for region 1 when treated as ocean |

Auto-inference: region 1 is treated as ocean when `flag == 1` **and**
ρ ≤ 1 Ω·m.  Override with `OCEAN = True / False` when this heuristic is
unreliable.

### Geographic / UTM origin

These parameters locate the mesh centre in geographic space.  They are used
to (a) derive the UTM zone number, and (b) convert lat/lon or UTM slice
positions to model-local metres.

| Variable | Default | Description |
|---|---|---|
| `UTM_ORIGIN_LAT` | `-16.409` | Latitude of mesh centre (decimal °, + = N) |
| `UTM_ORIGIN_LON` | `-71.537` | Longitude of mesh centre (decimal °, + = E) |
| `UTM_ORIGIN_E` | `229047.0` | UTM easting of mesh centre (m) |
| `UTM_ORIGIN_N` | `8184127.0` | UTM northing of mesh centre (m) |
| `UTM_ZONE_OVERRIDE` | `None` | Force a specific zone number (1–60); `None` = auto |

The UTM zone is derived automatically from `UTM_ORIGIN_LON` using the
standard 6° band rule (Norway/Svalbard special zones not handled; use
`UTM_ZONE_OVERRIDE` if needed).  The hemisphere is inferred from the sign
of `UTM_ORIGIN_LAT`.  The zone and active projection backend are printed at
startup.

### Mesh-centre estimation from calibration sites

If `UTM_ORIGIN_E` / `UTM_ORIGIN_N` are not known in advance, they can be
estimated from a small set of MT sites whose model-local positions (from
`observe.dat`) and geographic coordinates are both known.  Set
`ESTIMATE_ORIGIN = True` and populate `CALIBRATION_SITES`:

```python
ESTIMATE_ORIGIN = True

CALIBRATION_SITES = [
    dict(site=1,  crs="latlon", coords=[-71.500, -16.380]),  # [lon, lat]
    dict(site=10, crs="latlon", coords=[-71.620, -16.450]),
    dict(site=25, crs="utm",    coords=[224500., 8179300.]),  # [E, N]
]
```

| Key | Type | Description |
|---|---|---|
| `site` | int | Site number — matched against `observe.dat` |
| `crs` | str | `"latlon"` for decimal degrees; `"utm"` for UTM metres |
| `coords` | list | `[lon_deg, lat_deg]` for `"latlon"`; `[E_m, N_m]` for `"utm"` |

**Method** — the mesh is a pure translation of UTM space (FEMTIC aligns
model-local axes with UTM east/north), so each site yields one estimate of
the origin:

```
UTM_ORIGIN_E = E_site − x_m_site
UTM_ORIGIN_N = N_site − y_m_site
```

With N ≥ 1 sites the least-squares solution is the mean of the N implied
origins.  The estimated values and per-site residuals are printed; a large
residual on one site flags a coordinate error.  The result overwrites
`UTM_ORIGIN_E` / `UTM_ORIGIN_N` for the current run — copy the printed
values back into the Configuration block once satisfied.

Note: `UTM_ORIGIN_LAT` / `UTM_ORIGIN_LON` are still needed to derive the
UTM zone number; they only need to be approximately correct (any point in
the survey area within the correct 6° longitude band suffices).

### Display coordinate system

| `DISPLAY_COORDS` | Axis ticks |
|---|---|
| `"model"` (default) | model-local metres, origin at mesh centre |
| `"utm"` | absolute UTM metres |

### Site overlay

| Variable | Default | Description |
|---|---|---|
| `SITE_NUMBER` | `5` | 1-based site index from `observe.dat`; `None` = no overlay |
| `SITE_MARKER` | `dict(marker="v", color="black", ms=8, …)` | Matplotlib marker kwargs for map panels |

On **map panels** the site is plotted as a point marker at its (x, y)
model-local position.  On **NS / EW curtain panels** it is plotted as a
dashed vertical line at the projected horizontal coordinate (y or x
respectively).  Arbitrary-plane panels carry no automatic site projection.

### Output / figure

| Variable | Default | Description |
|---|---|---|
| `PLOT_FILE` | `resistivity_block_iter0.pdf` | Output path; `None` = interactive `show()` |
| `PLOT_DPI` | `600` | Figure DPI for saved file |
| `PLOT_CMAP` | `"turbo_r"` | Matplotlib colormap name |
| `PLOT_CLIM` | `[0.0, 4.0]` | Colour limits in log10(Ω·m); `None` = auto |
| `PLOT_OCEAN_COLOR` | `"lightgrey"` | Flat colour for ocean/lake cells; `None` = use colormap |
| `PLOT_AIR_BGCOLOR` | `None` | Axes facecolor shown through transparent air; `None` = figure default |
| `PLOT_XLIM` | `[-20000., 20000.]` | Global easting limits (model-local m); `None` = auto |
| `PLOT_YLIM` | `[-20000., 20000.]` | Global northing limits (model-local m); `None` = auto |
| `PLOT_ZLIM` | `[-6000., 15000.]` | Global depth limits (model-local m); `None` = auto |

---

## Slice specification (`PLOT_SLICES`)

`PLOT_SLICES` is a list of dicts, one per panel (left → right in the
figure).  Four slice kinds are supported:

| `kind` | Geometry | Horizontal axis | Vertical axis | Required key |
|---|---|---|---|---|
| `"map"` | Horizontal at z = const | x (easting) | y (northing) | `z0` |
| `"ns"` | N-S curtain at x = const | y (northing) | depth ↓ | `x0` |
| `"ew"` | E-W curtain at y = const | x (easting) | depth ↓ | `y0` |
| `"plane"` | Arbitrary strike/dip plane | along-strike | down-dip | `point`, `strike`, `dip` |

All panels accept the optional keys `xlim`, `ylim`, `zlim` (model-local m,
override the globals) and `title` (string, overrides the auto-generated
panel title).

### Position input coordinate systems

Every horizontal position key (`x0`, `y0`, and the horizontal components of
`point`) accepts input in three coordinate systems via an optional `"crs"`
tag:

```python
# Plain float — model-local metres (backward-compatible):
dict(kind="ns",  x0=0.0)

# (value, "crs") tuple — explicit coordinate system:
dict(kind="ns",  x0=(229047.0, "utm"))      # UTM easting [m]
dict(kind="ew",  y0=(-16.409, "latlon"))    # geographic latitude [°]
dict(kind="ns",  x0=(-71.537, "latlon"))    # geographic longitude [°]
```

| `crs` | `x0` meaning | `y0` meaning |
|---|---|---|
| `"model"` | model-local metres (+ east) | model-local metres (+ north) |
| `"utm"` | UTM easting [m] | UTM northing [m] |
| `"latlon"` | longitude [decimal °] | latitude [decimal °] |

`z0` and the depth component of `point` are **always** model-local metres
regardless of `crs`; depth has no geographic equivalent.

For `"plane"` slices the `point` key follows the same tagging convention,
with `[lon_deg, lat_deg, z_m]` ordering for `"latlon"`:

```python
dict(kind="plane", point=[0., 0., 5000.],               strike=45., dip=70.)
dict(kind="plane", point=([229047., 8184127., 5000.], "utm"),    strike=45., dip=70.)
dict(kind="plane", point=([-71.537, -16.409, 5000.],  "latlon"), strike=45., dip=70.)
```

All conversions are applied once by `resolve_slices()` before plotting.
The internal `plot_model_slices` function always receives model-local metres.

### Full example

```python
PLOT_SLICES = [
    # Two depth slices, model-local metres:
    dict(kind="map",  z0=5000.),
    dict(kind="map",  z0=20000., title="20 km depth"),

    # NS curtain positioned by UTM easting:
    dict(kind="ns",   x0=(229047., "utm"),   zlim=[-2000., 30000.]),

    # EW curtain positioned by geographic latitude:
    dict(kind="ew",   y0=(-16.409, "latlon"), zlim=[-2000., 30000.]),

    # Arbitrary plane through a geographic point:
    dict(kind="plane",
         point=([-71.537, -16.409, 0.], "latlon"),
         strike=30., dip=80.),
]
```

---

## `observe.dat` site positions

Each site in `observe.dat` occupies a block whose first line has the form:

```
<n>  <n>  <x_km>  <y_km>
```

where `n` is the site number (repeated in columns 1 and 2), and `x_km` /
`y_km` are model-local coordinates in **kilometres** (columns 3 and 4).
`read_site_position` scans the file linearly, identifies header lines by
the pattern `int int float float`, and returns `(x_m, y_m)` in metres.

---

## Coordinate conversion

### UTM zone derivation

```
zone = int((UTM_ORIGIN_LON + 180) / 6) + 1   [clamped to 1–60]
northern = UTM_ORIGIN_LAT >= 0
```

Override with `UTM_ZONE_OVERRIDE` for non-standard zones (Norway north of
56°, Svalbard).

### lat/lon → UTM → model-local

```
lat/lon ──[_latlon_to_utm]──► (E_m, N_m)
                                    │
                        [_utm_to_model]
                                    │
                                    ▼
                 (E_m − UTM_ORIGIN_E,  N_m − UTM_ORIGIN_N)
                 = model-local (x_m, y_m)
```

`_latlon_to_utm` uses **`pyproj.Transformer`** when available (primary
path; handles all edge cases, special projections, and datum shifts).
When `pyproj` is not importable the function falls back silently to a
built-in **Helmert/Bowring Transverse Mercator series**, which is accurate
to < 1 mm anywhere within a single UTM zone and requires no external
dependency.  The active backend is printed at startup when `OUT = True`.

---

## 3-D plotting (`PLOT3D`)

When `PLOT3D = True` the script renders a 3-D PyVista scene of the same
model after the 2-D slice figure.  PyVista must be installed
(`conda install -c conda-forge pyvista`); the step is silently skipped
when it is absent.

### Output format

| `PLOT3D_FILE` extension | Result |
|---|---|
| `.html` | Interactive WebGL scene — open in any browser, no PyVista needed at runtime |
| `.png` / `.jpg` | Static screenshot (anti-aliased at `screenshot_scale`× resolution) |
| `None` | Opens a live PyVista window (requires a display / VTK renderer) |

`.html` is the recommended format for sharing or archiving.

### 3-D configuration parameters

| Variable | Default | Description |
|---|---|---|
| `PLOT3D` | `False` | Enable / disable the 3-D step |
| `PLOT3D_FILE` | `*_3d.html` | Output path (see table above) |
| `PLOT3D_SCALAR` | `"log10_resistivity"` | Cell-data scalar to display (`"log10_resistivity"` or `"resistivity"`) |
| `PLOT3D_CLIM` | `[0.0, 4.0]` | Colour limits in the scalar's units; `None` = PyVista auto |
| `PLOT3D_CMAP` | `"turbo_r"` | Colormap for slices and iso-surfaces |
| `PLOT3D_SLICE_X` | `[0.0]` | x-positions of YZ (N-S) cutting planes (model-local m); `[]` = none |
| `PLOT3D_SLICE_Y` | `[0.0]` | y-positions of XZ (E-W) cutting planes; `[]` = none |
| `PLOT3D_SLICE_Z` | `[5000., 15000.]` | z-positions of XY (horizontal) cutting planes; `[]` = none |
| `PLOT3D_SLICE_PLANES` | `[]` | Oblique planes — list of `dict(origin=[x,y,z], normal=[nx,ny,nz])` |
| `PLOT3D_ISOVALUES` | `[1.0, 2.0, 3.0]` | Iso-surface levels (log10 Ω·m for default scalar); `[]` = none |
| `PLOT3D_ISO_OPACITY` | `0.35` | Iso-surface opacity (0 = transparent, 1 = opaque) |
| `PLOT3D_WINDOW_SIZE` | `[1600, 900]` | Window resolution in pixels |

### Axis-aligned slices

Each position in `PLOT3D_SLICE_X/Y/Z` places one infinite cutting plane
perpendicular to the named axis:

```python
PLOT3D_SLICE_X = [0.0]             # one YZ plane through the model centre
PLOT3D_SLICE_Y = [0.0]             # one XZ plane through the model centre
PLOT3D_SLICE_Z = [5000., 15000.]   # two horizontal maps at 5 km and 15 km depth
```

If all three lists are empty and no oblique planes or iso-surfaces are
defined, a default orthogonal triple (one slice per axis through the model
centre) is added automatically.

### Oblique plane slices

```python
PLOT3D_SLICE_PLANES = [
    dict(origin=[0., 0., 8000.], normal=[1., 1., 0.]),   # NE-striking vertical
    dict(origin=[0., 0., 5000.], normal=[0., 0., 1.]),   # horizontal at 5 km
]
```

`origin` is any point on the plane; `normal` need not be a unit vector.

### Iso-surfaces

```python
PLOT3D_ISOVALUES  = [1.0, 2.0, 3.0]   # conductor boundary / background / resistor
PLOT3D_ISO_OPACITY = 0.35              # semi-transparent
```

Iso-surfaces are coloured by the same scalar (and same clim/cmap) as the
slices, making conductor/resistor boundaries directly visible in 3-D.

---

## Ensemble slice plot (`PLOT_ENS`)

When `PLOT_ENS = True` the script produces a joint multi-row figure in which
each row shows one ensemble member across the same slice columns defined by
`PLOT_SLICES`.  Optional statistical summary rows (mean, std, median of
log₁₀(ρ) across all members) are appended below the member rows.

The mesh is parsed **once**; slice polygon geometry is precomputed **once**
per slice position; only the per-element resistivity vector is swapped for
each member.  This makes the function efficient even for large meshes.

All `PLOT_*` parameters (colormap, colour limits, axis limits, ocean colour,
etc.) are reused identically — no separate ensemble-specific colour config
is needed.

### Configuration parameters

| Variable | Default | Description |
|---|---|---|
| `PLOT_ENS` | `False` | Enable / disable the ensemble step |
| `ENS_FILES` | `[]` | List of resistivity block paths, one per member |
| `ENS_LABELS` | `None` | Row label strings; `None` → "Member 0", "Member 1", … |
| `ENS_STAT_ROWS` | `["mean", "std"]` | Stat rows appended after member rows: any subset of `"mean"`, `"std"`, `"median"` |
| `PLOT_ENS_FILE` | `*_ensemble.pdf` | Joint figure path; `None` → interactive show |
| `ENS_PER_MEMBER` | `False` | Also save one figure per member (named `*_memberN.pdf`) |

### Statistical summary rows

| `ENS_STAT_ROWS` entry | What is shown |
|---|---|
| `"mean"` | Cell-wise arithmetic mean of log₁₀(ρ) across all members; same colormap / clim as member rows |
| `"std"` | Cell-wise standard deviation of log₁₀(ρ); rendered on a separate sequential colormap (`cividis`) with its own colour scale |
| `"median"` | Cell-wise median of log₁₀(ρ); same colormap / clim as member rows |

NaN elements (air, missing) are excluded from all statistics.

### Populating `ENS_FILES`

```python
import glob, os
ENS_FILES = sorted(glob.glob(
    WORK_DIR + "ensemble/ubinas_rto_*/resistivity_block_iter10.dat"
))
ENS_LABELS = [os.path.basename(os.path.dirname(f)) for f in ENS_FILES]
```

### Per-member output

When `ENS_PER_MEMBER = True` and `PLOT_ENS_FILE` is set, one additional
single-row figure is saved per member alongside the joint figure:

```
resistivity_block_ensemble.pdf       ← joint (all members + stat rows)
resistivity_block_ensemble_member0.pdf
resistivity_block_ensemble_member1.pdf
…
```

---

## Air and ocean rendering

- **Air** (region 0): `prepare_rho_for_plotting` sets air to `NaN`.
  Intersection polygons are computed but not added to the `PolyCollection`;
  the axes facecolor (`PLOT_AIR_BGCOLOR`) shows through.  Correct
  topography is guaranteed because the plane intersection is geometrically
  exact (every tetrahedron that straddles the plane contributes an exact
  triangle or quadrilateral — no interpolation slab, no `dw` parameter).

- **Ocean / lake** (region 1 when active): set to the `OCEAN_RHO` sentinel
  and rendered as a separate flat-colour `PolyCollection` above the earth
  layer, visually distinct from the colormap range.  Set
  `PLOT_OCEAN_COLOR = None` to route ocean cells through the colormap
  instead.

---

## Dependencies

| Package | Role |
|---|---|
| NumPy | Array operations |
| Matplotlib | Figure rendering |
| PyVista | 3-D rendering, slices, iso-surfaces (`PLOT3D`; graceful skip when absent) |
| pyproj | `Transformer` for lat/lon → UTM (primary path; graceful built-in fallback when absent) |
| `femtic` (Py4MTX) | Environment bootstrap |
| `femtic_viz` (Py4MTX) | `read_femtic_mesh`, `read_resistivity_block`, `map_regions_to_element_rho`, `prepare_rho_for_plotting`, `plot_model_3d` |
| `util` (Py4MTX) | `print_title` |
| `version` (Py4MTX) | `versionstrg` |

Environment variables `PY4MTX_ROOT` and `PY4MTX_DATA` must be set.

---

## Borehole resistivity logs (`PLOT_BOREHOLE`)

When `PLOT_BOREHOLE = True` the script samples the resistivity model along
one or more vertical boreholes and produces a 1-D log₁₀(ρ) vs depth figure
(step 7, after the ensemble plot).

### Method — point-in-element

Each depth level `z` in the borehole is queried by a **point-in-element**
search on the tetrahedral mesh:

1. A lateral bounding-box pre-filter selects only elements whose x-y extents
   bracket `(x_m, y_m)` — computed once per borehole, reused for all depth
   levels.
2. Among those candidates, a z-range sub-filter discards elements whose depth
   extents do not bracket `z`.
3. The remaining candidates are tested with an exact **barycentric coordinate**
   test (`_point_in_tet`): `p = v0 + T·λ`; the point is inside when all
   `λᵢ ≥ 0` and `Σλᵢ ≤ 1` (tolerance 10⁻¹⁰).

The resistivity of the first containing element is returned.  Levels outside
every element (air, above or below the mesh) are set to NaN and appear as
gaps in the trace.

### Configuration parameters

| Variable | Default | Description |
|---|---|---|
| `PLOT_BOREHOLE` | `False` | Enable / disable the borehole step |
| `BOREHOLE_FILE` | `*_boreholes.pdf` | Output path; `None` → interactive show |
| `BOREHOLE_SITES` | `[]` | List of borehole spec dicts (see below) |
| `BOREHOLE_STYLE` | `dict(lw=1.2, marker="none")` | Matplotlib line kwargs for all traces |
| `BOREHOLE_XLIM` | `[0.0, 4.0]` | x-axis limits [log10(Ω·m)]; `None` = auto |
| `BOREHOLE_SHARED` | `True` | `True` = all on one axes; `False` = one panel per borehole |

### Borehole spec dict

Each entry in `BOREHOLE_SITES` is a dict with:

| Key | Type | Description |
|---|---|---|
| `"name"` | str | Label shown in the legend / panel title |
| `"x"` | float or `(value, "crs")` | Borehole easting — same CRS tagging as `PLOT_SLICES` |
| `"y"` | float or `(value, "crs")` | Borehole northing — same CRS tagging as `PLOT_SLICES` |
| `"z_top"` | float | Start depth [m, FEMTIC z-down]; 0 = surface |
| `"z_bot"` | float | End depth [m, z-down], e.g. `20000.0` for 20 km |
| `"dz"` | float | Sampling interval [m], e.g. `200.0` |

`"x"` and `"y"` accept the same three coordinate systems as the horizontal
slice positions: a plain float is model-local metres; a `(value, "utm")`
tuple is a UTM easting/northing; a `(value, "latlon")` tuple is a geographic
longitude/latitude.

### Example

```python
PLOT_BOREHOLE = True
BOREHOLE_FILE = WORK_DIR + "resistivity_block_iter0_boreholes.pdf"
BOREHOLE_SHARED = True    # all traces on one axes
BOREHOLE_XLIM = [0., 4.]  # log10(Ω·m)

BOREHOLE_SITES = [
    # Model-local metres — surface to 20 km, 200 m steps:
    dict(name="BH-centre",  x=0.0,    y=0.0,
         z_top=0., z_bot=20000., dz=200.),

    # UTM coordinates:
    dict(name="BH-north",   x=(229047., "utm"), y=(8190000., "utm"),
         z_top=0., z_bot=15000., dz=100.),

    # Geographic coordinates:
    dict(name="BH-east",    x=(-71.50, "latlon"), y=(-16.40, "latlon"),
         z_top=500., z_bot=10000., dz=250.),
]
```

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Created, modelled on `femtic_mod_edit.py` plotting section; site overlay from `observe.dat` |
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Added lat/lon and UTM slice-position input; `pyproj` primary path with pure-Python Helmert fallback; auto-derived UTM zone from mesh origin |
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Added `estimate_utm_origin`: least-squares mesh-centre estimation from N calibration sites with known model-local and geographic coordinates |
| 2026-05-13 | vrath / Claude Sonnet 4.6 | Harmonised plotting config block with `femtic_mod_edit.py`: unified variable names, comments, section header |
| 2026-05-13 | vrath / Claude Sonnet 4.6 | Added `PLOT3D` step (5): axis-aligned x/y/z slices, oblique planes, and iso-surfaces via `fviz.plot_model_3d`; HTML or screenshot output |
| 2026-05-13 | vrath / Claude Sonnet 4.6 | Added `PLOT_ENS` step (6): `plot_ensemble_slices` with `ENS_*` config block; mesh and geometry parsed once; per-member rows + optional mean/std/median stat rows; per-member file output |
| 2026-05-16 | vrath / Claude Sonnet 4.6 | Added `PLOT_BOREHOLE` step (7): `_point_in_tet` (barycentric), `extract_borehole_log` (bbox pre-filter + exact test), `plot_borehole_logs`; `BOREHOLE_*` config block; CRS tagging on x/y positions |
