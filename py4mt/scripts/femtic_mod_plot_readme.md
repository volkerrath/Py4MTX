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
| pyproj | `Transformer` for lat/lon → UTM (primary path; graceful built-in fallback when absent) |
| `femtic` (Py4MTX) | Environment bootstrap |
| `femtic_viz` (Py4MTX) | `read_femtic_mesh`, `read_resistivity_block`, `map_regions_to_element_rho`, `prepare_rho_for_plotting` |
| `util` (Py4MTX) | `print_title` |
| `version` (Py4MTX) | `versionstrg` |

Environment variables `PY4MTX_ROOT` and `PY4MTX_DATA` must be set.

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Created, modelled on `femtic_mod_edit.py` plotting section; site overlay from `observe.dat` |
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Added lat/lon and UTM slice-position input; `pyproj` primary path with pure-Python Helmert fallback; auto-derived UTM zone from mesh origin |
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Added `estimate_utm_origin`: least-squares mesh-centre estimation from N calibration sites with known model-local and geographic coordinates |
