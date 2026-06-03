# femtic_mod_plot_slice.py — README

**Purpose:** 2-D slice panels and borehole resistivity logs for a FEMTIC
tetrahedral resistivity model.

Sister script: [`femtic_mod_plot_3d.py`](femtic_mod_plot_3d_readme.md) —
PyVista 3-D rendering and VTK/VTU export from the same files.

---

## What this script does

```
mesh.dat + resistivity_block_iterX.dat
        │
        ├─(5)─► fviz.plot_model_slices(...)   [2-D map/curtain/plane panels]
        │        → PDF / PNG / interactive
        │
        └─(6)─► fviz.plot_borehole_logs(...)  [1-D ρ(z) traces, log x-axis]
                 → PDF / interactive           [optional, PLOT_BOREHOLE = True]
```

Both steps call functions from `femtic_viz.py`; no geometry code lives here.

---

## Execution steps

| Step | What happens |
|---|---|
| (1) | Optionally estimate UTM mesh-origin from `SITE_DAT` bounding-box or mean |
| (2) | Derive UTM zone from finalised `UTM_ORIGIN_LAT` / `UTM_ORIGIN_LON` |
| (3) | Resolve `PLOT_SLICES` positions to model-local metres (CRS conversion) |
| (4) | Read site positions from `SITE_DAT` or `OBSERVE_FILE` for marker overlay |
| (5) | Plot 2-D slice panels via `fviz.plot_model_slices` |
| (6) | Plot borehole logs via `fviz.plot_borehole_logs` (if `PLOT_BOREHOLE=True`) |

---

## Slice coordinate system

Every horizontal slice position (`x0`, `y0`, `point` horizontal component)
accepts a coordinate-system tag:

| Value | Meaning |
|---|---|
| plain `float` | model-local metres — origin at mesh centre (backward-compatible) |
| `(value, "utm")` | UTM metres in the mesh UTM zone |
| `(value, "latlon")` | decimal degrees (longitude for easting, latitude for northing) |

Depth `z0` is always model-local metres (z positive-down); no geographic
conversion applies.

Conversion chain: `lat/lon → UTM → model-local`.

---

## 2-D slice panels (`PLOT_SLICES`)

### Panel kinds

| `kind` | Position key | Description |
|---|---|---|
| `"map"` | `z0` | Horizontal slice at depth `z0` |
| `"ns"` | `x0` | N-S vertical section at easting `x0` |
| `"ew"` | `y0` | E-W vertical section at northing `y0` |
| `"plane"` | `point`, `strike`, `dip` | Arbitrary oblique plane |

### Per-panel optional keys

| Key | Description |
|---|---|
| `xlim` | `[xmin, xmax]` — easting / along-strike axis limit |
| `ylim` | `[ymin, ymax]` — northing / down-dip axis limit |
| `zlim` | `[zmin, zmax]` — depth axis limit (ns / ew / plane panels) |
| `invert_x` | `True` → flip horizontal axis left-to-right (default `False`) |
| `title` | Optional string override for panel title |

Per-panel `xlim`/`ylim`/`zlim` override the global `PLOT_XLIM`/`PLOT_YLIM`/`PLOT_ZLIM`.

### Example

```python
PLOT_SLICES = [
    dict(kind="map", z0=-4000.0),
    dict(kind="ns",  x0=(-70.90, "latlon")),
    dict(kind="ew",  y0=(-16.40, "latlon")),
    dict(kind="plane",
         point=([-71.5, -16.4, 5000.0], "latlon"),
         strike=45., dip=70.,
         invert_x=True),
]
```

---

## Configuration reference — 2-D slices

### Paths

| Variable | Description |
|---|---|
| `WORK_DIR` | Working directory prefix for all paths |
| `MODEL_FILE` | `resistivity_block_iterX.dat` to display |
| `MESH_FILE` | `mesh.dat` |
| `OBSERVE_FILE` | `observe.dat` — fallback site source |
| `SITE_DAT` | `mt_make_sitelist.py` CSV (`name,lat,lon,elev,sitenum,easting,northing`); `None` to disable |

### Ocean / air

| Variable | Default | Description |
|---|---|---|
| `OCEAN` | `None` | `None` = auto-infer; `True`/`False` = force |
| `AIR_RHO` | `1e9` | Air sentinel Ω·m (region 0) |
| `OCEAN_RHO` | `0.25` | Ocean sentinel Ω·m (region 1) |

### UTM / CRS

| Variable | Default | Description |
|---|---|---|
| `UTM_ORIGIN_LAT` | `None` | Mesh-centre latitude [°] |
| `UTM_ORIGIN_LON` | `None` | Mesh-centre longitude [°] |
| `UTM_ORIGIN_E` | `None` | Mesh-centre UTM easting [m] |
| `UTM_ORIGIN_N` | `None` | Mesh-centre UTM northing [m] |
| `UTM_ZONE_OVERRIDE` | `None` | Force UTM zone number; `None` = auto |
| `ORIGIN_METHOD` | `"box"` | `None` / `"box"` / `"average"` — origin estimation from `SITE_DAT` |
| `DISPLAY_COORDS` | `"model"` | `"model"` / `"utm"` / `"latlon"` |

### Site overlay

| Variable | Default | Description |
|---|---|---|
| `SITE_NAMES` | `None` | Site names from `SITE_DAT`; `None` = all |
| `SITE_NUMBER` | `[5,6,7]` | Fallback: site numbers from `observe.dat` |
| `PLOT_SITES_MAPS` | `True` | Show markers on map panels |
| `PLOT_SITES_SLICES` | `True` | Show markers on curtain / plane panels |
| `PROJECTION_DIST` | `1000.` | Max distance [m] from slice plane for site to appear |
| `SITE_MARKER` | `dict(marker="v",…)` | Marker style for map panels |
| `SITE_MARKER_SLICES` | `dict(marker="v",…)` | Marker style for curtain / plane panels |
| `MAP_MARKERS` | `[]` | Extra lat/lon point markers on map panels only |

### Plotting

| Variable | Default | Description |
|---|---|---|
| `PLOT_FILE` | `*_slices.pdf` | Output path; `None` = interactive |
| `PLOT_DPI` | `600` | Saved-figure DPI |
| `PLOT_CMAP` | `"turbo_r"` | Matplotlib colormap |
| `PLOT_CLIM` | `[0., 3.]` | `[log10_min, log10_max]` in log10(Ω·m); `None` = auto |
| `PLOT_OCEAN_COLOR` | `"lightgrey"` | Flat colour for ocean cells; `None` = colormap |
| `PLOT_AIR_COLOR` | `"whitesmoke"` | Flat colour for air polygons; `None` = blank |
| `PLOT_AIR_BGCOLOR` | `None` | Axes facecolor for air/background |

### Alpha / blanking

| Variable | Default | Description |
|---|---|---|
| `ALPHA_FILE` | `None` | Second block file whose `region_rho` values are log10 weights |
| `ALPHA_MODE` | `"fade"` | `"fade"` = proportional transparency; `"blank"` = hard cutoff |
| `ALPHA_BLANK_THRESH` | `0.0` | log10 threshold below which polygons are suppressed |

### Axis limits and aspect

| Variable | Default | Description |
|---|---|---|
| `PLOT_XLIM` | `[-15000., 15000.]` | Global easting limits [m] |
| `PLOT_YLIM` | `[-15000., 15000.]` | Global northing limits [m] |
| `PLOT_ZLIM` | `[-6000., 15000.]` | Global depth limits [m] |
| `PLOT_EQUAL_ASPECT` | `True` | Equal-aspect on map / curtain panels |
| `DEPTH_KM` | `True` | Depth axis in km on curtain / plane panels |
| `HORIZ_KM` | `True` | Horizontal axes in km when `DISPLAY_COORDS="model"` |

### Figure layout

| Variable | Default | Description |
|---|---|---|
| `PLOT_NROWS` | `None` | Subplot rows; `None` = 1 |
| `PLOT_NCOLS` | `None` | Subplot columns; `None` = `len(PLOT_SLICES)` |
| `PLOT_PANEL_HEIGHT` | `16.0` | Panel height [cm] |
| `PLOT_PANEL_WIDTH` | `None` | Fixed panel width [cm]; `None` = auto |
| `PLOT_FIGSIZE` | `None` | Full figure size `[w, h]` [cm]; overrides auto |

---

## Borehole resistivity logs (`PLOT_BOREHOLE`)

When `PLOT_BOREHOLE = True` the script samples the resistivity model along
one or more vertical boreholes and produces a **ρ vs depth** figure with a
**logarithmic x-axis** (Ohm·m).

### Spec dict keys

| Key | Type | Required | Description |
|---|---|---|---|
| `"name"` | str | yes | Label in legend / panel title |
| `"x"` | float or `(v, "crs")` | yes | Borehole easting — same CRS tagging as `PLOT_SLICES` |
| `"y"` | float or `(v, "crs")` | yes | Borehole northing |
| `"z_top"` | float or `"surface"` | no (def. 0) | Start depth [m, z-down].  `"surface"` → auto from mesh nodes (requires scipy) |
| `"z_bot"` | float | no (def. 20000) | End depth [m, z-down] |
| `"dz"` | float | no (def. 200) | Sampling interval [m] |
| `"lat"` | float | no | Latitude [°] shown in legend instead of model-local y |
| `"lon"` | float | no | Longitude [°] shown in legend instead of model-local x |
| `"color"`, `"ls"`, `"lw"`, `"marker"`, `"alpha"`, … | any | no | Matplotlib `Line2D` kwargs — override `BOREHOLE_STYLE` for this trace |

### Borehole configuration

| Variable | Default | Description |
|---|---|---|
| `PLOT_BOREHOLE` | `True` | Enable / disable |
| `BOREHOLE_FILE` | `*_boreholes.pdf` | Output path; `None` = interactive |
| `BOREHOLE_SITES` | `[]` | List of spec dicts |
| `BOREHOLE_STYLE` | `dict(lw=1.2, marker="none")` | Baseline line style; per-spec keys override |
| `BOREHOLE_XLIM` | `[1., 1e4]` | x-axis limits [Ω·m, log scale]; `None` = auto |
| `BOREHOLE_SHARED` | `True` | `True` = one shared axes; `False` = one panel per borehole |

### Example

```python
PLOT_BOREHOLE = True
BOREHOLE_FILE = WORK_DIR + "resistivity_block_iter17_boreholes.pdf"
BOREHOLE_SHARED = True
BOREHOLE_XLIM   = [1., 1e4]   # Ω·m

BOREHOLE_SITES = [
    dict(name="BH-centre", x=0.0, y=0.0,
         lat=-16.363, lon=-70.868,
         z_top="surface", z_bot=20000., dz=200.,
         color="steelblue", ls="-"),
    dict(name="BH-north",
         x=(229047., "utm"), y=(8190000., "utm"),
         z_top=0., z_bot=15000., dz=100.,
         color="firebrick", ls="--"),
]
```

---

## Dependencies

| Package | Role |
|---|---|
| `femtic` (Py4MTX) | `resolve_slice_positions`, `read_site_dat`, `read_site_position`, `utm_to_model`, `resolve_pos_x/y`, `extract_borehole_log` |
| `femtic_viz` (Py4MTX) | `plot_model_slices`, `plot_borehole_logs` |
| `util` (Py4MTX) | `utm_zone_from_latlon`, `utm_to_latlon_zn`, `print_title` |
| `matplotlib` | 2-D rendering (via `femtic_viz`) |
| `scipy` | Only for `z_top="surface"` KD-tree lookup |
| `numpy` | Array operations |

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Created as part of `femtic_mod_plot.py` |
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Lat/lon and UTM slice positions; auto UTM zone |
| 2026-05-13 | vrath / Claude Sonnet 4.6 | Config block harmonised with `femtic_mod_edit.py` |
| 2026-05-16 | vrath / Claude Sonnet 4.6 | Borehole step added; `BOREHOLE_*` config block |
| 2026-05-23 | vrath / Claude Sonnet 4.6 | Geographic helpers moved to `util.py`; model-local helpers to `femtic.py` |
| 2026-05-23 | vrath / Claude Sonnet 4.6 | `SITE_DAT` in `mt_make_sitelist.py` CSV format |
| 2026-05-24 | vrath / Claude Sonnet 4.6 | Ensemble step removed (→ `snippets.py`); borehole renumbered step 6 |
| 2026-05-25 | vrath / Claude Sonnet 4.6 | `PLOT_SITES_MAPS`/`_SLICES`, `PROJECTION_DIST`, `DEPTH_KM`, `HORIZ_KM`, `PLOT_NROWS`/`NCOLS`, panel sizing in cm |
| 2026-05-26 | Claude Sonnet 4.6 | `plot_model_slices`/`plot_borehole_logs` moved into `femtic_viz.py`; `ALPHA_FILE`/`ALPHA_MODE`/`ALPHA_BLANK_THRESH` added |
| 2026-05-27 | vrath / Claude Sonnet 4.6 | `PLOT_XLIM/YLIM/ZLIM` passed to slice panels |
| 2026-05-31 | vrath / Claude Sonnet 4.6 | `invert_x` per-panel key; origin estimation before UTM zone derivation |
| 2026-06-03 | Claude Sonnet 4.6 | **Split** from `femtic_mod_plot.py` → `femtic_mod_plot_slice.py` + `femtic_mod_plot_3d.py`. Borehole: `BOREHOLE_XLIM` now in Ω·m; `z_top="surface"`; lat/lon legend; per-trace line-style keys |
