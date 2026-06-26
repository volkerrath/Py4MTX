# femtic_mod_plot_slice.py — README

**Purpose:** 2-D slice panels (map, curtain, plane) for a FEMTIC tetrahedral
resistivity model, with optional embedded borehole columns.

Sister scripts:
- [`femtic_mod_plot_bh.py`](femtic_mod_plot_bh_readme.md) — standalone 1-D ρ(z) borehole log figures.
- [`femtic_mod_plot_3d.py`](femtic_mod_plot_3d_readme.md) — PyVista 3-D rendering and VTK/VTU export.

---

## What this script does

```
mesh.dat + resistivity_block_iterX.dat
        │
        └─(5)─► fviz.plot_model_slices(...)   [2-D map/curtain/plane panels]
                 → PDF / PNG / interactive
                 (optional embedded borehole columns when BOREHOLE_IN_SLICE=True)
```

All geometry lives in `femtic_viz.py`; no geometry code lives here.

---

## Execution steps

| Step | What happens |
|---|---|
| (1) | Optionally estimate UTM mesh-origin from `SITE_DAT` bounding-box or mean |
| (2) | Derive UTM zone from finalised `UTM_ORIGIN_LAT` / `UTM_ORIGIN_LON` |
| (3) | Resolve `PLOT_SLICES` positions to model-local metres (CRS conversion) |
| (4) | Read site positions from `SITE_DAT` or `OBSERVE_FILE` for marker overlay |
| (5) | Plot 2-D slice panels via `fviz.plot_model_slices` |

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
| `"profile"` | `p1`, `p2`, `z_top`, `z_bot` | Vertical fence section between two endpoints; strike auto-derived |

### `kind="profile"` keys

| Key | Type | Default | Description |
|---|---|---|---|
| `p1`, `p2` | position spec | required | Endpoint positions. Each accepts a bare 2-element list (model-local metres), `([x, y], "model")`, `([E, N], "utm")`, or `([lon, lat], "latlon")`. A 3-element `[x, y, z]` form is also accepted (z ignored). |
| `z_top` | float | `0.0` | Shallowest depth of the panel [m, z-positive-down] |
| `z_bot` | float | `20000.0` | Deepest depth of the panel [m, z-positive-down] |

The resolver converts a `"profile"` spec into a `kind="plane"` dict with `dip=90` before passing it to `fviz.plot_model_slices`. The horizontal axis of the panel runs along-strike from *p1* to *p2*; use `invert_x=True` to reverse the sense.

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
    dict(kind="profile",
         p1=([-71.536, -16.197], "latlon"),
         p2=([-71.406, -16.299], "latlon"),
         z_top=0.0, z_bot=25000.0,
         title="NW-SE profile"),
]
```

---

## Embedded borehole columns (`BOREHOLE_IN_SLICE`)

When `BOREHOLE_IN_SLICE = True` and `BOREHOLE_SITES` is non-empty, borehole
ρ(z) panels are appended as extra columns to the right of the slice grid
inside the **same figure**.  The borehole depth axis is linked to the leftmost
curtain/plane panel for synchronised zoom/pan.

For a standalone borehole figure (higher DPI, separate PDF), use
`femtic_mod_plot_bh.py` instead.

### Borehole spec dict keys

| Key | Type | Required | Description |
|---|---|---|---|
| `"name"` | str | yes | Label in legend / panel title |
| `"x"` | float or `(v, "crs")` | yes | Easting: float = model-local m; `(lon, "latlon")` = longitude [°]; `(E_m, "utm")` = UTM easting [m] |
| `"y"` | float or `(v, "crs")` | yes | Northing: float = model-local m; `(lat, "latlon")` = latitude [°]; `(N_m, "utm")` = UTM northing [m] |
| `"z_top"` | float or `"surface"` | no (def. 0) | Start depth [m, z-down]. `"surface"` → auto from mesh nodes (requires scipy) |
| `"z_bot"` | float | no (def. 20000) | End depth [m, z-down] |
| `"dz"` | float | no (def. 200) | Sampling interval [m] |
| `"lat"` | float | no | Override legend latitude [°] (auto-inferred for `"latlon"` / `"utm"` CRS) |
| `"lon"` | float | no | Override legend longitude [°] |
| `"color"`, `"ls"`, `"lw"`, `"marker"`, `"alpha"`, … | any | no | Matplotlib `Line2D` kwargs — override `BOREHOLE_STYLE` for this trace |

### Borehole configuration

| Variable | Default | Description |
|---|---|---|
| `BOREHOLE_SITES` | `[]` | List of borehole spec dicts |
| `BOREHOLE_STYLE` | `dict(lw=1.2, marker="none")` | Baseline line style; per-spec keys override |
| `BOREHOLE_XLIM` | `[1., 1e4]` | x-axis limits [Ω·m, log scale]; `None` = auto |
| `BOREHOLE_SHARED` | `True` | `True` = one shared column; `False` = one column per borehole |
| `BOREHOLE_IN_SLICE` | `True` | Embed borehole columns inside the slice figure |

---

## Configuration reference

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
| `TICK_FONTSIZE` | `7` | Font size for axis tick labels and colourbar ticks |
| `LABEL_FONTSIZE` | `8` | Font size for axis labels, panel titles, and colourbar label |

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

## Dependencies

| Package | Role |
|---|---|
| `femtic` (Py4MTX) | `resolve_slice_positions`, `read_site_dat`, `read_site_position`, `utm_to_model`, `latlon_to_model` |
| `femtic_viz` (Py4MTX) | `plot_model_slices` |
| `util` (Py4MTX) | `utm_zone_from_latlon`, `utm_to_latlon_zn`, `print_title` |
| `matplotlib` | 2-D rendering (via `femtic_viz`) |
| `scipy` | Only for `z_top="surface"` KD-tree lookup in embedded boreholes |
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
| 2026-06-03 | Claude Sonnet 4.6 | **Split** from `femtic_mod_plot.py` → `femtic_mod_plot_slice.py` + `femtic_mod_plot_3d.py`. `BOREHOLE_IN_SLICE`: embedded borehole columns via `plot_model_slices(borehole_sites=...)`; `BOREHOLE_XLIM` now in Ω·m; `z_top="surface"` supported; lat/lon legend; per-trace line-style keys; `BOREHOLE_NPZ` added |
| 2026-06-04 | vrath / Claude Sonnet 4.6 | **Split** from `femtic_mod_plot_slice.py`: standalone borehole step (step 6) and `PLOT_BOREHOLE` flag moved to new `femtic_mod_plot_bh.py`. `BOREHOLE_IN_SLICE` retained here for embedded columns only |
| 2026-06-19 | Claude Sonnet 4.6 | `kind="profile"` added to `PLOT_SLICES`: vertical fence section defined by two endpoint positions (`p1`, `p2`) each accepting model-local / UTM / latlon CRS tags; `strike` derived from p1→p2 azimuth; `dip` fixed at 90°. New helper `resolve_pos_two_point_profile()` in `femtic.py`; `resolve_slice_positions()` extended. |
| 2026-06-26 | vrath / Claude Sonnet 4.6 | Added `TICK_FONTSIZE` (default `7`) and `LABEL_FONTSIZE` (default `8`) config constants; forwarded to `plot_model_slices` as `tick_fontsize` / `label_fontsize`. |
