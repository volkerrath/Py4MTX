# femtic_ens_plot.py

Ensemble slice plot for a set of FEMTIC inversion runs, with optional
borehole resistivity logs.

---

## Purpose

`femtic_ens_plot.py` collects resistivity block files from a list of
ensemble run directories and produces:

1. A joint multi-row Matplotlib figure — one row per ensemble member —
   using the same slice geometry and `PLOT_*` parameters as
   `femtic_mod_plot.py`.  Optional statistical summary rows (mean, std,
   median of log₁₀(ρ) across all members) are appended.  Optionally one
   figure per member is saved alongside the joint figure.

2. Optionally, a borehole resistivity log figure (point-in-element sampling,
   identical to step (6) in `femtic_mod_plot.py`).

The ensemble plot step is taken directly from `snippets.py` (Snippet 1).
Slice positions, UTM/geographic coordinate handling, site overlay, and all
`PLOT_*` parameters follow the same conventions as `femtic_mod_plot.py`.

Typical use cases:

- comparing model structure across an RTO or Monte Carlo ensemble,
- producing a mean / std uncertainty figure alongside the member rows,
- checking spread in a specific region relative to MT site locations.

---

## Workflow

```
UTM_ORIGIN_LAT / LON  →  UTM zone (auto)
                                          [optional: ESTIMATE_ORIGIN = True]
        CALIBRATION_SITES  +  observe.dat  →  fem.estimate_utm_origin()
                                               UTM_ORIGIN_E / N  (printed)

ENS_DIRS  +  BLOCK_PATTERN  +  ENS_ITER
        |
        v  glob expand + sort → locate block file in each directory
   ENS_FILES  (list of resistivity block paths)
   ENS_LABELS_resolved  (directory basenames or user labels)
        |
        v  resolve_slices(PLOT_SLICES)       [optional CRS conversion]
   slice positions in model-local metres
        |                                    [optional]
        v  fem.read_site_dat(SITE_DAT)        [primary: mt_make_sitelist CSV]
   (name, x_m, y_m) per site
        OR  fem.read_site_position(OBSERVE_FILE)  [fallback: observe.dat]
        |
        v  fviz.plot_ensemble_slices(...)    [snippets.py Snippet 1]
joint PDF (all members + stat rows)  +  optional per-member PDFs
        |                                    [optional, PLOT_BOREHOLE = True]
        v  plot_borehole_logs(...)           [1-D ρ(z) — member 0 as reference]
borehole PDF / interactive window
```

---

## Configuration

All user-editable settings live in the **Configuration** block near the top
of the script.  No command-line arguments are used; edit the script directly.

### Paths

| Variable | Default | Description |
|---|---|---|
| `WORK_DIR` | `/home/vrath/FEMTIC_work/ubinas_rto/` | Working directory |
| `MESH_FILE` | `mesh.dat` | Mesh file — always required |
| `OBSERVE_FILE` | `observe.dat` | Used by `ESTIMATE_ORIGIN`; fallback site-overlay source when `SITE_DAT` is None |
| `SITE_DAT` | `site.dat` | Site list from `mt_make_sitelist.py`; `None` = use observe.dat fallback |

### Ensemble input

| Variable | Default | Description |
|---|---|---|
| `ENS_DIRS` | `[]` | List of ensemble run directories; glob patterns are expanded and sorted |
| `BLOCK_PATTERN` | `"resistivity_block_iter{iter}.dat"` | Filename pattern; `{iter}` is replaced by `ENS_ITER` |
| `ENS_ITER` | `10` | Inversion iteration whose block file is used from each directory |
| `ENS_LABELS` | `None` | Row label strings — one per directory; `None` → last component of the directory path |
| `ENS_STAT_ROWS` | `["mean", "std"]` | Stat rows appended after member rows: any subset of `"mean"`, `"std"`, `"median"` |
| `PLOT_ENS_FILE` | `ensemble.pdf` | Joint figure path; `None` → interactive show |
| `ENS_PER_MEMBER` | `False` | Also save one figure per member (`*_memberN.pdf`) |

#### Populating `ENS_DIRS`

```python
import glob
ENS_DIRS = sorted(glob.glob(WORK_DIR + "ubinas_rto_*/"))
```

Or list directories explicitly:

```python
ENS_DIRS = [
    WORK_DIR + "ubinas_rto_0/",
    WORK_DIR + "ubinas_rto_1/",
    WORK_DIR + "ubinas_rto_2/",
]
```

Glob patterns inside the list are also expanded at runtime, so mixed forms
work:

```python
ENS_DIRS = [
    WORK_DIR + "reference/",          # single explicit directory
    WORK_DIR + "ubinas_rto_[0-9]*/",  # glob for numbered runs
]
```

#### Block file resolution

For each directory `d`, the script looks for:

```
d / BLOCK_PATTERN.format(iter=ENS_ITER)
```

Missing files produce a warning and are skipped; if **no** files are found
the script exits.  Labels default to the last component of each directory
path (e.g. `"ubinas_rto_3"` for `.../ubinas_rto_3/`).

#### Statistical summary rows

| `ENS_STAT_ROWS` entry | What is shown |
|---|---|
| `"mean"` | Cell-wise mean of log₁₀(ρ); same colormap / clim as member rows |
| `"std"` | Cell-wise std of log₁₀(ρ); separate sequential colormap (`cividis`) with its own colour scale |
| `"median"` | Cell-wise median of log₁₀(ρ); same colormap / clim as member rows |

NaN elements (air, missing) are excluded from all statistics.

#### Per-member output

When `ENS_PER_MEMBER = True` and `PLOT_ENS_FILE` is set, one additional
single-row figure is saved per member:

```
ensemble.pdf            ← joint (all members + stat rows)
ensemble_member0.pdf
ensemble_member1.pdf
…
```

### Ocean / air handling

| Variable | Default | Description |
|---|---|---|
| `OCEAN` | `None` | `None` = auto-infer; `True` / `False` = force |
| `AIR_RHO` | `1e9` Ω·m | Sentinel value for region 0 (air) |
| `OCEAN_RHO` | `0.25` Ω·m | Sentinel value for region 1 when treated as ocean |

Auto-inference: region 1 is treated as ocean when `flag == 1` **and**
ρ ≤ 1 Ω·m.  Override with `OCEAN = True / False` when the heuristic is
unreliable.

### Geographic / UTM origin

| Variable | Default | Description |
|---|---|---|
| `UTM_ORIGIN_LAT` | `-16.409` | Latitude of mesh centre (decimal °, + = N) |
| `UTM_ORIGIN_LON` | `-71.537` | Longitude of mesh centre (decimal °, + = E) |
| `UTM_ORIGIN_E` | `229047.0` | UTM easting of mesh centre (m) |
| `UTM_ORIGIN_N` | `8184127.0` | UTM northing of mesh centre (m) |
| `UTM_ZONE_OVERRIDE` | `None` | Force a specific zone number (1–60); `None` = auto |

The UTM zone is derived automatically from `UTM_ORIGIN_LON` using the
standard 6° band rule.  The hemisphere is inferred from the sign of
`UTM_ORIGIN_LAT`.

### Mesh-centre estimation from calibration sites

Identical to `femtic_mod_plot.py`.  Set `ESTIMATE_ORIGIN = True` and
populate `CALIBRATION_SITES`:

```python
ESTIMATE_ORIGIN = True

CALIBRATION_SITES = [
    dict(site=1,  crs="latlon", coords=[-71.500, -16.380]),  # [lon, lat]
    dict(site=10, crs="utm",    coords=[224500., 8179300.]),  # [E, N]
]
```

When `CALIBRATION_SITES` is empty and `SITE_DAT` is set, the bounding-box
midpoint of all sites is used (femticPY-compatible).  The result overwrites
`UTM_ORIGIN_E` / `UTM_ORIGIN_N` for the current run.

### Display coordinate system

| `DISPLAY_COORDS` | Axis ticks |
|---|---|
| `"model"` (default) | model-local metres, origin at mesh centre |
| `"utm"` | absolute UTM metres |
| `"latlon"` | decimal degrees |

### Site overlay

| Variable | Default | Description |
|---|---|---|
| `SITE_DAT` | `site.dat` | Primary source: `mt_make_sitelist.py` CSV; `None` = use observe.dat |
| `SITE_NAMES` | `None` | Site names to select; `None` = all sites in the file |
| `SITE_NUMBER` | `None` | Fallback: 1-based site index(es) from `observe.dat`; used when `SITE_DAT` is None |
| `SITE_MARKER` | `dict(marker="v", color="black", ms=8, …)` | Matplotlib marker kwargs for map panels |

### Output / figure

| Variable | Default | Description |
|---|---|---|
| `PLOT_DPI` | `300` | Figure DPI for saved files |
| `PLOT_CMAP` | `"turbo_r"` | Matplotlib colormap name |
| `PLOT_CLIM` | `[0.0, 4.0]` | Colour limits in log10(Ω·m); `None` = auto |
| `PLOT_OCEAN_COLOR` | `"lightgrey"` | Flat colour for ocean/lake cells; `None` = use colormap |
| `PLOT_AIR_BGCOLOR` | `None` | Axes facecolor for air; `None` = figure default |
| `PLOT_XLIM` | `[-20000., 20000.]` | Global easting limits (model-local m); `None` = auto |
| `PLOT_YLIM` | `[-20000., 20000.]` | Global northing limits (model-local m); `None` = auto |
| `PLOT_ZLIM` | `[-6000., 15000.]` | Global depth limits (model-local m); `None` = auto |
| `PLOT_EQUAL_ASPECT` | `True` | Equal aspect ratio on map and curtain panels (model / utm only) |

---

## Slice specification (`PLOT_SLICES`)

Identical format to `femtic_mod_plot.py`.  Four slice kinds:

| `kind` | Geometry | Required key |
|---|---|---|
| `"map"` | Horizontal at z = const | `z0` |
| `"ns"` | N-S curtain at x = const | `x0` |
| `"ew"` | E-W curtain at y = const | `y0` |
| `"plane"` | Arbitrary strike/dip plane | `point`, `strike`, `dip` |

Position keys accept plain floats (model-local metres) or `(value, "crs")`
tuples where `crs` is `"model"`, `"utm"`, or `"latlon"`.

```python
PLOT_SLICES = [
    dict(kind="map",  z0=5000.0),
    dict(kind="map",  z0=15000.0),
    dict(kind="ns",   x0=(-70.87, "latlon")),
    dict(kind="ew",   y0=(-16.35, "latlon")),
]
```

See `femtic_mod_plot.py` and its README for the full position CRS reference.

---

## Site position sources

### Site list file (primary)

`fem.read_site_dat` parses the comma-separated file written by
`mt_make_sitelist.py` with `WHAT_FOR="femtic"`.  Column order
(no header; `#` = comment):

```
name, lat, lon, elev, sitenum, easting, northing
```

Easting/northing are UTM metres converted to model-local metres via
`fem.utm_to_model(easting, northing, UTM_ORIGIN_E, UTM_ORIGIN_N)`.
Filter by name with `SITE_NAMES`; `None` = all sites.

### `observe.dat` fallback

Used when `SITE_DAT = None` and `SITE_NUMBER` is set.
`fem.read_site_position` scans observe.dat for a site-header line
(`int int float float`) matching the site number and returns
`(x_m, y_m)` in metres.

---

## Coordinate conversion

Identical to `femtic_mod_plot.py`.

```
lat/lon ──[util.latlon_to_utm_zn]──► (E_m, N_m)
                                          │
                              [fem.utm_to_model]
                                          │
                                          ▼
                   (E_m − UTM_ORIGIN_E,  N_m − UTM_ORIGIN_N)
                   = model-local (x_m, y_m)
```

Pure geographic conversions live in `util.py`; model-local conversions in
`femtic.py`.  Script-level wrappers (`_utm_zone_from_origin`,
`resolve_slices`, etc.) bind the module globals so callers need not repeat
origin/zone arguments.

---

## Air and ocean rendering

- **Air** (region 0): set to NaN by `prepare_rho_for_plotting`; the axes
  facecolor (`PLOT_AIR_BGCOLOR`) shows through transparent polygons.
- **Ocean / lake** (region 1 when active): rendered as a separate flat-colour
  layer at `PLOT_OCEAN_COLOR`.  Set `PLOT_OCEAN_COLOR = None` to route ocean
  cells through the colormap instead.

---

## Borehole resistivity logs (`PLOT_BOREHOLE`)

When `PLOT_BOREHOLE = True`, a 1-D log₁₀(ρ) vs depth figure is produced
after the ensemble plot.  The **first ensemble member** (`ENS_FILES[0]`) is
used as the reference model for the borehole log.

Point-in-element search delegates to `fem.extract_borehole_log` (lateral
bounding-box pre-filter + exact barycentric containment test).

### Configuration parameters

| Variable | Default | Description |
|---|---|---|
| `PLOT_BOREHOLE` | `False` | Enable / disable the borehole step |
| `BOREHOLE_FILE` | `ensemble_boreholes.pdf` | Output path; `None` → interactive show |
| `BOREHOLE_SITES` | `[]` | List of borehole spec dicts (see below) |
| `BOREHOLE_STYLE` | `dict(lw=1.2, marker="none")` | Matplotlib line kwargs for all traces |
| `BOREHOLE_XLIM` | `[0.0, 4.0]` | x-axis limits [log10(Ω·m)]; `None` = auto |
| `BOREHOLE_SHARED` | `True` | `True` = all on one axes; `False` = one panel per borehole |

### Borehole spec dict

| Key | Type | Description |
|---|---|---|
| `"name"` | str | Label shown in the legend / panel title |
| `"x"` | float or `(value, "crs")` | Borehole easting — same CRS tagging as `PLOT_SLICES` |
| `"y"` | float or `(value, "crs")` | Borehole northing — same CRS tagging |
| `"z_top"` | float | Start depth [m, FEMTIC z-down]; 0 = surface |
| `"z_bot"` | float | End depth [m, z-down] |
| `"dz"` | float | Sampling interval [m] |

### Example

```python
PLOT_BOREHOLE = True
BOREHOLE_FILE = WORK_DIR + "ensemble_boreholes.pdf"
BOREHOLE_SHARED = True
BOREHOLE_XLIM   = [0., 4.]

BOREHOLE_SITES = [
    dict(name="BH-centre", x=0.0, y=0.0,
         z_top=0., z_bot=20000., dz=200.),
    dict(name="BH-summit", x=(-70.87, "latlon"), y=(-16.35, "latlon"),
         z_top=0., z_bot=15000., dz=100.),
]
```

---

## Dependencies

| Package | Role |
|---|---|
| NumPy | Array operations |
| Matplotlib | Figure rendering |
| pyproj | `Transformer` for lat/lon → UTM (primary path; graceful built-in fallback when absent) |
| `femtic` (Py4MTX) | `utm_to_model`, `read_site_dat`, `read_site_position`, `estimate_utm_origin`, `extract_borehole_log`, `resolve_slice_positions` |
| `femtic_viz` (Py4MTX) | `read_femtic_mesh`, `read_resistivity_block`, `map_regions_to_element_rho`, `prepare_rho_for_plotting`, `plot_ensemble_slices` |
| `util` (Py4MTX) | `print_title`, `utm_zone_from_latlon`, `latlon_to_utm_zn`, `utm_to_latlon_zn` |
| `version` (Py4MTX) | `versionstrg` |

Environment variables `PY4MTX_ROOT` and `PY4MTX_DATA` must be set.

---

## Relation to other scripts

| Script | Relationship |
|---|---|
| `femtic_mod_plot.py` | Sister script for single-model slice plots; shares all config conventions, helper functions, and `PLOT_*` parameters |
| `snippets.py` | Snippet 1 (ensemble plot step) is the direct source of step (5) here |
| `femtic.py` | Provides `read_site_dat`, `read_site_position`, `estimate_utm_origin`, `extract_borehole_log`, and all model-local coordinate helpers |
| `femtic_viz.py` | Provides `plot_ensemble_slices` and all mesh / block I/O |
| `mt_make_sitelist.py` | Produces `SITE_DAT` (the sitelist CSV) |

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-05-24 | vrath / Claude Sonnet 4.6 | Created, based on `femtic_mod_plot.py` and `snippets.py` Snippet 1. `ENS_DIRS` + `BLOCK_PATTERN` + `ENS_ITER` replace the flat `ENS_FILES` list; glob expansion and label auto-derivation at runtime. Borehole step uses member 0 as reference model. |
