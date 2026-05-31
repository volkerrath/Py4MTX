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
   figure per member is saved alongside.

2. Optionally, a borehole resistivity log figure (point-in-element sampling,
   identical to step (7) in `femtic_mod_plot.py`).

---

## Workflow

```
ORIGIN_METHOD + SITE_DAT  →  UTM_ORIGIN_E/N/LAT/LON (bounding-box midpoint)
                          →  UTM zone auto-derived from origin lat/lon

ENS_DIRS + BLOCK_PATTERN + ENS_ITER
        |
        v  glob expand + sort → locate block file in each directory
   ENS_FILES  +  ENS_LABELS_resolved
        |
        v  fem.resolve_slice_positions(PLOT_SLICES)
   slice positions in model-local metres
        |
        v  fem.read_site_dat(SITE_DAT)  [or fem.read_site_position fallback]
   site_xys: (name, x_m, y_m, elev_m) per site
        |
        v  fviz.plot_ensemble_slices(...)
joint PDF  +  optional per-member PDFs
        |                                    [PLOT_BOREHOLE = True]
        v  fviz.plot_borehole_logs(...)
borehole PDF / interactive window
```

---

## Key configuration variables

### Paths
| Variable | Description |
|---|---|
| `WORK_DIR` | Base directory for all relative paths |
| `MESH_FILE` | `mesh.dat` |
| `OBSERVE_FILE` | `observe.dat` (fallback site source) |
| `SITE_DAT` | `site.dat` CSV from `mt_make_sitelist.py` |

### Ensemble input
| Variable | Description |
|---|---|
| `ENS_DIRS` | List of ensemble run directories (glob patterns accepted) |
| `BLOCK_PATTERN` | Filename pattern, e.g. `"resistivity_block_iter{iter}.dat"` |
| `ENS_ITER` | Iteration number substituted into `BLOCK_PATTERN` |
| `ENS_LABELS` | Row labels; `None` → directory basenames |
| `ENS_STAT_ROWS` | Summary rows: any subset of `["mean", "std", "median"]` |

### Origin estimation
| Variable | Description |
|---|---|
| `ORIGIN_METHOD` | `None` / `"box"` / `"average"` — how to derive origin from `SITE_DAT` |
| `UTM_ORIGIN_LAT/LON` | Fallback geographic origin (used when `ORIGIN_METHOD=None`) |
| `UTM_ORIGIN_E/N` | Fallback UTM origin in metres |
| `UTM_ZONE_OVERRIDE` | Force a specific UTM zone number; `None` = auto |

### Display and layout
| Variable | Description |
|---|---|
| `DISPLAY_COORDS` | `"model"` / `"utm"` / `"latlon"` |
| `DEPTH_KM` | `True` → depth axis in km |
| `HORIZ_KM` | `True` → horizontal axes in km |
| `PLOT_EQUAL_ASPECT` | Equal aspect ratio on all panels |
| `PLOT_PANEL_HEIGHT` | Panel height in cm |
| `PLOT_NROWS/NCOLS` | Grid layout (`None` = auto) |

### Slice geometry
| Variable | Description |
|---|---|
| `PLOT_SLICES` | List of slice dicts; `kind` = `"map"` / `"ns"` / `"ew"` / `"plane"` |
| `PLOT_XLIM/YLIM/ZLIM` | Axis limits in model-local metres |

### Site overlay
| Variable | Description |
|---|---|
| `SITE_NAMES` | Site filter; `None` = all |
| `PLOT_SITES_MAPS` | Show sites on map panels |
| `PLOT_SITES_SLICES` | Show sites on curtain panels |
| `PROJECTION_DIST` | Max distance (m) from slice plane for curtain projection |
| `SITE_MARKER` | Marker style dict for map panels |
| `SITE_MARKER_SLICES` | Marker style for curtain panels (`None` → same as `SITE_MARKER`) |
| `MAP_MARKERS` | Additional map markers (known features, etc.) |

---

## Slice specification (`PLOT_SLICES`)

Each entry is a dict with `kind` and the matching position key:

```python
dict(kind="map",   z0=5000.0)                        # horizontal map at 5 km depth
dict(kind="ew",    y0=(-16.35, "latlon"))             # E-W section at lat −16.35°
dict(kind="ns",    x0=(300000., "utm"))               # N-S section at UTM easting
dict(kind="plane", point=[0,0,5000], strike=45, dip=60)
```

Position values accept:
- plain `float` → model-local metres
- `(value, "utm")` → UTM metres (easting for `x0`, northing for `y0`)
- `(value, "latlon")` → longitude for `x0`, latitude for `y0`

---

## Changes from previous version

- `ESTIMATE_ORIGIN` / `CALIBRATION_SITES` / `UPDATE_CONFIG` replaced by
  `ORIGIN_METHOD` (`None` | `"box"` | `"average"`).  Origin estimation
  now runs **before** UTM zone derivation, fixing a `TypeError` when
  `UTM_ORIGIN_LAT/LON` are `None`.
- Local coordinate helpers removed; `fem.*` and `utl.*` called directly.
- `site_xys` tuples now carry elevation (`elev_m`) as fourth element.
- `plot_ensemble_slices` call extended with `site_xys`, `utm_origin_*`,
  `utm_zone`, `utm_northern`, `utm_to_latlon_fn`, `latlon_to_model_fn`,
  `display_coords`, `depth_km`, `horiz_km`, `equal_aspect`,
  `panel_height`, `nrows`, `ncols`, `projection_dist`,
  `sites_in_maps`, `sites_in_slices`, `site_marker_slices`,
  `map_markers`, `obs_coords_only` kwargs.
- Added `DEPTH_KM`, `HORIZ_KM`, `PLOT_EQUAL_ASPECT`, `PLOT_PANEL_HEIGHT`,
  `PLOT_NROWS`, `PLOT_NCOLS`, `PLOT_SITES_MAPS`, `PLOT_SITES_SLICES`,
  `SITE_MARKER_SLICES`, `MAP_MARKERS`, `DISPLAY_COORDS` config vars.
