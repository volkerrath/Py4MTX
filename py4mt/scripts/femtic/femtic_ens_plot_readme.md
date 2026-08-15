# femtic_ens_plot.py

Ensemble slice plot for a set of FEMTIC inversion runs, with optional
borehole resistivity logs.

---

## Purpose

`femtic_ens_plot.py` discovers converged ensemble members exactly the way
`femtic_ens_post.py` does — scanning `ENSEMBLE_DIR` for `ENSEMBLE_NAME*`
sub-directories and filtering on `femtic.cnv` / `NRMS_MAX` — and produces:

1. **[default, `PER_MEMBER_PLOT=True`]** Per converged member, two
   `fviz.plot_model_slices()` figures in their own sub-directory
   `WORK_DIR/<label>/`, using the same slice geometry and `PLOT_*`
   parameters as `femtic_mod_plot.py`:
   - `iter0.<ext>` — the perturbed prior model
     (`resistivity_block_iter0.dat`)
   - `best.<ext>` — the best-fit model
     (`resistivity_block_iter{numit}.dat`, `numit` from `femtic.cnv`,
     with an `nRMS = ...` annotation on the figure)

   One file pair per entry in `PLOT_FORMAT` (e.g. `["pdf", "jpg"]`
   writes both a vector and a raster version — the figure is rebuilt
   once per format). When `"pdf"` is among the formats and
   `PER_MEMBER_PDF_CATALOG=True` (default), every per-member pdf figure
   is *also* collected, in plot order, into one multi-page catalog PDF
   via `matplotlib.backends.backend_pdf.PdfPages`, written to
   `PER_MEMBER_CATALOG_FILE`. This is in addition to, not instead of,
   the individual per-member files.

2. **[optional, `PLOT_JOINT=True`]** The previous joint multi-row figure —
   one row per member's best-fit model — via `fviz.plot_ensemble_slices()`,
   with optional statistical summary rows (mean, std, median of
   log₁₀(ρ) across all members).
   > **Known limitation:** this call currently passes several kwargs
   > (`site_xys`, CRS/display options, layout options, …) that are not
   > present in `plot_ensemble_slices()`'s current signature and will
   > raise `TypeError`. This pre-existing mismatch is unresolved — see
   > the code comment at the call site. `PLOT_JOINT` defaults to `False`
   > so it doesn't affect normal use.

3. Optionally, a borehole resistivity log figure (point-in-element sampling,
   identical to step (7) in `femtic_mod_plot.py`), sampled from the first
   converged member's best-fit model.

---

## Workflow

```
ORIGIN_METHOD + SITE_DAT  →  UTM_ORIGIN_E/N/LAT/LON (bounding-box midpoint)
                          →  UTM zone auto-derived from origin lat/lon

ENSEMBLE_DIR + ENSEMBLE_NAME + NRMS_MAX + FEMTIC
        |
        v  utl.get_filelist() → per-dir femtic.cnv → numit, nRMS
   model_list: [{label, dir, numit, nrms, iter0_file, best_file}, ...]
   (converged members only — same set femtic_ens_post.py includes)
        |
        v  fem.resolve_slice_positions(PLOT_SLICES)
   slice positions in model-local metres
        |
        v  fem.read_site_dat(SITE_DAT)  [or fem.read_site_position fallback]
   site_xys: (name, x_m, y_m, elev_m) per site
        |
        v  fviz.plot_model_slices(...)  × 2 per member × len(PLOT_FORMAT) [PER_MEMBER_PLOT]
WORK_DIR/<label>/iter0.<ext>  +  WORK_DIR/<label>/best.<ext>   (per converged member)
        |                                    ["pdf" in PLOT_FORMAT, PER_MEMBER_PDF_CATALOG]
        v  PdfPages(PER_MEMBER_CATALOG_FILE)
one multi-page pdf catalog, all member pdf figures in plot order
        |
        v  fviz.plot_ensemble_slices(...)                [PLOT_JOINT]
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

### Ensemble input — converged-member discovery
| Variable | Description |
|---|---|
| `FEMTIC` | FEMTIC version (`"4.3"` / `"5.0"`) — selects the `femtic.cnv` nRMS column; must match `femtic_ens_post.py` |
| `ENSEMBLE_DIR` | Directory containing one sub-directory per member |
| `ENSEMBLE_NAME` | Member sub-directories matched via `"<ENSEMBLE_NAME>*"` |
| `NRMS_MAX` | Max accepted nRMS from `femtic.cnv` — keep equal to `femtic_ens_post.py`'s value |
| `ENS_LABELS` | Labels for member plots/filenames; `None` → directory basenames |

### Per-member plots (default)
| Variable | Description |
|---|---|
| `PER_MEMBER_PLOT` | If `True` (default), plot iter0 + best-fit figures for every converged member |
| `PLOT_FORMAT` | Output format(s), e.g. `"pdf"` or `["pdf", "jpg"]` — normalised to `_PLOT_FORMATS` |
| `PER_MEMBER_PDF_CATALOG` | If `True` (default) and `"pdf"` is among `_PLOT_FORMATS`, combine every per-member pdf figure into one multi-page catalog |
| `PER_MEMBER_CATALOG_FILE` | Output path for the multi-page pdf catalog |

### Joint ensemble figure (optional extra)
| Variable | Description |
|---|---|
| `PLOT_JOINT` | If `True`, additionally build the joint multi-row figure (see Known limitation above) |
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
- **2026-08-13 (Claude Sonnet 5, Anthropic):** Added `femtic_ens_plot_summary.md`
  output at end of run: user-set (UPPERCASE) parameters, script path, and
  run date/time.
- **2026-08-14 (Claude Sonnet 5, Anthropic):** Replaced `ENS_DIRS` /
  `BLOCK_PATTERN` / `ENS_ITER` with `ENSEMBLE_DIR` / `ENSEMBLE_NAME` /
  `NRMS_MAX` / `FEMTIC` — converged members are now discovered the same
  way `femtic_ens_post.py` does, via `femtic.cnv`. Default behaviour is
  now two per-member figures (`PER_MEMBER_PLOT=True`): perturbed prior
  (`iter0`) and best-fit (`iterX`, `numit` from `femtic.cnv`), each via
  `fviz.plot_model_slices()`. The former joint multi-row figure is kept
  as an optional extra (`PLOT_JOINT=False` by default); its
  `plot_ensemble_slices()` call still has the previously flagged keyword
  mismatch and will raise `TypeError` if enabled — unresolved, out of
  scope for this change. Added `PER_MEMBER_PLOT`, `PER_MEMBER_FORMAT`,
  `PLOT_JOINT` config vars.
- **2026-08-15 (Claude Sonnet 5, Anthropic):** `PER_MEMBER_FORMAT`
  replaced by `PLOT_FORMAT`, which now accepts a list (e.g.
  `["pdf", "jpg"]`, normalised to `_PLOT_FORMATS`), mirroring
  `MOD_PLOT_FORMAT` in `femtic_ens_post.py`. Per-member files moved into
  their own sub-directory `WORK_DIR/<label>/` as `iter0.<ext>` /
  `best.<ext>` (previously the flat `<label>_iter0.<ext>` naming in
  `WORK_DIR`). Added `PER_MEMBER_PDF_CATALOG` (default `True`) and
  `PER_MEMBER_CATALOG_FILE`: when `"pdf"` is among `_PLOT_FORMATS`,
  every per-member pdf figure is additionally combined into one
  multi-page catalog via `matplotlib.backends.backend_pdf.PdfPages`.
