# Tacna ModEM Magnetotelluric (MT) Pipeline

GMT/PyGMT-free pipeline for plotting log₁₀(ρ) results from a ModEM 3-D MT
inversion of the Tacna region (southern Peru) on a UTM Zone 19S
(EPSG:32719) grid, sharing the same basemap engine, styling conventions,
and settings layout as the seismic-tomography pipeline (`README_seis.md`).

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline

```
tacna_precompute_modem.py  →  UTM-km NetCDF/CSV files  →  tacna_plot_modem_image.py  →  figures
                                                        (or tacna_plot_modem_mesh.py for an
                                                         exact, unresampled mesh cut)
```

**`plotpy.py`** must sit alongside the scripts — it's a small shared
module of plotting helpers (`import plotpy`), also used by
`tacna_plot_seis.py` (see `README_seis.md`'s Pipeline section for the
full list of what it covers). Both ModEM plot scripts additionally use
its sensitivity-alpha helpers (`sens_shade_alpha`/`sens_data_alpha`).

### 1. `tacna_precompute_modem.py`

Reads a ModEM resistivity model (`MODEL_FILE` + `.rho`) and data file
(`DATA_FILE` + `.dat`), and writes UTM-km NetCDF/CSV outputs analogous to
`tacna_precompute_seis.py`'s:

| Output file                        | Contents                                   |
|--------------------------------------|---------------------------------------------|
| `modem_model_utm.nc`                 | Full 3-D log₁₀(ρ) model on the UTM-km mesh |
| `modem_sens_utm.nc`                  | Full 3-D sensitivity/resolution field (if `USE_SENSITIVITY` and the `.sns` file is found) |
| `modem_topo_utm.nc`                  | 2-D surface topography extracted from the model (shallowest non-air cell per column) |
| `modem_sites_utm.nc`                 | MT site positions + names |
| `modem_grid_edges_utm.nc`            | True cumulative cell-edge coordinates (for exact, unresampled mesh rendering) |
| `modem_rho_utm_{depth}km.nc`         | Horizontal log₁₀(ρ) slice, one per entry in `DEPTH_SLICES_KM` |
| `modem_sens_utm_{depth}km.nc`        | Matching sensitivity slice (optional) |

ModEM's local Cartesian mesh (origin = the reference point read from the
`.rho` file's last non-comment line) is projected to absolute UTM
easting/northing via that reference point and a UTM transformer — **the
geographic reference must come from the `> lat lon` header line, not from
treating the mesh's local metre offsets as degrees.**

**Key settings:**

- `OUTPUT_DIR` (default `"."`) — directory all `.nc` outputs above are
  written to (created automatically if it doesn't exist). Keep this in
  sync with `NC_DIR` in `tacna_plot_modem_image.py`/
  `tacna_plot_modem_mesh.py` so the plot scripts read from wherever
  precompute actually wrote to.
- `MODEL_FILE`/`MODEL_EXT`, `DATA_FILE`/`DATA_EXT` — ModEM input files.
- `USE_SENSITIVITY`, `SENS_FILE`/`SENS_EXT`, `SENS_TRANSFORM`,
  `SENS_FLIP_EASTING`/`SENS_FLIP_NORTHING` — optional sensitivity field
  (for shading/blanking poorly-resolved cells in the plot script).
- `REFERENCE_LAT`/`REFERENCE_LON` — override the georeferencing point
  (default: read from the model file).
- `DEPTH_SLICES_KM` — depths to export; **must match `DEPTH_SLICES_KM` in
  `tacna_plot_modem_image.py`/`tacna_plot_modem_mesh.py`**, since both the resistivity
  and sensitivity depth slices are written from this one list.
- `TRIM_PAD` — padding cells dropped from each mesh face before output
  (ModEM padding cells grow geometrically toward the boundary, so this
  alone usually still leaves a domain far larger than the area of
  interest).
- `CROP_TO_REGION` (default `True`) — crop the trimmed grid further to
  `TAR_LON`/`TAR_LAT` before any output is written; `False` keeps the full
  trimmed extent. Same toggle/convention as `CROP_TO_REGION` in
  `tacna_precompute_seis.py`.
- `TAR_LON` / `TAR_LAT` — geographic crop box. **Kept identical to
  `TAR_LON`/`TAR_LAT` in `tacna_precompute_seis.py`** so both pipelines
  cover the same ground — currently the union of the two pipelines'
  original boxes, padded by ~0.05°. Must fully contain every `VSLICES`
  profile endpoint defined in `tacna_plot_modem_image.py`, or you'll get a
  silent no-data gap at the edge of a cross-section.
- `UTM_ZONE`/`UTM_HEMI` — manual UTM zone/hemisphere override (default:
  inferred from the reference longitude/latitude).

### 2. `tacna_plot_modem_image.py`

Reads the files above and produces the same two figure kinds as the seis
pipeline — horizontal log₁₀(ρ) depth-slice maps and arbitrary vertical
cross-sections — using **resampled** (interpolated or nearest-cell)
values on a regular sampling grid along each profile.

`tacna_plot_modem_mesh.py` is a companion script that instead renders the exact
ModEM mesh: every rendered patch is one real, unblended mesh cell at its
true position and size (`pcolormesh(..., shading="flat")` against
`modem_grid_edges_utm.nc`'s true cell-edge geometry, both for depth slices
and for the sequence of cells a profile actually crosses on a section) —
use it when the resampling/interpolation `tacna_plot_modem_image.py` applies
would misrepresent the mesh's real (non-uniform) cell geometry.

**Input/output directories:**

- `NC_DIR` (default `"."`) — directory to read precomputed NetCDF files
  from (`modem_model_utm.nc`, `NC_TOPO_MODEM`, per-depth slices, etc.).
  Must match `OUTPUT_DIR` in `tacna_precompute_modem.py`. Same setting,
  same behaviour, in both `tacna_plot_modem_image.py` and
  `tacna_plot_modem_mesh.py`.
- `PLOT_DIR` (default `"."`) — directory saved figures are written to
  (created automatically if it doesn't exist). Same in both plot scripts.
- `PLOT_FILENAME_SUFFIX` — appended to every saved figure's filename
  before the extension, so output from the two scripts never collides
  and is distinguishable at a glance: `"_img"` in
  `tacna_plot_modem_image.py` (resampled rendering), `"_msh"` in
  `tacna_plot_modem_mesh.py` (exact-mesh rendering), e.g.
  `modem_rho_1km_tacna_img.pdf` vs `modem_rho_1km_tacna_msh.pdf`. Set to
  `""` to disable.

**Map region & extent:**

- `REGION_SOURCE` — `"model"` (resistivity-grid extent, default) or
  `"topo"` (topo-grid extent, wider), combined with `REGION_MARGIN_KM`.
- `MAP_XLIM` / `MAP_YLIM` (UTM km, default `None`) — explicit override of
  the displayed map extent, applied *after* `REGION_SOURCE` computes the
  region — same mechanism as in `tacna_plot_seis.py`. Map axes are
  already in UTM km, driven off `modem_rho_utm_{depth}km.nc`'s own
  `easting`/`northing` coordinates.
- Maps always render at exact equal x/y scale (1 km in easting = 1 km in
  northing on the page), regardless of `MAP_XLIM`/`MAP_YLIM` or whether a
  colorbar is shown — guaranteed *by construction*: `create_map_figure()`
  places the map axes at an explicit, physically-computed size in inches
  matching the region's own aspect ratio, rather than relying on
  matplotlib's `ax.set_aspect("equal")` plus automatic colorbar
  space-stealing (which can desync from the actual rendered box). See
  `FIG_WIDTH` and the colorbar settings below. Same in both
  `tacna_plot_modem_image.py` and `tacna_plot_modem_mesh.py`.
- `AXES_UNITS` (`"km"` default, or `"latlon"`) — selects what the map's
  bottom/left tick labels show: UTM easting/northing in km, or longitude/
  latitude in degrees. It's an in-place relabelling of the primary axes
  (one unit system at a time), not a secondary overlay axis. Same
  parameter block, same behaviour, in all three plot scripts.
  `LATLON_NTICKS`/`LATLON_DECIMALS` control the lon/lat tick density and
  precision when `AXES_UNITS = "latlon"`. `AXES_KM_COMMA` (default
  `True`) adds a thousands comma to km-axis tick labels (American style,
  e.g. `8,000`) when `AXES_UNITS = "km"`; set `False` for plain numbers
  (`8000`). No effect when `AXES_UNITS = "latlon"`.

**Vertical sections (`VSLICES`):** same structure as the seis pipeline
(`name`, `p1`/`p2`, `coord`, `zmin_km`/`zmax_km`, `npts`/`nz`,
`swath_km`, optional per-slice `xlim`/`ylim`). `VSLICE_X_AXIS` switches
between `"utm"` and `"distance"`, exactly as in `tacna_plot_seis.py`.
`VSLICE_INTERP_METHOD` (`"nearest"` — true unblended cell values,
default — or `"linear"` — smoothed trilinear interpolation) controls how
the 3-D model is sampled onto a section's profile points in
`tacna_plot_modem_image.py` (not applicable to `tacna_plot_modem_mesh.py`, which
always cuts the exact mesh).

**Topography on sections:** the surface line is always drawn
(`VSLICE_TOPO_STYLE`), positioned from the section's own data
(`surf_depth`, from the model's air-cell mask) rather than a separately
referenced topography raster. Only an ocean fill is available (elevation
≤ 0, a genuine physical reference), bounded at `z = 0`; there is no land
fill for the ModEM sections.

**Vertical exaggeration:** `VSLICE_VE` (2.0 by default — MT structures are
usually flatter than a true-scale section shows). The "VE = …×" label is
drawn *before* the colour image (low z-order), positioned via
`VSLICE_VE_POS` (`"lower right"` default, or `"lower left"`/`"upper
left"`/`"upper right"`, or an explicit `(x, y, ha, va)` tuple) and styled
via `VSLICE_VE_STYLE` (default black). `VSLICE_EQUAL_SCALE` (default
`False`) overrides `VSLICE_VE` with `1.0` whenever `True`, forcing true
1:1 x/y (km) scale — off by default because real profiles are usually
much longer than they are deep, so a literal equal scale usually isn't
what you want day-to-day; this flag is for the occasional figure where
undistorted scale actually matters (e.g. comparing directly against a
map). Sections are built the same way as maps —
`create_section_figure()` places the panel and colorbar via explicit
inch-based axes placement, not `tight_layout()` plus a space-stealing
colorbar — which matters even more here, since that older approach could
produce a badly broken/overlapping layout specifically for the
wide-short panel shape a real profile tends to have. Same in all three
plot scripts — `tacna_plot_modem_mesh.py` in particular was missing
`VSLICE_VE_POS`/`VSLICE_VE_STYLE` (the VE label was drawn at a fixed
position/style) until this pass; it now matches the other two.

**Free-text annotation:** `ANNOTATION_TEXT` (default `None`), same
mechanism as `tacna_plot_seis.py`.

**Sensitivity shading/blanking** (only meaningful if precompute wrote a
sensitivity field): `USE_SENSITIVITY`, `SENS_BLANK_THRESHOLD` (blank
poorly-resolved cells to NaN), `SENS_SHADE_RANGE`/`SENS_SHADE_COLOR`/
`SENS_SHADE_MAX_ALPHA` (overlay a fading shade), `SENS_ALPHA_RANGE` (fade
the data layer itself toward transparent in poorly-resolved cells).

**Map feature layers:** `SHOW_PROFILE_LINES`, `SHOW_VSLICE_LINES`,
`SHOW_SEISMICITY`, `SHOW_MT_SITES`, `SHOW_VOLCANOES`,
`SHOW_VOLCANOES_ACTIVE`, `SHOW_CITIES`, `SHOW_NORTH_ARROW` — one boolean
per overlay layer, all default `True`. `SHOW_SEISMICITY`/`SHOW_MT_SITES`
also control the matching seismicity/MT-site scatter on vertical sections
(`VSLICE_EQ_STYLE`/`VSLICE_MT_STYLE`), so turning a layer off applies
everywhere it would otherwise appear, not just on the map. Same flags,
same behaviour, in all three plot scripts (`tacna_plot_seis.py`,
`tacna_plot_modem_image.py`, `tacna_plot_modem_mesh.py`).

Seismicity on maps can additionally be depth-filtered per slice via
`ZMIN_SEISM`/`ZMAX_SEISM` (km, one `(zmin, zmax)` pair per entry in
`DEPTH_SLICES_KM` — `None` in either slot means unbounded, i.e. no filter
on that side). All three lists must be the same length, or the script
exits with an explanatory error rather than silently mis-indexing — this
used to fail silently in `tacna_plot_modem_image.py` (stale,
longer-than-needed lists left over from an earlier `DEPTH_SLICES_KM`),
which was why its maps showed different seismicity than
`tacna_plot_modem_mesh.py`'s despite both reading the identical catalog;
fixed and now guarded the same way in all three scripts.

Volcano labels: `VOLC_LABEL_FULL_NAME` (default `False`) switches between
`VOLC_NAME_COL_FULL` (`"NAME"`) and `VOLC_NAME_COL_SHORT` (`"VOLCAN2"`)
in `volcanes.csv`; falls back to the short column with a warning if the
full-name column isn't present. City labels always use `cities.csv`'s
`Name` column — its only name field, and already the full city name — so
there's no separate full/short toggle for cities. Volcano and city labels
are plain black text (`VOLC_LABEL_STYLE`/`CITY_LABEL_STYLE`) with no
stroke/halo effect.

**Figure size & colorbar:**

- `FIG_WIDTH` (cm, default `10.0`) — controls only the map panel's
  width; height is always derived from it and the region's own aspect
  ratio. There's no manual height override — this is what makes the
  equal-scale guarantee above unconditional. (`VSLICE_WIDTH_CM`/
  `VSLICE_HEIGHT_CM` are separate and unaffected — cross-sections keep a
  settable height since `VSLICE_VE` deliberately makes them non-square.)
  Same in both `tacna_plot_modem_image.py` and `tacna_plot_modem_mesh.py`
  — the latter also applies it to its optional standalone sensitivity
  map (`PLOT_SENSITIVITY_MAPS`).
- `SHOW_COLORBAR` (default `True`) — set `False` to omit the colorbar
  entirely; the map panel itself is completely unaffected either way.
- `COLORBAR_POSITION` (`"right"` default, or `"left"`/`"bottom"`/
  `"top"`) — the colorbar is added as *extra* width (right/left) or
  height (bottom/top) beyond the map panel, so it never competes with
  the map for space and can never distort it.
- `COLORBAR_SIZE` (default `0.85`) — bar length, as a fraction of the
  map edge it's attached to. (Previously this was matplotlib's
  `fraction` parameter — bar *thickness* relative to the map, default
  `0.05` — the meaning changed along with the switch to explicit-axes
  placement; if you had a custom value, it needs rethinking under the
  new meaning.)
- `COLORBAR_ASPECT` (default `20`) — bar length ÷ bar thickness;
  thickness is derived from this and `COLORBAR_SIZE`.
- `COLORBAR_PAD` (inches), `COLORBAR_LABEL_SIZE`, `COLORBAR_TICK_SIZE`,
  `COLORBAR_NTICKS` — unchanged.

**Other notable settings:** `PLOT_FORMATS`/`PLOT_DPI`,
`CMIN_RHO`/`CMAX_RHO`/`CMAP_RHO` (log₁₀(Ω·m) colour scale), `NC_TOPO_SEIS`
(reuse the higher-resolution seis-pipeline topo instead of the
ModEM-derived one), `NC_BATH` (reuse ocean fill from the seis pipeline),
and the `*_MARKER_STYLE`/`*_LABEL_STYLE` dicts for every overlay
(seismicity, MT sites, volcanoes, cities, profile lines, north arrow —
MT sites also get a `VSLICE_MT_STYLE` for cross-section projection).

---

## Coordinate convention

All grids and figures use **UTM Zone 19S (EPSG:32719)**, distances in km.
Depth is km, positive down; `z = 0` is sea level / the top of the model
(the ModEM mesh's z=0 face).

## Dependencies

```
numpy, matplotlib, xarray, pandas, pyproj, scipy
```
plus the local `modem.py` helper library (`read_mod`, `read_data`,
`cells3d`, `get_topo`).

## Typical run

```bash
python3 tacna_precompute_modem.py   # writes *.nc/*.csv into the working directory
python3 tacna_plot_modem_image.py         # reads them, writes figures (resampled sections)
# or, for an exact unresampled mesh cut:
python3 tacna_plot_modem_mesh.py
```

Run precompute again whenever the `.rho`/`.dat` files, `TRIM_PAD`,
`CROP_TO_REGION`/`TAR_LON`/`TAR_LAT`, or `DEPTH_SLICES_KM` change.
Everything else (colours, styling, crop views, profile definitions,
annotations, sensitivity shading) only needs re-running the plot script.
