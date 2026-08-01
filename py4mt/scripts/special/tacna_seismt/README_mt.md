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
tacna_precompute.py (Part A)  →  UTM-km NetCDF files  →  tacna_plot_modem_image.py  →  figures
                                                        (or tacna_plot_modem_mesh.py for an
                                                         exact, unresampled mesh cut)
```

**`tacna_precompute_modem.py` and `tacna_precompute_seis.py` have been
merged into a single `tacna_precompute.py`** — Part A (ModEM/MT, this
README) and Part B (seismic tomography, `README_seis.md`) now live in one
script, one run, because they share region-of-interest settings
(`OUTPUT_DIR`/`CROP_TO_REGION`/`TAR_LON`/`TAR_LAT` — see SHARED SETTINGS in
the script) and because Part A now also resamples resistivity directly
onto Part B's grid (see "MT resistivity on the seismic grid" below). All
of Part A's own native-ModEM-mesh outputs are unchanged and still written
exactly as before.

**`plotpy.py`** must sit alongside the scripts — it's a small shared
module of plotting helpers (`import plotpy`), also used by
`tacna_plot_seis.py` (see `README_seis.md`'s Pipeline section for the
full list of what it covers). Both ModEM plot scripts additionally use
its sensitivity-alpha helpers (`sens_shade_alpha`/`sens_data_alpha`).

### 1. `tacna_precompute.py` — Part A (ModEM / MT)

Reads a ModEM resistivity model (`MODEL_FILE` + `.rho`) and data file
(`DATA_FILE` + `.dat`), and writes UTM-km NetCDF outputs analogous to Part
B's (`README_seis.md`):

| Output file                        | Contents                                   |
|--------------------------------------|---------------------------------------------|
| `modem_model_utm.nc`                 | Full 3-D log₁₀(ρ) model on the native UTM-km ModEM mesh |
| `modem_sens_utm.nc`                  | Full 3-D sensitivity/resolution field (if `USE_SENSITIVITY` and the `.sns` file is found) |
| `modem_topo_utm.nc`                  | 2-D surface topography extracted from the model (shallowest non-air cell per column) |
| `modem_sites_utm.nc`                 | MT site positions + names |
| `modem_grid_edges_utm.nc`            | True cumulative cell-edge coordinates (for exact, unresampled mesh rendering) |
| `modem_rho_utm_{depth}km.nc`         | Horizontal log₁₀(ρ) slice, one per entry in `DEPTH_SLICES_KM`, on the native ModEM mesh |
| `modem_sens_utm_{depth}km.nc`        | Matching sensitivity slice (optional), on the native ModEM mesh |
| `modem_submesh_points.nc`            | Full native ModEM submesh, flattened to one row per cell (easting/northing/depth + resistivity + sensitivity) — for `tacna_cluster.py` (`README_cluster.md`) |

ModEM's local Cartesian mesh (origin = the reference point read from the
`.rho` file's last non-comment line) is projected to absolute UTM
easting/northing via that reference point and a UTM transformer — **the
geographic reference must come from the `> lat lon` header line, not from
treating the mesh's local metre offsets as degrees.**

**No more MT-onto-seismic-grid resampling here.** Earlier versions of this
script resampled resistivity onto Part B's seismic Vp/Vs/density grid
directly in precompute (`modem_rho_on_seisgrid*.nc`, toggled by
`MT_TO_SEIS_GRID`). That step has moved to `tacna_cluster.py`, which now
RBF-interpolates resistivity/conductivity — read straight from
`modem_submesh_points.nc` below — onto a jointly-defined regular grid,
rather than onto the seismic tomography's own grid specifically. See
`README_cluster.md`.

**Key settings:**

- `OUTPUT_DIR` (default `"."`, both parts share this one setting now that
  precompute is a single script) — directory all `.nc` outputs above are
  written to (created automatically if it doesn't exist). Keep this in
  sync with `NC_DIR` in `tacna_plot_modem_image.py`/
  `tacna_plot_modem_mesh.py` so the plot scripts read from wherever
  precompute actually wrote to.
- `MODEL_FILE`/`MODEL_EXT`, `DATA_FILE`/`DATA_EXT` — ModEM input files.
- `USE_SENSITIVITY`, `SENS_FILE`/`SENS_EXT`, `SENS_TRANSFORM`,
  `SENS_FLIP_EASTING`/`SENS_FLIP_NORTHING` — optional sensitivity field
  (for shading/blanking poorly-resolved cells in the plot script, and now
  also included in `modem_submesh_points.nc` if enabled).
- `REFERENCE_LAT`/`REFERENCE_LON` — override the georeferencing point
  (default: read from the model file).
- `DEPTH_SLICES_KM` — depths to export **on the native ModEM mesh**; must
  match `DEPTH_SLICES_KM` in `tacna_plot_modem_image.py`/
  `tacna_plot_modem_mesh.py`, since both the resistivity and sensitivity
  depth slices are written from this one list.
- `EXPORT_SUBMESH_TABLE` (default `True`) — write `modem_submesh_points.nc`
  (see above); `False` skips it (`tacna_cluster.py` needs this file for
  its `"rho"`/`"cond"`/`"sens"` variables).
- `TRIM_PAD` — padding cells dropped from each mesh face before output
  (ModEM padding cells grow geometrically toward the boundary, so this
  alone usually still leaves a domain far larger than the area of
  interest).
- `CROP_TO_REGION` (default `True`, **shared** with Part B — one setting
  now, not two kept in sync) — crop the trimmed grid further to
  `TAR_LON`/`TAR_LAT` before any output is written; `False` keeps the full
  trimmed extent.
- `TAR_LON` / `TAR_LAT` (**shared** with Part B) — geographic crop box,
  currently the union of the two pipelines' original boxes, padded by
  ~0.05°. Must fully contain every `VSLICES` profile endpoint defined in
  `tacna_plot_modem_image.py`, or you'll get a silent no-data gap at the
  edge of a cross-section.
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
  Must match `OUTPUT_DIR` in `tacna_precompute.py`. Same setting,
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

**Map vs. section opacity:** depth-slice maps render the resistivity
layer at `1 - ALPHA_RHO` opacity by default, so the topography hillshade
basemap underneath shows through. Sections have no basemap underneath, so
they default to fully opaque (`ALPHA_RHO` doesn't apply there) —
`SENS_ALPHA_RANGE`, if set, is the only thing that fades a section's data
layer, to let the topo fill/line show through in poorly-resolved cells.
Same in both `tacna_plot_modem_image.py` and `tacna_plot_modem_mesh.py`.

**Map feature layers:** `SHOW_PROFILE_LINES`, `SHOW_VSLICE_LINES`,
`SHOW_SEISMICITY`, `SHOW_MT_SITES`, `SHOW_SEISMIC_SITES`,
`SHOW_VOLCANOES`, `SHOW_VOLCANOES_ACTIVE`, `SHOW_CITIES`,
`SHOW_NORTH_ARROW` — one boolean per overlay layer, all default `True`.
`SHOW_SEISMICITY`/`SHOW_MT_SITES` also control the matching
seismicity/MT-site scatter on vertical sections (`VSLICE_EQ_STYLE`/
`VSLICE_MT_STYLE`), so turning a layer off applies everywhere it would
otherwise appear, not just on the map. Same flags, same behaviour, in all
three plot scripts (`tacna_plot_seis.py`, `tacna_plot_modem_image.py`,
`tacna_plot_modem_mesh.py`).

**Marker sizing is linear (diameter), not area.** Every marker —
seismicity, MT sites, seismic sites, volcanoes, cities, both on maps and
on vertical sections — is drawn with `plotpy.clipped_markers`/
`plotpy.markers` (`ax.plot()`-based). A style dict's `s=18` means an
18 pt marker, not `ax.scatter()`'s 18 pt² area (which works out to a
much smaller ~4.8 pt diameter). There are no `ax.scatter()` calls
anywhere in any of the three plot scripts.

**Label text — full name, truncated, or none:** every `*_LABEL_STYLE`
dict (`VOLC_LABEL_STYLE`, `CITY_LABEL_STYLE`, `MT_LABEL_STYLE`) accepts
a `mode` key controlling how much of the feature's name is drawn as
text — the marker itself is unaffected either way:
- `"full"` — the complete name (default for volcanoes/cities)
- `"none"` — marker only, no text (default for MT sites — station
  names are usually too dense to label cleanly at map scale)
- `"firstN"` / `"lastN"` — first/last N characters, e.g. `"first3"`

**MT site labels** (`MT_LABEL_STYLE`, new) — off (`mode="none"`) by
default; switch to `"full"`/`"firstN"`/`"lastN"` to turn on. Defaults to
small, vertical text (`fontsize=5, rotation=90`) so close-together
station names don't overlap horizontally. Names come from
`modem_sites_utm.nc`'s `"name"` variable in `tacna_plot_modem_image.py`/
`tacna_plot_modem_mesh.py`.

**Seismic sites** (`CSV_SEISMIC_SITES`, default
`../features/seismic_sites.csv`) — seismometer station locations, a
plain CSV with **no header row**: columns are `network`, `station`,
`lat`, `lon`, `elev_m`. Default marker: filled green square, size 18
(`SEISMIC_SITES_MARKER_STYLE`). Map-only (no vertical-section
projection, unlike seismicity/MT sites) and not currently labelled with
station names — just the marker.

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
stroke/halo effect. `SHOW_VOLC_LABELS`/`SHOW_CITY_LABELS` (default
`True`) switch the name text off independently of `SHOW_VOLCANOES`/
`SHOW_CITIES` — set either to `False` to keep the marker with no label.

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
- **`VSLICE_WIDTH_CM` is the panel width of the *longest* profile in
  `VSLICES`** — every other profile's panel is proportionally narrower,
  scaled by its own length relative to that longest one, rather than
  every profile independently filling the same fixed width regardless of
  how long it actually is. This means every section in a run shares the
  exact same horizontal km-per-cm scale (and, since `VSLICE_VE`/depth
  range are normally the same across profiles, ends up the same height
  too) — so a short profile and a long profile drawn in the same run are
  now directly comparable at a glance, rather than the short one looking
  artificially "zoomed in" (visibly larger fonts/markers/colorbar
  relative to the data) purely because it was stretched to the same
  width as a much longer profile. Same behaviour, in all three plot
  scripts.
- Section depth range (and so the derived figure height) for a given
  `VSLICES` entry matches exactly between `tacna_plot_modem_image.py` and
  `tacna_plot_modem_mesh.py`: both clip to exactly `zmin_km`/`zmax_km`.
  `mesh` still selects real mesh cell edges internally (one cell of
  padding on each side, so a boundary cell isn't sliced in the middle of
  a real interior edge), but then truncates only the outermost edge of
  those two boundary cells to `zmin_km`/`zmax_km` — each boundary cell
  keeps its real value, just displayed clipped at the requested window
  (the same "keep the value, clip the display" approach
  `tacna_precompute.py` (Part A) uses for `CROP_TO_REGION`). The topography
  surface line/fill (`surf_depth`) is computed from those same cells'
  *true, unclamped* edges rather than the window-clamped ones — using
  the clamped edges there would flatten a segment's surface to exactly
  `zmin_km` whenever its shallowest valid cell happened to be one of the
  two clamped boundary cells, producing a hard, artificial flat plateau
  instead of the model's real (if blocky) topography.
- `VSLICE_SURFACE_MIN_RUN` (default `1`, i.e. off) — optionally require
  at least this many consecutive valid (non-air) cells, going deeper,
  before accepting a column's shallowest valid cell as its surface,
  rather than trusting that one cell on its own. Off by default: a
  cell's own air/rock classification is already a simple, direct fact
  (`log10(ρ) >= AIR_LOG10_RHO_THRESHOLD`), and this project has already
  been through enough guessed-at heuristics layered on top of it — turn
  it on only if you've confirmed (e.g. via `VSLICE_PRINT_SURFACE_CELLS`
  below) that a genuinely air-classified single cell is being read in
  isolation. Same setting, same behaviour, in both
  `tacna_plot_modem_image.py` and `tacna_plot_modem_mesh.py`.
- `tacna_plot_modem_mesh.py` now also checks, at load time, that
  `modem_grid_edges_utm.nc` and `modem_model_utm.nc` actually describe
  the *same* cells — every model cell centre must fall inside its
  supposedly matching edge interval — not just that their cell *counts*
  agree (which was already checked, but two files with matching shapes
  can still be shifted or cropped differently). If the two files were
  written by precompute runs with different `TRIM_PAD`/`CROP_TO_REGION`/
  `TAR_LON`/`TAR_LAT` (e.g. a settings change followed by only a partial
  re-run), this now exits with a clear message instead of silently
  letting `_locate_ij()` pick the wrong cell for some profile segments —
  which would look exactly like an isolated, flat-topped topography
  artifact, since neighbouring segments (still landing on a consistent,
  if wrong, cell) look fine. If you hit this, re-run
  `tacna_precompute.py` fully in one pass. This class of bug isn't
  possible in `tacna_plot_modem_image.py`, which only ever interpolates
  within one file's own self-consistent coordinate axes rather than
  combining two independently-computed files.
- `VSLICE_PRINT_SURFACE_CELLS` (default `True`, `tacna_plot_modem_mesh.py`
  only) — **diagnostic only, doesn't change what's plotted.** Prints,
  for every profile segment, its `(i, j)` cell, position, detected
  `surf_depth`, and the raw `log10(ρ)` values (with their air/rock
  classification against `AIR_LOG10_RHO_THRESHOLD`) at the cell
  `surf_depth` came from. No detection logic, no thresholds to tune —
  just the facts a column's classification is based on, to read
  directly rather than guess at. Set `False` once you're done
  inspecting it (verbose — one line or so per segment).
- **Everything above the model's real extent renders as plain white
  space, not a colour artifact.** A `.rho` file only contains cells up
  to the model's own real top edge (e.g. up to the maximum topography —
  nothing above that exists in the file at all, by construction);
  `VSLICES` entries commonly request more headroom than that
  (`zmin_km = -8` while the model's own top edge might be at, say,
  `-4.9`). `tacna_plot_modem_mesh.py`'s boundary-edge display clamp is
  now conditional on the model's real data actually reaching past the
  requested window — previously it was unconditional, stretching that
  boundary cell's displayed thickness out to fill the entire gap between
  the model's real edge and the requested window and painting its
  colour across a region with no real data behind it, which is what
  produced tall, flat-topped colour blocks appearing above the true
  topography. The requested headroom margin itself is preserved
  separately (via `zmin_km` directly, not the possibly-tightened display
  edge), so the amount of white space above the real topography is
  unchanged — it's just guaranteed genuinely blank now.
- `VSLICE_SHOW_DEM_TOPO_LINE` (default `False`) — an optional, purely
  visual black dashed line (`VSLICE_DEM_TOPO_STYLE`) showing the real
  DEM (`modem_topo_utm.nc`) alongside the model's own air/rock boundary
  (`VSLICE_TOPO_STYLE`, the dimgray line/staircase — this is what
  `surf_depth` actually is). Turn on to visually compare the two; it
  plays no part in defining the surface, the fill, the seismicity/
  MT-site depth cutoff, or the figure headroom either way — those always
  use `surf_depth` only. Same setting, same behaviour, in both
  `tacna_plot_modem_image.py` and `tacna_plot_modem_mesh.py`.
- **Air cells are masked to `NaN` at the source**, in
  `tacna_precompute.py` (Part A) `apply_transform()`, before the log
  transform and before saving `modem_model_utm.nc`/
  `modem_rho_utm_{tag}.nc` — controlled by `AIR_RHO_THRESHOLD` (Ω·m,
  linear, default `1e10`). `NaN` then propagates through everything
  downstream (map, section, `image`'s interpolation, `mesh`'s exact
  per-cell lookup) with one consistent, guaranteed-agreeing definition
  of "is this air", rather than each plot script re-deriving its own
  cutoff independently. The plot scripts' own
  `AIR_LOG10_RHO_THRESHOLD`-based masking is kept as a defensive
  fallback (harmless no-op against already-`NaN` cells) for older
  precompute output. **Re-run `tacna_precompute.py`** to pick this
  up — it only takes effect once the model/depth-slice files are
  regenerated with it.
- Section figure height is sized from the *actual* displayed depth span,
  not just the requested `zmin_km`/`zmax_km` window: when topography
  pushes the top of the axis higher (`surf_depth.min() -
  VSLICE_TOPO_HEADROOM_KM`, whenever that's above `zmin_km`), the extra
  headroom is included in the height calculation up front. Previously the
  figure box was sized from `zmin_km`/`zmax_km` alone and the axis was
  only discovered to be taller *after* the box was already fixed —
  quietly reducing the effective `VSLICE_VE` and leaving a different
  amount of "extra" white space above the topo line in each script,
  since how much taller depends on each script's own way of locating the
  surface (`tacna_plot_modem_image.py`'s uniformly-resampled axis vs.
  `tacna_plot_modem_mesh.py`'s exact mesh-cell edges). Same fix in both.
- `SHOW_COLORBAR` (default `True`) — set `False` to omit the colorbar
  entirely; the map panel itself is completely unaffected either way.
- `COLORBAR_POSITION` (`"right"` default, or `"left"`/`"bottom"`/
  `"top"`) — the colorbar is added as *extra* width (right/left) or
  height (bottom/top) beyond the map panel, so it never competes with
  the map for space and can never distort it. For `"left"`/`"bottom"`/
  `"top"`, extra clearance is automatically reserved beyond `COLORBAR_PAD`
  so the colorbar doesn't collide with whatever the main axes normally
  draws in that same space using ordinary (non-managed) layout — y
  tick labels/ylabel for `"left"`, x tick labels/xlabel for `"bottom"`,
  the plot title for `"top"` — sized from `AXIS_LABEL_SIZE`/
  `AXIS_TICK_SIZE`/`AXIS_TITLE_SIZE` below. (`tacna_plot_modem_mesh.py`
  defaults to `COLORBAR_POSITION = "bottom"`, so this fix matters there
  in particular.)
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

**Axis fonts:** `AXIS_LABEL_SIZE` (default `8`pt), `AXIS_TICK_SIZE`
(default `7`pt), `AXIS_TITLE_SIZE` (default `9`pt) — font sizes for the
map/section axis labels (`"Easting (km)"`/`"Northing (km)"`/
`"Depth (km)"`/distance or lon-lat labels), their tick annotations, and
the per-figure title, respectively. Separate from `COLORBAR_LABEL_SIZE`/
`COLORBAR_TICK_SIZE`, which only affect the colorbar's own label/ticks.
Same three settings, same behaviour, in all three plot scripts.

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

**Settings are kept in sync across all three plot scripts**, using
`tacna_plot_modem_mesh.py`'s values as the reference — colours, sizes,
fonts, thresholds, toggles, etc. Where a setting's value is inherently
script-specific (output directories, `PLOT_FILENAME_SUFFIX`, data-source
paths, colour-scale ranges for a different physical quantity), it's
deliberately left as-is rather than force-matched. Two settings use
different literal values that mean the same thing in each script's own
terms rather than a shared literal: `REGION_SOURCE` ("model" in the
ModEM scripts, "data" in `tacna_plot_seis.py` — `tacna_plot_seis.py` has
no "model" branch, so forcing the ModEM scripts' literal string would
silently fall through to different behaviour instead of matching it).
This now also includes the marker/label style dicts themselves
(`EQ_MARKER_STYLE`, `MT_MARKER_STYLE`, `SEISMIC_SITES_MARKER_STYLE`,
`VOLC_INACT_MARKER_STYLE`, `VOLC_ACT_MARKER_STYLE`, `CITY_MARKER_STYLE`,
`VOLC_LABEL_STYLE`, `CITY_LABEL_STYLE`, `MT_LABEL_STYLE`) and `VSLICES`
itself: all three scripts now define the same two profiles, `"profile
AA'"` and `"profile BB'"`, at the same coordinates. `tacna_plot_seis.py`
doesn't yet have MT-site-on-section support (`VSLICE_MT_STYLE` and the
matching projection/plotting), unlike the two ModEM scripts — that would
be a new feature rather than a settings sync, so it hasn't been added
automatically; let me know if you'd like it.

## Dependencies

```
numpy, matplotlib, xarray, pandas, pyproj, scipy
```
plus the local `modem.py` helper library (`read_mod`, `read_data`,
`cells3d`, `get_topo`).

## Typical run

```bash
python3 tacna_precompute.py         # writes *.nc for BOTH Part A (MT) and
                                     # Part B (seismic) into OUTPUT_DIR
python3 tacna_plot_modem_image.py   # reads them, writes figures (resampled sections)
# or, for an exact unresampled mesh cut:
python3 tacna_plot_modem_mesh.py
```

`tacna_precompute.py` runs Part A and Part B in one pass — but unlike
earlier versions, Part B no longer depends on Part A's resistivity beyond
both being in the same script; they're independent now. If you only need
to re-run the plot side, nothing here changes.

Run precompute again whenever the `.rho`/`.dat` files, `TRIM_PAD`,
`CROP_TO_REGION`/`TAR_LON`/`TAR_LAT`, `DEPTH_SLICES_KM`, or
`EXPORT_SUBMESH_TABLE` change (or anything in `README_seis.md`'s Part B
settings, since a single run covers both). Everything else (colours,
styling, crop views, profile definitions, annotations, sensitivity
shading) only needs re-running the plot script.
