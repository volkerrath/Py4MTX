# Tacna Seismic Tomography Pipeline

GMT/PyGMT-free pipeline for plotting Vp, Vs, and Vp/Vs seismic tomography
results for the Tacna region (southern Peru) on a UTM Zone 19S (EPSG:32719)
grid, using only NumPy / Matplotlib / xarray / pandas / pyproj / scipy.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline

```
tacna_precompute.py (Part B)  →  UTM-km NetCDF files  →  tacna_plot_seis.py  →  figures
```

**`tacna_precompute_seis.py` and `tacna_precompute_modem.py` have been
merged into a single `tacna_precompute.py`** — Part B (seismic tomography,
this README) and Part A (ModEM/MT, `README_mt.md`) now live in one script,
one run, because they share region-of-interest settings
(`OUTPUT_DIR`/`CROP_TO_REGION`/`TAR_LON`/`TAR_LAT` — see SHARED SETTINGS in
the script) and because Part B now also carries a density model
(`FNAME_DENS`, exported as `tacna_dens*.nc`), processed identically to
Vp/Vs. All of Part B's own outputs below are unchanged from the original
`tacna_precompute_seis.py`, apart from the density addition.

**`plotpy.py`** must sit alongside the scripts — it's a small shared module
of plotting helpers (`import plotpy`) used by all three plot scripts:
colormap loading (matplotlib name / GMT `.cpt` file / plain RGB(A) file),
UTM↔lon/lat conversion, hillshading, region-clipped scatter/label helpers,
the north arrow, the deterministic (equal-scale-by-construction) map/
section figure layout, the lon/lat tick overlay, the free-text annotation,
VE-label positioning, and generic profile-point sampling/projection
(shared by seismicity and, in the ModEM scripts, MT sites). Each plot
script owns its own settings and passes them in as arguments; nothing in
`plotpy.py` reads a calling script's globals directly.

Precompute and plot are deliberately separate: precompute does the slow
work once (reprojection, topo fetch, depth-slice extraction); plot only
reads the results and draws, so figure iteration (colours, styling, crop
boxes, profiles) is fast and doesn't re-touch the source data.

### 1. `tacna_precompute.py` — Part B (seismic tomography + density)

Reads the raw Vp/Vs/density finite-difference tomography models and produces:

| Output file                     | Contents                                   |
|----------------------------------|---------------------------------------------|
| `tacna_vp.nc`, `tacna_vs.nc`, `tacna_vps.nc`, `tacna_dens.nc` | Full geographic (lat/lon) subsets, with UTM easting/northing as auxiliary coords |
| `tacna_topo_utm.nc`              | Topography on a UTM-km grid                |
| `tacna_bath_utm.nc`              | Bathymetry mask (elevation ≤ 0) on the same grid |
| `tacna_vp_utm_{depth}km.nc`, `tacna_vs_utm_{depth}km.nc`, `tacna_vps_utm_{depth}km.nc`, `tacna_dens_utm_{depth}km.nc` | Per-depth UTM-km slices, one set per entry in `DEPTH_INDEX` |

**No more MT resistivity resampling here.** Earlier versions of this
script also resampled Part A's resistivity onto this grid directly
(`modem_rho_on_seisgrid*.nc`). That step has moved to `tacna_cluster.py`,
which RBF-interpolates resistivity (from `modem_submesh_points.nc`, Part
A) together with Vp/Vs/density onto its own jointly-defined regular grid
instead — see `README_cluster.md`.

**Density is named `dens`, not `rho`.** `FNAME_DENS` (default
`"../seistomo/FD_rho_model.nc"` — the source *file* is still named
`FD_rho_model.nc` on disk, matching the `FD_vp_model.nc`/`FD_vs_model.nc`
convention) is read, cropped, and sliced with exactly the same logic
already used for Vp/Vs. It's expected on the same `(lat, lon, depth)` grid
as `FD_vp_model.nc`; a non-fatal warning prints at run time if that grid
doesn't match (a silent mismatch there would misalign density against
velocity). Only the *output naming* was changed to `dens` — "rho" is
already used throughout Part A for MT resistivity (`modem_rho_utm_*.nc`),
and having the same-sounding name for two physically different
quantities on two different grids was a bug waiting to happen.

Topography comes from the `elevation` package (SRTM/ETOPO, via the `eio`
CLI), a local ETOPO NetCDF, or a local GeoTIFF — selected with
`TOPO_SOURCE`. Hillshade is **not** precomputed here; `tacna_plot_seis.py`
computes it on the fly with `matplotlib.colors.LightSource`.

**Key settings:**

- `OUTPUT_DIR` (default `"."`, **shared** with Part A now that precompute
  is a single script) — directory all `.nc` outputs above are
  written to (created automatically if it doesn't exist). Keep this in
  sync with `NC_DIR` in `tacna_plot_seis.py` so the plot script reads
  from wherever precompute actually wrote to.
- `FNAME_VP` / `FNAME_VS` / `FNAME_DENS` — input tomography NetCDF files.
- `CROP_TO_REGION` (default `True`, **shared** with Part A — one setting
  now, not two kept in sync) — if `True`, crop the velocity/density models
  to `TAR_LON`/`TAR_LAT` (and `DEPTH_RANGE`) before writing any output; if
  `False`, keep the full source-model extent (`DEPTH_RANGE` still
  applies).
- `TAR_LON` / `TAR_LAT` (**shared** with Part A) — geographic crop box for
  the velocity/density subset, currently the union of the two pipelines'
  original boxes, padded by ~0.05°. Must fully contain every `VSLICES`
  profile endpoint defined in `tacna_plot_seis.py`, or you'll get
  a silent no-data gap at the edge of a cross-section.
- `DEPTH_RANGE` — depth window (km) kept from the source model; the lower
  bound is intentionally negative (e.g. `-8`) so above-sea-level model
  coverage (e.g. under a volcanic edifice) isn't discarded.
- `DEPTH_INDEX` — which depth levels to export as UTM-km slices (must be
  kept in sync with `DEPTH_INDEX` in `tacna_plot_seis.py`).
- `MAP_LON` / `MAP_LAT` — topo/bathymetry fetch bounds; kept slightly
  wider than `TAR_LON`/`TAR_LAT` so the basemap isn't clipped tighter than
  the data.
- `TOPO_SOURCE` (`"elevation"` / `"etopo"` / `"geotiff"`) — topography
  backend, with `ETOPO_PATH` / `GEOTIFF_PATH` for the local-file backends.

### 2. `tacna_plot_seis.py`

Reads the files above and produces two kinds of figures:

- **Horizontal depth-slice maps** — one per entry in `PLOT_WHAT` (any
  subset of `["vp", "vs", "vps"]`) × `DEPTH_INDEX`, with hillshaded
  topography, ocean fill, seismicity/MT-site/volcano/city overlays, a
  north arrow, and an optional lon/lat tick overlay.
- **Vertical cross-sections** — one per entry in `VSLICES`, along an
  arbitrary profile (UTM or lon/lat endpoints), with seismicity projected
  onto the section and an optional topography line/fill.

**Input/output directories:**

- `NC_DIR` (default `"."`) — directory to read precomputed NetCDF files
  from (`tacna_vp.nc`, `NC_TOPO`, per-depth slices, etc.). Must match
  `OUTPUT_DIR` in `tacna_precompute.py`.
- `PLOT_DIR` (default `"."`) — directory saved figures are written to
  (created automatically if it doesn't exist).

**Map region & extent:**

- `REGION_SOURCE` — `"data"` (velocity-subset extent, default) or
  `"topo"` (topo-grid extent), combined with `REGION_MARGIN_KM`.
- `MAP_XLIM` / `MAP_YLIM` (UTM km, default `None`) — explicit override of
  the displayed map extent, applied *after* `REGION_SOURCE` computes the
  region. Also drives feature clipping, the figure's aspect ratio, and the
  lon/lat tick overlay, so everything stays consistent with what's drawn.
- Maps always render at exact equal x/y scale (1 km in easting = 1 km in
  northing on the page), regardless of `MAP_XLIM`/`MAP_YLIM` or whether a
  colorbar is shown — guaranteed *by construction*: `create_map_figure()`
  places the map axes at an explicit, physically-computed size in inches
  matching the region's own aspect ratio, rather than relying on
  matplotlib's `ax.set_aspect("equal")` plus automatic colorbar
  space-stealing (which can desync from the actual rendered box). See
  `FIG_WIDTH` and the colorbar settings below.
- The actual UTM min/max used for a given run are only known once
  precompute has produced `tacna_vp.nc` — `tacna_plot_seis.py` prints
  `UTM region (km): [xmin, xmax, ymin, ymax]` at runtime, or read them
  directly: `xr.open_dataset("tacna_vp.nc")["utm_easting"/"utm_northing"]`.
- `AXES_UNITS` (`"km"` default, or `"latlon"`) — selects what the map's
  bottom/left tick labels show: UTM easting/northing in km, or longitude/
  latitude in degrees. It's an in-place relabelling of the primary axes
  (one unit system at a time), not a secondary overlay axis.
  `LATLON_NTICKS`/`LATLON_DECIMALS` control the lon/lat tick density and
  precision when `AXES_UNITS = "latlon"`. `AXES_KM_COMMA` (default
  `True`) adds a thousands comma to km-axis tick labels (American style,
  e.g. `8,000`) when `AXES_UNITS = "km"`; set `False` for plain numbers
  (`8000`). No effect when `AXES_UNITS = "latlon"`.

**Vertical sections (`VSLICES`):** each entry is a dict with `name`, `p1`/
`p2` (endpoints, UTM km or lon/lat via `coord`), `zmin_km`/`zmax_km`,
`npts`/`nz` (sampling), `swath_km` (seismicity projection half-width), and
optional per-slice `xlim`/`ylim` to crop the *displayed* view without
resampling. `VSLICE_X_AXIS` switches the horizontal axis between `"utm"`
(easting/northing, matches the map) and `"distance"` (cumulative distance
from `p1`, starting at 0) — in `"distance"` mode, a slice's `xlim`
override is in distance-km, not UTM-km.

**Topography on sections:** the surface line is always drawn
(`VSLICE_TOPO_STYLE`); the shaded band between the surface and `z = 0`
(sea level / top of the model) is only drawn when
`VSLICE_SHOW_TOPO_FILL = True` (default `False`, line only), using
`VSLICE_TOPO_LAND_COLOR` / `VSLICE_TOPO_OCEAN_COLOR`.

**Vertical exaggeration:** `VSLICE_VE` (1 = true scale). The "VE = …×"
label is drawn *before* the colour image (low z-order, so the
semi-transparent data sits over it), positioned via `VSLICE_VE_POS`
(`"lower right"` default, or `"lower left"`/`"upper left"`/`"upper
right"`, or an explicit `(x, y, ha, va)` axes-fraction tuple) and styled
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
wide-short panel shape a real profile tends to have.

**Free-text annotation:** `ANNOTATION_TEXT` (default `None`) draws one
extra line — a version tag, processing note, "DRAFT" watermark, etc. — on
every figure the script produces, positioned/styled via `ANNOTATION_POS` /
`ANNOTATION_STYLE`.

**Map feature layers:** `SHOW_PROFILE_LINES`, `SHOW_VSLICE_LINES`,
`SHOW_SEISMICITY`, `SHOW_MT_SITES`, `SHOW_SEISMIC_SITES`,
`SHOW_VOLCANOES`, `SHOW_VOLCANOES_ACTIVE`, `SHOW_CITIES`,
`SHOW_NORTH_ARROW` — one boolean per overlay layer, all default `True`.
`SHOW_SEISMICITY` also controls the matching seismicity scatter on
vertical sections (`VSLICE_EQ_STYLE`), so turning it off applies
everywhere the layer would otherwise appear, not just on the map. Same
flags, same behaviour, in all three plot scripts.

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
station names don't overlap horizontally. In `tacna_plot_seis.py`, MT
site names are read from `CSV_MT_SITES` — the sitelist's own site-name
column isn't standardised across exports, so the loader tries `Site`,
`site`, `name`, `Name`, `station`, `Station` in turn and falls back to
blank labels (harmless, since the default is `mode="none"` anyway) if
none of those columns exist.

**Seismic sites** (`CSV_SEISMIC_SITES`, default
`../features/seismic_sites.csv`) — seismometer station locations, a
plain CSV with **no header row**: columns are `network`, `station`,
`lat`, `lon`, `elev_m`. Default marker: filled green square, size 18
(`SEISMIC_SITES_MARKER_STYLE`). Map-only (no vertical-section
projection, unlike seismicity/MT sites) and not currently labelled with
station names — just the marker.

Seismicity on maps can additionally be depth-filtered per slice via
`ZMIN_SEISM`/`ZMAX_SEISM` (km, one `(zmin, zmax)` pair per entry in
`DEPTH_INDEX` — `None` in either slot means unbounded, i.e. no filter on
that side). All three lists must be the same length, or the script exits
with an explanatory error rather than silently mis-indexing — this used
to fail silently in `tacna_plot_modem_image.py` (stale, longer-than-needed
lists left over from an earlier `DEPTH_SLICES_KM`), which was why its
maps showed different seismicity than `tacna_plot_modem_mesh.py`'s despite
both reading the identical catalog; fixed and now guarded the same way in
all three scripts.

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
  `AXIS_TICK_SIZE`/`AXIS_TITLE_SIZE` below.
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

**Other notable settings:** `PLOT_FORMATS`/`PLOT_DPI` (output formats and
resolution), `CMIN_*`/`CMAX_*` and `CMAP_*` (colour scales — accept a
matplotlib name, a GMT `.cpt` file, or a plain RGB(A) text/CSV file via
`load_colormap()`), and the `*_MARKER_STYLE`/`*_LABEL_STYLE` dicts for
every overlay (seismicity, MT sites, volcanoes, cities, profile lines,
north arrow).

---

## Coordinate convention

All grids and figures use **UTM Zone 19S (EPSG:32719)**, distances in km.
Depth is km, positive down; `z = 0` is sea level / the top of the velocity
model.

## Dependencies

```
numpy, matplotlib, xarray, pandas, pyproj, scipy, rioxarray, elevation (eio)
```
(`rioxarray`/`elevation` only needed for `TOPO_SOURCE="elevation"` or
`"geotiff"`.) `tacna_precompute.py` additionally needs the local `modem.py`
helper library (`read_mod`, `read_data`, `cells3d`, `get_topo`) on the
path — required for Part A even if you only care about Part B's
Vp/Vs/density outputs, since precompute is now one script covering both.

## Typical run

```bash
python3 tacna_precompute.py   # writes *.nc for BOTH Part A (MT) and
                               # Part B (seismic) into OUTPUT_DIR
python3 tacna_plot_seis.py    # reads Part B's files, writes figures (PLOT_FORMATS)
```

`tacna_precompute.py` runs Part A and Part B in one pass — see
`README_mt.md`'s "Typical run" for why they can't be split.

Run precompute again whenever the source tomography/density models,
`TAR_LON`/`TAR_LAT`, `DEPTH_RANGE`, or `DEPTH_INDEX` change (or anything in
`README_mt.md`'s Part A settings, since a single run covers both).
Everything else (colours, styling, crop views, profile definitions,
annotations) only needs re-running `tacna_plot_seis.py`.
