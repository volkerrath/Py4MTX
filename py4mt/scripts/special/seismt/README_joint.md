# `plot_joint.py` — Depth-Slice Maps & Sections of the Interpolated Grid

Plots every field on the common grid produced by `interpolate.py`
— MT resistivity/conductivity/sensitivity and/or seismic tomography
Vp/Vs/Vp-Vs-ratio/density, whichever were interpolated — **including
their spatial gradients**, as depth-slice maps and (where the grid is
regular) vertical cross-sections, using the same basemap engine and
styling conventions as the MT and seismic plot pipelines (`README_mt.md`,
`README_seis.md`).

This is the plotting counterpart to `cluster.py`: both scripts read
`interpolate.py`'s output and are completely agnostic to how that
grid was built or which interpolation method produced it. Where
`cluster.py` clusters the fields and plots the resulting class
labels, `plot_joint.py` plots each field's own value (and gradient
magnitude/components) directly — no clustering.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline

```
precompute.py  →  interpolate.py  →  {SITE_PREFIX}_interp_<method>.nc  →  plot_joint.py  →  figures
```

`SITE_PREFIX` (whichever site is the live, uncommented value — see
`precompute.py`/`interpolate.py`) selects which site's
`{SITE_PREFIX}_interp_<method>.nc`, topo/bath grids, and MT sitelist
this script reads, and prefixes its own output filenames.

`plot_joint.py` reads `INTERP_FILE` — no grid-building or
interpolation of its own — and is therefore agnostic to which
interpolation method (`rbf`/`kriging`/`idw`/`nearest`) or target-grid
choice (`joint`/`seismic`) produced it. Point `INTERP_FILE` at whichever
`interpolate.py` run you want to plot. You can also interpolate a
superset of variables (and their gradients) once and plot different
`PLOT_VARS` subsets without re-running `interpolate.py`.

`tomomt.py` must sit alongside `plot_joint.py` — the shared
plotting helper module also used by the MT, seismic, and clustering
scripts.

---

## Two grid modes

Which one applies is read from `INTERP_FILE`'s own `target_grid_mode`
attribute — nothing to set here.

| | `"joint"` | `"seismic"` |
|---|---|---|
| Grid | Genuinely regular UTM-km `(depth, northing, easting)` | Reused seismic-tomography native `(depth, row, col)`, **not** regular in UTM space — 2-D `utm_easting_km`/`utm_northing_km` aux coords instead |
| Depth-slice maps | `imshow(extent=...)` | `pcolormesh(shading="nearest")` against the 2-D coords |
| Vertical sections (`VSLICES`) | Supported — sampled via a 3-D `RegularGridInterpolator` directly in UTM space | **Not supported.** An arbitrary-angle straight-line cut through a grid that's irregular in UTM needs its own interpolation step. `VSLICES` is skipped with a printed warning; re-run `interpolate.py` with `TARGET_GRID="joint"` first if you need sections |
| Isolines | `ax.contour()` with 1-D axes | `ax.contour()` with the grid's own 2-D coords (matplotlib supports both) |

---

## Gradients (`PLOT_GRADIENT`)

If `interpolate.py`'s `COMPUTE_GRADIENT` was `True`, `INTERP_FILE`
carries, for each gradient variable, up to four extra data variables:
`{key}_grad_easting`, `{key}_grad_northing`, `{key}_grad_depth` (partial
derivatives) and `{key}_grad_mag` (their combined 3-D magnitude) — see
`README_interpolate.md`.

`PLOT_GRADIENT = True` (default) makes this script **also** produce
gradient maps/sections for those fields. `GRADIENT_COMPONENTS` picks
which of `"mag"`/`"easting"`/`"northing"`/`"depth"` to plot — `["mag"]`
only by default, since that's the one every field shares comparable
(always-positive) units for at a glance.

A field with no matching `{key}_grad_*` in `INTERP_FILE` — either
`COMPUTE_GRADIENT` was off, or that key wasn't in `interpolate.py`'s
`GRADIENT_VARS` — is silently skipped for gradient panels, with a note
printed at startup listing which fields that applies to.

---

## Standard layout: property + `|∇field|` paired (`PAIR_GRADIENT_WITH_FIELD`)

Whenever a field's gradient **magnitude** was plotted (`PLOT_GRADIENT=True`,
`"mag"` in `GRADIENT_COMPONENTS`, and that field actually has one), it's
drawn in the **same figure** as the field itself — one file per (field,
depth)/(field, profile) instead of two — rather than as two separate
figures. Each panel keeps its own colour scale/colorbar (`VAR_STYLE` /
`GRAD_STYLE` respectively); only the canvas is shared. This is the
default (`PAIR_GRADIENT_WITH_FIELD = True`); set it `False` to go back
to one file per panel (the pre-pairing behaviour).

**Layout is chosen automatically, not fixed**, from the aspect ratio a
*single* panel would have on its own (`FIG_WIDTH`-derived map size, or
the `VSLICE_WIDTH_CM`/`VSLICE_HEIGHT_CM`-derived section size):

| Single panel is… | Layout | Why |
|---|---|---|
| wide/landscape (`panel_w ≥ panel_h`) — e.g. a broad map, or a long vertical section | **stacked**, 2 rows × 1 column | avoids an implausibly *wide* combined figure |
| tall/portrait (`panel_h > panel_w`) | **side by side**, 1 row × 2 columns | avoids an implausibly *tall* combined figure |

`PAIR_GAP_CM` sets the gap between the two panels (horizontal gap if
side by side, vertical gap if stacked). The geometry math lives in
`_build_dual_panel_figure()` — a local sibling of `tomomt.build_panel_figure()`
(not added to `tomomt.py` itself, since it's only needed here); it
supports all four `COLORBAR_POSITION` choices (`right`/`left`/`top`/`bottom`)
in either orientation.

Falls back to a single-panel figure automatically wherever there's
nothing to pair with (no gradient computed, `"mag"` not requested, or
that field wasn't in `GRADIENT_VARS`) — `PLOT_GRADIENT`/
`GRADIENT_COMPONENTS` still control what's available to pair in the
first place. Any **other** gradient components you also request (e.g.
`"easting"`) are **not** paired — magnitude is the natural partner since
it's the one every field shares comparable units for; other components
remain their own separate single-panel figures, with a
`_grad<component>` filename suffix as before.

---

## Colour scales (`VAR_STYLE` / `GRAD_STYLE`)

Unlike `plot_seis.py` / `plot_dens.py` — each hard-coded to
one or three known fields — the set of fields actually present in
`INTERP_FILE` is only known at run time (whatever `interpolate.py`'s
`INTERP_VARS` happened to be). Colour scales are therefore dicts keyed by
field name:

- `VAR_STYLE = {"rho": dict(cmap=..., cmin=..., cmax=...), ...}` — a
  field not listed falls back to `DEFAULT_CMAP` and full auto-scaling.
- `GRAD_STYLE`, keyed by `"{var}_grad_{component}"` (e.g.
  `"rho_grad_mag"`) — falls back to `DEFAULT_GRAD_CMAP`.

`cmin`/`cmax` = `None` (the default for every entry shipped in the
script) means that particular panel auto-scales to its own finite data
range, computed fresh per depth slice/section — same convention as the
other plot scripts' `CMIN_GRAD=None`/`CMAX_GRAD=None` default. Gradient
panels have no natural fixed range (a spatial derivative), so they
auto-scale unless you give an explicit override in `GRAD_STYLE`.

**Before setting a fixed `cmin`/`cmax`**, check the units this script
prints for each field at startup (`resolved_units`, read from
`INTERP_FILE`'s own per-variable `units` attribute) — `plot_seis.py`
carries Vp/Vs ranges in m/s, while `interpolate.py`'s own
`VARIABLE_SOURCES` registry declares km/s for the same fields. Fixed
ranges here are left as placeholders (`None`) for exactly this reason.

---

## Volcano labels (`VOLC_NAME_COL` / `VOLC_LABEL_CHARS`)

Volcano labels always read the same `volcanes.csv` column
(`VOLC_NAME_COL`, default `"NAME"`) and are truncated to their first
`VOLC_LABEL_CHARS` characters (default `4`) via `VOLC_LABEL_STYLE`'s
`mode="firstN"` (see `tomomt.apply_label_mode`) — this replaced an
earlier two-column approach (a full-name column plus a separately
maintained abbreviated-code column) with one source of truth for the
name and a single number controlling how much of it is shown.

---

## What this script does

1. Reads `INTERP_FILE` — grid coords/dims, per-variable interpolated
   fields and their units, and the attributes needed to know how to plot
   it (`target_grid_mode`, `dim_names`, `interp_vars`,
   `gradient_computed`, `gradient_vars`).
2. Picks `PLOT_VARS`, a subset of whatever variables are actually in
   `INTERP_FILE` (`None` = every one of them).
3. For each `PLOT_DEPTHS_KM` entry (nearest available level used) and
   each active variable: plots a depth-slice map (paired with its
   `|∇field|` panel per `PAIR_GRADIENT_WITH_FIELD` above, where
   available) on the shared basemap/feature-layer engine, then any
   other requested `GRADIENT_COMPONENTS` as their own separate panels.
4. If `grid_mode == "joint"` and `VSLICES` is non-empty: same, but for
   vertical cross-sections along each defined profile.

Writes, into `PLOT_DIR`, `len(PLOT_FORMATS)` files per figure:

| Output file | Contents |
|---|---|
| `{var}_joint_{depth}km_{SITE_PREFIX}.{ext}` | Depth-slice map of `var` at the nearest available depth to each `PLOT_DEPTHS_KM` entry — paired with its `\|∇var\|` panel in the same file when `PAIR_GRADIENT_WITH_FIELD` applies (see above), otherwise `var` alone |
| `{var}_joint_{depth}km_{SITE_PREFIX}_grad<component>.{ext}` | A non-magnitude gradient component (e.g. `"easting"`), always its own separate figure |
| `{var}_section_{profile}_{SITE_PREFIX}.{ext}` | Vertical cross-section of `var` along one `VSLICES` entry (`"joint"` grid mode only) — same pairing rule as the maps above |
| `{var}_section_{profile}_{SITE_PREFIX}_grad<component>.{ext}` | A non-magnitude gradient component section, its own separate figure |

No NetCDF output — this script only plots what `interpolate.py`
already computed.

---

## Settings carried over unchanged

Most of `plot_joint.py`'s settings are the same dicts/toggles as
`plot_seis.py` and `cluster.py` — see those scripts' own
comments (and `README_seis.md`) for the full rationale behind each one:

- Basemap: `SHOW_TOPO_BASEMAP`, `HS_AZIMUTH`/`HS_ALTITUDE`/`HS_SIGMA`,
  `TOPO_VMIN`/`TOPO_VMAX`, `OCEAN_COLOR`
- Colorbar layout: `SHOW_COLORBAR`, `COLORBAR_POSITION`/`SIZE`/`PAD`/
  `ASPECT`/`LABEL_SIZE`/`TICK_SIZE`/`NTICKS`
- Marker/label style dicts: `EQ_MARKER_STYLE`, `MT_MARKER_STYLE`/
  `MT_LABEL_STYLE`, `SEISMIC_SITES_MARKER_STYLE`,
  `VOLC_INACT_MARKER_STYLE`/`VOLC_LABEL_STYLE`, `VOLC_ACT_MARKER_STYLE`,
  `CITY_MARKER_STYLE`/`CITY_LABEL_STYLE`, `ARROW_STYLE`/`ARROW_LABEL_STYLE`
- Feature CSV paths and on/off switches: `CSV_VOLCANES`,
  `CSV_SEISMCAT`, `CSV_MT_SITES`, `CSV_CITIES`, `CSV_SEISMIC_SITES`,
  `SHOW_SEISMICITY`/`SHOW_MT_SITES`/`SHOW_SEISMIC_SITES`/
  `SHOW_VOLCANOES`/`SHOW_VOLCANOES_ACTIVE`/`SHOW_CITIES`/
  `SHOW_NORTH_ARROW`
- `VSLICES` (same schema — `p1`/`p2`/`coord`/`zmin_km`/`zmax_km`/`npts`/
  `nz`/`swath_km`/`xlim`/`ylim`), `VSLICE_VE`, `VSLICE_EQUAL_SCALE`,
  `VSLICE_X_AXIS`, `VSLICE_TOPO_*`, `VSLICE_MAP_LINE_STYLE`
- Isolines: `ISO_LINES_MAP`/`ISO_LINES_VSLICE`, `ISO_LEVELS_MAP`/
  `ISO_LEVELS_VSLICE` (`"auto"` / explicit list / per-field dict — keyed
  the same way as `VAR_STYLE`, including `"{var}_grad_mag"` for a
  different setting on gradient panels), `ISO_AUTO_N`, `ISO_STYLE`,
  `ISO_LABEL*`
- `AXES_UNITS`/`LATLON_NTICKS`/`LATLON_DECIMALS`/`AXES_KM_COMMA`,
  `AXIS_LABEL_SIZE`/`AXIS_TICK_SIZE`/`AXIS_TITLE_SIZE`
- `ANNOTATION_TEXT`/`ANNOTATION_POS`/`ANNOTATION_STYLE`

Settings specific to this script: `SITE_PREFIX`, `INTERP_FILE`,
`PLOT_VARS`, `PLOT_GRADIENT`/`GRADIENT_COMPONENTS`,
`PAIR_GRADIENT_WITH_FIELD`/`PAIR_GAP_CM`, `VAR_STYLE`/`GRAD_STYLE`/
`DEFAULT_CMAP`/`DEFAULT_GRAD_CMAP`, `ALPHA_DATA`, `PLOT_DEPTHS_KM`,
`ZMIN_SEISM`/`ZMAX_SEISM` (one pair per `PLOT_DEPTHS_KM` entry — pad with
`None` to show all seismicity on a given slice, same convention as
`cluster.py`'s specific plot), `VOLC_NAME_COL`/`VOLC_LABEL_CHARS`.

---

## Coordinate convention

**UTM Zone 19S (EPSG:32719)**, distances in km, depth in km positive
down — see `README_interpolate.md` for the full `"joint"` vs `"seismic"`
grid-mode coordinate layout.

## Dependencies

```
numpy, xarray, pandas, matplotlib, scipy (RegularGridInterpolator —
only used for "joint"-mode vertical sections)
```
plus the local `tomomt.py` helper module.

## Typical run

```bash
python3 precompute.py      # must be run first (or already have been)
python3 interpolate.py     # see README_interpolate.md
python3 plot_joint.py      # reads {SITE_PREFIX}_interp_<method>.nc, plots
                            # every field (+ gradients, paired by default)
                            # as maps and, for a "joint" grid, sections
```

Re-run `plot_joint.py` (no need to re-run `interpolate.py`) whenever
`PLOT_VARS`, `PLOT_GRADIENT`/`GRADIENT_COMPONENTS`,
`PAIR_GRADIENT_WITH_FIELD`/`PAIR_GAP_CM`, `VAR_STYLE`/`GRAD_STYLE`,
`PLOT_DEPTHS_KM`, `VSLICES`, or any other plotting setting changes.
Re-run `interpolate.py` first (see `README_interpolate.md`) whenever the
interpolated grid itself — or whether gradients were computed at all —
needs to change.
