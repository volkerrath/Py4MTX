# femtic_ens_plot_slices.py

Slice figures of a FEMTIC ensemble, drawn from the results `.npz` of
`femtic_ens_post.py`.

Two kinds of panel (both configured by `MOD_STATS_WHAT`):

- `"best"`: the best-nRMS member (the former `MOD_QC` figure);
- the **statistics panels**: mean, median, error, MAD, percentiles,
  inter-quantile spreads, bootstrap error, sensitivity and SimRC maps (the
  former `MOD_STATS` figures) -- whatever the `.npz` holds.

Maps, N-S / E-W curtains and vertical planes (`MOD_SLICES`) are drawn with
`femtic_viz.plot_model_slices`. Nothing is recomputed and no member directory
is read; the panels come from the stored vectors. Needed on disk besides the
`.npz`: the mesh (`MOD_MESH`) and one resistivity block file for the
header / region structure (by default the best member's own file recorded in
the `.npz`; `TEMPLATE_BLOCK` replaces it, e.g. after moving the results to
another machine). Colour scales, slice positions, vertical exaggeration,
reference-model lines, region of interest or blanking by a statistic can
therefore be iterated in seconds.

Since 2026-11-08 this is a separate script; the code was part of
`femtic_ens_post.py` (sections 5-8) and the settings keep their old names and
defaults. The reusable parts moved to `femtic.py` (section 6c).

---

## Workflow

```
<PREFIX>_results.npz  (femtic_ens_post.py)
        |
        v
  fem.read_ensemble_results      statistics, best vector, member list
  fem.resolve_results_template   template block file (header, regions)
  origin / sites / ROI           fem.resolve_origin_and_sites, MOD_ROI_*
        |
        +--> "best":  block file from the stored best vector -> plot_model_slices
        +--> alpha mask (MOD_ALPHA_SOURCE)  -> resistivity_block_<P>_alpha.dat
        +--> each statistic: block file (sentinel air/ocean) -> plot_model_slices
```

---

## Configuration parameters

Defaults in the tables are the documented defaults; the UPPERCASE user section of
the script holds the values currently set. Every setting has a `#:` comment
there with the full text.

### Input

| Parameter | Type | Description |
|---|---|---|
| `ENSEMBLE_DIR` | str | Ensemble directory; `templates/mesh.dat`, `templates/site.dat` etc. are looked up here. |
| `ENSEMBLE_PREFIX` | str | `ENSEMBLE_PREFIX` of `femtic_ens_post.py`, **including** the suffix it added for `ENS_MODEL_SOURCE = "perturbed"` or for a random subsample (`ENS_SUBSAMPLE_N`). Names the results file and the figures. |
| `RESULTS_DIR` | str | Where `femtic_ens_post.py` wrote its results; the figures go here as well. |
| `ENSEMBLE_RESULTS` | str | The results `.npz`; default `RESULTS_DIR + ENSEMBLE_PREFIX.upper() + "_results.npz"` (same rule as in `femtic_ens_post.py`). |
| `NPZ_PREFIX` | str or None | Key prefix inside the `.npz`. `None` = detect (the file's only ensemble). Set it if the file holds several ensembles. |
| `TEMPLATE_BLOCK` | str or None | Resistivity block file that supplies header / region structure / flag columns. `None` = the best member's block file recorded in the `.npz`. Only its structure is used: all values come from the stored vectors. Set it when the member directories are gone or the results were moved: any resistivity block file of this ensemble (same mesh, same free regions) will do, e.g. a copy of the starting model. |

### Mesh, ocean and air

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_MESH` | str | `<ENSEMBLE_DIR>/templates/mesh.dat` | Tetrahedral mesh; required for all slice plots. |
| `MOD_MESH_SWAP_XY` | bool / None | `True` | Swap of the first two node-coordinate columns when the mesh is read (`femtic.read_femtic_mesh`). `mesh.dat` holds x = northing, y = easting; the plots, sites and slice positions use x = easting, y = northing, so `True` (default) gives the right orientation. `False` draws the model transposed (NNE structures appear as ENE) and is only for meshes already written as x = easting, y = northing; `None` = leave femtic's module default (no swap). |
| `MOD_OCEAN` | bool / None | `None` | `None` = auto-infer; `True`/`False` forces ocean-present/absent. Must match the inversion. |
| `MOD_AIR_RHO` | float | `1.0e9` | Ohm.m sentinel for air cells (region 0). |
| `MOD_OCEAN_RHO` | float | `0.25` | Ohm.m sentinel for ocean cells (region 1). |
| `MOD_PLOT_FORMAT` | list of str | `["pdf"]` | File extensions; every figure is saved once per entry (`"pdf"`, `"svg"`, `"png"`, `"jpg"`, ...). |

### Panels to plot

**Since 2026-09-26, one list drives everything.** `MOD_QC` used to be a
separate on/off switch that had to be kept in sync with `MOD_STATS_WHAT`
by hand (turn on QC *and* remember `MOD_STATS_WHAT` is still what it
should be, or vice versa). It is now just the special `"best"` entry in
`MOD_STATS_WHAT` — the *one* list of panel sources, alongside
`MOD_ALPHA_SOURCE` as the *one* blanking option (Shared slice / plot
parameters, below) — and `MOD_QC` / `MOD_STATS` are derived from it, not
set directly:

```python
MOD_QC    = "best" in MOD_STATS_WHAT
MOD_STATS = bool(set(MOD_STATS_WHAT) - {"best"})
```

To turn a panel off, edit the one list — drop `"best"` for no QC plot, or
set `MOD_STATS_WHAT = []` for nothing at all. A key the results file
does not hold (`femtic_ens_post.py` ran with that `COMPUTE_*` / `BOOTSTRAP_VAR`
switch off, say) is skipped with a printed message, never an error, so the
list can stay as it is.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_STATS_WHAT` | list of str | `["best","avg","med","err","mad"]` + one key per percentile level + one per `QDIFF_PAIRS` entry (+ `"err_boot"`) | Which panels to plot. `"best"` plots the best-nRMS converged member (to `MOD_QC_FILE`) — the former `MOD_QC` switch. Every other entry is a statistic key: subset of `"avg"`, `"var"`, `"err"`, `"med"`, `"mad"`, `"var_boot"`, `"err_boot"`, the log10 sensitivity keys (`"sens_mean_raw"`, `"sens_{mean,median,min,max,std}_{na,an}"`), `"sens_cv_na"`/`"sens_cv_an"` (linear), `"simrc_coef"`/`"simrc_corr"` (log10), plus auto-generated percentile keys (e.g. `2.3` → `"p2_3"`, `50.0` → `"p50"`, `97.7` → `"p97_7"`) and qdiff keys (e.g. `(15.9, 84.1)` → `"qdiff_15_9_84_1"`). `"err"` = `sqrt(var)` is plotted by default *instead of* `"var"`, since `var` is in (log10 Ω·m)² and isn't on the same scale as `MAD`/`QDIFF` (log10 Ω·m); add `"var"` back manually (with a `MOD_STATS_STYLE` entry) if you specifically want the raw-variance panel. |
| `MOD_QC` | bool | derived | `"best" in MOD_STATS_WHAT`. Not user-set — edit `MOD_STATS_WHAT` instead. |
| `MOD_QC_FILE` | str | `<prefix>_best` | Output path for the `"best"` panel; `_plot_slice()` appends `.<fmt>` per `MOD_PLOT_FORMAT` entry. |
| `MOD_STATS` | bool | derived | `bool(set(MOD_STATS_WHAT) - {"best"})`. Not user-set — edit `MOD_STATS_WHAT` instead. |
| `MOD_STATS_DIR` | str | `RESULTS_DIR` | Destination for block files and figures of the panels (the `"best"` panel too, unless `MOD_QC_FILE` is set). Add `+ "stats_plots/"` to keep them apart. |

Writes each selected statistic as a FEMTIC block file, then plots it.

**Per-panel style.** `MOD_CLIM` fixes the colour scale for the model
itself (log10 Ohm.m). Spread statistics (`VAR`/`ERR`/`MAD`/`QDIFF`) and
sensitivities live on entirely different scales, so each panel's colour
map, colour limits and colourbar label are set in `MOD_STATS_STYLE` (in
the shared slice block). `clim=None` now really auto-scales (it used to
fall back to `MOD_CLIM`).

**Sentinel air/ocean in statistics block files.** `plot_model_slices`
recognises air as log10 > threshold and ocean as log10 ~ log10(ocean
value). A statistic that happened to sit near those values (e.g. a log10
sensitivity of -0.6, the log10 of 0.25 Ohm.m) used to be painted as ocean.
MOD_STATS block files therefore write air as 1e30 and ocean as 1e-30 and
are plotted with matching `air_log10_thresh`/`ocean_value`.

Block files use the template block file (`TEMPLATE_BLOCK`, default the best
member's own file; preserves header, bounds, and flag columns).  Output
filenames follow the pattern `resistivity_block_<prefix>_<stat>.dat`.


### Sensitivity panels: ROI

Two display-time options for the log10 sensitivity panels (`sens_*`, not the
linear `sens_cv_*`). They never change the `.npz` or `MOD_ALPHA_SOURCE`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `SENS_ROI_NORMALIZE` | bool | `False` | When `True`, every `sens_*` panel actually plotted is re-centred, right before its block file is written, so the ROI's own max (the same box driving `MOD_XLIM`/`MOD_YLIM`/`MOD_ZLIM`, including `MOD_ROI_AUTO`) becomes log10 = 0 instead of the whole-mesh max -- equivalent to dividing by the max *within the ROI* rather than over the whole mesh. Cells outside the ROI can come out `> 0` (log10) once renormalised this way, since they may exceed the ROI's own max -- expected. `simrc_coef`/`simrc_corr` are **not** renormalised by this switch. |

**ROI diagnostic (always on, independent of `SENS_ROI_NORMALIZE`).** Every
`sens_*`/`simrc_*` panel that is listed in `MOD_STATS_WHAT` and actually
plotted has its *original* (whole-mesh-normalised, as stored) min/max printed
to the console, restricted to the free parameters whose volume-weighted
centroid falls inside the ROI: a quick read of the value range actually in
view, to make picking a `MOD_ALPHA_SOURCE` threshold easier. Degrades
gracefully (printed warning, diagnostic skipped) if the mesh or region
geometry cannot be read.

### Statistics definition used in the panels

| Parameter | Type | Default | Description |
|---|---|---|---|
| `QDIFF_PAIRS` | list of `(lo, hi)` | `[(15.9, 84.1), (2.3, 97.7)]` | Percentile pairs of the spread panels, as in `femtic_ens_post.py`'s `QDIFF_PAIRS`; the percentile levels themselves are taken from the file. |

### Shared slice / plot parameters

Both `MOD_QC` and `MOD_STATS` use the same slice/plot config block — this
block is identical (variable names and defaults) to `femtic_gst_prep.py`
and `femtic_rto_prep.py`.
The sensitivity-related alpha sources need the corresponding keys in the
results file (`COMPUTE_SENS` in `femtic_ens_post.py`).

| Parameter | Description |
|---|---|
| `MOD_SLICES` | List of slice-spec dicts (`kind`, `z0`/`x0`/`y0`). Kinds: `"map"`, `"ns"`, `"ew"`, `"plane"` (see "Vertical section" below). |
| `MOD_XLIM / YLIM / ZLIM` | Global axis limits (model-local **km**); `None` = auto. Overridden automatically by `MOD_ROI_AUTO` (below) when sites are available. |
| `MOD_ROI_AUTO` | Default `True`. When site positions are available, derives `MOD_XLIM`/`MOD_YLIM` from the site bounding box + `MOD_ROI_PAD_XY`, and sets `MOD_ZLIM` from `MOD_ROI_ZLIM` — overriding any literal values set above. Falls back to the literals (or full-mesh auto-scaling) when no sites are found. |
| `MOD_ROI_PAD_XY` | Default `2.0` km. Padding added around the site bounding box for `MOD_XLIM`/`MOD_YLIM`. |
| `MOD_ROI_ZLIM` | Default `[-1.0, 7.0]` km (positive-down). Depth range applied to `MOD_ZLIM` for `ns`/`ew`/`plane` panels when `MOD_ROI_AUTO=True`; `None` leaves `MOD_ZLIM` untouched. The negative lower bound leaves ~1 km of headroom above the z = 0 datum so topography is not clipped. |
| `MOD_CMAP` | Matplotlib colormap name. |
| `MOD_DPI` | Figure DPI, used by both `MOD_QC` and `MOD_STATS` plots. |
| `MOD_CLIM` | `[log10(ρ_min), log10(ρ_max)]`; `None` = auto. Default colour scale for `MOD_QC` and for any `MOD_STATS` panel not overridden by `MOD_STATS_STYLE`. |
| `MOD_OCEAN_COLOR` | Flat colour for ocean/lake cells; `None` = colormap. |
| `MOD_AIR_COLOR` | Flat colour for air cells. |
| `MOD_AIR_BGCOLOR` | Axes facecolor for air; `None` = figure default. |
| `MOD_ALPHA_SOURCE` | Single condition `"<name> <op> <value>"` selecting the cells to **show** (e.g. `"sens_mean_an > -3."`); `<name>` is any statistic key stored in the results file (see `MOD_STATS_WHAT`), `<op>` one of `> >= < <=`, `<value>` in that statistic's own units (log10 for sensitivities). Parsed with a regex, never `eval`ed. `None` = off. |
| `MOD_ALPHA_MODE` | `"blank"` (failing cells removed) or `"fade"` (failing cells fade linearly from opaque at the threshold to transparent `MOD_ALPHA_FADE_WIDTH` beyond it). |
| `MOD_ALPHA_FADE_WIDTH` | Fade width in the source's units (e.g. `1.0` = one decade). Default `1.0`. |
| `MOD_ALPHA_QC` | Also apply the mask to the `MOD_QC` best-member figure. Default `True`. |
| `MOD_ALPHA_EXCLUDE` | Tuple of statistic-key prefixes whose panels are drawn **without** the mask. Default `("sens_", "simrc_")`: a sensitivity map is not shaded by sensitivity (circular; the unmasked panel shows where the threshold cuts). `()` = mask every panel, as before 2026-10-28. |
| `MOD_STATS_STYLE` | Per-panel `dict(cmap=..., clim=..., label=...)` keyed as in `MOD_STATS_WHAT`; missing fields fall back to `MOD_CMAP`, `MOD_CLIM` (value-scale panels `avg`/`med`/`p*`) or auto-scaling (all others, `clim=None`), and a default colourbar label. Sensitivity/SimRC panels default to `viridis`; normalised sensitivities to `clim=[-4, 0]` (log10). |
| `MOD_EQUAL_ASPECT` | Equal aspect ratio on map/curtain panels (legacy test: acts for model/utm display only, not for `"latlon"`). |
| `MOD_VERT_EXAG` | `None` (default) = as `MOD_EQUAL_ASPECT` says, figures unchanged. A number VE > 0 fixes the aspect of the section panels (ns, ew, vertical plane) from the real axis units: `1.0` = true equal axes, `2.0` = vertical scale twice the horizontal, `0.5` = compressed — for every `MOD_DISPLAY_COORDS`, including `"latlon"`. Map panels stay 1:1 while `MOD_EQUAL_ASPECT` is `True`. Panel widths follow the real extents (×VE for the depth span). |
| `MOD_DEPTH_KM / HORIZ_KM` | Axis units. |
| `MOD_PANEL_HEIGHT` | Panel height in cm. |
| `MOD_PANEL_WIDTH` | Panel width in cm; `None` = auto from aspect ratio. With `MOD_EQUAL_ASPECT=True` and real `MOD_XLIM`/`MOD_YLIM`/`MOD_ZLIM` (e.g. from `MOD_ROI_AUTO`), map/ns/ew panels end up with genuinely different widths instead of being forced square. |
| `MOD_FIGSIZE` | `[w, h]` cm; overrides auto layout when set. |
| `MOD_NROWS / NCOLS` | Grid layout. Default `2 / 2`, matching the 4 default `MOD_SLICES` panels (2 maps + ns + ew); `None` = 1 row × N columns. Adjust if you change the number of panels. |
| `MOD_TICK_FONTSIZE` | Font size for axis tick labels and colourbar ticks. Default `7`, matching `fviz.plot_model_slices`' own default. |
| `MOD_LABEL_FONTSIZE` | Font size for axis labels, panel titles, and colourbar label. Default `8`, matching `fviz.plot_model_slices`' own default. |
| `MOD_TICK_DECIMALS` | Decimal digits shown on depth / easting-northing / lat-lon tick labels (all share this one value). Default `None` = `fviz.plot_model_slices`' own per-axis-type formatting unchanged. |
| `MOD_TITLE_FONTSIZE` | Font size of the figure title; `None` = `MOD_LABEL_FONTSIZE + 6`. |
| `MOD_SHOW_IN_SPYDER` | `True` (default) and running inside Spyder (detected via `utl.runtime_env() == "spyder"`) → every saved figure is also displayed inline in Spyder's Plots pane via `plt.show()`, in addition to being written to disk. No effect outside Spyder; set `False` to disable even under Spyder. |

### Titles

| Variable | Default | Description |
|---|---|---|
| `MOD_TITLE` | `"auto"` | Figure title of every figure of the script. `"auto"` = as before (the `"best"` figure: block-file name, best member, iteration, nRMS; the statistic figures: their block-file name). `None`, `False` or `""` = **no title**. A string = used as given, with the placeholders `{key}` (panel key of `MOD_STATS_WHAT`: `"best"`, `"avg"`, ...), `{label}` (description of the statistic), `{prefix}` (`ENSEMBLE_PREFIX`), `{n}` (number of members), `{source}` (`converged` / `perturbed`), `{best}`, `{iter}`, `{nrms}` (best member; `{nrms:.3f}` works); text with other braces is used as it stands. A dict = one entry per figure, e.g. `{"best": "Best member", "avg": "Mean", "err": None, "default": "auto"}` (keys = entries of `MOD_STATS_WHAT`; values as above; `"default"`, else `"auto"`, covers the keys not listed). |
| `MOD_PANEL_TITLES` | `True` | Panel titles (`Map  z = 1.0 km`, `N-S  easting = ...`). `False` = none. One panel only: add `title="text"` to its `MOD_SLICES` entry (`title=""` = just that panel untitled). |

### Colour bar

| Variable | Default | Description |
|---|---|---|
| `MOD_CBAR_LOCATION` | `"panel"` | `"panel"` = one colour bar next to every panel (as before). `"right"`, `"bottom"`, `"top"`, `"left"` = **one joint colour bar** for all panels of a figure (they share one colour scale), at that side of the whole panel grid. The panels give up a strip of the figure for it; their spacing is kept. Applies to the `"best"` figure and all statistic figures. |
| `MOD_CBAR_SHRINK` | `0.7` | Joint colour bar: length as a fraction of the side of the panel grid (`0 < value <= 1`). |
| `MOD_CBAR_THICKNESS` | `None` | Joint colour bar: thickness in cm; `None` = 1.5 x `MOD_LABEL_FONTSIZE` points. |

Both groups need the `femtic_viz.py` of 2026-11-09. At their defaults the figures are unchanged. The colour-bar label of each figure is still set per statistic in `MOD_STATS_STYLE` (`label`).


### Reference model (optional)

A reference model shown against the ensemble on the slice
figures: e.g. a deterministic inversion, the prior/starting model or a
published model. It is a resistivity block file on the **same mesh** as
`MOD_MESH`; its region structure may differ from the members'. It is one
fixed model for the whole ensemble: with the usual setting
`MOD_REF_FILE = ENSEMBLE_DIR + "/templates/referencemodel.dat"` it is the
**base (unperturbed) reference**, not a member's perturbed `referencemodel.dat`
(see `ENS_MODEL_SOURCE` in `femtic_ens_post.py` for plotting those).

| Parameter | Default | Description |
|---|---|---|
| `MOD_REF_FILE` | `None` | Block file of the reference model; `None` = off (everything below is then ignored). A missing file, or one whose element count differs from `MOD_MESH`'s, switches the reference off with a printed message; the run continues. |
| `MOD_REF_SLICES` | `True` | Draw the reference as iso-resistivity lines on every slice panel: the `"best"` panel and **all** statistics panels (average, spread, sensitivity, ...). The lines are the exact boundaries between slice cells lying on different sides of a level, so they follow the mesh without interpolation artefacts; air and ocean cells of the reference are ignored. |
| `MOD_REF_LEVELS` | `None` | Contour levels in log10(Ω·m), e.g. `[1.0, 2.0, 3.0]`. `None` = integer decades inside `MOD_CLIM` (inside the reference's own range when `MOD_CLIM` is `None`). The levels are always in the reference model's own units, whatever quantity a `MOD_STATS` panel shows, so a spread or sensitivity panel gets the same resistivity contours as the model panels. |
| `MOD_REF_STYLE` | `dict(colors="k", linewidths=0.8)` | `LineCollection` keywords of the slice lines, e.g. `linestyles="--"`. |
| `MOD_REF_LABEL` | `"reference model"` | Legend entry (lower right of each slice panel); `None` = none. |


### Plane sections (a `kind="plane"` entry in `MOD_SLICES`)

A section along any strike and dip is configured like every other slice,
as an entry of `MOD_SLICES` (units km / degrees). The default list ends
with a SW-NE vertical section through the model origin:

```python
dict(kind="plane", point=[0.0, 0.0], strike=45.0, dip=90.0)
dict(kind="plane", point=([6.12, 45.90, 0.0], "latlon"), strike=325.0, dip=70.0)
```

| Key | Default | Description |
|---|---|---|
| `point` | `[0, 0]` | A point on the plane; see "Position of `point`" below. `[0, 0]` = model origin at the surface. |
| `strike` | `0` | Geographic azimuth of the strike line, degrees clockwise from north, for every dip. For `dip = 90`, `45` = SW (left) to NE (right); `135` = NW to SE. |
| `dip` | `90` | Degrees below the horizontal, `0` .. `180`; the plane dips towards `strike + 90` (`strike=325, dip=70` dips 70 deg towards N55E). `90` = vertical section. |
| `xlim` | derived | Optional along-strike range `[min, max]` in km relative to `point` (negative = left end). |
| `ylim` | derived | Optional down-dip range `[min, max]` in km relative to `point` (dip != 90 only; positive = down the dip). |
| `invert_x`, `title` | | Passed through to `femtic_viz`. |

**Position of `point`.** Like `x0` / `y0` of the other kinds it can be given
in any of three coordinate systems; the vertical coordinate `z` is always
depth in model-local km, positive down (default 0 = surface), and matters
for dipping planes (a plane through `(x, y)` at depth `z`):

| Form | Meaning |
|---|---|
| `[x, y]` / `[x, y, z]` | model-local km (plain list, as before) |
| `([x, y, z], "model")` | model-local km |
| `([E, N, z], "utm")` | absolute UTM easting / northing in **metres**, `z` in km |
| `([lon, lat, z], "latlon")` | decimal degrees, **longitude first**, `z` in km |

UTM and latlon points are converted with the UTM zone and model origin the
script already uses for the other slices. They are independent of
`MOD_DISPLAY_COORDS`, which only sets the tick labels of the map / curtain
panels.

**Vertical sections (`dip = 90`).** The along-strike range is the
`MOD_XLIM` x `MOD_YLIM` box projected onto the strike direction (whole mesh
if either is `None`; after `MOD_ROI_AUTO` it follows the auto ROI), and the
depth range is `MOD_ZLIM`. The horizontal axis is in **model km** (0 at
`point`), independent of `MOD_DISPLAY_COORDS`.

**Dipping planes (`dip != 90`).** The axes are the true in-plane distances
from `point`: horizontal = along strike, vertical = down dip (positive
downwards), both in km if `MOD_DEPTH_KM` else m. The default extent is the
plane cut with the `MOD_XLIM` x `MOD_YLIM` x `MOD_ZLIM` box, so a plane
through the surface shows the air strip above down-dip 0 and ends at
`MOD_ZLIM`. The panel is drawn with true aspect where the other sections
get a fixed aspect (`MOD_EQUAL_ASPECT` with `"model"` / `"utm"` display, or
`MOD_VERT_EXAG` set; its value does not apply to a dipping plane), otherwise
it fills its cell. Check of the geometry: for a horizontally layered model a
layer boundary is a horizontal line at down-dip `(depth - z) / sin(dip)`.

The panel takes part in the normal `MOD_NROWS x MOD_NCOLS` grid, colour scale,
blanking and colourbar of its figure, and shows sites within
`MOD_PROJECTION_DIST` when `MOD_PLOT_SITES_SLICES` is `True`. Requires the
`femtic.py` / `femtic_viz.py` of 2026-11-09 (dipping planes; tagged `point`);
vertical sections need those of 2026-10-07 or later.


### Site overlay

| Parameter | Description |
|---|---|
| `MOD_SITE_DAT` | Path to `site.dat` CSV; `None` to fall back to `MOD_SITE_NUMBER`. |
| `MOD_SITE_NAMES` | `None` = all sites; list of strings = subset. |
| `MOD_SITE_NUMBER` | Fallback site number(s) from `observe.dat` (int or list of ints), used only when `MOD_SITE_DAT` is unavailable. |
| `MOD_PLOT_SITES_MAPS / SLICES` | Toggle site markers on map / curtain panels. |
| `MOD_PROJECTION_DIST` | Maximum projection distance (m) for curtain panels; `None` = show all sites on every panel. |
| `MOD_SITE_MARKER / MARKER_SLICES` | Marker style dicts for map / curtain overlays. |
| `MOD_MAP_MARKERS` | Extra point markers on map panels only. An entry with `"is_model_centre": True` (instead of `"latlon"`) overrides the model-centre marker's style rather than plotting as a regular marker. |
| `MOD_SHOW_MODEL_CENTRE` | `True` (default): marks the model origin on `"map"` panels whenever `MOD_DISPLAY_COORDS` is `"utm"`/`"latlon"` (no-op for `"model"`). Default style (since 2026-09-26): yellow star with a black edge, `ms=14`, at `zorder=15` (above every other map-panel artist); no legend entry. Override via `MOD_MAP_MARKERS` as above; set `False` to force off. |

### Geographic / UTM origin

| Parameter | Description |
|---|---|
| `MOD_ORIGIN_METHOD` | `"box"` (bounding-box midpoint) / `"average"` / `None` (use literal values). |
| `MOD_UTM_ORIGIN_LAT/LON/E/N` | Manual override; ignored when `MOD_ORIGIN_METHOD` estimates from `MOD_SITE_DAT`. |
| `MOD_UTM_ZONE_OVERRIDE` | Force a specific UTM zone string; `None` = auto-detect. |
| `MOD_DISPLAY_COORDS` | `"model"` / `"utm"` / `"latlon"`. |

---


## Output files

```
<MOD_STATS_DIR>/
  resistivity_block_<prefix>_avg.dat         block file of each statistic panel
  resistivity_block_<prefix>_err.dat
  resistivity_block_<prefix>_med.dat  ...
  resistivity_block_<prefix>_alpha.dat       (if MOD_ALPHA_SOURCE is set; raw values = alpha)
  <prefix>_best.<fmt>                        "best" panel (or MOD_QC_FILE.<fmt>)
  <prefix>_avg.<fmt>   <prefix>_err.<fmt>  ...      one figure per panel and MOD_PLOT_FORMAT entry
femtic_ens_plot_slices_summary.md            settings of the run (in ENSEMBLE_DIR)
```

The statistic block files are valid FEMTIC resistivity block files usable as
input to `femtic_mod_edit.py`, `femtic_mod_plot_slice.py`, or the VTK export
pipeline. Air and ocean cells carry the sentinel values `STATS_AIR_RHO` /
`STATS_OCEAN_RHO` of `femtic.py` (see "Panels to plot").

---

## Quick start

```bash
conda activate EM
python femtic_ens_post.py          # once: scan, compute, write the .npz
# edit ENSEMBLE_DIR, ENSEMBLE_PREFIX (and RESULTS_DIR) in femtic_ens_plot_slices.py
python femtic_ens_plot_slices.py   # repeat as often as needed
```

Plotting a prior (perturbed) run or a subsample: set `ENSEMBLE_PREFIX` to the
prefix with its suffix (`..._perturbed`, `..._sub32`), or `ENSEMBLE_RESULTS` to
the file.

---

## Relationship to other scripts

| Script | Purpose |
|---|---|
| `femtic_ens_post.py` | Writes the results `.npz` this script reads. |
| `femtic_ens_plot_bh.py` | Borehole profiles from the same `.npz`. |
| `femtic_data_misfit.py` | Quality control; its method `"nrms"` draws the nRMS histogram from the same `.npz`. |
| `femtic_gst_prep.py`, `femtic_rto_prep.py` | Share the `MOD_*` plot settings (names and semantics). |
| `femtic_mod_plot_slice.py` | Slice figures of single models ("mod" family): one block file, or a list / glob / run directories with an iteration selector, one figure per model and an optional catalog PDF. Replaces the per-member figures of the former `femtic_ens_plot.py`. |

---

## Changelog

| Date | Author | Change |
|---|---|---|
| 2026-11-08 | Claude Sonnet 5.5 (Anthropic) | Created: the `"best"` and statistics slice panels of `femtic_ens_post.py` (`MOD_QC` / `MOD_STATS`, alpha mask, reference lines, ROI) as a stand-alone script that reads the results `.npz`. `MOD_STATS_WHAT` is a plain list (keys missing from the file are skipped with a message); `MOD_STATS_STYLE` lists the standard keys explicitly. Helpers moved to `femtic.py` (`read_ensemble_results`, `ensemble_stat_map`, `resolve_results_template`, `resolve_origin_and_sites`, alpha helpers, `ReferenceModel`, `roi_free_mask`). |
| 2026-11-09 | Claude Sonnet 5.5 (Anthropic) | New `MOD_TITLE` (figure title: auto / none / text with placeholders / dict per figure), `MOD_PANEL_TITLES`, and the joint colour bar `MOD_CBAR_LOCATION` ("panel" / "right" / "left" / "top" / "bottom"), `MOD_CBAR_SHRINK`, `MOD_CBAR_THICKNESS`; passed to `fviz.plot_model_slices` (`panel_titles`, `cbar_location`, `cbar_shrink`, `cbar_thickness`; `figure_title=""` = no title). Defaults leave the figures unchanged. AI-generated; review before production use. |
| 2026-11-09b | Claude Sonnet 5.5 (Anthropic) | Plane sections: the `point` may be CRS-tagged (`([lon, lat, z], "latlon")`, `([E, N, z], "utm")`, `([x, y, z], "model")`, z in km), and dips other than 90 work: `strike` is the azimuth for every dip, axes are along-strike / down-dip distance from `point`, default extent = plane cut with the ROI box, optional `xlim` / `ylim`. Before, a dipping plane was a sheared, mislabelled panel. Needs `femtic.py` and `femtic_viz.py` of the same date. |
| 2026-11-10 | Claude Sonnet 5.5 (Anthropic) | `MOD_MESH_SWAP_XY` now defaults to `True` (was `None`) and its description is corrected (the old text claimed that no swap gives x = easting). Slice maps were drawn transposed (x = easting and y = northing exchanged; a NNE structure appeared as ENE). Cause: since 2026-10-11 `femtic.read_femtic_mesh` does not swap the node columns by default, but `mesh.dat` holds x = northing, y = easting. `None` keeps femtic's default, `False` forces no swap. Only the orientation changes. Checked on a synthetic mesh with a N30E block, not on real data. AI-generated; review before production use. |
