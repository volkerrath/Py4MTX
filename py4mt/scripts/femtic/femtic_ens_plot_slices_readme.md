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
| `MOD_STATS_WHAT` | list of str | `["best","avg","med","err","mad"]` + one key per percentile level + one per `QDIFF_PAIRS` entry (+ `"err_boot"`) | Which panels to plot. `"best"` plots the best-nRMS converged member (to `MOD_QC_FILE`) — the former `MOD_QC` switch. Every other entry is a statistic key: subset of `"avg"`, `"var"`, `"err"`, `"med"`, `"mad"`, `"var_boot"`, `"err_boot"`, the log10 sensitivity keys (`"sens_{mean,median,min,max,std}"`, each divided by its own max, so each peaks at 0), `"sens_cv"` (linear); with a results file that holds both kinds the same names are suffixed `_cum` / `_vn`, see *Raw or volume-normalised sensitivity*, `"simrc_coef"`/`"simrc_corr"` (log10), plus auto-generated percentile keys (e.g. `2.3` → `"p2_3"`, `50.0` → `"p50"`, `97.7` → `"p97_7"`) and qdiff keys (e.g. `(15.9, 84.1)` → `"qdiff_15_9_84_1"`). Also `"diff_prior"` / `"abs_diff_prior"` (not in the default list), see *Difference to the prior model*. `"err"` = `sqrt(var)` is plotted by default *instead of* `"var"`, since `var` is in (log10 Ω·m)² and isn't on the same scale as `MAD`/`QDIFF` (log10 Ω·m); add `"var"` back manually (with a `MOD_STATS_STYLE` entry) if you specifically want the raw-variance panel. |
| `MOD_SENS_KIND` | str | `"auto"` | Which sensitivity the `sens_*` entries of `MOD_STATS_WHAT` and the name in `MOD_ALPHA_SOURCE` stand for: `"raw"` (cumulative sum\|J\|, key suffix `_cum`), `"volume_normalised"` (raw / cell volume, suffix `_vn`; `"volume_normalized"` accepted), `"auto"` (the one kind of a single-kind file; the volume-normalised one of a file with both) or `"both"` (panels: every `sens_*` panel once per kind; the blanking uses the `"auto"` kind). See *Raw or volume-normalised sensitivity* below. |
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


### Raw or volume-normalised sensitivity

`femtic_ens_post.py` stores the sensitivity statistics of one kind under plain
keys (`SENS_KIND = "raw"` or `"volume_normalised"`) or of both kinds under
suffixed keys (`SENS_KIND = "both"`: `_cum` = raw, `_vn` =
volume-normalised). `MOD_SENS_KIND` picks among what the results file holds:

| `MOD_SENS_KIND` | file with one kind | file with both kinds | file from before 2026-11-15 |
|---|---|---|---|
| `"auto"` | its kind, plain names | volume-normalised (`..._vn`) | plain names, as before |
| `"raw"` | plain names if the file holds raw, else the entry is skipped with a message | `..._cum` | plain names, used with a note (the file does not say which kind it holds) |
| `"volume_normalised"` | plain names if the file holds that kind, else skipped with a message | `..._vn` | as above |
| `"both"` | its kind | every `sens_*` panel twice (`..._cum`, `..._vn`) | plain names |

- An entry that names the kind itself (`"sens_mean_cum"`, or
  `MOD_ALPHA_SOURCE = "sens_median_cum > -3."`) is used as it is whatever
  `MOD_SENS_KIND` says: e.g. blank by one kind and plot the other.
- A kind the file does not hold gives a message and no panel; the other kind's
  map is never drawn under the wrong name. If `MOD_ALPHA_SOURCE` cannot be
  resolved, blanking is switched off (printed).
- The two kinds have different spatial distributions, so a
  `MOD_ALPHA_SOURCE` threshold chosen for one is not valid for the other:
  look at the ROI diagnostic of each kind before choosing it.
- `MOD_STATS_STYLE` of the plain name (`"sens_mean"`) applies to both
  suffixed keys unless the suffixed key has its own entry. Colour bars of
  suffixed panels get the kind appended (`[raw]`, `[volume_normalised]`);
  block files and figures are named with the key actually plotted
  (`resistivity_block_<P>_sens_mean_vn.dat`, `<P>_sens_mean_vn.pdf`).
  Single-kind files keep the names and labels of earlier versions.
- **Normalisation.** Every `sens_*` statistic (`mean`, `median`, `min`,
  `max`, `std`) is stored by `femtic_ens_post.py` (2026-11-15 (c)) as log10
  of the statistic divided by **its own max** over the free blocks, so each
  peaks at exactly 0 and the default colour range is `[-5, 0]`;
  `sens_cv` (`std / mean`) is linear. The statistics therefore do not share
  a divisor: e.g. `sens_max` and `sens_mean` both peak at 0, and their native
  ratio is in `res["sens_ranges"][kind][stat] = (min, max)` (native value of
  a stored key = `10**key * max`). A threshold in `MOD_ALPHA_SOURCE` such as
  `"sens_median > -3."` is relative to the median's own maximum.
- **Results files from before 2026-11-15 (c)** hold `sens_<stat>_an` /
  `sens_<stat>_na` (and the native-units `sens_mean_raw`). The new plain
  names are then read from the nearest old key (`sens_mean` -> `sens_mean_an`,
  one common divisor for all statistics there) with a printed note; `_na`
  names have no equivalent. In the other direction, an old name such as
  `"sens_mean_an"` asked of a new file is read as `sens_mean`, with a note;
  `"sens_mean_na"` is reported and skipped. The legacy `MOD_STATS_STYLE`
  entries (`_na` / `_an`, native `sens_mean_raw`) are kept for old files.

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

### Difference to the prior model

When `femtic_ens_post.py` ran with `COMPUTE_PRIOR_DIFF` (its changelog
2026-11-15 (d)) the results file holds `diff_prior`, the **signed**
difference model(ens) - model(prior) in log10(Ohm.m), per free parameter.
Two names are then available in `MOD_STATS_WHAT`; neither is in the default
list:

| Name | Shows | Default style (`MOD_STATS_STYLE`) |
|---|---|---|
| `"diff_prior"` | `log10(rho_ens) - log10(rho_prior)`, signed; positive = the ensemble is more resistive than the prior | diverging `"RdBu"` (red = more conductive, blue = more resistive, as in the resistivity maps), `clim=[-1, 1]` (a factor of 10 either way), label `"log10(rho_ens / rho_prior)"` |
| `"abs_diff_prior"` | its absolute value (made when the file is read; the post script stores none) | `"viridis"`, `clim=[0, 1]`, label `"\|log10(rho_ens / rho_prior)\|"` |

```python
MOD_STATS_WHAT = MOD_STATS_WHAT + ["diff_prior", "abs_diff_prior"]
```

Either name can also be the `<name>` of `MOD_ALPHA_SOURCE`
(e.g. `"abs_diff_prior > 0.3"` shows only the cells that moved by more than a
factor 2), and either is masked by the alpha source like the `avg` panel
(`MOD_ALPHA_EXCLUDE` only exempts `sens_` / `simrc_` by default). A file
without the key (written before the change, `COMPUTE_PRIOR_DIFF = False`, or
no usable prior) skips the panel with a printed message. Which statistic
(mean / median) and which prior (the reference model file or the perturbed
references) it is, is in the line printed per panel (`STATS: plotting ...`)
and in the file's `<P>_diff_prior_info`. In a results file written with
`ENS_MODEL_SOURCE = "perturbed"` (prefix with `_perturbed`) the ensemble is
the set of perturbed models, so the difference exists only for a reference
prior and measures the sampling noise of the prior draws.

### Spread at iter0 and the ratio iterX / iter0

When `femtic_ens_post.py` ran with `COMPUTE_QDIFF_ITER0` and
`ENS_MODEL_SOURCE = "both"` (its changelog 2026-11-22) the results file holds,
for every `QDIFF_PAIRS` entry, the spread of the **iter0** models
(`qdiff0_<lo>_<hi>`) and the ratio of the converged spread to it
(`qratio_<lo>_<hi>` = `qdiff / qdiff0`, X = the best-nRMS iteration of each
member). Neither is in the default `MOD_STATS_WHAT`:

| Name | Shows | Default style (`MOD_STATS_STYLE`) |
|---|---|---|
| `"qdiff0_15_9_84_1"`, `"qdiff0_2_3_97_7"` | `\|P_hi - P_lo\|` of the iter0 / perturbed models, log10(Ohm.m): what the prior says | as `qdiff_*`: `clim=[0, 1]` |
| `"qratio_15_9_84_1"`, `"qratio_2_3_97_7"` | `qdiff / qdiff0`; below 1 the data narrowed the ensemble, near 1 the prior is what is left | `"coolwarm"`, `clim=[0, 2]` (1 is the neutral colour), label `"P84.1 - P15.9: iterX / iter0"` |

```python
MOD_STATS_WHAT = MOD_STATS_WHAT + ["qdiff0_15_9_84_1", "qratio_15_9_84_1"]
```

The ratio can also be the `<name>` of `MOD_ALPHA_SOURCE`
(`"qratio_15_9_84_1 < 0.5"` shows only the cells where the spread is less than
half the prior spread). It is `NaN` (not drawn) where the iter0 spread is zero,
e.g. when `referencemodel.dat` is a link and all members are identical. A file
without the keys (written before the change, `COMPUTE_QDIFF_ITER0 = False`, or
`ENS_MODEL_SOURCE` other than `"both"`) skips the panel with a printed
message; with `ENS_MODEL_SOURCE = "perturbed"` for the plots (the perturbed
models alone) the keys are dropped like `qdiff_*`.

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
| `MOD_SLICES` | List of slice-spec dicts (`kind`, `z0`/`x0`/`y0`). Kinds: `"map"`, `"ns"`, `"ew"`, `"plane"` (see "Vertical section" below) and `"profile"` (a fault from two surface points, see "Fault from two surface points" below). |
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
| `MOD_ALPHA_SOURCE` | Single condition `"<name> <op> <value>"` selecting the cells to **show** (e.g. `"sens_median > -3."`); `<name>` is any statistic key stored in the results file (see `MOD_STATS_WHAT`), `<op>` one of `> >= < <=`, `<value>` in that statistic's own units (log10 for sensitivities). Parsed with a regex, never `eval`ed. `None` = off. All names, units and examples: `femtic_ens_plot_slices_alpha.xlsx`, [Fading the model](#fading-the-model-mod_alpha_). |
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

### Fading the model (`MOD_ALPHA_*`)

`MOD_ALPHA_SOURCE` blanks or fades the model panels by one statistic of the
ensemble -- typically the cells the data hardly constrain. The settings are the
rows `MOD_ALPHA_SOURCE`, `MOD_ALPHA_MODE`, `MOD_ALPHA_FADE_WIDTH`,
`MOD_ALPHA_QC` and `MOD_ALPHA_EXCLUDE` of the table above, and
`MOD_SENS_KIND` for sensitivities. The workbook
**`femtic_ens_plot_slices_alpha.xlsx`** (in the folder of this readme) is the
reference for them:

| Sheet | Content |
|---|---|
| `About` | What the workbook is, the rule in three lines, provenance, colour legend. |
| `Settings` | Every setting that fades something: the six above, `SENS_KIND` of `femtic_ens_post.py`, and the fault keys `limit_km`, `alpha_outside`, `fade_km` -- with values, defaults and effect. |
| `Names` | The statistic names usable in `MOD_ALPHA_SOURCE` (`sens_*`, `avg`, `med`, `err`, `var`, `mad`, `p<level>`, `qdiff_<lo>_<hi>`, `err_boot`, `var_boot`, `simrc_*`): what each is, its units / range, the `femtic_ens_post.py` setting that stores it, an example condition. |
| `Name resolution` | What a name turns into for a results file with both sensitivity kinds, with `volume_normalised` only and with `raw` only, for each `MOD_SENS_KIND` -- including the cases in which nothing is found (generated with `fem.resolve_sens_stat`, the function the script calls). |
| `Fade calculator` | The alpha of a cell as a function of its value for a given name, operator, threshold, mode and width (live formulas; change the blue cells on yellow), and the three settings lines to paste into the script. |

The rules in short:

- `MOD_ALPHA_SOURCE = "<name> <op> <value>"`: **one** comparison, `<op>` one of
  `> >= < <=`, `<value>` in the units of the statistic (log10 for the
  sensitivities; they are divided by their own max, so <= 0). The cells that
  satisfy it are shown; NaN cells always get alpha 0. It is parsed with a
  regex, never evaluated, so two conditions cannot be combined.
- `MOD_ALPHA_MODE = "blank"`: the other cells are removed. `"fade"`: their alpha
  is `1 - distance / MOD_ALPHA_FADE_WIDTH` clipped to `0 ... 1`, `distance`
  being how far the value lies beyond the threshold. With `"sens_median > -5."`
  and width `1.`: alpha 1 down to -5, 0.5 at -5.5, 0 at -6 and below.
- **Sensitivity names depend on the file.** `femtic_ens_post.py` with
  `SENS_KIND = "volume_normalised"` (its default) or `"raw"` stores plain
  names (`sens_median`); with `"both"` it stores `sens_median_cum` (raw) and
  `sens_median_vn` (volume-normalised). A suffixed name is used only if the
  file holds that key: `"sens_median_vn > -5."` on a file with one kind is not
  resolved, a message is printed and the fading is **switched off**. The plain
  name `"sens_median > -5."` works for every file (for a file with both kinds
  it is the `_vn` map when `MOD_SENS_KIND = "auto"`).
- The `sens_*` and `simrc_*` panels themselves are drawn without the mask
  (`MOD_ALPHA_EXCLUDE`); `MOD_ALPHA_QC` extends it to the best-member figure.
- The mask is a model block file (`resistivity_block_<P>_alpha.dat`, raw values
  = alpha) that the slice plots read; the fault's own fade (`fade_km`) is a
  separate mechanism of the slice entry.

### Fault from two surface points

A `kind="profile"` entry with `dip` is the plane through the two surface
points `p1`, `p2`. The keys are degrees / metres even in this km script (`p1`,
`p2` = `([lon, lat], "latlon")`, `([E, N], "utm")` or `[x, y]` model-local
metres; `z_top`, `z_bot`, `xlim`, `ylim` in m), except `limit_km` and
`fade_km`, which are in km as their names say.

```python
dict(kind="profile",
     p1=([6.030869, 45.962166], "latlon"),
     p2=([6.115936, 45.910655], "latlon"),
     dip=110.0, z_top=0.0, z_bot=20000.0,
     overlay=dict(color="magenta", linestyle=":", linewidth=15.0, alpha=0.9),
     title="Vuache fault",
     limit_km=0.0, alpha_outside=0.25, fade_km=2.0,
     panel=False)
```

| Key | Default | Meaning |
|---|---|---|
| `dip` | `90` | The plane dips to the **right** of the direction p1 -> p2 for `dip < 90`; `dip = 180 - d` dips `d` degrees to the left. For the Vuache fault p1 -> p2 runs NW -> SE, so `dip=110.0` is **70 deg to the NE**; `dip=70.0` would dip SW. A third point of that plane: (6.079571, 45.941367) at 2 km depth. |
| `overlay` | none | `True` or `Line2D` keywords (`color`, `linestyle`, `linewidth`, `alpha`, `label`): draws the fault on every other panel of every figure (trace on the maps, dipping line on the sections). |
| `panel` | `True` | `False` = no panel of its own, only the lines; drop it to also get the fault panel (the model on the fault plane, along strike x down dip). |
| `limit_km` | `0.0` | **Limit to the defining points** (`femtic_viz.py` of 2026-11-15). The part of the fault between p1 and p2 is drawn as given; the rest of the plane -- the extrapolation along strike -- fades out. `limit_km` widens that stretch at both ends (`2.0` = 2 km beyond each point) or narrows it (`< 0`); `0` = exactly p1 ... p2; `None` = no distinction (one line, as before). |
| `alpha_outside` | `0.25` | Opacity (`0 ... 1`) of the fault **far outside** the stretch: of the overlay lines (relative to their own `alpha`: `0.9` x `0.25` = 0.225) and, on the fault panel, of the model (faded towards white). `0` = fully transparent there, `1` = no difference. |
| `fade_km` | `2.0` | Width (km) of the linear fade from full opacity at the ends of the stretch to `alpha_outside` -- the same ramp as `MOD_ALPHA_MODE = "fade"` for cells. `0` = abrupt step at the ends. With `alpha_outside = 0.0` the fault fades to nothing over `fade_km`. |

The default `limit_km = 0.0` applies to every fault entry (`dip != 90` or with
`overlay`), so an existing entry changes its look when the file is replaced:
the lines outside p1 ... p2 fade out. A plain vertical `kind="profile"`
without `overlay` keeps its look unless it carries `limit_km`. Details of the
geometry: `femtic_viz_readme.md`, "Fault planes".

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


### Polylines (optional)

`MOD_POLYLINES` is a dict `{name: entry}` of lines (faults, profiles,
boundaries, borehole traces, ...) drawn on the slice panels of every figure:
the `"best"` panel and all `MOD_STATS` panels. `{}` (default) draws nothing
and leaves the figures unchanged. Lengths are in **km** (like `MOD_SLICES`);
the script converts them with `fem.polylines_km_to_m` and hands them to
`fviz.plot_model_slices(polylines=...)`.

```python
MOD_POLYLINES = {
    "fault F1": dict(
        points=[[-4.0, -3.0, 0.5], [0.0, 0.0, 2.5], [4.0, 3.0, 4.5]],  # km
        style=dict(color="white", linewidth=2.0), max_dist=1.5),
    "profile": dict(points=([[6.05, 45.90], [6.15, 45.95]], "latlon"),
                    kinds="map", style=dict(color="k", linestyle=":")),
}
```

| Key of an entry | Default | Description |
|---|---|---|
| `points` | required | `[[x, y], ...]` or `[[x, y, z], ...]`, at least 2 rows, model-local km; z = depth, positive down (missing z = 0). Or CRS-tagged `(pts, "model" \| "utm" \| "latlon")`: `"utm"` = `[E, N(, z)]` in UTM **metres**, `"latlon"` = `[lon, lat(, z)]` in degrees (longitude first); z is km in every case. A bare point list instead of the entry dict is shorthand. |
| `style` | 1.5 pt solid, colour cycled | matplotlib `Line2D` keywords (`color`, `linewidth`, `linestyle`, `marker`, `alpha`, `zorder`, ...). |
| `label` | the dict key | Legend text (lower right of each panel, with the reference-model label); `None` = no entry. |
| `kinds` | all panels | `"map"`, `"ns"`, `"ew"`, `"plane"`, `"vertical"` (= ns + ew + plane) or a list of these. |
| `max_dist` | `MOD_PROJECTION_DIST` | km; vertical sections only: the parts of the polyline within `max_dist` of the section plane are projected into it (slab, like the site markers). `0` = only parts lying in the section plane. If `MOD_PROJECTION_DIST` is `None` the whole polyline is projected. |
| `z_range` | none | `[zmin, zmax]` km; maps only: draw only on maps whose `z0` lies in the range. Default: plan view on every map. |
| `panels` | all | A panel index or list of 0-based indices into `MOD_SLICES`; draw the line on those panels only (2026-11-12). |

Maps show the whole polyline in plan view. The polylines never change the axis
limits; parts outside the panel are clipped by the axes.

**Intersection line of two slices.** `fviz.slice_intersection_polyline(a, b,
xlim=..., ylim=..., zlim=...)` returns the line in which two slices meet
(clipped to the box and to the extents of the slices), in model-local metres;
`fviz.slice_intersection_polylines(slices, ...)` makes ready entries for many
pairs. Because the helpers work in metres, give the slices and limits in metres
and divide the result by 1000 for this km dict, e.g.

```python
_p = fviz.slice_intersection_polyline(
    dict(kind="map", z0=2000.0), dict(kind="ns", x0=0.0),
    xlim=[-20e3, 20e3], ylim=[-20e3, 20e3], zlim=[-1e3, 10e3])
MOD_POLYLINES["map 2 km x ns 0"] = dict(
    points=_p / 1000.0, max_dist=0.0, z_range=[2.0, 2.0])
```

`max_dist=0.0` draws the line only on the panels that contain it, `z_range`
only on the map at that depth. There is no automatic switch that draws the
intersections of all `MOD_SLICES`; a slice that carries `overlay` is drawn on
every other panel (next paragraph).

Needs the `femtic.py` (`polylines_km_to_m`) and `femtic_viz.py` of 2026-11-11.

**Fault from two surface points.** A `kind="profile"` entry takes an optional
`dip` (default 90 = vertical fence section as before); with `dip != 90` it is
the dipping plane through `p1` -> `p2`, which dips to the **right** of the
direction p1 -> p2 (`dip = 110` dips the same amount to the left). It is a
panel of its own (the model on the fault, along strike x down dip) and, with
`overlay`, also a line on every other panel of every figure of this script:
the trace on the maps, the dipping line on the ns / ew / plane sections.
With `panel=False` the entry is **not** a panel: you get the fault trace in all
the other slices and no fault panel (a slice along the fault is the same entry
without `panel=False`, or any `kind="plane"`).

```python
MOD_SLICES = [
    dict(kind="map", z0=0.0), dict(kind="map", z0=3.0),
    dict(kind="ns", x0=0.0), dict(kind="ew", y0=0.0),
    dict(kind="profile",
         p1=([6.030869, 45.962166], "latlon"),     # [lon, lat]
         p2=([6.115936, 45.910655], "latlon"),
         dip=70.0, z_top=0.0, z_bot=20000.0,       # metres
         overlay=dict(color="k", linewidth=1.5), title="fault",
         panel=False),                 # trace only; remove for a fault panel too
]
```

The profile keys (`p1`, `p2`, `z_top`, `z_bot`, `xlim`, `ylim`) are degrees /
UTM metres / model-local **metres** even in this km script. `z_top` is also
the depth at which the plane contains the trace (use `-elevation` for a trace
above sea level; an error `dz` shifts the plane horizontally by `dz / tan(dip)`).
`xlim=[-L/2, L/2]` (m along strike, relative to the midpoint of p1-p2) limits
the fault to the segment of length `L`. Details: `femtic_viz_readme.md`, "Fault
planes: a panel and an overlay". Needs `femtic.py` and `femtic_viz.py` of
2026-11-12.


### Point markers (`MOD_MAP_MARKERS`)

A list of dicts, drawn by `fviz.plot_model_slices(map_markers=...)` on every
figure of the script (`"best"` and all `MOD_STATS` panels). The script holds an
example:

```python
MOD_MAP_MARKERS = [
    # Epagny 1996: 45.9383 N, 6.0883 E, depth 2 +- 1 km
    dict(latlon=[45.9383, 6.0883], marker="*", color="black",
         markerfacecolor="yellow", markeredgecolor="black", ms=18, mew=1.0,
         zorder=14, depth=2.0, depth_err=1.0, name="Epagny 1996 (2 +- 1 km)"),
    # model centre (x = y = 0): black cross on maps and sections
    dict(is_model_centre=True, marker="+", color="black", ms=16, mew=2.2,
         kinds="all", depth=0.0, name="model centre"),
]
```

| Key | Meaning |
|---|---|
| `latlon` | `[lat, lon]`, degrees |
| `depth` | km, positive down (model z, like the z of a plane `point`). Maps ignore it; sections draw the marker at this depth |
| `depth_err` | km; the `+-` of a depth bar on the sections (`2 +- 1 km` = `depth=2.0, depth_err=1.0`); bar style `err_color` (default the marker's edge colour), `err_lw` (1.5), `err_cap` (cap width in points, 6; 0 = none) |
| `kinds` | panels that show it: `"map"`, `"ns"`, `"ew"`, `"plane"`, `"vertical"` (= ns + ew + plane), `"all"` or a list. Default `"all"` with a `depth`, else `"map"` |
| `max_dist` | km; a section shows the marker within this distance of its plane. Default `MOD_PROJECTION_DIST` (`None` there = on every section). The distance is measured to the marker's own position (x, y, depth) |
| `name` | legend label (lower right of the panel); `None` = none |
| `marker`, `color`, `ms`, `zorder` and any `Line2D` keyword | `markerfacecolor`, `markeredgecolor`, `mew`, ... |

- **Maps:** the markers are drawn on every map panel, whatever its depth. The
  model centre is drawn on maps only when `MOD_DISPLAY_COORDS` is `"utm"` or
  `"latlon"` (as before).
- **Sections:** a marker appears on an `ns`, `ew`, vertical or dipping `plane`
  panel (also a `kind="profile"` fault panel) when it is within `max_dist` of
  the section plane. In a dipping plane the axes are along-strike and down-dip
  distance, so the depth bar is the depth direction projected into the plane
  (`+- 1 km` of depth = `+- 1 km * sin(dip)` of down-dip distance). The
  markers never change the axis limits; one outside the view is clipped, one
  inside it is drawn whole, also on the frame.
- **Your slices:** with `MOD_PROJECTION_DIST = 1.0` the epicentre is shown on a
  section only within 1 km of it, so the example sections through it are in
  `MOD_SLICES` next to the Vuache entry (commented out): `dict(kind="ns",
  x0=(6.0883, "latlon"))` and `dict(kind="ew", y0=(45.9383, "latlon"))`;
  sections through the model origin are `dict(kind="ns", x0=0.0)` /
  `dict(kind="ew", y0=0.0)` (the cross is drawn at depth 0, on the top frame).
  The Vuache fault panel (`panel=True` on that entry) shows the epicentre too:
  computed from the two surface points and dip = 110, the plane at 2 km depth
  is about 0.2 km from it (the epicentre lies about 0.9 km NE of the surface
  trace and 5.1 km from p1 along it); the depth bar crosses the plane at about
  2.5 km depth.
- **Default centre:** without an `is_model_centre` entry the centre is the
  yellow star on the maps only (as before); `MOD_SHOW_MODEL_CENTRE = False`
  removes it everywhere.
- `MOD_MAP_MARKERS = []` = no markers; the figures are then unchanged. Entries
  of earlier versions (no `depth`) are drawn on the maps only, as before.
- `femtic_gst_prep.py` and `femtic_ens_plot.py` have their own marker settings
  and are not changed by this entry; their lists reach the same
  `plot_model_slices`, where the new keys are metres (no km conversion there).
- Needs `femtic.py` (`markers_km_to_m`) and `femtic_viz.py` of 2026-11-15
  (point markers).

### Site overlay

| Parameter | Description |
|---|---|
| `MOD_SITE_DAT` | Path to `site.dat` CSV; `None` to fall back to `MOD_SITE_NUMBER`. |
| `MOD_SITE_NAMES` | `None` = all sites; list of strings = subset. |
| `MOD_SITE_NUMBER` | Fallback site number(s) from `observe.dat` (int or list of ints), used only when `MOD_SITE_DAT` is unavailable. |
| `MOD_PLOT_SITES_MAPS / SLICES` | Toggle site markers on map / curtain panels. |
| `MOD_PROJECTION_DIST` | Maximum projection distance (m) for curtain panels; `None` = show all sites on every panel. |
| `MOD_SITE_MARKER / MARKER_SLICES` | Marker style dicts for map / curtain overlays. |
| `MOD_MAP_MARKERS` | Point markers (epicentres, wells, ...), a list of dicts; the default holds the example of the Epagny 1996 epicentre and the model centre, see "Point markers" below. On every map; on a section (`ns`, `ew`, `plane`, `profile` panel) those with a `depth` within `max_dist` of it. Lengths in km (`fem.markers_km_to_m`). `[]` = none. An entry with `"is_model_centre": True` (instead of `"latlon"`) sets the style, panels and depth of the model-centre marker. |
| `MOD_SHOW_MODEL_CENTRE` | `True` (default): marks the model origin on `"map"` panels whenever `MOD_DISPLAY_COORDS` is `"utm"`/`"latlon"` (no-op for `"model"`). Default style (since 2026-09-26): yellow star with a black edge, `ms=14`, at `zorder=15`; no legend entry. The `is_model_centre` entry of `MOD_MAP_MARKERS` overrides style, panels (`kinds`), depth and legend `name` (the script's default entry: black cross on the maps and the sections); `False` = no centre marker at all. |

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
femtic_ens_plot_slices_summary.md            settings of the run, then the contents of the results .npz that was read (in ENSEMBLE_DIR)
```

The contents of the results `.npz` (every key with shape, type and meaning) are printed
to the console after reading (if `OUT = True`) and appended to the summary file.

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
| 2026-11-11 | Claude Sonnet 5.5 (Anthropic) | New `MOD_POLYLINES = {name: entry}` (km; `{}` = none): lines drawn on the slice panels of every figure. Maps: plan view, optional `z_range`; ns / ew / plane sections: the parts within `max_dist` (default `MOD_PROJECTION_DIST`) of the section plane, projected into it. Points model-local km or CRS-tagged (`"utm"` metres, `"latlon"` degrees, longitude first); `style`, `label`, `kinds`. Passed to `fviz.plot_model_slices(polylines=...)` through `fem.polylines_km_to_m`. The intersection line of two slices is available as `fviz.slice_intersection_polyline` / `slice_intersection_polylines` (example in the Polylines section); it is not drawn automatically. With `MOD_POLYLINES = {}` the figures are unchanged. Needs `femtic.py` and `femtic_viz.py` of the same date. Tested on a synthetic mesh only. AI-generated; review before production use. |
| 2026-11-11b | Claude Sonnet 5.5 (Anthropic) | `_title_for()` now calls `fviz.resolve_title` (the shared title logic of all plot scripts); behaviour as before, plus: a dict key is also found by its text form (`0` finds `"0"`). The title and colour bar settings of this script (`MOD_TITLE`, `MOD_PANEL_TITLES`, `MOD_CBAR_*`) are unchanged. Needs the `femtic_viz.py` of the same date. |
| 2026-11-12 | Claude Sonnet 5.5 (Anthropic) | Fault from two surface points. A `kind="profile"` entry of `MOD_SLICES` takes `dip` (default 90 = as before; otherwise the dipping plane through p1 -> p2, dipping to the right of p1 -> p2) and any entry may carry `overlay` (`True` or `Line2D` keywords, + `label`): that plane is also drawn on every other panel (trace on maps, dipping line on sections). New polyline-entry key `panels`. Commented example in the script. Profile keys are degrees / metres, not km. Needs `femtic.py` and `femtic_viz.py` of the same date. Without `dip` / `overlay` the figures are unchanged. Tested on a synthetic mesh only. AI-generated; review before production use. New slice-spec key `panel=False`: the entry is no panel, it only draws its `overlay` line on all the other panels (fault trace without a fault panel). |
| 2026-11-15 | Claude Sonnet 5.5 (Anthropic) | Raw or volume-normalised sensitivity: new `MOD_SENS_KIND` (`"auto"` / `"raw"` / `"volume_normalised"` / `"both"`) chooses among the sensitivity kinds that `femtic_ens_post.py` (`SENS_KIND = "both"`) stores as `_cum` / `_vn` keys, for the `sens_*` entries of `MOD_STATS_WHAT` and for `MOD_ALPHA_SOURCE`. Names that carry the kind win; a kind the file does not hold is reported, never substituted. `MOD_STATS_STYLE` of the plain name applies to the suffixed keys. Single-kind and older results files give the same panels as before (checked against the previous script on synthetic results). Needs the `femtic.py` of the same date. Drawing itself not run here (tested with the drawing stubbed). AI-generated; review before production use. |
| 2026-11-15 (b) | Claude Sonnet 5.5 (Anthropic) | `sens_mean_raw` is stored max-normalised by `femtic_ens_post.py` (it was the one key in native units): its default `MOD_STATS_STYLE` is now `clim = [-5, 0]`, label `log10 S/Smax (mean, plain)`; results files from before that date (no `sens_ranges`) keep the native-units, auto-scaled style unless `MOD_STATS_STYLE["sens_mean_raw"]` is your own entry. Nothing else changed. Needs the `femtic.py` of the same date. Tested with the drawing stubbed out. AI-generated; review before production use. (Superseded by (c): `sens_mean_raw` is no longer stored.) |
| 2026-11-15 (c) | Claude Sonnet 5.5 (Anthropic) | Follows `femtic_ens_post.py` (c): the sensitivity statistics are `sens_{mean,median,min,max,std}` (log10, each divided by its own max) and `sens_cv` (linear), with `_cum` / `_vn` suffixes for a file with both kinds; `_na` / `_an` and `sens_mean_raw` are gone from new files. Defaults changed: `MOD_STATS_WHAT` has `"sens_mean"`, `"sens_median"`, `"sens_cv"`; `MOD_ALPHA_SOURCE = "sens_median > -3."` (relative to the median's own max; the old default was relative to the max of the mean, so the blanked area changes slightly); new `MOD_STATS_STYLE` entries `sens_*` (`clim = [-5, 0]`, label `log10 S/Smax (stat)`). Old names and old results files are mapped with a printed note (`fem.resolve_sens_stat`). Commented fault example in `MOD_SLICES`: `dip=110.0` (a profile dips to the RIGHT of p1->p2, so `dip=70` would have dipped to the SW for the Annecy fault). Needs the `femtic.py` of the same date. Tested with the drawing stubbed out and on synthetic results only. AI-generated; review before production use. |
| 2026-11-15 (f) | Claude Sonnet 5.5 (Anthropic) | **Workbook for the fading settings.** New `femtic_ens_plot_slices_alpha.xlsx` (sheets `About`, `Settings`, `Names`, `Name resolution`, `Fade calculator`): the settings that blank or fade the model, the statistic names usable in `MOD_ALPHA_SOURCE` with units and the `femtic_ens_post.py` setting that stores them, how a name is resolved for files with one or both sensitivity kinds (generated with `fem.resolve_sens_stat`), and a calculator of alpha from the cell value (checked against `fem.alpha_from_condition`: 200 cells, 4 operators, both modes, no difference). New section "Fading the model" above. Documentation only: no code change in the script. AI-generated; review before production use. |
| 2026-11-15 (e) | Claude Sonnet 5.5 (Anthropic) | **`alpha_outside` fades.** New optional key `fade_km` of the `kind="profile"` entry (km, default `2.0`): outside p1 ... p2 the opacity of the fault falls linearly from 1 at the ends of the stretch to `alpha_outside` over `fade_km`, like `MOD_ALPHA_MODE = "fade"` for cells (`fade_km = 0` = the abrupt step of (d); `alpha_outside = 0` fades to fully transparent). For the overlay lines `alpha_outside` is now relative to the line's own `alpha`; dotted / dashed lines keep their pattern across the fade. The Vuache entry carries `fade_km=2.0`. No code change in the script body; needs the `femtic_viz.py` of the same date. Tested on a synthetic mesh (drawing run); not on your data. AI-generated; review before production use. |
| 2026-11-15 (d) | Claude Sonnet 5.5 (Anthropic) | **Fault limited to its defining points.** The `kind="profile"` entry of `MOD_SLICES` takes `limit_km` (km, default `0.0`: the stretch p1 ... p2; `> 0` widens, `< 0` narrows it at both ends, `None` = off) and `alpha_outside` (default `0.25`): the fault between the two points is drawn as given, the rest fainter (overlay lines on the other panels; the model on the fault panel faded towards white). The Vuache entry carries both keys; the comment above it and the new section "Fault from two surface points" describe them. Based on the script as of 2026-11-15 16:12: all settings kept, `sens_*` names as in (c); MOD_ALPHA_SOURCE `"sens_median > -6."` (was `"sens_median_an > -6."`, now relative to the median's own max). No code change in the script; needs the `femtic_viz.py` of the same date. Tested on a synthetic mesh (drawing run); not on your data. AI-generated; review before production use. |
| 2026-11-15 (g) | Claude Sonnet 5.5 (Anthropic) | **Point markers on maps and sections.** `MOD_MAP_MARKERS` now holds the example of the Epagny 1996 epicentre (45.9383 N, 6.0883 E, depth 2 +- 1 km: yellow star with a depth bar) and the model centre (black cross, maps and sections). New entry keys `depth` / `depth_err` / `max_dist` (km), `kinds`, `err_color`, `err_lw`, `err_cap`; a marker with a depth is also drawn on `ns` / `ew` / `plane` / `profile` panels within `max_dist` (default `MOD_PROJECTION_DIST`) of the section. Commented example sections through both points next to the Vuache entry of `MOD_SLICES`. The call converts the list with the new `fem.markers_km_to_m`. With `MOD_MAP_MARKERS = []` the figures are unchanged. Needs `femtic.py` and `femtic_viz.py` of the same date. Tested on a synthetic mesh (drawing run, Agg backend); not on your data. AI-generated; review before production use. |
| 2026-11-15 (h) | Claude Sonnet 5.5 (Anthropic) | **Difference to the prior model.** `MOD_STATS_WHAT` accepts `"diff_prior"` (signed log10 difference model(ens) - model(prior), written by `femtic_ens_post.py` (d)) and `"abs_diff_prior"` (its absolute value); `MOD_STATS_STYLE` has entries for both (diverging `"RdBu"`, `clim=[-1, 1]`; `"viridis"`, `clim=[0, 1]`); either can be the name in `MOD_ALPHA_SOURCE`. Neither is in the default list. No code change in the script body. Needs the `femtic.py` of the same date and a results file from the post script of the same date. Tested on synthetic results (stubbed and real drawing, Agg); not on your data. AI-generated; review before production use. |
| 2026-11-22 | Claude Sonnet 5.5 (Anthropic) | **Spread at iter0 and the ratio iterX / iter0.** `MOD_STATS_WHAT` accepts `"qdiff0_<lo>_<hi>"` (spread of the iter0 / perturbed models) and `"qratio_<lo>_<hi>"` (= `qdiff / qdiff0`, written by `femtic_ens_post.py` of the same date); `MOD_STATS_STYLE` has entries for both (`qdiff0`: as `qdiff`; `qratio`: `"coolwarm"`, `clim=[0, 2]`); neither is in the default list (commented lines show them); either can be the name in `MOD_ALPHA_SOURCE`. A file without the keys skips the panel with a printed message. No code change in the script body; needs the `femtic.py` of the same date. Tested on synthetic results (reader, stat map, style entries); not on your data. AI-generated; review before production use. |
| 2026-11-22 (b) | Claude Sonnet 5.5 (Anthropic) | **Contents of the results `.npz`.** The list of the arrays of `ENSEMBLE_RESULTS` (key, shape, type, meaning) is printed after reading (if `OUT`) and appended to `femtic_ens_plot_slices_summary.md` (`fem.print_results_contents()`, `fem.append_results_contents()`; the `femtic.py` of the same date). Nothing else changes. Tested on synthetic results files; not on your data. AI-generated; review before production use. |
