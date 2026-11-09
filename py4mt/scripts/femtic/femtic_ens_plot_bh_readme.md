# femtic_ens_plot_bh.py

Borehole profiles of a FEMTIC ensemble, drawn from the results `.npz` of
`femtic_ens_post.py`.

Resistivity-depth profiles at vertical "boreholes" (`MOD_BH`): ensemble
statistics (mean / median / percentiles / sigma pairs), all members
(`"ensemble"` lines), a density heat map of the members (`"heatmap"`), the best
member and an optional reference model. If `femtic_ens_post.py` ran with
`ENS_MODEL_SOURCE = "both"`, the perturbed models of the same members are
drawn next to the converged ones on one common colour scale.

Nothing is recomputed: the element search is done once per borehole on the
mesh (`MOD_MESH`), the values are looked up in the stored ensemble. Needed on
disk besides the `.npz`: the mesh and one resistivity block file for the region
structure (by default the best member's own file recorded in the `.npz`;
`TEMPLATE_BLOCK` replaces it).

Since 2026-11-08 this is a separate script; the code was step (9) of
`femtic_ens_post.py`. Sampling (`femtic.sample_boreholes`, `BoreholeMesh`,
`heat_scale_*`) and drawing (`femtic_viz.plan_borehole_figures`,
`draw_borehole_figures`, `plot_boreholes`) moved to the modules; this script
holds the settings and the glue.

**One change in behaviour:** the default bottom of a profile (`z_bot`) is now
`MOD_ZLIM[1]` (else 20 km), not the region of interest derived from the sites
as in the old combined script. Entries that need another depth set `z_bot`.

---

## Workflow

```
<PREFIX>_results.npz  (femtic_ens_post.py)
        |
        v
  fem.read_ensemble_results, template block, origin / sites
        |
        v
  fem.sample_boreholes   element search per borehole (BoreholeMesh), values
        |                for avg / med / percentiles / members / best / ref
        v
  fviz.plan_borehole_figures -> fviz.draw_borehole_figures
        |
        +--> <MOD_BH_FILE>.<fmt>          figures
        +--> <MOD_BH_FILE>.npz            profiles (MOD_BH_NPZ)
        +--> <MOD_BH_FILE>_perturbed.npz  ("both" results files)
        +--> MOD_BH_HEAT_SCALE_FILE       JSON colour scale (optional)
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
| `ENS_PERTURBED_FILE` | str | Name of the perturbed reference file inside each member directory; only written into the header of the perturbed-profile file (the results file stores the name too, and that one wins). |

### Mesh, ocean and plot format

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_MESH` | str | `<ENSEMBLE_DIR>/templates/mesh.dat` | Tetrahedral mesh. |
| `MOD_MESH_SWAP_XY` | bool / None | `None` | Optional swap of the first two node-coordinate columns when reading `MOD_MESH`. |
| `MOD_OCEAN` | bool / None | `None` | Ocean flag; must match the inversion. `None` = auto. |
| `MOD_AIR_RHO`, `MOD_OCEAN_RHO` | float | `1.0e9`, `0.25` | Sentinel resistivities (Ohm.m) of air / ocean cells; ocean cells are drawn at `MOD_OCEAN_RHO`, air is a gap. |
| `MOD_PLOT_FORMAT` | list of str | `["pdf"]` | File extensions; one figure file per entry. |

### Profile defaults and fonts

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_ZLIM` | list or None | `None` | Default bottom of a profile (`z_bot` of a `MOD_BH` entry), `[zmin, zmax]` km, only zmax is used. `None` = 20 km. |
| `MOD_CLIM` | list or None | `[0.0, 4.0]` | `[log10_min, log10_max]` Ohm.m: resistivity range of the profile panels (default `xlim`). |
| `MOD_DPI` | int | `400` | Figure DPI. |
| `MOD_DEPTH_KM` | bool | `True` | Depth axis in km (`False`: m). |
| `MOD_TICK_FONTSIZE`, `MOD_LABEL_FONTSIZE`, `MOD_TITLE_FONTSIZE` | int | `16`, `16`, label + 10 | Axis annotation, label and figure title sizes. |
| `MOD_TITLE` | str, None or False | `"auto"` | Figure title. `"auto"` = `<ensemble directory name>  <ENSEMBLE_PREFIX>` (as before); `None`, `False` or `""` = **no title**; a string = used as given (`{prefix}`, `{dir}` are filled in; text with other braces is used as it stands). The tag `(converged | perturbed reference)` of a figure that shows both model sources is still appended to a title, not to "no title". The panel titles (borehole name, position) are not affected. Needs the `femtic_viz.py` of 2026-11-09. |
| `MOD_SHOW_IN_SPYDER` | bool | `True` | Also display saved figures inline when running inside Spyder. |

### Geographic / UTM origin and sites

Needed for `x`/`y` given as `"utm"` / `"latlon"` and for `sites`.

| Parameter | Description |
|---|---|
| `MOD_ORIGIN_METHOD` | `"box"` (bounding-box midpoint) / `"average"` / `None` (use literal values). |
| `MOD_UTM_ORIGIN_LAT/LON/E/N` | Manual override; ignored when `MOD_ORIGIN_METHOD` estimates from `MOD_SITE_DAT`. |
| `MOD_UTM_ZONE_OVERRIDE` | Force a specific UTM zone string; `None` = auto-detect. |
| `MOD_DISPLAY_COORDS` | `"model"` / `"utm"` / `"latlon"`; the coordinates shown in the panel titles. |
| `MOD_SITE_DAT` | Path to the `site.dat` CSV (`mt_make_sitelist.py`); source of `sites` in `MOD_BH` and of the origin estimate. |

### Reference model (optional)

A reference model drawn as a dashed line (`"reference model"`) in every
borehole panel: e.g. a deterministic inversion, the prior/starting model or a
published model. It is a resistivity block file on the **same mesh** as
`MOD_MESH`; its region structure may differ from the members'. With the usual
setting `MOD_REF_FILE = ENSEMBLE_DIR + "/templates/referencemodel.dat"` it is
the **base (unperturbed) reference**, not a member's perturbed
`referencemodel.dat`.

| Parameter | Default | Description |
|---|---|---|
| `MOD_REF_FILE` | `None` | Block file of the reference model; `None` = off. A missing file, or one whose element count differs from `MOD_MESH`'s, switches the reference off with a printed message; the run continues. |
| `MOD_REF_BH` | `True` | Add the reference profile to every `MOD_BH` panel. |
| `MOD_REF_LABEL` | `"reference model"` | Legend entry; `None` = none. |

In `MOD_BH`, the reference line is added last (drawn on top) unless the entry
has `ref=False`; listing `"ref"` in an entry's `what` chooses its position and
forces it on even when `MOD_REF_BH` is `False`. Its style is
`MOD_BH_STYLE["ref"]` (default black, `lw=2.2`, dashed). Reference values are
sampled in the same elements as the ensemble lines, with air as gaps and ocean
cells at `MOD_OCEAN_RHO`.

### Borehole profiles (`MOD_BH`)

1-D resistivity-depth profiles through the ensemble at one or more vertical
"boreholes": ensemble statistics, every member (drawn semi-transparent),
a density **heat map** of the ensemble (`"heatmap"`), and/or the best member,
one panel per borehole. `MOD_BH` is a list of dicts in the same spirit as
`MOD_SLICES`; an empty list (`MOD_BH = []`) draws nothing.

| Key | Meaning |
|---|---|
| `name` | Panel title and key in the `.npz` (spaces become `_`). Required. |
| `x`, `y` | Horizontal position. Plain float = model-local **km**; `(value, "utm")` = UTM metres in the mesh zone; `(value, "latlon")` = decimal degrees (x = longitude, y = latitude). Both must carry the same tag. Required unless `sites` is given. |
| `sites` | Instead of `x`/`y`: `"all"` (every site in `MOD_SITE_DAT`) or a list of site names. The entry is a template, expanded to one borehole per site (named after the site) with all its other keys (`z_top`, `z_bot`, `dz`, `what`, `alpha`, `xlim`, `style`) applied to each. Site UTM coordinates are used, so it needs `MOD_SITE_DAT` and a resolvable UTM origin. Names are made unique (`S1`, `S1_2`, ...) because they key the `.npz`. |
| `n_random`, `seed` | With `sites`: use only `n_random` sites drawn at random (without replacement) from the selected ones. `seed` (int) makes the draw reproducible; `None` (default) draws afresh each run. The chosen names are printed. |
| `z_top` | Float km (z down, model datum) or `"surface"` (default) = top of the first non-air element at (x, y); offshore that is the sea surface, so give a float to start at the sea floor. Resolved to about dz/5. |
| `z_bot` | Float km. Default `MOD_ZLIM[1]` if set, else 20 km. (There is no region-of-interest default in this script; the old combined script derived it from the sites.) |
| `dz` | Sampling interval, km. Default 0.05. |
| `what` | List of items to draw, in drawing order: `"avg"` (ensemble mean of log10 rho), `"med"`, `"percentiles"` (one line per `PERCENTILES` level), single percentile keys (`"p50"`, `"p15_9"`, ... as in `MOD_STATS_WHAT`), `"sigma1"` / `"sigma2"` (the 1- / 2-sigma quantile pairs P15.87 / P84.13 and P2.28 / P97.72 of the members in log10 rho, two lines each, computed directly, so independent of `PERCENTILES`), `"ensemble"` (all accepted members), `"heatmap"` (density heat map of the members, always drawn at the bottom; see "Heat map" below), `"best"` (best-nRMS member), `"ref"` (the reference model, `MOD_REF_FILE`; added automatically as the last item when `MOD_REF_BH` is `True`). Default `["avg", "percentiles"]`. Unknown names are reported and skipped. |
| `ref` | `False` switches the automatic reference line off for this entry (when `MOD_REF_BH` is `True`). Default `True`. |
| `alpha` | Opacity (0..1) of the `"ensemble"` lines. Default `MOD_BH_ALPHA`. Only used when `"ensemble"` is in `what`. |
| `xlim` | `[rho_min, rho_max]` in Ohm.m for this panel (default `MOD_BH_XLIM`). With `"heatmap"`: also the range of the bins. |
| `heat_bins` | Bins of the `"heatmap"` of this entry: an int (equal bins in log10 rho across the panel's x range) or an increasing list of log10(rho) bin edges. Default `MOD_BH_HEAT_BINS`. |
| `style` | Per-entry style overrides, keyed like `MOD_BH_STYLE`, e.g. `dict(avg=dict(color="navy", lw=3))`. |

Median and 1-sigma quantiles over a faint ensemble (the default entry);
items are drawn in the order of `what` and `"ensemble"` is always at the
bottom, so list it first:

```python
MOD_BH = [
    dict(sites="all", n_random=6, seed=1, z_bot=5.0,        # 6 random sites
         what=["ensemble", "med", "sigma1"], alpha=0.05),
    dict(sites="all", z_bot=5.0, what=["avg", "ensemble"], alpha=0.1),  # one panel per site
    dict(name="BH-1",
         x=(-70.868025, "latlon"), y=(-16.363436, "latlon"),
         z_top="surface", z_bot=4.0, dz=0.05,
         what=["avg", "percentiles", "ensemble"], alpha=0.15),
    dict(name="BH-2", x=2.0, y=-3.0, z_top=0.0, z_bot=6.0,
         what=["med", "p15_9", "p84_1", "best"]),
    dict(sites=["SITE01", "SITE07"], z_bot=5.0, xlim=[1.0, 1.0e4],      # heat map
         what=["heatmap", "med", "sigma1"]),                            # + lines on top
]
```

Global settings:

| Parameter | Default | Description |
|---|---|---|
| `MOD_BH_FILE` | `<dir><prefix>_boreholes` | Output path without extension; one figure per `MOD_PLOT_FORMAT` entry. |
| `MOD_BH_NPZ` | `True` | Also write `<MOD_BH_FILE>.npz`. |
| `MOD_BH_ALPHA` | `0.15` | Default opacity of the ensemble lines. |
| `MOD_BH_XLIM` | `None` | x range in Ohm.m (log axis). `None` = `10**MOD_CLIM`, shared with the slice figures; auto if `MOD_CLIM` is `None` too. |
| `MOD_BH_PANEL_SIZE` | `[10.0, 15.0]` | Panel width, height in cm. |
| `MOD_BH_NCOLS` | `None` | Panel columns; `None` = up to 4. |
| `MOD_BH_PER_FIG` | `None` | Maximum panels per figure; further figures get `_01`, `_02`, ... suffixes. Useful with `sites`. `None` = all in one. |
| `MOD_BH_SHARE_DEPTH` | `True` | One common depth range (union over all entries) for all panels. |
| `MOD_BH_LEGEND_LOC` | `"lower right"` | Legend position; `None` = no legend. |
| `MOD_BH_STYLE` | see script | Line style per item: `"avg"`, `"med"`, `"best"`, `"ensemble"`, `"percentiles"` (all percentile lines), one percentile key, or `"sigma1"` / `"sigma2"` (both lines of the pair; defaults navy dashed / steelblue dotted). Any Matplotlib `Line2D` keyword; `"percentiles"` may carry `"cmap"` (default `"viridis"`) to colour the lines by level. `"heatmap"` takes any `pcolormesh` keyword (default `dict(rasterized=True)`) and `cmap` to override `MOD_BH_HEAT_CMAP`; the colour scale itself is set by the `MOD_BH_HEAT_*` settings. |
| `MOD_BH_HEAT_BINS` | `60` | Heat map: number of equal bins in log10 rho across the panel's x range (entry `xlim`, else `MOD_BH_XLIM`, else `10**MOD_CLIM`; if none is set, the range of the member values at that borehole), or a list of log10(rho) bin edges. Per-entry `heat_bins` overrides it. |
| `MOD_BH_HEAT_NORM` | `"fraction"` | Colour scale: `"fraction"` = fraction of the members per bin (a depth level sums to 1 when all members are inside the range), `"row_max"` = every depth level scaled to its own maximum (0..1). Anything else stops the run at start. |
| `MOD_BH_HEAT_CMAP` | `"Greys"` | Colormap of the heat map (light = few members). A single-hue light-to-dark map (`"Greys"`, `"Blues"`, `"Purples"`, ...) keeps the lines on top readable. `MOD_BH_STYLE["heatmap"]` or an entry's `style=dict(heatmap=dict(cmap=...))` overrides it; the colour bar uses the cmap of the first panel with a heat map. |
| `MOD_BH_HEAT_SCALE` | `"linear"` | Shape of the colour scale: `"linear"`, `"sqrt"` (square root: lifts low densities, keeps peaks distinguishable) or `"log"` (logarithmic between `MOD_BH_HEAT_VMIN` and `MOD_BH_HEAT_VMAX`; empty bins are left blank; 1-2-5 colour-bar ticks when the range is under 3 decades). One scale per figure. Anything else stops the run at start. |
| `MOD_BH_HEAT_VMIN` | `None` | Lower end of the colour scale. `None` = 0 (`"linear"` / `"sqrt"`) or the smallest non-zero fraction of the figure (`"log"`, which needs a value > 0). A number sets it, e.g. `0.02` to blank out bins holding less than 2 % of the members (an arrow on the colour bar marks values below it). |
| `MOD_BH_HEAT_VMAX` | `None` | Upper end of the colour scale (`"row_max"` is always 1). `None` = for `"linear"` / `"sqrt"` the 99th percentile of the non-zero fractions of the figure (the colour bar ends in an arrow), for `"log"` the maximum; `1.0` = full range. |
| `MOD_BH_HEAT_CBAR` | `True` | Draw the shared colour bar of the heat map (one per figure). |
| `MOD_BH_HEAT_SHARE_SCALE` | `False` | `False` = one colour scale per figure (the default, as before). `True` = ONE scale for ALL figures of the run (every `MOD_BH_PER_FIG` chunk, converged and perturbed): the automatic `MOD_BH_HEAT_VMAX` is taken over all of them. With `ENS_MODEL_SOURCE = "both"` the converged and the perturbed heat maps of one figure always share the scale. See "Sharing the colour scale" below. |
| `MOD_BH_HEAT_FIGURE` | `"combined"` | Results file written with `ENS_MODEL_SOURCE = "both"` only. `"combined"` = converged and perturbed panel of each borehole side by side in one figure (`2 x MOD_BH_NCOLS` columns), one colour bar. `"separate"` = one figure per source (`<MOD_BH_FILE>` and `<MOD_BH_FILE>_perturbed`), same scale. |
| `MOD_BH_HEAT_SCALE_FILE` | `None` | JSON file that saves / reuses the colour scale across runs (same format as `BOREHOLE_HEAT_SCALE_FILE` of `femtic_ens_plot.py`); `None` = off. Values read from it replace `MOD_BH_HEAT_SCALE` / `_NORM` / `_VMIN` / `_VMAX` / `_CMAP` and, if it holds `"edges"`, the bins of every heat map. |
| `MOD_BH_HEAT_SCALE_IO` | `"auto"` | `"auto"` = read `MOD_BH_HEAT_SCALE_FILE` if it exists, else write it (one setting serves the first and every later run); `"read"` = read it, stop if missing; `"write"` = always (over)write it. |

Shared with the slice figures: `MOD_PLOT_FORMAT`, `MOD_DPI`, `MOD_CLIM`
(x range), `MOD_TICK_FONTSIZE` / `MOD_LABEL_FONTSIZE` / `MOD_TITLE_FONTSIZE`,
`MOD_DEPTH_KM`, `MOD_DISPLAY_COORDS` (panel-title coordinates),
`MOD_OCEAN` / `MOD_OCEAN_RHO`, `MOD_MESH`, and the UTM-origin resolution of
step (5) (needed for `"utm"` / `"latlon"` positions).

#### Heat map

`"heatmap"` in `what` shows the whole ensemble as a density image: at every
depth level the log10 resistivities of the members are histogrammed
(`MOD_BH_HEAT_BINS` bins) and drawn with `pcolormesh` behind all other items
(depth axis down, log resistivity axis). It sits on a single colour scale per
figure, with one colour bar; the colormap is taken from the first panel that
draws a heat map. It can be used alone or combined with any other item, e.g.
`what=["heatmap", "med", "sigma1"]` (lines on top of the image) or
`["heatmap", "ensemble"]`.

Colour scale examples (settings at the top of the script):

```python
MOD_BH_HEAT_SCALE = "log"       # logarithmic colours, empty bins blank
MOD_BH_HEAT_VMIN = 0.05         # ignore bins with < 5 % of the members
MOD_BH_HEAT_CMAP = "Blues"      # single-hue colormap
MOD_BH_HEAT_NORM = "row_max"    # or scale every depth to its own maximum
```

- **What a cell means.** With `MOD_BH_HEAT_NORM = "fraction"` the colour is
  the fraction of the valid members whose resistivity at that depth falls into
  the bin. This is comparable between depths and boreholes. `"row_max"`
  rescales each depth to its most populated bin and shows the shape of broad
  distributions better.
- **Range.** The bins span the panel's x range, so what you see is exactly
  what is binned. Members outside that range are not drawn (e.g. ocean cells
  at 0.25 Ohm.m with an x range of 1 ... 1e4 Ohm.m), but still count in the
  denominator, so such a level sums to less than 1 and ocean levels stay empty
  instead of piling up in the first bin. Widen `xlim` to see them.
- **Air / outside the mesh** are blank (NaN rows), as for the lines.
- **Colour scale.** `MOD_BH_HEAT_SCALE` (`"linear"` / `"sqrt"` / `"log"`),
  `MOD_BH_HEAT_VMIN` and `MOD_BH_HEAT_VMAX` set the scale of the whole figure;
  `MOD_BH_HEAT_CMAP` the colormap. Values outside an explicit range get the end
  colours and an arrow on the colour bar. On a log scale the empty bins (and
  the air) are blank, so the colours only show where members are.
- **Auto colour maximum.** Levels in which every member has the same value
  (the ocean, other fixed regions) have fraction 1 and would flatten the rest of
  the image on a linear scale, so `MOD_BH_HEAT_VMAX = None` uses the 99th
  percentile of the non-zero fractions (`"linear"` / `"sqrt"`) and marks the
  clipped end with an arrow on the colour bar; a log scale shows the whole range.
- **Cost.** The members are sampled once per borehole (one array lookup each),
  as for `"ensemble"`; the histogram is one `bincount` per borehole.
- **Export.** The `.npz` gets `heat_<name>` (`n_levels x n_bins` fractions,
  independent of `MOD_BH_HEAT_NORM`) and `heat_edges_<name>` (log10 Ohm.m bin
  edges). `"ensemble"` lines are only drawn / exported when `"ensemble"` is in
  `what`.
- An invalid `heat_bins` is reported and only that entry's heat map is skipped.

#### Sharing the colour scale

Colours of two heat maps are only comparable if the **bins** (bin width changes the fractions), the **colour limits** and the **colormap** agree. The defaults leave every figure on its own scale (output unchanged). To compare:

- **Converged and perturbed in one run:** a results file written with `ENS_MODEL_SOURCE = "both"`; both ensembles of a figure are binned on the same edges and share one scale (see above).
- **All figures of a run** (e.g. many sites over several `MOD_BH_PER_FIG` chunks): `MOD_BH_HEAT_SHARE_SCALE = True`. For identical bins in every borehole give an `xlim` (or `MOD_BH_XLIM` / `MOD_CLIM`) and an integer `heat_bins`, or a list of edges; without an `xlim` the bins follow each borehole's own value range.
- **Across runs** (a converged and a perturbed run, other ensembles, `femtic_ens_plot.py`): `MOD_BH_HEAT_SCALE_FILE`. The first run writes the resolved scale, every later run with the same setting reads it:

```python
MOD_BH_HEAT_SCALE_FILE = RESULTS_DIR + "heat_colorscale.json"
MOD_BH_HEAT_SCALE_IO = "auto"       # first run writes, later runs read
```

The file is plain JSON; every key is optional (a missing or `null` key keeps this run's `MOD_BH_HEAT_*` value), so a hand-written file with just `vmax` works:

```json
{"scale": "sqrt", "norm": "fraction", "vmin": 0.0, "vmax": 0.35, "cmap": "Greys",
 "edges": [0.0, 0.1, 0.2, 0.3, 0.4], "info": {"note": "ignored when reading"}}
```

`edges` are log10(Ohm.m) bin edges; `info` is ignored. When a file is read it applies to all heat maps of the run and the read is logged. Notes: a `cmap` set in `MOD_BH_STYLE["heatmap"]` or an entry's `style` still wins over the file's; the file is written only if there is a single scale for the run (`MOD_BH_HEAT_SHARE_SCALE = True`, or all profiles in one figure; otherwise a message says so), and `"edges"` only if all heat maps use the same bins; a missing file with `"read"`, or an invalid one (unknown `scale` / `norm`, negative `vmin`, `"log"` with `vmin <= 0`, unsorted `edges`), stops the run with a message.

Notes:

- **Depth convention:** z is positive downward, as everywhere in FEMTIC. The
  vertical axis increases downward and is labelled depth; values are model z,
  so levels above the datum (topography, `z_top` above sea level) are
  negative. `z_top`, `z_bot` and the figure axis use the same convention.
- Statistics are formed in log10(rho) space and shown in Ohm.m, so `"avg"`
  is the geometric mean, exactly as in the slice panels.
- Air and levels outside the mesh are gaps (NaN). Ocean cells show
  `MOD_OCEAN_RHO` when region 1 is an ocean block; other fixed regions keep
  their template value.
- The tetrahedron containing each level is found once per borehole (exact
  barycentric test, same as `fem.extract_borehole_log`), then reused for all
  statistics and all members.
- Only value-scale statistics (`avg`, `med`, percentiles) can be drawn; the
  spread and sensitivity statistics have different units.
- All members must share the region structure of the best member (as already
  assumed for the statistics block files); otherwise the block is skipped
  with a message.


---

## Output files

`<MOD_BH_FILE>.<fmt>` (one per `MOD_PLOT_FORMAT` entry) and, if `MOD_BH_NPZ`,
`<MOD_BH_FILE>.npz` with `depth_<name>` (m, z down), one array per drawn item
`<key>_<name>` (`avg`, `med`, `p50`, `best`, `ref`, ... in Ohm.m, NaN = air /
outside mesh), `ensemble_<name>` (`n_members x n_levels`, only if drawn),
`heat_<name>` (`n_levels x n_bins` fraction of the members per bin) and
`heat_edges_<name>` (`n_bins + 1` bin edges in log10 Ohm.m), both only with
`"heatmap"`, and a JSON `header` (`ensemble_source` (`"converged"` |
`"perturbed"`), member files, best member, `reference_file` (null without a
reference), percentile levels, `heat_unit`, per-borehole position, z range,
keys and `heat_bins`, and `colour_scale`: the heat-map `scale`, `norm`,
`cmap`, `vmin` / `vmax` (one number if the scale is shared by all figures,
else a list with one entry per figure) and `shared`; `null` without a heat
map). For a results file written with `ENS_MODEL_SOURCE = "both"` the same arrays for the perturbed
models are written to `<MOD_BH_FILE>_perturbed.npz` (header
`ensemble_source = "perturbed"`; the heat maps of both files use the same
`heat_edges_<name>`).


The settings of the run are written to `femtic_ens_plot_bh_summary.md` in `ENSEMBLE_DIR`.

---

## Quick start

```bash
conda activate EM
python femtic_ens_post.py          # once: scan, compute, write the .npz
# edit ENSEMBLE_DIR, ENSEMBLE_PREFIX and the MOD_BH list in femtic_ens_plot_bh.py
python femtic_ens_plot_bh.py       # repeat as often as needed
```

Converged and perturbed models together: run `femtic_ens_post.py` with
`ENS_MODEL_SOURCE = "both"`, then this script (`MOD_BH_HEAT_FIGURE`
`"combined"` or `"separate"`).

---

## Relationship to other scripts

| Script | Purpose |
|---|---|
| `femtic_ens_post.py` | Writes the results `.npz` this script reads. |
| `femtic_ens_plot_slices.py` | Slice figures from the same `.npz`. |
| `femtic_data_misfit.py` | Quality control; its method `"nrms"` draws the nRMS histogram from the same `.npz`. |
| `femtic_ens_plot.py` | Superseded (2026-11-08); its heat-map colour-scale JSON format (`MOD_BH_HEAT_SCALE_FILE`) is still read and written by this script. |
| `femtic_mod_plot_bh.py` | Borehole profiles of single models ("mod" family): one block file, or a list / glob / run directories with an iteration selector, one figure per model and an optional catalog PDF. |

---

## Changelog

| Date | Author | Change |
|---|---|---|
| 2026-11-08 | Claude Sonnet 5.5 (Anthropic) | Created: the borehole block (`MOD_BH`, step 9) of `femtic_ens_post.py` as a stand-alone script that reads the results `.npz` (including `<P>_ens_perturbed` for the converged-and-perturbed layout). Sampling in `femtic.py` (`sample_boreholes`, `BoreholeMesh`, `heat_scale_*`), drawing in `femtic_viz.py`. Default `z_bot` is `MOD_ZLIM[1]` (else 20 km) instead of the site-derived ROI. |
| 2026-11-09 | Claude Sonnet 5.5 (Anthropic) | New `MOD_TITLE` (figure title: auto / none / text); `BoreholeFigureOptions.suptitle = None` draws no title. Default unchanged. AI-generated; review before production use. |
