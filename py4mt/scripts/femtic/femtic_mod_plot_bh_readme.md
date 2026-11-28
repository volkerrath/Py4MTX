# femtic_mod_plot_bh.py - README

1-D resistivity-depth logs along boreholes for single FEMTIC resistivity
models.

The settings are those of `femtic_ens_plot_bh.py` (same `MOD_*` / `MOD_BH_*`
names, same `MOD_BH` entries, lengths in km) without everything that belongs to
ensemble statistics. Several runs can be given, as in `femtic_mod_misfit.py`:
every entry is a model of its own and gets its own figure(s) and profile file.

Sister scripts:
[`femtic_mod_plot_slice.py`](femtic_mod_plot_slice_readme.md),
[`femtic_mod_plot_3d.py`](femtic_mod_plot_3d_readme.md).
Sampling: `femtic.sample_model_boreholes()`; drawing:
`femtic_viz.draw_borehole_figures()` (the routine of the ensemble script).

The defaults are those of `femtic_ens_plot_bh.py` (2026-11-26).

AI-generated documentation (Claude Sonnet 5.5, Anthropic, 2026-11-26); review
before use.

---

## Usage

```
python femtic_mod_plot_bh.py                          # RUN_PATHS
python femtic_mod_plot_bh.py kra_rto_0 kra_rto_1      # two run directories
python femtic_mod_plot_bh.py "kra*/"                  # quote globs
python femtic_mod_plot_bh.py -w /data/ann ann_rto_orig-err_0/resistivity_block_iter10.dat
```

Command-line entries replace `RUN_PATHS`; `-w` / `--work-dir` replaces
`WORK_DIR`.

## Runs

| Setting | Default | Meaning |
|---|---|---|
| `WORK_DIR` | `.../annecy2026/RTO/ann_rto_orig-err/` | base of relative paths; `None` / `""` = current directory |
| `RUN_PATHS` | `["ann*/"]` | run directories, block files, `results_iterN.h5` files (= the block file of that iteration) or globs |
| `ITERATION` | `"best"` | for run directories: `"best"`, `"last"`, `N`, or a list |
| `CNV_FILE` | `["femtic.cnv", "FEMTIC-DABIC.cnv"]` | convergence file; first name that exists |

`MOD_MESH`, `MOD_SITE_DAT` and `MOD_REF_FILE` are looked up per run (run
directory first, then `WORK_DIR`; absolute paths serve every run). A model that
fails is reported and skipped. Return code: `0` all models found were plotted,
`1` one failed or none was found, `2` `WORK_DIR` missing.

## Output

```
<dir>/[<OUT_PREFIX>_]<label>_<tag>_boreholes.<fmt>
<dir>/[<OUT_PREFIX>_]<label>_<tag>_boreholes.npz
```

| Setting | Default | Meaning |
|---|---|---|
| `MOD_PLOT_FORMAT` | `["pdf"]` | one or more extensions |
| `OUT_DIR`, `OUT_PREFIX`, `OUT_PREFIX_DIR` | `None`, `None`, `True` | as in the slice script |
| `MOD_BH_FILE` | `None` | explicit path without extension (one model only) |
| `MOD_BH_NPZ` | `True` | write the profiles |
| `MOD_CATALOG` | `None` | multi-page PDF of all figures: `"auto"` (`catalog_boreholes.pdf`) or a path |
| `MOD_DPI` | `400` | resolution |
| `MOD_SHOW_IN_SPYDER` | `True` | show figures inline under Spyder |

More than `MOD_BH_PER_FIG` boreholes give figures `_01`, `_02`, ... The
parameter summary `femtic_mod_plot_bh_summary.md` is written to `WORK_DIR`.

### NPZ contents

| Key | Content |
|---|---|
| `depth_<name>` | m, positive down |
| `model_<name>` | Ohm.m, NaN = air / outside the mesh |
| `ref_<name>` | reference profile, if drawn |
| `header` | JSON: model, iteration, nRMS, alpha, mesh, reference, units, boreholes |

Blanks in `<name>` become `_`.

```python
import json
import numpy as np

z = np.load("ann_rto_orig-err_0_iter10_boreholes.npz")
info = json.loads(str(z["header"]))
depth, rho = z["depth_SITE01"], z["model_SITE01"]   # site name
```

## Mesh and origin

`MOD_MESH`, `MOD_MESH_SWAP_XY` (default `True`: `mesh.dat` holds x = northing,
the positions are x = easting), `MOD_OCEAN`, `MOD_AIR_RHO`, `MOD_OCEAN_RHO`,
`MOD_UTM_ORIGIN_LAT/LON/E/N` (E / N in UTM km), `MOD_UTM_ZONE_OVERRIDE`, `MOD_ORIGIN_METHOD`
(`"box"`, `"average"`, `None`), `MOD_SITE_DAT`, `MOD_DISPLAY_COORDS`
(`"latlon"` default; position text of titles and legend). Mesh and site list
default to `templates/mesh.dat` and `templates/site.dat`.

## Boreholes (`MOD_BH`)

| Key | Default | Meaning |
|---|---|---|
| `name` | required | panel title / legend, key in the NPZ |
| `x`, `y` | required | model-local km, or `(value, "utm")` (UTM km) / `(value, "latlon")` (x = longitude, y = latitude) |
| `z_top` | `"surface"` | km, positive down, or `"surface"` |
| `z_bot` | `MOD_ZLIM[1]`, else 20 | km |
| `dz` | `0.05` | km |
| `xlim` | `MOD_BH_XLIM` | Ohm.m, this panel |
| `what` | `["model"]` | may add `"ref"`; ensemble items (`"avg"`, `"ensemble"`, `"heatmap"`, ...) are ignored with a note |
| `ref` | | `False` = no reference line in this panel |
| `style` | | overrides of `MOD_BH_STYLE`, e.g. `dict(model=dict(color="red", ls=":"))`; `color`, `ls`, `lw`, `marker` may also be given directly (model line) |
| `sites` | | instead of `x` / `y`: `"all"` or a list of names from `MOD_SITE_DAT`, one borehole per site |
| `n_random`, `seed` | | with `sites`: random selection |

An `MOD_BH` list of `femtic_ens_plot_bh.py` can be used as it is.

Default: one panel per site of the site list, surface to 4 km, 1 to 10000
Ohm.m. Further forms:

```python
MOD_BH = [
    dict(name="SV-108",
         x=(6.053423, "latlon"), y=(45.92574, "latlon"),
         z_top="surface", z_bot=2.0, dz=0.05, xlim=[1.0, 10000.0],
         style=dict(model=dict(color="steelblue", ls="-", lw=3.2))),
    dict(name="local", x=2.0, y=-3.0, z_top=0.0, z_bot=6.0),
    dict(sites="all", n_random=6, seed=1, z_bot=5.0),
]
```

## Reference model

| Setting | Default | Meaning |
|---|---|---|
| `MOD_REF_FILE` | `"templates/referencemodel.dat"` | block file on the same mesh |
| `MOD_REF_BH` | `True` | add it to every panel |
| `MOD_REF_LABEL` | `"reference model"` | legend text |

The default reference line is black and dashed. Without `color` in
`MOD_BH_STYLE["ref"]` or the entry's style it takes the colour of the model
line.

## Figure

| Setting | Default | Meaning |
|---|---|---|
| `MOD_BH_STYLE` | solid model, black dashed reference | `{"model": {...}, "ref": {...}}`, Line2D keywords |
| `MOD_BH_SHARED` | `False` | `True` = all boreholes of a figure on one axes |
| `MOD_BH_XLIM` | `None` | Ohm.m, log axis; `None` = `10**MOD_CLIM`, else auto (an entry's `xlim` wins) |
| `MOD_CLIM` | `[0, 4]` | log10 range used when `MOD_BH_XLIM` is `None` |
| `MOD_ZLIM` | `None` | default `z_bot` = `MOD_ZLIM[1]` |
| `MOD_BH_PANEL_SIZE` | `[16, 20]` | cm |
| `MOD_BH_NCOLS`, `MOD_BH_PER_FIG` | `2`, `10` | layout |
| `MOD_BH_SHARE_DEPTH` | `True` | one depth range per figure |
| `MOD_BH_LEGEND_LOC` | `"best"` | `None` = no legend |
| `MOD_DEPTH_KM` | `True` | depth axis in km |
| `MOD_TICK_FONTSIZE`, `MOD_LABEL_FONTSIZE`, `MOD_TITLE_FONTSIZE` | `16`, `16`, label size + 10 | points; panel titles use tick size - 2, the legend tick size - 4 |
| `MOD_TITLE` | `"auto"` | none, text with `{label}`, `{tag}`, `{iter}`, `{nrms}`, `{alpha}`, `{file}`, `{prefix}`, or a dict per run label |
| `MOD_PANEL_TITLES` | `True` | name and position above each panel |

### One figure per borehole (optional)

| Setting | Default | Meaning |
|---|---|---|
| `MOD_SINGLE_PANELS` | `False` | `False` = panels in a grid; `True` = one figure per borehole |

With `True` the files are `..._boreholes_01_<name>.<fmt>`, `..._02_<name>`, ...
(characters other than letters, digits, `.`, `_`, `-` become `_`);
`MOD_BH_NCOLS` and `MOD_BH_PER_FIG` are not used. With
`MOD_BH_SHARE_DEPTH = True` all figures of a model have the same depth range.
`MOD_BH_PANEL_SIZE` is the size of each figure. The NPZ file is written once
per model as before. `MOD_BH_SHARED = True` (one axes) takes precedence.

## Annotations (`MOD_BH_MARKERS`)

A list of dicts; `[]` (default) / `None` = none. **Depths are km.**

| Key | Default | Meaning |
|---|---|---|
| `depth` | required | arrow tip, km positive down |
| `rho` | left edge | arrow tip, Ohm.m |
| `rho_text` | `3 * rho` | text position, Ohm.m |
| `depth_text` | `depth - 0.3` | text position, km |
| `text` | `""` | label |
| `borehole` | all panels | name or list of names |
| `arrowprops`, others | | passed to `Axes.annotate` |

A marker whose `depth` lies outside the depth range of a panel is not drawn.

## Migration from the earlier version (2026-11-26)

| Old | New |
|---|---|
| `MODEL_FILE`, `MODEL_ITER` | `RUN_PATHS`, `ITERATION` (default now `"best"`) |
| `BOREHOLE_FILE` | `MOD_BH_FILE` (no extension) + `MOD_PLOT_FORMAT` |
| `BATCH_OUT_DIR`, `BATCH_FORMATS`, `BATCH_CATALOG`, `BATCH_TITLE` | `OUT_DIR`, `MOD_PLOT_FORMAT`, `MOD_CATALOG`, `MOD_TITLE` |
| `MESH_FILE`, `MESH_SWAP_XY` | `MOD_MESH`, `MOD_MESH_SWAP_XY` |
| `BOREHOLE_SITES` (m; style keys in the entry) | `MOD_BH` (km; `style`) |
| `BOREHOLE_STYLE` | `MOD_BH_STYLE` (keyed `"model"`, `"ref"`) |
| `BOREHOLE_XLIM`, `BOREHOLE_SHARED`, `BOREHOLE_NPZ` | `MOD_BH_XLIM`, `MOD_BH_SHARED`, `MOD_BH_NPZ` |
| `BOREHOLE_MARKERS` (m) | `MOD_BH_MARKERS` (km) |
| `FIGURE_TITLE`, `PANEL_TITLES` | `MOD_TITLE`, `MOD_PANEL_TITLES` |
| `PLOT_DPI` | `MOD_DPI` |
| `LEGEND_FONTSIZE` | gone: legend and panel titles follow `MOD_TICK_FONTSIZE` |
| `TICK_FONTSIZE`, `LABEL_FONTSIZE`, `FIGURE_TITLE_FONTSIZE` | `MOD_TICK_FONTSIZE`, `MOD_LABEL_FONTSIZE`, `MOD_TITLE_FONTSIZE` |
| `SITE_DAT`, `AIR_RHO`, `OCEAN_RHO`, `UTM_*`, `ORIGIN_METHOD` | same name with `MOD_` in front |

The NPZ keys changed as listed above.

## Dependencies

`femtic.py`, `femtic_viz.py`, `util.py` (Py4MTX, of 2026-11-26 for the first
two), `numpy`, `matplotlib`; `scipy` for the mesh search.

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-05-16 | vrath / Claude Sonnet 4.6 | Borehole step created inside `femtic_mod_plot.py` |
| 2026-06-03 | Claude Sonnet 4.6 | Carried into `femtic_mod_plot_slice.py` after script split from `femtic_mod_plot.py`. `BOREHOLE_XLIM` in Ω·m; `z_top="surface"`; lat/lon legend; per-trace line-style keys; `BOREHOLE_NPZ` |
| 2026-06-04 | vrath / Claude Sonnet 4.6 | **Split** from `femtic_mod_plot_slice.py` into this dedicated script. `PLOT_BOREHOLE` flag removed (script is the flag). `BOREHOLE_IN_SLICE` removed (handled in `femtic_mod_plot_slice.py`). UTM origin preamble retained so script runs independently |
| 2026-06-04 | vrath / Claude Sonnet 4.6 | Added `BOREHOLE_MARKERS` (free arrow + text annotations) and `LEGEND_FONTSIZE`; forwarded to `fviz.plot_borehole_logs` |
| 2026-06-26 | vrath / Claude Sonnet 4.6 | Added `TICK_FONTSIZE` and `LABEL_FONTSIZE` (both default `None` → derived from `LEGEND_FONTSIZE`); forwarded to `fviz.plot_borehole_logs` as `tick_fontsize` / `label_fontsize`. |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_mod_plot_bh_summary.md` output at end of run: writes user-set (UPPERCASE) parameters, script path, and run date/time. |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | The parameter summary `femtic_mod_plot_bh_summary.md` is now written to `WORK_DIR` (the working directory set in the script) instead of next to the script: `utl.write_param_summary(fname, out_dir=WORK_DIR)`. Needs the `util.py` of the same date. AI-generated; review before production use. |
| 2026-11-08 | Claude Sonnet 5.5 (Anthropic) | **Several models in one run:** `MODEL_FILE` may be a list, a glob or run directories (`MODEL_ITER` selects the iteration), giving one figure and NPZ per model plus an optional catalog PDF (`BATCH_*`); new `femtic.resolve_model_files`, `fviz.append_catalog_page`; `fviz.plot_borehole_logs` now returns its figure. One model behaves exactly as before (checked: figure and NPZ identical). AI-generated; review before production use. |
| 2026-11-10 | Claude Sonnet 5.5 (Anthropic) | New `MESH_SWAP_XY` (default `True`), applied before the mesh is read; without it the mesh is transposed relative to the borehole positions and the logs are taken at the wrong places. AI-generated; review before production use. |
| 2026-11-11 | Claude Sonnet 5.5 (Anthropic) | `FIGURE_TITLE` takes the forms of all plot scripts (`"auto"`, none, text with placeholders, dict per model label); new `FIGURE_TITLE_FONTSIZE`, `PANEL_TITLES`; `BATCH_TITLE` also knows `{file}`. Defaults: figures unchanged. Needs the `femtic_viz.py` of the same date. |
| 2026-11-11b | Claude Sonnet 5.5 (Anthropic) | New `CNV_FILE` (one name or a list of candidate names; default `["femtic.cnv", "FEMTIC-DABIC.cnv"]`), passed to `fem.resolve_model_files()` as `cnv_name`. Needs the `femtic.py` of the same date. AI-generated; review before production use. |
| 2026-11-26 | Claude Sonnet 5.5 (Anthropic) | Rewritten with the settings of `femtic_ens_plot_bh.py` (without the ensemble statistics) and several runs as in `femtic_mod_misfit.py` (`RUN_PATHS`, command line, `-w`, `ITERATION`). `MOD_BH` / `MOD_BH_*` / `MOD_*` names, depths in km, `sites` templates, `MOD_REF_*`, `MOD_CATALOG`; per-run mesh, site list and reference. Profiles by `femtic.sample_model_boreholes()`, figures by `femtic_viz.draw_borehole_figures()`; NPZ arrays `depth_<name>`, `model_<name>`, `ref_<name>` and a JSON header. This readme was rewritten. Needs `femtic.py` and `femtic_viz.py` of the same date. Tested on synthetic files only. AI-generated; review before production use. |
| 2026-11-26b | Claude Sonnet 5.5 (Anthropic) | Optional single-panel output (`MOD_SINGLE_PANELS`, default `False`): one figure per borehole. Needs `femtic_viz.py` of the same date. Tested on synthetic files only. AI-generated; review before production use. |
| 2026-11-26c | Claude Sonnet 5.5 (Anthropic) | Defaults replaced by those of `femtic_ens_plot_bh.py` (paths in `templates/`, `RUN_PATHS = ["ann*/"]`, one panel per site to 4 km, black dashed reference, `MOD_CLIM` [0, 4], panel and font sizes). No change of the code. Tested on synthetic files only. AI-generated; review before production use. |
| 2026-11-26d | Claude Sonnet 5.5 (Anthropic) | All user settings are km, UTM coordinates too: `MOD_UTM_ORIGIN_E / _N` and `(value, "utm")` borehole positions (before: metres). A value that looks like metres stops with an error (`fem.KmUnitError`); smaller values given in metres cannot be recognised and are read as km. Needs `femtic.py` of the same date. Tested on synthetic data only: same output as before for the same positions. AI-generated; review before production use. |
