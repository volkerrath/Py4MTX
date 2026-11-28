# femtic_mod_plot_slice.py - README

2-D slice panels (map, N-S, E-W, oblique plane, two-point profile / fault) of
single FEMTIC resistivity models.

The settings are those of `femtic_ens_plot_slices.py` (same `MOD_*` names, same
`MOD_SLICES` entries, lengths in km) without everything that belongs to
ensemble statistics. Several runs can be given, as in `femtic_mod_misfit.py`:
every entry is a model of its own and gets its own figure(s): all slices in one
figure, or optionally one figure per slice (`MOD_SINGLE_PANELS`).

Sister scripts:
[`femtic_mod_plot_bh.py`](femtic_mod_plot_bh_readme.md) (borehole logs),
[`femtic_mod_plot_3d.py`](femtic_mod_plot_3d_readme.md) (3-D view, VTK).
All plotting is done by `femtic_viz.py`.

The defaults are those of `femtic_ens_plot_slices.py` (2026-11-26).

AI-generated documentation (Claude Sonnet 5.5, Anthropic, 2026-11-26); review
before use.

---

## Usage

```
python femtic_mod_plot_slice.py                       # RUN_PATHS
python femtic_mod_plot_slice.py kra_rto_0 kra_rto_1   # two run directories
python femtic_mod_plot_slice.py "kra*/"               # quote globs
python femtic_mod_plot_slice.py -w /data/ann ann_rto_orig-err_0/resistivity_block_iter10.dat
```

Command-line entries replace `RUN_PATHS`; `-w` / `--work-dir` replaces
`WORK_DIR`. In Spyder just set the user section and run.

## Runs

| Setting | Default | Meaning |
|---|---|---|
| `WORK_DIR` | `.../krafla2026/krafla_fine/krafla_ens_fine/` | base of relative paths; `None` / `""` = current directory |
| `RUN_PATHS` | `["kra*/"]` | run directories, block files, `results_iterN.h5` files (= the block file of that iteration; the HDF5 file is not read) or globs |
| `ITERATION` | `"best"` | for run directories: `"best"` (lowest nRMS), `"last"` (last row of the convergence file), `N` (`0` = starting model), or a list such as `[0, "best"]` |
| `CNV_FILE` | `["femtic.cnv", "FEMTIC-DABIC.cnv"]` | convergence file; first name that exists |

`MOD_MESH`, `MOD_SITE_DAT`, `MOD_OBSERVE_FILE`, `MOD_REF_FILE` and
`MOD_ALPHA_FILE` are looked up per run: a relative name in the run directory
first, then in `WORK_DIR`; an absolute path serves every run. Mesh centre and
sites are therefore determined run by run.

An entry without a model is reported and skipped; a model that fails is
reported and skipped, the others are still plotted. Return code: `0` all models
found were plotted, `1` one failed or none was found, `2` `WORK_DIR` missing.

## Output

```
<dir>/[<OUT_PREFIX>_]<label>_<tag>_slices.<fmt>
```

`label` = run directory name, `tag` = `iterN`. With `MOD_SINGLE_PANELS = True`
the names are `..._slices_01_<kind>.<fmt>`, `..._02_<kind>`, ... (number of the
displayed slice and its kind).

| Setting | Default | Meaning |
|---|---|---|
| `MOD_PLOT_FORMAT` | `["pdf"]` | one or more extensions |
| `OUT_DIR` | `None` | one directory for all runs; `None` = per run |
| `OUT_PREFIX` | `None` | file-name prefix; with `OUT_DIR = None` also a sub-directory of the run |
| `OUT_PREFIX_DIR` | `True` | `False` = no sub-directory |
| `MOD_QC_FILE` | `None` | explicit path without extension, used when exactly one model is plotted |
| `MOD_CATALOG` | `None` | multi-page PDF, one page per figure: `"auto"` (`catalog_slices.pdf` in `OUT_DIR` or the common parent of the runs) or a path; needs `"pdf"` among the formats |
| `MOD_DPI` | `400` | resolution |
| `MOD_SHOW_IN_SPYDER` | `True` | show figures inline under Spyder |

The parameter summary `femtic_mod_plot_slice_summary.md` is written to
`WORK_DIR`.

## Mesh, ocean, coordinates

| Setting | Default | Meaning |
|---|---|---|
| `MOD_MESH` | `"templates/mesh.dat"` | required |
| `MOD_MESH_SWAP_XY` | `True` | `mesh.dat` holds x = northing, y = easting; `True` gives x = easting. `False` draws such a mesh transposed; `None` = femtic's default |
| `MOD_OCEAN`, `MOD_AIR_RHO`, `MOD_OCEAN_RHO` | `None`, `1e9`, `0.25` | `None` = infer ocean from the block file |
| `MOD_UTM_ORIGIN_LAT/LON/E/N` | `None` | literal mesh-centre origin (degrees; E / N in UTM **km**) |
| `MOD_UTM_ZONE_OVERRIDE` | `None` | force the UTM zone |
| `MOD_ORIGIN_METHOD` | `"box"` | `"box"` / `"average"` of the sites in `MOD_SITE_DAT`; `None` = literals |
| `MOD_DISPLAY_COORDS` | `"utm"` | `"model"` / `"utm"` / `"latlon"` |
| `MOD_TICK_DECIMALS` | `0` | decimals of the tick labels |

Positions tagged `"utm"` or `"latlon"` need the origin; without it the model is
reported as failed.

## Slices (`MOD_SLICES`)

One dict per panel. Positions are model-local **km**, or
`(value, "utm")` (UTM km) / `(value, "latlon")` (degrees). Depths are km,
positive down.

| `kind` | Keys |
|---|---|
| `"map"` | `z0` |
| `"ns"` | `x0` |
| `"ew"` | `y0` |
| `"plane"` | `point` `[x, y(, z)]`, `strike` (azimuth, clockwise from north), `dip` (towards strike + 90; 90 = vertical) |
| `"profile"` | `p1`, `p2`, `z_top`, `z_bot`, `dip`, `limit_km`, `alpha_outside`, `fade_km` |

Common keys: `title`, `invert_x`, `overlay` (also draw this slice as a line on
the other panels), `panel=False` (overlay line only), `xlim` / `ylim` /
`zlim`.

**All lengths are km**, UTM coordinates too (as in the ensemble script):

- `kind="profile"`: `p1` / `p2` without tag are model-local km, tagged
  `"utm"` UTM km; `z_top` / `z_bot` are km.
- Per-panel `xlim` / `ylim` / `zlim` are km. For a `"plane"`, `xlim` / `ylim`
  are relative to `point` (along strike / down dip).

A value that looks like metres stops with an error (`fem.KmUnitError`); smaller values given in metres cannot be recognised and are read as km. Coordinates read from files (`site.dat`) stay in metres.

Default: maps at 0, 1, 2, 3, 4 and 5 km. Further forms:

```python
MOD_SLICES = [
    dict(kind="map", z0=2.0),                     # km
    dict(kind="ns",  x0=0.0),                     # km, model-local
    dict(kind="ew",  y0=(45.9383, "latlon")),
    dict(kind="plane", point=[0.0, 0.0], strike=45.0, dip=90.0),
    dict(kind="profile",
         p1=([6.030869, 45.962166], "latlon"),
         p2=([6.115936, 45.910655], "latlon"),
         dip=110.0, z_top=0.0, z_bot=20.0,        # km
         overlay=dict(color="k", linewidth=1.5), panel=False),
]
```

In a figure with several rows that mixes maps and sections, axis labels and
panel titles can overlap (layout of `fviz.plot_model_slices()`, which the
ensemble script uses too). Use one row (`MOD_NROWS = 1`, `MOD_NCOLS` = number
of slices) or `MOD_SINGLE_PANELS = True`.

Fault planes in detail: `femtic_viz_readme.md`, "Fault planes: a panel and an
overlay".

## Sites and markers

| Setting | Default | Meaning |
|---|---|---|
| `MOD_SITE_DAT` | `"templates/site.dat"` | `name,lat,lon,elev,sitenum,easting,northing`; `None` = off |
| `MOD_OBSERVE_FILE`, `MOD_SITE_NUMBER` | `"templates/observe.dat"`, `None` | fallback site source |
| `MOD_SITE_NAMES` | `None` | names to plot; `None` = all |
| `MOD_PLOT_SITES_MAPS`, `MOD_PLOT_SITES_SLICES` | `True`, `False` | draw site markers |
| `MOD_PROJECTION_DIST` | `1.0` | km from a section plane; `None` = all sites |
| `MOD_SITE_MARKER`, `MOD_SITE_MARKER_SLICES` | black triangles, `None` | marker style |
| `MOD_MAP_MARKERS` | model centre (black cross) | point markers: `latlon` `[lat, lon]`, `depth`, `depth_err`, `max_dist` (km), `kinds`, `name`, Line2D keywords |
| `MOD_SHOW_MODEL_CENTRE` | `True` | mark the origin on maps (`"utm"` / `"latlon"` display) |

## Polylines (`MOD_POLYLINES`)

`{name: entry}`; `{}` = none. `points` are model-local km (z = depth) or
`(pts, "utm")` (E, N in UTM km, z km) / `(pts, "latlon")` (lon, lat, z km); optional
`style`, `label`, `kinds`, `max_dist` (km), `z_range` (km), `panels`. See
`femtic_viz_readme.md` for the drawing rules.

## Reference model

| Setting | Default | Meaning |
|---|---|---|
| `MOD_REF_FILE` | `"templates/referencemodel.dat"` | block file on the same mesh |
| `MOD_REF_SLICES` | `False` | draw it as iso-resistivity lines |
| `MOD_REF_LEVELS` | `None` | log10 levels; `None` = integer decades inside `MOD_CLIM` |
| `MOD_REF_STYLE`, `MOD_REF_LABEL` | black lines | style and legend text |

## Limits, colours, blanking

| Setting | Default | Meaning |
|---|---|---|
| `MOD_XLIM`, `MOD_YLIM`, `MOD_ZLIM` | `None` | km; `None` = auto |
| `MOD_ROI_AUTO`, `MOD_ROI_PAD_XY`, `MOD_ROI_ZLIM` | `True`, `2.0`, `[-1, 6]` | x / y limits from the sites of each run, z limits as given |
| `MOD_CMAP`, `MOD_CLIM` | `"jet_r"`, `[0, 4]` | log10(Ohm.m) |
| `MOD_OCEAN_COLOR`, `MOD_AIR_COLOR`, `MOD_AIR_BGCOLOR` | grey, whitesmoke, `None` | |
| `MOD_ALPHA_FILE` | `None` | block file whose region values are log10 weights |
| `MOD_ALPHA_MODE`, `MOD_ALPHA_BLANK_THRESH` | `"fade"`, `0.0` | `"fade"` or `"blank"` below the threshold |
| `MOD_NRMS_ANNOTATE`, `MOD_NRMS_PANEL` | `True`, `None` | nRMS / alpha box, from the run's own convergence file; none if it has no row for the iteration |

## Layout, titles, colour bar

### One figure per slice (optional)

| Setting | Default | Meaning |
|---|---|---|
| `MOD_SINGLE_PANELS` | `False` | `False` = one figure per model with all slices; `True` = one figure per model and slice |

With `True`:

- `panel=False` entries get no figure; `overlay` lines (fault traces) appear on
  the other slices as in the combined figure.
- All figures of a model share the colour scale (automatic limits are taken from
  the whole model).
- `MOD_NROWS` / `MOD_NCOLS` are not used; `MOD_FIGSIZE` and the title apply to
  every figure.
- `MOD_NRMS_PANEL = None` puts the nRMS box on every figure; a number only on
  that slice.
- The mesh is read once per figure and format, so large meshes take longer.

| Setting | Default |
|---|---|
| `MOD_EQUAL_ASPECT`, `MOD_VERT_EXAG` | `True`, `None` |
| `MOD_DEPTH_KM`, `MOD_HORIZ_KM` | `True`, `True` |
| `MOD_NROWS`, `MOD_NCOLS` | `3`, `3` |
| `MOD_PANEL_HEIGHT`, `MOD_PANEL_WIDTH`, `MOD_FIGSIZE` | `20.0`, `None`, `None` (cm) |
| `MOD_TICK_FONTSIZE`, `MOD_LABEL_FONTSIZE`, `MOD_TITLE_FONTSIZE` | `16`, `16`, label size + 10 |
| `MOD_TITLE` | `"auto"` (block file; run, iteration, nRMS) |
| `MOD_PANEL_TITLES` | `True` |
| `MOD_CBAR_LOCATION`, `MOD_CBAR_SHRINK`, `MOD_CBAR_THICKNESS` | `"bottom"`, `0.7`, `None` |

`MOD_TITLE`: `"auto"`, `None` / `False` / `""` (none), text with `{label}`
(= `{run}`, `{key}`, `{best}`), `{tag}`, `{iter}`, `{nrms}`, `{alpha}`,
`{file}`, `{prefix}`, or a dict per run label with `"default"`.

## Migration from the earlier version (2026-11-26)

| Old | New |
|---|---|
| `MODEL_FILE`, `MODEL_ITER` | `RUN_PATHS`, `ITERATION` (default now `"best"`) |
| `PLOT_FILE` | `MOD_QC_FILE` (no extension) + `MOD_PLOT_FORMAT` |
| `BATCH_OUT_DIR`, `BATCH_FORMATS`, `BATCH_CATALOG`, `BATCH_TITLE` | `OUT_DIR`, `MOD_PLOT_FORMAT`, `MOD_CATALOG`, `MOD_TITLE` |
| `FIGURE_TITLE` (default none) | `MOD_TITLE` (default `"auto"`) |
| `MESH_FILE` | `MOD_MESH` |
| `PLOT_SLICES`, `PLOT_POLYLINES` (m) | `MOD_SLICES`, `MOD_POLYLINES` (km) |
| `PLOT_XLIM/YLIM/ZLIM` (m) | `MOD_XLIM/YLIM/ZLIM` (km) |
| `PROJECTION_DIST` (m) | `MOD_PROJECTION_DIST` (km) |
| `MAP_MARKERS` (`depth`, `depth_err`, `max_dist` in m) | `MOD_MAP_MARKERS` (km) |
| `CNV_PANEL` | `MOD_NRMS_PANEL` |
| `PLOT_<name>` | `MOD_<name>` |
| other plot settings (`SITE_*`, `ALPHA_*`, `CBAR_*`, `UTM_*`, ...) | same name with `MOD_` in front |

File names are now `<run>_iterN_slices.<fmt>` unless `MOD_QC_FILE` is set.

## Dependencies

`femtic.py`, `femtic_viz.py`, `util.py` (Py4MTX, of 2026-11-26 for the first
two), `numpy`, `matplotlib`.

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Created as part of `femtic_mod_plot.py` |
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Lat/lon and UTM slice positions; auto UTM zone |
| 2026-05-13 | vrath / Claude Sonnet 4.6 | Config block harmonised with `femtic_mod_edit.py` |
| 2026-05-16 | vrath / Claude Sonnet 4.6 | Borehole step added; `BOREHOLE_*` config block |
| 2026-05-23 | vrath / Claude Sonnet 4.6 | Geographic helpers moved to `util.py`; model-local helpers to `femtic.py` |
| 2026-05-23 | vrath / Claude Sonnet 4.6 | `SITE_DAT` in `mt_make_sitelist.py` CSV format |
| 2026-05-24 | vrath / Claude Sonnet 4.6 | Ensemble step removed (→ `snippets.py`); borehole renumbered step 6 |
| 2026-05-25 | vrath / Claude Sonnet 4.6 | `PLOT_SITES_MAPS`/`_SLICES`, `PROJECTION_DIST`, `DEPTH_KM`, `HORIZ_KM`, `PLOT_NROWS`/`NCOLS`, panel sizing in cm |
| 2026-05-26 | Claude Sonnet 4.6 | `plot_model_slices`/`plot_borehole_logs` moved into `femtic_viz.py`; `ALPHA_FILE`/`ALPHA_MODE`/`ALPHA_BLANK_THRESH` added |
| 2026-05-27 | vrath / Claude Sonnet 4.6 | `PLOT_XLIM/YLIM/ZLIM` passed to slice panels |
| 2026-05-31 | vrath / Claude Sonnet 4.6 | `invert_x` per-panel key; origin estimation before UTM zone derivation |
| 2026-06-03 | Claude Sonnet 4.6 | **Split** from `femtic_mod_plot.py` → `femtic_mod_plot_slice.py` + `femtic_mod_plot_3d.py`. `BOREHOLE_IN_SLICE`: embedded borehole columns via `plot_model_slices(borehole_sites=...)`; `BOREHOLE_XLIM` now in Ω·m; `z_top="surface"` supported; lat/lon legend; per-trace line-style keys; `BOREHOLE_NPZ` added |
| 2026-06-04 | vrath / Claude Sonnet 4.6 | **Split** from `femtic_mod_plot_slice.py`: standalone borehole step (step 6) and `PLOT_BOREHOLE` flag moved to new `femtic_mod_plot_bh.py`. `BOREHOLE_IN_SLICE` retained here for embedded columns only |
| 2026-06-19 | Claude Sonnet 4.6 | `kind="profile"` added to `PLOT_SLICES`: vertical fence section defined by two endpoint positions (`p1`, `p2`) each accepting model-local / UTM / latlon CRS tags; `strike` derived from p1→p2 azimuth; `dip` fixed at 90°. New helper `resolve_pos_two_point_profile()` in `femtic.py`; `resolve_slice_positions()` extended. |
| 2026-06-26 | vrath / Claude Sonnet 4.6 | Added `TICK_FONTSIZE` (default `7`) and `LABEL_FONTSIZE` (default `8`) config constants; forwarded to `plot_model_slices` as `tick_fontsize` / `label_fontsize`. |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_mod_plot_slice_summary.md` output at end of run: writes user-set (UPPERCASE) parameters, script path, and run date/time. |
| 2026-09-06 | Claude Sonnet 5 (Anthropic) | Added `SHOW_MODEL_CENTRE` (default `True`): marks the model origin on `"map"` panels whenever `DISPLAY_COORDS` is `"utm"`/`"latlon"`, via `fviz.plot_model_slices`'s new `show_model_centre` parameter. Override style with a `MAP_MARKERS` entry carrying `"is_model_centre": True`. |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | The parameter summary `femtic_mod_plot_slice_summary.md` is now written to `WORK_DIR` (the working directory set in the script) instead of next to the script: `utl.write_param_summary(fname, out_dir=WORK_DIR)`. Needs the `util.py` of the same date. AI-generated; review before production use. |
| 2026-11-08 | Claude Sonnet 5.5 (Anthropic) | **Several models in one run:** `MODEL_FILE` may be a list, a glob or run directories (`MODEL_ITER` selects the iteration), giving one figure per model plus an optional catalog PDF (`BATCH_*`); new `femtic.resolve_model_files`, `fviz.plot_model_slices_robust`, `fviz.append_catalog_page`. One model behaves exactly as before (checked: the figure is pixel-identical). Takes over the per-member figures of the former `femtic_ens_plot.py`. AI-generated; review before production use. |
| 2026-11-09 | Claude Sonnet 5.5 (Anthropic) | Dipping planes (`kind="plane"`, `dip != 90`) fixed in `femtic_viz.plot_model_slices`: `strike` is the azimuth for every dip, axes are along-strike / down-dip distance from `point`, default extent = plane cut with the `PLOT_XLIM` x `PLOT_YLIM` x `PLOT_ZLIM` box, per-slice `xlim` / `ylim` = along-strike / down-dip range (m, relative to `point`). The CRS-tagged `point` (`"model"` / `"utm"` / `"latlon"`) was already supported by this script. Needs the `femtic_viz.py` of the same date. |
| 2026-11-10 | Claude Sonnet 5.5 (Anthropic) | New `MESH_SWAP_XY` (default `True`), applied before the mesh is read. Slice maps were drawn transposed (x = easting and y = northing exchanged; a NNE structure appeared as ENE). Cause: since 2026-10-11 `femtic.read_femtic_mesh` does not swap the node columns by default, but `mesh.dat` holds x = northing, y = easting. `None` keeps femtic's default, `False` forces no swap. Only the orientation changes. Checked on a synthetic mesh with a N30E block, not on real data. AI-generated; review before production use. |
| 2026-11-11 | Claude Sonnet 5.5 (Anthropic) | New `PLOT_POLYLINES = {name: entry}` (metres; `{}` = none): lines drawn on the slice panels. Maps: plan view, optional `z_range`; ns / ew / plane / profile sections: the parts within `max_dist` (default `PROJECTION_DIST`) of the section plane, projected into it. Points model-local m or CRS-tagged (`"utm"` metres, `"latlon"` degrees, longitude first); `style`, `label`, `kinds`. Passed to `fviz.plot_model_slices(polylines=...)` in the shared options (every figure of a batch). The intersection line of two slices is available as `fviz.slice_intersection_polyline` / `slice_intersection_polylines` (example in the section above); it is not drawn automatically. With `PLOT_POLYLINES = {}` the figures are unchanged. Needs the `femtic_viz.py` of the same date. Tested on a synthetic mesh only. AI-generated; review before production use. |
| 2026-11-11b | Claude Sonnet 5.5 (Anthropic) | Titles and colour bars: `FIGURE_TITLE` takes the forms of all plot scripts (`"auto"`, none, text with `{label}`, `{tag}`, `{iter}`, `{nrms}`, `{alpha}`, `{file}`, dict per model label); `BATCH_TITLE` also knows `{alpha}`, `{file}`; new `PANEL_TITLES`, `CBAR_LOCATION` (`"panel"` default), `CBAR_SHRINK`, `CBAR_THICKNESS`. Defaults: figures unchanged. Needs the `femtic_viz.py` of the same date. |
| 2026-11-11c | Claude Sonnet 5.5 (Anthropic) | Convergence file from a list of names: `CNV_FILE` is now a name or a list of candidate names (default `["femtic.cnv", "FEMTIC-DABIC.cnv"]`), no longer `WORK_DIR + "femtic.cnv"`; it is passed to `fem.resolve_model_files()` and, for the annotation of a single model, searched first in the directory of the model file, then in `WORK_DIR`. The last row is read by column name (`fem.read_cnv_last()`); the setting `DISTORTION` and the guess from the number of columns are removed. Needs the `femtic.py` of the same date. AI-generated; review before production use. |
| 2026-11-12 | Claude Sonnet 5.5 (Anthropic) | Fault from two surface points. `kind="profile"` takes `dip` (default 90 = as before; otherwise the dipping plane through p1 -> p2, dipping to the right of p1 -> p2) and any slice may carry `overlay` (`True` or `Line2D` keywords, + `label`): that plane is also drawn on every other panel (trace on maps, dipping line on sections). New polyline-entry key `panels`. Documentation block and commented example in the script. Needs `femtic.py` and `femtic_viz.py` of the same date. Without `dip` / `overlay` the figures are unchanged. Tested on a synthetic mesh only. AI-generated; review before production use. New slice-spec key `panel=False`: the entry is no panel, it only draws its `overlay` line on all the other panels (fault trace without a fault panel). |
| 2026-11-26 | Claude Sonnet 5.5 (Anthropic) | Rewritten with the settings of `femtic_ens_plot_slices.py` (without the ensemble statistics) and several runs as in `femtic_mod_misfit.py` (`RUN_PATHS`, command line, `-w`, `ITERATION`). `MOD_*` names, lengths in km, new `OUT_DIR` / `OUT_PREFIX`, `MOD_QC_FILE`, `MOD_CATALOG`, `MOD_REF_*`, `MOD_ROI_*`; per-run mesh, site list, reference and alpha file; titles on by default. Fixes the broken `WORK_DIR` line (`NameError`). This readme was rewritten; its sections on embedded borehole columns described settings that the script no longer had. Needs `femtic.py` and `femtic_viz.py` of the same date. Tested on synthetic files only. AI-generated; review before production use. |
| 2026-11-26b | Claude Sonnet 5.5 (Anthropic) | Optional single-panel output (`MOD_SINGLE_PANELS`, default `False`): one figure per displayed slice. `RUN_PATHS` defaults to `["*_L2"]` (several runs) instead of one block file. Needs `femtic_viz.py` of the same date. Tested on synthetic files only. AI-generated; review before production use. |
| 2026-11-26c | Claude Sonnet 5.5 (Anthropic) | Defaults replaced by those of `femtic_ens_plot_slices.py` (paths in `templates/`, `RUN_PATHS = ["kra*/"]`, maps at 0-5 km, UTM display, sites on maps, region of interest from the sites, `jet_r` [0, 4], 3 x 3 layout, bottom colour bar). No change of the code. Tested on synthetic files only. AI-generated; review before production use. |
| 2026-11-26d | Claude Sonnet 5.5 (Anthropic) | All user settings are km, UTM coordinates too: `MOD_UTM_ORIGIN_E / _N`, `"utm"` positions, profile `p1` / `p2` / `z_top` / `z_bot`, per-panel limits and polyline points (before: metres). A value that looks like metres stops with an error (`fem.KmUnitError`); smaller values given in metres cannot be recognised and are read as km. Needs `femtic.py` of the same date. Tested on synthetic data only: same output as before for the same positions. AI-generated; review before production use. |
