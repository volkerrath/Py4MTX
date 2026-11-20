# femtic_rto_prep.py

Prepare and launch a Randomize-Then-Optimize (RTO) ensemble for FEMTIC
MT inversion.

---

## Purpose

`femtic_rto_prep.py` implements the outer loop of the RTO algorithm:

```
for i = 1 : N_SAMPLES do
    Draw perturbed data:  d̃ᵢ ~ N(d, Cᵈ)
    Draw perturbed model: m̃ᵢ = mᵣₑf + xᵢ,  xᵢ ~ N(0, (α RᵀR)⁻¹)   (roughness-matrix prior)
    Write member directory i with perturbed observe.dat / starting model
end
```

Each member directory contains a complete FEMTIC run setup (symlinks to
shared files, copied perturbed inputs) ready for `run_femtic_dias.sh`.

Optional diagnostic plots:

- **Data ensemble**: original vs. perturbed `observe.dat` for selected members.
- **Model ensemble**: reference vs. perturbed starting model slices.
- **QC slices**: `fviz.plot_model_slices` per member after perturbation.
- **Ensemble slices**: `fviz.plot_ensemble_slices` joint figure.

---

## Key configuration variables

### Paths
| Variable | Description |
|---|---|
| `ENSEMBLE_DIR` | Root directory for all member subdirectories |
| `ENSEMBLE_NAME` | Prefix for member directory names |
| `TEMPLATES` | Directory containing template files to copy/link |
| `COPY_LIST` | Files copied into each member directory |
| `LINK_LIST` | Files symlinked into each member directory |
| `RELATIVE_LINKS` | `True` → portable relative symlinks (default) |

### Ensemble size
| Variable | Description |
|---|---|
| `N_SAMPLES` | Number of ensemble members |
| `FROM_TO` | `[start, end]` to restart a subset; `None` = all |
| `RANDOM_SEED` | `None` (default) = fresh OS entropy, a different ensemble every run. An integer makes the run fully reproducible: the shared `rng = np.random.default_rng(RANDOM_SEED)` drives the roughness-matrix model draw (`generate_rto_model_ensemble`, already `rng`-aware), the data perturbation (`generate_data_ensemble`, now also `rng`-aware — see below), and the `VIZ_SAMPLES` diagnostic-plot draw. |

### Reproducibility

Before this update, `PERTURB_DAT` draws were **never** reproducible even
when `RANDOM_SEED`-style seeding was used for the model perturbation:
`ens.generate_data_ensemble` didn't accept an `rng` at all, so every call
to the underlying `femtic.modify_data` silently fell back to its own
fresh, unseeded generator. `generate_data_ensemble` now accepts `rng` and
forwards it to `modify_data`, and this script passes the same shared
`rng` used by `generate_rto_model_ensemble`, so setting `RANDOM_SEED`
makes the *entire* ensemble (data + model perturbation, plus the
`VIZ_SAMPLES` draw) reproducible together.


### Data perturbation (`PERTURB_DAT`)
`PERTURB_DAT = False`: no data ensemble is generated (`data_ensemble = None`); the `DAT_*` variables are then only needed for the data plot (which, with `PLOT_DATA = True`, shows the original against identical member data).

| Variable | Description |
|---|---|
| `DAT_METHOD` | `"add"` (additive noise) |
| `DAT_PDF` | `["normal", mean, std]` |
| `RESET_ERRORS` | Replace errors before perturbation |
| `ERRORS` | Error floors per data type |

### Model perturbation (`PERTURB_MOD`)
`PERTURB_MOD = False`: no model ensemble is generated and no `R` is loaded (`model_ensemble = None`); the `MOD_PERTURB_ITER0` / `MOD_ITER0_HOMOGENEOUS` options are switched off, and `MOD_REF` / `MOD_REF_BASE` are `<TEMPLATES>/referencemodel.dat` for the plots.

| Variable | Description |
|---|---|
| `MOD_REF` | Reference model file |
| `MOD_ALPHA` | **Required** (or `MOD_ALPHA_CNV`). Regularisation weight alpha of the member inversions; the model perturbation is `x ~ N(0, (alpha R^T R)^-1)`, std ~ `alpha^(-1/2)`. One fixed value for all members. The script exits before creating any directory if alpha is missing or `<= 0`. |
| `MOD_ALPHA_CNV` | Path of a `femtic.cnv` (or its directory) whose last-row `Alpha` is used when `MOD_ALPHA = None`. The script prints `Misfit + Alpha*Roughness` against `ObjFunc` and warns if they differ by more than 0.1 %: verify FEMTIC's convention before trusting the value. |
| `MOD_ALGO` | `"auto"` (default; sparse LU up to 60000 unknowns, Jacobi-CG above), `"splu"`, `"cg"`. |
| `MOD_LAM` | Fixed diagonal shift for `R^T R + lam I`; `0` = `MOD_LAM_ALPHA * median(diag(R^T R))`. (Not the regularisation weight.) |
| `MOD_LAM_ALPHA` | Scale of the automatic shift (default `1e-4`). |
| `MOD_RTOL` | CG relative tolerance (default `1e-8`; keep tight). |
| `MOD_START_FROM_PERTURBED` | `False` (default): members share the unperturbed `resistivity_block_iter0.dat`. `True`: each member also gets its perturbed model as starting model (real file replaces the symlink; the template is not modified). |
| `MOD_PERTURB_ITER0` | `False` (default). `True`: also perturb the starting model `MOD_ITER0_FILE` of every member with its **own** prior draw `x0 ~ N(0, (MOD_ALPHA R^T R)^-1)`: same procedure, alpha, `lam` and solver as the draw for `referencemodel.dat`, but **statistically independent** of it (both from one factorisation). Added to the iter0 template model; the member's symlink becomes a real file (template untouched), the unperturbed model is kept as `<name>_orig.dat`. The reference draws are unchanged by this option for a given `RANDOM_SEED`. Exclusive with `MOD_START_FROM_PERTURBED = True` (which copies the SAME draw) and with `MOD_ITER0_HOMOGENEOUS`; the script exits before touching any directory if both are set, if `MOD_ITER0_FILE` equals the reference file, is in neither `COPY_LIST` nor `LINK_LIST`, or is missing in `TEMPLATES`. The QC / model plots still show `MOD_REF_BASE`. |
| `MOD_ITER0_FILE` | `"resistivity_block_iter0.dat"`. File changed by `MOD_PERTURB_ITER0` / `MOD_ITER0_HOMOGENEOUS`. |
| `MOD_ITER0_HOMOGENEOUS` | `False` (default). `True`: fill `MOD_ITER0_FILE` of every member with **one homogeneous constant** (all free regions; air / ocean / fixed regions kept), drawn independently for each member, uniformly in log10(rho) in `MOD_ITER0_RANGE`. File handling as for `MOD_PERTURB_ITER0` (real file replaces the symlink, template untouched, backup `<name>_orig.dat`); the draws for `referencemodel.dat` do not depend on it for a given `RANDOM_SEED`. Exclusive with `MOD_PERTURB_ITER0` and `MOD_START_FROM_PERTURBED = True`; the script exits before touching any directory if both are set, if `MOD_ITER0_RANGE` is not two finite numbers with `lo <= hi`, or for the file checks listed under `MOD_PERTURB_ITER0`. |
| `MOD_ITER0_RANGE` | `[1.0, 3.0]`. `[lo, hi]` in log10(Ohm.m) for `MOD_ITER0_HOMOGENEOUS` (10 ... 1000 Ohm.m); `lo == hi` gives the same constant for all members. Choose it yourself: it sets how far from the data the inversions start. |
| `MOD_REMOVE_MEAN` | `False` (default). `True`: subtract each draw's global mean (offset set by the arbitrary shift `lam`, invisible to `R`). **Recommended together with `MOD_PERTURB_ITER0`**: the offset does not matter for the reference model but shifts the whole starting model. |

Removed (2026-10-21): `MOD_MU`, `MOD_PDF` (never passed on; implied alpha = 1),
`MOD_N_EIG`, `MOD_N_OVERSAMPLING`, `MOD_N_POWER_ITER`, `MOD_SIGMA2_RESIDUAL`,
`MOD_LAM_MODE`, `MOD_SOLVER`, `MOD_PRECOND`; `MOD_ALGO = "low rank"` is gone.

### Visualization
| Variable | Description |
|---|---|
| `PLOT_DATA` | Enable data ensemble plot |
| `PLOT_MODEL` | Enable model ensemble plot |
| `DAT_PHASE_WRAP_MODE` / `DAT_PHASE_WRAP_STYLE` | Data plots, phase panels: a phase curve that jumps by more than 180 deg between two periods (it leaves the +-180 deg window) is interrupted and the two points joined by a separate line. Mode `"direct"` (default, straight line), `"short"` (through the window edge), `"off"` (as before); style = dict of Matplotlib line properties over the dotted default (`None` = dotted, `False` = gap only). Passed to `fviz.plot_data_ensemble`. |
| `PLOT_SLICES_QC` | Enable per-member QC slice plot |
| `PLOT_SLICES_ENS` | Enable joint ensemble slice figure |
| `VIZ_N_SAMPLES` | Number of random members for the plots (default `N_SAMPLES`; capped at the number of members). Defined whatever the plot switches are; the draw of the members consumes the random generator in all cases, so the perturbed files do not depend on the plot switches. All five switches may be `False`: nothing is plotted and no plot variable is needed. |
| `MOD_SLICES` | Slice dicts for QC/model plots (model-local **km**, `kind` key; converted to metres by `_slices_km_to_m`). As of 2026-06-07 the QC plot uses the full shared `MOD_*` plotting config block (`MOD_CMAP`, `MOD_CLIM`, `MOD_XLIM`/`YLIM`/`ZLIM`, `MOD_OCEAN_COLOR`, `MOD_DPI`, site overlay, UTM origin, etc. — same as `femtic_gst_prep.py`/`femtic_ens_post.py`), not the older `QC_SLICES`/`QC_CMAP`/etc. variables. |
| `MOD_TICK_FONTSIZE` / `MOD_LABEL_FONTSIZE` | Axis tick/label font sizes for the QC/model slice plot. Defaults `7`/`8`, matching `fviz.plot_model_slices`' own defaults. |
| `MOD_XLIM` / `MOD_YLIM` / `MOD_ZLIM` | Literal plot extent, model-local **km**; `None` = auto. Overridden by `MOD_ROI_AUTO` (below) when site positions are available. |
| `MOD_ROI_AUTO` | Default `True`. With site positions available (`MOD_SITE_DAT` / `MOD_SITE_NUMBER`, subject to `MOD_PLOT_SITES_MAPS/SLICES`), `MOD_XLIM`/`MOD_YLIM` are derived from the site bounding box + `MOD_ROI_PAD_XY` and `MOD_ZLIM` is set from `MOD_ROI_ZLIM`, overriding the literals. Falls back to the literals if no sites are found. Identical to `femtic_ens_post.py` / `femtic_gst_prep.py`. Also drives the aspect-ratio panel sizing (`MOD_PANEL_WIDTH = None`). |
| `MOD_ROI_PAD_XY` | Default `2.0` km. Padding around the site bounding box. |
| `MOD_ROI_ZLIM` | Default `[-1.0, 7.0]` km, positive-down; `None` leaves `MOD_ZLIM` as set. Negative lower bound keeps ~1 km of headroom above the z = 0 datum so topography is not clipped. |
| `ENS_SLICES` | Slice dicts for ensemble plot |
| `ENS_CMAP/CLIM` | Colormap and limits for ensemble plot |
| `ENS_XLIM` / `ENS_YLIM` / `ENS_ZLIM` | Default to `MOD_XLIM`/`YLIM`/`ZLIM` and follow the `MOD_ROI_AUTO` result automatically, unless set to their own separate value. |
| `ENS_STAT_ROWS` | Summary rows: `["mean", "std", "median"]` subset |
| `ENS_TICK_FONTSIZE` / `ENS_LABEL_FONTSIZE` | Axis tick/label font sizes for the ensemble slice plot. Defaults `6`/`7`, matching `fviz.plot_ensemble_slices`' own defaults; independent of the `MOD_*` pair above since the joint member × slice grid needs smaller text to stay readable. |
| `MOD_SHOW_IN_SPYDER` | `True` (default) and running inside Spyder (detected via `utl.runtime_env() == "spyder"`) → every saved figure (QC, model, ensemble) is also displayed inline in Spyder's Plots pane via `plt.show()`, in addition to being written to disk. No effect outside Spyder; set `False` to disable even under Spyder. |
| `MOD_MAP_MARKERS` | Extra lat/lon point markers on QC/model map panels only. An entry with `"is_model_centre": True` (instead of `"latlon"`) overrides the model-centre marker's style rather than plotting as a regular marker. |
| `MOD_SHOW_MODEL_CENTRE` | `True` (default): marks the model origin on `"map"` panels whenever `MOD_DISPLAY_COORDS` is `"utm"`/`"latlon"` (no-op for `"model"`). Default style: black `"+"`, `ms=10`; no legend entry. Override via `MOD_MAP_MARKERS` as above; set `False` to force off. |

---

### Titles and colour bars of the slice figures

Added 2026-11-11 (Claude Sonnet 5.5 (Anthropic), AI-generated, review before use); the forms of the title settings (`"auto"`, `None` / `False` / `""`, text with `{placeholders}`, dict per figure) are described in `femtic_viz_readme.md`, *Titles and colour bars*. Defaults give the figures of before. The joint bars need the `femtic_viz.py` of the same date.

| Variable | Default | Meaning |
|---|---|---|
| `MOD_TITLE` | `"auto"` | Title of the QC and model slice figures. `"auto"` = the model file name (as before); `None`, `False`, `""` = none; text with `{label}` (member directory), `{index}` (member number), `{file}` (model file), `{key}` (`"qc"` or `"model"`); dict `{"qc": spec, "model": spec, 0: spec, "default": spec}` |
| `MOD_PANEL_TITLES` | `True` | `False` = no panel titles (`Map  z = 1.0 km`, ...) |
| `MOD_CBAR_LOCATION` | `None` | `None` / `"panel"` = one colour bar per panel (as before); `"right"`, `"left"`, `"top"`, `"bottom"` = ONE joint bar for the panel grid |
| `MOD_CBAR_SHRINK`, `MOD_CBAR_THICKNESS` | `0.7`, `None` | Joint bar: length as a fraction of the side of the panel grid; thickness in cm (`None` = automatic) |
| `ENS_TITLE` | `"auto"` | Title of the joint ensemble figure (and of the per-member figures with `ENS_PER_MEMBER`): `"auto"` = `Ensemble (N members)` / `Ensemble member i -- file`; text with `{n_members}` (`{index}`, `{label}`, `{file}` for the member figures); dict `{"joint": spec, 0: spec, "default": spec}` |
| `ENS_PANEL_TITLES`, `ENS_CBAR_LOCATION`, `ENS_CBAR_SHRINK`, `ENS_CBAR_THICKNESS` | `True`, `None`, `0.7`, `None` | As the `MOD_*` settings, for the ensemble figure: with a position there are joint bars, one for log10(rho) and one for the std row |

---

## Slice specification

All slice dicts use `kind` (not `type`) as the panel-type key. Positions in
`MOD_SLICES` / `ENS_SLICES` are model-local **km** (the script converts them
to the metres expected by `femtic_viz` / `fem.resolve_slice_positions`):

```python
dict(kind="map",  z0=5.0)                                  # km
dict(kind="ew",   y0=0.0)                                  # km
dict(kind="ns",   x0=0.0)                                  # km
dict(kind="plane", point=[0.0, 0.0, 5.0], strike=45, dip=60)  # [x, y, z] km
dict(kind="plane", point=[0.0, 0.0, 0.0], strike=45, dip=90)  # z irrelevant for dip=90
```

A plane `point` must have **three** elements `[x, y, z]`; a 2-element point
raises `ValueError: resolve_pos_point: 'point' must have 3 elements`.
CRS-tagged forms `([E, N, z], "utm")` and `([lon, lat, z], "latlon")` take
E/N in metres and lon/lat in degrees (not km); only `z` is in km.
The same rule applies to `p1` / `p2` of `kind="profile"` (z optional).

---

## Changes from previous version

- `MOD_SLICES` updated from `{"type": "map", ...}` to
  `dict(kind="map", ...)` — the `"type"` key is no longer accepted by
  `fviz.plot_model_slices`.
- `depth_km=True`, `horiz_km=True` added to the `plot_model_slices` (QC)
  call. **Correction (2026-07-25):** these were also mistakenly being
  passed to the `plot_ensemble_slices` call, which does not accept them
  (no km-scaling support there) — that call would have raised `TypeError`
  the first time `PLOT_SLICES_ENS` was set `True`. Removed from that call;
  `plot_ensemble_slices` output remains in plain metres.
- 2026-07-25 (Claude Sonnet 5, Anthropic): Added `RANDOM_SEED` for
  optional reproducible ensembles. Fixed a reproducibility gap where
  `generate_data_ensemble` silently used its own unseeded generator even
  when the model-perturbation `rng` was seeded — it now accepts and
  forwards `rng` to `femtic.modify_data`.
- 2026-07-25 (Claude Sonnet 5, Anthropic): Added `MOD_TICK_FONTSIZE`/
  `MOD_LABEL_FONTSIZE` (QC/model slice plots) and `ENS_TICK_FONTSIZE`/
  `ENS_LABEL_FONTSIZE` (ensemble slice plot) — axis tick/label font sizes
  were previously fixed at `femtic_viz.py`'s internal defaults with no way
  to override them here.
- 2026-07-25 (Claude Sonnet 5, Anthropic): Added `MOD_SHOW_IN_SPYDER`
  (default `True`): when running inside Spyder, every saved figure is
  also displayed inline via `plt.show()`, without changing what gets
  saved to disk. No effect outside Spyder.
- 2026-08-13 (Claude Sonnet 5, Anthropic): Added
  `femtic_rto_prep_summary.md` output at end of run: writes user-set
  (UPPERCASE) parameters, script path, and run date/time.
- 2026-09-06 (Claude Sonnet 5, Anthropic): Added `MOD_SHOW_MODEL_CENTRE`
  (default `True`): marks the model origin on `"map"` panels whenever
  `MOD_DISPLAY_COORDS` is `"utm"`/`"latlon"`, via
  `fviz.plot_model_slices`'s new `show_model_centre` parameter. Override
  style with a `MOD_MAP_MARKERS` entry carrying `"is_model_centre": True`.
- 2026-09-19 (Claude Sonnet 5, Anthropic): Plot extent made consistent with
  `femtic_ens_post.py`: added `MOD_ROI_AUTO` / `MOD_ROI_PAD_XY` / `MOD_ROI_ZLIM`
  (same names, defaults and semantics). `MOD_XLIM`/`MOD_YLIM` now come from the
  site bounding box + padding and `MOD_ZLIM` from `MOD_ROI_ZLIM` whenever sites
  are available, for QC/model and ensemble slice plots alike (`ENS_*LIM` were
  plain aliases of `MOD_*LIM` and are re-linked to the result unless given a
  separate value). The origin/site-resolution block now also runs for
  `PLOT_SLICES_ENS`, so the ensemble plot gets the ROI even when
  `PLOT_MODEL`/`PLOT_SLICES_QC` are off.
- 2026-10-21 (Claude Sonnet 5.5, Anthropic): **RTO prior perturbation
  corrected** (ensemble spreads were far too small compared with GST).
  (1) The randomized-SVD sampler kept the roughest instead of the smoothest
  modes (0.1-0.5 % of the prior variance in a lattice test); replaced by exact
  sampling (`ens.sample_rto_prior`). (2) The perturbation had an implicit
  alpha = 1 (`MOD_MU`/`MOD_PDF` were defined but never passed on); `MOD_ALPHA`
  (or `MOD_ALPHA_CNV`) is now required. New `MOD_ALGO` values, `MOD_LAM_ALPHA`,
  `MOD_RTOL`, `MOD_START_FROM_PERTURBED`, `MOD_REMOVE_MEAN`; the run prints the
  RMS std and the energy ratio (expect ~1) of the draws. AI-generated; review
  before production use. Details: `ensembles_readme.md`, changelog 2026-10-21.
  `femtic_rto_prep.py` itself could not be run end to end here (`femtic_viz`
  was not available); the alpha block and `ens.generate_rto_model_ensemble`
  were tested separately (`rto_integration_test.py`).
- 2026-11-17 (Claude Sonnet 5.5, Anthropic): **fixed `NameError` when plots are
  switched off** (first `VIZ_N_SAMPLES` in `_n_viz = min(VIZ_N_SAMPLES, ...)`, behind
  it `PLOT_STR`, `DAT_*`, `MOD_*`, `PLOT_SLICES_QC/ENS`, which were all only defined
  inside blocks that run when a plot is requested). `PLOT_DATA`, `PLOT_MODEL`,
  `PLOT_SLICES_QC`, `PLOT_SLICES_ENS` and `VIZ_N_SAMPLES` are now set once at the top
  of the plot section; the model-plot block runs if `PLOT_MODEL`, `PLOT_SLICES_QC` or
  `PLOT_SLICES_ENS` is set (so the two slice figures no longer need `PLOT_DATA` or
  `PLOT_MODEL` to be on, as they did before). The `VIZ_SAMPLES` draw stays unconditional: it uses the
  same random generator, so for a given `RANDOM_SEED` the perturbed files do not
  depend on the plot switches. Also guarded: `PERTURB_DAT = False` and
  `PERTURB_MOD = False` (no ensemble is generated; with `PERTURB_MOD = False` the
  plots use `<TEMPLATES>/referencemodel.dat`; before, `MOD_REF_BASE` was `None`).
  Tested: `rto_prep_flagmatrix.py` (11 switch combinations, each a fresh run of
  the whole script on a mock ensemble, with the `femtic_viz` plot functions
  replaced by no-ops; identical `referencemodel.dat` for all plot settings with the
  same seed; the old script fails the plots-off cases), `rto_prep_smoke.py` (now five
  cases, no plot names pre-defined) and `rto_iter0_test.py` (94 checks). The real
  plot functions were not run. AI-generated; review before production use.
- 2026-11-16, second change (Claude Sonnet 5.5, Anthropic): homogeneous random
  starting model. New `MOD_ITER0_HOMOGENEOUS` (default `False`) and
  `MOD_ITER0_RANGE = [lo, hi]` (log10 Ohm.m), passed to
  `ens.generate_rto_model_ensemble(iter0_const_range=...)`: the iter0 file of every
  member is filled with one constant, drawn per member, uniform in log10(rho) in the
  range. Exclusive with `MOD_PERTURB_ITER0` and `MOD_START_FROM_PERTURBED`; checks
  before any directory is created. Needs the `ensembles.py` of the same date.
  Tested with `rto_iter0_test.py` (94 checks in total, mock lattice) and
  `rto_prep_smoke.py` (whole script, three modes). Not run with real data / mesh.
  AI-generated; review before production use.
- 2026-11-16 (Claude Sonnet 5.5, Anthropic): optional independent perturbation of
  the starting model. New `MOD_PERTURB_ITER0` (default `False`) and `MOD_ITER0_FILE`,
  passed to `ens.generate_rto_model_ensemble(perturb_iter0=..., iter0_file=...)`; see
  the table above and `ensembles_readme.md`, changelog 2026-11-16. Checks of the
  option combination run before any directory is created. Fixed a `SyntaxError` in the
  uploaded copy: `ENS_LIST = np.arange{50)` -> `np.arange(50)`. Needs the
  `ensembles.py` of the same date. Tested: `rto_iter0_test.py` (59 checks at the time, mock
  lattice) and `rto_prep_smoke.py` (whole script on a mock ensemble, data
  perturbation and plots switched off, option on and off). Not run with real
  data / mesh. AI-generated; review before production use.
- 2026-10-28 (Claude Sonnet 5.5, Anthropic): `femtic_rto_prep_summary.md`
  is now written to `ENSEMBLE_DIR` (the working directory set in the
  script) instead of next to the script: `utl.write_param_summary(fname,
  out_dir=ENSEMBLE_DIR)`. Needs the `util.py` of the same date.
  AI-generated; review before production use.
- 2026-11-01 (Claude Sonnet 5.5, Anthropic): fixed `ValueError:
  resolve_pos_point: 'point' must have 3 elements, got 2` raised by
  `fem.resolve_slice_positions()`: the `kind="plane"` entry in `MOD_SLICES`
  now has a 3-element `point`. `_slices_km_to_m()` now also converts `point`,
  `p1`, `p2`, `z_top`, `z_bot` from km to m (for CRS-tagged `utm`/`latlon`
  positions only `z` is scaled); before, only `x0`/`y0`/`z0` were converted,
  so a plane `point` would have been 1000x too small. Tested the conversion
  and `fem.resolve_slice_positions()` on the changed specs (femtic.py with a
  stubbed `ensembles`); the full script was not run here (no site/mesh data).
  AI-generated; review before production use.
- **2026-11-11 (Claude Sonnet 5.5 (Anthropic)):** Titles and colour bars of the slice figures: `MOD_TITLE` / `ENS_TITLE`, `MOD_PANEL_TITLES` / `ENS_PANEL_TITLES`, `MOD_CBAR_LOCATION` / `ENS_CBAR_LOCATION`, `MOD_CBAR_SHRINK` / `ENS_CBAR_SHRINK`, `MOD_CBAR_THICKNESS` / `ENS_CBAR_THICKNESS` (see *Titles and colour bars of the slice figures* above). `_plot_member_slices()` got the arguments `kind` and `index` for the title placeholders. Defaults: figures unchanged. Needs the `femtic_viz.py` of the same date. Not run end to end (no site / mesh data); the changed calls were compile-checked. AI-generated; review before production use.
- **2026-11-12 (Claude Sonnet 5.5 (Anthropic)):** Fault trace in the slice figures. `ENS_SLICES` / `MOD_SLICES` may contain a fault from two surface points (`kind="profile"` with `p1`, `p2` (km here), `dip`; `overlay=True`, optionally `panel=False` for the trace without a fault panel). `ENS_SLICES` are resolved with `fem.resolve_slice_positions()` for `plot_ensemble_slices` only when an entry is a `"profile"` or has `overlay` / `panel=False` (otherwise unchanged). Needs `femtic.py` / `femtic_viz.py` of the same date; see `femtic_viz_readme.md`, "Fault planes". Script not run here (syntax check and a synthetic-mesh test of the plot function). AI-generated; review before production use.
- **2026-11-20 (Claude Sonnet 5.5 (Anthropic)):** Phase curves of the data plots that leave the +-180 deg window are interrupted and the two points joined by a dotted line: new `DAT_PHASE_WRAP_MODE` / `DAT_PHASE_WRAP_STYLE`, passed to `fviz.plot_data_ensemble`. Needs `femtic_viz.py` / `data_viz.py` of the same date. AI-generated; review before use.
