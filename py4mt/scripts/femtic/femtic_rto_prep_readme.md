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
| Variable | Description |
|---|---|
| `DAT_METHOD` | `"add"` (additive noise) |
| `DAT_PDF` | `["normal", mean, std]` |
| `RESET_ERRORS` | Replace errors before perturbation |
| `ERRORS` | Error floors per data type |

### Model perturbation (`PERTURB_MOD`)
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
| `MOD_REMOVE_MEAN` | `False` (default). `True`: subtract each draw's global mean (offset set by the arbitrary shift `lam`, invisible to `R`). |

Removed (2026-10-21): `MOD_MU`, `MOD_PDF` (never passed on; implied alpha = 1),
`MOD_N_EIG`, `MOD_N_OVERSAMPLING`, `MOD_N_POWER_ITER`, `MOD_SIGMA2_RESIDUAL`,
`MOD_LAM_MODE`, `MOD_SOLVER`, `MOD_PRECOND`; `MOD_ALGO = "low rank"` is gone.

### Visualization
| Variable | Description |
|---|---|
| `PLOT_DATA` | Enable data ensemble plot |
| `PLOT_MODEL` | Enable model ensemble plot |
| `PLOT_SLICES_QC` | Enable per-member QC slice plot |
| `PLOT_SLICES_ENS` | Enable joint ensemble slice figure |
| `MOD_SLICES` | Slice dicts for QC/model plots (model-local metres, `kind` key). As of 2026-06-07 the QC plot uses the full shared `MOD_*` plotting config block (`MOD_CMAP`, `MOD_CLIM`, `MOD_XLIM`/`YLIM`/`ZLIM`, `MOD_OCEAN_COLOR`, `MOD_DPI`, site overlay, UTM origin, etc. — same as `femtic_gst_prep.py`/`femtic_ens_post.py`), not the older `QC_SLICES`/`QC_CMAP`/etc. variables. |
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

## Slice specification

All slice dicts use `kind` (not `type`) as the panel-type key:

```python
dict(kind="map",  z0=5000.0)
dict(kind="ew",   y0=0.0)
dict(kind="ns",   x0=0.0)
dict(kind="plane", point=[0,0,5000], strike=45, dip=60)
```

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
