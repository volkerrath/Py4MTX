# femtic_rto_prep.py

Prepare and launch a Randomize-Then-Optimize (RTO) ensemble for FEMTIC
MT inversion.

---

## Purpose

`femtic_rto_prep.py` implements the outer loop of the RTO algorithm:

```
for i = 1 : N_SAMPLES do
    Draw perturbed data:  d̃ᵢ ~ N(d, Cᵈ)
    Draw perturbed model: m̃ᵢ ~ N(mᵣₑf, Cᵐ)   (via roughness-matrix prior)
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
| `MOD_LAM` | Roughness regularisation weight |
| `MOD_LAM_MODE` | `"fixed"` / `"auto"` |

### Visualization
| Variable | Description |
|---|---|
| `PLOT_DATA` | Enable data ensemble plot |
| `PLOT_MODEL` | Enable model ensemble plot |
| `PLOT_SLICES_QC` | Enable per-member QC slice plot |
| `PLOT_SLICES_ENS` | Enable joint ensemble slice figure |
| `MOD_SLICES` | Slice dicts for QC/model plots (model-local metres, `kind` key). As of 2026-06-07 the QC plot uses the full shared `MOD_*` plotting config block (`MOD_CMAP`, `MOD_CLIM`, `MOD_XLIM`/`YLIM`/`ZLIM`, `MOD_OCEAN_COLOR`, `MOD_DPI`, site overlay, UTM origin, etc. — same as `femtic_gst_prep.py`/`femtic_ens_post.py`), not the older `QC_SLICES`/`QC_CMAP`/etc. variables. |
| `MOD_TICK_FONTSIZE` / `MOD_LABEL_FONTSIZE` | Axis tick/label font sizes for the QC/model slice plot. Defaults `7`/`8`, matching `fviz.plot_model_slices`' own defaults. |
| `ENS_SLICES` | Slice dicts for ensemble plot |
| `ENS_CMAP/CLIM` | Colormap and limits for ensemble plot |
| `ENS_STAT_ROWS` | Summary rows: `["mean", "std", "median"]` subset |
| `ENS_TICK_FONTSIZE` / `ENS_LABEL_FONTSIZE` | Axis tick/label font sizes for the ensemble slice plot. Defaults `6`/`7`, matching `fviz.plot_ensemble_slices`' own defaults; independent of the `MOD_*` pair above since the joint member × slice grid needs smaller text to stay readable. |

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
