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
| `ENS_LIST` | `None` = all members; `[i, j, k, …]` = explicit index list |

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
| `QC_SLICES` | Slice dicts for QC plots (model-local metres, `kind` key) |
| `ENS_SLICES` | Slice dicts for ensemble plot |
| `ENS_CMAP/CLIM` | Colormap and limits for ensemble plot |
| `ENS_STAT_ROWS` | Summary rows: `["mean", "std", "median"]` subset |

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

- `FROM_TO` renamed to `ENS_LIST`; accepts `None` (all members) or an
  explicit list of member indices `[i, j, k, …]`.  Range semantics
  (`[start, stop]`) are not supported — a list always means exact indices.
  Matching change in `ensembles.py` (`_resolve_fromto` helper).
- `VIZ_SAMPLES` now drawn from `ENS_MEMBERS` (the resolved active index set)
  instead of `range(N_SAMPLES)`; file lists for data/model/QC/ensemble plots
  all scoped to `ENS_MEMBERS` — prevents `FileNotFoundError` when `ENS_LIST`
  restricts the run to a subset of members.
- `MOD_SLICES` updated from `{"type": "map", ...}` to
  `dict(kind="map", ...)` — the `"type"` key is no longer accepted by
  `fviz.plot_model_slices`.
- `depth_km=True`, `horiz_km=True` added to `plot_model_slices` (QC)
  and `plot_ensemble_slices` calls.
