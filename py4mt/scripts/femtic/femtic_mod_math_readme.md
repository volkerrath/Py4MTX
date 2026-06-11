# femtic_mod_math.py

Generate synthetic ensemble members from an N-subset of FEMTIC inversion
results.  Two derived models are produced:

| Output member | Description |
|---|---|
| `resistivity_block_avg.dat` | Element-wise arithmetic mean in log₁₀(ρ) |
| `resistivity_block_smooth_median.dat` | Element-wise median, then mesh-adaptive spatial smoothing |

Both files use the template (header, bounds, flag columns) of the lowest-nRMS
member in the subset.

---

## Workflow

```
Ensemble directories / block files
        │
        ▼
  nRMS filter (NRMS_MAX)
        │
        ▼
  SUBSET_LIST selection
        │
        ▼
  Load log₁₀(ρ)  →  N × M matrix
        │
        ├──▶  column-wise mean   ──▶  resistivity_block_avg.dat
        │
        └──▶  column-wise median
                    │
                    ▼
               spatial smooth   ──▶  resistivity_block_smooth_median.dat
        │
        ▼
  fviz.plot_model_slices  (optional, PLOT = True)
```

---

## Configuration parameters

### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ENSEMBLE_DIR` | str | — | Root directory containing ensemble sub-directories. |
| `ENSEMBLE_NAME` | str | `"rto_*"` | Glob matched against sub-directory names inside `ENSEMBLE_DIR`. |
| `BLOCK_FILES` | list or `None` | `None` | When not `None`, skip the directory scan and use this explicit list of block-file paths.  `NRMS_MAX` / `SUBSET_LIST` still apply; nRMS check is skipped for files without a sibling `femtic.cnv`. |
| `MESH_FILE` | str | — | Tetrahedral mesh (`mesh.dat`) — required for the smooth operation. |

### Subset selection

| Parameter | Type | Default | Description |
|---|---|---|---|
| `NRMS_MAX` | float | `1.4` | Members whose final nRMS (last line of `femtic.cnv`) exceeds this value are skipped.  Set to `np.inf` to disable. |
| `SUBSET_LIST` | list of int or `None` | `None` | 0-based indices into the nRMS-filtered, sorted file list.  `None` = all converged members.  Same format as `ENS_LIST` in `femtic_rto_prep.py`. |

### Output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `OUT_DIR` | str | — | Destination for output block files and optional PDFs. |
| `OUT` | bool | `True` | Verbose console output. |

### Ocean / air handling

| Parameter | Type | Default | Description |
|---|---|---|---|
| `OCEAN` | `None`/bool | `None` | `None` = auto-infer from region-1 heuristic. `True`/`False` = force. |
| `AIR_RHO` | float | `1e9` | Ω·m written for region 0 (air sentinel). |
| `OCEAN_RHO` | float | `0.25` | Ω·m written for region 1 when ocean is active. |

### Smoothing

The median model is smoothed using the same three kernel modes as
`femtic_mod_edit.py`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `SMOOTH_MODE` | str | `"physical"` | `"physical"` — global Gaussian, σ = `SMOOTH_SIGMA` m.<br>`"knn_uniform"` — flat average over K nearest neighbours.<br>`"knn_gauss"` — per-region Gaussian, σᵢ = `SMOOTH_KNN_SIGMA_FRAC` × d_{i,K}. |
| `SMOOTH_SIGMA` | float | `3000.0` | Gaussian σ in metres (`"physical"` mode only). |
| `SMOOTH_K` | int | `100` | Number of nearest neighbours (all modes). |
| `SMOOTH_KNN_SIGMA_FRAC` | float | `0.5` | Per-region σ fraction (`"knn_gauss"` only): σᵢ = frac × d_{i,K}. |
| `SMOOTH_MAX_GB` | float | `4.0` | Max RAM in GiB for chunked dense fallback (`"physical"`, no SciPy). |

`"knn_uniform"` and `"knn_gauss"` require SciPy (`scipy.spatial.cKDTree`).
`"physical"` falls back to a chunked dense matrix path when SciPy is absent.

**Kernel comparison**

| Mode | Physical reach | Memory | SciPy required |
|---|---|---|---|
| `physical` | Fixed σ everywhere | O(N²) chunk | No (fallback) |
| `knn_uniform` | Adapts to mesh density | O(N × K) | Yes |
| `knn_gauss` | Adapts; smooth decay | O(N × K) | Yes |

### Plotting

Controlled by the same config block used in `femtic_mod_edit.py` and
`femtic_mod_plot_slice.py`.  Both output members are plotted when `PLOT = True`.

| Parameter | Description |
|---|---|
| `PLOT` | `True` to produce slice figures. |
| `PLOT_DPI` | Figure resolution. |
| `PLOT_CMAP` | Matplotlib colormap name. |
| `PLOT_CLIM` | `[log10(ρ_min), log10(ρ_max)]`; `None` = auto. |
| `PLOT_OCEAN_COLOR` | Flat colour for ocean/lake cells. |
| `PLOT_SLICES` | List of slice dicts (`kind`, `z0`/`x0`/`y0`). |
| `PLOT_XLIM / YLIM / ZLIM` | Global axis limits (model-local metres). |
| `DEPTH_KM / HORIZ_KM` | Axis units. |
| `PLOT_EQUAL_ASPECT` | Equal aspect ratio on map/curtain panels. |
| `PLOT_PANEL_HEIGHT` | Panel height in cm. |
| `PLOT_NROWS / NCOLS` | Grid layout; `None` = 1 row × N columns. |

### Site overlay

| Parameter | Description |
|---|---|
| `SITE_DAT` | Path to `site.dat` CSV; `None` to disable. |
| `SITE_NAMES` | `None` = all sites; list of strings = subset. |
| `PLOT_SITES_MAPS / SLICES` | Toggle site markers on map / curtain panels. |
| `PROJECTION_DIST` | Maximum projection distance (m) for curtain panels. |
| `SITE_MARKER` | Marker style dict for map overlays. |

### Geographic / UTM origin

| Parameter | Description |
|---|---|
| `ORIGIN_METHOD` | `None` / `"box"` / `"average"` — how to estimate UTM origin from `SITE_DAT`. |
| `UTM_ORIGIN_LAT/LON/E/N` | Manual override; ignored when `ORIGIN_METHOD` estimates from sites. |
| `UTM_ZONE_OVERRIDE` | Force a specific UTM zone string; `None` = auto. |
| `DISPLAY_COORDS` | `"model"` / `"utm"` / `"latlon"`. |

---

## Input modes

### Directory scan (default)

`BLOCK_FILES = None`.  The script globs `ENSEMBLE_DIR` for sub-directories
matching `ENSEMBLE_NAME`, reads `femtic.cnv` from each, applies the nRMS
filter, then loads `resistivity_block_iter<N>.dat`.

```python
ENSEMBLE_DIR  = "/data/ubinas/ensemble/"
ENSEMBLE_NAME = "rto_*"
BLOCK_FILES   = None
```

### Explicit file list

Set `BLOCK_FILES` to bypass the directory scan entirely.  Useful when
the block files you want do not follow the standard naming convention, or
when mixing members from different ensemble runs.

```python
BLOCK_FILES = [
    "/data/ubinas/ensemble/rto_001/resistivity_block_iter12.dat",
    "/data/ubinas/ensemble/rto_007/resistivity_block_iter15.dat",
    "/data/ubinas/other_run/resistivity_block_iter10.dat",
]
```

---

## Output files

```
<OUT_DIR>/
  resistivity_block_avg.dat          # arithmetic mean in log₁₀ space
  resistivity_block_smooth_median.dat # median → smoothed
  math_avg.pdf                        # slice figure (PLOT=True)
  math_smooth_median.pdf              # slice figure (PLOT=True)
```

Both `.dat` files are valid FEMTIC resistivity block files and can be used
directly as starting models for subsequent inversions or as input to
`femtic_mod_edit.py`, `femtic_mod_plot_slice.py`, or ParaView via the VTK
export pipeline.

---

## Quick start

```bash
# 1. Activate the EM conda environment
conda activate EM

# 2. Edit the configuration section at the top of the script:
#    ENSEMBLE_DIR, NRMS_MAX, SUBSET_LIST, MESH_FILE, OUT_DIR, SMOOTH_*

# 3. Run
python femtic_mod_math.py
```

---

## Design notes

**Why two separate members?**
The average is the best linear estimator of the ensemble mean and minimises
squared error relative to the posterior.  The smoothed median is more robust
to outlier members and to highly resistive / conductive artefacts from
individual runs that did not converge cleanly.  Both are useful starting
points for further processing or as reference models.

**Why smooth the median but not the average?**
The mean already combines information from all members and tends to be
spatially smoother than any individual member.  The median picks the central
value independently per region; without smoothing, it can produce
block-by-block discontinuities at region boundaries.  Applying one smooth
pass restores spatial continuity while preserving the robustness advantage.

**Relationship to other scripts**

| Script | Purpose |
|---|---|
| `femtic_rto_prep.py` | Generate ensemble members via RTO. |
| `femtic_rto_post.py` | Compute aggregate statistics (mean, variance, MAD, percentiles) and covariance; save as `.npz`. |
| **`femtic_mod_math.py`** | Generate physically valid block-file members (average, smoothed median) from any N-subset. |
| `femtic_mod_edit.py` | Apply single arithmetic operations to one existing model. |
| `femtic_mod_plot_slice.py` | Plot slice figures from a single model file. |

---

## Changelog

| Date | Author | Change |
|---|---|---|
| 2026-06-11 | vrath / Claude Sonnet 4.6 (Anthropic) | Created. |
