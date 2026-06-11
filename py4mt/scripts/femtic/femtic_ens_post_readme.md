# femtic_ens_post.py

Algorithm-agnostic postprocessing of a FEMTIC ensemble.

Collects all converged inversion members, computes summary statistics in
log₁₀(ρ) space, assembles the empirical covariance, and saves everything
to a compressed `.npz` file.  Optionally produces slice figures for the
**best-nRMS member** (QC) and for the **ensemble statistics**
(mean, variance, median, MAD).

Supersedes `femtic_rto_post.py`.

---

## Workflow

```
Ensemble sub-directories  (rto_*, gst_*, member_*, …)
          │
          ▼
  nRMS filter  (NRMS_MAX)
          │
          ▼
  Load log₁₀(ρ)  →  N_members × N_free matrix
          │
          ├──▶  mean, variance, median, MAD, percentiles
          │
          └──▶  sklearn empirical covariance  →  optional sparse version
          │
          ▼
  <PREFIX>_results.npz
          │
          ├──▶  PLOT_QC = True   →  slice figure of best-nRMS member
          │
          └──▶  PLOT_STATS = True  →  block files + slice figures for each stat
```

---

## Configuration parameters

### Ensemble input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ENSEMBLE_DIR` | str | — | Root directory containing ensemble sub-directories. |
| `ENSEMBLE_NAME` | str | `"rto_*"` | Glob matched against sub-directory names. |
| `ENSEMBLE_PREFIX` | str | `"rto"` | Prefix for `.npz` output keys and default filenames. Set to `"gst"`, `"ens"`, etc. as appropriate. |
| `NRMS_MAX` | float | `1.4` | Members whose final nRMS exceeds this value are skipped. |

### Statistics

| Parameter | Type | Default | Description |
|---|---|---|---|
| `PERCENTILES` | list of float | `[2.3, 15.9, 50.0, 84.1, 97.7]` | Percentile levels (2-σ / 1-σ normal-equivalent). |

### Covariance

| Parameter | Type | Default | Description |
|---|---|---|---|
| `SPARSIFY` | bool | `True` | Threshold small entries in the covariance to create a CSR sparse version. |
| `SPARSE_THRESH` | float | `1e-8` | Relative threshold: entries with `|C_ij| / max|C| ≤ SPARSE_THRESH` are zeroed. |

### Output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ENSEMBLE_RESULTS` | str | `<PREFIX>_results.npz` | Path for the output `.npz` file. |
| `MESH_FILE` | str | `templates/mesh.dat` | Tetrahedral mesh — required for all slice plots. |
| `OUT` | bool | `True` | Verbose console output. |

### QC slice plot

Produces a single slice figure of the **lowest-nRMS** converged member.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `PLOT_QC` | bool | `False` | Enable QC slice plot. |
| `PLOT_QC_FILE` | str | `<prefix>_qc.pdf` | Output path; `None` → interactive `show()`. |
| `PLOT_QC_DPI` | int | `200` | Figure DPI. |

### Statistics slice plots

Writes each selected statistic as a FEMTIC block file, then plots it.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `PLOT_STATS` | bool | `False` | Enable statistics slice plots. |
| `PLOT_STATS_WHAT` | list of str | `["avg","var","med","mad"]` | Which statistics to plot. Subset of `"avg"`, `"var"`, `"med"`, `"mad"`. |
| `PLOT_STATS_DIR` | str | `stats_plots/` | Destination for block files and figures. |
| `PLOT_STATS_DPI` | int | `200` | Figure DPI. |

Block files are written using the lowest-nRMS member as format template
(preserves header, bounds, and flag columns).  Output filenames follow the
pattern `resistivity_block_<prefix>_<stat>.dat`.

### Shared slice parameters

Both `PLOT_QC` and `PLOT_STATS` use the same slice/plot config block.

| Parameter | Description |
|---|---|
| `PLOT_SLICES` | List of slice-spec dicts (`kind`, `z0`/`x0`/`y0`). Kinds: `"map"`, `"ns"`, `"ew"`, `"plane"`. |
| `PLOT_CMAP` | Matplotlib colormap name. |
| `PLOT_CLIM` | `[log10(ρ_min), log10(ρ_max)]`; `None` = auto. |
| `PLOT_XLIM / YLIM / ZLIM` | Global axis limits (model-local metres); `None` = auto. |
| `PLOT_OCEAN_COLOR` | Flat colour for ocean/lake cells. |
| `PLOT_OCEAN_RHO` | Ocean cell sentinel value (Ω·m). |
| `PLOT_AIR_BGCOLOR` | Axes facecolor for air; `None` = figure default. |
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
| `ORIGIN_METHOD` | `"box"` (bounding-box midpoint) / `"average"` / `None` (use literal values). |
| `UTM_ORIGIN_LAT/LON/E/N` | Manual override; ignored when `ORIGIN_METHOD` estimates from `SITE_DAT`. |
| `UTM_ZONE_OVERRIDE` | Force a specific UTM zone string; `None` = auto-detect. |
| `DISPLAY_COORDS` | `"model"` / `"utm"` / `"latlon"`. |

---

## Output files

### `.npz` archive

Keys follow the pattern `<PREFIX>_<stat>`:

| Key | Shape | Description |
|---|---|---|
| `<P>_model_list` | `(N, 3)` | `[block_file, n_iter, nRMS]` per accepted member. |
| `<P>_ens` | `(N_members, N_free)` | Stacked ensemble in log₁₀(Ω·m). |
| `<P>_cov` | `(N_free, N_free)` | Empirical covariance matrix. |
| `<P>_avg` | `(N_free,)` | Element-wise mean over members. |
| `<P>_var` | `(N_free,)` | Element-wise variance over members. |
| `<P>_med` | `(N_free,)` | Element-wise median over members. |
| `<P>_mad` | `(N_free,)` | Median absolute deviation. |
| `<P>_prc` | `(N_prc, N_free)` | Percentile values at `PERCENTILES` levels. |

### Statistics block files (PLOT_STATS = True)

```
<PLOT_STATS_DIR>/
  resistivity_block_<prefix>_avg.dat
  resistivity_block_<prefix>_var.dat
  resistivity_block_<prefix>_med.dat
  resistivity_block_<prefix>_mad.dat
  <prefix>_avg.pdf
  <prefix>_var.pdf
  <prefix>_med.pdf
  <prefix>_mad.pdf
```

All `.dat` files are valid FEMTIC resistivity block files usable as input
to `femtic_mod_edit.py`, `femtic_mod_plot_slice.py`, or the VTK export
pipeline.

---

## Quick start

```bash
conda activate EM
# Edit ENSEMBLE_DIR, ENSEMBLE_NAME, ENSEMBLE_PREFIX, NRMS_MAX,
#      MESH_FILE, PLOT_QC, PLOT_STATS at the top of the script.
python femtic_ens_post.py
```

---

## Bug fix vs. femtic_rto_post.py

The original script computed mean, variance, and median with `axis=1`,
which reduced over the **free-parameter** axis rather than the **member**
axis.  The ensemble matrix has shape `(N_members, N_free)`, so the
correct reduction axis is `axis=0`.  All aggregate statistics are now
correct.

---

## Relationship to other scripts

| Script | Purpose |
|---|---|
| `femtic_rto_prep.py` | Generate RTO ensemble members. |
| `femtic_gst_prep.py` | Generate GST ensemble members. |
| **`femtic_ens_post.py`** | Postprocess any ensemble: statistics, covariance, slice plots. |
| `femtic_mod_math.py` | Write average and smoothed-median as block files from an N-subset. |
| `femtic_mod_edit.py` | Apply arithmetic operations to a single model. |
| `femtic_mod_plot_slice.py` | Plot slice figures from a single model file. |

---

## Changelog

| Date | Author | Change |
|---|---|---|
| 2025-04-30 | vrath | Created as `femtic_rto_post.py`. |
| 2026-03-03 | Claude (Anthropic) | Renamed user-set parameters to UPPERCASE; generated README. |
| 2026-05-27 | vrath / Claude Sonnet 4.6 (Anthropic) | Added `femtic_viz` import; `PLOT_QC` block with minimal `plot_model_slices` call. |
| 2026-06-11 | vrath / Claude Sonnet 4.6 (Anthropic) | Renamed → `femtic_ens_post.py`; fixed `axis` bug in mean/var/median/MAD; replaced thin `PLOT_QC` block with full CRS-aware `_plot_slice()` helper; added `PLOT_STATS` block (writes block files + figures for avg/var/med/MAD); added `ENSEMBLE_PREFIX` config var for generic naming. |
