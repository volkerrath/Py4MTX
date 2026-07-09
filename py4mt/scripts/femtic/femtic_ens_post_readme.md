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
          ├──▶  MOD_QC = True   →  slice figure of best-nRMS member
          │
          └──▶  MOD_STATS = True  →  block files + slice figures for each stat
```

**Plotting config is shared with `femtic_gst_prep.py` / `femtic_rto_prep.py`.**
All `MOD_*` variables below (mesh, ocean/air, UTM origin, display coordinates,
site overlay, slice specs, colormap/limits, alpha/blanking, figure layout)
use the same names and semantics as the ensemble-generation scripts, so a
config block can be copied between scripts with no renaming.

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
| `MOD_MESH` | str | `templates/mesh.dat` | Tetrahedral mesh — required for all slice plots. |
| `OUT` | bool | `True` | Verbose console output. |

### Ocean / air handling

Must match the values used by the FEMTIC inversion that produced the ensemble.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_OCEAN` | bool / None | `None` | `None` = auto-infer; `True`/`False` forces ocean-present/absent. |
| `MOD_AIR_RHO` | float | `1.0e9` | Ω·m sentinel for air cells (region 0), used when writing stat block files. |
| `MOD_OCEAN_RHO` | float | `0.25` | Ω·m sentinel for ocean cells (region 1), used for both block-file writing and plotting. |

### QC slice plot

Produces a single slice figure of the **lowest-nRMS** converged member.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_QC` | bool | `False` | Enable QC slice plot. |
| `MOD_QC_FILE` | str | `<prefix>_qc.pdf` | Output path; `None` → interactive `show()`. |

### Statistics slice plots

Writes each selected statistic as a FEMTIC block file, then plots it.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_STATS` | bool | `False` | Enable statistics slice plots. |
| `MOD_STATS_WHAT` | list of str | `["avg","var","med","mad"]` | Which statistics to plot. Subset of `"avg"`, `"var"`, `"med"`, `"mad"`. |
| `MOD_STATS_DIR` | str | `stats_plots/` | Destination for block files and figures. |

Block files are written using the lowest-nRMS member as format template
(preserves header, bounds, and flag columns).  Output filenames follow the
pattern `resistivity_block_<prefix>_<stat>.dat`.

### Shared slice / plot parameters

Both `MOD_QC` and `MOD_STATS` use the same slice/plot config block — this
block is identical (variable names and defaults) to `femtic_gst_prep.py`
and `femtic_rto_prep.py`.

| Parameter | Description |
|---|---|
| `MOD_SLICES` | List of slice-spec dicts (`kind`, `z0`/`x0`/`y0`). Kinds: `"map"`, `"ns"`, `"ew"`, `"plane"`. |
| `MOD_XLIM / YLIM / ZLIM` | Global axis limits (model-local metres); `None` = auto. |
| `MOD_CMAP` | Matplotlib colormap name. |
| `MOD_DPI` | Figure DPI, used by both `MOD_QC` and `MOD_STATS` plots. |
| `MOD_CLIM` | `[log10(ρ_min), log10(ρ_max)]`; `None` = auto. |
| `MOD_OCEAN_COLOR` | Flat colour for ocean/lake cells; `None` = colormap. |
| `MOD_AIR_COLOR` | Flat colour for air cells. |
| `MOD_AIR_BGCOLOR` | Axes facecolor for air; `None` = figure default. |
| `MOD_ALPHA_FILE` | Path to a second (e.g. sensitivity) block file used to fade/blank low-sensitivity cells; `None` = disabled. |
| `MOD_ALPHA_MODE` | `"fade"` or `"blank"`. |
| `MOD_ALPHA_BLANK_THRESH` | Threshold below which cells are faded/blanked. |
| `MOD_EQUAL_ASPECT` | Equal aspect ratio on map/curtain panels. |
| `MOD_DEPTH_KM / HORIZ_KM` | Axis units. |
| `MOD_PANEL_HEIGHT` | Panel height in cm. |
| `MOD_PANEL_WIDTH` | Panel width in cm; `None` = auto from aspect ratio. |
| `MOD_FIGSIZE` | `[w, h]` cm; overrides auto layout when set. |
| `MOD_NROWS / NCOLS` | Grid layout; `None` = 1 row × N columns. |

### Site overlay

| Parameter | Description |
|---|---|
| `MOD_SITE_DAT` | Path to `site.dat` CSV; `None` to fall back to `MOD_SITE_NUMBER`. |
| `MOD_SITE_NAMES` | `None` = all sites; list of strings = subset. |
| `MOD_SITE_NUMBER` | Fallback site number(s) from `observe.dat` (int or list of ints), used only when `MOD_SITE_DAT` is unavailable. |
| `MOD_PLOT_SITES_MAPS / SLICES` | Toggle site markers on map / curtain panels. |
| `MOD_PROJECTION_DIST` | Maximum projection distance (m) for curtain panels; `None` = show all sites on every panel. |
| `MOD_SITE_MARKER / MARKER_SLICES` | Marker style dicts for map / curtain overlays. |
| `MOD_MAP_MARKERS` | Extra point markers on map panels only. |

### Geographic / UTM origin

| Parameter | Description |
|---|---|
| `MOD_ORIGIN_METHOD` | `"box"` (bounding-box midpoint) / `"average"` / `None` (use literal values). |
| `MOD_UTM_ORIGIN_LAT/LON/E/N` | Manual override; ignored when `MOD_ORIGIN_METHOD` estimates from `MOD_SITE_DAT`. |
| `MOD_UTM_ZONE_OVERRIDE` | Force a specific UTM zone string; `None` = auto-detect. |
| `MOD_DISPLAY_COORDS` | `"model"` / `"utm"` / `"latlon"`. |

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

### Statistics block files (MOD_STATS = True)

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
#      MOD_MESH, MOD_QC, MOD_STATS at the top of the script.
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
| 2026-07-07 | vrath / Claude Sonnet 5 (Anthropic) | Renamed the entire plotting config surface to match `femtic_gst_prep.py` / `femtic_rto_prep.py` exactly (`MOD_*` prefix throughout: mesh, ocean/air, UTM origin, display coords, site overlay, slice specs, colormap/limits, figure layout). Added `MOD_OCEAN`/`MOD_AIR_RHO`, `MOD_SITE_NUMBER` (observe.dat fallback), `MOD_AIR_COLOR`, `MOD_ALPHA_FILE/MODE/BLANK_THRESH`, `MOD_PANEL_WIDTH`, `MOD_FIGSIZE`. Removed a latent duplicate `MOD_XLIM/YLIM/ZLIM` assignment that silently discarded the first (non-`None`) values. A config block can now be copied between `femtic_ens_post.py` and the ensemble-generation scripts with no renaming. |
| 2026-07-09 | vrath / Claude Sonnet 5 (Anthropic) | Merged `MOD_QC_DPI` / `MOD_STATS_DPI` into a single `MOD_DPI` knob, matching `femtic_gst_prep.py` and `femtic_nss.py` (one figure-DPI setting per script, not one per plot type). `_plot_slice()` no longer takes a `dpi` argument; it reads `MOD_DPI` directly. |
