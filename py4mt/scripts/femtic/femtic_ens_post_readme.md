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
config block can be copied between scripts with no renaming. Region-of-
interest auto-scaling (`MOD_ROI_*`) is specific to this script.

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
| `QDIFF_PAIRS` | list of `(lo, hi)` | `[(15.9, 84.1)]` | Percentile-pair differences `\|P_hi - P_lo\|` computed as extra, outlier-robust spread statistics (both values must also appear in `PERCENTILES`). Saved as `<P>_qdiff_<lo>_<hi>` and plottable under the same key in `MOD_STATS_WHAT`. |

### Covariance

| Parameter | Type | Default | Description |
|---|---|---|---|
| `COMPUTE_COV` | bool | `True` | Set `False` to skip covariance estimation entirely. Statistics, percentiles, and slice plots are unaffected; only the `*_cov*` keys are omitted from the `.npz`. |
| `COV_METHOD` | str | `"full"` | `"full"` = dense empirical covariance. `"low_rank"` = thin SVD of the centred ensemble instead — see below. |
| `SPARSIFY` | bool | `True` | (`COV_METHOD="full"` only) Threshold small entries in the covariance to create a CSR sparse version. |
| `SPARSE_THRESH` | float | `1e-8` | Relative threshold: entries with `|C_ij| / max|C| ≤ SPARSE_THRESH` are zeroed. |

**`COV_METHOD="low_rank"`.** The empirical covariance of `n_members`
samples has rank ≤ `n_members - 1`. Since `n_members` (tens to a few
hundred) is normally far smaller than `n_free` (thousands to hundreds of
thousands of mesh cells), forming the dense `n_free × n_free` covariance
is both the slowest step in the script and, past a few thousand free
parameters, not something that fits in memory at all (e.g. `n_free=1e5`
→ 80 GB just to store it).

Instead, `"low_rank"` takes the thin SVD of the centred ensemble matrix
`Xc` (shape `n_members × n_free`):

```
Xc = U S Vᵀ         (economy SVD, cost O(n_members² · n_free))
C  = Xcᵀ Xc / (m-1) = Vᵀᵀ diag(S²/(m-1)) Vᵀ     — exact, not an approximation
```

This is stored as `f"{P}_cov_eigval"` (`(r,)`, `r = min(n_members, n_free)`)
and `f"{P}_cov_eigvec"` (`(n_free, r)`) in place of the dense `f"{P}_cov"`.
The full covariance reconstructs exactly as
`eigvec @ diag(eigval) @ eigvec.T`, and downstream consumers that only
need matrix-vector products, low-rank sampling, or leading eigenpairs
(e.g. `gdm.py`, prior sampling in `ensembles.py`) can use the factors
directly without ever forming the dense matrix. Cost drops to
`O(n_members² · n_free)` time and `O(n_members · n_free)` memory.

**Other ways to speed up `COV_METHOD="full"`**, if the dense matrix is
genuinely needed downstream:
- Make sure NumPy/SciPy are linked against a multi-threaded BLAS (OpenBLAS
  or MKL) — the `Xᵀ X` product inside `empirical_covariance` is a single
  `dgemm` call and will use all cores automatically; check with
  `numpy.show_config()` and set `OMP_NUM_THREADS` / `MKL_NUM_THREADS`
  before launching Python if it's pinned to one core.
- Cast `ens_matrix` to `float32` before the covariance call if the extra
  precision isn't needed — halves memory traffic and roughly doubles
  BLAS throughput on most hardware.
- If only a fixed block/neighbourhood structure of `C` is ever queried
  (e.g. covariance restricted to a region of interest), compute that
  sub-block directly (`Xc[:, idx].T @ Xc[:, idx]`) instead of the full
  matrix and slicing afterwards.
- For truly large `n_free` where even the low-rank factors are awkward,
  a randomized SVD (`sklearn.utils.extmath.randomized_svd` on `Xc`, or
  `scipy.sparse.linalg.svds`) approximates the same factorisation with
  fewer passes over the data — usually unnecessary here since the thin
  SVD above is already exact and cheap, but relevant if `n_members`
  itself becomes large (thousands).
- `joblib.Parallel` / `multiprocessing` can shard the outer product sum
  across process pools for the dense case, but for this problem shape
  (`n_members` small, `n_free` large) the algorithmic fix (`"low_rank"`)
  outperforms parallelizing the naive computation by orders of magnitude,
  so that's the recommended first step rather than adding process-level
  parallelism to the dense path.

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
| `MOD_STATS_WHAT` | list of str | `["avg","var","med","mad"]` + one key per `PERCENTILES` level + one per `QDIFF_PAIRS` entry | Which statistics to plot. Subset of `"avg"`, `"var"`, `"med"`, `"mad"`, plus auto-generated percentile keys (e.g. `2.3` → `"p2_3"`, `50.0` → `"p50"`, `97.7` → `"p97_7"`) and qdiff keys (e.g. `(15.9, 84.1)` → `"qdiff_15_9_84_1"`). Default includes all of them. |
| `MOD_STATS_DIR` | str | `stats_plots/` | Destination for block files and figures. |
| `MOD_STATS_CLIM` | dict | `{"var": [-2,2], "mad": [-2,2], "qdiff_...": [-2,2]}` | Per-statistic colour-scale override, keyed like `MOD_STATS_WHAT`. Each value is `[vmin, vmax]` or `None` (auto). `"avg"`/`"med"`/percentile keys aren't listed and fall back to `MOD_CLIM` automatically (same log10(Ω·m) space as the model). `"var"`/`"mad"`/`"qdiff_*"` (spread statistics on an unrelated, typically much narrower scale) default to a fixed `[-2, 2]` range; set a key to `None` to switch that one back to auto per-panel scaling. |

**Why `MOD_STATS_CLIM` exists.** `MOD_CLIM` fixes the colour scale for the
resistivity model itself (log10 Ω·m), typically something like `[0, 4]`.
Applying that same range to `VAR` (variance of log10ρ, usually well under 1)
or `MAD`/`QDIFF` (spread in log10ρ, also a small number) made those panels
come out essentially blank — everything mapped to the bottom of the
colour scale. `MOD_STATS_CLIM` lets `AVG`/`MED`/percentile panels keep
sharing `MOD_CLIM` (so they're visually comparable to each other and to
the QC plot) while `VAR`/`MAD`/`QDIFF_*` get their own, much narrower
fixed range (`[-2, 2]` by default) so they stay comparable to *each
other* across members/panels/runs; edit the values in `MOD_STATS_CLIM`
directly, or set a key to `None`, to change that behaviour.

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
| `MOD_XLIM / YLIM / ZLIM` | Global axis limits (model-local metres); `None` = auto. Overridden automatically by `MOD_ROI_AUTO` (below) when sites are available. |
| `MOD_ROI_AUTO` | Default `True`. When site positions are available, derives `MOD_XLIM`/`MOD_YLIM` from the site bounding box + `MOD_ROI_PAD_XY`, and sets `MOD_ZLIM` from `MOD_ROI_ZLIM` — overriding any literal values set above. Falls back to the literals (or full-mesh auto-scaling) when no sites are found. |
| `MOD_ROI_PAD_XY` | Default `5000.0` m. Padding added around the site bounding box for `MOD_XLIM`/`MOD_YLIM`. |
| `MOD_ROI_ZLIM` | Default `[0.0, 20000.0]` m (positive-down). Depth range applied to `MOD_ZLIM` for `ns`/`ew`/`plane` panels when `MOD_ROI_AUTO=True`; `None` leaves `MOD_ZLIM` untouched. |
| `MOD_CMAP` | Matplotlib colormap name. |
| `MOD_DPI` | Figure DPI, used by both `MOD_QC` and `MOD_STATS` plots. |
| `MOD_CLIM` | `[log10(ρ_min), log10(ρ_max)]`; `None` = auto. Default colour scale for `MOD_QC` and for any `MOD_STATS` panel not overridden by `MOD_STATS_CLIM`. |
| `MOD_OCEAN_COLOR` | Flat colour for ocean/lake cells; `None` = colormap. |
| `MOD_AIR_COLOR` | Flat colour for air cells. |
| `MOD_AIR_BGCOLOR` | Axes facecolor for air; `None` = figure default. |
| `MOD_ALPHA_FILE` | Path to a second (e.g. sensitivity) block file used to fade/blank low-sensitivity cells; `None` = disabled. |
| `MOD_ALPHA_MODE` | `"fade"` or `"blank"`. |
| `MOD_ALPHA_BLANK_THRESH` | Threshold below which cells are faded/blanked. |
| `MOD_EQUAL_ASPECT` | Equal aspect ratio on map/curtain panels. |
| `MOD_DEPTH_KM / HORIZ_KM` | Axis units. |
| `MOD_PANEL_HEIGHT` | Panel height in cm. |
| `MOD_PANEL_WIDTH` | Panel width in cm; `None` = auto from aspect ratio. With `MOD_EQUAL_ASPECT=True` and real `MOD_XLIM`/`MOD_YLIM`/`MOD_ZLIM` (e.g. from `MOD_ROI_AUTO`), map/ns/ew panels end up with genuinely different widths instead of being forced square. |
| `MOD_FIGSIZE` | `[w, h]` cm; overrides auto layout when set. |
| `MOD_NROWS / NCOLS` | Grid layout. Default `2 / 2`, matching the 4 default `MOD_SLICES` panels (2 maps + ns + ew); `None` = 1 row × N columns. Adjust if you change the number of panels. |
| `MOD_TICK_FONTSIZE` | Font size for axis tick labels and colourbar ticks. Default `7`, matching `fviz.plot_model_slices`' own default. |
| `MOD_LABEL_FONTSIZE` | Font size for axis labels, panel titles, and colourbar label. Default `8`, matching `fviz.plot_model_slices`' own default. |

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
| `<P>_cov` | `(N_free, N_free)` | Empirical covariance matrix. Present only if `COMPUTE_COV=True` and `COV_METHOD="full"`. |
| `<P>_cov_eigval` | `(r,)` | Covariance eigenvalues, `r = min(N_members, N_free)`. Present only if `COMPUTE_COV=True` and `COV_METHOD="low_rank"`. |
| `<P>_cov_eigvec` | `(N_free, r)` | Covariance eigenvectors; `C = eigvec @ diag(eigval) @ eigvec.T`. Present only if `COMPUTE_COV=True` and `COV_METHOD="low_rank"`. |
| `<P>_avg` | `(N_free,)` | Element-wise mean over members. |
| `<P>_var` | `(N_free,)` | Element-wise variance over members. |
| `<P>_med` | `(N_free,)` | Element-wise median over members. |
| `<P>_mad` | `(N_free,)` | Median absolute deviation. |
| `<P>_prc` | `(N_prc, N_free)` | Percentile values at `PERCENTILES` levels. |
| `<P>_prc_levels` | `(N_prc,)` | The `PERCENTILES` levels themselves, for self-describing output. |
| `<P>_qdiff_<lo>_<hi>` | `(N_free,)` | `\|P_hi - P_lo\|` per free parameter, one key per `QDIFF_PAIRS` entry (e.g. `<P>_qdiff_15_9_84_1`). |

If `COMPUTE_COV=False`, none of the `<P>_cov*` keys are present.

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
| 2026-07-17 | Claude Sonnet 5 (Anthropic) | `scipy.sparse`: migrated from legacy matrix to array-equivalent API — `scs.csr_matrix(tmp)` → `scs.csr_array(tmp)` when building the sparsified empirical covariance (`ens_covs`). No functional change; `ens_covs` is only used for its `.nnz` count. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `COMPUTE_COV` (skip covariance entirely) and `COV_METHOD="low_rank"` (exact thin-SVD factorisation of the centred ensemble, avoiding the dense `N_free × N_free` matrix — see Covariance section). `MOD_STATS` now also writes a block file and slice figure for each `PERCENTILES` level (`p2_3`, `p50`, `p97_7`, …), included by default in `MOD_STATS_WHAT`. Added `<P>_prc_levels` to the `.npz` output. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `MOD_STATS_CLIM` for per-statistic colour scaling (`VAR`/`MAD`/`QDIFF` default to a fixed `[-2, 2]` range instead of silently reusing `MOD_CLIM`, which made them blank; set a key to `None` for auto per-panel scaling instead). Added `QDIFF_PAIRS` (default `[(15.9, 84.1)]`): percentile-difference spread statistics, saved to the `.npz` and plottable via `MOD_STATS`. Added `MOD_ROI_AUTO`/`MOD_ROI_PAD_XY`/`MOD_ROI_ZLIM`: automatic `MOD_XLIM`/`MOD_YLIM`/`MOD_ZLIM` from the site bounding box, which also activates `femtic_viz.py`'s existing aspect-ratio panel-width logic so map/ns/ew panels size themselves differently. Changed `MOD_NROWS`/`MOD_NCOLS` defaults to `2`/`2`. Also fixed a sign bug in `femtic_viz.py`'s `plot_model_slices` (ns/ew curtain panels rendered blank/upside down whenever `MOD_ZLIM` was set) — the bug was dormant under the old `MOD_ZLIM=None` default and is fixed as part of enabling `MOD_ROI_AUTO`. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `MOD_TICK_FONTSIZE` / `MOD_LABEL_FONTSIZE`, passed through to every `_plot_slice()` call (`MOD_QC` and `MOD_STATS`). Axis tick labels, axis labels, panel titles, and colourbar text were previously fixed at `plot_model_slices`' internal defaults with no way to override them from this script. |
