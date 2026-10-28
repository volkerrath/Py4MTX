# femtic_nss.py — Nullspace Shuttle for FEMTIC Inversion Results

**Py4MTX framework · vrath / Claude Sonnet 4.6 · 2026-05-17 / 2026-06-23**

---

## Purpose

`femtic_nss.py` reads the final model and data from FEMTIC's own results file
(`results_iterX.h5`) and the Jacobian from its exchange file (`exchange.h5`),
decomposes the data-weighted Jacobian via randomised SVD with N singular
values / vectors, generates
a model perturbation, and projects that perturbation onto the **null space** of
the scaled Jacobian so that it leaves the predicted data unchanged.  This is
repeated for `NSS_N_MODELS` perturbations, **chunk by chunk**: each chunk of
`NSS_CHUNK_SIZE` shuttled models is merged into streaming statistics and then
discarded, so the ensemble never has to fit in memory.  The statistics are the
per-block mean and variance of log10(rho) (Welford / Chan pairwise merge), the
skewness and excess kurtosis (third and fourth central moments, same kind of
merge) and the per-block quantiles at the 1-sigma and 2-sigma probabilities
(fixed-bin histograms) with the Bowley quantile skewness derived from them.  The first shuttled model is written as a FEMTIC resistivity
block, together with the mean and quantile models.

Two perturbation modes are available, selected by `PERTURB_MODE`:

- **`"gst"`** *(default)* — geostatistical perturbation via pilot-point Ordinary
  Kriging (`ensembles.generate_gst_model_ensemble`).  Produces spatially coherent
  perturbations whose character is governed by a user-specified variogram model.
- **`"random"`** — uniform Gaussian placeholder (original behaviour).  The helper
  `_make_perturbation_random` can be edited to inject any user-defined prior-based
  perturbation.

The script is modelled on `femtic_mod_edit.py` and follows the same
configuration-block / helper-function pattern.

---

## Background

A 3-D MT inversion has many more model parameters than data.  The inversion
minimises data misfit but cannot constrain the null space N(J) — directions in
model space that produce no change in predicted data.  The **nullspace shuttle**
(Muñoz & Rath, 2006; Deal & Nolet, 1996) exploits this: any perturbation
δm̃ can be split into a data-sensitive component (row space of J) and a
data-invisible component (null space of J).  Adding only the null-space
component to the final model yields an equally valid model with different
resistivity structure.

This is useful for:

- testing how much structural variability is permitted by the data,
- constructing model ensembles for uncertainty visualisation,
- introducing geological constraints that the inversion itself could not recover.

### Mathematics

Let **J**_s = diag(1/σ) **J** be the error-scaled Jacobian (n_d × n_m).
Its truncated randomised SVD is:

```
Js ≈ U  S  Vt        U : (nd, r)   S : (r,)   Vt : (r, nm)
```

The **r_eff** right singular vectors with s_i ≥ τ · s_0 span the row space;
the remaining n_m − r_eff directions span the null space.  The null-space
projector is:

```
P_null = I  −  Vr Vr^T        Vr = Vt[:r_eff].T   (nm, r_eff)
```

The shuttle perturbation added to the model is:

```
δm_null = α · P_null · δm̃
```

where δm̃ comes from `_modify_model` and α = `NSS_AMPLITUDE`.

Verification: `‖Js · δm_null‖` is printed and should be near machine zero.

---

## Workflow

| Step | What happens |
|------|-------------|
| **1** | Read the final model (log10 rho of the free blocks) and misfit data from `results_iterX.h5`; read the Jacobian from `exchange.h5` |
| **2** | `Js = J` (FEMTIC writes the Jacobian already error-weighted; divided by the errors only if `JAC_WEIGHTED = False`); approximate RMS misfit from the results file |
| **3** | Randomised SVD of `Js` with N = `RSVD_RANK` singular values / vectors via `inverse.rsvd` (Halko et al., 2011) |
| **4** | Generate raw perturbation δm̃ via `PERTURB_MODE`: `"gst"` → Kriged model delta; `"random"` → user-editable Gaussian |
| **5** | For each chunk of `NSS_CHUNK_SIZE` models: draw δm̃ (step 4), project onto the null space, add to the model, merge the chunk into the running mean / variance (`chan_merge`) and the per-block histograms (`hist_update`); discard the chunk |
| **out** | Write the first model to `MODEL_OUT`, the mean and quantile models next to it, and all statistics to `NSS_STATS_FILE` |

---

## Input

### Results file - `results_iterX.h5`

Written by FEMTIC when `ACTIVATE_HDF5_RESULTS` is set in `control.dat` (build
with `HDF5_OUT=yes`); read here with `fem.read_results_hdf5`.

| Group / dataset | Used for |
|---|---|
| `/model/blocks` (`resistivity`, `type`) | Final model: log10(rho) of the **free** blocks (`type` not 1 or 2), in ascending blockID order, i.e. the order of the Jacobian columns |
| `/model/metadata` (`iterNum`, `anisotropic`) | Iteration number (compared with the Jacobian's); anisotropic (v5) files are rejected, because the model vector there is not one value per free block |
| `/data/data` (`re_val`, `im_val`, `re_err`, `im_err`, `cal_re`, `cal_im`) | Approximate error-weighted RMS misfit, printed in step 2 |

### Jacobian file - `exchange.h5`

Written by FEMTIC when `ACTIVATE_HDF5_EXCHANGE` is set (build with
`HDF5_JAC=yes`); it also holds the roughening matrix and the mesh.

| Dataset | Shape | Description |
|---------|-------|-------------|
| `/jacobian/values` | (n_d, n_m) | J = Cd^(-1/2) dF/dm, **already divided by the data errors** (`weighted` = 1) |
| `/jacobian/data_errors` | (n_d,) | Standard deviation of each row; used only if `JAC_WEIGHTED = False` |
| `/jacobian/metadata` | attributes | `iterNum`, `nData`, `nModel`, `weighted` |

- Because J is already weighted, `Js = J`.  Dividing by the errors again, as
  the earlier `Inversion_results.h5` version of this script did with its own
  unweighted Jacobian, would weight the data twice.
- The rows follow the inversion's global (PE-gathered) data ordering, **not**
  the row order of `/data/data` in the results file, so rows of J cannot be
  matched to rows of the results table.  The shuttle only needs the model
  (column) space of J, so this does not matter here; it is why the misfit in
  step 2 is computed from the results file alone.
- The columns are the free blocks in ascending blockID order.  The script
  stops if their number differs from the number of free blocks in the results
  file or in `MODEL_TEMPLATE`, and reports a differing iteration number.
- An older `jacobian.h5` (root-level dataset `/jacobian`, optionally
  `/data_errors`) is also accepted.  It carries no `weighted` attribute, so
  set `JAC_WEIGHTED` explicitly for such a file.

### Template resistivity block — `MODEL_TEMPLATE`

Used by `fem.read_model` / `fem.insert_model` to preserve the FEMTIC file
structure (header, region bounds, flag column, fixed regions, air, ocean).
Only the free-region values are replaced.  Any valid FEMTIC resistivity block
(e.g. `resistivity_block_iter0.dat`) works.

---

## Output

| File | Description |
|------|-------------|
| `resistivity_block_nss.dat` | First null-space-shuttled FEMTIC resistivity block (`MODEL_OUT`) |
| `resistivity_block_nss_mean.dat` | Per-block mean of log10(rho) over all shuttled models |
| `resistivity_block_nss_q0228.dat`, `_q1587.dat`, `_q5000.dat`, `_q8413.dat`, `_q9772.dat` | Quantile models at the probabilities Phi(-2), Phi(-1), 0.5, Phi(+1), Phi(+2) of the standard normal distribution (0.0228, 0.1587, 0.5, 0.8413, 0.9772); one file per level of `NSS_SIGMAS` (`NSS_WRITE_STAT_MODELS`) |
| `nss_stats.npz` | All statistics (see below) |
| `nss_models/resistivity_block_nss_0000.dat`, `_0001.dat`, ... | Only with `NSS_EXPORT_MODELS = True`: the first `min(NSS_N_MODELS, NSS_EXPORT_MAX)` shuttled models, one FEMTIC resistivity block each (same template, fixed regions, air and ocean as `MODEL_OUT`). Model 0000 is the same as `MODEL_OUT`. |
| `nss_models.npz` | Only with `NSS_EXPORT_MODELS = True`: `models` (n_export, nm) and `reference_model` (nm,), log10(rho) of the free blocks; row `i` is file `_i` |
| `nss_qc.pdf` | Optional QC slice figure of the output model (`MOD_QC = True`) |

`nss_stats.npz` (`NSS_STATS_FILE`) holds, all for log10(rho) of the free blocks
in ascending blockID order:

| Key | Shape | Content |
|-----|-------|---------|
| `n` | scalar | Number of models accumulated |
| `mean`, `std`, `m2` | (nm,) | Mean, sample standard deviation (ddof = 1), sum of squared deviations |
| `m3`, `m4` | (nm,) | Sums of the cubed and fourth-power deviations from the mean (to merge or continue a run) |
| `skew`, `ex_kurt` | (nm,) | Skewness and excess kurtosis (NaN for fewer than 4 models); `moments_unbiased` says which version |
| `bowley`, `bowley_sigma` | (nk, nm), (nk,) | Bowley skewness `(q(+k) + q(-k) - 2 q50) / (q(+k) - q(-k))` for each positive sigma level k |
| `sigma`, `probs` | (nq,) | Sigma levels (-2, -1, 0, 1, 2 for the default) and their normal-distribution probabilities |
| `quantiles` | (nq, nm) | Empirical quantiles from the histograms, row `i` for `probs[i]` |
| `gauss` | (nq, nm) | `mean + sigma[i] * std` for comparison (equal to the quantiles if the distribution is Gaussian) |
| `hist_lo`, `hist_hi`, `hist_nbins`, `clip_fraction` | scalars | Histogram range, bin count, and the fraction of values that fell outside the range |
| `reference_model` | (nm,) | The final model that the perturbations were added to |
| `r_eff`, `amplitude` | scalars | Effective rank used for the projection and `NSS_AMPLITUDE` |

The `.dat` files are in standard FEMTIC format and can be used directly as the
starting model for a further inversion run or passed to `femtic_mod_edit.py` /
`femtic_viz.py` for inspection.

---

## Plotting config (shared with `femtic_gst_prep.py` / `femtic_ens_post.py`)

Setting `MOD_QC = True` produces a single slice figure of the nullspace-
shuttled output model (`MODEL_OUT`), using the same `MOD_*` config surface --
variable names, defaults, and order -- as `femtic_gst_prep.py` and
`femtic_ens_post.py`.  A config block can be copied between the three
scripts with no renaming.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_MESH` | str | `WORK_DIR + "mesh.dat"` | Tetrahedral mesh -- required for the QC plot. |
| `MOD_OCEAN` | bool / None | `None` | `None` = auto-infer; `True`/`False` forces ocean-present/absent. |
| `MOD_AIR_RHO` | float | `1.0e9` | Ohm.m sentinel for air cells (region 0). |
| `MOD_OCEAN_RHO` | float | `0.25` | Ohm.m sentinel for ocean cells (region 1). |
| `MOD_QC` | bool | `False` | Enable the QC slice plot of `MODEL_OUT`. |
| `MOD_QC_FILE` | str | `WORK_DIR + "nss_qc.pdf"` | Output path. |
| `MOD_DPI` | int | `600` | Figure DPI. |
| `MOD_TICK_FONTSIZE` | int | `7` | Axis tick labels and colourbar ticks (matches `fviz.plot_model_slices`' own default). |
| `MOD_LABEL_FONTSIZE` | int | `8` | Axis labels, panel title, colourbar label (matches `fviz.plot_model_slices`' own default). |
| `MOD_SHOW_IN_SPYDER` | bool | `True` | When running inside Spyder (detected via `utl.runtime_env() == "spyder"`), the `MOD_QC` figure is also displayed inline in Spyder's Plots pane via `plt.show()`, in addition to being written to disk. No effect outside Spyder; set `False` to disable even under Spyder. |

The shared slice/plot, site-overlay, and geographic/UTM-origin parameters
(`MOD_SLICES`, `MOD_XLIM/YLIM/ZLIM`, `MOD_CMAP`, `MOD_CLIM`,
`MOD_OCEAN_COLOR`, `MOD_AIR_COLOR`, `MOD_AIR_BGCOLOR`, `MOD_ALPHA_*`,
`MOD_EQUAL_ASPECT`, `MOD_DEPTH_KM`/`MOD_HORIZ_KM`, `MOD_PANEL_HEIGHT`/
`MOD_PANEL_WIDTH`, `MOD_FIGSIZE`, `MOD_NROWS`/`MOD_NCOLS`, `MOD_SITE_DAT`,
`MOD_SITE_NAMES`, `MOD_SITE_NUMBER`, `MOD_PLOT_SITES_MAPS`/`SLICES`,
`MOD_PROJECTION_DIST`, `MOD_SITE_MARKER`/`MARKER_SLICES`, `MOD_MAP_MARKERS`,
`MOD_ORIGIN_METHOD`, `MOD_UTM_ORIGIN_LAT/LON/E/N`, `MOD_UTM_ZONE_OVERRIDE`,
`MOD_DISPLAY_COORDS`) have identical names, defaults, and semantics to
`femtic_ens_post.py`'s "Shared slice / plot parameters" -- see that README
for the full parameter table.

---

## Configuration

All user-facing parameters are collected in the **Configuration** block near
the top of the script.  No other edits are needed for a standard run.

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | `/home/vrath/Py4MTX/work/` | Directory for all I/O |
| `RESULTS_FILE` | `WORK_DIR + "results_iter10.h5"` | FEMTIC results file of the iteration the Jacobian belongs to (set the iteration number) |
| `JACOBIAN_FILE` | `WORK_DIR + "exchange.h5"` | FEMTIC exchange file with the Jacobian (or a legacy `jacobian.h5`) |
| `JAC_WEIGHTED` | `None` | Is J already divided by the data errors?  `None` = read the file's `weighted` attribute (exchange.h5: True; a legacy file without it: error); `True` = `Js = J`; `False` = `Js = diag(1/error) @ J` |
| `MODEL_TEMPLATE` | `WORK_DIR + "resistivity_block_iter0.dat"` | FEMTIC template block |
| `MODEL_OUT` | `WORK_DIR + "resistivity_block_nss.dat"` | Output block |
| `RANDOM_SEED` | `None` | `None` = fresh OS entropy (default, non-reproducible). An integer makes step 4's model perturbation reproducible: a shared `rng = np.random.default_rng(RANDOM_SEED)` drives both `PERTURB_MODE = "gst"` (passed to `ens.generate_gst_model_ensemble`) and `PERTURB_MODE = "random"` (used directly in `_make_perturbation_random`). **Does not** cover step 3's randomised SVD (`inv.rsvd`) — see the caveat below. |

### Perturbation mode  (step 4)

| Variable | Default | Description |
|----------|---------|-------------|
| `PERTURB_MODE` | `"gst"` | `"gst"` — pilot-point Kriging perturbation; `"random"` — Gaussian placeholder |

### GST model perturbation  (used when `PERTURB_MODE == "gst"`)

The GST block mirrors the equivalent config in `femtic_gst_prep.py`.

**Paths**

| Variable | Default | Description |
|----------|---------|-------------|
| `GST_REF_MOD` | `MODEL_TEMPLATE` | Reference model providing free-region structure |
| `GST_MESH` | `WORK_DIR + "mesh.dat"` | FEMTIC mesh file for barycentre computation |

**Pilot-point placement**

| Variable | Default | Description |
|----------|---------|-------------|
| `GST_PP_MODE` | `"random"` | Placement strategy: `"random"` / `"fixed"` / `"mixed"` / `"extrema"` |
| `GST_N_PP` | `100` | Number of random pilot points per realisation (used in `"random"`, `"mixed"`, `"extrema"`) |
| `GST_PP_N_RESET` | `None` | Reset interval of the random pilot-point layout, in models. `None` = N_e = `NSS_N_MODELS` (one layout for the whole run); `1` = fresh layout for every model; `n` = new layout every `n` models. Pilot-point **values** are redrawn for every model in any case. No effect for `GST_PP_MODE = "fixed"` or `PERTURB_MODE = "random"`. See below. |
| `GST_PP_BBOX` | `[-50000, 50000, -50000, 50000, 0, 80000]` | Bounding box [x_min, x_max, y_min, y_max, z_min, z_max] (m, z positive-down) |
| `GST_PP_COORDS` | `None` | Explicit (N, 3) array of pilot-point locations for `"fixed"` / `"mixed"` mode |
| `GST_PP_ROI` | `None` | Sub-volume for extremum search in `"extrema"` mode; `None` = full extent |
| `GST_PP_EXTREMA_K` | `9` | Neighbourhood size for local extremum test in `"extrema"` mode |
| `GST_PP_EXTREMA_WHICH` | `"both"` | Which extrema to seed: `"both"` / `"minima"` / `"maxima"` |

**`GST_PP_N_RESET`.** The random pilot-point *locations* (the whole point set
for `"random"`, the random-fill part for `"mixed"` / `"extrema"`) are drawn once per
block of `GST_PP_N_RESET` consecutive models and shared by the models in
that block; the pilot-point *values* are drawn anew for every model. With the
default (`None` = `NSS_N_MODELS`) the whole run uses one layout, so the
ensemble spread comes from value randomness at fixed geometry; with `1` every
model has its own layout. Counting is over the whole run, so it does not depend
on `NSS_CHUNK_SIZE`. The default differs from `MOD_PP_REGEN_EVERY` in
`femtic_gst_prep.py`, where `None` means a fresh layout for every member.
Values `< 1` raise `ValueError`; values larger than `NSS_N_MODELS` act like
`None`.

**Resistivity range**

| Variable | Default | Description |
|----------|---------|-------------|
| `GST_LOG_RHO_MIN` | `0.0` | Lower draw bound and post-Kriging clamp (log₁₀ Ω·m) |
| `GST_LOG_RHO_MAX` | `4.0` | Upper draw bound and post-Kriging clamp (log₁₀ Ω·m) |

**Variogram model**

| Variable | Default | Description |
|----------|---------|-------------|
| `GST_VARIO_MODEL` | `"Spherical"` | gstools covariance model class: `"Spherical"`, `"Gaussian"`, `"Exponential"`, `"Matern"`, … |
| `GST_VARIO_RANGE` | `(8000., 4000.)` | Correlation length(s) in m.  Scalar = isotropic; 2-tuple `(h_range, v_range)` = anisotropic |
| `GST_VARIO_SILL` | `0.5` | Sill (variance) in (log₁₀ Ω·m)² |
| `GST_VARIO_NUGGET` | `0.01` | Nugget in (log₁₀ Ω·m)²; keep ≤ 10 % of sill |
| `GST_VARIO_ANGLES` | `None` | Rotation [α, β, γ] in degrees; `None` = axis-aligned |

### Randomised SVD  (step 3)

| Variable | Default | Description |
|----------|---------|-------------|
| `RSVD_RANK` | `300` | N: number of singular values / vectors computed by the rSVD of Js (target rank). Increase until the singular-value spectrum flattens. Must be << min(n_d, n_m). |
| `RSVD_OVERSAMPLES` | `None` | Extra random columns. `None` → 2 × `RSVD_RANK` (Halko et al. default). |
| `RSVD_SUBSPACE_ITERS` | `2` | Power iterations for improved accuracy. More iterations = better approximation at higher cost. |

### Nullspace shuttle  (step 5)

| Variable | Default | Description |
|----------|---------|-------------|
| `NSS_SV_THRESH` | `1.0e-3` | Singular-value cutoff as a fraction of s₀. Vectors with s_i < τ · s₀ are treated as null-space. Smaller τ → larger effective rank → less null-space freedom. |
| `NSS_AMPLITUDE` | `1.0` | Scale factor α applied to δm_null before adding to the model. Start at 0.1 and increase to explore the null space more aggressively. |

### Ensemble statistics  (step 5)

| Variable | Default | Description |
|----------|---------|-------------|
| `NSS_N_MODELS` | `100` | Number of shuttled models. In `"gst"` mode each costs one Kriging run, which usually dominates the run time. |
| `NSS_CHUNK_SIZE` | `25` | Models generated, projected and merged at a time. Memory is about `NSS_CHUNK_SIZE * nm * 8` bytes for the chunk plus a few temporaries of that size. |
| `NSS_SIGMAS` | `(1.0, 2.0)` | Sigma levels. Quantiles are computed at Phi(-k), 0.5 and Phi(+k) of the standard normal distribution for each k (0.0228, 0.1587, 0.5, 0.8413, 0.9772). Use 1.96 for 2.5 / 97.5 %. |
| `NSS_HIST_NBINS` | `256` | Histogram bins per block. Memory is `nm * NSS_HIST_NBINS * 4` bytes; the quantile error is at most one bin width. |
| `NSS_HIST_RANGE` | `None` | `(lo, hi)` in log10(rho) common to all blocks. `None`: from the first chunk (min and max padded by half its span, at least 0.5 each side). Values outside go into the end bins. |
| `NSS_WRITE_STAT_MODELS` | `True` | Write the mean and quantile models as resistivity blocks. |
| `NSS_STATS_FILE` | `WORK_DIR + "nss_stats.npz"` | Archive with all statistics. |
| `NSS_EXPORT_MODELS` | `False` | Export the individual shuttled models, for testing with a small ensemble. |
| `NSS_EXPORT_MAX` | `100` | At most this many models are exported (the first ones); the statistics always use all `NSS_N_MODELS`. |
| `NSS_EXPORT_DIR` | `WORK_DIR + "nss_models/"` | Directory for the exported resistivity blocks (created if missing). |
| `NSS_EXPORT_NPZ` | `WORK_DIR + "nss_models.npz"` | All exported models as one array. |
| `NSS_MOMENTS_UNBIASED` | `False` | `False`: moment coefficients `g1 = sqrt(n) M3 / M2^1.5`, `g2 = n M4 / M2^2 - 3`. `True`: small-sample corrected `G1`, `G2`. Matters only below about 50 models. |

### Diagnostics

| Variable | Default | Description |
|----------|---------|-------------|
| `OUT` | `True` | Print shapes, norms, and verification metrics at each step. |

---

## Model Perturbation Functions  (step 4)

### `_make_perturbation_random(m)`

```python
def _make_perturbation_random(m: np.ndarray) -> np.ndarray:
    ...
```

Used when `PERTURB_MODE = "random"`.  Receives the final model vector `m`
(log₁₀(ρ), free regions only, shape `(n_m,)`) and returns a perturbation `dm`
of the same shape.  The perturbation is projected onto the null space of Js, so
any data-sensitive component is automatically removed before the result is added
to the model.

The default placeholder returns a unit-Gaussian random vector drawn from
the shared, module-level `rng` (see `RANDOM_SEED` above) — fresh OS entropy
by default, or a reproducible draw when `RANDOM_SEED` is set. (Prior to
2026-07-25 this was hardcoded to `np.random.default_rng(seed=0)`, always
the same draw regardless of any seed configured elsewhere — not
reproducible-by-choice, just permanently fixed.) Replace the body between
the `*** EDIT ***` markers:

```python
# Push a target region towards higher resistivity
dm = np.zeros_like(m)
dm[region_indices] = 1.0          # +1 log unit in target region

# Uniform bias across the whole model
dm = np.full_like(m, 0.5)         # +0.5 log units everywhere

# Random ensemble member, respecting RANDOM_SEED (just use the shared rng)
dm = rng.standard_normal(m.size) * 0.2

# Or an independent, explicitly-seeded draw if you want this placeholder
# to vary independently of RANDOM_SEED:
rng_local = np.random.default_rng(seed=42)
dm = rng_local.standard_normal(m.size) * 0.2
```

### `_make_perturbation_gst(m_ref, idx=0)`

```python
def _make_perturbation_gst(m_ref: np.ndarray, idx: int = 0) -> np.ndarray:
    ...
```

Used when `PERTURB_MODE = "gst"`.  Calls
`ens.generate_gst_model_ensemble()` for a single realisation in a temporary
subdirectory under `WORK_DIR`, passing the shared, module-level `rng` (see
`RANDOM_SEED` above) so the draw is reproducible when `RANDOM_SEED` is set.
(Prior to 2026-07-25 this constructed a fresh `np.random.default_rng()`
inline on every call — never reproducible, regardless of any seed
configured elsewhere.)  The Kriged resistivity block is read back,
the reference-model vector is subtracted, and the delta is returned:

```
dm = m_gst − m_ref       (log₁₀ Ω·m, free regions only)
```

`idx` is the global (0-based) index of the model over the whole run, not within
a chunk (`_draw_perturbations(k, i0)` passes `i0 + j`).  Together with the
module-level cache `_gst_pp_cache` it lets `ens.generate_gst_model_ensemble`
keep the `GST_PP_N_RESET` block structure although it is called with
`n_samples=1` once per model (`pp_regen_every=GST_PP_N_RESET`,
`pp_cache=_gst_pp_cache`, `pp_pos_offset=idx`; needs the 2026-10-23
`ensembles.py`).  In `"fixed"` mode none of these is passed.

Before each call the temporary member directory `_nss_gst_tmp_0/` is created
and `GST_REF_MOD` is copied into it as `referencemodel.dat` (a copy, not a
link), because `generate_gst_model_ensemble` reads its `insert_model` template
from there.  Earlier versions did neither and could not have run in `"gst"`
mode (see the provenance table).

The temporary directory is cleaned up regardless of success or failure.  The
spatial character of `dm` (correlation lengths, variance) is governed entirely
by the `GST_VARIO_*` settings.

---

## Nullspace Shuttle — `_nullspace_shuttle`

```python
def _nullspace_shuttle(dm, Vt, S, *, sv_thresh, amplitude) -> (dm_null, r_eff)
```

`dm` is one vector of shape `(nm,)` or a chunk of shape `(n_models, nm)`;
the chunk is projected row by row with one matrix product.

Internal helper.  Not intended for editing.

Computes the effective rank `r_eff` from `S` and `sv_thresh`, forms the
row-space basis `Vr = Vt[:r_eff].T`, and returns:

```
dm_null = amplitude * (dm  −  (dm @ Vr) @ Vr.T)
```

Also returns `r_eff` for diagnostic printing.

---

## Printed diagnostics

Example output (illustrative values):

```
Step 1: Reading FEMTIC results (model, data) and Jacobian
  model (free blocks) : (12345,)  (log10 rho)
  Jacobian            : (4800, 12345)  [exchange.h5, already error-weighted, used as is]
  iteration           : results 20, Jacobian 20
  nd=4800, nm=12345

Step 2: Data-weighted Jacobian Js and misfit summary
  Js shape : (4800, 12345)
  ||Js||_F : 5.1234e+03
  RMS (approx., from results file) : 1.0003  (9580 of 9600 real/imag entries used)

Step 3: Randomised SVD of Js
  Decomposition: U (4800, 300), S (300,), Vt (300, 12345)
  s[0]  = 3.4712e+02  (largest)
  s[-1] = 1.8843e-01  (smallest in truncated set)
  Effective rank at threshold 1.0e-03: 247 / 300

Step 5: Nullspace shuttle ensemble, 1000 models in chunks of 25  [PERTURB_MODE = 'gst']
  Effective rank used for projection : 247
  mean ||dm_row|| (first chunk)       : 1.9281e+01
  max ||Js @ dm_null|| (should be ~0) : 3.4e-12
  histogram range [-3.120, 7.310] log10(rho), 256 bins, bin width 0.0407, 12 MB
  models      1-25     of 1000: 78.20 s
  ...
  Models accumulated : 1000
  model_nss (first) range : [-0.124, 4.231]
  std (log10 rho) over blocks : median 0.1870, max 0.4410
  -2.0 sigma  p = 0.0228  quantile vs Gaussian mean +/- k*std: median |diff| 0.0310
  -1.0 sigma  p = 0.1587  quantile vs Gaussian mean +/- k*std: median |diff| 0.0190
  +0.0 sigma  p = 0.5000  quantile vs Gaussian mean +/- k*std: median |diff| 0.0090
  +1.0 sigma  p = 0.8413  quantile vs Gaussian mean +/- k*std: median |diff| 0.0190
  +2.0 sigma  p = 0.9772  quantile vs Gaussian mean +/- k*std: median |diff| 0.0310
  skewness g1 : median +0.020, 5-95 % of blocks [-0.610, +0.720]
  excess kurtosis g2 : median +0.150, 5-95 % of blocks [-0.700, +1.900]
  Bowley skewness (1.0 sigma quantiles) : median +0.030, median |.| 0.110
  Bowley skewness (2.0 sigma quantiles) : median +0.030, median |.| 0.100
  blocks with |g1| > 2*sqrt(6/n) = 0.126 : 35.0 % (Gaussian, independent samples; the models are correlated)
  values outside histogram range : 0.0000 %

Writing nullspace-shuttled model
  Written : /home/vrath/Py4MTX/work/resistivity_block_nss.dat  (first of 1000 shuttled models)
  Written : /home/vrath/Py4MTX/work/resistivity_block_nss_mean.dat
  Written : /home/vrath/Py4MTX/work/resistivity_block_nss_q0228.dat
  ...
  Written : /home/vrath/Py4MTX/work/nss_stats.npz
```

The key verification line is `‖Js · δm_null‖ ≈ 0`.  Values above ~ 10⁻⁶
indicate that `RSVD_RANK` is too small or `NSS_SV_THRESH` is too large.

---

## Dependencies

| Package | Role |
|---------|------|
| `numpy` | Array operations, norm computations |
| `h5py` | Reading `exchange.h5` / `jacobian.h5` (and, via `femtic`, `results_iterX.h5`) |
| `inverse` (Py4MTX) | `rsvd`, `find_range`, `subspace_iter`, `ortho_basis` |
| `femtic` (Py4MTX) | `read_results_hdf5`, `read_model`, `insert_model` |
| `ensembles` (Py4MTX) | `batch_moments4`, `merge_moments4`, `moments_variance`, `skew_kurtosis`, `quantile_skewness`, `hist_update`, `hist_quantiles` (needs the 2026-10-17 version); `generate_gst_model_ensemble` (GST mode only) |
| `util` (Py4MTX) | `print_title` |
| `version` (Py4MTX) | `versionstrg` |
| `gstools` | Variogram / Ordinary Kriging (via `ensembles`, GST mode only) |

---

## Tuning guide

**How large should `RSVD_RANK` be?**
Plot the singular-value spectrum of Js (values in `S` after step 3).  The
rank should cover the knee of the spectrum — where values transition from
large (data-sensitive) to small (effectively null).  Starting with 300 is
reasonable for typical FEMTIC problems; reduce if memory is tight, increase if
`r_eff / rank` is close to 1 (spectrum not yet fully captured).

**How to choose `NSS_SV_THRESH`?**
Values between 1 × 10⁻³ and 1 × 10⁻² are typical.  A smaller threshold
includes more singular vectors in the row space, leaving less null-space
freedom but producing a cleaner projection.  Watch the `r_eff` printed in
step 3 and step 5: if `r_eff` equals `RSVD_RANK`, lower the threshold or
increase the rank.

**How to choose `NSS_HIST_NBINS` and `NSS_HIST_RANGE`?**
The quantiles are accurate to one bin width, `(hi - lo) / NSS_HIST_NBINS`;
0.04 in log10(rho) (about 10 % in resistivity) is fine for plots, use more bins
if you need finer values. Check `values outside histogram range`: more than
0.1 % triggers a warning, and the affected quantiles are then only bounds. The
automatic range comes from the first chunk, so set `NSS_HIST_RANGE` explicitly
(e.g. `(-2.0, 6.0)`) when `NSS_AMPLITUDE` is large or the models are
heterogeneous.

**Testing with individual models.**
Set `NSS_EXPORT_MODELS = True` with a small `NSS_N_MODELS` (about 100). Every
model is written as a resistivity block, so it can be run through FEMTIC
(forward only, to confirm that the data fit is unchanged), plotted, or
compared with the mean and quantile models. `||Js @ dm_null||` printed in step
5 only checks the linearised data change; a forward run checks the full
non-linear response. For large runs leave it off: the files are as large as the
template and 100 000 models would not be useful.

**How many models are enough?**
The 1-sigma quantiles stabilise with a few hundred models, the 2-sigma
quantiles need a few thousand, and skewness and kurtosis more still: for
independent Gaussian samples their standard errors are about `sqrt(6/n)` and
`sqrt(24/n)`, i.e. 0.08 and 0.15 for n = 1000, and the correlation between the
shuttled models makes them larger. The Bowley skewness from the quantiles is
steadier and less sensitive to outlying models; if `g1` is large where the
Bowley value is small, a few models dominate the tail.  Compare the quantile models with the Gaussian
bands (`mean +/- k*std`) and with a run using different `RANDOM_SEED`. The
models share one reference model and one projector, so the quantiles describe
the null-space variability under the chosen perturbation prior (`PERTURB_MODE`,
variogram), not a posterior in the Bayesian sense.

**How to choose `NSS_AMPLITUDE`?**
Start at 0.1.  Increase to 1.0 or beyond to produce more distinct
null-space members.  Values ≫ 1 may push log₁₀(ρ) outside physically
plausible bounds; clip with `femtic_mod_edit.py` (operation `"clip"`) if
needed.

---

## Reproducibility

`RANDOM_SEED` (default `None`) covers **step 4 only** — the model
perturbation. Set it to an integer and both `PERTURB_MODE = "gst"` and
`PERTURB_MODE = "random"` become reproducible, driven by a single shared
`rng = np.random.default_rng(RANDOM_SEED)` constructed once at the top of
the script.  The models are drawn one after another, so a given seed gives the
same ensemble for any `NSS_CHUNK_SIZE`.  This includes the pilot-point layouts
of `GST_PP_N_RESET`: the blocks are counted over the whole run and the layout
is drawn from the same shared `rng`.  Changing `GST_PP_N_RESET` changes the
random stream, so it changes the ensemble for a given seed.

**Not covered:** step 3's randomised SVD of `Js` (`inv.rsvd`, Halko et al.
2011) uses its own internal randomness for the range-finder projection.
`inverse.py` is a separate module; whether `rsvd()` exposes an `rng`/`seed`
parameter wasn't verified here, so `RANDOM_SEED` does not currently reach
it. If you need bit-for-bit-identical `U`/`S`/`Vt` (not just an
identical perturbation `dm`) across runs, check `inv.rsvd`'s signature
and wire `RANDOM_SEED` through there as well.

---

## Provenance

| Date | Author | Change |
|------|--------|--------|
| 2026-05-17 | vrath / Claude Sonnet 4.6 | Created, modelled on `femtic_mod_edit.py` |
| 2026-06-23 | vrath / Claude Sonnet 4.6 | Merged GST model generation from `femtic_gst_prep.py`. Added `PERTURB_MODE` switch; `"gst"` path calls `ens.generate_gst_model_ensemble` for a single realisation and returns `m_gst − m_ref` as perturbation delta. Full GST config block (`GST_PP_*`, `GST_VARIO_*`). `"random"` retains original Gaussian placeholder in `_make_perturbation_random`. Added `ensembles` import. |
| 2026-07-09 | vrath / Claude Sonnet 5 (Anthropic) | Added the shared `MOD_*` plotting config block, `femtic_viz` import, `_resolve_origin_and_sites()` / `_plot_slice()` helpers, and an optional QC slice plot of `MODEL_OUT` (`MOD_QC = True`). Config surface is identical in name and order to `femtic_ens_post.py` and `femtic_gst_prep.py`, so a block can now be copied between all three scripts with no renaming. Uses a single `MOD_DPI` knob, matching `femtic_gst_prep.py` and the now-simplified `femtic_ens_post.py`. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `RANDOM_SEED` for optional reproducible model perturbations (step 4). Fixed two reproducibility bugs: `_make_perturbation_gst` was constructing a fresh, unseeded `np.random.default_rng()` inline on every call; `_make_perturbation_random` was hardcoded to `np.random.default_rng(seed=0)` regardless of any seed elsewhere. Both now use the shared, module-level `rng`. Step 3's `inv.rsvd` randomness is explicitly *not* covered — see Reproducibility section. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `MOD_TICK_FONTSIZE` / `MOD_LABEL_FONTSIZE`, passed through to the `MOD_QC` plot. Axis tick labels, axis labels, panel title, and colourbar text were previously fixed at `plot_model_slices`' internal defaults with no way to override them from this script. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `MOD_SHOW_IN_SPYDER` (default `True`): when running inside Spyder, the `MOD_QC` figure is also displayed inline via `plt.show()`, without changing what gets saved to disk. No effect outside Spyder. |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_nss_summary.md` output at end of run: writes user-set (UPPERCASE) parameters, script path, and run date/time. |
| 2026-10-11 | Claude Sonnet 5.5 (Anthropic) | Model I/O adapted to the current `femtic.py`: the start model is read with `fem.read_model(..., out=False)`, which returns the free-region log10(rho) vector directly (the former `read_model` + `extract_model` pair is gone), and the result is written with `fem.insert_model(template=..., model=..., model_file=..., out=...)`, which writes the file itself. |
| 2026-10-16 | Claude Sonnet 5.5 (Anthropic) | Steps 1-2 read FEMTIC's own HDF5 output: final model (log10 rho of the free blocks) from `results_iterX.h5` (`RESULTS_FILE`), Jacobian from `exchange.h5` (`JACOBIAN_FILE`; legacy `jacobian.h5` also accepted) instead of the ad-hoc `Inversion_results.h5` (`HDF5_FILE` removed). The exchange.h5 Jacobian is already SD-weighted, J = Cd^(-1/2) dF/dm, so `Js = J` (no second division by the errors; `JAC_WEIGHTED` overrides the file attribute). Jacobian rows follow the inversion's data ordering, not `/data/data`, so the step-2 RMS comes from the results file alone. Column count checked against the free blocks of the results file and `MODEL_TEMPLATE`; differing iteration numbers are reported; anisotropic (v5) results are rejected. Step 3 (rSVD with N = `RSVD_RANK` vectors) and the shuttle are unchanged. Verified with an in-memory h5py mock and a synthetic rank-15 Jacobian (`||Js dm_null||` about 3e-13); not run against a real FEMTIC file. AI-generated; review before production use. |
| 2026-10-16 | Claude Sonnet 5.5 (Anthropic) | Chunked ensemble statistics in step 5: `NSS_N_MODELS` shuttled models in chunks of `NSS_CHUNK_SIZE`, merged into streaming per-block mean / variance (`ensembles.chan_merge`, Chan, Golub and LeVeque 1983) and histogram quantiles at the +/-1 and +/-2 sigma probabilities (`ensembles.hist_update` / `hist_quantiles`); Gaussian bands for comparison. `_nullspace_shuttle` accepts a chunk. First model still goes to `MODEL_OUT`; mean and quantile models and `nss_stats.npz` are new outputs. Tested with a synthetic rank-15 Jacobian (mean and std agree with numpy to rounding, quantiles within one bin width of numpy's inverted-CDF quantiles, `||Js dm_null||` about 1e-12); the GST mode, `fem.insert_model` and a real FEMTIC run were not exercised. AI-generated; review before production use. |
| 2026-10-17 | Claude Sonnet 5.5 (Anthropic) | Optional export of the individual shuttled models for testing: `NSS_EXPORT_MODELS` (default `False`), `NSS_EXPORT_MAX` (100), `NSS_EXPORT_DIR`, `NSS_EXPORT_NPZ`. The first `min(NSS_N_MODELS, NSS_EXPORT_MAX)` models are written as resistivity blocks inside the chunk loop plus one `.npz` array; statistics and memory use are unchanged. Tested with the synthetic setup (230 models, limit 100; 60 models, limit 100; export off): file counts and `.npz` contents match the generated models; `fem.insert_model` was a stub. AI-generated; review before production use. |
| 2026-10-17 | Claude Sonnet 5.5 (Anthropic) | Higher moments: the streaming accumulator also carries M3 and M4 (`ensembles.batch_moments4` / `merge_moments4`, Pebay 2008; Terriberry 2007; Pebay et al. 2016). Per-block skewness and excess kurtosis (`NSS_MOMENTS_UNBIASED` selects the bias-corrected versions) and the Bowley quantile skewness for each sigma level are printed and written to `nss_stats.npz` (`skew`, `ex_kurt`, `m3`, `m4`, `bowley`, `bowley_sigma`). Needs the 2026-10-17 `ensembles.py`. Tested with the synthetic setup (230, 6, 3 and 2 models): skewness and kurtosis agree with `scipy.stats` to 1e-14, Bowley values agree with the quantile formula, models with fewer than 4 members give NaN; not run on real FEMTIC data. AI-generated; review before production use. |
| 2026-10-23 | Claude Sonnet 5.5 (Anthropic) | Configurable reset of the random pilot-point layout in the `"gst"` mode: new `GST_PP_N_RESET` (default `None` = N_e = `NSS_N_MODELS`; `1` = new layout per model; `n` = every `n` models), mirroring `MOD_PP_REGEN_EVERY` of `femtic_gst_prep.py` (which needed no change) but with a different default. Implemented through the new `pp_cache` / `pp_pos_offset` arguments of `ens.generate_gst_model_ensemble` (needs the 2026-10-23 `ensembles.py`), because the script calls it with `n_samples=1` once per model; `_draw_perturbations(k, i0)` and `_make_perturbation_gst(m_ref, idx)` carry the global model index. **Bug fixed:** `_make_perturbation_gst` never created `_nss_gst_tmp_0/` nor copied the template into it, so `generate_gst_model_ensemble` could not find its `insert_model` template; both are now done before the call. Tested with extracted helpers on a synthetic mesh and a stand-in for gstools (Kriging replaced by inverse-distance weighting, gstools could not be installed): layouts per block for `n_reset` = `None`, 1, 3 and > N_e, identical ensembles for chunk sizes 1, 2, 7 and 100, `"fixed"` and `"random"` modes, invalid values rejected, and the uploaded version failing for want of the template. The `"gst"` mode has **not** been run with real gstools, a real mesh or a FEMTIC file. AI-generated; review before production use. |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | The parameter summary `femtic_nss_summary.md` is now written to `WORK_DIR` (the working directory set in the script) instead of next to the script: `utl.write_param_summary(fname, out_dir=WORK_DIR)`. Needs the `util.py` of the same date. AI-generated; review before production use. |

**How to configure the GST variogram?**
The variogram controls the spatial coherence of the Kriged perturbation.
`GST_VARIO_RANGE` should be set to plausible geological correlation lengths —
roughly half the survey aperture horizontally and half the target depth
vertically.  `GST_VARIO_SILL` controls the overall variance of the field in
(log₁₀ Ω·m)²; values of 0.25–0.5 correspond to ±0.5–0.7 log units (1σ).
Keep `GST_VARIO_NUGGET` ≤ 10 % of the sill to maintain spatial coherence.

**How many pilot points?**
`GST_N_PP = 100` is a good starting point for typical 3-D MT survey volumes.
Too few (< 30) may leave large voids in the Kriging interpolation; too many
(> 500) slow down the Kriging without improving the spatial representation.

---



- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding structure with
  randomness: Probabilistic algorithms for constructing approximate matrix
  decompositions. *SIAM Review*, 53(2), 217–288.
- Deal, M. M., & Nolet, G. (1996). Nullspace shuttles. *Geophysical Journal
  International*, 124(2), 372–380.
- Muñoz, G., & Rath, V. (2006). Beyond smooth inversion: the use of nullspace
  projection for the exploration of non-uniqueness in MT. *Geophysical Journal
  International*, 164(2), 301–311.
- Suzuki, K., et al. (2025). Geostatistical initial-model ensemble for
  magnetotelluric uncertainty quantification. [full reference to be completed]
