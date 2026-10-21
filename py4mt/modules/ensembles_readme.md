# README\_ensembles.md

Module `ensembles.py` — roughness/precision matrix tools and ensemble generation for FEMTIC workflows.

---

## Overview

`ensembles.py` is the **single source of truth** for all matrix, roughness, and sampling utilities shared across the FEMTIC Python ecosystem. `femtic.py` imports these functions from here rather than duplicating them.

This module provides:

- **Matrix / roughness tools** — read `roughening_matrix.out`, build sparse `R`,
  construct and diagnose prior covariance proxies, sparsify dense matrices.
- **Prior covariance** (`make_prior_cov`) via approximate sparse inversion.
- **Precision solvers** — build callable `solve_Q(b)` for `Q = R^T R + λI`
  via CG, BiCGSTAB, Cholesky, or ILU.
- **Exact Gaussian prior sampling for RTO** (`sample_rto_prior`,
  alpha-aware, sparse LU or Jacobi-CG) plus the older full-rank sampler
  `sample_rtr_full_rank` (fixed 2026-10-21). The randomized-SVD sampler
  `sample_rtr_low_rank` is **deprecated** (it keeps the wrong end of the
  spectrum; see "RTO prior perturbation" below).
- **File-system helpers** — create per-member ensemble directories, copy
  template FEMTIC input files.
- **Data and model ensemble generators** — perturb `observe.dat` and
  `resistivity_block_iterX.dat` for RTO and related algorithms.
- **Geostatistical initial-model ensemble** (`generate_gst_model_ensemble`) —
  pilot-point Ordinary Kriging (gstools) for the GST uncertainty method.
- **Streaming mean / variance / skewness / kurtosis / quantiles**
  (`welford_update`, `batch_moments`, `chan_merge`, `moments_variance`,
  `batch_moments4`, `merge_moments4`, `skew_kurtosis`, `quantile_skewness`,
  `hist_update`, `hist_quantiles`) for ensembles too large to hold in memory;
  partial results merge across chunks or MPI ranks.
- **EOF / PCA analysis** (`compute_eofs`, `eof_reconstruct`, `eof_sample`,
  `fit_eof_model`, `sample_physical_ensemble`) for ensemble dimension reduction.
- **Weighted Karhunen–Loève decomposition** on unstructured FEMTIC meshes
  (`compute_weighted_kl`).
- **Polynomial Chaos Expansion (PCE) surrogate** for log-resistivity fields
  (`fit_kl_pce_model`, `evaluate_kl_pce_surrogate`, `KLPCEModel`).

All functions are importable; no code is executed on import.

---

## Automatic λ selection

Many roughness operators have a **nullspace** (e.g. constants), making
`R^T R` singular or poorly conditioned.  The module supports an optional
data-driven diagonal shift:

```
λ = α · median(diag(R^T R)),    α ≈ 1e-6 … 1e-3
```

### API

Available in `make_precision_solver()` and propagated to sampling helpers:

| Parameter       | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `lam_mode`      | `"fixed"` (default), `"scaled_median_diag"`, or `"auto"`.       |
| `lam_alpha`     | Scale factor α (recommended `1e-6 … 1e-3`).                     |
| `lam_statistic` | Currently `"median"` (future-proof hook).                        |
| `lam_min`       | Lower bound enforced on the computed λ.                          |

Convenience helper: `pick_lam_from_rtr_diag(R, alpha=..., statistic="median", min_lam=...)`.

---

## Core functions

### Directory and ensemble generation

| Function                         | Purpose                                               |
|----------------------------------|-------------------------------------------------------|
| `generate_directories()`         | Create numbered member directories, copy templates.   |
| `generate_data_ensemble()`       | Perturb `observe.dat` in each member directory. Accepts `rng` (forwarded to `femtic.modify_data`) for reproducible data perturbations. |
| `generate_rto_model_ensemble()`  | RTO: perturb the reference model with exact draws from `N(0, (alpha R^T R)^-1)`; **`alpha` is required**. |
| `generate_gst_model_ensemble()`  | Pilot-point Ordinary Kriging → geostatistical initial models (GST). |

### Roughness / precision matrix

| Function              | Purpose                                                    |
|-----------------------|------------------------------------------------------------|
| `get_roughness()`     | Parse `roughening_matrix.out` → sparse `R`.                |
| `make_prior_cov()`    | Approximate `M ≈ (R + εI)^{-1} (R + εI)^{-T}` proxy.     |
| `matrix_reduce()`     | Sparsify a dense or sparse matrix by dropping small entries. |
| `check_sparse_matrix()` | Print shape, nnz, symmetry and value-range diagnostics.  |
| `save_spilu()` / `load_spilu()` | Persist / restore ILU decomposition to/from NPZ. |

### Precision solvers and sampling

| Function                             | Purpose                                             |
|--------------------------------------|-----------------------------------------------------|
| `build_rtr_operator()`               | Matrix-free `Q = R^T R + λ I` LinearOperator.       |
| `make_rtr_preconditioner()`          | Jacobi / ILU / AMG preconditioner for CG.           |
| `make_precision_solver()`            | Build a callable `solve_Q(b)` for `Q x = b`.        |
| `make_sparse_cholesky_precision_solver()` | Direct Cholesky (CHOLMOD) or LU solver.        |
| `sample_rto_prior()`                 | **Exact** samples from `N(0, (alpha (R^T R + lam I))^-1)` (sparse LU or Jacobi-CG). Used by `generate_rto_model_ensemble`. |
| `prior_energy_ratio()`               | Diagnostic `mean(alpha*|R x|^2)/rank(R)`; ~1 for correct draws. |
| `sample_rtr_full_rank()`             | Sample `N(0, Q^{-1})` with a user-supplied solver (fixed 2026-10-21: missing `sqrt(lam)*xi2` term added). |
| `sample_rtr_low_rank()`              | **DEPRECATED** - randomized SVD of R keeps the roughest modes; warns when called. |
| `sample_prior_from_roughness()`      | Unified interface (full or low-rank); the low-rank branch is deprecated. |

### Example — iterative sampling with automatic λ

```python
from ensembles import make_precision_solver

solve_Q = make_precision_solver(
    R,
    lam=0.0,
    lam_mode="scaled_median_diag",
    lam_alpha=1e-5,
    precond="ilu",
    rtol=1e-4,
    maxiter=20000,
)
x = solve_Q(b)
```

### Streaming mean / variance / higher moments / quantiles

Per-cell mean and variance of a very large ensemble (e.g. many nullspace-
shuttle models) from three quantities per cell: `n`, `mean` and `M2` (sum of
squared deviations from the mean), with `var = M2 / (n - ddof)`.

| Function                                   | Purpose                                              |
|--------------------------------------------|------------------------------------------------------|
| `welford_update(n, mean, m2, x)`           | Add one sample (or one value per cell), Welford 1962. |
| `batch_moments(x, axis=0)`                 | `(n, mean, M2)` of an in-memory chunk (two-pass).    |
| `chan_merge(n_a, mean_a, m2_a, n_b, mean_b, m2_b)` | Pairwise merge of two accumulators (Chan et al. 1983). |
| `moments_variance(n, m2, ddof=1)`          | Variance from `(n, M2)`; NaN where `n <= ddof`.      |
| `hist_update(counts, x, lo, hi)`           | Add a chunk to per-cell fixed-bin histograms (in place); returns the number of clipped values. |
| `hist_quantiles(counts, lo, hi, probs)`    | Per-cell quantiles from the histograms, shape `(len(probs), ncell)`. |
| `batch_moments4(x, axis=0)`                | `(n, mean, M2, M3, M4)` of an in-memory chunk (two-pass). |
| `merge_moments4(acc_a, acc_b)`             | Pairwise merge of two `(n, mean, M2, M3, M4)` accumulators (Pebay 2008, Terriberry 2007). |
| `skew_kurtosis(n, m2, m3, m4, bias=True)`  | Skewness `g1` and excess kurtosis `g2` (`bias=False`: `G1`, `G2`); NaN where undefined. |
| `quantile_skewness(q_low, q_mid, q_high)`  | Bowley skewness `(q_high + q_low - 2 q_mid) / (q_high - q_low)` from three quantiles. |

```python
import numpy as np
from ensembles import batch_moments, chan_merge, moments_variance

n, mean, m2 = 0, np.zeros(ncell), np.zeros(ncell)
for chunk in chunks:                       # chunk shape (n_models, ncell)
    n, mean, m2 = chan_merge(n, mean, m2, *batch_moments(chunk, axis=0))
var = moments_variance(n, m2)
```

For MPI runs each rank accumulates its own chunks; reduce the per-rank
`(n, mean, M2)` triples with `chan_merge` (any reduction order gives the same
result to working precision). For resistivity, accumulate `log10(rho)`.

Quantiles need a summary larger than three numbers per cell. `hist_update`
keeps a fixed-bin histogram per cell, an `(ncell, nbins)` count array, in
`log10(rho)` over a common range `[lo, hi]`; histograms of the same range and
bin count merge by plain addition (also across MPI ranks). `hist_quantiles`
interpolates linearly inside the bin that holds the requested probability.
Memory is `ncell * nbins * 4` bytes (`uint32`).

```python
import math
from ensembles import hist_update, hist_quantiles

counts = np.zeros((ncell, 256), dtype=np.uint32)
for chunk in chunks:                       # same chunks as above
    n_clip = hist_update(counts, chunk, lo, hi)
phi = lambda k: 0.5 * (1.0 + math.erf(k / math.sqrt(2.0)))
probs = [phi(k) for k in (-2, -1, 0, 1, 2)]        # 0.0228 ... 0.9772
q = hist_quantiles(counts, lo, hi, probs)          # shape (5, ncell)
```

- The result is within one bin width `(hi - lo) / nbins` of the empirical
  inverse-CDF quantile (`numpy.quantile(..., method="inverted_cdf")`).
  numpy's default interpolates between neighbouring order statistics and can
  differ from it by their spacing, which is large in the tails of a small
  ensemble.
- Values outside `[lo, hi]` go into the first / last bin (`hist_update`
  returns how many); quantiles that fall in those bins are only bounds. Choose
  the range from a pilot chunk with some margin.
- For correlated ensembles (Markov-chain-like, or models shuttled from one
  starting point) the quantiles are those of the visited models, not error
  bars for independent samples.
- Higher moments: `Mk = sum((x - mean)**k)`; `g1 = sqrt(n) M3 / M2**1.5`,
  `g2 = n M4 / M2**2 - 3`. Start a run from `(0, zeros, zeros, zeros, zeros)`
  and `merge_moments4(acc, batch_moments4(chunk))` per chunk; MPI ranks merge
  the same way. `n`, `mean` and `M2` are identical to `chan_merge`. The
  sampling error of `g1` and `g2` for independent Gaussian samples is about
  `sqrt(6/n)` and `sqrt(24/n)`, so the higher moments need many more models
  than the 1-sigma quantiles. The Bowley coefficient from the quantiles
  above needs no extra accumulators and is less sensitive to outliers.
- The 1-sigma and 2-sigma probabilities are those of a normal distribution,
  0.1587 / 0.8413 and 0.0228 / 0.9772. For a non-Gaussian distribution these
  differ from `mean +/- k*std`; compare both.

### EOF / PCA

| Function                           | Purpose                                            |
|------------------------------------|----------------------------------------------------|
| `compute_eofs()`                   | SVD or sample-space eigendecomposition of ensemble. |
| `eof_reconstruct()`                | Truncated reconstruction from EOFs and PCs.        |
| `eof_sample()`                     | Draw a new sample in EOF space.                    |
| `fit_eof_model()` / `EOFModel`     | Fit an EOF model with optional prewhitening.       |
| `sample_physical_ensemble()`       | Generate new ensemble from fitted EOF model.       |

### Weighted KL + PCE surrogate

| Function                           | Purpose                                            |
|------------------------------------|----------------------------------------------------|
| `compute_weighted_kl()`            | Volume-weighted KL decomposition of log-ρ fields.  |
| `total_degree_multiindex()`        | Multi-index set for total-degree PCE basis.        |
| `fit_pce_for_kl_modes()`           | Least-squares PCE fit per KL mode.                 |
| `evaluate_kl_pce_surrogate()`      | Evaluate surrogate at new random inputs.           |
| `fit_kl_pce_model()` / `KLPCEModel` | One-call convenience wrapper.                    |

---

## Geostatistical initial-model ensemble (GST)

`generate_gst_model_ensemble` builds a distinct FEMTIC initial model for each
ensemble member by Ordinary Kriging from a sparse pilot-point cloud.  No
roughness matrix is required.

### Algorithm per member

1. Place pilot points (random / fixed / mixed — see `pp_mode`).
2. Draw log₁₀(ρ) values at pilot points, per `pp_value_mode`:
   - `"uniform"` (default) — Uniform(log_rho_min, log_rho_max), independent
     of location.
   - `"reference"` — reference_model(nearest free region) ± `pp_value_delta`,
     i.e. the reference log₁₀(ρ) looked up at the free-region barycentre
     nearest each pilot point, perturbed by Uniform(-pp_value_delta,
     +pp_value_delta).
3. Ordinary-Krig the values to all mesh cell centres (gstools).
4. Clamp the field to [log_rho_min, log_rho_max].
5. Write as `resistivity_block_iter0.dat` and/or `referencemodel.dat`.

### Key parameters

| Parameter        | Description                                                                              |
|------------------|------------------------------------------------------------------------------------------|
| `ref_mod_file`   | Full path to the template reference model (read once for cell-centre coordinates).       |
| `pp_mode`        | `"random"` — fresh random locations every member; `"fixed"` — fixed geometry, fresh values; `"mixed"` — fixed skeleton + random fill. |
| `n_pp`           | Number of randomly drawn pilot points per member (used when `pp_mode` ≠ `"fixed"`). Recommended: 50–200. |
| `pp_bbox`        | Bounding box `[x_min, x_max, y_min, y_max, z_min, z_max]` (m, z positive-down) for random placement. |
| `pp_coords`      | Explicit pilot-point array, shape `(N, 3)`.  Required for `"fixed"` / `"mixed"`.        |
| `log_rho_min`    | Lower bound in log₁₀(Ω·m) — draw floor and post-Kriging clamp.                          |
| `log_rho_max`    | Upper bound in log₁₀(Ω·m) — draw ceiling and post-Kriging clamp.                        |
| `pp_value_mode`  | `"uniform"` (default) — Uniform(log_rho_min, log_rho_max); `"reference"` — reference_model(nearest free region) ± `pp_value_delta`. |
| `pp_value_delta` | Half-width (log₁₀ Ω·m) of the perturbation around the reference value. Only used when `pp_value_mode="reference"`. Typical: 0.3–1.0. |
| `vario_model`    | gstools covariance class: `"Spherical"` (default), `"Gaussian"`, `"Exponential"`, etc.  |
| `vario_range`    | Correlation length (m).  Scalar = isotropic; 2-tuple `(h, v)` = geometric anisotropy.   |
| `vario_sill`     | Sill (variance) in (log₁₀ Ω·m)².  Recommended: 0.25–0.5.                               |
| `vario_nugget`   | Nugget in (log₁₀ Ω·m)².  Keep ≤ 10 % of sill.                                          |
| `vario_angles`   | Rotation `[α, β, γ]` in degrees; `None` = axis-aligned.                                 |
| `output_target`  | `"resistivity_block"` / `"referencemodel"` / `"both"` (recommended).                   |
| `rng`            | `numpy.random.Generator`, optional. Shared generator; pass a seeded one (e.g. `np.random.default_rng(seed)`) for a reproducible ensemble. `None` = fresh, non-reproducible generator. |
| `save_pilot_points` | `bool`, default `False`. Write every member's pilot-point coordinates and drawn log₁₀(ρ) values to `pilot_points_file` after the member loop. |
| `pilot_points_file` | `str`, optional. Output path for the pilot-point archive. `None` (default) → `f"{dir_base}pilot_points.npz"`. Ignored when `save_pilot_points=False`. |
| `seed`           | `int`, optional. Informational only — recorded in `pilot_points.npz` (`"seed"`, `-1` if omitted) for a self-describing archive. Does **not** seed the RNG; pass a seeded generator via `rng` for that. |
| `pp_regen_every` | `int`, optional. `None` (default) — the random pilot-point component (whole point set for `"random"`; random-fill portion of `"mixed"`/`"extrema"`) is redrawn every member, as before. An integer `K` instead redraws it once per block of `K` consecutive processed members, reusing it for the rest of the block; pilot-point **values** are still redrawn every member regardless. No effect for `pp_mode="fixed"`. See "Regenerating pilot points every *n*th member" below. |

### Regenerating pilot points every *n*th member

By default, `pp_mode="random"` draws a completely fresh pilot-point cloud
(locations **and** values) for every member, and `"mixed"`/`"extrema"`
redraw their random-fill points the same way (their fixed skeleton never
changes). `pp_regen_every` adds a middle ground: hold the **locations**
of that random component fixed across a block of `pp_regen_every`
consecutive members, while still drawing fresh **values** at those
locations for every member in the block. A new block gets a fresh set of
locations.

This decouples the two sources of ensemble spread that are normally
entangled: with `pp_regen_every=10`, members 0–9 all share one random
pilot-point layout and differ only in the resistivity values Kriged onto
it (spread from value randomness alone, at fixed geometry), members
10–19 share a different layout, and so on. Comparing within-block spread
to across-block spread is a direct diagnostic for which source dominates
the ensemble's spatial resolution — useful alongside
`gst_parameter_diagnostics` when choosing `n_pp`/variogram parameters.

Blocks are counted by **position within the processed member list**
(`fromto`/`range(n_samples)`), not by raw member index — only relevant
if `fromto` has gaps. `pp_regen_every=1` is equivalent to `None` (every
block has exactly one member, so nothing is ever reused). Recorded in
`pilot_points.npz` as `"pp_regen_every"` (`-1` if not set).

```python
# Members 0-9 share one random pilot-point layout, 10-19 a different one, etc.
mod_list = generate_gst_model_ensemble(
    ...,
    pp_mode         = "random",
    n_pp            = 100,
    pp_regen_every  = 10,
)
```

### Reproducibility & the `pilot_points.npz` archive

`generate_gst_model_ensemble` itself never seeds anything — reproducibility
is entirely up to the caller, via the `rng` it passes in:

```python
seed = 20260725
rng  = np.random.default_rng(seed)

mod_list = generate_gst_model_ensemble(
    ...,
    rng               = rng,
    save_pilot_points = True,
    pilot_points_file = None,        # -> f"{dir_base}pilot_points.npz"
    seed              = seed,        # recorded in the archive, not used to seed
)
```

With `save_pilot_points=True`, the coordinates and values actually drawn
for every member — the ground truth of what the Kriging step consumed —
are written to a single compressed `.npz`:

| Key | Shape | Description |
|---|---|---|
| `member_ids` | `(n_members,)` int | Processed member indices (`fromto_arr`). |
| `pp_x`, `pp_y`, `pp_z` | `(n_members, n_pp_total)` float | Pilot-point coordinates (easting, northing, depth) per member. `n_pp_total` is constant across members for a given `pp_mode` (fixed/extrema skeleton size + `n_pp` random fill, where applicable), so stacking into one array is always well-defined. |
| `pp_vals` | `(n_members, n_pp_total)` float | log₁₀(ρ) value drawn at each pilot point, per member. |
| `pp_mode`, `pp_value_mode`, `log_rho_min`, `log_rho_max`, `vario_model`, `vario_range`, `vario_sill`, `vario_nugget`, `seed`, `pp_regen_every` | scalar/array | Snapshot of the call's configuration (`pp_regen_every` is `-1` if not set). |

```python
d = np.load("ubinas_gst_pilot_points.npz")
d["pp_x"].shape        # (n_members, n_pp_total)
d["pp_vals"][3]        # log10(rho) values drawn for member 3
int(d["seed"])         # -1 if the run wasn't seeded
```

Useful for reproducibility audits, feeding the GST parameter-diagnostic
functions below (`gst_pilot_point_cv`, `gst_parameter_diagnostics`), or
re-plotting the pilot-point cloud without re-running Kriging.

### Tuning guidance

**Pilot-point density:** 50–150 points for a volume of order 100 km × 100 km × 50 km.
The variogram range should be at least 1.5× the typical inter-pilot-point spacing.

**Variogram range vs. survey geometry:** horizontal range ≈ 20–50 % of survey aperture;
vertical range ≈ 30–50 % of target depth.  MT sensitivity decays with depth, so
vertical ranges shorter than horizontal ranges are physically appropriate.

**Sill:** 0.25 (log₁₀ Ω·m)² gives ≈ ±0.5 log₁₀ units 1-sigma.  Increase for
larger expected resistivity contrasts.

**`output_target = "both"`** is recommended: the inversion then explores the misfit
surface starting from — and regularised toward — a structurally distinct spatial
pattern for every member.

### Example

```python
from ensembles import generate_gst_model_ensemble

# --- random pilot points (basic usage) ---
mod_list = generate_gst_model_ensemble(
    dir_base     = "./ubinas_gst_",
    n_samples    = 32,
    ref_mod_file = "./templates/referencemodel.dat",
    pp_mode      = "random",
    n_pp         = 100,
    pp_bbox      = [-50000, 50000, -50000, 50000, 0, 80000],
    log_rho_min  = 0.0,
    log_rho_max  = 4.0,
    vario_model  = "Spherical",
    vario_range  = (20000., 5000.),
    vario_sill   = 0.5,
    vario_nugget = 0.01,
    output_target = "both",
)

# --- extrema pilot points (structure-guided) ---
mod_list = generate_gst_model_ensemble(
    dir_base          = "./ubinas_gst_",
    n_samples         = 32,
    ref_mod_file      = "./templates/referencemodel.dat",
    pp_mode           = "extrema",
    n_pp              = 80,           # random fill on top of extrema
    pp_bbox           = [-50000, 50000, -50000, 50000, 0, 80000],
    pp_roi            = [-40000, 40000, -40000, 40000, 0, 60000],
    pp_extrema_k      = 30,
    pp_extrema_which  = "both",
    log_rho_min       = 0.0,
    log_rho_max       = 4.0,
    vario_model       = "Spherical",
    vario_range       = (20000., 5000.),
    vario_sill        = 0.5,
    vario_nugget      = 0.01,
    output_target     = "both",
)

# --- reference-anchored pilot-point values (stay close to reference model) ---
mod_list = generate_gst_model_ensemble(
    dir_base      = "./ubinas_gst_",
    n_samples     = 32,
    ref_mod_file  = "./templates/referencemodel.dat",
    pp_mode       = "random",
    n_pp          = 100,
    pp_bbox       = [-50000, 50000, -50000, 50000, 0, 80000],
    pp_value_mode  = "reference",
    pp_value_delta = 0.5,      # +-0.5 log10 Ohm.m around the reference value
    log_rho_min   = 0.0,
    log_rho_max   = 4.0,
    vario_model   = "Spherical",
    vario_range   = (20000., 5000.),
    vario_sill    = 0.5,
    vario_nugget  = 0.01,
    output_target = "both",
)
```

---

## RTO prior perturbation (`sample_rto_prior`)

Randomize-then-optimize (Bardsley et al., 2014) turns every ensemble member into
a perturbed inversion with ONE fixed regularisation weight alpha:

```
minimise   (d~ - F(m))^T Cd^-1 (d~ - F(m)) + alpha |R (m - m~_ref)|^2
           d~     = d + N(0, Cd)                         (generate_data_ensemble)
           m~_ref = m_ref + x,  x ~ N(0, (alpha R^T R)^-1)  (generate_rto_model_ensemble)
```

For linear F the minimisers are exact posterior samples; for nonlinear F they are
approximate (see "Notes" below). The prior perturbation `x` scales as
`alpha^(-1/2)`, so **alpha must be the regularisation weight of the member
inversions** (FEMTIC: the `Alpha` column of `femtic.cnv`; check there that
`ObjFunc = Misfit + Alpha*Roughness` holds for your FEMTIC version).

`sample_rto_prior(R, n_samples, alpha=...)` draws `x` exactly, without rank truncation:

```
Q = R^T R + lam I,    x = alpha^(-1/2) Q^(-1) ( R^T xi1 + sqrt(lam) xi2 )
xi1 ~ N(0, I_m),  xi2 ~ N(0, I_n)    =>    Cov(x) = (alpha Q)^(-1)
```

`generate_rto_model_ensemble` takes **R itself** (not `Q = R^T R`).

| Argument | Default | Meaning |
|---|---|---|
| `alpha` | **required** | Regularisation weight of the member inversions. `None`, `<= 0` or non-finite raises `ValueError`; there is deliberately no default (the former implicit alpha = 1 gave wrong spreads). |
| `algo` | `"auto"` | `"splu"`: sparse LU of Q, factored once. `"cg"`: Jacobi-preconditioned CG per draw. `"auto"`: LU up to 60000 unknowns, CG above. |
| `lam`, `lam_alpha` | `0`, `1e-4` | Diagonal shift (R^T R is singular for constants). `lam = 0` means `lam_alpha * median(diag(R^T R))`. |
| `solver_rtol` | `1e-8` | CG only. Keep it tight: CG converges last on the smoothest modes, which carry most of the prior variance. |
| `start_from_perturbed` | `False` | `True`: also write the perturbed model to `resistivity_block_iter0.dat` (a shared symlink is replaced by a real file; the template is never modified). |
| `remove_mean` | `False` | `True`: subtract each draw's global mean (see "Constant mode" below). |
| `rng` | `None` | Shared `numpy.random.Generator` for reproducibility. |

Removed arguments of the former implementation (`n_eig`, `n_oversampling`,
`n_power_iter`, `sigma2_residual`, `lam_mode`, `solver_method`, `precond`) are
still accepted and ignored with a `DeprecationWarning`; `algo="low rank"` maps to
`"auto"` and `algo="full rank"` to `"cg"`, also with a warning.

```python
import numpy as np
import scipy.sparse as scs
import femtic as fem            # import femtic before ensembles (circular import)
import ensembles as ens

R = scs.load_npz("R_coo.npz")
mods = ens.generate_rto_model_ensemble(
    "rto", "./ens_", 64, None, "referencemodel.dat", R,
    alpha=ALPHA,                # Alpha of the member inversions (femtic.cnv)
    algo="auto", rng=np.random.default_rng(1234), out=True)
```

### Diagnostics printed with `out=True`

```
RTO prior sampling done: n=..., lam=..., solver=splu, RMS std = ... (log10 units;
... after removing each draw's global mean), energy ratio = 1.00 (expect ~1).
```

The **energy ratio** `mean(alpha |R x|^2) / rank(R)` is the chi^2 per degree of
freedom of the roughness term of the draws; it is ~1 for correct draws. The former
randomized-SVD sampler gave ~ `n_eig / n_free` (0.05 in a lattice test with
n_eig = 128). A `RuntimeWarning` is issued outside (0.5, 2). The same function
can be applied to the perturbation actually written to disk (perturbed minus
`*_orig.dat` log10 model), see `prior_energy_ratio`.

### Constant mode

R annihilates constants, so `R^T R` is singular and `lam` is needed. The draws then
contain a global offset of the whole model with per-cell std ~ `1/sqrt(n*alpha*lam)`
that is fixed by the arbitrary `lam`, not by data or regularisation. If FEMTIC's
objective depends on the reference model only through `R (m - m_ref)`, this offset
has no influence on the member inversion; it does inflate the printed RMS std and,
with `start_from_perturbed=True`, moves the starting model. `remove_mean=True`
removes it (the energy ratio is unchanged; a warning is issued if R does not
annihilate constants). Whether FEMTIC's objective is indeed insensitive to such an
offset has not been checked against FEMTIC itself.

### Notes

- **Why the old low-rank sampler gave spreads far too small.** `randomized_svd(R)`
  returns the *largest* singular values s_k of R, i.e. the roughest modes. For
  `N(0, (R^T R)^-1)` the variance of mode k is `1/s_k^2`, so these modes carry the
  *least* variance. In a 3-D lattice test (n = 2744, n_eig = 128) they held
  0.08-0.5 % of the prior variance; an RTO ensemble built on them had 0.20-0.27 of
  the correct spread in a linear test. More `n_eig` or power iterations do not help.
  Missing alpha (implicit alpha = 1; `MOD_MU`/`MOD_PDF` in `femtic_rto_prep.py` were
  never passed on) is a second, independent error whenever the real alpha differs
  from 1.
- **Nonlinear F.** For nonlinear F, unweighted RTO samples are not guaranteed to
  follow the posterior (Bardsley et al., 2014; Blatter et al., Part I, Sec. 3.1;
  Oliver; Ba et al.). The direction of the bias is problem dependent.
- **Choice of alpha** sets the uncertainty level (Blatter et al., Part I, Sec. 3.3):
  taking the largest alpha consistent with a target RMS biases towards smooth
  models and tends to lower the posterior variance.
- **Comparison with GST.** The GST spread is set by the user-chosen prior amplitude
  (`vario_sill`, `pp_value_delta`); it need not equal the RTO posterior spread even
  when both are implemented correctly.
- Tested on lattices and on mock block files only (`rto_integration_test.py`);
  not run on a real FEMTIC mesh. LU memory (fill-in) on large 3-D meshes is
  untested; use `algo="cg"` if it becomes a problem.

## Quick-reference: recommended parameter values (FEMTIC)

| Parameter | Recommended | Location |
|-----------|-------------|----------|
| `alpha` | **Alpha of the member inversions (femtic.cnv); required** | `generate_rto_model_ensemble` |
| `algo` | `"auto"` | `generate_rto_model_ensemble` |
| `lam` / `lam_alpha` | `0` / `1e-4` | `generate_rto_model_ensemble`, `sample_rto_prior` |
| `solver_rtol` / `rtol` | `1e-8` (CG), never looser than `1e-6` | `generate_rto_model_ensemble`, `sample_rto_prior` |
| `remove_mean` | `False`; `True` to drop the lam-driven global offset | `generate_rto_model_ensemble` |
| `start_from_perturbed` | `False` (shared starting model) or `True` (each member starts at its own draw) | `generate_rto_model_ensemble` |
| `precond` | `"jacobi"` (ILU is not an SPD preconditioner) | `make_precision_solver` |
| `maxiter` | default `max(5000, 2n)` | `sample_rto_prior` |

## Solver performance and parameter guidance

### Choosing the sampling algorithm

| Scenario | Recommended |
|----------|-------------|
| Normal production run | `algo="auto"` (sparse LU for up to 60000 unknowns, factored once for all draws) |
| Very large mesh or LU memory problems | `algo="cg"`, `solver_rtol=1e-8` (Jacobi preconditioner) |
| Accuracy check of a result | `prior_energy_ratio(R, samples, alpha)` should be ~1 |

`"low rank"` is no longer available in `generate_rto_model_ensemble`.

### CG tolerance

A loose CG tolerance removes the smoothest modes, which carry the largest prior
variance: in the lattice tests `rtol=1e-2` gave 0.32-0.50 of the exact RMS std,
while `rtol=1e-8` reproduced it within 2 % (Monte Carlo error of the test). The former advice "rtol 1e-2 for
sampling" was wrong and has been removed.

### Legacy full-rank sampler (`sample_rtr_full_rank` + `make_precision_solver`)

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `rtol` | `1e-8`, `atol=0` | Not `1e-2` / `1e-3`; see above. These are now the defaults when `sample_rtr_full_rank` builds the solver itself (`solver=None`); a pre-built solver keeps its own tolerances. |
| `precond` | `"jacobi"` | `"ilu"` (`spilu`) is not an SPD preconditioner; CG+ILU did not converge on a Laplacian-type lattice. It may work on your mesh: check the iteration count and the energy ratio. |
| `solver_method` | `"cg"` | Q = R^T R + lam I is SPD. |
| `lam_mode`, `lam_alpha` | `"scaled_median_diag"`, `1e-4` | Pass the same `lam_mode`/`lam_alpha` to `make_precision_solver` and to `sample_rtr_full_rank`: the sampler needs the same lam for its `sqrt(lam)*xi2` term. |
| `maxiter` | 500-1000 if you want a hard cap | A cap that is hit means the draw is under-converged. |

## Solver convergence tips

If CG does not converge (`RuntimeWarning` from `sample_rto_prior` or
`RuntimeError: Iterative solver did not converge` from `make_precision_solver`):

1. Use `algo="auto"` / `"splu"` (direct solve), if memory allows.
2. Raise `maxiter`, keep `rtol` tight.
3. Raising `lam_alpha` improves conditioning but also shrinks the long-wavelength
   variance (mode variance `1/(e + lam)`); stay at or below `1e-3`.
4. Do not loosen `rtol` or switch to the deprecated low-rank sampler: both give
   spreads that are too small.

---

## Dependencies

| Package             | Role                                        |
|---------------------|---------------------------------------------|
| `numpy`             | Core array operations.                      |
| `scipy`             | Sparse matrices, iterative solvers, SVD, KD-tree neighbour search (`scipy.spatial`, used by `"extrema"` pilot-point mode). |
| `gstools`           | Variogram models and Ordinary Kriging (GST).|
| `joblib` (optional) | Kept for backward compatibility.            |
| `pyamg` (optional)  | AMG preconditioner.                         |
| `sksparse` (optional) | CHOLMOD direct solver.                    |

---

## Related scripts

| Script                  | README                          | Purpose                                |
|-------------------------|---------------------------------|----------------------------------------|
| `femtic_rto_rough.py`   | `README_femtic_rto_rough.md`    | Extract roughness matrix from FEMTIC.  |
| `femtic_rto_prior.py`   | `README_femtic_rto_prior.md`    | Build prior covariance proxy.          |
| `femtic_rto_prep.py`    | `README_femtic_rto_prep.md`     | Generate RTO ensemble.                 |
| `femtic_gst_prep.py`    | `README_femtic_gst_prep.md`     | Generate GST ensemble (pilot-point Kriging). |

---

## References

- Suzuki, A.
  *Assessing inversion uncertainty from initial-model variability in 3-D
  magnetotelluric inversion: Application to a geothermal field.*
  Journal of Applied Geophysics, 2026, **251**, 106320,
  doi:[10.1016/j.jappgeo.2026.106320](https://doi.org/10.1016/j.jappgeo.2026.106320).
  Preprint: [10.31223/X5NM9X](https://doi.org/10.31223/X5NM9X).

- Müller, S.; Schüler, L.; Zech, A. & Heße, F.
  *GSTools v1.3: a toolbox for geostatistical modelling in Python.*
  Geoscientific Model Development, 2022, **15**, 3161–3182,
  doi:[10.5194/gmd-15-3161-2022](https://doi.org/10.5194/gmd-15-3161-2022).

- Bardsley, J. M.; Solonen, A.; Haario, H. & Laine, M.
  *Randomize-Then-Optimize: a Method for Sampling from Posterior Distributions
  in Nonlinear Inverse Problems.*
  SIAM Journal on Scientific Computing, 2014, **36**, A1895–A1910,
  doi:[10.1137/140964023](https://doi.org/10.1137/140964023).

- Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
  *Uncertainty quantification for regularized inversion of electromagnetic
  geophysical data. Part I: Motivation and Theory* (2022; Part II: Application in
  1D and 2D problems). Geophysical Journal International. Authors and titles
  checked on the author-hosted manuscripts (submitted to GJI),
  e.g. <https://igppweb.ucsd.edu/~mmorzfeld/docs/RTO_Part_1.pdf>; **DOI,
  volume and pages of the published versions not confirmed** (section and
  equation numbers above refer to the manuscript).

- Oliver, D. S.
  *Metropolized randomized maximum likelihood for sampling from multimodal
  distributions.* arXiv:1507.08563 (sole author checked; journal reference not
  checked).

- Ba, Y.; de Wiljes, J.; Oliver, D. S. & Reich, S.
  *Randomized maximum likelihood based posterior sampling.* arXiv:2101.03612
  (authors checked; journal reference not checked).

---

## Version / provenance

Updated: 2026-10-21 (RTO prior perturbation corrected: `sample_rto_prior`, required `alpha`); 2026-10-17 (streaming skewness/kurtosis); 2026-10-16 (streaming mean/variance, histogram quantiles); 2026-04-27 (GST); 2026-04-11 (consolidation); 2026-04-02 (cleanup)

### Changelog (2026-10-21) -- RTO prior perturbation corrected
Claude Sonnet 5.5 (Anthropic). AI-generated; review before production use.
Cause: ensemble spreads from `femtic_rto_prep.py` were far too small compared
with GST.
- NEW `sample_rto_prior()` and `prior_energy_ratio()`: exact draws from
  `N(0, (alpha (R^T R + lam I))^-1)`, sparse LU or Jacobi-CG with tight tolerance.
- `generate_rto_model_ensemble`: uses it; new REQUIRED `alpha`; `algo` is
  `"auto"|"splu"|"cg"` (`"low rank"`/`"full rank"` mapped with a
  `DeprecationWarning`; removed low-rank arguments ignored with a warning);
  draws exactly `len(members)` samples; checks that R matches the free-region
  vector; prints RMS std and the energy ratio; new `start_from_perturbed` and
  `remove_mean`.
- `sample_rtr_low_rank` DEPRECATED (roughest instead of smoothest modes).
- `sample_rtr_full_rank` fixed: the draw `(R^T R + lam I)^-1 R^T xi` has covariance
  `Q^-1 R^T R Q^-1`; the missing `sqrt(lam) * xi2` term was added (shared helper
  `_resolve_lam`). Also affects `femtic.modify_model_npz(method="precision_sample")`.
  A solver built inside the sampler now defaults to `rtol=1e-8, atol=0` (was
  `1e-3 / 1e-4`; override with `solver_kwargs`).
- `make_precision_solver` docstring: tight `rtol` for sampling; ILU caveat.
- Tests: `rto_integration_test.py` (lattices, mock block files, prep-script alpha block).

### Changelog (2026-04-27)
- Added `generate_gst_model_ensemble` — geostatistical initial-model ensemble
  via pilot-point Ordinary Kriging (gstools).  No roughness matrix required.
  Supports `pp_mode = "random" | "fixed" | "mixed"`, fully configurable
  variogram, and `output_target = "resistivity_block" | "referencemodel" | "both"`.
  (`pp_mode = "extrema"` added 2026-06-06; see below.)
- Renamed `generate_model_ensemble` → `generate_rto_model_ensemble` for
  consistency with `generate_gst_model_ensemble`.
- Added `gstools` to the dependency table.
- Added `femtic_gst_prep.py` to the related-scripts table.
- Module docstring updated.

### Changelog (2026-04-11)
- `check_sparse_matrix` moved here from `femtic.py` — all matrix/roughness
  diagnostics are now in a single place.
- `femtic.py` Section 2 now imports these functions from `ensembles` rather
  than maintaining duplicate definitions.

### Changelog (2026-04-02)
- Fixed `FileNotFoundError` in `generate_model_ensemble`: `fem.insert_model`
  was called with `template=refmod` (bare basename), which failed when the
  current working directory was not the ensemble member directory. The template
  is now the full per-member path `<dir_base><iens>/<refmod>_orig.dat` (the
  backup just created by `shutil.copy`), matching the intent of the original
  code.

### Changelog (2026-03-31 cleanup)
- Removed debug `print("Sample #", ix, "solver is", solver)` from
  `sample_rtr_full_rank` (printed a callable object on every sample).
- Removed dead commented-out `fem.insert_model` block in
  `generate_model_ensemble`.
- Removed redundant per-sample `print('sample:', file)` inside the
  model-writing loop (list is already printed at the end).
- `_diag_rtr` removed; its sole call-site in `make_rtr_preconditioner`
  now uses the canonical `_rtr_diag` (better docstring, ndim guard).
- `estimate_low_rank_eigpairs` removed — used the deprecated `eigsh`-based
  approach and was no longer called anywhere.
- Docstrings for `generate_model_ensemble`, `sample_rtr_low_rank`,
  `sample_rtr_full_rank`, and `make_precision_solver` enriched with
  **Recommended** parameter values for FEMTIC workflows (see below).

Author: Volker Rath (DIAS)

### Changelog (2026-05-28)
- `put_files` and `generate_directories`: new `relative_links` parameter
  (default `True`).  When `True`, symlink targets are stored as
  `os.path.relpath` paths relative to each member directory, so the ensemble
  tree is portable after tgz/copy to another machine.  Set to `False` to
  restore the previous absolute-path behaviour.  The prep scripts expose this
  as the `RELATIVE_LINKS` config variable.

### Changelog (2026-05-28) — bug fix

- `generate_gst_model_ensemble`: fixed template-clobbering bug when
  `output_target="both"`.  The `resistivity_block` write used
  `member_dir + reference_file` as both template and (indirectly) output;
  if `resistivity_file == reference_file` or `insert_model` writes in-place,
  the template was corrupted before the `referencemodel` write could read it,
  raising `ValueError: Invalid resistivity block header: []`.  The template
  path is now resolved once into `template_path` before either write, so both
  branches always read the original un-modified file.

### Changelog (2026-06-06) — extrema pilot-point mode

- Added `_find_extrema_pilot_points()` private helper: KD-tree-based local
  extremum detection on free-region barycentres with optional ROI mask.
  Requires `scipy.spatial` (already a transitive dependency of gstools/numpy).
- `generate_gst_model_ensemble`: new `pp_mode = "extrema"` option.
  New parameters: `pp_roi`, `pp_extrema_k`, `pp_extrema_which` (all with
  safe defaults so existing call sites are unchanged).
  The extremum skeleton is computed once before the member loop (same geometry
  every member); `n_pp` random fill points are drawn fresh per member.
  Graceful fallback to `"random"` with `RuntimeWarning` if no extrema are found.
- `scipy.spatial` added explicitly to the dependency table.

### Changelog (2026-05-28) — GST parameter estimation

Added four functions for choosing variogram parameters before committing to a
full GST ensemble run.  All require `gstools`; all are purely diagnostic (no
FEMTIC runs needed unless noted).

| Function | Strategy | What it does |
|---|---|---|
| `gst_variogram_from_rto_samples` | 1 | Fits a gstools variogram to existing RTO (or GST) samples via `vario_estimate` + `fit_variogram`. Makes GST statistically consistent with RTO. |
| `gst_pilot_point_cv` | 2 | Leave-one-out Ordinary Kriging CV at the pilot points using the reference model as truth. Returns RMSE, MAE, and skill score without running FEMTIC. |
| `gst_sill_from_jacobian` | 3 | Linearised propagation of model covariance to data space via the Jacobian J. Calibrates the sill so that ensemble forward-response spread matches data noise at a target coverage level. Falls back to a diagonal C_m approximation for large meshes (n_cells > 5000). |
| `gst_parameter_diagnostics` | 4 | Integrating diagnostic: ensemble std maps, empirical variogram vs. target, optional LOO-CV (calls Strategy 2), optional data-space spread ratio (via J). Prints structured summary; optionally saves a multi-panel figure. |

### Changelog (2026-06-10) — explicit member-index list for `fromto`

- Added `_resolve_fromto(fromto, n_samples)` private helper that centralises
  the index-resolution logic previously duplicated in four functions.
- All four public ensemble functions now accept an **explicit list of member
  indices**; `None` means all members.  Range semantics (`[start, stop]`)
  are not supported — a list always means exact indices:

  | `fromto` value | Behaviour |
  |---|---|
  | `None` | all members `0 … n_samples-1` |
  | `[i, j, k, …]` | exactly those member indices |

  Affected functions: `generate_directories`, `generate_rto_model_ensemble`,
  `generate_gst_model_ensemble`, `generate_data_ensemble`.
- Type hints updated from `Optional[Tuple[int, int]]` to
  `Optional[List[int]]`; `Union`/`Tuple` removed from `fromto` signatures.
- Companion change in `femtic_rto_prep.py`: config variable `FROM_TO`
  renamed to `ENS_LIST`; `VIZ_SAMPLES` and all file lists scoped to the
  resolved active member set (`ENS_MEMBERS`) to prevent `FileNotFoundError`
  when `ENS_LIST` restricts the run to a subset of members.

### Changelog (2026-07-05) — reference-anchored pilot-point values

- `generate_gst_model_ensemble` gained `pp_value_mode` (`"uniform"` |
  `"reference"`) and `pp_value_delta` parameters controlling how pilot-point
  log₁₀(ρ) values are drawn:
  - `"uniform"` (default) preserves the original
    Uniform(log_rho_min, log_rho_max) draw.
  - `"reference"` draws reference_model(nearest free region) ±
    `pp_value_delta`, using a `scipy.spatial.KDTree` nearest-neighbour
    lookup against the free-region barycentres; result is still clamped to
    `[log_rho_min, log_rho_max]`.
- Reference log₁₀(ρ) at free regions is now computed unconditionally
  (previously only inside the `pp_mode="extrema"` branch), so it is
  available to both `"extrema"` pilot-point placement and `"reference"`
  pilot-point value mode.
- Companion change in `femtic_gst_prep.py`: added `MOD_PP_VALUE_MODE` /
  `MOD_PP_VALUE_DELTA` config variables, passed through as `pp_value_mode` /
  `pp_value_delta`.

### Changelog (2026-07-05) — larger default neighbourhood for extremum detection

- Raised the default neighbourhood size for extremum detection from `k=9`
  to `k=30` in both `_find_extrema_pilot_points` and
  `generate_gst_model_ensemble`'s `pp_extrema_k` — the strictly-less/
  greater-than-all-neighbours test was flagging too many spurious local
  minima/maxima at small `k` on typical FEMTIC meshes.
- Recommended range updated from 7–15 to 20–40; increase further (e.g. 30+)
  if the test still finds too many extrema on dense or noisy meshes.
- Companion change in `femtic_gst_prep.py`: default `MOD_PP_EXTREMA_K`
  raised from 9 to 30.

### Changelog (2026-07-17) — scipy.sparse: matrix → array API

- Migrated from legacy `scipy.sparse` matrix classes to the array-equivalent
  API throughout: removed unused `isspmatrix` import; `scipy.sparse.spmatrix`
  type hints replaced with `scipy.sparse.sparray` (`get_roughness`,
  `make_prior_cov`, `prune_inplace`, `prune_rebuild`, `dense_to_csr`,
  `matrix_reduce`, `check_sparse_matrix`, `build_rtr_operator`,
  `make_rtr_preconditioner`, `make_sparse_cholesky_precision_solver`,
  `_rtr_diag`, `pick_lam_from_rtr_diag`).
- Replaced `scipy.sparse.identity()` (returns a legacy matrix) with
  `scipy.sparse.eye_array()` in `make_rtr_preconditioner` and
  `make_sparse_cholesky_precision_solver`.
- Fixed a latent bug in `get_roughness` and `make_prior_cov` where `eye` was
  imported aliased as `eye_array`, silently returning a legacy sparse matrix
  instead of an array; now imports the genuine `eye_array` constructor.
- No functional/numerical change; the module already built matrices via
  `csr_array`/`csc_array`/`coo_array` internally, so this is purely a type
  and import cleanup plus the `eye_array` fix.

### Changelog (2026-07-25) — reproducible ensembles & pilot-point export

- `generate_gst_model_ensemble` gained `save_pilot_points` / `pilot_points_file`
  / `seed`. With `save_pilot_points=True`, every member's pilot-point
  coordinates (`pp_x`/`pp_y`/`pp_z`) and drawn log₁₀(ρ) values (`pp_vals`)
  are accumulated during the member loop and written to a single compressed
  `.npz` (default `f"{dir_base}pilot_points.npz"`) after it completes,
  alongside `pp_mode`/`pp_value_mode`/variogram/`log_rho_min`/`log_rho_max`/
  `seed` metadata — see the "Reproducibility & the `pilot_points.npz`
  archive" section above for the full key list and an example.
- `seed` is informational only: it's recorded in the archive but does not
  seed anything itself — pass a seeded `rng` for that. Reproducibility is
  therefore entirely the caller's responsibility; `femtic_gst_prep.py` now
  does this via a `RANDOM_SEED` config variable (see its README).
- `generate_data_ensemble` gained an `rng` parameter, forwarded to
  `femtic.modify_data` (which already accepted one) for each member. Before
  this change every `modify_data` call fell back to its own fresh, unseeded
  generator regardless of any `rng` the model-perturbation path was using —
  so `PERTURB_DAT` runs were never reproducible even when
  `generate_gst_model_ensemble`/`generate_rto_model_ensemble` were seeded.
  Passing the same shared `rng` to both now makes the full ensemble
  (model *and* data perturbation) reproducible together.

### Changelog (2026-09-09) — covariance-based ensemble sampling for repair

Three functions added after `sample_physical_ensemble()` (EOF/PCA
section), giving `femtic_ens_repair.py`'s new
`MOD_REPAIR_METHOD="eof_sample"` option a way to draw new candidate
starting models from an existing ensemble's own empirical covariance,
rather than only averaging two random members:

- `eof_model_from_covariance(mean, n_samples_fit, eigval=, eigvec=,
  cov=, ...)`: builds an `EOFModel` directly from a *precomputed*
  covariance instead of fitting one from a raw ensemble matrix —
  accepts either femtic_ens_post.py's `"low_rank"` factorisation
  (`eigval`/`eigvec`, exact eigenpairs, no decomposition needed) or its
  `"full"` dense `cov` (eigendecomposed and truncated here the same way
  `fit_eof_model` truncates its SVD). Returns the same `EOFModel`
  container `fit_eof_model()` does, so `sample_physical_ensemble()`
  works identically regardless of which path built the model.
- `sample_new_models_from_ensemble(ens_matrix, n_new, ...)` /
  `sample_new_models_from_covariance(mean, n_samples_fit, n_new, ...)`:
  thin femtic-shaped convenience wrappers (`fit_eof_model`/
  `eof_model_from_covariance` + `sample_physical_ensemble`) that handle
  the transpose between `femtic_ens_*.py`'s `ens_matrix` convention
  — samples as rows, `(n_members, n_free)` — and
  `fit_eof_model`/`sample_physical_ensemble`'s samples-as-columns
  `(ncells, nsamples)` convention, so callers never touch `EOFModel`
  directly. `femtic_ens_repair.py` itself calls the lower-level
  `fit_eof_model`/`eof_model_from_covariance`/`sample_physical_ensemble`
  directly instead (fitting once, then sampling once per non-converged
  member), but these wrappers are the simpler one-shot entry point for
  other callers.

**Note (pre-existing, unrelated):** `eof_generate_ensemble()`, defined
earlier in this section, has an empty body (only a docstring) and always
returns `None` — flagged here rather than fixed, pending confirmation of
whether it should be implemented or removed.

### Changelog (2026-09-09) — periodic pilot-point regeneration

- `generate_gst_model_ensemble` gained `pp_regen_every` (default `None` =
  unchanged behaviour). When set to an integer `K`, the *random*
  pilot-point component (`pp_mode="random"`'s whole point set, or the
  random-fill portion of `"mixed"`/`"extrema"`) is redrawn once per block
  of `K` consecutive processed members instead of every member, via a
  new position-keyed cache; pilot-point values are still redrawn every
  member regardless, so members sharing a block still differ. No effect
  for `pp_mode="fixed"` (a warning is printed if set anyway, since there
  is no random component to regenerate). See "Regenerating pilot points
  every *n*th member" above for the full explanation and an example.
- Recorded in `pilot_points.npz` as `"pp_regen_every"` (`-1` if not set).
- The member loop now iterates `enumerate(fromto_arr)` (tracking each
  member's position, needed for the block-caching above) instead of
  `fromto_arr` alone; no change to iteration order or per-member output.
- `femtic_gst_prep.py` gained a matching `MOD_PP_REGEN_EVERY` config
  variable — see its own README's matching changelog entry, which also
  notes an unrelated pre-existing bug fixed in the same pass
  (`output_target=MOD_OUTPUFalseT_TARGET`, an undefined-variable typo
  that raised `NameError` on every `PERTURB_MOD=True` run, corrected to
  `MOD_OUTPUT_TARGET`).

### Changelog (2026-10-16) -- streaming mean / variance

Added by Claude Sonnet 5.5 (Anthropic). AI-generated code -- please review
before production use.

- Added `welford_update()`, `batch_moments()`, `chan_merge()` and
  `moments_variance()` (after `generate_data_ensemble`, before the EOF/PCA
  section). Vectorised over cells; no change to existing functions.
  `chan_merge` is the pairwise update of Chan, Golub and LeVeque (1983) and
  is safe when either side is empty.
- Verification: against `numpy.var(ddof=1)` on data with mean 1e6 and sd 2
  (hard for naive formulas), chunked merge, row-by-row Welford and a binary
  tree of merges all agree to a relative error of about 1e-11.
- References: Welford, B. P. (1962), Technometrics 4(3), 419-420,
  doi:10.1080/00401706.1962.10490022 (DOI supplied by the user, not
  independently confirmed); Chan, T. F., Golub, G. H., LeVeque, R. J. (1983),
  The American Statistician 37(3), 242-247,
  doi:10.1080/00031305.1983.10483115.

### Changelog (2026-10-16) -- streaming quantiles

Added by Claude Sonnet 5.5 (Anthropic). AI-generated code -- please review
before production use.

- Added `hist_update()` and `hist_quantiles()` to the streaming section:
  per-cell fixed-bin histograms (mergeable by addition) and quantiles with
  linear interpolation inside the bin. No change to existing functions.
- Verification (synthetic Gaussian cells, 6000 samples, chunks of 700,
  400 bins over [-2, 6], bin width 0.02): chunked mean and variance agree
  with numpy to 1e-14; the quantiles at the +/-1 and +/-2 sigma
  probabilities differ from `numpy.quantile` by at most 0.007, i.e. less
  than one bin; histograms of two halves add up to the histogram of the
  whole; an empty cell gives NaN; clipping is counted.
- Used by `femtic_nss.py` (chunked nullspace-shuttle statistics).

### Changelog (2026-10-17) -- streaming higher moments

Added by Claude Sonnet 5.5 (Anthropic). AI-generated code -- please review
before production use.

- Added `batch_moments4()`, `merge_moments4()`, `skew_kurtosis()` and
  `quantile_skewness()` to the streaming section. No change to existing
  functions.
- Verification (31 chunks, 20000 samples, 40 lognormal cells with an offset
  of order 1e4): merged mean and variance agree with numpy to relative 1e-13;
  `g1`, `g2` and the bias-corrected `G1`, `G2` agree with `scipy.stats.skew` /
  `kurtosis` to 5e-10 and 2e-9 absolute (scipy itself loses digits with this
  offset); `n`, `mean`, `M2` are identical to `chan_merge`; the merge is
  independent of the order of the two accumulators; empty accumulators, `M2 = 0`
  and too small `n` give NaN.
- References: Terriberry, T. B. (2007, updated 2008), Computing Higher-Order
  Moments Online, https://people.xiph.org/~tterribe/notes/homs.html; Pebay, P.
  (2008), Sandia report SAND2008-6212, doi:10.2172/1028931; Pebay, P.,
  Terriberry, T. B., Kolla, H., Bennett, J. (2016), Computational Statistics
  31(4), 1305-1325, doi:10.1007/s00180-015-0637-z.
