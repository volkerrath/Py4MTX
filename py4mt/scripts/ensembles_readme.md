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
- **Gaussian sampling** from `N(0, Q^{-1})` — full-rank iterative and
  low-rank randomized SVD paths.
- **File-system helpers** — create per-member ensemble directories, copy
  template FEMTIC input files.
- **Data and model ensemble generators** — perturb `observe.dat` and
  `resistivity_block_iterX.dat` for RTO and related algorithms.
- **Geostatistical initial-model ensemble** (`generate_gst_model_ensemble`) —
  pilot-point Ordinary Kriging (gstools) for the GST uncertainty method.
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
| `generate_data_ensemble()`       | Perturb `observe.dat` in each member directory.       |
| `generate_rto_model_ensemble()`  | Sample from roughness precision and write perturbed models (RTO). |
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
| `sample_rtr_full_rank()`             | Sample `N(0, Q^{-1})` via full-rank iterative solves. |
| `sample_rtr_low_rank()`              | Approximate sampling via randomized SVD of R.        |
| `sample_prior_from_roughness()`      | Unified interface (full or low-rank).                |

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
2. Draw log₁₀(ρ) values at pilot points from Uniform(log_rho_min, log_rho_max).
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
| `vario_model`    | gstools covariance class: `"Spherical"` (default), `"Gaussian"`, `"Exponential"`, etc.  |
| `vario_range`    | Correlation length (m).  Scalar = isotropic; 2-tuple `(h, v)` = geometric anisotropy.   |
| `vario_sill`     | Sill (variance) in (log₁₀ Ω·m)².  Recommended: 0.25–0.5.                               |
| `vario_nugget`   | Nugget in (log₁₀ Ω·m)².  Keep ≤ 10 % of sill.                                          |
| `vario_angles`   | Rotation `[α, β, γ]` in degrees; `None` = axis-aligned.                                 |
| `output_target`  | `"resistivity_block"` / `"referencemodel"` / `"both"` (recommended).                   |

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

mod_list = generate_gst_model_ensemble(
    dir_base   = "./ubinas_gst_",
    n_samples  = 32,
    ref_mod_file = "./templates/referencemodel.dat",
    pp_mode    = "random",
    n_pp       = 100,
    pp_bbox    = [-50000, 50000, -50000, 50000, 0, 80000],
    log_rho_min = 0.0,
    log_rho_max = 4.0,
    vario_model = "Spherical",
    vario_range = (20000., 5000.),
    vario_sill  = 0.5,
    vario_nugget = 0.01,
    output_target = "both",
)
```

---

## Low-rank sampling (randomized SVD)

`sample_rtr_low_rank` now uses **randomized SVD** of R directly
(via `sklearn.utils.extmath.randomized_svd`) instead of the former
`eigsh`-based smallest-eigenvalue approach.

| Parameter        | Description                                                         |
|------------------|---------------------------------------------------------------------|
| `n_eig`          | Number of singular triplets retained (rank of approximation).       |
| `n_oversampling` | Extra columns in the randomized range-finder (default 10).          |
| `n_power_iter`   | Power-iteration steps to sharpen accuracy (default 2).              |
| `sigma2_residual`| Isotropic residual variance added per sample (default 0).           |

Key advantages over the old `eigsh` approach:

- Converges in O(n_eig) matvec passes with R and R^T — no SM eigenvalue
  convergence problem.
- Does not require forming Q = R^T R explicitly.
- Suitable for 4–100+ samples at typical FEMTIC mesh sizes.

`generate_rto_model_ensemble` now accepts **R directly** (not Q).
Q = R^T R is formed implicitly inside the sampling routines.

## Quick-reference: recommended parameter values (FEMTIC)

| Parameter | Recommended | Location |
|-----------|-------------|----------|
| `algo` | `"low rank"` | `generate_rto_model_ensemble` |
| `n_eig` | 128–256 | low-rank branch |
| `n_power_iter` | 3–4 | low-rank branch |
| `sigma2_residual` | 1e-3 | low-rank branch |
| `lam_mode` | `"scaled_median_diag"` | full-rank branch |
| `lam_alpha` | 1e-4 (raise to 1e-3 if slow) | full-rank branch |
| `precond` | `"ilu"` | full-rank branch |
| `solver_method` | `"cg"` | full-rank branch |
| `rtol` | 1e-2 (sampling) | `make_precision_solver` |
| `maxiter` | 500–1000 | `make_precision_solver` |

## Solver performance and parameter guidance

### Choosing the sampling algorithm

**Low-rank (`algo="low rank"`) is the recommended default.** It performs O(n_eig)
matvec passes with R and R^T and is typically 10–100× faster than full-rank CG for
the ensemble sizes and mesh dimensions typical in FEMTIC.  Full-rank CG is provided
for accuracy verification.

| Scenario | Recommended |
|----------|-------------|
| Normal production run | `"low rank"`, `n_eig=128–256` |
| Accuracy / distribution check | `"full rank"`, `precond="ilu"`, large `lam_alpha` |
| Very large mesh (>200k cells) | `"low rank"`, `n_eig=64–128`, `n_power_iter=4` |

### Low-rank parameter recommendations

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `n_eig` | 64 | **128–256** | More modes → smoother, more faithful samples; cost is linear in n_eig |
| `n_oversampling` | 10 | 10–15 | Rarely needs increasing beyond 15 |
| `n_power_iter` | 2 | **3–4** | FEMTIC roughness spectra decay slowly; 3–4 iterations sharpen the range-finder |
| `sigma2_residual` | 0.0 | **1e-4 – 1e-3** | Without residual variance samples live in the rank-k subspace only; a small isotropic residual restores short-wavelength variability (~10% of typical log10ρ variance) |

### Full-rank CG parameter recommendations

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `lam_alpha` | 1e-5 | **1e-4 – 1e-3** | Single biggest speed lever: larger shift dramatically improves CG conditioning |
| `lam_mode` | `"scaled_median_diag"` | keep | Auto diagonal shift is the correct default |
| `precond` | `"ilu"` | **keep `"ilu"`** | Best for SPD Q; Jacobi is cheaper to set up but needs 3–5× more CG iterations |
| `solver_method` | `"cg"` | **keep `"cg"`** | Q = R^T R is SPD so CG is optimal; BiCGSTAB gives no benefit |
| `rtol` | 1e-3 | **1e-2 for sampling** | Monte Carlo samples don't need high solve accuracy; 1e-2 often halves iteration count |
| `maxiter` | None | **500–1000** | Prevents runaway solves; with good λ + ILU convergence should be <200 iterations |

## Solver convergence tips

If you see `RuntimeError: Iterative solver did not converge` (full-rank mode):

1. Increase the diagonal shift: `lam_mode="scaled_median_diag"`, raise `lam_alpha` to 1e-4 or 1e-3.
2. Use stronger preconditioning: `precond="ilu"` or `precond="amg"`.
3. Relax tolerances for sampling: `rtol=1e-2`.
4. Switch to `"low rank"` mode — it avoids the convergence issue entirely.

---

## Dependencies

| Package             | Role                                        |
|---------------------|---------------------------------------------|
| `numpy`             | Core array operations.                      |
| `scipy`             | Sparse matrices, iterative solvers, SVD.    |
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

## Version / provenance

Updated: 2026-04-27 (GST); 2026-04-11 (consolidation); 2026-04-02 (cleanup)

### Changelog (2026-04-27)
- Added `generate_gst_model_ensemble` — geostatistical initial-model ensemble
  via pilot-point Ordinary Kriging (gstools).  No roughness matrix required.
  Supports `pp_mode = "random" | "fixed" | "mixed"`, fully configurable
  variogram, and `output_target = "resistivity_block" | "referencemodel" | "both"`.
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
