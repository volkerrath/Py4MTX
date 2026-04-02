# README\_ensembles.md

Module `ensembles.py` — high-level ensemble utilities for FEMTIC workflows.

---

## Overview

This module provides:

- **File-system helpers** for creating per-member ensemble directories and
  copying template FEMTIC input files.
- **Data and model ensemble generators** that perturb `observe.dat` and
  `resistivity_block_iterX.dat` files for RTO and related algorithms.
- **Roughness / precision matrix I/O** — read FEMTIC `roughening_matrix.out`
  and build sparse `R` or `Q = R^T R`.
- **Prior covariance proxy** (`make_prior_cov`) via approximate sparse
  inversion of the roughness matrix.
- **Gaussian sampling** from `N(0, Q^{-1})` where `Q = R^T R + λ I`, using
  iterative (CG, BiCGSTAB) or direct (Cholesky / LU) solvers with optional
  preconditioning.
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

| Function                    | Purpose                                               |
|-----------------------------|-------------------------------------------------------|
| `generate_directories()`    | Create numbered member directories, copy templates.   |
| `generate_data_ensemble()`  | Perturb `observe.dat` in each member directory.       |
| `generate_model_ensemble()` | Sample from precision and write perturbed models.     |

### Roughness / precision matrix

| Function              | Purpose                                                    |
|-----------------------|------------------------------------------------------------|
| `get_roughness()`     | Parse `roughening_matrix.out` → sparse `R`.                |
| `make_prior_cov()`    | Approximate `M ≈ (R + εI)^{-1} (R + εI)^{-T}` proxy.     |
| `matrix_reduce()`     | Sparsify a dense or sparse matrix by dropping small entries. |

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

`generate_model_ensemble` now accepts **R directly** (not Q).
Q = R^T R is formed implicitly inside the sampling routines.

## Quick-reference: recommended parameter values (FEMTIC)

| Parameter | Recommended | Location |
|-----------|-------------|----------|
| `algo` | `"low rank"` | `generate_model_ensemble` |
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

---

## Version / provenance

Updated: 2026-04-02 (code); 2026-04-02 (cleanup)

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
