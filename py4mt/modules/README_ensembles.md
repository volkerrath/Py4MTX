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
| `sample_rtr_low_rank()`              | Approximate sampling via `k` eigenpairs.             |
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

## Solver convergence tips

If you see `RuntimeError: Iterative solver did not converge`:

1. Enable a diagonal shift: `lam_mode="scaled_median_diag"`, `lam_alpha=1e-6 … 1e-3`.
2. Use stronger preconditioning: `precond="ilu"` or `precond="amg"`.
3. Relax tolerances for sampling: `rtol=1e-3 … 1e-4`.

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

Updated: 2026-03-01

Author: Volker Rath (DIAS)
