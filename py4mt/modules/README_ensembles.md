# ReadMe_ensembles.md

This module (`ensembles.py`) provides utilities for **Gaussian ensemble generation** and **precision / covariance-based sampling** used in FEMTIC-style workflows, in particular sampling from precisions of the form:

\[
Q = R^\top R + \lambda I
\]

where `R` is typically a FEMTIC *roughness* (regularization) operator.

---

## Key features

- Build **matrix-free** precision operators `Q` for large problems using `scipy.sparse.linalg.LinearOperator`.
- Solve `Q x = b` with iterative solvers (`cg`, `bicgstab`, …) and optional preconditioning.
- Generate **Gaussian draws** (e.g., for prior ensembles) driven by `R` / `Q`.
- Optional **automatic diagonal shift** selection (`lam`) to improve conditioning and avoid nullspace issues.

---

## Important update: automatic \(\lambda\) selection

Many roughness operators have a **nullspace** (e.g., constants), making `R.T @ R` singular or poorly conditioned.
To improve robustness, `ensembles.py` now supports an optional rule:

\[
\lambda = \alpha \cdot \mathrm{median}\big(\mathrm{diag}(R^\top R)\big),
\quad \alpha \approx 10^{-6} \ldots 10^{-3}.
\]

### API

The following new options are available (primarily in `make_precision_solver()` and propagated to sampling helpers):

- `lam_mode`: one of  
  - `"fixed"` (default): use the provided `lam`  
  - `"scaled_median_diag"` / `"median_diag"`: compute `lam` from `alpha * median(diag(R.T R))`  
  - `"auto"`: compute from the rule **only if** the provided `lam <= 0`
- `lam_alpha`: the \(\alpha\) factor (recommended range `1e-6 … 1e-3`)
- `lam_statistic`: currently `"median"` (future-proof hook)
- `lam_min`: lower bound to enforce `lam >= lam_min`

A convenience helper is provided:

- `pick_lam_from_rtr_diag(R, alpha=..., statistic="median", min_lam=...)`

---

## Core functions (high level)

### `make_precision_solver(...)`

Constructs a callable `solve_Q(b)` that solves `Q x = b` for multiple right-hand sides, where

- `Q = R.T@R + lam*I` (with optional automatic lam via `lam_mode`).

Key parameters:

- `R`: roughness operator (`numpy.ndarray` or `scipy.sparse` matrix)
- `solver_method` / legacy `msolver`: `"cg"` (default) or `"bicgstab"` etc.
- `precond` / legacy `mprec`: `"jacobi"` (default), `"ilu"`, `"amg"`, or `None`
- tolerances: `rtol`, `atol`, and `maxiter`
- **new**: `lam_mode`, `lam_alpha`, `lam_statistic`, `lam_min`

Example:

```python
from ensembles import make_precision_solver

solve_Q = make_precision_solver(
    R,
    lam=0.0,
    lam_mode="scaled_median_diag",
    lam_alpha=1e-5,      # try 1e-6 … 1e-3
    precond="ilu",       # typically much better than jacobi for large problems
    rtol=1e-4,
    maxiter=20000,
)

x = solve_Q(b)
```

### Sampling helpers

If you use functions that ultimately rely on `make_precision_solver` (for example `sample_rtr_full_rank(...)`),
you can pass the same `lam_mode/lam_alpha/...` arguments to get consistent behaviour.

---

## Notes on solver convergence

If you see:

> `RuntimeError: Iterative solver did not converge within ... iterations`

Typical fixes:

1. Enable/strengthen the diagonal shift: use `lam_mode="scaled_median_diag"` with `lam_alpha=1e-6 … 1e-3`.
2. Use stronger preconditioning (`precond="ilu"` or `precond="amg"` if available).
3. Relax tolerances for sampling (`rtol=1e-3 … 1e-4`) if exact solves are not required.

---

## Dependencies

- `numpy`
- `scipy`
- Optional:
  - `pyamg` (if you choose `precond="amg"`)
  - `sksparse` / CHOLMOD (if you use the CHOLMOD branch for direct solves)

---

## Version / provenance

This README corresponds to the `ensembles.py` found alongside it in this project snapshot.

- Updated: 2026-01-02 (UTC)

