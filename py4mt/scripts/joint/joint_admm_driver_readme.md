# `joint_admm_driver.py` — ADMM Outer Loop

ADMM driver for joint MT + seismic tomography inversion.

---

## Purpose

Implements the ADMM outer loop that alternates between physics-specific
model updates and a consensus/latent variable update. The coupling strategy
is entirely decoupled from the loop: any object exposing `update_z` and
optionally `report` is accepted.

---

## Public API

### `admm_joint_mt_seis`

```python
result = admm_joint_mt_seis(
    d_mt, d_sv,
    Wd_mt, Wd_sv,
    Wm_mt, Wm_sv,
    m_mt0, m_sv0,
    m_mt_ref, m_sv_ref,
    coupling,
    *,
    alpha_mt=1.0, alpha_sv=1.0,
    rho_mt=1.0,   rho_sv=1.0,
    max_outer=50,
    tol_primal=1e-3, tol_dual=1e-3,
    fix_method=None,
    verbose=True,
    solve_m_mt=None, solve_m_sv=None,
    apply_Gt_mt=None, apply_Gt_sv=None,
)
```

#### Required positional arguments

| Argument | Type | Description |
|---|---|---|
| `d_mt`, `d_sv` | ndarray | Observed data vectors |
| `Wd_mt`, `Wd_sv` | operator | Data-weighting operators |
| `Wm_mt`, `Wm_sv` | operator | Tikhonov regularisation operators |
| `m_mt0`, `m_sv0` | ndarray (N,) | Initial model vectors |
| `m_mt_ref`, `m_sv_ref` | ndarray (N,) | Reference models |
| `coupling` | object | Coupling strategy (see below) |

#### Keyword arguments

| Argument | Default | Description |
|---|---|---|
| `alpha_mt`, `alpha_sv` | 1.0 | Regularisation weights |
| `rho_mt`, `rho_sv` | 1.0 | ADMM penalty parameters |
| `max_outer` | 50 | Maximum ADMM iterations |
| `tol_primal`, `tol_dual` | 1e-3 | Stopping tolerances |
| `fix_method` | `None` | `"mt"` or `"sv"` to hold one method fixed |
| `verbose` | `True` | Print per-iteration diagnostics |
| `solve_m_mt`, `solve_m_sv` | `None` | Physics solvers; signature `solve(rhs, Wd, Wm, alpha, rho) -> m` |
| `apply_Gt_mt`, `apply_Gt_sv` | `None` | Adjoint operators; signature `apply_Gt(v) -> ndarray` |

#### Returns

`dict` with keys `m_mt`, `m_sv`, `z`, `y_mt`, `y_sv`, `n_iter`,
`converged`.

#### Raises

`ValueError` if `fix_method` is not `None`, `"mt"`, or `"sv"`.

---

## Coupling interface

The `coupling` argument must expose:

```python
coupling.update_z(m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv) -> ndarray
```

And optionally:

```python
coupling.report(m_mt, m_sv, z) -> dict   # printed when verbose=True
```

Ready-made wrappers are in `joint_coupling.py`.

---

## ADMM iteration structure

```
for it in range(max_outer):
    1. MT model update     (skipped if fix_method="mt")
    2. Seismic model update (skipped if fix_method="sv")
    3. z update            coupling.update_z(...)
    4. Dual update         y_mt += rho_mt * (m_mt - z)
                           y_sv += rho_sv * (m_sv - z)
    5. Stopping check      ||r|| < tol_primal and ||s|| < tol_dual
```

where `||r||² = ||m_mt - z||² + ||m_sv - z||²` (primal residual) and
`||s|| = rho_mt * ||z - z_old||` (dual residual).

---

## Authorship

Volker Rath (DIAS) — 2026-05-12, Claude Sonnet 4.6 (Anthropic)
