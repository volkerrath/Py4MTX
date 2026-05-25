# inverse.py

`inverse.py` collects numerical routines useful in inverse problems and
ensemble-based workflows, together with a deterministic 1-D Gauss–Newton
inversion for the simplified anisotropic MT model.

This README documents the merged drop-in module `inverse.py`, which
consolidates the former standalone `inverse.py` (general numerical utilities)
and `inv1d.py` (1-D MT inversion helpers) into a single file.

---

## Part A — General numerical routines

### TV / Split Bregman
- `soft_thresh(x, lam)`
- `splitbreg(J, y, lam, D, c=0, ...)`

### Covariance estimation
- `calc_covar_simple(x, y=None, ...)` — empirical cross-covariance
- `calc_covar_nice(x, y, fac, ...)` — NICE shrinkage (Vishny et al., 2024)

### SPD factors / square-roots
- `msqrt_sparse(M, method="chol"|"eigs"|"splu", ...)`
- `isspd(A)`

### Randomised SVD
- `rsvd`, `find_range`, `subspace_iter`, `ortho_basis`

### Spline utilities
- `make_spline`, `estimate_variance`, `bootstrap_confidence_band`

---

## Part B — Deterministic 1-D MT inversion

### Infrastructure
- `ensure_dir(path)` — create directory if needed
- `glob_inputs(pattern)` — sorted file list from glob

### Model parameterization and I/O
- `model_from_direct(model_direct)` — validate an inline model dict
- `normalize_model(model)` — normalize to (rho\_min, rho\_max, sigma\_min, sigma\_max, strike\_deg)
- `save_model_npz(model, path)`, `load_model_npz(path)`
- `load_site(path)` — load site data from `.npz` or `.edi`

### Phase tensor
- `ensure_phase_tensor(site)` — compute P = inv(Re(Z)) @ Im(Z); optionally bootstrap P\_err

### Impedance data packing
- `_pack_Z_obs(Z, Z_err, ...)` — complex impedance → real observation vector + sigma
- `_pack_Z_jac(dZ, ...)` — complex sensitivity → real Jacobian column

### Parameter spec
- `ParamSpec(nl, ...)` — bounds and masks for inversion; supports `param_domain` ∈ {rho, sigma} and `param_set` ∈ {minmax, max\_anifac}

### Gauss–Newton solver
- `invert_site(site, spec, model0, ...)` — single-site inversion with Tikhonov (ridge) or TSVD

### Result I/O
- `save_inversion_npz(res, path)`

---

## Minimal examples

### Split Bregman TV
```python
import numpy as np
import scipy.sparse as sp
from inverse import splitbreg

nd, nm = 200, 100
J = np.random.randn(nd, nm)
y = np.random.randn(nd)
D = sp.diags([1, -1], [0, 1], shape=(nm-1, nm), format="csr")

x = splitbreg(J, y, lam=1.0, D=D, maxiter=50)
```

### NICE covariance
```python
import numpy as np
from inverse import calc_covar_nice

Ne, Nx, Ny = 50, 500, 200
X = np.random.randn(Ne, Nx)
Y = np.random.randn(Ne, Ny)

Cov, Corr, L = calc_covar_nice(X, Y, fac=1.0)
```

### 1-D MT inversion
```python
from inverse import load_site, normalize_model, ParamSpec, invert_site

site = load_site("site001.npz")
model0 = normalize_model({
    "h_m":        [500, 1000, 2000],
    "rho_min":    [100, 100, 100],
    "rho_max":    [100, 500, 100],
    "strike_deg": [0, 30, 0],
})

spec = ParamSpec(nl=3, param_domain="rho", param_set="minmax")
result = invert_site(site, spec=spec, model0=model0, method="tikhonov", lam=1.0)
```

---

## Part C — Routines migrated from util.py

These functions lived in `util.py` but belong here conceptually.


### KL divergence
- `KLD(P, Q, epsilon=1e-8)` — Kullback-Leibler distance between two PDFs

### N-D discrete cosine transform
- `dctn(x, normused='ortho')` — N-D forward DCT (type II, all axes)
- `idctn(x, normused='ortho')` — N-D inverse DCT (type II, all axes)
- `fractrans(m, x, a=0.5)` — fractional derivative of order *a* (requires `differint`)

### L-curve corner estimation
- `calc_lc_corner(dnorm, mnorm)` — corner by maximum curvature in log-log space
- `curvature(x_data, y_data)` — κ at every interior point
- `circumradius(xvals, yvals)` — circumradius of three 2-D points
- `circumcenter(xvals, yvals)` — circumcenter of three 2-D points

### Residual norms
- `calc_resnorm(data_obs, data_calc, data_std, p=2)` — *p*-norm of weighted residuals
- `calc_rms(dcalc, dobs, Wd=1.0)` — NRMS and SRMS misfit metrics


---

## References

- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). SIAM Review.
- Vishny, D., Morzfeld, M., Gwirtz, K., Bach, E., Dunbar, O. R. A., & Hodyss, D. (2024).
  JAMES. doi:10.1029/2024MS004417
- Hansen, P. C. (2010). *Discrete Inverse Problems: Insight and Algorithms*. SIAM, Philadelphia.
- Hansen, P. C. (2001). The L-Curve and its Use in the Numerical Treatment of Inverse Problems.
  In P. Johnston (Ed.), *Computational Inverse Problems in Electrocardiology* (pp. 119–142). WIT Press.
- Hansen, P. C. (1998). *Rank Deficient and Discrete Ill-Posed Problems*. SIAM, Philadelphia.

Author: Volker Rath (DIAS)
Original numerical utilities created with GPT-5 Thinking on 2025-12-21 (UTC).
1-D inversion helpers created with GPT-5 Thinking on 2026-02-13 (UTC).
Cleaned up and merged by Claude (Anthropic, Opus 4.6) on 2026-03-02.
Part C migrated from util.py by Claude Sonnet 4.6 (Anthropic) on 2026-05-25.
