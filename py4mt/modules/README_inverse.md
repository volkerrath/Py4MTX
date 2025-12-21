# inverse.py

`inverse.py` collects numerical routines useful in inverse problems and ensemble-based workflows.

This README documents the updated drop-in module `inverse_updated.py`.

---

## Functions

### TV / Split Bregman
- `soft_thresh(x, lam)`
- `splitbreg(J, y, lam, D, c=0, ...)`

### Covariance estimation
- `calc_covar_simple(x, y=None, ...)`
- `calc_covar_nice(x, y, fac, ...)` (NICE shrinkage; Vishny et al., 2024)

### SPD factors / square-roots
- `msqrt_sparse(M, method="chol"|"eigs"|"splu", ...)`
- `isspd(A)`

### Randomised SVD
- `rsvd`, `find_range`, `subspace_iter`, `ortho_basis`

### Spline utilities
- `make_spline`, `estimate_variance`, `bootstrap_confidence_band`

---

## Minimal examples

### Split Bregman TV
```python
import numpy as np
import scipy.sparse as sp
from inverse_updated import splitbreg

nd, nm = 200, 100
J = np.random.randn(nd, nm)
y = np.random.randn(nd)
D = sp.diags([1, -1], [0, 1], shape=(nm-1, nm), format="csr")

x = splitbreg(J, y, lam=1.0, D=D, maxiter=50)
```

### NICE covariance
```python
import numpy as np
from inverse_updated import calc_covar_nice

Ne, Nx, Ny = 50, 500, 200
X = np.random.randn(Ne, Nx)
Y = np.random.randn(Ne, Ny)

Cov, Corr, L = calc_covar_nice(X, Y, fac=1.0)
```

---

## References

- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). SIAM Review.
- Vishny, D., Morzfeld, M., Gwirtz, K., Bach, E., Dunbar, O. R. A., & Hodyss, D. (2024).
  JAMES. doi:10.1029/2024MS004417

Author: Volker Rath (DIAS)  
Created by ChatGPT (GPT-5 Thinking) on 2025-12-21 (UTC)
