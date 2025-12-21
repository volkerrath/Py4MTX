# ReadMe_jacproc.md

This document describes the **Jacobian/sensitivity post-processing module**:

- `jacproc_updated.py` (rename to `jacproc.py` in your project if desired)

The module focuses on:
- **Sensitivity measures** derived from an (error-scaled) Jacobian.
- **Sparsification** for large Jacobians.
- **Streaming statistics** (online mean/variance).
- **Low-rank approximations** via randomized SVD (RSVD).

## Core functions

### Sensitivities
- `calc_sensitivity(Jac, Type='euclidean', ...)`
  - `Type='raw'`  : sum along data axis
  - `Type='cov'`  : sum of absolute values (often called “coverage”)
  - `Type='euc'`  : Euclidean / L2-type aggregation
  - `Type='cum'`  : cumulative sensitivity (Christiansen & Auken, 2012)

- `transform_sensitivity(S, Siz, Transform='max', ...)`
  - Supports transforms such as `max`, `siz` (size/volume normalisation), `log`, and `asinh`.

### Sparsification / scaling
- `normalize_jac(Jac, fn)` scales Jacobians either globally or via a left-diagonal.
- `sparsify_jac(Jac, sparse_thresh, normalized, ...)` thresholds small entries and returns a CSR matrix.

### Streaming statistics
- `update_avg(k, m_k, m_a, m_v)` updates running mean and variance accumulator.

### Low-rank approximation
- `rsvd(A, rank=..., n_subspace_iters=...)` provides a truncated randomized SVD.
  This is useful if you want a low-rank surrogate of a huge Jacobian.

## Dependencies

- `numpy`
- `scipy.sparse`
- `numba` (optional)

## Quick examples

### 1) Sensitivity from a sparse Jacobian

```python
import scipy.sparse as sp
from jacproc_updated import calc_sensitivity

# Jac is assumed to be *error-scaled* already
Jac = sp.random(1000, 20000, density=1e-4, format="csr")
S = calc_sensitivity(Jac, Type="euc")
```

### 2) Sparsify a dense Jacobian

```python
import numpy as np
from jacproc_updated import sparsify_jac

Jac = np.random.randn(2000, 5000)
Js, scale = sparsify_jac(Jac, sparse_thresh=1e-6, normalized=True, scalval=-1.0)
```

### 3) Low-rank approximation

```python
import numpy as np
from jacproc_updated import rsvd

A = np.random.randn(2000, 5000)
U, S, Vt = rsvd(A, rank=50, n_subspace_iters=1)
```

## Notes

Most functions assume your Jacobian is already **scaled** as:

\[
J_\mathrm{scaled} = C^{-1/2} J
\]

where \(C\) is the data covariance (or diagonal variance) matrix. That is consistent with the sensitivity
definitions implemented here.
