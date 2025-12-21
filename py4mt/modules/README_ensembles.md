# Ensemble utilities (`ensembles.py`)

`ensembles.py` is a companion module to `femtic.py` that focuses on **ensemble generation**
and **ensemble analysis** for FEMTIC workflows, including:

- Creating per-ensemble working directories
- Generating **data ensembles** (perturbed `observe.dat`)
- Generating **model ensembles** (perturbed resistivity blocks) via roughness/precision sampling
- Ensemble analysis: **EOFs/PCA**
- Weighted **KL decomposition**
- A practical **KL + PCE surrogate** builder for log-resistivity fields on FEMTIC meshes

The module is designed to be imported (no side effects on import).

---

## Relationship to `femtic.py`

`ensembles.py` calls several functions from `femtic.py` (imported as `femtic` / `fem`).

Make sure that either:

- both files live in the same directory and you run Python from there, **or**
- your project is a package and you import via package name, **or**
- `femtic.py` is on your `PYTHONPATH`.

---

## Requirements

Minimum:

- Python 3.10+
- `numpy`
- `scipy`

Optional:

- `joblib` (imported; used in some environments for parallel patterns)

---

## Directory/ensemble helpers

### Create a directory structure

```python
from ensembles import generate_directories

dirs = generate_directories(dir_base="./ens_", n_samples=50, out=True)
# creates: ./ens_0, ./ens_1, ..., ./ens_49
```

### Copy template files into each ensemble directory

```python
from ensembles import copy_files

copy_files(
    dir_base="./ens_",
    n_samples=50,
    files=["observe.dat", "mesh.dat", "resistivity_block_iter0.dat"],
)
```

---

## Data ensemble: perturb `observe.dat`

```python
from ensembles import generate_data_ensemble

obs_files = generate_data_ensemble(
    dir_base="./ens_",
    n_samples=50,
    file_in="observe.dat",
    draw_from=("normal", 0.0, 1.0),
    method="add",
    out=True,
)
```

This function:

1. expects `./ens_i/observe.dat` to exist for each `i`
2. makes `observe_orig.dat` backups
3. perturbs in-place using `femtic.modify_data(...)`

---

## Model ensemble: perturb resistivity blocks

### Full-rank sampling from a supplied precision/roughness matrix

```python
import scipy.sparse as sp
from ensembles import generate_model_ensemble

# q should represent a roughness matrix R (or a precision factor you use consistently)
# Example: q = sp.load_npz("roughness_R.npz")
q = sp.load_npz("roughness_R.npz")

models = generate_model_ensemble(
    dir_base="./ens_",
    n_samples=50,
    refmod="resistivity_block_iter0.dat",
    q=q,
    method="add",
    out=True,
)
```

This function writes a perturbed resistivity block into each `./ens_i/` directory
based on sampling from the implied Gaussian prior.

### Low-rank sampling workflow (when available)

`ensembles.py` also provides utilities for estimating low-rank eigenspaces and
sampling in that subspace:

- `estimate_low_rank_eigpairs(...)`
- `sample_rtr_low_rank(...)`

Use these when your number of cells is very large and you want faster approximate sampling.

---

## EOFs / PCA from an ensemble matrix

`compute_eofs(E, method=...)` assumes an ensemble matrix `E` with shape:

- `(ncells, nsamples)` where each **column** is one sample.

Two options are implemented:

- `method="svd"` (default): SVD-based EOFs
- `method="sample_space"`: eigen-decomposition in sample space (efficient when `nsamples << ncells`)

Example:

```python
import numpy as np
from ensembles import compute_eofs

# E shape (ncells, nsamples)
E = np.random.randn(100_000, 100)

eofs, pcs, evals, frac, mean = compute_eofs(E, k=10, method="svd")
```

Returned arrays:

- `eofs`: spatial EOF patterns, shape `(ncells, k)`
- `pcs`: principal components per sample, shape `(k, nsamples)`
- `evals`: eigenvalues (variance), shape `(k,)`
- `frac`: explained variance fraction, shape `(k,)`
- `mean`: removed mean field, shape `(ncells,)`

---

## Weighted KL decomposition

`compute_weighted_kl(...)` supports weighting (e.g., element volumes) in the inner product.

Typical use:

```python
from ensembles import compute_weighted_kl

# fields shape (ncells, nsamples)
# weights shape (ncells,) e.g. element volumes
Phi, scores, evals, mean = compute_weighted_kl(fields, weights, k=20)
```

---

## KL + PCE surrogate (non-intrusive)

For surrogate modelling, the module contains:

- multi-index generation (`total_degree_multiindex`)
- orthonormal Hermite/Legendre polynomials
- design matrix construction (`build_design_matrix`)
- per-mode PCE fitting (`fit_pce_for_kl_modes`)
- a compact model container: `KLPCEModel` with:
  - `fit_kl_pce_model(...)`
  - `evaluate_kl_pce_surrogate(...)`

High-level pattern:

```python
from ensembles import fit_kl_pce_model, evaluate_kl_pce_surrogate

model = fit_kl_pce_model(
    fields=fields,         # (ncells, nsamples)
    xi=xi,                 # (nsamples, ndims) input random variables
    weights=weights,       # (ncells,)
    k_kl=20,
    pce_degree=3,
)

# Evaluate surrogate at new random inputs xi_new (n_new, ndims)
fields_pred = evaluate_kl_pce_surrogate(model, xi_new)
```

---

## Precision solvers / preconditioning (advanced)

Sampling and linear solves involving `RᵀR` can be configured via:

- `make_rtr_preconditioner(...)`
- `make_sparse_cholesky_precision_solver(...)` (if sparse factorization is available)
- `make_precision_solver(...)` (dispatches between solvers)
- `sample_rtr_full_rank(...)`, `sample_rtr_low_rank(...)`

These are intended for large-scale problems where direct dense factorization is impossible.

---

## License / attribution

If you redistribute, please keep module headers/docstrings intact.

Author: Volker Rath (DIAS)  
Created by ChatGPT (GPT-5 Thinking) on 2025-12-21 (UTC)
