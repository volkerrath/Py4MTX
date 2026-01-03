# FEMTIC resistivity-block model workflow (read → NPZ → modify → write)

This refactor introduces a clean 3-step workflow for FEMTIC resistivity-block models
(``resistivity_block_iterX.dat``):

1. **Read** FEMTIC model → **NPZ** (self-contained, includes mapping, region table, fixed mask)
2. **Modify** the NPZ (updates **free regions only**, fixed regions preserved)
3. **Write** FEMTIC model from NPZ (+ optional updated NPZ copy)

The workflow is implemented in `femtic.py` via:

- `read_model_to_npz(...)`
- `modify_model_npz(...)`
- `write_model_from_npz(...)`

## Fixed regions and sampling compatibility

The NPZ stores:

- `fixed_mask` (boolean per region)
- `free_idx` (indices of regions that may change)
- `model_free` (free-vector in `model_trans` space, default: log10(ρ))

Fixed regions are defined as:

- region 0 (air) always fixed
- any region with `flag == 1` fixed
- region 1 treated as ocean-fixed if `ocean_present` is true

When using precision-matrix sampling, `modify_model_npz(method='precision_sample')`:

- accepts `R` directly or reads `roughening_matrix.out` via `get_roughness()`
- supports both `R.shape[1] == n_free` and `R.shape[1] == nreg` (auto-slices to `free_idx`)
- forwards `lam_mode`, `lam_alpha`, `lam_statistic`, `lam_min` to the sampler in `ensembles.py`

## Quick examples

### 1) Read model → NPZ

```python
from femtic import read_model_to_npz

read_model_to_npz(
    model_file="resistivity_block_iter0.dat",
    npz_file="model_iter0.npz",
    model_trans="log10",
)
```

### 2a) Modify by adding Gaussian noise (log10-space)

```python
from femtic import modify_model_npz

modify_model_npz(
    npz_in="model_iter0.npz",
    npz_out="model_iter0_noisy.npz",
    method="add_noise",
    add_sigma=0.05,   # log10(ρ) std-dev
)
```

### 2b) Modify by precision-matrix sampling (uses roughness matrix)

```python
from femtic import modify_model_npz

modify_model_npz(
    npz_in="model_iter0.npz",
    npz_out="model_iter0_prior_draw.npz",
    method="precision_sample",
    roughness="roughening_matrix.out",
    lam_mode="scaled_median_diag",
    lam_alpha=1e-5,
    solver_method="cg",
    scale=1.0,
    add_to_current=True,
)
```


### 2c) Precision sampling with preconditioning (recommended)

`solver_method="cg"` uses iterative solves of `Qx=b`. You can enable preconditioning via
either:

- `precond="jacobi" | "ilu" | "amg" | "identity" | None` (convenience args), or
- `solver_kwargs={"precond": "...", "precond_kwargs": {...}, "rtol": ..., "maxiter": ...}`.

Notes:
- `"jacobi"` does **not** form `Q` explicitly and is the safest default.
- `"ilu"` and `"amg"` require **sparse** `R` (because they build sparse `Q = R.T@R`).

```python
from femtic import modify_model_npz

modify_model_npz(
    npz_in="model_iter0.npz",
    npz_out="model_iter0_prior_draw_pcg.npz",
    method="precision_sample",
    roughness="roughening_matrix.out",
    lam_mode="scaled_median_diag",
    lam_alpha=1e-5,
    solver_method="cg",
    precond="jacobi",
    scale=1.0,
)
```

### 3) Write FEMTIC model from NPZ

```python
from femtic import write_model_from_npz

write_model_from_npz(
    npz_file="model_iter0_prior_draw.npz",
    model_file="resistivity_block_iter0_new.dat",
    also_write_npz="resistivity_block_iter0_new.npz",
)
```

