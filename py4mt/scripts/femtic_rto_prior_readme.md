# README\_femtic\_rto\_prior.md

Script for building a **prior covariance proxy** from a FEMTIC roughness matrix,
used in the Randomize-Then-Optimize (RTO) uncertainty-quantification workflow.

## Purpose

`femtic_rto_prior.py` reads a sparse roughness (regularisation) matrix `R` that
was previously extracted from FEMTIC output (via `femtic_rto_rough.py`), and
computes an approximate inverse covariance:

```
M ≈ α² · (R + ε I)⁻¹ · (R + ε I)⁻ᵀ
```

The resulting matrix `M` serves as a prior covariance proxy for sampling or
analysis steps in the RTO pipeline.  After construction, `M` is optionally
sparsified and saved as a compressed `.npz` file.

## Workflow position

```text
femtic_rto_rough.py   →   R_coo.npz  (roughness matrix)
                                 ↓
femtic_rto_prior.py   →   invRTR_*.npz  (prior covariance proxy)
                                 ↓
femtic_rto_prep.py    →   ensemble directories
```

## Configuration

| Variable     | Description                                                           |
|--------------|-----------------------------------------------------------------------|
| `WORK_DIR`    | Working directory containing the roughness matrix file.               |
| `MATRIX_IN`   | Base name of the input matrix (`'R'`).                                |
| `FORMAT_IN`   | Sparse format of the input file (`'coo'`).                           |
| `ALPHA`      | Regularisation weight; the output is scaled by `1/α²`.               |
| `REG_EPS`     | Small diagonal shift `ε` added to `R` before inversion.              |
| `SPARSIFY`   | `[drop_tol, fill_factor]` — controls ILU sparsification.             |
| `MATRIX_OUT`  | Base name for the output matrix (auto-generated from `SPARSIFY`).    |
| `FORMAT_OUT`  | Sparse format for the output (`'csr'`).                              |

## Key function

The heavy lifting is performed by:

```python
fem.make_prior_cov(
    rough=R,
    outmatrix=MATRIX_OUT,
    regeps=REG_EPS,
    spformat=FORMAT_OUT,
    spthresh=SPARSIFY[0],
    spfill=SPARSIFY[1],
    spsolver='ilu',
    spmeth='basic,area',
    nthreads=n_threads,
    out=True,
)
```

This function (defined in `ensembles.py` / `femtic.py`):

1. Adds `ε I` to the roughness matrix for numerical stability.
2. Computes an approximate inverse using incomplete LU (ILU).
3. Optionally drops small entries to maintain sparsity.
4. Forms `M = invR @ invR.T` (if the output name contains `"rtr"`).

## Dependencies

| Package        | Role                                          |
|----------------|-----------------------------------------------|
| `numpy`        | Array operations.                             |
| `scipy.sparse` | Sparse matrix I/O and arithmetic.             |
| `femtic`       | `make_prior_cov()` and sparse-matrix helpers. |
| `util`         | Print / version helpers.                      |

## References

See `README_femtic_rto_prep.md` for the full list of RTO references.

## Provenance

| Date       | Author | Change                                       |
|------------|--------|----------------------------------------------|
| 2025-07-24 | vrath  | Created.                                     |
| 2026-03-03 | Claude | Renamed user-set parameters to UPPERCASE.    |

## Author

Volker Rath (DIAS) — July 2025
