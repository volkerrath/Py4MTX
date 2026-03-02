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
| `WorkDir`    | Working directory containing the roughness matrix file.               |
| `MatrixIn`   | Base name of the input matrix (`'R'`).                                |
| `FormatIn`   | Sparse format of the input file (`'coo'`).                           |
| `Alpha`      | Regularisation weight; the output is scaled by `1/α²`.               |
| `RegEps`     | Small diagonal shift `ε` added to `R` before inversion.              |
| `Sparsify`   | `[drop_tol, fill_factor]` — controls ILU sparsification.             |
| `MatrixOut`  | Base name for the output matrix (auto-generated from `Sparsify`).    |
| `FormatOut`  | Sparse format for the output (`'csr'`).                              |

## Key function

The heavy lifting is performed by:

```python
fem.make_prior_cov(
    rough=R,
    outmatrix=MatrixOut,
    regeps=RegEps,
    spformat=FormatOut,
    spthresh=Sparsify[0],
    spfill=Sparsify[1],
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

## Author

Volker Rath (DIAS) — July 2025
