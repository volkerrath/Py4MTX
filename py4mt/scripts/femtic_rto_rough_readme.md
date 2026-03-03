# README\_femtic\_rto\_rough.md

Script for reading and converting a FEMTIC **roughening matrix** to a sparse
`.npz` file, used in the Randomize-Then-Optimize (RTO) workflow.

## Purpose

`femtic_rto_rough.py` is the first step in the RTO pipeline.  It reads the
raw text file `roughening_matrix.out` produced by FEMTIC, parses it into a
sparse matrix `R`, and optionally forms the precision matrix `Q = R^T R`.
The result is saved as a SciPy `.npz` sparse file for efficient re-use by
downstream scripts.

## Workflow position

```text
FEMTIC inversion output
    roughening_matrix.out
            ↓
femtic_rto_rough.py   →   R_coo.npz  or  Q_coo.npz
            ↓
femtic_rto_prior.py   →   prior covariance proxy
femtic_rto_prep.py    →   ensemble generation (if using R directly)
```

## Configuration

| Variable       | Description                                                        |
|----------------|--------------------------------------------------------------------|
| `WORK_DIR`      | Directory containing `roughening_matrix.out`.                      |
| `ROUGH_FILE`    | Full path to the FEMTIC roughening matrix text file.               |
| `OUT_ROUGH`     | Output matrix name: `'R'` saves the roughness, `'Q'` saves `R^T R`. |
| `SPARSE_FORMAT` | Sparse storage format for the output (`'coo'`, `'csr'`, `'csc'`). |

## Behaviour

- If `OUT_ROUGH` contains `'q'` (case-insensitive), the script computes
  `Q = R^T R` and saves the precision matrix.
- Otherwise it saves the roughness matrix `R` directly.
- A sparse-matrix diagnostic is printed via `fem.check_sparse_matrix()`.

## Key function

```python
R = fem.get_roughness(
    filerough=ROUGH_FILE,
    spformat=SPARSE_FORMAT,
    out=True,
)
```

This function (defined in `ensembles.py`) parses the FEMTIC two-pass text
format, builds COO triplets, and converts to the requested sparse format.

## Dependencies

| Package        | Role                                       |
|----------------|--------------------------------------------|
| `numpy`        | Array operations.                          |
| `scipy.sparse` | Sparse matrix construction and I/O.        |
| `femtic`       | `get_roughness()`, `check_sparse_matrix()`. |
| `util`         | Print / version helpers.                   |

## References

See `README_femtic_rto_prep.md` for the full list of RTO references.

## Provenance

| Date       | Author | Change                                       |
|------------|--------|----------------------------------------------|
| 2025-07-24 | vrath  | Created.                                     |
| 2026-03-03 | Claude | Renamed user-set parameters to UPPERCASE.    |

## Author

Volker Rath (DIAS) — July 2025
