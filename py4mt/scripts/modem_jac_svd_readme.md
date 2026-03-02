# modem_jac_svd.py

Compute randomised truncated SVD of a ModEM Jacobian.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_jac_svd.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads a processed (sparse) Jacobian and computes its truncated Singular
Value Decomposition using a randomised algorithm (`jac.rsvd`). Sweeps
over a grid of parameters (rank, oversampling factor, subspace
iterations) and reports the operator-norm accuracy as a percentage of
the full Jacobian norm.

The resulting U, S, V matrices are used downstream for null-space
projection (`modem_insert_multi.py`) and model-space analysis.

## Algorithm

For each parameter combination (k, o, s):

1. Compute U, S, Vt = rSVD(J^T, rank=k, oversamples=o·k, subspace_iters=s).
2. Form the residual D = U·diag(S)·Vt − J^T.
3. Estimate the operator norm via a random probe vector.
4. Report the percentage of the Jacobian's action explained.

## Inputs

| Item | Description |
|------|-------------|
| `JFile` | Base name for `_jac.npz` + `_info.npz` files. |

## Outputs

| File | Contents |
|------|----------|
| `<JFile>_SVD_k<K>_o<O>_s<S>_<pct>percent.npz` | U, S, V matrices and accuracy percentage. One file per parameter combination. |
| `<JFile>_SVD.dat` | Summary table: k, oversampling, subspace_iters, accuracy%, wall time. |

## Configuration

- `NumSingular` — list of ranks to compute (e.g. `[100, 200, 500, 1000]`).
- `OverSample` — list of oversampling factors (multiplied by rank).
- `SubspaceIt` — list of subspace iteration counts.
- `nthreads` — number of BLAS threads (set before numpy import).

## Dependencies

`numpy`, `scipy.sparse`, py4mt: `modem`, `jacproc`, `util`, `version`.
