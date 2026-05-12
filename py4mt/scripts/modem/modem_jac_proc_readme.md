# modem_jac_proc.py

Read, normalise, mask, and sparsify a ModEM Jacobian matrix.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_jac_proc.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

Primary Jacobian preprocessing script. Reads a raw ModEM Jacobian
(`.jac`), optionally normalises each row by the corresponding data
error, masks air cells, and sparsifies the matrix by zeroing entries
below a threshold. The processed Jacobian is saved in compressed sparse
format for downstream analysis.

## Processing steps

1. **Read model** — load the `.rho` file; identify air and sea cells.
2. **Read Jacobian** — load the `.jac` binary and its companion `_jac.dat` data file.
3. **Error normalisation** (`ERROR_SCALE=True`) — divide each row by its
   data error, producing J̃ = C_d^{-1/2} J.
4. **Air masking** — zero out Jacobian columns corresponding to air cells.
5. **Sparsification** — entries with |J̃_ij|/max < threshold are set to
   zero; the matrix is stored as a scipy sparse matrix.

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`SPARSE_THRESH`, `SPARSE`, `ERROR_SCALE`, `SCALE`, `RHOAIR`, `RHOSEA`, `WORK_DIR`, `J_FILES`, `M_FILE`). |
| **Unused variables** | Removed `rng`, `nan` (never used). |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `ERROR_SCALE` | `True` to normalise by data errors; `False` for raw Jacobian |
| `SPARSE_THRESH` | Sparsification threshold (e.g. `1e-6`); set to 0 for full matrix |
| `WORK_DIR`, `J_FILES`, `M_FILE` | Paths |

## Dependencies

`numpy`, `scipy.sparse`, py4mt: `modem`, `jac_proc`, `util`, `version`.
