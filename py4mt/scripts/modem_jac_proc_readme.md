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

## Purpose

Primary Jacobian preprocessing script. Reads a raw ModEM Jacobian
(`.jac`), optionally normalises each row by the corresponding data
error, masks air cells, and sparsifies the matrix by zeroing entries
below a threshold. The processed Jacobian is saved in compressed sparse
format for downstream analysis.

## Processing steps

1. **Read model** — load the `.rho` file; identify air and sea cells.
2. **Read Jacobian** — load the `.jac` binary and its companion `_jac.dat` data file.
3. **Error normalisation** (`ErrorScale=True`) — divide each row by its
   data error, producing J̃ = C_d^{-1/2} J.
4. **Air masking** — zero out Jacobian columns corresponding to air cells.
5. **Sparsification** — entries with |J̃_ij|/max < threshold are set to
   zero; the matrix is stored as a scipy sparse matrix.

## Inputs

| Item | Description |
|------|-------------|
| `MFile` | ModEM model file (without `.rho`). |
| `JFiles` | List of `.jac` Jacobian files to process. |

Each `.jac` file must have a companion `_jac.dat` data file.

## Outputs

Per Jacobian file:

| File | Contents |
|------|----------|
| `<name>_nerr_sp<N>_jac.npz` | Sparse Jacobian matrix. |
| `<name>_nerr_sp<N>_info.npz` | Freq, Data, Site, Comp, DTyp, Info, Scale arrays. |

## Configuration

- `ErrorScale` — `True` to normalise by data errors; `False` for raw Jacobian (e.g. from ModEM3DJE.x).
- `SparseThresh` — sparsification threshold (e.g. `1e-6`); set to 0 to keep full matrix.
- `WorkDir`, `JFiles`, `MFile` — paths.

## Dependencies

`numpy`, `scipy.sparse`, py4mt: `modem`, `jac_proc`, `util`, `version`.
