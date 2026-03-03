# femtic_ens_eof.py

Empirical Orthogonal Function (EOF) analysis of a FEMTIC model ensemble.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_ens_eof.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Provenance

| Date       | Author | Change                                                          |
|------------|--------|-----------------------------------------------------------------|
| 2025       | vrath  | Created (as femtic_ensemble_eof.py).                            |
| 2026-03-03 | Claude | Renamed file to femtic_ens_eof; params to UPPERCASE.            |

## Purpose

Collects converged FEMTIC inversion results into an ensemble matrix,
computes EOFs via SVD (demeaned), and generates reconstructed model
samples from truncated or individual EOF modes. This provides a
data-driven decomposition of the ensemble variability.

## Inputs

| Item | Description |
|------|-------------|
| `ENSEMBLE_DIR` | Root directory containing inversion sub-directories. |
| `ENSEMBLE_NAME` | Prefix pattern for directory matching (e.g. `ann_`). |
| `MIN_RMS` | Maximum acceptable nRMS for inclusion. |
| `MESH_FILE` | Relative path to the FEMTIC mesh file within each run directory. |

Each run directory must contain `femtic.cnv`, `mesh.dat`, and the final
`resistivity_block_iter<N>.dat`.

## Outputs

| File | Contents |
|------|----------|
| `<ENSEMBLE_FILE>.npz` | Raw ensemble matrix, mesh nodes and connectivity. |
| `<ENSEMBLE_EOF>.npz` | EOFs, principal components, weights, fractional variance, mean model, ensemble, and mesh. |
| `<ENSEMBLE_EOF>_trunc<N>.npz` | Truncated EOF reconstruction using modes 0…N (or individual mode N if `GET_COMPONENTS=True`). |

## Configuration

- `GET_COMPONENTS` — if `True`, store each EOF mode individually (`_comp`); if `False`, store cumulative truncations (`_trunc`).
- `MIN_RMS` — convergence threshold.
- `ENSEMBLE_FILE`, `ENSEMBLE_EOF` — base names for output files.

## Dependencies

`numpy`, py4mt modules: `femtic`, `util`, `ensembles`, `version`.
