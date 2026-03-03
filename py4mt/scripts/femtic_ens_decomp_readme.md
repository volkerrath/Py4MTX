# femtic_ens_decomp.py

Dimensionality reduction (PCA / ICA) on a FEMTIC model ensemble.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_ens_decomp.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Provenance

| Date       | Author | Change                                                          |
|------------|--------|-----------------------------------------------------------------|
| 2025       | vrath  | Created (as femtic_decomp_ens).                                 |
| 2026-03-03 | Claude | Renamed file to femtic_ens_decomp; params to UPPERCASE.         |

## Purpose

Scans a directory of FEMTIC inversion runs, collects every converged model
(nRMS below a threshold), stacks them into an ensemble matrix, and
performs PCA or ICA decomposition using scikit-learn. Prints explained
variance ratios and singular values for an increasing number of components.

## Inputs

| Item | Description |
|------|-------------|
| `ENSEMBLE_DIR` | Directory containing one sub-directory per inversion run. |
| `ENSEMBLE_NAME` | Glob pattern to match run directories (e.g. `rto_*`). |
| `NRMS_MAX` | Maximum normalised RMS; runs above this threshold are skipped. |
| `PROC` | Decomposition method: `'pca'`, `'increment'`, or `'ica'`. |

Each run directory must contain `femtic.cnv` (convergence log) and
`resistivity_block_iter<N>.dat` (final model).

## Outputs

| File | Contents |
|------|----------|
| `<ENSEMBLE_RESULTS>.npz` | `model_list` (file paths, iteration, nRMS) and `ensemble` (stacked model matrix). |

Console output: per-component explained variance, cumulative variance, and singular values.

## Configuration

Edit the **Configuration** section at the top of the script:

- `ENSEMBLE_DIR`, `ENSEMBLE_NAME` — where to find runs.
- `NRMS_MAX` — convergence filter.
- `PROC` — `'pca'` (default), `'increment'` (same loop), or `'ica'`.
- `ENSEMBLE_RESULTS` — output `.npz` path.

## Dependencies

`numpy`, `scikit-learn`, py4mt modules: `femtic`, `util`, `version`.
