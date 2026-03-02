# femtic_decomp_ens.py

Dimensionality reduction (PCA / ICA) on a FEMTIC model ensemble.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_decomp_ens.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Scans a directory of FEMTIC inversion runs, collects every converged model
(nRMS below a threshold), stacks them into an ensemble matrix, and
performs PCA or ICA decomposition using scikit-learn. Prints explained
variance ratios and singular values for an increasing number of components.

## Inputs

| Item | Description |
|------|-------------|
| `EnsembleDir` | Directory containing one sub-directory per inversion run. |
| `EnsembleName` | Glob pattern to match run directories (e.g. `rto_*`). |
| `NRMSmax` | Maximum normalised RMS; runs above this threshold are skipped. |
| `Proc` | Decomposition method: `'pca'`, `'increment'`, or `'ica'`. |

Each run directory must contain `femtic.cnv` (convergence log) and
`resistivity_block_iter<N>.dat` (final model).

## Outputs

| File | Contents |
|------|----------|
| `<EnsembleResults>.npz` | `model_list` (file paths, iteration, nRMS) and `ensemble` (stacked model matrix). |

Console output: per-component explained variance, cumulative variance, and singular values.

## Configuration

Edit the **Configuration** section at the top of the script:

- `EnsembleDir`, `EnsembleName` — where to find runs.
- `NRMSmax` — convergence filter.
- `Proc` — `'pca'` (default), `'increment'` (same loop), or `'ica'`.
- `EnsembleResults` — output `.npz` path.

## Dependencies

`numpy`, `scikit-learn`, py4mt modules: `femtic`, `util`, `version`.
