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
| 2026-08-13 | Claude Sonnet 5 | Added `femtic_ens_decomp_summary.md` output at end of run: user-set (UPPERCASE) parameters, script path, and run date/time. |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | The parameter summary `femtic_ens_decomp_summary.md` is now written to `ENSEMBLE_DIR` (the working directory set in the script) instead of next to the script: `utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR)`. Needs the `util.py` of the same date. AI-generated; review before production use. |
| 2026-11-11 | Claude Sonnet 5.5 (Anthropic) | New `CNV_FILE` (one name or a list of candidate names; default `["femtic.cnv", "FEMTIC-DABIC.cnv"]`). The file is found with `fem.find_cnv_file()` and its last row is read by column name (`fem.read_cnv_last()`) instead of the fixed positions 0 and 8, which fitted only the layout with distortion columns. Needs the `femtic.py` of the same date. AI-generated; review before production use. |

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
| `CNV_FILE` | Name of the convergence file in each run directory: one name or a list of candidate names (default `["femtic.cnv", "FEMTIC-DABIC.cnv"]`; the first one that exists is used). |
| `PROC` | Decomposition method: `'pca'`, `'increment'`, or `'ica'`. |

Each run directory must contain the convergence log (`CNV_FILE`, default `femtic.cnv` or `FEMTIC-DABIC.cnv`) and
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
