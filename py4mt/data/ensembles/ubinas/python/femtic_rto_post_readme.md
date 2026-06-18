# femtic_rto_post.py

Postprocessing of a Randomize-Then-Optimize (RTO) ensemble for FEMTIC.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_rto_post.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 3 March 2026 by Claude (Anthropic) |

## Provenance

| Date       | Author | Change                                       |
|------------|--------|----------------------------------------------|
| 2025-04-30 | vrath  | Created.                                     |
| 2026-03-03 | Claude | Renamed user-set parameters to UPPERCASE;    |
|            |        | generated README.                            |

## Purpose

Collects converged models from an RTO ensemble of FEMTIC inversion runs,
computes summary statistics (mean, variance, median, MAD, percentiles)
and the empirical covariance matrix, optionally sparsifies the
covariance, and saves everything to a compressed `.npz` file.

## Workflow position

```text
femtic_rto_prep.py    →   ensemble directories
                                 ↓
                          (run FEMTIC on each member)
                                 ↓
femtic_rto_post.py    →   RTO_results.npz
                          (covariance, statistics, ensemble)
                                 ↓
femtic_ens_from_covar.py  →   resample from covariance
```

## Configuration

| Variable          | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `ENSEMBLE_DIR`    | Root directory containing RTO inversion sub-directories.           |
| `ENSEMBLE_NAME`   | Glob pattern to match run directories (e.g. `rto_*`).             |
| `NRMS_MAX`        | Maximum nRMS; runs above this threshold are skipped.               |
| `PERCENTILES`     | List of percentiles to compute (default: 2-sigma / 1-sigma).      |
| `ENSEMBLE_RESULTS`| Output `.npz` file path.                                          |
| `SPARSIFY`        | If `True`, threshold the covariance to create a sparse version.    |
| `SPARSE_THRESH`   | Relative threshold for zeroing small covariance entries.           |

## Outputs

| Key in `.npz`  | Contents                                          |
|----------------|---------------------------------------------------|
| `model_list`   | List of `[file, iteration, nRMS]` per member.     |
| `rto_ens`      | Stacked ensemble matrix (N_members × N_params).   |
| `rto_cov`      | Empirical covariance matrix.                      |
| `rto_avg`      | Ensemble mean.                                    |
| `rto_var`      | Ensemble variance.                                |
| `rto_med`      | Ensemble median.                                  |
| `rto_mad`      | Median absolute deviation.                        |
| `rto_prc`      | Percentile values.                                |

## Dependencies

`numpy`, `scipy.sparse`, `scikit-learn` (empirical covariance),
py4mt modules: `femtic`, `ensembles`, `util`, `version`.

## References

See `femtic_rto_prep_readme.md` for the full list of RTO references.

## Author

Volker Rath (DIAS) — April 2025
