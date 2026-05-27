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
| 2026-03-03 | Claude | Renamed user-set parameters to UPPERCASE; generated README. |
| 2026-05-27 | vrath / Claude Sonnet 4.6 (Anthropic) | Added `femtic_viz` import; `MESH_FILE` / `PLOT_QC` / `PLOT_QC_FILE` / `PLOT_QC_SLICES` / `PLOT_QC_*` config; QC slice plot of best-nRMS member at end of main block (calls `fviz.plot_model_slices`). |

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

## QC slice plot (`PLOT_QC`)

When `PLOT_QC = True`, a slice figure is produced for the **best-converged** ensemble member (lowest nRMS among those that passed `NRMS_MAX`) using `fviz.plot_model_slices` (exact tetrahedron-plane intersection, model-local metres).

| Variable | Default | Description |
|---|---|---|
| `MESH_FILE` | `templates/mesh.dat` | Mesh file required for slicing |
| `PLOT_QC` | `False` | Enable / disable the QC slice plot |
| `PLOT_QC_FILE` | `rto_qc.pdf` | Output path; `None` → interactive show |
| `PLOT_QC_DPI` | `200` | Figure DPI |
| `PLOT_QC_SLICES` | 4 slices | Slice-spec list in model-local metres — same format as `femtic_mod_plot.PLOT_SLICES`; kinds: `"map"`, `"ns"`, `"ew"`, `"plane"` |
| `PLOT_QC_CMAP` | `"turbo_r"` | Matplotlib colormap |
| `PLOT_QC_CLIM` | `[0., 4.]` | log₁₀(Ω·m) colour limits; `None` = auto |
| `PLOT_QC_XLIM`, `PLOT_QC_YLIM`, `PLOT_QC_ZLIM` | `None` | Global axis limits in model-local metres; `None` = auto |
| `PLOT_QC_OCEAN_COLOR` | `"lightgrey"` | Flat colour for ocean cells |
| `PLOT_QC_OCEAN_RHO` | `0.25` Ω·m | Ocean cell sentinel value |

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
py4mt modules: `femtic`, `ensembles`, `util`, `version`, `femtic_viz` (optional, for QC slice plot).

## References

See `femtic_rto_prep_readme.md` for the full list of RTO references.

## Author

Volker Rath (DIAS) — April 2025
