# README\_femtic\_rto\_prep.md

Preparation script for the **Randomize-Then-Optimize (RTO)** algorithm applied
to FEMTIC magnetotelluric inversion.

## Purpose

`femtic_rto_prep.py` creates a complete ensemble of perturbed data and model
files that are ready to be submitted as independent FEMTIC inversion runs.
After all ensemble members have converged, the results can be collected and
analysed for uncertainty quantification (see references below).

## Algorithm outline

For each ensemble member *i = 1 … N*:

1. **Perturbed data** — draw `d̃ ~ N(d, C_d)` by adding scaled Gaussian noise
   to the observed data (`observe.dat`).
2. **Perturbed prior model** — draw `m̃ ~ N(0, (1/μ)(L^T L)^{-1})` by
   sampling from the precision matrix built from the FEMTIC roughness operator.
3. **Solve** — the deterministic FEMTIC inversion is run on each
   `(d̃, m̃)` pair (performed externally, not by this script).

## Workflow

```text
femtic_rto_rough.py   →   R / Q matrix  (.npz)
                                 ↓
femtic_rto_prep.py    →   ensemble directories with
                          perturbed observe.dat &
                          resistivity_block_iter0.dat
                          + diagnostic plots
                                 ↓
                          (run FEMTIC on each member)
                                 ↓
femtic_rto_post.py    →   collect results & statistics
```

## Configuration

All settings are at the top of the script:

| Variable        | Description                                                         |
|-----------------|---------------------------------------------------------------------|
| `N_SAMPLES`     | Number of ensemble members to generate.                             |
| `ENSEMBLE_DIR`   | Root directory for the ensemble.                                    |
| `TEMPLATES`     | Directory containing template FEMTIC input files.                   |
| `FILES`         | List of template files copied into each member directory.           |
| `ENSEMBLE_NAME`  | Prefix for member directories (e.g. `misti_rto_`).                  |
| `FROM_TO`        | `None` for 0…N−1, or `(start, stop)` to continue / patch members.  |

### Model perturbation

| Variable     | Description                                                          |
|--------------|----------------------------------------------------------------------|
| `PERTURB_MOD`         | Enable / disable model perturbation.                                        |
| `MOD_REF`             | Reference model file (e.g. converged iterate).                              |
| `MOD_METHOD`          | `'add'` — add perturbation to log₁₀(ρ).                                    |
| `MOD_PDF`             | Distribution parameters `['normal', mean, std]`.                            |
| `R_FILE`              | Base name of the sparse-matrix `.npz` file (without extension).             |
| `MOD_ALGO`            | `'low rank'` (randomized SVD, **recommended default**) or `'full rank'` (CG). |
| `MOD_N_EIG`           | Singular triplets for low-rank branch. Default **128**; increase to 256 for smoother samples (cost is linear). |
| `MOD_N_OVERSAMPLING`  | Extra range-finder columns (default 10; 10–15 is sufficient).               |
| `MOD_N_POWER_ITER`    | Power-iteration steps for low-rank branch. Default **3**; 3–4 recommended for FEMTIC roughness spectra. |
| `MOD_SIGMA2_RESIDUAL` | Isotropic residual variance per sample. Default **1e-3** (~10% of typical log10ρ variance); `0` disables. |
| `MOD_LAM`             | Diagonal shift seed for full-rank branch (default 0).                        |
| `MOD_LAM_MODE`        | `'scaled_median_diag'` (auto, default) or `'fixed'` — full-rank branch.     |
| `MOD_LAM_ALPHA`       | Scale factor α for the scaled-diagonal λ rule. Default **1e-4**; key speed lever — raise to 1e-3 if CG is slow. |
| `MOD_SOLVER`          | Iterative solver for full-rank branch: `'cg'` (optimal for SPD Q, default) or `'bicgstab'`. |
| `MOD_PRECOND`         | Preconditioner for full-rank branch: `'ilu'` (default, 3–5× fewer iterations than jacobi) or `'jacobi'`. |

R is passed **directly** to `generate_model_ensemble`; Q = R^T R is never
explicitly materialised — both branches form it implicitly via matvecs.

### Data perturbation

| Variable      | Description                                                        |
|---------------|--------------------------------------------------------------------|
| `PERTURB_DAT`  | Enable / disable data perturbation.                                |
| `DAT_PDF`     | Distribution parameters for noise `['normal', 0., 1.0]`.          |
| `RESET_ERRORS` | If `True`, overwrite error floors before perturbation.             |
| `ERRORS`      | Per-component error floors for impedance, VTF, and phase tensor.   |

### Visualization

All visualization parameters live in a single **Visualization config** section
near the top of the script, immediately after the perturbation config blocks.
`matplotlib.pyplot` is imported at the top level.

Ensemble members shown in the plots are drawn *randomly* without replacement
from 0 … `N_SAMPLES − 1` each run.  The drawn list is printed at runtime.

| Variable          | Description                                                              |
|-------------------|--------------------------------------------------------------------------|
| `PLOT_DATA`       | Enable / disable joint data plot (original vs. perturbed `observe.dat`). |
| `PLOT_MODEL`      | Enable / disable joint model plot (original vs. perturbed resistivity).  |
| `VIZ_N_SAMPLES`   | Number of ensemble members to draw for both plots (≤ `N_SAMPLES`).      |
| `VIZ_N_SITES`     | Number of MT sites drawn per data-plot row; `None` shows all sites.      |
| `DAT_WHAT`        | List of panel types, one column per entry: `'rho'`, `'phase'`, `'tipper'`, `'pt'`. E.g. `["rho", "phase", "tipper"]`. |
| `DAT_COMPS`       | List of component strings, one per entry in `DAT_WHAT` (ignored for `'tipper'`/`'pt'`). A single string is broadcast to all panels. E.g. `["xx,xy,yx,yy", "xy,yx"]`. |
| `DAT_SHOW_ERRORS_ORIG` | Show ±1σ error envelopes on the **original** curves (`True` / `False`). Raw template errors can be large at long periods — set `False` to hide them. |
| `DAT_SHOW_ERRORS_PERT` | Show ±1σ error envelopes on the **perturbed** curves (`True` / `False`). Perturbed files carry reset relative errors (compact ±σ bands). |
| `DAT_ALPHA_ORIG`  | Opacity of the **original** data curves (0–1; default 1.0).              |
| `DAT_ALPHA_PERT`  | Opacity of the **perturbed** data curves (0–1; default 0.6).             |
| `DAT_COMP_MARKERS` | Dict of marker symbol per component class (`'ii'`, `'ij'`, `'inv'`); `None` = use `DEFAULT_COMP_MARKERS` from `femtic_viz`; `{}` = disable markers. |
| `DAT_MARKERSIZE`  | Marker size in points (default `4.0`).                                   |
| `DAT_MARKEVERY`   | Plot a marker every N-th period; `None` = every period.                  |
| `DAT_ERROR_STYLE_ORIG` | Error rendering for **original** curves: `'shade'` (default), `'bar'`, or `'both'`. |
| `DAT_ERROR_STYLE_PERT` | Error rendering for **perturbed** curves: `'shade'` (default), `'bar'`, or `'both'`. |
| `DAT_PERLIMS`      | Period-axis limits `(T_min, T_max)` in seconds applied to every panel. `None` = auto.      |
| `DAT_RHOLIMS`      | y-axis limits for `'rho'` panels (Ω·m). `None` = auto.                                   |
| `DAT_PHSLIMS`      | y-axis limits for `'phase'` panels (degrees). `None` = auto.                              |
| `DAT_VTFLIMS`      | y-axis limits for `'tipper'` panels. `None` = auto.                                       |
| `DAT_PTLIMS`       | y-axis limits for `'pt'` panels. `None` = auto.                                           |
| `MOD_MESH`        | Path to the shared `mesh.dat`.                                           |
| `MOD_MODE`        | Slice rendering mode: `'tri'` \| `'scatter'` \| `'grid'`.               |
| `MOD_LOG10`       | Plot log₁₀(ρ) if `True`.                                                |
| `MOD_CMAP`        | Matplotlib colormap (default `'jet_r'`).                                 |
| `MOD_CLIM`        | `(vmin, vmax)` in log₁₀(Ω·m); `None` = auto-derived from original model.|
| `MOD_XLIM`        | `(xmin, xmax)` in metres for **map** slices (easting axis); `None` = auto.  |
| `MOD_YLIM`        | `(ymin, ymax)` in metres — northing axis for **map** slices; along-profile distance axis for **curtain** slices; `None` = auto. |
| `MOD_ZLIM`        | `(zmin, zmax)` in metres for **curtain** slices (depth axis, negative-down); `None` = auto. |
| `MOD_MESH_LINES`  | If `True`, overlay triangulation edges on filled patches (default `False`).  |
| `MOD_MESH_LW`     | Line width for mesh edge overlay (default `0.3` pt).                         |
| `MOD_MESH_COLOR`  | Colour for mesh edge overlay (default `'k'`).                                |
| `MOD_SLICES`      | List of 1–5 slice descriptors; each dict has `'type'`: `'map'` or `'curtain'`, plus kwargs forwarded to `femtic_viz`. |

`MOD_ORIG` is set to `MOD_REF` (they are the same path — the template reference
model).  `MOD_REF_BASE` (derived automatically via `os.path.basename`) holds
the bare filename and is used when constructing per-member file paths inside
`generate_model_ensemble` and `mod_ens_files`.

Diagnostic figures are saved to:

- `<ENSEMBLE_DIR>/rto_data_ensemble.pdf`
- `<ENSEMBLE_DIR>/rto_model_ensemble.pdf`

## Dependencies

| Package        | Role                                             |
|----------------|--------------------------------------------------|
| `numpy`        | Array operations and random draws.               |
| `scipy.sparse` | Sparse precision / roughness matrices.           |
| `ensembles`    | Directory generation, data & model ensembles.    |
| `femtic`       | FEMTIC I/O (model insertion, data modification). |
| `femtic_viz`   | Ensemble visualization helpers.                  |
| `util`         | Print / version helpers.                         |

## References

- Bardsley, J. M.; Solonen, A.; Haario, H. & Laine, M.
  *Randomize-Then-Optimize: a Method for Sampling from Posterior Distributions
  in Nonlinear Inverse Problems.*
  SIAM J. Sci. Comp., 2014, **36**, A1895–A1910.

- Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
  *Uncertainty quantification for regularized inversion of electromagnetic
  geophysical data. Part I: Motivation and Theory.*
  Geophysical Journal International, 2022, doi:10.1093/gji/ggac241.

- Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
  *Uncertainty quantification for regularized inversion of electromagnetic
  geophysical data — Part II: application in 1-D and 2-D problems.*
  Geophysical Journal International, 2022, doi:10.1093/gji/ggac242.

## Provenance

| Date       | Author | Change                                                      |
|------------|--------|-------------------------------------------------------------|
| 2025-04-30 | vrath  | Created.                                                    |
| 2026-03-03 | Claude | Renamed user-set parameters to UPPERCASE.                   |
| 2026-03-24 | Claude | Added visualization config blocks; helpers in `femtic_viz`. |
| 2026-03-28 | Claude | Moved viz blocks into perturbation sections; `matplotlib`   |
|            |        | imported at top level; `VIZ_SAMPLES` moved to base setup.   |
| 2026-03-29 | Claude | Consolidated all viz parameters into a single Visualization |
|            |        | config section; replaced fixed `VIZ_SAMPLES` list with      |
|            |        | `VIZ_N_SAMPLES` (random draw); added `VIZ_N_SITES` for      |
|            |        | random site sub-sampling in `plot_data_ensemble`.           |
| 2026-03-31 | Claude | Pass R directly (not Q); randomized SVD replaces eigsh in   |
|            |        | low-rank branch; new MOD_N_EIG / MOD_N_OVERSAMPLING /       |
|            |        | MOD_N_POWER_ITER / MOD_SIGMA2_RESIDUAL / MOD_LAM /          |
|            |        | MOD_LAM_MODE / MOD_LAM_ALPHA / MOD_SOLVER / MOD_PRECOND;    |
|            |        | MOD_R sentinel variable removed.                            |
| 2026-04-02 | Claude | Added `MOD_XLIM`, `MOD_YLIM`, `MOD_ZLIM` to visualization   |
|            |        | config; wired into `plot_model_ensemble` call.              |
| 2026-04-03 | Claude | Added `DAT_ALPHA_ORIG`, `DAT_ALPHA_PERT` to data-plot config; |
|            |        | added `MOD_MESH_LINES`, `MOD_MESH_LW`, `MOD_MESH_COLOR` to  |
|            |        | model-plot config; wired into respective function calls.     |
| 2026-04-12 | Claude | Added `DAT_COMP_MARKERS`, `DAT_MARKERSIZE`, `DAT_MARKEVERY`  |
|            |        | to data-plot config for per-class marker differentiation.   |
|            |        | Added `DAT_ERROR_STYLE_ORIG` / `DAT_ERROR_STYLE_PERT` for   |
|            |        | `'shade'` / `'bar'` / `'both'` error rendering per curve;  |
|            |        | no shared fallback variable.                                |
|            |        | `DAT_WHAT` changed to a list for multi-panel layouts        |
|            |        | (e.g. `["rho", "phase", "tipper"]`); `DAT_COMPS` changed   |
|            |        | to a parallel list (one entry per panel; single string      |
|            |        | still accepted and broadcast to all rho/phase panels).      |
| 2026-04-12 | Claude | Replaced `DAT_SHOW_ERRORS` with `DAT_SHOW_ERRORS_ORIG` /    |
|            |        | `DAT_SHOW_ERRORS_PERT` for independent per-curve control.   |
|            |        | Removed hardcoded `show_errors_orig=True` from the call.   |
|            |        | Fixed `PLOT_DIR` path (was erroneously prepended with       |
|            |        | `ENSEMBLE_DIR`).                                            |
| 2026-03-31 | Claude | Bug fixes: `DAT_METHOD` trailing comma (tuple → str);        |
|            |        | `MOD_ORIG` double-prepend of `TEMPLATES`; `refmod` and       |
|            |        | `mod_ens_files` used full path instead of basename;          |
|            |        | `MOD_REF_BASE` introduced to hold the filename component.   |
| 2026-04-12 | Claude | Per-sample plot loop: both `PLOT_DATA` and `PLOT_MODEL`      |
|            |        | blocks now iterate over `VIZ_SAMPLES`, calling               |
|            |        | `plot_data_ensemble` / `plot_model_ensemble` with            |
|            |        | `sample_indices=[i]` and saving `rto_data<PLOT_STR>.pdf` /  |
|            |        | `rto_model<PLOT_STR>.pdf` into each member's own subdir.     |
|            |        | `PLOT_DIR` no longer used for per-sample figures.            |
|            |        | `os.makedirs` replaces `os.mkdir` for safer dir creation.   |
| 2026-04-12 | Claude | Added `DAT_PERLIMS` / `DAT_RHOLIMS` / `DAT_PHSLIMS` /        |
|            |        | `DAT_VTFLIMS` / `DAT_PTLIMS` to data-plot config; wired      |
|            |        | into `plot_data_ensemble` as `perlims` / `rholims` /          |
|            |        | `phslims` / `vtflims` / `ptlims`. Applied post-hoc in        |
|            |        | `femtic_viz` after each cell is filled. `femtic_viz`          |
|            |        | provenance and docstring updated accordingly.                |

## Author

Volker Rath (DIAS) — April 2025
