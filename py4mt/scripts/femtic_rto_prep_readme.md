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
| `PERTURB_MOD` | Enable / disable model perturbation.                                 |
| `MOD_REF`    | Reference model file (e.g. converged iterate).                       |
| `MOD_METHOD` | `'add'` — add perturbation to log₁₀(ρ).                             |
| `MOD_PDF`    | Distribution parameters `['normal', mean, std]`.                     |
| `MOD_R`      | Source of roughness / precision: `'femtic R'` or `'Q'`.              |
| `R_FILE`     | Base name of the sparse-matrix `.npz` file (without extension).      |

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
| `DAT_WHAT`        | MT quantity: `'rho'`, `'phase'`, `'tipper'`, or `'pt'`.                 |
| `DAT_COMPS`       | Impedance components, comma-separated (e.g. `'xy,yx'`).                 |
| `DAT_SHOW_ERRORS` | Propagate error envelopes into the original-data curve.                  |
| `MOD_MESH`        | Path to the shared `mesh.dat`.                                           |
| `MOD_MODE`        | Slice rendering mode: `'tri'` \| `'scatter'` \| `'grid'`.               |
| `MOD_LOG10`       | Plot log₁₀(ρ) if `True`.                                                |
| `MOD_CMAP`        | Matplotlib colormap (default `'jet_r'`).                                 |
| `MOD_CLIM`        | `(vmin, vmax)` in log₁₀(Ω·m); `None` = auto-derived from original model.|
| `MOD_SLICES`      | List of 1–5 slice descriptors; each dict has `'type'`: `'map'` or `'curtain'`, plus kwargs forwarded to `femtic_viz`. |

`MOD_ORIG` is derived automatically from `MOD_REF` (defined in the model
perturbation block) and does not need to be set separately.

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

## Author

Volker Rath (DIAS) — April 2025
