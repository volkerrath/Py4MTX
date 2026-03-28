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

Visualization is performed immediately after each perturbation step, using
helpers from `femtic_viz` (`plot_data_ensemble`, `plot_model_ensemble`).
`matplotlib.pyplot` is imported at the top level.  `VIZ_SAMPLES` is set in
the **Base setup** block and applies to both plots.

| Variable          | Description                                                              |
|-------------------|--------------------------------------------------------------------------|
| `VIZ_SAMPLES`     | List of ensemble-member indices to include in diagnostic plots.          |
| `PLOT_DATA`       | Enable / disable joint data plot (original vs. perturbed `observe.dat`). |
| `DAT_WHAT`        | MT quantity: `'rho'`, `'phase'`, `'tipper'`, or `'pt'`.                 |
| `DAT_COMPS`       | Impedance components, comma-separated (e.g. `'xy,yx'`).                 |
| `DAT_SHOW_ERRORS` | Propagate error envelopes into the original-data curve.                  |
| `PLOT_MODEL`      | Enable / disable joint model plot (original vs. perturbed resistivity).  |
| `MOD_MESH`        | Path to the shared `mesh.dat`.                                           |
| `MOD_ORIG`        | Path to the reference (template) resistivity block.                      |
| `MOD_SLICES`      | List of 1–5 slice descriptors; each dict has `'type'`: `'map'` or `'curtain'`, plus kwargs forwarded to `femtic_viz`. |

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

## Author

Volker Rath (DIAS) — April 2025
