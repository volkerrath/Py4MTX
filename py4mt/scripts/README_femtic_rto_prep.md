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
                                 ↓
                          (run FEMTIC on each member)
                                 ↓
femtic_rto_post.py    →   collect results & statistics
```

## Configuration

All settings are at the top of the script:

| Variable        | Description                                                         |
|-----------------|---------------------------------------------------------------------|
| `N_samples`     | Number of ensemble members to generate.                             |
| `EnsembleDir`   | Root directory for the ensemble.                                    |
| `Templates`     | Directory containing template FEMTIC input files.                   |
| `Files`         | List of template files copied into each member directory.           |
| `EnsembleName`  | Prefix for member directories (e.g. `misti_rto_`).                  |
| `FromTo`        | `None` for 0…N−1, or `(start, stop)` to continue / patch members.  |

### Model perturbation

| Variable     | Description                                                          |
|--------------|----------------------------------------------------------------------|
| `PerturbMod` | Enable / disable model perturbation.                                 |
| `Mod_ref`    | Reference model file (e.g. converged iterate).                       |
| `Mod_method` | `'add'` — add perturbation to log₁₀(ρ).                             |
| `Mod_pdf`    | Distribution parameters `['normal', mean, std]`.                     |
| `Mod_R`      | Source of roughness / precision: `'femtic R'` or `'Q'`.              |
| `R_file`     | Base name of the sparse-matrix `.npz` file (without extension).      |

### Data perturbation

| Variable      | Description                                                        |
|---------------|--------------------------------------------------------------------|
| `PerturbDat`  | Enable / disable data perturbation.                                |
| `Dat_pdf`     | Distribution parameters for noise `['normal', 0., 1.0]`.          |
| `ResetErrors` | If `True`, overwrite error floors before perturbation.             |
| `Errors`      | Per-component error floors for impedance, VTF, and phase tensor.   |

## Dependencies

| Package        | Role                                          |
|----------------|-----------------------------------------------|
| `numpy`        | Array operations and random draws.            |
| `scipy.sparse` | Sparse precision / roughness matrices.        |
| `ensembles`    | Directory generation, data & model ensembles. |
| `femtic`       | FEMTIC I/O (model insertion, data modification). |
| `util`         | Print / version helpers.                      |

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

## Author

Volker Rath (DIAS) — April 2025
