# py4mt Processing Scripts

Python scripts for 3D magnetotelluric (MT) inversion post-processing, uncertainty analysis, and visualisation. Part of the **py4mt** (Python for Magnetotellurics) toolkit.

These scripts work with two 3D MT inversion codes:

- **ModEM** — Modular Electromagnetic Inversion (Egbert & Kelbert, 2012)
- **FEMTIC** — Finite Element Method for 3D Magnetotelluric Inversion with unstructured tetrahedral meshes (Usui, 2015)

## Repository Structure

```
├── femtic/                     # FEMTIC inversion utilities
│   ├── README.md
│   ├── femtic_decomp_ens.py        # PCA/ICA decomposition of model ensembles
│   ├── femtic_ensemble_eof.py      # Empirical Orthogonal Function analysis
│   ├── femtic_ens_from_covar.py    # Generate ensembles from posterior covariance
│   ├── femtic_jcn_prep.py          # Prepare jackknife uncertainty directories
│   ├── femtic_plot_convergence.py  # Plot iteration convergence curves
│   └── femtic_plot_lcurve.py       # Plot regularisation L-curves
│
├── modem/                      # ModEM inversion utilities
│   ├── README.md
│   ├── modem_insert_body.py        # Insert geometric bodies into models
│   ├── modem_insert_multi.py       # Generate multiple perturbed models
│   ├── modem_jac_proc.py           # Process raw Jacobians (normalise, sparsify)
│   ├── modem_jac_sens.py           # Compute sensitivity maps from Jacobians
│   ├── modem_jac_splitmerge.py     # Split/merge Jacobian matrices
│   ├── modem_jac_stats.py          # Jacobian statistics by component/site/freq
│   ├── modem_jac_svd.py            # Randomised SVD of Jacobians
│   ├── modem_mod_fill.py           # Fill model with uniform resistivity
│   ├── modem_mod_stats.py          # Cell-wise model statistics and histograms
│   ├── modem_mod_trans.py          # Convert models between formats
│   ├── modem_plot_rms.py           # Plot RMS convergence from log files
│   └── modem_plot_slices.py        # Plot slices through 3D models
│
└── README.md
```

## Dependencies

### Environment Variables

All scripts require two environment variables:

| Variable | Description |
|----------|-------------|
| `PY4MTX_ROOT` | Root directory of the py4mt toolkit (contains `py4mt/modules/` and `py4mt/scripts/`) |
| `PY4MTX_DATA` | Base directory for MT data and results |

### Python Packages

**Core** (all scripts): `numpy`, `scipy`

**Visualisation** (plotting scripts): `matplotlib`

**Jacobian processing**: `scipy.sparse`, `netCDF4`

**Ensemble/decomposition**: `scikit-learn`

**Covariance sampling** (`femtic_ens_from_covar.py`): `scikit-sparse`

### Internal py4mt Modules

| Module | Purpose |
|--------|---------|
| `modem` | Read/write ModEM model, data, and Jacobian files |
| `femtic` | Read/write FEMTIC model and mesh files |
| `jacproc` | Jacobian processing (normalisation, sensitivity, SVD) |
| `util` | General utilities (file listing, coordinate transforms) |
| `ensembles` | EOF computation and sampling |
| `version` | Version string generation |

## Usage

Each script is configured via variables in its header section (marked `# Configuration`). Edit these before running:

```bash
export PY4MTX_ROOT=/path/to/py4mt
export PY4MTX_DATA=/path/to/data
python modem/modem_jac_proc.py
```

## Output Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| `mod` | `.rho` | ModEM native format |
| `ubc` | `.mod` + `.mesh` | UBC-GIF mesh + model |
| `rlm` | `.rlm` | CGG/RLM format |

## Authors

- **vrath** — primary author
- **sbyrd** — contributor (modem_plot_rms, modem_mod_stats)

## Changes from Original Scripts

- **Bug fixes**: `=` vs `==` in `np.where()` (`modem_jac_splitmerge.py`); undefined `ipca` in ICA branch (`femtic_decomp_ens.py`); boolean scalar indexing for components (`modem_jac_sens.py`); syntax error in `write_mod_npz` (`modem_insert_multi.py`); undefined `N_samples` (`femtic_jcn_prep.py`)
- **Removed dead code**: commented-out blocks, unused imports (`jax`, `numba`, `netCDF4` where unneeded), duplicate code blocks
- **Consistent style**: uniform quoting, spacing, f-strings
- **Improved docstrings**: module-level docstrings describe purpose and methods
- **Consolidated module paths**: standardised to `py4mt/modules/` and `py4mt/scripts/`
- **Refactored repetition**: extracted `write_sensitivity()` helper in `modem_jac_sens.py`
- **Fixed plotting**: moved `plt.legend()` outside loop in `modem_plot_rms.py`; removed duplicate block in `femtic_ensemble_eof.py`
