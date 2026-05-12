# mt_aniso1d_sampler.py

PyMC driver for anisotropic 1-D MT inversion.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_aniso1d_sampler.py` |
| Author | Volker Rath (DIAS) |
| Part of | **py4mt** — Python for Magnetotellurics |
| Created | 2026-02-12, with ChatGPT (GPT-5 Thinking) |
| Modified | 2026-03-01 — Gaussian prior option, Claude (Opus 4.6, Anthropic) |
| Modified | 2026-03-01 — Matérn covariance kernels, Claude (Opus 4.6, Anthropic) |
| README generated | 3 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Script-style driver that runs Bayesian (MCMC) inversion of magnetotelluric
impedance (and optionally phase-tensor) data for a simplified anisotropic 1-D
earth model. It is intentionally **not** a CLI; all settings live in the
`USER CONFIG` section at the top of the file. Edit, then run:

```bash
python mt_aniso1d_sampler.py
```

All heavy lifting (model construction, forward modelling, sampling,
post-processing) is delegated to the companion library module `mcmc.py`.


## Environment setup

Two environment variables must be set before launching the script:

| Variable | Description |
|----------|-------------|
| `PY4MTX_ROOT` | Root of the Py4MTX installation (used to locate `py4mt/modules` and `py4mt/scripts`) |
| `PY4MTX_DATA` | Base data directory |


## Input data

The script expects one or more site files matched by `INPUT_GLOB` (default
`*.npz`). Each file must contain at least:

| Key | Shape | Description |
|-----|-------|-------------|
| `freq` | (n,) | Frequencies in Hz |
| `Z` | (n, 2, 2) | Complex impedance tensor |
| `Z_err` | (n, 2, 2) | Impedance uncertainties (optional) |
| `station` | scalar | Station name (optional; defaults to filename stem) |

If `USE_PT = True`, the phase tensor P and its errors P_err are computed
automatically from Z via `mcmc.ensure_phase_tensor`.


## Starting model

The starting model can come from two sources (controlled by `MODEL_DIRECT`):

1. **In-file template** — set `MODEL_DIRECT = MODEL0` to use the dict defined
   at the top of the script. The template is also saved to `MODEL_NPZ` for
   reproducibility.

2. **External NPZ** — set `MODEL_DIRECT = None` and point `MODEL_NPZ` to an
   existing file.

The starting model is normalised via `mcmc.normalize_model`, which ensures
both resistivity and conductivity fields are present and consistent.

### Example model (MODEL0)

An 8-layer isotropic halfspace with log-spaced thicknesses (500–3000 m) and a
uniform background resistivity of 300 Ω·m.


## USER CONFIG reference

### Data and I/O

| Constant | Type | Description |
|----------|------|-------------|
| `MCMC_DATA` | str | Base directory for input data and outputs |
| `INPUT_GLOB` | str | Glob pattern for site files |
| `MODEL_NPZ` | str | Path for saving / loading the starting model |
| `MODEL_DIRECT` | dict or None | In-file model template (set to `None` to load from NPZ) |
| `MODEL_DIRECT_SAVE_PATH` | str | Where to save the direct model |
| `MODEL_DIRECT_OVERWRITE` | bool | Overwrite existing model NPZ |
| `OUTDIR` | str | Output directory for NetCDF traces and summary NPZ files |

### Observation components

| Constant | Type | Description |
|----------|------|-------------|
| `USE_PT` | bool | Include phase tensor in the likelihood |
| `PT_ERR_NSIM` | int | Monte-Carlo samples for P_err bootstrap |
| `Z_COMPS` | tuple of str | Impedance components to fit (subset of `xx, xy, yx, yy`) |
| `PT_COMPS` | tuple of str | Phase-tensor components to fit |
| `PT_REG` | float | Diagonal regularisation for P = inv(Re(Z)) @ Im(Z) |
| `SIGMA_FLOOR_Z` | float | Error floor added in quadrature to Z_err |
| `SIGMA_FLOOR_P` | float | Error floor added in quadrature to P_err |

### Parameterisation and bounds

| Constant | Type | Description |
|----------|------|-------------|
| `FIX_H` | bool | Fix layer thicknesses at the starting model |
| `SAMPLE_LAST_THICKNESS` | bool | Sample the basement placeholder thickness (usually `False`) |
| `SAMPLE_H_M` | bool | Sample a single global depth scale H_m (requires `FIX_H=True`) |
| `LOG10_H_BOUNDS` | (float, float) | Bounds on log10(per-layer thickness) |
| `LOG10_H_TOTAL_BOUNDS` | (float, float) | Bounds on log10(H_m) |
| `LOG10_RHO_BOUNDS` | (float, float) | Bounds on log10(resistivity) |
| `STRIKE_BOUNDS_DEG` | (float, float) | Bounds on strike angle (degrees) |
| `PARAM_DOMAIN` | str | `"rho"` or `"sigma"` — domain for sampling |

### Sampler settings

| Constant | Type | Description |
|----------|------|-------------|
| `STEP_METHOD` | str | `"demetropolis"`, `"demetropolisz"`, `"metropolis"`, `"nuts"`, or `"hmc"` |
| `ENABLE_GRAD` | bool | Build gradient-enabled likelihood (required for NUTS/HMC) |
| `PROGRESSBAR` | bool | Show PyMC progress bar |

### Prior specification

| Constant | Type | Description |
|----------|------|-------------|
| `PRIOR_KIND` | str | `"default"`, `"uniform"`, or `"gaussian"` |
| `PRIOR_STD` | dict | Per-parameter standard deviations (diagonal covariance) |
| `PRIOR_COV` | ndarray or None | Full covariance matrix (overrides ρ/strike entries in `PRIOR_STD`) |

**`PRIOR_STD` keys:**

| Key | Applies to | Default |
|-----|-----------|---------|
| `"log10_rho"` | Both log10(ρ_min) and log10(ρ_max) | 1.0 |
| `"log10_rho_min"` | log10(ρ_min) only | — |
| `"log10_rho_max"` | log10(ρ_max) only | — |
| `"strike_deg"` | Strike angle (degrees) | 45.0 |
| `"log10_h"` | Per-layer log10(thickness) | 0.5 |
| `"log10_H"` | Global thickness scale log10(H_m) | 0.5 |

**`PRIOR_COV`** is a `(3*nl, 3*nl)` covariance matrix on the stacked vector
`[log10_ρ_min, log10_ρ_max, strike_deg]`. Build it with
`mcmc.build_gaussian_cov`, which supports these inter-layer correlation models:

| `corr_model` | Kernel | Formula (d = \|i−j\|) |
|---------------|--------|----------------------|
| `None` / `"identity"` | Independent layers | R = I |
| `"exponential"` | Matérn ν=½ | exp(−d / L) |
| `"matern32"` | Matérn ν=3⁄2 | (1 + √3 d/L) exp(−√3 d/L) |
| `"matern52"` | Matérn ν=5⁄2 | (1 + √5 d/L + 5d²/3L²) exp(−√5 d/L) |
| `"matern"` | General Matérn | Bessel-based (ν via `nu` kwarg) |
| `"gaussian"` | Matérn ν→∞ | exp(−d² / 2L²) |

### Quantile settings

| Constant | Type | Description |
|----------|------|-------------|
| `QPAIRS` | tuple of (float, float) | Quantile/percentile pairs for the posterior summary (values > 1 are interpreted as percentiles) |


## Pre-defined `pmc_dict` presets

Copy the desired preset into `pmc_dict` to activate it.

### Fixed-thickness (10-layer) presets

| Dict name | Step method | Draws | Tune | Chains |
|-----------|-------------|-------|------|--------|
| `PMC_DICT_NUTS_10LAYER_HFIXED` | NUTS | 4 000 | 4 000 | 4 |
| `PMC_DICT_METROPOLIS_10LAYER_HFIXED` | Metropolis | 120 000 | 30 000 | 4 |
| `PMC_DICT_DEMETROPOLISZ_10LAYER_HFIXED` | DEMetropolisZ | 60 000 | 20 000 | 8 |

### Thickness-sampled (6-layer) presets

| Dict name | Step method | Draws | Tune | Chains |
|-----------|-------------|-------|------|--------|
| `PMC_DICT_NUTS_6LAYER_LOGHM` | NUTS | 6 000 | 6 000 | 4 |
| `PMC_DICT_METROPOLIS_6LAYER_LOGHM` | Metropolis | 200 000 | 80 000 | 4 |
| `PMC_DICT_DEMETROPOLISZ_6LAYER_LOGHM` | DEMetropolisZ | 100 000 | 30 000 | 10 |


## Outputs

For each input site the script writes two files into `OUTDIR`:

| File | Format | Contents |
|------|--------|----------|
| `{station}_pmc_{method}.nc` | NetCDF (ArviZ) | Full posterior trace (InferenceData) |
| `{station}_pmc_{method}_summary.npz` | NumPy NPZ | Compact quantile envelopes for plotting |


## Dependencies

`numpy`, `PyMC`, `arviz`; py4mt: `data_proc`, `mcmc`, `util`, `version`.
