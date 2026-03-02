# `mt_aniso1d_sampler.py` — PyMC driver for anisotropic 1-D MT inversion

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

The script prepends the module and script directories to `sys.path`, then
imports `data_proc`, `mcmc`, `util`, and `version` from the local codebase.


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

1. **In-file template** — set `MODEL_DIRECT = Model0` to use the dict defined
   at the top of the script.  The template is also saved to `MODEL_NPZ` for
   reproducibility.

2. **External NPZ** — set `MODEL_DIRECT = None` and point `MODEL_NPZ` to an
   existing file.

The starting model is normalised via `mcmc.normalize_model`, which ensures
both resistivity and conductivity fields are present and consistent.

### Example model (Model0)

An 8-layer isotropic halfspace with log-spaced thicknesses (500–3000 m) and a
uniform background resistivity of 300 Ω·m.


## USER CONFIG reference

### Data and I/O

| Variable | Type | Description |
|----------|------|-------------|
| `MCMC_Data` | str | Base directory for input data and outputs |
| `INPUT_GLOB` | str | Glob pattern for site files |
| `MODEL_NPZ` | str | Path for saving / loading the starting model |
| `MODEL_DIRECT` | dict or None | In-file model template (set to `None` to load from NPZ) |
| `MODEL_DIRECT_SAVE_PATH` | str | Where to save the direct model |
| `MODEL_DIRECT_OVERWRITE` | bool | Overwrite existing model NPZ |
| `OUTDIR` | str | Output directory for NetCDF traces and summary NPZ files |

### Observation components

| Variable | Type | Description |
|----------|------|-------------|
| `USE_PT` | bool | Include phase tensor in the likelihood |
| `PT_ERR_NSIM` | int | Monte-Carlo samples for P_err bootstrap |
| `Z_COMPS` | tuple of str | Impedance components to fit (subset of `xx, xy, yx, yy`) |
| `PT_COMPS` | tuple of str | Phase-tensor components to fit |
| `PT_REG` | float | Diagonal regularisation for P = inv(Re(Z)) @ Im(Z) |
| `SIGMA_FLOOR_Z` | float | Error floor added in quadrature to Z_err |
| `SIGMA_FLOOR_P` | float | Error floor added in quadrature to P_err |

### Parameterisation and bounds

| Variable | Type | Description |
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

| Variable | Type | Description |
|----------|------|-------------|
| `STEP_METHOD` | str | `"demetropolis"`, `"demetropolisz"`, `"metropolis"`, `"nuts"`, or `"hmc"` |
| `ENABLE_GRAD` | bool | Build gradient-enabled likelihood (required for NUTS/HMC) |
| `PROGRESSBAR` | bool | Show PyMC progress bar |

### Prior specification

| Variable | Type | Description |
|----------|------|-------------|
| `PRIOR_KIND` | str | `"default"`, `"uniform"`, or `"gaussian"` |

#### Gaussian prior options (used only when `PRIOR_KIND = "gaussian"`)

| Variable | Type | Description |
|----------|------|-------------|
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

All values can be scalars (broadcast to all layers) or per-layer arrays.

**`PRIOR_COV`** is a `(3*nl, 3*nl)` covariance matrix on the stacked vector
`[log10_ρ_min, log10_ρ_max, strike_deg]`. Build it conveniently with
`mcmc.build_gaussian_cov`, which supports three inter-layer correlation models
selected via the `corr_model` keyword:

| `corr_model` | Kernel | Formula (d = \|i−j\|) |
|---------------|--------|----------------------|
| `None` / `"identity"` | Independent layers | R = I |
| `"exponential"` | Matérn ν=½ | exp(−d / L) |
| `"matern32"` | Matérn ν=3⁄2 | (1 + √3 d/L) exp(−√3 d/L) |
| `"matern52"` | Matérn ν=5⁄2 | (1 + √5 d/L + 5d²/3L²) exp(−√5 d/L) |
| `"matern"` | General Matérn | Bessel-based (ν via `nu` kwarg) |
| `"gaussian"` | Matérn ν→∞ | exp(−d² / 2L²) |

The smoothness parameter ν controls differentiability: ν = ½ is roughest,
ν = 3⁄2 is a good general-purpose default, and ν → ∞ (Gaussian) is smoothest.

The correlation length `L` (`corr_length`) is in layer-index units (L = 2 means
correlation extends over ~2 layers).  An optional `cross_corr` parameter
introduces uniform coupling between the three parameter blocks.

```python
# Exponential inter-layer correlation, length 2 layers
PRIOR_COV = mcmc.build_gaussian_cov(
    nlayer,
    std_log10_rho_min=0.5,
    std_log10_rho_max=0.5,
    std_strike_deg=30.0,
    corr_model="exponential",
    corr_length=2.0,
)

# Matérn ν=3/2 (once differentiable, good default)
PRIOR_COV = mcmc.build_gaussian_cov(
    nlayer,
    std_log10_rho_min=0.5,
    std_log10_rho_max=0.5,
    std_strike_deg=30.0,
    corr_model="matern32",
    corr_length=2.0,
)

# General Matérn with arbitrary ν (requires scipy)
PRIOR_COV = mcmc.build_gaussian_cov(
    nlayer,
    std_log10_rho_min=0.5,
    std_log10_rho_max=0.5,
    std_strike_deg=30.0,
    corr_model="matern",
    corr_length=2.0,
    nu=2.0,
)

# Gaussian (squared-exp) correlation, length 3, with cross-parameter coupling
PRIOR_COV = mcmc.build_gaussian_cov(
    nlayer,
    std_log10_rho_min=0.5,
    std_log10_rho_max=0.5,
    std_strike_deg=30.0,
    corr_model="gaussian",
    corr_length=3.0,
    cross_corr=0.2,
)
```

You can also build block correlation matrices manually:

```python
R_block = mcmc.exponential_corr(nlayer, corr_length=2.0)   # (nl, nl)
R_full  = mcmc.block_corr_matrix(nlayer, corr_within=R_block, cross_corr=0.1)
PRIOR_COV = mcmc.build_gaussian_cov(nlayer, ..., corr=R_full)
```

### Quantile settings

| Variable | Type | Description |
|----------|------|-------------|
| `QPAIRS` | tuple of (float, float) | Quantile/percentile pairs for the posterior summary (values > 1 are interpreted as percentiles) |


## Pre-defined `pmc_dict` presets

The script contains ready-made sampling-parameter dictionaries for common
scenarios. Copy the desired preset into `pmc_dict` to activate it.

### Fixed-thickness (10-layer) presets

| Dict name | Step method | Draws | Tune | Chains |
|-----------|-------------|-------|------|--------|
| `pmc_dict_nuts_10layer_hfixed` | NUTS | 4 000 | 4 000 | 4 |
| `pmc_dict_metropolis_10layer_hfixed` | Metropolis | 120 000 | 30 000 | 4 |
| `pmc_dict_demetropolisz_10layer_hfixed` | DEMetropolisZ | 60 000 | 20 000 | 8 |

### Thickness-sampled (6-layer) presets

| Dict name | Step method | Draws | Tune | Chains |
|-----------|-------------|-------|------|--------|
| `pmc_dict_nuts_6layer_loghm` | NUTS | 6 000 | 6 000 | 4 |
| `pmc_dict_metropolis_6layer_loghm` | Metropolis | 200 000 | 80 000 | 4 |
| `pmc_dict_demetropolisz_6layer_loghm` | DEMetropolisZ | 100 000 | 30 000 | 10 |


## Outputs

For each input site the script writes two files into `OUTDIR`:

| File | Format | Contents |
|------|--------|----------|
| `{station}_pmc_{method}.nc` | NetCDF (ArviZ) | Full posterior trace (InferenceData) |
| `{station}_pmc_{method}_summary.npz` | NumPy NPZ | Compact quantile envelopes for plotting |

The summary NPZ contains per-layer quantiles for ρ_min, ρ_max, σ_min, σ_max,
strike, depth interfaces, and the packed parameter vector θ.


## Execution flow

```
1. Set up environment / paths
2. Define starting model (Model0 or load from NPZ)
3. Normalise model, build ParamSpec
4. For each site file:
   a. Load site
   b. Optionally compute phase tensor
   c. Build PyMC model (with chosen prior kind)
   d. Run MCMC sampling
   e. Save InferenceData (.nc)
   f. Build and save summary (.npz)
```


## Quick-start examples

### Default priors with DEMetropolisZ (black-box)

```python
STEP_METHOD = "demetropolis"
ENABLE_GRAD = False
PRIOR_KIND  = "default"
FIX_H       = True
pmc_dict    = pmc_dict_demetropolisz_10layer_hfixed
```

### NUTS with gradient-enabled likelihood

```python
STEP_METHOD = "nuts"
ENABLE_GRAD = True
PRIOR_KIND  = "default"
FIX_H       = True
pmc_dict    = pmc_dict_nuts_10layer_hfixed
```

### Gaussian prior centred on starting model (diagonal)

```python
PRIOR_KIND = "gaussian"
PRIOR_STD  = {
    "log10_rho": 0.5,
    "strike_deg": 30.0,
}
PRIOR_COV  = None
```

### Gaussian prior with full covariance

```python
PRIOR_KIND = "gaussian"

# Matérn ν=3/2 inter-layer correlation, length 2 layers
PRIOR_COV  = mcmc.build_gaussian_cov(
    nlayer,
    std_log10_rho_min=0.5,
    std_log10_rho_max=0.5,
    std_strike_deg=30.0,
    corr_model="matern32",        # or "exponential", "matern52", "matern", "gaussian"
    corr_length=2.0,              # in layer-index units
    cross_corr=0.0,               # no cross-parameter coupling
)
```


## Provenance

- **Author:** Volker Rath (DIAS)
- **Created:** 2026-02-12 with the help of ChatGPT (GPT-5 Thinking)
- **Gaussian prior option:** added 2026-03-01 with the help of Claude (Opus 4.6, Anthropic)
- **Matérn covariance kernels:** added 2026-03-01 with the help of Claude (Opus 4.6, Anthropic)
- **This README:** generated 2026-03-01 with the help of Claude (Opus 4.6, Anthropic)
