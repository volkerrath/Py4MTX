# `mcmc.py` — PyMC utilities for anisotropic 1-D MT inversion

## Purpose

`mcmc.py` is the core library module for Bayesian inversion of magnetotelluric
(MT) data using a simplified anisotropic 1-D earth model. It is designed to be
imported by script-style drivers such as `mt_aniso1d_sampler.py` (sampling) and
`mt_aniso1d_plot.py` (plotting).

The module provides everything between reading observed data and writing
posterior summaries: model I/O and normalisation, data packing, forward-model
wrappers (black-box and gradient-enabled), prior specification, PyMC model
construction, sampling, and post-processing.


## Dependencies

| Package | Role |
|---------|------|
| **numpy** | Array operations (always required) |
| **pymc** | Probabilistic model and sampling (required at sampling time) |
| **pytensor** | Tensor graph, `wrap_py`, custom `Op` (required at sampling time) |
| **arviz** | InferenceData I/O (required for saving / summary) |
| **aniso** | Local forward model (`aniso1d_impedance_sens_simple`) |
| **data_proc** | Site I/O helpers provided by the Py4MTX codebase |

PyMC, PyTensor, and ArviZ are imported lazily so that lightweight tasks (model
normalisation, packing) can run without them.


## Per-layer parameterisation

Each layer is described by four physical quantities plus two boolean flags:

| Symbol | Array key | Unit | Description |
|--------|-----------|------|-------------|
| h | `h_m` | m | Layer thickness (last entry = basement placeholder, usually 0) |
| ρ_min | `rho_min` | Ω·m | Minimum horizontal resistivity |
| ρ_max | `rho_max` | Ω·m | Maximum horizontal resistivity |
| α | `strike_deg` | ° | Anisotropy strike angle |
| — | `is_iso` | bool | If `True`, enforce ρ_max = ρ_min (isotropic layer) |
| — | `is_fix` | bool | If `True`, freeze this layer at the starting model |

The module always stores **both** resistivity (`rho_min`, `rho_max`) and
conductivity (`sigma_min`, `sigma_max`) representations internally, regardless
of which domain is used for sampling.


## Likelihood implementations

Two forward-model wrappers are available, selected by the `enable_grad` flag
in `build_pymc_model`:

1. **Black-box** (`enable_grad=False`, default) — wraps the NumPy forward
   model with `pytensor.wrap_py`.  Very robust; no derivatives.  Use with
   `DEMetropolisZ` or `Metropolis`.

2. **Gradient-enabled** (`enable_grad=True`) — custom PyTensor `Op`
   (`_ForwardPackedOp`) whose `grad()` method returns vector–Jacobian products
   computed from analytic impedance sensitivities.  Supports NUTS / HMC for
   impedance likelihoods and, optionally, phase-tensor components.

The likelihood is always a multivariate Normal on the packed observation vector
(impedance Re/Im pairs, optionally followed by phase-tensor components).


## Prior kinds

The `prior_kind` argument of `build_pymc_model` selects the prior family:

### `"default"` (alias `"soft"`)

NUTS-friendly bounded transforms via sigmoid mappings, with an explicit
anisotropy-ratio parameter biased toward isotropy (δ near 0).  No hard walls
on log10(ρ); bounds are expressed as soft sigmoid saturation.

### `"uniform"`

Broad `pm.Uniform` priors on two independent log10 fields; the ordered pair
(ρ_min ≤ ρ_max) is enforced by taking element-wise min/max.

### `"gaussian"`

**Added 2026-03-01.** Multivariate or independent Normal prior centred on the
starting model with a user-prescribed covariance.  Two sub-modes:

| Mode | Trigger | Prior variables |
|------|---------|-----------------|
| **Diagonal** | `prior_std` dict, `prior_cov=None` | Independent `pm.Normal` per parameter |
| **Full covariance** | `prior_cov` array (shape 3·nl × 3·nl) | Single `pm.MvNormal` over `[log10_ρ_min, log10_ρ_max, strike]` |

In both cases the ordering ρ_min ≤ ρ_max is enforced after sampling via
`pt.minimum` / `pt.maximum`.

**`prior_std` keys** (all optional; missing keys receive defaults):

| Key | Applies to | Default |
|-----|-----------|---------|
| `"log10_rho"` | Both log10(ρ_min) and log10(ρ_max) | 1.0 |
| `"log10_rho_min"` | log10(ρ_min) only (overrides shared key) | — |
| `"log10_rho_max"` | log10(ρ_max) only (overrides shared key) | — |
| `"strike_deg"` | Strike (degrees) | 45.0 |
| `"log10_h"` | log10(h) per layer (when `fix_h=False`) | 0.5 |
| `"log10_H"` | log10(H_m) global scale (when `sample_H_m=True`) | 0.5 |

Every value can be a scalar (broadcast to all layers) or a per-layer array.

**`prior_cov`** is the full covariance matrix on the stacked vector
`[log10_ρ_min(0..nl-1), log10_ρ_max(0..nl-1), strike(0..nl-1)]`.
When provided, it takes precedence over the resistivity/strike entries in
`prior_std`; thickness priors (`log10_h`, `log10_H`) are still controlled
independently via `prior_std`.


## Convenience helper: `build_gaussian_cov`

```python
C = mcmc.build_gaussian_cov(
    nl,
    std_log10_rho_min=1.0,
    std_log10_rho_max=1.0,
    std_strike_deg=45.0,
    corr=None,            # explicit (3*nl, 3*nl) correlation matrix
    corr_model=None,      # or "exponential" / "gaussian"
    corr_length=1.0,      # in layer-index units (for corr_model)
    cross_corr=0.0,       # inter-block cross-correlation (for corr_model)
)
```

Constructs the covariance as `C = diag(σ) @ R @ diag(σ)` where σ is the
stacked standard-deviation vector and R is the correlation matrix.

The correlation matrix R is determined by (in order of priority):

1. `corr` — an explicit `(3*nl, 3*nl)` matrix.
2. `corr_model` — one of `"exponential"` or `"gaussian"`, built automatically
   via `block_corr_matrix` using `corr_length` and `cross_corr`.
3. If neither is given, R = I (fully independent parameters).


## Inter-layer correlation models

Two geostatistically motivated correlation kernels are provided, based on the
layer-index distance d = |i − j|:

| Model | Function | Formula | Adj. layer corr (L=1) |
|-------|----------|---------|----------------------|
| Exponential (Matérn-½) | `exponential_corr(nl, L)` | exp(−d / L) | 0.37 |
| Gaussian (squared-exp.) | `gaussian_corr(nl, L)` | exp(−d² / 2L²) | 0.61 |

The correlation length `L` is given in **layer-index units**:

- `L = 1` — significant correlation only between immediate neighbours.
- `L = 3` — correlation extends across roughly 3 layers.
- `L → ∞` — all layers perfectly correlated (approaches a 1-parameter model).

Both functions return an `(nl, nl)` symmetric positive-definite matrix with
unit diagonal.


## Block correlation assembly: `block_corr_matrix`

```python
R = mcmc.block_corr_matrix(
    nl,
    corr_within="exponential",   # or "gaussian", "identity", or an (nl,nl) array
    corr_length=2.0,
    cross_corr=0.1,              # optional coupling between parameter blocks
)
```

Assembles a `(3*nl, 3*nl)` correlation matrix with three diagonal blocks (one
per parameter family: ρ_min, ρ_max, strike) sharing the same `corr_within`
kernel and uniform off-diagonal cross-correlation `cross_corr` between blocks.

The result can be passed directly as the `corr` argument to
`build_gaussian_cov`, or to `build_pymc_model` via `prior_cov`.


## Low-level helpers

| Function | Returns | Description |
|----------|---------|-------------|
| `exponential_corr(nl, L)` | (nl, nl) | Exponential kernel on index distance |
| `gaussian_corr(nl, L)` | (nl, nl) | Squared-exponential kernel on index distance |
| `block_corr_matrix(nl, ...)` | (3·nl, 3·nl) | Block-diagonal assembly with optional cross-correlation |
| `_index_distance_matrix(nl)` | (nl, nl) | Raw \|i−j\| distance matrix (internal) |


## Thickness sampling modes

Controlled by `ParamSpec.fix_h` and `ParamSpec.sample_H_m`:

| `fix_h` | `sample_H_m` | Behaviour |
|---------|-------------|-----------|
| `True` | `False` | Thicknesses frozen at starting model |
| `True` | `True` | Single global scale H_m sampled; relative profile fixed |
| `False` | `False` | Per-layer log10(h) sampled (last layer optionally fixed) |


## Key public functions

| Function | Description |
|----------|-------------|
| `normalize_model(model)` | Validate and normalise a model dict (ensures both ρ and σ fields) |
| `model_from_direct(model)` | Convert an in-script template to a model dict |
| `save_model_npz` / `load_model_npz` | Persist / load model dicts as NPZ |
| `load_site(path)` | Load an MT site from `.edi` or `.npz` |
| `ensure_phase_tensor(site)` | Compute P = inv(Re(Z)) @ Im(Z) and optionally bootstrap P_err |
| `build_gaussian_cov(nl, ...)` | Build a covariance matrix for the Gaussian prior |
| `ParamSpec(...)` | Container for inversion bounds and flags |
| `build_pymc_model(site, spec, ...)` | Construct the PyMC model for one site |
| `sample_pymc(model, pmc_dict=...)` | Run PyMC sampling with a flexible dict-based interface |
| `save_idata(idata, path)` | Write ArviZ InferenceData to NetCDF |
| `build_summary_npz(...)` | Extract quantile envelopes from the posterior |
| `save_summary_npz(summary, path)` | Write summary dict to NPZ |

### Data-packing utilities

| Function | Description |
|----------|-------------|
| `pack_Z_vector(Z, comps)` | Pack complex Z into a real Re/Im vector |
| `pack_Z_sigma(Z_err, comps)` | Pack Z uncertainties to match `pack_Z_vector` |
| `pack_P_vector(P, comps)` | Pack real phase-tensor components |
| `pack_P_sigma(P_err, comps)` | Pack P uncertainties |
| `phase_tensor_from_Z(Z)` | Compute P = inv(Re(Z)) @ Im(Z) |
| `phase_tensor_dP_from_dZ(Z, dZ)` | Phase-tensor derivatives from impedance derivatives |


## Component labels

Impedance and phase-tensor components are addressed by short string labels
mapped to (i,j) matrix indices:

| Label | Aliases | Index |
|-------|---------|-------|
| `"xx"` | `"zxx"`, `"pxx"` | (0, 0) |
| `"xy"` | `"zxy"`, `"pxy"` | (0, 1) |
| `"yx"` | `"zyx"`, `"pyx"` | (1, 0) |
| `"yy"` | `"zyy"`, `"pyy"` | (1, 1) |


## Sampling workflow (high level)

```
load_site  →  ensure_phase_tensor  →  build_pymc_model  →  sample_pymc
                                                               ↓
                                                          save_idata
                                                               ↓
                                                       build_summary_npz  →  save_summary_npz
```


## Provenance

- **Author:** Volker Rath (DIAS)
- **Created:** 2026-02-13 with the help of ChatGPT (GPT-5 Thinking)
- **Gaussian prior option:** added 2026-03-01 with the help of Claude (Opus 4.6, Anthropic)
