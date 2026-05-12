# mt_transdim1d.py

Script driver for transdimensional (rjMCMC) 1-D MT inversion.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_transdim1d.py` |
| Author | Volker Rath (DIAS) / Claude (Opus 4.6, Anthropic) |
| Part of | **py4mt** — Python for Magnetotellurics |
| Created | 2026-03-07 |
| Modified | 2026-03-07 — anisotropic example block, viz split |
| Modified | 2026-03-07 — QC and posterior-model summary plots |
| Modified | 2026-03-07 — impedance/PT data fit, aniso histograms, depth scale options |
| Modified | 2026-03-08 — data I/O via data_proc (load_edi, load_npz, compute_rhophas, compute_pt) |
| README generated | 8 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Script-style driver that runs reversible-jump MCMC inversion of
magnetotelluric apparent-resistivity data for a 1-D layered earth where the
number of layers *k* is itself a free parameter.  The script supports both
**isotropic** and **anisotropic** forward models; switch between them by
setting `USE_ANISO` and choosing the appropriate starting model (`MODEL0` or
`MODEL0_ANISO`).

It is intentionally **not** a CLI; all settings live in the `USER CONFIG`
sections at the top of the file.  Edit, then run:

```bash
python mt_transdim1d.py
```

All sampling logic is delegated to `transdim.py`; all plotting to
`transdim_viz.py`.  The script follows the same conventions as
`mt_aniso1d_sampler.py` (UPPERCASE constants, PY4MTX environment variables,
startup banner, MODEL0 dicts, QPAIRS, OUTDIR).


## Environment setup

Two environment variables must be set before launching the script:

| Variable | Description |
|----------|-------------|
| `PY4MTX_ROOT` | Root of the Py4MTX installation |
| `PY4MTX_DATA` | Base data directory |


## Input data

Two input formats are supported, controlled by `INPUT_FORMAT`:

### EDI files (`INPUT_FORMAT = "edi"`)

All `.edi` files in `EDI_DIR` are discovered via
`data_proc.get_edi_list()` and read via `data_proc.load_edi()`.
Apparent resistivity and phase are computed automatically from the
impedance tensor Z via `data_proc.compute_rhophas()`.  The phase tensor
is computed via `data_proc.compute_pt()` when `COMPUTE_PT = True`.

### NPZ files (`INPUT_FORMAT = "npz"`, default)

Files matched by `INPUT_GLOB` are read via `data_proc.load_npz()`.
These are typically produced by `mt_data_processor.py` (via
`data_proc.save_npz()`).  If the NPZ contains `Z` but not `rho`,
apparent resistivity is computed automatically.

### Key lookup priority

The loader (`_load_site`) accepts data from either format and resolves
fields in this order:

| Field | Lookup (first match wins) |
|-------|--------------------------|
| Frequencies | `freq` → `frequencies` → `1/period` |
| Apparent resistivity | Compute from `Z` → `rho[:,0,1]` → `rho_a` → `rho_a_xy` |
| Uncertainties | Compute from `rho_err` → `sigma` → fall back to `NOISE_LEVEL` |
| Phase tensor | `P` → `PT` → compute from `Z` (when `COMPUTE_PT=True`) |

When Z is available, the impedance-level QC plot (Re/Im of Z components)
and phase-tensor panels are enabled automatically.


## Starting model

Three sources are supported (controlled by `MODEL_DIRECT`):

1. **`MODEL0`** (isotropic, default) — 4-layer half-space, 100 Ω·m.

2. **`MODEL0_ANISO`** (anisotropic) — 6-layer model with anisotropy
   ratios 1–5 and strikes 0°–45°.

3. **External NPZ** — set `MODEL_DIRECT = None` and point `MODEL_NPZ` to an
   existing file.

### MODEL0_ANISO layer table

| Layer | h [m] | ρ_max [Ω·m] | ρ_min [Ω·m] | Ratio | Strike [°] |
|-------|-------|-------------|-------------|-------|-----------|
| 1 | 50 | 200 | 200 | 1 | 0 |
| 2 | 150 | 300 | 100 | 3 | 30 |
| 3 | 500 | 500 | 100 | 5 | 45 |
| 4 | 800 | 800 | 200 | 4 | 45 |
| 5 | 1500 | 300 | 300 | 1 | 0 |
| 6 (basement) | — | 100 | 100 | 1 | 0 |


## USER CONFIG reference

### Forward model

| Constant | Type | Description |
|----------|------|-------------|
| `USE_ANISO` | bool | `False` = isotropic (default); `True` = anisotropic via `aniso.py` |

### Data and I/O

| Constant | Type | Description |
|----------|------|-------------|
| `MCMC_DATA` | str | Base directory for input data and outputs |
| `INPUT_FORMAT` | str | `"npz"` (default) or `"edi"` — selects input reader |
| `EDI_DIR` | str | Directory scanned for `.edi` files (when `INPUT_FORMAT="edi"`) |
| `INPUT_GLOB` | str | Glob pattern for `.npz` files (when `INPUT_FORMAT="npz"`) |
| `MODEL_NPZ` | str | Path for saving / loading the starting model |
| `MODEL_DIRECT` | dict or None | `MODEL0` (iso), `MODEL0_ANISO` (aniso), or `None` (load from NPZ) |
| `OUTDIR` | str | Output directory |
| `NOISE_LEVEL` | float | Default data uncertainty in log10(ρ_a) space |
| `SIGMA_FLOOR` | float | Minimum uncertainty (added as floor) |
| `ERR_METHOD` | str | Error method for `compute_rhophas`: `"none"`, `"analytic"`, `"bootstrap"`, `"both"` |
| `ERR_NSIM` | int | Monte-Carlo samples for bootstrap error estimation |
| `COMPUTE_PT` | bool | Compute phase tensor from Z for QC plots |

### Prior bounds

| Constant | Type | Description |
|----------|------|-------------|
| `K_MIN` / `K_MAX` | int | Bounds on number of internal interfaces |
| `DEPTH_MIN` / `DEPTH_MAX` | float | Shallowest / deepest allowed interface [m] |
| `LOG10_RHO_BOUNDS` | (float, float) | Bounds on log10(resistivity) |
| `LOG10_ANISO_BOUNDS` | (float, float) | Bounds on log10(ρ_max/ρ_min); aniso only |
| `STRIKE_BOUNDS_DEG` | (float, float) | Bounds on strike angle [deg]; aniso only |

### Sampler settings

| Constant | Type | Description |
|----------|------|-------------|
| `N_ITERATIONS` | int | Total MCMC iterations per chain |
| `BURN_IN` | int | Iterations to discard before collecting samples |
| `THIN` | int | Keep every N-th sample after burn-in |
| `PROPOSAL_WEIGHTS` | (float×4) | Relative weights for (birth, death, move, change) |

### Proposal standard deviations

| Constant | Applies to |
|----------|-----------|
| `SIGMA_BIRTH_RHO` | log10(ρ) perturbation on birth |
| `SIGMA_MOVE_Z` | Interface depth perturbation [m] |
| `SIGMA_CHANGE_RHO` | log10(ρ) perturbation on change |
| `SIGMA_BIRTH_ANISO` | log10(ratio) perturbation on birth (aniso) |
| `SIGMA_BIRTH_STRIKE` | Strike perturbation on birth [deg] (aniso) |
| `SIGMA_CHANGE_ANISO` | log10(ratio) perturbation on change (aniso) |
| `SIGMA_CHANGE_STRIKE` | Strike perturbation on change [deg] (aniso) |

### Parallel chains

| Constant | Type | Description |
|----------|------|-------------|
| `N_CHAINS` | int | Number of independent parallel chains |
| `N_JOBS` | int | `joblib` parallelism: `-1` = all CPUs, `1` = sequential |
| `BASE_SEED` | int | Chain *i* gets seed `BASE_SEED + i` |

### Output and plotting

| Constant | Type | Description |
|----------|------|-------------|
| `DEPTH_GRID_MAX` | float | Maximum depth for profile plots [m] |
| `QPAIRS` | tuple of (int, int) | Quantile pairs for the summary NPZ |
| `PROGRESSBAR` | bool | Verbose chain output |


## Outputs

For each input site the script writes five files into `OUTDIR`:

| File | Format | Contents |
|------|--------|----------|
| `{station}_rjmcmc.npz` | NumPy NPZ | Full posterior ensemble |
| `{station}_rjmcmc_summary.npz` | NumPy NPZ | Quantile envelopes on a depth grid |
| `{station}_rjmcmc.png` | PNG | Generic multi-panel diagnostic |
| `{station}_rjmcmc_qc.png` | PNG | **QC summary** (panels A–D, optionally PT) |
| `{station}_rjmcmc_model.png` | PNG | **Posterior model summary** (panels E/E1–E3 + F) |

### QC summary (`_rjmcmc_qc.png`)

The top-row mode is selected automatically based on the site data:

| Site data available | Panel A | Panel B |
|---------------------|---------|---------|
| rho_a only | Apparent resistivity vs period | Impedance phase vs period |
| Z tensor | Re(Z) components vs period | Im(Z) components vs period |

When `Z` is present and `USE_ANISO=True` (or `PT` is in the site file), an
additional row shows phase-tensor components (PT-1, PT-2).

The bottom row always contains panel C (data misfit vs sampling step,
per chain) and panel D (number-of-layers histogram).

### Posterior model summary (`_rjmcmc_model.png`)

| `USE_ANISO` | Layout |
|-------------|--------|
| `False` | E (ρ vs depth 2-D histogram) + F (change-point frequency) |
| `True` | E1 (ρ_max) + E2 (ρ_min) + E3 (strike) + F (change-point frequency) |

Each histogram panel overlays median (black), mean (blue), mode (green),
10th/90th percentiles (dashed), and optional true-model staircase (red).
The depth axis can be set to linear or logarithmic via the `depth_scale`
parameter in `plot_posterior_model`.


## Dependencies

`numpy`, `joblib` (via `transdim`), `matplotlib` (via `transdim_viz`);
optionally `aniso.py` (for anisotropic forward model);
py4mt: `data_proc`, `util`, `version`.
