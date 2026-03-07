# transdim.py / transdim_viz.py

Library modules for transdimensional (rjMCMC) 1-D layered-earth inversion.

## Provenance

| Field | Value |
|-------|-------|
| Files | `transdim.py`, `transdim_viz.py` |
| Author | Volker Rath (DIAS) / Claude (Opus 4.6, Anthropic) |
| Part of | **py4mt** — Python for Magnetotellurics |
| Created | 2026-03-07 — split from `transdimensional_mcmc_parallel.py` |
| Modified | 2026-03-07 — QC and posterior-model summary plots |
| Modified | 2026-03-07 — impedance/PT data fit, aniso histograms, depth/freq scale options |
| README generated | 7 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reusable library for reversible-jump MCMC where the number of layers *k* is
itself a free parameter.  The code is split into two files:

| File | Role | matplotlib? |
|------|------|-------------|
| `transdim.py` | Core — data structures, forward models, proposals, samplers, diagnostics, I/O | No |
| `transdim_viz.py` | Plotting — all figure-generation routines | Yes |

The split ensures the sampler can run on headless / HPC nodes without a
display backend.  Import only `transdim` for compute; add `transdim_viz`
when figures are needed.


## Algorithm

The sampler implements the reversible-jump MCMC algorithm (Green 1995) with
four proposal types:

| Proposal | Effect | Dimension change |
|----------|--------|-----------------|
| **Birth** | Insert a new interface at a random depth; split the parent layer's resistivity (and optionally anisotropy) | *k* → *k* + 1 |
| **Death** | Remove a random interface; merge the two neighbouring layers | *k* → *k* − 1 |
| **Move** | Perturb a random interface depth | *k* unchanged |
| **Change** | Perturb a random layer's resistivity, anisotropy ratio, or strike | *k* unchanged |

Acceptance follows the Metropolis–Hastings–Green criterion with log-proposal
ratios that account for the Jacobian of the birth/death dimension jump.

Multiple independent chains are run in parallel via `joblib` and merged.
Inter-chain convergence is assessed with the Gelman–Rubin R-hat statistic on
the log-likelihood trace.

### References

- Green (1995), *Biometrika* — Reversible jump MCMC
- Bodin & Sambridge (2009), *GJI* — Transdimensional tomography
- Malinverno (2002), *Geophysics* — Parsimonious Bayesian inversion


## `transdim.py` — public API

### Data structures

| Class | Purpose |
|-------|---------|
| `LayeredModel` | 1-D earth model: `interfaces` (sorted depths), `resistivities` (log10), optional `aniso_ratios` (ρ_max/ρ_min ≥ 1), `strikes` (degrees).  Properties: `k`, `n_layers`, `is_anisotropic`, `get_thicknesses()`, `get_resistivities()`, `copy()`. |
| `Prior` | Uniform prior bounds on *k*, interface depth, log10(ρ), log10(aniso ratio), strike.  Method: `log_prior(model)` → 0 or −∞. |
| `RjMCMCConfig` | Sampler tuning: `n_iterations`, `burn_in`, `thin`, `proposal_weights`, proposal standard deviations for resistivity, depth, anisotropy, and strike. |

### Forward models

**Apparent-resistivity level** (used by the sampler):

| Function | Returns |
|----------|---------|
| `mt_forward_1d_isotropic(thk, rho, freq)` | `rho_a` array (n_freq,) |
| `mt_forward_1d_isotropic_full(thk, rho, freq)` | dict `{'rho_a', 'phase_deg'}` |
| `mt_forward_1d_anisotropic(thk, rho, freq, aniso, strike)` | dict `{'rho_a_xy', 'rho_a_yx'}` |

**Impedance-tensor level** (used by QC plots):

| Function | Returns |
|----------|---------|
| `mt_forward_1d_isotropic_impedance(thk, rho, freq)` | dict `{'Z' (nf,2,2), 'rho_a', 'phase_deg'}` |
| `mt_forward_1d_anisotropic_impedance(thk, rho, freq, aniso, strike)` | dict `{'Z' (nf,2,2), 'rho_a_xy', 'rho_a_yx'}` |

For isotropic media: Zxx=Zyy=0, Zxy=Z, Zyx=−Z.  The anisotropic model
delegates to `aniso.aniso1d_impedance_sens_simple`; if `aniso.py` is not on
`PYTHONPATH`, only the isotropic models are available
(`has_aniso()` returns `False`).

### Phase tensor

| Function | Description |
|----------|-------------|
| `compute_phase_tensor(Z)` | PT = Re(Z)⁻¹ @ Im(Z) per frequency.  Input: (n_freq, 2, 2) complex → output: (n_freq, 2, 2) real. |

### Likelihood

| Function | Description |
|----------|-------------|
| `log_likelihood(model, freq, obs, sigma, ...)` | Gaussian log-likelihood in log10(ρ_a) space.  Supports both isotropic and anisotropic (xy + optional yx) data. |

### Proposals

Each returns `(new_model, log_proposal_ratio)`:

| Function | Proposal type |
|----------|--------------|
| `propose_birth(model, prior, ...)` | Add interface |
| `propose_death(model, prior, ...)` | Remove interface |
| `propose_move(model, prior, sigma_z)` | Perturb interface depth |
| `propose_change(model, prior, ...)` | Perturb layer property |

All proposal functions accept `use_aniso=True` to additionally perturb
anisotropy ratio and strike.

### Samplers

| Function | Description |
|----------|-------------|
| `run_rjmcmc(...)` | Single chain.  Returns dict with `models`, `log_likes`, `n_layers`, `full_ll_trace`, `acceptance`, `chain_id`. |
| `run_parallel_rjmcmc(...)` | N chains via `joblib`.  Merges posteriors, computes R-hat.  Returns merged dict plus `chains` list, `gelman_rubin`, and `burn_in`. |

`full_ll_trace` is a 1-D array of the log-likelihood at **every** iteration
(including burn-in), enabling misfit-vs-step convergence plots.

### Diagnostics

| Function | Description |
|----------|-------------|
| `gelman_rubin(chain_traces)` | R-hat convergence statistic for a list of 1-D traces.  Values near 1.0 indicate convergence. |

### Posterior analysis

| Function | Description |
|----------|-------------|
| `compute_posterior_profile(models, depth_grid)` | ρ_max(z) statistics: `mean`, `median`, `p05`, `p95`, `ensemble`. |
| `compute_posterior_rhomin_profile(models, depth_grid)` | ρ_min(z) statistics (ρ_max/aniso_ratio; equals ρ_max for isotropic models). |
| `compute_posterior_aniso_profile(models, depth_grid)` | Anisotropy-ratio and strike-angle statistics. |
| `compute_posterior_histogram(models, depth_grid, bins, prop)` | 2-D histogram of a layer property vs depth.  `prop` selects the quantity: `"rho"` (log10 ρ_max, default), `"rho_min"` (log10 ρ_min), or `"strike"` (degrees).  Returns `hist2d`, `value_bins`, `value_centres`, and `mode` profile. |
| `compute_changepoint_frequency(models, depth_grid)` | Interface-placement frequency at each depth. |

### I/O helpers

| Function | Description |
|----------|-------------|
| `save_results_npz(results, path)` | Save the full posterior ensemble to compressed NPZ. |
| `load_results_npz(path)` | Reload into the same dict format. |
| `generate_seed()` | Random integer seed. |


## `transdim_viz.py`

See **`transdim_viz_readme.md`** for the full plotting API.

Summary of available figures:

| Function | Description |
|----------|-------------|
| `plot_results(...)` | Generic 4- or 5-panel diagnostic |
| `plot_qc(...)` | QC summary — ρ_a/phase or Re/Im(Z) data fit, optional phase tensor, misfit trace, k histogram |
| `plot_posterior_model(...)` | Posterior model — 2-D histograms (ρ_max, ρ_min, strike in aniso mode) + change-point frequency; linear or log depth scale |
| `plot_resistivity_profile(ax, ...)` | Single-panel: posterior ρ(z) |
| `plot_dimension_histogram(ax, ...)` | Single-panel: k histogram |
| `plot_data_fit(ax, ...)` | Single-panel: data fit |
| `plot_chain_traces(ax, ...)` | Single-panel: LL traces |
| `plot_aniso_profile(ax, ...)` | Single-panel: anisotropy ratio |
| `plot_strike_profile(ax, ...)` | Single-panel: strike angle |


## Dependencies

| Module | Required by |
|--------|-------------|
| `numpy` | `transdim.py`, `transdim_viz.py` |
| `joblib` | `transdim.py` (imported lazily inside `run_parallel_rjmcmc`) |
| `aniso.py` | `transdim.py` (optional; isotropic mode works without it) |
| `matplotlib` | `transdim_viz.py` only |
