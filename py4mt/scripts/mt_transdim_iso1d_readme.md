# mt_transdim_iso1d.py — rjMCMC driver for isotropic 1-D MT inversion

Script-style driver for transdimensional (rjMCMC) **isotropic** 1-D MT
inversion.  The number of layers *k* is a free parameter sampled by the
reversible-jump MCMC algorithm.  Multiple independent chains run in parallel
via `joblib` and are merged with Gelman–Rubin R-hat convergence diagnostics.

This is intentionally **not a CLI** — edit the `USER CONFIG` block and run:

```bash
python mt_transdim_iso1d.py
```

Delegates sampling to `transdim.py`, plotting to `transdim_viz.py`, and data
I/O to `data_proc.py` (via `transdim.load_site()`).

---

## Likelihood modes

| `LIKELIHOOD_MODE` | Data fitted | Notes |
|---|---|---|
| `"Zdet"` | Re + Im of determinant impedance Z_det | Recommended; doubles effective data count vs ρ_a |
| `"rhoa"` | log₁₀ apparent resistivity from Z_det | Legacy / fallback |

If Z_det is not available at run time the script falls back to `"rhoa"`
automatically.  The anisotropic Z-component / phase-tensor likelihoods are
**not** available in this driver — use `mt_transdim_aniso1d.py` for those.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PY4MTX_ROOT` | Yes | Repository root; modules loaded from `$PY4MTX_ROOT/py4mt/modules/` |
| `PY4MTX_DATA` | Yes | Data root (informational; script exits if unset) |

---

## USER CONFIG reference

All configuration lives in UPPERCASE constants at the top of the script.

### Starting model

An isotropic 5-layer starting model (`MODEL0`) is provided in-file.  Set
`MODEL_DIRECT = MODEL0` to use it; set `MODEL_DIRECT = None` to load from
`MODEL_NPZ` instead.

| Constant | Default | Description |
|----------|---------|-------------|
| `MODEL_DIRECT` | `MODEL0` | In-script model dict, or `None` to load from NPZ |
| `MODEL_NPZ` | `MCMC_DATA + "model0.npz"` | Fallback NPZ path |
| `MODEL_DIRECT_SAVE_PATH` | `MODEL_NPZ` | Where to save the in-script model |
| `MODEL_DIRECT_OVERWRITE` | `True` | Overwrite NPZ even if it exists |

### Forward model / likelihood

| Constant | Default | Description |
|----------|---------|-------------|
| `USE_ANISO` | `False` | Must remain `False` for this isotropic driver |
| `LIKELIHOOD_MODE` | `"Zdet"` | `"Zdet"` or `"rhoa"` |

### Data

| Constant | Default | Description |
|----------|---------|-------------|
| `INPUT_GLOB` | `MCMC_DATA + "*proc.npz"` | Glob pattern for input files (EDI or NPZ) |
| `INPUT_FORMAT` | `"npz"` | `"npz"` or `"edi"` |
| `NOISE_LEVEL` | `0.02` | Relative noise floor for error estimation |
| `SIGMA_FLOOR` | `0.0` | Absolute sigma floor in log₁₀(ρ_a) units |
| `ERR_METHOD` | `"bootstrap"` | Error estimation method passed to `transdim.load_site` |
| `ERR_NSIM` | `200` | Bootstrap realisations |
| `COMPUTE_PT` | `True` | Compute phase tensor on load |

### Prior bounds

| Constant | Default | Description |
|----------|---------|-------------|
| `K_MIN` / `K_MAX` | `1` / `20` | Layer count bounds |
| `DEPTH_MIN` / `DEPTH_MAX` | `50.0` / `3000.0` m | Interface depth bounds |
| `LOG10_RHO_BOUNDS` | `(-1.0, 4.0)` | log₁₀(ρ) prior range [log₁₀ Ω·m] |
| `LOG10_ANISO_BOUNDS` | `(0.0, 1.5)` | log₁₀(aniso ratio) bounds (unused in iso mode) |
| `STRIKE_BOUNDS_DEG` | `(-90.0, 90.0)` | Strike bounds (unused in iso mode) |

### Sampler

| Constant | Default | Description |
|----------|---------|-------------|
| `N_ITERATIONS` | `250 000` | MCMC steps per chain |
| `BURN_IN` | `50 000` | Burn-in steps discarded per chain |
| `THIN` | `10` | Thinning interval |
| `PROPOSAL_WEIGHTS` | `(0.20, 0.20, 0.25, 0.35)` | Relative weights for Birth, Death, Move, Change |
| `SIGMA_BIRTH_RHO` | `0.03` | log₁₀(ρ) std for birth proposal |
| `SIGMA_MOVE_Z` | `50.0` m | Interface-depth std for move proposal |
| `SIGMA_CHANGE_RHO` | `0.05` | log₁₀(ρ) std for change proposal |

### Parallel chains

| Constant | Default | Description |
|----------|---------|-------------|
| `N_CHAINS` | `12` | Number of independent chains |
| `N_JOBS` | `12` | Parallel workers (joblib) |
| `BASE_SEED` | random | Master seed; per-chain seeds derived from it |

### Output

| Constant | Default | Description |
|----------|---------|-------------|
| `OUTDIR` | `MCMC_DATA + "rjmcmc_iso"` | Output directory (created if absent) |
| `DEPTH_GRID_MAX` | `3000.0` m | Maximum depth for posterior profile grid |
| `QPAIRS` | `((q5, q95), (q16, q84))` | Quantile pairs for summary (1-σ and 2-σ bands) |
| `PROGRESSBAR` | `True` | Show tqdm progress bar per chain |

---

## Per-station output files

For each input file matching `INPUT_GLOB`, the following files are written to
`OUTDIR`:

| File | Description |
|---|---|
| `{station}_rjmcmc.npz` | Full posterior ensemble (models, log-likelihoods, n_layers, R-hat) |
| `{station}_rjmcmc_summary.npz` | Pre-computed percentile profiles for `mcmc_viz.py` / `transdim_viz.py` |
| `{station}_rjmcmc.png` | Generic 4-panel diagnostic (profile, k histogram, data fit, LL traces) |
| `{station}_rjmcmc_qc.png` | QC summary (Z_det Re/Im fit, misfit trace, k histogram) |
| `{station}_rjmcmc_model.png` | 2-D posterior histogram + change-point frequency |
| `{station}_rjmcmc_metadata.json` | Full run metadata (prior, config, posterior summary) |
| `{station}_rjmcmc_metadata.npz` | Same metadata in NPZ format |

---

## Workflow

```
set PY4MTX_ROOT, PY4MTX_DATA
edit USER CONFIG (INPUT_GLOB, N_ITERATIONS, N_CHAINS, ...)
python mt_transdim_iso1d.py
    → transdim.load_site() — load EDI / NPZ, compute ρ_a, Z_det, PT
    → transdim.run_parallel_rjmcmc() — N_CHAINS × N_ITERATIONS steps
    → transdim.save_results_npz() — posterior ensemble
    → transdim.build_rjmcmc_summary() — percentile profiles
    → transdim_viz.plot_results / plot_qc / plot_posterior_model — figures
    → export_metadata() — JSON + NPZ provenance record
```

---

## Dependencies

| Module | Role |
|--------|------|
| `transdim` | Sampler, site loading, prior, config, I/O |
| `transdim_viz` | All figures |
| `util` | `print_title`, `get_percentile` |
| `version` | Version string |
| `numpy` | Array operations |
| `joblib` | Parallel chains (via `transdim`) |

---

## References

- Green (1995) *Biometrika* — Reversible jump MCMC
- Bodin & Sambridge (2009) *GJI* — Transdimensional tomography
- Malinverno (2002) *Geophysics* — Parsimonious Bayesian inversion

---

Author: Volker Rath (DIAS)
Created: 2026-03-07, Claude (Opus 4.6, Anthropic)
Modified: 2026-03-09 — Z_det likelihood; helpers moved to transdim.py
