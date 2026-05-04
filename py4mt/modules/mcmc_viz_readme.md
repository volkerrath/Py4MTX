# mcmc_viz.py — Axes-based posterior plots for anisotropic 1-D MT inversion

`mcmc_viz.py` provides **axes-based** (not figure-based) plotting helpers for
the simplified anisotropic 1-D MT posterior produced by `mcmc.py` and
`mt_aniso1d_sampler.py`.  All functions draw into a caller-supplied
`matplotlib.Axes` and return it; the calling script is responsible for figure
layout (`plt.subplots`, etc.).

Matplotlib is imported lazily so that the module can be imported on headless
nodes for non-plotting tasks (loading data, deriving quantities).

---

## Supported parameter sets

`plot_paramset_threepanel` handles all combinations of domain and parameter set:

| `param_domain` | `param_set` | Panels (left → right) |
|---|---|---|
| `"rho"` | `"minmax"` | log₁₀(ρ_min), log₁₀(ρ_max), strike [deg] |
| `"rho"` | `"max_anifac"` | log₁₀(ρ_max), log₁₀(a_ρ = √(ρ_max/ρ_min)), strike [deg] |
| `"sigma"` | `"minmax"` | log₁₀(σ_min), log₁₀(σ_max), strike [deg] |
| `"sigma"` | `"max_anifac"` | log₁₀(σ_max), log₁₀(a_σ = √(σ_max/σ_min)), strike [deg] |

Conductivity variables are derived from resistivity posterior samples when not
stored explicitly (σ = 1/ρ).  Anisotropy factors are derived from min/max
pairs if an explicit `*_anifac` variable is absent.

---

## Data sources

Each panel draws from whichever source is available (preference controlled by
`prefer_idata`):

1. **ArviZ `InferenceData`** (`*.nc` from `mcmc.save_idata`) — full posterior
   samples; supports multiple `qpairs` bands.
2. **Summary NPZ** (`*_pmc_summary.npz` from `mcmc.save_summary_npz`) — pre-
   computed `{base}_qlo`, `{base}_med`, `{base}_qhi` arrays; supports a
   single quantile band only.

---

## Key functions

### High-level

#### `plot_paramset_threepanel(axs, *, summary, idata, param_domain, param_set, ...)`

Plot a three-panel vertical profile figure for one of the parameter sets.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `axs` | sequence of 3 `Axes` | — | Target axes |
| `summary` | mapping or None | `None` | Summary dict from `load_summary_npz` |
| `idata` | `InferenceData` or None | `None` | Full posterior from `open_idata` |
| `param_domain` | `"rho"` \| `"sigma"` | `"rho"` | Resistivity or conductivity |
| `param_set` | `"minmax"` \| `"max_anifac"` | `"minmax"` | Parameter set |
| `qpairs` | sequence of (qlo, qhi) | `((0.1, 0.9),)` | Quantile pairs for shaded bands |
| `prefer_idata` | bool | `True` | Use `idata` samples when available |
| `show_quantile_lines` | bool | `False` | Also draw dashed band-edge lines |
| `overlay_single` | mapping or None | `None` | Single model to overlay (dict with same keys + `h_m`) |

Returns the input `axs` after plotting.

### I/O helpers

| Function | Description |
|----------|-------------|
| `open_idata(nc_path)` | Load ArviZ `InferenceData` from a `.nc` file |
| `load_summary_npz(path)` | Load a `*_pmc_summary.npz` into a plain dict |

### Low-level profile primitives

| Function | Description |
|----------|-------------|
| `plot_vertical_profile(ax, *, h_m, values, label, use_log10, ...)` | Single step profile |
| `plot_vertical_envelope(ax, *, h_m, qlo, med, qhi, label, use_log10, ...)` | Median + shaded (qlo, qhi) band from pre-computed quantiles |
| `plot_vertical_bands_from_samples(ax, *, h_m, samples, qpairs, ...)` | Median + one or more shaded bands computed from posterior samples |
| `depth_edges_from_h(h_m, *, ignore_basement)` | Thickness array → depth-edge array for step plots |

---

## Typical usage

```python
import matplotlib.pyplot as plt
import mcmc_viz as mviz

# Load posterior
idata   = mviz.open_idata("site001_pmc.nc")
summary = mviz.load_summary_npz("site001_pmc_summary.npz")

fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

mviz.plot_paramset_threepanel(
    axs,
    idata=idata,
    summary=summary,
    param_domain="rho",
    param_set="minmax",
    qpairs=[(0.25, 0.75), (0.05, 0.95)],   # inner and outer bands
)

fig.suptitle("Site 001 — posterior (ρ_min, ρ_max, strike)")
fig.tight_layout()
plt.savefig("site001_posterior.pdf")
```

---

## Dependencies

| Package | Required |
|---------|----------|
| `numpy` | Yes |
| `matplotlib` | Yes (lazy) |
| `arviz` | Only for `open_idata` and idata-based plotting |
| `xarray` | Only when stacking `idata.posterior` chains |

---

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-13 (UTC)
