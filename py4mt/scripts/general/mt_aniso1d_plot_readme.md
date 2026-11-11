# mt_aniso1d_plot.py

Plotting driver for anisotropic 1-D MT inversion results.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_aniso1d_plot.py` |
| Author | Volker Rath (DIAS), with ChatGPT (GPT-5 Thinking), 2026-02-13 |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 3 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Produces a three-panel figure per station showing vertical step profiles
of the sampled parameter sets with posterior uncertainty bands. Reads
sampler output (summary NPZ and optionally ArviZ NetCDF InferenceData)
and generates publication-ready plots.

## Workflow

1. Glob sampler summary files from `SUMM_DIR`.
2. For each station:
   - Load summary NPZ and optionally the NetCDF InferenceData.
   - Infer the parameterisation (domain, set) from metadata or variable names.
   - Plot a 3×1 figure (rho_min/sigma_min, rho_max/sigma_max, strike) with
     shaded uncertainty bands at the configured quantile pairs.
   - Save figure to `PLOT_DIR` in the requested format(s).

## Panel contents

Depends on the parameterisation used by the sampler:

| Domain | Param set | Panel 1 | Panel 2 | Panel 3 |
|--------|-----------|---------|---------|---------|
| `rho` | `minmax` | ρ_min | ρ_max | strike |
| `rho` | `max_anifac` | ρ_max | ρ_anifac | strike |
| `sigma` | `minmax` | σ_min | σ_max | strike |
| `sigma` | `max_anifac` | σ_max | σ_anifac | strike |

## Configuration constants

| Constant | Description |
|----------|-------------|
| `PARAM_DOMAIN` | `"auto"`, `"rho"`, or `"sigma"`. |
| `PARAM_SET` | `"auto"`, `"minmax"`, or `"max_anifac"`. |
| `BANDS` | Uncertainty band pairs as quantiles (0–1) or percentiles (0–100). |
| `SHOW_BAND_EDGES` | Show dashed lines at band edges in addition to shading. |
| `FIGURE_TITLE` | Figure title (2026-11-11). `"auto"` = `<station> \| <domain>:<set>` as before; `None`, `False`, `""` = none; text with `{station}`, `{domain}`, `{param_set}`, `{key}` (= the station); dict `{"<station>": spec, "default": spec}`. |
| `DATA_DIR` | Base data directory. |
| `SUMM_DIR` | Directory containing sampler summary NPZ files. |
| `PLOT_DIR` | Output directory for plots (created if missing). |
| `PLOT_FORMATS` | Tuple of file extensions (e.g. `(".pdf",)`). |
| `NAME_SUFFIX` | Suffix appended to plot filenames. |
| `SEARCH_GLOB` | Glob pattern to find summary NPZ files. |

## Inputs

| File | Description |
|------|-------------|
| `*_pmc_summary.npz` | Quantile envelopes and metadata from the sampler. |
| `*_pmc.nc` (optional) | Full ArviZ InferenceData for posterior-based bands. |

## Outputs

| File | Description |
|------|-------------|
| `<station><NAME_SUFFIX>.<fmt>` | Three-panel plot per station (PDF and/or PNG at 600 DPI). |

## Dependencies

`matplotlib`, `arviz`; py4mt: `mcmc`, `mcmc_viz`, `util`, `version`.

## Changes 2026-11-11 (titles)

`FIGURE_TITLE` (table above), via `util.resolve_title` (needs the `util.py` of the same date). The figure has no panel titles of its own, so there is no `PANEL_TITLES`. Claude Sonnet 5.5 (Anthropic): AI-generated; review before use. Defaults give the figures of before. Only compile-checked and tested at the level of the title logic (`util.resolve_title`, `test_util_titles.py`); the script was not run end to end.
