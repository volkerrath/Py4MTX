# transdim_viz.py

Plotting routines for transdimensional rjMCMC results.

## Provenance

| Field | Value |
|-------|-------|
| File | `transdim_viz.py` |
| Author | Volker Rath (DIAS) / Claude (Opus 4.6, Anthropic) |
| Part of | **py4mt** — Python for Magnetotellurics |
| Created | 2026-03-07 — split from `transdim.py` |
| Modified | 2026-03-07 — QC and posterior-model summary plots |
| Modified | 2026-03-07 — impedance/PT data fit, aniso histograms, depth/freq scale options |
| Modified | 2026-11-11 — titles and colour bar position (Claude Sonnet 5.5, AI-generated, review before use), see *Titles and colour bars* |
| README generated | 7 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Separated from `transdim.py` so that the core sampler module has no
matplotlib dependency (useful for headless / HPC environments).

All functions accept the results dict returned by
`transdim.run_rjmcmc` or `transdim.run_parallel_rjmcmc`.

## Component colour convention

When plotting individual impedance or phase-tensor components, the following
colour scheme is used consistently:

| Component | Colour |
|-----------|--------|
| xx | purple |
| xy | blue |
| yx | red |
| yy | green |


## Summary figures

### `plot_results`

Generic 4-panel diagnostic (5 panels when `use_aniso=True`): resistivity
profile, number-of-layers histogram, data fit, chain traces, and (optionally)
anisotropy-ratio profile.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | dict | — | Sampler output |
| `true_model` | LayeredModel or None | — | Optional true model for overlay |
| `frequencies` | (nf,) | — | Observation frequencies [Hz] |
| `observed` | (nf,) | — | Observed apparent resistivity [Ω·m] |
| `depth_max` | float | 3000 | Maximum depth for profile [m] |
| `use_aniso` | bool | False | Add anisotropy-ratio panel |
| `save_path` | str or None | None | Path to save PNG |


### `plot_qc`

QC summary with flexible data-fit panels.  The top-row mode is selected
automatically based on the arguments provided.

**Data-fit modes:**

| Mode | Trigger | Panels A & B |
|------|---------|-------------|
| **ρ_a + phase** | `observed_Z` is `None` (default) | Apparent resistivity and impedance phase vs period |
| **Impedance** | `observed_Z` provided | Re(Z) and Im(Z) for selected components vs period |

When `show_pt=True`, an additional middle row shows phase-tensor components
(two panels, PT-1 and PT-2).  The phase tensor is auto-computed from Z via
`transdim.compute_phase_tensor` if `observed_PT` is not supplied.

The bottom row always contains:

| Panel | Content |
|-------|---------|
| **C** | Data misfit (χ² = −2 LL) vs sampling step (log–log), one colour per chain; dotted vertical line at burn-in, horizontal line at *n*_data |
| **D** | Number-of-layers histogram (counts) |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | dict | — | Sampler output |
| `frequencies` | (nf,) | — | Observation frequencies [Hz] |
| `observed` | (nf,) | — | Observed apparent resistivity [Ω·m] |
| `sigma` | (nf,) | — | Data uncertainties in log10(ρ_a) space |
| `station` | str | `""` | Station name for title |
| `use_aniso` | bool | False | Use anisotropic forward model for best-fit predictions |
| `observed_Z` | (nf,2,2) complex or None | None | Observed impedance tensor; activates impedance mode |
| `observed_Z_err` | (nf,2,2) float or None | None | Impedance uncertainties (absolute) |
| `z_comps` | tuple of str | `("xy","yx")` | Z components to show (subset of `xx, xy, yx, yy`) |
| `show_pt` | bool | False | Add phase-tensor row |
| `observed_PT` | (nf,2,2) float or None | None | Observed phase tensor (auto-computed from Z if absent) |
| `observed_PT_err` | (nf,2,2) float or None | None | Phase-tensor uncertainties |
| `pt_comps` | tuple of str | `("xx","xy","yx","yy")` | PT components to show |
| `save_path` | str or None | None | Path to save PNG |


### `plot_posterior_model`

Posterior-model summary: 2-D colour histograms overlaid with statistical
profiles, plus change-point frequency.

**Isotropic vs anisotropic layout:**

| `use_aniso` | Panels |
|-------------|--------|
| `False` (default) | E (ρ vs depth histogram) + F (change-point frequency) |
| `True` | E1 (ρ_max) + E2 (ρ_min) + E3 (strike) + F (change-point frequency) |

Each histogram panel shows a colour-coded 2-D density plot (LogNorm,
`RdYlGn_r` colourmap) overlaid with:

| Overlay | Style |
|---------|-------|
| Median | solid black |
| Mean | solid blue |
| Mode | solid green |
| 10th & 90th percentiles | dashed black |
| True model (optional) | solid red staircase |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | dict | — | Sampler output |
| `depth_max` | float | 5000 | Maximum depth [m] |
| `log_rho_range` | (float, float) | (−1, 5) | Bin range for resistivity histograms [log10 Ω·m] |
| `n_rho_bins` | int | 200 | Number of resistivity bins |
| `true_model` | LayeredModel or None | None | Optional true model for overlay |
| `station` | str | `""` | Station name for title |
| `use_aniso` | bool | False | Switch to 3-histogram anisotropic layout |
| `depth_scale` | str | `"linear"` | `"linear"` or `"log"` — controls depth-axis spacing |
| `strike_range` | (float, float) | (−90, 90) | Bin range for strike histogram [deg] |
| `n_strike_bins` | int | 120 | Number of strike bins |
| `save_path` | str or None | None | Path to save PNG |


## Individual panels (reusable)

The single-panel functions take a `matplotlib.axes.Axes` as first argument,
so they can be composed into custom figure layouts.

| Function | Panel content |
|----------|--------------|
| `plot_resistivity_profile(ax, results, depth_grid, ...)` | Posterior ρ(z) with 90 % credible interval, optional true-model overlay |
| `plot_dimension_histogram(ax, results, ...)` | Posterior histogram of number of layers |
| `plot_data_fit(ax, results, freq, obs, ...)` | Observed vs. posterior-predicted ρ_a(f) |
| `plot_chain_traces(ax, results)` | Per-chain log-likelihood traces (colour-coded) |
| `plot_aniso_profile(ax, results, depth_grid, ...)` | Posterior anisotropy ratio vs. depth |
| `plot_strike_profile(ax, results, depth_grid, ...)` | Posterior strike angle vs. depth |

All single-panel functions accept `true_model` (optional) and `depth_max`
where applicable.


## Titles and colour bars

Added 2026-11-11. Defaults give the figures of before (checked pixel by pixel
for the five figure calls of the test).

| Function | New keyword | Meaning |
|----------|-------------|---------|
| `plot_results` | `figure_title=None`, `panel_titles=True` | `None` / `""` = no figure title (as before); text = title above all panels. `panel_titles=False` = no panel titles |
| `plot_qc` | `figure_title=None` | `None` = `MT Station <station> -- 1D rj-McMC Inversion` (`1D rj-McMC Inversion QC` without a station); `""` / `False` = no title; text |
| `plot_posterior_model` | `figure_title=None`, `cbar_location=None`, `cbar_shrink=0.5` | `None` = `Station <station>` above the first panel (none without a station); `""` / `False` = none. Colour bars: `None` = at the right of their panels as before, or `"right"`, `"left"`, `"top"`, `"bottom"`; `cbar_shrink` = their length as a fraction of the panel |
| `plot_resistivity_profile`, `plot_dimension_histogram`, `plot_data_fit`, `plot_chain_traces`, `plot_aniso_profile`, `plot_strike_profile` | `title=None` | `None` = the title of before; `""` / `False` = none; text |

For colour bars at the left, top or bottom the gap to the panel is larger
(`pad` 0.14) so that tick labels, titles and axis labels do not overlap.
`transdim_viz` has its own private `_check_cbar_location` and `_set_title`; it
needs neither `util` nor `femtic_viz`. The scripts `mt_transdim_iso1d.py` and
`mt_transdim_aniso1d.py` (`scripts/general`) pass their settings
`FIGURE_TITLE`, `PANEL_TITLES`, `CBAR_LOCATION`, `CBAR_SHRINK` to these
functions.

Test: `test_transdim_viz_titles.py` (12 tests, with a stub for `transdim`).

**Note (found, not changed):** `transdim_viz.py` and `transdim.py` do not
match in the version supplied (names that `transdim_viz` expects from
`transdim` are not all there), so the module could not be imported against the
real `transdim.py`; the tests use a small stand-in.


## Private helpers

| Function | Description |
|----------|-------------|
| `_overlay_true_model(ax, true_model, depth_max)` | Draw a resistivity staircase for the true model |
| `_overlay_true_staircase(ax, values, true_model, depth_max)` | Draw a generic per-layer staircase for arbitrary values (used by histogram panels for ρ_min, strike, etc.) |


## Dependencies

| Module | Required |
|--------|----------|
| `numpy` | Yes |
| `matplotlib` | Yes |
| `transdim` | Yes (data structures, forward models, posterior computation) |
