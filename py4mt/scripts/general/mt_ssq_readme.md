# mt_ssq.py

Zssq / Zdet impedance invariants — site overlay and log-average plots.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_ssq.py` |
| Author | Volker Rath (DIAS), 2026-06-12 |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2026-06-12 by Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-12 — initial version; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-12 — annotation to axes-fraction lower-left; `LEGEND_SITES` switch; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-12 — legend fontsize/ncol scaled by `LEGEND_SITES`; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-12 — `PLOT_SSQ` / `PLOT_DET` switches; unified `_rho_phi_from_zinvar` and `_plot_invar`; Claude Sonnet 4.6 (Anthropic) |

## Purpose

Reads a collection NPZ produced by `mt_data_processor.py`, extracts one or
both scalar impedance invariants (`Zssq`, `Zdet`) for every station, computes
apparent resistivity and phase, forms frequency-wise and global log-averages,
and writes one two-panel rho/phase plot per active invariant.

## Physics

### Sum-of-squares invariant

```
Zssq = sqrt( (Zxx² + Zxy² + Zyx² + Zyy²) / 2 )
```

### Determinant invariant

```
Zdet = sqrt( Zxx·Zyy − Zxy·Zyx )
```

Both are stored in the collection NPZ in MT field units (mV/km/nT). Apparent
resistivity and phase are derived identically for both:

```
rho_a(f) = mu0 * |Z_invar|² * 1e6 / (2π f)   [Ω·m]
phi(f)   = arg(Z_invar)                        [°]
```

using the same `_Z_MT_TO_SI_SQ` factor as `data_proc`.

## Workflow

1. **Load** the collection NPZ (`COLL_FILE`); extract `freq` and the active
   invariant key(s) per station.
2. **Compute** per-site `rho_a` and `phi` via `_rho_phi_from_zinvar`.
3. **Build** a merged frequency grid (union of all site grids, deduplicated at 1 %).
4. **Average** — frequency-wise geometric mean of `rho_a` and arithmetic mean
   of `phi` via nearest-bin matching (`BIN_TOL_DEC`).
5. **Global average** — scalar geometric mean over all sites and all frequencies.
6. **Plot** — one 2-panel figure per active invariant; save to `PLOT_DIR`.

## Configuration constants

| Constant | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | *(set per project)* | Root working directory |
| `COLL_FILE` | `WORK_DIR + "TEST_test_proc_collection.npz"` | Input collection NPZ |
| `PLOT_DIR` | `WORK_DIR + "../plots/"` | Output plot directory (created if missing) |
| `NAME_STR` | `"Ubinas_ssq2"` | Base name for output plot files |
| `PLOT_FORMAT` | `[".pdf", ".png"]` | List of Matplotlib-compatible format strings |
| `PLOT_SSQ` | `True` | Process and plot the Zssq invariant |
| `PLOT_DET` | `True` | Process and plot the Zdet invariant |
| `BIN_TOL_DEC` | `0.1` | Nearest-bin match tolerance in decades |

Output filenames are `{NAME_STR}_ssq{fmt}` and `{NAME_STR}_det{fmt}`.

### Plot appearance

| Constant | Default | Description |
|----------|---------|-------------|
| `XLIM` | `(1e-4, 1e+4)` | Period axis limits [s]; `None` = auto |
| `YLIM_RHO` | `(1e+0, 1e+4)` | Rho axis limits [Ω·m]; `None` = auto |
| `YLIM_PHASE` | `(0., 90.)` | Phase axis limits [°]; `None` = auto |
| `LEGEND_SITES` | `True` | If `True`, all station names in legend; if `False`, averages only (larger font) |
| `SITE_ALPHA` | `0.35` | Opacity of individual site curves |
| `SITE_LW` | `1.8` | Line width of individual site curves |
| `AVG_LW` | `3.0` | Line width of the frequency-wise average curve |
| `AVG_COLOR` | `"red"` | Colour of the frequency-wise average curve |
| `SITE_CMAP` | `"tab20"` | Matplotlib colormap used to colour site curves |

## Internal helpers

| Function | Description |
|----------|-------------|
| `_rho_phi_from_zinvar(freq, Zinvar)` | Compute `rho_a` [Ω·m] and `phi` [°] from any scalar complex invariant |
| `_period(freq)` | Return `1/freq` |
| `_log_avg(values)` | Scalar geometric mean over all finite positive values in a list of arrays |
| `_build_freq_grid(all_freqs)` | Union of all site frequency vectors, sorted ascending, deduplicated at 1 % ratio tolerance |
| `_freqwise_log_avg(grid, all_freqs, all_rho, all_phi, tol_dec)` | Per-grid-node geometric mean of `rho_a` and arithmetic mean of `phi`; nodes with no contributing site are `NaN` |
| `_plot_invar(tag, label, stations, all_freqs, all_rho, all_phi)` | Produce and save the 2-panel rho/phase figure for one invariant |

## Output

One figure per active invariant (`_ssq` / `_det` suffix), each with:

| Panel | x-axis | y-axis | Contents |
|-------|--------|--------|----------|
| Top | Period [s] (log) | ρ_a [Ω·m] (log) | Per-site curves (coloured, semi-transparent); frequency-wise average (dashed, `AVG_COLOR`); global scalar average as horizontal dotted line with annotation at lower-left |
| Bottom | Period [s] (log) | Phase [°] (linear) | Per-site curves; frequency-wise average; reference lines at 0° and 45° |

The global log-average ρ̄_a for each invariant is also printed to stdout.

## Legend behaviour

| `LEGEND_SITES` | Font size | Columns | Entries |
|----------------|-----------|---------|---------|
| `False` | 10 | 1 | freq-wise avg only |
| `True` | 5 | 4 | all station names + freq-wise avg |

## Dependencies

`numpy`, `matplotlib`; py4mt: `data_proc` (collection NPZ format).
`Zssq` and `Zdet` must have been computed by `mt_data_processor.py`
with `INVARS = True` before saving the collection.
