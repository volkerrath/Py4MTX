# mt_get_averages.py

Zssq / Zdet / Zavg — site overlay, log-average plots, and NPZ export.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_get_averages.py` |
| Author | Volker Rath (DIAS), 2026-06-12 |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2026-06-12 by Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-12 — initial version; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-12 — annotation to axes-fraction lower-left; `LEGEND_SITES` switch; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-12 — legend fontsize/ncol scaled by `LEGEND_SITES`; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-12 — `PLOT_SSQ` / `PLOT_DET` switches; unified `_rho_phi_from_zinvar` and `_plot_invar`; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-13 — `PLOT_AVG`: Zavg = sqrt(Zxy·Zyx) off-diagonal geometric mean; `_plot_invar` returns averages; NPZ save per quantity; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-13 — removed `PLOT_XY` / `PLOT_YX`; added `FREQ_RANGE` frequency mask; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-13 — `PLOT_AVG` switched to direct `"Zavg"` key lookup; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-13 — `FIG_SIZE_CM`; `PLOT_ERRORS` / `ERR_ALPHA` shaded std bands on all three averages; `rho_std` / `phi_std` added to NPZ; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-13 — Zavg phase folded to [0°, 90°] via `fold_phase` flag; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-13 — `ERRORS` switch (`"std"`/`"bootstrap"`/`None`); bootstrap std-of-the-mean per bin and globally; global interval in print/annotation; `rho_global_std` in NPZ; Claude Sonnet 4.6 (Anthropic) |

## Purpose

Reads a collection NPZ produced by `mt_data_processor.py`, extracts any
combination of scalar impedance quantities (`Zssq`, `Zdet`, `Zavg`) for every
station, optionally restricts the frequency range, computes apparent
resistivity and phase, forms frequency-wise and global log-averages with
inter-site spread estimates, writes one two-panel rho/phase plot per active
quantity, and saves the averaged curves to a per-quantity NPZ.

## Physics

### Invariants

```
Zssq = sqrt( (Zxx² + Zxy² + Zyx² + Zyy²) / 2 )     sum-of-squares
Zdet = sqrt( Zxx·Zyy − Zxy·Zyx )                     determinant
Zavg = sqrt( Zxy · Zyx )                              off-diagonal geometric mean
```

`Zavg` recovers the Berdichevsky average: for a 1-D earth `rho_a(Zavg)`
equals the arithmetic mean of `rho_a(Zxy)` and `rho_a(Zyx)`.

All quantities are in MT field units (mV/km/nT). Apparent resistivity and
phase are derived identically:

```
rho_a(f) = mu0 * |Z|² * 1e6 / (2π f)   [Ω·m]
phi(f)   = arg(Z)                        [°]
```

using the same `_Z_MT_TO_SI_SQ` factor as `data_proc`.

### Phase convention for Zavg

Because `Zavg = sqrt(Zxy·Zyx)`, the complex square root introduces a branch
cut that can place `arg(Zavg)` anywhere in (−180°, 180°]. For a 1-D earth
`Zyx = −Zxy`, so `Zxy·Zyx` is negative real and `arg(Zavg) = ±90°`; in
general the sign depends on the choice of square-root branch. The physically
meaningful quantity is the magnitude of the phase, which lies in [0°, 90°]
for passive linear media. The script therefore folds `phi` into [0°, 90°]
using

```
phi_folded = 90° − |( phi mod 180° ) − 90°|
```

which maps the full circle correctly: −90° → 90°, −45° → 45°, 135° → 45°, etc.

## Error statistics

### Estimators

Two estimators are available, selected by `ERRORS`:

**`"std"` — sample standard deviation (spatial spread)**

Characterises the *heterogeneity* across the survey area, not the uncertainty
of the mean. Computed in a single analytical pass:

```
rho_std_g = exp( std_sample( ln rho_a ) )       geometric std factor
phi_std   = std_sample( phi )                    [°]
```

The ±1 σ band for rho is `[rho_avg / rho_std_g,  rho_avg · rho_std_g]` on
the log scale. This is the standard log-normal spread estimator (Limpert et al. 2001).
It estimates how *spread out* the individual sites are, not how well the mean
is determined.

**`"bootstrap"` — bootstrap standard deviation of the mean (default)**

Estimates the *uncertainty of the mean estimator* itself. At each frequency
bin, the N contributing site values are resampled with replacement `ERR_NSIM`
times; the std of the resulting distribution of bootstrap means is the reported
uncertainty:

```
rho_std_g = exp( std_bootstrap( mean( ln rho_a_b ) ) )    b = 1…N_sim
phi_std   = std_bootstrap( mean( phi_b ) )                 [°]
```

This is ≈ σ/√N for large N (the standard error of the mean), but is
distribution-free and correct for the log-normal geometry of resistivity.
For N = 41 sites the bootstrap std is about √41 ≈ 6× smaller than the sample
std — the shaded band represents how precisely the mean is located, not how
scattered the sites are.

The same bootstrap is applied globally to compute `rho_global_std`, printed
to screen and shown in the plot annotation.

### Comparison

| Estimator | Measures | Scales with N | Use when |
|-----------|----------|--------------|----------|
| `"std"` | Lateral heterogeneity | Does not shrink | Characterising structural variation |
| `"bootstrap"` | Uncertainty of mean | ∝ 1/√N | Reporting average as a data product |

### References

- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*.
  Chapman & Hall/CRC. — authoritative reference for bootstrap resampling and
  the bootstrap standard error.
- Efron, B. (1979). Bootstrap methods: another look at the jackknife.
  *Annals of Statistics*, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552 —
  original bootstrap paper; also introduces the jackknife connection.
- Limpert, E., Stahel, W. A., & Abbt, M. (2001). Log-normal distributions
  across the sciences: keys and clues. *BioScience*, 51(5), 341–352.
  https://doi.org/10.1641/0006-3568(2001)051[0341:LNDATS]2.0.CO;2 — reference
  for the geometric standard deviation as a spread measure on log-normal data.
- Berdichevsky, M. N., & Dmitriev, V. I. (2008). *Models and Methods of
  Magnetotellurics*. Springer. ISBN 978-3-540-77811-0. — foundational
  treatment of regional averages and the determinant/ssq invariants.
- Szarka, L., & Menvielle, M. (1997). Analysis of rotational invariants of
  the magnetotelluric impedance tensor. *Geophysical Journal International*,
  129(1), 133–142. https://doi.org/10.1111/j.1365-246X.1997.tb00942.x —
  defines the ssq invariant used here.
- Rung-Arunwan, T., Siripunvaraporn, W., & Utada, H. (2016). On the Berdichevsky
  average. *Physics of the Earth and Planetary Interiors*, 253, 1–4.
  https://doi.org/10.1016/j.pepi.2016.01.006 — establishes the relationship
  between Zavg and the classical Berdichevsky average.

## Workflow

1. **Load** the collection NPZ (`COLL_FILE`); extract `freq` and the active
   quantity per station. `Zssq`, `Zdet`, `Zavg` are read from direct keys
   (require `INVARS = True` in `mt_data_processor.py`).
   Frequencies outside `FREQ_RANGE` are masked out before any processing.
   For `Zavg`, phase is folded into [0°, 90°] after computation.
2. **Compute** per-site `rho_a` and `phi` via `_rho_phi_from_zinvar`.
3. **Build** a merged frequency grid (union of all site grids, deduplicated at 1 %).
4. **Average** — frequency-wise geometric mean of `rho_a` and arithmetic mean
   of `phi` via nearest-bin matching (`BIN_TOL_DEC`); inter-site std computed
   in a second pass.
5. **Global average** — scalar geometric mean over all sites and all frequencies.
6. **Plot** — one 2-panel figure per active quantity with optional std shading;
   save to `PLOT_DIR`.
7. **Export** — per-quantity NPZ with averaged curves and std arrays; save to
   `DATA_DIR`.

## Configuration constants

| Constant | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | *(set per project)* | Root working directory |
| `COLL_FILE` | `WORK_DIR + "TEST_test_proc_collection.npz"` | Input collection NPZ |
| `PLOT_DIR` | `WORK_DIR + "../plots/"` | Output plot directory (created if missing) |
| `DATA_DIR` | `WORK_DIR` | Output NPZ directory (created if missing) |
| `NAME_STR` | `"Ubinas_ssq2"` | Base name for all output files |
| `PLOT_FORMAT` | `[".pdf", ".png"]` | List of Matplotlib-compatible format strings |
| `PLOT_SSQ` | `True` | Process Zssq invariant |
| `PLOT_DET` | `True` | Process Zdet invariant |
| `PLOT_AVG` | `True` | Process Zavg = sqrt(Zxy·Zyx) off-diagonal geometric mean |
| `FREQ_RANGE` | `(None, None)` | Frequency range [Hz] applied before averaging; `None` = no limit on that bound |
| `BIN_TOL_DEC` | `0.1` | Nearest-bin match tolerance in decades |

Output filenames: `{NAME_STR}_{tag}{fmt}` for plots and
`{NAME_STR}_{tag}_avg.npz` for averaged data,
where `tag` ∈ {`ssq`, `det`, `avg`}.

### Plot appearance

| Constant | Default | Description |
|----------|---------|-------------|
| `XLIM` | `(1e-4, 1e+4)` | Period axis limits [s]; `None` = auto |
| `YLIM_RHO` | `(1e+0, 1e+4)` | Rho axis limits [Ω·m]; `None` = auto |
| `YLIM_PHASE` | `(0., 90.)` | Phase axis limits [°]; `None` = auto |
| `FIG_SIZE_CM` | `(22., 18.)` | Figure size (width, height) in cm |
| `PLOT_ERRORS` | `True` | Enable error band shading |
| `ERRORS` | `"bootstrap"` | Error estimator: `"std"` (sample std, spatial spread), `"bootstrap"` (bootstrap std-of-the-mean), or `None` |
| `ERR_NSIM` | `500` | Number of bootstrap resamples (used only when `ERRORS="bootstrap"`) |
| `ERR_ALPHA` | `0.20` | Opacity of the std shading |
| `LEGEND_SITES` | `True` | If `True`, all station names in legend; if `False`, averages only (larger font) |
| `SITE_ALPHA` | `0.35` | Opacity of individual site curves |
| `SITE_LW` | `1.8` | Line width of individual site curves |
| `AVG_LW` | `3.0` | Line width of the frequency-wise average curve |
| `AVG_COLOR` | `"red"` | Colour of the frequency-wise average curve |
| `SITE_CMAP` | `"tab20"` | Matplotlib colormap used to colour site curves |

## Internal helpers

| Function | Description |
|----------|-------------|
| `_rho_phi_from_zinvar(freq, Zinvar)` | Compute `rho_a` [Ω·m] and `phi` [°] from any scalar complex quantity |
| `_period(freq)` | Return `1/freq` |
| `_log_avg(values)` | Scalar geometric mean over all finite positive values in a list of arrays |
| `_build_freq_grid(all_freqs)` | Union of all site frequency vectors, sorted ascending, deduplicated at 1 % ratio tolerance |
| `_freqwise_log_avg(grid, all_freqs, all_rho, all_phi, tol_dec, errors, n_sim)` | Per-grid-node geometric mean of `rho_a` and arithmetic mean of `phi`; per-bin std via `"std"` (sample std, two analytical passes) or `"bootstrap"` (std-of-the-mean from `n_sim` resamples); `NaN` where count < 2 |
| `_plot_invar(tag, label, stations, all_freqs, all_rho, all_phi)` | Produce and save the 2-panel figure; return averaged-curve dict for NPZ export |

## NPZ output format

Each `{NAME_STR}_{tag}_avg.npz` contains:

| Key | Shape | Units | Description |
|-----|-------|-------|-------------|
| `freq_avg` | `(n_grid,)` | Hz | Merged frequency grid |
| `period_avg` | `(n_grid,)` | s | `1/freq_avg` |
| `rho_avg` | `(n_grid,)` | Ω·m | Frequency-wise geometric mean rho_a (`NaN` where no site contributes) |
| `phi_avg` | `(n_grid,)` | ° | Frequency-wise arithmetic mean phase |
| `rho_std` | `(n_grid,)` | — | Geometric std factor (multiplicative); band = `rho_avg / rho_std` … `rho_avg * rho_std`; `NaN` where count < 2 |
| `phi_std` | `(n_grid,)` | ° | Std of phase across contributing sites; `NaN` where count < 2 |
| `rho_global` | scalar | Ω·m | Global geometric mean rho_a over all sites and frequencies |
| `rho_global_std` | scalar | Ω·m | Bootstrap (or sample) std of the global mean; `NaN` if errors disabled |
| `n_sites` | scalar | — | Number of stations that contributed |

## Plot panels

| Panel | x-axis | y-axis | Contents |
|-------|--------|--------|----------|
| Top | Period [s] (log) | ρ_a [Ω·m] (log) | Per-site curves (coloured, semi-transparent); frequency-wise average (dashed, `AVG_COLOR`); global scalar average as horizontal dotted line with annotation at lower-left |
| Bottom | Period [s] (log) | Phase [°] (linear) | Per-site curves; frequency-wise average; reference lines at 0° and 45° |

## Legend behaviour

| `LEGEND_SITES` | Font size | Columns | Entries |
|----------------|-----------|---------|---------|
| `False` | 10 | 1 | freq-wise avg only |
| `True` | 5 | 4 | all station names + freq-wise avg |

## Dependencies

`numpy`, `matplotlib`; py4mt: `data_proc` (collection NPZ format).
`Zssq` and `Zdet` require `INVARS = True` in `mt_data_processor.py`.
`Zavg` requires only that `Z` is present (always the case).
