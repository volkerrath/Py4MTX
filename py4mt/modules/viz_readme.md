# viz.py — Matplotlib visualization for MT data and models

`viz.py` provides Matplotlib-based plotting functions for magnetotelluric
data, inversion results, and model visualizations. All plotters accept
`**pltargs` keyword arguments for flexible style control.

> **Note:** For simple transfer-function plots from DataFrames, prefer
> `data_viz.py`. This module is aimed at richer, publication-ready figures
> and model-space visualizations.

---

## MT data plotters

### `plot_impedance(thisaxis=None, data=None, **pltargs)`

Plot impedance tensor components (real/imaginary parts of Zxx, Zxy, Zyx, Zyy)
as a function of period, with optional observed/calculated comparison and
error bars.

### `plot_rhophas(thisaxis=None, data=None, **pltargs)`

Plot apparent resistivity (log-log) and phase (semilog-x) curves with
optional model comparison and error envelopes.

Optional keys of `**pltargs` for phase curves that leave the +-180 deg window
(2026-11-20; off by default, nothing changes without them):

| Key | Description |
|-----|-------------|
| `phase_wrap_mode` | `"off"` (default), `"direct"` (a curve that jumps by more than half the window between two periods is interrupted and the two points are joined by a straight line) or `"short"` (joined the short way, through the window edge) |
| `phase_wrap_style` | dict of Matplotlib line properties of the joining line over the dotted default (`data_viz.PHASE_WRAP_STYLE`); `None` = dotted, `False` = gap only |
| `phase_wrap_range` | the window, default `(-180, 180)` |
| `phase_wrap_curves` | which of the two plotted quantities are phases: `(1,)` (default; data column 2 / 6, the phase of a rho/phase pair), `(0, 1)` for a panel with two phases, `(0,)` |

The handling is `data_viz.mark_phase_wraps` applied to the lines of the observed
(errorbar) and calculated curves of the selected quantities; the other plotters
of the module ignore the keys, so one `PLTARGS` dict can be shared.

### `plot_phastens(thisaxis=None, data=None, **pltargs)`

Plot phase tensor components (Φxx, Φxy, Φyx, Φyy) as a function of period.

### `plot_vtf(thisaxis=None, data=None, **pltargs)`

Plot vertical transfer functions (tipper) — both real and imaginary parts of
Tx and Ty.

---

## Model visualization

### `plot_sparsity_pattern(matrix, ...)`

Visualize the sparsity pattern of a matrix (e.g. Jacobian, regularization).

### `plot_plane_cross(ax, position, ...)`

Draw vertical cross-section planes for 3-D model visualization.

---

## PDF catalog

### `make_pdf_catalog(workdir, pdflist, filename)`

Merge multiple PDF files into a single catalog document using PyMuPDF (fitz).

---

## `pltargs` convention

All plotters read their configuration from `**pltargs`. Common keys:

| Key | Type | Description |
|-----|------|-------------|
| `plotformat` | list[str] | Output formats, e.g. `['.png', '.pdf']` |
| `plotfile` | str | Base filename for saving |
| `suptitle` | str | Figure super-title |
| `figsize` | list[float] | Figure size `[width, height]` |
| `fontsizes` | list[int] | `[tick, label, title, legend]` |
| `c_obs`, `c_cal` | list[str] | Colors for observed/calculated |
| `m_obs`, `m_cal` | list[str] | Markers for observed/calculated |
| `l_obs`, `l_cal` | list | Line styles |
| `xlimits`, `ylimits` | list | Axis limits |

---

## Changelog

### Changelog (2026-11-20) — phase wraps in `plot_rhophas` (Claude Sonnet 5.5 (Anthropic), AI-generated, review before use)
- New opt-in keys `phase_wrap_mode`, `phase_wrap_style`, `phase_wrap_range`, `phase_wrap_curves` of `**pltargs` (see `plot_rhophas`); default `"off"`, so existing figures are unchanged. Needs `data_viz.py` of the same date (`mark_phase_wraps`).
- Tests: `scripts/general/test_phase_wrap_plotters.py`.

### Changelog (2026-10-11) — FEMTIC I/O alignment
- `plot_sparsity_pattern`: the sparse-matrix check is now `ensembles.check_sparse_matrix` (the function lives in `ensembles.py`; the former `femtic.check_sparse_matrix` call raised `AttributeError`).

### Changelog (2026-07-17) — scipy.sparse: matrix → array API
- Migrated from legacy `scipy.sparse` matrix classes to the array-equivalent
  API in `plot_sparsity_pattern`: `coo_matrix` → `coo_array`; removed a
  redundant, now-unused duplicate import line that still referenced the
  legacy `csr_matrix`/`csc_matrix`/`coo_matrix` names.

Author: Volker Rath (DIAS)
scipy.sparse matrix → array migration by Claude Sonnet 5 (Anthropic) on 2026-07-17.
