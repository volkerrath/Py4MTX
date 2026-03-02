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

Author: Volker Rath (DIAS)
