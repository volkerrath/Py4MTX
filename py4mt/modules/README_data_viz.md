# data_viz.py — Matplotlib helpers for MT transfer-function plots

`data_viz.py` contains small, composable Matplotlib plotters for
magnetotelluric transfer-function data. All plotters work with a **tidy
`pandas.DataFrame`** produced by `data_proc.dataframe_from_edi()`, and can
also accept an EDI-style data dict directly (auto-converted internally via
`datadict_to_plot_df()`).

## MT plotters

| Function | Plot type | Axes scale |
|----------|-----------|------------|
| `add_rho` | Apparent resistivity curves | log–log |
| `add_phase` | Impedance phase | semilog-x |
| `add_tipper` | Tipper real/imaginary | semilog-x |
| `add_pt` | Phase tensor components | semilog-x |

All plotters accept an optional Matplotlib `Axes` and return the axes. If
`ax=None`, a new figure/axes pair is created automatically.

---

## Expected DataFrame columns

**Required:** `freq` [Hz]

**Optional but recommended:** `period` [s] (computed from `freq` if missing)

### Apparent resistivity / phase

- `rho_xx`, `rho_xy`, `rho_yx`, `rho_yy` [Ω·m]
- `phi_xx`, `phi_xy`, `phi_yx`, `phi_yy` [deg]
- Optional error columns: `rho_xy_err`, `phi_xy_err`, … (suffix configurable
  via `error_suffix`)

### Tipper

- `Tx_re`, `Tx_im`, `Ty_re`, `Ty_im`
- Optional: `Tx_re_err`, `Tx_im_err`, `Ty_re_err`, `Ty_im_err`

### Phase tensor

Phase tensor components are real; column names keep the historical `_re` suffix:

- `ptxx_re`, `ptxy_re`, `ptyx_re`, `ptyy_re`
- Optional: `ptxx_re_err`, `ptxy_re_err`, `ptyx_re_err`, `ptyy_re_err`

---

## Common plotter arguments

All MT plotters share these keyword arguments:

- `show_errors` (bool) — show ±1σ fill-between envelopes (default `False`)
- `error_suffix` (str) — suffix for error columns (default `"_err"`)
- `error_alpha` (float) — transparency for error envelopes (default `0.25`)
- `comps` (str) — comma-separated component list, e.g. `"xy,yx"` (default
  varies by plotter)
- `legend` (bool) — show legend (default `True`)

---

## Data dict support

In addition to DataFrames, the main plotters accept a data dict (the dict
returned by `data_proc.load_edi()`) with keys `freq`, `Z`, `Z_err`, `T`,
`T_err`, `P`, `P_err`. Conversion to the plotting DataFrame happens
automatically via `datadict_to_plot_df()`.

---

## Recommended workflow

```python
import matplotlib.pyplot as plt
import data_proc
import data_viz

site = data_proc.load_edi("SITE.edi")
site["P"], site["P_err"] = data_proc.compute_pt(site["Z"], site.get("Z_err"))
df = data_proc.dataframe_from_edi(site)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

data_viz.add_rho(df, ax=axs[0, 0], comps="xy,yx", show_errors=True)
data_viz.add_phase(df, ax=axs[0, 1], comps="xy,yx", show_errors=True)
data_viz.add_tipper(df, ax=axs[1, 0], show_errors=True)
data_viz.add_pt(df, ax=axs[1, 1], show_errors=True)

fig.tight_layout()
plt.show()
```

---

## Generic subplot helpers

Two utility functions for rendering lists of small datasets into subplot grids:

- `plot_gridx(data_list, nrows, ncols, figsize)` — supports
  `type ∈ {"line", "scatter", "image", "hist"}`
- `plot_grid(data_list, nrows, ncols, figsize)` — simple `{"x", "y", "style"}`
  line datasets

Both remove empty axes automatically.

---

Author: Volker Rath (DIAS)
