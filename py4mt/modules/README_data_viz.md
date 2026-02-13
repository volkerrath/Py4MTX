# data_viz.py — Matplotlib helpers for MT transfer-function plots

`data_viz.py` contains small, composable Matplotlib plotters designed to work
with a **tidy `pandas.DataFrame`** produced from EDI dictionaries (typically via
`data_proc.dataframe_from_edi`).

Main MT plotters:

- `add_rho` — apparent resistivity curves (log–log)
- `add_phase` — impedance phase (semilog-x)
- `add_tipper` — tipper real/imag (semilog-x)
- `add_pt` — phase tensor components (semilog-x)

All plotters accept an optional Matplotlib `Axes` and return the axes.

> Note: This module **does not** take an EDI dict directly. Convert to a DataFrame first.

---

## Expected DataFrame columns

Required:

- `freq` [Hz] (frequency array)

Optional but strongly recommended:

- `period` [s] (if missing, `period = 1/freq` is computed internally)

### Apparent resistivity / phase

- `rho_xx`, `rho_xy`, `rho_yx`, `rho_yy` [Ω·m]
- `phi_xx`, `phi_xy`, `phi_yx`, `phi_yy` [deg]

Optional 1‑sigma envelopes:

- `rho_xy_err`, `phi_xy_err`, … (suffix is configurable via `error_suffix`)

### Tipper

- `Tx_re`, `Tx_im`, `Ty_re`, `Ty_im`
- optional: `Tx_re_err`, `Tx_im_err`, `Ty_re_err`, `Ty_im_err`

### Phase tensor

Phase tensor components are real, but column names keep the historical “`_re`”:

- `ptxx_re`, `ptxy_re`, `ptyx_re`, `ptyy_re`
- optional: `ptxx_re_err`, `ptxy_re_err`, `ptyx_re_err`, `ptyy_re_err`

---

## Recommended workflow (EDI dict → DataFrame → plot)

```python
import matplotlib.pyplot as plt
import data_proc
import data_viz

site = data_proc.load_edi("SITE.edi")

# Optional: compute PT from Z (or do this elsewhere)
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

## Convenience: quick subplot grids (`plot_gridx`, `plot_grid`)

The module also contains two generic helpers that render lists of small
datasets into an `(nrows, ncols)` subplot grid and remove empty axes.

- `plot_gridx` supports `type ∈ {"line","scatter","image","hist"}`
- `plot_grid` supports simple `{"x","y","style"}` line datasets

These are optional utilities and unrelated to MT-specific plotting.

---

Author: Volker Rath (DIAS)  
Updated with the help of ChatGPT (GPT-5.2 Thinking) on 2026-02-13 (UTC)
