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
| `add_rhoplus` | D⁺/rho-plus test: ρ_a and ρ⁺ with violation markers | log–log |
| `add_phase` | Impedance phase | semilog-x |
| `add_tipper` | Tipper real/imaginary | semilog-x |
| `add_pt` | Phase tensor components | semilog-x |

All plotters accept an optional Matplotlib `Axes` and return the axes. If
`ax=None`, a new figure/axes pair is created automatically.

---

## D⁺ / rho-plus test plot

### `add_rhoplus(data, comps='xy,yx', ax=None, ...)`

Overlays the measured apparent resistivity ρ_a (solid line) and the D⁺
upper bound ρ⁺ (dashed line, same colour) for each requested component.
Periods where ρ_a > ρ⁺ — i.e. violations of the 1-D consistency test — are
highlighted with red crosses.

The D⁺ (also known as **dplus** in Cordell's mtcode) is the necessary and
sufficient condition for a scalar impedance dataset to be compatible with any
1-D conductivity profile (Parker 1980; Parker & Whaler 1981). A violation at
even one period proves no 1-D model can explain the data.

```python
# Quickest usage — pass a raw EDI dict
ax = data_viz.add_rhoplus(site, comps="xy,yx")

# With a pre-built DataFrame
df = data_proc.dataframe_from_edi(site)
ax = data_viz.add_rhoplus(df, comps="xy,yx", show_errors=True)
```

Key extra arguments:

| Argument | Default | Effect |
|----------|---------|--------|
| `n_lambda_per_freq` | `4` | D⁺ λ-grid density |
| `violation_marker` | `"x"` | Matplotlib marker for violations |
| `violation_color` | `"red"` | Colour for violation markers |
| `violation_size` | `120` | Scatter marker size |

Requires `data_proc.compute_rhoplus` (lazy import).

**References:** Parker (1980) *JGR* 85; Parker & Whaler (1981) *JGR* 86;
Parker & Booker (1996) *PEPI* 98;
Cordell et al. (2022) mtcode doi:10.5281/zenodo.6784201
https://github.com/darcycordell/mtcode

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

fig, axs = plt.subplots(2, 3, figsize=(14, 8))

data_viz.add_rho(df, ax=axs[0, 0], comps="xy,yx", show_errors=True)
data_viz.add_rhoplus(site, ax=axs[0, 1], comps="xy,yx")   # D+ test
data_viz.add_phase(df, ax=axs[0, 2], comps="xy,yx", show_errors=True)
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
Modified: 2026-03-16 — add_rhoplus (D+/rho+ test plot); Claude Sonnet 4.6 (Anthropic)
