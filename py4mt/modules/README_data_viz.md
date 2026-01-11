# data_viz.py

Matplotlib helpers for magnetotelluric (MT) transfer-function plots.

This module provides small, composable plotters for:

- apparent resistivity (`add_rho`)
- impedance phase (`add_phase`)
- tipper components (`add_tipper`)
- phase tensor components (`add_pt`)

All plotters accept an optional Matplotlib `Axes`. If `ax` is `None`, a new
figure/axes is created.

## Inputs: DataFrame *or* EDI dict

The plotters accept either:

1) a `pandas.DataFrame` with *flat* columns (traditional workflow), or  
2) an **EDI dict** (`Mapping[str, Any]`) as produced by your EDI parsers.

When an EDI dict is passed, it is converted internally to a plotting
DataFrame via `edidict_to_plot_df()`.

### Supported EDI dict keys

Required:

- `freq` : `(nf,)` frequency array [Hz]

Optional MT transfer functions and errors:

- `Z` : `(nf, 2, 2)` complex impedance tensor
- `Z_err` : `(nf, 2, 2)` complex or real 1-sigma errors
- `T` : `(nf, 2)` complex tipper `(Tx, Ty)`
- `T_err` : `(nf, 2)` complex or real 1-sigma errors
- `P` : `(nf, 2, 2)` real phase tensor
- `P_err` : `(nf, 2, 2)` real 1-sigma errors

Convenience: `Z`, `Z_err`, `P`, `P_err` are also accepted as `(nf, 4)` and will
be reshaped to `(nf, 2, 2)`.

Metadata commonly found in EDI dicts (e.g., `station`, `lat_deg`, `lon_deg`,
`rot`, …) is copied into `df.attrs`.

## Flat DataFrame column naming

If you pass a DataFrame directly, the plotters expect:

- frequency / period:
  - `freq` [Hz] (required unless `period` exists)
  - `period` [s] (optional; used if present)

- apparent resistivity / phase:
  - `rho_xx`, `rho_xy`, `rho_yx`, `rho_yy` [Ω·m]
  - `phi_xx`, `phi_xy`, `phi_yx`, `phi_yy` [deg]
  - optional 1-sigma envelopes (same unit as the base column):
    - `rho_xy_err`, `phi_xy_err`, …

- tipper:
  - `Tx_re`, `Tx_im`, `Ty_re`, `Ty_im`
  - optional: `Tx_re_err`, `Tx_im_err`, `Ty_re_err`, `Ty_im_err`

- phase tensor:
  - `ptxx_re`, `ptxy_re`, `ptyx_re`, `ptyy_re`
  - optional: `ptxx_re_err`, `ptxy_re_err`, `ptyx_re_err`, `ptyy_re_err`

## What the EDI-dict conversion produces

`edidict_to_plot_df()` creates at least:

- `freq`, `period`

If `Z` is present:

- `rho_<comp>` and `phi_<comp>` for `comp in {xx,xy,yx,yy}`

If `Z_err` is present with a compatible shape:

- `rho_<comp>_err` and `phi_<comp>_err`

Error propagation used (approximate, first-order):

- `rho = |Z|^2 / (μ0 ω)`
- `σ_rho ≈ 2|Z| σ_|Z| / (μ0 ω)`
- `φ = atan2(Im(Z), Re(Z))` (degrees)
- `σ_φ(deg) ≈ (σ_|Z| / |Z|) * 180/π`

If `T` is present:

- `Tx_re`, `Tx_im`, `Ty_re`, `Ty_im`

If `T_err` is present:

- `Tx_re_err`, `Tx_im_err`, `Ty_re_err`, `Ty_im_err`

If `P` is present:

- `ptxx_re`, `ptxy_re`, `ptyx_re`, `ptyy_re`

If `P_err` is present:

- `ptxx_re_err`, `ptxy_re_err`, `ptyx_re_err`, `ptyy_re_err`

## Examples

### Plot from an EDI dict (recommended)

```python
import matplotlib.pyplot as plt
from data_viz_patched import add_rho, add_phase, add_tipper, add_pt

# edi_dict contains keys: freq, Z, T, P, ... (see above)
fig, ax = plt.subplots()
add_rho(edi_dict, ax=ax, show_errors=True)

fig, ax = plt.subplots()
add_phase(edi_dict, ax=ax, show_errors=True)

fig, ax = plt.subplots()
add_tipper(edi_dict, ax=ax, show_errors=True)

fig, ax = plt.subplots()
add_pt(edi_dict, ax=ax, show_errors=True)
plt.show()
```

### Convert explicitly and reuse the DataFrame

```python
from data_viz_patched import edidict_to_plot_df, add_rho

df = edidict_to_plot_df(edi_dict)
ax = add_rho(df, comps="xy,yx", show_errors=True)
```

## Notes

- If a requested component column is missing, the plotters simply skip it.
- `period` is taken from the DataFrame if present; otherwise it is computed as
  `1/freq`.
- The converter assumes SI units for the impedance-derived quantities.
  (μ0 is taken as `4π×10⁻⁷` H/m.)

---

Author: Volker Rath (DIAS)  
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-11
