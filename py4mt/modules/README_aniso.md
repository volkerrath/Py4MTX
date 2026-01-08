# aniso_sens — 1‑D anisotropic MT impedance + sensitivities

This module computes the complex 2×2 MT surface impedance tensor **Z** for a stack of
horizontally layered, electrically anisotropic media, with an optional sensitivity
output suitable for inversion / gradient checks.

The implementation follows a legacy-style recursion for the layered impedance and
adds derivatives by combining:
- analytic recursion derivatives w.r.t. effective horizontal anisotropy parameters (AL, AT, BLT), and
- finite-difference derivatives of the **conversion** from “principal resistivities + Euler angles”
  to (AL, AT, BLT), i.e. a chain rule through `cpanis()`.

## File

In this project snapshot the module lives as:

- `aniso_clean_param_only_hm.py`  (CLI `prog` string is `aniso_sens.py`, but you can run the file as-is)

## Model parameterization (only supported form)

The **only** supported model parameterization is:

2) **Principal resistivities + Euler angles** (converted via `cpanis`)

Per layer (nl layers):

- `h_m` (nl,) — layer thicknesses in **meters**
- `rop` (nl, 3) — principal resistivities **[Ohm·m]**
- `ustr_deg`, `udip_deg`, `usla_deg` (nl,) — Euler angles in **degrees**
  (strike, dip, slant)

### Unit conventions

- Thickness input is `h_m` (meters). No unit conversion is applied internally.
- Periods are in seconds (`periods_s`).
- Resistivities are in Ohm·m; conductivities derived internally are in S/m.

### Basement convention

The recursion does not use the “thickness” of the last layer (basement). You may set
`h_m[-1] = 0.0` (or any value).

## Public API

### `aniso1d_impedance_sens(...)`

```python
from aniso_clean_param_only import aniso1d_impedance_sens

res = aniso1d_impedance_sens(
    periods_s=periods,     # (nper,)
    h_m=h_m,             # (nl,)
    rop=rop,               # (nl,3)
    ustr_deg=ustr_deg,     # (nl,)
    udip_deg=udip_deg,     # (nl,)
    usla_deg=usla_deg,     # (nl,)
    compute_sens=True,
)
Z = res['Z']                 # (nper,2,2)
```

Returned object: `dict`

Always present:
- `Z` — complex impedance tensor, shape `(nper, 2, 2)`

If `compute_sens=True`, additionally:
- `dZ_drop` — ∂Z/∂rop, shape `(nper, nl, 3, 2, 2)`
- `dZ_dustr_deg` — ∂Z/∂ustr_deg, shape `(nper, nl, 2, 2)`
- `dZ_dudip_deg` — ∂Z/∂udip_deg, shape `(nper, nl, 2, 2)`
- `dZ_dusla_deg` — ∂Z/∂usla_deg, shape `(nper, nl, 2, 2)`
- `dZ_dh_m` — ∂Z/∂h_m, shape `(nper, nl, 2, 2)`

For debugging / intermediate use (also returned when sensitivities are computed):
- `dZ_dal`, `dZ_dat`, `dZ_dblt` — derivatives w.r.t. effective parameters (AL, AT, BLT)

### Sensitivity step sizes (advanced)

- Thickness derivative uses a centered finite difference with relative step `dh_rel` (default `1e-6`).
- `rop` derivatives through `cpanis()` use relative step `fd_rel_rop` (default `1e-6`).
- Euler-angle derivatives through `cpanis()` use absolute step `fd_abs_angle_deg` (default `1e-4` degrees).

You can tune these if you do gradient checks against your inversion code.

## CLI usage

### 1) Create a `model.npz`

Example:

```python
import numpy as np

h_m = np.array([1000.0, 2000.0, 0.0])           # 3 layers (last = basement)
rop  = np.array([[100, 200, 300],
                 [ 50,  50, 100],
                 [ 10,  10,  10]], dtype=float)

ustr_deg = np.array([  0,  30, 0.0])
udip_deg = np.array([  0,  10, 0.0])
usla_deg = np.array([  0,   0, 0.0])

np.savez("model.npz",
         h_m=h_m, rop=rop,
         ustr_deg=ustr_deg, udip_deg=udip_deg, usla_deg=usla_deg)
```

### 2) Run forward (and sensitivities)

```bash
python aniso_clean_param_only_hm.py run   --model model.npz   --periods 0.1,1,10,100   --out out.npz   --sens
```

Output `out.npz` contains at least:
- `periods_s`
- `Z`

If `--sens` is used, it also contains:
- `dZ_drop`, `dZ_dustr_deg`, `dZ_dudip_deg`, `dZ_dusla_deg`, `dZ_dh_m`
- (and intermediate) `dZ_dal`, `dZ_dat`, `dZ_dblt`

## Notes

- This is a 1‑D *layered* model. Lateral structure is not represented.
- Angles are handled in degrees at the public API level; internally radians are used.
- If you need **ρ/φ** or other MT derived quantities, compute them from `Z` in your
  post-processing (or extend the module with a small helper).

Author: Volker Rath (DIAS)  
Created with the help of ChatGPT (GPT‑5 Thinking) on 2026‑01‑08 (UTC)
