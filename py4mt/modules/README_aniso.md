# aniso.py — anisotropic 1‑D MT forward model (+ sensitivities)

`aniso.py` implements a 1‑D layered MT forward model with horizontal electrical
anisotropy (strike angle per layer). It provides the impedance tensor `Z(ω)`
and (optionally) sensitivities.

This project currently uses two “front ends”:

1. **Public / legacy parameterization** (implemented here):
   - `rop` (principal resistivities) + Euler angles  
   - function: `aniso1d_impedance_sens(...)`

2. **Simplified inversion parameterization** (expected by `mcmc.py` and `inv1d.py`):
   - `(rho_min, rho_max, strike_deg)` per layer  
   - function: `aniso1d_impedance_sens_simple(...)` (thin wrapper)

> If your local `aniso.py` snapshot does not yet contain
> `aniso1d_impedance_sens_simple`, `mcmc.py` / `inv1d.py` will fail at import
> time. Add the wrapper (or adjust the import) so that all modules agree.

---

## Common conventions

- `h_m` is `(nl,)` thickness array in **meters**.
- The **last entry** in `h_m` is a “basement placeholder” and is ignored by the
  recursion. Keep it at `0.0` so cumulative-depth plots remain well-defined.
- `periods_s` is `(nper,)` in seconds.

---

## 1) Public API: `aniso1d_impedance_sens` (rop + Euler angles)

```python
from aniso import aniso1d_impedance_sens

res = aniso1d_impedance_sens(
    periods_s=periods_s,          # (nper,)
    h_m=h_m,                      # (nl,)
    rop=rop,                      # (nl,3) principal resistivities [Ω·m]
    ustr_deg=ustr_deg,            # (nl,) strike (deg)
    udip_deg=udip_deg,            # (nl,) dip (deg)
    usla_deg=usla_deg,            # (nl,) slant (deg)
    compute_sens=True,
)
Z = res["Z"]                      # (nper,2,2), complex
```

If `compute_sens=True`, the returned dict also contains (see `aniso.py` docstrings):

- `dZ_drop` : `(nper, nl, 3, 2, 2)` derivatives w.r.t. the 3 principal resistivities
- `dZ_dustr_deg`, `dZ_dudip_deg`, `dZ_dusla_deg` : `(nper, nl, 2, 2)` angle derivatives
- `dZ_dh_m` : `(nper, nl, 2, 2)` thickness derivatives

---

## 2) Simplified wrapper: `aniso1d_impedance_sens_simple` (rho_min/max + strike)

The Bayesian sampler (`mcmc.py`) and the deterministic inverter (`inv1d.py`)
operate in a reduced parameter space:

- `rho_min` : `(nl,)` minimum horizontal resistivity
- `rho_max` : `(nl,)` maximum horizontal resistivity
- `strike_deg` : `(nl,)` strike angle

Expected call pattern:

```python
import aniso

fwd = aniso.aniso1d_impedance_sens_simple(
    periods_s,
    h_m,
    rho_max,
    rho_min,
    strike_deg,
    compute_sens=True,
)
Z = fwd["Z"]   # (nper,2,2)
```

Expected sensitivity keys (used by `inv1d.py`):

- `dZ_drho_min` : `(nper, nl, 2, 2)`
- `dZ_drho_max` : `(nper, nl, 2, 2)`
- `dZ_dstrike_deg` : `(nper, nl, 2, 2)`

(Thickness sensitivities are currently not required by `inv1d.py`, but are
useful when sampling/inverting `h_m`.)

---

## Units and stability notes

- Resistivity is in Ω·m; conductivity is S/m.
- Any phase-tensor computations in this project use  
  `P = inv(Re(Z)) @ Im(Z)` (see `mcmc.ensure_phase_tensor` or `data_proc.compute_pt`).

---

Author: Volker Rath (DIAS)  
Updated with the help of ChatGPT (GPT-5.2 Thinking) on 2026-02-13 (UTC)
