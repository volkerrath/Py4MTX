# mcmc.py — Bayesian inversion (PyMC) for simplified anisotropic 1‑D MT

This mini-package contains a **script-style** PyMC workflow for layered 1‑D MT
impedance inversion with optional phase-tensor likelihood:

- **`mcmc.py`** — PyMC model builder + sampling + compact summaries
- **`mcmc_viz.py`** — axes-based plotting helpers for the summaries / posteriors
- **`mt_aniso1d_sampler.py`** — non-CLI driver script (edit USER CONFIG and run)
- **`aniso.py`** — forward model (see `README_aniso.md`)
- **`data_proc.py`** — EDI/NPZ I/O and MT derived quantities (see `README_data_proc.md`)

The current inversion code uses a **simplified per-layer parameterization**:

- `h_m` (m) — thicknesses (last entry is a basement placeholder)
- `rho_min` (Ω·m) — minimum horizontal resistivity
- `rho_max` (Ω·m) — maximum horizontal resistivity (enforced `rho_max >= rho_min`)
- `strike_deg` (deg) — anisotropy strike angle

Two boolean masks are supported:

- `is_iso` — isotropic layer (forces `rho_min == rho_max`, strike ignored)
- `is_fix` — fixed layer (parameters not sampled)

---

## Dependencies

Sampling:

- `numpy`
- `pymc` (v5+)
- `pytensor` (backend)
- `arviz` (InferenceData I/O, diagnostics)

Plotting:

- `matplotlib` (via `mcmc_viz.py`)

---

## Input site format (flat dict)

All functions in `mcmc.py` operate on a **flat** Python dict (this is the
`data_dict` used throughout the project):

Required keys:

- `freq` — `(n,)` frequency array in **Hz**
- `Z` — `(n,2,2)` complex impedance tensor

Optional uncertainty keys:

- `Z_err` — `(n,2,2)` float (std or var; interpreted by `err_kind` if present)
- `err_kind` — `"std"` or `"var"` (optional; if missing, code assumes std-like use)

Optional phase tensor keys (only needed if you already have them):

- `P` — `(n,2,2)` float
- `P_err` — `(n,2,2)` float

### Loading sites (`.edi` or `.npz`)

Use:

```python
import mcmc

site = mcmc.load_site("SITE.edi")    # via data_proc.load_edi(...)
site = mcmc.load_site("SITE.npz")    # NPZ with a single key "data_dict"
```

For project-style NPZ storage (single key only):

```python
import data_proc

data_proc.save_npz(site, "SITE.npz")     # stores under key="data_dict"
site = data_proc.load_npz("SITE.npz")    # returns the dict
```

---

## Phase tensor handling

If you set `use_pt=True` in `build_pymc_model`, the sampler expects `P` (and
optionally `P_err`) in the site dict.

Recommended: always call

```python
site = mcmc.ensure_phase_tensor(site, nsim=200)
```

Notes:

- `ensure_phase_tensor` **always recomputes** the phase tensor from the observed `Z` using  
  `P = inv(Re(Z)) @ Im(Z)`.
- If `P_err` is missing and `Z_err` exists, a **Monte‑Carlo** estimate of `P_err` is attempted.

---

## Starting model dictionary (`model0`)

A valid starting model is a plain dict with (at minimum):

```python
model0 = dict(
    h_m=...,          # (nl,) meters, last entry is basement placeholder
    rho_min=...,      # (nl,) Ω·m
    rho_max=...,      # (nl,) Ω·m
    strike_deg=...,   # (nl,) deg
    is_iso=...,       # (nl,) bool  (optional; defaults False)
    is_fix=...,       # (nl,) bool  (optional; defaults False)
)
```

You may also provide conductivities instead of resistivities:

- `sigma_min`, `sigma_max` (S/m) + `strike_deg`

`mcmc.normalize_model(model0)` converts everything into a canonical form
containing both rho and sigma fields and guarantees consistent array lengths.

---

## Building and sampling a PyMC model

```python
import mcmc

spec = mcmc.ParamSpec(
    nl=nl,
    fix_h=True,                 # keep per-layer thicknesses fixed
    sample_H_m=False,           # optional global thickness scale
    log10_rho_bounds=(0.0, 5.0),
    strike_bounds_deg=(-180, 180),
)

pm_model, info = mcmc.build_pymc_model(
    site,
    spec=spec,
    model0=model0,
    use_pt=True,
    z_comps=("xx","xy","yx","yy"),
    pt_comps=("xx","xy","yx","yy"),
    prior_kind="default",
    param_domain="rho",         # or "sigma"
    enable_grad=False,          # set True only if you want NUTS/HMC
)

idata = mcmc.sample_pymc(
    pm_model,
    draws=10_000,
    tune=1_000,
    chains=8,
    cores=8,
    step_method="demetropolis", # "nuts" only when enable_grad=True
)
```

### Step methods

`sample_pymc(..., step_method=...)` accepts:

- `"demetropolis"` (default, robust for black-box likelihood)
- `"metropolis"`
- `"nuts"` (requires `enable_grad=True`)
- `"hmc"` (requires `enable_grad=True`)

---

## Outputs

### 1) InferenceData (`*.nc`)

```python
mcmc.save_idata(idata, "SITE_pmc.nc")
```

### 2) Compact summary (`*_pmc_summary.npz`)

```python
summary = mcmc.build_summary_npz(
    station="SITE",
    site=site,
    idata=idata,
    spec=spec,
    model0=model0,
    info=info,
    qpairs=((10,90),(25,75)),   # percentiles or quantiles
)
mcmc.save_summary_npz(summary, "SITE_pmc_summary.npz")
```

The summary NPZ is designed for plotting vertical profiles. Typical keys:

- `periods_s` — `(n,)`
- `h_m0`, `rho_min0`, `rho_max0`, `strike0` — starting model arrays
- `z_m` — `(nl,)` depth-to-interface array (m), derived from the reference thickness
- `q` — quantile grid used for envelopes (includes 0.5)
- `rho_min_q`, `rho_max_q`, `strike_q` — `(nq,nl)` quantiles
- `sigma_min_q`, `sigma_max_q` — `(nq,nl)` quantiles
- optional: `z_m_q` (interface depth quantiles) and `H_m_q` (global thickness scale)

---

## Plotting (three-panel profiles)

`mcmc_viz.py` is **axes-based**; the calling script creates the figure.

```python
import matplotlib.pyplot as plt
import mcmc_viz as mv

s = mv.load_summary_npz("SITE_pmc_summary.npz")
idata = mv.open_idata("SITE_pmc.nc")

fig, axs = plt.subplots(3, 1, figsize=(8, 8))

mv.plot_paramset_threepanel(
    axs,
    summary=s,
    idata=idata,
    param_domain="rho",
    param_set="minmax",         # or "max_anifac"
    qpairs=((0.1,0.9),(0.25,0.75)),
)

fig.tight_layout()
plt.show()
```

---

Author: Volker Rath (DIAS)  
Updated with the help of ChatGPT (GPT-5.2 Thinking) on 2026-02-13 (UTC)
