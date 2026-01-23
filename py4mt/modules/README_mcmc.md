# README_mcmc.md

Bayesian inversion (PyMC) for **anisotropic 1‑D MT** (impedance; optional phase tensor)

This mini-package glues together:

- **`aniso.py`**: anisotropic 1‑D MT forward model (and optional sensitivities).
- **`mcmc.py`**: PyMC model builder + sampling wrappers.
- **`mt_aniso1d_sampler.py`**: script-style driver (not a CLI) to run per-site inversions.
- **`mcmc_viz.py`**: axis-based plotting helpers for the sampler outputs.

The intended workflow is:

1. Load MT data (EDI or NPZ) with `data_proc.py`.
2. Build and sample a PyMC model (impedance; optionally phase tensor).
3. Save:
   - an ArviZ `InferenceData` NetCDF (`*_pmc.nc`)
   - a compact NPZ summary (`*_pmc_summary.npz`) for plotting/archiving
4. Plot results using `mcmc_viz.py`.

---

## Installation & dependencies

Runtime (sampling):

- `numpy`
- `pymc`
- `arviz` (for `InferenceData` + diagnostics)

Plotting:

- `matplotlib`

Local project modules used as imports:

- `aniso.py` (forward model + sensitivities)
- `data_proc.py` (EDI/NPZ I/O; MT derived quantities like phase tensor, rho/phase)
- `mcmc.py`, `mcmc_viz.py`
- `mt_aniso1d_forward.py` (example forward runs / model templates)

Notes on backends:

- Recent PyMC versions use **PyTensor**; older installations used **Aesara**.
  The code follows whatever backend your installed PyMC provides.

---

## Input data format

`mt_aniso1d_sampler.py` accepts a glob of either:

- **EDI** files (`*.edi`) — loaded via `data_proc.load_edi(...)`
- **NPZ** site files (`*.npz`) — expected to contain at least `freq` and `Z` (+ errors)

The sampler uses these keys from the site dict:

- `freq` (Hz)
- `Z` (complex impedance tensor, shape `(nper, 2, 2)`)
- `Z_err` (same shape as `Z`, either **std** or **var**, controlled by `err_kind`)
- `err_kind` (`"std"` or `"var"`; applies to `*_err` arrays)

Optional phase tensor likelihood (if enabled):

- `P` (phase tensor, shape `(nper, 2, 2)`)
- `P_err` (same shape, std or var, controlled by `err_kind`)

If `P` is not present, it can be computed from `Z` using `data_proc.compute_pt(..., err_method="none")`.

---

## Model parameterization

The forward model API (and therefore the inversion) uses the public parameterization
implemented in `aniso.aniso1d_impedance_sens(...)`:

Per layer:

- `h_m` : thickness in meters, shape `(nl,)`
- `rop` : principal resistivities `[Ω·m]`, shape `(nl, 3)`
- `ustr_deg, udip_deg, usla_deg` : Euler angles (strike, dip, slant) in degrees, shape `(nl,)`
- `is_iso` : optional isotropy flag per layer (if present/used in your forward model wrapper)

Sampling parameterization (typical):

- thicknesses sampled as `log10(h_m)`
- resistivities sampled as `log10(rop)`
- angles sampled in degrees with bounds

Thickness can be held fixed (`FIX_H=True`), in which case thicknesses are removed from the sampled vector.

---

## Running the sampler

`mt_aniso1d_sampler.py` is **not** a CLI: edit the “USER CONFIG” section and run it:

```bash
python mt_aniso1d_sampler.py
```

Key settings you typically adjust:

- `INPUT_GLOB`, `OUTDIR`
- model template: `MODEL_NPZ` or `MODEL_DIRECT` (saved to NPZ for reproducibility)
- likelihood switches: `USE_PT`, `Z_COMPS`, `PT_COMPS`
- thickness handling: `FIX_H`
- sampler controls: `STEP_METHOD`, `DRAWS`, `TUNE`, `CHAINS`, `CORES`

Default step method is gradient-free (e.g., `DEMetropolisZ`) which is robust for
black-box likelihoods.

---

## Outputs

For each input station/site file, the sampler writes:

1. `"<outdir>/<station>_pmc.nc"`
   - ArviZ `InferenceData` NetCDF with posterior samples and sampler stats.

2. `"<outdir>/<station>_pmc_summary.npz"`
   - Compact summary NPZ designed for quick plotting and archiving.

### Summary NPZ: typical keys

Observed data:

- `freq` : `(nper,)`
- `Z_obs` : `(nper, 2, 2)` complex
- (optional) `P_obs` : `(nper, 2, 2)`

Model summary (layered parameters):

- `z_bot_med` : `(nl,)` depth to bottom of each layer (m)
- `rop_med` : `(nl, 3)` principal resistivities (Ω·m)
- `ustr_deg_med`, `udip_deg_med`, `usla_deg_med` : `(nl,)`
- `h_m_med` : `(nl,)` thickness (m) — may be absent if thickness is fixed
- optional credible bands (when requested): `*_qlo`, `*_qhi`

Predictive summaries (data space):

- `rho_med`, `rho_qlo`, `rho_qhi` : apparent resistivity for all tensor components
- `phase_deg_med`, `phase_deg_qlo`, `phase_deg_qhi` : phase (deg)
- (optional) `P_med`, `P_qlo`, `P_qhi` : phase tensor element summaries

Quantile-pair metadata (stored as Python objects in NPZ, load with `allow_pickle=True`):

- `theta_qpairs` : quantile pairs used for parameter-space summaries
- `model_qpairs` : quantile pairs used for model profiles
- `pred_qpairs` : quantile pairs used for predictive summaries

---

## Plotting

`mcmc_viz.py` contains **axis-based** plotting functions (it does not create figures).

Typical pattern:

```python
import matplotlib.pyplot as plt
import mcmc_viz as mv

s = mv.load_summary_npz("SITE_pmc_summary.npz")
idata = mv.open_idata("SITE_pmc.nc")

fig, axs = plt.subplots(3, 1, figsize=(6, 10))

# traces / marginal densities
mv.plot_theta_trace(axs[0], idata, idx=0, name="theta[0]")
mv.plot_theta_density(axs[1], idata, idx=0, qpairs=s.get("theta_qpairs"))

# depth profiles (positive down by default)
mv.plot_depth_resistivity(axs[2], s, comp=0, use_log10=True)
# or conductivity:
# mv.plot_depth_resistivity(axs[2], s, comp=0, quantity="conductivity", use_log10=True)

fig.tight_layout()
fig.show()
```

Useful plot families:

- **Depth credible intervals** (depth increases downward by default):
  - `plot_depth_resistivity(...)` (ρ1..ρ3)
  - `plot_depth_angle(...)` (strike/dip/slant or any stored angle key)
  - `plot_depth_thickness(...)` (h)

- **Data fit (ρ/phase)**:
  - `plot_rho_phase_fit(...)` for one impedance component (e.g., `"xy"`)
  - `plot_phase_tensor_element(...)` for one PT element (if present in summary)

Deprecated compatibility wrappers remain available:

- `plot_vertical_resistivity`, `plot_vertical_angle`, `plot_vertical_thickness`

---

## Notes on gradients & parallelism

- Default sampling is **gradient-free** (e.g., DEMetropolisZ) and works without
  differentiating the forward model.
- If/when analytic gradients are enabled (via sensitivities), you may try
  gradient-based samplers (e.g., NUTS), but this is optional and model-dependent.
- Parallelism is handled by PyMC via the `cores`/`chains` settings; heavy forward
  calculations can still dominate runtime.

---

## Troubleshooting

- If PyMC import fails due to optional `numba/coverage` interactions in your environment,
  `mcmc.py` includes defensive import preparation to reduce such issues.
- If you run on minimal systems without a C compiler / Python headers, the backend may
  attempt C compilation; `mcmc.py` forces a pure-Python execution mode where possible.

