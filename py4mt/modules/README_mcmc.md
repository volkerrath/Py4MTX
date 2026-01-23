# README_mcmc.md

Bayesian inversion (PyMC) for **anisotropic 1‑D MT** (impedance; optional phase tensor)
=====================================================================================

This mini-package glues together:

- `aniso.py`: anisotropic 1‑D MT forward model (and optional sensitivities).
- `mcmc.py`: PyMC model builder + sampling wrappers.
- `mt_aniso1d_sampler.py`: *script-style* driver (not a CLI) to run per-site inversions.
- `mcmc_viz.py`: axis-based plotting helpers for sampler outputs.

Typical workflow
----------------

1. Load MT data (EDI or NPZ) with `data_proc.py`.
2. Build and sample a PyMC model (impedance; optionally phase tensor).
3. Save:
   - an ArviZ `InferenceData` NetCDF (`*_pmc.nc`)
   - a compact NPZ summary (`*_pmc_summary.npz`) for quick plotting/archiving
4. Plot results using `mcmc_viz.py`.

See `mt_aniso1d_sampler.py` for a ready-to-run template.


Contents
--------

- Installation & dependencies
- Input data format
- Model parameterization
- Running the sampler
- Outputs
- Plotting
- Notes on gradients & parallelism
- Troubleshooting


Installation & dependencies
---------------------------

### Python packages

Required for sampling:

- `numpy`
- `pymc`
- `arviz`

Required for plotting:

- `matplotlib`

`mcmc.py` contains defensive import logic for environments where optional dependencies
can cause import issues, and it configures Aesara to run in a “no C compilation”
mode (useful on minimal systems / HPC login nodes).

### Local project modules

These are used as local imports:

- `aniso.py` (forward model)
- `data_proc.py` (EDI/NPZ I/O; optional phase tensor computation)
- `mcmc.py`, `mcmc_viz.py`
- `mt_aniso1d_forward.py` (example forward runs / template models)


Input data format
-----------------

`mt_aniso1d_sampler.py` accepts a glob of either:

- EDI files (`*.edi`) loaded via `data_proc.load_edi(...)`, or
- NPZ site files (`*.npz`) containing at least `freq` and `Z` (+ errors).

A “site dictionary” is expected to contain:

- `freq` (Hz), shape `(nper,)`
- `Z` (complex impedance tensor), shape `(nper, 2, 2)`
- `Z_err` (same shape as `Z`), either **std** or **var**
- `err_kind` string: `"std"` or `"var"`

Optionally:

- `P` and `P_err` (phase tensor and its uncertainty), if you enable PT likelihood.

If you produce NPZ site files via your `data_proc.py` utilities, keep `err_kind`
consistent with how you store `Z_err`.


Model parameterization
----------------------

The inversion uses the public parameterization expected by the forward model:

Per layer:

- `h_m` : thickness in meters, shape `(nl,)`
- `rop` : principal resistivities [Ω·m], shape `(nl, 3)`
- `ustr_deg, udip_deg, usla_deg` : Euler angles in degrees, shape `(nl,)`

This is converted internally (via `aniso.cpanis(...)`) to the effective anisotropic
parameters used in the layered recursion.

Thickness can be **fixed** during sampling (`FIX_H=True`), which removes `h_m` from
the sampled parameter vector.


Running the sampler
-------------------

### 1) Configure `mt_aniso1d_sampler.py`

This driver is *not a CLI*: edit the “USER CONFIG” section and run it:

```bash
python mt_aniso1d_sampler.py
```

Key settings in the driver:

- `INPUT_GLOB`: EDI or NPZ input files (one “site” per file)
- `OUTDIR`: output directory
- `MODEL_NPZ`: starting model template
- `MODEL_DIRECT`: alternatively define the starting model in-code and have it saved to NPZ
- `USE_PT`: include phase tensor likelihood (optional)
- `Z_COMPS`, `PT_COMPS`: select components used in likelihood
- `FIX_H`: sample or fix thickness
- `STEP_METHOD`: `"demetropolis" | "metropolis" | "nuts" | "auto"`
- `DRAWS`, `TUNE`, `CHAINS`, `CORES`
- `ENABLE_GRAD`: optionally expose gradients (needed for NUTS experiments)

The default driver uses the gradient-free **DEMetropolisZ** step method, which is robust
for “black-box” forward models.

### 2) (Optional) create / test a starting model

`mt_aniso1d_forward.py` contains example models and shows how to call the forward model
directly. In the sampler, if `MODEL_DIRECT` is not `None`, it will be written to
`MODEL_DIRECT_SAVE_PATH` (defaulting to `MODEL_NPZ`) and used as the template model.


Outputs
-------

For each input station/site, `mt_aniso1d_sampler.py` writes:

1. `<outdir>/<station>_pmc.nc`  
   ArviZ `InferenceData` NetCDF with posterior samples and sampling stats.

2. `<outdir>/<station>_pmc_summary.npz`  
   A compact NPZ summary designed for fast plotting and archiving.

### Summary NPZ: keys written by the current driver

As of the current `mt_aniso1d_sampler.py`, the summary contains:

Observed data:

- `station` (str)
- `freq` : `(nper,)`
- `Z_obs` : `(nper, 2, 2)` complex
- `Z_err` : `(nper, 2, 2)` (or `None`)
- optionally `P_obs`, `P_err` (or `None`)
- `err_kind` (str)
- `z_comps`, `pt_comps` (arrays of component names)

Posterior summaries:

- `theta_med` : 1-D parameter vector median
- `theta_qpairs` : quantile pairs used (e.g. `[(0.1, 0.9), (0.25, 0.75), ...]`)
- `theta_qlo`, `theta_qhi` : low/high quantiles for each pair
- `param_names` : names aligned with `theta_*`

Model-space medians:

- `h_m_med` (absent/empty if `FIX_H=True`)
- `rop_med`
- `ustr_deg_med`, `udip_deg_med`, `usla_deg_med`
- `is_iso` (copied from template model; layer isotropy flags)

A single “forward run” at the median model:

- `Z_pred` : predicted impedance tensor at the median model

If you want vertical-profile plots with credible intervals (qlo/qhi) in model space,
extend the summary writer in `mt_aniso1d_sampler.py` to also store (for example)
`rop_qlo`, `rop_qhi`, etc., plus `z_bot_med = cumsum(h_m_med)`.


Plotting
--------

`mcmc_viz.py` contains **axis-based** plotting functions (it does not create figures).
You typically:

- load the summary NPZ
- open the NetCDF `InferenceData`
- plot traces/densities from `InferenceData`
- plot model/data summaries from the NPZ

Example:

```python
import matplotlib.pyplot as plt
import mcmc_viz as mv

s = mv.load_summary_npz("SITE_pmc_summary.npz")
idata = mv.open_idata("SITE_pmc.nc")

fig, axs = plt.subplots(2, 1, figsize=(7, 7))
mv.plot_theta_trace(axs[0], idata, idx=0, name="theta[0]")
mv.plot_theta_density(axs[1], idata, idx=0)
fig.tight_layout()
fig.show()
```

Notes:

- Depth-profile plots in `mcmc_viz.py` currently expect keys like `z_bot_med` and
  `*_qlo/*_qhi` for model parameters. If your summary NPZ does not contain them yet,
  either (a) add them in the sampler, or (b) plot medians only (or work directly from
  `InferenceData`).

- Data-fit plots can be made either from `Z_pred` (median model) or by computing
  apparent resistivity/phase from observed/predicted Z inside your plotting scripts.


Notes on gradients & parallelism
--------------------------------

### Gradients

- Default sampling is gradient-free (`DEMetropolisZ`) and works with a pure-Python
  forward model.
- If `ENABLE_GRAD=True`, you can experiment with `STEP_METHOD="nuts"` (NUTS needs gradients).

### Parallelism (chains / cores)

Parallelism is controlled by:

- `CHAINS`: number of Markov chains
- `CORES`: number of processes used to run chains in parallel

PyMC evaluates chains concurrently via multiprocessing. If the forward model is CPU-bound
and pure-Python, speedups depend on how much time is spent inside NumPy vs Python loops.


Troubleshooting
---------------

- **Aesara tries to compile C code and fails** (missing headers / no compiler):  
  `mcmc.py` configures Aesara to avoid C compilation.

- **Variance vs std confusion**:  
  Ensure `err_kind` matches how `Z_err` is stored (`"var"` vs `"std"`). The sampler and
  likelihood must interpret the uncertainty consistently.

- **Numerical problems from tiny uncertainties**:  
  Set/raise `SIGMA_FLOOR_Z` (and `SIGMA_FLOOR_P` if using PT) in the sampler driver.

- **PT likelihood unstable / too strong**:  
  Start impedance-only (`USE_PT=False`, `Z_COMPS=("xy","yx")`), then enable PT once the
  basic run behaves, and make sure `P_err` is realistic (bootstrap size matters).

Notes on depth axis and conductivity
------------------------------------

- Depth is plotted **positive downward** (the y-axis is inverted) for all depth-profile functions.
- You can plot **conductivity** instead of resistivity in model profiles via:
  `plot_depth_resistivity(..., quantity="conductivity")`.
- In data-fit plots you can plot **apparent conductivity** via:
  `plot_rho_phase_fit(..., rho_quantity="conductivity")`.
