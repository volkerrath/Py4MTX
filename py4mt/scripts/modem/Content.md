# ModEM Scripts — Catalogue

**py4mt / py4mtx** — ModEM (3-D MT inversion, non-linear conjugate gradients)
workflow scripts.  All scripts live in `scripts/modem/`.

---

## Model manipulation

| Script | Purpose |
|--------|---------|
| `modem_mod_fill.py` | Replace all subsurface cells in a ModEM `.rho` model with a constant resistivity, preserving air and sea cells. Standard way to create a homogeneous starting model. |
| `modem_mod_stats.py` | Compute cell-wise statistics (mean, variance, median, percentiles) across a ModEM model ensemble, or print summary statistics and histograms for a single model. |
| `modem_mod_trans.py` | Convert a ModEM `.rho` model to UBC (`.mod` + `.mesh`) or RLM/CGG (`.rlm`) format. |
| `modem_improc.py` | Apply spatial image-processing filters (Gaussian smoothing, median filter, or anisotropic diffusion) to a ModEM 3-D resistivity model. |
| `modem_insert_body.py` | Insert synthetic geometric bodies (ellipsoids or boxes) into a ModEM model with optional smoothing; supports replace and additive modes with optional conditions. |
| `modem_insert_multi.py` | Generate multiple perturbed models (checkerboard or random anomalies) and project each through the Jacobian null-space via pre-computed SVD for resolution testing. |
| `modem_compare.py` | Compare two ModEM 3-D models by computing log₁₀(ρ) difference and cross-gradient. |
| `modem_compress.py` | Apply spectral or basis-function compression to a ModEM resistivity model.  Six methods: 3-D DCT (radial or separable), 3-D DWT (wavelet), Legendre-z × DCT-xy, B-spline-z × DCT-xy, and Karhunen–Loève / PCA.  Reports RMS, relative RMS, and max reconstruction error; optional truncation sweep. |

---

## Jacobian analysis

| Script | Purpose |
|--------|---------|
| `modem_jac_proc.py` | Read a raw ModEM Jacobian, optionally normalise rows by data errors, mask air cells, and sparsify.  Primary Jacobian preprocessing step. |
| `modem_jac_grad.py` | Read a processed Jacobian and compute sensitivity-weighted gradient quantities for model-update analysis (numba-accelerated). |
| `modem_jac_stats.py` | Compute and print summary statistics on a processed Jacobian for the full matrix and for subsets by component, site, or frequency band. |
| `modem_jac_sens.py` | Compute sensitivity / coverage maps from a Jacobian for the full dataset and subsets (data type, component, site, frequency band); supports raw, absolute, and Euclidean types with volume normalisation. |
| `modem_jac_svd.py` | Compute a randomised truncated SVD of a processed Jacobian over a grid of rank / oversampling / subspace-iteration parameters.  U, S, V used by `modem_insert_multi.py`. |
| `modem_jac_splitmerge.py` | Merge separate Jacobian files (Z, tipper, PT) into one, or split a merged Jacobian by frequency band, transfer-function component, or data type. |

---

## Data and inversion utilities

| Script | Purpose |
|--------|---------|
| `modem_data_split.py` | Split ModEM data files into separate files by period band for band-by-band inversion or data quality inspection. |
| `modem_generate_alphas.py` | Compute depth-dependent smoothing alpha parameters that vary linearly with depth between user-specified bounds, for ModEM regularisation. |
| `modem_generate_sites_synthetic.py` | Create a rectangular grid of synthetic MT station locations and write one EDI file per site from a template, for forward-modelling studies. |
| `modem_plot_rms.py` | Parse ModEM `.log` files to extract nRMS at each iteration and plot all convergence curves on a single figure; writes per-run `.csv` files. |
| `modem_plot_slices.py` | Read a ModEM model and prepare horizontal and vertical cross-sections for visualisation. *(Work-in-progress stub — model loading works; slice plotting not yet implemented.)* |

---

## Summary

| Group | Scripts |
|-------|---------|
| Model manipulation | 8 |
| Jacobian analysis | 6 |
| Data / inversion utilities | 5 |
| **Total** | **19** |

---

*Generated 2026-06-26 from README files in `scripts/modem/`.*
