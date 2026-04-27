# py4mt Script Catalogue

**Python for Magnetotellurics** — script index generated from README files.

All scripts are part of the `py4mt` package (DIAS / Volker Rath).
They are organised by inversion code and functional group.

---

## FEMTIC scripts

FEMTIC is a 3-D finite-element MT inversion code (Yoshiya Usui).

### Uncertainty quantification — Randomize-Then-Optimize (RTO)

| Script | Purpose |
|--------|---------|
| `femtic_rto_rough.py` | Read FEMTIC `roughening_matrix.out` and save the sparse roughness matrix `R` (or precision matrix `Q = RᵀR`) as a compressed `.npz` file. First step in the RTO pipeline. |
| `femtic_rto_prior.py` | Build a prior covariance proxy `M ≈ α²(R + εI)⁻¹(R + εI)⁻ᵀ` from the roughness matrix. Optionally sparsify and save as `.npz`. |
| `femtic_rto_prep.py` | Generate the full RTO ensemble: perturb `observe.dat` with Gaussian noise and draw prior model perturbations from `N(0, (RᵀR)⁻¹)` via randomized SVD. Creates one directory per member ready for FEMTIC submission. |
| `femtic_rto_post.py` | Postprocess a converged RTO ensemble: collect models, compute cell-wise statistics (mean, variance, median, MAD, percentiles) and the empirical covariance matrix, and save to `RTO_results.npz`. |

**RTO workflow:**
```
femtic_rto_rough.py  →  R_coo.npz
femtic_rto_prior.py  →  invRTR_*.npz   (optional)
femtic_rto_prep.py   →  ensemble directories
                         (run FEMTIC on each member)
femtic_rto_post.py   →  RTO_results.npz
femtic_ens_from_covar.py  →  resample from covariance
```

---

### Uncertainty quantification — Ensemble analysis

| Script | Purpose |
|--------|---------|
| `femtic_ens_decomp.py` | Scan a directory of converged FEMTIC runs, filter by nRMS threshold, and perform PCA or ICA decomposition (scikit-learn) on the stacked ensemble matrix. Prints explained-variance ratios and singular values. |
| `femtic_ens_eof.py` | Compute Empirical Orthogonal Functions (EOFs) via SVD on a demeaned ensemble matrix. Reconstructs models from truncated or individual EOF modes for data-driven uncertainty characterisation. |
| `femtic_ens_from_covar.py` | Draw new model samples from a posterior covariance via Cholesky decomposition: `m_new = m_ref + L·z`, `z ~ N(0,I)`. Follows the method of Osypov et al. (2013). |

---

### Uncertainty quantification — Jackknife

| Script | Purpose |
|--------|---------|
| `femtic_jcn_prep.py` | Set up a jackknife uncertainty analysis: create N member directories from template files and generate reduced `observe.dat` files (leave-one-site-out or random subsets). |

---

### Diagnostics and visualisation

| Script | Purpose |
|--------|---------|
| `femtic_plot_convergence.py` | Read `femtic.cnv` convergence files from one or more inversion directories and plot misfit, nRMS, or model roughness vs. iteration number. |
| `femtic_plot_lcurve.py` | Collect final-iteration roughness and misfit from multiple FEMTIC runs at different regularisation parameters (alpha) and plot the L-curve with annotated alpha values. |

---

## ModEM scripts

ModEM is a 3-D MT inversion code based on non-linear conjugate gradients.

### Model manipulation

| Script | Purpose |
|--------|---------|
| `modem_mod_fill.py` | Replace all subsurface cells in a ModEM `.rho` model with a constant resistivity, preserving air and sea cells. Standard way to create a homogeneous starting model. |
| `modem_mod_stats.py` | Compute cell-wise statistics (mean, variance, median, percentiles) across a ModEM model ensemble, or print summary statistics and histograms for a single model. |
| `modem_mod_trans.py` | Convert a ModEM `.rho` model to UBC (`.mod` + `.mesh`) or RLM/CGG (`.rlm`) format. |
| `modem_improc.py` | Apply spatial image-processing filters (Gaussian smoothing, median filter, or anisotropic diffusion) to a ModEM 3-D resistivity model. |
| `modem_insert_body.py` | Insert synthetic geometric bodies (ellipsoids or boxes) into a ModEM model with optional smoothing. Supports replace and additive modes with optional conditions. |
| `modem_insert_multi.py` | Generate multiple perturbed models (checkerboard or random anomalies) and project each through the Jacobian null-space via pre-computed SVD for resolution testing. |
| `modem_compare.py` | Compare two ModEM 3-D models by computing log₁₀(ρ) difference and cross-gradient. |

---

### Jacobian analysis

| Script | Purpose |
|--------|---------|
| `modem_jac_proc.py` | Read a raw ModEM Jacobian, optionally normalise rows by data errors, mask air cells, and sparsify. Primary Jacobian preprocessing step. |
| `modem_jac_stats.py` | Compute and print summary statistics on a processed Jacobian for the full matrix and for subsets split by component, site, or frequency band. |
| `modem_jac_sens.py` | Compute sensitivity / coverage maps from a Jacobian for the full dataset and subsets (by data type, component, site, frequency band). Supports raw, absolute, and Euclidean sensitivity types with volume normalisation. |
| `modem_jac_svd.py` | Compute a randomised truncated SVD of a processed Jacobian over a grid of rank / oversampling / subspace-iteration parameters. Output U, S, V used by `modem_insert_multi.py`. |
| `modem_jac_splitmerge.py` | Merge separate Jacobian files (Z, tipper, PT) into one, or split a merged Jacobian by frequency band, transfer-function component, or data type. |

---

### Data and inversion utilities

| Script | Purpose |
|--------|---------|
| `modem_data_split.py` | Split ModEM data files into separate files by period band for band-by-band inversion or data quality inspection. |
| `modem_generate_alphas.py` | Compute depth-dependent smoothing alpha parameters that vary linearly with depth between user-specified bounds, for ModEM regularisation. |
| `modem_generate_sites_synthetic.py` | Create a rectangular grid of synthetic MT station locations and write one EDI file per site from a template, for forward-modelling studies. |
| `modem_plot_rms.py` | Parse ModEM `.log` files to extract nRMS at each iteration and plot all convergence curves on a single figure. Also writes per-run `.csv` files. |
| `modem_plot_slices.py` | Read a ModEM model and prepare horizontal and vertical cross-sections for visualisation. *(Work-in-progress stub — model loading works; slice plotting not yet implemented.)* |

---

## General MT scripts

Scripts applicable across inversion codes or for data pre/post-processing.

### Data processing

| Script | Purpose |
|--------|---------|
| `mt_data_processor.py` | Batch-process `.edi` files: compute phase tensor, impedance invariants, and apparent resistivity/phase; set or estimate errors; optionally interpolate or rotate; export to EDI, NPZ, HDF, or MAT format. |
| `mt_make_sitelist.py` | Read all `.edi` files in a directory and write a site list (name, latitude, longitude, elevation) in WALDIM, FEMTIC, or general CSV format. |
| `mt_calc_ptdim.py` | Use phase-tensor analysis to classify each MT station / frequency as 1-D, 2-D, or 3-D. Writes per-site dimensionality tables and a combined summary. |
| `mt_kmz_waldim.py` | Export WALDIM dimensionality classification results as a KMZ file for Google Earth, with colour-coded site markers (3-class or 10-class scheme). |
| `mt_archive_run.py` | Clean and archive FEMTIC-style iteration directories: keep lowest/highest N iterations, protect specified files, delete the rest (dry-run by default), and create a compressed archive (`.zip` / `.tgz`). |

---

### Plotting

| Script | Purpose |
|--------|---------|
| `mt_plot_data.py` | Generate per-station 3×2 diagnostic subplot figures (apparent resistivity, phase, tipper, phase tensor) from `.npz`, `.edi`, or `.dat` files. Optionally assembles a PDF catalogue. |
| `mt_plot_strikes.py` | Generate strike-direction rose diagrams (aggregate, per-decade, per-station) from MT impedance data using mtpy. |

---

### 1-D inversion — anisotropic (deterministic + Bayesian)

| Script | Purpose |
|--------|---------|
| `mt_aniso1d_forward.py` | Compute full impedance tensor, apparent resistivity/phase, and phase tensor for a layered anisotropic conductivity model. Exports to text files and plots. |
| `mt_aniso1d_inversion.py` | Batch deterministic Gauss-Newton inversion of anisotropic 1-D MT data with Tikhonov or TSVD regularisation and automatic regularisation parameter selection (GCV, L-curve, ABIC). |
| `mt_aniso1d_sampler.py` | PyMC driver for Bayesian (MCMC) anisotropic 1-D MT inversion. Supports Gaussian and Matérn covariance priors. All heavy lifting delegated to `mcmc.py`. |
| `mt_aniso1d_plot.py` | Plot posterior parameter distributions from anisotropic 1-D sampler results as three-panel depth profiles (ρ_min/ρ_max/strike or equivalent) with uncertainty bands. |

---

### 1-D inversion — transdimensional (rjMCMC)

| Script | Purpose |
|--------|---------|
| `mt_transdim1d.py` | Reversible-jump MCMC inversion of MT apparent-resistivity data for a 1-D layered earth with free number of layers. Supports isotropic and anisotropic forward models. Delegates sampling to `transdim.py` and plotting to `transdim_viz.py`. |
| `mt_plot_rjmcmc.py` | Visualise posterior resistivity-depth distributions from the rjmcmc-MT transdimensional sampler. Optionally adds site coordinates from EDI files and assembles a PDF catalogue. |

---

## Summary by count

| Group | Scripts |
|-------|---------|
| FEMTIC — RTO uncertainty | 4 |
| FEMTIC — ensemble analysis | 3 |
| FEMTIC — jackknife | 1 |
| FEMTIC — diagnostics | 2 |
| ModEM — model manipulation | 7 |
| ModEM — Jacobian analysis | 5 |
| ModEM — data / inversion utilities | 5 |
| General MT — data processing | 5 |
| General MT — plotting | 2 |
| General MT — anisotropic 1-D | 4 |
| General MT — transdimensional 1-D | 2 |
| **Total** | **40** |

---

*Generated 2026-04-27 from README files in `scripts.zip`.*
