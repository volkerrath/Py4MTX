# py4mt Script Catalogue

**Python for Magnetotellurics** — script index generated from README files.

All scripts are part of the `py4mt` / `py4mtx` package (DIAS / Volker Rath).
They are organised into subdirectories by inversion code and functional group:
`femtic/`, `modem/`, `general/`, and `joint/` (each with its own `Content.md`
holding the detailed listing), plus `utils/` and `special/` (listed at the end).
The template `py4mt_template.py` and the helper `py2ipynb` remain at the scripts root.

---

## FEMTIC scripts  →  [`femtic/Content.md`](femtic/Content.md)

FEMTIC is a 3-D finite-element MT inversion code (Yoshiya Usui).

### Model visualisation

| Script | Purpose |
|--------|---------|
| `femtic/femtic_mod_plot_slice.py` | 2-D slice panels (map, N-S, E-W, arbitrary plane) with site overlays, compass corner labels, and nRMS annotation from `femtic.cnv`. |
| `femtic/femtic_mod_plot_bh.py` | Standalone 1-D ρ(z) borehole resistivity logs from a tetrahedral mesh. |
| `femtic/femtic_plot_nrms.py` | Map of per-site nRMS misfit (best iteration from `femtic.cnv`, residuals from `results_iter<N>.h5`). |
| `femtic/femtic_plot_distort.py` | Map of static galvanic distortion matrices as ellipses (optional Groom–Bailey panels). |
| `femtic/femtic_mod_plot_3d.py` | PyVista 3-D rendering and VTK/VTU export; slice planes, iso-surfaces, spatial clipping. |

---

### Model manipulation

| Script | Purpose |
|--------|---------|
| `femtic/femtic_mod_edit.py` | Apply arithmetic operations to a FEMTIC resistivity block in log₁₀ space (null, fill, clip, shift, smooth, ellipsoid/brick insertion, mean, median, standardise). |
| `femtic/femtic_mod_math.py` | Generate ensemble-derived synthetic members: element-wise log₁₀(ρ) mean and mesh-adaptive smoothed median. |

---

### Diagnostics and L-curve

| Script | Purpose |
|--------|---------|
| `femtic/femtic_convergence_plot.py` | Plot misfit, nRMS, or roughness vs. iteration from one or more `femtic.cnv` files. |
| `femtic/femtic_plot_conv.py` | Plot any `femtic.cnv` field(s) vs. iteration, overlaid across run directories. |
| `femtic/femtic_data_misfit.py` | Crossplots, normalised-residual histograms and Q-Q plots (and optional response curves) from `results_iter<N>.h5`; single run or ensemble. |
| `femtic/femtic_lcurve_plot.py` | Plot L-curve (roughness vs. nRMS / misfit) across multiple α runs; auto-detects distortion column layout. |

---

### Uncertainty quantification — RTO

| Script | Purpose |
|--------|---------|
| `femtic/femtic_rto_rough.py` | Read `roughening_matrix.out`, save sparse R / Q = RᵀR as `.npz`. |
| `femtic/femtic_rto_prior.py` | Build prior covariance proxy from roughness matrix. |
| `femtic/femtic_rto_prep.py` | Generate full RTO ensemble (perturb data + draw prior model perturbations). |

**RTO workflow:** `rto_rough` → `rto_prior` → `rto_prep` → *(run FEMTIC)* → `ens_post`

---

### Uncertainty quantification — GST

| Script | Purpose |
|--------|---------|
| `femtic/femtic_gst_prep.py` | Generate geostatistical ensemble via Ordinary Kriging pilot points (gstools). |

---

### Uncertainty quantification — Ensemble analysis

| Script | Purpose |
|--------|---------|
| `femtic/femtic_ens_post.py` | Algorithm-agnostic ensemble postprocessing: nRMS filter, statistics, covariance, optional slice figures and borehole profiles (`MOD_BH`). Replaces the former `femtic_rto_post.py`. |
| `femtic/femtic_ens_plot.py` | Joint multi-row slice figure for ensemble runs with optional statistics rows and borehole columns. |
| `femtic/femtic_ens_repair.py` | Convergence diagnostic for all ensemble directories; restart directories for non-converged members. |
| `femtic/femtic_ens_decomp.py` | PCA / ICA decomposition of a filtered ensemble matrix. |
| `femtic/femtic_ens_eof.py` | EOFs via SVD on a demeaned ensemble matrix. |
| `femtic/femtic_ens_from_covar.py` | Draw samples from posterior covariance via Cholesky decomposition. |

---

### Uncertainty quantification — Jackknife

| Script | Purpose |
|--------|---------|
| `femtic/femtic_jcn_prep.py` | Create N member directories with reduced `observe.dat` files (leave-one-site-out or random site subsets). |

---

### Nullspace shuttle

| Script | Purpose |
|--------|---------|
| `femtic/femtic_nss.py` | Project a model perturbation onto the null space of the data-weighted Jacobian, leaving predicted data unchanged. |

---

### Utilities

| Script | Purpose |
|--------|---------|
| `femtic/femtic_archive_run.py` | Clean and archive FEMTIC iteration directories; keep selected iterations, protect specified files, compress. |
| `femtic/femtic_summarize_model_cells.py` | Print element composition (air, ocean, fixed, free) for one or more `resistivity_block_iter*.dat` files. |
| `femtic/femtic_summarize_observe_dat.py` | Print data content (sites, frequencies, types: MT/VTF/PT) for one or more `observe.dat` files. |

---

## ModEM scripts  →  [`modem/Content.md`](modem/Content.md)

ModEM is a 3-D MT inversion code based on non-linear conjugate gradients.

### Model manipulation

| Script | Purpose |
|--------|---------|
| `modem/modem_mod_fill.py` | Replace subsurface cells with a constant resistivity (homogeneous starting model). |
| `modem/modem_mod_stats.py` | Cell-wise statistics across a model ensemble or summary statistics for a single model. |
| `modem/modem_mod_trans.py` | Convert ModEM `.rho` model to UBC or RLM/CGG format. |
| `modem/modem_nss.py` | Nullspace shuttle for ModEM (counterpart of `femtic_nss.py`). |
| `modem/modem_improc.py` | Apply spatial filters (Gaussian, median, anisotropic diffusion) to a 3-D ModEM model. |
| `modem/modem_insert_body.py` | Insert synthetic ellipsoids or boxes into a ModEM model with optional smoothing. |
| `modem/modem_insert_multi.py` | Generate perturbed models (checkerboard / random) and project through Jacobian null-space via pre-computed SVD. |
| `modem/modem_compare.py` | Compute log₁₀(ρ) difference and cross-gradient between two ModEM models. |
| `modem/modem_compress.py` | Spectral/basis-function compression: 3-D DCT, DWT, Legendre-z × DCT-xy, B-spline-z × DCT-xy, KL/PCA. |

---

### Jacobian analysis

| Script | Purpose |
|--------|---------|
| `modem/modem_jac_proc.py` | Normalise, mask air cells, and sparsify a raw ModEM Jacobian. Primary preprocessing step. |
| `modem/modem_jac_grad.py` | Sensitivity-weighted gradient quantities from a processed Jacobian (numba-accelerated). |
| `modem/modem_jac_stats.py` | Summary statistics on a processed Jacobian by component, site, or frequency band. |
| `modem/modem_jac_sens.py` | Sensitivity / coverage maps (raw, absolute, Euclidean) with volume normalisation. |
| `modem/modem_jac_svd.py` | Randomised truncated SVD over a parameter grid; output used by `modem_insert_multi.py`. |
| `modem/modem_jac_splitmerge.py` | Merge or split Jacobian files by data type, component, or frequency band. |

---

### Data and inversion utilities

| Script | Purpose |
|--------|---------|
| `modem/modem_data_split.py` | Split ModEM data files by period band. |
| `modem/modem_generate_alphas.py` | Compute depth-dependent smoothing alpha parameters. |
| `modem/modem_generate_sites_synthetic.py` | Create a rectangular grid of synthetic MT station locations. |
| `modem/modem_data_misfit.py` | Crossplots, normalised-residual histograms and Q-Q plots for ModEM observed/calculated data (counterpart of `femtic_data_misfit.py`). |
| `modem/modem_plot_rms.py` | Parse ModEM `.log` files and plot all nRMS convergence curves. |
| `modem/modem_plot_slices.py` | ModEM model slice visualisation. *(Work-in-progress.)* |

---

## General MT scripts  →  [`general/Content.md`](general/Content.md)

Scripts applicable across inversion codes or for data pre/post-processing.

### Data processing

| Script | Purpose |
|--------|---------|
| `general/mt_data_processor.py` | Batch-process `.edi` files: compute PT/invariants/ρ_a/phase, set/estimate errors, interpolate, rotate, export. |
| `general/mt_get_averages.py` | Compute Z_ssq, Z_det, and Z_avg = √(Z_xy·Z_yx) log-averages across a site collection with spread estimates and plots. |
| `general/mt_make_sitelist.py` | Read `.edi` files and write site list in WALDIM, FEMTIC, or CSV format. |
| `general/mt_calc_ptdim.py` | Phase-tensor dimensionality classification (1-D/2-D/3-D) per station and frequency. |
| `general/mt_kmz_waldim.py` | Export WALDIM dimensionality classification as a KMZ file for Google Earth. |

---

### Plotting

| Script | Purpose |
|--------|---------|
| `general/mt_plot_data.py` | Per-station 3×2 diagnostic subplot figures (ρ_a, phase, tipper, PT) with optional PDF catalogue. |
| `general/mt_plot_strikes.py` | Strike-direction rose diagrams (aggregate, per-decade, per-station) from MT impedance data. |

---

### 1-D inversion — anisotropic

| Script | Purpose |
|--------|---------|
| `general/mt_aniso1d_forward.py` | Full impedance tensor, ρ_a/phase, and PT for a layered anisotropic model. |
| `general/mt_aniso1d_inversion.py` | Batch deterministic Gauss-Newton inversion with Tikhonov/TSVD and auto-α selection. |
| `general/mt_aniso1d_sampler.py` | PyMC Bayesian MCMC driver for anisotropic 1-D inversion; Gaussian / Matérn priors. |
| `general/mt_aniso1d_plot.py` | Three-panel posterior depth profiles (ρ_min/ρ_max/strike) with uncertainty bands. |

---

### 1-D inversion — transdimensional (rjMCMC)

| Script | Purpose |
|--------|---------|
| `general/mt_transdim_iso1d.py` | rjMCMC isotropic 1-D inversion; likelihood modes `"Zdet"` and `"rhoa"`. |
| `general/mt_transdim_aniso1d.py` | rjMCMC anisotropic 1-D inversion; likelihood modes `"Z_comps"` and `"rhoa"`; parallel chains. |
| `general/mt_plot_rjmcmc.py` | Visualise rjMCMC posterior ρ(z) distributions; optional PDF catalogue. |

---

## Joint inversion scripts  →  [`joint/Content.md`](joint/Content.md)

ADMM-based joint MT + seismic tomography inversion.

| File | Role |
|------|------|
| `joint/joint_admm_driver.py` | ADMM outer loop; coupling-agnostic. |
| `joint/joint_coupling.py` | ADMM-compatible wrappers for each coupling strategy. |
| `joint/coupling_crossgrad.py` | Cross-gradient structural coupling. |
| `joint/coupling_gramian.py` | Structural Gramian coupling. |
| `joint/coupling_entropy.py` | Mutual-information (entropy) coupling. |
| `joint/coupling_fcm.py` | Fuzzy C-Means latent-field coupling. |
| `joint/coupling_interp.py` | Shared mesh interpolation utilities. |
| `joint/inversion_state.py` | Shared inversion state container. |
| `joint/mt_fwd.py` | FEMTIC MT forward-model wrapper (uses `femtic` model/mesh I/O). |
| `joint/seistomo_fwd.py` | Seismic tomography forward-model wrapper. |
| `joint/seistomo_prep.py` | Seismic tomography data / mesh preprocessing. |

---

## Other directories

These have no `Content.md` of their own.

| Path | Contents |
|------|----------|
| `utils/inspect_hdf5.py` | Recursively list the groups, datasets (shape, dtype, layout, compression) and attributes of an HDF5 file without loading the data. |
| `utils/femtic_modem_trans.py` | Volume-weighted interpolation between a ModEM `.rho` model and a FEMTIC mesh + resistivity block, in both directions (`modem2femtic`, `femtic2modem`). Readme: `utils/femtic_modem_trans_readme.md`. |
| `special/seismt/` | Tacna (southern Peru) imaging pipeline for seismic tomography (Vp, Vs, Vp/Vs) and ModEM resistivity: `precompute.py`, `interpolate.py`, `cluster.py`, `plot_seis.py`, `plot_modem*.py`, `plot_joint.py`, `plotpy.py`, `modem.py` (local copy of ModEM I/O), `make_cpts.py`. See `special/seismt/README*.md`. |
| `py2ipynb` | Shell helper: converts `MT*.py` scripts to notebooks with `jupytext`. |

---

## Summary by count

| Group | Scripts |
|-------|---------|
| FEMTIC — model visualisation | 5 |
| FEMTIC — model manipulation | 2 |
| FEMTIC — diagnostics / L-curve | 4 |
| FEMTIC — RTO uncertainty | 3 |
| FEMTIC — GST uncertainty | 1 |
| FEMTIC — ensemble analysis | 6 |
| FEMTIC — jackknife | 1 |
| FEMTIC — nullspace shuttle | 1 |
| FEMTIC — utilities | 3 |
| ModEM — model manipulation | 9 |
| ModEM — Jacobian analysis | 6 |
| ModEM — data / inversion utilities | 6 |
| General MT — data processing | 5 |
| General MT — plotting | 2 |
| General MT — anisotropic 1-D | 4 |
| General MT — transdimensional 1-D | 3 |
| Joint inversion | 11 |
| **Total** | **72** |

(`utils/` and `special/` are not counted.)

---

*Generated 2026-06-26 from README files in `scripts.zip`; updated 2026-10-11.*
