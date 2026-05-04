# py4mt Module Catalogue

**Python for Magnetotellurics** — module index generated from README files.

All modules are part of the `py4mt` / `py4mtx` package (DIAS / Volker Rath).
They are organised by functional group.

---

## Data I/O and processing

### `data_proc.py` — EDI / MT data processing

Core I/O and processing library for MT transfer functions.  All downstream
modules treat the dict returned by `load_edi` as the standard **site container**.

| Capability | Key functions |
|---|---|
| Read / write EDI (Phoenix/SPECTRA and table formats) | `load_edi`, `save_edi` |
| Derived quantities | `compute_rhophas`, `compute_pt`, `compute_zdet`, `compute_zssq` |
| D⁺ / rho-plus 1-D consistency test (Parker 1980) | `compute_rhoplus` |
| Multi-format export | `save_npz`, `save_hdf`, `save_ncd`, `save_mat` |
| DataFrame conversion (for plotting) | `dataframe_from_edi` |
| Experimental EMTF-XML support | `read_emtf_xml`, `write_emtf_xml`, `edi_to_emtf`, `emtf_to_edi` |
| Error setting / flooring | `set_errors` (`mode="fix"` or `"floor"`) |
| Error estimation from curve scatter | `estimate_errors` (spline-residual, smoothed, MAD) |
| Interpolation | `interpolate_data` |
| Rotation | `rotate_data` |
| 1-D forward modelling | `mt1dfwd`, `wait1d` |

**FT sign-convention:** `load_edi` accepts `manufacturer="phoenix"` and
automatically conjugates Z and T to the standard e⁻ⁱωᵗ convention.

**Z units:** stored in mV/km/nT throughout; `ρ_a = |Z|² × 10⁶ / (μ₀ω)`.

---

## Visualisation

### `data_viz.py` — Matplotlib helpers for MT transfer-function plots

Composable plotters working with tidy `pandas.DataFrame` from
`data_proc.dataframe_from_edi()` or directly with EDI-style dicts.

| Function | Plot type |
|----------|-----------|
| `add_rho` | Apparent resistivity (log–log) |
| `add_rhoplus` | D⁺/rho-plus test: ρ_a and upper bound with violation markers |
| `add_phase` | Impedance phase (semilog-x) |
| `add_tipper` | Tipper Re/Im (semilog-x) |
| `add_pt` | Phase tensor components (semilog-x) |
| `plot_gridx` / `plot_grid` | Generic subplot-grid helpers |

All plotters share `show_errors`, `comps`, `xlim`, `ylim`, `linestyle`,
`marker`, `legend` keyword arguments.  A shared `PLTARGS` dict is safe to
pass to every plotter without collision.

---

### `viz.py` — Matplotlib visualisation (publication-ready)

Richer figures for MT data, inversion results, and model-space displays.
All plotters accept `**pltargs` keyword arguments.

| Function | Description |
|----------|-------------|
| `plot_impedance` | Z components (Re/Im) vs period with obs/cal comparison |
| `plot_rhophas` | ρ_a and phase with model comparison and error envelopes |
| `plot_phastens` | Phase tensor components vs period |
| `plot_vtf` | Tipper (Re/Im of Tx and Ty) |
| `plot_sparsity_pattern` | Sparsity pattern of a matrix |
| `plot_plane_cross` | Vertical cross-section planes for 3-D model display |
| `make_pdf_catalog` | Merge PDFs into a single catalog (PyMuPDF) |

> For simple transfer-function plots from DataFrames, prefer `data_viz.py`.

---

### `mcmc_viz.py` — Axes-based posterior plots for anisotropic 1-D inversion

Axes-based (not figure-based) plotting helpers for the simplified anisotropic
1-D MT posterior produced by `mcmc.py`.  All functions draw into a
caller-supplied `Axes`; the calling script manages figure layout.
Matplotlib is imported lazily — safe to import on headless nodes.

**Supported parameter sets** (`param_domain` × `param_set`):

| `param_domain` | `param_set` | Panels |
|---|---|---|
| `"rho"` | `"minmax"` | log₁₀(ρ_min), log₁₀(ρ_max), strike [deg] |
| `"rho"` | `"max_anifac"` | log₁₀(ρ_max), log₁₀(a_ρ), strike [deg] |
| `"sigma"` | `"minmax"` | log₁₀(σ_min), log₁₀(σ_max), strike [deg] |
| `"sigma"` | `"max_anifac"` | log₁₀(σ_max), log₁₀(a_σ), strike [deg] |

Conductivity variables and anisotropy factors are derived automatically from
stored resistivity min/max posterior samples when not present explicitly.

| Function | Description |
|----------|-------------|
| `open_idata(nc_path)` | Load ArviZ `InferenceData` from `.nc` |
| `load_summary_npz(path)` | Load `*_pmc_summary.npz` into a plain dict |
| `plot_paramset_threepanel(axs, *, summary, idata, param_domain, param_set, qpairs, ...)` | Three-panel step profiles with shaded quantile bands; draws from `idata` samples (supports multiple `qpairs`) or falls back to summary qlo/med/qhi arrays |
| `plot_vertical_profile(ax, *, h_m, values, ...)` | Single step profile |
| `plot_vertical_envelope(ax, *, h_m, qlo, med, qhi, ...)` | Median + shaded (qlo, qhi) band from pre-computed quantiles |
| `plot_vertical_bands_from_samples(ax, *, h_m, samples, qpairs, ...)` | Median + one or more shaded bands from posterior samples |
| `depth_edges_from_h(h_m, ...)` | Thickness array → depth-edge array for step plots |

---

## Forward models

### `aniso.py` — Anisotropic 1-D MT forward model

1-D layered MT forward model with horizontal electrical anisotropy (strike
angle per layer).  Returns impedance tensor Z(ω) and optional sensitivities.

Two parameterisations:

| API | Parameters | Users |
|-----|-----------|-------|
| `aniso1d_impedance_sens` | `rop` (principal resistivities) + Euler angles | Public / legacy |
| `aniso1d_impedance_sens_simple` | `(rho_min, rho_max, strike_deg)` per layer | `mcmc.py`, `inverse.py`, `transdim.py` |

Sensitivities include ∂Z/∂ρ, ∂Z/∂angles, and ∂Z/∂h for all parameters.

---

### `mt1d_aniso` — Compiled Fortran 1-D anisotropic kernel

`mt1d_aniso.cpython-311-x86_64-linux-gnu.so` — f2py-compiled extension
built from `aniso_f2py/mt1d_aniso.f90`.  Provides the low-level recursive
impedance kernel called by `aniso.py`.

Source, build script, and test results are in the `aniso_f2py/` subdirectory.

---

## Inversion

### `mcmc.py` — PyMC utilities for Bayesian anisotropic 1-D inversion

Core library for Bayesian inversion of MT data via a simplified anisotropic
1-D earth model.  Imported by `mt_aniso1d_sampler.py` and `mt_aniso1d_plot.py`.

| Capability | Functions |
|---|---|
| Model I/O and normalisation | `normalize_model`, `model_from_direct`, `save_model_npz`, `load_model_npz` |
| Site loading | `load_site`, `ensure_phase_tensor` |
| Prior construction | `build_gaussian_cov`, `build_pymc_model` (`prior_kind` = `"default"`, `"uniform"`, `"gaussian"`) |
| Correlation kernels | `exponential_corr`, `gaussian_corr`, `matern_corr`, `block_corr_matrix` |
| Sampling | `sample_pymc`, `ParamSpec` |
| Post-processing | `save_idata`, `build_summary_npz`, `save_summary_npz` |
| Data packing | `pack_Z_vector`, `pack_Z_sigma`, `pack_P_vector`, `pack_P_sigma` |

Two likelihood modes: **black-box** (`pytensor.wrap_py`, robust, DEMetropolisZ /
Metropolis) and **gradient-enabled** (custom `Op` with analytic VJPs, supports NUTS).

Matérn correlation kernels (ν = ½, 3⁄2, 5⁄2, ∞) for inter-layer covariance in
the Gaussian prior.

---

### `inverse.py` — Numerical inversion utilities + deterministic 1-D inversion

Merged module consolidating general numerical routines and 1-D MT inversion helpers.

**Part A — General numerical routines:**

| Function | Description |
|----------|-------------|
| `splitbreg` | Split Bregman / TV regularisation |
| `soft_thresh` | Soft-thresholding |
| `calc_covar_simple` | Empirical cross-covariance |
| `calc_covar_nice` | NICE shrinkage (Vishny et al. 2024) |
| `msqrt_sparse` | SPD matrix square root (Cholesky / EIGS / SPLU) |
| `rsvd` | Randomised SVD with subspace iteration |
| `make_spline`, `estimate_variance`, `bootstrap_confidence_band` | Spline utilities |

**Part B — Deterministic 1-D MT inversion:**

| Function | Description |
|----------|-------------|
| `load_site` | Load site from `.edi` or `.npz` |
| `normalize_model` | Validate and normalise a model dict |
| `ParamSpec` | Bounds and masks; `param_domain` ∈ {`rho`, `sigma`}, `param_set` ∈ {`minmax`, `max_anifac`} |
| `invert_site` | Gauss-Newton inversion with Tikhonov or TSVD |
| `ensure_phase_tensor` | Compute P = Re(Z)⁻¹ Im(Z) with optional bootstrap error |

---

### `transdim.py` — Transdimensional (rjMCMC) 1-D inversion

Core library for reversible-jump MCMC where the number of layers *k* is itself
a free parameter (Green 1995; Bodin & Sambridge 2009; Malinverno 2002).
**No matplotlib dependency** — safe for headless / HPC nodes.

| Class / function | Description |
|---|---|
| `LayeredModel` | 1-D earth: interfaces, log10(ρ), optional aniso_ratios and strikes |
| `Prior` | Uniform bounds on k, depth, log10(ρ), aniso ratio, strike |
| `RjMCMCConfig` | Sampler tuning: iterations, burn-in, thin, proposal weights and σ |
| `mt_forward_1d_isotropic` / `_anisotropic` | ρ_a-level forward models |
| `mt_forward_1d_isotropic_impedance` / `_anisotropic_impedance` | Full Z-tensor forward models |
| `compute_phase_tensor` | PT = Re(Z)⁻¹ Im(Z) per frequency |
| `log_likelihood` | Gaussian log-likelihood in log10(ρ_a) space |
| `propose_birth` / `propose_death` / `propose_move` / `propose_change` | RJMCMC proposals |
| `run_rjmcmc` | Single chain |
| `run_parallel_rjmcmc` | N chains via joblib; Gelman–Rubin R-hat |
| `compute_posterior_profile` / `_rhomin_profile` / `_aniso_profile` / `_histogram` | Posterior analysis |
| `compute_changepoint_frequency` | Interface-placement frequency vs depth |
| `save_results_npz` / `load_results_npz` | Posterior I/O |

---

### `transdim_viz.py` — rjMCMC results visualisation

All figure-generation routines for `transdim.py` posteriors.
Separated to keep the sampler matplotlib-free.

| Function | Description |
|----------|-------------|
| `plot_results` | Generic 4- or 5-panel diagnostic |
| `plot_qc` | QC summary: ρ_a/phase **or** Re/Im(Z) data fit, optional phase-tensor row, misfit trace, k histogram |
| `plot_posterior_model` | 2-D histograms (ρ_max, ρ_min, strike) + change-point frequency; linear or log depth scale |
| `plot_resistivity_profile` | Posterior ρ(z) with credible interval |
| `plot_dimension_histogram` | Posterior k histogram |
| `plot_data_fit` | Observed vs posterior-predicted ρ_a |
| `plot_chain_traces` | Per-chain log-likelihood traces |
| `plot_aniso_profile` | Posterior anisotropy ratio vs depth |
| `plot_strike_profile` | Posterior strike angle vs depth |

---

### `plotrjmcmc.py` — Legacy rjMCMC plotting (Geoscience Australia)

Python translation of Ross Brodie's original MATLAB plotting routines for
rjMCMC inversion results (Geoscience Australia), adapted by Volker Rath (2019).

> **Status: legacy.** For new work, prefer `transdim_viz.py`.
> `plotrjmcmc.py` is retained for reading GA-format rjMCMC output files.

Defines a `Results` class that reads a GA-format rjMCMC output directory and
generates multi-panel diagnostic figures.

```python
Results(path, outputFileName,
        plotSizeInches='11x8', maxDepth=2000,
        resLims=[1., 100000.], zLog=False, colormap='gray_r')
```

Plot flags on the instance: `_plotLowestMisfitModels`, `_plotMisfits`
(default `True`), `_plotSynthetic`.

---

## FEMTIC interface

### `femtic.py` — FEMTIC-specific I/O, model conversion, and format utilities

FEMTIC file I/O, model read/write, mesh parsing, and format conversion.
Matrix/roughness tools and ensemble generation are delegated to `ensembles.py`
(re-exported here for backward compatibility).

| Capability | Key functions |
|---|---|
| Data I/O (`observe.dat`) | `read_observe_dat`, `write_observe_dat`, `modify_data` |
| EDI → observe.dat | `edi_list_to_observe_dat`, `observe_to_site_viz_list` |
| Resistivity-block model workflow | `read_model_to_npz`, `modify_model_npz`, `write_model_from_npz` |
| Distortion file I/O | `read_distortion_file` |
| Mesh I/O | tetrahedral `mesh.dat` parsing |
| NPZ ↔ VTK / VTU | `unstructured_grid_from_femtic` (via PyVista) |
| NPZ ↔ NetCDF / HDF5 | CF-compliant and HDF5 export/import |
| CLI | `femtic-to-npz`, `npz-to-vtk`, `npz-to-femtic`, `edi-to-observe` |

**z-convention:** FEMTIC uses z positive-downward; `edi_list_to_observe_dat`
applies `z_femtic = -elev_m` and warns on positive site-header z values.

**Unit conversion:** Z is stored as SI Ω in `observe.dat`;
`Z_SI = Z_MT × μ₀ × 10³` applied automatically on EDI → observe conversion.

---

### `femtic_viz.py` — FEMTIC model visualisation

Visualisation for FEMTIC resistivity models operating directly on
`mesh.dat` + `resistivity_block_iterX.dat` (no intermediate NPZ required).

| Capability | Functions |
|---|---|
| PyVista grid / VTU export | `unstructured_grid_from_femtic`, CLI `export-vtu` |
| Map slices (Matplotlib) | `map_slice_from_cells`; modes: `tri`, `scatter`, `grid` |
| Curtain slices (Matplotlib) | `curtain_from_cells`; modes: `tri`, `scatter`, `grid` |
| IDW curtain on regular s–z grid | `curtain_grid_idw`, `plot_curtain_matplotlib` |
| PyVista surface sampling | `build_curtain_surface`, `build_map_surface`, `sample_grid_on_surface` |
| RTO/GST ensemble data plots | `plot_data_ensemble` |
| RTO/GST ensemble model plots | `plot_model_ensemble` |

`plot_data_ensemble` supports multi-panel layouts (`what` list), independent
error-style control per curve (`error_style_orig` / `error_style_pert`),
component markers (`comp_markers`), and per-panel component selection.

`plot_model_ensemble` supports map and curtain slices, global colour limits,
mesh-edge overlay, and axis limits per slice type.

Ocean cells rendered in flat light-grey by default; controlled via `ocean_color`.

---

### `ensembles.py` — Roughness / precision matrix tools and ensemble generation

Single source of truth for all matrix, roughness, and sampling utilities shared
across the FEMTIC ecosystem.  `femtic.py` imports from here.

| Group | Key functions |
|---|---|
| Directory / ensemble setup | `generate_directories`, `generate_data_ensemble` |
| RTO model ensemble | `generate_rto_model_ensemble` (roughness-matrix prior draw) |
| GST model ensemble | `generate_gst_model_ensemble` (pilot-point Ordinary Kriging, gstools) |
| Roughness / precision matrix | `get_roughness`, `make_prior_cov`, `matrix_reduce`, `check_sparse_matrix` |
| ILU persistence | `save_spilu`, `load_spilu` |
| Precision solvers | `build_rtr_operator`, `make_rtr_preconditioner`, `make_precision_solver`, `make_sparse_cholesky_precision_solver` |
| Gaussian sampling | `sample_rtr_full_rank`, `sample_rtr_low_rank`, `sample_prior_from_roughness` |
| EOF / PCA | `compute_eofs`, `eof_reconstruct`, `eof_sample`, `fit_eof_model`, `sample_physical_ensemble` |
| Weighted KL decomposition | `compute_weighted_kl` |
| PCE surrogate | `fit_kl_pce_model`, `evaluate_kl_pce_surrogate`, `KLPCEModel` |

**Automatic λ selection:** diagonal shift `λ = α · median(diag(RᵀR))`
via `lam_mode="scaled_median_diag"` suppresses nullspace instability in `RᵀR`.

**Recommended defaults (FEMTIC):** `algo="low rank"`, `n_eig=128–256`,
`n_power_iter=3–4`, `lam_alpha=1e-4–1e-3`, `precond="ilu"`, `rtol=1e-2`.

---

## ModEM interface

### `modem.py` — ModEM I/O, processing, and model compression

File I/O, smoothing operators, and six model-compression methods for the
ModEM 3-D MT inversion package.

**File I/O:**

| Capability | Functions |
|---|---|
| Jacobian read (Fortran unformatted) | `read_jac`, `read_data_jac` |
| Jacobian export | `write_jac_ncd` |
| Data file read/write | `read_data`, `write_data`, `write_data_ncd` |
| Model read/write (`.rho`) | `read_mod`, `write_mod`, `write_mod_npz`, `write_mod_ncd` |
| UBC and RLM conversion | `write_ubc`, `read_ubc`, `write_rlm` |

**Model compression:**

| Method | `METHOD` | Best for |
|--------|----------|---------|
| 3-D DCT-II, radial truncation | `"dct"` | Smooth, stationary models |
| 3-D DCT-II, separable (box) truncation | `"dct_sep"` | Anisotropic compression needs |
| 3-D DWT (wavelet) | `"wavelet"` | Spatially localised anomalies |
| Legendre-z × DCT-xy | `"legdct"` | Models with strong depth gradients |
| B-spline-z × DCT-xy | `"bspdct"` | Real models with expanding cell size |
| Karhunen–Loève / PCA | `"kl"` | Ensemble-based uncertainty analysis |

Each method provides: forward transform, inverse transform, one-call
compress-and-reconstruct wrapper, and a truncation sweep for parameter selection.

---

### `jac_proc.py` — Jacobian / sensitivity post-processing

| Capability | Functions |
|---|---|
| Sensitivity maps | `calc_sensitivity` (types: `raw`, `cov`, `euc`, `cum`) |
| Transform and normalise | `transform_sensitivity`, `normalize_jac` |
| Sparsification | `sparsify_jac` |
| Streaming statistics | `update_avg` |
| Low-rank approximation | `rsvd` |

All functions assume Jacobian is pre-scaled as `J_scaled = C⁻¹/² J`.

---

## General utilities

### `util.py` — General-purpose utilities

| Section | Key functions |
|---|---|
| HDF5 workspace persistence | `save_workspace_hdf5`, `load_workspace_hdf5` |
| Module introspection | `list_module_callables`, `runtime_env`, `running_in_notebook` |
| Coordinate projections (pyproj) | WGS84 ↔ UTM / ITM / Gauss-Krüger; geoid correction |
| File and string utilities | `get_filelist`, `strreplace`, `symlink`, `filecopy`, `make_pdf_catalog` |
| Archive unpacking | `unpack_compressed` (zip, tar, gz, bz2, xz) |
| Archive packing | `pack_compressed` (zip, tar, tgz, tbz2, txz) |
| Script queue runner | `run_queue` (glob expansion, timestamped log, strict/lenient mode) |
| Grid generation | `gen_grid_latlon`, `gen_grid_utm`, `gen_searchgrid` |
| Geometry | `point_inside_polygon`, `choose_data_poly`, `proj_to_line` |
| Numerical | `KLD`, `calc_lc_corner`, `calc_rms`, `dctn`/`idctn`, `fractrans` |
| Rotation matrices | `rot_x`, `rot_y`, `rot_z`, `rot_full` |
| Petrophysical models | `brine_conductivity_sen_goode`, `archie`, `simandoux`, `dual_porosity`, `rgpz`, `kozeny_carman`, `hashin_shtrikman_two_phase`, `hashin_shtrikman_n_phase` |

---

### `topo.py` — SRTM DEM download, mosaicking, and conversion

Download SRTM elevation tiles (USGS SRTM v2.1, 3 arc-second ≈ 90 m),
mosaic them into a single GeoTIFF, rotate, and export to XYZ.

| Function | Description |
|----------|-------------|
| `tiles_for_bbox(lat_min, lat_max, lon_min, lon_max)` | Enumerate SRTM tile names for a bounding box |
| `download_srtm_tile(tile, continent, out_dir)` | Download and unzip a single SRTM `.hgt.zip` tile |
| `mosaic_hgt(hgt_paths, out_path)` | Mosaic `.hgt` files into a GeoTIFF (`rasterio.merge`) |
| `rotate_raster(in_path, out_path, angle_deg)` | Affine-transform rotation (note: not a full reprojection) |
| `geotiff_to_xyz(in_path, out_path, as_dataframe, step)` | Export GeoTIFF to XYZ array or DataFrame |
| `process_srtm(lat_min, lat_max, lon_min, lon_max, angle_deg, ...)` | One-call pipeline: enumerate → download → mosaic → rotate |

Dependencies: `numpy`, `pandas`, `rasterio`, `requests`.

---

### `cluster.py` — K-means clustering with missing values

#### `kmeans_missing(X, n_clusters=3, max_iter=10)`

K-means on data with NaN / non-finite values via EM-style imputation.

**Algorithm:** missing entries initialised to column means → fit K-means →
impute missing from cluster centroids → repeat until label stability or
`max_iter` reached.  Iteration 0 uses `KMeans` (parallel inits); subsequent
iterations use `MiniBatchKMeans` seeded from the previous centroids.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `X` | — | `(n_samples, n_features)` array; NaN / ±inf treated as missing |
| `n_clusters` | `3` | Number of clusters |
| `max_iter` | `10` | Maximum EM iterations |

Returns `(labels, centroids, X_hat)` — cluster assignments, centroids, and
X with missing values filled in.

---

### `version.py` — Package version string

`versionstrg()` — returns `(version_string, release_date)` for the Py4MTX
package.  Used by `print_title()` in `util.py` and imported by scripts for
header output.

---

## Module dependency graph

```
data_proc  ──────────────────────────────────────┐
aniso  ──────────────────────────────────────┐   │
                                             ▼   ▼
                                           mcmc   transdim ──► transdim_viz
                                             │
                                           inverse
                                             │
                                           mcmc_viz

femtic ──► ensembles
femtic_viz ──► femtic

data_proc ──► data_viz
modem ──► jac_proc (via modem scripts)

util  ◄── (imported by all scripts and most modules)
```

---

## Summary by count

| Group | Modules |
|---|---|
| Data I/O and processing | 1 |
| Visualisation | 4 |
| Forward models | 2 |
| Inversion | 4 |
| FEMTIC interface | 3 |
| ModEM interface | 2 |
| General utilities | 4 |
| **Total** | **20** |

---

*Generated 2026-05-04 from README files in `modules.zip`; missing READMEs synthesised from source.*
