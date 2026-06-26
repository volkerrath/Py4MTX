# General MT Scripts — Catalogue

**py4mt / py4mtx** — scripts applicable across inversion codes or for
data pre/post-processing.  All scripts live in `scripts/general/`.

---

## Data processing

| Script | Purpose |
|--------|---------|
| `mt_data_processor.py` | Batch-process `.edi` files: compute phase tensor, impedance invariants, apparent resistivity/phase; set or estimate errors; optionally interpolate or rotate; export to EDI, NPZ, HDF, or MAT. |
| `mt_get_averages.py` | Compute scalar impedance averages (Z_ssq, Z_det, Z_avg = √(Z_xy·Z_yx)) across a site collection from NPZ files; form frequency-wise and global log-averages with inter-site spread (std or bootstrap); write two-panel ρ_a / phase plots per quantity and NPZ per quantity. Supports `FREQ_RANGE` masking, shaded error bands, and site-overlay legend. |
| `mt_make_sitelist.py` | Read all `.edi` files in a directory and write a site list (name, lat, lon, elev) in WALDIM, FEMTIC, or general CSV format. |
| `mt_calc_ptdim.py` | Use phase-tensor analysis to classify each MT station / frequency as 1-D, 2-D, or 3-D; writes per-site dimensionality tables and a combined summary. |
| `mt_kmz_waldim.py` | Export WALDIM dimensionality classification results as a KMZ file for Google Earth, with colour-coded site markers (3-class or 10-class scheme). |
| `mt_archive_run.py` | Clean and optionally archive iteration directories: keep lowest / highest N iterations, protect specified files (by token or suffix), delete the rest (dry-run by default), compress to `.zip` or `.tgz`. |

---

## Plotting

| Script | Purpose |
|--------|---------|
| `mt_plot_data.py` | Generate per-station 3×2 diagnostic subplot figures (apparent resistivity, phase, tipper, phase tensor) from `.npz`, `.edi`, or `.dat` files; optionally assembles a PDF catalogue. |
| `mt_plot_strikes.py` | Generate strike-direction rose diagrams (aggregate, per-decade, per-station) from MT impedance data using mtpy. |

---

## 1-D inversion — anisotropic (deterministic + Bayesian)

| Script | Purpose |
|--------|---------|
| `mt_aniso1d_forward.py` | Compute full impedance tensor, apparent resistivity/phase, and phase tensor for a layered anisotropic conductivity model; exports to text files and plots. |
| `mt_aniso1d_inversion.py` | Batch deterministic Gauss-Newton inversion of anisotropic 1-D MT data with Tikhonov or TSVD regularisation and automatic α selection (GCV, L-curve, ABIC). |
| `mt_aniso1d_sampler.py` | PyMC driver for Bayesian (MCMC) anisotropic 1-D MT inversion; supports Gaussian and Matérn covariance priors. Delegates to `mcmc.py`. |
| `mt_aniso1d_plot.py` | Plot posterior parameter distributions from anisotropic 1-D sampler results as three-panel depth profiles (ρ_min/ρ_max/strike or equivalent) with uncertainty bands. |

---

## 1-D inversion — transdimensional (rjMCMC)

| Script | Purpose |
|--------|---------|
| `mt_transdim_iso1d.py` | rjMCMC driver for isotropic 1-D MT inversion; likelihood modes `"Zdet"` and `"rhoa"`; number of layers is a free parameter. Delegates to `transdim.py` and `transdim_viz.py`. |
| `mt_transdim_aniso1d.py` | rjMCMC driver for anisotropic 1-D MT inversion; likelihood modes `"Z_comps"` and `"rhoa"`; multiple independent chains via joblib. Delegates to `transdim.py` and `transdim_viz.py`. |
| `mt_plot_rjmcmc.py` | Visualise posterior resistivity-depth distributions from the rjmcmc-MT transdimensional sampler; optionally adds site coordinates from EDI files and assembles a PDF catalogue. |

---

## Summary

| Group | Scripts |
|-------|---------|
| Data processing | 6 |
| Plotting | 2 |
| Anisotropic 1-D (deterministic + Bayesian) | 4 |
| Transdimensional 1-D (rjMCMC) | 3 |
| **Total** | **15** |

---

*Generated 2026-06-26 from README files in `scripts/general/`.*
