# FEMTIC Scripts — Catalogue

**py4mt / py4mtx** — FEMTIC (3-D finite-element MT inversion, Yoshiya Usui)
workflow scripts.  All scripts live in `scripts/femtic/`.

---

## Model visualisation

| Script | Purpose |
|--------|---------|
| `femtic_mod_plot_slice.py` | 2-D slice panels (map, N-S curtain, E-W curtain, arbitrary strike/dip plane) with optional embedded borehole columns, site overlays, compass corner labels, and nRMS annotation from `femtic.cnv`. |
| `femtic_mod_plot_bh.py` | Standalone 1-D ρ(z) borehole resistivity logs; point-in-element sampling from a tetrahedral mesh (mesh read with `fem.read_femtic_mesh`). For ensembles see `MOD_BH` in `femtic_ens_post.py`. |
| `femtic_plot_nrms.py` | Map of per-site nRMS misfit as colour-coded glyphs, optionally over an SRTM hillshade. The best iteration comes from `femtic.cnv`; per-site residuals from `results_iter<N>.h5` via `fem.read_results_data`. |
| `femtic_plot_distort.py` | Map of the static galvanic distortion matrices as ellipses (optional Groom–Bailey decomposition panels, optional SRTM hillshade). |
| `femtic_mod_plot_3d.py` | PyVista 3-D rendering and VTK/VTU export; axis-aligned and oblique slice planes, iso-surfaces, spatial clipping. |

---

## Model manipulation

| Script | Purpose |
|--------|---------|
| `femtic_mod_edit.py` | Apply arithmetic operations to a FEMTIC resistivity block in log₁₀ space (null, fill, clip, shift, smooth, ellipsoid/brick insertion, mean, median, standardise). Air, ocean, and fixed cells are never modified. Optional slice figure output. |
| `femtic_mod_math.py` | Generate ensemble-derived synthetic members: element-wise log₁₀(ρ) mean and mesh-adaptive smoothed median. Accepts an nRMS filter and an optional subset list. |

---

## Diagnostics and L-curve

| Script | Purpose |
|--------|---------|
| `femtic_convergence_plot.py` | Read `femtic.cnv` from one or more inversion directories and plot misfit, nRMS, or roughness vs. iteration number. |
| `femtic_plot_conv.py` | Plot any `femtic.cnv` field(s) — by name, via `fem.read_cnv(columnar=True)` — vs. iteration/row index, overlaid across one or more run directories; no version/column-count switch. |
| `femtic_data_misfit.py` | Data-misfit diagnostics from `results_iter<N>.h5`: observed-vs-calculated crossplots, normalised-residual histograms and normal Q-Q plots in frequency bands, optional per-site response curves; single run or ensemble. Uses `fem.read_results_data`. |
| `femtic_lcurve_plot.py` | Harvest final-iteration α / roughness / nRMS from multiple directories and plot the L-curve; annotates each point with its α value. Column positions read from each `femtic.cnv`'s own header row via `fem.read_cnv()`. |

---

## Uncertainty quantification — Randomize-Then-Optimize (RTO)

| Script | Purpose |
|--------|---------|
| `femtic_rto_rough.py` | Read FEMTIC `roughening_matrix.out` (`ens.get_roughness`) and save the sparse roughness matrix R (or Q = RᵀR) as `.npz`. First step in the RTO pipeline. |
| `femtic_rto_prior.py` | Build a prior covariance proxy M ≈ α²(R + εI)⁻¹(R + εI)⁻ᵀ from the roughness matrix. Optionally sparsify and save as `.npz`. |
| `femtic_rto_prep.py` | Generate the full RTO ensemble: perturb `observe.dat` with Gaussian noise and draw prior model perturbations from N(0,(RᵀR)⁻¹) via randomized SVD. Creates one directory per member. |

**RTO workflow:**
```
femtic_rto_rough.py  →  R_coo.npz
femtic_rto_prior.py  →  invRTR_*.npz   (optional)
femtic_rto_prep.py   →  ensemble directories
                         (run FEMTIC on each member)
femtic_ens_post.py   →  ensemble statistics (see below)
```

---

## Uncertainty quantification — Geostatistical (GST)

| Script | Purpose |
|--------|---------|
| `femtic_gst_prep.py` | Generate a geostatistical ensemble: perturb `observe.dat` and draw initial resistivity models via Ordinary Kriging from a sparse pilot-point cloud (gstools). Supports `"random"`, `"fixed"`, and `"mixed"` pilot-point strategies. |

---

## Uncertainty quantification — Ensemble analysis

| Script | Purpose |
|--------|---------|
| `femtic_ens_post.py` | Algorithm-agnostic postprocessing of a converged ensemble: nRMS filter, log₁₀(ρ) statistics (mean, variance, median, MAD, percentiles), empirical covariance, optional QC and statistics slice figures, and optional borehole profiles (`MOD_BH`: ensemble statistics and/or all members at chosen or random positions, or at all sites). Replaces the former `femtic_rto_post.py`. |
| `femtic_ens_plot.py` | Joint multi-row slice figure for a set of ensemble run directories, with optional statistical summary rows (mean, std, median) and optional borehole columns. |
| `femtic_ens_repair.py` | Convergence diagnostic (nRMS bar chart / histogram, per status) for every directory of an ensemble; for non-converged members writes a restart directory `<dir>_restart` whose starting model is the mean of two random converged members. |
| `femtic_ens_decomp.py` | Filter ensemble by nRMS threshold then perform PCA or ICA decomposition (scikit-learn); prints explained-variance ratios. |
| `femtic_ens_eof.py` | Compute EOFs via SVD on a demeaned ensemble matrix; reconstruct from truncated or individual EOF modes. |
| `femtic_ens_from_covar.py` | Draw new model samples from a posterior covariance via Cholesky decomposition: `m_new = m_ref + L·z`, z ~ N(0,I). |

---

## Uncertainty quantification — Jackknife

| Script | Purpose |
|--------|---------|
| `femtic_jcn_prep.py` | Create N member directories (`ens.generate_directories`) and write reduced `observe.dat` files into them (leave-one-site-out, or random site subsets) using `fem.read_observe_dat` / `fem.write_observe_dat`. |

---

## Nullspace shuttle

| Script | Purpose |
|--------|---------|
| `femtic_nss.py` | Read final model and data from an HDF5 inversion archive, compute the data-weighted Jacobian, decompose via randomised SVD, generate a model perturbation (geostatistical or random), and project it onto the null space of the scaled Jacobian so predicted data are unchanged. Writes a new FEMTIC resistivity block. |

---

## Utilities

| Script | Purpose |
|--------|---------|
| `femtic_archive_run.py` | Clean and optionally archive FEMTIC iteration directories: keep lowest / highest N iterations, protect specified files (by token or suffix), delete the rest (dry-run by default), compress to `.zip` or `.tgz`. |
| `femtic_summarize_model_cells.py` | Print element composition (air, ocean, other-fixed, free) for one or more `resistivity_block_iterXX.dat` files; optional `--ocean yes/no` override. Falls back to a self-contained parser if `femtic.py` is unavailable. |
| `femtic_summarize_observe_dat.py` | Print data content (sites, frequencies, data-vector size, observation types: MT/VTF/PT) for one or more `observe.dat` files. Falls back to a minimal parser if `femtic.py` is unavailable. |

---

## FEMTIC I/O conventions used by these scripts

| Task | Call | Notes |
|------|------|-------|
| Read mesh | `nodes, conn = fem.read_femtic_mesh(path)` | Only reader; lives in `femtic.py`. `mesh.dat` is x = northing, y = easting, z positive down. Optional `swap_xy=True` (or module default `fem.MESH_SWAP_XY`) swaps x/y for old conventions. `femtic_viz` wraps it in `FemticMesh`. |
| Read resistivity block | `fem.read_model(...)` | Returns the free-region log₁₀(ρ) vector only. |
| Write resistivity block | `fem.insert_model(template=..., model=..., model_file=..., out=...)` | Writes the file; returns nothing. |
| Region structure / centroids | `fem._read_resistivity_block_struct`, `fem.build_region_geometry` | |
| Read / write `observe.dat` | `fem.read_observe_dat`, `fem.write_observe_dat` | Block header counts must be updated when sites are dropped. |
| Read `results_iter<N>.h5` | `fem.read_results_data`, `read_results_model`, `read_results_hdf5` | Replaces the former `femtic_hdf5_readers`. |
| Directories, roughness, prior, data ensembles | `ens.generate_directories`, `ens.get_roughness`, `ens.make_prior_cov`, `ens.check_sparse_matrix`, `ens.generate_data_ensemble` | Live in `ensembles.py`, not `femtic.py`. |

---

## Summary

| Group | Scripts |
|-------|---------|
| Model visualisation | 5 |
| Model manipulation | 2 |
| Diagnostics and L-curve | 4 |
| RTO uncertainty | 3 |
| GST uncertainty | 1 |
| Ensemble analysis | 6 |
| Jackknife | 1 |
| Nullspace shuttle | 1 |
| Utilities | 3 |
| **Total** | **26** |

---

*Generated 2026-06-26 from README files in `scripts/femtic/`; updated 2026-10-11 (new scripts, FEMTIC I/O table, removed `femtic_rto_post.py`).*
