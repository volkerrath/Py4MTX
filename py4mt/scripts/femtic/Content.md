# FEMTIC Scripts — Catalogue

**py4mt / py4mtx** — FEMTIC (3-D finite-element MT inversion, Yoshiya Usui)
workflow scripts.  All scripts live in `scripts/femtic/`.

---

## Model visualisation

| Script | Purpose |
|--------|---------|
| `femtic_mod_plot_slice.py` | 2-D slice panels (map, N-S curtain, E-W curtain, arbitrary strike/dip plane) with optional embedded borehole columns, site overlays, compass corner labels, and nRMS annotation from `femtic.cnv`. |
| `femtic_mod_plot_bh.py` | Standalone 1-D ρ(z) borehole resistivity logs; point-in-element sampling from a tetrahedral mesh. |
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
| `femtic_lcurve_plot.py` | Harvest final-iteration α / roughness / nRMS from multiple directories and plot the L-curve; annotates each point with its α value. `DISTORTION=None` auto-detects column layout from cnv line length. |

---

## Uncertainty quantification — Randomize-Then-Optimize (RTO)

| Script | Purpose |
|--------|---------|
| `femtic_rto_rough.py` | Read FEMTIC `roughening_matrix.out` and save the sparse roughness matrix R (or Q = RᵀR) as `.npz`. First step in the RTO pipeline. |
| `femtic_rto_prior.py` | Build a prior covariance proxy M ≈ α²(R + εI)⁻¹(R + εI)⁻ᵀ from the roughness matrix. Optionally sparsify and save as `.npz`. |
| `femtic_rto_prep.py` | Generate the full RTO ensemble: perturb `observe.dat` with Gaussian noise and draw prior model perturbations from N(0,(RᵀR)⁻¹) via randomized SVD. Creates one directory per member. |
| `femtic_rto_post.py` | Postprocess a converged RTO ensemble: collect models, compute cell-wise statistics (mean, variance, median, MAD, percentiles) and the empirical covariance matrix, save to `RTO_results.npz`. |

**RTO workflow:**
```
femtic_rto_rough.py  →  R_coo.npz
femtic_rto_prior.py  →  invRTR_*.npz   (optional)
femtic_rto_prep.py   →  ensemble directories
                         (run FEMTIC on each member)
femtic_rto_post.py   →  RTO_results.npz
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
| `femtic_ens_post.py` | Algorithm-agnostic postprocessing of a converged ensemble: nRMS filter, log₁₀(ρ) statistics (mean, variance, median, MAD, percentiles), empirical covariance, optional QC and statistics slice figures. Supersedes `femtic_rto_post.py`. |
| `femtic_ens_plot.py` | Joint multi-row slice figure for a set of ensemble run directories, with optional statistical summary rows (mean, std, median) and optional borehole columns. |
| `femtic_ens_decomp.py` | Filter ensemble by nRMS threshold then perform PCA or ICA decomposition (scikit-learn); prints explained-variance ratios. |
| `femtic_ens_eof.py` | Compute EOFs via SVD on a demeaned ensemble matrix; reconstruct from truncated or individual EOF modes. |
| `femtic_ens_from_covar.py` | Draw new model samples from a posterior covariance via Cholesky decomposition: `m_new = m_ref + L·z`, z ~ N(0,I). |

---

## Uncertainty quantification — Jackknife

| Script | Purpose |
|--------|---------|
| `femtic_jcn_prep.py` | Create N member directories from template files and generate reduced `observe.dat` files (leave-one-site-out or random subsets). |

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

## Summary

| Group | Scripts |
|-------|---------|
| Model visualisation | 3 |
| Model manipulation | 2 |
| Diagnostics and L-curve | 2 |
| RTO uncertainty | 4 |
| GST uncertainty | 1 |
| Ensemble analysis | 5 |
| Jackknife | 1 |
| Nullspace shuttle | 1 |
| Utilities | 3 |
| **Total** | **22** |

---

*Generated 2026-06-26 from README files in `scripts/femtic/`.*
