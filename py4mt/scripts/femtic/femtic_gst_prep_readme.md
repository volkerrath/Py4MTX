# README\_femtic\_gst\_prep.md

Preparation script for the **Geostatistical (GST)** ensemble algorithm applied
to FEMTIC magnetotelluric inversion.

## Purpose

`femtic_gst_prep.py` creates a complete ensemble of perturbed data files and
geostatistically generated initial resistivity models that are ready to be
submitted as independent FEMTIC inversion runs.  After all ensemble members
have converged, the results can be collected and analysed for uncertainty
quantification (see references below).

The key difference from the RTO approach (`femtic_rto_prep.py`) is in the
**model perturbation**: instead of sampling from a roughness-matrix prior, each
member starts the inversion from a structurally distinct initial model built by
Kriging random resistivity values from a sparse pilot-point cloud to every mesh
cell.  The data perturbation is unchanged.

## Algorithm outline

For each ensemble member *i = 1 … N*:

1. **Perturbed data** — draw `d̃ ~ N(d, C_d)` by adding scaled Gaussian noise
   to the observed data (`observe.dat`).  Identical to the RTO step.
2. **Pilot-point values** — draw log₁₀(ρ) values at a set of pilot points
   uniformly from `[log_rho_min, log_rho_max]`.
3. **Kriging** — interpolate and extrapolate the pilot-point values to all
   FEMTIC mesh cell centres via Ordinary Kriging (gstools).
4. **Clamp** — clip the Kriged field to `[log_rho_min, log_rho_max]`.
5. **Write** — save the field as `resistivity_block_iter0.dat` and/or
   `referencemodel.dat` in the member directory.
6. **Solve** — the deterministic FEMTIC inversion is run on each
   `(d̃, m₀⁽ⁱ⁾)` pair (performed externally, not by this script).

## Workflow

```text
femtic_gst_prep.py   →   ensemble directories with
                          perturbed observe.dat  &
                          resistivity_block_iter0.dat
                          (+ referencemodel.dat if MOD_OUTPUT_TARGET = "both")
                          + diagnostic plots
                                 ↓
                          (run FEMTIC on each member)
                                 ↓
femtic_gst_post.py   →   collect results & statistics
```

No upstream roughness-matrix computation is required (contrast with
`femtic_rto_rough.py` → `femtic_rto_prep.py`).

## Configuration

All settings are at the top of the script.

### Base setup

| Variable         | Description                                                         |
|------------------|---------------------------------------------------------------------|
| `N_SAMPLES`      | Number of ensemble members to generate.                             |
| `ENSEMBLE_DIR`   | Root directory for the ensemble.                                    |
| `TEMPLATES`      | Directory containing template FEMTIC input files.                   |
| `COPY_LIST`      | Template files copied into each member directory.                   |
| `LINK_LIST`      | Template files symlinked into each member directory.                |
| `ENSEMBLE_NAME`  | Prefix for member directories (e.g. `ubinas_gst_`).                |
| `FROM_TO`        | `None` for 0 … N−1, or `(start, stop)` to continue / patch members.|

### Model perturbation

#### Pilot points

| Variable         | Description                                                                          |
|------------------|--------------------------------------------------------------------------------------|
| `PERTURB_MOD`    | Enable / disable model perturbation.                                                 |
| `MOD_PP_MODE`    | Pilot-point placement strategy: `"random"`, `"fixed"`, or `"mixed"`.                |
| `MOD_N_PP`       | Number of randomly drawn pilot points per member (used when `MOD_PP_MODE` ≠ `"fixed"`). |
| `MOD_PP_BBOX`    | Bounding box `[x_min, x_max, y_min, y_max, z_min, z_max]` (m) for random placement. |
| `MOD_PP_COORDS`  | Explicit pilot-point coordinates, shape `(N, 3)` (easting, northing, depth). Required when `MOD_PP_MODE = "fixed"` or `"mixed"`. |

**`MOD_PP_MODE` options:**

| Value     | Behaviour                                                                              |
|-----------|----------------------------------------------------------------------------------------|
| `"random"` | `MOD_N_PP` points drawn uniformly inside `MOD_PP_BBOX` — different locations each run. |
| `"fixed"`  | Locations taken from `MOD_PP_COORDS` — same geometry every run, only values change.   |
| `"mixed"`  | `MOD_PP_COORDS` plus `MOD_N_PP` additional random points — fixed skeleton + random fill. |

#### Resistivity range

| Variable           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| `MOD_LOG_RHO_MIN`  | Minimum pilot-point value in log₁₀(Ω·m). Also applied as a clamp after Kriging. |
| `MOD_LOG_RHO_MAX`  | Maximum pilot-point value in log₁₀(Ω·m). Also applied as a clamp after Kriging. |

#### Variogram

Ordinary Kriging is performed with a **gstools** covariance model.  All
three spatial dimensions (easting, northing, depth) are handled natively.

| Variable            | Description                                                                                           |
|---------------------|-------------------------------------------------------------------------------------------------------|
| `MOD_VARIO_MODEL`   | gstools covariance model class name: `"Spherical"` (default), `"Gaussian"`, `"Exponential"`, `"Matern"`, `"Linear"`, `"PowerLaw"`, etc. |
| `MOD_VARIO_RANGE`   | Correlation length (m). Scalar = isotropic. Tuple `(h_range, v_range)` = geometric anisotropy with separate horizontal and vertical ranges. |
| `MOD_VARIO_SILL`    | Sill (variance) in (log₁₀ Ω·m)². Controls total variability of the Kriged field. Typical range: 0.1 – 1.0. |
| `MOD_VARIO_NUGGET`  | Nugget (short-range discontinuity) in (log₁₀ Ω·m)². Keep ≤ 10% of sill for spatial coherence.      |
| `MOD_VARIO_ANGLES`  | Rotation angles `[α, β, γ]` in degrees orienting the anisotropy axes (converted to radians internally). `None` = axis-aligned. |

**Recommended starting values:**

| Parameter          | Recommended value / rule of thumb                                     |
|--------------------|-----------------------------------------------------------------------|
| `MOD_VARIO_MODEL`  | `"Spherical"` — moderate smoothness, widely used in geophysics.       |
| `MOD_VARIO_RANGE`  | `(L_h, L_v)` where L_h ≈ half the survey aperture, L_v ≈ half the target depth. |
| `MOD_VARIO_SILL`   | 0.25 – 0.5 (log₁₀ Ω·m)² — corresponds to ±0.5 – 0.7 log₁₀ units 1-sigma spread. |
| `MOD_VARIO_NUGGET` | 0.01 – 0.05 (≤ 10% of sill).                                         |
| `MOD_VARIO_ANGLES` | `None` unless the geology has a known strike/dip that should be reflected in the prior. |

#### Output target

| Variable              | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `MOD_OUTPUT_TARGET`   | Which file(s) receive the Kriged model per member: `"resistivity_block"`, `"referencemodel"`, or `"both"`. |
| `MOD_RESISTIVITY_FILE`| Filename for the initial model (default: `resistivity_block_iter0.dat`).|
| `MOD_REFERENCE_FILE`  | Filename for the reference / prior model (default: `referencemodel.dat`).|
| `MOD_REF`             | Full path to the template reference model — read once to obtain mesh cell-centre coordinates. |

**`MOD_OUTPUT_TARGET` guidance:**

| Value                  | Effect                                                                 |
|------------------------|------------------------------------------------------------------------|
| `"resistivity_block"`  | Only the starting iterate varies; the regularisation prior is shared.  |
| `"referencemodel"`     | Only the prior varies; all members start from the same iterate.        |
| `"both"`               | Starting iterate **and** prior both equal the Kriged field (recommended for a fully geostatistical prior). |

### Data perturbation

| Variable       | Description                                                         |
|----------------|---------------------------------------------------------------------|
| `PERTURB_DAT`  | Enable / disable data perturbation.                                 |
| `DAT_PDF`      | Distribution parameters for noise: `['normal', 0., 1.0]`.          |
| `RESET_ERRORS` | If `True`, overwrite error floors before perturbation.              |
| `ERRORS`       | Per-component error floors for impedance, VTF, and phase tensor.    |

### Visualization

All visualization parameters live in a single **Visualization config** section.
`matplotlib.pyplot` is imported at the top level.

Ensemble members shown in the plots are drawn *randomly* without replacement
from 0 … `N_SAMPLES − 1` each run.  The drawn list is printed at runtime.

| Variable               | Description                                                                        |
|------------------------|------------------------------------------------------------------------------------|
| `PLOT_DATA`            | Enable / disable joint data plot (original vs. perturbed `observe.dat`).           |
| `PLOT_MODEL`           | Enable / disable joint model plot (reference vs. Kriged initial model).            |
| `VIZ_N_SAMPLES`        | Number of ensemble members to draw for both plots (≤ `N_SAMPLES`).                |
| `VIZ_N_SITES`          | Number of MT sites drawn per data-plot row; `None` shows all sites.                |
| `DAT_WHAT`             | List of panel types: `'rho'`, `'phase'`, `'tipper'`, `'pt'`.                      |
| `DAT_COMPS`            | Parallel list of component strings (one per entry in `DAT_WHAT`).                  |
| `DAT_SHOW_ERRORS_ORIG` | Show ±1σ error envelopes on original curves.                                       |
| `DAT_SHOW_ERRORS_PERT` | Show ±1σ error envelopes on perturbed curves.                                      |
| `DAT_ALPHA_ORIG`       | Opacity of original data curves (0–1).                                             |
| `DAT_ALPHA_PERT`       | Opacity of perturbed data curves (0–1).                                            |
| `DAT_COMP_MARKERS`     | Dict of marker symbol per component class; `None` = `femtic_viz` defaults.         |
| `DAT_MARKERSIZE`       | Marker size in points.                                                              |
| `DAT_MARKEVERY`        | Plot a marker every N-th period; `None` = every period.                            |
| `DAT_ERROR_STYLE_ORIG` | Error rendering for original curves: `'shade'`, `'bar'`, or `'both'`.              |
| `DAT_ERROR_STYLE_PERT` | Error rendering for perturbed curves: `'shade'`, `'bar'`, or `'both'`.             |
| `DAT_PERLIMS`          | Period-axis limits `(T_min, T_max)` in seconds; `None` = auto.                     |
| `DAT_RHOLIMS`          | y-axis limits for `'rho'` panels (Ω·m); `None` = auto.                             |
| `DAT_PHSLIMS`          | y-axis limits for `'phase'` panels (degrees); `None` = auto.                       |
| `DAT_VTFLIMS`          | y-axis limits for `'tipper'` panels; `None` = auto.                                |
| `DAT_PTLIMS`           | y-axis limits for `'pt'` panels; `None` = auto.                                    |
| `MOD_MESH`             | Path to the shared `mesh.dat`.                                                     |
| `MOD_MODE`             | Slice rendering mode: `'tri'` \| `'scatter'` \| `'grid'`.                         |
| `MOD_LOG10`            | Plot log₁₀(ρ) if `True`.                                                          |
| `MOD_CMAP`             | Matplotlib colormap (default `'jet_r'`).                                           |
| `MOD_CLIM`             | `(vmin, vmax)` in log₁₀(Ω·m); `None` = auto from reference model.                |
| `MOD_XLIM`             | `(xmin, xmax)` in metres for map slices; `None` = auto.                            |
| `MOD_YLIM`             | `(ymin, ymax)` in metres — northing (map) or along-profile (curtain); `None` = auto.|
| `MOD_ZLIM`             | `(zmin, zmax)` in metres for curtain slices (negative-down); `None` = auto.        |
| `MOD_MESH_LINES`       | Overlay triangulation edges on filled patches (default `False`).                   |
| `MOD_MESH_LW`          | Line width for mesh edge overlay (pt).                                             |
| `MOD_MESH_COLOR`       | Colour for mesh edge overlay.                                                      |
| `MOD_SLICES`           | List of 1–5 slice dicts, each with `'type'`: `'map'` or `'curtain'`.              |

Model diagnostic figures compare the **uniform reference model** (original) against
the **Kriged initial model** for each selected member, showing how much structural
variability the geostatistical prior injects.  Figures are saved as
`gst_model<PLOT_STR>.pdf` in each member's own subdirectory.

## Tuning guidance

### Pilot-point density

Too few pilot points → the Kriged field is overly smooth, ensemble spread
is low.  Too many → computational cost of Kriging rises (O(N³) with N pilot
points), and the field may over-fit random fluctuations.

A practical rule: use 50–150 points for a survey volume of order 100 km × 100 km
× 50 km.  Check that the variogram range is at least 1.5× the typical
inter-pilot-point spacing to ensure adequate correlation.

### Variogram range vs. survey geometry

The horizontal range should span a significant fraction of the survey aperture
(e.g. 20–50 km for a 100 km survey).  The vertical range controls depth
smoothing and should reflect the expected depth resolution of the MT data
(typically 3–10× shallower than the horizontal range).

### Sill vs. resistivity contrast

A sill of 0.25 (log₁₀ Ω·m)² gives a 1-sigma spread of ~0.5 log₁₀ units,
corresponding to roughly a factor of 3 in resistivity.  Increase the sill
if you expect large contrasts (e.g. conductor vs. resistor) to be sampled
in the prior.

### `MOD_OUTPUT_TARGET`

For full geostatistical prior sampling use `"both"`: the inversion then explores
the misfit surface starting from — and regularised toward — a distinct spatial
pattern for every member.  Use `"resistivity_block"` only if you want to hold
the regularisation prior fixed (shared reference model) while varying the
starting point.

## Dependencies

| Package        | Role                                                   |
|----------------|--------------------------------------------------------|
| `numpy`        | Array operations and random draws.                     |
| `gstools`      | Variogram models and Ordinary Kriging.                 |
| `ensembles`    | Directory generation and data ensemble.                |
| `femtic`       | FEMTIC I/O (`read_model`, `write_model`).              |
| `femtic_viz`   | Ensemble visualization helpers.                        |
| `util`         | Print / version helpers.                               |

No sparse-matrix file (`.npz`) is required.

## References

- Suzuki, K. et al.
  *Geostatistical initial-model ensemble for uncertainty quantification in
  magnetotelluric inversion.*
  [full reference to be completed]

- Isaaks, E. H. & Srivastava, R. M.
  *An Introduction to Applied Geostatistics.*
  Oxford University Press, New York, 1989.

- Müller, S.; Schüler, L.; Zech, A. & Heße, F.
  *GSTools v1.3: a toolbox for geostatistical modelling in Python.*
  Geoscientific Model Development, 2022, **15**, 3161–3182,
  doi:10.5194/gmd-15-3161-2022.

- Bardsley, J. M.; Solonen, A.; Haario, H. & Laine, M.
  *Randomize-Then-Optimize: a Method for Sampling from Posterior Distributions
  in Nonlinear Inverse Problems.*
  SIAM J. Sci. Comp., 2014, **36**, A1895–A1910.

## Provenance

| Date       | Author        | Change                                                        |
|------------|---------------|---------------------------------------------------------------|
| 2026-04-27 | vrath / Claude | Created, modelled on `femtic_rto_prep.py`. GST model          |
|            |               | perturbation replaces roughness-matrix prior draw with        |
|            |               | pilot-point Ordinary Kriging (gstools). Data perturbation     |
|            |               | block unchanged from RTO. `MOD_PP_MODE` supports `"random"`,  |
|            |               | `"fixed"`, and `"mixed"` pilot-point strategies.              |
|            |               | `MOD_OUTPUT_TARGET` controls which FEMTIC files are written.  |

## Author

Volker Rath (DIAS) — April 2026
