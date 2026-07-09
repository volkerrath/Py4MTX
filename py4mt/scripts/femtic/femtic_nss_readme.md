# femtic_nss.py — Nullspace Shuttle for FEMTIC Inversion Results

**Py4MTX framework · vrath / Claude Sonnet 4.6 · 2026-05-17 / 2026-06-23**

---

## Purpose

`femtic_nss.py` reads the final model and data from an HDF5 inversion archive,
computes the data-weighted Jacobian, decomposes it via randomised SVD, generates
a model perturbation, and projects that perturbation onto the **null space** of
the scaled Jacobian so that it leaves the predicted data unchanged.  The result
is written as a new FEMTIC resistivity block.

Two perturbation modes are available, selected by `PERTURB_MODE`:

- **`"gst"`** *(default)* — geostatistical perturbation via pilot-point Ordinary
  Kriging (`ensembles.generate_gst_model_ensemble`).  Produces spatially coherent
  perturbations whose character is governed by a user-specified variogram model.
- **`"random"`** — uniform Gaussian placeholder (original behaviour).  The helper
  `_make_perturbation_random` can be edited to inject any user-defined prior-based
  perturbation.

The script is modelled on `femtic_mod_edit.py` and follows the same
configuration-block / helper-function pattern.

---

## Background

A 3-D MT inversion has many more model parameters than data.  The inversion
minimises data misfit but cannot constrain the null space N(J) — directions in
model space that produce no change in predicted data.  The **nullspace shuttle**
(Muñoz & Rath, 2006; Deal & Nolet, 1996) exploits this: any perturbation
δm̃ can be split into a data-sensitive component (row space of J) and a
data-invisible component (null space of J).  Adding only the null-space
component to the final model yields an equally valid model with different
resistivity structure.

This is useful for:

- testing how much structural variability is permitted by the data,
- constructing model ensembles for uncertainty visualisation,
- introducing geological constraints that the inversion itself could not recover.

### Mathematics

Let **J**_s = diag(1/σ) **J** be the error-scaled Jacobian (n_d × n_m).
Its truncated randomised SVD is:

```
Js ≈ U  S  Vt        U : (nd, r)   S : (r,)   Vt : (r, nm)
```

The **r_eff** right singular vectors with s_i ≥ τ · s_0 span the row space;
the remaining n_m − r_eff directions span the null space.  The null-space
projector is:

```
P_null = I  −  Vr Vr^T        Vr = Vt[:r_eff].T   (nm, r_eff)
```

The shuttle perturbation added to the model is:

```
δm_null = α · P_null · δm̃
```

where δm̃ comes from `_modify_model` and α = `NSS_AMPLITUDE`.

Verification: `‖Js · δm_null‖` is printed and should be near machine zero.

---

## Workflow

| Step | What happens |
|------|-------------|
| **1** | Read `model`, `observed`, `calculated`, `errors`, `jacobian` from `Inversion_results.h5` |
| **2** | Form `Js = diag(1/error) @ J` and normalised residual `rs` |
| **3** | Randomised SVD of `Js` via `inverse.rsvd` (Halko et al., 2011) |
| **4** | Generate raw perturbation δm̃ via `PERTURB_MODE`: `"gst"` → Kriged model delta; `"random"` → user-editable Gaussian |
| **5** | Project δm̃ onto null space; add to model; write output block |

---

## Input

### HDF5 file — `Inversion_results.h5`

| Dataset | Shape | Description |
|---------|-------|-------------|
| `model` | (n_m,) | Final model — log₁₀(ρ) for free regions |
| `observed` | (n_d,) | Observed data vector |
| `calculated` | (n_d,) | Forward-modelled data at final model |
| `errors` | (n_d,) | Data errors (positive, same units as data) |
| `jacobian` | (n_d, n_m) | Sensitivity matrix at final model |

### Template resistivity block — `MODEL_TEMPLATE`

Used by `fem.read_model` / `fem.insert_model` to preserve the FEMTIC file
structure (header, region bounds, flag column, fixed regions, air, ocean).
Only the free-region values are replaced.  Any valid FEMTIC resistivity block
(e.g. `resistivity_block_iter0.dat`) works.

---

## Output

| File | Description |
|------|-------------|
| `resistivity_block_nss.dat` | Null-space-shuttled FEMTIC resistivity block |
| `nss_qc.pdf` | Optional QC slice figure of the output model (`MOD_QC = True`) |

The `.dat` file is in standard FEMTIC format and can be used directly as the
starting model for a further inversion run or passed to `femtic_mod_edit.py` /
`femtic_viz.py` for inspection.

---

## Plotting config (shared with `femtic_gst_prep.py` / `femtic_ens_post.py`)

Setting `MOD_QC = True` produces a single slice figure of the nullspace-
shuttled output model (`MODEL_OUT`), using the same `MOD_*` config surface --
variable names, defaults, and order -- as `femtic_gst_prep.py` and
`femtic_ens_post.py`.  A config block can be copied between the three
scripts with no renaming.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_MESH` | str | `WORK_DIR + "mesh.dat"` | Tetrahedral mesh -- required for the QC plot. |
| `MOD_OCEAN` | bool / None | `None` | `None` = auto-infer; `True`/`False` forces ocean-present/absent. |
| `MOD_AIR_RHO` | float | `1.0e9` | Ohm.m sentinel for air cells (region 0). |
| `MOD_OCEAN_RHO` | float | `0.25` | Ohm.m sentinel for ocean cells (region 1). |
| `MOD_QC` | bool | `False` | Enable the QC slice plot of `MODEL_OUT`. |
| `MOD_QC_FILE` | str | `WORK_DIR + "nss_qc.pdf"` | Output path. |
| `MOD_DPI` | int | `600` | Figure DPI. |

The shared slice/plot, site-overlay, and geographic/UTM-origin parameters
(`MOD_SLICES`, `MOD_XLIM/YLIM/ZLIM`, `MOD_CMAP`, `MOD_CLIM`,
`MOD_OCEAN_COLOR`, `MOD_AIR_COLOR`, `MOD_AIR_BGCOLOR`, `MOD_ALPHA_*`,
`MOD_EQUAL_ASPECT`, `MOD_DEPTH_KM`/`MOD_HORIZ_KM`, `MOD_PANEL_HEIGHT`/
`MOD_PANEL_WIDTH`, `MOD_FIGSIZE`, `MOD_NROWS`/`MOD_NCOLS`, `MOD_SITE_DAT`,
`MOD_SITE_NAMES`, `MOD_SITE_NUMBER`, `MOD_PLOT_SITES_MAPS`/`SLICES`,
`MOD_PROJECTION_DIST`, `MOD_SITE_MARKER`/`MARKER_SLICES`, `MOD_MAP_MARKERS`,
`MOD_ORIGIN_METHOD`, `MOD_UTM_ORIGIN_LAT/LON/E/N`, `MOD_UTM_ZONE_OVERRIDE`,
`MOD_DISPLAY_COORDS`) have identical names, defaults, and semantics to
`femtic_ens_post.py`'s "Shared slice / plot parameters" -- see that README
for the full parameter table.

---

## Configuration

All user-facing parameters are collected in the **Configuration** block near
the top of the script.  No other edits are needed for a standard run.

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | `/home/vrath/Py4MTX/work/` | Directory for all I/O |
| `HDF5_FILE` | `WORK_DIR + "Inversion_results.h5"` | Input HDF5 archive |
| `MODEL_TEMPLATE` | `WORK_DIR + "resistivity_block_iter0.dat"` | FEMTIC template block |
| `MODEL_OUT` | `WORK_DIR + "resistivity_block_nss.dat"` | Output block |

### Perturbation mode  (step 4)

| Variable | Default | Description |
|----------|---------|-------------|
| `PERTURB_MODE` | `"gst"` | `"gst"` — pilot-point Kriging perturbation; `"random"` — Gaussian placeholder |

### GST model perturbation  (used when `PERTURB_MODE == "gst"`)

The GST block mirrors the equivalent config in `femtic_gst_prep.py`.

**Paths**

| Variable | Default | Description |
|----------|---------|-------------|
| `GST_REF_MOD` | `MODEL_TEMPLATE` | Reference model providing free-region structure |
| `GST_MESH` | `WORK_DIR + "mesh.dat"` | FEMTIC mesh file for barycentre computation |

**Pilot-point placement**

| Variable | Default | Description |
|----------|---------|-------------|
| `GST_PP_MODE` | `"random"` | Placement strategy: `"random"` / `"fixed"` / `"mixed"` / `"extrema"` |
| `GST_N_PP` | `100` | Number of random pilot points per realisation (used in `"random"`, `"mixed"`, `"extrema"`) |
| `GST_PP_BBOX` | `[-50000, 50000, -50000, 50000, 0, 80000]` | Bounding box [x_min, x_max, y_min, y_max, z_min, z_max] (m, z positive-down) |
| `GST_PP_COORDS` | `None` | Explicit (N, 3) array of pilot-point locations for `"fixed"` / `"mixed"` mode |
| `GST_PP_ROI` | `None` | Sub-volume for extremum search in `"extrema"` mode; `None` = full extent |
| `GST_PP_EXTREMA_K` | `9` | Neighbourhood size for local extremum test in `"extrema"` mode |
| `GST_PP_EXTREMA_WHICH` | `"both"` | Which extrema to seed: `"both"` / `"minima"` / `"maxima"` |

**Resistivity range**

| Variable | Default | Description |
|----------|---------|-------------|
| `GST_LOG_RHO_MIN` | `0.0` | Lower draw bound and post-Kriging clamp (log₁₀ Ω·m) |
| `GST_LOG_RHO_MAX` | `4.0` | Upper draw bound and post-Kriging clamp (log₁₀ Ω·m) |

**Variogram model**

| Variable | Default | Description |
|----------|---------|-------------|
| `GST_VARIO_MODEL` | `"Spherical"` | gstools covariance model class: `"Spherical"`, `"Gaussian"`, `"Exponential"`, `"Matern"`, … |
| `GST_VARIO_RANGE` | `(8000., 4000.)` | Correlation length(s) in m.  Scalar = isotropic; 2-tuple `(h_range, v_range)` = anisotropic |
| `GST_VARIO_SILL` | `0.5` | Sill (variance) in (log₁₀ Ω·m)² |
| `GST_VARIO_NUGGET` | `0.01` | Nugget in (log₁₀ Ω·m)²; keep ≤ 10 % of sill |
| `GST_VARIO_ANGLES` | `None` | Rotation [α, β, γ] in degrees; `None` = axis-aligned |

### Randomised SVD  (step 3)

| Variable | Default | Description |
|----------|---------|-------------|
| `RSVD_RANK` | `300` | Target rank for rSVD of Js. Increase until the singular-value spectrum flattens. Must be ≪ min(n_d, n_m). |
| `RSVD_OVERSAMPLES` | `None` | Extra random columns. `None` → 2 × `RSVD_RANK` (Halko et al. default). |
| `RSVD_SUBSPACE_ITERS` | `2` | Power iterations for improved accuracy. More iterations = better approximation at higher cost. |

### Nullspace shuttle  (step 5)

| Variable | Default | Description |
|----------|---------|-------------|
| `NSS_SV_THRESH` | `1.0e-3` | Singular-value cutoff as a fraction of s₀. Vectors with s_i < τ · s₀ are treated as null-space. Smaller τ → larger effective rank → less null-space freedom. |
| `NSS_AMPLITUDE` | `1.0` | Scale factor α applied to δm_null before adding to the model. Start at 0.1 and increase to explore the null space more aggressively. |

### Diagnostics

| Variable | Default | Description |
|----------|---------|-------------|
| `OUT` | `True` | Print shapes, norms, and verification metrics at each step. |

---

## Model Perturbation Functions  (step 4)

### `_make_perturbation_random(m)`

```python
def _make_perturbation_random(m: np.ndarray) -> np.ndarray:
    ...
```

Used when `PERTURB_MODE = "random"`.  Receives the final model vector `m`
(log₁₀(ρ), free regions only, shape `(n_m,)`) and returns a perturbation `dm`
of the same shape.  The perturbation is projected onto the null space of Js, so
any data-sensitive component is automatically removed before the result is added
to the model.

The default placeholder returns a unit-Gaussian random vector (seeded for
reproducibility).  Replace the body between the `*** EDIT ***` markers:

```python
# Push a target region towards higher resistivity
dm = np.zeros_like(m)
dm[region_indices] = 1.0          # +1 log unit in target region

# Uniform bias across the whole model
dm = np.full_like(m, 0.5)         # +0.5 log units everywhere

# Seeded random ensemble member
rng = np.random.default_rng(seed=42)
dm = rng.standard_normal(m.size) * 0.2
```

### `_make_perturbation_gst(m_ref)`

```python
def _make_perturbation_gst(m_ref: np.ndarray) -> np.ndarray:
    ...
```

Used when `PERTURB_MODE = "gst"`.  Calls
`ens.generate_gst_model_ensemble()` for a single realisation in a temporary
subdirectory under `WORK_DIR`.  The Kriged resistivity block is read back,
the reference-model vector is subtracted, and the delta is returned:

```
dm = m_gst − m_ref       (log₁₀ Ω·m, free regions only)
```

The temporary directory is cleaned up regardless of success or failure.  The
spatial character of `dm` (correlation lengths, variance) is governed entirely
by the `GST_VARIO_*` settings.

---

## Nullspace Shuttle — `_nullspace_shuttle`

```python
def _nullspace_shuttle(dm, Vt, S, *, sv_thresh, amplitude) -> (dm_null, r_eff)
```

Internal helper.  Not intended for editing.

Computes the effective rank `r_eff` from `S` and `sv_thresh`, forms the
row-space basis `Vr = Vt[:r_eff].T`, and returns:

```
dm_null = amplitude * (dm  −  Vr @ (Vr.T @ dm))
```

Also returns `r_eff` for diagnostic printing.

---

## Printed diagnostics

```
Step 1: Reading inversion results from HDF5
  model      : (12345,)
  jacobian   : (4800, 12345)
  nd=4800, nm=12345

Step 2: Computing scaled Jacobian Js = diag(1/error) @ J
  Js shape : (4800, 12345)
  ||rs||   : 69.2832
  RMS      : 1.0003

Step 3: Randomised SVD of Js
  Decomposition: U (4800, 300), S (300,), Vt (300, 12345)
  s[0]  = 3.4712e+02  (largest)
  s[-1] = 1.8843e-01  (smallest in truncated set)
  Effective rank at threshold 1.0e-03: 247 / 300

Step 4: Model perturbation  [PERTURB_MODE = 'gst']
  ||dm_raw||  = 9.8431e+01
  dm_raw range: [-1.823, 2.417]
  elapsed     : 3.14 s

Step 5: Nullspace shuttle
  Effective rank used for projection : 247
  ||dm_null||  = 1.0934e+02
  ||dm_row ||  = 1.9281e+01
  ||Js @ dm_null|| (should be ~0) = 3.4e-12
  model_nss range : [-0.124, 4.231]

Writing nullspace-shuttled model
  Written: /home/vrath/Py4MTX/work/resistivity_block_nss.dat
```

The key verification line is `‖Js · δm_null‖ ≈ 0`.  Values above ~ 10⁻⁶
indicate that `RSVD_RANK` is too small or `NSS_SV_THRESH` is too large.

---

## Dependencies

| Package | Role |
|---------|------|
| `numpy` | Array operations, norm computations |
| `h5py` | Reading `Inversion_results.h5` |
| `inverse` (Py4MTX) | `rsvd`, `find_range`, `subspace_iter`, `ortho_basis` |
| `femtic` (Py4MTX) | `read_model`, `extract_model`, `insert_model`, `write_model` |
| `ensembles` (Py4MTX) | `generate_gst_model_ensemble` (GST mode only) |
| `util` (Py4MTX) | `print_title` |
| `version` (Py4MTX) | `versionstrg` |
| `gstools` | Variogram / Ordinary Kriging (via `ensembles`, GST mode only) |

---

## Tuning guide

**How large should `RSVD_RANK` be?**
Plot the singular-value spectrum of Js (values in `S` after step 3).  The
rank should cover the knee of the spectrum — where values transition from
large (data-sensitive) to small (effectively null).  Starting with 300 is
reasonable for typical FEMTIC problems; reduce if memory is tight, increase if
`r_eff / rank` is close to 1 (spectrum not yet fully captured).

**How to choose `NSS_SV_THRESH`?**
Values between 1 × 10⁻³ and 1 × 10⁻² are typical.  A smaller threshold
includes more singular vectors in the row space, leaving less null-space
freedom but producing a cleaner projection.  Watch the `r_eff` printed in
step 3 and step 5: if `r_eff` equals `RSVD_RANK`, lower the threshold or
increase the rank.

**How to choose `NSS_AMPLITUDE`?**
Start at 0.1.  Increase to 1.0 or beyond to produce more distinct
null-space members.  Values ≫ 1 may push log₁₀(ρ) outside physically
plausible bounds; clip with `femtic_mod_edit.py` (operation `"clip"`) if
needed.

---

## Provenance

| Date | Author | Change |
|------|--------|--------|
| 2026-05-17 | vrath / Claude Sonnet 4.6 | Created, modelled on `femtic_mod_edit.py` |
| 2026-06-23 | vrath / Claude Sonnet 4.6 | Merged GST model generation from `femtic_gst_prep.py`. Added `PERTURB_MODE` switch; `"gst"` path calls `ens.generate_gst_model_ensemble` for a single realisation and returns `m_gst − m_ref` as perturbation delta. Full GST config block (`GST_PP_*`, `GST_VARIO_*`). `"random"` retains original Gaussian placeholder in `_make_perturbation_random`. Added `ensembles` import. |
| 2026-07-09 | vrath / Claude Sonnet 5 (Anthropic) | Added the shared `MOD_*` plotting config block, `femtic_viz` import, `_resolve_origin_and_sites()` / `_plot_slice()` helpers, and an optional QC slice plot of `MODEL_OUT` (`MOD_QC = True`). Config surface is identical in name and order to `femtic_ens_post.py` and `femtic_gst_prep.py`, so a block can now be copied between all three scripts with no renaming. Uses a single `MOD_DPI` knob, matching `femtic_gst_prep.py` and the now-simplified `femtic_ens_post.py`. |

**How to configure the GST variogram?**
The variogram controls the spatial coherence of the Kriged perturbation.
`GST_VARIO_RANGE` should be set to plausible geological correlation lengths —
roughly half the survey aperture horizontally and half the target depth
vertically.  `GST_VARIO_SILL` controls the overall variance of the field in
(log₁₀ Ω·m)²; values of 0.25–0.5 correspond to ±0.5–0.7 log units (1σ).
Keep `GST_VARIO_NUGGET` ≤ 10 % of the sill to maintain spatial coherence.

**How many pilot points?**
`GST_N_PP = 100` is a good starting point for typical 3-D MT survey volumes.
Too few (< 30) may leave large voids in the Kriging interpolation; too many
(> 500) slow down the Kriging without improving the spatial representation.

---



- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding structure with
  randomness: Probabilistic algorithms for constructing approximate matrix
  decompositions. *SIAM Review*, 53(2), 217–288.
- Deal, M. M., & Nolet, G. (1996). Nullspace shuttles. *Geophysical Journal
  International*, 124(2), 372–380.
- Muñoz, G., & Rath, V. (2006). Beyond smooth inversion: the use of nullspace
  projection for the exploration of non-uniqueness in MT. *Geophysical Journal
  International*, 164(2), 301–311.
- Suzuki, K., et al. (2025). Geostatistical initial-model ensemble for
  magnetotelluric uncertainty quantification. [full reference to be completed]
