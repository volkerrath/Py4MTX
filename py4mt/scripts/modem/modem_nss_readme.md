# modem_nss.py — Nullspace Shuttle for ModEM Inversion Results

**py4mt framework · vrath / Claude Sonnet 5 · 2026-07-19**

---

## Purpose

`modem_nss.py` generalises `femtic_nss.py` to ModEM's structured
rectilinear grid. It reads a converged ModEM model and its processed
Jacobian archive, decomposes the (data-weighted) Jacobian via randomised
SVD, generates a candidate model perturbation, projects that perturbation
onto the **null space** of the Jacobian so the predicted data are left
unchanged, and writes the result as a new `.rho` model.

Two perturbation modes, selected by `PERTURB_MODE`:

- **`"gst"`** *(default)* — geostatistical perturbation via pilot-point
  Ordinary Kriging, generated directly on the ModEM grid's cell centres
  (`ensembles.generate_gst_perturbation_modem`).
- **`"random"`** — Gaussian placeholder (edit `_make_perturbation_random`
  to inject any other prior-based perturbation).

This is a companion script to `femtic_nss.py`, sharing its null-space
mathematics but built for ModEM's I/O and grid geometry.

---

## What's new: the mesh-agnostic GST perturbation machine

`femtic_nss.py`'s `_make_perturbation_gst` generates one perturbation by
calling `ensembles.generate_gst_model_ensemble` for a single realisation
inside a temporary directory, then reading the written file back — a
round-trip built for FEMTIC's directory-based, file-I/O ensemble workflow.
ModEM has no unstructured mesh, no free-region concept, and no need for
that file round-trip for a single NSS candidate.

`ensembles.py` therefore gained (2026-07-19) a mesh-agnostic **perturbation
machine** that factors the pilot-point-placement + Ordinary-Kriging core
out of `generate_gst_model_ensemble`, so a single realisation can be
generated in memory given nothing but a target point cloud:

| Function | Role |
|---|---|
| `krige_pilot_points_to_targets()` | Ordinary-Krig pilot-point values onto an arbitrary point cloud. |
| `_draw_pilot_points()` | Shared random / fixed / mixed / extrema pilot-point placement logic. |
| `generate_gst_perturbation()` | One realisation, given a target point cloud — the mesh-agnostic "machine". Used directly for FEMTIC-style targets (e.g. free-region barycentres). |
| `modem_gst_cell_centers()` | ModEM rectilinear-grid cell centres from `dx, dy, dz` (+ optional free-cell mask), the ModEM analogue of FEMTIC's free-region barycentres. |
| `generate_gst_perturbation_modem()` | ModEM wrapper: computes cell centres, calls `generate_gst_perturbation`, scatters the result back into an `(nx, ny, nz)` array (air cells unchanged). |

`generate_gst_model_ensemble` itself (FEMTIC's directory-based ensemble
generator) is unchanged — these are additive. `femtic_nss.py`'s
`_make_perturbation_gst` could be pointed at `generate_gst_perturbation`
directly in a future cleanup to drop its temp-directory round-trip, but
that edit was not made here to avoid touching a working FEMTIC code path.

---

## Workflow

| Step | What happens |
|------|---------------|
| **1** | Read `model` (`modem.read_mod`, `.rho`, `trans="LOGE"`) and the Jacobian archive (`<JFile>_jac.npz`, `<JFile>_info.npz`, `jac_proc`/`modem_jac_svd.py` layout). Determine whether Jacobian columns index the full grid or free (non-air) cells only. |
| **2** | Form `Js` — the archived Jacobian as-is (`JAC_ALREADY_SCALED = True`, default) or `diag(Scale) @ Jac` if not. |
| **3** | Randomised SVD of `Js.T` via `jac_proc.rsvd` (Halko et al., 2011) — same call as `modem_jac_svd.py`. |
| **4** | Generate raw perturbation δm̃ via `PERTURB_MODE`: `"gst"` → Kriged grid delta (log10 Kriging, converted to loge); `"random"` → user-editable Gaussian. |
| **5** | Project δm̃ onto the null space (using `U`, the model-space singular vectors — see "U vs Vt" below); add to the model; write output `.rho`. |

---

## ⚠️ Assumptions to verify

This script was written without direct access to `jac_proc.py` or the
current `modem.py` source in this session, only to `modem_jac_svd.py` (which
consumes the same archive) and prior py4mt conversation history establishing
`modem.read_mod` / `modem.write_mod` conventions. Three points are flagged
inline with `# ASSUMPTION:` comments and **should be checked against your
actual modules before trusting the output**:

1. **Jacobian scaling** (step 2). `modem_jac_svd.py` calls
   `jac.rsvd(Jac.T, ...)` directly with no visible error-weighting step,
   which is why `JAC_ALREADY_SCALED = True` is the default here. If your
   `_jac.npz` Jacobian is *not* already error-scaled, set
   `JAC_ALREADY_SCALED = False`; the script will then form
   `Js = diag(Scale) @ Jac` from `Dat["Scale"]` — verify that `Scale` is
   really a multiplicative per-datum weight (e.g. `1/error`) and not
   something else (e.g. a raw error, which would need inverting).

2. **Model ↔ Jacobian column correspondence** (step 1). It is not
   established here whether the archived Jacobian's `nm` columns index the
   full grid (`nx*ny*nz`) or free/non-air cells only. The script checks
   `Jac.shape[1]` against both possibilities and **raises an error if
   neither matches** rather than silently mis-indexing — if that happens,
   inspect `jac_proc`'s column-construction logic and adjust Step 1.

3. **U vs Vt after `jac.rsvd(Js.T, ...)`.** `modem_jac_svd.py` decomposes
   `Jac.T`, not `Jac`. Because of the transpose, the *model-space*
   singular vectors come back as `U` (shape `(nm, rank)`), not `Vt` as in
   `femtic_nss.py` (which decomposes `Js` directly, unstransposed). Step 5
   projects using `U`, not `Vt` — this is called out inline where the SVD
   is computed. Double-check this against `jac_proc.rsvd`'s actual
   return-value convention.

4. **`GST_PP_BBOX`** defaults to a placeholder box; set it to bracket your
   model's actual free-cell extent (`dx.sum()`, `dy.sum()`, `dz.sum()`
   from the cell widths, in the model-local coordinate system defined by
   `ensembles.modem_gst_cell_centers`).

None of these affect the null-space *mathematics* (steps 2 core and 5,
which are copied unchanged from `femtic_nss.py`) — only the I/O plumbing
around them.

---

## Unit convention (log10 vs loge)

`modem.read_mod(..., trans="LOGE")` returns the model in natural-log
resistivity, matching the convention used throughout this py4mt codebase
(see `modem_compress.py`, `modem_jac_svd.py`). The GST tooling in
`ensembles.py` (inherited from the FEMTIC pipeline) expresses
`log_rho_min`/`log_rho_max`/`vario_sill`/etc. in **log10(Ω·m)**, the
standard unit in the geophysical literature this tooling is built on.

`modem_nss.py` therefore:

1. Converts the reference model to log10 before Kriging:
   `ref_log10 = model_loge / ln(10)`.
2. Krigs and clamps entirely in log10 space (unchanged FEMTIC-style
   parameters/defaults apply directly).
3. Converts only the resulting **delta** back to loge before adding it to
   the model: `dm_loge = (field_log10 - ref_log10) * ln(10)`.

This keeps `GST_LOG_RHO_MIN/MAX`, `GST_VARIO_SILL`, etc. directly
comparable to the FEMTIC config blocks (`femtic_gst_prep.py`,
`femtic_nss.py`) while leaving the model itself in loge throughout, as
`modem.read_mod`/`write_mod` expect.

---

## Configuration

### Paths

| Variable | Default | Description |
|----------|---------|--------------|
| `WORK_DIR` | `/home/vrath/ModEM_work/...` | Directory for all I/O |
| `JFile` | `WORK_DIR + "Ub25_ZPT_nerr_sp-6"` | Base name for `_jac.npz` / `_info.npz` |
| `MFile` | `WORK_DIR + "Ub_600ZT4_PT_NLCG_009"` | Final ModEM model (no extension) |
| `MODEL_OUT` | `..._nss` | Output resistivity model (no extension) |
| `RHOAIR` | `1.0e17` | Air sentinel — must match the value used when `MFile` was written |

### Jacobian scaling (step 2)

| Variable | Default | Description |
|----------|---------|--------------|
| `JAC_ALREADY_SCALED` | `True` | See "Assumptions to verify" #1 |

### Randomised SVD (step 3)

Same knobs as `modem_jac_svd.py`.

| Variable | Default | Description |
|----------|---------|--------------|
| `RSVD_RANK` | `300` | Target rank. Should be ≪ min(nd, nm). |
| `RSVD_OVERSAMPLE_FACTOR` | `2` | Oversampling = factor × rank |
| `RSVD_SUBSPACE_ITERS` | `2` | Power iterations |

### Nullspace shuttle (step 5)

| Variable | Default | Description |
|----------|---------|--------------|
| `NSS_SV_THRESH` | `1.0e-3` | Fraction of s₀ below which a mode is null |
| `NSS_AMPLITUDE` | `1.0` | Scale factor on the null-space perturbation; start at 0.1 |

### GST perturbation (step 4, `PERTURB_MODE = "gst"`)

Mirrors `femtic_nss.py`'s GST block, in log10(Ω·m) units (see "Unit
convention" above):

| Variable | Default | Description |
|----------|---------|--------------|
| `GST_PP_MODE` | `"random"` | `"random"` \| `"fixed"` \| `"mixed"` \| `"extrema"` |
| `GST_N_PP` | `100` | Pilot points per realisation |
| `GST_PP_BBOX` | *(placeholder — set to your model extent)* | `[x_min,x_max,y_min,y_max,z_min,z_max]` m |
| `GST_PP_COORDS` | `None` | Required for `"fixed"` / `"mixed"` |
| `GST_PP_ROI`, `GST_PP_EXTREMA_K`, `GST_PP_EXTREMA_WHICH` | `None`, `30`, `"both"` | `"extrema"` mode only |
| `GST_LOG_RHO_MIN` / `MAX` | `0.0` / `4.0` | log10 Ω·m draw bounds and clamp |
| `GST_PP_VALUE_MODE` | `"uniform"` | `"uniform"` \| `"reference"` |
| `GST_PP_VALUE_DELTA` | `0.5` | Half-width for `"reference"` mode |
| `GST_VARIO_MODEL` | `"Spherical"` | gstools covariance class |
| `GST_VARIO_RANGE` | `(8000., 4000.)` | (horizontal, vertical) metres |
| `GST_VARIO_SILL` | `0.5` | (log10 Ω·m)² |
| `GST_VARIO_NUGGET` | `0.01` | ≤ 10% of sill |
| `GST_VARIO_ANGLES` | `None` | Rotation, degrees |

---

## Printed diagnostics (example)

```
Step 1: Reading model and Jacobian archive
  model grid  : nx=..., ny=..., nz=...  (N cells, F free / A air)
  Jacobian    : (nd, nm)  (nd=..., nm=...)
  column mode : 'full_grid'  (or 'free_cells')

Step 2: Forming the (data-)scaled Jacobian Js
  JAC_ALREADY_SCALED = True -> using the archived Jacobian as-is
  Js shape : (nd, nm)

Step 3: Randomised SVD of Js
  Decomposition: U (nm, rank), S (rank,), Vt (rank, nd)
  s[0]  = ...   s[-1] = ...
  Effective rank at threshold 1.0e-03: r / rank

Step 4: Model perturbation  [PERTURB_MODE = 'gst']
  generate_gst_perturbation: P pilot points (random/uniform) -> F targets, ...
  ||dm_raw||  = ...

Step 5: Nullspace shuttle
  Effective rank used for projection : r
  ||dm_null||  = ...
  ||Js @ dm_null|| (should be ~0) = ...

Writing nullspace-shuttled model
  Written : .../..._nss.rho
```

The key verification line is `||Js @ dm_null|| ≈ 0`. Values well above
machine-precision-scale indicate `RSVD_RANK` is too small, `NSS_SV_THRESH`
is too large, or one of the "Assumptions to verify" above does not hold
for your archive.

---

## Tuning guide

Same guidance as `femtic_nss.py`:

- **`RSVD_RANK`**: plot `S` from step 3; pick a rank past the spectrum's
  knee (data-sensitive → effectively-null transition). 300 is a
  reasonable start for typical ModEM problems.
- **`NSS_SV_THRESH`**: 1e-3 – 1e-2 typical. If `r_eff` prints equal to
  `RSVD_RANK`, lower the threshold or raise the rank.
- **`NSS_AMPLITUDE`**: start at 0.1, increase toward 1.0+ to explore the
  null space more aggressively; very large values may push log-resistivity
  outside physically plausible bounds.
- **GST variogram**: `GST_VARIO_RANGE` ≈ half the survey aperture
  (horizontal) / half the target depth (vertical); `GST_VARIO_SILL` of
  0.25–0.5 (log10 Ω·m)² gives ≈ ±0.5–0.7 log₁₀-unit (1σ) spread.

---

## Dependencies

| Package | Role |
|---------|------|
| `numpy` | Array operations, norms |
| `scipy` | `scipy.sparse` (Jacobian archive I/O) |
| `gstools` | Variogram / Ordinary Kriging (via `ensembles`, GST mode only) |
| `jac_proc` (py4mt) | `rsvd` — randomised SVD of the ModEM Jacobian |
| `modem` (py4mt) | `read_mod`, `write_mod` |
| `ensembles` (py4mt) | `generate_gst_perturbation_modem`, `modem_gst_cell_centers` (GST mode only) |
| `util`, `version` (py4mt) | Print / version helpers |

---

## References

- Deal, M. M., & Nolet, G. (1996). Nullspace shuttles. *Geophysical
  Journal International*, 124(2), 372–380.
  doi:[10.1111/j.1365-246X.1996.tb07027.x](https://doi.org/10.1111/j.1365-246X.1996.tb07027.x)
- Muñoz, G., & Rath, V. (2006). Beyond smooth inversion: the use of
  nullspace projection for the exploration of non-uniqueness in MT.
  *Geophysical Journal International*, 164(2), 301–311.
  doi:[10.1111/j.1365-246X.2005.02825.x](https://doi.org/10.1111/j.1365-246X.2005.02825.x)
- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding structure
  with randomness: Probabilistic algorithms for constructing approximate
  matrix decompositions. *SIAM Review*, 53(2), 217–288.
  doi:[10.1137/090771806](https://doi.org/10.1137/090771806)

---

## Provenance

| Date | Author | Change |
|------|--------|--------|
| 2026-07-19 | vrath / Claude Sonnet 5 (Anthropic) | Created. Generalises `femtic_nss.py` to ModEM's structured rectilinear grid. Jacobian I/O modelled on `modem_jac_svd.py`. GST perturbation (step 4) uses the new mesh-agnostic `ensembles.generate_gst_perturbation_modem`, added to `ensembles.py` on the same date, instead of FEMTIC's directory-based `generate_gst_model_ensemble`. Nullspace-shuttle mathematics unchanged from `femtic_nss.py`. Several I/O-layer assumptions (Jacobian scaling, column indexing, U-vs-Vt after transposed rSVD) are flagged inline and in "Assumptions to verify" above pending confirmation against `jac_proc.py`. |

## Author

Volker Rath (DIAS) — July 2026
