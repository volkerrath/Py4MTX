# femtic_nss.py — Nullspace Shuttle for FEMTIC Inversion Results

**Py4MTX framework · vrath / Claude Sonnet 4.6 · 2026-05-17**

---

## Purpose

`femtic_nss.py` reads the final model and data from an HDF5 inversion archive,
computes the data-weighted Jacobian, decomposes it via randomised SVD, applies
a user-defined model perturbation, and projects that perturbation onto the
**null space** of the scaled Jacobian so that it leaves the predicted data
unchanged.  The result is written as a new FEMTIC resistivity block.

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
| **4** | Call `_modify_model(model)` → raw perturbation δm̃  ← **edit this** |
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

The file is in standard FEMTIC format and can be used directly as the starting
model for a further inversion run or passed to `femtic_mod_edit.py` /
`femtic_viz.py` for inspection.

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

## Model Modification — `_modify_model`

```python
def _modify_model(m: np.ndarray) -> np.ndarray:
    ...
```

This is the **only function the user is expected to edit**.  It receives the
final model vector `m` (log₁₀(ρ), free regions only, shape `(n_m,)`) and
returns a perturbation `dm` of the same shape.  The perturbation is then
projected onto the null space of Js, so any data-sensitive component is
automatically removed before the result is added to the model.

The default placeholder returns a unit-Gaussian random vector (seeded for
reproducibility).  Replace the body between the `*** EDIT ***` markers with
a physically motivated perturbation, for example:

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

Step 4: Model modification
  ||dm_raw||  = 1.1102e+02

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
| `femtic` (Py4MTX) | `read_model`, `insert_model`, `write_model` |
| `util` (Py4MTX) | `print_title` |
| `version` (Py4MTX) | `versionstrg` |

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

---

## References

- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding structure with
  randomness: Probabilistic algorithms for constructing approximate matrix
  decompositions. *SIAM Review*, 53(2), 217–288.
- Deal, M. M., & Nolet, G. (1996). Nullspace shuttles. *Geophysical Journal
  International*, 124(2), 372–380.
- Muñoz, G., & Rath, V. (2006). Beyond smooth inversion: the use of nullspace
  projection for the exploration of non-uniqueness in MT. *Geophysical Journal
  International*, 164(2), 301–311.
