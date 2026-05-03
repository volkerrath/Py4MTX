# femtic_mod_edit.py

Apply arithmetic operations to a FEMTIC resistivity model in log10 space and
rewrite the model file.

---

## Purpose

`femtic_mod_edit.py` reads a FEMTIC `resistivity_block_iterX.dat`, operates on
the **free** log10(rho) parameter vector, and writes the result back to a
(optionally different) block file.  Typical use cases are:

- creating a homogeneous or constant-value starting model (fill, mean, median),
- enforcing physical bounds (clip),
- shifting the global resistivity level by a known offset (shift),
- spatially smoothing a rough inversion result before re-inversion (smooth),
- inserting a local anomaly — conductor or resistor — as an ellipsoidal body (ellipsoid),
- normalising the model vector for statistical post-processing (standardise).

**Air, ocean, and any other fixed region are never modified**, regardless of
the chosen operation.  `fem.read_model` excludes them from the free vector
before it reaches the operation function; `fem.insert_model` writes their
canonical values (`AIR_RHO`, `OCEAN_RHO`) back unconditionally when
reconstructing the block file.

---

## Workflow

```
resistivity_block_iterX.dat   [+ mesh.dat for "smooth" / "ellipsoid" / "brick" / "wmean"]
        |
        v  fem.read_model(model_trans="log10")
  log10(rho)  [free regions only — air/ocean/fixed excluded]
        |
        v  _OPERATIONS[OPERATION](log_m)
  log10(rho)  [modified, free regions only]
        |
        v  fem.insert_model(template=MODEL_IN, ...)
                 writes air_rho / ocean_rho unconditionally
resistivity_block_edited.dat
```

---

## Configuration

All user-editable settings live in the **Configuration** block near the top of
the script.  No command-line arguments are used; edit the script directly.

### Paths

| Variable | Default | Description |
|---|---|---|
| `WORK_DIR` | `/home/vrath/Py4MTX/py4mt/data/example/` | Working directory |
| `MODEL_IN` | `resistivity_block_iter10.dat` | Source block file (also used as format template) |
| `MESH_FILE` | `mesh.dat` | Mesh file — required for `"wmean"`, `"smooth"`, `"ellipsoid"` and `"brick"`; ignored otherwise |
| `MODEL_OUT` | `resistivity_block_edited.dat` | Output block file; set equal to `MODEL_IN` to overwrite in-place |

### Ocean / fixed-region handling

| Variable | Default | Description |
|---|---|---|
| `OCEAN` | `None` | `None` = auto-infer; `True` / `False` = force |
| `AIR_RHO` | `1e9` Ohm·m | Written for region 0 (air) |
| `OCEAN_RHO` | `0.25` Ohm·m | Written for region 1 when treated as ocean |

Ocean auto-inference: region 1 is treated as ocean if `flag == 1` **and**
rho ≤ 1 Ohm·m.  Override with `OCEAN = True / False` when unreliable.

### Operation selection

| Variable | Default | Description |
|---|---|---|
| `OPERATION` | `"mean"` | Key selecting the operation (see table below) |
| `OP_FILL_VALUE` | `2.0` | Constant fill value in log10(Ohm·m) — `"fill"` |
| `OP_CLIP_MIN` | `0.0` | Lower bound in log10(Ohm·m) — `"clip"` |
| `OP_CLIP_MAX` | `4.0` | Upper bound in log10(Ohm·m) — `"clip"` |
| `OP_SHIFT_VALUE` | `0.5` | Additive offset in log10(Ohm·m) — `"shift"` |
| `OP_SMOOTH_SIGMA` | `5000.0` | Gaussian length scale σ in metres — `"smooth"` |
| `OP_SMOOTH_K` | `100` | K nearest neighbours per region — `"smooth"` |
| `OP_SMOOTH_MAX_GB` | `4.0` | RAM cap (GiB) for fallback dense path when SciPy absent — `"smooth"` |

### Body lists (ellipsoid and brick)

Both `"ellipsoid"` and `"brick"` read from a Python list of body dicts.
Each body is applied in order; later bodies overwrite earlier ones where
masks overlap, allowing layered construction.

**Ellipsoid** — `OP_ELLIPSOID_BODIES`  
**Brick** — `OP_BRICK_BODIES`

Every body dict shares the same five keys:

| Key | Type | Description |
|---|---|---|
| `mode` | str | `"replace"` — set to absolute value; `"add"` — add signed offset |
| `value` | float | log10(Ohm·m) — absolute resistivity or signed offset |
| `center` | [x, y, z] | Body centre in metres; z positive-down |
| `axes` | [a, b, c] | Ellipsoid: semi-axes in metres. Brick: half-extents in metres. All > 0 |
| `angles` | [α, β, γ] | ZYX rotation angles in degrees (yaw, pitch, roll). `[0,0,0]` = no rotation |

Example — two ellipsoids:

```python
OP_ELLIPSOID_BODIES = [
    dict(mode="replace", value=0.0,
         center=[0., 0., 5000.], axes=[10000., 10000., 5000.], angles=[0., 0., 0.]),
    dict(mode="add", value=-1.0,
         center=[5000., 0., 8000.], axes=[3000., 3000., 3000.], angles=[30., 0., 0.]),
]
```

Example — one rotated brick:

```python
OP_BRICK_BODIES = [
    dict(mode="replace", value=3.0,
         center=[0., 0., 10000.], axes=[8000., 4000., 3000.], angles=[45., 0., 0.]),
]
```

---

## Available operations

All operations act on the **free** log10(rho) vector only.  Air, ocean, and
flag-fixed regions are excluded before the operation and restored afterwards.

| Key | Effect | Typical use |
|---|---|---|
| `"fill"` | m ← OP_FILL_VALUE | Constant half-space starting model |
| `"mean"` | m ← mean(m) | Homogeneous model at arithmetic-mean resistivity |
| `"wmean"` | m ← Σ(wₖ mₖ)/Σwₖ, wₖ=1/Vₖ | Same, but small cells outweigh large cells |
| `"median"` | m ← median(m) | Same, robust to outlier regions |
| `"clip"` | m ← clamp(m, min, max) | Enforce physical bounds |
| `"shift"` | m ← m + δ | Global resistivity offset (e.g. +0.5 → ×3.2 in Ohm·m) |
| `"smooth"` | m̃ᵢ = Σⱼ∈KNN wᵢⱼ mⱼ / Σ wᵢⱼ | Spatial low-pass; reduce artefacts before re-inversion |
| `"ellipsoid"` | m[inside] replace/+= value | Insert rotated ellipsoidal body/bodies |
| `"brick"` | m[inside] replace/+= value | Insert rotated rectangular prism body/bodies |
| `"standardise"` | m ← (m − μ) / σ | Zero-mean / unit-variance; combine with `"clip"` |

### Ellipsoid (`"ellipsoid"`) and Brick (`"brick"`)

Both operations share the same multi-body engine (`_apply_bodies`).  Regions
whose centroid falls inside at least one body are modified; all others are
unchanged.  Bodies are applied in order — later entries win on overlap.

**Shared ZYX rotation convention:**

```
Local frame:  p' = R^T (p - center)
R = Rz(alpha) @ Ry(beta) @ Rx(gamma)   [intrinsic ZYX / yaw-pitch-roll]
```

**Ellipsoid mask** (quadratic form in local frame):

```
(x'/a)^2 + (y'/b)^2 + (z'/c)^2 <= 1
```

**Brick mask** (box test in local frame, half-extents a, b, c):

```
|x'| <= a  AND  |y'| <= b  AND  |z'| <= c
```

**Modes** (per body):

| Mode | Effect |
|---|---|
| `"replace"` | m[inside] = value (absolute log10(Ohm·m)) |
| `"add"` | m[inside] += value (signed offset in log10 space) |

A runtime warning is printed for any body whose mask is empty (geometry
mismatch or body too small for the mesh resolution).

### Inverse-volume-weighted mean (`"wmean"`)

Computes a single scalar value to replace all free-region values, weighted by
the inverse of each region's total volume:

```
w_k = 1 / V_k
m_tilde = ( sum_k  w_k * m_k ) / ( sum_k  w_k )
```

In a typical FEMTIC mesh the volume ratio between the largest background
region and the smallest near-surface region can exceed 10⁶.  The arithmetic
`"mean"` is dominated by those large deep cells; `"wmean"` gives proportionally
more influence to the fine cells in the target depth range — which is usually
what is wanted when resetting a model to a representative starting value.

Requires `MESH_FILE`.  No additional parameters.

### Spatial smoothing (`"smooth"`)

Isotropic Gaussian kernel evaluated between free-region centroids, restricted
to each region's K nearest neighbours:

```
w_ij = exp( -||c_i - c_j||^2 / (2 sigma^2) )

m_tilde_i = ( sum_{j in KNN(i)}  w_ij * m_j ) / ( sum_{j in KNN(i)} w_ij )
```

**Why K-NN and not `query_ball_point`** — variable-length neighbour lists from
`query_ball_point` cause a segfault at ~125 k regions because the combined
lists exhaust RAM before any computation begins.  `cKDTree.query(k=K)` returns
a **fixed-shape** `(n, K)` array; memory is exactly `n × K × 8` bytes —
predictable and bounded.  For K = 100 and n = 125 k that is ~100 MB.
Computation is fully vectorised (no Python loop over regions).

If SciPy is unavailable the code falls back to a chunked dense path capped at
`OP_SMOOTH_MAX_GB` gigabytes (no K restriction in that path).

| Parameter | Default | Effect |
|---|---|---|
| `OP_SMOOTH_SIGMA` | `5000.0` m | Gaussian length scale σ |
| `OP_SMOOTH_K` | `100` | Number of nearest neighbours per region |
| `OP_SMOOTH_MAX_GB` | `4.0` GiB | RAM cap for fallback chunked path |

**Choosing σ** — 1–2× the typical element edge length at the target depth.

**Choosing K** — K should cover the neighbourhood within roughly 2–3σ.
A rough guide: if the average inter-region spacing is `d` metres,
set `K ≈ (2σ/d)³ × π/6` (volume of sphere / average region volume).
Start with K = 50–200 and check that `dist[:, -1].max() < 3σ` (the
K-th neighbour is within 3σ for all regions).  If not, increase K.
Regions beyond K get zero weight regardless of distance.

---

## Mesh requirement for `"smooth"`, `"ellipsoid"`, `"brick"`, and `"wmean"`

These operations read `MESH_FILE` (same `mesh.dat` used during inversion) to
compute free-region centroids (and volumes for `"wmean"`) via `_build_region_geometry`.  The
element→region mapping is obtained from `MODEL_IN` via
`fem._read_resistivity_block_struct`.  Mesh loading is shared: if a future
operation also needs geometry, add its key to `_NEEDS_MESH` in the main block.

---

## Adding new operations

```python
def _op_my_transform(m: np.ndarray) -> np.ndarray:
    """Brief description."""
    return ...                       # same shape as m

_OPERATIONS["my_transform"] = _op_my_transform
```

Operations needing geometry (mesh / centroids / volumes) should add their key to
`_NEEDS_MESH` and populate a dedicated context dict in the `if OPERATION in
_NEEDS_MESH:` block, following the `_ellipsoid_ctx` / `_smooth_ctx` pattern.

---

## Dependencies

| Package | Role |
|---|---|
| NumPy | Array operations, rotation matrices, Gaussian kernel |
| `femtic` (Py4MTX) | `read_model`, `insert_model`, `read_femtic_mesh`, `_read_resistivity_block_struct` |
| `util` (Py4MTX) | `print_title` |
| `version` (Py4MTX) | `versionstrg` |

Environment variables `PY4MTX_ROOT` and `PY4MTX_DATA` must be set.
SciPy is **not** required.

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-04-30 | vrath / Claude Sonnet 4.6 | Created, modelled on `femtic_gst_prep.py` |
| 2026-04-30 | vrath / Claude Sonnet 4.6 | Added `"smooth"` (Gaussian region-centroid weighting) |
| 2026-04-30 | vrath / Claude Sonnet 4.6 | Added `"fill"`; clarified air/ocean safety guarantee |
| 2026-04-30 | vrath / Claude Sonnet 4.6 | Added `"ellipsoid"` (replace/add, rotated ZYX geometry); refactored mesh loading into shared `_NEEDS_MESH` block |
| 2026-04-30 | vrath / Claude Sonnet 4.6 | Added `"wmean"` (inverse-volume-weighted mean); refactored `_build_region_centroids` → `_build_region_geometry` (centroids + volumes in one pass) |
| 2026-05-03 | vrath / Claude Sonnet 4.6 | Added `"brick"` (rotated rectangular prism); refactored `_op_ellipsoid` into shared `_apply_bodies` engine supporting lists of bodies for both `"ellipsoid"` and `"brick"` |
