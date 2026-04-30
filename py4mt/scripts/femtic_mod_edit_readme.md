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
resistivity_block_iterX.dat   [+ mesh.dat for "smooth" / "ellipsoid"]
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
| `MESH_FILE` | `mesh.dat` | Mesh file — required for `"wmean"`, `"smooth"` and `"ellipsoid"`, ignored otherwise |
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
| `OP_SMOOTH_CUTOFF` | `4.0` | Cutoff radius in multiples of σ; beyond this weight = 0 — `"smooth"` |
| `OP_SMOOTH_MAX_GB` | `4.0` | RAM cap (GiB) for fallback dense path when SciPy absent — `"smooth"` |

### Ellipsoid parameters

| Variable | Default | Description |
|---|---|---|
| `OP_ELLIPSOID_MODE` | `"replace"` | `"replace"` or `"add"` — see below |
| `OP_ELLIPSOID_VALUE` | `0.0` | Value applied in log10(Ohm·m) |
| `OP_ELLIPSOID_CENTER` | `[0., 0., 5000.]` | Centre [x, y, z] in metres (z positive-down) |
| `OP_ELLIPSOID_AXES` | `[10000., 10000., 5000.]` | Semi-axes [a, b, c] in metres, must be > 0 |
| `OP_ELLIPSOID_ANGLES` | `[0., 0., 0.]` | ZYX rotation angles [α, β, γ] in degrees |

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
| `"smooth"` | m̃ᵢ = Σⱼ wᵢⱼ mⱼ / Σⱼ wᵢⱼ | Spatial low-pass; reduce artefacts before re-inversion |
| `"ellipsoid"` | m[inside] replace/+= value | Insert or adjust a local anomalous body |
| `"standardise"` | m ← (m − μ) / σ | Zero-mean / unit-variance; combine with `"clip"` |

### Ellipsoid (`"ellipsoid"`)

Regions whose **centroid** (mean of all element centroids for that region)
falls inside the rotated ellipsoid are modified; all others are unchanged.

**Geometry** — the ellipsoid is defined in the FEMTIC model frame (z positive
downward, same as the mesh coordinate system):

```
Quadratic form (in ellipsoid-local frame after ZYX rotation):
  (x'/a)^2 + (y'/b)^2 + (z'/c)^2 <= 1

Local frame: p' = R^T (p - center)
R = Rz(alpha) @ Ry(beta) @ Rx(gamma)   [intrinsic ZYX / yaw-pitch-roll]
```

**Modes:**

| Mode | Formula | Use case |
|---|---|---|
| `"replace"` | m[inside] = OP_ELLIPSOID_VALUE | Insert a body at a fixed absolute resistivity |
| `"add"` | m[inside] += OP_ELLIPSOID_VALUE | Nudge an existing anomaly up or down |

Examples:

```python
# Spherical conductor of 1 Ohm·m, radius 8 km, centred at 5 km depth
OP_ELLIPSOID_MODE   = "replace"
OP_ELLIPSOID_VALUE  = 0.0          # log10(1 Ohm·m)
OP_ELLIPSOID_CENTER = [0., 0., 5000.]
OP_ELLIPSOID_AXES   = [8000., 8000., 8000.]
OP_ELLIPSOID_ANGLES = [0., 0., 0.]

# Make an elongated NE-SW body one decade more resistive (add mode)
OP_ELLIPSOID_MODE   = "add"
OP_ELLIPSOID_VALUE  = 1.0          # +1 log10 unit
OP_ELLIPSOID_CENTER = [0., 0., 10000.]
OP_ELLIPSOID_AXES   = [20000., 5000., 4000.]
OP_ELLIPSOID_ANGLES = [45., 0., 0.]   # 45° yaw → NE-SW alignment
```

A runtime warning is printed if no free-region centroids fall inside the
ellipsoid (geometry mismatch or ellipsoid too small for the mesh resolution).

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

Isotropic Gaussian kernel evaluated between free-region centroids:

```
w_ij = exp( -||c_i - c_j||^2 / (2 sigma^2) )
m_tilde_i = ( sum_j  w_ij * m_j ) / ( sum_j  w_ij )
```

The self-weight (i = j) is always included.

**Memory-efficient implementation** — the naive dense approach allocates an
(n, n, 3) difference array (~350 GiB for n = 125 k), which is infeasible.
Two strategies are combined:

1. **Distance cutoff** (`OP_SMOOTH_CUTOFF`, default 4σ): neighbours beyond
   `cutoff × sigma` get zero weight (Gaussian < exp(-8) ≈ 0.03 %).
   Only the non-zero neighbourhood is ever computed per region.

2. **SciPy cKDTree** (preferred): `query_ball_point` returns the neighbour
   index list per region; weighted accumulation is row-by-row in O(n_nbrs)
   memory.  If SciPy is unavailable the code falls back to a chunked
   dense path capped at `OP_SMOOTH_MAX_GB` gigabytes.

| Parameter | Default | Effect |
|---|---|---|
| `OP_SMOOTH_SIGMA` | `5000.0` m | Gaussian length scale |
| `OP_SMOOTH_CUTOFF` | `4.0` (× σ) | Beyond this radius weight is zero |
| `OP_SMOOTH_MAX_GB` | `4.0` GiB | RAM cap for fallback chunked path |

**Choosing σ** — 1–2× the typical element edge length at the target depth.
**Choosing cutoff** — 4σ is accurate; 3σ is faster (weight < 1.1 % at boundary).

### Note on multiplicative scaling

Multiplying log10(rho) by a constant has no clean physical interpretation and
is not provided.  Use `"shift"` for global rescaling (e.g. `OP_SHIFT_VALUE =
1.0` → ×10 in Ohm·m), or `"ellipsoid"` in `"add"` mode for local rescaling.

---

## Mesh requirement for `"smooth"` and `"ellipsoid"`

Both operations read `MESH_FILE` (same `mesh.dat` used during inversion) to
compute free-region centroids via `_build_region_centroids`.  The
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
