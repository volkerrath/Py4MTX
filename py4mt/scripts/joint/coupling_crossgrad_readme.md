# `coupling_crossgrad.py` — Cross-Gradient Structural Coupling

Self-contained module providing mesh gradient operators, mesh-to-mesh
interpolation, cross-gradient computation, and an ADMM proximal term.

Inlined from `crossgrad/cross_gradient.py`, `crossgrad/mesh_operators.py`,
and `crossgrad/interpolation.py`.

---

## Mesh interface

### `GradientMesh` (abstract base class)

All mesh objects passed to `compute_cross_gradient` must inherit from
`GradientMesh` or implement the same interface:

```python
mesh.grad(m: ndarray (N,)) -> ndarray (N, dim)
```

### `StructuredGridMesh`

Finite-difference gradient on a regular tensor grid (2-D or 3-D).
Internal points use central differences; boundary points use one-sided.

```python
StructuredGridMesh(shape, spacing)
```

| Parameter | Description |
|---|---|
| `shape` | tuple of int — `(nx,)`, `(nx, ny)`, or `(nx, ny, nz)` |
| `spacing` | tuple of float — `(dx,)`, `(dx, dy)`, or `(dx, dy, dz)` |

```python
mesh = StructuredGridMesh(shape=(40, 40, 20), spacing=(500., 500., 250.))
g    = mesh.grad(m)   # (N, 3)
```

### `UnstructuredMesh`

Wraps a user-supplied gradient callable. Use when the mesh and gradient
operator come from an external library (FEMTIC, SimPEG, Devito, etc.).

```python
UnstructuredMesh(grad_operator, dim, N=None)
```

| Parameter | Description |
|---|---|
| `grad_operator` | callable `(N,) -> (N, dim)` |
| `dim` | int — spatial dimension |
| `N` | int or None — cell count for input validation |

---

## Interpolation

### `interpolate_mesh_to_mesh(src_mesh, dst_mesh, values, method="nearest", p=2.0, rbf_sigma=1.0)`

Map cell-centred values from one mesh to another. Meshes must expose
`cell_centers` (ndarray) or `get_cell_centers()`.

| `method` | Description |
|---|---|
| `"nearest"` | Nearest-neighbour (exact at source nodes) |
| `"idw"` | Inverse distance weighting with exponent `p` |
| `"rbf"` | Gaussian RBF with width `rbf_sigma` |

Returns `ndarray (N_dst,)`.

---

## Cross-gradient

### `compute_cross_gradient(m_mt, mesh_mt, m_sv, mesh_sv, coupling_mesh=None, interp_method="nearest", interp_power=2.0)`

Compute the cross-gradient field:

```
X = ∇m_mt × ∇m_sv      (N_c, 3)
```

When MT and seismic live on different meshes, one or both models are
interpolated to `coupling_mesh` before gradient computation.
`coupling_mesh` defaults to `mesh_sv`. A third independent mesh can be
passed if desired.

Returns `(X, coupling_mesh)`.

---

### `cross_gradient_proximal_term(z, m_target, weight)`

ADMM-compatible proximal shrinkage:

```
z ← (z + weight · m_target) / (1 + weight)
```

Nudges the consensus variable `z` toward `m_target` (typically `m_mt`).
Setting `weight = 0` leaves `z` unchanged.

Returns `ndarray (N,)`.

---

## Usage example

```python
from coupling_crossgrad import StructuredGridMesh, compute_cross_gradient

mesh = StructuredGridMesh(shape=(50, 50, 25), spacing=(1000., 1000., 500.))

X, _ = compute_cross_gradient(m_mt, mesh, m_sv, mesh)
cg_rms = float(np.sqrt(np.mean(np.sum(X**2, axis=-1))))
print(f"Cross-gradient RMS: {cg_rms:.3e}")
```

---

## Dependencies

NumPy only. `scipy.spatial.cKDTree` is not used here; see
`coupling_gramian` for KD-tree-based operations.

---

## Authorship

Volker Rath (DIAS) — 2026-05-12, Claude Sonnet 4.6 (Anthropic)
