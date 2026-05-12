# `coupling_interp.py` — Joint Mesh Interpolation

Consolidated mesh-to-mesh interpolation for the joint MT + seismic coupling
layer. All interpolation logic previously duplicated across `coupling_crossgrad`
and `coupling_gramian` now lives here.

Imported by: `coupling_crossgrad.py`, `coupling_gramian.py`, `coupling_entropy.py`.

---

## Contents

| Class / Function | Role |
|---|---|
| `ModelGrid` | Model parameter container (dataclass) |
| `build_common_grid` | Regular voxel common grid spanning two model extents |
| `interpolate_mesh_to_mesh` | Simple point-query interpolation (nearest / IDW / RBF) |
| `MultiscaleResampler` | KD-tree IDW + Gaussian pre-smoothing + adjoint |

---

## `ModelGrid`

Dataclass container for a geophysical model.

```python
ModelGrid(coords, values, name="model")
```

| Field | Type | Description |
|---|---|---|
| `coords` | (N, 3) array | Cell centroids [m], z positive-down |
| `values` | (N,) array | Model parameter (log₁₀(Ω·m) or Vp [km/s]) |
| `name` | str | Label, e.g. `"MT"` or `"seismic"` |

Property `n` returns the number of cells.

---

## `build_common_grid`

```python
common_coords = build_common_grid(coords_a, coords_b, *, dx,
                                   extent=None, out=True)
```

Builds a regular voxel grid with spacing `dx` [m] covering the bounding
box of both input coordinate arrays. Pass `extent=[xmin,xmax,ymin,ymax,zmin,zmax]`
to override the automatic bounding box.

Returns `ndarray (M, 3)` of voxel centroids.

---

## `interpolate_mesh_to_mesh`

Simple, pure-NumPy mesh-to-mesh interpolation. Used internally by
`coupling_crossgrad` and suitable for small meshes (N ≲ 10 000).

```python
values_dst = interpolate_mesh_to_mesh(
    src_mesh, dst_mesh, values,
    method="nearest",   # or "idw" or "rbf"
    p=2.0,              # IDW exponent
    rbf_sigma=1.0,      # RBF width [same units as cell_centers]
)
```

Meshes must expose `cell_centers` (ndarray) or `get_cell_centers()`.

| Method | Description | Complexity |
|---|---|---|
| `"nearest"` | Nearest-neighbour lookup | O(N_src · N_dst) |
| `"idw"` | Inverse distance weighting, exponent `p` | O(N_src · N_dst) |
| `"rbf"` | Gaussian RBF, width `rbf_sigma` | O(N_src · N_dst) |

For large meshes use `MultiscaleResampler` instead.

---

## `MultiscaleResampler`

KD-tree IDW interpolation with optional Gaussian pre-smoothing and an
exact adjoint. Preferred when:
- Meshes have very different resolutions (use `sigma` to balance).
- A gradient chain rule is needed (Gramian, MI coupling).
- N is large (O(k log N) query after O(N log N) construction).

```python
resampler = MultiscaleResampler(
    source_coords, target_coords,
    k=8,          # IDW nearest neighbours
    p=2.0,        # IDW distance exponent
    sigma=3000.,  # Gaussian pre-smoothing [m]; 0 = off
    K_smooth=50,  # neighbours for smoothing kernel
)

u     = resampler(m)           # forward:  (N_src,) → (N_tgt,)
g_src = resampler.adjoint(gu)  # adjoint:  (N_tgt,) → (N_src,)
```

### Resolution balancing

Use a wider `sigma` for the coarser-resolution model (MT) and a narrower
`sigma` for the finer-resolution model (seismic), so that both have
comparable effective resolution on the common grid before the Gramian or
MI coupling is evaluated.

```python
mt_to_g   = MultiscaleResampler(mt_coords,   common_coords, sigma=3000.)
seis_to_g = MultiscaleResampler(seis_coords, common_coords, sigma=500.)
```

### Adjoint

The adjoint satisfies `<R u, v> = <u, R^T v>` exactly, enabling correct
gradient back-propagation through the interpolation:

```
∂Φ/∂m_src = R^T (∂Φ/∂u)
```

Requires `scipy.spatial.cKDTree`.

---

## Module dependency graph

```
coupling_interp.py
       ├── coupling_crossgrad.py   (interpolate_mesh_to_mesh)
       ├── coupling_gramian.py     (ModelGrid, MultiscaleResampler,
       │                            build_common_grid)
       └── coupling_entropy.py     (ModelGrid, MultiscaleResampler)
```

`coupling_gramian` also exports `_fd_gradient_magnitude` to
`coupling_entropy` — the only remaining cross-module dependency outside
of `coupling_interp`.

---

## Dependencies

NumPy (all). `scipy.spatial.cKDTree` (`MultiscaleResampler` only).

---

## Authorship

Volker Rath (DIAS) — 2026-05-12, Claude Sonnet 4.6 (Anthropic)
