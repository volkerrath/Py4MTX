# `coupling_gramian.py` — Structural Gramian Coupling

Self-contained module for the structural Gramian constraint coupling
MT and seismic models via a shared common grid.

Inlined from `gramian/modules/model_interp.py` and
`gramian/modules/joint_gramian.py` (the `entropy/modules/` copies are
identical). Also exports `MultiscaleResampler` and `_fd_gradient_magnitude`
for use by `coupling_entropy.py`.

**Reference:** Zhdanov, M. S., Gribenko, A. V., & Wilson, G. (2012).
*Multinary inversion for tunnel detection*. GRL 39, L09301.
https://doi.org/10.1029/2012GL051233

---

## Grid infrastructure

### `ModelGrid`

Dataclass container for a geophysical model parameterisation.

```python
ModelGrid(coords, values, name="model")
```

| Field | Type | Description |
|---|---|---|
| `coords` | (N, 3) array | Cell centroids [m], z positive-down |
| `values` | (N,) array | Model parameter (log₁₀(Ω·m) or Vp [km/s]) |
| `name` | str | Label |

Property `n` returns the cell count.

---

### `build_common_grid(coords_a, coords_b, *, dx, extent=None, out=True)`

Build a regular voxel common grid covering the bounding box of both
input coordinate arrays.

| Parameter | Description |
|---|---|
| `coords_a`, `coords_b` | (N, 3) cell centroid arrays |
| `dx` | float — voxel edge length [m] |
| `extent` | `[xmin, xmax, ymin, ymax, zmin, zmax]` or `None` (auto) |
| `out` | bool — print node count |

Returns `ndarray (M, 3)` of voxel centroids.

---

### `MultiscaleResampler`

IDW interpolation from a source grid to a target grid with optional
Gaussian pre-smoothing. Use a wider `sigma` for the coarser method (MT)
and a narrower `sigma` for the finer one (seismic) so that both
representations have comparable effective resolution before the Gramian
is computed (Tu & Zhdanov 2021, GJI 226, 1058–1085).

```python
MultiscaleResampler(source_coords, target_coords, *,
                    k=8, p=2.0, sigma=0.0, K_smooth=50)
```

| Parameter | Description |
|---|---|
| `source_coords` | (N_src, 3) |
| `target_coords` | (N_tgt, 3) |
| `k` | IDW nearest neighbours |
| `p` | IDW distance exponent |
| `sigma` | Gaussian pre-smoothing length scale [m]; 0 = off |
| `K_smooth` | Neighbours for smoothing kernel |

**Methods:**

`resampler(values)` — forward interpolation `(N_src,) → (N_tgt,)`.

`resampler.adjoint(g_tgt)` — adjoint interpolation `(N_tgt,) → (N_src,)`
for gradient chain rule.

Requires `scipy.spatial.cKDTree`.

---

## Gram matrix utilities

### `gram_matrix(u, v)`

2×2 Gram matrix of two real vectors:

```
G = [[u·u,  u·v],
     [u·v,  v·v]]
```

Returns `ndarray (2, 2)`.

### `gramian(u, v)`

`det(G) = (u·u)(v·v) − (u·v)² ≥ 0`. Zero iff `u ∥ v`.

### `gramian_gradient(u, v)`

Gradient of `det(G)` w.r.t. `u` and `v`:

```
∂det/∂u = 2(v·v)u − 2(u·v)v
∂det/∂v = 2(u·u)v − 2(u·v)u
```

Returns `(grad_u, grad_v)`, each `ndarray (N,)`.

---

## `StructuralGramian`

Full coupling class.

```python
StructuralGramian(mt_to_g, seis_to_g, *,
                  beta=1.0, mode="gradient",
                  common_coords, k_nn=6)
```

| Parameter | Description |
|---|---|
| `mt_to_g` | `MultiscaleResampler` — MT grid → common grid |
| `seis_to_g` | `MultiscaleResampler` — seismic grid → common grid |
| `beta` | Gramian weight β |
| `mode` | Attribute transform applied before Gramian |
| `common_coords` | (M, 3) common grid centroids |
| `k_nn` | Neighbours for FD gradient/Laplacian |

**`mode` options:**

| Value | Attribute T(m) |
|---|---|
| `"value"` | raw resampled values |
| `"gradient"` | FD gradient magnitude ‖∇m‖ |
| `"laplacian"` | FD Laplacian |

**Methods:**

`value(m_mt, m_seis) → float` — objective β · det(Gram(T(u), T(v))).

`gradient(m_mt, m_seis, eps=1e-5) → (grad_mt, grad_seis)` — gradients
on native model grids. For `"value"` mode the chain rule is exact; for
`"gradient"` and `"laplacian"` modes the diff-op Jacobian is approximated
by finite differences.

`report(m_mt, m_seis) → dict` — keys: `gramian` (det), `correlation`,
`gram_matrix`.

---

## Usage example

```python
from coupling_gramian import (
    build_common_grid, MultiscaleResampler, StructuralGramian
)

common_coords = build_common_grid(mt_coords, seis_coords, dx=2000.)
mt_to_g   = MultiscaleResampler(mt_coords,   common_coords, sigma=3000.)
seis_to_g = MultiscaleResampler(seis_coords, common_coords, sigma=500.)

gramian = StructuralGramian(
    mt_to_g, seis_to_g,
    beta=1.0, mode="gradient",
    common_coords=common_coords,
)

phi          = gramian.value(m_mt, m_seis)
g_mt, g_seis = gramian.gradient(m_mt, m_seis)
```

---

## Dependencies

NumPy, `scipy.spatial.cKDTree`.

---

## Authorship

Volker Rath (DIAS) — 2026-05-12, Claude Sonnet 4.6 (Anthropic)
