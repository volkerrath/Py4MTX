# `coupling_fcm.py` — Fuzzy C-Means Coupling Primitives

Self-contained FCM clustering and latent field update for joint ADMM inversion.

No subpackage imports. Inlined from `fcm/fcm.py` and `fcm/latent_field.py`
(the `crossgrad/` copies are identical).

---

## Functions

### `update_centroids(z, U, q)`

Update FCM cluster centroids.

```
c_k = (Σ_i u_ik^q z_i) / (Σ_i u_ik^q)
```

| Argument | Shape | Description |
|---|---|---|
| `z` | (N,) | Latent petrophysical field |
| `U` | (N, K) | Membership matrix |
| `q` | float | Fuzziness exponent |

Returns `ndarray (K,)`.

---

### `update_memberships(z, c, q)`

Update FCM membership matrix using the standard FCM formula. Handles
exact centroid matches (zero distance) without division by zero.

```
u_ik = 1 / Σ_j (d_ik / d_ij)^(2/(q-1))
```

| Argument | Shape | Description |
|---|---|---|
| `z` | (N,) | Latent field |
| `c` | (K,) | Centroids |
| `q` | float | Fuzziness exponent (must be > 1) |

Returns `ndarray (N, K)` with row sums = 1.

---

### `compute_distances(z, c)`

Squared distances between latent field values and centroids. Diagnostic
use only; not called internally by `update_z_fcm`.

Returns `ndarray (N, K)`.

---

### `update_z_fcm(m_mt, m_sv, y_mt, y_sv, U, c, beta, rho_mt, rho_sv, q, w_mt=0.5, w_sv=0.5)`

Closed-form update of the latent petrophysical field `z`. Minimises:

```
β Σ_i Σ_k u_ik^q (z_i − c_k)²
+ (ρ_mt/2) ‖m_mt − z + y_mt/ρ_mt‖²
+ (ρ_sv/2) ‖m_sv − z + y_sv/ρ_sv‖²
```

The solution is analytic (element-wise):

```
z_i = (2β Σ_k u_ik^q c_k  +  w_mt ρ_mt v_mt_i  +  w_sv ρ_sv v_sv_i)
    / (2β Σ_k u_ik^q        +  w_mt ρ_mt         +  w_sv ρ_sv)
```

where `v_mt = m_mt + y_mt/ρ_mt`, `v_sv = m_sv + y_sv/ρ_sv`.

| Argument | Description |
|---|---|
| `m_mt`, `m_sv` | ndarray (N,) — model vectors |
| `y_mt`, `y_sv` | ndarray (N,) — dual variables |
| `U` | ndarray (N, K) — membership matrix |
| `c` | ndarray (K,) — centroids |
| `beta` | float — FCM coupling weight |
| `rho_mt`, `rho_sv` | float — ADMM penalty parameters |
| `q` | float — fuzziness exponent |
| `w_mt`, `w_sv` | float — relative ADMM weights (sum to 1) |

Returns `ndarray (N,)`.

---

### `update_z_fcm_on_mesh(m_mt_c, m_sv_c, y_mt_c, y_sv_c, U, c, beta, rho_mt, rho_sv, q, w_mt=0.5, w_sv=0.5)`

Identical in form to `update_z_fcm` but makes explicit that all inputs
have already been interpolated to a dedicated coupling mesh before the
call. Used by `FCMCouplingMesh` in `joint_coupling.py`.

The distinction matters for heterogeneous-mesh configurations: `m_mt`
and `m_sv` live on their native physics meshes, whereas `z`, `U`, `c`,
and the duals all live on `mesh_z`.

Returns `ndarray (N_c,)`.

---

## Usage

These primitives are called by `FCMCoupling` in `joint_coupling.py`.
They can also be used directly for custom FCM update schedules.

```python
from coupling_fcm import update_centroids, update_memberships, update_z_fcm

z = update_z_fcm(m_mt, m_sv, y_mt, y_sv, U, c, beta, rho_mt, rho_sv, q)
c = update_centroids(z, U, q)
U = update_memberships(z, c, q)
```

---

## Dependencies

NumPy only.

---

## Authorship

Volker Rath (DIAS) — 2026-05-12, Claude Sonnet 4.6 (Anthropic)
