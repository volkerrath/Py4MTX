# `joint_coupling.py` — ADMM Coupling Wrappers

ADMM-compatible wrappers for all joint regularisation strategies.

Imports from the four self-contained `coupling_*.py` sibling modules.

---

## Classes

### `FCMCouplingMesh`

FCM coupling with a dedicated coupling mesh. `z` is defined on `mesh_z`,
which may differ from the MT and seismic meshes. Each `update_z` call
interpolates the models to `mesh_z`, runs FCM there, updates coupling-mesh
duals as instance state, and returns `z` on `mesh_z`. The driver then
interpolates `z` back to the physics meshes when building the model-update
RHS.

```python
FCMCouplingMesh(K, N_c, mesh_mt, mesh_sv, mesh_z, *,
                beta=1.0, q=2.0, w_mt=0.5, w_sv=0.5, n_inner=2,
                interp_method="nearest", interp_power=2.0)
```

| Parameter | Description |
|---|---|
| `K` | Number of FCM clusters |
| `N_c` | Number of cells on `mesh_z` |
| `mesh_mt`, `mesh_sv`, `mesh_z` | Mesh objects (must expose `cell_centers`) |
| `beta` | FCM coupling weight |
| `q` | Fuzziness exponent |
| `w_mt`, `w_sv` | Relative ADMM weights |
| `n_inner` | FCM sweeps per iteration |
| `interp_method` | `"nearest"`, `"idw"`, or `"rbf"` |
| `interp_power` | IDW exponent |

**Instance state:** `U` (memberships), `c` (centroids), `y_mt_c`,
`y_sv_c` (coupling-mesh duals) are maintained across iterations.

`report` keys: `r_mt_c`, `r_sv_c` (residual norms on coupling mesh),
`fcm_var`.

**Use this when** the MT and seismic meshes differ significantly in
resolution or geometry. Use `FCMCoupling` when all three meshes are the
same (or close enough that the distinction doesn't matter).

---

### `FCMCoupling`

Fuzzy C-Means latent petrophysical field. `z` is updated in closed form
by minimising the FCM objective together with the ADMM quadratic penalty.
Memberships `U` and centroids `c` are refined for `n_inner` sweeps per
ADMM iteration and stored as instance state.

```python
FCMCoupling(K, N, *, beta=1.0, q=2.0, w_mt=0.5, w_sv=0.5, n_inner=2)
```

| Parameter | Description |
|---|---|
| `K` | Number of FCM clusters |
| `N` | Number of model cells |
| `beta` | FCM coupling weight |
| `q` | Fuzziness exponent (must be > 1) |
| `w_mt`, `w_sv` | Relative ADMM weights (should sum to 1) |
| `n_inner` | FCM sweeps per ADMM outer iteration |

`report` keys: `r_mt`, `r_sv` (residual norms), `fcm_var` (weighted
intra-cluster variance).

---

### `CrossGradientCoupling`

Structural cross-gradient proximal coupling. `z` is computed as the
weighted ADMM consensus mean, then a proximal shrinkage step nudges it
toward the cross-gradient null-space.

```python
CrossGradientCoupling(mesh, *, cg_weight=0.1, w_mt=0.5, w_sv=0.5)
```

| Parameter | Description |
|---|---|
| `mesh` | `GradientMesh` instance (from `coupling_crossgrad`) |
| `cg_weight` | Proximal weight; 0 disables the shrinkage step |
| `w_mt`, `w_sv` | Consensus mean weights |

`report` keys: `cg_rms` (RMS cross-gradient magnitude on the coupling mesh).

---

### `GramianCoupling`

Structural Gramian coupling (Zhdanov 2012). `z` is the plain ADMM
weighted mean. The Gramian gradient is exposed via `.gradient()` so it
can be injected into the physics-model RHS by the caller.

```python
GramianCoupling(gramian, *, w_mt=0.5, w_sv=0.5)
```

| Parameter | Description |
|---|---|
| `gramian` | `StructuralGramian` instance (from `coupling_gramian`) |
| `w_mt`, `w_sv` | Consensus mean weights |

Additional method: `gradient(m_mt, m_sv) -> (grad_mt, grad_sv)`.

`report` keys: all keys from `StructuralGramian.report` plus
`r_consensus` (distance from consensus mean to `z`).

---

### `MutualInfoCoupling`

Mutual-information / entropy coupling. `z` is the plain ADMM weighted
mean. The MI gradient is exposed via `.gradient()` for solver injection.

```python
MutualInfoCoupling(mi_coupling, *, w_mt=0.5, w_sv=0.5)
```

| Parameter | Description |
|---|---|
| `mi_coupling` | `MutualInformationCoupling` instance (from `coupling_entropy`) |
| `w_mt`, `w_sv` | Consensus mean weights |

Additional method: `gradient(m_mt, m_sv) -> (grad_mt, grad_sv)`.

`report` keys: all keys from `MutualInformationCoupling.report` plus
`r_consensus`.

---

### `CombinedCoupling`

Weighted combination of any number of coupling strategies. `update_z`
returns the weighted average of each component's `z` update. `report`
merges all component diagnostics, prefixing keys with `c{i}_`.

```python
CombinedCoupling(components, *, weights=None)
```

| Parameter | Description |
|---|---|
| `components` | List of coupling objects (each must have `update_z`) |
| `weights` | Per-component weights; equal weights if `None` |

```python
coupling = CombinedCoupling(
    [FCMCoupling(K=3, N=N, beta=0.5),
     CrossGradientCoupling(mesh, cg_weight=0.05)],
    weights=[0.7, 0.3],
)
```

---

## Imports

```python
from coupling_fcm      import update_centroids, update_memberships, update_z_fcm
from coupling_crossgrad import compute_cross_gradient, cross_gradient_proximal_term
from coupling_gramian  import StructuralGramian
from coupling_entropy  import MutualInformationCoupling
```

---

## Authorship

Volker Rath (DIAS) — 2026-05-12, Claude Sonnet 4.6 (Anthropic)
