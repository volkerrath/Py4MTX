# Joint MT + Seismic Inversion — ADMM Coupling Package

**Python for Magnetotellurics** — py4mt / DIAS / Volker Rath

Alternating Direction Method of Multipliers (ADMM) driver and pluggable
coupling strategies for joint MT + seismic tomography inversion.

---

## File overview

| File | Role |
|---|---|
| `joint_admm_driver.py` | ADMM outer loop; coupling-agnostic |
| `joint_coupling.py` | ADMM-compatible wrappers for each strategy |
| `coupling_interp.py` | All mesh interpolation — shared by crossgrad, gramian, entropy |
| `coupling_fcm.py` | Fuzzy C-Means latent field — primitives |
| `coupling_crossgrad.py` | Cross-gradient + mesh operators |
| `coupling_gramian.py` | Structural Gramian |
| `coupling_entropy.py` | Mutual-information (entropy) coupling |

No subpackage imports, no `jointinv.*` namespace. The only cross-module
dependencies are: `coupling_crossgrad`, `coupling_gramian`, and
`coupling_entropy` all import from `coupling_interp`; `coupling_entropy`
also imports `_fd_gradient_magnitude` from `coupling_gramian`.

---

## Design

The driver calls a single method on the coupling object every iteration:

```
coupling.update_z(m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv) → z
```

An optional `report(m_mt, m_sv, z) → dict` is called when `verbose=True`.
This interface is the only contract between the driver and any coupling
strategy; switching strategies requires only changing which wrapper is
instantiated.

### ADMM update roles by strategy

| Strategy | How `z` is updated | Gradient coupling |
|---|---|---|
| FCMCoupling | Closed-form FCM + ADMM objective (same mesh) | Via `z` |
| FCMCouplingMesh | Same, but `z` on dedicated `mesh_z`; interp each iter | Via `z` |
| CrossGradient | ADMM mean + proximal shrinkage | Via `z` |
| Gramian | ADMM mean | `.gradient()` → injected into solvers |
| MutualInfo | ADMM mean | `.gradient()` → injected into solvers |
| CombinedCoupling | Weighted average of components | Per component |

---

## Quick-start examples

### FCM coupling (same mesh)

```python
from joint_admm_driver import admm_joint_mt_seis
from joint_coupling import FCMCoupling

coupling = FCMCoupling(K=4, N=len(m_mt0), beta=1.0, q=2.0, n_inner=3)

result = admm_joint_mt_seis(
    d_mt, d_sv, Wd_mt, Wd_sv, Wm_mt, Wm_sv,
    m_mt0, m_sv0, m_mt_ref, m_sv_ref,
    coupling,
    rho_mt=1.0, rho_sv=1.0, max_outer=50,
    solve_m_mt=my_mt_solver, solve_m_sv=my_seis_solver,
    apply_Gt_mt=my_mt_adjoint, apply_Gt_sv=my_seis_adjoint,
)
```

### FCM coupling (dedicated coupling mesh)

```python
from joint_coupling import FCMCouplingMesh

coupling = FCMCouplingMesh(
    K=4, N_c=len(mesh_z_cells),
    mesh_mt=mesh_mt, mesh_sv=mesh_sv, mesh_z=mesh_z,
    beta=1.0, interp_method="idw",
)
# z is returned on mesh_z; the driver interpolates it back to physics meshes
```

### Cross-gradient coupling

```python
from joint_coupling import CrossGradientCoupling
from coupling_crossgrad import StructuredGridMesh

mesh = StructuredGridMesh(shape=(nx, ny, nz), spacing=(dx, dy, dz))
coupling = CrossGradientCoupling(mesh, cg_weight=0.1)
```

### Gramian coupling

```python
from joint_coupling import GramianCoupling
from coupling_gramian import StructuralGramian, MultiscaleResampler, build_common_grid

common_coords = build_common_grid(mt_coords, seis_coords, dx=2000.0)
mt_to_g   = MultiscaleResampler(mt_coords,   common_coords, sigma=3000.0)
seis_to_g = MultiscaleResampler(seis_coords, common_coords, sigma=500.0)

gramian  = StructuralGramian(mt_to_g, seis_to_g,
                              beta=1.0, mode="gradient",
                              common_coords=common_coords)
coupling = GramianCoupling(gramian)
```

### Mutual-information coupling

```python
from joint_coupling import MutualInfoCoupling
from coupling_entropy import MutualInformationCoupling

mi_obj   = MutualInformationCoupling(mt_to_g, seis_to_g, beta=1.0,
                                      mode="rank", n_bins=32)
coupling = MutualInfoCoupling(mi_obj)
```

### Combined coupling

```python
from joint_coupling import CombinedCoupling, FCMCoupling, CrossGradientCoupling

coupling = CombinedCoupling(
    [FCMCoupling(K=3, N=N, beta=0.5),
     CrossGradientCoupling(mesh, cg_weight=0.05)],
    weights=[0.7, 0.3],
)
```

### Fixing one method

```python
# Hold MT fixed; optimise seismic + z only
result = admm_joint_mt_seis(..., coupling=coupling, fix_method="sv")
```

---

## Dependencies

| Package | Required by |
|---|---|
| NumPy | all |
| SciPy (`spatial.cKDTree`, `stats.norm`) | `coupling_gramian`, `coupling_entropy` |

---

## Authorship

Volker Rath (DIAS) — 2026-05-12, Claude Sonnet 4.6 (Anthropic)
