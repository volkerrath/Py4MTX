# Joint Inversion Scripts — Catalogue

**py4mt / py4mtx** — ADMM-based joint MT + seismic tomography inversion.
All scripts live in `scripts/joint/`.

---

## Driver and coupling wrappers

| File | Role |
|------|------|
| `joint_admm_driver.py` | ADMM outer loop; coupling-agnostic.  Calls `coupling.update_z(m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv) → z` every iteration. |
| `joint_coupling.py` | ADMM-compatible wrapper classes for each coupling strategy; provides a uniform interface to the driver. |

---

## Coupling strategies

| File | Strategy |
|------|----------|
| `coupling_crossgrad.py` | Cross-gradient structural coupling + mesh operators (`StructuredGridMesh`); ADMM mean + proximal shrinkage `z` update. |
| `coupling_gramian.py` | Structural Gramian (`StructuralGramian`, `MultiscaleResampler`, `build_common_grid`); injects gradient into solvers. |
| `coupling_entropy.py` | Mutual-information (entropy) coupling; ranks-based or histogram MI; imports `_fd_gradient_magnitude` from `coupling_gramian`. |
| `coupling_fcm.py` | Fuzzy C-Means latent-field coupling; closed-form FCM + ADMM objective; supports a dedicated coupling mesh (`FCMCouplingMesh`). |
| `coupling_interp.py` | Shared mesh interpolation (IDW, nearest-neighbour) used by crossgrad, gramian, and entropy. |

---

## Supporting files

| File | Role |
|------|------|
| `inversion_state.py` | Shared inversion state container passed between driver and coupling objects. |
| `mt_fwd.py` | MT forward-model wrapper called by the ADMM driver. |
| `seistomo_fwd.py` | Seismic tomography forward-model wrapper called by the ADMM driver. |
| `seistomo_prep.py` | Preprocessing utilities for seismic tomography data and meshes. |

---

## ADMM update roles by strategy

| Strategy | `z` update | Gradient coupling |
|---|---|---|
| `FCMCoupling` | Closed-form FCM + ADMM objective (same mesh) | Via `z` |
| `FCMCouplingMesh` | Same, but `z` on dedicated `mesh_z`; interpolated each iter | Via `z` |
| `CrossGradientCoupling` | ADMM mean + proximal shrinkage | Via `z` |
| `GramianCoupling` | ADMM mean | `.gradient()` injected into solvers |
| `MutualInfoCoupling` | ADMM mean | `.gradient()` injected into solvers |
| `CombinedCoupling` | Weighted average of components | Per component |

---

## Dependencies

| Package | Required by |
|---------|-------------|
| NumPy | all |
| SciPy (`spatial.cKDTree`, `stats.norm`) | `coupling_gramian`, `coupling_entropy` |

---

## Summary

| Group | Files |
|-------|-------|
| Driver + coupling wrappers | 2 |
| Coupling strategies | 5 |
| Supporting files | 3 |
| **Total** | **10** |

---

*Generated 2026-06-26 from `README.md` and source files in `scripts/joint/`.*
