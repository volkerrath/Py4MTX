# `coupling_entropy.py` — Mutual-Information (Entropy) Coupling

Self-contained module for mutual-information (MI) coupling between MT
and seismic models via Parzen-window kernel density estimation.

Inlined from `entropy/modules/cross_entropy_coupling.py`. Imports
`MultiscaleResampler`, `ModelGrid`, and `_fd_gradient_magnitude` from
`coupling_gramian` (the only cross-module dependency).

**References:**
- Haber & Oldenburg (1997). *Inverse Problems* 13, 63–77. https://doi.org/10.1088/0266-5611/13/1/006
- Moorkamp et al. (2011). *GJI* 184, 477–493. https://doi.org/10.1111/j.1365-246X.2010.04856.x
- Viola & Wells (1997). *IJCV* 24, 137–154. https://doi.org/10.1023/A:1007958904918

---

## Theory

The coupling objective is:

```
Φ_MI = −β · I(T(R_mt m_mt), T(R_seis m_seis))
```

where `I` is mutual information estimated by a Parzen-window (Gaussian
kernel) KDE on a 2-D histogram grid, `T` is an attribute transform, and
`R` denotes the `MultiscaleResampler` that maps each model to the common
grid. Minimising `Φ_MI` maximises MI, driving statistical dependence
between the two model attributes without assuming a specific petrophysical
relationship.

---

## Functions

### `mutual_information_kde(u, v, n_bins=32, hu=0.0, hv=0.0, ug=None, vg=None)`

Estimate `I(u, v)` via Parzen-window KDE on a 2-D evaluation grid.

| Parameter | Description |
|---|---|
| `u`, `v` | (M,) model attributes on the common grid |
| `n_bins` | Bins per axis for the evaluation grid |
| `hu`, `hv` | KDE bandwidths; 0 → Scott's rule |
| `ug`, `vg` | Pre-built evaluation grids (fix grids across FD steps) |

Returns `float` in nats (≥ 0).

Complexity: O(M · n_bins²) per call. Tractable for M ≲ 5000 and
n_bins ≤ 64.

---

## `MutualInformationCoupling`

```python
MutualInformationCoupling(
    mt_to_g, seis_to_g, *,
    beta=1.0, n_bins=32,
    bandwidth_u=0.0, bandwidth_v=0.0,
    mode="value", grad_mode="analytic",
    fd_eps=1e-4, common_coords=None, k_nn=6,
)
```

| Parameter | Description |
|---|---|
| `mt_to_g` | `MultiscaleResampler` — MT grid → common grid |
| `seis_to_g` | `MultiscaleResampler` — seismic grid → common grid |
| `beta` | Coupling weight β (≥ 0) |
| `n_bins` | KDE bins per axis |
| `bandwidth_u`, `bandwidth_v` | KDE bandwidths; 0 = Scott's rule |
| `mode` | Attribute transform (see below) |
| `grad_mode` | Gradient method (see below) |
| `fd_eps` | FD step size for `grad_mode="fd"` |
| `common_coords` | (M, 3) — required for `mode="gradient"` |
| `k_nn` | Neighbours for FD spatial gradient |

**`mode` options:**

| Value | Attribute T(m) |
|---|---|
| `"value"` | Raw resampled values (default) |
| `"rank"` | Normal-score transform → standard normal scores |
| `"gradient"` | FD gradient magnitude ‖∇m‖ on common grid |

`"rank"` makes bandwidth selection scale-independent and handles
heterogeneous parameter ranges (log₁₀(Ω·m) vs Vp [km/s]).

**`grad_mode` options:**

| Value | Method |
|---|---|
| `"analytic"` | Exact Parzen-window VJP — O(M · n_bins) |
| `"fd"` | Forward finite differences — slower, useful for validation |

**Methods:**

`value(m_mt, m_seis) → float` — Φ_MI (to be minimised).

`gradient(m_mt, m_seis) → (grad_mt, grad_seis)` — gradients on native
model grids via adjoint chain rule through the resampler.

`report(m_mt, m_seis) → dict` — keys: `mutual_information`, `pearson_r`,
`phi_MI`, `u_range`, `v_range`.

---

## `CombinedCoupling` (Gramian + MI)

Additive combination of `StructuralGramian` and `MutualInformationCoupling`.

```python
CombinedCoupling(gramian, mi)
```

Exposes `value`, `gradient`, and `report` (merged, no key conflicts).
`report` adds `phi_total = gramian + phi_MI`.

---

## Usage example

```python
from coupling_gramian import MultiscaleResampler, build_common_grid
from coupling_entropy import MutualInformationCoupling

common_coords = build_common_grid(mt_coords, seis_coords, dx=2000.)
mt_to_g   = MultiscaleResampler(mt_coords,   common_coords, sigma=3000.)
seis_to_g = MultiscaleResampler(seis_coords, common_coords, sigma=500.)

mi = MutualInformationCoupling(
    mt_to_g, seis_to_g,
    beta=1.0, mode="rank", n_bins=32,
)

phi          = mi.value(m_mt, m_seis)
g_mt, g_seis = mi.gradient(m_mt, m_seis)
diag         = mi.report(m_mt, m_seis)
```

---

## Dependencies

NumPy, `scipy.stats.norm` (for normal-score transform in `mode="rank"`).
`coupling_gramian` for `MultiscaleResampler` and `_fd_gradient_magnitude`.

---

## Authorship

Volker Rath (DIAS) — 2026-05-12, Claude Sonnet 4.6 (Anthropic)
