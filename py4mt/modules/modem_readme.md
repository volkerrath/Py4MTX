# ReadMe_modem.md

This document describes the **ModEM helper module**:

- `modem_updated.py` (rename to `modem.py` in your codebase if desired)

The module provides **file I/O**, **processing utilities**, and **model compression** commonly
needed when working with the ModEM 3-D magnetotelluric inversion package.

## Key features

### Jacobian I/O
- Read ModEM Jacobians stored as Fortran unformatted files:
  - `read_jac(Jacfile, out=False)`
  - `read_data_jac(Datfile, out=True)`
- Export Jacobian + metadata to NetCDF:
  - `write_jac_ncd(NCfile, Jac, Dat, Site, Comp, ...)`

### Data file helpers
- Read/Write ModEM `.dat`-style data files:
  - `read_data(Datfile, modext=".dat", out=True)`
  - `write_data(Datfile, Dat, Site, Comp, Head, out=True)`
- NetCDF export:
  - `write_data_ncd(NCfile, Dat, Site, Comp, ...)`

### Model file helpers
- Read/Write ModEM `.rho` models:
  - `read_mod(file, modext=".rho", trans="LINEAR", ...)`
  - `write_mod(file, modext=".rho", dx, dy, dz, mval, ...)`
  - `write_mod_npz(...)` (NPZ convenience export)
  - `write_mod_ncd(...)` (NetCDF export)
- UBC and RLM conversion utilities:
  - `write_ubc(...)`, `read_ubc(...)`
  - `write_rlm(...)`

### Numerical utilities
The module also contains a set of filters / smoothing operators and model-preparation helpers
used in earlier project scripts (e.g., diffusion / median filters, anisotropic diffusion, etc.).

### Model compression and alternative parameterisations

Four compression / basis-expansion methods are implemented, all sharing the
same `out=True` verbosity convention and a common error metric
(`dct_reconstruction_error`).  Each provides a forward transform, an inverse
transform, a one-call compress-and-reconstruct wrapper, and a truncation
sweep for choosing compression parameters.

#### DCT-II (Discrete Cosine Transform)

Represents the model as a superposition of cosine modes.  Zero-flux
(Neumann) boundary conditions match ModEM's background-tapering convention.
Best for smooth, spatially stationary models.

**Radial truncation** — keep the *n* lowest-wavenumber coefficients
(sphere in wavenumber space); truncation by `n_keep`, `frac_keep`, or
cutoff `kmax`.  **Separable (box) truncation** — keep the first
nx_keep × ny_keep × nz_keep coefficients; allows different horizontal and
vertical resolution.

| Function | Description |
|---|---|
| `model_to_dct(mval, ...)` | Forward 3-D DCT-II + radial truncation |
| `dct_to_model(coeff, shape, keep_mask)` | Inverse DCT-II |
| `dct_compress(mval, ...)` | One-call wrapper + stats |
| `dct_reconstruction_error(mval, mval_rec, norm)` | RMS / L-inf / rel-RMS error |
| `dct_spectrum(mval, n_bins)` | Radially averaged power spectrum |
| `dct_truncation_analysis(mval, n_levels)` | Accuracy vs compression sweep |
| `model_to_dct_separable(mval, nx_keep, ny_keep, nz_keep)` | Separable (box) forward truncation |
| `dct_separable_to_model(coeff_block, shape_full)` | Inverse of separable truncation |

#### Wavelet (3-D DWT)

Spatially *localised* frequency decomposition via `pywt.dwtn`.  The key
advantage over DCT: a wavelet coefficient encodes a frequency at a specific
location, so a conductive anomaly in one corner of the model is represented
compactly regardless of the background.  Thresholding by count (`n_keep`),
fraction (`frac_keep`), or amplitude (`thresh`).  Requires
`PyWavelets` (`pip install PyWavelets`).  Recommended wavelets for smooth
geophysical models: `'db4'`, `'sym4'`, `'coif2'`.

| Function | Description |
|---|---|
| `model_to_wavelet(mval, wavelet, ...)` | Forward DWT + hard thresholding |
| `wavelet_to_model(coeffs, wavelet, shape)` | Inverse DWT |
| `wavelet_compress(mval, wavelet, ...)` | One-call wrapper + stats |
| `wavelet_truncation_analysis(mval, wavelet, n_levels)` | Accuracy vs compression sweep |

#### Legendre-z × DCT-xy (separable mixed basis)

Respects the physical anisotropy of MT models: the depth axis has strong
resistivity gradients that Legendre polynomials represent compactly, while
horizontal structure is quasi-periodic and well-suited to the DCT.  The
transform applies a 1-D Legendre projection along z and a 2-D DCT-II along
x and y.  Independent compression ratios can be set for each axis.

| Function | Description |
|---|---|
| `model_to_legdct(mval, n_leg, nx_dct, ny_dct, ...)` | Forward Legendre-z × DCT-xy |
| `legdct_to_model(C, shape_full, params)` | Inverse transform |
| `legdct_compress(mval, ...)` | One-call wrapper + stats |
| `legdct_truncation_analysis(mval, n_levels)` | Accuracy vs compression sweep |

#### B-spline-z × DCT-xy (separable mixed basis)

Structurally identical to Legendre-z × DCT-xy but uses locally supported
cubic B-splines along the depth axis instead of global Legendre polynomials.

Key advantages over the Legendre variant:

- **Local support** — a B-spline control point at 10 km depth has zero
  influence below ~30 km (depending on knot spacing).  Changes in the deep
  model do not disturb the near-surface fit, and vice versa.
- **Adaptive knot placement** — the knot vector follows the actual cell-size
  distribution via `knot_style='quantile'`, concentrating basis functions
  near the surface where ModEM cells are fine and the MT data are most
  sensitive, and spacing them coarsely at depth where cells are large.
- **Clamped boundary conditions** — coincident end knots enforce zero slope
  at the top and bottom of the model, which is physically reasonable.

Three knot placement strategies are available:

| `knot_style` | Knot positions | When to use |
|---|---|---|
| `'quantile'` | Depth quantiles of cell centres (default) | Real models with expanding cell size |
| `'uniform'` | Equally spaced in normalised depth | Models with uniform cell sizes |
| `'log'` | Logarithmically spaced | When depth spans many decades |

| Function | Description |
|---|---|
| `model_to_bspdct(mval, dz, n_basis, k, knot_style, ...)` | Forward B-spline-z × DCT-xy |
| `bspdct_to_model(C, shape_full, params)` | Inverse transform |
| `bspdct_compress(mval, dz, ...)` | One-call wrapper + stats |
| `bspdct_truncation_analysis(mval, dz, ...)` | Accuracy vs compression sweep |

#### KL / PCA (ensemble-based Karhunen–Loève)

Theoretically optimal linear basis for a given model ensemble.  Given a
collection of prior or posterior models the empirical covariance is
diagonalised by SVD; the resulting eigenmodes are sorted by descending
explained variance.  Any new model is expressed as a score vector of length
`n_modes`.  Most powerful for uncertainty quantification and for
parameterising gradient-based inversion in a reduced subspace.

Three SVD backends are supported via the `svd_method` argument of
`ensemble_to_kl`:

| `svd_method` | Algorithm | Cost | When to use |
|---|---|---|---|
| `'auto'` | Selects below automatically | — | Default; always a safe choice |
| `'exact'` | `numpy.linalg.svd` (economy) | O(n_m² × n_c) | All modes needed, or n_modes ≈ n_models |
| `'randomized'` | Halko et al. (2011) via sklearn | O(n_m × n_c × n_modes) | n_modes ≪ n_models — preferred for large ensembles |
| `'truncated'` | ARPACK via `scipy.sparse.linalg.svds` | O(n_m × n_c × n_modes) | No sklearn; sparse matrices |

`'auto'` selects `'randomized'` when `n_modes < 0.5 * min(n_models, n_cells)`,
otherwise `'exact'`.  If sklearn is absent the randomized path falls back
automatically to `'truncated'` with a printed warning.  Accuracy of the
randomized path is controlled by `n_oversamples` (default 10) and
`n_power_iter` (default 4).

| Function | Description |
|---|---|
| `ensemble_to_kl(ensemble, n_modes, svd_method, ...)` | SVD of centred ensemble → modes, singular values, mean |
| `model_to_kl(mval, modes, mean_model)` | Project model onto KL basis → coefficients α |
| `kl_to_model(alpha, modes, mean_model, shape)` | Reconstruct model from α |
| `kl_variance_spectrum(singular_values)` | Per-mode variance fraction table |
| `kl_truncation_analysis(mval, modes, ...)` | Accuracy vs number of modes sweep |

## Dependencies

Minimum:
- `numpy`
- `scipy` (`scipy.fft` for DCT/Legendre compression; `scipy.special` for Legendre basis)
- `netCDF4`

Optional:
- `PyWavelets` / `pywt` — wavelet compression (`pip install PyWavelets`); all wavelet functions raise `ImportError` with an install hint if absent
- `numba` (some functions import/use it)
- `pyproj` is only required if the project-local `util.py` is not available and you call UTM helpers.

### About `util.py`
The original module expects `import util as utl`. In `modem_updated.py` this import is wrapped in a
`try/except` and a **minimal fallback** is provided (UTM zone and WGS84↔UTM projection via `pyproj`).

If you work in the southern hemisphere, set:

```bash
export MODEM_UTM_HEMISPHERE=S
```

before using `proj_utm_to_latlon` via the fallback.

## Quick examples

### 1) Read a ModEM model

```python
from modem_updated import read_mod

dx, dy, dz, rho, ref, trans = read_mod("resistivity_block_iter0", modext=".dat", trans="LINEAR")
```

### 2) Export a Jacobian to NetCDF

```python
from modem_updated import read_jac, read_data_jac, write_jac_ncd

Jac, inf = read_jac("Jacobian.bin", out=True)
Dat, Site, Freq, Comp, Dtyp, Head = read_data_jac("data.dat", out=True)

write_jac_ncd("jacobian.nc", Jac=Jac, Dat=Dat, Site=Site, Comp=Comp, out=True)
```

### 3) Compress a model with DCT and check accuracy

```python
from modem import read_mod, model_to_dct, dct_to_model, dct_truncation_analysis

dx, dy, dz, rho, ref, _ = read_mod("TAC_100", modext=".rho", trans="LOGE")

# Sweep compression levels to find an acceptable ratio
dct_truncation_analysis(rho, n_levels=15)

# Keep the 5 % lowest-wavenumber coefficients
coeff, shape, mask = model_to_dct(rho, frac_keep=0.05)
rho_rec = dct_to_model(coeff, shape, mask)
```

### 4) Anisotropic (separable) DCT compression

```python
from modem import model_to_dct_separable, dct_separable_to_model

# Retain full horizontal resolution but compress vertically 4×
coeff_block, shape_full, shape_keep = model_to_dct_separable(
    rho, nx_keep=None, ny_keep=None, nz_keep=rho.shape[2] // 4
)
rho_rec = dct_separable_to_model(coeff_block, shape_full)
```

### 5) Wavelet compression

```python
from modem import wavelet_compress, wavelet_truncation_analysis

# Diagnose accuracy vs compression for db4 wavelet
wavelet_truncation_analysis(rho, wavelet='db4', n_levels=15)

# Keep the 5 % largest-magnitude wavelet coefficients
rho_rec, coeffs, n_kept = wavelet_compress(rho, wavelet='db4', frac_keep=0.05)
```

### 6) Legendre-z × DCT-xy mixed-basis compression

```python
from modem import legdct_compress, legdct_truncation_analysis

# Sweep combined compression levels
legdct_truncation_analysis(rho, n_levels=15)

# 30 % Legendre along z, 40 % DCT along x and y
rho_rec, C, params = legdct_compress(rho, frac_leg=0.3, frac_dct=0.4)
```

### 7) B-spline-z × DCT-xy mixed-basis compression

```python
from modem import read_mod, bspdct_compress, bspdct_truncation_analysis

dx, dy, dz, rho, ref, _ = read_mod("TAC_100", modext=".rho", trans="LOGE")

# Sweep compression levels — knots follow the cell-size distribution
bspdct_truncation_analysis(rho, dz=dz, knot_style='quantile', n_levels=15)

# 40 % B-spline basis along z, 40 % DCT along x and y, cubic splines
rho_rec, C, params = bspdct_compress(
    rho, dz=dz,
    frac_basis=0.4,          # fraction of nz cells kept as spline basis functions
    k=3,                     # cubic splines
    knot_style='quantile',   # knots at depth quantiles of cell centres
    frac_dct=0.4,
)

# Explicit knot count and explicit DCT size
rho_rec, C, params = bspdct_compress(
    rho, dz=dz, n_basis=12, k=3, knot_style='quantile',
    nx_dct=20, ny_dct=20,
)
```

### 8) KL / PCA ensemble-based compression

```python
import numpy as np
from modem import (read_mod, ensemble_to_kl, model_to_kl,
                   kl_to_model, kl_variance_spectrum, kl_truncation_analysis)

# Build ensemble from multiple inversion results
files = ["iter10_run1", "iter10_run2", "iter10_run3"]
models = []
for f in files:
    _, _, _, m, _, _ = read_mod(f, modext=".rho", trans="LOGE")
    models.append(m)
ensemble = np.stack(models)                 # (n_models, nx, ny, nz)

# Exact SVD — use when all modes are needed or n_modes ≈ n_models
modes, sv, mean_m, shape = ensemble_to_kl(ensemble, svd_method='exact')

# Randomized SVD — strongly preferred when keeping only a few modes
# (requires sklearn; falls back to truncated ARPACK automatically if absent)
modes, sv, mean_m, shape = ensemble_to_kl(
    ensemble, n_modes=20,
    svd_method='randomized',   # or 'auto' to select automatically
    n_oversamples=10,          # extra projections: more → more accurate
    n_power_iter=4,            # power iterations: more → better on flat spectra
    random_state=42,
)

# Inspect variance spectrum to choose n_modes
kl_variance_spectrum(sv)

# Project a target model and reconstruct with 10 modes
dx, dy, dz, rho_target, ref, _ = read_mod("TAC_100", modext=".rho", trans="LOGE")
alpha = model_to_kl(rho_target, modes, mean_m)
rho_rec = kl_to_model(alpha[:10], modes, mean_m, shape=shape)

# Reconstruction accuracy vs number of modes
kl_truncation_analysis(rho_target, modes, mean_m, shape=shape, singular_values=sv)
```

- ModEM stores models in a **Fortran order** layout; keep an eye on `order='F'` reshaping in readers/writers.
- Be consistent about whether a model is stored in **linear resistivity** or **log-transformed** values.
  The `trans` argument in readers/writers controls this.
