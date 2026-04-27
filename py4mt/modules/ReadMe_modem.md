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

### DCT model compression
A suite of functions for representing a ModEM resistivity model in the
Discrete Cosine Transform (DCT-II) domain, enabling low-rank compression,
spectral diagnostics, and smooth parameterisations for null-space analysis
or gradient-based optimisation.

**Radial (L2-wavenumber) truncation** — keeps the *n* lowest-wavenumber
coefficients ranked by k = √(kx² + ky² + kz²), i.e. a sphere in wavenumber
space.  Truncation can be specified by count (`n_keep`), fraction
(`frac_keep`), or cutoff wavenumber (`kmax`).

**Separable (box) truncation** — keeps the first nx_keep × ny_keep × nz_keep
coefficients as a rectangular sub-block of the coefficient array.  Useful
when different horizontal and vertical resolutions are desired.

| Function | Description |
|---|---|
| `model_to_dct(mval, ...)` | Forward 3-D DCT-II + optional radial truncation → `(coeff, shape, keep_mask)` |
| `dct_to_model(coeff, shape, keep_mask)` | Inverse DCT-II → reconstructed model array |
| `dct_compress(mval, ...)` | One-call compress + reconstruct + print stats |
| `dct_reconstruction_error(mval, mval_rec, norm)` | RMS / L-inf / relative-RMS error |
| `dct_spectrum(mval, n_bins)` | Radially averaged DCT power spectrum |
| `dct_truncation_analysis(mval, n_levels)` | Sweep over compression levels; print accuracy table |
| `model_to_dct_separable(mval, nx_keep, ny_keep, nz_keep)` | Separable (box) forward truncation |
| `dct_separable_to_model(coeff_block, shape_full)` | Inverse of separable truncation |

The DCT-II with `norm='ortho'` is used throughout (unitary, zero-flux
boundary conditions), which matches ModEM's convention of tapering models to
a background resistivity at the mesh boundary.

Additional compression methods (e.g. wavelet, SVD-based) will be added in
future releases under the same section header.

## Dependencies

Minimum:
- `numpy`
- `scipy` (including `scipy.fft` for DCT compression)
- `netCDF4`

Optional:
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

- ModEM stores models in a **Fortran order** layout; keep an eye on `order='F'` reshaping in readers/writers.
- Be consistent about whether a model is stored in **linear resistivity** or **log-transformed** values.
  The `trans` argument in readers/writers controls this.
