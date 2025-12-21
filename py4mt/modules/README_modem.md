# ReadMe_modem.md

This document describes the **ModEM helper module**:

- `modem_updated.py` (rename to `modem.py` in your codebase if desired)

The module provides **file I/O** and **processing utilities** commonly needed when working with
the ModEM 3-D magnetotelluric inversion package.

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

## Dependencies

Minimum:
- `numpy`

Common:
- `scipy`
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

## Conventions and pitfalls

- ModEM stores models in a **Fortran order** layout; keep an eye on `order='F'` reshaping in readers/writers.
- Be consistent about whether a model is stored in **linear resistivity** or **log-transformed** values.
  The `trans` argument in readers/writers controls this.
