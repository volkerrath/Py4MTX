# ReadMe_dataproc.md

This package contains lightweight Python helpers for **magnetotelluric (MT) data preparation** and a few
small **forward-modelling / derived-quantity** utilities.

The main module is:

- `mtproc_updated.py` (replace/rename to `mtproc.py` in your project if desired)

## What it does

### EDI discovery
- `get_edi_list(edirname, sort=False, fullpath=True)` collects `.edi` files from a directory (ignores hidden files).

### MT data/collection wrappers (mtpy)
- `make_data(...)` reads all EDI files in a directory into an `MTData` object and can store the result in an
  `MTCollection` (HDF5).
- `make_collection(...)` creates/updates an `MTCollection` from EDI files and can convert it back to `MTData`.

> These functions require **mtpy**. If mtpy is not importable, the functions raise a `RuntimeError`
  with an installation hint.

### Derived quantities
- `calc_rhoa_phas(freq, Z)` computes apparent resistivity and phase from complex impedance.

### Quick 1-D forward checks
- `mt1dfwd(freq, sig, d, ...)` computes layered-earth 1D MT response using a recursion.
- `wait1d(periods, thick, res)` provides an alternative legacy recursion.

## Installation / dependencies

Minimum:
- `numpy`

Recommended / optional:
- `xarray` (used by some workflows)
- `pyproj` (coordinate utilities, if used elsewhere)
- `mtpy` (required for `make_data` / `make_collection`)

## Quick examples

### 1) List EDI files

```python
from mtproc_updated import get_edi_list

edis = get_edi_list("./edis/", sort=True, fullpath=True)
print(len(edis), edis[:3])
```

### 2) Build an MTData object (requires mtpy)

```python
from mtproc_updated import make_data

mtd = make_data(
    edirname="./edis/",
    collection="./My_Collection.h5",
    survey="my_survey",
    savedata=True,
    utm_epsg=32629,
)
```

### 3) Apparent resistivity + phase from impedance

```python
import numpy as np
from mtproc_updated import calc_rhoa_phas

freq = np.logspace(-3, 3, 50)
Z = (1.0 + 1.0j) * 1e-3  # example only
rhoa, phi = calc_rhoa_phas(freq=freq, Z=Z)
```

## Notes

- The module is intentionally lightweight and does **not** try to replace your EDI parsing pipeline
  (`ediproc/edidat` etc.)—it mainly provides convenience wrappers and small utilities.
