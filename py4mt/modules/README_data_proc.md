# README — `data_proc.py`

Utilities for **magnetotelluric (MT) transfer-function I/O and light processing**.

The module centers on an **EDI dictionary** representation (frequency/period + impedance/tipper + optional errors + derived quantities). It supports two common EDI “flavours”:

- **Phoenix / SPECTRA EDIs** (`>SPECTRA` blocks): parses spectra matrices and reconstructs **Z** (and **T** if present).
- **Classical table EDIs** (`>FREQ`, `>ZXXR`, … blocks): reads tabulated Z/T (and optional variance columns).

---

## Dependencies

Required:

- `numpy`
- `pandas`
- `scipy` (used for smoothing-spline utilities)

Optional:

- `tables` (PyTables) — needed for `save_hdf()`
- `xarray` — needed for `save_ncd()`

---

## Core data structure: the “EDI dict”

A typical dictionary returned by `load_edi()` looks like:

```python
edi = {
    "freq":    (n,),              # Hz (ascending)
    "Z":       (n,2,2) complex,   # impedance tensor
    "T":       (n,1,2) complex,   # tipper (optional)
    "Z_err":   (n,2,2) float,     # variance or std (optional)
    "T_err":   (n,1,2) float,     # variance or std (optional)
    "P":       (n,2,2) float,     # phase tensor (optional, via compute_pt)
    "P_err":   (n,2,2) float,     # optional
    "rot":     (n,) float,        # degrees (optional)
    "err_kind": "var" | "std",
    "source_kind": "spectra" | "tables",
    "station": str | None,
    "lat_deg": float | None,
    "lon_deg": float | None,
    "elev_m": float | None,
}
```

---

## Main API

- `load_edi(path, ...)` — read an EDI file (Phoenix spectra or classic tables) into an EDI dict.
- `save_edi(edi, path, ...)` — write an EDI dict back to a classical table-style EDI.
- `compute_pt(edi, ...)` — compute phase tensor (Φ) and optionally propagate impedance uncertainty.

Persistence helpers:

- `save_hdf(data_dict_or_df, path, key="mt", ...)` — write a table + metadata to HDF5 via pandas.
- `save_ncd(data_dict_or_df, path, ...)` — write a NetCDF file via xarray.
- `save_npz(obj, path, key="data_dict")` / `load_npz(path, key=...)` — store/load arbitrary picklable Python objects as a single NPZ entry.
- `save_list_of_dicts_npz(records, path, key="records")` / `load_list_of_dicts_npz(...)` — convenience wrappers for a collection like `sites = [site0, site1, ...]`.

---

## Common workflows

### 1) Read an EDI

```python
from data_proc import load_edi

edi = load_edi("SITE001.edi")
Z = edi["Z"]
freq = edi["freq"]
```

### 2) Compute phase tensor (Φ)

```python
from data_proc import compute_pt

edi = compute_pt(edi, n_mc=200)  # MC propagation optional
P = edi["P"]
```

### 3) Write a classical EDI

```python
from data_proc import save_edi

save_edi(edi, "SITE001_out.edi")
```

---

## Persisting results

### A) NPZ: save/load arbitrary Python objects (pickle-based)

```python
from data_proc import save_npz, load_npz

save_npz(edi, "site001.npz", key="edi")
edi2 = load_npz("site001.npz", key="edi")
```

### B) NPZ: save/load a list of dicts (Option A)

```python
from data_proc import save_list_of_dicts_npz, load_list_of_dicts_npz

sites = [
    {"station": "A001", "freq": [1.0, 0.1], "Z": Z0},
    {"station": "A002", "freq": [1.0, 0.1], "Z": Z1},
]

save_list_of_dicts_npz(sites, "sites.npz", key="records")
sites2 = load_list_of_dicts_npz("sites.npz", key="records")
```

### C) HDF5: save a table plus metadata (`save_hdf`)

`save_hdf()` writes two datasets:

- `f"{key}/table"` — a DataFrame with tabular variables (freq/period + expanded columns)
- `f"{key}/meta"` — a single-row DataFrame with metadata (`df.attrs` / non-tabular entries)

```python
from data_proc import save_hdf

save_hdf(edi, "site001.h5", key="mt")
```

#### Avoiding the PyTables “pickling object types” warning

Pandas/PyTables warns when writing columns of dtype `object` that contain **mixed Python objects** (for example `header_raw` as a list). In that case it falls back to **pickling**, which is slower and less portable.

To avoid this, `save_hdf()` sanitizes metadata using:

- `_sanitize_meta_for_hdf(meta)` — converts non-scalar metadata to JSON strings (fallback: `repr(...)`).

### D) NetCDF: save via xarray (`save_ncd`)

```python
from data_proc import save_ncd

save_ncd(edi, "site001.nc", dim="freq", dataset_name="mt")
```

Notes:

- Complex arrays are stored as `<name>_re` and `<name>_im`.
- Metadata is flattened into NetCDF attributes (`ds.attrs`).

---

## Other utilities

- Smoothing / uncertainty helpers (smoothing splines, GCV-like lambda choice, bootstrap CI utilities)
- Experimental EMTF-XML reader: `read_emtf_xml(...)`

---

## Notes and gotchas

- Frequencies are returned in **ascending** order.
- Table-style EDI variance blocks are interpreted as variances by default (`err_kind="var"`).
- The NPZ helpers store Python objects (NumPy object arrays). Loading uses `allow_pickle=True` internally.

### Security note (NPZ pickling)

Pickle can execute arbitrary code when loading malicious files. Only load `.npz` files created by you or from sources you trust.
