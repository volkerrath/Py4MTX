# data_proc.py — EDI / MT data processing utilities

`data_proc.py` provides light-weight I/O and processing helpers for
magnetotelluric (MT) transfer functions.

## Core capabilities

- **Read/write EDI files** — both Phoenix/SPECTRA and classical table formats
  (`load_edi`, `save_edi`)
- **Compute derived quantities** — apparent resistivity and phase
  (`compute_rhophas`), phase tensor (`compute_pt`), impedance invariants
  (`compute_zdet`, `compute_zssq`)
- **DataFrame conversion** — tidy `pandas.DataFrame` for plotting
  (`dataframe_from_edi`)
- **Multi-format export** — NPZ, HDF5/NetCDF, MATLAB `.mat`
  (`save_npz`, `save_hdf`, `save_ncd`, `save_mat`)
- **Experimental EMTF-XML support** — read/write simplified EMTF-XML
  (`read_emtf_xml`, `write_emtf_xml`, `edi_to_emtf`, `emtf_to_edi`)
- **Processing helpers** — error estimation, interpolation, rotation, and
  manual error setting (`estimate_errors`, `interpolate_data`, `rotate_data`,
  `set_errors`)
- **1-D forward modelling** — `mt1dfwd` and `wait1d` for layered-Earth
  impedance calculations

---

## EDI dictionary ("data_dict") layout

`load_edi()` returns a flat dict whose key set depends on what the EDI
contains.

Common keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `freq` | `(n,)` | Frequency array [Hz] |
| `Z` | `(n,2,2)` | Complex impedance tensor |
| `Z_err` | `(n,2,2)` | Error (variance or std; see `err_kind`) |
| `T` | `(n,1,2)` | Complex tipper (if present) |
| `T_err` | `(n,1,2)` | Tipper error |
| `P` | `(n,2,2)` | Phase tensor (after `compute_pt`) |
| `P_err` | `(n,2,2)` | Phase tensor error |
| `rot` | `(n,)` | Rotation angle [deg], if present |
| `err_kind` | `str` | `"var"` or `"std"` |

Metadata (if present): `station`, `lat_deg`, `lon_deg`, `elev_m`, and
convenience aliases `lat`, `lon`, `elev`.

All downstream modules treat this dict as the **site container**.

---

## Reading / writing EDI

### `load_edi(path, prefer_spectra=True, ...)`

Supports Phoenix/SPECTRA EDIs (reconstructed Z/T from `>SPECTRA` blocks) and
classical table EDIs (`>ZXXR`, `>ZXYI`, …). Frequencies are returned in
ascending order.

### `save_edi(edi, path, ...)`

Writes a classical table-style EDI from an in-memory dict.

---

## Apparent resistivity + phase

### `compute_rhophas(freq, Z, Z_err=None, *, err_kind="var", err_method=..., nsim=200, ...)`

Computes apparent resistivity (`rho_a = |Z|² / (μ₀ω)`) and phase
(`φ = angle(Z)` in degrees) for each impedance component.

Supported `err_method` values: `"none"`, `"analytic"`, `"bootstrap"`, `"both"`.

```python
rho, phi, rho_err, phi_err = data_proc.compute_rhophas(
    site["freq"], site["Z"], site.get("Z_err"),
    err_kind=site.get("err_kind", "var"),
    err_method="bootstrap", nsim=500,
)
```

---

## Phase tensor and invariants

### `compute_pt(Z, Z_err=None, *, err_kind="var", err_method=..., nsim=200, ...)`

Computes the phase tensor `P = inv(Re(Z)) @ Im(Z)` per period, with optional
error propagation (analytic delta-method, bootstrap, or both).

### `compute_zdet(Z, ...)` and `compute_zssq(Z, ...)`

Impedance determinant and sum-of-squares invariants, using the same
`err_method`/`err_kind` pattern.

---

## Converting to a DataFrame (for plotting)

### `dataframe_from_edi(edi, include_rho_phi=True, include_tipper=True, include_pt=True, ...)`

Produces a tidy DataFrame with columns compatible with `data_viz.py`:
`freq`, `period`, `rho_xx`…`rho_yy`, `phi_xx`…`phi_yy`,
`Tx_re`/`Tx_im`/`Ty_re`/`Ty_im`, `ptxx_re`…`ptyy_re`, plus `*_err` columns
when error arrays exist.

```python
import data_proc, data_viz
import matplotlib.pyplot as plt

site = data_proc.load_edi("SITE.edi")
site["P"], site["P_err"] = data_proc.compute_pt(site["Z"], site.get("Z_err"))
df = data_proc.dataframe_from_edi(site)

fig, ax = plt.subplots()
data_viz.add_rho(df, ax=ax, show_errors=True)
plt.show()
```

---

## Export formats

### Project-style NPZ

Stores the site dict under a single `data_dict` key in a compressed `.npz`.

```python
data_proc.save_npz(site, "SITE.npz")
site2 = data_proc.load_npz("SITE.npz")
```

For collections: `save_list_of_dicts_npz(records, path)` /
`load_list_of_dicts_npz(path)`.

### MATLAB `.mat`

`save_mat(data_dict, path, *, key="mt", include_raw=True, do_compression=True)`
writes a `.mat` containing a struct table, metadata, and (optionally) the raw
EDI-style arrays for direct MATLAB use.

```matlab
S = load('SITE.mat');
mt = S.mt_raw;
f  = mt.freq;
Z  = mt.Z;       % (n,2,2) complex
```

### HDF5 and NetCDF

`save_hdf(data_dict, path)` and `save_ncd(data_dict, path)` write
HDF5/NetCDF representations of the site dict.

---

## 1-D forward modelling

### `mt1dfwd(freq, sig, d, inmod="r", out="imp", magfield="b")`

Compute 1-D MT forward response for a layered Earth using recursive
impedance propagation.

### `wait1d(periods, thick, res)`

Alternative 1-D implementation (legacy interface).

---

## Notes on "bootstrap" uncertainty

Bootstrap is implemented as **parametric Monte-Carlo**: `Z` is perturbed
using the provided `Z_err`, assuming independent Gaussian perturbations of
the complex entries.

---

## References

- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. *Annals of Statistics*, 7(1), 1–26.
- Eisel, M. & Egbert, G. D. (2001). On the stability of magnetotelluric transfer function estimates. *GJI*, 144(1), 65–82.
- Neukirch, M. & García, X. (2014). Nonstationary magnetotelluric data processing. *JGR: Solid Earth*, 119(3), 1634–1654.
- Chen, J. et al. (2012). Using empirical mode decomposition to process marine MT data. *GJI*, 190(1), 293–309.
- Usui, Y. et al. (2024). New robust remote reference estimator. *GJI*, 238(2), 943–959.
- Usui, Y. et al. (2025). Fast and robust bootstrap for MT transfer functions. *GJI*, 242(1).
- Salibian-Barrera, M. et al. (2008). Fast and robust bootstrap. *Statistical Methods & Applications*, 17, 41–71.

---

Author: Volker Rath (DIAS)
