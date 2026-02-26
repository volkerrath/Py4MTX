# data_proc.py — EDI / MT data processing utilities

`data_proc.py` provides light-weight I/O + processing helpers for MT transfer
functions:

- read/write EDI (`load_edi`, `save_edi`)
- compute derived quantities (`compute_rhophas`, `compute_pt`, `compute_zdet`, `compute_zssq`)
- convert an EDI dict to a tidy `pandas.DataFrame` (`dataframe_from_edi`)
- persist data as **project-style NPZ** with a single `data_dict` key (`save_npz`, `load_npz`)

This README describes the **dict conventions actually used by tonight’s modules**
(in particular `mcmc.py` / `inv1d.py` / `mt_aniso1d_sampler.py`).

---

## EDI dictionary (“data_dict”) layout

`load_edi()` returns a *flat* dict. Key set depends on what the EDI contains.

Common keys:

- `freq` : `(n,)` frequency array [Hz]
- `Z` : `(n,2,2)` complex impedance tensor
- `Z_err` : `(n,2,2)` float (variance or std; see `err_kind`)
- `T` : `(n,1,2)` complex tipper (if present)
- `T_err` : `(n,1,2)` float
- `P` : `(n,2,2)` float phase tensor (only after calling `compute_pt`)
- `P_err` : `(n,2,2)` float
- `rot` : `(n,)` float rotation angle (deg), if present in the EDI
- `err_kind` : `"var"` or `"std"` (how `*_err` arrays are interpreted)
- metadata (if present): `station`, `lat_deg`, `lon_deg`, `elev_m`, …

All downstream modules treat this dict as the **site container**.

---

## Reading / writing EDI

### `load_edi(path, prefer_spectra=True, ...)`

Supports:

- Phoenix / SPECTRA EDIs (`>SPECTRA` blocks; reconstructed Z/T)
- classical table EDIs (`>ZXXR`, `>ZXYI`, …)

### `save_edi(edi, path, ...)`

Writes a **classical table-style** EDI from an in-memory dict.

---


---

## MATLAB .mat export

### `save_mat(data_dict, path, *, key="mt", include_raw=True, do_compression=True)`

Writes a MATLAB ``.mat`` file containing:

- ``<key>_table``: struct with one field per DataFrame column (e.g. ``freq``, ``Z_re_0``, ...)
- ``<key>_table_cols``: ordered column-name list
- ``<key>_table_data``: numeric matrix ``(n, ncol)`` (same order as ``*_table_cols``)
- ``<key>_meta``: metadata (sanitized)
- ``<key>_raw``: **raw EDI-style dict** (arrays/scalars/strings), for MATLAB users who prefer
  to work with complex tensors like ``Z`` directly. Controlled by ``include_raw``.

Example:

```python
import data_proc

site = data_proc.load_edi("SITE.edi")
data_proc.save_mat(site, "SITE.mat", include_raw=True)
```

MATLAB usage:

```matlab
S = load('SITE.mat');
mt = S.mt_raw;        % raw dict-like struct
f  = mt.freq;         % (n,1) or (1,n)
Z  = mt.Z;            % (n,2,2) complex
meta = S.mt_meta;     % metadata struct
```

Notes:

- This uses ``scipy.io.savemat`` (MATLAB v7.3 / HDF5 MAT files are **not** written by SciPy).
- For very large collections consider HDF5 (``save_hdf``) or NetCDF (``save_ncd``).


## Project-style NPZ (flat, single key)

In this project, an “NPZ site file” is a compressed `*.npz` that contains
**one key only**:

- `data_dict` → the Python dict described above (stored as a NumPy object array)

Use:

```python
import data_proc

site = data_proc.load_edi("SITE.edi")
data_proc.save_npz(site, "SITE.npz")     # writes key="data_dict"

site2 = data_proc.load_npz("SITE.npz")   # returns the dict
```

For collections (e.g., lists of sites or records):

- `save_list_of_dicts_npz(records, path, key="records")`
- `load_list_of_dicts_npz(path, key="records")`

Notes:

- NPZ loading requires `allow_pickle=True` internally (handled in `load_npz`).

---

## Apparent resistivity + phase

### `compute_rhophas(freq, Z, Z_err=None, *, err_kind="var", err_method="analytic"|"bootstrap"|..., nsim=200, ...)`

Computes apparent resistivity and phase for each impedance component:

- `rho_a = |Z|^2 / (mu0 * omega)` with `omega = 2π f`
- `phi = angle(Z)` (degrees)

Optional uncertainty propagation:

- `err_method="none"`: no errors returned
- `err_method="analytic"`: fast first-order propagation
- `err_method="bootstrap"`: parametric Monte‑Carlo perturbation of `Z`
- `err_method="both"`: returns both analytic and bootstrap error estimates

Example:

```python
rho, phi, rho_err, phi_err = data_proc.compute_rhophas(
    site["freq"], site["Z"], site.get("Z_err"),
    err_kind=site.get("err_kind", "var"),
    err_method="bootstrap",
    nsim=500,
)
```

---

## Phase tensor and invariants

### `compute_pt(Z, Z_err=None, *, err_kind="var", err_method="bootstrap"|..., nsim=200, ...)`

Computes the phase tensor as:

- `P = inv(Re(Z)) @ Im(Z)`   (per period)

If `Z_err` is provided and `err_method != "none"`, `P_err` is estimated either
analytically (finite-difference delta method), via bootstrap, or both.

### `compute_zdet(Z, ...)` and `compute_zssq(Z, ...)`

Convenience helpers to compute two common impedance invariants, with the same
`err_method` / `err_kind` pattern as `compute_pt`.

---

## Converting to a DataFrame (for plotting)

### `dataframe_from_edi(edi, include_rho_phi=True, include_tipper=True, include_pt=True, ...)`

Produces a tidy DataFrame with columns compatible with `data_viz.py`:

- frequency/period: `freq`, `period`
- resistivity: `rho_xx`, `rho_xy`, `rho_yx`, `rho_yy`
- phase: `phi_xx`, `phi_xy`, `phi_yx`, `phi_yy`
- tipper: `Tx_re`, `Tx_im`, `Ty_re`, `Ty_im`
- phase tensor: `ptxx_re`, `ptxy_re`, `ptyx_re`, `ptyy_re`
- plus optional `*_err` columns when error arrays exist in the dict

Example:

```python
import data_proc
import data_viz
import matplotlib.pyplot as plt

site = data_proc.load_edi("SITE.edi")
site["P"], site["P_err"] = data_proc.compute_pt(site["Z"], site.get("Z_err"))

df = data_proc.dataframe_from_edi(site)

fig, ax = plt.subplots()
data_viz.add_rho(df, ax=ax, show_errors=True)
plt.show()
```

---

## Notes on “bootstrap” here

Bootstrap is implemented as **parametric Monte‑Carlo**: `Z` is perturbed using
the provided `Z_err`, assuming independent Gaussian perturbations of the complex
entries.

---

## Bootstrap / Monte‑Carlo uncertainty references (BibTeX)

```bibtex
@article{Efron1979Bootstrap,
  title   = {Bootstrap Methods: Another Look at the Jackknife},
  author  = {Efron, Bradley},
  journal = {The Annals of Statistics},
  year    = {1979},
  volume  = {7},
  number  = {1},
  pages   = {1--26},
  doi     = {10.1214/aos/1176344552}
}

@article{EiselEgbert2001Stability,
  title   = {On the stability of magnetotelluric transfer function estimates and the reliability of their variances},
  author  = {Eisel, Markus and Egbert, Gary D.},
  journal = {Geophysical Journal International},
  year    = {2001},
  volume  = {144},
  number  = {1},
  pages   = {65--82},
  doi     = {10.1046/j.1365-246x.2001.00292.x}
}

@article{NeukirchGarcia2014Nonstationary,
  title   = {Nonstationary magnetotelluric data processing with instantaneous parameter},
  author  = {Neukirch, M. and Garc{\'i}a, X.},
  journal = {Journal of Geophysical Research: Solid Earth},
  year    = {2014},
  volume  = {119},
  number  = {3},
  pages   = {1634--1654},
  doi     = {10.1002/2013JB010494}
}

@article{Chen2012EMDMarineMT,
  title   = {Using empirical mode decomposition to process marine magnetotelluric data},
  author  = {Chen, J. and others},
  journal = {Geophysical Journal International},
  year    = {2012},
  volume  = {190},
  number  = {1},
  pages   = {293--309},
  doi     = {10.1111/j.1365-246X.2012.05536.x}
}

@article{UsuiEtAl2024RRMS,
  title   = {New robust remote reference estimator using robust multivariate linear regression},
  author  = {Usui, Yoshiya and Uyeshima, Makoto and Sakanaka, Shin'ya and Hashimoto, Tasuku and Ichiki, Masahiro and Kaida, Toshiki and Yamaya, Yusuke and Ogawa, Yasuo and Masuda, Masataka and Akiyama, Takahiro},
  journal = {Geophysical Journal International},
  year    = {2024},
  volume  = {238},
  number  = {2},
  pages   = {943--959},
  doi     = {10.1093/gji/ggae199}
}

@article{UsuiEtAl2025FRB_MT,
  title   = {Application of the fast and robust bootstrap method to the uncertainty analysis of the magnetotelluric transfer function},
  author  = {Usui, Yoshiya and Uyeshima, Makoto and Sakanaka, Shin'ya and Hashimoto, Tasuku and Ichiki, Masahiro and Kaida, Toshiki and Yamaya, Yusuke and Ogawa, Yasuo and Masuda, Masataka and Akiyama, Takahiro},
  journal = {Geophysical Journal International},
  year    = {2025},
  volume  = {242},
  number  = {1},
  doi     = {10.1093/gji/ggaf162}
}

@article{SalibianBarrera2008FRB,
  title   = {Fast and robust bootstrap},
  author  = {Salibi{\'a}n-Barrera, Mat{\'i}as and Van Aelst, Stefan and Willems, Gert},
  journal = {Statistical Methods \& Applications},
  year    = {2008},
  volume  = {17},
  pages   = {41--71},
  doi     = {10.1007/s10260-007-0048-6}
}
```

---

Author: Volker Rath (DIAS)  
Updated with the help of ChatGPT (GPT-5.2 Thinking) on 2026-02-13 (UTC)
