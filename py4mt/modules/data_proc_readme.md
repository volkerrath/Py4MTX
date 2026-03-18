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
| `Z` | `(n,2,2)` | Complex impedance tensor [mV km⁻¹ nT⁻¹] |
| `Z_err` | `(n,2,2)` | Error (variance or std; see `err_kind`) |
| `Z_orig` | `(n,2,2)` | Unperturbed Z (only present after `set_errors` with `add_noise=True`) |
| `T` | `(n,1,2)` | Complex tipper (if present) |
| `T_err` | `(n,1,2)` | Tipper error |
| `T_orig` | `(n,1,2)` | Unperturbed T (only present after `set_errors` with `add_noise=True`) |
| `P` | `(n,2,2)` | Phase tensor (after `compute_pt`) |
| `P_err` | `(n,2,2)` | Phase tensor error |
| `P_orig` | `(n,2,2)` | Unperturbed P (only present after `set_errors` with `add_noise=True`) |
| `rot` | `(n,)` | Rotation angle [deg], if present |
| `err_kind` | `str` | `"var"` or `"std"` |
| `freq_order` | `str` | `"inc"`, `"dec"`, or `"keep"` — order used when loading |
| `Z_units` | `str` | Always `"mV/km/nT"` (MT field units); `ρ_a = \|Z\|²×10⁶/(μ₀ω)` |

Metadata (if present): `station`, `lat_deg`, `lon_deg`, `elev_m`, and
convenience aliases `lat`, `lon`, `elev`.

All downstream modules treat this dict as the **site container**.

---

## Reading / writing EDI

### `load_edi(path, prefer_spectra=True, freq_order='inc', ...)`

Supports Phoenix/SPECTRA EDIs (reconstructed Z/T from `>SPECTRA` blocks) and
classical table EDIs (`>ZXXR`, `>ZXYI`, …).

The `freq_order` parameter controls the order of frequencies in the returned
dictionary:

| Value | Effect |
|-------|--------|
| `'inc'` | Ascending frequency — *default*, equivalent to the previous behaviour |
| `'dec'` | Descending frequency (i.e. ascending period) |
| `'keep'` | Preserve the order as found in the EDI file |

```python
# Default — ascending frequency
site = data_proc.load_edi("SITE.edi")

# Descending frequency (ascending period)
site = data_proc.load_edi("SITE.edi", freq_order="dec")

# Preserve EDI file order without any sorting
site = data_proc.load_edi("SITE.edi", freq_order="keep")
```

### `save_edi(edi, path, ...)`

Writes a classical table-style EDI from an in-memory dict.  The `>FREQ` line
includes `NFREQ=`, `ORDER=`, and a count comment matching the format of the
input EDI, e.g.:

```
>FREQ NFREQ=23 ORDER=DEC //23
```

The `ORDER` tag is `freq_order` uppercased directly (`"inc"` → `INC`,
`"dec"` → `DEC`, `"keep"` → `KEEP`).  When `freq_order`
is absent, the order is inferred automatically from the frequency array.

**Phase tensor blocks** — written with the `PT` prefix:
`>PTXX`, `>PTXY`, `>PTYX`, `>PTYY` (and `>PTXX.VAR` etc. when errors are present).

**Apparent resistivity and phase blocks** — always written, derived from
`edi["rho"]` / `edi["phi"]` when present, otherwise computed from `edi["Z"]`:
`>RHOXX`, `>RHOXY`, `>RHOYX`, `>RHOYY` and `>PHASEXX`, `>PHASEXY`,
`>PHASEYX`, `>PHASEYY` (plus `>.VAR` variants when `rho_err` / `phi_err` are present).

## Apparent resistivity + phase

### `compute_rhophas(freq, Z, Z_err=None, *, err_kind="var", err_method=..., nsim=200, ...)`

Computes apparent resistivity (`rho_a`) and phase (`φ = angle(Z)` in degrees)
for each impedance component.

**Unit convention:** Z is in MT field units (mV/km per nT), the py4mt
convention.  The correct formula is:

    rho_a = |Z|² × 10⁶ / (μ₀ω)

which is equivalent to converting Z to SI Ohm first (Z_SI = Z × μ₀ × 10³)
and applying `|Z_SI|² / (μ₀ω)`.  The factor 10⁶ is defined as the
module-level constant `_Z_MT_TO_SI_SQ`.

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

All save functions accept the site dictionary in two ways:

```python
# Positional dict:
save_xxx(data_dict, path="out.xxx")

# Splatted dict (preferred in scripts):
save_xxx(**data_dict, path="out.xxx")
```

In both cases, `path` and all format-specific options are **keyword-only**.
When using the `**data_dict` form, the function reconstructs the dictionary
internally from the keyword arguments.  Functions that also accept a
`pandas.DataFrame` (save_hdf, save_ncd, save_mat) must receive it as the
positional first argument.

### Project-style NPZ

Each dict entry is stored as a separate array so that individual fields can
be accessed directly:

```python
data_proc.save_npz(site, path="SITE.npz")
# or equivalently:
data_proc.save_npz(**site, path="SITE.npz")

# Direct access to individual arrays:
data = np.load("SITE.npz", allow_pickle=True)
Z    = data["Z"]          # (n, 2, 2) complex
freq = data["freq"]       # (n,) float
station = data["station"].item()  # str

# Or reload as a dict:
site2 = data_proc.load_npz("SITE.npz")
```

Non-array values (strings, `None`, scalars) are stored as 0-d object arrays
and require `allow_pickle=True` on load; use `.item()` to unwrap them.
`load_npz` does the unwrapping automatically.

For collections: `save_list_of_dicts_npz(records, path)` /
`load_list_of_dicts_npz(path)`.

### MATLAB `.mat`

`save_mat(data_dict, *, path, do_compression=True)`
stores each dict entry as a MATLAB variable (complex arrays supported
natively).

```matlab
S = load('SITE.mat');
Z    = S.Z;        % (n,2,2) complex
freq = S.freq;     % (n,1) double
station = S.station;
```

### HDF5

`save_hdf(data_dict, *, path)` stores each dict entry as an HDF5 dataset
(arrays) or attribute (scalars/strings).  Complex arrays are supported
natively by HDF5.  Requires `h5py`.

```python
import h5py
with h5py.File("SITE.hdf") as f:
    Z    = f["mt"]["Z"][:]        # (n, 2, 2) complex
    freq = f["mt"]["freq"][:]     # (n,) float
    station = f["mt"].attrs["station"]  # str
```

### NetCDF

`save_ncd(data_dict, *, path)` stores arrays as NetCDF variables.  Complex
arrays are split into `<key>_re` and `<key>_im` (NetCDF limitation).
Scalars and strings go to global attributes.  Requires `xarray`.

```python
import xarray as xr
ds = xr.open_dataset("SITE.ncd")
Z_re = ds["Z_re"].values    # (n, 2, 2)
Z_im = ds["Z_im"].values
freq = ds["freq"].values     # (n,)
```

---

## Setting / flooring errors

### `set_errors(edi_dict, *, mode, Z_rel, Z_rel_mode, T_abs, PT_abs, err_kind, add_noise, random_state)`

Sets or floors error arrays for Z, tipper, and phase tensor.
All parameters except `edi_dict` and `mode` are keyword-only and optional.

| `mode` | Behaviour |
|--------|-----------|
| `"fix"` | Replace existing errors unconditionally |
| `"floor"` | Only raise existing errors; values already larger are kept |

**Z errors — relative to the data** (`Z_rel`: length-4 `[xx, xy, yx, yy]`)

| `Z_rel_mode` | Formula |
|-------------|---------|
| `"ij"` | σ_ij(ω) = Z_rel_ij × \|Z_ij(ω)\| |
| `"ij*ii"` | σ_ij(ω) = Z_rel_ij × √(\|Z_ii(ω)\| × \|Z_ij(ω)\|) off-diagonal; diagonal as `"ij"` |

**Tipper / PT errors — absolute constant** (`T_abs`: `[Tx, Ty]`; `PT_abs`: `[xx, xy, yx, yy]`)

**Noise perturbation** (`add_noise`: bool, only valid with `mode="fix"`)

When `add_noise=True`, the data arrays are perturbed using the fixed σ values
computed for this call, and both the perturbed and original arrays are stored
in the returned dict:

| Array | Perturbed key | Original preserved as | Perturbation |
|-------|---------------|-----------------------|-------------|
| `Z` | `Z` | `Z_orig` | Real and imaginary parts each drawn from N(0, σ/√2); total complex noise has std = σ |
| `T` | `T` | `T_orig` | Same complex splitting as Z |
| `P` | `P` | `P_orig` | Real-valued; single draw from N(0, σ) per element |

`*_orig` keys are only added for components that were actually perturbed (i.e.
only when the corresponding `Z_rel` / `T_abs` / `PT_abs` argument was supplied
and the matching data array was present in the dict).

Pass `random_state` (a `numpy.random.Generator`) for reproducible draws.
Raises `ValueError` if `add_noise=True` and `mode != "fix"`.

```python
# In script — passed as **err_pars
err_pars = {
    "Z_rel":      [0.05, 0.05, 0.05, 0.05],
    "Z_rel_mode": "ij",
    "T_abs":      [0.03, 0.03],
    "PT_abs":     [0.05, 0.05, 0.05, 0.05],
    "add_noise":  True,
    "random_state": np.random.default_rng(42),  # omit for non-reproducible
}
data_dict = data_proc.set_errors(data_dict, mode="fix", **err_pars)
data_dict = data_proc.set_errors(data_dict, mode="floor", **err_pars)
```

---

## Interpolation

### `interpolate_data(edi_dict, *, newfreqs, freq_per_dec, interp_method)`

Resamples all arrays (Z, T, P, errors, rot) onto a new frequency grid using
spline interpolation in log-frequency space.

| Parameter | Description |
|-----------|-------------|
| `newfreqs` | Explicit target frequency array [Hz]; takes priority |
| `freq_per_dec` | Frequencies per decade for a log-spaced grid over the data range |
| `interp_method` | `"gcvspline"` or `"linear"` (default); passed to `make_spline` |

```python
# In script — passed as **interp_pars
interp_pars = {
    "freq_per_dec":  6,
    "interp_method": "gcvspline",
}
data_dict = data_proc.interpolate_data(data_dict, **interp_pars)
```

---

## D⁺ / rho-plus test

### `compute_rhoplus(freq, Z_scalar, Z_scalar_err=None, *, n_lambda_per_freq=4, mu0=_MU0)`

Computes the D⁺ upper bound ρ⁺ on apparent resistivity (Parker 1980;
Parker & Whaler 1981).  Also known as **dplus** in Cordell's mtcode.

For a 1-D earth the measured ρ_a must satisfy ρ_a(ω) ≤ ρ⁺(ω) at every
frequency.  Any single violation proves the data cannot be explained by any
1-D model.

The D⁺ model has the form c(ω) = Σ Δaₖ/(λₖ + iω), Δaₖ ≥ 0, where
c = Z/(iωμ₀) is the Schmucker c-response.  For a fixed log-spaced λ grid the
Δaₖ are found by non-negative least-squares (NNLS); ρ⁺ = μ₀ω|c⁺|² follows
directly.

```python
# Test Zxy for 1-D consistency
rho_plus, rho_a, ok = data_proc.compute_rhoplus(
    site['freq'], site['Z'][:, 0, 1],
    Z_scalar_err=site['Z_err'][:, 0, 1] if site['Z_err'] is not None else None,
)
print(f"D+ violations: {(~ok).sum()} of {len(ok)}")

# Also test Zyx, Z_det, etc.
z_det, _ = data_proc.compute_zdet(site['Z'])
rho_plus_det, rho_a_det, ok_det = data_proc.compute_rhoplus(site['freq'], z_det)
```

Returns `(rho_plus, rho_a, pass_test)` where `pass_test` is a boolean array
(`True` = passes, `False` = violation).

**References:** Parker (1980) *JGR* 85; Parker & Whaler (1981) *JGR* 86;
Parker & Booker (1996) *PEPI* 98; Cordell et al. (2022) mtcode.

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
Modified: 2026-03-07 — unified save_xxx(**data_dict, path=...) calling convention, Claude (Opus 4.6, Anthropic)
Modified: 2026-03-16 — freq_order parameter (load_edi, save_edi), compute_rhoplus (D+/rho+ test), PTXX/PTXY phase tensor blocks, RHOXY/PHASEXY rho-phase blocks in save_edi, MT unit fix (rho_a = |Z|²×1e6/(μ₀ω) for Z in mV/km/nT); Claude Sonnet 4.6 (Anthropic)
