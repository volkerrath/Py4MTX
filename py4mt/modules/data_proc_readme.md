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
- **One preset frequency grid for a whole survey** — `make_freq_grid`,
  `make_freq_grid_n`, `common_freq_grid`, `freq_grid_coverage`; each site gets the part of the grid
  that its band covers, optional extrapolation just outside the band
- **FT-convention detection** — `detect_ft_convention`,
  `scan_edi_header_convention`, `load_edi(manufacturer="auto")`
- **1-D forward modelling** — `mt1dfwd` and `wait1d` for layered-Earth
  impedance calculations
- **Site-list I/O** — parse FEMTIC sitelist CSV produced by
  `mt_make_sitelist.py` (`read_sitelist`)

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
| `manufacturer` | `str` | `"metronix"`, `"phoenix"`, `"delta"`, or `"auto"` |
| `ft_convention` | `str` | `"e-iwt"` (standard) or `"e+iwt_corrected"` (Phoenix, after fix) |
| `ft_detected`, `ft_detect_fraction`, `ft_detect_nvotes`, `ft_detect_source` | `str`, `float`, `int`, `str` | only with `manufacturer="auto"`: detected convention (`"e-iwt"`, `"e+iwt"`, `"undetermined"`), share of the votes, number of votes, and source (`"data"`, `"header"`, `"fallback:<manufacturer>"`) |

Metadata (if present): `station`, `lat_deg`, `lon_deg`, `elev_m`, and
convenience aliases `lat`, `lon`, `elev`.

All downstream modules treat this dict as the **site container**.

---

## Reading / writing EDI

### `load_edi(path, prefer_spectra=True, freq_order='inc', manufacturer='metronix', auto_fallback='metronix', ft_detect_pars=None, ...)`

Supports Phoenix/SPECTRA EDIs (reconstructed Z/T from `>SPECTRA` blocks) and
classical table EDIs (`>ZXXR`, `>ZXYI`, …).

The `freq_order` parameter controls the order of frequencies in the returned
dictionary:

| Value | Effect |
|-------|--------|
| `'inc'` | Ascending frequency — *default*, equivalent to the previous behaviour |
| `'dec'` | Descending frequency (i.e. ascending period) |
| `'keep'` | Preserve the order as found in the EDI file |

The `manufacturer` parameter declares which instrument produced the EDI and
controls the FT sign-convention correction applied on load:

| Value | FT convention | Correction applied |
|-------|---------------|--------------------|
| `'metronix'` | e⁻ⁱωᵗ (standard) | None |
| `'delta'` | e⁻ⁱωᵗ (standard) | None |
| `'phoenix'` | e⁺ⁱωᵗ | `Im(Z)` and `Im(T)` negated (conjugation) |
| `'auto'` | detected per file | negated only if the file is found to be e⁺ⁱωᵗ (see below) |

After loading, **all returned dicts are in the standard e⁻ⁱωᵗ convention**
regardless of manufacturer.  The keys `"manufacturer"` and `"ft_convention"`
record what was done.

```python
# Metronix / DELTA — no correction needed (default)
site = data_proc.load_edi("SITE.edi")

# Phoenix — FT convention corrected automatically
site = data_proc.load_edi("SITE_PHX.edi", manufacturer="phoenix")
print(site["ft_convention"])   # "e+iwt_corrected"

# Writer unknown (many processing codes do not put the manufacturer into
# the EDI): detect the convention from the data
site = data_proc.load_edi("SITE_UNKNOWN.edi", manufacturer="auto")
print(site["ft_detected"], site["ft_detect_fraction"], site["ft_detect_source"])

# Descending frequency (ascending period)
site = data_proc.load_edi("SITE.edi", freq_order="dec")

# Preserve EDI file order without any sorting
site = data_proc.load_edi("SITE.edi", freq_order="keep")
```

### FT-convention detection (`detect_ft_convention`, `scan_edi_header_convention`)

An EDI does not say whether its Z and T are e⁻ⁱωᵗ or e⁺ⁱωᵗ, and not every
processing code writes its manufacturer into the file.  The two conventions
are complex conjugates, and they can be told apart from the data (x = north,
y = east, z = down): for a layered, 2-D or 3-D Earth the phase of the
off-diagonal impedances lies in the 1st/3rd quadrant for e⁺ⁱωᵗ
(`Zxy` = +45° over a half-space) and in the 4th/2nd quadrant for e⁻ⁱωᵗ
(`Zxy` = −45°).

`detect_ft_convention(data, *, header=None, margin_deg=5, min_votes=6,
decisive=0.75)` counts, per frequency, the sign of `sin(arg Zdet)`
(rotation-invariant), `sin(2 arg Zxy)` and `sin(2 arg Zyx)`.  A reversed
sensor polarity (phase + 180°) or exchanged x/y axes do not change these signs.
Points within `margin_deg` of a quadrant boundary are ignored.  The decision
is taken from the Zdet votes (≥ `min_votes` and a share of ≥ `decisive`),
else from the pooled Zxy/Zyx votes (twice as many needed), else from an
explicit `exp(±iwt)` tag in the header, else the result is `None`
(inconclusive).  If the data and a header tag disagree the data win
(`agree=False`).

`scan_edi_header_convention(text)` looks for such an explicit tag
(`exp(-iwt)`, `exp(+i*omega*t)`, `e^{-iwt}`, `e-iwt`, …), a manufacturer
fingerprint (informative only) and the `PROGVERS` string.  Files written by
`save_edi` carry such a tag (below).

```python
r = data_proc.detect_ft_convention(site)         # site["Z"] as read, before any correction
print(r["convention"], r["fraction"], r["n_votes"], r["reason"])
```

Limits: not reliable if most phases are outside 0–90° (strong current
channelling, wrong polarity of one channel only) or with very few
frequencies — check `fraction`; inconclusive results fall back to
`auto_fallback` in `load_edi` with a `UserWarning`.  Tested on synthetic data
(layered Earth, rotated 0–137°, noise up to 20 %): never wrong, inconclusive
only above ~40 % noise.

### `save_edi(edi, path, ..., write_convention=True)`

Writes a classical table-style EDI from an in-memory dict.

**Invalid data** — every NaN, ±inf or |value| ≥ 1e32 is written as
`1.000000E+32`; the `>HEAD` block says `EMPTY=1.0E32`; `load_edi` treats
everything above `invalid_sentinel` (1e30) as invalid.  Two cases to know:
with `drop_invalid_periods=True` (the default) a frequency with an invalid
value in a component that is otherwise valid is dropped on loading, so it never
reaches the output file; a component that is invalid at *all* frequencies
(e.g. both diagonals) is set to 0 on loading, so it is written as `0.0` with
variance `0.0`, not as 1e32.  With `drop_invalid_periods=False` the 1e32
markers are kept.

**FT convention** — with `write_convention=True` (default) an `>INFO` line
`Fourier convention of Z and T: exp(-iwt); Z in mV/km/nT; invalid values: 1.0E32`
is written when `edi["ft_convention"]` is known (`"e-iwt"` and
`"e+iwt_corrected"` give `exp(-iwt)`, `"e+iwt"` gives `exp(+iwt)`), so that the
file can be recognised later by `scan_edi_header_convention`.  The `>FREQ` line
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

## Error estimation from curve scatter

### `estimate_errors(edi_dict, method)`

Replaces `Z_err` (and `T_err`, `P_err`) with data-driven estimates derived
from the scatter of each component relative to a smooth curve fitted on the
original frequency grid.  **`freq` and `Z` are not modified.**

The `method` dict controls the strategy:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `"kind"` | str | `"spline_residual"` | Estimation method (see below) |
| `"lam"` | float\|None | `None` | Primary spline λ; `None` → GCV selection |
| `"lam_smooth"` | float\|None | `None` | Envelope smoothing λ (smoothed variant only) |
| `"window_dec"` | float | `1.0` | MAD window half-width in decades |
| `"floor_frac"` | float\|None | `None` | Fractional floor: σ_ij ≥ floor_frac × \|Z_ij\| |
| `"components"` | str | `"ZT"` | Which arrays to process: `"Z"`, `"ZT"`, or `"ZTP"` |

**Methods (`"kind"`):**

| Value | Description |
|-------|-------------|
| `"spline_residual"` | GCV smoothing spline fitted to Re and Im separately; σ = √(r_re² + r_im²) pointwise. Fast, locally adaptive. |
| `"spline_residual_smoothed"` | Same as above, then smooth the σ envelope with a second spline (`lam_smooth`). Suppresses single-point spikes driving adjacent errors. |
| `"mad"` | Sliding-window median absolute deviation in log-frequency space, combined as √(mad_re² + mad_im²). Most robust to clustered outliers; requires sufficient point density per window. |

All methods set `err_kind = "std"` in the returned dict.

The `floor_frac` option prevents near-zero errors on extremely clean sections
of the curve where the spline fits almost perfectly.  A value of `0.02`
(2 % of \|Z\|) is a reasonable starting point.

```python
# Basic spline-residual estimate
method = {"kind": "spline_residual", "floor_frac": 0.02}
site = data_proc.estimate_errors(site, method)

# Smoothed envelope — less sensitive to isolated noisy periods
method = {
    "kind":       "spline_residual_smoothed",
    "lam_smooth": 1e-2,
    "floor_frac": 0.02,
}
site = data_proc.estimate_errors(site, method)

# MAD — robust alternative, wide window
method = {"kind": "mad", "window_dec": 1.5, "floor_frac": 0.03}
site = data_proc.estimate_errors(site, method)

# Combine with set_errors to floor against a minimum level
site = data_proc.estimate_errors(site, {"kind": "spline_residual"})
site = data_proc.set_errors(site, mode="floor",
                             Z_rel=[0.02]*4, Z_rel_mode="ij")
```

---

## Interpolation and common frequency grids

Inversions that are parallelised per frequency need the sites on **the same
frequencies**.  Preset one grid for the survey and give it to every site; each
site is written on the part of the grid that its own band covers:

```python
grid = data_proc.common_freq_grid(sites, freq_per_dec=6)          # band of the whole survey
grid = data_proc.common_freq_grid(sites, n_freq=25, freq_range=(1e-2, 1e2))   # exactly 25 points
sites = [data_proc.interpolate_data(d, newfreqs=grid) for d in sites]
# with a margin of 1/6 decade for extrapolation just outside a site's band:
sites = [data_proc.interpolate_data(d, newfreqs=grid, extrap_decades=1/6) for d in sites]
```

Every frequency of every site is then a member of the grid, sites with the same
band get the same frequencies, and a site with a short band has fewer
frequencies (no data are made up outside its band, except within
`extrap_decades`).  FEMTIC's `observe.dat` keeps its own frequency list per
station, so this is what `edi_list_to_observe_dat` writes.

### `make_freq_grid(freq_per_dec, fmin, fmax, *, anchor=1.0, rtol=1e-3, order="inc")`

Grid points `f_k = anchor * 10**(k / freq_per_dec)` with **integer k**, those in
`[fmin, fmax]` (a point within `rtol` outside is kept).  A grid point does not
depend on the band, so grids of different sites are subsets of one grid, and
with `anchor = 1` the decades (..., 0.1, 1, 10, ...) are always members.

### `make_freq_grid_n(n_freq, fmin, fmax, *, order="inc")`

**Exactly `n_freq` frequencies**, equally spaced in log f, with `fmin` and `fmax`
themselves as the first and last point (exactly, not only within rounding); the
spacing is `log10(fmax/fmin)/(n_freq-1)` decades.  Unlike `make_freq_grid` the
frequencies are not anchored on the absolute decades, so a grid made for another
band is not a subset of this one; that does not matter when one grid is made for
the whole survey.  `n_freq` must be an integer >= 2 (1 only for `fmin == fmax`),
else `ValueError`.

### `common_freq_grid(freqs, freq_per_dec=None, *, n_freq=None, freq_range="survey", anchor=1.0, rtol=1e-3, order="inc", extrap_decades=0.0, names=None, verbose=True)`

`freqs`: one frequency array (or site dict) per site.  Give **exactly one** of
`freq_per_dec` (points per decade, `make_freq_grid`) and `n_freq` (exactly that
many frequencies between the limits, `make_freq_grid_n`; `anchor` and `rtol`
then do not shape the grid); otherwise `ValueError`.  A site gets the grid points
that lie in its own band, so with `n_freq` and `freq_range=(fmin, fmax)` every
site whose band covers the limits has exactly `n_freq` frequencies.

| `freq_range` | Grid band |
|--------------|-----------|
| `"survey"` (default) | lowest minimum to highest maximum of all sites: no site limits the grid, sites with a shorter band get a subset |
| `"common"` | only the band that **all** sites cover (highest minimum to lowest maximum): every site has every frequency; `ValueError` if there is none |
| `(fmin, fmax)` | explicit band [Hz]: the preset grid |

With `verbose=True` it prints the grid, the band of the survey, how many sites
cover the whole grid, and the sites that get only part of it (number of points,
band, number extrapolated); a site without any grid frequency is flagged.
`extrap_decades` only changes this report.

### `freq_grid_coverage(grid, freqs, *, extrap_decades=0.0, rtol=1e-3, names=None)`

Which grid frequencies does each site get?  Returns a dict with the boolean
arrays `keep` and `extrapolated` (nsite, ngrid), the counts `n_keep`,
`n_extrap`, and the bands `fmin`, `fmax`.  The selection is the one made by
`interpolate_data`, so it can be used to plan or check an inversion (which
frequency has how many sites).

### `interpolate_data(edi_dict, *, newfreqs, freq_per_dec, interp_method, anchor, rtol, extrap_decades, outside)`

Resamples Z, T, P, errors and rot onto the new frequencies in log-frequency.

| Parameter | Description |
|-----------|-------------|
| `newfreqs` | The preset grid [Hz]; the site gets those of its frequencies that its band covers, in the order given; takes priority over `freq_per_dec` |
| `freq_per_dec` | Points per decade on the absolute-decade grid (`make_freq_grid`) inside this site's own band; use `common_freq_grid` + `newfreqs` for a survey |
| `anchor`, `rtol` | As in `make_freq_grid` |
| `extrap_decades` | Default `0.0`.  Grid frequencies up to this far **outside** the band are kept and extrapolated (`1/x` for "less than 1/x of a decade") |
| `outside` | `"clip"` (default): grid frequencies beyond band + `extrap_decades` are left out; `"raise"`: `ValueError`.  If nothing is left, `ValueError` in both cases |
| `interp_method` | currently ignored (Z, T, P, rot: GCV smoothing spline) |

The result has the new boolean key `extrapolated` (one flag per frequency).

**Extrapolation**: Z, T, P and rot continue as a straight line in log f, using
the value and slope of the smoothing spline at the edge of the band (a
cubic would run away); the error arrays keep their value at the edge.  Extrapolated
values are guesses carrying the error of the last measured frequency: keep the
margin small (about one grid step), and inflate their errors using the flag if
the inversion should not trust them.  Gaps *inside* a band are bridged by the
spline.

Changes of 2026-11-20:

- `freq_per_dec` used to give a `linspace` from the site's own lowest to highest
  frequency, i.e. different frequencies at every site.
- Target frequencies outside the band were first refused (`allow_extrapolation`,
  removed the same day) and are now clipped or extrapolated as described above.
- The `freq_order` tag now matches the new array (the old code left `"dec"`
  with an ascending array); `freq_per_dec` grids follow the stored order.
- Error arrays (`Z_err`, `T_err`, `P_err`) are interpolated **linearly in
  log-frequency**: the spline made variances negative (down to -10 in a noisy
  test), which gave NaN in every later error propagation.
- Derived per-frequency keys (`Zdet`, `Zssq`, `Zavg`, `rho`, `phi`, their
  errors, `dplus`) are removed because they refer to the old grid; compute them
  after the interpolation (as `mt_data_processor.py` does now).

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

## Site-list I/O

### `read_sitelist(path, site_names=None)`

Parses a comma-separated FEMTIC sitelist produced by `mt_make_sitelist.py`
with `WHAT_FOR="femtic_site"`.

**Column order** (no column header; lines starting with `#` are skipped.
Since 2026-11-20 the list starts with two `#` lines naming the UTM zone,
e.g. `# utm_zone=33 hemisphere=N mode=common central_meridian=15.0`; all
`easting`/`northing` are in that one zone, also if the sites lie on both sides
of a zone boundary, see `mt_make_sitelist_readme.md`):

```
name, lat, lon, elev, sitenum, easting, northing
```

| Column | Type | Description |
|--------|------|-------------|
| `name` | str | Station name |
| `lat` | float | WGS-84 latitude [°] |
| `lon` | float | WGS-84 longitude [°] |
| `elev` | float | Elevation [m] |
| `sitenum` | int | 0-based site index |
| `easting` | float | UTM easting [m] |
| `northing` | float | UTM northing [m] |

Returns a list of dicts, one per (matching) site.  The `site_names` argument
accepts a single name string or a list of strings; matching is
case-insensitive.  `None` (default) returns all sites.

**No coordinate-system conversion is applied.**  Callers that need
model-local metres should follow up with `fem.utm_to_model` (femtic.py).

```python
import data_proc as dp

# All sites
sites = dp.read_sitelist("Sitelist_femtic.txt")

# Filtered subset
sites = dp.read_sitelist("Sitelist_femtic.txt",
                          site_names=["MT01", "MT05", "MT12"])

# Convert to model-local metres (requires mesh-centre origin)
import femtic as fem
for s in sites:
    x_m, y_m = fem.utm_to_model(s["easting"], s["northing"],
                                  UTM_ORIGIN_E, UTM_ORIGIN_N)
```

---

## Building a station collection from a directory of EDI files

### `make_collection(edi_dir, *, ext=".edi", save_path=None, manufacturer="metronix", drop_invalid_periods=True, freq_order="inc", ft_correction=None, compute_invariants=True, compute_phase_tensor=False, use_filename_as_station=True, station_upper=False, verbose=True)`

Reads every EDI file in `edi_dir`, computes the impedance invariants
(`Zdet`, `Zssq`, `Zavg`, plus their `*_err` companions) via `compute_zdet`,
`compute_zssq`, `compute_zavg`, and returns the resulting list of station
dicts — the exact `records` layout expected by `mt_get_averages.py`.

This factors out the "read EDI directory → compute invariants → collect"
portion of `mt_data_processor.py`'s per-station loop into one reusable call,
without any of that script's export/plotting machinery (no EDI/NPZ/HDF/MAT
export, no diagnostic plots, no error setting/interpolation/rotation).

```python
import data_proc as dp

# In-memory only
records = dp.make_collection("/path/to/edi_dir/")

# Also cache it to disk as a COLL_FILE for mt_get_averages.py
records = dp.make_collection(
    "/path/to/edi_dir/",
    save_path="/path/to/Review_collection.npz",
)
```

| Argument | Default | Effect |
|----------|---------|--------|
| `ext` | `".edi"` | File extension searched for in `edi_dir` |
| `save_path` | `None` | If given, write `records` via `save_list_of_dicts_npz` (a `COLL_FILE`) |
| `manufacturer` | `"metronix"` | Forwarded to `load_edi` |
| `drop_invalid_periods` | `True` | Forwarded to `load_edi` |
| `freq_order` | `"inc"` | Forwarded to `load_edi` |
| `ft_correction` | `None` | If given, e.g. `{"from_convention": "e+iwt", "to_convention": "e-iwt"}`, applied via `correct_ft_convention` right after loading |
| `compute_invariants` | `True` | Compute `Zdet`/`Zssq`/`Zavg` (+ `*_err`) |
| `compute_phase_tensor` | `False` | Also compute `P`/`P_err` via `compute_pt` (not needed by `mt_get_averages.py`) |
| `use_filename_as_station` | `True` | Station name taken from the EDI filename stem rather than the EDI header (mirrors `STAT_FILE` in `mt_data_processor.py`) |
| `station_upper` | `False` | Upper-case the station name |
| `verbose` | `True` | Print one line per station, plus a summary if `save_path` is given |

Returns the list of station dicts (also written to `save_path` if given, in
the same format read by `load_list_of_dicts_npz`).

`mt_get_averages.py` can call this directly (`BUILD_FROM_EDI = True`) to
skip the intermediate `COLL_FILE` entirely, or use it to (re)generate a
`COLL_FILE` cache.

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
Modified: 2026-03-25 — manufacturer parameter in load_edi (phoenix/metronix/delta); FT-convention correction (conjugation of Z and T for Phoenix); manufacturer and ft_convention keys in data_dict; Claude Sonnet 4.6 (Anthropic)
Modified: 2026-04-27 — estimate_errors section rewritten: spline_residual, spline_residual_smoothed, and mad methods; Claude Sonnet 4.6 (Anthropic)
Modified: 2026-05-23 — added read_sitelist(): parse FEMTIC sitelist CSV (name,lat,lon,elev,sitenum,easting,northing) with optional name filter; raw values only, no CRS conversion; Claude Sonnet 4.6 (Anthropic)
Modified: 2026-09-07 — added make_collection(): build a station-dict collection directly from a directory of EDI files (load_edi + optional FT correction + Zdet/Zssq/Zavg invariants), returned in-memory and optionally saved via save_list_of_dicts_npz() as the COLL_FILE consumed by mt_get_averages.py; Claude Sonnet 5 (Anthropic)
Modified: 2026-11-20 -- common frequency grids (make_freq_grid, common_freq_grid; interpolate_data anchored grid, anchor/rtol, freq_order tag, linear interpolation of the error arrays, derived keys removed); FT-convention detection (detect_ft_convention, scan_edi_header_convention, load_edi manufacturer="auto", auto_fallback, ft_detect_pars, keys ft_detected / ft_detect_fraction / ft_detect_nvotes / ft_detect_source); save_edi write_convention (>INFO line) and documentation of the invalid-value marker; Claude Sonnet 5.5 (Anthropic), AI-generated, review before use
Modified: 2026-11-20 (b) -- one preset survey grid, each site on the part its band covers: common_freq_grid freq_range "survey" (default) / "common" / (fmin, fmax) with a coverage report; new freq_grid_coverage; interpolate_data clips (outside="clip") or raises, extrap_decades (1/x decade) extrapolates just outside the band, new key "extrapolated"; allow_extrapolation removed; Claude Sonnet 5.5 (Anthropic), AI-generated, review before use
Modified: 2026-11-20 (c) -- `read_sitelist()` docs: one UTM zone for all sites of a list (also across zone boundaries), zone named in the `# utm_zone=...` comment line, which is skipped; Claude Sonnet 5.5 (Anthropic); AI-generated, review before production use
Modified: 2026-11-20 (d) -- make_freq_grid_n(): exactly n_freq log-spaced frequencies including both limits; common_freq_grid(n_freq=...), freq_per_dec now optional (exactly one of the two); Claude Sonnet 5.5 (Anthropic), AI-generated, review before use
