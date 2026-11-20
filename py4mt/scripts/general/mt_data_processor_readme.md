# mt_data_processor.py

Batch MT station processing script.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_data_processor.py` |
| Author | Volker Rath (DIAS), with ChatGPT (GPT-5 Thinking), 2026-02-13 |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 3 March 2026 by Claude (Anthropic), from cleaned source |
| Modified | 2026-03-07 — unified save_xxx(**data_dict, path=...) calling convention |
| Modified | 2026-03-16 — freq_order, D+/rho+ test (DPLUS), add_rhoplus plot; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-03-18 — add_noise option in SET_ERRORS (mode='fix' only), random_state passthrough; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-03-30 — adapt plot block to updated data_viz: per-call ylim overrides (no PLTARGS mutation), DPLUS subplot row, removed bogus required-key guard from add_tipper/add_pt; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-12 — `correct_ft_convention` imported from `data_proc`; Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-06-13 — `compute_zavg` added to `INVARS` block (`Zavg`, `Zavg_err`); Claude Sonnet 4.6 (Anthropic) |
| Modified | 2026-11-20 — one preset frequency grid for all stations (`freq_range`, `anchor`, `extrap_decades`), each station on the part of it that its band covers, interpolation first; `MANUFACTURER = "auto"` (FT convention detected from the data); `observe` token in `OUT_FILES` (FEMTIC `observe.dat` with dummy VTF stations); Claude Sonnet 5.5 (Anthropic), AI-generated, review before use |

## Purpose

Reads all `.edi` files from an input directory, computes derived quantities
(phase tensor, impedance invariants, apparent resistivity/phase), optionally
sets errors, estimates errors, interpolates, or rotates, then exports each
station in one or more formats. A collection NPZ containing all stations is
saved at the end.

## Workflow

1. **Read** all `.edi` files from `DATA_DIR_IN` via `data_proc.load_edi()`
   (with `MANUFACTURER = "auto"` the FT convention is detected per station and a
   summary table is printed), apply the FT correction.
2. **Frequency grid** (`INTERPOLATE = True`): one preset grid for the survey,
   `data_proc.common_freq_grid()`; the report names the stations that get only
   part of it.
3. **Per station**: interpolate onto the part of the grid that the station's
   band covers (first, so that everything below lives on these frequencies),
   phase tensor, invariants, errors (set / floor), rotation, rho/phi,
   D+/rho+ test, then the exports and plots.  A station without any grid
   frequency is skipped with a warning.
4. **Coverage report** (when `INTERPOLATE`): stations per frequency, and
   `<COLL_NAME>_freq_coverage.csv` (0 = no data, 1 = data, 2 = extrapolated).
5. **Export** each station in one or more formats (EDI, NPZ, HDF, MAT), and, with
   the token `observe`, one FEMTIC `observe.dat` for all stations.
6. **Save** a collection NPZ containing all stations.

## Configuration constants

| Constant | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | *(set per project)* | Root working directory |
| `EDI_DIR` | `WORK_DIR + "/orig/"` | Input EDI directory |
| `DATA_DIR` | `WORK_DIR + "/proc/"` | Output directory (created if missing) |
| `OUT_FILES` | `"edi, npz, hdf, mat"` | Comma-separated output formats: `edi`, `npz`, `hdf`, `mat`, `ncd`, and `observe` (one FEMTIC `observe.dat` for all stations, see below) |
| `NAME_STR` | `"_proc"` | Suffix appended to output filenames |
| `COLL_NAME` | `"ANN_DJ_aniso"` | Name for the collection NPZ |
| `FREQ_ORDER` | `"inc"` | Frequency order passed to `load_edi()`: `"inc"`, `"dec"`, or `"keep"`; the common grid follows it |
| `MANUFACTURER` | `"metronix"` | `"metronix"`, `"phoenix"`, `"delta"`, or `"auto"` (detect e-iwt / e+iwt from the data; see below) |

### Processing switches

| Constant | Default | Description |
|----------|---------|-------------|
| `PHAS_TENS` | `True` | Compute phase tensor via `compute_pt()` |
| `INVARS` | `True` | Compute Zdet, Zssq, and Zavg invariants |
| `DPLUS` | `True` | D+/rho+ test via `compute_rhoplus()` on Zxy, Zyx, Zdet; prints violation counts; stores results in `data_dict["dplus"]` |
| `SET_ERRORS` | `False` | Override errors with fixed relative values |
| `ESTIMATE_ERRORS` | `False` | Estimate errors from data (work in progress) |
| `INTERPOLATE` | `False` | Interpolate to uniform frequency sampling |
| `ROTATE` | `False` | Rotate impedance/tipper by a fixed angle |
| `PLOT` | `False` | Generate per-station diagnostic plots |
| `STAT_FILE` | `True` | Use EDI filename (not station header) for output names |

### Error configuration (when `SET_ERRORS = True` or `SET_ERROR_FLOORS = True`)

```python
ADD_NOISE = False   # perturb data by N(0, σ); only valid with mode="fix"
err_pars = {
    "Z_rel":        [0.1, 0.1, 0.1, 0.1],  # relative Z errors [xx, xy, yx, yy]
    "Z_rel_mode":   "ij",                    # "ij" or "ij*ii"
    "T_abs":        [0.03, 0.03],            # absolute tipper errors [Tx, Ty]
    "PT_abs":       [0.1, 0.1, 0.1, 0.1],   # absolute PT errors [xx, xy, yx, yy]
    "add_noise":    ADD_NOISE,
    "random_state": rng,                     # module-level Generator; seed for reproducibility
}
```

Passed as `set_errors(data_dict, mode="fix", **err_pars)` or `mode="floor"`.
`SET_ERRORS` and `SET_ERROR_FLOORS` are independent and can both be active
(floor applied after set).

When `ADD_NOISE = True`, each data array is perturbed by independent Gaussian
noise drawn from N(0, σ), where σ is the fixed error level computed for that
component.  For the complex arrays Z and T the noise is split equally between
real and imaginary parts (each N(0, σ/√2)); for the real-valued phase tensor P
a single draw from N(0, σ) is used.  The unperturbed originals are preserved in
the dict as `Z_orig`, `T_orig`, and `P_orig` (only for components that were
actually perturbed).  Noise perturbation is only permitted with `mode="fix"` and
raises a `ValueError` if combined with `mode="floor"`.

### FT convention (`MANUFACTURER`)

`"metronix"` / `"delta"` are e-iwt, `"phoenix"` is e+iwt (Im Z, Im T negated on
loading).  Use `MANUFACTURER = "auto"` for EDIs of an unknown writer (many
processing codes do not put their name into the file):

```python
MANUFACTURER = "auto"
FT_AUTO_FALLBACK = "metronix"          # used if the detection is inconclusive
FT_DETECT_PARS = {"margin_deg": 5.0, "min_votes": 6, "decisive": 0.75}
```

The convention is decided from the data (sign of the phase quadrant of Zdet,
Zxy, Zyx; see `data_proc.detect_ft_convention`), an explicit `exp(+-iwt)` tag in
the EDI is used when the data are inconclusive, and the data win when the two
disagree.  A table (station, detected convention, share of the votes, number of
votes, source) is printed; check the lines marked as fallback.  After loading,
all data are e-iwt: keep `ftc_pars["from_convention"] = "e-iwt"`, otherwise
the data are flipped twice.  Files written by this script carry the line
`Fourier convention of Z and T: exp(-iwt)` in `>INFO`.

### Interpolation (when `INTERPOLATE = True`)

All inversions are parallelised per frequency, so the stations are put on **one
preset grid**, and each station is written on the part of it that its own band
covers:

```python
interp_pars = {
    "freq_per_dec":   6,          # points per decade; None if n_freq is used
    "n_freq":         None,       # exactly this many frequencies; None if freq_per_dec is used
    "freq_range":     "survey",   # "survey", "common", or (fmin, fmax) in Hz
    "anchor":         1.0,        # Hz, f = anchor * 10**(k/freq_per_dec), k integer (not used with n_freq)
    "interp_method":  "gcvspline",   # currently ignored by interpolate_data
    "extrap_decades": 0.0,        # 1/x: extrapolate up to 1/x decade outside a station's band
}
```

- Two ways to define the grid; set **one** of `freq_per_dec` / `n_freq` and the
  other to `None` (both set, or neither, stops the script with a message):
  - `freq_per_dec` points per decade on absolute decades (the decades 0.1, 1, 10,
    ... are always grid points); the number of frequencies depends on the band.
  - `n_freq = N` **exactly N frequencies**, equally spaced in log f, with **both
    limits of the band as grid points** (`data_proc.make_freq_grid_n`); `anchor`
    is not used.  E.g. `n_freq = 25` with `freq_range = (0.01, 100.)` gives 25
    frequencies from 0.01 to 100 Hz (0.21 decade apart).  With `"survey"` or
    `"common"` the limits are the extreme frequencies found in the data
    (e.g. 0.00173 Hz), not rounded values; give `(fmin, fmax)` if you want round
    limits.  N counts the grid, not what a station gets: a station whose band is
    shorter than the limits gets only the grid points inside its band (every
    station gets exactly N if its band covers the limits, e.g. always with
    `"common"`).
- `freq_range = "survey"` (default) spans the band of the whole survey, so no
  station limits the grid; `"common"` is only the band that all stations cover; a
  tuple presets the band explicitly.
- A station gets exactly those grid frequencies that lie in its own band.  Nothing
  is made up outside the band, except for grid frequencies within
  `extrap_decades` of it (`1.0/6.0` = one grid step at 6 per decade; `0.0` = none).
  Those are **extrapolated** (straight line in log f from the value and slope at
  the band edge, errors keep their edge value) and flagged in
  `data_dict["extrapolated"]`; the script reports their number and the observe
  export notes them.
- `"newfreqs"` (explicit array [Hz]) may be given instead; it is used as it is.
- Stations with the same band have identical frequencies; every frequency of
  every station is a member of the grid.  The script prints the stations that get
  only part of the grid, the number of stations per frequency, and writes
  `<COLL_NAME>_freq_coverage.csv`.
- The interpolation runs **first**: phase tensor, invariants, errors and rho/phi
  are computed on the new frequencies.  Error arrays are interpolated linearly in
  log f (the spline gave negative variances).

### Rotation (when `ROTATE = True`)

```python
ANGLE = 0.0
DEC_DEG = True
```

## Output formats

For each station, the script can write: EDI (classical table-style),
NPZ (NumPy compressed archive), HDF (HDF5), MAT (MATLAB `.mat`), and
NCD (NetCDF, if added to `OUT_FILES`).

All save functions are called with the `**data_dict` splatting convention
(`save_xxx(**data_dict, path="...")`), where `path` is keyword-only.

After all stations are processed, a single collection NPZ is written via
`save_list_of_dicts_npz()`.

### FEMTIC `observe.dat` (token `observe` in `OUT_FILES`)

One `observe.dat` for all stations (`femtic.edi_list_to_observe_dat`), plus
`observe_ids.csv` (station, ID, H-ID):

```python
OUT_FILES = "edi, npz, observe"
OBSERVE_FILE = DATA_DIR_OUT + "observe.dat"
SITE_DAT = DATA_DIR_IN + "site.dat"     # name,lat,lon,elev,sitenum,easting,northing
OBS_ORIGIN_METHOD = "box"               # "box", "average", or None + OBS_ORIGIN_E/N
OBS_UTM_ZONE_OVERRIDE = None
OBS_DUMMY_VTF = True                    # add dummy VTF stations (OWNER_ELEMENT -1)
OBS_SITE_TYPE = "land"                  # "land" or "seafloor"
OBS_DUMMY_ID_OFFSET = 1000              # raised to a power of ten > number of stations
OBS_INVALID = "drop"                    # "drop" or "mark": how invalid data are written
OBS_INVALID_SE = -1.0                   # error written for a marked component (must be < 0)
```

- Positions are model-local E/N [m] (x = easting, y = northing, as in
  `fem.utm_to_model`) from `site.dat`, matched to the EDI file names
  (case-insensitive).  The origin must be the origin of the mesh the file is used
  with.
- With `OBS_DUMMY_VTF` every MT station gets `H-ID = ID + offset` and a dummy VTF
  station (0 frequencies) at the same place; set `OWNER_ELEMENT -1` in
  `control.dat`.  Land: MT owner 0, VTF owner 1.
- Each station is written with its frequencies of the grid (ascending; `observe.dat`
  keeps a frequency list per station), converted to SI (Ohm).  Warnings are printed for missing positions, e+iwt storage, and zero or
  non-finite errors.
- The name is `observe.dat`, as FEMTIC reads it.
- **Invalid data.**  FEMTIC has no marker value: a datum (each of the 8 real numbers
  Re/Im Zxx..Zyy per frequency) is used only if its error is `> 0`, and a negative
  error means "not used" (manual).  `OBS_INVALID = "drop"` leaves out every frequency
  with a non-finite value or error (as before); `"mark"` keeps the frequency and
  writes only the bad component with `OBS_INVALID_SE` (value 0.0 if it was not
  finite), so the valid components of that frequency are kept.  Components with an
  error `<= 0` (e.g. both diagonals zeroed because they were invalid at all
  frequencies) are marked in `"mark"` and written as 0 in `"drop"`; FEMTIC does not
  use them either way.  A frequency with nothing usable is always left out.  The script
  prints the counts per station.  Invalid values reach this stage only with
  `DROP_INVALID = False`, which must not be combined with `INTERPOLATE`
  (the 1e32 markers would go into the spline; the script warns).

### Invalid data in the written EDIs

`save_edi` writes every NaN, +-inf or |value| >= 1e32 as `1.000000E+32`
(`EMPTY=1.0E32` in the header).  With `DROP_INVALID = True` (default) a frequency
with an invalid value in an otherwise valid component (or in T or the variances)
is dropped on loading, so no marker reaches the output; a component that is
invalid at all frequencies (e.g. both diagonals) is set to 0 on loading and is
written as `0.0` with variance `0.0`, not as 1e32.  With `DROP_INVALID = False`
the markers are kept, and the phase tensor of those rows is NaN (written as 1e32).

## Diagnostic plots (when `PLOT = True`)

A 3×2 subplot figure is generated per station (4×2 when `DPLUS = True`) showing:

| Row | Left | Right |
|-----|------|-------|
| 1 | ρ_a (xy, yx) | Phase (xy, yx) |
| 2 | ρ_a (xx, yy) | Phase (xx, yy) |
| 3 *(DPLUS only)* | D+/rho+ test (xy, yx) | *(hidden)* |
| 3 (or 4) | Tipper | Phase tensor |

`nrows` is computed as `3 + (1 if DPLUS else 0)` so the figure height scales
automatically.  Per-panel `ylim` overrides are passed as `**{**PLTARGS, "ylim": (...)}` —
the shared `PLTARGS` dict is never mutated between calls.

Phase wrap-around (2026-11-20): where a phase curve leaves the +-180 deg window
and re-enters on the other side, the solid line and the error band are
interrupted and the jump is drawn as a dotted connection instead of a long
vertical streak.  Configuration (inside the `if PLOT:` block, passed to the
two `add_phase` calls only, not through `PLTARGS`):

| Constant | Default | Description |
|----------|---------|-------------|
| `PHASE_WRAP_MODE` | `"direct"` | `"direct"` = one line from point to point; `"short"` = two short stubs through the window edge; `"off"` = solid line through the jump (old) |
| `PHASE_WRAP_STYLE` | `{"linestyle": ":"}` | dict of Line2D keywords of the connection, e.g. `{"linestyle": (0, (1, 3)), "alpha": 0.5, "color": "0.4"}`; `False` = no connection |

See `data_viz_readme.md` for the details.

Figure title (2026-11-11): `FIGURE_TITLE` (default `"auto"` = `<station><NAME_STR with " | " for "_">` as before; `None`, `False`, `""` = none; text with `{station}`, `{name_str}`, `{key}` (= the station); dict `{"<station>": spec, "default": spec}`). It is read inside the `if PLOT:` block and resolved by `util.resolve_title` (needs the `util.py` of the same date).

Empty axes are removed. Plots are saved in all formats listed in `PLOT_FORMAT`
(default: PDF at 600 DPI).

## Dependencies

`numpy`, `matplotlib`; py4mt: `data_proc` (`load_edi`, `compute_rhoplus`, …), `data_viz` (`add_rhoplus`, …), `util`, `version`.

## Changes 2026-11-11 (titles)

`FIGURE_TITLE` for the station figures (see *Diagnostic plots*). Claude Sonnet 5.5 (Anthropic): AI-generated; review before use. Defaults give the figures of before. Only compile-checked and tested at the level of the title logic (`util.resolve_title`, `test_util_titles.py`); the script was not run end to end.

## Changes 2026-11-20 (frequency grid, FT detection, observe.dat)

Claude Sonnet 5.5 (Anthropic): AI-generated; review before use.

- **One preset frequency grid**: `interp_pars` has `freq_range` ("survey" default,
  "common", or a band), `anchor`, `extrap_decades`; the grid is built once by
  `data_proc.common_freq_grid`, and every station is written on the part of it
  that its band covers.  Before, `freq_per_dec` gave a `linspace` between the lowest
  and highest frequency of each station.  A coverage report and
  `<COLL_NAME>_freq_coverage.csv` replace the former "identical frequencies" check.
- **Interpolation first**: the derived quantities were left on the old grid, the
  D+ block used stale Z and errors, and the `freq_order` tag did not match the
  array; all fixed.
- **`MANUFACTURER = "auto"`**: FT convention from the data, summary table.
- **`observe` token**: FEMTIC `observe.dat` with dummy VTF stations.
- `PLOT_DIR` is only created when `PLOT = True` (it raised a `NameError` otherwise);
  the EDI list is sorted.
- Tests: `test_data_proc_freq_ft.py` (82 cases, synthetic layered Earth; they pass).
  The script was run end to end on a synthetic survey of 5 stations with different
  bands and mixed conventions (with stand-ins for `h5py` and `pyproj`, which were
  not installed where this was written); not yet run on your data.

## Changes 2026-11-20 (c) (invalid data in observe.dat)

Claude Sonnet 5.5 (Anthropic): AI-generated; review before use.  `OBS_INVALID`
("drop" as before, or "mark") and `OBS_INVALID_SE`; see *FEMTIC observe.dat*.
`femtic.edi_list_to_observe_dat` also treats 1e32 markers as invalid (before they were
written as huge valid numbers) and no longer writes `nan` errors.

## Changes 2026-11-20 (e) (exactly N frequencies)

Claude Sonnet 5.5 (Anthropic): AI-generated; review before use.  New `interp_pars`
key `n_freq` (see *Interpolation*); `freq_per_dec` and `n_freq` are alternatives.
Needs the `data_proc.py` of the same date (`make_freq_grid_n`, `common_freq_grid(n_freq=...)`).
Tests: `test_data_proc_freq_ft.py` (124 cases now); ran end to end on a synthetic
survey of 5 stations (explicit limits: every station exactly N identical
frequencies; "survey": subsets of one N-point grid; both / neither set: the script
stops with a message).

## Changes 2026-11-20 (d) (phase wrap-around in the plots)

Claude Sonnet 5.5 (Anthropic): AI-generated; review before use.  `PHASE_WRAP_MODE`
and `PHASE_WRAP_STYLE` (plot configuration) are passed to `data_viz.add_phase`; see
*Diagnostic plots*.  Needs the `data_viz.py` of the same date.  Ran end to end on a
synthetic survey (plots written); the phase-wrap logic itself is covered by
`test_data_viz_phase_wrap.py`.
