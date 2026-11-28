# mt_dimensionality.py - dimensionality indicators of a survey

Author: Claude (Anthropic), model Claude Sonnet 5.5, generated 2026-11-28.
The code is AI-generated and should be reviewed before production use.

Reads EDI files or py4mt `.npz` collections, estimates the dimensionality
indicators with the module `dimens` and writes a survey summary and one `.npz`
file per site. Method, units, limits and function reference: `dimens_readme.md`.
It replaces the earlier driver `mt_dimens.py` (delete that file and its readme).

| indicator | per frequency | per band |
|---|---|---|
| WAL class (WALDIM 1.2), strike, twist, shear | yes | class and averages |
| reason of "undetermined", flags | yes | counts |
| phase tensor, principal phases, alpha, beta, azimuth, ellipticity | yes | means |
| phase tensor class 1D / 2D / 3D | yes | most frequent |
| Swift skew, Bahr skew | yes | medians |
| class at other thresholds | yes | yes |
| bootstrap agreement (optional) | yes | yes |

Contents: [Requirements](#requirements), [Quick start](#quick-start),
[Configuration](#configuration), [Examples](#examples), [Input](#input),
[What the script does](#what-the-script-does),
[Screen report](#screen-report), [Output files](#output-files),
[Class codes](#class-codes), [Reading the results](#reading-the-results),
[Messages](#messages), [Test run](#test-run), [References](#references),
[Changelog](#changelog).

## Requirements

- Python 3 with NumPy.
- py4mt modules in `$PY4MTX_ROOT/py4mt/modules/`: `dimens.py` (new), `util.py`,
  `version.py`, and `data_proc.py` for EDI input.
- Their packages: `h5py` and `pyproj` (imported by `util.py`); `pandas` and
  `scipy` (imported by `data_proc.py`).
- Environment variables `PY4MTX_ROOT` and `PY4MTX_DATA`.

## Quick start

1. Copy `dimens.py` to `py4mt/modules/` and `mt_dimensionality.py` to
   `py4mt/scripts/general/`.
2. Put the EDI files in `$PY4MTX_DATA/dimensionality/edi/`, or set `WORK_DIR` and
   `INPUT`.
3. Set `EDI_OPTIONS` for your instrument.
4. Run:

```
python mt_dimensionality.py
```

5. Read `$PY4MTX_DATA/dimensionality/mt_dimensionality/mt_dimensionality.txt`.

There are no command line options: all settings are in the configuration block
at the top of the script.

## Configuration

**Input**

| name | default | meaning |
|---|---|---|
| `WORK_DIR` | `PY4MTX_DATA + "/dimensionality/"` | working directory |
| `INPUT` | `WORK_DIR + "edi/"` | EDI directory, EDI file, `.npz` collection or station file, pattern with `*` / `?`, or a list of these |
| `EDI_OPTIONS` | `{"manufacturer": "metronix"}` | further `load_edi` keywords (EDI input only) |
| `USE_FILENAME_AS_STATION` | `True` | EDI files: name from the file name; `False`: DATAID of the header |

**Output**

| name | default | meaning |
|---|---|---|
| `OUT_DIR` | `WORK_DIR + "mt_dimensionality/"` | summary and tables |
| `SUMMARY_NAME` | `"mt_dimensionality"` | name of the summary files and tables |
| `SITE_NPZ_DIR` | `OUT_DIR + "sites/"` | one file per site; `None`: none |
| `SITE_NPZ_SUFFIX` | `"_dim"` | file name `<site><suffix>.npz` |
| `WALDIM_TABLES` | `True` | write the tables for `mt_kmz_waldim.py` |

**Errors**

| name | default | meaning |
|---|---|---|
| `ERROR_PERCENT` | `None` | number: ignore the data errors; standard error = this percentage of max(\|Re\|, \|Im\|) of each element |
| `MISSING_ERROR_PERCENT` | `5.0` | the same, only for sites without errors; `None`: stop at such a site |

**WAL classification**

| name | default | meaning |
|---|---|---|
| `THRESHOLD` | `0.15` | limit of the invariants I3 to I7 |
| `Q_THRESHOLD` | `0.10` | limit of the invariant Q |
| `N_REALIZATIONS` | `100` | noisy tensors for the bias test and the errors |
| `SEED` | `0` | random seed |
| `WALDIM_COMPAT` | `False` | `True`: rules of WALDIM 1.2 |

**Phase tensor**

| name | default | meaning |
|---|---|---|
| `FT_CONVENTION` | `"auto"` | time dependence of the data: `"auto"`, `"e+iwt"`, `"e-iwt"` |
| `PT_BETA_THRESHOLD` | `3.0` | deg; \|beta\| above it: 3D |
| `PT_ELLIPTICITY_THRESHOLD` | `0.1` | otherwise 2D above it, else 1D |

**Reliability of the WAL class**

| name | default | meaning |
|---|---|---|
| `STABILITY_THRESHOLDS` | `(0.10, 0.12, 0.18, 0.20)` | other values of `THRESHOLD` to try; `None`: none |
| `N_BOOTSTRAP` | `0` | repetitions with added noise, e.g. 200; `0`: none |

**Bands**

| name | default | meaning |
|---|---|---|
| `BANDS` | `True` | band summaries; `False`: one summary over all data |
| `BANDS_PER_DECADE` | `1.0` | bands per decade |
| `PERIOD_RANGE` | `None` | (Tmin, Tmax) in s; `None`: all data |

Without `PERIOD_RANGE` the band edges are 10**(k / `BANDS_PER_DECADE`), cover all
data and are common to all sites. With it, bands start at Tmin and an incomplete
last band is left out, as in WALDIM. Each band is (lower, upper]: a period equal
to Tmin is in no band. Periods outside the bands remain in the per-frequency
results and in the counts per site.

## Examples

Change these lines in the configuration block.

Phoenix EDI files, 5 percent errors where there are none:

```python
EDI_OPTIONS = {"manufacturer": "phoenix"}
MISSING_ERROR_PERCENT = 5.0
```

A collection written by `mt_data_processor.py`:

```python
INPUT = WORK_DIR + "proc/MySurvey_proc_collection.npz"
```

Two bands per decade from 1 ms to 1000 s, with bootstrap:

```python
BANDS_PER_DECADE = 2.0
PERIOD_RANGE = (0.001, 1000.0)
N_BOOTSTRAP = 200
```

Rules and errors as in WALDIM:

```python
ERROR_PERCENT = 5.0
WALDIM_COMPAT = True
```

`WALDIM_COMPAT` restores the rules of the original, not its random numbers or its
corrected source errors (`dimens_readme.md`).

## Input

- EDI files are read with `data_proc.load_edi`. `manufacturer` is `"metronix"`
  (default), `"phoenix"`, `"delta"` or `"auto"`.
- A directory is searched for `*.edi` files (not its subfolders). A file matched
  by a pattern is read as `.npz` or else as EDI.
- `.npz` collections (key `records`) and station files are read as stored;
  `EDI_OPTIONS` does not apply to them. A collection made with the `load_edi`
  default `drop_invalid_periods=True` may lack rows that the EDI files hold.
  **`.npz` files hold pickled objects: read only files that you trust.**
- Impedances are expected in mV/km/nT.
- Station names must be unique over all inputs, also ignoring case if site files
  are written (an EDI directory and the collection made from it usually clash).
  The script then stops with a message and writes nothing.

## What the script does

1. Expands patterns in `INPUT`.
2. Runs `dimens.classify_dimensionality` on all sites.
3. Prints a report per site.
4. Writes the site files (this checks the names), then the summary, then the
   WALDIM tables. Existing files are overwritten; files of earlier runs with
   other names remain.

Return code 0 on success, otherwise 1.

## Screen report

```
 DELTA_20: 14 frequencies (0 removed, errors: Z_err)
   WAL classes : undetermined 4, 1D 4, 2D 2, 3D/1D2D 4
   undetermined: no rule 4
   PT classes  : 1D 14  (data read as e+iwt, 100 % of frequencies agree)
   medians     : |beta| 0.2 deg, ellipticity 0.05, Swift 0.07, Bahr 0.06
   mean threshold agreement: 0.69
       0.001 -      0.01 s  2D           PT 1D        strike  87.3 (3.0)
        0.01 -       0.1 s  1D           PT 1D
         0.1 -         1 s  3D/1D2D      PT 1D
```

| item | meaning |
|---|---|
| removed | rows with invalid frequency, impedance, error or rotation |
| errors | `Z_err`: from the data; `percent`: percentage errors |
| undetermined | reasons: `bias test`, `error bars`, `no rule` |
| data read as | time convention used for the phase tensor, and the share of frequencies that agree |
| threshold agreement | share of the tried thresholds that give the class (1: stable) |
| band lines | period range, WAL class, phase tensor class; strike, twist and shear (error) where the class has them |

## Output files

All in `OUT_DIR`.

| file | content |
|---|---|
| `mt_dimensionality.txt` | settings and summary tables |
| `mt_dimensionality.csv` | one row per site and band with data |
| `mt_dimensionality.npz` | the summary as arrays: (sites,), (bands,), (sites, bands) |
| `sites/<site>_dim.npz` | all stored parameters of one site, per frequency and per band |
| `mt_dimensionality_DIM_<th>.dat` | WALDIM table, one line per site and frequency (WAL only) |
| `mt_dimensionality_BANDCLASS_<th>.dat` | WALDIM table, one line per site and band (WAL only; needs bands) |

`<th>` is `THRESHOLD` with two decimals. The keys of both kinds of `.npz` file are
listed in `dimens_readme.md` ("One npz file per site", "Survey summary").

### Summary text file

1. Settings of the run.
2. Number of sites per band and class: WAL, then phase tensor.
3. Number of frequencies per site and class: WAL, then phase tensor (`n` used,
   `out` removed, `conv` and `agree` as in the screen report).
4. One line per site and band:

| column | meaning |
|---|---|
| n | frequencies in the band |
| class, PT | WAL band class, phase tensor class |
| strike, err | averaged strike and its error [deg] |
| twist, shear | averaged angles [deg] |
| PTazim | mean azimuth of the major phase tensor axis [deg] |
| \|beta\|, ellip | median skew angle [deg] and ellipticity |
| swift, bahr | median skews |
| thr | threshold agreement (only with `STABILITY_THRESHOLDS`) |
| boot | bootstrap agreement (only with `N_BOOTSTRAP`) |

"-" means undefined. Short class names: `undet` undetermined, `3/2tw` 3D/2Dtw,
`3/2D` 3D/2D, `3/2dg` 3D/2Ddiag, `3/12D` 3D/1D2D, `ANIS1`, `ANIS2`, `undef`
undefined. Strikes and azimuths include the rotation angle of the data and carry
the 90 degree ambiguity.

### CSV file

Columns: `station`, `lat_deg`, `lon_deg`, `band`, `period_min`, `period_max`,
`class_name`, `pt_class_name`, `n_periods`, `n_classified`, `class_id`,
`pt_class`, `strike_deg`, `strike_err_deg`, `twist_deg`, `twist_err_deg`,
`shear_deg`, `shear_err_deg`, `pt_azimuth_deg`, `pt_azimuth_err_deg`,
`pt_abs_beta_deg`, `pt_ellipticity_median`, `swift_skew`, `bahr_skew`, then
`threshold_agreement` and `bootstrap_agreement` if computed. An empty field is
undefined.

### Loading the arrays

Run these lines in `$PY4MTX_DATA/dimensionality/`; replace `SITE` by a station
name.

```python
import numpy as np

s = np.load("mt_dimensionality/mt_dimensionality.npz")   # no pickle needed
k = 5                                                    # a band
print(s["band_label"][k])
for name, wal, pt, beta in zip(s["station"], s["class_id"][:, k],
                               s["pt_class"][:, k], s["pt_abs_beta_deg"][:, k]):
    print(name, wal, pt, round(float(beta), 1))   # -1 / nan: no data

d = np.load("mt_dimensionality/sites/SITE_dim.npz")
print(d["period"], d["class_name"], d["pt_class_name"], d["bahr_skew"])
```

### Maps

`femtic_plot_sitemap.py` maps the same indicators (measures `dim`, `ptdim`,
`ptskew`, `ellipt`, `swift`, `bahr`); see its readme. `mt_kmz_waldim.py` reads
the two WALDIM tables.

## Class codes

| code | WAL class | meaning |
|---|---|---|
| -1 | no data | empty band |
| 0 | undetermined | no class could be assigned |
| 1 | 1D | |
| 2 | 2D | strike defined |
| 3 | 3D/2Dtw | 3D/2D, twist only (counts as 4 in bands) |
| 4 | 3D/2D | galvanic distortion over a 2D structure |
| 5 | 3D | |
| 6 | 3D/2Ddiag | diagonally distorted case |
| 7 | 3D/1D2D | galvanic distortion, strike not recoverable |
| 8 | ANIS-HINT1 | anisotropy hint (over homogeneous medium) |
| 9 | ANIS-HINT2 | anisotropy hint (over 2D medium) |

| code | phase tensor class | rule (defaults) |
|---|---|---|
| -1 | no data | empty band |
| 0 | undefined | no phase tensor |
| 1 | 1D | \|beta\| <= 3 deg and ellipticity <= 0.1 |
| 2 | 2D | \|beta\| <= 3 deg and ellipticity > 0.1 |
| 3 | 3D | \|beta\| > 3 deg |

The threshold applies to \|beta\|, not to 2 \|beta\|.

## Reading the results

- Check "data read as ..." on the screen (columns `conv`, `agree`). A low share
  means that the phase tensor values of that site need a look; `FT_CONVENTION`
  forces a convention.
- WAL and phase tensor classes answer different questions and often differ: the
  WAL class uses the errors and separates galvanic distortion; the phase tensor
  class is blind to distortion and ignores the errors.
- WAL classes depend on the errors. State the errors (data, floor or percentage)
  with any result.
- Many "undetermined": see the reasons, and `thr` in the summary.
- The bootstrap agreement is a flag of reliability, not a probability
  (`dimens_readme.md`, "Why undetermined, and how stable is a class").

## Messages

| message (part) | cause | remedy |
|---|---|---|
| `KeyError: 'PY4MTX_DATA'` (or `ROOT`) | variable not set | export it |
| `No module named 'dimens'` (or `util`, `h5py`, ...) | module or package missing | see "Requirements" |
| `no file matches` | pattern in `INPUT` finds nothing | check the path |
| `no EDI files in` | wrong or empty directory | check `INPUT` |
| `no Z_err; set error_percent or missing_error_percent` | data without errors | set `MISSING_ERROR_PERCENT` |
| `duplicate site name` | two inputs with one name | remove one input |
| `site names do not give unique file names` | names differ only in case, blanks or separators | rename the sites |
| `PERIOD_RANGE is narrower than one band` | range too small | widen it or raise `BANDS_PER_DECADE` |
| `no sites in the input` | empty input | check `INPUT` |

Other messages of the module, and its warnings: `dimens_readme.md`, "Messages".

## Test run

The four files of `py4mt/data/edi/orig`, default settings (2026-11-28), number of
frequencies:

| site | used | WAL: undetermined, 1D, 2D, other | phase tensor: 1D, 2D, 3D |
|---|---|---|---|
| BONN_22_synth_proc | 26 | 6, 4, 0, 16 | 17, 9, 0 |
| DELTA_20 | 14 | 4, 4, 2, 4 | 14, 0, 0 |
| Metronix | 26 (2 removed) | 18, 0, 0, 8 | 0, 13, 13 |
| Phoenix2025 | 104 | 6, 17, 5, 76 | 34, 23, 47 |

Phoenix2025.edi has no impedance errors that `load_edi` returns here; 5 percent
were assumed. The runs used a stand-in for `util.py`, because `h5py` and
`pyproj` were not installed where this was written.

## References

- Marti, A., Queralt, P., Ledo, J. (2009). WALDIM: A code for the
  dimensionality analysis of magnetotelluric data using the rotational
  invariants of the magnetotelluric tensor. Computers & Geosciences, 35(12),
  2295-2303. doi:10.1016/j.cageo.2009.03.004
- Caldwell, T. G., Bibby, H. M., Brown, C. (2004). The magnetotelluric phase
  tensor. Geophysical Journal International, 158(2), 457-469.
  doi:10.1111/j.1365-246X.2004.02281.x
- Booker, J. R. (2014). The magnetotelluric phase tensor: a critical review.
  Surveys in Geophysics, 35(1), 7-40. doi:10.1007/s10712-013-9234-2
- Bahr, K. (1991). Geological noise in magnetotelluric data: a classification of
  distortion types. Physics of the Earth and Planetary Interiors, 66(1-2),
  24-38. doi:10.1016/0031-9201(91)90101-M
- No DOI confirmed: Swift, C. M. (1967). A magnetotelluric investigation of an
  electrical conductivity anomaly in the southwestern United States. PhD thesis,
  Massachusetts Institute of Technology. Bahr, K. (1988). Interpretation of the
  magnetotelluric impedance tensor: regional induction and local telluric
  distortion. Journal of Geophysics, 62, 119-127.

## Changelog

- 2026-11-28  Claude Sonnet 5.5 (Anthropic): initial version; replaces
  mt_dimens.py. AI-generated; review before production use.
- 2026-11-28  Claude Sonnet 5.5 (Anthropic): (b) readme completed (requirements,
  examples, screen report, columns, class codes, messages, references); script
  text on percentage errors corrected. AI-generated; review before production
  use.
