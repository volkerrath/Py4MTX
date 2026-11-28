# dimens.py - dimensionality of magnetotelluric impedances

WAL rotational invariants (port of WALDIM 1.2), phase tensor, Swift and Bahr
skews; one result per site, per frequency and per period band.

Author of this module: Claude (Anthropic), model Claude Sonnet 5.5, generated
2026-11-28. The code is AI-generated and should be reviewed before production
use. Original Fortran code (WAL part): A. Marti, P. Queralt and J. Ledo.

Script on top of it: `mt_dimensionality.py` (see
`mt_dimensionality_readme.md`). Maps: `femtic_plot_sitemap.py`.

Until 2026-11-28 the module was called `waldim.py` (script `mt_waldim.py`,
then `mt_dimens.py`): delete these older files.

Contents: [Files](#files),
[Requirements and installation](#requirements-and-installation),
[Quick start](#quick-start), [Functions](#functions),
[Parameters](#parameters), [Input](#input), [Output](#output),
[Indicators](#indicators), [Bands](#bands),
[Time convention](#time-convention),
[Why undetermined, and how stable is a class](#why-undetermined-and-how-stable-is-a-class),
[One npz file per site](#one-npz-file-per-site),
[Survey summary](#survey-summary), [WALDIM tables](#waldim-tables),
[Method of the WAL classification](#method-of-the-wal-classification-as-in-the-source),
[`waldim_compat`](#waldim_compat),
[Source errors corrected in both modes](#source-errors-corrected-in-both-modes),
[Other differences from the Fortran](#other-differences-from-the-fortran),
[Messages](#messages), [Limitations](#limitations),
[Observations on py4mt](#observations-on-py4mt-commit-21807c5-nothing-was-changed-there),
[Verification](#verification-2026-11-28), [Not ported](#not-ported),
[References](#references), [Changelog](#changelog).

## Files

| file in the zip | content |
|---|---|
| py4mt/modules/dimens.py | module |
| py4mt/modules/dimens_readme.md | this file |
| py4mt/modules/dimens.bib | references |
| py4mt/scripts/general/mt_dimensionality.py | script: EDI files or collections to summary and npz |
| py4mt/scripts/general/mt_dimensionality_readme.md | notes on the script |
| py4mt/scripts/femtic/femtic_plot_sitemap.py | site map script, with dimensionality maps |
| py4mt/scripts/femtic/femtic_plot_sitemap_readme.md | notes on that script |
| tests/test_dimens.py | pytest suite of the module |
| tests/check_against_fortran.py | cross-check with the Fortran source |

## Requirements and installation

- Python 3 with NumPy. Tests run with Python 3.13 and NumPy 2.5; the files also
  compile with Python 3.11.
- To read EDI files: the py4mt module `data_proc` (`load_edi`) and what it
  imports, or your own function passed as `loader`. Data dicts and `.npz` files
  need NumPy only.
- Install: copy `dimens.py`, `dimens_readme.md` and `dimens.bib` to
  `<py4mtx>/py4mt/modules/` and have that folder on `sys.path`, for example

```
export PYTHONPATH=$PY4MTX_ROOT/py4mt/modules:$PYTHONPATH
```

## Quick start

```python
from dimens import (classify_dimensionality, save_site_npz, write_results,
                    write_summary)

# EDI directory, EDI file, .npz collection or data dicts; one band per decade
res = classify_dimensionality("edi/", bands=True, missing_error_percent=5.0)

write_summary(res, "out/")                # mt_dimensionality.txt/.csv/.npz
save_site_npz(res, "out/sites/")          # <site>_dim.npz
write_results(res, "out/mt_dimensionality")   # WALDIM tables

for name, site in res.items():
    print(name, site["counts"], site["pt_counts"])
    for b in site["bands"]:
        print(f'{b["period_min"]:10.3g} {b["period_max"]:10.3g} '
              f'{b["n_periods"]:4d}  {b["class_name"]:12s} '
              f'{b["pt_abs_beta_deg"]:7.2f} {b["bahr_skew"]:6.3f}')
```

Data dicts and the output of `data_proc.make_collection` work the same way
(`missing_error_percent` is used only where a site has no `Z_err`):

```python
res = classify_dimensionality(load_edi(path), missing_error_percent=5.0)
res = classify_dimensionality(make_collection("edi/"),
                              missing_error_percent=5.0)
res = classify_dimensionality(data, error_percent=5.0)   # ignore Z_err
```

## Functions

| function | purpose | writes files |
|---|---|---|
| `classify_dimensionality(sites, ...)` | all indicators, per site | no |
| `summarise_band(result, members, ...)` | summary of one site over chosen rows | no |
| `site_parameters(result)` | one site as a flat dict | no |
| `save_site_npz(results, directory, ...)` | one `.npz` per site | yes |
| `survey_summary(results)` | band summaries as (site, band) arrays | no |
| `write_summary(results, directory, ...)` | summary as `.txt`, `.csv`, `.npz` | yes |
| `write_results(results, root)` | WALDIM `_DIM_` and `_BANDCLASS_` tables | yes |

`results` is the dict returned by `classify_dimensionality`; `result` is one of
its values. Options after `*` must be given by name. Existing files are
overwritten.

### classify_dimensionality

```
classify_dimensionality(sites, *, threshold=0.15, q_threshold=0.10,
    error_percent=None, missing_error_percent=None, bands=False,
    bands_per_decade=1.0, band_edges=None, band_domain="period",
    n_realizations=100, seed=0, waldim_compat=False,
    stability_thresholds=(0.10, 0.12, 0.18, 0.20), n_bootstrap=0,
    pt_beta_threshold=3.0, pt_ellipticity_threshold=0.1, ft_convention="auto",
    use_filename_as_station=True, edi_options=None, loader=None)
```

Returns `{site name: result}`. See "Input", "Parameters" and "Output".

### summarise_band

```
summarise_band(result, members, *, waldim_compat=None)
```

`members` are row numbers (positions in `result["period"]`); the list may be
empty. Returns the dict of one band (see "Bands") without the band edges.
`waldim_compat=None` takes the setting of the site; with True the rows should be
consecutive.

```python
import numpy as np
from dimens import classify_dimensionality, summarise_band

res = classify_dimensionality("edi/DELTA_20.edi")
site = res["DELTA_20"]
rows = np.flatnonzero((site["period"] > 0.001) & (site["period"] < 0.1))
band = summarise_band(site, rows)
print(band["class_name"], band["n_periods"], band["pt_abs_beta_deg"])
```

### site_parameters, save_site_npz

```
site_parameters(result)
save_site_npz(results, directory, *, suffix="_dim")
```

`site_parameters` returns the dict that `save_site_npz` stores (see "One npz
file per site"). `save_site_npz` creates the directory and returns the paths. It
raises `ValueError`, before writing, if two sites get the same file name (case is
ignored).

### survey_summary, write_summary

```
survey_summary(results)
write_summary(results, directory, *, name="mt_dimensionality", header=())
```

All sites need the same bands, or none has bands. `write_summary` creates the
directory and returns the three paths; `header` is a list of extra lines for the
top of the text file. See "Survey summary".

### write_results

```
write_results(results, root)
```

`root` is the output path without ending; its directory must exist. All sites
need the same `threshold`. Returns one path, or two if the results hold bands.

### Constants

| name | content |
|---|---|
| `CLASS_NAMES` | WAL class code (0 to 9) to name |
| `PT_CLASS_NAMES` | phase tensor class code (0 to 3) to name |
| `REASON_NAMES` | reason code (0 to 3) to name |
| `NO_DATA`, `NO_DATA_NAME` | -1, "no data" (empty band) |
| `INVARIANT_NAMES` | columns of `invariants` |
| `STABILITY_THRESHOLDS`, `PT_BETA_THRESHOLD`, `PT_ELLIPTICITY_THRESHOLD` | defaults |
| `FT_CONVENTIONS` | "auto", "e+iwt", "e-iwt" |

## Parameters

All keywords of `classify_dimensionality`. Last column: counterpart in the
`param.cfg` file of WALDIM.

| keyword | default | meaning | WALDIM |
|---|---|---|---|
| `threshold` | 0.15 | limit of the invariants I3 to I7, 0 to 1 | threshold |
| `q_threshold` | 0.10 | limit of Q, 0 to 1 | Q threshold |
| `error_percent` | None | ignore `Z_err`; standard error = this percentage of max(\|Re\|, \|Im\|) of each element, all sites | error percentage |
| `missing_error_percent` | None | the same, only for sites without errors; None: such a site raises an error | - |
| `bands` | False | also summarise in period bands | band option |
| `bands_per_decade` | 1.0 | bands per decade; edges at 10**(k/n), common to all sites | bands per decade |
| `band_edges` | None | own edges (switches banding on, overrides `bands_per_decade`) | min/max period |
| `band_domain` | "period" | `band_edges` in s ("period") or Hz ("frequency") | - |
| `n_realizations` | 100 | noisy tensors for the bias test and the errors, at least 2 | fixed to 100 |
| `seed` | 0 | random seed, used afresh for each site; None: not reproducible | from the clock |
| `waldim_compat` | False | True: rules of WALDIM 1.2 (see "waldim_compat") | - |
| `stability_thresholds` | (0.10, 0.12, 0.18, 0.20) | also classify with these `threshold` values; None or empty: not done | - |
| `n_bootstrap` | 0 | repetitions of the classification with added noise; 0: none | - |
| `pt_beta_threshold` | 3.0 | deg; phase tensor class 3D if \|beta\| exceeds it | - |
| `pt_ellipticity_threshold` | 0.1 | otherwise 2D if the ellipticity exceeds it | - |
| `ft_convention` | "auto" | time convention of Z for the phase tensor | - |
| `use_filename_as_station` | True | EDI files: site name from the file name, else DATAID | - |
| `edi_options` | None | further keywords of the EDI reader | EDI reading |
| `loader` | None | own reader `(path, **edi_options) -> dict` | - |

## Input

- A py4mt data dict: `freq` (n,), `Z` (n,2,2) complex in mV/km/nT, and
  optionally `Z_err`, `err_kind` ("var" default, or "std"), `rot` (deg),
  `station`, `lat_deg`, `lon_deg`. A mapping with a key `Z` is one such dict.
- A sequence of dicts or paths, or a mapping {site name: dict or path}.
- EDI files, or a directory holding *.edi files, read with `data_proc.load_edi`:
  `err_kind="var"`, `drop_invalid_periods=False`, `freq_order="inc"` are set;
  `edi_options` adds to or overrides these, for example
  `{"manufacturer": "phoenix"}`. Invalid impedance rows are removed by dimens.
- A `.npz` collection (`data_proc.save_list_of_dicts_npz`, key `records`) or a
  station file (`data_proc.save_npz`). **These hold pickled objects: read only
  files that you trust.**
- Names: EDI files after the file (`use_filename_as_station=True`) or after
  DATAID (False). Data dicts, also the records of a `.npz` file, after their
  `station` entry; a record without one after the file (`<file>_001`, ... if
  there are several). A site name must not occur twice.
- Files without impedance errors (for example Phoenix spectra files) need
  `missing_error_percent`.
- A warning is issued for a component that is zero at all frequencies (a missing
  block), missing rotation angles (0 is assumed), unexpected `Z_units`, and sites
  without a valid frequency.

### Error convention

As in WALDIM, the variance of an element applies to its real part and to its
imaginary part. The bootstrap functions of data_proc split sigma by sqrt(2)
between the two parts. The choice changes the classes: with halved variances,
7 of 26 (Metronix.edi), 4 of 26 (BONN_22_synth_proc.edi) and 0 of 14
(DELTA_20.edi) classes differ. To treat Z_err as the total variance:

```python
def load_total_variance(path, **kwargs):
    data = load_edi(path, **kwargs)
    if data.get("Z_err") is not None:
        data["Z_err"] = 0.5 * data["Z_err"]
    return data

res = classify_dimensionality(
    "edi/", loader=load_total_variance, missing_error_percent=5.0,
    edi_options={"err_kind": "var", "drop_invalid_periods": False,
                 "freq_order": "inc"})   # not set for a custom loader
```

## Output

`{site name: result}`. Per-frequency arrays are sorted by ascending period.

| key | content |
|---|---|
| station, lat_deg, lon_deg | site |
| threshold, q_threshold, waldim_compat | settings |
| freq, period, index | (m,) valid data; index is the input row |
| class_id, class_name | (m,) WAL class |
| counts | number of frequencies per class name (all ten) |
| reason_id, reason_name, reason_counts, flags | why undetermined |
| invariants | (m,8) \|I1\| ... \|I7\|, Q |
| rho_a, rho_a_err | (m,) ohm m |
| phase_deg, phase_err_deg | (m,) in [0, 90] |
| strike_deg, strike_err_deg | (m,) strike of the class, else NaN |
| twist_deg, shear_deg (+ _err) | (m,) classes 3 and 4, else NaN |
| strikes_deg, strikes_err_deg | (m,3) from Re, from Im, third estimate |
| distortion | dict: angles for every frequency, whatever its class |
| analytic_errors | first-order errors of rho_a, phase, three strikes |
| pt..., swift_skew..., bahr_skew... | see "Indicators" |
| pt_class, pt_class_name, pt_counts | phase tensor class |
| pt_ft_convention, pt_ft_agreement | time convention used |
| pt_beta_threshold_deg, pt_ellipticity_threshold | settings |
| threshold_values, class_by_threshold, threshold_agreement | threshold test |
| n_bootstrap, bootstrap_class_id, bootstrap_class_prob, bootstrap_agreement | only with the bootstrap |
| n_dropped, error_source | rows removed; "Z_err" or "percent" |
| bands | list of dicts (only with banding) |

Units and conventions: rho_a = 0.2 |Z|^2 / f. Angles in degrees. Strikes include
`rot` and carry the 90 degree ambiguity. The sign of the shear angle belongs to
the strike reported with it. All errors are 1 sigma; the errors without the
prefix `analytic_errors` are Monte Carlo standard deviations.

## Indicators

### WAL class (per frequency and band)

| code | name | meaning | parameters |
|---|---|---|---|
| 0 | undetermined | no class could be assigned | - |
| 1 | 1D | 1D | - |
| 2 | 2D | 2D | strike (a) |
| 3 | 3D/2Dtw | 3D/2D, twist only | strike (b), twist, shear |
| 4 | 3D/2D | 3D/2D, general galvanic distortion | strike (b), twist, shear |
| 5 | 3D | 3D | - |
| 6 | 3D/2Ddiag | diagonally distorted case | - |
| 7 | 3D/1D2D | galvanic distortion, strike not recoverable | - |
| 8 | ANIS-HINT1 | anisotropy hint (source: "over homogeneous") | strike (a) |
| 9 | ANIS-HINT2 | anisotropy hint (source: "over 2D") | strike (b) |

(a) mean of the strikes from the real and the imaginary part; (b) strike from
the distortion decomposition. Apparent resistivity and phase (from I1 and I2)
are given for all classes. In band averages, class 3 counts as class 4.

### Phase tensor

Phi = X^-1 Y with Z = X + iY (Caldwell et al. 2004; see also Booker 2014).

| key | definition |
|---|---|
| `pt`, `pt_err` | tensor and standard deviations, (m,2,2), data coordinates |
| `pt_phimax_deg`, `pt_phimin_deg` | principal phases; Phi_min keeps its sign (negative if det(Phi) < 0) |
| `pt_alpha_deg` | atan2(P12 + P21, P11 - P22) / 2 + rot |
| `pt_beta_deg` | atan2(P12 - P21, P11 + P22) / 2 (skew angle) |
| `pt_azimuth_deg` | major axis, alpha - beta (including rot), [0, 180) |
| `pt_ellipticity` | (max - min) / (max + min) of the principal values |
| `pt_class` | 0 undefined, 1 1D, 2 2D, 3 3D |

Class: 3D if |beta| > `pt_beta_threshold` (3 deg); else 2D if the ellipticity
> `pt_ellipticity_threshold` (0.1); else 1D. **The threshold applies to |beta|,
not to 2 |beta|** (the defaults are those of the constants of
`mt_calc_ptdim.py` in py4mt). A band gets the most frequent class (the higher
one in a tie). These thresholds ignore the errors: noisy data look 3D.

### Skews

With S1 = Zxx + Zyy, S2 = Zxy + Zyx, D1 = Zxx - Zyy, D2 = Zxy - Zyx and
[a, b] = Re a Im b - Im a Re b:

- `swift_skew` = |S1| / |D2| (Swift 1967)
- `bahr_skew` = sqrt(|[D1, S2] - [S1, D2]|) / |D2| (Bahr 1988, 1991)

They are reported without a class. A tensor with D2 near 0 (for example a
purely diagonal one) has very large values.

### Errors of phase tensor and skews

`..._err` of the phase tensor and skews are standard deviations over
`n_realizations` noisy tensors (angle differences modulo 180 deg). Near a
singular tensor they are large and unreliable.

## Bands

Each entry of `result["bands"]` holds `period_min`, `period_max`, `freq_min`,
`freq_max`, `data_period_min`, `data_period_max`, `n_periods`, `n_classified`,
`n_averaged`, `class_id`, `class_name`, `pt_class`, `counts`, `pt_counts`,
`reason_counts` and the values described below.

- Edges lie at 10**(k / bands_per_decade), cover all data and are common to all
  sites. Each band is (period_min, period_max].
- `summarise_band(result, rows)` gives the same summary for any set of rows.
- `counts` holds the unmerged class counts of the band.
- `class_id` is the WALDIM band class: 3 counts as 4; undetermined values are
  ignored; the most frequent class wins and ties go to the lowest code; if 3D
  wins, another class with at least half of the classified values replaces it.
  0 means nothing classified, -1 means no data.
- `strike_deg`, `twist_deg`, `shear_deg` are averaged over the frequencies of
  the band class (classes 2, 4, 8, 9; twist and shear for class 4). By default strikes are
  averaged with 90 degree periodicity and shear signs are matched to the mean
  strike.
- Phase tensor means over **all** frequencies of the band: `pt_phimax_deg`,
  `pt_phimin_deg`, `pt_beta_deg`, `pt_azimuth_deg` (both modulo 180),
  `pt_ellipticity`.
- Medians over all frequencies, without errors: `pt_abs_beta_deg`,
  `pt_ellipticity_median`, `swift_skew`, `bahr_skew`.
- Also `pt_class`, `pt_counts`, `reason_counts`; with the threshold test
  `class_by_threshold` and `threshold_agreement`; with the bootstrap
  `bootstrap_class_prob` and `bootstrap_agreement`.

**Errors of the averages.** `..._err_prop` is propagated from the data errors,
sqrt(sum err^2) / N; `..._err_scatter` is the standard error from the scatter of
the values in the band (NaN for fewer than two); `..._err` is the larger of the
two. On the sample files the scatter term was the larger one in 7 to 10 of 22
band averages of the phase tensor angles and ellipticity: propagated errors
alone understate the uncertainty there. With `waldim_compat=True` the errors of
strike, twist and shear are the propagated ones.

## Time convention

The WAL classes, strikes, twist, shear, phases and skews are the same for
exp(+iwt) and exp(-iwt) data. The **phase tensor is not**: it needs phases of Zxy
in the first quadrant (exp(+iwt)). For exp(-iwt) data the plain formula gives
-Phi and |beta| near 90 deg. `load_edi` of py4mt stores exp(-iwt) as its
standard, but the sample files differ: Phoenix2025.edi arrives as exp(-iwt),
BONN_22_synth_proc.edi, DELTA_20.edi and Metronix.edi as exp(+iwt) (read with
`manufacturer="metronix"`).

`ft_convention="auto"` (default) takes for each site the sign of the trace of
Phi at most frequencies and conjugates exp(-iwt) data for the phase tensor.
The result is reported in `pt_ft_convention` and `pt_ft_agreement` (share of
frequencies that agree; 0.94 for Phoenix2025, 1.0 for the other three). A low
agreement means data near the quadrant limits, errors or mixed conventions: set
`ft_convention` yourself then. The Monte Carlo errors of the WAL part differ
marginally between the two conventions (other random signs).

`data_proc.compute_pt` applies X^-1 Y to the stored Z, so for exp(-iwt) data
it returns -Phi (see "Observations on py4mt").

## Why undetermined, and how stable is a class

**Reason codes** (`reason_id`, `reason_name`, `reason_counts`):

| code | name | meaning |
|---|---|---|
| 0 | classified | - |
| 1 | bias test | true and Monte Carlo mean of one of I3 ... I6 differ by more than a standard deviation |
| 2 | error bars | one of I3 ... I6 cannot be decided against the threshold within its error |
| 3 | no rule | the combination of flags has no class |

`flags` (m,5) shows I3 ... I7 below (0) or above (1) the threshold, -1 undecided.

**Thresholds** (`stability_thresholds`, default 0.10, 0.12, 0.18, 0.20 plus the
one in use): `class_by_threshold` (m,k) and `threshold_agreement`, the share of
these thresholds that give the class of the run. `q_threshold` stays fixed.

On the 170 frequencies of the four sample files (default settings, 5 percent
errors for Phoenix2025): 34 undetermined (22 no rule, 10 error bars, 2 bias
test); the class differs from that at 0.15 for 40 frequencies at 0.10, 21 at
0.12, 27 at 0.18 and 37 at 0.20.

**Bootstrap** (`n_bootstrap`, default 0): the classification is repeated with
noise of the size of the errors added to Z. `bootstrap_agreement` is the share
of repetitions with the class of the data.

Test with synthetic data (600 tensors each of 1D, 2D, 3D/2D and 3D, 200
repetitions, seed 5; percent of all tensors with the correct class):

| noise (% of largest element) | single classification | most frequent bootstrap class |
|---|---|---|
| 0.5 | 98.4 | 98.5 |
| 1 | 92.6 | 91.8 |
| 2 | 77.0 | 68.6 |
| 5 | 35.3 | 23.6 |

(Seed 11: 98.3 / 98.4, 91.6 / 90.9, 77.0 / 68.1, 35.7 / 23.0.) At 2 percent
noise 2D falls from 57.5 to 41.3 and 3D/2D from 58.3 to 40.2 percent.

- The bootstrap majority is **not** a better estimate: the noise is added a
  second time, which favours complex classes. `class_id` is never replaced by it.
- The agreement is useful as a flag. Of the classified tensors, 86, 58, 36 and
  14 percent (noise 0.5, 1, 2, 5) reach an agreement of 0.95 or more, and of
  these 100, 99.9, 99.1 and 98.0 percent are correct (seed 11 at 5 percent: 95).
  As a ranking of right against wrong classes it did better than
  `threshold_agreement` (area under the ROC curve 0.98, 0.88, 0.72, 0.61
  against 0.85, 0.66, 0.58, 0.50).
- It is not a calibrated probability: the labels of simple structures stay
  nearly always right even with low agreement. The main use is for "3D": at 1 to
  2 percent noise such a label is right about 98 percent of the time with an
  agreement of 0.95 or more, and about 16 to 24 percent otherwise.
- Cost: about `n_bootstrap` times that of one classification (some 5 s for 2400
  tensors and 200 repetitions here).

## One npz file per site

`save_site_npz(results, directory, suffix="_dim")` writes
`<directory>/<site><suffix>.npz`, creating the directory and overwriting
existing files. Blanks, commas and path separators in site names become "_".
Two names that differ only in case are refused before anything is written.

Each file holds the dict returned by `site_parameters(result)`; a strike, twist
or shear angle that does not apply to the class of a frequency or band is NaN,
as is its error.

| keys | shape |
|---|---|
| station, lat_deg, lon_deg, error_source, n_dropped, threshold, q_threshold, waldim_compat | scalar |
| pt_ft_convention, pt_ft_agreement, pt_beta_threshold_deg, pt_ellipticity_threshold; n_bootstrap | scalar |
| freq, period, index, class_id, class_name, reason_id, reason_name | (m,) |
| flags; invariants, invariant_names | (m,5); (m,8), (8,) |
| rho_a, rho_a_err, phase_deg, phase_err_deg, strike_deg, twist_deg, shear_deg (+ _err) | (m,) |
| pt, pt_err | (m,2,2) |
| pt_phimax_deg, pt_phimin_deg, pt_alpha_deg, pt_beta_deg, pt_azimuth_deg, pt_ellipticity, swift_skew, bahr_skew (+ _err) | (m,) |
| pt_class, pt_class_name | (m,) |
| class_by_threshold, threshold_values, threshold_agreement | (m,k), (k,), (m,) |
| bootstrap_class_prob, bootstrap_agreement | (m,10), (m,) |
| class_codes/names/counts; reason_codes/names/counts; pt_class_codes/names/counts | (10,); (4,); (4,) |
| band_<name> for the band entries, including ..._err, ..._err_prop, ..._err_scatter | (nb,) |
| band_class_counts, band_pt_class_counts, band_reason_counts | (nb,10), (nb,4), (nb,4) |
| band_class_by_threshold, band_threshold_agreement | (nb,k), (nb,) |
| band_bootstrap_class_prob, band_bootstrap_agreement | (nb,10), (nb,) |

Band, threshold and bootstrap keys exist only if these were computed. All bands
are stored, including empty ones (class -1). The three raw strikes, the
distortion angles of every frequency, the analytic errors and the single
bootstrap draws are not stored; they remain in the returned results.

```python
d = np.load("out/sites/SITE_dim.npz")          # no pickle needed
three_d = (d["class_id"] == 5) & (d["pt_class"] == 3)
print(d["period"][three_d], d["pt_beta_deg"][three_d])

d = data_proc.load_npz("out/sites/SITE_dim.npz")  # scalars as Python values
```

## Survey summary

`write_summary(results, directory, name="mt_dimensionality", header=())` writes

- `<name>.txt`: sites per class and band (WAL and phase tensor), frequencies per
  site and class (with `n` used, `out` removed, time convention), one line per
  site and band;
- `<name>.csv`: one row per site and band with data (empty field: undefined);
- `<name>.npz`: the arrays of `survey_summary(results)`: per site (ns,), per
  band (nb,), per site and band (ns,nb), and counts of sites per band and class
  (nb,10) and (nb,4). Loads without pickle.

Arrays in `<name>.npz`:

| keys | shape |
|---|---|
| station, lat_deg, lon_deg, n_freq, n_dropped, error_source, pt_ft_convention, pt_ft_agreement | (ns,) |
| site_class_counts, site_pt_class_counts | (ns,10), (ns,4) |
| band_label, band_period_min, band_period_max | (nb,) |
| n_periods, n_classified, class_id (-1: no data), pt_class | (ns,nb) |
| strike_deg, twist_deg, shear_deg, pt_azimuth_deg (+ _err_deg) | (ns,nb) |
| pt_abs_beta_deg, pt_ellipticity_median, swift_skew, bahr_skew | (ns,nb) |
| threshold_agreement, bootstrap_agreement (if computed) | (ns,nb) |
| class_codes, class_names; pt_class_codes, pt_class_names | (10,); (4,) |
| class_site_counts, pt_class_site_counts (sites per band and class) | (nb,10), (nb,4) |

All sites need the same bands. Without bands, each site is summarised over all
its frequencies.

## WALDIM tables

`write_results(results, root)` writes the two WALDIM tables that
`mt_kmz_waldim.py` reads (WAL results only):

- `root_DIM_<th>.dat`: Site, Longitude, Latitude, F, Per, DIM, then value and
  error of st3, st4, st5 (strikes), st6 (twist + shear) and st7 (twist -
  shear).
- `root_BANDCLASS_<th>.dat`: Site, Longitude, Latitude, BAND, Tmin, Tmax,
  nper, DIM, strike, errstr, twi, errtwi, shear, errsh, cont, scale,
  strikecomp. Only bands holding data. Tmin and Tmax are those of the data.

Undefined values are written as "nan" (WALDIM writes 0 in the band table).

## Method of the WAL classification (as in the source)

1. Exact zeros are replaced by tiny random numbers; Zyx is perturbed if it
   equals -Zxy; null frequencies are removed.
2. Invariants I1 to I7 and Q, with propagated errors for I3 to I6.
3. Monte Carlo test (100 realizations): a frequency is undetermined if the
   true and the mean value of any of I3 to I6 differ by more than one
   standard deviation (unless both are below the threshold).
4. I3 to I6 are flagged using value and error bar; I7 using Q and threshold.
5. Rules on the flags, on |xi4| and |eta4| (class 6) and on the three
   strike estimates (classes 2, 8, 9) give the class.

## `waldim_compat`

The default (False) changes the following rules. True restores them.

### Strike wrap-around (classes 2 and 8)

The source accepts Re and Im strikes near 0 and 90 degrees as aligned, but
then takes their arithmetic mean (89 and 1 give 45). The third strike is
compared with that mean, so regular 2D data with a strike near 0 or 90
degrees are often labelled ANIS-HINT1. Synthetic 2D tensor, 2000 noisy
samples per row, noise 1 percent of the largest element:

| strike (deg) | compat: 2D / HINT1 (%) | default: 2D / HINT1 (%) |
|---|---|---|
| 0 | 44.8 / 54.5 | 99.3 / 0.0 |
| 3 | 87.4 / 11.9 | 99.3 / 0.0 |
| 30 | 99.8 / 0.0 | 99.8 / 0.0 |
| 89 | 20.3 / 79.2 | 99.7 / 0.0 |

The default compares and averages all strikes modulo 90 degrees.

### Percentage errors

The source takes the percentage of the signed max(Re, Im) of each element. If
both parts are negative this is the smaller one in magnitude. The default uses
max(|Re|, |Im|).

### Shear sign

The source computes the shear angle on the unfolded third strike but reports
the folded one, so its sign may belong to the strike 90 degrees away. The
default computes it on the reported strike.

### Band errors

The source gives the propagated error only (see "Bands").

## Source errors corrected in both modes

| WALDIM1.2.f lines | error | effect |
|---|---|---|
| 751 | resistivity error uses a leftover index, F(i) | wrong rho_a error |
| 858 | rotation converted with 180/pi | wrong Monte Carlo twist, shear |
| 859-878 | factors of 2 missing, signs reversed | wrong Monte Carlo twist, shear |
| 1400 | band strike average reads site index 2 | wrong band mirroring |
| 768-785 | incomplete derivative of strike 3 | small: original/exact 0.95 to 1.02 (5 to 95 percentile) |
| (table output) | class 9 strike error not set | undefined value |

## Other differences from the Fortran

- Monte Carlo errors take angle differences modulo 90 (strikes) or 180 degrees
  (distortion angles).
- All null frequencies are removed. The source skips the second of two
  consecutive null frequencies.
- Random numbers come from NumPy with a fixed default seed. Frequencies close to
  the bias-test limit can differ from a Fortran run (and between Fortran runs,
  which are seeded from the clock).
- The bias test allows a round-off of 1e-10, relevant for error-free data.
- Pi is double precision.
- Automatic bands follow the data. The source uses a minimum and maximum period
  and truncates the number of bands to an integer (mt_dimensionality.py does this
  if PERIOD_RANGE is set).
- Missing errors raise an exception unless a percentage is given.
- Added: phase tensor, skews, reason codes, threshold test, bootstrap, survey
  summary.

## Messages

**Errors** (exception, nothing is returned or written):

| message (start) | cause | remedy |
|---|---|---|
| `<site>: no Z_err; set error_percent or ...` | data without errors | set `missing_error_percent` |
| `<site>: Z must have shape` (also `Z_err`, `rot`) | wrong array shapes | check the data dict |
| `<site>: unknown err_kind` | `err_kind` not "var" or "std" | correct the data dict |
| `duplicate site name` | two inputs with one name | rename, or pass a mapping {name: input} |
| `<name>: <path> holds N stations` / `is a directory` | a mapping entry must be one site | pass the path without a name |
| `no EDI files in` | empty or wrong directory | check the path |
| `neither a collection ... nor a station file` | unknown `.npz` layout | use `save_list_of_dicts_npz` or `save_npz` files |
| `reading EDI paths needs py4mt data_proc.load_edi` | py4mt modules not found | extend `sys.path`, or pass `loader` |
| `... must lie in [0, 1]`, `... must be >= 0` and similar | invalid parameter | see "Parameters" |
| `band_edges ...` | edges not positive, or fewer than two | correct the edges |
| `some sites have bands and others not`, `the bands of the sites differ` | summary of mixed results | classify all sites in one call, or give the same `band_edges` |
| `results were computed with different thresholds` | `write_results` | classify with one `threshold` |
| `site names ... unique` | table or file names clash | rename the sites |

**Warnings** (the run goes on):

| message (part) | meaning |
|---|---|
| `no valid frequency` | all rows were removed; check impedances and errors (a component without any valid error removes them unless `missing_error_percent` is set) |
| `... is zero at all frequencies` | probably a missing block in the EDI file: do not use this site |
| `no valid rotation angle, 0 is assumed` | strikes refer to the data coordinates |
| `Z_units is ..., expected mV/km/nT` | apparent resistivity will be wrong |

## Limitations

- The WAL thresholds and the phase tensor thresholds are conventions; classes
  change with them and with the errors (see above). Indicators that disagree are
  information, not a fault: WAL uses the errors and distortion, the phase tensor
  class does not.
- The phase tensor is blind to galvanic distortion of the electric field; its 1D
  or 2D refers to the regional structure.
- No test against data of independently known dimensionality (synthetic tensors
  and sample files only).
- Sites are processed one after the other.

## Observations on py4mt (commit 21807c5; nothing was changed there)

1. `data_proc.compute_pt` forms X^-1 Y from the stored Z. For exp(-iwt) data,
   the standard of `load_edi`, the result is -Phi (a 1-D tensor gives -I and
   |beta| = 90 deg). Check the scripts that use it with your data.
2. `data_proc.rotate_data` rotates Z_err like the tensor, R E R^T. For
   variances this gives negative values: BONN_22_synth_proc.edi rotated by 30
   degrees has 26 of 104 negative, in all 26 rows (60 degrees: 73 of 104).
   dimens treats a negative variance as invalid, so such rotated data lose
   rows. For uncorrelated errors the rotated variances are
   `(R**2) @ V @ (R**2).T`. The code also rotates T_err and P_err like the
   quantities they belong to.
3. The impedances in `data/edi/proc/PHOENIX2025_*_proc.edi` are a factor 1000
   smaller than in `data/edi/orig/Phoenix2025.edi` (median |Zxy| 0.0105 and
   10.5), so rho_a is 1e6 too small. The cause was not investigated.
4. With its default `drop_invalid_periods=True`, `load_edi` keeps 20 of the
   28 rows of `orig/Metronix.edi`. Only 2 rows have an invalid impedance;
   dimens keeps the other 26.
5. `mt_kmz_waldim.py`, frequency mode, selects rows with `np.isclose` on
   log10(f) with rtol 1e-2, so one row can enter several frequency folders: 170
   table rows gave 225 points in 156 folders.

## Verification (2026-11-28)

Commands, run from the top folder of the unpacked zip:

    PYTHONPATH=py4mt/modules PY4MTX_ROOT=<repo> python -m pytest tests/test_dimens.py
    PYTHONPATH=py4mt/modules python tests/check_against_fortran.py <dir with the .f files>

`<repo>` is a py4mtx checkout (for load_edi and the sample data; the tests that
need it are skipped without it).

- Tests: 192 passed. Synthetic tensors, small EDI files written by the tests,
  the sample collections of py4mt (phase tensor compared with
  `data_proc.compute_pt`, after conjugation of exp(-iwt) data).
- Cross-check: line ranges of the original source, translated mechanically to
  Python (lines 751 and 1400 corrected first), compared with the module in compat
  mode using identical random deviates. 24000 tensors, all ten codes present, no
  class mismatch; invariants, resistivity and their errors agree to 7e-16; phase,
  distortion angles and the compared angle errors to 3e-8 and strikes to 1.3e-6
  (single-precision pi in the source); 40000 band class cases without mismatch;
  15628 band averages to 1.6e-15.
- The WAL results equal those of the previous version in both modes (6871
  entries), except the band errors of the default mode, which now include the
  scatter.
- Not done: no Fortran compiler was available, so the compiled program was never
  run and no WALDIM output was compared; the phase tensor and skews were not
  compared with another program (formulas checked against the literature and
  `compute_pt`).

## Not ported

The Fortran EDI reader (replaced by load_edi), the param.cfg file (replaced by
keyword arguments and mt_dimensionality.py), the output tables other than _DIM_
and _BANDCLASS_, and the comparison tables with Swift and Bahr parameters.

## References

- Marti, A., Queralt, P., Ledo, J. (2009). WALDIM: A code for the
  dimensionality analysis of magnetotelluric data using the rotational
  invariants of the magnetotelluric tensor. Computers & Geosciences, 35(12),
  2295-2303. doi:10.1016/j.cageo.2009.03.004
- Marti, A., Queralt, P., Ledo, J., Farquharson, C. (2010). Dimensionality
  imprint of electrical anisotropy in magnetotelluric responses. Physics of the
  Earth and Planetary Interiors, 182(3-4), 139-151.
  doi:10.1016/j.pepi.2010.07.007
- Marti, A., Queralt, P., Jones, A. G., Ledo, J. (2005). Improving Bahr's
  invariant parameters using the WAL approach. Geophysical Journal
  International, 163(1), 38-41. doi:10.1111/j.1365-246X.2005.02748.x
- Weaver, J. T., Agarwal, A. K., Lilley, F. E. M. (2000). Characterization of
  the magnetotelluric tensor in terms of its invariants. Geophysical Journal
  International, 141(2), 321-336. DOI not confirmed;
  record: http://hdl.handle.net/1885/89903
- Caldwell, T. G., Bibby, H. M., Brown, C. (2004). The magnetotelluric phase
  tensor. Geophysical Journal International, 158(2), 457-469.
  doi:10.1111/j.1365-246X.2004.02281.x
- Booker, J. R. (2014). The magnetotelluric phase tensor: a critical review.
  Surveys in Geophysics, 35(1), 7-40. doi:10.1007/s10712-013-9234-2
- Bahr, K. (1991). Geological noise in magnetotelluric data: a classification of
  distortion types. Physics of the Earth and Planetary Interiors, 66(1-2),
  24-38. doi:10.1016/0031-9201(91)90101-M

Not in dimens.bib (no DOI confirmed):

- Bahr, K. (1988). Interpretation of the magnetotelluric impedance tensor:
  regional induction and local telluric distortion. Journal of Geophysics, 62,
  119-127.
- Swift, C. M. (1967). A magnetotelluric investigation of an electrical
  conductivity anomaly in the southwestern United States. PhD thesis,
  Massachusetts Institute of Technology.

## Changelog

- 2026-11-28  Claude Sonnet 5.5 (Anthropic): initial Python port of the
  WALDIM 1.2 classification, with optional period/frequency bands.
- 2026-11-28  Claude Sonnet 5.5 (Anthropic): (b) EDI reading through py4mt
  load_edi; corrected rules are the default (waldim_compat=False); apparent
  resistivity, phase, strike, twist, shear, errors and band averages;
  write_results(); source errors corrected; driver script mt_dimens.py.
- 2026-11-28  Claude Sonnet 5.5 (Anthropic): (c) site_parameters() and
  save_site_npz(), one npz file per site.
- 2026-11-28  Claude Sonnet 5.5 (Anthropic): (d) renamed to dimens.py; phase
  tensor, skews, phase tensor class and time convention; reason codes; threshold
  test; optional bootstrap; band errors with scatter; .npz input;
  summarise_band(), survey_summary(), write_summary(); script
  mt_dimensionality.py replaces mt_dimens.py; dimensionality maps in
  femtic_plot_sitemap.py. AI-generated; review before production use.
- 2026-11-28  Claude Sonnet 5.5 (Anthropic): readme completed (requirements,
  function reference, parameter meanings, summary arrays, messages) and
  reordered; no change of the module.
