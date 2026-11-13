# femtic_distort_stat.py

Statistics and histograms of the galvanic-distortion matrices of a FEMTIC
ensemble (`ens` family): for every accepted member the file
`distortion_iter<N>.dat` of its best iteration is read, the distortion matrices
of all members and sites are collected, plotted as histograms and summarised.

---

## Purpose

If the inversions of an ensemble (RTO, jackknife, ...) estimate the distortion
matrices, each member gives its own `C` for every site. The script shows how
the estimated distortion is distributed over the members (the uncertainty of
the distortion of one site) and over the sites (how strong the distortion is in
the survey). It answers questions such as: is `C_xx` of site A well determined?
how large are the condition numbers? are there sites whose `C` is close to
singular?

To correct the data of a single run with its distortion result, see
`femtic_distcorr.py`; to map the distortion of one run, see
`femtic_plot_distort.py`.

## Members and iterations

The members are chosen as in `femtic_ens_post.py`, with the same settings
(`ENSEMBLE_DIR`, `ENSEMBLE_NAME`, `CNV_FILE`, `NRMS_MAX`, `ITER_FALLBACK`), so
that the distortion statistics belong to the same members as the model
statistics:

1. every sub-directory of `ENSEMBLE_DIR` whose name starts with
   `ENSEMBLE_NAME` is a candidate (files are ignored);
2. its best iteration is the iteration **> 0** with the smallest nRMS in the
   convergence file (ties: the lower iteration; iteration 0 is the starting
   model and is not eligible);
3. members with a best nRMS above `NRMS_MAX` are left out;
4. if `distortion_iter<best>.dat` does not exist, `ITER_FALLBACK = True` takes
   the iteration > 0 with the smallest nRMS **that has the file** (its nRMS must
   also be `<= NRMS_MAX`); with `False` the member is left out.

Members that are not used are listed with the reason (`no_cnv_file`,
`cnv_unreadable`, `cnv_no_iterations`, `rejected_nrms`, `no_distortion_file`,
`distortion_unreadable`). If no member is left, the script stops and prints the
first reasons.

## Quantities

FEMTIC writes the difference matrix `C'` of the type
`ESTIMATE_DISTORTION_MATRIX_DIFFERENCE`:

```
nsites
siteID  C'xx  C'xy  C'yx  C'yy
```

and the distortion matrix is `C = I + C'` (`fem.read_distortion_file`).
`QUANTITIES` may contain:

| Quantity | Meaning |
|---|---|
| `"xx"`, `"xy"`, `"yx"`, `"yy"` | Elements of `C` (`DISTORTION_KIND = "full"`) or of `C'` (`"prime"`; the diagonal is then `C - 1`). |
| `"condition"` | `cond(C) = sigma_max / sigma_min` (singular values of `C`); 1 = no distortion. A singular `C` gives `inf` and is not drawn. |
| `"determinant"` | `det(C)`; negative = the distortion reverses orientation. |
| `"gain"` | Mean singular value of `C`, `(sigma_max + sigma_min) / 2`. |

The last three are always those of the full matrix `C`. The singular values are
computed in closed form from the four elements (`sigma_max * sigma_min = |det|`,
`sigma_max^2 + sigma_min^2 = ||C||_F^2`), checked against `numpy.linalg.svd` in
the tests.

## Usage

Edit the configuration block at the top of the script and run

```bash
python femtic_distort_stat.py
```

Output (in `OUT_DIR`, default `ENSEMBLE_DIR/results/`):

| File | Content |
|---|---|
| `femtic_distort_stat.png` | Histograms of all members and sites pooled, one panel per quantity. |
| `femtic_distort_stat_site<ID>.png` | One figure per site of `PER_SITE`: the members of that site; the member with the smallest nRMS is marked by a triangle. |
| `femtic_distort_stat.csv` | `n`, `mean`, `std` (ddof = 1) and the `PERCENTILES` of every quantity, pooled (`site_id = ALL`) and per site. |
| `femtic_distort_stat.npz` | `members`, `iterations`, `nrms`, `site_ids`, `C` (members x sites x 2 x 2, NaN where a member has no such site), `kind`, `quantities`. |
| `femtic_distort_stat_summary.md` | The user-set parameters of the run (`utl.write_param_summary`), in `ENSEMBLE_DIR`. |

The pooled table is also printed (ASCII).

## Reading the histograms

- Each panel: bars = counts (or density), solid line = median, dashed lines =
  15.9 / 84.1 percentiles (the 1-sigma bounds of a normal distribution),
  dotted line = value without distortion (`C = I`); `n` and the median are
  above the panel. A caption names the line styles.
- **The pooled histograms mix two spreads**: the differences between the sites
  and the differences between the members. With distortions that differ from
  site to site they are broad or multimodal even if every site is well
  determined. Use `PER_SITE` and the per-site rows of the CSV for the
  uncertainty of one site.
- The spread over the members is the spread of the *estimates* of the
  ensemble, as produced by its perturbation (data, prior, ...); it is not
  automatically the full posterior uncertainty of the distortion.
- The condition number is the usual single number for the strength of a
  distortion matrix (Lilley, 2016); values near 1 mean little distortion.

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `ENSEMBLE_DIR`, `ENSEMBLE_NAME` | (user paths) | Directory of the ensemble and the start of the member names. |
| `CNV_FILE` | `["femtic.cnv", "FEMTIC-DABIC.cnv"]` | Convergence file(s) per member; first existing one is used. |
| `NRMS_MAX` | `1.5` | Members with a larger best nRMS are left out. |
| `ITER_FALLBACK` | `True` | See "Members and iterations". |
| `DISTORTION_FILE` | `"distortion_iter{iter}.dat"` | `{iter}` is replaced by the iteration number. |
| `RESULTS_H5_FILE` | `"results_iter{iter}.h5"` | Used only to check the type of the distortion parameters (first member that has the file); `None` = no check. Skipped without the file or without `h5py`. |
| `SITE_DAT` | `None` | Site list CSV (`mt_make_sitelist.py` format) for site names in titles and the CSV. |
| `DISTORTION_KIND` | `"full"` | `"full"` (`C`) or `"prime"` (`C'`). |
| `QUANTITIES` | `["xx", "xy", "yx", "yy"]` | See "Quantities". |
| `PERCENTILES` | `[2.3, 15.9, 50.0, 84.1, 97.7]` | For the CSV and the printed table. |
| `BINS` | `30` | Number of bins, a numpy rule (`"auto"`, `"fd"`, ...) or bin edges. |
| `DENSITY` | `False` | `True` = density instead of counts. |
| `RANGE` | `None` | `None` or `{quantity: (lo, hi)}` (also sets the x limits). |
| `SHOW_PERCENTILES`, `SHOW_REFERENCE` | `True`, `True` | Median / percentile lines; "no distortion" line. |
| `PER_SITE` | `None` | `None`, `"all"`, or a list of site IDs and / or names. |
| `MARK_BEST` | `True` | Triangle for the best-nRMS member in the per-site figures. |
| `FIGSIZE_PANEL`, `DPI`, colours, font sizes | see script | One panel's size; the figure is built from it. |
| `TITLE` | `"auto"` | `"auto"`; `None` / `False` / `""` = none; text with `{n_members}`, `{n_sites}`, `{site}`, `{id}`; or a dict `{"pooled": ..., "site": ..., "default": ...}`. |
| `SHOW` | `False` | Also show the figures on screen. |
| `OUT_DIR`, `OUTPUT_BASENAME`, `OUTPUT_FORMATS`, `STATS_CSV`, `RESULTS_NPZ` | see script | Output location and names. |

## Limits and things to check

- **Distortion type.** Only `ESTIMATE_DISTORTION_MATRIX_DIFFERENCE` is
  implemented. The files of the gains-only and gains-and-rotations types hold
  other parameters (see `fem.read_results_distortion`) and are not understood
  here. If a member's `results_iter<N>.h5` is present and `h5py` is available,
  the `/distortion` type is read (first member with the file only) and the
  script stops for any other type; otherwise it prints that the type was not
  checked. Make sure of the type for your runs.
- **Same sites in all members.** Sites are matched by ID. A site missing in
  some members gets fewer values (`n` in the CSV) and a warning is printed.
- **Site names.** The distortion files only hold IDs; names come from
  `SITE_DAT` (sitelist CSV) if given.
- **Large ensembles.** All matrices are kept in memory (members x sites x 4
  numbers), which is small. One figure per site (`PER_SITE = "all"`) can mean
  hundreds of files.
- **Not tested** with a real FEMTIC ensemble or with real `results_iter<N>.h5`
  files (no `h5py` in the test environment).

## Verification performed

Synthetic ensembles only; 15 tests in `test_femtic_distort_stat.py`, all
passing (pytest or `python test_femtic_distort_stat.py`):

- Elements of `C` and `C'`; condition number, gain and determinant against
  `numpy.linalg.svd` / `det` for 50 random matrices; NaN stays NaN, a singular
  matrix gives `inf`; wrong quantity or kind -> `ValueError`.
- Best-iteration rule (iteration 0 excluded, ties -> lower iteration).
- Member scan with every status: accepted, accepted by fallback, rejected by
  nRMS, fallback with nRMS above the limit, no distortion file, no / unreadable
  convergence file, no iteration > 0, `FEMTIC-DABIC.cnv`, a file and a
  non-matching directory in the ensemble directory; `ITER_FALLBACK = False`;
  `NRMS_MAX` applied to the best iteration.
- The file of the best iteration is read, not a later one (a different `C` was
  written for iteration 7); members with different site sets give NaN; a
  malformed file drops that member only.
- Statistics (n, mean, std, percentiles) with NaN values and with one value;
  pooled and per-site rows; mean and spread of 40 synthetic members agree with
  the true values; CSV is plain ASCII with the expected header.
- Choice of sites for the per-site figures (`None`, `"all"`, IDs and names,
  unknown site -> warning).
- Distortion-type check with a mocked `read_results_distortion` (type 1 passes,
  other types stop, no group / no `h5py` -> note); the real HDF5 path was
  **not** run.
- Figures: number of panels and hidden panels, title, no legend for one series,
  all values counted in the bins, constant and all-NaN data, `RANGE`, caption.
- `main()` end to end: files written, rejected member left out, iteration used,
  `C` in the `.npz` equal to the files, pooled count and mean in the CSV,
  titles (auto, text with placeholders, dict, none), `"prime"` kind, stop
  without members, stop for another distortion type; site names from a
  sitelist CSV (`SITE_DAT`) in the CSV and for `PER_SITE`.
- Both source files contain plain ASCII only.
- Figures of an 80-member, 6-site synthetic ensemble were looked at (pooled and
  per-site); an overlap of the "no distortion" label with the statistics text
  was found and removed (caption instead of per-panel labels).

## References

- Lilley, F. E. M. (2016). The distortion tensor of magnetotellurics: a
  tutorial on some properties. *Exploration Geophysics*, 47(2), 85-99.
  `doi:10.1071/EG14093` (DOI confirmed on the publisher's page; the paper uses
  the ratio of the singular values as a condition number to quantify
  distortion).
- Groom, R. W., Bailey, R. C. (1989). Decomposition of magnetotelluric
  impedance tensors in the presence of local three-dimensional galvanic
  distortion. *Journal of Geophysical Research*, 94(B2), 1913-1925.
  `doi:10.1029/JB094iB02p01913` (confirmed on an aggregator record,
  not on the publisher's page; EarthRef lists the same paper under
  `10.1029/88JB03636`, which was not checked).

## Provenance

| Date | Author | Change |
|---|---|---|
| 2026-11-13 | Claude Sonnet 5.5 (Anthropic) | Created: `femtic_distort_stat.py` reads `distortion_iter<best>.dat` of the accepted ensemble members (selection as in `femtic_ens_post.py`), pooled and per-site histograms of the elements, condition number, determinant and gain of `C`, statistics CSV and results `.npz`. Tests with synthetic ensembles in `test_femtic_distort_stat.py`. AI-generated; review before production use. |

Author: Volker Rath (DIAS)
