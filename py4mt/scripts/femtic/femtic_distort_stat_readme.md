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
| `femtic_distort_stat.<fmt>` | Histograms of all members and sites pooled, one panel per quantity. |
| `femtic_distort_stat_site<ID>.<fmt>` | One figure per site of `PER_SITE`: the members of that site; the member with the smallest nRMS is marked by a triangle. |
| `femtic_distort_stat.csv` | `n`, `mean`, `std` (ddof = 1) and the `PERCENTILES` of every quantity, pooled (`site_id = ALL`) and per site. |
| `femtic_distort_stat.npz` | `members`, `iterations`, `nrms`, `site_ids`, `C` (members x sites x 2 x 2, NaN where a member has no such site), `kind`, `quantities`. |
| `femtic_distort_stat_summary.md` | The user-set parameters of the run (`utl.write_param_summary`), in `ENSEMBLE_DIR`. |

`<fmt>` is every format of `OUTPUT_FORMATS` (one file per format and figure).
The pooled table is also printed (ASCII), followed by the distortion of the
best-fit member (see below).

## Plotting only some members

`PLOT_MEMBERS` limits the figures to one member or a short list; `None` (or
`"all"`) shows all accepted members. A selector is one of

| Selector | Selects |
|---|---|
| `"best"` | the best-fit member (smallest nRMS; ties: the first by name) |
| `"ann_rto_orig-err_017"` | the member with this directory name |
| `"ann_*_01?"` | all members whose name matches the pattern (`*`, `?`, `[..]`) |
| `17` or `"17"` | the member whose name ends in the number 17 (`_017`, `_17`, ...); stops with a message if several members match - then use the full name |

and `PLOT_MEMBERS` is one selector or a list of them, e.g. `"best"` or
`["best", 17, "ann_rto_orig-err_023"]`. Repeats are dropped; a selector that
matches nothing stops the script at the start (before anything is written) and
lists the available names.

- Only the figures use the selection. The statistics (CSV, NPZ, printed table)
  and the best-fit table always cover all accepted members.
- The histograms of **one** member show the distribution over its sites (so few
  values that you may want a smaller `BINS`). `PER_SITE` needs two or more
  members (one member gives one value per site) and is skipped with a note
  otherwise.
- With a selection the figure files carry the members' names, so that they do
  not overwrite the figures of the whole ensemble:
  `femtic_distort_stat_<member>.png`, `femtic_distort_stat_<a>+<b>.png` (or
  `femtic_distort_stat_<k>members.png` if the names are long), and
  `..._site<ID>.png` for `PER_SITE`. A selection that contains every member
  counts as no selection.
- The marker in the per-site figures is the member with the smallest nRMS *of
  the plotted members*. The title says "member X" or "k selected members";
  in a text `TITLE`, `{n_members}` is the number of plotted members and
  `{members}` their names (`all` without selection).

## Distortion of the best-fit member

With `PRINT_BEST = True` the script prints, for every site, the distortion
matrix of the best-fit member: the accepted member with the smallest nRMS (at
the iteration used; ties: the first by name; the same member as the triangle in
the per-site figures):

```
Distortion of the best-fit member ann_000 (iteration 6, nRMS 1.1000): elements of C = I + C'
====================================================================
    site       Cxx       Cxy       Cyx       Cyy   cond(C)    det(C)
--------------------------------------------------------------------
     201    0.9879   -0.0209    0.0410    0.5682     1.739    0.5622
     ...
```

The four columns are the elements of `C` (`DISTORTION_KIND = "full"`) or of
`C'` (`"prime"`); `cond(C)` and `det(C)` are always those of the full `C`. A
name column appears if `SITE_DAT` gives names; a negative determinant is marked
`det<0`; a site that is not in the member's distortion file is listed as such.
The example is from the synthetic test ensemble, not from real data.

## Output formats

`OUTPUT_FORMATS` is a list, e.g. `["png", "pdf", "jpg"]`, or one string
(`"pdf"`, or `"pdf, jpg"`). Any format that matplotlib writes can be used:

| Kind | Formats | Note |
|---|---|---|
| Raster | `png`, `jpg` (`jpeg`), `tif` (`tiff`), `webp` | Resolution `DPI`; `JPEG_QUALITY` (1 - 100, default 95) for `jpg` / `jpeg`. |
| Vector | `pdf`, `svg`, `eps`, `ps` | `DPI` has no effect. Text stays text in `pdf` / `eps` / `ps` (fonts embedded as Type 42, so it can be edited in a vector program). |

Names are not case sensitive and a leading dot is ignored (`".PDF"` = `"pdf"`);
repeated names are dropped; `"jpg"` and `"jpeg"` give files with the extension
as written. An unknown name or an empty list stops the script at the start,
before the ensemble is read, with a message that lists the allowed formats.
A JPEG has no transparency and is lossy; use `png` or a vector format for line
art that is processed further.

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
| `BINS` | `20` | Number of bins, a numpy rule (`"auto"`, `"fd"`, ...) or bin edges. |
| `DENSITY` | `False` | `True` = density instead of counts. |
| `RANGE` | `None` | `None` or `{quantity: (lo, hi)}` (also sets the x limits). |
| `SHOW_PERCENTILES`, `SHOW_REFERENCE` | `True`, `True` | Median / percentile lines; "no distortion" line. |
| `PER_SITE` | `None` | `None`, `"all"`, or a list of site IDs and / or names. |
| `MARK_BEST` | `True` | Triangle for the best-nRMS member in the per-site figures. |
| `PLOT_MEMBERS` | `None` | Members shown in the figures, see "Plotting only some members". |
| `PRINT_BEST` | `True` | Print the distortion matrices of the best-fit member, see above. |
| `FIGSIZE_PANEL` | `(3.6, 2.9)` | One panel's size in inches; the figure is made from it, title and caption are added to the height and wrapped to the width. Holds font sizes up to 12; make it larger for larger fonts (matplotlib warns "Tight layout not applied" if it is too small). |
| `DPI` | `400` | Resolution of the raster formats. |
| `TICK_FONTSIZE`, `LABEL_FONTSIZE_AXES`, colours | `12`, `12`, see script | Font sizes and the colours of bars, text, lines, grid and background. |
| `TITLE` | `""` (no title) | `"auto"`; `None` / `False` / `""` = none; text with `{n_members}`, `{n_sites}`, `{site}`, `{id}`; or a dict `{"pooled": ..., "site": ..., "default": ...}`. |
| `SHOW` | `False` | Also show the figures on screen. |
| `OUT_DIR`, `OUTPUT_BASENAME`, `STATS_CSV`, `RESULTS_NPZ` | see script | Output location and names. |
| `OUTPUT_FORMATS` | `["png"]` | Figure formats, see "Output formats". |
| `JPEG_QUALITY` | `95` | Quality of `jpg` / `jpeg` files, 1 - 100. |

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

Synthetic ensembles only; 27 tests in `test_femtic_distort_stat.py`, all
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
- Output formats: names normalised (case, dot, separators, duplicates, bad and
  empty input); the files of png, pdf, jpg, svg, tif and eps have the right
  signature; pdf text is embedded as Type 42 (and the default would not be);
  JPEG quality changes the file size; `DPI` changes only raster files;
  `main()` with `"pdf, .JPG"` writes the pooled and the per-site figure in both
  formats; an unknown format or a `JPEG_QUALITY` outside 1 - 100 stops before
  anything is read or written.
- Layout: title and caption lie inside the figure and do not overlap the
  panels (font sizes 8 and 12, 1, 2 and 5 panels, a long title); count axes
  have whole-number ticks.
- Best-fit table: values, condition number and determinant against `numpy`,
  `det<0` flag, names column, a site missing in the member, `"prime"` kind,
  fallback note; `main()` prints the member with the smallest nRMS (with the
  matrix of the right site) and nothing with `PRINT_BEST = False`.
- Member selection: every selector form (`"best"`, name, pattern, number as int
  and as string, order, repeats, numpy integer), the errors (unknown name,
  pattern without match, ambiguous number, empty list, `True`, float, `None` in
  a list), names consisting of digits; file-name tags. `main()`: one member
  (figure data equal to that member's matrices, statistics and `.npz` still
  from all members, `PER_SITE` skipped), a list of two (pooled and per-site
  figure from exactly those members, marker = the better of the two, file
  names), a selection of every member (default names and title), `"best"` with
  the `{members}` / `{n_members}` placeholders, and a failing selector that
  stops before anything is written.
- Both source files contain plain ASCII only.
- Figures of an 80-member, 6-site synthetic ensemble were looked at (pooled and
  per-site, as jpg and png): an overlap of the "no distortion" label with the
  statistics text was found and removed (caption instead of per-panel labels);
  with font size 12 the caption was clipped at both edges and the count axis
  showed 2.5 / 7.5, both fixed (caption and title wrapped to the figure width,
  whole-number count ticks). The pdf files were checked for signature and
  fonts, not looked at.

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
| 2026-11-13b | Claude Sonnet 5.5 (Anthropic) | `OUTPUT_FORMATS` accepts a list or one string and any format matplotlib writes (png, pdf, jpg, svg, eps, tif, ...); names are normalised and checked at the start; new `JPEG_QUALITY`; text stays text in pdf / eps / ps. Title and caption are wrapped to the figure width (the caption was clipped with font size 12); count axes have whole-number ticks. Built on the user's version of the script (`BINS = 20`, `DPI = 400`, font sizes 12, `TITLE = ""` kept). AI-generated; review before production use. |
| 2026-11-13c | Claude Sonnet 5.5 (Anthropic) | New `PRINT_BEST`: table of the distortion matrices (elements of `C` or `C'`, `cond(C)`, `det(C)`) of the best-fit member for every site. AI-generated; review before production use. |
| 2026-11-14 | Claude Sonnet 5.5 (Anthropic) | New `PLOT_MEMBERS`: figures for one member or a short list of members (`"best"`, directory names, patterns, numbers); figure file names carry the members' names; `PER_SITE` needs two or more members; statistics and best-fit table still cover all members. AI-generated; review before production use. |

Author: Volker Rath (DIAS)
