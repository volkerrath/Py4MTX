# femtic_distcorr.py

Apply the galvanic-distortion result of a FEMTIC inversion to the observations:
read `observe.dat` and `distortion_iter<N>.dat` (N = the best iteration, from
`femtic.cnv`) and save a new observations file in which the impedances of every
MT site are corrected for the estimated distortion matrix of that site.

---

## Purpose

When FEMTIC inverts for the distortion matrices, the observed impedances are
fitted as the distorted response of the model. The script removes the estimated
distortion from the observations (`MODE = "remove"`), so that the new file holds
the data as the model "sees" them. It can also do the opposite
(`MODE = "apply"`), for example to test the direction of the correction.

## Method

FEMTIC's model of the data, with the standard form of galvanic distortion
(`E_obs = C E`, the magnetic field is not distorted):

```
Z_obs = C Z_model ,      C = I + C'
```

FEMTIC estimates the difference matrix `C'` (type
`ESTIMATE_DISTORTION_MATRIX_DIFFERENCE`) and writes it to
`distortion_iter<N>.dat`:

```
nsites
siteID  C'xx  C'xy  C'yx  C'yy
```

`C = C' + I` is formed by `fem.read_distortion_file()`. Per MT site and
frequency, with `M = C^-1` (`"remove"`) or `M = C` (`"apply"`):

```
Z_new = M Z_obs
```

**Errors** (`ERROR_MODE = "propagate"`): first-order propagation, the four
impedance elements taken as independent (as FEMTIC does, Re and Im separately),
`C` taken as exact:

```
err(Re Z_new_ij) = sqrt( sum_k M_ik^2 err(Re Z_kj)^2 )      (same for Im)
```

`ERROR_MODE = "keep"` leaves the error columns unchanged.

**Other blocks.** Only MT blocks are changed. VTF blocks (magnetic quantities)
are not affected by galvanic distortion. PT blocks are left as they are: the
phase tensor does not change under a real distortion matrix (Caldwell et al.,
2004). Station lines, preamble, `END` line and sites with 0 frequencies are
written back as they were. All data rows of the new file are rewritten in the
format `%.8E` (`fem.write_observe_dat`), which changes the number format of the
VTF / PT rows but not their values (relative difference below 1e-8).

## Usage

Edit the configuration block at the top of the script and run

```bash
python femtic_distcorr.py
```

Output (iteration 7 as an example):

| File | Content |
|---|---|
| `observe_distcorr_iter7.dat` | The new observations file. |
| `observe_distcorr_iter7_report.txt` | ASCII table, one line per MT site: ID, name, number of frequencies, `cond(C)`, `det(C)`, `C`, status. |
| `observe_distcorr_iter7_plot_overview.png` | All MT sites, original (solid) and corrected (dashed), apparent resistivity and phase (`femtic_viz.plot_data_ensemble`). |
| `observe_distcorr_iter7_plot_site<ID>.png` | The same for one site, title with `cond(C)`; the `PLOT_SITES` most distorted sites. |
| `femtic_distcorr_summary.md` | The user-set parameters of the run (`utl.write_param_summary`). |

The original `observe.dat` is never overwritten (the script stops if
`OUTPUT_FILE` is the input file).

## Plots

With `PLOT = True` the original and the new file are plotted with
`femtic_viz.plot_data_ensemble` (original solid, new dashed; both files are read
back from disk, so the figures show what was written). The overview figure
draws all MT sites; the per-site figures (`PLOT_SITES`) are made from small
temporary single-site files. `PLOT_SITES`: an integer N = the N corrected sites
with the largest `||C - I||` (default 6), `"all"`, a list of IDs and / or names,
or `None`. The plot step is the last one: if it fails, the message is printed
and the observation file stays.

`plot_data_ensemble` continues the colour cycle over all curves it draws, so
the dashed curve of a component would get another colour than its solid curve.
The script therefore recolours the curves of the `"rho"` / `"phase"` panels
per component (XY, YX, ...) and rebuilds the legend; nothing else of the figure
is changed. (The same behaviour is in `plot_data_ensemble` itself, e.g. for
original / perturbed data.)

For a distortion matrix near the identity the phases hardly change: `C` mostly
changes the amplitude (static-shift-like) of the dominant off-diagonal
elements.

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `WORK_DIR` | `/home/vrath/work/femtic_run/` | Run directory (inputs and outputs). |
| `OBSERVE_FILE` | `"observe.dat"` | Observations to be corrected; name inside `WORK_DIR` or a full path. |
| `CNV_FILE` | `["femtic.cnv", "FEMTIC-DABIC.cnv"]` | Convergence file(s); first existing one is used. |
| `ITERATION` | `"best"` | `"best"` (smallest nRMS in the convergence file, `fem.get_nrms`, iteration 0 included), `"last"` (last row), or an integer. |
| `DISTORTION_FILE` | `"distortion_iter{iter}.dat"` | `{iter}` is replaced by the iteration number. |
| `OUTPUT_FILE` | `"observe_distcorr_iter{iter}.dat"` | New observations file. |
| `REPORT_FILE` | `"observe_distcorr_iter{iter}_report.txt"` | `None` = no report file (the table is printed anyway). |
| `RESULTS_H5_FILE` | `"results_iter{iter}.h5"` | Used only to check the type of the distortion parameters; `None` = no check. Skipped without the file or without `h5py`. |
| `MODE` | `"remove"` | `"remove"`: `Z_new = C^-1 Z_obs`; `"apply"`: `Z_new = C Z_obs`. |
| `ERROR_MODE` | `"propagate"` | `"propagate"` or `"keep"`. |
| `MATCH_BY` | `"auto"` | `"id"` (site ID), `"order"` (i-th MT site = i-th row; equal counts needed), `"auto"` (`"id"` if all MT station lines have the FEMTIC layout `ID H-ID ... X Y`, else `"order"`). |
| `MISSING_SITES` | `"keep"` | MT site without a row in the distortion file: `"keep"` (unchanged, warning) or `"error"`. |
| `MAX_COND` | `1.0e6` | Sites whose `C` has a larger condition number (or is singular) are kept unchanged. |
| `VERIFY_OUTPUT` | `True` | Re-read the written file and compare with the expected result. |
| `PLOT` | `True` | `False` = no figures. |
| `PLOT_OVERVIEW`, `PLOT_OVERVIEW_NSITES` | `True`, `None` | Overview figure; draw only this many randomly chosen sites (`None` = all). |
| `PLOT_SITES` | `6` | Per-site figures, see above. |
| `PLOT_WHAT` | `["rho", "phase"]` | Panels (columns): `"rho"`, `"phase"`, `"tipper"`, `"pt"` (the last two are not changed by the correction). |
| `PLOT_COMPS` | `"xy,yx"` | Components of the `"rho"` / `"phase"` panels (subset of `xx,xy,yx,yy`). |
| `PLOT_ERRORS` | `True` | Error envelopes of both data sets. |
| `PLOT_PERIOD_LIMITS` | `None` | `(T_min, T_max)` in s. |
| `PLOT_FILE`, `PLOT_FORMATS`, `PLOT_DPI` | `"observe_distcorr_iter{iter}_plot"`, `["png"]`, `200` | File name stem (`_overview` / `_site<ID>` and the format are appended), formats, resolution. |
| `PLOT_TITLE` | `"auto"` | `"auto"`; `None` / `False` / `""` = none; text with `{site}`, `{id}`, `{cond}`, `{iter}`, `{new}`; or a dict `{"overview": ..., "site": ..., "default": ...}`. |
| `SHOW` | `False` | Also show the figures on screen. |

## Limits and things to check

- **Distortion type.** Only `ESTIMATE_DISTORTION_MATRIX_DIFFERENCE` is
  implemented, the type for which `fem.read_distortion_file` gives `C = C' + I`.
  The files of the gains-only and gains-and-rotations types hold other
  parameters (see `fem.read_results_distortion`); their definition was not
  checked here, so they are not implemented. If `results_iter<N>.h5` is present
  and `h5py` is available, its `/distortion` type is read and the script stops
  for any other type. Without that file the type is not checked, so make sure
  of it for your run.
- **Direction.** `Z_obs = C Z_model` is the standard galvanic-distortion model.
  That FEMTIC applies `C` in exactly this form was not checked against the
  FEMTIC source or manual here. A cheap check on real data: run the script with
  `MODE = "apply"` on the corrected file (same `DISTORTION_FILE`) and compare
  with the original `observe.dat` (values agree; errors do not, they are
  propagated twice).
- **Site IDs.** With `MATCH_BY = "id"` the first column of the distortion file
  must be the ID in the first column of the MT station lines of `observe.dat`.
  If no ID matches, the script stops and suggests `MATCH_BY = "order"`. IDs that
  are only on one side are listed as warnings (the distortion file of the
  dummy-VTF workflow may list fewer stations than `observe.dat` has MT sites;
  see `femtic_add_dummy_vtf_readme.md`).
- **Singular `C`, `det(C) < 0`.** A site with `cond(C) > MAX_COND` is left
  unchanged and reported as `singular`. A corrected site with `det(C) < 0` is
  marked in the report; such a distortion matrix reverses orientation and is
  worth a look (Lilley, 2016).
- **Errors.** The propagated errors ignore the uncertainty of `C` itself and
  correlations between the impedance elements. After the correction, the
  off-diagonal errors enter the diagonal elements and the other way round.
- **Blocks.** `fem.read_observe_dat` reads MT, VTF and PT only; a file with
  `APP_RES_AND_PHS` or `HTF` blocks stops with that function's error message.
- **Re-running FEMTIC** with the corrected file: the distortion is already
  removed, so switch the distortion estimation off, or expect a distortion
  result close to the identity.

## Verification performed

Synthetic files only (no real FEMTIC run was available); 20 tests in
`test_femtic_distcorr.py`, all passing:

- `Z_obs = C Z` built from a known `Z`: `"remove"` recovers `Z` to 1e-15.
- Propagated errors compared with a Monte Carlo estimate (400 000 draws): agree
  within 2 %. One value also checked by hand (site 1, first frequency).
- `"remove"` followed by `"apply"` gives back the original data (1e-9).
- Statuses `corrected` / `identity` / `singular` / `no distortion row`; VTF and
  PT data unchanged; unchanged sites bit-identical.
- Station lines (also with owner column and dummy VTF stations with 0
  frequencies), preamble and `END` line preserved; original file untouched.
- Legacy `name x y z` layout via `MATCH_BY = "auto"` -> `"order"`; unequal
  counts, no matching IDs, `MISSING_SITES = "error"`, output equal to input:
  all refused with a clear message.
- Iteration choice (`"best"`, `"last"`, number) from a synthetic `femtic.cnv`.
- Distortion-type check with a mocked `read_results_distortion` (type 1 passes,
  type 2 stops); the real HDF5 path was **not** run (no `h5py` here).
- The built-in `VERIFY_OUTPUT` detects a tampered output file.
- The report is plain ASCII.
- Plots: file names and formats (png, pdf), file contents (PNG header, size),
  titles (auto, none, text with placeholders, dict), choice of sites (N most
  distorted, all, IDs / names, unknown name -> warning), `"apply"` wording,
  recolouring per component, and a failing plot step leaves the output file in
  place. The figures of one synthetic site were also looked at (dashed curve of
  a component has the colour of its solid curve, legend correct).

Not tested: real FEMTIC files, files with many sites (the overview with many
sites is crowded; use `PLOT_OVERVIEW_NSITES`), the `h5py` path.

## References

- Caldwell, T. G., Bibby, H. M., Brown, C. (2004). The magnetotelluric phase
  tensor. *Geophysical Journal International*, 158(2), 457-469.
  `doi:10.1111/j.1365-246X.2004.02281.x`
- Groom, R. W., Bailey, R. C. (1989). Decomposition of magnetotelluric
  impedance tensors in the presence of local three-dimensional galvanic
  distortion. *Journal of Geophysical Research*, 94(B2), 1913-1925.
  `doi:10.1029/JB094iB02p01913`
- Lilley, F. E. M. (2016). The distortion tensor of magnetotellurics: a
  tutorial on some properties. *Exploration Geophysics*, 47(2), 85-99.
  `doi:10.1071/EG14093`

(DOI checks: Caldwell et al. confirmed on the paper itself; Lilley confirmed on
the publisher's page (2026-11-13); Groom and Bailey confirmed on an aggregator
record (Scilit: volume, pages and DOI agree), not on the publisher's page -
EarthRef lists the same paper under `10.1029/88JB03636`, which was not checked.)

## Provenance

| Date | Author | Change |
|---|---|---|
| 2026-11-13 | Claude Sonnet 5.5 (Anthropic) | Created: `femtic_distcorr.py` reads `observe.dat` and `distortion_iter<N>.dat` (N from `femtic.cnv`: best / last / number), corrects (or applies) the distortion matrices of the MT sites with first-order error propagation, writes a new observations file and a per-site report, and re-reads the written file as a check. Tests with synthetic files in `test_femtic_distcorr.py`. AI-generated; review before production use. |
| 2026-11-13b | Claude Sonnet 5.5 (Anthropic) | Plots of the original and the corrected data with `femtic_viz.plot_data_ensemble`: overview of all MT sites and one figure per site for the most distorted sites; curves recoloured per component so that solid and dashed curves match. New settings `PLOT`, `PLOT_OVERVIEW`, `PLOT_OVERVIEW_NSITES`, `PLOT_SITES`, `PLOT_WHAT`, `PLOT_COMPS`, `PLOT_ERRORS`, `PLOT_PERIOD_LIMITS`, `PLOT_FILE`, `PLOT_FORMATS`, `PLOT_DPI`, `PLOT_TITLE`, `SHOW`. Needs `femtic_viz.py` and `data_viz.py` of the project. AI-generated; review before use. |

Author: Volker Rath (DIAS)
