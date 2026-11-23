# femtic_initial_post

Post-process a starting-model study with FEMTIC: for every run made by
`femtic_initial_prep.py`, take the **best nRMS** and plot it against
**log10 of the fill value** (the starting resistivity). The run with the
smallest best nRMS is marked and reported.

Built on the pattern of `femtic_lcurve_post.py` (run harvesting by glob,
`CNV_FILE`, plot style, report, npz, parameter summary).

---

## Workflow

| Step | Action |
|------|--------|
| 1 | `femtic_initial_prep.py` creates `ini_<rho>/` for every value in `RHO_LIST` |
| 2 | Run FEMTIC in every `ini_<rho>/` |
| 3 | `femtic_initial_post.py` reads `femtic.cnv` and the starting model of each run, plots best nRMS versus log10(fill value), writes the report |

Needs at least 2 usable runs. If the minimum is at the smallest or largest fill
value tried, the script warns: the minimum may lie outside the sampled range;
extend `RHO_LIST` and re-run step 1 (`SKIP_FINISHED` leaves finished runs
alone).

---

## What is taken from each run

| Quantity | Source |
|----------|--------|
| fill value | `FILL_SOURCE = "model"` (default): median resistivity of the free regions of `MODEL_FILE` in the run directory (`fem.read_model`). The number in the directory name is compared with it (warning if log10 differs by more than 1e-3; the model value is used). `"name"`: the number after `ENSEMBLE_NAME` in the directory name, no block file is read (faster for a large mesh) |
| best nRMS, best iteration | smallest nRMS of all lines of the convergence file: `fem.get_nrms()`, the definition also used for the best iteration in `femtic_plot_nrms.py` |
| last nRMS, last iteration, number of lines | last line of the same file; shown in the report to expose runs that end worse than their best iteration (diverging, or not converged) |

Column positions come from the file's own header row (`fem.read_cnv`).

A run is reported and skipped if it has no usable convergence file, non-finite
or non-positive nRMS, an unreadable starting model, or (with
`FILL_SOURCE = "model"`) a starting model that is **not homogeneous** (spread of
log10 rho above 1e-5 among the free regions, e.g. a link to the template left
from a wrong setup). Two runs with the same fill value stop the script.

How to read the plot: a flat curve means that the result does not depend on the
start; a clear minimum or trend means that it does, i.e. runs are (partly) stuck
in local minima or not converged within the iterations allowed.

---

## Output

Stem `PLOT_NAME` (default `<ENSEMBLE_DIR>initial_nrms`):

| File | Description |
|------|-------------|
| `<PLOT_NAME>.<ext>` | Best nRMS (y) versus log10(fill value / Ohm.m) (x) (`PLOT_FORMATS`), every point labelled with its fill value in Ohm.m, best run marked by a star |
| `<PLOT_NAME>_best.txt` | Best run and table of all runs: fill value, log10, best nRMS and iteration, last nRMS and iteration, number of lines, directory (same text as printed) |
| `<PLOT_NAME>_best.npz` | `fill_rho`, `fill_log10`, `nrms_best`, `iter_best`, `nrms_last`, `iter_last`, `n_lines`, `dirs`, `best_index`, `best_fill_rho`, `best_nrms`, `fill_source` |
| `femtic_initial_post_summary.md` | User-set (UPPERCASE) parameters, script path, run date/time, written to `ENSEMBLE_DIR` |

---

## Configuration variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENSEMBLE_DIR` | `str` | `.../sabancaya2026/initial/` | Directory with the run directories (as in `femtic_initial_prep.py`) |
| `ENSEMBLE_NAME` | `str` | `"ini_"` | Run directories are `<ENSEMBLE_DIR><ENSEMBLE_NAME>*` |
| `CNV_FILE` | `str` or `list` | `["femtic.cnv", "FEMTIC-DABIC.cnv"]` | Convergence file name(s); first existing wins |
| `MODEL_FILE` | `str` | `"resistivity_block_iter0.dat"` | Starting model in each run directory |
| `FILL_SOURCE` | `str` | `"model"` | `"model"` or `"name"`, see above |
| `PLOT_NAME` | `str` | `ENSEMBLE_DIR + "initial_nrms"` | Stem of the output files |
| `PLOT_TITLE` | `str`, `dict`, `None` | `"auto"` | `"auto"` = "Starting model test"; `None`, `False`, `""` = no title; a string; a dict `{"default": spec}` (`fviz.resolve_title`) |
| `FONTSIZE` | `int` | `9` | Base font size |
| `PLOT_LOG_Y` | `bool` | `False` | Log10 axis for the nRMS |
| `PLOT_XLIM`, `PLOT_YLIM` | `[min, max]` or `None` | `None` | Axis limits (x in log10 Ohm.m, y in nRMS) |
| `PLOT_ANNOTATE` | `bool` | `True` | Label every point with its fill value |
| `PLOT_SIZE_CM` | `[w, h]` | `[8.5, 8.5]` | Figure size in cm |
| `BEST_COLOR`, `BEST_MARKER`, `BEST_SIZE` | | `"yellow"`, `"*"`, `14` | Marker of the best run |
| `PLOT_FORMATS` | `list` of `str` | `["pdf", "svg", "jpg"]` | Output formats |
| `PLOT_DPI` | `int` | `600` | Resolution of raster formats |

---

## Dependencies

| Module | Source | Purpose |
|--------|--------|---------|
| `numpy`, `matplotlib` | PyPI | Arrays, plot |
| `femtic` | Py4MTX | `read_cnv`, `get_nrms`, `read_model` |
| `femtic_viz` | Py4MTX | `resolve_title` |
| `util` | Py4MTX | `print_title`, `write_param_summary` |
| `version` | Py4MTX | Version string for the title banner |

Needs the environment variable `PY4MTX_ROOT`.

---

## Provenance

| Date | Author | Description |
|------|--------|-------------|
| 2026-11-23 | Claude Sonnet 5.5 (Anthropic) | Created. Built from `femtic_lcurve_post.py` and `fem.get_nrms`. AI-generated; review before production use. |
