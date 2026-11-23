# femtic_initial_prep

Prepare the run directories for a starting-model study with FEMTIC. One
directory is created per resistivity in `RHO_LIST`. Each gets its own
`resistivity_block_iter0.dat` in which **every free region is set to that
resistivity** (a homogeneous starting model); everything else is symlinked to
the templates, as in `femtic_lcurve_prep.py` and `femtic_rto_prep.py`.

The directories are named `<ENSEMBLE_NAME><rho>` (default `ini_10`, `ini_100`,
...), with rho in Ohm.m; `femtic_initial_post.py` finds them with the glob
`ini_*`.

The script is the deterministic, list-driven counterpart of
`MOD_ITER0_HOMOGENEOUS` in `femtic_rto_prep.py` (same block-file handling via
`fem.insert_model`; air, ocean and fixed regions kept).

---

## Workflow

| Step | Action |
|------|--------|
| 1 | `femtic_initial_prep.py` creates `ini_<rho>/` for every value in `RHO_LIST` |
| 2 | Run FEMTIC in every `ini_<rho>/` |
| 3 | `femtic_initial_post.py` takes the best nRMS of every run, plots it against log10(fill value) and reports the best starting resistivity |

Everything that is common to all runs (`ITERATION`, `STEP_LENGTH`,
`TRADE_OFF_PARAMETER`, ...) is set once in `templates/control.dat`.

---

## Input

| File | Location | Description |
|------|----------|-------------|
| `resistivity_block_iter0.dat` (`MODEL_FILE`) | `TEMPLATES` | Template starting model; v4 or v5 block format. Only read |
| files of `COPY_LIST` / `LINK_LIST` | `TEMPLATES` | Copied / linked into every run directory; names missing in `TEMPLATES` are reported and skipped |

## Output

| Item | Description |
|------|-------------|
| `<ENSEMBLE_DIR><ENSEMBLE_NAME><rho>/` | One run directory per resistivity: a private `resistivity_block_iter0.dat` plus links |
| `femtic_initial_prep_summary.md` | User-set (UPPERCASE) parameters, script path, run date/time, written to `ENSEMBLE_DIR` |

A table (run number, rho, log10 rho, status, directory) is printed at the end.

---

## What is written

The starting model of a run is the template with:

| Part of the block file | In the run |
|------------------------|------------|
| element-to-region map | unchanged |
| free regions | resistivity = fill value (all of them) |
| air (region 0), ocean, any other region with flag 1 | unchanged, values taken from the template |
| lower / upper bounds, `n`, flags | unchanged |

Free regions are those that `fem.read_model` returns: not region 0, not flagged
fixed (flag 1; this includes an ocean block). The template's own air and ocean
resistivities are passed to `fem.insert_model`, so they are **not** replaced by
the defaults of that function (1e9 and 0.25 Ohm.m).

For anisotropic (v5, TI / GA) free regions `fem.insert_model` sets `rhoXX` to
the fill value and scales `rhoYY` / `rhoZZ` by the same factor: the anisotropy
ratio of each region is kept, strike, dip and fix flags are unchanged. A note is
printed in that case.

### Reference model

`referencemodel.dat` (the regularisation prior) is **not** changed. With it in
`LINK_LIST` it is the same file in every run, so only the starting model
differs. If the template has none, the script prints a note: check in the
FEMTIC documentation what is then used as reference; if it is the starting
model, the runs differ in start **and** prior.

---

## Checks and safety

Before anything is created:

- `MODEL_FILE` must be in neither `COPY_LIST` nor `LINK_LIST` (it is written in
  every run directory; a link would write into the template);
- `RHO_LIST` non-empty, finite, resistivities > 0, no two values with the same
  directory name (6 significant digits);
- the template has free regions, and `fem.read_model` agrees on their number;
- starting resistivity against the `[lower, upper]` bounds of the free regions
  in the template: reported, and the script stops (`ABORT_ON_BOUNDS`).

While writing:

- a link (or file) of the name `MODEL_FILE` left in a run directory by an
  earlier setup is removed first, so the template is never written through a
  link;
- with `VERIFY`, each written model is read back and compared with the template:
  element map, bounds, `n`, flags and fixed regions unchanged, free regions all
  equal to the fill value (relative tolerance 1e-5, the file holds 7 digits);
- the checksum of the template is compared before and after.

---

## Configuration variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENSEMBLE_DIR` | `str` | `.../sabancaya2026/initial/` | Directory with `templates/`; run directories are created here |
| `ENSEMBLE_NAME` | `str` | `"ini_"` | Prefix of the run directories |
| `TEMPLATES` | `str` | `ENSEMBLE_DIR + "templates/"` | Template directory |
| `RHO_LIST` | `list` of `float` | `[10, 30, 100, 300, 1000]` | Fill values, one run each; finite and > 0 Ohm.m |
| `RHO_LOG10` | `bool` | `False` | `True`: `RHO_LIST` is in log10(Ohm.m), e.g. `[1, 2, 3]`; directory names still use Ohm.m |
| `MODEL_FILE` | `str` | `"resistivity_block_iter0.dat"` | The starting model that is filled |
| `COPY_LIST` | `list` of `str` | `[]` | Files copied into every run directory |
| `LINK_LIST` | `list` of `str` | control, mesh, observe, distortion, reference model, site, run scripts | Files symlinked into every run directory |
| `RELATIVE_LINKS` | `bool` | `True` | Relative symlinks (portable, survive tgz); `False` = absolute |
| `ABORT_ON_BOUNDS` | `bool` | `True` | Stop before creating anything if a fill value is outside the bounds of a free region; `False` = report only |
| `VERIFY` | `bool` | `True` | Read every written model back and compare it with the template |
| `SKIP_FINISHED` | `bool` | `True` | Leave a run directory alone if it already holds a convergence file, so extending `RHO_LIST` never rewrites finished runs |
| `CNV_FILE` | `str` or `list` | `["femtic.cnv", "FEMTIC-DABIC.cnv"]` | Convergence file name(s) used for `SKIP_FINISHED` |

The directory name holds the value in Ohm.m with 6 significant digits
(`ini_100`, `ini_31.6228`, `ini_1e+06`).

---

## Dependencies

| Module | Source | Purpose |
|--------|--------|---------|
| `numpy` | PyPI | Value list handling |
| `femtic` | Py4MTX | `read_resistivity_block`, `read_model`, `insert_model`; imported before `ensembles` (the two import each other) |
| `ensembles` | Py4MTX | `put_files` (copy / link) |
| `util` | Py4MTX | `print_title`, `write_param_summary` |
| `version` | Py4MTX | Version string for the title banner |

Needs the environment variable `PY4MTX_ROOT`.

---

## Provenance

| Date | Author | Description |
|------|--------|-------------|
| 2026-11-23 | Claude Sonnet 5.5 (Anthropic) | Created as `femtic_initial.py`, renamed to `femtic_initial_prep.py` the same day. Modelled on `femtic_lcurve_prep.py`; model written with `fem.insert_model`. AI-generated; review before production use. |
