# femtic_lcurve_prep

Prepare the run directories for an L-curve study with FEMTIC. One directory
is created per trade-off parameter (alpha) in `TRADE_OFF_LIST`. Each gets its
own copy of `control.dat` (the only entry of `COPY_LIST`) in which the value
of `TRADE_OFF_PARAMETER` is replaced; everything else is symlinked to the
templates, as in `femtic_rto_prep.py`.

The directories are named `<ENSEMBLE_NAME><alpha>` (default `lc_0.3`, `lc_10`,
...), which is what `femtic_lcurve_plot.py` expects (`SEARCH_STRNG = "lc_*"`).

---

## Workflow

| Step | Action |
|------|--------|
| 1 | `femtic_lcurve_prep.py` creates `lc_<alpha>/` for every value in `TRADE_OFF_LIST` |
| 2 | Run FEMTIC in every `lc_<alpha>/` (fixed alpha: `INV_METHOD 1`, `TYPE_OF_TRADE_OFF_PARAMETER 0`) |
| 3 | `femtic_lcurve_plot.py` reads the last line of `femtic.cnv` of each run and plots roughness vs. nRMS |
| 4 | `femtic_lcurve_post.py` does the same and also estimates the alpha at the corner (maximum curvature), see `femtic_lcurve_post_readme.md` |

Everything that is common to all runs (`ITERATION`, `STEP_LENGTH`,
`DISTORTION`, ...) is set once in `templates/control.dat`.

---

## Input

| File | Location | Description |
|------|----------|-------------|
| `control.dat` | `TEMPLATES` | Template control file, must contain the trade-off keyword exactly once |
| files of `LINK_LIST` | `TEMPLATES` | Linked into every run directory; names missing in `TEMPLATES` are reported and skipped |

## Output

| Item | Description |
|------|-------------|
| `<ENSEMBLE_DIR><ENSEMBLE_NAME><alpha>/` | One run directory per alpha: a private `control.dat` plus links |
| `femtic_lcurve_prep_summary.md` | User-set (UPPERCASE) parameters, script path, run date/time, written to `ENSEMBLE_DIR` |

A table (run number, alpha, old -> new value, directory) is printed at the end.

---

## How control.dat is edited

The rules follow the FEMTIC-DABIC v2.7.1 `control.dat`:

- keywords and values are whitespace-separated tokens; a value may be on the
  same line as its keyword or on a later line;
- a comment is a whole line starting with `#`; reading stops at `END`;
- each keyword may appear once.

Consequences for the editor:

- comment lines that mention `TRADE_OFF_PARAMETER` are never touched, and the
  different keyword `TYPE_OF_TRADE_OFF_PARAMETER` is not matched (exact token
  match, no substring search);
- only the **first value token** after the keyword is replaced. Further tokens
  of that value (older layouts such as `100.0 1.0`) and every other byte of the
  file, line endings included, are kept;
- the keyword is searched in `TRADE_OFF_KEYS`, first hit wins:
  `TRADE_OFF_PARAMETER` (v2.7.1) or `TRADE_OFF_PARAM` (older templates);
- no keyword, a keyword twice, no value or a non-numeric value: the script
  stops **before** any directory is created.

Safety checks: `control.dat` must be in `COPY_LIST` and not in `LINK_LIST`
(a link would edit the template); a leftover symlink in a run directory is
removed before copying; the copy is made user-writable (a read-only template
gives read-only copies); a checksum of the template is compared before and
after the run.

A warning is printed if the template has `INV_METHOD` other than 1 or
`TYPE_OF_TRADE_OFF_PARAMETER` other than 0: separate runs for an L-curve
normally use a fixed alpha.

---

## Configuration variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENSEMBLE_DIR` | `str` | `.../misti2026/l-curve/` | Directory with `templates/`; run directories are created here |
| `ENSEMBLE_NAME` | `str` | `"lc_"` | Prefix of the run directories; keep `"lc_"` for `femtic_lcurve_plot.py` |
| `TEMPLATES` | `str` | `ENSEMBLE_DIR + "templates/"` | Template directory |
| `TRADE_OFF_LIST` | `list` of `float` | `[0.01, 0.03, ..., 100.]` | Alpha values, one run each. Finite and > 0; values giving the same directory name (6 significant digits) are refused |
| `TRADE_OFF_KEYS` | `list` of `str` | `["TRADE_OFF_PARAMETER", "TRADE_OFF_PARAM"]` | Keyword(s) of the trade-off parameter; first one found is used |
| `CONTROL_FILE` | `str` | `"control.dat"` | The edited file |
| `COPY_LIST` | `list` of `str` | `[CONTROL_FILE]` | Files copied into every run directory |
| `LINK_LIST` | `list` of `str` | mesh, observe, starting model, distortion, reference model, site, run scripts | Files symlinked into every run directory |
| `RELATIVE_LINKS` | `bool` | `True` | Relative symlinks (portable, survive tgz); `False` = absolute |
| `SKIP_FINISHED` | `bool` | `True` | Leave a run directory alone if it already holds a convergence file, so extending `TRADE_OFF_LIST` never rewrites finished runs |
| `CNV_FILE` | `str` or `list` | `["femtic.cnv", "FEMTIC-DABIC.cnv"]` | Convergence file name(s) used for `SKIP_FINISHED` |

The value is written with up to 10 significant digits (`0.5`, `10`, `1e-05`);
the directory name uses 6 (`lc_0.5`, `lc_10`, `lc_1e-05`).

---

## Dependencies

| Module | Source | Purpose |
|--------|--------|---------|
| `numpy` | PyPI | Value list handling |
| `femtic` | Py4MTX | Imported first (it and `ensembles` import each other); not called |
| `ensembles` | Py4MTX | `put_files` (copy / link) |
| `util` | Py4MTX | `print_title`, `write_param_summary` |
| `version` | Py4MTX | Version string for the title banner |

Needs the environment variable `PY4MTX_ROOT`.

---

## Provenance

| Date | Author | Description |
|------|--------|-------------|
| 2026-11-17 | Claude Sonnet 5.5 (Anthropic) | Created. AI-generated; review before production use. |
| 2026-11-17 | Claude Sonnet 5.5 (Anthropic) | Renamed from `femtic_lcurve.py` to `lcurve_prep.py`; no code change. |
| 2026-11-17 | Claude Sonnet 5.5 (Anthropic) | Renamed again, `lcurve_prep.py` to `femtic_lcurve_prep.py` (same naming as `femtic_lcurve_plot.py`); no code change. |
| 2026-11-17 | Claude Sonnet 5.5 (Anthropic) | Readme only: workflow table extended by `femtic_lcurve_post.py`; no code change. |
