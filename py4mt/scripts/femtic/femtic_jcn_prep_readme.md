# femtic_jcn_prep.py

Prepare jackknife uncertainty analysis directories for FEMTIC.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_jcn_prep.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 3 March 2026 by Claude (Anthropic), from cleaned source |
| README updated | 25 July 2026 by Claude Sonnet 5 (Anthropic) — added `RANDOM_SEED`; flagged a known issue (see below) |
| README updated | 13 August 2026 by Claude Sonnet 5 (Anthropic) — added `femtic_jcn_prep_summary.md` output at end of run |

## ⚠ Known issue (unrelated to the reproducibility update)

`fem.generate_directories()` and `fem.generate_data_fcn()`, called near the
bottom of this script, **do not exist** in the current `femtic.py` /
`ensembles.py`. This script predates the consolidation of directory- and
data-ensemble generation into `ensembles.py` — compare `femtic_rto_prep.py`
/ `femtic_gst_prep.py`, which call `ens.generate_directories()` /
`ens.generate_data_ensemble()` instead. As written, both calls will raise
`AttributeError` before the RNG below is ever consumed. `RANDOM_SEED` /
`rng` have been wired through anyway so the script is consistent with the
rest of the project once the calls are migrated, or a dedicated
jackknife/leave-one-site-out generator is added to `ensembles.py` — that
migration has not been done here since it needs a design decision (a new
`ensembles.py` function, or restoring the old `femtic.py` functions) rather
than a mechanical fix.

## Purpose

Sets up the directory structure and reduced data files needed for a
jackknife-style uncertainty analysis.  For each jackknife sample a
complete inversion directory is created from template files, and a
reduced data set (e.g. leave-one-site-out) is generated.

## Workflow

1. Reads `control.dat` to determine the number of sites / sample count.
2. Creates *N* sub-directories (`jcn_0`, `jcn_1`, …) by copying template files.
3. Generates reduced `observe.dat` files with one site (or subset) removed.

## Configuration constants

| Constant | Description |
|----------|-------------|
| `ENSEMBLE_DIR` | Base directory for jackknife runs. |
| `TEMPLATES` | Path to the template directory. |
| `FILES` | List of template file names (control.dat, observe.dat, mesh.dat, …). |
| `CHOICE_MODE` | `["site"]` for leave-one-site-out, or `["subset", N]` for random subsets. |
| `N_SAMPLES` | Number of jackknife samples (read from `control.dat` or set manually). |
| `RANDOM_SEED` | `None` (default) = fresh OS entropy. An integer makes the run reproducible — relevant only when `CHOICE_MODE = ["subset", N]` (random selection); `["site"]` leave-one-out is deterministic and doesn't consume any random draws. |

## Inputs

| Item | Description |
|------|-------------|
| `TEMPLATES/` directory | Contains the template files to copy into each run directory. |
| `control.dat` | First line provides the site count used for jackknife sampling. |

## Outputs

| Item | Description |
|------|-------------|
| `jcn_<N>/` | One inversion directory per jackknife sample, ready to run FEMTIC. |
| Modified `observe.dat` | Each directory contains a reduced data file with one site removed. |
| `femtic_jcn_prep_summary.md` | Markdown summary of user-set (UPPERCASE) parameters, script path, and run date/time. |

## Dependencies

`numpy`; py4mt modules: `femtic`, `util`, `version`.
