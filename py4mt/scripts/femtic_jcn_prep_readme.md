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

## Dependencies

`numpy`; py4mt modules: `femtic`, `util`, `version`.
