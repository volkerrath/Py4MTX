# femtic_jcn_prep.py

Prepare jackknife uncertainty analysis directories for FEMTIC.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_jcn_prep.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Sets up the directory structure and reduced data files needed for a
jackknife-style uncertainty analysis. For each jackknife sample, a
complete inversion directory is created from template files, and a
reduced data set (e.g. leave-one-site-out) is generated.

## Workflow

1. Reads `control.dat` to determine the number of sites / sample count.
2. Creates N sub-directories (`jcn_0`, `jcn_1`, …) by copying template files.
3. Generates reduced `observe.dat` files with one site (or subset) removed.

## Inputs

| Item | Description |
|------|-------------|
| `Templates` | Directory containing the template files to copy into each run directory. |
| `Files` | List of template file names (control.dat, observe.dat, mesh.dat, etc.). |
| `ChoiceMode` | `['site']` for leave-one-site-out, or `['subset', N]` for random subsets. |

## Outputs

| Item | Description |
|------|-------------|
| `jcn_<N>/` | One inversion directory per jackknife sample, ready to run FEMTIC. |
| Modified `observe.dat` | Each directory contains a reduced data file with one site removed. |

## Configuration

- `EnsembleDir` — base directory for jackknife runs.
- `Templates` — path to the template directory.
- `Files` — list of files to copy.
- `ChoiceMode` — `['site']` or `['subset', N_samples]`.

## Dependencies

`numpy`, py4mt modules: `femtic`, `util`, `version`.
