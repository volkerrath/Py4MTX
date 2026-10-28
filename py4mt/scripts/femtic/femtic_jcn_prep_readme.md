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
| README updated | 11 October 2026 by Claude Sonnet 5.5 (Anthropic) — script migrated to `ensembles.py`; known issue resolved |
| README updated | 13 August 2026 by Claude Sonnet 5 (Anthropic) — added `femtic_jcn_prep_summary.md` output at end of run |
| README updated | 28 October 2026 by Claude Sonnet 5.5 (Anthropic) -- `femtic_jcn_prep_summary.md` is now written to `ENSEMBLE_DIR` instead of next to the script |

## Migration (11 October 2026)

The script used `fem.generate_directories()` and `fem.generate_data_fcn()`,
neither of which exists any more. It now calls `ens.generate_directories()`
(`ensembles.py`) and a local function `make_jackknife_data()`, built on
`fem.read_observe_dat` / `fem.write_observe_dat`. It was also fixed that
directories were created as `jcn_<i>` while the data call used `ens_<i>`
(both now use `DIR_BASE`). Tested on a synthetic four-site MT + VTF
`observe.dat` (both modes); not run on real data.

Things to be aware of:

- `observe.dat` is **copied** into each member (the original stays next to
  it as `observe_orig.dat`); `control.dat`, `mesh.dat`, the starting model
  and `distortion_iter0.dat` are symlinked.
- `distortion_iter0.dat` is **not** trimmed when sites are dropped.
- `"subset"` mode is a reinterpretation of an undocumented old option:
  each of `N_SAMPLES` members keeps `SUBSET_SIZE` random sites.

## Purpose

Sets up the directory structure and reduced data files needed for a
jackknife-style uncertainty analysis.  For each jackknife sample a
complete inversion directory is created from template files, and a
reduced data set (e.g. leave-one-site-out) is generated.

## Workflow

1. `"site"` mode: counts the unique site names in the template `observe.dat` (a differing count in line 1 of `control.dat` is reported, not fatal).
2. Creates *N* sub-directories (`jcn_0`, `jcn_1`, …); copies `COPY_FILES`, symlinks `LINK_FILES`.
3. Writes a reduced `observe.dat` into each (one site removed, or a random subset kept); block header counts are updated in every data block (MT, VTF, PT).

## Configuration constants

| Constant | Description |
|----------|-------------|
| `ENSEMBLE_DIR` | Base directory for jackknife runs. |
| `TEMPLATES` | Path to the template directory. |
| `FILES` | List of template file names (control.dat, observe.dat, mesh.dat, …). |
| `COPY_FILES` / `LINK_FILES` | Files copied into / symlinked in each member; `LINK_FILES` is `FILES` minus `COPY_FILES`. |
| `CHOICE_MODE` | `["site"]` for leave-one-site-out, or `["subset"]` for random subsets. |
| `N_SAMPLES` | Number of members. In `"site"` mode it is overwritten with the number of sites in `observe.dat`; in `"subset"` mode set it manually. |
| `SUBSET_SIZE` | `"subset"` mode only: number of sites kept per member. |
| `DIR_BASE` | Member directory prefix (`ENSEMBLE_DIR + "jcn_"`). |
| `RANDOM_SEED` | `None` (default) = fresh OS entropy. An integer makes the run reproducible — relevant only when `CHOICE_MODE = ["subset"]` (random selection); `["site"]` leave-one-out is deterministic and doesn't consume any random draws. |

## Inputs

| Item | Description |
|------|-------------|
| `TEMPLATES/` directory | Contains the template files to copy into each run directory. |
| `observe.dat` | Source of the site list and data (read and rewritten with `fem.read_observe_dat` / `fem.write_observe_dat`). |

## Outputs

| Item | Description |
|------|-------------|
| `jcn_<N>/` | One inversion directory per jackknife sample, ready to run FEMTIC. |
| Modified `observe.dat` | Each directory contains a reduced data file with one site removed. |
| `femtic_jcn_prep_summary.md` | Markdown summary of user-set (UPPERCASE) parameters, script path, and run date/time; written to `ENSEMBLE_DIR`. |

## Dependencies

`numpy`; py4mt modules: `femtic`, `ensembles`, `util`, `version`.
