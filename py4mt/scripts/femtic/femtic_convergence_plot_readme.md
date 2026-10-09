# femtic_plot_convergence.py

Plot iteration-by-iteration convergence curves from FEMTIC inversions.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_plot_convergence.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Provenance

| Date       | Author | Change                                       |
|------------|--------|----------------------------------------------|
| 2025       | vrath  | Created.                                     |
| 2026-03-03 | Claude | Renamed user-set parameters to UPPERCASE.    |
| 2026-08-13 | Claude Sonnet 5 | Added `femtic_convergence_plot_summary.md` output at end of run: user-set (UPPERCASE) parameters, script path, and run date/time. |
| 2026-10-08 | Claude Sonnet 5.5 | Convergence file name configurable via `CNV_FILE` (default `femtic.cnv`; list of candidates allowed); column positions now read from the header row via `fem.read_cnv()` instead of fixed indices. AI-generated; review before production use. |

## Purpose

Reads `femtic.cnv` convergence files from one or more inversion
directories and generates a convergence plot (PDF) showing how the
chosen quantity evolves with iteration number.

## Plot options

| `PLOT_WHAT` | Y-axis | Scale |
|------------|--------|-------|
| `'misfit'` | Weighted data misfit ‖C_d^{-1/2}(d_obs − d_calc)‖₂ | log |
| `'rms'` | Normalised RMS | linear |
| `'rough'` | Model roughness ‖C_m^{-1/2} m‖₂ | log |

## Inputs

| Item | Description |
|------|-------------|
| `WORK_DIR` | Directory containing inversion sub-directories. |
| `SEARCH_STRNG` | Glob pattern to find sub-directories (e.g. `kra*`). |
| `CNV_FILE` | Convergence file name in each sub-directory (default `"femtic.cnv"`; a list gives candidates, first existing wins). |

Each sub-directory must contain the convergence file (`CNV_FILE`, default `femtic.cnv`) with a header row naming (at least) Alpha, Roughness, Misfit and RMS; positions are read from the header via `fem.read_cnv()`.

## Outputs

One PDF per inversion directory:
`<WORK_DIR>/<PLOT_NAME>_<PLOT_WHAT>_alpha<value>.pdf`

## Configuration

- `WORK_DIR` — root directory.
- `PLOT_NAME` — base name for the plot title and filename.
- `PLOT_WHAT` — `'misfit'`, `'rms'`, or `'rough'`.
- `SEARCH_STRNG` — glob pattern.
- `CNV_FILE` — name of the convergence file (or list of candidates).

## Dependencies

`numpy`, `matplotlib`, py4mt: `femtic`, `util`, `version`.
