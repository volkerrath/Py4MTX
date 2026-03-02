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

## Purpose

Reads `femtic.cnv` convergence files from one or more inversion
directories and generates a convergence plot (PDF) showing how the
chosen quantity evolves with iteration number.

## Plot options

| `PlotWhat` | Y-axis | Scale |
|------------|--------|-------|
| `'misfit'` | Weighted data misfit ‖C_d^{-1/2}(d_obs − d_calc)‖₂ | log |
| `'rms'` | Normalised RMS | linear |
| `'rough'` | Model roughness ‖C_m^{-1/2} m‖₂ | log |

## Inputs

| Item | Description |
|------|-------------|
| `WorkDir` | Directory containing inversion sub-directories. |
| `SearchStrng` | Glob pattern to find sub-directories (e.g. `kra*`). |

Each sub-directory must contain `femtic.cnv` with columns:
iteration, retry, alpha, …, roughness, …, misfit, nRMS.

## Outputs

One PDF per inversion directory:
`<WorkDir>/<PlotName>_<PlotWhat>_alpha<value>.pdf`

## Configuration

- `WorkDir` — root directory.
- `PlotName` — base name for the plot title and filename.
- `PlotWhat` — `'misfit'`, `'rms'`, or `'rough'`.
- `SearchStrng` — glob pattern.

## Dependencies

`numpy`, `matplotlib`, py4mt: `femtic`, `util`, `version`.
