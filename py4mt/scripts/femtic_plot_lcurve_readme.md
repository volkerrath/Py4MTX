# femtic_plot_lcurve.py

Plot the L-curve for FEMTIC regularisation parameter selection.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_plot_lcurve.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Provenance

| Date       | Author | Change                                       |
|------------|--------|----------------------------------------------|
| 2025       | vrath  | Created.                                     |
| 2026-03-03 | Claude | Renamed user-set parameters to UPPERCASE.    |

## Purpose

Collects the final-iteration roughness and misfit (or nRMS) from
multiple FEMTIC inversion runs performed at different regularisation
parameters (alpha). Plots roughness vs. misfit as an L-curve, with
each point annotated by its alpha value, to help identify the
optimal trade-off.

## Inputs

| Item | Description |
|------|-------------|
| `WORK_DIR` | Directory containing one sub-directory per alpha value. |
| `SEARCH_STRNG` | Glob pattern to find sub-directories. |

Each sub-directory must contain `femtic.cnv`. The last line of
the file provides the final-iteration alpha, roughness, misfit, and nRMS.

## Outputs

| File | Contents |
|------|----------|
| `<PLOT_NAME>.pdf` | L-curve plot (PDF). |
| `<PLOT_NAME>.png` | L-curve plot (PNG). |

## Configuration

- `WORK_DIR` — root directory.
- `PLOT_NAME` — base filename for the output plots.
- `PLOT_WHAT` — `'nrms'` (plot nRMS on x-axis) or `'misfit'` (plot raw misfit).

## Dependencies

`numpy`, `matplotlib`, py4mt: `femtic`, `util`, `version`.
