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

## Purpose

Collects the final-iteration roughness and misfit (or nRMS) from
multiple FEMTIC inversion runs performed at different regularisation
parameters (alpha). Plots roughness vs. misfit as an L-curve, with
each point annotated by its alpha value, to help identify the
optimal trade-off.

## Inputs

| Item | Description |
|------|-------------|
| `WorkDir` | Directory containing one sub-directory per alpha value. |
| `SearchStrng` | Glob pattern to find sub-directories. |

Each sub-directory must contain `femtic.cnv`. The last line of
the file provides the final-iteration alpha, roughness, misfit, and nRMS.

## Outputs

| File | Contents |
|------|----------|
| `<PlotName>.pdf` | L-curve plot (PDF). |
| `<PlotName>.png` | L-curve plot (PNG). |

## Configuration

- `WorkDir` — root directory.
- `PlotName` — base filename for the output plots.
- `PlotWhat` — `'nrms'` (plot nRMS on x-axis) or `'misfit'` (plot raw misfit).

## Dependencies

`numpy`, `matplotlib`, py4mt: `femtic`, `util`, `version`.
