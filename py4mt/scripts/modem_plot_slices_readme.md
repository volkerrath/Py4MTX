# modem_plot_slices.py

Plot cross-sections through a 3-D ModEM resistivity model.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_plot_slices.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| Status | **Work-in-progress stub** — model loading works; slice plotting is not yet implemented. |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads a (possibly anisotropic) ModEM model and prepares horizontal
and vertical slices for visualisation.

## Planned functionality

- Compute cell centres from `dx`, `dy`, `dz` arrays.
- Build a `RegularGridInterpolator` for arbitrary slice planes.
- Plot horizontal slices at specified depths.
- Plot vertical slices along specified profiles.

## Inputs

| Item | Description |
|------|-------------|
| `ModFile` | List of ModEM model file paths (supports anisotropic multi-component models). |
| `Components` | Number of resistivity components (1 = isotropic, 3 = anisotropic). |

## Outputs

None yet. Planned: PDF/PNG slice plots.

## Configuration

- `WorkDir` — working directory.
- `ModFile` — model file path(s).
- `Components` — 1 or 3.
- `UseAniso` — flag for anisotropic reading.
- `PlotFile` — output plot base name (not yet used).

## Dependencies

`numpy`, `scipy.interpolate`, py4mt: `modem`, `util`, `version`.
