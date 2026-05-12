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
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

Reads a (possibly anisotropic) ModEM model and prepares horizontal
and vertical slices for visualisation.

## Planned functionality

- Compute cell centres from `dx`, `dy`, `dz` arrays.
- Build a `RegularGridInterpolator` for arbitrary slice planes.
- Plot horizontal slices at specified depths.
- Plot vertical slices along specified profiles.

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`RHOAIR`, `COMPONENTS`, `USE_ANISO`, `WORK_DIR`, `MOD_FILE`, `PLOT_FILE`). |
| **Unused variables** | Removed `rng = np.random.default_rng()` and `nan = np.nan` (never used). |
| **Hard-coded paths** | Replaced hard-coded `mypath` with `PY4MTX_ROOT` env var (consistent with other scripts). |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `WORK_DIR` | Working directory |
| `MOD_FILE` | Model file path(s) |
| `COMPONENTS` | Number of resistivity components (1 = isotropic, 3 = anisotropic) |
| `USE_ANISO` | Flag for anisotropic reading |
| `PLOT_FILE` | Output plot base name (not yet used) |

## Dependencies

`numpy`, `scipy.interpolate`, py4mt: `modem`, `util`, `version`.
