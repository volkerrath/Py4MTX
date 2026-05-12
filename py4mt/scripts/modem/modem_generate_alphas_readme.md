# modem_generate_alphas.py

Generate depth-dependent smoothing alpha parameters for ModEM inversion.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_generate_alphas.py` |
| Author | vrath (Nov 2024) |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 4 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads a ModEM model to obtain the vertical mesh discretisation, then
computes alpha_x and alpha_y smoothing parameters that vary linearly
with depth between user-specified upper and lower bounds. These
parameters control the spatial smoothness regularisation in ModEM
inversions.

## Workflow

1. Read the model file to obtain `dz` (vertical cell thicknesses).
2. Compute depth from `dz` via `mod.set_mesh`.
3. Call `mod.generate_alphas` with begin/end linear bounds.
4. Print the resulting alpha arrays.

## Issues fixed during cleanup (4 Mar 2026)

| Issue | Description |
|-------|-------------|
| **Missing import** | `inspect` was used but not imported. Added. |
| **Hard-coded path override** | `ModDir_in` was overwritten with a second hard-coded path. Removed override; uses `PY4MTX_DATA`-based path. |
| **Commented-out variables** | `rng`, `nan`, `blank`, `rhoair` were commented out but still present. Removed. |
| **Minimal docstring** | Only had `'Created on …'`. Replaced with proper description. |
| **Single-quoted strings** | Normalised to double quotes for consistency with other scripts. |
| **Unused variable** | `total = 0` defined but never used. Removed. |
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`MOD_DIR_IN`, `MOD_DIR_OUT`, `MOD_FILE_IN`, `MOD_FILE_OUT`, `BEG_LIN`, `END_LIN`). |

## Configuration

| Constant | Description |
|----------|-------------|
| `MOD_DIR_IN` / `MOD_DIR_OUT` | Input/output directories |
| `MOD_FILE_IN` / `MOD_FILE_OUT` | Model file path (without `.rho`) |
| `BEG_LIN` | `[depth, alpha_x, alpha_y]` at shallowest bound |
| `END_LIN` | `[depth, alpha_x, alpha_y]` at deepest bound |

## Dependencies

`numpy`, py4mt: `modem`, `util`, `version`.
