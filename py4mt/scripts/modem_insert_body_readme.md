# modem_insert_body.py

Insert geometric anomaly bodies into a ModEM resistivity model.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_insert_body.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

Reads a ModEM `.rho` model file, inserts one or more synthetic bodies
(ellipsoids or boxes), applies optional spatial smoothing, and writes
the perturbed model. Useful for resolution testing and synthetic studies.

## Body types

- **Ellipsoid** (`'ell'`): defined by centre, semi-axes, and rotation angles.
- **Box** (`'box'`): defined by centre, half-widths, and rotation angles.

## Insertion modes

| `ACTION[0]` | Behaviour |
|--------------|-----------|
| `'rep'` | Replace cell values with the body resistivity (conditional or unconditional). |
| `'add'` | Add the body resistivity to existing values; bodies are applied sequentially. |

A `CONDITION` string (e.g. `'val <= np.log(1.)'`) can restrict replacement
to cells meeting the criterion.

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`RHOAIR`, `MOD_FILE_IN`, `MOD_FILE_OUT`, `ACTION`, `CONDITION`, `ELL`, `BODIES`, `ADDITIVE`, `SMOOTHER`). |
| **Unused imports** | Removed `jac_proc` (not used in this script). |
| **Unused variables** | Removed `rng`, `nan` (never used). |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `MOD_FILE_IN` / `MOD_FILE_OUT` | Input/output model paths (without `.rho`) |
| `BODIES` | List of body definitions |
| `ACTION` | Insertion mode (`'rep'` or `'add'`) and value |
| `CONDITION` | Optional condition string for conditional replacement |
| `SMOOTHER` | Smoothing type and parameter |

## Dependencies

`numpy`, py4mt: `modem`, `util`, `version`.
