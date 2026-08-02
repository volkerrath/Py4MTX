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
| Feature added | 2 Aug 2026 by Claude (Anthropic): `outside` parameter for `insert_body` / `insert_body_condition` |

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

## Masking the exterior (`outside`)

`modem.insert_body` and `modem.insert_body_condition` (in `modem.py`) now
accept an `outside` keyword, exposed in this driver script as the `OUTSIDE`
config constant:

```python
# Default: exterior cells keep their original rho_in values (previous behaviour)
rho_out = mod.insert_body_condition(
    dx, dy, dz, rho_in, body,
    smooth=SMOOTHER, reference=refmod,
    outside=None,
)

# Replace everything outside the body with a fixed resistivity
rho_out = mod.insert_body_condition(
    dx, dy, dz, rho_in, body,
    smooth=SMOOTHER, reference=refmod,
    outside=100.0,   # Ohm.m
)

# Blank everything outside the body with NaN
rho_out = mod.insert_body_condition(
    dx, dy, dz, rho_in, body,
    smooth=None, reference=refmod,   # see note below on smoothing + NaN
    outside=np.nan,
)
```

| `outside` | Behaviour |
|-----------|-----------|
| `None` (default) | Exterior cells keep their original `rho_in` values — unchanged from previous behaviour. |
| a finite resistivity value (Ohm.m) | Every cell whose centre falls outside the body geometry is set to this value. |
| `np.nan` | Every cell outside the body is set to `NaN` — e.g. to produce an isolated-body mask/template for downstream processing that treats NaN as "no data". |

Notes:
- "Outside" is determined from the same per-cell geometric test used for
  insertion (`in_ellipsoid` / `in_box`), evaluated over the entire model
  grid — cells in the `pad` margin are also treated as exterior.
- The fill is applied **before** smoothing. Smoothing a NaN-filled exterior
  will propagate NaN into the body's edge cells; both functions print a
  warning if `smooth` is set together with `outside=np.nan`. Use
  `smooth=None` in that case.
- The fill value is applied in log-resistivity space, consistent with how
  the body value itself is applied; the returned array is converted back
  to linear resistivity as before.
- Available in `modem.py` for both `insert_body` and `insert_body_condition`,
  and wired into this driver script via `OUTSIDE`.

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`RHOAIR`, `MOD_FILE_IN`, `MOD_FILE_OUT`, `ACTION`, `CONDITION`, `ELL`, `BODIES`, `ADDITIVE`, `SMOOTHER`). |
| **Unused imports** | Removed `jac_proc` (not used in this script). |
| **Unused variables** | Removed `rng`, `nan` (never used). |
| **Provenance line** | Added cleanup date to docstring. |

## Changes 2 Aug 2026

| Change | Description |
|--------|-------------|
| **`outside` parameter** | `modem.insert_body` and `modem.insert_body_condition` gained an `outside=None` keyword to replace every cell outside the inserted body's geometry with a fixed resistivity value or `NaN`. See "Masking the exterior" above. Default `None` preserves prior behaviour exactly. |
| **`OUTSIDE` config constant** | Added to `modem_insert_body.py` and threaded into both the `insert_body_condition` (non-additive) and `insert_body` (additive) calls. |

## Configuration

| Constant | Description |
|----------|-------------|
| `MOD_FILE_IN` / `MOD_FILE_OUT` | Input/output model paths (without `.rho`) |
| `BODIES` | List of body definitions |
| `ACTION` | Insertion mode (`'rep'` or `'add'`) and value |
| `CONDITION` | Optional condition string for conditional replacement |
| `SMOOTHER` | Smoothing type and parameter |
| `OUTSIDE` | `None` (default, exterior unchanged), a resistivity value (Ohm.m), or `np.nan`; replaces every cell outside the body geometry |

## Dependencies

`numpy`, py4mt: `modem`, `util`, `version`.
