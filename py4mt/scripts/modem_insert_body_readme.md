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

## Purpose

Reads a ModEM `.rho` model file, inserts one or more synthetic bodies
(ellipsoids or boxes), applies optional spatial smoothing, and writes
the perturbed model. Useful for resolution testing and synthetic studies.

## Body types

- **Ellipsoid** (`'ell'`): defined by centre, semi-axes, and rotation angles.
- **Box** (`'box'`): defined by centre, half-widths, and rotation angles.

## Insertion modes

| `action[0]` | Behaviour |
|--------------|-----------|
| `'rep'` | Replace cell values with the body resistivity (conditional or unconditional). |
| `'add'` | Add the body resistivity to existing values; bodies are applied sequentially. |

A `condition` string (e.g. `'val <= np.log(1.)'`) can restrict replacement
to cells meeting the criterion.

## Inputs

| Item | Description |
|------|-------------|
| `ModFile_in` | Path to the input ModEM model (without `.rho` extension). |
| `bodies` | List of body definitions. |
| `smoother` | Smoothing type and parameter: `['gaussian', sigma]` or `['uniform', width]`. |

## Outputs

One `.rho` file per body (non-additive mode) or a single `_final.rho` (additive mode).

## Configuration

Edit the **Configuration** section: `ModFile_in`, `ModFile_out`, `bodies`,
`action`, `condition`, `smoother`, `additive`.

## Dependencies

`numpy`, py4mt: `modem`, `util`, `jac_proc`, `version`.
