# modem_jac_sens.py

Compute sensitivity (coverage) maps from a ModEM Jacobian.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_jac_sens.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads a processed Jacobian and computes sensitivity for the full data
set and for subsets split by data type, transfer-function component,
measurement site, and/or frequency band. Sensitivities can be normalised
by cell volume and/or maximum value. Output is written as 3-D model
files in one or more formats.

## Sensitivity types

| `Type` | Formula | Description |
|--------|---------|-------------|
| `'raw'` | Σ_i J_ij | Sum of raw sensitivities. |
| `'abs'` / `'cov'` | Σ_i \|J_ij\| | Absolute-value sum (coverage). |
| `'euc'` | Σ_i J_ij² | Sum of squared sensitivities. |

## Transforms (applied in order)

| Keyword | Effect |
|---------|--------|
| `'vol'` / `'siz'` | Divide by cell volume (or area, horizontal/vertical size). |
| `'max'` | Normalise by maximum absolute value. |
| `'sur'` | Normalise by surface-layer value. |
| `'log'` | Take the logarithm (apply last). |

## Splits

| `Splits` keyword | Produces one sensitivity file per… |
|-------------------|------------------------------------|
| `'total'` | Entire data set (single file). |
| `'dtyp'` | Data type (Full_Z, Off_Diag_Z, Tipper, PT). |
| `'comp'` | Component (ZXY, ZYX, TXR, PTXX, …). |
| `'site'` | Measurement site. |
| `'freq'` | Frequency band (defined by `PerIntervals`). |

## Inputs

| Item | Description |
|------|-------------|
| `MFile` | ModEM model file (for mesh and air mask). |
| `JFile` | Jacobian base name (`_jac.npz` + `_info.npz`). |

## Outputs

Sensitivity files are written to `SensDir` in the configured formats
(ModEM `.rho`, UBC `.sns` + `.msh`, RLM `.rlm`).

Optionally also writes cell-size files (`.siz`) and topography (`.top`).

## Configuration

- `Type` — sensitivity type.
- `Transform` — space-separated transform chain (e.g. `'siz vol max'`).
- `Splits` — space-separated split list.
- `NormLocal` — if `True`, each subset is normalised independently;
  if `False`, all subsets share the total-sensitivity maximum.
- `OutFormat` — `'mod'`, `'ubc'`, `'rlm'`, or combinations.

## Dependencies

`numpy`, `scipy.sparse`, py4mt: `modem`, `jacproc`, `util`, `version`.
