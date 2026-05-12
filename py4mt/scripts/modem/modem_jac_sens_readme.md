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
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

Reads a processed Jacobian and computes sensitivity for the full data
set and for subsets split by data type, transfer-function component,
measurement site, and/or frequency band. Sensitivities can be normalised
by cell volume and/or maximum value. Output is written as 3-D model
files in one or more formats.

## Sensitivity types

| `TYPE` | Formula | Description |
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

| `SPLITS` keyword | Produces one sensitivity file per… |
|-------------------|------------------------------------|
| `'total'` | Entire data set (single file). |
| `'dtyp'` | Data type (Full_Z, Off_Diag_Z, Tipper, PT). |
| `'comp'` | Component (ZXY, ZYX, TXR, PTXX, …). |
| `'site'` | Measurement site. |
| `'freq'` | Frequency band (defined by `PER_INTERVALS`). |

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`BLANK`, `RHOAIR`, `INP_FORMAT`, `OUT_FORMAT`, `MOD_EXT`, `WORK_DIR`, `J_NAME`, `J_FILE`, `M_FILE`, `M_ORIG`, `SIZ_EXTRACT`, `TOPO_EXTRACT`, `SPLITS`, `NO_REIM`, `NORM_LOCAL`, `PER_INTERVALS`, `TYPE`, `TRANSFORM`, `SENS_DIR`). |
| **Unused variables** | Removed `rng` (never used). |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `TYPE` | Sensitivity type (`'raw'`, `'cov'`, `'euc'`) |
| `TRANSFORM` | Space-separated transform chain (e.g. `'siz vol max'`) |
| `SPLITS` | Space-separated split list |
| `NORM_LOCAL` | If `True`, each subset normalised independently |
| `OUT_FORMAT` | `'mod'`, `'ubc'`, `'rlm'`, or combinations |

## Dependencies

`numpy`, `scipy.sparse`, py4mt: `modem`, `jac_proc`, `util`, `version`.
