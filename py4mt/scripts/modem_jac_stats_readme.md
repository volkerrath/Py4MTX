# modem_jac_stats.py

Compute and print statistics on a ModEM Jacobian matrix.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_jac_stats.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads a processed Jacobian (sparse or raw) and computes summary
statistics (via `jac.print_stats`) for the full matrix and for
subsets split by data-type component, measurement site, and/or
frequency band. Results are printed to the console and written to
a text file.

## Splits available

| Keyword | Description |
|---------|-------------|
| `'comp'` | By data-type code (Full_Z, Off_Diag_Z, Tipper, PT, …). |
| `'site'` | By individual measurement site name. |
| `'freq'` | By period/frequency band defined in `PerIntervals`. |

## Inputs

| Item | Description |
|------|-------------|
| `MFile` | ModEM model file path (for reading dimensions and air mask). |
| `JFile` | Jacobian file base name (`_jac.npz` + `_info.npz`, or `.jac` + `_jac.dat`). |

## Outputs

| File | Contents |
|------|----------|
| `<JFile>_stats.dat` | Text file with statistics for each subset. |

## Configuration

- `InpFormat` — `'sparse'` (reads `_jac.npz`) or raw (reads `.jac`).
- `Splits` — list of split criteria, e.g. `['comp', 'site', 'freq']`.
- `PerIntervals` — frequency/period bands for the `'freq'` split.

## Dependencies

`numpy`, `scipy.sparse`, py4mt: `modem`, `jacproc`, `util`, `version`.
