# modem_jac_splitmerge.py

Merge or split processed ModEM Jacobian files.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_jac_splitmerge.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

**Merge:** Combines separate Jacobian files (e.g. impedance Z, tipper T,
phase tensor PT) into a single file for joint analysis.

**Split:** Decomposes a merged Jacobian into subsets by one or more
criteria: frequency band, transfer-function component, data type, or site.

## Merge mode (`Task = 'merge'`)

Reads N sparse Jacobian + info file pairs and vertically stacks them:

    J_merged = [J_Z; J_P; J_T]

Saves a single `_jac.npz` + `_info.npz` pair.

## Split mode (`Task = 'split'`)

Reads one merged Jacobian and writes subsets based on the `Split` string:

| Keyword | Splits by |
|---------|-----------|
| `'freq'` | Period / frequency bands defined by `PerIntervals`. |
| `'comp'` | Transfer-function component (ZXY, ZYX, TXR, PTXX, …). |
| `'dtype'` | Data type code (1=Full_Z, 2=Off_Diag_Z, 3=Tipper, 6=PT, …). |
| `'site'` | Individual measurement site. |

Multiple keywords can be combined: `Split = 'dtype site freq comp'`.

## Inputs / Outputs

All I/O uses the `_jac.npz` + `_info.npz` pair format produced by
`modem_jac_proc.py`.

## Configuration

- `Task` — `'merge'` or `'split'`.
- `MFiles` — list of file base names to merge.
- `MergedFile` — output base name for the merged file.
- `SFile` — input base name for splitting.
- `Split` — space-separated list of split criteria.
- `PerIntervals` — frequency/period bands for frequency splitting.

## Dependencies

`numpy`, `scipy.sparse`, py4mt: `modem`, `jacproc`, `util`, `version`.
