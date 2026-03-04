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
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

**Merge:** Combines separate Jacobian files (e.g. impedance Z, tipper T,
phase tensor PT) into a single file for joint analysis.

**Split:** Decomposes a merged Jacobian into subsets by one or more
criteria: frequency band, transfer-function component, data type, or site.

## Merge mode (`TASK = 'merge'`)

Reads N sparse Jacobian + info file pairs and vertically stacks them:

    J_merged = [J_Z; J_P; J_T]

Saves a single `_jac.npz` + `_info.npz` pair.

## Split mode (`TASK = 'split'`)

Reads one merged Jacobian and writes subsets based on the `SPLIT` string:

| Keyword | Splits by |
|---------|-----------|
| `'freq'` | Period / frequency bands defined by `PER_INTERVALS`. |
| `'comp'` | Transfer-function component (ZXY, ZYX, TXR, PTXX, …). |
| `'dtype'` | Data type code (1=Full_Z, 2=Off_Diag_Z, 3=Tipper, 6=PT, …). |
| `'site'` | Individual measurement site. |

Multiple keywords can be combined: `SPLIT = 'dtype site freq comp'`.

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`WORK_DIR`, `TASK`, `MERGED_FILE`, `M_FILES`, `S_FILE`, `SPLIT`, `PER_INTERVALS`). |
| **Shadow variable** | Renamed local `nF` in split-by-freq to `nBands` to avoid shadowing the outer `nF`. |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `TASK` | `'merge'` or `'split'` |
| `M_FILES` | List of file base names to merge |
| `MERGED_FILE` | Output base name for the merged file |
| `S_FILE` | Input base name for splitting |
| `SPLIT` | Space-separated list of split criteria |
| `PER_INTERVALS` | Frequency/period bands for frequency splitting |

## Dependencies

`numpy`, `scipy.sparse`, py4mt: `modem`, `jac_proc`, `util`, `version`.
