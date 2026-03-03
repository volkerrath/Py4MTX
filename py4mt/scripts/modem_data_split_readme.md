# modem_data_split.py

Split ModEM data files into period-band subsets.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_data_split.py` |
| Author | vrath (Feb 2021 / May 2024) |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 3 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads ModEM-format data files and splits each into separate files by
period band.  This is useful for band-by-band inversions, data quality
inspection, or frequency-dependent sensitivity studies.

## Workflow

1. For each input data file and each period interval:
   - Parse header lines (lines starting with `#` or `>`).
   - Select data lines whose period falls within the current interval.
   - Count unique periods and sites in the selection.
2. If data exist for the band, rewrite the header with updated counts and
   write the subset to a new file named `*_perband<N>.dat`.

## Configuration constants

| Constant | Description |
|----------|-------------|
| `DAT_DIR_IN` | Input directory containing ModEM data files. |
| `DAT_DIR_OUT` | Output directory (defaults to same as input). |
| `DAT_FILES_IN` | List of input data file names (impedance, phase tensor, tipper, …). |
| `PER_INTERVALS` | List of `[low, high]` period bounds (seconds) defining each band. |
| `PER_NUM_MIN` | Minimum number of periods required (reserved for future filtering). |
| `NUM_BANDS` | Number of period bands (derived from `PER_INTERVALS`). |

## Inputs

| Item | Description |
|------|-------------|
| `FOG_Z_in.dat` | Full-band impedance data (ModEM format). |
| `FOG_P_in.dat` | Full-band phase-tensor data. |
| `FOG_T_in.dat` | Full-band tipper data. |

## Outputs

| Item | Description |
|------|-------------|
| `*_perband<N>.dat` | One file per input × period band, containing only the data within that band. |

## Notes

- The last entry in the original `PerIntervals` list was not nested as a
  two-element list.  The cleaned version fixes this to
  `[10000.0, 1000000.0]` for consistency.
- The unused `jax` import from the original script has been removed.

## Dependencies

`numpy`; py4mt modules: `util`, `version`.
