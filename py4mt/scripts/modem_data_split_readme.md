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
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

Reads ModEM-format data files and splits each into separate files by
period band. This is useful for band-by-band inversions, data quality
inspection, or frequency-dependent sensitivity studies.

## Workflow

1. For each input data file and each period interval:
   - Parse header lines (lines starting with `#` or `>`).
   - Select data lines whose period falls within the current interval.
   - Count unique periods and sites in the selection.
2. If data exist for the band, rewrite the header with updated counts and
   write the subset to a new file named `*_perband<N>.dat`.

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **Unused imports** | Removed `time`, `datetime`, `warnings` (none used). |
| **Unused variables** | Removed `rng`, `nan` (never used). |
| **Provenance line** | Added cleanup date to docstring. |

Note: Configuration constants were already `UPPER_SNAKE_CASE` in the
previous cleanup pass.

## Configuration

| Constant | Description |
|----------|-------------|
| `DAT_DIR_IN` | Input directory containing ModEM data files |
| `DAT_DIR_OUT` | Output directory (defaults to same as input) |
| `DAT_FILES_IN` | List of input data file names |
| `PER_INTERVALS` | List of `[low, high]` period bounds (seconds) |
| `PER_NUM_MIN` | Minimum number of periods required (reserved for future filtering) |

## Dependencies

`numpy`, py4mt: `util`, `version`.
