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
| NRMS option added | 30 July 2026 by Claude (Anthropic) |

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

## NRMS / SRMS option (added 30 Jul 2026)

If `COMPUTE_NRMS = True`, the script additionally computes total and
subgroup NRMS/SRMS misfit statistics by comparing each observed data
file against a matching calculated (forward-response) data file, using
`inverse.calc_rms` (real and imaginary parts of each datum are treated
as independent real-valued observations, normalised by the datum's
error).

### Configuration

| Constant | Description |
|----------|--------------|
| `COMPUTE_NRMS` | Toggle NRMS/SRMS computation on/off |
| `CALC_FILES_IN` | Dict mapping each observed file to its calculated/predicted counterpart |
| `NRMS_GROUP_BY` | Subgroup keys to report, any of `"datatype"`, `"component"`, `"site"`, `"band"` |
| `NRMS_SITE_LIST` | Optional explicit list of site codes; if non-empty, only these sites are included |
| `NRMS_COMP_LIST` | Optional explicit list of component codes (e.g. `"ZXY"`); if non-empty, only these are included |
| `NRMS_FREQ_LIST` | Optional explicit list of frequencies (Hz); if non-empty, only matching periods are included |
| `NRMS_FREQ_RTOL` | Relative tolerance used when matching `NRMS_FREQ_LIST` against data periods |
| `NRMS_OUT_FILE` | Path of the written text summary |

### Workflow

1. Parse each observed file and its matching calculated file with
   `parse_modem_dat()`.
2. If `NRMS_SITE_LIST`, `NRMS_COMP_LIST`, or `NRMS_FREQ_LIST` are
   non-empty, discard any observed record not in the corresponding
   list before matching (frequencies are compared as `1/period` within
   `NRMS_FREQ_RTOL`).
3. Match remaining records by `(period, site, component)`.
4. Build normalised residual arrays (`Wd = 1/error`) and call
   `inverse.calc_rms(dcalc, dobs, Wd)` for the total, then again per
   subgroup value.
5. Print and write a text summary (`modem_nrms_summary.txt`).

Note: the subset filters and the `NRMS_GROUP_BY` breakdown compose —
e.g. setting `NRMS_COMP_LIST = ["ZXY", "ZYX"]` and
`NRMS_GROUP_BY = ["site"]` reports per-site NRMS using only those two
components.

### Flagged assumptions (unverified)

- **Column layout**: observed and calculated files are assumed to follow
  the standard ModEM layout
  `period code lat lon x y z component real imag error` (11
  whitespace-separated columns). This has not been checked against every
  ModEM data type (e.g. some Tipper/PT variants may differ).
- **Calculated-file naming**: `CALC_FILES_IN` defaults to replacing
  `"_in.dat"` with `"_calc.dat"` for each entry in `DAT_FILES_IN`. ModEM
  itself has no fixed naming convention for forward-response files
  (commonly something like `*_NLCG_050.dat`); **this mapping must be
  edited to match the actual output filenames** before relying on the
  results.
- **Error convention**: the single error column is assumed to apply
  identically to both the real and imaginary parts of each datum.

## Dependencies

`numpy`, py4mt: `util`, `version`, `inverse` (for `calc_rms`).
