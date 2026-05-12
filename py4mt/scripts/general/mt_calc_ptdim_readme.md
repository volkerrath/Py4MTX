# mt_calc_ptdim.py

Calculate phase-tensor dimensionality for each MT station.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_calc_ptdim.py` |
| Authors | sb & vr (Dec 2019) |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

Reads EDI files and uses the impedance tensor's phase-tensor analysis
to classify each frequency as 1-D, 2-D, or 3-D. Writes per-site
dimensionality tables (`<site>_dims.dat`), a combined table
(`All_dims.dat`), and a summary file (`Dimensions.dat`) with
per-site and overall statistics.

## Workflow

1. Scan `EDI_DIR` for `.edi` files.
2. For each EDI file, build an `MT` object, extract the impedance
   tensor, and call `estimate_dimensionality()`.
3. Write per-site and combined dimensionality tables.
4. Print and save a summary of 1-D / 2-D / 3-D counts.

## Issues fixed during cleanup

| Issue | Description |
|-------|-------------|
| **Wrong docstring** | Docstring said "produces a site list" (copy-paste). Replaced with correct description. |
| **Hard-coded `PY4MTX_DATA`** | Overwrote env-var value. Removed override. |
| **Unused imports** | `matplotlib`, `mpl`, `plt`, `Z` class imported but never used. Removed. |
| **Unused variable** | `cm = 1/2.54` defined but never used (commented-out figure). Removed. |
| **Fragile `sit` counter** | Manual `sit = -1` + increment replaced with `enumerate()`. |
| **Fragile `dims` init** | `sit == 0` check replaced with `dims is None` guard. |
| **Lambda counting** | `sum(map(lambda x: x == 1, dim))` replaced with `np.sum(dim == 1)`. |
| **Jupytext header** | Removed obsolete Jupytext cell metadata block. |
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE`. |
| **Threshold constants** | `skew_threshold` and `eccentricity_threshold` extracted as named constants `SKEW_THRESHOLD`, `ECCENTRICITY_THRESHOLD`. |

## Configuration

| Constant | Description |
|----------|-------------|
| `WORK_DIR` / `EDI_DIR` | Path to EDI files |
| `DIM_FILE` | Output summary file |
| `SKEW_THRESHOLD` | Skew angle threshold (degrees) for 3-D classification |
| `ECCENTRICITY_THRESHOLD` | Phase ellipse eccentricity threshold for 2-D vs 1-D |

## Dependencies

`numpy`, `mtpy`, py4mt: `data_proc`, `util`, `version`.
