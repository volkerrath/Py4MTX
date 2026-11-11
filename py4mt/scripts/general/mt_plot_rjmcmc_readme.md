# mt_plot_rjmcmc.py

Visualise transdimensional (rjmcmc-MT) 1-D inversion results.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_plot_rjmcmc.py` |
| Authors | R. Hassan (GA, 2017), V. Rath (2019–2024) |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

Plots posterior resistivity-depth probability distributions produced
by the rjmcmc-MT transdimensional sampler. For each result file, the
script generates a summary plot (via `plotrjmcmc.Results`) and
optionally augments the output with site coordinates from the
corresponding EDI file. A combined all-stations `.dat` file and
a PDF catalogue can also be produced.

## Workflow

1. Scan `RES_DIR` for result files and `EDI_DIR` for matching EDI files.
2. For each result file:
   a. Plot the posterior distribution (resistivity vs. depth).
   b. Read station coordinates from the corresponding EDI.
   c. Prepend lat/lon columns to the `.dat` output.
3. Assemble a combined data file with all stations.
4. Optionally create a PDF catalogue.

## Issues fixed during earlier cleanup (2 Mar 2026)

| Issue | Description |
|-------|-------------|
| **Unused imports** | `datetime`, `mt_metadata.TF_XML` removed. |
| **Variable name** | `myfilename` renamed to `mypath`. |
| **Hard-coded `PY4MTX_DATA`** | Removed override. |
| **Fragile `count==1` check** | Replaced with `data_all is None`. |

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`WORK_DIR`, `EDI_DIR`, `RES_DIR`, `PLT_DIR`, `PLOT_FMT`, `RHO_PLOT_LIM`, `DEPTH_PLOT_LIM`, `LOG_DEPTH`, `COLOR_MAP`, `PDF_CATALOG`, `PDF_CATALOG_NAME`, `OUT_STRNG`, `DATA_OUT`, `DATA_NAME`, `W_REF`). |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `WORK_DIR`, `EDI_DIR`, `RES_DIR`, `PLT_DIR` | Directory paths |
| `PLOT_FMT` | Output format (`.pdf`, `.png`) |
| `RHO_PLOT_LIM` | Resistivity axis limits |
| `DEPTH_PLOT_LIM` | Maximum depth (m) |
| `DATA_OUT` | Write augmented `.dat` files and combined file |
| `W_REF` | If `True`, convert depth to elevation (elev − depth) |
| `PDF_CATALOG` | Assemble a PDF catalogue |
| `FIGURE_TITLE` | Figure title (2026-11-11), passed to `plotrjmcmc.Results(figureTitle=...)`. `"auto"` = `MT Station <station> - 1D rj-McMC Inversion` as before; `None`, `False`, `""` = none; text with `{station}`; dict `{"<station>": spec, "default": spec}`. Needs the `plotrjmcmc.py` of the same date. |

## Dependencies

`numpy`, py4mt: `data_proc` (`load_edi`), `plotrjmcmc`, `util`, `version`.

## Changes 2026-11-11 (titles)

`FIGURE_TITLE` (table above). The position of the colour bar cannot be chosen in `plotrjmcmc` (hand-placed axes); see `plotrjmcmc_readme.md`. Claude Sonnet 5.5 (Anthropic): AI-generated; review before use. Defaults give the figures of before. Only compile-checked and tested at the level of the title logic (`util.resolve_title`, `test_util_titles.py`); the script was not run end to end.
