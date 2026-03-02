# mt_plot_rjmcmc.py

Visualise transdimensional (rjmcmc-MT) 1-D inversion results.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_plot_rjmcmc.py` |
| Authors | R. Hassan (GA, 2017), V. Rath (2019–2024) |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Plots posterior resistivity-depth probability distributions produced
by the rjmcmc-MT transdimensional sampler. For each result file, the
script generates a summary plot (via `plotrjmcmc.Results`) and
optionally augments the output with site coordinates from the
corresponding EDI file. A combined all-stations `.dat` file and
a PDF catalogue can also be produced.

## Workflow

1. Scan `ResDir` for result files and `EdiDir` for matching EDI files.
2. For each result file:
   a. Plot the posterior distribution (resistivity vs. depth).
   b. Read station coordinates from the corresponding EDI.
   c. Prepend lat/lon columns to the `.dat` output.
3. Assemble a combined data file with all stations.
4. Optionally create a PDF catalogue.

## Issues fixed during cleanup

| Issue | Description |
|-------|-------------|
| **Unused imports** | `datetime`, `mt_metadata.TF_XML` removed. |
| **Variable name** | `myfilename` renamed to `mypath` (consistent with other scripts). |
| **Hard-coded PY4MTX_DATA** | Overwrote env-var value; removed override. |
| **Fragile count==1 check** | Replaced with `data_all is None` initialisation. |
| **Commented-out docstring** | `load_edi` signature in comments removed. |

## Configuration

- `WorkDir`, `EdiDir`, `ResDir`, `PltDir` — directory paths.
- `PlotFmt` — output format (`.pdf`, `.png`).
- `RhoPlotLim` — resistivity axis limits.
- `DepthPlotLim` — maximum depth (m).
- `DataOut` — write augmented `.dat` files and combined file.
- `WRef` — if `True`, convert depth to elevation (elev − depth).
- `PDFCatalog` — assemble a PDF catalogue.

## Dependencies

`numpy`, py4mt: `data_proc` (`load_edi`), `plotrjmcmc`, `util`, `version`.
