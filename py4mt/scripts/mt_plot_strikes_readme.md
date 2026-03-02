# mt_plot_strikes.py

Plot impedance strike directions from MT station data.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_plot_strikes.py` |
| Authors | sb & vr (Dec 2019) |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Uses mtpy to generate strike-direction rose diagrams from MT impedance
data. Produces station map, aggregate strike plot, per-decade strike
plot, and individual per-station strike diagrams. Optionally assembles
a PDF catalogue.

## Workflow

1. Build an `MTData` dataset from EDI files (or load from an HDF5 collection).
2. Plot station locations on a map.
3. Plot aggregate strike rose diagram (all data).
4. Plot per-decade strike roses (impedance, tipper, phase tensor, invariants).
5. Loop over stations and plot individual strike diagrams.
6. Optionally build a PDF catalogue.

## Bugs fixed during cleanup

| Bug | Description |
|-----|-------------|
| **Wrong docstring** | Docstring said "produces a site list" (copy-paste from `mt_make_sitelist`). Replaced with correct description. |
| **Unused imports** | `getpass`, `datetime`, `contextily`, `PdfPages` imported but never used. Removed. |
| **Hard-coded PY4MTX_DATA** | Overwrote env-var value on line 70. Removed override. |
| **Unused variables** | `rng`, `nan`, `blank`, `rhoair` defined but never used. Removed. |
| **Hard-coded survey name** | `'enfield'` in the `else` branch did not match `surveyname = 'Annecy'`. Changed to use `surveyname`. |

## Configuration

- `EPSG` — UTM zone for coordinate projection.
- `WorkDir` / `EdiDir` — path to EDI files.
- `surveyname` — survey identifier for mtpy metadata.
- `FromEdis` — `True` to build dataset from EDI files; `False` to load from HDF5 collection.
- `PlotFmt` — list of plot formats (e.g. `['.png']`).
- `PDFCatalog` — assemble per-station plots into a PDF.

## Dependencies

`numpy`, `matplotlib`, `mtpy`, py4mt: `mtproc`, `util`, `version`.
