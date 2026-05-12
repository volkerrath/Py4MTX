# mt_plot_strikes.py

Plot impedance strike directions from MT station data.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_plot_strikes.py` |
| Authors | sb & vr (Dec 2019) |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

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

## Bugs fixed during earlier cleanup (2 Mar 2026)

| Bug | Description |
|-----|-------------|
| **Wrong docstring** | Said "produces a site list". Replaced. |
| **Unused imports** | `getpass`, `datetime`, `contextily`, `PdfPages` removed. |
| **Hard-coded `PY4MTX_DATA`** | Removed override. |
| **Unused variables** | `rng`, `nan`, `blank`, `rhoair` removed. |
| **Hard-coded survey name** | `'enfield'` changed to use `SURVEY_NAME`. |

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`EPSG`, `WORK_DIR`, `EDI_DIR`, `SURVEY_NAME`, `COLLECTION`, `FROM_EDIS`, `PLT_DIR`, `PLOT_FMT`, `DPI`, `PDF_CATALOG`, `PDF_CATALOG_NAME`). |
| **Import fix** | `from dtpy import …` corrected to `from mtpy import …` (dtpy is not a real package; this was likely a typo or alias). |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `EPSG` | UTM zone for coordinate projection |
| `WORK_DIR` / `EDI_DIR` | Path to EDI files |
| `SURVEY_NAME` | Survey identifier for mtpy metadata |
| `FROM_EDIS` | `True` to build dataset from EDI files; `False` to load from HDF5 |
| `PLOT_FMT` | List of plot formats (e.g. `['.png']`) |
| `PDF_CATALOG` | Assemble per-station plots into a PDF |

## Dependencies

`numpy`, `matplotlib`, `mtpy`, py4mt: `data_proc`, `util`, `version`.
