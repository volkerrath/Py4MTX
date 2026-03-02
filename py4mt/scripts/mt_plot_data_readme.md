# mt_plot_data.py

Plot MT station data (apparent resistivity, phase, tipper, phase tensor).

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_plot_data.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads MT station data from `.npz`, `.edi`, or `.dat` files and generates
per-station 3×2 diagnostic subplot figures showing apparent resistivity,
impedance phase, tipper, and phase tensor. Optionally assembles all plots
into a PDF catalogue.

## Plot layout

| Row | Left | Right |
|-----|------|-------|
| 1 | ρ_a (xy, yx) | Phase (xy, yx) |
| 2 | ρ_a (xx, yy) | Phase (xx, yy) |
| 3 | Tipper | Phase tensor |

Empty axes are automatically removed.

## Bugs fixed during cleanup

| Bug | Description |
|-----|-------------|
| **Wrong docstring** | Docstring described the RTO algorithm (copy-paste from another script). Replaced with correct description. |
| **Axis iteration** | `for ax in axs` iterated over rows (arrays) instead of individual axes. Fixed: `for ax in axs.flat`. |
| **Missing path separator** | `PltDir` lacked trailing `/`, causing filenames to run together. |
| **EDI branch dead code** | EDI branch computed `edi_files` then ignored them, hardcoding an NPZ path instead. |
| **Unused imports** | Removed 14 unused imports: `shutil`, `functools`, `time`, `warnings`, `csv`, `scipy.sparse`, `scipy.interpolate`, `scipy`, `femtic`, `save_edi`, `save_ncd`, `save_hdf`, `save_npz`, `util.stop`, and more. |

## Configuration

- `DatDir` — directory containing data files.
- `UseEDI` / `UseNPZ` / `UseDAT` — select input format (only one should be `True`).
- `PltFmt` — list of output formats (e.g. `['.png', '.pdf']`).
- `Catalog` — assemble a PDF catalogue of all plots.

## Dependencies

`numpy`, `pandas`, `matplotlib`, py4mt: `data_viz`, `util`, `version`.
