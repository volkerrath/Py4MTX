# mt_plot_data.py

Plot MT station data (apparent resistivity, phase, tipper, phase tensor).

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_plot_data.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

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

## Bugs fixed during earlier cleanup (2 Mar 2026)

| Bug | Description |
|-----|-------------|
| **Wrong docstring** | Described the RTO algorithm. Replaced. |
| **Axis iteration** | `for ax in axs` iterated over rows. Fixed: `for ax in axs.flat`. |
| **Missing path separator** | `PLT_DIR` lacked trailing `/`. |
| **EDI branch dead code** | EDI branch computed `edi_files` then ignored them. |
| **Unused imports** | Removed 14 unused imports. |

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`DAT_DIR`, `PLT_DIR`, `USE_EDI`, `USE_NPZ`, `USE_DAT`, `DAT_LIST`, `STRNG_OUT`, `FILES_ONLY`, `PLT_FMT`, `CAT_NAME`, `CATALOG`, `NCOLS`, `PLTARGS`). |
| **Directory creation** | Added `os.makedirs(PLT_DIR)` guard before plotting. |
| **Removed unused RNG** | `rng = np.random.default_rng()` and `nan = np.nan` were unused. Removed. |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `DAT_DIR` | Directory containing data files |
| `USE_EDI` / `USE_NPZ` / `USE_DAT` | Select input format (only one should be `True`) |
| `PLT_FMT` | List of output formats (e.g. `['.png', '.pdf']`) |
| `CATALOG` | Assemble a PDF catalogue of all plots |
| `FIGURE_TITLE` | Figure title of the station figures (2026-11-11). `"auto"` = the station name as before; `None`, `False`, `""` = none; text with `{station}`, `{key}`; dict `{"<station>": spec, "default": spec}`. |

## Dependencies

`numpy`, `pandas`, `matplotlib`, py4mt: `data_viz`, `util`, `version`.

## Changes 2026-11-11 (titles)

`FIGURE_TITLE` (table above), via `util.resolve_title` (needs the `util.py` of the same date). Claude Sonnet 5.5 (Anthropic): AI-generated; review before use. Defaults give the figures of before. Only compile-checked and tested at the level of the title logic (`util.resolve_title`, `test_util_titles.py`); the script was not run end to end.
