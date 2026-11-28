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

Reads MT station data from `.npz` files (see *Input* below) and generates
per-station 3×2 diagnostic subplot figures showing apparent resistivity,
impedance phase, tipper, and phase tensor. Optionally assembles all plots
into a PDF catalogue.

## Input

Station files as written by `data_proc.save_npz` (`mt_data_processor.py`, token
`npz`): `freq` `(n,)`, `Z` `(n,2,2)`, optionally `Z_err`, `T`, `P` and their errors,
and `station`. Files with one 1-D array per column (`freq`, `rho_xy`, `phi_xy`, ...,
`station`) work as well. `*_collection.npz` files are not read. The files are loaded
with `allow_pickle=True` (needed for the text entries): only use files you trust.

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
| `USE_EDI` / `USE_NPZ` / `USE_DAT` | Select input format (only one should be `True`). Only `USE_NPZ` works: the plotting loop does not read `.edi` or `.dat` files |
| `PLT_FMT` | List of output formats (e.g. `['.png', '.pdf']`) |
| `CATALOG` | Assemble a PDF catalogue of all plots |
| `PHASE_WRAP_MODE`, `PHASE_WRAP_STYLE` | Phase panels (2026-11-20): a phase curve that jumps by more than 180 deg between two periods (it leaves the +-180 deg window) is interrupted and the two points joined by a separate line. Mode `"direct"` (default, straight line), `"short"` (through the window edge), `"off"` (as before); style = dict of Matplotlib line properties over the dotted default (`None` = dotted, `False` = gap only). Passed to `add_phase` only |
| `LINE_KW` | Matplotlib line keywords for the `add_rho` / `add_phase` curves, taken from `PLTARGS` (`{"linewidth": PLTARGS["Linewidths"][0]}`) |
| `FIGURE_TITLE` | Figure title of the station figures (2026-11-11). `"auto"` = the station name as before; `None`, `False`, `""` = none; text with `{station}`, `{key}`; dict `{"<station>": spec, "default": spec}`. |

## Dependencies

`numpy`, `matplotlib`, py4mt: `data_proc` (needs `pandas`, `scipy`), `data_viz`, `util`,
`version`.

## Changes 2026-11-11 (titles)

`FIGURE_TITLE` (table above), via `util.resolve_title` (needs the `util.py` of the same date). Claude Sonnet 5.5 (Anthropic): AI-generated; review before use. Defaults give the figures of before. Only compile-checked and tested at the level of the title logic (`util.resolve_title`, `test_util_titles.py`); the script was not run end to end.

## Changes 2026-11-20 (phase wraps)

`PHASE_WRAP_MODE` / `PHASE_WRAP_STYLE` (table above), through `data_viz.add_phase` (needs the `data_viz.py` of the same
date). Claude Sonnet 5.5 (Anthropic): AI-generated; review before use.

One more change, needed to run the script at all: the whole `PLTARGS` dict (keys such as `"Fontsize"`, `"Labelsize"`,
`"Colors"`) was passed to `add_rho` / `add_phase`, which hand unknown keywords on to Matplotlib (`AttributeError:
Line2D.set() got an unexpected keyword argument 'Fontsize'`). The curves now get `LINE_KW` (the line width of `PLTARGS`);
`PLTARGS` itself is unchanged. Ran end to end on a synthetic `observation.npz` (PNG, PDF and catalogue written).
Not changed: `pd.DataFrame(np.load(...))` on an `NpzFile` gave an empty-looking frame with pandas 3.0 in the test
environment, and `mpl.backends.backend_pdf` is not always imported by then; both depend on the installed versions.

## Changes 2026-11-27 (figure size in cm)

Claude Sonnet 5.5 (Anthropic): AI-generated; review before use.  New setting
`FIG_SIZE = (20.32, 35.56)` [cm] (width, height of the station figures; before:
8 x 14 inch, fixed in the code).

## Changes 2026-11-28 (script stopped before plotting)

Claude Sonnet 5.5 (Anthropic): AI-generated; review before use.

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: module 'matplotlib.backends' has no attribute 'backend_pdf'` | the submodule was used without importing it | `from matplotlib.backends.backend_pdf import PdfPages` |
| `ValueError: Object arrays cannot be loaded when allow_pickle=False` (processor files) | `np.load` without `allow_pickle` | `data_proc.load_npz`; `station` is a `str` |
| `KeyError: 'freq'` (pandas 3.0.5) | `pd.DataFrame` on the open `.npz` file | `data_viz.datadict_to_plot_df` (one column per component from `Z`, `T`, `P`) |

Settings, panel layout and output file names are unchanged. New test:
`test_mt_plot_data.py` (runs the script on small station files).
Not changed: `USE_EDI` / `USE_DAT` (see *Configuration*).
