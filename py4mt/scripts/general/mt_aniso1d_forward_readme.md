# mt_aniso1d_forward.py

Anisotropic 1-D MT forward modelling and plotting.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_aniso1d_forward.py` |
| Author | Volker Rath (DIAS), with ChatGPT (GPT-5 Thinking), 2026-02-13 |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 3 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Computes the full impedance tensor, apparent resistivity/phase, and
phase tensor for a layered anisotropic conductivity model. Results
are written to text files and optionally plotted.

## Workflow

1. Define a 1-D anisotropic model (thickness, ρ_x/ρ_y/ρ_z, strike/dip/slant per layer).
2. Call `aniso1d_impedance_sens()` to compute the impedance tensor Z(T).
3. Derive phase tensor P from Z.
4. Export impedance (Re/Im interlaced), ρ_a / φ, and phase tensor to `.dat` files.
5. Generate 2×2 subplot figures for impedance, ρ_a/φ, and phase tensor.

## Configuration constants

| Constant | Description |
|----------|-------------|
| `WORK_DIR` | Working / output directory. |
| `SUM_OUT`, `RES_FILE` | Write model summary to file. |
| `IMP_OUT`, `IMP_FILE`, `IMP_PLT` | Impedance output and plot switches. |
| `RHO_OUT`, `RHO_FILE`, `RHO_PLT` | Apparent resistivity / phase output and plot switches. |
| `PHT_OUT`, `PHT_FILE`, `PHT_PLT` | Phase tensor output and plot switches. |
| `PLOT_FORMAT`, `PLOT_FILE` | Plot file format(s) and base name. |
| `PLTARGS` | Dict of plotting parameters (sizes, fonts, colours, markers). |
| `FIGURE_TITLE` | Figure titles (2026-11-11). `"auto"` = `PLTARGS["suptitle"]` as before; `None`, `False`, `""` = none; text with `{title}` (the automatic title) and `{key}` (`"imped"`, `"rhophas"` or `"phstens"`); dict `{"imped": spec, "rhophas": spec, "phstens": spec, "default": spec}`. |
| `PANEL_TITLES` | `False` = no titles of the panels of the three figures (default `True`). |
| `PHASE_WRAP_MODE`, `PHASE_WRAP_STYLE` | Phase panels of the rho/phase figure (2026-11-20): a phase curve that jumps by more than 180 deg between two periods is interrupted and the two points joined by a separate line. Mode `"direct"` (default, straight line), `"short"` (through the window edge; meant for phases inside +-180 deg, the panels show phases shifted by +-90 deg), `"off"`; style = dict of Matplotlib line properties over the dotted default (`None` = dotted, `False` = gap only). Passed to `viz.plot_rhophas` (keys `phase_wrap_*` of `PLTARGS`, set per panel: off for the rho panels). |
| `PERIODS`, `N_LAYER`, `MODEL` | Model definition (period array, layer count, parameter matrix). |

## Outputs

| File | Contents |
|------|----------|
| `summary.dat` | Model parameters (thickness, resistivities, angles). |
| `impedance.dat` | Re/Im of all four impedance components vs. period. |
| `rhophas.dat` | Apparent resistivity and phase vs. period. |
| `phstens.dat` | Phase tensor elements vs. period. |
| `*_imped.png` | Impedance plots. |
| `*_rhophas.png` | Apparent resistivity / phase plots. |
| `*_phstens.png` | Phase tensor plots. |

## Changes 2026-11-11 (titles)

`FIGURE_TITLE` and `PANEL_TITLES` (table above); both work through `util.resolve_title` (needs the `util.py` of the same date). Claude Sonnet 5.5 (Anthropic): AI-generated; review before use. Defaults give the figures of before. Only compile-checked and tested at the level of the title logic (`util.resolve_title`, `test_util_titles.py`); the script was not run end to end.

## Changes 2026-11-20 (phase wraps)

`PHASE_WRAP_MODE` / `PHASE_WRAP_STYLE` (table above), through `viz.plot_rhophas` (needs the `viz.py` and `data_viz.py` of the same date). Claude Sonnet 5.5 (Anthropic): AI-generated; review before use. Ran end to end on the model of the script (all three figures written); the dotted connections show in the xx/yy phase panel.

## Bugs fixed during cleanup

| Bug | Description |
|-----|-------------|
| **PhTFile = RhoFile** | Phase tensor output filename was identical to `RhoFile`, causing overwrite. Fixed: separate `phstens.dat`. |
| **Legend typo** | ρ_a legend showed "xy, xy" instead of "xy, yx". |
| **Impedance interlace write** | `ImpOut` block applied `.real`/`.imag` to already-real interlaced array; rewrote to index columns directly. |
| **Cross-block variable dependency** | `RhoPlt` block used `rhoa2`/`phas2` from `RhoOut` block; merged into single compute-once block. |
| **Dead code** | ~100 lines of commented-out alternative models and old loop removed. |
| **Unused import** | `mu0` defined but never used; removed. |

## Dependencies

`numpy`, `matplotlib`; py4mt: `aniso`, `data_proc`, `viz`, `util`, `version`.

## Changes 2026-11-27 (figure size in cm)

Claude Sonnet 5.5 (Anthropic): AI-generated; review before use.  The figure sizes are
settings in cm: `FIG_SIZE = (40.64, 40.64)` (impedance and rho/phase figures)
and `FIG_SIZE_PT = (40.64, 20.32)` (phase tensor figure). Before, they were
fixed in the code in inch; the defaults give the same figures.
