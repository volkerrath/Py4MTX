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
