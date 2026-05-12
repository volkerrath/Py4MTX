# mt_aniso1d_inversion.py

Deterministic anisotropic 1-D MT inversion driver.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_aniso1d_inversion.py` |
| Author | Volker Rath (DIAS), with ChatGPT (GPT-5 Thinking), 2026-02-13 |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 3 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Batch deterministic inversion of anisotropic 1-D MT data. Reads `.npz`
station files, builds a starting model, and runs Gauss-Newton inversion
with Tikhonov or TSVD regularisation. Supports automatic regularisation
parameter selection (GCV, L-curve, ABIC).

## Workflow

1. Define or load a starting model (layer thicknesses, σ_min/σ_max, strike).
2. Build a `ParamSpec` describing the inversion parameterisation.
3. Loop over station `.npz` files:
   - Optionally compute phase tensor and bootstrap errors.
   - Run `inverse.invert_site()` with the chosen method.
   - Save results as `.npz`.

## Configuration constants

| Constant | Description |
|----------|-------------|
| `N_LAYER`, `H_M` | Number of layers and thickness vector for the starting model. |
| `MODEL0` | In-file starting model dict. |
| `DATA_DIR` | Base data directory. |
| `INPUT_GLOB` | Glob pattern for input `.npz` station files. |
| `OUTDIR` | Output directory. |
| `MODEL_DIRECT` | Set to `MODEL0` to use in-file model, or `None` to load from `MODEL_NPZ`. |
| `INV_METHOD` | `"tikhonov"` or `"tsvd"`. |
| `ALPHA_SELECT` | `"fixed"`, `"lcurve"`, `"gcv"`, `"abic"`. |
| `PARAM_DOMAIN` | `"rho"` or `"sigma"`. |
| `PARAM_SET` | `"minmax"` or `"max_anifac"`. |
| `USE_PT` | Include phase tensor data in inversion. |
| `PLOT_RESULTS` | Enable result plotting (not yet implemented). |

## Outputs

Per station: `<station>_inverse_<method>_<select>.npz` containing the
inversion result (model, residuals, diagnostics).

## Status

Result plotting (`PLOT_RESULTS`) is declared but not yet implemented.

## Dependencies

`numpy`; py4mt: `inverse`, `util`, `version`.
