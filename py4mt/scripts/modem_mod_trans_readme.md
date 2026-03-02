# modem_mod_trans.py

Convert a ModEM model file to UBC and/or RLM (CGG) format.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_mod_trans.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads a ModEM `.rho` model file and writes it in one or more alternative
formats used by other modelling/visualisation packages.

## Supported output formats

| Format | Files produced | Used by |
|--------|---------------|---------|
| UBC | `.mod` + `.mesh` | UBC-GIF software suite |
| RLM / CGG | `.rlm` | CGG proprietary tools |

## Inputs

| Item | Description |
|------|-------------|
| `InMod` | Path to the ModEM model (without `.rho`). |
| `lat`, `lon` | Geographic coordinates of the model origin (needed for UBC). |

## Outputs

Files are written alongside the input with format-specific extensions.

## Configuration

- `InFmt` — input format (currently only `'mod'` is implemented).
- `OutFmt` — list of output formats, e.g. `['rlm', 'ubc']`.
- `lat`, `lon` — origin coordinates for UBC output.

## Dependencies

`numpy`, py4mt: `modem`, `util`, `version`.
