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
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

Reads a ModEM `.rho` model file and writes it in one or more alternative
formats used by other modelling/visualisation packages.

## Supported output formats

| Format | Files produced | Used by |
|--------|---------------|---------|
| UBC | `.mod` + `.mesh` | UBC-GIF software suite |
| RLM / CGG | `.rlm` | CGG proprietary tools |

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`RHOAIR`, `BLANK`, `IN_FMT`, `OUT_FMT`, `IN_MOD`, `LAT`, `LON`). |
| **Runtime variable** | `OutMod` renamed to `OUT_MOD` for consistency. |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `IN_FMT` | Input format (currently only `'mod'` is implemented) |
| `OUT_FMT` | List of output formats, e.g. `['rlm', 'ubc']` |
| `IN_MOD` | Path to the ModEM model (without `.rho`) |
| `LAT`, `LON` | Origin coordinates for UBC output |

## Dependencies

`numpy`, py4mt: `modem`, `util`, `version`.
