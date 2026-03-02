# modem_mod_fill.py

Fill a ModEM model with a uniform resistivity value.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_mod_fill.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads an existing ModEM model, replaces all subsurface cells with a
constant resistivity while preserving air and sea cells, and writes
the result. This is the standard way to create a homogeneous
prior / starting model that inherits the mesh and topography from
a previous inversion.

## Workflow

1. Read the input `.rho` model.
2. Identify air cells (ρ ≈ `rhoair`) and sea cells (ρ ≈ `rhosea`).
3. Set all cells to `FillVal`.
4. Restore air and sea cells to their original values.
5. Write the filled model.

## Inputs

| Item | Description |
|------|-------------|
| `InMod` | Path to the input ModEM model (without `.rho`). |

## Outputs

| Item | Description |
|------|-------------|
| `OutMod.rho` | Model with uniform subsurface resistivity. |

## Configuration

- `rhosea` — sea-water resistivity for identification (e.g. 0.3 Ωm).
- `rhoair` — air resistivity for identification (e.g. 1e17 Ωm).
- `FillVal` — uniform fill value in Ωm (e.g. 100).
- `InMod`, `OutMod` — input/output file paths.

## Dependencies

`numpy`, py4mt: `modem`, `util`, `version`.
