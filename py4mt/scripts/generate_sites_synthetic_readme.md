# generate_sites_synthetic.py

Generate synthetic MT site locations and EDI files on a regular grid.

## Provenance

| Field | Value |
|-------|-------|
| Script | `generate_sites_synthetic.py` |
| Authors | sb & vr (July 2020) |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 3 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Creates a rectangular grid of magnetotelluric (MT) station locations and
writes one EDI file per site, using an existing EDI template for frequency
content and data structure.  Designed for setting up forward-modelling
studies where a controlled site layout is needed.

## Workflow

1. Define geographic bounds (lat/lon limits) and grid spacing.
2. Convert to UTM, compute grid dimensions, and ensure even counts.
3. Build a meshgrid of UTM coordinates and project back to lat/lon.
4. For each grid node, clone the EDI template with updated coordinates and
   station name, then write the new EDI file.

## Configuration constants

| Constant | Description |
|----------|-------------|
| `EDI_TEMPLATE` | Path to the EDI file used as a template. |
| `EDI_OUT_DIR` | Output directory for generated EDI files. |
| `OUT_NAME` | Prefix for station names and file names (e.g. `"Krafla_"`). |
| `EDI_GEN` | Generation mode — `"rect regular"` or `"rect random"`. |
| `LAT_LIMITS` | Latitude range of the study area (south, north). |
| `LON_LIMITS` | Longitude range of the study area (west, east). |
| `CENTER_LATLON` | Centre of the grid (computed from limits by default). |
| `DX`, `DY` | Grid spacing in metres. |

## Inputs

| Item | Description |
|------|-------------|
| EDI template file | Provides frequency list, data structure, and metadata skeleton. |

## Outputs

| Item | Description |
|------|-------------|
| `<OUT_NAME><N>.edi` | One EDI file per grid node, written to `EDI_OUT_DIR`. |

## Dependencies

`numpy`, `mtpy`; py4mt modules: `util`, `modem`, `version`.
