# mt_make_sitelist.py

Generate a site list (names, coordinates, elevations) from EDI files.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_make_sitelist.py` |
| Authors | sb & vr (Dec 2019) |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads all `.edi` files in a directory and writes a CSV/text site list
with station name, latitude, longitude, and optionally elevation and
site number. Output format is tailored for the target application.

## Output formats

| `whatfor` | Delimiter | Columns | Header |
|-----------|-----------|---------|--------|
| `"wal"` | space | name, lat, lon | WALDIM-style (with site count row) |
| `"femtic"` | comma | name, lat, lon, elev, sitenum | FEMTIC input |
| other | comma | name, lat, lon, elev | General purpose |

## Issues fixed during cleanup

| Issue | Description |
|-------|-------------|
| **Redundant env read** | `PY4MTX_DATA` was read from environment twice (lines 37 and 49). Removed duplicate. |
| **Unused imports** | `save_edi`, `save_npz` imported but never used. Removed. |
| **Dead UTM branch** | When EPSG is `None`, code computed UTM zone then immediately called `sys.exit`. Simplified to exit directly. |
| **Unused variable** | `dialect = 'unix'` defined but never passed to csv.writer. Removed. |

## Configuration

- `whatfor` — `"wal"`, `"femtic"`, or `"kml"`.
- `Coords` — `"latlon"` (default for WALDIM) or `"utm"`.
- `EPSG` — UTM zone EPSG code (required if `Coords = "utm"`).
- `EdiDir` — directory containing `.edi` files.

## Dependencies

`numpy`, `csv`, py4mt: `data_proc` (`load_edi`), `util`, `version`.
