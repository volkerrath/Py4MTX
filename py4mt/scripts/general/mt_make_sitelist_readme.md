# mt_make_sitelist.py

Generate a site list (names, coordinates, elevations) from EDI files.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_make_sitelist.py` |
| Authors | sb & vr (Dec 2019) |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

Reads all `.edi` files in a directory and writes a CSV/text site list
with station name, latitude, longitude, and optionally elevation and
site number. Output format is tailored for the target application.

## Output formats

| `WHAT_FOR` | Delimiter | Columns | Header |
|------------|-----------|---------|--------|
| `"wal"` | space | name, lat, lon | WALDIM-style (with site count row) |
| `"femtic"` | comma | name, lat, lon, elev, sitenum | FEMTIC input |
| other | comma | name, lat, lon, elev | General purpose |

## Issues fixed during earlier cleanup (2 Mar 2026)

| Issue | Description |
|-------|-------------|
| **Redundant env read** | `PY4MTX_DATA` read twice. Removed duplicate. |
| **Unused imports** | `save_edi`, `save_npz` removed. |
| **Dead UTM branch** | Simplified to exit directly when EPSG is `None`. |
| **Unused variable** | `dialect = 'unix'` removed. |

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`COORDS`, `EPSG`, `DELIM`, `WHAT_FOR`, `WORK_DIR`, `EDI_DIR`, `CSV_FILE`). |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `WHAT_FOR` | `"wal"`, `"femtic"`, or `"kml"` |
| `COORDS` | `"latlon"` (default for WALDIM) or `"utm"` |
| `EPSG` | UTM zone EPSG code (required if `COORDS = "utm"`) |
| `EDI_DIR` | Directory containing `.edi` files |

## Dependencies

`numpy`, `csv`, py4mt: `data_proc` (`load_edi`), `util`, `version`.
