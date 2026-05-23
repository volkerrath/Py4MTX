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
| Modified | 2026-05-23 — UTM zone auto-derived per site via `util.utm_zone_from_latlon` / `util.latlon_to_utm_zn`; FEMTIC output now includes easting & northing; removed COORDS/EPSG; Claude Sonnet 4.6 (Anthropic) |

## Purpose

Reads all `.edi` files in a directory and writes a CSV/text site list
with station name, latitude, longitude, and optionally elevation and
site number. Output format is tailored for the target application.

## Output formats

| `WHAT_FOR` | Delimiter | Columns | Header |
|------------|-----------|---------|--------|
| `"wal"` | space | name, lat, lon | WALDIM-style (with site count row) |
| `"femtic"` | comma | name, lat, lon, elev, sitenum, easting, northing | FEMTIC input + UTM coords |
| other | comma | name, lat, lon, elev | General purpose |

The FEMTIC sitelist format is consumed by `data_proc.read_sitelist()` and
by `femtic_mod_plot.py` (via `read_sitelist`) to overlay site positions on
model slice figures.

## Configuration

| Constant | Description |
|----------|-------------|
| `WHAT_FOR` | `"wal"`, `"femtic"`, or `"kml"` |
| `UTM_ZONE_OVERRIDE` | Integer zone number to force for all sites, or `None` = auto-derive per site from its own lat/lon |
| `EDI_DIR` | Directory containing `.edi` files |

## UTM zone derivation

The UTM zone is derived **per site** by calling `util.utm_zone_from_latlon(lat, lon)`
after reading each EDI file.  `UTM_ZONE_OVERRIDE` forces a fixed zone when
sites span a zone boundary or when the survey convention differs from the
standard 6° rule.

## Changes (2026-05-23)

| Change | Description |
|--------|-------------|
| **Zone auto-derive** | `utm_zone_from_latlon(lat, lon, override=UTM_ZONE_OVERRIDE)` replaces old EPSG-integer path |
| **UTM conversion** | `latlon_to_utm_zn(lat, lon, zone, northern)` replaces `proj_latlon_to_utm(EPSG=...)` |
| **FEMTIC columns** | Row is now `[name, lat, lon, elev, sitenum, easting, northing]` |
| **Removed** | `COORDS`, `EPSG` config variables; `sys.exit` on missing EPSG |
| **Added** | `UTM_ZONE_OVERRIDE = None` config; per-site diagnostic print of zone and UTM coords |

## Issues fixed during earlier cleanup (2 Mar 2026)

| Issue | Description |
|-------|-------------|
| **Redundant env read** | `PY4MTX_DATA` read twice. Removed duplicate. |
| **Unused imports** | `save_edi`, `save_npz` removed. |
| **Dead UTM branch** | Simplified to exit directly when EPSG is `None`. |
| **Unused variable** | `dialect = 'unix'` removed. |

## Dependencies

`numpy`, `csv`, py4mt: `data_proc` (`load_edi`), `util` (`utm_zone_from_latlon`,
`latlon_to_utm_zn`), `version`.
