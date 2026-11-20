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
| Modified | 2026-11-20 -- sites may lie in different UTM zones: one common zone for the whole survey (`UTM_ZONE_MODE = "common"`, default); zone report, scale warning, zone comment line; Claude Sonnet 5.5 (Anthropic); AI-generated, review before production use |

## Purpose

Reads all `.edi` files in a directory and writes a CSV/text site list
with station name, latitude, longitude, and optionally elevation and
site number. Output format is tailored for the target application.

## Output formats

| `WHAT_FOR` | Delimiter | Columns | First lines |
|------------|-----------|---------|-------------|
| `"wal"` | space | name, lat, lon | WALDIM-style (`Sitename Latitude Longitude`, then the site count); no UTM |
| `"fem"` / `"femtic"` | comma | name, lat, lon, elev, sitenum, easting, northing | two `#` comment lines with the UTM zone (see below) |
| `"kml"` or other | comma | the same seven columns as FEMTIC | the same two `#` lines |

The FEMTIC sitelist format is consumed by `data_proc.read_sitelist()`, by
`femtic.read_site_dat()` and by the FEMTIC plot scripts to overlay site
positions on model figures.  All of them skip lines starting with `#`.

## Configuration

| Constant | Description |
|----------|-------------|
| `WHAT_FOR` | `"wal"`, `"femtic"`, or `"kml"` |
| `EDI_DIR` | Directory containing `.edi` files |
| `UTM_ZONE_MODE` | `"common"` (default): one UTM zone for all sites. `"per_site"`: each site in its own zone (old behaviour) |
| `UTM_ZONE_OVERRIDE` | `None`, or an integer zone (1-60) forced for all sites |
| `UTM_MAX_SCALE_SPREAD` | Warn if the UTM scale factor varies by more than this across the sites (default `1e-3` = 1 m/km) |
| `WRITE_ZONE_HEADER` | Write the `# utm_zone=...` comment lines (default `True`; set `False` for programs that do not skip `#` lines) |

## Sites in different UTM zones (2026-11-20)

Easting and northing are only comparable if they are in the same zone.
Before 2026-11-20 every site got the zone of its own longitude, so a survey
that crosses a zone boundary (e.g. 12 deg E, between zones 32 and 33) got
eastings of about 7.3e5 m on one side and 2.7e5 m on the other for sites
a few km apart.

Now (`UTM_ZONE_MODE = "common"`):

- All EDI files are read first.  `util.utm_common_zone()` picks the zone of
  the **mean position** of the sites (mean latitude; mean longitude taken
  relative to the first site and wrapped, so surveys across the antimeridian
  work).  The hemisphere follows the mean latitude; sites on the other side of
  the equator are projected with the false northing of that hemisphere, so the
  northing stays continuous.
- Every site is projected into that zone with `util.latlon_to_utm_zn()`
  (pyproj / PROJ `utm`).  Sites outside the zone are simply projected with the
  zone's central meridian ("extended UTM").  The default PROJ algorithm
  (Poder/Engsager) is accurate far from the central meridian; the faster
  `+approx` variant, which this script does not use, is not (diverges beyond
  about 3 deg).
- `UTM_ZONE_OVERRIDE = 32` forces the zone instead of the mean.
- The zone is the one that downstream FEMTIC scripts derive from the mean of
  the site latitudes/longitudes, so both agree except for surveys that lie
  almost on a boundary; see the caveat below.

**Distortion.** The farther a site is from the central meridian, the larger
the UTM scale factor k (grid distance = k x ground distance;
k = 0.9996 / sqrt(1 - (cos(lat) sin(dlon))^2) on the sphere).  The script
prints k for every site, the range of k over the survey and the largest
distance from the central meridian, and warns if the range exceeds
`UTM_MAX_SCALE_SPREAD`.  Example (47 N, sites between 11.8 and 12.25 E in
zone 33): k = 1.00013 to 1.00033, a spread of 0.19 m/km, no warning.  Near the
equator, 9 deg off the central meridian, k is about 1.012; then a local
projection is the better choice.

**Output.** The first lines of the FEMTIC/CSV list are comments:

```
# utm_zone=33 hemisphere=N mode=common central_meridian=15.0
# easting/northing [m] WGS84 UTM; sites outside the zone are projected into it (extended UTM)
```

With `UTM_ZONE_MODE = "per_site"` the zones differ and the file says
`# utm_zone=per_site zones=32N/33N ...`; the script warns that these
coordinates must not be used together.

**Caveat (downstream).** The FEMTIC plot/prep scripts (`femtic_mod_plot_slice.py`,
`femtic_mod_plot_3d.py`, ...) read easting/northing from the list but derive
the zone for lat/lon conversions themselves, from the mean site position or
from the mesh origin (`UTM_ZONE_OVERRIDE`).  If the survey sits close to a
zone boundary the zone they find can differ from the one in the list; set
their `UTM_ZONE_OVERRIDE` to the zone in the first line of `site.dat`.

## Tests

`test_utm_common_zone.py` (55 cases): zone choice (boundary, majority side,
antimeridian, equator, override, non-finite values), central meridian and scale
factor, grid distance = ground distance x k across the boundary, agreement with
an independent Krueger-series (Karney 2011) reference up to 9 deg from the
central meridian, and end-to-end runs of this script on a synthetic survey
across 12 deg E in both modes.  Run from `scripts/general`:
`python -m pytest -q test_utm_common_zone.py`.

## Changes (2026-11-20)

| Change | Description |
|--------|-------------|
| **Common zone** | `UTM_ZONE_MODE = "common"` (default) puts all sites in one zone, chosen by `util.utm_common_zone()`; `"per_site"` keeps the old behaviour |
| **Report** | Per-site table (zone, own zone, E, N, distance from the central meridian, scale factor) and a summary: zones used and zones the sites belong to, number of sites outside the zone, scale-factor range |
| **Warnings** | Different zones in `"per_site"` mode; scale-factor range above `UTM_MAX_SCALE_SPREAD` |
| **Zone comment** | `# utm_zone=...` lines at the top of FEMTIC/CSV lists (`WRITE_ZONE_HEADER`) |
| **Order of work** | All EDI files are read first, the list is written afterwards (no half-written file if an EDI has no valid LAT/LONG, which now stops the script with the file name) |
| **Formats** | `"kml"`/other write the same seven columns as FEMTIC (as the code always did; the table in the old README was wrong); `"wal"` no longer computes UTM coordinates |

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
`utm_common_zone`, `utm_central_meridian`, `utm_scale_factor`, `latlon_to_utm_zn`), `version`.
