# mt_kmz_waldim.py

Export WALDIM dimensionality analysis results as a KMZ file for Google Earth.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_kmz_waldim.py` |
| Authors | sb & vr (May 2023) |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads WALDIM dimensionality classification output and creates a
Google Earth KMZ file with colour-coded site markers. Each site is
coloured by its WALDIM dimensionality class. Supports per-frequency
or per-band classification, and a simplified 3-class or full 10-class
colour scheme.

## WALDIM classification (10-class)

| Code | Class |
|------|-------|
| 0 | Undetermined |
| 1 | 1-D |
| 2 | 2-D |
| 3 | 3-D/2-D (twist only) |
| 4 | 3-D/2-D (general) |
| 5 | 3-D |
| 6 | 3-D/2-D with diagonal regional tensor |
| 7 | 3-D/2-D or 3-D/1-D indistinguishable |
| 8 | Anisotropy hint: homogeneous anisotropic medium |
| 9 | Anisotropy hint: anisotropic body within 2-D medium |

## Bugs fixed during cleanup

| Bug | Description |
|-----|-------------|
| **`kml` variable shadow** | `kml = False` (boolean flag) was overwritten by `kml = simplekml.Kml(...)` (object). Renamed: `SaveKml`/`SaveKmz` for flags, `kml_obj` for the object. |
| **Duplicated styling code** | Per-frequency and per-band branches had identical placemark styling. Extracted into `_style_site()` helper. |
| **Unused variable** | `bnd_name` defined but never used. Removed. |

## Configuration

- `UseFreqs` — `True` for per-frequency mode, `False` for per-band mode.
- `Class3` — `True` for 3-class scheme (1-D/2-D/3-D), `False` for full 10-class.
- `DimFile` — path to the WALDIM output `.dat` file.

## References

Martí, Queralt & Ledo (2009), Computers & Geosciences, 35, 2295-2303.
Martí, Queralt, Ledo & Farquharson (2010), PEPI, 182, 139-151.

## Dependencies

`numpy`, `simplekml`, `matplotlib` (colormaps for 10-class mode), py4mt: `util`, `version`.
