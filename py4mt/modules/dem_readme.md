# dem.py

Digital elevation models (DEM) for maps and sections: tile names, download and
cache, reading, mosaic, crop, thinning out and shaded relief.

The routines were moved out of `scripts/femtic/femtic_plot_sitemap.py` so that
any script can use them. They are stateless: every option is an argument,
nothing is read from script settings.

## DEM dictionary

All readers return (and the other functions take) a dict:

| Key | Content |
|---|---|
| `lon` | 1-D, degrees, increasing |
| `lat` | 1-D, degrees, decreasing (row 0 = north) |
| `z` | `z[lat, lon]`, elevation in m, float32, NaN = no value |

A box is `(lon_min, lon_max, lat_min, lat_max)` in degrees.

## Example

```python
import matplotlib.pyplot as plt
import dem

box = (5.95, 6.20, 45.85, 46.05)                      # lon_min, lon_max, lat_min, lat_max
d = dem.load_dem("copernicus30", box, cache_dir="/data/dem_cache")
fig, ax = plt.subplots()
if d is not None:
    relief = dem.shade_dem(d, style="hillshade", azdeg=315.0, altdeg=45.0)
    kw = dict(cmap="Greys_r", vmin=0.0, vmax=1.0) if relief["kind"] == "gray" else {}
    ax.imshow(relief["img"], extent=dem.dem_extent(d), origin="upper", **kw)
    ax.set_aspect("auto")
    note = dem.dem_credit("copernicus30")             # the notice to put on the map
```

## Sources

| `source` | Data | Format | Needs |
|---|---|---|---|
| `"copernicus30"` | Copernicus DEM GLO-30 (1 arc second), AWS bucket `copernicus-dem-30m` | GeoTIFF (COG) | `rasterio` |
| `"copernicus90"` | Copernicus DEM GLO-90 (3 arc seconds), bucket `copernicus-dem-90m` | GeoTIFF (COG) | `rasterio` |
| `"skadi"` | AWS Terrain Tiles (Mapzen), a composite of several data sets | `.hgt.gz` | numpy |
| `"tiles"` | own tiles, URL or path pattern (`url_template`) | `.tif`, `.hgt` | `rasterio` for `.tif` |
| `"file"` | one file (`file`) | `.tif`, `.npz` (`lon`, `lat`, `z`), `.hgt` | `rasterio` for `.tif` |
| `"none"` | no DEM | | |

Tile patterns use the placeholders `{name}` (e.g. `N65W017`), `{NS}`, `{lat}`,
`{EW}`, `{lon}` (lower left corner of a 1 x 1 degree tile).

Notes on the data (check the terms of the provider for your use):

- The Copernicus DEM is a surface model (buildings, vegetation) with heights
  above the EGM2008 geoid. There are no tiles over the open ocean, and a few
  tiles of some countries are not public in GLO-30.
- The Copernicus licence asks for a notice (`COPERNICUS_NOTICE`, returned by
  `dem_credit`).
- Information on the AWS Terrain Tiles:
  https://registry.opendata.aws/terrain-tiles

## Functions

| Function | Purpose |
|---|---|
| `load_dem(source, box, *, file=None, url_template=None, cache_dir="dem_cache", download=True, timeout=60.0, max_tiles=16, fill=0.0, strict=False, max_pixels=1600, user_agent=USER_AGENT, say=say_default)` | main entry: DEM of a box from any source, cropped and thinned out |
| `shade_dem(dem, *, style="hillshade", azdeg=315.0, altdeg=45.0, vert_exag=1.0, cmap="gist_earth")` | dict(`kind`, `img`) for `imshow`: `"gray"` (intensity 0..1) or `"rgba"` (style `"terrain"`) |
| `dem_extent(dem)` | `(west, east, south, north)` of the pixel edges |
| `dem_credit(source, url_template=None, credit="auto")` | credit line of a map; `"auto"` = that of the source (none for own sources) |
| `crop_dem(dem, lon_min, lon_max, lat_min, lat_max)` | the part that covers a box |
| `decimate_dem(dem, max_pixels)` | every s-th pixel |
| `load_dem_file(path, box=None, max_pixels=None)` | `.npz`, GeoTIFF or one `.hgt` tile |
| `load_dem_raster(...)`, `load_dem_hgt(...)` | a box from GeoTIFF or `.hgt` tiles |
| `read_raster_window(path, lon_min, lon_max, lat_min, lat_max, *, max_pixels=None)` | window of a GeoTIFF; projected rasters are warped to lon / lat |
| `merge_dems(parts, *, fill_boxes=(), fill=0.0)`, `mosaic_tiles(tiles, *, fill=0.0, sea=())` | join windows or tiles |
| `fetch_tile(...)`, `fetch_raster_tile(...)` | one tile from the cache, a path or a URL |
| `tiles_for_extent(...)`, `tile_name(lat0, lon0)`, `tile_url(template, lat0, lon0)`, `dem_template(source, template)`, `tile_kind(template)`, `is_url(text)` | names and patterns |
| `parse_hgt(raw, label)`, `read_hgt_file(path)` | the `.hgt` format |
| `tile_report(status, *, fill, strict, say)`, `raster_errors()`, `say_default(msg)` | messages and errors |

Constants: `DEM_SOURCES`, `DEM_STYLES`, `DEM_PRESETS`, `COPERNICUS_NOTICE`,
`EARTH_RADIUS_M`, `HGT_EXTENSIONS`, `RASTER_EXTENSIONS`, `USER_AGENT`.

## Behaviour

- Tiles of a URL are downloaded once into `cache_dir` and read from there
  afterwards; `download=False` uses only what is there.
- A tile the server does not have (HTTP 403 / 404: open ocean) gets the value
  `fill` (`None` = no value). Other failures give a message and no value there;
  with `strict=True` they raise `RuntimeError`.
- `load_dem` returns `None` for `"none"` or if no tile could be had. It raises
  `ValueError`, `OSError`, `RuntimeError` or `ImportError` for an unusable source
  or file, a DEM that does not cover the box, failed tiles with `strict`, or a
  GeoTIFF without `rasterio`.
- Some error texts name `DEM_MAX_TILES`, `DEM_URL_TEMPLATE` and `DEM_STYLE`,
  the script settings for `max_tiles`, `url_template` and `style`.
- Extents that cross the antimeridian are not supported.

## Dependencies

numpy; `rasterio` (imported only when a GeoTIFF is read); matplotlib (only in
`shade_dem`). `topo.py` is an older, separate SRTM helper and is not used here.

## Testing

Tested through `scripts/general/test_femtic_plot_sitemap.py` on synthetic files,
with a stand-in for `rasterio` and without network access. The reading of real
GeoTIFF files, the warping of projected rasters and the downloads have not been
run.

## Provenance

| Date | Author | Change |
|---|---|---|
| 2026-11-27 | Claude Sonnet 5.5 (Anthropic) | Created from the DEM functions of `femtic_plot_sitemap.py` (written 2026-11-21, Claude): explicit `say` and `user_agent` arguments; `_raster_errors()` / `_tile_report()` are now `raster_errors()` / `tile_report()`; constants without leading underscore; new `load_dem()` and `dem_credit()`. AI-generated; review before production use. |

Author of py4mt: Volker Rath (DIAS).
