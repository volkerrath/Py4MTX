# topo.py — SRTM DEM download, mosaicking, and conversion

`topo.py` (internally documented as `srtm_tools.py`) provides a self-contained
pipeline for fetching SRTM elevation tiles from the USGS archive, mosaicking
them into a single DEM, rotating the result, and exporting to various formats.

Tiles are fetched from the **USGS SRTM v2.1 archive** (SRTM3, 3 arc-second
≈ 90 m resolution).  Filenames follow the convention
`N45E006.hgt.zip` under continent directories (e.g. `Eurasia`).

---

## Functions

### `tiles_for_bbox(lat_min, lat_max, lon_min, lon_max)`

Enumerate all SRTM tile names required to cover a geographic bounding box.

Returns a list of tile name strings (e.g. `["N45E006", "N45E007", ...]`).

---

### `download_srtm_tile(tile, continent="Eurasia", out_dir="srtm")`

Download and unzip a single SRTM tile from the USGS archive.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tile` | — | Tile name, e.g. `"N45E006"` |
| `continent` | `"Eurasia"` | Subdirectory on the USGS server |
| `out_dir` | `"srtm"` | Local directory for downloaded files |

Returns the path to the unpacked `.hgt` file.

---

### `mosaic_hgt(hgt_paths, out_path="srtm_mosaic.tif")`

Mosaic multiple `.hgt` files into a single GeoTIFF using `rasterio.merge`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hgt_paths` | — | List of `.hgt` file paths |
| `out_path` | `"srtm_mosaic.tif"` | Output GeoTIFF path |

Returns the output path.

---

### `rotate_raster(in_path, out_path, angle_deg)`

Rotate a raster by an angle using an affine transform.

> **Note:** This applies a simple affine rotation and does not reproject the
> data.  For grid-aligned rotation (preserving correct georeferencing),
> use `rasterio.warp.reproject` instead.

| Parameter | Description |
|-----------|-------------|
| `in_path` | Input GeoTIFF |
| `out_path` | Output GeoTIFF |
| `angle_deg` | Rotation angle in degrees (positive = counter-clockwise) |

---

### `geotiff_to_xyz(in_path, out_path=None, as_dataframe=False, step=1)`

Convert a GeoTIFF raster to XYZ coordinates (longitude, latitude, elevation).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_path` | — | Input GeoTIFF |
| `out_path` | `None` | Optional path to write a `.xyz` ASCII file |
| `as_dataframe` | `False` | If `True`, return a `pandas.DataFrame`; otherwise an `(N, 3)` ndarray |
| `step` | `1` | Subsampling step (1 = every pixel, 2 = every other, …) |

---

### `process_srtm(lat_min, lat_max, lon_min, lon_max, angle_deg, continent="Eurasia", out_dir="srtm")`

Full one-call pipeline: enumerate tiles → download → mosaic → rotate.

Returns the path to the rotated GeoTIFF.

---

## Typical usage

```python
from topo import process_srtm, geotiff_to_xyz

# Download and mosaic SRTM tiles for a survey area, then rotate
dem_path = process_srtm(
    lat_min=14.5, lat_max=16.5,
    lon_min=-71.5, lon_max=-69.5,
    angle_deg=30.0,
    continent="South_America",
    out_dir="srtm_ubinas",
)

# Export to XYZ for further use
xyz = geotiff_to_xyz(dem_path, step=3, as_dataframe=True)
print(xyz.head())
```

---

## Dependencies

| Package | Role |
|---------|------|
| `numpy` | Array operations |
| `pandas` | DataFrame output (optional) |
| `rasterio` | GeoTIFF read/write, mosaicking |
| `requests` | HTTP download of SRTM tiles |
| `zipfile` | Unpack downloaded `.hgt.zip` files |

---

Author: Volker Rath (DIAS)
Generated with Copilot v1.0, 2025-11-27
