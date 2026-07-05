#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tacna_precompute_seis.py
=======================
Pre-computation script for tacna_plot_seis.py (GMT-free pipeline).

Produces the same UTM-km NetCDF files as tacna_precompute_gmt.py except:
  * topography/bathymetry is fetched via the `elevation` package (SRTM/ETOPO)
    instead of pygmt.datasets.load_earth_relief — no GMT required.
  * tacna_topo_shade_utm.nc is NOT written; hillshade is computed on-the-fly
    inside tacna_plot_seis.py using matplotlib.colors.LightSource.
  * tacna_svelocity_subset.nc and tacna_psratio_subset.nc are still written
    for completeness but are not read by tacna_plot_seis.py.

Output files consumed by tacna_plot_seis.py
-------------------------------------------
  tacna_topo_utm.nc             — topography on UTM-km grid
  tacna_bath_utm.nc             — bathymetry mask (topo ≤ 0) on UTM-km grid
  tacna_pvelocity_subset.nc     — Vp subset (depth coord + UTM aux coords)
  tacna_vp_utm_{tag}.nc  }
  tacna_vs_utm_{tag}.nc  }      — per-depth UTM-km velocity slices
  tacna_vps_utm_{tag}.nc }

Topography back-end
-------------------
Requires the `elevation` package (pip install elevation) and its cli tool
`eio`, which downloads SRTM 30 m (land) / ETOPO1 (ocean) tiles on first use
and caches them locally.  Alternatively, set TOPO_SOURCE = "etopo" to use
the ETOPO1 global relief NetCDF directly (see USER SETTINGS).

Dependencies
------------
    numpy, xarray, pandas, pyproj, scipy, rioxarray, elevation (eio)
    — OR — numpy, xarray, pyproj, scipy, rioxarray  (with a local GeoTIFF)

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic), 2026-06-29.
License: GNU General Public License v3 (GPL-3.0-or-later).
AI-generated code — review before use in production.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

# =====================================================================
# USER SETTINGS
# =====================================================================

# Input velocity model files
FNAME_VP = "./seistomo/FD_vp_model.nc"
FNAME_VS = "./seistomo/FD_vs_model.nc"

# Velocity subset geographic bounds
TAR_LON = [-70.46, -69.566]
TAR_LAT = [-18.25, -17.0867]
DEPTH_RANGE = [0, 100]          # km

# Depth indices to export as per-depth UTM-km slices
DEPTH_INDEX = [5, 9, 13]

# Topo/bath geographic bounds (slightly wider than velocity subset)
MAP_LON = [-70.798043, -69.46973]
MAP_LAT = [-18.3423,   -17.0034]

# Output UTM-km grid spacing (km)
TOPO_SPACING_KM = 1.0

# Topography source:
#   "elevation" — use the `elevation` Python package (SRTM + ETOPO tiles)
#   "etopo"     — read a local ETOPO1/ETOPO2022 NetCDF file (set ETOPO_PATH)
#   "geotiff"   — read a local GeoTIFF (set GEOTIFF_PATH)
TOPO_SOURCE = "elevation"

ETOPO_PATH   = ""    # path to local ETOPO NetCDF  (used when TOPO_SOURCE="etopo")
GEOTIFF_PATH = ""    # path to local GeoTIFF       (used when TOPO_SOURCE="geotiff")

# =====================================================================
# END USER SETTINGS
# =====================================================================


# ------------------------------------------------------------------
# UTM projection (Zone 19S, EPSG:32719)
# ------------------------------------------------------------------
_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
_to_geo = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)


# ------------------------------------------------------------------
# UTM coordinate helpers
# ------------------------------------------------------------------
def add_utm_coords(ds):
    """
    Add 2D UTM easting/northing arrays (metres, EPSG:32719) as auxiliary
    coordinates to a Dataset that has 1D 'lat' and 'lon' dim coordinates.
    """
    lons = ds["lon"].values
    lats = ds["lat"].values
    lon2d, lat2d = np.meshgrid(lons, lats)
    easting, northing = _to_utm.transform(lon2d, lat2d)
    ds = ds.assign_coords(
        utm_easting=(
            ("lat", "lon"), easting,
            {"long_name": "UTM easting (Zone 19S)", "units": "m",
             "grid_mapping": "crs", "crs": "EPSG:32719"},
        ),
        utm_northing=(
            ("lat", "lon"), northing,
            {"long_name": "UTM northing (Zone 19S)", "units": "m",
             "grid_mapping": "crs", "crs": "EPSG:32719"},
        ),
    )
    return ds


def geo_to_utm_km_nc(da, outpath, spacing_km=1.0):
    """
    Reproject a geographic (lon/lat) DataArray to a regular UTM Zone 19S
    grid (km) via bilinear interpolation and write to NetCDF.

    Parameters
    ----------
    da         : xr.DataArray with 'lat'/'lon' (or 'y'/'x') dim coordinates
    outpath    : output NetCDF path
    spacing_km : output grid spacing in km
    """
    lat_dim = "lat" if "lat" in da.dims else "y"
    lon_dim = "lon" if "lon" in da.dims else "x"

    lats   = da[lat_dim].values.copy()
    lons   = da[lon_dim].values.copy()
    values = da.values.copy().astype(float)

    # RegularGridInterpolator requires strictly ascending axes
    if lats[0] > lats[-1]:
        lats   = lats[::-1]
        values = values[::-1, :]
    if lons[0] > lons[-1]:
        lons   = lons[::-1]
        values = values[:, ::-1]

    interp = RegularGridInterpolator(
        (lats, lons), values, method="linear",
        bounds_error=False, fill_value=np.nan,
    )

    # UTM extent from the four geographic corners
    corner_lons = [lons[0],  lons[-1], lons[0],  lons[-1]]
    corner_lats = [lats[0],  lats[0],  lats[-1], lats[-1]]
    ce, cn = _to_utm.transform(corner_lons, corner_lats)
    ce = np.asarray(ce) / 1e3
    cn = np.asarray(cn) / 1e3

    e_1d = np.arange(np.floor(ce.min()), np.ceil(ce.max()) + spacing_km, spacing_km)
    n_1d = np.arange(np.floor(cn.min()), np.ceil(cn.max()) + spacing_km, spacing_km)

    E2d, N2d     = np.meshgrid(e_1d, n_1d)
    lon2d, lat2d = _to_geo.transform(E2d * 1e3, N2d * 1e3)
    pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    out = interp(pts).reshape(E2d.shape).astype(np.float32)

    result = xr.DataArray(
        out,
        dims=["y", "x"],
        coords={
            "y": xr.Variable("y", n_1d,
                             attrs={"units": "km", "long_name": "Northing (UTM 19S)"}),
            "x": xr.Variable("x", e_1d,
                             attrs={"units": "km", "long_name": "Easting (UTM 19S)"}),
        },
        attrs=da.attrs,
    )
    result.to_netcdf(outpath)
    print(f"  Saved: {outpath}")


def slice_to_utm_km_nc(da, outpath):
    """
    Convert a 2D (lat, lon) velocity DataArray that carries utm_easting /
    utm_northing auxiliary coordinates (metres) to a Cartesian UTM-km NetCDF.

    Uses the middle row/column of the 2D UTM arrays as 1D axes
    (small-region approximation; error < 0.1 % over ~150 km E-W).
    """
    e2d = da["utm_easting"].values    # metres, shape (nlat, nlon)
    n2d = da["utm_northing"].values

    mid_lat = e2d.shape[0] // 2
    mid_lon = n2d.shape[1] // 2

    e_1d = e2d[mid_lat, :] / 1e3     # km
    n_1d = n2d[:, mid_lon] / 1e3     # km

    result = xr.DataArray(
        da.values.astype(np.float32),
        dims=["y", "x"],
        coords={
            "y": xr.Variable("y", n_1d,
                             attrs={"units": "km", "long_name": "Northing (UTM 19S)"}),
            "x": xr.Variable("x", e_1d,
                             attrs={"units": "km", "long_name": "Easting (UTM 19S)"}),
        },
        attrs=da.attrs,
    )
    result.to_netcdf(outpath)
    print(f"  Saved: {outpath}")


# ------------------------------------------------------------------
# Topography loader (no GMT)
# ------------------------------------------------------------------
def load_topo_geographic(lon_range, lat_range):
    """
    Return a (lats, lons, values_metres) tuple for the requested region
    using the backend selected by TOPO_SOURCE.

    'elevation' backend
        Uses the `elevation` CLI tool (eio) to clip SRTM/ETOPO tiles to a
        temporary GeoTIFF, then reads it with rioxarray.  Tiles are cached
        in ~/elevation by default on first use.

    'etopo' backend
        Reads a local ETOPO1/ETOPO2022 NetCDF (set ETOPO_PATH).

    'geotiff' backend
        Reads a local GeoTIFF (set GEOTIFF_PATH) via rioxarray.
    """
    lon0, lon1 = lon_range
    lat0, lat1 = lat_range

    if TOPO_SOURCE == "elevation":
        try:
            import rioxarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "rioxarray is required for TOPO_SOURCE='elevation'. "
                "Install with: pip install rioxarray"
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tif = str(Path(tmpdir) / "topo.tif")
            bounds  = f"{lon0} {lat0} {lon1} {lat1}"
            cmd = [
                "eio", "clip",
                "-o", out_tif,
                "--bounds", str(lon0), str(lat0), str(lon1), str(lat1),
            ]
            print(f"  Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            import rioxarray
            da = rioxarray.open_rasterio(out_tif).squeeze()
            # rioxarray uses x=lon, y=lat
            lons = da["x"].values
            lats = da["y"].values
            vals = da.values.astype(float)
            # Ensure ascending lat
            if lats[0] > lats[-1]:
                lats = lats[::-1]
                vals = vals[::-1, :]
        return lats, lons, vals

    elif TOPO_SOURCE == "etopo":
        if not ETOPO_PATH:
            raise ValueError("Set ETOPO_PATH when TOPO_SOURCE='etopo'.")
        ds = xr.open_dataset(ETOPO_PATH)
        # Common ETOPO variable names: 'z', 'Band1', 'elevation'
        var = [v for v in ds.data_vars][0]
        da  = ds[var]
        lat_dim = "lat" if "lat" in da.dims else "y"
        lon_dim = "lon" if "lon" in da.dims else "x"
        lats = da[lat_dim].values
        lons = da[lon_dim].values
        # Subset to region
        lat_mask = (lats >= lat0) & (lats <= lat1)
        lon_mask = (lons >= lon0) & (lons <= lon1)
        lats = lats[lat_mask]
        lons = lons[lon_mask]
        vals = da.values[np.ix_(lat_mask, lon_mask)].astype(float)
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            vals = vals[::-1, :]
        return lats, lons, vals

    elif TOPO_SOURCE == "geotiff":
        if not GEOTIFF_PATH:
            raise ValueError("Set GEOTIFF_PATH when TOPO_SOURCE='geotiff'.")
        try:
            import rioxarray
        except ImportError:
            raise ImportError("rioxarray is required for TOPO_SOURCE='geotiff'.")
        da   = rioxarray.open_rasterio(GEOTIFF_PATH).squeeze()
        lons = da["x"].values
        lats = da["y"].values
        vals = da.values.astype(float)
        # Clip to region
        lon_mask = (lons >= lon0) & (lons <= lon1)
        lat_mask = (lats >= lat0) & (lats <= lat1)
        lons = lons[lon_mask]
        lats = lats[lat_mask]
        vals = vals[np.ix_(lat_mask, lon_mask)]
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            vals = vals[::-1, :]
        return lats, lons, vals

    else:
        raise ValueError(
            f"Unknown TOPO_SOURCE={TOPO_SOURCE!r}. "
            "Choose 'elevation', 'etopo', or 'geotiff'."
        )


# ==================================================================
# Main
# ==================================================================

left,  right  = TAR_LON
bottom, top   = TAR_LAT

# ------------------------------------------------------------------
# 1. Velocity subsets
# ------------------------------------------------------------------
print("Reading velocity models …")
vtomop = xr.open_dataset(FNAME_VP)
vtomos = xr.open_dataset(FNAME_VS)

vp  = vtomop.sel(lat=slice(bottom, top), lon=slice(left, right),
                 depth=slice(*DEPTH_RANGE))
vs  = vtomos.sel(lat=slice(bottom, top), lon=slice(left, right),
                 depth=slice(*DEPTH_RANGE))
vps = vp / vs

vp  = add_utm_coords(vp)
vs  = add_utm_coords(vs)
vps = add_utm_coords(vps)

vp.to_netcdf("tacna_vp.nc")
vs.to_netcdf("tacna_vs.nc")
vps.to_netcdf("tacna_vps.nc")
print("Saved velocity subsets:")
print("  tacna_vp.nc")
print("  tacna_vs.nc")
print("  tacna_vps.nc")

# ------------------------------------------------------------------
# 2. Topography and bathymetry (no GMT)
# ------------------------------------------------------------------
print(f"\nLoading topography via TOPO_SOURCE='{TOPO_SOURCE}' …")
topo_lats, topo_lons, topo_vals = load_topo_geographic(MAP_LON, MAP_LAT)

topo_da = xr.DataArray(
    topo_vals.astype(np.float32),
    dims=["lat", "lon"],
    coords={"lat": topo_lats, "lon": topo_lons},
    attrs={"long_name": "Elevation", "units": "m"},
)

# Bathymetry: keep only sub-zero cells
bath_vals = np.where(topo_vals <= 0, topo_vals, np.nan).astype(np.float32)
bath_da = xr.DataArray(
    bath_vals,
    dims=["lat", "lon"],
    coords={"lat": topo_lats, "lon": topo_lons},
    attrs={"long_name": "Bathymetry", "units": "m"},
)

print("\nReprojecting topo/bath to UTM-km grids …")
geo_to_utm_km_nc(topo_da, "tacna_topo_utm.nc", spacing_km=TOPO_SPACING_KM)
geo_to_utm_km_nc(bath_da, "tacna_bath_utm.nc", spacing_km=TOPO_SPACING_KM)

# Note: tacna_topo_shade_utm.nc is intentionally NOT written here.
# Hillshade is computed on-the-fly in tacna_plot_seis.py via
# matplotlib.colors.LightSource, which gives equivalent results.

# ------------------------------------------------------------------
# 3. Per-depth UTM-km velocity slices
# ------------------------------------------------------------------
depth_coord = vp["depth"]

print("\nPre-computing per-depth UTM-km velocity slices …")
for d_index in DEPTH_INDEX:
    depth_km = int(depth_coord.item(d_index))
    tag = f"{depth_km}km"
    slice_to_utm_km_nc(vp["data"].isel(depth=d_index),  f"tacna_vp_utm_{tag}.nc")
    slice_to_utm_km_nc(vs["data"].isel(depth=d_index),  f"tacna_vs_utm_{tag}.nc")
    slice_to_utm_km_nc(vps["data"].isel(depth=d_index), f"tacna_vps_utm_{tag}.nc")

print("\nDone. All UTM-km grids ready for tacna_plot_seis.py.")
