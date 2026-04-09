#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
netcdf2vtk.py
====================

Read P- and S-wave velocity grids (NetCDF/.grd), crop to a sub-volume around
Ubinas volcano, compute Vp/Vs, save the subset as NetCDF, and export VTK files
for 3-D visualisation.

Two VTK export modes are available:

    geo  — RectilinearGrid in geographic coordinates (lon, lat, depth/km).
           Straightforward; axes are lon/lat/depth as in the source files.

    utm  — StructuredGrid reprojected to UTM (EPSG:32719 by default).
           Coordinates are shifted to an anchor point so that the origin is
           near the volcano.  Depth is converted from km to m.

Provenance
----------
Original author : Svetlana Byrdina (2024-07-18)
Merged/extended : 2025-07-18 — combined ubinas_netcdf2vtk.py (geographic mode)
                  and ubinas_netcdf2vtk_UTM.py (UTM mode) into a single script;
                  added ``USE_UTM`` flag, ``utm_epsg`` / anchor parameters,
                  and this docstring.  Logic unchanged.

Usage
-----
Set ``USE_UTM = True`` (or ``False``) in the USER CONFIG section below, adjust
the file paths and region bounds, then run::

    python ubinas_netcdf2vtk.py
"""

import numpy as np
import xarray as xr
import pandas as pd
import pyvista as pv
from pyproj import Transformer, CRS

# =============================================================================
# USER CONFIG
# =============================================================================

# --- Output mode ---
USE_UTM = True          # True → UTM StructuredGrid; False → geographic RectilinearGrid

# --- UTM options (only used when USE_UTM = True) ---
EPSG = 32719
crs = CRS.from_epsg(32719)
print(crs)
print(crs.area_of_use)

UTM_EPSG    = "EPSG:"+str(EPSG)   # UTM zone 19S (southern Peru)
ANCHOR_LON  = -70.868069     # subtracted from Easting  [m]
ANCHOR_LAT  = -16.367828     # subtracted from Northing [m]
ANCHOR_X_M  = 300472         # pre-computed UTM easting  of anchor [m]
ANCHOR_Y_M  = 8189946        # pre-computed UTM northing of anchor [m]

# --- Region of interest ---
LEFT   = -71.7130
RIGHT  = -70.023
TOP    = -15.55
BOTTOM = -17.18

LATLIMS = [BOTTOM, TOP]
LONLIMS = [LEFT, RIGHT]
ZLIMS   = [-100, 100]        # depth range in km (positive = up in source grid)

# --- Input files ---
FNAME_P = "/home/sbyrd/Desktop/PEROU/CHEVROT/Chevrot_orig/velocity_model_P.grd"
FNAME_S = "/home/sbyrd/Desktop/PEROU/CHEVROT/Chevrot_orig/velocity_model_S.grd"

# --- Station list (optional; loaded but not used in VTK export) ---
STATION_CSV = "./Features/done/Ubinas_Sitelist.csv"

# --- Output file names ---
NETCDF_OUT = "ubinas_velocity_subset.nc"
VTK_SUFFIX = "_utm" if USE_UTM else ""

VTK_VP  = f"ubinas_vp_subset{VTK_SUFFIX}.vtk"
VTK_VS  = f"ubinas_vs_subset{VTK_SUFFIX}.vtk"
VTK_VPS = f"ubinas_vps_subset{VTK_SUFFIX}.vtk"


# =============================================================================
# HELPERS
# =============================================================================

def ordered_slice(coord, vmin, vmax):
    """Return a slice that works for both ascending and descending coordinates."""
    coord = np.asarray(coord)
    return slice(vmin, vmax) if coord[0] < coord[-1] else slice(vmax, vmin)


def save_rectilinear_vtk(data_array, vtk_name, field_name):
    """Export a DataArray as a geographic-coordinate RectilinearGrid VTK file.

    Parameters
    ----------
    data_array : xr.DataArray
        3-D array with dimensions ``x`` (lon), ``y`` (lat), ``z`` (depth/km).
    vtk_name : str
        Output file path.
    field_name : str
        Name of the scalar field stored in the VTK file.
    """
    x = data_array["x"].values
    y = data_array["y"].values
    z = data_array["z"].values

    values_xyz = data_array.transpose("x", "y", "z").values

    grid = pv.RectilinearGrid(x, y, z)
    grid.point_data[field_name] = np.ascontiguousarray(values_xyz).ravel(order="F")
    grid.save(vtk_name)
    print(f"Saved: {vtk_name}")


def save_structured_vtk_utm(data_array, vtk_name, field_name,
                             utm_epsg=UTM_EPSG,
                             anchor_lon=ANCHOR_LON, anchor_lat=ANCHOR_LAT,
                             anchor_x_m=ANCHOR_X_M, anchor_y_m=ANCHOR_Y_M):
    """Export a DataArray as a UTM-projected StructuredGrid VTK file.

    Geographic coordinates (lon, lat) are reprojected to the requested UTM
    zone.  An anchor point is subtracted so that the grid origin sits near the
    volcano.  Depth (km in source) is converted to metres.

    Parameters
    ----------
    data_array : xr.DataArray
        3-D array with dimensions ``x`` (lon), ``y`` (lat), ``z`` (depth/km).
    vtk_name : str
        Output file path.
    field_name : str
        Name of the scalar field stored in the VTK file.
    utm_epsg : str
        EPSG code string for the target UTM projection.
    anchor_lon, anchor_lat : float
        Geographic coordinates of the origin anchor.
    anchor_x_m, anchor_y_m : float
        Pre-computed UTM easting / northing of the anchor [m].
    """
    lon = data_array["x"].values
    lat = data_array["y"].values
    z   = data_array["z"].values

    values_xyz = data_array.transpose("x", "y", "z").values

    transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)

    LON, LAT, Z = np.meshgrid(lon, lat, z, indexing="ij")
    Xutm0, Yutm0 = transformer.transform(LON, LAT)

    Xutm = Xutm0 - anchor_x_m
    Yutm = Yutm0 - anchor_y_m
    Zutm = Z * 1000.0          # km → m

    grid = pv.StructuredGrid(Yutm, Xutm, Zutm)
    grid.point_data[field_name] = np.ascontiguousarray(values_xyz).ravel(order="F")
    grid.field_data["crs"] = np.array([utm_epsg])

    print(f"  Grid dimensions : {grid.dimensions}")
    print(f"  X range (m)     : {Xutm.min():.1f}  {Xutm.max():.1f}")
    print(f"  Y range (m)     : {Yutm.min():.1f}  {Yutm.max():.1f}")
    print(f"  Z range (m)     : {Zutm.min():.1f}  {Zutm.max():.1f}")

    grid.save(vtk_name)
    print(f"Saved: {vtk_name}")


# =============================================================================
# MAIN
# =============================================================================

# --- Read input grids ---
vtomop = xr.open_dataset(FNAME_P)
vtomos = xr.open_dataset(FNAME_S)

vp3d = vtomop["velocity"].rename("vp")
vs3d = vtomos["velocity"].rename("vs")

# Station list (informational; not written to VTK)
ubinas    = pd.read_csv(STATION_CSV, delimiter=" ")
lon_ub    = ubinas["x"]
lat_ub    = ubinas["y"]

# --- Crop to requested sub-volume ---
xs = ordered_slice(vp3d["x"].values, LONLIMS[0], LONLIMS[1])
ys = ordered_slice(vp3d["y"].values, LATLIMS[0], LATLIMS[1])
zs = ordered_slice(vp3d["z"].values, ZLIMS[0],   ZLIMS[1])

vp_sub  = vp3d.sel(x=xs, y=ys, z=zs)
vs_sub  = vs3d.sel(x=xs, y=ys, z=zs)
vps_sub = (vp_sub / vs_sub).rename("vps")

vp_sub.attrs["long_name"]  = "P-wave velocity"
vs_sub.attrs["long_name"]  = "S-wave velocity"
vps_sub.attrs["long_name"] = "Vp/Vs ratio"

# Drop all-NaN edge slices
for dim in ["x", "y", "z"]:
    vp_sub  = vp_sub.dropna(dim=dim,  how="all")
    vs_sub  = vs_sub.dropna(dim=dim,  how="all")
    vps_sub = vps_sub.dropna(dim=dim, how="all")

print("Subset shape     :", vps_sub.shape)
print("Longitude range  :", float(vps_sub.x.min()), float(vps_sub.x.max()))
print("Latitude  range  :", float(vps_sub.y.min()), float(vps_sub.y.max()))
print("Depth     range  :", float(vps_sub.z.min()), float(vps_sub.z.max()))

# --- Save NetCDF ---
ds_out = xr.Dataset({"vp": vp_sub, "vs": vs_sub, "vps": vps_sub})
ds_out.to_netcdf(NETCDF_OUT)
print(f"Saved: {NETCDF_OUT}")

# --- Save VTK files ---
if USE_UTM:
    print(f"\nExporting UTM StructuredGrid VTK ({UTM_EPSG}) ...")
    save_structured_vtk_utm(vp_sub,  VTK_VP,  "vp")
    save_structured_vtk_utm(vs_sub,  VTK_VS,  "vs")
    save_structured_vtk_utm(vps_sub, VTK_VPS, "vps")
else:
    print("\nExporting geographic RectilinearGrid VTK ...")
    save_rectilinear_vtk(vp_sub,  VTK_VP,  "vp")
    save_rectilinear_vtk(vs_sub,  VTK_VS,  "vs")
    save_rectilinear_vtk(vps_sub, VTK_VPS, "vps")

print("\nDone.")
