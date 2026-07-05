#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tacna_precompute_modem.py
===================
Pre-computation script for ModEM 3-D MT inversion results.

Reads a ModEM resistivity model file (.rho) and a ModEM data file (.dat),
then writes a set of UTM-km NetCDF files analogous to those produced by
tacna_precompute_seis.py for seismic tomography.  The output files are
designed to be consumed by a companion plotting script (e.g. modem_plot.py).

What this script produces
--------------------------
  modem_model_utm.nc         — full 3-D log10(ρ) model on the UTM-km mesh
  modem_topo_utm.nc          — 2-D surface topography extracted from the model
                               (z of the shallowest non-air cell per column)
  modem_sites_utm.nc         — MT site positions (easting, northing, elevation)
                               plus unique site names
  modem_rho_utm_{D}km.nc     — horizontal log10(ρ) slice at depth D km
                               (one file per entry in DEPTH_SLICES_KM)

Coordinate convention
----------------------
ModEM uses a local Cartesian mesh whose origin is the *reference point*
(lat_ref, lon_ref) stored in the last non-comment line of the .rho file.
The mesh x-axis points North, y-axis East, z-axis Down (positive downward).
Cell sizes dx, dy, dz are in metres.

This script:
  1. Builds cell-centre coordinates in local metres (North, East, Down)
     using modem.cells3d() with center=True.
  2. Projects (North, East) offsets onto absolute geographic coordinates
     via the reference point and a UTM transformer, yielding absolute
     UTM easting / northing in km on a regular grid.
  3. Saves each output as an xarray DataArray/Dataset NetCDF with
     x ↔ UTM easting (km), y ↔ UTM northing (km), z ↔ depth (km, positive down).

Helpers used from modem.py
---------------------------
  read_mod(file, modext, trans)   — reads .rho file → dx, dy, dz, mval, reference
  read_data(Datfile, modext)      — reads .dat file → Site, Comp, Data, Head
  cells3d(dx, dy, dz, center)     — cumulative cell-centre coordinates
  get_topo(dx, dy, dz, mval, ref) — extracts 2-D surface topography

Dependencies
------------
    numpy, xarray, pyproj, scipy, modem (project-local)

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic), 2026-06-29.
License: GNU General Public License v3 (GPL-3.0-or-later).
AI-generated code — review before use in production.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer

# modem.py must be on the Python path or in the working directory
try:
    import modem as mdm
except ImportError:
    sys.exit(
        "Cannot import modem.py — place it in the working directory or on PYTHONPATH."
    )

# =====================================================================
# USER SETTINGS
# =====================================================================

# --- Input files (without extension) ---
MODEL_FILE = "./mt/Tacna_final"    # reads MODEL_FILE + MODEL_EXT
DATA_FILE  = "./mt/Tacna_final"    # reads DATA_FILE  + DATA_EXT
MODEL_EXT  = ".rho"
DATA_EXT   = ".dat"

# --- Reference point (must match the model file; sign-checked at runtime) ---
# ModEM stores [lat, lon, elevation_m] in the last data line of the .rho file.
# Leave as None to read directly from the file (recommended).
REFERENCE_LAT = None   # degrees, WGS84; None → read from model file
REFERENCE_LON = None   # degrees, WGS84; None → read from model file

# --- Resistivity transform for output ---
# "LOG10"  : save log10(ρ)  [most common for visualisation]
# "LOGE"   : save ln(ρ)
# "LINEAR" : save ρ in Ω·m
OUTPUT_TRANSFORM = "LOG10"

# --- Depth slices to export as 2-D horizontal NetCDF grids (km, positive down) ---
DEPTH_SLICES_KM = [2.0, 5.0, 10.0, 20.0, 40.0]

# --- Air-cell resistivity threshold (Ω·m) used by get_topo ---
RHO_AIR = 1.0e17

# --- Padding cells to trim from each face before output ---
# ModEM models have large padding cells at the boundary that distort colour scales.
# [trim_x0, trim_x1, trim_y0, trim_y1, trim_z0] — number of cells to drop
# from the -x, +x, -y, +y faces and the top (z=0) face respectively.
# Set all to 0 to keep the full model.
TRIM_PAD = [7, 7, 7, 7, 0]

# --- UTM zone override ---
# By default the zone is inferred from REFERENCE_LON.  Set manually if needed.
UTM_ZONE     = None        # e.g. 19;  None → auto-detect
UTM_HEMI     = None        # "N" or "S"; None → infer from REFERENCE_LAT

# =====================================================================
# END USER SETTINGS
# =====================================================================


# ------------------------------------------------------------------
# UTM projection helpers
# ------------------------------------------------------------------

def _build_transformer(lat_ref: float, lon_ref: float,
                        zone_override=None, hemi_override=None):
    """Return a pyproj Transformer (WGS84 → UTM) for the reference point."""
    if zone_override is not None:
        zone = int(zone_override)
    else:
        zone = int((lon_ref + 180.0) // 6.0) + 1

    if hemi_override is not None:
        hemi = hemi_override.upper()
    else:
        hemi = "N" if lat_ref >= 0.0 else "S"

    epsg = (32600 + zone) if hemi == "N" else (32700 + zone)
    print(f"  UTM zone {zone}{hemi}  (EPSG:{epsg})")
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return transformer, epsg


def _ref_to_utm(transformer, lat_ref, lon_ref):
    """Return reference point easting/northing in km."""
    e, n = transformer.transform(lon_ref, lat_ref)
    return e / 1e3, n / 1e3


# ------------------------------------------------------------------
# Mesh coordinate builder
# ------------------------------------------------------------------

def build_utm_axes(dx, dy, reference, transformer, lat_ref, lon_ref):
    """
    Convert ModEM local-North / local-East cell-centre arrays to absolute
    UTM easting / northing (km).

    ModEM convention
    ----------------
    dx  → North direction (x-axis, index 0), metres
    dy  → East  direction (y-axis, index 1), metres
    The model centre (x=0, y=0 in local coords) corresponds to the
    geographic reference point (lat_ref, lon_ref).

    Parameters
    ----------
    dx, dy      : 1-D arrays of cell sizes in metres (after trimming)
    reference   : raw reference from read_mod (local Cartesian metres, unused here)
    transformer : pyproj Transformer WGS84→UTM
    lat_ref     : geographic latitude of model centre (degrees)
    lon_ref     : geographic longitude of model centre (degrees)

    Returns
    -------
    utm_e_km : 1-D array, UTM easting  of cell centres (km)
    utm_n_km : 1-D array, UTM northing of cell centres (km)
    """
    # Cell-centre offsets from the model centre in metres (North / East)
    x_local = np.cumsum(dx) - np.sum(dx) / 2.0   # North offsets, m
    y_local = np.cumsum(dy) - np.sum(dy) / 2.0   # East  offsets, m
    x_local -= dx / 2.0
    y_local -= dy / 2.0

    # UTM coordinates of the geographic reference point
    ref_e, ref_n = _ref_to_utm(transformer, lat_ref, lon_ref)   # km

    # North offset → +northing;  East offset → +easting
    utm_n_km = ref_n + x_local / 1e3   # shape (nx,)
    utm_e_km = ref_e + y_local / 1e3   # shape (ny,)

    return utm_e_km, utm_n_km


# ------------------------------------------------------------------
# Depth axis builder
# ------------------------------------------------------------------

def build_depth_axis_km(dz):
    """Return cell-centre depth array in km (positive down, from surface)."""
    nz = len(dz)
    z_edges = np.concatenate([[0.0], np.cumsum(dz)])
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])   # metres
    return z_centres / 1e3


# ------------------------------------------------------------------
# Apply transform to resistivity values
# ------------------------------------------------------------------

def apply_transform(mval, trans):
    """Return mval (Ω·m, physical) converted to the requested representation."""
    trans = trans.upper()
    if trans == "LOG10":
        return np.log10(np.where(mval > 0, mval, np.nan))
    elif trans in ("LOGE", "LN"):
        return np.log(np.where(mval > 0, mval, np.nan))
    elif trans == "LINEAR":
        return mval.copy()
    else:
        raise ValueError(f"Unknown OUTPUT_TRANSFORM={trans!r}")


# ------------------------------------------------------------------
# Trim padding cells
# ------------------------------------------------------------------

def trim_model(dx, dy, dz, mval, trim):
    """
    Remove padding cells from the model periphery.

    Parameters
    ----------
    trim : [tx0, tx1, ty0, ty1, tz0]
        Cells to drop from -x, +x, -y, +y, top-z faces.
    """
    tx0, tx1, ty0, ty1, tz0 = trim
    nx, ny, nz = mval.shape

    sl_x = slice(tx0, nx - tx1 if tx1 else None)
    sl_y = slice(ty0, ny - ty1 if ty1 else None)
    sl_z = slice(tz0, None)

    mval_t = mval[sl_x, sl_y, sl_z]
    dx_t   = dx[sl_x]
    dy_t   = dy[sl_y]
    dz_t   = dz[sl_z]

    print(f"  After trimming: {mval_t.shape[0]}×{mval_t.shape[1]}×{mval_t.shape[2]} cells")
    return dx_t, dy_t, dz_t, mval_t


# ------------------------------------------------------------------
# NetCDF writers
# ------------------------------------------------------------------

def save_3d_model(utm_e_km, utm_n_km, depth_km, rho_transformed, outpath):
    """Write full 3-D resistivity model to NetCDF."""
    long_name = {
        "LOG10": "log10 resistivity",
        "LOGE":  "ln resistivity",
        "LINEAR": "resistivity",
    }.get(OUTPUT_TRANSFORM.upper(), "resistivity")
    units = {
        "LOG10": "log10(Ohm.m)",
        "LOGE":  "ln(Ohm.m)",
        "LINEAR": "Ohm.m",
    }.get(OUTPUT_TRANSFORM.upper(), "Ohm.m")

    # rho_transformed has shape (nx, ny, nz) — ModEM (N, E, Down)
    # Reorder to (z, y, x) = (depth, northing, easting) for NetCDF convention
    data = np.transpose(rho_transformed, (2, 0, 1))   # (nz, nx, ny)

    da = xr.DataArray(
        data.astype(np.float32),
        dims=["depth", "northing", "easting"],
        coords={
            "depth":    xr.Variable("depth",    depth_km,
                                    attrs={"units": "km", "positive": "down",
                                           "long_name": "Depth below surface"}),
            "northing": xr.Variable("northing", utm_n_km,
                                    attrs={"units": "km",
                                           "long_name": "UTM northing"}),
            "easting":  xr.Variable("easting",  utm_e_km,
                                    attrs={"units": "km",
                                           "long_name": "UTM easting"}),
        },
        attrs={"long_name": long_name, "units": units,
               "transform": OUTPUT_TRANSFORM},
    )
    da.to_netcdf(outpath)
    print(f"  Saved: {outpath}")


def save_topo(utm_e_km, utm_n_km, topo_m, outpath):
    """
    Write 2-D surface topography (metres) to NetCDF.

    topo_m has shape (nx, ny) in ModEM (North, East) order, matching
    utm_n_km (length nx) and utm_e_km (length ny).
    ModEM z is positive downward; get_topo returns the z of the shallowest
    non-air cell face, which is 0 or negative for surface above the model
    top.  Negate to get elevation positive up.
    """
    elev_m = -topo_m   # shape (nx, ny) = (northing, easting), positive up

    da = xr.DataArray(
        elev_m.astype(np.float32),
        dims=["northing", "easting"],
        coords={
            "northing": xr.Variable("northing", utm_n_km,
                                    attrs={"units": "km",
                                           "long_name": "UTM northing"}),
            "easting":  xr.Variable("easting",  utm_e_km,
                                    attrs={"units": "km",
                                           "long_name": "UTM easting"}),
        },
        attrs={"long_name": "Surface elevation", "units": "m",
               "positive": "up",
               "note": "Derived from shallowest non-air cell in ModEM model"},
    )
    da.to_netcdf(outpath)
    print(f"  Saved: {outpath}")


def save_depth_slice(utm_e_km, utm_n_km, depth_km_axis, rho_transformed,
                     target_depth_km, outpath):
    """
    Interpolate the 3-D model to a target depth and save as 2-D NetCDF.

    Uses nearest-neighbour selection on the depth axis (no interpolation
    artefacts across large depth intervals).

    rho_transformed shape: (nx, ny, nz)  ModEM (N, E, Down)
    """
    iz = int(np.argmin(np.abs(depth_km_axis - target_depth_km)))
    actual_depth = depth_km_axis[iz]
    print(f"  Depth slice {target_depth_km} km → nearest cell centre {actual_depth:.2f} km")

    # Extract slice: shape (nx, ny) = (northing, easting) — no transpose needed
    slc = rho_transformed[:, :, iz]

    long_name = f"{OUTPUT_TRANSFORM} resistivity at {actual_depth:.1f} km depth"
    units = {
        "LOG10": "log10(Ohm.m)",
        "LOGE":  "ln(Ohm.m)",
        "LINEAR": "Ohm.m",
    }.get(OUTPUT_TRANSFORM.upper(), "Ohm.m")

    da = xr.DataArray(
        slc.astype(np.float32),
        dims=["northing", "easting"],
        coords={
            "northing": xr.Variable("northing", utm_n_km,
                                    attrs={"units": "km",
                                           "long_name": "UTM northing"}),
            "easting":  xr.Variable("easting",  utm_e_km,
                                    attrs={"units": "km",
                                           "long_name": "UTM easting"}),
        },
        attrs={
            "long_name":   long_name,
            "units":       units,
            "depth_km":    float(actual_depth),
            "target_depth_km": float(target_depth_km),
            "transform":   OUTPUT_TRANSFORM,
        },
    )
    da.to_netcdf(outpath)
    print(f"  Saved: {outpath}")


def save_sites(utm_e_km_sites, utm_n_km_sites, elev_m_sites, site_names, outpath):
    """Write MT site positions to NetCDF."""
    n = len(site_names)
    ds = xr.Dataset(
        {
            "easting":  xr.DataArray(utm_e_km_sites.astype(np.float32),
                                     dims=["site"],
                                     attrs={"units": "km",
                                            "long_name": "UTM easting"}),
            "northing": xr.DataArray(utm_n_km_sites.astype(np.float32),
                                     dims=["site"],
                                     attrs={"units": "km",
                                            "long_name": "UTM northing"}),
            "elevation": xr.DataArray(elev_m_sites.astype(np.float32),
                                      dims=["site"],
                                      attrs={"units": "m",
                                             "long_name": "Site elevation",
                                             "positive": "up"}),
            "name":     xr.DataArray(np.array(site_names, dtype=object),
                                     dims=["site"],
                                     attrs={"long_name": "Site name"}),
        }
    )
    ds.to_netcdf(outpath)
    print(f"  Saved: {outpath}  ({n} sites)")


# ==================================================================
# Main
# ==================================================================

# ------------------------------------------------------------------
# 1. Read model
# ------------------------------------------------------------------
print("\n=== Reading ModEM model ===")
dx, dy, dz, mval, reference, trans_in = mdm.read_mod(
    file=MODEL_FILE, modext=MODEL_EXT, trans="LINEAR", out=True
)
# mval is now in physical Ω·m, shape (nx, ny, nz)

# Read the data file to get the geographic reference point and site coordinates.
# The .dat header line "> lat lon" gives the model origin in geographic coords.
# Data columns: Period Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error
#   col 2 = GG_Lat (°), col 3 = GG_Lon (°)
#   col 4 = X (m, North), col 5 = Y (m, East), col 6 = Z (m, positive down)
Site, Comp, Data, Head = mdm.read_data(Datfile=DATA_FILE, modext=DATA_EXT, out=True)

# Extract geographic reference from the "> lat lon" header line
_ref_line = [l for l in Head if l.startswith(">") and
             len(l.split()) == 3 and
             not any(c.isalpha() for c in l.replace(".", "").replace("-", "").replace(">", "").strip())]
if _ref_line and REFERENCE_LAT is None:
    _parts = _ref_line[0].split()
    lat_ref = float(_parts[1])
    lon_ref = float(_parts[2])
    print(f"  Geographic reference from .dat header: lat={lat_ref:.4f}°  lon={lon_ref:.4f}°")
else:
    lat_ref = REFERENCE_LAT if REFERENCE_LAT is not None else float(np.mean(Data[:, 1]))
    lon_ref = REFERENCE_LON if REFERENCE_LON is not None else float(np.mean(Data[:, 2]))
    print(f"  Geographic reference (override/fallback): lat={lat_ref:.4f}°  lon={lon_ref:.4f}°")

# ------------------------------------------------------------------
# 2. UTM transformer
# ------------------------------------------------------------------
print("\n=== Setting up UTM projection ===")
transformer, epsg = _build_transformer(lat_ref, lon_ref, UTM_ZONE, UTM_HEMI)

# ------------------------------------------------------------------
# 3. Trim padding
# ------------------------------------------------------------------
print("\n=== Trimming padding cells ===")
print(f"  Raw model: {mval.shape[0]}×{mval.shape[1]}×{mval.shape[2]} cells")
dx, dy, dz, mval = trim_model(dx, dy, dz, mval, TRIM_PAD)

# ------------------------------------------------------------------
# 4. Build coordinate axes
# ------------------------------------------------------------------
print("\n=== Building UTM coordinate axes ===")
utm_e_km, utm_n_km = build_utm_axes(dx, dy, reference, transformer,
                                     lat_ref, lon_ref)
depth_km = build_depth_axis_km(dz)

print(f"  Easting  range: {utm_e_km.min():.1f} – {utm_e_km.max():.1f} km")
print(f"  Northing range: {utm_n_km.min():.1f} – {utm_n_km.max():.1f} km")
print(f"  Depth    range: {depth_km.min():.2f} – {depth_km.max():.1f} km")

# ------------------------------------------------------------------
# 5. Apply output transform
# ------------------------------------------------------------------
print(f"\n=== Applying output transform: {OUTPUT_TRANSFORM} ===")
rho_out = apply_transform(mval, OUTPUT_TRANSFORM)

# ------------------------------------------------------------------
# 6. Topography from model
# ------------------------------------------------------------------
print("\n=== Extracting model topography ===")
# get_topo expects physical mval (Ω·m) and reference in metres
xcnt, ycnt, topo_m = mdm.get_topo(
    dx=dx, dy=dy, dz=dz, mval=mval,
    ref=[0., 0., reference[2]],    # x/y are relative; z ref is elevation
    mvalair=RHO_AIR, out=True,
)
# xcnt/ycnt are local offsets in metres, matching dx/dy after trimming.
# We use the already-built UTM axes instead.
save_topo(utm_e_km, utm_n_km, topo_m, "modem_topo_utm.nc")

# ------------------------------------------------------------------
# 7. Full 3-D model
# ------------------------------------------------------------------
print("\n=== Saving 3-D model ===")
save_3d_model(utm_e_km, utm_n_km, depth_km, rho_out, "modem_model_utm.nc")

# ------------------------------------------------------------------
# 8. Depth slices
# ------------------------------------------------------------------
print("\n=== Saving depth slices ===")
for d_km in DEPTH_SLICES_KM:
    tag = f"{d_km:.0f}km" if d_km == int(d_km) else f"{d_km:.1f}km"
    save_depth_slice(utm_e_km, utm_n_km, depth_km, rho_out,
                     d_km, f"modem_rho_utm_{tag}.nc")

# ------------------------------------------------------------------
# 9. Site positions from data file
# ------------------------------------------------------------------
print("\n=== Saving MT site positions ===")
# Site, Comp, Data, Head already loaded above.
# Use GG_Lat/GG_Lon (cols 1,2) for site geographic positions, and
# X(m)/Y(m)/Z(m) (cols 4,5,6) as the model Cartesian coordinates.
# Elevation: Z(m) is positive down in ModEM; negate for positive-up.
_, unique_idx = np.unique(Site, return_index=True)
unique_names  = Site[unique_idx]
site_lats     = Data[unique_idx, 1]    # GG_Lat, degrees
site_lons     = Data[unique_idx, 2]    # GG_Lon, degrees
site_elevs    = -Data[unique_idx, 6]   # Z(m) positive down → negate for positive up

# Project geographic coords to UTM km
site_e_raw, site_n_raw = transformer.transform(site_lons, site_lats)
site_e_km = site_e_raw / 1e3
site_n_km = site_n_raw / 1e3

save_sites(site_e_km, site_n_km, site_elevs, unique_names.tolist(),
           "modem_sites_utm.nc")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n=== Done — output files ===")
outputs = [
    "modem_topo_utm.nc",
    "modem_model_utm.nc",
    "modem_sites_utm.nc",
] + [
    "modem_rho_utm_{}.nc".format(
        f"{d:.0f}km" if d == int(d) else f"{d:.1f}km"
    )
    for d in DEPTH_SLICES_KM
]
for f in outputs:
    print(f"  {f}")
