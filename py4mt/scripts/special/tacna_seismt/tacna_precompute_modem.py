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
  modem_sens_utm.nc          — full 3-D sensitivity/resolution field (optional,
                               same mesh; only if USE_SENSITIVITY and the
                               .sns file is found), for shading/blanking
                               poorly-resolved regions in the plot script
  modem_topo_utm.nc          — 2-D surface topography extracted from the model
                               (z of the shallowest non-air cell per column)
  modem_sites_utm.nc         — MT site positions (easting, northing, elevation)
                               plus unique site names
  modem_rho_utm_{D}km.nc     — horizontal log10(ρ) slice at depth D km
                               (one file per entry in DEPTH_SLICES_KM)
  modem_sens_utm_{D}km.nc    — matching sensitivity slice at depth D km
                               (optional, see above)

Coordinate convention
----------------------
ModEM uses a local Cartesian mesh whose origin is the *reference point*
(lat_ref, lon_ref) stored in the last non-comment line of the .rho file.
The mesh x-axis points North, y-axis East, z-axis Down (positive downward).
Cell sizes dx, dy, dz are in metres.

Region of interest
-------------------
TRIM_PAD drops a fixed number of padding cells from each face, but ModEM
padding cells grow geometrically toward the model boundary, so a small
TRIM_PAD count can still leave a domain far larger than the area actually
of interest. If CROP_TO_REGION is True (default), the trimmed grid is
further cropped to the geographic box TAR_LON/TAR_LAT — the same convention
used in tacna_precompute_seis.py — so the resulting map matches the seismic
tomography region rather than the full (trimmed) model extent.

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

# Directory for all NetCDF outputs written by this script (created if it
# doesn't exist). Default "." keeps everything in the current directory,
# matching the previous (fixed) behaviour. tacna_plot_modem_image.py /
# tacna_plot_modem_mesh.py have a matching NC_DIR setting to read from
# wherever this is pointed at.
# OUTPUT_DIR = "."
OUTPUT_DIR = "./precompute/"

# --- Input files (without extension) ---
MODEL_FILE = "./mt/Tacna_final"    # reads MODEL_FILE + MODEL_EXT
DATA_FILE  = "./mt/Tacna_final"    # reads DATA_FILE  + DATA_EXT
MODEL_EXT  = ".rho"
DATA_EXT   = ".dat"

# --- Sensitivity/resolution file (optional, for shading/blanking) ---
# Same grid format as the .rho model file — read with the same reader
# (mdm.read_mod), so it must share the .rho file's mesh (dx/dy/dz, cell
# counts). Typically shares its base name too, but can be set separately.
# Set USE_SENSITIVITY = False to skip reading/writing it entirely.
USE_SENSITIVITY = True
# SENS_FILE = MODEL_FILE      # base name (without extension)
SENS_FILE = "/home/vrath/MT_Data/Tacna/TAC_30_JAC/TAC30_nerr_sp-8_sens_cov_max/TAC30_nerr_sp-8_total_cov_max"
SENS_EXT  = ".sns"
# "LOG10" is usually more useful for sensitivity, which commonly spans many
# orders of magnitude; "LINEAR" keeps raw values as stored in the file.
SENS_TRANSFORM = "LOG10"

# The consistency check below only compares *shape* and *cell sizes*
# between the .sns and .rho meshes — it can't detect an axis that's simply
# stored back-to-front, since a reversed cell-width array can still have
# identical shape and (if padding is roughly symmetric) still pass an
# allclose comparison on cell sizes. If your sensitivity file comes from a
# different tool than the .rho model and ends up mirrored east-west or
# north-south relative to the resistivity model (e.g. compared against a
# GeoTools/other-software rendering of the same file), flip the relevant
# axis here rather than in the plot script, so every downstream product
# (3-D field, depth slices) is corrected once, consistently.
SENS_FLIP_EASTING  = False  # this file doesn't need an East-West flip
SENS_FLIP_NORTHING = True  # empirically confirmed against real station
                              # positions for TAC_G2_ZT1_nerr_sp-8_Dtype_
                              # zfull_sqr_max.sns — a different .sns file
                              # may need different settings; re-validate
                              # if you switch files again.

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
# Must match DEPTH_SLICES_KM in tacna_plot_modem_mesh.py — both the resistivity
# (modem_rho_utm_{tag}.nc) and sensitivity (modem_sens_utm_{tag}.nc) depth
# slices are written from this one list, in the same loop (see "8. Depth
# slices" below), so keeping this in sync automatically keeps sensitivity
# covering the same depths as resistivity.
DEPTH_SLICES_KM = [1., 5.0, 9.0]

# --- Air-cell resistivity threshold (Ω·m) used by get_topo ---
RHO_AIR = 1.0e17

# --- Padding cells to trim from each face before output ---
# ModEM models have large padding cells at the boundary that distort colour scales.
# [trim_x0, trim_x1, trim_y0, trim_y1, trim_z0] — number of cells to drop
# from the -x, +x, -y, +y faces and the top (z=0) face respectively.
# Set all to 0 to keep the full model.
TRIM_PAD = [7, 7, 7, 7, 0]

# --- Geographic region of interest ---
# TRIM_PAD only drops a fixed number of cells and typically still leaves a
# domain far larger than the area of interest (ModEM padding cells grow
# geometrically toward the boundary). Unlike tacna_precompute_seis.py, which
# crops its velocity subset directly to TAR_LON/TAR_LAT, the model grid here
# was previously left at the full (trimmed) extent, so the plotted map did
# not match the seismic-tomography region.
#
# Set CROP_TO_REGION = True to crop the trimmed grid to this geographic box
# before any NetCDF output is written. Set to False to keep the full
# trimmed extent.
#
# IMPORTANT: this box must fully contain every VSLICES profile endpoint
# defined in tacna_plot_modem_image.py (PROFILE_CD_LON/LAT etc.), not just the
# seis-script's own region — the two are defined independently and can
# silently drift apart. profile_CD's endpoints, [-70.476, -18.255] and
# [-69.499, -17.048], both fell *outside* the box below on every side when
# it was copied straight from tacna_precompute_seis.py, leaving no model
# data at all near either end of the profile (a white gap at both edges of
# the section, unrelated to the depth-axis fix). Widened with margin here;
# if you add profiles reaching further, widen this to match.
#
# Kept identical to TAR_LON/TAR_LAT in tacna_precompute_seis.py so both
# pipelines cover the same geographic area — the union of the ModEM box
# [-70.55, -69.40] x [-18.35, -16.95] and the seismic box
# [-70.79, -69.48] x [-18.34, -17.01], padded by ~0.05°.
CROP_TO_REGION = True
TAR_LON = [-70.84, -69.35]
TAR_LAT = [-18.40, -16.90]

# --- UTM zone override ---
# By default the zone is inferred from REFERENCE_LON.  Set manually if needed.
UTM_ZONE     = None        # e.g. 19;  None → auto-detect
UTM_HEMI     = None        # "N" or "S"; None → infer from REFERENCE_LAT

# =====================================================================
# END USER SETTINGS
# =====================================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def outpath(name):
    """Join a bare output filename onto OUTPUT_DIR."""
    return str(Path(OUTPUT_DIR) / name)


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


def build_utm_edges(dx, dy, reference, transformer, lat_ref, lon_ref):
    """
    Same convention as build_utm_axes, but returns cell EDGE coordinates
    (n+1 values for n cells) rather than cell centres — the true, exact
    boundaries between adjacent ModEM cells (which are NOT evenly spaced,
    since padding cells grow geometrically outward from the fine core
    region). These are what a caller needs for an exact, non-interpolated
    pcolormesh(edges_e, edges_n, field, shading="flat") rendering that
    shows a true cut through the mesh's actual cells rather than smoothing
    or resampling them onto a uniform pixel grid.

    Must be called with the SAME dx/dy (i.e. same trim/crop state) used
    for the matching build_utm_axes() call, so edges and centres describe
    the same set of cells.
    """
    x_edges_local = np.concatenate([[0.0], np.cumsum(dx)]) - np.sum(dx) / 2.0
    y_edges_local = np.concatenate([[0.0], np.cumsum(dy)]) - np.sum(dy) / 2.0

    ref_e, ref_n = _ref_to_utm(transformer, lat_ref, lon_ref)   # km

    utm_n_edges_km = ref_n + x_edges_local / 1e3   # shape (nx+1,)
    utm_e_edges_km = ref_e + y_edges_local / 1e3   # shape (ny+1,)

    return utm_e_edges_km, utm_n_edges_km


# ------------------------------------------------------------------
# Depth axis builder
# ------------------------------------------------------------------

def build_depth_axis_km(dz, ref_z=0.0):
    """
    Return cell-centre depth array in km, positive down *from sea level*.

    ref_z must be the same reference used for get_topo()'s `ref[2]`
    (i.e. reference[2] from read_mod) — get_topo computes its surface
    elevation as cumsum(dz) + ref[2] (see its "z ref is elevation" call
    site), so without adding the same ref_z here, this axis and the
    model's own topography are anchored to two different zero points: this
    one to the mesh's arbitrary top face, that one to sea level. That
    mismatch made it impossible for any interpolated depth to come out
    negative (above sea level) regardless of how negative a caller's
    zmin_km was — the model's own z-axis simply didn't extend there.
    """
    nz = len(dz)
    z_edges = np.concatenate([[0.0], np.cumsum(dz)]) + ref_z
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])   # metres
    return z_centres / 1e3


def build_depth_edges_km(dz, ref_z=0.0):
    """
    Return cell EDGE depths in km (n+1 values for n cells), on the same
    datum as build_depth_axis_km (see its docstring for ref_z). These are
    the true, non-uniform depth-cell boundaries — dz grows with depth just
    like dx/dy grow in the horizontal padding — needed for an exact,
    non-interpolated vertical-section cut through the mesh's real cells.
    """
    z_edges = np.concatenate([[0.0], np.cumsum(dz)]) + ref_z
    return z_edges / 1e3


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

    Returns
    -------
    dx_t, dy_t, dz_t, mval_t, z_trim_offset_m
        z_trim_offset_m is the total thickness (metres) trimmed off the top
        of the z-axis (0.0 if tz0=0) — pass this into build_depth_axis_km's
        ref_z (added to reference[2]) so the depth axis still lines up with
        get_topo's own elevation reference even if z is trimmed.
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
    z_trim_offset_m = float(np.sum(dz[:tz0])) if tz0 else 0.0

    print(f"  After trimming: {mval_t.shape[0]}×{mval_t.shape[1]}×{mval_t.shape[2]} cells")
    return dx_t, dy_t, dz_t, mval_t, z_trim_offset_m


# ------------------------------------------------------------------
# NetCDF writers
# ------------------------------------------------------------------

def save_grid_edges(utm_e_edges_km, utm_n_edges_km, depth_edges_km, outfile):
    """
    Write the true (non-uniform) UTM cell EDGE coordinates shared by every
    field on this mesh — resistivity, sensitivity, all depth slices — plus
    the depth-cell edges, to a small standalone NetCDF. The plot script
    uses these for an exact, non-interpolated pcolormesh(edges, edges,
    field, shading="flat") rendering — for depth slices AND vertical
    sections: each rendered patch is then a true cut through one actual
    mesh cell, rather than a value resampled/interpolated onto a uniform
    pixel grid or blended between neighbouring cells.
    """
    ds = xr.Dataset(
        {
            "easting_edges": (
                "easting_edge", utm_e_edges_km.astype(np.float32),
                {"units": "km", "long_name": "UTM easting cell edges"},
            ),
            "northing_edges": (
                "northing_edge", utm_n_edges_km.astype(np.float32),
                {"units": "km", "long_name": "UTM northing cell edges"},
            ),
            "depth_edges": (
                "depth_edge", depth_edges_km.astype(np.float32),
                {"units": "km", "long_name": "Depth cell edges",
                 "positive": "down"},
            ),
        }
    )
    ds.to_netcdf(outfile)
    print(f"  Saved: {outfile}")


def save_3d_model(utm_e_km, utm_n_km, depth_km, rho_transformed, outfile):
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
    da.to_netcdf(outfile)
    print(f"  Saved: {outfile}")


def save_3d_field(utm_e_km, utm_n_km, depth_km, field, outfile,
                  long_name, units, transform):
    """
    Write an arbitrary 3-D field (same mesh/orientation as the resistivity
    model) to NetCDF. Generic version of save_3d_model, parameterised by
    name/units/transform instead of assuming resistivity — used for the
    sensitivity/resolution field.
    """
    data = np.transpose(field, (2, 0, 1))   # (nx,ny,nz) -> (nz, nx, ny)

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
        attrs={"long_name": long_name, "units": units, "transform": transform},
    )
    da.to_netcdf(outfile)
    print(f"  Saved: {outfile}")


def save_depth_slice_field(utm_e_km, utm_n_km, depth_km_axis, field,
                           target_depth_km, outfile, long_name, units,
                           transform):
    """
    Generic version of save_depth_slice, parameterised by name/units/
    transform instead of assuming resistivity — used for the
    sensitivity/resolution field.
    """
    iz = int(np.argmin(np.abs(depth_km_axis - target_depth_km)))
    actual_depth = depth_km_axis[iz]
    print(f"  Depth slice {target_depth_km} km → nearest cell centre {actual_depth:.2f} km")

    slc = field[:, :, iz]   # (nx, ny) = (northing, easting)

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
            "long_name":       f"{long_name} at {actual_depth:.1f} km depth",
            "units":           units,
            "depth_km":        float(actual_depth),
            "target_depth_km": float(target_depth_km),
            "transform":       transform,
        },
    )
    da.to_netcdf(outfile)
    print(f"  Saved: {outfile}")


def save_topo(utm_e_km, utm_n_km, topo_m, outfile):
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
    da.to_netcdf(outfile)
    print(f"  Saved: {outfile}")


def save_depth_slice(utm_e_km, utm_n_km, depth_km_axis, rho_transformed,
                     target_depth_km, outfile):
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
    da.to_netcdf(outfile)
    print(f"  Saved: {outfile}")


def save_sites(utm_e_km_sites, utm_n_km_sites, elev_m_sites, site_names, outfile):
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
    ds.to_netcdf(outfile)
    print(f"  Saved: {outfile}  ({n} sites)")


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

# ------------------------------------------------------------------
# 1b. Read sensitivity/resolution model (optional)
# ------------------------------------------------------------------
# Same mesh/format as the .rho file — read with the same function. Kept
# in lock-step with mval through every subsequent trim/crop step below so
# the two stay aligned on identical cells; sens is None if disabled,
# missing, or on a mesh that doesn't match the resistivity model.
sens = None
if USE_SENSITIVITY:
    print("\n=== Reading sensitivity/resolution model ===")
    # Guard against a common mistake: if SENS_FILE already ends with
    # SENS_EXT (e.g. someone pasted a full filename including ".sns"),
    # the naive SENS_FILE + SENS_EXT concatenation below would silently
    # build a nonexistent double-extension path (".sns.sns") and just
    # look like "file not found" with no clue why.
    if SENS_FILE.endswith(SENS_EXT):
        print(f"  NOTE: SENS_FILE already ends with {SENS_EXT!r} — "
              f"stripping it before appending SENS_EXT, to avoid building "
              f"a nonexistent '...{SENS_EXT}{SENS_EXT}' path.")
        SENS_FILE = SENS_FILE[: -len(SENS_EXT)]
    sens_path = Path(SENS_FILE + SENS_EXT)
    if not sens_path.exists():
        print(f"  WARNING: {sens_path} not found — skipping sensitivity "
              f"shading/blanking (set USE_SENSITIVITY = False to silence).")
    else:
        sdx, sdy, sdz, sens, sref, strans_in = mdm.read_mod(
            file=SENS_FILE, modext=SENS_EXT, trans="LINEAR", out=True
        )
        if sens.shape != mval.shape:
            print(f"  WARNING: {sens_path} has shape {sens.shape}, resistivity "
                  f"model has {mval.shape} — meshes don't match, cannot use "
                  f"for shading/blanking. Skipping.")
            sens = None
        elif not (np.allclose(sdx, dx) and np.allclose(sdy, dy) and np.allclose(sdz, dz)):
            print(f"  WARNING: {sens_path} cell sizes don't match the "
                  f"resistivity model's — meshes may not be identical. "
                  f"Proceeding, but double-check this file is the right one.")

        if sens is not None and (SENS_FLIP_EASTING or SENS_FLIP_NORTHING):
            # sens has shape (North, East, Down) — same convention as mval
            # (see build_utm_axes). Flipping here, before trim/crop, keeps
            # every downstream product (3-D field, depth slices) consistent
            # without needing a second fix in the plot script.
            if SENS_FLIP_EASTING:
                sens = sens[:, ::-1, :]
                print("  Flipped sensitivity along East-West axis "
                      "(SENS_FLIP_EASTING = True)")
            if SENS_FLIP_NORTHING:
                sens = sens[::-1, :, :]
                print("  Flipped sensitivity along North-South axis "
                      "(SENS_FLIP_NORTHING = True)")

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
if sens is not None:
    # Trim with the same TRIM_PAD and the same (pre-trim) dx/dy/dz used for
    # mval, so both fields are cut at identical cell indices.
    _, _, _, sens, _ = trim_model(dx, dy, dz, sens, TRIM_PAD)
dx, dy, dz, mval, z_trim_offset_m = trim_model(dx, dy, dz, mval, TRIM_PAD)

# ------------------------------------------------------------------
# 4. Build coordinate axes
# ------------------------------------------------------------------
print("\n=== Building UTM coordinate axes ===")
utm_e_km, utm_n_km = build_utm_axes(dx, dy, reference, transformer,
                                     lat_ref, lon_ref)
utm_e_edges_km, utm_n_edges_km = build_utm_edges(dx, dy, reference, transformer,
                                                  lat_ref, lon_ref)
depth_km = build_depth_axis_km(dz, ref_z=reference[2] + z_trim_offset_m)
depth_edges_km = build_depth_edges_km(dz, ref_z=reference[2] + z_trim_offset_m)

print(f"  Easting  range: {utm_e_km.min():.1f} – {utm_e_km.max():.1f} km")
print(f"  Northing range: {utm_n_km.min():.1f} – {utm_n_km.max():.1f} km")
print(f"  Depth    range: {depth_km.min():.2f} – {depth_km.max():.1f} km")

# ------------------------------------------------------------------
# 4b. Crop to geographic region of interest
# ------------------------------------------------------------------
if CROP_TO_REGION:
    print("\n=== Cropping to region of interest ===")
    corner_lons = [TAR_LON[0], TAR_LON[1], TAR_LON[0], TAR_LON[1]]
    corner_lats = [TAR_LAT[0], TAR_LAT[0], TAR_LAT[1], TAR_LAT[1]]
    ce, cn = transformer.transform(corner_lons, corner_lats)
    ce = np.asarray(ce) / 1e3
    cn = np.asarray(cn) / 1e3
    e_min, e_max = ce.min(), ce.max()
    n_min, n_max = cn.min(), cn.max()

    idx_e = np.where((utm_e_km >= e_min) & (utm_e_km <= e_max))[0]
    idx_n = np.where((utm_n_km >= n_min) & (utm_n_km <= n_max))[0]

    if idx_e.size == 0 or idx_n.size == 0:
        sys.exit(
            "CROP_TO_REGION removed all cells — TAR_LON/TAR_LAT does not "
            "overlap the trimmed model domain. Check the bounds, or the "
            "TRIM_PAD setting, or set CROP_TO_REGION = False."
        )

    # Grids are monotonic (cumulative cell offsets), so the index ranges are
    # contiguous — slice rather than fancy-index so dx/dy/mval stay aligned.
    sl_e = slice(idx_e.min(), idx_e.max() + 1)
    sl_n = slice(idx_n.min(), idx_n.max() + 1)
    # Edge arrays have one more element than centres (n cells -> n+1
    # edges), so their matching slice needs to extend one index further.
    sl_e_edges = slice(idx_e.min(), idx_e.max() + 2)
    sl_n_edges = slice(idx_n.min(), idx_n.max() + 2)

    utm_e_km = utm_e_km[sl_e]
    utm_n_km = utm_n_km[sl_n]
    utm_e_edges_km = utm_e_edges_km[sl_e_edges]
    utm_n_edges_km = utm_n_edges_km[sl_n_edges]
    dy       = dy[sl_e]     # East cell widths, indexed like easting
    dx       = dx[sl_n]     # North cell widths, indexed like northing
    mval     = mval[sl_n, sl_e, :]
    if sens is not None:
        sens = sens[sl_n, sl_e, :]

    # CROP_TO_REGION above selects cells by *centre* falling inside the
    # box — the boundary cell on each side can still be a large padding
    # cell (tens of km, near the mesh edge), so its true, non-uniform
    # edge can extend well past e_min/e_max/n_min/n_max. The topography
    # raster (a separate DEM, cropped tightly to the same box) always
    # stops exactly at the box — so left un-clipped, the exact-geometry
    # resistivity/sensitivity rendering would visibly overhang past
    # where the topo basemap ends. Clip only the outermost edge of the
    # boundary cells to the requested box: the cell keeps its real
    # value, just truncated at the window the box defines, matching the
    # topo raster's own hard cutoff there.
    print(f"  Boundary-cell edge overhang before clipping: "
          f"easting [{e_min - utm_e_edges_km[0]:+.2f}, "
          f"{utm_e_edges_km[-1] - e_max:+.2f}] km, "
          f"northing [{n_min - utm_n_edges_km[0]:+.2f}, "
          f"{utm_n_edges_km[-1] - n_max:+.2f}] km")
    utm_e_edges_km[0]  = max(utm_e_edges_km[0],  e_min)
    utm_e_edges_km[-1] = min(utm_e_edges_km[-1], e_max)
    utm_n_edges_km[0]  = max(utm_n_edges_km[0],  n_min)
    utm_n_edges_km[-1] = min(utm_n_edges_km[-1], n_max)

    print(f"  Cropped easting  range: {utm_e_km.min():.1f} – {utm_e_km.max():.1f} km"
          f"  ({mval.shape[1]} cells)")
    print(f"  Cropped northing range: {utm_n_km.min():.1f} – {utm_n_km.max():.1f} km"
          f"  ({mval.shape[0]} cells)")

# ------------------------------------------------------------------
# 5. Apply output transform
# ------------------------------------------------------------------
print(f"\n=== Applying output transform: {OUTPUT_TRANSFORM} ===")
rho_out = apply_transform(mval, OUTPUT_TRANSFORM)

sens_out = None
if sens is not None:
    print(f"=== Applying sensitivity transform: {SENS_TRANSFORM} ===")
    sens_out = apply_transform(sens, SENS_TRANSFORM)

# ------------------------------------------------------------------
# 6. Topography from model
# ------------------------------------------------------------------
print("\n=== Extracting model topography ===")
# get_topo expects physical mval (Ω·m) and reference in metres
xcnt, ycnt, topo_m = mdm.get_topo(
    dx=dx, dy=dy, dz=dz, mval=mval,
    ref=[0., 0., reference[2] + z_trim_offset_m],  # keep in sync with
    # build_depth_axis_km's ref_z above — both must use the same z
    # reference (reference[2] plus any top-z trim offset) or this
    # topography and the model's own depth axis go back out of alignment.
    mvalair=RHO_AIR, out=True,
)
# xcnt/ycnt are local offsets in metres, matching dx/dy after trimming.
# We use the already-built UTM axes instead.
save_topo(utm_e_km, utm_n_km, topo_m, outpath("modem_topo_utm.nc"))

# ------------------------------------------------------------------
# 7. Full 3-D model
# ------------------------------------------------------------------
print("\n=== Saving 3-D model ===")
save_3d_model(utm_e_km, utm_n_km, depth_km, rho_out, outpath("modem_model_utm.nc"))
save_grid_edges(utm_e_edges_km, utm_n_edges_km, depth_edges_km,
                outpath("modem_grid_edges_utm.nc"))

if sens_out is not None:
    print("\n=== Saving 3-D sensitivity/resolution field ===")
    sens_long_name = {
        "LOG10": "log10 sensitivity", "LOGE": "ln sensitivity",
        "LINEAR": "sensitivity",
    }.get(SENS_TRANSFORM.upper(), "sensitivity")
    sens_units = {
        "LOG10": "log10(sensitivity)", "LOGE": "ln(sensitivity)",
        "LINEAR": "sensitivity",
    }.get(SENS_TRANSFORM.upper(), "sensitivity")
    save_3d_field(utm_e_km, utm_n_km, depth_km, sens_out, outpath("modem_sens_utm.nc"),
                 sens_long_name, sens_units, SENS_TRANSFORM)

# ------------------------------------------------------------------
# 8. Depth slices
# ------------------------------------------------------------------
print("\n=== Saving depth slices ===")
for d_km in DEPTH_SLICES_KM:
    tag = f"{d_km:.0f}km" if d_km == int(d_km) else f"{d_km:.1f}km"
    save_depth_slice(utm_e_km, utm_n_km, depth_km, rho_out,
                     d_km, outpath(f"modem_rho_utm_{tag}.nc"))
    if sens_out is not None:
        save_depth_slice_field(utm_e_km, utm_n_km, depth_km, sens_out, d_km,
                               outpath(f"modem_sens_utm_{tag}.nc"),
                               sens_long_name, sens_units, SENS_TRANSFORM)

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
           outpath("modem_sites_utm.nc"))

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n=== Done — output files ===")
outputs = [
    "modem_topo_utm.nc",
    "modem_model_utm.nc",
    "modem_grid_edges_utm.nc",
    "modem_sites_utm.nc",
] + [
    "modem_rho_utm_{}.nc".format(
        f"{d:.0f}km" if d == int(d) else f"{d:.1f}km"
    )
    for d in DEPTH_SLICES_KM
]
if sens_out is not None:
    outputs.append("modem_sens_utm.nc")
    outputs += [
        "modem_sens_utm_{}.nc".format(
            f"{d:.0f}km" if d == int(d) else f"{d:.1f}km"
        )
        for d in DEPTH_SLICES_KM
    ]
for f in outputs:
    print(f"  {outpath(f)}")
