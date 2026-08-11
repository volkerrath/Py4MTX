#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precompute.py
====================
Combined pre-computation script for the imaging pipeline.

Site-specific settings (input files, region box, depth lists, etc.) are
kept as capitalized constants below; where they differ between sites, the
currently-active value is live and any other site's value is kept as a
labeled, commented-out alternative (e.g. "# TACNA") right next to it, so
switching sites is a matter of swapping which line is commented.

This merges what were previously two separate scripts:
  * precompute_modem.py  (Part A below) — ModEM 3-D MT resistivity
    inversion results (+ optional sensitivity/resolution field).
  * precompute_seis.py   (Part B below) — seismic tomography
    Vp / Vs / Vp-Vs-ratio, topography/bathymetry, and (new) density.

Both parts read their own model files, reproject/crop/slice them onto UTM
(Zone 19S, EPSG:32719) grids in km, and write NetCDF files consumed by the
companion plot scripts (plot_modem_image.py, plot_modem_mesh.py,
plot_seis.py). They are combined into one script because they share
region-of-interest settings (TAR_LON/TAR_LAT/CROP_TO_REGION — see SHARED
SETTINGS below) and because a density model, added here, sits alongside the
seismic Vp/Vs tomography and is processed exactly the same way.

What this script produces
--------------------------
Part A (ModEM / MT, native ModEM mesh):
  modem_model_utm.nc         — full 3-D log10(ρ) resistivity model
  modem_sens_utm.nc          — full 3-D sensitivity/resolution field (optional)
  modem_topo_utm.nc          — 2-D surface topography extracted from the model
  modem_sites_utm.nc         — MT site positions
  modem_rho_utm_{D}km.nc     — horizontal log10(ρ) slice at depth D km
  modem_sens_utm_{D}km.nc    — matching sensitivity slice (optional)
  modem_grid_edges_utm.nc    — exact (non-uniform) mesh cell edges
  modem_submesh_points.nc    — full submesh flattened to one row per cell
                                (easting/northing/depth + resistivity +
                                sensitivity), for clustering

Part B (seismic tomography + density, seismic grid):
  {SITE_PREFIX}_topo_utm.nc          — topography on UTM-km grid
  {SITE_PREFIX}_bath_utm.nc          — bathymetry mask (topo ≤ 0) on UTM-km grid
  {SITE_PREFIX}_vp.nc / {SITE_PREFIX}_vs.nc / {SITE_PREFIX}_vps.nc / {SITE_PREFIX}_dens.nc
                              — full Vp / Vs / Vp-Vs-ratio / density subsets
                                (depth coord + UTM aux coords)
  {SITE_PREFIX}_vp_utm_{tag}.nc  }
  {SITE_PREFIX}_vs_utm_{tag}.nc  }
  {SITE_PREFIX}_vps_utm_{tag}.nc }    — per-depth UTM-km slices
  {SITE_PREFIX}_dens_utm_{tag}.nc}    — per-depth UTM-km density slice

Density (Part B) — renamed "dens", not "rho"
------------------------------------------------
FNAME_DENS (default "../seistomo/FD_rho_model.nc" — the source *file* is
still named FD_rho_model.nc on disk) is read, cropped, and sliced with
exactly the same logic already used for Vp/Vs — same region box, same
DEPTH_INDEX depths, same UTM reprojection helper (slice_to_utm_km_nc). It
is expected to sit on the same (lat, lon, depth) grid as FD_vp_model.nc; a
non-fatal warning is printed at run time if that grid doesn't match, since
a silent mismatch there would misalign density slices against velocity
ones. Exported as "dens" ({SITE_PREFIX}_dens.nc, {SITE_PREFIX}_dens_utm_{tag}.nc)
rather than "rho", since "rho" is already used throughout Part A for MT
resistivity (modem_rho_utm_*.nc) — same-sounding names on two physically
different quantities, on two different grids, was a bug waiting to happen.

Note — no more MT-onto-seismic-grid resampling here
------------------------------------------------------------------
Earlier versions of this script resampled resistivity onto the seismic
Vp/Vs/density grid here in precompute (modem_rho_on_seisgrid*.nc). That
step has moved to cluster.py, which now RBF-interpolates every
clustering variable — resistivity/conductivity from modem_submesh_points.nc
below, Vp/Vs/density from Part B's own outputs — onto one jointly-defined
regular grid, rather than onto the seismic tomography grid specifically.
See cluster.py / README_cluster.md.

Full submesh point table, for clustering (Part A, new)
------------------------------------------------------------------
See save_submesh_table() below. modem_submesh_points.nc flattens the full
3-D resistivity (and sensitivity) field on its own native ModEM submesh —
at full native resolution — into one row per cell: easting_km/northing_km/
depth_km as ordinary columns alongside resistivity/sensitivity. This is
now cluster.py's actual source for resistivity: it loads this point
table directly and RBF-interpolates it onto the joint clustering grid
(rather than this script resampling it onto the seismic grid first, as it
used to). Air-masked/out-of-mesh cells stay NaN (with a `valid` flag)
rather than being dropped, so the clustering code can apply whatever NaN
policy it prefers. Toggle with EXPORT_SUBMESH_TABLE (Part A settings).

Coordinate convention (Part A / ModEM)
----------------------------------------
ModEM uses a local Cartesian mesh whose origin is the *reference point*
(lat_ref, lon_ref) stored in the last non-comment line of the .rho file.
The mesh x-axis points North, y-axis East, z-axis Down (positive downward).
Cell sizes dx, dy, dz are in metres. See build_utm_axes/build_utm_edges
below for how these are turned into absolute UTM easting/northing (km).

Region of interest
-------------------
TRIM_PAD (Part A only) drops a fixed number of padding cells from each
face, but ModEM padding cells grow geometrically toward the model
boundary, so a small TRIM_PAD count can still leave a domain far larger
than the area actually of interest. CROP_TO_REGION (shared, see below)
further crops both parts to the same geographic box TAR_LON/TAR_LAT, so
the ModEM and seismic outputs cover matching areas.

Helpers used from modem.py (Part A)
--------------------------------------
  read_mod(file, modext, trans)   — reads .rho file → dx, dy, dz, mval, reference
  read_data(Datfile, modext)      — reads .dat file → Site, Comp, Data, Head
  cells3d(dx, dy, dz, center)     — cumulative cell-centre coordinates
  get_topo(dx, dy, dz, mval, ref) — extracts 2-D surface topography

Topography back-end (Part B)
--------------------------------
Requires the `elevation` package (pip install elevation) and its cli tool
`eio`, which downloads SRTM 30 m (land) / ETOPO1 (ocean) tiles on first use
and caches them locally. Alternatively, set TOPO_SOURCE = "etopo" to use
the ETOPO1 global relief NetCDF directly, or "geotiff" for a local file
(see SEISMIC SETTINGS below).

Dependencies
------------
    numpy, xarray, pandas, pyproj, scipy, rioxarray, elevation (eio), modem
    (project-local) — OR, without the "elevation" backend: numpy, xarray,
    pyproj, scipy, rioxarray (with a local GeoTIFF), modem (project-local)

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic), 2026-06-29. Merged from
precompute_modem.py / precompute_seis.py, extended with density
(FD_rho_model.nc, exported as "dens") support, and extended again with MT
resistivity resampled onto the seismic grid: Claude (Anthropic), 2026-07-31.
License: GNU General Public License v3 (GPL-3.0-or-later).
AI-generated code — review before use in production.
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

# modem.py must be on the Python path or in the working directory
try:
    import modem as mdm
except ImportError:
    sys.exit(
        "Cannot import modem.py — place it in the working directory or on PYTHONPATH."
    )

# =====================================================================
# SHARED SETTINGS  (used identically by Part A and Part B)
# =====================================================================

# --- Site selector ---
# Prefixes every Part-B seismic/density output filename
# ({SITE_PREFIX}_vp.nc, {SITE_PREFIX}_topo_utm.nc, etc. — see module
# docstring). Change this (and the site-specific settings marked with a
# "# TACNA" comment throughout this file) to switch which site's inputs
# and outputs are active.
# SITE_PREFIX = "saba"
SITE_PREFIX = "tacna"  # TACNA

# Directory for all NetCDF outputs written by this script (created if it
# doesn't exist). Default "." keeps everything in the current directory.
# plot_modem_image.py / plot_modem_mesh.py / plot_seis.py have a matching
# NC_DIR setting to read from wherever this is pointed at.
# OUTPUT_DIR = "."
OUTPUT_DIR = "../precompute/tacna/"

# --- Geographic region of interest ---
# TRIM_PAD (Part A) only drops a fixed number of cells and typically still
# leaves a domain far larger than the area of interest (ModEM padding
# cells grow geometrically toward the boundary). Set CROP_TO_REGION = True
# to crop both the ModEM grid and the seismic/density subset to this
# geographic box before any NetCDF output is written. Set to False to keep
# each part's full (trimmed, for ModEM) source extent instead.
#
# IMPORTANT: this box must fully contain every VSLICES profile endpoint
# defined in plot_modem_image.py AND plot_seis.py (PROFILE_CD_LON/LAT
# etc.) — profile_CD's endpoints, [-70.476, -18.255] and [-69.499,
# -17.048], both fell *outside* an earlier, narrower version of this box
# on every side, leaving no data at all near either end of the profile (a
# white gap at both edges of the section). Widened with margin here; if
# you add profiles reaching further, widen this to match.
#
# This is the union of the original ModEM box [-70.55, -69.40] x
# [-18.35, -16.95] and the original seismic box [-70.79, -69.48] x
# [-18.34, -17.01], padded by ~0.05° — now shared by both parts of the
# script so they always cover the same geographic area by construction.
CROP_TO_REGION = True
TAR_LON = [-70.84, -69.35]  # TACNA
TAR_LAT = [-18.40, -16.90]  # TACNA
#TAR_LON = [-72.62, -71.271]
#TAR_LAT = [-16.62, -15.109]  # passing by Sabancaya


Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def outpath(name):
    """Join a bare output filename onto OUTPUT_DIR."""
    return str(Path(OUTPUT_DIR) / name)


def safe_to_netcdf(obj, path):
    """
    Write a Dataset/DataArray to NetCDF, overwriting any existing file at
    `path` even if it's read-only — e.g. left over from an earlier run
    (possibly by a different user/process, or with different permissions),
    which otherwise makes xarray's own to_netcdf() raise PermissionError
    instead of just overwriting it. Removes the stale file first (fixing
    its permissions first if needed), then writes normally.
    """
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
        except PermissionError:
            os.chmod(p, 0o644)
            p.unlink()
    obj.to_netcdf(path)


# =====================================================================
# PART A SETTINGS — ModEM / MT (resistivity, sensitivity, sites)
# =====================================================================

# --- Input files (without extension) ---
MODEL_FILE = "../mt/tacna/TACG26b_Z1_NLCG_006_clip"  # TACNA
DATA_FILE = "../mt/tacna/TAC_100_smooth2_short/TACG26b_100ZT_Alpha03_smooth_NLCG_007"  # TACNA
#MODEL_FILE = "../mt/saba/SABA13_Z_Alpha01_priorP8_NLCG_016_clip"  # reads MODEL_FILE + MODEL_EXT
#DATA_FILE = "../mt/saba/SABA13a_Z"  # reads DATA_FILE  + DATA_EXT
MODEL_EXT = ".rho"
DATA_EXT = ".dat"

# --- Sensitivity/resolution file (optional, for shading/blanking) ---
# Same grid format as the .rho model file — read with the same reader
# (mdm.read_mod), so it must share the .rho file's mesh (dx/dy/dz, cell
# counts). Typically shares its base name too, but can be set separately.
# Set USE_SENSITIVITY = False to skip reading/writing it entirely.
USE_SENSITIVITY = False
# SENS_FILE = MODEL_FILE      # base name (without extension)
# SENS_FILE = "../mt/TAC30_nerr_sp-8_anco_cov_max"  # TACNA
SENS_FILE = ("../mt/tacna/TAC_G2_ZT1_nerr_sp-8_Dtype_zfull_sqr_max.sns")  # TACNA
#SENS_FILE = (
    #"../mt/saba/SABA13a_total_sns"
#)
SENS_EXT = ".sns"
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
SENS_FLIP_EASTING = False  # this file doesn't need an East-West flip
SENS_FLIP_NORTHING = False  # empirically confirmed against real station
# positions for TAC_G2_ZT1_nerr_sp-8_Dtype_
# zfull_sqr_max.sns — a different .sns file
# may need different settings; re-validate
# if you switch files again.

# --- Reference point (must match the model file; sign-checked at runtime) ---
# ModEM stores [lat, lon, elevation_m] in the last data line of the .rho file.
# Leave as None to read directly from the file (recommended).
REFERENCE_LAT = None  # degrees, WGS84; None → read from model file
REFERENCE_LON = None  # degrees, WGS84; None → read from model file

# --- Resistivity transform for output ---
# "LOG10"  : save log10(ρ)  [most common for visualisation]
# "LOGE"   : save ln(ρ)
# "LINEAR" : save ρ in Ω·m
OUTPUT_TRANSFORM = "LOG10"

# --- Depth slices to export as 2-D horizontal NetCDF grids (km, positive down) ---
# Must match DEPTH_SLICES_KM in plot_modem_mesh.py — both the resistivity
# (modem_rho_utm_{tag}.nc) and sensitivity (modem_sens_utm_{tag}.nc) depth
# slices are written from this one list, in the same loop (see "8. Depth
# slices" below), so keeping this in sync automatically keeps sensitivity
# covering the same depths as resistivity.
# DEPTH_SLICES_KM = [-3., -1., 1.0, 5.0, 9.0]  # TACNA
DEPTH_SLICES_KM = [-3., 1.0, 6.0, 11.0, 16.0, 21.0, 26.0, 31.0, 36]
# --- Export the full native ModEM submesh as a flat point table, for
# clustering ---
# Produces modem_submesh_points.nc: one row per ModEM cell (after
# TRIM_PAD/CROP_TO_REGION), with easting/northing/depth (km) as ordinary
# columns alongside resistivity (and sensitivity, if USE_SENSITIVITY) —
# ready to feed straight into a clustering script as an (n_points,
# n_features) table, no reshaping needed. This is the FULL native ModEM
# mesh resolution, as a point cloud — cluster.py reads this
# directly and RBF-interpolates it onto its own jointly-defined regular
# grid (see README_cluster.md), rather than this script resampling it
# onto any other grid first.
EXPORT_SUBMESH_TABLE = True

# --- Air-cell resistivity threshold (Ω·m) used by get_topo ---
RHO_AIR = 1.0e17

# --- Air-cell masking threshold (Ω·m) for the saved model itself ---
# Any cell at or above this resistivity is masked to NaN before the
# output transform is applied — baked into modem_rho_utm_{tag}.nc and
# modem_model_utm.nc directly, rather than left for each plot script to
# re-derive its own air/rock cutoff from the raw resistivity value. NaN
# then propagates naturally through everything downstream: map color
# scales, RegularGridInterpolator (image script), and the exact per-cell
# lookup (mesh script) all just see "no data" and treat it consistently,
# with no risk of one consumer's threshold disagreeing with another's.
# Deliberately far below RHO_AIR itself (1e10 vs 1e17) — real rock in
# this model tops out at a few thousand Ω·m at most, so this has a huge
# margin on both sides and won't accidentally catch resistive rock.
AIR_RHO_THRESHOLD = 1.0e10

# --- "Illegal"/sentinel-value threshold, near-zero end (Ω·m or
# sensitivity units — see note below) ---
# Some .rho/.sns files mark invalid/no-data cells with a tiny non-zero
# placeholder near the OTHER end of the scale from AIR_RHO_THRESHOLD
# above — e.g. 1e-32 — rather than (or in addition to) a huge air-like
# value. Applied by apply_transform() the same way AIR_RHO_THRESHOLD is:
# once, here, before the transform, to whichever field is passed in
# (resistivity or sensitivity — apply_transform() is shared by both), so
# every downstream consumer sees a single consistent NaN rather than
# re-deriving its own cutoff. Any cell with |value| <= this threshold is
# masked to NaN — abs() because some codes use a negative sentinel too,
# and because for the LOG10/LOGE transforms a genuine physical value
# this close to zero would itself already blow up into a huge negative
# log the moment the transform is applied, so there's no real resistivity
# regime this could be mistaken for. Deliberately far below any real Ω·m
# or sensitivity value (which never legitimately reach anywhere near
# double precision's ~1e-300 underflow floor) — set to None to disable.
ILLEGAL_LOW_THRESHOLD = 1.0e-20

# --- Padding cells to trim from each face before output ---
# ModEM models have large padding cells at the boundary that distort colour scales.
# [trim_x0, trim_x1, trim_y0, trim_y1, trim_z0] — number of cells to drop
# from the -x, +x, -y, +y faces and the top (z=0) face respectively.
# Set all to 0 to keep the full model.
# TRIM_PAD = [7, 7, 7, 7, 16]  # TACNA
TRIM_PAD = [7, 7, 7, 7, 0]

# --- Geographic region of interest ---
# CROP_TO_REGION / TAR_LON / TAR_LAT now live in the SHARED SETTINGS
# section above (used identically by both the ModEM and seismic parts
# of this script, which is exactly why they were merged into one).
# IMPORTANT: this box must fully contain every VSLICES profile endpoint
# defined in plot_modem_image.py (PROFILE_CD_LON/LAT etc.) — see
# the SHARED SETTINGS comment for the history of why this box has the
# margins it does.

# --- UTM zone override ---
# By default the zone is inferred from REFERENCE_LON.  Set manually if needed.
UTM_ZONE = None  # e.g. 19;  None → auto-detect
UTM_HEMI = None  # "N" or "S"; None → infer from REFERENCE_LAT

# =====================================================================
# END PART A SETTINGS
# =====================================================================


# =====================================================================
# PART B SETTINGS — seismic tomography (Vp, Vs, Vp/Vs) + density
# =====================================================================

# Input velocity model files
FNAME_VP = "../seistomo/FD_vp_model.nc"
FNAME_VS = "../seistomo/FD_vs_model.nc"
# Density model (kg/m^3 or g/cm^3, as stored in the source file — this
# script does not rescale it, only reprojects/crops/slices it exactly
# like Vp/Vs). Expected on the same (lat, lon, depth) grid as FNAME_VP/
# FNAME_VS — a soft consistency check at read time warns (does not
# abort) if the grids disagree. Exported here as "dens" ({SITE_PREFIX}_dens.nc,
# {SITE_PREFIX}_dens_utm_{tag}.nc) rather than "rho" — "rho" is already used
# throughout Part A for MT resistivity (modem_rho_utm_*.nc etc.), and the
# two are physically different quantities on different grids. The source
# *file* itself is still named FD_rho_model.nc on disk (matching the
# FD_vp_model.nc / FD_vs_model.nc naming convention) — only this script's
# internal variable/output naming has been disambiguated.
FNAME_DENS = "../seistomo/FD_rho_model.nc"

# Velocity subset geographic bounds
# CROP_TO_REGION / TAR_LON / TAR_LAT now live in the SHARED SETTINGS
# section above (used identically by both the ModEM and seismic parts
# of this script). See that section's comment for why the box has the
# margins it does (must fully contain every VSLICES profile endpoint
# defined in plot_seis.py, PROFILE_CD_LON/LAT etc.).

# TAR_LON = [-70.79, -69.50]  # TACNA, superseded by SHARED SETTINGS above
# TAR_LAT = [-18.34, -16.99]  # TACNA, superseded by SHARED SETTINGS above
# Lower bound was 0 (sea level), which silently discarded any above-sea-
# level coverage the source model has (e.g. under a volcanic edifice) —
# the same VSLICES zmin_km=-8.0 fix in plot_seis.py can't recover
# data that was already cropped out here at the source. -8 gives the same
# margin used there for the catalogue's shallowest events (z = -5.75 km).
# xarray's .sel(slice(...)) clips gracefully to whatever the source
# actually covers, so this is safe even if FD_vp_model.nc/FD_vs_model.nc
# don't extend that high — worth checking the printed depth range below
# after re-running to see whether real coverage was gained or not.

DEPTH_RANGE = [-8, 100]          # km
# DEPTH_INDEX = [1, 5, 10, 15, 20, 25, 30]  # TACNA
DEPTH_INDEX = [1, 5, 10, 15, 20, 25, 30, 35, 40]
# Depth indices to export as per-depth UTM-km slices
# Topo/bath geographic bounds (slightly wider than velocity subset)
# MAP_LON = [-70.94, -69.25]  # TACNA
# MAP_LAT = [-18.50, -16.80]  # TACNA
MAP_LON = [TAR_LON[0], TAR_LON[1]]
MAP_LAT = [TAR_LAT[0], TAR_LAT[1]]

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
# END PART B SETTINGS
# =====================================================================


# ------------------------------------------------------------------
# PART A HELPER FUNCTIONS — ModEM / MT
# ------------------------------------------------------------------
# (OUTPUT_DIR is created and outpath() defined once, in SHARED SETTINGS above.)


# ------------------------------------------------------------------
# UTM projection helpers
# ------------------------------------------------------------------


def _build_transformer(
    lat_ref: float, lon_ref: float, zone_override=None, hemi_override=None
):
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
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
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
    x_local = np.cumsum(dx) - np.sum(dx) / 2.0  # North offsets, m
    y_local = np.cumsum(dy) - np.sum(dy) / 2.0  # East  offsets, m
    x_local -= dx / 2.0
    y_local -= dy / 2.0

    # UTM coordinates of the geographic reference point
    ref_e, ref_n = _ref_to_utm(transformer, lat_ref, lon_ref)  # km

    # North offset → +northing;  East offset → +easting
    utm_n_km = ref_n + x_local / 1e3  # shape (nx,)
    utm_e_km = ref_e + y_local / 1e3  # shape (ny,)

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

    ref_e, ref_n = _ref_to_utm(transformer, lat_ref, lon_ref)  # km

    utm_n_edges_km = ref_n + x_edges_local / 1e3  # shape (nx+1,)
    utm_e_edges_km = ref_e + y_edges_local / 1e3  # shape (ny+1,)

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
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])  # metres
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
    """
    Return mval (Ω·m, physical) converted to the requested representation.

    Air cells (resistivity >= AIR_RHO_THRESHOLD) are masked to NaN here,
    once, before the transform — not left for each plot script to
    re-derive its own air/rock cutoff from the transformed value. NaN
    then propagates naturally through every downstream consumer (map
    color scales, interpolation, exact per-cell lookups) with a single,
    consistent source of truth for "is this air".

    Values at the other end of the scale (|mval| <= ILLEGAL_LOW_THRESHOLD,
    e.g. a 1e-32 sentinel some .rho/.sns files use to mark invalid/
    no-data cells) are masked to NaN the same way, for the same reason —
    see ILLEGAL_LOW_THRESHOLD's own comment above. Shared by both the
    resistivity and the sensitivity model (apply_transform() is called
    on whichever is passed in), so this catches the same kind of sentinel
    in either file.
    """
    mval = mval.copy().astype(float)

    n_air = int(np.sum(mval >= AIR_RHO_THRESHOLD))
    mval[mval >= AIR_RHO_THRESHOLD] = np.nan

    n_illegal = 0
    if ILLEGAL_LOW_THRESHOLD is not None:
        illegal = np.isfinite(mval) & (np.abs(mval) <= ILLEGAL_LOW_THRESHOLD)
        n_illegal = int(np.sum(illegal))
        mval[illegal] = np.nan

    if n_illegal:
        print(
            f"    Masked {n_illegal} cell(s) with |value| <= "
            f"ILLEGAL_LOW_THRESHOLD ({ILLEGAL_LOW_THRESHOLD:g}) to NaN "
            f"(sentinel/no-data placeholder, not physical)"
        )

    trans = trans.upper()
    if trans == "LOG10":
        return np.log10(np.where(mval > 0, mval, np.nan))
    elif trans in ("LOGE", "LN"):
        return np.log(np.where(mval > 0, mval, np.nan))
    elif trans == "LINEAR":
        return mval
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
    dx_t = dx[sl_x]
    dy_t = dy[sl_y]
    dz_t = dz[sl_z]
    z_trim_offset_m = float(np.sum(dz[:tz0])) if tz0 else 0.0

    print(
        f"  After trimming: {mval_t.shape[0]}×{mval_t.shape[1]}×{mval_t.shape[2]} cells"
    )
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
                "easting_edge",
                utm_e_edges_km.astype(np.float32),
                {"units": "km", "long_name": "UTM easting cell edges"},
            ),
            "northing_edges": (
                "northing_edge",
                utm_n_edges_km.astype(np.float32),
                {"units": "km", "long_name": "UTM northing cell edges"},
            ),
            "depth_edges": (
                "depth_edge",
                depth_edges_km.astype(np.float32),
                {
                    "units": "km",
                    "long_name": "Depth cell edges",
                    "positive": "down",
                },
            ),
        }
    )
    safe_to_netcdf(ds, outfile)
    print(f"  Saved: {outfile}")


def save_submesh_table(
    utm_e_km, utm_n_km, depth_km, rho_field, sens_field,
    rho_long_name, rho_units, sens_long_name, sens_units, outfile,
):
    """
    Flatten the full 3-D ModEM field(s) — on their own native (cropped/
    trimmed) submesh, at full resolution, NOT resampled onto the coarser
    seismic grid — into a one-row-per-cell point table: easting_km,
    northing_km, depth_km, resistivity (and sensitivity, if available).
    Intended as a ready-to-use feature table for clustering (e.g.
    fuzzy_cmeans.py) — every ModEM cell becomes one point/sample, with its
    coordinates kept as ordinary columns rather than dimension coordinates,
    so it loads directly as an (n_points, n_features) array
    (`ds.to_dataframe()` or `np.column_stack([ds["resistivity"].values,
    ds["sensitivity"].values])`), no reshaping needed.

    Air-masked / out-of-mesh cells remain NaN (same convention as
    modem_model_utm.nc) rather than being dropped — this keeps every
    point's coordinates meaningful and lets the clustering code apply
    whatever NaN policy it prefers, rather than this script deciding for
    it. `valid` is 1 where resistivity is finite, 0 where NaN, so a caller
    can filter without re-deriving the mask.
    """
    # utm_n_km (nx), utm_e_km (ny), depth_km (nz) — same order as
    # rho_field/sens_field (nx, ny, nz) = (North, East, Down).
    Ngrid, Egrid, Dgrid = np.meshgrid(
        utm_n_km, utm_e_km, depth_km, indexing="ij"
    )

    n_points = rho_field.size
    data_vars = {
        "easting": (
            "point", Egrid.ravel().astype(np.float32),
            {"units": "km", "long_name": "UTM easting"},
        ),
        "northing": (
            "point", Ngrid.ravel().astype(np.float32),
            {"units": "km", "long_name": "UTM northing"},
        ),
        "depth": (
            "point", Dgrid.ravel().astype(np.float32),
            {"units": "km", "long_name": "Depth below surface", "positive": "down"},
        ),
        "resistivity": (
            "point", rho_field.ravel().astype(np.float32),
            {"long_name": rho_long_name, "units": rho_units},
        ),
        "valid": (
            "point", np.isfinite(rho_field.ravel()).astype(np.int8),
            {"long_name": "1 if resistivity is finite (not air/out-of-mesh), else 0"},
        ),
    }
    if sens_field is not None:
        data_vars["sensitivity"] = (
            "point", sens_field.ravel().astype(np.float32),
            {"long_name": sens_long_name, "units": sens_units},
        )

    ds = xr.Dataset(data_vars, attrs={
        "description": (
            "Full native-ModEM-submesh point table (one row per mesh cell, "
            "after TRIM_PAD/CROP_TO_REGION), at full native ModEM "
            "resolution. Intended as a feature table for clustering — see "
            "cluster.py, which RBF-interpolates this onto its own "
            "jointly-defined regular grid."
        ),
    })
    safe_to_netcdf(ds, outfile)
    n_valid = int(np.isfinite(rho_field).sum())
    print(
        f"  Saved: {outfile}  ({n_points} points, {n_valid} valid / "
        f"{n_points - n_valid} NaN)"
    )


def save_3d_model(utm_e_km, utm_n_km, depth_km, rho_transformed, outfile):
    """Write full 3-D resistivity model to NetCDF."""
    long_name = {
        "LOG10": "log10 resistivity",
        "LOGE": "ln resistivity",
        "LINEAR": "resistivity",
    }.get(OUTPUT_TRANSFORM.upper(), "resistivity")
    units = {
        "LOG10": "log10(Ohm.m)",
        "LOGE": "ln(Ohm.m)",
        "LINEAR": "Ohm.m",
    }.get(OUTPUT_TRANSFORM.upper(), "Ohm.m")

    # rho_transformed has shape (nx, ny, nz) — ModEM (N, E, Down)
    # Reorder to (z, y, x) = (depth, northing, easting) for NetCDF convention
    data = np.transpose(rho_transformed, (2, 0, 1))  # (nz, nx, ny)

    da = xr.DataArray(
        data.astype(np.float32),
        dims=["depth", "northing", "easting"],
        coords={
            "depth": xr.Variable(
                "depth",
                depth_km,
                attrs={
                    "units": "km",
                    "positive": "down",
                    "long_name": "Depth below surface",
                },
            ),
            "northing": xr.Variable(
                "northing",
                utm_n_km,
                attrs={"units": "km", "long_name": "UTM northing"},
            ),
            "easting": xr.Variable(
                "easting",
                utm_e_km,
                attrs={"units": "km", "long_name": "UTM easting"},
            ),
        },
        attrs={
            "long_name": long_name,
            "units": units,
            "transform": OUTPUT_TRANSFORM,
        },
    )
    safe_to_netcdf(da, outfile)
    print(f"  Saved: {outfile}")


def save_3d_field(
    utm_e_km, utm_n_km, depth_km, field, outfile, long_name, units, transform
):
    """
    Write an arbitrary 3-D field (same mesh/orientation as the resistivity
    model) to NetCDF. Generic version of save_3d_model, parameterised by
    name/units/transform instead of assuming resistivity — used for the
    sensitivity/resolution field.
    """
    data = np.transpose(field, (2, 0, 1))  # (nx,ny,nz) -> (nz, nx, ny)

    da = xr.DataArray(
        data.astype(np.float32),
        dims=["depth", "northing", "easting"],
        coords={
            "depth": xr.Variable(
                "depth",
                depth_km,
                attrs={
                    "units": "km",
                    "positive": "down",
                    "long_name": "Depth below surface",
                },
            ),
            "northing": xr.Variable(
                "northing",
                utm_n_km,
                attrs={"units": "km", "long_name": "UTM northing"},
            ),
            "easting": xr.Variable(
                "easting",
                utm_e_km,
                attrs={"units": "km", "long_name": "UTM easting"},
            ),
        },
        attrs={"long_name": long_name, "units": units, "transform": transform},
    )
    safe_to_netcdf(da, outfile)
    print(f"  Saved: {outfile}")


def save_depth_slice_field(
    utm_e_km,
    utm_n_km,
    depth_km_axis,
    field,
    target_depth_km,
    outfile,
    long_name,
    units,
    transform,
):
    """
    Generic version of save_depth_slice, parameterised by name/units/
    transform instead of assuming resistivity — used for the
    sensitivity/resolution field.
    """
    iz = int(np.argmin(np.abs(depth_km_axis - target_depth_km)))
    actual_depth = depth_km_axis[iz]
    print(
        f"  Depth slice {target_depth_km} km → nearest cell centre {actual_depth:.2f} km"
    )

    slc = field[:, :, iz]  # (nx, ny) = (northing, easting)

    da = xr.DataArray(
        slc.astype(np.float32),
        dims=["northing", "easting"],
        coords={
            "northing": xr.Variable(
                "northing",
                utm_n_km,
                attrs={"units": "km", "long_name": "UTM northing"},
            ),
            "easting": xr.Variable(
                "easting",
                utm_e_km,
                attrs={"units": "km", "long_name": "UTM easting"},
            ),
        },
        attrs={
            "long_name": f"{long_name} at {actual_depth:.1f} km depth",
            "units": units,
            "depth_km": float(actual_depth),
            "target_depth_km": float(target_depth_km),
            "transform": transform,
        },
    )
    safe_to_netcdf(da, outfile)
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
    elev_m = -topo_m  # shape (nx, ny) = (northing, easting), positive up

    da = xr.DataArray(
        elev_m.astype(np.float32),
        dims=["northing", "easting"],
        coords={
            "northing": xr.Variable(
                "northing",
                utm_n_km,
                attrs={"units": "km", "long_name": "UTM northing"},
            ),
            "easting": xr.Variable(
                "easting",
                utm_e_km,
                attrs={"units": "km", "long_name": "UTM easting"},
            ),
        },
        attrs={
            "long_name": "Surface elevation",
            "units": "m",
            "positive": "up",
            "note": "Derived from shallowest non-air cell in ModEM model",
        },
    )
    safe_to_netcdf(da, outfile)
    print(f"  Saved: {outfile}")


def save_depth_slice(
    utm_e_km,
    utm_n_km,
    depth_km_axis,
    rho_transformed,
    target_depth_km,
    outfile,
):
    """
    Interpolate the 3-D model to a target depth and save as 2-D NetCDF.

    Uses nearest-neighbour selection on the depth axis (no interpolation
    artefacts across large depth intervals).

    rho_transformed shape: (nx, ny, nz)  ModEM (N, E, Down)
    """
    iz = int(np.argmin(np.abs(depth_km_axis - target_depth_km)))
    actual_depth = depth_km_axis[iz]
    print(
        f"  Depth slice {target_depth_km} km → nearest cell centre {actual_depth:.2f} km"
    )

    # Extract slice: shape (nx, ny) = (northing, easting) — no transpose needed
    slc = rho_transformed[:, :, iz]

    long_name = (
        f"{OUTPUT_TRANSFORM} resistivity at {actual_depth:.1f} km depth"
    )
    units = {
        "LOG10": "log10(Ohm.m)",
        "LOGE": "ln(Ohm.m)",
        "LINEAR": "Ohm.m",
    }.get(OUTPUT_TRANSFORM.upper(), "Ohm.m")

    da = xr.DataArray(
        slc.astype(np.float32),
        dims=["northing", "easting"],
        coords={
            "northing": xr.Variable(
                "northing",
                utm_n_km,
                attrs={"units": "km", "long_name": "UTM northing"},
            ),
            "easting": xr.Variable(
                "easting",
                utm_e_km,
                attrs={"units": "km", "long_name": "UTM easting"},
            ),
        },
        attrs={
            "long_name": long_name,
            "units": units,
            "depth_km": float(actual_depth),
            "target_depth_km": float(target_depth_km),
            "transform": OUTPUT_TRANSFORM,
        },
    )
    safe_to_netcdf(da, outfile)
    print(f"  Saved: {outfile}")


def save_sites(
    utm_e_km_sites, utm_n_km_sites, elev_m_sites, site_names, outfile
):
    """Write MT site positions to NetCDF."""
    n = len(site_names)
    ds = xr.Dataset(
        {
            "easting": xr.DataArray(
                utm_e_km_sites.astype(np.float32),
                dims=["site"],
                attrs={"units": "km", "long_name": "UTM easting"},
            ),
            "northing": xr.DataArray(
                utm_n_km_sites.astype(np.float32),
                dims=["site"],
                attrs={"units": "km", "long_name": "UTM northing"},
            ),
            "elevation": xr.DataArray(
                elev_m_sites.astype(np.float32),
                dims=["site"],
                attrs={
                    "units": "m",
                    "long_name": "Site elevation",
                    "positive": "up",
                },
            ),
            "name": xr.DataArray(
                np.array(site_names, dtype=object),
                dims=["site"],
                attrs={"long_name": "Site name"},
            ),
        }
    )
    safe_to_netcdf(ds, outfile)
    print(f"  Saved: {outfile}  ({n} sites)")


# ------------------------------------------------------------------
# PART B HELPER FUNCTIONS — seismic tomography + density
# ------------------------------------------------------------------
# (OUTPUT_DIR is created and outpath() defined once, in SHARED SETTINGS
# above.) UTM projection (Zone 19S, EPSG:32719) — fixed, unlike Part A's
# dynamically-detected transformer, since the seismic/density source
# grids are always geographic (lat/lon) rather than a ModEM local mesh.
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


def geo_to_utm_km_nc(da, outfile, spacing_km=1.0):
    """
    Reproject a geographic (lon/lat) DataArray to a regular UTM Zone 19S
    grid (km) via bilinear interpolation and write to NetCDF.

    Parameters
    ----------
    da         : xr.DataArray with 'lat'/'lon' (or 'y'/'x') dim coordinates
    outfile    : output NetCDF path (use outpath(name) to place it in OUTPUT_DIR)
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
    safe_to_netcdf(result, outfile)
    print(f"  Saved: {outfile}")


def slice_to_utm_km_nc(da, outfile):
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
    safe_to_netcdf(result, outfile)
    print(f"  Saved: {outfile}")


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
# PART A — ModEM (MT) processing
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
        print(
            f"  NOTE: SENS_FILE already ends with {SENS_EXT!r} — "
            f"stripping it before appending SENS_EXT, to avoid building "
            f"a nonexistent '...{SENS_EXT}{SENS_EXT}' path."
        )
        SENS_FILE = SENS_FILE[: -len(SENS_EXT)]
    sens_path = Path(SENS_FILE + SENS_EXT)
    if not sens_path.exists():
        print(
            f"  WARNING: {sens_path} not found — skipping sensitivity "
            f"shading/blanking (set USE_SENSITIVITY = False to silence)."
        )
    else:
        sdx, sdy, sdz, sens, sref, strans_in = mdm.read_mod(
            file=SENS_FILE, modext=SENS_EXT, trans="LINEAR", out=True
        )
        if sens.shape != mval.shape:
            print(
                f"  WARNING: {sens_path} has shape {sens.shape}, resistivity "
                f"model has {mval.shape} — meshes don't match, cannot use "
                f"for shading/blanking. Skipping."
            )
            sens = None
        elif not (
            np.allclose(sdx, dx)
            and np.allclose(sdy, dy)
            and np.allclose(sdz, dz)
        ):
            print(
                f"  WARNING: {sens_path} cell sizes don't match the "
                f"resistivity model's — meshes may not be identical. "
                f"Proceeding, but double-check this file is the right one."
            )

        if sens is not None and (SENS_FLIP_EASTING or SENS_FLIP_NORTHING):
            # sens has shape (North, East, Down) — same convention as mval
            # (see build_utm_axes). Flipping here, before trim/crop, keeps
            # every downstream product (3-D field, depth slices) consistent
            # without needing a second fix in the plot script.
            if SENS_FLIP_EASTING:
                sens = sens[:, ::-1, :]
                print(
                    "  Flipped sensitivity along East-West axis "
                    "(SENS_FLIP_EASTING = True)"
                )
            if SENS_FLIP_NORTHING:
                sens = sens[::-1, :, :]
                print(
                    "  Flipped sensitivity along North-South axis "
                    "(SENS_FLIP_NORTHING = True)"
                )

# Read the data file to get the geographic reference point and site coordinates.
# The .dat header line "> lat lon" gives the model origin in geographic coords.
# Data columns: Period Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error
#   col 2 = GG_Lat (°), col 3 = GG_Lon (°)
#   col 4 = X (m, North), col 5 = Y (m, East), col 6 = Z (m, positive down)
Site, Comp, Data, Head = mdm.read_data(
    Datfile=DATA_FILE, modext=DATA_EXT, out=True
)

# Extract geographic reference from the "> lat lon" header line
_ref_line = [
    l
    for l in Head
    if l.startswith(">")
    and len(l.split()) == 3
    and not any(
        c.isalpha()
        for c in l.replace(".", "").replace("-", "").replace(">", "").strip()
    )
]
if _ref_line and REFERENCE_LAT is None:
    _parts = _ref_line[0].split()
    lat_ref = float(_parts[1])
    lon_ref = float(_parts[2])
    print(
        f"  Geographic reference from .dat header: lat={lat_ref:.4f}°  lon={lon_ref:.4f}°"
    )
else:
    lat_ref = (
        REFERENCE_LAT
        if REFERENCE_LAT is not None
        else float(np.mean(Data[:, 1]))
    )
    lon_ref = (
        REFERENCE_LON
        if REFERENCE_LON is not None
        else float(np.mean(Data[:, 2]))
    )
    print(
        f"  Geographic reference (override/fallback): lat={lat_ref:.4f}°  lon={lon_ref:.4f}°"
    )

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
utm_e_km, utm_n_km = build_utm_axes(
    dx, dy, reference, transformer, lat_ref, lon_ref
)
utm_e_edges_km, utm_n_edges_km = build_utm_edges(
    dx, dy, reference, transformer, lat_ref, lon_ref
)
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
    dy = dy[sl_e]  # East cell widths, indexed like easting
    dx = dx[sl_n]  # North cell widths, indexed like northing
    mval = mval[sl_n, sl_e, :]
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
    print(
        f"  Boundary-cell edge overhang before clipping: "
        f"easting [{e_min - utm_e_edges_km[0]:+.2f}, "
        f"{utm_e_edges_km[-1] - e_max:+.2f}] km, "
        f"northing [{n_min - utm_n_edges_km[0]:+.2f}, "
        f"{utm_n_edges_km[-1] - n_max:+.2f}] km"
    )
    utm_e_edges_km[0] = max(utm_e_edges_km[0], e_min)
    utm_e_edges_km[-1] = min(utm_e_edges_km[-1], e_max)
    utm_n_edges_km[0] = max(utm_n_edges_km[0], n_min)
    utm_n_edges_km[-1] = min(utm_n_edges_km[-1], n_max)

    print(
        f"  Cropped easting  range: {utm_e_km.min():.1f} – {utm_e_km.max():.1f} km"
        f"  ({mval.shape[1]} cells)"
    )
    print(
        f"  Cropped northing range: {utm_n_km.min():.1f} – {utm_n_km.max():.1f} km"
        f"  ({mval.shape[0]} cells)"
    )

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
    dx=dx,
    dy=dy,
    dz=dz,
    mval=mval,
    ref=[0.0, 0.0, reference[2] + z_trim_offset_m],  # keep in sync with
    # build_depth_axis_km's ref_z above — both must use the same z
    # reference (reference[2] plus any top-z trim offset) or this
    # topography and the model's own depth axis go back out of alignment.
    mvalair=RHO_AIR,
    out=True,
)
# xcnt/ycnt are local offsets in metres, matching dx/dy after trimming.
# We use the already-built UTM axes instead.
save_topo(utm_e_km, utm_n_km, topo_m, outpath("modem_topo_utm.nc"))

# ------------------------------------------------------------------
# 7. Full 3-D model
# ------------------------------------------------------------------
print("\n=== Saving 3-D model ===")
save_3d_model(
    utm_e_km, utm_n_km, depth_km, rho_out, outpath("modem_model_utm.nc")
)
save_grid_edges(
    utm_e_edges_km,
    utm_n_edges_km,
    depth_edges_km,
    outpath("modem_grid_edges_utm.nc"),
)

if sens_out is not None:
    print("\n=== Saving 3-D sensitivity/resolution field ===")
    sens_long_name = {
        "LOG10": "log10 sensitivity",
        "LOGE": "ln sensitivity",
        "LINEAR": "sensitivity",
    }.get(SENS_TRANSFORM.upper(), "sensitivity")
    sens_units = {
        "LOG10": "log10(sensitivity)",
        "LOGE": "ln(sensitivity)",
        "LINEAR": "sensitivity",
    }.get(SENS_TRANSFORM.upper(), "sensitivity")
    save_3d_field(
        utm_e_km,
        utm_n_km,
        depth_km,
        sens_out,
        outpath("modem_sens_utm.nc"),
        sens_long_name,
        sens_units,
        SENS_TRANSFORM,
    )

# ------------------------------------------------------------------
# 7b. Full submesh point table (for clustering)
# ------------------------------------------------------------------
if EXPORT_SUBMESH_TABLE:
    print("\n=== Saving full submesh point table (for clustering) ===")
    _rho_long_name = {
        "LOG10": "log10 resistivity",
        "LOGE": "ln resistivity",
        "LINEAR": "resistivity",
    }.get(OUTPUT_TRANSFORM.upper(), "resistivity")
    _rho_units = {
        "LOG10": "log10(Ohm.m)",
        "LOGE": "ln(Ohm.m)",
        "LINEAR": "Ohm.m",
    }.get(OUTPUT_TRANSFORM.upper(), "Ohm.m")
    save_submesh_table(
        utm_e_km, utm_n_km, depth_km, rho_out,
        sens_out if sens_out is not None else None,
        _rho_long_name, _rho_units,
        sens_long_name if sens_out is not None else None,
        sens_units if sens_out is not None else None,
        outpath("modem_submesh_points.nc"),
    )

# ------------------------------------------------------------------
# 8. Depth slices
# ------------------------------------------------------------------
print("\n=== Saving depth slices ===")
for d_km in DEPTH_SLICES_KM:
    tag = f"{d_km:.0f}km" if d_km == int(d_km) else f"{d_km:.1f}km"
    save_depth_slice(
        utm_e_km,
        utm_n_km,
        depth_km,
        rho_out,
        d_km,
        outpath(f"modem_rho_utm_{tag}.nc"),
    )
    if sens_out is not None:
        save_depth_slice_field(
            utm_e_km,
            utm_n_km,
            depth_km,
            sens_out,
            d_km,
            outpath(f"modem_sens_utm_{tag}.nc"),
            sens_long_name,
            sens_units,
            SENS_TRANSFORM,
        )

# ------------------------------------------------------------------
# 9. Site positions from data file
# ------------------------------------------------------------------
print("\n=== Saving MT site positions ===")
# Site, Comp, Data, Head already loaded above.
# Use GG_Lat/GG_Lon (cols 1,2) for site geographic positions, and
# X(m)/Y(m)/Z(m) (cols 4,5,6) as the model Cartesian coordinates.
# Elevation: Z(m) is positive down in ModEM; negate for positive-up.
_, unique_idx = np.unique(Site, return_index=True)
unique_names = Site[unique_idx]
site_lats = Data[unique_idx, 1]  # GG_Lat, degrees
site_lons = Data[unique_idx, 2]  # GG_Lon, degrees
site_elevs = -Data[
    unique_idx, 6
]  # Z(m) positive down → negate for positive up

# Project geographic coords to UTM km
site_e_raw, site_n_raw = transformer.transform(site_lons, site_lats)
site_e_km = site_e_raw / 1e3
site_n_km = site_n_raw / 1e3

save_sites(
    site_e_km,
    site_n_km,
    site_elevs,
    unique_names.tolist(),
    outpath("modem_sites_utm.nc"),
)

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
    "modem_rho_utm_{}.nc".format(f"{d:.0f}km" if d == int(d) else f"{d:.1f}km")
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


# ==================================================================
# PART B — Seismic tomography processing (Vp, Vs, Vp/Vs, density)
# ==================================================================

left,  right  = TAR_LON
bottom, top   = TAR_LAT

# ------------------------------------------------------------------
# 1. Velocity subsets
# ------------------------------------------------------------------
print("Reading velocity models …")
vtomop  = xr.open_dataset(FNAME_VP)
vtomos  = xr.open_dataset(FNAME_VS)
print("Reading density model …")
vtomodens = xr.open_dataset(FNAME_DENS)

# Soft consistency check: dens is expected on the same (lat, lon, depth)
# grid as Vp/Vs, since it is not reprojected/regridded separately — only
# cropped and sliced exactly like them (see slice_to_utm_km_nc below,
# which assumes matching utm_easting/utm_northing coordinates). A mismatch
# here would silently misalign the density slices against the velocity
# ones without this warning.
for _dim in ("lat", "lon", "depth"):
    if _dim in vtomodens.dims and _dim in vtomop.dims:
        if not (vtomodens.sizes[_dim] == vtomop.sizes[_dim] and
                np.allclose(vtomodens[_dim].values, vtomop[_dim].values)):
            print(
                f"  WARNING: FD_rho_model.nc '{_dim}' grid does not match "
                f"FD_vp_model.nc — density slices may not align with "
                f"velocity slices. Proceeding anyway."
            )

if CROP_TO_REGION:
    vp  = vtomop.sel(lat=slice(bottom, top), lon=slice(left, right),
                     depth=slice(*DEPTH_RANGE))
    vs  = vtomos.sel(lat=slice(bottom, top), lon=slice(left, right),
                     depth=slice(*DEPTH_RANGE))
    dens = vtomodens.sel(lat=slice(bottom, top), lon=slice(left, right),
                     depth=slice(*DEPTH_RANGE))
    print(f"  Region: cropped to TAR_LON={TAR_LON}, TAR_LAT={TAR_LAT}")
else:
    vp  = vtomop.sel(depth=slice(*DEPTH_RANGE))
    vs  = vtomos.sel(depth=slice(*DEPTH_RANGE))
    dens = vtomodens.sel(depth=slice(*DEPTH_RANGE))
    print("  Region: full source-model extent (CROP_TO_REGION=False)")
vps = vp / vs

print(f"  Requested depth range: {DEPTH_RANGE[0]} – {DEPTH_RANGE[1]} km")
print(f"  Actual depth range in source model (Vp): "
      f"{float(vp['depth'].min()):.2f} – {float(vp['depth'].max()):.2f} km")
print(f"  Actual depth range in source model (dens): "
      f"{float(dens['depth'].min()):.2f} – {float(dens['depth'].max()):.2f} km")

vp  = add_utm_coords(vp)
vs  = add_utm_coords(vs)
vps = add_utm_coords(vps)
dens = add_utm_coords(dens)

safe_to_netcdf(vp, outpath(f"{SITE_PREFIX}_vp.nc"))
safe_to_netcdf(vs, outpath(f"{SITE_PREFIX}_vs.nc"))
safe_to_netcdf(vps, outpath(f"{SITE_PREFIX}_vps.nc"))
safe_to_netcdf(dens, outpath(f"{SITE_PREFIX}_dens.nc"))
print("Saved velocity/density subsets:")
print(f"  {outpath(f'{SITE_PREFIX}_vp.nc')}")
print(f"  {outpath(f'{SITE_PREFIX}_vs.nc')}")
print(f"  {outpath(f'{SITE_PREFIX}_vps.nc')}")
print(f"  {outpath(f'{SITE_PREFIX}_dens.nc')}")

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
geo_to_utm_km_nc(topo_da, outpath(f"{SITE_PREFIX}_topo_utm.nc"), spacing_km=TOPO_SPACING_KM)
geo_to_utm_km_nc(bath_da, outpath(f"{SITE_PREFIX}_bath_utm.nc"), spacing_km=TOPO_SPACING_KM)

# Note: {SITE_PREFIX}_topo_shade_utm.nc is intentionally NOT written here.
# Hillshade is computed on-the-fly in plot_seis.py via
# matplotlib.colors.LightSource, which gives equivalent results.

# ------------------------------------------------------------------
# 3. Per-depth UTM-km velocity slices
# ------------------------------------------------------------------
depth_coord = vp["depth"]

print("\nPre-computing per-depth UTM-km velocity/density slices …")
for d_index in DEPTH_INDEX:
    depth_km_val = int(depth_coord.item(d_index))
    tag = f"{depth_km_val}km"
    slice_to_utm_km_nc(vp["data"].isel(depth=d_index),  outpath(f"{SITE_PREFIX}_vp_utm_{tag}.nc"))
    slice_to_utm_km_nc(vs["data"].isel(depth=d_index),  outpath(f"{SITE_PREFIX}_vs_utm_{tag}.nc"))
    slice_to_utm_km_nc(vps["data"].isel(depth=d_index), outpath(f"{SITE_PREFIX}_vps_utm_{tag}.nc"))
    slice_to_utm_km_nc(dens["data"].isel(depth=d_index), outpath(f"{SITE_PREFIX}_dens_utm_{tag}.nc"))

print("\nDone (Part B). All seismic + density UTM-km grids ready.")
