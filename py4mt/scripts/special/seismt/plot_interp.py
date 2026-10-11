#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_interp.py
==============
Plotting script for interpolate.py output files
(``{SITE_PREFIX}_interp_{METHOD}.nc`` — one dataset per site/method run).

Produces depth-slice maps and vertical cross-sections of any variable
stored in the interpolated NetCDF, sharing the same basemap engine,
styling conventions, and USER SETTINGS layout as the rest of this
pipeline's plot scripts (``plot_seis.py``, ``plot_femtic_mesh.py``).

Data format
-----------
Reads the UTM-km 3-D grid written by ``interpolate.py``::

    dims   : (depth, northing, easting)
    coords : depth    [km, positive down — uniform OR geometrically
                       increasing, depending on GRID_DEPTH_KM in
                       interpolate.py]
             northing [km, UTM, uniform]
             easting  [km, UTM, uniform]
    vars   : one per interpolated source variable (e.g. rho_femtic, vps,
             dens) plus optional gradient fields (``*_grad_easting``, etc.)

The depth coordinate may be non-uniform (geometric layer spacing).
All plotting routines use the actual coordinate values from the file, so
they render correctly for both uniform and non-uniform depth axes.
Each variable gets its own colour-scale / colormap entry in CMAP_SETTINGS.

Depth slices
------------
Horizontal maps at each depth listed in DEPTH_SLICES_KM.  Each figure is
a single map panel with optional topo basemap, feature overlays, and
colorbar — same layout as plot_seis.py / plot_femtic_mesh.py depth maps.

Vertical sections
-----------------
Defined in VSLICES (same schema as the other scripts).  Values are
interpolated bilinearly along the profile line from the 3-D grid.  Topo
line and seismicity projection follow the same conventions as plot_seis.py.

Dependencies
------------
    numpy, matplotlib, xarray, pandas, scipy
plus the local ``tomomt.py`` module.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic), 2026-08.
License: GNU General Public License v3 (GPL-3.0-or-later).
AI-generated code — review before use in production.
"""

import os
import sys
import warnings

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker
from matplotlib.path import Path as MplPath
from scipy.interpolate import RegularGridInterpolator

import tomomt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# Colormap / save helpers forwarded from tomomt
# ---------------------------------------------------------------------
load_colormap = tomomt.load_colormap
export_colormap_to_cpt = tomomt.export_colormap_to_cpt


# =====================================================================
# USER SETTINGS
# =====================================================================

# --- Site / file selection ---
# Must match SITE_PREFIX in precompute.py / interpolate.py.
SITE_PREFIX = "ubinas"

# Interpolation method tag used in the filename
# (e.g. "nearest", "linear", "kriging" — whatever interpolate.py wrote).
INTERP_METHOD = "nearest"

# Full path to the interpolated NetCDF (leave None to auto-build from
# SITE_PREFIX and INTERP_METHOD: ``{SITE_PREFIX}_interp_{INTERP_METHOD}.nc``).
INTERP_FILE = None

# Directory to read the interpolated NetCDF from (also used for topo/bath
# files when NC_DIR_TOPO is None — set that below if they live elsewhere).
NC_DIR = "../precompute/ubinas_small/"

# Directory for topo/bath NetCDF files. None = same as NC_DIR.
NC_DIR_TOPO = None

# Directory for saved figures (created if it doesn't exist).
PLOT_DIR = "../plots_interp/"

PLOT_FORMATS = [".pdf"]
PLOT_DPI = 600
SHOW_PLOTS = False


# --- Variables to plot ---
# List of variable names from the NetCDF to plot (e.g. "rho_femtic",
# "vps", "dens", "rho_femtic_grad_mag", …).  Every name must exist in
# the dataset; the script exits with an error for any that don't.
PLOT_VARS = ["rho_femtic", "dens", "vps", "vp", "dens_grad_mag",  "vp_grad_mag"]


# --- Per-variable colour-scale and colormap ---
# Keys are variable names (matching PLOT_VARS).  Each entry is a dict:
#   cmap  : matplotlib name or path to a .cpt / plain-RGB file
#   cmin  : colour-scale minimum
#   cmax  : colour-scale maximum
#   label : colorbar label string
#   alpha : data transparency (0 = opaque, 1 = invisible)
#
# Variables not listed here fall back to CMAP_DEFAULT below.
CMAP_SETTINGS = {
    "rho_femtic": dict(
        cmap  = "jet_r",
        cmin  = 0.0,
        cmax  = 3.0,
        label = "log₁₀ ρ (Ω·m)",
        alpha = 0.35,
    ),
    "vps": dict(
        cmap  = "../colormaps/hotr_vps_ub.cpt",
        cmin  = 1.70,
        cmax  = 2.10,
        label = "Vp/Vs",
        alpha = 0.50,
    ),
    "dens": dict(
        cmap  = "jet_r",
        cmin  = 1500,
        cmax  = 3300,
        label = "Density (kg/m³)",
        alpha = 0.35,
    ),
    "rho_femtic_grad_mag": dict(
        cmap  = "jet",
        cmin  = 0.0,
        cmax  = 0.5,
        label = "‖∇log₁₀ρ‖ ((log₁₀Ω·m)/km)",
        alpha = 0.35,
    ),
    "vps_grad_mag": dict(
        cmap  = "hot_r",
        cmin  = 0.0,
        cmax  = 0.02,
        label = "‖∇(Vp/Vs)‖ ((m/s)/km)",
        alpha = 0.35,
    ),
    "dens_grad_mag": dict(
        cmap  = "jet",
        cmin  = None,
        cmax  = 40,
        label = "‖∇dens‖ (kg/m3)/km)",
        alpha = 0.35,
    ),
}

# Fallback for any variable not in CMAP_SETTINGS.
CMAP_DEFAULT = dict(
    cmap  = "viridis",
    cmin  = None,   # None → auto from finite data range
    cmax  = None,
    label = "",     # falls back to the variable's long_name attr
    alpha = 0.35,
)

# --- Depth slices ---
# Requested depths in km (positive down). Each value is matched to the
# nearest depth level actually present in the NetCDF — no interpolation.
# Works for both uniform and geometrically-spaced depth axes (set in
# GRID_DEPTH_KM in interpolate.py); the actual matched depth is printed
# and shown in the figure title, so approximate values are fine here.
DEPTH_SLICES_KM = [-3, 2, 6, 14]

# Seismicity depth window for each depth slice (km), one pair per entry
# in DEPTH_SLICES_KM.  Set both to None to show all seismicity.
ZMIN_SEISM = [None, None, None, None]
ZMAX_SEISM = [None, None, None, None]

if not (len(ZMIN_SEISM) == len(ZMAX_SEISM) == len(DEPTH_SLICES_KM)):
    sys.exit(
        f"ZMIN_SEISM ({len(ZMIN_SEISM)}), ZMAX_SEISM ({len(ZMAX_SEISM)}), and "
        f"DEPTH_SLICES_KM ({len(DEPTH_SLICES_KM)}) must all have the same length."
    )

# --- Isolines ---
ISO_LINES_MAP    = False
ISO_LINES_VSLICE = False
ISO_LEVELS_MAP    = "auto"
ISO_LEVELS_VSLICE = "auto"
ISO_AUTO_N = 6
ISO_STYLE  = dict(colors="black", linewidths=0.6, linestyles="solid", zorder=7)
ISO_LABEL           = True
ISO_LABEL_FMT       = "%.2g"
ISO_LABEL_FONTSIZE  = 6

# --- Figure sizes ---
FIG_WIDTH = 10.0        # cm — map panel width

VSLICE_WIDTH_CM  = None
VSLICE_HEIGHT_CM = 10.0  # None → derived from VE and profile length

# --- Map region ---
# "data" uses the interpolated grid's own extent; "topo" uses the topo grid.
REGION_SOURCE    = "data"
REGION_MARGIN_KM = 0.0

# Explicit override (UTM km). None → use REGION_SOURCE extent.
MAP_XLIM = None   # e.g. [280., 320.]
MAP_YLIM = None   # e.g. [8170., 8215.]

# --- Topo / basemap ---
NC_TOPO = f"{SITE_PREFIX}_topo_utm.nc"
NC_BATH = f"{SITE_PREFIX}_bath_utm.nc"

SHOW_TOPO_BASEMAP = False
TOPO_VMIN = 1000
TOPO_VMAX = 6000
OCEAN_COLOR = "#6baed6"

HS_AZIMUTH  = 315
HS_ALTITUDE = 45
HS_SIGMA    = 1.0

# --- Axes units ---
AXES_UNITS      = "km"    # "km" | "latlon"
LATLON_NTICKS   = 5
LATLON_DECIMALS = 2
AXES_KM_COMMA   = True

# =====================================================================
# COLORBAR SETTINGS
# =====================================================================
SHOW_COLORBAR       = True
COLORBAR_POSITION   = "right"
COLORBAR_SIZE       = 0.85
COLORBAR_PAD        = 0.10
COLORBAR_ASPECT     = 20
COLORBAR_LABEL_SIZE = 12
COLORBAR_TICK_SIZE  = 12
COLORBAR_NTICKS     = 7

# =====================================================================
# AXIS FONT SETTINGS
# =====================================================================
AXIS_LABEL_SIZE = 12
AXIS_TICK_SIZE  = 12
AXIS_TITLE_SIZE = 12

# =====================================================================
# MAP FEATURE LAYERS
# =====================================================================
SHOW_PROFILE_LINES    = True
SHOW_VSLICE_LINES     = True
SHOW_SEISMICITY       = False
SHOW_MT_SITES         = True
SHOW_SEISMIC_SITES    = True
SHOW_VOLCANOES        = True
SHOW_VOLCANOES_ACTIVE = True
SHOW_CITIES           = True
SHOW_NORTH_ARROW      = True

# --- Feature CSV paths ---
CSV_VOLCANES      = "../features/volcanes.csv"
CSV_SEISMCAT      = "../features/catalog_welllocated_15_simple5.csv"
CSV_MT_SITES      = "../features/done/MTubinas_Sitelist.csv"
CSV_CITIES        = "../features/cities.csv"
CSV_SEISMIC_SITES = "../features/seismic_sites.csv"

VOLC_LABEL_IDX   = [5, 12, 13]
VOLC_NAME_COL    = "NAME"

# North arrow anchor (lon, lat) and shaft length (km)
ARROW_LON    = -73.6
ARROW_LAT    = -18.1
ARROW_LEN_KM = 4.0

# --- Static profile lines (drawn on every map) ---
PROFILE_1_LON = [-70.79, -70.95]
PROFILE_1_LAT = [-16.52, -16.22]
PROFILE_2_LON  = [-71.05, -70.73]
PROFILE_2_LAT  = [-16.459, -16.21]

# =====================================================================
# MARKER & LABEL STYLE SETTINGS
# =====================================================================
PROFILE_1_STYLE = dict(color="black", lw=0.4, zorder=10)
PROFILE_2_STYLE  = dict(color="gray",  lw=0.4, zorder=10)

EQ_MARKER_STYLE = dict(
    marker="o", s=4.5,
    facecolors="white", edgecolors="black", linewidths=0.2, zorder=11,
)

_MT_PIN_VERTS = [(0.0, 0.0), (-1.0, 1.732), (1.0, 1.732), (0.0, 0.0)]
_MT_PIN_CODES = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
MT_PIN_MARKER = MplPath(_MT_PIN_VERTS, _MT_PIN_CODES)

MT_MARKER_STYLE = dict(
    marker="v", s=10,
    facecolors="yellow", edgecolors="black", linewidths=0.7, zorder=12,
)
MT_LABEL_STYLE = dict(
    fontsize=5, color="black", zorder=14, rotation=90,
    offset_x=0.3, offset_y=0.3, mode="none",
)

SEISMIC_SITES_MARKER_STYLE = dict(
    marker="v", s=10,
    facecolors="green", edgecolors="black", linewidths=0.7, zorder=12,
)

VOLC_LABEL_CHARS = 4
VOLC_INACT_MARKER_STYLE = dict(
    marker="^", s=10,
    facecolors="blue", edgecolors="black", linewidths=0.7, zorder=13,
)
VOLC_LABEL_STYLE = dict(
    fontsize=6, fontweight="bold", color="black", zorder=14,
    offset_x=0.3, offset_y=0.3,
    mode=f"first{VOLC_LABEL_CHARS}",
)

VOLC_ACT_MARKER_STYLE = dict(
    marker="^", s=10,
    facecolors="red", edgecolors="black", linewidths=0.7, zorder=13,
)

CITY_MARKER_STYLE = dict(
    marker="s", s=6,
    facecolors="black", edgecolors="black", linewidths=0.2, zorder=13,
)
CITY_LABEL_STYLE = dict(
    fontsize=6, color="white", zorder=14,
    offset_x=0.3, offset_y=-0.3, mode="full",
)

ARROW_STYLE       = dict(color="dimgray", lw=2, mutation_scale=14)
ARROW_LABEL_STYLE = dict(fontsize=9, fontweight="bold", color="dimgray")

# =====================================================================
# VERTICAL SLICE SETTINGS
#
# Same schema as plot_seis.py / plot_femtic_mesh.py:
#   name     : str   — label used in title and output filename
#   p1, p2   : [x, y] — endpoints, UTM km or lon/lat (see coord)
#   coord    : "utm" | "latlon"
#   zmin_km  : top  of section (km, positive down; negative = above sea level)
#   zmax_km  : base of section (km)
#   npts     : number of sample points along the profile
#   swath_km : half-width for seismicity projection
#   xlim     : optional [xmin, xmax] — crop displayed x-axis (units from
#              VSLICE_X_AXIS: UTM km or distance km).  None = full range.
#   ylim     : optional [ytop, ybottom] — crop displayed depth-axis (km).
#              None = topo/headroom-derived top to zmax_km.
# =====================================================================
VSLICES = [
    dict(
        name="profile AA'", 
        p1 = [PROFILE_1_LON[0], PROFILE_1_LAT[0]],   # lon, lat
        p2 = [PROFILE_1_LON[1], PROFILE_1_LAT[1]],
        coord="latlon", 
        zmin_km=-8.0, 
        zmax_km=20.0, 
        swath_km=10.0,
    ),
    dict(
        name="profile BB'", 
        p1 = [PROFILE_2_LON[0], PROFILE_2_LAT[0]],   # lon, lat
        p2 = [PROFILE_2_LON[1], PROFILE_2_LAT[1]],
        coord="latlon", 
        zmin_km=-8.0, 
        zmax_km=20.0, 
        swath_km=10.0,
    ),
]


# Colour-scale for vertical slices: defaults to the same as the maps;
# override per-variable by adding entries here (same dict keys as CMAP_SETTINGS).
VSLICE_CMAP_OVERRIDES = {}

VSLICE_VE           = 1.0
VSLICE_EQUAL_SCALE  = False
VSLICE_VE_POS       = "lower right"
VSLICE_VE_STYLE     = dict(fontsize=7, color="black")
VSLICE_X_AXIS       = "distance"   # "distance" | "utm"

VSLICE_EQ_STYLE = dict(
    marker="o", s=4.5,
    facecolors="white", edgecolors="black", linewidths=0.2, zorder=11,
)
VSLICE_TOPO_STYLE        = dict(color="dimgray", lw=0.5, zorder=12)
VSLICE_SHOW_TOPO_FILL    = False
VSLICE_TOPO_LAND_COLOR   = "gray"
VSLICE_TOPO_OCEAN_COLOR  = "#6baed6"
VSLICE_TOPO_HEADROOM_KM  = 1.0
VSLICE_MAP_LINE_STYLE    = dict(color="magenta", lw=0.8, ls="--", zorder=15)

# --- Free-text annotation ---
ANNOTATION_TEXT  = None
ANNOTATION_POS   = (0.01, 0.99)
ANNOTATION_STYLE = dict(fontsize=7, color="gray", ha="left", va="top")

# =====================================================================
# END USER SETTINGS
# =====================================================================

os.makedirs(PLOT_DIR, exist_ok=True)

_NC_DIR_TOPO = NC_DIR_TOPO if NC_DIR_TOPO is not None else NC_DIR


def ncpath(name, topo=False):
    """Join a bare filename onto NC_DIR (or NC_DIR_TOPO for topo/bath files)."""
    d = _NC_DIR_TOPO if topo else NC_DIR
    return tomomt.resolve_path(d, name)


def _interp_path():
    if INTERP_FILE is not None:
        return INTERP_FILE
    return ncpath(f"{SITE_PREFIX}_interp_{INTERP_METHOD}.nc")


# ------------------------------------------------------------------
# Utility helpers delegated to tomomt
# ------------------------------------------------------------------
load_colormap       = tomomt.load_colormap
to_utm_km           = tomomt.to_utm_km
compute_hillshade   = tomomt.compute_hillshade


def save_fig(fig, stem):
    return tomomt.save_fig(fig, stem, PLOT_DIR, PLOT_FORMATS, PLOT_DPI)


def _maybe_show():
    tomomt.maybe_show(SHOW_PLOTS)


def draw_annotation(ax):
    tomomt.draw_annotation(ax, ANNOTATION_TEXT, ANNOTATION_POS, ANNOTATION_STYLE)


def _region():
    return (xmin, xmax, ymin, ymax)


def _colorbar_settings():
    return dict(
        show=SHOW_COLORBAR, position=COLORBAR_POSITION,
        size=COLORBAR_SIZE, pad=COLORBAR_PAD, aspect=COLORBAR_ASPECT,
        label_size=COLORBAR_LABEL_SIZE, tick_size=COLORBAR_TICK_SIZE,
        nticks=COLORBAR_NTICKS, title_size=AXIS_TITLE_SIZE,
    )


_resolve_ve_pos = tomomt.resolve_ve_pos


def _in_region(xe, yn):
    return tomomt.in_region(xe, yn, _region())


def clipped_markers(ax, xe, yn, **kwargs):
    tomomt.clipped_markers(ax, xe, yn, _region(), **kwargs)


clipped_scatter = clipped_markers  # backward-compat alias


def clipped_labels(ax, xe, yn, labels, style_dict):
    tomomt.clipped_labels(ax, xe, yn, labels, style_dict, _region())


def draw_north_arrow(ax, x_km, y_km, length_km=4.0):
    tomomt.draw_north_arrow(ax, x_km, y_km, _region(),
                             ARROW_STYLE, ARROW_LABEL_STYLE, length_km)


def draw_iso_contours(ax, x, y, data2d, levels_spec, var):
    return tomomt.draw_iso_contours(
        ax, x, y, data2d, levels_spec, ISO_STYLE,
        n_auto=ISO_AUTO_N, label=ISO_LABEL,
        label_fmt=ISO_LABEL_FMT, label_fontsize=ISO_LABEL_FONTSIZE, key=var,
    )


def add_latlon_ticks(ax):
    tomomt.add_latlon_ticks(ax, _region(), LATLON_NTICKS, LATLON_DECIMALS,
                             AXIS_LABEL_SIZE, AXIS_TICK_SIZE)


# ------------------------------------------------------------------
# Figure factory helpers
# ------------------------------------------------------------------
def create_map_figure():
    map_w_in = FIG_WIDTH / 2.54
    map_h_in = map_w_in * (ymax - ymin) / (xmax - xmin)
    return tomomt.build_panel_figure(map_w_in, map_h_in,
                                      _colorbar_settings(), size_label="map")


def _vslice_colorbar_settings():
    settings = _colorbar_settings()
    ref_len_in = (VSLICE_HEIGHT_CM if VSLICE_HEIGHT_CM is not None
                  else VSLICE_WIDTH_CM) / 2.54
    settings["thickness_in"] = (COLORBAR_SIZE * ref_len_in) / COLORBAR_ASPECT
    return settings


def create_section_figure(w_in, h_in):
    return tomomt.build_panel_figure(w_in, h_in,
                                      _vslice_colorbar_settings(), size_label="section")


def finish_panel_colorbar(cax, mappable, label):
    return tomomt.finish_panel_colorbar(cax, mappable, label, _colorbar_settings())


def finish_section_colorbar(cax, mappable, label):
    return tomomt.finish_panel_colorbar(cax, mappable, label,
                                         _vslice_colorbar_settings())


# ------------------------------------------------------------------
# Per-variable colour-scale resolution
# ------------------------------------------------------------------
def _var_cfg(var, ds):
    """Return (cmap, cmin, cmax, label, alpha) for a variable."""
    cfg = dict(CMAP_DEFAULT)
    cfg.update(CMAP_SETTINGS.get(var, {}))

    cmap = load_colormap(cfg["cmap"])

    # Auto-range from finite data if not specified
    cmin, cmax = cfg["cmin"], cfg["cmax"]
    if cmin is None or cmax is None:
        vals = ds[var].values.ravel()
        finite = vals[np.isfinite(vals)]
        if len(finite):
            cmin = float(finite.min()) if cmin is None else cmin
            cmax = float(finite.max()) if cmax is None else cmax
        else:
            cmin, cmax = 0.0, 1.0

    label = cfg["label"] or ds[var].attrs.get("long_name", var)
    alpha = cfg["alpha"]
    return cmap, cmin, cmax, label, alpha


def _var_cfg_vslice(var, ds):
    """Like _var_cfg but applies VSLICE_CMAP_OVERRIDES first."""
    cfg = dict(CMAP_DEFAULT)
    cfg.update(CMAP_SETTINGS.get(var, {}))
    cfg.update(VSLICE_CMAP_OVERRIDES.get(var, {}))

    cmap = load_colormap(cfg["cmap"])
    cmin, cmax = cfg["cmin"], cfg["cmax"]
    if cmin is None or cmax is None:
        vals = ds[var].values.ravel()
        finite = vals[np.isfinite(vals)]
        cmin = float(finite.min()) if cmin is None and len(finite) else (cmin or 0.0)
        cmax = float(finite.max()) if cmax is None and len(finite) else (cmax or 1.0)
    label = cfg["label"] or ds[var].attrs.get("long_name", var)
    alpha = cfg["alpha"]
    return cmap, cmin, cmax, label, alpha


# ------------------------------------------------------------------
# lon/lat secondary axes
# ------------------------------------------------------------------
_profile_utm_km     = tomomt.profile_utm_km
_profile_labels     = tomomt.profile_labels
_sample_profile_points = tomomt.sample_profile_points


# ==================================================================
# Vertical slice engine
# ==================================================================

def compute_vertical_slice(vslice, var, ds):
    """
    Sample ``var`` along a vertical profile from the 3-D UTM-km grid in
    ``ds``.  Easting and northing are interpolated bilinearly at each
    horizontal position; depth is sampled at the grid's own depth levels
    (which may be non-uniform if GRID_DEPTH_KM used the geometric 'inc'
    mode in interpolate.py).  pcolormesh with shading="nearest" then
    renders each depth layer at its correct height automatically.

    Returns
    -------
    dist_km   : 1-D distance along profile (km)
    depth_km  : 1-D depth axis from ds (km)
    section   : 2-D array (ndepth, npts)
    e_ends    : [e0, e1] UTM km
    n_ends    : [n0, n1] UTM km
    topo_prof : 1-D surface elevation (m) along profile, or None
    utm_x     : 1-D x-axis for "utm" mode
    utm_xlabel: str label for that axis
    """
    e_ends, n_ends = _profile_utm_km(vslice)
    npts = vslice.get("npts", 200)

    dist_km, e_pts, n_pts, utm_x, utm_xlabel = \
        _sample_profile_points(e_ends, n_ends, npts)

    depth_km  = ds["depth"].values.astype(float)
    north_km  = ds["northing"].values.astype(float)
    east_km   = ds["easting"].values.astype(float)
    vals_3d   = ds[var].values.astype(float)   # (ndepth, nnorth, neast)

    # Ensure ascending coordinate axes for the interpolator
    if north_km[0] > north_km[-1]:
        north_km = north_km[::-1]
        vals_3d  = vals_3d[:, ::-1, :]
    if east_km[0] > east_km[-1]:
        east_km = east_km[::-1]
        vals_3d = vals_3d[:, :, ::-1]
    if depth_km[0] > depth_km[-1]:
        depth_km = depth_km[::-1]
        vals_3d  = vals_3d[::-1, :, :]

    interp = RegularGridInterpolator(
        (depth_km, north_km, east_km), vals_3d,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    nz = len(depth_km)
    d_q = np.repeat(depth_km, npts)
    n_q = np.tile(n_pts, nz)
    e_q = np.tile(e_pts, nz)
    section = interp(np.column_stack([d_q, n_q, e_q])).reshape(nz, npts)

    # Topography along profile
    topo_prof = None
    if topo_z is not None:
        ty = topo_y if topo_y[0] < topo_y[-1] else topo_y[::-1]
        tz = topo_z if topo_y[0] < topo_y[-1] else topo_z[::-1, :]
        topo_interp = RegularGridInterpolator(
            (ty, topo_x), tz,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        topo_prof = topo_interp(np.column_stack([n_pts, e_pts]))

    return dist_km, depth_km, section, e_ends, n_ends, topo_prof, utm_x, utm_xlabel


def _project_seismicity_to_profile(e_ends, n_ends, swath_km, zmin_km, zmax_km):
    return tomomt.project_points_to_profile(
        eq_e0, eq_n0, e_ends, n_ends, swath_km,
        z0=zeqs, zmin_km=zmin_km, zmax_km=zmax_km,
    )


def plot_vertical_slice(dist_km, depth_km, section,
                        e_ends, n_ends, topo_prof, utm_x, utm_xlabel,
                        lbl_start, lbl_end, vslice,
                        var, cmap, cmin, cmax, cbar_label, alpha,
                        stem):
    """Produce and save one vertical cross-section figure."""
    swath  = vslice.get("swath_km", 10.0)
    zmin_s = vslice.get("zmin_km", depth_km[0])
    zmax_s = vslice.get("zmax_km", depth_km[-1])
    ve     = 1.0 if VSLICE_EQUAL_SCALE else VSLICE_VE
    name   = vslice.get("name", "profile")

    if VSLICE_X_AXIS == "distance":
        x_arr   = dist_km
        x_label = "Distance along profile (km)"
    else:
        x_arr   = utm_x
        x_label = utm_xlabel

    eq_dist, eq_dep = _project_seismicity_to_profile(
        e_ends, n_ends, swath, zmin_s, zmax_s)
    if len(eq_dist):
        eq_x = np.interp(eq_dist, dist_km, x_arr)
    else:
        eq_x = eq_dist

    profile_len = dist_km[-1]
    depth_range = depth_km[-1] - depth_km[0]
    if VSLICE_HEIGHT_CM is not None:
        h_in = VSLICE_HEIGHT_CM / 2.54
        w_in = h_in * profile_len / (depth_range * ve)
    else:
        w_in = VSLICE_WIDTH_CM / 2.54
        h_in = w_in * (depth_range * ve) / profile_len
    print(f"    Section size: {w_in:.2f} × {h_in:.2f} in")

    fig, ax, cax = create_section_figure(w_in, h_in)

    if ve != 1.0:
        vx, vy, vha, vva = _resolve_ve_pos(VSLICE_VE_POS)
        ax.text(vx, vy, f"VE = {ve:.1f}×",
                transform=ax.transAxes, ha=vha, va=vva,
                zorder=21, **VSLICE_VE_STYLE)

    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    # shading="nearest": one solid rectangle per sample point, no blending —
    # matches the flat-cell rendering used for depth-slice maps.
    im = ax.pcolormesh(
        x_arr, depth_km, section,
        cmap=cmap, norm=norm, shading="nearest",
        alpha=1.0 - alpha, zorder=5,
    )

    if ISO_LINES_VSLICE:
        draw_iso_contours(ax, x_arr, depth_km, section, ISO_LEVELS_VSLICE, var)

    y_top = depth_km[0]
    if topo_prof is not None:
        surf_depth = -topo_prof / 1e3
        finite_surf = surf_depth[np.isfinite(surf_depth)]
        if finite_surf.size > 0:
            # At least some of the profile is inside the topo grid.
            y_top = float(finite_surf.min()) - VSLICE_TOPO_HEADROOM_KM

            if VSLICE_SHOW_TOPO_FILL:
                land  = topo_prof >  0
                ocean = topo_prof <= 0
                if land.any():
                    ax.fill_between(x_arr, 0.0, surf_depth, where=land,
                                    color=VSLICE_TOPO_LAND_COLOR, alpha=0.5,
                                    zorder=6, interpolate=True)
                if ocean.any():
                    ax.fill_between(x_arr, 0.0, surf_depth, where=ocean,
                                    color=VSLICE_TOPO_OCEAN_COLOR, alpha=0.5,
                                    zorder=6, interpolate=True)
            ax.plot(x_arr, surf_depth, **VSLICE_TOPO_STYLE)
        else:
            # Profile falls entirely outside the topo grid — skip topo
            # line/fill and keep y_top = depth_km[0].
            print(f"    WARNING: topo profile is all-NaN for '{vslice.get('name', 'profile')}' "
                  f"— profile may be outside the topo grid extent. Topo line skipped.")

    if SHOW_SEISMICITY and len(eq_x):
        tomomt.markers(ax, eq_x, eq_dep, **VSLICE_EQ_STYLE)

    x0, x1 = x_arr[0], x_arr[-1]
    xlim = vslice.get("xlim", None)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(min(x0, x1), max(x0, x1))
    ylim = vslice.get("ylim", None)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(y_top, depth_km[-1])
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Depth (km)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=AXIS_TICK_SIZE)

    for xpos, lbl in ((x0, lbl_start), (x1, lbl_end)):
        ax.text(xpos, y_top, lbl,
                ha="center", va="bottom",
                fontsize=AXIS_LABEL_SIZE, fontweight="bold",
                color="black", clip_on=False, zorder=20)

    ax.set_title(f"{var} — {name}", fontsize=AXIS_TITLE_SIZE)
    finish_section_colorbar(cax, im, cbar_label)
    draw_annotation(ax)
    save_fig(fig, stem)
    _maybe_show()
    plt.close(fig)


# ==================================================================
# Load static grids
# ==================================================================
print("Loading topo grid …")
_topo_da = xr.open_dataarray(ncpath(NC_TOPO, topo=True))
if "x" in _topo_da.dims:
    topo_x = _topo_da["x"].values
    topo_y = _topo_da["y"].values
elif "easting" in _topo_da.dims:
    topo_x = _topo_da["easting"].values
    topo_y = _topo_da["northing"].values
else:
    raise ValueError(f"Cannot identify spatial dims in {NC_TOPO}")
topo_z = _topo_da.values
_topo_da.close()

if topo_z.shape[0] != len(topo_y):
    topo_z = topo_z.T
if topo_y[0] > topo_y[-1]:
    topo_y = topo_y[::-1]
    topo_z = topo_z[::-1, :]

dx_km = float(np.median(np.diff(topo_x)))
dy_km = float(np.median(np.diff(topo_y)))
topo_extent = [topo_x.min(), topo_x.max(), topo_y.min(), topo_y.max()]

if SHOW_TOPO_BASEMAP:
    print("Computing hillshade …")
    topo_hs   = compute_hillshade(topo_z, dx_km, dy_km, HS_AZIMUTH, HS_ALTITUDE, HS_SIGMA)
    topo_norm = mcolors.Normalize(vmin=TOPO_VMIN, vmax=TOPO_VMAX)
else:
    topo_hs, topo_norm = None, None
CMAP_TOPO = plt.get_cmap("gray")

print("Loading bathymetry grid …")
_use_bath = True
try:
    _bath_da = xr.open_dataarray(ncpath(NC_BATH, topo=True))
    bath_x = (_bath_da["x"].values if "x" in _bath_da.dims
               else _bath_da["easting"].values)
    bath_y = (_bath_da["y"].values if "y" in _bath_da.dims
               else _bath_da["northing"].values)
    bath_z = _bath_da.values
    if bath_z.shape[0] != len(bath_y):
        bath_z = bath_z.T
    bath_extent = [bath_x.min(), bath_x.max(), bath_y.min(), bath_y.max()]
    _bath_da.close()
except FileNotFoundError:
    print(f"  WARNING: {NC_BATH} not found — ocean fill disabled.")
    _use_bath = False
    bath_x = bath_y = bath_z = bath_extent = None


# ==================================================================
# Load interpolated data (for region + variable validation)
# ==================================================================
print(f"Loading interpolated data from: {_interp_path()}")
_ds_meta = xr.open_dataset(_interp_path())
_missing = [v for v in PLOT_VARS if v not in _ds_meta]
if _missing:
    sys.exit(f"Variables not found in dataset: {_missing}\n"
             f"Available: {list(_ds_meta.data_vars)}")
print(f"  dims: {dict(_ds_meta.dims)}")
print(f"  vars: {list(_ds_meta.data_vars)}")
_ds_meta.close()


# ==================================================================
# Map region
# ==================================================================
if REGION_SOURCE == "data":
    _ds_r = xr.open_dataset(_interp_path())
    _e = _ds_r["easting"].values
    _n = _ds_r["northing"].values
    _ds_r.close()
    xmin = float(_e.min()) - REGION_MARGIN_KM
    xmax = float(_e.max()) + REGION_MARGIN_KM
    ymin = float(_n.min()) - REGION_MARGIN_KM
    ymax = float(_n.max()) + REGION_MARGIN_KM
    print(f"Region: interpolated grid extent + {REGION_MARGIN_KM} km margin")
else:
    xmin, xmax = topo_x.min(), topo_x.max()
    ymin, ymax = topo_y.min(), topo_y.max()
    print("Region: topo grid extent")

if MAP_XLIM is not None:
    xmin, xmax = MAP_XLIM
if MAP_YLIM is not None:
    ymin, ymax = MAP_YLIM
print(f"Map region (km): easting [{xmin:.2f}, {xmax:.2f}], "
      f"northing [{ymin:.2f}, {ymax:.2f}]")


# ==================================================================
# Feature layers
# ==================================================================
volcanes = pd.read_csv(CSV_VOLCANES)
utmv_e, utmv_n = to_utm_km(
    volcanes["LONG"][VOLC_LABEL_IDX].values,
    volcanes["LAT"][VOLC_LABEL_IDX].values,
)
if VOLC_NAME_COL in volcanes.columns:
    namev = volcanes[VOLC_NAME_COL][VOLC_LABEL_IDX].values
else:
    print(f"  WARNING: volcano name column {VOLC_NAME_COL!r} not found.")
    namev = [""] * len(VOLC_LABEL_IDX)

volc_act_e, volc_act_n = [], []
for i in range(len(volcanes)):
    if "ACT" in str(volcanes["ESTADO"][i]):
        ae, an = to_utm_km([volcanes["LONG"][i]], [volcanes["LAT"][i]])
        volc_act_e.append(ae[0])
        volc_act_n.append(an[0])

eqs    = pd.read_csv(CSV_SEISMCAT, delimiter=" ")
eq_e0, eq_n0 = to_utm_km(eqs["x"].values, eqs["y"].values)
zeqs   = eqs["z"].values

_mt    = pd.read_csv(CSV_MT_SITES, delimiter=" ")
mt_e, mt_n = to_utm_km(_mt["x"].values, _mt["y"].values)
for _nc in ("Site", "site", "name", "Name", "station", "Station"):
    if _nc in _mt.columns:
        mt_names = _mt[_nc].astype(str).tolist()
        break
else:
    mt_names = [""] * len(_mt)

seis_sites   = pd.read_csv(CSV_SEISMIC_SITES, header=None,
                             names=["network", "station", "lat", "lon", "elev_m"])
seis_site_e, seis_site_n = to_utm_km(seis_sites["lon"].values,
                                       seis_sites["lat"].values)
seis_site_names = seis_sites["station"].values

cities   = pd.read_csv(CSV_CITIES)
cit_e, cit_n = to_utm_km(cities["x"].values, cities["y"].values)
name_cit = cities["Name"].values

prof_1_e, prof_1_n = to_utm_km(PROFILE_1_LON, PROFILE_1_LAT)
prof2_e,   prof2_n   = to_utm_km(PROFILE_2_LON,  PROFILE_2_LAT)
arr_e,     arr_n     = to_utm_km([ARROW_LON], [ARROW_LAT])


# ==================================================================
# Basemap and feature drawing
# ==================================================================
def draw_basemap(ax):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    if SHOW_TOPO_BASEMAP:
        ax.imshow(CMAP_TOPO(topo_norm(topo_z)),
                  origin="lower", extent=topo_extent,
                  aspect="auto", interpolation="bilinear", zorder=1)
        ax.imshow(topo_hs, cmap="gray", origin="lower", extent=topo_extent,
                  alpha=0.45, aspect="auto", interpolation="bilinear", zorder=2)
    if _use_bath:
        bath_mask = np.where(bath_z <= 0, 1.0, np.nan)
        ax.imshow(bath_mask, origin="lower", extent=bath_extent,
                  cmap=mcolors.ListedColormap([OCEAN_COLOR]),
                  vmin=0, vmax=1, alpha=0.85, aspect="auto",
                  interpolation="none", zorder=3)

    ax.set_xlabel("Easting (km)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Northing (km)", fontsize=AXIS_LABEL_SIZE)
    if AXES_UNITS == "km" and AXES_KM_COMMA:
        _comma_fmt = mpl.ticker.StrMethodFormatter("{x:,.0f}")
        ax.xaxis.set_major_formatter(_comma_fmt)
        ax.yaxis.set_major_formatter(_comma_fmt)
    ax.tick_params(labelsize=AXIS_TICK_SIZE)


def draw_features(ax, eq_e, eq_n):
    if SHOW_PROFILE_LINES:
        ax.plot(prof_1_e, prof_1_n, clip_on=True, **PROFILE_1_STYLE)
        ax.plot(prof2_e,   prof2_n,   clip_on=True, **PROFILE_2_STYLE)

    if SHOW_VSLICE_LINES:
        for vi, vs in enumerate(VSLICES):
            ve_ends, vn_ends = _profile_utm_km(vs)
            lbl_start, lbl_end = _profile_labels(vi)
            ax.plot(ve_ends, vn_ends, clip_on=True, **VSLICE_MAP_LINE_STYLE)
            for xy, lbl in zip(zip(ve_ends, vn_ends), (lbl_start, lbl_end)):
                if _in_region(np.array([xy[0]]), np.array([xy[1]]))[0]:
                    ax.text(xy[0], xy[1], lbl,
                            fontsize=AXIS_TICK_SIZE, fontweight="bold",
                            color=VSLICE_MAP_LINE_STYLE["color"],
                            ha="center", va="bottom", clip_on=True, zorder=16)

    if SHOW_SEISMICITY:
        clipped_markers(ax, eq_e, eq_n, label="Seismicity", **EQ_MARKER_STYLE)
    if SHOW_MT_SITES:
        clipped_markers(ax, mt_e, mt_n, label="MT site", **MT_MARKER_STYLE)
        clipped_labels(ax, mt_e, mt_n, mt_names, MT_LABEL_STYLE)
    if SHOW_SEISMIC_SITES:
        clipped_markers(ax, seis_site_e, seis_site_n,
                        label="Seismic site", **SEISMIC_SITES_MARKER_STYLE)
    if SHOW_VOLCANOES:
        clipped_markers(ax, utmv_e, utmv_n, **VOLC_INACT_MARKER_STYLE)
        clipped_labels(ax, utmv_e, utmv_n, namev, VOLC_LABEL_STYLE)
    if SHOW_VOLCANOES_ACTIVE and volc_act_e:
        clipped_markers(ax, volc_act_e, volc_act_n,
                        label="Active volcano", **VOLC_ACT_MARKER_STYLE)
    if SHOW_CITIES:
        clipped_markers(ax, cit_e, cit_n, label="City", **CITY_MARKER_STYLE)
        clipped_labels(ax, cit_e, cit_n, name_cit, CITY_LABEL_STYLE)
    if SHOW_NORTH_ARROW:
        draw_north_arrow(ax, arr_e[0], arr_n[0], length_km=ARROW_LEN_KM)


# ==================================================================
# Main loop — depth slices
# ==================================================================
out_list = []

print("\n=== Depth slices ===")
ds = xr.open_dataset(_interp_path())
depth_axis = ds["depth"].values.astype(float)
east_axis  = ds["easting"].values.astype(float)
north_axis = ds["northing"].values.astype(float)

for ii, d_km in enumerate(DEPTH_SLICES_KM):
    # Nearest depth level
    idx = int(np.argmin(np.abs(depth_axis - d_km)))
    d_actual  = float(depth_axis[idx])
    tag       = f"{d_actual:.0f}km" if d_actual == int(d_actual) else f"{d_actual:.1f}km"
    depth_lbl = f"{d_actual:.0f} km" if d_actual == int(d_actual) else f"{d_actual:.1f} km"
    print(f"Depth slice at {depth_lbl} (requested {d_km} km, index {idx}) …")

    zmin_s = ZMIN_SEISM[ii]
    zmax_s = ZMAX_SEISM[ii]
    if zmin_s is None:
        zmin_s = -np.inf
    if zmax_s is None:
        zmax_s = np.inf
    mask_eqs = (zeqs > zmin_s) & (zeqs < zmax_s)
    eq_e = eq_e0[mask_eqs]
    eq_n = eq_n0[mask_eqs]

    for var in PLOT_VARS:
        print(f"  Plotting {var} …")
        cmap, cmin, cmax, cbar_label, alpha = _var_cfg(var, ds)

        # Extract 2-D slice (northing × easting)
        slice_2d = ds[var].isel(depth=idx).values.astype(float)  # (nnorth, neast)

        fig, ax, cax = create_map_figure()
        draw_basemap(ax)

        norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
        # pcolormesh with shading="nearest" renders each grid cell as a
        # solid-coloured rectangle centred on its coordinate — no
        # interpolation between cells, exactly reproducing the interpolated
        # model's own discrete grid without adding any smoothing on top.
        # (shading="flat" requires corner coordinates and silently drops the
        # last row/column; shading="nearest" keeps every cell and accepts
        # centre coordinates directly, which is what easting/northing are.)
        im = ax.pcolormesh(
            east_axis, north_axis, slice_2d,
            cmap=cmap, norm=norm, shading="nearest",
            alpha=1.0 - alpha, zorder=5,
        )
        if ISO_LINES_MAP:
            draw_iso_contours(ax, east_axis, north_axis, slice_2d,
                              ISO_LEVELS_MAP, var)

        draw_features(ax, eq_e, eq_n)
        ax.set_title(f"{var} at {depth_lbl}", fontsize=AXIS_TITLE_SIZE)
        finish_panel_colorbar(cax, im, cbar_label)
        if AXES_UNITS == "latlon":
            add_latlon_ticks(ax)
        draw_annotation(ax)
        stem = f"{var}_interp_{INTERP_METHOD}_{tag}_{SITE_PREFIX}"
        save_fig(fig, stem)
        _maybe_show()
        plt.close(fig)
        out_list.append(stem)

ds.close()

print("\nDepth-slice stems:")
for s in out_list:
    print(f"  {s}")


# ==================================================================
# Vertical slices
# ==================================================================
if VSLICES:
    print("\n=== Vertical slices ===")
    ds = xr.open_dataset(_interp_path())

    for vi, vslice in enumerate(VSLICES):
        name = vslice.get("name", "profile")
        lbl_start, lbl_end = _profile_labels(vi)
        print(f"  Section: {name} …")

        for var in PLOT_VARS:
            print(f"    {var} …")
            cmap, cmin, cmax, cbar_label, alpha = _var_cfg_vslice(var, ds)

            dist_km, depth_km, section, e_ends, n_ends, topo_prof, \
                utm_x, utm_xlabel = compute_vertical_slice(vslice, var, ds)

            stem = f"{var}_interp_{INTERP_METHOD}_section_{name}_{SITE_PREFIX}"
            plot_vertical_slice(
                dist_km, depth_km, section,
                e_ends, n_ends, topo_prof, utm_x, utm_xlabel,
                lbl_start, lbl_end, vslice,
                var, cmap, cmin, cmax, cbar_label, alpha,
                stem,
            )
            out_list.append(stem)

    ds.close()

    print("\nVertical slice stems:")
    for s in out_list:
        if "section" in s:
            print(f"  {s}")

print("\nAll done.")
