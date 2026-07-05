#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tacna_plot_seis.py
=================
Matplotlib replacement for tacna_plot_gmt.py.

Produces comparable depth-slice maps of Vp, Vs, and Vp/Vs ratio for the
Tacna seismic-tomography study area using only standard Python scientific
packages (NumPy, Matplotlib, xarray, pandas, pyproj, scipy).

Reads the same pre-computed UTM-km NetCDF grids and CSV feature files
produced by tacna_precompute.py; no GMT/PyGMT installation required.

Differences from the GMT version
---------------------------------
* Topographic shading is reproduced via matplotlib LightSource hillshade.
* The bathymetry ocean-fill is a masked imshow with a single blue-grey colour.
* North arrow is drawn with matplotlib's annotate mechanism.
* Markers, label offsets, fonts, and line styles are all user parameters
  (see MARKER STYLE SETTINGS section below).
* Features outside the map region are clipped via ax.set_clip_on + the
  Axes bounding box, so no markers or labels bleed outside the plot area.

Dependencies
------------
    numpy, matplotlib, xarray, pandas, pyproj, scipy

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic), 2026-06-29.
License: GNU General Public License v3 (GPL-3.0-or-later).
AI-generated code — review before use in production.
"""

import warnings
import matplotlib.patheffects as pe

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker
from matplotlib.colors import LightSource
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================================================
# USER SETTINGS
# =====================================================================

PLOT_WHAT    = ["vps"]           # any subset of ["vp", "vs", "vps"]
PLOT_FORMATS = [".pdf", ".jpg"]  # output formats
PLOT_DPI     = 600

# Figure sizes (cm).  Set HEIGHT to None to derive automatically.
# Horizontal maps: height derived from UTM aspect ratio if None.
HORZ_WIDTH_CM  = 10.0
HORZ_HEIGHT_CM = None   # None → derived from UTM aspect ratio

# Vertical sections: height derived from (depth_range * VE / profile_len)
# scaled to VSLICE_WIDTH_CM if None.
VSLICE_WIDTH_CM  = 14.0
VSLICE_HEIGHT_CM = None  # None → derived from VE and profile length

DEPTH_INDEX = [5, 9, 13]
ZMIN_SEISM  = [-7, 1,  9]
ZMAX_SEISM  = [ 1, 9, 30]

# Colour-scale limits
CMIN_VS,  CMAX_VS  = 2400, 4000
CMIN_VP,  CMAX_VP  = 4500, 6500
CMIN_VPS, CMAX_VPS = 1.7,  2.1

# Velocity overlay transparency (0 = opaque, 1 = invisible)
ALPHA_VELOCITY = 0.50

# Colourmap names (matplotlib built-ins; swap for any registered name)
CMAP_VS  = "viridis_r"   # stand-in for viridisr_vs.cpt
CMAP_VP  = "viridis_r"   # stand-in for viridisr_vp.cpt
CMAP_VPS = "hot_r"       # stand-in for hotr_vps.cpt

# Profile lines (lon/lat endpoint pairs)
PROFILE_CD_LON = [-70.476, -69.499213]
PROFILE_CD_LAT = [-18.255, -17.0481]
PROFILE_2_LON  = [-69.670, -70.034]
PROFILE_2_LAT  = [-17.695, -17.267]

# North-arrow anchor (lon, lat) and shaft length (km)
ARROW_LON    = -73.6
ARROW_LAT    = -18.1
ARROW_LEN_KM = 4.0

# Feature CSV paths
CSV_VOLCANES = "./features/volcanes.csv"
CSV_SEISMCAT = "./features/catalog_welllocated_15_simple5.csv"
CSV_MT_SITES = "./features/done/MTTacna_Sitelist.csv"
CSV_CITIES   = "./features/cities.csv"

# Volcano rows to label (indices into volcanes.csv)
VOLC_LABEL_IDX = [5, 12, 13]

# Pre-computed UTM-km NetCDF files (produced by tacna_precompute.py)
NC_TOPO       = "tacna_topo_utm.nc"
NC_BATH       = "tacna_bath_utm.nc"
NC_TOPO_SHADE = "tacna_topo_shade_utm.nc"  # presence noted; hillshade recomputed

# Region source: "topo" uses topo-grid extent; "data" uses velocity-subset extent
REGION_SOURCE    = "data"
REGION_MARGIN_KM = -0.001

# Hillshade parameters
HS_AZIMUTH  = 315   # Sun azimuth (degrees)
HS_ALTITUDE = 45    # Sun elevation (degrees)
HS_SIGMA    = 1.0   # Gaussian pre-smooth sigma (pixels); 0 = no smoothing

# Topo colour normalisation range (metres); matches grayC_dsc CPT limits
TOPO_VMIN = 1000
TOPO_VMAX = 6000

# Ocean fill colour (stand-in for oleron CPT)
OCEAN_COLOR = "#6baed6"

# =====================================================================
# LAT/LON SECONDARY AXES
# When True, a secondary top x-axis (longitude) and right y-axis (latitude)
# are added after the UTM-km plot.  The primary bottom/left axes keep their
# UTM km labels; lon/lat appear on top/right as a cosmetic overlay.
# LATLON_NTICKS : number of tick positions on each secondary axis
# LATLON_DECIMALS : decimal places on lon/lat tick labels
# =====================================================================
SHOW_LATLON_AXES = True
LATLON_NTICKS    = 5
LATLON_DECIMALS  = 2

# =====================================================================
# COLORBAR SETTINGS
# COLORBAR_POSITION : "right" | "left" | "bottom" | "top"
#   right / left  -> vertical bar, placed outside the map on that side
#   bottom / top  -> horizontal bar, placed outside the map on that side
# COLORBAR_SIZE    : fraction of the map edge occupied by the bar (0-1)
# COLORBAR_PAD     : gap between map and bar in inches
# COLORBAR_ASPECT  : length-to-thickness ratio of the bar
# COLORBAR_LABEL_* : font sizes for the bar label and tick annotations
# =====================================================================
COLORBAR_POSITION   = "right"   # "right" | "left" | "bottom" | "top"
COLORBAR_SIZE       = 0.05      # fraction of map edge length
COLORBAR_PAD        = 0.10      # inches between map axes and colorbar
COLORBAR_ASPECT     = 20        # bar length / bar thickness
COLORBAR_LABEL_SIZE = 8         # pt, label font size
COLORBAR_TICK_SIZE  = 7         # pt, tick annotation font size
COLORBAR_NTICKS     = 5         # approximate number of tick intervals

# =====================================================================
# MARKER & LABEL STYLE SETTINGS
# All sizes in points (s= for scatter uses pt²; markersize uses pt).
# Label offsets in km (map coordinates).
# =====================================================================

# --- Profile lines ---
PROFILE_CD_STYLE = dict(color="black", lw=0.4, zorder=10)
PROFILE_2_STYLE  = dict(color="gray",  lw=0.4, zorder=10)

# --- Seismicity ---
EQ_MARKER_STYLE = dict(
    s=4, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=11,
)

# --- MT sites ---
MT_MARKER_STYLE = dict(
    marker="^", s=14, facecolors="blue", edgecolors="black",
    linewidths=0.4, alpha=0.6, zorder=12,
)

# --- Inactive volcanoes ---
VOLC_INACT_MARKER_STYLE = dict(
    marker="^", s=16, facecolors="black", edgecolors="black",
    linewidths=0.2, zorder=13,
)
VOLC_LABEL_STYLE = dict(
    fontsize=6, fontweight="bold", color="white", zorder=14,
    offset_x=0.3, offset_y=0.3,   # km offsets from marker centre
)
VOLC_LABEL_STROKE = dict(linewidth=1.5, foreground="black")

# --- Active volcanoes ---
VOLC_ACT_MARKER_STYLE = dict(
    marker="^", s=16, facecolors="red", edgecolors="black",
    linewidths=0.2, zorder=13,
)

# --- Cities ---
CITY_MARKER_STYLE = dict(
    marker="s", s=18, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=13,
)
CITY_LABEL_STYLE = dict(
    fontsize=6, color="white", zorder=14,
    offset_x=0.3, offset_y=-0.3,  # km offsets from marker centre
)
CITY_LABEL_STROKE = dict(linewidth=1.5, foreground="black")

# --- North arrow ---
ARROW_STYLE = dict(color="dimgray", lw=2, mutation_scale=14)
ARROW_LABEL_STYLE = dict(fontsize=9, fontweight="bold", color="dimgray")

# =====================================================================
# VERTICAL SLICE SETTINGS
#
# VSLICES defines a list of arbitrary vertical cross-sections.
# Each entry is a dict with:
#
#   name    : str   — label used in title and output filename
#   p1      : [x, y] — first  endpoint, in UTM km OR lon/lat (see coord below)
#   p2      : [x, y] — second endpoint, same convention
#   coord   : "utm"     — p1/p2 are [easting_km, northing_km]
#             "latlon"  — p1/p2 are [lon_deg, lat_deg]
#   zmin_km : float — top  of section (km, positive down; 0 = surface)
#   zmax_km : float — base of section (km, positive down)
#   npts    : int   — number of sample points along the profile
#   nz      : int   — number of depth levels (interpolated)
#   swath_km: float — half-width (km) for projecting seismicity onto section
#
# Set VSLICES = [] to skip all vertical sections.
# =====================================================================
VSLICES = [
    dict(
        name    = "profile_CD",
        p1      = [-70.476, -18.255],   # lon, lat
        p2      = [-69.499, -17.048],
        coord   = "latlon",
        zmin_km = 0.0,
        zmax_km = 60.0,
        npts    = 200,
        nz      = 150,
        swath_km= 10.0,
    ),
    # Add further profiles here, e.g.:
    # dict(name="profile_2", p1=[...], p2=[...], coord="utm", ...),
]

# Colour-scale limits for vertical slices
# (defaults to the same as the horizontal maps; override here if needed)
VSLICE_CMIN_VPS = CMIN_VPS
VSLICE_CMAX_VPS = CMAX_VPS
VSLICE_CMIN_VP  = CMIN_VP
VSLICE_CMAX_VP  = CMAX_VP
VSLICE_CMIN_VS  = CMIN_VS
VSLICE_CMAX_VS  = CMAX_VS

# Vertical exaggeration (1 = true scale)
VSLICE_VE = 1.0

# Horizontal axis for vertical sections:
#   "utm"      — UTM easting or northing (km), xlim = profile endpoint coords
#   "distance" — cumulative distance from p1 (km), xlim = [0, profile_length]
VSLICE_X_AXIS = "utm"

# Seismicity marker style on cross-section (overrides EQ_MARKER_STYLE)
VSLICE_EQ_STYLE = dict(
    s=4, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=11,
)

# Topographic surface line style on cross-section
VSLICE_TOPO_STYLE = dict(color="dimgray", lw=0.5, zorder=12)

# Fill colours for the topography band above the section
VSLICE_TOPO_LAND_COLOR  = "gray"    # z > 0 (above sea level)
VSLICE_TOPO_OCEAN_COLOR = "#6baed6" # z <= 0 (below sea level)

# Extra headroom above the highest topographic point (km).
# The panel top is set to (max_elevation_km + VSLICE_TOPO_HEADROOM_KM)
# above the section zmin, so the topo fill is not flush with the axes edge.
VSLICE_TOPO_HEADROOM_KM = 1.0

# Style of the profile line drawn on the map figures
VSLICE_MAP_LINE_STYLE = dict(color="magenta", lw=0.8, ls="--", zorder=15)

# =====================================================================
# END USER SETTINGS
# =====================================================================


# ------------------------------------------------------------------
# Coordinate helper
# ------------------------------------------------------------------
_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
_to_geo = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)


def to_utm_km(lon, lat):
    """Convert geographic lon/lat to UTM Zone 19S easting/northing in km."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    e, n = _to_utm.transform(lon, lat)
    return e / 1e3, n / 1e3


# ------------------------------------------------------------------
# Hillshade
# ------------------------------------------------------------------
def compute_hillshade(z2d, dx_km, dy_km, azimuth=315, altitude=45, sigma=1.0):
    """Return a [0, 1] hillshade array for a 2-D elevation grid (metres)."""
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    if sigma > 0:
        z2d = gaussian_filter(z2d, sigma=sigma)
    return ls.hillshade(z2d, dx=dx_km * 1e3, dy=dy_km * 1e3, vert_exag=1.0)


# ------------------------------------------------------------------
# Save helper
# ------------------------------------------------------------------
def save_fig(fig, stem):
    for fmt in PLOT_FORMATS:
        out = stem + fmt
        fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {out}")


# ------------------------------------------------------------------
# Clip-aware scatter / text helpers
# ------------------------------------------------------------------
def _in_region(xe, yn):
    """Boolean mask: True where (xe, yn) fall inside the map region."""
    return (
        (xe >= xmin) & (xe <= xmax) &
        (yn >= ymin) & (yn <= ymax)
    )


def clipped_scatter(ax, xe, yn, **kwargs):
    """scatter() restricted to points inside the map region."""
    mask = _in_region(np.asarray(xe), np.asarray(yn))
    if not np.any(mask):
        return
    xe, yn = np.asarray(xe)[mask], np.asarray(yn)[mask]
    # Pop label so it still goes to legend via the first call
    ax.scatter(xe, yn, **kwargs)


def clipped_labels(ax, xe, yn, labels, style_dict):
    """
    Draw text labels for points inside the map region.

    style_dict must include 'offset_x' and 'offset_y' (km); remaining keys
    are passed to ax.text().  An optional 'stroke' key (dict) activates a
    withStroke path-effect.
    """
    ox = style_dict.pop("offset_x", 0.0)
    oy = style_dict.pop("offset_y", 0.0)
    stroke = style_dict.pop("stroke", None)
    path_effects = [pe.withStroke(**stroke)] if stroke else []

    xe = np.asarray(xe, dtype=float)
    yn = np.asarray(yn, dtype=float)
    mask = _in_region(xe, yn)

    for x, y, lbl, inside in zip(xe, yn, labels, mask):
        if not inside:
            continue
        ax.text(
            x + ox, y + oy, lbl,
            path_effects=path_effects if path_effects else None,
            **style_dict,
        )


# ------------------------------------------------------------------
# North arrow
# ------------------------------------------------------------------
def draw_north_arrow(ax, x_km, y_km, length_km=4.0):
    """Draw a north arrow at UTM position (x_km, y_km) if inside region."""
    if not _in_region(np.array([x_km]), np.array([y_km]))[0]:
        return
    ax.annotate(
        "", xy=(x_km, y_km + length_km), xytext=(x_km, y_km),
        arrowprops=dict(arrowstyle="-|>", **ARROW_STYLE),
        annotation_clip=True,
    )
    ax.text(
        x_km, y_km + length_km + 0.8, "N",
        ha="center", va="bottom", **ARROW_LABEL_STYLE,
        clip_on=True,
    )


# ------------------------------------------------------------------
# Colourbar  (position controlled by COLORBAR_* user settings)
# ------------------------------------------------------------------
def add_colorbar(fig, ax, mappable, label):
    """
    Attach a colourbar to *ax* according to COLORBAR_POSITION.

    "right" / "left"   -> vertical bar
    "bottom" / "top"   -> horizontal bar

    Uses fig.colorbar with the axes-stealing (fraction/pad) API so that
    the map axes shrinks by the right amount and tight_layout still works.
    """
    pos = COLORBAR_POSITION.lower()
    if pos in ("right", "left"):
        orientation = "vertical"
        # For left placement we still use fraction/pad but flip the location
        location = pos
    elif pos in ("bottom", "top"):
        orientation = "horizontal"
        location = pos
    else:
        raise ValueError(
            f"COLORBAR_POSITION={COLORBAR_POSITION!r} is not valid. "
            "Choose 'right', 'left', 'bottom', or 'top'."
        )

    cbar = fig.colorbar(
        mappable, ax=ax,
        orientation=orientation,
        location=location,        # matplotlib >= 3.7 respects this with ax=
        fraction=COLORBAR_SIZE,
        pad=COLORBAR_PAD / fig.get_size_inches()[0],   # convert inches -> fig fraction
        aspect=COLORBAR_ASPECT,
        shrink=0.85,
    )
    cbar.set_label(label, fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    cbar.locator = mpl.ticker.MaxNLocator(nbins=COLORBAR_NTICKS)
    cbar.update_ticks()
    return cbar


# ------------------------------------------------------------------
# Secondary lon/lat axes  (cosmetic overlay on UTM-km plot)
# ------------------------------------------------------------------
def add_latlon_ticks(ax):
    """
    Replace the UTM-km tick labels on the primary axes with lon/lat values.
    No extra axes are created — the existing bottom x-axis and left y-axis
    ticks are reformatted in-place using FuncFormatter.

    Controlled by SHOW_LATLON_AXES, LATLON_NTICKS, LATLON_DECIMALS.
    """
    e_mid_m = (xmin + xmax) / 2.0 * 1e3
    n_mid_m = (ymin + ymax) / 2.0 * 1e3
    fmt = f"{{:.{LATLON_DECIMALS}f}}°"

    # Place ticks at chosen UTM km positions
    e_ticks_km = np.linspace(xmin, xmax, LATLON_NTICKS)
    n_ticks_km = np.linspace(ymin, ymax, LATLON_NTICKS)

    lon_labels = [fmt.format(_to_geo.transform(e * 1e3, n_mid_m)[0])
                  for e in e_ticks_km]
    lat_labels = [fmt.format(_to_geo.transform(e_mid_m, n * 1e3)[1])
                  for n in n_ticks_km]

    ax.set_xticks(e_ticks_km)
    ax.set_xticklabels(lon_labels, fontsize=COLORBAR_TICK_SIZE)
    ax.set_xlabel("Longitude", fontsize=COLORBAR_LABEL_SIZE)

    ax.set_yticks(n_ticks_km)
    ax.set_yticklabels(lat_labels, fontsize=COLORBAR_TICK_SIZE)
    ax.set_ylabel("Latitude", fontsize=COLORBAR_LABEL_SIZE)


# ==================================================================
# Vertical slice engine
# ==================================================================

def _profile_utm_km(vslice):
    """
    Return (e_km, n_km) endpoint arrays for a VSLICES entry,
    normalising coord="latlon" → UTM km.
    """
    p1, p2 = np.asarray(vslice["p1"], float), np.asarray(vslice["p2"], float)
    if vslice.get("coord", "latlon").lower() == "latlon":
        e1, n1 = to_utm_km([p1[0]], [p1[1]])
        e2, n2 = to_utm_km([p2[0]], [p2[1]])
        return np.array([e1[0], e2[0]]), np.array([n1[0], n2[0]])
    else:
        return np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]])


def _profile_labels(index):
    """
    Return (start_label, end_label) for the profile at position *index*
    in VSLICES.  Index 0 → ('A', 'A\''), 1 → ('B', 'B\''), etc.
    """
    letter = chr(ord('A') + index)
    return letter, letter + "'"


def _sample_profile_points(e_ends, n_ends, npts):
    """
    Return (dist_km, e_pts, n_pts, utm_x, utm_xlabel) for npts evenly spaced
    points along the profile.

    utm_x      : 1-D array — easting when |Δe| >= |Δn|, northing otherwise.
    utm_xlabel : matching axis label.
    dist_km    : cumulative distance from p1 (km).
    """
    e_pts = np.linspace(e_ends[0], e_ends[1], npts)
    n_pts = np.linspace(n_ends[0], n_ends[1], npts)
    dist_km = np.sqrt((e_pts - e_ends[0])**2 + (n_pts - n_ends[0])**2)

    de = abs(e_ends[1] - e_ends[0])
    dn = abs(n_ends[1] - n_ends[0])
    if de >= dn:
        utm_x      = e_pts
        utm_xlabel = "Easting (km)"
    else:
        utm_x      = n_pts
        utm_xlabel = "Northing (km)"

    return dist_km, e_pts, n_pts, utm_x, utm_xlabel


def compute_vertical_slice_seis(vslice, var):
    """
    Sample a seismic variable along a vertical profile.

    Reads tacna_pvelocity_subset.nc which has dims (depth, lat, lon).
    Interpolates bilinearly in (lat, lon) at each depth, then resamples
    to a regular depth grid.

    Parameters
    ----------
    vslice : dict  — one entry from VSLICES
    var    : str   — "vp", "vs", or "vps" (variable name in the dataset)

    Returns
    -------
    dist_km : 1-D array, distance along profile (km)
    depth_km: 1-D array, depth axis (km, positive down)
    section : 2-D array (nz, npts), interpolated values
    e_ends  : 1-D array [e0, e1] in UTM km (for map overlay)
    n_ends  : 1-D array [n0, n1] in UTM km
    topo_prof: 1-D array, surface elevation (m) along profile (or None)
    """
    e_ends, n_ends = _profile_utm_km(vslice)
    npts  = vslice.get("npts", 200)
    nz    = vslice.get("nz",   150)
    zmin  = vslice.get("zmin_km", 0.0)
    zmax  = vslice.get("zmax_km", 60.0)

    dist_km, e_pts, n_pts, utm_x, utm_xlabel = \
        _sample_profile_points(e_ends, n_ends, npts)

    # Convert profile points back to lat/lon for geographic-grid interpolation
    lon_pts, lat_pts = _to_geo.transform(e_pts * 1e3, n_pts * 1e3)

    # Load 3-D velocity subset
    nc_var = {"vp": "data", "vs": "data", "vps": "data"}[var]
    fname  = {"vp": "tacna_vp.nc",
              "vs": "tacna_vs.nc",
              "vps": "tacna_vps.nc"}[var]
    _ds = xr.open_dataset(fname)
    lats_grid = _ds["lat"].values
    lons_grid = _ds["lon"].values
    deps_grid = _ds["depth"].values   # km
    vals_3d   = _ds["data"].values    # (ndepth, nlat, nlon)
    _ds.close()

    # Ensure ascending lat
    if lats_grid[0] > lats_grid[-1]:
        lats_grid = lats_grid[::-1]
        vals_3d   = vals_3d[:, ::-1, :]

    # Interpolator over (lat, lon) — one per depth level is too slow;
    # build one 3-D interpolator instead
    interp = RegularGridInterpolator(
        (deps_grid, lats_grid, lons_grid), vals_3d,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    depth_km = np.linspace(zmin, zmax, nz)
    D, L, O  = np.meshgrid(depth_km, lat_pts, lon_pts, indexing="ij")
    # We want section[iz, ix] = value at depth_km[iz], profile point ix
    # Build query points: iterate depth outer, profile inner
    d_q = np.repeat(depth_km, npts)
    l_q = np.tile(lat_pts, nz)
    o_q = np.tile(lon_pts, nz)
    pts = np.column_stack([d_q, l_q, o_q])
    section = interp(pts).reshape(nz, npts)

    # Topo along profile (interpolate topo grid)
    topo_prof = None
    if topo_z is not None:
        topo_interp = RegularGridInterpolator(
            (topo_y, topo_x), topo_z,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        topo_prof = topo_interp(np.column_stack([n_pts, e_pts]))

    return dist_km, depth_km, section, e_ends, n_ends, topo_prof, utm_x, utm_xlabel


def _project_seismicity_to_profile(e_ends, n_ends, swath_km, zmin_km, zmax_km):
    """
    Return (along_km, depth_km) for catalogue events within swath_km of
    the profile line and within the section depth range [zmin_km, zmax_km].

    Catalogue z convention: positive downward (km).  Events above the surface
    (negative z) are included when zmin_km < 0.
    """
    de = e_ends[1] - e_ends[0]
    dn = n_ends[1] - n_ends[0]
    L  = np.sqrt(de**2 + dn**2)
    if L == 0:
        return np.array([]), np.array([])
    ue, un = de / L, dn / L

    ve = eq_e0 - e_ends[0]
    vn = eq_n0 - n_ends[0]

    along  = ve * ue + vn * un
    across = np.abs(ve * (-un) + vn * ue)

    mask = (
        (across <= swath_km) &
        (along  >= 0) &
        (along  <= L) &
        (zeqs   >= zmin_km) &
        (zeqs   <= zmax_km)
    )
    return along[mask], zeqs[mask]


def plot_vertical_slice(dist_km, depth_km, section, e_ends, n_ends,
                        topo_prof, utm_x, utm_xlabel,
                        lbl_start, lbl_end,
                        vslice, var, cmap, cmin, cmax, cbar_label,
                        stem):
    """
    Produce and save a vertical cross-section figure.

    x-axis : UTM easting/northing (km) when VSLICE_X_AXIS="utm",
             or cumulative distance from p1 when VSLICE_X_AXIS="distance".
             xlim defaults to the two profile endpoint coordinates.
    y-axis : depth (km, positive down, increasing downward).
    Endpoint labels (lbl_start, lbl_end) are annotated on the x-axis.
    """
    swath  = vslice.get("swath_km", 10.0)
    zmin_s = vslice.get("zmin_km", depth_km[0])
    zmax_s = vslice.get("zmax_km", depth_km[-1])
    ve     = VSLICE_VE
    name   = vslice.get("name", "profile")

    # Choose horizontal coordinate
    if VSLICE_X_AXIS == "distance":
        x_arr    = dist_km
        x_label  = "Distance along profile (km)"
    else:
        x_arr    = utm_x
        x_label  = utm_xlabel

    # --- project seismicity ---
    eq_dist, eq_dep = _project_seismicity_to_profile(
        e_ends, n_ends, swath, zmin_s, zmax_s)
    if len(eq_dist):
        eq_x = np.interp(eq_dist, dist_km, x_arr)
    else:
        eq_x = eq_dist

    # --- figure dimensions ---
    profile_len = dist_km[-1]
    depth_range = depth_km[-1] - depth_km[0]
    w_in = VSLICE_WIDTH_CM / 2.54
    h_in = w_in * (depth_range * ve) / profile_len \
           if VSLICE_HEIGHT_CM is None else VSLICE_HEIGHT_CM / 2.54
    print(f"  Section figure size: {w_in:.2f} × {h_in:.2f} in")

    fig, ax = plt.subplots(figsize=(w_in, h_in))

    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    im = ax.pcolormesh(
        x_arr, depth_km, section,
        cmap=cmap, norm=norm, shading="auto",
        alpha=1.0 - ALPHA_VELOCITY, zorder=5,
    )

    # Topo fill
    y_top = depth_km[0]
    if topo_prof is not None:
        surf_depth = depth_km[0] - topo_prof / 1e3
        y_top = surf_depth.min() - VSLICE_TOPO_HEADROOM_KM

        land  = topo_prof >  0
        ocean = topo_prof <= 0

        if land.any():
            ax.fill_between(x_arr, depth_km[0], surf_depth,
                            where=land,
                            color=VSLICE_TOPO_LAND_COLOR, alpha=0.5,
                            zorder=6, interpolate=True)
        if ocean.any():
            ax.fill_between(x_arr, depth_km[0], surf_depth,
                            where=ocean,
                            color=VSLICE_TOPO_OCEAN_COLOR, alpha=0.5,
                            zorder=6, interpolate=True)
        ax.plot(x_arr, surf_depth, **VSLICE_TOPO_STYLE)

    # Seismicity
    if len(eq_x):
        ax.scatter(eq_x, eq_dep, **VSLICE_EQ_STYLE)

    # Axes
    x0, x1 = x_arr[0], x_arr[-1]
    ax.set_xlim(min(x0, x1), max(x0, x1))
    ax.set_ylim(y_top, depth_km[-1])
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel("Depth (km)", fontsize=8)
    ax.tick_params(labelsize=7)

    # Endpoint labels at the top of the section (A / A', B / B', …)
    # y_top is the topmost data coordinate (smallest depth, possibly negative)
    for xpos, lbl in ((x0, lbl_start), (x1, lbl_end)):
        ax.text(xpos, y_top, lbl,
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="black",
                clip_on=False, zorder=20)

    title_var = {"vp": "Vp", "vs": "Vs", "vps": "Vp/Vs"}[var]
    ax.set_title(f"{title_var} — {name}  (swath ±{swath} km)", fontsize=9)

    if ve != 1.0:
        ax.text(0.99, 0.01, f"VE = {ve:.1f}×",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="gray")

    add_colorbar(fig, ax, im, cbar_label)
    fig.tight_layout()
    save_fig(fig, stem)
    plt.close(fig)


# ==================================================================
# Load static grids
# ==================================================================
print("Loading topo grid …")
_topo_da = xr.open_dataarray(NC_TOPO)
topo_x = _topo_da["x"].values   # 1-D easting  in km
topo_y = _topo_da["y"].values   # 1-D northing in km
topo_z = _topo_da.values        # shape (ny, nx), metres
_topo_da.close()

dx_km = float(np.median(np.diff(topo_x)))
dy_km = float(np.median(np.diff(topo_y)))

print("Computing hillshade …")
topo_hs = compute_hillshade(topo_z, dx_km, dy_km,
                             HS_AZIMUTH, HS_ALTITUDE, HS_SIGMA)

print("Loading bathymetry grid …")
_bath_da = xr.open_dataarray(NC_BATH)
bath_x = _bath_da["x"].values
bath_y = _bath_da["y"].values
bath_z = _bath_da.values
_bath_da.close()

topo_extent = [topo_x.min(), topo_x.max(), topo_y.min(), topo_y.max()]
bath_extent = [bath_x.min(), bath_x.max(), bath_y.min(), bath_y.max()]
topo_norm   = mcolors.Normalize(vmin=TOPO_VMIN, vmax=TOPO_VMAX)
CMAP_TOPO   = plt.get_cmap("gray")


# ==================================================================
# Map region
# ==================================================================
if REGION_SOURCE == "data":
    _ds = xr.open_dataset("tacna_vp.nc")
    _e  = _ds["utm_easting"].values  / 1e3
    _n  = _ds["utm_northing"].values / 1e3
    _ds.close()
    utm_region = [
        float(_e.min()) - REGION_MARGIN_KM,
        float(_e.max()) + REGION_MARGIN_KM,
        float(_n.min()) - REGION_MARGIN_KM,
        float(_n.max()) + REGION_MARGIN_KM,
    ]
    print(f"Region: velocity subset + {REGION_MARGIN_KM} km margin")
else:
    utm_region = [topo_x.min(), topo_x.max(), topo_y.min(), topo_y.max()]
    print("Region: topo grid extent")

xmin, xmax, ymin, ymax = utm_region
print(f"UTM region (km): {utm_region}")

fig_w = HORZ_WIDTH_CM / 2.54
fig_h = fig_w * (ymax - ymin) / (xmax - xmin) \
        if HORZ_HEIGHT_CM is None else HORZ_HEIGHT_CM / 2.54
print(f"Figure size (map): {fig_w:.2f} × {fig_h:.2f} in")


# ==================================================================
# Feature layers
# ==================================================================
volcanes   = pd.read_csv(CSV_VOLCANES)
utmv_e, utmv_n = to_utm_km(
    volcanes["LONG"][VOLC_LABEL_IDX].values,
    volcanes["LAT"][VOLC_LABEL_IDX].values,
)
namev = volcanes["VOLCAN2"][VOLC_LABEL_IDX].values

volc_act_e, volc_act_n = [], []
for i in range(len(volcanes)):
    if "ACT" in str(volcanes["ESTADO"][i]):
        ae, an = to_utm_km([volcanes["LONG"][i]], [volcanes["LAT"][i]])
        volc_act_e.append(ae[0])
        volc_act_n.append(an[0])

eqs    = pd.read_csv(CSV_SEISMCAT, delimiter=" ")
eq_e0, eq_n0 = to_utm_km(eqs["x"].values, eqs["y"].values)
zeqs   = eqs["z"].values

tacna  = pd.read_csv(CSV_MT_SITES, delimiter=" ")
mt_e, mt_n = to_utm_km(tacna["x"].values, tacna["y"].values)

cities    = pd.read_csv(CSV_CITIES)
cit_e, cit_n = to_utm_km(cities["x"].values, cities["y"].values)
name_cit  = cities["Name"].values

prof_cd_e, prof_cd_n = to_utm_km(PROFILE_CD_LON, PROFILE_CD_LAT)
prof2_e,   prof2_n   = to_utm_km(PROFILE_2_LON,  PROFILE_2_LAT)
arr_e,     arr_n     = to_utm_km([ARROW_LON], [ARROW_LAT])


# ==================================================================
# Depth coordinate
# ==================================================================
depth_coord = xr.open_dataset("tacna_vp.nc")["depth"]


# ==================================================================
# Basemap and feature drawing
# ==================================================================
def draw_basemap(ax):
    """Topo greyscale + hillshade + ocean fill; enforce map limits."""
    # Set limits and aspect BEFORE imshow calls.
    # adjustable="box" is the only mode compatible with shared axes created
    # later by add_latlon_ticks (twiny/twinx).  Passing aspect="auto" to
    # each imshow lets the axes-level aspect control geometry instead.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    ax.imshow(
        CMAP_TOPO(topo_norm(topo_z)),
        origin="lower", extent=topo_extent,
        aspect="auto", interpolation="bilinear", zorder=1,
    )
    ax.imshow(
        topo_hs, cmap="gray", origin="lower", extent=topo_extent,
        alpha=0.45, aspect="auto", interpolation="bilinear", zorder=2,
    )
    bath_mask = np.where(bath_z <= 0, 1.0, np.nan)
    ax.imshow(
        bath_mask, origin="lower", extent=bath_extent,
        cmap=mcolors.ListedColormap([OCEAN_COLOR]),
        vmin=0, vmax=1, alpha=0.85, aspect="auto",
        interpolation="none", zorder=3,
    )
    ax.set_xlabel("Easting (km)", fontsize=8)
    ax.set_ylabel("Northing (km)", fontsize=8)
    ax.tick_params(labelsize=7)


def draw_features(ax, eq_e, eq_n):
    """Overlay all feature layers; all markers/labels clipped to map region."""

    # Profile lines — clip at axes boundary automatically via clip_on
    ax.plot(prof_cd_e, prof_cd_n, clip_on=True, **PROFILE_CD_STYLE)
    ax.plot(prof2_e,   prof2_n,   clip_on=True, **PROFILE_2_STYLE)

    # Vertical slice profile lines drawn on the map
    for vi, vs in enumerate(VSLICES):
        ve_ends, vn_ends = _profile_utm_km(vs)
        lbl_start, lbl_end = _profile_labels(vi)
        ax.plot(ve_ends, vn_ends, clip_on=True,
                label=vs.get("name", "slice"), **VSLICE_MAP_LINE_STYLE)
        for xy, lbl in zip(zip(ve_ends, vn_ends), (lbl_start, lbl_end)):
            if _in_region(np.array([xy[0]]), np.array([xy[1]]))[0]:
                ax.text(xy[0], xy[1], lbl, fontsize=7, fontweight="bold",
                        color=VSLICE_MAP_LINE_STYLE["color"],
                        ha="center", va="bottom", clip_on=True, zorder=16)

    # Seismicity
    clipped_scatter(ax, eq_e, eq_n, label="Seismicity", **EQ_MARKER_STYLE)

    # MT sites
    clipped_scatter(ax, mt_e, mt_n, label="MT site", **MT_MARKER_STYLE)

    # Inactive volcanoes
    clipped_scatter(ax, utmv_e, utmv_n, **VOLC_INACT_MARKER_STYLE)
    clipped_labels(
        ax, utmv_e, utmv_n, namev,
        {**VOLC_LABEL_STYLE, "stroke": VOLC_LABEL_STROKE},
    )

    # Active volcanoes
    if volc_act_e:
        clipped_scatter(ax, volc_act_e, volc_act_n,
                        label="Active volcano", **VOLC_ACT_MARKER_STYLE)

    # Cities
    clipped_scatter(ax, cit_e, cit_n, label="City", **CITY_MARKER_STYLE)
    clipped_labels(
        ax, cit_e, cit_n, name_cit,
        {**CITY_LABEL_STYLE, "stroke": CITY_LABEL_STROKE},
    )

    # North arrow
    draw_north_arrow(ax, arr_e[0], arr_n[0], length_km=ARROW_LEN_KM)


# ==================================================================
# Main loop
# ==================================================================
out_list = []

for ii, d_index in enumerate(DEPTH_INDEX):

    depth_km    = int(depth_coord.item(d_index))
    depth_label = f"{depth_km} km"
    tag         = f"{depth_km}km"

    mask_eqs = (zeqs > ZMIN_SEISM[ii]) & (zeqs < ZMAX_SEISM[ii])
    eq_e = eq_e0[mask_eqs]
    eq_n = eq_n0[mask_eqs]

    vps_nc = f"tacna_vps_utm_{tag}.nc"
    vp_nc  = f"tacna_vp_utm_{tag}.nc"
    vs_nc  = f"tacna_vs_utm_{tag}.nc"

    # ------------------------------------------------------------------
    # Vp/Vs
    # ------------------------------------------------------------------
    if "vps" in PLOT_WHAT:
        print(f"Plotting Vp/Vs at {depth_label} …")
        _da  = xr.open_dataarray(vps_nc)
        vx, vy = _da["x"].values, _da["y"].values
        vz = np.clip(_da.values.copy().astype(float), CMIN_VPS, CMAX_VPS)
        _da.close()
        # Mask clipped edges (replicates grdclip nan_transparent behaviour)
        vz[(vz <= CMIN_VPS) | (vz >= CMAX_VPS)] = np.nan

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        draw_basemap(ax)

        norm = mcolors.Normalize(vmin=CMIN_VPS, vmax=CMAX_VPS)
        im = ax.imshow(
            vz, cmap=CMAP_VPS, norm=norm, origin="lower",
            extent=[vx.min(), vx.max(), vy.min(), vy.max()],
            alpha=1.0 - ALPHA_VELOCITY,
            aspect="equal", interpolation="bilinear", zorder=5,
        )
        draw_features(ax, eq_e, eq_n)
        ax.set_title(f"Vp/Vs at {depth_label}", fontsize=9)
        add_colorbar(fig, ax, im, "Vp/Vs")
        if SHOW_LATLON_AXES:
            add_latlon_ticks(ax)
        fig.tight_layout()
        stem = f"vps_tomo_{tag}_tacna"
        save_fig(fig, stem)
        plt.close(fig)
        out_list.append(stem)

    # ------------------------------------------------------------------
    # Vp
    # ------------------------------------------------------------------
    if "vp" in PLOT_WHAT:
        print(f"Plotting Vp at {depth_label} …")
        _da  = xr.open_dataarray(vp_nc)
        vx, vy = _da["x"].values, _da["y"].values
        vz = np.clip(_da.values.copy().astype(float), CMIN_VP, CMAX_VP)
        _da.close()

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        draw_basemap(ax)

        norm = mcolors.Normalize(vmin=CMIN_VP, vmax=CMAX_VP)
        im = ax.imshow(
            vz, cmap=CMAP_VP, norm=norm, origin="lower",
            extent=[vx.min(), vx.max(), vy.min(), vy.max()],
            alpha=1.0 - ALPHA_VELOCITY,
            aspect="equal", interpolation="bilinear", zorder=5,
        )
        draw_features(ax, eq_e, eq_n)
        ax.set_title(f"Vp at {depth_label}", fontsize=9)
        add_colorbar(fig, ax, im, "Vp (m/s)")
        if SHOW_LATLON_AXES:
            add_latlon_ticks(ax)
        fig.tight_layout()
        stem = f"vp_tomo_{tag}_tacna"
        save_fig(fig, stem)
        plt.close(fig)
        out_list.append(stem)

    # ------------------------------------------------------------------
    # Vs
    # ------------------------------------------------------------------
    if "vs" in PLOT_WHAT:
        print(f"Plotting Vs at {depth_label} …")
        _da  = xr.open_dataarray(vs_nc)
        vx, vy = _da["x"].values, _da["y"].values
        vz = np.clip(_da.values.copy().astype(float), CMIN_VS, CMAX_VS)
        _da.close()

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        draw_basemap(ax)

        norm = mcolors.Normalize(vmin=CMIN_VS, vmax=CMAX_VS)
        im = ax.imshow(
            vz, cmap=CMAP_VS, norm=norm, origin="lower",
            extent=[vx.min(), vx.max(), vy.min(), vy.max()],
            alpha=1.0 - ALPHA_VELOCITY,
            aspect="equal", interpolation="bilinear", zorder=5,
        )
        draw_features(ax, eq_e, eq_n)
        ax.set_title(f"Vs at {depth_label}", fontsize=9)
        add_colorbar(fig, ax, im, "Vs (m/s)")
        if SHOW_LATLON_AXES:
            add_latlon_ticks(ax)
        fig.tight_layout()
        stem = f"vs_tomo_{tag}_tacna"
        save_fig(fig, stem)
        plt.close(fig)
        out_list.append(stem)

print("\nDone. Output stems:")
for s in out_list:
    print(f"  {s}")


# ==================================================================
# Vertical slices
# ==================================================================
if VSLICES:
    print("\n=== Vertical slices ===")

    var_cfg = []
    if "vps" in PLOT_WHAT:
        var_cfg.append(("vps", CMAP_VPS, VSLICE_CMIN_VPS, VSLICE_CMAX_VPS, "Vp/Vs"))
    if "vp"  in PLOT_WHAT:
        var_cfg.append(("vp",  CMAP_VP,  VSLICE_CMIN_VP,  VSLICE_CMAX_VP,  "Vp (m/s)"))
    if "vs"  in PLOT_WHAT:
        var_cfg.append(("vs",  CMAP_VS,  VSLICE_CMIN_VS,  VSLICE_CMAX_VS,  "Vs (m/s)"))

    for vi, vslice in enumerate(VSLICES):
        name = vslice.get("name", "profile")
        print(f"  Computing section: {name} …")
        lbl_start, lbl_end = _profile_labels(vi)
        for var, cmap, cmin, cmax, cbar_label in var_cfg:
            dist_km, depth_km, section, e_ends, n_ends, topo_prof, \
                utm_x, utm_xlabel = compute_vertical_slice_seis(vslice, var)
            stem = f"{var}_section_{name}_tacna"
            plot_vertical_slice(dist_km, depth_km, section, e_ends, n_ends,
                                topo_prof, utm_x, utm_xlabel,
                                lbl_start, lbl_end,
                                vslice, var, cmap, cmin, cmax,
                                cbar_label, stem)
            out_list.append(stem)

    print("\nVertical slice stems:")
    for s in out_list:
        if "section" in s:
            print(f"  {s}")
