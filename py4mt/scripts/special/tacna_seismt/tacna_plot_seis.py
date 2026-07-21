#! /usr/bin/env python3
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
import os
import matplotlib.patheffects as pe

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker
from matplotlib.colors import LightSource
from matplotlib.path import Path as MplPath
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------
# Colormap import helper
# ---------------------------------------------------------------------
def load_colormap(spec, name=None):
    """
    Resolve a colourmap spec into a matplotlib Colormap.

    Accepts, in order of precedence:
      - an existing Colormap instance — returned unchanged
      - a path to a GMT ``.cpt`` file — parsed directly, preserving the
        file's own (possibly non-uniform) colour-stop spacing. This lets
        you use the *actual* original palette (e.g. viridisr_vp.cpt) for
        an exact visual comparison against GMT-produced figures, instead
        of a same-ish matplotlib named stand-in.
      - a path to a plain text/CSV file of RGB(A) rows (0-255 or 0-1,
        whitespace- or comma-separated, one colour per line) — built into
        an evenly-spaced ListedColormap. Useful for reusing an exact
        palette exported from another tool (e.g. ParaView, Generic
        Mapping Tools' makecpt, a colleague's colour list) so two
        different figures use pixel-identical colours for comparison.
      - any matplotlib-registered colormap name (built-in, or registered
        by a third-party package such as cmcrameri/cmocean if that
        package has been imported elsewhere in the process) — resolved
        via plt.get_cmap, unchanged from the original behaviour.

    Parameters
    ----------
    spec : str or matplotlib.colors.Colormap
    name : str, optional — name to register the resulting colormap under
           (defaults to the file's base name, or the spec string itself)

    Returns
    -------
    matplotlib.colors.Colormap
    """
    if isinstance(spec, mcolors.Colormap):
        return spec

    spec = str(spec)
    ext = os.path.splitext(spec)[1].lower()
    cmap_name = name or os.path.splitext(os.path.basename(spec))[0]

    if ext == ".cpt":
        return _load_cpt_colormap(spec, cmap_name)
    if ext in (".txt", ".csv", ".dat") and os.path.exists(spec):
        return _load_rgb_list_colormap(spec, cmap_name)

    # Not a recognised file — treat as a matplotlib-registered name
    # (built-in, or from a third-party package already imported).
    return plt.get_cmap(spec)


def _parse_cpt_color(tokens):
    """Parse a single .cpt colour field: 'R G B', 'R/G/B', '#hex', or grey."""
    if len(tokens) >= 3:
        r, g, b = (float(t) for t in tokens[:3])
        return (r / 255, g / 255, b / 255)
    tok = tokens[0]
    if tok.startswith("#"):
        return mcolors.to_rgb(tok)
    if "/" in tok:
        r, g, b = (float(t) for t in tok.split("/"))
        return (r / 255, g / 255, b / 255)
    v = float(tok)
    return (v / 255, v / 255, v / 255)


def _load_cpt_colormap(path, name):
    """Parse a GMT .cpt colour-palette file into a LinearSegmentedColormap,
    preserving its own colour-stop spacing (not assumed to be uniform)."""
    stops = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line[0] in "BFNbfn":
                continue
            parts = line.split()
            try:
                if len(parts) >= 8:
                    z0 = float(parts[0]); c0 = _parse_cpt_color(parts[1:4])
                    z1 = float(parts[4]); c1 = _parse_cpt_color(parts[5:8])
                elif len(parts) == 4:
                    z0 = float(parts[0]); c0 = _parse_cpt_color([parts[1]])
                    z1 = float(parts[2]); c1 = _parse_cpt_color([parts[3]])
                else:
                    continue
            except ValueError:
                continue
            stops.append((z0, c0))
            stops.append((z1, c1))

    if not stops:
        raise ValueError(f"No colour stops parsed from .cpt file: {path}")

    zs = np.array([s[0] for s in stops], dtype=float)
    zmin, zmax = zs.min(), zs.max()
    span = zmax - zmin if zmax > zmin else 1.0
    seen = {}
    for z, c in stops:
        seen[round((z - zmin) / span, 6)] = c
    positions_colors = sorted(seen.items())
    return mcolors.LinearSegmentedColormap.from_list(name, positions_colors)


def _load_rgb_list_colormap(path, name):
    """Build a ListedColormap from a plain text/CSV file of RGB(A) rows.
    Values may be 0-255 or 0-1; whitespace- or comma-separated."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                rows.append([float(p) for p in parts[:4]])

    if not rows:
        raise ValueError(f"No colour rows parsed from: {path}")

    arr = np.array(rows, dtype=float)
    if arr.max() > 1.0:
        arr[:, :3] /= 255.0
        if arr.shape[1] == 4:
            arr[:, 3] /= 255.0
    return mcolors.ListedColormap(arr, name=name)


def export_colormap_to_cpt(cmap, vmin, vmax, outpath, n_steps=32):
    """
    Export a matplotlib Colormap to a GMT-style .cpt file over [vmin, vmax].

    Reverse of load_colormap()'s .cpt import — samples n_steps+1 points
    across the colourmap and writes them as n_steps colour segments, so a
    colourmap actually used here (a matplotlib built-in name, or something
    already imported from a file/package via load_colormap) can be
    re-exported for use in GMT, or shared with a colleague for an exact
    comparison against a figure made with a named/registered colourmap
    rather than a hand-picked .cpt.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap — resolved via load_colormap
           if not already a Colormap instance
    vmin, vmax : float — data range the colourmap is stretched over; the
           .cpt's own z breakpoints are written in this range so it's
           directly usable for the same data in GMT
    outpath : str — output .cpt file path
    n_steps : int — number of colour segments (n_steps+1 sample points)
    """
    cmap = load_colormap(cmap) if not isinstance(cmap, mcolors.Colormap) else cmap
    zs = np.linspace(vmin, vmax, n_steps + 1)
    fracs = np.linspace(0.0, 1.0, n_steps + 1)
    rgb = (np.array([cmap(f)[:3] for f in fracs]) * 255).round().astype(int)

    lines = ["# COLOR_MODEL = RGB",
             f"# Exported from matplotlib colormap {cmap.name!r} "
             f"over [{vmin}, {vmax}]"]
    for i in range(n_steps):
        z0, z1 = zs[i], zs[i + 1]
        r0, g0, b0 = rgb[i]
        r1, g1, b1 = rgb[i + 1]
        lines.append(f"{z0:<12.6g} {r0:3d} {g0:3d} {b0:3d}   "
                     f"{z1:<12.6g} {r1:3d} {g1:3d} {b1:3d}")

    r0, g0, b0 = rgb[0]
    r1, g1, b1 = rgb[-1]
    lines.append(f"B {r0} {g0} {b0}")
    lines.append(f"F {r1} {g1} {b1}")
    lines.append("N 128 128 128")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Exported colourmap to: {outpath}")


# =====================================================================
# USER SETTINGS
# =====================================================================

# Directory to read precomputed NetCDF files from (must match OUTPUT_DIR
# in tacna_precompute_seis.py). Default "." reads from the current
# directory, matching the previous (fixed) behaviour.
NC_DIR = "./precompute/"

# Directory for saved figures (created if it doesn't exist). Default "."
# writes into the current directory, matching the previous behaviour.
PLOT_DIR = "./plots/"

PLOT_WHAT    = ["vps"]           # any subset of ["vp", "vs", "vps"]
PLOT_FORMATS = [".pdf", ".jpg"]  # output formats
PLOT_DPI     = 600

# Figure sizes (cm).
# Horizontal maps: FIG_WIDTH controls the *map panel's* width only.
# FIG_HEIGHT is always derived from it and the UTM data aspect ratio —
# there is no manual height override — so 1 km in easting always renders
# as exactly the same length as 1 km in northing, guaranteed by
# construction (see create_map_figure()), not merely by an aspect setting
# a colorbar could throw off.
FIG_WIDTH = 10.0   # cm — map panel width; a colorbar (if shown) adds its
                    # own extra width/height beyond this, never competing
                    # with the map for space.

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

# Colourmap names: matplotlib built-in name, OR a path to a colourmap file
# to import (.cpt = GMT colour palette table, .txt/.csv = plain RGB(A)
# list) — useful to reproduce the exact original palette for comparison
# against GMT-produced figures, e.g.:
#   CMAP_VP = "./cpt/viridisr_vp.cpt"
CMAP_VS  = "viridis_r"   # stand-in for viridisr_vs.cpt
CMAP_VP  = "viridis_r"   # stand-in for viridisr_vp.cpt
# CMAP_VPS = "hot_r"       # stand-in for hotr_vps.cpt
CMAP_VPS = "./colormaps/hotr_vps.cpt"       # stand-in for hotr_vps.cpt
CMAP_VS  = load_colormap(CMAP_VS)
CMAP_VP  = load_colormap(CMAP_VP)
CMAP_VPS = load_colormap(CMAP_VPS)

# Export the resolved colourmaps above (each over its own CMIN/CMAX) to GMT
# .cpt files — reverse of the import above. Useful to get an exact GMT
# equivalent of whatever colourmap/range is actually used for this run
# (a matplotlib built-in, or something already imported from a file).
EXPORT_CPT = False
EXPORT_CPT_NSTEPS = 32
EXPORT_CPT_PATHS = {
    "vs":  "seis_vs_cmap.cpt",
    "vp":  "seis_vp_cmap.cpt",
    "vps": "seis_vps_cmap.cpt",
}
if EXPORT_CPT:
    export_colormap_to_cpt(CMAP_VS,  CMIN_VS,  CMAX_VS,
                           EXPORT_CPT_PATHS["vs"],  EXPORT_CPT_NSTEPS)
    export_colormap_to_cpt(CMAP_VP,  CMIN_VP,  CMAX_VP,
                           EXPORT_CPT_PATHS["vp"],  EXPORT_CPT_NSTEPS)
    export_colormap_to_cpt(CMAP_VPS, CMIN_VPS, CMAX_VPS,
                           EXPORT_CPT_PATHS["vps"], EXPORT_CPT_NSTEPS)

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

# Volcano label text: full name vs. short/abbreviated name.
# VOLC_LABEL_FULL_NAME : True  -> use VOLC_NAME_COL_FULL
#                        False -> use VOLC_NAME_COL_SHORT (default)
VOLC_LABEL_FULL_NAME = False
VOLC_NAME_COL_FULL   = "NAME"      # column used when VOLC_LABEL_FULL_NAME=True
VOLC_NAME_COL_SHORT  = "VOLCAN2"   # column used when VOLC_LABEL_FULL_NAME=False

# Pre-computed UTM-km NetCDF files (produced by tacna_precompute.py)
NC_TOPO       = "tacna_topo_utm.nc"
NC_BATH       = "tacna_bath_utm.nc"
NC_TOPO_SHADE = "tacna_topo_shade_utm.nc"  # presence noted; hillshade recomputed

# Region source: "topo" uses topo-grid extent; "data" uses velocity-subset extent
REGION_SOURCE    = "data"
REGION_MARGIN_KM = -0.001

# Explicit override of the map's displayed x/y range (UTM km), applied
# *after* REGION_SOURCE/REGION_MARGIN_KM compute the region above — crops
# (or expands) the displayed view without touching how that region is
# computed. Also feeds the feature-clipping (_in_region), the map figure's
# aspect ratio, and the lon/lat tick overlay, so all stay consistent with
# what's actually drawn. Set to None (default) to use the REGION_SOURCE
# extent unchanged. Analogous to the per-slice "xlim" in VSLICES below.
# MAP_XLIM = None   # e.g. [300.0, 420.0]  (easting,  km)
# MAP_YLIM = None   # e.g. [7960.0, 8080.0] (northing, km)
MAP_XLIM = [310.0, 445.0]    # e.g. [300.0, 420.0]  (easting,  km)
MAP_YLIM = [7971.6, 8120.5]  # e.g. [7960.0, 8080.0] (northing, km)

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
# MAP AXES UNITS
# Selects what the map's bottom/left tick labels show — one or the other,
# not both (the tick *positions* are simply relabelled in place; no extra
# axes are added).
# AXES_UNITS : "km"     — UTM easting/northing in km (default).
#              "latlon" — longitude/latitude in degrees.
# LATLON_NTICKS   : number of tick positions when AXES_UNITS="latlon"
# LATLON_DECIMALS : decimal places on the lon/lat tick labels
# AXES_KM_COMMA   : when AXES_UNITS="km", add a thousands comma
#                   (American style, e.g. "8,000"). False -> plain "8000".
#                   Has no effect when AXES_UNITS="latlon".
# =====================================================================
AXES_UNITS       = "km"   # "km" | "latlon"
LATLON_NTICKS    = 5
LATLON_DECIMALS  = 2
AXES_KM_COMMA    = True

# =====================================================================
# COLORBAR SETTINGS
# SHOW_COLORBAR      : False omits the colorbar entirely — the map panel
#                      itself is completely unaffected either way.
# COLORBAR_POSITION  : "right" | "left" | "bottom" | "top"
#   right / left  -> vertical bar, placed outside the map on that side
#   bottom / top  -> horizontal bar, placed outside the map on that side
# The colorbar is placed in its own explicitly-sized axes, added as EXTRA
# width (right/left) or height (bottom/top) beyond the map panel — it
# never steals space from the map, so it can never distort its scale.
# COLORBAR_SIZE      : bar length, as a fraction (0-1) of the map edge
#                      it's attached to
# COLORBAR_PAD       : gap between map and bar, in inches
# COLORBAR_ASPECT    : bar length / bar thickness (thickness is derived)
# COLORBAR_LABEL_*   : font sizes for the bar label and tick annotations
# =====================================================================
SHOW_COLORBAR       = True
COLORBAR_POSITION   = "right"   # "right" | "left" | "bottom" | "top"
COLORBAR_SIZE       = 0.85      # bar length, fraction of the map edge
COLORBAR_PAD        = 0.10      # inches between map axes and colorbar
COLORBAR_ASPECT     = 20        # bar length / bar thickness
COLORBAR_LABEL_SIZE = 8         # pt, label font size
COLORBAR_TICK_SIZE  = 7         # pt, tick annotation font size
COLORBAR_NTICKS     = 5         # approximate number of tick intervals

# =====================================================================
# MAP FEATURE LAYERS — simple on/off switches
# Each flag controls one overlay layer on the map. SHOW_SEISMICITY and
# SHOW_MT_SITES also control the matching projection onto vertical
# sections (VSLICE_EQ_STYLE), so turning a feature off applies everywhere
# it would otherwise appear, not just on the map.
# =====================================================================
SHOW_PROFILE_LINES    = True   # static profile_CD / profile_2 lines
SHOW_VSLICE_LINES     = True   # VSLICES cross-section lines + endpoint labels
SHOW_SEISMICITY       = True
SHOW_MT_SITES         = True
SHOW_VOLCANOES        = True   # inactive volcano markers + labels
SHOW_VOLCANOES_ACTIVE = True   # active volcano markers
SHOW_CITIES           = True
SHOW_NORTH_ARROW      = True

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

# Inverted-triangle marker for MT sites whose *apex* — not its centroid —
# lands exactly on the site coordinate, like a map pin pointing down at the
# true position. A plain marker="v" is centred on the point instead, so the
# tip would sit visibly below the real location.
_MT_PIN_VERTS = [(0.0, 0.0), (-1.0, 1.732), (1.0, 1.732), (0.0, 0.0)]
_MT_PIN_CODES = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
MT_PIN_MARKER = MplPath(_MT_PIN_VERTS, _MT_PIN_CODES)

# --- MT sites ---
MT_MARKER_STYLE = dict(
    marker=MT_PIN_MARKER, s=14, facecolors="blue", edgecolors="black",
    linewidths=0.4, alpha=0.6, zorder=12,
)

# --- Inactive volcanoes ---
VOLC_INACT_MARKER_STYLE = dict(
    marker="^", s=16, facecolors="black", edgecolors="black",
    linewidths=0.2, zorder=13,
)
VOLC_LABEL_STYLE = dict(
    fontsize=6, fontweight="bold", color="black", zorder=14,
    offset_x=0.3, offset_y=0.3,   # km offsets from marker centre
)

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
    fontsize=6, color="black", zorder=14,
    offset_x=0.3, offset_y=-0.3,  # km offsets from marker centre
)

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
#   xlim    : optional [xmin, xmax] — crop the *displayed* x-axis range
#             without recomputing anything. Units must match VSLICE_X_AXIS
#             below: UTM easting/northing (km) for "utm", or cumulative
#             distance from p1 (km) for "distance". The full profile is
#             still sampled/interpolated (npts/nz unaffected) — this only
#             narrows the plotted view, so it's cheap to iterate on for
#             fine-tuning a figure. Omit or set to None for the full
#             profile (default).
#   ylim    : optional [ytop, ybottom] — crop the *displayed* depth-axis
#             range without recomputing anything. Depth in km, positive
#             down, given top-first (ytop is the shallower/smaller value,
#             e.g. negative to include topography). Omit or set to None
#             for the default range (topo/headroom-derived top, zmax_km
#             at the bottom).
#
# Set VSLICES = [] to skip all vertical sections.
# =====================================================================
VSLICES = [
    dict(
        name    = "profile_CD",
        p1      = [-70.476, -18.255],   # lon, lat
        p2      = [-69.499, -17.048],
        coord   = "latlon",
        # zmin_km must be negative enough to reach above sea level, or real
        # seismicity there (e.g. within a volcanic edifice — inside the
        # grey topo fill) gets silently excluded by the (zeqs >= zmin_km)
        # filter in _project_seismicity_to_profile — the catalogue used
        # here has events down to z = -5.75 km. -8.0 gives some margin.
        # (Same fix as tacna_plot_modem_image.py's VSLICES.)
        zmin_km = -8.0,
        zmax_km = 30.0,
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
VSLICE_VE = 3.0

# Force true equal x/y (km) scale on sections, overriding VSLICE_VE with
# 1.0 whenever True. Off by default: real profiles are typically much
# longer than they are deep, so a literal 1:1 scale usually isn't what
# you want day-to-day — VSLICE_VE stays the normal, always-available
# control for how exaggerated (or not) a section looks. This flag exists
# only for the occasional figure where true, undistorted scale actually
# matters (e.g. comparing directly against a map at the same scale).
VSLICE_EQUAL_SCALE = False

# VE-label placement on cross-section figures.
# VSLICE_VE_POS : one of "lower right", "lower left", "upper right",
#                 "upper left", or an explicit (x, y, ha, va) tuple in
#                 axes-fraction coordinates.
# VSLICE_VE_STYLE : remaining ax.text() kwargs (fontsize, color, etc.)
VSLICE_VE_POS   = "lower right"
VSLICE_VE_STYLE = dict(fontsize=7, color="black")

# Horizontal axis for vertical sections:
#   "utm"      — UTM easting or northing (km), xlim = profile endpoint coords
#   "distance" — cumulative distance from p1 (km), xlim = [0, profile_length]
VSLICE_X_AXIS = "distance" #"utm"

# Seismicity marker style on cross-section (overrides EQ_MARKER_STYLE)
VSLICE_EQ_STYLE = dict(
    s=4, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=11,
)

# Topographic surface line style on cross-section
VSLICE_TOPO_STYLE = dict(color="dimgray", lw=0.5, zorder=12)

# Whether to shade the band between the topographic surface and z = 0
# (sea level / top of the velocity model) — i.e. the part of the section
# that lies above the model, not itself part of the data. False (default)
# draws only the topography line (VSLICE_TOPO_STYLE) with no fill; True
# fills that band using VSLICE_TOPO_LAND_COLOR / VSLICE_TOPO_OCEAN_COLOR.
VSLICE_SHOW_TOPO_FILL = False

# Fill colours for the topography band above the section (only drawn when
# VSLICE_SHOW_TOPO_FILL is True)
VSLICE_TOPO_LAND_COLOR  = "gray"    # z > 0 (above sea level)
VSLICE_TOPO_OCEAN_COLOR = "#6baed6" # z <= 0 (below sea level)

# Extra headroom above the highest topographic point (km).
# The panel top is set to (max_elevation_km + VSLICE_TOPO_HEADROOM_KM)
# above the section zmin, so the topo fill is not flush with the axes edge.
VSLICE_TOPO_HEADROOM_KM = 1.0

# Style of the profile line drawn on the map figures
VSLICE_MAP_LINE_STYLE = dict(color="magenta", lw=0.8, ls="--", zorder=15)

# --- Free-text annotation (optional) ---
# Draws one extra line of arbitrary text on every figure this script
# produces (both depth slices and vertical sections) — e.g. a version tag,
# a processing note, or a "DRAFT" watermark. Set to None or "" to disable.
ANNOTATION_TEXT  = None                  # e.g. "Preliminary — v3 mesh"
ANNOTATION_POS   = (0.01, 0.99)          # (x, y) in axes-fraction coords
ANNOTATION_STYLE = dict(fontsize=7, color="gray", ha="left", va="top")

# =====================================================================
# END USER SETTINGS
# =====================================================================

os.makedirs(PLOT_DIR, exist_ok=True)


def ncpath(name):
    """Join a bare precomputed-NetCDF filename onto NC_DIR."""
    return os.path.join(NC_DIR, name)


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
        out = os.path.join(PLOT_DIR, stem + fmt)
        fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {out}")


def draw_annotation(ax):
    """Draw the optional free-text annotation (ANNOTATION_TEXT), if set."""
    if ANNOTATION_TEXT:
        ax.text(*ANNOTATION_POS, ANNOTATION_TEXT,
                transform=ax.transAxes, zorder=25, **ANNOTATION_STYLE)


# ------------------------------------------------------------------
# VE-label position resolver (used by plot_vertical_slice)
# ------------------------------------------------------------------
_VE_POS_PRESETS = {
    "upper right": (0.99, 0.99, "right", "top"),
    "upper left":  (0.01, 0.99, "left",  "top"),
    "lower right": (0.99, 0.01, "right", "bottom"),
    "lower left":  (0.01, 0.01, "left",  "bottom"),
}


def _resolve_ve_pos(spec):
    """Resolve VSLICE_VE_POS into an (x, y, ha, va) tuple in axes fraction."""
    if isinstance(spec, str):
        try:
            return _VE_POS_PRESETS[spec.lower()]
        except KeyError:
            raise ValueError(
                f"VSLICE_VE_POS={spec!r} not recognised; choose one of "
                f"{list(_VE_POS_PRESETS)} or an explicit (x, y, ha, va) tuple."
            )
    return spec


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
    withStroke path-effect. Callers may pass a shared/global style dict
    directly — it's copied internally, never mutated.
    """
    style_dict = dict(style_dict)
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
# Map figure creation — guarantees equal x/y (km) scale BY CONSTRUCTION
# ------------------------------------------------------------------
def _build_panel_figure(panel_w_in, panel_h_in, size_label="panel"):
    """
    Shared machinery behind create_map_figure() and create_section_figure():
    given a panel's exact physical size in inches, place it (and an
    optional colorbar, added as EXTRA canvas beyond the panel) via
    explicit inch-based axes placement — never matplotlib's automatic
    colorbar space-stealing (fig.colorbar(..., ax=...)) or tight_layout(),
    both of which can produce a badly broken layout for panels with an
    extreme aspect ratio (e.g. a long, shallow cross-section).

    Returns (fig, ax, cax) — cax is the colorbar axes, or None if
    SHOW_COLORBAR is False.
    """
    pos = COLORBAR_POSITION.lower()
    if pos not in ("right", "left", "bottom", "top"):
        raise ValueError(
            f"COLORBAR_POSITION={COLORBAR_POSITION!r} is not valid. "
            "Choose 'right', 'left', 'bottom', or 'top'."
        )

    pad_in = COLORBAR_PAD
    bar_len_in = bar_thick_in = 0.0
    cbar_w_in = cbar_h_in = 0.0
    if SHOW_COLORBAR:
        if pos in ("right", "left"):
            bar_len_in = COLORBAR_SIZE * panel_h_in
            cbar_w_in = bar_thick_in = bar_len_in / COLORBAR_ASPECT
        else:
            bar_len_in = COLORBAR_SIZE * panel_w_in
            cbar_h_in = bar_thick_in = bar_len_in / COLORBAR_ASPECT

    fig_w_in = panel_w_in + (cbar_w_in + pad_in if cbar_w_in else 0.0)
    fig_h_in = panel_h_in + (cbar_h_in + pad_in if cbar_h_in else 0.0)
    print(f"Figure size ({size_label}): {fig_w_in:.2f} × {fig_h_in:.2f} in "
          f"({size_label} {panel_w_in:.2f} × {panel_h_in:.2f} in)")

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    panel_left   = (cbar_w_in + pad_in) / fig_w_in if (SHOW_COLORBAR and pos == "left") else 0.0
    panel_bottom = (cbar_h_in + pad_in) / fig_h_in if (SHOW_COLORBAR and pos == "bottom") else 0.0
    panel_w_frac = panel_w_in / fig_w_in
    panel_h_frac = panel_h_in / fig_h_in
    ax = fig.add_axes([panel_left, panel_bottom, panel_w_frac, panel_h_frac])

    cax = None
    if SHOW_COLORBAR:
        bar_len_frac = (bar_len_in / fig_h_in) if pos in ("right", "left") \
            else (bar_len_in / fig_w_in)
        if pos == "right":
            cax = fig.add_axes([
                (panel_w_in + pad_in) / fig_w_in,
                panel_bottom + (panel_h_frac - bar_len_frac) / 2,
                cbar_w_in / fig_w_in, bar_len_frac,
            ])
        elif pos == "left":
            cax = fig.add_axes([
                0.0,
                panel_bottom + (panel_h_frac - bar_len_frac) / 2,
                cbar_w_in / fig_w_in, bar_len_frac,
            ])
        elif pos == "top":
            cax = fig.add_axes([
                panel_left + (panel_w_frac - bar_len_frac) / 2,
                (panel_h_in + pad_in) / fig_h_in,
                bar_len_frac, cbar_h_in / fig_h_in,
            ])
        elif pos == "bottom":
            cax = fig.add_axes([
                panel_left + (panel_w_frac - bar_len_frac) / 2,
                0.0,
                bar_len_frac, cbar_h_in / fig_h_in,
            ])

    return fig, ax, cax


def create_map_figure():
    """
    Build a horizontal-map figure whose map axes is sized in physical
    inches to exactly match the UTM data aspect ratio ((ymax-ymin) /
    (xmax-xmin)) — so 1 km in easting always renders as exactly the same
    length as 1 km in northing, guaranteed by explicit inch-based axes
    placement rather than relying on matplotlib's 'equal' aspect setting
    plus automatic colorbar space-stealing (fig.colorbar(..., ax=...)),
    which can desync from the actual rendered box in some layouts.

    FIG_WIDTH controls only the map panel's width; height is always
    derived. If SHOW_COLORBAR is True, the colorbar is added as EXTRA
    width (right/left) or height (bottom/top) beyond the map panel, so it
    never competes with the map for space and can never distort it.

    Returns (fig, ax, cax) — cax is the colorbar axes, or None if
    SHOW_COLORBAR is False.
    """
    map_w_in = FIG_WIDTH / 2.54
    map_h_in = map_w_in * (ymax - ymin) / (xmax - xmin)
    return _build_panel_figure(map_w_in, map_h_in, size_label="map")


def create_section_figure(w_in, h_in):
    """
    Build a vertical-section figure at the given panel size (in inches —
    already computed by the caller from VSLICE_WIDTH_CM and the profile's
    own depth-range/VE-derived aspect ratio). Same explicit inch-based
    axes placement as create_map_figure(), which matters most here: real
    profiles are usually much longer than they are deep, and the old
    tight_layout()-plus-space-stealing-colorbar approach could produce a
    badly broken/overlapping layout for that kind of wide, short panel.

    Returns (fig, ax, cax) — cax is the colorbar axes, or None if
    SHOW_COLORBAR is False.
    """
    return _build_panel_figure(w_in, h_in, size_label="section")


def finish_panel_colorbar(cax, mappable, label):
    """Render the colorbar into the cax returned by create_map_figure()
    or create_section_figure()."""
    if cax is None:
        return None
    pos = COLORBAR_POSITION.lower()
    orientation = "vertical" if pos in ("right", "left") else "horizontal"
    cbar = cax.figure.colorbar(mappable, cax=cax, orientation=orientation)
    cbar.set_label(label, fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    cbar.locator = mpl.ticker.MaxNLocator(nbins=COLORBAR_NTICKS)
    cbar.update_ticks()
    if pos == "left":
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
    if pos == "top":
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
    return cbar


# ------------------------------------------------------------------
# Secondary lon/lat axes  (cosmetic overlay on UTM-km plot)
# ------------------------------------------------------------------
def add_latlon_ticks(ax):
    """
    Replace the UTM-km tick labels on the primary axes with lon/lat values.
    No extra axes are created — the existing bottom x-axis and left y-axis
    ticks are reformatted in-place using FuncFormatter.

    Tick *positions* are chosen at round lon/lat values (e.g. 0.1/0.2/0.5°
    steps, picked automatically via matplotlib's MaxNLocator) rather than at
    evenly spaced UTM-km positions — the round geographic values are then
    converted back to UTM km to place the ticks.

    Controlled by AXES_UNITS, LATLON_NTICKS, LATLON_DECIMALS.
    """
    e_mid_m = (xmin + xmax) / 2.0 * 1e3
    n_mid_m = (ymin + ymax) / 2.0 * 1e3
    fmt = f"{{:.{LATLON_DECIMALS}f}}°"

    # Geographic extent of the map along each edge (mid-line of the other axis)
    lon_min, _ = _to_geo.transform(xmin * 1e3, n_mid_m)
    lon_max, _ = _to_geo.transform(xmax * 1e3, n_mid_m)
    _, lat_min = _to_geo.transform(e_mid_m, ymin * 1e3)
    _, lat_max = _to_geo.transform(e_mid_m, ymax * 1e3)

    # Round tick values (nice 1/2/5-type steps), clipped to the map extent
    lon_locator = mpl.ticker.MaxNLocator(nbins=LATLON_NTICKS, steps=[1, 2, 5, 10])
    lat_locator = mpl.ticker.MaxNLocator(nbins=LATLON_NTICKS, steps=[1, 2, 5, 10])
    lon_vals = [v for v in lon_locator.tick_values(min(lon_min, lon_max), max(lon_min, lon_max))
                if min(lon_min, lon_max) <= v <= max(lon_min, lon_max)]
    lat_vals = [v for v in lat_locator.tick_values(min(lat_min, lat_max), max(lat_min, lat_max))
                if min(lat_min, lat_max) <= v <= max(lat_min, lat_max)]

    # Convert round lon/lat values back to UTM km for tick placement
    e_ticks_km = np.array([_to_utm.transform(lon, (lat_min + lat_max) / 2.0)[0]
                            for lon in lon_vals]) / 1e3
    n_ticks_km = np.array([_to_utm.transform((lon_min + lon_max) / 2.0, lat)[1]
                            for lat in lat_vals]) / 1e3

    lon_labels = [fmt.format(v) for v in lon_vals]
    lat_labels = [fmt.format(v) for v in lat_vals]

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
    _ds = xr.open_dataset(ncpath(fname))
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
             ylim defaults to the topo/headroom-derived top and zmax_km.
    Endpoint labels (lbl_start, lbl_end) are annotated on the x-axis.
    """
    swath  = vslice.get("swath_km", 10.0)
    zmin_s = vslice.get("zmin_km", depth_km[0])
    zmax_s = vslice.get("zmax_km", depth_km[-1])
    ve     = 1.0 if VSLICE_EQUAL_SCALE else VSLICE_VE
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

    fig, ax, cax = create_section_figure(w_in, h_in)

    # VE label drawn first (low zorder) so the (semi-transparent) data
    # image sits over it rather than being covered by it.
    if ve != 1.0:
        vx, vy, vha, vva = _resolve_ve_pos(VSLICE_VE_POS)
        ax.text(vx, vy, f"VE = {ve:.1f}×",
                transform=ax.transAxes, ha=vha, va=vva,
                zorder=1, **VSLICE_VE_STYLE)

    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    im = ax.pcolormesh(
        x_arr, depth_km, section,
        cmap=cmap, norm=norm, shading="gouraud",
        alpha=1.0 - ALPHA_VELOCITY, zorder=5,
    )
    # NOTE: previously used shading="auto" + set_rasterized(True) to fix
    # vector-format seams between cells, but rasterized artists render
    # upside-down under matplotlib's PDF/PS backends when combined with
    # ax.invert_yaxis() (used below for the depth axis) — a known
    # matplotlib bug, not something in this pipeline. Gouraud shading
    # (smooth per-vertex interpolation, no discrete cell polygons) removes
    # the seams without rasterizing anything, so the inverted axis renders
    # correctly.

    # Topo line / optional fill.
    # surf_depth is the topographic surface expressed on the depth axis,
    # anchored to z = 0 (sea level / top of the velocity model). Previously
    # this was offset by depth_km[0] (the section's requested zmin_km)
    # instead of 0, which shifted both the line and the fill upward by
    # however far zmin_km sat above sea level — fixed here.
    y_top = depth_km[0]
    if topo_prof is not None:
        surf_depth = -topo_prof / 1e3
        y_top = surf_depth.min() - VSLICE_TOPO_HEADROOM_KM

        if VSLICE_SHOW_TOPO_FILL:
            land  = topo_prof >  0
            ocean = topo_prof <= 0

            if land.any():
                ax.fill_between(x_arr, 0.0, surf_depth,
                                where=land,
                                color=VSLICE_TOPO_LAND_COLOR, alpha=0.5,
                                zorder=6, interpolate=True)
            if ocean.any():
                ax.fill_between(x_arr, 0.0, surf_depth,
                                where=ocean,
                                color=VSLICE_TOPO_OCEAN_COLOR, alpha=0.5,
                                zorder=6, interpolate=True)
        ax.plot(x_arr, surf_depth, **VSLICE_TOPO_STYLE)

    # Seismicity
    if SHOW_SEISMICITY and len(eq_x):
        ax.scatter(eq_x, eq_dep, **VSLICE_EQ_STYLE)

    # Axes
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

    finish_panel_colorbar(cax, im, cbar_label)
    draw_annotation(ax)
    save_fig(fig, stem)
    plt.close(fig)


# ==================================================================
# Load static grids
# ==================================================================
print("Loading topo grid …")
_topo_da = xr.open_dataarray(ncpath(NC_TOPO))
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
_bath_da = xr.open_dataarray(ncpath(NC_BATH))
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
    _ds = xr.open_dataset(ncpath("tacna_vp.nc"))
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

if MAP_XLIM is not None:
    xmin, xmax = MAP_XLIM
if MAP_YLIM is not None:
    ymin, ymax = MAP_YLIM
if MAP_XLIM is not None or MAP_YLIM is not None:
    print(f"UTM region overridden by MAP_XLIM/MAP_YLIM: "
          f"[{xmin}, {xmax}, {ymin}, {ymax}]")
else:
    print(f"UTM region (km): {utm_region}")


# ==================================================================
# Feature layers
# ==================================================================
volcanes   = pd.read_csv(CSV_VOLCANES)
utmv_e, utmv_n = to_utm_km(
    volcanes["LONG"][VOLC_LABEL_IDX].values,
    volcanes["LAT"][VOLC_LABEL_IDX].values,
)
_volc_name_col = VOLC_NAME_COL_FULL if VOLC_LABEL_FULL_NAME else VOLC_NAME_COL_SHORT
if _volc_name_col not in volcanes.columns:
    print(f"  WARNING: volcano name column {_volc_name_col!r} not found in "
          f"{CSV_VOLCANES} — falling back to {VOLC_NAME_COL_SHORT!r}.")
    _volc_name_col = VOLC_NAME_COL_SHORT
namev = volcanes[_volc_name_col][VOLC_LABEL_IDX].values

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
depth_coord = xr.open_dataset(ncpath("tacna_vp.nc"))["depth"]


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
    if AXES_UNITS == "km" and AXES_KM_COMMA:
        _comma_fmt = mpl.ticker.StrMethodFormatter("{x:,.0f}")
        ax.xaxis.set_major_formatter(_comma_fmt)
        ax.yaxis.set_major_formatter(_comma_fmt)
    ax.tick_params(labelsize=7)


def draw_features(ax, eq_e, eq_n):
    """Overlay all feature layers; all markers/labels clipped to map region."""

    # Profile lines — clip at axes boundary automatically via clip_on
    if SHOW_PROFILE_LINES:
        ax.plot(prof_cd_e, prof_cd_n, clip_on=True, **PROFILE_CD_STYLE)
        ax.plot(prof2_e,   prof2_n,   clip_on=True, **PROFILE_2_STYLE)

    # Vertical slice profile lines drawn on the map
    if SHOW_VSLICE_LINES:
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
    if SHOW_SEISMICITY:
        clipped_scatter(ax, eq_e, eq_n, label="Seismicity", **EQ_MARKER_STYLE)

    # MT sites
    if SHOW_MT_SITES:
        clipped_scatter(ax, mt_e, mt_n, label="MT site", **MT_MARKER_STYLE)

    # Inactive volcanoes
    if SHOW_VOLCANOES:
        clipped_scatter(ax, utmv_e, utmv_n, **VOLC_INACT_MARKER_STYLE)
        clipped_labels(ax, utmv_e, utmv_n, namev, VOLC_LABEL_STYLE)

    # Active volcanoes
    if SHOW_VOLCANOES_ACTIVE and volc_act_e:
        clipped_scatter(ax, volc_act_e, volc_act_n,
                        label="Active volcano", **VOLC_ACT_MARKER_STYLE)

    # Cities
    if SHOW_CITIES:
        clipped_scatter(ax, cit_e, cit_n, label="City", **CITY_MARKER_STYLE)
        clipped_labels(ax, cit_e, cit_n, name_cit, CITY_LABEL_STYLE)

    # North arrow
    if SHOW_NORTH_ARROW:
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

    vps_nc = ncpath(f"tacna_vps_utm_{tag}.nc")
    vp_nc  = ncpath(f"tacna_vp_utm_{tag}.nc")
    vs_nc  = ncpath(f"tacna_vs_utm_{tag}.nc")

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

        fig, ax, cax = create_map_figure()
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
        finish_panel_colorbar(cax, im, "Vp/Vs")
        if AXES_UNITS == "latlon":
            add_latlon_ticks(ax)
        draw_annotation(ax)
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

        fig, ax, cax = create_map_figure()
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
        finish_panel_colorbar(cax, im, "Vp (m/s)")
        if AXES_UNITS == "latlon":
            add_latlon_ticks(ax)
        draw_annotation(ax)
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

        fig, ax, cax = create_map_figure()
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
        finish_panel_colorbar(cax, im, "Vs (m/s)")
        if AXES_UNITS == "latlon":
            add_latlon_ticks(ax)
        draw_annotation(ax)
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
