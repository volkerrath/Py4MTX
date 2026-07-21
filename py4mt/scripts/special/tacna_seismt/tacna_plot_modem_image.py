#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tacna_plot_modem_image.py
=========================
Companion plotting script for tacna_precompute_modem.py.

Produces depth-slice maps of log10(ρ) (or linear ρ) from a ModEM 3-D MT
inversion result for the Tacna region.  Reads the UTM-km NetCDF files
produced by tacna_precompute_modem.py; no GMT/PyGMT required.

Shares the same basemap engine (topo hillshade, ocean fill, feature overlays,
colourbar placement, clipping) as tacna_plot_seis.py.  The main differences
are:

* Data overlay: log10(ρ) depth slices from modem_rho_utm_{D}km.nc
  instead of seismic-velocity grids.
* Sensitivity-based shading/blanking: if tacna_precompute_modem.py found a
  .sns sensitivity/resolution file, modem_sens_utm.nc /
  modem_sens_utm_{D}km.nc are used to blank (NaN) and/or shade
  poorly-resolved cells on both the horizontal slices and the vertical
  sections — see USE_SENSITIVITY, SENS_BLANK_THRESHOLD, SENS_SHADE_RANGE.
* MT sites: read from modem_sites_utm.nc (produced by the precompute script)
  instead of a plain CSV — avoids redundant coordinate conversion.
* No seismicity depth-window loop: a single seismicity catalogue depth filter
  per slice, controlled by ZMIN_SEISM / ZMAX_SEISM lists.
* Topo NetCDF: modem_topo_utm.nc (elevation in metres, positive up,
  dims northing/easting) produced by the ModEM precompute script.
* No bathymetry file is required from the precompute script; the seis-pipeline
  bath grid (tacna_bath_utm.nc) can be reused if available, otherwise the
  ocean fill is skipped gracefully.

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
import sys

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker
from matplotlib.path import Path as MplPath
from scipy.interpolate import RegularGridInterpolator

import plotpy

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------
# Colormap import helper (matplotlib name / GMT .cpt file / plain RGB(A)
# file) — see plotpy.load_colormap for the full docstring.
# ---------------------------------------------------------------------
load_colormap = plotpy.load_colormap
export_colormap_to_cpt = plotpy.export_colormap_to_cpt


# =====================================================================
# USER SETTINGS
# =====================================================================


# Directory to read precomputed NetCDF files from (must match OUTPUT_DIR
# in tacna_precompute_modem.py). Default "." reads from the current
# directory, matching the previous (fixed) behaviour.
NC_DIR = "../precompute/"

# Directory for saved figures (created if it doesn't exist). Default "."
# writes into the current directory, matching the previous behaviour.
PLOT_DIR = "../plots/"

# Appended to every saved figure's filename (before the extension) — lets
# output from this script (resampled "image" rendering) be told apart at
# a glance from tacna_plot_modem_mesh.py's exact-mesh output, e.g.
# "modem_rho_1km_tacna_img.pdf" vs "..._msh.pdf". Set to "" to disable.
PLOT_FILENAME_SUFFIX = "_img"


PLOT_FORMATS = [".pdf", ".jpg"]
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

VSLICE_WIDTH_CM  = 14.0
VSLICE_HEIGHT_CM = None  # None → derived from VE and profile length

# Depth slices to plot — must match values used in tacna_precompute_modem.py.
# Each entry corresponds to one DEPTH_SLICES_KM value; tag strings are
# constructed the same way as in the precompute script.
DEPTH_SLICES_KM = [1.0, 5.0, 9.0]

# Seismicity depth windows (km), one pair per entry in DEPTH_SLICES_KM.
# Set both to None to show all seismicity on every slice.
# Contiguous, non-overlapping bins split at the midpoint between adjacent
# slice depths (1, 5, 9 km -> split at 3 and 7 km), so every earthquake
# is assigned to exactly one slice — the one closest to its own depth —
# rather than showing everything on every map or leaving gaps. The
# outermost bins are left open (None) so no shallow/deep event is
# arbitrarily excluded. Same scheme as ZMIN_SEISM/ZMAX_SEISM in
# tacna_plot_seis.py and tacna_plot_modem_mesh.py.
ZMIN_SEISM = [None, 3.0, 7.0]
ZMAX_SEISM = [3.0,  7.0, None]

if not (len(ZMIN_SEISM) == len(ZMAX_SEISM) == len(DEPTH_SLICES_KM)):
    sys.exit(
        f"ZMIN_SEISM ({len(ZMIN_SEISM)}), ZMAX_SEISM ({len(ZMAX_SEISM)}), "
        f"and DEPTH_SLICES_KM ({len(DEPTH_SLICES_KM)}) must all be the same "
        f"length — one seismicity depth-window pair per depth slice. Pad "
        f"the shorter list(s) with None (= show all seismicity) for any "
        f"slice that doesn't need a filter."
    )

# Colour-scale limits for log10(ρ) [Ω·m]; adjust to your model range
CMIN_RHO = 0.    # log10(Ω·m) — ~3 Ω·m
CMAX_RHO = 3.    # log10(Ω·m) — ~10,000 Ω·m

# Air cells above the model surface are stored with RHO_AIR (~1e17 Ω·m,
# i.e. log10(ρ) ≈ 17) in tacna_precompute_modem.py. Any cell at or above this
# threshold is treated as air/no-data and masked to NaN — independent of
# CMIN_RHO/CMAX_RHO, so changing the display colour range never re-exposes
# the air layer. Used for both the horizontal depth slices and the vertical
# sections.
AIR_LOG10_RHO_THRESHOLD = 10.0

# --- Sensitivity-based shading/blanking (optional) ---
# Reads modem_sens_utm.nc / modem_sens_utm_{D}km.nc from
# tacna_precompute_modem.py (only produced if USE_SENSITIVITY was True
# there, and a .sns file was found). Units match SENS_TRANSFORM chosen in
# that script (default log10(sensitivity)). Applies to both the horizontal
# depth slices and the vertical sections.
#
# Missing sensitivity data (NaN — e.g. outside the .sns file's own
# coverage) is always treated as fully shaded/blanked: we have no basis for
# claiming a cell is well-resolved just because we have no information.
USE_SENSITIVITY = False

# Cells with sensitivity below this are fully blanked (set to NaN,
# transparent) — poorly-resolved regions disappear entirely, the same way
# air cells do. Set to None to disable blanking.
# NOTE: kept ON *in addition to* SENS_ALPHA_RANGE below — belt-and-braces.
# SENS_ALPHA_RANGE already makes these cells fully transparent by itself,
# but blanking to NaN also removes them from the colour-scale/data array,
# which matters if this array is reused elsewhere (e.g. exported).
SENS_BLANK_THRESHOLD = -4   # log10(1e-4) — sensitivity < 1e-4

# Cells with sensitivity between these two values get a smooth semi-
# transparent white overlay fading from fully shaded (SENS_SHADE_MAX_ALPHA,
# at or below the first value) to unshaded (0 alpha, at or above the
# second) — a softer "how much to trust this" cue than a hard blank cutoff.
# Set to None to disable shading. Values are in the same units as
# SENS_TRANSFORM.
# Switched OFF for now: it was washing everything from -4 to 0 with an
# opaque-ish white layer, which is a *different* effect from "make the
# low-sensitivity area transparent" (it actually makes it more opaque, just
# white instead of coloured) and was likely what made sensitivity masking
# look like it wasn't doing anything useful. Re-enable if you also want
# that softer cue on top of the hard cutoff below.
SENS_SHADE_RANGE = None   # e.g. (-2.0, 0.0)
SENS_SHADE_COLOR = "white"
SENS_SHADE_MAX_ALPHA = 0.85

# Cells with sensitivity between these two values fade the *data layer
# itself* (the resistivity colour, not an overlay on top of it) from fully
# transparent (at/below the first value) to its normal opacity
# (1 - ALPHA_RHO, at/above the second). Unlike SENS_SHADE_RANGE — which
# washes poorly-resolved cells with an extra flat colour on top — this
# lets whatever is drawn underneath (the topography hillshade basemap, in
# particular) show straight through in poorly-resolved areas, which reads
# better than grey/white wash when the point is to relate resistivity
# structure to topography. Can be used together with SENS_SHADE_RANGE/
# SENS_BLANK_THRESHOLD, or on its own. Set to None to disable (data alpha
# stays the constant 1 - ALPHA_RHO everywhere, as before). Values are in
# the same units as SENS_TRANSFORM (log10 sensitivity by default).
#
# NOTE on the upper bound: with SENS_TRANSFORM="LOG10", sensitivity itself
# is normalised to [0, 1], so log10(sensitivity) never exceeds 0 — the
# best-resolved cell you can ever have sits at exactly 0. That means the
# *second* value of this range (where opacity reaches its normal maximum)
# should essentially always stay 0; only the first value (how lenient the
# cutoff is) is worth changing.
#
# Example (a) — sharp cutoff, no fade: everything below the threshold is
# fully transparent, everything at/above it is fully opaque, nothing in
# between. This is what's currently active below.
SENS_ALPHA_RANGE = (-4., -4.)
# SENS_ALPHA_RANGE = (-3., -3.)    # more lenient cutoff (sens < 1e-3)
# SENS_ALPHA_RANGE = (-2., -2.)    # stricter cutoff (sens < 1e-2)
#
# Example (b) — soft fade from -2 up to 0: cells fade in gradually as
# sensitivity improves from 1e-2 to 1 (fully resolved), rather than
# snapping straight from invisible to fully opaque.
# SENS_ALPHA_RANGE = (-2., 0.)

# Colourmap: matplotlib built-in name ("jet_r", "RdBu", "turbo_r", "bwr_r",
# etc.), OR a path to a colourmap file to import (.cpt = GMT colour palette
# table, .txt/.csv = plain RGB(A) list) — useful to match a specific
# published palette, or the same palette used in tacna_plot_seis.py, for
# direct visual comparison, e.g.:
#   CMAP_RHO = "../cpt/rho_gmt.cpt"
CMAP_RHO = "jet_r"
CMAP_RHO = load_colormap(CMAP_RHO)

# Export the resolved colourmap above (over [CMIN_RHO, CMAX_RHO]) to a GMT
# .cpt file — reverse of the import above. Useful to get an exact GMT
# equivalent of whatever colourmap/range is actually used for this run
# (a matplotlib built-in, or something already imported from a file).
EXPORT_CPT = False
EXPORT_CPT_PATH = "modem_rho_cmap.cpt"
EXPORT_CPT_NSTEPS = 32
if EXPORT_CPT:
    export_colormap_to_cpt(CMAP_RHO, CMIN_RHO, CMAX_RHO, EXPORT_CPT_PATH,
                           EXPORT_CPT_NSTEPS)

# Resistivity overlay transparency (0 = opaque, 1 = invisible)
ALPHA_RHO = 0.45

# Pre-computed UTM-km NetCDF files from tacna_precompute_modem.py
NC_TOPO_MODEM = "modem_topo_utm.nc"    # 2-D elevation, dims (northing, easting)
NC_SITES      = "modem_sites_utm.nc"   # MT site positions

# Optional: reuse bathymetry from the seis pipeline if available
NC_BATH = "tacna_bath_utm.nc"          # set to "" to skip ocean fill

# Optional: reuse external topo/hillshade from seis pipeline if available.
# If set, overrides NC_TOPO_MODEM for the greyscale basemap (the ModEM topo
# is coarser and may show mesh artefacts).
# Set to "" to use the ModEM topography extracted from the model.
NC_TOPO_SEIS = "tacna_topo_utm.nc"     # set to "" to use NC_TOPO_MODEM

# Region source:
#   "model"  — use the extent of the resistivity grid (recommended)
#   "topo"   — use the extent of the topo grid (wider)
REGION_SOURCE    = "model"
REGION_MARGIN_KM = 0.0

# Explicit override of the map's displayed x/y range (UTM km), applied
# *after* REGION_SOURCE/REGION_MARGIN_KM compute the region above — crops
# (or expands) the displayed view without touching how that region is
# computed. Also feeds the feature-clipping (_in_region), the map figure's
# aspect ratio, and the lon/lat tick overlay, so all stay consistent with
# what's actually drawn. Set to None (default) to use the REGION_SOURCE
# extent unchanged. Analogous to the per-slice "xlim" in VSLICES below.
MAP_XLIM = [310.0, 445.0]    # e.g. [300.0, 420.0]  (easting,  km)
MAP_YLIM = [7971.6, 8120.5]  # e.g. [7960.0, 8080.0] (northing, km)

# Hillshade parameters
HS_AZIMUTH  = 315
HS_ALTITUDE = 45
HS_SIGMA    = 1.0   # Gaussian pre-smooth sigma (pixels); 0 = off

# Topo colour normalisation range (metres)
TOPO_VMIN = 1000
TOPO_VMAX = 6000

# Ocean fill colour
OCEAN_COLOR = "#6baed6"

# =====================================================================
# COLORBAR SETTINGS
# SHOW_COLORBAR      : False omits the colorbar entirely — the map panel
#                      itself is completely unaffected either way.
# COLORBAR_POSITION  : "right" | "left" | "bottom" | "top"
# The colorbar is placed in its own explicitly-sized axes, added as EXTRA
# width (right/left) or height (bottom/top) beyond the map panel — it
# never steals space from the map, so it can never distort its scale.
# COLORBAR_SIZE      : bar length, as a fraction (0-1) of the map edge
#                      it's attached to
# COLORBAR_ASPECT    : bar length / bar thickness (thickness is derived)
# =====================================================================
SHOW_COLORBAR       = True
COLORBAR_POSITION   = "right"   # "right" | "left" | "bottom" | "top"
COLORBAR_SIZE       = 0.85      # bar length, fraction of the map edge
COLORBAR_PAD        = 0.10      # inches
COLORBAR_ASPECT     = 20
COLORBAR_LABEL_SIZE = 8
COLORBAR_TICK_SIZE  = 7
COLORBAR_NTICKS     = 5

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
# FEATURE OVERLAY SETTINGS
# =====================================================================

# --- Profile lines (lon/lat endpoint pairs; set to [] to disable) ---
PROFILE_CD_LON = [-70.476, -69.499213]
PROFILE_CD_LAT = [-18.255, -17.0481]
PROFILE_2_LON  = [-69.670, -70.034]
PROFILE_2_LAT  = [-17.695, -17.267]

# --- North-arrow anchor (lon, lat) and shaft length (km) ---
ARROW_LON    = -73.6
ARROW_LAT    = -18.1
ARROW_LEN_KM = 4.0

# --- Seismicity CSV (space-delimited; columns x=lon, y=lat, z=depth km) ---
CSV_SEISMCAT = "../features/catalog_welllocated_15_simple5.csv"

# --- Volcanoes CSV ---
CSV_VOLCANES    = "../features/volcanes.csv"
VOLC_LABEL_IDX  = [5, 12, 13]   # row indices to label

# Volcano label text: full name vs. short/abbreviated name.
# VOLC_LABEL_FULL_NAME : True  -> use VOLC_NAME_COL_FULL
#                        False -> use VOLC_NAME_COL_SHORT (default)
VOLC_LABEL_FULL_NAME = False
VOLC_NAME_COL_FULL   = "NAME"      # column used when VOLC_LABEL_FULL_NAME=True
VOLC_NAME_COL_SHORT  = "VOLCAN2"   # column used when VOLC_LABEL_FULL_NAME=False

# --- Cities CSV (columns x=lon, y=lat, Name) ---
CSV_CITIES = "../features/cities.csv"

# =====================================================================
# MAP FEATURE LAYERS — simple on/off switches
# Each flag controls one overlay layer on the map. SHOW_SEISMICITY and
# SHOW_MT_SITES also control the matching projection onto vertical
# sections (VSLICE_EQ_STYLE / VSLICE_MT_STYLE), so turning a feature off
# applies everywhere it would otherwise appear, not just on the map.
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
# =====================================================================

PROFILE_CD_STYLE = dict(color="black", lw=0.4, zorder=10)
PROFILE_2_STYLE  = dict(color="gray",  lw=0.4, zorder=10)

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

# MT sites — loaded from modem_sites_utm.nc, no CSV needed
MT_MARKER_STYLE = dict(
    marker=MT_PIN_MARKER, s=16, facecolors="cyan", edgecolors="black",
    linewidths=0.5, alpha=0.85, zorder=12,
)

VOLC_INACT_MARKER_STYLE = dict(
    marker="^", s=16, facecolors="black", edgecolors="black",
    linewidths=0.2, zorder=13,
)
VOLC_LABEL_STYLE  = dict(fontsize=6, fontweight="bold", color="black",
                          zorder=14, offset_x=0.3, offset_y=0.3)

VOLC_ACT_MARKER_STYLE = dict(
    marker="^", s=16, facecolors="red", edgecolors="black",
    linewidths=0.2, zorder=13,
)

CITY_MARKER_STYLE = dict(
    marker="s", s=18, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=13,
)
CITY_LABEL_STYLE  = dict(fontsize=6, color="black", zorder=14,
                          offset_x=0.3, offset_y=-0.3)

ARROW_STYLE       = dict(color="dimgray", lw=2, mutation_scale=14)
ARROW_LABEL_STYLE = dict(fontsize=9, fontweight="bold", color="dimgray")

# =====================================================================
# VERTICAL SLICE SETTINGS
#
# VSLICES defines a list of arbitrary vertical cross-sections.
# Each entry is a dict with:
#
#   name    : str   — label used in title and output filename
#   p1      : [x, y] — first  endpoint, in UTM km OR lon/lat (see coord)
#   p2      : [x, y] — second endpoint, same convention
#   coord   : "utm"     — p1/p2 are [easting_km, northing_km]
#             "latlon"  — p1/p2 are [lon_deg, lat_deg]
#   zmin_km : float — top  of section (km, positive down; 0 = surface)
#   zmax_km : float — base of section (km, positive down)
#   npts    : int   — sample points along the profile
#   nz      : int   — depth levels (interpolated between model cells)
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
        name     = "profile_CD",
        p1       = [-70.476, -18.255],   # lon, lat
        p2       = [-69.499, -17.048],
        coord    = "latlon",
        # zmin_km must be negative enough to reach above sea level, or real
        # seismicity there (e.g. within a volcanic edifice) gets silently
        # excluded by the (zeqs >= zmin_km) filter in
        # _project_seismicity_to_profile — the catalogue used here has
        # events down to z = -5.75 km. -8.0 gives some margin.
        zmin_km  = -8.0,
        zmax_km  = 30.0,
        npts     = 200,
        nz       = 150,
        swath_km = 10.0,
    ),
    # Add further profiles here.
]

# Colour-scale limits for vertical slices (defaults to horizontal map limits)
VSLICE_CMIN_RHO = CMIN_RHO
VSLICE_CMAX_RHO = CMAX_RHO

# Vertical exaggeration (1 = true scale)
VSLICE_VE = 2.0

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

# How the 3-D model is sampled onto a section's profile points:
#   "nearest" — use the value of whichever mesh cell actually contains
#               each sample point, with no blending across cell
#               boundaries. Piecewise-constant, i.e. every colour in the
#               section is a real, unmodified cell value — a true cut
#               through the mesh's own cells, not a smoothed resampling.
#               (The along-profile *sample spacing* — VSLICES' npts/nz —
#               is still a regular grid, not the mesh's own irregular cell
#               boundaries, so this isn't a full geometric mesh-polygon
#               cut for an arbitrary-angle profile; but the values plotted
#               are exact, unblended cell values, which is the part that
#               actually matters visually.)
#   "linear"  — smooth trilinear interpolation between neighbouring cells
#               (the previous default) — nicer-looking but can visually
#               blur sharp resistivity/sensitivity contrasts across real
#               cell boundaries, and can imply resolution the mesh doesn't
#               have.
# The section's pcolormesh shading follows this automatically: "nearest"
# draws flat, undecorated cell blocks; "linear" uses "gouraud" (smooth
# per-vertex shading) so the rendering matches how the values were
# produced either way.
VSLICE_INTERP_METHOD = "nearest"

# --- Free-text annotation (optional) ---
# Draws one extra line of arbitrary text on every figure this script
# produces (both depth slices and vertical sections) — e.g. a version tag,
# a processing note, or a "DRAFT" watermark. Set to None or "" to disable.
# Default position is top-left; the VE label now sits lower-right on
# sections (see VSLICE_VE_POS above), so the two no longer collide.
ANNOTATION_TEXT  = None                  # e.g. "Preliminary — v3 mesh"
ANNOTATION_POS   = (0.01, 0.99)          # (x, y) in axes-fraction coords
ANNOTATION_STYLE = dict(fontsize=7, color="gray", ha="left", va="top")

# Horizontal axis for vertical sections:
#   "utm"      — UTM easting or northing (km)
#   "distance" — cumulative distance from p1 (km)
VSLICE_X_AXIS = "utm"

# Seismicity marker style on cross-section
VSLICE_EQ_STYLE = dict(
    s=2, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=11,
)

# MT site marker style on cross-section (projected within swath)
VSLICE_MT_STYLE = dict(
    marker=MT_PIN_MARKER, s=16, facecolors="cyan", edgecolors="black",
    linewidths=0.5, zorder=12,
)

# Topographic surface line style
VSLICE_TOPO_STYLE = dict(color="dimgray", lw=0.5, zorder=12)

# Fill colours for the topography band above the section
VSLICE_TOPO_LAND_COLOR  = "gray"    # z > 0 (above sea level)
VSLICE_TOPO_OCEAN_COLOR = "#6baed6" # z <= 0 (below sea level)

# Extra headroom above the highest topographic point (km).
VSLICE_TOPO_HEADROOM_KM = 1.0

# Style of the profile line drawn on the map figures
VSLICE_MAP_LINE_STYLE = dict(color="magenta", lw=0.8, ls="--", zorder=15)

# =====================================================================
# END USER SETTINGS
# =====================================================================

os.makedirs(PLOT_DIR, exist_ok=True)


def ncpath(name):
    """Join a bare precomputed-NetCDF filename onto NC_DIR."""
    return os.path.join(NC_DIR, name)


# ------------------------------------------------------------------
# Coordinate helper / hillshade / sensitivity-alpha helpers — see
# plotpy for implementation
# ------------------------------------------------------------------
to_utm_km = plotpy.to_utm_km
compute_hillshade = plotpy.compute_hillshade
sens_shade_alpha = plotpy.sens_shade_alpha
sens_data_alpha = plotpy.sens_data_alpha


def draw_sens_shade_overlay(ax, vx, vy, alpha_2d, zorder,
                            e_edges=None, n_edges=None):
    """Draw a solid-colour overlay (SENS_SHADE_COLOR) whose per-cell alpha
    comes from alpha_2d, to visually de-emphasise poorly-resolved cells.
    Uses pcolormesh with the true (non-uniform) ModEM grid rather than
    imshow+extent, for the same reason the resistivity raster does — see
    the note by the resistivity pcolormesh call above. Uses exact cell
    edges (shading="flat") when e_edges/n_edges are supplied and match
    vx/vy in size, else falls back to centre-based shading="nearest"."""
    rgb = mcolors.to_rgb(SENS_SHADE_COLOR)
    shade_cmap = mcolors.ListedColormap([rgb])
    if (e_edges is not None and n_edges is not None and
            e_edges.size == vx.size + 1 and n_edges.size == vy.size + 1):
        ax.pcolormesh(
            e_edges, n_edges, np.zeros_like(alpha_2d),
            cmap=shade_cmap, vmin=0, vmax=1, shading="flat",
            alpha=alpha_2d, zorder=zorder,
        )
    else:
        ax.pcolormesh(
            vx, vy, np.zeros_like(alpha_2d),
            cmap=shade_cmap, vmin=0, vmax=1, shading="nearest",
            alpha=alpha_2d, zorder=zorder,
        )


def load_sens_depth_slice(tag, ref_shape, ref_northing, ref_easting):
    """
    Load modem_sens_utm_{tag}.nc for the horizontal-slice loop, re-oriented
    to match the resistivity slice's own (northing, easting) orientation.
    Returns None if sensitivity is disabled or the file doesn't exist.
    """
    if not USE_SENSITIVITY:
        return None
    path = ncpath(f"modem_sens_utm_{tag}.nc")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — sensitivity masking/shading "
              f"is disabled for this depth slice. Check that "
              f"tacna_precompute_modem.py found the .sns file (look for its "
              f"own WARNING) and that OUTPUT_DIR there matches NC_DIR here.")
        return None
    _da = xr.open_dataarray(path)
    sy = _da["northing"].values
    sx = _da["easting"].values
    sv = _da.values.copy().astype(float)
    _da.close()

    if sv.shape[0] != len(sy):
        sv = sv.T
    if sy[0] > sy[-1]:
        sy = sy[::-1]
        sv = sv[::-1, :]
    if sx[0] > sx[-1]:
        sx = sx[::-1]
        sv = sv[:, ::-1]

    if sv.shape != ref_shape or not (np.allclose(sy, ref_northing) and
                                     np.allclose(sx, ref_easting)):
        print(f"  WARNING: {path} grid doesn't match the resistivity slice "
              f"— skipping shading/blanking for this depth.")
        return None
    return sv


# ------------------------------------------------------------------
# Save helper
# ------------------------------------------------------------------
def save_fig(fig, stem):
    for fmt in PLOT_FORMATS:
        out = os.path.join(PLOT_DIR, stem + PLOT_FILENAME_SUFFIX + fmt)
        fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {out}")


def draw_annotation(ax):
    """Draw the optional free-text annotation (ANNOTATION_TEXT), if set."""
    plotpy.draw_annotation(ax, ANNOTATION_TEXT, ANNOTATION_POS, ANNOTATION_STYLE)


def _region():
    return (xmin, xmax, ymin, ymax)


def _colorbar_settings():
    return dict(show=SHOW_COLORBAR, position=COLORBAR_POSITION,
                size=COLORBAR_SIZE, pad=COLORBAR_PAD, aspect=COLORBAR_ASPECT,
                label_size=COLORBAR_LABEL_SIZE, tick_size=COLORBAR_TICK_SIZE,
                nticks=COLORBAR_NTICKS)


_resolve_ve_pos = plotpy.resolve_ve_pos


def _in_region(xe, yn):
    """Boolean mask: True where (xe, yn) fall inside the map region."""
    return plotpy.in_region(xe, yn, _region())


def clipped_scatter(ax, xe, yn, **kwargs):
    """scatter() restricted to points inside the map region."""
    plotpy.clipped_scatter(ax, xe, yn, _region(), **kwargs)


def clipped_labels(ax, xe, yn, labels, style_dict):
    """Draw text labels for points inside the map region — see
    plotpy.clipped_labels for the full docstring."""
    plotpy.clipped_labels(ax, xe, yn, labels, style_dict, _region())


def draw_north_arrow(ax, x_km, y_km, length_km=4.0):
    """Draw a north arrow at UTM position (x_km, y_km) if inside region."""
    plotpy.draw_north_arrow(ax, x_km, y_km, _region(),
                             ARROW_STYLE, ARROW_LABEL_STYLE, length_km)


# ------------------------------------------------------------------
# Map/section figure creation — guarantees equal x/y (km) scale BY
# CONSTRUCTION; see plotpy.build_panel_figure for the full docstring.
# ------------------------------------------------------------------
def create_map_figure():
    map_w_in = FIG_WIDTH / 2.54
    map_h_in = map_w_in * (ymax - ymin) / (xmax - xmin)
    return plotpy.build_panel_figure(map_w_in, map_h_in, _colorbar_settings(),
                                      size_label="map")


def create_section_figure(w_in, h_in):
    return plotpy.build_panel_figure(w_in, h_in, _colorbar_settings(),
                                      size_label="section")


def finish_panel_colorbar(cax, mappable, label):
    """Render the colorbar into the cax returned by create_map_figure()
    or create_section_figure()."""
    return plotpy.finish_panel_colorbar(cax, mappable, label, _colorbar_settings())

# ------------------------------------------------------------------
# Secondary lon/lat axes  (cosmetic overlay on UTM-km plot)
# ------------------------------------------------------------------
def add_latlon_ticks(ax):
    """Replace UTM-km tick labels with lon/lat values — see
    plotpy.add_latlon_ticks for the full docstring. Controlled by
    AXES_UNITS, LATLON_NTICKS, LATLON_DECIMALS."""
    plotpy.add_latlon_ticks(ax, _region(), LATLON_NTICKS, LATLON_DECIMALS,
                             COLORBAR_LABEL_SIZE, COLORBAR_TICK_SIZE)


# ==================================================================
# Vertical slice engine
# ==================================================================

_profile_utm_km = plotpy.profile_utm_km
_profile_labels = plotpy.profile_labels
_sample_profile_points = plotpy.sample_profile_points


def compute_vertical_slice_modem(vslice):
    """
    Sample the ModEM resistivity model (modem_model_utm.nc) along a
    vertical profile.

    modem_model_utm.nc has dims (depth, northing, easting) in UTM km —
    no coordinate reprojection is needed; the interpolator works directly
    in UTM km space.

    Returns
    -------
    dist_km    : 1-D (npts,)   distance along profile (km)
    depth_km   : 1-D (nz,)     depth axis (km, positive down)
    section    : 2-D (nz, npts) interpolated log10(ρ)
    e_ends     : 1-D [e0, e1]  UTM km, for map overlay
    n_ends     : 1-D [n0, n1]  UTM km
    surf_depth : 1-D (npts,)   depth (km) of the shallowest valid (non-air)
                 cell in `section` at each profile position — this is the
                 model's *own* surface, guaranteed consistent with the
                 colour data (see note below).
    topo_prof  : 1-D (npts,)   surface elevation (m) from modem_topo_utm.nc,
                 or None — kept only for the land/ocean colour distinction,
                 not for positioning the surface line (see note below).
    sens_section : 2-D (nz, npts) interpolated sensitivity field, or None
                 if USE_SENSITIVITY is False or modem_sens_utm.nc is
                 missing — used by plot_vertical_slice for shading.
    """
    e_ends, n_ends = _profile_utm_km(vslice)
    npts  = vslice.get("npts", 200)
    nz    = vslice.get("nz",   150)
    zmin  = vslice.get("zmin_km", 0.0)
    zmax  = vslice.get("zmax_km", 60.0)

    dist_km, e_pts, n_pts, utm_x, utm_xlabel = \
        _sample_profile_points(e_ends, n_ends, npts)

    # Load 3-D model (dims: depth, northing, easting)
    _da  = xr.open_dataarray(ncpath("modem_model_utm.nc"))
    e_ax = _da["easting"].values    # km
    n_ax = _da["northing"].values   # km
    d_ax = _da["depth"].values      # km
    vals = _da.values               # (ndepth, nnorthing, neasting)
    _da.close()

    # Ensure all axes strictly ascending for RegularGridInterpolator
    if n_ax[0] > n_ax[-1]:
        n_ax = n_ax[::-1];  vals = vals[:, ::-1, :]
    if e_ax[0] > e_ax[-1]:
        e_ax = e_ax[::-1];  vals = vals[:, :, ::-1]

    interp = RegularGridInterpolator(
        (d_ax, n_ax, e_ax), vals,
        method=VSLICE_INTERP_METHOD, bounds_error=False, fill_value=np.nan,
    )

    depth_km = np.linspace(zmin, zmax, nz)
    d_q = np.repeat(depth_km, npts)
    n_q = np.tile(n_pts, nz)
    e_q = np.tile(e_pts, nz)
    section = interp(np.column_stack([d_q, n_q, e_q])).reshape(nz, npts)

    # Mask air cells (log10(ρ) ≈ 17 for RHO_AIR) so they show as no-data
    # rather than a saturated colour — the horizontal depth slices already
    # do this via colour-range clipping; the section used a raw interpolated
    # value with nothing removing the air layer above the topography.
    section[section >= AIR_LOG10_RHO_THRESHOLD] = np.nan

    # Surface derived directly from the section's own air/rock mask, i.e.
    # the shallowest non-NaN depth in each column. This is the model's own
    # depth-axis reference by construction, so it always sits exactly where
    # the colour data starts.
    #
    # We previously derived the surface line from modem_topo_utm.nc's
    # elevation (via get_topo, referenced to the .rho file's reference-point
    # elevation) and assumed depth_km's own 0 was sea level. But
    # build_depth_axis_km() (precompute) rebases the model's own top face to
    # depth=0 regardless of its true elevation — if that top face sits well
    # above the reference point (as it typically does, to leave air padding
    # above the tallest terrain), the two "zero" reference points don't
    # coincide. That mismatch was exactly the floating topo line and the
    # large spurious no-data gap: the line was drawn using an elevation
    # value that didn't correspond to depth_km's own datum. Deriving the
    # surface from `section` itself sidesteps the whole reference-frame
    # question — it's self-consistent by definition.
    #
    # IMPORTANT: this must be computed from the *air* mask only, before any
    # sensitivity-based blanking below — the physical surface shouldn't
    # move just because a well-resolved cell happens to have low
    # sensitivity coverage from the .sns file.
    valid = ~np.isnan(section)
    has_data = valid.any(axis=0)
    first_valid_idx = np.argmax(valid, axis=0)
    surf_depth = np.full(npts, depth_km[-1])
    surf_depth[has_data] = depth_km[first_valid_idx[has_data]]

    # Sensitivity-based blanking/shading (see USE_SENSITIVITY etc.). Sampled
    # at the exact same query points as `section` above, so the two stay
    # pixel-aligned. Applied only to the *displayed* section, after
    # surf_depth has already been fixed from the air mask.
    sens_section = None
    if USE_SENSITIVITY and not os.path.exists(ncpath("modem_sens_utm.nc")):
        print("  WARNING: modem_sens_utm.nc not found — sensitivity "
              "masking/shading is disabled for this section. Check that "
              "tacna_precompute_modem.py found the .sns file (look for its "
              "own WARNING) and that OUTPUT_DIR there matches NC_DIR here.")
    if USE_SENSITIVITY and os.path.exists(ncpath("modem_sens_utm.nc")):
        _sda  = xr.open_dataarray(ncpath("modem_sens_utm.nc"))
        se_ax = _sda["easting"].values
        sn_ax = _sda["northing"].values
        sd_ax = _sda["depth"].values
        svals = _sda.values
        _sda.close()

        print(' sens min = ', np.amin(svals))

        if sn_ax[0] > sn_ax[-1]:
            sn_ax = sn_ax[::-1]; svals = svals[:, ::-1, :]
        if se_ax[0] > se_ax[-1]:
            se_ax = se_ax[::-1]; svals = svals[:, :, ::-1]

        sens_interp = RegularGridInterpolator(
            (sd_ax, sn_ax, se_ax), svals,
            method=VSLICE_INTERP_METHOD, bounds_error=False, fill_value=np.nan,
        )
        sens_section = sens_interp(
            np.column_stack([d_q, n_q, e_q])).reshape(nz, npts)

        if SENS_BLANK_THRESHOLD is not None:
            _blank_mask = sens_section < SENS_BLANK_THRESHOLD
            print(f"  sens blanking: {np.sum(_blank_mask)}/{_blank_mask.size} "
                  f"cells below threshold ({SENS_BLANK_THRESHOLD})")
            section = np.where(_blank_mask, np.nan, section)

    # Topo along profile — kept only to flag ocean (elevation <= 0) for the
    # bathymetry fill colour; NOT used to position the surface line/MT
    # sites/seismicity any more (see surf_depth above).
    topo_prof = None
    if modem_topo_z is not None:
        topo_interp = RegularGridInterpolator(
            (modem_topo_y, modem_topo_x), modem_topo_z,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        topo_prof = topo_interp(np.column_stack([n_pts, e_pts]))

    return (dist_km, depth_km, section, e_ends, n_ends, surf_depth, topo_prof,
            sens_section, utm_x, utm_xlabel)


def _project_seismicity_to_profile(e_ends, n_ends, swath_km, zmin_km, zmax_km):
    """Events within swath_km of the profile and within depth range;
    returns (along_km, depth_km)."""
    return plotpy.project_points_to_profile(
        eq_e0, eq_n0, e_ends, n_ends, swath_km, z0=zeqs,
        zmin_km=zmin_km, zmax_km=zmax_km)


def _project_mt_sites_to_profile(e_ends, n_ends, swath_km):
    """MT sites within swath_km; returns along-profile distance (km).

    Depth is not returned here — MT sites sit on the true surface, not at a
    fixed z=0, so their plotted depth is taken from the section's own
    surf_depth (interpolated at each site's along-profile position) by the
    caller instead.
    """
    return plotpy.project_points_to_profile(mt_e, mt_n, e_ends, n_ends, swath_km)


def plot_vertical_slice(dist_km, depth_km, section, e_ends, n_ends,
                        surf_depth, topo_prof, sens_section, utm_x, utm_xlabel,
                        lbl_start, lbl_end,
                        vslice, cmap, cmin, cmax, cbar_label, stem):
    """Produce and save a vertical cross-section figure (UTM km or distance vs depth)."""
    swath  = vslice.get("swath_km", 10.0)
    zmin_s = vslice.get("zmin_km", depth_km[0])
    zmax_s = vslice.get("zmax_km", depth_km[-1])
    ve     = 1.0 if VSLICE_EQUAL_SCALE else VSLICE_VE
    name   = vslice.get("name", "profile")

    # Choose horizontal coordinate
    if VSLICE_X_AXIS == "distance":
        x_arr   = dist_km
        x_label = "Distance along profile (km)"
    else:
        x_arr   = utm_x
        x_label = utm_xlabel

    # x_arr must be ascending for np.interp below regardless of profile
    # direction (northing/easting can decrease along the profile).
    if x_arr[0] > x_arr[-1]:
        x_arr_asc, surf_depth_asc = x_arr[::-1], surf_depth[::-1]
    else:
        x_arr_asc, surf_depth_asc = x_arr, surf_depth

    eq_dist, eq_dep = _project_seismicity_to_profile(
        e_ends, n_ends, swath, zmin_s, zmax_s)
    mt_dist = _project_mt_sites_to_profile(e_ends, n_ends, swath)

    eq_x = np.interp(eq_dist, dist_km, x_arr) if len(eq_dist) else eq_dist
    mt_x = np.interp(mt_dist, dist_km, x_arr) if len(mt_dist) else mt_dist

    # MT sites sit on the true surface, not at a fixed z=0 — take their
    # plotted depth from the section's own surface at that position.
    mt_dep = np.interp(mt_x, x_arr_asc, surf_depth_asc) if len(mt_x) else mt_x

    # An event can only be real if it's at or below the local surface.
    # Events above it here are a projection artefact (the swath can be wide
    # enough that an event's true, off-profile position has different local
    # relief than the profile line itself) rather than physically "in the
    # air" — but drawing them above the surface line is misleading either
    # way, so they're dropped from this particular figure.
    if len(eq_x):
        local_surf_at_eq = np.interp(eq_x, x_arr_asc, surf_depth_asc)
        keep = eq_dep >= local_surf_at_eq
        eq_x, eq_dep = eq_x[keep], eq_dep[keep]

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
    # Per-cell alpha for the data layer itself: constant (1-ALPHA_RHO) as
    # before, unless SENS_ALPHA_RANGE is set — then it fades toward fully
    # transparent in poorly-resolved cells, letting whatever is drawn
    # underneath (the topo fill/line) show through instead of being
    # covered by data.
    data_alpha = 1.0 - ALPHA_RHO
    if sens_section is not None and SENS_ALPHA_RANGE is not None:
        data_alpha = sens_data_alpha(sens_section, SENS_ALPHA_RANGE[0],
                                     SENS_ALPHA_RANGE[1], 1.0 - ALPHA_RHO)
    # Shading follows how the data was sampled: "nearest" (piecewise-
    # constant, unblended real cell values — see VSLICE_INTERP_METHOD)
    # gets flat, undecorated blocks so the rendering doesn't re-introduce
    # smoothing the sampling deliberately avoided; "linear" (smoothed
    # trilinear interpolation) keeps the previous "gouraud" look, which
    # matches how those values were produced.
    _section_shading = "nearest" if VSLICE_INTERP_METHOD == "nearest" else "gouraud"
    im = ax.pcolormesh(
        x_arr, depth_km, section,
        cmap=cmap, norm=norm, shading=_section_shading,
        alpha=data_alpha, zorder=5,
        antialiased=(_section_shading != "nearest"),
    )
    # NOTE: previously used shading="auto" + set_rasterized(True) to fix
    # vector-format seams between cells, but rasterized artists render
    # upside-down under matplotlib's PDF/PS backends when combined with
    # ax.invert_yaxis() (used below for the depth axis) — a known
    # matplotlib bug, not something in this pipeline. Gouraud shading
    # (smooth per-vertex interpolation, no discrete cell polygons) removes
    # the seams without rasterizing anything, so the inverted axis renders
    # correctly. When VSLICE_INTERP_METHOD="nearest" reintroduces flat
    # shading (deliberately, for a true unblended cut), antialiased=False
    # is used instead to suppress the same hairline-seam artifact — this
    # doesn't rasterize anything, so the invert_yaxis bug doesn't apply.

    if sens_section is not None and SENS_SHADE_RANGE is not None:
        alpha_2d = sens_shade_alpha(sens_section, SENS_SHADE_RANGE[0],
                                    SENS_SHADE_RANGE[1], SENS_SHADE_MAX_ALPHA)
        rgb = mcolors.to_rgb(SENS_SHADE_COLOR)
        shade_cmap = mcolors.ListedColormap([rgb])
        # pcolormesh accepts a per-cell alpha array (not just a scalar),
        # so this stays on the exact same (x_arr, depth_km) grid as the
        # main data — no separate imshow/extent/origin bookkeeping needed.
        ax.pcolormesh(
            x_arr, depth_km, np.zeros_like(alpha_2d),
            cmap=shade_cmap, vmin=0, vmax=1, shading=_section_shading,
            alpha=alpha_2d, zorder=6,
            antialiased=(_section_shading != "nearest"),
        )

    # Surface line/fill now come from the section's own data (surf_depth),
    # not a separately-referenced topography raster — see the note in
    # compute_vertical_slice_modem for why that mismatch was producing a
    # floating topo line and a large spurious no-data gap.
    y_top = min(depth_km[0], surf_depth.min() - VSLICE_TOPO_HEADROOM_KM)

    if topo_prof is not None:
        # Ocean fill only (a genuine physical reference — elevation <= 0 —
        # not a masked/no-data region), positioned using surf_depth so it
        # never floats apart from the colour data.
        ocean = topo_prof <= 0
        if ocean.any():
            ax.fill_between(x_arr, 0.0, surf_depth,
                            where=ocean,
                            color=VSLICE_TOPO_OCEAN_COLOR, alpha=0.5,
                            zorder=6, interpolate=True)

    ax.plot(x_arr, surf_depth, **VSLICE_TOPO_STYLE)

    if SHOW_SEISMICITY and len(eq_x):
        ax.scatter(eq_x, eq_dep, **VSLICE_EQ_STYLE)
    if SHOW_MT_SITES and len(mt_x):
        ax.scatter(mt_x, mt_dep, **VSLICE_MT_STYLE)

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

    # Endpoint labels at the top of the section
    for xpos, lbl in ((x0, lbl_start), (x1, lbl_end)):
        ax.text(xpos, y_top, lbl,
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="black",
                clip_on=False, zorder=20)

    ax.set_title(f"log$_{{10}}$ρ — {name}  (swath ±{swath} km)", fontsize=9)

    finish_panel_colorbar(cax, im, cbar_label)
    draw_annotation(ax)
    save_fig(fig, stem)
    plt.close(fig)

# --- Topography ---
# Prefer the higher-resolution seis-pipeline topo if available
_topo_path = ncpath(NC_TOPO_SEIS) if (NC_TOPO_SEIS and os.path.exists(ncpath(NC_TOPO_SEIS))) \
             else ncpath(NC_TOPO_MODEM)
print(f"Loading topography from: {_topo_path}")
_topo_da = xr.open_dataarray(_topo_path)

# Both seis (dims x/y) and modem (dims easting/northing) grids are supported
if "x" in _topo_da.dims:
    topo_x = _topo_da["x"].values       # easting km
    topo_y = _topo_da["y"].values       # northing km
elif "easting" in _topo_da.dims:
    topo_x = _topo_da["easting"].values
    topo_y = _topo_da["northing"].values
else:
    raise ValueError(f"Cannot identify spatial dims in {_topo_path}")

topo_z = _topo_da.values               # (ny, nx) or (northing, easting)
_topo_da.close()

# Ensure orientation is (northing, easting) with northing increasing upward
if topo_z.shape[0] != len(topo_y):
    topo_z = topo_z.T
if topo_y[0] > topo_y[-1]:
    topo_y = topo_y[::-1]
    topo_z = topo_z[::-1, :]

dx_km = float(np.median(np.diff(topo_x)))
dy_km = float(np.median(np.diff(topo_y)))

print("Computing hillshade …")
topo_hs = compute_hillshade(topo_z, dx_km, dy_km, HS_AZIMUTH, HS_ALTITUDE, HS_SIGMA)
topo_extent = [topo_x.min(), topo_x.max(), topo_y.min(), topo_y.max()]
topo_norm   = mcolors.Normalize(vmin=TOPO_VMIN, vmax=TOPO_VMAX)

# --- ModEM's own topography (for vertical-section masking only) ---
# The vertical-section surface mask must line up with wherever mval itself
# switches from air to rock — that boundary is only guaranteed to match
# modem_topo_utm.nc (extracted directly from the model). NC_TOPO_SEIS above
# may be a different, unrelated DEM used purely to make the map basemap
# look nicer; reusing it for the section mask left thin unmasked slivers of
# real resistivity poking through wherever the two surfaces disagreed.
_modem_topo_da = xr.open_dataarray(ncpath(NC_TOPO_MODEM))
if "easting" in _modem_topo_da.dims:
    modem_topo_x = _modem_topo_da["easting"].values
    modem_topo_y = _modem_topo_da["northing"].values
else:
    modem_topo_x = _modem_topo_da["x"].values
    modem_topo_y = _modem_topo_da["y"].values
modem_topo_z = _modem_topo_da.values
_modem_topo_da.close()

if modem_topo_z.shape[0] != len(modem_topo_y):
    modem_topo_z = modem_topo_z.T
if modem_topo_y[0] > modem_topo_y[-1]:
    modem_topo_y = modem_topo_y[::-1]
    modem_topo_z = modem_topo_z[::-1, :]
CMAP_TOPO   = plt.get_cmap("gray")

# --- Bathymetry (optional) ---
_use_bath = bool(NC_BATH and os.path.exists(ncpath(NC_BATH)))
if _use_bath:
    print(f"Loading bathymetry from: {ncpath(NC_BATH)}")
    _bath_da = xr.open_dataarray(ncpath(NC_BATH))
    bath_x = _bath_da["x"].values if "x" in _bath_da.dims else _bath_da["easting"].values
    bath_y = _bath_da["y"].values if "y" in _bath_da.dims else _bath_da["northing"].values
    bath_z = _bath_da.values
    _bath_da.close()
    if bath_z.shape[0] != len(bath_y):
        bath_z = bath_z.T
    bath_extent = [bath_x.min(), bath_x.max(), bath_y.min(), bath_y.max()]
else:
    print("Bathymetry file not found — ocean fill skipped.")

# --- MT site positions from NetCDF ---
print(f"Loading MT sites from: {ncpath(NC_SITES)}")
_sites_ds = xr.open_dataset(ncpath(NC_SITES))
mt_e      = _sites_ds["easting"].values
mt_n      = _sites_ds["northing"].values
mt_names  = _sites_ds["name"].values.tolist()
_sites_ds.close()

# ==================================================================
# Map region
# ==================================================================
# Derive region from first depth-slice grid
_tag0 = f"{DEPTH_SLICES_KM[0]:.0f}km" \
        if DEPTH_SLICES_KM[0] == int(DEPTH_SLICES_KM[0]) \
        else f"{DEPTH_SLICES_KM[0]:.1f}km"
_nc0 = ncpath(f"modem_rho_utm_{_tag0}.nc")
_da0 = xr.open_dataarray(_nc0)
_e0  = _da0["easting"].values
_n0  = _da0["northing"].values
_da0.close()

if REGION_SOURCE == "model":
    utm_region = [
        float(_e0.min()) - REGION_MARGIN_KM,
        float(_e0.max()) + REGION_MARGIN_KM,
        float(_n0.min()) - REGION_MARGIN_KM,
        float(_n0.max()) + REGION_MARGIN_KM,
    ]
    print("Region source: resistivity grid")
else:
    utm_region = [topo_x.min(), topo_x.max(), topo_y.min(), topo_y.max()]
    print("Region source: topo grid")

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
# Feature layers (CSV-based)
# ==================================================================
volcanes = pd.read_csv(CSV_VOLCANES)
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

eqs  = pd.read_csv(CSV_SEISMCAT, delimiter=" ")
eq_e0, eq_n0 = to_utm_km(eqs["x"].values, eqs["y"].values)
zeqs = eqs["z"].values

cities   = pd.read_csv(CSV_CITIES)
cit_e, cit_n = to_utm_km(cities["x"].values, cities["y"].values)
name_cit = cities["Name"].values

prof_cd_e, prof_cd_n = to_utm_km(PROFILE_CD_LON, PROFILE_CD_LAT) \
    if PROFILE_CD_LON else ([], [])
prof2_e, prof2_n = to_utm_km(PROFILE_2_LON, PROFILE_2_LAT) \
    if PROFILE_2_LON else ([], [])
arr_e, arr_n = to_utm_km([ARROW_LON], [ARROW_LAT])


# ==================================================================
# Basemap and feature drawing
# ==================================================================
def draw_basemap(ax):
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
    if _use_bath:
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
    if SHOW_PROFILE_LINES:
        if len(prof_cd_e):
            ax.plot(prof_cd_e, prof_cd_n, clip_on=True, **PROFILE_CD_STYLE)
        if len(prof2_e):
            ax.plot(prof2_e, prof2_n, clip_on=True, **PROFILE_2_STYLE)

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

    if SHOW_SEISMICITY:
        clipped_scatter(ax, eq_e, eq_n, label="Seismicity", **EQ_MARKER_STYLE)

    # MT sites from NetCDF (already in UTM km)
    if SHOW_MT_SITES:
        clipped_scatter(ax, mt_e, mt_n, label="MT site", **MT_MARKER_STYLE)

    if SHOW_VOLCANOES:
        clipped_scatter(ax, utmv_e, utmv_n, **VOLC_INACT_MARKER_STYLE)
        clipped_labels(ax, utmv_e, utmv_n, namev, VOLC_LABEL_STYLE)

    if SHOW_VOLCANOES_ACTIVE and volc_act_e:
        clipped_scatter(ax, volc_act_e, volc_act_n,
                        label="Active volcano", **VOLC_ACT_MARKER_STYLE)

    if SHOW_CITIES:
        clipped_scatter(ax, cit_e, cit_n, label="City", **CITY_MARKER_STYLE)
        clipped_labels(ax, cit_e, cit_n, name_cit, CITY_LABEL_STYLE)

    if SHOW_NORTH_ARROW:
        draw_north_arrow(ax, arr_e[0], arr_n[0], length_km=ARROW_LEN_KM)


# ==================================================================
# Main loop
# ==================================================================
import matplotlib.ticker   # noqa: E402 — needed for add_colorbar above

# True (non-uniform) ModEM cell-edge coordinates, if tacna_precompute_modem.py
# produced them (modem_grid_edges_utm.nc). These let the depth-slice raster
# be drawn as an exact, non-interpolated cut through the mesh's actual cells
# (pcolormesh + shading="flat") instead of being resampled onto a uniform
# pixel grid. Falls back to the approximate shading="nearest" (cell centres
# only, edges reconstructed as midpoints) if the file isn't there yet — e.g.
# output from an older precompute run.
HAVE_GRID_EDGES = os.path.exists(ncpath("modem_grid_edges_utm.nc"))
if HAVE_GRID_EDGES:
    _edges_da = xr.open_dataset(ncpath("modem_grid_edges_utm.nc"))
    grid_e_edges = _edges_da["easting_edges"].values
    grid_n_edges = _edges_da["northing_edges"].values
    _edges_da.close()
    print("Using exact cell-edge geometry from modem_grid_edges_utm.nc "
          "for depth-slice rendering.")
else:
    grid_e_edges = grid_n_edges = None
    print("WARNING: modem_grid_edges_utm.nc not found — falling back to "
          "approximate cell boundaries (shading='nearest') for depth "
          "slices. Re-run tacna_precompute_modem.py to get exact cell "
          "edges.")

out_list = []

for ii, d_km in enumerate(DEPTH_SLICES_KM):
    tag   = f"{d_km:.0f}km" if d_km == int(d_km) else f"{d_km:.1f}km"
    nc    = ncpath(f"modem_rho_utm_{tag}.nc")
    label = f"{d_km:.0f} km" if d_km == int(d_km) else f"{d_km:.1f} km"

    print(f"Plotting log10(ρ) at {label} …")
    _da = xr.open_dataarray(nc)
    vx  = _da["easting"].values
    vy  = _da["northing"].values
    vz  = _da.values.copy().astype(float)
    _da.close()

    # Ensure (northing, easting) orientation with northing increasing upward
    if vz.shape[0] != len(vy):
        vz = vz.T
    if vy[0] > vy[-1]:
        vy = vy[::-1]
        vz = vz[::-1, :]

    # Mask true air/no-data cells only (see AIR_LOG10_RHO_THRESHOLD),
    # independent of the display colour range. Previously this clipped to
    # [CMIN_RHO, CMAX_RHO] and then NaN'd anything at either boundary —
    # which also hid genuinely resistive volcanic rock (fresh, unaltered
    # edifice material routinely exceeds CMAX_RHO) as if it were air.
    # imshow + Normalize already clamp in-range display to the colour
    # extremes for anything outside [CMIN_RHO, CMAX_RHO], so no manual
    # clip is needed here.
    vz[vz >= AIR_LOG10_RHO_THRESHOLD] = np.nan

    # Sensitivity-based blanking/shading (see USE_SENSITIVITY etc.)
    sens_vz = load_sens_depth_slice(tag, vz.shape, vy, vx)
    if sens_vz is not None:
        if SENS_BLANK_THRESHOLD is not None:
            _blank_mask = sens_vz < SENS_BLANK_THRESHOLD
            print(f"  sens blanking: {np.sum(_blank_mask)}/{_blank_mask.size} "
                  f"cells below threshold ({SENS_BLANK_THRESHOLD})")
            vz = np.where(_blank_mask, np.nan, vz)

    # Seismicity depth filter
    zmin = ZMIN_SEISM[ii] if ZMIN_SEISM[ii] is not None else -np.inf
    zmax = ZMAX_SEISM[ii] if ZMAX_SEISM[ii] is not None else  np.inf
    mask_eqs = (zeqs > zmin) & (zeqs < zmax)
    eq_e = eq_e0[mask_eqs]
    eq_n = eq_n0[mask_eqs]

    fig, ax, cax = create_map_figure()
    draw_basemap(ax)

    norm = mcolors.Normalize(vmin=CMIN_RHO, vmax=CMAX_RHO)
    # Per-cell alpha for the data layer itself: constant (1-ALPHA_RHO) as
    # before, unless SENS_ALPHA_RANGE is set — then it fades toward fully
    # transparent in poorly-resolved cells, letting the topography
    # basemap underneath show through instead of being covered by data.
    data_alpha = 1.0 - ALPHA_RHO
    if sens_vz is not None and SENS_ALPHA_RANGE is not None:
        data_alpha = sens_data_alpha(sens_vz, SENS_ALPHA_RANGE[0],
                                     SENS_ALPHA_RANGE[1], 1.0 - ALPHA_RHO)
    edges_ok = (HAVE_GRID_EDGES and
                grid_e_edges.size == vx.size + 1 and
                grid_n_edges.size == vy.size + 1)
    if edges_ok:
        im = ax.pcolormesh(
            grid_e_edges, grid_n_edges, vz, cmap=CMAP_RHO, norm=norm,
            alpha=data_alpha, shading="flat", zorder=5,
        )
    else:
        if HAVE_GRID_EDGES:
            print(f"  WARNING: modem_grid_edges_utm.nc size doesn't match "
                  f"this slice ({grid_e_edges.size-1}×{grid_n_edges.size-1} "
                  f"cells vs {vx.size}×{vy.size}) — falling back to "
                  f"approximate cell boundaries for this figure.")
        im = ax.pcolormesh(
            vx, vy, vz, cmap=CMAP_RHO, norm=norm,
            alpha=data_alpha, shading="nearest", zorder=5,
        )
    # NOTE: previously ax.imshow(vz, extent=[vx.min(), vx.max(), ...]).
    # imshow always assumes uniform pixel spacing across that extent, but
    # ModEM meshes are NOT uniform — dx/dy grow geometrically in the
    # padding cells outside the fine core region. Passing a non-uniform
    # grid's min/max as an imshow extent silently stretches/compresses
    # individual cells to fit a uniform pixel grid, which is exactly the
    # kind of "resistivity is shifted relative to topography" misalignment
    # this was producing — the topography raster (already on a uniform
    # grid) was positioned correctly, but the ModEM raster wasn't.
    # With modem_grid_edges_utm.nc available, shading="flat" against the
    # true cumulative cell edges gives an exact cut through the actual
    # mesh cells — no interpolation, no resampling, each rendered patch is
    # one real ModEM cell. Without it, shading="nearest" against cell
    # centres is used as an approximation (matplotlib reconstructs
    # boundaries as midpoints between centres, which is only exact where
    # neighbouring cells happen to be the same width).
    if sens_vz is not None and SENS_SHADE_RANGE is not None:
        alpha_2d = sens_shade_alpha(sens_vz, SENS_SHADE_RANGE[0],
                                    SENS_SHADE_RANGE[1], SENS_SHADE_MAX_ALPHA)
        draw_sens_shade_overlay(ax, vx, vy, alpha_2d, zorder=6,
                               e_edges=grid_e_edges, n_edges=grid_n_edges)
    draw_features(ax, eq_e, eq_n)
    ax.set_title(f"log$_{{10}}$ρ at {label}", fontsize=9)
    finish_panel_colorbar(cax, im, "log$_{10}$(ρ / Ω·m)")
    if AXES_UNITS == "latlon":
        add_latlon_ticks(ax)
    draw_annotation(ax)

    stem = f"modem_rho_{tag}_tacna"
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
    for vi, vslice in enumerate(VSLICES):
        name = vslice.get("name", "profile")
        print(f"  Computing section: {name} …")
        lbl_start, lbl_end = _profile_labels(vi)
        dist_km, depth_km, section, e_ends, n_ends, surf_depth, topo_prof, \
            sens_section, utm_x, utm_xlabel = compute_vertical_slice_modem(vslice)
        stem = f"modem_section_{name}_tacna"
        plot_vertical_slice(
            dist_km, depth_km, section, e_ends, n_ends,
            surf_depth, topo_prof, sens_section, utm_x, utm_xlabel,
            lbl_start, lbl_end, vslice,
            CMAP_RHO, VSLICE_CMIN_RHO, VSLICE_CMAX_RHO,
            "log$_{10}$(ρ / Ω·m)", stem,
        )
        out_list.append(stem)

    print("\nVertical slice stems:")
    for s in out_list:
        if "section" in s:
            print(f"  {s}")
