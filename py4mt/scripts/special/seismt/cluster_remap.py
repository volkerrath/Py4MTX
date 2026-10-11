#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster.py
=================
Clusters MT resistivity/conductivity + seismic tomography properties
(Vp, Vs, Vp/Vs, density) via fuzzy c-means or a self-organizing map
(SOM), and plots depth-slice maps. The grid clustered on can come from
EITHER of two upstream scripts (INPUT_KIND below):

  "interp" — interpolate.py's point-cloud RBF/kriging/IDW/nearest
    interpolation (a "joint" freshly-built regular UTM-km grid, or a
    reused seismic-tomography native grid).
  "remap"  — remapping.py's conservative, cell-volume-weighted
    regrid (always a "joint" regular UTM-km grid -- see that script's
    own header for why a "seismic"-mode equivalent doesn't exist for a
    genuine conservative remap). Point interpolate.py and remapping.py
    at the same precompute.py outputs and cluster each one's output to
    see directly how much interpolate.py's point-cloud cell-size bias
    actually changes a given site's clustering.

This script does NOT build any grid or interpolate/remap anything
itself — it just reads whichever grid the chosen upstream script
already produced, and is therefore completely agnostic to how that
grid was built or which method produced it. CLUSTERING_METHOD picks
fuzzy c-means or SOM independently of any of that.

Pipeline
--------
precompute.py → interpolate.py → {SITE_PREFIX}_interp_<method>.nc  ─┐
             └→ remapping.py   → {SITE_PREFIX}_remap.npz          ─┼→ cluster.py → {SITE_PREFIX}_clusters.nc + figures

What this script does
----------------------
1. Reads INTERP_FILE or REMAP_FILE (per INPUT_KIND) — grid coords/dims,
   per-variable fields and their units, and enough metadata
   (target_grid_mode, interp_vars/remap_vars) to know how to plot it.
2. Picks CLUSTER_VARS, a subset of whatever variables are actually
   available (defaults to all of them) — so you can interpolate/remap
   a superset once and cluster on different subsets without re-running
   interpolate.py/remapping.py.
3. Builds one feature table, drops any grid cell with a NaN in a
   selected variable, optionally standardizes (z-score) each feature,
   then weights (CLUSTER_WEIGHTS).
4. Clusters via CLUSTERING_METHOD:
   - "fcm" — a self-contained (NumPy-only) fuzzy c-means (Bezdek, 1981),
     N_CLUSTERS discrete classes, reporting the fuzzy partition
     coefficient (FPC) as a quick quality check.
   - "som" — a self-contained (NumPy-only) Kohonen self-organizing map,
     SOM_ROWS x SOM_COLS neurons, every point labelled with its
     best-matching unit (BMU) over the FULL neuron grid (not collapsed
     to N clusters), colored with a topological colormap so visually
     similar map colors reflect genuinely similar feature-space
     neighbors.
5. Reconstructs the hard label + membership/quantization-error back onto
   the grid and saves {SITE_PREFIX}_clusters.nc / {SITE_PREFIX}_cluster_centers.csv.
6. Plots horizontal cluster maps at PLOT_DEPTHS_KM, on the same
   topography/bathymetry basemap as plot_seis.py. If the grid is
   regular in UTM space ("joint" -- always true for INPUT_KIND="remap",
   optionally true for INPUT_KIND="interp") this uses a plain
   imshow(extent=...); if it was built on a reused seismic-tomography
   ("seismic", INPUT_KIND="interp" only) grid — not regular in UTM
   space — this uses pcolormesh(shading="nearest") against that grid's
   own 2-D utm_easting_km/utm_northing_km coordinates instead.

Dependencies
------------
    numpy, xarray, pandas, matplotlib
plus the local `tomomt.py` helper module (also used by the plot
scripts). The fuzzy c-means / SOM implementations are self-contained
(NumPy only, no scikit-fuzzy/MiniSom).

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic). Split out of
cluster_rbf.py / cluster_kriging.py / cluster_som.py
into a single, interpolation-method-agnostic clustering script that
reads interpolate.py's output instead of building any grid or
interpolating itself: Claude (Anthropic), 2026-08-02. INPUT_KIND
added so this script reads either interpolate.py's NetCDF or
remapping.py's conservative-remap NPZ interchangeably: Claude
(Anthropic), 2026-09-06.
License: GNU General Public License v3 (GPL-3.0-or-later).
AI-generated code — review before use in production.
"""

import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator

import tomomt

to_utm_km = tomomt.to_utm_km
compute_hillshade = tomomt.compute_hillshade
clipped_markers = tomomt.clipped_markers
clipped_labels = tomomt.clipped_labels
draw_north_arrow = tomomt.draw_north_arrow

# =====================================================================
# USER SETTINGS
# =====================================================================

# --- Input/output directories ---
# --- Site selector ---
# Must match SITE_PREFIX in precompute.py / interpolate.py — used below
# to find their outputs and to name this script's own outputs.
SITE_PREFIX = "tacna"
# SITE_PREFIX = "ubinas"  # ubinas

# --- Input/output directories ---
NC_DIR = "../precompute/"+ f"{SITE_PREFIX}/"   # must match interpolate.py's NC_DIR;
                             # {SITE_PREFIX}_clusters.nc / {SITE_PREFIX}_cluster_centers.csv
                             # are also written here.
PLOT_DIR = "../plots_cluster/"
PLOT_FORMATS = [".pdf", ".png"]
PLOT_DPI = 600

# --- Which upstream output to read ---
# "interp" (default) — interpolate.py's point-cloud-interpolated
#   NetCDF (RBF/kriging/IDW/nearest); grid_mode is read from the file's
#   own target_grid_mode attribute ("joint" or "seismic").
# "remap"  — remapping.py's conservatively volume-weighted NPZ, always
#   on a regular UTM-km grid (there is no "seismic"-mode equivalent for
#   a conservative remap — see remapping.py's own header). Everything
#   below this point (feature-table build, clustering, plotting) is
#   otherwise completely agnostic to which of the two produced its
#   input, since remapping.py's output always looks like a "joint"-mode
#   interpolate.py one once loaded.
INPUT_KIND = "interp"   # "interp" | "remap"

# Bare filename, looked up under NC_DIR:
#   INPUT_KIND == "interp" — e.g. "{SITE_PREFIX}_interp_rbf.nc",
#     "{SITE_PREFIX}_interp_kriging.nc", "{SITE_PREFIX}_interp_idw.nc",
#     or whatever interpolate.py's OUTPUT_FILE was set to.
#   INPUT_KIND == "remap"  — e.g. "{SITE_PREFIX}_remap.npz", or
#     whatever remapping.py's OUTPUT_FILE was set to.
INTERP_FILE = f"{SITE_PREFIX}_interp_kriging.nc"
REMAP_FILE = f"{SITE_PREFIX}_remap.npz"

# --- Which variables to actually cluster on ---
# None = use every variable available in the chosen input file (its
# own interp_vars/remap_vars metadata). Set an explicit subset to
# cluster on fewer than what was interpolated/remapped — must all be
# present in that file, or this raises telling you to add them to
# interpolate.py's INTERP_VARS (or remapping.py's REMAP_VARS) and
# re-run it first.
CLUSTER_VARS =  None # ["rho",  "dens", "vps", "vp"] #None

# --- Gradient-based clustering features (optional) ---
# Two independent, opt-in mechanisms for bringing spatial-derivative
# information into the clustering feature table.
#
# (A) Per-variable gradient MAGNITUDE: no special setting -- just list
# "{key}_grad_mag" in CLUSTER_VARS above (e.g. CLUSTER_VARS = ["rho",
# "vps", "rho_grad_mag"]). Only available for INPUT_KIND="interp", and
# only if interpolate.py was itself run with COMPUTE_GRADIENT=True and
# that key in its own GRADIENT_VARS -- it stores "{key}_grad_mag" as an
# ordinary data variable even though it's not listed in interp_vars
# (see README_interpolate.md). remapping.py does not compute gradients
# -- referencing a "_grad_mag" variable under INPUT_KIND="remap" raises
# a clear error rather than a generic "not found".
#
# (B) Structural-ALIGNMENT features between variable PAIRS -- computed
# HERE (self-contained; works under either INPUT_KIND; "joint" grid
# mode only) via cosine similarity of two variables' own gradient
# vectors:
#     cos(theta) = (grad_A . grad_B) / (|grad_A| |grad_B|)
# in [-1, 1] -- +1 = co-aligned structure (the same boundary shows up
# in both models), -1 = anti-aligned (itself often geologically
# meaningful, e.g. resistive-where-fast vs. conductive-where-slow),
# 0 = unrelated structure that happens to cross. NaN wherever EITHER
# field's own local gradient magnitude falls below ITS OWN
# STRUCTURAL_MIN_GRADIENT_PERCENTILE-th percentile -- a per-variable,
# scale-free noise floor (not a fixed absolute threshold, since e.g.
# "rho" (log10 Ohm.m) and "vps" (dimensionless) have unrelated natural
# gradient scales), and deliberately NOT a threshold on the cross-
# gradient/cosine value itself: a near-zero cosine value where both
# gradients are genuinely large is a real (anti-)alignment signal that
# must be kept, not masked away just because it's "near zero" --
# see gradient_components_joint()/structural_alignment() below for
# the full reasoning.
#
# List each pair once; add "{a}_{b}_align" to CLUSTER_VARS to actually
# cluster on it (a listed pair that CLUSTER_VARS never references is
# not computed). CLUSTER_VARS must be given EXPLICITLY (not None)
# whenever STRUCTURAL_ALIGNMENT_PAIRS is non-empty -- "use everything"
# can't know to include a feature this script derives rather than one
# the input file itself provides.
STRUCTURAL_ALIGNMENT_PAIRS = []   # e.g. [("rho", "vps"), ("rho", "dens")]
STRUCTURAL_MIN_GRADIENT_PERCENTILE = 20.0

# --- Per-variable clustering weights ---
# Multiplies each (already standardized, if STANDARDIZE) feature by
# sqrt(weight) before clustering, so a larger weight makes that variable
# count more toward cluster assignment — equivalent to a weighted
# Euclidean distance d^2 = sum_j weight_j * (x_j - c_j)^2. Any
# CLUSTER_VARS entry not listed here falls back to 1.0 (no effect).
CLUSTER_WEIGHTS = {}
#CLUSTER_WEIGHTS = {
#    "rho_femtic": 2.0,   # counts 2× more than weight-1 variables (sqrt(4)=2)
#    "vps":        2.0,   # counts ~1.4× more
#    "vp":         1.0,   # neutral
#    "vs":         1.0,   # neutral
#    "dens":       1.0,  # counts 2× less (sqrt(0.25)=0.5)
#}


# --- Standardize (z-score) each variable before clustering ---
# Strongly recommended: rho/cond/vps/dens have very different numeric
# ranges and units, and Euclidean distance would otherwise be dominated
# by whichever variable happens to have the largest raw range.
STANDARDIZE = True

# --- Clustering method ---
# "fcm" = fuzzy c-means (Bezdek, 1981), N_CLUSTERS discrete classes.
# "som" = self-organizing map (Kohonen, 1982): trains a SOM_ROWS x
#   SOM_COLS grid of neurons and labels every point with its
#   best-matching unit (BMU) -- i.e. the FULL neuron grid is kept as the
#   output classification (SOM_ROWS*SOM_COLS classes), not collapsed
#   down to N_CLUSTERS. N_CLUSTERS/FUZZINESS/MAX_ITER/TOL below are only
#   used when CLUSTERING_METHOD == "fcm"; SOM_* settings are only used
#   when CLUSTERING_METHOD == "som".
CLUSTERING_METHOD = "fcm"

# --- Fuzzy c-means settings (CLUSTERING_METHOD == "fcm") ---
N_CLUSTERS = 5
FUZZINESS = 1.8      # "m" in the FCM literature; > 1, 2.0 is conventional
MAX_ITER = 300
TOL = 1e-5            # stop once max membership change between iterations < TOL
RANDOM_SEED = 42

# --- Self-organizing map settings (CLUSTERING_METHOD == "som") ---
SOM_ROWS = 5
SOM_COLS = 5
SOM_N_ITER = 12000     # online (one random sample per iteration) updates,
                       # NOT epochs over the full dataset
SOM_LR0 = 0.5          # initial learning rate; decays exponentially to
                       # 1% of this by the final iteration
SOM_SIGMA0 = None     # initial neighborhood radius, grid units; None =
                       # max(SOM_ROWS, SOM_COLS) / 2. Decays exponentially
                       # to a floor of 0.5.
SOM_SEED = 42

# --- Plotting ---

CLUSTER_CMAP = "tab10"             # qualitative colormap, used only for
                                    # CLUSTERING_METHOD == "fcm" (SOM uses
                                    # a topological colormap instead, see
                                    # som_grid_colormap())
if "tac" in SITE_PREFIX: 
    PLOT_DEPTHS_KM = [1.0, 5.0, 9.0]   # nearest available depth level is used for each
    ZMIN_SEISM = [-7, 1, 9]
    ZMAX_SEISM = [1, 9, 30]
if "ubi" in SITE_PREFIX: 
    PLOT_DEPTHS_KM = [-1.0, 5.0, 14.0, 28]
    ZMIN_SEISM = [-7, -1, 5, 14]
    ZMAX_SEISM = [-1, 5, 14, 28]


CLUSTER_ALPHA = 0.80

SHOW_TOPO_BASEMAP = True
HS_AZIMUTH, HS_ALTITUDE, HS_SIGMA = 315, 45, 1.0
TOPO_VMIN, TOPO_VMAX = 1000, 6000
OCEAN_COLOR = "#6baed6"

MAP_XLIM = None    # e.g. [310.0, 455.]  (easting,  km); None = auto from the grid
MAP_YLIM = None    # e.g. [7971.6, 8125] (northing, km); None = auto from the grid
REGION_MARGIN_KM = 0.0
FIG_WIDTH = 10.0    # cm — map panel width; height is derived, equal-scale by construction

AXES_UNITS = "km"          # "km" | "latlon"
AXES_KM_COMMA = True
LATLON_NTICKS = 5
LATLON_DECIMALS = 2

SHOW_COLORBAR = True
COLORBAR_POSITION = "right"
COLORBAR_SIZE = 0.85
COLORBAR_ASPECT = 20
COLORBAR_PAD = 0.10
COLORBAR_LABEL_SIZE = 10
COLORBAR_TICK_SIZE = 9

SHOW_NORTH_ARROW = True
ARROW_LON, ARROW_LAT, ARROW_LEN_KM = -73.6, -18.1, 4.0
ARROW_STYLE = dict(color="dimgray", lw=2, mutation_scale=14)
ARROW_LABEL_STYLE = dict(fontsize=9, fontweight="bold", color="dimgray")

AXIS_LABEL_SIZE = 12
AXIS_TICK_SIZE = 12
AXIS_TITLE_SIZE = 12

ANNOTATION_TEXT = None
ANNOTATION_POS = (0.01, 0.99)
ANNOTATION_STYLE = dict(fontsize=7, color="gray", ha="left", va="top")

# =====================================================================
# SPECIFIC PLOT SETTINGS — borrowed from plot_seis.py
# =====================================================================
SHOW_SPECIFIC_PLOT = True

CSV_VOLCANES      = "../features/volcanes.csv"
CSV_SEISMCAT      = "../features/catalog_welllocated_15_simple5.csv"
CSV_MT_SITES      = "../features/done/MTubinas_Sitelist.csv"
# CSV_MT_SITES      = "../features/done/MTubinas_Sitelist.csv"  # ubinas
CSV_CITIES        = "../features/cities.csv"
CSV_SEISMIC_SITES = "../features/seismic_sites.csv"  # no header row; columns
                                                       # are network, station,
                                                       # lat, lon, elev_m



if "tac" in SITE_PREFIX:                                                       
    PROFILE_1_LON  = [-70.48, -69.580]
    PROFILE_1_LAT  = [ -18.245, -17.135]
    PROFILE_2_LON = [ -70.034, -69.670]
    PROFILE_2_LAT = [ -17.267, -17.695]                                                       

if "ubi" in SITE_PREFIX:   
    PROFILE_1_LON = [-70.79, -70.95]
    PROFILE_1_LAT = [-16.52, -16.22]
    PROFILE_2_LON  = [-71.05, -70.73]
    PROFILE_2_LAT  = [-16.459, -16.21]





SHOW_SEISMICITY       = True
SHOW_MT_SITES         = True
SHOW_SEISMIC_SITES    = True
SHOW_VOLCANOES        = True   # inactive volcano markers + labels
SHOW_VOLCANOES_ACTIVE = True   # active volcano markers
SHOW_CITIES           = True


VOLC_LABEL_IDX = [5, 12, 13]
# Volcano name column (volcanes.csv). Labels are truncated to their
# first VOLC_LABEL_CHARS characters via VOLC_LABEL_STYLE's mode="firstN"
# (see tomomt.apply_label_mode) rather than reading a separate
# already-abbreviated column (e.g. "VOLCAN2") — one source of truth for
# the name, with truncation as a display-only concern.
VOLC_NAME_COL = "NAME"

EQ_MARKER_STYLE = dict(
    marker="o", s=4.5, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=11,
)
MT_MARKER_STYLE = dict(
    marker="v", s=10, facecolors="yellow", edgecolors="black",
    linewidths=0.7, zorder=12,
)
MT_LABEL_STYLE = dict(
    fontsize=5, color="black", zorder=14, rotation=90,
    offset_x=0.3, offset_y=0.3, mode="none",
)
SEISMIC_SITES_MARKER_STYLE = dict(
    marker="v", s=10, facecolors="green", edgecolors="black",
    linewidths=0.7, zorder=12,
)
VOLC_INACT_MARKER_STYLE = dict(
    marker="^", s=10, facecolors="blue", edgecolors="black",
    linewidths=0.7, zorder=13,
)
VOLC_LABEL_CHARS = 4   # number of leading characters shown (mode="firstN"
                        # below) — was previously a separate "VOLCAN2"
                        # abbreviated-code column; adjust to taste.
VOLC_LABEL_STYLE = dict(
    fontsize=6, fontweight="bold", color="black", zorder=14,
    offset_x=0.3, offset_y=0.3, mode=f"first{VOLC_LABEL_CHARS}",
)
VOLC_ACT_MARKER_STYLE = dict(
    marker="^", s=10, facecolors="red", edgecolors="black",
    linewidths=0.7, zorder=13,
)
CITY_MARKER_STYLE = dict(
    marker="s", s=6, facecolors="black", edgecolors="black",
    linewidths=0.2, zorder=13,
)
CITY_LABEL_STYLE = dict(
    fontsize=6, color="white", zorder=14,
    offset_x=0.3, offset_y=-0.3, mode="full",
)

if SHOW_SPECIFIC_PLOT and SHOW_SEISMICITY:
    if not (len(ZMIN_SEISM) == len(ZMAX_SEISM) == len(PLOT_DEPTHS_KM)):
        raise SystemExit(
            f"ZMIN_SEISM ({len(ZMIN_SEISM)}), ZMAX_SEISM ({len(ZMAX_SEISM)}), "
            f"and PLOT_DEPTHS_KM ({len(PLOT_DEPTHS_KM)}) must all be the same "
            f"length — one seismicity depth-window pair per depth slice. Pad "
            f"the shorter list(s) with None (= show all seismicity) for any "
            f"slice that doesn't need a filter."
        )

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
        zmin_km=-6.0, 
        zmax_km=30.0, 
        swath_km=10.0,
    ),
    dict(
        name="profile BB'", 
        p1 = [PROFILE_2_LON[0], PROFILE_2_LAT[0]],   # lon, lat
        p2 = [PROFILE_2_LON[1], PROFILE_2_LAT[1]],
        coord="latlon", 
        zmin_km=-6.0, 
        zmax_km=30.0, 
        swath_km=10.0,
    ),
]


# Colour-scale for vertical slices: defaults to the same as the maps;
# override per-variable by adding entries here (same dict keys as CMAP_SETTINGS).
VSLICE_CMAP_OVERRIDES = {}

if "tac" in SITE_PREFIX:
    VSLICE_VE           = 3.0
else:
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

# --- Section plotting toggle ---
# True  : compute and plot a cluster vertical section for every VSLICES entry.
# False : skip section plotting entirely (only depth-slice maps are produced).
PLOT_SECTIONS = True

# Section figure size: fixed height in cm (same convention as plot_interp_ubinas).
# Width is derived from profile length / depth range / VSLICE_VE so that
# the physical aspect ratio is always correct.
# VSLICE_HEIGHT_CM = None → use VSLICE_WIDTH_CM as the fixed dimension instead.
VSLICE_HEIGHT_CM = 15.24   # 6 inches — matches plot_interp_ubinas / structure_ubinas
VSLICE_WIDTH_CM  = None    # None → derived from height


# =====================================================================
# END USER SETTINGS
# =====================================================================

Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)


def ncpath(name):
    """Join a bare NetCDF filename onto NC_DIR."""
    return tomomt.resolve_path(NC_DIR, name)


safe_to_netcdf = tomomt.safe_to_netcdf
safe_open_w = tomomt.safe_open_w


# ------------------------------------------------------------------
# Structural-alignment clustering feature (STRUCTURAL_ALIGNMENT_PAIRS)
# ------------------------------------------------------------------
def gradient_components_joint(field, e_axis, n_axis, d_axis):
    """
    Component-wise spatial gradient of `field` (dims depth, northing,
    easting) via plain numpy.gradient along each axis' own real
    coordinate spacing -- the same joint-grid-only, unsmoothed approach
    structure.py's own gradient_components() already uses (see
    README_tomomt.md) -- used here purely to build the
    STRUCTURAL_ALIGNMENT_PAIRS cosine-similarity feature below. This is
    a deliberately separate, simpler implementation from
    interpolate.py's own (spline-based, seismic-grid-aware) gradient
    machinery -- see tomomt.py's own module docstring for why that
    duplication is intentional rather than an oversight.

    NaNs in `field` propagate to every finite-difference stencil that
    touches them, same as everywhere else in this pipeline.
    """
    field = np.asarray(field, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        grad_depth = np.gradient(field, d_axis, axis=0, edge_order=1)
        grad_northing = np.gradient(field, n_axis, axis=1, edge_order=1)
        grad_easting = np.gradient(field, e_axis, axis=2, edge_order=1)
    return grad_easting, grad_northing, grad_depth


def structural_alignment(field_a, field_b, e_axis, n_axis, d_axis,
                          min_gradient_percentile):
    """
    Cosine similarity of field_a's and field_b's own gradient vectors,
    in [-1, 1]: cos(theta) = (grad_A . grad_B) / (|grad_A| |grad_B|).

    NaN wherever EITHER field's own local gradient magnitude falls
    below ITS OWN `min_gradient_percentile`-th percentile (computed
    separately for each field, over its own finite values) -- this is
    the noise floor below which a gradient "direction" isn't
    meaningfully defined, and it deliberately does NOT threshold the
    cosine value itself: a near-zero cosine at a location where BOTH
    gradients are large is a genuine (anti-)alignment signal and must
    be kept, not discarded just because the number is small -- see
    STRUCTURAL_ALIGNMENT_PAIRS's own settings comment for the full
    reasoning (this is the "near-zero masking" question CLUSTER_VARS
    users most often get backwards).

    Returns (cosine, floor_a, floor_b) -- the two floors are returned
    too, purely so the caller can print them for a sanity check.
    """
    ge_a, gn_a, gd_a = gradient_components_joint(field_a, e_axis, n_axis, d_axis)
    ge_b, gn_b, gd_b = gradient_components_joint(field_b, e_axis, n_axis, d_axis)
    mag_a = np.sqrt(ge_a ** 2 + gn_a ** 2 + gd_a ** 2)
    mag_b = np.sqrt(ge_b ** 2 + gn_b ** 2 + gd_b ** 2)

    dot = ge_a * ge_b + gn_a * gn_b + gd_a * gd_b
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = dot / (mag_a * mag_b)
    cosine = np.clip(cosine, -1.0, 1.0)

    finite_a = mag_a[np.isfinite(mag_a)]
    finite_b = mag_b[np.isfinite(mag_b)]
    floor_a = float(np.percentile(finite_a, min_gradient_percentile)) if finite_a.size else np.inf
    floor_b = float(np.percentile(finite_b, min_gradient_percentile)) if finite_b.size else np.inf

    undefined = (
        ~np.isfinite(mag_a) | ~np.isfinite(mag_b)
        | (mag_a < floor_a) | (mag_b < floor_b)
    )
    cosine = np.where(undefined, np.nan, cosine)
    return cosine.astype(np.float32), floor_a, floor_b


def align_feature_name(pair):
    a, b = pair
    return f"{a}_{b}_align"


# ------------------------------------------------------------------
# Fuzzy c-means (self-contained, NumPy only)
# ------------------------------------------------------------------
def fuzzy_cmeans(X, n_clusters, m=2.0, max_iter=300, tol=1e-5, seed=42):
    """
    Standard (Bezdek, 1981) fuzzy c-means clustering.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Already NaN-free.
    n_clusters : int
    m : float
        Fuzziness exponent, > 1. 2.0 is the conventional default.
    max_iter, tol : stopping criteria.
    seed : random seed for the initial membership matrix.

    Returns
    -------
    centers, U, fpc, n_iter — see body for details.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    U = rng.random((n_samples, n_clusters))
    U /= U.sum(axis=1, keepdims=True)

    centers = None
    for it in range(max_iter):
        Um = U ** m
        centers = (Um.T @ X) / Um.sum(axis=0)[:, None]

        dist = np.empty((n_samples, n_clusters))
        for j in range(n_clusters):
            dist[:, j] = np.linalg.norm(X - centers[j], axis=1)
        dist = np.fmax(dist, 1e-12)

        inv = dist ** (-2.0 / (m - 1.0))
        U_new = inv / inv.sum(axis=1, keepdims=True)

        diff = float(np.max(np.abs(U_new - U)))
        U = U_new
        if diff < tol:
            break

    fpc = float(np.sum(U ** 2) / n_samples)
    return centers, U, fpc, it + 1


# ------------------------------------------------------------------
# Self-organizing map (self-contained, NumPy only)
# ------------------------------------------------------------------
def train_som(X, rows, cols, n_iter=8000, lr0=0.5, sigma0=None, seed=42):
    """
    Online (sequential) Kohonen self-organizing map. See
    cluster_som.py's original docstring for the full parameter
    rationale; unchanged here.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    n_neurons = rows * cols
    if sigma0 is None:
        sigma0 = max(rows, cols) / 2.0

    data_min, data_max = X.min(axis=0), X.max(axis=0)
    weights = rng.uniform(data_min, data_max, size=(n_neurons, n_features))

    grid_r, grid_c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    grid_coords = np.column_stack([grid_r.ravel(), grid_c.ravel()]).astype(np.float64)

    sample_idx = rng.integers(0, n_samples, size=n_iter)

    for it in range(n_iter):
        frac = it / max(n_iter - 1, 1)
        lr = lr0 * (0.01 ** frac)
        sigma = max(sigma0 * (0.05 ** frac), 0.5)

        x = X[sample_idx[it]]
        dists = np.linalg.norm(weights - x, axis=1)
        bmu = int(np.argmin(dists))

        grid_dist_sq = np.sum((grid_coords - grid_coords[bmu]) ** 2, axis=1)
        neighborhood = np.exp(-grid_dist_sq / (2.0 * sigma ** 2))

        weights += lr * neighborhood[:, None] * (x - weights)

    return weights, grid_coords


def som_bmu_assign(X, weights):
    """Assign every row of X to its best-matching unit (BMU)."""
    dists = np.linalg.norm(X[:, None, :] - weights[None, :, :], axis=2)
    bmu_idx = np.argmin(dists, axis=1)
    qe = dists[np.arange(len(X)), bmu_idx]
    return bmu_idx, qe


def som_topographic_error(X, weights, grid_coords):
    """Fraction of points whose best- and second-best-matching units are
    NOT adjacent on the SOM grid (4-neighborhood, Manhattan distance 1)."""
    dists = np.linalg.norm(X[:, None, :] - weights[None, :, :], axis=2)
    order = np.argsort(dists, axis=1)
    c1 = grid_coords[order[:, 0]]
    c2 = grid_coords[order[:, 1]]
    grid_dist = np.abs(c1 - c2).sum(axis=1)
    return float(np.mean(grid_dist > 1))


def som_grid_colormap(rows, cols):
    """Topological ListedColormap: nearby SOM neurons get visually
    similar colors — see cluster_som.py's original docstring."""
    colors = np.zeros((rows * cols, 3))
    for r in range(rows):
        for c in range(cols):
            hue = c / max(cols - 1, 1)
            val = 0.55 + 0.45 * (r / max(rows - 1, 1))
            colors[r * cols + c] = mcolors.hsv_to_rgb([hue, 0.75, val])
    return mcolors.ListedColormap(colors)


# ==================================================================
# Resolve which base variables actually need to be loaded from the
# upstream file: CLUSTER_VARS entries that are not themselves a
# structural-alignment feature name, plus (regardless of whether
# CLUSTER_VARS asks for it directly) every variable referenced by
# STRUCTURAL_ALIGNMENT_PAIRS, since those are needed as gradient
# INPUTS even when only the alignment score itself is being clustered
# on.
# ==================================================================
_align_name_to_pair = {align_feature_name(p): tuple(p) for p in STRUCTURAL_ALIGNMENT_PAIRS}

if STRUCTURAL_ALIGNMENT_PAIRS and CLUSTER_VARS is None:
    raise ValueError(
        "STRUCTURAL_ALIGNMENT_PAIRS is set but CLUSTER_VARS is None -- "
        "'use everything from the input file' can't know to also include "
        "a feature this script derives rather than one the file itself "
        f"provides. Set CLUSTER_VARS explicitly (known alignment feature "
        f"names from the pairs configured: {sorted(_align_name_to_pair)})."
    )

if CLUSTER_VARS is not None:
    unknown_align = [v for v in CLUSTER_VARS
                      if v.endswith("_align") and v not in _align_name_to_pair]
    if unknown_align:
        raise KeyError(
            f"CLUSTER_VARS references {unknown_align}, which looks like a "
            f"structural-alignment feature name but has no matching entry "
            f"in STRUCTURAL_ALIGNMENT_PAIRS -- add that pair there first "
            f"(currently configured pairs produce: "
            f"{sorted(_align_name_to_pair) or '(none configured)'})."
        )
    _requested_align_names = [v for v in CLUSTER_VARS if v in _align_name_to_pair]
    _base_vars_requested = [v for v in CLUSTER_VARS if v not in _align_name_to_pair]
    _pair_base_vars = sorted({
        v for name in _requested_align_names for v in _align_name_to_pair[name]
    })
    _vars_to_load = sorted(set(_base_vars_requested) | set(_pair_base_vars))
else:
    _requested_align_names = []
    _vars_to_load = None   # sentinel: "everything the input file itself lists"

# ==================================================================
# Read the upstream grid -- interpolate.py's NetCDF or remapping.py's
# NPZ (INPUT_KIND) -- into one common set of variables (grid_mode,
# dim_names, d_axis/row_coord/col_coord, _loaded, resolved_units,
# utm_easting_2d/utm_northing_2d, grid_shape) so everything from here
# on is unaware of which one actually produced them.
# ==================================================================
if INPUT_KIND == "interp":
    interp_path = ncpath(INTERP_FILE)
    print(f"Reading interpolated grid: {interp_path} …")
    interp_ds = xr.open_dataset(interp_path)

    grid_mode = interp_ds.attrs["target_grid_mode"]        # "joint" | "seismic"
    dim_names = tuple(interp_ds.attrs["dim_names"].split(", "))
    dim_depth, dim_row, dim_col = dim_names
    vars_available = interp_ds.attrs["interp_vars"].split(", ")

    _vars_to_load_resolved = list(vars_available) if _vars_to_load is None else _vars_to_load
    missing = [v for v in _vars_to_load_resolved if v not in interp_ds.data_vars]
    if missing:
        raise KeyError(
            f"CLUSTER_VARS/STRUCTURAL_ALIGNMENT_PAIRS reference {missing}, "
            f"not present in {INTERP_FILE!r} (it has "
            f"{list(interp_ds.data_vars)}) — add them to interpolate.py's "
            f"INTERP_VARS/GRADIENT_VARS and re-run it first."
        )

    d_axis = interp_ds[dim_depth].values.astype(np.float64)
    row_coord = interp_ds[dim_row].values
    col_coord = interp_ds[dim_col].values

    _loaded = {}
    resolved_units = {}
    for key in _vars_to_load_resolved:
        da = interp_ds[key]
        _loaded[key] = da.values.astype(np.float32)
        resolved_units[key] = da.attrs.get("units", "")

    if grid_mode == "joint":
        e_axis = interp_ds[dim_col].values.astype(np.float64)     # "easting"
        n_axis = interp_ds[dim_row].values.astype(np.float64)     # "northing"
        utm_easting_2d = None
        utm_northing_2d = None
    else:  # "seismic"
        e_axis = None
        n_axis = None
        utm_easting_2d = interp_ds["utm_easting_km"].values
        utm_northing_2d = interp_ds["utm_northing_km"].values

    grid_shape = _loaded[_vars_to_load_resolved[0]].shape  # (depth, row, col)
    interp_ds.close()

elif INPUT_KIND == "remap":
    remap_path = ncpath(REMAP_FILE)
    print(f"Reading remapped grid: {remap_path} …")
    with np.load(remap_path, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta_json"]))
        vars_available = list(meta["remap_vars"])

        _vars_to_load_resolved = list(vars_available) if _vars_to_load is None else _vars_to_load
        missing = [v for v in _vars_to_load_resolved if v not in npz.files]
        if missing:
            grad_mag_missing = [v for v in missing if v.endswith("_grad_mag")]
            if grad_mag_missing:
                raise KeyError(
                    f"{grad_mag_missing} requested, but remapping.py does "
                    f"not compute gradients -- '_grad_mag' clustering "
                    f"features are only available for INPUT_KIND='interp' "
                    f"(interpolate.py, with COMPUTE_GRADIENT=True and that "
                    f"key in GRADIENT_VARS). Use STRUCTURAL_ALIGNMENT_PAIRS "
                    f"instead for a gradient-based feature under "
                    f"INPUT_KIND='remap' -- that one is computed here, not "
                    f"read from the file."
                )
            raise KeyError(
                f"CLUSTER_VARS/STRUCTURAL_ALIGNMENT_PAIRS reference "
                f"{missing}, not present in {REMAP_FILE!r} (it has "
                f"{vars_available}) — add them to remapping.py's "
                f"REMAP_VARS and re-run it first."
            )

        # remapping.py always targets a regular UTM-km grid -- no
        # "seismic" (reused native grid) mode exists for a conservative
        # remap, see that script's own header -- so this always plugs
        # into the "joint" branch below.
        grid_mode = "joint"
        dim_names = tuple(meta["dim_names"])   # ("depth", "northing", "easting")
        dim_depth, dim_row, dim_col = dim_names

        d_axis = npz["depth"].astype(np.float64)
        n_axis = npz["northing"].astype(np.float64)
        e_axis = npz["easting"].astype(np.float64)
        row_coord = n_axis
        col_coord = e_axis
        utm_easting_2d = None
        utm_northing_2d = None

        _loaded = {}
        resolved_units = {}
        for key in _vars_to_load_resolved:
            _loaded[key] = npz[key].astype(np.float32)
            resolved_units[key] = meta["units"].get(key, "")

        grid_shape = _loaded[_vars_to_load_resolved[0]].shape  # (depth, northing, easting)

else:
    raise ValueError(f"INPUT_KIND must be 'interp' or 'remap', got {INPUT_KIND!r}.")

_source_file = INTERP_FILE if INPUT_KIND == "interp" else REMAP_FILE

# ==================================================================
# Structural-alignment features (STRUCTURAL_ALIGNMENT_PAIRS), if any
# were actually requested via CLUSTER_VARS
# ==================================================================
if _requested_align_names:
    if grid_mode != "joint":
        raise NotImplementedError(
            "STRUCTURAL_ALIGNMENT_PAIRS requires a 'joint' (regular "
            "UTM-km) grid -- this input is grid_mode='seismic' "
            "(curvilinear); a proper spatial gradient there needs the "
            "same row/col -> easting/northing Jacobian correction "
            "interpolate.py's own gradient code already does, not "
            "duplicated here. Re-run interpolate.py with "
            "TARGET_GRID='joint', or use remapping.py's output instead "
            "(always 'joint')."
        )
    print(f"\nComputing structural-alignment features: {_requested_align_names} …")
    for name in _requested_align_names:
        a, b = _align_name_to_pair[name]
        cosine, floor_a, floor_b = structural_alignment(
            _loaded[a].astype(np.float64), _loaded[b].astype(np.float64),
            e_axis, n_axis, d_axis, STRUCTURAL_MIN_GRADIENT_PERCENTILE,
        )
        n_defined = int(np.isfinite(cosine).sum())
        print(
            f"    {name!r}: {n_defined} / {cosine.size} cells have a "
            f"defined alignment ({100.0 * n_defined / cosine.size:.1f}%); "
            f"gradient-magnitude noise floors "
            f"(P{STRUCTURAL_MIN_GRADIENT_PERCENTILE:g}) -- "
            f"{a}: {floor_a:.4g}, {b}: {floor_b:.4g}"
        )
        _loaded[name] = cosine
        resolved_units[name] = "cos(theta)"

active_cluster_vars = list(CLUSTER_VARS) if CLUSTER_VARS is not None else list(vars_available)
print(f"\nClustering on: {active_cluster_vars}  "
      f"(grid_mode={grid_mode!r}, from {_source_file!r})")

n_total = int(np.prod(grid_shape))
print(f"Grid shape {dim_names}: {grid_shape} ({n_total} cells total)")

# ==================================================================
# Build feature table
# ==================================================================
feature_stack = np.stack(
    [_loaded[k].ravel() for k in active_cluster_vars], axis=1
)  # (n_total, n_features)
valid_mask = np.all(np.isfinite(feature_stack), axis=1)
n_valid = int(valid_mask.sum())
print(
    f"Valid (finite in every selected variable): {n_valid} / {n_total} "
    f"({100.0 * n_valid / n_total:.1f}%)"
)
if CLUSTERING_METHOD not in ("fcm", "som"):
    raise ValueError(f"CLUSTERING_METHOD must be 'fcm' or 'som', got {CLUSTERING_METHOD!r}.")
n_labels = N_CLUSTERS if CLUSTERING_METHOD == "fcm" else SOM_ROWS * SOM_COLS

if n_valid < n_labels:
    raise RuntimeError(
        f"Only {n_valid} valid points — fewer than the number of classes "
        f"({n_labels}). Check CLUSTER_VARS or interpolate.py's masking settings."
    )

X_raw = feature_stack[valid_mask]  # (n_valid, n_features)

if STANDARDIZE:
    feat_mean = X_raw.mean(axis=0)
    feat_std = X_raw.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    X = (X_raw - feat_mean) / feat_std
else:
    feat_mean = np.zeros(X_raw.shape[1])
    feat_std = np.ones(X_raw.shape[1])
    X = X_raw

weight_vector = np.array(
    [CLUSTER_WEIGHTS.get(k, 1.0) for k in active_cluster_vars], dtype=float
)
if np.any(weight_vector <= 0):
    raise ValueError(
        f"CLUSTER_WEIGHTS must be positive; got "
        f"{dict(zip(active_cluster_vars, weight_vector))}."
    )
print(
    "Cluster weights: " +
    ", ".join(f"{k}={w:g}" for k, w in zip(active_cluster_vars, weight_vector))
)
sqrt_weight = np.sqrt(weight_vector)
X_weighted = X * sqrt_weight[None, :]

# ==================================================================
# Cluster: fuzzy c-means or self-organizing map
# ==================================================================
diagnostics = {}

if CLUSTERING_METHOD == "fcm":
    print(f"\nRunning fuzzy c-means: n_clusters={N_CLUSTERS}, m={FUZZINESS} …")
    centers_weighted, U, fpc, n_iter = fuzzy_cmeans(
        X_weighted, N_CLUSTERS, m=FUZZINESS, max_iter=MAX_ITER, tol=TOL, seed=RANDOM_SEED
    )
    hard_label = np.argmax(U, axis=1)
    membership_max = np.max(U, axis=1)
    centers_std = centers_weighted / sqrt_weight[None, :]
    centers_raw = centers_std * feat_std + feat_mean

    print(
        f"Converged after {n_iter} iterations. "
        f"Fuzzy partition coefficient (FPC): {fpc:.3f} "
        f"(1/{N_CLUSTERS} = {1.0 / N_CLUSTERS:.3f} = fuzziest, 1 = crisp)"
    )
    diagnostics["fuzziness_m"] = FUZZINESS
    diagnostics["fpc"] = fpc
    diagnostics["n_iter"] = n_iter
    diagnostics["random_seed"] = RANDOM_SEED

else:  # CLUSTERING_METHOD == "som"
    print(
        f"\nTraining SOM: grid={SOM_ROWS}x{SOM_COLS} ({n_labels} neurons), "
        f"n_iter={SOM_N_ITER}, lr0={SOM_LR0} …"
    )
    weights_weighted, grid_coords = train_som(
        X_weighted, SOM_ROWS, SOM_COLS,
        n_iter=SOM_N_ITER, lr0=SOM_LR0, sigma0=SOM_SIGMA0, seed=SOM_SEED,
    )
    hard_label, qe = som_bmu_assign(X_weighted, weights_weighted)
    membership_max = qe  # SOM's quantization error, NOT a [0, 1] membership value
    topo_err = som_topographic_error(X_weighted, weights_weighted, grid_coords)
    centers_std = weights_weighted / sqrt_weight[None, :]
    centers_raw = centers_std * feat_std + feat_mean

    print(
        f"Done. Mean quantization error: {qe.mean():.3f} (standardized units); "
        f"topographic error: {topo_err:.3f} "
        f"(0 = perfectly neighborhood-preserving, higher = worse)"
    )
    diagnostics["som_grid"] = f"{SOM_ROWS}x{SOM_COLS}"
    diagnostics["som_n_iter"] = SOM_N_ITER
    diagnostics["som_lr0"] = SOM_LR0
    diagnostics["som_sigma0"] = SOM_SIGMA0 if SOM_SIGMA0 is not None else max(SOM_ROWS, SOM_COLS) / 2.0
    diagnostics["som_seed"] = SOM_SEED
    diagnostics["mean_quantization_error"] = float(qe.mean())
    diagnostics["topographic_error"] = topo_err

print("\nCluster centers (raw units) and sizes:")
header = "  cluster |    n    |  frac  | " + " | ".join(
    f"{k} ({resolved_units[k] or '-'})" for k in active_cluster_vars
)
print(header)
for c in range(n_labels):
    n_c = int(np.sum(hard_label == c))
    frac = n_c / n_valid
    vals = "  ".join(f"{centers_raw[c, j]:9.3f}" for j in range(len(active_cluster_vars)))
    print(f"  {c:7d} | {n_c:7d} | {frac:5.1%} | {vals}")

# ==================================================================
# Reconstruct grid + save
# ==================================================================
label_flat = np.full(n_total, -1, dtype=np.int16)
label_flat[valid_mask] = hard_label.astype(np.int16)
label_grid = label_flat.reshape(grid_shape)

membership_flat = np.full(n_total, np.nan, dtype=np.float32)
membership_flat[valid_mask] = membership_max.astype(np.float32)
membership_grid = membership_flat.reshape(grid_shape)

_method_label = "Fuzzy c-means" if CLUSTERING_METHOD == "fcm" else "SOM (self-organizing map)"
_label_long_name = (
    "Fuzzy c-means hard cluster label (argmax membership)" if CLUSTERING_METHOD == "fcm"
    else "SOM best-matching-unit (BMU) index, flat row*cols+col over the SOM_ROWS x SOM_COLS grid"
)
_membership_long_name = (
    "Membership of the assigned (hard-label) cluster" if CLUSTERING_METHOD == "fcm"
    else "Quantization error: distance (standardized/weighted feature space) from the point to "
         "its BMU weight vector -- smaller is a better fit, NOT a [0, 1] membership value"
)

out_coords = {
    dim_depth: (dim_depth, d_axis, {"units": "km", "positive": "down"}),
    dim_row: (dim_row, row_coord),
    dim_col: (dim_col, col_coord),
}
if grid_mode == "seismic":
    out_coords["utm_easting_km"] = ((dim_row, dim_col), utm_easting_2d)
    out_coords["utm_northing_km"] = ((dim_row, dim_col), utm_northing_2d)

out_ds = xr.Dataset(
    {
        "cluster_label": (
            dim_names, label_grid,
            {
                "long_name": _label_long_name,
                "flag_value_missing": -1,
                "cluster_vars": ", ".join(active_cluster_vars),
                "n_clusters": n_labels,
                "clustering_method": CLUSTERING_METHOD,
            },
        ),
        "membership": (
            dim_names, membership_grid,
            {
                "long_name": _membership_long_name,
                "units": "1" if CLUSTERING_METHOD == "fcm" else "(standardized/weighted feature units)",
            },
        ),
    },
    coords=out_coords,
    attrs={
        "description": (
            f"{_method_label} clustering of " + ", ".join(active_cluster_vars) +
            f" on the grid read from {_source_file} "
            f"(see {'interpolate.py' if INPUT_KIND == 'interp' else 'remapping.py'})."
        ),
        "input_kind": INPUT_KIND,
        "source_file": _source_file,
        "target_grid_mode": grid_mode,
        "clustering_method": CLUSTERING_METHOD,
        "cluster_weights": ", ".join(
            f"{k}={w:g}" for k, w in zip(active_cluster_vars, weight_vector)
        ),
        "standardized": str(STANDARDIZE),
        **{k: ("" if v is None else v) for k, v in diagnostics.items()},
    },
)
out_nc = ncpath(f"{SITE_PREFIX}_clusters.nc")
safe_to_netcdf(out_ds, out_nc)
print(f"\nSaved: {out_nc}")

centers_csv = ncpath(f"{SITE_PREFIX}_cluster_centers.csv")
with safe_open_w(centers_csv, newline="") as f:
    w = csv.writer(f)
    w.writerow(["cluster", "n_points", "fraction"] + active_cluster_vars)
    w.writerow(["weight", "", ""] + [f"{wt:g}" for wt in weight_vector])
    for c in range(n_labels):
        n_c = int(np.sum(hard_label == c))
        w.writerow(
            [c, n_c, f"{n_c / n_valid:.6f}"]
            + [f"{centers_raw[c, j]:.6g}" for j in range(len(active_cluster_vars))]
        )
print(f"Saved: {centers_csv}")

# ==================================================================
# Load topo/bath basemap
# ==================================================================
print("\nLoading topo/bath grids …")
_topo_da = xr.open_dataarray(ncpath(f"{SITE_PREFIX}_topo_utm.nc"))
topo_x = _topo_da["x"].values
topo_y = _topo_da["y"].values
topo_z = _topo_da.values
_topo_da.close()
dx_km = float(np.median(np.diff(topo_x)))
dy_km = float(np.median(np.diff(topo_y)))

if SHOW_TOPO_BASEMAP:
    topo_hs = compute_hillshade(topo_z, dx_km, dy_km, HS_AZIMUTH, HS_ALTITUDE, HS_SIGMA)
else:
    topo_hs = None

_bath_da = xr.open_dataarray(ncpath(f"{SITE_PREFIX}_bath_utm.nc"))
bath_x = _bath_da["x"].values
bath_y = _bath_da["y"].values
bath_z = _bath_da.values
_bath_da.close()

topo_extent = [topo_x.min(), topo_x.max(), topo_y.min(), topo_y.max()]
bath_extent = [bath_x.min(), bath_x.max(), bath_y.min(), bath_y.max()]
topo_norm = mcolors.Normalize(vmin=TOPO_VMIN, vmax=TOPO_VMAX) if SHOW_TOPO_BASEMAP else None
CMAP_TOPO = plt.get_cmap("gray")

# ==================================================================
# Map region
# ==================================================================
if grid_mode == "joint":
    _e_min, _e_max = float(e_axis.min()), float(e_axis.max())
    _n_min, _n_max = float(n_axis.min()), float(n_axis.max())
else:
    _e_min, _e_max = float(np.nanmin(utm_easting_2d)), float(np.nanmax(utm_easting_2d))
    _n_min, _n_max = float(np.nanmin(utm_northing_2d)), float(np.nanmax(utm_northing_2d))

if MAP_XLIM is not None:
    xmin, xmax = MAP_XLIM
else:
    xmin = _e_min - REGION_MARGIN_KM
    xmax = _e_max + REGION_MARGIN_KM
if MAP_YLIM is not None:
    ymin, ymax = MAP_YLIM
else:
    ymin = _n_min - REGION_MARGIN_KM
    ymax = _n_max + REGION_MARGIN_KM
print(f"Map region (km): [{xmin}, {xmax}, {ymin}, {ymax}]")


def _region():
    return (xmin, xmax, ymin, ymax)


def _colorbar_settings():
    return dict(
        show=SHOW_COLORBAR, position=COLORBAR_POSITION,
        size=COLORBAR_SIZE, pad=COLORBAR_PAD, aspect=COLORBAR_ASPECT,
        label_size=COLORBAR_LABEL_SIZE, tick_size=COLORBAR_TICK_SIZE,
        nticks=n_labels, title_size=AXIS_TITLE_SIZE,
    )


def create_map_figure():
    return tomomt.build_map_figure(
        FIG_WIDTH, xmin, xmax, ymin, ymax, _colorbar_settings(), size_label="map"
    )


def draw_basemap(ax):
    """Topo greyscale + hillshade + ocean fill; enforce map limits."""
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    if SHOW_TOPO_BASEMAP:
        ax.imshow(
            CMAP_TOPO(topo_norm(topo_z)), origin="lower", extent=topo_extent,
            aspect="auto", interpolation="bilinear", zorder=1,
        )
        ax.imshow(
            topo_hs, cmap="gray", origin="lower", extent=topo_extent,
            alpha=0.45, aspect="auto", interpolation="bilinear", zorder=2,
        )
    bath_mask = np.where(bath_z <= 0, 1.0, np.nan)
    ax.imshow(
        bath_mask, origin="lower", extent=bath_extent,
        cmap=mcolors.ListedColormap([OCEAN_COLOR]), vmin=0, vmax=1,
        alpha=0.85, aspect="auto", interpolation="none", zorder=3,
    )
    ax.set_xlabel("Easting (km)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Northing (km)", fontsize=AXIS_LABEL_SIZE)
    if AXES_UNITS == "km" and AXES_KM_COMMA:
        _comma_fmt = mpl.ticker.StrMethodFormatter("{x:,.0f}")
        ax.xaxis.set_major_formatter(_comma_fmt)
        ax.yaxis.set_major_formatter(_comma_fmt)
    ax.tick_params(labelsize=AXIS_TICK_SIZE)
    if SHOW_NORTH_ARROW:
        arr_e, arr_n = to_utm_km([ARROW_LON], [ARROW_LAT])
        tomomt.draw_north_arrow(
            ax, arr_e[0], arr_n[0], _region(),
            ARROW_STYLE, ARROW_LABEL_STYLE, ARROW_LEN_KM,
        )


def save_fig(fig, stem):
    return tomomt.save_fig(fig, stem, PLOT_DIR, PLOT_FORMATS, PLOT_DPI)


# ==================================================================
# Vertical section helpers
# ==================================================================

def _section_colorbar_settings():
    s = _colorbar_settings()
    ref_in = (VSLICE_HEIGHT_CM if VSLICE_HEIGHT_CM is not None
              else VSLICE_WIDTH_CM) / 2.54
    s["thickness_in"] = (COLORBAR_SIZE * ref_in) / COLORBAR_ASPECT
    return s


def _topo_interp():
    """Build a RegularGridInterpolator on the topo grid (northing, easting)."""
    ty = topo_y if topo_y[0] < topo_y[-1] else topo_y[::-1]
    tz = topo_z if topo_y[0] < topo_y[-1] else topo_z[::-1, :]
    return RegularGridInterpolator(
        (ty, topo_x), tz, method="linear", bounds_error=False, fill_value=np.nan,
    )


def compute_cluster_section(vslice, label_grid_3d, d_axis_in, e_ax, n_ax):
    """
    Sample the integer cluster-label grid along a profile by nearest-neighbour
    lookup (NearestNDInterpolator on the 3-D grid — identical to the method
    used for cluster *maps*, which already use interpolation="nearest").

    Returns
    -------
    dist_km      : 1-D along-profile distance
    depth_plot   : 1-D depth axis clipped to [zmin_km, zmax_km]
    section      : 2-D float array (ndepth_clipped, npts), NaN where label == -1
    e_ends, n_ends : profile endpoint UTM km
    topo_prof    : 1-D surface elevation (m) along profile, or None
    utm_x        : x-axis array (same as dist_km when VSLICE_X_AXIS=="distance")
    utm_xlabel   : label string for that axis
    """
    npts    = vslice.get("npts", 200)
    zmin_km = vslice.get("zmin_km", float(d_axis_in[0]))
    zmax_km = vslice.get("zmax_km", float(d_axis_in[-1]))

    e_ends, n_ends = tomomt.profile_utm_km(vslice)
    dist_km, e_pts, n_pts, utm_x, utm_xlabel = \
        tomomt.sample_profile_points(e_ends, n_ends, npts)

    # Clip depth axis to display window
    d_ax = d_axis_in.copy()
    if d_ax[0] > d_ax[-1]:
        d_ax = d_ax[::-1]
        label_grid_3d = label_grid_3d[::-1, :, :]

    # Ensure horizontal axes ascending for the interpolator
    lv3 = label_grid_3d.astype(float)
    lv3[lv3 < 0] = np.nan   # -1 (missing) → NaN

    n_asc = n_ax if n_ax[0] < n_ax[-1] else n_ax[::-1]
    e_asc = e_ax if e_ax[0] < e_ax[-1] else e_ax[::-1]
    if n_ax[0] > n_ax[-1]:
        lv3 = lv3[:, ::-1, :]
    if e_ax[0] > e_ax[-1]:
        lv3 = lv3[:, :, ::-1]

    interp = RegularGridInterpolator(
        (d_ax, n_asc, e_asc), lv3,
        method="nearest", bounds_error=False, fill_value=np.nan,
    )

    # Clip to display window
    d_mask    = (d_ax >= zmin_km) & (d_ax <= zmax_km)
    depth_plot = d_ax[d_mask]

    nz = len(depth_plot)
    d_q = np.repeat(depth_plot, npts)
    n_q = np.tile(n_pts, nz)
    e_q = np.tile(e_pts, nz)
    section = interp(np.column_stack([d_q, n_q, e_q])).reshape(nz, npts)

    # Topography along profile
    topo_prof = None
    if topo_z is not None:
        topo_prof = _topo_interp()(np.column_stack([n_pts, e_pts]))

    return dist_km, depth_plot, section, e_ends, n_ends, topo_prof, utm_x, utm_xlabel


def plot_cluster_section(vslice, vi, label_grid_3d, d_axis_in, e_ax, n_ax,
                         annotated=False):
    """
    Produce and save one cluster vertical-section figure.

    Parameters
    ----------
    vslice        : VSLICES dict entry
    vi            : index into VSLICES (used for profile endpoint labels)
    label_grid_3d : (ndepth, nrow, ncol) int array of cluster labels
    d_axis_in     : 1-D depth axis (km)
    e_ax, n_ax    : 1-D easting / northing axes (km)
    annotated     : if True, add seismicity projection (requires SHOW_SPECIFIC_PLOT)
    """
    name    = vslice.get("name", "profile")
    swath   = vslice.get("swath_km", 10.0)
    zmin_km = vslice.get("zmin_km", float(d_axis_in[0]))
    zmax_km = vslice.get("zmax_km", float(d_axis_in[-1]))
    ve      = 1.0 if VSLICE_EQUAL_SCALE else VSLICE_VE

    dist_km, depth_plot, section, e_ends, n_ends, topo_prof, utm_x, utm_xlabel = \
        compute_cluster_section(vslice, label_grid_3d, d_axis_in, e_ax, n_ax)

    lbl_start, lbl_end = tomomt.profile_labels(vi)

    if VSLICE_X_AXIS == "distance":
        x_arr, x_label = dist_km, "Distance along profile (km)"
    else:
        x_arr, x_label = utm_x, utm_xlabel

    # Figure size: same formula as plot_interp_ubinas / structure_ubinas
    profile_len = dist_km[-1]
    depth_range = (float(depth_plot[-1]) - float(depth_plot[0])) \
                  if len(depth_plot) > 1 else (zmax_km - zmin_km)
    if VSLICE_HEIGHT_CM is not None:
        h_in = VSLICE_HEIGHT_CM / 2.54
        w_in = h_in * profile_len / max(depth_range * ve, 1e-6)
    else:
        w_in = VSLICE_WIDTH_CM / 2.54
        h_in = w_in * (depth_range * ve) / profile_len

    fig, ax, cax = tomomt.build_panel_figure(w_in, h_in,
                                              _section_colorbar_settings(),
                                              size_label="section")

    # VE label
    if ve != 1.0:
        vx, vy, vha, vva = tomomt.resolve_ve_pos(VSLICE_VE_POS)
        ax.text(vx, vy, f"VE = {ve:.1f}×",
                transform=ax.transAxes, ha=vha, va=vva,
                zorder=21, **VSLICE_VE_STYLE)

    # Cluster section (nearest-neighbour colours — no blending between classes)
    im = ax.pcolormesh(
        x_arr, depth_plot, section,
        cmap=cluster_cmap, norm=cluster_norm,
        shading="nearest", alpha=CLUSTER_ALPHA, zorder=5,
    )

    # Topo line
    y_top = float(depth_plot[0]) if len(depth_plot) else zmin_km
    if topo_prof is not None:
        surf_depth = -topo_prof / 1e3
        finite_surf = surf_depth[np.isfinite(surf_depth)]
        if finite_surf.size > 0:
            y_top = float(finite_surf.min()) - VSLICE_TOPO_HEADROOM_KM
            if VSLICE_SHOW_TOPO_FILL:
                land  = topo_prof > 0
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
            print(f"    WARNING: topo all-NaN for '{name}' — topo line skipped.")

    # Seismicity projection (annotated variant only)
    if annotated and SHOW_SPECIFIC_PLOT and SHOW_SEISMICITY:
        eq_dist, eq_dep = tomomt.project_points_to_profile(
            eq_e0, eq_n0, e_ends, n_ends, swath,
            z0=zeqs, zmin_km=zmin_km, zmax_km=zmax_km,
        )
        if len(eq_dist):
            eq_x = np.interp(eq_dist, dist_km, x_arr)
            tomomt.markers(ax, eq_x, eq_dep, **VSLICE_EQ_STYLE)

    # MT sites on section (annotated variant only)
    if annotated and SHOW_SPECIFIC_PLOT and SHOW_MT_SITES:
        de = e_ends[1] - e_ends[0]
        dn = n_ends[1] - n_ends[0]
        L  = np.sqrt(de**2 + dn**2)
        if L > 0:
            ue, un = de / L, dn / L
            ve_mt = mt_e - e_ends[0]
            vn_mt = mt_n - n_ends[0]
            along  = ve_mt * ue + vn_mt * un
            across = np.abs(ve_mt * (-un) + vn_mt * ue)
            in_swath = (across <= swath) & (along >= 0) & (along <= L)
            if in_swath.any():
                # Place MT sites at topo surface depth
                mt_topo = _topo_interp()(
                    np.column_stack([mt_n[in_swath], mt_e[in_swath]])
                )
                mt_dep = np.where(np.isfinite(mt_topo), -mt_topo / 1e3, 0.0)
                mt_x = np.interp(along[in_swath], dist_km, x_arr)
                tomomt.markers(ax, mt_x, mt_dep, **MT_MARKER_STYLE)

    # Axes
    xlim = vslice.get("xlim", None)
    ax.set_xlim(xlim if xlim is not None else (x_arr[0], x_arr[-1]))
    ylim = vslice.get("ylim", None)
    ax.set_ylim(ylim if ylim is not None else (y_top, zmax_km))
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Depth (km)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=AXIS_TICK_SIZE)

    # Profile endpoint labels
    x0, x1 = x_arr[0], x_arr[-1]
    for xpos, lbl in ((x0, lbl_start), (x1, lbl_end)):
        ax.text(xpos, y_top, lbl,
                ha="center", va="bottom",
                fontsize=AXIS_LABEL_SIZE, fontweight="bold",
                color="black", clip_on=False, zorder=20)

    ax.set_title(
        f"{_method_label} classes ({', '.join(active_cluster_vars)}) — {name}",
        fontsize=AXIS_TITLE_SIZE,
    )
    tomomt.finish_panel_colorbar(cax, im, "Class", _section_colorbar_settings())
    tomomt.draw_annotation(ax, ANNOTATION_TEXT, ANNOTATION_POS, ANNOTATION_STYLE)

    ann_tag = "_annotated" if annotated else ""
    safe_name = name.replace(" ", "_").replace("'", "p")
    stem = f"clusters_section_{safe_name}_{SITE_PREFIX}{ann_tag}"
    save_fig(fig, stem)
    plt.show()
    plt.close(fig)


# ==================================================================
# Feature layers for the specific (annotated) plot — same CSVs/loading
# as plot_seis.py
# ==================================================================
if SHOW_SPECIFIC_PLOT:
    print("\nLoading feature layers for the specific (annotated) plot …")

    volcanes = pd.read_csv(CSV_VOLCANES)
    utmv_e, utmv_n = to_utm_km(
        volcanes["LONG"][VOLC_LABEL_IDX].values,
        volcanes["LAT"][VOLC_LABEL_IDX].values,
    )
    if VOLC_NAME_COL not in volcanes.columns:
        print(f"  WARNING: volcano name column {VOLC_NAME_COL!r} not found in "
              f"{CSV_VOLCANES} — labels will be blank.")
        namev = [""] * len(VOLC_LABEL_IDX)
    else:
        namev = volcanes[VOLC_NAME_COL][VOLC_LABEL_IDX].values

    volc_act_e, volc_act_n = [], []
    for _i in range(len(volcanes)):
        if "ACT" in str(volcanes["ESTADO"][_i]):
            _ae, _an = to_utm_km([volcanes["LONG"][_i]], [volcanes["LAT"][_i]])
            volc_act_e.append(_ae[0])
            volc_act_n.append(_an[0])

    eqs = pd.read_csv(CSV_SEISMCAT, delimiter=" ")
    eq_e0, eq_n0 = to_utm_km(eqs["x"].values, eqs["y"].values)
    zeqs = eqs["z"].values

    _mt = pd.read_csv(CSV_MT_SITES, delimiter=" ")
    mt_e, mt_n = to_utm_km(_mt["x"].values, _mt["y"].values)
    for _name_col in ("Site", "site", "name", "Name", "station", "Station"):
        if _name_col in _mt.columns:
            mt_names = _mt[_name_col].astype(str).tolist()
            break
    else:
        mt_names = [""] * len(_mt)

    seis_sites = pd.read_csv(CSV_SEISMIC_SITES, header=None,
                              names=["network", "station", "lat", "lon", "elev_m"])
    seis_site_e, seis_site_n = to_utm_km(seis_sites["lon"].values, seis_sites["lat"].values)
    seis_site_names = seis_sites["station"].values

    cities = pd.read_csv(CSV_CITIES)
    cit_e, cit_n = to_utm_km(cities["x"].values, cities["y"].values)
    name_cit = cities["Name"].values


def draw_specific_features(ax, eq_e, eq_n):
    """Overlay seismicity/MT-site/seismic-site/volcano/city feature layers
    borrowed from plot_seis.py's draw_features()."""
    if SHOW_SEISMICITY:
        clipped_markers(ax, eq_e, eq_n, _region(), label="Seismicity", **EQ_MARKER_STYLE)

    if SHOW_MT_SITES:
        clipped_markers(ax, mt_e, mt_n, _region(), label="MT site", **MT_MARKER_STYLE)
        clipped_labels(ax, mt_e, mt_n, mt_names, MT_LABEL_STYLE, _region())

    if SHOW_SEISMIC_SITES:
        clipped_markers(ax, seis_site_e, seis_site_n, _region(), label="Seismic site",
                         **SEISMIC_SITES_MARKER_STYLE)

    if SHOW_VOLCANOES:
        clipped_markers(ax, utmv_e, utmv_n, _region(), **VOLC_INACT_MARKER_STYLE)
        clipped_labels(ax, utmv_e, utmv_n, namev, VOLC_LABEL_STYLE, _region())

    if SHOW_VOLCANOES_ACTIVE and volc_act_e:
        clipped_markers(ax, volc_act_e, volc_act_n, _region(),
                         label="Active volcano", **VOLC_ACT_MARKER_STYLE)

    if SHOW_CITIES:
        clipped_markers(ax, cit_e, cit_n, _region(), label="City", **CITY_MARKER_STYLE)
        clipped_labels(ax, cit_e, cit_n, name_cit, CITY_LABEL_STYLE, _region())


# ==================================================================
# Plot cluster maps
# ==================================================================
if CLUSTERING_METHOD == "som":
    cluster_cmap = som_grid_colormap(SOM_ROWS, SOM_COLS)
else:
    cluster_cmap = plt.get_cmap(CLUSTER_CMAP, N_CLUSTERS)
bounds = np.arange(-0.5, n_labels + 0.5, 1.0)
cluster_norm = mcolors.BoundaryNorm(bounds, cluster_cmap.N)

_tick_stride = max(1, -(-n_labels // 12))  # ceil(n_labels / 12)
_cbar_ticks = list(range(0, n_labels, _tick_stride))


def _draw_cluster_overlay(ax, cax, label_slice, actual_depth):
    """Cluster map + title + colorbar + lon/lat ticks + free-text
    annotation — the part shared by both the plain and the specific
    (annotated) cluster maps. Uses imshow(extent=...) for a regular
    "joint" grid, pcolormesh(shading="nearest") against the grid's own
    2-D UTM coords for a reused "seismic" grid (not regular in UTM
    space)."""
    if grid_mode == "joint":
        im = ax.imshow(
            label_slice, cmap=cluster_cmap, norm=cluster_norm, origin="lower",
            extent=[e_axis.min(), e_axis.max(), n_axis.min(), n_axis.max()],
            alpha=CLUSTER_ALPHA, aspect="equal", interpolation="nearest", zorder=5,
        )
    else:
        im = ax.pcolormesh(
            utm_easting_2d, utm_northing_2d, label_slice,
            cmap=cluster_cmap, norm=cluster_norm, shading="nearest",
            alpha=CLUSTER_ALPHA, zorder=5,
        )
    ax.set_title(
        f"{_method_label} classes ({', '.join(active_cluster_vars)}) at {actual_depth:.1f} km",
        fontsize=AXIS_TITLE_SIZE,
    )
    if SHOW_COLORBAR:
        cbar = tomomt.finish_panel_colorbar(cax, im, "Class", _colorbar_settings())
        cbar.set_ticks(_cbar_ticks)
    if AXES_UNITS == "latlon":
        tomomt.add_latlon_ticks(
            ax, _region(), LATLON_NTICKS, LATLON_DECIMALS,
            AXIS_LABEL_SIZE, AXIS_TICK_SIZE,
        )
    tomomt.draw_annotation(ax, ANNOTATION_TEXT, ANNOTATION_POS, ANNOTATION_STYLE)


for i_depth, target_depth in enumerate(PLOT_DEPTHS_KM):
    iz = int(np.argmin(np.abs(d_axis - target_depth)))
    actual_depth = float(d_axis[iz])
    print(
        f"\nPlotting clusters at {target_depth} km "
        f"(nearest available: {actual_depth:.2f} km) …"
    )

    label_slice = label_grid[iz].astype(float)  # (row, col)
    label_slice[label_slice < 0] = np.nan  # -1 (missing) -> NaN, transparent

    tag = f"{actual_depth:.0f}km" if actual_depth == int(actual_depth) else f"{actual_depth:.1f}km"

    # --- Plain cluster map ---
    fig, ax, cax = create_map_figure()
    draw_basemap(ax)
    _draw_cluster_overlay(ax, cax, label_slice, actual_depth)
    save_fig(fig, f"clusters_{tag}_{SITE_PREFIX}")
    plt.show()
    plt.close(fig)

    # --- Specific (annotated) cluster map ---
    if SHOW_SPECIFIC_PLOT:
        if SHOW_SEISMICITY:
            zmin, zmax = ZMIN_SEISM[i_depth], ZMAX_SEISM[i_depth]
            if zmin is None and zmax is None:
                eq_e, eq_n = eq_e0, eq_n0
            else:
                _zmask = np.ones(len(zeqs), dtype=bool)
                if zmin is not None:
                    _zmask &= zeqs >= zmin
                if zmax is not None:
                    _zmask &= zeqs <= zmax
                eq_e, eq_n = eq_e0[_zmask], eq_n0[_zmask]
        else:
            eq_e, eq_n = np.array([]), np.array([])

        fig, ax, cax = create_map_figure()
        draw_basemap(ax)
        _draw_cluster_overlay(ax, cax, label_slice, actual_depth)
        draw_specific_features(ax, eq_e, eq_n)
        save_fig(fig, f"clusters_{tag}_{SITE_PREFIX}_annotated")
        plt.show()
        plt.close(fig)

print("\nDone.")


# ==================================================================
# Plot cluster vertical sections
# ==================================================================
if PLOT_SECTIONS and VSLICES:
    # For "joint" grids the 3-D label array has dims (depth, northing, easting)
    # with 1-D coordinate axes e_axis / n_axis.  For "seismic" grids the
    # horizontal layout is irregular (2-D utm coords) so vertical sections are
    # not implemented (a regular UTM grid is required for the RegularGridInterpolator).
    if grid_mode != "joint":
        print(
            "\nNote: PLOT_SECTIONS is only supported for a 'joint' "
            "(regular UTM-km) grid. The loaded input file uses a 'seismic' "
            "grid with 2-D UTM coordinates — section plotting skipped. "
            "(Only possible with INPUT_KIND='interp'; remapping.py's NPZ "
            "output is always 'joint'.)"
        )
    else:
        print(f"\nPlotting {len(VSLICES)} cluster vertical section(s) …")
        for vi, vslice in enumerate(VSLICES):
            name = vslice.get("name", f"profile{vi}")
            print(f"  Section: {name!r}")

            # Plain section (no seismicity / MT overlay)
            plot_cluster_section(
                vslice, vi, label_grid, d_axis, e_axis, n_axis,
                annotated=False,
            )

            # Annotated section (seismicity + MT sites) — only when
            # SHOW_SPECIFIC_PLOT is True and the feature layers are loaded
            if SHOW_SPECIFIC_PLOT:
                plot_cluster_section(
                    vslice, vi, label_grid, d_axis, e_axis, n_axis,
                    annotated=True,
                )
