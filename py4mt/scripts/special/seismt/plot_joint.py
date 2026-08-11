#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_joint.py
=====================
Depth-slice maps (and, where the grid is regular, vertical cross-
sections) of every field on the common grid produced by
interpolate.py — MT resistivity/conductivity/sensitivity and/or
seismic tomography Vp/Vs/Vp-Vs-ratio/density, whichever were
interpolated — INCLUDING their spatial gradients, if
interpolate.py's COMPUTE_GRADIENT was on.

This is the plotting counterpart to cluster.py: both scripts read
interpolate.py's output (INTERP_FILE) and are completely agnostic
to how that grid was built (a fresh joint UTM grid, or a reused
seismic-tomography native grid) or which interpolation method produced
it (RBF/kriging/IDW/nearest). Where cluster.py clusters the fields
and plots the resulting class labels, this script plots each field's own
value (and gradient magnitude/components) directly — no clustering.

Pipeline
--------
precompute.py → interpolate.py → {SITE_PREFIX}_interp_<method>.nc
                                                     ↓
                                    plot_joint.py → figures

Two grid modes (see interpolate.py's TARGET_GRID)
---------------------------------------------------------
  "joint"    — a genuinely regular UTM-km (depth, northing, easting)
    grid. Depth-slice maps are drawn with imshow(extent=...); vertical
    cross-sections (VSLICES, same schema as plot_seis.py) are
    supported, sampled via a 3-D RegularGridInterpolator directly in UTM
    space (simpler than plot_seis.py's own lat/lon-based sampler,
    since this grid is already in UTM).
  "seismic"  — a reused seismic-tomography native (depth, row, col) grid,
    NOT regular in UTM space (2-D utm_easting_km/utm_northing_km aux
    coords instead). Depth-slice maps are drawn with
    pcolormesh(shading="nearest") against those 2-D coords, matching
    cluster.py's own rendering. Vertical cross-sections are NOT
    supported in this mode (an arbitrary-angle straight-line cut through
    an irregular-in-UTM grid needs its own interpolation step — build
    INTERP_FILE with TARGET_GRID="joint" instead if you need sections);
    VSLICES is simply skipped with a printed warning.

Gradients (PLOT_GRADIENT)
----------------------------
If interpolate.py's COMPUTE_GRADIENT was True, INTERP_FILE also
carries, for each gradient variable, up to four extra data variables:
"{key}_grad_easting", "{key}_grad_northing", "{key}_grad_depth" (partial
derivatives) and "{key}_grad_mag" (their combined 3-D magnitude) — see
that script's own docstring. PLOT_GRADIENT here controls whether this
script ALSO produces gradient maps/sections for those fields.
GRADIENT_COMPONENTS picks which of "mag"/"easting"/"northing"/"depth" to
plot — "mag" only by default, since that is the one every other field
shares comparable units for at a glance. A field with no matching
"{key}_grad_*" in INTERP_FILE (either COMPUTE_GRADIENT was off, or that
key wasn't in interpolate.py's GRADIENT_VARS) is silently skipped.

Standard layout: property + |∇field| paired (PAIR_GRADIENT_WITH_FIELD)
------------------------------------------------------------------------------
Whenever a field's gradient MAGNITUDE was plotted (PLOT_GRADIENT=True,
"mag" in GRADIENT_COMPONENTS, and that field has one), it's drawn in the
SAME figure as the field itself, each keeping its own colour scale/
colorbar (VAR_STYLE / GRAD_STYLE respectively) — rather than as two
separate files. This is the default (PAIR_GRADIENT_WITH_FIELD=True); set
it False to go back to one file per panel. The two panels are arranged
side by side (1 row × 2 columns) or stacked (2 rows × 1 column)
automatically, based on the aspect ratio a SINGLE panel would have on
its own — a wide/landscape panel (e.g. a broad map, or a long vertical
section) stacks vertically so the combined figure doesn't end up
implausibly wide; a tall/portrait panel goes side by side so it doesn't
end up implausibly tall (see _build_dual_panel_figure). Falls back to a
single-panel figure automatically wherever there's nothing to pair with
(no gradient computed, "mag" not requested, or that field wasn't in
GRADIENT_VARS) — PLOT_GRADIENT/GRADIENT_COMPONENTS still control what's
available to pair in the first place. Any OTHER gradient components you
also request (e.g. "easting") are NOT paired — magnitude is the natural
partner; other components remain their own separate single-panel
figures, with a "_grad<component>" filename suffix as before.

Colour scales (VAR_STYLE)
-----------------------------
Unlike plot_seis.py / plot_dens.py — each hard-coded to one
or three known fields — the set of fields actually present in
INTERP_FILE is only known at run time (whatever interpolate.py's
INTERP_VARS happened to be). VAR_STYLE below is therefore a dict keyed
by field name, giving each a colormap and a (cmin, cmax) pair; a field
not listed there falls back to DEFAULT_CMAP and auto-scaling (cmin/cmax
= None -> the finite min/max of that particular depth slice/section,
computed fresh per panel, same convention as the other plot scripts'
CMIN_GRAD=None/CMAX_GRAD=None default). Gradient magnitude/component
panels always auto-scale (there is no natural fixed range for a spatial
derivative) unless given an explicit override in GRAD_STYLE.

Most settings below (basemap, colorbar layout, marker/label styles,
CSV feature paths, VSLICES schema, isolines) are carried over unchanged
from plot_seis.py / cluster.py — see those scripts' own
comments for the full rationale behind each one; only the parts specific
to plotting an arbitrary set of interpolated fields are re-explained
here.

Dependencies
------------
    numpy, xarray, pandas, matplotlib, scipy (RegularGridInterpolator,
    only used for "joint"-mode vertical sections)
plus the local `plotpy.py` helper module.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic). Written from
plot_seis.py (settings/plotting style) and cluster.py
(grid-mode-aware data loading) as models: Claude (Anthropic), 2026-08-05.
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
from scipy.interpolate import RegularGridInterpolator

import plotpy

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# Colormap import helper (matplotlib name / GMT .cpt file / plain RGB(A)
# file) — see plotpy.load_colormap for the full docstring.
# ---------------------------------------------------------------------
load_colormap = plotpy.load_colormap


# =====================================================================
# USER SETTINGS
# =====================================================================

# --- Site selector ---
# Prefixes every precompute.py/interpolate.py output filename this
# script reads and this script's own output filenames. Must match
# SITE_PREFIX in precompute.py / interpolate.py.
#SITE_PREFIX = "saba"
SITE_PREFIX = "tacna"  # TACNA

# --- Input/output directories ---
NC_DIR = "../precompute/tacna/"     # must match precompute.py's OUTPUT_DIR
                               # and interpolate.py's NC_DIR.
PLOT_DIR = "../plots_joint/"

PLOT_FORMATS = [".pdf", ".jpg"]
PLOT_DPI     = 600

# Whether to also display each figure on screen after saving it (in
# addition to always writing PLOT_FORMATS to disk). Previously this
# script called plt.show() unconditionally, which is only harmless in an
# environment that keeps a live GUI event loop open between calls (e.g.
# Spyder's own console). Run the same script from a plain terminal, a
# batch job, or the DIAS cluster and plt.show() either blocks waiting on
# a display that will never advance, or errors outright on a machine
# with no display at all -- neither of which has anything to do with
# whether the figures themselves were produced correctly. SHOW_PLOTS
# defaults to False so headless/batch runs "just work"; _maybe_show()
# below additionally only calls plt.show() when matplotlib is actually
# running an interactive backend, so turning this on is still safe if
# the script happens to run somewhere headless.
SHOW_PLOTS = False

# --- Which interpolate.py output to read ---
# Bare filename, looked up under NC_DIR — e.g. "{SITE_PREFIX}_interp_rbf.nc",
# "{SITE_PREFIX}_interp_kriging.nc", "{SITE_PREFIX}_interp_idw.nc", or whatever
# interpolate.py's OUTPUT_FILE was set to.
INTERP_FILE = f"{SITE_PREFIX}_interp_kriging.nc"

# --- Interpolation-method tag for output filenames ---
# Derived automatically from INTERP_FILE's own "_interp_<method>" naming
# (e.g. "kriging"/"krig" -> "krig", "rbf" -> "rbf", "idw" -> "idw",
# "nearest"/"nn" -> "nn") so every figure this script writes is tagged
# with both the area (SITE_PREFIX) and the interpolation method that
# produced INTERP_FILE, matching cluster.py's own filenames. An
# unrecognised method string is flagged (printed warning) and passed
# through unchanged rather than guessed at.
_INTERP_METHOD_TAG_MAP = {
    "kriging": "krig", "krige": "krig", "krig": "krig",
    "rbf": "rbf",
    "idw": "idw",
    "nearest_neighbour": "nn", "nearest_neighbor": "nn",
    "nearest": "nn", "nn": "nn",
}


def _derive_interp_tag(filename):
    stem_name = os.path.splitext(os.path.basename(filename))[0]
    marker = "_interp_"
    method_str = stem_name.split(marker, 1)[1] if marker in stem_name else stem_name
    method_str = method_str.lower()
    for key, tag in _INTERP_METHOD_TAG_MAP.items():
        if key in method_str:
            return tag
    print(f"  WARNING: could not recognise interpolation method from "
          f"INTERP_FILE={filename!r} (looked for one of "
          f"{sorted(set(_INTERP_METHOD_TAG_MAP.values()))}) — using raw "
          f"string {method_str!r} in output filenames instead.")
    return method_str


INTERP_TAG = _derive_interp_tag(INTERP_FILE)

# --- Which fields to plot ---
# None = every field in INTERP_FILE's own interp_vars attribute (i.e.
# everything interpolate.py actually interpolated). Set an explicit
# subset to plo2t fewer — must all be present in INTERP_FILE, or this
# raises telling you to add them to interpolate.py's INTERP_VARS
# and re-run it first.
PLOT_VARS = None

# --- Gradients ---
# See module docstring. PLOT_GRADIENT=False (default) skips gradient
# panels entirely, regardless of what's in INTERP_FILE.
PLOT_GRADIENT = True
GRADIENT_COMPONENTS = ["mag"]   # subset of "mag" | "easting" | "northing" | "depth"

# Standard layout: whenever a field's gradient MAGNITUDE was plotted
# (PLOT_GRADIENT=True and "mag" in GRADIENT_COMPONENTS and that field
# actually has one), draw it together with the field itself in ONE
# figure/file instead of two separate figures. Each panel keeps its own
# colour scale and colorbar (VAR_STYLE / GRAD_STYLE respectively); only
# the figure/canvas is shared. The two panels are arranged side by side
# or stacked automatically, based on what aspect ratio a single panel
# would have on2 its own (see _build_dual_panel_figure) — not a fixed
# choice here. Falls back to a single-panel figure (the pre-pairing
# behaviour) whenever there's no gradient magnitude to pair with — no
# gradient computed at all, "mag" not requested, or that particular
# field wasn't in interpolate.py's GRADIENT_VARS — so PLOT_GRADIENT
# / GRADIENT_COMPONENTS above still control what's available to pair.
# Any OTHER gradient components you also request (e.g. "easting") are
# NOT paired — magnitude is the one every field shares comparable units
# for, so it's the natural pairing partner; other components are
# still plotted as their own separate single-panel figures.
PAIR_GRADIENT_WITH_FIELD = True
PAIR_GAP_CM = 4.0   # gap between the two panels, in cm (horizontal gap if
                     # side by side, vertical gap if stacked)

# Figure sizes (cm). Horizontal maps: FIG_WIDTH controls EACH map
# panel's width (whether shown alone or paired) — height is
# always derived from it and the UTM data aspect ratio (equal x/y km
# scale by construction; see create_map_figure()/create_map_figure_pair()).
FIG_WIDTH = 10.0   # cm

# Vertical sections ("joint" grid mode only): height derived from
# (depth_range * VE / profile_len) scaled to VSLICE_WIDTH_CM if None.
# Same per-panel sizing applies whether shown alone or paired.
VSLICE_WIDTH_CM  = None
VSLICE_HEIGHT_CM = 10.0

# --- Depth slices to plot (km); nearest available level in INTERP_FILE
# is used for each ---
PLOT_DEPTHS_KM = [1.0, 5.0, 9.0]

# Seismicity depth windows (km), one pair per entry in PLOT_DEPTHS_KM.
# Set both to None to show all seismicity on every slice.
ZMIN_SEISM = [-7, 1,  9]
ZMAX_SEISM = [ 1, 9, 30]

if not (len(ZMIN_SEISM) == len(ZMAX_SEISM) == len(PLOT_DEPTHS_KM)):
    sys.exit(
        f"ZMIN_SEISM ({len(ZMIN_SEISM)}), ZMAX_SEISM ({len(ZMAX_SEISM)}), "
        f"and PLOT_DEPTHS_KM ({len(PLOT_DEPTHS_KM)}) must all be the same "
        f"length — one seismicity depth-window pair per depth slice. Pad "
        f"the shorter list(s) with None (= show all seismicity) for any "
        f"slice that doesn't need a filter."
    )

# --- Per-field colour scale / colormap ---
# Keyed by the field name as it appears in INTERP_FILE (e.g. "rho",
# "cond", "sens", "vp", "vs", "vps", "dens"). cmin/cmax = None -> that
# particular panel auto-scales to its own finite data range (computed
# fresh per depth slice/section). cmap accepts anything plotpy.load_colormap
# does (matplotlib name, .cpt file, plain RGB(A) file). A field not
# listed here falls back to DEFAULT_CMAP and full auto-scaling.
#
# NOTE: cmin/cmax units must match whatever's actually in INTERP_FILE
# (its own per-variable `units` attribute, printed at run time below) —
# e.g. Vp/Vs may be stored in km/s or m/s depending on the source model;
# adjust the placeholder ranges below to match before relying on them.
VAR_STYLE = {
    "rho":  dict(cmap="viridis_r",  cmin=None, cmax=None),
    "cond": dict(cmap="viridis",    cmin=None, cmax=None),
    "sens": dict(cmap="viridis_r",  cmin=None, cmax=None),
    "vp":   dict(cmap="viridis_r",  cmin=None, cmax=None),
    "vs":   dict(cmap="viridis_r",  cmin=None, cmax=None),
    "vps":  dict(cmap="hot_r",      cmin=None, cmax=None),
    "dens": dict(cmap="cividis",    cmin=None, cmax=None),
}
DEFAULT_CMAP = "viridis"

# --- Gradient panel colour scale / colormap ---
# Keyed by "{var}_grad_{component}" (e.g. "rho_grad_mag") for a specific
# override, else DEFAULT_GRAD_CMAP and full auto-scaling (there's no
# natural fixed range for a spatial derivative).
GRAD_STYLE = {}
DEFAULT_GRAD_CMAP = "magma"

# Data-layer transparency (0 = opaque, 1 = invisible) — applies to every
# field/gradient panel alike.
ALPHA_DATA = 0.50

# --- Isolines (contours) on top of the field images ---
# Applies to both the depth-slice maps and the vertical sections below.
ISO_LINES_MAP    = False
ISO_LINES_VSLICE = False

# Contour levels. May be given EITHER:
#   "auto"                 -> ISO_AUTO_N evenly spaced levels spanning
#                              the finite data range of each panel
#   [v1, v2, ...]           -> applied to every field, as-is
#   {"rho": ..., "vp": ...} -> per-field spec (each value itself "auto"
#                              or an explicit list); fields not present
#                              default to "auto". Keyed the same way as
#                              VAR_STYLE — including "{var}_grad_mag" etc.
#                              for gradient panels, if you want different
#                              levels there.
ISO_LEVELS_MAP    = "auto"
ISO_LEVELS_VSLICE = "auto"
ISO_AUTO_N = 6

ISO_STYLE = dict(colors="black", linewidths=0.6, linestyles="solid", zorder=7)
ISO_LABEL          = True
ISO_LABEL_FMT      = "%.2g"
ISO_LABEL_FONTSIZE = 6

# =====================================================================
# BASEMAP SETTINGS
# =====================================================================
NC_TOPO = f"{SITE_PREFIX}_topo_utm.nc"
NC_BATH = f"{SITE_PREFIX}_bath_utm.nc"

SHOW_TOPO_BASEMAP = True
HS_AZIMUTH  = 315
HS_ALTITUDE = 45
HS_SIGMA    = 1.0

TOPO_VMIN = 1000
TOPO_VMAX = 6000

OCEAN_COLOR = "#6baed6"

MAP_XLIM = None    # e.g. [310.0, 455.]  (easting,  km); None = auto from the grid
MAP_YLIM = None    # e.g. [7971.6, 8125] (northing, km); None = auto from the grid
REGION_MARGIN_KM = 0.0

# =====================================================================
# MAP AXES UNITS — see plot_seis.py for the full docstring.
# =====================================================================
AXES_UNITS       = "km"   # "km" | "latlon"
LATLON_NTICKS    = 5
LATLON_DECIMALS  = 2
AXES_KM_COMMA    = True

# =====================================================================
# COLORBAR SETTINGS — see plot_seis.py for the full docstring.
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
# MAP FEATURE LAYERS — simple on/off switches; see plot_seis.py.
# =====================================================================
SHOW_PROFILE_LINES    = True
SHOW_VSLICE_LINES     = True
SHOW_SEISMICITY       = True
SHOW_MT_SITES         = True
SHOW_SEISMIC_SITES    = True
SHOW_VOLCANOES        = True
SHOW_VOLCANOES_ACTIVE = True
SHOW_CITIES           = True
SHOW_NORTH_ARROW      = True

# Profile lines (lon/lat endpoint pairs) — static reference lines, same
# as plot_seis.py. NOT re-verified for Sabancaya (same as PROFILE_LON/LAT
# in VSLICES further below, and ARROW_LON/LAT, VOLC_LABEL_IDX): these are
# Tacna's own coordinates, carried over from plot_seis.py. Unlike
# MAP_XLIM/MAP_YLIM above (which default to None/auto here), these
# directly determine what VSLICES actually samples, so a "saba" run will
# silently slice through the wrong part of the grid until these are
# updated with real Sabancaya coordinates.
PROFILE_CD_LON = [-70.034, -69.670]
PROFILE_CD_LAT = [-17.267, -17.695]
PROFILE_2_LON  = [-69.580, -70.48]
PROFILE_2_LAT  = [-17.135, -18.245]

ARROW_LON    = -73.6
ARROW_LAT    = -18.1
ARROW_LEN_KM = 4.0

# Feature CSV paths
CSV_VOLCANES      = "../features/volcanes.csv"
CSV_SEISMCAT      = "../features/catalog_welllocated_15_simple5.csv"
CSV_MT_SITES      = "../features/done/MTsaba_Sitelist.csv"
# CSV_MT_SITES      = "../features/done/MTTacna_Sitelist.csv"  # TACNA
CSV_CITIES        = "../features/cities.csv"
CSV_SEISMIC_SITES = "../features/seismic_sites.csv"  # no header row; columns
                                                       # are network, station,
                                                       # lat, lon, elev_m

VOLC_LABEL_IDX = [5, 12, 13]
# VOLC_NAME_COL : volcano name column (volcanes.csv). Labels are
# truncated to their first VOLC_LABEL_CHARS characters via
# VOLC_LABEL_STYLE's mode="firstN" (see plotpy.apply_label_mode) rather
# than reading a separate already-abbreviated column (e.g. "VOLCAN2").
VOLC_NAME_COL = "NAME"

# =====================================================================
# MARKER & LABEL STYLE SETTINGS — identical to plot_seis.py
# =====================================================================
PROFILE_CD_STYLE = dict(color="black", lw=0.4, zorder=10)
PROFILE_2_STYLE  = dict(color="gray",  lw=0.4, zorder=10)

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

ARROW_STYLE       = dict(color="dimgray", lw=2, mutation_scale=14)
ARROW_LABEL_STYLE = dict(fontsize=9, fontweight="bold", color="dimgray")

# =====================================================================
# VERTICAL SLICE SETTINGS ("joint" grid mode only — see module docstring)
#
# Same schema as plot_seis.py's VSLICES. Set VSLICES = [] to skip
# all vertical sections regardless of grid mode.
#
# NOT re-verified for Sabancaya: the p1/p2 endpoints below are Tacna's
# own profile lines (copied from plot_seis.py) — see the note above
# PROFILE_CD_LON for why this matters more here than MAP_XLIM/MAP_YLIM.
# =====================================================================
VSLICES = [
    dict(
        name    = "profile AA'",
        p1      = [-70.48, -18.245],
        p2      = [-69.580, -17.135],
        coord   = "latlon",
        zmin_km = -8.0,
        zmax_km = 30.0,
        npts    = 200,
        nz      = 150,
        swath_km= 10.0,
    ),
    dict(
        name    = "profile BB'",
        p1      = [-70.034, -17.267],
        p2      = [-69.670, -17.695],
        coord   = "latlon",
        zmin_km = -8.0,
        zmax_km = 30.0,
        npts    = 200,
        nz      = 150,
        swath_km= 10.0,
    ),
]

VSLICE_VE            = 3.0
VSLICE_EQUAL_SCALE   = False
VSLICE_VE_POS        = "lower right"
VSLICE_VE_STYLE      = dict(fontsize=7, color="black")
VSLICE_X_AXIS        = "distance"   # "utm" | "distance"

VSLICE_EQ_STYLE = dict(
    marker="o", s=4.5, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=11,
)
VSLICE_TOPO_STYLE = dict(color="dimgray", lw=0.5, zorder=12)
VSLICE_SHOW_TOPO_FILL   = False
VSLICE_TOPO_LAND_COLOR  = "gray"
VSLICE_TOPO_OCEAN_COLOR = "#6baed6"
VSLICE_TOPO_HEADROOM_KM = 1.0
VSLICE_MAP_LINE_STYLE   = dict(color="magenta", lw=0.8, ls="--", zorder=15)

# --- Free-text annotation (optional) ---
ANNOTATION_TEXT  = None
ANNOTATION_POS   = (0.01, 0.99)
ANNOTATION_STYLE = dict(fontsize=7, color="gray", ha="left", va="top")

# =====================================================================
# END USER SETTINGS
# =====================================================================

os.makedirs(PLOT_DIR, exist_ok=True)


def ncpath(name):
    """Join a bare precomputed-NetCDF filename onto NC_DIR."""
    return os.path.join(NC_DIR, name)


# ------------------------------------------------------------------
# Colormap / style resolution helpers
# ------------------------------------------------------------------
_cmap_cache = {}


def _resolve_cmap(spec):
    """load_colormap() a spec once, cache by identity of the spec string."""
    key = spec if isinstance(spec, str) else id(spec)
    if key not in _cmap_cache:
        _cmap_cache[key] = load_colormap(spec)
    return _cmap_cache[key]


def _field_style(key):
    """Resolve (cmap, cmin, cmax) for a plain field, applying DEFAULT_CMAP
    if `key` isn't in VAR_STYLE."""
    spec = VAR_STYLE.get(key, {})
    cmap = _resolve_cmap(spec.get("cmap", DEFAULT_CMAP))
    return cmap, spec.get("cmin"), spec.get("cmax")


def _grad_style(grad_key):
    """Resolve (cmap, cmin, cmax) for a gradient field ('{var}_grad_{comp}'),
    applying DEFAULT_GRAD_CMAP if `grad_key` isn't in GRAD_STYLE."""
    spec = GRAD_STYLE.get(grad_key, {})
    cmap = _resolve_cmap(spec.get("cmap", DEFAULT_GRAD_CMAP))
    return cmap, spec.get("cmin"), spec.get("cmax")


def _slice_range(data2d, cmin, cmax):
    """Resolve an explicit/partial/None (cmin, cmax) override into a
    concrete (vmin, vmax) pair, auto-filling either side from the finite
    data range of this particular panel when left as None."""
    if cmin is not None and cmax is not None:
        return cmin, cmax
    finite = data2d[np.isfinite(data2d)]
    if finite.size == 0:
        return (cmin if cmin is not None else 0.0,
                cmax if cmax is not None else 1.0)
    auto_min, auto_max = float(finite.min()), float(finite.max())
    if auto_min == auto_max:
        auto_min -= 0.5
        auto_max += 0.5
    return (cmin if cmin is not None else auto_min,
            cmax if cmax is not None else auto_max)


def _resolve_iso_spec(spec, key):
    """ISO_LEVELS_MAP/ISO_LEVELS_VSLICE may be a single "auto"/list
    (applied to every field) or a dict keyed by field name for per-field
    control. Fields absent from the dict default to "auto"."""
    if isinstance(spec, dict):
        return spec.get(key, "auto")
    return spec


def resolve_iso_levels(data2d, levels_spec, n_auto=ISO_AUTO_N):
    """Resolve an (already per-field-resolved) ISO_LEVELS_* setting into
    an explicit list of contour levels for one panel. "auto"/None picks
    n_auto evenly spaced levels spanning the finite data range of this
    particular panel. Returns [] if there's no usable finite data."""
    if levels_spec is None or (isinstance(levels_spec, str) and levels_spec.lower() == "auto"):
        finite = data2d[np.isfinite(data2d)]
        if finite.size == 0:
            return []
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            return []
        return list(np.linspace(vmin, vmax, n_auto + 2)[1:-1])
    return list(levels_spec)


def draw_iso_contours(ax, x, y, data2d, levels_spec, key, n_auto=ISO_AUTO_N):
    """Overlay isolines of data2d on ax, using ISO_STYLE/ISO_LABEL*. key
    is the field name, used to resolve a per-field entry if levels_spec
    is a dict. x/y may be 1-D (regular "joint" grid) or 2-D matching
    data2d's shape (curvilinear "seismic" grid's own utm_easting_km/
    utm_northing_km) — ax.contour() accepts either. No-op if there are
    no usable levels."""
    spec = _resolve_iso_spec(levels_spec, key)
    levels = resolve_iso_levels(data2d, spec, n_auto)
    if not levels:
        return None
    cs = ax.contour(x, y, data2d, levels=levels, **ISO_STYLE)
    if ISO_LABEL:
        ax.clabel(cs, fmt=ISO_LABEL_FMT, fontsize=ISO_LABEL_FONTSIZE, inline=True)
    return cs


# ------------------------------------------------------------------
# Coordinate helper / hillshade — see plotpy for implementation
# ------------------------------------------------------------------
to_utm_km = plotpy.to_utm_km
compute_hillshade = plotpy.compute_hillshade


def save_fig(fig, stem):
    for fmt in PLOT_FORMATS:
        out = os.path.join(PLOT_DIR, stem + fmt)
        fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {out}")


def _maybe_show():
    """Display the current figure only if SHOW_PLOTS=True *and* matplotlib
    is actually running an interactive backend (mpl.is_interactive() --
    true in Spyder's own console/Qt backend, false for the default
    non-interactive "Agg" backend a plain terminal or batch job gets).
    Saved output (save_fig, called just before this in every caller) is
    unaffected either way -- this only controls the on-screen pop-up,
    never whether files get written."""
    if SHOW_PLOTS and mpl.is_interactive():
        plt.show()


def draw_annotation(ax):
    plotpy.draw_annotation(ax, ANNOTATION_TEXT, ANNOTATION_POS, ANNOTATION_STYLE)


def _region():
    return (xmin, xmax, ymin, ymax)


def _colorbar_settings():
    return dict(show=SHOW_COLORBAR, position=COLORBAR_POSITION,
                size=COLORBAR_SIZE, pad=COLORBAR_PAD, aspect=COLORBAR_ASPECT,
                label_size=COLORBAR_LABEL_SIZE, tick_size=COLORBAR_TICK_SIZE,
                nticks=COLORBAR_NTICKS, title_size=AXIS_TITLE_SIZE)


_resolve_ve_pos = plotpy.resolve_ve_pos


def _in_region(xe, yn):
    return plotpy.in_region(xe, yn, _region())


def clipped_markers(ax, xe, yn, **kwargs):
    plotpy.clipped_markers(ax, xe, yn, _region(), **kwargs)


def clipped_labels(ax, xe, yn, labels, style_dict):
    plotpy.clipped_labels(ax, xe, yn, labels, style_dict, _region())


def draw_north_arrow(ax, x_km, y_km, length_km=4.0):
    plotpy.draw_north_arrow(ax, x_km, y_km, _region(),
                             ARROW_STYLE, ARROW_LABEL_STYLE, length_km)


def create_map_figure():
    map_w_in = FIG_WIDTH / 2.54
    map_h_in = map_w_in * (ymax - ymin) / (xmax - xmin)
    return plotpy.build_panel_figure(map_w_in, map_h_in, _colorbar_settings(),
                                      size_label="map")


def _build_dual_panel_figure(panel_w_in, panel_h_in, colorbar, gap_in=0.6,
                              size_label="panel"):
    """
    Sibling of plotpy.build_panel_figure(): lays out TWO identically-sized
    panels (each panel_w_in × panel_h_in), each with its own colorbar per
    `colorbar` settings, in one figure — used to put a field and its
    gradient magnitude next to each other (see PAIR_GRADIENT_WITH_FIELD
    in the settings above). Not in plotpy.py itself since it's only
    needed here; mirrors build_panel_figure's inch-exact placement math,
    just duplicated per panel and offset by one panel-plus-colorbar
    "block" size plus gap_in.

    Orientation is chosen automatically from the SINGLE panel's own
    aspect ratio, not fixed: panel_w_in >= panel_h_in (a wide/landscape
    single panel, e.g. a broad map or a long vertical section) stacks
    the two blocks vertically (2 rows × 1 column) so the combined figure
    doesn't end up implausibly wide; a tall/portrait single panel
    (panel_h_in > panel_w_in) places them side by side (1 row × 2
    columns) so the combined figure doesn't end up implausibly tall.
    Either way the two panels keep exactly their usual single-panel size
    — only how they're arranged changes.

    Returns (fig, ax_1, cax_1, ax_2, cax_2) — reading order is left-then-
    right for the side-by-side case, top-then-bottom for the stacked
    case. cax_* is None if colorbar["show"] is False, same convention as
    plotpy.build_panel_figure.
    """
    show = colorbar["show"]
    pos = colorbar["position"].lower()
    if pos not in ("right", "left", "bottom", "top"):
        raise ValueError(
            f"colorbar position {colorbar['position']!r} is not valid. "
            "Choose 'right', 'left', 'bottom', or 'top'."
        )

    pad_in = colorbar["pad"]
    if show and pos != "right":
        tick_size = colorbar.get("tick_size", 7)
        label_size = colorbar.get("label_size", 8)
        title_size = colorbar.get("title_size", 9)
        text_in = lambda *sizes: sum(sizes) / 72.0 * 1.6 + 0.05
        extra_clearance = {
            "left":   text_in(tick_size, label_size),
            "bottom": text_in(tick_size, label_size),
            "top":    text_in(title_size),
        }[pos]
        pad_in += extra_clearance

    bar_len_in = bar_thick_in = 0.0
    cbar_w_in = cbar_h_in = 0.0
    if show:
        if pos in ("right", "left"):
            bar_len_in = colorbar["size"] * panel_h_in
            cbar_w_in = bar_thick_in = bar_len_in / colorbar["aspect"]
        else:
            bar_len_in = colorbar["size"] * panel_w_in
            cbar_h_in = bar_thick_in = bar_len_in / colorbar["aspect"]

    # One "block" = one panel + its own colorbar space — same accounting
    # as build_panel_figure's fig_w_in/fig_h_in for a single panel.
    block_w_in = panel_w_in + (cbar_w_in + pad_in if cbar_w_in else 0.0)
    block_h_in = panel_h_in + (cbar_h_in + pad_in if cbar_h_in else 0.0)

    stacked = panel_w_in >= panel_h_in   # wide panel -> stack (2,1); tall -> side by side (1,2)
    if stacked:
        fig_w_in = block_w_in
        fig_h_in = 2 * block_h_in + gap_in
        # First block on top, so it reads first (top-to-bottom).
        block_origins = [(0.0, block_h_in + gap_in), (0.0, 0.0)]
        layout_label = "stacked (2×1)"
    else:
        fig_w_in = 2 * block_w_in + gap_in
        fig_h_in = block_h_in
        block_origins = [(0.0, 0.0), (block_w_in + gap_in, 0.0)]
        layout_label = "side-by-side (1×2)"

    print(f"Figure size ({size_label}, {layout_label} pair): "
          f"{fig_w_in:.2f} × {fig_h_in:.2f} in "
          f"(each panel {panel_w_in:.2f} × {panel_h_in:.2f} in)")

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    axes, caxes = [], []
    for bx_in, by_in in block_origins:
        panel_left_in = bx_in + \
            ((cbar_w_in + pad_in) if (show and pos == "left") else 0.0)
        panel_bottom_in = by_in + \
            ((cbar_h_in + pad_in) if (show and pos == "bottom") else 0.0)
        panel_left = panel_left_in / fig_w_in
        panel_bottom = panel_bottom_in / fig_h_in
        panel_w_frac = panel_w_in / fig_w_in
        panel_h_frac = panel_h_in / fig_h_in
        ax = fig.add_axes([panel_left, panel_bottom, panel_w_frac, panel_h_frac])
        axes.append(ax)

        cax = None
        if show:
            bar_len_frac = (bar_len_in / fig_h_in) if pos in ("right", "left") \
                else (bar_len_in / fig_w_in)
            if pos == "right":
                cax = fig.add_axes([
                    (bx_in + panel_w_in + pad_in) / fig_w_in,
                    panel_bottom + (panel_h_frac - bar_len_frac) / 2,
                    cbar_w_in / fig_w_in, bar_len_frac,
                ])
            elif pos == "left":
                cax = fig.add_axes([
                    bx_in / fig_w_in,
                    panel_bottom + (panel_h_frac - bar_len_frac) / 2,
                    cbar_w_in / fig_w_in, bar_len_frac,
                ])
            elif pos == "top":
                cax = fig.add_axes([
                    panel_left + (panel_w_frac - bar_len_frac) / 2,
                    (by_in + panel_h_in + pad_in) / fig_h_in,
                    bar_len_frac, cbar_h_in / fig_h_in,
                ])
            elif pos == "bottom":
                cax = fig.add_axes([
                    panel_left + (panel_w_frac - bar_len_frac) / 2,
                    by_in / fig_h_in,
                    bar_len_frac, cbar_h_in / fig_h_in,
                ])
        caxes.append(cax)

    return fig, axes[0], caxes[0], axes[1], caxes[1]


def create_map_figure_pair():
    map_w_in = FIG_WIDTH / 2.54
    map_h_in = map_w_in * (ymax - ymin) / (xmax - xmin)
    return _build_dual_panel_figure(map_w_in, map_h_in, _colorbar_settings(),
                                     gap_in=PAIR_GAP_CM / 2.54, size_label="map")


def create_section_figure_pair(w_in, h_in):
    return _build_dual_panel_figure(w_in, h_in, _vslice_colorbar_settings(),
                                     gap_in=PAIR_GAP_CM / 2.54, size_label="section")


def _vslice_colorbar_settings():
    settings = _colorbar_settings()
    ref_len_in = (VSLICE_HEIGHT_CM if VSLICE_HEIGHT_CM is not None
                  else VSLICE_WIDTH_CM) / 2.54
    settings["thickness_in"] = (COLORBAR_SIZE * ref_len_in) / COLORBAR_ASPECT
    return settings


def create_section_figure(w_in, h_in):
    return plotpy.build_panel_figure(w_in, h_in, _vslice_colorbar_settings(),
                                      size_label="section")


def finish_panel_colorbar(cax, mappable, label):
    return plotpy.finish_panel_colorbar(cax, mappable, label, _colorbar_settings())


def finish_section_colorbar(cax, mappable, label):
    return plotpy.finish_panel_colorbar(cax, mappable, label, _vslice_colorbar_settings())


def add_latlon_ticks(ax):
    plotpy.add_latlon_ticks(ax, _region(), LATLON_NTICKS, LATLON_DECIMALS,
                             AXIS_LABEL_SIZE, AXIS_TICK_SIZE)


_profile_utm_km = plotpy.profile_utm_km
_profile_labels = plotpy.profile_labels
_sample_profile_points = plotpy.sample_profile_points


# ==================================================================
# Read interpolate.py's output
# ==================================================================
interp_path = ncpath(INTERP_FILE)
print(f"Reading interpolated grid: {interp_path} …")
interp_ds = xr.open_dataset(interp_path)

grid_mode = interp_ds.attrs["target_grid_mode"]        # "joint" | "seismic"
dim_names = tuple(interp_ds.attrs["dim_names"].split(", "))
dim_depth, dim_row, dim_col = dim_names
interp_vars_available = interp_ds.attrs["interp_vars"].split(", ")
gradient_computed = interp_ds.attrs.get("gradient_computed", "False") == "True"
gradient_vars_available = (
    interp_ds.attrs["gradient_vars"].split(", ") if gradient_computed else []
)

if PLOT_VARS is None:
    active_vars = list(interp_vars_available)
else:
    active_vars = list(PLOT_VARS)
    missing = [v for v in active_vars if v not in interp_ds.data_vars]
    if missing:
        raise KeyError(
            f"PLOT_VARS {missing} not present in {INTERP_FILE!r} (it has "
            f"{list(interp_ds.data_vars)}) — add them to interpolate.py's "
            f"INTERP_VARS and re-run it first."
        )

print(f"Plotting: {active_vars}  (grid_mode={grid_mode!r}, from {INTERP_FILE!r})")

if PLOT_GRADIENT:
    if not gradient_computed:
        print("  PLOT_GRADIENT=True but INTERP_FILE has no gradients "
              "(interpolate.py's COMPUTE_GRADIENT was False) — "
              "no gradient panels will be produced.")
    else:
        grad_vars_present = [v for v in active_vars if v in gradient_vars_available]
        skipped = [v for v in active_vars if v not in gradient_vars_available]
        if skipped:
            print(f"  Note: no gradient computed for {skipped} "
                  f"(not in interpolate.py's GRADIENT_VARS) — skipping "
                  f"gradient panels for those.")
        print(f"  Gradient panels: {grad_vars_present} × {GRADIENT_COMPONENTS}")
else:
    grad_vars_present = []

d_axis = interp_ds[dim_depth].values.astype(np.float64)

_loaded = {}
resolved_units = {}
for key in active_vars:
    da = interp_ds[key]
    _loaded[key] = da.values.astype(np.float32)
    resolved_units[key] = da.attrs.get("units", "")
    print(f"    {key!r}: units={resolved_units[key]!r}")

_grad_loaded = {}
resolved_grad_units = {}
if PLOT_GRADIENT and gradient_computed:
    for key in grad_vars_present:
        for comp in GRADIENT_COMPONENTS:
            grad_key = f"{key}_grad_{comp}"
            if grad_key not in interp_ds.data_vars:
                print(f"    WARNING: {grad_key!r} not found in {INTERP_FILE!r} "
                      f"— skipping.")
                continue
            da = interp_ds[grad_key]
            _grad_loaded[grad_key] = da.values.astype(np.float32)
            resolved_grad_units[grad_key] = da.attrs.get("units", "")

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

grid_shape = _loaded[active_vars[0]].shape  # (depth, row, col)
interp_ds.close()
print(f"Grid shape {dim_names}: {grid_shape}")

if grid_mode != "joint" and VSLICES:
    print(f"\nNOTE: grid_mode={grid_mode!r} — vertical sections (VSLICES) "
          f"are only supported for grid_mode='joint' (a genuinely regular "
          f"UTM grid). Skipping all {len(VSLICES)} VSLICES entries. Re-run "
          f"interpolate.py with TARGET_GRID='joint' if you need "
          f"sections.")


# ==================================================================
# Load static grids
# ==================================================================
print("\nLoading topo grid …")
_topo_da = xr.open_dataarray(ncpath(NC_TOPO))
topo_x = _topo_da["x"].values
topo_y = _topo_da["y"].values
topo_z = _topo_da.values
_topo_da.close()

dx_km = float(np.median(np.diff(topo_x)))
dy_km = float(np.median(np.diff(topo_y)))

if SHOW_TOPO_BASEMAP:
    print("Computing hillshade …")
    topo_hs = compute_hillshade(topo_z, dx_km, dy_km, HS_AZIMUTH, HS_ALTITUDE, HS_SIGMA)
else:
    topo_hs = None

print("Loading bathymetry grid …")
_bath_da = xr.open_dataarray(ncpath(NC_BATH))
bath_x = _bath_da["x"].values
bath_y = _bath_da["y"].values
bath_z = _bath_da.values
_bath_da.close()

topo_extent = [topo_x.min(), topo_x.max(), topo_y.min(), topo_y.max()]
bath_extent = [bath_x.min(), bath_x.max(), bath_y.min(), bath_y.max()]
topo_norm   = mcolors.Normalize(vmin=TOPO_VMIN, vmax=TOPO_VMAX) if SHOW_TOPO_BASEMAP else None
CMAP_TOPO   = plt.get_cmap("gray")


# ==================================================================
# Map region — from the interpolated grid itself (unlike
# plot_seis.py's REGION_SOURCE, there's only one sensible source
# here: whatever INTERP_FILE actually covers)
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


# ==================================================================
# Feature layers
# ==================================================================
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
for i in range(len(volcanes)):
    if "ACT" in str(volcanes["ESTADO"][i]):
        ae, an = to_utm_km([volcanes["LONG"][i]], [volcanes["LAT"][i]])
        volc_act_e.append(ae[0])
        volc_act_n.append(an[0])

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

cities = pd.read_csv(CSV_CITIES)
cit_e, cit_n = to_utm_km(cities["x"].values, cities["y"].values)
name_cit = cities["Name"].values

prof_cd_e, prof_cd_n = to_utm_km(PROFILE_CD_LON, PROFILE_CD_LAT)
prof2_e,   prof2_n   = to_utm_km(PROFILE_2_LON,  PROFILE_2_LAT)
arr_e,     arr_n     = to_utm_km([ARROW_LON], [ARROW_LAT])


# ==================================================================
# Basemap and feature drawing
# ==================================================================
def draw_basemap(ax):
    """Topo greyscale + hillshade + ocean fill; enforce map limits."""
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    if SHOW_TOPO_BASEMAP:
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
    ax.set_xlabel("Easting (km)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Northing (km)", fontsize=AXIS_LABEL_SIZE)
    if AXES_UNITS == "km" and AXES_KM_COMMA:
        _comma_fmt = mpl.ticker.StrMethodFormatter("{x:,.0f}")
        ax.xaxis.set_major_formatter(_comma_fmt)
        ax.yaxis.set_major_formatter(_comma_fmt)
    ax.tick_params(labelsize=AXIS_TICK_SIZE)


def draw_features(ax, eq_e, eq_n):
    """Overlay all feature layers; all markers/labels clipped to map region."""

    if SHOW_PROFILE_LINES:
        ax.plot(prof_cd_e, prof_cd_n, clip_on=True, **PROFILE_CD_STYLE)
        ax.plot(prof2_e,   prof2_n,   clip_on=True, **PROFILE_2_STYLE)

    if SHOW_VSLICE_LINES and grid_mode == "joint":
        for vi, vs in enumerate(VSLICES):
            ve_ends, vn_ends = _profile_utm_km(vs)
            lbl_start, lbl_end = _profile_labels(vi)
            ax.plot(ve_ends, vn_ends, clip_on=True,
                    label=vs.get("name", "slice"), **VSLICE_MAP_LINE_STYLE)
            for xy, lbl in zip(zip(ve_ends, vn_ends), (lbl_start, lbl_end)):
                if _in_region(np.array([xy[0]]), np.array([xy[1]]))[0]:
                    ax.text(xy[0], xy[1], lbl, fontsize=AXIS_TICK_SIZE, fontweight="bold",
                            color=VSLICE_MAP_LINE_STYLE["color"],
                            ha="center", va="bottom", clip_on=True, zorder=16)

    if SHOW_SEISMICITY:
        clipped_markers(ax, eq_e, eq_n, label="Seismicity", **EQ_MARKER_STYLE)

    if SHOW_MT_SITES:
        clipped_markers(ax, mt_e, mt_n, label="MT site", **MT_MARKER_STYLE)
        clipped_labels(ax, mt_e, mt_n, mt_names, MT_LABEL_STYLE)

    if SHOW_SEISMIC_SITES:
        clipped_markers(ax, seis_site_e, seis_site_n, label="Seismic site",
                        **SEISMIC_SITES_MARKER_STYLE)

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
# Depth-slice map: one field, one depth level
# ==================================================================
def _render_depth_slice_panel(ax, cax, field3d, iz, key, cmap, cmin, cmax,
                               cbar_label, title, eq_e, eq_n):
    """Draw one depth-slice panel (basemap + data + features + colorbar +
    title) onto a GIVEN ax/cax — the shared rendering core behind both
    plot_depth_slice_map() (one panel, one figure) and
    plot_depth_slice_pair() (two panels, one figure). Does not create a
    figure or save/show/close anything — callers own the figure
    lifecycle. Rendering split between imshow (regular "joint" grid) and
    pcolormesh(shading="nearest") (curvilinear "seismic" grid) mirrors
    cluster.py's _draw_cluster_overlay()."""
    data2d = field3d[iz].astype(float)
    vmin, vmax = _slice_range(data2d, cmin, cmax)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    draw_basemap(ax)

    if grid_mode == "joint":
        im = ax.imshow(
            data2d, cmap=cmap, norm=norm, origin="lower",
            extent=[e_axis.min(), e_axis.max(), n_axis.min(), n_axis.max()],
            alpha=1.0 - ALPHA_DATA, aspect="equal",
            interpolation="bilinear", zorder=5,
        )
        iso_x, iso_y = e_axis, n_axis
    else:
        im = ax.pcolormesh(
            utm_easting_2d, utm_northing_2d, data2d,
            cmap=cmap, norm=norm, shading="nearest",
            alpha=1.0 - ALPHA_DATA, zorder=5,
        )
        iso_x, iso_y = utm_easting_2d, utm_northing_2d

    if ISO_LINES_MAP:
        draw_iso_contours(ax, iso_x, iso_y, data2d, ISO_LEVELS_MAP, key)

    draw_features(ax, eq_e, eq_n)
    ax.set_title(title, fontsize=AXIS_TITLE_SIZE)
    finish_panel_colorbar(cax, im, cbar_label)
    if AXES_UNITS == "latlon":
        add_latlon_ticks(ax)
    draw_annotation(ax)


def plot_depth_slice_map(field3d, iz, actual_depth, key, cmap, cmin, cmax,
                          cbar_label, title, stem, eq_e, eq_n):
    """Render and save ONE depth-slice map for `field3d[iz]` as a
    single-panel figure — used whenever there's no gradient magnitude to
    pair it with (see plot_depth_slice_pair for the paired/standard
    case)."""
    fig, ax, cax = create_map_figure()
    _render_depth_slice_panel(ax, cax, field3d, iz, key, cmap, cmin, cmax,
                               cbar_label, title, eq_e, eq_n)
    save_fig(fig, stem)
    _maybe_show()
    plt.close(fig)


def plot_depth_slice_pair(field3d, grad3d, iz, key, grad_key,
                           cmap, cmin, cmax, cbar_label, title,
                           gcmap, gcmin, gcmax, gcbar_label, gtitle,
                           stem, eq_e, eq_n):
    """Render and save ONE paired depth-slice figure: `field3d[iz]` first,
    `grad3d[iz]` (its gradient magnitude) second — the standard layout
    (PAIR_GRADIENT_WITH_FIELD) whenever a field's gradient magnitude was
    computed and requested. Arranged side by side or stacked depending
    on the single panel's own aspect ratio — see create_map_figure_pair
    / _build_dual_panel_figure."""
    fig, ax_1, cax_1, ax_2, cax_2 = create_map_figure_pair()
    _render_depth_slice_panel(ax_1, cax_1, field3d, iz, key, cmap, cmin, cmax,
                               cbar_label, title, eq_e, eq_n)
    _render_depth_slice_panel(ax_2, cax_2, grad3d, iz, grad_key, gcmap, gcmin, gcmax,
                               gcbar_label, gtitle, eq_e, eq_n)
    save_fig(fig, stem)
    _maybe_show()
    plt.close(fig)


# ==================================================================
# Vertical slice engine ("joint" grid mode only)
# ==================================================================
def compute_vertical_slice_joint(vslice, field3d):
    """
    Sample a field (dims: depth, northing, easting, on the regular
    "joint" UTM-km grid) along a vertical profile, via a 3-D
    RegularGridInterpolator directly in UTM space — simpler than
    plot_seis.py's own lat/lon-based sampler, since this grid is
    already in UTM (no reprojection needed per query point).

    Returns
    -------
    dist_km, depth_km, section, e_ends, n_ends, topo_prof, utm_x, utm_xlabel
    — same signature as plot_seis.py's compute_vertical_slice_seis().
    """
    e_ends, n_ends = _profile_utm_km(vslice)
    npts = vslice.get("npts", 200)
    nz   = vslice.get("nz",   150)
    zmin = vslice.get("zmin_km", float(d_axis[0]))
    zmax = vslice.get("zmax_km", float(d_axis[-1]))

    dist_km, e_pts, n_pts, utm_x, utm_xlabel = \
        _sample_profile_points(e_ends, n_ends, npts)

    interp = RegularGridInterpolator(
        (d_axis, n_axis, e_axis), field3d.astype(np.float64),
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    depth_km = np.linspace(zmin, zmax, nz)
    d_q = np.repeat(depth_km, npts)
    n_q = np.tile(n_pts, nz)
    e_q = np.tile(e_pts, nz)
    pts = np.column_stack([d_q, n_q, e_q])
    section = interp(pts).reshape(nz, npts)

    topo_prof = None
    if topo_z is not None:
        topo_interp = RegularGridInterpolator(
            (topo_y, topo_x), topo_z,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        topo_prof = topo_interp(np.column_stack([n_pts, e_pts]))

    return dist_km, depth_km, section, e_ends, n_ends, topo_prof, utm_x, utm_xlabel


def _project_seismicity_to_profile(e_ends, n_ends, swath_km, zmin_km, zmax_km):
    return plotpy.project_points_to_profile(
        eq_e0, eq_n0, e_ends, n_ends, swath_km, z0=zeqs,
        zmin_km=zmin_km, zmax_km=zmax_km)


def _render_vertical_slice_panel(ax, cax, x_arr, x_label, depth_km, section,
                                  eq_x, eq_dep, topo_prof, y_top, vslice,
                                  key, cmap, cmin, cmax, cbar_label, title,
                                  lbl_start, lbl_end, ve):
    """Draw one vertical-section panel (VE label + data + topo fill +
    seismicity + colorbar + title + profile-endpoint labels) onto a
    GIVEN ax/cax — the shared rendering core behind both
    plot_vertical_slice() (one panel, one figure) and
    plot_vertical_slice_pair() (two panels, one figure). Does not create
    a figure or save/show/close — callers own the figure lifecycle."""
    if ve != 1.0:
        vx, vy, vha, vva = _resolve_ve_pos(VSLICE_VE_POS)
        ax.text(vx, vy, f"VE = {ve:.1f}×",
                transform=ax.transAxes, ha=vha, va=vva,
                zorder=21, **VSLICE_VE_STYLE)

    vmin, vmax = _slice_range(section, cmin, cmax)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(
        x_arr, depth_km, section,
        cmap=cmap, norm=norm, shading="gouraud",
        alpha=1.0 - ALPHA_DATA, zorder=5,
    )
    # Gouraud shading (not shading="auto" + set_rasterized(True)) — see
    # plot_seis.py's own note: rasterized artists render upside-down
    # under matplotlib's PDF/PS backends combined with invert_yaxis().

    if ISO_LINES_VSLICE:
        draw_iso_contours(ax, x_arr, depth_km, section, ISO_LEVELS_VSLICE, key)

    if topo_prof is not None:
        surf_depth = -topo_prof / 1e3
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

    if SHOW_SEISMICITY and len(eq_x):
        plotpy.markers(ax, eq_x, eq_dep, **VSLICE_EQ_STYLE)

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

    ax.set_title(title, fontsize=AXIS_TITLE_SIZE)
    finish_section_colorbar(cax, im, cbar_label)
    draw_annotation(ax)


def plot_vertical_slice(dist_km, depth_km, section, e_ends, n_ends,
                        topo_prof, utm_x, utm_xlabel,
                        lbl_start, lbl_end,
                        vslice, key, cmap, cmin, cmax, cbar_label,
                        title, stem):
    """Produce and save ONE vertical cross-section as a single-panel
    figure — used whenever there's no gradient magnitude to pair it with
    (see plot_vertical_slice_pair for the paired/standard case)."""
    swath  = vslice.get("swath_km", 10.0)
    zmin_s = vslice.get("zmin_km", depth_km[0])
    zmax_s = vslice.get("zmax_km", depth_km[-1])
    ve     = 1.0 if VSLICE_EQUAL_SCALE else VSLICE_VE

    if VSLICE_X_AXIS == "distance":
        x_arr, x_label = dist_km, "Distance along profile (km)"
    else:
        x_arr, x_label = utm_x, utm_xlabel

    eq_dist, eq_dep = _project_seismicity_to_profile(
        e_ends, n_ends, swath, zmin_s, zmax_s)
    eq_x = np.interp(eq_dist, dist_km, x_arr) if len(eq_dist) else eq_dist

    profile_len = dist_km[-1]
    depth_range = depth_km[-1] - depth_km[0]
    if VSLICE_HEIGHT_CM is not None:
        h_in = VSLICE_HEIGHT_CM / 2.54
        w_in = h_in * profile_len / (depth_range * ve)
    else:
        w_in = VSLICE_WIDTH_CM / 2.54
        h_in = w_in * (depth_range * ve) / profile_len
    print(f"  Section figure size: {w_in:.2f} × {h_in:.2f} in")

    y_top = depth_km[0]
    if topo_prof is not None:
        y_top = (-topo_prof / 1e3).min() - VSLICE_TOPO_HEADROOM_KM

    fig, ax, cax = create_section_figure(w_in, h_in)
    _render_vertical_slice_panel(ax, cax, x_arr, x_label, depth_km, section,
                                  eq_x, eq_dep, topo_prof, y_top, vslice,
                                  key, cmap, cmin, cmax, cbar_label, title,
                                  lbl_start, lbl_end, ve)
    save_fig(fig, stem)
    _maybe_show()
    plt.close(fig)


def plot_vertical_slice_pair(dist_km, depth_km, section, grad_section,
                             e_ends, n_ends, topo_prof, utm_x, utm_xlabel,
                             lbl_start, lbl_end, vslice, key, grad_key,
                             cmap, cmin, cmax, cbar_label, title,
                             gcmap, gcmin, gcmax, gcbar_label, gtitle,
                             stem):
    """Produce and save ONE paired vertical-section figure: `section` (the
    field) first, `grad_section` (its gradient magnitude) second — the
    standard layout (PAIR_GRADIENT_WITH_FIELD) whenever a field's
    gradient magnitude was computed and requested. Arranged side by side
    or stacked depending on the single panel's own aspect ratio — see
    create_section_figure_pair / _build_dual_panel_figure. Both panels
    share the same profile geometry (x-axis, seismicity projection, topo
    profile) since they're sampled along the same VSLICES entry — only
    the plotted field and its colour scale differ."""
    swath  = vslice.get("swath_km", 10.0)
    zmin_s = vslice.get("zmin_km", depth_km[0])
    zmax_s = vslice.get("zmax_km", depth_km[-1])
    ve     = 1.0 if VSLICE_EQUAL_SCALE else VSLICE_VE

    if VSLICE_X_AXIS == "distance":
        x_arr, x_label = dist_km, "Distance along profile (km)"
    else:
        x_arr, x_label = utm_x, utm_xlabel

    eq_dist, eq_dep = _project_seismicity_to_profile(
        e_ends, n_ends, swath, zmin_s, zmax_s)
    eq_x = np.interp(eq_dist, dist_km, x_arr) if len(eq_dist) else eq_dist

    profile_len = dist_km[-1]
    depth_range = depth_km[-1] - depth_km[0]
    if VSLICE_HEIGHT_CM is not None:
        h_in = VSLICE_HEIGHT_CM / 2.54
        w_in = h_in * profile_len / (depth_range * ve)
    else:
        w_in = VSLICE_WIDTH_CM / 2.54
        h_in = w_in * (depth_range * ve) / profile_len
    print(f"  Section figure size (paired): {w_in:.2f} × {h_in:.2f} in each")

    y_top = depth_km[0]
    if topo_prof is not None:
        y_top = (-topo_prof / 1e3).min() - VSLICE_TOPO_HEADROOM_KM

    fig, ax_1, cax_1, ax_2, cax_2 = create_section_figure_pair(w_in, h_in)
    _render_vertical_slice_panel(ax_1, cax_1, x_arr, x_label, depth_km, section,
                                  eq_x, eq_dep, topo_prof, y_top, vslice,
                                  key, cmap, cmin, cmax, cbar_label, title,
                                  lbl_start, lbl_end, ve)
    _render_vertical_slice_panel(ax_2, cax_2, x_arr, x_label, depth_km, grad_section,
                                  eq_x, eq_dep, topo_prof, y_top, vslice,
                                  grad_key, gcmap, gcmin, gcmax, gcbar_label, gtitle,
                                  lbl_start, lbl_end, ve)
    save_fig(fig, stem)
    _maybe_show()
    plt.close(fig)


# ==================================================================
# Label / title / stem helpers
# ==================================================================
def _field_label(key):
    units = resolved_units.get(key, "")
    return f"{key} ({units})" if units else key


def _grad_label(grad_key, base_key, comp):
    units = resolved_grad_units.get(grad_key, "")
    symbol = f"|∇{base_key}|" if comp == "mag" else f"∂{base_key}/∂{comp}"
    return f"{symbol} ({units})" if units else symbol


def _title_suffix():
    """Area + interpolation-method tag appended to every plot title,
    mirroring the same two components already carried in the output
    filenames (see INTERP_TAG above)."""
    return f" [{SITE_PREFIX}, {INTERP_TAG}]"


# ==================================================================
# Main loop — depth-slice maps
# ==================================================================
print("\n=== Depth-slice maps ===")
out_list = []

for i_depth, target_depth in enumerate(PLOT_DEPTHS_KM):
    iz = int(np.argmin(np.abs(d_axis - target_depth)))
    actual_depth = float(d_axis[iz])
    tag = f"{actual_depth:.0f}km" if actual_depth == int(actual_depth) else f"{actual_depth:.1f}km"

    _zmin, _zmax = ZMIN_SEISM[i_depth], ZMAX_SEISM[i_depth]
    if _zmin is None and _zmax is None:
        eq_e, eq_n = eq_e0, eq_n0
    else:
        mask_eqs = np.ones(len(zeqs), dtype=bool)
        if _zmin is not None:
            mask_eqs &= zeqs >= _zmin
        if _zmax is not None:
            mask_eqs &= zeqs <= _zmax
        eq_e, eq_n = eq_e0[mask_eqs], eq_n0[mask_eqs]

    print(f"\nDepth {target_depth} km (nearest available: {actual_depth:.2f} km) …")

    for key in active_vars:
        cmap, cmin, cmax = _field_style(key)
        field_title = f"{key} at {actual_depth:.1f} km{_title_suffix()}"
        stem = f"{key}_joint_{tag}_{SITE_PREFIX}_{INTERP_TAG}"

        mag_key = f"{key}_grad_mag"
        pair_this = (PAIR_GRADIENT_WITH_FIELD and PLOT_GRADIENT
                     and "mag" in GRADIENT_COMPONENTS and mag_key in _grad_loaded)

        if pair_this:
            print(f"  Plotting {key!r} + |∇{key}| (paired) at {actual_depth:.1f} km …")
            gcmap, gcmin, gcmax = _grad_style(mag_key)
            gtitle = f"|∇{key}| at {actual_depth:.1f} km{_title_suffix()}"
            plot_depth_slice_pair(
                _loaded[key], _grad_loaded[mag_key], iz, key, mag_key,
                cmap, cmin, cmax, _field_label(key), field_title,
                gcmap, gcmin, gcmax, _grad_label(mag_key, key, "mag"), gtitle,
                stem, eq_e, eq_n,
            )
        else:
            print(f"  Plotting {key!r} at {actual_depth:.1f} km …")
            plot_depth_slice_map(
                _loaded[key], iz, actual_depth, key, cmap, cmin, cmax,
                _field_label(key), field_title, stem, eq_e, eq_n,
            )
        out_list.append(stem)

        if PLOT_GRADIENT:
            for comp in GRADIENT_COMPONENTS:
                if pair_this and comp == "mag":
                    continue   # already drawn as the right-hand panel above
                grad_key = f"{key}_grad_{comp}"
                if grad_key not in _grad_loaded:
                    continue
                comp_suffix = "" if comp == "mag" else comp
                print(f"    Plotting {grad_key!r} at {actual_depth:.1f} km …")
                gcmap, gcmin, gcmax = _grad_style(grad_key)
                gstem = f"{key}_joint_{tag}_{SITE_PREFIX}_{INTERP_TAG}_grad{comp_suffix}"
                gtitle = (f"|∇{key}| at {actual_depth:.1f} km{_title_suffix()}" if comp == "mag"
                          else f"∂{key}/∂{comp} at {actual_depth:.1f} km{_title_suffix()}")
                plot_depth_slice_map(
                    _grad_loaded[grad_key], iz, actual_depth, grad_key,
                    gcmap, gcmin, gcmax, _grad_label(grad_key, key, comp),
                    gtitle, gstem, eq_e, eq_n,
                )
                out_list.append(gstem)

print("\nDone with depth-slice maps. Output stems:")
for s in out_list:
    print(f"  {s}")


# ==================================================================
# Vertical slices ("joint" grid mode only)
# ==================================================================
if VSLICES and grid_mode == "joint":
    print("\n=== Vertical slices ===")

    for vi, vslice in enumerate(VSLICES):
        name = vslice.get("name", "profile")
        print(f"  Computing section: {name} …")
        lbl_start, lbl_end = _profile_labels(vi)

        for key in active_vars:
            cmap, cmin, cmax = _field_style(key)
            dist_km, depth_km, section, e_ends, n_ends, topo_prof, \
                utm_x, utm_xlabel = compute_vertical_slice_joint(vslice, _loaded[key])
            stem = f"{key}_section_{name}_{SITE_PREFIX}_{INTERP_TAG}"
            field_title = f"{key} — {name}{_title_suffix()}"

            mag_key = f"{key}_grad_mag"
            pair_this = (PAIR_GRADIENT_WITH_FIELD and PLOT_GRADIENT
                         and "mag" in GRADIENT_COMPONENTS and mag_key in _grad_loaded)

            if pair_this:
                print(f"    Computing {key!r} + |∇{key}| (paired) section …")
                gcmap, gcmin, gcmax = _grad_style(mag_key)
                gtitle = f"|∇{key}| — {name}{_title_suffix()}"
                _, _, gsection, _, _, _, _, _ = compute_vertical_slice_joint(
                    vslice, _grad_loaded[mag_key])
                plot_vertical_slice_pair(
                    dist_km, depth_km, section, gsection,
                    e_ends, n_ends, topo_prof, utm_x, utm_xlabel,
                    lbl_start, lbl_end, vslice, key, mag_key,
                    cmap, cmin, cmax, _field_label(key), field_title,
                    gcmap, gcmin, gcmax, _grad_label(mag_key, key, "mag"), gtitle,
                    stem,
                )
            else:
                print(f"    Computing {key!r} section …")
                plot_vertical_slice(
                    dist_km, depth_km, section, e_ends, n_ends,
                    topo_prof, utm_x, utm_xlabel, lbl_start, lbl_end,
                    vslice, key, cmap, cmin, cmax, _field_label(key),
                    field_title, stem,
                )
            out_list.append(stem)

            if PLOT_GRADIENT:
                for comp in GRADIENT_COMPONENTS:
                    if pair_this and comp == "mag":
                        continue   # already drawn as the right-hand panel above
                    grad_key = f"{key}_grad_{comp}"
                    if grad_key not in _grad_loaded:
                        continue
                    comp_suffix = "" if comp == "mag" else comp
                    gcmap, gcmin, gcmax = _grad_style(grad_key)
                    dist_km, depth_km, gsection, e_ends, n_ends, topo_prof, \
                        utm_x, utm_xlabel = compute_vertical_slice_joint(
                            vslice, _grad_loaded[grad_key])
                    gstem = f"{key}_section_{name}_{SITE_PREFIX}_{INTERP_TAG}_grad{comp_suffix}"
                    gtitle = (f"|∇{key}| — {name}{_title_suffix()}" if comp == "mag"
                              else f"∂{key}/∂{comp} — {name}{_title_suffix()}")
                    plot_vertical_slice(
                        dist_km, depth_km, gsection, e_ends, n_ends,
                        topo_prof, utm_x, utm_xlabel, lbl_start, lbl_end,
                        vslice, grad_key, gcmap, gcmin, gcmax,
                        _grad_label(grad_key, key, comp), gtitle, gstem,
                    )
                    out_list.append(gstem)

    print("\nVertical slice stems:")
    for s in out_list:
        if "section" in s:
            print(f"  {s}")

print("\nDone.")
