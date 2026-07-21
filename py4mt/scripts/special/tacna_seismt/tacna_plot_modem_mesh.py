#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tacna_plot_modem_mesh.py
========================
Companion plotting script for tacna_precompute_modem.py.

Produces depth-slice maps and vertical cross-sections of log10(ρ) (or
linear ρ) from a ModEM 3-D MT inversion result for the Tacna region. Reads
the UTM-km NetCDF files produced by tacna_precompute_modem.py; no
GMT/PyGMT required.

Exact mesh rendering
---------------------
ModEM meshes are non-uniform (dx/dy/dz grow geometrically in the padding
cells outside the fine core region), so this script never resamples the
model onto a uniform pixel grid or interpolates/blends values across cell
boundaries. Every figure is a true, unblended cut through the mesh's own
cells, using the exact cell-edge geometry from modem_grid_edges_utm.nc
(written by tacna_precompute_modem.py):
* Depth slices: pcolormesh(easting_edges, northing_edges, field,
  shading="flat") — each rendered patch is exactly one real mesh cell, at
  its true position and true size.
* Vertical sections: the profile line is intersected with the mesh's own
  grid lines to find the exact sequence of cells it actually crosses (see
  compute_vertical_slice_modem) — each rendered patch is one real 3-D
  cell's value, with true along-profile and true depth boundaries. No
  RegularGridInterpolator, no resampling.

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
        you use the *actual* original palette for an exact visual
        comparison against GMT-produced figures, instead of a same-ish
        matplotlib named stand-in.
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
# in tacna_precompute_modem.py). Default "." reads from the current
# directory, matching the previous (fixed) behaviour.
NC_DIR = "./precompute/"

# Directory for saved figures (created if it doesn't exist). Default "."
# writes into the current directory, matching the previous behaviour.
PLOT_DIR = "./plots/"

# Appended to every saved figure's filename (before the extension) — lets
# output from this script (exact-mesh rendering) be told apart at a
# glance from tacna_plot_modem_image.py's resampled output, e.g.
# "modem_rho_1km_tacna_msh.pdf" vs "..._img.pdf". Set to "" to disable.
PLOT_FILENAME_SUFFIX = "_msh"

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
DEPTH_SLICES_KM = [-3., -1., 0, 3.0, 5.0, 9.0, 13.0,]

# Seismicity depth windows (km), one pair per entry in DEPTH_SLICES_KM.
# Set both to None to show all seismicity on every slice.
# NOTE: reset to "no filter" for every slice below — the previous
# ZMIN_SEISM/ZMAX_SEISM values were already a leftover from an earlier,
# differently-sized DEPTH_SLICES_KM and no longer had a reliable
# correspondence to today's 7 slices, so rather than guess, every slice
# now shows all seismicity by default. Re-add specific (zmin, zmax) pairs
# per slice below if you want per-depth filtering back.
ZMIN_SEISM = [None] * len(DEPTH_SLICES_KM)
ZMAX_SEISM = [None] * len(DEPTH_SLICES_KM)

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
CMAX_RHO = 3.0    # log10(Ω·m) — ~10,000 Ω·m

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
USE_SENSITIVITY = True

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
SENS_SHADE_MAX_ALPHA = 0.99

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
SENS_ALPHA_RANGE = (-3., -3.)
# SENS_ALPHA_RANGE = (-3., -3.)    # more lenient cutoff (sens < 1e-3)
# SENS_ALPHA_RANGE = (-2., -2.)    # stricter cutoff (sens < 1e-2)
#
# Example (b) — soft fade from -2 up to 0: cells fade in gradually as
# sensitivity improves from 1e-2 to 1 (fully resolved), rather than
# snapping straight from invisible to fully opaque.
# SENS_ALPHA_RANGE = (-2., 0.)

# --- Optional standalone sensitivity plots ---
# In addition to using sensitivity to blank/shade/fade the resistivity
# plots above, optionally produce separate maps and sections showing the
# raw sensitivity field itself — same basemap, feature overlays, and
# exact mesh-cell geometry as the resistivity plots, just a different
# field and colour scale. Useful for judging where to set
# SENS_BLANK_THRESHOLD/SENS_ALPHA_RANGE directly, rather than only seeing
# their effect secondhand on the resistivity plot. Requires
# USE_SENSITIVITY and a sensitivity file to actually be available; has no
# effect otherwise.
PLOT_SENSITIVITY_MAPS = True

CMAP_SENS = "jet_r"
# Same units as SENS_TRANSFORM (log10 sensitivity by default). CMAX_SENS
# is left at 0 since log10(sensitivity) can't exceed that (see the
# SENS_ALPHA_RANGE note above) — CMIN_SENS is the one worth tuning, and
# is set wider than SENS_BLANK_THRESHOLD/SENS_ALPHA_RANGE on purpose, so
# this plot shows the falloff you're deciding a cutoff against, not just
# a clipped version of it.
CMIN_SENS = -6.0
CMAX_SENS = 0.0
SENS_CBAR_LABEL = "log$_{10}$(sensitivity)"

# Colourmap: matplotlib built-in name ("jet_r", "RdBu", "turbo_r", "bwr_r",
# etc.), OR a path to a colourmap file to import (.cpt = GMT colour palette
# table, .txt/.csv = plain RGB(A) list) — useful to match a specific
# published palette, or the same palette used in tacna_plot_seis.py, for
# direct visual comparison, e.g.:
#   CMAP_RHO = "./cpt/rho_gmt.cpt"
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
CSV_SEISMCAT = "./features/catalog_welllocated_15_simple5.csv"

# --- Volcanoes CSV ---
CSV_VOLCANES    = "./features/volcanes.csv"
VOLC_LABEL_IDX  = [5, 12, 13]   # row indices to label

# Volcano label text: full name vs. short/abbreviated name.
# VOLC_LABEL_FULL_NAME : True  -> use VOLC_NAME_COL_FULL
#                        False -> use VOLC_NAME_COL_SHORT (default)
VOLC_LABEL_FULL_NAME = False
VOLC_NAME_COL_FULL   = "NAME"      # column used when VOLC_LABEL_FULL_NAME=True
VOLC_NAME_COL_SHORT  = "VOLCAN2"   # column used when VOLC_LABEL_FULL_NAME=False

# --- Cities CSV (columns x=lon, y=lat, Name) ---
CSV_CITIES = "./features/cities.csv"

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
#   swath_km: float — half-width (km) for projecting seismicity onto section
#   xlim    : optional [xmin, xmax] — crop the *displayed* x-axis range
#             without recomputing anything. Units must match VSLICE_X_AXIS
#             below: UTM easting/northing (km) for "utm", or cumulative
#             distance from p1 (km) for "distance". This only narrows the
#             plotted view, so it's cheap to iterate on for fine-tuning a
#             figure. Omit or set to None for the full profile (default).
#
# There is no npts/nz sample-resolution setting: sections are cut using
# the mesh's own exact cell geometry (see compute_vertical_slice_modem),
# so the resolution along the profile and in depth is whatever the real
# mesh has at each point — not a user-chosen sampling density.
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

# --- Free-text annotation (optional) ---
# Draws one extra line of arbitrary text on every figure this script
# produces (both depth slices and vertical sections) — e.g. a version tag,
# a processing note, or a "DRAFT" watermark. Set to None or "" to disable.
# Default position is top-left so it doesn't collide with the VE label,
# which sits top-right on sections.
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
# Coordinate helper (for feature CSVs that are in lon/lat)
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
    """Return a [0, 1] hillshade array for a 2-D elevation grid (metres).

    LightSource.hillshade() computes the surface gradient with
    numpy.gradient, which uses one-sided (lower-quality) differences at
    the true boundary rows/columns of whatever array it's given, instead
    of the centred differences used everywhere else — producing a
    visibly different thin stripe right along the top, bottom, and side
    edges of the rendered hillshade, unrelated to the real terrain.
    Padding the elevation by a few pixels before filtering/shading (using
    an odd/slope-preserving reflection, which continues the local
    gradient rather than mirroring values — plain mirroring introduces
    its own artificial kink right at the seam) and cropping the result
    back down to the original shape gives every true-edge pixel a proper
    two-sided gradient too, removing the stripe.
    """
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    pad = max(int(np.ceil(3 * sigma)), 3) if sigma > 0 else 3
    z_padded = np.pad(z2d, pad, mode="reflect", reflect_type="odd")
    if sigma > 0:
        z_padded = gaussian_filter(z_padded, sigma=sigma)
    hs_padded = ls.hillshade(z_padded, dx=dx_km * 1e3, dy=dy_km * 1e3, vert_exag=1.0)
    return hs_padded[pad:-pad, pad:-pad]


# ------------------------------------------------------------------
# Sensitivity-based shading/blanking
# ------------------------------------------------------------------
def sens_shade_alpha(sens, low, high, max_alpha):
    """
    Map a sensitivity array to a shading alpha in [0, max_alpha]:
    max_alpha at/below `low`, 0 at/above `high`, linearly interpolated
    in between. NaN (missing sensitivity data) is treated as max_alpha —
    conservative, since missing information is not evidence of good
    resolution.
    """
    sens = np.asarray(sens, dtype=float)
    if high == low:
        alpha = np.where(sens <= low, max_alpha, 0.0)
    else:
        frac = np.clip((high - sens) / (high - low), 0.0, 1.0)
        alpha = frac * max_alpha
    return np.where(np.isnan(sens), max_alpha, alpha)


def sens_data_alpha(sens, low, high, base_alpha):
    """
    Map a sensitivity array to a per-cell alpha for the *data layer
    itself* (as opposed to sens_shade_alpha's overlay-on-top alpha): 0
    (fully transparent — whatever is drawn underneath, e.g. the
    topography basemap, shows straight through) at/below `low`,
    base_alpha (the normal data opacity) at/above `high`, linearly
    interpolated in between. NaN (missing sensitivity data) is treated
    as 0 — conservative, same reasoning as blanking.
    """
    sens = np.asarray(sens, dtype=float)
    if high == low:
        alpha = np.where(sens >= high, base_alpha, 0.0)
    else:
        frac = np.clip((sens - low) / (high - low), 0.0, 1.0)
        alpha = frac * base_alpha
    return np.where(np.isnan(sens), 0.0, alpha)


def draw_sens_shade_overlay(ax, e_edges, n_edges, alpha_2d, zorder):
    """Draw a solid-colour overlay (SENS_SHADE_COLOR) whose per-cell alpha
    comes from alpha_2d, to visually de-emphasise poorly-resolved cells.
    Uses the exact (non-uniform) ModEM cell edges, same as the
    resistivity raster."""
    rgb = mcolors.to_rgb(SENS_SHADE_COLOR)
    shade_cmap = mcolors.ListedColormap([rgb])
    ax.pcolormesh(
        e_edges, n_edges, np.zeros_like(alpha_2d),
        cmap=shade_cmap, vmin=0, vmax=1, shading="flat",
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
    return (
        (xe >= xmin) & (xe <= xmax) &
        (yn >= ymin) & (yn <= ymax)
    )


def clipped_scatter(ax, xe, yn, **kwargs):
    mask = _in_region(np.asarray(xe), np.asarray(yn))
    if not np.any(mask):
        return
    ax.scatter(np.asarray(xe)[mask], np.asarray(yn)[mask], **kwargs)


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
        ax.text(x + ox, y + oy, lbl,
                path_effects=path_effects if path_effects else None,
                **style_dict)


# ------------------------------------------------------------------
# North arrow
# ------------------------------------------------------------------
def draw_north_arrow(ax, x_km, y_km, length_km=4.0):
    if not _in_region(np.array([x_km]), np.array([y_km]))[0]:
        return
    ax.annotate("", xy=(x_km, y_km + length_km), xytext=(x_km, y_km),
                arrowprops=dict(arrowstyle="-|>", **ARROW_STYLE),
                annotation_clip=True)
    ax.text(x_km, y_km + length_km + 0.8, "N",
            ha="center", va="bottom", clip_on=True, **ARROW_LABEL_STYLE)


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
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=COLORBAR_NTICKS)
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
    Replace UTM-km tick labels on the primary axes with lon/lat values.
    No extra axes created — existing ticks are reformatted in-place.

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
    """Return (e_km, n_km) endpoint arrays, converting latlon → UTM if needed."""
    p1 = np.asarray(vslice["p1"], float)
    p2 = np.asarray(vslice["p2"], float)
    if vslice.get("coord", "latlon").lower() == "latlon":
        e1, n1 = to_utm_km([p1[0]], [p1[1]])
        e2, n2 = to_utm_km([p2[0]], [p2[1]])
        return np.array([e1[0], e2[0]]), np.array([n1[0], n2[0]])
    else:
        return np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]])


def _profile_labels(index):
    """A/A' for index 0, B/B' for index 1, etc."""
    letter = chr(ord('A') + index)
    return letter, letter + "'"


def _grid_line_crossings(p0, p1, edges):
    """t in (0,1) where the segment p0->p1 crosses each value in `edges`.

    Used to find every point where a profile line crosses a real ModEM
    grid line (a cell boundary) — the basis of the exact section cut.
    """
    d = p1 - p0
    if d == 0:
        return np.array([])
    t = (edges - p0) / d
    return t[(t > 1e-12) & (t < 1.0 - 1e-12)]


def _profile_breakpoints(e1, n1, e2, n2, e_edges, n_edges):
    """
    Sorted t in [0,1] at every grid-line crossing along the profile
    (e1,n1)->(e2,n2), including the endpoints t=0 and t=1. Between any two
    consecutive values the profile stays within a single (i,j) mesh
    column, so each such interval corresponds to one real horizontal cell
    the profile actually passes through.
    """
    ts = np.concatenate([
        [0.0, 1.0],
        _grid_line_crossings(e1, e2, e_edges),
        _grid_line_crossings(n1, n2, n_edges),
    ])
    return np.unique(ts)


def _locate_ij(e, n, e_edges, n_edges):
    """(i, j) mesh indices of the cell containing (e, n), or None if the
    point falls outside the grid. i = northing index, j = easting index
    (ModEM convention — see build_utm_axes in the precompute script)."""
    j = np.searchsorted(e_edges, e, side="right") - 1
    i = np.searchsorted(n_edges, n, side="right") - 1
    if j < 0 or j >= len(e_edges) - 1 or i < 0 or i >= len(n_edges) - 1:
        return None
    return i, j


def _step_lookup(s_query, s_edges, values):
    """Piecewise-constant lookup: the value of whichever segment (defined
    by s_edges, length len(values)+1) contains each point in s_query."""
    idx = np.searchsorted(s_edges, s_query, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def _s_to_xarr(s, L, e_ends, n_ends, mode):
    """Convert along-profile distance(s) (km) to the section's horizontal
    plot coordinate, exactly (the profile is a straight line, so this is
    linear — no lookup table needed)."""
    s = np.asarray(s, dtype=float)
    if mode == "distance" or L == 0:
        return s
    t = np.clip(s / L, 0.0, 1.0)
    if mode == "easting":
        return e_ends[0] + t * (e_ends[1] - e_ends[0])
    else:
        return n_ends[0] + t * (n_ends[1] - n_ends[0])


def compute_vertical_slice_modem(vslice):
    """
    Cut the ModEM resistivity model (modem_model_utm.nc) along a vertical
    profile using the mesh's own, exact cell geometry — no interpolation.

    The profile line is intersected with the mesh's real grid lines
    (modem_grid_edges_utm.nc) to find the exact sequence of (i,j) cell
    columns it passes through; within each such along-profile segment,
    every depth cell is drawn with its own real value and its own real
    depth-edge boundaries (also exact — z edges are the same everywhere
    in x,y for a ModEM tensor mesh, so no approximation is needed there
    either). Every colour in the resulting section is therefore an
    unmodified value from one specific real 3-D cell.

    Returns
    -------
    s_edges    : 1-D (nseg+1,) along-profile distance edges (km)
    d_edges    : 1-D (ndepth+1,) depth-cell edges (km, positive down),
                 restricted to the profile's zmin_km/zmax_km window
    section    : 2-D (ndepth, nseg) exact log10(ρ), one real cell per patch
    e_ends     : 1-D [e0, e1]  UTM km, for map overlay
    n_ends     : 1-D [n0, n1]  UTM km
    surf_depth : 1-D (nseg,)  depth (km) of the top edge of the shallowest
                 valid (non-air) cell in each segment's column — the
                 model's own surface, exact by construction.
    topo_prof  : 1-D (nseg,)  surface elevation (m) from modem_topo_utm.nc
                 at each segment's midpoint, or None — kept only for the
                 land/ocean colour distinction (a real DEM is a genuinely
                 continuous field, so interpolating it is appropriate,
                 unlike the FD mesh's own piecewise-constant cells).
    sens_section : 2-D (ndepth, nseg) exact sensitivity field, or None if
                 USE_SENSITIVITY is False or modem_sens_utm.nc is missing.
    L          : along-profile length (km), for VE/figure-size and for
                 mapping seismicity/MT-site along-profile distances.
    air_valid  : 2-D (ndepth, nseg) boolean, True where the cell is real
                 rock (not air/no-data) — the mask surf_depth was derived
                 from, before any sensitivity-based blanking. Lets a
                 caller mask air cells out of a standalone sensitivity
                 plot without conflating that with sensitivity blanking.
    """
    e_ends, n_ends = _profile_utm_km(vslice)
    e1, e2 = e_ends
    n1, n2 = n_ends
    zmin_s = vslice.get("zmin_km", 0.0)
    zmax_s = vslice.get("zmax_km", 60.0)
    L = float(np.hypot(e2 - e1, n2 - n1))

    # Load the full 3-D model + its exact grid geometry
    _da  = xr.open_dataarray(ncpath("modem_model_utm.nc"))
    e_ax = _da["easting"].values    # km
    n_ax = _da["northing"].values   # km
    vals = _da.values               # (ndepth, nnorthing, neasting)
    _da.close()

    # Ensure axes ascending — vals/e_ax/n_ax are reordered together if
    # needed; grid_e_edges/grid_n_edges/grid_d_edges are always ascending
    # by construction (cumulative sums of positive cell widths), so once
    # vals is in ascending order the two describe the same cells.
    if n_ax[0] > n_ax[-1]:
        vals = vals[:, ::-1, :]
    if e_ax[0] > e_ax[-1]:
        vals = vals[:, :, ::-1]

    e_edges = grid_e_edges
    n_edges = grid_n_edges
    d_edges = grid_d_edges
    if vals.shape != (len(d_edges) - 1, len(n_edges) - 1, len(e_edges) - 1):
        sys.exit(
            f"modem_grid_edges_utm.nc doesn't match modem_model_utm.nc "
            f"(edges describe {len(d_edges)-1}×{len(n_edges)-1}×"
            f"{len(e_edges)-1} cells, model has {vals.shape}) — re-run "
            f"tacna_precompute_modem.py."
        )

    # Restrict the depth range to what's needed (one cell of padding on
    # each side) for the requested zmin_km/zmax_km window.
    k_lo = max(np.searchsorted(d_edges, zmin_s, side="right") - 1, 0)
    k_hi = min(np.searchsorted(d_edges, zmax_s, side="left") + 1,
              len(d_edges) - 1)
    d_edges_c = d_edges[k_lo:k_hi + 1]
    vals_c = vals[k_lo:k_hi, :, :]

    # Exact along-profile breakpoints: every real grid-line crossing
    ts = _profile_breakpoints(e1, n1, e2, n2, e_edges, n_edges)
    n_seg = len(ts) - 1
    s_edges = ts * L

    seg_e_mid = e1 + ((ts[:-1] + ts[1:]) / 2.0) * (e2 - e1)
    seg_n_mid = n1 + ((ts[:-1] + ts[1:]) / 2.0) * (n2 - n1)

    n_depth = vals_c.shape[0]
    section = np.full((n_depth, n_seg), np.nan)
    seg_ij = [None] * n_seg
    for k in range(n_seg):
        ij = _locate_ij(seg_e_mid[k], seg_n_mid[k], e_edges, n_edges)
        seg_ij[k] = ij
        if ij is not None:
            i, j = ij
            section[:, k] = vals_c[:, i, j]

    # Mask air cells (log10(ρ) ≈ 17 for RHO_AIR) so they show as no-data
    # rather than a saturated colour.
    section[section >= AIR_LOG10_RHO_THRESHOLD] = np.nan

    # Surface = the TOP EDGE of the shallowest valid (non-air) cell in
    # each segment's column — exact, since it's a real cell boundary, not
    # an interpolated sample point.
    valid = ~np.isnan(section)
    has_data = valid.any(axis=0)
    first_valid_idx = np.argmax(valid, axis=0)
    surf_depth = np.full(n_seg, d_edges_c[-1])
    surf_depth[has_data] = d_edges_c[first_valid_idx[has_data]]

    # Sensitivity — same exact per-segment column lookup, sharing seg_ij
    # so the two fields stay pixel-for-pixel aligned.
    sens_section = None
    if USE_SENSITIVITY and not os.path.exists(ncpath("modem_sens_utm.nc")):
        print("  WARNING: modem_sens_utm.nc not found — sensitivity "
              "masking/shading is disabled for this section. Check that "
              "tacna_precompute_modem.py found the .sns file (look for its "
              "own WARNING) and that OUTPUT_DIR there matches NC_DIR here.")
    if USE_SENSITIVITY and os.path.exists(ncpath("modem_sens_utm.nc")):
        _sda  = xr.open_dataarray(ncpath("modem_sens_utm.nc"))
        svals = _sda.values
        _sda.close()
        if n_ax[0] > n_ax[-1]:
            svals = svals[:, ::-1, :]
        if e_ax[0] > e_ax[-1]:
            svals = svals[:, :, ::-1]
        svals_c = svals[k_lo:k_hi, :, :]

        sens_section = np.full((n_depth, n_seg), np.nan)
        for k, ij in enumerate(seg_ij):
            if ij is not None:
                i, j = ij
                sens_section[:, k] = svals_c[:, i, j]
        print("  sens min = ", np.nanmin(sens_section))

        if SENS_BLANK_THRESHOLD is not None:
            _blank_mask = sens_section < SENS_BLANK_THRESHOLD
            print(f"  sens blanking: {np.sum(_blank_mask)}/{_blank_mask.size} "
                  f"cells below threshold ({SENS_BLANK_THRESHOLD})")
            section = np.where(_blank_mask, np.nan, section)

    # Topo along profile — a real DEM, genuinely continuous, so
    # interpolating it (at each segment's midpoint) is appropriate; kept
    # only to flag ocean (elevation <= 0) for the bathymetry fill colour,
    # not to position the surface line (surf_depth, above, is exact).
    topo_prof = None
    if modem_topo_z is not None:
        topo_interp = RegularGridInterpolator(
            (modem_topo_y, modem_topo_x), modem_topo_z,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        topo_prof = topo_interp(np.column_stack([seg_n_mid, seg_e_mid]))

    return (s_edges, d_edges_c, section, e_ends, n_ends, surf_depth,
            topo_prof, sens_section, L, valid)


def _project_seismicity_to_profile(e_ends, n_ends, swath_km, zmin_km, zmax_km):
    """Events within swath_km of the profile and within depth range; returns (along_km, depth_km)."""
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
        (across <= swath_km) & (along >= 0) & (along <= L) &
        (zeqs >= zmin_km) & (zeqs <= zmax_km)
    )
    return along[mask], zeqs[mask]


def _project_mt_sites_to_profile(e_ends, n_ends, swath_km):
    """MT sites within swath_km; returns along-profile distance (km).

    Depth is not returned here — MT sites sit on the true surface, not at a
    fixed z=0, so their plotted depth is taken from the section's own
    surf_depth (interpolated at each site's along-profile position) by the
    caller instead.
    """
    de = e_ends[1] - e_ends[0]
    dn = n_ends[1] - n_ends[0]
    L  = np.sqrt(de**2 + dn**2)
    if L == 0:
        return np.array([])
    ue, un = de / L, dn / L
    ve = mt_e - e_ends[0]
    vn = mt_n - n_ends[0]
    along  = ve * ue + vn * un
    across = np.abs(ve * (-un) + vn * ue)
    mask   = (across <= swath_km) & (along >= 0) & (along <= L)
    return along[mask]


def plot_vertical_slice(s_edges, d_edges, section, e_ends, n_ends,
                        surf_depth, topo_prof, sens_section, L,
                        lbl_start, lbl_end,
                        vslice, cmap, cmin, cmax, cbar_label, stem,
                        title_field="log$_{10}$ρ"):
    """Produce and save a vertical cross-section figure (UTM km or distance
    vs depth) as an exact cut through the mesh's real cells — every patch
    is one true 3-D cell, at its true along-profile and true depth
    boundaries (see compute_vertical_slice_modem)."""
    swath  = vslice.get("swath_km", 10.0)
    zmin_s = vslice.get("zmin_km", d_edges[0])
    zmax_s = vslice.get("zmax_km", d_edges[-1])
    ve     = 1.0 if VSLICE_EQUAL_SCALE else VSLICE_VE
    name   = vslice.get("name", "profile")

    # Choose horizontal coordinate mode and convert the exact along-profile
    # edges to it (an exact linear map — the profile is a straight line).
    if VSLICE_X_AXIS == "distance":
        x_mode, x_label = "distance", "Distance along profile (km)"
    else:
        de, dn = abs(e_ends[1] - e_ends[0]), abs(n_ends[1] - n_ends[0])
        if de >= dn:
            x_mode, x_label = "easting", "Easting (km)"
        else:
            x_mode, x_label = "northing", "Northing (km)"
    x_edges = _s_to_xarr(s_edges, L, e_ends, n_ends, x_mode)

    # x_edges must be ascending for pcolormesh/stairs regardless of profile
    # direction (northing/easting can decrease along the profile); reorder
    # section/surf_depth/sens_section together so everything stays aligned.
    if x_edges[0] > x_edges[-1]:
        x_edges = x_edges[::-1]
        section = section[:, ::-1]
        surf_depth = surf_depth[::-1]
        if sens_section is not None:
            sens_section = sens_section[:, ::-1]

    eq_dist, eq_dep = _project_seismicity_to_profile(
        e_ends, n_ends, swath, zmin_s, zmax_s)
    mt_dist = _project_mt_sites_to_profile(e_ends, n_ends, swath)

    # eq_dist/mt_dist are along-profile distances (km) in the same s units
    # as s_edges — look up each one's segment directly (piecewise-constant,
    # exact) rather than interpolating, then convert to the plot's x mode.
    mt_dep = _step_lookup(mt_dist, s_edges, surf_depth) if len(mt_dist) else mt_dist
    eq_x = _s_to_xarr(eq_dist, L, e_ends, n_ends, x_mode)
    mt_x = _s_to_xarr(mt_dist, L, e_ends, n_ends, x_mode)

    # An event can only be real if it's at or below the local surface.
    # Events above it here are a projection artefact (the swath can be wide
    # enough that an event's true, off-profile position has different local
    # relief than the profile line itself) rather than physically "in the
    # air" — but drawing them above the surface line is misleading either
    # way, so they're dropped from this particular figure.
    if len(eq_x):
        local_surf_at_eq = _step_lookup(eq_dist, s_edges, surf_depth)
        keep = eq_dep >= local_surf_at_eq
        eq_x, eq_dep = eq_x[keep], eq_dep[keep]

    profile_len = L
    depth_range = d_edges[-1] - d_edges[0]
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
    # Per-cell alpha for the data layer itself. Sections have no basemap
    # underneath (unlike the depth-slice maps, where ALPHA_RHO lets the
    # topography hillshade show through) — so default to fully opaque,
    # and only reduce alpha when SENS_ALPHA_RANGE is actually in use to
    # fade poorly-resolved cells and let the topo fill/line show through.
    data_alpha = 1.0
    if sens_section is not None and SENS_ALPHA_RANGE is not None:
        data_alpha = sens_data_alpha(sens_section, SENS_ALPHA_RANGE[0],
                                     SENS_ALPHA_RANGE[1], 1.0)
    # Each rendered patch is exactly one real 3-D mesh cell — true
    # along-profile boundaries (x_edges, from the profile/grid-line
    # intersection) and true depth boundaries (d_edges). No interpolation,
    # no smoothing; antialiased=False avoids hairline seams between
    # adjacent patches in vector (PDF/EPS) output.
    im = ax.pcolormesh(
        x_edges, d_edges, section,
        cmap=cmap, norm=norm, shading="flat",
        alpha=data_alpha, zorder=5, antialiased=False,
    )

    if sens_section is not None and SENS_SHADE_RANGE is not None:
        alpha_2d = sens_shade_alpha(sens_section, SENS_SHADE_RANGE[0],
                                    SENS_SHADE_RANGE[1], SENS_SHADE_MAX_ALPHA)
        rgb = mcolors.to_rgb(SENS_SHADE_COLOR)
        shade_cmap = mcolors.ListedColormap([rgb])
        ax.pcolormesh(
            x_edges, d_edges, np.zeros_like(alpha_2d),
            cmap=shade_cmap, vmin=0, vmax=1, shading="flat",
            alpha=alpha_2d, zorder=6, antialiased=False,
        )

    # Surface line/fill come from the section's own exact data (surf_depth
    # — the true top edge of each segment's shallowest real rock cell), so
    # they're drawn as a proper staircase (ax.stairs), not a smoothed
    # curve — the model's own resolution is genuinely blocky, and this
    # shows that honestly rather than implying more precision than the
    # mesh actually has.
    y_top = min(d_edges[0], surf_depth.min() - VSLICE_TOPO_HEADROOM_KM)

    if topo_prof is not None:
        # Ocean fill only (a genuine physical reference — elevation <= 0 —
        # not a masked/no-data region), positioned using surf_depth so it
        # never floats apart from the colour data. step="post" matches
        # surf_depth's piecewise-constant (per-segment) nature.
        ocean = topo_prof <= 0
        if ocean.any():
            ax.fill_between(x_edges[:-1], 0.0, surf_depth,
                            where=ocean, step="post",
                            color=VSLICE_TOPO_OCEAN_COLOR, alpha=0.5,
                            zorder=6, interpolate=False)

    ax.stairs(surf_depth, x_edges, baseline=None, fill=False,
              **VSLICE_TOPO_STYLE)

    if SHOW_SEISMICITY and len(eq_x):
        ax.scatter(eq_x, eq_dep, **VSLICE_EQ_STYLE)
    if SHOW_MT_SITES and len(mt_x):
        ax.scatter(mt_x, mt_dep, **VSLICE_MT_STYLE)

    x0, x1 = x_edges[0], x_edges[-1]
    xlim = vslice.get("xlim", None)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(min(x0, x1), max(x0, x1))
    ax.set_ylim(y_top, d_edges[-1])
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

    ax.set_title(f"{title_field} — {name}  (swath ±{swath} km)", fontsize=9)

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

# True (non-uniform) ModEM cell-edge coordinates, written by
# tacna_precompute_modem.py. Every raster in this script — depth slices
# and vertical sections — is rendered as an exact cut through these real
# mesh cells (pcolormesh + shading="flat"), never resampled onto a
# uniform pixel grid or interpolated across cell boundaries, so this file
# is required rather than optional.
if not os.path.exists(ncpath("modem_grid_edges_utm.nc")):
    sys.exit(
        f"{ncpath('modem_grid_edges_utm.nc')} not found. This script "
        "requires the exact cell-edge geometry written by "
        "tacna_precompute_modem.py — re-run it (with the current version, "
        "and matching OUTPUT_DIR/NC_DIR) before plotting."
    )
_edges_da = xr.open_dataset(ncpath("modem_grid_edges_utm.nc"))
grid_e_edges = _edges_da["easting_edges"].values
grid_n_edges = _edges_da["northing_edges"].values
grid_d_edges = _edges_da["depth_edges"].values
_edges_da.close()
print("Loaded exact mesh cell-edge geometry from modem_grid_edges_utm.nc")

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
    air_mask = vz >= AIR_LOG10_RHO_THRESHOLD
    vz[air_mask] = np.nan

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
    if grid_e_edges.size != vx.size + 1 or grid_n_edges.size != vy.size + 1:
        sys.exit(
            f"modem_grid_edges_utm.nc size ({grid_e_edges.size-1}×"
            f"{grid_n_edges.size-1} cells) doesn't match this slice "
            f"({vx.size}×{vy.size} cells) — re-run "
            f"tacna_precompute_modem.py so the edges file matches the "
            f"current model/crop settings."
        )
    # Each rendered patch is exactly one real ModEM cell, at its true
    # (non-uniform) position and size — see the module docstring.
    im = ax.pcolormesh(
        grid_e_edges, grid_n_edges, vz, cmap=CMAP_RHO, norm=norm,
        alpha=data_alpha, shading="flat", zorder=5,
    )
    if sens_vz is not None and SENS_SHADE_RANGE is not None:
        alpha_2d = sens_shade_alpha(sens_vz, SENS_SHADE_RANGE[0],
                                    SENS_SHADE_RANGE[1], SENS_SHADE_MAX_ALPHA)
        draw_sens_shade_overlay(ax, grid_e_edges, grid_n_edges, alpha_2d, zorder=6)
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

    # --- Optional standalone sensitivity map (same settings as above,
    # different field/colour scale — see PLOT_SENSITIVITY_MAPS) ---
    if PLOT_SENSITIVITY_MAPS and sens_vz is not None:
        print(f"Plotting sensitivity at {label} …")
        sens_vz_masked = np.where(air_mask, np.nan, sens_vz)

        fig_s, ax_s, cax_s = create_map_figure()
        draw_basemap(ax_s)
        norm_s = mcolors.Normalize(vmin=CMIN_SENS, vmax=CMAX_SENS)
        im_s = ax_s.pcolormesh(
            grid_e_edges, grid_n_edges, sens_vz_masked,
            cmap=CMAP_SENS, norm=norm_s,
            alpha=1.0 - ALPHA_RHO, shading="flat", zorder=5,
        )
        draw_features(ax_s, eq_e, eq_n)
        ax_s.set_title(f"Sensitivity at {label}", fontsize=9)
        finish_panel_colorbar(cax_s, im_s, SENS_CBAR_LABEL)
        if AXES_UNITS == "latlon":
            add_latlon_ticks(ax_s)
        draw_annotation(ax_s)

        stem_s = f"modem_sens_map_{tag}_tacna"
        save_fig(fig_s, stem_s)
        plt.close(fig_s)
        out_list.append(stem_s)

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
        s_edges, d_edges, section, e_ends, n_ends, surf_depth, topo_prof, \
            sens_section, L, air_valid = compute_vertical_slice_modem(vslice)
        stem = f"modem_section_{name}_tacna"
        plot_vertical_slice(
            s_edges, d_edges, section, e_ends, n_ends,
            surf_depth, topo_prof, sens_section, L,
            lbl_start, lbl_end, vslice,
            CMAP_RHO, VSLICE_CMIN_RHO, VSLICE_CMAX_RHO,
            "log$_{10}$(ρ / Ω·m)", stem,
        )
        out_list.append(stem)

        # --- Optional standalone sensitivity section (same settings as
        # above, different field/colour scale — see PLOT_SENSITIVITY_MAPS).
        # sens_section=None on this call disables the sensitivity-based
        # alpha fade, so the sensitivity field is shown fully opaque —
        # fading it by itself would be circular.
        if PLOT_SENSITIVITY_MAPS and sens_section is not None:
            print(f"  Computing sensitivity section: {name} …")
            sens_section_masked = np.where(air_valid, sens_section, np.nan)
            stem_s = f"modem_section_{name}_sens_tacna"
            plot_vertical_slice(
                s_edges, d_edges, sens_section_masked, e_ends, n_ends,
                surf_depth, topo_prof, None, L,
                lbl_start, lbl_end, vslice,
                CMAP_SENS, CMIN_SENS, CMAX_SENS,
                SENS_CBAR_LABEL, stem_s,
                title_field="Sensitivity",
            )
            out_list.append(stem_s)

    print("\nVertical slice stems:")
    for s in out_list:
        if "section" in s:
            print(f"  {s}")
