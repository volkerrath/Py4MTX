#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tacna_plot_modem.py
===================
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

PLOT_FORMATS = [".pdf", ".jpg"]
PLOT_DPI     = 600

# Figure sizes (cm).  Set HEIGHT to None to derive automatically.
HORZ_WIDTH_CM  = 10.0
HORZ_HEIGHT_CM = None   # None → derived from UTM aspect ratio

VSLICE_WIDTH_CM  = 14.0
VSLICE_HEIGHT_CM = None  # None → derived from VE and profile length

# Depth slices to plot — must match values used in tacna_precompute_modem.py.
# Each entry corresponds to one DEPTH_SLICES_KM value; tag strings are
# constructed the same way as in the precompute script.
DEPTH_SLICES_KM = [5.0, 9.0, 13.0,]

# Seismicity depth windows (km), one pair per entry in DEPTH_SLICES_KM.
# Set both to None to show all seismicity on every slice.
ZMIN_SEISM = [-5,  0,  5, 15, 30]
ZMAX_SEISM = [ 5, 10, 20, 30, 55]

# Colour-scale limits for log10(ρ) [Ω·m]; adjust to your model range
CMIN_RHO = 0.5    # log10(Ω·m) — ~3 Ω·m
CMAX_RHO = 4.0    # log10(Ω·m) — ~10,000 Ω·m

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
SENS_BLANK_THRESHOLD = None

# Cells with sensitivity between these two values get a smooth semi-
# transparent overlay fading from fully shaded (SENS_SHADE_MAX_ALPHA, at or
# below the first value) to unshaded (0 alpha, at or above the second) — a
# softer "how much to trust this" cue than a hard blank cutoff. Set to None
# to disable shading. Values are in the same units as SENS_TRANSFORM.
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
# the same units as SENS_TRANSFORM.
SENS_ALPHA_RANGE = None   # e.g. (-2.0, 0.0)

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
# =====================================================================
COLORBAR_POSITION   = "right"   # "right" | "left" | "bottom" | "top"
COLORBAR_SIZE       = 0.05
COLORBAR_PAD        = 0.10      # inches
COLORBAR_ASPECT     = 20
COLORBAR_LABEL_SIZE = 8
COLORBAR_TICK_SIZE  = 7
COLORBAR_NTICKS     = 5

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

# --- Cities CSV (columns x=lon, y=lat, Name) ---
CSV_CITIES = "./features/cities.csv"

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
VOLC_LABEL_STYLE  = dict(fontsize=6, fontweight="bold", color="white",
                          zorder=14, offset_x=0.3, offset_y=0.3)
VOLC_LABEL_STROKE = dict(linewidth=1.5, foreground="black")

VOLC_ACT_MARKER_STYLE = dict(
    marker="^", s=16, facecolors="red", edgecolors="black",
    linewidths=0.2, zorder=13,
)

CITY_MARKER_STYLE = dict(
    marker="s", s=18, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=13,
)
CITY_LABEL_STYLE  = dict(fontsize=6, color="white", zorder=14,
                          offset_x=0.3, offset_y=-0.3)
CITY_LABEL_STROKE = dict(linewidth=1.5, foreground="black")

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
        zmax_km  = 60.0,
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
VSLICE_VE = 1.0

# Horizontal axis for vertical sections:
#   "utm"      — UTM easting or northing (km)
#   "distance" — cumulative distance from p1 (km)
VSLICE_X_AXIS = "utm"

# Seismicity marker style on cross-section
VSLICE_EQ_STYLE = dict(
    s=4, facecolors="white", edgecolors="black",
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
    """Return a [0, 1] hillshade array for a 2-D elevation grid (metres)."""
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    if sigma > 0:
        z2d = gaussian_filter(z2d, sigma=sigma)
    return ls.hillshade(z2d, dx=dx_km * 1e3, dy=dy_km * 1e3, vert_exag=1.0)


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


def draw_sens_shade_overlay(ax, alpha_2d, extent, zorder):
    """Draw a solid-colour overlay (SENS_SHADE_COLOR) whose per-pixel alpha
    comes from alpha_2d, to visually de-emphasise poorly-resolved cells."""
    rgb = mcolors.to_rgb(SENS_SHADE_COLOR)
    rgba = np.zeros((*alpha_2d.shape, 4), dtype=float)
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = alpha_2d
    ax.imshow(rgba, origin="lower", extent=extent, zorder=zorder,
             interpolation="nearest")


def load_sens_depth_slice(tag, ref_shape, ref_northing, ref_easting):
    """
    Load modem_sens_utm_{tag}.nc for the horizontal-slice loop, re-oriented
    to match the resistivity slice's own (northing, easting) orientation.
    Returns None if sensitivity is disabled or the file doesn't exist.
    """
    if not USE_SENSITIVITY:
        return None
    path = f"modem_sens_utm_{tag}.nc"
    if not os.path.exists(path):
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
        out = stem + fmt
        fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {out}")


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
# Colourbar
# ------------------------------------------------------------------
def add_colorbar(fig, ax, mappable, label):
    pos = COLORBAR_POSITION.lower()
    if pos in ("right", "left"):
        orientation = "vertical"
    elif pos in ("bottom", "top"):
        orientation = "horizontal"
    else:
        raise ValueError(f"COLORBAR_POSITION={COLORBAR_POSITION!r} invalid.")
    cbar = fig.colorbar(
        mappable, ax=ax,
        orientation=orientation,
        location=pos,
        fraction=COLORBAR_SIZE,
        pad=COLORBAR_PAD / fig.get_size_inches()[0],
        aspect=COLORBAR_ASPECT,
        shrink=0.85,
    )
    cbar.set_label(label, fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=COLORBAR_NTICKS)
    cbar.update_ticks()
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

    Controlled by SHOW_LATLON_AXES, LATLON_NTICKS, LATLON_DECIMALS.
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


def _sample_profile_points(e_ends, n_ends, npts):
    """
    Evenly spaced points along profile.
    Returns (dist_km, e_pts, n_pts, utm_x, utm_xlabel).
    utm_x is easting when |Δe| >= |Δn|, northing otherwise.
    """
    e_pts   = np.linspace(e_ends[0], e_ends[1], npts)
    n_pts   = np.linspace(n_ends[0], n_ends[1], npts)
    dist_km = np.sqrt((e_pts - e_ends[0])**2 + (n_pts - n_ends[0])**2)

    de = abs(e_ends[1] - e_ends[0])
    dn = abs(n_ends[1] - n_ends[0])
    if de >= dn:
        utm_x, utm_xlabel = e_pts, "Easting (km)"
    else:
        utm_x, utm_xlabel = n_pts, "Northing (km)"

    return dist_km, e_pts, n_pts, utm_x, utm_xlabel


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
    _da  = xr.open_dataarray("modem_model_utm.nc")
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
        method="linear", bounds_error=False, fill_value=np.nan,
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
    if USE_SENSITIVITY and os.path.exists("modem_sens_utm.nc"):
        _sda  = xr.open_dataarray("modem_sens_utm.nc")
        se_ax = _sda["easting"].values
        sn_ax = _sda["northing"].values
        sd_ax = _sda["depth"].values
        svals = _sda.values
        _sda.close()

        if sn_ax[0] > sn_ax[-1]:
            sn_ax = sn_ax[::-1]; svals = svals[:, ::-1, :]
        if se_ax[0] > se_ax[-1]:
            se_ax = se_ax[::-1]; svals = svals[:, :, ::-1]

        sens_interp = RegularGridInterpolator(
            (sd_ax, sn_ax, se_ax), svals,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        sens_section = sens_interp(
            np.column_stack([d_q, n_q, e_q])).reshape(nz, npts)

        if SENS_BLANK_THRESHOLD is not None:
            section = np.where(sens_section < SENS_BLANK_THRESHOLD,
                               np.nan, section)

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


def plot_vertical_slice(dist_km, depth_km, section, e_ends, n_ends,
                        surf_depth, topo_prof, sens_section, utm_x, utm_xlabel,
                        lbl_start, lbl_end,
                        vslice, cmap, cmin, cmax, cbar_label, stem):
    """Produce and save a vertical cross-section figure (UTM km or distance vs depth)."""
    swath  = vslice.get("swath_km", 10.0)
    zmin_s = vslice.get("zmin_km", depth_km[0])
    zmax_s = vslice.get("zmax_km", depth_km[-1])
    ve     = VSLICE_VE
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

    fig, ax = plt.subplots(figsize=(w_in, h_in))
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
    im = ax.pcolormesh(
        x_arr, depth_km, section,
        cmap=cmap, norm=norm, shading="gouraud",
        alpha=data_alpha, zorder=5,
    )
    # NOTE: previously used shading="auto" + set_rasterized(True) to fix
    # vector-format seams between cells, but rasterized artists render
    # upside-down under matplotlib's PDF/PS backends when combined with
    # ax.invert_yaxis() (used below for the depth axis) — a known
    # matplotlib bug, not something in this pipeline. Gouraud shading
    # (smooth per-vertex interpolation, no discrete cell polygons) removes
    # the seams without rasterizing anything, so the inverted axis renders
    # correctly.

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
            cmap=shade_cmap, vmin=0, vmax=1, shading="gouraud",
            alpha=alpha_2d, zorder=6,
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

    if len(eq_x):
        ax.scatter(eq_x, eq_dep, **VSLICE_EQ_STYLE)
    if len(mt_x):
        ax.scatter(mt_x, mt_dep, **VSLICE_MT_STYLE)

    x0, x1 = x_arr[0], x_arr[-1]
    xlim = vslice.get("xlim", None)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(min(x0, x1), max(x0, x1))
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

    if ve != 1.0:
        ax.text(0.99, 0.01, f"VE = {ve:.1f}×",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="gray")

    add_colorbar(fig, ax, im, cbar_label)
    fig.tight_layout()
    save_fig(fig, stem)
    plt.close(fig)

# --- Topography ---
# Prefer the higher-resolution seis-pipeline topo if available
_topo_path = NC_TOPO_SEIS if (NC_TOPO_SEIS and os.path.exists(NC_TOPO_SEIS)) \
             else NC_TOPO_MODEM
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
_modem_topo_da = xr.open_dataarray(NC_TOPO_MODEM)
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
_use_bath = bool(NC_BATH and os.path.exists(NC_BATH))
if _use_bath:
    print(f"Loading bathymetry from: {NC_BATH}")
    _bath_da = xr.open_dataarray(NC_BATH)
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
print(f"Loading MT sites from: {NC_SITES}")
_sites_ds = xr.open_dataset(NC_SITES)
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
_nc0 = f"modem_rho_utm_{_tag0}.nc"
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
print(f"UTM region (km): {utm_region}")

fig_w = HORZ_WIDTH_CM / 2.54
fig_h = fig_w * (ymax - ymin) / (xmax - xmin) \
        if HORZ_HEIGHT_CM is None else HORZ_HEIGHT_CM / 2.54
print(f"Figure size (map): {fig_w:.2f} × {fig_h:.2f} in")

# ==================================================================
# Feature layers (CSV-based)
# ==================================================================
volcanes = pd.read_csv(CSV_VOLCANES)
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
    ax.tick_params(labelsize=7)


def draw_features(ax, eq_e, eq_n):
    if len(prof_cd_e):
        ax.plot(prof_cd_e, prof_cd_n, clip_on=True, **PROFILE_CD_STYLE)
    if len(prof2_e):
        ax.plot(prof2_e, prof2_n, clip_on=True, **PROFILE_2_STYLE)

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

    clipped_scatter(ax, eq_e, eq_n, label="Seismicity", **EQ_MARKER_STYLE)

    # MT sites from NetCDF (already in UTM km)
    clipped_scatter(ax, mt_e, mt_n, label="MT site", **MT_MARKER_STYLE)

    clipped_scatter(ax, utmv_e, utmv_n, **VOLC_INACT_MARKER_STYLE)
    clipped_labels(ax, utmv_e, utmv_n, namev,
                   {**VOLC_LABEL_STYLE, "stroke": VOLC_LABEL_STROKE})

    if volc_act_e:
        clipped_scatter(ax, volc_act_e, volc_act_n,
                        label="Active volcano", **VOLC_ACT_MARKER_STYLE)

    clipped_scatter(ax, cit_e, cit_n, label="City", **CITY_MARKER_STYLE)
    clipped_labels(ax, cit_e, cit_n, name_cit,
                   {**CITY_LABEL_STYLE, "stroke": CITY_LABEL_STROKE})

    draw_north_arrow(ax, arr_e[0], arr_n[0], length_km=ARROW_LEN_KM)


# ==================================================================
# Main loop
# ==================================================================
import matplotlib.ticker   # noqa: E402 — needed for add_colorbar above

out_list = []

for ii, d_km in enumerate(DEPTH_SLICES_KM):
    tag   = f"{d_km:.0f}km" if d_km == int(d_km) else f"{d_km:.1f}km"
    nc    = f"modem_rho_utm_{tag}.nc"
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
            vz = np.where(sens_vz < SENS_BLANK_THRESHOLD, np.nan, vz)

    # Seismicity depth filter
    zmin = ZMIN_SEISM[ii] if ZMIN_SEISM[ii] is not None else -np.inf
    zmax = ZMAX_SEISM[ii] if ZMAX_SEISM[ii] is not None else  np.inf
    mask_eqs = (zeqs > zmin) & (zeqs < zmax)
    eq_e = eq_e0[mask_eqs]
    eq_n = eq_n0[mask_eqs]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
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
    im = ax.imshow(
        vz, cmap=CMAP_RHO, norm=norm, origin="lower",
        extent=[vx.min(), vx.max(), vy.min(), vy.max()],
        alpha=data_alpha,
        aspect="equal", interpolation="bilinear", zorder=5,
    )
    if sens_vz is not None and SENS_SHADE_RANGE is not None:
        alpha_2d = sens_shade_alpha(sens_vz, SENS_SHADE_RANGE[0],
                                    SENS_SHADE_RANGE[1], SENS_SHADE_MAX_ALPHA)
        draw_sens_shade_overlay(ax, alpha_2d,
                               [vx.min(), vx.max(), vy.min(), vy.max()],
                               zorder=6)
    draw_features(ax, eq_e, eq_n)
    ax.set_title(f"log$_{{10}}$ρ at {label}", fontsize=9)
    add_colorbar(fig, ax, im, "log$_{10}$(ρ / Ω·m)")
    if SHOW_LATLON_AXES:
        add_latlon_ticks(ax)
    fig.tight_layout()

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
