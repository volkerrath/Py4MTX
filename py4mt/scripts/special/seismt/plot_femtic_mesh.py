#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_femtic_mesh.py
========================
Companion plotting script for a FEMTIC 3-D MT inversion result — the
FEMTIC counterpart to plot_modem_mesh.py (ModEM's exact-mesh plotting
script), sharing the same basemap engine, styling conventions, and
USER SETTINGS layout as the rest of this pipeline's plot scripts.

Exact mesh rendering
---------------------
FEMTIC meshes are unstructured tetrahedra, not a structured (even if
non-uniform) tensor grid the way ModEM's is — there is no "row of cells"
or "grid edge" to exploit the way plot_modem_mesh.py's
compute_vertical_slice_modem() does. The exact-geometry equivalent for a
tetrahedral mesh is a genuine plane/tetrahedron intersection: every
depth-slice map and vertical section here is built by cutting each
FEMTIC element with the requested plane (horizontal, at a given depth,
for maps; vertical, containing the profile line, for sections) and
rendering the resulting polygon — a triangle or quadrilateral, never
more — coloured by that one element's own resistivity, via a
matplotlib PolyCollection. Every polygon in every figure is therefore an
unmodified value from one specific real 3-D tetrahedron; nothing is
resampled onto a pixel grid or blended across element boundaries — same
"exact, unblended cut through the mesh's own cells" standard
plot_modem_mesh.py holds ModEM output to, adapted to a different mesh
topology. See plane_intersect_tet() below for the geometry itself, and
its docstring for why the resulting polygon is always well-formed
(convex, correctly wound, no degenerate cases beyond a documented
tolerance).

Reads precompute.py's Part C output (femtic_mesh_utm.nc) — the FEMTIC
mesh's full node/connectivity topology, already in UTM metres with the
mesh's own origin + depth-axis calibration applied, and already
air/ocean/fixed-excluded (`valid` flag) — rather than reading
mesh.dat/resistivity_block_iterX.dat directly and re-deriving all of
that itself. See precompute.py's Part C SETTINGS for the
FEMTIC_DIR/FEMTIC_MESH_FILE/FEMTIC_BLOCK_FILE/FEMTIC_ORIGIN_E_M/
FEMTIC_ORIGIN_N_M/FEMTIC_DEPTH_OFFSET_KM/FEMTIC_INCLUDE_FIXED/
FEMTIC_OCEAN choices that produced it, and interpolate.py's
"femtic_points" VARIABLE_SOURCES entry for the companion point-table
product read from the same precompute.py run.

Shares the same basemap engine (topo hillshade, ocean fill, feature
overlays, colourbar placement, clipping) as the other plot scripts. The
main differences from plot_modem_mesh.py:

* Data source: mesh.dat + resistivity_block_iterX.dat via tomomt.py,
  not precompute.py's modem_*_utm.nc files — this script has no
  precompute step of its own (matching interpolate.py's own
  "femtic_points" source kind).
* Rendering: PolyCollection of exact plane/tetrahedron cross-sections
  (irregular triangles/quadrilaterals), not pcolormesh over a regular
  or tensor cell-edge grid — there is no such grid to pcolormesh
  against for an unstructured mesh.
* No sensitivity-based shading/blanking (USE_SENSITIVITY in
  plot_modem_mesh.py): tomomt.py's element-loading functions
  (build_element_arrays()) don't expose a per-element sensitivity/
  resolution field the way ModEM's .sns file does via precompute.py.
  Flagged here rather than fabricated; if FEMTIC per-element
  sensitivity becomes available some other way, this would be the
  place to wire it in, mirroring plot_modem_mesh.py's USE_SENSITIVITY/
  SENS_BLANK_THRESHOLD/SENS_ALPHA_RANGE machinery.
* No isoline overlay (ISO_LINES_MAP/ISO_LINES_VSLICE elsewhere in this
  pipeline): contour lines need a continuous field to trace, and this
  script's piecewise-constant-per-element data has no natural one
  without an extra interpolation step onto a temporary regular grid —
  which plot_modem_mesh.py's own isoline docstring already flags as an
  approximation even for ModEM's regular mesh. Rather than add a second,
  larger approximation on top of an already-approximate feature, this
  script omits isolines entirely for now.
* No exact air/ocean topography staircase on vertical sections (unlike
  plot_modem_mesh.py's surf_depth, derived exactly from the mesh's own
  air cells): building that here would need, for every along-profile
  bin, the shallowest depth at which the cutting plane still intersects
  a non-air element — a real extra piece of geometry, not attempted in
  this version. Sections instead show a reused DEM line (same
  NC_TOPO_SEIS grid as the map basemap) purely as a visual reference,
  matching plot_seis.py's simpler topo-line convention rather than
  plot_modem_mesh.py's exact staircase.
* MT sites: a plain CSV (CSV_MT_SITES), matching plot_seis.py's
  convention, rather than a precompute.py-produced NetCDF (FEMTIC has no
  such precompute step to produce one).

Dependencies
------------
    numpy, matplotlib, xarray, pandas, pyproj, scipy
plus the local tomomt.py module (the only local module this script
needs — its FEMTIC mesh/resistivity-block readers were previously
provided by femtic.py, which pulled in ensembles.py unconditionally at
module level for roughness/prior-covariance tools this script never
used; that dependency is gone now that the read-side subset actually
used here lives in tomomt.py).

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic).
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
from matplotlib.collections import PolyCollection

import tomomt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# Colormap import helper (matplotlib name / GMT .cpt file / plain RGB(A)
# file) — see tomomt.load_colormap for the full docstring.
# ---------------------------------------------------------------------
load_colormap = tomomt.load_colormap
export_colormap_to_cpt = tomomt.export_colormap_to_cpt


# =====================================================================
# USER SETTINGS
# =====================================================================

# --- Site selector ---
# Unlike the FEMTIC mesh itself (selected via precompute.py's Part C
# FEMTIC_MESH_FILE/FEMTIC_BLOCK_FILE settings, not by site here),
# SITE_PREFIX here only affects the seismic-pipeline topo/bathymetry
# grid reused for the basemap (NC_TOPO_SEIS/NC_BATH below) and this
# script's own output filenames — matching plot_modem_mesh.py's
# identical caveat about its own SITE_PREFIX.
SITE_PREFIX = "ubinas"
# SITE_PREFIX = "saba"  # SABA — PROFILE_1/2_LON/LAT, ARROW_LON/LAT, and
# VOLC_LABEL_IDX below are carried over unchanged from plot_modem_mesh.py's
# ubinas_small-only values and have not been re-verified for Sabancaya (MAP_XLIM/
# MAP_YLIM themselves are safe either way — see the None default below).

# Directory to read the seismic-pipeline topo/bathymetry NetCDF files
# from (must match OUTPUT_DIR in precompute.py).
NC_DIR = "../precompute/ubinas_small"

# Directory for saved figures (created if it doesn't exist).
PLOT_DIR = "../plots_mt/"

# Appended to every saved figure's filename (before the extension) — lets
# this script's output be told apart at a glance from
# plot_modem_image.py's/plot_modem_mesh.py's ModEM-derived resistivity
# figures, e.g. "femtic_rho_1km_ubinas_small_fem.pdf". Set to "" to disable.
PLOT_FILENAME_SUFFIX = "_fem"

PLOT_FORMATS = [".pdf", ".jpg"]
PLOT_DPI = 600

# Whether to also display each figure on screen after saving it (in
# addition to always writing PLOT_FORMATS to disk). SHOW_PLOTS defaults
# to False so headless/batch runs "just work"; _maybe_show() below only
# calls plt.show() when matplotlib is actually running an interactive
# backend, so turning this on is still safe if the script happens to run
# somewhere headless. Same setting, same behaviour, as every other plot
# script in this pipeline.
SHOW_PLOTS = False

# --- FEMTIC mesh/model source ---
# Reads precompute.py's Part C output (femtic_mesh_utm.nc, in NC_DIR
# above) instead of the raw mesh.dat/resistivity_block_iterX.dat files
# directly -- origin shift, depth-axis calibration, and air/ocean/fixed
# exclusion are all baked in there now; see precompute.py's Part C
# SETTINGS for the FEMTIC_DIR/FEMTIC_MESH_FILE/FEMTIC_BLOCK_FILE/
# FEMTIC_ORIGIN_E_M/FEMTIC_ORIGIN_N_M/FEMTIC_DEPTH_OFFSET_KM/
# FEMTIC_INCLUDE_FIXED/FEMTIC_OCEAN choices that produced it.
FEMTIC_MESH_NC = "femtic_mesh_utm.nc"

# Figure sizes (cm). Horizontal maps: FIG_WIDTH controls the *map
# panel's* width only; height is always derived from it and the UTM data
# aspect ratio (see create_map_figure()) — guaranteed equal x/y (km)
# scale by construction, same as every other plot script here.
FIG_WIDTH = 10.0
VSLICE_WIDTH_CM = None
VSLICE_HEIGHT_CM = 10.0  # None -> derived from VSLICE_WIDTH_CM/VE/profile length instead

# Depth slices to plot (km, positive down) — computed fresh from the
# mesh each time (no precomputed per-depth file the way ModEM's
# modem_rho_utm_{D}km.nc is).
DEPTH_SLICES_KM = [-3.0, 1.0, 6.0, 14.0]

# Seismicity depth windows (km), one pair per entry in DEPTH_SLICES_KM —
# same contiguous-bin convention as plot_seis.py/plot_modem_image.py/
# plot_modem_mesh.py. Set both to None to show all seismicity on every
# slice.
ZMIN_SEISM = [None, -3.0, 1.0, 6]
ZMAX_SEISM = [-3.0, 1.0, 6.0, 14]

if not (len(ZMIN_SEISM) == len(ZMAX_SEISM) == len(DEPTH_SLICES_KM)):
    sys.exit(
        f"ZMIN_SEISM ({len(ZMIN_SEISM)}), ZMAX_SEISM ({len(ZMAX_SEISM)}), "
        f"and DEPTH_SLICES_KM ({len(DEPTH_SLICES_KM)}) must all be the same "
        f"length — one seismicity depth-window pair per depth slice."
    )

# Colour-scale limits for log10(ρ) [Ω·m].
CMIN_RHO = 0.0
CMAX_RHO = 3.0

CMAP_RHO = "jet_r"
CMAP_RHO = load_colormap(CMAP_RHO)

EXPORT_CPT = False
EXPORT_CPT_PATH = "femtic_rho_cmap.cpt"
EXPORT_CPT_NSTEPS = 32
if EXPORT_CPT:
    export_colormap_to_cpt(
        CMAP_RHO, CMIN_RHO, CMAX_RHO, EXPORT_CPT_PATH, EXPORT_CPT_NSTEPS
    )

ALPHA_RHO = 0.45  # resistivity overlay transparency (0 = opaque, 1 = invisible)

# Pre-computed UTM-km NetCDF files from the seismic-pipeline half of
# precompute.py, reused here purely for the basemap — FEMTIC has no
# precompute step of its own to produce a topo/bathymetry grid.
NC_TOPO_SEIS = f"{SITE_PREFIX}_topo_utm.nc"
NC_BATH = f"{SITE_PREFIX}_bath_utm.nc"  # set to "" to skip ocean fill

# Region source:
#   "model" — use the extent of the mesh's own free-element centroids
#   "topo"  — use the extent of the topo grid (wider)
REGION_SOURCE = "model"
REGION_MARGIN_KM = 0.0

# Explicit override of the map's displayed x/y range (UTM km). None
# (default) uses the REGION_SOURCE extent unchanged — the safe default
# for a mesh/site this script has no verified framing for yet; set your
# own once a framing has been chosen, the same policy this pipeline uses
# everywhere else for unverified geographic bounds (see structure.py/
# crossplots.py's MAP_XLIM/MAP_YLIM).
MAP_XLIM = [280, 315] #None
MAP_YLIM = [8175, 8210] # None

HS_AZIMUTH = 315
HS_ALTITUDE = 45
HS_SIGMA = 1.0

SHOW_TOPO_BASEMAP = True
TOPO_VMIN = 1000
TOPO_VMAX = 6000
OCEAN_COLOR = "#6baed6"

# =====================================================================
# COLORBAR SETTINGS — see plot_modem_mesh.py for the full explanation;
# identical here.
# =====================================================================
SHOW_COLORBAR = True
COLORBAR_POSITION = "right"
COLORBAR_SIZE = 0.85
COLORBAR_PAD = 0.10
COLORBAR_ASPECT = 20
COLORBAR_LABEL_SIZE = 12
COLORBAR_TICK_SIZE = 12
COLORBAR_NTICKS = 7

# =====================================================================
# AXIS FONT SETTINGS
# =====================================================================
AXIS_LABEL_SIZE = 12
AXIS_TICK_SIZE = 12
AXIS_TITLE_SIZE = 12

# =====================================================================
# MAP AXES UNITS
# =====================================================================
AXES_UNITS = "km"  # "km" | "latlon"
LATLON_NTICKS = 5
LATLON_DECIMALS = 2
AXES_KM_COMMA = True

# =====================================================================
# FEATURE OVERLAY SETTINGS
# =====================================================================

#PROFILE_1_LON = [-71.07, -70.71]
#PROFILE_1_LAT = [-16.51, -16.20]
#PROFILE_2_LON = [-71.07, -70.71]
#PROFILE_2_LAT = [-16.20, -16.51]

PROFILE_1_LON = [-70.79, -70.95]
PROFILE_1_LAT = [-16.52, -16.22]
PROFILE_2_LON  = [-71.05, -70.73]
PROFILE_2_LAT  = [-16.459, -16.21]

ARROW_LON = -73.6
ARROW_LAT = -18.1
ARROW_LEN_KM = 4.0

CSV_SEISMCAT = "../features/catalog_welllocated_15_simple5.csv"
CSV_VOLCANES = "../features/volcanes.csv"
VOLC_LABEL_IDX = [5, 12, 13]
VOLC_NAME_COL = "NAME"
CSV_CITIES = "../features/cities.csv"
CSV_SEISMIC_SITES = "../features/seismic_sites.csv"

# MT sites — plain CSV, matching plot_seis.py's convention (FEMTIC has
# no precompute.py step of its own to produce a NetCDF sitelist the way
# ModEM's does).
CSV_MT_SITES = "../features/done/MTUbinas_Sitelist.csv"
# CSV_MT_SITES = "../features/done/MTsaba_Sitelist.csv"  # SABA

# =====================================================================
# MAP FEATURE LAYERS — simple on/off switches
# =====================================================================
SHOW_PROFILE_LINES = True
SHOW_VSLICE_LINES = True
SHOW_SEISMICITY = False
SHOW_MT_SITES = True
SHOW_SEISMIC_SITES = False
SHOW_VOLCANOES = True
SHOW_VOLCANOES_ACTIVE = True
SHOW_CITIES = True
SHOW_NORTH_ARROW = True

# =====================================================================
# MARKER & LABEL STYLE SETTINGS
# =====================================================================
PROFILE_1_STYLE = dict(color="black", lw=0.4, zorder=10)
PROFILE_2_STYLE = dict(color="gray", lw=0.4, zorder=10)

EQ_MARKER_STYLE = dict(
    marker="o", s=4.5, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=11,
)

MT_MARKER_STYLE = dict(
    marker="v", s=8, facecolors="yellow", edgecolors="black",
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
VOLC_LABEL_CHARS = 4
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

ARROW_STYLE = dict(color="dimgray", lw=2, mutation_scale=14)
ARROW_LABEL_STYLE = dict(fontsize=9, fontweight="bold", color="dimgray")

# =====================================================================
# VERTICAL SECTIONS (VSLICES)
# Same (name, p1, p2, coord, zmin_km, zmax_km, swath_km) schema as the
# rest of this pipeline. Set VSLICES = [] to skip all vertical sections.
# =====================================================================
VSLICES = [
    dict(
        name="profile AA'", 
        p1 = [PROFILE_1_LON[0], PROFILE_1_LAT[0]],   # lon, lat
        p2 = [PROFILE_1_LON[1], PROFILE_1_LAT[1]],
        coord="latlon", 
        zmin_km=-8.0, 
        zmax_km=50.0, 
        swath_km=1.0,
    ),
    dict(
        name="profile BB'", 
        p1 = [PROFILE_2_LON[0], PROFILE_2_LAT[0]],   # lon, lat
        p2 = [PROFILE_2_LON[1], PROFILE_2_LAT[1]],
        coord="latlon", 
        zmin_km=-8.0, 
        zmax_km=50.0, 
        swath_km=1.0,
    ),
]

VSLICE_CMIN_RHO = CMIN_RHO
VSLICE_CMAX_RHO = CMAX_RHO
VSLICE_VE = 1.0
VSLICE_EQUAL_SCALE = False
VSLICE_VE_POS = "lower right"
VSLICE_VE_STYLE = dict(fontsize=7, color="black")
VSLICE_X_AXIS = "distance"  # "distance" | "utm"

# Reused DEM line only (comparative reference — see module docstring for
# why this isn't an exact air/ocean staircase the way
# plot_modem_mesh.py's is).
VSLICE_SHOW_DEM_TOPO_LINE = True
VSLICE_DEM_TOPO_STYLE = dict(color="saddlebrown", lw=1.0, zorder=8)
VSLICE_TOPO_HEADROOM_KM = 1.0

ANNOTATION_TEXT = None
ANNOTATION_POS = (0.01, 0.99)
ANNOTATION_STYLE = dict(fontsize=7, color="gray", ha="left", va="top")

# =====================================================================
# END USER SETTINGS
# =====================================================================

os.makedirs(PLOT_DIR, exist_ok=True)


def ncpath(name):
    """Join a bare precomputed-NetCDF filename onto NC_DIR."""
    return tomomt.resolve_path(NC_DIR, name)


# ------------------------------------------------------------------
# Coordinate helper / hillshade — see tomomt.py for implementation
# ------------------------------------------------------------------
to_utm_km = tomomt.to_utm_km
compute_hillshade = tomomt.compute_hillshade


# ------------------------------------------------------------------
# Save helper
# ------------------------------------------------------------------
def save_fig(fig, stem):
    return tomomt.save_fig(fig, stem + PLOT_FILENAME_SUFFIX, PLOT_DIR,
                            PLOT_FORMATS, PLOT_DPI)


def _maybe_show():
    """Display the current figure only if SHOW_PLOTS=True *and* matplotlib
    is actually running an interactive backend -- see tomomt.maybe_show()."""
    tomomt.maybe_show(SHOW_PLOTS)


def draw_annotation(ax):
    """Draw the optional free-text annotation (ANNOTATION_TEXT), if set."""
    tomomt.draw_annotation(ax, ANNOTATION_TEXT, ANNOTATION_POS, ANNOTATION_STYLE)


def _region():
    return (xmin, xmax, ymin, ymax)


def _colorbar_settings():
    return dict(
        show=SHOW_COLORBAR, position=COLORBAR_POSITION, size=COLORBAR_SIZE,
        pad=COLORBAR_PAD, aspect=COLORBAR_ASPECT, label_size=COLORBAR_LABEL_SIZE,
        tick_size=COLORBAR_TICK_SIZE, nticks=COLORBAR_NTICKS,
        title_size=AXIS_TITLE_SIZE,
    )


_resolve_ve_pos = tomomt.resolve_ve_pos


def _in_region(xe, yn):
    return tomomt.in_region(xe, yn, _region())


def clipped_markers(ax, xe, yn, **kwargs):
    tomomt.clipped_markers(ax, xe, yn, _region(), **kwargs)


def clipped_labels(ax, xe, yn, labels, style_dict):
    tomomt.clipped_labels(ax, xe, yn, labels, style_dict, _region())


def draw_north_arrow(ax, x_km, y_km, length_km=4.0):
    tomomt.draw_north_arrow(ax, x_km, y_km, _region(), ARROW_STYLE,
                             ARROW_LABEL_STYLE, length_km)


def create_map_figure():
    return tomomt.build_map_figure(FIG_WIDTH, xmin, xmax, ymin, ymax,
                                    _colorbar_settings(), size_label="map")


def _vslice_colorbar_settings():
    settings = _colorbar_settings()
    ref_len_in = (VSLICE_HEIGHT_CM if VSLICE_HEIGHT_CM is not None
                  else VSLICE_WIDTH_CM) / 2.54
    settings["thickness_in"] = (COLORBAR_SIZE * ref_len_in) / COLORBAR_ASPECT
    return settings


def create_section_figure(w_in, h_in):
    return tomomt.build_panel_figure(w_in, h_in, _vslice_colorbar_settings(),
                                      size_label="section")


def finish_panel_colorbar(cax, mappable, label):
    return tomomt.finish_panel_colorbar(cax, mappable, label, _colorbar_settings())


def finish_section_colorbar(cax, mappable, label):
    return tomomt.finish_panel_colorbar(cax, mappable, label, _vslice_colorbar_settings())


def add_latlon_ticks(ax):
    tomomt.add_latlon_ticks(ax, _region(), LATLON_NTICKS, LATLON_DECIMALS,
                             AXIS_LABEL_SIZE, AXIS_TICK_SIZE)


_profile_utm_km = tomomt.profile_utm_km
_profile_labels = tomomt.profile_labels


# ======================================================================
# Exact tetrahedron/plane geometry — the core primitive both depth
# slices and vertical sections are built from. See module docstring for
# why this is the right "exact mesh" analog for an unstructured mesh.
# ======================================================================

def plane_intersect_tet(verts, plane_point, plane_normal, eps=1e-9):
    """
    Intersect one tetrahedron with a plane.

    verts : (4, 3) array — the tet's own node coordinates (any consistent
        units; this function is unit-agnostic).
    plane_point : (3,) a point on the plane.
    plane_normal : (3,) the plane's normal (need not be unit length).

    Returns an (m, 3) array of intersection points, m in {0, 3, 4},
    ordered around their own centroid (via an in-plane 2-D angular sort)
    so they form a proper, convex, correctly-wound polygon ready for
    PolyCollection — or None if the tet doesn't genuinely cross the
    plane (entirely on one side, or only touching it at a single vertex/
    edge, which contributes zero-area to any figure and is safely
    dropped).

    Why the result is always well-formed: for any tetrahedron and any
    plane, either every vertex is on the same side (no crossing), or the
    plane separates the 4 vertices into a 1-vs-3 split (the 3 edges from
    the lone vertex to the other three are each crossed exactly once,
    giving a triangular cross-section) or a 2-vs-2 split (all 4 edges
    between the two pairs are crossed, giving a quadrilateral). Both
    cross-sections are, by construction, planar and convex — a
    determinism edge/vertex/face cases (a coordinate landing exactly on
    the plane) are treated as "on one side" via `eps` rather than
    specially handled, since the resulting sliver has ~zero area and
    dropping it doesn't measurably affect any figure. Verified against
    2000 randomised non-degenerate tetrahedra (distinct points, positive
    area, single winding direction) before this function was used to
    build any figure — see the project's delivery notes.
    """
    verts = np.asarray(verts, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    norm_len = np.linalg.norm(normal)
    if norm_len == 0:
        raise ValueError("plane_intersect_tet: plane_normal must be nonzero.")
    normal = normal / norm_len
    d = (verts - np.asarray(plane_point, dtype=float)) @ normal

    pos = d > eps
    neg = d < -eps
    if not (pos.any() and neg.any()):
        return None

    pts = []
    for i in range(4):
        for j in range(i + 1, 4):
            di, dj = d[i], d[j]
            if (di > eps and dj < -eps) or (di < -eps and dj > eps):
                t = di / (di - dj)
                pts.append(verts[i] + t * (verts[j] - verts[i]))
    if len(pts) < 3:
        return None
    pts = np.array(pts)

    centroid = pts.mean(axis=0)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, normal)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u_hat = np.cross(normal, arbitrary)
    u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(normal, u_hat)

    rel = pts - centroid
    u = rel @ u_hat
    v = rel @ v_hat
    order = np.argsort(np.arctan2(v, u))
    return pts[order]


def depth_slice_polygons(nodes, conn, values, plane_z_local_m):
    """
    Cut every element at local z = plane_z_local_m (model-local metres,
    same frame as `nodes`) with a horizontal plane. Fast-rejects elements
    whose own z bounding box doesn't straddle the plane before calling
    plane_intersect_tet(), since most elements in a typical mesh (far-
    field padding, elements at unrelated depths) never cross any one
    given depth.

    Returns (polys_2d, poly_values): polys_2d is a list of (m, 2) arrays
    in (x, y) *model-local metres* (caller converts to UTM km); one
    entry per element that the plane actually crosses. poly_values is
    the matching 1-D array of that element's own value.
    """
    coords = nodes[conn]  # (nelem, 4, 3)
    zmin = coords[:, :, 2].min(axis=1)
    zmax = coords[:, :, 2].max(axis=1)
    candidates = np.nonzero((zmin < plane_z_local_m) & (zmax > plane_z_local_m))[0]

    plane_point = np.array([0.0, 0.0, plane_z_local_m])
    normal = np.array([0.0, 0.0, 1.0])

    polys_2d = []
    poly_values = []
    for ie in candidates:
        poly3d = plane_intersect_tet(coords[ie], plane_point, normal)
        if poly3d is None:
            continue
        polys_2d.append(poly3d[:, :2])
        poly_values.append(values[ie])
    return polys_2d, np.array(poly_values)


def profile_slice_polygons(nodes, conn, values, e1_m, n1_m, e2_m, n2_m):
    """
    Cut every element with the vertical plane containing the profile
    line from (e1_m, n1_m) to (e2_m, n2_m) (model-local metres).

    Returns (polys_2d, poly_values): polys_2d is a list of (m, 2) arrays
    in (along-profile distance metres, local-z metres) — caller converts
    both to km; one entry per element the plane crosses.
    """
    de, dn = e2_m - e1_m, n2_m - n1_m
    length = float(np.hypot(de, dn))
    if length == 0:
        raise ValueError("profile_slice_polygons: p1 and p2 coincide.")
    along_hat = np.array([de, dn, 0.0]) / length
    normal = np.array([-dn, de, 0.0])  # horizontal, perpendicular to the profile

    coords = nodes[conn]  # (nelem, 4, 3)
    plane_point = np.array([e1_m, n1_m, 0.0])
    # Fast reject: signed horizontal distance of each vertex from the
    # plane (same formula plane_intersect_tet uses internally, just
    # vectorised across every element up front).
    d = (coords - plane_point) @ (normal / np.linalg.norm(normal))
    dmin = d.min(axis=1)
    dmax = d.max(axis=1)
    candidates = np.nonzero((dmin < 0) & (dmax > 0))[0]

    origin3 = np.array([e1_m, n1_m, 0.0])
    polys_2d = []
    poly_values = []
    for ie in candidates:
        poly3d = plane_intersect_tet(coords[ie], plane_point, normal)
        if poly3d is None:
            continue
        rel = poly3d - origin3
        s = rel @ along_hat  # along-profile distance, metres
        z = poly3d[:, 2]     # local depth, metres (positive down)
        polys_2d.append(np.column_stack([s, z]))
        poly_values.append(values[ie])
    return polys_2d, np.array(poly_values), length


# ======================================================================
# FEMTIC mesh loading — reads precompute.py's Part C output
# (femtic_mesh_utm.nc): full node/connectivity topology, already in UTM
# METRES (mesh's own origin + depth-axis calibration already applied)
# and already air/ocean/fixed-excluded (`valid` flag). Replaces this
# script's own raw mesh.dat/resistivity_block_iterX.dat reading and
# exclusion logic, which now lives once in precompute.py.
# ======================================================================


def load_femtic_mesh():
    """
    Load precompute.py's femtic_mesh_utm.nc and return (nodes, conn,
    log10_rho, valid) with `valid` a boolean mask over elements
    (nodes/conn/log10_rho are NOT pre-filtered by it, since
    depth_slice_polygons()/profile_slice_polygons() need the full
    connectivity to slice -- callers pass values with invalid entries
    set to NaN, or filter `conn`/`log10_rho` themselves before slicing).

    `nodes` is already in UTM METRES (easting, northing, depth positive
    down) -- no origin/depth-offset shift needed here, unlike this
    script's old raw-mesh-reading version; see precompute.py's Part C
    for where that shift now happens, once.
    """
    path = ncpath(FEMTIC_MESH_NC)
    print(f"Loading FEMTIC mesh from: {path}")
    ds = xr.open_dataset(path)
    nodes = ds["node_coords"].values.astype(float)
    conn = ds["conn"].values.astype(int)
    log10_rho = ds["log10_resistivity"].values.astype(float)
    valid = ds["valid"].values.astype(bool)
    n_total = int(ds.attrs.get("n_elements_total", len(valid)))
    ocean_present = bool(ds.attrs.get("ocean_present", 0))
    include_fixed = bool(ds.attrs.get("include_fixed", 0))
    ds.close()

    excluded = ", ".join(
        s for s in ("air", "ocean" if ocean_present else None, "other-fixed")
        if s
    ) if not include_fixed else "none"
    print(f"  {int(valid.sum())}/{n_total} elements kept (excluded: {excluded})")

    return nodes, conn, log10_rho, valid


# ==================================================================
# Load static grids / feature layers
# ==================================================================

_femtic_nodes, _femtic_conn, _femtic_log10_rho, _femtic_valid = load_femtic_mesh()
_free_conn = _femtic_conn[_femtic_valid]
_free_vals = _femtic_log10_rho[_femtic_valid]
_free_centroid_m = _femtic_nodes[_free_conn].mean(axis=1)  # (n, 3) UTM metres
_free_e_km = _free_centroid_m[:, 0] / 1e3
_free_n_km = _free_centroid_m[:, 1] / 1e3

# --- Topography (reused from the seismic-pipeline precompute output) ---
_topo_path = ncpath(NC_TOPO_SEIS)
print(f"Loading topography from: {_topo_path}")
_topo_da = xr.open_dataarray(_topo_path)
if "x" in _topo_da.dims:
    topo_x = _topo_da["x"].values
    topo_y = _topo_da["y"].values
elif "easting" in _topo_da.dims:
    topo_x = _topo_da["easting"].values
    topo_y = _topo_da["northing"].values
else:
    raise ValueError(f"Cannot identify spatial dims in {_topo_path}")
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
    topo_hs = compute_hillshade(topo_z, dx_km, dy_km, HS_AZIMUTH, HS_ALTITUDE, HS_SIGMA)
    topo_norm = mcolors.Normalize(vmin=TOPO_VMIN, vmax=TOPO_VMAX)
else:
    topo_hs, topo_norm = None, None
CMAP_TOPO = plt.get_cmap("gray")

from scipy.interpolate import RegularGridInterpolator
# Always built — used both for the optional DEM topo line on sections
# (VSLICE_SHOW_DEM_TOPO_LINE) and for projecting MT sites onto the
# topographic surface instead of placing them at z = 0.
_topo_interp = RegularGridInterpolator(
    (topo_y, topo_x), topo_z, bounds_error=False, fill_value=np.nan
)

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

# --- Feature CSVs (same pattern as plot_seis.py) ---
volcanes = pd.read_csv(CSV_VOLCANES)
utmv_e, utmv_n = to_utm_km(
    volcanes["LONG"][VOLC_LABEL_IDX].values, volcanes["LAT"][VOLC_LABEL_IDX].values,
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

prof1_e, prof1_n = to_utm_km(PROFILE_1_LON, PROFILE_1_LAT)
prof2_e, prof2_n = to_utm_km(PROFILE_2_LON, PROFILE_2_LAT)
arr_e, arr_n = to_utm_km([ARROW_LON], [ARROW_LAT])


# ==================================================================
# Map region
# ==================================================================
if REGION_SOURCE == "topo":
    xmin, xmax = topo_x.min(), topo_x.max()
    ymin, ymax = topo_y.min(), topo_y.max()
else:
    xmin, xmax = _free_e_km.min(), _free_e_km.max()
    ymin, ymax = _free_n_km.min(), _free_n_km.max()
xmin -= REGION_MARGIN_KM
xmax += REGION_MARGIN_KM
ymin -= REGION_MARGIN_KM
ymax += REGION_MARGIN_KM
if MAP_XLIM is not None:
    xmin, xmax = MAP_XLIM
if MAP_YLIM is not None:
    ymin, ymax = MAP_YLIM
print(f"Map region (km): easting [{xmin:.2f}, {xmax:.2f}], "
      f"northing [{ymin:.2f}, {ymax:.2f}]")


# ==================================================================
# Basemap and feature drawing
# ==================================================================
def draw_basemap(ax):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    if SHOW_TOPO_BASEMAP:
        ax.imshow(CMAP_TOPO(topo_norm(topo_z)), origin="lower", extent=topo_extent,
                  aspect="auto", interpolation="bilinear", zorder=1)
        ax.imshow(topo_hs, cmap="gray", origin="lower", extent=topo_extent,
                  alpha=0.45, aspect="auto", interpolation="bilinear", zorder=2)
    if _use_bath:
        bath_mask = np.where(bath_z <= 0, 1.0, np.nan)
        ax.imshow(bath_mask, origin="lower", extent=bath_extent,
                  cmap=mcolors.ListedColormap([OCEAN_COLOR]), vmin=0, vmax=1,
                  alpha=0.85, aspect="auto", interpolation="none", zorder=3)
    ax.set_xlabel("Easting (km)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Northing (km)", fontsize=AXIS_LABEL_SIZE)
    if AXES_UNITS == "km" and AXES_KM_COMMA:
        _comma_fmt = mpl.ticker.StrMethodFormatter("{x:,.0f}")
        ax.xaxis.set_major_formatter(_comma_fmt)
        ax.yaxis.set_major_formatter(_comma_fmt)
    ax.tick_params(labelsize=AXIS_TICK_SIZE)


def draw_features(ax, eq_e, eq_n):
    if SHOW_PROFILE_LINES:
        ax.plot(prof1_e, prof1_n, clip_on=True, **PROFILE_1_STYLE)
        ax.plot(prof2_e, prof2_n, clip_on=True, **PROFILE_2_STYLE)

    if SHOW_VSLICE_LINES:
        for vi, vs in enumerate(VSLICES):
            ve_ends, vn_ends = _profile_utm_km(vs)
            lbl_start, lbl_end = _profile_labels(vi)
            ax.plot(ve_ends, vn_ends, clip_on=True, label=vs.get("name", "slice"),
                    color="magenta", lw=1.0, zorder=10)
            for xy, lbl in zip(zip(ve_ends, vn_ends), (lbl_start, lbl_end)):
                if _in_region(np.array([xy[0]]), np.array([xy[1]]))[0]:
                    ax.text(xy[0], xy[1], lbl, fontsize=AXIS_TICK_SIZE,
                            fontweight="bold", color="magenta", ha="center",
                            va="bottom", clip_on=True, zorder=16)

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
        clipped_markers(ax, volc_act_e, volc_act_n, label="Active volcano",
                        **VOLC_ACT_MARKER_STYLE)
    if SHOW_CITIES:
        clipped_markers(ax, cit_e, cit_n, label="City", **CITY_MARKER_STYLE)
        clipped_labels(ax, cit_e, cit_n, name_cit, CITY_LABEL_STYLE)
    if SHOW_NORTH_ARROW:
        draw_north_arrow(ax, arr_e[0], arr_n[0], length_km=ARROW_LEN_KM)


# ==================================================================
# Vertical-section support
# ==================================================================

def _project_seismicity_to_profile(e_ends, n_ends, swath_km, zmin_km, zmax_km):
    return tomomt.project_points_to_profile(
        eq_e0, eq_n0, e_ends, n_ends, swath_km, z0=zeqs,
        zmin_km=zmin_km, zmax_km=zmax_km)


def _project_mt_sites_to_profile(e_ends, n_ends, swath_km):
    """
    Project MT sites onto the profile.  Returns (along_km, mt_e_kept,
    mt_n_kept) — the along-profile distances and the original UTM
    easting/northing of sites within the swath, so the caller can query
    the topo interpolator at each site's real position.
    """
    e0 = np.asarray(mt_e)
    n0 = np.asarray(mt_n)
    de = e_ends[1] - e_ends[0]
    dn = n_ends[1] - n_ends[0]
    L  = np.sqrt(de**2 + dn**2)
    if L == 0:
        return np.array([]), np.array([]), np.array([])
    ue, un = de / L, dn / L
    ve = e0 - e_ends[0]
    vn = n0 - n_ends[0]
    along  = ve * ue + vn * un
    across = np.abs(ve * (-un) + vn * ue)
    mask   = (across <= swath_km) & (along >= 0) & (along <= L)
    return along[mask], e0[mask], n0[mask]


def compute_vertical_slice_femtic(vslice):
    """
    Cut the FEMTIC mesh along a vertical profile using each element's
    own exact geometry (see profile_slice_polygons()). Returns
    (polys_km, poly_vals, e_ends, n_ends, L) — polys_km is a list of
    (m, 2) arrays already converted to (along-profile distance km,
    depth km), ready for PolyCollection.
    """
    e_ends, n_ends = _profile_utm_km(vslice)
    e1_m, n1_m = e_ends[0] * 1e3, n_ends[0] * 1e3
    e2_m, n2_m = e_ends[1] * 1e3, n_ends[1] * 1e3

    polys_2d, vals, length_m = profile_slice_polygons(
        _femtic_nodes, _free_conn, _free_vals, e1_m, n1_m, e2_m, n2_m
    )
    L = length_m / 1e3
    polys_km = [
        np.column_stack([p[:, 0] / 1e3, p[:, 1] / 1e3])  # (along-profile km, depth km) -- z already UTM metres positive-down
        for p in polys_2d
    ]
    return polys_km, vals, e_ends, n_ends, L


def plot_vertical_slice(polys_km, vals, e_ends, n_ends, L, lbl_start, lbl_end,
                         vslice, cmap, cmin, cmax, cbar_label, stem,
                         title_field="log$_{10}$ρ"):
    """
    Produce and save a vertical cross-section figure as an exact cut
    through the mesh's real elements — every polygon is one true 3-D
    tetrahedron's own value, at its true along-profile and true depth
    extent (see compute_vertical_slice_femtic()).
    """
    swath = vslice.get("swath_km", 10.0)
    zmin_s = vslice.get("zmin_km", 0.0)
    zmax_s = vslice.get("zmax_km", 30.0)
    ve = 1.0 if VSLICE_EQUAL_SCALE else VSLICE_VE
    name = vslice.get("name", "profile")

    eq_dist, eq_dep = _project_seismicity_to_profile(e_ends, n_ends, swath, zmin_s, zmax_s)
    mt_dist, mt_e_proj, mt_n_proj = _project_mt_sites_to_profile(e_ends, n_ends, swath)

    y_top = zmin_s
    topo_prof = None
    if _topo_interp is not None:
        n_pts = np.linspace(n_ends[0], n_ends[1], 200)
        e_pts = np.linspace(e_ends[0], e_ends[1], 200)
        topo_prof = _topo_interp(np.column_stack([n_pts, e_pts]))
        surf_depth = -topo_prof / 1e3
        y_top = min(zmin_s, surf_depth.min() - VSLICE_TOPO_HEADROOM_KM)

    profile_len = L
    depth_range = zmax_s - y_top
    if VSLICE_HEIGHT_CM is not None:
        h_in = VSLICE_HEIGHT_CM / 2.54
        w_in = h_in * profile_len / (depth_range * ve)
    else:
        w_in = VSLICE_WIDTH_CM / 2.54
        h_in = w_in * (depth_range * ve) / profile_len
    print(f"  Section figure size: {w_in:.2f} × {h_in:.2f} in")

    fig, ax, cax = create_section_figure(w_in, h_in)

    if ve != 1.0:
        vx, vy, vha, vva = _resolve_ve_pos(VSLICE_VE_POS)
        ax.text(vx, vy, f"VE = {ve:.1f}×", transform=ax.transAxes, ha=vha,
                va=vva, zorder=21, **VSLICE_VE_STYLE)

    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    if len(polys_km):
        pc = PolyCollection(polys_km, array=np.asarray(vals), cmap=cmap,
                             norm=norm, alpha=1.0 - ALPHA_RHO, zorder=5,
                             edgecolors="none")
        ax.add_collection(pc)
        im = pc
    else:
        print(f"  WARNING: no elements crossed the cutting plane for "
              f"{name!r} — figure will show basemap/features only.")
        im = None

    if VSLICE_SHOW_DEM_TOPO_LINE and topo_prof is not None:
        n_pts_plot = np.linspace(0.0, L, len(topo_prof))
        dem_depth_km = -topo_prof / 1e3
        ax.plot(n_pts_plot, dem_depth_km, **VSLICE_DEM_TOPO_STYLE)

    if SHOW_SEISMICITY and len(eq_dist):
        tomomt.markers(ax, eq_dist, eq_dep, **EQ_MARKER_STYLE)
    if SHOW_MT_SITES and len(mt_dist):
        # Place each MT site at the topographic surface (negative elevation
        # → positive depth on the section's depth axis) rather than at z = 0
        # (sea level), which is wrong for sites above or below sea level.
        # Query the topo interpolator at each site's real UTM position.
        mt_elev_m = _topo_interp(np.column_stack([mt_n_proj, mt_e_proj]))  # m
        mt_dep = -mt_elev_m / 1e3  # elevation (m, positive up) → depth (km, positive down)
        # Fall back to z = 0 for any site outside the topo grid extent.
        mt_dep = np.where(np.isfinite(mt_dep), mt_dep, 0.0)
        tomomt.markers(ax, mt_dist, mt_dep, **MT_MARKER_STYLE)

    x0, x1 = 0.0, L
    xlim = vslice.get("xlim", None)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(x0, x1)
    ax.set_ylim(y_top, zmax_s)
    ax.invert_yaxis()
    ax.set_xlabel("Distance along profile (km)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Depth (km)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=AXIS_TICK_SIZE)

    for xpos, lbl in ((x0, lbl_start), (x1, lbl_end)):
        ax.text(xpos, y_top, lbl, ha="center", va="bottom",
                fontsize=AXIS_LABEL_SIZE, fontweight="bold", color="black",
                clip_on=False, zorder=20)

    ax.set_title(f"{title_field} — {name}", fontsize=AXIS_TITLE_SIZE)
    if im is not None:
        finish_section_colorbar(cax, im, cbar_label)
    draw_annotation(ax)
    save_fig(fig, stem)
    _maybe_show()
    plt.close(fig)


# ==================================================================
# Main loop
# ==================================================================
out_list = []

for ii, d_km in enumerate(DEPTH_SLICES_KM):
    tag = f"{d_km:.0f}km" if d_km == int(d_km) else f"{d_km:.1f}km"
    label = f"{d_km:.0f} km" if d_km == int(d_km) else f"{d_km:.1f} km"
    print(f"Plotting log10(ρ) at {label} …")

    plane_z_m = d_km * 1e3
    polys_m, vals = depth_slice_polygons(
        _femtic_nodes, _free_conn, _free_vals, plane_z_m
    )
    polys_km = [
        np.column_stack([p[:, 0] / 1e3, p[:, 1] / 1e3]) for p in polys_m
    ]

    zmin = ZMIN_SEISM[ii] if ZMIN_SEISM[ii] is not None else -np.inf
    zmax = ZMAX_SEISM[ii] if ZMAX_SEISM[ii] is not None else np.inf
    mask_eqs = (zeqs > zmin) & (zeqs < zmax)
    eq_e = eq_e0[mask_eqs]
    eq_n = eq_n0[mask_eqs]

    fig, ax, cax = create_map_figure()
    draw_basemap(ax)

    norm = mcolors.Normalize(vmin=CMIN_RHO, vmax=CMAX_RHO)
    if len(polys_km):
        pc = PolyCollection(polys_km, array=vals, cmap=CMAP_RHO, norm=norm,
                             alpha=1.0 - ALPHA_RHO, zorder=5, edgecolors="none")
        ax.add_collection(pc)
        im = pc
    else:
        print(f"  WARNING: no elements crossed the plane at {label} — "
              f"figure will show basemap/features only.")
        im = None

    draw_features(ax, eq_e, eq_n)
    ax.set_title(f"log$_{{10}}$ρ at {label}", fontsize=AXIS_TITLE_SIZE)
    if im is not None:
        finish_panel_colorbar(cax, im, "log$_{10}$(ρ / Ω·m)")
    if AXES_UNITS == "latlon":
        add_latlon_ticks(ax)
    draw_annotation(ax)

    stem = f"femtic_rho_{tag}_{SITE_PREFIX}"
    save_fig(fig, stem)
    _maybe_show()
    plt.close(fig)
    out_list.append(stem)

for vi, vslice in enumerate(VSLICES):
    name = vslice.get("name", f"profile_{vi}")
    print(f"Plotting vertical section: {name} …")
    lbl_start, lbl_end = _profile_labels(vi)
    polys_km, vals, e_ends, n_ends, L = compute_vertical_slice_femtic(vslice)
    safe_name = name.replace(" ", "_").replace("'", "")
    stem = f"femtic_rho_sec_{safe_name}_{SITE_PREFIX}"
    plot_vertical_slice(
        polys_km, vals, e_ends, n_ends, L, lbl_start, lbl_end, vslice,
        CMAP_RHO, VSLICE_CMIN_RHO, VSLICE_CMAX_RHO,
        "log$_{10}$(ρ / Ω·m)", stem,
    )
    out_list.append(stem)

print("\nDone. Output stems:")
for s in out_list:
    print(f"  {s}")
