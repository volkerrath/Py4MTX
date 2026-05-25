#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_plot.py — Read and plot slice panels of a FEMTIC resistivity model.

Optionally reads one site position from observe.dat (given its site number)
and overplots it on every relevant panel.  The site's model-local coordinates
(km) are converted to UTM metres using a user-supplied UTM origin for the
mesh centre.

Slice positions can be given in three equivalent systems
---------------------------------------------------------
Every horizontal slice position key (``x0``, ``y0``, and the horizontal
components of ``point``) can optionally carry a coordinate-system tag by
replacing the plain scalar with a two-element tuple:

    (value, "crs")

where ``crs`` is one of:

    "model"   model-local metres, origin at mesh centre (default; a bare
              float is treated as "model" — fully backward-compatible)
    "utm"     UTM metres in the same zone as the mesh origin (zone number
              is auto-derived from UTM_ORIGIN_LAT / UTM_ORIGIN_LON)
    "latlon"  geographic decimal degrees (longitude for x0 / NS slices,
              latitude for y0 / EW slices)

The conversion chain is always:

    lat/lon ──► UTM(m) ──► model-local(m)
    UTM(m)            ──► model-local(m)
    model-local(m)    ──► (no-op)

Depth (z0) is always in model-local metres; no geographic conversion applies.

Examples (inside PLOT_SLICES)
------------------------------
    # Plain float — model-local metres (unchanged from previous version):
    dict(kind="map",  z0=5000.0)
    dict(kind="ns",   x0=0.0)

    # UTM easting for the NS curtain:
    dict(kind="ns",   x0=(229047.0, "utm"))

    # Geographic longitude for the NS curtain:
    dict(kind="ns",   x0=(-71.537, "latlon"))

    # Geographic latitude for the EW curtain:
    dict(kind="ew",   y0=(-16.409, "latlon"))

    # Arbitrary plane through a geographic point:
    dict(kind="plane",
         point=([−71.5, −16.4, 5000.0], "latlon"),
         strike=45., dip=70.)

UTM zone derivation
--------------------
    Zone number is computed from ``UTM_ORIGIN_LON`` (standard 6° bands,
    ignoring Norway / Svalbard exceptions).  Override with
    ``UTM_ZONE_OVERRIDE`` (positive integer) when needed.

Display coordinate system
--------------------------
    DISPLAY_COORDS = "model"  — axis ticks in model-local metres (default)
    DISPLAY_COORDS = "utm"    — axis ticks in absolute UTM metres

Provenance
----------
    2026-05-06  vrath / Claude Sonnet 4.6   Created, modelled on
                femtic_mod_edit.py plotting section.
    2026-05-06  vrath / Claude Sonnet 4.6   Added lat/lon and UTM slice-
                position input; pure-Python UTM forward projection;
                auto-derived UTM zone from mesh origin coordinates.
    2026-05-06  vrath / Claude Sonnet 4.6   Added estimate_utm_origin:
                least-squares mesh-centre estimation from N calibration
                sites with known model-local and geographic coordinates.
    2026-05-13  vrath / Claude Sonnet 4.6   Harmonised plotting config block
                with femtic_mod_edit.py: unified variable names, comments,
                and section header.
    2026-05-13  vrath / Claude Sonnet 4.6   Added 3-D PyVista plot step (5):
                PLOT3D config block with axis-aligned x/y/z slices, oblique
                planes, and iso-surfaces via fviz.plot_model_3d.  Output:
                interactive HTML or static screenshot.

    2026-05-16  vrath / Claude Sonnet 4.6   Added borehole resistivity log
                step (7): _point_in_tet (barycentric), extract_borehole_log
                (bbox pre-filter + exact test), plot_borehole_logs; BOREHOLE_*
                config block; CRS tagging (model/utm/latlon) on x/y positions.
    2026-05-23  vrath / Claude Sonnet 4.6   Moved pure geographic helpers
                (_latlon_to_utm, _utm_to_latlon, _utm_zone_from_origin) to
                util.py (utm_zone_from_latlon, latlon_to_utm_zn,
                utm_to_latlon_zn).  Moved model-local helpers (_utm_to_model,
                _latlon_to_model, _parse_pos, _resolve_x0/y0/point,
                resolve_slices) to femtic.py (utm_to_model, latlon_to_model,
                parse_pos_crs, resolve_pos_x/y/point, resolve_slice_positions).
                Script-level functions are now thin wrappers that supply the
                module globals.  Imported femtic as fem.
                SITE_NUMBER now accepts a list; plot_model_slices takes
                site_xys [(sn, x_m, y_m), …] and loops over all sites.
    2026-05-23  vrath / Claude Sonnet 4.6   SITE_DAT now uses the
                mt_make_sitelist.py CSV format (name, lat, lon, elev,
                sitenum, easting, northing); replaces SITELIST_FILE.
                read_site_dat() rewritten accordingly; read_sitelist()
                removed.  estimate_utm_origin kwarg renamed site_dat;
                bounding-box origin method wired to read_site_dat().
    2026-05-23  vrath / Claude Sonnet 4.6   PLOT_EQUAL_ASPECT config flag;
                equal_aspect kwarg in plot_model_slices; set_aspect("equal",
                adjustable="box") on map/ns/ew panels when DISPLAY_COORDS is
                model or utm; figsize auto-computed from xlim/ylim/zlim ratios.
    2026-05-24  vrath / Claude Sonnet 4.6   Removed ENS_* config block and
                step (6) ensemble plot; moved to snippets.py.  Step (7)
                borehole log renumbered to step (6).
    2026-05-24  vrath / Claude Sonnet 4.6   Moved read_site_position(),
                read_site_dat(), estimate_utm_origin(), _point_in_tet(),
                extract_borehole_log() to femtic.py; script calls fem.*
                directly.  import data_proc as dp removed.
    2026-05-25  vrath / Claude Sonnet 4.6   Added PLOT_SITES master switch.
                Site markers (inverted triangle, depth=0) now on all panel
                kinds incl. plane (along-strike projection via u-axis dot
                product).  Legend guard extended to "plane".
    2026-05-25  vrath / Claude Sonnet 4.6   Replaced ESTIMATE_ORIGIN /
                CALIBRATION_SITES / UPDATE_CONFIG with ORIGIN_METHOD
                (None | "box" | "average").  Origin estimated from SITE_DAT
                UTM coords only; observe.dat fallback is model-local only.
    2026-05-25  vrath / Claude Sonnet 4.6   UTM display mode now in km
                (_display_scale 1e-3).  Curtain/plane site markers at true
                mesh surface (z_surf = nodes[:,2].min()) not depth=0.
                PROJECTION_DIST filters sites per curtain/plane panel.
                DISPLAY_COORDS utm/latlon suppressed when sites from
                observe.dat (obs_coords_only flag).
    2026-05-25  vrath / Claude Sonnet 4.6   DEPTH_KM flag for km depth axis
                on curtain/plane panels.  PLOT_PANEL_HEIGHT, PLOT_PANEL_WIDTH,
                PLOT_FIGSIZE config vars; passed to plot_model_slices as
                panel_height, panel_width, figsize.  Aspect-ratio auto-width
                accounts for depth_km and horizontal sc scales.
    2026-05-25  vrath / Claude Sonnet 4.6   HORIZ_KM flag: km on horizontal
                axes in "model" mode.  PLOT_NROWS / PLOT_NCOLS grid layout;
                subplots flattened to 1-D, surplus cells hidden.  _do_equal
                checks horiz/vert km consistency.
    2026-05-25  vrath / Claude Sonnet 4.6   Sites removed from curtain/plane
                panels (map panels only).  PLOT_PANEL_HEIGHT/WIDTH/FIGSIZE now
                in cm (converted to inches at call site).  Cmap deprecation
                fixed: matplotlib.colormaps[cmap] replaces get_cmap().
    2026-05-25  vrath / Claude Sonnet 4.6   PLOT_SITES replaced by
                PLOT_SITES_MAPS / PLOT_SITES_SLICES for independent control.

@author: vrath
"""

import os
import sys
from pathlib import Path
import math
import inspect
import numpy as np

# ---------------------------------------------------------------------------
# Py4MTX-specific settings and imports
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from version import versionstrg
import util as utl
import femtic as fem

try:
    import femtic_viz as fviz
except ImportError:
    fviz = None

try:
    from pyproj import Transformer as _Transformer
    _HAVE_PYPROJ = True
except ImportError:
    _HAVE_PYPROJ = False

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# Configuration
# ===========================================================================

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK_DIR = r"/home/vrath/Py4MTX/work/rto/ubinas_data/"

#: Resistivity block to display (any iteration).
MODEL_FILE = WORK_DIR + "resistivity_block_iter17.dat"

#: Mesh file — always required for plotting.
MESH_FILE = WORK_DIR + "mesh.dat"

#: observe.dat — used by ESTIMATE_ORIGIN to look up model-local site positions.
#: Also used as fallback site-overlay source when SITE_DAT is None and
#: SITE_NUMBER is not None.
OBSERVE_FILE = WORK_DIR + "observe.dat"

#: Site list produced by mt_make_sitelist.py (WHAT_FOR="femtic").
#: Format (comma-separated, no header):
#:   name, lat, lon, elev, sitenum, easting, northing
#: Easting/northing are UTM metres; model-local x/y is derived via
#: fem.utm_to_model using the mesh-centre origin.
#: When ESTIMATE_ORIGIN is True and CALIBRATION_SITES is empty, the
#: bounding-box centre of all sites is used to estimate the mesh origin.
#: Set to None to fall back to the observe.dat / SITE_NUMBER path.
SITE_DAT = WORK_DIR + "site.dat"   # set to None to disable

# ---------------------------------------------------------------------------
# Ocean / air handling (must match the inversion setup)
# ---------------------------------------------------------------------------
#: None → auto-infer from region 1 heuristic (ρ ≤ 1 Ω·m AND flag==1).
#: True / False → force ocean-present / ocean-absent.
OCEAN = None

AIR_RHO = 1.0e9   # Ω·m  (region 0)
OCEAN_RHO = 0.25    # Ω·m  (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# Geographic / UTM origin of the mesh centre
# ---------------------------------------------------------------------------
#: Geographic coordinates (WGS-84) of the FEMTIC mesh origin.
#: Used to derive the UTM zone number and to convert lat/lon slice positions.
UTM_ORIGIN_LAT = -16.409   # decimal degrees, positive = North
UTM_ORIGIN_LON = -71.537   # decimal degrees, positive = East

#: UTM coordinates of the mesh origin in metres (same zone as above).
#: Used for model-local ↔ UTM conversions and for display tick offsets.
UTM_ORIGIN_E = 229047.0   # easting  [m]
UTM_ORIGIN_N = 8184127.0  # northing [m]

#: Override the auto-derived UTM zone number.  None = auto from origin lat/lon.
#: Example: UTM_ZONE_OVERRIDE = 19  →  force zone 19 (ignoring special zones).
UTM_ZONE_OVERRIDE = None

# ---------------------------------------------------------------------------
# Display coordinate system
# ---------------------------------------------------------------------------
#: "model"  — axis ticks in model-local metres (origin = 0, default)
#: "utm"    — axis ticks in absolute UTM metres
#: "latlon" — axis ticks in decimal degrees (lon for easting, lat for northing)
DISPLAY_COORDS = "utm"

# ---------------------------------------------------------------------------
# Site overlay
# ---------------------------------------------------------------------------
#: Primary source: site names to overlay from SITE_DAT.
#: May be a single string or a list of strings matching the "name" column.
#: Set to None to overlay all sites in the file.
SITE_NAMES = None   # e.g. ["MT01", "MT05", "MT12"]  or None = all sites

#: Fallback source (used when SITE_DAT is None): site number(s) to
#: extract from observe.dat (integer, 1-based).
#: May be a single int or a list of ints.  Set to None to skip site overlay.
SITE_NUMBER = [5, 6, 7]

#: Show site markers on horizontal map panels.
PLOT_SITES_MAPS = True
#: Show site markers on vertical curtain (ns/ew) and plane panels.
PLOT_SITES_SLICES = True

#: Maximum distance [m] from a vertical slice plane for a site to be plotted
#: on that panel.  Sites further than this are omitted.
#: For NS panels: distance in x (easting); for EW panels: distance in y (northing).
#: For plane panels: perpendicular distance from the slice plane.
#: None = plot all sites on every panel regardless of distance.
PROJECTION_DIST = 3000.  # e.g. 5000.0  (5 km)

#: Marker style for map panels (inverted triangle = MT convention).
SITE_MARKER = dict(marker="v", color="black", ms=4, zorder=10,
                   label=None)   # label filled in automatically

#: Marker style for curtain and plane panels (centered symbol at site elevation).
SITE_MARKER_SLICES = dict(marker="o", color="black", ms=4, zorder=10,
                           label=None)

# ---------------------------------------------------------------------------
# Additional map markers
# ---------------------------------------------------------------------------
#: Arbitrary point markers overlaid on map panels only.
#: Each entry is a dict with:
#:   "latlon"  : [lat_deg, lon_deg]  — position in geographic coordinates
#:   "marker"  : Matplotlib marker string  (e.g. "P", "+", "x", "*", "^")
#:   "color"   : colour string
#:   "ms"      : marker size in points
#:   "name"    : label string (shown in legend); None = no legend entry
#: Any additional Matplotlib plot kwargs (mew, mfc, zorder, …) are accepted.
MAP_MARKERS = [
    dict(latlon=[-16.34861, -70.90222], marker="*", color="red", ms=10,
         name="Ubinas summit"),
]

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Plotting — requires femtic_viz and Matplotlib
# ---------------------------------------------------------------------------
#: Output file path — None → interactive show().
PLOT_FILE = WORK_DIR + "resistivity_block_iter17.pdf"

#: Figure DPI for saved file.
PLOT_DPI = 600

#: Matplotlib colormap name.
PLOT_CMAP = "turbo_r"

#: Colour limits [log10(ρ_min), log10(ρ_max)] — None = auto.
PLOT_CLIM = [0.0, 4.0]      # log10(Ω·m)

#: Flat colour for ocean / lake cells.  None → use colormap.
PLOT_OCEAN_COLOR = "lightgrey"

#: Axes facecolor for air / background.  None = figure default.
PLOT_AIR_BGCOLOR = None

#: Slice specification — a list of dicts, one per panel (left to right).
#:
#: Slices use exact tetrahedron-plane intersection — no selection slab,
#: no dw, no dz.  The plane is infinitely thin; every tetrahedron that
#: straddles it contributes an exact triangle or quadrilateral polygon.
#:
#: Each dict must contain:
#:   kind   : "map"   — horizontal slice at z = z0
#:            "ns"    — N-S vertical section at x = x0   (y vs depth)
#:            "ew"    — E-W vertical section at y = y0   (x vs depth)
#:            "plane" — arbitrary plane by strike / dip / point
#:   z0     : (map   only)  depth in metres
#:   x0     : (ns    only)  easting — plain float = model-local metres;
#:            or (value, "utm") / (value, "latlon") for CRS tagging
#:   y0     : (ew    only)  northing — plain float = model-local metres;
#:            or (value, "utm") / (value, "latlon") for CRS tagging
#:   point  : (plane only)  [x, y, z] any point on the plane (metres)
#:            or ([lon, lat, z], "latlon") / ([E, N, z], "utm")
#:   strike : (plane only)  clockwise from North, degrees (0=N, 90=E)
#:   dip    : (plane only)  downward inclination from horizontal, degrees
#:   xlim   : [xmin, xmax] — easting or along-strike axis limit
#:   ylim   : [ymin, ymax] — northing or down-dip axis limit
#:   zlim   : [zmin, zmax] — depth axis limit (ns/ew panels)
#:   title  : optional string override
#:
#: Per-panel xlim/ylim/zlim override the global PLOT_XLIM/PLOT_YLIM/PLOT_ZLIM.
#: Ubinas summit: 16.34861° S, 70.90222° W (16°20′55″S 70°54′08″W), elevation 5672 m.

PLOT_SLICES = [
    # Plain float — model-local metres (backward-compatible):
    dict(kind="map", z0=-4000.0),
    dict(kind="map", z0=15000.0),
    # UTM easting for the NS curtain:
    dict(kind="ns",   x0=(-70.85576, "latlon")),
    # dict(kind="ns", x0=(-70.90222, "latlon")),
    # Geographic latitude for the EW curtain:
    # dict(kind="ew", y0=(-16.34861, "latlon")),    
    dict(kind="ew", y0=(-16.39606, "latlon")),
]


#: Global axis limits in model-local metres — used for panels that do not
#: specify their own.  None → auto (inferred from data extent).
PLOT_XLIM = [-20000., 20000.]   # [xmin, xmax] metres — easting
PLOT_YLIM = [-20000., 20000.]   # [ymin, ymax] metres — northing
PLOT_ZLIM = [-6000., 15000.]  # [zmin, zmax] metres — depth (z positive-down)

#: Equal aspect ratio for map and curtain panels.
#: True  → ax.set_aspect("equal") on map (x/y), ns (y/z), and ew (x/z) panels
#:         so that 1 m horizontal = 1 m vertical on screen.  Applies only when
#:         DISPLAY_COORDS is "model" or "utm" (both axes in metres).  Ignored
#:         for plane slices and when DISPLAY_COORDS is "latlon".
#: False → Matplotlib default (axes fill the available space).
PLOT_EQUAL_ASPECT = True

#: Display depth axis in km instead of metres on curtain and plane panels.
#: Does not affect the horizontal axes (use HORIZ_KM for those).
DEPTH_KM = True

#: Display horizontal axes in km instead of metres when DISPLAY_COORDS is
#: "model".  Has no effect for "utm" (already in km) or "latlon" (degrees).
#: When both HORIZ_KM and DEPTH_KM are True, equal aspect is restored in
#: "model" mode because both axes carry the same km scale.
HORIZ_KM = True

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
#: Number of subplot rows and columns.  None → auto:
#:   PLOT_NROWS = None  →  1 row (all panels in a single row)
#:   PLOT_NCOLS = None  →  len(PLOT_SLICES) columns
#: Set both explicitly for a grid, e.g. PLOT_NROWS=2, PLOT_NCOLS=4 for 8 maps.
#: Total cells must be >= len(PLOT_SLICES); surplus cells are hidden.
PLOT_NROWS = 2   # e.g. 2
PLOT_NCOLS = 2  # e.g. 4

#: Panel height in cm (height of one row of panels).
PLOT_PANEL_HEIGHT = 8.0   # cm

#: Fixed panel width in cm.  None → auto-computed from aspect ratio when
#: PLOT_EQUAL_ASPECT = True, or equal to PLOT_PANEL_HEIGHT otherwise.
PLOT_PANEL_WIDTH = None   # e.g. 10.0 to force a fixed width per panel

#: Full figure size [width, height] in cm.  Overrides all auto-computed
#: sizing when set.  None → computed from panel sizes and grid shape.
PLOT_FIGSIZE = None   # e.g. [40., 25.]

# ---------------------------------------------------------------------------
# 3-D plotting — requires PyVista  (conda install -c conda-forge pyvista)
# ---------------------------------------------------------------------------
#: Set True to produce a 3-D PyVista scene after the 2-D slice figure.
PLOT3D = False

#: Output file for the 3-D scene.
#:   .html  → interactive WebGL in a browser (recommended).
#:   .png / .jpg → static screenshot.
#:   None   → open an interactive PyVista window (requires a display).
PLOT3D_FILE = WORK_DIR + "resistivity_block_iter0_3d.html"

#: Scalar field to display.  "log10_resistivity" or "resistivity".
PLOT3D_SCALAR = "log10_resistivity"

#: Colour limits [vmin, vmax] for the scalar.  None → PyVista auto.
PLOT3D_CLIM = [0.0, 4.0]       # log10(Ω·m)

#: Matplotlib / PyVista colormap for slices.
PLOT3D_CMAP = "turbo_r"

#: Axis-aligned slice positions in model-local metres (z positive-down).
#: Each list entry places one cutting plane perpendicular to that axis.
#: Empty list or None → no slices along that axis.
PLOT3D_SLICE_X = [0.0]                    # YZ planes — N-S sections
PLOT3D_SLICE_Y = [0.0]                    # XZ planes — E-W sections
PLOT3D_SLICE_Z = [5000.0, 15000.0]        # XY planes — horizontal maps

#: Arbitrary oblique plane slices.  Each entry is a dict with:
#:   "origin" : [x, y, z]  — any point on the plane (model-local m).
#:   "normal" : [nx, ny, nz] — plane normal vector (need not be unit).
#: Empty list or None → no oblique slices.
PLOT3D_SLICE_PLANES = [
    # dict(origin=[0., 0., 8000.], normal=[1., 1., 0.]),   # NE-trending vertical
]

#: Iso-surface levels in the same units as PLOT3D_SCALAR.
#: For log10_resistivity: 1.0 = 10 Ω·m, 2.0 = 100 Ω·m, 3.0 = 1000 Ω·m.
#: Empty list or None → no iso-surfaces.
PLOT3D_ISOVALUES = [1.0, 2.0, 3.0]

#: Opacity for iso-surfaces (0 = transparent, 1 = solid).
PLOT3D_ISO_OPACITY = 0.35

#: Window size in pixels [width, height] — used for screenshot modes.
PLOT3D_WINDOW_SIZE = [1600, 900]

# ---------------------------------------------------------------------------
# Borehole resistivity logs  (optional)
# ---------------------------------------------------------------------------
#: Set True to produce a 1-D log₁₀(ρ) vs depth figure from point-in-element
#: sampling along vertical boreholes.
PLOT_BOREHOLE = False

#: Output file for the borehole figure.
#:   None → interactive show().
BOREHOLE_FILE = WORK_DIR + "resistivity_block_iter0_boreholes.pdf"

#: List of borehole specifications.  Each entry is a dict with:
#:   "name"   : str   — label shown in the legend / panel title
#:   "x"      : float — model-local easting  [m]  (or use (value, "utm") /
#:              (value, "latlon") tuples — same CRS tagging as PLOT_SLICES)
#:   "y"      : float — model-local northing [m]  (same CRS tagging applies)
#:   "z_top"  : float — start depth [m, FEMTIC z-down convention; positive-down]
#:              0 = surface; negative = above datum (e.g. elevated topography)
#:   "z_bot"  : float — end depth [m, positive-down], e.g. 20000.0 for 20 km
#:   "dz"     : float — sampling interval [m]; e.g. 100.0 for 100 m steps
#:
#: Example:
BOREHOLE_SITES = [
    # dict(name="BH-01", x=0.0,    y=0.0,    z_top=0.0, z_bot=20000., dz=200.),
    # dict(name="BH-02", x=(229100., "utm"), y=(8184000., "utm"),
    #      z_top=0.0, z_bot=15000., dz=100.),
]

#: Matplotlib line / marker style for the borehole traces.
BOREHOLE_STYLE = dict(lw=1.2, marker="none")

#: x-axis limits for the borehole log panels [log10 min, log10 max].
#:   None → auto.
BOREHOLE_XLIM = [0.0, 4.0]   # log10(Ω·m)

#: Whether to draw all boreholes in a single shared axes (True) or
#: one panel per borehole (False).
BOREHOLE_SHARED = True

# ---------------------------------------------------------------------------
# Mesh-centre estimation from site.dat UTM coordinates  (optional)
# ---------------------------------------------------------------------------
#: Method used to estimate UTM_ORIGIN_E / UTM_ORIGIN_N from SITE_DAT:
#:
#:   None      — use the hard-coded UTM_ORIGIN_E / UTM_ORIGIN_N above (default)
#:   "box"     — midpoint of the UTM bounding box of all sites in SITE_DAT
#:               (femticPY-compatible)
#:   "average" — arithmetic mean of all site UTM coordinates in SITE_DAT
#:
#: Requires SITE_DAT to be set and readable.  The result overwrites
#: UTM_ORIGIN_E / UTM_ORIGIN_N (and UTM_ORIGIN_LAT / UTM_ORIGIN_LON) for
#: this run; the printed values can be copied back into the config above.
#: Falls back to hard-coded values if SITE_DAT is None or unreadable.
ORIGIN_METHOD =  "box"   # None | "box" | "average"


# ===========================================================================
# Coordinate conversion helpers
# ===========================================================================
# Pure geographic conversions (latlon ↔ UTM, zone derivation) live in
# util.py.  Model-local conversions (utm/latlon → model-local metres,
# CRS-tagged position resolution, slice resolution) live in femtic.py.
# The script-level wrappers below bind the module globals so callers
# throughout this script need not repeat the origin/zone arguments.
# ===========================================================================

def _utm_zone_from_origin() -> tuple[int, bool]:
    """Derive UTM zone and hemisphere from module-level mesh-origin lat/lon."""
    return utl.utm_zone_from_latlon(UTM_ORIGIN_LAT, UTM_ORIGIN_LON,
                                    override=UTM_ZONE_OVERRIDE)


def _latlon_to_utm(lat_deg: float, lon_deg: float,
                   zone: int, northern: bool) -> tuple[float, float]:
    """WGS-84 → UTM [m].  Delegates to util.latlon_to_utm_zn."""
    return utl.latlon_to_utm_zn(lat_deg, lon_deg, zone, northern)


def _utm_to_model(E_m: float, N_m: float) -> tuple[float, float]:
    """UTM [m] → model-local [m] using module-level mesh-centre origin."""
    return fem.utm_to_model(E_m, N_m, UTM_ORIGIN_E, UTM_ORIGIN_N)


def _latlon_to_model(lat_deg: float, lon_deg: float,
                     zone: int, northern: bool) -> tuple[float, float]:
    """Geographic [°] → model-local [m] using module-level origin."""
    return fem.latlon_to_model(lat_deg, lon_deg, zone, northern,
                               UTM_ORIGIN_E, UTM_ORIGIN_N)


def _parse_pos(raw) -> tuple:
    """Parse ``(value, crs)`` position spec.  Delegates to fem.parse_pos_crs."""
    return fem.parse_pos_crs(raw)


def _resolve_x0(raw, zone: int, northern: bool) -> float:
    """Resolve x0 to model-local metres.  Delegates to fem.resolve_pos_x."""
    return fem.resolve_pos_x(raw, zone, northern,
                             UTM_ORIGIN_E, UTM_ORIGIN_N,
                             UTM_ORIGIN_LAT, UTM_ORIGIN_LON)


def _resolve_y0(raw, zone: int, northern: bool) -> float:
    """Resolve y0 to model-local metres.  Delegates to fem.resolve_pos_y."""
    return fem.resolve_pos_y(raw, zone, northern,
                             UTM_ORIGIN_E, UTM_ORIGIN_N,
                             UTM_ORIGIN_LAT, UTM_ORIGIN_LON)


def _resolve_point(raw, zone: int, northern: bool) -> list[float]:
    """Resolve plane point to model-local metres.  Delegates to fem.resolve_pos_point."""
    return fem.resolve_pos_point(raw, zone, northern, UTM_ORIGIN_E, UTM_ORIGIN_N)


def resolve_slices(slices: list, zone: int, northern: bool) -> list:
    """Resolve all CRS-tagged positions in *slices* to model-local metres.

    Thin wrapper around ``fem.resolve_slice_positions`` that supplies the
    module-level mesh-origin globals automatically.
    """
    return fem.resolve_slice_positions(
        slices, zone, northern,
        UTM_ORIGIN_E, UTM_ORIGIN_N,
        UTM_ORIGIN_LAT, UTM_ORIGIN_LON,
        verbose=OUT,
    )


# ===========================================================================
# Helper: display-coordinate offset and axis label suffix
# ===========================================================================

def _display_offset() -> tuple[float, float]:
    """Return (dE, dN) to add to model-local metres before applying scale.

    For 'latlon' the polygons are still drawn in UTM metres (offset by the
    UTM origin) and tick labels are reformatted by ``_display_formatters``.
    """
    if DISPLAY_COORDS in ("utm", "latlon"):
        return UTM_ORIGIN_E, UTM_ORIGIN_N
    return 0.0, 0.0


def _display_scale() -> float:
    """Return multiplicative scale applied after offset.

    'utm'  → 1e-3  (display in km)
    others → 1.0   (metres or degrees via formatter)
    """
    return 1e-3 if DISPLAY_COORDS == "utm" else 1.0


def _display_suffix() -> str:
    """Return axis label suffix reflecting the display coordinate system."""
    if DISPLAY_COORDS == "utm":
        return " [UTM km]"
    if DISPLAY_COORDS == "latlon":
        return " [°]"
    return " [m]"


def _utm_to_latlon(E_m: float, N_m: float,
                   zone: int, northern: bool) -> tuple[float, float]:
    """UTM [m] → (lat, lon) [°].  Delegates to util.utm_to_latlon_zn."""
    return utl.utm_to_latlon_zn(E_m, N_m, zone, northern)


def _display_formatters(zone: int, northern: bool):
    """Return (x_formatter, y_formatter) for the chosen DISPLAY_COORDS.

    For 'latlon' returns FuncFormatter objects that convert UTM tick values
    (in metres, already offset by UTM_ORIGIN) to lon / lat degree strings.
    For other modes returns (None, None) — default Matplotlib formatting.
    """
    if DISPLAY_COORDS != "latlon":
        return None, None

    import matplotlib.ticker as mticker

    def _lon_fmt(val, _pos):
        _, lon = _utm_to_latlon(val, UTM_ORIGIN_N, zone, northern)
        return f"{lon:.3f}"

    def _lat_fmt(val, _pos):
        lat, _ = _utm_to_latlon(UTM_ORIGIN_E, val, zone, northern)
        return f"{lat:.3f}"

    return (mticker.FuncFormatter(_lon_fmt),
            mticker.FuncFormatter(_lat_fmt))


# ===========================================================================
# Plotting
# ===========================================================================

def plot_model_slices(
    model_file: str,
    mesh_file: str,
    slices: list,
    *,
    cmap: str = "turbo_r",
    clim=None,
    xlim=None,
    ylim=None,
    zlim=None,
    ocean_color="lightgrey",
    ocean_value: float = OCEAN_RHO,
    air_bgcolor=None,
    site_xys: list | None = None,
    obs_coords_only: bool = False,
    projection_dist: float | None = None,
    sites_in_maps: bool = True,
    sites_in_slices: bool = False,
    map_markers: list | None = None,
    plot_file=None,
    dpi: int = 200,
    equal_aspect: bool = True,
    depth_km: bool = False,
    horiz_km: bool = False,
    nrows: int | None = None,
    ncols: int | None = None,
    panel_height: float = 5.0,
    panel_width: float | None = None,
    figsize: list | tuple | None = None,
    out: bool = True,
):
    """Produce a multi-panel figure of axis-parallel model slices.

    All position values in *slices* must already be in model-local metres
    (pre-process with ``resolve_slices`` before calling).

    Parameters
    ----------
    model_file  : resistivity block file
    mesh_file   : mesh.dat used during inversion
    slices      : list of slice-spec dicts with model-local positions
    cmap        : Matplotlib colormap name
    clim        : [log10_min, log10_max]; None = auto
    xlim, ylim, zlim : global axis limits (model-local m); per-panel override
    ocean_color : flat colour for ocean polygons; None → colormap
    ocean_value : Ω·m sentinel for ocean (must match OCEAN_RHO)
    air_bgcolor : axes facecolor for air / background; None = figure default
    site_xys    : list of (label, x_m, y_m, elev_m) tuples in model-local
                  metres; elev_m is the site elevation above datum [m].
                  None or empty list → no markers.
    obs_coords_only : True when site_xys came from observe.dat (model-local
                  only); suppresses DISPLAY_COORDS "utm"/"latlon" display.
    projection_dist : maximum distance [m] from a curtain/plane for a site to
                  appear on that panel.  None = all sites on all panels.
    sites_in_maps   : plot site markers on horizontal map panels
    sites_in_slices : plot site markers on curtain (ns/ew) and plane panels
                  at the mesh surface elevation
    map_markers : list of dicts (latlon, marker, color, ms, name, …);
                  plotted on map panels only; lat/lon converted to
                  model-local at render time
    plot_file   : save path; None = interactive show()
    dpi         : saved-figure DPI
    equal_aspect : if True, call ``ax.set_aspect("equal", adjustable="box")``
                  on map/ns/ew panels.  Requires both axes to carry the same
                  physical scale (both metres, or both km).
    depth_km    : show depth axis in km on curtain and plane panels
    horiz_km    : show horizontal (easting/northing) axes in km when
                  DISPLAY_COORDS is "model"; no effect for "utm" (already km)
                  or "latlon" (degrees)
    nrows, ncols : subplot grid shape; None = 1 × n_panels.  Surplus cells
                  are hidden.  Total cells must be >= len(slices).
    panel_height : row height in inches (pass PLOT_PANEL_HEIGHT / 2.54 from cm)
    panel_width  : fixed column width in inches; None = auto from aspect ratio
                  when equal_aspect is True, else equal to panel_height
    figsize     : explicit [width, height] in inches; overrides all auto sizing
    out         : verbose progress
    """
    if fviz is None:
        print("  plot_model_slices: femtic_viz not available — skipping.")
        return

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.collections import PolyCollection
    except ImportError:
        print("  plot_model_slices: Matplotlib not available — skipping.")
        return

    # ── internal geometry helpers ────────────────────────────────────────────

    def _axis_slice_params(axis: int, val: float):
        """Return (normal, point, u_ax, v_ax, invert_v) for an axis-aligned cut."""
        normals = [np.array([1., 0., 0.]), np.array(
            [0., 1., 0.]), np.array([0., 0., 1.])]
        inv = [False, False, False]
        pt = np.zeros(3)
        pt[axis] = val
        n = normals[axis]
        ref = np.array([0., 0., 1.]) if axis != 2 else np.array([1., 0., 0.])
        u = np.cross(n, ref)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        v /= np.linalg.norm(v)
        return n, pt, u, v, inv[axis]

    def _tet_plane_intersection(verts, normal, d):
        dots = verts @ normal - d
        pos = dots >= 0
        if pos.all() or (~pos).all():
            return []
        pts = []
        for i in range(4):
            for j in range(i + 1, 4):
                if pos[i] != pos[j]:
                    t = dots[i] / (dots[i] - dots[j])
                    pts.append(verts[i] + t * (verts[j] - verts[i]))
        c = np.mean(pts, axis=0)
        u2d = np.cross(normal,
                       np.array([0., 0., 1.]) if abs(normal[2]) < 0.9
                       else np.array([1., 0., 0.]))
        if np.linalg.norm(u2d) < 1e-12:
            return pts
        u2d /= np.linalg.norm(u2d)
        v2d = np.cross(normal, u2d)
        angles = [np.arctan2((p - c) @ v2d, (p - c) @ u2d) for p in pts]
        return [pts[k] for k in np.argsort(angles)]

    def _slice_geometry(nodes, conn, rho_arr, normal, point, u_ax, v_ax):
        d = float(normal @ point)
        verts_all = nodes[conn]
        polys, vals = [], []
        for k, verts in enumerate(verts_all):
            pts3d = _tet_plane_intersection(verts, normal, d)
            if not pts3d:
                continue
            polys.append([(float(p @ u_ax), float(p @ v_ax)) for p in pts3d])
            with np.errstate(divide="ignore", invalid="ignore"):
                vals.append(math.log10(
                    rho_arr[k]) if rho_arr[k] > 0 else float("nan"))
        return polys, np.asarray(vals, dtype=float)

    def _plot_slice_panel(ax, polys, vals, *, cmap_obj, norm,
                          ocean_color, ocean_value, invert_v):
        if not polys:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            ov_log = math.log10(
                ocean_value) if ocean_value > 0 else float("nan")
        is_ocean = np.isclose(vals, ov_log, atol=0.05)
        is_air = ~np.isfinite(vals)
        is_data = ~is_ocean & ~is_air
        mappable = None
        if is_data.any():
            pc = PolyCollection(
                [polys[i] for i in np.where(is_data)[0]],
                array=vals[is_data], cmap=cmap_obj, norm=norm,
                linewidths=0, zorder=2, rasterized=True)
            ax.add_collection(pc)
            mappable = pc
        if is_ocean.any() and ocean_color is not None:
            oc = PolyCollection(
                [polys[i] for i in np.where(is_ocean)[0]],
                facecolor=ocean_color, linewidths=0, zorder=3, rasterized=True)
            ax.add_collection(oc)
        ax.autoscale_view()
        if invert_v:
            ax.invert_yaxis()
        return mappable

    def _strike_dip_to_normal(strike_deg, dip_deg):
        s, d = math.radians(strike_deg), math.radians(dip_deg)
        return np.array([-math.sin(d) * math.sin(s),
                         math.sin(d) * math.cos(s),
                         -math.cos(d)])

    def _plane_basis(normal):
        ref = np.array([0., 1., 0.]) if abs(
            normal[1]) < 0.9 else np.array([1., 0., 0.])
        u = np.cross(normal, ref)
        u /= np.linalg.norm(u)
        v = np.cross(u, normal)
        v /= np.linalg.norm(v)
        return u, v

    # ── display offset / scale ───────────────────────────────────────────────
    # When sites come from observe.dat (model-local only), UTM and lat/lon
    # display modes cannot be applied to site markers — fall back to "model".
    _disp = "model" if (obs_coords_only and DISPLAY_COORDS in ("utm", "latlon")) \
            else DISPLAY_COORDS
    if obs_coords_only and DISPLAY_COORDS in ("utm", "latlon") and out:
        print(f"  Note: site positions from observe.dat; "
              f"DISPLAY_COORDS={DISPLAY_COORDS!r} ignored for site markers "
              f"(model-local only).")
    dE = (UTM_ORIGIN_E if _disp in ("utm", "latlon") else 0.0)
    dN = (UTM_ORIGIN_N if _disp in ("utm", "latlon") else 0.0)
    # Horizontal scale factor: utm always km; model→km when HORIZ_KM; else 1.
    if _disp == "utm":
        sc = 1e-3
    elif _disp == "model" and horiz_km:
        sc = 1e-3
    else:
        sc = 1.0
    # Axis label suffix for horizontal axes.
    if _disp == "utm":
        sfx = " [UTM km]"
    elif _disp == "model" and horiz_km:
        sfx = " [km]"
    elif _disp == "latlon":
        sfx = " [°]"
    else:
        sfx = " [m]"
    _fmt_x, _fmt_y = _display_formatters(UTM_ZONE, UTM_NORTHERN)

    # Equal aspect is valid when both horizontal and vertical axes carry the
    # same physical scale.  Mismatches that break it:
    #   depth_km=True  + horiz_km=False + _disp="model"  → m horiz, km vert
    #   depth_km=False + horiz_km=True  + _disp="model"  → km horiz, m vert
    # Compatible combinations: both False (m/m), both True (km/km),
    # _disp="utm" (km/km regardless of depth_km), _disp="latlon" (never equal).
    _horiz_km_eff = (sc == 1e-3)   # True whenever horizontal axis is in km
    _vert_km_eff  = depth_km
    _do_equal = (equal_aspect
                 and _disp in ("model", "utm")
                 and _horiz_km_eff == _vert_km_eff)

    # ── load model ───────────────────────────────────────────────────────────
    if out:
        print(f"  plot: reading model {os.path.basename(model_file)}")
    mesh = fviz.read_femtic_mesh(mesh_file)
    block = fviz.read_resistivity_block(model_file)
    rho_elem = fviz.map_regions_to_element_rho(
        block.region_of_elem, block.region_rho)
    rho_plot = fviz.prepare_rho_for_plotting(
        rho_elem,
        air_is_nan=True,
        ocean_value=float(ocean_value),
        region_of_elem=block.region_of_elem,
    )
    nodes = mesh.nodes   # (nn, 3)
    conn = mesh.conn    # (nelem, 4)
    # Surface elevation: minimum z among nodes of non-air elements only.
    # Air padding elements (region 0) extend far above the topography and
    # would give a spuriously large (negative) z_min if included.
    _non_air_mask = block.region_of_elem != 0
    _non_air_node_idx = np.unique(conn[_non_air_mask])
    _non_air_nodes = nodes[_non_air_node_idx]
    z_surf = float(_non_air_nodes[:, 2].min())
    if out:
        _hp_idx = int(np.argmin(_non_air_nodes[:, 2]))
        _hp_x, _hp_y, _hp_z = _non_air_nodes[_hp_idx]
        _hp_E = _hp_x + UTM_ORIGIN_E
        _hp_N = _hp_y + UTM_ORIGIN_N
        _hp_lat, _hp_lon = utl.utm_to_latlon_zn(_hp_E, _hp_N,
                                                  UTM_ZONE, UTM_NORTHERN)
        print(f"  plot: mesh highest point (non-air):  "
              f"elev = {-_hp_z:.1f} m  "
              f"lat = {_hp_lat:.5f}°  lon = {_hp_lon:.5f}°  "
              f"(model-local x={_hp_x:.1f} m  y={_hp_y:.1f} m)")
        print(f"  plot: {len(slices)} panel(s), exact plane-intersection method")

    # ── colormap ─────────────────────────────────────────────────────────────
    cmap_obj = matplotlib.colormaps[cmap].copy()
    cmap_obj.set_bad(alpha=0.0)

    # ── colour normalisation ─────────────────────────────────────────────────
    if clim is not None:
        norm = mcolors.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            _lall = np.log10(rho_plot[np.isfinite(rho_plot)])
        _lall = _lall[np.isfinite(_lall)]
        norm = mcolors.Normalize(vmin=float(np.nanmin(_lall)),
                                 vmax=float(np.nanmax(_lall)))

    # ── figure layout ────────────────────────────────────────────────────────
    n_panels = len(slices)
    _dz_sc = 1e-3 if depth_km else 1.0   # scale for depth span in aspect calc

    # Grid shape — default to one row.
    _nrows = int(nrows) if nrows is not None else 1
    _ncols = int(ncols) if ncols is not None else n_panels
    if _nrows * _ncols < n_panels:
        raise ValueError(
            f"plot_model_slices: grid {_nrows}×{_ncols} = {_nrows*_ncols} cells "
            f"< {n_panels} slices — increase PLOT_NROWS/PLOT_NCOLS."
        )

    if figsize is not None:
        _fig_w, _fig_h = float(figsize[0]), float(figsize[1])
    else:
        _panel_h = float(panel_height)
        if panel_width is not None:
            _col_w = float(panel_width)
            _col_widths = [_col_w] * _ncols
        elif _do_equal:
            # Per-panel width from aspect ratio; use max width per column.
            _pw = []
            for spec in slices:
                kind = spec.get("kind", "map")
                _xl = spec.get("xlim", xlim)
                _yl = spec.get("ylim", ylim)
                _zl = spec.get("zlim", zlim)
                if kind == "map":
                    hspan = (_xl[1] - _xl[0]) * sc if _xl is not None else _panel_h * 200
                    vspan = (_yl[1] - _yl[0]) * sc if _yl is not None else _panel_h * 200
                elif kind == "ns":
                    hspan = (_yl[1] - _yl[0]) * sc    if _yl is not None else _panel_h * 200
                    vspan = (_zl[1] - _zl[0]) * _dz_sc if _zl is not None else _panel_h * 200
                elif kind == "ew":
                    hspan = (_xl[1] - _xl[0]) * sc    if _xl is not None else _panel_h * 200
                    vspan = (_zl[1] - _zl[0]) * _dz_sc if _zl is not None else _panel_h * 200
                else:
                    hspan = vspan = 1.0
                ratio = hspan / vspan if vspan > 0 else 1.0
                _pw.append(_panel_h * ratio)
            # Pad to full grid, assign panels column-by-column.
            _pw += [_panel_h] * (_nrows * _ncols - len(_pw))
            _col_widths = [
                max(_pw[c::_ncols]) for c in range(_ncols)
            ]
        else:
            _col_widths = [_panel_h] * _ncols
        _fig_w = sum(_col_widths)
        _fig_h = _panel_h * _nrows

    fig, axes = plt.subplots(_nrows, _ncols,
                             figsize=(_fig_w, _fig_h),
                             squeeze=False)
    # Flatten to 1-D list; hide surplus cells beyond n_panels.
    _ax_flat = [axes[r][c] for r in range(_nrows) for c in range(_ncols)]
    for ax in _ax_flat[n_panels:]:
        ax.set_visible(False)
    axes = _ax_flat[:n_panels]
    if air_bgcolor is not None:
        for ax in axes:
            ax.set_facecolor(air_bgcolor)

    _site_xys = site_xys or []   # list of (sn, x_m, y_m); empty → no markers

    # ── render each panel ────────────────────────────────────────────────────
    for ax, spec in zip(axes, slices):
        kind = spec.get("kind", "map")
        title = spec.get("title", None)
        _xlim = spec.get("xlim", xlim)
        _ylim = spec.get("ylim", ylim)
        _zlim = spec.get("zlim", zlim)
        mappable = None

        # ── map (z = const) ───────────────────────────────────────────────
        if kind == "map":
            z0 = float(spec.get("z0", 0.0))
            if out:
                print(f"    map slice z={z0:.0f} m …")
            normal = np.array([0., 0., 1.])
            point = np.array([0., 0., z0])
            u_ax = np.array([1., 0., 0.])
            v_ax = np.array([0., 1., 0.])
            polys, vals = _slice_geometry(nodes, conn, rho_plot,
                                          normal, point, u_ax, v_ax)
            polys_d = [[((px + dE) * sc, (py + dN) * sc) for px, py in poly]
                       for poly in polys]
            mappable = _plot_slice_panel(ax, polys_d, vals,
                                         cmap_obj=cmap_obj, norm=norm,
                                         ocean_color=ocean_color, ocean_value=ocean_value, invert_v=False)
            ax.set_xlabel(f"x (easting){sfx}")
            ax.set_ylabel(f"y (northing){sfx}")
            if _xlim is not None:
                ax.set_xlim([(v + dE) * sc for v in _xlim])
            if _ylim is not None:
                ax.set_ylim([(v + dN) * sc for v in _ylim])
            if _fmt_x is not None:
                ax.xaxis.set_major_formatter(_fmt_x)
            if _fmt_y is not None:
                ax.yaxis.set_major_formatter(_fmt_y)
            if _do_equal:
                ax.set_aspect("equal", adjustable="box")
            if title is None:
                title = f"Map  z = {z0 / 1000:.1f} km"
            for sn, sx_m, sy_m, _elev in (_site_xys if sites_in_maps else []):
                mk = dict(SITE_MARKER)
                mk.setdefault("label", f"Site {sn}")
                ax.plot((sx_m + dE) * sc, (sy_m + dN) * sc,
                        linestyle="none", **mk)
            for _mm in (map_markers or []):
                _lat, _lon = _mm["latlon"]
                _mx_m, _my_m = fem.latlon_to_model(
                    _lat, _lon, UTM_ZONE, UTM_NORTHERN,
                    UTM_ORIGIN_E, UTM_ORIGIN_N)
                _mk = dict(
                    marker=_mm.get("marker", "+"),
                    color=_mm.get("color", "black"),
                    ms=_mm.get("ms", 8),
                    zorder=_mm.get("zorder", 11),
                    label=_mm.get("name", None),
                )
                _mk.update({k: v for k, v in _mm.items()
                             if k not in ("latlon", "marker", "color",
                                          "ms", "zorder", "name")})
                ax.plot((_mx_m + dE) * sc, (_my_m + dN) * sc,
                        linestyle="none", **_mk)

        # ── NS curtain (x = const) ────────────────────────────────────────
        elif kind == "ns":
            x0 = float(spec.get("x0", 0.0))
            if out:
                print(f"    NS slice x={x0:.0f} m …")
            normal, point, u_ax, v_ax, inv = _axis_slice_params(0, x0)
            polys, vals = _slice_geometry(nodes, conn, rho_plot,
                                          normal, point, u_ax, v_ax)
            # u_ax points in y (northing) direction; vertical axis: -pz (depth positive-down)
            polys_d = [[((py + dN) * sc, -pz * _dz_sc) for py, pz in poly] for poly in polys]
            mappable = _plot_slice_panel(ax, polys_d, vals,
                                         cmap_obj=cmap_obj, norm=norm,
                                         ocean_color=ocean_color, ocean_value=ocean_value, invert_v=inv)
            ax.set_xlabel(f"y (northing){sfx}")
            ax.set_ylabel("depth (km)" if depth_km else "depth (m)")
            if _ylim is not None:
                ax.set_xlim([(v + dN) * sc for v in _ylim])
            if _zlim is not None:
                ax.set_ylim([_zlim[1] * _dz_sc, _zlim[0] * _dz_sc])
            if _fmt_y is not None:
                ax.xaxis.set_major_formatter(_fmt_y)
            if _do_equal:
                ax.set_aspect("equal", adjustable="box")
            if title is None:
                title = f"N-S  easting = {(x0 + UTM_ORIGIN_E) / 1000:.1f} km"
            if sites_in_slices:
                _pd = projection_dist
                for sn, sx_m, sy_m, _elev in _site_xys:
                    if _pd is not None and abs(sx_m - x0) > _pd:
                        continue
                    mk = dict(SITE_MARKER_SLICES)
                    mk.setdefault("label", f"Site {sn}")
                    ax.plot((sy_m + dN) * sc, -_elev * _dz_sc, linestyle="none", **mk)

        # ── EW curtain (y = const) ────────────────────────────────────────
        elif kind == "ew":
            y0 = float(spec.get("y0", 0.0))
            if out:
                print(f"    EW slice y={y0:.0f} m …")
            normal, point, u_ax, v_ax, inv = _axis_slice_params(1, y0)
            polys, vals = _slice_geometry(nodes, conn, rho_plot,
                                          normal, point, u_ax, v_ax)
            # u_ax points in x (easting) direction; vertical axis: -pz
            polys_d = [[((px + dE) * sc, -pz * _dz_sc) for px, pz in poly] for poly in polys]
            mappable = _plot_slice_panel(ax, polys_d, vals,
                                         cmap_obj=cmap_obj, norm=norm,
                                         ocean_color=ocean_color, ocean_value=ocean_value, invert_v=inv)
            ax.set_xlabel(f"x (easting){sfx}")
            ax.set_ylabel("depth (km)" if depth_km else "depth (m)")
            if _xlim is not None:
                ax.set_xlim([(v + dE) * sc for v in _xlim])
            if _zlim is not None:
                ax.set_ylim([_zlim[1] * _dz_sc, _zlim[0] * _dz_sc])
            if _fmt_x is not None:
                ax.xaxis.set_major_formatter(_fmt_x)
            if _do_equal:
                ax.set_aspect("equal", adjustable="box")
            if title is None:
                title = f"E-W  northing = {(y0 + UTM_ORIGIN_N) / 1000:.1f} km"
            if sites_in_slices:
                _pd = projection_dist
                for sn, sx_m, sy_m, _elev in _site_xys:
                    if _pd is not None and abs(sy_m - y0) > _pd:
                        continue
                    mk = dict(SITE_MARKER_SLICES)
                    mk.setdefault("label", f"Site {sn}")
                    ax.plot((sx_m + dE) * sc, -_elev * _dz_sc, linestyle="none", **mk)

        # ── arbitrary plane ────────────────────────────────────────────────
        elif kind == "plane":
            _pt = np.asarray(spec.get("point", [0., 0., 0.]), dtype=float)
            _strike = float(spec.get("strike", 0.0))
            _dip = float(spec.get("dip", 90.0))
            if out:
                print(f"    plane slice strike={_strike:.0f}° dip={_dip:.0f}° …")
            normal = _strike_dip_to_normal(_strike, _dip)
            u_ax, v_ax = _plane_basis(normal)
            polys, vals = _slice_geometry(nodes, conn, rho_plot,
                                          normal, _pt, u_ax, v_ax)
            mappable = _plot_slice_panel(ax, polys, vals,
                                         cmap_obj=cmap_obj, norm=norm,
                                         ocean_color=ocean_color, ocean_value=ocean_value, invert_v=True)
            ax.set_xlabel("along-strike (m)")
            ax.set_ylabel("down-dip (km)" if depth_km else "down-dip (m)")
            if _xlim is not None:
                ax.set_xlim(_xlim)
            if _ylim is not None:
                ax.set_ylim(_ylim)
            if title is None:
                title = f"Plane  str={_strike:.0f}°  dip={_dip:.0f}°"
            if sites_in_slices:
                _pd = projection_dist
                for sn, sx_m, sy_m, _elev in _site_xys:
                    site_xyz = np.array([sx_m, sy_m, -_elev]) - _pt
                    perp_dist = abs(float(np.dot(site_xyz, normal)))
                    if _pd is not None and perp_dist > _pd:
                        continue
                    u_coord = float(np.dot(site_xyz, u_ax))
                    mk = dict(SITE_MARKER_SLICES)
                    mk.setdefault("label", f"Site {sn}")
                    ax.plot(u_coord, -_elev * _dz_sc, linestyle="none", **mk)

        else:
            ax.set_visible(False)
            print(f"  plot: unknown slice kind {kind!r} — skipped.")
            continue

        if mappable is not None:
            cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("log₁₀(ρ / Ω·m)", fontsize=8)
            cb.ax.tick_params(labelsize=7)

        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        _show_legend = (
            (_site_xys and (
                (sites_in_maps and kind == "map") or
                (sites_in_slices and kind in ("ns", "ew", "plane"))
            )) or
            (map_markers and kind == "map" and
             any(m.get("name") for m in map_markers))
        )
        if _show_legend:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Model: {os.path.basename(model_file)}", fontsize=10)
    fig.tight_layout()

    if plot_file is not None:
        fig.savefig(plot_file, dpi=dpi, bbox_inches="tight")
        if out:
            print(f"  plot: saved → {plot_file}")
    else:
        plt.show()


def _resolve_borehole_xy(spec: dict, zone: int, northern: bool) -> tuple[float, float]:
    """Resolve borehole x/y position specs to model-local metres.

    Accepts the same (value, "crs") tagging as PLOT_SLICES position keys.

    Parameters
    ----------
    spec     : borehole dict from BOREHOLE_SITES
    zone     : UTM zone number
    northern : hemisphere flag

    Returns
    -------
    x_m, y_m : model-local metres
    """
    return _resolve_x0(spec["x"], zone, northern), \
        _resolve_y0(spec["y"], zone, northern)


def plot_borehole_logs(
    model_file: str,
    mesh_file: str,
    borehole_sites: list,
    *,
    zone: int,
    northern: bool,
    clim=None,
    borehole_style: dict | None = None,
    shared: bool = True,
    plot_file=None,
    dpi: int = 200,
    out: bool = True,
):
    """Produce a 1-D log₁₀(ρ) vs depth figure for a list of boreholes.

    For each borehole spec in *borehole_sites* the resistivity is sampled
    at regular depth intervals by point-in-element search (``extract_borehole_log``).
    Air / out-of-mesh levels are plotted as gaps (NaN).

    Parameters
    ----------
    model_file     : resistivity block file
    mesh_file      : mesh.dat file
    borehole_sites : list of borehole spec dicts (from ``BOREHOLE_SITES``)
    zone           : UTM zone number (for CRS conversion of x/y specs)
    northern       : hemisphere flag
    clim           : [log10_min, log10_max] for the x-axis; None = auto
    borehole_style : Matplotlib line kwargs applied to every trace; None = defaults
    shared         : True → all boreholes on one axes; False → one panel each
    plot_file      : save path; None = interactive show()
    dpi            : figure DPI
    out            : verbose progress
    """
    if fviz is None:
        print("  plot_borehole_logs: femtic_viz not available — skipping.")
        return
    if not borehole_sites:
        print("  plot_borehole_logs: BOREHOLE_SITES is empty — skipping.")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  plot_borehole_logs: Matplotlib not available — skipping.")
        return

    # ── load model ──────────────────────────────────────────────────────────
    if out:
        print(f"  boreholes: reading model {os.path.basename(model_file)}")
    mesh = fviz.read_femtic_mesh(mesh_file)
    block = fviz.read_resistivity_block(model_file)
    rho_elem = fviz.map_regions_to_element_rho(
        block.region_of_elem, block.region_rho)
    rho_plot = fviz.prepare_rho_for_plotting(
        rho_elem,
        air_is_nan=True,
        ocean_value=float(OCEAN_RHO),
        region_of_elem=block.region_of_elem,
    )
    nodes = mesh.nodes
    conn = mesh.conn

    style = dict(lw=1.2, marker="none")
    if borehole_style:
        style.update(borehole_style)

    # ── sample each borehole ────────────────────────────────────────────────
    n = len(borehole_sites)
    logs = []
    for spec in borehole_sites:
        name = spec.get("name", "?")
        x_m, y_m = _resolve_borehole_xy(spec, zone, northern)
        z_top = float(spec.get("z_top", 0.0))
        z_bot = float(spec.get("z_bot", 20000.0))
        dz = float(spec.get("dz", 200.0))
        if out:
            print(f"  borehole {name!r}  x={x_m:.0f} m  y={y_m:.0f} m "
                  f"  z=[{z_top:.0f}..{z_bot:.0f}]  dz={dz:.0f} m")
        depths, rho = fem.extract_borehole_log(
            nodes, conn, rho_plot, x_m, y_m, z_top, z_bot, dz, out=out
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rho = np.where(rho > 0, np.log10(rho), np.nan)
        logs.append(dict(name=name, depths=depths, log_rho=log_rho))

    # ── figure layout ───────────────────────────────────────────────────────
    if shared:
        fig, ax_arr = plt.subplots(1, 1, figsize=(4, 6))
        ax_arr = [ax_arr] * n
    else:
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 6), sharey=True,
                                 squeeze=False)
        ax_arr = list(axes[0])

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [c["color"] for c in prop_cycle]

    for idx, (spec, log) in enumerate(zip(borehole_sites, logs)):
        ax = ax_arr[idx]
        col = colors[idx % len(colors)]
        ax.plot(log["log_rho"], log["depths"],
                color=col, label=log["name"], **style)
        ax.invert_yaxis()
        ax.set_ylabel("depth (m)")
        ax.set_xlabel("log₁₀(ρ / Ω·m)")
        if clim is not None:
            ax.set_xlim(clim)
        if not shared:
            ax.set_title(log["name"], fontsize=9)

    if shared:
        ax_arr[0].legend(fontsize=8)
        ax_arr[0].set_title("Borehole resistivity logs", fontsize=9)

    # x-grid on every axes (set comprehension avoids duplicates for shared)
    for ax in set(ax_arr):
        ax.grid(axis="x", lw=0.4, alpha=0.5)

    fig.suptitle(f"Model: {os.path.basename(model_file)}", fontsize=10)
    fig.tight_layout()

    if plot_file is not None:
        fig.savefig(plot_file, dpi=dpi, bbox_inches="tight")
        if out:
            print(f"  boreholes: saved → {plot_file}")
    else:
        plt.show()


# ===========================================================================
# Main
# ===========================================================================

# --- (1) Derive UTM zone from mesh-origin coordinates ---------------------
# Zone is derived from UTM_ORIGIN_LAT/LON (approximate geographic centre).
# These do not need to be exact for zone derivation; any representative point
# in the survey area suffices.  UTM_ORIGIN_E/N are refined in step (1b)
# when ESTIMATE_ORIGIN = True.
UTM_ZONE, UTM_NORTHERN = _utm_zone_from_origin()
hemi = "N" if UTM_NORTHERN else "S"
_proj_backend = "pyproj" if _HAVE_PYPROJ else "built-in Helmert series"
print(f"UTM zone: {UTM_ZONE}{hemi}  "
      f"(origin lat={UTM_ORIGIN_LAT:.4f}°, lon={UTM_ORIGIN_LON:.4f}°)  "
      f"[projection: {_proj_backend}]")
print()

# --- (1b) Optionally estimate UTM_ORIGIN_E / UTM_ORIGIN_N from site.dat ---
if ORIGIN_METHOD is not None:
    if SITE_DAT is None or not os.path.isfile(SITE_DAT):
        print(f"  WARNING: ORIGIN_METHOD={ORIGIN_METHOD!r} requested but "
              f"SITE_DAT is not available — using hard-coded origin.")
    else:
        _sdat = fem.read_site_dat(SITE_DAT)
        if not _sdat:
            print(f"  WARNING: SITE_DAT is empty — using hard-coded origin.")
        else:
            _Es = np.array([d["easting"]  for d in _sdat])
            _Ns = np.array([d["northing"] for d in _sdat])
            if ORIGIN_METHOD == "box":
                UTM_ORIGIN_E = 0.5 * (_Es.min() + _Es.max())
                UTM_ORIGIN_N = 0.5 * (_Ns.min() + _Ns.max())
            elif ORIGIN_METHOD == "average":
                UTM_ORIGIN_E = float(_Es.mean())
                UTM_ORIGIN_N = float(_Ns.mean())
            else:
                sys.exit(f"Unknown ORIGIN_METHOD {ORIGIN_METHOD!r}; "
                         f"use None, 'box', or 'average'.")
            UTM_ORIGIN_LAT, UTM_ORIGIN_LON = utl.utm_to_latlon_zn(
                UTM_ORIGIN_E, UTM_ORIGIN_N, UTM_ZONE, UTM_NORTHERN
            )
            UTM_ZONE, UTM_NORTHERN = _utm_zone_from_origin()
            if OUT:
                print(f"Origin estimated ({ORIGIN_METHOD}, {len(_sdat)} sites):")
                print(f"  UTM_ORIGIN_E   = {UTM_ORIGIN_E:.1f} m")
                print(f"  UTM_ORIGIN_N   = {UTM_ORIGIN_N:.1f} m")
                print(f"  UTM_ORIGIN_LAT = {UTM_ORIGIN_LAT:.6f}°")
                print(f"  UTM_ORIGIN_LON = {UTM_ORIGIN_LON:.6f}°")
                print(f"  UTM_ZONE       = {UTM_ZONE}{'N' if UTM_NORTHERN else 'S'}")
                print()

# --- (2) Resolve slice positions to model-local metres ---------------------
slices_resolved = resolve_slices(PLOT_SLICES, UTM_ZONE, UTM_NORTHERN)
if OUT:
    print()

# --- (3) Optionally read site position(s) ----------------------------
# Strategy:
#   (a) site.dat present → UTM easting/northing converted to model-local via
#       current UTM_ORIGIN_E/N.  Full coordinate display available.
#   (b) observe.dat fallback → model-local coordinates read directly.
#       No UTM or lat/lon display possible.
site_xys = []       # list of (label, x_m, y_m)
_sites_from_obs = False   # flag: observe.dat path used (model-local only)
_need_sites = PLOT_SITES_MAPS or PLOT_SITES_SLICES
if _need_sites and SITE_DAT is not None:
    print(f"Reading site positions from site.dat: {SITE_DAT}")
    _rows = fem.read_site_dat(SITE_DAT, site_names=SITE_NAMES)
    for row in _rows:
        sx_m, sy_m = fem.utm_to_model(row["easting"], row["northing"],
                                      UTM_ORIGIN_E, UTM_ORIGIN_N)
        site_xys.append((row["name"], sx_m, sy_m, float(row.get("elev", 0.0))))
        print(f"  {row['name']}: model-local x = {sx_m / 1000:.3f} km,"
              f"  y = {sy_m / 1000:.3f} km")
        if DISPLAY_COORDS in ("utm", "latlon"):
            print(f"           UTM         E = {sx_m + UTM_ORIGIN_E:.1f} m,"
                  f"  N = {sy_m + UTM_ORIGIN_N:.1f} m")
    if not site_xys:
        print("  (no matching sites found in site.dat)")
    print()
elif _need_sites and SITE_NUMBER is not None:
    _site_nums = (SITE_NUMBER if isinstance(SITE_NUMBER, (list, tuple))
                  else [SITE_NUMBER])
    print(f"Reading site positions from observe.dat: {OBSERVE_FILE}")
    print(f"  Note: observe.dat provides model-local coordinates only.")
    for _sn in _site_nums:
        sx_m, sy_m = fem.read_site_position(OBSERVE_FILE, _sn)
        site_xys.append((_sn, sx_m, sy_m, 0.0))
        print(f"  site {_sn}: model-local x = {sx_m / 1000:.3f} km,"
              f"  y = {sy_m / 1000:.3f} km")
    _sites_from_obs = True
    print()

# --- (4) Plot slices -------------------------------------------------------
if fviz is None:
    sys.exit("femtic_viz not available — cannot plot.  Check your installation.")

print(f"Plotting model: {MODEL_FILE}")
plot_model_slices(
    model_file=MODEL_FILE,
    mesh_file=MESH_FILE,
    slices=slices_resolved,
    cmap=PLOT_CMAP,
    clim=PLOT_CLIM,
    xlim=PLOT_XLIM,
    ylim=PLOT_YLIM,
    zlim=PLOT_ZLIM,
    ocean_color=PLOT_OCEAN_COLOR,
    ocean_value=OCEAN_RHO,
    air_bgcolor=PLOT_AIR_BGCOLOR,
    site_xys=site_xys,
    obs_coords_only=_sites_from_obs,
    projection_dist=PROJECTION_DIST,
    sites_in_maps=PLOT_SITES_MAPS,
    sites_in_slices=PLOT_SITES_SLICES,
    map_markers=MAP_MARKERS,
    plot_file=PLOT_FILE,
    dpi=PLOT_DPI,
    equal_aspect=PLOT_EQUAL_ASPECT,
    depth_km=DEPTH_KM,
    horiz_km=HORIZ_KM,
    nrows=PLOT_NROWS,
    ncols=PLOT_NCOLS,
    panel_height=PLOT_PANEL_HEIGHT / 2.54,
    panel_width=PLOT_PANEL_WIDTH / 2.54 if PLOT_PANEL_WIDTH is not None else None,
    figsize=[v / 2.54 for v in PLOT_FIGSIZE] if PLOT_FIGSIZE is not None else None,
    out=OUT,
)
print("Done.")

# --- (5) 3-D PyVista plot --------------------------------------------------
if PLOT3D:
    if fviz is None:
        print("  3-D plot skipped: femtic_viz not available.")
    else:
        print(f"Rendering 3-D model: {MODEL_FILE}")
        fviz.plot_model_3d(
            mesh_file=MESH_FILE,
            block_file=MODEL_FILE,
            scalar=PLOT3D_SCALAR,
            clim=PLOT3D_CLIM,
            cmap=PLOT3D_CMAP,
            slice_x=PLOT3D_SLICE_X,
            slice_y=PLOT3D_SLICE_Y,
            slice_z=PLOT3D_SLICE_Z,
            slice_planes=PLOT3D_SLICE_PLANES,
            isovalues=PLOT3D_ISOVALUES,
            iso_opacity=PLOT3D_ISO_OPACITY,
            iso_cmap=PLOT3D_CMAP,
            ocean_value=OCEAN_RHO,
            air_region_index=0,
            ocean_region_index=1,
            window_size=PLOT3D_WINDOW_SIZE,
            plot_file=PLOT3D_FILE,
            out=OUT,
        )
        print("3-D plot done.")

# --- (6) Borehole resistivity logs ----------------------------------------
if PLOT_BOREHOLE:
    if fviz is None:
        print("  Borehole plot skipped: femtic_viz not available.")
    elif not BOREHOLE_SITES:
        print("  Borehole plot skipped: BOREHOLE_SITES is empty.")
    else:
        print(f"Sampling {len(BOREHOLE_SITES)} borehole(s) …")
        plot_borehole_logs(
            model_file=MODEL_FILE,
            mesh_file=MESH_FILE,
            borehole_sites=BOREHOLE_SITES,
            zone=UTM_ZONE,
            northern=UTM_NORTHERN,
            clim=BOREHOLE_XLIM,
            borehole_style=BOREHOLE_STYLE,
            shared=BOREHOLE_SHARED,
            plot_file=BOREHOLE_FILE,
            dpi=PLOT_DPI,
            out=OUT,
        )
        print("Borehole plot done.")
