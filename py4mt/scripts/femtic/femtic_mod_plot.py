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
    2026-05-26  Claude Sonnet 4.6 (Anthropic)
                Moved plot_model_slices (all inner geometry helpers) and
                plot_borehole_logs into femtic_viz.py; script now calls
                fviz.plot_model_slices / fviz.plot_borehole_logs directly.
                Removed script-level coordinate-conversion and display
                helpers (_utm_zone_from_origin, resolve_slices, _display_*,
                _display_formatters, etc.); main section calls utl/fem
                directly.  Removed math and pyproj imports. Added
                PLOT3D_VTU_FILE config var; changed PLOT3D_FILE default
                from .html to .png.  Script reduced from ~1520 to ~690
                lines.
    2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
                Passed PLOT_XLIM/YLIM/ZLIM to plot_model_3d (xlim/ylim/zlim)
                so VTU export and 3-D scene are spatially clipped to the
                same box as the 2-D slice panels.
                Added ALPHA_FILE / ALPHA_MODE / ALPHA_BLANK_THRESH config
                vars; passed to fviz.plot_model_slices for per-element
                polygon fading/blanking driven by a second block file.

@author: vrath
"""

import os
import sys
from pathlib import Path
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

# ---------------------------------------------------------------------------
# Alpha / blanking by second block file  (optional)
# ---------------------------------------------------------------------------
#: Path to a second resistivity_block_iterX.dat with the SAME mesh and
#: region structure as MODEL_FILE.  Its region_rho values are interpreted
#: directly as log10 weights (not as physical resistivities).  Cells with
#: a weight < ALPHA_BLANK_THRESH are suppressed on all slice panels.
#: Set to None to disable.
ALPHA_FILE = None   # e.g. WORK_DIR + "sensitivity_log10.dat"

#: How alpha weights drive visibility:
#:   "fade"  — proportional transparency between ALPHA_BLANK_THRESH (=0) and 0
#:   "blank" — hard cutoff: below ALPHA_BLANK_THRESH the polygon is omitted
ALPHA_MODE = "fade"

#: Log10 threshold at or below which polygons are blanked / fully faded.
#: Should be <= 0 (e.g. -1.0).  Default 0.0 removes anything < 0.
ALPHA_BLANK_THRESH = 0.0

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
PLOT_NCOLS = 2   # e.g. 4

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
PLOT3D = True

#: Output file for the 3-D rendered view.
#:   .vtu / .vtk → VTK unstructured-grid for ParaView (no rendering needed).
#:   .html       → interactive WebGL in a browser (requires pyvista[jupyter] /
#:                 trame_vtk; use ParaView's own File → Export Scene instead).
#:   .png / .jpg → static screenshot.
#:   None        → open an interactive PyVista window (requires a display).
PLOT3D_FILE = WORK_DIR + "resistivity_block_iter0_3d.png"

#: Optional separate VTK export for ParaView / Zenodo deposit.
#:   .vtu  → VTK XML unstructured grid  (recommended for ParaView).
#:   .vtk  → legacy VTK binary/ASCII.
#:   None  → no grid file exported.
PLOT3D_VTU_FILE = WORK_DIR + "resistivity_block_iter0.vtu"

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
# Main
# ===========================================================================

# --- (1) Derive UTM zone from mesh-origin coordinates ---------------------
UTM_ZONE, UTM_NORTHERN = utl.utm_zone_from_latlon(
    UTM_ORIGIN_LAT, UTM_ORIGIN_LON, override=UTM_ZONE_OVERRIDE)
hemi = "N" if UTM_NORTHERN else "S"
print(f"UTM zone: {UTM_ZONE}{hemi}  "
      f"(origin lat={UTM_ORIGIN_LAT:.4f}°, lon={UTM_ORIGIN_LON:.4f}°)")
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
                UTM_ORIGIN_E, UTM_ORIGIN_N, UTM_ZONE, UTM_NORTHERN)
            UTM_ZONE, UTM_NORTHERN = utl.utm_zone_from_latlon(
                UTM_ORIGIN_LAT, UTM_ORIGIN_LON, override=UTM_ZONE_OVERRIDE)
            if OUT:
                print(f"Origin estimated ({ORIGIN_METHOD}, {len(_sdat)} sites):")
                print(f"  UTM_ORIGIN_E   = {UTM_ORIGIN_E:.1f} m")
                print(f"  UTM_ORIGIN_N   = {UTM_ORIGIN_N:.1f} m")
                print(f"  UTM_ORIGIN_LAT = {UTM_ORIGIN_LAT:.6f}°")
                print(f"  UTM_ORIGIN_LON = {UTM_ORIGIN_LON:.6f}°")
                print(f"  UTM_ZONE       = {UTM_ZONE}{'N' if UTM_NORTHERN else 'S'}")
                print()

# --- (2) Resolve slice positions to model-local metres ---------------------
slices_resolved = fem.resolve_slice_positions(
    PLOT_SLICES, UTM_ZONE, UTM_NORTHERN,
    UTM_ORIGIN_E, UTM_ORIGIN_N,
    UTM_ORIGIN_LAT, UTM_ORIGIN_LON,
    verbose=OUT,
)
if OUT:
    print()

# --- (3) Optionally read site position(s) ----------------------------------
site_xys = []
_sites_from_obs = False
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
fviz.plot_model_slices(
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
    site_marker=SITE_MARKER,
    site_marker_slices=SITE_MARKER_SLICES,
    map_markers=MAP_MARKERS,
    display_coords=DISPLAY_COORDS,
    utm_origin_e=UTM_ORIGIN_E,
    utm_origin_n=UTM_ORIGIN_N,
    utm_zone=UTM_ZONE,
    utm_northern=UTM_NORTHERN,
    utm_to_latlon_fn=utl.utm_to_latlon_zn,
    latlon_to_model_fn=fem.latlon_to_model,
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
    alpha_file=ALPHA_FILE,
    alpha_mode=ALPHA_MODE,
    alpha_blank_thresh=ALPHA_BLANK_THRESH,
    out=OUT,
)
print("Done.")

# --- (5) 3-D PyVista plot --------------------------------------------------
if PLOT3D:
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
        xlim=PLOT_XLIM,
        ylim=PLOT_YLIM,
        zlim=PLOT_ZLIM,
        window_size=PLOT3D_WINDOW_SIZE,
        plot_file=PLOT3D_FILE,
        vtu_file=PLOT3D_VTU_FILE,
        out=OUT,
    )
    print("3-D plot done.")

# --- (6) Borehole resistivity logs ----------------------------------------
if PLOT_BOREHOLE:
    if not BOREHOLE_SITES:
        print("  Borehole plot skipped: BOREHOLE_SITES is empty.")
    else:
        def _resolve_borehole_xy(spec):
            return (
                fem.resolve_pos_x(spec["x"], UTM_ZONE, UTM_NORTHERN,
                                  UTM_ORIGIN_E, UTM_ORIGIN_N,
                                  UTM_ORIGIN_LAT, UTM_ORIGIN_LON),
                fem.resolve_pos_y(spec["y"], UTM_ZONE, UTM_NORTHERN,
                                  UTM_ORIGIN_E, UTM_ORIGIN_N,
                                  UTM_ORIGIN_LAT, UTM_ORIGIN_LON),
            )
        print(f"Sampling {len(BOREHOLE_SITES)} borehole(s) …")
        fviz.plot_borehole_logs(
            model_file=MODEL_FILE,
            mesh_file=MESH_FILE,
            borehole_sites=BOREHOLE_SITES,
            resolve_xy_fn=_resolve_borehole_xy,
            ocean_value=OCEAN_RHO,
            clim=BOREHOLE_XLIM,
            borehole_style=BOREHOLE_STYLE,
            shared=BOREHOLE_SHARED,
            plot_file=BOREHOLE_FILE,
            dpi=PLOT_DPI,
            out=OUT,
        )
        print("Borehole plot done.")
