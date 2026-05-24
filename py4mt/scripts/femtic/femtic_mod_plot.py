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

# try:
#     from pyproj import Transformer as _Transformer
#     _HAVE_PYPROJ = True
# except ImportError:
#     _HAVE_PYPROJ = False

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
MESH_FILE  = WORK_DIR + "mesh.dat"

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

AIR_RHO   = 1.0e9   # Ω·m  (region 0)
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
DISPLAY_COORDS = "model"

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

#: Marker style for map panels; dashed vertical line for curtain panels.
SITE_MARKER = dict(marker="v", color="black", ms=8, zorder=10,
                   label=None)   # label filled in automatically

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
#: Ubinas summit: -16.3500, -70.8700

PLOT_SLICES = [
    # Plain float — model-local metres (backward-compatible):
    dict(kind="map",  z0=5000.0),
    dict(kind="map",  z0=15000.0),
    # UTM easting for the NS curtain:
    # dict(kind="ns",   x0=(229047.0, "utm")),
    dict(kind="ns",   x0=(-70.8700, "latlon")),
    # Geographic latitude for the EW curtain:
    dict(kind="ew",   y0=(-16.3500, "latlon")),
]

#: Global axis limits in model-local metres — used for panels that do not
#: specify their own.  None → auto (inferred from data extent).
PLOT_XLIM = [-20000., 20000.]   # [xmin, xmax] metres — easting
PLOT_YLIM = [-20000., 20000.]   # [ymin, ymax] metres — northing
PLOT_ZLIM = [  -6000., 15000.]  # [zmin, zmax] metres — depth (z positive-down)

#: Equal aspect ratio for map and curtain panels.
#: True  → ax.set_aspect("equal") on map (x/y), ns (y/z), and ew (x/z) panels
#:         so that 1 m horizontal = 1 m vertical on screen.  Applies only when
#:         DISPLAY_COORDS is "model" or "utm" (both axes in metres).  Ignored
#:         for plane slices and when DISPLAY_COORDS is "latlon".
#: False → Matplotlib default (axes fill the available space).
PLOT_EQUAL_ASPECT = True

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
# Mesh-centre estimation from known site coordinates  (optional)
# ---------------------------------------------------------------------------
#: Set ESTIMATE_ORIGIN = True to compute UTM_ORIGIN_E / UTM_ORIGIN_N.
#:
#: Two methods are available (selected automatically):
#:
#:   Bounding-box centre (femticPY-compatible, DEFAULT when
#:   CALIBRATION_SITES is empty and SITE_DAT is set):
#:     origin = midpoint of the bounding box of all site UTM coordinates
#:     read from SITE_DAT.  No observe.dat is needed.
#:
#:   Calibration-site pairs (classic):
#:     Each entry in CALIBRATION_SITES provides a site whose model-local
#:     position (from observe.dat or site.dat) and geographic position are
#:     both known.  UTM_ORIGIN_E/N = mean(E_site − x_m, N_site − y_m).
#:     At least 1 site is required; 3+ is recommended.
#:
#: Each entry in CALIBRATION_SITES is a dict with:
#:   "site"    : int    site number (matched against observe.dat)
#:   "crs"     : str    "latlon" or "utm"
#:   "coords"  : for "latlon" → [lon_deg, lat_deg]
#:               for "utm"    → [E_m, N_m]
#:
#: The result always overwrites UTM_ORIGIN_E / UTM_ORIGIN_N for this run.
#: Set UPDATE_CONFIG = True to also feed UTM_ORIGIN_LAT / UTM_ORIGIN_LON back
#: into the module globals and re-derive the UTM zone — so all subsequent
#: coordinate conversions in this run use the estimated centre rather than
#: the hard-coded values above.  The printed values can then be copied back
#: into the Configuration block for future runs.
ESTIMATE_ORIGIN = False
UPDATE_CONFIG   = True    # feed estimated lat/lon back into globals (requires ESTIMATE_ORIGIN)

CALIBRATION_SITES = [
    # dict(site=1,  crs="latlon", coords=[-71.500, -16.380]),
    # dict(site=10, crs="latlon", coords=[-71.620, -16.450]),
    # dict(site=25, crs="utm",    coords=[224500., 8179300.]),
]


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
    """Return (dE, dN) to add to model-local metres for display axis ticks.

    For 'latlon' the polygons are still drawn in UTM metres (offset by the
    UTM origin) and tick labels are reformatted by ``_display_formatters``.
    """
    if DISPLAY_COORDS in ("utm", "latlon"):
        return UTM_ORIGIN_E, UTM_ORIGIN_N
    return 0.0, 0.0


def _display_suffix() -> str:
    """Return axis label suffix reflecting the display coordinate system."""
    if DISPLAY_COORDS == "utm":
        return " [UTM m]"
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
    plot_file=None,
    dpi: int = 200,
    equal_aspect: bool = True,
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
    site_xys    : list of (site_number, x_m, y_m) tuples in model-local metres;
                  each site is overplotted on every map/ns/ew panel.
                  None or empty list → no markers.
    plot_file   : save path; None = interactive show()
    dpi         : saved-figure DPI
    equal_aspect : if True, call ``ax.set_aspect("equal", adjustable="box")``
                  on map (x/y), ns (y/z), and ew (x/z) panels so that
                  1 m horizontal = 1 m vertical.  Applied only when
                  DISPLAY_COORDS is "model" or "utm"; ignored for plane panels
                  and when axes carry different units (e.g. latlon).
    out         : verbose progress
    """
    if fviz is None:
        print("  plot_model_slices: femtic_viz not available — skipping.")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.cm as mcm
        from matplotlib.collections import PolyCollection
    except ImportError:
        print("  plot_model_slices: Matplotlib not available — skipping.")
        return

    # ── internal geometry helpers ────────────────────────────────────────────

    def _axis_slice_params(axis: int, val: float):
        """Return (normal, point, u_ax, v_ax, invert_v) for an axis-aligned cut."""
        normals = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
        inv     = [False, False, False]
        pt      = np.zeros(3); pt[axis] = val
        n       = normals[axis]
        ref     = np.array([0., 0., 1.]) if axis != 2 else np.array([1., 0., 0.])
        u       = np.cross(n, ref); u /= np.linalg.norm(u)
        v       = np.cross(n, u);   v /= np.linalg.norm(v)
        return n, pt, u, v, inv[axis]

    def _tet_plane_intersection(verts, normal, d):
        dots = verts @ normal - d
        pos  = dots >= 0
        if pos.all() or (~pos).all():
            return []
        pts = []
        for i in range(4):
            for j in range(i + 1, 4):
                if pos[i] != pos[j]:
                    t = dots[i] / (dots[i] - dots[j])
                    pts.append(verts[i] + t * (verts[j] - verts[i]))
        c   = np.mean(pts, axis=0)
        u2d = np.cross(normal,
                       np.array([0., 0., 1.]) if abs(normal[2]) < 0.9
                       else np.array([1., 0., 0.]))
        if np.linalg.norm(u2d) < 1e-12:
            return pts
        u2d /= np.linalg.norm(u2d)
        v2d  = np.cross(normal, u2d)
        angles = [np.arctan2((p - c) @ v2d, (p - c) @ u2d) for p in pts]
        return [pts[k] for k in np.argsort(angles)]

    def _slice_geometry(nodes, conn, rho_arr, normal, point, u_ax, v_ax):
        d         = float(normal @ point)
        verts_all = nodes[conn]
        polys, vals = [], []
        for k, verts in enumerate(verts_all):
            pts3d = _tet_plane_intersection(verts, normal, d)
            if not pts3d:
                continue
            polys.append([(float(p @ u_ax), float(p @ v_ax)) for p in pts3d])
            with np.errstate(divide="ignore", invalid="ignore"):
                vals.append(math.log10(rho_arr[k]) if rho_arr[k] > 0 else float("nan"))
        return polys, np.asarray(vals, dtype=float)

    def _plot_slice_panel(ax, polys, vals, *, cmap_obj, norm,
                          ocean_color, ocean_value, invert_v):
        if not polys:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            ov_log = math.log10(ocean_value) if ocean_value > 0 else float("nan")
        is_ocean = np.isclose(vals, ov_log, atol=0.05)
        is_air   = ~np.isfinite(vals)
        is_data  = ~is_ocean & ~is_air
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
        ref = np.array([0., 1., 0.]) if abs(normal[1]) < 0.9 else np.array([1., 0., 0.])
        u   = np.cross(normal, ref); u /= np.linalg.norm(u)
        v   = np.cross(u, normal);   v /= np.linalg.norm(v)
        return u, v

    # ── display offset ───────────────────────────────────────────────────────
    dE, dN   = _display_offset()
    sfx      = _display_suffix()
    _fmt_x, _fmt_y = _display_formatters(UTM_ZONE, UTM_NORTHERN)

    # Equal aspect applies only when both axes are in metres (model or utm).
    # latlon display has different scales on x/y so aspect="equal" is wrong.
    _do_equal = equal_aspect and DISPLAY_COORDS.lower() in ("model", "utm")

    # ── load model ───────────────────────────────────────────────────────────
    if out:
        print(f"  plot: reading model {os.path.basename(model_file)}")
    mesh     = fviz.read_femtic_mesh(mesh_file)
    block    = fviz.read_resistivity_block(model_file)
    rho_elem = fviz.map_regions_to_element_rho(block.region_of_elem, block.region_rho)
    rho_plot = fviz.prepare_rho_for_plotting(
        rho_elem,
        air_is_nan=True,
        ocean_value=float(ocean_value),
        region_of_elem=block.region_of_elem,
    )
    nodes = mesh.nodes   # (nn, 3)
    conn  = mesh.conn    # (nelem, 4)
    if out:
        print(f"  plot: {len(slices)} panel(s), exact plane-intersection method")

    # ── colormap ─────────────────────────────────────────────────────────────
    cmap_obj = (mcm.colormaps[cmap] if hasattr(mcm, "colormaps")
                else mcm.get_cmap(cmap)).copy()
    cmap_obj.set_bad(alpha=0.0)

    # ── colour normalisation ─────────────────────────────────────────────────
    if clim is not None:
        norm = mcolors.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            _lall = np.log10(rho_plot[np.isfinite(rho_plot)])
        _lall = _lall[np.isfinite(_lall)]
        norm  = mcolors.Normalize(vmin=float(np.nanmin(_lall)),
                                  vmax=float(np.nanmax(_lall)))

    # ── figure layout ────────────────────────────────────────────────────────
    n_panels = len(slices)
    _panel_h = 5.0   # inches — height of each panel

    if _do_equal:
        # Compute each panel's width from the aspect ratio of its limits so
        # that set_aspect("equal") does not waste whitespace.
        _panel_widths = []
        for spec in slices:
            kind  = spec.get("kind", "map")
            _xl   = spec.get("xlim", xlim)
            _yl   = spec.get("ylim", ylim)
            _zl   = spec.get("zlim", zlim)
            if kind == "map":
                hspan = (_xl[1] - _xl[0]) if _xl is not None else _panel_h * 200
                vspan = (_yl[1] - _yl[0]) if _yl is not None else _panel_h * 200
            elif kind == "ns":
                hspan = (_yl[1] - _yl[0]) if _yl is not None else _panel_h * 200
                vspan = (_zl[1] - _zl[0]) if _zl is not None else _panel_h * 200
            elif kind == "ew":
                hspan = (_xl[1] - _xl[0]) if _xl is not None else _panel_h * 200
                vspan = (_zl[1] - _zl[0]) if _zl is not None else _panel_h * 200
            else:
                # plane or unknown — use square panel
                hspan = vspan = 1.0
            ratio = hspan / vspan if vspan > 0 else 1.0
            _panel_widths.append(_panel_h * ratio)
        _fig_w = sum(_panel_widths)
    else:
        _panel_widths = [5.0] * n_panels
        _fig_w = 5.0 * n_panels

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(_fig_w, _panel_h),
                             squeeze=False)
    axes = axes[0]
    if air_bgcolor is not None:
        for ax in axes:
            ax.set_facecolor(air_bgcolor)

    _site_xys = site_xys or []   # list of (sn, x_m, y_m); empty → no markers

    # ── render each panel ────────────────────────────────────────────────────
    for ax, spec in zip(axes, slices):
        kind  = spec.get("kind", "map")
        title = spec.get("title", None)
        _xlim = spec.get("xlim", xlim)
        _ylim = spec.get("ylim", ylim)
        _zlim = spec.get("zlim", zlim)
        mappable = None

        # ── map (z = const) ───────────────────────────────────────────────
        if kind == "map":
            z0     = float(spec.get("z0", 0.0))
            if out: print(f"    map slice z={z0:.0f} m …")
            normal = np.array([0., 0., 1.])
            point  = np.array([0., 0., z0])
            u_ax   = np.array([1., 0., 0.])
            v_ax   = np.array([0., 1., 0.])
            polys, vals = _slice_geometry(nodes, conn, rho_plot,
                                          normal, point, u_ax, v_ax)
            polys_d = [[(px + dE, py + dN) for px, py in poly] for poly in polys]
            mappable = _plot_slice_panel(ax, polys_d, vals,
                cmap_obj=cmap_obj, norm=norm,
                ocean_color=ocean_color, ocean_value=ocean_value, invert_v=False)
            ax.set_xlabel(f"x (easting){sfx}")
            ax.set_ylabel(f"y (northing){sfx}")
            if _xlim is not None: ax.set_xlim([v + dE for v in _xlim])
            if _ylim is not None: ax.set_ylim([v + dN for v in _ylim])
            if _fmt_x is not None: ax.xaxis.set_major_formatter(_fmt_x)
            if _fmt_y is not None: ax.yaxis.set_major_formatter(_fmt_y)
            if _do_equal: ax.set_aspect("equal", adjustable="box")
            if title is None: title = f"Map  z = {z0/1000:.1f} km"
            for sn, sx_m, sy_m in _site_xys:
                mk = dict(SITE_MARKER)
                mk.setdefault("label", f"Site {sn}")
                ax.plot(sx_m + dE, sy_m + dN, linestyle="none", **mk)

        # ── NS curtain (x = const) ────────────────────────────────────────
        elif kind == "ns":
            x0 = float(spec.get("x0", 0.0))
            if out: print(f"    NS slice x={x0:.0f} m …")
            normal, point, u_ax, v_ax, inv = _axis_slice_params(0, x0)
            polys, vals = _slice_geometry(nodes, conn, rho_plot,
                                          normal, point, u_ax, v_ax)
            # u_ax points in y (northing) direction
            polys_d = [[(py + dN, -pz) for py, pz in poly] for poly in polys]
            mappable = _plot_slice_panel(ax, polys_d, vals,
                cmap_obj=cmap_obj, norm=norm,
                ocean_color=ocean_color, ocean_value=ocean_value, invert_v=inv)
            ax.set_xlabel(f"y (northing){sfx}")
            ax.set_ylabel("depth (m)")
            if _ylim is not None: ax.set_xlim([v + dN for v in _ylim])
            if _zlim is not None: ax.set_ylim([_zlim[1], _zlim[0]])
            if _fmt_y is not None: ax.xaxis.set_major_formatter(_fmt_y)
            if _do_equal: ax.set_aspect("equal", adjustable="box")
            if title is None: title = f"N-S  x = {x0/1000:.1f} km"
            for sn, sx_m, sy_m in _site_xys:
                ax.axvline(sy_m + dN, color=SITE_MARKER["color"],
                           lw=1.2, ls="--", zorder=9,
                           label=f"Site {sn} (y)")

        # ── EW curtain (y = const) ────────────────────────────────────────
        elif kind == "ew":
            y0 = float(spec.get("y0", 0.0))
            if out: print(f"    EW slice y={y0:.0f} m …")
            normal, point, u_ax, v_ax, inv = _axis_slice_params(1, y0)
            polys, vals = _slice_geometry(nodes, conn, rho_plot,
                                          normal, point, u_ax, v_ax)
            # u_ax points in x (easting) direction
            polys_d = [[(px + dE, -pz) for px, pz in poly] for poly in polys]
            mappable = _plot_slice_panel(ax, polys_d, vals,
                cmap_obj=cmap_obj, norm=norm,
                ocean_color=ocean_color, ocean_value=ocean_value, invert_v=inv)
            ax.set_xlabel(f"x (easting){sfx}")
            ax.set_ylabel("depth (m)")
            if _xlim is not None: ax.set_xlim([v + dE for v in _xlim])
            if _zlim is not None: ax.set_ylim([_zlim[1], _zlim[0]])
            if _fmt_x is not None: ax.xaxis.set_major_formatter(_fmt_x)
            if _do_equal: ax.set_aspect("equal", adjustable="box")
            if title is None: title = f"E-W  y = {y0/1000:.1f} km"
            for sn, sx_m, sy_m in _site_xys:
                ax.axvline(sx_m + dE, color=SITE_MARKER["color"],
                           lw=1.2, ls="--", zorder=9,
                           label=f"Site {sn} (x)")

        # ── arbitrary plane ────────────────────────────────────────────────
        elif kind == "plane":
            _pt     = np.asarray(spec.get("point", [0., 0., 0.]), dtype=float)
            _strike = float(spec.get("strike", 0.0))
            _dip    = float(spec.get("dip", 90.0))
            if out:
                print(f"    plane slice strike={_strike:.0f}° dip={_dip:.0f}° …")
            normal      = _strike_dip_to_normal(_strike, _dip)
            u_ax, v_ax  = _plane_basis(normal)
            polys, vals = _slice_geometry(nodes, conn, rho_plot,
                                          normal, _pt, u_ax, v_ax)
            mappable = _plot_slice_panel(ax, polys, vals,
                cmap_obj=cmap_obj, norm=norm,
                ocean_color=ocean_color, ocean_value=ocean_value, invert_v=True)
            ax.set_xlabel("along-strike (m)")
            ax.set_ylabel("down-dip (m)")
            if _xlim is not None: ax.set_xlim(_xlim)
            if _ylim is not None: ax.set_ylim(_ylim)
            if title is None:
                title = f"Plane  str={_strike:.0f}°  dip={_dip:.0f}°"

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
        if _site_xys and kind in ("map", "ns", "ew"):
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
    mesh     = fviz.read_femtic_mesh(mesh_file)
    block    = fviz.read_resistivity_block(model_file)
    rho_elem = fviz.map_regions_to_element_rho(block.region_of_elem, block.region_rho)
    rho_plot = fviz.prepare_rho_for_plotting(
        rho_elem,
        air_is_nan=True,
        ocean_value=float(OCEAN_RHO),
        region_of_elem=block.region_of_elem,
    )
    nodes = mesh.nodes
    conn  = mesh.conn

    style = dict(lw=1.2, marker="none")
    if borehole_style:
        style.update(borehole_style)

    # ── sample each borehole ────────────────────────────────────────────────
    n = len(borehole_sites)
    logs = []
    for spec in borehole_sites:
        name  = spec.get("name", "?")
        x_m, y_m = _resolve_borehole_xy(spec, zone, northern)
        z_top = float(spec.get("z_top", 0.0))
        z_bot = float(spec.get("z_bot", 20000.0))
        dz    = float(spec.get("dz",    200.0))
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
        ax  = ax_arr[idx]
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

# --- (1b) Optionally estimate UTM_ORIGIN_E / UTM_ORIGIN_N from sites ------
if ESTIMATE_ORIGIN:
    _cal_sites = list(CALIBRATION_SITES)
    if SITE_DAT is not None and os.path.isfile(SITE_DAT):
        _sdat = fem.read_site_dat(SITE_DAT)
        # merge: site.dat entries take priority; CALIBRATION_SITES fills gaps
        _sdat_ids = {d["sitenum"] for d in _sdat}
        _extra    = [d for d in _cal_sites if d.get("site") not in _sdat_ids]
        _cal_sites = _sdat + _extra
        if OUT:
            print(f"  site.dat: loaded {len(_sdat)} site(s) from {SITE_DAT}")
    UTM_ORIGIN_E, UTM_ORIGIN_N = fem.estimate_utm_origin(
        _cal_sites, OBSERVE_FILE, UTM_ZONE, UTM_NORTHERN,
        site_dat=SITE_DAT, out=OUT
    )
    if UPDATE_CONFIG:
        UTM_ORIGIN_LAT, UTM_ORIGIN_LON = utl.utm_to_latlon_zn(
            UTM_ORIGIN_E, UTM_ORIGIN_N, UTM_ZONE, UTM_NORTHERN
        )
        UTM_ZONE, UTM_NORTHERN = _utm_zone_from_origin()
        if OUT:
            print(f"Config updated:  UTM_ORIGIN_LAT = {UTM_ORIGIN_LAT:.6f}")
            print(f"                 UTM_ORIGIN_LON = {UTM_ORIGIN_LON:.6f}")
            print(f"                 UTM_ZONE       = {UTM_ZONE}"
                  f"{'N' if UTM_NORTHERN else 'S'}")
            print()

# --- (2) Resolve slice positions to model-local metres ---------------------
slices_resolved = resolve_slices(PLOT_SLICES, UTM_ZONE, UTM_NORTHERN)
if OUT:
    print()

# --- (3) Optionally read site position(s) ----------------------------
# Primary: SITE_DAT (mt_make_sitelist.py format).  Fallback: observe.dat.
site_xys = []   # list of (label, x_m, y_m)
if SITE_DAT is not None:
    print(f"Reading site position(s) from site.dat: {SITE_DAT}")
    _rows = fem.read_site_dat(SITE_DAT, site_names=SITE_NAMES)
    for row in _rows:
        sx_m, sy_m = fem.utm_to_model(row["easting"], row["northing"],
                                       UTM_ORIGIN_E, UTM_ORIGIN_N)
        site_xys.append((row["name"], sx_m, sy_m))
        print(f"  {row['name']}: model-local x = {sx_m/1000:.3f} km,  y = {sy_m/1000:.3f} km")
        if DISPLAY_COORDS == "utm":
            print(f"           UTM         E = {sx_m + UTM_ORIGIN_E:.1f} m,  "
                  f"N = {sy_m + UTM_ORIGIN_N:.1f} m")
    if not site_xys:
        print("  (no matching sites found in site.dat)")
    print()
elif SITE_NUMBER is not None:
    _site_nums = (SITE_NUMBER if isinstance(SITE_NUMBER, (list, tuple))
                  else [SITE_NUMBER])
    print(f"Reading site position(s) from: {OBSERVE_FILE}")
    for _sn in _site_nums:
        sx_m, sy_m = fem.read_site_position(OBSERVE_FILE, _sn)
        site_xys.append((_sn, sx_m, sy_m))
        print(f"  site {_sn}: model-local x = {sx_m/1000:.3f} km,  y = {sy_m/1000:.3f} km")
        if DISPLAY_COORDS == "utm":
            print(f"           UTM         E = {sx_m + UTM_ORIGIN_E:.1f} m,  "
                  f"N = {sy_m + UTM_ORIGIN_N:.1f} m")
    print()

# --- (4) Plot slices -------------------------------------------------------
if fviz is None:
    sys.exit("femtic_viz not available — cannot plot.  Check your installation.")

print(f"Plotting model: {MODEL_FILE}")
plot_model_slices(
    model_file  = MODEL_FILE,
    mesh_file   = MESH_FILE,
    slices      = slices_resolved,
    cmap        = PLOT_CMAP,
    clim        = PLOT_CLIM,
    xlim        = PLOT_XLIM,
    ylim        = PLOT_YLIM,
    zlim        = PLOT_ZLIM,
    ocean_color = PLOT_OCEAN_COLOR,
    ocean_value = OCEAN_RHO,
    air_bgcolor = PLOT_AIR_BGCOLOR,
    site_xys    = site_xys,
    plot_file   = PLOT_FILE,
    dpi         = PLOT_DPI,
    equal_aspect = PLOT_EQUAL_ASPECT,
    out         = OUT,
)
print("Done.")

# --- (5) 3-D PyVista plot --------------------------------------------------
if PLOT3D:
    if fviz is None:
        print("  3-D plot skipped: femtic_viz not available.")
    else:
        print(f"Rendering 3-D model: {MODEL_FILE}")
        fviz.plot_model_3d(
            mesh_file      = MESH_FILE,
            block_file     = MODEL_FILE,
            scalar         = PLOT3D_SCALAR,
            clim           = PLOT3D_CLIM,
            cmap           = PLOT3D_CMAP,
            slice_x        = PLOT3D_SLICE_X,
            slice_y        = PLOT3D_SLICE_Y,
            slice_z        = PLOT3D_SLICE_Z,
            slice_planes   = PLOT3D_SLICE_PLANES,
            isovalues      = PLOT3D_ISOVALUES,
            iso_opacity    = PLOT3D_ISO_OPACITY,
            iso_cmap       = PLOT3D_CMAP,
            ocean_value    = OCEAN_RHO,
            air_region_index   = 0,
            ocean_region_index = 1,
            window_size    = PLOT3D_WINDOW_SIZE,
            plot_file      = PLOT3D_FILE,
            out            = OUT,
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
            model_file     = MODEL_FILE,
            mesh_file      = MESH_FILE,
            borehole_sites = BOREHOLE_SITES,
            zone           = UTM_ZONE,
            northern       = UTM_NORTHERN,
            clim           = BOREHOLE_XLIM,
            borehole_style = BOREHOLE_STYLE,
            shared         = BOREHOLE_SHARED,
            plot_file      = BOREHOLE_FILE,
            dpi            = PLOT_DPI,
            out            = OUT,
        )
        print("Borehole plot done.")
