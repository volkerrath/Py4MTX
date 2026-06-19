#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_plot_slice.py — 2-D slice panels for a FEMTIC resistivity model.

Produces one figure with horizontal map, N-S curtain, E-W curtain, and/or
arbitrary plane slices using exact tetrahedron-plane intersection
(``fviz.plot_model_slices``).

Sister scripts:
  ``femtic_mod_plot_bh.py``  — 1-D ρ(z) borehole log figures.
  ``femtic_mod_plot_3d.py``  — PyVista 3-D rendering and VTK/VTU export.

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
         point=([-71.5, -16.4, 5000.0], "latlon"),
         strike=45., dip=70.)

    # Two-point vertical profile (fence section) — strike auto-derived:
    dict(kind="profile",
         p1=([-71.50, -16.30], "latlon"),  # start endpoint [lon, lat]
         p2=([-70.90, -16.40], "latlon"),  # end   endpoint [lon, lat]
         z_top=0.0, z_bot=20000.0)         # depth extent [m, z-down]

UTM zone derivation
--------------------
    Zone number is computed from ``UTM_ORIGIN_LON`` (standard 6° bands,
    ignoring Norway / Svalbard exceptions).  Override with
    ``UTM_ZONE_OVERRIDE`` (positive integer) when needed.

Display coordinate system
--------------------------
    DISPLAY_COORDS = "model"  — axis ticks in model-local metres (default)
    DISPLAY_COORDS = "utm"    — axis ticks in absolute UTM metres
    DISPLAY_COORDS = "latlon" — axis ticks in decimal degrees

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
    2026-05-16  vrath / Claude Sonnet 4.6   Added borehole resistivity log
                step: BOREHOLE_* config block; CRS tagging on x/y positions.
    2026-05-23  vrath / Claude Sonnet 4.6   Moved pure geographic helpers to
                util.py; model-local helpers to femtic.py.  Script-level
                wrappers delegate to both.  SITE_NUMBER accepts list.
    2026-05-23  vrath / Claude Sonnet 4.6   SITE_DAT in mt_make_sitelist.py
                CSV format; read_site_dat() rewritten.
    2026-05-23  vrath / Claude Sonnet 4.6   PLOT_EQUAL_ASPECT config flag.
    2026-05-24  vrath / Claude Sonnet 4.6   Removed ENS_* block (moved to
                snippets.py).
    2026-05-24  vrath / Claude Sonnet 4.6   Moved read_site_position(),
                read_site_dat(), estimate_utm_origin(), _point_in_tet(),
                extract_borehole_log() to femtic.py.
    2026-05-25  vrath / Claude Sonnet 4.6   Added PLOT_SITES_MAPS /
                PLOT_SITES_SLICES, PROJECTION_DIST, SITE_MARKER_SLICES,
                DEPTH_KM, HORIZ_KM, PLOT_NROWS/NCOLS, PLOT_PANEL_HEIGHT/
                WIDTH/FIGSIZE.  UTM display in km.  Cmap deprecation fix.
    2026-05-26  Claude Sonnet 4.6 (Anthropic)
                Moved plot_model_slices / plot_borehole_logs into
                femtic_viz.py.  Added ALPHA_FILE / ALPHA_MODE /
                ALPHA_BLANK_THRESH config vars.
    2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
                Passed PLOT_XLIM/YLIM/ZLIM to slice panels.
    2026-05-31  vrath / Claude Sonnet 4.6 (Anthropic)
                Added invert_x per-panel key in PLOT_SLICES.
                Origin estimation now runs before UTM zone derivation.
                Hard-coded UTM_ORIGIN_* set to None (derived at runtime).
    2026-06-03  Claude Sonnet 4.6 (Anthropic)
                Split from femtic_mod_plot.py into femtic_mod_plot_slice.py
                (2-D slices + boreholes) and femtic_mod_plot_3d.py (PyVista
                3-D rendering).  Borehole: BOREHOLE_XLIM now in Ohm*m;
                z_top="surface" supported; lat/lon legend annotation and
                per-trace line-style keys in BOREHOLE_SITES.
                BOREHOLE_NPZ config var for NPZ data export.
                BOREHOLE_IN_SLICE config var: when True, borehole panels are
                embedded inside the slice figure as extra columns to the right
                via plot_model_slices(borehole_sites=...).
                _resolve_borehole_xy removed: CRS conversion for boreholes now
                handled natively in _sample_borehole_logs via utm_zone /
                utm_origin_e/n parameters (no longer needed in script).
                Standalone plot_borehole_logs call updated to pass utm_zone /
                utm_northern / utm_origin_e / utm_origin_n directly.
    2026-06-04  vrath / Claude Sonnet 4.6 (Anthropic)
                Borehole config block (BOREHOLE_*, PLOT_BOREHOLE) and step (6)
                moved to new sister script femtic_mod_plot_bh.py.
                plot_model_slices call: borehole_* parameters removed.
    2026-06-19  Claude Sonnet 4.6 (Anthropic)
                Added kind="profile" to PLOT_SLICES: a vertical fence section
                defined by two endpoint positions (p1, p2) each accepting
                model-local / UTM / latlon CRS tags; strike is derived from
                the p1→p2 azimuth; dip is fixed at 90.  New helper
                resolve_pos_two_point_profile() added to femtic.py;
                resolve_slice_positions() extended with the "profile" branch.

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
WORK_DIR = r"/home/vrath/Py4MTX/py4mt/data/rto/misti/PrepRun8/"

#: Resistivity block to display (any iteration).
MODEL_FILE = WORK_DIR + "resistivity_block_iter15.dat"

#: Mesh file — always required for plotting.
MESH_FILE = WORK_DIR + "mesh.dat"

#: observe.dat — used as fallback site-overlay source when SITE_DAT is None
#: and SITE_NUMBER is not None.
OBSERVE_FILE = WORK_DIR + "observe.dat"

#: Site list produced by mt_make_sitelist.py (WHAT_FOR="femtic").
#: Format (comma-separated, no header):
#:   name, lat, lon, elev, sitenum, easting, northing
#: Easting/northing are UTM metres; model-local x/y is derived via
#: fem.utm_to_model using the mesh-centre origin.
#: Set to None to fall back to the observe.dat / SITE_NUMBER path.
SITE_DAT = WORK_DIR + "site.dat"   # set to None to disable

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

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
UTM_ORIGIN_LAT = None   # decimal degrees, positive = North
UTM_ORIGIN_LON = None   # decimal degrees, positive = East

#: UTM coordinates of the mesh origin in metres.
UTM_ORIGIN_E = None   # easting  [m]
UTM_ORIGIN_N = None   # northing [m]

#: Override the auto-derived UTM zone number.  None = auto from origin lat/lon.
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

#: Show site markers on horizontal map panels.
PLOT_SITES_MAPS = True
#: Show site markers on vertical curtain (ns/ew) and plane panels.
PLOT_SITES_SLICES = True

#: Maximum distance [m] from a vertical slice plane for a site to be plotted
#: on that panel.  None = plot all sites on every panel regardless of distance.
PROJECTION_DIST = 1000.   # e.g. 5000.0  (5 km)

#: Marker style for map panels (inverted triangle = MT convention).
SITE_MARKER = dict(marker="v", color="black", ms=4, zorder=10,
                   label=None)   # label filled in automatically

#: Marker style for curtain and plane panels.
SITE_MARKER_SLICES = dict(marker="v", color="black", ms=4, zorder=10,
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

## Ubinas
# MAP_MARKERS = [
#     dict(latlon=[-16.3169, -70.9673], marker="x", color="red", ms=8,
#          name="test point, 4457m"),
#     dict(latlon=[-16.3450, -70.8972], marker="*", color="red", ms=8,
#          name="ubinas crater"),
#     dict(latlon=[-16.363436, -70.868025], marker="+", color="blue", ms=8,
#          name="mesh origin, borehole1"),
#     dict(latlon=[-16.351, -70.9016], marker="^", color="magenta", ms=6,
#          name="max elev"),
# ]

## Misti
MAP_MARKERS = [
    dict(latlon=[-16.2991, -71.4056], marker="*", color="red", ms=12,
         name="misti crater"),
    dict(latlon=[-16.248170, -71.506431], marker="+", color="blue", ms=12,
         name="mesh origin, borehole1"),
    # dict(latlon=[-16.351, -70.9016], marker="^", color="magenta", ms=6,
    #      name="max elev"),
]

# ---------------------------------------------------------------------------
# Plotting — requires femtic_viz and Matplotlib
# ---------------------------------------------------------------------------
#: Output file path — None → interactive show().
PLOT_FILE = WORK_DIR + "resistivity_block_iter15.pdf"
#: Figure DPI for saved file.
PLOT_DPI = 600
#: Matplotlib colormap name.
PLOT_CMAP = "turbo_r"
#: Colour limits [log10(ρ_min), log10(ρ_max)] — None = auto.
PLOT_CLIM = [0.0, 3.0]      # log10(Ω·m)
#: Flat colour for ocean / lake cells.  None → use colormap.
PLOT_OCEAN_COLOR = "lightgrey"
#: Flat colour for air polygons on slice panels.  None → blank gaps.
PLOT_AIR_COLOR = "whitesmoke"
#: Axes facecolor for air / background.  None = figure default.
PLOT_AIR_BGCOLOR = None

# ---------------------------------------------------------------------------
# Alpha / blanking by second block file  (optional)
# ---------------------------------------------------------------------------
#: Path to a second resistivity_block_iterX.dat with the SAME mesh and
#: region structure as MODEL_FILE.  Its region_rho values are interpreted
#: as log10 weights.  Cells with weight < ALPHA_BLANK_THRESH are suppressed.
#: Set to None to disable.
ALPHA_FILE = None   # e.g. WORK_DIR + "sensitivity_log10.dat"

#: How alpha weights drive visibility:
#:   "fade"  — proportional transparency between ALPHA_BLANK_THRESH (=0) and 0
#:   "blank" — hard cutoff: below ALPHA_BLANK_THRESH the polygon is omitted
ALPHA_MODE = "fade"

#: Log10 threshold at or below which polygons are blanked / fully faded.
ALPHA_BLANK_THRESH = 0.0

# ---------------------------------------------------------------------------
# Slice specification
# ---------------------------------------------------------------------------
#: List of slice dicts — one per panel (left to right).
#:
#: Slices use exact tetrahedron-plane intersection (no selection slab).
#: Each dict must contain:
#:   kind   : "map"     — horizontal slice at z = z0
#:            "ns"      — N-S vertical section at x = x0   (y vs depth)
#:            "ew"      — E-W vertical section at y = y0   (x vs depth)
#:            "plane"   — arbitrary plane by strike / dip / point
#:            "profile" — vertical fence section defined by two endpoints;
#:                        strike is computed automatically from p1→p2 azimuth
#:   z0     : (map     only)  depth in metres
#:   x0     : (ns      only)  easting — plain float = model-local metres;
#:            or (value, "utm") / (value, "latlon") for CRS tagging
#:   y0     : (ew      only)  northing — plain float / CRS tuple
#:   point  : (plane   only)  [x, y, z] any point on the plane
#:            or ([lon, lat, z], "latlon") / ([E, N, z], "utm")
#:   p1, p2 : (profile only)  endpoint position specs; each accepts:
#:              [x, y]                  model-local metres (bare list)
#:              ([x, y], "model")       explicit model-local
#:              ([E, N], "utm")         UTM easting / northing [m]
#:              ([lon, lat], "latlon")  decimal degrees
#:            A 3-element [x, y, z] variant is also accepted (z ignored).
#:   z_top  : (profile only)  shallowest depth of fence panel [m, z-down]
#:            default 0.0
#:   z_bot  : (profile only)  deepest depth of fence panel [m, z-down]
#:            default 20 000 m
#:   strike : (plane   only)  clockwise from North, degrees
#:   dip    : (plane   only)  downward inclination from horizontal, degrees
#:   xlim   : [xmin, xmax] — easting or along-strike axis limit
#:   ylim   : [ymin, ymax] — northing or down-dip axis limit
#:   zlim   : [zmin, zmax] — depth axis limit (ns/ew/profile panels)
#:   invert_x : True → flip the horizontal axis left-to-right after rendering
#:   title  : optional string override
#:
#: Per-panel xlim/ylim/zlim override PLOT_XLIM/PLOT_YLIM/PLOT_ZLIM.
PLOT_SLICES = [
    dict(kind="ns",  x0=(-71.536322, "latlon")),
    dict(kind="ew",  y0=(-16.196900, "latlon")),
    dict(kind="map", z0=-4000.0),
    dict(kind="map", z0= 10000.0),
    # Two-point vertical profile example (uncomment to use):
    # dict(kind="profile",
    #      p1=([-71.536, -16.197], "latlon"),
    #      p2=([-71.406, -16.299], "latlon"),
    #      z_top=0.0, z_bot=25000.0,
    #      title="NW-SE profile"),
]

#: Global axis limits in model-local metres.  None → auto.
PLOT_XLIM = [-25000., 25000.]   # [xmin, xmax] metres — easting
PLOT_YLIM = [-25000., 25000.]   # [ymin, ymax] metres — northing
PLOT_ZLIM = [-10000.,  25000.]   # [zmin, zmax] metres — depth (z positive-down)

#: Equal aspect ratio for map and curtain panels.
PLOT_EQUAL_ASPECT = True

#: Display depth axis in km instead of metres on curtain / plane panels.
DEPTH_KM = True

#: Display horizontal axes in km when DISPLAY_COORDS is "model".
HORIZ_KM = True

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
#: Number of subplot rows and columns.  None → auto (1 row, len(PLOT_SLICES) cols).
PLOT_NROWS = 2 #None
PLOT_NCOLS = 2 #None

#: Panel height in cm.
PLOT_PANEL_HEIGHT = 16.0   # cm

#: Fixed panel width in cm.  None → auto from aspect ratio.
PLOT_PANEL_WIDTH = None

#: Full figure size [width, height] in cm.  Overrides auto-sizing when set.
PLOT_FIGSIZE = None   # e.g. [40., 25.]

# ---------------------------------------------------------------------------
# Mesh-centre estimation from site.dat UTM coordinates  (optional)
# ---------------------------------------------------------------------------
#: Method used to estimate UTM_ORIGIN_E / UTM_ORIGIN_N from SITE_DAT:
#:   None      — use the hard-coded values above
#:   "box"     — midpoint of the UTM bounding box of all sites in SITE_DAT
#:   "average" — arithmetic mean of all site UTM coordinates in SITE_DAT
ORIGIN_METHOD = "box"   # None | "box" | "average"


# ===========================================================================
# Main
# ===========================================================================

# --- (1) Optionally estimate UTM origin from site.dat ----------------------
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
            _lats = np.array([d["lat"] for d in _sdat])
            _lons = np.array([d["lon"] for d in _sdat])
            UTM_ZONE, UTM_NORTHERN = utl.utm_zone_from_latlon(
                float(_lats.mean()), float(_lons.mean()),
                override=UTM_ZONE_OVERRIDE)
            UTM_ORIGIN_LAT, UTM_ORIGIN_LON = utl.utm_to_latlon_zn(
                UTM_ORIGIN_E, UTM_ORIGIN_N, UTM_ZONE, UTM_NORTHERN)
            if OUT:
                print(f"Origin estimated ({ORIGIN_METHOD}, {len(_sdat)} sites):")
                print(f"  UTM_ORIGIN_E   = {UTM_ORIGIN_E:.1f} m")
                print(f"  UTM_ORIGIN_N   = {UTM_ORIGIN_N:.1f} m")
                print(f"  UTM_ORIGIN_LAT = {UTM_ORIGIN_LAT:.6f}°")
                print(f"  UTM_ORIGIN_LON = {UTM_ORIGIN_LON:.6f}°")
                print(f"  UTM_ZONE       = {UTM_ZONE}{'N' if UTM_NORTHERN else 'S'}")
                print()

# --- (2) Derive UTM zone from finalised mesh-origin coordinates ------------
UTM_ZONE, UTM_NORTHERN = utl.utm_zone_from_latlon(
    UTM_ORIGIN_LAT, UTM_ORIGIN_LON, override=UTM_ZONE_OVERRIDE)
hemi = "N" if UTM_NORTHERN else "S"
print(f"UTM zone: {UTM_ZONE}{hemi}  "
      f"(origin lat={UTM_ORIGIN_LAT:.4f}°, lon={UTM_ORIGIN_LON:.4f}°)")
print()

# --- (3) Resolve slice positions to model-local metres ---------------------
slices_resolved = fem.resolve_slice_positions(
    PLOT_SLICES, UTM_ZONE, UTM_NORTHERN,
    UTM_ORIGIN_E, UTM_ORIGIN_N,
    UTM_ORIGIN_LAT, UTM_ORIGIN_LON,
    verbose=OUT,
)
if OUT:
    print()

# --- (4) Optionally read site position(s) ----------------------------------
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

# --- (5) Plot 2-D slice panels --------------------------------------------
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
    air_color=PLOT_AIR_COLOR,
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
print("Slice plot done.")

