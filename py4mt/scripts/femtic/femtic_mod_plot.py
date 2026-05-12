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
WORK_DIR = r"/home/vrath/Py4MTX/work/"

#: Resistivity block to display (any iteration).
MODEL_FILE = WORK_DIR + "resistivity_block_iter0.dat"

#: Mesh file — always required for plotting.
MESH_FILE  = WORK_DIR + "mesh.dat"

#: observe.dat — required only when SITE_NUMBER is not None.
OBSERVE_FILE = WORK_DIR + "observe.dat"

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
DISPLAY_COORDS = "model"

# ---------------------------------------------------------------------------
# Site overlay from observe.dat
# ---------------------------------------------------------------------------
#: Site number to extract from observe.dat (integer, 1-based).
#: Set to None to skip site overlay.
SITE_NUMBER = 5

#: Marker style for the site on map panels; dashed line on curtain panels.
SITE_MARKER = dict(marker="v", color="black", ms=8, zorder=10,
                   label=None)   # label filled in automatically

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
#: Output file path — None → interactive show().
PLOT_FILE = WORK_DIR + "resistivity_block_iter0.pdf"

#: Figure DPI for saved file.
PLOT_DPI = 600

#: Matplotlib colormap name.
PLOT_CMAP = "turbo_r"

#: Colour limits [log10(ρ_min), log10(ρ_max)] — None = auto.
PLOT_CLIM = [0.0, 4.0]

#: Flat colour for ocean / lake cells.  None → use colormap.
PLOT_OCEAN_COLOR = "lightgrey"

#: Axes facecolor for air / background.  None = figure default.
PLOT_AIR_BGCOLOR = None

#: Slice specification.
#:
#: Each dict must contain:
#:   kind   : "map"   — horizontal slice at z = z0
#:            "ns"    — N-S vertical section at x = x0   (y vs depth)
#:            "ew"    — E-W vertical section at y = y0   (x vs depth)
#:            "plane" — arbitrary plane by strike / dip / point
#:
#: Horizontal position keys (x0, y0, point[:2]) accept:
#:   plain float          → model-local metres ("model" crs)
#:   (value, "model")     → model-local metres
#:   (value, "utm")       → UTM metres  (auto zone from UTM_ORIGIN_LAT/LON)
#:   (value, "latlon")    → decimal degrees  (lon for x0, lat for y0)
#:
#: For "plane":
#:   point = [x, y, z]            → all model-local metres
#:   point = ([lon, lat, z], "latlon")  → lon/lat in degrees, z in model-local m
#:   point = ([E,   N,   z], "utm")     → E/N in UTM m, z in model-local m
#:
#: z0 is always model-local metres.
#: Per-panel xlim/ylim/zlim (model-local m) override the globals below.
PLOT_SLICES = [
    # Plain float — model-local metres (backward-compatible):
    dict(kind="map",  z0=5000.0),
    dict(kind="map",  z0=15000.0),
    # UTM easting for the NS curtain:
    dict(kind="ns",   x0=(229047.0, "utm")),
    # Geographic latitude for the EW curtain:
    dict(kind="ew",   y0=(-16.409, "latlon")),
]

#: Global axis limits in model-local metres; None = auto.
PLOT_XLIM = [-20000., 20000.]
PLOT_YLIM = [-20000., 20000.]
PLOT_ZLIM = [  -6000., 15000.]

# ---------------------------------------------------------------------------
# Mesh-centre estimation from known site coordinates  (optional)
# ---------------------------------------------------------------------------
#: Set ESTIMATE_ORIGIN = True to compute UTM_ORIGIN_E / UTM_ORIGIN_N from
#: a set of calibration sites whose model-local positions (from observe.dat)
#: and geographic coordinates are both known.  The result overwrites the
#: hard-coded UTM_ORIGIN_E / UTM_ORIGIN_N above for this run and is printed
#: so you can copy the values back into the script.
#:
#: Each entry in CALIBRATION_SITES is a dict with:
#:   "site"    : int    site number (matched against observe.dat)
#:   "crs"     : str    "latlon" or "utm"
#:   "coords"  : for "latlon" → [lon_deg, lat_deg]
#:               for "utm"    → [E_m, N_m]
#:
#: At least 1 site is required; 3+ is recommended to detect gross errors.
#: The fit is a pure translation (no rotation/scale) in UTM space, solved
#: by least squares.  Residuals per site are printed when OUT = True.
ESTIMATE_ORIGIN = False

CALIBRATION_SITES = [
    # dict(site=1,  crs="latlon", coords=[-71.500, -16.380]),
    # dict(site=10, crs="latlon", coords=[-71.620, -16.450]),
    # dict(site=25, crs="utm",    coords=[224500., 8179300.]),
]


# ===========================================================================
# Coordinate conversion helpers
# ===========================================================================

def _utm_zone_from_origin() -> tuple[int, bool]:
    """Derive UTM zone number and hemisphere flag from mesh-origin lat/lon.

    Standard 6° band rule; ignores Norway/Svalbard special zones.
    Override with UTM_ZONE_OVERRIDE when non-standard zones are needed.

    Returns
    -------
    zone     : int   UTM zone number (1–60)
    northern : bool  True if origin is in the Northern hemisphere
    """
    if UTM_ZONE_OVERRIDE is not None:
        zone = int(UTM_ZONE_OVERRIDE)
        if not 1 <= zone <= 60:
            raise ValueError(f"UTM_ZONE_OVERRIDE={zone} out of range 1–60.")
        return zone, UTM_ORIGIN_LAT >= 0.0
    zone = min(int((UTM_ORIGIN_LON + 180.0) / 6.0) + 1, 60)
    return zone, UTM_ORIGIN_LAT >= 0.0


def _latlon_to_utm(lat_deg: float, lon_deg: float,
                   zone: int, northern: bool) -> tuple[float, float]:
    """Convert WGS-84 geographic coordinates to UTM easting/northing (m).

    Uses pyproj when available (primary path); falls back to the standard
    Transverse Mercator / Helmert series (accurate to < 1 mm within a zone)
    when pyproj is not installed.

    Parameters
    ----------
    lat_deg, lon_deg : decimal degrees (positive = N / E)
    zone             : UTM zone number 1–60
    northern         : True → Northern hemisphere (false northing = 0)

    Returns
    -------
    E_m, N_m : UTM easting and northing in metres
    """
    # ------------------------------------------------------------------
    # Primary path: pyproj  (accurate, handles all edge cases)
    # ------------------------------------------------------------------
    if _HAVE_PYPROJ:
        hemi = "north" if northern else "south"
        crs  = f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m"
        tr   = _Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        E_m, N_m = tr.transform(lon_deg, lat_deg)
        return float(E_m), float(N_m)

    # ------------------------------------------------------------------
    # Fallback: Helmert / Bowring series (no external dependency)
    # Accurate to < 1 mm anywhere within a single UTM zone.
    # ------------------------------------------------------------------
    a   = 6_378_137.0             # WGS-84 semi-major axis [m]
    f   = 1.0 / 298.257_223_563
    k0  = 0.9996                  # central-meridian scale factor
    E0  = 500_000.0               # false easting [m]
    N0  = 0.0 if northern else 10_000_000.0

    e2       = 2.0 * f - f * f
    lon0_deg = (zone - 1) * 6 - 180 + 3   # central meridian [°]

    lat  = math.radians(lat_deg)
    lon  = math.radians(lon_deg)
    lon0 = math.radians(lon0_deg)

    N_r = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    T   = math.tan(lat) ** 2
    C   = e2 / (1.0 - e2) * math.cos(lat) ** 2
    A2  = math.cos(lat) * (lon - lon0)

    e4, e6 = e2 ** 2, e2 ** 3
    M = a * (
        (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * math.sin(2.0 * lat)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * math.sin(4.0 * lat)
        - (35.0 * e6 / 3072.0) * math.sin(6.0 * lat)
    )

    E_m = E0 + k0 * N_r * (
        A2
        + (1.0 - T + C) * A2 ** 3 / 6.0
        + (5.0 - 18.0 * T + T ** 2 + 72.0 * C - 58.0 * e2 / (1.0 - e2)) * A2 ** 5 / 120.0
    )
    N_m = N0 + k0 * (
        M
        + N_r * math.tan(lat) * (
            A2 ** 2 / 2.0
            + (5.0 - T + 9.0 * C + 4.0 * C ** 2) * A2 ** 4 / 24.0
            + (61.0 - 58.0 * T + T ** 2 + 600.0 * C - 330.0 * e2 / (1.0 - e2)) * A2 ** 6 / 720.0
        )
    )
    return E_m, N_m


def _utm_to_model(E_m: float, N_m: float) -> tuple[float, float]:
    """UTM metres → model-local metres (subtract mesh-centre UTM origin)."""
    return E_m - UTM_ORIGIN_E, N_m - UTM_ORIGIN_N


def _latlon_to_model(lat_deg: float, lon_deg: float,
                     zone: int, northern: bool) -> tuple[float, float]:
    """Geographic degrees → model-local metres via UTM."""
    E_m, N_m = _latlon_to_utm(lat_deg, lon_deg, zone, northern)
    return _utm_to_model(E_m, N_m)


def _parse_pos(raw) -> tuple:
    """Parse a position spec into (value, crs_str).

    Accepts:
      scalar (int / float)   → (float, "model")
      (value, "crs")         → (value, crs)
    """
    if isinstance(raw, (int, float)):
        return float(raw), "model"
    try:
        val, crs = raw
        return val, str(crs)
    except (TypeError, ValueError):
        raise ValueError(
            f"Position spec {raw!r} must be a scalar or (value, 'crs') tuple."
        )


def _resolve_x0(raw, zone: int, northern: bool) -> float:
    """Resolve x0 (NS curtain, easting direction) to model-local metres.

    crs "latlon": *raw* is a longitude in decimal degrees.
    crs "utm":    *raw* is a UTM easting in metres.
    crs "model":  *raw* is already in model-local metres.
    """
    val, crs = _parse_pos(raw)
    if crs == "model":
        return float(val)
    if crs == "utm":
        x_m, _ = _utm_to_model(float(val), UTM_ORIGIN_N)
        return x_m
    if crs == "latlon":
        # val = longitude; use origin latitude for the northing
        x_m, _ = _latlon_to_model(UTM_ORIGIN_LAT, float(val), zone, northern)
        return x_m
    raise ValueError(f"Unknown crs={crs!r} for x0.  Choose 'model', 'utm', or 'latlon'.")


def _resolve_y0(raw, zone: int, northern: bool) -> float:
    """Resolve y0 (EW curtain, northing direction) to model-local metres.

    crs "latlon": *raw* is a latitude in decimal degrees.
    crs "utm":    *raw* is a UTM northing in metres.
    crs "model":  *raw* is already in model-local metres.
    """
    val, crs = _parse_pos(raw)
    if crs == "model":
        return float(val)
    if crs == "utm":
        _, y_m = _utm_to_model(UTM_ORIGIN_E, float(val))
        return y_m
    if crs == "latlon":
        # val = latitude; use origin longitude for the easting
        _, y_m = _latlon_to_model(float(val), UTM_ORIGIN_LON, zone, northern)
        return y_m
    raise ValueError(f"Unknown crs={crs!r} for y0.  Choose 'model', 'utm', or 'latlon'.")


def _resolve_point(raw, zone: int, northern: bool) -> list[float]:
    """Resolve a plane-slice point [x, y, z] to model-local metres.

    Accepts:
      [x, y, z]                   plain list → model-local metres
      ([x, y, z], "model")        model-local metres
      ([E, N, z], "utm")          UTM metres for E/N; z model-local
      ([lon, lat, z], "latlon")   degrees for lon/lat; z model-local
    """
    crs = "model"
    pt  = raw
    if (isinstance(raw, (list, tuple)) and len(raw) == 2
            and isinstance(raw[-1], str)):
        pt, crs = raw[0], raw[1]

    pt = list(pt)
    if len(pt) != 3:
        raise ValueError(f"plane 'point' must have 3 elements, got {len(pt)}.")
    z_m = float(pt[2])   # depth always model-local

    if crs == "model":
        return [float(pt[0]), float(pt[1]), z_m]
    if crs == "utm":
        x_m, y_m = _utm_to_model(float(pt[0]), float(pt[1]))
        return [x_m, y_m, z_m]
    if crs == "latlon":
        # pt = [lon_deg, lat_deg, z_m]
        x_m, y_m = _latlon_to_model(float(pt[1]), float(pt[0]), zone, northern)
        return [x_m, y_m, z_m]
    raise ValueError(f"Unknown crs={crs!r} for point.  Choose 'model', 'utm', or 'latlon'.")


def resolve_slices(slices: list, zone: int, northern: bool) -> list:
    """Return a new list of slice specs with all positions in model-local metres.

    Only horizontal position keys (x0, y0, point) are converted; z0 and all
    other keys (xlim, ylim, zlim, title, strike, dip, kind) pass through
    unchanged.  The original *slices* list is not mutated.

    Parameters
    ----------
    slices   : list of slice-spec dicts (from PLOT_SLICES)
    zone     : UTM zone number
    northern : hemisphere flag

    Returns
    -------
    list of slice-spec dicts with model-local float values for x0, y0, point
    """
    resolved = []
    for i, spec in enumerate(slices):
        s    = dict(spec)
        kind = s.get("kind", "map")

        if kind == "ns" and "x0" in s:
            raw      = s["x0"]
            s["x0"]  = _resolve_x0(raw, zone, northern)
            _, crs   = _parse_pos(raw)
            if OUT and crs != "model":
                print(f"  slice[{i}] ns  x0: {raw!r}  →  {s['x0']:.1f} m (model-local)")

        if kind == "ew" and "y0" in s:
            raw      = s["y0"]
            s["y0"]  = _resolve_y0(raw, zone, northern)
            _, crs   = _parse_pos(raw)
            if OUT and crs != "model":
                print(f"  slice[{i}] ew  y0: {raw!r}  →  {s['y0']:.1f} m (model-local)")

        if kind == "plane" and "point" in s:
            raw         = s["point"]
            s["point"]  = _resolve_point(raw, zone, northern)
            crs = (raw[1] if (isinstance(raw, (list, tuple))
                              and len(raw) == 2
                              and isinstance(raw[-1], str))
                   else "model")
            if OUT and crs != "model":
                print(f"  slice[{i}] plane point: {raw!r}  →  {s['point']} (model-local)")

        resolved.append(s)
    return resolved


# ===========================================================================
# Helper: read one site's model-local position from observe.dat
# ===========================================================================

def read_site_position(observe_file: str, site_number: int) -> tuple[float, float]:
    """Return (x_m, y_m) model-local position for *site_number* from observe.dat.

    The file format alternates between site-header lines::

        <n>  <n>  <x_km>  <y_km>

    (where the first two tokens are the site number repeated) and data blocks.
    This function scans the file linearly, identifies site-header lines by the
    pattern ``int int float float``, and returns the (x, y) pair that matches
    *site_number*.

    Parameters
    ----------
    observe_file : path to observe.dat
    site_number  : 1-based site index to find

    Returns
    -------
    x_m, y_m : model-local coordinates in **metres** (converted from km)

    Raises
    ------
    FileNotFoundError
        If *observe_file* does not exist.
    ValueError
        If *site_number* is not found in the file.
    """
    if not os.path.isfile(observe_file):
        raise FileNotFoundError(f"observe.dat not found: {observe_file}")

    with open(observe_file) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                n1   = int(parts[0])
                int(parts[1])
                x_km = float(parts[2])
                y_km = float(parts[3])
            except ValueError:
                continue
            if n1 == site_number:
                return x_km * 1000.0, y_km * 1000.0

    raise ValueError(
        f"Site {site_number} not found in {observe_file}.  Check SITE_NUMBER."
    )


# ===========================================================================
# Helper: display-coordinate offset and axis label suffix
# ===========================================================================

def _display_offset() -> tuple[float, float]:
    """Return (dE, dN) to add to model-local metres for display axis ticks."""
    if DISPLAY_COORDS == "utm":
        return UTM_ORIGIN_E, UTM_ORIGIN_N
    return 0.0, 0.0


def _display_suffix() -> str:
    """Return axis label suffix reflecting the display coordinate system."""
    return " [UTM m]" if DISPLAY_COORDS == "utm" else " [m]"


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
    site_xy: tuple | None = None,
    plot_file=None,
    dpi: int = 200,
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
    site_xy     : (x_m, y_m) model-local position of the selected site;
                  None = no marker
    plot_file   : save path; None = interactive show()
    dpi         : saved-figure DPI
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
        normals = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        inv     = [True, True, False]
        pt      = np.zeros(3); pt[axis] = val
        n       = normals[axis]
        ref     = np.array([0, 0, 1]) if axis != 2 else np.array([1, 0, 0])
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
                       np.array([0, 0, 1]) if abs(normal[2]) < 0.9
                       else np.array([1, 0, 0]))
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
        ref = np.array([0, 1, 0]) if abs(normal[1]) < 0.9 else np.array([1, 0, 0])
        u   = np.cross(normal, ref); u /= np.linalg.norm(u)
        v   = np.cross(u, normal);   v /= np.linalg.norm(v)
        return u, v

    # ── display offset ───────────────────────────────────────────────────────
    dE, dN = _display_offset()
    sfx    = _display_suffix()

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
    cmap_obj = (mcm.colormaps[cmap].copy()
                if hasattr(mcm, "colormaps")
                else mcm.get_cmap(cmap).copy())
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
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5 * n_panels, 5),
                             squeeze=False)
    axes = axes[0]
    if air_bgcolor is not None:
        for ax in axes:
            ax.set_facecolor(air_bgcolor)

    sx_m, sy_m = (site_xy if site_xy is not None else (None, None))

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
            normal = np.array([0, 0, 1])
            point  = np.array([0, 0, z0])
            u_ax   = np.array([1, 0, 0])
            v_ax   = np.array([0, 1, 0])
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
            if title is None: title = f"Map  z = {z0/1000:.1f} km"
            if sx_m is not None:
                mk = dict(SITE_MARKER)
                mk.setdefault("label", f"Site {SITE_NUMBER}")
                ax.plot(sx_m + dE, sy_m + dN, linestyle="none", **mk)

        # ── NS curtain (x = const) ────────────────────────────────────────
        elif kind == "ns":
            x0 = float(spec.get("x0", 0.0))
            if out: print(f"    NS slice x={x0:.0f} m …")
            normal, point, u_ax, v_ax, inv = _axis_slice_params(0, x0)
            polys, vals = _slice_geometry(nodes, conn, rho_plot,
                                          normal, point, u_ax, v_ax)
            # u_ax points in y (northing) direction
            polys_d = [[(py + dN, pz) for py, pz in poly] for poly in polys]
            mappable = _plot_slice_panel(ax, polys_d, vals,
                cmap_obj=cmap_obj, norm=norm,
                ocean_color=ocean_color, ocean_value=ocean_value, invert_v=inv)
            ax.set_xlabel(f"y (northing){sfx}")
            ax.set_ylabel("depth (m)")
            if _ylim is not None: ax.set_xlim([v + dN for v in _ylim])
            if _zlim is not None: ax.set_ylim([_zlim[1], _zlim[0]])
            if title is None: title = f"N-S  x = {x0/1000:.1f} km"
            if sx_m is not None:
                ax.axvline(sy_m + dN, color=SITE_MARKER["color"],
                           lw=1.2, ls="--", zorder=9,
                           label=f"Site {SITE_NUMBER} (y)")

        # ── EW curtain (y = const) ────────────────────────────────────────
        elif kind == "ew":
            y0 = float(spec.get("y0", 0.0))
            if out: print(f"    EW slice y={y0:.0f} m …")
            normal, point, u_ax, v_ax, inv = _axis_slice_params(1, y0)
            polys, vals = _slice_geometry(nodes, conn, rho_plot,
                                          normal, point, u_ax, v_ax)
            # u_ax points in x (easting) direction
            polys_d = [[(px + dE, pz) for px, pz in poly] for poly in polys]
            mappable = _plot_slice_panel(ax, polys_d, vals,
                cmap_obj=cmap_obj, norm=norm,
                ocean_color=ocean_color, ocean_value=ocean_value, invert_v=inv)
            ax.set_xlabel(f"x (easting){sfx}")
            ax.set_ylabel("depth (m)")
            if _xlim is not None: ax.set_xlim([v + dE for v in _xlim])
            if _zlim is not None: ax.set_ylim([_zlim[1], _zlim[0]])
            if title is None: title = f"E-W  y = {y0/1000:.1f} km"
            if sx_m is not None:
                ax.axvline(sx_m + dE, color=SITE_MARKER["color"],
                           lw=1.2, ls="--", zorder=9,
                           label=f"Site {SITE_NUMBER} (x)")

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
        if sx_m is not None and kind in ("map", "ns", "ew"):
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Model: {os.path.basename(model_file)}", fontsize=10)
    fig.tight_layout()

    if plot_file is not None:
        fig.savefig(plot_file, dpi=dpi, bbox_inches="tight")
        if out:
            print(f"  plot: saved → {plot_file}")
    else:
        plt.show()


# ===========================================================================
# Mesh-centre estimation helper
# ===========================================================================

def estimate_utm_origin(calibration_sites: list,
                        observe_file: str,
                        zone: int,
                        northern: bool,
                        *,
                        out: bool = True) -> tuple[float, float]:
    """Estimate UTM coordinates of the mesh centre from calibration sites.

    Each calibration site provides a pair of observations:

    - its model-local position (x_m, y_m), read from *observe_file*,
    - its known geographic position in lat/lon or UTM.

    The mesh centre satisfies:

        E_site = UTM_ORIGIN_E + x_m_site
        N_site = UTM_ORIGIN_N + y_m_site

    Rearranging:

        UTM_ORIGIN_E = E_site − x_m_site
        UTM_ORIGIN_N = N_site − y_m_site

    This is a pure translation — no rotation or scale is assumed (model-local
    axes are aligned with UTM east/north by FEMTIC convention).  With N ≥ 1
    sites the system is solved as a least-squares mean; residuals per site
    expose gross coordinate errors.

    Parameters
    ----------
    calibration_sites : list of dicts with keys
                        "site"   : int   site number (matched in observe.dat)
                        "crs"    : str   "latlon" or "utm"
                        "coords" : list  [lon_deg, lat_deg] or [E_m, N_m]
    observe_file      : path to observe.dat
    zone              : UTM zone number (used for lat/lon → UTM conversion)
    northern          : hemisphere flag
    out               : print per-site residuals and result when True

    Returns
    -------
    origin_E, origin_N : estimated UTM coordinates of the mesh centre [m]

    Raises
    ------
    ValueError  if calibration_sites is empty or any site is not found.
    """
    if not calibration_sites:
        raise ValueError("CALIBRATION_SITES is empty — nothing to estimate.")

    offsets_E = []
    offsets_N = []

    if out:
        print("Estimating mesh-centre UTM origin from calibration sites:")
        print(f"  {'site':>5}  {'x_model':>10}  {'y_model':>10}  "
              f"{'E_utm':>12}  {'N_utm':>14}  {'dE':>8}  {'dN':>8}")
        print("  " + "-" * 77)

    for entry in calibration_sites:
        site_num = int(entry["site"])
        crs      = str(entry["crs"])
        coords   = list(entry["coords"])

        # model-local position [m] from observe.dat
        x_m, y_m = read_site_position(observe_file, site_num)

        # geographic → UTM [m]
        if crs == "latlon":
            lon_deg, lat_deg = coords
            E_site, N_site = _latlon_to_utm(lat_deg, lon_deg, zone, northern)
        elif crs == "utm":
            E_site, N_site = float(coords[0]), float(coords[1])
        else:
            raise ValueError(
                f"Calibration site {site_num}: unknown crs={crs!r}. "
                f"Use 'latlon' or 'utm'."
            )

        # implied origin from this site
        oE = E_site - x_m
        oN = N_site - y_m
        offsets_E.append(oE)
        offsets_N.append(oN)

        if out:
            print(f"  {site_num:>5}  {x_m/1000:>10.3f}  {y_m/1000:>10.3f}  "
                  f"{E_site:>12.1f}  {N_site:>14.1f}  "
                  f"{oE:>8.1f}  {oN:>8.1f}")

    # least-squares estimate = mean of implied origins
    origin_E = float(np.mean(offsets_E))
    origin_N = float(np.mean(offsets_N))

    if out and len(calibration_sites) > 1:
        # per-site residuals relative to the LS mean
        print()
        print(f"  {'site':>5}  {'res_E (m)':>10}  {'res_N (m)':>10}")
        print("  " + "-" * 30)
        for entry, oE, oN in zip(calibration_sites, offsets_E, offsets_N):
            print(f"  {int(entry['site']):>5}  "
                  f"{oE - origin_E:>10.2f}  {oN - origin_N:>10.2f}")
        rms_E = float(np.sqrt(np.mean((np.array(offsets_E) - origin_E) ** 2)))
        rms_N = float(np.sqrt(np.mean((np.array(offsets_N) - origin_N) ** 2)))
        print(f"  {'RMS':>5}  {rms_E:>10.2f}  {rms_N:>10.2f}")

    print()
    print(f"  Estimated mesh-centre UTM origin:")
    print(f"    UTM_ORIGIN_E = {origin_E:.1f}")
    print(f"    UTM_ORIGIN_N = {origin_N:.1f}")
    print(f"  Copy these values into the Configuration block.")
    print()

    return origin_E, origin_N


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
    UTM_ORIGIN_E, UTM_ORIGIN_N = estimate_utm_origin(
        CALIBRATION_SITES, OBSERVE_FILE, UTM_ZONE, UTM_NORTHERN, out=OUT
    )

# --- (2) Resolve slice positions to model-local metres ---------------------
slices_resolved = resolve_slices(PLOT_SLICES, UTM_ZONE, UTM_NORTHERN)
if OUT:
    print()

# --- (3) Optionally read site position from observe.dat -------------------
site_xy = None
if SITE_NUMBER is not None:
    print(f"Reading site {SITE_NUMBER} position from: {OBSERVE_FILE}")
    sx_m, sy_m = read_site_position(OBSERVE_FILE, SITE_NUMBER)
    site_xy = (sx_m, sy_m)
    print(f"  model-local : x = {sx_m/1000:.3f} km,  y = {sy_m/1000:.3f} km")
    if DISPLAY_COORDS == "utm":
        print(f"  UTM         : E = {sx_m + UTM_ORIGIN_E:.1f} m,  "
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
    site_xy     = site_xy,
    plot_file   = PLOT_FILE,
    dpi         = PLOT_DPI,
    out         = OUT,
)
print("Done.")
