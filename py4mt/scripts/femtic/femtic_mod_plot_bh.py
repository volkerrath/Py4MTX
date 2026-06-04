#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_plot_bh.py — 1-D ρ(z) borehole resistivity logs for a FEMTIC model.

Samples resistivity along one or more vertical boreholes and produces a ρ(z)
figure with a logarithmic x-axis (Ohm·m), using
``fviz.plot_borehole_logs``.

Sister scripts:
  ``femtic_mod_plot_slice.py`` — 2-D map / curtain / plane slice panels.
  ``femtic_mod_plot_3d.py``   — PyVista 3-D rendering and VTK/VTU export.

Borehole position coordinate systems
--------------------------------------
Each spec dict ``"x"`` / ``"y"`` field accepts:

    plain float          model-local metres (origin at mesh centre)
    (value, "utm")       UTM metres in the mesh UTM zone
    (value, "latlon")    decimal degrees (longitude for x, latitude for y)

``"z_top"`` may be the float depth [m, z-down] or the string ``"surface"``
to auto-detect the mesh surface elevation at the borehole location (requires
scipy).

Execution steps
---------------
(1) Optionally estimate UTM mesh-origin from ``SITE_DAT`` bounding-box / mean
(2) Derive UTM zone from finalised ``UTM_ORIGIN_LAT`` / ``UTM_ORIGIN_LON``
(3) Plot borehole logs via ``fviz.plot_borehole_logs``

Provenance
----------
    2026-06-04  vrath / Claude Sonnet 4.6 (Anthropic)
                Split from femtic_mod_plot_slice.py.  Borehole config block
                (BOREHOLE_*, PLOT_BOREHOLE) and execution step moved here.
                fviz (femtic_viz) used directly.
    2026-06-04  vrath / Claude Sonnet 4.6 (Anthropic)
                Added BOREHOLE_MARKERS config block for free annotations
                (arrows + text) and LEGEND_FONTSIZE for the legend / panel
                title font size.  Both forwarded to
                ``fviz.plot_borehole_logs``.

@author: vrath
@project: Py4MTX
@created: 2026-06-04
@modified: 2026-06-04  vrath / Claude Sonnet 4.6 (Anthropic)
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

#: Resistivity block to sample (any iteration).
MODEL_FILE = WORK_DIR + "resistivity_block_iter17.dat"

#: Mesh file — always required.
MESH_FILE = WORK_DIR + "mesh.dat"

#: Site list produced by mt_make_sitelist.py (WHAT_FOR="femtic").
#: Format (comma-separated, no header):
#:   name, lat, lon, elev, sitenum, easting, northing
#: Used for origin estimation only (no site overlay in borehole figures).
#: Set to None to disable.
SITE_DAT = WORK_DIR + "site.dat"

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Ocean / air handling (must match the inversion setup)
# ---------------------------------------------------------------------------
AIR_RHO   = 1.0e9   # Ω·m  (region 0)
OCEAN_RHO = 0.25    # Ω·m  (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# Geographic / UTM origin of the mesh centre
# ---------------------------------------------------------------------------
UTM_ORIGIN_LAT = None   # decimal degrees, positive = North
UTM_ORIGIN_LON = None   # decimal degrees, positive = East

UTM_ORIGIN_E = None   # easting  [m]
UTM_ORIGIN_N = None   # northing [m]

#: Override the auto-derived UTM zone number.  None = auto from origin lat/lon.
UTM_ZONE_OVERRIDE = None

# ---------------------------------------------------------------------------
# Mesh-centre estimation from site.dat UTM coordinates  (optional)
# ---------------------------------------------------------------------------
#: Method used to estimate UTM_ORIGIN_E / UTM_ORIGIN_N from SITE_DAT:
#:   None      — use the hard-coded values above
#:   "box"     — midpoint of the UTM bounding box of all sites in SITE_DAT
#:   "average" — arithmetic mean of all site UTM coordinates in SITE_DAT
ORIGIN_METHOD = "box"   # None | "box" | "average"

# ---------------------------------------------------------------------------
# Borehole resistivity logs
# ---------------------------------------------------------------------------
#: Output file for the borehole figure.  None → interactive show().
BOREHOLE_FILE = WORK_DIR + "resistivity_block_iter17_boreholes.pdf"

#: List of borehole specifications.  Each entry is a dict with:
#:   "name"   : str   — label shown in the legend / panel title
#:   "x"      : float — model-local easting  [m]  (or CRS-tagged tuple)
#:   "y"      : float — model-local northing [m]  (or CRS-tagged tuple)
#:   "z_top"  : float | "surface"
#:              float — start depth [m, z-down]; 0 = datum surface
#:              "surface" — auto-detect from mesh nodes via KD-tree (scipy req.)
#:   "z_bot"  : float — end depth [m, z-down], e.g. 20000.0 for 20 km
#:   "dz"     : float — sampling interval [m], e.g. 100.0
#:
#:   Optional keys:
#:   "lat"    : float — geographic latitude  [°]  shown in legend instead of y_m
#:   "lon"    : float — geographic longitude [°]  shown in legend instead of x_m
#:              Both must be provided together.
#:   Any Matplotlib Line2D kwarg ("color", "ls", "lw", "marker", "alpha", …)
#:   placed in the spec dict overrides BOREHOLE_STYLE for that trace only.
BOREHOLE_SITES = [
    dict(name="borehole1",
         x=(-70.868025, "latlon"), y=(-16.363436, "latlon"),
         z_top="surface", z_bot=5000., dz=50.,
         color="steelblue", ls="-", lw=3.2),
    dict(name="borehole crater",
         x=(-70.8972, "latlon"), y=(-16.3450, "latlon"),
         z_top="surface", z_bot=4000., dz=50.,
         color="red", ls=":", lw=3.2),
    # dict(name="BH-01",
    #      x=(-70.90, "latlon"), y=(-16.40, "latlon"),
    #      z_top="surface", z_bot=20000., dz=200.,
    #      color="firebrick", ls="--"),
    # dict(name="BH-02",
    #      x=(229100., "utm"), y=(8184000., "utm"),
    #      z_top="surface", z_bot=15000., dz=100.,
    #      color="seagreen", ls="-.", lw=1.5),
]

#: Baseline Matplotlib line / marker style for all borehole traces.
#: Per-spec keys in BOREHOLE_SITES override these for individual traces.
BOREHOLE_STYLE = dict(lw=1.2, marker="none")

#: x-axis limits in Ohm*m (log scale).  None → auto.
BOREHOLE_XLIM = [1.0, 1e4]   # Ohm*m

#: True = all boreholes on one shared axes; False = one panel per borehole.
BOREHOLE_SHARED = False


#: Free annotations (arrows + text) added to the borehole depth panels.
#: Each entry is a dict with:
#:   "depth"    : float  — depth in metres (z-down); REQUIRED.
#:   "rho"      : float  — x-position of the arrow tip in Ohm·m (optional;
#:                         defaults to the left x-axis edge).
#:   "text"     : str    — annotation label (optional; default "").
#:   "borehole" : str or list of str — borehole name(s) to annotate
#:                (optional; None = all panels).
#:   "xytext"   : (dx_log_factor, dy_km) — text offset relative to tip
#:                (optional; default (1.5, -0.3)).
#:   "arrowprops" : dict — forwarded to ax.annotate (optional; default thin
#:                         black -> arrow).
#:   Any additional keys are passed verbatim to ax.annotate
#:   (e.g. "color", "fontsize", "fontweight", "ha", "va", "zorder").
#:
#: Set to [] or None to place no markers.
BOREHOLE_MARKERS = [
    # dict(depth=1500., rho=10., text="conductor",
    #      borehole="borehole1",
    #      color="red", fontsize=8, fontweight="bold",
    #      arrowprops=dict(arrowstyle="->", color="red", lw=1.2)),
    # dict(depth=3200., text="resistive basement",
    #      color="navy", fontsize=8),
]

#: Font size for the borehole legend (shared mode) and panel titles
#: (per-panel mode).  Tick labels are set to legend_fontsize - 2.
LEGEND_FONTSIZE = 9


#: Export sampled depth / rho arrays to an NPZ file.
#:   True  — save; path is derived from BOREHOLE_FILE (same stem, .npz extension)
#:            or "borehole_logs.npz" when BOREHOLE_FILE is None.
#:   False — skip NPZ export.
#:   str / Path — explicit output path (overrides auto-derive).
BOREHOLE_NPZ = True

#: Figure DPI for saved file.
PLOT_DPI = 600


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

# --- (3) Borehole resistivity logs ----------------------------------------
if fviz is None:
    sys.exit("femtic_viz not available — cannot plot.  "
             "Check your installation.")

if not BOREHOLE_SITES:
    print("  Borehole plot skipped: BOREHOLE_SITES is empty.")
else:
    print(f"Sampling {len(BOREHOLE_SITES)} borehole(s) …")
    fviz.plot_borehole_logs(
        model_file     = MODEL_FILE,
        mesh_file      = MESH_FILE,
        borehole_sites = BOREHOLE_SITES,
        resolve_xy_fn  = None,
        utm_zone       = UTM_ZONE,
        utm_northern   = UTM_NORTHERN,
        utm_origin_e   = UTM_ORIGIN_E,
        utm_origin_n   = UTM_ORIGIN_N,
        ocean_value    = OCEAN_RHO,
        clim           = BOREHOLE_XLIM,
        borehole_style = BOREHOLE_STYLE,
        shared         = BOREHOLE_SHARED,
        markers        = BOREHOLE_MARKERS or None,
        legend_fontsize= LEGEND_FONTSIZE,
        npz_file       = (None   if BOREHOLE_NPZ is True
                          else (False if BOREHOLE_NPZ is False
                                else BOREHOLE_NPZ)),
        plot_file      = BOREHOLE_FILE,
        dpi            = PLOT_DPI,
        out            = OUT,
    )
    print("Borehole plot done.")
