#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_plot_3d.py — PyVista 3-D rendering and VTK/VTU export for a FEMTIC
resistivity model.

Produces:
  • An interactive PyVista scene or static screenshot with axis-aligned slice
    planes, oblique planes, and iso-surfaces (``fviz.plot_model_3d``).
  • Optionally a VTU/VTK file of the full unstructured grid for ParaView /
    Zenodo deposit.

Sister script ``femtic_mod_plot_slice.py`` handles 2-D slice panels and
borehole logs from the same mesh and block file.

UTM zone derivation
--------------------
    Zone number is computed from ``UTM_ORIGIN_LON`` (standard 6° bands).
    Override with ``UTM_ZONE_OVERRIDE`` when needed.

Provenance
----------
    2026-05-13  vrath / Claude Sonnet 4.6   Created as part of
                femtic_mod_plot.py; PLOT3D config block with axis-aligned
                x/y/z slices, oblique planes, and iso-surfaces via
                fviz.plot_model_3d.  Output: interactive HTML or screenshot.
    2026-05-26  Claude Sonnet 4.6 (Anthropic)
                Added PLOT3D_VTU_FILE config var; changed PLOT3D_FILE default
                from .html to .png.  plot_model_3d moved into femtic_viz.py.
    2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
                Passed PLOT_XLIM/YLIM/ZLIM to plot_model_3d for spatial
                clipping of the VTU export and PyVista scene.
    2026-05-31  vrath / Claude Sonnet 4.6 (Anthropic)
                Origin estimation now runs before UTM zone derivation.
                Hard-coded UTM_ORIGIN_* set to None (derived at runtime).
    2026-06-03  Claude Sonnet 4.6 (Anthropic)
                Split from femtic_mod_plot.py into femtic_mod_plot_slice.py
                (2-D slices + boreholes) and femtic_mod_plot_3d.py (this
                script, PyVista 3-D rendering only).

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

#: Mesh file — always required.
MESH_FILE = WORK_DIR + "mesh.dat"

#: Site list produced by mt_make_sitelist.py — used only for origin estimation.
#: Set to None if UTM_ORIGIN_E/N are hard-coded.
SITE_DAT = WORK_DIR + "site.dat"   # set to None to disable

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Ocean / air handling (must match the inversion setup)
# ---------------------------------------------------------------------------
#: None → auto-infer from region 1 heuristic (ρ ≤ 1 Ω·m AND flag==1).
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
# Mesh-centre estimation from site.dat UTM coordinates  (optional)
# ---------------------------------------------------------------------------
#: Method used to estimate UTM_ORIGIN_E / UTM_ORIGIN_N from SITE_DAT:
#:   None      — use the hard-coded values above
#:   "box"     — midpoint of the UTM bounding box of all sites in SITE_DAT
#:   "average" — arithmetic mean of all site UTM coordinates in SITE_DAT
ORIGIN_METHOD = "box"   # None | "box" | "average"

# ---------------------------------------------------------------------------
# Spatial clip box  (applied to both VTU export and PyVista scene)
# ---------------------------------------------------------------------------
#: Model-local metres.  None → no clipping on that axis.
PLOT_XLIM = [-15000., 15000.]   # [xmin, xmax] easting
PLOT_YLIM = [-15000., 15000.]   # [ymin, ymax] northing
PLOT_ZLIM = [-6000.,  15000.]   # [zmin, zmax] depth (z positive-down)

# ---------------------------------------------------------------------------
# 3-D rendering  (requires PyVista; conda install -c conda-forge pyvista)
# ---------------------------------------------------------------------------
#: Output file for the rendered view.
#:   .vtu / .vtk → VTK unstructured-grid for ParaView (no rendering done here).
#:   .html       → interactive WebGL in browser (requires pyvista[jupyter] /
#:                 trame_vtk; falls back to .png if trame_vtk absent).
#:   .png / .jpg → static screenshot.
#:   None        → open an interactive PyVista window (requires a display).
PLOT3D_FILE = WORK_DIR + "resistivity_block_iter17_3d.png"

#: Optional separate VTK export for ParaView / Zenodo.
#:   .vtu  → VTK XML unstructured grid  (recommended).
#:   .vtk  → legacy VTK binary/ASCII.
#:   None  → no grid file exported.
PLOT3D_VTU_FILE = WORK_DIR + "resistivity_block_iter17.vtu"

#: Scalar field to display.  "log10_resistivity" or "resistivity".
PLOT3D_SCALAR = "log10_resistivity"

#: Colour limits [vmin, vmax] for the scalar.  None → PyVista auto.
PLOT3D_CLIM = [0.0, 3.0]       # log10(Ω·m)

#: Matplotlib / PyVista colormap.
PLOT3D_CMAP = "turbo_r"

#: Axis-aligned slice positions in model-local metres (z positive-down).
#: Each list entry places one cutting plane perpendicular to that axis.
#: Empty list or None → no slices along that axis.
PLOT3D_SLICE_X = [0.0]                     # YZ planes — N-S sections
PLOT3D_SLICE_Y = [0.0]                     # XZ planes — E-W sections
PLOT3D_SLICE_Z = [5000.0, 15000.0]         # XY planes — horizontal maps

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

#: Figure DPI for saved screenshot files.
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

# --- (3) 3-D PyVista plot --------------------------------------------------
if fviz is None:
    sys.exit("femtic_viz not available — cannot plot.  Check your installation.")

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
