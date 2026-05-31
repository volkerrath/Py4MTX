#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_plot_ensemble.py — Ensemble slice plot for a set of FEMTIC inversion runs.

Loops over a list of ensemble directories, collects the resistivity block
file from each, and produces:

  (1) A joint multi-row figure — one row per ensemble member — using the
      same slice geometry and PLOT_* parameters as femtic_mod_plot.py.
      Optional statistical summary rows (mean, std, median of log₁₀(ρ))
      are appended.  Optionally one figure per member is saved alongside.

  (2) Optionally, a borehole resistivity log figure (same as step (6) in
      femtic_mod_plot.py).

The ensemble step is taken directly from snippets.py (Snippet 1).

Slice positions, UTM/geographic coordinate handling, site overlay, and
all PLOT_* parameters follow the same conventions as femtic_mod_plot.py.
See that script and its README for full documentation.

Provenance
----------
    2026-05-24  vrath / Claude Sonnet 4.6   Created, based on
                femtic_mod_plot.py and snippets.py (Snippet 1).
                ENS_DIRS replaces ENS_FILES: the script loops over
                directories and builds the file list automatically.
    2026-05-31  vrath / Claude Sonnet 4.6   Aligned with femtic_mod_plot.py:
                replaced ESTIMATE_ORIGIN/CALIBRATION_SITES/UPDATE_CONFIG
                with ORIGIN_METHOD (None|"box"|"average"); origin estimation
                now runs before UTM zone derivation.  Removed local
                coordinate helpers (delegated to fem/utl).  site_xys tuples
                now carry elev.  plot_ensemble_slices call extended with
                site_xys, utm_origin_e/n, utm_zone, utm_northern,
                utm_to_latlon_fn, latlon_to_model_fn, display_coords,
                depth_km, horiz_km, equal_aspect, panel_height, nrows,
                ncols kwargs.  Added DEPTH_KM, HORIZ_KM, PLOT_EQUAL_ASPECT,
                PLOT_PANEL_HEIGHT, PLOT_NROWS, PLOT_NCOLS, PLOT_SITES_MAPS,
                PLOT_SITES_SLICES, SITE_MARKER_SLICES, MAP_MARKERS,
                DISPLAY_COORDS config vars.

@author: vrath
"""

import os
import sys
import glob
import math
import inspect
from pathlib import Path

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

#: Mesh file — always required for plotting.
MESH_FILE = WORK_DIR + "mesh.dat"

#: observe.dat — used by ESTIMATE_ORIGIN and as fallback for SITE_NUMBER.
OBSERVE_FILE = WORK_DIR + "observe.dat"

#: Site list produced by mt_make_sitelist.py (WHAT_FOR="femtic").
#: Format (comma-separated, no header):
#:   name, lat, lon, elev, sitenum, easting, northing
#: Easting/northing are UTM metres; model-local x/y is derived via
#: fem.utm_to_model using the mesh-centre origin.
#: Set to None to fall back to the observe.dat / SITE_NUMBER path.
SITE_DAT = WORK_DIR + "site.dat"   # set to None to disable

# ---------------------------------------------------------------------------
# Ensemble input
# ---------------------------------------------------------------------------
#: List of ensemble run directories.  Each directory must contain a
#: resistivity block file named according to BLOCK_PATTERN (see below).
#: Entries may be:
#:   - absolute or WORK_DIR-relative paths
#:   - glob patterns (e.g. WORK_DIR + "ubinas_rto_*") — expanded and sorted
#:
#: Example:
#:   ENS_DIRS = sorted(glob.glob(WORK_DIR + "ubinas_rto_*/"))
ENS_DIRS = [
    # WORK_DIR + "ubinas_rto_0/",
    # WORK_DIR + "ubinas_rto_1/",
    # WORK_DIR + "ubinas_rto_2/",
]

#: Filename pattern for the resistivity block inside each ensemble directory.
#: The placeholder {iter} is replaced by ENS_ITER.
#: Example: "resistivity_block_iter{iter}.dat"  →  resistivity_block_iter10.dat
BLOCK_PATTERN = "resistivity_block_iter{iter}.dat"

#: Inversion iteration whose block file is used from each directory.
ENS_ITER = 10

#: Labels for the member rows — one string per directory.
#: None → use the last component of each directory path.
ENS_LABELS = None

#: Statistical summary rows appended after the member rows.
#: Any subset of: "mean", "std", "median".
#: "mean"   → cell-wise mean   of log10(ρ) across all members
#: "std"    → cell-wise std    of log10(ρ); separate colormap (cividis)
#: "median" → cell-wise median of log10(ρ) across all members
ENS_STAT_ROWS = ["mean", "std"]

#: Output file for the joint ensemble figure.
#:   None → interactive show().
PLOT_ENS_FILE = WORK_DIR + "ensemble.pdf"

#: If True, also save one figure per member alongside the joint figure.
#: Per-member files are named by replacing ".pdf" with "_memberN.pdf".
ENS_PER_MEMBER = False

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
#: Fallback values used only when ORIGIN_METHOD is None or SITE_DAT is
#: unavailable.  When ORIGIN_METHOD is "box" or "average" these are
#: overwritten at runtime from site.dat and may be left as None.
UTM_ORIGIN_LAT = None      # decimal degrees, positive = North  (None → derived)
UTM_ORIGIN_LON = None      # decimal degrees, positive = East   (None → derived)

UTM_ORIGIN_E   = None      # easting  [m]  (None → derived from site.dat)
UTM_ORIGIN_N   = None      # northing [m]  (None → derived from site.dat)

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
# Axis scaling and layout
# ---------------------------------------------------------------------------
#: True → depth axis in km; False → metres.
DEPTH_KM = True

#: True → horizontal axes in km (model/utm modes); False → metres.
HORIZ_KM = True

#: Equal aspect ratio on map and curtain panels (model/utm coords only).
PLOT_EQUAL_ASPECT = True

#: Panel height in cm.  Width auto-computed from axis limits when PLOT_EQUAL_ASPECT.
PLOT_PANEL_HEIGHT = 16.0   # cm

#: Grid layout.  None → 1 row / len(PLOT_SLICES) columns.
PLOT_NROWS = None
PLOT_NCOLS = None

# ---------------------------------------------------------------------------
# Site overlay
# ---------------------------------------------------------------------------
#: Site names to overlay from SITE_DAT.  None = all sites in the file.
SITE_NAMES = None   # e.g. ["MT01", "MT05", "MT12"]  or None = all sites

#: Fallback (when SITE_DAT is None): 1-based site number(s) from observe.dat.
#: Int or list of int.  None = no overlay.
SITE_NUMBER = None

#: Show site markers on map panels.
PLOT_SITES_MAPS   = True
#: Show site markers on curtain (ns/ew) panels.
PLOT_SITES_SLICES = False

#: Maximum distance (m) from slice plane for site projection onto curtains.
PROJECTION_DIST = 5000.

#: Marker style for map panels.
SITE_MARKER = dict(marker="v", color="black", ms=8, zorder=10, label=None)

#: Marker style for curtain panels (None → same as SITE_MARKER).
SITE_MARKER_SLICES = None

#: Additional map markers (e.g. known features).  List of dicts:
#:   dict(pos=(x, y), marker="*", color="red", ms=10, label="label")
#: pos accepts model-local metres or (value, "utm"/"latlon") tuples.
MAP_MARKERS = []

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
#: Figure DPI for saved files.
PLOT_DPI = 300

#: Matplotlib colormap name.
PLOT_CMAP = "turbo_r"

#: Colour limits [log10(ρ_min), log10(ρ_max)] — None = auto.
PLOT_CLIM = [0.0, 4.0]      # log10(Ω·m)

#: Flat colour for ocean / lake cells.  None → use colormap.
PLOT_OCEAN_COLOR = "lightgrey"

#: Axes facecolor for air / background.  None = figure default.
PLOT_AIR_BGCOLOR = None

#: Slice specification — same format as femtic_mod_plot.py PLOT_SLICES.
#: Each dict must have 'kind' and the matching position key:
#:   kind="map"   → z0   (depth in model-local metres)
#:   kind="ns"    → x0   (easting;  plain float = model-local m,
#:                        or (value, "utm") / (value, "latlon"))
#:   kind="ew"    → y0   (northing; same CRS tagging)
#:   kind="plane" → point, strike, dip
#:   invert_x     → True to flip horizontal axis on ns/ew/plane panels
#:                  (for comparison with sections using opposite convention)
PLOT_SLICES = [
    dict(kind="map",  z0=5000.0),
    dict(kind="map",  z0=15000.0),
    dict(kind="ns",   x0=(-70.8700, "latlon")),
    dict(kind="ew",   y0=(-16.3500, "latlon")),
]

#: Global axis limits in model-local metres.  None → auto.
PLOT_XLIM = [-20000., 20000.]
PLOT_YLIM = [-20000., 20000.]
PLOT_ZLIM = [  -6000., 15000.]

#: Equal aspect ratio on map and curtain panels (model / utm coords only).
PLOT_EQUAL_ASPECT = True

# ---------------------------------------------------------------------------
# Borehole resistivity logs  (optional — same as step 6 in femtic_mod_plot.py)
# ---------------------------------------------------------------------------
#: Set True to produce a borehole figure after the ensemble plot.
PLOT_BOREHOLE = False

#: Output file for the borehole figure.  None → interactive show().
BOREHOLE_FILE = WORK_DIR + "ensemble_boreholes.pdf"

#: List of borehole spec dicts — same format as femtic_mod_plot.py.
#: Keys: "name", "x", "y", "z_top", "z_bot", "dz".
#: x/y accept plain float (model-local m) or (value, "utm"/"latlon") tuples.
BOREHOLE_SITES = [
    # dict(name="BH-01", x=0.0, y=0.0, z_top=0.0, z_bot=20000., dz=200.),
]

#: Matplotlib line style for borehole traces.
BOREHOLE_STYLE = dict(lw=1.2, marker="none")

#: x-axis limits [log10 min, log10 max] for borehole panels.  None = auto.
BOREHOLE_XLIM = [0.0, 4.0]

#: True = all boreholes on one axes; False = one panel per borehole.
BOREHOLE_SHARED = True

# ---------------------------------------------------------------------------
# Mesh-centre estimation from site.dat  (optional)
# ---------------------------------------------------------------------------
#: Method used to estimate UTM_ORIGIN_E / UTM_ORIGIN_N from SITE_DAT:
#:   None      — use the hard-coded UTM_ORIGIN_E / UTM_ORIGIN_N above
#:   "box"     — midpoint of the UTM bounding box of all sites (femticPY-compatible)
#:   "average" — arithmetic mean of all site UTM coordinates
#: Requires SITE_DAT to be set and readable.
ORIGIN_METHOD = "box"   # None | "box" | "average"


# ===========================================================================
# Borehole helper
# ===========================================================================

def _resolve_borehole_xy(spec: dict, zone: int, northern: bool) -> tuple[float, float]:
    """Resolve borehole x/y position specs to model-local metres."""
    return (
        fem.resolve_pos_x(spec["x"], zone, northern,
                          UTM_ORIGIN_E, UTM_ORIGIN_N,
                          UTM_ORIGIN_LAT, UTM_ORIGIN_LON),
        fem.resolve_pos_y(spec["y"], zone, northern,
                          UTM_ORIGIN_E, UTM_ORIGIN_N,
                          UTM_ORIGIN_LAT, UTM_ORIGIN_LON),
    )


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

    Delegates point-in-element search to ``fem.extract_borehole_log``.
    Parameters are identical to femtic_mod_plot.py.
    """
    if fviz is None:
        print("  plot_borehole_logs: femtic_viz not available — skipping.")
        return
    if not borehole_sites:
        print("  plot_borehole_logs: BOREHOLE_SITES is empty — skipping.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  plot_borehole_logs: Matplotlib not available — skipping.")
        return

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

    n    = len(borehole_sites)
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

# --- (1) Estimate origin from site.dat (before zone derivation) -----------
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
            _zone_boot, _north_boot = utl.utm_zone_from_latlon(
                float(_lats.mean()), float(_lons.mean()), override=UTM_ZONE_OVERRIDE)
            UTM_ORIGIN_LAT, UTM_ORIGIN_LON = utl.utm_to_latlon_zn(
                UTM_ORIGIN_E, UTM_ORIGIN_N, _zone_boot, _north_boot)
            if OUT:
                print(f"Origin estimated ({ORIGIN_METHOD}, {len(_sdat)} sites):")
                print(f"  UTM_ORIGIN_E   = {UTM_ORIGIN_E:.1f} m")
                print(f"  UTM_ORIGIN_N   = {UTM_ORIGIN_N:.1f} m")
                print(f"  UTM_ORIGIN_LAT = {UTM_ORIGIN_LAT:.6f}°")
                print(f"  UTM_ORIGIN_LON = {UTM_ORIGIN_LON:.6f}°")
                print()

# --- (2) Derive UTM zone from finalised origin -----------------------------
UTM_ZONE, UTM_NORTHERN = utl.utm_zone_from_latlon(
    UTM_ORIGIN_LAT, UTM_ORIGIN_LON, override=UTM_ZONE_OVERRIDE)
hemi = "N" if UTM_NORTHERN else "S"
print(f"UTM zone: {UTM_ZONE}{hemi}  "
      f"(origin lat={UTM_ORIGIN_LAT:.4f}°, lon={UTM_ORIGIN_LON:.4f}°)")
print()

# --- (3) Resolve slice positions to model-local metres --------------------
slices_resolved = fem.resolve_slice_positions(
    PLOT_SLICES, UTM_ZONE, UTM_NORTHERN,
    UTM_ORIGIN_E, UTM_ORIGIN_N,
    UTM_ORIGIN_LAT, UTM_ORIGIN_LON,
    verbose=OUT,
)
if OUT:
    print()

# --- (4) Read site positions ----------------------------------------------
site_xys = []
_sites_from_obs = False
if SITE_DAT is not None and os.path.isfile(SITE_DAT):
    print(f"Reading site positions from site.dat: {SITE_DAT}")
    _rows = fem.read_site_dat(SITE_DAT, site_names=SITE_NAMES)
    for row in _rows:
        sx_m, sy_m = fem.utm_to_model(row["easting"], row["northing"],
                                      UTM_ORIGIN_E, UTM_ORIGIN_N)
        site_xys.append((row["name"], sx_m, sy_m, float(row.get("elev", 0.0))))
        if OUT:
            print(f"  {row['name']}: model-local x = {sx_m/1000:.3f} km,"
                  f"  y = {sy_m/1000:.3f} km")
    if not site_xys:
        print("  (no matching sites found in site.dat)")
    print()
elif SITE_NUMBER is not None:
    _site_nums = (SITE_NUMBER if isinstance(SITE_NUMBER, (list, tuple))
                  else [SITE_NUMBER])
    print(f"Reading site positions from observe.dat: {OBSERVE_FILE}")
    for _sn in _site_nums:
        sx_m, sy_m = fem.read_site_position(OBSERVE_FILE, _sn)
        site_xys.append((_sn, sx_m, sy_m, 0.0))
        if OUT:
            print(f"  site {_sn}: model-local x = {sx_m/1000:.3f} km,"
                  f"  y = {sy_m/1000:.3f} km")
    _sites_from_obs = True
    print()

# --- (5) Build ensemble file list from ENS_DIRS ---------------------------
# Expand any glob patterns, sort, then locate the block file in each dir.
_expanded_dirs = []
for _d in ENS_DIRS:
    _matches = sorted(glob.glob(str(_d)))
    if _matches:
        _expanded_dirs.extend(_matches)
    else:
        _expanded_dirs.append(str(_d))   # keep as-is; will fail gracefully below

if not _expanded_dirs:
    sys.exit("ENS_DIRS is empty — nothing to plot.  Set at least one directory.")

_block_name = BLOCK_PATTERN.format(iter=ENS_ITER)
ENS_FILES  = []
ENS_LABELS_resolved = []
_missing   = []
for _d in _expanded_dirs:
    _f = os.path.join(_d, _block_name)
    if os.path.isfile(_f):
        ENS_FILES.append(_f)
        _label = (ENS_LABELS[len(ENS_FILES) - 1]
                  if ENS_LABELS and len(ENS_FILES) <= len(ENS_LABELS)
                  else os.path.basename(os.path.normpath(_d)))
        ENS_LABELS_resolved.append(_label)
    else:
        _missing.append(_f)

if _missing:
    print(f"WARNING: {len(_missing)} block file(s) not found:")
    for _mf in _missing:
        print(f"  {_mf}")

if not ENS_FILES:
    sys.exit(f"No block files found (pattern: {_block_name}).  "
             f"Check ENS_DIRS and ENS_ITER.")

if OUT:
    print(f"Ensemble: {len(ENS_FILES)} member(s)  "
          f"(block: {_block_name})")
    for _lbl, _f in zip(ENS_LABELS_resolved, ENS_FILES):
        print(f"  {_lbl:30s}  {_f}")
    print()

# --- (6) Ensemble slice plot  [from snippets.py Snippet 1] ----------------
if fviz is None:
    sys.exit("femtic_viz not available — cannot plot.  Check your installation.")

print(f"Plotting ensemble: {len(ENS_FILES)} member(s) …")
fviz.plot_ensemble_slices(
    member_files       = ENS_FILES,
    mesh_file          = MESH_FILE,
    slices             = slices_resolved,
    labels             = ENS_LABELS_resolved,
    stat_rows          = ENS_STAT_ROWS,
    cmap               = PLOT_CMAP,
    clim               = PLOT_CLIM,
    xlim               = PLOT_XLIM,
    ylim               = PLOT_YLIM,
    zlim               = PLOT_ZLIM,
    ocean_color        = PLOT_OCEAN_COLOR,
    ocean_value        = OCEAN_RHO,
    air_bgcolor        = PLOT_AIR_BGCOLOR,
    site_xys           = site_xys,
    obs_coords_only    = _sites_from_obs,
    sites_in_maps      = PLOT_SITES_MAPS,
    sites_in_slices    = PLOT_SITES_SLICES,
    site_marker        = SITE_MARKER,
    site_marker_slices = SITE_MARKER_SLICES,
    map_markers        = MAP_MARKERS,
    projection_dist    = PROJECTION_DIST,
    display_coords     = DISPLAY_COORDS,
    utm_origin_e       = UTM_ORIGIN_E,
    utm_origin_n       = UTM_ORIGIN_N,
    utm_zone           = UTM_ZONE,
    utm_northern       = UTM_NORTHERN,
    utm_to_latlon_fn   = utl.utm_to_latlon_zn,
    latlon_to_model_fn = fem.latlon_to_model,
    depth_km           = DEPTH_KM,
    horiz_km           = HORIZ_KM,
    equal_aspect       = PLOT_EQUAL_ASPECT,
    panel_height       = PLOT_PANEL_HEIGHT / 2.54,
    nrows              = PLOT_NROWS,
    ncols              = PLOT_NCOLS,
    plot_file          = PLOT_ENS_FILE,
    per_member_file    = ENS_PER_MEMBER,
    dpi                = PLOT_DPI,
    out                = OUT,
)
print("Ensemble plot done.")

# --- (7) Borehole resistivity logs ----------------------------------------
if PLOT_BOREHOLE:
    if not BOREHOLE_SITES:
        print("  Borehole plot skipped: BOREHOLE_SITES is empty.")
    else:
        # Use the first ensemble member as the reference model for the log.
        print(f"Sampling {len(BOREHOLE_SITES)} borehole(s) "
              f"from member 0 ({ENS_LABELS_resolved[0]}) …")
        plot_borehole_logs(
            model_file     = ENS_FILES[0],
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
