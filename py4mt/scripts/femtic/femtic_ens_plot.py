#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_plot.py — Ensemble slice plot for a set of FEMTIC inversion runs.

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
WORK_DIR = r"/home/vrath/FEMTIC_work/ubinas_rto/"

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
#: Geographic coordinates (WGS-84) of the FEMTIC mesh origin.
UTM_ORIGIN_LAT = -16.409   # decimal degrees, positive = North
UTM_ORIGIN_LON = -71.537   # decimal degrees, positive = East

#: UTM coordinates of the mesh origin in metres (same zone as above).
UTM_ORIGIN_E = 229047.0   # easting  [m]
UTM_ORIGIN_N = 8184127.0  # northing [m]

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
#: Site names to overlay from SITE_DAT.  None = all sites in the file.
SITE_NAMES = None   # e.g. ["MT01", "MT05", "MT12"]  or None = all sites

#: Fallback (when SITE_DAT is None): 1-based site number(s) from observe.dat.
#: Int or list of int.  None = no overlay.
SITE_NUMBER = None

#: Marker style for map panels; dashed vertical line for curtain panels.
SITE_MARKER = dict(marker="v", color="black", ms=8, zorder=10,
                   label=None)

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
# Mesh-centre estimation  (optional — same as femtic_mod_plot.py)
# ---------------------------------------------------------------------------
ESTIMATE_ORIGIN = False
UPDATE_CONFIG   = True

CALIBRATION_SITES = [
    # dict(site=1,  crs="latlon", coords=[-71.500, -16.380]),
    # dict(site=10, crs="utm",    coords=[224500., 8179300.]),
]


# ===========================================================================
# Coordinate conversion helpers  (identical to femtic_mod_plot.py)
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
    """Resolve plane point to model-local metres."""
    return fem.resolve_pos_point(raw, zone, northern, UTM_ORIGIN_E, UTM_ORIGIN_N)


def resolve_slices(slices: list, zone: int, northern: bool) -> list:
    """Resolve all CRS-tagged positions in *slices* to model-local metres."""
    return fem.resolve_slice_positions(
        slices, zone, northern,
        UTM_ORIGIN_E, UTM_ORIGIN_N,
        UTM_ORIGIN_LAT, UTM_ORIGIN_LON,
        verbose=OUT,
    )


# ===========================================================================
# Display helpers  (identical to femtic_mod_plot.py)
# ===========================================================================

def _display_offset() -> tuple[float, float]:
    """Return (dE, dN) to add to model-local metres for display axis ticks."""
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
    """UTM [m] → (lat, lon) [°]."""
    return utl.utm_to_latlon_zn(E_m, N_m, zone, northern)


def _display_formatters(zone: int, northern: bool):
    """Return (x_formatter, y_formatter) for the chosen DISPLAY_COORDS."""
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
# Borehole helper  (identical to femtic_mod_plot.py)
# ===========================================================================

def _resolve_borehole_xy(spec: dict, zone: int, northern: bool) -> tuple[float, float]:
    """Resolve borehole x/y position specs to model-local metres."""
    return (_resolve_x0(spec["x"], zone, northern),
            _resolve_y0(spec["y"], zone, northern))


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

# --- (1) Derive UTM zone from mesh-origin coordinates ---------------------
UTM_ZONE, UTM_NORTHERN = _utm_zone_from_origin()
hemi = "N" if UTM_NORTHERN else "S"
print(f"UTM zone: {UTM_ZONE}{hemi}  "
      f"(origin lat={UTM_ORIGIN_LAT:.4f}°, lon={UTM_ORIGIN_LON:.4f}°)")
print()

# --- (1b) Optionally estimate UTM_ORIGIN_E / UTM_ORIGIN_N from sites ------
if ESTIMATE_ORIGIN:
    _cal_sites = list(CALIBRATION_SITES)
    if SITE_DAT is not None and os.path.isfile(SITE_DAT):
        _sdat = fem.read_site_dat(SITE_DAT)
        _sdat_ids = {d["sitenum"] for d in _sdat}
        _extra    = [d for d in _cal_sites if d.get("site") not in _sdat_ids]
        _cal_sites = _sdat + _extra
        if OUT:
            print(f"  site.dat: loaded {len(_sdat)} site(s) from {SITE_DAT}")
    UTM_ORIGIN_E, UTM_ORIGIN_N = fem.estimate_utm_origin(
        _cal_sites, OBSERVE_FILE, UTM_ZONE, UTM_NORTHERN,
        site_dat=SITE_DAT, out=OUT,
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

# --- (3) Optionally read site position(s) ----------------------------------
site_xys = []
if SITE_DAT is not None:
    print(f"Reading site position(s) from site.dat: {SITE_DAT}")
    _rows = fem.read_site_dat(SITE_DAT, site_names=SITE_NAMES)
    for row in _rows:
        sx_m, sy_m = fem.utm_to_model(row["easting"], row["northing"],
                                       UTM_ORIGIN_E, UTM_ORIGIN_N)
        site_xys.append((row["name"], sx_m, sy_m))
        print(f"  {row['name']}: model-local x = {sx_m/1000:.3f} km,"
              f"  y = {sy_m/1000:.3f} km")
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
        print(f"  site {_sn}: model-local x = {sx_m/1000:.3f} km,"
              f"  y = {sy_m/1000:.3f} km")
    print()

# --- (4) Build ensemble file list from ENS_DIRS ---------------------------
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

# --- (5) Ensemble slice plot  [from snippets.py Snippet 1] ----------------
if fviz is None:
    sys.exit("femtic_viz not available — cannot plot.  Check your installation.")

print(f"Plotting ensemble: {len(ENS_FILES)} member(s) …")
fviz.plot_ensemble_slices(
    member_files    = ENS_FILES,
    mesh_file       = MESH_FILE,
    slices          = slices_resolved,
    labels          = ENS_LABELS_resolved,
    stat_rows       = ENS_STAT_ROWS,
    cmap            = PLOT_CMAP,
    clim            = PLOT_CLIM,
    xlim            = PLOT_XLIM,
    ylim            = PLOT_YLIM,
    zlim            = PLOT_ZLIM,
    ocean_color     = PLOT_OCEAN_COLOR,
    ocean_value     = OCEAN_RHO,
    air_bgcolor     = PLOT_AIR_BGCOLOR,
    plot_file       = PLOT_ENS_FILE,
    per_member_file = ENS_PER_MEMBER,
    dpi             = PLOT_DPI,
    out             = OUT,
)
print("Ensemble plot done.")

# --- (6) Borehole resistivity logs ----------------------------------------
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
