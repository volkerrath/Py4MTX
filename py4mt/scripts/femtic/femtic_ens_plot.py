#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_plot_ensemble.py — Ensemble slice plot for a set of FEMTIC inversion runs.

Scans ENSEMBLE_DIR for converged ensemble members exactly the way
femtic_ens_post.py does (ENSEMBLE_NAME* sub-directories, femtic.cnv,
NRMS_MAX threshold), then produces:

  (1) [default, PER_MEMBER_PLOT=True] Two fviz.plot_model_slices()
      figures per converged member: the perturbed prior model
      (resistivity_block_iter0.dat) and the best-fit model
      (resistivity_block_iter{numit}.dat, numit from femtic.cnv), saved
      as "<label>_iter0.<ext>" / "<label>_best.<ext>" in WORK_DIR.

  (2) [optional, PLOT_JOINT=True] The previous joint multi-row figure —
      one row per member's best-fit model — with optional mean/std/median
      summary rows, via fviz.plot_ensemble_slices().

  (3) Optionally, a borehole resistivity log figure (same as step (6) in
      femtic_mod_plot.py), sampled from the first converged member's
      best-fit model.

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
    2026-08-13  Claude Sonnet 5 (Anthropic)
                Added femtic_ens_plot_summary.md output at end of run:
                writes user-set (UPPERCASE) parameters, script path, and
                run date/time via utl.write_param_summary().
    2026-08-14  Claude Sonnet 5 (Anthropic)
                Rewrote member handling to match femtic_ens_post.py:
                converged members are now discovered by scanning
                ENSEMBLE_DIR for "ENSEMBLE_NAME*" sub-directories and
                reading femtic.cnv (same NRMS_MAX threshold, same
                nRMS-column FEMTIC-version logic), replacing the old
                fixed ENS_DIRS/BLOCK_PATTERN/ENS_ITER file list.
                For each converged member, two single-model figures are
                now produced via fviz.plot_model_slices(): the perturbed
                prior (resistivity_block_iter0.dat) and the best-fit
                model (resistivity_block_iter{numit}.dat, numit from
                femtic.cnv), saved as "<label>_iter0.<ext>" and
                "<label>_best.<ext>" in WORK_DIR. Controlled by new
                PER_MEMBER_PLOT flag (default True). The previous joint
                multi-row figure (fviz.plot_ensemble_slices, with
                mean/std/median summary rows) is kept as an optional
                extra, now gated by PLOT_JOINT (default False) and still
                built from the same converged-member file list. Its
                fviz.plot_ensemble_slices() call still has the
                previously flagged keyword mismatch against the
                function's current signature (would raise TypeError if
                PLOT_JOINT=True) — this remains out of scope here and
                is called out in a code comment at the call site.

@author: vrath
"""

import os
import sys
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
# Ensemble input — converged-member discovery
# ---------------------------------------------------------------------------
#: FEMTIC version used for the run — controls which column of the last
#: femtic.cnv line holds nRMS.  Must match femtic_ens_post.py's setting
#: for the same ensemble so both scripts agree on what "converged" means.
FEMTIC = "5.0"   # "4.3" | "5.0"

#: Directory containing one sub-directory per ensemble member.
ENSEMBLE_DIR = WORK_DIR

#: Member sub-directories are matched via glob "<ENSEMBLE_NAME>*".
ENSEMBLE_NAME = "ubinas_rto"

#: Maximum normalised RMS accepted from femtic.cnv.  Keep this equal to
#: NRMS_MAX in femtic_ens_post.py so this script plots exactly the
#: members ens_post included in its ensemble statistics.
NRMS_MAX = 1.5

#: Labels for the member plots/filenames — one string per converged
#: member, in the order directories are found.  None → use each member
#: directory's basename.
ENS_LABELS = None

# ---------------------------------------------------------------------------
# Per-member plots (default): iter0 (perturbed prior) + best-fit model
# ---------------------------------------------------------------------------
#: If True, produce two fviz.plot_model_slices() figures for every
#: converged member: "<label>_iter0.<ext>" (resistivity_block_iter0.dat,
#: the perturbed/prior model) and "<label>_best.<ext>"
#: (resistivity_block_iter{numit}.dat, the best-fit model at the
#: iteration femtic.cnv reports).  Saved into WORK_DIR.
PER_MEMBER_PLOT = True

#: File extension for per-member plots (passed to plot_model_slices).
PER_MEMBER_FORMAT = "pdf"

# ---------------------------------------------------------------------------
# Joint ensemble figure (optional extra)
# ---------------------------------------------------------------------------
#: If True, additionally build the old joint multi-row figure — one row
#: per converged member (best-fit model) — via fviz.plot_ensemble_slices,
#: with optional mean/std/median summary rows.
PLOT_JOINT = False

#: Statistical summary rows appended after the member rows in the joint
#: figure.  Any subset of: "mean", "std", "median".
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


def _plot_member_slice(
    block_file: str,
    out_file: str,
    *,
    slices_resolved: list,
    site_xys: list,
    obs_coords_only: bool,
    figure_title: str,
    nrms_annotation: dict | None = None,
) -> None:
    """Call fviz.plot_model_slices once for a single block file.

    Mirrors femtic_ens_post.py's ``_plot_slice`` helper: same PLOT_*
    options (CRS handling, site overlay, figure layout) used throughout
    this script, applied to one resistivity block file at a time.
    """
    if fviz is None:
        print("  plot_member_slice: femtic_viz not available — skipping.")
        return

    fviz.plot_model_slices(
        model_file          = block_file,
        mesh_file           = MESH_FILE,
        slices              = slices_resolved,
        cmap                = PLOT_CMAP,
        clim                = PLOT_CLIM,
        xlim                = PLOT_XLIM,
        ylim                = PLOT_YLIM,
        zlim                = PLOT_ZLIM,
        ocean_color         = PLOT_OCEAN_COLOR,
        ocean_value         = OCEAN_RHO,
        air_bgcolor         = PLOT_AIR_BGCOLOR,
        site_xys            = site_xys,
        obs_coords_only     = obs_coords_only,
        sites_in_maps       = PLOT_SITES_MAPS,
        sites_in_slices     = PLOT_SITES_SLICES,
        site_marker         = SITE_MARKER,
        site_marker_slices  = SITE_MARKER_SLICES,
        map_markers         = MAP_MARKERS,
        projection_dist     = PROJECTION_DIST,
        display_coords      = DISPLAY_COORDS,
        utm_origin_e        = UTM_ORIGIN_E,
        utm_origin_n        = UTM_ORIGIN_N,
        utm_zone            = UTM_ZONE,
        utm_northern        = UTM_NORTHERN,
        utm_to_latlon_fn    = utl.utm_to_latlon_zn,
        latlon_to_model_fn  = fem.latlon_to_model,
        depth_km            = DEPTH_KM,
        horiz_km            = HORIZ_KM,
        equal_aspect        = PLOT_EQUAL_ASPECT,
        panel_height        = PLOT_PANEL_HEIGHT / 2.54,
        nrows               = PLOT_NROWS,
        ncols               = PLOT_NCOLS,
        nrms_annotation     = nrms_annotation,
        figure_title        = figure_title,
        plot_file           = out_file,
        dpi                 = PLOT_DPI,
        out                 = OUT,
    )
    if OUT:
        print(f"    saved → {out_file}")


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

# --- (5) Scan ensemble directories for converged members -------------------
# Mirrors femtic_ens_post.py Step (1) exactly: same NRMS_MAX threshold,
# same femtic.cnv column logic per FEMTIC version, same
# resistivity_block_iter{numit}.dat naming for the best-fit model — so
# this script plots precisely the members ens_post included in its
# ensemble statistics.
dir_list = utl.get_filelist(
    searchstr=[ENSEMBLE_NAME + "*"],
    searchpath=ENSEMBLE_DIR,
    fullpath=True,
)
print(f"Found {len(dir_list)} sub-directory/ies matching '{ENSEMBLE_NAME}'.")

model_list = []   # list of dicts: label, dir, numit, nrms, iter0_file, best_file
for _d in dir_list:
    if not os.path.isdir(_d):
        print(f"\n  {_d}: not a directory — skipped (not an ensemble run).")
        continue

    print(f"\n  Inversion run: {_d}")
    _cnv_file = os.path.join(_d, "femtic.cnv")
    if not os.path.isfile(_cnv_file):
        print(f"    femtic.cnv not found — skipped.")
        continue

    with open(_cnv_file) as _fh:
        _cnv = _fh.readlines()
    _info = _cnv[-1].split()
    if "4.3" in FEMTIC:
        _numit = int(_info[0])
        _nrms  = float(_info[6])
    elif "5." in FEMTIC:
        _numit = int(_info[0])
        _nrms  = float(_info[8])
    else:
        sys.exit(f"FEMTIC version {FEMTIC!r} not recognised. Exit.")

    if _nrms > NRMS_MAX:
        print(f"    nRMS={_nrms:.4f} > NRMS_MAX={NRMS_MAX} — skipped.")
        continue

    _best_file  = os.path.join(_d, f"resistivity_block_iter{_numit}.dat")
    _iter0_file = os.path.join(_d, "resistivity_block_iter0.dat")

    if not os.path.isfile(_best_file):
        print(f"    {_best_file} not found — skipped.")
        continue
    if not os.path.isfile(_iter0_file):
        print(f"    {_iter0_file} not found — skipped.")
        continue

    _idx   = len(model_list)
    _label = (ENS_LABELS[_idx] if ENS_LABELS and _idx < len(ENS_LABELS)
              else os.path.basename(os.path.normpath(_d)))

    print(f"    iter={_numit}  nRMS={_nrms:.4f}  {_best_file}")
    model_list.append(dict(
        label=_label, dir=_d, numit=_numit, nrms=_nrms,
        iter0_file=_iter0_file, best_file=_best_file,
    ))

n_members = len(model_list)
print(f"\nConverged members: {n_members}")

if n_members == 0:
    sys.exit("No converged members found. Nothing to do.")

ENS_FILES = [m["best_file"] for m in model_list]   # kept for PLOT_JOINT / borehole
ENS_LABELS_resolved = [m["label"] for m in model_list]

if OUT:
    print(f"\nEnsemble: {n_members} converged member(s)")
    for _m in model_list:
        print(f"  {_m['label']:30s}  iter0={os.path.basename(_m['iter0_file'])}"
              f"  best=iter{_m['numit']} (nRMS={_m['nrms']:.4f})")
    print()

# --- (6) Per-member plots: iter0 (perturbed prior) + best-fit model -------
if PER_MEMBER_PLOT:
    if fviz is None:
        sys.exit("femtic_viz not available — cannot plot.  Check your installation.")

    print(f"Plotting {n_members} converged member(s), "
          f"2 figures each (iter0 + best) …")
    for _m in model_list:
        _label = _m["label"]

        _iter0_out = os.path.join(WORK_DIR, f"{_label}_iter0.{PER_MEMBER_FORMAT}")
        print(f"\n  member {_label!r}: perturbed prior (iter0)")
        _plot_member_slice(
            block_file      = _m["iter0_file"],
            out_file        = _iter0_out,
            slices_resolved = slices_resolved,
            site_xys        = site_xys,
            obs_coords_only = _sites_from_obs,
            figure_title    = f"{_label} — perturbed prior (iter0)",
        )

        _best_out = os.path.join(WORK_DIR, f"{_label}_best.{PER_MEMBER_FORMAT}")
        print(f"  member {_label!r}: best fit (iter{_m['numit']}, "
              f"nRMS={_m['nrms']:.4f})")
        _plot_member_slice(
            block_file      = _m["best_file"],
            out_file        = _best_out,
            slices_resolved = slices_resolved,
            site_xys        = site_xys,
            obs_coords_only = _sites_from_obs,
            figure_title    = f"{_label} — best fit (iter{_m['numit']})",
            nrms_annotation = dict(nrms=_m["nrms"]),
        )
    print("\nPer-member plots done.")

# --- (6b) Joint multi-row ensemble figure (optional extra) ----------------
# NOTE: this call passes several kwargs (site_xys, obs_coords_only,
# sites_in_maps/slices, site_marker*, map_markers, projection_dist,
# display_coords, utm_origin_e/n, utm_zone, utm_northern,
# utm_to_latlon_fn, latlon_to_model_fn, depth_km, horiz_km, equal_aspect,
# panel_height, nrows, ncols) that are NOT present in the current
# fviz.plot_ensemble_slices() signature and will raise TypeError if
# PLOT_JOINT=True. This is the previously flagged signature mismatch;
# it is out of scope for this rewrite (PER_MEMBER_PLOT is now the
# default, working path) and needs a separate design decision — either
# extend plot_ensemble_slices() to accept these kwargs, or trim this
# call down to what it currently supports.
if PLOT_JOINT:
    if fviz is None:
        sys.exit("femtic_viz not available — cannot plot.  Check your installation.")

    print(f"\nPlotting joint ensemble figure: {len(ENS_FILES)} member(s) …")
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
    print("Joint ensemble plot done.")

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


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname)
