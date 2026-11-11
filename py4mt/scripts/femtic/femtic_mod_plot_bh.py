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

Family: the "mod" scripts draw SINGLE models; the "ens" scripts
(femtic_ens_post, femtic_ens_plot_slices, femtic_ens_plot_bh and the quality-
control script femtic_data_misfit) work on the ensemble. For the ensemble
view of the same boreholes (heat map, percentiles, all members) use
femtic_ens_plot_bh.py.

One model or several (batch)
----------------------------
MODEL_FILE is, as before, ONE resistivity block file; BOREHOLE_FILE is its
figure. It can also be a list of block files, a glob pattern, or run
directories (a directory, a list, or a glob such as "ens/ens_*"); for run
directories MODEL_ITER chooses the block file ("last" = iteration of the last
row of femtic.cnv, "best" = lowest nRMS in femtic.cnv, N =
resistivity_block_iterN.dat, 0 = starting model / perturbed prior, or a list
such as [0, "last"]). With more than one model there is one figure (and one
NPZ) per model, named <label>_<tag>_boreholes.<ext> in BATCH_OUT_DIR (None =
the model's own directory), plus an optional multi-page PDF (BATCH_CATALOG).
BOREHOLE_FILE is not used then.

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
@modified: 2026-08-13  Claude Sonnet 5 (Anthropic) -- Added
    femtic_mod_plot_bh_summary.md output at end of run: writes user-set
    (UPPERCASE) parameters, script path, and run date/time via
    utl.write_param_summary().
@modified: 2026-10-28  Claude Sonnet 5.5 (Anthropic) -- The parameter
    summary femtic_mod_plot_bh_summary.md is now written to WORK_DIR
    instead of next to the script (utl.write_param_summary(fname,
    out_dir=WORK_DIR)). Needs the util.py of the same date. AI-generated;
    review before production use.
@modified: 2026-11-08  Claude Sonnet 5.5 (Anthropic) -- Several models in one
    run: MODEL_FILE may be a list, a glob or run directories (MODEL_ITER
    selects the iteration), giving one figure per model plus an optional
    catalog PDF (BATCH_* settings; femtic.resolve_model_files,
    fviz.append_catalog_page). One model behaves exactly as before.
    fviz.plot_borehole_logs now returns its figure. AI-generated; review
    before production use.
@modified: 2026-11-10  Claude Sonnet 5.5 (Anthropic) -- New setting
    MESH_SWAP_XY (default True): sets fem.MESH_SWAP_XY before the mesh is
    read. mesh.dat holds x = northing, y = easting, but the borehole
    positions (site.dat, UTM -> model) are in a geographic frame
    (x = easting, y = northing); without the swap (the default of
    femtic.py since 2026-10-11) the mesh is transposed relative to the
    positions and the logs are taken at the wrong places. None leaves
    femtic's default, False forces no swap. AI-generated; review before
    production use.
@modified: 2026-11-11  Claude Sonnet 5.5 (Anthropic) -- Titles in the form of
    all plot scripts. New FIGURE_TITLE ("auto" = the block-file name, as
    before; None / False / "" = no title; text with {placeholders}; dict per
    model) for the single-model figure and, when BATCH_TITLE is unset, for
    the batch figures; BATCH_TITLE now also takes "auto", False and a dict.
    New PANEL_TITLES (False = no panel titles) and FIGURE_TITLE_FONTSIZE.
    Defaults give the same figures as before. Needs the femtic_viz.py of the
    same date. AI-generated; review before production use.
@modified: 2026-11-11b  Claude Sonnet 5.5 (Anthropic) -- New setting CNV_FILE
    (one name or a list of candidate names of the convergence file, default
    ["femtic.cnv", "FEMTIC-DABIC.cnv"]; the first one that exists is used),
    passed to fem.resolve_model_files() as cnv_name; it matters for
    MODEL_ITER = "last" / "best" with run directories. Needs the femtic.py of
    the same date. AI-generated; review before production use.
"""

import os
import sys
from pathlib import Path
import gc
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

#: Resistivity block to sample (any iteration).  Also a list of block files,
#: a glob pattern, or run directories for a batch (see MODEL_ITER below).
MODEL_FILE = WORK_DIR + "resistivity_block_iter17.dat"

#: For run directories in MODEL_FILE: which block file of each run.
#:   "last" - iteration of the last row of femtic.cnv (best-fit model of a
#:            converged run)
#:   "best" - iteration with the lowest nRMS in femtic.cnv
#:   N      - resistivity_block_iterN.dat (0 = starting model / perturbed prior)
#:   a list - e.g. [0, "last"]: one figure per entry and run
#: Does not apply to block files given as files.
MODEL_ITER = "last"

#: Name of the FEMTIC convergence file inside each run directory (used for
#: MODEL_ITER = "last" / "best"): one name or a list of candidate names (the
#: first one that exists is used; names are also matched ignoring upper /
#: lower case). Same setting as in all femtic_* scripts.
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

#: With more than one model: one figure and NPZ per model, named
#: <label>_<tag>_boreholes.<ext> (label = run directory, tag = "iterN").
#: Directory for them; None = the model's own directory.
BATCH_OUT_DIR = None
#: Figure formats (without dot), e.g. ("pdf", "png"); None = the extension of
#: BOREHOLE_FILE ("pdf" if that is None).
BATCH_FORMATS = None
#: Multi-page PDF with one page per model: None = none; "auto" =
#: <BATCH_OUT_DIR or the common parent directory>/catalog_boreholes.pdf; or a
#: path. Needs "pdf" among the formats.
BATCH_CATALOG = None
#: Figure title template; {label}, {iter}, {tag}, {nrms} are filled in.
#: Same forms as FIGURE_TITLE below ("auto", False, a dict ...); None or ""
#: = use FIGURE_TITLE instead.
BATCH_TITLE = "{label} - iteration {iter}"

#: Mesh file — always required.
MESH_FILE = WORK_DIR + "mesh.dat"


#: x/y column swap applied when the mesh is read (femtic.read_femtic_mesh).
#: FEMTIC stores mesh.dat nodes as x = northing, y = easting, z down; the
#: plotting code (femtic_viz, sites from site.dat, slice positions) works in
#: a geographic frame (x = easting, y = northing). True (default) swaps the
#: two columns on read, which gives that frame: maps, curtains and sites then
#: line up, and strikes come out with the right bearing. False draws the
#: model transposed (a N30E structure appears as N60E, x/y exchanged); it is
#: only for meshes that were already written as x = easting, y = northing.
#: None = leave femtic's module default (False) untouched.
MESH_SWAP_XY = True
if MESH_SWAP_XY is not None:
    fem.MESH_SWAP_XY = bool(MESH_SWAP_XY)

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
#:   "depth"      : float — arrow tip depth [m, z-down]; REQUIRED.
#:   "rho"        : float — arrow tip x-position [Ohm·m]; optional,
#:                          defaults to left x-axis edge.
#:   "rho_text"   : float — text x-position [Ohm·m]; optional,
#:                          defaults to rho * 3 (≈ half a decade right).
#:   "depth_text" : float — text y-position [m, z-down]; optional,
#:                          defaults to depth - 300 m (slightly above tip).
#:   Both text positions are in natural axis display units (Ohm·m / metres)
#:   so you can read the values directly off the plot axes when choosing them.
#:   "text"       : str   — annotation label (optional; default "").
#:   "borehole"   : str or list of str — borehole name(s) to annotate
#:                  (optional; None = all panels).
#:   "arrowprops" : dict  — forwarded to ax.annotate (optional; default
#:                          thin black -> arrow).
#:   Any additional keys forwarded verbatim to ax.annotate
#:   (e.g. "color", "fontsize", "fontweight", "ha", "va", "zorder").
#:
#: Set to [] or None to place no markers.
BOREHOLE_MARKERS = [
    dict(depth=-4000., rho=40., text="conductor",
         rho_text=100., depth_text=-4300.,
         borehole="borehole crater",
         color="red", fontsize=12, fontweight="bold",
         arrowprops=dict(arrowstyle="->", color="red", lw=2)),
    # dict(depth=3200., rho_text=500., depth_text=2900.,
    #      text="resistive basement", color="navy", fontsize=8),
]

#: Figure title (above all panels), one model (several: see BATCH_TITLE).
#:   "auto" or True     the block-file name (as before);
#:   None, False or ""  no title;
#:   a string           used as given. Placeholders: {label} run directory name,
#:                      {tag} "iterN", {iter} the iteration ("?" if unknown),
#:                      {nrms} nRMS (3 decimals; "n/a" if unknown), {file} the
#:                      block-file name, {key} = {label}. Text with other
#:                      braces is used as it stands;
#:   a dict             one entry per model, keyed by its label, e.g.
#:                        {"run1": "Reference", "default": "auto"}
#:                      values as above; "default" (else "auto") covers the rest.
FIGURE_TITLE = "auto"
#: Font size of the figure title; None = LEGEND_FONTSIZE + 1.
FIGURE_TITLE_FONTSIZE = None
#: Panel titles ("Borehole resistivity logs" in the shared panel, borehole name
#: and position above every panel otherwise): True = as before, False = none.
PANEL_TITLES = True

#: Font size for the borehole legend (shared mode) and panel titles
#: (per-panel mode).  Tick labels are set to legend_fontsize - 2.
LEGEND_FONTSIZE = 12
#: Font size for axis tick labels.  None → LEGEND_FONTSIZE - 2 (≥ 6).
TICK_FONTSIZE = None
#: Font size for axis labels (xlabel / ylabel).  None → LEGEND_FONTSIZE.
LABEL_FONTSIZE = None


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

# --- (0) Resolve the model(s) to plot --------------------------------------
MODELS = fem.resolve_model_files(MODEL_FILE, MODEL_ITER,
                                 cnv_name=CNV_FILE, out=OUT)
if not MODELS:
    sys.exit(f"No model found for MODEL_FILE = {MODEL_FILE!r} "
             f"(MODEL_ITER = {MODEL_ITER!r}).")
BATCH = len(MODELS) > 1
print(f"{len(MODELS)} model(s) to plot"
      + (": one figure per model (batch)" if BATCH else ""))
print()

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

# Options shared by every figure of this run.
_common = dict(
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
    panel_titles   = bool(PANEL_TITLES),
    figure_title_fontsize = FIGURE_TITLE_FONTSIZE,
    tick_fontsize  = TICK_FONTSIZE,
    label_fontsize = LABEL_FONTSIZE,
    dpi            = PLOT_DPI,
    out            = OUT,
)

def _model_title(m, spec):
    """Figure title of model ``m`` (an entry of MODELS) from ``spec``
    (FIGURE_TITLE / BATCH_TITLE): None = the block-file name, "" = none,
    else the text."""
    return fviz.resolve_title(
        spec, m["label"], None, name="FIGURE_TITLE",
        label=m["label"], tag=m["tag"],
        iter="?" if m["iter"] is None else m["iter"],
        nrms="n/a" if m["nrms"] is None else f"{m['nrms']:.3f}",
        file=os.path.basename(str(m["file"])))


if not BOREHOLE_SITES:
    print("  Borehole plot skipped: BOREHOLE_SITES is empty.")
elif not BATCH:
    print(f"Sampling {len(BOREHOLE_SITES)} borehole(s) …")
    fviz.plot_borehole_logs(
        model_file     = MODELS[0]["file"],
        figure_title   = _model_title(MODELS[0], FIGURE_TITLE),
        npz_file       = (None   if BOREHOLE_NPZ is True
                          else (False if BOREHOLE_NPZ is False
                                else BOREHOLE_NPZ)),
        plot_file      = BOREHOLE_FILE,
        **_common,
    )
    print("Borehole plot done.")
else:
    # --- several models: one figure (and NPZ) per model + optional catalog --
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    _fmts = BATCH_FORMATS if BATCH_FORMATS is not None else (
        (os.path.splitext(BOREHOLE_FILE)[1].lstrip(".") if BOREHOLE_FILE else "")
        or "pdf",)
    _fmts = [str(_f).lstrip(".") for _f in
             ((_fmts,) if isinstance(_fmts, str) else _fmts)]

    _cat_path = None
    if BATCH_CATALOG:
        if "pdf" not in _fmts:
            print("  catalog: \"pdf\" is not among the formats - skipped.")
        elif BATCH_CATALOG == "auto":
            _cat_path = os.path.join(
                BATCH_OUT_DIR or os.path.commonpath([_m["dir"] for _m in MODELS]),
                "catalog_boreholes.pdf")
        else:
            _cat_path = str(BATCH_CATALOG)
    if BATCH_OUT_DIR:
        os.makedirs(BATCH_OUT_DIR, exist_ok=True)
    _pdf = None
    if _cat_path:
        os.makedirs(os.path.dirname(os.path.abspath(_cat_path)), exist_ok=True)
        _pdf = PdfPages(_cat_path)
    _n_pages = 0
    print(f"Sampling {len(BOREHOLE_SITES)} borehole(s) in {len(MODELS)} model(s) ...")
    try:
        for _i, _m in enumerate(MODELS, 1):
            _title = _model_title(
                _m, BATCH_TITLE if BATCH_TITLE not in (None, "")
                else FIGURE_TITLE)
            _stem = os.path.join(BATCH_OUT_DIR or _m["dir"],
                                 f"{_m['label']}_{_m['tag']}_boreholes")
            print(f"[{_i}/{len(MODELS)}] {_m['label']} / {_m['tag']}")
            for _k, _fmt in enumerate(_fmts):
                # the NPZ holds the same data for every format: write it once
                if BOREHOLE_NPZ is False or _k > 0:
                    _npz = False
                elif BOREHOLE_NPZ is True:
                    _npz = None            # derived from the figure file
                else:
                    _npz = (os.path.splitext(str(BOREHOLE_NPZ))[0]
                            + f"_{_m['label']}_{_m['tag']}.npz")
                _fig = fviz.plot_borehole_logs(
                    model_file   = _m["file"],
                    figure_title = _title,
                    npz_file     = _npz,
                    plot_file    = f"{_stem}.{_fmt}",
                    **_common,
                )
                if _fig is None:
                    continue
                if _fmt == "pdf" and _pdf is not None:
                    _n_pages += bool(fviz.append_catalog_page(
                        _pdf, _fig, _m["label"]))
                plt.close(_fig)
                del _fig
            gc.collect()   # Matplotlib figures hold reference cycles
    finally:
        if _pdf is not None:
            _info = _pdf.infodict()
            _info["Title"] = "femtic_mod_plot_bh catalog"
            _info["Author"] = "femtic_mod_plot_bh.py"
            _pdf.close()
    print(f"Borehole plots done: {len(MODELS)} model(s)")
    if _cat_path:
        print(f"  catalog: {_n_pages} page(s) -> {_cat_path}")


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname, out_dir=WORK_DIR)
