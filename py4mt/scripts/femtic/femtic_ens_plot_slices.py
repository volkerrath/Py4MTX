#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_plot_slices.py -- Slice figures of a FEMTIC ensemble

Reads the results .npz written by femtic_ens_post.py and draws the slice
figures (maps, N-S / E-W curtains, vertical planes):

  - "best":  the best-nRMS member (MOD_STATS_WHAT entry "best");
  - the statistics panels: mean, median, error, MAD, percentiles,
    inter-quantile spreads, bootstrap error, sensitivity and SimRC maps --
    whatever the .npz holds and MOD_STATS_WHAT asks for (a key the file does
    not hold is skipped with a message).

Nothing is recomputed and no member directory is read: all panels are drawn
from the stored vectors. Needed on disk besides the .npz: the mesh (MOD_MESH)
and one resistivity block file for the header / region structure (by default
the best member's own file recorded in the .npz; TEMPLATE_BLOCK replaces it,
e.g. when the results were moved to another machine). Slice geometry, colour
scales, vertical exaggeration, reference-model lines, region of interest,
blanking by a statistic etc. can therefore be iterated in seconds.

Companion scripts: femtic_ens_post.py (calculation), femtic_ens_plot_bh.py
(borehole profiles), femtic_data_misfit.py (quality control; method "nrms" =
nRMS histogram). Plots of single models: femtic_mod_plot_slice.py.

Provenance
----------
Author      : Claude (Claude Sonnet 5.5, Anthropic)
Generated   : 2026-11-08
Origin      : the "best" and statistics slice panels of femtic_ens_post.py
              (sections 5-8, _plot_slice, blanking and reference helpers),
              moved here unchanged in behaviour; the settings carry their old
              names and defaults. The statistic map, results reader, origin /
              site resolution, alpha helpers and reference model now live in
              femtic.py (Section 6c).
Changes     : MOD_STATS_WHAT is a plain list (the old conditionals on
              BOOTSTRAP_VAR / COMPUTE_* became "skip what the file lacks");
              MOD_STATS_STYLE lists the standard keys explicitly.
AI-generated code -- review before production use.

2026-11-09  Claude Sonnet 5.5 (Anthropic)
              (1) Titles: new MOD_TITLE (figure title of every figure:
              "auto" = as before, None / False / "" = no title, a string with
              placeholders {key} {label} {prefix} {n} {source} {best} {iter}
              {nrms}, or a dict per figure key) and MOD_PANEL_TITLES (False =
              no panel titles; single panels: title="" / title="text" in the
              MOD_SLICES entry). (2) Joint colour bar: new MOD_CBAR_LOCATION
              ("panel" = one bar per panel, as before; "right" / "left" /
              "top" / "bottom" = ONE colour bar for all panels of a figure),
              MOD_CBAR_SHRINK, MOD_CBAR_THICKNESS. Needs the femtic_viz.py of
              the same date (plot_model_slices: panel_titles, cbar_location,
              cbar_shrink, cbar_thickness; figure_title="" = no title). With
              the new settings at their defaults the figures are unchanged.
              AI-generated; review before production use.

2026-11-09  Claude Sonnet 5.5 (Anthropic)
              Plane slices. (1) The `point` of a kind="plane" entry in
              MOD_SLICES may now be CRS-tagged like x0 / y0: ([lon, lat, z],
              "latlon"), ([E, N, z], "utm") (UTM metres) or ([x, y, z],
              "model"); z is always model-local km, depth positive down;
              plain [x, y(, z)] stays model-local km. (2) Dips other than
              90 deg no longer give a sheared, mislabelled panel: strike is
              the geographic azimuth for every dip, the plane dips towards
              strike + 90, and the axes are along-strike and down-dip
              distance from `point` (see the readme). Needs femtic.py and
              femtic_viz.py of the same date. Entries with dip = 90 and all
              other kinds are unchanged.
              AI-generated; review before production use.

2026-11-10  Claude Sonnet 5.5 (Anthropic)
              MOD_MESH_SWAP_XY now defaults to True (was None). Since
              2026-10-11 femtic.read_femtic_mesh does not swap the node
              columns, but mesh.dat holds x = northing, y = easting and
              femtic_viz / the sites use x = easting, y = northing, so the
              maps, curtains and borehole positions were drawn transposed
              (a NNE structure appeared as ENE). The comment, which claimed
              that no swap gives x = easting, is corrected. None = leave
              femtic's default, False = force no swap. Checked on a
              synthetic mesh with a N30E block, not on real data.
              AI-generated; review before production use.

2026-11-11  Claude Sonnet 5.5 (Anthropic)
              Polylines. New MOD_POLYLINES = {name: entry} (km; {} = none):
              lines (faults, profiles, boundaries, ...) drawn on the slice
              panels of every figure. Maps show the polyline in plan view
              (optional z_range); vertical sections (ns, ew, plane) show
              the parts within max_dist of the section plane, projected into
              it. Points may be model-local km or CRS-tagged ("utm" metres,
              "latlon" degrees, longitude first). Passed to
              fviz.plot_model_slices(polylines=...) via
              fem.polylines_km_to_m. The intersection line of two slices
              can be computed with fviz.slice_intersection_polyline
              (comment under MOD_POLYLINES). Needs femtic.py and
              femtic_viz.py of the same date. With MOD_POLYLINES = {} the
              figures are unchanged. Tested on a synthetic mesh only.
              AI-generated; review before production use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import femtic_viz as fviz
import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# USER SECTION -- all user-set parameters below are UPPERCASE
# ===========================================================================
# ---------------------------------------------------------------------------
# Input: results of femtic_ens_post.py
# ---------------------------------------------------------------------------
#: Ensemble directory (templates/mesh.dat, templates/site.dat etc. are looked
#: up here by default).
ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/"

#: Prefix of femtic_ens_post.py's ENSEMBLE_PREFIX (including the suffix it adds
#: for ENS_MODEL_SOURCE = "perturbed" or for a random subsample). It names the
#: results file and the figures.
ENSEMBLE_PREFIX = "annecy_rto_orig-err"

#: Where femtic_ens_post.py wrote its results, and where the figures and the
#: panels' block files go (MOD_STATS_DIR below).
RESULTS_DIR = os.path.join(ENSEMBLE_DIR, "results256") + os.sep

#: The results .npz (same name rule as femtic_ens_post.py).
ENSEMBLE_RESULTS = RESULTS_DIR + ENSEMBLE_PREFIX.upper() + "_results256.npz"

#: Key prefix inside the .npz. None = detect (the file's only ensemble). Set it
#: if the file holds several ensembles.
NPZ_PREFIX = None

#: Template block file for header / region structure (None = the best member's
#: block file recorded in the .npz). Only its structure is used; all panels are
#: drawn from the stored vectors. Set it when the member directories are gone
#: or the results were moved: any resistivity block file of this ensemble (same
#: mesh, same free regions) will do, e.g. a copy of the starting model. The
#: "best" panel is then rebuilt from the stored best vector.
TEMPLATE_BLOCK = None

# ---------------------------------------------------------------------------
# Mesh (required for any slice plot)
# ---------------------------------------------------------------------------
MOD_MESH = ENSEMBLE_DIR + "templates/mesh.dat"

#: Swap of the first two node-coordinate columns when reading MOD_MESH
#: (femtic.read_femtic_mesh, which is used by every plot and by the borehole
#: block). FEMTIC stores mesh.dat nodes as x = northing, y = easting, z down;
#: the plotting code (femtic_viz, sites from site.dat, slice positions) works
#: in a geographic frame (x = easting, y = northing). True (default) swaps the
#: two columns on read, which gives that frame: maps, curtains and sites line
#: up and strikes come out with the right bearing. False draws the model
#: transposed (a N30E structure appears as N60E); use it only for meshes that
#: were already written as x = easting, y = northing. None = leave femtic's
#: module default (False) untouched.
MOD_MESH_SWAP_XY = True
if MOD_MESH_SWAP_XY is not None:
    fem.MESH_SWAP_XY = bool(MOD_MESH_SWAP_XY)

# --- Ocean / air handling (must match the inversion setup) ----------------
MOD_OCEAN = None
MOD_AIR_RHO = 1.0e9  # Ω·m  (region 0)
MOD_OCEAN_RHO = 0.25  # Ω·m  (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# Figure output format
# ---------------------------------------------------------------------------
#: One or more file extensions. Every QC / MOD_STATS figure is saved once
#: per entry (slice geometry resolved once and reused; the figure itself
#: is still rebuilt per format, since plot_model_slices doesn't expose a
#: way to re-save an already-built figure), so e.g. ["pdf", "jpg"] writes
#: both a vector version for print and a raster preview in one run. A bare
#: string ("pdf") also works and is treated as a single-entry list.
#: MOD_QC_FILE / MOD_STATS_DIR filenames below are built without an
#: extension; _plot_slice() appends each MOD_PLOT_FORMAT entry in turn.
#: Supported values (Matplotlib Agg-backend savefig formats):
#:   "pdf"  -- vector, multi-page-safe, default; best for print/publication
#:   "svg"  -- vector, editable in Illustrator/Inkscape
#:   "eps"  -- vector, legacy PostScript (no transparency support)
#:   "png"  -- raster, lossless, transparency-capable; good for slides/web
#:   "jpg" / "jpeg" -- raster, lossy, no transparency; small file size
#:   "tif" / "tiff"  -- raster, lossless, common in publication workflows
#:   "webp" -- raster, modern lossy/lossless web format
#: Raster formats (png/jpg/tif/webp) are rendered at MOD_DPI; vector
#: formats (pdf/svg/eps) are resolution-independent and MOD_DPI is ignored
#: except for any embedded raster elements (e.g. rasterized markers).
MOD_PLOT_FORMAT = ["pdf",] # "jpg"]

#: Normalised to a list regardless of whether a bare string or a list was
#: set above.
_MOD_PLOT_FORMATS = (
    [MOD_PLOT_FORMAT]
    if isinstance(MOD_PLOT_FORMAT, str)
    else list(MOD_PLOT_FORMAT)
)

# ---------------------------------------------------------------------------
# Panels to plot -- ONE list of sources (MOD_STATS_WHAT) plus ONE blanking
# option (MOD_ALPHA_SOURCE, shared slice block below).
# ---------------------------------------------------------------------------
#: Percentile pairs of the spread panels, as in femtic_ens_post.py's
#: QDIFF_PAIRS: (15.9, 84.1) -> key "qdiff_15_9_84_1" etc.
QDIFF_PAIRS = [(15.9, 84.1), (2.3, 97.7)]

#: Which panels to plot. The special entry "best" plots the best-nRMS
#: converged member (to MOD_QC_FILE). Every other entry is a statistic key:
#: subset of "avg", "var", "err", "med", "mad", one key per percentile level
#: stored in the file (e.g. 2.3 -> "p2_3", 50.0 -> "p50", 97.7 -> "p97_7"),
#: one per QDIFF_PAIRS entry (e.g. (15.9, 84.1) -> "qdiff_15_9_84_1"),
#: "err_boot" (+ "var_boot") when femtic_ens_post.py ran with BOOTSTRAP_VAR,
#: and -- when it ran with COMPUTE_SENS / COMPUTE_SIMRC -- the log10
#: sensitivity panels "sens_mean_raw", "sens_mean_na", "sens_mean_an",
#: "sens_median_na"/"_an", "sens_min_na"/"_an", "sens_max_na"/"_an",
#: "sens_std_na"/"_an", the linear ratios "sens_cv_na"/"_an", and
#: "simrc_coef"/"simrc_corr" (log10). A key requested here but not held by the
#: results file is skipped with a printed message, never an error.
#: "err" = sqrt(var) is plotted by default instead of "var", since var is in
#: (log10 Ohm.m)^2 and not on the same scale as MAD/QDIFF (log10 Ohm.m).
#: Colour map, colour limits and colourbar label of every statistic panel
#: are set per key in MOD_STATS_STYLE (shared slice block below); "best"
#: always plots at MOD_CLIM/MOD_CMAP directly (no MOD_STATS_STYLE entry).
MOD_STATS_WHAT = (
      ["best", "avg", "med", "err", "mad"]
#    + ["p" + f"{_p:g}".replace(".", "_") for _p in (2.3, 15.9, 50.0, 84.1, 97.7)]
    + [f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_") for _lo, _hi in QDIFF_PAIRS]
    + ["err_boot"]
    + [
        "sens_mean_raw",
        "sens_mean_an",
        "sens_median_an",
        "sens_cv_an",
    ]
    + ["simrc_corr"]
    + ["simrc_coef"]
)

#: Extension-less base path for the "best" panel; _plot_slice() appends
#: ".<fmt>" per MOD_PLOT_FORMAT entry. None = MOD_STATS_DIR/<prefix>_best.
MOD_QC_FILE = None

#: Output directory for the panels' block files and figures.
MOD_STATS_DIR = RESULTS_DIR   #+ "stats_plots/"

#: Derived from MOD_STATS_WHAT above -- not user-set. Remove "best" from
#: the list (or set MOD_STATS_WHAT = []) to turn either off, instead of a
#: separate boolean.
MOD_QC = "best" in MOD_STATS_WHAT
MOD_STATS = bool(set(MOD_STATS_WHAT) - {"best"})

#: ROI diagnostic for the log10 sensitivity panels ("sens_mean_raw",
#: "sens_{mean,median,min,max,std}_{na,an}" -- NOT "sens_cv_na"/"_an",
#: which are linear ratios, and NOT "simrc_coef"/"simrc_corr", left
#: untouched pending more experience with that method).
#:
#: Regardless of SENS_ROI_NORMALIZE below, every such panel that is both
#: listed in MOD_STATS_WHAT and actually plotted has its *original*
#: (whole-mesh-normalised, as currently stored/plotted) min/max printed
#: to the console, restricted to the free parameters inside the ROI (the
#: same box driving the MOD_STATS slice plots: MOD_XLIM/MOD_YLIM/MOD_ZLIM,
#: including any MOD_ROI_AUTO override) -- a quick read of the value range
#: actually in view, to help pick a MOD_ALPHA_SOURCE threshold.
#:
#: SENS_ROI_NORMALIZE (default False) additionally *re-centres* each such
#: panel, right before it is written/plotted, so that the ROI's own max
#: becomes log10 = 0 instead of the whole-mesh max (equivalent to dividing
#: the underlying linear quantity by its max within the ROI rather than
#: over the whole mesh). This is a display-only transform: it never
#: touches the .npz output (sens_mean_an etc. are saved exactly as before)
#: or MOD_ALPHA_SOURCE (which still reads the untransformed _stat_map
#: values). Cells outside the ROI can come out > 0 (log10) once
#: renormalised this way -- expected, since they may exceed the ROI's max.
SENS_ROI_NORMALIZE = True

# ---------------------------------------------------------------------------
# Shared slice / plot parameters
# (identical config surface to femtic_gst_prep.py / femtic_rto_prep.py /
#  femtic_mod_plot_slice.py — used by both MOD_QC and MOD_STATS below)
# ---------------------------------------------------------------------------

# --- Geographic / UTM origin of the mesh centre ----------------------------
#: Set to None when MOD_ORIGIN_METHOD will estimate the origin from MOD_SITE_DAT.
MOD_UTM_ORIGIN_LAT = None  # decimal degrees, positive = North
MOD_UTM_ORIGIN_LON = None  # decimal degrees, positive = East
MOD_UTM_ORIGIN_E = None  # UTM easting  [m]
MOD_UTM_ORIGIN_N = None  # UTM northing [m]
MOD_UTM_ZONE_OVERRIDE = None  # override auto-derived zone; None = auto

#: "box"     → midpoint of UTM bounding box of all sites in MOD_SITE_DAT.
#: "average" → arithmetic mean of UTM coordinates in MOD_SITE_DAT.
#: None      → use the hard-coded literals above.
MOD_ORIGIN_METHOD = "box"

# --- Display coordinate system ---------------------------------------------
#: "model"  — axis ticks in model-local metres (default)
#: "utm"    — axis ticks in absolute UTM metres
#: "latlon" — axis ticks in decimal degrees
MOD_DISPLAY_COORDS = "latlon"

# --- Site overlay ------------------------------------------------------------
#: Primary source: mt_make_sitelist.py CSV (name,lat,lon,elev,sitenum,E,N).
#: Set to None to fall back to observe.dat / MOD_SITE_NUMBER.
MOD_SITE_DAT = ENSEMBLE_DIR + "templates/site.dat"
MOD_SITE_NAMES = None  # list of names to plot, or None = all sites
#: Fallback: site number(s) from observe.dat (int or list of ints).
MOD_SITE_NUMBER = None

MOD_PLOT_SITES_MAPS = True  # show markers on map panels
MOD_PLOT_SITES_SLICES = False  # show markers on curtain / plane panels
#: Max distance [km] from a curtain plane for a site to appear on it.
MOD_PROJECTION_DIST = 1.0  # km; None = show all sites on every panel

MOD_SITE_MARKER = dict(marker="v", color="black", ms=8, zorder=10, label=None)
MOD_SITE_MARKER_SLICES = None
#: Extra point markers on map panels only (each dict: latlon, marker, color, ms, name).
MOD_MAP_MARKERS = []
#: Mark the model origin on every "map" panel whenever MOD_DISPLAY_COORDS
#: is "utm"/"latlon" (no effect for "model"). Override style via a
#: MOD_MAP_MARKERS entry with "is_model_centre": True.
MOD_SHOW_MODEL_CENTRE = True

# --- Slice specification ----------------------------------------------------
#: Slice positions accept plain floats (model-local m) or CRS-tagged tuples:
#:   (value, "utm") | (value, "latlon")
#: Depth z0 is always model-local metres (no CRS tagging).
# MOD_SLICES = [
#     dict(kind="map", z0=0.0),    # km
#     dict(kind="map", z0=5.0),    # km
#     dict(kind="map", z0=10.0),   # km
#     dict(kind="map", z0=15.0),   # km
#     dict(kind="map", z0=20.0),   # km
#     dict(kind="map", z0=25.0),   # km
#     dict(kind="ns",  x0=(-71.40723, 'latlon')),    # km
#     dict(kind="ew",  y0=(-16.299593, 'latlon')),    # km
# ]
MOD_SLICES = [
    # dict(kind="map", z0=-0.25),  # km
    dict(kind="map", z0=0.0),  # km
    # dict(kind="map", z0=0.5),  # km
    dict(kind="map", z0=1.0),  # km
    dict(kind="map", z0=2.0),  # km
    # dict(kind="map", z0=2.5),  # km
    dict(kind="map", z0=3.0),  # km
    dict(kind="map", z0=4.0),  # km
    dict(kind="map", z0=5.0),  # km
    dict(kind="plane", point=[0.0, 0.0], strike=45.0, dip=90.0),
    dict(kind="ns", x0=0.0),  # km
    dict(kind="ew", y0=0.0),  # km
    #: Vertical section along a strike through ``point`` (model-local km,
    #: [0, 0] = model origin). strike = geographic azimuth, degrees
    #: clockwise from north: 45 = SW (left) -> NE (right), 135 = NW -> SE.
    #: Horizontal axis = along-strike distance from ``point`` in model km;
    #: its range follows MOD_XLIM x MOD_YLIM (projected onto the strike),
    #: its depth range MOD_ZLIM. Optional: xlim=[min, max] (km) to override.
    # dict(kind="plane", point=[0.0, 0.0], strike=45.0, dip=90.0),
    #: Dipping plane (dip < 90): strike = azimuth (deg clockwise from north)
    #: of the strike line, the plane dips towards strike + 90. ``point`` may
    #: be model-local km or CRS-tagged (z = depth in km, positive down):
    #:   point=[x, y, z]                       model-local km
    #:   point=([x, y, z], "model")            model-local km
    #:   point=([E, N, z], "utm")              UTM metres, z in km
    #:   point=([lon, lat, z], "latlon")       degrees (longitude first)
    #: Axes = along-strike and down-dip distance from ``point`` (km if
    #: MOD_DEPTH_KM else m, true aspect, see the readme). Default extent = the plane cut
    #: with the MOD_XLIM x MOD_YLIM x MOD_ZLIM box; optional per-entry
    #: xlim=[min, max] (along strike) and ylim=[min, max] (down dip), km
    #: relative to ``point``.
    # dict(kind="plane", point=([6.12, 45.90, 0.0], "latlon"), strike=325.0, dip=70.0),
    # dict(kind="plane", point=([E_utm, N_utm, 2.0], "utm"), strike=325.0, dip=45.0),
    # dict(kind="ns",  x0=(-71.40723, 'latlon')),    # km
    # dict(kind="ew",  y0=(-16.299593, 'latlon')),    # km
]
#: Epagny (45.9383, 6.0883) 
# --- Polylines (optional) -----------------------------------------------------
#: Lines drawn on the slice panels of every figure ("best" and MOD_STATS):
#: faults, profiles, borehole traces, licence boundaries, ... A dict
#: {name: entry}; {} = none. An entry is the point list itself or a dict:
#:   points    [[x, y], ...] or [[x, y, z], ...], at least 2 rows, model-local
#:             km; z = depth positive down (missing z = 0). CRS-tagged like
#:             the `point` of a plane: (pts, "model" | "utm" | "latlon") with
#:             "utm" = [E, N(, z)] UTM metres, "latlon" = [lon, lat(, z)]
#:             degrees (longitude first); z is km in every case
#:   style     matplotlib Line2D keywords, e.g. dict(color="k", linestyle="--",
#:             linewidth=1.5); default 1.5 pt solid, colour cycled per entry
#:   label     legend text (lower right of the panel); default = the key,
#:             None = no legend entry
#:   kinds     where to draw it: "map", "ns", "ew", "plane", "vertical"
#:             (= ns + ew + plane) or a list of them; default = everywhere
#:   max_dist  km; vertical sections only: the parts of the polyline within
#:             max_dist of the section plane are projected into it. Default
#:             MOD_PROJECTION_DIST (the whole polyline if that is None);
#:             0 = only parts lying in the section plane
#:   z_range   [zmin, zmax] km; maps only: draw only on maps whose z0 lies in
#:             the range (default: plan view on every map)
#: Maps always show the whole polyline in plan view. The axis limits are
#: never changed by the polylines.
#: Intersection line of two slices (metres in, metres out; ready-made
#: entries for many pairs: fviz.slice_intersection_polylines):
#:   _p = fviz.slice_intersection_polyline(
#:       dict(kind="map", z0=2000.0), dict(kind="ns", x0=0.0),
#:       xlim=[-20e3, 20e3], ylim=[-20e3, 20e3], zlim=[-1e3, 10e3])
#:   MOD_POLYLINES["map 2 km x ns 0"] = dict(
#:       points=_p / 1000.0, max_dist=0.0, z_range=[2.0, 2.0])
MOD_POLYLINES = {}
# MOD_POLYLINES = {
#     "fault F1": dict(
#         points=[[-4.0, -3.0, 0.5], [0.0, 0.0, 2.5], [4.0, 3.0, 4.5]],
#         style=dict(color="white", linewidth=2.0), max_dist=1.5),
#     "profile": dict(points=([[6.05, 45.90], [6.15, 45.95]], "latlon"),
#                     kinds="map", style=dict(color="k", linestyle=":")),
# }
# --- Reference model (optional) -----------------------------------------------
#: A reference model to compare the ensemble with: a resistivity block file
#: (resistivity_block_iterX.dat) on the SAME mesh as MOD_MESH -- e.g. a
#: deterministic inversion, the prior/starting model or a published model.
#: Its region structure may differ from the members'. None = no reference
#: (everything below is then ignored). A missing file, or one with another
#: element count than MOD_MESH, switches the reference off with a message.
# MOD_REF_FILE = None  # e.g. ENSEMBLE_DIR + "reference/resistivity_block_iter20.dat"
MOD_REF_FILE =ENSEMBLE_DIR + "/templates/referencemodel.dat" 
#: Slices: draw the reference as iso-resistivity lines on every slice panel
#: (the "best" panel and all MOD_STATS panels). The lines are the exact
#: boundaries between slice cells lying on different sides of a level; air
#: and ocean cells of the reference are ignored.
MOD_REF_SLICES = False
#: Contour levels in log10(Ohm.m), e.g. [1.0, 2.0, 3.0]. None = the integer
#: decades inside MOD_CLIM (the reference model's own range if MOD_CLIM is
#: None). Always in the reference model's units, whatever quantity a
#: MOD_STATS panel shows.
MOD_REF_LEVELS = None
#: LineCollection keywords of the slice lines, e.g.
#: dict(colors="k", linewidths=0.8, linestyles="--").
MOD_REF_STYLE = dict(colors="k", linewidths=0.8)
#: Legend entry of the slice lines (lower right of each panel); None = none.
MOD_REF_LABEL = "reference model"

MOD_XLIM = None  # [xmin, xmax] model-local km; None = auto
MOD_YLIM = None  # [ymin, ymax] model-local km; None = auto
MOD_ZLIM = None  # [zmin, zmax] model-local km; None = auto

# --- Region of interest (auto xlim/ylim/zlim from site positions) ----------
#: When True and site positions are available (MOD_SITE_DAT / MOD_SITE_NUMBER,
#: subject to MOD_PLOT_SITES_MAPS/SLICES as usual), MOD_XLIM/MOD_YLIM are
#: derived automatically from the site bounding box + MOD_ROI_PAD_XY, and
#: MOD_ZLIM is set from MOD_ROI_ZLIM -- overriding the literal MOD_XLIM/
#: MOD_YLIM/MOD_ZLIM values above. Falls back to those literals (or to
#: full-mesh auto-scaling if they're also None) when no sites are found.
#: Also drives the per-panel aspect-ratio sizing below (MOD_PANEL_WIDTH),
#: since that sizing needs an actual extent to compute widths from.
#: The same three MOD_ROI_* variables (same defaults, same semantics) now
#: exist in femtic_rto_prep.py and femtic_gst_prep.py, so the plot extent
#: is identical between prep-time and post-time figures.
MOD_ROI_AUTO = True
MOD_ROI_PAD_XY = 4.0  # km of padding around the site bbox
MOD_ROI_ZLIM = [-1.0, 5.0]
#: depth range (km, positive-down) for ns/ew/plane panels; None = leave MOD_ZLIM as-is
#: Lower bound is negative (above the z=0 datum) to give ~1 km of headroom
#: so topography (mesh cells with z < 0) is not clipped out of the ns/ew/
#: plane panels. Previously [0.0, 20000.0] cut panels off exactly at the
#: datum, hiding any topography above it.

MOD_DPI = 400  # figure DPI, used by both MOD_QC and MOD_STATS
MOD_CMAP = "jet_r"
MOD_CLIM = [0.0, 4.0]  # [log10_min, log10_max] Ω·m; None = auto
MOD_OCEAN_COLOR = "lightgrey"  # flat colour for ocean cells; None = colormap
MOD_AIR_COLOR = "whitesmoke"
MOD_AIR_BGCOLOR = None  # axes facecolor for air; None = figure default

# --- Per-panel style (MOD_STATS) ------------------------------------------
#: Per-statistic overrides for every MOD_STATS panel, keyed as in
#: MOD_STATS_WHAT. Each value is a dict with any of
#:   "cmap"  : Matplotlib colormap name                (default MOD_CMAP)
#:   "clim"  : [vmin, vmax] in the panel's own units,
#:             None = auto-scale from the data         (default see below)
#:   "label" : colourbar label                         (default see below)
#: Keys/fields not given fall back to the defaults: value-scale panels
#: (avg, med, p*) use MOD_CMAP / MOD_CLIM / "log10(rho / Ohm*m)"; every
#: other panel uses MOD_CMAP / auto clim / the statistic's own name.
#: Sensitivity panels are log10 (see the Sensitivity statistics section),
#: so e.g. clim=[-4, 0] shows four decades below the normalised maximum.
MOD_STATS_STYLE = {
    "var": dict(clim=[0.0, 1.0], label="var log10(rho)"),
    "err": dict(clim=[0.0, 1.0], label="std log10(rho)"),
    "mad": dict(clim=[0.0, 1.0], label="MAD log10(rho)"),
}
for _lo, _hi in QDIFF_PAIRS:
    MOD_STATS_STYLE[f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_")] = dict(
        clim=[0.0, 1.0], label=f"P{_hi:g} - P{_lo:g}  log10(rho)"
    )
MOD_STATS_STYLE["var_boot"] = dict(
    clim=[0.0, 1.0], label="bootstrap var log10(rho)"
)
MOD_STATS_STYLE["err_boot"] = dict(
    clim=[0.0, 1.0], label="bootstrap std log10(rho)"
)
#: Native sensitivity units (femtic_ens_post.py SENS_KIND) -- run- and
#: mesh-dependent, auto-scaled.
MOD_STATS_STYLE["sens_mean_raw"] = dict(
    cmap="viridis", clim=None, label="log10 S (native units)"
)
for _sfx in ("na", "an"):
    for _st in ("mean", "median", "min", "max", "std"):
        MOD_STATS_STYLE[f"sens_{_st}_{_sfx}"] = dict(
            cmap="viridis",
            clim=[-5.0, 0.0],
            label=f"log10 S/Smax ({_st}, {_sfx})",
        )
    MOD_STATS_STYLE[f"sens_cv_{_sfx}"] = dict(
        cmap="viridis", clim=None, label=f"CV of S ({_sfx})"
    )
MOD_STATS_STYLE["simrc_coef"] = dict(
    cmap="viridis", clim=None, label="log10 sum|SimRC|"
)
MOD_STATS_STYLE["simrc_corr"] = dict(
    cmap="viridis", clim=None, label="log10 sum|corr|"
)
# --- Blanking / fading (optional) ------------------------------------------
#: One condition "<name> <op> <value>" selecting the cells to SHOW; cells
#: failing it are blanked or faded. <name> is any statistic key known to
#: MOD_STATS (see MOD_STATS_WHAT; computed statistics are available here
#: even if not listed for plotting), <op> is one of > >= < <=, and <value>
#: is in that statistic's own units (log10 for sensitivities). Examples:
#:   "sens_mean_an > -3."    keep cells within 3 decades of max sensitivity
#:   "err < 0.2"             keep cells with ensemble std below 0.2 decades
#: None disables blanking/fading.
# MOD_ALPHA_SOURCE = "sens_median_an > -5.5"
# #: "blank" -- failing cells are removed (hard cut).
# #: "fade"  -- failing cells fade linearly from opaque at the threshold to
# #:            fully transparent MOD_ALPHA_FADE_WIDTH beyond it (same units
# #:            as <value>, e.g. 1.0 = one decade for log10 sensitivities).
# MOD_ALPHA_MODE = "blank"
MOD_ALPHA_SOURCE = "sens_median_an > -3."
#: "blank" -- failing cells are removed (hard cut).
#: "fade"  -- failing cells fade linearly from opaque at the threshold to
#:            fully transparent MOD_ALPHA_FADE_WIDTH beyond it (same units
#:            as <value>, e.g. 1.0 = one decade for log10 sensitivities).
MOD_ALPHA_MODE = "fade"
MOD_ALPHA_FADE_WIDTH = 1.
#: Apply the same mask to the MOD_QC (best-member) figure as well.
MOD_ALPHA_QC = True

#: Statistic panels whose key starts with one of these prefixes are drawn
#: WITHOUT the mask. Default: the sensitivity panels ("sens_*", "simrc_*") --
#: shading a sensitivity map by sensitivity is circular, and the unmasked map
#: is what shows where the MOD_ALPHA_SOURCE threshold cuts. All other panels
#: (avg, err, ..., and the best panel via MOD_ALPHA_QC) are masked as before.
#: () = mask every panel (the behaviour of earlier versions).
MOD_ALPHA_EXCLUDE = ("sens_", "simrc_")

# --- Figure layout -----------------------------------------------------------
MOD_EQUAL_ASPECT = True
#: Vertical exaggeration of the section panels (ns, ew and vertical "plane").
#: None = as set by MOD_EQUAL_ASPECT (unchanged). A number fixes the aspect of
#: every section from the real axis units: 1.0 = true equal axes, 2.0 = the
#: vertical scale twice the horizontal one. Works for every
#: MOD_DISPLAY_COORDS, also "latlon", where MOD_EQUAL_ASPECT alone does not
#: act (its test needs model/utm display); map panels stay 1:1 while
#: MOD_EQUAL_ASPECT is True.
MOD_VERT_EXAG = None
MOD_DEPTH_KM = True
MOD_HORIZ_KM = True
#: 2x2 grid matching the 4 default MOD_SLICES panels (2 maps + ns + ew).
#: Adjust to len(MOD_SLICES) if you change the number of panels; None/None
#: falls back to a single row of len(MOD_SLICES) columns.
MOD_NROWS = 3  # None = auto (1 row)
MOD_NCOLS = 3  # None = auto (len(MOD_SLICES) cols)
MOD_PANEL_HEIGHT = 20.0  # cm
#: None = auto per-column width from each panel's own aspect ratio (needs
#: MOD_EQUAL_ASPECT=True and real xlim/ylim/zlim -- supplied automatically
#: by MOD_ROI_AUTO above -- so map, ns, and ew panels naturally end up
#: different widths instead of being forced square).
MOD_PANEL_WIDTH = None  # cm; None = auto from aspect ratio
MOD_FIGSIZE = None  # [w, h] cm; overrides auto when set

#: Axis annotation font sizes, passed through to fviz.plot_model_slices.
#: Defaults match plot_model_slices' own defaults.
MOD_TICK_FONTSIZE = 16  # axis tick labels, colourbar ticks
MOD_LABEL_FONTSIZE = 16  # axis labels, panel titles, colourbar label
#: Font size of the figure title (the block-file name above the panels);
#: None = MOD_LABEL_FONTSIZE + 6.
MOD_TITLE_FONTSIZE = MOD_LABEL_FONTSIZE + 10

# --- Titles -----------------------------------------------------------------
#: Figure title (above all panels) of every figure of this script.
#:   "auto"            as before: the "best" figure shows the block-file name,
#:                     the best member, its iteration and nRMS; every
#:                     statistic figure shows its block-file name.
#:   None, False or "" no figure title.
#:   a string          used as given. Placeholders, filled in per figure:
#:                     {key} panel key of MOD_STATS_WHAT ("best", "avg", ...),
#:                     {label} description of the statistic, {prefix}
#:                     ENSEMBLE_PREFIX, {n} number of members, {source}
#:                     "converged" / "perturbed", {best} name of the best
#:                     member, {iter} its iteration, {nrms} its nRMS (a
#:                     number: {nrms:.3f} works). Text with other braces is
#:                     used as it stands.
#:   a dict            one entry per figure, e.g.
#:                       {"best": "Best member", "avg": "Mean", "err": None,
#:                        "default": "auto"}
#:                     keys = entries of MOD_STATS_WHAT; values as above;
#:                     "default" (else "auto") covers the keys not listed.
MOD_TITLE = "auto"
#: Panel titles ("Map  z = 1.0 km", "N-S  easting = 0.0 km", ...): True = as
#: before, False = none. A single panel is titled by adding title="text" to its
#: MOD_SLICES entry (title="" leaves just that panel untitled).
MOD_PANEL_TITLES = True

# --- Colour bar ----------------------------------------------------------------
#: "panel"  one colour bar next to every panel (as before);
#: "right", "bottom", "top", "left"
#:          ONE colour bar for all panels of a figure (they share one colour
#:          scale), at that side of the whole panel grid. The panels give up a
#:          strip of the figure for it; their spacing is kept.
MOD_CBAR_LOCATION = "panel"
#: Joint colour bar only: its length as a fraction of the side of the panel
#: grid it runs along (0 < value <= 1), and its thickness in cm (None =
#: 1.5 x MOD_LABEL_FONTSIZE points).
MOD_CBAR_SHRINK = 0.7
MOD_CBAR_THICKNESS = None  # cm

#: Decimal digits shown on axis tick labels (depth, map/curtain
#: easting-northing, and lat/lon all share this one setting). None (default)
#: keeps plot_model_slices' own per-axis-type formatting unchanged.
MOD_TICK_DECIMALS = 2

#: When True (default) and this script is running inside Spyder, every
#: saved figure is also displayed inline (Spyder's Plots pane) via
#: plt.show(), in addition to being written to disk -- no change to what
#: gets saved. Has no effect outside Spyder (plain python / other IDEs
#: still save-only, matching prior behaviour); set False to disable even
#: under Spyder.
MOD_SHOW_IN_SPYDER = True

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

#: Detected once at import time; utl.runtime_env() returns 'spyder' when
#: running inside Spyder's IPython console (SPYDER_KERNEL env var / spyder_
#: kernels module), 'jupyter'/'ipython-*'/'python' otherwise.
_IN_SPYDER = utl.runtime_env() == "spyder"
_SHOW_PLOTS = MOD_SHOW_IN_SPYDER and _IN_SPYDER
if _IN_SPYDER:
    print(
        f"Detected Spyder — inline figure display {'enabled' if _SHOW_PLOTS else 'disabled (MOD_SHOW_IN_SPYDER=False)'}.\n"
    )


# ===========================================================================
# Helpers
# ===========================================================================

#: The optional reference model (MOD_REF_FILE): checked once, switched off with
#: a message if the file is missing or lives on another mesh.
_REF = fem.ReferenceModel(MOD_REF_FILE, ocean=MOD_OCEAN, ocean_rho=MOD_OCEAN_RHO)


def _plot_slice(
    block_file: str,
    pdf_file: str,
    utm_e,
    utm_n,
    utm_lat,
    utm_lon,
    utm_zone,
    utm_north,
    site_xys: list,
    obs_coords_only: bool = False,
    clim="model",
    cmap=None,
    cbar_label=None,
    alpha_file=None,
    stats_panel: bool = False,
    figure_title=None,
) -> None:
    """Call fviz.plot_model_slices once per MOD_PLOT_FORMAT entry.

    Mirrors the plotting call in femtic_gst_prep.py / femtic_rto_prep.py,
    so QC and statistics figures share the MOD_SLICES geometry, CRS
    handling, site overlay and figure layout.

    Parameters
    ----------
    pdf_file : str
        Output path *without* an extension; ".<fmt>" is appended for each
        MOD_PLOT_FORMAT entry.
    clim : "model" | [vmin, vmax] | None
        "model" (default) = MOD_CLIM; None = auto-scale from the data;
        a pair = explicit range (from MOD_STATS_STYLE). Previously None
        silently fell back to MOD_CLIM, so auto-scaling never happened.
    cmap, cbar_label : optional
        Per-panel overrides (from MOD_STATS_STYLE); None falls back to
        MOD_CMAP / plot_model_slices' default label.
    alpha_file : str | None
        Alpha block written by fem.write_alpha_block() (raw values = alpha in
        [0, 1]); plotted with alpha_mode="direct". None = no blanking.
    stats_panel : bool
        True for MOD_STATS panels. Their block files use sentinel
        resistivities for air/ocean (fem.STATS_AIR_RHO / fem.STATS_OCEAN_RHO) so
        that a statistic value can never be mistaken for air (log10 > 8)
        or ocean (log10 ~ log10(MOD_OCEAN_RHO)) by plot_model_slices.
    figure_title : str | None
        Figure title passed to plot_model_slices; None = block-file name,
        "" = no title (see MOD_TITLE and _title_for()).

    Module settings used besides the MOD_* layout ones: MOD_REF_FILE /
    MOD_REF_SLICES / MOD_REF_LEVELS / MOD_REF_STYLE / MOD_REF_LABEL (reference
    iso-lines on every panel, stats panels included; the reference is read
    from its own block file and always contoured in its own resistivity) and
    MOD_VERT_EXAG (vertical exaggeration of the section panels).
    """
    _ref_kw = (
        _REF.slice_kwargs(
            block_file,
            levels=MOD_REF_LEVELS,
            clim=MOD_CLIM,
            style=MOD_REF_STYLE,
            label=MOD_REF_LABEL,
        )
        if MOD_REF_SLICES
        else {}
    )
    _clim = MOD_CLIM if (isinstance(clim, str) and clim == "model") else clim
    _cmap = MOD_CMAP if cmap is None else cmap
    _ocean_value = fem.STATS_OCEAN_RHO if stats_panel else MOD_OCEAN_RHO
    _air_thresh = fem.STATS_AIR_LOG10_THRESH if stats_panel else 8.0

    # "plane" entries are converted here (km -> m; `point` may be CRS-tagged,
    # see fem.plane_point_km_to_m); all other kinds go through
    # fem.resolve_slice_positions
    # exactly as before. Original MOD_SLICES order is preserved.
    _others = [sp for sp in MOD_SLICES if sp.get("kind") != "plane"]
    _others_res = iter(
        fem.resolve_slice_positions(
            fem.slices_km_to_m(_others),
            utm_zone,
            utm_north,
            utm_e,
            utm_n,
            utm_lat,
            utm_lon,
            verbose=OUT,
        )
        if _others
        else []
    )
    _slices_resolved = [
        fem.plane_spec_km_to_m(
            sp,
            MOD_XLIM,
            MOD_YLIM,
            zone=utm_zone,
            northern=utm_north,
            origin_E=utm_e,
            origin_N=utm_n,
        )
        if sp.get("kind") == "plane"
        else next(_others_res)
        for sp in MOD_SLICES
    ]
    for _fmt_i, _fmt in enumerate(_MOD_PLOT_FORMATS):
        _fmt_file = f"{pdf_file}.{_fmt}"
        # Only pop up the interactive window (if MOD_SHOW_IN_SPYDER) on the
        # first format; re-displaying the same figure once per extra
        # format would just spam the Plots pane for no benefit. Note the
        # slice geometry (_slices_resolved, above) is computed once and
        # reused, but plot_model_slices still builds and renders its own
        # figure fresh on each call -- it doesn't expose a way to re-save
        # an already-built figure under a second extension.
        _show_this = _SHOW_PLOTS if _fmt_i == 0 else False
        fviz.plot_model_slices(
            model_file=block_file,
            mesh_file=MOD_MESH,
            slices=_slices_resolved,
            cmap=_cmap,
            clim=_clim,
            cbar_label=cbar_label,
            xlim=fem.lim_km_to_m(MOD_XLIM),
            ylim=fem.lim_km_to_m(MOD_YLIM),
            zlim=fem.lim_km_to_m(MOD_ZLIM),
            ocean_color=MOD_OCEAN_COLOR,
            ocean_value=_ocean_value,
            air_log10_thresh=_air_thresh,
            air_color=MOD_AIR_COLOR,
            air_bgcolor=MOD_AIR_BGCOLOR,
            site_xys=site_xys,
            obs_coords_only=obs_coords_only,
            sites_in_maps=MOD_PLOT_SITES_MAPS,
            sites_in_slices=MOD_PLOT_SITES_SLICES,
            site_marker=MOD_SITE_MARKER,
            site_marker_slices=MOD_SITE_MARKER_SLICES,
            map_markers=MOD_MAP_MARKERS,
            show_model_centre=MOD_SHOW_MODEL_CENTRE,
            polylines=fem.polylines_km_to_m(MOD_POLYLINES),
            projection_dist=fem.km_to_m(MOD_PROJECTION_DIST),
            display_coords=MOD_DISPLAY_COORDS,
            utm_origin_e=utm_e,
            utm_origin_n=utm_n,
            utm_zone=utm_zone,
            utm_northern=utm_north,
            utm_to_latlon_fn=utl.utm_to_latlon_zn,
            latlon_to_model_fn=fem.latlon_to_model,
            depth_km=MOD_DEPTH_KM,
            horiz_km=MOD_HORIZ_KM,
            equal_aspect=MOD_EQUAL_ASPECT,
            vert_exag=MOD_VERT_EXAG,
            panel_height=MOD_PANEL_HEIGHT / 2.54,
            panel_width=(
                MOD_PANEL_WIDTH / 2.54 if MOD_PANEL_WIDTH is not None else None
            ),
            figsize=(
                [v / 2.54 for v in MOD_FIGSIZE]
                if MOD_FIGSIZE is not None
                else None
            ),
            nrows=MOD_NROWS,
            ncols=MOD_NCOLS,
            tick_fontsize=MOD_TICK_FONTSIZE,
            label_fontsize=MOD_LABEL_FONTSIZE,
            figure_title_fontsize=MOD_TITLE_FONTSIZE,
            figure_title=figure_title,
            panel_titles=bool(MOD_PANEL_TITLES),
            cbar_location=MOD_CBAR_LOCATION,
            cbar_shrink=MOD_CBAR_SHRINK,
            cbar_thickness=(
                MOD_CBAR_THICKNESS / 2.54
                if MOD_CBAR_THICKNESS is not None
                else None
            ),
            tick_decimals=MOD_TICK_DECIMALS,
            alpha_file=alpha_file,
            alpha_mode="direct",
            **_ref_kw,
            plot_file=_fmt_file,
            dpi=MOD_DPI,
            show=_show_this,
            out=OUT,
        )
        if OUT:
            print(f"  saved -> {_fmt_file}")

# ===========================================================================
# Main
# ===========================================================================
if not (MOD_QC or MOD_STATS):
    sys.exit("MOD_STATS_WHAT is empty - nothing to plot.")

# --- (1) Read the results written by femtic_ens_post.py ---------------------
try:
    res = fem.read_ensemble_results(ENSEMBLE_RESULTS, NPZ_PREFIX, out=OUT)
    tpl, _from_vector = fem.resolve_results_template(
        res, TEMPLATE_BLOCK, ocean=MOD_OCEAN
    )
except (FileNotFoundError, KeyError, ValueError) as _exc:
    sys.exit(f"femtic_ens_plot_slices.py: {_exc}")

P = res["prefix"]
ens_matrix = res["ens"]
model_list = res["model_list"]  # [[block file, iteration, nRMS], ...]
_ENS_SRC = res["model_source"]  # "converged" | "perturbed"
_best_i = res["best_index"]
_best_file, _best_iter, _best_nrms = model_list[_best_i]
_best_member = res["best_member"]

#: key -> (vector over the free parameters, description); and the keys that are
#: on the resistivity scale (avg, med, percentiles).
stat_map, _value_scale_keys = fem.ensemble_stat_map(res)
if MOD_QC_FILE is None:
    MOD_QC_FILE = os.path.join(MOD_STATS_DIR, P + "_best")
if _from_vector:
    print(f"\n  template block file: {tpl}  (not the best member's own file)")

# --- (2) Resolve UTM origin and sites ----------------------------------------
(
    utm_e,
    utm_n,
    utm_lat,
    utm_lon,
    utm_zone,
    utm_north,
    site_xys,
    obs_coords_only,
) = fem.resolve_origin_and_sites(
    site_dat=MOD_SITE_DAT,
    site_names=MOD_SITE_NAMES,
    site_number=MOD_SITE_NUMBER,
    observe_file=ENSEMBLE_DIR + "templates/observe.dat",
    origin_e=MOD_UTM_ORIGIN_E,
    origin_n=MOD_UTM_ORIGIN_N,
    origin_lat=MOD_UTM_ORIGIN_LAT,
    origin_lon=MOD_UTM_ORIGIN_LON,
    zone_override=MOD_UTM_ZONE_OVERRIDE,
    origin_method=MOD_ORIGIN_METHOD,
    plot_sites=MOD_PLOT_SITES_MAPS or MOD_PLOT_SITES_SLICES,
)

# --- Region of interest: override MOD_XLIM/YLIM/ZLIM from site bbox ---------
if MOD_ROI_AUTO and site_xys:
    _sx = np.array([s[1] for s in site_xys])  # model-local metres
    _sy = np.array([s[2] for s in site_xys])  # model-local metres
    MOD_XLIM = [
        float(_sx.min() / 1000.0 - MOD_ROI_PAD_XY),
        float(_sx.max() / 1000.0 + MOD_ROI_PAD_XY),
    ]
    MOD_YLIM = [
        float(_sy.min() / 1000.0 - MOD_ROI_PAD_XY),
        float(_sy.max() / 1000.0 + MOD_ROI_PAD_XY),
    ]
    if MOD_ROI_ZLIM is not None:
        MOD_ZLIM = list(MOD_ROI_ZLIM)
    print(f"\nROI (from {len(site_xys)} sites, pad={MOD_ROI_PAD_XY:.2f} km):")
    print(f"  MOD_XLIM = {MOD_XLIM} km")
    print(f"  MOD_YLIM = {MOD_YLIM} km")
    print(f"  MOD_ZLIM = {MOD_ZLIM} km")
elif MOD_ROI_AUTO:
    print(
        "\nROI: MOD_ROI_AUTO=True but no sites available - "
        "using literal MOD_XLIM/MOD_YLIM/MOD_ZLIM instead."
    )

# --- (3) Panel style and blanking/fading mask ---------------------------------


def _panel_style(key):
    """(cmap, clim, label) for a MOD_STATS panel: MOD_STATS_STYLE over defaults."""
    st = MOD_STATS_STYLE.get(key, {})
    if key in _value_scale_keys:
        d_clim, d_label = MOD_CLIM, "log10(rho / Ohm*m)"
    else:
        d_clim, d_label = None, key
    return (
        st.get("cmap", MOD_CMAP),
        st.get("clim", d_clim),
        st.get("label", d_label),
    )


def _title_for(key, auto, label=""):
    """Figure title of the figure `key`, from MOD_TITLE.

    Returns `auto` for "auto" (None = file name of the block file), "" for
    no title, otherwise the text with its placeholders filled in (the
    function is fviz.resolve_title, shared by the plot scripts).
    """
    return fviz.resolve_title(
        MOD_TITLE, key, auto, name="MOD_TITLE",
        label=label,
        prefix=P,
        n=len(model_list),
        source=_ENS_SRC,
        best=_best_member,
        iter=_best_iter,
        nrms=_best_nrms,
    )


_alpha_block = None
if MOD_ALPHA_SOURCE:
    _a_name, _a_op, _a_thr = fem.parse_alpha_source(MOD_ALPHA_SOURCE)
    if _a_name not in stat_map:
        print(
            f"\n  MOD_ALPHA_SOURCE: '{_a_name}' not available - not stored in "
            f"the results file (femtic_ens_post.py did not compute it). "
            f"Available: {sorted(stat_map)}. Blanking disabled."
        )
    else:
        _alpha = fem.alpha_from_condition(
            stat_map[_a_name][0],
            _a_op,
            _a_thr,
            MOD_ALPHA_MODE,
            MOD_ALPHA_FADE_WIDTH,
        )
        os.makedirs(MOD_STATS_DIR, exist_ok=True)
        _alpha_block = os.path.join(
            MOD_STATS_DIR, f"resistivity_block_{P}_alpha.dat"
        )
        fem.write_alpha_block(
            _alpha, tpl, _alpha_block, ocean=MOD_OCEAN, out=OUT
        )
        print(
            f"\n  MOD_ALPHA_SOURCE '{MOD_ALPHA_SOURCE}' "
            f"(mode='{MOD_ALPHA_MODE}'): "
            f"{int(np.sum(_alpha >= 1.0))} shown, "
            f"{int(np.sum((_alpha > 0.0) & (_alpha < 1.0)))} faded, "
            f"{int(np.sum(_alpha <= 0.0))} blanked "
            f"of {_alpha.size} cells."
        )

# --- (4) QC slice plot -- best-nRMS member ------------------------------------
if MOD_QC:
    _best_name = os.path.basename(_best_file)
    print(
        f"\nQC: best member  {_best_member}  nRMS={_best_nrms:.4f}  "
        f"iter={_best_iter}"
    )
    _qc_block, _qc_stats = _best_file, False
    if _from_vector:
        # The template is not the best member's own file, so draw the stored
        # best vector into it (same route and sentinels as the statistic
        # panels).
        os.makedirs(MOD_STATS_DIR, exist_ok=True)
        _qc_block = os.path.join(MOD_STATS_DIR, f"resistivity_block_{P}_best.dat")
        fem.insert_model(
            template=tpl,
            model=ens_matrix[_best_i],
            model_file=_qc_block,
            ocean=MOD_OCEAN,
            air_rho=fem.STATS_AIR_RHO,
            ocean_rho=fem.STATS_OCEAN_RHO,
            out=OUT,
        )
        _qc_stats = True
    _plot_slice(
        block_file=_qc_block,
        stats_panel=_qc_stats,
        pdf_file=MOD_QC_FILE,
        utm_e=utm_e,
        utm_n=utm_n,
        utm_lat=utm_lat,
        utm_lon=utm_lon,
        utm_zone=utm_zone,
        utm_north=utm_north,
        site_xys=site_xys,
        obs_coords_only=obs_coords_only,
        alpha_file=_alpha_block if MOD_ALPHA_QC else None,
        figure_title=_title_for(
            "best",
            f"{_best_name}"
            + (" (perturbed reference)" if _ENS_SRC == "perturbed" else "")
            + "\n"
            f"best member: {_best_member}  iter {_best_iter}  "
            f"nRMS {_best_nrms:.3f}",
            "best member",
        ),
    )

# --- (5) Statistics slice plots -----------------------------------------------
if MOD_STATS:
    os.makedirs(MOD_STATS_DIR, exist_ok=True)

    for _key in MOD_STATS_WHAT:
        if _key == "best":
            continue  # handled by the QC block above
        if _key not in stat_map:
            print(
                f"  MOD_STATS: '{_key}' not in the results file - skipped. "
                f"Available: {sorted(stat_map)}"
            )
            continue
        _vec, _label = stat_map[_key]
        _roi_normalised = False

        # --- ROI diagnostic / optional ROI-max renormalisation ------------
        # Print the original (whole-mesh-normalised, as stored) ROI min/max
        # for every sens_*/simrc_* panel actually plotted -- a quick read of
        # the value range in view, to help pick a MOD_ALPHA_SOURCE
        # threshold. SENS_ROI_NORMALIZE additionally re-centres the panel to
        # the ROI's own max, but only for sens_* (not sens_cv_*, a linear
        # ratio; not simrc_coef/_corr, left as-is pending more experience
        # with that method).
        _is_sens = _key.startswith("sens_") and not _key.startswith("sens_cv")
        _is_simrc = _key.startswith("simrc_")
        if _is_sens or _is_simrc:
            _roi_mask = fem.roi_free_mask(
                tpl,
                _vec.shape[0],
                MOD_MESH,
                xlim_km=MOD_XLIM,
                ylim_km=MOD_YLIM,
                zlim_km=MOD_ZLIM,
                ocean=MOD_OCEAN,
            )
            if _roi_mask is not None and np.any(_roi_mask):
                _roi_vals = _vec[_roi_mask]
                _roi_vals = _roi_vals[np.isfinite(_roi_vals)]
                if _roi_vals.size:
                    print(
                        f"  {_key}: original ROI min={_roi_vals.min():.3f}, "
                        f"max={_roi_vals.max():.3f} (log10, whole-mesh-"
                        f"normalised, n={_roi_vals.size} free params)"
                    )
                    if _is_sens and SENS_ROI_NORMALIZE:
                        _roi_max = _roi_vals.max()
                        _vec = _vec - _roi_max
                        _label = f"{_label}, ROI-renormalised"
                        _roi_normalised = True
                else:
                    print(
                        f"  {_key}: no finite values inside the ROI - "
                        f"skipping ROI diagnostic for this panel."
                    )

        _cmap, _clim, _cbl = _panel_style(_key)
        if _roi_normalised:
            _cbl = (
                f"{_cbl} [ROI-renorm.]" if _cbl else "ROI-renormalised (log10)"
            )
            # ROI renormalisation can push values above the whole-mesh-
            # normalised default clim (e.g. [-4, 0]) since cells outside the
            # ROI may now exceed the ROI's own max -- auto-scale unless the
            # user explicitly overrode clim for this key in MOD_STATS_STYLE.
            if _key not in MOD_STATS_STYLE or "clim" not in MOD_STATS_STYLE[_key]:
                _clim = None
        _block_out = os.path.join(MOD_STATS_DIR, f"resistivity_block_{P}_{_key}.dat")
        _pdf_out = os.path.join(MOD_STATS_DIR, f"{P}_{_key}")
        print(f"\nSTATS: writing {_label} -> {_block_out}")
        # insert_model writes 10**v; plot_model_slices log10s it back, so the
        # panel shows _vec itself (NaN -> 0 raw -> not drawn).
        fem.insert_model(
            template=tpl,
            model=np.where(np.isfinite(_vec), _vec, -np.inf),
            model_file=_block_out,
            ocean=MOD_OCEAN,
            air_rho=fem.STATS_AIR_RHO,
            ocean_rho=fem.STATS_OCEAN_RHO,
            out=OUT,
        )
        print(
            f"STATS: plotting {_label} -> {_pdf_out}  "
            f"(cmap={_cmap}, clim={_clim})"
        )
        _plot_slice(
            block_file=_block_out,
            pdf_file=_pdf_out,
            utm_e=utm_e,
            utm_n=utm_n,
            utm_lat=utm_lat,
            utm_lon=utm_lon,
            utm_zone=utm_zone,
            utm_north=utm_north,
            site_xys=site_xys,
            obs_coords_only=obs_coords_only,
            clim=_clim,
            cmap=_cmap,
            cbar_label=_cbl,
            alpha_file=(
                None
                if _key.startswith(tuple(MOD_ALPHA_EXCLUDE))
                else _alpha_block
            ),
            stats_panel=True,
            figure_title=_title_for(_key, None, _label),
        )

print("\nfemtic_ens_plot_slices.py complete.")

# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
_summary_path = utl.write_param_summary(__file__, out_dir=ENSEMBLE_DIR)
