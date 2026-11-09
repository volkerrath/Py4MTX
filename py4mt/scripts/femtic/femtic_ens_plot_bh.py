#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_plot_bh.py -- Borehole profiles of a FEMTIC ensemble

Reads the results .npz written by femtic_ens_post.py and draws resistivity-
depth profiles at vertical "boreholes" (MOD_BH): the ensemble statistics
(mean / median / percentiles / sigma pairs), all members ("ensemble" lines), a
density heat map of the members ("heatmap"), the best member and an optional
reference model. If femtic_ens_post.py ran with ENS_MODEL_SOURCE = "both", the
perturbed models of the same members are drawn next to the converged ones on
one common colour scale.

Nothing is recomputed: the element search is done once per borehole on the
mesh (MOD_MESH), the values are looked up in the stored ensemble. Needed on
disk besides the .npz: the mesh and one resistivity block file for the region
structure (by default the best member's own file recorded in the .npz;
TEMPLATE_BLOCK replaces it).

Outputs (MOD_BH_FILE, extension-less): the figures (one per MOD_BH_PER_FIG
boreholes), the profiles as .npz (MOD_BH_NPZ; "_perturbed.npz" for the
perturbed models) and, if MOD_BH_HEAT_SCALE_FILE is set, the heat-map colour
scale as JSON.

Companion scripts: femtic_ens_post.py (calculation), femtic_ens_plot_slices.py
(slice figures), femtic_data_misfit.py (quality control; method "nrms" = nRMS
histogram). Plots of single models: femtic_mod_plot_bh.py.

Provenance
----------
Author      : Claude (Claude Sonnet 5.5, Anthropic)
Generated   : 2026-11-08
Origin      : the borehole block (MOD_BH) of femtic_ens_post.py, section 9 and
              its _bh_* helpers. The sampling now lives in femtic.py
              (sample_boreholes, BoreholeMesh, heat_scale_*), the drawing in
              femtic_viz.py (plan_borehole_figures, draw_borehole_figures,
              plot_boreholes); this script holds the settings and the glue.
Changes     : the profile bottom defaults to MOD_ZLIM (below) and no longer to
              the region of interest derived from the sites in the old
              combined script; entries that need another depth set "z_bot".
AI-generated code -- review before production use.

2026-11-09  Claude Sonnet 5.5 (Anthropic)
              New MOD_TITLE: figure title of the borehole figures ("auto" = as
              before, None / False / "" = no title, a string with the
              placeholders {prefix} and {dir}). The panel titles (borehole name
              and position) are not affected. Needs the femtic_viz.py of the
              same date (BoreholeFigureOptions.suptitle = None: no title). With
              the default the figures are unchanged. AI-generated; review before
              production use.
"""

from __future__ import annotations

import json
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

#: Where femtic_ens_post.py wrote its results; the figures go here too.
RESULTS_DIR = os.path.join(ENSEMBLE_DIR, "results256") + os.sep

#: The results .npz (same name rule as femtic_ens_post.py).
ENSEMBLE_RESULTS = RESULTS_DIR + ENSEMBLE_PREFIX.upper() + "_results256.npz"

#: Key prefix inside the .npz. None = detect (the file's only ensemble).
NPZ_PREFIX = None

#: Template block file for the region structure (None = the best member's
#: block file recorded in the .npz). Set it when the member directories are
#: gone or the results were moved: any resistivity block file of this ensemble
#: (same mesh, same free regions) will do.
TEMPLATE_BLOCK = None

# ---------------------------------------------------------------------------
# Mesh (required for any slice plot)
# ---------------------------------------------------------------------------
MOD_MESH = ENSEMBLE_DIR + "templates/mesh.dat"

#: Optional swap of the first two node-coordinate columns when reading
#: MOD_MESH (femtic.read_femtic_mesh, which is used by every plot and by the
#: borehole block). None = leave femtic's default (no swap: coordinates as
#: stored in mesh.dat, with x = easting, y = northing); True = swap, for
#: meshes stored as x = northing, y = easting; False = force no swap.
MOD_MESH_SWAP_XY = None
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

# --- Reference model (optional) -----------------------------------------------
#: A reference model to compare the ensemble with: a resistivity block file
#: (resistivity_block_iterX.dat) on the SAME mesh as MOD_MESH -- e.g. a
#: deterministic inversion, the prior/starting model or a published model.
#: Its region structure may differ from the members'. None = no reference
#: (everything below is then ignored). A missing file, or one with another
#: element count than MOD_MESH, switches the reference off with a message.
# MOD_REF_FILE = None  # e.g. ENSEMBLE_DIR + "reference/resistivity_block_iter20.dat"
MOD_REF_FILE =ENSEMBLE_DIR + "/templates/referencemodel.dat" 
#: Boreholes: add the reference profile as a dashed line "reference model"
#: to every MOD_BH panel. A MOD_BH entry can switch it off with ref=False, or
#: list "ref" in its "what" to force it on and to set its drawing position.
#: Style: MOD_BH_STYLE["ref"].
MOD_REF_BH = True
#: Legend entry of the reference line (drawn in every panel); None = none.
MOD_REF_LABEL = "reference model"

# ---------------------------------------------------------------------------
# Figure parameters (same names and meaning as in femtic_ens_plot_slices.py)
# ---------------------------------------------------------------------------
#: Default bottom of a profile ("z_bot" of a MOD_BH entry), [zmin, zmax] km,
#: only zmax is used. None = 20 km.
MOD_ZLIM = None

#: [log10_min, log10_max] Ohm.m: resistivity range of the profile panels and of
#: the heat-map bins unless MOD_BH_XLIM / an entry's "xlim" says otherwise;
#: None = from the data.
MOD_CLIM = [0.0, 4.0]

MOD_DPI = 400  # figure DPI
MOD_DEPTH_KM = True  # depth axis in km (False: m)
#: Axis annotation font sizes and the figure title size.
MOD_TICK_FONTSIZE = 16
MOD_LABEL_FONTSIZE = 16
MOD_TITLE_FONTSIZE = MOD_LABEL_FONTSIZE + 10
#: Figure title (above all panels).
#:   "auto"            as before: "<ensemble directory name>  <ENSEMBLE_PREFIX>"
#:   None, False or "" no figure title.
#:   a string          used as given; {prefix} = ENSEMBLE_PREFIX, {dir} = name
#:                     of ENSEMBLE_DIR. Text with other braces is used as it
#:                     stands.
#: When the figure shows the converged and the perturbed models together, the
#: tag "(converged | perturbed reference)" is still appended to a title (not to
#: "no title"). The panel titles (borehole name, position) are not affected.
MOD_TITLE = "auto"

#: Name of the perturbed reference file inside each member directory (only
#: written into the header of the perturbed-profile file; the results file
#: stores the name too, and that one wins).
ENS_PERTURBED_FILE = "referencemodel.dat"

# --- Borehole profiles (optional) --------------------------------------------
#: 1-D resistivity-depth profiles through the ensemble at vertical "boreholes".
#: One panel per entry, all panels in one figure (MOD_BH_FILE). An empty list
#: ([]) switches the block off. Entries are dicts, in the same spirit as
#: MOD_SLICES (plain horizontal positions are model-local km, depths km):
#:
#:   "name"  : str    -- panel title / key in the .npz file         (required)
#:   "x","y" : horizontal position. Plain float = model-local km (origin =
#:             mesh centre); (value, "utm") = UTM metres in the mesh zone;
#:             (value, "latlon") = decimal degrees (x = longitude,
#:             y = latitude). x and y must carry the same tag. (required,
#:             unless "sites" is given)
#:   "sites" : instead of "x"/"y": "all" (every site in MOD_SITE_DAT) or a
#:             list of site names. The entry is then a template: it is
#:             expanded to one borehole per site, named after the site, with
#:             all other keys ("z_top", "z_bot", "dz", "what", "alpha",
#:             "xlim", "style") applied to each. Needs MOD_SITE_DAT.
#:   "n_random", "seed" : with "sites": use only N sites drawn at random
#:             (without replacement) from the selected ones; "seed" = int
#:             makes the draw reproducible, None (default) draws afresh
#:             every run. The chosen names are printed.
#:   "z_top" : float km (z-down, model datum) or "surface" = top of the first
#:             non-air element at (x, y) (sea level offshore). Default "surface".
#:   "z_bot" : float km, bottom of the profile. Default: MOD_ZLIM[1] if set,
#:             else 20 km.
#:   "dz"    : float km, sampling interval. Default 0.05 (50 m).
#:   "what"  : list of what to draw (order = drawing order). Any of
#:               "avg"          ensemble mean of log10(rho) (geometric mean)
#:               "med"          ensemble median
#:               "percentiles"  one line per level in PERCENTILES
#:               "p50", "p15_9" ... single percentile(s), keys as in
#:                              MOD_STATS_WHAT
#:               "sigma1"       the 1-sigma quantile pair, P15.87 and P84.13 of
#:                              the members (log10 rho), two lines
#:               "sigma2"       the 2-sigma pair, P2.28 and P97.72
#:               "ensemble"     every accepted member, thin lines drawn with
#:                              opacity "alpha"
#:               "best"         the best-nRMS member
#:               "heatmap"      density heat map of the ensemble: at every
#:                              depth level the members' log10(rho) are
#:                              histogrammed (MOD_BH_HEAT_BINS bins across
#:                              the panel's resistivity range) and shown as
#:                              a colour image behind all other items; one
#:                              colour bar per figure. See MOD_BH_HEAT_*.
#:                              Combine freely with "ensemble" (lines on
#:                              top of it), "med", "sigma1", ... Put it
#:                              anywhere in "what" -- it is always drawn at
#:                              the bottom.
#:               "ref"          the reference model (MOD_REF_FILE); added
#:                              automatically (drawn last, on top) when
#:                              MOD_REF_BH is True; list it here to choose its
#:                              position, or to force it on when MOD_REF_BH is
#:                              False
#:             Default ["avg", "percentiles"].
#:   "ref"   : False = no reference line in this panel although MOD_REF_BH is
#:             True. Default True.
#:   "alpha" : opacity of the "ensemble" lines, 0..1. Default MOD_BH_ALPHA.
#:   "xlim"  : [rho_min, rho_max] in Ohm.m for this panel (default MOD_BH_XLIM).
#:   "heat_bins" : bins of the "heatmap" for this entry: an int (number of
#:             equal bins in log10(rho) across the panel's xlim) or an
#:             increasing list of log10(rho) bin edges. Default
#:             MOD_BH_HEAT_BINS.
#:   "style" : dict of per-entry style overrides, keyed like MOD_BH_STYLE.
#: Air and out-of-mesh levels are left as gaps.
MOD_BH = [
    # dict(
    #     name="model centre",
    #     x=0.0,
    #     y=0.0,  # km, model-local
    #     z_top="surface",
    #     z_bot=4.0,
    #     dz=0.05,  # km
    #     what=["ensemble", "med", "sigma1". "sigma2",],  # ensemble first = underneath
    #     alpha=0.3,
    # ),
    # dict(sites="all", n_random=6, seed=1,   # 6 random sites, reproducible
    #      z_bot=5.0, what=["ensemble", "med", "sigma1"], alpha=0.05),
    #     what=["ensemble", "med", "sigma1". "sigma2",],  # ensemble first = underneath
    #     alpha=0.3,
    dict(sites="all", z_top="surface", z_bot=4.0, dz=0.05,   # one panel per site
             what=["heatmap", "med", "sigma1", "sigma2",],  # ensemble first = underneath
             alpha=0.4, xlim=[1., 10000.],),
    # dict(sites=["SITE01", "SITE07"], z_bot=5.0, what=["med", "best"]),
    dict(name="SV-108",
         x=(6.053423 , "latlon"), y=(45.92574, "latlon"),
         z_top="surface", z_bot=2., dz=0.05,
         what=["heatmap", "med", "sigma1", "sigma2", ],
         alpha=0.4, xlim=[1., 10000.],), 
    # 108: 6.05342229365832 	45.9257397106266
    # 109: 6.05340119257481 	45.9404667527012
    dict(name="SV-109",
         x=(6.05340119, "latlon"), y=(45.940467, "latlon"),
         z_top="surface", z_bot=2., dz=0.05,
         what=["heatmap", "med", "sigma1", "sigm2", ],
         alpha=0.4, xlim=[1., 10000.],),
    # dict(name="BH-X",
    #      x=(229100.0, "utm"), y=(8184000.0, "utm"),
    #      z_top=0.0, z_bot=6.0, dz=0.1,
    #      what=["avg", "ensemble"], alpha=0.08,
    #      style=dict(avg=dict(color="navy", lw=3))),
]

#: Output path (no extension) of the borehole figure; one file per
#: MOD_PLOT_FORMAT entry is written.
MOD_BH_FILE = RESULTS_DIR + ENSEMBLE_PREFIX + "_boreholes"
#: Export the sampled profiles (depth, statistics, all members) to an .npz
#: next to MOD_BH_FILE (<MOD_BH_FILE>.npz). False = skip.
MOD_BH_NPZ = True
#: Default opacity of the "ensemble" lines (per-entry "alpha" overrides).
MOD_BH_ALPHA = 0.15
#: x-axis (resistivity) range in Ohm.m, log scale. None = 10**MOD_CLIM
#: (shared with the slice figures); if MOD_CLIM is None too, auto.
MOD_BH_XLIM = None
#: Panel size [width, height] in cm and number of panel columns
#: (None = up to 4 columns). MOD_BH_PER_FIG = maximum panels per figure
#: (None = all in one); further figures get a _01, _02, ... suffix. Useful
#: with "sites".
MOD_BH_PANEL_SIZE = [16.0, 20.0]
MOD_BH_NCOLS = 2
MOD_BH_PER_FIG = 10
#: True = all panels use one common depth range (union of the entries'
#: ranges), so profiles can be compared by eye.
MOD_BH_SHARE_DEPTH = True
#: Matplotlib legend location per panel; None = no legend.
MOD_BH_LEGEND_LOC = "best"
#: Heat map ("heatmap" in a MOD_BH entry's "what"): number of bins in
#: log10(rho) across the panel's resistivity range (the entry's "xlim", else
#: MOD_BH_XLIM, else 10**MOD_CLIM; if all are None, the range of the
#: members' values at that borehole). May also be a list of log10(rho) bin
#: edges. Members outside the range are not shown (e.g. ocean cells at
#: 0.25 Ohm.m with xlim [1, 1e4]); the fractions still refer to all valid
#: members, so such a level sums to less than 1. A per-entry "heat_bins"
#: overrides this.
MOD_BH_HEAT_BINS = 40
#: Colour scale of the heat map, one scale (and one colour bar) per figure:
#:   "fraction" : fraction of the valid members falling into a bin (a depth
#:                level sums to 1 if all members are inside the range) --
#:                comparable between depths, boreholes and ensembles of
#:                different size.
#:   "row_max"  : every depth level scaled to its own maximum (1 = most
#:                populated bin) -- shows the shape of the distribution
#:                even where it is broad (small fractions).
#: The .npz always holds the fractions; "row_max" is a display option only.
MOD_BH_HEAT_NORM = "fraction"
#: Colormap of the heat map (any Matplotlib name; light = few members). A
#: single-hue light-to-dark map ("Greys", "Blues", "Purples", ...) keeps the
#: lines on top readable. MOD_BH_STYLE["heatmap"] or an entry's
#: style=dict(heatmap=dict(cmap=...)) overrides it; the colour bar uses the
#: cmap of the first panel that draws a heat map.
MOD_BH_HEAT_CMAP = "Greys"
#: Shape of the colour scale between MOD_BH_HEAT_VMIN and MOD_BH_HEAT_VMAX:
#:   "linear" : colour proportional to the value (default; unchanged output)
#:   "sqrt"   : square-root scale -- lifts the low densities, keeps the
#:              peaks distinguishable
#:   "log"    : logarithmic scale, over the decades between VMIN and VMAX
#:              (default VMIN = smallest non-zero value of the figure, i.e.
#:              one member); empty bins (0) are left blank.
MOD_BH_HEAT_SCALE = "linear"
#: Lower end of the colour scale. None = 0 ("linear" / "sqrt"); the smallest
#: non-zero value of the figure ("log"; a log scale needs a value > 0). A
#: number sets it explicitly, e.g. 0.02 to blank out bins holding less than
#: 2 % of the members; values below it get the lowest colour (an arrow on the
#: colour bar), and with "log" they are left out.
MOD_BH_HEAT_VMIN = None
#: Upper end of the colour scale ("fraction" norm; "row_max" is always 1).
#: None = for "linear" / "sqrt" the 99th percentile of the non-zero fractions
#: of all panels of the figure, so that a few levels in which every member has
#: the same value (the ocean, fixed regions: fraction 1) do not wash out the
#: rest (the colour bar then ends in an arrow); for "log" the maximum. A
#: number sets it explicitly (1.0 = the full range).
MOD_BH_HEAT_VMAX = None
#: Draw the colour bar of the heat map (one per figure).
MOD_BH_HEAT_CBAR = True

# --- Sharing the heat-map colour scale ---------------------------------------
#: Colours of two heat maps are only comparable if the bin edges (bin width
#: changes the fractions), the colour limits and the colormap are the same.
#:   False (default) = one colour scale per figure (as before). With
#:       ENS_MODEL_SOURCE = "both" the converged and the perturbed heat maps of a
#:       figure always share it.
#:   True = ONE scale for ALL figures of the run (every MOD_BH_PER_FIG chunk,
#:       converged and perturbed): the automatic MOD_BH_HEAT_VMAX is taken over
#:       all of them, so equal colours mean equal fractions in every panel.
MOD_BH_HEAT_SHARE_SCALE = False

#: ENS_MODEL_SOURCE = "both" only. "combined" (default) = ONE figure per chunk
#: with the converged and the perturbed panel of each borehole side by side
#: (the figure is twice as wide: 2 x MOD_BH_NCOLS columns) and ONE colour bar,
#: written to MOD_BH_FILE. "separate" = one figure per source, MOD_BH_FILE
#: (converged) and MOD_BH_FILE + ENS_PERTURBED_SUFFIX (perturbed), same scale,
#: each with its own (identical) colour bar.
MOD_BH_HEAT_FIGURE = "combined"

#: JSON file to save / reuse the colour scale across runs (a converged and a
#: perturbed run, other ensembles, femtic_ens_plot.py -- same format); None =
#: off. It holds the resolved "scale", "norm", "vmin", "vmax", "cmap" and, when
#: all panels use the same bins, the "edges" (log10 Ohm.m); a hand-written file
#: may hold only some of these keys, the rest then comes from MOD_BH_HEAT_*.
#: Values read from the file replace MOD_BH_HEAT_SCALE / _NORM / _VMIN / _VMAX /
#: _CMAP and, if "edges" is present, the bins of EVERY heat map (the entries'
#: "heat_bins" / "xlim"-derived bins); they apply to all heat maps of the run.
#: Example: RESULTS_DIR + "heat_colorscale.json"
MOD_BH_HEAT_SCALE_FILE = None
#: "write" = always (over)write the file with the scale of this run; "read" =
#: use it (stop if it is missing); "auto" (default) = read it if it exists,
#: else write it -- the same setting then serves the first run (writes) and
#: every later one (reads).
MOD_BH_HEAT_SCALE_IO = "auto"
#: Line style per drawn item, keyed "avg", "med", "best", "ensemble",
#: "percentiles" (all percentile lines), a single percentile key such as
#: "p50", or "sigma1" / "sigma2" (both lines of the pair). Any Matplotlib Line2D keyword is allowed. "percentiles" may carry
#: "cmap" instead of "color": the lines are then coloured by level from that
#: colormap. Per-entry "style" dicts override these.
MOD_BH_STYLE = {
    "avg": dict(color="black", lw=2.0, ls="-"),
    "med": dict(color="crimson", lw=2.0, ls="-"),
    "best": dict(color="tab:orange", lw=1.6, ls="-"),
    "ensemble": dict(color="0.5", lw=0.7, ls="-"),
    "percentiles": dict(cmap="viridis", lw=1.2, ls="--"),
    "sigma1": dict(color="navy", lw=1.6, ls="--"),
    "sigma2": dict(color="steelblue", lw=1.2, ls=":"),
    "ref": dict(color="black", lw=2.2, ls="--"),
    # heat map: any pcolormesh keyword, and "cmap" to override
    # MOD_BH_HEAT_CMAP (the colour scale itself: MOD_BH_HEAT_SCALE / _VMIN /
    # _VMAX; the colour bar uses the cmap of the first panel with a heat map)
    "heatmap": dict(rasterized=True),
}
if MOD_BH_HEAT_NORM not in ("fraction", "row_max"):
    raise ValueError(
        f"MOD_BH_HEAT_NORM must be 'fraction' or 'row_max', "
        f"got {MOD_BH_HEAT_NORM!r}."
    )
if MOD_BH_HEAT_SCALE not in ("linear", "sqrt", "log"):
    raise ValueError(
        f"MOD_BH_HEAT_SCALE must be 'linear', 'sqrt' or 'log', "
        f"got {MOD_BH_HEAT_SCALE!r}."
    )
if MOD_BH_HEAT_VMIN is not None and MOD_BH_HEAT_VMIN < 0:
    raise ValueError("MOD_BH_HEAT_VMIN must be >= 0 (or None).")
if MOD_BH_HEAT_SCALE == "log" and MOD_BH_HEAT_VMIN is not None:
    if MOD_BH_HEAT_VMIN <= 0:
        raise ValueError("MOD_BH_HEAT_SCALE='log' needs MOD_BH_HEAT_VMIN > 0.")
if MOD_BH_HEAT_FIGURE not in ("combined", "separate"):
    raise ValueError(
        f"MOD_BH_HEAT_FIGURE must be 'combined' or 'separate', "
        f"got {MOD_BH_HEAT_FIGURE!r}."
    )
if MOD_BH_HEAT_SCALE_IO not in ("auto", "read", "write"):
    raise ValueError(
        f"MOD_BH_HEAT_SCALE_IO must be 'auto', 'read' or 'write', "
        f"got {MOD_BH_HEAT_SCALE_IO!r}."
    )
if (
    MOD_BH
    and MOD_BH_HEAT_SCALE_FILE
    and MOD_BH_HEAT_SCALE_IO == "read"
    and not os.path.isfile(MOD_BH_HEAT_SCALE_FILE)
):
    raise ValueError(
        f"MOD_BH_HEAT_SCALE_IO='read' but MOD_BH_HEAT_SCALE_FILE does not "
        f"exist: {MOD_BH_HEAT_SCALE_FILE}"
    )


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
# Main
# ===========================================================================
if not MOD_BH:
    sys.exit("MOD_BH is empty - nothing to plot.")

# --- (1) Read the results written by femtic_ens_post.py ---------------------
try:
    res = fem.read_ensemble_results(ENSEMBLE_RESULTS, NPZ_PREFIX, out=OUT)
    tpl, _from_vector = fem.resolve_results_template(
        res, TEMPLATE_BLOCK, ocean=MOD_OCEAN
    )
    struct = fem._read_resistivity_block_struct(tpl, ocean=MOD_OCEAN, out=False)
    mesh = fem.BoreholeMesh(MOD_MESH)
except (FileNotFoundError, KeyError, ValueError) as _exc:
    sys.exit(f"femtic_ens_plot_bh.py: {_exc}")

P = res["prefix"]
ens_matrix = res["ens"]
pert_matrix = res["pert"]  # perturbed models of the same members, or None
model_list = res["model_list"]  # [[block file, iteration, nRMS], ...]
_ENS_SRC = res["model_source"]  # "converged" | "perturbed"
_best_i = res["best_index"]
_best_file = res["best_file"]
_pert_name = res["perturbed_file"] or ENS_PERTURBED_FILE
stat_map, _value_keys = fem.ensemble_stat_map(res)

# --- (2) UTM origin (needed for positions given in utm / latlon) -------------
(
    utm_e,
    utm_n,
    utm_lat,
    utm_lon,
    utm_zone,
    utm_north,
    _site_xys,
    _obs_coords_only,
) = fem.resolve_origin_and_sites(
    site_dat=MOD_SITE_DAT,
    site_names=None,
    site_number=None,
    observe_file=ENSEMBLE_DIR + "templates/observe.dat",
    origin_e=MOD_UTM_ORIGIN_E,
    origin_n=MOD_UTM_ORIGIN_N,
    origin_lat=MOD_UTM_ORIGIN_LAT,
    origin_lon=MOD_UTM_ORIGIN_LON,
    zone_override=MOD_UTM_ZONE_OVERRIDE,
    origin_method=MOD_ORIGIN_METHOD,
    plot_sites=False,
)

# --- (3) Reference model, colour scale, borehole list ------------------------
print("\nBOREHOLES:")

# Reference model (MOD_REF_FILE): log10(rho) per element, air = NaN, ocean =
# MOD_OCEAN_RHO (as for the ensemble lines). Wanted when MOD_REF_BH is on or a
# MOD_BH entry lists "ref" in its "what".
_REF = fem.ReferenceModel(MOD_REF_FILE, ocean=MOD_OCEAN, ocean_rho=MOD_OCEAN_RHO)
_ref_el = None
if MOD_REF_BH or any("ref" in (_t.get("what") or ()) for _t in MOD_BH):
    _ref_el = _REF.element_log10(mesh.n_elem)
    if _ref_el is not None:
        print(f"  reference model for the profiles: {MOD_REF_FILE}")

# Heat-map colour scale (configuration, laid over by MOD_BH_HEAT_SCALE_FILE
# when that is read: its bin edges decide how the profiles are binned).
try:
    cs, io = fem.heat_scale_init(
        fem.heat_scale_defaults(
            scale=MOD_BH_HEAT_SCALE,
            norm=MOD_BH_HEAT_NORM,
            vmin=MOD_BH_HEAT_VMIN,
            vmax=MOD_BH_HEAT_VMAX,
            cmap=MOD_BH_HEAT_CMAP,
        ),
        MOD_BH_HEAT_SCALE_FILE,
        MOD_BH_HEAT_SCALE_IO,
        any("heatmap" in (_t.get("what") or ()) for _t in MOD_BH),
        out=OUT,
    )
except (FileNotFoundError, ValueError) as _exc:
    sys.exit(f"femtic_ens_plot_bh.py: {_exc}")

# Expand "sites" templates (one borehole per site, from MOD_SITE_DAT).
specs = fem.expand_site_templates(MOD_BH, MOD_SITE_DAT, utm_e, out=OUT)

# --- (4) Sample the ensemble along the boreholes -------------------------------
try:
    smp = fem.sample_boreholes(
        specs,
        mesh=mesh,
        struct=struct,
        ens=ens_matrix,
        stat_map=stat_map,
        value_keys=_value_keys,
        best_i=_best_i,
        percentiles=res["prc_levels"],
        utm=(utm_e, utm_n, utm_zone, utm_north),
        cs=cs,
        pert=pert_matrix,
        ref_el=_ref_el,
        ref_default=MOD_REF_BH,
        ocean_rho=MOD_OCEAN_RHO,
        zlim_km=MOD_ZLIM,
        display_coords=MOD_DISPLAY_COORDS,
        heat_bins=MOD_BH_HEAT_BINS,
        xlim=fem.borehole_global_xlim(MOD_BH_XLIM, MOD_CLIM),
    )
except ValueError as _exc:
    sys.exit(f"femtic_ens_plot_bh.py: {_exc}")
profiles = smp["profiles"]
if not profiles:
    sys.exit("  MOD_BH: no usable borehole - nothing plotted.")

# --- (5) Figures, sources and colour scales (also writes the scale file) ------
plan = fviz.plan_borehole_figures(
    profiles,
    cs,
    io,
    per_fig=MOD_BH_PER_FIG,
    layout=MOD_BH_HEAT_FIGURE,
    share_scale=MOD_BH_HEAT_SHARE_SCALE,
    scale_file=MOD_BH_HEAT_SCALE_FILE,
    default_source=_ENS_SRC,
    ensemble=P,
    n_members=int(ens_matrix.shape[0]),
)

# --- (6) Profiles as .npz ------------------------------------------------------
os.makedirs(os.path.dirname(MOD_BH_FILE) or ".", exist_ok=True)
if MOD_BH_NPZ:
    _bh_meta = dict(
        ensemble_prefix=P,
        ensemble_source=_ENS_SRC,
        depth_unit="m, z positive-down, model datum",
        rho_unit="Ohm.m (NaN = air / outside mesh)",
        heat_unit=(
            "heat_<name>: (n_levels, n_bins) fraction of the "
            "valid members per bin (rows sum to 1 when all "
            "members are inside the edges); "
            "heat_edges_<name>: bin edges in log10(Ohm.m)"
        ),
        members=[m[0] for m in model_list],
        best_member=_best_file,
        reference_file=(MOD_REF_FILE if _ref_el is not None else None),
        percentiles=[float(v) for v in res["prc_levels"]],
        boreholes=smp["hdr"],
        colour_scale=plan["header"],
    )
    _bh_npz = dict(smp["npz"])
    _bh_npz["header"] = np.array(json.dumps(_bh_meta, indent=2))
    np.savez_compressed(MOD_BH_FILE + ".npz", **_bh_npz)
    print(f"  profiles saved -> {MOD_BH_FILE}.npz")
    if smp["npz_pert"]:
        # the perturbed models of the same members, same layout
        _bh_npz_p = dict(smp["npz_pert"])
        _bh_npz_p["header"] = np.array(
            json.dumps(
                dict(
                    _bh_meta,
                    ensemble_source="perturbed",
                    members=[
                        os.path.join(os.path.dirname(m[0]), _pert_name)
                        for m in model_list
                    ],
                    best_member=os.path.join(
                        os.path.dirname(_best_file), _pert_name
                    ),
                    boreholes=smp["hdr_pert"],
                ),
                indent=2,
            )
        )
        _fp = MOD_BH_FILE + "_perturbed.npz"
        np.savez_compressed(_fp, **_bh_npz_p)
        print(f"  profiles saved -> {_fp}")

# --- (7) Draw ------------------------------------------------------------------
def _figure_title():
    """Figure title from MOD_TITLE: a string, or None for no title."""
    _dir = os.path.basename(os.path.normpath(ENSEMBLE_DIR))
    _auto = _dir + "  " + P
    if MOD_TITLE is None or MOD_TITLE is False:
        return None
    if MOD_TITLE is True:
        return _auto
    _t = str(MOD_TITLE)
    if _t.strip() == "":
        return None
    if _t.strip().lower() == "auto":
        return _auto
    try:
        return _t.format(prefix=P, dir=_dir)
    except (KeyError, IndexError, ValueError, AttributeError) as _exc:
        print(f"  MOD_TITLE: cannot fill in {_t!r} ({_exc!r}) - used as it stands.")
        return _t


_opts = fviz.BoreholeFigureOptions(
    line_style=MOD_BH_STYLE,
    alpha=MOD_BH_ALPHA,
    xlim=fem.borehole_global_xlim(MOD_BH_XLIM, MOD_CLIM),
    ncols=MOD_BH_NCOLS,
    panel_size=MOD_BH_PANEL_SIZE,
    share_depth=MOD_BH_SHARE_DEPTH,
    legend_loc=MOD_BH_LEGEND_LOC,
    heat_cbar=MOD_BH_HEAT_CBAR,
    depth_km=MOD_DEPTH_KM,
    tick_fontsize=MOD_TICK_FONTSIZE,
    label_fontsize=MOD_LABEL_FONTSIZE,
    title_fontsize=MOD_TITLE_FONTSIZE,
    dpi=MOD_DPI,
    formats=_MOD_PLOT_FORMATS,
    show=_SHOW_PLOTS,
    ref_label=MOD_REF_LABEL if _ref_el is not None else None,
    suptitle=_figure_title(),
    tag_sources=pert_matrix is not None,
    out=OUT,
)
fviz.draw_borehole_figures(plan, cs, MOD_BH_FILE, _opts, perturbed_suffix="_perturbed")

print("\nfemtic_ens_plot_bh.py complete.")

# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
_summary_path = utl.write_param_summary(__file__, out_dir=ENSEMBLE_DIR)
