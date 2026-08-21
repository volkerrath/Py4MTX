#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_plot_ensemble.py — Ensemble slice plot for a set of FEMTIC inversion runs.

Scans ENSEMBLE_DIR for converged ensemble members exactly the way
femtic_ens_post.py does (ENSEMBLE_NAME* sub-directories, femtic.cnv,
NRMS_MAX threshold), then produces:

  (1) [default, PER_MEMBER_PLOT=True] Per converged member, two
      fviz.plot_model_slices() figures in their own sub-directory
      WORK_DIR/<label>/: the perturbed prior model (iter0.<ext>,
      from resistivity_block_iter0.dat, colormap/scale PLOT_CMAP_ITER0
      / PLOT_CLIM_ITER0) and the best-fit model (best.<ext>, from
      resistivity_block_iter{numit}.dat, numit from femtic.cnv,
      colormap/scale PLOT_CMAP_BEST / PLOT_CLIM_BEST). One file per format in PLOT_FORMAT (e.g.
      ["pdf", "jpg"]). When "pdf" is among the formats,
      PER_MEMBER_PDF_CATALOG_MODE selects which per-member pdf pages
      (none / iter0 / best / both, interlaced per member) are also
      combined into one multi-page catalog PDF
      (PER_MEMBER_CATALOG_FILE).

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
    2026-08-15  Claude Sonnet 5 (Anthropic)
                Per-member output reworked: (1) PLOT_FORMAT replaces
                PER_MEMBER_FORMAT and now accepts a list, e.g.
                ["pdf", "jpg"] (normalised to _PLOT_FORMATS), mirroring
                MOD_PLOT_FORMAT in femtic_ens_post.py -- one savefig()
                call per format, figure rebuilt per format since
                plot_model_slices doesn't expose a re-save path.
                (2) Per-member files now live in their own sub-directory
                WORK_DIR/<label>/ as "iter0.<ext>" / "best.<ext>",
                replacing the flat "<label>_iter0.<ext>" naming.
                (3) New PER_MEMBER_PDF_CATALOG flag (default True):
                when "pdf" is among _PLOT_FORMATS, every per-member pdf
                Figure (iter0 + best, in plot order) is additionally
                collected and written as one multi-page catalog via
                matplotlib.backends.backend_pdf.PdfPages, to
                PER_MEMBER_CATALOG_FILE.
    2026-08-16  Claude Sonnet 5 (Anthropic)
                Reduced peak memory for large ensembles: the pdf catalog
                is now opened once before the per-member loop and each
                pdf figure is written via PdfPages.savefig() and
                released (del + gc.collect()) immediately after it's
                built, instead of accumulating every member's Figure
                object in a Python list and writing them all at the end.
                _plot_member_slice() gained a pdf_catalog= kwarg and no
                longer returns a Figure.
    2026-08-20  Claude Sonnet 5 (Anthropic)
                _plot_member_slice(): the pdf-format savefig call is
                now wrapped in mpl.rc_context({"pdf.compression": 0})
                to disable matplotlib's PDF indexed-colour image path
                (backend_pdf.PdfFile._writeImg / np.searchsorted
                palette mapping, gated on the pdf.compression
                rcParam) at the source, plus a try/except IndexError
                fallback (render PNG, convert to PDF with Pillow) as
                a safety net in case the same class of bug (same code
                path as matplotlib/matplotlib#25806) is hit some other
                way. Data-/render-dependent, not a bug in this script
                or in plot_model_slices -- generated/reviewed by
                Claude, should be checked before relying on it in
                production.
    2026-08-20  Claude Sonnet 5 (Anthropic)
                All user-facing length parameters in the Configuration
                section (PLOT_SLICES z0/x0/y0/point, PLOT_XLIM/YLIM/ZLIM,
                PROJECTION_DIST, MAP_MARKERS positions, BOREHOLE_SITES
                x/y/z_top/z_bot/dz) are now entered in kilometres instead
                of metres. A new "Unit conversion" block, run once right
                after the config section, converts everything to
                model-local metres before fem.resolve_slice_positions()
                or fviz.plot_model_slices()/plot_ensemble_slices() see
                it -- those functions and femtic_ens_post.py's own
                convention are unchanged (still metres internally).
                CRS-tagged specs, e.g. (value, "utm") / (value,
                "latlon"), are absolute coordinates and are left
                unchanged by the conversion; only plain numbers /
                implicit-"model" specs are treated as model-local km.
                Fixes a bug where PLOT_SLICES z0 values and
                PLOT_XLIM/YLIM/ZLIM were written with "# km" comments
                but silently consumed as metres (e.g. z0=25.0 plotted a
                slice at 25 m depth, not 25 km, and PLOT_XLIM=[-25, 25]
                restricted the map panels to a 50 m-wide box instead of
                50 km). PROJECTION_DIST and the BOREHOLE_SITES example
                were also updated from metres to km.
    2026-08-20  Claude Sonnet 5 (Anthropic)
                (1) PER_MEMBER_PDF_CATALOG (bool) replaced by
                PER_MEMBER_PDF_CATALOG_MODE ("none" | "iter0" | "best" |
                "both", default "both"), selecting which per-member pdf
                pages go into the catalog: none, prior-only, best-only,
                or both interlaced per member in plot order (iter0_A,
                best_A, iter0_B, best_B, ...) -- "both" reproduces the
                previous PER_MEMBER_PDF_CATALOG=True behaviour, since
                the per-member loop already visits iter0 then best for
                each member in turn. Validated against
                _VALID_CATALOG_MODES at load time.
                (2) PLOT_CLIM split into PLOT_CLIM_ITER0 and
                PLOT_CLIM_BEST so the perturbed-prior and best-fit
                model plots can use distinct log10(rho) colour scales
                (the prior's resistivity range often differs
                substantially from the inverted result). _plot_member_
                slice() gained a required clim= kwarg -- the two call
                sites in the per-member loop now pass PLOT_CLIM_ITER0 /
                PLOT_CLIM_BEST respectively; the joint ensemble figure
                (PLOT_JOINT), which only plots best-fit models, uses
                PLOT_CLIM_BEST.
                (3) PLOT_CMAP likewise split into PLOT_CMAP_ITER0 and
                PLOT_CMAP_BEST, mirroring (2), so the colormap itself
                -- not just the colour limits -- can differ between
                the prior and best-fit plots. _plot_member_slice()
                gained a required cmap= kwarg alongside clim=; the
                joint ensemble figure uses PLOT_CMAP_BEST.

@author: vrath
"""

import os
import sys
import math
import gc
import inspect
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

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
WORK_DIR = r"/media/vrath/LargeBack/Ensembles/misti2026/gst/"
#: Mesh file — always required for plotting.
MESH_FILE = WORK_DIR + "/templates/mesh.dat"

#: observe.dat — used by ESTIMATE_ORIGIN and as fallback for SITE_NUMBER.
OBSERVE_FILE = WORK_DIR +  "/templates/observe.dat"

#: Site list produced by mt_make_sitelist.py (WHAT_FOR="femtic").
#: Format (comma-separated, no header):
#:   name, lat, lon, elev, sitenum, easting, northing
#: Easting/northing are UTM metres; model-local x/y is derived via
#: fem.utm_to_model using the mesh-centre origin.
#: Set to None to fall back to the observe.dat / SITE_NUMBER path.
SITE_DAT = WORK_DIR + "/templates/site.dat"   # set to None to disable

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
ENSEMBLE_NAME = "misti_gst_suzuki_rnd"

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
#: converged member: "iter0.<ext>" (resistivity_block_iter0.dat, the
#: perturbed/prior model) and "best.<ext>" (resistivity_block_iter{numit}.dat,
#: the best-fit model at the iteration femtic.cnv reports).  Each member
#: gets its own sub-directory WORK_DIR/<label>/.
PER_MEMBER_PLOT = True

#: Output format(s) for per-member plots (Matplotlib Agg-backend savefig
#: formats).  A list saves one file per format per plot (the figure is
#: still rebuilt per format, since plot_model_slices doesn't expose a way
#: to re-save an already-built figure), e.g. ["pdf", "jpg"] writes both a
#: vector version and a raster preview. A bare string ("pdf") also works
#: and is treated as a single-entry list.
#: Supported values: "pdf", "svg", "eps" (vector); "png", "jpg"/"jpeg",
#: "tif"/"tiff", "webp" (raster, rendered at PLOT_DPI).
PLOT_FORMAT = ["pdf"]

#: Normalised to a list regardless of whether a bare string or a list was
#: set above.
_PLOT_FORMATS = (
    [PLOT_FORMAT] if isinstance(PLOT_FORMAT, str) else list(PLOT_FORMAT)
)

#: Controls which per-member figures (if any) are additionally collected
#: into one multi-page catalog PDF via matplotlib.backends.backend_pdf.
#: PdfPages, saved as PER_MEMBER_CATALOG_FILE. Only takes effect when
#: "pdf" is among _PLOT_FORMATS. Does not affect the individual
#: per-member files, which are still written as usual regardless of
#: this setting. One of:
#:   "none"  — no catalog is built
#:   "iter0" — catalog contains only the perturbed-prior (iter0) pages
#:   "best"  — catalog contains only the best-fit pages
#:   "both"  — catalog contains both, interlaced per member in plot
#:             order (iter0_A, best_A, iter0_B, best_B, ...)
PER_MEMBER_PDF_CATALOG_MODE = "both"   # "none" | "iter0" | "best" | "both"

#: Output path for the multi-page pdf catalog.
PER_MEMBER_CATALOG_FILE = WORK_DIR + ENSEMBLE_NAME+"_catalog.pdf"

_VALID_CATALOG_MODES = ("none", "iter0", "best", "both")
if PER_MEMBER_PDF_CATALOG_MODE not in _VALID_CATALOG_MODES:
    raise ValueError(
        f"PER_MEMBER_PDF_CATALOG_MODE={PER_MEMBER_PDF_CATALOG_MODE!r} — "
        f"must be one of {_VALID_CATALOG_MODES}."
    )

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
DISPLAY_COORDS = "latlon"

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
PLOT_NROWS = 4
PLOT_NCOLS = 2

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

#: Maximum distance (km) from slice plane for site projection onto curtains.
PROJECTION_DIST = 5.0

#: Marker style for map panels.
SITE_MARKER = dict(marker="v", color="black", ms=8, zorder=10, label=None)

#: Marker style for curtain panels (None → same as SITE_MARKER).
SITE_MARKER_SLICES = None

#: Additional map markers (e.g. known features).  List of dicts:
#:   dict(pos=(x, y), marker="*", color="red", ms=10, label="label")
#: pos accepts model-local km or (value, "utm"/"latlon") tuples (the
#: latter are absolute coordinates -- UTM metres / decimal degrees --
#: and are not affected by the km convention).
MAP_MARKERS = []

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
#: Figure DPI for saved files.
PLOT_DPI = 600

#: Matplotlib colormap name — set separately for the perturbed-prior
#: (iter0) and best-fit models, same reasoning as PLOT_CLIM_ITER0 /
#: PLOT_CLIM_BEST below. PLOT_CMAP_BEST is also used for the joint
#: ensemble figure (PLOT_JOINT), which only plots best-fit models.
PLOT_CMAP_ITER0 = "turbo_r"
PLOT_CMAP_BEST  = "turbo_r"

#: Colour limits [log10(ρ_min), log10(ρ_max)] — None = auto. Set
#: separately for the perturbed-prior (iter0) and best-fit models, since
#: the prior's resistivity range often differs from the inverted result.
#: PLOT_CLIM_BEST is also used for the joint ensemble figure (PLOT_JOINT),
#: which only plots best-fit models.
PLOT_CLIM_ITER0 = [0.0, 4.0]      # log10(Ω·m)
PLOT_CLIM_BEST  = [0.0, 4.0]      # log10(Ω·m)

#: Flat colour for ocean / lake cells.  None → use colormap.
PLOT_OCEAN_COLOR = "lightgrey"

#: Axes facecolor for air / background.  None = figure default.
PLOT_AIR_BGCOLOR = None

#: Slice specification — same format as femtic_mod_plot.py PLOT_SLICES.
#: All lengths below are given in KILOMETRES; the "Unit conversion" block
#: further down converts them to model-local metres before fem/fviz see
#: them.  CRS-tagged specs -- (value, "utm") in UTM metres, (value,
#: "latlon") in decimal degrees -- are absolute coordinates and are left
#: unchanged by that conversion.
#: Each dict must have 'kind' and the matching position key:
#:   kind="map"   → z0   (depth in model-local km)
#:   kind="ns"    → x0   (easting;  plain float = model-local km,
#:                        or (value, "utm") / (value, "latlon"))
#:   kind="ew"    → y0   (northing; same CRS tagging)
#:   kind="plane" → point ([x, y, z] model-local km, or ([x,y,z], "utm"
#:                  / "latlon") with z still model-local km), strike, dip
#:   invert_x     → True to flip horizontal axis on ns/ew/plane panels
#:                  (for comparison with sections using opposite convention)
# PLOT_SLICES = [
#     dict(kind="map",  z0=5.0),
#     dict(kind="map",  z0=15.0),
#     dict(kind="ns",   x0=(-70.8700, "latlon")),
#     dict(kind="ew",   y0=(-16.3500, "latlon")),
# ]
PLOT_SLICES = [    
    dict(kind="map", z0=0.0),    # km
    dict(kind="map", z0=5.0),    # km
    dict(kind="map", z0=10.0),   # km
    dict(kind="map", z0=15.0),   # km
    dict(kind="map", z0=20.0),   # km
    dict(kind="map", z0=25.0),   # km
    dict(kind="ns",  x0=(-71.40723, 'latlon')),    # deg
    dict(kind="ew",  y0=(-16.299593, 'latlon')),    # deg
]

#: Global axis limits in model-local KILOMETRES.  None → auto.
PLOT_XLIM = [-25., 25.]
PLOT_YLIM = [-25., 25.]
PLOT_ZLIM = [  -6., 30.]

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
#: Keys: "name", "x", "y", "z_top", "z_bot", "dz" — all lengths in km.
#: x/y accept plain float (model-local km) or (value, "utm"/"latlon")
#: tuples (absolute coordinates, left unchanged by the km conversion).
BOREHOLE_SITES = [
    # dict(name="BH-01", x=0.0, y=0.0, z_top=0.0, z_bot=20.0, dz=0.2),
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
# Unit conversion: user parameters above are in km  →  metres below
# ===========================================================================
# fem.resolve_slice_positions() and fviz.plot_model_slices() /
# plot_ensemble_slices() require model-local metres (the convention shared
# with femtic_ens_post.py and femtic.py, left unchanged here). Everything
# the user sets above -- PLOT_SLICES z0/x0/y0/point, PLOT_XLIM/YLIM/ZLIM,
# PROJECTION_DIST, MAP_MARKERS positions, BOREHOLE_SITES x/y/z_top/z_bot/dz
# -- is in km, so it is converted exactly once here, immediately after the
# config section. Nothing past this point needs to change; the rest of the
# script (including femtic.py / femtic_viz.py) still works in metres.
#
# CRS-tagged position specs -- (value, "utm") in absolute UTM metres, or
# (value, "latlon") in decimal degrees -- are absolute coordinates, not
# model-local lengths, so they pass through unchanged. Only plain numbers
# (or explicit ("model") tags) are treated as model-local km.

def _km_to_m(v):
    """Convert a plain scalar length from km to m; None passes through."""
    return None if v is None else float(v) * 1000.0


def _km_to_m_lim(lim):
    """Convert an [lo, hi] limit pair from km to m; None (either) passes."""
    if lim is None:
        return None
    return [None if x is None else float(x) * 1000.0 for x in lim]


def _km_to_m_pos(raw):
    """Convert an x0/y0-style position spec from km to m.

    Plain numbers and explicit ("model") tags are model-local km and are
    converted; ("utm") / ("latlon") tags are absolute coordinates and are
    returned unchanged.
    """
    if isinstance(raw, (int, float)):
        return float(raw) * 1000.0
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and isinstance(raw[1], str):
        val, crs = raw
        return (float(val) * 1000.0, crs) if crs == "model" else raw
    raise ValueError(f"_km_to_m_pos: unexpected position spec {raw!r}")


def _km_to_m_point(raw):
    """Convert a plane-slice 'point' spec ([x, y, z], optionally CRS-tagged)
    from km to m. z is always model-local and is always converted; x/y are
    converted only under an implicit/explicit "model" CRS -- ("utm") /
    ("latlon") x/y are absolute and left unchanged.
    """
    tagged = (isinstance(raw, (list, tuple)) and len(raw) == 2
              and isinstance(raw[1], str))
    pt, crs = (raw[0], raw[1]) if tagged else (raw, "model")
    pt = list(pt)
    if crs == "model":
        pt = [pt[0] * 1000.0, pt[1] * 1000.0, pt[2] * 1000.0]
    else:
        pt = [pt[0], pt[1], pt[2] * 1000.0]
    return (pt, crs) if tagged else pt


def _km_to_m_slice(spec: dict) -> dict:
    """Return a copy of one PLOT_SLICES entry with lengths converted km → m."""
    s = dict(spec)
    if "z0" in s:
        s["z0"] = _km_to_m(s["z0"])
    if "x0" in s:
        s["x0"] = _km_to_m_pos(s["x0"])
    if "y0" in s:
        s["y0"] = _km_to_m_pos(s["y0"])
    if "point" in s:
        s["point"] = _km_to_m_point(s["point"])
    for _k in ("xlim", "ylim", "zlim"):
        if _k in s:
            s[_k] = _km_to_m_lim(s[_k])
    return s


PLOT_SLICES = [_km_to_m_slice(_s) for _s in PLOT_SLICES]

PLOT_XLIM = _km_to_m_lim(PLOT_XLIM)
PLOT_YLIM = _km_to_m_lim(PLOT_YLIM)
PLOT_ZLIM = _km_to_m_lim(PLOT_ZLIM)

PROJECTION_DIST = _km_to_m(PROJECTION_DIST)

_converted_markers = []
for _mk in MAP_MARKERS:
    _mk = dict(_mk)
    if "pos" in _mk:
        _mk["pos"] = _km_to_m_pos(_mk["pos"])
    _converted_markers.append(_mk)
MAP_MARKERS = _converted_markers

_converted_boreholes = []
for _bh in BOREHOLE_SITES:
    _bh = dict(_bh)
    if "x" in _bh:
        _bh["x"] = _km_to_m_pos(_bh["x"])
    if "y" in _bh:
        _bh["y"] = _km_to_m_pos(_bh["y"])
    for _k in ("z_top", "z_bot", "dz"):
        if _k in _bh:
            _bh[_k] = _km_to_m(_bh[_k])
    _converted_boreholes.append(_bh)
BOREHOLE_SITES = _converted_boreholes


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
    out_stem: str,
    *,
    slices_resolved: list,
    site_xys: list,
    obs_coords_only: bool,
    figure_title: str,
    cmap: str,
    clim: list | None,
    nrms_annotation: dict | None = None,
    pdf_catalog: "PdfPages | None" = None,
) -> None:
    """Call fviz.plot_model_slices once per _PLOT_FORMATS entry.

    Mirrors femtic_ens_post.py's ``_plot_slice`` helper: same PLOT_*
    options (CRS handling, site overlay, figure layout) used throughout
    this script, applied to one resistivity block file at a time.

    Parameters
    ----------
    out_stem : str
        Output path *without* an extension. ".<fmt>" is appended for
        each entry in PLOT_FORMAT / _PLOT_FORMATS (e.g. base "foo" +
        ["pdf", "jpg"] -> "foo.pdf" and "foo.jpg", same figure, one
        savefig() call per format — plot_model_slices doesn't expose a
        way to re-save an already-built figure under a second
        extension, so it is rebuilt fresh for each format.
    cmap : str
        Matplotlib colormap name for this call — pass PLOT_CMAP_ITER0
        or PLOT_CMAP_BEST from the caller so the perturbed-prior and
        best-fit plots can use distinct colormaps.
    clim : list or None
        Colour limits [log10(ρ_min), log10(ρ_max)] for this call —
        pass PLOT_CLIM_ITER0 or PLOT_CLIM_BEST from the caller so the
        perturbed-prior and best-fit plots can use distinct scales.
    pdf_catalog : PdfPages, optional
        If given and "pdf" is among _PLOT_FORMATS, the pdf-format
        Figure is written to this already-open PdfPages immediately
        after it's built, then the local reference is dropped so it
        can be garbage-collected before the next member is plotted —
        this keeps peak memory to one figure at a time instead of
        holding every member's figure until the whole ensemble is
        done. plot_model_slices already closes the figure
        (plt.close(fig)) before returning it; the object is still
        valid for a PdfPages.savefig() call right after.
    """
    if fviz is None:
        print("  plot_member_slice: femtic_viz not available — skipping.")
        return

    for _fmt in _PLOT_FORMATS:
        _out_file = f"{out_stem}.{_fmt}"
        _kwargs = dict(
            model_file          = block_file,
            mesh_file           = MESH_FILE,
            slices              = slices_resolved,
            cmap                = cmap,
            clim                = clim,
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
            dpi                 = PLOT_DPI,
            out                 = OUT,
        )

        # matplotlib's PDF backend only tries to write images as an
        # "Indexed" PDF colour space (backend_pdf.PdfFile._writeImg,
        # gated on the pdf.compression rcParam) when pdf.compression is
        # truthy. That indexed-colour path is what triggers the
        # np.searchsorted IndexError below (composited raster with few
        # distinct colours; an anti-aliased pixel not in the
        # pre-counted palette maps past the end of the palette array —
        # data-/render-dependent, not a bug in plot_model_slices or in
        # this script). Disabling compression for just this savefig()
        # call forces the always-safe plain-RGB branch instead, at the
        # cost of a somewhat larger pdf file. Scoped with rc_context so
        # it doesn't affect global matplotlib state or non-pdf formats.
        _rc_overrides = {"pdf.compression": 0} if _fmt == "pdf" else {}

        try:
            with mpl.rc_context(_rc_overrides):
                _fig = fviz.plot_model_slices(plot_file=_out_file, **_kwargs)
        except IndexError as _err:
            # Belt-and-braces fallback in case some other render path
            # still hits the same class of matplotlib PDF-backend bug
            # (see matplotlib/matplotlib#25806) despite pdf.compression
            # being disabled above. Render PNG instead (unaffected
            # code path) and convert that to PDF with Pillow.
            if _fmt != "pdf":
                raise
            print(f"    ! pdf save hit matplotlib indexed-colour bug "
                  f"({_err}) -- falling back to PNG->PDF for {_out_file}")
            _png_fallback = f"{out_stem}__pdf_fallback.png"
            _fig = fviz.plot_model_slices(plot_file=_png_fallback, **_kwargs)
            from PIL import Image
            Image.open(_png_fallback).convert("RGB").save(
                _out_file, "PDF", resolution=PLOT_DPI)
            os.remove(_png_fallback)

        if OUT:
            print(f"    saved -> {_out_file}")

        if _fmt == "pdf" and pdf_catalog is not None:
            pdf_catalog.savefig(_fig)

        # Drop the reference immediately so the figure (axes, colorbars,
        # gridded image data) can be garbage-collected before the next
        # format/member is built, instead of accumulating across the
        # whole ensemble. plt.close(fig) inside plot_model_slices already
        # detached it from pyplot's figure registry; this just releases
        # the last strong reference held here.
        del _fig


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
# Each member gets its own sub-directory WORK_DIR/<label>/ containing
# iter0.<ext> and best.<ext> (one pair of files per entry in _PLOT_FORMATS).
#
# Memory note: the pdf catalog (if enabled) is opened ONCE before this loop
# and every pdf figure is written to it and released immediately after it's
# built (see pdf_catalog= in _plot_member_slice), rather than accumulating
# all figures in a Python list and writing them at the end. For a 50-member
# ensemble that's the difference between ~2 live figures at a time and ~100
# — the earlier approach held every member's Matplotlib Figure (axes,
# colorbars, gridded slice arrays) in memory simultaneously until the very
# last member was plotted.
if PER_MEMBER_PLOT:
    if fviz is None:
        sys.exit("femtic_viz not available — cannot plot.  Check your installation.")

    print(f"Plotting {n_members} converged member(s), "
          f"2 figures each (iter0 + best), formats={_PLOT_FORMATS} …")

    _use_catalog = PER_MEMBER_PDF_CATALOG_MODE != "none" and "pdf" in _PLOT_FORMATS
    _catalog_iter0 = _use_catalog and PER_MEMBER_PDF_CATALOG_MODE in ("iter0", "both")
    _catalog_best  = _use_catalog and PER_MEMBER_PDF_CATALOG_MODE in ("best", "both")
    _pdf_catalog = PdfPages(PER_MEMBER_CATALOG_FILE) if _use_catalog else None
    _n_catalog_pages = 0

    try:
        for _m in model_list:
            _label      = _m["label"]
            _member_dir = os.path.join(WORK_DIR, _label)
            os.makedirs(_member_dir, exist_ok=True)

            print(f"\n  member {_label!r}: perturbed prior (iter0)")
            _plot_member_slice(
                block_file      = _m["iter0_file"],
                out_stem        = os.path.join(_member_dir, "iter0"),
                slices_resolved = slices_resolved,
                site_xys        = site_xys,
                obs_coords_only = _sites_from_obs,
                figure_title    = f"{_label} — perturbed prior (iter0)",
                cmap            = PLOT_CMAP_ITER0,
                clim            = PLOT_CLIM_ITER0,
                pdf_catalog     = _pdf_catalog if _catalog_iter0 else None,
            )
            if _catalog_iter0:
                _n_catalog_pages += 1

            print(f"  member {_label!r}: best fit (iter{_m['numit']}, "
                  f"nRMS={_m['nrms']:.4f})")
            _plot_member_slice(
                block_file      = _m["best_file"],
                out_stem        = os.path.join(_member_dir, "best"),
                slices_resolved = slices_resolved,
                site_xys        = site_xys,
                obs_coords_only = _sites_from_obs,
                figure_title    = f"{_label} — best fit (iter{_m['numit']})",
                cmap            = PLOT_CMAP_BEST,
                clim            = PLOT_CLIM_BEST,
                nrms_annotation = dict(nrms=_m["nrms"]),
                pdf_catalog     = _pdf_catalog if _catalog_best else None,
            )
            if _catalog_best:
                _n_catalog_pages += 1

            # Matplotlib Figure/Axes hold reference cycles, so refcounting
            # alone won't free them right away; force a cyclic-gc pass
            # after each member so peak memory stays bounded by ~one
            # member's figures rather than drifting upward over the run.
            gc.collect()
    finally:
        if _pdf_catalog is not None:
            _pdf_info = _pdf_catalog.infodict()
            _pdf_info["Title"] = "femtic_ens_plot ensemble catalog"
            _pdf_info["Author"] = "femtic_ens_plot.py"
            _pdf_catalog.close()

    print("\nPer-member plots done.")
    if _use_catalog:
        print(f"  pdf catalog ({PER_MEMBER_PDF_CATALOG_MODE}): "
              f"{_n_catalog_pages} page(s) → {PER_MEMBER_CATALOG_FILE}")
    elif PER_MEMBER_PDF_CATALOG_MODE != "none":
        print("  pdf catalog: \"pdf\" not in PLOT_FORMAT — skipped.")

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
        cmap               = PLOT_CMAP_BEST,
        clim               = PLOT_CLIM_BEST,
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
