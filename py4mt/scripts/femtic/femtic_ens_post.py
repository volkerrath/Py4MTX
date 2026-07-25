#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_post.py — Ensemble postprocessing for FEMTIC

Collects all converged members of a FEMTIC ensemble (RTO, GST, or any
directory-based ensemble), computes summary statistics, assembles the
empirical covariance, and saves everything to a compressed ``.npz`` file.

Optionally produces slice figures for the **best-nRMS member** (QC) and
for the **ensemble statistics** (mean, variance, median, MAD).

References
----------
Bardsley, J. M.; Solonen, A.; Haario, H. & Laine, M.
    Randomize-Then-Optimize: a Method for Sampling from Posterior
    Distributions in Nonlinear Inverse Problems.
    SIAM J. Sci. Comp., 2014, 36, A1895-A1910.

Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
    Uncertainty quantification for regularized inversion of electromagnetic
    geophysical data. Part I: Motivation and Theory.
    Geophysical Journal International, doi:10.1093/gji/ggac241, 2022.

Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
    Uncertainty quantification for regularized inversion of electromagnetic
    geophysical data – Part II: application in 1-D and 2-D problems.
    Geophysical Journal International, doi:10.1093/gji/ggac242, 2022.

Suzuki, K.; Assessing inversion uncertainty from initial-model variability in 
    3-D magnetotelluric inversion: Application to a geothermal field
    Journal of Applied Geophysics, 251, 106320
    doi:10.1016/j.jappgeo.2026.106320, 2026


@author: vrath

Provenance
----------
2025-04-30  vrath
            Created as femtic_rto_post.py.
2026-03-03  Claude (Anthropic)
            Renamed user-set parameters to UPPERCASE; generated README.
2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
            Added femtic_viz import; MESH_FILE / MOD_QC / MOD_QC_FILE /
            MOD_QC_SLICES / MOD_QC_* config vars; QC slice plot of
            best-nRMS member at end of main block (calls
            fviz.plot_model_slices).
2026-06-11  vrath / Claude Sonnet 4.6 (Anthropic)
            Renamed femtic_rto_post.py → femtic_ens_post.py for
            algorithm-agnostic use.  Fixed axis bug: mean/var/median/MAD
            were computed over axis=1 (free parameters) instead of axis=0
            (members) — statistics are now correct.  MOD_QC block
            replaced by full MOD_SLICES framework matching
            femtic_mod_edit.py / femtic_mod_math.py: UTM origin
            resolution, CRS-aware fem.resolve_slice_positions, site
            overlay via _resolve_origin_and_sites(); new MOD_STATS block
            plots mean/variance/median/MAD as individual slice figures.
            ENSEMBLE_PREFIX added for generic naming of output keys.
2026-07-07  vrath / Claude Sonnet 5 (Anthropic)
            Aligned the entire plotting config surface with
            femtic_gst_prep.py / femtic_rto_prep.py: MESH_FILE →
            MOD_MESH; UTM_ORIGIN_* → MOD_UTM_ORIGIN_*; UTM_ZONE_OVERRIDE →
            MOD_UTM_ZONE_OVERRIDE; ORIGIN_METHOD → MOD_ORIGIN_METHOD;
            DISPLAY_COORDS → MOD_DISPLAY_COORDS; SITE_DAT/SITE_NAMES →
            MOD_SITE_DAT/MOD_SITE_NAMES; MOD_SITES_MAPS/SLICES →
            MOD_PLOT_SITES_MAPS/SLICES; PROJECTION_DIST →
            MOD_PROJECTION_DIST; SITE_MARKER(_SLICES) →
            MOD_SITE_MARKER(_SLICES); MAP_MARKERS → MOD_MAP_MARKERS;
            DEPTH_KM/HORIZ_KM → MOD_DEPTH_KM/MOD_HORIZ_KM.  Added
            MOD_OCEAN/MOD_AIR_RHO, MOD_SITE_NUMBER (observe.dat fallback,
            same as femtic_gst_prep.py), MOD_AIR_COLOR, MOD_ALPHA_FILE/
            MODE/BLANK_THRESH, MOD_PANEL_WIDTH, MOD_FIGSIZE.  Removed a
            latent duplicate MOD_XLIM/YLIM/ZLIM assignment that silently
            discarded the first (non-None) values.  _resolve_origin_and_
            sites() and _plot_slice() now match femtic_gst_prep.py's
            origin-resolution and plot_model_slices() call byte-for-byte
            in option coverage, so QC and statistics figures render
            identically to the ensemble-generation scripts given the
            same MOD_* settings.
2026-07-09  vrath / Claude Sonnet 5 (Anthropic)
            Merged MOD_QC_DPI / MOD_STATS_DPI into a single MOD_DPI knob
            (matching femtic_gst_prep.py / femtic_nss.py — one figure-DPI
            setting per script, not one per plot type).  _plot_slice() no
            longer takes a dpi argument; it reads MOD_DPI directly.
2026-07-17  Claude Sonnet 5 (Anthropic)
            scipy.sparse: migrated from legacy matrix to array-equivalent
            API — scs.csr_matrix(tmp) → scs.csr_array(tmp) when building
            the sparsified empirical covariance (ens_covs). No functional
            change; ens_covs is only used for its .nnz count.
2026-07-25  Claude Sonnet 5 (Anthropic)
            Covariance estimation made optional (COMPUTE_COV; skips step
            (3) entirely, omitting the *_cov* keys from the .npz output).
            Added COV_METHOD="low_rank": thin SVD of the centred ensemble
            matrix (n_members, n_free) instead of the dense empirical
            covariance. Since the sample covariance of n_members draws
            has rank <= n_members-1, this is an *exact* factorisation
            (not an approximation) whenever n_members << n_free — the
            usual case — cutting cost from O(n_free^2 * n_members) time /
            O(n_free^2) memory to O(n_members^2 * n_free) time /
            O(n_members * n_free) memory. Stores f"{P}_cov_eigval" /
            f"{P}_cov_eigvec" in place of the dense f"{P}_cov"; full
            covariance reconstructs exactly as
            eigvec @ diag(eigval) @ eigvec.T. Also saved f"{P}_prc_levels"
            (the PERCENTILES list itself) alongside f"{P}_prc" for
            self-describing output. MOD_STATS now writes a block file and
            slice figure for each PERCENTILES level in addition to
            avg/var/med/mad, keyed as "p2_3", "p50", "p97_7", etc.
            (default MOD_STATS_WHAT includes all of them).
2026-07-25  Claude Sonnet 5 (Anthropic)
            Added MOD_STATS_CLIM: per-statistic colour-scale override for
            MOD_STATS plots. AVG/MED/percentile panels default to the
            shared MOD_CLIM (same log10(Ω·m) space as the model); VAR/MAD/
            QDIFF panels — a different scale entirely — default to a fixed
            [-2, 2] range instead of silently reusing MOD_CLIM (which
            previously made them blank or meaningless); set any of those
            keys to None in MOD_STATS_CLIM to fall back to auto per-panel
            scaling instead. _plot_slice() gained a clim= parameter to
            carry this through.
            Added QDIFF_PAIRS (default [(15.9, 84.1)]): computes
            |P_hi - P_lo| per free parameter as an outlier-robust spread
            statistic, saved to the .npz as f"{P}_qdiff_<lo>_<hi>" and
            available in MOD_STATS under the same key. Added
            MOD_ROI_AUTO/MOD_ROI_PAD_XY/MOD_ROI_ZLIM: MOD_XLIM/MOD_YLIM
            are now derived automatically from the site bounding box (+
            padding) and MOD_ZLIM from MOD_ROI_ZLIM, applied once after
            _resolve_origin_and_sites() and picked up by every subsequent
            _plot_slice() call. This also feeds femtic_viz.py's existing
            aspect-ratio panel-width logic (MOD_PANEL_WIDTH=None +
            MOD_EQUAL_ASPECT=True), so map vs. ns vs. ew panels now size
            themselves differently instead of coming out uniformly square
            whenever xlim/ylim/zlim were previously left at their
            full-mesh-auto None default. Changed MOD_NROWS/MOD_NCOLS
            defaults from None/None (1 row) to 2/2, matching the 4 default
            MOD_SLICES panels. Also fixed a matching sign bug in
            femtic_viz.py's plot_model_slices (ns/ew curtain panels came
            out blank/upside down whenever MOD_ZLIM was actually set,
            which is why the bug was invisible until this ROI change made
            MOD_ZLIM non-None by default) — see femtic_viz.py provenance.
2026-07-25  Claude Sonnet 5 (Anthropic)
            Added MOD_TICK_FONTSIZE / MOD_LABEL_FONTSIZE (defaults 7 / 8,
            matching fviz.plot_model_slices' own defaults), passed through
            to every _plot_slice() call (MOD_QC and MOD_STATS). Axis tick
            labels, axis labels, panel titles, and colourbar text were
            previously fixed at plot_model_slices' internal defaults with
            no way to override them from this script.
"""
from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

import numpy as np

import sklearn.covariance
import scipy.sparse as scs

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import ensembles as ens
import util as utl
from version import versionstrg

try:
    import femtic_viz as fviz
except ImportError:
    fviz = None

rng = np.random.default_rng()
nan = np.nan

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# Configuration
# ===========================================================================
FEMTIC="4.3"
# ---------------------------------------------------------------------------
# Ensemble input
# ---------------------------------------------------------------------------

# ENSEMBLE_DIR = r"/home/vrath/Py4MTX/py4mt/data/ensembles/misti/ensemble/"
#ENSEMBLE_DIR = r"/media/vrath/LargeBack/misti/ensemble/"
ENSEMBLE_DIR = r"//FEMTIC_work/Ensembles/misti_gst/ensemble/"
ENSEMBLE_NAME = "misti_gst_suzuki_"
# ENSEMBLE_DIR = r"/home/vrath/Py4MTX/py4mt/data/ensembles/misti/ensemble/"
# ENSEMBLE_NAME = "misti_gst_suzuki_"

#: Prefix used for .npz output keys and default file/figure names.
#: e.g. "rto" → keys rto_ens, rto_avg, …  and file RTO_results.npz.
ENSEMBLE_PREFIX = "Misti_gstat"

#: Maximum normalised RMS accepted from femtic.cnv.
NRMS_MAX = 1.5

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
#: Percentile levels. Default: 2-σ / 1-σ normal-equivalent.
PERCENTILES = [2.3, 15.9, 50.0, 84.1, 97.7]

#: Percentile-pair differences to compute as extra spread statistics, e.g.
#: (15.9, 84.1) -> the 1-sigma-equivalent interquantile range
#: |P84.1 - P15.9|ᵢ per free parameter (an outlier-robust alternative to
#: VAR/MAD). Each value in a pair must also appear in PERCENTILES. Stored
#: in the .npz as f"{P}_qdiff_<lo>_<hi>" and available for MOD_STATS
#: plotting under the same key.
QDIFF_PAIRS = [(15.9, 84.1)]

# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
#: Set False to skip covariance estimation entirely.  Mean/var/median/MAD/
#: percentiles and slice plots are unaffected — only the *_cov* /
#: *_cov_eigval* / *_cov_eigvec* keys are omitted from the .npz output.
#: Skipping is worthwhile whenever the covariance itself isn't needed
#: downstream (e.g. plain ensemble statistics runs), since it is by far
#: the most expensive step for large meshes.
COMPUTE_COV = True

#: "full"     — dense empirical covariance (n_free x n_free) via
#:              sklearn.covariance.empirical_covariance.  Cost
#:              O(n_free^2 * n_members) time, O(n_free^2) memory — fine
#:              for a few thousand free parameters, prohibitive beyond
#:              that (e.g. n_free=1e5 -> 80 GB just to store it).
#: "low_rank" — thin SVD of the centred ensemble matrix instead.  Since
#:              the empirical covariance of n_members samples has rank
#:              <= n_members-1, this is an *exact* factorisation (not an
#:              approximation) whenever n_members << n_free, which is the
#:              usual case here.  Cost drops to
#:              O(n_members^2 * n_free) time and O(n_members * n_free)
#:              memory — orders of magnitude cheaper.  Stores
#:              f"{P}_cov_eigval" (r,) and f"{P}_cov_eigvec" (n_free, r)
#:              with r = n_members instead of a dense f"{P}_cov"; the
#:              full covariance can be reconstructed exactly as
#:              eigvec @ diag(eigval) @ eigvec.T.  Same spirit as the
#:              randomized-SVD-on-R approach used for low-rank prior
#:              sampling in ensembles.py.
COV_METHOD = "full"

#: Sparsify the dense covariance (COV_METHOD="full" only). Ignored for
#: "low_rank", which is already a compact factorisation.
SPARSIFY     = True
SPARSE_THRESH = 1.0e-8   # relative threshold for zeroing small entries

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
ENSEMBLE_RESULTS = ENSEMBLE_DIR + ENSEMBLE_PREFIX.upper() + "_results.npz"

# ---------------------------------------------------------------------------
# Mesh (required for any slice plot)
# ---------------------------------------------------------------------------
MOD_MESH = ENSEMBLE_DIR + "templates/mesh.dat"

# --- Ocean / air handling (must match the inversion setup) ----------------
MOD_OCEAN     = None
MOD_AIR_RHO   = 1.0e9   # Ω·m  (region 0)
MOD_OCEAN_RHO = 0.25    # Ω·m  (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# QC slice plot — best-nRMS converged member
# ---------------------------------------------------------------------------
#: Set True to plot the best-nRMS member.
MOD_QC      = False
MOD_QC_FILE = ENSEMBLE_DIR + ENSEMBLE_PREFIX + "_qc.pdf"

# ---------------------------------------------------------------------------
# Statistics slice plots — mean / variance / median / MAD
# ---------------------------------------------------------------------------
#: Set True to write derived stat members as block files and plot them.
#: Requires MOD_MESH and a valid template file (taken from best member).
MOD_STATS      = False
#: Which statistics to plot.  Subset of: "avg", "var", "med", "mad", plus
#: one auto-generated key per PERCENTILES level (e.g. 2.3 -> "p2_3",
#: 50.0 -> "p50", 97.7 -> "p97_7") and one per QDIFF_PAIRS entry (e.g.
#: (15.9, 84.1) -> "qdiff_15_9_84_1"). Default below includes all of them.
MOD_STATS_WHAT = ["avg", "var", "med", "mad"] + [
    "p" + f"{_p:g}".replace(".", "_") for _p in PERCENTILES
] + [
    f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_") for _lo, _hi in QDIFF_PAIRS
]
#: Output directory for stat block files and figures.
MOD_STATS_DIR  = ENSEMBLE_DIR + "/stats_plots/"

#: Per-statistic colour-scale override, keyed the same as MOD_STATS_WHAT
#: (e.g. "var", "p50", "qdiff_15_9_84_1"). Each value is an explicit
#: [vmin, vmax] pair, or None for automatic per-panel scaling from that
#: statistic's own data range. AVG/MED and the percentile fields aren't
#: listed here — they fall back to MOD_CLIM automatically (same log10(Ω·m)
#: space as the model itself). VAR/MAD/QDIFF are spread statistics on a
#: completely different, typically much narrower scale, so they get their
#: own fixed range below rather than auto-scaling per panel — set to
#: [-2, 2] as a sensible starting range; adjust per-key, or set a key to
#: None to fall back to auto-scaling for that one statistic.
MOD_STATS_CLIM = {
    "var": [-2.0, 2.0],
    "mad": [-2.0, 2.0],
}
for _lo, _hi in QDIFF_PAIRS:
    MOD_STATS_CLIM[f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_")] = [-2.0, 2.0]

# ---------------------------------------------------------------------------
# Shared slice / plot parameters
# (identical config surface to femtic_gst_prep.py / femtic_rto_prep.py /
#  femtic_mod_plot_slice.py — used by both MOD_QC and MOD_STATS below)
# ---------------------------------------------------------------------------

# --- Geographic / UTM origin of the mesh centre ----------------------------
#: Set to None when MOD_ORIGIN_METHOD will estimate the origin from MOD_SITE_DAT.
MOD_UTM_ORIGIN_LAT    = None   # decimal degrees, positive = North
MOD_UTM_ORIGIN_LON    = None   # decimal degrees, positive = East
MOD_UTM_ORIGIN_E      = None   # UTM easting  [m]
MOD_UTM_ORIGIN_N      = None   # UTM northing [m]
MOD_UTM_ZONE_OVERRIDE = None   # override auto-derived zone; None = auto

#: "box"     → midpoint of UTM bounding box of all sites in MOD_SITE_DAT.
#: "average" → arithmetic mean of UTM coordinates in MOD_SITE_DAT.
#: None      → use the hard-coded literals above.
MOD_ORIGIN_METHOD = "box"

# --- Display coordinate system ---------------------------------------------
#: "model"  — axis ticks in model-local metres (default)
#: "utm"    — axis ticks in absolute UTM metres
#: "latlon" — axis ticks in decimal degrees
MOD_DISPLAY_COORDS = "model"

# --- Site overlay ------------------------------------------------------------
#: Primary source: mt_make_sitelist.py CSV (name,lat,lon,elev,sitenum,E,N).
#: Set to None to fall back to observe.dat / MOD_SITE_NUMBER.
MOD_SITE_DAT    = ENSEMBLE_DIR + "templates/site.dat"
MOD_SITE_NAMES  = None   # list of names to plot, or None = all sites
#: Fallback: site number(s) from observe.dat (int or list of ints).
MOD_SITE_NUMBER = None

MOD_PLOT_SITES_MAPS   = True    # show markers on map panels
MOD_PLOT_SITES_SLICES = False   # show markers on curtain / plane panels
#: Max distance [m] from a curtain plane for a site to appear on it.
MOD_PROJECTION_DIST = 5000.0    # metres; None = show all sites on every panel

MOD_SITE_MARKER        = dict(marker="v", color="black", ms=8, zorder=10, label=None)
MOD_SITE_MARKER_SLICES = None
#: Extra point markers on map panels only (each dict: latlon, marker, color, ms, name).
MOD_MAP_MARKERS = []

# --- Slice specification ----------------------------------------------------
#: Slice positions accept plain floats (model-local m) or CRS-tagged tuples:
#:   (value, "utm") | (value, "latlon")
#: Depth z0 is always model-local metres (no CRS tagging).
MOD_SLICES = [
    dict(kind="map", z0=5000.0),
    dict(kind="map", z0=15000.0),
    dict(kind="ns",  x0=0.0),
    dict(kind="ew",  y0=0.0),
]
MOD_XLIM = None    # [xmin, xmax] model-local metres; None = auto
MOD_YLIM = None    # [ymin, ymax] model-local metres; None = auto
MOD_ZLIM = None    # [zmin, zmax] model-local metres; None = auto

# --- Region of interest (auto xlim/ylim/zlim from site positions) ----------
#: When True and site positions are available (MOD_SITE_DAT / MOD_SITE_NUMBER,
#: subject to MOD_PLOT_SITES_MAPS/SLICES as usual), MOD_XLIM/MOD_YLIM are
#: derived automatically from the site bounding box + MOD_ROI_PAD_XY, and
#: MOD_ZLIM is set from MOD_ROI_ZLIM -- overriding the literal MOD_XLIM/
#: MOD_YLIM/MOD_ZLIM values above. Falls back to those literals (or to
#: full-mesh auto-scaling if they're also None) when no sites are found.
#: Also drives the per-panel aspect-ratio sizing below (MOD_PANEL_WIDTH),
#: since that sizing needs an actual extent to compute widths from.
MOD_ROI_AUTO   = True
MOD_ROI_PAD_XY = 5000.0             # metres of padding around the site bbox
MOD_ROI_ZLIM   = [0.0, 20000.0]     # depth range (m, positive-down) for ns/ew/plane panels; None = leave MOD_ZLIM as-is

MOD_DPI         = 600            # figure DPI, used by both MOD_QC and MOD_STATS
MOD_CMAP        = "turbo_r"
MOD_CLIM        = [0.0, 4.0]     # [log10_min, log10_max] Ω·m; None = auto
MOD_OCEAN_COLOR = "lightgrey"    # flat colour for ocean cells; None = colormap
MOD_AIR_COLOR   = "whitesmoke"
MOD_AIR_BGCOLOR = None           # axes facecolor for air; None = figure default

# --- Alpha / blanking by second block file (optional) -----------------------
MOD_ALPHA_FILE         = None    # path to sensitivity block; None = disabled
MOD_ALPHA_MODE         = "fade"  # "fade" | "blank"
MOD_ALPHA_BLANK_THRESH = 0.0

# --- Figure layout -----------------------------------------------------------
MOD_EQUAL_ASPECT = True
MOD_DEPTH_KM     = True
MOD_HORIZ_KM     = True
#: 2x2 grid matching the 4 default MOD_SLICES panels (2 maps + ns + ew).
#: Adjust to len(MOD_SLICES) if you change the number of panels; None/None
#: falls back to a single row of len(MOD_SLICES) columns.
MOD_NROWS        = 2      # None = auto (1 row)
MOD_NCOLS        = 2      # None = auto (len(MOD_SLICES) cols)
MOD_PANEL_HEIGHT = 16.0   # cm
#: None = auto per-column width from each panel's own aspect ratio (needs
#: MOD_EQUAL_ASPECT=True and real xlim/ylim/zlim -- supplied automatically
#: by MOD_ROI_AUTO above -- so map, ns, and ew panels naturally end up
#: different widths instead of being forced square).
MOD_PANEL_WIDTH  = None   # cm; None = auto from aspect ratio
MOD_FIGSIZE      = None   # [w, h] cm; overrides auto when set

#: Axis annotation font sizes, passed through to fviz.plot_model_slices.
#: Defaults match plot_model_slices' own defaults.
MOD_TICK_FONTSIZE  = 7    # axis tick labels, colourbar ticks
MOD_LABEL_FONTSIZE = 8    # axis labels, panel titles, colourbar label

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ===========================================================================
# Helpers
# ===========================================================================

def _resolve_origin_and_sites():
    """Estimate UTM origin from MOD_SITE_DAT; collect site model-local coords.

    Mirrors the origin-resolution block in femtic_gst_prep.py /
    femtic_rto_prep.py so all three scripts behave identically, including
    the observe.dat / MOD_SITE_NUMBER fallback when MOD_SITE_DAT is absent.

    Returns
    -------
    utm_e, utm_n, utm_lat, utm_lon : float | None
    utm_zone, utm_northern : str | None, bool | None
    site_xys : list of (name, x_m, y_m, elev)
    obs_coords_only : bool
        True if site_xys was populated from observe.dat / MOD_SITE_NUMBER
        rather than MOD_SITE_DAT.
    """
    _e   = MOD_UTM_ORIGIN_E
    _n   = MOD_UTM_ORIGIN_N
    _lat = MOD_UTM_ORIGIN_LAT
    _lon = MOD_UTM_ORIGIN_LON
    _zone, _north = None, None

    if MOD_ORIGIN_METHOD is not None and MOD_SITE_DAT and os.path.isfile(MOD_SITE_DAT):
        _sdat = fem.read_site_dat(MOD_SITE_DAT)
        if _sdat:
            _Es  = np.array([d["easting"]  for d in _sdat])
            _Ns  = np.array([d["northing"] for d in _sdat])
            if MOD_ORIGIN_METHOD == "box":
                _e = 0.5 * (_Es.min() + _Es.max())
                _n = 0.5 * (_Ns.min() + _Ns.max())
            elif MOD_ORIGIN_METHOD == "average":
                _e = float(_Es.mean())
                _n = float(_Ns.mean())
            _lats = np.array([d["lat"] for d in _sdat])
            _lons = np.array([d["lon"] for d in _sdat])
            _zone, _north = utl.utm_zone_from_latlon(
                float(_lats.mean()), float(_lons.mean()),
                override=MOD_UTM_ZONE_OVERRIDE,
            )
            _lat, _lon = utl.utm_to_latlon_zn(_e, _n, _zone, _north)

    if _lat is not None and _lon is not None:
        _zone, _north = utl.utm_zone_from_latlon(
            _lat, _lon, override=MOD_UTM_ZONE_OVERRIDE
        )

    site_xys = []
    obs_coords_only = False
    _need_sites = MOD_PLOT_SITES_MAPS or MOD_PLOT_SITES_SLICES
    if _need_sites and MOD_SITE_DAT and os.path.isfile(MOD_SITE_DAT):
        for row in fem.read_site_dat(MOD_SITE_DAT, site_names=MOD_SITE_NAMES):
            sx, sy = fem.utm_to_model(
                row["easting"], row["northing"], _e, _n
            )
            site_xys.append(
                (row["name"], sx, sy, float(row.get("elev", 0.0)))
            )
    elif _need_sites and MOD_SITE_NUMBER is not None:
        _obs_file = ENSEMBLE_DIR + "templates/observe.dat"
        _site_nums = (MOD_SITE_NUMBER if isinstance(MOD_SITE_NUMBER, (list, tuple))
                      else [MOD_SITE_NUMBER])
        for _sn in _site_nums:
            sx, sy = fem.read_site_position(_obs_file, _sn)
            site_xys.append((_sn, sx, sy, 0.0))
        obs_coords_only = True

    return _e, _n, _lat, _lon, _zone, _north, site_xys, obs_coords_only


def _plot_slice(block_file: str, pdf_file: str,
                utm_e, utm_n, utm_lat, utm_lon,
                utm_zone, utm_north, site_xys: list,
                obs_coords_only: bool = False,
                clim=None) -> None:
    """Call fviz.plot_model_slices with the shared MOD_* config.

    Mirrors the plotting call in femtic_gst_prep.py / femtic_rto_prep.py
    exactly, so QC and statistics figures use the same options (CRS
    handling, site overlay, alpha/blanking, figure layout) as the
    ensemble-generation scripts.

    Parameters
    ----------
    clim : [vmin, vmax] | None
        Per-call colour-scale override. ``None`` (default) falls back to
        the module-level ``MOD_CLIM``, unchanged from previous behaviour.
        Used by the MOD_STATS block to give VAR/MAD/QDIFF panels their own
        automatic scale instead of forcing MOD_CLIM onto them.
    """
    if fviz is None:
        print("  plot_slice: femtic_viz not available — skipping.")
        return

    _clim = MOD_CLIM if clim is None else clim

    _slices_resolved = fem.resolve_slice_positions(
        MOD_SLICES, utm_zone, utm_north,
        utm_e, utm_n, utm_lat, utm_lon,
        verbose=OUT,
    )
    fviz.plot_model_slices(
        model_file          = block_file,
        mesh_file           = MOD_MESH,
        slices              = _slices_resolved,
        cmap                = MOD_CMAP,
        clim                = _clim,
        xlim                = MOD_XLIM,
        ylim                = MOD_YLIM,
        zlim                = MOD_ZLIM,
        ocean_color         = MOD_OCEAN_COLOR,
        ocean_value         = MOD_OCEAN_RHO,
        air_color           = MOD_AIR_COLOR,
        air_bgcolor         = MOD_AIR_BGCOLOR,
        site_xys            = site_xys,
        obs_coords_only     = obs_coords_only,
        sites_in_maps       = MOD_PLOT_SITES_MAPS,
        sites_in_slices     = MOD_PLOT_SITES_SLICES,
        site_marker         = MOD_SITE_MARKER,
        site_marker_slices  = MOD_SITE_MARKER_SLICES,
        map_markers         = MOD_MAP_MARKERS,
        projection_dist     = MOD_PROJECTION_DIST,
        display_coords      = MOD_DISPLAY_COORDS,
        utm_origin_e        = utm_e,
        utm_origin_n        = utm_n,
        utm_zone            = utm_zone,
        utm_northern        = utm_north,
        utm_to_latlon_fn    = utl.utm_to_latlon_zn,
        latlon_to_model_fn  = fem.latlon_to_model,
        depth_km            = MOD_DEPTH_KM,
        horiz_km            = MOD_HORIZ_KM,
        equal_aspect        = MOD_EQUAL_ASPECT,
        panel_height        = MOD_PANEL_HEIGHT / 2.54,
        panel_width         = MOD_PANEL_WIDTH / 2.54 if MOD_PANEL_WIDTH is not None else None,
        figsize             = [v / 2.54 for v in MOD_FIGSIZE] if MOD_FIGSIZE is not None else None,
        nrows               = MOD_NROWS,
        ncols               = MOD_NCOLS,
        tick_fontsize       = MOD_TICK_FONTSIZE,
        label_fontsize      = MOD_LABEL_FONTSIZE,
        alpha_file          = MOD_ALPHA_FILE,
        alpha_mode          = MOD_ALPHA_MODE,
        alpha_blank_thresh  = MOD_ALPHA_BLANK_THRESH,
        plot_file           = pdf_file,
        dpi                 = MOD_DPI,
        out                 = OUT,
    )
    if OUT:
        print(f"  saved → {pdf_file}")


# ===========================================================================
# Main
# ===========================================================================

# --- (1) Scan ensemble directories ----------------------------------------
dir_list = utl.get_filelist(
    searchstr=[ENSEMBLE_NAME+"*"],
    searchpath=ENSEMBLE_DIR,
    fullpath=True,
)
print(f"Found {len(dir_list)} sub-directory/ies matching '{ENSEMBLE_NAME}'.")

model_list  = []          # list of [block_file, n_iter, nRMS]
model_count = 0
ens_matrix  = None        # will become (n_members, n_free) float64

for d in dir_list:
    print(f"\n  Inversion run: {d}")
    cnv_file = os.path.join(d, "femtic.cnv")
    if not os.path.isfile(cnv_file):
        print(f"    femtic.cnv not found — skipped.")
        continue

    with open(cnv_file) as _fh:
        cnv = _fh.readlines()
    info  = cnv[-1].split()
    if "4.3" in FEMTIC:
        numit = int(info[0])
        nrms  = float(info[6])
    elif "5." in FEMTIC:
        numit = int(info[0])
        nrms  = float(info[8])
    else:
        sys.exit("FEMTIC version"+__file__+": does not exist! Exit.")

    if nrms > NRMS_MAX:
        print(f"    nRMS={nrms:.4f} > NRMS_MAX={NRMS_MAX} — skipped.")
        continue

    mod_file = os.path.join(d, f"resistivity_block_iter{numit}.dat")
    if not os.path.isfile(mod_file):
        print(f"    {mod_file} not found — skipped.")
        continue

    print(f"    iter={numit}  nRMS={nrms:.4f}  {mod_file}")
    model_list.append([mod_file, numit, nrms])

    log_m = fem.read_model(model_file=mod_file, model_trans="log10", out=OUT)

    if ens_matrix is None:
        ens_matrix = log_m[np.newaxis, :]         # (1, n_free)
    else:
        ens_matrix = np.vstack((ens_matrix, log_m))   # (k, n_free)

    model_count += 1

n_members = model_count
print(f"\nConverged members: {n_members}")

if n_members == 0:
    sys.exit("No converged members found. Nothing to do.")

# ens_matrix shape: (n_members, n_free)
# axis=0 → reduce over members  (correct for all aggregate statistics)
# axis=1 → reduce over free parameters (was the bug in the original script)

# --- (2) Summary statistics -----------------------------------------------
P        = ENSEMBLE_PREFIX
ne       = ens_matrix.shape

ens_avg  = np.mean  (ens_matrix, axis=0)                           # (n_free,)
ens_var  = np.var   (ens_matrix, axis=0)                           # (n_free,)
ens_med  = np.median(ens_matrix, axis=0)                           # (n_free,)
ens_mad  = np.median(np.abs(ens_matrix - ens_med[np.newaxis, :]),
                     axis=0)                                        # (n_free,)
ens_prc  = np.percentile(ens_matrix, PERCENTILES, axis=0)          # (n_prc, n_free)

# --- Percentile-pair differences (robust spread, e.g. 1-sigma-equivalent IQR) ---
ens_qdiff = {}   # key -> (n_free,) array
for _lo, _hi in QDIFF_PAIRS:
    if _lo not in PERCENTILES or _hi not in PERCENTILES:
        print(f"  QDIFF_PAIRS: ({_lo}, {_hi}) not both in PERCENTILES — skipped.")
        continue
    _ilo = PERCENTILES.index(_lo)
    _ihi = PERCENTILES.index(_hi)
    _qkey = f"qdiff_{_lo:g}_{_hi:g}".replace(".", "_")
    ens_qdiff[_qkey] = np.abs(ens_prc[_ihi] - ens_prc[_ilo])       # (n_free,)

print(f"\nStatistics (over {n_members} members, {ne[1]} free parameters):")
print(f"  mean   log10(ρ): [{ens_avg.min():.3f}, {ens_avg.max():.3f}]")
print(f"  var    log10(ρ): [{ens_var.min():.4f}, {ens_var.max():.4f}]")
print(f"  median log10(ρ): [{ens_med.min():.3f}, {ens_med.max():.3f}]")
print(f"  MAD    log10(ρ): [{ens_mad.min():.4f}, {ens_mad.max():.4f}]")
for _qkey, _qval in ens_qdiff.items():
    print(f"  {_qkey}: [{_qval.min():.4f}, {_qval.max():.4f}]")

# --- (3) Empirical covariance (optional) -----------------------------------
ens_cov       = None
ens_covs      = None
ens_cov_eigval = None
ens_cov_eigvec = None

if COMPUTE_COV:
    if COV_METHOD == "low_rank":
        print("\nComputing low-rank covariance factorisation (thin SVD) …")
        _Xc = ens_matrix - ens_avg[np.newaxis, :]           # (m, n_free), centred
        _m  = _Xc.shape[0]
        # Thin SVD of the (m, n_free) centred ensemble: cost O(m^2 * n_free),
        # memory O(m * n_free) — never forms the n_free x n_free covariance.
        # C = Xc^T Xc / (m-1) = Vt.T @ diag(S^2/(m-1)) @ Vt, exactly (rank <= m-1).
        _U, _S, _Vt = np.linalg.svd(_Xc, full_matrices=False)
        ens_cov_eigval = (_S ** 2) / max(_m - 1, 1)          # (r,)  r = min(m, n_free)
        ens_cov_eigvec = _Vt.T                               # (n_free, r)
        print(f"  rank r={ens_cov_eigval.size}  "
              f"eigval range=[{ens_cov_eigval.min():.3e}, {ens_cov_eigval.max():.3e}]")
        print("  Full covariance can be reconstructed exactly as "
              "eigvec @ diag(eigval) @ eigvec.T")
    else:
        print("\nComputing empirical covariance …")
        ens_cov = sklearn.covariance.empirical_covariance(ens_matrix)

        if SPARSIFY:
            tmp    = ens_cov.copy()
            tmp[np.abs(tmp) / np.amax(np.abs(tmp)) <= SPARSE_THRESH] = 0.0
            ens_covs = scs.csr_array(tmp)
            nnz      = ens_covs.nnz
            total    = ens_cov.size
            print(f"  Sparse covariance: {nnz}/{total} non-zeros "
                  f"({100.0*nnz/total:.2f}%), threshold={SPARSE_THRESH:.1e}")
else:
    print("\nCOMPUTE_COV=False — skipping covariance estimation.")

# --- (4) Save .npz --------------------------------------------------------
ens_dict = {
    f"{P}_model_list": model_list,
    f"{P}_ens":        ens_matrix,
    f"{P}_avg":        ens_avg,
    f"{P}_var":        ens_var,
    f"{P}_med":        ens_med,
    f"{P}_mad":        ens_mad,
    f"{P}_prc":        ens_prc,
    f"{P}_prc_levels": np.asarray(PERCENTILES),
}
for _qkey, _qval in ens_qdiff.items():
    ens_dict[f"{P}_{_qkey}"] = _qval
if ens_cov is not None:
    ens_dict[f"{P}_cov"] = ens_cov
if ens_cov_eigval is not None:
    ens_dict[f"{P}_cov_eigval"] = ens_cov_eigval
    ens_dict[f"{P}_cov_eigvec"] = ens_cov_eigvec

np.savez_compressed(ENSEMBLE_RESULTS, **ens_dict)
print(f"\nResults saved → {ENSEMBLE_RESULTS}")

# --- (5) Resolve UTM origin and sites (needed for any plot) ---------------
_need_plot = MOD_QC or MOD_STATS
if _need_plot:
    (utm_e, utm_n, utm_lat, utm_lon,
     utm_zone, utm_north, site_xys, obs_coords_only) = _resolve_origin_and_sites()

    # --- Region of interest: override MOD_XLIM/YLIM/ZLIM from site bbox ---
    if MOD_ROI_AUTO and site_xys:
        _sx = np.array([s[1] for s in site_xys])
        _sy = np.array([s[2] for s in site_xys])
        MOD_XLIM = [float(_sx.min() - MOD_ROI_PAD_XY), float(_sx.max() + MOD_ROI_PAD_XY)]
        MOD_YLIM = [float(_sy.min() - MOD_ROI_PAD_XY), float(_sy.max() + MOD_ROI_PAD_XY)]
        if MOD_ROI_ZLIM is not None:
            MOD_ZLIM = list(MOD_ROI_ZLIM)
        print(f"\nROI (from {len(site_xys)} sites, pad={MOD_ROI_PAD_XY:.0f} m):")
        print(f"  MOD_XLIM = {MOD_XLIM}")
        print(f"  MOD_YLIM = {MOD_YLIM}")
        print(f"  MOD_ZLIM = {MOD_ZLIM}")
    elif MOD_ROI_AUTO:
        print("\nROI: MOD_ROI_AUTO=True but no sites available — "
              "using literal MOD_XLIM/MOD_YLIM/MOD_ZLIM instead.")

# --- (6) QC slice plot — best-nRMS member ---------------------------------
if MOD_QC:
    if fviz is None:
        print("\n  MOD_QC: femtic_viz not available — skipping.")
    elif not model_list:
        print("\n  MOD_QC: no converged members — skipping.")
    else:
        _best      = min(model_list, key=lambda x: x[2])
        _best_file, _best_iter, _best_nrms = _best
        print(f"\nQC: best member  nRMS={_best_nrms:.4f}  "
              f"iter={_best_iter}")
        _plot_slice(
            block_file      = _best_file,
            pdf_file        = MOD_QC_FILE,
            utm_e           = utm_e,
            utm_n           = utm_n,
            utm_lat         = utm_lat,
            utm_lon         = utm_lon,
            utm_zone        = utm_zone,
            utm_north       = utm_north,
            site_xys        = site_xys,
            obs_coords_only = obs_coords_only,
        )

# --- (7) Statistics slice plots -------------------------------------------
if MOD_STATS:
    if fviz is None:
        print("\n  MOD_STATS: femtic_viz not available — skipping.")
    elif not model_list:
        print("\n  MOD_STATS: no converged members — skipping.")
    else:
        os.makedirs(MOD_STATS_DIR, exist_ok=True)

        # Template = lowest-nRMS member (preserves header / flag columns)
        _best_file = min(model_list, key=lambda x: x[2])[0]

        _stat_map = {
            "avg": (ens_avg, "mean"),
            "var": (ens_var, "variance"),
            "med": (ens_med, "median"),
            "mad": (ens_mad, "MAD"),
        }
        # "Value-scale" keys share the model's own log10(Ω·m) range and
        # default to MOD_CLIM; everything else ("var", "mad", "qdiff_*")
        # is a spread statistic on a different scale and defaults to None
        # (auto per-panel). MOD_STATS_CLIM always takes precedence.
        _value_scale_keys = {"avg", "med"}

        # One entry per PERCENTILES level, keyed e.g. 2.3 -> "p2_3", 50.0 -> "p50".
        for _i, _pval in enumerate(PERCENTILES):
            _pkey = "p" + f"{_pval:g}".replace(".", "_")
            _stat_map[_pkey] = (ens_prc[_i], f"{_pval:g}th percentile")
            _value_scale_keys.add(_pkey)

        # One entry per QDIFF_PAIRS entry (spread statistic — auto scale).
        for _qkey, _qval in ens_qdiff.items():
            _stat_map[_qkey] = (_qval, f"|{_qkey}| spread")

        for _key in MOD_STATS_WHAT:
            if _key not in _stat_map:
                print(f"  MOD_STATS: unknown stat '{_key}' — skipped.")
                continue
            _vec, _label = _stat_map[_key]
            _default_clim = MOD_CLIM if _key in _value_scale_keys else None
            _clim = MOD_STATS_CLIM.get(_key, _default_clim)
            _block_out = os.path.join(
                MOD_STATS_DIR,
                f"resistivity_block_{P}_{_key}.dat",
            )
            _pdf_out = os.path.join(
                MOD_STATS_DIR,
                f"{P}_{_key}.pdf",
            )
            print(f"\nSTATS: writing {_label} → {_block_out}")
            fem.insert_model(
                template   = _best_file,
                model      = _vec,
                model_file = _block_out,
                ocean      = MOD_OCEAN,
                air_rho    = MOD_AIR_RHO,
                ocean_rho  = MOD_OCEAN_RHO,
                out        = OUT,
            )
            print(f"STATS: plotting {_label} → {_pdf_out}  (clim={_clim})")
            _plot_slice(
                block_file      = _block_out,
                pdf_file        = _pdf_out,
                utm_e           = utm_e,
                utm_n           = utm_n,
                utm_lat         = utm_lat,
                utm_lon         = utm_lon,
                utm_zone        = utm_zone,
                utm_north       = utm_north,
                site_xys        = site_xys,
                obs_coords_only = obs_coords_only,
                clim            = _clim,
            )

print("\nfemtic_ens_post.py complete.")
