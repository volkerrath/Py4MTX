#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_misfit.py

Data-misfit diagnostics for an ENSEMBLE of FEMTIC runs (ens family).

All entries of RUN_PATHS (run directories with results_iter*.h5, results
files, or globs) form one ensemble: the chosen iteration of each run is one
member, members are matched row by row (datatype, site, component, frequency).
Methods (METHODS):

  "crossplot"  observed vs. calculated, one panel per data type / component
  "histogram"  normalized-residual histograms in frequency (or period) bands,
               RMS1 and RMS2 after Baba (2023), with the synthesized error
               taken from the spread of the members
  "qq"         normal Q-Q plots of the same residuals in the same bands
  "curves"     optional: response curves per site; the members drawn as thin
               curves, as a per-period density, or as percentile bands
  "nrms"       histogram of the best nRMS of ALL scanned members (accepted /
               rejected), from the results .npz of femtic_ens_post.py; needs
               no results_iterN.h5

The single-model counterpart is femtic_mod_misfit.py. Both call
femtic_misfit.py (the engine of femtic_data_misfit.py, which stays as it is
and does both jobs in one script).

Provenance
----------
Author      : Claude (Anthropic)
Generated   : 2026-11-09 (from femtic_data_misfit.py, which is unchanged)
Notice      : This code is AI-generated. Review and test it before any
              production use.

Reference
---------
Baba, K. (2023): A simple method to evaluate the uncertainty of
magnetotelluric forward modeling for practical three-dimensional
conductivity structure models. Earth, Planets and Space 75, 67.
https://doi.org/10.1186/s40623-023-01832-5

Usage
-----
    python femtic_ens_misfit.py                         # settings below
    python femtic_ens_misfit.py "./ens/member_*/"       # paths from the command line
    python femtic_ens_misfit.py ./m01/ ./m02/ ./m03/

Quote globs so the shell does not expand them (the script expands them).
Output: one set of figures and tables, prefix ensemble_M<M> (OUT_PREFIX),
written into the common parent directory of the members (OUT_DIR forces one
directory; PLOT_DIR separates the figures from the tables).

Changelog
---------
2026-11-09  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            created. Ensemble half of femtic_data_misfit.py (RUN_MODE = "ensemble"): same
            settings, same output; RUN_MODE is gone, the script is the ensemble mode.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# py4mt environment: make the modules importable (femtic_misfit.py, femtic.py)
# ---------------------------------------------------------------------------
PY4MTX_ROOT = os.environ.get("PY4MTX_ROOT", "")
if PY4MTX_ROOT:
    for _pth in (os.path.join(PY4MTX_ROOT, "py4mt", "modules"),
                 os.path.join(PY4MTX_ROOT, "py4mt", "scripts")):
        if os.path.isdir(_pth) and _pth not in sys.path:
            sys.path.insert(0, _pth)

import femtic_misfit as fm  # noqa: E402

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
# --- what to run ---
# All entries of RUN_PATHS together form ONE ensemble: the chosen iteration of
# every run is one member (members must share the same data rows).
RUN_PATHS: List[str] = [
    r"/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/ann*/"
    ]        # run dirs, results files, or globs (overridden by argv)

ITERATION: Union[str, int] = "best"      # "best" | "last" | integer
ITER_PATTERN: str = "results_iter*.h5"   # searched in each run directory
METHODS: Tuple[str, ...] = ("crossplot", "histogram", "qq", "curves", "nrms", )
#   "curves" is optional; "nrms" needs NRMS_RESULTS or NRMS_CSV (see below)

# --- ensemble handling ---
ENSEMBLE_OBS: str = "mean"     # observed values/errors: "mean" over members
                               # (RTO-perturbed data average back to the
                               # original) or "first" member
REF_MEMBER: Optional[int] = None   # RMS1 reference: None = ensemble mean;
                                   # integer = calculated response of that
                                   # member (as in Baba's "6th model")

# --- sites ---
SITE_MERGE_BY_POSITION: bool = True   # data types often number the same station
                                      # differently (e.g. MT 1..48, VTF 1001..1048).
                                      # True: all rows at the same position get the
                                      # site_id of the FIRST occurrence of that
                                      # position (row order of the results file,
                                      # usually the impedance data): VTF 1001 -> 1.
                                      # Affects the per-site table and the "curves"
                                      # figures (one figure per position, all data
                                      # types together); not the other methods
SITE_MERGE_TOL: float = 1.0           # [m] positions closer than this are the same

# --- output ---
OUT_DIR: Optional[str] = (
    "/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/results256/"
)
# None      # None -> into the run directory (single:
                                   # each run's own; ensemble: common parent
                                   # of the member directories); or a path
OUT_PREFIX: Optional[str] = "Misfits"  # None -> derived from run / ensemble;
                                   # if set with several singles it is
                                   # prepended: <OUT_PREFIX>_<run>_iter<N>
PLOT_FORMATS: Tuple[str, ...] = (".pdf",)   # a tuple: note the comma!
PLOT_DPI: int = 300
PLOT_DIR: Optional[str] = None  # where the figures go (tables stay in
                            # the output directory). None = same directory
                            # as the tables; a relative name = sub-directory
                            # of the output directory (per run); an absolute
                            # path = that one directory for all figures
                            # (file names stay distinct via the prefix)
WRITE_STATS: bool = True
FIT_MEASURES: bool = True   # overall table of fit measures (per data type
                            # and pooled) -> <prefix>_fit_measures.txt
SHOW: bool = False

# --- selection (None = everything) ---
DATATYPES: Optional[Sequence[str]] = None     # e.g. ["MT", "VTF"]
COMPONENTS: Optional[Sequence[str]] = None    # e.g. ["Zxy", "Zyx"]

# --- crossplot ---
NCOLS: int = 4
PANEL_SIZE: float = 3.3
EQUAL_ASPECT: bool = True
CMAP: str = "turbo"
COLOR_RANGE: str = "global"   # "global" | "panel" (range of log10 f)
MARKER_SIZE: float = 14.0
ALPHA: float = 0.85
LOG_RHO: bool = True          # log-log axes for rho* components
SHOW_ERRORBARS: bool = False

# --- residual histograms ---
HIST_BAND_AXIS: str = "frequency"   # "frequency" | "period"
HIST_BANDS_PER_DECADE: int = 2
HIST_BY_COMPONENT: bool = False     # False: one figure per data type
                                    # (components pooled, as in Baba 2023)
HIST_LIMIT: float = 10.0            # histogram range +/- (values outside
                                    # are counted, not drawn)
HIST_BIN_WIDTH: float = 0.5
HIST_MIN_N: int = 5                 # skip bands with fewer values
HIST_NCOLS: int = 4
HIST_PANEL_SIZE: float = 3.0

# --- Q-Q plots (same normalized residuals as the histograms) ---
QQ_BAND_AXIS: str = "frequency"     # "frequency" | "period"
QQ_BANDS_PER_DECADE: int = 2
QQ_BY_COMPONENT: bool = False       # False: per data type, components pooled
QQ_LIMIT: float = 10.0              # max |sample quantile| shown
QQ_EQUAL_AXES: bool = True         # True: same range on both axes, square
                                    # panels (1:1 line at 45 degrees)
QQ_CONFIDENCE: Optional[float] = 0.95   # pointwise envelope; None = off
QQ_MIN_N: int = 10                  # skip bands with fewer values
QQ_NCOLS: int = 4
QQ_PANEL_SIZE: float = 3.0

# --- ensemble nRMS histogram (method "nrms") ---
# Needs only the results of femtic_ens_post.py; no results_iterN.h5, and
# RUN_MODE / RUN_PATHS are not used by this method. Output: <OUT_DIR or the
# directory of NRMS_RESULTS>/<OUT_PREFIX or the ensemble prefix>_nrms_hist.<ext>
# (figure directory: PLOT_DIR; formats and DPI: PLOT_FORMATS, PLOT_DPI).
NRMS_RESULTS: Optional[str] = (
    "/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/"
    "results256/ANNECY_RTO_ORIG-ERR_results256.npz"
)   # results .npz of femtic_ens_post.py (holds the nRMS table of ALL scanned
    # members, <P>_nrms_table); None = use NRMS_CSV only
NRMS_NPZ_PREFIX: Optional[str] = None   # key prefix inside the .npz; None = the
                                        # file's only ensemble
NRMS_CSV: Optional[str] = None   # fallback for results files written before
                                 # 2026-11-08 (no table inside): the CSV
                                 # <ENSEMBLE_PREFIX>_nrms.csv of femtic_ens_post.py.
                                 # If neither gives a table, only the accepted
                                 # members of the .npz are shown
NRMS_MAX: Optional[float] = None   # acceptance threshold line; None = the
                                   # value stored in the .npz (NRMS_MAX of
                                   # femtic_ens_post.py), none if absent
NRMS_BINS: Union[int, str] = "auto"   # an int, or a numpy rule ("auto", "fd", ...)
NRMS_HIST_FILE: Optional[str] = None   # extension-less output path;
                                       # None = <figure dir>/<prefix>_nrms_hist
NRMS_TICK_FONTSIZE: int = 16
NRMS_LABEL_FONTSIZE: int = 16
NRMS_TITLE_FONTSIZE: Optional[int] = None   # None = Matplotlib's default title size

# --- response curves (optional method "curves"; add it to METHODS) ---
# Needs data_viz.py on the path. One figure per site, one panel per entry of
# CURVES_WHAT; PLOT_FORMATS/PLOT_DPI/OUT_DIR/OUT_PREFIX apply as usual.
CURVES_WHAT: Tuple[str, ...] = ("rho", "phase", "tipper")  # + "pt"
CURVES_COMPS: str = (
    "xx, xy, yx, yy"  # impedance components (rho and phase panels)
)
CURVES_SITE_DAT: Optional[str] = (
    "/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_rto_orig-err/templates/site.dat"
)
# "auto"   # site.dat from mt_make_sitelist.py
# (name,lat,lon,elev,sitenum,E,N): gives the
# site names. "auto" = look for site.dat in
# the results directory and up to two
# levels above; a path; None = no names
CURVES_SITE_ID_OFFSET: int = 0  # site_id in the results file = sitenum
# in site.dat + this offset (only used by
# CURVES_SITE_MATCH = "id" and as the fallback)
CURVES_SITE_MATCH: str = "auto"  # how the site.dat names are assigned to the
# site_id values of the results file:
# "auto"   by coordinates (site_x / site_y of the
#          results file against easting / northing of
#          site.dat; a common shift is found by the
#          script); by id (sitenum + offset) only if
#          the results file has no coordinates
# "coords" by coordinates only
# "id"     by sitenum + CURVES_SITE_ID_OFFSET only
CURVES_SITE_TOL: float = 15.0   # [m] largest distance for a coordinate match
CURVES_SITES: Union[None, int, Sequence[Union[int, str]]] = None
# None = all sites; int N = N sites (chosen
# by CURVES_SITE_ORDER); list = these
                                   # site_id values and/or site names
CURVES_SITE_ORDER: str = "even"    # for an int CURVES_SITES: "even" (spread
                                   # over the site list), "id" (first N),
                                   # "worst" / "best" (by site nRMS)
CURVES_MAX_SITES: int = 200        # safety cap on the number of figures

# ensemble display (ignored for a single run: observed + calculated line)
CURVES_ENS_MODE: str = "auto"      # "auto" | "curves" | "density" | "bands",
                                   # or "+"-joined, e.g. "density+bands"
CURVES_MAX_CURVES: int = 50        # auto: curves up to this many members,
                                   # density above
CURVES_SUBSAMPLE: int = 100        # "curves": at most this many members drawn
CURVES_MEMBER_ALPHA: Optional[float] = None   # None = from number of curves
CURVES_MEDIAN: bool = True         # ensemble median on top
CURVES_BAND_PCT: Tuple[float, ...] = (5.0, 25.0)  # bands: p..100-p percentiles
CURVES_DENSITY_NX: int = 200       # density: log-period grid points
CURVES_DENSITY_NY: int = 60        # density: y bins
CURVES_DENSITY_GAMMA: float = 0.5  # < 1 makes sparse regions more visible
CURVES_SEED: int = 0               # member subsampling

# observed data
CURVES_OBS_ERRORS: str = "bar"     # "bar" | "shade" | "none"
CURVES_OBSERVED_ONLY: bool = False # True: draw only the observed data (no calculated response, no ensemble layers)
CURVES_OBS_MARKERSIZE: float = 3.5
CURVES_LINEWIDTH: float = 1.2      # calculated / median line

# axes
CURVES_INVERT_X: bool = True       # True = as data_viz (period decreasing
                                   # to the right); False = period increasing
CURVES_PERIOD_LIM: Optional[Tuple[float, float]] = None   # (Tmin, Tmax) [s]
CURVES_YLIM: Dict[str, Optional[Tuple[float, float]]] = {
    "rho": None, "phase": (-180.,+180.,), "tipper": (-0.5,+0.5,), "pt": None}
CURVES_PANEL_SIZE: Tuple[float, float] = (4.2, 3.4)       # width, height [in]
CURVES_LEGEND_LOC: Optional[str] = "below"   # "below" (under each panel), a
                                   # matplotlib loc such as "upper right", or
                                   # None for no legend. Avoid "best" with
                                   # density: matplotlib then needs seconds
                                   # per panel to place it
CURVES_MULTIPAGE: bool = True      # ".pdf": one file, one page per site;
                                   # other formats: one file per site in a
                                   # sub-directory <prefix>_curves/
CURVES_APPRES_PHASE_DEG: bool = True   # APP_RES_AND_PHS phase in degrees
                                       # (False: radians)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def settings() -> "fm.SimpleNamespace":
    """The uppercase settings of this script as a femtic_misfit settings object."""
    ours = {k: v for k, v in globals().items() if k in fm.SETTING_NAMES}
    stray = [k for k in globals() if k.isupper() and k not in fm.SETTING_NAMES
             and k not in ("PY4MTX_ROOT",)]
    if stray:
        print(f"femtic_ens_misfit: ignoring names that are no setting of "
              f"femtic_misfit: {stray}")
    return fm.make_settings(**ours)


def main(argv: Sequence[str]) -> int:
    return fm.run_ensemble(settings(), argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
