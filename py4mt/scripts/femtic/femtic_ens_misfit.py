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
               RMS1 after Baba (2023); RMS2 (with the synthesized error taken
               from the spread of the members, i.e. a modelling error) only if
               SHOW_RMS2 = True, which is not the default
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
    python femtic_ens_misfit.py "ens/member_*/"         # paths from the command line
    python femtic_ens_misfit.py m01/ m02/ m03/
    python femtic_ens_misfit.py -w /data/ens "member_*/"   # another WORK_DIR

Quote globs so the shell does not expand them (the script expands them).

Working directory
-----------------
All paths refer to WORK_DIR: RUN_PATHS, OUT_DIR, NRMS_RESULTS, NRMS_CSV,
NRMS_HIST_FILE, CURVES_SITE_DAT and the paths on the command line are taken
relative to WORK_DIR; an absolute path (or one that starts with ~) is used as
given. WORK_DIR is the setting below, or -w / --work-dir DIR on the command line
(which replaces it); None or "" = the current directory. It must exist (else
return code 2). The current directory is WORK_DIR during the run (and is
restored afterwards), so the last place of the "auto" site.dat search is
WORK_DIR. Two things are not paths into the data and stay as they were: a
relative PLOT_DIR is a sub-directory of the OUTPUT directory, not of WORK_DIR,
and PY4MTX_ROOT (environment variable; where the code is) is not touched.

Output: one set of figures and tables, prefix OUT_PREFIX (None: ensemble_M<M>),
written into the directory <common parent directory of the members>/<OUT_PREFIX>/,
made if it does not exist (OUT_DIR, if set, is used as given instead;
OUT_PREFIX_DIR = False writes into the common parent itself; PLOT_DIR separates
the figures from the tables).

Curves of chosen sites only: CURVES_SITES = ["A01", "a02", 17] (names of the
local site.dat, any case, and/or site ids) writes one file per site and format
(.pdf, .jpg, ...) into <output directory>/<prefix>_curves/.

Changelog
---------
2026-11-21b Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            "curves": optional small map of the sites in every figure, the
            current site marked, all sites coloured by their nRMS (one
            colour scale for all pages; grey and no colour bar with
            CURVES_OBSERVED_ONLY). New settings CURVES_MAP (False),
            CURVES_MAP_PLACEMENT ("right" | "left" = own column; "upper
            right" | "upper left" | "lower right" | "lower left" = inset in
            a corner of the last / first panel), CURVES_MAP_SIZE,
            CURVES_MAP_TICKS (model coordinates in km, or none),
            CURVES_MAP_AXES, CURVES_MAP_CMAP, CURVES_MAP_NRMS_RANGE and
            CURVES_MAP_MARKER_SIZE. With CURVES_MAP = False the output is
            unchanged. The work is done in femtic_misfit.py of the same date
            (see its changelog).
2026-11-21  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            all paths refer to a WORK_DIR (new setting, as in femtic_clean.py):
            relative RUN_PATHS, OUT_DIR, NRMS_RESULTS, NRMS_CSV, NRMS_HIST_FILE,
            CURVES_SITE_DAT and command-line paths are taken from it, absolute
            ones as given. New option -w / --work-dir DIR (replaces WORK_DIR).
            The script changes into WORK_DIR for the run, so the "auto" site.dat
            search ends there instead of in wherever the script was started.
            CHANGED: paths on the command line are no longer relative to the
            current directory but to WORK_DIR. The paths in the settings block
            are written relative to WORK_DIR (the ensemble directory
            krafla_ens_fine, the parent of the members); femtic_misfit.py is
            unchanged. RUN_PATHS resolves to the same members as before.
            CHANGED: OUT_DIR is "results/" under WORK_DIR (was the
            krafla_ens_fine_paired/results/ directory, now the commented
            alternative); NRMS_RESULTS (was a file in annecy2026) is None, so
            the method "nrms" is skipped, and CURVES_SITE_DAT (same) is
            "auto"; set both to files of this ensemble.
2026-11-20  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            phase curves that leave the +-180 deg window are interrupted
            and the two points joined by a dotted line (bands and density
            skip the jump). New settings CURVES_PHASE_WRAP_MODE ("direct" |
            "short" | "off"), CURVES_PHASE_WRAP_STYLE (dict of Matplotlib
            line properties; None = dotted; False = gap only) and
            CURVES_PHASE_WRAP_RANGE. The work is done in femtic_misfit.py
            and data_viz.py of the same date (see their changelogs).
2026-11-11c Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            nRMS2 switched off: no modelling error is used. New setting
            SHOW_RMS2 (default False): the crossplot panels show "nRMS" only,
            the histograms and Q-Q plots use the conventional RMS1
            normalization, and the tables drop the err+syn rows. True gives
            the former output (RMS1 and RMS2). The work is done in
            femtic_misfit.py of the same date (see its changelog).
2026-11-11b Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            titles and colour bar position, in the form of all plot scripts:
            FIGURE_TITLE ("auto" = as before; None / False / "" = none; text
            with {placeholders}; dict per kind of figure), PANEL_TITLES,
            CROSSPLOT_CBAR_LOCATION (None = at the right as before; "right",
            "left", "top", "bottom"), CROSSPLOT_CBAR_SHRINK. The work is done
            in femtic_misfit.py of the same date (see its changelog).
2026-11-11  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            CURVES_INVERT_X now defaults to False: the response curves have
            the period increasing from left to right (it was True: period
            decreasing to the right, the data_viz orientation; True still
            gives that). The band panels of the histograms / Q-Q plots with
            *_BAND_AXIS = "frequency" now run from high to low frequency
            (were low to high). The work is done in femtic_misfit.py (see its
            changelog).
2026-11-10  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            output goes to the directory <common parent>/<OUT_PREFIX>/ unless
            OUT_DIR is set (new OUT_PREFIX_DIR); CURVES_SITES list -> single
            pdf / jpg files (CURVES_MULTIPAGE = None = auto), names from the
            local site.dat in any case; new CURVES_NAME_CASE, title "site NAME
            (site#)"; std, skew and excess kurtosis in the statistics tables.
            The work is done in femtic_misfit.py (see its changelog).
2026-11-09  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            created. Ensemble half of femtic_data_misfit.py (RUN_MODE = "ensemble"): same
            settings, same output; RUN_MODE is gone, the script is the ensemble mode.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
# --- working directory ---
# Every relative path below (RUN_PATHS, OUT_DIR, NRMS_RESULTS, NRMS_CSV,
# NRMS_HIST_FILE, CURVES_SITE_DAT) and every path on the command line is taken
# relative to WORK_DIR; absolute paths (and ~) are used as given. None or "" =
# the current directory. -w / --work-dir DIR on the command line replaces it.
# WORK_DIR: Optional[str] = r"/media/vrath/LargeBack/Ensembles/krafla2026/krafla_fine/krafla_ens_fine_paired/"
# WORK_DIR: Optional[str] = r"/media/vrath/LargeBack/Ensembles/krafla2026/krafla_fine/krafla_ens_fine/"
WORK_DIR: Optional[str] = r"/media/vrath/LargeBack/Ensembles/krafla2026/krafla_ens/"
                                # the ensemble directory: the parent of the members

# --- what to run ---
# All entries of RUN_PATHS together form ONE ensemble: the chosen iteration of
# every run is one member (members must share the same data rows).
RUN_PATHS: List[str] = [
    r"kra*/"
    ]        # run dirs, results files, or globs, relative to WORK_DIR (overridden by argv)

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
SHOW_RMS2: bool = False            # Baba's RMS2 (observed error plus the error
                                   # synthesized from the ensemble spread, i.e. a
                                   # modelling error): False = not shown, plots and
                                   # tables use RMS1 only; True = RMS1 and RMS2

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
    "results/"
    # "../krafla_ens_fine_paired/results/"
)
# None      # None -> into the directory <common parent of the member
                                   # directories>/<OUT_PREFIX>/ (made if missing);
                                   # or a path (relative to WORK_DIR), used as
                                   # given (as here: no OUT_PREFIX sub-directory;
                                   # set None to get it)
OUT_PREFIX: Optional[str] = "KraflaMisfits"  # None -> derived from the ensemble
                                   # (ensemble_M<M>); also the name of the
                                   
                                   # output directory while OUT_DIR is None
OUT_PREFIX_DIR: bool = True     # False: with OUT_DIR None write into the common
                                # parent itself (no OUT_PREFIX sub-directory)
PLOT_FORMATS: Tuple[str, ...] = (".pdf",)   # a tuple: note the comma!
PLOT_DPI: int = 300
PLOT_DIR: Optional[str] = None  # where the figures go (tables stay in
                            # the output directory). None = same directory
                            # as the tables; a relative name = sub-directory
                            # of the OUTPUT directory (per run; not of WORK_DIR);
                            # an absolute path = that one directory for all
                            # figures (file names stay distinct via the prefix)
WRITE_STATS: bool = True
FIT_MEASURES: bool = True   # overall table of fit measures (per data type
                            # and pooled) -> <prefix>_fit_measures.txt
SHOW: bool = False

# --- titles (all figures) ---
# "auto" = the titles of before; None, False or "" = no figure title; a string,
# used as given ({title} = the run description, {datatype} / {group} = data
# type or group of the figure, {axis}, {per_decade} for histogram / Q-Q,
# {site}, {sid}, {nrms} for curves, {key} = the kind of figure); a dict
# {"crossplot": spec, "histogram": spec, "qq": spec, "curves": spec,
# "default": spec}
FIGURE_TITLE: Union[str, bool, None, dict] = "auto"
PANEL_TITLES: bool = True     # titles of the panels (crossplot, histogram, Q-Q)

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
CROSSPLOT_CBAR_LOCATION: Optional[str] = None   # None = at the right (as before);
                                                # "right", "left", "top", "bottom"
CROSSPLOT_CBAR_SHRINK: float = 0.85   # length / side of the panel grid

# --- residual histograms ---
HIST_BAND_AXIS: str = "frequency"   # "frequency" | "period"
HIST_BANDS_PER_DECADE: int = 1
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
NRMS_RESULTS: Optional[str] ="results/krafla_rto_ref_results.npz" # e.g. "results/<name>.npz"
    # results .npz of femtic_ens_post.py (holds the nRMS table of ALL scanned
    # members, <P>_nrms_table); relative to WORK_DIR; None = use NRMS_CSV only
    # (with both None the method "nrms" is skipped, with a message)
NRMS_NPZ_PREFIX: Optional[str] = None   # key prefix inside the .npz; None = the
                                        # file's only ensemble
NRMS_CSV: Optional[str] = None   # fallback for results files written before
                                 # 2026-11-08 (no table inside): the CSV
                                 # <ENSEMBLE_PREFIX>_nrms.csv of femtic_ens_post.py
                                 # (relative to WORK_DIR). If neither gives a
                                 # table, only the accepted members of the .npz
                                 # are shown
NRMS_MAX: Optional[float] = None   # acceptance threshold line; None = the
                                   # value stored in the .npz (NRMS_MAX of
                                   # femtic_ens_post.py), none if absent
NRMS_BINS: Union[int, str] = "auto"   # an int, or a numpy rule ("auto", "fd", ...)
NRMS_HIST_FILE: Optional[str] = None   # extension-less output path (relative to
                                       # WORK_DIR); None = <figure dir>/<prefix>_nrms_hist
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
CURVES_SITE_DAT: Optional[str] = "auto"   # or e.g. "templates/site.dat"
# "auto"   # site.dat from mt_make_sitelist.py
# (name,lat,lon,elev,sitenum,E,N): gives the
# site names (None = no names). "auto" = the
# local site.dat: looked for in the results
# directory and up to two levels above, then
# in WORK_DIR; or a path (relative to WORK_DIR)
CURVES_NAME_CASE: Optional[str] = None   # site names in the figure titles and the
# file names: None = as in site.dat, "upper"
# or "lower". With a name the title reads
# "site NAME (site#)", else "site site#"
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
# by CURVES_SITE_ORDER); list = exactly these
# sites, as site_id values and/or site names
# of the local site.dat (case-insensitive;
# "A01, A02" as one string also works). A
# list gives one file per site (see
# CURVES_MULTIPAGE), e.g. ["A01", "a02", 17]
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
CURVES_INVERT_X: bool = False      # False = period increasing to the right
                                   # (long periods at the right; the same as frequency
                                   # decreasing to the right); True = flipped (the
                                   # old data_viz orientation)
CURVES_PERIOD_LIM: Optional[Tuple[float, float]] = None   # (Tmin, Tmax) [s]
CURVES_YLIM: Dict[str, Optional[Tuple[float, float]]] = {
    "rho": None, "phase": (-180.,+180.,), "tipper": (-0.5,+0.5,), "pt": None}
CURVES_PANEL_SIZE: Tuple[float, float] = (4.2, 3.4)       # width, height [in]
CURVES_LEGEND_LOC: Optional[str] = "below"   # "below" (under each panel), a
                                   # matplotlib loc such as "upper right", or
                                   # None for no legend. Avoid "best" with
                                   # density: matplotlib then needs seconds
                                   # per panel to place it
CURVES_MULTIPAGE: Optional[bool] = None   # None = auto: all sites / N sites -> as
                                   # True; a LIST of sites (CURVES_SITES) -> as
                                   # False. True: ".pdf" is one file, one page
                                   # per site; False: the pdfs are single files
                                   # too. Other formats (.jpg, .png, ...) are
                                   # always one file per site, in a
                                   # sub-directory <prefix>_curves/
CURVES_APPRES_PHASE_DEG: bool = True   # APP_RES_AND_PHS phase in degrees
                                       # (False: radians)
# phase curves that leave the +-180 deg window (a jump of more than half the
# window between two periods): the line is interrupted there and the two
# points are joined by a separate line; bands and density skip the jump
CURVES_PHASE_WRAP_MODE: str = "direct"   # "direct" (straight line between the two
                                       # points) | "short" (the short way, through
                                       # the window edge) | "off" (no handling)
CURVES_PHASE_WRAP_STYLE: Union[None, bool, Dict[str, Any]] = None
                                       # Matplotlib line properties of the connection,
                                       # e.g. {"linestyle": "--", "alpha": 0.5};
                                       # colour / width / alpha of the curve are
                                       # inherited. None = data_viz.PHASE_WRAP_STYLE
                                       # (dotted); False = gap only, no connection
CURVES_PHASE_WRAP_RANGE: Tuple[float, float] = (-180.0, 180.0)
                                       # phase window; (-3.14159, 3.14159) if the
                                       # phases are in radians

# small map of the sites in every figure (current site marked, all sites
# coloured by their nRMS, the same colour scale on every page)
CURVES_MAP: bool = True           # True = draw the map
CURVES_MAP_PLACEMENT: str = "left"   # "right" | "left": an own column next to the
                                   # panels (the figure gets wider by CURVES_MAP_SIZE);
                                   # "upper right" | "upper left" | "lower right" |
                                   # "lower left": inset in the corner of the last
                                   # (right) / first (left) panel, on top of the
                                   # curves (do not put the legend there)
CURVES_MAP_SIZE: float = 3.2       # [in] side of the map block (map and colour bar)
CURVES_MAP_TICKS: bool = True      # True = ticks and labels in model coordinates
                                   # [km]; False = no ticks, no labels (frame only)
CURVES_MAP_AXES: str = "ne"        # "ne": site_x = north (vertical), site_y = east
                                   # (horizontal), the FEMTIC convention; "en":
                                   # site_x = east, site_y = north
CURVES_MAP_CMAP: Optional[str] = "jet" #None   # None = single-hue blue ramp, light = low
                                   # nRMS, dark = high; or a Matplotlib colormap
CURVES_MAP_NRMS_RANGE: Optional[Tuple[float, float]] = None   # (lo, hi) of the colour
                                   # scale; None = smallest / largest nRMS of the sites
CURVES_MAP_MARKER_SIZE: float = 10.0   # [pt^2] of a site (the current one is larger)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def resolve_work_dir(work_dir: Optional[str] = None) -> str:
    """
    Absolute WORK_DIR: `work_dir` if given, else the setting WORK_DIR; None or
    "" = the current directory. ~ is expanded; a relative name is taken from
    the current directory.
    """
    wd = WORK_DIR if work_dir is None else work_dir
    return os.path.abspath(os.path.expanduser(str(wd or ".")))


def anchor(path: str, work_dir: str) -> str:
    """
    `path` with ~ expanded; a relative path is joined to `work_dir`, an absolute
    one is returned as it is. Not normalized, so that a glob pattern and a
    trailing "/" (a glob that matches directories only) are kept.
    """
    p = os.path.expanduser(str(path))
    return p if os.path.isabs(p) else os.path.join(work_dir, p)


def settings(work_dir: Optional[str] = None) -> "fm.SimpleNamespace":
    """
    The uppercase settings of this script as a femtic_misfit settings object,
    every path of them anchored at WORK_DIR (`work_dir`, else the setting).
    PLOT_DIR is passed on unchanged: a relative one is a sub-directory of the
    output directory (see femtic_misfit.resolve_plot_dir).
    """
    wd = resolve_work_dir(work_dir)
    ours = {k: v for k, v in globals().items() if k in fm.SETTING_NAMES}
    stray = [k for k in globals() if k.isupper() and k not in fm.SETTING_NAMES
             and k not in ("PY4MTX_ROOT", "WORK_DIR")]
    if stray:
        print(f"femtic_ens_misfit: ignoring names that are no setting of "
              f"femtic_misfit: {stray}")

    runs = ours.get("RUN_PATHS") or []
    if isinstance(runs, str):                    # one path given as a plain string
        runs = [runs]
    ours["RUN_PATHS"] = [anchor(p, wd) for p in runs]
    for key in ("OUT_DIR", "NRMS_RESULTS", "NRMS_CSV", "NRMS_HIST_FILE"):
        if ours.get(key):
            ours[key] = anchor(ours[key], wd)
    site = ours.get("CURVES_SITE_DAT")
    if site and site != "auto":                  # "auto": searched for by the engine
        ours["CURVES_SITE_DAT"] = anchor(site, wd)
    return fm.make_settings(**ours)


def main(argv: Sequence[str]) -> int:
    """
    Run the ensemble misfit diagnostics with all paths taken from WORK_DIR.
    Return code: 0 success, 1 no usable member / unreadable file, 2 usage error
    (WORK_DIR missing).
    """
    ap = argparse.ArgumentParser(
        prog="femtic_ens_misfit.py",
        description="Data-misfit diagnostics for an ensemble of FEMTIC runs. "
                    "All paths are taken relative to WORK_DIR.")
    ap.add_argument("paths", nargs="*", metavar="PATH",
                    help="run directories, results files or quoted globs, relative to "
                         "WORK_DIR (replace RUN_PATHS)")
    ap.add_argument("-w", "--work-dir", default=None, metavar="DIR",
                    help="base of all relative paths (replaces WORK_DIR; default: the "
                         "setting WORK_DIR, or the current directory if that is None)")
    args = ap.parse_args(list(argv))

    wd = resolve_work_dir(args.work_dir)
    if not os.path.isdir(wd):
        print(f"femtic_ens_misfit: WORK_DIR is not a directory: {wd}", file=sys.stderr)
        return 2
    print(f"femtic_ens_misfit: WORK_DIR = {wd}")

    opts = settings(wd)
    paths = [anchor(p, wd) for p in args.paths]
    cwd = os.getcwd()
    os.chdir(wd)               # the engine's last-resort "auto" site.dat search uses it
    try:
        return fm.run_ensemble(opts, paths)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
