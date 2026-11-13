#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_misfit.py

Data-misfit diagnostics for single FEMTIC models / runs (mod family).

Every entry of RUN_PATHS (run directories with results_iter*.h5, results
files, or globs) is processed on its own, they do NOT form an ensemble: one
set of figures and tables per run, for the iteration chosen by ITERATION
("best" = smallest error-normalized RMS, "last", or an integer). Methods
(METHODS):

  "crossplot"  observed vs. calculated, one panel per data type / component
  "histogram"  normalized-residual histograms in frequency (or period) bands
               (conventional RMS1 normalization; there is no synthesized
               error for one model)
  "qq"         normal Q-Q plots of the same residuals in the same bands
  "curves"     optional: observed data with errors against the calculated
               response, per site

The ensemble counterpart is femtic_ens_misfit.py (also: RMS2 on request, member
curves, nRMS histogram of all scanned members). Both call femtic_misfit.py (the
engine of femtic_data_misfit.py, which stays as it is and does both jobs in
one script).

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
    python femtic_mod_misfit.py                         # settings below
    python femtic_mod_misfit.py ./run01/ ./run02/       # two runs, separately
    python femtic_mod_misfit.py "./ens/run_*/"          # glob: one set each
    python femtic_mod_misfit.py ./run01/results_iter25.h5   # a results file

Output of each run: prefix <OUT_PREFIX>_<run>_iter<N> (OUT_PREFIX = None:
<run>_iter<N>), written into the directory <directory of the chosen results
file>/<OUT_PREFIX>/, made if it does not exist (OUT_DIR forces one directory,
used as given; OUT_PREFIX_DIR = False writes into the results directory itself;
PLOT_DIR separates the figures from the tables).

Curves of chosen sites only: CURVES_SITES = ["A01", "a02", 17] (names of the
local site.dat, any case, and/or site ids) writes one file per site and format
(.pdf, .jpg, ...) into <OUT_PREFIX>/<prefix>_curves/.

Changelog
---------
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
            output goes to the directory <run dir>/<OUT_PREFIX>/ (new
            OUT_PREFIX_DIR); CURVES_SITES list -> single pdf / jpg files
            (CURVES_MULTIPAGE = None = auto), names from the local site.dat
            in any case; new CURVES_NAME_CASE, title "site NAME (site#)";
            std, skew and excess kurtosis in the statistics tables. The
            work is done in femtic_misfit.py (see its changelog).
2026-11-09  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            created. Single-model half of femtic_data_misfit.py (RUN_MODE = "single"): same
            settings (without the ensemble ones), same output; RUN_MODE and the
            "nrms" method are gone.
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
# Every entry of RUN_PATHS is a model of its own (a list of singles, NOT an
# ensemble): one set of figures and tables per entry.
RUN_PATHS: List[str] = [
    "/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/annecy_rnd_2_59/",
    "/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/annecy_rnd_2_60/",
    ]        # run dirs, results files, or globs (overridden by argv)

ITERATION: Union[str, int] = "best"      # "best" | "last" | integer
ITER_PATTERN: str = "results_iter*.h5"   # searched in each run directory
METHODS: Tuple[str, ...] = ("crossplot", "histogram", "qq", "curves")
#   "curves" is optional (needs data_viz.py); "nrms" is for ensembles only
#   (femtic_ens_misfit.py)

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
OUT_DIR: Optional[str] = None   # None -> into the directory <run dir>/<OUT_PREFIX>/
                                # (made if missing); or one path for all runs,
                                # used as given (no OUT_PREFIX sub-directory)
OUT_PREFIX: Optional[str] = "Misfits"  # None -> derived from run / ensemble;
                                   # if set with several singles it is
                                   # prepended: <OUT_PREFIX>_<run>_iter<N>;
                                   # also the name of the output directory
                                   # while OUT_DIR is None
OUT_PREFIX_DIR: bool = True     # False: with OUT_DIR None write into the run
                                # directory itself (no OUT_PREFIX sub-directory)
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

# --- response curves (optional method "curves"; add it to METHODS) ---
# Needs data_viz.py on the path. One figure per site, one panel per entry of
# CURVES_WHAT; PLOT_FORMATS/PLOT_DPI/OUT_DIR/OUT_PREFIX apply as usual.
CURVES_WHAT: Tuple[str, ...] = ("rho", "phase", "tipper")  # + "pt"
CURVES_COMPS: str = (
    "xx, xy, yx, yy"  # impedance components (rho and phase panels)
)
CURVES_SITE_DAT: Optional[str] = "auto"
# "auto"   # site.dat from mt_make_sitelist.py
# (name,lat,lon,elev,sitenum,E,N): gives the
# site names (None = no names). "auto" = the
# local site.dat: looked for in the results
# directory and up to two levels above, then
# in the current working directory; or a path
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

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def settings() -> "fm.SimpleNamespace":
    """The uppercase settings of this script as a femtic_misfit settings object."""
    ours = {k: v for k, v in globals().items() if k in fm.SETTING_NAMES}
    stray = [k for k in globals() if k.isupper() and k not in fm.SETTING_NAMES
             and k not in ("PY4MTX_ROOT",)]
    if stray:
        print(f"femtic_mod_misfit: ignoring names that are no setting of "
              f"femtic_misfit: {stray}")
    return fm.make_settings(**ours)


def main(argv: Sequence[str]) -> int:
    return fm.run_models(settings(), argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
