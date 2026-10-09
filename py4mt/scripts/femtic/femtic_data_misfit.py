#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_data_misfit.py

Data-misfit diagnostics for FEMTIC results_iterN.h5 files:

  1. "crossplot"  - scatter plots of observed vs. calculated data, one panel
                    per data type and component, log10 frequency as colour.
  2. "histogram"  - normalized-residual histograms in frequency (or period)
                    bands, following Baba (2023): residuals normalized by the
                    observed error only (conventional, RMS1) and by the
                    observed and the synthesized error (proposed, RMS2).
  3. "qq"         - normal Q-Q plots of the same normalized residuals in the
                    same frequency (or period) bands, with the ideal 1:1
                    line, a quartile line (slope = robust scale, intercept
                    = bias), and a pointwise confidence envelope.

  4. "curves"    - OPTIONAL (not in the default METHODS): response curves per
                    site, observed data with errors against the calculated
                    response, drawn with the data_viz plotters (add_rho,
                    add_phase, add_tipper, add_pt). For an ensemble the
                    members can be drawn as many thin curves, as a per-period
                    density of all member curves, as percentile bands, or a
                    combination ("auto" switches from curves to density when
                    the ensemble is large).

All four work on single runs (best iteration of each) or on an ensemble of
runs (best iteration of each member).

@author   vrath
@project  py4mt
@created  2026-09-18
@modified 2026-09-30

Provenance
----------
Author      : Claude (Anthropic)
Generated   : 2026-09-19 (last modified 2026-09-30)
Notice      : This code is AI-generated. Review and test it before any
              production use.

Reference
---------
Baba, K. (2023): A simple method to evaluate the uncertainty of
magnetotelluric forward modeling for practical three-dimensional
conductivity structure models. Earth, Planets and Space 75, 67.
https://doi.org/10.1186/s40623-023-01832-5

Run modes
---------
single    RUN_PATHS is a list of runs (run directories containing
          results_iter*.h5, results files, or glob patterns). Every entry
          is processed on its own -- they do NOT form an ensemble. For a
          directory the iteration is chosen by ITERATION ("best" = smallest
          overall error-normalized RMS of the /data rows, "last", or an
          integer).
ensemble  All entries of RUN_PATHS together form one ensemble. The chosen
          iteration of every run is one member; members are matched row by
          row (datatype, site, component, frequency).

Fit measures (from inverse.py): for the error-normalized residuals r,
    nrms  = sqrt(mean(r^2))            expected 1    for r ~ N(0,1)
    r_mae = sqrt(pi/2) * mean(|r|)     expected 1
    r_med = median(|r|) / 0.6745       expected 1
    q95   = 95th percentile of |r|     expected 1.96
plus the mean of r (bias). They are given for both normalizations (err =
conventional, err+syn = Baba's RMS2; the latter only if M > 1) in the overall
table (<prefix>_fit_measures.txt) and in the crossplot and histogram tables.
Robust measures (r_mae, r_med, q95) that stay near their expected values
while nrms is large point to a few outliers rather than a general misfit.

Output location: OUT_DIR = None (default) writes into the run directory
(single: the directory of each chosen results file; ensemble: the common
parent directory of the member directories); a string forces one directory.

Method notes (see also the readme)
----------------------------------
Baba (2023), Eqs. (10)-(14), for M syntheses of the same model:

    mu      = mean of the M calculated values          (complex mean)
    sigma   = sqrt( 1/(M-1) * sum |Z_j - mu|^2 )       (complex modulus)
    eps_syn = sigma / sqrt(M)
    RMS1    = sqrt( 1/(2N) sum |Zobs - Zsyn|^2 / eps_obs^2 )
    RMS2    = sqrt( 1/(2N) sum |Zobs - mu|^2 / (eps_obs^2 + eps_syn^2) )

Here the M syntheses are the ensemble members. The histograms show the
per-component (real/imaginary) normalized residuals, as in Baba's Fig. 10a,
with N(0, RMS^2) overlaid. In single mode M = 1, eps_syn = 0, and RMS2 is
identical to RMS1 (only the conventional histogram is drawn).

Optional plotting section ("curves", see also the CURVES_* settings)
--------------------------------------------------------------------
Needs data_viz.py (and, if importable, femtic_viz.py for the component
marker convention) on the path; nothing else changes if "curves" is not in
METHODS. One figure (one page of <prefix>_curves.pdf) per selected site, one
panel per quantity in CURVES_WHAT ("rho", "phase", "tipper", "pt"):

    single run   observed data (markers, error bars or shading) and the
                 calculated response (line).
    ensemble     the calculated responses of all M members, as
                 "curves"   thin transparent lines (at most CURVES_SUBSAMPLE),
                 "density"  per-period density of all member curves (member
                            curves are interpolated onto a fine log-period
                            grid and binned in y; the colour opacity is the
                            fraction of members per bin -- this scales to
                            hundreds of members),
                 "bands"    percentile envelopes (CURVES_BAND_PCT),
                 or a "+"-joined combination such as "density+bands";
                 "auto" = curves for M <= CURVES_MAX_CURVES, else density.
                 The ensemble median (CURVES_MEDIAN) and the observed data
                 (mean over members, as in the other methods) are drawn on
                 top.

Quantities are derived from the /data rows in the same way as
data_viz.datadict_to_plot_df (rho_a = |Z|^2 / (mu0 omega) with Z in SI ohm as
stored by FEMTIC; phase = atan2(Im Z, Re Z); the error of rho_a and of the
phase from the complex modulus of the errors); the only difference is that
the calculated phase is put on the branch of the observed phase (no 360
degree jumps for members crossing +-180, common for yx). Apparent-resistivity/phase
data types (APP_RES_AND_PHS) are taken as stored (rho in ohm.m, phase in
degrees unless CURVES_APPRES_PHASE_DEG = False). Data types without a
data_viz plotter (HTF, NMT) are skipped.

Usage
-----
    python femtic_data_misfit.py                       # config below
    python femtic_data_misfit.py single   ./run01/ ./run02/   # separately
    python femtic_data_misfit.py ensemble "./ens/member_*/"

Examples: "single" (each entry is its own run, one output set per run)
    # config block: RUN_MODE = "single"; ITERATION = "best"
    RUN_PATHS = ["/data/ens/run01/", "/data/ens/run02/"]
    #   -> two independent evaluations, prefixes Misfits_run01_iter<N> and
    #      Misfits_run02_iter<N> (OUT_PREFIX = "Misfits"; None -> run01_iter<N>),
    #      each written into the directory of its chosen results file.
    python femtic_data_misfit.py single ./run01/                # one run
    python femtic_data_misfit.py single ./run01/ ./run02/       # two runs
    python femtic_data_misfit.py single "./ens/run_*/"          # glob: one each
    # a fixed iteration instead of the best one: ITERATION = 25 ("last" and
    # "best" are also allowed); a results file may be given directly:
    python femtic_data_misfit.py single ./run01/results_iter25.h5

Examples: "ensemble" (all entries together form ONE ensemble, M members)
    # config block: RUN_MODE = "ensemble"; ITERATION = "last"
    RUN_PATHS = ["/data/ens/member_*/"]       # one member per directory
    python femtic_data_misfit.py ensemble "./ens/member_*/"
    python femtic_data_misfit.py ensemble ./m01/ ./m02/ ./m03/
    #   -> one output set, prefix ensemble_M<M>, written into the common
    #      parent directory of the members (OUT_DIR = "/some/dir" forces
    #      one directory). The chosen iteration of each member is one
    #      member of the ensemble; members must share the same data rows.
    #      Gives RMS1 and RMS2 (Baba 2023; RMS2 needs M > 1).
    #   Quote globs so the shell does not expand them (the script expands them).

Changelog
---------
2026-09-18  Claude Sonnet 5 (Anthropic): initial version (as
            femtic_crossplot.py): observed-vs-calculated crossplots.
2026-09-19  Claude Sonnet 5 (Anthropic): renamed to femtic_data_misfit.py;
            added the normalized-residual histogram method of Baba (2023)
            in frequency/period bands (default 2 per decade); added single
            run (best iteration) and ensemble modes; METHODS switch.
2026-09-19b Claude Sonnet 5 (Anthropic): added third method "qq" (normal Q-Q
            plots of the normalized residuals in frequency/period bands);
            band selection factored out of the histogram function.
2026-09-21  Claude Sonnet 5 (Anthropic): (1) OUT_DIR = None writes into the
            run directory (ensemble: common parent of the members);
            (2) single mode processes every entry of RUN_PATHS separately
            (a list of singles, not an ensemble); (3) QQ_EQUAL_AXES gives
            optionally equal (square) Q-Q axes.
2026-09-21b Claude Sonnet 5 (Anthropic): (4) several measures of data fit
            (nrms, r_mae, r_med, q95 from inverse.py, plus mean) in an
            overall table per data type and in the crossplot and histogram
            tables; FIT_MEASURES switch. Crossplot statistics now use the
            same residuals as the histograms (RMS1 reference = REF_MEMBER
            response if set).
2026-09-21c Claude Sonnet 5 (Anthropic): the run name (directory of the
            chosen results file) and iteration are shown in the title of
            every plot in single mode (ensemble: "ensemble of M runs").
2026-09-21d Claude Sonnet 5 (Anthropic): fixed PLOT_FORMATS handling: a plain
            string such as (".pdf") (no trailing comma) was iterated
            character by character (".", "p", "d", "f") and produced four
            PNG files with names ending in "", "p", "d", "f"; _save() now
            accepts a string, adds a missing dot and passes the format
            explicitly (unsupported extensions raise).
2026-09-30  Claude (Anthropic): optional fourth method "curves": per-site
            response curves (observed vs. calculated) built on the data_viz
            plotters; ensemble-aware (many thin curves, per-period density,
            percentile bands, ensemble median); CURVES_* settings; site
            selection (even / first / worst / best by nRMS, or explicit ids);
            multi-page PDF; per-site nRMS table (<prefix>_site_nrms.txt).
            build_dataset() now also keeps the per-member calculated
            responses (cal_re_m, cal_im_m, shape (M, rows)) and the site
            coordinates (site_x, site_y); the other three methods are
            unchanged.
2026-09-30b Claude (Anthropic): site names for "curves" from the site.dat of
            mt_make_sitelist.py (femtic.read_site_dat): shown in the figure
            titles, the file names and <prefix>_site_nrms.txt, and accepted in
            CURVES_SITES; CURVES_SITE_DAT ("auto" searches the results
            directory and two levels up) and CURVES_SITE_ID_OFFSET.
2026-09-30c Claude (Anthropic): CURVES_OBSERVED_ONLY = True draws only the
            observed data (markers + errors) in "curves"; calculated
            response, ensemble layers and nRMS in the title are omitted.
2026-10-08  Claude (Anthropic), AI-generated, review before use: documentation
            only -- explicit "single" and "ensemble" usage examples (config
            block and command line) added to the Usage section and readme.
2026-10-08b Claude (Anthropic), AI-generated, review before use: resolve_run()
            skips files h5py cannot open (OSError, e.g. "file signature not
            found") like unreadable results; ensemble mode skips members
            without a usable file and continues with the rest.
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# py4mt environment: make the modules importable (femtic.py, ensembles.py)
# ---------------------------------------------------------------------------
PY4MTX_ROOT = os.environ.get("PY4MTX_ROOT", "")
if PY4MTX_ROOT:
    for _pth in (os.path.join(PY4MTX_ROOT, "py4mt", "modules"),
                 os.path.join(PY4MTX_ROOT, "py4mt", "scripts")):
        if os.path.isdir(_pth) and _pth not in sys.path:
            sys.path.insert(0, _pth)

import femtic as fem  # noqa: E402
import inverse as inv  # noqa: E402  (nrms, r_mae, r_med, q95)

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
# --- what to run ---

# "single":   every entry of RUN_PATHS is processed on its own (a list of
#             singles; they do NOT form an ensemble)
# "ensemble": all entries together form one ensemble
#  RUN_MODE: str = "ensemble"               # "single" | "ensemble"
RUN_MODE: str = "ensemble"               # "single" | "ensemble"
RUN_PATHS: List[str] = [
 r"/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_0/ann*/"
    ]        # run dirs, results files, or globs (overridden by argv)
# RUN_MODE: str = "single"
# RUN_PATHS: List[str] = [
#     "/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/annecy_rnd_2_59/",
#     "/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/annecy_rnd_2_60/",
#     ]
ITERATION: Union[str, int] = "best"      # "best" | "last" | integer
ITER_PATTERN: str = "results_iter*.h5"   # searched in each run directory
METHODS: Tuple[str, ...] = ("crossplot", "histogram", "qq")
# METHODS = ("crossplot", "histogram", "qq", "curves")   # "curves" is optional

# --- ensemble handling ---
ENSEMBLE_OBS: str = "mean"     # observed values/errors: "mean" over members
                               # (RTO-perturbed data average back to the
                               # original) or "first" member
REF_MEMBER: Optional[int] = None   # RMS1 reference: None = ensemble mean;
                                   # integer = calculated response of that
                                   # member (as in Baba's "6th model")

# --- output ---
OUT_DIR: Optional[str] = None      # None -> into the run directory (single:
                                   # each run's own; ensemble: common parent
                                   # of the member directories); or a path
OUT_PREFIX: Optional[str] = "Misfits"  # None -> derived from run / ensemble;
                                   # if set with several singles it is
                                   # prepended: <OUT_PREFIX>_<run>_iter<N>
PLOT_FORMATS: Tuple[str, ...] = (".pdf",)   # a tuple: note the comma!
PLOT_DPI: int = 300
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

# --- response curves (optional method "curves"; add it to METHODS) ---
# Needs data_viz.py on the path. One figure per site, one panel per entry of
# CURVES_WHAT; PLOT_FORMATS/PLOT_DPI/OUT_DIR/OUT_PREFIX apply as usual.
CURVES_WHAT: Tuple[str, ...] = ("rho", "phase", "tipper")   # + "pt"
CURVES_COMPS: str = "xx, xy, yx, yy"        # impedance components (rho and phase panels)
CURVES_SITE_DAT: Optional[str] = "auto"   # site.dat from mt_make_sitelist.py
                                   # (name,lat,lon,elev,sitenum,E,N): gives the
                                   # site names. "auto" = look for site.dat in
                                   # the results directory and up to two
                                   # levels above; a path; None = no names
CURVES_SITE_ID_OFFSET: int = 0     # site_id in the results file = sitenum
                                   # in site.dat + this offset
CURVES_SITES: Union[None, int, Sequence[Union[int, str]]] = 12
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
# Constants
# ---------------------------------------------------------------------------
REAL_ONLY_TYPES = ("APP_RES_AND_PHS", "NMT2_APP_RES_AND_PHS", "PT")


# ---------------------------------------------------------------------------
# Run / iteration selection
# ---------------------------------------------------------------------------
def _iter_from_name(path: str) -> int:
    m = re.search(r"iter(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def data_rms(data: dict) -> float:
    """
    Overall error-normalized RMS of a results file's /data rows.

    Real and imaginary parts count separately; the imaginary part is left
    out for real-valued data types. Non-finite values and non-positive
    errors are skipped. Returns NaN if nothing is usable.
    """
    rows = data["rows"]
    if rows.size == 0:
        return float("nan")
    dn = data["datatype_name"]
    cpx = ~np.isin(dn, REAL_ONLY_TYPES)
    r_all = []
    for obs, cal, err, m in (
        (rows["re_val"], rows["cal_re"], rows["re_err"], np.ones(rows.size, bool)),
        (rows["im_val"], rows["cal_im"], rows["im_err"], cpx),
    ):
        obs = np.asarray(obs, float)
        cal = np.asarray(cal, float)
        err = np.asarray(err, float)
        ok = m & np.isfinite(obs) & np.isfinite(cal) & np.isfinite(err) & (err > 0)
        r_all.append((obs[ok] - cal[ok]) / err[ok])
    r = np.concatenate(r_all)
    return float(np.sqrt(np.mean(r**2))) if r.size else float("nan")


def resolve_run(path: str, *, iteration: Union[str, int], pattern: str) -> dict:
    """
    Pick one results file from a run and read its /data group.

    Parameters
    ----------
    path : str
        Run directory (searched with `pattern`) or a results file.
    iteration : "best", "last" or int
        Ignored if `path` is a file.

    Returns
    -------
    dict with keys run, path, iter, rms, data, n_candidates.
    """
    if os.path.isfile(path):
        cands = [path]
    elif os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, pattern)))
    else:
        raise FileNotFoundError(f"resolve_run: {path} not found.")
    if not cands:
        raise FileNotFoundError(f"resolve_run: no '{pattern}' in {path}.")

    best = None
    best_score = float("inf")
    n_ok = 0
    for c in cands:
        try:
            d = fem.read_results_data(c, out=False)
        except (KeyError, OSError) as exc:
            # OSError: not an HDF5 file / truncated / still being written
            print(f"femtic_data_misfit: skipping {c}: {exc}")
            continue
        if d["nRows"] == 0:
            continue
        n_ok += 1
        it = d["iterNum"] if d["iterNum"] >= 0 else _iter_from_name(c)
        rms = data_rms(d)
        if os.path.isfile(path):
            score = 0.0
        elif iteration == "best":
            score = rms if np.isfinite(rms) else float("inf")
        elif iteration == "last":
            score = -float(it)
        else:
            score = 0.0 if it == int(iteration) else float("inf")
        if best is None or score < best_score:
            best, best_score = dict(run=path, path=c, iter=it, rms=rms, data=d), score
    if best is None or (best_score == float("inf") and n_ok > 0 and iteration not in ("best", "last")):
        raise ValueError(f"resolve_run: no usable file (iteration={iteration!r}) in {path}.")
    best["n_candidates"] = n_ok
    return best


def expand_paths(paths: Sequence[str]) -> List[str]:
    """Expand glob patterns; keep order, drop duplicates."""
    out: List[str] = []
    for p in paths:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        for h in hits:
            if h not in out:
                out.append(h)
    return out


# ---------------------------------------------------------------------------
# Building the (ensemble) dataset
# ---------------------------------------------------------------------------
def _row_keys(rows: np.ndarray) -> Tuple[np.ndarray, ...]:
    f = np.asarray(rows["freq"], float)
    logf = np.round(np.log10(np.where(f > 0, f, np.nan)), 9)
    return (rows["datatype"].astype(np.int64), rows["site_id"].astype(np.int64),
            rows["component"].astype(np.int64), np.nan_to_num(logf, nan=-999.0))


def align_members(datas: Sequence[dict]) -> List[np.ndarray]:
    """
    Row indices into each member's rows such that row k is the same datum
    (datatype, site_id, component, frequency) in every member.
    """
    keys = [_row_keys(d["rows"]) for d in datas]
    k0 = keys[0]
    same = all(len(k[0]) == len(k0[0]) and all(np.array_equal(a, b) for a, b in zip(k0, k))
               for k in keys[1:])
    if same:
        return [np.arange(len(k0[0])) for _ in datas]
    lists = [list(zip(*[a.tolist() for a in k])) for k in keys]
    maps = [{key: i for i, key in enumerate(lst)} for lst in lists]
    common = [key for key in lists[0] if all(key in m for m in maps[1:])]
    print(f"femtic_data_misfit: members differ in row layout; "
          f"{len(common)} of {len(lists[0])} rows are common to all.")
    return [np.array([m[key] for key in common], dtype=np.int64) for m in maps]


def build_dataset(datas: Sequence[dict], *, obs_mode: str = "mean",
                  ref_member: Optional[int] = None) -> dict:
    """
    Combine M >= 1 results datasets into one set of aligned arrays.

    Returns a dict with M and per-row arrays: freq, site_id, component,
    datatype_name, component_name, obs_re/im, err_re/im, cal_re/im (ensemble
    mean), ref_re/im (reference response for RMS1), eps_syn (Baba Eq. 13,
    zero for M = 1), plus n_dropped_nan.
    """
    M = len(datas)
    idx = align_members(datas)
    rows = [d["rows"][ix] for d, ix in zip(datas, idx)]

    def stack(name: str) -> np.ndarray:
        return np.vstack([np.asarray(r[name], float) for r in rows])

    cal_re, cal_im = stack("cal_re"), stack("cal_im")
    ok = np.all(np.isfinite(cal_re) & np.isfinite(cal_im), axis=0)
    n_drop = int(np.count_nonzero(~ok))

    def obs_like(name: str) -> np.ndarray:
        s = stack(name)
        return s.mean(axis=0) if obs_mode == "mean" else s[0]

    mu_re, mu_im = cal_re.mean(axis=0), cal_im.mean(axis=0)
    if M > 1:
        var = (((cal_re - mu_re) ** 2 + (cal_im - mu_im) ** 2).sum(axis=0)) / (M - 1)
        eps = np.sqrt(var / M)
    else:
        eps = np.zeros(mu_re.shape)
    if ref_member is None:
        ref_re, ref_im = mu_re, mu_im
    else:
        ref_re, ref_im = cal_re[ref_member], cal_im[ref_member]

    r0 = rows[0]
    ds = dict(
        M=M,
        freq=np.asarray(r0["freq"], float),
        site_id=np.asarray(r0["site_id"]),
        component=np.asarray(r0["component"]),
        datatype_name=np.asarray(datas[0]["datatype_name"])[idx[0]],
        component_name=np.asarray(datas[0]["component_name"])[idx[0]],
        obs_re=obs_like("re_val"), obs_im=obs_like("im_val"),
        err_re=obs_like("re_err"), err_im=obs_like("im_err"),
        cal_re=mu_re, cal_im=mu_im, ref_re=ref_re, ref_im=ref_im,
        eps_syn=eps,
        site_x=np.asarray(r0["site_x"], float),
        site_y=np.asarray(r0["site_y"], float),
    )
    for k in list(ds):
        if isinstance(ds[k], np.ndarray) and ds[k].shape[:1] == ok.shape:
            ds[k] = ds[k][ok]
    # per-member calculated responses, shape (M, n_rows): for the "curves"
    # method (added after the 1-D filter above, which would mis-handle 2-D)
    ds["cal_re_m"] = cal_re[:, ok]
    ds["cal_im_m"] = cal_im[:, ok]
    ds["n_dropped_nan"] = n_drop
    return ds


# ---------------------------------------------------------------------------
# Panels (one per datatype / component / part)
# ---------------------------------------------------------------------------
def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def build_panels(ds: dict, *, datatypes: Optional[Sequence[str]] = None,
                 components: Optional[Sequence[str]] = None) -> Dict[str, List[dict]]:
    """Split a dataset into panels: {datatype_name: [panel, ...]}."""
    dt_all = ds["datatype_name"]
    if dt_all.size == 0:
        raise ValueError("build_panels: dataset is empty.")
    _, first = np.unique(dt_all, return_index=True)
    dtype_order = [dt_all[i] for i in np.sort(first)]

    panels: Dict[str, List[dict]] = {}
    for dname in dtype_order:
        if datatypes is not None and dname not in datatypes:
            continue
        m_dt = dt_all == dname
        for code in np.unique(ds["component"][m_dt]):
            m = m_dt & (ds["component"] == code)
            cname = str(ds["component_name"][m][0])
            if components is not None and cname not in components:
                continue
            has_im = (dname not in REAL_ONLY_TYPES) and (
                np.any(np.nan_to_num(ds["obs_im"][m]) != 0.0)
                or np.any(np.nan_to_num(ds["cal_im"][m]) != 0.0))
            parts = [("re", "obs_re", "cal_re", "ref_re", "err_re")]
            if has_im:
                parts.append(("im", "obs_im", "cal_im", "ref_im", "err_im"))
            freq = ds["freq"][m]
            for part, ko, kc, kr, ke in parts:
                obs, cal, ref = ds[ko][m], ds[kc][m], ds[kr][m]
                err, eps = ds[ke][m], ds["eps_syn"][m]
                ok = (np.isfinite(obs) & np.isfinite(cal) & np.isfinite(ref)
                      & np.isfinite(freq) & (freq > 0.0))
                if has_im:
                    label = f"{'Re' if part == 're' else 'Im'} {cname}"
                    tag = part
                else:
                    label, tag = cname, "real"
                panels.setdefault(dname, []).append(dict(
                    datatype=dname, component=cname, part=tag, label=label,
                    freq=freq[ok], obs=obs[ok], cal=cal[ok], ref=ref[ok],
                    err=err[ok], eps=eps[ok], n_dropped=int(np.count_nonzero(~ok)),
                ))
    return panels


def fit_measures(r: np.ndarray) -> dict:
    """
    Fit measures of error-normalized residuals (non-finite values ignored):
    n, nrms, r_mae, r_med, q95 (all from inverse.py) and the mean (bias).
    Expected for N(0,1) residuals: nrms = r_mae = r_med = 1, q95 = 1.96.
    """
    r = np.asarray(r, dtype=float).ravel()
    r = r[np.isfinite(r)]
    return dict(
        n=int(r.size),
        nrms=float(inv.nrms(r)),
        r_mae=float(inv.r_mae(r)),
        r_med=float(inv.r_med(r)),
        q95=float(inv.q95(r)),
        mean=float(np.mean(r)) if r.size else float("nan"),
    )


def panel_stats(p: dict) -> dict:
    """
    n, plain rms, and the fit measures of the panel's normalized residuals:
    m1 (conventional, obs error only) and m2 (obs + synthesized error);
    nrms1 / nrms2 are shortcuts for m1["nrms"] / m2["nrms"].
    """
    res = p["obs"] - p["cal"]
    n = res.size
    rms = float(np.sqrt(np.mean(res**2))) if n else float("nan")
    _, r1, r2 = normalized_residuals(p)
    m1, m2 = fit_measures(r1), fit_measures(r2)
    return dict(n=n, n_dropped=p["n_dropped"], rms=rms, m1=m1, m2=m2,
                nrms1=m1["nrms"], nrms2=m2["nrms"])


def normalized_residuals(p: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (freq, r1, r2) for the rows with finite, positive error:
      r1 = (obs - ref) / err                       (conventional)
      r2 = (obs - mean) / sqrt(err^2 + eps_syn^2)  (Baba 2023, RMS2)
    """
    ok = np.isfinite(p["err"]) & (p["err"] > 0.0)
    err, eps = p["err"][ok], p["eps"][ok]
    r1 = (p["obs"][ok] - p["ref"][ok]) / err
    r2 = (p["obs"][ok] - p["cal"][ok]) / np.sqrt(err**2 + eps**2)
    return p["freq"][ok], r1, r2


def global_logf_range(panels: Dict[str, List[dict]]) -> Optional[Tuple[float, float]]:
    fs = [p["freq"] for lst in panels.values() for p in lst if p["freq"].size]
    if not fs:
        return None
    f = np.concatenate(fs)
    return float(np.log10(f.min())), float(np.log10(f.max()))


# ---------------------------------------------------------------------------
# Method 1: crossplots
# ---------------------------------------------------------------------------
def _axis_limits(x: np.ndarray, y: np.ndarray, *, log: bool) -> Tuple[float, float]:
    v = np.concatenate([x, y])
    lo, hi = float(np.min(v)), float(np.max(v))
    if log:
        if lo == hi:
            return lo / 2.0, hi * 2.0
        f = (hi / lo) ** 0.05
        return lo / f, hi * f
    if lo == hi:
        d = abs(lo) * 0.1 if lo != 0.0 else 1.0
        return lo - d, hi + d
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def plot_crossplot_datatype(
    dname: str, panels: Sequence[dict], *, title_extra: str, has_syn: bool,
    ncols: int, panel_size: float, cmap: str,
    color_range: Optional[Tuple[float, float]], marker_size: float,
    alpha: float, log_rho: bool, show_errorbars: bool, equal_aspect: bool,
):
    """One figure with all crossplot panels of a data type."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    n = len(panels)
    nc = max(1, min(ncols, n))
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(
        nr, nc, figsize=(panel_size * nc + 1.2, panel_size * nr + 0.6),
        squeeze=False, constrained_layout=True)
    axf = axes.ravel()

    sm_norm = None
    for k, p in enumerate(panels):
        ax = axf[k]
        x, y, err = p["obs"], p["cal"], p["err"]
        if x.size == 0:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(p["label"], fontsize=10)
            continue
        logf = np.log10(p["freq"])
        if color_range is not None:
            norm = Normalize(vmin=color_range[0], vmax=color_range[1])
        else:
            norm = Normalize(vmin=float(logf.min()), vmax=float(logf.max()))
        if sm_norm is None:
            sm_norm = norm
        use_log = (log_rho and p["component"].lower().startswith("rho")
                   and bool(np.all(x > 0.0)) and bool(np.all(y > 0.0)))
        if show_errorbars:
            xerr = np.where(np.isfinite(err) & (err > 0.0), err, 0.0)
            ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="0.75",
                        elinewidth=0.5, zorder=1)
        ax.scatter(x, y, c=logf, cmap=cmap, norm=norm, s=marker_size,
                   alpha=alpha, edgecolors="none", zorder=2)
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        lo, hi = _axis_limits(x, y, log=use_log)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, zorder=3)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        st = panel_stats(p)
        txt = (f"n={st['n']}\nnRMS1={st['nrms1']:.2f}\nnRMS2={st['nrms2']:.2f}"
               if has_syn else f"n={st['n']}\nnRMS={st['nrms1']:.2f}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
        ax.set_title(p["label"], fontsize=10)
        ax.set_xlabel("observed", fontsize=9)
        ax.set_ylabel("calculated (ensemble mean)" if has_syn else "calculated", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major", lw=0.3, alpha=0.5)
    for ax in axf[n:]:
        ax.set_visible(False)
    if sm_norm is not None:
        sm = cm.ScalarMappable(norm=sm_norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=list(axf[:n]), shrink=0.85, pad=0.02)
        cb.set_label("log10 frequency (Hz)")
    fig.suptitle(f"{dname}: observed vs. calculated, {title_extra}", fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# Method 2: normalized-residual histograms in frequency / period bands
# ---------------------------------------------------------------------------
def band_index(freq: np.ndarray, *, per_decade: int, axis: str) -> np.ndarray:
    """Integer band number; bands are anchored at whole decades."""
    x = np.log10(freq) if axis == "frequency" else -np.log10(freq)
    return np.floor(x * per_decade + 1e-9).astype(int)


def band_label(k: int, *, per_decade: int, axis: str) -> str:
    lo, hi = 10.0 ** (k / per_decade), 10.0 ** ((k + 1) / per_decade)
    unit = "Hz" if axis == "frequency" else "s"
    return f"{lo:.3g} - {hi:.3g} {unit}"


def select_bands(freq: np.ndarray, *, per_decade: int, axis: str,
                 min_n: int) -> List[Tuple[str, np.ndarray]]:
    """[("all bands", mask), (band label, mask), ...]; sparse bands dropped."""
    bidx = band_index(freq, per_decade=per_decade, axis=axis)
    sel = [("all bands", np.ones(freq.size, bool))]
    for k in np.unique(bidx):
        m = bidx == k
        if np.count_nonzero(m) >= min_n:
            sel.append((band_label(int(k), per_decade=per_decade, axis=axis), m))
    return sel


def _rms(a: np.ndarray) -> float:
    return float(inv.nrms(a))


def plot_residual_histograms(
    title: str, freq: np.ndarray, r1: np.ndarray, r2: np.ndarray, *,
    has_syn: bool, per_decade: int, axis: str, ncols: int, panel_size: float,
    limit: float, bin_width: float, min_n: int, run_label: str = "",
):
    """
    Histograms of normalized residuals: first panel pools all bands, then
    one panel per frequency (period) band. Bars: residuals normalized by
    sqrt(err^2 + eps_syn^2) (RMS2) if `has_syn`, else by err (RMS1). With
    `has_syn` the RMS1 histogram is added as a step outline. Red: N(0, RMS^2)
    of the bars; blue dashed: +/- RMS of the bars (Baba 2023, Fig. 10a).

    Returns (figure, stats) where stats is a list of dicts with keys
    band, n, m1, m2 (fit_measures of the RMS1- / RMS2-normalized residuals),
    n_out (values of the plotted variant outside +/- limit).
    """
    import matplotlib.pyplot as plt

    sel = select_bands(freq, per_decade=per_decade, axis=axis, min_n=min_n)

    n = len(sel)
    nc = max(1, min(ncols, n))
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(
        nr, nc, figsize=(panel_size * nc + 0.4, panel_size * nr * 0.85 + 0.6),
        squeeze=False, sharex=True, constrained_layout=True)
    axf = axes.ravel()

    edges = np.arange(-limit, limit + 0.5 * bin_width, bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    xs = np.linspace(-limit, limit, 400)
    stats: List[dict] = []

    for k, (label, m) in enumerate(sel):
        ax = axf[k]
        a1, a2 = r1[m], r2[m]
        main = a2 if has_syn else a1
        nn = main.size
        rms1, rms2 = _rms(a1), _rms(a2)
        rms_main = rms2 if has_syn else rms1
        n_out = int(np.count_nonzero(np.abs(main) > limit))

        counts, _ = np.histogram(main, bins=edges)
        ax.bar(centres, 100.0 * counts / nn, width=bin_width, color="0.75",
               edgecolor="0.3", linewidth=0.4,
               label="RMS2-normalized" if has_syn else "normalized")
        if has_syn:
            c1, _ = np.histogram(a1, bins=edges)
            ax.step(centres, 100.0 * c1 / nn, where="mid", color="tab:orange",
                    linewidth=1.0, label="RMS1-normalized")
        if np.isfinite(rms_main) and rms_main > 0.0:
            g = 100.0 * bin_width * np.exp(-xs**2 / (2.0 * rms_main**2)) \
                / (rms_main * math.sqrt(2.0 * math.pi))
            ax.plot(xs, g, color="red", linewidth=1.0, label="N(0, RMS^2)")
            for s in (-rms_main, rms_main):
                ax.axvline(s, color="tab:blue", linestyle="--", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        txt = (f"n={nn}\nRMS1={rms1:.2f}\nRMS2={rms2:.2f}" if has_syn
               else f"n={nn}\nRMS={rms1:.2f}")
        if n_out:
            txt += f"\nout: {n_out}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("normalized residual", fontsize=9)
        ax.set_ylabel("frequency [%]", fontsize=9)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
        stats.append(dict(band=label, n=nn, m1=fit_measures(a1),
                          m2=fit_measures(a2), n_out=n_out))
    for ax in axf[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{title}: normalized residuals, {axis} bands "
                 f"({per_decade} per decade)" + (f"\n{run_label}" if run_label else ""),
                 fontsize=12)
    return fig, stats


def qq_points(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normal Q-Q coordinates: theoretical N(0,1) quantiles at plotting
    positions p_i = (i - 0.5) / n, and the sorted sample.
    """
    from scipy.stats import norm
    a = np.sort(a[np.isfinite(a)])
    p = (np.arange(1, a.size + 1) - 0.5) / a.size
    return norm.ppf(p), a


def qq_quartile_line(sample_sorted: np.ndarray) -> Tuple[float, float]:
    """(slope, intercept) of the line through the sample's and the standard
    normal's first and third quartiles (robust scale and location)."""
    from scipy.stats import norm
    q1, q3 = np.quantile(sample_sorted, [0.25, 0.75])
    z1, z3 = norm.ppf([0.25, 0.75])
    slope = float((q3 - q1) / (z3 - z1))
    return slope, float(q1 - slope * z1)


def plot_qq_bands(
    title: str, freq: np.ndarray, r1: np.ndarray, r2: np.ndarray, *,
    has_syn: bool, per_decade: int, axis: str, ncols: int, panel_size: float,
    limit: float, confidence: Optional[float], min_n: int,
    equal_axes: bool = False, run_label: str = "",
):
    """
    Normal Q-Q plots of the normalized residuals: first panel pools all
    bands, then one panel per frequency (period) band.

    Dots: residuals normalized by sqrt(err^2 + eps_syn^2) (RMS2) if
    `has_syn`, else by err (RMS1); with `has_syn` the RMS1 residuals are
    added as orange dots. Black line: ideal N(0,1) (calibrated errors).
    Red dashed: quartile line of the plotted dots (slope = robust scale,
    intercept = bias). Gray band: pointwise confidence envelope around the
    quartile line, from the asymptotic standard error of order statistics,
    se_i = sqrt(p_i (1 - p_i) / n) / phi(z_i), so it judges the Gaussian
    shape irrespective of scale and bias. With `equal_axes` both axes get
    the same range (at most +/- `limit`) and the panels are square.

    Returns (figure, stats); stats rows have keys band, n, slope1, slope2,
    icpt, out_pct (percentage of the dots outside the envelope).
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    sel = select_bands(freq, per_decade=per_decade, axis=axis, min_n=min_n)
    n = len(sel)
    nc = max(1, min(ncols, n))
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(
        nr, nc, figsize=(panel_size * nc + 0.4, panel_size * nr + 0.6),
        squeeze=False, sharex=True, sharey=True, constrained_layout=True)
    axf = axes.ravel()

    main_all = r2 if has_syn else r1
    ymax = min(limit, 1.05 * float(np.max(np.abs(main_all)))) if main_all.size else limit
    zc = norm.ppf(0.5 + 0.5 * confidence) if confidence else None
    stats: List[dict] = []

    for k, (label, m) in enumerate(sel):
        ax = axf[k]
        a1, a2 = r1[m], r2[m]
        main = a2 if has_syn else a1
        z, q = qq_points(main)
        nn = q.size
        slope2, icpt = qq_quartile_line(q)
        slope1 = qq_quartile_line(np.sort(a1))[0] if has_syn else slope2

        if has_syn:
            z1, q1 = qq_points(a1)
            ax.scatter(z1, q1, s=6, color="tab:orange", alpha=0.7,
                       edgecolors="none", label="RMS1-normalized", zorder=2)
        ax.scatter(z, q, s=6, color="0.25", edgecolors="none", zorder=3,
                   label="RMS2-normalized" if has_syn else "normalized")
        zl = np.array([z.min(), z.max()])
        ax.plot(zl, zl, color="k", linewidth=0.8, label="N(0,1)", zorder=4)
        fit = icpt + slope2 * z
        ax.plot(z, fit, color="red", linestyle="--", linewidth=0.9,
                label="quartile line", zorder=4)
        out_pct = float("nan")
        if zc is not None:
            p = (np.arange(1, nn + 1) - 0.5) / nn
            se = np.sqrt(p * (1.0 - p) / nn) / norm.pdf(z)
            lo, hi = fit - zc * slope2 * se, fit + zc * slope2 * se
            ax.fill_between(z, lo, hi, color="0.5", alpha=0.25, linewidth=0, zorder=1)
            out_pct = 100.0 * float(np.count_nonzero((q < lo) | (q > hi))) / nn
        txt = (f"n={nn}\nslope1={slope1:.2f}\nslope2={slope2:.2f}" if has_syn
               else f"n={nn}\nslope={slope2:.2f}")
        txt += f"\nicpt={icpt:.2f}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("theoretical N(0,1) quantile", fontsize=9)
        ax.set_ylabel("sample quantile", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        if k == 0:
            ax.legend(fontsize=7, loc="lower right")
        stats.append(dict(band=label, n=nn, slope1=slope1, slope2=slope2,
                          icpt=icpt, out_pct=out_pct))
    if equal_axes:
        zmax = float(np.max(np.abs(qq_points(main_all)[0]))) if main_all.size else 1.0
        lim = min(limit, max(ymax, 1.05 * zmax))
        axf[0].set_xlim(-lim, lim)
        axf[0].set_ylim(-lim, lim)
        for ax in axf[:n]:
            ax.set_aspect("equal", adjustable="box")
    else:
        axf[0].set_ylim(-ymax, ymax)
    for ax in axf[n:]:
        ax.set_visible(False)
    env = f", {100 * confidence:.0f}% envelope" if confidence else ""
    fig.suptitle(f"{title}: normal Q-Q, {axis} bands ({per_decade} per decade){env}"
                 + (f"\n{run_label}" if run_label else ""), fontsize=12)
    return fig, stats


def group_panels(panels: Dict[str, List[dict]], *, by_component: bool) -> Dict[str, List[dict]]:
    """Group panels for the histograms: per data type, or per data type + component."""
    groups: Dict[str, List[dict]] = {}
    for dname, lst in panels.items():
        if by_component:
            for p in lst:
                groups.setdefault(f"{dname}_{p['component']}", []).append(p)
        else:
            groups[dname] = list(lst)
    return groups


# ---------------------------------------------------------------------------
# Method 4 (optional): response curves per site -- observed data and the
# calculated response, single run or ensemble (curves / density / bands)
# ---------------------------------------------------------------------------
CURVE_PANELS = ("rho", "phase", "tipper", "pt")
_CURVE_PLOTTER = {"rho": "add_rho", "phase": "add_phase",
                  "tipper": "add_tipper", "pt": "add_pt"}
_COMP_COLOR = {"xy": "tab:blue", "yx": "tab:red", "xx": "tab:green",
               "yy": "tab:orange"}
_IDX_COLOR = ("tab:blue", "tab:orange", "tab:green", "tab:red")
_IDX_MARKER = ("o", "^", "s", "d")                 # as data_viz.add_tipper/add_pt
_TIPPER_COLS = ("Tx_re", "Tx_im", "Ty_re", "Ty_im")
_PT_COLS = ("ptxx_re", "ptxy_re", "ptyx_re", "ptyy_re")
_CURVE_TYPES = ("MT", "NMT2", "APP_RES_AND_PHS", "NMT2_APP_RES_AND_PHS",
                "VTF", "PT")                       # data types that can be drawn
_CURVE_LAYERS = ("curves", "density", "bands")


def _curves_cfg() -> dict:
    """Snapshot of the CURVES_* settings (read when the method is run)."""
    return dict(
        what=tuple(CURVES_WHAT), comps=CURVES_COMPS, sites=CURVES_SITES,
        site_order=CURVES_SITE_ORDER, max_sites=CURVES_MAX_SITES,
        ens_mode=CURVES_ENS_MODE, max_curves=CURVES_MAX_CURVES,
        subsample=CURVES_SUBSAMPLE, member_alpha=CURVES_MEMBER_ALPHA,
        median=CURVES_MEDIAN, band_pct=tuple(CURVES_BAND_PCT),
        dens_nx=CURVES_DENSITY_NX, dens_ny=CURVES_DENSITY_NY,
        dens_gamma=CURVES_DENSITY_GAMMA, seed=CURVES_SEED,
        obs_only=bool(CURVES_OBSERVED_ONLY),
        obs_errors=CURVES_OBS_ERRORS, obs_ms=CURVES_OBS_MARKERSIZE,
        lw=CURVES_LINEWIDTH, invert_x=CURVES_INVERT_X,
        period_lim=CURVES_PERIOD_LIM, ylim=dict(CURVES_YLIM),
        panel_size=tuple(CURVES_PANEL_SIZE), multipage=CURVES_MULTIPAGE,
        appres_deg=CURVES_APPRES_PHASE_DEG, legend_loc=CURVES_LEGEND_LOC,
        site_dat=CURVES_SITE_DAT, id_offset=CURVES_SITE_ID_OFFSET,
    )


def load_site_names(spec: Optional[str], search_dirs: Sequence[str], offset: int,
                    site_ids: Sequence[int]) -> Dict[int, str]:
    """
    Site names by site_id from a site.dat written by mt_make_sitelist.py
    (read with femtic.read_site_dat): site_id = sitenum + offset.

    spec: None -> {}; "auto" -> first site.dat found in `search_dirs`; else a
    path. A missing or unreadable file only gives a message and {}.
    """
    if not spec:
        return {}
    path = None
    if spec == "auto":
        for d in search_dirs:
            p = os.path.join(d, "site.dat")
            if os.path.isfile(p):
                path = p
                break
        if path is None:
            print("femtic_data_misfit: curves: no site.dat found "
                  f"(looked in {list(search_dirs)}); sites are labelled by id.")
            return {}
    else:
        path = spec
    try:
        rows = fem.read_site_dat(path)
    except (OSError, ValueError) as exc:
        print(f"femtic_data_misfit: curves: cannot read site.dat ({exc}); "
              "sites are labelled by id.")
        return {}
    names = {int(r["sitenum"]) + int(offset): str(r["name"]) for r in rows}
    hit = sum(1 for s in site_ids if s in names)
    print(f"femtic_data_misfit: curves: {len(names)} site names from {path}; "
          f"{hit} of {len(site_ids)} site ids matched (site_id = sitenum + {offset}).")
    if hit < len(site_ids):
        print("femtic_data_misfit: curves: WARNING: not all site ids have a name; "
              "check CURVES_SITE_ID_OFFSET and that site.dat belongs to this data set.")
    return names


def _curve_modules():
    """
    Import data_viz (required) and femtic_viz (optional; only its component
    marker convention is used). Returns (data_viz, markers, comp_class).
    """
    import data_viz as dv  # noqa: WPS433 (imported only when "curves" runs)
    try:
        import femtic_viz as fv
        markers = dict(fv.DEFAULT_COMP_MARKERS)
        comp_class = fv._comp_class
    except Exception:
        markers = {"ii": "o", "ij": "s", "inv": "^"}

        def comp_class(comp: str) -> str:
            c = str(comp).strip().lower()
            return "ii" if c in ("xx", "yy") else ("ij" if c in ("xy", "yx") else "inv")
    return dv, markers, comp_class


def site_nrms(ds: dict) -> Dict[int, dict]:
    """
    Per site: number of residual values n, conventional nRMS (RMS1 reference,
    real and imaginary parts, as in the other methods) and the site x, y as
    stored in the results file. Keys are the integer site ids.
    """
    n = ds["freq"].size
    cpx = ~np.isin(ds["datatype_name"], REAL_ONLY_TYPES)
    parts = []
    for ko, kr, ke, msk in (("obs_re", "ref_re", "err_re", np.ones(n, bool)),
                            ("obs_im", "ref_im", "err_im", cpx)):
        err = ds[ke]
        ok = (msk & np.isfinite(err) & (err > 0.0)
              & np.isfinite(ds[ko]) & np.isfinite(ds[kr]))
        r = np.full(n, np.nan)
        r[ok] = (ds[ko][ok] - ds[kr][ok]) / err[ok]
        parts.append(r)
    sid = ds["site_id"]
    out: Dict[int, dict] = {}
    for s in np.unique(sid):
        m = sid == s
        r = np.concatenate([p[m] for p in parts])
        r = r[np.isfinite(r)]
        i0 = int(np.flatnonzero(m)[0])
        out[int(s)] = dict(
            n=int(r.size),
            nrms=float(inv.nrms(r)) if r.size else float("nan"),
            x=float(ds["site_x"][i0]), y=float(ds["site_y"][i0]))
    return out


def select_sites(info: Dict[int, dict], spec, order: str, max_sites: int) -> List[int]:
    """
    Choose the sites to plot from the per-site table `info` (see site_nrms).

    spec  None -> all; int N -> N sites; sequence -> exactly these sites, given
          as site ids (int) and/or site names (str, if info has "name").
    order for an int spec: "even" (evenly spread over the sorted site ids),
          "id" (first N), "worst" / "best" (largest / smallest nRMS first).
          Without an int spec "worst"/"best" still order the whole list.
    """
    ids = sorted(info)
    if spec is not None and not isinstance(spec, (int, np.integer)):
        by_name = {d["name"]: s for s, d in info.items() if d.get("name")}
        sel, miss = [], []
        for item in spec:
            if isinstance(item, str):
                if item in by_name:
                    sel.append(by_name[item])
                elif item.strip().isdigit() and int(item) in info:
                    sel.append(int(item))
                else:
                    miss.append(item)
            elif int(item) in info:
                sel.append(int(item))
            else:
                miss.append(item)
        if miss:
            print(f"femtic_data_misfit: curves: sites not in the data: {miss}")
    else:
        if order not in ("even", "id", "worst", "best"):
            raise ValueError(f"CURVES_SITE_ORDER {order!r} not in even/id/worst/best.")

        def key(s: int) -> float:
            v = info[s]["nrms"]
            return v if np.isfinite(v) else float("-inf")

        if order == "worst":
            ids = sorted(ids, key=lambda s: -key(s))
        elif order == "best":
            ids = sorted(ids, key=lambda s: (not np.isfinite(info[s]["nrms"]), key(s)))
        if spec is None:
            sel = ids
        else:
            n = max(0, min(int(spec), len(ids)))
            if order == "even" and n > 0:
                pos = np.unique(np.round(np.linspace(0, len(ids) - 1, n)).astype(int))
                sel = [ids[i] for i in pos]
            else:
                sel = ids[:n]
    if max_sites and len(sel) > max_sites:
        print(f"femtic_data_misfit: curves: {len(sel)} sites selected, "
              f"limited to CURVES_MAX_SITES = {max_sites}.")
        sel = sel[:max_sites]
    return sel


def site_series(ds: dict, sid: int, *, appres_deg: bool = True) -> Dict[str, dict]:
    """
    Plot-ready series of one site, keyed by data_viz column name (rho_xy,
    phi_xy, Tx_re, Tx_im, ptxy_re, ...). Each value is a dict with panel
    ("rho"/"phase"/"tipper"/"pt"), freq, period (ascending), obs, err (NaN
    where not usable) and cal, shape (M, n) -- one row per ensemble member.

    Same conventions as data_viz.datadict_to_plot_df: rho_a = |Z|^2/(mu0*omega)
    (FEMTIC's Z is in SI ohm), phase = atan2(Im, Re) in degrees, and the
    errors of rho_a and phase from the complex modulus of the errors of Z.
    One deliberate difference: the calculated phase is put on the branch of
    the observed phase (differences wrapped to +-180 degrees), so members that
    cross +-180 do not jump by 360.
    """
    mu0 = 4.0e-7 * np.pi
    sel = ds["site_id"] == sid
    dn, cn = ds["datatype_name"], ds["component_name"]
    out: Dict[str, dict] = {}

    def put(col: str, panel: str, f, obs, err, cal) -> None:
        good = np.isfinite(obs) & np.isfinite(f) & (f > 0.0)
        if not good.any():
            return
        order = np.argsort(-f[good])                   # period ascending
        f_ = f[good][order]
        e_ = err[good][order]
        o_ = obs[good][order]
        c_ = cal[:, good][:, order]
        if panel == "phase":
            # calculated phase on the branch of the observed phase (degrees):
            # members that cross +-180 (typical for the yx component, ~ -180)
            # would otherwise jump by 360 and ruin medians and densities
            c_ = o_ + (c_ - o_ + 180.0) % 360.0 - 180.0
        out[col] = dict(
            panel=panel, freq=f_, period=1.0 / f_, obs=o_,
            err=np.where(np.isfinite(e_) & (e_ > 0.0), e_, np.nan), cal=c_)

    with np.errstate(divide="ignore", invalid="ignore"):
        for dname in np.unique(dn[sel]):
            md = sel & (dn == dname)
            for cname in np.unique(cn[md]):
                m = md & (cn == cname)
                f = ds["freq"][m]
                o_re, o_im = ds["obs_re"][m], ds["obs_im"][m]
                e_re, e_im = ds["err_re"][m], ds["err_im"][m]
                c_re, c_im = ds["cal_re_m"][:, m], ds["cal_im_m"][:, m]
                lc = str(cname).lower()
                if dname in ("MT", "NMT2") and str(cname).startswith("Z"):
                    c = lc[1:]
                    om = 2.0 * np.pi * f
                    zabs = np.hypot(o_re, o_im)
                    sig = np.hypot(e_re, e_im)
                    put(f"rho_{c}", "rho", f, zabs**2 / (mu0 * om),
                        2.0 * zabs * sig / (mu0 * om),
                        (c_re**2 + c_im**2) / (mu0 * om))
                    put(f"phi_{c}", "phase", f, np.degrees(np.arctan2(o_im, o_re)),
                        np.degrees(sig / np.where(zabs > 0.0, zabs, np.nan)),
                        np.degrees(np.arctan2(c_im, c_re)))
                elif dname in ("APP_RES_AND_PHS", "NMT2_APP_RES_AND_PHS"):
                    if lc.startswith("rho"):
                        put("rho_" + lc[3:], "rho", f, o_re, e_re, c_re)
                    elif lc.startswith("phs"):
                        k = 1.0 if appres_deg else 180.0 / np.pi
                        put("phi_" + lc[3:], "phase", f, o_re * k, e_re * k, c_re * k)
                elif dname == "VTF" and str(cname).startswith("Tz"):
                    c = lc[2:]                         # "x" | "y"
                    put(f"T{c}_re", "tipper", f, o_re, e_re, c_re)
                    put(f"T{c}_im", "tipper", f, o_im, e_im, c_im)
                elif dname == "PT" and str(cname).startswith("PT"):
                    put(f"pt{lc[2:]}_re", "pt", f, o_re, e_re, c_re)
    return out


def _series_label(col: str) -> str:
    if col.startswith("rho_"):
        return "rho " + col[4:].upper()
    if col.startswith("phi_"):
        return "phase " + col[4:].upper()
    if col.startswith("pt"):
        return "PT" + col[2:4]
    return ("Re(" if col.endswith("_re") else "Im(") + col[:2] + ")"


def resolve_layers(mode: str, M: int, max_curves: int) -> List[str]:
    """Ensemble display layers for M members ([] for a single run)."""
    if M <= 1:
        return []
    mode = str(mode).strip().lower()
    if mode == "auto":
        return ["curves"] if M <= max_curves else ["density"]
    layers = [s.strip() for s in mode.split("+") if s.strip()]
    bad = [s for s in layers if s not in _CURVE_LAYERS]
    if bad or not layers:
        raise ValueError(f"CURVES_ENS_MODE {mode!r}: use auto or a '+'-joined "
                         f"combination of {_CURVE_LAYERS}.")
    return layers


def _draw(ax, plotter, frame, col: str, panel: str, *, color, ls, lw, alpha,
          marker, ms, zorder, show_errors: bool = False) -> None:
    """
    Draw one series with a data_viz plotter and then normalize what it drew.

    The plotters pick their own marker/colour defaults, toggle the x-axis
    direction on every call and colour error shading from the colour cycle,
    so after the call: the axis direction is restored, the new lines get the
    requested marker/size/zorder, and new shading gets `color`.
    """
    n_lines, n_coll = len(ax.lines), len(ax.collections)
    was_inv = ax.xaxis_inverted()
    kw = dict(ax=ax, legend=False, show_errors=show_errors, color=color,
              linestyle=ls, linewidth=lw, alpha=alpha)
    if show_errors:
        kw["error_alpha"] = 0.2
    if panel in ("rho", "phase"):
        kw["comps"] = col.split("_", 1)[1]
    plotter(frame, **kw)
    if ax.xaxis_inverted() != was_inv:
        ax.invert_xaxis()
    for ln in list(ax.lines)[n_lines:]:
        ln.set_marker(marker if marker else "none")
        ln.set_markersize(ms)
        ln.set_zorder(zorder)
    for cl in list(ax.collections)[n_coll:]:
        cl.set_facecolor(color)
        cl.set_zorder(zorder - 0.5)


def draw_density(ax, period: np.ndarray, y: np.ndarray, *, log_y: bool, color,
                 nx: int, ny: int, gamma: float, zorder: float = 1.0) -> bool:
    """
    Per-period density of many curves. `y` has shape (M, n) at the ascending
    `period` (n,). Every member curve is linearly interpolated (in log period,
    and in log10 y if `log_y`) onto `nx` grid points, binned in `ny` y-bins,
    and drawn with pcolormesh in `color` whose opacity is the fraction of
    members in the bin (PowerNorm with `gamma`; gamma < 1 lifts sparse
    regions). Returns False if nothing could be drawn.
    """
    import matplotlib.colors as mcolors

    n = period.size
    if n < 2 or y.shape[0] < 2:
        return False
    lx = np.log10(period)
    xg = np.linspace(lx[0], lx[-1], nx)
    idx = np.clip(np.searchsorted(lx, xg, side="right") - 1, 0, n - 2)
    dx = lx[idx + 1] - lx[idx]
    t = np.where(dx > 0.0, (xg - lx[idx]) / np.where(dx > 0.0, dx, 1.0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        yv = np.where(y > 0.0, np.log10(y), np.nan) if log_y else np.asarray(y, float)
    yi = yv[:, idx] * (1.0 - t) + yv[:, idx + 1] * t          # (M, nx)
    fin = np.isfinite(yi)
    if not fin.any():
        return False
    lo, hi = np.nanpercentile(yi, [0.25, 99.75])
    pad = 0.05 * (hi - lo) if hi > lo else (0.25 if log_y else max(1e-3, 0.05 * abs(lo)))
    lo, hi = lo - pad, hi + pad
    dy = (hi - lo) / ny
    b = np.floor((np.where(fin, yi, lo) - lo) / dy)
    ok = fin & (b >= 0) & (b < ny)
    col = np.broadcast_to(np.arange(nx), yi.shape)
    flat = b[ok].astype(np.int64) + ny * col[ok]
    counts = np.bincount(flat, minlength=ny * nx).reshape(nx, ny).T.astype(float)
    nvalid = fin.sum(axis=0).astype(float)
    dens = counts / np.where(nvalid > 0.0, nvalid, 1.0)
    if not np.any(dens > 0.0):
        return False
    step = xg[1] - xg[0]
    xe = np.concatenate(([xg[0] - 0.5 * step], 0.5 * (xg[:-1] + xg[1:]),
                         [xg[-1] + 0.5 * step]))
    ye = lo + dy * np.arange(ny + 1)
    X = 10.0**xe
    Y = 10.0**ye if log_y else ye
    rgb = mcolors.to_rgb(color)
    cmap = mcolors.LinearSegmentedColormap.from_list("dens", [rgb + (0.0,), rgb + (1.0,)])
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    norm = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=float(dens.max()))
    ax.pcolormesh(X, Y, np.ma.masked_less_equal(dens, 0.0), cmap=cmap, norm=norm,
                  shading="flat", rasterized=True, zorder=zorder)
    return True


def draw_bands(ax, period: np.ndarray, y: np.ndarray, *, pct, color,
               zorder: float = 1.5) -> None:
    """Percentile envelopes p..100-p of the member curves (one fill per p)."""
    for p in pct:
        lo, hi = np.nanpercentile(y, [p, 100.0 - p], axis=0)
        ax.fill_between(period, lo, hi, color=color, alpha=0.18, linewidth=0,
                        zorder=zorder)


def plot_site_curves(sid: int, series: Dict[str, dict], info: dict, *, M: int,
                     layers: Sequence[str], cfg: dict, title_extra: str,
                     dv, markers: dict, comp_class, rng):
    """
    One figure for one site: a panel per quantity in cfg["what"] that has data.
    Returns the figure, or None if the site has nothing to draw.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    comps = [c.strip().lower() for c in str(cfg["comps"]).split(",") if c.strip()]
    panels: List[Tuple[str, List[str]]] = []
    for w in cfg["what"]:
        if w == "rho":
            cols = [f"rho_{c}" for c in comps]
        elif w == "phase":
            cols = [f"phi_{c}" for c in comps]
        elif w == "tipper":
            cols = list(_TIPPER_COLS)
        else:
            cols = list(_PT_COLS)
        cols = [c for c in cols if c in series]
        if cols:
            panels.append((w, cols))
    if not panels:
        return None

    nc = len(panels)
    fig, axes = plt.subplots(1, nc, figsize=(cfg["panel_size"][0] * nc, cfg["panel_size"][1]),
                             squeeze=False, constrained_layout=True)
    obs_only = cfg["obs_only"]
    if obs_only:
        layers = []
    if M > 1 and "curves" in layers:
        sub = (np.arange(M) if M <= cfg["subsample"]
               else np.sort(rng.choice(M, size=cfg["subsample"], replace=False)))
        alpha_m = cfg["member_alpha"] or float(np.clip(3.0 / np.sqrt(sub.size), 0.05, 0.6))
    else:
        sub, alpha_m = np.arange(0), 0.3

    def frame(s: dict, col: str, vals, err=None):
        d = {"freq": s["freq"], "period": s["period"], col: vals}
        if err is not None:
            d[col + "_err"] = err
        return pd.DataFrame(d)

    for ax, (w, cols) in zip(axes[0], panels):
        plotter = getattr(dv, _CURVE_PLOTTER[w])
        log_y = (w == "rho")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        handles = []
        for k, col in enumerate(cols):
            s = series[col]
            comp = col.split("_", 1)[1] if w in ("rho", "phase") else ""
            color = (_COMP_COLOR.get(comp, _IDX_COLOR[k % 4]) if w in ("rho", "phase")
                     else _IDX_COLOR[k % 4])
            mk = (markers.get(comp_class(comp), "o") if w in ("rho", "phase")
                  else _IDX_MARKER[k % 4])
            handles.append(Line2D([], [], color=color, lw=1.5, label=_series_label(col)))
            cal = s["cal"]
            if obs_only:
                pass
            elif M > 1:
                if "density" in layers:
                    draw_density(ax, s["period"], cal, log_y=log_y, color=color,
                                 nx=cfg["dens_nx"], ny=cfg["dens_ny"],
                                 gamma=cfg["dens_gamma"])
                if "bands" in layers:
                    draw_bands(ax, s["period"], cal, pct=cfg["band_pct"], color=color)
                if "curves" in layers:
                    for j in sub:
                        _draw(ax, plotter, frame(s, col, cal[j]), col, w, color=color,
                              ls="-", lw=0.6, alpha=alpha_m, marker=None, ms=0.0,
                              zorder=2.0)
                if cfg["median"]:
                    _draw(ax, plotter, frame(s, col, np.nanmedian(cal, axis=0)), col, w,
                          color=color, ls="-", lw=cfg["lw"] + 0.4, alpha=1.0,
                          marker=None, ms=0.0, zorder=3.0)
            else:
                _draw(ax, plotter, frame(s, col, cal[0]), col, w, color=color, ls="-",
                      lw=cfg["lw"], alpha=1.0, marker=None, ms=0.0, zorder=3.0)
            # observed data on top
            emode = cfg["obs_errors"]
            _draw(ax, plotter, frame(s, col, s["obs"], s["err"] if emode == "shade" else None),
                  col, w, color=color, ls="none", lw=0.0, alpha=1.0, marker=mk,
                  ms=cfg["obs_ms"], zorder=5.0, show_errors=(emode == "shade"))
            if emode == "bar":
                err, y = s["err"], s["obs"]
                good = np.isfinite(err) & (err > 0.0)
                lo = np.minimum(err, 0.9 * y) if log_y else err
                if good.any():
                    ax.errorbar(s["period"][good], y[good],
                                yerr=[lo[good], err[good]], fmt="none", ecolor=color,
                                elinewidth=0.7, capsize=1.5, zorder=4.5)
        # style entries (what the marker / line / shading means)
        style = [Line2D([], [], color="0.3", marker="o", ls="none", ms=3.5, label="observed")]
        if obs_only:
            pass
        elif M > 1:
            if "density" in layers:
                style.append(Patch(facecolor="0.6", alpha=0.6, label=f"density (M={M})"))
            if "bands" in layers:
                style.append(Patch(facecolor="0.6", alpha=0.35, label="percentile bands"))
            if "curves" in layers:
                style.append(Line2D([], [], color="0.5", lw=0.8, label=f"members ({sub.size})"))
            if cfg["median"]:
                style.append(Line2D([], [], color="0.2", lw=1.6, label="ensemble median"))
        else:
            style.append(Line2D([], [], color="0.3", lw=1.4, label="calculated"))
        allh = handles + style
        if cfg["legend_loc"] == "below":
            ax.legend(handles=allh, fontsize=6.5, loc="upper center",
                      bbox_to_anchor=(0.5, -0.22), frameon=False,
                      ncol=min(4, len(allh)))
        elif cfg["legend_loc"]:
            ax.legend(handles=allh, fontsize=6.5, loc=cfg["legend_loc"], framealpha=0.75,
                      ncol=2 if len(allh) > 5 else 1)
        ax.tick_params(labelsize=8)
        if cfg["period_lim"] is not None:
            cur = sorted(ax.get_xlim())
            lo, hi = cfg["period_lim"]
            ax.set_xlim(cur[0] if lo is None else lo, cur[1] if hi is None else hi)
        yl = cfg["ylim"].get(w)
        if yl is not None:
            cur = sorted(ax.get_ylim())
            ax.set_ylim(cur[0] if yl[0] is None else yl[0], cur[1] if yl[1] is None else yl[1])
        if cfg["invert_x"] != ax.xaxis_inverted():
            ax.invert_xaxis()
    who = f"site {info['name']} (id {sid}" if info.get("name") else f"site {sid} ("
    fit = "" if obs_only else f"nRMS = {info['nrms']:.2f}, "
    fig.suptitle(f"{who}, x={info['x']:g}, y={info['y']:g}), {fit}"
                 f"n = {info['n']}{' (observed only)' if obs_only else ''}\n{title_extra}",
                 fontsize=10)
    return fig


def format_site_nrms(info: Dict[int, dict]) -> str:
    hdr = (f"{'site_id':>8} {'name':<16} {'site_x':>14} {'site_y':>14} {'n':>7} "
           f"{'nrms':>8}")
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for s in sorted(info):
        d = info[s]
        lines.append(f"{s:>8d} {d.get('name', ''):<16} {d['x']:>14.6g} {d['y']:>14.6g} "
                     f"{d['n']:>7d} {d['nrms']:>8.3f}")
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def run_curves(ds: dict, *, out_dir: str, prefix: str, title_extra: str,
               search_dirs: Sequence[str] = ()) -> int:
    """
    Method "curves": one figure per selected site (see the CURVES_* settings).
    Returns 0 on success, 1 if data_viz is missing or the settings are invalid.
    """
    import matplotlib.pyplot as plt

    try:
        dv, markers, comp_class = _curve_modules()
    except ImportError as exc:
        print(f"femtic_data_misfit: curves: needs data_viz.py on the path ({exc}).")
        return 1
    cfg = _curves_cfg()
    bad = [w for w in cfg["what"] if w not in CURVE_PANELS]
    if bad or not cfg["what"]:
        print(f"femtic_data_misfit: curves: CURVES_WHAT entries must be from "
              f"{CURVE_PANELS}; got {tuple(cfg['what'])}.")
        return 1
    if cfg["obs_errors"] not in ("bar", "shade", "none"):
        print("femtic_data_misfit: curves: CURVES_OBS_ERRORS must be bar/shade/none.")
        return 1
    M = ds["M"]
    try:
        layers = resolve_layers(cfg["ens_mode"], M, cfg["max_curves"])
    except ValueError as exc:
        print(f"femtic_data_misfit: curves: {exc}")
        return 1

    skipped = sorted(set(np.unique(ds["datatype_name"]).tolist()) - set(_CURVE_TYPES))
    if skipped:
        print(f"femtic_data_misfit: curves: no plotter for data types {skipped}, skipped.")
    info = site_nrms(ds)
    names = load_site_names(cfg["site_dat"], search_dirs, cfg["id_offset"], sorted(info))
    for s, nm in names.items():
        if s in info:
            info[s]["name"] = nm
    if WRITE_STATS:
        _write(os.path.join(out_dir, f"{prefix}_site_nrms.txt"),
               f"# {title_extra}; per-site nRMS, conventional normalization (RMS1)\n"
               "# site_x / site_y as stored in the results file; name from site.dat",
               format_site_nrms(info))
    try:
        sids = select_sites(info, cfg["sites"], cfg["site_order"], cfg["max_sites"])
    except ValueError as exc:
        print(f"femtic_data_misfit: curves: {exc}")
        return 1
    if cfg["obs_only"]:
        layers = []
    print(f"femtic_data_misfit: curves: {len(sids)} of {len(info)} sites, M={M}, "
          f"layers={'observed only' if cfg['obs_only'] else '+'.join(layers) if layers else 'single run'}")
    if not sids:
        print("femtic_data_misfit: curves: no sites selected, nothing to plot.")
        return 0

    exts = ["." + str(e).lstrip(".") for e in
            ((PLOT_FORMATS,) if isinstance(PLOT_FORMATS, str) else PLOT_FORMATS)]
    multipage = bool(cfg["multipage"]) and ".pdf" in exts
    pdf_path = os.path.join(out_dir, f"{prefix}_curves.pdf")
    pdf = None                                    # opened with the first page
    other = [e for e in exts if not (multipage and e == ".pdf")]
    sub_dir = os.path.join(out_dir, f"{prefix}_curves")
    if other:
        os.makedirs(sub_dir, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    n_done = 0
    try:
        for sid in sids:
            series = site_series(ds, sid, appres_deg=cfg["appres_deg"])
            fig = plot_site_curves(sid, series, info[sid], M=M, layers=layers, cfg=cfg,
                                   title_extra=title_extra, dv=dv, markers=markers,
                                   comp_class=comp_class, rng=rng)
            if fig is None:
                print(f"femtic_data_misfit: curves: site {sid}: nothing to draw, skipped.")
                continue
            if multipage:
                if pdf is None:
                    from matplotlib.backends.backend_pdf import PdfPages
                    pdf = PdfPages(pdf_path)
                pdf.savefig(fig, dpi=PLOT_DPI)
            if other:
                tag = f"{sid}_{_safe_name(info[sid]['name'])}" if info[sid].get("name") else f"{sid}"
                _save(fig, os.path.join(sub_dir, f"{prefix}_curves_site{tag}"), other, PLOT_DPI)
            if SHOW:
                plt.show()
            plt.close(fig)
            n_done += 1
    finally:
        if pdf is not None:
            pdf.close()
            print(f"femtic_data_misfit: wrote {pdf_path} ({n_done} pages)")
    return 0


# ---------------------------------------------------------------------------
# Text tables (plain ASCII)
# ---------------------------------------------------------------------------
FIT_MEASURE_NOTE = (
    "# r = error-normalized residuals; norm 'err' = conventional (Baba RMS1),\n"
    "# 'err+syn' = obs + synthesized error (Baba RMS2, ensemble only).\n"
    "# nrms = sqrt(mean(r^2)); r_mae = sqrt(pi/2)*mean(|r|); r_med = median(|r|)/0.6745;\n"
    "# q95 = 95th percentile of |r|; mean = bias.\n"
    "# Expected for N(0,1) residuals: nrms = r_mae = r_med = 1, q95 = 1.96, mean = 0."
)
_MK = ("nrms", "r_mae", "r_med", "q95")
_MN = {"nrms": "nrms", "r_mae": "rmae", "r_med": "rmed", "q95": "q95"}


def format_crossplot_stats(panels: Dict[str, List[dict]], *, has_syn: bool) -> str:
    cols = [f"{_MN[k]}{sfx}" for k in _MK for sfx in (("1", "2") if has_syn else ("",))]
    hdr = (f"{'datatype':<22} {'component':<8} {'part':<5} {'n':>7} "
           f"{'dropped':>8} {'rms':>12}" + "".join(f" {c:>8}" for c in cols))
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for dname, lst in panels.items():
        for p in lst:
            s = panel_stats(p)
            vals = [s[m][k] for k in _MK for m in (("m1", "m2") if has_syn else ("m1",))]
            lines.append(f"{dname:<22} {p['component']:<8} {p['part']:<5} {s['n']:>7d} "
                         f"{s['n_dropped']:>8d} {s['rms']:>12.4e}"
                         + "".join(f" {v:>8.3f}" for v in vals))
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def _measure_rows(label_a: str, label_b: str, n: int, m1: dict, m2: dict, *,
                  has_syn: bool, extra: Optional[Tuple[str, str]] = None) -> List[str]:
    """Table rows (norm 'err' and, if has_syn, 'err+syn') for one residual set."""
    out = []
    variants = [("err", m1)] + ([("err+syn", m2)] if has_syn else [])
    for k, (nm, m) in enumerate(variants):
        ex = "" if extra is None else f" {extra[k] if k < len(extra) else '':>6}"
        out.append(f"{label_a:<26} " + (f"{label_b:<24} " if label_b else "")
                   + f"{nm:<8} {m['n']:>7d} "
                   f"{m['nrms']:>8.3f} {m['r_mae']:>8.3f} {m['r_med']:>8.3f} "
                   f"{m['q95']:>8.3f} {m['mean']:>8.3f}{ex}")
    return out


def _measure_header(first: str, second: str, *, extra: str = "") -> str:
    return (f"{first:<26} " + (f"{second:<24} " if second else "")
            + f"{'norm':<8} {'n':>7} {'nrms':>8} {'r_mae':>8} "
            f"{'r_med':>8} {'q95':>8} {'mean':>8}" + (f" {extra:>6}" if extra else ""))


def format_hist_stats(all_stats: Dict[str, List[dict]], *, has_syn: bool) -> str:
    hdr = _measure_header("group", "band", extra="n_out")
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for g, lst in all_stats.items():
        for s in lst:
            # n_out refers to the plotted variant (err+syn if has_syn, else err)
            extra = ("", str(s["n_out"])) if has_syn else (str(s["n_out"]),)
            lines.extend(_measure_rows(g, s["band"], s["n"], s["m1"], s["m2"],
                                       has_syn=has_syn, extra=extra))
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def format_fit_measures(panels: Dict[str, List[dict]], *, has_syn: bool) -> str:
    """Overall fit measures: pooled over everything ('ALL') and per data type."""
    scopes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for dname, lst in panels.items():
        parts = [normalized_residuals(p) for p in lst]
        scopes[dname] = (np.concatenate([q[1] for q in parts]),
                         np.concatenate([q[2] for q in parts]))
    scopes = {"ALL": (np.concatenate([v[0] for v in scopes.values()]),
                      np.concatenate([v[1] for v in scopes.values()])), **scopes}
    hdr = _measure_header("scope", "")
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for name, (a1, a2) in scopes.items():
        lines.extend(_measure_rows(name, "", a1.size, fit_measures(a1),
                                   fit_measures(a2), has_syn=has_syn))
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def format_qq_stats(all_stats: Dict[str, List[dict]], *, has_syn: bool) -> str:
    hdr = (f"{'group':<26} {'band':<24} {'n':>7} {'slope1':>8}"
           + (f" {'slope2':>8}" if has_syn else "") + f" {'icpt':>8} {'out%':>6}")
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for g, lst in all_stats.items():
        for s in lst:
            lines.append(f"{g:<26} {s['band']:<24} {s['n']:>7d} {s['slope1']:>8.3f}"
                         + (f" {s['slope2']:>8.3f}" if has_syn else "")
                         + f" {s['icpt']:>8.3f} {s['out_pct']:>6.1f}")
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def format_runs(runs: Sequence[dict]) -> str:
    hdr = f"{'#':>4} {'iter':>5} {'cands':>6} {'rms':>9}  path"
    lines = ["=" * 78, hdr, "-" * 78]
    for i, r in enumerate(runs):
        lines.append(f"{i:>4d} {r['iter']:>5d} {r['n_candidates']:>6d} {r['rms']:>9.3f}  {r['path']}")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------
def _save(fig, stem: str, formats: Union[str, Sequence[str]], dpi: int) -> None:
    """
    Save `fig` as stem + ext for every extension in `formats`. A plain string
    (e.g. ".pdf") counts as one format; extensions may be given with or
    without the leading dot; the file format is passed explicitly, so an
    unsupported extension raises instead of silently producing a PNG.
    """
    if isinstance(formats, str):
        formats = (formats,)
    for ext in formats:
        ext = "." + str(ext).lstrip(".")
        out = stem + ext
        fig.savefig(out, dpi=dpi, format=ext[1:])
        print(f"femtic_data_misfit: wrote {out}")


def _write(path: str, header: str, body: str) -> None:
    with open(path, "w", encoding="ascii") as fh:
        fh.write(header + "\n" + body + "\n")
    print(f"femtic_data_misfit: wrote {path}")


def default_prefix(runs: Sequence[dict], mode: str) -> str:
    """File-name prefix: <run>_iter<N> (single), ensemble_M<M> (ensemble)."""
    if mode == "single":
        base = os.path.basename(os.path.normpath(runs[0]["run"]))
        return base if os.path.isfile(runs[0]["run"]) else f"{base}_iter{runs[0]['iter']}"
    return f"ensemble_M{len(runs)}"


def default_out_dir(runs: Sequence[dict]) -> str:
    """
    Directory of the chosen results file(s): the run directory for one run;
    for an ensemble their common parent directory.
    """
    dirs = [os.path.dirname(os.path.abspath(r["path"])) for r in runs]
    if len(set(dirs)) == 1:
        return dirs[0]
    try:
        return os.path.commonpath(dirs)
    except ValueError:
        return os.getcwd()


def process_runs(runs: Sequence[dict], *, mode: str, out_dir: str, prefix: str) -> int:
    """
    Run the selected METHODS on one dataset built from `runs` (one run for
    "single", all members for "ensemble"). Returns 0 on success.
    """
    import matplotlib.pyplot as plt

    ds = build_dataset([r["data"] for r in runs], obs_mode=ENSEMBLE_OBS,
                       ref_member=REF_MEMBER)
    M = ds["M"]
    has_syn = M > 1
    print(f"femtic_data_misfit: mode={mode}, members M={M}, rows={ds['freq'].size}, "
          f"dropped (non-finite calc)={ds['n_dropped_nan']}")
    if mode == "ensemble" and M == 1:
        print("femtic_data_misfit: WARNING: ensemble with one member; RMS2 = RMS1.")
    runs_txt = format_runs(runs)
    if M <= 30:
        print(runs_txt)
    else:
        rr = np.array([r["rms"] for r in runs], float)
        print(f"femtic_data_misfit: member RMS min/median/max = "
              f"{np.nanmin(rr):.3f} / {np.nanmedian(rr):.3f} / {np.nanmax(rr):.3f}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"femtic_data_misfit: output directory {out_dir}")
    if M > 1:
        title_extra = f"ensemble of {M} runs (best iterations)"
    else:
        run_name = os.path.basename(os.path.dirname(os.path.abspath(runs[0]["path"])))
        title_extra = f"{run_name}, iteration {runs[0]['iter']}"
    if WRITE_STATS:
        _write(os.path.join(out_dir, f"{prefix}_runs.txt"), f"# mode={mode}", runs_txt)

    panels = build_panels(ds, datatypes=DATATYPES, components=COMPONENTS)
    if not panels:
        print("femtic_data_misfit: nothing left after DATATYPES/COMPONENTS selection.")
        return 1

    if FIT_MEASURES:
        table = format_fit_measures(panels, has_syn=has_syn)
        print(table)
        if WRITE_STATS:
            _write(os.path.join(out_dir, f"{prefix}_fit_measures.txt"),
                   f"# {title_extra}\n{FIT_MEASURE_NOTE}", table)

    if "crossplot" in METHODS:
        crange = global_logf_range(panels) if COLOR_RANGE == "global" else None
        for dname, lst in panels.items():
            fig = plot_crossplot_datatype(
                dname, lst, title_extra=title_extra, has_syn=has_syn, ncols=NCOLS,
                panel_size=PANEL_SIZE, cmap=CMAP, color_range=crange,
                marker_size=MARKER_SIZE, alpha=ALPHA, log_rho=LOG_RHO,
                show_errorbars=SHOW_ERRORBARS, equal_aspect=EQUAL_ASPECT)
            _save(fig, os.path.join(out_dir, f"{prefix}_crossplot_{_safe_name(dname)}"),
                  PLOT_FORMATS, PLOT_DPI)
            if SHOW:
                plt.show()
            plt.close(fig)
        table = format_crossplot_stats(panels, has_syn=has_syn)
        print(table)
        if WRITE_STATS:
            _write(os.path.join(out_dir, f"{prefix}_crossplot_stats.txt"),
                   f"# {title_extra}\n{FIT_MEASURE_NOTE}\n"
                   "# columns with suffix 1 / 2: conventional / obs+syn normalization "
                   "(ensemble only); rmae = r_mae, rmed = r_med", table)

    if "histogram" in METHODS:
        if not has_syn:
            print("femtic_data_misfit: single run: no synthesized error available, "
                  "histograms show the conventional (RMS1) normalization only.")
        all_stats: Dict[str, List[dict]] = {}
        for gname, lst in group_panels(panels, by_component=HIST_BY_COMPONENT).items():
            parts = [normalized_residuals(p) for p in lst]
            freq = np.concatenate([q[0] for q in parts])
            r1 = np.concatenate([q[1] for q in parts])
            r2 = np.concatenate([q[2] for q in parts])
            if r1.size == 0:
                print(f"femtic_data_misfit: {gname}: no rows with positive error, skipped.")
                continue
            fig, stats = plot_residual_histograms(
                gname, freq, r1, r2, has_syn=has_syn,
                per_decade=HIST_BANDS_PER_DECADE, axis=HIST_BAND_AXIS,
                ncols=HIST_NCOLS, panel_size=HIST_PANEL_SIZE, limit=HIST_LIMIT,
                bin_width=HIST_BIN_WIDTH, min_n=HIST_MIN_N,
                run_label=title_extra)
            all_stats[gname] = stats
            _save(fig, os.path.join(out_dir, f"{prefix}_hist_{_safe_name(gname)}"),
                  PLOT_FORMATS, PLOT_DPI)
            if SHOW:
                plt.show()
            plt.close(fig)
        table = format_hist_stats(all_stats, has_syn=has_syn)
        print(table)
        if WRITE_STATS:
            _write(os.path.join(out_dir, f"{prefix}_hist_stats.txt"),
                   f"# {title_extra}; bands: {HIST_BANDS_PER_DECADE} per decade "
                   f"({HIST_BAND_AXIS})\n{FIT_MEASURE_NOTE}\n"
                   "# n_out: values of the plotted variant outside +/- HIST_LIMIT", table)

    if "qq" in METHODS:
        qq_stats: Dict[str, List[dict]] = {}
        for gname, lst in group_panels(panels, by_component=QQ_BY_COMPONENT).items():
            parts = [normalized_residuals(p) for p in lst]
            freq = np.concatenate([q[0] for q in parts])
            r1 = np.concatenate([q[1] for q in parts])
            r2 = np.concatenate([q[2] for q in parts])
            if r1.size < 4:
                print(f"femtic_data_misfit: {gname}: too few rows for Q-Q plots, skipped.")
                continue
            fig, stats = plot_qq_bands(
                gname, freq, r1, r2, has_syn=has_syn,
                per_decade=QQ_BANDS_PER_DECADE, axis=QQ_BAND_AXIS,
                ncols=QQ_NCOLS, panel_size=QQ_PANEL_SIZE, limit=QQ_LIMIT,
                confidence=QQ_CONFIDENCE, min_n=QQ_MIN_N,
                equal_axes=QQ_EQUAL_AXES, run_label=title_extra)
            qq_stats[gname] = stats
            _save(fig, os.path.join(out_dir, f"{prefix}_qq_{_safe_name(gname)}"),
                  PLOT_FORMATS, PLOT_DPI)
            if SHOW:
                plt.show()
            plt.close(fig)
        table = format_qq_stats(qq_stats, has_syn=has_syn)
        print(table)
        if WRITE_STATS:
            _write(os.path.join(out_dir, f"{prefix}_qq_stats.txt"),
                   f"# {title_extra}; bands: {QQ_BANDS_PER_DECADE} per decade "
                   f"({QQ_BAND_AXIS})", table)

    if "curves" in METHODS:
        # where to look for site.dat: the results directories and up to two
        # levels above them (ensemble members sit in a common parent)
        sdirs: List[str] = []
        for r in runs:
            d = os.path.dirname(os.path.abspath(r["path"]))
            for _ in range(3):
                if d not in sdirs:
                    sdirs.append(d)
                d = os.path.dirname(d)
        return run_curves(ds, out_dir=out_dir, prefix=prefix, title_extra=title_extra,
                          search_dirs=sdirs)
    return 0


def main(argv: Sequence[str]) -> int:
    import matplotlib
    if not SHOW:
        matplotlib.use("Agg")

    argv = list(argv)
    mode = RUN_MODE
    if argv and argv[0] in ("single", "ensemble"):
        mode = argv.pop(0)
    paths = expand_paths(argv if argv else RUN_PATHS)
    if not paths:
        print("femtic_data_misfit: no run paths given.")
        return 1

    if mode == "ensemble":
        runs = []
        for p in paths:
            try:
                runs.append(resolve_run(p, iteration=ITERATION, pattern=ITER_PATTERN))
            except (FileNotFoundError, ValueError) as exc:
                print(f"femtic_data_misfit: skipping member {p}: {exc}")
        if not runs:
            print("femtic_data_misfit: no usable ensemble member found.")
            return 1
        print(f"femtic_data_misfit: ensemble of {len(runs)} of {len(paths)} runs")
        prefix = OUT_PREFIX or default_prefix(runs, mode)
        out_dir = OUT_DIR or default_out_dir(runs)
        return process_runs(runs, mode=mode, out_dir=out_dir, prefix=prefix)

    # single: every entry on its own (not an ensemble)
    rc = 0
    for k, p in enumerate(paths):
        print(f"femtic_data_misfit: ===== run {k + 1} of {len(paths)}: {p}")
        try:
            run = resolve_run(p, iteration=ITERATION, pattern=ITER_PATTERN)
        except (FileNotFoundError, ValueError) as exc:
            print(f"femtic_data_misfit: skipping {p}: {exc}")
            rc = 1
            continue
        derived = default_prefix([run], "single")
        if not OUT_PREFIX:
            prefix = derived
        elif len(paths) == 1:
            prefix = OUT_PREFIX
        else:
            prefix = f"{OUT_PREFIX}_{derived}"
        out_dir = OUT_DIR or default_out_dir([run])
        try:
            rc = max(rc, process_runs([run], mode="single", out_dir=out_dir, prefix=prefix))
        except (ValueError, KeyError) as exc:
            print(f"femtic_data_misfit: {p} failed: {exc}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
