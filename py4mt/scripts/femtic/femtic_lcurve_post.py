#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process an L-curve study: plot the L-curve and estimate the trade-off
parameter at its corner (point of maximum curvature).

Companion of femtic_lcurve_prep.py (creates the lc_<alpha>/ run directories)
and extension of femtic_lcurve_plot.py (harvest + plot).  From the last line
of the convergence file (femtic.cnv) of every run directory the script takes
alpha, roughness, misfit and nRMS, plots the L-curve in the style of
femtic_lcurve_plot.py (x = model roughness, y = nRMS or misfit, every point
labelled with its alpha), and marks the corner.

Corner estimate (CORNER_METHOD):

    "menger"  (default) The method of lc.py, taken from inverse.py of
              Py4MTX (calc_lc_corner, curvature, circumradius): the corner
              is the run whose point has the largest Menger curvature 1/R,
              R = radius of the circle through the point and its two
              neighbours, in the plane of (ln data norm, ln model norm).
              Only runs can be chosen, and not the first and the last one
              (no neighbours).  Two things are handled here:
              calc_lc_corner returns an index into the n-2 interior values,
              i.e. one less than the index of the run (corrected, +1); and
              if any three consecutive points are collinear or coincident,
              inverse.curvature gives up for the whole curve (it prints a
              message and returns 0), which the script reports as "no
              menger estimate" (CORNER_METHOD "spline" still works).

    "spline"  Option.  Cubic splines u(t), v(t) of ln data norm and ln model
              norm versus t = log10(alpha) and the analytic signed curvature
                  kappa(t) = (u' v'' - u'' v') / (u'^2 + v'^2)^(3/2),
              taken positive for an L-shaped bend.  Differences to
              "menger": alpha is found between the runs (not restricted to
              the sampled values), the result is less tied to how the
              alphas happen to be spaced, only L-shaped bends count (an
              opposite bend cannot win), and SPLINE_SMOOTH allows smoothing.
              In a test on a synthetic Tikhonov problem (known corner, 0.5
              decade spacing of alpha) the spline was clearly closer to the
              true corner for smooth data; with noisy runs the advantage
              was small or gone, and "gcv" smoothing helped a little.  The
              curve through a sharp corner can oscillate, so look at the
              curvature plot.  Needs 4 runs (5 with smoothing).  The search
              is restricted to the range between the 2nd and the
              last-but-one run, as for "menger": a corner has to be
              bracketed by runs.
    Both estimates are always calculated and printed; CORNER_METHOD selects
    the one that is marked in the plot and reported first.

Norms: the corner is searched in the plane of (data norm, model norm), the
classical L-curve of Hansen, independent of what is plotted (PLOT_WHAT):
data norm = nRMS (proportional to the residual norm), model norm =
sqrt(Roughness) if ROUGH_IS_SQUARED (the cnv column holds the squared
roughness term Phi_m) else Roughness.  Curvature in log-log space changes if
only one axis is squared, so ROUGH_IS_SQUARED matters; check it for your
FEMTIC version (compare the two settings).  The L-curve criterion is a
heuristic (it does not give a convergent parameter choice rule in general;
Engl and Grever 1994): look at the plot and at the curvature plot.

lambda: FEMTIC-DABIC v2.7.1 minimises Phi = Phi_d + alpha^2 Phi_m + ..., so
the weight of the roughness term is lambda = alpha^LAMBDA_POWER with
LAMBDA_POWER = 2; use 1 for older versions (Phi_d + alpha Phi_m), None to
skip.  The value to put into TRADE_OFF_PARAMETER of control.dat is alpha.

Output files (stem PLOT_NAME):
    <PLOT_NAME>.<ext>             L-curve with corner (PLOT_FORMATS)
    <PLOT_NAME>_curvature.<ext>   curvature versus alpha (PLOT_CURVATURE)
    <PLOT_NAME>_corner.txt        corner estimates and table of all runs
    <PLOT_NAME>_corner.npz        the same numbers as arrays
    femtic_lcurve_post_summary.md parameter summary, written to ENSEMBLE_DIR

References:
    Hansen, P. C. (1992), Analysis of discrete ill-posed problems by means of
        the L-curve, SIAM Review 34(4), 561-580, doi:10.1137/1034115.
    Hansen, P. C. and O'Leary, D. P. (1993), The use of the L-curve in the
        regularization of discrete ill-posed problems, SIAM J. Sci. Comput.
        14(6), 1487-1503, doi:10.1137/0914086.
    Engl, H. and Grever, W. (1994), Using the L-curve for determining
        optimal regularization parameters, Numer. Math. 69, 25-31,
        doi:10.1007/s002110050078.
    and the references of inverse.calc_lc_corner (Hansen 2010, 2001, 1998;
    circumradius code by Hunter Ratliff, 2019; lc.py by VR, June 2022).

@author:   Claude (Anthropic), for vrath
@project:  py4mt -- Python for Magnetotellurics
@inversion: FEMTIC

Provenance:
    2026-11-17  Claude Sonnet 5.5 (Anthropic)
                Created (date generated: 2026-11-17).  Author: Claude,
                Anthropic.  Built from femtic_lcurve_plot.py (run
                harvesting, plot style, CNV_FILE, PLOT_TITLE via
                fviz.resolve_title) and lc.py (Menger curvature corner);
                spline corner added as option.  Not carried over from
                femtic_lcurve_plot.py: SCALE_ROUGH / SCALE_MISFIT (the plot
                is log-log by default) and the LC_dat.npz file (replaced by
                <PLOT_NAME>_corner.npz).
                AI-generated code; review before production use.
    2026-11-17  Claude Sonnet 5.5 (Anthropic)
                Revised on request: the Menger corner now comes from
                inverse.py (calc_lc_corner, curvature) and the smoothing
                splines from inverse.make_spline, instead of local copies;
                the local vectorised Menger code was removed.  Differences
                in behaviour: the +1 index offset of calc_lc_corner is
                applied here, and collinear or coincident points give "no
                menger estimate" (no per-point treatment any more).
                AI-generated code; review before production use.
    2026-11-17  Claude Sonnet 5.5 (Anthropic)
                Corner marker in the L-curve plot is now a yellow star with
                black edge by default (was red); CORNER_COLOR,
                CORNER_MARKER and CORNER_SIZE added to the USER SECTION.
                AI-generated code; review before production use.
    2026-11-18  Claude Sonnet 5.5 (Anthropic)
                Scale factors for linear axes (SCALE_X, SCALE_Y; "auto" or
                a number) with the factor shown as "/ 10^n" in the axis
                titles; PLOT_XLIM / PLOT_YLIM are now in data units.
                AI-generated code; review before production use.
"""

import os
import sys
import glob
from pathlib import Path
import inspect

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import femtic_viz as fviz
import inverse as inv
import util as utl
from version import versionstrg

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
# USER SECTION -- all user-set parameters below are UPPERCASE
# =============================================================================

#: Directory with the run directories; the same as ENSEMBLE_DIR of
#: femtic_lcurve_prep.py (WORK_DIR of femtic_lcurve_plot.py).
ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/sabancaya2026/l-curve/"
#: Run directories are <ENSEMBLE_DIR><ENSEMBLE_NAME>*, as made by
#: femtic_lcurve_prep.py.
ENSEMBLE_NAME = "lc_"
#: Name of the FEMTIC convergence file in each run directory; a single name
#: or a list of candidates (first existing wins).
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

# --- corner estimate ---------------------------------------------------------
#: "menger" = inverse.calc_lc_corner, i.e. lc.py (default); "spline" =
#: cubic-spline curvature (option, finds alpha between the runs).  Both are
#: printed either way.
CORNER_METHOD = "spline" #"menger"
#: True: the cnv column "Roughness" holds the squared roughness term
#: (Phi_m = |R m|^2, as a sum-of-squares objective function suggests), the
#: model norm used for the corner is sqrt(Roughness).  False: the column is
#: already a norm.  The corner can move with this setting: check it.
ROUGH_IS_SQUARED = True
#: Smoothing of the splines ("spline" method only): 0 = interpolating cubic
#: spline; "gcv" = smoothing spline with the penalty chosen by generalised
#: cross validation (an option for noisy runs); a positive number = penalty
#: lam of inverse.make_spline / scipy's make_smoothing_spline (larger =
#: smoother; depends on the
#: scale of the data, a fixed value can oversmooth).  Smoothing needs at
#: least 5 runs. 
SPLINE_SMOOTH = "gcv"  #0.0
#: lambda = alpha**LAMBDA_POWER is reported next to alpha: 2 for
#: FEMTIC-DABIC v2.7.1 (Phi = Phi_d + alpha^2 Phi_m), 1 for older versions,
#: None = do not report.
LAMBDA_POWER = 2

# --- plot --------------------------------------------------------------------
PLOT_NAME = os.path.join(ENSEMBLE_DIR, "L-curve_gcv")
#: Quantity on the y axis: "nrms" or "misfit" (as femtic_lcurve_plot.py).
#: The corner does not depend on this setting.
PLOT_WHAT = "nrms"
#: Figure title: "auto" = "L-curve"; None, False or "" = no title; a string,
#: used as given ({what} = PLOT_WHAT); a dict {"default": spec}.
PLOT_TITLE = "auto"
FONTSIZE = 9
#: Log10 axes.  True is the natural choice here: the corner is searched in
#: log-log space.
PLOT_LOG_X = False #True
PLOT_LOG_Y = False
#: Axis limits [min, max] in data units (before SCALE_X / SCALE_Y are
#: applied), or None for matplotlib auto.
PLOT_XLIM = None
PLOT_YLIM = None
#: Scale factors of LINEAR axes (ignored for a log axis).  The plotted values
#: are data / SCALE and the axis title gets the suffix "/ 10^n".  SCALE_X is
#: for the roughness (x), SCALE_Y for the plotted misfit measure (y).
#: "auto": power of ten, multiple of 3, chosen from the data if its largest
#: value is >= 1e4 or < 1e-2; None or 1: no scaling; a number > 0: fixed
#: factor (like SCALE_ROUGH / SCALE_MISFIT in femtic_lcurve_plot.py).
SCALE_X = "auto"
SCALE_Y = "auto"
#: Figure size [width, height] in cm (femtic_lcurve_plot.py: 8.5 x 8.5).
PLOT_SIZE_CM = [8.5, 8.5]
#: Also draw curvature versus alpha (second figure, <PLOT_NAME>_curvature).
PLOT_CURVATURE = True
#: Marker of the corner in the L-curve: fill colour (any matplotlib colour),
#: marker style ("*" = star) and size in points; the edge is black.
CORNER_COLOR = "yellow"
CORNER_MARKER = "*"
CORNER_SIZE = 14
PLOT_FORMATS = ["pdf", "svg", "jpg"]
PLOT_DPI = 600

# =============================================================================
# END OF USER SECTION
# =============================================================================

#: Number of points of the fine alpha grid for the spline curvature.
_N_FINE = 2000


def _turn(u, v, i):
    """Sign of the turn of the curve (u, v) at point i (neighbours i-1, i+1).

    Positive for a left turn, which is the L-shaped bend when u increases and
    v decreases along the curve; the same sign convention as
    spline_curvature().  inverse.curvature() returns unsigned values only.
    """
    return ((u[i] - u[i - 1]) * (v[i + 1] - v[i])
            - (v[i] - v[i - 1]) * (u[i + 1] - u[i]))


def _make_spline(t, y, smooth):
    """Cubic spline of y(t): interpolating (smooth 0/None) or smoothing
    (inverse.make_spline: "gcv" = penalty by GCV, a number = lam)."""
    if isinstance(smooth, str):
        return inv.make_spline(t, y, lam=None)      # lam = None: GCV
    if smooth is None or smooth == 0:
        return CubicSpline(t, y)
    return inv.make_spline(t, y, lam=float(smooth))


def _check_scale(name, val):
    """Validate SCALE_X / SCALE_Y: None, "auto" or a finite number > 0."""
    if val is None:
        return None
    if isinstance(val, str):
        if val.lower() != "auto":
            sys.exit(" %s must be None, 'auto' or a number > 0, got %r"
                     % (name, val))
        return "auto"
    if isinstance(val, bool) or not np.isfinite(val) or val <= 0.0:
        sys.exit(" %s must be None, 'auto' or a number > 0, got %r"
                 % (name, val))
    return float(val)


def _axis_scale(name, spec, values, is_log):
    """Scale factor of an axis (1.0 = none); plotted values = data / factor.
    A log axis is never scaled."""
    if spec is None or spec == 1.0:
        return 1.0
    if is_log:
        if spec != "auto":
            print("Note: %s is ignored, the axis is logarithmic." % name)
        return 1.0
    if spec == "auto":
        m = float(np.max(np.abs(values)))
        if not np.isfinite(m) or m <= 0.0 or 1.0e-2 <= m < 1.0e4:
            return 1.0
        return 10.0 ** (3.0 * np.floor(np.floor(np.log10(m)) / 3.0))
    return float(spec)


def _scale_suffix(scale):
    """Axis-title suffix for a scale factor: " / 10^n" (empty if 1)."""
    if scale == 1.0:
        return ""
    e = np.log10(scale)
    if np.isclose(e, np.round(e)):
        return r" $/\,10^{%d}$" % int(np.round(e))
    return r" $/\,%g$" % scale


def spline_curvature(su, sv, tt):
    """Signed curvature of the curve (u(t), v(t)) at t = tt.

    Positive for an L-shaped bend when u (data norm) increases and v (model
    norm) decreases with t = log10(alpha).  NaN where the speed is zero.
    """
    du, dv = su(tt, 1), sv(tt, 1)
    ddu, ddv = su(tt, 2), sv(tt, 2)
    den = (du ** 2 + dv ** 2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0.0, (du * ddv - ddu * dv) / den, np.nan)


def _fmt(x):
    return format(x, ".6g")


# =============================================================================
# Check the settings
# =============================================================================
ens_dir = os.path.join(ENSEMBLE_DIR, "")
if not os.path.isdir(ens_dir):
    sys.exit(" Directory: %s does not exist." % ens_dir)

CORNER_METHOD = str(CORNER_METHOD).lower()
if CORNER_METHOD not in ("menger", "spline"):
    sys.exit(" CORNER_METHOD must be 'menger' or 'spline', got %r"
             % CORNER_METHOD)

if isinstance(SPLINE_SMOOTH, str):
    if SPLINE_SMOOTH.lower() != "gcv":
        sys.exit(" SPLINE_SMOOTH must be a number >= 0 or 'gcv', got %r"
                 % SPLINE_SMOOTH)
    SPLINE_SMOOTH = "gcv"
elif SPLINE_SMOOTH is not None and SPLINE_SMOOTH < 0:
    sys.exit(" SPLINE_SMOOTH must be >= 0, got %r" % SPLINE_SMOOTH)
smooth_on = isinstance(SPLINE_SMOOTH, str) or bool(SPLINE_SMOOTH)

SCALE_X = _check_scale("SCALE_X", SCALE_X)
SCALE_Y = _check_scale("SCALE_Y", SCALE_Y)

plot_nrms = "nrms" in str(PLOT_WHAT).lower()

# =============================================================================
# Read the final convergence values of every run
# =============================================================================
# Column positions come from each file's own header row (fem.read_cnv), as in
# femtic_lcurve_plot.py.  The last line of the file is used.
runs = []
for directory in sorted(glob.glob(ens_dir + ENSEMBLE_NAME + "*")):
    if not os.path.isdir(directory):
        continue
    try:
        rows = fem.read_cnv(directory, filename=CNV_FILE)["rows"]
    except (ValueError, OSError):
        rows = []
    if not rows:
        print(directory, "does not contain a valid .cnv file, skipped")
        continue
    last = rows[-1]
    run = dict(dir=directory,
               alpha=float(last["Alpha"]),
               rough=float(last["Roughness"]),
               misfit=float(last["Misfit"]),
               nrms=float(last["RMS"]),
               objfc=float(last["ObjFunc"]),
               iter=int(last["Iter"]) if "Iter" in last else -1)
    if min(run["alpha"], run["rough"], run["misfit"], run["nrms"]) <= 0.0:
        print(directory, "has non-positive alpha / roughness / misfit / "
              "nRMS in its last line, skipped")
        continue
    runs.append(run)

if len(runs) < 3:
    sys.exit(" %d usable run(s) found in %s%s*; at least 3 are needed."
             % (len(runs), ens_dir, ENSEMBLE_NAME))

runs.sort(key=lambda r: r["alpha"])
alpha = np.array([r["alpha"] for r in runs])
rough = np.array([r["rough"] for r in runs])
misfit = np.array([r["misfit"] for r in runs])
nrms = np.array([r["nrms"] for r in runs])
objfc = np.array([r["objfc"] for r in runs])
iters = np.array([r["iter"] for r in runs])
dirs = [r["dir"] for r in runs]
n = len(alpha)

_same = ["%s = %s" % (os.path.basename(dirs[k - 1]), os.path.basename(dirs[k]))
         for k in range(1, n) if alpha[k] == alpha[k - 1]]
if _same:
    sys.exit(" Runs with the same alpha: %s; remove or rename one of each "
             "pair." % ", ".join(_same))

ydata = nrms if plot_nrms else misfit

# =============================================================================
# Corner of the L-curve
# =============================================================================
# Plane of the classical L-curve: u = ln(data norm), v = ln(model norm), runs
# ordered by alpha (t = log10 alpha).  Natural logarithms, as in
# inverse.calc_lc_corner.
t = np.log10(alpha)
dnorm = nrms
mnorm = np.sqrt(rough) if ROUGH_IS_SQUARED else rough
u = np.log(dnorm)
v = np.log(mnorm)

if not (u[-1] > u[0] and v[-1] < v[0]):
    print("WARNING: data norm does not increase / model norm does not "
          "decrease from the smallest to the largest alpha.\n"
          "         The curvature sign convention assumes an L-shaped curve "
          "in that direction; check the runs (convergence?).\n")

# --- method "menger" (inverse.calc_lc_corner) --------------------------------
kap_m = np.full(n, np.nan)
alpha_m = None
i_m = None
menger_note = "collinear or coincident points among the runs"
_kap = inv.curvature(u, v)            # n-2 values; the number 0 if it failed
if np.ndim(_kap) == 0:
    print("WARNING (menger): %s; no menger estimate.\n" % menger_note)
else:
    kap_m[1:-1] = _kap
    # calc_lc_corner returns an index into the n-2 interior points
    i_m = int(inv.calc_lc_corner(dnorm, mnorm)) + 1
    alpha_m = alpha[i_m]
    if _turn(u, v, i_m) <= 0.0:
        print("WARNING (menger): the curve bends the other way at the point "
              "of maximum curvature (not an L-shaped bend);\n"
              "         check the plot and the runs.\n")
    if n >= 4 and i_m in (1, n - 2):
        print("WARNING (menger): maximum curvature at the 2nd / last-but-one "
              "run; the corner may lie outside the\n"
              "         sampled range of alpha.  Add runs beyond %s.\n"
              % ("the smallest alpha" if i_m == 1 else "the largest alpha"))
if alpha_m is None and CORNER_METHOD == "menger":
    sys.exit(" CORNER_METHOD 'menger' gives no estimate: %s; try 'spline'."
             % menger_note)

# --- method "spline" ---------------------------------------------------------
n_min = 5 if smooth_on else 4
have_spline = n >= n_min
alpha_s = None
spline_note = "needs at least %d runs (found %d)" % (n_min, n)
if have_spline:
    su = _make_spline(t, u, SPLINE_SMOOTH)
    sv = _make_spline(t, v, SPLINE_SMOOTH)
    sx = _make_spline(t, np.log(rough), SPLINE_SMOOTH)    # plotted x
    sy = _make_spline(t, np.log(ydata), SPLINE_SMOOTH)    # plotted y
    tf = np.linspace(t[0], t[-1], _N_FINE)
    kap_s = spline_curvature(su, sv, tf)
    inside = np.flatnonzero((tf >= t[1]) & (tf <= t[-2])
                            & np.isfinite(kap_s))
    j = inside[int(np.argmax(kap_s[inside]))] if inside.size else None
    if j is None or kap_s[j] <= 0.0:
        spline_note = "no L-shaped bend (curvature <= 0 in the searched range)"
        print("WARNING (spline): %s; no spline estimate.\n" % spline_note)
    else:
        t_s = tf[j]
        alpha_s = 10.0 ** t_s
        if j in (inside[0], inside[-1]):
            print("WARNING (spline): maximum curvature at the edge of the "
                  "searched range (2nd / last-but-one run);\n"
                  "         the corner may lie outside the sampled range of "
                  "alpha.  Add runs there.\n")
if alpha_s is None and CORNER_METHOD == "spline":
    sys.exit(" CORNER_METHOD 'spline' gives no estimate: %s." % spline_note)

# --- selected corner ---------------------------------------------------------
if CORNER_METHOD == "menger":
    alpha_c = alpha_m
    x_c, y_c = rough[i_m], ydata[i_m]
    nrms_c, rough_c = nrms[i_m], rough[i_m]
else:
    alpha_c = alpha_s
    x_c, y_c = np.exp(sx(t_s)), np.exp(sy(t_s))
    nrms_c, rough_c = np.exp(su(t_s)), np.exp(sx(t_s))
i_near = int(np.argmin(np.abs(t - np.log10(alpha_c))))
lam_c = None if not LAMBDA_POWER else alpha_c ** LAMBDA_POWER

# =============================================================================
# Report
# =============================================================================
def _kap_s_at_runs():
    out = np.full(n, np.nan)
    if have_spline:
        out[1:-1] = spline_curvature(su, sv, t[1:-1])
    return out


kap_s_runs = _kap_s_at_runs()

rep = []
rep.append("@=@")
rep.append("L-curve corner (maximum curvature)")
rep.append("@=@")
rep.append("  runs                : %d in %s%s*" % (n, ens_dir, ENSEMBLE_NAME))
rep.append("  method (selected)   : %s" % CORNER_METHOD)
rep.append("  alpha               : %s   (TRADE_OFF_PARAMETER)" % _fmt(alpha_c))
if lam_c is not None:
    rep.append("  alpha^%-3s (lambda)  : %s" % (format(LAMBDA_POWER, "g"), _fmt(lam_c)))
rep.append("  nearest run         : %s  (alpha = %s)"
           % (os.path.basename(dirs[i_near]), _fmt(alpha[i_near])))
rep.append("  nRMS at the corner  : %s" % _fmt(nrms_c))
rep.append("  roughness (cnv)     : %s" % _fmt(rough_c))
rep.append("@-@")
if alpha_m is not None:
    rep.append("  menger              : alpha = %s   (run %s)"
               % (_fmt(alpha_m), os.path.basename(dirs[i_m])))
else:
    rep.append("  menger              : no estimate: %s" % menger_note)
if alpha_s is not None:
    rep.append("  spline              : alpha = %s   (SPLINE_SMOOTH = %s)"
               % (_fmt(alpha_s), SPLINE_SMOOTH))
else:
    rep.append("  spline              : no estimate: %s" % spline_note)
rep.append("  norms               : data = nRMS, model = %s"
           % ("sqrt(Roughness)" if ROUGH_IS_SQUARED else "Roughness"))
rep.append("@=@")
rep.append("%4s  %-11s  %-12s  %-11s  %-12s  %-10s  %-10s  %4s  %s"
           % ("run", "alpha", "roughness", "nRMS", "misfit",
              "kap_menger", "kap_spline", "iter", "directory"))
rep.append("@-@")
for k in range(n):
    rep.append("%4d  %-11s  %-12s  %-11s  %-12s  %-10s  %-10s  %4d  %s%s"
               % (k, _fmt(alpha[k]), _fmt(rough[k]), _fmt(nrms[k]),
                  _fmt(misfit[k]),
                  "-" if np.isnan(kap_m[k]) else format(kap_m[k], ".4g"),
                  "-" if np.isnan(kap_s_runs[k]) else format(kap_s_runs[k], ".4g"),
                  iters[k], os.path.basename(dirs[k]),
                  ("   <- corner" if CORNER_METHOD == "menger"
                   else "   <- nearest to corner") if k == i_near else ""))
rep.append("@=@")
_w = max(len(line) for line in rep)
report = "\n".join(("=" * _w if line == "@=@" else "-" * _w if line == "@-@"
                    else line) for line in rep)
print(report)

with open(PLOT_NAME + "_corner.txt", "w") as _f:
    _f.write(report + "\n")

_npz = dict(alpha=alpha, roughness=rough, misfit=misfit, nrms=nrms,
            objfc=objfc, iteration=iters, kappa_menger=kap_m,
            kappa_spline_runs=kap_s_runs,
            corner_alpha=alpha_c, corner_method=np.array(CORNER_METHOD),
            corner_alpha_menger=np.nan if alpha_m is None else alpha_m,
            corner_alpha_spline=np.nan if alpha_s is None else alpha_s,
            rough_is_squared=bool(ROUGH_IS_SQUARED))
if have_spline:
    _npz.update(alpha_fine=10.0 ** tf, kappa_spline=kap_s)
np.savez_compressed(PLOT_NAME + "_corner.npz", **_npz)

# =============================================================================
# Plot: L-curve with corner
# =============================================================================
# Linear axes: plotted values = data / scale; the axis title shows the factor.
scale_x = _axis_scale("SCALE_X", SCALE_X, rough, PLOT_LOG_X)
scale_y = _axis_scale("SCALE_Y", SCALE_Y, ydata, PLOT_LOG_Y)
if scale_x != 1.0 or scale_y != 1.0:
    print("Axis scale: x / %g, y / %g (see the axis titles)\n"
          % (scale_x, scale_y))
rx = rough / scale_x
ry = ydata / scale_y

cm = 1 / 2.54  # cm -> inch
fig, ax = plt.subplots(figsize=(PLOT_SIZE_CM[0] * cm, PLOT_SIZE_CM[1] * cm))

if CORNER_METHOD == "spline":
    ax.plot(np.exp(sx(tf)) / scale_x, np.exp(sy(tf)) / scale_y,
            color="0.6", linewidth=0.8, zorder=1, label="spline")

ax.plot(rx, ry, color="green", marker="o", linestyle="dashed",
        linewidth=1, markersize=7, markeredgecolor="red",
        markerfacecolor="white", zorder=2, label="runs")

ann_offset = (6, 6)
for k in range(n):
    ax.annotate(format(alpha[k], ".3g"), xy=(rx[k], ry[k]),
                xytext=ann_offset, textcoords="offset points",
                fontsize=FONTSIZE - 1)

ax.plot([x_c / scale_x], [y_c / scale_y], linestyle="none",
        marker=CORNER_MARKER,
        markersize=CORNER_SIZE, markerfacecolor=CORNER_COLOR,
        markeredgecolor="black", zorder=5,
        label=r"$\alpha^* = %s$ (%s)" % (format(alpha_c, ".3g"), CORNER_METHOD))

if PLOT_LOG_X:
    ax.set_xscale("log")
if PLOT_LOG_Y:
    ax.set_yscale("log")
if PLOT_XLIM is not None:
    ax.set_xlim([x / scale_x for x in PLOT_XLIM])
if PLOT_YLIM is not None:
    ax.set_ylim([y / scale_y for y in PLOT_YLIM])

_title = fviz.resolve_title(PLOT_TITLE, "default", "L-curve",
                            name="PLOT_TITLE", what=PLOT_WHAT)
if _title:
    ax.set_title(_title, fontsize=FONTSIZE + 1)
ax.tick_params(labelsize=FONTSIZE - 1)
ax.set_xlabel("model roughness" + _scale_suffix(scale_x),
              fontsize=FONTSIZE)
ax.set_ylabel(("nRMS" if plot_nrms else "misfit") + _scale_suffix(scale_y),
              fontsize=FONTSIZE)
ax.grid(True)
ax.legend(fontsize=FONTSIZE - 1, frameon=False, loc="best")
fig.tight_layout()
for _ext in PLOT_FORMATS:
    fig.savefig("%s.%s" % (PLOT_NAME, _ext), dpi=PLOT_DPI)
print("Saved %s.{%s}" % (PLOT_NAME, ",".join(PLOT_FORMATS)))

# =============================================================================
# Plot: curvature versus alpha
# =============================================================================
if PLOT_CURVATURE:
    fig2, ax2 = plt.subplots(figsize=(PLOT_SIZE_CM[0] * cm,
                                      0.75 * PLOT_SIZE_CM[1] * cm))
    if alpha_m is not None:
        ax2.plot(alpha[1:-1], kap_m[1:-1], color="green", marker="o",
                 linestyle="dashed", linewidth=1, markersize=5,
                 markeredgecolor="red", markerfacecolor="white",
                 label="menger (runs)")
    if have_spline:
        ax2.plot(10.0 ** tf, kap_s, color="0.4", linewidth=1.0,
                 label="spline")
    ax2.axvline(alpha_c, color="red", linestyle=":", linewidth=1.0,
                label=r"$\alpha^* = %s$" % format(alpha_c, ".3g"))
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\alpha$ (trade-off parameter)", fontsize=FONTSIZE)
    ax2.set_ylabel("curvature of the L-curve", fontsize=FONTSIZE)
    ax2.tick_params(labelsize=FONTSIZE - 1)
    ax2.grid(True)
    ax2.legend(fontsize=FONTSIZE - 1, frameon=False, loc="best")
    fig2.tight_layout()
    for _ext in PLOT_FORMATS:
        fig2.savefig("%s_curvature.%s" % (PLOT_NAME, _ext), dpi=PLOT_DPI)
    print("Saved %s_curvature.{%s}" % (PLOT_NAME, ",".join(PLOT_FORMATS)))

# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR)
