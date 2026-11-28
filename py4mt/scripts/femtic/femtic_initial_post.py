#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process a starting-model study: find the best nRMS of every run and
plot it against log10 of the fill value (the starting resistivity).

Companion of femtic_initial_prep.py (creates the ini_<rho>/ run directories,
each with a homogeneous starting model).  Built on the pattern of
femtic_lcurve_post.py (run harvesting, plot style, report, npz, summary).
For every run directory <ENSEMBLE_DIR><ENSEMBLE_NAME>* the script takes

    fill value   the resistivity of the free regions of MODEL_FILE
                 (resistivity_block_iter0.dat) in the run directory
                 (FILL_SOURCE = "model"), or the number in the directory name
                 (FILL_SOURCE = "name"); with "model" the number in the
                 directory name is compared with it as a check.  A starting
                 model that is not homogeneous (e.g. a link to the template
                 left from a wrong setup) has no fill value; the run is
                 reported and skipped.
    best nRMS    the smallest nRMS in the convergence file (femtic.cnv) of
                 the run, and the iteration where it occurs: the definition
                 of fem.get_nrms(), which is also used for the best iteration
                 in femtic_plot_nrms.py.  The nRMS of the last line of the
                 file is reported next to it (a run that diverges late shows
                 up as a large difference).

and plots the best nRMS (y) against log10(fill value in Ohm.m) (x), every
point labelled with its resistivity.  The run with the smallest best nRMS
is marked by a star and reported: this is the starting resistivity that
fits the data best.  The script warns if it is the smallest or the largest
value tried, because the minimum may then lie outside the sampled range.

A flat curve means that the inversion result does not depend on the start;
a clear minimum or a trend means that it does, and that the runs are
(partly) stuck in local minima or not converged within ITERATION steps.

Output files (stem PLOT_NAME):
    <PLOT_NAME>.<ext>             best nRMS versus log10(fill value)
                                  (PLOT_FORMATS)
    <PLOT_NAME>_best.txt          best run and table of all runs
    <PLOT_NAME>_best.npz          the same numbers as arrays
    femtic_initial_post_summary.md  parameter summary, written to ENSEMBLE_DIR

@author:   Claude (Anthropic), for vrath
@project:  py4mt -- Python for Magnetotellurics
@inversion: FEMTIC

Provenance:
    2026-11-23  Claude Sonnet 5.5 (Anthropic)
                Created (date generated: 2026-11-23).  Author: Claude,
                Anthropic.  Built from femtic_lcurve_post.py (run
                harvesting by glob, CNV_FILE, plot style, PLOT_TITLE via
                fviz.resolve_title, report / npz / parameter summary) and
                fem.get_nrms (best nRMS).
                AI-generated code; review before production use.
"""

import os
import sys
import glob
from pathlib import Path
import inspect

import numpy as np
import matplotlib.pyplot as plt

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
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
# USER SECTION -- all user-set parameters below are UPPERCASE
# =============================================================================

#: Directory with the run directories; the same as ENSEMBLE_DIR of
#: femtic_initial_prep.py.
ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/sabancaya2026/initial_nodist/"
#: Run directories are <ENSEMBLE_DIR><ENSEMBLE_NAME>*, as made by
#: femtic_initial_prep.py.
ENSEMBLE_NAME = "ini_"
#: Name of the FEMTIC convergence file in each run directory; a single name
#: or a list of candidates (first existing wins).
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

# --- fill value --------------------------------------------------------------
#: The starting model of the run directories (MODEL_FILE of
#: femtic_initial_prep.py).
MODEL_FILE = "resistivity_block_iter0.dat"
#: Where the fill value comes from: "model" = read from MODEL_FILE of the run
#: directory (the free regions; the directory name is only compared), "name"
#: = the number after ENSEMBLE_NAME in the directory name (Ohm.m), which does
#: not need the block files (faster for a large mesh).
FILL_SOURCE = "model"

# --- plot --------------------------------------------------------------------
PLOT_NAME = os.path.join(ENSEMBLE_DIR, "initial_nrms")
#: Figure title: "auto" = "Starting model test"; None, False or "" = no
#: title; a string, used as given; a dict {"default": spec}.
PLOT_TITLE = "auto"
FONTSIZE = 9
#: Log10 axis for the nRMS (the x axis is log10(fill value) already).
PLOT_LOG_Y = False
#: Axis limits [min, max] (x: log10 Ohm.m, y: nRMS), or None for matplotlib
#: auto.
PLOT_XLIM = None
PLOT_YLIM = None
#: Label every point with its fill value in Ohm.m.
PLOT_ANNOTATE = True
#: Figure size [width, height] in cm (femtic_lcurve_post.py: 8.5 x 8.5).
PLOT_SIZE_CM = [8.5, 8.5]
#: Marker of the best run: fill colour, marker style ("*" = star) and size in
#: points; the edge is black.
BEST_COLOR = "yellow"
BEST_MARKER = "*"
BEST_SIZE = 14
PLOT_FORMATS = ["pdf", "svg", "jpg"]
PLOT_DPI = 600

# =============================================================================
# END OF USER SECTION
# =============================================================================

#: Largest spread of log10(rho) among the free regions of a starting model
#: that is still called homogeneous (the block file holds 7 digits).
_HOMOGENEOUS_TOL = 1.0e-5
#: Largest difference of log10(fill value) between the model and the
#: directory name that is accepted without a warning.
_NAME_TOL = 1.0e-3


def _fmt(x):
    return format(x, ".6g")


def _name_value(directory):
    """Fill value (Ohm.m) from the directory name, or None if it has none."""
    base = os.path.basename(os.path.normpath(directory))
    try:
        v = float(base[len(ENSEMBLE_NAME):])
    except ValueError:
        return None
    return v if np.isfinite(v) and v > 0.0 else None


def _model_value(directory):
    """(rho, spread) of the free regions of MODEL_FILE in directory.

    rho in Ohm.m (median), spread = largest |log10(rho) - median| in the
    free regions.  Raises OSError / ValueError if the file cannot be read.
    """
    m = fem.read_model(os.path.join(directory, MODEL_FILE),
                       model_trans="log10", out=False)
    if m.size == 0:
        raise ValueError("no free regions")
    med = float(np.median(m))
    return 10.0 ** med, float(np.max(np.abs(m - med)))


# =============================================================================
# Check the settings
# =============================================================================
ens_dir = os.path.join(ENSEMBLE_DIR, "")
if not os.path.isdir(ens_dir):
    sys.exit(" Directory: %s does not exist." % ens_dir)

FILL_SOURCE = str(FILL_SOURCE).lower()
if FILL_SOURCE not in ("model", "name"):
    sys.exit(" FILL_SOURCE must be 'model' or 'name', got %r" % FILL_SOURCE)

# =============================================================================
# Read the runs
# =============================================================================
# Column positions come from each file's own header row (fem.read_cnv).
runs = []
for directory in sorted(glob.glob(ens_dir + ENSEMBLE_NAME + "*")):
    if not os.path.isdir(directory):
        continue
    base = os.path.basename(directory)

    try:
        rows = fem.read_cnv(directory, filename=CNV_FILE)["rows"]
    except (ValueError, OSError):
        rows = []
    if not rows:
        print(base, "does not contain a valid .cnv file, skipped")
        continue
    nrms_all = np.array([r["RMS"] for r in rows], dtype=float)
    if not np.all(np.isfinite(nrms_all)) or np.any(nrms_all <= 0.0):
        print(base, "has non-finite or non-positive nRMS in its .cnv file, "
              "skipped")
        continue
    # best = smallest nRMS of all lines, the definition of fem.get_nrms()
    iter_best, nrms_best = fem.get_nrms(directory, CNV_FILE)

    v_name = _name_value(directory)
    if FILL_SOURCE == "name":
        if v_name is None:
            print(base, "has no number after %r in its name, skipped"
                  % ENSEMBLE_NAME)
            continue
        fill = v_name
    else:
        try:
            fill, spread = _model_value(directory)
        except (OSError, ValueError) as exc:
            print(base, "starting model %s cannot be read (%s), skipped"
                  % (MODEL_FILE, exc))
            continue
        if spread > _HOMOGENEOUS_TOL:
            print(base, "starting model %s is not homogeneous (log10 rho "
                  "spread %.3g), skipped.\n    Is it a link to the template?"
                  " (FILL_SOURCE = 'name' takes the value from the "
                  "directory name.)" % (MODEL_FILE, spread))
            continue
        if v_name is not None and abs(np.log10(fill) - np.log10(v_name)) > _NAME_TOL:
            print("WARNING: %s: starting model has %s Ohm.m, the directory "
                  "name says %s.  The model value is used."
                  % (base, _fmt(fill), _fmt(v_name)))

    runs.append(dict(dir=directory, fill=float(fill),
                     best=float(nrms_best), iter_best=int(iter_best),
                     last=float(nrms_all[-1]),
                     iter_last=int(round(rows[-1]["Iter"])) if "Iter" in rows[-1] else -1,
                     nrows=len(rows)))

if len(runs) < 2:
    sys.exit(" %d usable run(s) found in %s%s*; at least 2 are needed."
             % (len(runs), ens_dir, ENSEMBLE_NAME))

runs.sort(key=lambda r: r["fill"])
fill = np.array([r["fill"] for r in runs])
logf = np.log10(fill)
best = np.array([r["best"] for r in runs])
it_best = np.array([r["iter_best"] for r in runs])
last = np.array([r["last"] for r in runs])
it_last = np.array([r["iter_last"] for r in runs])
nrows = np.array([r["nrows"] for r in runs])
dirs = [r["dir"] for r in runs]
n = len(fill)

_same = ["%s = %s" % (os.path.basename(dirs[k - 1]), os.path.basename(dirs[k]))
         for k in range(1, n) if abs(logf[k] - logf[k - 1]) < 1.0e-9]
if _same:
    sys.exit(" Runs with the same fill value: %s; remove or rename one of "
             "each pair." % ", ".join(_same))

# =============================================================================
# Best run
# =============================================================================
i_best = int(np.argmin(best))

if n >= 3 and i_best in (0, n - 1):
    print("WARNING: the smallest nRMS is at the %s fill value tried (%s Ohm.m); "
          "the minimum may lie outside the sampled range.\n"
          "         Add runs beyond %s (extend RHO_LIST in "
          "femtic_initial_prep.py).\n"
          % ("smallest" if i_best == 0 else "largest", _fmt(fill[i_best]),
             "the smallest value" if i_best == 0 else "the largest value"))

# =============================================================================
# Report
# =============================================================================
rep = []
rep.append("@=@")
rep.append("Starting-model test: best nRMS versus fill value")
rep.append("@=@")
rep.append("  runs                : %d in %s%s*" % (n, ens_dir, ENSEMBLE_NAME))
rep.append("  fill value from     : %s" % ("starting model " + MODEL_FILE
                                          if FILL_SOURCE == "model"
                                          else "directory name"))
rep.append("  best run            : %s" % os.path.basename(dirs[i_best]))
rep.append("  fill value          : %s Ohm.m   (log10 = %s)"
           % (_fmt(fill[i_best]), _fmt(logf[i_best])))
rep.append("  best nRMS           : %s   (iteration %d)"
           % (_fmt(best[i_best]), it_best[i_best]))
rep.append("  nRMS range, all runs: %s ... %s   (largest / smallest = %s)"
           % (_fmt(best.min()), _fmt(best.max()), _fmt(best.max() / best.min())))
rep.append("@=@")
rep.append("%4s  %-12s  %-9s  %-11s  %-9s  %-11s  %-9s  %5s  %s"
           % ("run", "fill [Ohm.m]", "log10", "best nRMS", "best iter",
              "last nRMS", "last iter", "lines", "directory"))
rep.append("@-@")
for k in range(n):
    rep.append("%4d  %-12s  %9.4f  %-11s  %9d  %-11s  %9d  %5d  %s%s"
               % (k, _fmt(fill[k]), logf[k], _fmt(best[k]), it_best[k],
                  _fmt(last[k]), it_last[k], nrows[k],
                  os.path.basename(dirs[k]),
                  "   <- best" if k == i_best else ""))
rep.append("@=@")
_w = max(len(line) for line in rep)
report = "\n".join(("=" * _w if line == "@=@" else "-" * _w if line == "@-@"
                    else line) for line in rep)
print(report)

with open(PLOT_NAME + "_best.txt", "w") as _f:
    _f.write(report + "\n")

np.savez_compressed(PLOT_NAME + "_best.npz",
                    fill_rho=fill, fill_log10=logf, nrms_best=best,
                    iter_best=it_best, nrms_last=last, iter_last=it_last,
                    n_lines=nrows, dirs=np.array(dirs),
                    best_index=i_best, best_fill_rho=fill[i_best],
                    best_nrms=best[i_best],
                    fill_source=np.array(FILL_SOURCE))

# =============================================================================
# Plot: best nRMS versus log10(fill value)
# =============================================================================
cm = 1 / 2.54  # cm -> inch
fig, ax = plt.subplots(figsize=(PLOT_SIZE_CM[0] * cm, PLOT_SIZE_CM[1] * cm))


ax.plot(logf, best, color="green", marker="o", linestyle="dashed",
        linewidth=1, markersize=7, markeredgecolor="red",
        markerfacecolor="white", zorder=4, label="runs")

if PLOT_ANNOTATE:
    for k in range(n):
        ax.annotate(format(fill[k], ".2g"), xy=(logf[k], best[k]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=FONTSIZE - 1, arrowprops=dict(
            arrowstyle="-",
            linewidth=0.5,
            color="0.5",
            shrinkA=0.1,
            shrinkB=0.1, relpos=(0.0, 0.0),))
        
    ax.margins(x=0.10, y=0.12)      # room for the labels at the edges

ax.plot([logf[i_best]], [best[i_best]], linestyle="none",
        marker=BEST_MARKER, markersize=BEST_SIZE,
        markerfacecolor=BEST_COLOR, markeredgecolor="black", zorder=5,
        label=r"best: %s $\Omega$m, nRMS %s"
              % (format(fill[i_best], ".2g"), format(best[i_best], ".2g")))

if PLOT_LOG_Y:
    ax.set_yscale("log")
if PLOT_XLIM is not None:
    ax.set_xlim(PLOT_XLIM)
if PLOT_YLIM is not None:
    ax.set_ylim(PLOT_YLIM)

_title = fviz.resolve_title(PLOT_TITLE, "default", "Starting model test",
                            name="PLOT_TITLE")
if _title:
    ax.set_title(_title, fontsize=FONTSIZE + 1)
ax.tick_params(labelsize=FONTSIZE - 1)
ax.set_xlabel(r"$\log_{10}$ (fill value / $\Omega$m)", fontsize=FONTSIZE)
ax.set_ylabel("best nRMS", fontsize=FONTSIZE)
ax.grid(True)
ax.legend(fontsize=FONTSIZE - 1, frameon=False, loc="best")
fig.tight_layout()
for _ext in PLOT_FORMATS:
    fig.savefig("%s.%s" % (PLOT_NAME, _ext), dpi=PLOT_DPI)
print("Saved %s.{%s}" % (PLOT_NAME, ",".join(PLOT_FORMATS)))

# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR)
