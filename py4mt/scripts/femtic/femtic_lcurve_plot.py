#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the L-curve (roughness vs. misfit/nRMS) from FEMTIC inversion runs
at different regularisation parameters (alpha).

@author: vrath

Provenance:
    2025       vrath   Created.
    2026-03-03 Claude  Renamed user-set parameters to UPPERCASE.
    2026-06-17 Claude  Added PLOT_LOG_X / PLOT_LOG_Y for independent log10 axes.
    2026-06-18 Claude  Added PLOT_XLIM / PLOT_YLIM; offset annotations from markers.
    2026-06-26 Claude  DISTORTION defaults to None; auto-detected from cnv column
                       count (10 cols → distortion, 8 cols → no distortion).
                       Explicit True/False still overrides.
    2026-07-05 Claude  Added SCALE_ROUGH / SCALE_MISFIT optional scaling factors,
                       shown as "x10^n" next to the roughness / misfit axis label.
                       Not applied when PLOT_WHAT = "nrms". PLOT_XLIM / PLOT_YLIM
                       are now auto-rescaled to match, so limits stay valid.
    2026-07-05 Claude  Flipped exponent sign in scale-factor label: since data
                       is divided by SCALE_*, a scale of 1e3 now shows "x10^-3"
                       (previously showed "x10^3").
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np
import matplotlib.pyplot as plt

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import util as utl
from version import versionstrg

rng = np.random.default_rng()
nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
WORK_DIR = r"/home/vrath/MT_Data/Ubinas/ubinas_10_LC/"
PLOT_NAME = WORK_DIR + "Fig04_L-curve"
PLOT_WHAT = "nrms"  # 'nrms' or 'misfit'
PLOT_TITLE = r"Ubinas | ini = 10 $\Omega \cdot m$ " #"| distcorr"
DISTORTION = None   # None → auto-detect from cnv column count (10 → distortion, 8 → no distortion)

FONTSIZE = 10
#: Use log10 scale on the x-axis (misfit / nRMS).
PLOT_LOG_X = False
#: Use log10 scale on the y-axis (roughness).
PLOT_LOG_Y = False

#: x-axis limits [min, max]; set to None for matplotlib auto-scaling.
PLOT_XLIM = [0., 80000.] # None   # e.g. [0.0, 1e6]
#: y-axis limits [min, max]; set to None for matplotlib auto-scaling.
PLOT_YLIM = [0.5, 6.] # None   # e.g. [0.9, 5.0]

#: Optional scaling factor for roughness (x-axis data divided by this value).
#: Shown as "x10^n" appended to the axis label. Set to None or 1 to disable.
SCALE_ROUGH = 1.e3   # e.g. 1e3
#: Optional scaling factor for misfit (y-axis data, only when PLOT_WHAT != "nrms").
#: Shown as "x10^n" appended to the axis label. Set to None or 1 to disable.
#: Ignored when PLOT_WHAT = "nrms".
SCALE_MISFIT = None   # e.g. 1e4


SEARCH_STRNG = "*L2"
dir_list = utl.get_filelist(
    searchstr=[SEARCH_STRNG], searchpath=WORK_DIR,
    sortedlist=True, fullpath=True,
)

# =============================================================================
#  Read final convergence values
# =============================================================================
#  Iter#,  Retrial#, Alpha, Damp, Rough, Misfit, RMS, ObjFunc
#  Iter#,  Retrial#, Alpha, Beta, Damp,  Rough,  Dist,  Misfit, nRMS,  ObjFunc
#  0,      1,        2,     3,    4,     5,      6,     7,      8      9
l_curve = []
for directory in dir_list:
    with open(directory + "/femtic.cnv") as cnv:
        content = cnv.readlines()

    line = content[-1].split()
    print(line)
    # Auto-detect distortion from column count; override with DISTORTION if set.
    # 8 cols (no distortion): alpha=2, rough=4, misfit=5, nRMS=6, obj=7
    # 10 cols (distortion):   alpha=2, rough=5, dist=6,   misfit=7, nRMS=8, obj=9
    _has_dist = (DISTORTION if DISTORTION is not None else len(line) == 10)
    if _has_dist:
        alpha = float(line[2])
        rough = float(line[5])
        misft = float(line[7])
        nrmse = float(line[8])
        objfc = float(line[9])
    else:
        alpha = float(line[2])
        rough = float(line[4])
        misft = float(line[5])
        nrmse = float(line[6])
        objfc = float(line[7])
        
    l_curve.append([alpha, rough, misft, nrmse, objfc])

lc = np.array(l_curve).reshape((-1, 5))
ind = np.argsort(lc[:, 0])
lc_sorted = lc[ind]

a = lc_sorted[:, 0]
r = lc_sorted[:, 1]
m = lc_sorted[:, 2]
n = lc_sorted[:, 3]
o = lc_sorted[:, 4]

print("alpha", a)
print("rough", r)
print("misfit", m)
print("nrmse", n)
print("objfc", o)

lcurve = dict(alpha=a, rough=r, misfit=m, nrmse=n, objfc=o) 
np.savez_compressed(WORK_DIR + "LC_dat.npz", **lcurve)

# =============================================================================
#  Plot L-curve
# =============================================================================


def _scale_suffix(scale):
    """Return a LaTeX '$\\times 10^{-n}$' suffix for a given scale factor,
    or an empty string if scaling is disabled (None or 1). Data is divided
    by `scale`, so the label shows the equivalent x10^-n multiplier."""
    if scale is None or scale == 1:
        return ""
    exponent = -np.log10(scale)
    if np.isclose(exponent, np.round(exponent)):
        return fr" $\times\,10^{{{int(np.round(exponent))}}}$"
    return fr" $\times\,{1/scale:g}$"


xformula = r"model roughness" + _scale_suffix(SCALE_ROUGH)
#" $\mathrm{model roughness}$ = \quad \Vert\mathbf{C}_m^{-1/2} \mathbf{m}\Vert_2$"

if "nrms" in PLOT_WHAT.lower():
    ydata = n
    yformula = r"nRMS"
        #"\quad\left(\frac{1}{N}\sum_{i=1,N}\left[\mathbf{C}_d^{-1/2}\left(d_i^{\mathrm{o}}-d_i^{\mathrm{c}}\right)\right]^2\right)^{1/2}$"
else:
    ydata = m
    if SCALE_MISFIT is not None and SCALE_MISFIT != 1:
        ydata = ydata / SCALE_MISFIT
    yformula = r"misfit" + _scale_suffix(SCALE_MISFIT)
        #"$\mathrm{misfit} = \quad \Vert\mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})\Vert_2$"

if SCALE_ROUGH is not None and SCALE_ROUGH != 1:
    r = r / SCALE_ROUGH
    if PLOT_XLIM is not None:
        PLOT_XLIM = [x / SCALE_ROUGH for x in PLOT_XLIM]

if ("nrms" not in PLOT_WHAT.lower() and SCALE_MISFIT is not None
        and SCALE_MISFIT != 1 and PLOT_YLIM is not None):
    PLOT_YLIM = [y / SCALE_MISFIT for y in PLOT_YLIM]




plot_kwargs = dict(
    color="green", marker="o", linestyle="dashed",
    linewidth=1, markersize=7,
    markeredgecolor="red", markerfacecolor="white",
)


cm = 1/2.54  # cm -> inch
fig, ax = plt.subplots(figsize=(8.5*cm, 8.5*cm))

# fig, ax = plt.subplots()


plt.plot(r, ydata, **plot_kwargs)

if PLOT_LOG_X:
    ax.set_xscale("log")
if PLOT_LOG_Y:
    ax.set_yscale("log")

if PLOT_XLIM is not None:
    ax.set_xlim(PLOT_XLIM)
if PLOT_YLIM is not None:
    ax.set_ylim(PLOT_YLIM)

ann_offset = (plot_kwargs["markersize"] -1, plot_kwargs["markersize"] -1)
for k in np.arange(len(lc_sorted)):
    alph = np.round(a[k], 1)
    ax.annotate(
        str(alph), xy=(r[k], ydata[k]),
        xytext=ann_offset, textcoords="offset points",
        fontsize=FONTSIZE,
    )

plt.title(PLOT_TITLE,fontsize=FONTSIZE+1 )
plt.xticks(fontsize=FONTSIZE-1)
plt.yticks(fontsize=FONTSIZE-1)
plt.xlabel(xformula, fontsize=FONTSIZE)
plt.ylabel(yformula, fontsize=FONTSIZE)
plt.grid("on")
plt.tight_layout()

plt.savefig(PLOT_NAME + ".pdf")
plt.savefig(PLOT_NAME + ".svg")
plt.savefig(PLOT_NAME + ".jpg", dpi=600., transparent=True)

print(f"Saved {PLOT_NAME}.pdf and {PLOT_NAME}.jpg")
