#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the L-curve (roughness vs. misfit/nRMS) from FEMTIC inversion runs
at different regularisation parameters (alpha).

@author: vrath

Provenance:
    2025       vrath   Created.
    2026-03-03 Claude  Renamed user-set parameters to UPPERCASE.
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
WORK_DIR = r"/home/vrath/MT_Data/Ubinas/ubinas_35/no_corr/"

PLOT_NAME = WORK_DIR + "Ubinas_Ini35"
PLOT_WHAT = "nrms"  # 'nrms' or 'misfit'

SEARCH_STRNG = "*L2"
dir_list = utl.get_filelist(
    searchstr=[SEARCH_STRNG], searchpath=WORK_DIR,
    sortedlist=True, fullpath=True,
)

# =============================================================================
#  Read final convergence values
# =============================================================================
#  Iter#,  Retrial#, Alpha, Damp, Roughness, Misfit, RMS, ObjFunc
#  0,      1,        2,     3,    4,         5,      6,   7
l_curve = []
for directory in dir_list:
    with open(directory + "/femtic.cnv") as cnv:
        content = cnv.readlines()

    line = content[-1].split()
    print(line)
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

# =============================================================================
#  Plot L-curve
# =============================================================================
plot_kwargs = dict(
    color="green", marker="o", linestyle="dashed",
    linewidth=1, markersize=7,
    markeredgecolor="red", markerfacecolor="white",
)

yformula = r"$\mathrm{roughness}=\Vert\mathbf{C}_m^{-1/2} \mathbf{m}\Vert_2$"

fig, ax = plt.subplots()

if "nrms" in PLOT_WHAT.lower():
    xdata = n
    xformula = \
        r"$\mathrm{nRMS}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left[\mathbf{C}_d^{-1/2}\left(d_i^{\mathrm{obs}}-d_i^{\mathrm{pred}}\right)\right]^2}$"
else:
    xdata = m
    xformula = \
        r"$\mathrm{nRMS}=\Vert\mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})\Vert_2$"

plt.plot(xdata, r, **plot_kwargs)

for k in np.arange(len(lc_sorted)):
    alph = round(a[k], -int(np.floor(np.log10(abs(a[k])))))
    plt.annotate(str(alph), [xdata[k], r[k]])

plt.title(PLOT_NAME.replace("_", " "))
plt.xlabel(xformula, fontsize=12)
plt.ylabel(yformula, fontsize=12)
plt.grid("on")
plt.tight_layout()

plt.savefig(PLOT_NAME + ".pdf")
plt.savefig(PLOT_NAME + ".png")
print(f"Saved {PLOT_NAME}.pdf and {PLOT_NAME}.png")
