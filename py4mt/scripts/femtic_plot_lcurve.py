#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the L-curve (roughness vs. misfit/nRMS) from FEMTIC inversion runs
at different regularisation parameters (alpha).

@author: vrath
"""

import os
import sys
import inspect

import numpy as np
import matplotlib.pyplot as plt

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

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
WorkDir = r"/home/vrath/FEMTIC_work/krafla6big_L2_L_curve/"
PlotName = WorkDir + "Krafla_L2_L-Curve"
PlotWhat = "nrms"  # 'nrms' or 'misfit'

SearchStrng = "kra*"
dir_list = utl.get_filelist(
    searchstr=[SearchStrng], searchpath=WorkDir,
    sortedlist=True, fullpath=True,
)

# =============================================================================
#  Read final convergence values
# =============================================================================
l_curve = []
for directory in dir_list:
    with open(directory + "/femtic.cnv") as cnv:
        content = cnv.readlines()

    line = content[-1].split()
    print(line)
    alpha = float(line[2])
    rough = float(line[5])
    misft = float(line[7])
    nrmse = float(line[8])
    l_curve.append([alpha, rough, misft, nrmse])

lc = np.array(l_curve).reshape((-1, 4))
ind = np.argsort(lc[:, 0])
lc_sorted = lc[ind]

a = lc_sorted[:, 0]
r = lc_sorted[:, 1]
m = lc_sorted[:, 2]
n = lc_sorted[:, 3]

print("alpha", a)
print("rough", r)
print("misfit", m)
print("nrmse", n)

# =============================================================================
#  Plot L-curve
# =============================================================================
plot_kwargs = dict(
    color="green", marker="o", linestyle="dashed",
    linewidth=1, markersize=7,
    markeredgecolor="red", markerfacecolor="white",
)

xformula = r"$\Vert\mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})\Vert_2$"
yformula = r"$\Vert\mathbf{C}_m^{-1/2} \mathbf{m}\Vert_2$"

fig, ax = plt.subplots()

if "nrms" in PlotWhat.lower():
    xdata = n
else:
    xdata = m

plt.plot(xdata, r, **plot_kwargs)

for k in np.arange(len(lc_sorted)):
    alph = round(a[k], -int(np.floor(np.log10(abs(a[k])))))
    plt.annotate(str(alph), [xdata[k], r[k]])

plt.title(PlotName.replace("_", " "))
plt.xlabel(r"misfit " + xformula, fontsize=14)
plt.ylabel(r"roughness " + yformula, fontsize=14)
plt.grid("on")
plt.tight_layout()

plt.savefig(PlotName + ".pdf")
plt.savefig(PlotName + ".png")
print(f"Saved {PlotName}.pdf and {PlotName}.png")
