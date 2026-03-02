#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot convergence curves (misfit, nRMS, or roughness vs. iteration)
from FEMTIC inversion runs.

Reads femtic.cnv files from a set of inversion directories and
generates convergence plots as PDF.

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
PlotName = r"Krafla_L2_Convergence"
PlotWhat = "rough"  # Options: 'misfit', 'rms', 'rough'

SearchStrng = "kra*"
dir_list = utl.get_filelist(
    searchstr=[SearchStrng], searchpath=WorkDir,
    sortedlist=True, fullpath=True,
)

# =============================================================================
#  Read convergence data and plot
# =============================================================================
for directory in dir_list:
    convergence = []
    iteration = -1

    with open(directory + "/femtic.cnv") as cnv:
        content = cnv.readlines()
        for line in content:
            if "#" in line:
                continue
            iteration += 1
            nline = line.split()
            itern = int(nline[0])
            retry = int(nline[1])
            if retry > 0:
                itern = itern + retry
            alpha = float(nline[2])
            rough = float(nline[5])
            misft = float(nline[7])
            nrmse = float(nline[8])
            convergence.append([iteration, alpha, rough, misft, nrmse])

    if len(convergence) == 0:
        print(directory, "/femtic.cnv is empty!")
        continue

    c = np.array(convergence)
    itern = c[:, 0]
    alpha = c[:, 1]
    rough = c[:, 2]
    misft = c[:, 3]
    nrmse = c[:, 4]

    print("#iter", itern)
    print("#misfit", misft)
    print("#nrmse", nrmse)
    print("#rough", rough)

    # Plotting parameters
    plot_kwargs = dict(
        color="green", marker="o", linestyle="dashed",
        linewidth=1, markersize=7,
        markeredgecolor="red", markerfacecolor="white",
    )

    fig, ax = plt.subplots()
    plot_what = PlotWhat.lower()

    if "mis" in plot_what:
        print("plotting misfit")
        formula = r"$\Vert\mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})\Vert_2$"
        plt.semilogy(itern, misft, **plot_kwargs)
        plt.ylabel(r"misfit " + formula, fontsize=14)

    elif "rms" in plot_what:
        print("plotting nrmse")
        formula = r"$\sqrt{N^{-1} \mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})_2}$"
        plt.plot(itern, nrmse, **plot_kwargs)
        plt.ylabel(r"nRMS " + formula, fontsize=14)

    elif "rough" in plot_what:
        print("plotting roughness")
        formula = r"$\Vert\mathbf{C}_m^{-1/2} \mathbf{m}\Vert_2$"
        plt.semilogy(itern, rough, **plot_kwargs)
        plt.ylabel(r"roughness " + formula, fontsize=14)

    else:
        sys.exit(
            f"plot_convergence: plotting parameter '{plot_what}' not implemented! Exit."
        )

    plt.title(
        PlotName.replace("_", " ") + r" |   $\alpha$ = "
        + str(round(alpha[0], 2))
    )
    plt.xlabel(r"iteration", fontsize=14)
    plt.grid("on")
    plt.tight_layout()

    outfile = (
        WorkDir + PlotName + "_" + plot_what
        + "_alpha" + str(round(alpha[0], 2)) + ".pdf"
    )
    plt.savefig(outfile)
    print(f"Saved {outfile}")
