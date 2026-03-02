#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute cell-wise statistics on ModEM model files.

Reads one or more ModEM model files and computes mean, variance, median,
and quantile statistics. Optionally generates histograms and a PDF catalog.

Percentile reference (normal distribution):
    -2 sigma -> 2.28th,  -1 sigma -> 15.87th,  median -> 50th,
    +1 sigma -> 84.13th, +2 sigma -> 97.72nd

@author: vrath, sbyrd
"""

import os
import sys
import inspect
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from version import versionstrg
import util as utl
import modem as mod

rng = np.random.default_rng()
blank = np.nan
rhoair = 17.0  # log10(rhoair)

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
hpads = 10
vpads = 50
OutFormat = "modem"

ModHist = True
PlotFormat = [".png", ".pdf"]

WorkDir = "/home/vrath/work/MT_Data/Sabancaya/Saba_best/SABA13a/"
if not WorkDir.endswith("/"):
    WorkDir = WorkDir + "/"

MFile = WorkDir + "SABA13a"
Models = [MFile]

PDFCatalog = False
if len(Models) > 1:
    ModFileEns = "ModEns"
    if ".pdf" in PlotFormat:
        PDFCatalog = True
        PDFCatalogName = "ModEnsCatalog.pdf"
        catalog = PdfPages(PDFCatalogName)

# =============================================================================
#  Read models and compute statistics
# =============================================================================
imod = -1
for f in Models:
    imod += 1
    dx, dy, dz, rho, ref, trans = mod.read_mod(
        file=f, modext=".rho", trans="LOG10", blank=1.0e-30, out=True,
    )
    dims = np.shape(rho)
    aircells = np.where(rho > rhoair)
    print(f + ".rho, shape is", dims)

    rtmp = rho.ravel()
    if imod == 0:
        ModEns = rtmp
    else:
        ModEns = np.vstack((ModEns, rtmp))

    # Blank out padding and air cells
    rho[aircells] = blank
    rho[:hpads, :, :] = blank
    rho[dims[0] - hpads : dims[0], :, :] = blank
    rho[:, :hpads, :] = blank
    rho[:, dims[1] - hpads : dims[1], :] = blank
    rho[:, :, dims[2] - vpads : dims[2]] = blank

    # Print statistics
    rhoavg = np.nanmean(rho)
    print(f"Mean resistivity is {np.power(10.0, rhoavg):.2f}")
    print(f"Mean log resistivity is {rhoavg:.4f}")
    rhostd = np.sqrt(np.nanvar(rho))
    print(f"Std log resistivity is {rhostd:.4f}")

    rhomed = np.nanmedian(rho)
    print(f"Median resistivity is {np.power(10.0, rhomed):.2f}")
    print(f"Median log resistivity is {rhomed:.4f}")
    print("1-sigma quantiles:", np.nanquantile(rho, [0.16, 0.84]))
    print("2-sigma quantiles:", np.nanquantile(rho, [0.023, 0.977]))

    if ModHist:
        rtmp = rho.ravel()
        rtmp = rtmp[np.isfinite(rtmp)]
        fig, ax = plt.subplots()
        counts, bins = np.histogram(rtmp, bins=51, range=(-1, 4))
        plt.stairs(counts, bins, fill=True)
        plt.xlabel(r"log resistivity $\Omega m$")
        plt.ylabel(r"counts")
        plt.grid("on")
        plt.title(os.path.splitext(os.path.basename(f))[0])
        plt.tight_layout()

        pname = os.path.splitext(f)[0]
        for fmt in PlotFormat:
            plt.savefig(pname + fmt)

        if PDFCatalog:
            catalog.savefig(fig)

# Ensemble statistics (multi-model case)
if len(Models) > 1:
    ModAvg = np.mean(ModEns, axis=1).reshape(dims)
    ModVar = np.var(ModEns, axis=1).reshape(dims)

    if "mod" in OutFormat.lower():
        mod.write_mod(
            ModFileEns, modext="_avg.rho",
            dx=dx, dy=dy, dz=dz, mval=ModAvg,
            reference=ref, mvalair=blank, aircells=aircells,
            header="Model log-average",
        )
        print("Averages written to " + ModFileEns + "_avg.rho")
        mod.write_mod(
            ModFileEns, modext="_var.rho",
            dx=dx, dy=dy, dz=dz, mval=np.sqrt(ModVar),
            reference=ref, mvalair=blank, aircells=aircells,
            header="Model log-std",
        )
        print("Std dev written to " + ModFileEns + "_var.rho")

    if "rlm" in OutFormat.lower():
        mod.write_rlm(
            ModFileEns, modext="_avg.rlm",
            dx=dx, dy=dy, dz=dz, mval=ModAvg,
            reference=ref, mvalair=blank, aircells=aircells,
            comment="Model log-average",
        )
        mod.write_rlm(
            ModFileEns, modext="_var.rlm",
            dx=dx, dy=dy, dz=dz, mval=np.sqrt(ModVar),
            reference=ref, mvalair=blank, aircells=aircells,
            comment="Model log-std",
        )

if PDFCatalog:
    catalog.close()
