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
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import sys
import inspect

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

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
HPADS = 10
VPADS = 50
OUT_FORMAT = "modem"
BLANK = np.nan
RHOAIR = 17.0  # log10(rhoair)

MOD_HIST = True
PLOT_FORMAT = [".png", ".pdf"]

WORK_DIR = "/home/vrath/work/MT_Data/Sabancaya/Saba_best/SABA13a/"
if not WORK_DIR.endswith("/"):
    WORK_DIR = WORK_DIR + "/"

M_FILE = WORK_DIR + "SABA13a"
MODELS = [M_FILE]

PDF_CATALOG = False
if len(MODELS) > 1:
    MOD_FILE_ENS = "ModEns"
    if ".pdf" in PLOT_FORMAT:
        PDF_CATALOG = True
        PDF_CATALOG_NAME = "ModEnsCatalog.pdf"
        catalog = PdfPages(PDF_CATALOG_NAME)

# =============================================================================
#  Read models and compute statistics
# =============================================================================
ModEns = None

for imod, f in enumerate(MODELS):
    dx, dy, dz, rho, ref, trans = mod.read_mod(
        file=f, modext=".rho", trans="LOG10", blank=1.0e-30, out=True,
    )
    dims = np.shape(rho)
    aircells = np.where(rho > RHOAIR)
    print(f + ".rho, shape is", dims)

    rtmp = rho.ravel()
    if ModEns is None:
        ModEns = rtmp
    else:
        ModEns = np.vstack((ModEns, rtmp))

    # Blank out padding and air cells
    rho[aircells] = BLANK
    rho[:HPADS, :, :] = BLANK
    rho[dims[0] - HPADS : dims[0], :, :] = BLANK
    rho[:, :HPADS, :] = BLANK
    rho[:, dims[1] - HPADS : dims[1], :] = BLANK
    rho[:, :, dims[2] - VPADS : dims[2]] = BLANK

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

    if MOD_HIST:
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
        for fmt in PLOT_FORMAT:
            plt.savefig(pname + fmt)

        if PDF_CATALOG:
            catalog.savefig(fig)

# =============================================================================
#  Ensemble statistics (multi-model case)
# =============================================================================
if len(MODELS) > 1:
    ModAvg = np.mean(ModEns, axis=1).reshape(dims)
    ModVar = np.var(ModEns, axis=1).reshape(dims)

    if "mod" in OUT_FORMAT.lower():
        mod.write_mod(
            MOD_FILE_ENS, modext="_avg.rho",
            dx=dx, dy=dy, dz=dz, mval=ModAvg,
            reference=ref, mvalair=BLANK, aircells=aircells,
            header="Model log-average",
        )
        print("Averages written to " + MOD_FILE_ENS + "_avg.rho")
        mod.write_mod(
            MOD_FILE_ENS, modext="_var.rho",
            dx=dx, dy=dy, dz=dz, mval=np.sqrt(ModVar),
            reference=ref, mvalair=BLANK, aircells=aircells,
            header="Model log-std",
        )
        print("Std dev written to " + MOD_FILE_ENS + "_var.rho")

    if "rlm" in OUT_FORMAT.lower():
        mod.write_rlm(
            MOD_FILE_ENS, modext="_avg.rlm",
            dx=dx, dy=dy, dz=dz, mval=ModAvg,
            reference=ref, mvalair=BLANK, aircells=aircells,
            comment="Model log-average",
        )
        mod.write_rlm(
            MOD_FILE_ENS, modext="_var.rlm",
            dx=dx, dy=dy, dz=dz, mval=np.sqrt(ModVar),
            reference=ref, mvalair=BLANK, aircells=aircells,
            comment="Model log-std",
        )

if PDF_CATALOG:
    catalog.close()
