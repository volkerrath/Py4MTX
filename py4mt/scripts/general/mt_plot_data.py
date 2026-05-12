#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot MT station data (rho_a, phase, tipper, phase tensor).

Reads .npz, .edi, or .dat station files and generates per-station
diagnostic subplots. Optionally assembles a PDF catalogue.

@author: vrath
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import sys
from pathlib import Path
import inspect
import getpass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from version import versionstrg
import util as utl
from data_viz import add_phase, add_rho, add_tipper, add_pt

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
DAT_DIR = r"/home/vrath/FEMTIC_work/ens_misti/misti_rto_01/"
PLT_DIR = DAT_DIR + "/plots/"

USE_EDI = False
USE_NPZ = True
USE_DAT = False

if USE_DAT:
    DAT_LIST = [DAT_DIR + "/observation.dat"]
elif USE_EDI:
    DAT_LIST = utl.get_filelist(
        searchstr=[".edi"], searchpath=DAT_DIR, sortedlist=True, fullpath=True
    )
elif USE_NPZ:
    DAT_LIST = [DAT_DIR + "observation.npz"]

STRNG_OUT = ""

FILES_ONLY = False
PLT_FMT = [".png", ".pdf"]

CAT_NAME = PLT_DIR + "Annecy_processed.pdf"
CATALOG = True
if ".pdf" not in PLT_FMT:
    print(" No pdf files generated. No catalog possible!")
    CATALOG = False

# =============================================================================
#  Matplotlib settings
# =============================================================================
plt.style.use("seaborn-v0_8-paper")

if FILES_ONLY:
    mpl.use("cairo")

if CATALOG:
    pdf_list = []
    catalog = mpl.backends.backend_pdf.PdfPages(CAT_NAME)

mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["savefig.facecolor"] = "none"
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["savefig.bbox"] = "tight"

NCOLS = 11
colors = plt.cm.jet(np.linspace(0, 1, NCOLS))
PLTARGS = {
    "Fontsize": 8,
    "Labelsize": 10,
    "Titlesize": 10,
    "Linewidths": [0.6],
    "Markersize": 4,
    "Colors": colors,
    "Grey": 0.7,
}

# =============================================================================
#  Ensure output directory exists
# =============================================================================
if not os.path.isdir(PLT_DIR):
    print(" Directory %s does not exist, creating." % PLT_DIR)
    os.makedirs(PLT_DIR)

# =============================================================================
#  Plot data sets
# =============================================================================
for site in DAT_LIST:
    data = np.load(site)
    station = data["station"]
    df = pd.DataFrame(data)

    fig, axs = plt.subplots(3, 2, figsize=(8, 14), sharex=True)

    add_rho(df, comps="xy,yx", ax=axs[0, 0], **PLTARGS)
    add_phase(df, comps="xy,yx", ax=axs[0, 1], **PLTARGS)
    add_rho(df, comps="xx,yy", ax=axs[1, 0], **PLTARGS)
    add_phase(df, comps="xx,yy", ax=axs[1, 1], **PLTARGS)
    add_tipper(df, ax=axs[2, 0])
    add_pt(df, ax=axs[2, 1])
    fig.suptitle(station)

    # Remove empty axes
    for ax in axs.flat:
        if not ax.lines and not ax.images and not ax.collections:
            fig.delaxes(ax)

    for fmt in PLT_FMT:
        pltfilename = PLT_DIR + station + STRNG_OUT + fmt
        plt.savefig(pltfilename, dpi=600)

    if CATALOG:
        pdf_list.append(pltfilename)
        catalog.savefig(fig)

    plt.show()

if CATALOG:
    print(pdf_list)
    d = catalog.infodict()
    d["Title"] = CAT_NAME
    d["Author"] = getpass.getuser()
    d["CreationDate"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    catalog.close()

print("Data plots ready!")
