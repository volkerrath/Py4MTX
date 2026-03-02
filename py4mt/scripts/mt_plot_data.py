#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot MT station data (rho_a, phase, tipper, phase tensor).

Reads .npz, .edi, or .dat station files and generates per-station
diagnostic subplots. Optionally assembles a PDF catalogue.

@author: vrath
"""

import os
import sys
import inspect
import getpass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from version import versionstrg
import util as utl
from data_viz import add_phase, add_rho, add_tipper, add_pt

rng = np.random.default_rng()
nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
DatDir = r"/home/vrath/FEMTIC_work/ens_misti/misti_rto_01/"
PltDir = DatDir + "/plots/"

UseEDI = False
UseNPZ = True
UseDAT = False

if UseDAT:
    DatList = [DatDir + "/observation.dat"]
elif UseEDI:
    edi_files = utl.get_filelist(
        searchstr=[".edi"], searchpath=DatDir, sortedlist=True, fullpath=True
    )
    DatList = edi_files
elif UseNPZ:
    DatList = [DatDir + "observation.npz"]

StrngOut = ""

FilesOnly = False
PltFmt = [".png", ".pdf"]

CatName = PltDir + "Annecy_processed.pdf"
Catalog = True
if ".pdf" not in PltFmt:
    print(" No pdf files generated. No catalog possible!")
    Catalog = False

# =============================================================================
#  Matplotlib settings
# =============================================================================
plt.style.use("seaborn-v0_8-paper")

if FilesOnly:
    mpl.use("cairo")

if Catalog:
    pdf_list = []
    catalog = mpl.backends.backend_pdf.PdfPages(CatName)

mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["savefig.facecolor"] = "none"
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["savefig.bbox"] = "tight"

ncols = 11
colors = plt.cm.jet(np.linspace(0, 1, ncols))
pltargs = {
    "Fontsize": 8,
    "Labelsize": 10,
    "Titlesize": 10,
    "Linewidths": [0.6],
    "Markersize": 4,
    "Colors": colors,
    "Grey": 0.7,
}

# =============================================================================
#  Plot data sets
# =============================================================================
for site in DatList:
    data = np.load(site)
    station = data["station"]
    df = pd.DataFrame(data)

    fig, axs = plt.subplots(3, 2, figsize=(8, 14), sharex=True)

    add_rho(df, comps="xy,yx", ax=axs[0, 0], **pltargs)
    add_phase(df, comps="xy,yx", ax=axs[0, 1], **pltargs)
    add_rho(df, comps="xx,yy", ax=axs[1, 0], **pltargs)
    add_phase(df, comps="xx,yy", ax=axs[1, 1], **pltargs)
    add_tipper(df, ax=axs[2, 0])
    add_pt(df, ax=axs[2, 1])
    fig.suptitle(station)

    # Remove empty axes
    for ax in axs.flat:
        if not ax.lines and not ax.images and not ax.collections:
            fig.delaxes(ax)

    for fmt in PltFmt:
        pltfilename = PltDir + station + StrngOut + fmt
        plt.savefig(pltfilename, dpi=600)

    if Catalog:
        pdf_list.append(pltfilename)
        catalog.savefig(fig)

    plt.show()

if Catalog:
    print(pdf_list)
    d = catalog.infodict()
    d["Title"] = CatName
    d["Author"] = getpass.getuser()
    d["CreationDate"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    catalog.close()

print("data plots ready!")
