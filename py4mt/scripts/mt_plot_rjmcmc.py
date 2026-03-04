#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise rjmcmc-MT transdimensional inversion results.

Plots posterior resistivity-depth distributions from rjmcmc output,
optionally augments each station with EDI coordinates, and assembles
a combined data file and PDF catalogue.

Based on Ross Brodie's original MATLAB plotting routines.

Revision History:
    2017/10  R. Hassan (GA) — original MATLAB version
    2019/10  V. Rath — adapted colorbar, sizes
    2020/03  V. Rath — minor improvements, less memory
    2022/03  V. Rath — more options, graphical improvements
    2024/04  V. Rath — python 3.11 / mtpy-v2

@author: vrath
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import sys
import inspect

import numpy as np

from data_proc import load_edi

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
PY4MTX_DATA = os.environ["PY4MTX_DATA"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import plotrjmcmc as pmc
import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
WORK_DIR = PY4MTX_DATA + "/Ubaye_best/"
EDI_DIR = WORK_DIR + "/edis/"
print("Edifiles read from: %s" % EDI_DIR)

RES_DIR = WORK_DIR + "/transdim/results_impedance/"
PLT_DIR = WORK_DIR + "/plots/"

PLOT_FMT = ".pdf"
RHO_PLOT_LIM = [0.1, 10000]
DEPTH_PLOT_LIM = 15000.0
LOG_DEPTH = False
COLOR_MAP = "rainbow"

PDF_CATALOG = True
PDF_CATALOG_NAME = WORK_DIR + "Ubaye_results.pdf"
if ".pdf" not in PLOT_FMT:
    PDF_CATALOG = False
    print("No PDF catalog because no pdf output!")
OUT_STRNG = "_model"

DATA_OUT = True
DATA_NAME = WORK_DIR + "Ubaye_1dresults.dat"
W_REF = False

# =============================================================================
#  Collect file lists
# =============================================================================
edi_files = sorted([
    entry for entry in os.listdir(EDI_DIR)
    if entry.endswith(".edi") and not entry.startswith(".")
])

result_files = sorted([
    entry for entry in os.listdir(RES_DIR)
    if not entry.startswith(".")
])
nfiles = len(result_files)

if PDF_CATALOG:
    pdf_list = []
    for filename in result_files:
        name, _ = os.path.splitext(filename)
        pdf_list.append(PLT_DIR + name + OUT_STRNG + ".pdf")

if not os.path.isdir(PLT_DIR):
    print(" Directory %s does not exist, creating." % PLT_DIR)
    os.mkdir(PLT_DIR)

# =============================================================================
#  Process each result file
# =============================================================================
data_all = None

for count, filename in enumerate(result_files, start=1):
    print(f"\n{count} of {nfiles}")

    infile = RES_DIR + filename
    name, _ = os.path.splitext(filename)
    outfile = PLT_DIR + name + OUT_STRNG + PLOT_FMT
    print(infile)
    print(outfile)

    r = pmc.Results(
        infile,
        outfile,
        plotSizeInches="11x8",
        maxDepth=DEPTH_PLOT_LIM,
        resLims=RHO_PLOT_LIM,
        zLog=LOG_DEPTH,
        colormap=COLOR_MAP,
    )
    r.plot()

    if DATA_OUT:
        file_i = EDI_DIR + name + ".edi"
        mt_dict = load_edi(file_i)

        lat = mt_dict["lat"]
        lon = mt_dict["lon"]
        elev = mt_dict["elev"]

        name_result, _ = os.path.splitext(outfile)

        data_in = np.loadtxt(name_result + ".dat")
        sd = np.shape(data_in)
        lon_col = np.ones((sd[0], 1)) * lon
        lat_col = np.ones((sd[0], 1)) * lat

        if W_REF:
            elev_col = np.ones_like(data_in[:, 0]) * elev
            data_in[:, 0] = elev_col - data_in[:, 0]

        data_out = np.append(lat_col, lon_col, axis=1)
        data_out = np.append(data_out, data_in, axis=1)
        header = (
            name.split("_")[0]
            + "  lat, lon, depth, median, q10, q90, mean, mode"
        )
        np.savetxt(name_result + ".dat", data_out, delimiter="  ", header=header)

        sit = np.full_like(lat_col, name.split("_")[0], dtype=object)
        tmp = np.append(sit, data_out, axis=1)

        if data_all is None:
            data_all = tmp
        else:
            data_all = np.append(data_all, tmp, axis=0)

# =============================================================================
#  Final outputs
# =============================================================================
if DATA_OUT and data_all is not None:
    header = "All data:  site, lat, lon, depth, median, q10, q90, mean, mode"
    fmt = "%s  %14.7f  %14.7f  %15.5f  %18.5e  %18.5e %18.5e  %18.5e  %18.5e"
    np.savetxt(DATA_NAME, data_all, delimiter="  ", header=header, fmt=fmt)

if PDF_CATALOG:
    utl.make_pdf_catalog(PLT_DIR, pdflist=pdf_list, filename=PDF_CATALOG_NAME)
