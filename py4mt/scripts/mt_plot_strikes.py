#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot impedance strike directions from MT station data.

Generates station maps, aggregate strike rose diagrams, per-decade
strike plots, and per-station strike diagrams using mtpy. Optionally
assembles a PDF catalogue.

@author: sb & vr dec 2019
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import sys
import inspect

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from mtpy import MT, MTCollection, MTData

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import data_proc as dtp
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
EPSG = 32629
WORK_DIR = PY4MTX_DATA + "/France/annecy_2025_dist/edi_files/"
EDI_DIR = WORK_DIR
SURVEY_NAME = "Annecy"

file_list = dtp.get_edi_list(EDI_DIR)
ns = len(file_list)

COLLECTION = WORK_DIR + "/ubaye_collection.h5"

FROM_EDIS = True
if FROM_EDIS:
    print(" Edifiles read from: %s" % EDI_DIR)
    dataset = dtp.make_data(
        edirname=EDI_DIR,
        collection=COLLECTION,
        metaid="ubaye",
        survey=SURVEY_NAME,
        savedata=True,
        utm_epsg=EPSG,
    )
else:
    with MTCollection() as mtc:
        mtc.open_collection(COLLECTION)
        mtc.working_dataframe = mtc.master_dataframe.loc[
            mtc.master_dataframe.survey == SURVEY_NAME
        ]
        mtc.utm_crs = EPSG
        dataset = mtc.to_mt_data()

PLT_DIR = WORK_DIR + "/plots/"
print(" Plots written to: %s" % PLT_DIR)
if not os.path.isdir(PLT_DIR):
    print(" Directory %s does not exist, creating." % PLT_DIR)
    os.mkdir(PLT_DIR)

# =============================================================================
#  Plot settings
# =============================================================================
PLOT_FMT = [".png"]
DPI = 600
PDF_CATALOG = True
PDF_CATALOG_NAME = "Ubaye_strikes.pdf"
if ".pdf" not in PLOT_FMT:
    PDF_CATALOG = False
    print("No PDF catalog because no pdf output!")

# =============================================================================
#  Generate plots
# =============================================================================
stations_plot = dataset.plot_stations(pad=0.005)
for fmt in PLOT_FMT:
    stations_plot.save_plot(PLT_DIR + "AllSites" + fmt, fig_dpi=DPI)

strike_plot_all = dataset.plot_strike()
for fmt in PLOT_FMT:
    strike_plot_all.save_plot(PLT_DIR + "StrikesAllData" + fmt, fig_dpi=DPI)

strike_plot_dec = dataset.plot_strike(
    plot_type=1,
    print_stats=True,
    text_pad=0.005,
    plot_pt=True,
    plot_tipper=True,
    plot_invariant=True,
    plot_orientation="v",
)
for fmt in PLOT_FMT:
    strike_plot_dec.save_plot(PLT_DIR + "StrikesPerDec" + fmt, fig_dpi=DPI)

# =============================================================================
#  Per-station strike plots
# =============================================================================
if PDF_CATALOG:
    pdf_list = []

for sit in file_list:
    site, _ = os.path.splitext(os.path.basename(sit))
    data = dataset.get_subset([SURVEY_NAME + "." + site])
    strike_plot_site = data.plot_strike()
    for fmt in PLOT_FMT:
        strike_plot_site.save_plot(PLT_DIR + site + "_strikes" + fmt, fig_dpi=DPI)

    if PDF_CATALOG:
        pdf_list.append(PLT_DIR + site + "_strikes.pdf")

if PDF_CATALOG:
    utl.make_pdf_catalog(PLT_DIR, pdflist=pdf_list, filename=PDF_CATALOG_NAME)
