#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot impedance strike directions from MT station data.

Generates station maps, aggregate strike rose diagrams, per-decade
strike plots, and per-station strike diagrams using dtpy. Optionally
assembles a PDF catalogue.

@author: sb & vr dec 2019
"""

import os
import sys
import inspect

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from dtpy import MT, MTCollection, MTData

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
WorkDir = PY4MTX_DATA + "/France/annecy_2025_dist/edi_files/"
EdiDir = WorkDir
surveyname = "Annecy"

file_list = dtp.get_edi_list(EdiDir)
ns = len(file_list)

Collection = WorkDir + "/ubaye_collection.h5"

FromEdis = True
if FromEdis:
    print(" Edifiles read from: %s" % EdiDir)
    dataset = dtp.make_data(
        edirname=EdiDir,
        collection=Collection,
        metaid="ubaye",
        survey=surveyname,
        savedata=True,
        utm_epsg=EPSG,
    )
else:
    with MTCollection() as mtc:
        mtc.open_collection(Collection)
        mtc.working_dataframe = mtc.master_dataframe.loc[
            mtc.master_dataframe.survey == surveyname
        ]
        mtc.utm_crs = EPSG
        dataset = mtc.to_mt_data()

PltDir = WorkDir + "/plots/"
print(" Plots written to: %s" % PltDir)
if not os.path.isdir(PltDir):
    print(" File: %s does not exist, but will be created" % PltDir)
    os.mkdir(PltDir)

# =============================================================================
#  Plot settings
# =============================================================================
PlotFmt = [".png"]
DPI = 600
PDFCatalog = True
PDFCatalogName = "Ubaye_strikes.pdf"
if ".pdf" not in PlotFmt:
    PDFCatalog = False
    print("No PDF catalog because no pdf output!")

# =============================================================================
#  Generate plots
# =============================================================================
stations_plot = dataset.plot_stations(pad=0.005)
for fmt in PlotFmt:
    stations_plot.save_plot(PltDir + "AllSites" + fmt, fig_dpi=DPI)

strike_plot_all = dataset.plot_strike()
for fmt in PlotFmt:
    strike_plot_all.save_plot(PltDir + "StrikesAllData" + fmt, fig_dpi=DPI)

strike_plot_dec = dataset.plot_strike(
    plot_type=1,
    print_stats=True,
    text_pad=0.005,
    plot_pt=True,
    plot_tipper=True,
    plot_invariant=True,
    plot_orientation="v",
)
for fmt in PlotFmt:
    strike_plot_dec.save_plot(PltDir + "StrikesPerDec" + fmt, fig_dpi=DPI)

# =============================================================================
#  Per-station strike plots
# =============================================================================
if PDFCatalog:
    pdf_list = []

for sit in file_list:
    site, _ = os.path.splitext(os.path.basename(sit))
    data = dataset.get_subset([surveyname + "." + site])
    strike_plot_site = data.plot_strike()
    for fmt in PlotFmt:
        strike_plot_site.save_plot(PltDir + site + "_strikes" + fmt, fig_dpi=DPI)

    if PDFCatalog:
        pdf_list.append(PltDir + site + "_strikes.pdf")

if PDFCatalog:
    utl.make_pdf_catalog(PltDir, pdflist=pdf_list, filename=PDFCatalogName)
