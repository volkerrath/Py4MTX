#!/usr/bin/env python3

"""
This script constructs a list of edifiles in a given directory, and produces
plots for all of them.

@author: sb & vr oct 2019
adapted by jc feb 2025

"""

# Import required modules

import os
import sys
import inspect


import getpass
import datetime

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mtpy import MT, MTCollection, MTData

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util as utl
import mtproc as mtp
from version import versionstrg


version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=inspect.getfile(inspect.currentframe()), out=False)
print(titstrng+"\n\n")

PY4MTX_DATA =  "C:/Users/charroyj/Documents/MT/software/Py4MTX_DATA/"
WorkDir = PY4MTX_DATA+"/annecy_2025/"
EdiDir = WorkDir+"edis/"


# Define the path to your MTCollection file:
# CollFile = WorkDir+"/enfield_collection.h5"
# EPSG = 32629
# FromEdis = True
# if FromEdis:
#     print(" Edifiles read from: %s" % EdiDir)
#     dataset = mtp.make_collection(edirname=EdiDir,
#                         collection="Collfile",
#                         metaid="enfield",
#                         survey="enfield",
#                         returndata=True,
#                         utm_epsg=EPSG
#                         )
# else:
#     with MTCollection() as mtc:
#         mtc.open_collection(CollFile)
#         mtc.working_dataframe = mtc.master_dataframe.loc[mtc.master_dataframe.survey == "enfield"]
#         mtc.utm_crs = EPSG
#         dataset = mtc.to_mt_data()


# Define the  path for saving  plots:
PltDir = WorkDir +"/plots/"
print(" Plots written to: %s" % PltDir)
if not os.path.isdir(PltDir):
    print(" File: %s does not exist, but will be created" % PltDir)
    os.mkdir(PltDir)

# Graphical paramter. Determine the plot formats produced,
# and the required resolution:

PlotFmt = [".png", ".pdf"]
DPI = 1200
PDFCatalog = True
PDFCatalogName  = PltDir+"Annecy_2025.pdf"
if not ".pdf" in PlotFmt:
    PDFCatalog= False
    print("No PDF catalog because no pdf output!")




PlotStrng="_data"
# PerLimits = np.array([0.00003, 3.]) # AMT
# PerLimits = np.array([0.001,100000.]) #BBMT
PerLimits = (0.001,30.) #Annecy MT

# What should be plotted?
# 1 = yx and xy; 2 = all 4 components
# 3 = off diagonal + determinant

PlotType = 1
PlotTipp="yri"
TipLimits = np.array([-.5, 0.5])

#  RhoLimits = None
RhoLimits = np.array([0.1, 50000.])


PT_colorby = "skew"  #'phimin'
PT_cmap = "mt_bl2gr2rd"
PT_range = [-10.,10.,5.]

#Do we also plot depth of investigation (Niblett)-Bostick transformation)?
Plotdoi=True

# No changes required after this line!

# Construct list of EDI-files:

edi_files = mtp.get_edi_list(EdiDir)


if PDFCatalog:
    pdf_list = []
    # catalog = PdfPages(PDFCatalogName)


for filename in edi_files:
    name, ext = os.path.splitext(os.path.basename(filename))
    file_i = filename
    # Create an MT object
    mt_obj = MT()
    mt_obj.read(file_i)

    lat = mt_obj.station_metadata.location.latitude
    lon = mt_obj.station_metadata.location.longitude
    elev = mt_obj.station_metadata.location.elevation
    print(" site %s at :  % 10.6f % 10.6f % 8.1f" % (name, lat, lon, elev ))

    plot_response = mt_obj.plot_mt_response(
            plot_num = PlotType, fig_num = 1,
            x_limits = PerLimits,
            res_limits = RhoLimits,
            tipper_limits = TipLimits,
            plot_tipper=PlotTipp,
            ellipse_colorby = PT_colorby,  #'phimin'
            ellipse_cmap = PT_cmap,
            ellipse_range = PT_range,
            close_plot=False)

    plt.show()
    for F in PlotFmt:
        plot_response.save_plot(PltDir+name+PlotStrng+F, fig_dpi=DPI)

    if PDFCatalog:
        pdf_list.append(PltDir+name+PlotStrng+".pdf")
        # catalog.savefig()
    if Plotdoi:
        plot_doi = mt_obj.plot_depth_of_penetration()
        plt.show()
        for F in PlotFmt:
            plot_doi.save_plot(PltDir+name+'_doi'+F, fig_dpi=DPI)

    plt.clf()



# Finally save to multipage catalog
if PDFCatalog:
    utl.make_pdf_catalog(PltDir, PdfList=pdf_list, FileName=PDFCatalogName)
    print(pdf_list)
    # d = catalog.infodict()
    # d["Title"] =  PDFCatalogName
    # d["Author"] = getpass.getuser()
    # d["CreationDate"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # catalog.close()
