#!/usr/bin/env python3

"""
This script constructs a list of edifiles in a given directory, and produces
plots for all of them.

@author: sb & vr oct 2019
adapte

"""

# Import required modules

import os
import sys
import numpy as np
from mtpy.core.mt import MT

import matplotlib as mpl
import matplotlib.pyplot as plt

PY4MT_DATA = os.environ["PY4MT_DATA"]
PY4MT_ROOT = os.environ["PY4MT_ROOT"]

mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util
from version import versionstrg


version, _ = versionstrg()
titstrng = util.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

# Graphical paramter. Determine the plot formats produced,
# and the required resolution:

PlotFmt = [".png", ".pdf"]
dpi = 400
PdfCat = True
if not ".pdf" in PlotFmt:
    PdfC = False
    print("No PDF catalog because no pdf output!")
    
PdfCName  = "Opf2023_data.pdf"


# Define the path to your EDI-files:
EdiDir = PY4MT_ROOT+"/work/orig/"
print(" Edifiles read from: %s" % EdiDir)
    
# Define the  path for saving  plots:
PltDir = EdiDir +"/plots/"
print(" Plots written to: %s" % PltDir)
if not os.path.isdir(PltDir):
    print(" File: %s does not exist, but will be created" % PltDir)
    os.mkdir(PltDir)


PlotStrng="_orig"
# PerLimits = np.array([]) #None
PerLimits = np.array([0.00003, 3000.]) # AMT
# PerLimits = np.array([0.001,100000.]) #BBMT
# PerLimits = (0.00003,10000.) #AMT+BBMT

# What should be plotted?
# 1 = yx and xy; 2 = all 4 components
# 3 = off diagonal + determinant

PlotType = 2
PlotTipp="yri"


# RhoLimits = None
RhoLimits = np.array([0.1, 10000.])

Plottipper="yri"
TipLimits = np.array([-.5, 0.5])

                                
PT_colorby = "skew"  #'phimin'
PT_cmap = "mt_bl2gr2rd"
PT_range = [-10.,10.,5.]


# No changes required after this line!

# Construct list of EDI-files:


edi_files = []
files = os.listdir(EdiDir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)
edi_files = sorted(edi_files)

if PdfCat:
    pdf_list= []
# Create an MT object

for filename in edi_files:
    name, ext = os.path.splitext(filename)
    file_i = EdiDir + filename
    
    mt_obj = MT()
    mt_obj.read(file_i)
    
    lat = mt_obj.station_metadata.location.latitude
    lon = mt_obj.station_metadata.location.longitude
    elev = mt_obj.station_metadata.location.elevation
    print(" site %s at :  % 10.6f % 10.6f % 8.1f" % (name, lat, lon, elev ))
    
    # z = mt.Z.z
    # t = mt.Tipper.tipper

    # for ii in np.arange(fs[0]):

    #     if (abs(z[ii,:,:]).any()>1.e30):    z[ii,:,:] = 1.e30
    #     if (abs(z[ii,:,:]).any()<1.e-30):   z[ii,:,:] = 1.e-30
    #     if (abs(t[ii,:]).any()>1.e30):      t[ii,:] = 1.e30
    #     if (abs(t[ii,:]).any()<1.e-30):     t[ii,:] = 1.e-30


    # if no_err is True:
    #     # mt.Z.z_err = 0.0001*np.ones_like( freq_list = mt.Z.freqnp.real(mt.Z.z))
    #     # mt.Tipper.tipper_err = 0.0001*np.ones_like(np.real(mt.Tipper.tipper))
    #     mt.Z.z_err = 0.001 * np.real(mt.Z.z)
    #     mt.Tipper.tipper_err = 0.001 * np.real(mt.Tipper.tipper)
        

    zplot = mt_obj.plot_mt_response(plot_num = PlotType,
                                    fig_num = 2, 
                                    x_limits = PerLimits,
                                    res_limits = RhoLimits, 
                                    tipper_limits = TipLimits,
                                    plot_tipper=PlotTipp,
                                    ellipse_colorby = PT_colorby,  #'phimin'
                                    ellipse_cmap = PT_cmap, 
                                    ellipse_range = PT_range
                                    )  

        
    for F in PlotFmt:
        zplot.save_plot(PltDir+name+PlotStrng+"_z"+F, fig_dpi=600)
    
    if PdfCat: 
        pdf_list.append(PltDir+name+PlotStrng+"_z.pdf")
    

# Finally save figure
if PdfCat:
    util.make_pdf_catalog(PltDir, PdfList=pdf_list, FileName=PltDir+PdfCName)
